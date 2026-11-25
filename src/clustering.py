import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_samples, silhouette_score)
from sklearn.preprocessing import StandardScaler

from .config import CONFIG, MODEL_DIR, PROCESSED_DIR, REPORT_DIR
from .io_utils import save_csv, save_json, save_parquet
from .features import build_panel, build_kab_features, feature_registry

RS = 42
CORE = ["l_produksi", "l_pembudidaya", "l_harga", "shannon"]
LABEL_KHUSUS = "Spesialis Rumput Laut (outlier struktural)"


def robust_z(s):
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    return (s - med) / mad if mad > 0 else s * 0


def _n_iter():
    fast = os.environ.get("PIPE_FAST") == "1"
    return (10, 20, 50) if fast else (30, 60, 300)      # n_init evaluasi, n bootstrap, n_init final


def run_clustering(df: pd.DataFrame = None, config=None) -> dict:
    warnings.filterwarnings("ignore")
    C = {**{"min_size_frac": 0.13, "min_size_abs": 3, "stability_ari_min": 0.70, "sil_tolerance": 0.02, "special_laut_share": 0.9,
            "special_harga_z": -3.5, "kmax": 8, "label_harga_tinggi_x": 1.35}, **((config or CONFIG).get("cluster") or {})}
    N_INIT_EVAL, N_BOOT, N_INIT_FINAL = _n_iter()

    if df is not None and "total_pembesaran" in df.columns:          # keluaran build_features_model (DB + kolom tambahan) -> panel & profil kabupaten
        reg = feature_registry(df).set_index("fitur")
        panel = build_panel(df)
        F = build_kab_features(panel)
    else:                                                              # mode lama: baca berkas hasil notebook 02
        F = pd.read_parquet(PROCESSED_DIR / "features_kab.parquet")
        panel = df if df is not None else pd.read_parquet(PROCESSED_DIR / "panel_kab_tahun.parquet")
        reg = pd.read_csv(REPORT_DIR / "feature_registry.csv").set_index("fitur")
    usable = lambda f: bool(reg.loc[f, "usable_cluster"]) if f in reg.index else False

    tidak_diklaster = F[F[CORE].isna().any(axis=1)].kabupaten.tolist()          # data inti belum lengkap
    if tidak_diklaster:
        print(f"  Belum diklaster (data inti belum lengkap): {tidak_diklaster}")
        F = F[~F.kabupaten.isin(tidak_diklaster)].reset_index(drop=True)
    N = len(F)
    if N < 4:
        print(f"  Clustering dilewati: hanya {N} kabupaten dengan data inti lengkap")
        return {"k_opt": None, "silhouette": None, "eval_df": None, "profil": None, "label_map": {}, "df": None}
    MIN_SIZE = max(C["min_size_abs"], int(round(C["min_size_frac"] * N)))
    print(f"  N={N} kabupaten -> ukuran klaster minimum = {MIN_SIZE}")

    # --- 1. segmen khusus: habitat laut > 90% DAN harga robust-z < -3,5
    Fi = F.set_index("kabupaten")
    spesialis = Fi.index[(Fi.h_laut.fillna(0) > C["special_laut_share"]) & (robust_z(Fi.l_harga) < C["special_harga_z"])].tolist()
    print(f"  Segmen khusus: {spesialis if spesialis else '(tidak ada)'}")
    F_fit = F[~F.kabupaten.isin(spesialis)].reset_index(drop=True)
    med_imp = {}
    for c in ["h_kolam", "h_tambak", "h_laut", "l_pembenihan", "l_ikan_hias"]:
        med_imp[c] = F_fit[c].median()
        F_fit[c] = F_fit[c].fillna(med_imp[c] if pd.notna(med_imp[c]) else 0.0)      # kolom kosong total -> 0 (agar StandardScaler tidak gagal)
    if len(F_fit) < 4:
        print("  Clustering dilewati: data setelah segmen khusus < 4 kabupaten")
        return {"k_opt": None, "silhouette": None, "eval_df": None, "profil": None, "label_map": {}, "df": None}

    # --- 2. pemilihan set fitur
    SETS = {"A": ["l_produksi", "l_pembudidaya", "l_harga", "shannon"],
            "B": ["l_produksi", "l_pembudidaya", "l_harga", "shannon", "h_kolam", "h_tambak"],
            "C": ["l_produksi", "l_pembudidaya", "l_harga", "shannon", "h_kolam", "h_tambak", "l_pembenihan", "l_ikan_hias"]}
    if usable("luas_lahan_ha") and F_fit.l_luas.notna().mean() >= 0.8:
        med_imp["l_luas"] = F_fit.l_luas.median()
        F_fit["l_luas"] = F_fit.l_luas.fillna(med_imp["l_luas"])
        SETS["D"] = SETS["B"] + ["l_luas"]
    else:
        print("  Set D (luas lahan) tidak diuji: cakupan luas belum memadai.")

    def boot_stability(X, k, n=N_BOOT, seed=0):
        base = KMeans(k, n_init=N_INIT_EVAL, random_state=RS).fit(X).labels_
        rng = np.random.default_rng(seed)
        a = []
        for b in range(n):
            idx = rng.choice(len(X), len(X), replace=True)
            a.append(adjusted_rand_score(base, KMeans(k, n_init=5, random_state=b).fit(X[idx]).predict(X)))
        return float(np.mean(a))

    def eval_k(X, ks):
        rows = []
        for k in ks:
            lab = KMeans(k, n_init=N_INIT_EVAL, random_state=RS).fit_predict(X)
            rows.append(dict(k=k, silhouette=silhouette_score(X, lab), davies_bouldin=davies_bouldin_score(X, lab),
                             calinski=calinski_harabasz_score(X, lab), ukuran_min=int(np.bincount(lab).min()),
                             stabilitas_ARI=boot_stability(X, k)))
        return pd.DataFrame(rows)

    KS = list(range(2, max(2, min(C["kmax"], len(F_fit) // MIN_SIZE)) + 1))
    KS = [k for k in KS if k < len(F_fit)]
    ev_all = []
    for nm, cols in SETS.items():
        e = eval_k(StandardScaler().fit_transform(F_fit[cols]), KS)
        e.insert(0, "set_fitur", nm)
        e["lolos_syarat"] = (e.ukuran_min >= MIN_SIZE) & (e.stabilitas_ARI >= C["stability_ari_min"])
        ev_all.append(e)
    ev_all = pd.concat(ev_all, ignore_index=True)

    best_by_set = ev_all[ev_all.lolos_syarat].groupby("set_fitur").silhouette.max()
    if best_by_set.empty:
        print("  PERINGATAN: tidak ada set/k yang lolos syarat (data terlalu sedikit/tak stabil) -> pakai Set B, k=2")
        SET_PILIH = "B"
    else:
        SET_PILIH = "B" if ("B" in best_by_set.index and best_by_set["B"] >= best_by_set.max() - C["sil_tolerance"]) else best_by_set.idxmax()
    print(f"  Set fitur terpilih: {SET_PILIH} (silhouette terbaik per set yang lolos: {best_by_set.round(3).to_dict()})")

    # --- 3. memilih k
    FITUR = SETS[SET_PILIH]
    scaler = StandardScaler().fit(F_fit[FITUR])
    X = scaler.transform(F_fit[FITUR])
    ev = ev_all[ev_all.set_fitur == SET_PILIH].reset_index(drop=True)
    feas = ev[ev.lolos_syarat]
    if feas.empty:
        k_opt = 2
        ev["dipilih"] = ev.k == 2
    else:
        best = feas.silhouette.max()
        k_opt = int(feas[feas.silhouette >= best - C["sil_tolerance"]].k.max())
        ev["dipilih"] = ev.k == k_opt
    save_csv(ev, REPORT_DIR / "clustering_eval.csv")
    save_csv(ev_all, REPORT_DIR / "clustering_eval_semua_set.csv")

    # --- 4. model akhir + ketahanan
    km = KMeans(k_opt, n_init=N_INIT_FINAL, random_state=RS).fit(X)
    F_fit["klaster"] = km.labels_
    sil = float(silhouette_score(X, km.labels_))
    ward = AgglomerativeClustering(k_opt, linkage="ward").fit_predict(X)
    F_fit["silhouette_i"] = silhouette_samples(X, km.labels_)
    print(f"  Silhouette={sil:.3f} | Davies-Bouldin={davies_bouldin_score(X, km.labels_):.2f} | ARI vs Ward={adjusted_rand_score(km.labels_, ward):.2f} "
          f"| silhouette negatif: {int((F_fit.silhouette_i < 0).sum())}")

    # --- 5. profil & pelabelan (ambang relatif)
    prof = F_fit.groupby("klaster").agg(
        n_kab=("kabupaten", "size"), produksi_ton_median=("produksi_ton", "median"), pembudidaya_median=("pembudidaya", "median"),
        harga_rp_kg_median=("harga_rp_kg", "median"), shannon=("shannon", "mean"), habitat_kolam=("h_kolam", "mean"),
        habitat_tambak=("h_tambak", "mean"), pembenihan_median=("pembenihan_ribu_ekor", "median"),
        ikan_hias_median=("ikan_hias_ekor", "median"), luas_lahan_median=("luas_lahan_ha", "median"), rtp_median=("rtp", "median"))
    rank_prod = prof.produksi_ton_median.rank(ascending=False, method="first").astype(int)
    med_h, med_p = F_fit.harga_rp_kg.median(), F_fit.produksi_ton.median()

    def beri_label(r, rk):
        if r.habitat_tambak > 0.5:
            return "Tambak Payau – Bernilai Tinggi (Udang)" if r.harga_rp_kg_median > C["label_harga_tinggi_x"] * med_h else "Tambak Payau – Skala Menengah"
        if rk == 1:
            return "Skala Besar & Beragam"
        if r.habitat_kolam > 0.7 and r.produksi_ton_median > med_p:
            return "Kolam Air Tawar – Skala Menengah"
        return "Skala Kecil (Perkotaan / Terbatas)"

    labels = {int(k): beri_label(prof.loc[k], rank_prod[k]) for k in prof.index}
    sl = pd.Series(labels)
    dup = sl[sl.duplicated(keep=False)]
    if len(dup):
        labels = {k: (f"{v} (K{k + 1})" if k in dup.index else v) for k, v in labels.items()}
        print(f"  Label ganda -> diberi akhiran (K#): {dup.to_dict()}")
    prof["label"] = pd.Series(labels)
    F_fit["label"] = F_fit.klaster.map(labels)

    # --- 6. segmen khusus & penempatan per tahun (pusat klaster tetap)
    ID_KHUSUS = k_opt
    assign = pd.concat([F_fit[["kabupaten", "klaster", "label"]],
                        pd.DataFrame({"kabupaten": spesialis, "klaster": ID_KHUSUS, "label": LABEL_KHUSUS})], ignore_index=True)
    assign = assign.sort_values("kabupaten").reset_index(drop=True)
    assert assign.kabupaten.is_unique
    labels_all = {**labels, **({ID_KHUSUS: LABEL_KHUSUS} if spesialis else {})}

    pj = panel[["kabupaten", "tahun", "l_produksi_ton", "l_pembudidaya", "l_harga_rp_kg", "shannon", "luas_lahan_ha"]].rename(
        columns={"l_produksi_ton": "l_produksi", "l_harga_rp_kg": "l_harga"})
    pj["l_luas"] = np.log1p(pj.pop("luas_lahan_ha"))                              # sama dengan definisi di features_kab
    hab_med = F_fit[["h_kolam", "h_tambak"]].median()
    pj = pj.join(F.set_index("kabupaten")[["h_kolam", "h_tambak"]].fillna(hab_med), on="kabupaten")
    fb = F.set_index("kabupaten")[["l_produksi", "l_pembudidaya", "l_harga", "shannon", "l_luas"]]
    for c in ["l_produksi", "l_pembudidaya", "l_harga", "shannon", "l_luas"]:
        pj[c] = pj[c].fillna(pj.kabupaten.map(fb[c]))
    if "l_luas" in FITUR:
        pj["l_luas"] = pj["l_luas"].fillna(med_imp["l_luas"])
    pj = pj[pj.kabupaten.isin(assign.kabupaten) & ~pj.kabupaten.isin(tidak_diklaster)]
    pj = pj.dropna(subset=FITUR).copy()
    pj["klaster"] = km.predict(scaler.transform(pj[FITUR]))
    pj.loc[pj.kabupaten.isin(spesialis), "klaster"] = ID_KHUSUS
    pj["label"] = pj.klaster.map(labels_all)
    print(f"  kab-tahun tanpa klaster: {len(panel) - len(pj)}")

    # --- 7. simpan (format dashboard)
    out = assign.merge(F[["kabupaten"] + [c for c in FITUR if c in F] + ["produksi_ton", "pembudidaya", "harga_rp_kg"]].drop_duplicates("kabupaten"),
                       on="kabupaten", how="left")
    save_parquet(out, REPORT_DIR / "cluster_assignments.parquet")
    cy = pj[["kabupaten", "tahun", "klaster", "label"]].rename(columns={"label": "label_klaster"}).merge(
        panel[["kabupaten", "tahun", "produksi_media_ton", "rtp", "pembudidaya", "luas_lahan_ha", "produksi_ton", "harga_rp_kg", "shannon", "h_kolam", "h_tambak"]].rename(
            columns={"produksi_media_ton": "produksi_ton", "produksi_ton": "total_pembesaran"}), on=["kabupaten", "tahun"], how="left")
    cy["klaster"] = cy.klaster.astype(int)
    cy["segmen_khusus"] = cy.kabupaten.isin(spesialis)
    save_parquet(cy, REPORT_DIR / "cluster_assignments_by_year.parquet")
    save_csv(prof.reset_index().rename(columns={"klaster": "id"}), REPORT_DIR / "cluster_profile.csv")
    joblib.dump({"scaler": scaler, "kmeans": km, "fitur": FITUR, "labels": labels_all, "khusus": spesialis, "set": SET_PILIH, "k": k_opt},
                MODEL_DIR / "clustering_kmeans.pkl")
    save_json({"set": SET_PILIH, "k": k_opt, "silhouette": sil, "fitur": FITUR, "khusus": spesialis,
               "n_klaster_tampil": int(k_opt + (1 if spesialis else 0)), "min_size": MIN_SIZE, "tidak_diklaster": tidak_diklaster},
              REPORT_DIR / "clustering_meta.json")

    pca = PCA(2, random_state=RS).fit(X)
    return {"k_opt": k_opt, "silhouette": sil, "eval_df": ev, "profil": prof, "label_map": labels_all,
            "df": cy, "assign": assign, "kmeans": km, "scaler": scaler, "pca": pca, "X_pca": pca.transform(X), "fitur": FITUR}