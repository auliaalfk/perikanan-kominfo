import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, REPORT_DIR, CONFIG, KOMODITAS_GRUP
from .io_utils import save_parquet, save_csv

FITUR_NUM = ["produksi_ton", "nilai_rp_juta", "rtp", "pembudidaya", "luas_lahan_ha", "total_pembesaran"]
DB_SHARES = ["rumput_laut", "udang", "bandeng", "lele", "nila", "gurami", "patin", "mas"]
DB_COLS = ["kabupaten", "tahun"] + FITUR_NUM + ["share_" + s for s in DB_SHARES]
EXTRA_COLS = ["luas_sumber", "nilai_kkp_juta", "harga_rp_kg", "shannon", "n_komoditas", "top_share", "grp_lainnya",
              "h_kolam", "h_tambak", "h_laut", "h_minapadi", "h_kja", "pembenihan_ribu_ekor", "ikan_hias_ekor", "flag_lonjakan_pembudidaya"]
_FIT = {"cover_cluster": 0.8, "cover_panel": 0.5, "nilai_fallback_kkp": False, **(CONFIG.get("fitur") or {})}
_IDX = ["kabupaten", "tahun"]


def _empty(name):
    return pd.Series(dtype=float, name=name, index=pd.MultiIndex.from_arrays([[], []], names=_IDX))


def _agg(df, name):
    if df is None or df.empty:
        return _empty(name)
    return df.groupby(_IDX)["nilai"].sum(min_count=1).rename(name)


def _pembudidaya(fact):
    """Pembudidaya valid + nilai lonjakan (is_suspect_scale, bukan tahun rusak) diisi median kabupaten dari titik valid."""
    pm = fact[fact["metrik"] == "pembudidaya"]
    good = pm[~pm.is_suspect_scale & ~pm.is_missing_year & pm.nilai.notna()]
    med = good.groupby("kabupaten")["nilai"].median()
    bad = pm[pm.is_suspect_scale & ~pm.is_missing_year].copy()
    bad["nilai"] = bad["kabupaten"].map(med)
    bad = bad.dropna(subset=["nilai"])
    s = _agg(good, "pembudidaya")
    imp = _agg(bad, "pembudidaya")
    out = pd.concat([s, imp[~imp.index.isin(s.index)]]).sort_index()
    flag = pd.Series(out.index.isin(imp.index) & ~out.index.isin(s.index), index=out.index, name="flag_lonjakan_pembudidaya")
    return out, flag


def _luas_kanonik(fact):
    """Satu nilai luas per kab-tahun. KKP diutamakan; OpenData mengisi hanya bila satuannya terbukti sebanding (rasio median 0,5-2 pada kab-tahun yang sama-sama valid).
       Kab-tahun yang salah satu barisnya ditandai janggal tidak dipakai (jumlah parsial = menyesatkan)."""
    lu = fact[(fact["metrik"] == "luas_lahan_ha") & fact["tahun"].notna()]
    if lu.empty:
        return _empty("luas_lahan_ha"), pd.Series(dtype=object, name="luas_sumber", index=_empty("x").index)
    parsial = set(zip(lu[lu.is_suspect_scale]["sumber"], lu[lu.is_suspect_scale]["kabupaten"], lu[lu.is_suspect_scale]["tahun"]))
    ok = lu[~lu.is_suspect_scale & ~lu.is_missing_year]
    od, kk = _agg(ok[ok.sumber == "luas"], "od"), _agg(ok[ok.sumber == "luas_kkp"], "kk")
    for s, nm in ((od, "luas"), (kk, "luas_kkp")):
        drop = [i for i in s.index if (nm, i[0], i[1]) in parsial]
        if drop:
            s.loc[drop] = np.nan
            print(f"  [WARN] luas ({nm}): {len(drop)} kab-tahun tidak dipakai karena ada baris luasnya yang janggal (jumlah parsial menyesatkan)")
    both = pd.concat([kk, od], axis=1)
    ov = both.dropna()
    ov = ov[(ov.kk > 0) & (ov.od > 0)]
    comparable = len(ov) >= 5 and 0.5 <= (ov.od / ov.kk).median() <= 2
    if len(ov) >= 5:
        print(f"  luas: rasio median OpenData/KKP = {(ov.od / ov.kk).median():.2f} (n={len(ov)}) -> " + ("sebanding, OpenData boleh mengisi" if comparable else "TIDAK sebanding, tidak dicampur"))
    use_od = both.od if (comparable or both.kk.isna().all()) else pd.Series(np.nan, index=both.index)
    luas = both.kk.where(both.kk.notna(), use_od).rename("luas_lahan_ha")
    sumber = pd.Series(np.where(both.kk.notna(), "kkp", np.where(luas.notna(), "opendata", None)), index=both.index, name="luas_sumber")
    return luas, sumber


def _pembesaran(fact):
    """Dari KKP pembesaran: total volume, nilai, bagian per grup komoditas, shannon, top_share, n_komoditas."""
    kp = fact[fact["sumber"] == "kkp_pembesaran"]
    vol = kp[(kp["metrik"] == "produksi_ton") & kp["nilai"].notna()].copy()
    nil = kp[kp["metrik"] == "nilai_rp_juta"]
    grp = vol["komoditas"].map(KOMODITAS_GRUP)
    baru = sorted(set(vol.loc[grp.isna(), "komoditas"].dropna()))
    if baru:
        print(f"  [INFO] komoditas belum terdaftar di KOMODITAS_GRUP -> masuk 'lainnya': {baru}")
    vol["grup"] = grp.fillna("lainnya")
    G = vol.pivot_table(index=_IDX, columns="grup", values="nilai", aggfunc="sum", fill_value=0)
    tot = G.sum(axis=1)
    P = G.div(tot.where(tot > 0), axis=0)
    out = pd.DataFrame(index=G.index)
    out["total_kkp"] = tot.where(tot > 0)
    for s in DB_SHARES:
        out["share_" + s] = P[s] if s in P else np.where(out.total_kkp.notna(), 0.0, np.nan)
    out["grp_lainnya"] = P["lainnya"] if "lainnya" in P else np.where(out.total_kkp.notna(), 0.0, np.nan)
    out["shannon"] = -(P * np.log(P.where(P > 0))).sum(axis=1)
    out["top_share"] = P.max(axis=1)
    out["n_komoditas"] = vol[vol.nilai > 0].groupby(_IDX)["komoditas"].nunique()
    out["nilai_kkp_juta"] = _agg(nil, "nilai_kkp_juta")
    return out


def _habitat(fact, kabs):
    """Komposisi habitat per kabupaten dari produksi per jenis budidaya (tahun valid saja)."""
    j = fact[(fact["sumber"] == "jumlah") & (fact["metrik"] == "produksi_ton") & fact.jenis_budidaya.notna()]
    j = j.groupby(["kabupaten", "jenis_budidaya"])["nilai"].sum().unstack()
    H = pd.DataFrame(index=kabs)
    for c in ["KOLAM", "TAMBAK", "LAUT", "MINA PADI", "JARING APUNG", "KARAMBA"]:
        H[c] = j[c] if c in j else np.nan
    H = H.reindex(kabs)
    tot = H.sum(axis=1, min_count=1)
    H = H.div(tot.where(tot > 0), axis=0)
    return pd.DataFrame({"h_kolam": H["KOLAM"], "h_tambak": H["TAMBAK"], "h_laut": H["LAUT"], "h_minapadi": H["MINA PADI"],
                         "h_kja": H[["JARING APUNG", "KARAMBA"]].sum(axis=1, min_count=1)})


def _median_vol(fact, sumber, metrik, kabs):
    """Median volume antar-tahun VALID; kabupaten tak tercatat = 0 struktural. Tahun rusak (baris sudah terfilter) tidak ikut."""
    d = fact[(fact["sumber"] == sumber) & (fact["metrik"] == metrik)]
    if d.empty:
        return pd.Series(0.0, index=kabs)
    W = d.groupby(["kabupaten", "tahun"])["nilai"].sum(min_count=1).unstack()
    W = W[[y for y in W.columns if W[y].notna().any()]]
    return W.reindex(kabs).fillna(0).median(axis=1) if W.shape[1] else pd.Series(0.0, index=kabs)


def build_features_model(fact: pd.DataFrame) -> pd.DataFrame:
    fall = fact[fact["tahun"].notna()].copy()
    pemb, pemb_flag = _pembudidaya(fall)
    f = fall[(~fall["is_suspect_scale"]) & (~fall["is_missing_year"])]

    prod = _agg(f[(f["sumber"] == "jumlah") & (f["metrik"] == "produksi_ton")], "produksi_ton")
    rtp = _agg(f[f["metrik"] == "rtp"], "rtp")
    luas, luas_sumber = _luas_kanonik(fall)
    pb = _pembesaran(f)
    bk = _agg(f[(f["sumber"] == "bps_kegiatan") & (f["kegiatan"] == "pembesaran") & (f["metrik"] == "produksi_ton")], "bps_pembesaran")
    nilai_bps = (_agg(f[(f["sumber"] == "bps_nilai") & (f["metrik"] == "nilai_rp")], "x") / 1e6).rename("nilai_bps")

    feat = pd.concat([prod, rtp, pemb, pemb_flag, luas, luas_sumber, pb, bk, nilai_bps], axis=1)
    feat["total_pembesaran"] = feat["total_kkp"].combine_first(feat["bps_pembesaran"])      # KKP utama; BPS hanya pengisi (TIDAK dijumlahkan)
    feat["nilai_rp_juta"] = feat["nilai_bps"]
    if _FIT["nilai_fallback_kkp"]:
        feat["nilai_rp_juta"] = feat["nilai_bps"].combine_first(feat["nilai_kkp_juta"])
    feat["harga_rp_kg"] = np.where((feat["total_kkp"] > 0) & (feat["nilai_kkp_juta"] > 0), feat["nilai_kkp_juta"] * 1000 / feat["total_kkp"], np.nan)
    feat["flag_lonjakan_pembudidaya"] = feat["flag_lonjakan_pembudidaya"].fillna(False).astype(bool)
    feat = feat.reset_index()

    kabs = sorted(feat["kabupaten"].unique())
    feat = feat.merge(_habitat(fall[(~fall.is_suspect_scale) & (~fall.is_missing_year)], kabs).rename_axis("kabupaten").reset_index(), on="kabupaten", how="left")
    feat["pembenihan_ribu_ekor"] = feat["kabupaten"].map(_median_vol(f, "kkp_pembenihan", "volume_ribu_ekor", kabs))
    feat["ikan_hias_ekor"] = feat["kabupaten"].map(_median_vol(f, "kkp_ikan_hias", "volume_ekor", kabs))

    feat = feat[feat[FITUR_NUM].notna().any(axis=1)].copy()
    feat["tahun"] = feat["tahun"].astype(int)
    feat = feat.sort_values(_IDX).reset_index(drop=True)
    for c in DB_COLS + EXTRA_COLS:
        if c not in feat:
            feat[c] = np.nan
    feat = feat[DB_COLS + EXTRA_COLS]
    save_parquet(feat, PROCESSED_DIR / "features_model.parquet")
    feature_registry(feat)
    return feat


# ----------------------------------------------------------------------------- registri fitur & panel untuk model
def feature_registry(feat):
    """Fitur mana yang BOLEH dipakai model (dihitung ulang tiap data bertambah). Disimpan ke output/reports/feature_registry.csv."""
    p = feat[feat["total_pembesaran"] > 0]
    cand = {"pembudidaya": "pembudidaya", "rtp": "rtp", "luas_lahan_ha": "luas_lahan_ha", "harga_rp_kg": "harga_rp_kg", "produksi_media_ton": "produksi_ton"}
    rows = []
    for nama, col in cand.items():
        rows.append({"fitur": nama,
                     "kab_cover": float(p.groupby("kabupaten")[col].apply(lambda s: s.notna().any()).mean()) if len(p) else 0.0,
                     "row_cover": float(p[col].notna().mean()) if len(p) else 0.0,
                     "tahun_valid": ", ".join(str(y) for y in sorted(p.loc[p[col].notna(), "tahun"].unique()))})
    reg = pd.DataFrame(rows)
    reg["usable_cluster"] = reg.kab_cover >= _FIT["cover_cluster"]
    reg["usable_panel"] = reg.row_cover >= _FIT["cover_panel"]
    save_csv(reg.round(3), REPORT_DIR / "feature_registry.csv")
    return reg


def build_panel(feat):
    """Panel kabupaten x tahun (tahun KKP yang valid) dengan nama kolom seperti notebook 02: produksi_ton = KKP, produksi_media_ton = OpenData."""
    p = feat[feat["total_pembesaran"] > 0].copy()
    p = p.rename(columns={"produksi_ton": "produksi_media_ton", "total_pembesaran": "produksi_ton", "nilai_kkp_juta": "nilai_juta",
                          "grp_lainnya": "share_lainnya"})
    pos = lambda s: s.where(s > 0)
    for c in ["produksi_ton", "nilai_juta", "pembudidaya", "harga_rp_kg"]:
        p["l_" + c] = np.log(pos(p[c]))
    p["l_luas"] = np.log(pos(p["luas_lahan_ha"]))
    p["l_rtp"] = np.log(pos(p["rtp"]))
    p["prod_per_pembudidaya"] = p["produksi_ton"] / pos(p["pembudidaya"])
    p["prod_per_ha"] = p["produksi_ton"] / pos(p["luas_lahan_ha"])
    return p.sort_values(_IDX).reset_index(drop=True)


def build_kab_features(panel):
    """Profil 1 baris per kabupaten (seluruh tahun valid) untuk clustering = `features_kab` di notebook 02."""
    pos = lambda s: s.where(s > 0)
    F = panel.groupby("kabupaten").agg(produksi_ton=("produksi_ton", "mean"), nilai_juta=("nilai_juta", "mean"), pembudidaya=("pembudidaya", "median"),
                                       rtp=("rtp", "median"), luas_lahan_ha=("luas_lahan_ha", "median"), shannon=("shannon", "mean"),
                                       n_komoditas=("n_komoditas", "mean"), top_share=("top_share", "mean"), n_tahun=("tahun", "nunique"),
                                       h_kolam=("h_kolam", "first"), h_tambak=("h_tambak", "first"), h_laut=("h_laut", "first"),
                                       h_minapadi=("h_minapadi", "first"), h_kja=("h_kja", "first"),
                                       pembenihan_ribu_ekor=("pembenihan_ribu_ekor", "first"), ikan_hias_ekor=("ikan_hias_ekor", "first"))
    F["harga_rp_kg"] = np.where(F.produksi_ton > 0, F.nilai_juta * 1000 / F.produksi_ton, np.nan)
    F["hab_imputed"] = F.h_kolam.isna()
    F["l_produksi"] = np.log(pos(F.produksi_ton)); F["l_pembudidaya"] = np.log(pos(F.pembudidaya)); F["l_harga"] = np.log(pos(F.harga_rp_kg))
    F["l_prod_per_pembudidaya"] = F.l_produksi - F.l_pembudidaya
    F["l_pembenihan"] = np.log1p(F.pembenihan_ribu_ekor); F["l_ikan_hias"] = np.log1p(F.ikan_hias_ekor)
    F["l_luas"] = np.log1p(F.luas_lahan_ha); F["l_rtp"] = np.log1p(F.rtp)
    return F.reset_index()


# ----------------------------------------------------------------------------- dataset tambahan untuk regresi (notebook 02)
def build_reg_komoditas(fact):
    """kab x komoditas x tahun (KKP pembesaran, hanya baris valid, volume & nilai > 0). Kunci unik."""
    kp = fact[(fact["sumber"] == "kkp_pembesaran") & fact["tahun"].notna() & ~fact["is_suspect_scale"] & ~fact["is_missing_year"]]
    k = ["tahun", "kabupaten", "komoditas"]
    v = kp[kp["metrik"] == "produksi_ton"].groupby(k)["nilai"].sum(min_count=1).rename("vol_ton")
    n = kp[kp["metrik"] == "nilai_rp_juta"].groupby(k)["nilai"].sum(min_count=1).rename("nilai_juta")
    rk = pd.concat([v, n], axis=1).reset_index()
    rk = rk[(rk.vol_ton > 0) & (rk.nilai_juta > 0)].copy()
    rk["tahun"] = rk["tahun"].astype(int)
    rk["l_vol"] = np.log(rk.vol_ton)
    rk["l_nilai"] = np.log(rk.nilai_juta)
    return rk.reset_index(drop=True)


def build_validasi_tahun_baru(fact, kkp_years):
    """Produksi OpenData per kab-tahun untuk tahun yang BELUM ada di KKP (mis. tahun terbaru): pembanding untuk validasi prakiraan."""
    j = fact[(fact["sumber"] == "jumlah") & (fact["metrik"] == "produksi_ton") & fact["tahun"].notna() & ~fact["is_suspect_scale"] & ~fact["is_missing_year"]]
    j = j[~j["tahun"].isin(set(int(y) for y in kkp_years))]
    out = j.groupby(["kabupaten", "tahun"])["nilai"].sum(min_count=1).rename("produksi_ton").reset_index().dropna()
    return out[out.produksi_ton > 0].reset_index(drop=True)


def outlier_kab(panel, z=3.5):
    """Kabupaten ekstrem asli (robust-z produksi atau harga rata-rata > z). Dipakai hanya untuk metrik 'tanpa outlier'; data TIDAK dibuang."""
    def rz(s):
        med = s.median(); mad = (s - med).abs().median() * 1.4826
        return (s - med) / mad if mad > 0 else s * 0
    kab = panel.groupby("kabupaten").agg(l_prod=("l_produksi_ton", "mean"), l_harga=("l_harga_rp_kg", "mean")).dropna(how="all")
    Z = kab.apply(rz)
    return sorted(Z.index[(Z.l_prod.abs() > z) | (Z.l_harga.abs() > z)].tolist()) if len(Z) else []