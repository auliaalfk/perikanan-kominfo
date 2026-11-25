import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CONFIG, MODEL_DIR, PROCESSED_DIR, REPORT_DIR
from .features import build_kab_features, build_panel, build_reg_komoditas, build_validasi_tahun_baru, feature_registry, outlier_kab
from .io_utils import save_csv, save_json

_R_DEFAULT = {"set_tolerance": 0.01, "ml_beat_baseline": 0.10, "min_rows_m2": 30, "min_kab_m2": 10, "min_rows_m3": 30, "horizon": 3}


# ----------------------------------------------------------------------------- evaluasi
def _metrics(y, p, mask=None):
    """y, p skala log. R2_asli = R2 pada skala asli (dipakai kolom r2_test dashboard)."""
    if mask is not None:
        y, p = y[mask], p[mask]
    return dict(R2_log=r2_score(y, p), R2_asli=r2_score(np.exp(y), np.exp(p)), RMSE_log=float(np.sqrt(np.mean((y - p) ** 2))),
                MAE_asli=float(np.mean(np.abs(np.exp(y) - np.exp(p)))), medAPE=float(np.median(np.abs(np.exp(p) - np.exp(y)) / np.exp(y))))


def _oof_grouped(model, X, y, groups, repeats, n_splits=5, seed=0):
    """Grouped K-fold berulang: seluruh baris satu kabupaten selalu di fold yang sama -> (repeats, n) prediksi out-of-fold."""
    ug = np.array(sorted(set(groups)))
    n_splits = min(n_splits, len(ug))
    out = np.zeros((repeats, len(y)))
    for r in range(repeats):
        fold_of = dict(zip(ug, np.random.default_rng(seed + r).permutation(len(ug)) % n_splits))
        f = np.array([fold_of[g] for g in groups])
        for k in range(n_splits):
            tr, te = np.where(f != k)[0], np.where(f == k)[0]
            out[r, te] = clone(model).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    return out


def _summarize(name, y, oof, mask=None):
    d = pd.DataFrame([_metrics(y, oof[r], mask) for r in range(oof.shape[0])])
    return {"model": name, **{c: d[c].mean() for c in d}, **{c + "_sd": d[c].std() for c in ["R2_log", "medAPE"]}}


def _prep(num, cat=()):
    tr = [("num", StandardScaler(), list(num))]
    if cat:
        tr.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), list(cat)))
    return ColumnTransformer(tr)


# ----------------------------------------------------------------------------- M1
def _m1(rk, REPEATS, N_TREE, N_BOOT):
    y1, g1 = rk.l_nilai.values, rk.kabupaten.values
    M1 = {"Volume saja (Ridge)": (Pipeline([("p", _prep(["l_vol"])), ("m", Ridge(1.0))]), ["l_vol"]),
          "Volume + komoditas (Ridge)": (Pipeline([("p", _prep(["l_vol"], ["komoditas"])), ("m", Ridge(1.0))]), ["l_vol", "komoditas"]),
          "Volume + komoditas (Huber)": (Pipeline([("p", _prep(["l_vol"], ["komoditas"])), ("m", HuberRegressor(alpha=1e-3, max_iter=2000))]), ["l_vol", "komoditas"]),
          "Volume + komoditas (RandomForest)": (Pipeline([("p", _prep(["l_vol"], ["komoditas"])), ("m", RandomForestRegressor(N_TREE, min_samples_leaf=3, random_state=0, n_jobs=-1))]), ["l_vol", "komoditas"])}
    res, oof = [], {}
    for nm, (mdl, cols) in M1.items():
        oof[nm] = _oof_grouped(mdl, rk[cols], y1, g1, REPEATS)
        res.append(_summarize(nm, y1, oof[nm]))
    res = pd.DataFrame(res).set_index("model")

    def elas(df, dengan_kom):
        X = pd.get_dummies(df[["l_vol", "komoditas"]], columns=["komoditas"], drop_first=True, dtype=float) if dengan_kom else df[["l_vol"]]
        return LinearRegression().fit(X, df.l_nilai).coef_[list(X.columns).index("l_vol")]
    rng = np.random.default_rng(0)
    kabs = rk.kabupaten.unique()
    groups = {k: d for k, d in rk.groupby("kabupaten")}
    el = {}
    for dk in [False, True]:
        b = [elas(pd.concat([groups[k] for k in rng.choice(kabs, len(kabs), replace=True)]), dk) for _ in range(N_BOOT)]
        el[dk] = (float(elas(rk, dk)), *[float(x) for x in np.percentile(b, [2.5, 97.5])])
    best = res.R2_log.idxmax()
    p1 = oof[best].mean(axis=0)
    resid = rk.assign(pred_nilai_juta=np.exp(p1), resid_log=y1 - p1)
    return {"res": res, "el": el, "best": best, "models": M1, "resid": resid}


# ----------------------------------------------------------------------------- M2
def _m2(panel, reg, R, REPEATS, N_TREE, outliers):
    d2 = panel.copy()
    for c in ["h_kolam", "h_tambak", "h_laut"]:
        d2[c] = d2[c].fillna(d2[c].median() if d2[c].notna().any() else 0.0)
    share = [c for c in d2.columns if c.startswith("share_") and d2[c].std() > 0]
    if "share_lainnya" in share:
        share.remove("share_lainnya")
    elif share:
        share = share[:-1]                       # share berjumlah 1 -> buang satu (acuan) agar tidak kolinear sempurna
    S = {"S1 pembudidaya": ["l_pembudidaya"], "S2 + komposisi komoditas": ["l_pembudidaya", "shannon"] + share}
    S["S3 + habitat"] = S["S2 + komposisi komoditas"] + ["h_kolam", "h_tambak", "h_laut"]
    luas_ok = bool(reg.loc["luas_lahan_ha", "usable_panel"]) and d2.l_luas.notna().sum() >= 40
    if luas_ok:
        S["S4 + luas lahan"] = S["S3 + habitat"] + ["l_luas"]
    need = sorted(set(sum(S.values(), [])))
    d2c = d2.dropna(subset=need).reset_index(drop=True)
    print(f"  M2: sampel bersama {len(d2c)} dari {len(d2)} baris ({d2c.kabupaten.nunique()} kab) | set: {list(S)}" + ("" if luas_ok else " | luas belum memadai -> S4 tidak diuji"))
    if not (len(d2c) >= R["min_rows_m2"] and d2c.kabupaten.nunique() >= R["min_kab_m2"]):
        print("  M2 dilewati: data lengkap terlalu sedikit.")
        return None
    MD = {"Ridge": Ridge(3.0), "Huber": HuberRegressor(alpha=1.0, max_iter=3000),
          "RandomForest": RandomForestRegressor(N_TREE, min_samples_leaf=3, random_state=0, n_jobs=-1),
          "GradBoost": GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)}
    y2, g2 = d2c.l_produksi_ton.values, d2c.kabupaten.values
    ex_out = ~np.isin(g2, outliers)
    oof, rows = {}, []
    oof["Baseline rata-rata"] = _oof_grouped(DummyRegressor(), d2c[["l_pembudidaya"]], y2, g2, REPEATS)
    r = _summarize("Baseline rata-rata", y2, oof["Baseline rata-rata"])
    r["medAPE_tanpa_outlier"] = _metrics(y2, oof["Baseline rata-rata"].mean(0), ex_out)["medAPE"]
    rows.append(r)
    for sn, cols in S.items():
        for mn, m in MD.items():
            nm = f"{sn} | {mn}"
            oof[nm] = _oof_grouped(Pipeline([("s", StandardScaler()), ("m", m)]), d2c[cols], y2, g2, REPEATS)
            r = _summarize(nm, y2, oof[nm])
            r["medAPE_tanpa_outlier"] = float(np.mean([_metrics(y2, oof[nm][i], ex_out)["medAPE"] for i in range(oof[nm].shape[0])]))
            rows.append(r)
    tab = pd.DataFrame(rows).set_index("model")
    sc = tab.drop("Baseline rata-rata").copy()
    sc["set"] = [i.split(" | ")[0] for i in sc.index]
    best_per_set = sc.groupby("set").R2_log.max()
    top = best_per_set.max()
    set2 = [s for s in S if s in best_per_set.index and best_per_set[s] >= top - R["set_tolerance"]][0]      # set paling sederhana yang tidak kalah > toleransi
    best2 = sc[sc.set == set2].R2_log.idxmax()
    print(f"  M2: set terpilih {set2} (R2 terbaik antar-set {top:.3f}) | {best2} | R2(log)={tab.loc[best2, 'R2_log']:.3f}, medAPE={tab.loc[best2, 'medAPE']:.2f}")
    if "S4 + luas lahan" in best_per_set.index:
        print(f"  M2: nilai tambah luas lahan: S4 {best_per_set['S4 + luas lahan']:.3f} vs S3 {best_per_set['S3 + habitat']:.3f} ({best_per_set['S4 + luas lahan'] - best_per_set['S3 + habitat']:+.3f})")
    d2c["resid_oof"] = y2 - oof[best2].mean(axis=0)
    rkab = d2c.groupby("kabupaten").agg(produksi_ton=("produksi_ton", "mean"), resid_log=("resid_oof", "mean")).sort_values("resid_log")
    rkab["produksi_ekspektasi_ton"] = rkab.produksi_ton / np.exp(rkab.resid_log)
    sn_, mn_ = best2.split(" | ")
    dfit = d2.dropna(subset=S[sn_])
    final = Pipeline([("s", StandardScaler()), ("m", clone(MD[mn_]))]).fit(dfit[S[sn_]], dfit.l_produksi_ton.values)
    return {"tab": tab, "sc": sc, "best": best2, "set": set2, "S": S, "rkab": rkab, "n": len(d2c), "final": final, "fitur": S[sn_]}


# ----------------------------------------------------------------------------- M3
def _mdl_growth(kind, N_TREE):
    if kind == "Ridge pertumbuhan":
        return Pipeline([("s", StandardScaler()), ("m", Ridge(10.0))])
    if kind == "Huber pertumbuhan":
        return Pipeline([("s", StandardScaler()), ("m", HuberRegressor(alpha=1.0, max_iter=3000))])
    return RandomForestRegressor(N_TREE, min_samples_leaf=3, random_state=0, n_jobs=-1)


def _m3(pan, vnew, R, N_TREE, outliers, LAST):
    lagd = pan[["kabupaten", "tahun", "l_produksi_ton", "l_pembudidaya", "shannon", "top_share"]].copy()
    lagd["tahun"] += 1
    lagd.columns = ["kabupaten", "tahun", "lag", "lpem_lag", "shan_lag", "top_lag"]
    g = pan.merge(lagd, on=["kabupaten", "tahun"], how="left")
    g["d"] = g.l_produksi_ton - g.lag
    g["ppp_lag"] = g.lag - g.lpem_lag
    FEc = ["lag", "lpem_lag", "shan_lag", "top_lag", "ppp_lag"]
    dd = g.dropna(subset=["lag"]).reset_index(drop=True)
    dd_fe = dd.dropna(subset=FEc)
    if not (dd.tahun.nunique() >= 2 and len(dd) >= R["min_rows_m3"]):
        print("  M3 dilewati: perlu >= 3 tahun data & >= %d baris pertumbuhan." % R["min_rows_m3"])
        return None
    KINDS = ["Ridge pertumbuhan", "Huber pertumbuhan", "RF pertumbuhan"]
    rows, pl = [], []
    for t in sorted(dd.tahun.unique())[1:]:
        tr, te = dd[dd.tahun < t], dd[dd.tahun == t]
        trf, tef = dd_fe[dd_fe.tahun < t], te.dropna(subset=FEc)
        cand = {"Naif (tahun lalu)": (te, te.lag.values), "Naif + rata-rata pertumbuhan": (te, te.lag.values + tr.d.mean())}
        for k in KINDS:
            if len(trf) >= 20 and len(tef):
                cand[k] = (tef, tef.lag.values + clone(_mdl_growth(k, N_TREE)).fit(trf[FEc], trf.d).predict(tef[FEc]))
        for nm, (tt, p) in cand.items():
            m = _metrics(tt.l_produksi_ton.values, p)
            nx = ~tt.kabupaten.isin(outliers).values
            m_ex = _metrics(tt.l_produksi_ton.values, p, nx) if nx.sum() > 3 else m
            rows.append({"tahun_uji": t, "model": nm, "n_latih": len(tr), **m, "medAPE_tanpa_outlier": m_ex["medAPE"]})
            pl.append(pd.DataFrame({"tahun": t, "model": nm, "kabupaten": tt.kabupaten.values, "y": tt.l_produksi_ton.values, "p": p}))
    roll, preds = pd.DataFrame(rows), pd.concat(pl)
    agg3 = roll.groupby("model")[["R2_log", "R2_asli", "MAE_asli", "medAPE", "medAPE_tanpa_outlier"]].mean().sort_values("medAPE")
    ref = agg3.loc["Naif + rata-rata pertumbuhan", "medAPE"]
    ml = agg3.loc[[k for k in KINDS if k in agg3.index], "medAPE"]
    if len(ml) and ml.min() <= (1 - R["ml_beat_baseline"]) * ref:
        FINAL3 = ml.idxmin()
        alasan = f"{FINAL3} mengalahkan baseline >= {R['ml_beat_baseline']:.0%} ({ml.min():.3%} vs {ref:.3%})"
    else:
        FINAL3 = "Naif + rata-rata pertumbuhan"
        alasan = f"model ML tidak mengungguli baseline >= {R['ml_beat_baseline']:.0%} ({(f'{ml.min():.3%}' if len(ml) else 'n/a')} vs {ref:.3%}) -> baseline (lebih sederhana, sama akurat)"
    print(f"  M3: model {FINAL3} | medAPE rolling-origin {agg3.loc[FINAL3, 'medAPE']:.2%}")

    last = pan[pan.tahun == LAST].copy()
    last["lag"], last["lpem_lag"], last["shan_lag"], last["top_lag"] = last.l_produksi_ton, last.l_pembudidaya, last.shannon, last.top_share
    last["ppp_lag"] = last.lag - last.lpem_lag
    if FINAL3 == "Naif + rata-rata pertumbuhan":
        mu = dd.d.mean(); pred = last.lag.values + mu; final_obj = {"tipe": "naif_growth", "growth": float(mu)}
    elif FINAL3 == "Naif (tahun lalu)":
        pred = last.lag.values; final_obj = {"tipe": "naif"}
    else:
        m = clone(_mdl_growth(FINAL3, N_TREE)).fit(dd_fe[FEc], dd_fe.d)
        last = last.dropna(subset=FEc); pred = last.lag.values + m.predict(last[FEc]); final_obj = {"tipe": FINAL3, "model": m, "fitur": FEc}
    err = preds[preds.model == FINAL3]
    err = (err.y - err.p).values
    q_lo, q_hi = np.quantile(err, [0.10, 0.90])
    fc = pd.DataFrame({"kabupaten": last.kabupaten.values, f"produksi_{LAST}_ton": np.exp(last.l_produksi_ton.values), f"prakiraan_{LAST + 1}_ton": np.exp(pred),
                       "batas_bawah_80": np.exp(pred + q_lo), "batas_atas_80": np.exp(pred + q_hi)})
    va = vnew[vnew.tahun == LAST + 1].rename(columns={"produksi_ton": "aktual_tahun_baru_opendata"})[["kabupaten", "aktual_tahun_baru_opendata"]]
    fc = fc.merge(va, on="kabupaten", how="left")
    v = fc.dropna(subset=["aktual_tahun_baru_opendata"])
    ape = pd.Series(dtype=float)
    if len(v):
        ape = (v[f"prakiraan_{LAST + 1}_ton"] - v.aktual_tahun_baru_opendata).abs() / v.aktual_tahun_baru_opendata
        cover = ((v.aktual_tahun_baru_opendata >= v.batas_bawah_80) & (v.aktual_tahun_baru_opendata <= v.batas_atas_80)).mean()
        print(f"  M3: validasi {LAST + 1}: {len(v)}/{len(fc)} kab | median APE {ape.median():.1%} | cakupan interval 80% {cover:.0%}")
    else:
        print(f"  M3: belum ada data pembanding untuk {LAST + 1}.")

    # proyeksi 3 tahun (random walk with drift; 3 skenario dari sebaran pertumbuhan antar-kabupaten)
    HORIZON = [LAST + h for h in range(1, int(R["horizon"]) + 1)]
    gk = dd.d.values
    SK = {"Pesimis": float(np.quantile(gk, 0.25)), "Dasar": float(np.mean(gk)), "Optimis": float(np.quantile(gk, 0.75))}
    lastp = pan[pan.tahun == LAST].set_index("kabupaten")
    out = []
    for sk, mu_s in SK.items():
        for yr in HORIZON:
            h = yr - LAST
            lp = lastp.l_produksi_ton + mu_s * h
            d = pd.DataFrame({"kabupaten": lastp.index, "tahun": yr, "skenario": sk, "produksi_ton": np.exp(lp.values)})
            d["batas_bawah_80"] = np.exp(lp.values + q_lo * np.sqrt(h)); d["batas_atas_80"] = np.exp(lp.values + q_hi * np.sqrt(h))
            d[f"harga_rp_kg_{LAST}"] = lastp.harga_rp_kg.values
            out.append(d)
    proj = pd.concat(out, ignore_index=True)
    pt_ = pan.groupby("tahun").agg(n=("nilai_juta", "sum"), p=("produksi_ton", "sum"))
    hr = pt_.n / pt_.p
    g_harga = float(np.log(hr.iloc[-1] / hr.iloc[0]) / max(1, len(hr) - 1))
    proj["nilai_juta_harga_tetap"] = proj.produksi_ton * proj[f"harga_rp_kg_{LAST}"] / 1000
    proj["nilai_juta_harga_tren"] = proj.nilai_juta_harga_tetap * np.exp(g_harga * (proj.tahun - LAST))
    prov = proj.groupby(["skenario", "tahun"])[["produksi_ton", "nilai_juta_harga_tetap", "nilai_juta_harga_tren"]].sum().reset_index()
    return {"agg3": agg3, "final": FINAL3, "alasan": alasan, "final_obj": final_obj, "fc": fc, "ape": ape, "n_valid": len(v), "proj": proj, "prov": prov,
            "SK": SK, "g_harga": g_harga, "HORIZON": HORIZON}


# ----------------------------------------------------------------------------- utama
def run_all_regression(feat: pd.DataFrame = None, fact: pd.DataFrame = None, config=None) -> dict:
    warnings.filterwarnings("ignore")
    R = {**_R_DEFAULT, **((config or CONFIG).get("regresi") or {})}
    fast = os.environ.get("PIPE_FAST") == "1"
    N_TREE, N_BOOT, REPEATS = (100, 60, 2) if fast else (300, 200, 5)
    if feat is None:
        feat = pd.read_parquet(PROCESSED_DIR / "features_model.parquet")
    if fact is None:
        fact = pd.read_parquet(PROCESSED_DIR / "fact_budidaya.parquet")

    pan = build_panel(feat)
    reg = feature_registry(feat).set_index("fitur")
    rk = build_reg_komoditas(fact)
    if len(rk) < 30 or pan.empty:
        print(f"  Regresi dilewati: data terlalu sedikit (reg_komoditas={len(rk)}, panel={len(pan)})")
        return {"utama": None, "struktural": None, "forecast": None, "eval_all": pd.DataFrame()}
    YEARS = sorted(int(y) for y in pan.tahun.unique())
    LAST = YEARS[-1]
    outliers = outlier_kab(pan)
    vnew = build_validasi_tahun_baru(fact, YEARS)
    print(f"  reg_komoditas {rk.shape} | panel {pan.shape} | tahun {YEARS} | outlier struktural: {outliers}")

    # ---- M1
    m1 = _m1(rk, REPEATS, N_TREE, N_BOOT)
    res1, el, best1 = m1["res"], m1["el"], m1["best"]
    print(f"  M1: {best1} | CV-R2 {res1.loc[best1, 'R2_log']:.3f} (volume saja {res1.loc['Volume saja (Ridge)', 'R2_log']:.3f}) | elastisitas {el[True][0]:.3f} (CI {el[True][1]:.3f}-{el[True][2]:.3f})")
    # ---- M2, M3
    m2 = _m2(pan, reg, R, REPEATS, N_TREE, outliers)
    m3 = _m3(pan, vnew, R, N_TREE, outliers, LAST)

    # ---- simulator what-if (nilai_baru = nilai_dasar * (1+delta)^elastisitas)
    base_nilai, base_prod = float(pan[pan.tahun == LAST].nilai_juta.sum()), float(pan[pan.tahun == LAST].produksi_ton.sum())
    e_pt, e_lo, e_hi = el[True]

    def sim(delta):
        f = 1 + delta / 100
        lohi = sorted([base_nilai * f ** e_lo / 1e6, base_nilai * f ** e_hi / 1e6])
        return {"delta_volume_pct": delta, "nilai_baru_Rp_T": base_nilai * f ** e_pt / 1e6, "nilai_batas_bawah_Rp_T": lohi[0], "nilai_batas_atas_Rp_T": lohi[1], "tambahan_Rp_T": base_nilai * (f ** e_pt - 1) / 1e6}
    save_csv(pd.DataFrame([sim(x) for x in [-20, -10, -5, 5, 10, 15, 20, 30]]), REPORT_DIR / "scenario_table.csv")
    save_json({"elastisitas": e_pt, "ci95": [e_lo, e_hi], "tahun_dasar": LAST, "base_nilai_juta": base_nilai, "base_produksi_ton": base_prod,
               "cv_r2_M1": float(res1.loc[best1, "R2_log"]), "rumus": "nilai_baru = nilai_dasar * (1 + delta)^elastisitas"}, REPORT_DIR / "scenario_params.json")

    # ---- regression_eval.csv (SKEMA DASHBOARD). Dataset pertama = dataset utama; 'coef' baris terbaik HARUS terisi (elastisitas simulator).
    rows = []
    for nm in res1.index:
        rows.append({"dataset": "nilai_vs_volume", "model": nm, "cv_r2_mean": res1.loc[nm, "R2_log"], "r2_test": res1.loc[nm, "R2_asli"], "coef": el["komoditas" in nm][0],
                     "medape": res1.loc[nm, "medAPE"], "rmse_log": res1.loc[nm, "RMSE_log"], "n": len(rk), "terpilih": nm == best1})
    if m2:
        for s_, grp in m2["sc"].groupby("set", sort=False):
            b = grp.R2_log.idxmax()
            rows.append({"dataset": "produksi_struktural", "model": b.replace(" | ", " / "), "cv_r2_mean": grp.loc[b, "R2_log"], "r2_test": grp.loc[b, "R2_asli"], "coef": np.nan,
                         "medape": grp.loc[b, "medAPE"], "rmse_log": grp.loc[b, "RMSE_log"], "n": m2["n"], "terpilih": b == m2["best"]})
    reg_eval = pd.DataFrame(rows)
    top = reg_eval[reg_eval.dataset == "nilai_vs_volume"].sort_values("cv_r2_mean", ascending=False).iloc[0]
    assert pd.notna(top.coef), "coef baris regresi terbaik kosong -> dashboard akan memakai elastisitas 1.0"
    save_csv(reg_eval, REPORT_DIR / "regression_eval.csv")

    detail = [res1.reset_index().assign(tugas="M1 nilai", terpilih=lambda d: d.model == best1)]
    if m2:
        detail.append(m2["tab"].reset_index().assign(tugas="M2 produksi struktural", terpilih=lambda d: d.model == m2["best"]))
    if m3:
        detail.append(m3["agg3"].reset_index().assign(tugas="M3 forecast 1 tahun (rolling-origin)", terpilih=lambda d: d.model == m3["final"]))
    save_csv(pd.concat(detail, ignore_index=True), REPORT_DIR / "regression_eval_detail.csv")
    save_csv(pd.DataFrame([["M1 nilai ~ volume (tanpa komoditas)", "elastisitas", *el[False]], ["M1 nilai ~ volume (dengan komoditas)", "elastisitas", *el[True]]],
                          columns=["model", "parameter", "estimasi", "ci95_lo", "ci95_hi"]), REPORT_DIR / "regression_coef.csv")
    if m2:
        m2["rkab"].to_csv(REPORT_DIR / "produksi_residual_kab.csv")
        joblib.dump({"model": m2["final"], "fitur": m2["fitur"], "set": m2["set"]}, MODEL_DIR / "regressor_M2_produksi_struktural.pkl")
    if m3:
        save_csv(m3["fc"], REPORT_DIR / f"forecast_produksi_{LAST + 1}.csv"); save_csv(m3["proj"], REPORT_DIR / "proyeksi_kabupaten.csv"); save_csv(m3["prov"], REPORT_DIR / "proyeksi_provinsi.csv")
        joblib.dump(m3["final_obj"], MODEL_DIR / "forecast_M3.pkl")
    mdl, cols = m1["models"][best1]
    joblib.dump(mdl.fit(rk[cols], rk.l_nilai.values), MODEL_DIR / "regressor_M1_nilai.pkl")

    ins = [f"M1 Nilai ~ volume + komoditas: CV-R2 {res1.loc[best1, 'R2_log']:.3f} (volume saja {res1.loc['Volume saja (Ridge)', 'R2_log']:.3f}); elastisitas {e_pt:.2f} (CI {e_lo:.2f}-{e_hi:.2f}).",
           (f"M2 Produksi struktural ({m2['best']}): CV-R2 {m2['tab'].loc[m2['best'], 'R2_log']:.3f}, median APE {m2['tab'].loc[m2['best'], 'medAPE']:.0%} - EXPLANATORY (membandingkan kabupaten), TIDAK cocok untuk prediksi titik." if m2 else "M2 dilewati (data kurang)."),
           (f"M3 Prakiraan 1 tahun: {m3['final']}; median APE rolling-origin {m3['agg3'].loc[m3['final'], 'medAPE']:.1%}. {m3['alasan']}." if m3 else "M3 dilewati (butuh >= 3 tahun data).")]
    if m3 and len(m3["ape"]):
        ins.append(f"Validasi {LAST + 1} (opendata): median APE {m3['ape'].median():.1%} pada {m3['n_valid']} kabupaten.")
    if m3:
        pv = m3["prov"][m3["prov"].skenario == "Dasar"].sort_values("tahun")
        ins.append("Proyeksi produksi Jatim (skenario dasar): " + "; ".join(f"{int(r.tahun)} {r.produksi_ton:,.0f}" for r in pv.itertuples()) + f" ton. Elastisitas simulator {e_pt:.2f}.")
    save_json({"insights": ins}, REPORT_DIR / "regression_insight.json")
    meta = {}
    try:
        import json
        meta = json.load(open(REPORT_DIR / "run_meta.json"))
    except Exception:
        pass
    save_json({"data_hash": meta.get("data_hash", ""), "model_name": f"M1:{best1}", "cv_r2": float(res1.loc[best1, "R2_log"]), "notes": " | ".join(ins)[:900]}, REPORT_DIR / "model_version_row.json")
    for s_ in ins:
        print("  -", s_)

    best_row = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in top.to_dict().items()}
    return {"utama": {"name": "Nilai vs Volume", "best_row": best_row, "res_df": res1, "elastisitas": el, "coef_info": {"coef_unscaled_per_unit": e_pt, "interpretasi": f"Elastisitas: +1% volume -> +{e_pt:.2f}% nilai"}},
            "struktural": ({"name": "Produksi struktural", "best_row": {**m2["tab"].loc[m2["best"]].to_dict(), "cv_r2_mean": float(m2["tab"].loc[m2["best"], "R2_log"])}, "res_df": m2["tab"]} if m2 else None),
            "forecast": m3, "eval_all": reg_eval}