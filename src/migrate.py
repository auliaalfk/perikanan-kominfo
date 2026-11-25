import argparse
import json
import re

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, REPORT_DIR
from . import db


def _read_csv(name):
    p = REPORT_DIR / name
    return pd.read_csv(p) if p.exists() else None


def _read_json(name):
    p = REPORT_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ----------------------------------------------------------------------------- tabel hasil analisis
def _model_eval_frame():
    """Gabungkan regression_eval_detail.csv (semua model & metrik) dengan regression_eval.csv/regression_coef.csv (koefisien & selang)."""
    det, ev, cf = _read_csv("regression_eval_detail.csv"), _read_csv("regression_eval.csv"), _read_csv("regression_coef.csv")
    if det is None:
        return None
    nm = {"M1 nilai": "nilai_vs_volume", "M2 produksi struktural": "produksi_struktural", "M3 forecast 1 tahun (rolling-origin)": "prakiraan_1_tahun"}
    d = det.copy()
    d["dataset"] = d["tugas"].map(nm)
    d["model"] = d["model"].astype(str).str.replace(" | ", " / ", regex=False)
    out = pd.DataFrame({"dataset": d.dataset, "model": d.model, "cv_r2_mean": d.get("R2_log"), "r2_test": d.get("R2_asli"), "rmse_log": d.get("RMSE_log"),
                        "mae_asli": d.get("MAE_asli"), "med_ape": d.get("medAPE"), "med_ape_tanpa_outlier": d.get("medAPE_tanpa_outlier"),
                        "cv_r2_sd": d.get("R2_log_sd"), "med_ape_sd": d.get("medAPE_sd"), "terpilih": d.get("terpilih", False)})
    out["coef"] = np.nan; out["coef_lo"] = np.nan; out["coef_hi"] = np.nan
    if ev is not None:
        c = ev[ev.dataset == "nilai_vs_volume"].set_index("model")["coef"]
        m = out.dataset == "nilai_vs_volume"
        out.loc[m, "coef"] = out.loc[m, "model"].map(c)
    if cf is not None:
        for _, r in cf.iterrows():
            dengan = "dengan komoditas" in r["model"]
            m = (out.dataset == "nilai_vs_volume") & (out.model.str.contains("komoditas") == dengan)
            out.loc[m, "coef_lo"], out.loc[m, "coef_hi"] = r["ci95_lo"], r["ci95_hi"]
    return out


def sync_results(verbose=True):
    """Isi tabel hasil analisis dari berkas di output/reports (tabel ini diganti seluruhnya tiap kali)."""
    n = {}
    me = _model_eval_frame()
    if me is not None:
        n["model_eval"] = db.save_model_eval(me)
    ce = _read_csv("clustering_eval.csv")
    if ce is not None:
        ce = ce.rename(columns={"stabilitas_ARI": "stabilitas_ari", "dipilih": "terpilih"})
        n["cluster_eval"] = db.save_cluster_eval(ce)
    pr = _read_csv("proyeksi_kabupaten.csv")
    if pr is not None:
        n["proyeksi_kabupaten"] = db.save_proyeksi(pr)
    fcs = sorted(REPORT_DIR.glob("forecast_produksi_*.csv"), key=lambda p: int(re.findall(r"(\d{4})", p.stem)[-1]))
    if fcs:
        n["validasi_prakiraan"] = db.save_validasi_prakiraan(pd.read_csv(fcs[-1]))
    rs = _read_csv("produksi_residual_kab.csv")
    if rs is not None:
        n["residual_kabupaten"] = db.save_residual(rs)
    ql = _read_csv("data_quality_log.csv")
    if ql is not None:
        n["data_quality_log"] = db.save_quality_log(ql)
    params = {k: v for k, v in {"skenario": _read_json("scenario_params.json"), "clustering": _read_json("clustering_meta.json"),
                                "ringkasan_regresi": _read_json("regression_insight.json")}.items() if v is not None}
    if params:
        db.save_model_params(params)
    mv = _read_json("model_version_row.json")
    if mv:
        with db.get_engine().connect() as conn:
            last = conn.execute(db.text("SELECT data_hash, model_name FROM model_versions ORDER BY version_id DESC LIMIT 1")).fetchone()
        if last and last[0] == mv["data_hash"] and last[1] == mv["model_name"][:100]:
            print("✓ model_versions: data & model sama dengan catatan terakhir, tidak dicatat ulang.")
        else:
            db.save_model_version(mv["data_hash"], mv["model_name"][:100], mv["cv_r2"], mv.get("notes", ""), tugas="M1_nilai")
    return n


# ----------------------------------------------------------------------------- utama
def main(reset=False, prune=False):
    print("=" * 60); print("  SINKRONISASI HASIL PIPELINE -> POSTGRESQL"); print("=" * 60)
    print("\n[1/5] Init schema...")
    db.init_schema()
    if reset:
        print("\n[reset] Mengosongkan tabel utama...")
        db.reset_data()

    fp = PROCESSED_DIR / "fact_budidaya.parquet"
    print("\n[2/5] fact_budidaya + raw_budidaya...")
    if fp.exists():
        fact = pd.read_parquet(fp)
        db.upsert_fact(fact, prune=prune)
        db.upsert_raw(fact, batch_name="pipeline_latest")
    else:
        print("  ⚠️ fact_budidaya.parquet tidak ditemukan (jalankan pipeline dulu).")

    print("\n[3/5] features_model...")
    ftp = PROCESSED_DIR / "features_model.parquet"
    if ftp.exists():
        db.upsert_features(pd.read_parquet(ftp), prune=prune)
    else:
        print("  ⚠️ features_model.parquet tidak ditemukan.")

    print("\n[4/5] cluster_assignments (per kabupaten-tahun)...")
    cp = REPORT_DIR / "cluster_assignments_by_year.parquet"
    if cp.exists():
        db.upsert_cluster(pd.read_parquet(cp), prune=prune)
    else:
        print("  ⚠️ cluster_assignments_by_year.parquet tidak ditemukan.")

    print("\n[5/5] hasil analisis (model, klaster, proyeksi, validasi, kualitas data)...")
    try:
        n = sync_results()
    except Exception as e:
        print("  ⚠️ sebagian hasil analisis tidak tersimpan:", e)
        n = {}
    print("\n" + "=" * 60); print("  ✓ SELESAI"); print("=" * 60)
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reset", action="store_true", help="kosongkan fact/features/cluster sebelum mengisi")
    ap.add_argument("--prune", action="store_true", help="hapus baris DB yang tidak ada lagi di data baru")
    ap.add_argument("--run", action="store_true", help="jalankan pipeline dulu")
    a = ap.parse_args()
    if a.run:
        from .pipeline import run_all
        run_all()
    main(reset=a.reset, prune=a.prune)