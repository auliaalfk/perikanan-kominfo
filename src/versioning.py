import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MODEL_DIR
from .io_utils import save_json, save_parquet

FACT_KEY = ["kabupaten", "tahun", "metrik", "sumber", "jenis_budidaya", "kegiatan", "komoditas"]
_NA = "∅"


def hash_df(df) -> str:
    """Sidik jari isi DataFrame (10 hex). Sama untuk isi yang sama, tidak peduli urutan kolom/indeks."""
    d = df.reindex(sorted(df.columns), axis=1)
    h = pd.util.hash_pandas_object(d, index=False).values
    return hashlib.md5(h.tobytes() + ",".join(map(str, d.columns)).encode()).hexdigest()[:10]


def save_run(meta: dict, prefix: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return save_json(meta, MODEL_DIR / f"{prefix}_{ts}.json")


def list_runs(prefix: str):
    return sorted(MODEL_DIR.glob(f"{prefix}_*.json"))


def _keys(df, key_cols):
    """Himpunan kunci (NaN-aman: NaN dianggap sama dengan NaN)."""
    k = df[key_cols].astype(object).where(df[key_cols].notna(), _NA)
    return pd.MultiIndex.from_frame(k)


def detect_changes(old_df, new_df, key_cols=("kabupaten", "tahun"), value_cols=None):
    key_cols = list(key_cols)
    ok, nk = set(_keys(old_df, key_cols)), set(_keys(new_df, key_cols))
    added, updated, removed = nk - ok, nk & ok, ok - nk
    vc = value_cols or [c for c in new_df.columns if c in old_df.columns and c not in key_cols and not str(c).startswith(("updated", "source"))]
    changed = set()
    if vc and updated:
        o = old_df.assign(_k=_keys(old_df, key_cols)).drop_duplicates("_k", keep="last").set_index("_k")[vc]
        n = new_df.assign(_k=_keys(new_df, key_cols)).drop_duplicates("_k", keep="last").set_index("_k")[vc]
        idx = [i for i in n.index if i in o.index]
        a, b = o.loc[idx], n.loc[idx]
        num = [c for c in vc if pd.api.types.is_numeric_dtype(b[c]) and pd.api.types.is_numeric_dtype(a[c])]
        diff = pd.Series(False, index=idx)
        for c in vc:
            if c in num:
                diff |= ~np.isclose(a[c].astype(float), b[c].astype(float), rtol=1e-9, atol=1e-12, equal_nan=True)
            else:
                diff |= ~((a[c] == b[c]) | (a[c].isna() & b[c].isna()))
        changed = set(diff.index[diff])
    return {"added": added, "updated": updated, "removed": removed, "changed": changed,
            "n_added": len(added), "n_updated": len(updated), "n_removed": len(removed), "n_changed": len(changed)}


def upsert_data(existing_path: Path, new_df: pd.DataFrame, key_cols=("kabupaten", "tahun")):
    key_cols = list(key_cols)
    existing_path = Path(existing_path)
    if existing_path.exists():
        old_df = pd.read_parquet(existing_path)
        keep = ~_keys(old_df, key_cols).isin(_keys(new_df, key_cols))
        combined = pd.concat([old_df[keep], new_df], ignore_index=True)
    else:
        combined = new_df.copy()
    save_parquet(combined, existing_path)
    return combined


def upsert_fact(existing_path: Path, new_df: pd.DataFrame, key_cols=tuple(FACT_KEY)):
    key_cols = [c for c in key_cols if c in new_df.columns]
    existing_path = Path(existing_path)
    if existing_path.exists():
        n_before = len(pd.read_parquet(existing_path))
        combined = upsert_data(existing_path, new_df, key_cols)
        n_after = len(combined)
        n_new = len(new_df)
        n_upd = n_before + n_new - n_after
        print(f"UPSERT: {n_before} -> {n_after} baris (update {n_upd}, insert {n_new - n_upd})")
    else:
        combined = upsert_data(existing_path, new_df, key_cols)
        print(f"INSERT: {len(combined)} baris (file baru)")
    return combined


def upsert_features(existing_path: Path, new_df: pd.DataFrame, key_cols=("kabupaten", "tahun")):
    return upsert_fact(existing_path, new_df, key_cols=key_cols)


def save_snapshot(df: pd.DataFrame, version: int, prefix: str = "fact", directory: Path = None):
    version_dir = Path(directory) if directory else MODEL_DIR / "versions"
    return save_parquet(df, version_dir / f"{prefix}_v{version}.parquet")


def record_run(res: dict, extra: dict = None):
    """Catat ringkasan satu kali jalan pipeline (hash data, ukuran, tahun, k klaster, model terbaik) ke output/models/run_<waktu>.json."""
    fact, feat, clu, reg = res.get("fact"), res.get("features"), res.get("clustering") or {}, res.get("regression") or {}
    utama = (reg.get("utama") or {}).get("best_row") or {}
    meta = {"waktu": datetime.now().isoformat(),
            "fact": {"baris": int(len(fact)), "hash": hash_df(fact)} if fact is not None else None,
            "features": {"baris": int(len(feat)), "hash": hash_df(feat[["kabupaten", "tahun", "produksi_ton", "rtp", "pembudidaya", "luas_lahan_ha", "total_pembesaran"]]),
                         "tahun": sorted(int(y) for y in feat.tahun.unique())} if feat is not None else None,
            "clustering": {"k": clu.get("k_opt"), "silhouette": clu.get("silhouette")},
            "regresi_utama": {"model": utama.get("model"), "cv_r2_mean": utama.get("cv_r2_mean"), "coef": utama.get("coef")}, **(extra or {})}
    return save_run(meta, "run")