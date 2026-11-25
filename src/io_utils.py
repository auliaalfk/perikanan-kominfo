import json
import math
import os
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _tmp(path: Path) -> Path:
    return path.with_name(path.name + f".{os.getpid()}.tmp")


def _atomic(path: Path, writer):
    """writer(tmp_path) menulis berkas; sukses -> ganti berkas tujuan; gagal -> berkas sementara dibuang, berkas lama utuh."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def save_parquet(df: pd.DataFrame, path: Path):
    return _atomic(path, lambda t: df.to_parquet(t, index=False))


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_csv(df: pd.DataFrame, path: Path):
    return _atomic(path, lambda t: df.to_csv(t, index=False))


def _json_safe(o):
    """Ubah tipe numpy/pandas/Path/set dan NaN/inf menjadi tipe JSON standar."""
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()
    if isinstance(o, pd.Series):
        return _json_safe(o.to_dict())
    if isinstance(o, pd.DataFrame):
        return _json_safe(o.to_dict(orient="records"))
    if isinstance(o, Path):
        return str(o)
    if o is pd.NaT or (not isinstance(o, (str, bytes)) and o is not None and pd.isna(o) is True):
        return None
    return o


def save_json(data: dict, path: Path):
    def _w(t):
        with open(t, "w", encoding="utf-8") as f:
            json.dump(_json_safe(data), f, indent=2, ensure_ascii=False, allow_nan=False)
    return _atomic(path, _w)


def save_xlsx(sheets: dict, path: Path):
    def _w(t):
        with pd.ExcelWriter(t, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
    return _atomic(path, _w)