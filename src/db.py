import json
from functools import lru_cache

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from .config import DB_CONFIG


# ============================================================ KONEKSI
@lru_cache(maxsize=1)
def get_engine():
    miss = [k for k in ("host", "dbname", "user") if not DB_CONFIG.get(k)]
    if miss:
        raise RuntimeError(f"DB_CONFIG belum lengkap ({', '.join(miss)}). Isi lewat environment/.env: PG_HOST, PG_PORT, PG_DBNAME, PG_USER, PG_PASSWORD.")
    url = URL.create(
        "postgresql+psycopg2",
        username=DB_CONFIG["user"], password=DB_CONFIG["password"],
        host=DB_CONFIG["host"], port=DB_CONFIG["port"], database=DB_CONFIG["dbname"],
    )
    return create_engine(url, pool_pre_ping=True)


def reset_engine():
    """Buang engine ter-cache (mis. setelah DB_CONFIG diubah)."""
    get_engine.cache_clear()


# ============================================================ SKEMA
FEATURE_COLS = [
    "produksi_ton", "nilai_rp_juta", "rtp", "pembudidaya", "luas_lahan_ha", "total_pembesaran",
    "share_rumput_laut", "share_udang", "share_bandeng", "share_lele", "share_nila", "share_gurami",
    "share_patin", "share_mas",
    "harga_rp_kg", "shannon", "n_komoditas", "top_share", "h_kolam", "h_tambak", "h_laut",
    "flag_lonjakan_pembudidaya",
]
CLUSTER_COLS = [
    "klaster", "label_klaster", "produksi_ton", "rtp", "pembudidaya", "luas_lahan_ha", "total_pembesaran",
    "harga_rp_kg", "shannon", "h_kolam", "h_tambak", "segmen_khusus",
]
FACT_COLS = ["kabupaten_id", "tahun", "jenis_budidaya", "metrik", "nilai", "satuan", "sumber",
             "kegiatan", "komoditas", "is_suspect_scale", "is_missing_year", "is_stale"]
FACT_KEY = ["kabupaten_id", "tahun", "metrik", "sumber", "jenis_budidaya", "kegiatan", "komoditas"]

# kolom tambahan yang ditambahkan ke tabel lama (ALTER ... IF NOT EXISTS), aman dijalankan berulang
_ADD_COLUMNS = {
    "features_model": [("share_mas", "FLOAT"), ("harga_rp_kg", "FLOAT"), ("shannon", "FLOAT"),
                       ("n_komoditas", "FLOAT"), ("top_share", "FLOAT"), ("h_kolam", "FLOAT"),
                       ("h_tambak", "FLOAT"), ("h_laut", "FLOAT"), ("flag_lonjakan_pembudidaya", "BOOLEAN")],
    "cluster_assignments": [("harga_rp_kg", "FLOAT"), ("shannon", "FLOAT"), ("h_kolam", "FLOAT"),
                            ("h_tambak", "FLOAT"), ("segmen_khusus", "BOOLEAN")],
    "model_versions": [("tugas", "VARCHAR(60)"), ("med_ape", "FLOAT")],
}

# tabel hasil analisis: diisi ulang seluruhnya tiap pembaruan (replace_table)
_REPORT_DDL = [
    """CREATE TABLE IF NOT EXISTS model_eval (
        id SERIAL PRIMARY KEY,
        dataset VARCHAR(60) NOT NULL,          -- M1_nilai | M2_produksi_struktural | M3_forecast_1th
        model VARCHAR(120) NOT NULL,
        cv_r2_mean FLOAT, r2_test FLOAT,
        coef FLOAT, coef_lo FLOAT, coef_hi FLOAT,   -- elastisitas (M1) + selang kepercayaan 95%
        rmse_log FLOAT, mae_asli FLOAT, med_ape FLOAT, med_ape_tanpa_outlier FLOAT,
        cv_r2_sd FLOAT, med_ape_sd FLOAT,
        terpilih BOOLEAN DEFAULT FALSE,
        updated_at TIMESTAMP DEFAULT NOW()
    );""",
    """CREATE TABLE IF NOT EXISTS cluster_eval (
        id SERIAL PRIMARY KEY,
        k INTEGER NOT NULL, silhouette FLOAT, davies_bouldin FLOAT, calinski FLOAT,
        ukuran_min INTEGER, stabilitas_ari FLOAT, lolos_syarat BOOLEAN,
        terpilih BOOLEAN DEFAULT FALSE,
        updated_at TIMESTAMP DEFAULT NOW()
    );""",
    """CREATE TABLE IF NOT EXISTS proyeksi_kabupaten (
        id SERIAL PRIMARY KEY,
        kabupaten_id INTEGER REFERENCES dim_kabupaten(kabupaten_id),
        tahun INTEGER NOT NULL, skenario VARCHAR(20) NOT NULL,
        produksi_ton FLOAT, batas_bawah_80 FLOAT, batas_atas_80 FLOAT,
        harga_rp_kg_dasar FLOAT, nilai_juta_harga_tetap FLOAT, nilai_juta_harga_tren FLOAT,
        updated_at TIMESTAMP DEFAULT NOW(),
        UNIQUE (kabupaten_id, tahun, skenario)
    );""",
    """CREATE TABLE IF NOT EXISTS validasi_prakiraan (
        kabupaten_id INTEGER NOT NULL REFERENCES dim_kabupaten(kabupaten_id),
        tahun_dasar INTEGER, tahun_target INTEGER,
        produksi_dasar_ton FLOAT, prakiraan_ton FLOAT, batas_bawah_80 FLOAT, batas_atas_80 FLOAT,
        aktual_ton FLOAT,
        updated_at TIMESTAMP DEFAULT NOW()
    );""",
    """CREATE TABLE IF NOT EXISTS residual_kabupaten (
        kabupaten_id INTEGER PRIMARY KEY REFERENCES dim_kabupaten(kabupaten_id),
        produksi_ton FLOAT, resid_log FLOAT, produksi_ekspektasi_ton FLOAT,
        updated_at TIMESTAMP DEFAULT NOW()
    );""",
    """CREATE TABLE IF NOT EXISTS model_params (
        key VARCHAR(60) PRIMARY KEY,
        value JSONB,
        updated_at TIMESTAMP DEFAULT NOW()
    );""",
    """CREATE TABLE IF NOT EXISTS data_quality_log (
        id SERIAL PRIMARY KEY,
        sumber TEXT, masalah TEXT, bukti TEXT, tindakan TEXT,
        updated_at TIMESTAMP DEFAULT NOW()
    );""",
]

_RENAME_SQL = """
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='model_eval' AND column_name='med_ape_tanpa_sumenep') THEN
    ALTER TABLE model_eval RENAME COLUMN med_ape_tanpa_sumenep TO med_ape_tanpa_outlier;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='proyeksi_kabupaten' AND column_name='harga_rp_kg_2024') THEN
    ALTER TABLE proyeksi_kabupaten RENAME COLUMN harga_rp_kg_2024 TO harga_rp_kg_dasar;
  END IF;
  IF to_regclass('validasi_2025') IS NOT NULL AND to_regclass('validasi_prakiraan') IS NULL THEN
    ALTER TABLE validasi_2025 RENAME TO validasi_prakiraan;
    ALTER TABLE validasi_prakiraan RENAME COLUMN produksi_2024_ton TO produksi_dasar_ton;
    ALTER TABLE validasi_prakiraan RENAME COLUMN prakiraan_2025_ton TO prakiraan_ton;
    ALTER TABLE validasi_prakiraan RENAME COLUMN aktual_2025 TO aktual_ton;
    ALTER TABLE validasi_prakiraan ADD COLUMN IF NOT EXISTS tahun_dasar INTEGER;
    ALTER TABLE validasi_prakiraan ADD COLUMN IF NOT EXISTS tahun_target INTEGER;
  END IF;
END $$;
"""

_UQ_FACT = ("CREATE UNIQUE INDEX IF NOT EXISTS uq_fact_budidaya ON fact_budidaya "
            "(kabupaten_id, tahun, metrik, COALESCE(sumber,''), COALESCE(jenis_budidaya,''), "
            "COALESCE(kegiatan,''), COALESCE(komoditas,''))")
_CONFLICT_FACT = ("kabupaten_id, tahun, metrik, COALESCE(sumber,''), COALESCE(jenis_budidaya,''), "
                  "COALESCE(kegiatan,''), COALESCE(komoditas,'')")


def init_schema():
    """Buat tabel bila belum ada, dan migrasikan tabel lama (aman dijalankan berulang)."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""CREATE TABLE IF NOT EXISTS dim_kabupaten (
            kabupaten_id SERIAL PRIMARY KEY, nama_kabupaten VARCHAR(100) UNIQUE NOT NULL);"""))

        conn.execute(text("""CREATE TABLE IF NOT EXISTS raw_budidaya (
            id SERIAL PRIMARY KEY, upload_batch VARCHAR(50), kabupaten VARCHAR(100), tahun INTEGER,
            jenis_budidaya VARCHAR(50), metrik VARCHAR(50), nilai FLOAT, satuan VARCHAR(20),
            sumber VARCHAR(50), uploaded_at TIMESTAMP DEFAULT NOW());"""))

        # fact_budidaya: TANPA UNIQUE inline (NULL dianggap berbeda -> duplikat); pakai indeks COALESCE
        conn.execute(text("""CREATE TABLE IF NOT EXISTS fact_budidaya (
            id SERIAL PRIMARY KEY,
            kabupaten_id INTEGER REFERENCES dim_kabupaten(kabupaten_id),
            tahun INTEGER NOT NULL, jenis_budidaya VARCHAR(50), metrik VARCHAR(50) NOT NULL,
            nilai FLOAT, satuan VARCHAR(20), sumber VARCHAR(50), kegiatan VARCHAR(50), komoditas VARCHAR(50),
            is_suspect_scale BOOLEAN DEFAULT FALSE, is_missing_year BOOLEAN DEFAULT FALSE,
            is_stale BOOLEAN DEFAULT FALSE, updated_at TIMESTAMP DEFAULT NOW());"""))

        conn.execute(text("""CREATE TABLE IF NOT EXISTS features_model (
            id SERIAL PRIMARY KEY,
            kabupaten_id INTEGER REFERENCES dim_kabupaten(kabupaten_id), tahun INTEGER NOT NULL,
            produksi_ton FLOAT, nilai_rp_juta FLOAT, rtp FLOAT, pembudidaya FLOAT, luas_lahan_ha FLOAT,
            total_pembesaran FLOAT, share_rumput_laut FLOAT, share_udang FLOAT, share_bandeng FLOAT,
            share_lele FLOAT, share_nila FLOAT, share_gurami FLOAT, share_patin FLOAT,
            updated_at TIMESTAMP DEFAULT NOW(), UNIQUE (kabupaten_id, tahun));"""))

        conn.execute(text("""CREATE TABLE IF NOT EXISTS cluster_assignments (
            id SERIAL PRIMARY KEY,
            kabupaten_id INTEGER REFERENCES dim_kabupaten(kabupaten_id), tahun INTEGER NOT NULL,
            klaster INTEGER, label_klaster VARCHAR(100), produksi_ton FLOAT, rtp FLOAT, pembudidaya FLOAT,
            luas_lahan_ha FLOAT, total_pembesaran FLOAT,
            updated_at TIMESTAMP DEFAULT NOW(), UNIQUE (kabupaten_id, tahun));"""))

        conn.execute(text("""CREATE TABLE IF NOT EXISTS model_versions (
            version_id SERIAL PRIMARY KEY, timestamp TIMESTAMP DEFAULT NOW(), data_hash VARCHAR(20),
            model_name VARCHAR(100), cv_r2 FLOAT, notes TEXT);"""))

        # ---- migrasi: kolom tambahan
        for tbl, cols in _ADD_COLUMNS.items():
            for c, typ in cols:
                conn.execute(text(f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS {c} {typ}"))

        # ---- migrasi fact_budidaya: buang UNIQUE lama (NULL-distinct), bersihkan duplikat, pasang indeks baru
        if conn.execute(text("SELECT to_regclass('uq_fact_budidaya')")).scalar() is None:
            conn.execute(text("""
                DO $$
                DECLARE r record;
                BEGIN
                  FOR r IN SELECT conname FROM pg_constraint
                           WHERE conrelid = CAST('fact_budidaya' AS regclass) AND contype = 'u'
                  LOOP
                    EXECUTE 'ALTER TABLE fact_budidaya DROP CONSTRAINT ' || quote_ident(r.conname);
                  END LOOP;
                END $$;"""))
            n = conn.execute(text("""
                DELETE FROM fact_budidaya a USING fact_budidaya b
                WHERE a.id < b.id AND a.kabupaten_id = b.kabupaten_id AND a.tahun = b.tahun
                  AND a.metrik = b.metrik
                  AND COALESCE(a.sumber,'') = COALESCE(b.sumber,'')
                  AND COALESCE(a.jenis_budidaya,'') = COALESCE(b.jenis_budidaya,'')
                  AND COALESCE(a.kegiatan,'') = COALESCE(b.kegiatan,'')
                  AND COALESCE(a.komoditas,'') = COALESCE(b.komoditas,'')""")).rowcount
            if n:
                print(f"  migrasi: {n} baris duplikat lama di fact_budidaya dibersihkan.")
            conn.execute(text(_UQ_FACT))

        conn.execute(text(_RENAME_SQL))
        for ddl in _REPORT_DDL:
            conn.execute(text(ddl))
    print("✓ Schema initialized.")


# ============================================================ HELPER
def _py(v):
    """Nilai numpy/pandas -> tipe Python murni; NaN/NaT/inf -> None (psycopg2 tidak mengerti NaN dengan benar)."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if (np.isnan(v) or np.isinf(v)) else float(v)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _rows(df, cols):
    return [tuple(_py(v) for v in r) for r in df[cols].itertuples(index=False, name=None)]


def _bulk(conn, sql, rows, fetch=False, page_size=2000):
    """execute_values di dalam transaksi SQLAlchemy yang sama."""
    cur = conn.connection.cursor()
    try:
        return execute_values(cur, sql, rows, page_size=page_size, fetch=fetch)
    finally:
        cur.close()


def _ensure_cols(df, cols):
    d = df.copy()
    for c in cols:
        if c not in d.columns:
            d[c] = None
    return d


def _with_kab_id(df):
    df = df.drop(columns=["kabupaten_id"], errors="ignore")      # kolom kosong dari _ensure_cols akan bentrok saat merge
    upsert_kabupaten(df)
    d = df.merge(get_kabupaten_map(), left_on="kabupaten", right_on="nama_kabupaten", how="left")
    miss = d["kabupaten_id"].isna().sum()
    if miss:
        print(f"  ! {miss} baris dibuang karena kabupaten tidak ada di dim_kabupaten")
    d = d.dropna(subset=["kabupaten_id"]).copy()
    d["kabupaten_id"] = d["kabupaten_id"].astype(int)
    return d


def _upsert(table, d, cols, conflict, update_cols, label, prune_where=None, prune_params=None):
    """INSERT ... ON CONFLICT DO UPDATE. prune_where (SQL) -> hapus baris tabel yang cocok tetapi TIDAK tersentuh upsert ini (updated_at < awal transaksi)."""
    rows = _rows(d, cols)
    if not rows:
        print(f"✓ {label}: 0 baris.")
        return 0
    sets = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
    sql = (f"INSERT INTO {table} ({', '.join(cols)}) VALUES %s "
           f"ON CONFLICT ({conflict}) DO UPDATE SET {sets}, updated_at = NOW() RETURNING (xmax = 0)")
    n_del = 0
    with get_engine().begin() as conn:
        t0 = conn.execute(text("SELECT NOW()")).scalar()          # NOW() konstan dalam satu transaksi
        res = _bulk(conn, sql, rows, fetch=True)
        if prune_where:
            n_del = conn.execute(text(f"DELETE FROM {table} WHERE updated_at < :t0 AND ({prune_where})"), {**(prune_params or {}), "t0": t0}).rowcount
    n_ins = sum(1 for r in res if r[0])
    print(f"✓ {label}: {n_ins} inserted, {len(res) - n_ins} updated" + (f", {n_del} dihapus (tidak ada lagi di data baru)." if prune_where else "."))
    return len(res)


def replace_table(table, df, cols, with_kab=False):
    """Kosongkan tabel lalu isi ulang dalam SATU transaksi (untuk tabel hasil analisis)."""
    d = _with_kab_id(df) if with_kab else df.copy()
    cols = [c for c in cols if c in d.columns or c == "kabupaten_id"]
    rows = _rows(d, cols)
    with get_engine().begin() as conn:
        conn.execute(text(f"DELETE FROM {table}"))
        if rows:
            _bulk(conn, f"INSERT INTO {table} ({', '.join(cols)}) VALUES %s", rows)
    print(f"✓ {table}: {len(rows)} baris (diganti seluruhnya).")
    return len(rows)


# ============================================================ DIMENSI & RAW
def upsert_kabupaten(df: pd.DataFrame):
    uniq = [(n,) for n in df["kabupaten"].dropna().unique()]
    if uniq:
        with get_engine().begin() as conn:
            _bulk(conn, "INSERT INTO dim_kabupaten (nama_kabupaten) VALUES %s ON CONFLICT (nama_kabupaten) DO NOTHING", uniq)
    print(f"✓ dim_kabupaten: {len(uniq)} kabupaten di-cek.")


def get_kabupaten_map() -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(text("SELECT kabupaten_id, nama_kabupaten FROM dim_kabupaten"), conn)


def upsert_raw(df: pd.DataFrame, batch_name: str = "manual_upload"):
    df = df.copy()
    df["upload_batch"] = batch_name
    cols = ["upload_batch", "kabupaten", "tahun", "jenis_budidaya", "metrik", "nilai", "satuan", "sumber"]
    df = df[[c for c in cols if c in df.columns]]
    with get_engine().begin() as conn:                                  # idempoten: batch yang sama diganti, bukan ditambah
        conn.execute(text("DELETE FROM raw_budidaya WHERE upload_batch = :b"), {"b": batch_name})
        df.to_sql("raw_budidaya", conn, if_exists="append", index=False, method="multi", chunksize=2000)
    print(f"✓ raw_budidaya: {len(df)} baris di-append (batch={batch_name}).")


# ============================================================ FAKTA / FITUR / KLASTER
def upsert_fact(df: pd.DataFrame, prune: bool = False) -> int:
    d = _ensure_cols(_with_kab_id(df), FACT_COLS)
    for c in ("is_suspect_scale", "is_missing_year", "is_stale"):
        d[c] = d[c].fillna(False).astype(bool)
    for c in ("jenis_budidaya", "kegiatan", "komoditas", "sumber"):          # '' -> NULL agar kunci konsisten
        d[c] = d[c].where(d[c].notna() & (d[c].astype(str).str.strip() != ""), None)
    d["tahun"] = d["tahun"].astype(int)
    kk = d[FACT_KEY].astype(object).where(d[FACT_KEY].notna(), "")
    dup = kk.duplicated(keep="last")
    if dup.any():
        print(f"  ! {int(dup.sum())} baris kembar di dalam batch dibuang (dipakai yang terakhir)")
        d = d[~dup.values]
    upd = [c for c in FACT_COLS if c not in FACT_KEY]
    src = [x for x in d["sumber"].dropna().unique().tolist()]
    return _upsert("fact_budidaya", d, FACT_COLS, _CONFLICT_FACT, upd, "fact_budidaya",
                   prune_where="sumber = ANY(:src)" if prune and src else None, prune_params={"src": src})


def upsert_features(df: pd.DataFrame, prune: bool = False) -> int:
    d = _ensure_cols(_with_kab_id(df), FEATURE_COLS)
    d["tahun"] = d["tahun"].astype(int)
    d = d.drop_duplicates(["kabupaten_id", "tahun"], keep="last")
    return _upsert("features_model", d, ["kabupaten_id", "tahun"] + FEATURE_COLS,
                   "kabupaten_id, tahun", FEATURE_COLS, "features_model", prune_where="TRUE" if prune else None)


def upsert_cluster(df: pd.DataFrame, prune: bool = False) -> int:
    d = _ensure_cols(_with_kab_id(df), CLUSTER_COLS)
    d["tahun"] = d["tahun"].astype(int)
    d["klaster"] = d["klaster"].astype(int)
    d = d.drop_duplicates(["kabupaten_id", "tahun"], keep="last")
    return _upsert("cluster_assignments", d, ["kabupaten_id", "tahun"] + CLUSTER_COLS,
                   "kabupaten_id, tahun", CLUSTER_COLS, "cluster_assignments", prune_where="TRUE" if prune else None)


def reset_data(tables=("fact_budidaya", "features_model", "cluster_assignments")):
    """Kosongkan tabel utama (dim_kabupaten dibiarkan). Dipakai bila DB pernah terisi data v1."""
    with get_engine().begin() as conn:
        for t in tables:
            conn.execute(text(f"TRUNCATE {t} RESTART IDENTITY"))
    print("✓ dikosongkan:", ", ".join(tables))


# ============================================================ HASIL ANALISIS
def save_model_eval(df):
    cols = ["dataset", "model", "cv_r2_mean", "r2_test", "coef", "coef_lo", "coef_hi", "rmse_log", "mae_asli",
            "med_ape", "med_ape_tanpa_outlier", "cv_r2_sd", "med_ape_sd", "terpilih"]
    return replace_table("model_eval", _ensure_cols(df, cols), cols)


def save_cluster_eval(df):
    cols = ["k", "silhouette", "davies_bouldin", "calinski", "ukuran_min", "stabilitas_ari", "lolos_syarat", "terpilih"]
    return replace_table("cluster_eval", _ensure_cols(df, cols), cols)


def save_proyeksi(df):
    df = df.rename(columns={c: "harga_rp_kg_dasar" for c in df.columns if str(c).startswith("harga_rp_kg_") and c != "harga_rp_kg_dasar"})   # harga_rp_kg_<tahun> -> harga_rp_kg_dasar
    cols = ["kabupaten_id", "tahun", "skenario", "produksi_ton", "batas_bawah_80", "batas_atas_80",
            "harga_rp_kg_dasar", "nilai_juta_harga_tetap", "nilai_juta_harga_tren"]
    return replace_table("proyeksi_kabupaten", _ensure_cols(df, cols), cols, with_kab=True)


def save_validasi_prakiraan(df, tahun_dasar=None, tahun_target=None):
    """df dari forecast_produksi_<tahun>.csv (kolom produksi_<dasar>_ton, prakiraan_<target>_ton, aktual_tahun_baru_opendata) atau sudah bernama generik."""
    d = df.copy()
    for c in list(d.columns):
        m1, m2 = re_year(r"^produksi_(\d{4})_ton$", c), re_year(r"^prakiraan_(\d{4})_ton$", c)
        if m1: tahun_dasar = tahun_dasar or m1; d = d.rename(columns={c: "produksi_dasar_ton"})
        if m2: tahun_target = tahun_target or m2; d = d.rename(columns={c: "prakiraan_ton"})
    d = d.rename(columns={"aktual_tahun_baru_opendata": "aktual_ton"})
    d["tahun_dasar"], d["tahun_target"] = tahun_dasar, tahun_target
    cols = ["kabupaten_id", "tahun_dasar", "tahun_target", "produksi_dasar_ton", "prakiraan_ton", "batas_bawah_80", "batas_atas_80", "aktual_ton"]
    return replace_table("validasi_prakiraan", _ensure_cols(d, cols), cols, with_kab=True)


def re_year(pat, s):
    import re
    m = re.match(pat, str(s))
    return int(m.group(1)) if m else None


def save_validasi_2025(df):
    """LEGACY: dipertahankan agar kode lama tidak patah. Gunakan save_validasi_prakiraan."""
    d = df.rename(columns={"produksi_2024_ton": "produksi_dasar_ton", "prakiraan_2025_ton": "prakiraan_ton", "aktual_2025": "aktual_ton"})
    return save_validasi_prakiraan(d, tahun_dasar=2024, tahun_target=2025)


def save_residual(df):
    cols = ["kabupaten_id", "produksi_ton", "resid_log", "produksi_ekspektasi_ton"]
    return replace_table("residual_kabupaten", _ensure_cols(df, cols), cols, with_kab=True)


def save_quality_log(df):
    cols = ["sumber", "masalah", "bukti", "tindakan"]
    return replace_table("data_quality_log", _ensure_cols(df, cols), cols)


def save_model_params(params: dict):
    """params: {kunci: objek-JSON}. Kunci yang sama ditimpa."""
    rows = [(k, json.dumps(v, ensure_ascii=False, default=float)) for k, v in params.items()]
    with get_engine().begin() as conn:
        _bulk(conn, "INSERT INTO model_params (key, value) VALUES %s ON CONFLICT (key) "
                    "DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()", rows)
    print(f"✓ model_params: {len(rows)} kunci disimpan.")


def save_model_version(data_hash: str, model_name: str, cv_r2: float, notes: str = "",
                       tugas: str = None, med_ape: float = None):
    with get_engine().begin() as conn:
        conn.execute(text("""INSERT INTO model_versions (data_hash, model_name, cv_r2, notes, tugas, med_ape)
                             VALUES (:h, :m, :r2, :n, :t, :ma)"""),
                     {"h": data_hash, "m": model_name, "r2": _py(cv_r2), "n": notes, "t": tugas, "ma": _py(med_ape)})
    print(f"✓ model_versions: {model_name} dicatat.")


# ============================================================ QUERY DASHBOARD
def _read(sql, params=None):
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def query_fact(tahun: int = None) -> pd.DataFrame:
    q = """SELECT f.tahun, k.nama_kabupaten AS kabupaten, f.jenis_budidaya, f.metrik, f.nilai, f.satuan,
                  f.sumber, f.kegiatan, f.komoditas, f.is_suspect_scale, f.is_missing_year, f.is_stale
           FROM fact_budidaya f JOIN dim_kabupaten k ON f.kabupaten_id = k.kabupaten_id"""
    return _read(q + (" WHERE f.tahun = :t" if tahun else ""), {"t": int(tahun)} if tahun else None)


def query_cluster(tahun: int = None) -> pd.DataFrame:
    q = f"""SELECT c.tahun, k.nama_kabupaten AS kabupaten, {', '.join('c.' + x for x in CLUSTER_COLS)}
            FROM cluster_assignments c JOIN dim_kabupaten k ON c.kabupaten_id = k.kabupaten_id"""
    return _read(q + (" WHERE c.tahun = :t" if tahun else ""), {"t": int(tahun)} if tahun else None)


def query_features(tahun: int = None) -> pd.DataFrame:
    q = f"""SELECT f.tahun, k.nama_kabupaten AS kabupaten, {', '.join('f.' + x for x in FEATURE_COLS)}
            FROM features_model f JOIN dim_kabupaten k ON f.kabupaten_id = k.kabupaten_id"""
    return _read(q + (" WHERE f.tahun = :t" if tahun else ""), {"t": int(tahun)} if tahun else None)


def query_model_eval() -> pd.DataFrame:
    return _read("SELECT * FROM model_eval ORDER BY id")


def query_cluster_eval() -> pd.DataFrame:
    return _read("SELECT * FROM cluster_eval ORDER BY k")


def query_proyeksi() -> pd.DataFrame:
    return _read("""SELECT k.nama_kabupaten AS kabupaten, p.tahun, p.skenario, p.produksi_ton, p.batas_bawah_80,
                           p.batas_atas_80, p.harga_rp_kg_dasar, p.nilai_juta_harga_tetap, p.nilai_juta_harga_tren
                    FROM proyeksi_kabupaten p JOIN dim_kabupaten k USING (kabupaten_id) ORDER BY p.tahun, k.nama_kabupaten""")


def query_validasi_prakiraan() -> pd.DataFrame:
    return _read("""SELECT k.nama_kabupaten AS kabupaten, v.tahun_dasar, v.tahun_target, v.produksi_dasar_ton, v.prakiraan_ton,
                           v.batas_bawah_80, v.batas_atas_80, v.aktual_ton
                    FROM validasi_prakiraan v JOIN dim_kabupaten k USING (kabupaten_id) ORDER BY k.nama_kabupaten""")


def query_validasi_2025() -> pd.DataFrame:
    """LEGACY (nama kolom lama). Gunakan query_validasi_prakiraan."""
    return query_validasi_prakiraan().rename(columns={"produksi_dasar_ton": "produksi_2024_ton", "prakiraan_ton": "prakiraan_2025_ton", "aktual_ton": "aktual_2025"})


def query_residual() -> pd.DataFrame:
    return _read("""SELECT k.nama_kabupaten AS kabupaten, r.produksi_ton, r.resid_log, r.produksi_ekspektasi_ton
                    FROM residual_kabupaten r JOIN dim_kabupaten k USING (kabupaten_id) ORDER BY r.resid_log""")


def query_model_params() -> dict:
    with get_engine().connect() as conn:
        return {k: v for k, v in conn.execute(text("SELECT key, value FROM model_params")).fetchall()}


def query_quality_log() -> pd.DataFrame:
    return _read("SELECT sumber, masalah, bukti, tindakan FROM data_quality_log ORDER BY id")