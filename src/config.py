import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "output"
REPORT_DIR = OUTPUT_DIR / "reports"
MODEL_DIR = OUTPUT_DIR / "models"
FIG_DIR = OUTPUT_DIR / "figures"

for d in [RAW_DIR, PROCESSED_DIR, REPORT_DIR, MODEL_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PROVINSI_TARGET = "JAWA TIMUR"

# Link sumber data
SUMBER = {
    "produksi_jenis_budidaya": "https://opendata.jatimprov.go.id/dataset/jumlah-produksi-perikanan-budidaya-berdasarkan-jenis-budidaya",
    "rtp": "https://opendata.jatimprov.go.id/dataset/jumlah-rtp-rumah-tangga-perikanan--perusahaan-perikanan-pp-berdasarkan-jenis-budidaya",
    "luas_bersih": "https://opendata.jatimprov.go.id/dataset/luas-bersih-lahan-perikanan-budidaya-berdasarkan-jenis-budidaya",
    "luas_lahan_kkp": "https://portaldata.kkp.go.id/portals/data-statistik/lahan_budidaya/summary",
    "pembudidaya": "https://portaldata.kkp.go.id/portals/data-statistik/budidaya_ikan/summary",
    "ikan_hias": "https://portaldata.kkp.go.id/datainsight/produksi-ikan-hias",
    "pembenihan": "https://portaldata.kkp.go.id/datainsight/produksi-ikan-pembenihan",
    "pembesaran": "https://portaldata.kkp.go.id/datainsight/produksi-ikan-budidaya",
    "bps_kegiatan": "https://jatim.bps.go.id/id/statistics-table/3/WjFab0sxVjVkR3hOVUdjMWNIRnNkbWd3ZWl0bWR6MDkjMw==/produksi-perikanan-budidaya-menurut-kabupaten-kota-dan-jenis-kegiatan-di-provinsi-jawa-timur--2021.html",
    "bps_nilai": "https://jatim.bps.go.id/id/statistics-table/3/Wldsb1EyRktNMDl2VUVSalEzVllaM2xQWTFGTmR6MDkjMw%3D%3D/volume-produksi-dan-nilai-produksi-perikanan-budidaya-menurut-kabupaten-kota-dan-jenis-budidaya-di-provinsi-jawa-timur--2023.html",
}

KOMODITAS_SINONIM = {
    "GURAME; GURAMI": "GURAMI",
    "GURAME": "GURAMI",
    "MAS KOKI": "MAS",
    "UDANG VANAME": "UDANG",
    "LOBSTER PASIR": "LOBSTER AIR LAUT",
    "RUMPUT LAUT EUCHEUMA COTTONII": "RUMPUT LAUT",
    "IKAN LAINNYA": "LAINNYA",
    "BAWAL AIR TAWAR": "BAWAL",
    "LALAWAK": "TAWES",
}

JENIS_BUDIDAYA_STD = {
    "JARING APUNG": "JARING APUNG",
    "JARING APUNG LAUT": "JARING APUNG",
    "JARING APUNG TAWAR": "JARING APUNG",
    "KARAMBA": "KARAMBA",
    "KOLAM": "KOLAM",
    "KOLAM AIR TENANG": "KOLAM",
    "KOLAM AIR DERAS": "KOLAM",
    "LAUT": "LAUT",
    "LAUT LAINNYA": "LAUT",
    "MINA PADI": "MINA PADI",
    "MINAPADI SAWAH": "MINA PADI",
    "MINAPADI (SAWAH)": "MINA PADI",
    "TAMBAK": "TAMBAK",
    "TAMBAK INTENSIF": "TAMBAK",
    "TAMBAK SEMI INTENSIF": "TAMBAK",
    "TAMBAK SEMI INSENTIF": "TAMBAK",
    "TAMBAK SEDERHANA": "TAMBAK",
    "RUMPUT LAUT": "RUMPUT LAUT",
    "JARING TANCAP TAWAR": "JARING TANCAP",
}

# Komoditas (SETELAH KOMODITAS_SINONIM diterapkan) -> grup untuk kolom share_* di features_model.
# Komoditas yang belum terdaftar masuk grup 'lainnya' (dan dicatat di QC), jadi komoditas baru tidak merusak pipeline.
KOMODITAS_GRUP = {
    "RUMPUT LAUT": "rumput_laut",
    "UDANG": "udang", "UDANG LAINNYA": "udang", "UDANG WINDU": "udang",
    "BANDENG": "bandeng", "LELE": "lele", "NILA": "nila", "GURAMI": "gurami", "PATIN": "patin", "MAS": "mas",
}

# Penemuan file mentah OTOMATIS: nama file boleh berubah / rentang tahun bertambah; cukup memuat kata kunci.
# kunci: (kata yang WAJIB ada di nama file, kata yang TIDAK BOLEH ada). Dibandingkan setelah huruf dikecilkan & simbol dibuang.
FILE_RULES = {
    "produksi_jenis": (["jumlah", "produksi", "jenis budidaya"], ["kegiatan", "pembenihan", "pembesaran", "hias"]),
    "rtp":            (["rtp"], []),
    "luas":           (["luas bersih"], []),
    "luas_kkp":       (["luas", "lahan"], ["bersih"]),
    "pembudidaya":    (["pembudidaya"], []),
    "hias":           (["ikan hias"], []),
    "bps_kegiatan":   (["jenis kegiatan"], []),
    "pembenihan":     (["pembenihan"], ["jenis kegiatan", "ikan hias"]),
    "pembesaran":     (["pembesaran"], ["jenis kegiatan"]),
    "bps_nilai":      (["nilai produksi"], []),
}
FILE_WAJIB = ["produksi_jenis", "pembesaran", "pembudidaya"]    # tanpa ini panel tidak dapat dibangun -> pipeline berhenti dengan pesan jelas

# Parameter preprocessing dan pemodelan
CONFIG = {
    # --- kunci lama (jangan dihapus; dipakai modul lama sampai diganti) ---
    "coverage_threshold": 0.90,
    "metric_coverage_min": 0.50,
    "min_metrik_lengkap": 3,
    "min_samples_regression": 10,
    "min_kab_clustering": 10,
    "label_top_share_min": 0.20,
    "outlier_mad_k": 5.0,
    "outlier_iqr_k_fallback": 1.5,
    "suspect_scale_produksi_ton": 100_000,   # LEGACY: ambang mutlak; diganti audit relatif ("audit") di cleaning v3
    "suspect_scale_luas_ha": 100_000,        # LEGACY
    "suspect_scale_rtp": 50_000,             # LEGACY

    # --- v3: audit data (notebook 01) ---
    "audit": {
        "min_ref_years": 2,          # tahun pembanding minimum agar sebuah tahun bisa diverifikasi
        "min_kab": 5,                # kabupaten minimum yang punya pembanding
        "zero_share_incomplete": 0.8,
        "log_tol_ok": 0.35,          # |median log10 rasio| < ini = wajar (~2,2x)
        "scale_exponents": (3, 4, 6),  # salah satuan: x1000, m2<->ha, x1e6
        "scale_tol": 0.25, "scale_iqr_max": 0.6,
        "stale_threshold": 0.9, "stale_min_n": 8, "stale_base_max": 0.5,
        "spike_ratio": 50, "spike_global_ratio": 200, "spike_max_share": 0.10,
        "pembudidaya_spike": 3.0,
    },
    # --- v3: kelayakan fitur untuk model (notebook 02) ---
    "fitur": {"cover_cluster": 0.8, "cover_panel": 0.5, "nilai_fallback_kkp": False},   # cover_panel 0.5: luas valid 3 dari 5 tahun (=60%) mudah jatuh di bawah 0,6 karena satu kab-tahun
    # --- v3: clustering (notebook 03) ---
    "cluster": {"min_size_frac": 0.13, "min_size_abs": 3, "stability_ari_min": 0.70, "sil_tolerance": 0.02,
                "special_laut_share": 0.9, "special_harga_z": -3.5, "kmax": 8, "label_harga_tinggi_x": 1.35},
    # --- v3: regresi (notebook 04) ---
    "regresi": {"set_tolerance": 0.01, "ml_beat_baseline": 0.10, "min_rows_m2": 30, "min_kab_m2": 10, "min_rows_m3": 30, "horizon": 3},
}


def _load_dotenv(path):
    """Pembaca .env sederhana (KEY=VALUE), tanpa dependensi. Variabel yang sudah ada di environment TIDAK ditimpa."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

# Bentuk dict TETAP sama (db.py membacanya apa adanya). Isi diambil dari environment / .env:
#   PG_HOST, PG_PORT, PG_DBNAME, PG_USER, PG_PASSWORD
DB_CONFIG = {
    "host": os.environ.get("PG_HOST", "localhost"),
    "port": int(os.environ.get("PG_PORT", "5432")),
    "dbname": os.environ.get("PG_DBNAME", "perikanan_jatim_v2"),
    "user": os.environ.get("PG_USER", "postgres"),
    "password": os.environ.get("PG_PASSWORD", "auliahaha"),
}