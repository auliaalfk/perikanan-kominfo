import os
import time
from datetime import datetime
from contextlib import contextmanager

from .config import CONFIG, REPORT_DIR
from .io_utils import save_json
from .cleaning import build_fact
from .features import build_features_model
from .clustering import run_clustering
from .regression import run_all_regression
from .versioning import record_run

LOCK_FILE = REPORT_DIR / ".pipeline.lock"
LOCK_STALE_S = 15 * 60
RUN_LOG = REPORT_DIR / "run_log.json"


@contextmanager
def _lock():
    """Satu pipeline pada satu waktu. Lock yang lebih tua dari LOCK_STALE_S dianggap sisa proses yang mati."""
    for percobaan in range(2):
        try:
            fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()} mulai={datetime.now().isoformat()}".encode())
            os.close(fd)
            break
        except FileExistsError:
            umur = time.time() - LOCK_FILE.stat().st_mtime
            if umur > LOCK_STALE_S and percobaan == 0:
                LOCK_FILE.unlink(missing_ok=True)
                continue
            raise RuntimeError(f"Pipeline lain sedang berjalan (lock berumur {umur / 60:.0f} menit). "
                               f"Tunggu selesai, atau jika yakin tidak ada proses berjalan hapus berkas: {LOCK_FILE}")
    try:
        yield
    finally:
        LOCK_FILE.unlink(missing_ok=True)


class _Jejak:
    """Mencatat langkah -> run_log.json."""
    def __init__(self):
        self.mulai = datetime.now(); self.langkah = []; self.status = "berjalan"

    def jalankan(self, nama, fn, *args, ringkas=None):
        t0 = time.time()
        try:
            hasil = fn(*args)
        except Exception as e:
            self.langkah.append({"langkah": nama, "status": "gagal", "detik": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {e}"})
            self.status = f"gagal di: {nama}"
            raise RuntimeError(f'Langkah "{nama}" gagal: {type(e).__name__}: {e}') from e
        self.langkah.append({"langkah": nama, "status": "ok", "detik": round(time.time() - t0, 1), **(ringkas(hasil) if ringkas else {})})
        return hasil

    def simpan(self):
        try:
            save_json({"mulai": self.mulai.isoformat(), "selesai": datetime.now().isoformat(), "status": self.status, "langkah": self.langkah}, RUN_LOG)
        except Exception as e:   # catatan gagal tidak boleh menggagalkan pipeline
            print(f"  (peringatan: run_log.json tidak tersimpan: {e})")


def _shape(df):
    return {"baris": int(df.shape[0]), "kolom": int(df.shape[1])}


def run_all(config=None):
    cfg = CONFIG if config is None else config
    with _lock():
        jejak = _Jejak()
        try:
            print('=== STEP 1: Cleaning & Integrasi ===')
            fact = jejak.jalankan("cleaning", build_fact, cfg, ringkas=_shape)
            print(f'  fact shape: {fact.shape}')

            print('\n=== STEP 2: Buat Fitur ===')
            feat = jejak.jalankan("fitur", build_features_model, fact, ringkas=_shape)
            print(f'  features shape: {feat.shape}')

            print('\n=== STEP 3: Clustering ===')
            clust = jejak.jalankan("clustering", run_clustering, feat, ringkas=lambda c: {"k_opt": c.get("k_opt"), "silhouette": c.get("silhouette")})
            sil = clust.get("silhouette")
            print(f'  k optimal: {clust.get("k_opt")}, silhouette: ' + (f'{sil:.4f}' if sil is not None else 'n/a (clustering dilewati)'))
            df_c = clust.get('df')
            for kl, lbl in (clust.get('label_map') or {}).items():
                n = int((df_c['klaster'] == kl).sum()) if df_c is not None and 'klaster' in df_c else 0
                print(f'  K{kl} ({lbl}): {n} baris')

            print('\n=== STEP 4: Regresi ===')
            reg = jejak.jalankan("regresi", run_all_regression, feat, fact)
            for key in ['utama', 'struktural']:
                r = (reg or {}).get(key)
                if r:
                    br = r.get('best_row') or {}
                    cv = br.get('cv_r2_mean')
                    print(f'  {r.get("name", key)}: CV R² = ' + (f'{cv:.4f}' if cv is not None else 'n/a'))

            jejak.status = "sukses"
            hasil = {'fact': fact, 'features': feat, 'clustering': clust, 'regression': reg, 'run_log': jejak.langkah}
            try:
                record_run(hasil)                      # ringkasan versi data & model -> output/models/run_<waktu>.json
            except Exception as e:
                print(f"  (peringatan: record_run gagal: {e})")
            print('\n=== SELESAI ===')
            return hasil
        finally:
            jejak.simpan()


if __name__ == '__main__':      # jalankan dari folder proyek:  python -m src.pipeline
    run_all()