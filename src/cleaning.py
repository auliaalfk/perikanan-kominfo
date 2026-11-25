import hashlib
import json
import re
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd

from .config import (RAW_DIR, PROCESSED_DIR, REPORT_DIR, PROVINSI_TARGET, KOMODITAS_SINONIM,
                     JENIS_BUDIDAYA_STD, FILE_RULES, FILE_WAJIB)
from .io_utils import save_parquet, save_csv, save_json

FACT_COLS = ["tahun", "kabupaten", "jenis_budidaya", "kegiatan", "komoditas", "metrik", "nilai", "satuan",
             "sumber", "source_file", "is_missing_year", "is_stale", "is_suspect_scale"]
FLAGS = ["is_missing_year", "is_stale", "is_suspect_scale"]

_AUDIT_DEFAULT = {"min_ref_years": 2, "min_kab": 5, "zero_share_incomplete": 0.8, "log_tol_ok": 0.35, "scale_exponents": (3, 4, 6),
                  "scale_tol": 0.25, "scale_iqr_max": 0.6, "stale_threshold": 0.9, "stale_min_n": 8, "stale_base_max": 0.5,
                  "spike_ratio": 50, "spike_global_ratio": 200, "spike_max_share": 0.10, "pembudidaya_spike": 3.0,
                  "scale_action": "correct"}      # 'correct' = kalikan faktor & catat; 'flag' = biarkan nilai, tandai is_suspect_scale


# ----------------------------------------------------------------------------- helper dasar
def parse_number(x):
    """Angka dari excel/teks. '-' dll -> NaN; (*) dibuang; mendukung '1,234.5' dan '1.234,5'."""
    if x is None or (not isinstance(x, str) and pd.isna(x)):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = re.sub(r"\(\*+\)", "", str(x)).strip()
    if s in {"-", "–", "—", "", "nan", "NaN", "NA", "n/a", "..", "..."}:
        return np.nan
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+", s) else s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def norm_txt(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def norm_kab(s):
    if s is None or (not isinstance(s, str) and pd.isna(s)):
        return s
    s = re.sub(r"\s+", " ", str(s).upper().strip())
    s = re.sub(r"^(KABUPATEN|KAB\.?)\s+", "", s)
    s = re.sub(r"^(KOTA|KOTAMADYA)\s+", "KOTA ", s)
    return s.title()


def norm_komoditas(s):
    if s is None or (not isinstance(s, str) and pd.isna(s)):
        return np.nan
    s = str(s).upper().strip()
    return KOMODITAS_SINONIM.get(s, s)


def norm_budidaya(s):
    if s is None or (not isinstance(s, str) and pd.isna(s)):
        return np.nan
    s = str(s).upper().strip()
    return JENIS_BUDIDAYA_STD.get(s, s)


def pick(df, *pats, required=True):
    """Cari kolom lewat regex pada nama kolom ternormalisasi; pola awal = prioritas."""
    cols = {c: norm_txt(c) for c in df.columns if not str(c).startswith("_")}
    for pat in pats:
        for c, n in cols.items():
            if re.search(pat, n):
                return c
    if required:
        raise KeyError(f"kolom {pats} tidak ditemukan; kolom yang ada: {list(cols.values())}")


def _read_any(p):
    if p.suffix.lower() == ".csv":
        try:
            return pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(p, sep=None, engine="python", encoding="latin-1")
    return pd.read_excel(p)


def _newest_per_year(df, ycol="tahun"):
    """Tahun yang ada di beberapa file: pakai hanya file TERBARU (urutan _file = tertua -> terbaru)."""
    if df["_file"].nunique() < 2:
        return df
    claimed, keep = set(), []
    for f in list(dict.fromkeys(df["_file"]))[::-1]:
        sub = df[df["_file"] == f]
        ys = set(sub[ycol].dropna())
        keep.append(sub[~sub[ycol].isin(claimed)])
        claimed |= ys
    return pd.concat(keep).sort_index()


# ----------------------------------------------------------------------------- konteks satu kali jalan
class _Ctx:
    def __init__(self, config):
        self.A = {**_AUDIT_DEFAULT, **((config or {}).get("audit") or {})}
        self.qc_rows, self.log_rows, self.ya, self.files, self.bps_tot = [], [], [], {}, []

    def qc(self, level, sumber, pesan, tahun=None):
        self.qc_rows.append({"level": level, "sumber": sumber, "tahun": tahun, "pesan": pesan})
        print(f"  [{level}] {sumber}: {pesan}")

    def log(self, sumber, masalah, bukti, tindakan):
        self.log_rows.append({"sumber": sumber, "masalah": masalah, "bukti": bukti, "tindakan": tindakan})

    def find(self, key):
        must, never = FILE_RULES[key]
        hit = []
        if RAW_DIR.exists():
            for p in RAW_DIR.rglob("*"):
                if not p.is_file() or p.suffix.lower() not in (".xlsx", ".xls", ".csv") or p.name.startswith(("~$", ".")):
                    continue
                if any(part.startswith(".") for part in p.relative_to(RAW_DIR).parts):
                    continue
                t = norm_txt(p.stem)
                if all(norm_txt(m) in t for m in must) and not any(norm_txt(n) in t for n in never):
                    hit.append(p)
        hit = sorted(hit, key=lambda p: (p.stat().st_mtime, p.name))      # tertua -> terbaru
        self.files[key] = hit
        return hit

    def read(self, key):
        fs = self.find(key)
        if not fs:
            self.qc("ERROR" if key in FILE_WAJIB else "INFO", key, "file tidak ditemukan di data/raw" + (" (WAJIB)" if key in FILE_WAJIB else " (opsional, dilewati)"))
            return None
        parts = []
        for p in fs:
            d = _read_any(p)
            d.columns = [str(c).strip() for c in d.columns]
            d["_file"] = p.name
            parts.append(d)
        if len(fs) > 1:
            self.qc("INFO", key, f"{len(fs)} file digabung; tahun yang sama -> file terbaru menang: " + ", ".join(p.name for p in fs))
        return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------- detektor generik
def audit_years(W, A, lenient=False):
    """W: kabupaten x tahun. Status tiap tahun: ok | sebagian | skala | tidak_lengkap | tidak_konsisten | tak_terverifikasi."""
    W = W.copy(); W.columns = [int(c) for c in W.columns]; years = sorted(W.columns)
    res = {y: dict(tahun=y, status="ok", faktor=1.0, median_log10=np.nan, iqr_log10=np.nan, zero_share=np.nan,
                   n_kab_ada=int((W[y] > 0).sum())) for y in years}
    bad = set()
    for _ in range(4):
        new_bad = set()
        for y in years:
            r = res[y]; r.update(status="ok", faktor=1.0, median_log10=np.nan, iqr_log10=np.nan, zero_share=np.nan)
            oth = [z for z in years if z != y and z not in bad]
            if len(oth) < A["min_ref_years"]:
                r["status"] = "tak_terverifikasi"; continue
            ref = W[oth].where(W[oth] > 0).median(axis=1); has = ref.notna()
            if has.sum() < A["min_kab"]:
                r["status"] = "tak_terverifikasi"; continue
            v = W.loc[has, y].fillna(0); zs = float((v <= 0).mean()); r["zero_share"] = zs
            if zs >= A["zero_share_incomplete"]:
                r["status"] = "tidak_lengkap"; new_bad.add(y); continue
            lr = np.log10(v[v > 0] / ref[has][v > 0]); m = float(lr.median()); iq = float(lr.quantile(.75) - lr.quantile(.25)); k = int(round(m))
            r["median_log10"], r["iqr_log10"] = m, iq
            if abs(m) < A["log_tol_ok"]:
                r["status"] = "sebagian" if zs > 0.2 else "ok"
            elif abs(k) in A["scale_exponents"] and abs(m - k) < A["scale_tol"] and iq < A["scale_iqr_max"]:
                r["status"] = "skala"; r["faktor"] = 10.0 ** (-k); new_bad.add(y)
            elif lenient:
                r["status"] = "sebagian"          # data jarang (hanya kabupaten aktif): jangan terlalu galak
            else:
                r["status"] = "tidak_konsisten"; new_bad.add(y)
        if new_bad == bad:
            break
        bad = new_bad
    return pd.DataFrame(list(res.values())).set_index("tahun")


def stale_years(W, A):
    """Tahun y yang >= stale_threshold nilainya identik dengan tahun lebih awal, padahal pasangan tahun lain jarang identik."""
    W = W.copy(); W.columns = [int(c) for c in W.columns]; ys = sorted(W.columns)

    def same(a, b):
        m = (W[a] > 0) & (W[b] > 0)
        return float(np.isclose(W.loc[m, a], W.loc[m, b]).mean()) if m.sum() >= A["stale_min_n"] else np.nan
    adj = [same(ys[i - 1], ys[i]) for i in range(1, len(ys))]; out = {}
    for i, y in enumerate(ys):
        for z in ys[:i]:
            s = same(z, y)
            if pd.notna(s) and s >= A["stale_threshold"]:
                others = [a for a in adj if pd.notna(a) and a < A["stale_threshold"]]
                if others and np.median(others) < A["stale_base_max"]:
                    out[y] = z; break
    return out


def _spike_mask(df, keys, A, vcol="nilai"):
    """Baris yang melonjak > spike_ratio x median baris yang sama di tahun lain (atau > spike_global_ratio x median jenisnya di tahun itu).
       Hanya bila <= spike_max_share baris tahun itu; bila lebih banyak itu masalah TAHUN (ditangani audit_years)."""
    flag = pd.Series(False, index=df.index)
    for y in df["tahun"].dropna().unique():
        oth = df[(df["tahun"] != y) & (df[vcol] > 0)].groupby(keys)[vcol].median().rename("_ref")
        cur = df[(df["tahun"] == y) & (df[vcol] > 0)].join(oth, on=keys)
        gm = cur.groupby(keys[-1])[vcol].transform("median")
        bad = ((cur[vcol] > A["spike_ratio"] * cur["_ref"]) | (cur[vcol] > A["spike_global_ratio"] * gm)) & (cur[vcol] > 100)
        if len(cur) and bad.sum() / len(cur) <= A["spike_max_share"]:
            flag[bad.index[bad]] = True
    return flag


def _ensure_flags(df):
    for f in FLAGS:
        if f not in df:
            df[f] = False
    return df


def audit_frame(ctx, df, name, main="nilai", also=(), stale=False, lenient=False):
    """Audit tahun pada total kab x tahun kolom `main`; koreksi/NaN-kan baris (dan kolom `also`) sesuai status. Mengembalikan df baru."""
    d = _ensure_flags(df.copy())
    W = d.groupby(["kabupaten", "tahun"])[main].sum(min_count=1).unstack("tahun")
    if W.shape[1] == 0 or W.dropna(how="all").empty:
        return d
    aud = audit_years(W, ctx.A, lenient); st = stale_years(W, ctx.A) if stale else {}
    cols = [main, *also]
    for y, r in aud.iterrows():
        m = d["tahun"] == y
        if y in st:
            aud.loc[y, "status"] = "salinan"; d.loc[m, cols] = np.nan; d.loc[m, ["is_missing_year", "is_stale"]] = True
            ctx.qc("WARN", name, f"tahun {y} salinan tahun {st[y]} (>= {ctx.A['stale_threshold']:.0%} kabupaten identik) -> NaN", y)
            ctx.log(name, f"tahun {y} salinan tahun {st[y]}", "nilai identik di hampir semua kabupaten, padahal pasangan tahun lain jarang identik", "tahun -> NaN (is_stale, is_missing_year)")
        elif r.status == "skala":
            if ctx.A["scale_action"] == "correct":
                d.loc[m, cols] = d.loc[m, cols] * r.faktor; tindakan = f"dikalikan {r.faktor:g}"
            else:
                d.loc[m, "is_suspect_scale"] = True; tindakan = "nilai dibiarkan, ditandai is_suspect_scale"
            ctx.qc("WARN", name, f"tahun {y} berskala x{1 / r.faktor:g} -> {tindakan}", y)
            ctx.log(name, f"tahun {y} berskala x{1 / r.faktor:g}", f"median log10 rasio={r.median_log10:.2f}, IQR={r.iqr_log10:.2f} (seragam di semua kabupaten)", tindakan)
        elif r.status in ("tidak_lengkap", "tidak_konsisten"):
            d.loc[m, cols] = np.nan; d.loc[m, "is_missing_year"] = True
            ctx.qc("WARN", name, f"tahun {y} {r.status} (zero_share={r.zero_share:.0%}, median_log10={r.median_log10:.2f}) -> NaN", y)
            ctx.log(name, f"tahun {y} {r.status}", f"zero_share={r.zero_share:.0%}; median log10 rasio ke tahun lain={r.median_log10:.2f}; IQR={r.iqr_log10:.2f}", "tahun dijadikan NaN (bukan 0)")
        elif r.status == "tak_terverifikasi":
            ctx.qc("INFO", name, f"tahun {y} tak terverifikasi (pembanding < {ctx.A['min_ref_years']} tahun)", y)
    a = aud.reset_index(); a.insert(0, "dataset", name); ctx.ya.append(a)
    return d


def _mark_spikes(ctx, d, name, keys, cols=("nilai",)):
    m = _spike_mask(d, keys, ctx.A)
    if m.any():
        d.loc[m, list(cols)] = np.nan; d.loc[m, "is_suspect_scale"] = True
        ctx.qc("WARN", name, f"{int(m.sum())} baris lonjakan > {ctx.A['spike_ratio']}x median barisnya sendiri -> NaN (tahun & baris lain tetap dipakai)")
        ctx.log(name, "lonjakan baris tunggal", f"{int(m.sum())} baris", "baris -> NaN + is_suspect_scale; tahun & kabupaten lain tetap dipakai")
    return d


# ----------------------------------------------------------------------------- opendata Jatim (produksi, RTP, luas)
def _opendata(ctx, key, metrik, sumber):
    d = ctx.read(key)
    if d is None:
        return pd.DataFrame()
    kc = pick(d, r"^kode kabupaten kota$", r"kode kab", required=False)
    nc = pick(d, r"^nama kabupaten kota$", r"nama kab", r"kabupaten|kota")
    vc = pick(d, r"^jumlah$", r"^nilai$", r"jumlah"); tc = pick(d, r"^tahun$"); gc = pick(d, r"^kategori$", r"jenis")
    sc = pick(d, r"^satuan$", required=False)
    o = pd.DataFrame({"tahun": pd.to_numeric(d[tc], errors="coerce"),
                      "kode": d[kc].astype(str).str.replace(r"\.0$", "", regex=True) if kc else "",
                      "nama_raw": d[nc].astype(str), "jenis_raw": d[gc].astype(str).str.upper().str.strip(),
                      "nilai": d[vc].map(parse_number), "satuan": d[sc] if sc else None, "_file": d["_file"]}).dropna(subset=["tahun"])
    o["tahun"] = o.tahun.astype(int); o = _newest_per_year(o)
    o["agregat"] = o.kode.isin(["0", "", "nan"]) | o.nama_raw.map(norm_txt).isin(["jawa timur", "provinsi jawa timur", "prov jawa timur"])
    o["blok"] = o.groupby(["tahun", "kode", "jenis_raw"]).cumcount()
    nonag = o[~o.agregat]
    cnt = nonag.groupby(["kode", "tahun", "jenis_raw"]).size(); share2 = (cnt >= 2).groupby(level="kode").mean()
    twin = set(share2[share2 >= 0.9].index) & set(nonag[nonag.nama_raw.str.upper().str.startswith("KOTA")].kode)
    extra = (o.blok >= np.where(o.kode.isin(twin), 2, 1)) & ~o.agregat
    if extra.any():
        ctx.qc("WARN", key, f"{int(extra.sum())} baris ganda (kunci tahun-kode-jenis sama) dibuang; baris pertama dipakai")
        o = o[~extra].copy()
    o["kabupaten"] = o.nama_raw.map(norm_kab)
    m = o.kode.isin(twin) & (o.blok == 0) & ~o.agregat
    o.loc[m, "kabupaten"] = o.loc[m, "nama_raw"].str.upper().str.replace(r"^KOTA\s+", "", regex=True).str.title()
    if twin:
        ctx.qc("INFO", key, f"{len(twin)} kode kembar Kab/Kota dipisahkan (kemunculan pertama = Kabupaten): " + ", ".join(sorted(o.loc[m, "kabupaten"].unique())))
    o = o[~o.agregat].copy()
    o["jenis_budidaya"] = o.jenis_raw.map(norm_budidaya)
    baru = sorted(set(o.jenis_budidaya.dropna()) - set(JENIS_BUDIDAYA_STD.values()))
    if baru:
        ctx.qc("INFO", key, f"jenis budidaya di luar daftar standar (dipakai apa adanya; tambahkan ke JENIS_BUDIDAYA_STD bila perlu): {baru}")
    g = o.groupby(["tahun", "kabupaten", "jenis_budidaya"], as_index=False).agg(nilai=("nilai", lambda s: s.sum(min_count=1)), satuan=("satuan", "first"), source_file=("_file", "last"))
    g["metrik"] = metrik; g["sumber"] = sumber
    return _ensure_flags(g)


def _luas_vs_rtp(ctx, luas, rtp):
    """Baris luas yang PERSIS sama dengan angka RTP (kabupaten-jenis-tahun sama) = angka rumah tangga, bukan hektar."""
    if luas.empty or rtp.empty:
        return luas
    k = ["tahun", "kabupaten", "jenis_budidaya"]
    m = luas[k + ["nilai"]].merge(rtp[k + ["nilai"]].rename(columns={"nilai": "rtp"}), on=k, how="left")
    eq = ((m.nilai > 0) & (m.nilai == m.rtp)).values
    if eq.any():
        per = dict(luas.loc[eq, "tahun"].value_counts().sort_index())
        luas = luas.copy(); luas.loc[eq, "nilai"] = np.nan; luas.loc[eq, "is_suspect_scale"] = True
        ctx.qc("WARN", "luas", f"{int(eq.sum())} baris luas persis sama dengan RTP -> NaN (per tahun: {per})")
        ctx.log("luas", "luas tercampur angka RTP", f"{int(eq.sum())} baris nilai == RTP pada kabupaten-jenis-tahun yang sama", "baris -> NaN + is_suspect_scale")
    return luas


def _luas_kkp(ctx):
    """File ke-10 (KKP Lahan Budidaya). Struktur belum pasti -> deteksi kolom otomatis; gagal = dilewati (bukan error)."""
    d = ctx.read("luas_kkp")
    if d is None:
        return pd.DataFrame()
    try:
        tc = pick(d, r"^tahun"); kc = pick(d, r"kabupaten|kota|wilayah"); pc = pick(d, r"provinsi", required=False)
        jc = pick(d, r"jenis|kategori|kelompok|media|komoditas", required=False)
        vcols = [c for c in d.columns if not str(c).startswith("_") and re.search(r"luas|lahan|hektar|\bha\b", norm_txt(c)) and c not in (kc, jc)]
        if not vcols:
            raise KeyError("kolom luas tidak terdeteksi")
        o = pd.DataFrame({"tahun": pd.to_numeric(d[tc], errors="coerce"), "kabupaten": d[kc].map(norm_kab), "source_file": d["_file"], "_file": d["_file"]})
        o["nilai"] = pd.concat([d[c].map(parse_number) for c in vcols], axis=1).sum(axis=1, min_count=1)
        o["jenis_budidaya"] = d[jc].map(norm_budidaya) if jc else np.nan
        if pc:
            o = o[d[pc].map(norm_txt).str.contains(norm_txt(PROVINSI_TARGET))]
        o = o.dropna(subset=["tahun", "kabupaten"]); o["tahun"] = o.tahun.astype(int); o = _newest_per_year(o)
        if jc:
            o = o[~o.jenis_budidaya.map(norm_txt).isin(["total", "jumlah", "jumlah total"])]
        ctx.qc("INFO", "luas_kkp", f"terbaca: {len(o)} baris, kolom luas={vcols}, kolom jenis={jc}, tahun={sorted(o.tahun.unique())}")
        o = o.groupby(["tahun", "kabupaten", "jenis_budidaya"], as_index=False, dropna=False).agg(nilai=("nilai", lambda s: s.sum(min_count=1)), source_file=("source_file", "last"))
        o["metrik"] = "luas_lahan_ha"; o["satuan"] = "ha"; o["sumber"] = "luas_kkp"
        return _ensure_flags(o)
    except Exception as e:
        ctx.qc("WARN", "luas_kkp", f"file ada tetapi tidak bisa dibaca otomatis ({type(e).__name__}: {e}); dilewati. Cek nama kolom.")
        return pd.DataFrame()


# ----------------------------------------------------------------------------- BPS
def _year_assign(ctx, key, d, p, kab_col):
    """Tahun per baris: kolom 'Tahun' bila ada; satu tahun di nama file; atau blok bertumpuk yang diakhiri baris 'Jawa Timur' (rentang di nama file)."""
    tc = pick(d, r"^tahun$", required=False)
    if tc:
        return pd.to_numeric(d[tc], errors="coerce")
    yrs = re.findall(r"(20\d\d)", p.stem)
    if len(set(yrs)) == 1:
        return pd.Series(int(yrs[0]), index=d.index)
    if len(yrs) >= 2:
        a, b = int(yrs[-2]), int(yrs[-1]); years = list(range(a, b + 1))
        tot = d[kab_col].map(norm_txt) == "jawa timur"
        blk = tot.shift(fill_value=False).cumsum()
        if blk.nunique() == len(years):
            return blk.map(dict(zip(sorted(blk.unique()), years)))
        ctx.qc("WARN", key, f"{p.name}: jumlah blok ({blk.nunique()}) != jumlah tahun di nama file ({len(years)}) -> file dilewati")
    else:
        ctx.qc("WARN", key, f"{p.name}: tahun tidak dapat ditentukan -> file dilewati")
    return pd.Series(np.nan, index=d.index)


def _bps_files(ctx, key):
    out = []
    for p in ctx.find(key):
        try:
            d = _read_any(p); d.columns = [str(c).strip() for c in d.columns]
            kc = pick(d, r"kabupaten|kota")
            d["tahun"] = _year_assign(ctx, key, d, p, kc)
            out.append((p, d, kc))
        except Exception as e:
            ctx.qc("WARN", key, f"{p.name} tidak terbaca ({type(e).__name__}: {e})")
    return out


EXCL = {"JAWA TIMUR", "KETERANGAN", "ANGKA TETAP", "-", "NAN"}


def _bps_kegiatan(ctx):
    long = []
    for p, d, kc in _bps_files(ctx, "bps_kegiatan"):
        d = d.dropna(subset=["tahun"]); d["tahun"] = d.tahun.astype(int)
        col_map = {}
        for c in d.columns:
            cu = str(c).upper()
            if "PEMBESARAN" in cu: col_map[c] = ("pembesaran", "produksi_ton")
            elif "PEMBENIHAN" in cu: col_map[c] = ("pembenihan", "volume_ribu_ekor")
            elif "IKAN HIAS" in cu or cu.strip() == "HIAS": col_map[c] = ("ikan_hias", "volume_ekor")
        if not col_map:
            ctx.qc("WARN", "bps_kegiatan", f"{p.name}: kolom pembesaran/pembenihan/ikan hias tidak ditemukan -> dilewati"); continue
        tot = d[d[kc].astype(str).str.upper().str.strip() == "JAWA TIMUR"]
        for c, (keg, met) in col_map.items():
            if keg == "pembesaran":
                for _, r in tot.iterrows():
                    ctx.bps_tot.append({"tahun": int(r.tahun), "bps_jatim_ton": parse_number(r[c]), "sumber": "bps_kegiatan"})
        d = d[~d[kc].astype(str).str.upper().str.strip().isin(EXCL)]
        for c, (keg, met) in col_map.items():
            t = pd.DataFrame({"tahun": d.tahun, "kabupaten": d[kc].map(norm_kab), "nilai": d[c].map(parse_number), "source_file": p.name})
            t["kegiatan"] = keg; t["metrik"] = met; t["satuan"] = "mixed"; long.append(t)
    if not long:
        return pd.DataFrame()
    out = pd.concat(long, ignore_index=True); out["sumber"] = "bps_kegiatan"
    out = out.groupby(["tahun", "kabupaten", "kegiatan", "metrik", "satuan", "sumber"], as_index=False).agg(nilai=("nilai", lambda s: s.sum(min_count=1)), source_file=("source_file", "last"))
    return _ensure_flags(out)


def _bps_nilai(ctx):
    long = []
    for p, d, kc in _bps_files(ctx, "bps_nilai"):
        d = d.dropna(subset=["tahun"]); d["tahun"] = d.tahun.astype(int)
        is_tot = d[kc].astype(str).str.upper().str.strip() == "JAWA TIMUR"
        for c in d.columns:
            cu = str(c).upper()
            if "JUMLAH" in cu and ("PRODUKSI" in cu or "VOLUME" in cu) and "NILAI" not in cu:
                for _, r in d[is_tot].iterrows():
                    ctx.bps_tot.append({"tahun": int(r.tahun), "bps_jatim_ton": parse_number(r[c]) / 1000, "sumber": "bps_nilai"})
        d = d[~d[kc].astype(str).str.upper().str.strip().isin(EXCL)]
        for c in d.columns:
            if c in {kc, "tahun", "_file"}:
                continue
            cu = str(c).upper()
            jb = next((v for k, v in JENIS_BUDIDAYA_STD.items() if k in cu), None)
            if jb is None or "JUMLAH" in cu:
                continue
            met = "nilai_rp" if "NILAI" in cu else ("produksi_kg" if ("PRODUKSI" in cu or "VOLUME" in cu) else None)
            if met is None:
                continue
            t = pd.DataFrame({"tahun": d.tahun, "kabupaten": d[kc].map(norm_kab), "nilai": d[c].map(parse_number), "source_file": p.name})
            t["jenis_budidaya"] = jb; t["metrik"] = met; t["satuan"] = "kg" if met == "produksi_kg" else "Rp"; long.append(t)
    if not long:
        return pd.DataFrame()
    out = pd.concat(long, ignore_index=True); out["sumber"] = "bps_nilai"
    out = out.groupby(["tahun", "kabupaten", "jenis_budidaya", "metrik", "satuan", "sumber"], as_index=False).agg(nilai=("nilai", lambda s: s.sum(min_count=1)), source_file=("source_file", "last"))
    return _ensure_flags(out)


# ----------------------------------------------------------------------------- pembudidaya (KKP)
def _pembudidaya(ctx):
    d = ctx.read("pembudidaya")
    if d is None:
        return pd.DataFrame()
    kc, tc, vc = pick(d, r"kabupaten|kota"), pick(d, r"^tahun"), pick(d, r"jumlah", r"pembudidaya")
    o = pd.DataFrame({"kabupaten": d[kc].map(norm_kab), "tahun": pd.to_numeric(d[tc], errors="coerce"), "nilai": d[vc].map(parse_number), "source_file": d["_file"], "_file": d["_file"]})
    o = o.dropna(subset=["tahun", "kabupaten"]); o["tahun"] = o.tahun.astype(int); o = _newest_per_year(o)
    o = o[~o.kabupaten.map(norm_txt).isin(["jawa timur"])].drop_duplicates(["kabupaten", "tahun"], keep="last")
    o["metrik"] = "pembudidaya"; o["satuan"] = "orang"; o["sumber"] = "pembudidaya"
    o = audit_frame(ctx, _ensure_flags(o), "pembudidaya", stale=True)
    med = o.groupby("kabupaten")["nilai"].transform("median"); n = o.groupby("kabupaten")["nilai"].transform("count")
    sp = ((o.nilai / med > ctx.A["pembudidaya_spike"]) | (o.nilai / med < 1 / ctx.A["pembudidaya_spike"])) & (n >= 3)
    if sp.any():
        o.loc[sp, "nilai"] = np.nan; o.loc[sp, "is_suspect_scale"] = True
        ctx.qc("WARN", "pembudidaya", f"{int(sp.sum())} titik > {ctx.A['pembudidaya_spike']:g}x / < 1/{ctx.A['pembudidaya_spike']:g}x median kabupatennya -> NaN + is_suspect_scale (features.py mengisi dengan median kabupaten)")
        ctx.log("pembudidaya", "lonjakan melawan median kabupaten sendiri", f"{int(sp.sum())} titik", "nilai -> NaN + is_suspect_scale (tidak ada angka karangan di fact)")
    return o.drop(columns=["_file"])


# ----------------------------------------------------------------------------- KKP (pembesaran / pembenihan / ikan hias)
def _kkp(ctx, key, kegiatan, vol_col, metrik_vol):
    d = ctx.read(key)
    if d is None:
        return pd.DataFrame()
    try:
        pv, kb, kl = pick(d, r"^provinsi$", r"provinsi"), pick(d, r"^kab kota$", r"kabupaten|kota"), pick(d, r"^kelompok ikan$", r"kelompok|komoditas")
        tc, nv = pick(d, r"^tahun$"), pick(d, r"^nilai rp juta$", r"^nilai")
        vv = pick(d, "^" + re.escape(norm_txt(vol_col)) + "$", r"^volume")
    except KeyError:
        cols = [c for c in d.columns if c != "_file"]
        if len(cols) not in (7, 8):
            ctx.qc("ERROR" if key in FILE_WAJIB else "WARN", key, f"nama kolom tidak dikenali & jumlah kolom {len(cols)} bukan 7/8 -> dilewati. Kolom: {cols}")
            return pd.DataFrame()
        names = ["tahun", "prov", "kab", "kelompok"] + (["jenis_ikan"] if len(cols) == 8 else []) + ["vol", "nilai", "harga"]
        d = d.rename(columns=dict(zip(cols, names))); pv, kb, kl, tc, nv, vv = "prov", "kab", "kelompok", "tahun", "nilai", "vol"
        ctx.qc("INFO", key, "nama kolom tidak dikenali -> dibaca berdasarkan posisi kolom")
    o = pd.DataFrame({"tahun": pd.to_numeric(d[tc], errors="coerce"), "prov": d[pv], "kabupaten": d[kb].map(norm_kab), "komoditas": d[kl].map(norm_komoditas),
                      "vol": d[vv].map(parse_number), "nilai_juta": d[nv].map(parse_number), "source_file": d["_file"], "_file": d["_file"]})
    o = o.dropna(subset=["tahun"]); o["tahun"] = o.tahun.astype(int); o = _newest_per_year(o)
    n0 = len(o); o = o[o.prov.map(norm_txt) == norm_txt(PROVINSI_TARGET)]
    if n0 != len(o):
        ctx.qc("INFO", key, f"{n0 - len(o)} baris non-{PROVINSI_TARGET} dibuang")
    # beberapa 'jenis ikan' dalam satu kelompok -> dijumlahkan agar kunci (kab, tahun, komoditas) unik
    o = o.groupby(["tahun", "kabupaten", "komoditas"], as_index=False, dropna=False).agg(
        vol=("vol", lambda s: s.sum(min_count=1)), nilai_juta=("nilai_juta", lambda s: s.sum(min_count=1)), source_file=("source_file", "last"))
    o = audit_frame(ctx, _ensure_flags(o), f"{key}_volume", main="vol", also=("nilai_juta",), lenient=(key != "pembesaran"))
    o = audit_frame(ctx, o, f"{key}_nilai", main="nilai_juta", lenient=(key != "pembesaran"))
    out = []
    for met, col, sat in [("nilai_rp_juta", "nilai_juta", "Rp Juta"), (metrik_vol, "vol", vol_col)]:
        t = o[["tahun", "kabupaten", "komoditas", "source_file", *FLAGS]].copy(); t["nilai"] = o[col]; t["metrik"] = met; t["satuan"] = sat; out.append(t)
    res = pd.concat(out, ignore_index=True); res["kegiatan"] = kegiatan; res["sumber"] = f"kkp_{kegiatan}"
    return res


# ----------------------------------------------------------------------------- rakit
def build_fact(config) -> pd.DataFrame:
    ctx = _Ctx(config)
    print("  membaca & mengaudit file mentah ...")
    prod = _opendata(ctx, "produksi_jenis", "produksi_ton", "jumlah")
    if not prod.empty:
        prod = audit_frame(ctx, prod, "produksi_jenis")
    rtp = _opendata(ctx, "rtp", "rtp", "jumlah")
    if not rtp.empty:
        rtp = audit_frame(ctx, rtp, "rtp", stale=True)
    luas = _opendata(ctx, "luas", "luas_lahan_ha", "luas")
    if not luas.empty:
        luas = _luas_vs_rtp(ctx, luas, rtp)
        luas = audit_frame(ctx, luas, "luas_opendata", stale=True)
        luas = _mark_spikes(ctx, luas, "luas_opendata", ["kabupaten", "jenis_budidaya"])
    luas_k = _luas_kkp(ctx)
    if not luas_k.empty:
        luas_k = audit_frame(ctx, luas_k, "luas_kkp", stale=True)
        luas_k = _mark_spikes(ctx, luas_k, "luas_kkp", ["kabupaten", "jenis_budidaya"]) if luas_k.jenis_budidaya.notna().any() else luas_k

    parts = [prod, rtp, luas, luas_k, _bps_kegiatan(ctx), _bps_nilai(ctx),
             _kkp(ctx, "hias", "ikan_hias", "Volume (Ekor)", "volume_ekor"),
             _kkp(ctx, "pembenihan", "pembenihan", "Volume (Ribu Ekor)", "volume_ribu_ekor"),
             _kkp(ctx, "pembesaran", "pembesaran", "Volume (ton)", "produksi_ton"),
             _pembudidaya(ctx)]

    ada = {p.sumber.iloc[0] for p in parts if not p.empty}
    hilang = [k for k in FILE_WAJIB if not ctx.files.get(k)]
    if hilang:
        _simpan_laporan(ctx, pd.DataFrame())
        raise FileNotFoundError(f"File WAJIB tidak ditemukan di {RAW_DIR}: {hilang}. Periksa nama file (kata kunci di config.FILE_RULES).")

    fact = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    for c in FACT_COLS:
        if c not in fact:
            fact[c] = np.nan
    fact = fact[FACT_COLS].copy()
    fact = fact.dropna(subset=["tahun", "kabupaten"]); fact["tahun"] = fact.tahun.astype(int)
    for f in FLAGS:
        fact[f] = fact[f].fillna(False).astype(bool)
    key = ["kabupaten", "tahun", "jenis_budidaya", "metrik", "kegiatan", "komoditas", "sumber"]
    dup = fact.assign(**{c: fact[c].fillna("∅") for c in ("jenis_budidaya", "kegiatan", "komoditas")}).duplicated(key, keep="first")
    if dup.any():
        ctx.qc("WARN", "fact", f"{int(dup.sum())} baris kunci ganda (kabupaten,tahun,jenis,metrik,kegiatan,komoditas,sumber) dibuang; baris pertama dipakai")
        fact = fact[~dup.values]
    fact = fact.sort_values(["sumber", "metrik", "tahun", "kabupaten"]).reset_index(drop=True)
    save_parquet(fact, PROCESSED_DIR / "fact_budidaya.parquet")
    _simpan_laporan(ctx, fact)
    return fact


def _simpan_laporan(ctx, fact):
    q = pd.DataFrame(ctx.qc_rows, columns=["level", "sumber", "tahun", "pesan"])
    ya = pd.concat(ctx.ya, ignore_index=True) if ctx.ya else pd.DataFrame()
    save_csv(q, REPORT_DIR / "qc_report.csv"); save_csv(ya, REPORT_DIR / "year_audit.csv"); save_csv(pd.DataFrame(ctx.log_rows, columns=["sumber", "masalah", "bukti", "tindakan"]), REPORT_DIR / "data_quality_log.csv")
    save_parquet(pd.DataFrame(ctx.bps_tot, columns=["tahun", "bps_jatim_ton", "sumber"]), PROCESSED_DIR / "bps_totals.parquet")
    files = {k: [{"nama": p.name, "ukuran": p.stat().st_size, "mtime": int(p.stat().st_mtime)} for p in v] for k, v in ctx.files.items()}
    h = hashlib.sha1(json.dumps(files, sort_keys=True).encode()).hexdigest()[:12]
    save_json({"dibuat": datetime.now().isoformat(), "files": files, "data_hash": h, "tahun": sorted(int(y) for y in fact.tahun.unique()) if len(fact) else []}, REPORT_DIR / "run_meta.json")
    print(f"  QC: {q.level.value_counts().to_dict() if len(q) else {}} | data_hash={h}")