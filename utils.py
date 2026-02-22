# utils.py
# Hulpfuncties voor het Streamlit-dashboard mosselkartering
from __future__ import annotations
import os
from pathlib import Path
import math
import json
import pandas as pd
import numpy as np
import re

# Optioneel DuckDB (valt terug op Parquet als niet aanwezig)
USE_DUCKDB = os.getenv('USE_DUCKDB', '0') == '1'

try:
    import duckdb  # type: ignore
except Exception:
    duckdb = None
    USE_DUCKDB = False

DATA_DIR = Path(os.getenv('DATA_DIR', 'processed'))
DB_PATH = Path(os.getenv('DB_PATH', DATA_DIR / 'mussels.duckdb'))

# ---------- RD -> WGS84 ----------
# Gebaseerd op RDNAP-transformatie (benadering). Voor nauwkeurige productie-doeleinden
# verdient pyproj/geopandas de voorkeur.
def rd_to_wgs84(x: float, y: float) -> tuple[float, float]:
    p = (x - 155000.0) / 100000.0
    q = (y - 463000.0) / 100000.0
    K = [
        (0, 1, 3235.65389),(2, 0, -32.58297),(0, 2, -0.2475),(2, 1, -0.84978),
        (0, 3, -0.0655),(2, 2, -0.01709),(1, 0, -0.00738),(4, 0, 0.0053),
        (2, 3, -0.00039),(4, 1, 0.00033),(1, 1, -0.00012)]
    L = [
        (1, 0, 5260.52916),(1, 1, 105.94684),(1, 2, 2.45656),(3, 0, -0.81885),
        (1, 3, 0.05594),(3, 1, -0.05607),(0, 1, 0.01199),(3, 2, -0.00256),
        (1, 4, 0.00128),(0, 2, 0.00022),(2, 0, -0.00022),(5, 0, 0.00026)]
    phi = 52.15517440
    lam = 5.38720621
    for a,b,c in K:
        phi += c * (p**a) * (q**b) / 3600.0
    for a,b,c in L:
        lam += c * (p**a) * (q**b) / 3600.0
    return phi, lam

# ---------- Data laden ----------

def _parquet_path(name: str) -> Path:
    return DATA_DIR / name


def ensure_duckdb_from_parquet() -> None:
    """Bouw (eenmalig) een DuckDB db op basis van de Parquet-bestanden voor performance."""
    if duckdb is None:
        return
    if DB_PATH.exists():
        return
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    try:
        for fname, table in [
            ('measurements.parquet','measurements'),
            ('populatie_counts.parquet','populatie_counts'),
            ('populatie_percent.parquet','populatie_percent'),
            ('adv_lenclass.parquet','adv_lenclass'),
            ('adv_m2_locations.parquet','adv_m2_locations'),
        ]:
            fpath = _parquet_path(fname)
            if fpath.exists():
                con.execute(f"CREATE TABLE {table} AS SELECT * FROM read_parquet('{fpath.as_posix()}')")
    finally:
        con.close()


def load_table(name: str) -> pd.DataFrame:
    if USE_DUCKDB and duckdb is not None and DB_PATH.exists():
        con = duckdb.connect(str(DB_PATH), read_only=True)
        try:
            return con.execute(f"SELECT * FROM {name}").df()
        finally:
            con.close()
    # Parquet fallback
    return pd.read_parquet(_parquet_path(f"{name}.parquet"))

import re  # toevoegen bovenaan

CANONICAL_DEELGEBIEDEN = {
    "Hoornse Hop",
    "IJmeer",
    "Markermeer Noord",
    "Markermeer Midden",
    "Markermeer Zuid",
}

def normalize_deelgebied(val):
    if pd.isna(val):
        return pd.NA
    s = str(val).strip()
    key = re.sub(r"[^a-z0-9]", "", s.lower())

    if "ijmeer" in key:
        return "IJmeer"
    if "hoornsehop" in key:
        return "Hoornse Hop"
    if "markermeer" in key and "noord" in key:
        return "Markermeer Noord"
    if ("markermeer" in key and "midden" in key) or ("markeermeer" in key and "midden" in key) or ("markerneer" in key and "midden" in key):
        return "Markermeer Midden"
    if "markermeer" in key and "zuid" in key:
        return "Markermeer Zuid"

    return s

def normalize_deelgebied_col(df: pd.DataFrame, keep_only_canonical: bool = True) -> pd.DataFrame:
    if df is None or df.empty or "Deelgebied" not in df.columns:
        return df
    out = df.copy()
    out["Deelgebied"] = out["Deelgebied"].apply(normalize_deelgebied)
    if keep_only_canonical:
        out = out[out["Deelgebied"].isin(CANONICAL_DEELGEBIEDEN)].copy()
    return out

def load_data() -> dict[str, pd.DataFrame]:
    ensure_duckdb_from_parquet()
    data = {
        "measurements": load_table("measurements"),
        "adv_m2": load_table("adv_m2_locations"),
    }
    for t in ["populatie_counts", "populatie_percent", "adv_lenclass"]:
        try:
            data[t] = load_table(t)
        except Exception:
            pass

    # Safety-net normalisatie voor tabellen met Deelgebied-kolom
    for k in list(data.keys()):
        data[k] = normalize_deelgebied_col(data[k], keep_only_canonical=True)

    return data

# ---------- Afgeleide berekeningen ----------

def avg_lutum_percentage(df: pd.DataFrame) -> pd.Series:
    """Zet lutum-klassen (tekst: '8-12', '>35' etc.) om naar numerieke middenwaarden en neem gemiddelde per meetpunt."""
    def to_mid(s: str) -> float | None:
        if pd.isna(s):
            return None
        s = str(s).strip()
        if s.startswith('>'):
            try:
                return float(s[1:]) + 2.5  # arbitraire marge
            except Exception:
                return None
        if '-' in s:
            a,b = s.split('-',1)
            try:
                return (float(a)+float(b))/2
            except Exception:
                return None
        try:
            return float(s)
        except Exception:
            return None
    lut_cols = [c for c in df.columns if c.startswith('lutum_')]
    tmp = df[lut_cols].applymap(to_mid)
    return tmp.mean(axis=1, skipna=True)

# ---------- Visual helpers ----------

SPECIES_COLORS = {
    'driehoek': '#1f77b4',
    'quagga': '#ff7f0e'
}
