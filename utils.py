# utils.py
# Hulpfuncties voor het Streamlit-dashboard mosselkartering
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import numpy as np
import re

# Optioneel DuckDB (valt terug op Parquet als niet aanwezig)
USE_DUCKDB = os.getenv("USE_DUCKDB", "0") == "1"
try:
    import duckdb  # type: ignore
except Exception:
    duckdb = None
    USE_DUCKDB = False

# Root directory met outputs van dataprep.py
# NB: bij multi-year schrijft dataprep.py naar processed/<jaar>/*.parquet.
DATA_DIR = Path(os.getenv("DATA_DIR", "processed"))

# ------------------------------------------------------------
# RD -> WGS84 (benadering)
# ------------------------------------------------------------
# Gebaseerd op RDNAP-transformatie (benadering). Voor nauwkeurige productie-doeleinden
# verdient pyproj/geopandas de voorkeur.

def rd_to_wgs84(x: float, y: float) -> tuple[float, float]:
    p = (x - 155000.0) / 100000.0
    q = (y - 463000.0) / 100000.0
    K = [
        (0, 1, 3235.65389),
        (2, 0, -32.58297),
        (0, 2, -0.2475),
        (2, 1, -0.84978),
        (0, 3, -0.0655),
        (2, 2, -0.01709),
        (1, 0, -0.00738),
        (4, 0, 0.0053),
        (2, 3, -0.00039),
        (4, 1, 0.00033),
        (1, 1, -0.00012),
    ]
    L = [
        (1, 0, 5260.52916),
        (1, 1, 105.94684),
        (1, 2, 2.45656),
        (3, 0, -0.81885),
        (1, 3, 0.05594),
        (3, 1, -0.05607),
        (0, 1, 0.01199),
        (3, 2, -0.00256),
        (1, 4, 0.00128),
        (0, 2, 0.00022),
        (2, 0, -0.00022),
        (5, 0, 0.00026),
    ]
    phi = 52.15517440
    lam = 5.38720621
    for a, b, c in K:
        phi += c * (p**a) * (q**b) / 3600.0
    for a, b, c in L:
        lam += c * (p**a) * (q**b) / 3600.0
    return phi, lam


# ------------------------------------------------------------
# Deelgebied normalisatie
# ------------------------------------------------------------
# Uitgebreid met extra meren/wateren die in andere monitoringsjaren voorkomen.
CANONICAL_DEELGEBIEDEN = {
    "Hoornse Hop",
    "IJmeer",
    "Markermeer Noord",
    "Markermeer Midden",
    "Markermeer Zuid",
    "Zwartemeer",
    "Ketelmeer",
    "Vossemeer",
    "Drontermeer",
    "Veluwemeer",
    "Wolderwijd",
    "Nuldernauw",
    "Eemmeer",
    "Gooimeer",
    "Nijkerkernauw",
    "Reevediep",
}


def normalize_deelgebied(val):
    """Normaliseer schrijfwijzen/typos van deelgebieden.

    Let op: onbekende deelgebieden blijven intact (worden niet weggefilterd).
    """
    if pd.isna(val):
        return pd.NA
    s = str(val).strip()
    key = re.sub(r"[^a-z0-9]", "", s.lower())

    # Midden-Nederland varianten
    if "ijmeer" in key:
        return "IJmeer"
    if "hoornsehop" in key:
        return "Hoornse Hop"
    if "markermeer" in key and "noord" in key:
        return "Markermeer Noord"
    if (
        ("markermeer" in key and "midden" in key)
        or ("markeermeer" in key and "midden" in key)
        or ("markerneer" in key and "midden" in key)
    ):
        return "Markermeer Midden"
    if "markermeer" in key and "zuid" in key:
        return "Markermeer Zuid"

    # Overige expliciete wateren (ruime match)
    # Hiermee vangen we o.a. varianten als 'Ketelmeer (oost)' of 'Gooimeer-West'.
    for dg in [
        "Zwartemeer",
        "Ketelmeer",
        "Vossemeer",
        "Drontermeer",
        "Veluwemeer",
        "Wolderwijd",
        "Nuldernauw",
        "Eemmeer",
        "Gooimeer",
        "Nijkerkernauw",
        "Reevediep",
    ]:
        if re.sub(r"[^a-z0-9]", "", dg.lower()) in key:
            return dg

    return s


def normalize_deelgebied_col(df: pd.DataFrame, keep_only_canonical: bool = False) -> pd.DataFrame:
    """Normaliseer Deelgebied-kolom in een DataFrame.

    - keep_only_canonical=False (default): behoud alle deelgebieden (multi-year).
    - keep_only_canonical=True: behoud alleen CANONICAL_DEELGEBIEDEN (oude gedrag).

    Functionaliteit blijft behouden doordat je het oude gedrag nog expliciet kunt kiezen.
    """
    if df is None or df.empty or "Deelgebied" not in df.columns:
        return df
    out = df.copy()
    out["Deelgebied"] = out["Deelgebied"].apply(normalize_deelgebied)
    if keep_only_canonical:
        out = out[out["Deelgebied"].isin(CANONICAL_DEELGEBIEDEN)].copy()
    return out


# ------------------------------------------------------------
# Multi-year: years discovery / path helpers
# ------------------------------------------------------------

def list_years(data_dir: Path | None = None) -> list[str]:
    """Vind jaardirectories (processed/<jaar>) op basis van submappen met 4 cijfers."""
    root = data_dir or DATA_DIR
    if not root.exists():
        return []
    years: list[str] = []
    for p in root.iterdir():
        if p.is_dir() and re.fullmatch(r"\d{4}", p.name):
            years.append(p.name)
    years.sort()
    return years


def resolve_year_dir(year: str | None, data_dir: Path | None = None) -> Path:
    """Geef de directory terug waar Parquet-bestanden staan.

    - Als year is None: root (DATA_DIR)
    - Als year is not None: DATA_DIR/<year>
    """
    root = data_dir or DATA_DIR
    return root / year if year else root


def _parquet_path(name: str, year: str | None = None, data_dir: Path | None = None) -> Path:
    return resolve_year_dir(year, data_dir) / name


def _duckdb_path(year: str | None = None, data_dir: Path | None = None) -> Path:
    """DuckDB pad per jaar (processed/<jaar>/mussels.duckdb), of in root als year=None."""
    d = resolve_year_dir(year, data_dir)
    return Path(os.getenv("DB_PATH", str(d / "mussels.duckdb"))) if year is None else (d / "mussels.duckdb")


# ------------------------------------------------------------
# Data laden (Parquet / DuckDB)
# ------------------------------------------------------------

def ensure_duckdb_from_parquet(year: str | None = None, data_dir: Path | None = None) -> None:
    """Bouw (eenmalig) een DuckDB db op basis van de Parquet-bestanden voor performance.

    Voor multi-year bouwen we een DB per jaar-directory.
    """
    if duckdb is None or not USE_DUCKDB:
        return

    db_path = _duckdb_path(year, data_dir)
    if db_path.exists():
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        for fname, table in [
            ("measurements.parquet", "measurements"),
            ("populatie_counts.parquet", "populatie_counts"),
            ("populatie_percent.parquet", "populatie_percent"),
            ("adv_lenclass.parquet", "adv_lenclass"),
            ("adv_m2_locations.parquet", "adv_m2_locations"),
        ]:
            fpath = _parquet_path(fname, year, data_dir)
            if fpath.exists():
                con.execute(
                    f"CREATE TABLE {table} AS SELECT * FROM read_parquet('{fpath.as_posix()}')"
                )
    finally:
        con.close()


def load_table(name: str, *, year: str | None = None, data_dir: Path | None = None) -> pd.DataFrame:
    """Laad een tabel.

    - name: logical table name (bijv. 'measurements', 'adv_m2_locations', ...)
    - year: None (root) of '2024' etc.

    Backwards compatible: callers die alleen load_table("measurements") gebruiken blijven werken.
    """
    if USE_DUCKDB and duckdb is not None:
        ensure_duckdb_from_parquet(year, data_dir)
        db_path = _duckdb_path(year, data_dir)
        if db_path.exists():
            con = duckdb.connect(str(db_path), read_only=True)
            try:
                return con.execute(f"SELECT * FROM {name}").df()
            finally:
                con.close()

    # Parquet fallback
    return pd.read_parquet(_parquet_path(f"{name}.parquet", year, data_dir))


def load_data(
    *,
    year: str | None = None,
    years: list[str] | None = None,
    combine_years: bool = False,
    keep_only_canonical: bool | None = None,
    data_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Laad de benodigde tabellen.

    Ondersteunt multi-year:
    - year='2024' laadt alleen dat jaar (processed/2024)
    - years=['2021','2023','2024'] laadt meerdere jaren.
    - combine_years=True plakt jaren onder elkaar en voegt kolom 'jaar' toe.

    keep_only_canonical:
    - None: lees uit env KEEP_ONLY_CANONICAL (default '0')
    - True/False: forceer gedrag

    Backwards compatible:
    - load_data() werkt nog steeds en laadt root (DATA_DIR) zoals voorheen.
    """

    root = data_dir or DATA_DIR

    # default canonical gedrag via env (oude app had canonical filtering aan).
    if keep_only_canonical is None:
        keep_only_canonical = os.getenv("KEEP_ONLY_CANONICAL", "0") == "1"

    # Bepaal welke jaren
    if years is None:
        years = [year] if year else [None]  # type: ignore[list-item]

    tables = {
        "measurements": "measurements",
        "adv_m2": "adv_m2_locations",
        "populatie_counts": "populatie_counts",
        "populatie_percent": "populatie_percent",
        "adv_lenclass": "adv_lenclass",
    }

    data: dict[str, pd.DataFrame] = {}

    if combine_years and years != [None]:
        # Multi-year concat
        for key, tname in tables.items():
            frames: list[pd.DataFrame] = []
            for y in years:
                if y is None:
                    continue
                try:
                    df = load_table(tname, year=y, data_dir=root)
                    df = normalize_deelgebied_col(df, keep_only_canonical=bool(keep_only_canonical))
                    df["jaar"] = str(y)
                    frames.append(df)
                except Exception:
                    continue
            data[key] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return data

    # Single-year or multi-year (separate)
    # Als meerdere jaren gevraagd zijn zonder combine_years, geven we keys met suffix terug.
    if len(years) > 1 and years != [None]:
        for y in years:
            if y is None:
                continue
            for key, tname in tables.items():
                try:
                    df = load_table(tname, year=y, data_dir=root)
                    df = normalize_deelgebied_col(df, keep_only_canonical=bool(keep_only_canonical))
                    data[f"{key}_{y}"] = df
                except Exception:
                    data[f"{key}_{y}"] = pd.DataFrame()
        return data

    # Default: één directory (root of year)
    y0 = years[0]
    for key, tname in tables.items():
        try:
            df = load_table(tname, year=y0 if y0 is not None else None, data_dir=root)
            df = normalize_deelgebied_col(df, keep_only_canonical=bool(keep_only_canonical))
            data[key] = df
        except Exception:
            data[key] = pd.DataFrame()

    return data


# ------------------------------------------------------------
# Afgeleide berekeningen
# ------------------------------------------------------------

def avg_lutum_percentage(df: pd.DataFrame) -> pd.Series:
    """Zet lutum-klassen (tekst: '8-12', '>35' etc.) om naar numerieke middenwaarden en neem gemiddelde per meetpunt."""

    def to_mid(s: str) -> float | None:
        if pd.isna(s):
            return None
        s = str(s).strip()
        if s.startswith(">"):
            try:
                return float(s[1:]) + 2.5  # arbitraire marge
            except Exception:
                return None
        if "-" in s:
            a, b = s.split("-", 1)
            try:
                return (float(a) + float(b)) / 2
            except Exception:
                return None
        try:
            return float(s)
        except Exception:
            return None

    lut_cols = [c for c in df.columns if str(c).startswith("lutum_")]
    if not lut_cols:
        return pd.Series([np.nan] * len(df), index=df.index)

    tmp = df[lut_cols].applymap(to_mid)
    return tmp.mean(axis=1, skipna=True)


# ------------------------------------------------------------
# Visual helpers
# ------------------------------------------------------------
SPECIES_COLORS = {"driehoek": "#1f77b4", "quagga": "#ff7f0e"}


# ------------------------------------------------------------
# Streamlit helper: gedeelde sidebar
# ------------------------------------------------------------

def render_sidebar(
    *,
    title: str = "Mosselkartering",
    default_years: list[str] | None = None,
    allow_multi_year: bool = True,
    allow_combine: bool = True,
) -> dict[str, object]:
    """Render een gedeelde sidebar voor alle Streamlit pagina's.

    Dit is bedoeld om in elke pagina aan te roepen:
        from utils import render_sidebar
        state = render_sidebar()

    Retourneert o.a.:
    - years: list[str]
    - combine_years: bool
    - keep_only_canonical: bool

    Let op: Streamlit is optioneel. Als streamlit niet beschikbaar is, geeft deze functie defaults terug.
    """
    try:
        import streamlit as st  # type: ignore
    except Exception:
        # Non-streamlit context (tests / scripts)
        yrs = default_years or list_years() or []
        return {
            "years": yrs,
            "combine_years": False,
            "keep_only_canonical": False,
        }

    st.sidebar.title(title)

    years_available = list_years()
    if not years_available:
        st.sidebar.info(
            "Geen jaardirectories gevonden in 'processed/'. Verwacht structuur processed/<jaar>/*.parquet."
        )

    if default_years is None:
        # Default: laatste jaar geselecteerd (als beschikbaar)
        default_years = [years_available[-1]] if years_available else []

    if allow_multi_year:
        years_sel = st.sidebar.multiselect(
            "Monitoringsjaar(en)",
            options=years_available,
            default=[y for y in default_years if y in years_available],
        )
    else:
        y = st.sidebar.selectbox(
            "Monitoringsjaar",
            options=years_available,
            index=(len(years_available) - 1) if years_available else 0,
        )
        years_sel = [y] if y else []

    combine_years = False
    if allow_combine and allow_multi_year:
        combine_years = st.sidebar.checkbox(
            "Combineer geselecteerde jaren (voeg kolom 'jaar' toe)",
            value=False,
        )

    keep_only_canonical = st.sidebar.checkbox(
        "Filter deelgebieden op vaste set (legacy)",
        value=(os.getenv("KEEP_ONLY_CANONICAL", "0") == "1"),
        help="Zet aan om alleen bekende deelgebieden te tonen (oude dashboard-gedrag).",
    )

    return {
        "years": years_sel,
        "combine_years": combine_years,
        "keep_only_canonical": keep_only_canonical,
    }
