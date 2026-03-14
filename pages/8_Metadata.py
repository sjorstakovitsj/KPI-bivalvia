# pages/08_✅_Datakwaliteit_en_metadata.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import load_data, render_sidebar

st.set_page_config(page_title="Datakwaliteit & metadata", layout="wide")

# -----------------------------
# Gedeelde sidebar (zelfde patroon als 4_ADV.py)
# -----------------------------
ui = render_sidebar(title="Mosselkartering")
years_sel = list(ui.get("years", []))
combine_years = bool(ui.get("combine_years", False))
keep_only_canonical = bool(ui.get("keep_only_canonical", False))

st.title("✅ Datakwaliteit en metadata")

if not years_sel:
    st.warning("Selecteer in de sidebar ten minste één monitoringsjaar.")
    st.stop()


# -----------------------------
# Data helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def _load(years: tuple[str, ...], combine: bool, keep_only: bool) -> dict[str, pd.DataFrame]:
    return load_data(
        years=list(years),
        combine_years=combine,
        keep_only_canonical=keep_only,
    )


def _get_table(data: dict[str, pd.DataFrame], base_key: str, years: list[str]) -> pd.DataFrame:
    """
    Haal een tabel op uit load_data(), robuust voor single-year en multi-year.
    - Als combine_years=True: verwacht key base_key met kolom 'jaar'
    - Als combine_years=False en meerdere jaren: verwacht keys base_key_<jaar>
    """
    if base_key in data and isinstance(data[base_key], pd.DataFrame) and not data[base_key].empty:
        df = data[base_key].copy()
        if "jaar" not in df.columns and len(years) == 1:
            df["jaar"] = str(years[0])
        return df

    frames: list[pd.DataFrame] = []
    for y in years:
        k = f"{base_key}_{y}"
        dfy = data.get(k)
        if isinstance(dfy, pd.DataFrame) and not dfy.empty:
            tmp = dfy.copy()
            tmp["jaar"] = str(y)
            frames.append(tmp)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _canonicalize_measurements_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maak measurements-kolommen robuust tegen lower/upper case en kleine naamvarianten.
    Hierdoor blijft de pagina werken, ook als load_data() intern kolommen normaliseert.
    """
    out = df.copy()
    col_map = {}

    for c in out.columns:
        c_norm = str(c).strip().lower()

        if c_norm in {"datum", "date", "datum_tijd", "datumtijd"}:
            col_map[c] = "Datum"
        elif c_norm in {"deelgebied", "deel_gebied"}:
            col_map[c] = "Deelgebied"
        elif c_norm in {"locatie", "meetpunt", "meetpunten"}:
            col_map[c] = "Locatie"
        elif c_norm in {"lat", "latitude"}:
            col_map[c] = "lat"
        elif c_norm in {"lon", "lng", "longitude"}:
            col_map[c] = "lon"
        elif c_norm in {"opmerkingen", "opmerking", "remarks"}:
            col_map[c] = "Opmerkingen"
        elif c_norm in {"jaar", "year"}:
            col_map[c] = "jaar"
        elif c_norm in {"x_planned_rd", "xplannedrd"}:
            col_map[c] = "x_planned_rd"
        elif c_norm in {"y_planned_rd", "yplannedrd"}:
            col_map[c] = "y_planned_rd"
        elif c_norm in {"x_rd", "xrd"}:
            col_map[c] = "x_rd"
        elif c_norm in {"y_rd", "yrd"}:
            col_map[c] = "y_rd"
        elif c_norm.startswith("sedimenttype_"):
            col_map[c] = str(c).strip()
        elif c_norm.startswith("pas_"):
            suffix = str(c).strip().split("_", 1)[1] if "_" in str(c).strip() else ""
            col_map[c] = f"PAS_{suffix}" if suffix else "PAS"
        elif c_norm.startswith("lutum_"):
            col_map[c] = str(c).strip()

    return out.rename(columns=col_map)


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _unique_nonempty_values(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    if not columns:
        return []
    vals = pd.Series(pd.unique(frame[columns].astype(str).values.ravel("K")))
    vals = vals[~vals.isin(["nan", "None", "-", "", "<NA>"])]
    vals = vals.dropna()
    return sorted(vals.tolist())


# -----------------------------
# Data ophalen
# -----------------------------
DATA = _load(tuple(years_sel), combine_years, keep_only_canonical)
meas = _get_table(DATA, "measurements", years_sel)

if meas is None or meas.empty:
    st.warning(
        "Geen measurements data gevonden "
        "(measurements.parquet ontbreekt/is leeg of load_data() levert geen measurements op)."
    )
    st.stop()

meas = _canonicalize_measurements_columns(meas)

required_cols = {"Datum", "Deelgebied", "Locatie"}
missing = required_cols - set(meas.columns)
if missing:
    st.error(
        "Ontbrekende kolommen in measurements: "
        f"{', '.join(sorted(missing))}\n\n"
        f"Gevonden kolommen: {list(meas.columns)}"
    )
    st.stop()

meas["Datum"] = pd.to_datetime(meas["Datum"], errors="coerce")
meas = meas[meas["Datum"].notna()].copy()
meas["Locatie"] = meas["Locatie"].astype("string").str.strip()
meas = meas[meas["Locatie"].notna()].copy()
meas["Deelgebied"] = meas["Deelgebied"].astype("string").str.strip()

# Jaar robuust afleiden uit Datum (origineel gedrag) en, indien aanwezig, ook als string laten staan
meas["jaar_num"] = meas["Datum"].dt.year.astype("Int64")
if "jaar" not in meas.columns:
    meas["jaar"] = meas["jaar_num"].astype("string")
else:
    meas["jaar"] = meas["jaar"].astype("string")

# Numeriek voorbereiden waar relevant
for c in ["lat", "lon", "x_planned_rd", "y_planned_rd", "x_rd", "y_rd"]:
    if c in meas.columns:
        meas[c] = _safe_numeric(meas[c])

# -----------------------------
# Paginafilters (onder gedeelde sidebar)
# -----------------------------
with st.sidebar:
    st.markdown("---")
    st.subheader("📌 Paginafilters")

    deelgebieden = sorted(meas["Deelgebied"].dropna().astype(str).unique().tolist())
    sel_deelgebied = st.multiselect(
        "Deelgebied",
        options=deelgebieden,
        default=deelgebieden,
        key="metadata_deelgebied",
    )

    show_missing_top_n = st.slider(
        "Aantal kolommen in missing-plot",
        min_value=10,
        max_value=100,
        value=25,
        step=5,
        key="metadata_missing_top_n",
    )

view = meas[meas["Deelgebied"].astype(str).isin(sel_deelgebied)].copy()
if view.empty:
    st.warning("Geen data na selectie.")
    st.stop()

# -----------------------
# Observaties per jaar
# -----------------------
st.subheader("Aantal waarnemingen per jaar")
counts_year = (
    view.dropna(subset=["jaar_num"])
    .groupby("jaar_num", as_index=False)
    .size()
    .rename(columns={"size": "n"})
    .sort_values("jaar_num")
)

if counts_year.empty:
    st.info("Geen geldige datums/jaren beschikbaar in de huidige selectie.")
else:
    fig = px.bar(counts_year, x="jaar_num", y="n", labels={"jaar_num": "Jaar", "n": "Aantal"})
    st.plotly_chart(fig, use_container_width=True)

# per deelgebied
st.subheader("Aantal waarnemingen per jaar per deelgebied")
counts_year_area = (
    view.dropna(subset=["jaar_num", "Deelgebied"])
    .groupby(["jaar_num", "Deelgebied"], as_index=False)
    .size()
    .rename(columns={"size": "n"})
)

if counts_year_area.empty:
    st.info("Geen geldige jaar/deelgebied-combinaties beschikbaar.")
else:
    fig2 = px.bar(
        counts_year_area,
        x="jaar_num",
        y="n",
        color="Deelgebied",
        barmode="stack",
        labels={"jaar_num": "Jaar", "n": "Aantal"},
    )
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------
# Meetgaten
# -----------------------
st.subheader("Meetgaten (locaties zonder waarneming in een jaar)")
years = sorted(view["jaar_num"].dropna().astype(int).unique().tolist())
if len(years) < 2:
    st.info("Meetgaten-analyse wordt pas zinvol bij meerdere jaren data.")
else:
    pivot = (
        view.assign(waarde=1)
        .pivot_table(index="Locatie", columns="jaar_num", values="waarde", aggfunc="max", fill_value=0)
        .sort_index()
    )
    st.dataframe(pivot, use_container_width=True)

    heat_df = pivot.reset_index().melt(id_vars="Locatie", var_name="jaar", value_name="gemeten")
    fig_gap = px.density_heatmap(
        heat_df,
        x="jaar",
        y="Locatie",
        z="gemeten",
        histfunc="avg",
        color_continuous_scale="Viridis",
        title="Meetgaten-heatmap (1 = gemeten, 0 = geen waarneming)",
    )
    st.plotly_chart(fig_gap, use_container_width=True)

# -----------------------
# Consistentie codes (sediment / PAS / lutum)
# -----------------------
st.subheader("Consistentie van codes (sedimenttype / PAS)")
sed_cols = [c for c in view.columns if str(c).startswith("sedimenttype_")]
pas_cols = [c for c in view.columns if str(c).startswith("PAS_")]
lut_cols = [c for c in view.columns if str(c).startswith("lutum_")]

col1, col2, col3 = st.columns(3)
with col1:
    if sed_cols:
        sed_vals = _unique_nonempty_values(view, sed_cols)
        st.write("Sedimenttype unieke waarden:")
        st.write(sed_vals)
    else:
        st.info("Geen sedimenttype-kolommen gevonden.")

with col2:
    if pas_cols:
        pas_vals = _unique_nonempty_values(view, pas_cols)
        st.write("PAS unieke waarden:")
        st.write(pas_vals)
    else:
        st.info("Geen PAS-kolommen gevonden.")

with col3:
    if lut_cols:
        lut_vals = _unique_nonempty_values(view, lut_cols)
        st.write("Lutum-klassen unieke waarden:")
        st.write(lut_vals)
    else:
        st.info("Geen lutum-kolommen gevonden.")

# -----------------------
# Ruimtelijke dekking
# -----------------------
st.subheader("Ruimtelijke dekking")
if {"lat", "lon"}.issubset(view.columns) and view[["lat", "lon"]].dropna().shape[0] > 0:
    st.map(view[["lat", "lon"]].dropna())
else:
    st.info("Geen lat/lon-kolommen of geen geldige coördinaten beschikbaar in de huidige selectie.")

# -----------------------
# Coördinaat-verschil gepland vs uitgevoerd (RD)
# -----------------------
st.subheader("Planned vs. uitgevoerd (RD) – afwijking (m)")
rd_required = {"x_planned_rd", "y_planned_rd", "x_rd", "y_rd"}
if rd_required.issubset(set(view.columns)):
    d = view.copy()
    for c in ["x_planned_rd", "y_planned_rd", "x_rd", "y_rd"]:
        d[c] = _safe_numeric(d[c])

    d["delta_m"] = np.sqrt((d["x_rd"] - d["x_planned_rd"]) ** 2 + (d["y_rd"] - d["y_planned_rd"]) ** 2)
    valid_d = d.dropna(subset=["delta_m"]).copy()

    if valid_d.empty:
        st.info("Geen geldige RD-coördinaten beschikbaar om afwijkingen te berekenen.")
    else:
        figd = px.histogram(valid_d, x="delta_m", nbins=30, title="Verdeling afwijking (m)")
        st.plotly_chart(figd, use_container_width=True)

        st.write("Top 10 grootste afwijkingen:")
        top_cols = [c for c in ["jaar", "Deelgebied", "Locatie", "Datum", "delta_m", "Opmerkingen"] if c in valid_d.columns]
        top = valid_d.sort_values("delta_m", ascending=False).head(10)[top_cols]
        st.dataframe(top, use_container_width=True, hide_index=True)
else:
    st.info("Planned/uitgevoerd RD-coördinaten zijn niet compleet in de dataset.")

# -----------------------
# Completeness / missing values
# -----------------------
st.subheader("Completeness (missende waarden per kolom)")
miss = view.isna().mean().sort_values(ascending=False).reset_index()
miss.columns = ["kolom", "missing_fractie"]
miss["missing_%"] = (miss["missing_fractie"] * 100).round(1)

figm = px.bar(
    miss.head(show_missing_top_n),
    x="missing_%",
    y="kolom",
    orientation="h",
    title=f"Top missende kolommen (max {show_missing_top_n})",
)
st.plotly_chart(figm, use_container_width=True)

with st.expander("Volledige missing-tabel"):
    st.dataframe(miss, use_container_width=True, hide_index=True)

csv = miss.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download missing-tabel als CSV",
    data=csv,
    file_name="metadata_missing_tabel.csv",
    mime="text/csv",
)
