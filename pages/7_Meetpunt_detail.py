# pages/07_🔍_Meetpunt_detail.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium

from utils import load_data, render_sidebar

st.set_page_config(page_title="Meetpunt detail", layout="wide")

# -----------------------------
# Gedeelde sidebar (zelfde patroon als 4_ADV.py)
# -----------------------------
ui = render_sidebar(title="Mosselkartering")
years_sel = list(ui.get("years", []))
combine_years = bool(ui.get("combine_years", False))

st.title("🔍 Deep-dive per meetlocatie")

if not years_sel:
    st.warning("Selecteer in de sidebar ten minste één monitoringsjaar.")
    st.stop()


# -----------------------------
# Data helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def _load(years: tuple[str, ...], combine: bool) -> dict[str, pd.DataFrame]:
    return load_data(
        years=list(years),
        combine_years=combine,
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
        elif c_norm in {"diepte_m", "diepte", "waterdiepte", "waterdiepte_m"}:
            col_map[c] = "diepte_m"
        elif c_norm in {"biovol_totaal_ml", "biovol totaal ml", "biovol_totaal"}:
            col_map[c] = "biovol_totaal_ml"
        elif c_norm in {"biovol_driehoek_ml", "biovol driehoek ml", "biovol_driehoek"}:
            col_map[c] = "biovol_driehoek_ml"
        elif c_norm in {"biovol_quagga_ml", "biovol quagga ml", "biovol_quagga"}:
            col_map[c] = "biovol_quagga_ml"
        elif c_norm in {"opmerkingen", "opmerking", "remarks"}:
            col_map[c] = "Opmerkingen"
        elif c_norm in {"jaar", "year"}:
            col_map[c] = "jaar"
        elif c_norm.startswith("sedimenttype_"):
            col_map[c] = str(c).strip()
        elif c_norm.startswith("lutum_"):
            col_map[c] = str(c).strip()
        elif c_norm.startswith("pas_"):
            # standaardiseer hoofdletters voor PAS
            suffix = str(c).strip().split("_", 1)[1] if "_" in str(c).strip() else ""
            col_map[c] = f"PAS_{suffix}" if suffix else "PAS"

    out = out.rename(columns=col_map)
    return out


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# -----------------------------
# Data ophalen
# -----------------------------
DATA = _load(tuple(years_sel), combine_years)
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

# Numerieke kolommen veilig voorbereiden indien aanwezig
for c in ["lat", "lon", "diepte_m", "biovol_driehoek_ml", "biovol_quagga_ml", "biovol_totaal_ml"]:
    if c in meas.columns:
        meas[c] = _safe_numeric(meas[c])

# Pagina-specifieke filters in sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("📌 Paginafilters")

    deelgebieden = sorted(meas["Deelgebied"].dropna().astype(str).unique().tolist())
    sel_deelgebied = st.selectbox(
        "Deelgebied",
        options=["(alle)"] + deelgebieden,
        key="meetpunt_detail_deelgebied",
    )

    base = meas.copy()
    if sel_deelgebied != "(alle)":
        base = base[base["Deelgebied"].astype(str) == str(sel_deelgebied)].copy()

    locaties = sorted(base["Locatie"].dropna().astype(str).unique().tolist())
    sel = st.selectbox(
        "Kies meetlocatie",
        options=locaties,
        key="meetpunt_detail_locatie",
    )

if not sel:
    st.warning("Geen meetlocaties beschikbaar voor de huidige selectie.")
    st.stop()

m = meas[meas["Locatie"].astype(str) == str(sel)].sort_values("Datum").copy()
if sel_deelgebied != "(alle)":
    m = m[m["Deelgebied"].astype(str) == str(sel_deelgebied)].copy()

if m.empty:
    st.warning("Geen data voor deze locatie.")
    st.stop()

# -----------------------------
# Info block + kaart
# -----------------------------
col1, col2 = st.columns([1, 2])

with col1:
    if {"lat", "lon"}.issubset(m.columns) and m[["lat", "lon"]].dropna().shape[0] > 0:
        map_df = m.dropna(subset=["lat", "lon"]).copy()
        center = [float(map_df["lat"].mean()), float(map_df["lon"].mean())]
        mp = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

        for _, r in map_df.iterrows():
            jaar_txt = f" | jaar {r['jaar']}" if "jaar" in map_df.columns and pd.notna(r.get("jaar", None)) else ""
            folium.CircleMarker(
                location=[float(r["lat"]), float(r["lon"])],
                radius=6,
                color="#1f77b4",
                fill=True,
                fill_opacity=0.8,
                tooltip=f"Locatie {str(r['Locatie'])} – {pd.to_datetime(r['Datum']).date()}{jaar_txt}",
            ).add_to(mp)

        st_folium(mp, height=360, width="stretch")
    else:
        st.info("Geen kaartcoördinaten (lat/lon) beschikbaar voor deze locatie.")

with col2:
    st.subheader("Samenvatting")

    opmerkingen = []
    if "Opmerkingen" in m.columns:
        opmerkingen = (
            m["Opmerkingen"]
            .dropna()
            .astype(str)
            .map(str.strip)
            .replace("", pd.NA)
            .dropna()
            .unique()
            .tolist()
        )

    summary = {
        "Deelgebied": m["Deelgebied"].iloc[0] if "Deelgebied" in m.columns and len(m) else None,
        "Locatie": str(sel),
        "Aantal observaties": int(len(m)),
        "Periode": f"{m['Datum'].min().date()} — {m['Datum'].max().date()}",
    }

    if "jaar" in m.columns:
        summary["Jaren"] = sorted(m["jaar"].dropna().astype(str).unique().tolist())
    if "diepte_m" in m.columns:
        summary["Gem. diepte (m)"] = float(_safe_numeric(m["diepte_m"]).mean()) if _safe_numeric(m["diepte_m"]).notna().any() else None
    if "biovol_totaal_ml" in m.columns:
        summary["Totaal biovolume (ml)"] = float(_safe_numeric(m["biovol_totaal_ml"]).sum())
    if opmerkingen:
        summary["Opmerkingen (uniek)"] = opmerkingen

    st.write(summary)

# -----------------------------
# Tijdreeks
# -----------------------------
st.subheader("Tijdreeks biovolume")

plot_cols = [c for c in ["biovol_driehoek_ml", "biovol_quagga_ml", "biovol_totaal_ml"] if c in m.columns]
if not plot_cols:
    st.info("Geen biovolume-kolommen gevonden voor deze locatie.")
else:
    plot_df = m.copy()
    facet_by_year = "jaar" in plot_df.columns and plot_df["jaar"].nunique() > 1

    if facet_by_year:
        long_df = plot_df.melt(
            id_vars=[c for c in ["Datum", "jaar"] if c in plot_df.columns],
            value_vars=plot_cols,
            var_name="Reeks",
            value_name="ml",
        )
        fig = px.line(
            long_df,
            x="Datum",
            y="ml",
            color="Reeks",
            facet_row="jaar",
            markers=True,
            title="Biovolume door de tijd",
            labels={"ml": "ml", "Reeks": "Reeks", "jaar": "Jaar"},
        )
    else:
        fig = px.line(
            plot_df,
            x="Datum",
            y=plot_cols,
            markers=True,
            title="Biovolume door de tijd",
            labels={"value": "ml", "variable": "Reeks"},
        )

    st.plotly_chart(fig, width="stretch")

# -----------------------------
# Hapdetails
# -----------------------------
st.subheader("Sediment / lutum / PAS per hap")
hap_cols = [
    c for c in m.columns
    if str(c).startswith("sedimenttype_") or str(c).startswith("lutum_") or str(c).startswith("PAS_")
]

if hap_cols:
    detail_cols = [c for c in ["jaar", "Datum"] if c in m.columns] + hap_cols
    st.dataframe(m[detail_cols].reset_index(drop=True), width="stretch", hide_index=True)
else:
    st.info("Geen hapdetails gevonden in de dataset.")

# -----------------------------
# Volledige records + download
# -----------------------------
with st.expander("Toon volledige record(s)"):
    st.dataframe(m.reset_index(drop=True), width="stretch", hide_index=True)

csv = m.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download geselecteerde meetpuntdata als CSV",
    data=csv,
    file_name=f"meetpunt_detail_{str(sel)}.csv",
    mime="text/csv",
)
