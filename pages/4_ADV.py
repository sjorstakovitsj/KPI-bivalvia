# pages/04_⚖️_ADV.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import load_data, render_sidebar

st.set_page_config(page_title="ADV", layout="wide")

# -----------------------------
# Gedeelde sidebar (zelfde op alle pagina's)
# -----------------------------
ui = render_sidebar(title="Mosselkartering")
years_sel = list(ui.get("years", []))
combine_years = bool(ui.get("combine_years", False))
keep_only_canonical = bool(ui.get("keep_only_canonical", False))

st.title("⚖️ Asvrij drooggewicht (ADV) – Bijlage 6 (ADV/mossel, N, SL)")

if not years_sel:
    st.warning("Selecteer in de sidebar ten minste één monitoringsjaar.")
    st.stop()


@st.cache_data(show_spinner=False)
def _load(years: tuple[str, ...], combine: bool, keep_only: bool) -> dict[str, pd.DataFrame]:
    return load_data(years=list(years), combine_years=combine, keep_only_canonical=keep_only)


def _get_table(data: dict[str, pd.DataFrame], base_key: str, years: list[str]) -> pd.DataFrame:
    """Haal een tabel op uit load_data(), robuust voor multi-year.

    - Als combine_years=True: verwacht key base_key met kolom 'jaar'
    - Als combine_years=False en meerdere jaren: verwacht keys base_key_<jaar>

    Let op: jaarselectie gebeurt via gedeelde sidebar; hier geen extra jaarfilter.
    """
    if base_key in data and isinstance(data[base_key], pd.DataFrame) and not data[base_key].empty:
        df = data[base_key].copy()
        if "jaar" not in df.columns and len(years) == 1:
            df["jaar"] = years[0]
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


DATA = _load(tuple(years_sel), combine_years, keep_only_canonical)
adv_len = _get_table(DATA, "adv_lenclass", years_sel)

if adv_len is None or adv_len.empty:
    st.warning(
        "Geen ADV lengteklasse data gevonden (adv_lenclass.parquet ontbreekt of is leeg).\n\n"
        "Run eerst: `python dataprep.py` zodat Bijlage 6 wordt omgezet naar adv_lenclass.parquet."
    )
    st.stop()

# -----------------------------
# Type safety
# -----------------------------
df = adv_len.copy()
for c in ["sl_mm", "adv_mg_per_mossel", "n_verast"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Zorg dat deze kolommen bestaan
if "deelgebied" in df.columns:
    df["deelgebied"] = df["deelgebied"].astype("string")
if "soort" in df.columns:
    df["soort"] = df["soort"].astype("string")
if "jaar" in df.columns:
    df["jaar"] = df["jaar"].astype("string")

# Opties voor filters
if "deelgebied" not in df.columns or "soort" not in df.columns:
    st.error(
        "ADV dataset mist verplichte kolommen ('deelgebied' en/of 'soort'). "
        f"Gevonden kolommen: {list(df.columns)}"
    )
    st.stop()


deel_opts = sorted(df["deelgebied"].dropna().unique().tolist())
soort_opts = sorted(df["soort"].dropna().unique().tolist())

# =========================
# 1) FILTERS (voor alles)
# =========================
st.subheader("Filters")
fcol1, fcol2, fcol3, fcol4 = st.columns(4)

with fcol1:
    sel_deel = st.multiselect("Deelgebied", options=deel_opts, default=deel_opts, key="deel_filter")
with fcol2:
    sel_soort = st.multiselect("Soort", options=soort_opts, default=soort_opts, key="soort_filter")
with fcol3:
    view_mode = st.selectbox("Visualisatie", ["Lijn (ADV vs SL)", "Boxplot"], index=0, key="viz_mode")
with fcol4:
    min_n = st.number_input(
        "Minimum aantal veraste individuen",
        value=10,
        step=1,
        key="min_n_filter",
    )

# Data selectie op basis van filters
view = df[df["deelgebied"].isin(sel_deel) & df["soort"].isin(sel_soort)].copy()
if min_n > 0 and "n_verast" in view.columns:
    view = view[(view["n_verast"].fillna(0) >= min_n)].copy()

# Voor grafieken: vereis beide kolommen aanwezig
need_cols = [c for c in ["sl_mm", "adv_mg_per_mossel"] if c in view.columns]
view_plot = view.dropna(subset=need_cols).copy() if need_cols else pd.DataFrame()

st.divider()

# =========================
# 2) VISUALISATIES
# =========================
st.subheader("ADV/mossel per schelplengte (SL)")

if view_plot.empty:
    st.info("Geen data na selectie/filters (voor visualisaties).")
else:
    facet_by_year = "jaar" in view_plot.columns and view_plot["jaar"].nunique() > 1

    if view_mode == "Lijn (ADV vs SL)":
        fig = px.line(
            view_plot.sort_values(["jaar", "sl_mm"] if "jaar" in view_plot.columns else ["sl_mm"]),
            x="sl_mm",
            y="adv_mg_per_mossel",
            color="soort",
            line_group="deelgebied",
            facet_col="deelgebied",
            facet_col_wrap=3,
            facet_row="jaar" if facet_by_year else None,
            markers=True,
            labels={
                "sl_mm": "SL (mm)",
                "adv_mg_per_mossel": "ADV/mossel (mg)",
                "soort": "Soort",
                "deelgebied": "Deelgebied",
                "jaar": "Jaar",
            },
            title="ADV/mossel vs SL (per deelgebied en soort)",
        )
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

    elif view_mode == "Boxplot":
        fig = px.box(
            view_plot,
            x="deelgebied",
            y="adv_mg_per_mossel",
            color="soort",
            facet_row="jaar" if facet_by_year else None,
            points="all",
            labels={
                "deelgebied": "Deelgebied",
                "adv_mg_per_mossel": "ADV/mossel (mg)",
                "soort": "Soort",
                "jaar": "Jaar",
            },
            title="Verdeling ADV/mossel per deelgebied (boxplot)",
        )
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# =========================
# 3) TABEL
# =========================
st.subheader("Tabel (ADV/mossel, N, SL)")

sort_cols = [c for c in ["jaar", "deelgebied", "soort", "sl_mm"] if c in view.columns]
view_table = view.sort_values(sort_cols) if sort_cols else view
st.dataframe(view_table, use_container_width=True)
