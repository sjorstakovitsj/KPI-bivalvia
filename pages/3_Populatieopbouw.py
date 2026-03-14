# pages/03_👥_Populatieopbouw.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import load_data, render_sidebar

st.set_page_config(page_title="Populatieopbouw", layout="wide")

# -----------------------------
# Gedeelde sidebar (zelfde op alle pagina's)
# -----------------------------
ui = render_sidebar(title="Mosselkartering")
years_sel = list(ui.get("years", []))
combine_years = bool(ui.get("combine_years", False))
keep_only_canonical = bool(ui.get("keep_only_canonical", False))

st.title("👥 Populatieopbouw per deelgebied per soort")

if not years_sel:
    st.warning("Selecteer in de sidebar ten minste één monitoringsjaar.")
    st.stop()


@st.cache_data(show_spinner=False)
def _load(years: tuple[str, ...], combine: bool, keep_only: bool) -> dict[str, pd.DataFrame]:
    return load_data(years=list(years), combine_years=combine, keep_only_canonical=keep_only)


DATA = _load(tuple(years_sel), combine_years, keep_only_canonical)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
REQUIRED_COLS = {"lengteklasse_mm", "deelgebied", "soort", "waarde", "metric"}


def _validate(df: pd.DataFrame, name: str) -> None:
    if df is None or df.empty:
        st.warning(f"Geen data gevonden voor: {name} (parquet ontbreekt of is leeg).")
        st.stop()
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        st.error(
            f"Populatie dataset '{name}' heeft niet de verwachte tidy structuur.\n\n"
            f"Ontbrekende kolommen: {sorted(missing)}\n"
            f"Gevonden kolommen: {list(df.columns)}\n\n"
            "Tip: run `python dataprep.py` om tidy parquets te genereren."
        )
        st.stop()


def _get_table(data: dict[str, pd.DataFrame], base_key: str, years: list[str]) -> pd.DataFrame:
    """Haal een tabel op uit load_data(), robuust voor multi-year.

    - Als combine_years=True: verwacht key base_key met kolom 'jaar'
    - Als combine_years=False en meerdere jaren: verwacht keys base_key_<jaar>

    Let op: De jaarselectie gebeurt via de gedeelde sidebar; op deze pagina tonen we geen extra jaarfilter.
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


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
counts_raw = _get_table(DATA, "populatie_counts", years_sel)
perc_raw = _get_table(DATA, "populatie_percent", years_sel)

# Tabs

tab1, tab2 = st.tabs(["Aantallen", "Percentages"])

with tab1:
    _validate(counts_raw, "populatie_counts")
    df = counts_raw.copy()

    df = df.rename(
        columns={
            "lengteklasse_mm": "Lengteklasse",
            "deelgebied": "Deelgebied",
            "soort": "Soort",
            "waarde": "Aantal",
            "jaar": "Jaar",
        }
    )

    # Filters (geen jaarfilter hier; dat zit in de gedeelde sidebar)
    deel_opts = sorted(df["Deelgebied"].dropna().astype(str).unique().tolist())
    soort_opts = sorted(df["Soort"].dropna().astype(str).unique().tolist())

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_deel = st.multiselect("Deelgebieden", deel_opts, default=deel_opts)
    with col2:
        sel_soort = st.multiselect("Soorten", soort_opts, default=soort_opts)
    with col3:
        mode = st.selectbox("Weergave", ["Gestapeld", "Groepen"], index=0)

    view = df[df["Deelgebied"].isin(sel_deel) & df["Soort"].isin(sel_soort)].copy()

    view["Aantal"] = pd.to_numeric(view["Aantal"], errors="coerce")
    view["Lengteklasse"] = pd.to_numeric(view["Lengteklasse"], errors="coerce")

    if view.empty:
        st.info("Geen data na selectie.")
    else:
        barmode = "stack" if mode == "Gestapeld" else "group"

        # Facets:
        # - Altijd per Deelgebied
        # - Als meerdere jaren via sidebar: facet ook per Jaar
        facet_cols = ["Deelgebied"]
        if "Jaar" in view.columns and view["Jaar"].nunique() > 1:
            facet_cols = ["Jaar", "Deelgebied"]

        fig = px.bar(
            view,
            x="Lengteklasse",
            y="Aantal",
            color="Soort",
            facet_col=facet_cols[-1],
            facet_row=facet_cols[0] if len(facet_cols) == 2 else None,
            barmode=barmode,
            title="Aantallen per lengteklasse (per deelgebied en soort)",
            labels={"Lengteklasse": "Lengteklasse (mm)", "Aantal": "Aantal", "Soort": "Soort"},
        )
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Totaal per deelgebied en soort")
        group_cols = ["Deelgebied", "Soort"]
        if "Jaar" in view.columns and view["Jaar"].nunique() > 1:
            group_cols = ["Jaar"] + group_cols
        tot = view.groupby(group_cols, as_index=False)["Aantal"].sum()
        st.dataframe(tot, use_container_width=True)

with tab2:
    _validate(perc_raw, "populatie_percent")
    df = perc_raw.copy()

    df = df.rename(
        columns={
            "lengteklasse_mm": "Lengteklasse",
            "deelgebied": "Deelgebied",
            "soort": "Soort",
            "waarde": "Percentage",
            "jaar": "Jaar",
        }
    )

    df["Percentage"] = pd.to_numeric(df["Percentage"], errors="coerce")
    if df["Percentage"].dropna().max() <= 1.5:
        df["Percentage"] = df["Percentage"] * 100.0

    # Filters (geen jaarfilter hier; dat zit in de gedeelde sidebar)
    deel_opts = sorted(df["Deelgebied"].dropna().astype(str).unique().tolist())
    soort_opts = sorted(df["Soort"].dropna().astype(str).unique().tolist())

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_deel = st.multiselect("Deelgebieden", deel_opts, default=deel_opts, key="deel_perc")
    with col2:
        sel_soort = st.multiselect("Soorten", soort_opts, default=soort_opts, key="soort_perc")
    with col3:
        as_line = st.checkbox("Toon als lijnen", value=False)

    view = df[df["Deelgebied"].isin(sel_deel) & df["Soort"].isin(sel_soort)].copy()
    view["Lengteklasse"] = pd.to_numeric(view["Lengteklasse"], errors="coerce")

    if view.empty:
        st.info("Geen data na selectie.")
    else:
        facet_cols = ["Deelgebied"]
        if "Jaar" in view.columns and view["Jaar"].nunique() > 1:
            facet_cols = ["Jaar", "Deelgebied"]

        if as_line:
            fig = px.line(
                view,
                x="Lengteklasse",
                y="Percentage",
                color="Soort",
                facet_col=facet_cols[-1],
                facet_row=facet_cols[0] if len(facet_cols) == 2 else None,
                markers=True,
                title="Procentuele verdeling per lengteklasse",
                labels={"Lengteklasse": "Lengteklasse (mm)", "Percentage": "%", "Soort": "Soort"},
            )
        else:
            fig = px.area(
                view,
                x="Lengteklasse",
                y="Percentage",
                color="Soort",
                facet_col=facet_cols[-1],
                facet_row=facet_cols[0] if len(facet_cols) == 2 else None,
                title="Procentuele verdeling per lengteklasse",
                labels={"Lengteklasse": "Lengteklasse (mm)", "Percentage": "%", "Soort": "Soort"},
            )

        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Controle: som per deelgebied en soort (≈100%)")
        group_cols = ["Deelgebied", "Soort"]
        if "Jaar" in view.columns and view["Jaar"].nunique() > 1:
            group_cols = ["Jaar"] + group_cols
        chk = view.groupby(group_cols, as_index=False)["Percentage"].sum()
        st.dataframe(chk, use_container_width=True)
