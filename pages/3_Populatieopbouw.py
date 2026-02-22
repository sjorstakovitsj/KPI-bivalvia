# pages/03_👥_Populatieopbouw.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import load_data

st.set_page_config(page_title="Populatieopbouw", layout="wide")
st.title("👥 Populatieopbouw per deelgebied per soort")

DATA = load_data()
counts = DATA.get("populatie_counts")
perc = DATA.get("populatie_percent")

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

tab1, tab2 = st.tabs(["Aantallen", "Percentages"])

with tab1:
    _validate(counts, "populatie_counts")

    df = counts.copy()
    df = df.rename(columns={"lengteklasse_mm": "Lengteklasse", "deelgebied": "Deelgebied", "soort": "Soort", "waarde": "Aantal"})

    # Filters
    deel_opts = sorted(df["Deelgebied"].dropna().unique().tolist())
    soort_opts = sorted(df["Soort"].dropna().unique().tolist())

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_deel = st.multiselect("Deelgebieden", deel_opts, default=deel_opts)
    with col2:
        sel_soort = st.multiselect("Soorten", soort_opts, default=soort_opts)
    with col3:
        mode = st.selectbox("Weergave", ["Gestapeld", "Groepen"], index=0)

    view = df[df["Deelgebied"].isin(sel_deel) & df["Soort"].isin(sel_soort)].copy()
    view["Aantal"] = pd.to_numeric(view["Aantal"], errors="coerce")

    if view.empty:
        st.info("Geen data na selectie.")
    else:
        barmode = "stack" if mode == "Gestapeld" else "group"
        fig = px.bar(
            view,
            x="Lengteklasse",
            y="Aantal",
            color="Soort",
            facet_col="Deelgebied",
            facet_col_wrap=3,
            barmode=barmode,
            title="Aantallen per lengteklasse (per deelgebied en soort)",
            labels={"Lengteklasse": "Lengteklasse (mm)", "Aantal": "Aantal", "Soort": "Soort"}
        )
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Totaal per deelgebied en soort")
        tot = view.groupby(["Deelgebied", "Soort"], as_index=False)["Aantal"].sum()
        st.dataframe(tot, use_container_width=True)

with tab2:
    _validate(perc, "populatie_percent")

    df = perc.copy()
    df = df.rename(columns={"lengteklasse_mm": "Lengteklasse", "deelgebied": "Deelgebied", "soort": "Soort", "waarde": "Percentage"})

    # In bron is percentage vaak fractie (0..1). We zetten hier om naar %.
    df["Percentage"] = pd.to_numeric(df["Percentage"], errors="coerce")
    if df["Percentage"].dropna().max() <= 1.5:
        df["Percentage"] = df["Percentage"] * 100.0

    # Filters
    deel_opts = sorted(df["Deelgebied"].dropna().unique().tolist())
    soort_opts = sorted(df["Soort"].dropna().unique().tolist())

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_deel = st.multiselect("Deelgebieden", deel_opts, default=deel_opts, key="deel_perc")
    with col2:
        sel_soort = st.multiselect("Soorten", soort_opts, default=soort_opts, key="soort_perc")
    with col3:
        as_line = st.checkbox("Toon als lijnen", value=False)

    view = df[df["Deelgebied"].isin(sel_deel) & df["Soort"].isin(sel_soort)].copy()

    if view.empty:
        st.info("Geen data na selectie.")
    else:
        if as_line:
            fig = px.line(
                view,
                x="Lengteklasse",
                y="Percentage",
                color="Soort",
                facet_col="Deelgebied",
                facet_col_wrap=3,
                markers=True,
                title="Procentuele verdeling per lengteklasse",
                labels={"Lengteklasse": "Lengteklasse (mm)", "Percentage": "%", "Soort": "Soort"}
            )
        else:
            fig = px.area(
                view,
                x="Lengteklasse",
                y="Percentage",
                color="Soort",
                facet_col="Deelgebied",
                facet_col_wrap=3,
                title="Procentuele verdeling per lengteklasse",
                labels={"Lengteklasse": "Lengteklasse (mm)", "Percentage": "%", "Soort": "Soort"}
            )

        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Controle: som per deelgebied en soort (≈100%)")
        chk = view.groupby(["Deelgebied", "Soort"], as_index=False)["Percentage"].sum()
        st.dataframe(chk, use_container_width=True)