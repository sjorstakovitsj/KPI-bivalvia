# pages/04_⚖️_ADV.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import load_data

st.set_page_config(page_title="ADV", layout="wide")
st.title("⚖️ Asvrij drooggewicht (ADV) – Bijlage 6 (ADV/mossel, N, SL)")

DATA = load_data()
adv_len = DATA.get("adv_lenclass")  # moet nu bestaan vanuit dataprep.py

if adv_len is None or adv_len.empty:
    st.warning(
        "Geen ADV lengteklasse data gevonden (adv_lenclass.parquet ontbreekt of is leeg).\n\n"
        "Run eerst: `python dataprep.py` zodat Bijlage 6 wordt omgezet naar adv_lenclass.parquet."
    )
    st.stop()

# Type safety
df = adv_len.copy()
for c in ["sl_mm", "adv_mg_per_mossel", "n_verast"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df["deelgebied"] = df["deelgebied"].astype("string")
df["soort"] = df["soort"].astype("string")

deel_opts = sorted(df["deelgebied"].dropna().unique().tolist())
soort_opts = sorted(df["soort"].dropna().unique().tolist())

# =========================
# 1) GEDEELDE FILTERS (voor alles)
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
        key="min_n_filter"
    )

# Data selectie op basis van gedeelde filters
view = df[df["deelgebied"].isin(sel_deel) & df["soort"].isin(sel_soort)].copy()

if min_n > 0:
    view = view[(view["n_verast"].fillna(0) >= min_n)].copy()

# Voor grafieken: vereis beide kolommen aanwezig
view_plot = view.dropna(subset=["sl_mm", "adv_mg_per_mossel"]).copy()

st.divider()

# =========================
# 2) VISUALISATIES
# =========================
st.subheader("ADV/mossel per schelplengte (SL)")

if view_plot.empty:
    st.info("Geen data na selectie/filters (voor visualisaties).")
else:
    if view_mode == "Lijn (ADV vs SL)":
        fig = px.line(
            view_plot.sort_values("sl_mm"),
            x="sl_mm",
            y="adv_mg_per_mossel",
            color="soort",
            line_group="deelgebied",
            facet_col="deelgebied",
            facet_col_wrap=3,
            markers=True,
            labels={
                "sl_mm": "SL (mm)",
                "adv_mg_per_mossel": "ADV/mossel (mg)",
                "soort": "Soort",
            },
            title="ADV/mossel vs SL (per deelgebied en soort)",
        )
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)

    elif view_mode == "Boxplot":
        fig = px.box(
            view_plot,
            x="deelgebied",
            y="adv_mg_per_mossel",
            color="soort",
            points="all",
            labels={
                "deelgebied": "Deelgebied",
                "adv_mg_per_mossel": "ADV/mossel (mg)",
                "soort": "Soort",
            },
            title="Verdeling ADV/mossel per deelgebied (boxplot)",
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# =========================
# 3) TABEL & KWALITEITSCHECK (N)
#    (voorheen tab 2, nu gekoppeld aan dezelfde filters)
# =========================
st.subheader("Tabel (ADV/mossel, N, SL)")

# Voor de tabel: we tonen ook regels zonder adv/sl als je die wilt zien.
# In de oorspronkelijke tab2-code werd niet gedropt op NaN; dat houden we zo.
view_table = view.sort_values(["deelgebied", "soort", "sl_mm"])

st.dataframe(view_table, use_container_width=True)