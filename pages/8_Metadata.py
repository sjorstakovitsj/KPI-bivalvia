# pages/08_✅_Datakwaliteit_en_metadata.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import load_data

st.set_page_config(page_title="Datakwaliteit & metadata", layout="wide")
st.title("✅ Datakwaliteit en metadata")

DATA = load_data()
meas = DATA["measurements"].copy()
meas["Datum"] = pd.to_datetime(meas["Datum"], errors="coerce")

# -----------------------
# Observaties per jaar
# -----------------------
st.subheader("Aantal waarnemingen per jaar")
meas["jaar"] = meas["Datum"].dt.year
counts_year = meas.groupby("jaar").size().reset_index(name="n").sort_values("jaar")
fig = px.bar(counts_year, x="jaar", y="n", labels={"jaar": "Jaar", "n": "Aantal"})
st.plotly_chart(fig, use_container_width=True)

# per deelgebied
st.subheader("Aantal waarnemingen per jaar per deelgebied")
counts_year_area = meas.groupby(["jaar", "Deelgebied"]).size().reset_index(name="n")
fig2 = px.bar(
    counts_year_area,
    x="jaar",
    y="n",
    color="Deelgebied",
    barmode="stack",
    labels={"jaar": "Jaar", "n": "Aantal"}
)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------
# Meetgaten (placeholder)
# -----------------------
st.subheader("Meetgaten (locaties zonder waarneming in een jaar)")
years = sorted(meas["jaar"].dropna().unique().tolist())
if len(years) < 2:
    st.info("Meetgaten-analyse wordt pas zinvol bij meerdere jaren data.")
else:
    # heatmap: locatie x jaar (1=gemeten, 0=gat)
    pivot = (
        meas.assign(waarde=1)
        .pivot_table(index="Locatie", columns="jaar", values="waarde", aggfunc="max", fill_value=0)
    )
    st.dataframe(pivot)

# -----------------------
# Consistentie codes (sediment / PAS)
# -----------------------
st.subheader("Consistentie van codes (sedimenttype / PAS)")

sed_cols = [c for c in meas.columns if c.startswith("sedimenttype_")]
pas_cols = [c for c in meas.columns if c.startswith("PAS_")]
lut_cols = [c for c in meas.columns if c.startswith("lutum_")]

col1, col2, col3 = st.columns(3)

with col1:
    if sed_cols:
        sed_vals = pd.Series(pd.unique(meas[sed_cols].astype(str).values.ravel("K")))
        sed_vals = sed_vals[~sed_vals.isin(["nan", "None", "-", ""])]
        st.write("Sedimenttype unieke waarden:")
        st.write(sorted(sed_vals.tolist()))
    else:
        st.info("Geen sedimenttype-kolommen gevonden.")

with col2:
    if pas_cols:
        pas_vals = pd.Series(pd.unique(meas[pas_cols].astype(str).values.ravel("K")))
        pas_vals = pas_vals[~pas_vals.isin(["nan", "None", "-", ""])]
        st.write("PAS unieke waarden:")
        st.write(sorted(pas_vals.tolist()))
    else:
        st.info("Geen PAS-kolommen gevonden.")

with col3:
    if lut_cols:
        lut_vals = pd.Series(pd.unique(meas[lut_cols].astype(str).values.ravel("K")))
        lut_vals = lut_vals[~lut_vals.isin(["nan", "None", "-", ""])]
        st.write("Lutum-klassen unieke waarden:")
        st.write(sorted(lut_vals.tolist()))
    else:
        st.info("Geen lutum-kolommen gevonden.")

# -----------------------
# Ruimtelijke dekking
# -----------------------
st.subheader("Ruimtelijke dekking")
st.map(meas[["lat", "lon"]].dropna())

# -----------------------
# Coördinaat-verschil gepland vs uitgevoerd (RD)
# -----------------------
st.subheader("Planned vs. uitgevoerd (RD) – afwijking (m)")

if {"x_planned_rd", "y_planned_rd", "x_rd", "y_rd"}.issubset(set(meas.columns)):
    d = meas.copy()
    for c in ["x_planned_rd", "y_planned_rd", "x_rd", "y_rd"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["delta_m"] = np.sqrt((d["x_rd"] - d["x_planned_rd"]) ** 2 + (d["y_rd"] - d["y_planned_rd"]) ** 2)
    figd = px.histogram(d.dropna(subset=["delta_m"]), x="delta_m", nbins=30, title="Verdeling afwijking (m)")
    st.plotly_chart(figd, use_container_width=True)

    st.write("Top 10 grootste afwijkingen:")
    top = d.sort_values("delta_m", ascending=False).head(10)[
        ["Deelgebied", "Locatie", "Datum", "delta_m", "Opmerkingen"]
    ]
    st.dataframe(top)
else:
    st.info("Planned/uitgevoerd RD-coördinaten zijn niet compleet in de dataset.")

# -----------------------
# Completeness / missing values
# -----------------------
st.subheader("Completeness (missende waarden per kolom)")
miss = meas.isna().mean().sort_values(ascending=False).reset_index()
miss.columns = ["kolom", "missing_fractie"]
miss["missing_%"] = (miss["missing_fractie"] * 100).round(1)

figm = px.bar(miss.head(25), x="missing_%", y="kolom", orientation="h", title="Top missende kolommen (max 25)")
st.plotly_chart(figm, use_container_width=True)

with st.expander("Volledige missing-tabel"):
    st.dataframe(miss)