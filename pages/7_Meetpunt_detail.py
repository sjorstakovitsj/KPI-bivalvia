# pages/07_🔍_Meetpunt_detail.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium

from utils import load_data

st.set_page_config(page_title="Meetpunt detail", layout="wide")
st.title("🔍 Deep-dive per meetlocatie")

DATA = load_data()
meas = DATA["measurements"].copy()
meas["Datum"] = pd.to_datetime(meas["Datum"])

locaties = sorted(meas["Locatie"].dropna().unique().tolist())
sel = st.selectbox("Kies meetlocatie", options=locaties)

m = meas[meas["Locatie"] == sel].sort_values("Datum").copy()
if m.empty:
    st.warning("Geen data voor deze locatie.")
    st.stop()

# Info block
col1, col2 = st.columns([1, 2])

with col1:
    center = [float(m["lat"].mean()), float(m["lon"].mean())]
    mp = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    for _, r in m.iterrows():
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=6,
            color="#1f77b4",
            fill=True,
            fill_opacity=0.8,
            tooltip=f"Locatie {int(r['Locatie'])} – {r['Datum'].date()}"
        ).add_to(mp)

    st_folium(mp, height=360, use_container_width=True)

with col2:
    st.subheader("Samenvatting")
    st.write({
        "Deelgebied": m["Deelgebied"].iloc[0],
        "Locatie": int(sel),
        "Aantal observaties": int(len(m)),
        "Periode": f"{m['Datum'].min().date()} — {m['Datum'].max().date()}",
        "Gem. diepte (m)": float(pd.to_numeric(m["diepte_m"], errors="coerce").mean()),
        "Totaal biovolume (ml)": float(pd.to_numeric(m["biovol_totaal_ml"], errors="coerce").sum()),
        "Opmerkingen (uniek)": sorted(m["Opmerkingen"].astype(str).unique().tolist())
    })

# Time series
st.subheader("Tijdreeks biovolume")
fig = px.line(
    m,
    x="Datum",
    y=["biovol_driehoek_ml", "biovol_quagga_ml", "biovol_totaal_ml"],
    markers=True,
    title="Biovolume door de tijd",
    labels={"value": "ml", "variable": "Reeks"}
)
st.plotly_chart(fig, use_container_width=True)

# Hapdetails
st.subheader("Sediment / lutum / PAS per hap")
hap_cols = [c for c in m.columns if c.startswith("sedimenttype_") or c.startswith("lutum_") or c.startswith("PAS_")]
if hap_cols:
    st.dataframe(m[["Datum"] + hap_cols].reset_index(drop=True))
else:
    st.info("Geen hapdetails gevonden in de dataset.")

# Raw record
with st.expander("Toon volledige record(s)"):
    st.dataframe(m.reset_index(drop=True))