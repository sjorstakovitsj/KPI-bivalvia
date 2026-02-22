# Home.py
import streamlit as st
import pandas as pd
import numpy as np
from utils import load_data

st.set_page_config(page_title="Mosselkartering dashboard", layout="wide")
st.title("📊 Mosselkartering – overzicht")

DATA = load_data()
meas = DATA["measurements"].copy()

# -----------------------------
# Type-safety: maak numeriek waar nodig
# -----------------------------
for c in ["Locatie", "biovol_totaal_ml", "biovol_driehoek_ml", "biovol_quagga_ml", "diepte_m"]:
    if c in meas.columns:
        meas[c] = pd.to_numeric(meas[c], errors="coerce") if c != "Deelgebied" else meas[c]

# Altijd alle deelgebieden (geen filters)
m = meas

# -----------------------------
# 1) Eerst aggregeren naar unieke locaties (unit = locatie)
#    - Als er meerdere meetmomenten per locatie zijn: neem gemiddelde per locatie
# -----------------------------
per_loc = (
    m.groupby(["Deelgebied", "Locatie"], as_index=False)
     .agg(
         biovol_totaal_ml=("biovol_totaal_ml", "mean"),
         biovol_driehoek_ml=("biovol_driehoek_ml", "mean"),
         biovol_quagga_ml=("biovol_quagga_ml", "mean"),
         diepte_m=("diepte_m", "mean"),
     )
)

# Ratio per locatie (op basis van de per-locatie gemiddelden)
per_loc["ratio_driehoek_quagga"] = np.where(
    per_loc["biovol_quagga_ml"] > 0,
    per_loc["biovol_driehoek_ml"] / per_loc["biovol_quagga_ml"],
    np.nan
)

# -----------------------------
# KPI's (gemiddelden over unieke locaties)
# -----------------------------
colk1, colk2, colk3, colk4 = st.columns(4)

with colk1:
    st.metric("Meetpunten (unieke locaties)", f"{per_loc['Locatie'].nunique()}")

with colk2:
    avg_biovol = per_loc["biovol_totaal_ml"].mean()
    st.metric("Gemiddeld biovolume per locatie (ml)", f"{avg_biovol:.1f}" if pd.notna(avg_biovol) else "n.v.t.")

with colk3:
    avg_ratio = per_loc["ratio_driehoek_quagga"].mean()
    st.metric("Gem. verhouding driehoek/quagga (per locatie)", f"{avg_ratio:.2f}" if pd.notna(avg_ratio) else "n.v.t.")

with colk4:
    avg_diepte = per_loc["diepte_m"].mean()
    st.metric("Gem. diepte per locatie (m)", f"{avg_diepte:.2f}" if pd.notna(avg_diepte) else "n.v.t.")

# -----------------------------
# Tabel per deelgebied (ook gebaseerd op unieke locaties)
# -----------------------------
st.subheader("Per deelgebied (op basis van unieke locaties)")

agg = (
    per_loc.groupby("Deelgebied", as_index=False)
           .agg(
               meetpunten=("Locatie", "nunique"),
               gem_biovol_ml=("biovol_totaal_ml", "mean"),
               gem_ratio_driehoek_quagga=("ratio_driehoek_quagga", "mean"),
               gem_diepte_m=("diepte_m", "mean"),
           )
)

# Afronden voor leesbaarheid
show = agg.copy()
show["gem_biovol_ml"] = show["gem_biovol_ml"].round(2)
show["gem_ratio_driehoek_quagga"] = show["gem_ratio_driehoek_quagga"].round(3)
show["gem_diepte_m"] = show["gem_diepte_m"].round(2)

# Sorteer op gemiddeld biovolume
show = show.sort_values("gem_biovol_ml", ascending=False)

st.dataframe(show, use_container_width=True)

# Optioneel: toelichting
st.caption(
    "Alle KPI’s en de tabel zijn berekend op basis van unieke locaties: "
    "per locatie worden (eventuele) meerdere meetmomenten eerst gemiddeld, "
    "daarna wordt het gemiddelde over locaties genomen."
)