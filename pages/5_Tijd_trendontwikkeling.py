# pages/05_📈_Tijd_en_trendontwikkeling.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import load_data  # <-- alleen load_data importeren (geen slope_trend/trend_label)

st.set_page_config(page_title="Tijd & trend", layout="wide")
st.title("📈 Tijd en trendanalyse")


# -----------------------------
# Helpers: trendberekening
# -----------------------------
def slope_trend(dates, values):
    """
    Bepaal lineaire trend (slope) per jaar op basis van (datum, waarde).
    Retourneert float (slope/jaar) of None bij te weinig geldige punten.
    """
    d = pd.to_datetime(pd.Series(dates), errors="coerce")
    y = pd.to_numeric(pd.Series(values), errors="coerce")

    mask = d.notna() & y.notna()
    if mask.sum() < 2:
        return None

    d2 = d[mask]
    y2 = y[mask].astype(float)

    # x in jaren sinds eerste meetmoment (numeriek stabieler dan absolute ordinals)
    x_days = (d2 - d2.min()).dt.total_seconds() / (24 * 3600.0)
    x_years = x_days / 365.25

    # Bij degenerate situatie (alle datums gelijk) geen slope
    if np.isclose(x_years.max(), x_years.min()):
        return None

    slope, _intercept = np.polyfit(x_years.to_numpy(), y2.to_numpy(), 1)
    return float(slope)


def trend_label(slope, tol=0.0):
    """
    Geef label aan trend obv tolerance (slope/jaar).
    """
    if slope is None or not np.isfinite(slope):
        return "onbekend"
    if slope > tol:
        return "stijgend"
    if slope < -tol:
        return "dalend"
    return "stabiel"


# -----------------------------
# Data
# -----------------------------
DATA = load_data()
meas = DATA["measurements"].copy()  # gebruikt in originele pagina [1](https://rijkswaterstaat-my.sharepoint.com/personal/ben_bildirici_rws_nl/Documents/Microsoft%20Copilot%20Chat%20Files/5_Tijd_trendontwikkeling.py)

# Basis-validatie kolommen
required_cols = {"Datum", "Deelgebied", "Locatie"}
missing = required_cols - set(meas.columns)
if missing:
    st.error(f"Ontbrekende kolommen in measurements: {', '.join(sorted(missing))}")
    st.stop()

meas["Datum"] = pd.to_datetime(meas["Datum"], errors="coerce")
meas = meas[meas["Datum"].notna()].copy()


# -----------------------------
# Sidebar selectors
# -----------------------------
with st.sidebar:
    st.header("Selecties")

    sel_deelgebied = st.selectbox(
        "Deelgebied",
        options=["(alle)"] + sorted(meas["Deelgebied"].dropna().unique().tolist()),
    )

    all_loc = sorted(meas["Locatie"].dropna().unique().tolist())
    sel_locaties = st.multiselect("Meetpunten (Locatie)", options=all_loc, default=[])

    metric = st.selectbox(
        "Indicator",
        options=["biovol_totaal_ml", "biovol_driehoek_ml", "biovol_quagga_ml"],
        index=0,
    )

    tol = st.number_input("Trend tolerantie (slope/jaar)", value=0.0, step=0.1)
    show_points = st.checkbox("Toon punten", value=True)

# Check metric aanwezig
if metric not in meas.columns:
    st.error(f"Indicator-kolom '{metric}' ontbreekt in measurements.")
    st.stop()

# -----------------------------
# Apply filter
# -----------------------------
m = meas.copy()

if sel_deelgebied != "(alle)":
    m = m[m["Deelgebied"] == sel_deelgebied].copy()

if sel_locaties:
    m = m[m["Locatie"].isin(sel_locaties)].copy()

# Forceer numeriek voor gekozen metric
m[metric] = pd.to_numeric(m[metric], errors="coerce")

if m.empty:
    st.warning("Geen data na selectie.")
    st.stop()

# -----------------------------
# Trend lines (per locatie)
# -----------------------------
st.subheader("Trendlijnen / tijdreeks")
m = m.sort_values("Datum")

fig = px.line(
    m,
    x="Datum",
    y=metric,
    color="Locatie",
    markers=show_points,
    title=f"Tijdreeks: {metric}",
    labels={metric: metric, "Datum": "Datum"},
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Slope analyse per locatie
# -----------------------------
st.subheader("Slope-analyse (lineaire trend per jaar)")

rows = []
for loc, g in m.groupby("Locatie"):
    g = g.sort_values("Datum")
    s = slope_trend(g["Datum"], g[metric])
    rows.append(
        {
            "Locatie": loc,
            "Deelgebied": g["Deelgebied"].iloc[0] if len(g) else None,
            "n": int(len(g)),
            "slope_per_jaar": s,
            "trend": trend_label(s, tol),
            "min_datum": g["Datum"].min().date() if len(g) else None,
            "max_datum": g["Datum"].max().date() if len(g) else None,
        }
    )

trend_df = pd.DataFrame(rows).sort_values(["Deelgebied", "Locatie"])
st.dataframe(trend_df, use_container_width=True)

# -----------------------------
# Voor/na vergelijking
# -----------------------------
st.subheader("Voor/na vergelijking")
dates = sorted(m["Datum"].dt.date.unique().tolist())

if len(dates) < 2:
    st.info("Voor/na vergelijking vereist minimaal 2 meetmomenten.")
else:
    col1, col2 = st.columns(2)
    with col1:
        d1 = st.selectbox("Voor (datum)", options=dates, index=0)
    with col2:
        d2 = st.selectbox("Na (datum)", options=dates, index=len(dates) - 1)

    before = m[m["Datum"].dt.date == d1].set_index("Locatie")[metric]
    after = m[m["Datum"].dt.date == d2].set_index("Locatie")[metric]

    comp = pd.DataFrame({"voor": before, "na": after})
    comp["delta"] = comp["na"] - comp["voor"]
    comp = comp.reset_index()

    figc = px.bar(
        comp,
        x="Locatie",
        y="delta",
        title=f"Delta (na - voor) voor {metric}",
        labels={"delta": "Verschil", "Locatie": "Locatie"},
    )
    st.plotly_chart(figc, use_container_width=True)
    st.dataframe(comp, use_container_width=True)