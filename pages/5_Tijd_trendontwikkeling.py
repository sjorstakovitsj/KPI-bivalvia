# pages/05_📈_Tijd_en_trendontwikkeling.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import load_data, render_sidebar

st.set_page_config(page_title="Tijd & trend", layout="wide")

# -----------------------------
# Gedeelde sidebar (zelfde patroon als 4_ADV.py)
# -----------------------------
ui = render_sidebar(title="Mosselkartering")
years_sel = list(ui.get("years", []))
combine_years = bool(ui.get("combine_years", False))

st.title("📈 Tijd en trendanalyse")

if not years_sel:
    st.warning("Selecteer in de sidebar ten minste één monitoringsjaar.")
    st.stop()


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
# Data laden (zelfde patroon als 4_ADV.py)
# -----------------------------
@st.cache_data(show_spinner=False)
def _load(years: tuple[str, ...], combine: bool) -> dict[str, pd.DataFrame]:
    return load_data(
        years=list(years),
        combine_years=combine,
    )


def _get_table(data: dict[str, pd.DataFrame], base_key: str, years: list[str]) -> pd.DataFrame:
    """
    Haal een tabel op uit load_data(), robuust voor multi-year.
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

    # map op lowercase/gestripte namen -> gewenste naam
    col_map = {}
    for c in out.columns:
        c_norm = str(c).strip().lower()

        if c_norm in {"datum", "date", "datum_tijd", "datumtijd"}:
            col_map[c] = "Datum"
        elif c_norm in {"deelgebied", "deel_gebied"}:
            col_map[c] = "Deelgebied"
        elif c_norm in {"locatie", "meetpunt", "meetpunten"}:
            col_map[c] = "Locatie"
        elif c_norm in {"biovol_totaal_ml", "biovol totaal ml", "biovol_totaal"}:
            col_map[c] = "biovol_totaal_ml"
        elif c_norm in {"biovol_driehoek_ml", "biovol driehoek ml", "biovol_driehoek"}:
            col_map[c] = "biovol_driehoek_ml"
        elif c_norm in {"biovol_quagga_ml", "biovol quagga ml", "biovol_quagga"}:
            col_map[c] = "biovol_quagga_ml"
        elif c_norm in {"jaar", "year"}:
            col_map[c] = "jaar"

    out = out.rename(columns=col_map)
    return out


def _prepare_measurements(meas: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Normaliseer en aggregeer measurements voor veilige visualisatie/analyse.
    """
    m = meas.copy()

    # Datum robuust naar datetime
    m["Datum"] = pd.to_datetime(m["Datum"], errors="coerce")
    m = m[m["Datum"].notna()].copy()

    # Gekozen metric naar numeric
    m[metric] = pd.to_numeric(m[metric], errors="coerce")

    # Alleen relevante kolommen
    base_cols = ["Datum", "Deelgebied", "Locatie", metric]
    extra_cols = ["jaar"] if "jaar" in m.columns else []
    cols = [c for c in base_cols + extra_cols if c in m.columns]
    m = m[cols].copy()

    # Lege locaties eruit
    m = m[m["Locatie"].notna()].copy()

    # Aggregeren per datum / deelgebied / locatie (/ jaar indien aanwezig)
    group_cols = ["Datum", "Deelgebied", "Locatie"]
    if "jaar" in m.columns:
        group_cols = ["jaar"] + group_cols

    m = (
        m.groupby(group_cols, dropna=False, as_index=False)[metric]
        .sum(min_count=1)
        .sort_values(group_cols)
    )

    return m


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

metric_options = [c for c in ["biovol_totaal_ml", "biovol_driehoek_ml", "biovol_quagga_ml"] if c in meas.columns]
if not metric_options:
    st.error(
        "Geen van de verwachte biovolume-kolommen gevonden in measurements.\n\n"
        f"Gevonden kolommen: {list(meas.columns)}"
    )
    st.stop()

# -----------------------------
# Paginafilters (onder gedeelde sidebar)
# -----------------------------
with st.sidebar:
    st.markdown("---")
    st.subheader("📌 Paginafilters")

    deelgebieden = sorted(meas["Deelgebied"].dropna().astype(str).unique().tolist())
    sel_deelgebied = st.selectbox(
        "Deelgebied",
        options=["(alle)"] + deelgebieden,
        key="tijdtrend_deelgebied",
    )

    if sel_deelgebied != "(alle)":
        loc_base = meas.loc[meas["Deelgebied"] == sel_deelgebied, "Locatie"]
    else:
        loc_base = meas["Locatie"]

    locatie_opts = sorted(loc_base.dropna().astype(str).unique().tolist())
    sel_locaties = st.multiselect(
        "Meetpunten (Locatie)",
        options=locatie_opts,
        default=[],
        key="tijdtrend_locaties",
    )

    metric = st.selectbox(
        "Indicator",
        options=metric_options,
        index=0,
        key="tijdtrend_metric",
    )

    tol = st.number_input(
        "Trend tolerantie (slope/jaar)",
        value=0.0,
        step=0.1,
        key="tijdtrend_tol",
    )

    show_points = st.checkbox(
        "Toon punten",
        value=True,
        key="tijdtrend_show_points",
    )

# -----------------------------
# Data voorbereiden en filteren
# -----------------------------
m = _prepare_measurements(meas, metric)

if sel_deelgebied != "(alle)":
    m = m[m["Deelgebied"] == sel_deelgebied].copy()

if sel_locaties:
    m = m[m["Locatie"].astype(str).isin(sel_locaties)].copy()

if m.empty:
    st.warning("Geen data na selectie.")
    st.stop()

# -----------------------------
# Samenvatting
# -----------------------------
st.subheader("Samenvatting")

n_loc = int(m["Locatie"].nunique())
n_dg = int(m["Deelgebied"].nunique())
n_rows = int(len(m))
date_min = m["Datum"].min()
date_max = m["Datum"].max()

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Aantal meetpunten", n_loc)
col_b.metric("Aantal deelgebieden", n_dg)
col_c.metric("Aantal meetmomenten", n_rows)
col_d.metric(
    "Periode",
    f"{date_min.date().isoformat()} → {date_max.date().isoformat()}"
    if pd.notna(date_min) and pd.notna(date_max)
    else "—",
)

# -----------------------------
# Trendlijnen / tijdreeks
# -----------------------------
st.subheader("Trendlijnen / tijdreeks")

facet_by_year = "jaar" in m.columns and m["jaar"].nunique() > 1

fig = px.line(
    m.sort_values(["jaar", "Datum"] if "jaar" in m.columns else ["Datum"]),
    x="Datum",
    y=metric,
    color="Locatie",
    facet_row="jaar" if facet_by_year else None,
    markers=show_points,
    title=f"Tijdreeks: {metric}",
    labels={metric: metric, "Datum": "Datum", "jaar": "Jaar"},
)

fig.update_layout(
    legend_title_text="Locatie",
    hovermode="x unified",
    margin=dict(l=10, r=10, t=60, b=10),
)

st.plotly_chart(fig, width="stretch")

# -----------------------------
# Slope analyse per locatie
# -----------------------------
st.subheader("Slope-analyse (lineaire trend per jaar)")

trend_rows = []

group_cols = ["Locatie"]
if "jaar" in m.columns and facet_by_year:
    group_cols = ["jaar", "Locatie"]

for keys, g in m.groupby(group_cols, dropna=False):
    g = g.sort_values("Datum")
    s = slope_trend(g["Datum"], g[metric])

    row = {
        "Locatie": g["Locatie"].iloc[0] if len(g) else None,
        "Deelgebied": g["Deelgebied"].iloc[0] if len(g) else None,
        "n": int(len(g)),
        "slope_per_jaar": s,
        "trend": trend_label(s, tol),
        "min_datum": g["Datum"].min().date() if len(g) else None,
        "max_datum": g["Datum"].max().date() if len(g) else None,
    }
    if "jaar" in g.columns and facet_by_year:
        row["jaar"] = g["jaar"].iloc[0]

    trend_rows.append(row)

trend_df = pd.DataFrame(trend_rows)

if trend_df.empty:
    st.info("Geen trendanalyse beschikbaar voor de huidige selectie.")
else:
    sort_cols = [c for c in ["jaar", "Deelgebied", "Locatie"] if c in trend_df.columns]
    trend_df = trend_df.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    trend_counts = trend_df["trend"].value_counts(dropna=False).to_dict()
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric("Stijgend", int(trend_counts.get("stijgend", 0)))
    tc2.metric("Dalend", int(trend_counts.get("dalend", 0)))
    tc3.metric("Stabiel", int(trend_counts.get("stabiel", 0)))
    tc4.metric("Onbekend", int(trend_counts.get("onbekend", 0)))

    st.dataframe(trend_df, width="stretch", hide_index=True)

# -----------------------------
# Voor/na vergelijking
# -----------------------------
st.subheader("Voor/na vergelijking")

available_dates = sorted(m["Datum"].dt.date.unique().tolist())

if len(available_dates) < 2:
    st.info("Voor/na vergelijking vereist minimaal 2 meetmomenten.")
else:
    col1, col2 = st.columns(2)

    with col1:
        d1 = st.selectbox(
            "Voor (datum)",
            options=available_dates,
            index=0,
            key="tijdtrend_before_date",
        )

    with col2:
        d2 = st.selectbox(
            "Na (datum)",
            options=available_dates,
            index=len(available_dates) - 1,
            key="tijdtrend_after_date",
        )

    if d1 == d2:
        st.warning("Kies twee verschillende datums voor de voor/na vergelijking.")
    else:
        group_cols = ["Locatie", "Deelgebied"]
        if "jaar" in m.columns and facet_by_year:
            group_cols = ["jaar"] + group_cols

        before = (
            m.loc[m["Datum"].dt.date == d1]
            .groupby(group_cols, dropna=False, as_index=False)[metric]
            .sum(min_count=1)
            .rename(columns={metric: "voor"})
        )

        after = (
            m.loc[m["Datum"].dt.date == d2]
            .groupby(group_cols, dropna=False, as_index=False)[metric]
            .sum(min_count=1)
            .rename(columns={metric: "na"})
        )

        comp = pd.merge(before, after, on=group_cols, how="outer")

        comp["voor"] = pd.to_numeric(comp["voor"], errors="coerce")
        comp["na"] = pd.to_numeric(comp["na"], errors="coerce")
        comp["delta"] = comp["na"] - comp["voor"]

        sort_cols = [c for c in ["jaar", "Deelgebied", "Locatie"] if c in comp.columns]
        comp = comp.sort_values(sort_cols, na_position="last").reset_index(drop=True)

        figc = px.bar(
            comp,
            x="Locatie",
            y="delta",
            color="Deelgebied",
            facet_row="jaar" if ("jaar" in comp.columns and facet_by_year) else None,
            title=f"Delta (na - voor) voor {metric}",
            labels={"delta": "Verschil", "Locatie": "Locatie", "jaar": "Jaar"},
        )

        figc.update_layout(
            legend_title_text="Deelgebied",
            margin=dict(l=10, r=10, t=60, b=10),
        )

        st.plotly_chart(figc, width="stretch")
        st.dataframe(comp, width="stretch", hide_index=True)

        csv = comp.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download voor/na vergelijking als CSV",
            data=csv,
            file_name=f"voor_na_{metric}_{d1}_{d2}.csv",
            mime="text/csv",
        )
