# pages/06_🔗_Relaties_en_clusteranalyses.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from utils import load_data, avg_lutum_percentage, render_sidebar

st.set_page_config(page_title="Relaties & PCA", layout="wide")

# -----------------------------
# Gedeelde sidebar (zelfde patroon als 4_ADV.py)
# -----------------------------
ui = render_sidebar(title="Mosselkartering")
years_sel = list(ui.get("years", []))
combine_years = bool(ui.get("combine_years", False))
keep_only_canonical = bool(ui.get("keep_only_canonical", False))

st.title("🔗 Relaties en verklaringen")

if not years_sel:
    st.warning("Selecteer in de sidebar ten minste één monitoringsjaar.")
    st.stop()

# ---------------------------------------------------------
# Optioneel: check of statsmodels beschikbaar is
# Plotly trendline='ols' vereist statsmodels.
# ---------------------------------------------------------
HAS_STATSMODELS = True
try:
    import statsmodels.api as sm  # noqa: F401
except Exception:
    HAS_STATSMODELS = False


# ---------------------------------------------------------
# Data helpers
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load(years: tuple[str, ...], combine: bool, keep_only: bool) -> dict[str, pd.DataFrame]:
    return load_data(
        years=list(years),
        combine_years=combine,
        keep_only_canonical=keep_only,
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
        elif c_norm in {"diepte_m", "diepte", "waterdiepte", "waterdiepte_m"}:
            col_map[c] = "diepte_m"
        elif c_norm in {"biovol_totaal_ml", "biovol totaal ml", "biovol_totaal"}:
            col_map[c] = "biovol_totaal_ml"
        elif c_norm in {"biovol_driehoek_ml", "biovol driehoek ml", "biovol_driehoek"}:
            col_map[c] = "biovol_driehoek_ml"
        elif c_norm in {"biovol_quagga_ml", "biovol quagga ml", "biovol_quagga"}:
            col_map[c] = "biovol_quagga_ml"
        elif c_norm in {"jaar", "year"}:
            col_map[c] = "jaar"
        elif c_norm.startswith("lutum_"):
            col_map[c] = str(c).strip()
        elif c_norm.startswith("sedimenttype_"):
            col_map[c] = str(c).strip()

    out = out.rename(columns=col_map)
    return out


# ---------------------------------------------------------
# Helper: voeg een simpele lineaire fit-lijn toe (zonder statsmodels)
# ---------------------------------------------------------
def add_numpy_trendline(fig, df, xcol, ycol, color="#111111", name="Lineaire fit (numpy)"):
    """
    Voeg een lineaire regressielijn toe aan een bestaande plotly figuur
    op basis van numpy.polyfit.
    """
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return fig

    x2 = x[mask].to_numpy(dtype=float)
    y2 = y[mask].to_numpy(dtype=float)

    a, b = np.polyfit(x2, y2, 1)
    x_line = np.linspace(np.min(x2), np.max(x2), 50)
    y_line = a * x_line + b

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            line=dict(color=color, width=3),
            name=name,
            hoverinfo="skip",
        )
    )
    return fig


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------
DATA = _load(tuple(years_sel), combine_years, keep_only_canonical)
meas = _get_table(DATA, "measurements", years_sel)

if meas is None or meas.empty:
    st.warning(
        "Geen measurements data gevonden "
        "(measurements.parquet ontbreekt/is leeg of load_data() levert geen measurements op)."
    )
    st.stop()

meas = _canonicalize_measurements_columns(meas)

# Derive lutum% from lutum_1..n via helper
try:
    meas["lutum_%"] = avg_lutum_percentage(meas)
except Exception:
    lutum_cols = [c for c in meas.columns if str(c).startswith("lutum_")]
    if lutum_cols:
        tmp = meas[lutum_cols].apply(pd.to_numeric, errors="coerce")
        meas["lutum_%"] = tmp.mean(axis=1, skipna=True)
    else:
        meas["lutum_%"] = np.nan

# Derive dominant sediment mode
sed_cols = [c for c in meas.columns if str(c).startswith("sedimenttype_")]
if sed_cols:
    sed_tmp = meas[sed_cols].astype("string").replace({"nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    try:
        sed_mode = sed_tmp.mode(axis=1, dropna=True)
        meas["sediment_mode"] = sed_mode[0] if not sed_mode.empty else pd.NA
    except Exception:
        meas["sediment_mode"] = pd.NA
else:
    meas["sediment_mode"] = pd.NA

# Veiligheid: check of verwachte kolommen bestaan
required = {
    "Deelgebied",
    "Locatie",
    "diepte_m",
    "biovol_driehoek_ml",
    "biovol_quagga_ml",
    "biovol_totaal_ml",
}
missing = required - set(meas.columns)
if missing:
    st.error(
        "Ontbrekende kolommen in measurements: "
        f"{', '.join(sorted(missing))}\n\n"
        f"Gevonden kolommen: {list(meas.columns)}"
    )
    st.stop()

# Numerieke kolommen voorbereiden
for c in ["diepte_m", "lutum_%", "biovol_driehoek_ml", "biovol_quagga_ml", "biovol_totaal_ml"]:
    if c in meas.columns:
        meas[c] = pd.to_numeric(meas[c], errors="coerce")

# ---------------------------------------------------------
# Sidebar (paginafilters onder gedeelde sidebar)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.subheader("📌 Paginafilters")

    deelgebieden = sorted(meas["Deelgebied"].dropna().astype(str).unique().tolist())
    sel_deelgebied = st.multiselect(
        "Deelgebied",
        options=deelgebieden,
        default=deelgebieden,
        key="relaties_deelgebied",
    )

    yvar_options = [
        c for c in ["biovol_totaal_ml", "biovol_driehoek_ml", "biovol_quagga_ml"] if c in meas.columns
    ]
    yvar = st.selectbox(
        "Y-variabele",
        yvar_options,
        index=0,
        key="relaties_yvar",
    )

    show_trendline = st.checkbox("Trendline", value=True, key="relaties_trendline")

    st.divider()
    st.subheader("PCA / clustering")

    color_mode_options = ["Deelgebied", "Cluster"]
    if "jaar" in meas.columns and meas["jaar"].nunique() > 1:
        color_mode_options.insert(1, "Jaar")

    n_clusters_ui = st.slider(
        "Aantal clusters (KMeans)",
        min_value=2,
        max_value=8,
        value=4,
        key="relaties_nclusters",
    )
    color_mode = st.selectbox("Kleur op", color_mode_options, index=0, key="relaties_colormode")

# Filter view
view = meas[meas["Deelgebied"].astype(str).isin(sel_deelgebied)].copy()
if view.empty:
    st.warning("Geen data na selectie.")
    st.stop()

# ---------------------------------------------------------
# 1) Scatter: diepte vs biovol + lutum vs biovol
# ---------------------------------------------------------
st.subheader("Bivariate relaties")

col1, col2 = st.columns(2)
use_px_ols = bool(show_trendline and HAS_STATSMODELS)
facet_by_year = "jaar" in view.columns and view["jaar"].nunique() > 1

if show_trendline and not HAS_STATSMODELS:
    st.info(
        "📌 `statsmodels` is niet geïnstalleerd. Ik toon een lineaire trendlijn via `numpy` (fallback). "
        "Wil je OLS-trendlijnen van Plotly? Installeer dan `statsmodels`."
    )

hover_cols = [c for c in ["Locatie", "lutum_%", "sediment_mode", "jaar"] if c in view.columns]

with col1:
    df1 = view.dropna(subset=["diepte_m", yvar]).copy()
    fig1 = px.scatter(
        df1,
        x="diepte_m",
        y=yvar,
        color="Deelgebied",
        trendline="ols" if use_px_ols else None,
        facet_row="jaar" if facet_by_year else None,
        hover_data=hover_cols,
        labels={"diepte_m": "Diepte (m)", yvar: yvar, "jaar": "Jaar"},
        title=f"Diepte vs {yvar}",
    )
    if show_trendline and not HAS_STATSMODELS:
        fig1 = add_numpy_trendline(
            fig1,
            df=df1,
            xcol="diepte_m",
            ycol=yvar,
            color="#111111",
            name="Lineaire fit (numpy)",
        )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    df2 = view.dropna(subset=["lutum_%", yvar]).copy()
    hover_cols2 = [c for c in ["Locatie", "diepte_m", "sediment_mode", "jaar"] if c in view.columns]
    fig2 = px.scatter(
        df2,
        x="lutum_%",
        y=yvar,
        color="Deelgebied",
        trendline="ols" if use_px_ols else None,
        facet_row="jaar" if facet_by_year else None,
        hover_data=hover_cols2,
        labels={"lutum_%": "Lutum (%)", yvar: yvar, "jaar": "Jaar"},
        title=f"Lutum (%) vs {yvar}",
    )
    if show_trendline and not HAS_STATSMODELS:
        fig2 = add_numpy_trendline(
            fig2,
            df=df2,
            xcol="lutum_%",
            ycol=yvar,
            color="#111111",
            name="Lineaire fit (numpy)",
        )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------
# 2) PCA + clustering
# ---------------------------------------------------------
st.subheader("Multivariate samenvatting (PCA/ordination-achtig) + clustering")

features_num = ["diepte_m", "lutum_%", "biovol_driehoek_ml", "biovol_quagga_ml"]
df_num = view[features_num].copy().apply(pd.to_numeric, errors="coerce")

# one-hot sediment mode
if view["sediment_mode"].notna().any():
    dummies = pd.get_dummies(view["sediment_mode"].fillna("onbekend"), prefix="sed")
else:
    dummies = pd.DataFrame(index=view.index)

df_feat = pd.concat([df_num, dummies], axis=1)

# drop rows with missing numeric fundamentals
base_mask = df_num.notna().all(axis=1)
df_feat = df_feat.loc[base_mask].copy()
meta_cols = [c for c in ["Locatie", "Deelgebied", "jaar"] if c in view.columns]
meta = view.loc[df_feat.index, meta_cols].copy()

if len(df_feat) < 3:
    st.info("Onvoldoende complete records voor PCA (minimaal 3 rijen met complete kernvariabelen nodig).")
    st.stop()

# Scale + PCA
X = StandardScaler().fit_transform(df_feat.values)
pca = PCA(n_components=2, random_state=42)
pc = pca.fit_transform(X)

pca_df = pd.DataFrame(pc, columns=["PC1", "PC2"], index=df_feat.index)
pca_df = pd.concat([pca_df, meta], axis=1)

# Clustering op PCA ruimte
n_samples = len(pca_df)
max_clusters = max(2, min(8, n_samples - 1))
n_clusters = min(n_clusters_ui, max_clusters)
if n_clusters != n_clusters_ui:
    st.info(
        f"Aantal clusters aangepast van {n_clusters_ui} naar {n_clusters} "
        f"omdat er {n_samples} complete records beschikbaar zijn."
    )

km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
pca_df["Cluster"] = km.fit_predict(pca_df[["PC1", "PC2"]].values).astype(str)

if color_mode == "Cluster":
    color_col = "Cluster"
elif color_mode == "Jaar" and "jaar" in pca_df.columns:
    color_col = "jaar"
else:
    color_col = "Deelgebied"

hover_pca = [c for c in ["Locatie", "Deelgebied", "jaar", "Cluster"] if c in pca_df.columns]
figp = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color=color_col,
    hover_data=hover_pca,
    title="PCA 2D – meetpunten met vergelijkbare toestand",
)
st.plotly_chart(figp, use_container_width=True)

st.caption(
    f"Verklaarde variantie: PC1={pca.explained_variance_ratio_[0]:.2f}, "
    f"PC2={pca.explained_variance_ratio_[1]:.2f}"
)

st.subheader("Cluster-overzicht")
cluster_group_cols = [c for c in ["jaar", "Deelgebied", "Cluster"] if c in pca_df.columns]
clu = (
    pca_df.groupby(cluster_group_cols, as_index=False)
    .size()
    .rename(columns={"size": "n"})
)
st.dataframe(clu, use_container_width=True, hide_index=True)

csv = clu.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download cluster-overzicht als CSV",
    data=csv,
    file_name="cluster_overzicht.csv",
    mime="text/csv",
)
