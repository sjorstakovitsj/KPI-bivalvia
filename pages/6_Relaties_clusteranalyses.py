# pages/06_🔗_Relaties_en_PCA.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from utils import load_data, avg_lutum_percentage

st.set_page_config(page_title="Relaties & PCA", layout="wide")
st.title("🔗 Relaties en verklaringen")

# ---------------------------------------------------------
# Optioneel: check of statsmodels beschikbaar is
# Plotly trendline="ols" vereist statsmodels.
# ---------------------------------------------------------
HAS_STATSMODELS = True
try:
    import statsmodels.api as sm  # noqa: F401
except Exception:
    HAS_STATSMODELS = False


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
        return fig  # te weinig punten voor fit

    x2 = x[mask].to_numpy(dtype=float)
    y2 = y[mask].to_numpy(dtype=float)

    # Lineaire fit: y = a*x + b
    a, b = np.polyfit(x2, y2, 1)

    # teken lijn over het bereik van x
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
DATA = load_data()
meas = DATA["measurements"].copy()

# Derive lutum% from lutum_1..5
meas["lutum_%"] = avg_lutum_percentage(meas)

# Derive dominant sediment mode
sed_cols = [c for c in meas.columns if c.startswith("sedimenttype_")]
if sed_cols:
    # mode(axis=1) geeft DataFrame; [0] pakt eerste mode
    meas["sediment_mode"] = (
        meas[sed_cols]
        .astype(str)
        .replace("nan", np.nan)
        .mode(axis=1, dropna=True)[0]
    )
else:
    meas["sediment_mode"] = np.nan

# Veiligheid: check of verwachte kolommen bestaan
required = {"Deelgebied", "Locatie", "diepte_m", "biovol_driehoek_ml", "biovol_quagga_ml", "biovol_totaal_ml"}
missing = required - set(meas.columns)
if missing:
    st.error(f"Ontbrekende kolommen in measurements: {', '.join(sorted(missing))}")
    st.stop()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("Selecties")

    deelgebieden = sorted(meas["Deelgebied"].dropna().unique().tolist())
    sel_deelgebied = st.multiselect(
        "Deelgebied",
        options=deelgebieden,
        default=deelgebieden
    )

    yvar = st.selectbox(
        "Y-variabele",
        ["biovol_totaal_ml", "biovol_driehoek_ml", "biovol_quagga_ml"],
        index=0
    )

    show_trendline = st.checkbox("Trendline", value=True)
    st.divider()

    st.subheader("PCA / clustering")
    n_clusters = st.slider("Aantal clusters (KMeans)", min_value=2, max_value=8, value=4)
    color_mode = st.selectbox("Kleur op", ["Deelgebied", "Cluster"], index=0)

# Filter view
view = meas[meas["Deelgebied"].isin(sel_deelgebied)].copy()
if view.empty:
    st.warning("Geen data na selectie.")
    st.stop()

# Maak numeriek (belangrijk voor regressies/plots)
view["diepte_m"] = pd.to_numeric(view["diepte_m"], errors="coerce")
view["lutum_%"] = pd.to_numeric(view["lutum_%"], errors="coerce")
view[yvar] = pd.to_numeric(view[yvar], errors="coerce")

# ---------------------------------------------------------
# 1) Scatter: diepte vs biovol + lutum vs biovol
# ---------------------------------------------------------
st.subheader("Bivariate relaties")
col1, col2 = st.columns(2)

# Als statsmodels ontbreekt, gebruiken we géén px trendline="ols"
use_px_ols = bool(show_trendline and HAS_STATSMODELS)
if show_trendline and not HAS_STATSMODELS:
    st.info(
        "📌 `statsmodels` is niet geïnstalleerd. "
        "Ik toon een lineaire trendlijn via `numpy` (fallback). "
        "Wil je OLS-trendlijnen van Plotly? Installeer dan `statsmodels`."
    )

with col1:
    fig1 = px.scatter(
        view,
        x="diepte_m",
        y=yvar,
        color="Deelgebied",
        trendline="ols" if use_px_ols else None,
        hover_data=["Locatie", "lutum_%", "sediment_mode"],
        labels={"diepte_m": "Diepte (m)", yvar: yvar},
        title=f"Diepte vs {yvar}"
    )

    # Fallback trendline (numpy) als statsmodels ontbreekt
    if show_trendline and not HAS_STATSMODELS:
        fig1 = add_numpy_trendline(
            fig1,
            df=view,
            xcol="diepte_m",
            ycol=yvar,
            color="#111111",
            name="Lineaire fit (numpy)"
        )

    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.scatter(
        view,
        x="lutum_%",
        y=yvar,
        color="Deelgebied",
        trendline="ols" if use_px_ols else None,
        hover_data=["Locatie", "diepte_m", "sediment_mode"],
        labels={"lutum_%": "Lutum (%)", yvar: yvar},
        title=f"Lutum (%) vs {yvar}"
    )

    if show_trendline and not HAS_STATSMODELS:
        fig2 = add_numpy_trendline(
            fig2,
            df=view,
            xcol="lutum_%",
            ycol=yvar,
            color="#111111",
            name="Lineaire fit (numpy)"
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
df_feat = df_feat[base_mask].copy()
meta = view.loc[df_feat.index, ["Locatie", "Deelgebied"]].copy()

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
km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
pca_df["Cluster"] = km.fit_predict(pca_df[["PC1", "PC2"]].values).astype(str)

color_col = "Deelgebied" if color_mode == "Deelgebied" else "Cluster"

figp = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color=color_col,
    hover_data=["Locatie", "Deelgebied", "Cluster"],
    title="PCA 2D – meetpunten met vergelijkbare toestand"
)
st.plotly_chart(figp, use_container_width=True)

st.caption(
    f"Verklaarde variantie: PC1={pca.explained_variance_ratio_[0]:.2f}, "
    f"PC2={pca.explained_variance_ratio_[1]:.2f}"
)

st.subheader("Cluster-overzicht")
clu = (
    pca_df.groupby(["Deelgebied", "Cluster"], as_index=False)
    .size()
    .rename(columns={"size": "n"})
)
st.dataframe(clu, use_container_width=True)