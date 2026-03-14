# 2_Kaart.py
# Kaartpagina (Folium) – mosselkartering

import re

import branca.colormap as cm
import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium import FeatureGroup
from folium.features import DivIcon
from streamlit_folium import st_folium

from utils import load_data, render_sidebar, rd_to_wgs84

st.set_page_config(page_title="Kaart – Mosselkartering", layout="wide")

# -----------------------------
# Gedeelde sidebar (zelfde op alle pagina's)
# -----------------------------
ui = render_sidebar(title="Mosselkartering")
years_sel = list(ui.get("years", []))
combine_years = bool(ui.get("combine_years", False))
keep_only_canonical = bool(ui.get("keep_only_canonical", False))

st.title("🗺️ Kaart en ruimtelijke analyse")

if not years_sel:
    st.warning("Selecteer in de sidebar ten minste één monitoringsjaar.")
    st.stop()


@st.cache_data(show_spinner=False)
def _load(years: tuple[str, ...], combine: bool, keep_only: bool):
    return load_data(years=list(years), combine_years=combine, keep_only_canonical=keep_only)


DATA = _load(tuple(years_sel), combine_years, keep_only_canonical)

meas = DATA.get("measurements", pd.DataFrame()).copy()
adv7 = DATA.get("adv_m2", pd.DataFrame()).copy()

if meas.empty:
    st.error(
        "Geen meetdata gevonden voor de geselecteerde jaren. "
        "Controleer of processed/<jaar>/measurements.parquet bestaat en of dataprep.py succesvol heeft gedraaid."
    )
    st.stop()


# -----------------------------
# Helpers
# -----------------------------
def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def _find_col(df: pd.DataFrame, candidates):
    """Zoek een kolomnaam in df op basis van kandidaatnamen (case-insensitive)."""
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        c2 = str(cand).lower()
        if c2 in lower_map:
            return lower_map[c2]
    return None


def _make_cross_icon(size_px: int = 18, color: str = "#111111") -> DivIcon:
    """Kruis (✖) als DivIcon."""
    html = f"""<div style=\"font-size:{size_px}px;color:{color};line-height:{size_px}px;\">✖</div>"""
    return folium.DivIcon(html=html)


def _scale_radius(v, vmin, vmax, rmin=4, rmax=18):
    """Schaal radius obv waarde (sqrt) voor beter zicht."""
    if not np.isfinite(v):
        return rmin
    t = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.0
    t = max(0.0, min(1.0, t))
    t = np.sqrt(t)
    return rmin + (rmax - rmin) * t


def _loc_id(series: pd.Series) -> pd.Series:
    """Maak een stabiele locatie-id (string) voor merges/labels."""
    return series.astype(str).str.strip()


def _hap_indices(df: pd.DataFrame, prefix: str) -> list[int]:
    """Detecteer welke hap-indexen aanwezig zijn in de dataset.

    Voorbeelden:
      - prefix='PAS_' matcht kolommen PAS_1..PAS_10
      - prefix='lutum_' matcht kolommen lutum_1..lutum_10
      - prefix='sedimenttype_' matcht kolommen sedimenttype_1..sedimenttype_10

    Dit is cruciaal omdat 2021 10 happen bevat terwijl 2023/2024 vaak 5 happen hebben.
    """
    if df is None or df.empty:
        return []
    patt = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    out = []
    for c in df.columns:
        m = patt.match(str(c))
        if m:
            out.append(int(m.group(1)))
    out = sorted(set(out))
    return out


# Bepaal max aantal happen per type op basis van aanwezige kolommen
PAS_HAPS = _hap_indices(meas, "PAS_")
SED_HAPS = _hap_indices(meas, "sedimenttype_")
LUT_HAPS = _hap_indices(meas, "lutum_")

# Fallback als kolommen ontbreken (oude parquets): gebruik 5
PAS_HAPS = PAS_HAPS or list(range(1, 6))
SED_HAPS = SED_HAPS or list(range(1, 6))
LUT_HAPS = LUT_HAPS or list(range(1, 6))


# --------- 2-delige taart (soortverdeling) ----------
def _pie_svg_two(
    p_a: float,
    r: int = 14,
    color_a: str = "#1f77b4",
    color_b: str = "#ff7f0e",
    stroke: str = "#333333",
) -> str:
    """2-delige taart (inline SVG). p_a = aandeel A (0..1), rest B."""
    p_a = 0.0 if not np.isfinite(p_a) else max(0.0, min(1.0, p_a))
    cx, cy = r, r
    a = 2 * np.pi * p_a
    x = cx + r * np.sin(a)
    y = cy - r * np.cos(a)
    large_arc = 1 if p_a > 0.5 else 0
    d1 = f"M {cx},{cy} L {cx},{cy-r} A {r},{r} 0 {large_arc},1 {x:.2f},{y:.2f} Z"
    svg = f"""
<svg width="{2*r}" height="{2*r}" viewBox="0 0 {2*r} {2*r}" xmlns="http://www.w3.org/2000/svg">
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="{color_b}" stroke="{stroke}" stroke-width="1" />
  <path d="{d1}" fill="{color_a}" stroke="{stroke}" stroke-width="1" />
</svg>
"""
    return svg.strip()


# --------- Multi-slice taart (PAS/Sediment/Lutum) ----------
def _polar_to_xy(cx, cy, r, angle):
    return cx + r * np.sin(angle), cy - r * np.cos(angle)


def _wedge_path(cx, cy, r, start_angle, end_angle):
    if end_angle <= start_angle:
        end_angle = start_angle + 1e-6
    x1, y1 = _polar_to_xy(cx, cy, r, start_angle)
    x2, y2 = _polar_to_xy(cx, cy, r, end_angle)
    large_arc = 1 if (end_angle - start_angle) > np.pi else 0
    return f"M {cx},{cy} L {x1:.2f},{y1:.2f} A {r},{r} 0 {large_arc},1 {x2:.2f},{y2:.2f} Z"


def _pie_svg_multi(shares: dict, colors: dict, r: int = 14, stroke: str = "#333333") -> str:
    """Multi-slice taart (inline SVG). shares somt ~1."""
    cx, cy = r, r
    items = [(k, float(v)) for k, v in shares.items() if np.isfinite(v) and v > 0]
    items.sort(key=lambda t: t[0])
    if not items:
        return f"""<svg width="{2*r}" height="{2*r}" xmlns="http://www.w3.org/2000/svg"></svg>""".strip()

    svg_parts = [f'<svg width="{2*r}" height="{2*r}" viewBox="0 0 {2*r} {2*r}" xmlns="http://www.w3.org/2000/svg">']
    svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#ffffff" stroke="{stroke}" stroke-width="1" />')

    angle = 0.0
    for k, frac in items:
        end = angle + 2 * np.pi * frac
        path = _wedge_path(cx, cy, r, angle, end)
        col = colors.get(k, "#999999")
        svg_parts.append(f'<path d="{path}" fill="{col}" stroke="{stroke}" stroke-width="1" />')
        angle = end

    svg_parts.append("</svg>")
    return "\n".join(svg_parts).strip()


# -----------------------------
# PAS parsing (letters / Nvt)
# -----------------------------
PAS_KEYS = ["d", "m", "c", "z", "n", "w", "o", "nvt"]
PAS_LABELS = {
    "d": "Dreissena (d)",
    "m": "Mytilus (m)",
    "c": "Corbicula (c)",
    "z": "Zuiderzeeschelpen (z)",
    "n": "Najaden (n)",
    "w": "Waterplanten (w)",
    "o": "Overig (o)",
    "nvt": "Nvt",
}
PAS_COLORS = {
    "d": "#1f77b4",
    "m": "#9467bd",
    "c": "#ff7f0e",
    "z": "#8c564b",
    "n": "#d62728",
    "w": "#2ca02c",
    "o": "#7f7f7f",
    "nvt": "#c7c7c7",
}


def _parse_pas_cell(val) -> list[str]:
    if pd.isna(val):
        return []
    s = str(val).strip().lower()
    if s in ("", "nan", "none"):
        return []
    s = s.replace(" ", "")
    if "nvt" in s:
        return ["nvt"]
    tokens = re.split(r"[^a-z]+", s)
    out = []
    for t in tokens:
        if not t:
            continue
        if t in PAS_KEYS:
            out.append(t)
            continue
        if all(ch in PAS_KEYS for ch in t) and len(t) > 1:
            out.extend(list(t))
    return list(dict.fromkeys(out))


def _pas_distribution_for_row(row) -> tuple[dict, float, int]:
    weights = {k: 0.0 for k in PAS_KEYS}
    used = 0.0
    for i in PAS_HAPS:
        col = f"PAS_{i}"
        if col not in row.index:
            continue
        cats = _parse_pas_cell(row.get(col))
        if not cats:
            continue
        w = 1.0 / len(cats)
        for c in cats:
            if c in weights:
                weights[c] += w
        used += 1.0
    total = sum(weights.values())
    if total <= 0:
        return {}, used, len(PAS_HAPS)
    shares = {k: v / total for k, v in weights.items() if v > 0}
    return shares, used, len(PAS_HAPS)


# -----------------------------
# Sedimenttype parsing
# -----------------------------
SED_KEYS = ["k", "z", "z/s", "v", "s", "s/z", "g", "x"]
SED_LABELS = {
    "k": "Klei (k)",
    "z": "Zand (z)",
    "z/s": "Meeste zand met slib (z/s)",
    "v": "Veen (v)",
    "s": "Slib (s)",
    "s/z": "Meeste slib met zand (s/z)",
    "g": "Grind (g)",
    "x": "Grof materiaal (x)",
}
SED_COLORS = {
    "k": "#a65628",
    "z": "#fdbf6f",
    "z/s": "#ffdd99",
    "v": "#6a3d9a",
    "s": "#bdbdbd",
    "s/z": "#969696",
    "g": "#7f7f7f",
    "x": "#1b1b1b",
}


def _norm_sed_token(t: str):
    if t is None:
        return None
    s = str(t).strip().lower().replace(" ", "")
    if not s:
        return None
    if s in ("z/s", "z_s", "z-s", "zs"):
        return "z/s"
    if s in ("s/z", "s_s", "s-z", "sz"):
        return "s/z"
    if s in ("k", "z", "v", "s", "g", "x"):
        return s
    if len(s) > 1 and all(ch in ("k", "z", "v", "s", "g", "x") for ch in s):
        return s
    return None


def _parse_sed_cell(val) -> list[str]:
    if pd.isna(val):
        return []
    s = str(val).strip().lower()
    if s in ("", "nan", "none"):
        return []
    parts = re.split(r"[;,/\\+&]|\ben\b|\bof\b", s)
    tokens = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        tok = _norm_sed_token(p)
        if tok is None:
            continue
        if tok in SED_KEYS:
            tokens.append(tok)
        else:
            for ch in tok:
                if ch in ("k", "z", "v", "s", "g", "x"):
                    tokens.append(ch)
    tokens = list(dict.fromkeys(tokens))
    return [t for t in tokens if t in SED_KEYS]


def _sed_distribution_for_row(row) -> tuple[dict, float, int]:
    weights = {k: 0.0 for k in SED_KEYS}
    used = 0.0
    for i in SED_HAPS:
        col = f"sedimenttype_{i}"
        if col not in row.index:
            continue
        cats = _parse_sed_cell(row.get(col))
        if not cats:
            continue
        w = 1.0 / len(cats)
        for c in cats:
            if c in weights:
                weights[c] += w
        used += 1.0
    total = sum(weights.values())
    if total <= 0:
        return {}, used, len(SED_HAPS)
    shares = {k: v / total for k, v in weights.items() if v > 0}
    return shares, used, len(SED_HAPS)


# -----------------------------
# Lutumgehalte parsing
# -----------------------------
LUT_KEYS = ["0-2", "2-5", "5-8", "8-12", "12-17", "17-25", "25-35", ">35"]
LUT_LABELS = {
    "0-2": "0–2% klei-arm zand",
    "2-5": "2–5% kleihoudend zand",
    "5-8": "5–8% kleiig-/slibbig zand",
    "8-12": "8–12% zeer lichte zavel",
    "12-17": "12–17% matig lichte zavel",
    "17-25": "17–25% zware zavel",
    "25-35": "25–35% lichte klei",
    ">35": ">35% zware klei",
}
LUT_COLORS = {
    "0-2": "#fff7bc",
    "2-5": "#fee391",
    "5-8": "#fec44f",
    "8-12": "#fe9929",
    "12-17": "#ec7014",
    "17-25": "#cc4c02",
    "25-35": "#993404",
    ">35": "#662506",
}


def _lut_bin_from_number(x: float):
    if not np.isfinite(x) or x < 0:
        return None
    if x < 2:
        return "0-2"
    if x < 5:
        return "2-5"
    if x < 8:
        return "5-8"
    if x < 12:
        return "8-12"
    if x < 17:
        return "12-17"
    if x < 25:
        return "17-25"
    if x < 35:
        return "25-35"
    return ">35"


def _parse_lutum_cell(val) -> list[str]:
    if pd.isna(val):
        return []
    s = str(val).strip().lower()
    if s in ("", "nan", "none"):
        return []

    s_num = s.replace("%", "").replace(",", ".")
    try:
        x = float(s_num)
        b = _lut_bin_from_number(x)
        return [b] if b else []
    except Exception:
        pass

    parts = re.split(r"[;,/\\+&]|\ben\b|\bof\b", s)
    out = []
    for p in parts:
        p = p.strip().replace(" ", "")
        if not p:
            continue
        if p.startswith(">"):
            nums = re.findall(r"\d+(?:\.\d+)?", p)
            if nums:
                x = float(nums[0])
                out.append(_lut_bin_from_number(max(35.0, x)))
            continue
        m = re.search(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)", p)
        if m:
            a = float(m.group(1))
            b = float(m.group(2))
            out.append(_lut_bin_from_number((a + b) / 2.0))
            continue
        nums = re.findall(r"\d+(?:\.\d+)?", p)
        if nums:
            x = float(nums[0])
            out.append(_lut_bin_from_number(x))

    out = [x for x in out if x in LUT_KEYS]
    return list(dict.fromkeys(out))


def _lut_distribution_for_row(row) -> tuple[dict, float, int]:
    weights = {k: 0.0 for k in LUT_KEYS}
    used = 0.0
    for i in LUT_HAPS:
        col = f"lutum_{i}"
        if col not in row.index:
            continue
        cats = _parse_lutum_cell(row.get(col))
        if not cats:
            continue
        w = 1.0 / len(cats)
        for c in cats:
            if c in weights:
                weights[c] += w
        used += 1.0
    total = sum(weights.values())
    if total <= 0:
        return {}, used, len(LUT_HAPS)
    shares = {k: v / total for k, v in weights.items() if v > 0}
    return shares, used, len(LUT_HAPS)


# -----------------------------
# Prepare data: lat/lon en ids
# -----------------------------
if "Locatie" in meas.columns:
    meas["Locatie_id"] = _loc_id(meas["Locatie"])
else:
    meas["Locatie_id"] = ""

# Jaar aanwezig? (bij combine_years in utils)
if "jaar" not in meas.columns and len(years_sel) == 1:
    meas["jaar"] = years_sel[0]

# lat/lon fallback: als ontbreekt, probeer RD->WGS84
if "lat" not in meas.columns or "lon" not in meas.columns:
    x_col = _find_col(meas, ["x_rd", "X.1", "x.1"])
    y_col = _find_col(meas, ["y_rd", "Y.1", "y.1"])
    if x_col and y_col:
        latlon = meas[[x_col, y_col]].apply(
            lambda r: rd_to_wgs84(float(r[x_col]), float(r[y_col]))
            if pd.notna(r[x_col]) and pd.notna(r[y_col])
            else (np.nan, np.nan),
            axis=1,
        )
        meas["lat"] = [t[0] for t in latlon]
        meas["lon"] = [t[1] for t in latlon]

# ADV voorbereiden
if not adv7.empty and "Locatie" in adv7.columns:
    adv7["Locatie_id"] = _loc_id(adv7["Locatie"])


# -----------------------------
# Sidebar (pagina-specifieke filters)
# -----------------------------
with st.sidebar:
    st.header("Filters & lagen")

    deel_opt = ["(alle)"]
    if "Deelgebied" in meas.columns:
        deel_opt += sorted(meas["Deelgebied"].dropna().astype(str).unique().tolist())

    sel_deelgebied = st.selectbox("Deelgebied", options=deel_opt, key="deelgebied_sel")

    kaartlaag = st.selectbox(
        "Kaartlaag",
        options=[
            "Meetpunten",
            "Soortverdeling",
            "Asvrijdrooggewicht",
            "Biovolumina",
            "PAS (primair aanhechtingssubstraat)",
            "Sedimenttype",
            "Lutumgehalte",
        ],
        index=0,
        key="kaartlaag_sel",
    )

    if kaartlaag == "Biovolumina":
        st.subheader("Biovolumina selectie")
        show_tri = st.checkbox("Driehoeksmossel", value=True, key="bio_tri_chk")
        show_qua = st.checkbox("Quaggamossel", value=True, key="bio_qua_chk")
    else:
        show_tri, show_qua = True, True


# -----------------------------
# Apply filter
# -----------------------------
m = meas.copy()
if sel_deelgebied != "(alle)" and "Deelgebied" in m.columns:
    m = m[m["Deelgebied"] == sel_deelgebied].copy()

if m.empty:
    st.warning("Geen meetpunten gevonden voor de gekozen selectie.")
    st.stop()


# -----------------------------
# ADV (Bijlage 7): merge op Locatie_id + (Deelgebied) + (jaar)
# -----------------------------
ADV_STD_COL = "adv_mg_m2_total"

if adv7 is None or adv7.empty:
    m[ADV_STD_COL] = np.nan
else:
    adv7_df = adv7.copy()

    bug_col = _find_col(adv7_df, ["Berekend ADV bugensis (mg/m2)"])
    pol_col = _find_col(adv7_df, ["Berekend ADV polymorpha (mg/m2)"])

    if bug_col is not None:
        adv7_df[bug_col] = pd.to_numeric(adv7_df[bug_col], errors="coerce")
    if pol_col is not None:
        adv7_df[pol_col] = pd.to_numeric(adv7_df[pol_col], errors="coerce")

    parts = []
    if bug_col is not None:
        parts.append(adv7_df[bug_col])
    if pol_col is not None:
        parts.append(adv7_df[pol_col])

    adv7_df[ADV_STD_COL] = pd.concat(parts, axis=1).sum(axis=1, min_count=1) if parts else np.nan

    group_cols = ["Locatie_id"]
    if "Deelgebied" in adv7_df.columns and "Deelgebied" in m.columns:
        group_cols.append("Deelgebied")
    if "jaar" in adv7_df.columns and "jaar" in m.columns:
        group_cols.append("jaar")

    adv7_agg = adv7_df.groupby(group_cols, dropna=False)[ADV_STD_COL].mean().reset_index()

    merge_cols = ["Locatie_id"]
    if "Deelgebied" in adv7_agg.columns and "Deelgebied" in m.columns:
        merge_cols.append("Deelgebied")
    if "jaar" in adv7_agg.columns and "jaar" in m.columns:
        merge_cols.append("jaar")

    m = m.merge(adv7_agg, how="left", on=merge_cols)


# -----------------------------
# Map center
# -----------------------------
if "lat" not in m.columns or "lon" not in m.columns:
    st.error("Geen lat/lon informatie beschikbaar om een kaart te tekenen.")
    st.stop()

center = [
    float(pd.to_numeric(m["lat"], errors="coerce").mean()),
    float(pd.to_numeric(m["lon"], errors="coerce").mean()),
]
zoom = 10 if sel_deelgebied == "(alle)" else 11
mp = folium.Map(location=center, zoom_start=zoom, tiles="cartodbpositron")


# -----------------------------
# Layers
# -----------------------------
fg_points = FeatureGroup(name="Meetpunten", show=(kaartlaag == "Meetpunten"))
fg_pies = FeatureGroup(name="Soortverdeling (driehoek/quagga)", show=(kaartlaag == "Soortverdeling"))
fg_adv = FeatureGroup(name="Asvrij drooggewicht (ADV, mg/m²)", show=(kaartlaag == "Asvrijdrooggewicht"))
fg_bio = FeatureGroup(name="Biovolumina", show=(kaartlaag == "Biovolumina"))
fg_pas = FeatureGroup(name="PAS (primair aanhechtingssubstraat)", show=(kaartlaag.startswith("PAS")))
fg_sed = FeatureGroup(name="Sedimenttype", show=(kaartlaag == "Sedimenttype"))
fg_lut = FeatureGroup(name="Lutumgehalte", show=(kaartlaag == "Lutumgehalte"))

POINT_COLOR = "#1f77b4"

# ADV colormap (lichtgroen -> donkergroen), excl. 0
adv_vals = pd.to_numeric(m.get(ADV_STD_COL), errors="coerce")
finite_adv = adv_vals[np.isfinite(adv_vals)]
finite_adv_pos = finite_adv[finite_adv > 0]
adv_cmap = None
adv_min = None
adv_max = None
if len(finite_adv_pos) > 0:
    adv_min = float(finite_adv_pos.min())
    adv_max = float(finite_adv_pos.max())
    if adv_max == adv_min:
        adv_max = adv_min + 1.0
    adv_cmap = cm.LinearColormap(colors=["#e8f5e9", "#1b5e20"], vmin=adv_min, vmax=adv_max)
    adv_cmap.caption = "Asvrij drooggewicht (ADV, mg/m²)"

# Biovolume scaling
bio_tri_vals = pd.to_numeric(m.get("biovol_driehoek_ml"), errors="coerce")
bio_qua_vals = pd.to_numeric(m.get("biovol_quagga_ml"), errors="coerce")

bio_vals_pool = []
if kaartlaag == "Biovolumina":
    if show_tri:
        bio_vals_pool.append(bio_tri_vals)
    if show_qua:
        bio_vals_pool.append(bio_qua_vals)

if bio_vals_pool:
    bio_all = pd.concat(bio_vals_pool, axis=0)
    bio_all = bio_all[np.isfinite(bio_all) & (bio_all > 0)]
    bio_min = float(bio_all.min()) if len(bio_all) else 0.0
    bio_max = float(bio_all.max()) if len(bio_all) else 1.0
    if bio_max == bio_min:
        bio_max = bio_min + 1.0
else:
    bio_min, bio_max = 0.0, 1.0


def _bio_radius(v):
    if not np.isfinite(v) or v <= 0:
        return 0
    t = (v - bio_min) / (bio_max - bio_min) if bio_max > bio_min else 0.0
    t = max(0.0, min(1.0, t))
    t = np.sqrt(t)
    return 4 + (20 - 4) * t


# -----------------------------
# Add markers
# -----------------------------
for _, row in m.iterrows():
    lat = _safe_float(row.get("lat"))
    lon = _safe_float(row.get("lon"))
    if not np.isfinite(lat) or not np.isfinite(lon):
        continue

    dg = row.get("Deelgebied", "onbekend")
    loc_txt = row.get("Locatie_id", row.get("Locatie", ""))

    biovol_total = _safe_float(row.get("biovol_totaal_ml", 0.0), 0.0)
    b_tri = _safe_float(row.get("biovol_driehoek_ml", 0.0), 0.0)
    b_qua = _safe_float(row.get("biovol_quagga_ml", 0.0), 0.0)

    if kaartlaag == "Meetpunten":
        tooltip_point = (
            f"{dg} – Locatie {loc_txt}<br>"
            f"Biovolume totaal: {biovol_total:.2f} ml<br>"
            f"Driehoek: {b_tri:.2f} ml – Quagga: {b_qua:.2f} ml"
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=POINT_COLOR,
            fill=True,
            fill_opacity=0.9,
            weight=2,
            tooltip=folium.Tooltip(tooltip_point, sticky=True),
        ).add_to(fg_points)

    if kaartlaag == "Soortverdeling":
        denom = (b_tri + b_qua)
        p_tri = (b_tri / denom) if denom > 0 else 0.0
        p_qua = 1.0 - p_tri
        svg = _pie_svg_two(p_tri)
        icon = folium.DivIcon(html=f"""<div style='transform: translate(-50%, -50%);'>{svg}</div>""")
        tooltip_pie = (
            f"{dg} – Locatie {loc_txt}<br>"
            f"Aandeel driehoek: {p_tri:.0%}<br>"
            f"Aandeel quagga: {p_qua:.0%}<br>"
            f"Driehoek (ml): {b_tri:.2f} – Quagga (ml): {b_qua:.2f}"
        )
        folium.Marker(
            location=[lat, lon],
            icon=icon,
            tooltip=folium.Tooltip(tooltip_pie, sticky=True),
        ).add_to(fg_pies)

    if kaartlaag == "Asvrijdrooggewicht":
        adv_v = _safe_float(row.get(ADV_STD_COL), np.nan)
        if np.isfinite(adv_v) and adv_v > 0 and adv_cmap is not None:
            radius = _scale_radius(adv_v, adv_min, adv_max, rmin=4, rmax=18)
            color = adv_cmap(adv_v)
            tooltip_adv = (
                f"{dg} – Locatie {loc_txt}<br>"
                f"ADV: {adv_v:.2f} mg/m²<br>"
                f"Biovolume totaal: {biovol_total:.2f} ml"
            )
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                weight=2,
                tooltip=folium.Tooltip(tooltip_adv, sticky=True),
            ).add_to(fg_adv)
        else:
            reason = "ADV = 0 (geen aangetroffen)" if np.isfinite(adv_v) and adv_v == 0 else "geen waarde gevonden (Bijlage 7)"
            tooltip_none = f"{dg} – Locatie {loc_txt}<br>ADV: {reason}"
            folium.Marker(
                location=[lat, lon],
                icon=_make_cross_icon(size_px=18, color="#111111"),
                tooltip=folium.Tooltip(tooltip_none, sticky=True),
            ).add_to(fg_adv)

    if kaartlaag == "Biovolumina":
        candidates = []
        if show_tri:
            candidates.append(("Driehoeksmossel", b_tri, "#1f77b4"))
        if show_qua:
            candidates.append(("Quaggamossel", b_qua, "#ff7f0e"))
        candidates.sort(key=lambda t: (t[1] if np.isfinite(t[1]) else -1), reverse=True)

        for label, val, col in candidates:
            if np.isfinite(val) and val > 0:
                r = _bio_radius(val)
                tooltip_bio = (
                    f"{dg} – Locatie {loc_txt}<br>"
                    f"{label}: {val:.2f} ml<br>"
                    f"Totaal: {biovol_total:.2f} ml"
                )
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=r,
                    color=col,
                    fill=True,
                    fill_color=col,
                    fill_opacity=0.45,
                    weight=2,
                    tooltip=folium.Tooltip(tooltip_bio, sticky=True),
                ).add_to(fg_bio)
            else:
                tooltip_none = f"{dg} – Locatie {loc_txt}<br>{label}: 0 of geen waarde"
                folium.Marker(
                    location=[lat, lon],
                    icon=_make_cross_icon(size_px=16, color=col),
                    tooltip=folium.Tooltip(tooltip_none, sticky=True),
                ).add_to(fg_bio)

    if kaartlaag.startswith("PAS"):
        shares, used_haps, total_haps = _pas_distribution_for_row(row)
        if not shares:
            tooltip_none = f"{dg} – Locatie {loc_txt}<br>PAS: geen (bruikbare) data gevonden"
            folium.Marker(
                location=[lat, lon],
                icon=_make_cross_icon(size_px=18, color="#111111"),
                tooltip=folium.Tooltip(tooltip_none, sticky=True),
            ).add_to(fg_pas)
        else:
            svg = _pie_svg_multi(shares, PAS_COLORS, r=14)
            icon = folium.DivIcon(html=f"""<div style='transform: translate(-50%, -50%);'>{svg}</div>""")
            lines = [f"{PAS_LABELS.get(k, k)}: {frac:.0%}" for k, frac in sorted(shares.items(), key=lambda kv: kv[1], reverse=True)]
            tooltip_pas = (
                f"{dg} – Locatie {loc_txt}<br>"
                f"Bodemhappen met PAS-info: {int(used_haps)}/{int(total_haps)}<br>" + "<br>".join(lines)
            )
            folium.Marker(
                location=[lat, lon],
                icon=icon,
                tooltip=folium.Tooltip(tooltip_pas, sticky=True),
            ).add_to(fg_pas)

    if kaartlaag == "Sedimenttype":
        shares, used_haps, total_haps = _sed_distribution_for_row(row)
        if not shares:
            tooltip_none = f"{dg} – Locatie {loc_txt}<br>Sedimenttype: geen (bruikbare) data gevonden"
            folium.Marker(
                location=[lat, lon],
                icon=_make_cross_icon(size_px=18, color="#111111"),
                tooltip=folium.Tooltip(tooltip_none, sticky=True),
            ).add_to(fg_sed)
        else:
            svg = _pie_svg_multi(shares, SED_COLORS, r=14)
            icon = folium.DivIcon(html=f"""<div style='transform: translate(-50%, -50%);'>{svg}</div>""")
            lines = [f"{SED_LABELS.get(k, k)}: {frac:.0%}" for k, frac in sorted(shares.items(), key=lambda kv: kv[1], reverse=True)]
            tooltip_sed = (
                f"{dg} – Locatie {loc_txt}<br>"
                f"Bodemhappen met sedimentinfo: {int(used_haps)}/{int(total_haps)}<br>" + "<br>".join(lines)
            )
            folium.Marker(
                location=[lat, lon],
                icon=icon,
                tooltip=folium.Tooltip(tooltip_sed, sticky=True),
            ).add_to(fg_sed)

    if kaartlaag == "Lutumgehalte":
        shares, used_haps, total_haps = _lut_distribution_for_row(row)
        if not shares:
            tooltip_none = f"{dg} – Locatie {loc_txt}<br>Lutum: geen (bruikbare) data gevonden"
            folium.Marker(
                location=[lat, lon],
                icon=_make_cross_icon(size_px=18, color="#111111"),
                tooltip=folium.Tooltip(tooltip_none, sticky=True),
            ).add_to(fg_lut)
        else:
            svg = _pie_svg_multi(shares, LUT_COLORS, r=14)
            icon = folium.DivIcon(html=f"""<div style='transform: translate(-50%, -50%);'>{svg}</div>""")
            lines = [f"{LUT_LABELS.get(k, k)}: {frac:.0%}" for k, frac in sorted(shares.items(), key=lambda kv: kv[1], reverse=True)]
            tooltip_lut = (
                f"{dg} – Locatie {loc_txt}<br>"
                f"Bodemhappen met lutuminfo: {int(used_haps)}/{int(total_haps)}<br>" + "<br>".join(lines)
            )
            folium.Marker(
                location=[lat, lon],
                icon=icon,
                tooltip=folium.Tooltip(tooltip_lut, sticky=True),
            ).add_to(fg_lut)


# -----------------------------
# Add selected layer to map
# -----------------------------
if kaartlaag == "Meetpunten":
    mp.add_child(fg_points)
elif kaartlaag == "Soortverdeling":
    mp.add_child(fg_pies)
elif kaartlaag == "Asvrijdrooggewicht":
    mp.add_child(fg_adv)
    if adv_cmap is not None:
        adv_cmap.add_to(mp)
elif kaartlaag == "Biovolumina":
    mp.add_child(fg_bio)
elif kaartlaag.startswith("PAS"):
    mp.add_child(fg_pas)
elif kaartlaag == "Sedimenttype":
    mp.add_child(fg_sed)
elif kaartlaag == "Lutumgehalte":
    mp.add_child(fg_lut)

folium.LayerControl(collapsed=False).add_to(mp)

st_folium(mp, height=700, use_container_width=True)


# -----------------------------
# Caption
# -----------------------------
if kaartlaag == "Meetpunten":
    st.caption("Meetpunten hebben één uniforme kleur. Tooltip toont biovolume-informatie.")
elif kaartlaag == "Soortverdeling":
    st.caption("Taartdiagram toont aandeel driehoek én quagga per meetpunt.")
elif kaartlaag == "Asvrijdrooggewicht":
    st.caption(
        "Bollen tonen ADV uit Bijlage 7 (mg/m²): groter en donkerder groen = hogere waarde. "
        "Kruis = geen ADV-waarde of ADV=0 op die locatie."
    )
elif kaartlaag == "Biovolumina":
    st.caption("Biovolumina: bollen per geselecteerde soort (ml). Kruis = 0 of geen waarde. Kleuren: blauw=driehoek, oranje=quagga.")
elif kaartlaag.startswith("PAS"):
    st.caption(
        f"PAS: taartdiagram toont verdeling van primair aanhechtingssubstraat over {len(PAS_HAPS)} bodemhappen. "
        "Bij meerdere substraten in één hap wordt die hap gelijk verdeeld."
    )
elif kaartlaag == "Sedimenttype":
    st.caption(
        f"Sedimenttype: taartdiagram toont verdeling over {len(SED_HAPS)} bodemhappen. "
        "Bij meerdere typen in één hap wordt die hap gelijk verdeeld."
    )
else:
    st.caption(
        f"Lutumgehalte: taartdiagram toont verdeling van lutumklassen over {len(LUT_HAPS)} bodemhappen. "
        "Variatie tussen happen binnen één locatie wordt zichtbaar als meerdere taartsegmenten."
    )
