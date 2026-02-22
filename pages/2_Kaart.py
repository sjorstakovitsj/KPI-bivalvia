# pages/02_🗺️_Kaart.py
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium import FeatureGroup
from folium.features import DivIcon
from streamlit_folium import st_folium
import branca.colormap as cm
from pathlib import Path
import re

from utils import load_data  # geen toestand/trend/bedekking imports

st.set_page_config(page_title="Kaart – Mosselkartering", layout="wide")
st.title("🗺️ Kaart en ruimtelijke analyse")

DATA = load_data()
meas = DATA["measurements"].copy()

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
    html = f"""
    <div style="
        font-size:{size_px}px;
        color:{color};
        font-weight:700;
        transform: translate(-50%, -50%);
        text-shadow: 0 0 2px rgba(255,255,255,0.85);
    ">✖</div>
    """
    return folium.DivIcon(html=html)

def _scale_radius(v, vmin, vmax, rmin=4, rmax=18):
    """Schaal radius obv waarde (sqrt) voor beter zicht."""
    if not np.isfinite(v):
        return rmin
    t = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.0
    t = max(0.0, min(1.0, t))
    t = np.sqrt(t)
    return rmin + (rmax - rmin) * t

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
    <svg xmlns="http://www.w3.org/2000/svg" width="{2*r}" height="{2*r}">
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="{color_b}" stroke="{stroke}" stroke-width="1"/>
      <path d="{d1}" fill="{color_a}" stroke="{stroke}" stroke-width="0.5"/>
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
        return f"""
        <svg xmlns="http://www.w3.org/2000/svg" width="{2*r}" height="{2*r}">
          <circle cx="{cx}" cy="{cy}" r="{r}" fill="#eeeeee" stroke="{stroke}" stroke-width="1"/>
        </svg>
        """.strip()

    svg_parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{2*r}" height="{2*r}">']
    svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#ffffff" stroke="{stroke}" stroke-width="1"/>')

    angle = 0.0
    for k, frac in items:
        end = angle + 2 * np.pi * frac
        path = _wedge_path(cx, cy, r, angle, end)
        col = colors.get(k, "#999999")
        svg_parts.append(f'<path d="{path}" fill="{col}" stroke="{stroke}" stroke-width="0.3"/>')
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
            continue
    return list(dict.fromkeys(out))

def _pas_distribution_for_row(row) -> tuple[dict, float]:
    weights = {k: 0.0 for k in PAS_KEYS}
    used = 0.0
    for i in range(1, 6):
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
        return {}, used
    shares = {k: v / total for k, v in weights.items() if v > 0}
    return shares, used

# -----------------------------
# Sedimenttype parsing (k, z, z/s, v, s, s/z, g, x)
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

def _norm_sed_token(t: str) -> str | None:
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
        return s  # caller split
    return None

def _parse_sed_cell(val) -> list[str]:
    if pd.isna(val):
        return []
    s = str(val).strip().lower()
    if s in ("", "nan", "none"):
        return []

    parts = re.split(r"[;,/\\+&]| en | of |\|", s)
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

def _sed_distribution_for_row(row) -> tuple[dict, float]:
    weights = {k: 0.0 for k in SED_KEYS}
    used = 0.0
    for i in range(1, 6):
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
        return {}, used
    shares = {k: v / total for k, v in weights.items() if v > 0}
    return shares, used

# -----------------------------
# Lutumgehalte parsing (percent/klassen)
# -----------------------------
LUT_KEYS = [
    "0-2", "2-5", "5-8", "8-12", "12-17", "17-25", "25-35", ">35"
]
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
    "0-2":  "#fff7bc",
    "2-5":  "#fee391",
    "5-8":  "#fec44f",
    "8-12": "#fe9929",
    "12-17":"#ec7014",
    "17-25":"#cc4c02",
    "25-35":"#993404",
    ">35":  "#662506",
}

def _lut_bin_from_number(x: float) -> str | None:
    if not np.isfinite(x):
        return None
    if x < 0:
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
    """
    Parseer één lutum-cel. Ondersteunt:
    - numeriek: 14, 14%
    - tekst: '0-2%', '>35%', '8-12', '12-17%' etc.
    - meerdere waarden in één cel: '8-12 en 12-17' / '8-12;12-17'
    """
    if pd.isna(val):
        return []
    s = str(val).strip().lower()
    if s in ("", "nan", "none"):
        return []

    # probeer direct numeriek (ook "14%")
    s_num = s.replace("%", "").replace(",", ".")
    try:
        x = float(s_num)
        b = _lut_bin_from_number(x)
        return [b] if b else []
    except Exception:
        pass

    # split in delen (meerdere in 1 cel)
    parts = re.split(r"[;,/\\+&]| en | of |\|", s)
    out = []
    for p in parts:
        p = p.strip().replace(" ", "")
        if not p:
            continue

        # >35 etc.
        if p.startswith(">"):
            nums = re.findall(r"\d+(?:\.\d+)?", p)
            if nums:
                x = float(nums[0])
                out.append(_lut_bin_from_number(max(35.0, x)))
            continue

        # bereik 0-2 / 0–2 / 0-2%
        m = re.search(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)", p)
        if m:
            a = float(m.group(1))
            b = float(m.group(2))
            mid = (a + b) / 2.0
            out.append(_lut_bin_from_number(mid))
            continue

        # losse getal in tekst
        nums = re.findall(r"\d+(?:\.\d+)?", p)
        if nums:
            x = float(nums[0])
            out.append(_lut_bin_from_number(x))

    out = [x for x in out if x in LUT_KEYS]
    return list(dict.fromkeys(out))

def _lut_distribution_for_row(row) -> tuple[dict, float]:
    """
    Verdeling over 5 bodemhappen: lutum_1..lutum_5
    Bij meerdere klassen in één hap: gelijke verdeling (1/k).
    """
    weights = {k: 0.0 for k in LUT_KEYS}
    used = 0.0
    for i in range(1, 6):
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
        return {}, used
    shares = {k: v / total for k, v in weights.items() if v > 0}
    return shares, used

# -----------------------------
# Sidebar filters / kaartlaag
# -----------------------------
with st.sidebar:
    st.header("Filters & lagen")

    sel_deelgebied = st.selectbox(
        "Deelgebied",
        options=["(alle)"] + sorted(meas["Deelgebied"].dropna().unique().tolist()),
        key="deelgebied_sel",
    )

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

    # Alleen tonen als Biovolumina is geselecteerd
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
if sel_deelgebied != "(alle)":
    m = m[m["Deelgebied"] == sel_deelgebied].copy()

if m.empty:
    st.warning("Geen meetpunten gevonden voor de gekozen selectie.")
    st.stop()

# -----------------------------
# ADV (Bijlage 7): merge op Locatie (+ Deelgebied indien beschikbaar)
# -----------------------------
ADV_STD_COL = "adv_mg_m2_total"

adv7 = None
ADV_KEYS = ["adv_m2_locations", "adv_m2_location", "adv_m2", "adv_m2_locations.parquet"]
for k in ADV_KEYS:
    tmp = DATA.get(k)
    if tmp is not None:
        adv7 = tmp
        break

# Fallback: direct uit processed lezen (dataprep schrijft dit bestand) [1](https://rijkswaterstaat-my.sharepoint.com/personal/ben_bildirici_rws_nl/Documents/Microsoft%20Copilot%20Chat%20Files/dataprep.py)
if adv7 is None:
    p = Path("processed") / "adv_m2_locations.parquet"
    if p.exists():
        try:
            adv7 = pd.read_parquet(p)
        except Exception:
            adv7 = None

if adv7 is None or (hasattr(adv7, "empty") and adv7.empty):
    m[ADV_STD_COL] = np.nan
else:
    adv7_df = adv7.copy()
    adv7_loc_col = _find_col(adv7_df, ["Locatie", "locatie", "LOCATIE"])
    adv7_deel_col = _find_col(adv7_df, ["Deelgebied", "deelgebied"])
    bug_col = _find_col(adv7_df, ["Berekend ADV bugensis (mg/m2)"])
    pol_col = _find_col(adv7_df, ["Berekend ADV polymorpha (mg/m2)"])
    meas_loc_col = _find_col(m, ["Locatie", "locatie", "LOCATIE"])
    meas_deel_col = _find_col(m, ["Deelgebied", "deelgebied"])

    if adv7_loc_col is None or meas_loc_col is None or (bug_col is None and pol_col is None):
        m[ADV_STD_COL] = np.nan
    else:
        adv7_df[adv7_loc_col] = pd.to_numeric(adv7_df[adv7_loc_col], errors="coerce")
        m[meas_loc_col] = pd.to_numeric(m[meas_loc_col], errors="coerce")

        if bug_col is not None:
            adv7_df[bug_col] = pd.to_numeric(adv7_df[bug_col], errors="coerce")
        if pol_col is not None:
            adv7_df[pol_col] = pd.to_numeric(adv7_df[pol_col], errors="coerce")

        parts = []
        if bug_col is not None:
            parts.append(adv7_df[bug_col])
        if pol_col is not None:
            parts.append(adv7_df[pol_col])

        adv7_df[ADV_STD_COL] = pd.concat(parts, axis=1).sum(axis=1, min_count=1)

        group_cols = [adv7_loc_col]
        if adv7_deel_col is not None:
            group_cols.append(adv7_deel_col)

        adv7_agg = (
            adv7_df
            .groupby(group_cols, dropna=False)[ADV_STD_COL]
            .mean()
            .reset_index()
        )

        if adv7_deel_col is not None and meas_deel_col is not None:
            m = m.merge(
                adv7_agg,
                how="left",
                left_on=[meas_loc_col, meas_deel_col],
                right_on=[adv7_loc_col, adv7_deel_col],
            )
        else:
            m = m.merge(
                adv7_agg,
                how="left",
                left_on=[meas_loc_col],
                right_on=[adv7_loc_col],
            )

# -----------------------------
# Map center
# -----------------------------
center = [float(m["lat"].mean()), float(m["lon"].mean())]
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

# ADV colormap (lichtgroen -> donkergroen), excl. 0 (want 0 => kruis)
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
    adv_cmap = cm.LinearColormap(
        colors=["#e8f5e9", "#1b5e20"],
        vmin=adv_min,
        vmax=adv_max,
    )
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
    loc = row.get("Locatie", row.get("locatie", ""))

    biovol_total = _safe_float(row.get("biovol_totaal_ml", 0.0), 0.0)
    b_tri = _safe_float(row.get("biovol_driehoek_ml", 0.0), 0.0)
    b_qua = _safe_float(row.get("biovol_quagga_ml", 0.0), 0.0)

    try:
        loc_txt = f"{int(loc)}"
    except Exception:
        loc_txt = f"{loc}"

    # (1) Meetpunten
    if kaartlaag == "Meetpunten":
        tooltip_point = (
            f"{dg} – Locatie {loc_txt}<br>"
            f"Biovolume totaal: {biovol_total:.2f} ml<br>"
            f"Driehoek: {b_tri:.2f} ml "
            f"Quagga: {b_qua:.2f} ml"
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

    # (2) Soortverdeling
    if kaartlaag == "Soortverdeling":
        denom = (b_tri + b_qua)
        p_tri = (b_tri / denom) if denom > 0 else 0.0
        p_qua = 1.0 - p_tri

        svg = _pie_svg_two(p_tri)
        icon = folium.DivIcon(
            html=f"""
            <div style="width: 28px; height: 28px; transform: translate(-14px, -14px);">
              {svg}
            </div>
            """
        )

        tooltip_pie = (
            f"{dg} – Locatie {loc_txt}<br>"
            f"Aandeel driehoek: {p_tri:.0%}<br>"
            f"Aandeel quagga: {p_qua:.0%}<br>"
            f"Driehoek (ml): {b_tri:.2f} "
            f"Quagga (ml): {b_qua:.2f}"
        )

        folium.Marker(
            location=[lat, lon],
            icon=icon,
            tooltip=folium.Tooltip(tooltip_pie, sticky=True),
        ).add_to(fg_pies)

    # (3) Asvrijdrooggewicht (Bijlage 7) -> kruis bij NaN of 0
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

    # (4) Biovolumina kaartlaag
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

    # (5) PAS kaartlaag
    if kaartlaag.startswith("PAS"):
        shares, used_haps = _pas_distribution_for_row(row)

        if not shares:
            tooltip_none = f"{dg} – Locatie {loc_txt}<br>PAS: geen (bruikbare) data gevonden"
            folium.Marker(
                location=[lat, lon],
                icon=_make_cross_icon(size_px=18, color="#111111"),
                tooltip=folium.Tooltip(tooltip_none, sticky=True),
            ).add_to(fg_pas)
        else:
            svg = _pie_svg_multi(shares, PAS_COLORS, r=14)
            icon = folium.DivIcon(
                html=f"""
                <div style="width: 28px; height: 28px; transform: translate(-14px, -14px);">
                  {svg}
                </div>
                """
            )
            lines = [f"{PAS_LABELS.get(k, k)}: {frac:.0%}" for k, frac in sorted(shares.items(), key=lambda kv: kv[1], reverse=True)]
            tooltip_pas = (
                f"{dg} – Locatie {loc_txt}<br>"
                f"Bodemhappen met PAS-info: {int(used_haps)}/5<br>"
                + "<br>".join(lines)
            )
            folium.Marker(
                location=[lat, lon],
                icon=icon,
                tooltip=folium.Tooltip(tooltip_pas, sticky=True),
            ).add_to(fg_pas)

    # (6) Sedimenttype kaartlaag
    if kaartlaag == "Sedimenttype":
        shares, used_haps = _sed_distribution_for_row(row)

        if not shares:
            tooltip_none = f"{dg} – Locatie {loc_txt}<br>Sedimenttype: geen (bruikbare) data gevonden"
            folium.Marker(
                location=[lat, lon],
                icon=_make_cross_icon(size_px=18, color="#111111"),
                tooltip=folium.Tooltip(tooltip_none, sticky=True),
            ).add_to(fg_sed)
        else:
            svg = _pie_svg_multi(shares, SED_COLORS, r=14)
            icon = folium.DivIcon(
                html=f"""
                <div style="width: 28px; height: 28px; transform: translate(-14px, -14px);">
                  {svg}
                </div>
                """
            )
            lines = [f"{SED_LABELS.get(k, k)}: {frac:.0%}" for k, frac in sorted(shares.items(), key=lambda kv: kv[1], reverse=True)]
            tooltip_sed = (
                f"{dg} – Locatie {loc_txt}<br>"
                f"Bodemhappen met sedimentinfo: {int(used_haps)}/5<br>"
                + "<br>".join(lines)
            )
            folium.Marker(
                location=[lat, lon],
                icon=icon,
                tooltip=folium.Tooltip(tooltip_sed, sticky=True),
            ).add_to(fg_sed)

    # (7) Lutumgehalte kaartlaag
    if kaartlaag == "Lutumgehalte":
        shares, used_haps = _lut_distribution_for_row(row)

        if not shares:
            tooltip_none = f"{dg} – Locatie {loc_txt}<br>Lutum: geen (bruikbare) data gevonden"
            folium.Marker(
                location=[lat, lon],
                icon=_make_cross_icon(size_px=18, color="#111111"),
                tooltip=folium.Tooltip(tooltip_none, sticky=True),
            ).add_to(fg_lut)
        else:
            svg = _pie_svg_multi(shares, LUT_COLORS, r=14)
            icon = folium.DivIcon(
                html=f"""
                <div style="width: 28px; height: 28px; transform: translate(-14px, -14px);">
                  {svg}
                </div>
                """
            )
            lines = [f"{LUT_LABELS.get(k, k)}: {frac:.0%}" for k, frac in sorted(shares.items(), key=lambda kv: kv[1], reverse=True)]
            tooltip_lut = (
                f"{dg} – Locatie {loc_txt}<br>"
                f"Bodemhappen met lutuminfo: {int(used_haps)}/5<br>"
                + "<br>".join(lines)
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
    st.caption("Bollen tonen ADV uit Bijlage 7 (mg/m²): groter en donkerder groen = hogere waarde. Kruis = geen ADV-waarde of ADV=0 op die locatie.")
elif kaartlaag == "Biovolumina":
    if not (show_tri or show_qua):
        st.caption("Selecteer minimaal één soort (driehoeksmossel en/of quaggamossel) om biovolumina te tonen.")
    else:
        st.caption("Biovolumina: bollen per geselecteerde soort (ml). Kruis = 0 of geen waarde. Kleuren: blauw=driehoek, oranje=quagga.")
elif kaartlaag.startswith("PAS"):
    st.caption("PAS: taartdiagram toont verdeling van primair aanhechtingssubstraat over 5 bodemhappen. Bij meerdere substraten in één hap wordt die hap gelijk verdeeld.")
elif kaartlaag == "Sedimenttype":
    st.caption("Sedimenttype: taartdiagram toont verdeling over 5 bodemhappen. Bij meerdere typen in één hap wordt die hap gelijk verdeeld.")
else:
    st.caption("Lutumgehalte: taartdiagram toont verdeling van lutumklassen over 5 bodemhappen. Variatie tussen happen binnen één locatie wordt zichtbaar als meerdere taartsegmenten.")