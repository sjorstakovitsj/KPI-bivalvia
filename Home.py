# Home.py
import streamlit as st
import pandas as pd
import numpy as np

from utils import load_data, render_sidebar

st.set_page_config(page_title="Mosselkartering dashboard", layout="wide")

# -----------------------------
# Gedeelde sidebar (zelfde op alle pagina's)
# -----------------------------
ui = render_sidebar(title="Mosselkartering")
years_sel = list(ui.get("years", []))
combine_years = bool(ui.get("combine_years", False))
keep_only_canonical = bool(ui.get("keep_only_canonical", False))

st.title("📊 Mosselkartering – overzicht")

if not years_sel:
    st.warning("Selecteer in de sidebar ten minste één monitoringsjaar.")
    st.stop()


@st.cache_data(show_spinner=False)
def _load(years: tuple[str, ...], combine: bool, keep_only: bool):
    return load_data(years=list(years), combine_years=combine, keep_only_canonical=keep_only)


def _get_measurements(data: dict, years: list[str]) -> pd.DataFrame:
    """Maak één measurements dataframe, ook als utils.load_data() keys met suffix teruggeeft."""
    if "measurements" in data and isinstance(data["measurements"], pd.DataFrame) and not data["measurements"].empty:
        return data["measurements"].copy()

    # fallback: concat measurements_<year>
    frames: list[pd.DataFrame] = []
    for y in years:
        k = f"measurements_{y}"
        df = data.get(k)
        if isinstance(df, pd.DataFrame) and not df.empty:
            tmp = df.copy()
            if "jaar" not in tmp.columns:
                tmp["jaar"] = str(y)
            frames.append(tmp)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _ensure_measurement_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compat: maak kolommen consistent over jaargangen.

    Oplossingen:
    - Locatie kan numeriek of tekst (ZWRT_1, EEM77, ...). We behandelen hem als string-id.
    - Oudere jaargangen gebruiken Bugensis/Polymorpha i.p.v. Driehoeksmossel/Quaggamossel.
    - Waterdiepte kan 'Waterdiepte' of 'Waterdiepte (m)' heten.
    """
    out = df.copy()

    # Locatie als string id
    if "Locatie" in out.columns:
        out["Locatie_id"] = out["Locatie"].astype(str).str.strip()
    else:
        out["Locatie_id"] = pd.Series(["" for _ in range(len(out))])

    # Diepte
    if "diepte_m" not in out.columns:
        for cand in ["Waterdiepte (m)", "Waterdiepte", "Waterdiepte  (m)"]:
            if cand in out.columns:
                out["diepte_m"] = pd.to_numeric(out[cand], errors="coerce")
                break

    # Biovolumes
    # Doelkolommen: biovol_driehoek_ml, biovol_quagga_ml, biovol_totaal_ml
    if "biovol_driehoek_ml" not in out.columns:
        # 2023/2024: Driehoeksmossel
        if "Driehoeksmossel" in out.columns:
            out["biovol_driehoek_ml"] = pd.to_numeric(out["Driehoeksmossel"], errors="coerce")
        # 2021: Polymorpha (D. polymorpha = driehoek)
        elif "Polymorpha" in out.columns:
            out["biovol_driehoek_ml"] = pd.to_numeric(out["Polymorpha"], errors="coerce")

    if "biovol_quagga_ml" not in out.columns:
        # 2023/2024: Quaggamossel
        if "Quaggamossel" in out.columns:
            out["biovol_quagga_ml"] = pd.to_numeric(out["Quaggamossel"], errors="coerce")
        # 2021: Bugensis (D. bugensis = quagga)
        elif "Bugensis" in out.columns:
            out["biovol_quagga_ml"] = pd.to_numeric(out["Bugensis"], errors="coerce")

    if "biovol_totaal_ml" not in out.columns:
        if "biovol_driehoek_ml" in out.columns or "biovol_quagga_ml" in out.columns:
            a = out["biovol_driehoek_ml"] if "biovol_driehoek_ml" in out.columns else 0
            b = out["biovol_quagga_ml"] if "biovol_quagga_ml" in out.columns else 0
            out["biovol_totaal_ml"] = pd.to_numeric(a, errors="coerce").fillna(0) + pd.to_numeric(b, errors="coerce").fillna(0)

    # Ensure numeric types where applicable (niet Locatie!)
    for c in ["biovol_totaal_ml", "biovol_driehoek_ml", "biovol_quagga_ml", "diepte_m"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Deelgebied normaliseren gebeurt al in utils.load_data(); maar guard:
    if "Deelgebied" in out.columns:
        out["Deelgebied"] = out["Deelgebied"].astype(str).str.strip()

    return out


DATA = _load(tuple(years_sel), combine_years, keep_only_canonical)
meas_raw = _get_measurements(DATA, years_sel)

if meas_raw.empty:
    st.error(
        "Geen meetdata gevonden voor de geselecteerde jaren. "
        "Controleer of processed/<jaar>/measurements.parquet bestaat en of dataprep.py succesvol heeft gedraaid."
    )
    # Debug info
    with st.expander("Debug: beschikbare keys in DATA"):
        st.write(sorted(DATA.keys()))
    st.stop()

meas = _ensure_measurement_columns(meas_raw)

# Extra debug: laat kolommen zien als er iets ontbreekt
required_cols = ["Deelgebied", "Locatie_id", "biovol_totaal_ml", "biovol_driehoek_ml", "biovol_quagga_ml", "diepte_m"]
missing = [c for c in required_cols if c not in meas.columns]
if missing:
    st.warning(
        "Niet alle verwachte kolommen zijn aanwezig in deze jaargang(en). "
        f"Ontbrekend: {missing}. Ik probeer waar mogelijk alternatieve kolommen te gebruiken."
    )
    with st.expander("Debug: kolommen in measurements"):
        st.write(sorted(meas.columns.tolist()))

# -----------------------------
# 1) Eerst aggregeren naar unieke locaties (unit = locatie)
# - Als er meerdere meetmomenten per locatie zijn: neem gemiddelde per locatie
# - Bij multi-year: behandel (jaar, locatie) als unieke combinatie
# -----------------------------
if "Deelgebied" not in meas.columns:
    st.error("Kolom 'Deelgebied' ontbreekt in measurements. Kan niet aggregeren.")
    with st.expander("Debug: kolommen"):
        st.write(sorted(meas.columns.tolist()))
    st.stop()

# Grouping keys
keys = ["Deelgebied", "Locatie_id"]
if "jaar" in meas.columns:
    keys = ["jaar"] + keys

# Agg: gebruik alleen kolommen die bestaan (voorkomt KeyError)
agg_map = {}
for col in ["biovol_totaal_ml", "biovol_driehoek_ml", "biovol_quagga_ml", "diepte_m"]:
    if col in meas.columns:
        agg_map[col] = (col, "mean")

if not agg_map:
    st.error(
        "Geen biovolume/diepte kolommen beschikbaar om KPI's te berekenen voor deze selectie. "
        "Controleer Bijlage 4 kolomnamen of draai dataprep opnieuw."
    )
    with st.expander("Debug: voorbeelddata"):
        st.dataframe(meas.head(20))
    st.stop()

per_loc = meas.groupby(keys, as_index=False).agg(**agg_map)

# Ratio per locatie
if "biovol_quagga_ml" in per_loc.columns and "biovol_driehoek_ml" in per_loc.columns:
    per_loc["ratio_driehoek_quagga"] = np.where(
        per_loc["biovol_quagga_ml"] > 0,
        per_loc["biovol_driehoek_ml"] / per_loc["biovol_quagga_ml"],
        np.nan,
    )
else:
    per_loc["ratio_driehoek_quagga"] = np.nan

# Unieke locatie-telling
if "jaar" in per_loc.columns:
    n_unique_locations = (per_loc["jaar"].astype(str) + "_" + per_loc["Locatie_id"].astype(str)).nunique(dropna=True)
else:
    n_unique_locations = per_loc["Locatie_id"].nunique(dropna=True)

# -----------------------------
# KPI's
# -----------------------------
colk1, colk2, colk3, colk4 = st.columns(4)
with colk1:
    st.metric("Meetpunten (unieke locaties)", f"{n_unique_locations}")
with colk2:
    avg_biovol = per_loc["biovol_totaal_ml"].mean() if "biovol_totaal_ml" in per_loc.columns else np.nan
    st.metric("Gemiddeld biovolume per locatie (ml)", f"{avg_biovol:.1f}" if pd.notna(avg_biovol) else "n.v.t.")
with colk3:
    avg_ratio = per_loc["ratio_driehoek_quagga"].mean()
    st.metric("Gem. verhouding driehoek/quagga (per locatie)", f"{avg_ratio:.2f}" if pd.notna(avg_ratio) else "n.v.t.")
with colk4:
    avg_diepte = per_loc["diepte_m"].mean() if "diepte_m" in per_loc.columns else np.nan
    st.metric("Gem. diepte per locatie (m)", f"{avg_diepte:.2f}" if pd.notna(avg_diepte) else "n.v.t.")

# -----------------------------
# Tabel per deelgebied
# -----------------------------
st.subheader("Per deelgebied (op basis van unieke locaties)")

agg_keys = ["Deelgebied"]
if "jaar" in per_loc.columns:
    agg_keys = ["jaar"] + agg_keys

# Bouw aggregaties veilig
agg2 = {"meetpunten": ("Locatie_id", "nunique")}
if "biovol_totaal_ml" in per_loc.columns:
    agg2["gem_biovol_ml"] = ("biovol_totaal_ml", "mean")
if "ratio_driehoek_quagga" in per_loc.columns:
    agg2["gem_ratio_driehoek_quagga"] = ("ratio_driehoek_quagga", "mean")
if "diepte_m" in per_loc.columns:
    agg2["gem_diepte_m"] = ("diepte_m", "mean")

agg = per_loc.groupby(agg_keys, as_index=False).agg(**agg2)

# Afronden
show = agg.copy()
if "gem_biovol_ml" in show.columns:
    show["gem_biovol_ml"] = show["gem_biovol_ml"].round(2)
if "gem_ratio_driehoek_quagga" in show.columns:
    show["gem_ratio_driehoek_quagga"] = show["gem_ratio_driehoek_quagga"].round(3)
if "gem_diepte_m" in show.columns:
    show["gem_diepte_m"] = show["gem_diepte_m"].round(2)

# Sorteren
sort_cols = [c for c in ["jaar", "gem_biovol_ml"] if c in show.columns]
if sort_cols:
    ascending = [True] * len(sort_cols)
    if "gem_biovol_ml" in sort_cols:
        ascending[sort_cols.index("gem_biovol_ml")] = False
    show = show.sort_values(sort_cols, ascending=ascending)

st.dataframe(show, use_container_width=True)

st.caption(
    "Alle KPI’s en de tabel zijn berekend op basis van unieke locaties: per locatie worden (eventuele) meerdere meetmomenten eerst gemiddeld. "
    "Bij meerdere jaren worden locaties per jaar als uniek beschouwd (jaar+locatie)."
)
