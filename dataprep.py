# dataprep.py
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
XLSX = Path("velddata_mosselkartering_midden-nederland.xlsx")
OUTDIR = Path("processed")
OUTDIR.mkdir(exist_ok=True)

CANONICAL_DEELGEBIEDEN = {
    "Hoornse Hop",
    "IJmeer",
    "Markermeer Noord",
    "Markermeer Midden",
    "Markermeer Zuid",
}

# ------------------------------------------------------------
# Normalisatie helpers
# ------------------------------------------------------------

def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def normalize_deelgebied(val) -> str | None:
    if pd.isna(val):
        return None
    s = str(val).strip()
    k = _key(s)
    mapping = {
        "hoornsehop": "Hoornse Hop",
        "ijmeer": "IJmeer",
        "markermeernoord": "Markermeer Noord",
        "markermeermidden": "Markermeer Midden",
        "markeermeermidden": "Markermeer Midden",
        "markerneermidden": "Markermeer Midden",
        "markermeerzuid": "Markermeer Zuid",
    }
    if k in mapping:
        return mapping[k]
    if "ijmeer" in k:
        return "IJmeer"
    if "hoornsehop" in k:
        return "Hoornse Hop"
    if "markermeer" in k and "noord" in k:
        return "Markermeer Noord"
    if ("markermeer" in k and "midden" in k) or ("markeermeer" in k and "midden" in k) or (
        "markerneer" in k and "midden" in k
    ):
        return "Markermeer Midden"
    if "markermeer" in k and "zuid" in k:
        return "Markermeer Zuid"
    return s


def normalize_soort(val) -> str:
    if pd.isna(val):
        return "onbekend"
    s = str(val).strip().lower()
    if "bugensis" in s:
        return "D. bugensis (quagga)"
    if "polymorpha" in s:
        return "D. polymorpha (driehoek)"
    return str(val).strip()


def _clean_header_token(x) -> str:
    if x is None or pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower().startswith("unnamed"):
        return ""
    return s


def _has_lengteklasse_token(*tokens: str) -> bool:
    txt = " ".join([t for t in tokens if t]).lower()
    return "lengteklasse" in txt


# ------------------------------------------------------------
# RD -> WGS84 (benadering)
# ------------------------------------------------------------

def rd_to_wgs84(x: float, y: float) -> tuple[float, float]:
    p = (x - 155000.0) / 100000.0
    q = (y - 463000.0) / 100000.0

    K = [
        (0, 1, 3235.65389),
        (2, 0, -32.58297),
        (0, 2, -0.2475),
        (2, 1, -0.84978),
        (0, 3, -0.0655),
        (2, 2, -0.01709),
        (1, 0, -0.00738),
        (4, 0, 0.0053),
        (2, 3, -0.00039),
        (4, 1, 0.00033),
        (1, 1, -0.00012),
    ]
    L = [
        (1, 0, 5260.52916),
        (1, 1, 105.94684),
        (1, 2, 2.45656),
        (3, 0, -0.81885),
        (1, 3, 0.05594),
        (3, 1, -0.05607),
        (0, 1, 0.01199),
        (3, 2, -0.00256),
        (1, 4, 0.00128),
        (0, 2, 0.00022),
        (2, 0, -0.00022),
        (5, 0, 0.00026),
    ]

    phi = 52.15517440
    lam = 5.38720621

    for a, b, c in K:
        phi += c * (p**a) * (q**b) / 3600.0
    for a, b, c in L:
        lam += c * (p**a) * (q**b) / 3600.0

    return phi, lam


# ------------------------------------------------------------
# Bijlage 4 -> measurements.parquet
# ------------------------------------------------------------

def read_bijlage4_measurements() -> pd.DataFrame:
    df = pd.read_excel(XLSX, sheet_name="Bijlage 4", header=3, engine="openpyxl")

    df = df[df["Deelgebied"].notna()].copy()
    df["Deelgebied"] = df["Deelgebied"].apply(normalize_deelgebied)
    df = df[df["Deelgebied"].isin(CANONICAL_DEELGEBIEDEN)].copy()

    df = df.rename(
        columns={
            "x": "x_planned_rd",
            "y": "y_planned_rd",
            "x.1": "x_rd",
            "y.1": "y_rd",
            "Waterdiepte (m)": "diepte_m",
            "Driehoeksmossel": "biovol_driehoek_ml",
            "Quaggamossel": "biovol_quagga_ml",
            "Mytilidae": "aantallen_mytilidae",
            "Najaden": "aantallen_najaden",
        }
    )

    if "Corbicula" in df.columns:
        df = df.rename(columns={"Corbicula": "aantallen_corbicula"})
    if "Corbicula.1" in df.columns:
        df = df.rename(columns={"Corbicula.1": "aantallen_corbicula_dup"})

    for i in range(1, 6):
        if i in df.columns:
            df = df.rename(columns={i: f"sedimenttype_{i}"})
        if f"{i}.1" in df.columns:
            df = df.rename(columns={f"{i}.1": f"lutum_{i}"})
        if f"{i}.2" in df.columns:
            df = df.rename(columns={f"{i}.2": f"PAS_{i}"})

    df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce").dt.date

    num_cols = [
        "Locatie",
        "x_planned_rd",
        "y_planned_rd",
        "x_rd",
        "y_rd",
        "diepte_m",
        "biovol_driehoek_ml",
        "biovol_quagga_ml",
        "aantallen_mytilidae",
        "aantallen_corbicula",
        "aantallen_corbicula_dup",
        "aantallen_najaden",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["biovol_totaal_ml"] = df["biovol_driehoek_ml"].fillna(0) + df["biovol_quagga_ml"].fillna(0)
    df["ratio_driehoek_quagga"] = np.where(
        df["biovol_quagga_ml"] > 0,
        df["biovol_driehoek_ml"] / df["biovol_quagga_ml"],
        np.nan,
    )

    latlon = df[["x_rd", "y_rd"]].apply(
        lambda r: rd_to_wgs84(float(r["x_rd"]), float(r["y_rd"])),
        axis=1,
    )
    df["lat"] = [t[0] for t in latlon]
    df["lon"] = [t[1] for t in latlon]

    return df


# ------------------------------------------------------------
# Bijlage 7 -> adv_m2_locations.parquet (optioneel)
# ------------------------------------------------------------

def read_bijlage7_adv_m2() -> pd.DataFrame:
    df = pd.read_excel(XLSX, sheet_name="Bijlage 7", header=3, engine="openpyxl")

    df = df[df["Deelgebied"].notna()].copy()
    df["Deelgebied"] = df["Deelgebied"].apply(normalize_deelgebied)
    df = df[df["Deelgebied"].isin(CANONICAL_DEELGEBIEDEN)].copy()

    for c in ["Berekend ADV bugensis (mg/m2)", "Berekend ADV polymorpha (mg/m2)"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Locatie" in df.columns:
        df["Locatie"] = pd.to_numeric(df["Locatie"], errors="coerce")

    return df


# ------------------------------------------------------------
# Helpers: vind header-rij in sheet (robust)
# ------------------------------------------------------------

def _find_header_row(
    sheet: str,
    needle: str,
    *,
    prefer_first_col: bool = True,
    whole_word: bool = True,
    max_rows: int = 80,
) -> int | None:
    """Zoek de header-rij in een Excel-sheet.

    Robuustheid:
    - In Bijlage 5b komt het woord 'lengteklasse' ook voor in titelregels als 'lengteklassen'.
      Met `whole_word=True` matchen we alleen op het *woord* 'Lengteklasse' (\b...\b).
    - We zoeken eerst in de eerste kolom, omdat de echte header daar meestal staat.
    """

    raw = pd.read_excel(XLSX, sheet_name=sheet, header=None, engine="openpyxl", nrows=max_rows)

    patt = re.escape(needle)
    if whole_word:
        patt = rf"\b{patt}\b"

    if prefer_first_col and raw.shape[1] > 0:
        s0 = raw.iloc[:, 0].astype(str)
        hits0 = s0.str.contains(patt, case=False, na=False, regex=True)
        if hits0.any():
            return int(hits0.idxmax())

    for i in range(len(raw)):
        row = raw.iloc[i].astype(str)
        if row.str.contains(patt, case=False, na=False, regex=True).any():
            return int(i)

    return None


# ------------------------------------------------------------
# Populatie Bijlage 5a/5b -> tidy (ROBUST)
# ------------------------------------------------------------

def _ffill_multiindex_level0(cols: pd.MultiIndex) -> pd.MultiIndex:
    """Forward-fill het eerste level van een 2-level MultiIndex.

    Excel gebruikt vaak merged cells voor deelgebieden (bijv. 'Markermeer Noord' over 2 kolommen).
    Pandas leest dat als lege waarden/NaN in de vervolgkolommen. Door level-0 te ffill'en krijgen
    beide subkolommen weer hetzelfde deelgebied.
    """

    lvl0 = pd.Series(cols.get_level_values(0))
    lvl1 = pd.Series(cols.get_level_values(1))

    # maak lege tokens NaN zodat ffill werkt
    lvl0 = lvl0.replace(to_replace=r"^Unnamed:.*", value=np.nan, regex=True)
    lvl0 = lvl0.replace("", np.nan)
    lvl0 = lvl0.ffill().fillna("")

    return pd.MultiIndex.from_arrays([lvl0.values, lvl1.values])


def _parse_numeric_series(s: pd.Series) -> pd.Series:
    """Maak een kolom robuust numeriek.

    Ondersteunt o.a. waarden als '18,12%' of '18,12 %' (komma-decimaal en percent-teken).
    Als de kolom al numeriek is, wordt deze onveranderd teruggegeven.
    """

    if pd.api.types.is_numeric_dtype(s):
        return s

    ss = s.astype(str).str.strip()
    ss = ss.str.replace("\u00a0", " ", regex=False)  # non-breaking spaces
    ss = ss.str.replace("%", "", regex=False)
    ss = ss.str.replace(",", ".", regex=False)
    ss = ss.replace({"": np.nan, "nan": np.nan, "None": np.nan})

    return pd.to_numeric(ss, errors="coerce")


def read_populatie_tidy(sheet: str, metric: str) -> pd.DataFrame:
    """Robuust tidy maken voor Bijlage 5a/5b.

    Lost op:
    - Header-detectie: voorkomt dat titelregels met 'lengteklassen' per ongeluk als header worden gebruikt.
    - Merged cells: forward-fill van deelgebied in MultiIndex level 0.
    - Percentages als tekst: '18,12%' wordt numeriek.

    Output kolommen:
    - lengteklasse_mm (int)
    - deelgebied (canonical)
    - soort (canonical)
    - waarde (float)
    - metric ("count" of "percent")
    """

    header_row = _find_header_row(sheet, "Lengteklasse", prefer_first_col=True, whole_word=True)
    if header_row is None:
        return pd.DataFrame(columns=["lengteklasse_mm", "deelgebied", "soort", "waarde", "metric"])

    df = pd.read_excel(
        XLSX,
        sheet_name=sheet,
        header=[header_row, header_row + 1],
        engine="openpyxl",
    )

    df = df.dropna(axis=1, how="all")

    # Clean MultiIndex headers (strip + remove 'Unnamed')
    cleaned_cols: list[tuple[str, str]] = []
    for c0, c1 in df.columns:
        a = _clean_header_token(c0)
        b = _clean_header_token(c1)
        cleaned_cols.append((a, b))
    df.columns = pd.MultiIndex.from_tuples(cleaned_cols)

    # Herstel merged headers
    df.columns = _ffill_multiindex_level0(df.columns)

    # 1) Vind alle lengteklasse kolommen (kan >1 zijn door duplicaten)
    lengte_cols = [col for col in df.columns if _has_lengteklasse_token(col[0], col[1])]
    if not lengte_cols:
        lengte_cols = [col for col in df.columns if "lengteklasse" in str(col).lower()]
    if not lengte_cols:
        raise ValueError(
            f"Kon geen Lengteklasse-kolom vinden in sheet '{sheet}'. "
            f"Gevonden kolommen: {list(df.columns)}"
        )

    # 2) Pak een 1D Series uit (kan DataFrame zijn bij duplicate MultiIndex keys)
    lk_sel = df[lengte_cols[0]]
    if isinstance(lk_sel, pd.DataFrame):
        chosen = None
        for j in range(lk_sel.shape[1]):
            cand = lk_sel.iloc[:, j]
            if cand.notna().any():
                chosen = cand
                break
        lk_series = chosen if chosen is not None else lk_sel.iloc[:, 0]
    else:
        lk_series = lk_sel

    lk = pd.to_numeric(lk_series, errors="coerce")
    keep = lk.notna()
    df = df.loc[keep].copy()
    lk = lk.loc[keep].astype(int)

    # drop alle lengteklasse kolommen (ook duplicaten)
    df = df.drop(columns=lengte_cols, errors="ignore")

    # 3) Zet overige kolommen om naar tidy
    parts: list[pd.DataFrame] = []
    for (h0, h1) in df.columns:
        deelgebied = None
        soort = None

        dg0 = normalize_deelgebied(h0)
        dg1 = normalize_deelgebied(h1)

        if dg0 in CANONICAL_DEELGEBIEDEN and ("bugensis" in str(h1).lower() or "polymorpha" in str(h1).lower()):
            deelgebied = dg0
            soort = normalize_soort(h1)
        elif dg1 in CANONICAL_DEELGEBIEDEN and ("bugensis" in str(h0).lower() or "polymorpha" in str(h0).lower()):
            deelgebied = dg1
            soort = normalize_soort(h0)
        else:
            # fallback: zoek tokens in beide velden
            if dg0 in CANONICAL_DEELGEBIEDEN:
                deelgebied = dg0
            elif dg1 in CANONICAL_DEELGEBIEDEN:
                deelgebied = dg1

            token = (str(h0).lower() + " " + str(h1).lower())
            if "bugensis" in token:
                soort = "D. bugensis (quagga)"
            elif "polymorpha" in token:
                soort = "D. polymorpha (driehoek)"

        if deelgebied is None or soort is None:
            continue
        if deelgebied not in CANONICAL_DEELGEBIEDEN:
            continue

        values = _parse_numeric_series(df[(h0, h1)])

        parts.append(
            pd.DataFrame(
                {
                    "lengteklasse_mm": lk.values,
                    "deelgebied": deelgebied,
                    "soort": soort,
                    "waarde": values.values,
                    "metric": metric,
                }
            )
        )

    if not parts:
        return pd.DataFrame(columns=["lengteklasse_mm", "deelgebied", "soort", "waarde", "metric"])

    out = pd.concat(parts, ignore_index=True)
    out = out[out["waarde"].notna()].copy()

    return out


# ------------------------------------------------------------
# Bijlage 6 -> adv_lenclass.parquet (tidy/long)
# ------------------------------------------------------------

def read_bijlage6_adv_lenclass_tidy() -> pd.DataFrame:
    header_row = _find_header_row("Bijlage 6", "SL", prefer_first_col=False, whole_word=False)
    if header_row is None:
        return pd.DataFrame(columns=["sl_mm", "deelgebied", "soort", "adv_mg_per_mossel", "n_verast"])

    df = pd.read_excel(
        XLSX,
        sheet_name="Bijlage 6",
        header=[header_row, header_row + 1, header_row + 2],
        engine="openpyxl",
    )

    cleaned = []
    for c0, c1, c2 in df.columns:
        a = _clean_header_token(c0)
        b = _clean_header_token(c1)
        c = _clean_header_token(c2)
        if "sl" in (a + " " + b + " " + c).lower():
            cleaned.append(("SL", "SL", "SL"))
        else:
            cleaned.append((a, b, c))
    df.columns = pd.MultiIndex.from_tuples(cleaned)

    sl_col = ("SL", "SL", "SL")
    sl = pd.to_numeric(df[sl_col], errors="coerce")
    keep = sl.notna()
    df = df.loc[keep].copy()
    sl = sl.loc[keep].astype(int)

    cols = [c for c in df.columns if c != sl_col]

    pair_map: dict[tuple[str, str], dict[str, tuple[str, str, str]]] = {}
    for dg_raw, soort_raw, metric_raw in cols:
        dg = normalize_deelgebied(dg_raw)
        if dg is None or dg not in CANONICAL_DEELGEBIEDEN:
            continue
        soort = normalize_soort(soort_raw)
        metric_name = str(metric_raw).strip().lower()

        pair_map.setdefault((dg, soort), {})
        if "adv" in metric_name:
            pair_map[(dg, soort)]["adv"] = (dg_raw, soort_raw, metric_raw)
        if metric_name.startswith("n") or metric_name == "n":
            pair_map[(dg, soort)]["n"] = (dg_raw, soort_raw, metric_raw)

    records = []
    for (dg, soort), maps in pair_map.items():
        if "adv" not in maps:
            continue
        adv_col = maps["adv"]
        n_col = maps.get("n")

        adv_vals = pd.to_numeric(df[adv_col], errors="coerce")
        n_vals = pd.to_numeric(df[n_col], errors="coerce") if n_col else pd.Series([np.nan] * len(df), index=df.index)

        records.append(
            pd.DataFrame(
                {
                    "sl_mm": sl.values,
                    "deelgebied": dg,
                    "soort": soort,
                    "adv_mg_per_mossel": adv_vals.values,
                    "n_verast": n_vals.values,
                }
            )
        )

    if not records:
        return pd.DataFrame(columns=["sl_mm", "deelgebied", "soort", "adv_mg_per_mossel", "n_verast"])

    out = pd.concat(records, ignore_index=True)
    out = out[out["adv_mg_per_mossel"].notna()].copy()

    return out


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    if not XLSX.exists():
        raise FileNotFoundError(f"Excelbestand niet gevonden: {XLSX.resolve()}")

    measurements = read_bijlage4_measurements()
    measurements.to_parquet(OUTDIR / "measurements.parquet", index=False)

    adv_m2 = read_bijlage7_adv_m2()
    adv_m2.to_parquet(OUTDIR / "adv_m2_locations.parquet", index=False)

    pop_counts = read_populatie_tidy("Bijlage 5a", metric="count")
    pop_counts.to_parquet(OUTDIR / "populatie_counts.parquet", index=False)

    pop_percent = read_populatie_tidy("Bijlage 5b", metric="percent")
    pop_percent.to_parquet(OUTDIR / "populatie_percent.parquet", index=False)

    adv_len = read_bijlage6_adv_lenclass_tidy()
    adv_len.to_parquet(OUTDIR / "adv_lenclass.parquet", index=False)

    print("✅ Parquet-bestanden geschreven naar:", OUTDIR.resolve())
    print("✅ Deelgebieden measurements:", sorted(measurements["Deelgebied"].unique().tolist()))
    print(
        "✅ Deelgebieden populatie_counts:",
        sorted(pop_counts["deelgebied"].unique().tolist()) if not pop_counts.empty else [],
    )
    print(
        "✅ Deelgebieden populatie_percent:",
        sorted(pop_percent["deelgebied"].unique().tolist()) if not pop_percent.empty else [],
    )
    print("✅ adv_lenclass kolommen:", list(adv_len.columns))


if __name__ == "__main__":
    main()