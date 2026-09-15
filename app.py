"""
================================================================================
 app.py -- Streamlit front end for gas condensate decline curve analysis
================================================================================

Run locally:
    streamlit run app.py

Deploy on Streamlit Cloud: point the app at this file and make sure
requirements.txt sits beside it at the repository root.

Layout
------
  sidebar : fluid definition (PVT), CVD table, processing split, run settings
  tabs    : Data & QC -> Decline fit -> Yield -> Material balance ->
            Forecast -> Uncertainty -> Field -> Export

The heavy lifting all lives in `gas_condensate_dca.py`; this file is intake,
state and presentation. Expensive steps are cached on their inputs so moving a
slider does not re-run the whole field.
================================================================================
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import sys
import traceback
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

# -- local module import ------------------------------------------------------
# Streamlit Cloud redacts import errors, so a missing sibling file shows up as
# an unexplained ModuleNotFoundError. Look in the obvious places first, then
# fail with a message that actually says what is missing and what is present.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _candidate in (_HERE, os.path.join(_HERE, "streamlit_app"),
                   os.path.join(_HERE, "src"),
                   os.path.dirname(_HERE)):
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

_import_error: Optional[BaseException] = None
try:
    import gas_condensate_dca as dca
    import dca_charts as ch
except BaseException as _exc:       # noqa: BLE001 - reported to the user below
    _import_error = _exc
    dca = ch = None                 # type: ignore[assignment]

# ==============================================================================
# Page setup
# ==============================================================================

st.set_page_config(
    page_title="Gas Condensate DCA",
    page_icon=":material/show_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

if _import_error is not None:
    missing = getattr(_import_error, "name", None) or "a required module"
    st.error(f"Could not import **{missing}**.")
    if missing in ("gas_condensate_dca", "dca_charts"):
        st.markdown(
            f"`{missing}.py` has to sit in the same folder as `app.py`. "
            "The files below are what the app can actually see."
        )
    else:
        st.markdown(
            f"`{missing}` is a third-party package, so it belongs in "
            "`requirements.txt` at the **repository root**. Streamlit Cloud "
            "installs nothing beyond its base image on its own. After adding "
            "it, use **Manage app -> Reboot app**."
        )
    try:
        listing = "\n".join(sorted(
            f"{'DIR ' if os.path.isdir(os.path.join(_HERE, n)) else '    '}{n}"
            for n in os.listdir(_HERE)))
    except Exception:
        listing = "(could not list the application directory)"
    st.code(f"{_HERE}\n\n{listing}", language="text")
    with st.expander("Full traceback"):
        st.code("".join(traceback.format_exception(
            type(_import_error), _import_error,
            _import_error.__traceback__)), language="text")
    st.stop()


def current_theme() -> str:
    """Which theme the viewer is actually in, across Streamlit versions."""
    try:
        t = st.context.theme.type            # Streamlit >= 1.46
        if t in ("light", "dark"):
            return t
    except Exception:
        pass
    try:
        base = st.get_option("theme.base")
        if base in ("light", "dark"):
            return base
    except Exception:
        pass
    return "light"


def show_fig(fig, key: Optional[str] = None) -> None:
    """st.plotly_chart across the width-API change in Streamlit 1.49."""
    try:
        st.plotly_chart(fig, width="stretch", theme=None, key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, theme=None, key=key)


def show_df(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


THEME = current_theme()
C = ch.palette(THEME)

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; }}
      div[data-testid="stMetricValue"] {{ font-size: 1.45rem; }}
      .dca-note {{
          color: {C['ink2']}; font-size: 0.86rem; line-height: 1.45;
          border-left: 3px solid {C['grid']}; padding: 0.15rem 0 0.15rem 0.7rem;
          margin: 0.35rem 0 0.9rem 0;
      }}
      .dca-warn {{
          color: {C['critical']}; font-size: 0.86rem; line-height: 1.45;
          border-left: 3px solid {C['critical']};
          padding: 0.15rem 0 0.15rem 0.7rem; margin: 0.35rem 0 0.9rem 0;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


def note(text: str) -> None:
    st.markdown(f'<div class="dca-note">{text}</div>', unsafe_allow_html=True)


def warn(text: str) -> None:
    st.markdown(f'<div class="dca-warn">{text}</div>', unsafe_allow_html=True)


# ==============================================================================
# Cached compute
# ==============================================================================

DEFAULT_CVD = pd.DataFrame({
    "pressure_psia": [5100.0, 4500.0, 3800.0, 3100.0, 2400.0, 1800.0,
                      1200.0, 700.0],
    "cum_produced_molfrac": [0.000, 0.078, 0.176, 0.283, 0.402, 0.515,
                             0.641, 0.757],
    "liquid_dropout_frac": [0.000, 0.081, 0.134, 0.152, 0.146, 0.131,
                            0.108, 0.086],
})


PASTE_ROWS = 14
PASTE_COLUMNS = ["date", "well", "days_on", "q_gas", "q_cond", "q_water",
                 "p_wf", "p_res"]


def blank_paste_frame(n: int = PASTE_ROWS) -> pd.DataFrame:
    """An empty grid with the canonical headers, ready to paste into."""
    return pd.DataFrame({
        "date": pd.Series([""] * n, dtype="object"),
        "well": pd.Series([""] * n, dtype="object"),
        **{c: pd.Series([np.nan] * n, dtype="float64")
           for c in PASTE_COLUMNS[2:]},
    })


def paste_column_config() -> dict:
    """Headers carry their units, so a pasted column cannot be misread."""
    num = st.column_config.NumberColumn
    return {
        "date": st.column_config.TextColumn(
            "date", help="Any parseable date: 2021-03-01, 01/03/2021, Mar-2021",
            width="small"),
        "well": st.column_config.TextColumn(
            "well", help="Leave blank to treat the whole table as one entity",
            width="small"),
        "days_on": num("days_on", help="Producing days in the period",
                       format="%.1f"),
        "q_gas": num("q_gas (Mscf/d)", help="Separator gas rate — required",
                     format="%.1f"),
        "q_cond": num("q_cond (STB/d)", help="Condensate rate", format="%.2f"),
        "q_water": num("q_water (STB/d)", format="%.1f"),
        "p_wf": num("p_wf (psia)", help="Flowing bottomhole pressure",
                    format="%.0f"),
        "p_res": num("p_res (psia)", help="Average reservoir pressure",
                     format="%.0f"),
    }


def align_to_paste_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fit an arbitrary table onto the grid's columns, keeping the dtypes.

    Column names go through the same alias matcher the file loader uses, so a
    block headed `Gas Rate (Mscf/d)` lands in `q_gas` rather than being lost.
    """
    d = dca.map_columns(df).copy()
    out = pd.DataFrame(index=range(len(d)))
    for c in PASTE_COLUMNS:
        if c in ("date", "well"):
            out[c] = (d[c].astype(str) if c in d.columns
                      else pd.Series([""] * len(d), dtype="object"))
            if c == "date" and c in d.columns:
                parsed = pd.to_datetime(d[c], errors="coerce", format="mixed")
                out[c] = parsed.dt.strftime("%Y-%m-%d").fillna(
                    d[c].astype(str))
        else:
            # Always float: an int column cannot hold the blanks the editor
            # writes back for empty cells.
            out[c] = (pd.to_numeric(d[c], errors="coerce").astype("float64")
                      if c in d.columns
                      else pd.Series([np.nan] * len(d), dtype="float64"))
    return out.reset_index(drop=True)


def parse_paste_grid(edited: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Turn the edited grid into a production table, reporting what was lost."""
    messages: list[str] = []
    d = edited.copy()
    for c in PASTE_COLUMNS:
        if c not in d.columns:
            d[c] = np.nan
    d["date"] = d["date"].astype(str).str.strip().replace(
        {"": np.nan, "nan": np.nan, "None": np.nan, "NaT": np.nan})
    d["well"] = d["well"].astype(str).str.strip().replace(
        {"": np.nan, "nan": np.nan, "None": np.nan})
    for c in PASTE_COLUMNS[2:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=["date", "q_gas"], how="all")
    n_rows = len(d)
    if n_rows == 0:
        return pd.DataFrame(), messages

    parsed = pd.to_datetime(d["date"], errors="coerce", format="mixed")
    bad_date = parsed.isna()
    if bad_date.any():
        messages.append(f"{int(bad_date.sum())} row(s) dropped: the date could "
                        "not be read.")
    d["date"] = parsed

    bad_gas = d["q_gas"].isna() | (d["q_gas"] <= 0)
    if bad_gas.any():
        messages.append(f"{int(bad_gas.sum())} row(s) dropped: no positive gas "
                        "rate.")

    d = d[~bad_date & ~bad_gas]
    # Drop columns that were left entirely blank so they do not clutter the
    # mapping UI or masquerade as supplied-but-missing data.
    for c in PASTE_COLUMNS[1:]:
        if c in d.columns and d[c].isna().all():
            d = d.drop(columns=[c])
    return d.reset_index(drop=True), messages


@st.cache_resource(show_spinner=False)
def build_pvt(gas_gravity: float, temperature_F: float, condensate_api: float,
              condensate_mw: Optional[float], p_dew: Optional[float],
              p_init: Optional[float], y_n2: float, y_co2: float, y_h2s: float,
              initial_cgr: Optional[float], use_wellstream: bool,
              cvd_key: Optional[tuple]) -> dca.PVT:
    """PVT objects pre-compute property tables, so they are cached by value."""
    cvd = None
    if cvd_key:
        p, n = zip(*cvd_key)
        cvd = dca.CVDTable(pressure=np.array(p),
                           cum_produced_molfrac=np.array(n))
    return dca.PVT(gas_gravity=gas_gravity, temperature_F=temperature_F,
                   condensate_api=condensate_api, condensate_mw=condensate_mw,
                   p_dew=p_dew, p_init=p_init, y_n2=y_n2, y_co2=y_co2,
                   y_h2s=y_h2s, cvd=cvd, initial_cgr=initial_cgr,
                   use_wellstream_gravity=use_wellstream)


@st.cache_data(show_spinner=False)
def make_demo_field(n_wells: int, seed: int, pvt_sig: tuple,
                    _pvt: dca.PVT) -> pd.DataFrame:
    # pvt_sig is in the cache key so changing the fluid regenerates the field;
    # _pvt itself is excluded from hashing by the leading underscore.
    return dca.make_synthetic_field(_pvt, n_wells=n_wells, seed=seed)


@st.cache_data(show_spinner=False)
def read_upload(data: bytes, name: str) -> pd.DataFrame:
    buf = io.BytesIO(data)
    if name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(buf)
    return pd.read_csv(buf)


def _sniff_separator(text: str) -> Optional[str]:
    """Pick the delimiter that appears the same number of times on every line.

    Pasted data arrives from wherever the engineer copied it: Excel gives
    tabs, an export gives commas, a European locale gives semicolons, a
    terminal dump gives runs of spaces. Guessing from the header alone is
    unreliable - a header can contain a comma inside a unit label - so the
    delimiter has to be consistent down the block to be believed.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()][:15]
    if len(lines) < 2:
        return None
    best, best_count = None, 0
    for sep in ("\t", ",", ";", "|"):
        counts = {ln.count(sep) for ln in lines}
        if len(counts) == 1 and counts != {0} and counts.pop() > best_count:
            best, best_count = sep, max(ln.count(sep) for ln in lines)
    return best


@st.cache_data(show_spinner=False)
def parse_pasted_text(text: str) -> pd.DataFrame:
    """Turn a pasted text block into a dataframe, or explain why it could not.

    Used to seed the paste grid from a text blob, and as the fallback for
    people who would rather paste a whole block including its header row than
    fill a grid.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n ")
    if not text:
        raise ValueError("Nothing was pasted.")
    if len(text.splitlines()) < 3:
        raise ValueError("Paste a header row and at least two data rows.")

    sep = _sniff_separator(text)
    attempts = []
    if sep:
        # A decimal comma only makes sense when the comma is not the delimiter.
        attempts.append({"sep": sep, "decimal": "."})
        if sep != ",":
            attempts.append({"sep": sep, "decimal": ",", "thousands": None})
    attempts.append({"sep": r"\s+", "decimal": "."})

    last = None
    for kw in attempts:
        try:
            df = pd.read_csv(io.StringIO(text), engine="python",
                             skipinitialspace=True, **kw)
        except Exception as exc:
            last = exc
            continue
        if df.shape[1] < 2 or len(df) < 2:
            continue
        df.columns = [str(c).strip() for c in df.columns]
        # A block of prose splits happily on whitespace into something shaped
        # like a table, so require at least one genuinely numeric column
        # before believing the parse.
        numeric_cols = sum(
            pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.6
            for c in df.columns)
        if numeric_cols == 0:
            continue
        probe = dca.map_columns(df)
        # And if a gas column was recognised, it has to hold numbers.
        if "q_gas" in probe.columns:
            q = pd.to_numeric(probe["q_gas"], errors="coerce")
            if q.notna().mean() < 0.5:
                continue
        return df

    raise ValueError(
        "Could not read that as a table. Expected a header row and one row "
        "per period, separated by tabs, commas or semicolons."
        + (f" (pandas said: {last})" if last else ""))


@st.cache_data(show_spinner=False, max_entries=8)
def run_analysis(df: pd.DataFrame, _pvt: dca.PVT, pvt_sig: tuple,
                 settings: tuple, products: tuple, well_col: Optional[str]):
    """Analyse every well. Keyed on the dataframe, PVT signature and settings."""
    (q_econ, model, terminal, t_max, run_mc, n_mc, use_mb, use_fmb, cap,
     rate_basis, min_uptime, outlier_sigma, fit_from_bdf, window) = settings
    split = dca.ProductSplit(inert_fraction=products[0],
                             fuel_flare_fraction=products[1],
                             ngl_yield_gal_per_mscf=products[2],
                             ngl_shrinkage_fraction=products[3])

    raw = dca.map_columns(df)
    groups = ({str(k): v for k, v in raw.groupby(well_col)}
              if well_col and well_col in raw.columns else {"FIELD": raw})

    prepared, prep_errors = {}, {}
    for name, grp in groups.items():
        try:
            prepared[name] = dca.ProductionData.prepare(
                grp, _pvt, well=name, rate_basis=rate_basis,
                min_uptime_frac=min_uptime,
                outlier_sigma=(None if outlier_sigma <= 0 else outlier_sigma))
        except Exception as exc:
            prep_errors[name] = str(exc)

    results, errors = {}, dict(prep_errors)
    for name, pdata in prepared.items():
        try:
            results[name] = dca.analyse_well(
                pdata, _pvt, well=name, q_econ_mscfd=q_econ, select=model,
                terminal_decline_pct_yr=terminal, t_max_years=t_max,
                products=split, run_monte_carlo=run_mc, n_mc=n_mc,
                use_material_balance=use_mb, use_fmb=use_fmb,
                apply_ogip_cap=cap, fit_from_bdf=fit_from_bdf,
                fit_window_days=window, verbose=False)
        except Exception as exc:
            errors[name] = str(exc)

    rows = []
    for name, r in results.items():
        row = {
            "well": name,
            "model": r.settings["selected_model"],
            "points_fitted": r.best_fit.n_points,
            "R2_log": r.best_fit.r2,
            "b": r.best_fit.params.get("b", np.nan),
            "Di_pct_yr": (100 * (1 - math.exp(-r.best_fit.params["Di"]
                                              * dca.DAYS_PER_YEAR))
                          if "Di" in r.best_fit.params else np.nan),
            "Gp_to_date_mmscf": float(r.data.Gp_ws[-1]),
            "Np_cond_to_date_mstb": float(r.data.Np_cond[-1]),
            "EUR_wellstream_mmscf": r.forecast.eur_wellstream_mmscf,
            "EUR_sales_gas_mmscf": r.forecast.eur_sales_gas_mmscf,
            "EUR_condensate_mstb": r.forecast.eur_condensate_mstb,
            "EUR_ngl_mstb": r.forecast.eur_ngl_mstb,
            "remaining_gas_mmscf": r.forecast.remaining_wellstream_mmscf,
            "life_yr": r.forecast.economic_life_years,
            "OGIP_matbal_mmscf": (r.matbal.ogip_mmscf if r.matbal else np.nan),
        }
        if r.mc_stats:
            for k, s in r.mc_stats.items():
                tag = k.split("(")[0].strip().replace(" ", "_")
                row[f"{tag}_P90"], row[f"{tag}_P50"], row[f"{tag}_P10"] = (
                    s["P90"], s["P50"], s["P10"])
        rows.append(row)
    summary = (pd.DataFrame(rows).sort_values("EUR_wellstream_mmscf",
                                              ascending=False)
               .reset_index(drop=True) if rows else pd.DataFrame())
    profile = dca.field_profile(results) if results else pd.DataFrame()
    return results, summary, profile, errors


# ==============================================================================
# Sidebar -- fluid, processing, run settings
# ==============================================================================

with st.sidebar:
    st.markdown("### Fluid definition")
    note("Pseudo-pressure and the two-phase z-factor are built from these. "
         "Get the dew point and initial CGR right before anything else.")

    c1, c2 = st.columns(2)
    gas_gravity = c1.number_input("Gas gravity (air=1)", 0.55, 1.40, 0.72,
                                  0.01, format="%.3f")
    temperature_F = c2.number_input("Reservoir temp (deg F)", 60.0, 450.0,
                                    248.0, 1.0)
    c1, c2 = st.columns(2)
    condensate_api = c1.number_input("Condensate API", 30.0, 75.0, 52.0, 0.5)
    mw_override = c2.number_input("Condensate MW (0 = Standing)", 0.0, 300.0,
                                  0.0, 1.0)
    c1, c2 = st.columns(2)
    p_init = c1.number_input("Initial pressure (psia)", 200.0, 20000.0, 6400.0,
                             50.0)
    p_dew = c2.number_input("Dew point (psia)", 0.0, 20000.0, 5100.0, 50.0)
    initial_cgr = st.number_input("Initial CGR (STB/MMscf)", 0.0, 500.0, 78.0,
                                  1.0)
    use_wellstream = st.toggle(
        "Use wellstream gravity for PVT", value=True,
        help="Correct above the dew point: the reservoir flows one phase, so "
             "properties should use the wet-gas gravity, not the separator "
             "gas gravity.")

    with st.expander("Inerts"):
        y_n2 = st.number_input("N2 mole fraction", 0.0, 0.5, 0.012, 0.001,
                               format="%.3f")
        y_co2 = st.number_input("CO2 mole fraction", 0.0, 0.5, 0.031, 0.001,
                                format="%.3f")
        y_h2s = st.number_input("H2S mole fraction", 0.0, 0.3, 0.0, 0.001,
                                format="%.3f")

    with st.expander("CVD table (two-phase z)"):
        note("From the lab CVD report. Without it the Rayes correlation is "
             "used, which is a regression rather than your fluid.")
        use_cvd = st.toggle("Use CVD table", value=True)
        cvd_df = st.data_editor(DEFAULT_CVD, num_rows="dynamic",
                                key="cvd_editor",
                                disabled=not use_cvd)

    st.divider()
    st.markdown("### Surface processing")
    c1, c2 = st.columns(2)
    inert_frac = c1.number_input("Inerts removed", 0.0, 0.5, 0.043, 0.001,
                                 format="%.3f")
    fuel_frac = c2.number_input("Fuel & flare", 0.0, 0.3, 0.020, 0.001,
                                format="%.3f")
    c1, c2 = st.columns(2)
    ngl_yield = c1.number_input("NGL yield (gal/Mscf)", 0.0, 10.0, 1.8, 0.1)
    ngl_shrink = c2.number_input("NGL shrinkage", 0.0, 0.3, 0.03, 0.005,
                                 format="%.3f")

    st.divider()
    st.markdown("### Analysis settings")
    q_econ = st.number_input("Economic limit (Mscf/d)", 1.0, 100000.0, 400.0,
                             25.0)
    model_choice = st.selectbox(
        "Decline model", ["modified_hyperbolic", "arps", "ple", "sepd",
                          "duong", "auto"], index=0,
        help="Modified hyperbolic is the default because it is the one with a "
             "defensible late-time limit. 'auto' picks the lowest AIC, which "
             "is a statistical choice, not a physical one.")
    terminal = st.slider("Terminal decline (%/yr)", 2.0, 25.0, 7.0, 0.5)
    t_max = st.slider("Max forecast life (years)", 5, 60, 40, 1)

    with st.expander("QC thresholds"):
        rate_basis = st.radio("Rate basis", ["stream-day", "calendar-day"],
                              horizontal=True,
                              help="Mixing the two across a history is the "
                                   "commonest source of fake hyperbolic "
                                   "curvature.")
        min_uptime = st.slider("Min uptime fraction", 0.0, 0.9, 0.35, 0.05)
        outlier_sigma = st.slider("Outlier rejection (MAD sigma, 0 = off)",
                                  0.0, 8.0, 4.0, 0.5)

    with st.expander("Cross-checks and uncertainty"):
        use_mb = st.toggle("Material balance (needs p_res)", value=True)
        use_fmb = st.toggle("Flowing material balance (needs p_wf)",
                            value=False,
                            help="Biased low on a condensate well below the "
                                 "dew point - read it as a lower bound.")
        cap_ogip = st.toggle("Cap forecast at material-balance OGIP",
                             value=True)
        run_mc = st.toggle("Run Monte Carlo", value=True)
        n_mc = st.slider("Monte Carlo realisations", 200, 4000, 800, 100,
                         disabled=not run_mc)

# ==============================================================================
# PVT object
# ==============================================================================

cvd_key = None
if use_cvd and cvd_df is not None and len(cvd_df.dropna()) >= 3:
    clean = cvd_df.dropna(subset=["pressure_psia", "cum_produced_molfrac"])
    cvd_key = tuple(zip(clean["pressure_psia"].astype(float),
                        clean["cum_produced_molfrac"].astype(float)))

try:
    pvt = build_pvt(gas_gravity, temperature_F, condensate_api,
                    (mw_override or None), (p_dew or None), (p_init or None),
                    y_n2, y_co2, y_h2s, (initial_cgr or None),
                    use_wellstream, cvd_key)
except Exception as exc:
    st.error(f"The fluid definition is not valid: {exc}")
    st.stop()

pvt_sig = (gas_gravity, temperature_F, condensate_api, mw_override, p_dew,
           p_init, y_n2, y_co2, y_h2s, initial_cgr, use_wellstream, cvd_key)

# ==============================================================================
# Header and data intake
# ==============================================================================

st.title("Gas condensate decline curve analysis")
note("Everything is fitted on a <b>wellstream (gas-equivalent)</b> basis, then "
     "re-split into products, so the gas and condensate forecasts can never "
     "drift apart. Below the dew point the material balance uses a "
     "<b>two-phase z-factor</b>.")

# A short, physically consistent history (plateau then decline, pressure
# surveys every six months) so the sample actually analyses rather than
# failing on too few points. Material balance recovers its true 46,000
# MMscf to within 1%.
SAMPLE_PASTE = """date	well	days_on	q_gas	q_cond	p_wf	p_res
2021-01-01	A-1	31	27585	2301	1408	6400
2021-02-01	A-1	9	28731	2349	1405	
2021-03-01	A-1	28	29297	2441	1427	
2021-04-01	A-1	19	26744	2092	1356	
2021-05-01	A-1	30	29578	2468	1394	
2021-06-01	A-1	31	28736	2321	1395	
2021-07-01	A-1	30	29768	2252	1358	5212
2021-08-01	A-1	31	28767	2322	1431	
2021-09-01	A-1	31	30440	2321	1406	
2021-10-01	A-1	30	26957	1884	1388	
2021-11-01	A-1	25	31807	2105	1389	
2021-12-01	A-1	30	30836	1893	1406	
2022-01-01	A-1	31	27584	1822	1348	4178
2022-02-01	A-1	31	29883	1786	1419	
2022-03-01	A-1	28	29659	1549	1409	
2022-04-01	A-1	31	25972	1404	1387	
2022-05-01	A-1	30	29270	1484	1361	
2022-06-01	A-1	31	24887	1370	1467	
2022-07-01	A-1	11	34937	1868	1392	3304
2022-08-01	A-1	31	21924	1075	1411	
2022-09-01	A-1	31	20526	929	1393	
2022-10-01	A-1	30	19665	936	1388	
2022-11-01	A-1	31	18451	767	1365	
2022-12-01	A-1	11	26226	1193	1391	
2023-01-01	A-1	31	16228	709	1438	2816
2023-02-01	A-1	31	14270	627	1380	
2023-03-01	A-1	28	13534	583	1462	
2023-04-01	A-1	31	13532	546	1451	
2023-05-01	A-1	30	13605	620	1396	
2023-06-01	A-1	31	12632	519	1440	
2023-07-01	A-1	30	11031	463	1385	2439
2023-08-01	A-1	31	10363	468	1405	
2023-09-01	A-1	31	9800	406	1401	
2023-10-01	A-1	30	9880	394	1403	
2023-11-01	A-1	31	9828	388	1414	
2023-12-01	A-1	30	8802	371	1431	
2024-01-01	A-1	31	8705	367	1443	2192
2024-02-01	A-1	31	8081	332	1383	
2024-03-01	A-1	29	8241	340	1355	
2024-04-01	A-1	31	6377	240	1408	
2024-05-01	A-1	30	6071	217	1442	
2024-06-01	A-1	31	6515	239	1359	"""

src_col, a_col, b_col = st.columns([1.1, 1, 2.2])
with src_col:
    source = st.radio("Data source",
                      ["Example field", "Upload file", "Paste data"],
                      horizontal=False, label_visibility="collapsed")

upload = None
n_wells, seed = 4, 5
if source == "Upload file":
    with b_col:
        upload = st.file_uploader("Production history (CSV or Excel)",
                                  type=["csv", "xlsx", "xlsm", "xls"],
                                  label_visibility="collapsed")
elif source == "Paste data":
    with b_col:
        st.caption("A table with the headers already in place. Paste a block "
                   "straight from Excel or Google Sheets, or type into it.")
else:
    n_wells = a_col.slider("Wells", 1, 8, 4)
    seed = b_col.number_input("Seed", 0, 9999, 5, 1)

paste_grid = None
if source == "Paste data":
    st.caption("Select your data in the spreadsheet **without** its header "
               "row, click the first cell below and paste. Rows are added as "
               "you need them. Only `date` and `q_gas` are required — leave "
               "any other column blank.")

    if st.session_state.pop("_load_sample", False):
        st.session_state.pop("paste_editor", None)
        st.session_state["paste_frame"] = align_to_paste_columns(
            parse_pasted_text(SAMPLE_PASTE))
    if st.session_state.pop("_clear_grid", False):
        st.session_state.pop("paste_editor", None)
        st.session_state["paste_frame"] = blank_paste_frame()

    paste_grid = st.data_editor(
        st.session_state.get("paste_frame", blank_paste_frame()),
        column_config=paste_column_config(), column_order=PASTE_COLUMNS,
        num_rows="dynamic", hide_index=True, key="paste_editor",
        height=440)

    b1, b2, b3 = st.columns([1, 1, 3])
    b1.button("Load a sample", on_click=lambda: st.session_state.update(
        _load_sample=True),
        help="Fills the grid with a small worked example.")
    b2.button("Clear table", on_click=lambda: st.session_state.update(
        _clear_grid=True))
    b3.download_button(
        "Blank CSV template",
        blank_paste_frame(0).to_csv(index=False).encode(),
        "production_template.csv", "text/csv",
        help="Fill this in a spreadsheet, then paste the rows back here.")

    with st.expander("Rather paste a whole block, header row and all?"):
        blob = st.text_area(
            "Paste the table as text", height=150,
            label_visibility="collapsed",
            placeholder=("date\twell\tdays_on\tq_gas\tq_cond\tp_wf\tp_res\n"
                         "2019-01-01\tA-1\t31\t28450\t2210\t1420\t6180\n"
                         "...  header row first, one row per period"))
        if st.button("Load into the table", disabled=not blob.strip()):
            try:
                st.session_state["paste_frame"] = align_to_paste_columns(
                    parse_pasted_text(blob))
                st.session_state.pop("paste_editor", None)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

raw_df: Optional[pd.DataFrame] = None
if source == "Example field":
    with st.spinner("Simulating a synthetic condensate field..."):
        try:
            raw_df = make_demo_field(int(n_wells), int(seed), pvt_sig, pvt)
        except Exception as exc:
            st.error(f"Could not generate example data: {exc}")
            st.stop()
    # The simulator knows the true gas in place, so the app can show what the
    # material balance actually recovered instead of asking you to trust it.
    st.session_state["truth_ogip"] = {
        r["well"]: r["true_ogip_mmscf"] for r in (raw_df.attrs.get("truth") or [])
    }
    st.caption("Synthetic data from a single-tank reservoir model: "
               "pseudo-steady deliverability, two-phase material balance, a "
               "condensate-bank mobility penalty below the dew point and a "
               "facility plateau, with noise and downtime applied inside the "
               "loop so rates, cumulatives and pressures stay consistent.")
elif source == "Upload file" and upload is not None:
    st.session_state["truth_ogip"] = {}
    try:
        raw_df = read_upload(upload.getvalue(), upload.name)
    except Exception as exc:
        st.error(f"Could not read that file: {exc}")
        st.stop()
elif source == "Paste data" and paste_grid is not None:
    st.session_state["truth_ogip"] = {}
    parsed_df, paste_msgs = parse_paste_grid(paste_grid)
    for msg in paste_msgs:
        st.caption(f"· {msg}")
    if len(parsed_df) < 4:
        st.info("Enter or paste at least 4 rows with a date and a positive "
                "gas rate to run the analysis.")
        st.stop()
    n_w = parsed_df["well"].nunique() if "well" in parsed_df.columns else 1
    st.success(
        f"{len(parsed_df):,} usable row(s) · {n_w} well(s) · "
        f"{parsed_df['date'].min():%b %Y} to {parsed_df['date'].max():%b %Y}")
    raw_df = parsed_df

if raw_df is None or raw_df.empty:
    st.info("Upload a production history, paste one in, or switch to the "
            "example field to see the whole workflow.")
    with st.expander("Expected columns"):
        st.markdown("""
| Column | Units | Required | Notes |
|---|---|---|---|
| `date` | — | yes | any parseable date |
| `q_gas` | Mscf/d | yes | separator gas |
| `well` | — | no | one entity assumed if absent |
| `q_cond` | STB/d | no | absent means dry gas |
| `days_on` | days | no | strongly recommended |
| `p_wf` | psia | no | needed for flowing material balance |
| `p_res` | psia | no | needed for material balance |

Column names are matched case- and punctuation-insensitively, so
`Gas Rate (Mscf/d)`, `gas_mscfd` and `qg` all resolve to `q_gas`.
""")
    st.stop()

# -- column mapping -----------------------------------------------------------
mapped_preview = dca.map_columns(raw_df)
unresolved = [c for c in ("date", "q_gas") if c not in mapped_preview.columns]

with st.expander("Column mapping", expanded=bool(unresolved)):
    st.caption("Auto-detected names are shown first. Override anything the "
               "matcher got wrong.")
    cols = ["(none)"] + list(raw_df.columns)
    overrides: Dict[str, str] = {}
    grid = st.columns(4)
    for i, canon in enumerate(["date", "well", "q_gas", "q_cond", "days_on",
                               "p_wf", "p_res", "q_water"]):
        auto = None
        for orig in raw_df.columns:
            if dca.map_columns(raw_df[[orig]]).columns[0] == canon:
                auto = orig
                break
        idx = cols.index(auto) if auto in cols else 0
        pick = grid[i % 4].selectbox(canon, cols, index=idx,
                                     key=f"map_{canon}")
        if pick != "(none)":
            overrides[canon] = pick

work_df = raw_df.rename(columns={v: k for k, v in overrides.items()})
if "date" not in work_df.columns or "q_gas" not in work_df.columns:
    st.error("A date column and a gas rate column are required. Set them in "
             "**Column mapping** above.")
    st.stop()

well_col = "well" if "well" in work_df.columns else None

# -- fit window override ------------------------------------------------------
window = None
with st.expander("Decline fit window"):
    st.caption("By default the facility-limited plateau and any transient "
               "flow are excluded automatically. Rate on plateau is set by "
               "the contract, not the reservoir, and fitting Arps through it "
               "drags b to zero.")
    auto_window = st.toggle("Detect the window automatically", value=True)
if not auto_window:
    try:
        probe = work_df if well_col is None else work_df[
            work_df[well_col] == sorted(work_df[well_col].astype(str).unique())[0]]
        span_days = float((pd.to_datetime(probe["date"]).max()
                           - pd.to_datetime(probe["date"]).min()).days)
    except Exception:
        span_days = 3650.0
    span_yr = max(span_days / dca.DAYS_PER_YEAR, 1.0)
    lo, hi = st.slider("Fit window (years on production)", 0.0,
                       round(span_yr, 1), (0.0, round(span_yr, 1)), 0.1)
    window = (lo * dca.DAYS_PER_YEAR, hi * dca.DAYS_PER_YEAR)
    note("A manual window applies to every well. With wells of different "
         "vintages, prefer automatic detection.")

# ==============================================================================
# Run
# ==============================================================================

settings = (q_econ, model_choice, terminal, float(t_max), run_mc, int(n_mc),
            use_mb, use_fmb, cap_ogip, rate_basis, min_uptime, outlier_sigma,
            auto_window, window)
products = (inert_frac, fuel_frac, ngl_yield, ngl_shrink)

with st.spinner("Fitting declines, yield models and material balance..."):
    try:
        results, summary, profile, errors = run_analysis(
            work_df, pvt, pvt_sig, settings, products, well_col)
    except Exception:
        st.error("The analysis failed.")
        st.code(traceback.format_exc())
        st.stop()

if errors:
    with st.expander(f"{len(errors)} well(s) could not be analysed",
                     expanded=not results):
        for name, msg in errors.items():
            st.markdown(f"**{name}** — {msg}")

if not results:
    st.error("No well could be analysed. Check the column mapping, the rate "
             "units (gas in Mscf/d), and the QC thresholds in the sidebar.")
    st.stop()

well_names = list(summary["well"]) if not summary.empty else list(results)
sel = st.selectbox("Well", well_names, index=0)
res = results[sel]
fit = res.best_fit

# ==============================================================================
# Headline metrics
# ==============================================================================

m = st.columns(5)
m[0].metric("EUR wellstream gas",
            f"{res.forecast.eur_wellstream_mmscf:,.0f} MMscf",
            f"{res.forecast.remaining_wellstream_mmscf:,.0f} remaining")
m[1].metric("EUR sales gas", f"{res.forecast.eur_sales_gas_mmscf:,.0f} MMscf")
m[2].metric("EUR condensate",
            f"{res.forecast.eur_condensate_mstb:,.0f} Mstb",
            f"{res.forecast.remaining_condensate_mstb:,.0f} remaining")
m[3].metric("Economic life", f"{res.forecast.economic_life_years:.1f} yr")
m[4].metric("OGIP (material balance)",
            f"{res.matbal.ogip_mmscf:,.0f} MMscf" if res.matbal else "n/a")

if fit.at_bounds:
    if fit.at_bounds == ["b"] and fit.params.get("b", 1.0) < 1e-6:
        note("<b>b sits at zero</b>, so the data are exponential. That is a "
             "legitimate answer for a depleting gas well, not a failed fit.")
    else:
        warn("<b>" + ", ".join(fit.at_bounds) + "</b> pinned to a bound — the "
             "fit is not identifiable. Narrow the window, fix a parameter, or "
             "try another model.")

tabs = st.tabs(["Data & QC", "Decline fit", "Yield", "Material balance",
                "Forecast", "Uncertainty", "Field", "Export"])

# ------------------------------------------------------------------- Data & QC
with tabs[0]:
    qc = res.data.qc
    k = st.columns(5)
    k[0].metric("Rows in", qc.n_input)
    k[1].metric("Rows fitted on", qc.n_after_screen)
    k[2].metric("Outliers removed", qc.n_outliers_removed)
    k[3].metric("Low-uptime rows", qc.n_low_uptime_removed)
    k[4].metric("Plateau ends",
                f"{qc.plateau_end_days / dca.DAYS_PER_YEAR:.2f} yr"
                if qc.plateau_end_days is not None else "n/a")
    note("Cumulatives are accumulated over <b>every</b> producing month before "
         "any row is excluded. A month dropped from the fit still put gas in "
         "the pipe, and removing its volume would corrupt the material "
         "balance.")
    for n in qc.notes:
        st.caption(f"· {n}")
    show_fig(ch.chart_rate_time(res, THEME), key="qc_rate")
    with st.expander("Cleaned history"):
        show_df(res.data.df, height=380)
    with st.expander("Fluid properties in use"):
        show_df(res.pvt.describe().rename("value").to_frame())

# ---------------------------------------------------------------- Decline fit
with tabs[1]:
    left, right = st.columns([3, 2])
    with left:
        show_fig(ch.chart_rate_time(res, THEME), key="fit_rate")
    with right:
        st.markdown("**Selected fit**")
        st.code(fit.summary(), language="text")
    c1, c2 = st.columns(2)
    with c1:
        show_fig(ch.chart_b_diagnostic(res, THEME), key="fit_b")
    with c2:
        show_fig(ch.chart_decline_rate(res, THEME), key="fit_d")
    note("A genuinely hyperbolic segment shows b roughly constant. A b that "
         "keeps climbing means transient flow or a changing flowing pressure, "
         "and the Arps fit over that interval will not predict. b near zero "
         "means exponential, which is normal for a depleting gas well.")
    c1, c2 = st.columns([3, 2])
    with c1:
        show_fig(ch.chart_rate_cum(res, THEME), key="fit_cum")
    with c2:
        st.markdown("**Model ranking**")
        st.caption("Lower AIC is better, but statistics is not physics: "
                   "prefer the model whose late-time behaviour you can "
                   "defend. Duong often wins on fit for a transient history "
                   "and still gives an indefensible EUR.")
        show_df(res.model_table.style.format({
            "R2_log": "{:.4f}", "RMSE_log": "{:.4f}",
            "AIC": "{:.1f}", "BIC": "{:.1f}"}), hide_index=True)

# ---------------------------------------------------------------------- Yield
with tabs[2]:
    c1, c2 = st.columns([3, 2])
    with c1:
        show_fig(ch.chart_cgr(res, THEME), key="yield_cgr")
    with c2:
        st.code(res.yield_model.summary(), language="text")
        note("The floor matters. A pure exponential decays to zero, which no "
             "real condensate reservoir does, and it will understate "
             "late-life liquid. Use the CVD dropout curve to check the "
             "<i>shape</i>, not to set the level.")
    show_fig(ch.chart_condensate(res, THEME), key="yield_cond")

# ----------------------------------------------------------- Material balance
with tabs[3]:
    if res.matbal is None:
        st.info("No reservoir pressure data. Supply a `p_res` column to run "
                "the material balance.")
    else:
        truth = st.session_state.get("truth_ogip", {}).get(sel)
        if truth:
            k = st.columns(3)
            k[0].metric("True OGIP (simulated)", f"{truth:,.0f} MMscf")
            k[1].metric("Recovered by material balance",
                        f"{res.matbal.ogip_mmscf:,.0f} MMscf",
                        f"{100 * (res.matbal.ogip_mmscf / truth - 1):+.1f}%")
            if res.matbal.ogip_single_phase:
                k[2].metric("If single-phase z were used",
                            f"{res.matbal.ogip_single_phase:,.0f} MMscf",
                            f"{100 * (res.matbal.ogip_single_phase / truth - 1):+.1f}%",
                            delta_color="inverse")
            note("The example data comes from a reservoir model whose gas in "
                 "place is known, so you can see what the two-phase material "
                 "balance recovers and what the single-phase shortcut costs.")
        show_fig(ch.chart_pz(res, THEME, height=440), key="mb_pz")
        if res.matbal.ogip_single_phase:
            delta = 100 * (res.matbal.ogip_single_phase
                           / res.matbal.ogip_mmscf - 1)
            note(f"Using the single-phase z-factor would change OGIP by "
                 f"<b>{delta:+.1f}%</b>. Below the dew point part of the "
                 "hydrocarbon has condensed, so the two-phase z is lower, the "
                 "single-phase points fall below the true line, and the "
                 "extrapolation is biased. The direction and size depend on "
                 "the fluid — which is why this is measured rather than "
                 "assumed.")
        with st.expander("Material balance detail"):
            st.code(res.matbal.summary(), language="text")
    if res.fmb is not None:
        st.markdown("**Flowing material balance**")
        f = res.fmb
        k = st.columns(4)
        k[0].metric("Contacted gas",
                    f"{f['ogip_contacted_mmscf']:,.0f} MMscf")
        k[1].metric("Converged", str(bool(f["converged"])))
        k[2].metric("R2 of the line", f"{f['r2']:.3f}")
        k[3].metric("p_avg at last point",
                    f"{f['p_avg_last_psia']:,.0f} psia")
        if f.get("bias_warning"):
            warn("The reservoir is below the dew point over part of this "
                 "window, so condensate banking degrades mobility faster than "
                 "depletion alone and this figure is biased <b>low</b>. Treat "
                 "it as a lower bound, not a competing OGIP.")

# ------------------------------------------------------------------- Forecast
with tabs[4]:
    st.code(res.forecast.summary(), language="text")
    c1, c2 = st.columns(2)
    with c1:
        show_fig(ch.chart_rate_time(res, THEME, show_separator=False),
                 key="fx_gas")
    with c2:
        show_fig(ch.chart_condensate(res, THEME), key="fx_cond")
    with st.expander("Forecast table"):
        show_df(res.forecast.table, height=400)

# ---------------------------------------------------------------- Uncertainty
with tabs[5]:
    if res.mc is None:
        st.info("Monte Carlo is switched off. Enable it in the sidebar under "
                "**Cross-checks and uncertainty**.")
    else:
        note("Petroleum convention: <b>P90 is the low case</b> (90% chance of "
             "exceeding) and P10 the high case. Sampled parameter standard "
             "deviations are capped at 35% of value — production-data "
             "regression covariances are enormous because the parameters "
             "trade off, and uncapped the sampler spends every draw outside "
             "physical bounds.")
        for key, col, label in (
                ("EUR wellstream gas (MMscf)", "eur_wellstream_mmscf",
                 "EUR wellstream gas (MMscf)"),
                ("EUR condensate (Mstb)", "eur_condensate_mstb",
                 "EUR condensate (Mstb)")):
            s = (res.mc_stats or {}).get(key)
            if not s:
                continue
            st.markdown(f"**{key}**")
            k = st.columns(4)
            k[0].metric("P90 (low)", f"{s['P90']:,.0f}")
            k[1].metric("P50", f"{s['P50']:,.0f}")
            k[2].metric("P10 (high)", f"{s['P10']:,.0f}")
            k[3].metric("P10 / P90", f"{s['P10'] / max(s['P90'], 1e-9):.2f}x")
            show_fig(ch.chart_eur_cdf(res, col, label, key, THEME),
                     key=f"mc_{col}")
        drivers = [c for c in res.mc.columns if c.startswith("p_")]
        if drivers:
            d_sel = st.selectbox("Driver", drivers, index=0)
            show_fig(ch.chart_mc_scatter(res, d_sel, "eur_wellstream_mmscf",
                                         d_sel.replace("p_", ""),
                                         "EUR wellstream gas (MMscf)", THEME),
                     key="mc_scatter")

# ---------------------------------------------------------------------- Field
with tabs[6]:
    tot = summary[["EUR_wellstream_mmscf", "EUR_sales_gas_mmscf",
                   "EUR_condensate_mstb"]].sum()
    k = st.columns(4)
    k[0].metric("Wells analysed", len(summary))
    k[1].metric("Field EUR wellstream gas",
                f"{tot['EUR_wellstream_mmscf']:,.0f} MMscf")
    k[2].metric("Field EUR sales gas",
                f"{tot['EUR_sales_gas_mmscf']:,.0f} MMscf")
    k[3].metric("Field EUR condensate",
                f"{tot['EUR_condensate_mstb']:,.0f} Mstb")
    note("The field profile is the sum of well forecasts on a common time "
         "axis. A single field-level decline would hide well additions, "
         "workovers and liquid loading.")
    c1, c2 = st.columns(2)
    with c1:
        show_fig(ch.chart_field_profile(profile, "q_wellstream_mscfd",
                                        "Field wellstream gas profile",
                                        "Gas rate (Mscf/d)", THEME),
                 key="fld_gas")
    with c2:
        show_fig(ch.chart_field_profile(profile, "q_condensate_stbd",
                                        "Field condensate profile",
                                        "Condensate rate (STB/d)", THEME),
                 key="fld_cond")
    c1, c2 = st.columns(2)
    with c1:
        show_fig(ch.chart_well_bars(summary, "EUR_wellstream_mmscf",
                                    "EUR by well", "EUR wellstream (MMscf)",
                                    THEME), key="fld_bar_gas")
    with c2:
        show_fig(ch.chart_well_bars(summary, "EUR_condensate_mstb",
                                    "Condensate EUR by well",
                                    "EUR condensate (Mstb)", THEME),
                 key="fld_bar_cond")
    show_df(summary, hide_index=True)

# --------------------------------------------------------------------- Export
with tabs[7]:
    st.markdown("**Downloads**")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Well summary (CSV)",
                       summary.to_csv(index=False).encode(),
                       "well_summary.csv", "text/csv")
    c2.download_button("Field profile (CSV)",
                       profile.to_csv(index=False).encode(),
                       "field_profile.csv", "text/csv")
    c3.download_button(f"{sel} forecast (CSV)",
                       res.forecast.table.to_csv(index=False).encode(),
                       f"{sel}_forecast.csv", "text/csv")

    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as xl:
            summary.to_excel(xl, sheet_name="well_summary", index=False)
            profile.to_excel(xl, sheet_name="field_profile", index=False)
            for name, r in results.items():
                tag = str(name)[:24]
                r.data.df.to_excel(xl, sheet_name=f"{tag}_hist"[:31],
                                   index=False)
                r.forecast.table.to_excel(xl, sheet_name=f"{tag}_fcst"[:31],
                                          index=False)
        st.download_button("Full workbook (XLSX)", buf.getvalue(),
                           "gas_condensate_dca.xlsx",
                           "application/vnd.openxmlformats-officedocument."
                           "spreadsheetml.sheet")
    except Exception as exc:
        st.caption(f"Excel export unavailable ({exc}). Install `openpyxl`.")

    st.markdown("**Full text report for this well**")
    sio = io.StringIO()
    with contextlib.redirect_stdout(sio):
        res.summary()
    st.download_button(f"{sel} report (TXT)", sio.getvalue().encode(),
                       f"{sel}_report.txt", "text/plain")
    st.code(sio.getvalue(), language="text")
