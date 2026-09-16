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
  sidebar : fluid definition (PVT), CVD table, run settings
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


# ==============================================================================
# Deferred widget writes
# ==============================================================================
# Streamlit refuses to let session state be written under a widget's key once
# that widget has been instantiated in the current run. The estimated initial
# pressure is decided far down the page, long after the sidebar has rendered,
# so it is parked under a staging key and moved onto the widget here, at the
# top of the following run, before any widget exists.

_STAGED = {"_stage_p_init": "w_p_init", "_stage_mb_pi": "w_mb_pi"}
for _stage, _target in _STAGED.items():
    if _stage in st.session_state:
        st.session_state[_target] = st.session_state.pop(_stage)


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


NGL_COLUMNS = ["q_ngl_stbd", "cum_ngl_mstb"]
SEP_RENAMES = {"q_sales_gas_mscfd": "q_sep_gas_mscfd",
               "cum_sales_gas_mmscf": "cum_sep_gas_mmscf"}


def tidy_forecast(table: pd.DataFrame) -> pd.DataFrame:
    """Drop the plant-NGL columns and rename sales gas to separator gas.

    With no surface processing declared, sales gas is identically the separator
    gas and NGL is identically zero. Carrying both through the tables invites
    someone to quote a sales-gas number that has had no processing applied.
    """
    t = table.drop(columns=[c for c in NGL_COLUMNS if c in table.columns])
    # q_sep_gas_mscfd already exists from the engine and equals sales gas here,
    # so drop the duplicate rather than collide on the rename.
    dupes = [old for old, new in SEP_RENAMES.items()
             if old in t.columns and new in t.columns]
    t = t.drop(columns=dupes)
    return t.rename(columns={k: v for k, v in SEP_RENAMES.items()
                             if k in t.columns})


def tidy_summary_text(text: str) -> str:
    """Strip the NGL lines from a Forecast.summary() block and relabel gas.

    The replacement eats four trailing spaces so the colons stay aligned:
    "sales gas" is 9 characters, "separator gas" is 13.
    """
    keep = [ln for ln in text.splitlines() if "NGL" not in ln]
    return "\n".join(ln.replace("sales gas    ", "separator gas")
                     for ln in keep)


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
                parsed = dca.parse_dates(d[c])
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

    d = d.dropna(subset=["date", "q_gas", "p_res"], how="all")
    n_rows = len(d)
    if n_rows == 0:
        return pd.DataFrame(), messages

    parsed = dca.parse_dates(d["date"])
    bad_date = parsed.isna()
    if bad_date.any():
        messages.append(f"{int(bad_date.sum())} row(s) dropped: the date could "
                        "not be read.")
    d["date"] = parsed

    # A row with a pressure and no rate is a static survey, and gauges are run
    # precisely on the months a well is shut in. Dropping it for want of a rate
    # throws away the only measurement of the reservoir in that row — which is
    # how the material balance ends up reporting no pressure data on a well
    # that plainly has it.
    survey_only = d["p_res"].notna() & (d["p_res"] > 0) & (
        d["q_gas"].isna() | (d["q_gas"] <= 0))
    d.loc[survey_only, "q_gas"] = 0.0
    if survey_only.any():
        messages.append(f"{int(survey_only.sum())} row(s) kept as pressure "
                        "surveys: a reservoir pressure with no rate.")
        # A blank well on a survey row would be dropped by the per-well
        # grouping downstream, taking the survey with it.
        if d["well"].notna().any():
            fill = d["well"].ffill().bfill()
            d.loc[survey_only & d["well"].isna(), "well"] = fill[
                survey_only & d["well"].isna()]

    bad_gas = (d["q_gas"].isna() | (d["q_gas"] <= 0)) & ~survey_only
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


def prepend_initial_pressure_row(frame: pd.DataFrame,
                                 p_initial: float) -> pd.DataFrame:
    """Write an estimated p_i into the grid as a survey at zero cumulative.

    It goes in as its own row dated the day before first production rather
    than into the first producing month's `p_res`, because those are different
    quantities: the first month's cell means "pressure once that month's gas
    had been produced", and the whole point of the estimate is the pressure
    BEFORE any of it was. A row of its own is also visible and deletable,
    which a number quietly dropped into an existing cell is not.
    """
    fr = align_to_paste_columns(frame) if not set(PASTE_COLUMNS) <= set(
        frame.columns) else frame.copy()
    dates = dca.parse_dates(fr["date"])
    if dates.notna().any():
        first = dates.min()
        new_date = (first - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        well = ""
        first_rows = fr.loc[dates == first, "well"]
        if len(first_rows):
            well = str(first_rows.iloc[0] or "")
    else:
        new_date, well = "", ""

    # Replace a previous estimate rather than stacking another one on top.
    keep = ~((dates.notna()) & (dates == dates.min())
             & (pd.to_numeric(fr["q_gas"], errors="coerce").fillna(0) <= 0)
             & (pd.to_numeric(fr["p_res"], errors="coerce").notna())
             ) if dates.notna().any() else pd.Series(True, index=fr.index)
    fr = fr[keep]

    row = {c: np.nan for c in PASTE_COLUMNS}
    row.update({"date": new_date, "well": well, "days_on": 0.0,
                "p_res": float(p_initial)})
    out = pd.concat([pd.DataFrame([row]), fr], ignore_index=True)
    return align_to_paste_columns(out)


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
def estimate_pi(df: pd.DataFrame, _pvt: dca.PVT, pvt_sig: tuple,
                well_col: Optional[str], rate_basis: str, min_uptime: float,
                outlier_sigma: float):
    """Field initial pressure from the early p/z trend, or why it cannot be."""
    raw = dca.map_columns(df)
    groups = ({str(k): v for k, v in raw.groupby(well_col)}
              if well_col and well_col in raw.columns else {"FIELD": raw})
    prepared = {}
    for name, grp in groups.items():
        try:
            prepared[name] = dca.ProductionData.prepare(
                grp, _pvt, well=name, rate_basis=rate_basis,
                min_uptime_frac=min_uptime,
                outlier_sigma=(None if outlier_sigma <= 0 else outlier_sigma),
                detect_bdf=False)
        except Exception:
            continue
    if not prepared:
        return None
    try:
        return dca.estimate_initial_pressure_from_wells(prepared, _pvt)
    except Exception:
        return None


@st.cache_data(show_spinner=False, max_entries=8)
def run_analysis(df: pd.DataFrame, _pvt: dca.PVT, pvt_sig: tuple,
                 settings: tuple, well_col: Optional[str]):
    """Analyse every well. Keyed on the dataframe, PVT signature and settings."""
    (q_econ, model, terminal, t_max, run_mc, n_mc, use_mb, use_fmb, cap,
     rate_basis, min_uptime, outlier_sigma, fit_from_bdf, window,
     mb_pi, mb_skip, use_aq, ogip_mode) = settings
    # No surface processing is applied, so "sales gas" is the separator gas and
    # plant NGL is zero. Both are reported on the separator-gas basis below.
    split = dca.ProductSplit(inert_fraction=0.0, fuel_flare_fraction=0.0,
                             ngl_yield_gal_per_mscf=0.0,
                             ngl_shrinkage_fraction=0.0)

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
                use_material_balance=use_mb,
                mb_p_initial=(mb_pi if mb_pi and mb_pi > 0 else None),
                mb_skip_early=int(mb_skip), use_aquifer=use_aq,
                use_fmb=use_fmb,
                apply_ogip_cap=cap, ogip_cap_mode=ogip_mode,
                fit_from_bdf=fit_from_bdf,
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
            "EUR_sep_gas_mmscf": r.forecast.eur_sales_gas_mmscf,
            "EUR_condensate_mstb": r.forecast.eur_condensate_mstb,
            "remaining_gas_mmscf": r.forecast.remaining_wellstream_mmscf,
            "life_yr": r.forecast.economic_life_years,
            "OGIP_pz_mmscf": (r.matbal.ogip_mmscf if r.matbal else np.nan),
            "OGIP_ceiling_mmscf": (r.matbal.g_ceiling_mmscf
                                  if r.matbal else np.nan),
            "OGIP_fetkovich_mmscf": (
                r.matbal.fetkovich["G_mmscf"]
                if r.matbal and r.matbal.fetkovich else np.nan),
            "OGIP_cap_mmscf": (r.ogip_choice.value
                               if r.ogip_choice else np.nan),
            "OGIP_cap_source": (r.ogip_choice.source
                                if r.ogip_choice else "none"),
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
    # Passing a default alongside a key that session state has already set
    # makes Streamlit log a warning on every run, so the default is offered
    # only the first time round.
    _pi_default = ({} if "w_p_init" in st.session_state
                   else {"value": 6400.0})
    p_init = c1.number_input(
        "Initial pressure (psia)", min_value=200.0, max_value=20000.0,
        step=50.0, **_pi_default,
        key="w_p_init",
        help="Reservoir pressure at zero cumulative production — not the "
             "first survey, which is almost always months into the life. If "
             "it was never measured, the app offers to estimate it from the "
             "early p/z trend once production data is loaded.")
    p_dew = c2.number_input("Dew point (psia)", 0.0, 20000.0, 5100.0, 50.0)
    initial_cgr = st.number_input(
        "Initial CGR (STB/MMscf)", 0.0, 500.0, 78.0, 1.0,
        help="Stock tank condensate yield of the produced wellstream at "
             "initial reservoir pressure. If initial pressure is below dew "
             "point, use the vapour/produced CGR at the initial pressure "
             "rather than the above dewpoint fluid CGR.")
    use_wellstream = st.toggle(
        "Use wellstream gravity for PVT", value=True,
        help="Correct above the dew point: the reservoir flows one phase, so "
             "properties should use the wet-gas gravity, not the separator "
             "gas gravity.")

    with st.expander("CVD table (two-phase z)"):
        note("From the lab CVD report. Without it the Rayes correlation is "
             "used, which is a regression rather than your fluid.")
        use_cvd = st.toggle("Use CVD table", value=True)
        cvd_df = st.data_editor(DEFAULT_CVD, num_rows="dynamic",
                                key="cvd_editor",
                                disabled=not use_cvd)

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
        _mb_default = ({} if "w_mb_pi" in st.session_state
                       else {"value": 0.0})
        mb_pi = st.number_input(
            "Material balance p_i (0 = extrapolate)", min_value=0.0,
            max_value=2.0e5, step=50.0, **_mb_default,
            key="w_mb_pi",
            help="Leave at 0 and the reference is extrapolated from the p/z "
                 "line to zero cumulative, which is what makes the answer "
                 "ORIGINAL gas in place rather than gas in place on the day "
                 "of the first survey. Set it only if a real initial pressure "
                 "is known.")
        use_aq = st.toggle(
            "Fit a Fetkovich aquifer", value=True,
            help="Adds a second or two per well. Worth it whenever the drive "
                 "is not volumetric; on a closed tank it returns negligible "
                 "influx, which is the cross-check passing.")
        mb_skip = st.number_input(
            "Drop earliest surveys", 0, 20, 0, 1,
            help="Use this when the consistency guard fires and the first "
                 "survey predates a reliable reference.")
        cap_ogip = st.toggle("Cap forecast at material-balance OGIP",
                             value=True)
        OGIP_MODES = {"Auto (by drive)": "auto", "p/z line": "p/z",
                      "Fetkovich G": "fetkovich", "We ≥ 0 ceiling": "ceiling"}
        ogip_mode_label = st.selectbox(
            "Which gas in place", list(OGIP_MODES), index=0,
            disabled=not cap_ogip,
            help="The p/z intercept is only gas in place when the tank is "
                 "closed; under pressure support it is inflated by the "
                 "influx, which is exactly when a cap matters. Auto takes the "
                 "p/z line on a volumetric drive, the Fetkovich G when the "
                 "drive is supported and the aquifer fit is usable, and the "
                 "min(F/Eg) ceiling otherwise. Every choice is clipped to the "
                 "ceiling, which no gas in place may exceed.")
        ogip_mode = OGIP_MODES[ogip_mode_label]
        run_mc = st.toggle("Run Monte Carlo", value=True)
        n_mc = st.slider("Monte Carlo realisations", 200, 4000, 800, 100,
                         disabled=not run_mc)

# ==============================================================================
# PVT object
# ==============================================================================

# Inert content is not exposed in the UI; the pseudo-criticals are built on the
# hydrocarbon gravity alone. For a sour or nitrogen-rich gas, pass the mole
# fractions to gas_condensate_dca.PVT directly instead of using this app.
Y_N2 = Y_CO2 = Y_H2S = 0.0

cvd_key = None
if use_cvd and cvd_df is not None and len(cvd_df.dropna()) >= 3:
    clean = cvd_df.dropna(subset=["pressure_psia", "cum_produced_molfrac"])
    cvd_key = tuple(zip(clean["pressure_psia"].astype(float),
                        clean["cum_produced_molfrac"].astype(float)))

try:
    pvt = build_pvt(gas_gravity, temperature_F, condensate_api,
                    (mw_override or None), (p_dew or None), (p_init or None),
                    Y_N2, Y_CO2, Y_H2S, (initial_cgr or None),
                    use_wellstream, cvd_key)
except Exception as exc:
    st.error(f"The fluid definition is not valid: {exc}")
    st.stop()

pvt_sig = (gas_gravity, temperature_F, condensate_api, mw_override, p_dew,
           p_init, Y_N2, Y_CO2, Y_H2S, initial_cgr, use_wellstream, cvd_key)

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
    _pending_pi = st.session_state.pop("_stage_grid_pi", None)
    if _pending_pi is not None:
        # The editor's live contents, stashed on the previous run: rebuilding
        # from `paste_frame` alone would silently undo anything typed since.
        base_fr = st.session_state.get("_paste_current")
        if base_fr is None:
            base_fr = st.session_state.get("paste_frame", blank_paste_frame())
        try:
            st.session_state["paste_frame"] = prepend_initial_pressure_row(
                base_fr, float(_pending_pi))
            st.session_state.pop("paste_editor", None)
        except Exception as exc:
            st.warning(f"The estimate could not be written into the table "
                       f"({exc}). It is still applied to the fluid definition.")

    paste_grid = st.data_editor(
        st.session_state.get("paste_frame", blank_paste_frame()),
        column_config=paste_column_config(), column_order=PASTE_COLUMNS,
        num_rows="dynamic", hide_index=True, key="paste_editor",
        height=440)
    st.session_state["_paste_current"] = paste_grid

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

# ==============================================================================
# Initial pressure
# ==============================================================================
# A gauge is almost never run before a well produces. The first static survey
# arrives months in, by which time the reservoir has already given up pressure
# nobody recorded. Taking that survey as p_i does not fail loudly - it quietly
# redefines G as the gas in place on the day of the survey, drops everything
# produced before it, and makes every Eg too small. So the app looks for the
# gap and offers to close it rather than waiting to be asked.

PI_MIN_GAP_DAYS = 45.0
PI_MIN_RISE_FRAC = 0.01


def pi_signature(df: pd.DataFrame, col: Optional[str]) -> str:
    """Identify this dataset, so the prompt appears once per table."""
    try:
        d = dca.parse_dates(df["date"])
        pres = pd.to_numeric(df.get("p_res"), errors="coerce") if (
            "p_res" in df.columns) else pd.Series(dtype=float)
        return "|".join(str(x) for x in (
            len(df), col, d.min(), d.max(),
            int(np.isfinite(pres).sum()) if len(pres) else 0,
            round(float(np.nansum(pres)), 3) if len(pres) else 0.0,
            round(float(pd.to_numeric(df["q_gas"], errors="coerce").sum()), 3)))
    except Exception:
        return str(len(df))


def apply_pi(value: float) -> None:
    """Park the estimate for the sidebar, the material balance and the grid."""
    v = float(np.clip(value, 200.0, 20000.0))
    st.session_state["_stage_p_init"] = v
    st.session_state["_stage_mb_pi"] = float(np.clip(value, 0.0, 2.0e5))
    st.session_state["_stage_grid_pi"] = v
    st.session_state["_pi_applied"] = v
    # Accepting the estimate edits the table, which changes its signature.
    # Without this the next run would read that as a brand new dataset, clear
    # the banner and offer the estimate all over again.
    st.session_state["_pi_absorb_sig"] = True


pi_sig = pi_signature(work_df, well_col)
if st.session_state.get("_pi_data_sig") != pi_sig:
    # A new table invalidates any estimate made against the old one.
    st.session_state["_pi_data_sig"] = pi_sig
    if st.session_state.pop("_pi_absorb_sig", False):
        st.session_state["_pi_prompt_seen"] = pi_sig
    else:
        st.session_state.pop("_pi_prompt_seen", None)
        st.session_state.pop("_pi_applied", None)

pi_est = None
if "p_res" in work_df.columns:
    with st.spinner("Checking the pressure record..."):
        try:
            pi_est = estimate_pi(work_df, pvt, pvt_sig, well_col, rate_basis,
                                 min_uptime, outlier_sigma)
        except Exception:
            pi_est = None


def pi_gap_matters(e) -> bool:
    if e is None or not e.ok:
        return False
    late = (np.isfinite(e.gap_days) and e.gap_days > PI_MIN_GAP_DAYS) or (
        np.isfinite(e.gap_frac_ogip) and e.gap_frac_ogip > 0.02)
    lifted = e.rise_psi > PI_MIN_RISE_FRAC * max(e.p_first_survey, 1.0)
    return bool(late and lifted)


def pi_body(e) -> None:
    """The shared explanation, used by the dialog and by the inline panel."""
    st.markdown(
        f"The earliest static survey reads **{e.p_first_survey:,.0f} psia**, "
        f"but it was taken **{e.gap_months:,.1f} months** into production, "
        f"after **{e.gap_mmscf:,.0f} MMscf** had already been produced. "
        "That is a pressure during depletion, not the initial pressure.")
    g = st.columns(3)
    g[0].metric("Estimated p_i", f"{e.p_initial:,.0f} psia",
                f"{e.rise_psi:,.0f} psi above survey 1", delta_color="off")
    g[1].metric("Range", f"{e.low:,.0f} – {e.high:,.0f}",
                help="Spread across every early window that fits the same "
                     "line, not a confidence interval.")
    g[2].metric("Surveys used", f"{e.n_used} of {e.n_surveys}",
                f"R² {e.r2:.4f}" if np.isfinite(e.r2) else None,
                delta_color="off")
    note(f"Method: {e.method}. p/z is straight against cumulative production "
         "while depletion is volumetric, so the value where that line crosses "
         "zero cumulative is p_i/z_i, and inverting it gives p_i. Only the "
         "early surveys are used — pressure support flattens the late trend, "
         "which pulls the intercept down and the OGIP up at the same time.")
    for w in e.warnings:
        warn(w[0].upper() + w[1:])


@st.dialog("Initial pressure was never measured")
def pi_dialog(e) -> None:
    pi_body(e)
    st.caption("Accepting writes the estimate into the fluid definition, uses "
               "it as the material balance reference at zero cumulative, and "
               "adds it to the table as a survey dated before first gas.")
    a, b = st.columns(2)
    if a.button("Estimate it", type="primary", width="stretch"):
        apply_pi(e.p_initial)
        st.session_state["_pi_prompt_seen"] = pi_sig
        st.rerun()
    if b.button("Leave it alone", width="stretch"):
        st.session_state["_pi_prompt_seen"] = pi_sig
        st.rerun()


applied = st.session_state.get("_pi_applied")
if applied is not None:
    c1, c2 = st.columns([5, 1])
    c1.success(
        f"**Initial pressure estimated at {applied:,.0f} psia** and applied to "
        "the fluid definition, the material balance reference and the table.")
    if c2.button("Undo", width="stretch"):
        st.session_state["_stage_p_init"] = 6400.0
        st.session_state["_stage_mb_pi"] = 0.0
        st.session_state.pop("_pi_applied", None)
        st.rerun()
elif pi_gap_matters(pi_est):
    if st.session_state.get("_pi_prompt_seen") != pi_sig:
        pi_dialog(pi_est)
    else:
        with st.expander(
                f"Initial pressure was never measured — the first survey is "
                f"{pi_est.gap_months:,.1f} months late", expanded=False):
            pi_body(pi_est)
            if st.button("Estimate it", type="primary", key="pi_late"):
                apply_pi(pi_est.p_initial)
                st.rerun()
else:
    reason = None
    if "p_res" not in work_df.columns:
        reason = ("there is no reservoir pressure in the data at all. "
                  "Initial pressure cannot be inferred from rates alone — it "
                  "needs at least two static surveys, a pre-production "
                  "RFT/DST, or a regional pressure gradient.")
    elif pi_est is not None and not pi_est.ok:
        reason = pi_est.reason
    if reason:
        note(f"<b>No initial pressure.</b> {reason[0].upper()}{reason[1:]} "
             "Enter it in the sidebar under <b>Initial pressure</b> if it is "
             "known from an offset well or a regional gradient. The decline "
             "fit, the yield model and the forecast do not need it; the "
             "material balance does.")

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
        pd_ = dca.parse_dates(probe["date"])
        span_days = float((pd_.max() - pd_.min()).days)
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
            auto_window, window, float(mb_pi), int(mb_skip), use_aq,
            ogip_mode)
with st.spinner("Fitting declines, yield models and material balance..."):
    try:
        results, summary, profile, errors = run_analysis(
            work_df, pvt, pvt_sig, settings, well_col)
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
m[1].metric("EUR separator gas",
            f"{res.forecast.eur_sales_gas_mmscf:,.0f} MMscf",
            help="Wellstream gas less the condensate gas equivalent. No "
                 "surface processing is applied.")
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
    k = st.columns(6)
    k[0].metric("Rows in", qc.n_input)
    k[1].metric("Rows fitted on", qc.n_after_screen)
    k[2].metric("Outliers removed", qc.n_outliers_removed)
    k[3].metric("Low-uptime rows", qc.n_low_uptime_removed)
    k[5].metric("Pressure surveys", qc.n_pressure_surveys,
                help="Counted on the full record. Surveys are kept for the "
                     "material balance even when the rate QC drops that "
                     "month.")
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
        st.info(res.settings.get("matbal_note")
                or "No reservoir pressure data. Supply a `p_res` column to "
                   "run the material balance.")
        surv = res.data.surveys
        if not surv.empty:
            st.caption(f"{len(surv)} pressure survey(s) found in the record:")
            show_df(surv, hide_index=True)
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
        mb = res.matbal
        k = st.columns(5)
        k[0].metric("OGIP (p/z line)", f"{mb.ogip_mmscf:,.0f} MMscf",
                    f"R² {mb.r2:.3f}", delta_color="off")
        k[1].metric("G ceiling (We ≥ 0)",
                    f"{mb.g_ceiling_mmscf:,.0f} MMscf"
                    if np.isfinite(mb.g_ceiling_mmscf) else "n/a",
                    help="min(F/Eg). Influx can only add to the withdrawal, "
                         "so this is a hard upper bound on gas in place "
                         "however straight the p/z plot looks.")
        k[2].metric("F/Eg rise",
                    f"{mb.ho_rise:.2f}×" if np.isfinite(mb.ho_rise) else "n/a")
        k[3].metric("Drive", mb.drive.title())
        k[4].metric("Produced", f"{mb.gp_now:,.0f} MMscf")

        if np.isfinite(mb.p_initial):
            src = ("as entered in the sidebar" if mb.p_initial_known else
                   "extrapolated from the p/z line to zero cumulative")
            note(f"Reference pressure: <b>{mb.p_initial:,.0f} psia</b> at zero "
                 f"cumulative, {src}. Every Eg and every F/Eg below is "
                 "measured from it, so it is the single number this tab is "
                 "most sensitive to.")

        if mb.impossible:
            warn(f"<b>Consistency guard.</b> Material balance requires "
                 f"G ≤ min(F/Eg) = {mb.g_ceiling_mmscf:,.0f} MMscf because "
                 f"We ≥ 0, but {mb.gp_now:,.0f} MMscf has already been "
                 "produced. The surveys and the volumes cannot both be right, "
                 "and no aquifer model rescues that. The usual cause is a "
                 "reference pressure taken <b>after</b> first production — p_i "
                 "is then too low, every Eg downstream too small and F/Eg too "
                 "large everywhere. Try dropping the earliest survey below.")
        elif mb.drive == "volumetric":
            st.success(f"**Volumetric depletion.** F/Eg is flat "
                       f"({mb.ho_rise:.2f}× across the record), so the gas is "
                       "producing by its own expansion and the p/z intercept "
                       "is a real number.")
        elif mb.drive == "borderline":
            warn(f"<b>Borderline — F/Eg climbs {mb.ho_rise:.2f}×.</b> Under "
                 "the 1.25× at which this calls water drive, but not flat "
                 "either, and a rise this size is what early or weak pressure "
                 "support looks like before it becomes obvious. Treat the p/z "
                 "intercept as an upper bound and check whether produced "
                 "water is accelerating.")
        elif mb.drive == "water drive":
            warn(f"<b>Water drive or pressure support.</b> F/Eg climbs "
                 f"{mb.ho_rise:.2f}×, so something outside the gas is "
                 "supplying energy. The p/z intercept is an <b>artefact, not "
                 "a volume</b> — support holds pressure up, which flattens the "
                 "trend and inflates the intercept. Use the ceiling instead.")

        oc = res.ogip_choice
        if oc is not None:
            if np.isfinite(oc.value):
                bits = " · ".join(
                    (f"<b>{k} {v:,.0f}</b>" if k == oc.source
                     else f"{k} {v:,.0f}")
                    for k, v in (oc.candidates or {}).items())
                note(f"<b>Forecast cap: {oc.value:,.0f} MMscf</b> "
                     f"({oc.source}, ±{100 * oc.rel_sigma:.0f} % in the Monte "
                     f"Carlo). {oc.reason}<br>Candidates — {bits} MMscf.")
            else:
                note(f"<b>The forecast is not capped.</b> {oc.reason}")

        if mb.ogip_exceeds_ceiling:
            warn(f"<b>The p/z intercept is above what material balance "
                 f"allows.</b> Since We ≥ 0, G ≤ min(F/Eg) = "
                 f"<b>{mb.g_ceiling_mmscf:,.0f} MMscf</b>, but the straight "
                 f"line reads {mb.ogip_mmscf:,.0f} — "
                 f"{mb.ogip_mmscf / mb.g_ceiling_mmscf:.2f}× the ceiling. Take "
                 "the ceiling as the upper bound on this tank and treat the "
                 "intercept as what it is: a line fitted through a trend that "
                 "is curving.")

        c1, c2 = st.columns(2)
        with c1:
            show_fig(ch.chart_pz(res, THEME), key="mb_pz")
            st.caption("**p/z vs Gp.** Straight for a volumetric tank, and the "
                       "intercept is OGIP. But a straight-*looking* p/z plot "
                       "is not evidence of one — the panel beside it is.")
        with c2:
            show_fig(ch.chart_havlena_odeh(res, THEME), key="mb_ho")
            st.caption("**Havlena–Odeh.** F = G·Eg + We. Flat means We ≈ 0 and "
                       "the level *is* G. Rising means influx. This is the "
                       "discriminator.")

        show_fig(ch.chart_apparent_g(res, THEME), key="mb_appg")
        ho = mb.ho_table
        u = ho["apparent_G_usable"].to_numpy()
        if u.sum() >= 2:
            vals = ho.loc[u, "apparent_G_mmscf"].to_numpy()
            spread = float(vals[-1] / vals[0]) if vals[0] > 0 else float("nan")
            if np.isfinite(spread) and spread > 1.10:
                warn(f"<b>Apparent G climbs {spread:.2f}× across the "
                     f"surveys</b>, from {vals[0]:,.0f} to {vals[-1]:,.0f} "
                     "MMscf. A closed tank returns the same number every time; "
                     "support holds p/z up, which inflates every estimate and "
                     "the later ones most. So the <b>smallest</b> value is the "
                     f"tightest bound: <b>G ≤ {mb.g_bound_mmscf:,.0f} "
                     "MMscf</b>, and the straight-line intercept is the least "
                     "reliable reading of the set because it is dominated by "
                     "the latest, most inflated points.")
            elif np.isfinite(spread) and spread < 0.91:
                warn(f"<b>Apparent G falls {spread:.2f}× across the "
                     "surveys.</b> Influx only accumulates, so it cannot "
                     "produce a falling trend. Suspect the reference pressure, "
                     "the datum correction, or production allocated to this "
                     "well.")
            else:
                st.success(f"**Apparent G is level ({spread:.2f}× across the "
                           f"surveys)** at about {np.median(vals):,.0f} MMscf. "
                           "That is what a closed tank looks like, and it is "
                           "independent confirmation of the p/z intercept.")
            tbl = ho.loc[u, ["p", "Gp_mmscf", "depleted_frac",
                             "F_over_Eg_mmscf", "apparent_G_mmscf"]].copy()
            tbl.columns = ["p (psia)", "Gp (MMscf)", "Depleted",
                           "F/Eg (MMscf)", "Apparent G (MMscf)"]
            show_df(tbl.style.format({
                "p (psia)": "{:,.0f}", "Gp (MMscf)": "{:,.0f}",
                "Depleted": "{:.1%}", "F/Eg (MMscf)": "{:,.0f}",
                "Apparent G (MMscf)": "{:,.0f}"}), hide_index=True)

        fk = mb.fetkovich
        if fk:
            st.markdown("#### Fetkovich aquifer fit")
            a = st.columns(5)
            a[0].metric("G", f"{fk['G_mmscf']:,.0f} MMscf")
            a[1].metric("Wei", f"{fk['Wei_mmbbl']:,.0f} MMbbl")
            a[2].metric("J", f"{fk['J_bbl_d_psi']:,.2f} bbl/d/psi")
            a[3].metric("We to date", f"{fk['We_mmbbl']:,.1f} MMbbl")
            a[4].metric("rms", f"{fk['rms_pct']:.2f} %")

            lo, hi = fk["g_range_mmscf"]
            wef = 100 * fk.get("we_frac_hcpv", float("nan"))
            if wef < 2.0:
                st.success(
                    f"The aquifer fit finds **{fk['We_mmbbl']:,.2f} MMbbl of "
                    f"influx — {wef:.1f}% of the reservoir's hydrocarbon pore "
                    "volume**, which is nothing. That is the cross-check "
                    "passing: this reservoir does not need an aquifer to "
                    f"explain its pressure history, and its independent G of "
                    f"{fk['G_mmscf']:,.0f} MMscf sits beside the p/z intercept "
                    f"of {mb.ogip_mmscf:,.0f} MMscf.")
            else:
                st.info(
                    f"**Accounting for influx, G is {fk['G_mmscf']:,.0f} MMscf** "
                    f"— against {mb.ogip_mmscf:,.0f} MMscf from the straight "
                    "line, which assumed there was none. The model needs "
                    f"{fk['We_mmbbl']:,.1f} MMbbl of water to have entered the "
                    f"reservoir — {wef:.0f}% of its hydrocarbon pore volume — "
                    "to hold the pressure up as observed.")

            width = hi / lo if lo > 0 else float("inf")
            if width > 1.15:
                warn("<b>The solution is not unique.</b> A large aquifer with "
                     "a small J and a small aquifer with a large J bend the "
                     "same pressure history over a finite record; only late "
                     "depletion of the aquifer itself separates them. The data "
                     f"constrain G only to <b>{lo:,.0f} – {hi:,.0f} MMscf</b>, "
                     "and that range — not the single best triplet above — is "
                     "the honest output. The chart below is the valley being "
                     "quoted.")
            else:
                note(f"The fit is well constrained here: G lies between "
                     f"<b>{lo:,.0f} and {hi:,.0f} MMscf</b> "
                     f"({100 * (width - 1):.0f}% wide). That is unusual — the "
                     "aquifer solution is normally a long valley rather than a "
                     "point, because a big aquifer with a small J and a small "
                     "one with a big J bend the same pressure history.")

            f1, f2 = st.columns(2)
            with f1:
                show_fig(ch.chart_aquifer_match(res, THEME), key="aq_match")
            with f2:
                show_fig(ch.chart_aquifer_locus(res, THEME), key="aq_locus")
            show_fig(ch.chart_aquifer_influx(res, THEME), key="aq_we")
            if fk["rms_pct"] > 10:
                warn(f"rms {fk['rms_pct']:.1f}% is above 10% — the model "
                     "cannot reproduce the pressure history. Treat the "
                     "parameters as indicative only.")
            with st.expander("Locus — best achievable fit at each G"):
                show_df(fk["locus"].style.format({
                    "G_mmscf": "{:,.0f}", "Wei_mmbbl": "{:,.1f}",
                    "tau_days": "{:,.0f}", "J_bbl_d_psi": "{:,.2f}",
                    "rms_pct": "{:.2f}"}), hide_index=True)

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
    st.code(tidy_summary_text(res.forecast.summary()), language="text")
    c1, c2 = st.columns(2)
    with c1:
        show_fig(ch.chart_rate_time(res, THEME, show_separator=False),
                 key="fx_gas")
    with c2:
        show_fig(ch.chart_condensate(res, THEME), key="fx_cond")
    with st.expander("Forecast table"):
        show_df(tidy_forecast(res.forecast.table), height=400)

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
    tot = summary[["EUR_wellstream_mmscf", "EUR_sep_gas_mmscf",
                   "EUR_condensate_mstb"]].sum()
    k = st.columns(4)
    k[0].metric("Wells analysed", len(summary))
    k[1].metric("Field EUR wellstream gas",
                f"{tot['EUR_wellstream_mmscf']:,.0f} MMscf")
    k[2].metric("Field EUR separator gas",
                f"{tot['EUR_sep_gas_mmscf']:,.0f} MMscf")
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
                       tidy_forecast(res.forecast.table).to_csv(index=False).encode(),
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
                tidy_forecast(r.forecast.table).to_excel(
                    xl, sheet_name=f"{tag}_fcst"[:31], index=False)
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
