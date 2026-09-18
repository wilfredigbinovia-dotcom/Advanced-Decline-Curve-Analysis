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
from typing import Dict, Optional, Tuple

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
            "date", help="ISO 8601 only: 2021-03-01. 01/03/2021 is refused "
                         "because it is 1 March in most of the world and "
                         "3 January in the US, and nothing says which.",
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
    # reset_index matters: `out` is built on a fresh RangeIndex, so assigning
    # a column that still carries the caller's index makes pandas align on it
    # and silently fill NaN wherever the two disagree. Any frame that has been
    # filtered - a row dropped, rows reordered - arrives here with gaps in its
    # index and loses its first rows to that alignment.
    d = dca.map_columns(df).copy().reset_index(drop=True)
    out = pd.DataFrame(index=range(len(d)))
    for c in PASTE_COLUMNS:
        if c in ("date", "well"):
            out[c] = (d[c].astype(str) if c in d.columns
                      else pd.Series([""] * len(d), dtype="object"))
            if c == "date" and c in d.columns:
                # A non-ISO column is left exactly as it was written, so the
                # grid shows the offending text rather than a wall of blanks.
                try:
                    parsed = dca.parse_dates(d[c])
                    out[c] = parsed.dt.strftime("%Y-%m-%d").fillna(
                        d[c].astype(str))
                except dca.DateFormatError:
                    pass
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

    parsed = dca.parse_dates(d["date"])      # DateFormatError handled by caller
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


def date_format_stop(exc: "dca.DateFormatError") -> None:
    """One clear stop for a date column the app refuses to guess at."""
    st.error(f"**The date column is not in ISO format.** {exc}")
    with st.expander("Why this is an error and not a best guess",
                     expanded=True):
        st.markdown(
            "`01/05/2012` is 1 May in most of the world and 5 January in the "
            "United States, and nothing in the file says which. Read the "
            "wrong way round, a monthly history becomes a fortnight: the rows "
            "stay in order, the cumulative still adds up, and every rate, "
            "decline and pressure gap is out by a factor of thirty. Nothing "
            "looks broken — only the answer changes.\n\n"
            "Detecting the order from the shape of the column was tried and "
            "is not reliable enough. It needs more than twelve months of data "
            "before day-first is distinguishable at all, so a short paste is "
            "read backwards in silence, and a column of survey dates that "
            "happen to fall on low day numbers stays ambiguous however long "
            "it is.\n\n"
            "**The fix, in Excel:** select the date column → Format Cells → "
            "Custom → `yyyy-mm-dd`, then copy again.\n\n"
            "**In pandas** — set `dayfirst` to match your file, not mine:\n")
        st.code("df['date'] = (pd.to_datetime(df['date'], dayfirst=True)\n"
                "                .dt.strftime('%Y-%m-%d'))", language="python")
        st.markdown(
            "A column that is already a real date type — an Excel date cell, "
            "a parquet timestamp — is accepted as it is. No text is parsed, "
            "so there is nothing to be ambiguous about.")
    st.stop()


def write_initial_pressure(frame: pd.DataFrame, p_initial: float,
                           ours: Optional[float] = None,
                           clear: bool = False) -> Tuple[pd.DataFrame, str]:
    """Put the estimated p_i on the FIRST row of the data, in `p_res`.

    Strictly the two are not the same quantity - `p_res` on the first row means
    the pressure once that row's gas had been produced, and the estimate is the
    pressure before any of it was. The gap is one period's cumulative, which on
    a monthly history is a fraction of a percent of gas in place: on the worked
    well it moved the p/z intercept by 0.14 % and the We >= 0 ceiling not at
    all. The material balance reference is set separately and explicitly from
    the sidebar, so nothing downstream depends on this cell being at exactly
    zero cumulative.

    Against that, a value written into the row the data already has is easier
    to see, easier to edit and easier to delete than an extra row appearing
    above the record - which is the whole reason for preferring it.

    A first row that ALREADY carries a pressure is never overwritten. That is
    a measurement, and the estimator would not have offered in the first place
    if the record began with one. The one exception is a value this app wrote
    itself: pass it as `ours` and it is replaced, so re-estimating updates the
    cell instead of refusing it. `clear` blanks that same cell, which is what
    Undo needs - otherwise the estimate stays in the table after being
    withdrawn from everywhere else.

    Returns the frame and one of: "written", "cleared", "kept_existing",
    "no_rows".
    """
    fr = align_to_paste_columns(frame) if not set(PASTE_COLUMNS) <= set(
        frame.columns) else frame.copy()
    try:
        dates = dca.parse_dates(fr["date"])
    except dca.DateFormatError:
        dates = pd.Series(pd.NaT, index=fr.index, dtype="datetime64[ns]")
    if not dates.notna().any():
        return align_to_paste_columns(fr), "no_rows"

    # Clear out an estimate written by an older version of this app, which put
    # it on a row of its own dated before first gas. Left in place it would sit
    # there as a duplicate of the value about to be written below.
    gas = pd.to_numeric(fr["q_gas"], errors="coerce").fillna(0.0)
    pres = pd.to_numeric(fr["p_res"], errors="coerce")
    legacy = dates.notna() & (dates == dates.min()) & (gas <= 0) & pres.notna()
    if legacy.any() and int((~legacy).sum()) > 0:
        fr = fr[~legacy].reset_index(drop=True)
        dates = dates[~legacy].reset_index(drop=True)
        pres = pd.to_numeric(fr["p_res"], errors="coerce")

    first_idx = dates.idxmin()
    current = float(pres.get(first_idx, np.nan))
    is_ours = (ours is not None and np.isfinite(current)
               and abs(current - float(ours)) < 1e-6)
    if clear:
        if is_ours:
            fr.loc[first_idx, "p_res"] = np.nan
            return align_to_paste_columns(fr), "cleared"
        return align_to_paste_columns(fr), "kept_existing"
    if np.isfinite(current) and not is_ours:
        return align_to_paste_columns(fr), "kept_existing"
    fr.loc[first_idx, "p_res"] = float(p_initial)
    return align_to_paste_columns(fr), "written"


@st.cache_resource(show_spinner=False)
def build_pvt(gas_gravity: float, temperature_F: float, condensate_api: float,
              condensate_mw: Optional[float], p_dew: Optional[float],
              p_init: Optional[float], y_n2: float, y_co2: float, y_h2s: float,
              initial_cgr: Optional[float], use_wellstream: bool,
              cvd_key: Optional[tuple],
              rp_key: Optional[tuple] = None) -> dca.PVT:
    """PVT objects pre-compute property tables, so they are cached by value."""
    cvd = None
    if cvd_key:
        p, n, ld = zip(*cvd_key)
        ld_arr = np.array(ld, dtype=float)
        cvd = dca.CVDTable(pressure=np.array(p),
                           cum_produced_molfrac=np.array(n),
                           liquid_dropout=(ld_arr
                                           if np.any(np.isfinite(ld_arr))
                                           else None))
    relperm = None
    if rp_key:
        swi, sorg, ng, no, ratio = rp_key
        relperm = dca.RelPerm(swi=swi, sorg=sorg, ng=ng, no=no,
                              bank_saturation_ratio=ratio)
    return dca.PVT(gas_gravity=gas_gravity, temperature_F=temperature_F,
                   condensate_api=condensate_api, condensate_mw=condensate_mw,
                   p_dew=p_dew, p_init=p_init, y_n2=y_n2, y_co2=y_co2,
                   y_h2s=y_h2s, cvd=cvd, initial_cgr=initial_cgr,
                   use_wellstream_gravity=use_wellstream, relperm=relperm)


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
     mb_pi, mb_skip, use_aq, ogip_mode, aquifer_model,
     q_water_lim, wcut_lim, pi_estimated) = settings
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
                aquifer_model=aquifer_model,
                q_water_econ_stbd=(q_water_lim if q_water_lim > 0 else None),
                water_cut_econ=(wcut_lim / 100.0 if wcut_lim > 0 else None),
                use_fmb=use_fmb,
                apply_ogip_cap=cap, ogip_cap_mode=ogip_mode,
                fit_from_bdf=fit_from_bdf,
                fit_window_days=window, verbose=False,
                p_initial_estimated=pi_estimated)
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
            "ended_by": r.forecast.abandonment_reason,
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

    with st.expander("Condensate bank (two-phase pseudo-pressure)"):
        note("Fevang-Whitson. Retrograde liquid around the wellbore takes "
             "relative permeability from the gas; the single-phase m(p) knows "
             "nothing about it, so a flowing material balance attributes the "
             "lost deliverability to a smaller reservoir. Needs the CVD "
             "<b>liquid dropout</b> column below.")
        use_bank = st.toggle("Use two-phase pseudo-pressure m*(p)", value=False)
        bc1, bc2 = st.columns(2)
        rp_swi = bc1.number_input("Swi", 0.0, 0.7, 0.20, 0.01,
                                  disabled=not use_bank)
        rp_sorg = bc2.number_input("Sorg (immobile condensate)", 0.0, 0.5,
                                   0.10, 0.01, disabled=not use_bank)
        bc1, bc2 = st.columns(2)
        rp_ng = bc1.number_input("Corey n_g", 1.0, 6.0, 3.0, 0.5,
                                 disabled=not use_bank)
        rp_no = bc2.number_input("Corey n_o", 1.0, 6.0, 3.0, 0.5,
                                 disabled=not use_bank)
        rp_ratio = st.slider(
            "Bank saturation ratio", 1.0, 3.0, 1.0, 0.1, disabled=not use_bank,
            help="The CVD dropout column is a cell AVERAGE. The bank around "
                 "the wellbore is richer, because it is fed by gas flowing in "
                 "from everywhere. At 1.0 you get the reservoir average, "
                 "which understates the bank — a deliberate lower bound. "
                 "Fevang and Whitson's published examples sit nearer 1.5–2.5. "
                 "Setting it properly needs a black-oil table this app does "
                 "not ask for, so it is yours to judge.")

    with st.expander("CVD table (two-phase z)"):
        note("From the lab CVD report. Without it the Rayes correlation is "
             "used, which is a regression rather than your fluid. The top row "
             "is the <b>dew point</b>, where nothing has been produced yet, "
             "and the table should reach below the lowest reservoir pressure "
             "you expect. Lab reports usually quote both right-hand columns "
             "as <b>percentages</b> — paste them either way, they are "
             "converted if they exceed 1.")
        use_cvd = st.toggle("Use CVD table", value=True)
        cvd_df = st.data_editor(DEFAULT_CVD, num_rows="dynamic",
                                key="cvd_editor",
                                disabled=not use_cvd)

    st.divider()
    st.markdown("### Analysis settings")
    q_econ = st.number_input("Economic limit (Mscf/d)", 1.0, 100000.0, 400.0,
                             25.0)
    with st.expander("Water and liquid-handling limits"):
        note("A gas rate is rarely what ends a wet gas well. Set what the "
             "facility can actually take and the forecast stops at whichever "
             "constraint binds first — the tab reports which one, and when "
             "each of the others would have.")
        q_water_lim = st.number_input(
            "Water rate limit (STB/d, 0 = off)", 0.0, 1.0e6, 0.0, 100.0,
            help="Produced water is forecast from a log-linear trend fitted "
                 "to its own history, not from the gas decline. The trend is "
                 "only used when it is statistically significant.")
        wcut_lim = st.slider(
            "Water cut limit (% of liquid, 0 = off)", 0, 99, 0, 1,
            help="Water as a fraction of total produced liquid — water plus "
                 "condensate, on a stock-tank basis.")
    model_choice = st.selectbox(
        "Decline model", ["modified_hyperbolic", "arps", "ple", "sepd",
                          "duong", "auto"], index=0,
        help="Modified hyperbolic is the default because it is the one with a "
             "defensible late-time limit. 'auto' picks the lowest AIC, which "
             "is a statistical choice, not a physical one.")
    terminal = st.slider("Terminal decline (%/yr)", 2.0, 25.0, 7.0, 0.5)
    t_max = st.slider(
        "Forecast horizon (years from last record)", 1, 60, 30, 1,
        help="How far past the end of the history to roll the decline "
             "forward. This is forecast length, not total well life: a well "
             "with 14 years of history and a 10-year horizon is abandoned at "
             "24 years on production.")

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
        aq_label = st.selectbox(
            "Aquifer model", ["Fetkovich (pseudo-steady)",
                              "Carter-Tracy (transient)"], index=0,
            help="Fetkovich is pseudo-steady from t = 0, so it cannot produce "
                 "the large early influx of an aquifer still in transient "
                 "flow and compensates by inflating Wei — on a synthetic "
                 "transient aquifer it reads G 27 % high. Carter-Tracy "
                 "carries the transient explicitly and recovers it to within "
                 "1 %. Fetkovich is the safer default on a long, "
                 "pseudo-steady record; try both and compare the locus.")
        aquifer_model = ("carter_tracy" if aq_label.startswith("Carter")
                         else "fetkovich")
        use_aq = st.toggle(
            "Fit an aquifer", value=True,
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

rp_key = ((float(rp_swi), float(rp_sorg), float(rp_ng), float(rp_no),
           float(rp_ratio)) if use_bank else None)

cvd_key = None
if use_cvd and cvd_df is not None and len(cvd_df.dropna()) >= 3:
    clean = cvd_df.dropna(subset=["pressure_psia", "cum_produced_molfrac"])
    ld_col = ("liquid_dropout_frac" if "liquid_dropout_frac" in clean.columns
              else None)
    ld_vals = (pd.to_numeric(clean[ld_col], errors="coerce").astype(float)
               if ld_col else pd.Series([np.nan] * len(clean)))
    cvd_key = tuple(zip(clean["pressure_psia"].astype(float),
                        clean["cum_produced_molfrac"].astype(float),
                        ld_vals))

try:
    pvt = build_pvt(gas_gravity, temperature_F, condensate_api,
                    (mw_override or None), (p_dew or None), (p_init or None),
                    Y_N2, Y_CO2, Y_H2S, (initial_cgr or None),
                    use_wellstream, cvd_key, rp_key)
except Exception as exc:
    st.error(f"The fluid definition is not valid: {exc}")
    st.stop()

if pvt.cvd is not None and getattr(pvt.cvd, "unit_note", ""):
    with st.sidebar:
        note(f"<b>CVD units.</b> {pvt.cvd.unit_note[0].upper()}"
             f"{pvt.cvd.unit_note[1:]}")

if getattr(pvt, "cvd_warning", ""):
    with st.sidebar:
        warn(f"<b>CVD table.</b> {pvt.cvd_warning[0].upper()}"
             f"{pvt.cvd_warning[1:]}")

pvt_sig = (gas_gravity, temperature_F, condensate_api, mw_override, p_dew,
           p_init, Y_N2, Y_CO2, Y_H2S, initial_cgr, use_wellstream, cvd_key,
           rp_key)

# ==============================================================================
# Header and data intake
# ==============================================================================

st.title("Gas condensate decline curve analysis")
st.caption(f"v{dca.__version__}")
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
               "any other column blank. Dates must be **ISO: YYYY-MM-DD** "
               "(`2012-05-01`); `01/05/2012` is ambiguous and is refused "
               "rather than guessed at.")

    if st.session_state.pop("_load_sample", False):
        st.session_state.pop("paste_editor", None)
        st.session_state["paste_frame"] = align_to_paste_columns(
            parse_pasted_text(SAMPLE_PASTE))
    if st.session_state.pop("_clear_grid", False):
        st.session_state.pop("paste_editor", None)
        st.session_state["paste_frame"] = blank_paste_frame()
    _clear_pi = st.session_state.pop("_stage_grid_clear", None)
    _pending_pi = st.session_state.pop("_stage_grid_pi", None)
    if _clear_pi is not None:
        base_fr = st.session_state.get("_paste_current")
        if base_fr is None:
            base_fr = st.session_state.get("paste_frame", blank_paste_frame())
        try:
            _fr, _ = write_initial_pressure(base_fr, 0.0, ours=float(_clear_pi),
                                            clear=True)
            st.session_state["paste_frame"] = _fr
            st.session_state.pop("paste_editor", None)
        except Exception:
            pass
    if _pending_pi is not None:
        # The editor's live contents, stashed on the previous run: rebuilding
        # from `paste_frame` alone would silently undo anything typed since.
        base_fr = st.session_state.get("_paste_current")
        if base_fr is None:
            base_fr = st.session_state.get("paste_frame", blank_paste_frame())
        try:
            _fr, _status = write_initial_pressure(
                base_fr, float(_pending_pi),
                ours=st.session_state.get("_pi_written"))
            st.session_state["paste_frame"] = _fr
            st.session_state.pop("paste_editor", None)
            if _status == "written":
                st.session_state["_pi_written"] = float(_pending_pi)
            if _status == "kept_existing":
                st.warning(
                    "The first row of the table already carries a reservoir "
                    "pressure, so it was left alone — that is a measurement, "
                    "and the estimate does not belong on top of it. The "
                    "estimate is still applied to the fluid definition and to "
                    "the material balance reference.")
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
                         "...  header row first, one row per period, "
                         "dates as YYYY-MM-DD"))
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
    try:
        parsed_df, paste_msgs = parse_paste_grid(paste_grid)
    except dca.DateFormatError as exc:
        date_format_stop(exc)
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
| `date` | — | yes | **ISO only: `YYYY-MM-DD`** |
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

# Check the dates once, here, where the message can be a page rather than a
# traceback buried in a per-well error list. Everything downstream may assume
# the column is ISO and parsed.
try:
    dca.parse_dates(work_df["date"])
except dca.DateFormatError as exc:
    date_format_stop(exc)

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
               "puts it in the `p_res` cell on the first row of the table.")
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
        f"**Initial pressure estimated at {applied:,.0f} psia** — applied to "
        "the fluid definition, used as the material balance reference, and "
        "written into `p_res` on the first row of the table.")
    if c2.button("Undo", width="stretch"):
        st.session_state["_stage_p_init"] = 6400.0
        st.session_state["_stage_mb_pi"] = 0.0
        # Withdraw it from the table as well. Leaving the estimate sitting in
        # p_res after undoing it everywhere else would turn a withdrawn guess
        # into what looks like a measurement.
        _w = st.session_state.pop("_pi_written", None)
        if _w is not None:
            st.session_state["_stage_grid_clear"] = _w
            st.session_state["_pi_absorb_sig"] = True
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
            ogip_mode, aquifer_model, float(q_water_lim),
            float(wcut_lim),
            # Whether the p_i sitting in the table was ESTIMATED by this tool
            # rather than measured. It has to travel with the settings so the
            # cached analysis is invalidated when it changes, and so the
            # material balance can say that its earliest survey is its own
            # extrapolation rather than an independent gauge reading.
            bool(st.session_state.get("_pi_written") is not None))
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
m[3].metric("Economic life", f"{res.forecast.economic_life_years:.1f} yr",
            f"ended by {res.forecast.abandonment_reason}", delta_color="off",
            help="From first production. Which constraint bound, and when "
                 "each of the others would have, is on the Forecast tab.")
# This tile used to read `matbal.ogip_mmscf` — the p/z intercept — whatever
# the sidebar said, so selecting Fetkovich G changed nothing here and a record
# with no intercept printed "nan MMscf". It now shows the gas in place actually
# chosen, and names which one it is.
_oc = res.ogip_choice
if _oc is not None and np.isfinite(_oc.value):
    m[4].metric("OGIP (material balance)", f"{_oc.value:,.0f} MMscf",
                _oc.source, delta_color="off",
                help="Follows **Which gas in place** in the sidebar. See the "
                     "Material balance tab for every candidate and why this "
                     "one was used.")
else:
    m[4].metric("OGIP (material balance)", "n/a", delta_color="off",
                help=((_oc.reason if _oc is not None else "")
                      or "No material balance on this well."))

if fit.at_bounds:
    if fit.at_bounds == ["b"] and fit.params.get("b", 1.0) < 1e-6:
        note("<b>b sits at zero</b>, so the data are exponential. That is a "
             "legitimate answer for a depleting gas well, not a failed fit.")
    else:
        warn("<b>" + ", ".join(fit.at_bounds) + "</b> pinned to a bound — the "
             "fit is not identifiable. Narrow the window, fix a parameter, or "
             "try another model.")

tabs = st.tabs(["Data & QC", "Decline fit", "Yield", "Material balance",
                "Forecast", "Uncertainty", "Field", "Export", "About"])

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

    lc, wl = res.liquid_check, res.water_in_liquid
    if lc is not None and lc.ok and lc.exceeds:
        warn(f"<b>The reported liquid is more than this fluid can produce.</b> "
             f"The recent produced CGR of <b>{lc.cgr_median_recent:,.0f} "
             f"STB/MMscf</b> is <b>{lc.ratio_recent:.1f}×</b> the initial CGR "
             f"of {lc.initial_cgr:,.0f}, in {lc.n_periods_over} of "
             f"{lc.n_periods} periods. Below the dew point a retrograde gas "
             "gets <i>leaner</i> — the heavy ends drop out in the reservoir "
             "and stay there — so the produced ratio falls away from its "
             "initial value and cannot climb back past it. That implies "
             f"<b>{100 * lc.implied_non_condensate:.0f}%</b> of the reported "
             "liquid is not condensate. The usual causes are water the "
             "separator never split out, a test that measured total liquid, "
             "or a drifting allocation factor — and all three inflate "
             "reserves.")
        if lc.by_year is not None and len(lc.by_year):
            with st.expander("Produced CGR against the fluid's ceiling, by year"):
                t = lc.by_year.rename(columns={
                    "cgr": "median CGR (STB/MMscf)", "ratio": "× initial",
                    "implied_non_condensate": "implied non-condensate"})
                show_df(t.style.format({
                    "median CGR (STB/MMscf)": "{:,.0f}", "× initial": "{:,.1f}",
                    "implied non-condensate": "{:.0%}"}), hide_index=True)
    elif lc is not None and lc.ok:
        st.success(f"**The liquid stream is consistent with the fluid.** "
                   f"Recent produced CGR {lc.cgr_median_recent:,.0f} against "
                   f"an initial {lc.initial_cgr:,.0f} STB/MMscf, which is the "
                   "ceiling a depleting retrograde gas cannot exceed.")

    if wl is not None and wl.ok:
        if wl.water_driven:
            warn(f"<b>The CGR is tracking water cut</b> (Spearman ρ = "
                 f"{wl.correlation:+.2f}, p = {wl.p_value:.1e}, n = {wl.n}). A "
                 "retrograde reservoir has no mechanism to raise its yield as "
                 "water arrives, so this is the water being measured as "
                 "liquid rather than a change in the reservoir. Extrapolating "
                 f"the line back to zero water cut gives <b>{wl.clean_cgr:,.0f} "
                 "STB/MMscf</b> — compare that against the initial CGR above.")
        else:
            note(f"CGR does not track water cut (ρ = {wl.correlation:+.2f}, "
                 f"p = {wl.p_value:.1e}), so whatever is moving the yield, it "
                 "is not simply produced water arriving in the liquid stream.")

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
        if truth and np.isfinite(res.matbal.ogip_mmscf):
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
        k[0].metric("OGIP (p/z line)",
                    f"{mb.ogip_mmscf:,.0f} MMscf"
                    if np.isfinite(mb.ogip_mmscf) else "no intercept",
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

        if np.isfinite(mb.ogip_mmscf) and mb.ogip_mmscf > 0 and np.isfinite(
                mb.ogip_stderr):
            _rse = mb.ogip_stderr / mb.ogip_mmscf
            if _rse > dca.PZ_MAX_REL_SE:
                warn(f"<b>The p/z intercept is not a usable number.</b> It "
                     f"reads {mb.ogip_mmscf:,.0f} MMscf ± "
                     f"{100 * _rse:,.0f} % — a big number divided by a slope "
                     "that is nearly zero. A line this flat has no meaningful "
                     "x-intercept, so it is excluded from the forecast cap "
                     "and should not be quoted. The ceiling and the aquifer "
                     "fit are the numbers to read here.")

        if pvt.cvd is not None and len(mb.pressure):
            _cov = pvt.cvd.coverage_note(float(np.min(mb.pressure)),
                                         float(np.max(mb.pressure)))
            if _cov:
                warn(f"<b>The CVD table does not span these surveys.</b> "
                     f"{_cov[0].upper()}{_cov[1:]}")

        if mb.pz_note:
            warn(f"<b>No straight-line OGIP.</b> {mb.pz_note[0].upper()}"
                 f"{mb.pz_note[1:]}<br>Everything else on this tab still "
                 "applies: F/Eg, the We ≥ 0 ceiling, the apparent-G sequence "
                 "and the Fetkovich fit need no straight line, and they are "
                 "the right instruments when the pressure is being held up.")

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
        fk_bad = dca.fetkovich_health(fk) if fk else []
        if fk and fk_bad:
            st.markdown("#### Fetkovich aquifer fit")
            warn("<b>This fit did not converge on anything meaningful, so the "
                 "numbers below are not reported.</b><br>"
                 + "<br>".join(f"· {b[0].upper()}{b[1:]}." for b in fk_bad)
                 + "<br>An optimiser always returns a triplet. On data that "
                 "cannot constrain it — a degenerate two-phase z, no initial "
                 "pressure, or too few surveys far enough into depletion — it "
                 "returns one that looks like every other answer this tab "
                 "prints. Fix the input the checks above point at and the fit "
                 "becomes readable; until then there is nothing here to use.")
            with st.expander("Show the failed fit anyway"):
                st.caption("For diagnosis only — do not quote these.")
                b = st.columns(5)
                b[0].metric("G", f"{fk['G_mmscf']:,.0f} MMscf")
                b[1].metric("Wei", f"{fk['Wei_mmbbl']:,.0f} MMbbl")
                b[2].metric("J", f"{fk['J_bbl_d_psi']:,.2f} bbl/d/psi")
                b[3].metric("We to date", f"{fk['We_mmbbl']:,.1f} MMbbl")
                b[4].metric("rms", f"{fk['rms_pct']:.2f} %")
        elif fk:
            _mdl = ("Carter-Tracy" if fk.get("model") == "carter_tracy"
                    else "Fetkovich")
            st.markdown(f"#### {_mdl} aquifer fit")
            a = st.columns(5)
            a[0].metric("G", f"{fk['G_mmscf']:,.0f} MMscf")
            if fk.get("model") == "carter_tracy":
                a[1].metric("B'", f"{fk['Bprime_bbl_psi']:,.0f} bbl/psi",
                            f"Wei equiv {fk['Wei_mmbbl']:,.0f} MMbbl",
                            delta_color="off")
                a[2].metric("t_D at end", f"{fk['td_at_end']:,.1f}",
                            "still transient" if fk.get("transient")
                            else "pseudo-steady reached", delta_color="off",
                            help="Below about 100 the aquifer is still in "
                                 "transient flow, which is where Carter-Tracy "
                                 "earns its keep over Fetkovich.")
            else:
                a[1].metric("Wei", f"{fk['Wei_mmbbl']:,.0f} MMbbl")
                a[2].metric("J", f"{fk['J_bbl_d_psi']:,.2f} bbl/d/psi")
            a[3].metric("We to date", f"{fk['We_mmbbl']:,.1f} MMbbl")
            a[4].metric("rms", f"{fk['rms_pct']:.2f} %")
            if (fk.get("model") == "carter_tracy" and fk.get("transient")
                    and 100 * fk.get("we_frac_hcpv", 0.0) >= 2.0):
                note("The aquifer is still in <b>transient</b> flow over this "
                     "record (t_D below 100). A pseudo-steady law cannot "
                     "produce the early influx that implies, so Fetkovich "
                     "would inflate Wei to compensate and read gas in place "
                     "high. This is the case Carter-Tracy exists for — worth "
                     "running both and comparing the locus.")

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
                    + (f"of {mb.ogip_mmscf:,.0f} MMscf."
                       if np.isfinite(mb.ogip_mmscf)
                       else "— which this record does not have."))
            else:
                st.info(
                    f"**Accounting for influx, G is {fk['G_mmscf']:,.0f} MMscf** "
                    + (f"— against {mb.ogip_mmscf:,.0f} MMscf from the "
                       "straight line, which assumed there was none. "
                       if np.isfinite(mb.ogip_mmscf) else
                       "— the straight line gives no intercept on this "
                       "record, so there is nothing to compare it with. ")
                    + "The model needs "
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
    bank = res.bank
    if bank is not None and bank.ok:
        st.markdown("**Condensate bank — productivity index**")
        show_fig(ch.chart_bank_pi(res, THEME), key="mb_pi")
        note(f"q / [m(p_avg) − m(p_wf)] divides out the drawdown and the gas "
             f"properties, so what is left is mobility and contacted volume. "
             f"It has fallen <b>{100 * bank.loss_frac:,.0f}%</b> "
             f"({bank.trend_pct_per_year:+,.0f} %/yr, R² {bank.r2:.2f})"
             + (f", and the reservoir crossed the dew point at "
                f"{bank.p_dew_crossed_days / dca.DAYS_PER_YEAR:,.1f} years"
                if bank.p_dew_crossed_days is not None else "")
             + ". A fall that begins at that crossing is the condensate bank, "
             "in the only units that matter to a forecast — lost "
             "deliverability.")
    elif bank is not None:
        note(f"<b>No bank diagnostic.</b> {bank.reason[0].upper()}"
             f"{bank.reason[1:]}. It needs a flowing pressure column, a gas "
             "in place to set the average reservoir pressure, and an initial "
             "pressure.")

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
        if f.get("pseudo_pressure", "").startswith("two-phase"):
            loss = 100 * f.get("bank_mobility_loss_last", float("nan"))
            spread = 100 * f.get("bank_loss_spread", float("nan"))
            corr = 100 * f.get("bank_correction", float("nan"))
            r2_1p = f.get("r2_single_phase", float("nan"))
            msg = (f"<b>Fevang-Whitson two-phase pseudo-pressure.</b> The bank "
                   f"has taken <b>{loss:.0f} %</b> of the gas mobility by the "
                   f"last point, and m*(p) puts that where it belongs — in "
                   f"the mobility, not the volume. R² "
                   f"{r2_1p:.3f} → {f['r2']:.3f}.")
            if np.isfinite(spread) and spread < 10:
                msg += (f"<br>Gas in place moved {corr:+.1f} %, and that is "
                        f"the honest result rather than a disappointment: the "
                        f"loss varies by only {spread:.0f} points across the "
                        "window, and multiplying Δm by something close to a "
                        "constant leaves the x-intercept exactly where it "
                        "was. A bank corrects the <b>line</b>; it only moves "
                        "the <b>answer</b> when its severity changes as the "
                        "reservoir depletes.")
            else:
                msg += (f"<br>Gas in place moved <b>{corr:+.1f} %</b> — the "
                        f"loss varies by {spread:.0f} points across the "
                        "window, enough to change the slope and so the "
                        "intercept.")
            note(msg)
        elif f.get("bias_warning"):
            warn("The reservoir is below the dew point over part of this "
                 "window, so condensate banking degrades mobility faster than "
                 "depletion alone and this figure is biased <b>low</b>. Treat "
                 "it as a lower bound, not a competing OGIP. Switch on "
                 "<b>Condensate bank</b> in the sidebar to correct it with a "
                 "two-phase pseudo-pressure.")

# ------------------------------------------------------------------- Forecast
with tabs[4]:
    fcast = res.forecast
    cy = fcast.constraint_years or {}
    if len(cy) > 1:
        rows = sorted(cy.items(), key=lambda kv: kv[1])
        binding = rows[0][0]
        k = st.columns(len(rows))
        for i, (name, yr) in enumerate(rows):
            k[i].metric(name.title(), f"{yr:,.1f} yr",
                        "binding" if name == binding else "not reached",
                        delta_color=("inverse" if name == binding else "off"))
        nxt = rows[1]
        note(f"The forecast ends on <b>{binding}</b> at "
             f"{rows[0][1]:,.1f} years. The next constraint, <b>{nxt[0]}</b>, "
             f"would not have bound until {nxt[1]:,.1f} years — so quoting "
             f"the life off {nxt[0]} alone would have overstated it by "
             f"<b>{nxt[1] - rows[0][1]:,.1f} years</b>.")
    elif fcast.water_trend is None and "q_water" in res.data.df.columns:
        note("No water constraint applied. Produced water is in the data but "
             "either its trend is not statistically significant or no limit "
             "is set — see <b>Water and liquid-handling limits</b> in the "
             "sidebar.")
    st.code(tidy_summary_text(res.forecast.summary()), language="text")
    c1, c2 = st.columns(2)
    with c1:
        show_fig(ch.chart_rate_time(res, THEME, show_separator=False),
                 key="fx_gas")
    with c2:
        show_fig(ch.chart_condensate(res, THEME), key="fx_cond")
    if fcast.water_trend is not None:
        wt = fcast.water_trend
        show_fig(ch.chart_water_forecast(res, THEME, q_limit=q_water_lim),
                 key="fx_water")
        note(f"Water is fitted on its own history, not derived from the gas "
             f"decline: <b>{wt.growth_pct_per_year:+,.0f} %/yr</b> "
             f"(R² {wt.r2:.2f}, p {wt.p_value:.1e}, n = {wt.n_points}), last "
             f"measured {wt.q_last_stbd:,.0f} STB/d. The trend is only used "
             "when it is significant; a log-linear extrapolation of water is "
             "crude, and it is shown so it can be disbelieved.")
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


# ---------------------------------------------------------------------- About
with tabs[8]:
    # Numbers in this tab come from one of two places and must never be
    # confused. VERIFY_CASE figures are from the module's own synthetic
    # reservoirs, where the answer is known and the reader can reproduce them;
    # live() figures are computed from whatever well is selected right now.
    # A number from some other well, quoted as though it were general, is the
    # one thing this tab must not do.
    VERIFY = ("in the module's verification case — a synthetic reservoir whose "
              "answer is known, reproducible with "
              "`python gas_condensate_dca.py --selftest`")

    def live(text: str) -> None:
        note(f"<b>On {sel}, as currently loaded:</b> {text}")

    st.markdown("## What decline curve analysis is")
    st.markdown(
        "A producing well's rate falls over time in a way that is remarkably "
        "regular once the pressure disturbance has reached the boundaries of "
        "its drainage area — boundary-dominated flow. Decline curve analysis "
        "fits a curve to that history and integrates it forward to an economic "
        "limit. The area under the curve is the estimated ultimate recovery.\n\n"
        "It is an **empirical** method. It does not know the rock, the fluid, "
        "or the geometry; it knows what the well has done. That is its great "
        "strength — it needs only rates and dates, which every well has — and "
        "its central weakness: a curve fitted to the past has no idea what the "
        "reservoir is capable of, and will happily forecast more gas than "
        "exists.")

    st.markdown("### The classic method")
    st.markdown(
        "Arps (1945) wrote the decline rate as a power of the rate itself, "
        "which integrates to one family of curves:")
    st.latex(r"q(t)=\frac{q_i}{\left(1+b\,D_i\,t\right)^{1/b}}"
             r"\qquad D(t)=-\frac{1}{q}\frac{dq}{dt}=\frac{D_i}{1+b\,D_i\,t}")
    st.markdown(
        "The exponent **b** is the whole argument. `b = 0` is exponential — a "
        "constant fractional decline, the classic single-phase depletion case. "
        "`b = 1` is harmonic. Between them lies hyperbolic behaviour, which is "
        "what most real wells show. Above `b = 1` the integral to infinite "
        "time diverges: the curve predicts infinite gas. That is not a rounding "
        "problem, it is the model being used outside its derivation.\n\n"
        "Arps assumed boundary-dominated flow, a well on constant bottomhole "
        "pressure, one phase flowing, and no change in the way the well is "
        "operated. A gas condensate well satisfies none of those cleanly.")

    st.markdown("### Why a gas condensate well breaks it")
    st.markdown(
        "**One reservoir, two reported streams.** Gas and condensate arrive at "
        "the separator as two numbers, but they left the reservoir as a single "
        "phase. Declining them independently lets the two forecasts drift "
        "apart, and the implied condensate yield drifts with them — usually "
        "into something the fluid cannot produce.\n\n"
        "**Retrograde condensation.** Below the dew point liquid drops out "
        "*inside the reservoir*, where most of it stays. The produced yield "
        "falls for reasons that have nothing to do with the rate, and the "
        "single-phase z-factor stops describing what is in the pore space.\n\n"
        "**The condensate bank.** That liquid accumulates near the wellbore "
        "and takes relative permeability away from the gas. The well loses "
        "deliverability from a mechanism no Arps curve contains, and the "
        "apparent `b` absorbs it.\n\n"
        "**The plateau.** Early rate is usually set by a contract or a "
        "compressor, not the reservoir. Fitting Arps through a flat plateau "
        "drags `b` toward zero and the forecast with it.")

    st.markdown("## What this tool does differently")
    st.markdown(
        "| | Conventional DCA | This tool |\n"
        "|---|---|---|\n"
        "| What is declined | separator gas; condensate fitted separately or "
        "at a fixed ratio | one **wellstream** (gas-equivalent) rate, then "
        "re-split into products |\n"
        "| z-factor | single-phase | **two-phase** below the dew point, from "
        "your CVD table or the Rayes correlation |\n"
        "| Condensate | its own curve | a **CGR(Gp) yield model** evaluated on "
        "the gas forecast, so the two cannot drift apart |\n"
        "| Late time | `b` as fitted, often > 1 | **modified hyperbolic** with "
        "a terminal decline you set |\n"
        "| Fit window | all history | plateau and transient flow **detected "
        "and excluded** |\n"
        "| Initial pressure | the first survey | **back-extrapolated** from "
        "the early p/z trend |\n"
        "| Drive mechanism | assumed volumetric | **Havlena-Odeh F/Eg**, the "
        "**We ≥ 0 ceiling**, and a **Fetkovich** or **Carter-Tracy** "
        "aquifer fit |\n"
        "| Condensate bank | ignored | **Fevang-Whitson two-phase "
        "pseudo-pressure** m*(p) from your CVD dropout column |\n"
        "| Reserves bound | none | forecast **capped** at an independently "
        "estimated gas in place |\n"
        "| Uncertainty | one deterministic case | **Monte Carlo** P90/P50/P10 "
        "on the fit parameters and the cap |\n"
        "| What ends the well | the gas rate | the **earliest** of gas rate, "
        "water rate, water cut and gas in place — with water forecast on its "
        "own trend |\n"
        "| The liquid stream | taken on trust | checked against the **ceiling "
        "the fluid sets**, and against water cut |\n")

    st.markdown("### The wellstream basis")
    st.markdown(
        "Every rate is converted to the gas that actually left the reservoir "
        "before anything is fitted. Condensate is turned back into its gas "
        "equivalent:")
    st.latex(r"V_{eq}=\frac{133{,}000\,\gamma_o}{M_o}\ \mathrm{scf/STB}"
             r"\qquad q_{ws}=q_g+q_o\,V_{eq}")
    st.markdown(
        f"For the fluid currently defined that is **{pvt.v_eq:,.0f} scf/STB**. "
        "The decline is fitted to `q_ws`, and the forecast is split back into "
        "separator gas and condensate through the yield model. This is the "
        "single most important difference from a conventional workflow: it "
        "makes the gas and liquid forecasts arithmetically consistent by "
        "construction rather than by inspection.")

    st.markdown("### The two-phase z-factor")
    st.markdown(
        "Below the dew point the material balance has to account for the gas "
        "*and* the retrograde liquid still in the reservoir. The two-phase "
        "z-factor is defined so that p/z stays linear in moles produced:")
    st.latex(r"\frac{p}{z_{2\phi}}=\frac{p_d}{z_d}\left(1-n_p\right)")
    st.markdown(
        "where `n_p` is the cumulative wellstream mole fraction produced, read "
        "from your CVD table. Using the single-phase z instead **understates** "
        "gas in place once well below the dew point, because `z₂φ < z₁φ` "
        "steepens the apparent p/z trend. How much depends on how far below "
        "the dew point the reservoir has gone and how rich the fluid is, so "
        "the tool reports both numbers rather than quoting a rule of thumb.")
    if (res.matbal is not None and res.matbal.ogip_single_phase
            and np.isfinite(res.matbal.ogip_mmscf)):
        _d = 100 * (res.matbal.ogip_single_phase / res.matbal.ogip_mmscf - 1)
        live(f"single-phase z would read {res.matbal.ogip_single_phase:,.0f} "
             f"MMscf against the two-phase {res.matbal.ogip_mmscf:,.0f} — "
             f"a difference of <b>{_d:+.1f} %</b>.")
    else:
        live("no p/z line on this record, so the two cannot be compared here.")

    st.markdown("### Which aquifer model")
    st.markdown(
        "**Fetkovich** treats the aquifer as a tank draining at pseudo-steady "
        "state from t = 0. It is robust, cheap and right once the aquifer has "
        "felt its own boundaries — but it cannot produce the large early "
        "influx of an aquifer still in *transient* flow, and it compensates "
        f"by inflating Wei. On a transient aquifer {VERIFY}, it reads gas in "
        "place **27 % high** and asks for an aquifer 66× larger than the real "
        "one.\n\n"
        "**Carter-Tracy** carries the transient explicitly through the "
        "van Everdingen-Hurst dimensionless pressure, evaluated by a "
        "recursion that needs no superposition — which matters, because "
        "superposition needs a dense pressure history and a handful of "
        "build-ups is not one. On that same case it recovers G to **0.5 %**. "
        "Which model is right for *your* well is not a matter of preference: "
        "the tab reports `t_D` at the end of the record, and below about 100 "
        "the aquifer is still transient and Fetkovich is the wrong tool.")
    _fk = res.matbal.fetkovich if res.matbal else None
    if _fk and _fk.get("model") == "carter_tracy":
        live(f"t_D reaches <b>{_fk['td_at_end']:,.0f}</b> by the last survey, "
             + ("so the aquifer is **still transient** and Carter-Tracy is "
                "the right choice here." if _fk.get("transient") else
                "so the aquifer has reached pseudo-steady state — which is "
                "why Fetkovich also works on this well. Run both and compare "
                "the locus."))
    elif _fk:
        live(f"fitted with Fetkovich: G {_fk['G_mmscf']:,.0f} MMscf, rms "
             f"{_fk['rms_pct']:.2f} %. Switch the sidebar to Carter-Tracy to "
             "see whether the aquifer is still transient over this record — "
             "if it is, this G is high.")
    else:
        live("no aquifer fit on this record, so neither model applies.")

    st.markdown("### The condensate bank")
    st.markdown(
        "Below the dew point retrograde liquid drops out around the wellbore "
        "and takes relative permeability from the gas. The single-phase m(p) "
        "knows nothing about it, so a flowing material balance attributes the "
        "lost deliverability to a smaller reservoir. Following Fevang and "
        "Whitson, the pseudo-pressure is weighted by the gas relative "
        "permeability along the depletion path:")
    st.latex(r"m^{*}(p)=\int_0^{p}\frac{k_{rg}(p')}{k_{rg,max}}"
             r"\,\frac{2p'}{\mu z}\,dp'")
    st.markdown(
        "The saturation comes from **your CVD liquid-dropout column** and the "
        "curves from a Corey pair you set in the sidebar. Two honest caveats. "
        "The solution-gas term of the full integral is omitted, because it "
        "needs a black-oil table this app does not ask for. And the CVD "
        "dropout is a *cell average* — the bank near the wellbore is richer, "
        "so the **bank saturation ratio** slider scales it, with 1.0 giving "
        "the reservoir average and therefore a lower bound.\n\n"
        "What it buys is worth stating precisely, because it is not what you "
        "might expect. The x-intercept of q/Δm against Gp is **unchanged** by "
        "multiplying Δm by a constant. So a mobility loss that is large but "
        "roughly uniform across the fitted window straightens the line "
        "without moving gas in place at all; only a loss whose severity "
        "*changes* as the reservoir depletes shifts the intercept. A bank "
        "corrects the *line*; whether it corrects the *answer* depends on "
        "that spread, which the Material balance tab reports for your well.\n\n"
        f"{VERIFY[0].upper()}{VERIFY[1:]}, a 68 % mobility loss lifts R² from "
        "0.919 to 0.937 and moves gas in place by 0 % — the uniform case. "
        "Expect a different answer on a well whose dropout curve is still "
        "climbing steeply over the window you are fitting.")
    _f = res.fmb
    if _f and str(_f.get("pseudo_pressure", "")).startswith("two-phase"):
        live(f"mobility loss {100 * _f['bank_mobility_loss_last']:.0f} % at "
             f"the last point, varying by "
             f"{100 * _f['bank_loss_spread']:.0f} points across the window, "
             f"which moved gas in place {100 * _f['bank_correction']:+.1f} %.")
    elif _f:
        live("the flowing material balance is running on the single-phase "
             "m(p). Switch on <b>Condensate bank</b> in the sidebar to see "
             "the correction on this well.")
    elif not use_fmb:
        live("the flowing material balance is switched off in the sidebar, so "
             "there is nothing here to correct.")
    elif "p_wf" not in res.data.df.columns:
        live("this record has no flowing-pressure column, so the flowing "
             "material balance cannot run at all.")
    else:
        live("the flowing material balance did not converge on this record, "
             "so there is nothing to correct.")

    st.markdown("### What actually ends the well")
    st.markdown(
        "A gas rate is rarely it. Water climbs while gas falls, and the "
        "handling limit arrives first. Produced water is forecast from a "
        "log-linear trend fitted to **its own history** — not derived from "
        "the gas decline, which knows nothing about it — and used only when "
        "that trend is statistically significant. The forecast then stops at "
        "the earliest of the gas rate, the water rate, the water cut and the "
        "gas in place, and reports which one bound **and when each of the "
        "others would have**.\n\n"
        "How much that matters is entirely a property of your well. On a dry, "
        "stable producer it changes nothing. On one where water is climbing "
        "faster than gas is falling, it is the difference between a life set "
        "by the reservoir and a life set by the facility — and the second is "
        "usually much shorter. The point of reporting every constraint, not "
        "just the binding one, is that you can see which of those you have "
        "without being told.")
    _cy = res.forecast.constraint_years or {}
    if len(_cy) > 1:
        _rows = sorted(_cy.items(), key=lambda kv: kv[1])
        live("ends on <b>" + _rows[0][0] + f"</b> at {_rows[0][1]:,.1f} yr; "
             + ", ".join(f"{k} would bind at {v:,.1f} yr"
                         for k, v in _rows[1:]) + ".")
    else:
        live("only the gas rate is constraining this forecast. Set a water "
             "or water-cut limit in the sidebar to test the others.")

    st.markdown("### Is the liquid really condensate?")
    st.markdown(
        "Below the dew point a retrograde gas gets **leaner**: the heavy ends "
        "drop out in the reservoir and stay there, so the produced "
        "condensate-gas ratio falls away from its initial value and cannot "
        "climb back past it. Above the dew point it sits *at* that value. "
        "Either way the initial CGR is a hard ceiling on the produced CGR, "
        "and the fluid report already on file is enough to test it.\n\n"
        "A produced ratio above that ceiling is not reservoir behaviour. It "
        "is usually water the separator never split out, a test that measured "
        "total liquid, or a drifting allocation factor — and all three "
        "inflate reserves without showing up as an error anywhere else. The "
        "app reports the excess as an implied non-condensate fraction, and "
        "separately regresses CGR against water cut: a yield that climbs in "
        "step with water is measuring the water, and extrapolating that line "
        "back to zero water cut gives the CGR the stream would have had "
        "without it.")

    st.markdown("### The bank, measured")
    st.markdown(
        "Productivity index, q / [m(p_avg) − m(p_wf)], divides out the "
        "drawdown and the gas properties, so what is left is mobility and "
        "contacted volume. A fall that begins where the reservoir crosses the "
        "dew point is the condensate bank, in the units that matter to a "
        "forecast — lost deliverability.\n\n"
        "It is measured on the post-plateau window only, and that is not a "
        "detail. On plateau the choke holds the rate constant while the "
        "reservoir depletes, so the drawdown needed to deliver it shrinks and "
        "the index **rises** — by enough, on a well with a long plateau, to "
        f"cancel the later fall completely. {VERIFY[0].upper()}{VERIFY[1:]}, "
        "the whole record reports +4 %/yr and the post-plateau window "
        "−33 %/yr, on the same data.")
    _b = res.bank
    if _b is not None and _b.ok:
        live(f"PI has fallen <b>{100 * _b.loss_frac:,.0f} %</b> "
             f"({_b.trend_pct_per_year:+,.0f} %/yr, R² {_b.r2:.2f}).")
    elif _b is not None:
        live(f"not available — {_b.reason}.")

    st.markdown("## The workflow, in order")
    st.markdown(
        "1. **QC and conversion** — units checked, uptime applied, outliers "
        "flagged, rates converted to wellstream. Cumulatives are accumulated "
        "over *every* producing month before any row is filtered, so QC never "
        "changes a volume.\n"
        "2. **Window detection** — the facility plateau and any transient flow "
        "are found and excluded, because neither obeys Arps.\n"
        "3. **Decline fit** — Arps, modified hyperbolic, Duong, power-law "
        "exponential and stretched exponential are all fitted and ranked by "
        "AICc. Parameters sitting at a bound are reported as such.\n"
        "4. **Yield model** — CGR against cumulative gas, fitted to the "
        "history, used to produce condensate from the gas forecast.\n"
        "5. **Material balance** — two-phase p/z, Havlena-Odeh, the We ≥ 0 "
        "ceiling, the apparent-G sequence, and a Fetkovich aquifer fit when "
        "the drive is not volumetric.\n"
        "6. **Forecast** — rolled forward to the economic limit and capped at "
        "the chosen gas in place.\n"
        "7. **Uncertainty** — Monte Carlo over the fit parameters, the "
        "terminal decline and the cap.")

    st.markdown("### Which gas-in-place number to believe")
    st.markdown(
        "Three are reported and they mean different things. The **p/z "
        "intercept** is gas in place only if the tank is closed; under "
        "pressure support the influx holds p/z up, flattens the trend and "
        "inflates the intercept. The **We ≥ 0 ceiling**, `min(F/Eg)`, is not "
        "an estimate at all but a hard upper bound — influx can only add to "
        "the withdrawal, so gas in place cannot exceed it however straight the "
        "plot looks. The **Fetkovich G** models the influx explicitly instead "
        "of absorbing it into the intercept, and is the right answer when the "
        "drive is not volumetric — but it is non-unique, so a range is quoted "
        "alongside it.\n\n"
        "The **Which gas in place** setting in the sidebar picks between them; "
        "on *Auto* the choice follows the drive diagnosis, and whatever is "
        "chosen is clipped to the ceiling.")

    st.markdown("## What it refuses to do")
    st.markdown(
        "Most of the work in this tool is in the refusals. A fitting routine "
        "always returns something, and a plausible-looking number from a "
        "calculation that quietly failed is worse than no number at all.")
    st.markdown(
        "- **Ambiguous dates.** `01/05/2012` is 1 May or 5 January depending "
        "on where the file came from. Read the wrong way round, a monthly "
        "history becomes a fortnight and every rate is out by a factor of "
        "thirty, with nothing looking broken. Dates must be ISO `YYYY-MM-DD`.\n"
        "- **An initial pressure that was never measured.** The first survey "
        "is typically months into the life. Taking it as p_i silently "
        "redefines G as the gas in place on the day of the survey. The tool "
        "offers to back-extrapolate instead, and declines to guess from a "
        "single survey.\n"
        "- **A p/z intercept from a line that is nearly flat.** A big number "
        "over a slope near zero is a finite intercept and a meaningless one. "
        "It is excluded from the cap on its own standard error.\n"
        "- **An aquifer fit that did not converge.** Checked on rms, on influx "
        "as a fraction of pore volume, and on whether the best-fit G falls "
        "inside its own locus.\n"
        "- **A Carter-Tracy solution that pushes the tank above its initial "
        "pressure.** The recursion assumes a constant influx rate over each "
        "step and will oscillate if driven too hard; an aquifer whose own "
        "pressure never exceeds p_i cannot drive the reservoir above it "
        "either, so those parameter sets are rejected rather than fitted.\n"
        "- **A liquid stream the fluid cannot produce.** Checked against the "
        "initial CGR, which is a ceiling below the dew point and an equality "
        "above it.\n"
        "- **A water trend that is only scatter with a slope.** It is fitted, "
        "reported with its R² and p-value, and used for the forecast only "
        "when it is significant.\n"
        "- **A CVD table that does not fit the data** — quoted in percent, "
        "not starting at the dew point, or not spanning the survey pressures. "
        "Any of these can flatten p/z to a constant and empty the material "
        "balance without raising anything.")

    st.markdown("## Limitations, honestly")
    st.markdown(
        "Each of these is written so you can tell whether it applies to the "
        "well in front of you, rather than having to take it on trust. The "
        "*when it bites* line is the observable condition on your own data.")
    st.markdown(
        "**The material balance is a single tank.** Layered or "
        "compartmentalised reservoirs are not represented, and commingled "
        "production will not behave.\n"
        "> *When it bites:* p/z shows two straight segments with a kink, or "
        "the apparent-G sequence steps rather than drifts. Both are visible "
        "on the Material balance tab. Compartments can be **detected** that "
        "way but not **allocated** without production logs.\n\n"
        "**Aquifer fits are non-unique**, whichever model is used. A large "
        "aquifer with a small conductivity and a small one with a large "
        "conductivity bend the same pressure history; only late depletion of "
        "the aquifer itself separates them, and a finite record rarely "
        "reaches it.\n"
        "> *When it bites:* always, to some degree — which is why a locus is "
        "quoted rather than a triplet. Read the width of that range on your "
        "well; if it spans a factor of two, it spans a factor of two.\n\n"
        "**The condensate bank is measured and corrected, but not modelled in "
        "the decline.** Its effect there is absorbed into the fitted "
        "parameters — adequate for forecasting, useless for diagnosing a "
        "completion change. The near-wellbore saturation is scaled from a "
        "cell average rather than computed from the flowing CGR, which would "
        "need a black-oil table.\n"
        "> *When it bites:* a reservoir well below its dew point with a "
        "steeply climbing dropout curve over the fitted window. Check the "
        "mobility-loss spread on the Material balance tab.\n\n"
        "**Rising CGR cannot be represented by the yield model**, and the fit "
        "will say so rather than bend to it. That is deliberate: the checks "
        "above exist to identify *why* it is rising, and a curve that flexed "
        "to accommodate contaminated liquid would bury the finding instead of "
        "surfacing it.\n"
        "> *When it bites:* the Yield tab reports a negative R² or a rate "
        "constant at its bound. Treat that as a result, not a failure.\n\n"
        "**The decline models assume the well is operated the same way "
        "throughout** — boundary-dominated flow at roughly constant flowing "
        "pressure. Long shut-ins, choke changes and compression all violate "
        "it, and the fit absorbs them into b and D.\n"
        "> *When it bites:* a low R² on the log-rate fit, a parameter sitting "
        "at a bound, or a QC report that has dropped a large share of the "
        "record. All three are on the Data & QC and Decline fit tabs.\n\n"
        "**The Monte Carlo samples the fitted parameters and the cap, not the "
        "choice of model.** Two models can fit the same history equally well "
        "and forecast very differently, and that spread does not appear in "
        "P90/P10.\n"
        "> *When it bites:* when the model ranking on the Decline fit tab "
        "shows several models within a point or two of AICc. Compare their "
        "EURs directly instead of reading the band.\n\n"
        "**Decline curve analysis cannot see a change that has not happened "
        "yet.** A compressor or a workover will break the forecast and "
        "neither is in the data. Water handling reaching its limit no longer "
        "belongs on that list — it is forecast and applied — but the water "
        "trend is a log-linear extrapolation, which is crude, and is shown "
        "with its R² and p-value so it can be disbelieved.\n"
        "> *When it bites:* whenever something is planned. No diagnostic will "
        "tell you; you have to.")

    st.markdown("## References")
    st.markdown(
        "- Arps, J.J. (1945) *Analysis of decline curves.* Trans. AIME 160.\n"
        "- Fetkovich, M.J. (1971) *A simplified approach to water influx "
        "calculations — finite aquifer systems.* JPT 23(7).\n"
        "- Havlena, D. and Odeh, A.S. (1963) *The material balance as an "
        "equation of a straight line.* JPT 15(8).\n"
        "- Fevang, Ø. and Whitson, C.H. (1996) *Modeling gas condensate well "
        "deliverability.* SPE Reservoir Engineering 11(4).\n"
        "- Carter, R.D. and Tracy, G.W. (1960) *An improved method for "
        "calculating water influx.* Trans. AIME 219.\n"
        "- van Everdingen, A.F. and Hurst, W. (1949) *The application of the "
        "Laplace transformation to flow problems in reservoirs.* Trans. AIME "
        "186.\n"
        "- Edwardson, M.J. et al. (1962) *Calculation of formation "
        "temperature disturbances caused by mud circulation.* JPT 14(4) "
        "— source of the p_D approximations.\n"
        "- Rayes, D.G. et al. (1992) *Two-phase compressibility factors for "
        "retrograde gases.* SPE Formation Evaluation 7(1).\n"
        "- Dranchuk, P.M. and Abou-Kassem, J.H. (1975) *Calculation of z "
        "factors for natural gases using equations of state.* JCPT 14(3).\n"
        "- Ilk, D. et al. (2008) *Exponential vs. hyperbolic decline in tight "
        "gas sands.* SPE 116731.\n"
        "- Valkó, P.P. (2009) *Assigning value to stimulation in the Barnett "
        "Shale* (stretched exponential). SPE 119369.")

    st.divider()
    st.caption(f"Version {dca.__version__} · {len(dca.DECLINE_MODELS)} decline "
               "models · self-tests run with "
               "`python gas_condensate_dca.py --selftest`")
