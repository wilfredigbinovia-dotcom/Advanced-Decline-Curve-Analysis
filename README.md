# Gas Condensate Decline Curve Analysis

`gas_condensate_dca.py` — a single-file Python engine for defensible DCA on
retrograde gas condensate wells and fields, plus a Streamlit front end.

```bash
pip install -r requirements.txt
python gas_condensate_dca.py --selftest   # 20 internal consistency checks
python gas_condensate_dca.py --demo       # worked example on synthetic data
streamlit run app.py                      # the web app
```

## Deploying the Streamlit app

Everything the app needs is in `streamlit_app/`. Copy its contents to the root
of your repository:

```
your-repo/
├── app.py                  # Streamlit front end
├── dca_charts.py           # Plotly charts
├── gas_condensate_dca.py   # the analysis engine
├── requirements.txt        # MUST be at the repo root
└── .streamlit/
    └── config.toml         # theme
```

Point Streamlit Cloud at `app.py`. If a deploy fails with `ModuleNotFoundError`,
it is almost always because `requirements.txt` is missing, is in a subfolder, or
does not list the package that failed — Streamlit Cloud installs nothing beyond
its base image on its own. After pushing a change to it, use **Manage app →
Reboot app**; a cached environment sometimes survives a plain commit.

The app opens on a synthetic field, so it works before you upload anything. On
that example data it also shows the *true* gas in place from the simulator
beside what the material balance recovered — which is the quickest way to see
what the two-phase z-factor is buying you.

---

## Why this isn't just Arps

Applying dry-gas Arps to a condensate well goes wrong in three specific ways.
The module is built around fixing each one.

**1. The thing that declines is the wellstream, not the separator products.**
Above the dew point the reservoir produces a single phase. Separator gas and
separator condensate are *products* of that stream, not independent quantities.
Fitting them separately gives two inconsistent EURs and a nonsensical implied
yield. Everything here is converted to a gas-equivalent wellstream basis
(`V_eq = 133,000 γ_o / M_o` scf/STB) before anything is fitted, then re-split
into products afterwards.

**2. Yield is not constant.** Below the dew point, liquid drops out in the
reservoir and produced CGR falls. The module fits a CGR-versus-cumulative model
with a dew-point break and a floor, then applies it to the gas forecast — so
gas and liquid can never drift apart.

**3. Single-phase z corrupts the material balance.** Once below the dew point,
part of the hydrocarbon has condensed, so the two-phase z (which accounts for
*all* remaining moles) is lower than the single-phase gas z. Points plotted with
the single-phase value fall below the true line, the trend looks too steep, and
the extrapolated OGIP comes out **too small** — around 9–13% on the demo fluid.
The module supports two-phase z from a CVD table or the Rayes correlation, and
reports the size of the error for your own fluid rather than assuming a rule of
thumb. (Note the direction: this understates OGIP. Some references state the
opposite; the code measures it so you don't have to rely on memory.)

---

## Quick start

```python
from gas_condensate_dca import *

cvd = CVDTable(
    pressure             = [5100, 4500, 3800, 3100, 2400, 1800, 1200, 700],
    cum_produced_molfrac = [0.000, 0.078, 0.176, 0.283, 0.402, 0.515, 0.641, 0.757],
)

pvt = PVT(gas_gravity=0.72, temperature_F=248, condensate_api=52.0,
          p_dew=5100, p_init=6400, y_n2=0.012, y_co2=0.031,
          cvd=cvd, initial_cgr=78.0)

wells = load_production("field_history.csv", pvt)      # {well: ProductionData}

results, summary = analyse_field(
    wells, pvt,
    q_econ_mscfd            = 400,
    terminal_decline_pct_yr = 7.0,
    products                = ProductSplit(inert_fraction=0.043,
                                           fuel_flare_fraction=0.02,
                                           ngl_yield_gal_per_mscf=1.8,
                                           ngl_shrinkage_fraction=0.03),
    run_monte_carlo         = True,
)

results["GC-1"].summary()
plot_diagnostics(results["GC-1"], path="GC-1.png")
results["GC-1"].to_excel("GC-1.xlsx")
print(field_profile(results).head())
```

Or from the shell:

```bash
python gas_condensate_dca.py --input field_history.csv \
  --gas-gravity 0.72 --temperature-f 248 --condensate-api 52 \
  --p-dew 5100 --p-init 6400 --initial-cgr 78 \
  --y-n2 0.012 --y-co2 0.031 --q-econ 400 --terminal-decline 7
```

---

## Input data

Column names are matched case- and punctuation-insensitively against a list of
aliases, so `Gas Rate (Mscf/d)`, `gas_mscfd` and `qg` all map to `q_gas`.
Pass `column_map={"q_gas": "MY_WEIRD_COLUMN"}` for anything unusual.

| Column    | Units    | Required | Notes |
|-----------|----------|----------|-------|
| `date`    | —        | yes      | any parseable date |
| `q_gas`   | Mscf/d   | yes      | separator gas |
| `well`    | —        | no       | one entity assumed if absent |
| `q_cond`  | STB/d    | no       | absent ⇒ treated as dry gas |
| `days_on` | days     | no       | strongly recommended |
| `p_wf`    | psia     | no       | needed for FMB |
| `p_res`   | psia     | no       | needed for material balance |
| `q_water` | STB/d    | no       | carried through for QC |

`rate_basis` is explicit (`"stream-day"` or `"calendar-day"`) because silently
mixing the two across a history is the single most common source of spurious
hyperbolic curvature.

---

## What the workflow does, in order

1. **QC and wellstream conversion.** Screens non-physical values, drops
   low-uptime periods and rate outliers (rolling-median MAD), and converts to a
   gas-equivalent wellstream. Cumulatives are accumulated over *every* producing
   month before any row is excluded — a month dropped from the fit still put gas
   in the pipe, and removing its volume corrupts the material balance.
2. **Flow-regime diagnosis.** Detects where the plateau ends (rate set by
   facilities, not the reservoir) and where boundary-dominated flow begins, and
   fits only the freely-declining part. Never discards more than 40% of the
   history to BDF detection.
3. **Decline fitting.** Arps, modified hyperbolic, Duong, PLE and SEPD, each
   with bounded multi-start `least_squares` on log-rate residuals under a robust
   loss. Ranked by AIC. Parameters are referenced to the *start of the fit
   window*, not first production — otherwise `qi` and `Di` become almost
   perfectly correlated and the fit slides onto its bounds. Parameters pinned to
   a bound are flagged.
4. **Yield model.** `CGR(Gp) = cgr_min + (cgr_i − cgr_min)·exp(−k·(Gp − Gp_dew))`.
   The floor matters: a pure exponential decays to zero, which no real
   condensate reservoir does, and it will understate late-life liquid.
5. **Material balance.** Two-phase p/z regression for OGIP with a curvature
   diagnostic (water influx vs. volumetric), plus optional flowing material
   balance. The OGIP becomes a cap on the forecast — the cheapest way to stop a
   high-b hyperbolic producing more gas than the reservoir holds.
6. **Forecast and product split.** Economic-limit solve, then sales gas, plant
   NGL and condensate. EURs include history.
7. **Monte Carlo.** Samples the fit covariance plus explicit priors on terminal
   decline, b, yield and the OGIP cap. Reports P90/P50/P10 in the petroleum
   convention (P90 = low case).

---

## Validation

`--selftest` runs 20 checks: DAK z against its own implicit equation, every
analytic cumulative against a fine numerical integral, continuity of the
modified hyperbolic at its switch point, parameter and EUR recovery from clean
and noisy data, and OGIP recovery.

The synthetic data generator is a single-tank reservoir model — pseudo-steady
deliverability `q = J·bank(p)·[m(p̄) − m(p_wf)]`, two-phase material balance, a
condensate-bank mobility multiplier below the dew point, and a facility plateau
— with noise and downtime applied *inside* the loop so rates, cumulatives and
reported pressures stay mutually consistent. That makes the checks real tests
rather than circular ones. On a four-well synthetic field the two-phase material
balance recovers the true OGIP to within ±1%, and FMB to within about +6%.

---

## Things worth knowing before you trust a number

- **Condensate banking is a deliverability effect, not a reserves effect.** It
  steepens the apparent decline without destroying gas in place. Don't let it
  inflate your decline exponent; the module's OGIP cap is the guardrail.
- **FMB reads low on a condensate well.** Below the dew point the bank degrades
  mobility faster than depletion alone, so the normalised-rate line is too steep
  and its x-intercept lands below true OGIP. Treat it as a lower bound; the
  result carries a `bias_warning` flag.
- **b at its lower bound just means exponential.** That's a legitimate answer
  for a depleting gas well, and the summary says so rather than crying failure.
- **b > 1 in a conventional setting is almost always an artefact** — transient
  flow, changing flowing pressure, added wells, or mixed rate bases.
- **The CGR model is empirical in cumulative space.** Its asymptote is a fitting
  parameter, not the CVD liquid-dropout floor. Use CVD to check the *shape*.
- **Pseudo-pressure uses the wellstream gravity** when `initial_cgr` is given,
  which is the physically correct choice above the dew point.
- This is a screening and reserves tool. It does not replace compositional
  simulation for a field where the bank, deliverability and yield all matter to
  a development decision.

---

## API map

| Area | Entry points |
|---|---|
| Fluids | `PVT`, `CVDTable`, `z_factor`, `gas_viscosity`, `rayes_two_phase_z` |
| Data | `load_production`, `ProductionData.prepare`, `QCReport` |
| Diagnostics | `diagnose_b`, `detect_decline_start`, `detect_bdf_start` |
| Models | `Arps`, `ModifiedHyperbolic`, `Duong`, `PowerLawExponential`, `StretchedExponential` |
| Fitting | `fit_decline`, `rank_models`, `FitResult` |
| Yield | `YieldModel`, `fit_yield_model` |
| Balance | `material_balance_pz`, `flowing_material_balance` |
| Forecast | `ProductSplit`, `forecast_products`, `monte_carlo_eur`, `percentiles_petroleum` |
| Orchestration | `analyse_well`, `analyse_field`, `field_profile`, `WellResult` |
| Output | `plot_diagnostics`, `WellResult.to_excel` |
| Synthetic | `simulate_tank`, `make_synthetic_well`, `make_synthetic_field` |

MIT licensed. Oilfield units throughout (psia, °F, Mscf/d, STB/d, MMscf,
STB/MMscf); time is days internally, years at the reporting boundary.
