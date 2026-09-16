"""
================================================================================
 gas_condensate_dca.py
 Decline Curve Analysis (DCA) toolkit for gas condensate fields
================================================================================

A single-file, dependency-light engine for performing defensible decline curve
analysis on retrograde gas condensate wells and fields.

Why gas condensate needs its own toolkit
----------------------------------------
Applying dry-gas Arps directly to a condensate well is wrong in three ways:

1. The reservoir produces a single-phase wellstream above the dew point. The
   separator gas and separator condensate are *products*, not the thing that
   declines. This module converts everything to a **wellstream (gas-equivalent)**
   basis before fitting, then re-splits the forecast into products.

2. Condensate yield (CGR) is not constant. Below the dew point, liquid drops out
   in the reservoir and the produced yield falls, tracking the CVD liquid-dropout
   curve. This module fits a **CGR-vs-cumulative yield model** and applies it to
   the gas forecast, so gas and liquid forecasts stay internally consistent.

3. Material balance with a single-phase z-factor over-states OGIP once below the
   dew point. This module supports a **two-phase z-factor** (from a CVD table, or
   the Rayes et al. correlation) for the p/z check.

What is included
----------------
  * PVT engine
      - Sutton pseudo-criticals with inert (N2/CO2/H2S) mixing
      - Wichert-Aziz sour-gas correction
      - Dranchuk-Abou-Kassem z-factor (bounded root solve, vectorised)
      - Lee-Gonzalez-Eakin gas viscosity
      - Real-gas pseudo-pressure m(p) and material-balance pseudo-time
      - Gas equivalent of condensate, wellstream gravity
      - Two-phase z-factor: CVD table interpolation or Rayes correlation
  * Data handling
      - Tolerant loader (CSV/Excel) with column mapping and unit conversion
      - QC: uptime normalisation, non-physical value screening, outlier
        rejection (rolling-median MAD), flow-regime / BDF start detection
  * Decline models (all with rate, cumulative, D(t), b(t), inverse)
      - Arps (exponential / hyperbolic / harmonic)
      - Modified hyperbolic (hyperbolic -> terminal exponential at D_min)
      - Duong
      - Power-Law Exponential (Ilk et al.)
      - Stretched Exponential (SEPD)
  * Robust fitting
      - Bounded, multi-start `scipy.optimize.least_squares`
      - Log-space residuals with robust (soft-L1 / Huber) loss
      - Parameter covariance, standard errors, AIC/BIC, model ranking
  * Yield model
      - CGR(Gp) with dew-point break and exponential decay to a floor
  * Material balance
      - p/z and two-phase p/z* regression for OGIP, with drive diagnostics
      - Flowing material balance (iterative, pseudo-pressure based)
  * Forecasting
      - Economic-limit solve, EUR, product splitting (sales gas, NGL, condensate)
      - Monte Carlo P90/P50/P10 from the fit covariance plus user priors
  * Outputs
      - Diagnostic plot suite, tidy result objects, CSV/Excel export
  * Synthetic data generator so the whole thing is runnable out of the box

Units
-----
Oilfield units throughout, unless a name says otherwise:
    pressure        psia
    temperature     degR internally (degF at the API boundary)
    gas rate        Mscf/d
    gas cumulative  MMscf
    liquid rate     STB/d
    liquid cum      Mstb
    CGR             STB/MMscf
    time            days internally; years at the reporting boundary
    decline rate    1/day (nominal) internally; %/yr at the reporting boundary

Quick start
-----------
    python gas_condensate_dca.py --demo

    from gas_condensate_dca import *
    df   = load_production("well_A.csv")
    pvt  = PVT(gas_gravity=0.72, temperature_F=248, condensate_api=52.0,
               y_n2=0.01, y_co2=0.03)
    res  = analyse_well(df, pvt, well="A-1")
    res.summary()

Author: built for a gas condensate DCA workflow.
License: MIT.
================================================================================
"""

from __future__ import annotations

import argparse
import math
import re
import os
import sys
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy import optimize, special, stats
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import interp1d

# Version history is tracked so a screenshot or an exported workbook can be
# traced back to the code that produced it.
#   v1  baseline: DCA, yield model, two-phase material balance, Havlena-Odeh,
#       We>=0 ceiling, Fetkovich aquifer, initial-pressure estimator, OGIP cap
#       selection.
#   v2  dates fixed to ISO YYYY-MM-DD rather than guessed; the estimated-p_i
#       row dated to the start of a month; a non-declining p/z diagnosed
#       instead of aborting the material balance.
__version__ = "2.0"

__all__ = [
    "__version__", "PVT", "CVDTable",
    "ProductionData", "load_production", "make_synthetic_field",
    "Arps", "ModifiedHyperbolic", "Duong", "PowerLawExponential",
    "StretchedExponential", "DECLINE_MODELS",
    "FitResult", "fit_decline", "rank_models",
    "diagnose_b", "detect_bdf_start",
    "YieldModel", "fit_yield_model",
    "material_balance_pz", "flowing_material_balance",
    "Forecast", "forecast_products", "monte_carlo_eur", "ProductSplit",
    "fetkovich_aquifer_fit", "havlena_odeh_gas", "gas_fvf_rb_per_scf",
    "OGIPChoice", "select_ogip",
    "InitialPressureEstimate", "estimate_initial_pressure",
    "estimate_initial_pressure_from_wells", "field_survey_table",
    "parse_dates", "DateFormatError", "percentiles_petroleum",
    "PZ_MAX_REL_SE",
    "WellResult", "analyse_well", "analyse_field", "field_profile",
    "detect_decline_start", "simulate_tank", "make_synthetic_well",
    "MaterialBalanceResult", "QCReport", "run_self_tests", "demo",
    "plot_diagnostics", "PALETTE",
]

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

R_GAS = 10.7316              # psia.ft3/(lbmol.degR)
AIR_MW = 28.9625             # lb/lbmol
DAYS_PER_YEAR = 365.25
SCF_PER_MSCF = 1.0e3
MSCF_PER_MMSCF = 1.0e3

# Critical properties of inerts (degR, psia)
INERT_PROPS = {
    "n2":  {"mw": 28.0134, "tc": 227.16, "pc": 492.84},
    "co2": {"mw": 44.0100, "tc": 547.58, "pc": 1071.00},
    "h2s": {"mw": 34.0800, "tc": 672.35, "pc": 1299.97},
}

# Dranchuk-Abou-Kassem (1975) coefficients
_DAK = (0.3265, -1.0700, -0.5339, 0.01569, -0.05165,
        0.5475, -0.7361, 0.1844, 0.1056, 0.6134, 0.7210)

# Rayes et al. (1992) two-phase z-factor correlation coefficients
_RAYES = (2.24353, -0.0375281, -3.56539, 0.000829231, 1.53428, 0.131987)

# dataviz reference palette (light mode), used in fixed slot order
PALETTE = {
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "critical": "#d03b3b",
    "good": "#0ca30c",
}


# ==============================================================================
# SECTION 1 -- PVT / FLUID PROPERTIES
# ==============================================================================

def standing_condensate_mw(api: float) -> float:
    """Standing's correlation for stock-tank condensate molecular weight.

    M_o = 5954 / (API - 8.811)     [lb/lbmol]

    Valid roughly 40 <= API <= 70, the usual gas condensate range. Prefer a
    measured C7+ / stock-tank molecular weight when one exists.
    """
    api = float(api)
    if api <= 9.0:
        raise ValueError("Condensate API must exceed ~9 for Standing's correlation.")
    return 5954.0 / (api - 8.811)


def api_to_sg(api: float) -> float:
    """Stock-tank liquid specific gravity from API gravity."""
    return 141.5 / (float(api) + 131.5)


def gas_equivalent_of_condensate(condensate_sg: float,
                                 condensate_mw: float) -> float:
    """Gas equivalent volume of one stock-tank barrel of condensate.

        V_eq = 133,000 * gamma_o / M_o     [scf / STB]

    This is the volume the condensate would occupy as gas at standard
    conditions, and is what lets separator liquid be folded back into a
    single-phase reservoir wellstream.
    """
    if condensate_mw <= 0:
        raise ValueError("Condensate molecular weight must be positive.")
    return 133000.0 * float(condensate_sg) / float(condensate_mw)


def wellstream_gravity(gas_gravity: float, gor_scf_stb: float,
                       condensate_sg: float, condensate_mw: float) -> float:
    """Specific gravity of the reservoir wellstream (wet gas).

        gamma_w = (R * gamma_g + 4584 * gamma_o) / (R + V_eq)

    with R the producing gas-oil ratio in scf/STB and V_eq the gas equivalent.
    For a very high GOR (lean gas) this collapses to gamma_g, as it should.
    """
    veq = gas_equivalent_of_condensate(condensate_sg, condensate_mw)
    R = float(gor_scf_stb)
    if not np.isfinite(R) or R <= 0:
        return float(gas_gravity)
    return (R * gas_gravity + 4584.0 * condensate_sg) / (R + veq)


def sutton_pseudocriticals(gas_gravity: float,
                           y_n2: float = 0.0, y_co2: float = 0.0,
                           y_h2s: float = 0.0) -> Tuple[float, float]:
    """Pseudo-critical T (degR) and p (psia) via Sutton, with inert handling.

    Inerts are stripped out, Sutton's wet-gas correlation is applied to the
    hydrocarbon fraction alone, the inerts are mixed back by Kay's rule, and
    the Wichert-Aziz correction is applied for CO2/H2S. This is the standard
    treatment and matters for sour or nitrogen-rich condensate gas.
    """
    y_n2, y_co2, y_h2s = float(y_n2), float(y_co2), float(y_h2s)
    y_inert = y_n2 + y_co2 + y_h2s
    if y_inert < 0 or y_inert >= 1.0:
        raise ValueError("Inert mole fractions must sum to a value in [0, 1).")

    gg = float(gas_gravity)

    if y_inert > 1e-12:
        inert_mw = (y_n2 * INERT_PROPS["n2"]["mw"]
                    + y_co2 * INERT_PROPS["co2"]["mw"]
                    + y_h2s * INERT_PROPS["h2s"]["mw"])
        gg_hc = (gg - inert_mw / AIR_MW) / (1.0 - y_inert)
        # Guard against an inconsistent gravity/composition pair.
        gg_hc = float(np.clip(gg_hc, 0.55, 1.6))
    else:
        gg_hc = gg

    # Sutton (1985) wet-gas pseudo-criticals for the hydrocarbon fraction
    tpc_hc = 169.2 + 349.5 * gg_hc - 74.0 * gg_hc ** 2
    ppc_hc = 756.8 - 131.0 * gg_hc - 3.6 * gg_hc ** 2

    tpc = (1.0 - y_inert) * tpc_hc
    ppc = (1.0 - y_inert) * ppc_hc
    for key, y in (("n2", y_n2), ("co2", y_co2), ("h2s", y_h2s)):
        tpc += y * INERT_PROPS[key]["tc"]
        ppc += y * INERT_PROPS[key]["pc"]

    # Wichert-Aziz sour correction
    A = y_co2 + y_h2s
    B = y_h2s
    if A > 1e-12:
        eps = 120.0 * (A ** 0.9 - A ** 1.6) + 15.0 * (B ** 0.5 - B ** 4)
        tpc_c = tpc - eps
        ppc_c = ppc * tpc_c / (tpc + B * (1.0 - B) * eps)
        tpc, ppc = tpc_c, ppc_c

    return float(tpc), float(ppc)


def _dak_z_from_rhor(rho_r: float, tpr: float) -> float:
    """DAK z-factor evaluated at a given reduced density."""
    a = _DAK
    t = tpr
    z = (1.0
         + (a[0] + a[1] / t + a[2] / t ** 3 + a[3] / t ** 4 + a[4] / t ** 5) * rho_r
         + (a[5] + a[6] / t + a[7] / t ** 2) * rho_r ** 2
         - a[8] * (a[6] / t + a[7] / t ** 2) * rho_r ** 5
         + a[9] * (1.0 + a[10] * rho_r ** 2) * (rho_r ** 2 / t ** 3)
         * math.exp(-a[10] * rho_r ** 2))
    return z


def _z_scalar(ppr: float, tpr: float) -> float:
    """Solve DAK for z at one (ppr, tpr) by bracketing on reduced density."""
    if ppr <= 1e-9:
        return 1.0
    tpr = float(np.clip(tpr, 1.0, 3.2))

    def f(rho_r: float) -> float:
        return 0.27 * ppr / (tpr * _dak_z_from_rhor(rho_r, tpr)) - rho_r

    lo, hi = 1e-12, 3.0
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        # Expand once; DAK is well behaved so this is a safety net only.
        hi = 5.0
        fhi = f(hi)
        if flo * fhi > 0:
            return float("nan")
    rho_r = optimize.brentq(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200)
    return _dak_z_from_rhor(rho_r, tpr)


_z_vec = np.vectorize(_z_scalar, otypes=[float])


def z_factor(p: np.ndarray, T_R: float, gas_gravity: float,
             y_n2: float = 0.0, y_co2: float = 0.0,
             y_h2s: float = 0.0) -> np.ndarray:
    """Real-gas compressibility factor z(p) by Dranchuk-Abou-Kassem."""
    tpc, ppc = sutton_pseudocriticals(gas_gravity, y_n2, y_co2, y_h2s)
    p = np.asarray(p, dtype=float)
    return _z_vec(p / ppc, T_R / tpc)


def gas_viscosity(p: np.ndarray, T_R: float, gas_gravity: float,
                  z: Optional[np.ndarray] = None,
                  **inerts) -> np.ndarray:
    """Gas viscosity (cp) by Lee-Gonzalez-Eakin."""
    p = np.asarray(p, dtype=float)
    if z is None:
        z = z_factor(p, T_R, gas_gravity, **inerts)
    z = np.asarray(z, dtype=float)

    M = AIR_MW * float(gas_gravity)
    T = float(T_R)
    K = (9.379 + 0.01607 * M) * T ** 1.5 / (209.2 + 19.26 * M + T)
    X = 3.448 + 986.4 / T + 0.01009 * M
    Y = 2.447 - 0.2224 * X
    rho = 1.4935e-3 * p * M / (z * T)          # g/cm3
    return 1.0e-4 * K * np.exp(X * rho ** Y)


def gas_compressibility(p: np.ndarray, T_R: float, gas_gravity: float,
                        **inerts) -> np.ndarray:
    """Isothermal gas compressibility cg = 1/p - (1/z)(dz/dp), 1/psi."""
    p = np.asarray(p, dtype=float)
    dp = np.maximum(1.0, 0.001 * np.maximum(p, 1.0))
    z0 = z_factor(p, T_R, gas_gravity, **inerts)
    z1 = z_factor(p + dp, T_R, gas_gravity, **inerts)
    dzdp = (z1 - z0) / dp
    with np.errstate(divide="ignore", invalid="ignore"):
        cg = 1.0 / np.maximum(p, 1e-9) - dzdp / z0
    return cg


@dataclass
class CVDTable:
    """Constant-volume-depletion laboratory data for a retrograde gas.

    Parameters
    ----------
    pressure : psia, descending from the dew point
    cum_produced_molfrac : cumulative wellstream moles produced, as a fraction
        of the original moles in the cell (0 at dew point, -> 1 at abandonment)
    z_gas : single-phase (equilibrium gas) z-factor, optional
    liquid_dropout : retrograde liquid volume as a fraction of dew-point volume,
        optional but used to sanity-check the CGR yield model
    """
    pressure: np.ndarray
    cum_produced_molfrac: np.ndarray
    z_gas: Optional[np.ndarray] = None
    liquid_dropout: Optional[np.ndarray] = None

    def __post_init__(self):
        self.pressure = np.asarray(self.pressure, dtype=float)
        self.cum_produced_molfrac = np.asarray(self.cum_produced_molfrac, dtype=float)
        if self.pressure.size != self.cum_produced_molfrac.size:
            raise ValueError("CVD pressure and cum_produced_molfrac must be same length.")
        order = np.argsort(-self.pressure)      # descending pressure
        self.pressure = self.pressure[order]
        self.cum_produced_molfrac = self.cum_produced_molfrac[order]
        if self.z_gas is not None:
            self.z_gas = np.asarray(self.z_gas, dtype=float)[order]
        if self.liquid_dropout is not None:
            self.liquid_dropout = np.asarray(self.liquid_dropout, dtype=float)[order]

    def two_phase_z(self, p: np.ndarray, z_dew: float, p_dew: float) -> np.ndarray:
        """Two-phase z from CVD, defined so that p/z_2p is linear in moles produced.

            (p / z_2p) = (p_dew / z_dew) * (1 - n_p)

        where n_p is the cumulative wellstream mole fraction produced. This is
        the definition that makes the p/z plot usable below the dew point.
        """
        p = np.asarray(p, dtype=float)
        np_interp = np.interp(p, self.pressure[::-1],
                              self.cum_produced_molfrac[::-1])
        np_interp = np.clip(np_interp, 0.0, 0.999)
        denom = (p_dew / z_dew) * (1.0 - np_interp)
        return np.where(denom > 0, p / denom, np.nan)


def rayes_two_phase_z(p: np.ndarray, T_R: float, gas_gravity: float,
                      **inerts) -> np.ndarray:
    """Rayes et al. (1992) two-phase z-factor correlation.

    A fallback for when no CVD table exists. Nominal validity is
    0.7 < p_pr < 20 and 1.1 < T_pr < 2.1; values outside are returned but
    flagged via a warning, because extrapolation here is not reliable.
    """
    tpc, ppc = sutton_pseudocriticals(gas_gravity, **inerts)
    ppr = np.asarray(p, dtype=float) / ppc
    tpr = float(T_R) / tpc
    if not (1.1 <= tpr <= 2.1):
        warnings.warn(f"Rayes two-phase z: T_pr={tpr:.2f} outside 1.1-2.1.")
    a0, a1, a2, a3, a4, a5 = _RAYES
    return (a0 + a1 * ppr + a2 / tpr + a3 * ppr ** 2
            + a4 / tpr ** 2 + a5 * ppr / tpr)


@dataclass
class PVT:
    """Fluid property container and pseudo-pressure engine for a condensate gas.

    Parameters
    ----------
    gas_gravity : separator / dry gas specific gravity (air = 1)
    temperature_F : reservoir temperature, degF
    condensate_api : stock-tank condensate API gravity
    condensate_mw : stock-tank condensate molecular weight; if omitted it is
        estimated from API by Standing's correlation
    p_dew : dew point pressure, psia (optional but needed for the yield break)
    p_init : initial reservoir pressure, psia
    y_n2, y_co2, y_h2s : inert mole fractions
    cvd : optional CVDTable for the two-phase z-factor
    use_wellstream_gravity : if True, pseudo-pressure is computed on the
        wellstream (wet) gravity rather than the separator gas gravity. This is
        the physically correct choice above the dew point.
    initial_cgr : initial condensate-gas ratio, STB/MMscf (used to derive
        wellstream gravity when use_wellstream_gravity is True)
    """
    gas_gravity: float = 0.70
    temperature_F: float = 220.0
    condensate_api: float = 52.0
    condensate_mw: Optional[float] = None
    p_dew: Optional[float] = None
    p_init: Optional[float] = None
    y_n2: float = 0.0
    y_co2: float = 0.0
    y_h2s: float = 0.0
    cvd: Optional[CVDTable] = None
    use_wellstream_gravity: bool = True
    initial_cgr: Optional[float] = None
    p_max_table: float = 15000.0

    # populated in __post_init__
    T_R: float = field(init=False)
    condensate_sg: float = field(init=False)
    v_eq: float = field(init=False)
    effective_gravity: float = field(init=False)
    tpc: float = field(init=False)
    ppc: float = field(init=False)

    def __post_init__(self):
        self.T_R = float(self.temperature_F) + 459.67
        if self.T_R <= 459.67:
            raise ValueError("Reservoir temperature must be above 0 degF.")
        self.condensate_sg = api_to_sg(self.condensate_api)
        if self.condensate_mw is None:
            self.condensate_mw = standing_condensate_mw(self.condensate_api)
        self.v_eq = gas_equivalent_of_condensate(self.condensate_sg,
                                                 self.condensate_mw)

        if self.use_wellstream_gravity and self.initial_cgr:
            gor = 1.0e6 / float(self.initial_cgr)       # scf gas per STB
            self.effective_gravity = wellstream_gravity(
                self.gas_gravity, gor, self.condensate_sg, self.condensate_mw)
        else:
            self.effective_gravity = float(self.gas_gravity)

        self.tpc, self.ppc = sutton_pseudocriticals(
            self.effective_gravity, self.y_n2, self.y_co2, self.y_h2s)

        self._build_tables()

    # -- property tables -------------------------------------------------
    @property
    def _inerts(self) -> Dict[str, float]:
        return {"y_n2": self.y_n2, "y_co2": self.y_co2, "y_h2s": self.y_h2s}

    def _build_tables(self, n: int = 600) -> None:
        """Pre-compute z, mu, cg and m(p) on a fine grid for fast interpolation."""
        p = np.linspace(14.7, float(self.p_max_table), n)
        z = z_factor(p, self.T_R, self.effective_gravity, **self._inerts)
        mu = gas_viscosity(p, self.T_R, self.effective_gravity, z=z,
                           **self._inerts)
        integrand = 2.0 * p / (mu * z)
        m = np.concatenate([[0.0], cumulative_trapezoid(integrand, p)])
        cg = gas_compressibility(p, self.T_R, self.effective_gravity,
                                 **self._inerts)

        self._p_grid = p
        self._z_grid = z
        self._mu_grid = mu
        self._m_grid = m
        self._cg_grid = cg
        self._f_z = interp1d(p, z, kind="cubic", bounds_error=False,
                            fill_value=(z[0], z[-1]))
        self._f_mu = interp1d(p, mu, kind="cubic", bounds_error=False,
                             fill_value=(mu[0], mu[-1]))
        self._f_m = interp1d(p, m, kind="cubic", bounds_error=False,
                            fill_value=(m[0], m[-1]))
        self._f_cg = interp1d(p, cg, kind="cubic", bounds_error=False,
                             fill_value=(cg[0], cg[-1]))
        self._f_muct = interp1d(p, mu * cg, kind="cubic", bounds_error=False,
                               fill_value=(mu[0] * cg[0], mu[-1] * cg[-1]))

        # Pre-sorted p/z -> p inverse. Built once here rather than rebuilt and
        # re-sorted on every call: the Fetkovich search inverts p/z tens of
        # thousands of times and that sort dominated the runtime.
        for attr, zz in (("_inv2", self.z_two_phase(p)), ("_inv1", z)):
            pzv = p / zz
            srt = np.argsort(pzv)
            setattr(self, attr, (np.ascontiguousarray(pzv[srt]),
                                 np.ascontiguousarray(p[srt])))

    # -- public property accessors --------------------------------------
    def z(self, p) -> np.ndarray:
        return np.asarray(self._f_z(np.asarray(p, dtype=float)), dtype=float)

    def mu(self, p) -> np.ndarray:
        return np.asarray(self._f_mu(np.asarray(p, dtype=float)), dtype=float)

    def cg(self, p) -> np.ndarray:
        return np.asarray(self._f_cg(np.asarray(p, dtype=float)), dtype=float)

    def m(self, p) -> np.ndarray:
        """Real-gas pseudo-pressure, psia^2/cp."""
        return np.asarray(self._f_m(np.asarray(p, dtype=float)), dtype=float)

    def mu_cg(self, p) -> np.ndarray:
        return np.asarray(self._f_muct(np.asarray(p, dtype=float)), dtype=float)

    def z_two_phase(self, p, method: str = "auto") -> np.ndarray:
        """Two-phase z-factor below the dew point.

        method = 'cvd'    use the supplied CVD table (preferred)
                 'rayes'  use the Rayes et al. correlation
                 'auto'   CVD if available, else Rayes
                 'single' force the single-phase z (for comparison only)
        """
        p = np.asarray(p, dtype=float)
        if method == "single":
            return self.z(p)
        if method in ("auto", "cvd") and self.cvd is not None and self.p_dew:
            z_dew = float(self.z(self.p_dew))
            z2 = self.cvd.two_phase_z(p, z_dew, float(self.p_dew))
            single = self.z(p)
            return np.where(p >= self.p_dew, single, z2)
        if method == "cvd":
            raise ValueError("No CVD table supplied for method='cvd'.")
        z2 = rayes_two_phase_z(p, self.T_R, self.effective_gravity, **self._inerts)
        single = self.z(p)
        if self.p_dew:
            # Anchor the correlation to the single-phase z AT the dew point.
            # By definition there is no liquid there, so the two values must
            # agree; Rayes is an independent regression and does not honour
            # that on its own. Left unanchored it puts a step in z at the dew
            # point - a couple of percent for a typical fluid - which makes p/z
            # discontinuous, breaks the inverse mapping used to turn an
            # intercept back into a pressure, and quietly corrupts every
            # apparent-G calculation that straddles the dew point.
            zd_single = float(self.z(np.array([self.p_dew]))[0])
            zd_rayes = float(rayes_two_phase_z(
                np.array([float(self.p_dew)]), self.T_R,
                self.effective_gravity, **self._inerts)[0])
            if np.isfinite(zd_rayes) and zd_rayes > 0:
                z2 = z2 * (zd_single / zd_rayes)
            return np.where(p >= self.p_dew, single, z2)
        return z2

    def pressure_from_pz(self, pz, two_phase: bool = True) -> np.ndarray:
        """Invert p/z -> p on the internal grid (the mapping is monotonic)."""
        xs, ys = self._inv2 if two_phase else self._inv1
        return np.interp(np.asarray(pz, dtype=float), xs, ys)

    def pseudo_time(self, t_days: np.ndarray, p_avg: np.ndarray,
                    p_ref: Optional[float] = None) -> np.ndarray:
        """Material-balance pseudo-time, t_a = (mu*cg)_i * int dt / (mu*cg)(p_avg)."""
        t_days = np.asarray(t_days, dtype=float)
        p_avg = np.asarray(p_avg, dtype=float)
        p_ref = float(p_ref if p_ref is not None else (self.p_init or p_avg[0]))
        muct_i = float(self.mu_cg(p_ref))
        integrand = muct_i / np.maximum(self.mu_cg(p_avg), 1e-12)
        ta = np.concatenate([[0.0], cumulative_trapezoid(integrand, t_days)])
        return ta

    def condensate_to_gas_equiv(self, q_cond_stb_d: np.ndarray) -> np.ndarray:
        """Convert condensate rate (STB/d) to gas equivalent rate (Mscf/d)."""
        return np.asarray(q_cond_stb_d, dtype=float) * self.v_eq / SCF_PER_MSCF

    def describe(self) -> pd.Series:
        return pd.Series({
            "gas_gravity": self.gas_gravity,
            "effective_gravity_used": self.effective_gravity,
            "temperature_F": self.temperature_F,
            "condensate_API": self.condensate_api,
            "condensate_MW": self.condensate_mw,
            "condensate_SG": self.condensate_sg,
            "gas_equivalent_scf_per_STB": self.v_eq,
            "p_dew_psia": self.p_dew,
            "p_init_psia": self.p_init,
            "Tpc_R": self.tpc,
            "Ppc_psia": self.ppc,
            "y_N2": self.y_n2, "y_CO2": self.y_co2, "y_H2S": self.y_h2s,
        })


# ==============================================================================
# SECTION 2 -- PRODUCTION DATA: LOADING, QC, WELLSTREAM CONVERSION
# ==============================================================================

# Canonical column names the rest of the module expects.
CANONICAL_COLUMNS = {
    "date": ["date", "prod_date", "production_date", "month", "day",
             "time_stamp", "period", "report_date"],
    "well": ["well", "well_name", "uwi", "api", "wellid", "well_id",
             "completion", "string"],
    "days_on": ["days_on", "days_online", "uptime_days", "producing_days",
                "onstream_days", "prod_days", "uptime", "on_stream_days",
                "days", "days_on_stream", "op_days", "days_produced"],
    "q_gas": ["q_gas", "gas_rate", "gas", "gas_mscfd", "qg", "gas_mscf_d",
              "gas_volume_rate", "sep_gas", "separator_gas", "gas_production",
              "q_g", "gas_prod", "dry_gas"],
    "q_cond": ["q_cond", "cond_rate", "condensate", "oil_rate", "qo",
               "cond_stbd", "oil", "condensate_rate", "liquid_rate",
               "condensate_production", "cond", "qc", "liquid", "cond_prod",
               "condensate_volume"],
    "q_water": ["q_water", "water_rate", "water", "qw", "water_stbd",
                "water_production", "wtr", "brine"],
    "p_wf": ["p_wf", "bhp", "fbhp", "bottomhole_pressure", "pwf",
             "flowing_bhp", "bottom_hole_pressure", "bhfp",
             "flowing_pressure"],
    "p_wh": ["p_wh", "thp", "fthp", "wellhead_pressure", "pwh",
             "tubing_pressure", "tubing_head_pressure"],
    "p_res": ["p_res", "reservoir_pressure", "p_avg", "static_pressure",
              "shut_in_pressure", "average_pressure", "sibhp", "pres",
              "p_bar", "static_bhp", "avg_reservoir_pressure"],
    "gas_cum": ["gas_cum", "cum_gas", "gp", "cumulative_gas"],
    "cond_cum": ["cond_cum", "cum_cond", "np", "cum_oil", "cumulative_condensate"],
}


def _normalise(name: str) -> str:
    """Reduce a column header to a comparable key.

    Real headers carry their units - "Gas Rate (Mscf/d)", "Condensate [STB/d]"
    - and separate words however the author felt like. Parenthesised or
    bracketed annotations are dropped, then everything that is not a letter or
    digit goes, so "Gas Rate (Mscf/d)", "gas_rate" and "GasRate" all collapse
    to the same key. Units are dropped rather than matched because this module
    fixes the units by contract; a header claiming different ones needs
    converting, not renaming.
    """
    s = str(name).lower().strip()
    s = re.sub(r"[\(\[\{][^\)\]\}]*[\)\]\}]", " ", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def map_columns(df: pd.DataFrame,
                overrides: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Rename a raw dataframe's columns onto the canonical schema.

    Matching is case- and punctuation-insensitive. Anything not recognised is
    left alone, so extra columns survive untouched.
    """
    overrides = overrides or {}
    lookup: Dict[str, str] = {}
    for canon, aliases in CANONICAL_COLUMNS.items():
        for alias in aliases + [canon]:
            lookup[_normalise(alias)] = canon

    rename: Dict[str, str] = {}
    for col in df.columns:
        if col in overrides:
            rename[col] = overrides[col]
            continue
        key = _normalise(col)
        if key in lookup:
            rename[col] = lookup[key]
    out = df.rename(columns=rename)
    # User overrides expressed as {canonical: raw_column}
    for canon, raw in overrides.items():
        if raw in df.columns and canon in CANONICAL_COLUMNS:
            out = out.rename(columns={raw: canon})
    return out


class DateFormatError(ValueError):
    """A date column that is not unambiguous. Carries advice, not just a name."""


ISO_DATE_RULE = (
    "Dates must be written ISO 8601: YYYY-MM-DD (2012-05-01). "
    "In Excel, select the column and use Format Cells -> Custom -> yyyy-mm-dd, "
    "or in pandas: df['date'] = pd.to_datetime(df['date'], dayfirst=True)"
    ".dt.strftime('%Y-%m-%d')  -- set dayfirst to match YOUR file."
)

_ISO_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}"
                     r"(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?\s*$")
_ISO_SLASH_RE = re.compile(r"^\s*\d{4}/\d{2}/\d{2}\s*$")
_SLASHED_RE = re.compile(r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})\s*$")
_MONTHNAME_RE = re.compile(
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.I)


def _diagnose_dates(values: Sequence[str]) -> str:
    """Say what the column looks like, so the fix is obvious rather than guessed."""
    sample = [str(v).strip() for v in values if str(v).strip()
              and str(v).strip().lower() not in ("nan", "nat", "none")][:200]
    if not sample:
        return "the column is empty."
    shown = ", ".join(sample[:3])

    nums = pd.to_numeric(pd.Series(sample), errors="coerce")
    if nums.notna().all():
        v = nums.astype(float)
        if v.between(19000101, 21001231).all() and (v % 1 == 0).all():
            return (f"the column holds plain numbers like {shown}, which look "
                    "like YYYYMMDD. Insert the dashes.")
        if v.between(20000, 80000).all():
            lo = pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(v.min()))
            hi = pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(v.max()))
            return (f"the column holds plain numbers like {shown}, which are "
                    f"Excel date serials ({lo:%Y-%m-%d} to {hi:%Y-%m-%d}). The "
                    "cells lost their date formatting on the way out of the "
                    "spreadsheet. Reformat the column as a date before "
                    "copying, or write it out as text in ISO form.")
        return (f"the column holds plain numbers like {shown}, which are not "
                "dates at all. Check that the right column was mapped.")

    if any(_MONTHNAME_RE.search(v) for v in sample):
        return (f"the column uses month names, like {shown}. That is readable "
                "but not ISO, and this app does not guess.")

    m = [_SLASHED_RE.match(v) for v in sample]
    if all(m):
        a = [int(x.group(1)) for x in m]
        b = [int(x.group(2)) for x in m]
        if max(a) > 12 and max(b) <= 12:
            order = ("It is day-first (DD/MM/YYYY): the first field goes above "
                     "12.")
        elif max(b) > 12 and max(a) <= 12:
            order = ("It is month-first (MM/DD/YYYY): the second field goes "
                     "above 12.")
        else:
            first_day = f"{a[0]:02d}/{b[0]:02d}"
            order = (f"Both fields stay at 12 or below, so {first_day} could "
                     "be either day-first or month-first and nothing in the "
                     "file says which. Reading it the wrong way round turns a "
                     "monthly history into a fortnight and silently divides "
                     "every rate by thirty, so the app will not choose for "
                     "you.")
        return (f"the column is slash- or dot-separated, like {shown}. {order}")

    return (f"the column is not in a recognised date format; it reads like "
            f"{shown}.")


def parse_dates(series: pd.Series, *, strict: bool = True) -> pd.Series:
    """Parse a date column. ISO 8601 only - ambiguity is refused, not guessed.

    `01/05/2012` is 1 May in most of the world and 5 January in the United
    States, and NOTHING in the string says which. Pandas picks month-first and
    does not complain, so a monthly history written day-first comes back as a
    dozen consecutive days in January - the rows stay in order, the cumulative
    still adds up, and every rate, decline and pressure gap downstream is wrong
    by a factor of thirty. It is the most dangerous class of bug there is: the
    answer changes, nothing looks broken.

    Guessing from the shape of the column was tried and is not good enough. It
    needs more than twelve months of data before day-first becomes detectable
    at all, so a short paste is read backwards in silence; and a column of
    survey dates that happen to fall on low day numbers is ambiguous no matter
    how much of it there is. So the contract is ISO, and anything else is an
    error that names what it found.

    Columns that are already a datetime dtype - an Excel date cell, a parquet
    timestamp - pass through untouched. No text was parsed, so there is
    nothing to be ambiguous about.
    """
    s = pd.Series(series)
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")

    # astype(str) leaves a real None as a float NaN rather than the string
    # "None", so the missing mask has to cover both.
    txt = s.astype(str).str.strip()
    blank = (s.isna() | txt.isna() | txt.eq("")
             | txt.str.lower().isin(("nan", "nat", "none", "<na>")).fillna(True))
    real = txt[~blank]
    if real.empty:
        return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    iso = real.str.match(_ISO_RE) | real.str.match(_ISO_SLASH_RE)
    if not bool(iso.all()):
        if strict:
            raise DateFormatError(
                f"{int((~iso).sum())} of {len(real)} dates are not ISO: "
                f"{_diagnose_dates(real[~iso].tolist())} {ISO_DATE_RULE}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.to_datetime(s, errors="coerce", format="mixed")

    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    parsed = pd.to_datetime(real.str.replace("/", "-", regex=False),
                            errors="coerce", format="ISO8601")
    out.loc[real.index] = parsed
    if strict and parsed.isna().any():
        bad = real[parsed.isna()].tolist()[:3]
        raise DateFormatError(
            f"{int(parsed.isna().sum())} date(s) are ISO-shaped but not real "
            f"dates: {', '.join(bad)}. Check for month 13 or a day the month "
            "does not have.")
    return out


@dataclass
class QCReport:
    """What the QC step did, so it can be reported rather than hidden."""
    n_input: int = 0
    n_after_screen: int = 0
    n_zero_or_negative_gas: int = 0
    n_nonfinite: int = 0
    n_outliers_removed: int = 0
    n_low_uptime_removed: int = 0
    n_pressure_surveys: int = 0
    bdf_start_index: Optional[int] = None
    bdf_start_days: Optional[float] = None
    plateau_end_index: Optional[int] = None
    plateau_end_days: Optional[float] = None
    fit_start_days: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  rows in                 : {self.n_input}",
            f"  rows retained           : {self.n_after_screen}",
            f"  zero/negative gas       : {self.n_zero_or_negative_gas}",
            f"  non-finite values       : {self.n_nonfinite}",
            f"  rate outliers removed   : {self.n_outliers_removed}",
            f"  low-uptime rows removed : {self.n_low_uptime_removed}",
        ]
        if self.plateau_end_days is not None:
            lines.append(f"  plateau ends            : day {self.plateau_end_days:.0f}"
                         f" ({self.plateau_end_days / DAYS_PER_YEAR:.2f} yr)")
        if self.bdf_start_days is not None:
            lines.append(f"  BDF start estimate      : day {self.bdf_start_days:.0f}"
                         f" ({self.bdf_start_days / DAYS_PER_YEAR:.2f} yr)")
        if self.fit_start_days is not None:
            lines.append(f"  decline fit starts      : day {self.fit_start_days:.0f}"
                         f" ({self.fit_start_days / DAYS_PER_YEAR:.2f} yr)")
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


@dataclass
class ProductionData:
    """Cleaned, wellstream-converted production history for one well or entity.

    Attributes created by :meth:`prepare`:
        t            days since first production
        q_gas        separator gas rate, Mscf/d (calendar- or stream-day)
        q_cond       condensate rate, STB/d
        q_ws         wellstream (gas equivalent) rate, Mscf/d
        Gp_ws        cumulative wellstream gas, MMscf
        Gp_sep       cumulative separator gas, MMscf
        Np_cond      cumulative condensate, Mstb
        cgr          condensate-gas ratio, STB/MMscf

    `df` is the QC-filtered frame used for decline fitting. `full_df` is the
    same frame before the rate filters, and is what pressure surveys must be
    read from - see :attr:`surveys`.
    """
    df: pd.DataFrame
    pvt: PVT
    well: str = "WELL"
    qc: QCReport = field(default_factory=QCReport)
    rate_basis: str = "stream-day"      # or 'calendar-day'
    full_df: Optional[pd.DataFrame] = None

    # ---- construction ----------------------------------------------------
    @classmethod
    def prepare(cls,
                df: pd.DataFrame,
                pvt: PVT,
                well: str = "WELL",
                rate_basis: str = "stream-day",
                min_uptime_frac: float = 0.35,
                outlier_sigma: float = 4.0,
                outlier_window: int = 9,
                detect_bdf: bool = True,
                drop_leading_zeros: bool = True) -> "ProductionData":
        """Clean, unit-check and wellstream-convert a raw production table.

        Parameters
        ----------
        rate_basis : 'stream-day' divides volumes by producing days (the basis
            Arps expects); 'calendar-day' divides by elapsed days. Mixing the
            two across a history is the single most common source of spurious
            hyperbolic curvature, so the choice is explicit and recorded.
        min_uptime_frac : rows with less than this fraction of the period
            on-stream are dropped; partial months bias the decline low.
        outlier_sigma : rolling-median MAD threshold for rate outlier rejection.
            Set to None to disable.
        """
        qc = QCReport()
        d = map_columns(df).copy()
        qc.n_input = len(d)

        if "q_gas" not in d.columns:
            raise ValueError("A gas rate column is required (e.g. 'q_gas' in Mscf/d).")
        if "date" not in d.columns:
            raise ValueError("A date column is required.")

        d["date"] = parse_dates(d["date"])
        d = d.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        for col in ("q_gas", "q_cond", "q_water", "p_wf", "p_wh", "p_res", "days_on"):
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")
        if "q_cond" not in d.columns:
            d["q_cond"] = 0.0
            qc.notes.append("No condensate column found; treated as dry gas.")
        if "q_water" not in d.columns:
            d["q_water"] = np.nan

        # Period length and uptime
        # The period a row represents runs FORWARD from its own date to the
        # next one. Differencing backwards labels each row with the previous
        # month's length, so a 31-day month following February gets a 28-day
        # period and its days-on is clipped to 28 - quietly losing real volume.
        dt = (d["date"].shift(-1) - d["date"]).dt.days.astype(float)
        if len(dt) > 1:
            dt.iloc[-1] = dt.iloc[-2] if np.isfinite(dt.iloc[-2]) else 30.4375
        else:
            dt.iloc[-1] = 30.4375
        d["period_days"] = dt.fillna(30.4375).clip(lower=1.0)
        if "days_on" in d.columns:
            # Allow a little slack before clipping: reported days-on can exceed
            # a nominal period by a day without being wrong.
            d["days_on"] = d["days_on"].fillna(d["period_days"]).clip(
                lower=0.0, upper=d["period_days"] + 1.0)
        else:
            d["days_on"] = d["period_days"]
            qc.notes.append("No uptime column; assumed fully on-stream.")
        d["uptime_frac"] = d["days_on"] / d["period_days"]

        # Volumes for the period, from the rate as supplied (assumed stream-day
        # if an uptime column exists, otherwise calendar-day).
        eff_days = d["days_on"].where(d["days_on"] > 0, d["period_days"])
        d["vol_gas"] = d["q_gas"] * eff_days                    # Mscf
        d["vol_cond"] = d["q_cond"].fillna(0.0) * eff_days      # STB

        if rate_basis == "calendar-day":
            d["q_gas"] = d["vol_gas"] / d["period_days"]
            d["q_cond"] = d["vol_cond"] / d["period_days"]
        elif rate_basis != "stream-day":
            raise ValueError("rate_basis must be 'stream-day' or 'calendar-day'.")

        # -- screening ------------------------------------------------------
        nonfinite = (~np.isfinite(d["q_gas"])).sum()
        qc.n_nonfinite = int(nonfinite)
        d["q_gas"] = d["q_gas"].where(np.isfinite(d["q_gas"]), 0.0)

        nonpos = (d["q_gas"] <= 0).sum()
        qc.n_zero_or_negative_gas = int(nonpos)
        if drop_leading_zeros:
            first_pos = d["q_gas"].gt(0).idxmax() if (d["q_gas"] > 0).any() else None
            if first_pos is not None and first_pos > 0:
                # A zero-rate row BEFORE first gas is usually a pre-production
                # static survey - an RFT, a DST build-up, or an estimated p_i
                # written back in. That is the single most valuable pressure
                # point there is, because it is the only one paired with a
                # cumulative of exactly zero. Dropping it with the other
                # leading blanks throws away the reference the whole material
                # balance is measured from.
                lead = d.loc[:first_pos - 1]
                keep_lead = np.zeros(len(lead), dtype=bool)
                if "p_res" in lead.columns:
                    keep_lead = np.isfinite(
                        pd.to_numeric(lead["p_res"], errors="coerce")
                    ).to_numpy()
                d = pd.concat([lead[keep_lead], d.loc[first_pos:]])
            elif first_pos is not None:
                d = d.loc[first_pos:]
        if (d["q_gas"] > 0).sum() < 4:
            raise ValueError(f"[{well}] Fewer than 4 usable points after QC.")
        d = d.reset_index(drop=True)

        # -- derived series, computed on the FULL record ---------------------
        # Shut-in months are kept this far on purpose. They contribute nothing
        # to the cumulative, but a static pressure survey is almost always run
        # on a shut-in or short month - dropping those rows here is what made
        # the material balance report "no pressure data" on wells that plainly
        # had it. Zero-rate rows are filtered out of the fitting frame further
        # down, after the surveys have been snapshotted.
        t0 = d["date"].iloc[0]
        d["t"] = (d["date"] - t0).dt.days.astype(float)
        if d["t"].iloc[0] == 0:          # Duong / power-law models need t > 0
            d.loc[d.index[0], "t"] = max(0.5, 0.5 * d["period_days"].iloc[0])

        d["q_cond"] = d["q_cond"].fillna(0.0).clip(lower=0.0)
        d["q_ge"] = pvt.condensate_to_gas_equiv(d["q_cond"].to_numpy())
        d["q_ws"] = d["q_gas"] + d["q_ge"]

        eff = d["days_on"].where(d["days_on"] > 0, d["period_days"])
        d["Gp_sep"] = np.cumsum(d["q_gas"] * eff) / MSCF_PER_MMSCF       # MMscf
        d["Gp_ws"] = np.cumsum(d["q_ws"] * eff) / MSCF_PER_MMSCF         # MMscf
        d["Np_cond"] = np.cumsum(d["q_cond"] * eff) / 1.0e3              # Mstb
        if "q_water" in d.columns:
            d["Wp_water"] = np.cumsum(
                pd.to_numeric(d["q_water"], errors="coerce").fillna(0.0).clip(
                    lower=0.0) * eff) / 1.0e3                             # Mstb
        with np.errstate(divide="ignore", invalid="ignore"):
            d["cgr"] = np.where(d["q_gas"] > 0,
                                d["q_cond"] / (d["q_gas"] / MSCF_PER_MMSCF),
                                np.nan)                                   # STB/MMscf

        # -- exclusions (cumulatives above are already correct) --------------
        keep_mask = d["q_gas"].to_numpy() > 0          # shut-in months go now

        low_uptime = d["uptime_frac"].to_numpy() < min_uptime_frac
        # Count only producing months here; a shut-in month is already counted
        # under zero/negative gas and should not be double-reported.
        qc.n_low_uptime_removed = int(
            (low_uptime & (d["q_gas"].to_numpy() > 0)).sum())
        keep_mask &= ~low_uptime

        producing = d["q_gas"].to_numpy() > 0
        if outlier_sigma is not None and producing.sum() >= outlier_window:
            # Shut-in months are NaN here, not -inf: log(0) would poison the
            # rolling median and take the neighbouring months out with it.
            q = d["q_gas"].to_numpy(float)
            lq = np.where(producing, np.log(np.where(producing, q, 1.0)), np.nan)
            med = pd.Series(lq).rolling(outlier_window, center=True,
                                        min_periods=3).median().to_numpy()
            resid = lq - med
            mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
            scale = 1.4826 * mad if mad > 0 else np.nanstd(resid)
            if scale and np.isfinite(scale) and scale > 0:
                out = np.isfinite(resid) & (np.abs(resid) > outlier_sigma * scale)
                qc.n_outliers_removed = int(out.sum())
                keep_mask &= ~out

        # Always keep the last PRODUCING point: it carries the cumulative to
        # date. Forcing the last row would readmit a trailing shut-in month.
        last_prod = np.flatnonzero(producing)
        if last_prod.size:
            keep_mask[last_prod[-1]] = True
        if keep_mask.sum() < 4:
            raise ValueError(f"[{well}] Fewer than 4 usable points after QC.")

        # Keep the unfiltered record. Static pressure surveys land on whatever
        # month the gauge was run, which is very often a short or shut-in month
        # that the rate filters above discard. Reading the material balance off
        # the filtered frame silently throws those surveys away - and a survey
        # is a measurement of the reservoir, not of the rate, so the rate
        # filters have no business removing it.
        full = d.copy()
        d = d[keep_mask].reset_index(drop=True)
        qc.n_after_screen = len(d)
        if "p_res" in full.columns:
            qc.n_pressure_surveys = int(
                np.isfinite(pd.to_numeric(full["p_res"],
                                          errors="coerce")).sum())

        obj = cls(df=d, pvt=pvt, well=well, qc=qc, rate_basis=rate_basis,
                  full_df=full.reset_index(drop=True))

        if detect_bdf:
            p_idx, p_t = detect_decline_start(obj.t, obj.q_ws)
            qc.plateau_end_index, qc.plateau_end_days = p_idx, p_t
            # Diagnose flow regime on the freely-declining part only.
            mask = obj.t >= (p_t if p_t is not None else obj.t[0])
            if mask.sum() >= 12:
                b_idx, b_t = detect_bdf_start(obj.t[mask], obj.q_ws[mask])
            else:
                b_idx, b_t = None, None
            qc.bdf_start_index, qc.bdf_start_days = b_idx, b_t
            starts = [v for v in (p_t, b_t) if v is not None]
            qc.fit_start_days = max(starts) if starts else None
            if p_t is not None and p_idx and p_idx > 0:
                qc.notes.append(
                    f"Plateau/constrained period of {p_idx} points excluded from "
                    "the decline fit (rate set by facilities, not the reservoir).")
        return obj

    # ---- convenient array views -----------------------------------------
    @property
    def t(self) -> np.ndarray:
        return self.df["t"].to_numpy(float)

    @property
    def q_gas(self) -> np.ndarray:
        return self.df["q_gas"].to_numpy(float)

    @property
    def q_cond(self) -> np.ndarray:
        return self.df["q_cond"].to_numpy(float)

    @property
    def q_ws(self) -> np.ndarray:
        return self.df["q_ws"].to_numpy(float)

    @property
    def Gp_ws(self) -> np.ndarray:
        return self.df["Gp_ws"].to_numpy(float)

    @property
    def Gp_sep(self) -> np.ndarray:
        return self.df["Gp_sep"].to_numpy(float)

    @property
    def Np_cond(self) -> np.ndarray:
        return self.df["Np_cond"].to_numpy(float)

    @property
    def cgr(self) -> np.ndarray:
        return self.df["cgr"].to_numpy(float)

    @property
    def p_wf(self) -> Optional[np.ndarray]:
        return self.df["p_wf"].to_numpy(float) if "p_wf" in self.df else None

    @property
    def p_res(self) -> Optional[np.ndarray]:
        return self.df["p_res"].to_numpy(float) if "p_res" in self.df else None

    @property
    def surveys(self) -> pd.DataFrame:
        """Static pressure surveys with their cumulative, from the FULL record.

        Returns columns date, t, p_res, p_wf, Gp_ws for every row carrying a
        finite reservoir pressure - including rows the rate QC dropped, because
        a gauge reading is a measurement of the reservoir and has nothing to do
        with whether that month's rate was usable.
        """
        src = self.full_df if self.full_df is not None else self.df
        cols = ["date", "t", "p_res", "p_wf", "Gp_ws", "Gp_sep", "Wp_water"]
        if "p_res" not in src.columns:
            return pd.DataFrame(columns=cols)
        pres = pd.to_numeric(src["p_res"], errors="coerce")
        keep = np.isfinite(pres) & (pres > 0)
        out = src.loc[keep, [c for c in cols if c in src.columns]].copy()
        return out.reset_index(drop=True)

    def window(self, t_min: Optional[float] = None,
               t_max: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return (t, q_wellstream) restricted to a fitting window."""
        m = np.ones(len(self.df), dtype=bool)
        if t_min is not None:
            m &= self.t >= t_min
        if t_max is not None:
            m &= self.t <= t_max
        return self.t[m], self.q_ws[m]


def load_production(path: str,
                    pvt: Optional[PVT] = None,
                    well_column: Optional[str] = "well",
                    column_map: Optional[Dict[str, str]] = None,
                    sheet_name: int | str = 0,
                    **prepare_kwargs) -> pd.DataFrame | Dict[str, ProductionData]:
    """Read a CSV/Excel production file and (if a PVT is given) prepare it.

    Expected columns (aliases are accepted, see CANONICAL_COLUMNS):
        date      production date
        well      well identifier (optional; one entity assumed if absent)
        q_gas     separator gas rate, Mscf/d
        q_cond    condensate rate, STB/d
        days_on   producing days in the period (optional but recommended)
        p_wf      flowing bottomhole pressure, psia (optional)
        p_res     average reservoir pressure, psia (optional, for material balance)

    Returns the raw dataframe if `pvt` is None, otherwise a dict of
    {well: ProductionData}.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        raw = pd.read_excel(path, sheet_name=sheet_name)
    else:
        raw = pd.read_csv(path)
    raw = map_columns(raw, column_map)
    if pvt is None:
        return raw

    out: Dict[str, ProductionData] = {}
    if well_column and well_column in raw.columns:
        for name, grp in raw.groupby(well_column):
            try:
                out[str(name)] = ProductionData.prepare(
                    grp, pvt, well=str(name), **prepare_kwargs)
            except ValueError as exc:
                warnings.warn(f"Skipping {name}: {exc}")
    else:
        out["FIELD"] = ProductionData.prepare(raw, pvt, well="FIELD",
                                              **prepare_kwargs)
    return out


# ==============================================================================
# SECTION 3 -- DECLINE MODELS
# ==============================================================================

class DeclineModel:
    """Base class. Time in days, rate in Mscf/d, cumulative in MMscf.

    **Time origin.** Every model carries a `t0`, the time at which its
    parameters are referenced. This matters: if you fit only the
    boundary-dominated tail of a history but anchor the model at first
    production, `qi` becomes a meaningless back-extrapolation, `Di` and `qi`
    become near-perfectly correlated, and the fit slides onto its bounds. By
    referencing the model to the start of the fitting window instead, `qi` is
    the rate there, the parameters are identifiable, and the covariance is
    usable for uncertainty work. A hyperbolic restricted to t >= t0 is still a
    hyperbolic with the same b, so nothing is lost.

    Subclasses implement `rate`; `cum` is analytic where a closed form exists
    and numerically integrated otherwise. `cum(t)` is always measured from t0.
    """
    name: str = "base"
    param_names: Tuple[str, ...] = ()
    default_bounds: Dict[str, Tuple[float, float]] = {}

    def __init__(self, t0: float = 0.0, **params):
        missing = set(self.param_names) - set(params)
        if missing:
            raise ValueError(f"{self.name}: missing parameters {sorted(missing)}")
        self.t0 = float(t0)
        self.params = {k: float(params[k]) for k in self.param_names}
        for k, v in self.params.items():
            setattr(self, k, v)

    # -- time handling ----------------------------------------------------
    def _tau(self, t) -> np.ndarray:
        """Elapsed time since the model's reference time, clipped at zero."""
        return np.maximum(np.asarray(t, dtype=float) - self.t0, 0.0)

    # -- core ------------------------------------------------------------
    def rate(self, t: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def cum(self, t: np.ndarray) -> np.ndarray:
        """Cumulative production, MMscf, from t0 to t. Numeric fallback."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        tmax = float(max(np.max(t), self.t0))
        grid = np.unique(np.concatenate([
            np.linspace(self.t0, tmax, 4000), np.maximum(t, self.t0)]))
        q = self.rate(grid)
        c = np.concatenate([[0.0], cumulative_trapezoid(q, grid)]) / MSCF_PER_MMSCF
        return np.interp(np.maximum(t, self.t0), grid, c)

    def D(self, t: np.ndarray) -> np.ndarray:
        """Instantaneous nominal decline rate, 1/day."""
        t = np.maximum(np.asarray(t, dtype=float), self.t0)
        h = np.maximum(1e-3, 1e-4 * np.maximum(t - self.t0, 1.0))
        q0 = self.rate(t)
        q1 = self.rate(t + h)
        return -(np.log(np.maximum(q1, 1e-300))
                 - np.log(np.maximum(q0, 1e-300))) / h

    def b_exponent(self, t: np.ndarray) -> np.ndarray:
        """Instantaneous loss-ratio derivative b(t) = d(1/D)/dt."""
        t = np.maximum(np.asarray(t, dtype=float), self.t0)
        h = np.maximum(1e-1, 1e-3 * np.maximum(t - self.t0, 1.0))
        inv0 = 1.0 / np.maximum(self.D(t), 1e-12)
        inv1 = 1.0 / np.maximum(self.D(t + h), 1e-12)
        return (inv1 - inv0) / h

    def time_to_rate(self, q_target: float,
                     t_max: float = 100 * DAYS_PER_YEAR) -> float:
        """Absolute time (days) at which the rate falls to q_target."""
        q_target = float(q_target)
        t_hi = self.t0 + t_max
        if float(self.rate(np.array([self.t0]))[0]) <= q_target:
            return float(self.t0)
        if float(self.rate(np.array([t_hi]))[0]) > q_target:
            return float("nan")

        def f(t):
            return float(self.rate(np.array([t]))[0]) - q_target

        return float(optimize.brentq(f, self.t0, t_hi, xtol=1e-6, maxiter=300))

    def eur(self, q_econ: float, t_max: float = 50 * DAYS_PER_YEAR,
            cum_to_date: float = 0.0) -> Tuple[float, float]:
        """(EUR in MMscf, abandonment time in days) at a given economic limit."""
        t_ab = self.time_to_rate(q_econ, t_max=t_max)
        if not np.isfinite(t_ab):
            t_ab = self.t0 + t_max
        t_ab = min(t_ab, self.t0 + t_max)
        return float(self.cum(np.array([t_ab]))[0] + cum_to_date), float(t_ab)

    # -- helpers ---------------------------------------------------------
    def __repr__(self) -> str:
        ps = ", ".join(f"{k}={v:.6g}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({ps}, t0={self.t0:.4g})"

    def as_dict(self) -> Dict[str, float]:
        return dict(self.params, t0=self.t0)

    @classmethod
    def initial_guess(cls, t: np.ndarray, q: np.ndarray) -> Dict[str, float]:
        raise NotImplementedError

    @classmethod
    def bounds(cls, t: np.ndarray, q: np.ndarray) -> Dict[str, Tuple[float, float]]:
        return dict(cls.default_bounds)


class Arps(DeclineModel):
    """Classic Arps: q = qi / (1 + b*Di*t)^(1/b), with b=0 exponential.

    For a volumetric gas reservoir produced at constant bottomhole pressure,
    theory puts b near 0.4-0.5. b greater than about 1 in a conventional
    setting almost always means the data are still in transient flow, the
    flowing pressure is changing, or wells were added - not a real hyperbolic.
    """
    name = "Arps"
    param_names = ("qi", "Di", "b")
    default_bounds = {"qi": (1e-3, 1e9), "Di": (1e-7, 0.05), "b": (0.0, 2.0)}

    def rate(self, t):
        tau = self._tau(t)
        if self.b < 1e-8:
            return self.qi * np.exp(-self.Di * tau)
        arg = np.maximum(1.0 + self.b * self.Di * tau, 1e-300)
        return self.qi * arg ** (-1.0 / self.b)

    def cum(self, t):
        q = self.rate(t)
        if self.b < 1e-8:
            c = (self.qi - q) / self.Di
        elif abs(self.b - 1.0) < 1e-8:
            c = (self.qi / self.Di) * np.log(np.maximum(self.qi / q, 1e-300))
        else:
            c = (self.qi / ((1.0 - self.b) * self.Di)
                 * (1.0 - (q / self.qi) ** (1.0 - self.b)))
        return c / MSCF_PER_MMSCF

    def D(self, t):
        tau = self._tau(t)
        if self.b < 1e-8:
            return np.full_like(tau, self.Di)
        return self.Di / (1.0 + self.b * self.Di * tau)

    def b_exponent(self, t):
        return np.full_like(self._tau(t), self.b)

    def time_to_rate(self, q_target, t_max=100 * DAYS_PER_YEAR):
        q_target = float(q_target)
        if q_target >= self.qi:
            return float(self.t0)
        if self.b < 1e-8:
            return float(self.t0 + np.log(self.qi / q_target) / self.Di)
        return float(self.t0
                     + ((self.qi / q_target) ** self.b - 1.0) / (self.b * self.Di))

    @classmethod
    def initial_guess(cls, t, q):
        qi = float(np.max(q[:max(3, len(q) // 10)]))
        span = max(float(t[-1] - t[0]), 1.0)
        Di = float(np.log(max(q[0], 1e-9) / max(q[-1], 1e-9)) / span)
        Di = float(np.clip(Di if np.isfinite(Di) and Di > 0 else 1e-3, 1e-6, 0.02))
        return {"qi": qi, "Di": Di, "b": 0.5}

    @classmethod
    def bounds(cls, t, q):
        qmax = float(np.max(q))
        return {"qi": (0.1 * qmax, 10.0 * qmax), "Di": (1e-7, 0.05), "b": (0.0, 2.0)}


class ModifiedHyperbolic(DeclineModel):
    """Hyperbolic decline that switches to exponential at a terminal D_min.

    This is the standard reserves-book fix for the unbounded EUR of a
    hyperbolic with b >= 1. D_min is entered as a nominal 1/day; the helper
    `from_annual` lets you specify it as an effective %/yr instead.
    """
    name = "ModifiedHyperbolic"
    param_names = ("qi", "Di", "b", "Dmin")
    default_bounds = {"qi": (1e-3, 1e9), "Di": (1e-7, 0.05),
                      "b": (0.0, 2.0), "Dmin": (1e-6, 0.01)}

    def __init__(self, t0: float = 0.0, **params):
        super().__init__(t0=t0, **params)
        self.Dmin = min(self.Dmin, self.Di * 0.999999)
        self.params["Dmin"] = self.Dmin
        if self.b > 1e-8 and self.Di > self.Dmin:
            self.tau_switch = (self.Di / self.Dmin - 1.0) / (self.b * self.Di)
        else:
            self.tau_switch = np.inf
        self.t_switch = self.t0 + self.tau_switch
        self._hyp = Arps(t0=self.t0, qi=self.qi, Di=self.Di, b=self.b)
        if np.isfinite(self.t_switch):
            self.q_switch = float(self._hyp.rate(np.array([self.t_switch]))[0])
            self.cum_switch = float(self._hyp.cum(np.array([self.t_switch]))[0])
        else:
            self.q_switch, self.cum_switch = np.nan, np.nan

    @staticmethod
    def annual_effective_to_nominal(d_eff_per_year: float) -> float:
        """Convert an effective annual decline (e.g. 0.08 for 8%/yr) to 1/day."""
        return -math.log(1.0 - float(d_eff_per_year)) / DAYS_PER_YEAR

    @staticmethod
    def nominal_to_annual_effective(d_nominal_per_day: float) -> float:
        return 1.0 - math.exp(-float(d_nominal_per_day) * DAYS_PER_YEAR)

    def rate(self, t):
        t = np.maximum(np.asarray(t, dtype=float), self.t0)
        q = self._hyp.rate(t)
        if np.isfinite(self.t_switch):
            tail = t > self.t_switch
            if np.any(tail):
                q = np.where(tail,
                             self.q_switch * np.exp(-self.Dmin
                                                    * (t - self.t_switch)),
                             q)
        return q

    def cum(self, t):
        t = np.maximum(np.asarray(t, dtype=float), self.t0)
        c = self._hyp.cum(t)
        if np.isfinite(self.t_switch):
            tail = t > self.t_switch
            if np.any(tail):
                q_tail = self.q_switch * np.exp(
                    -self.Dmin * (np.maximum(t, self.t_switch) - self.t_switch))
                c_tail = (self.cum_switch
                          + (self.q_switch - q_tail) / self.Dmin / MSCF_PER_MMSCF)
                c = np.where(tail, c_tail, c)
        return c

    def D(self, t):
        t = np.maximum(np.asarray(t, dtype=float), self.t0)
        d = self._hyp.D(t)
        if np.isfinite(self.t_switch):
            d = np.where(t > self.t_switch, self.Dmin, d)
        return d

    def b_exponent(self, t):
        t = np.maximum(np.asarray(t, dtype=float), self.t0)
        bb = np.full_like(t, self.b)
        if np.isfinite(self.t_switch):
            bb = np.where(t > self.t_switch, 0.0, bb)
        return bb

    def time_to_rate(self, q_target, t_max=100 * DAYS_PER_YEAR):
        q_target = float(q_target)
        if q_target >= self.qi:
            return float(self.t0)
        if np.isfinite(self.t_switch) and q_target < self.q_switch:
            return float(self.t_switch
                         + np.log(self.q_switch / q_target) / self.Dmin)
        return self._hyp.time_to_rate(q_target, t_max)

    @classmethod
    def initial_guess(cls, t, q):
        g = Arps.initial_guess(t, q)
        g["b"] = 0.8
        g["Dmin"] = cls.annual_effective_to_nominal(0.08)
        return g

    @classmethod
    def bounds(cls, t, q):
        bb = Arps.bounds(t, q)
        bb["Dmin"] = (cls.annual_effective_to_nominal(0.02),
                      cls.annual_effective_to_nominal(0.30))
        return bb


class Duong(DeclineModel):
    """Duong (2011) - built for long transient / fracture-dominated flow.

        q(t) = q1 * t^(-m) * exp[ a/(1-m) * (t^(1-m) - 1) ]
        Gp(t) = (q1/a) * exp[ a/(1-m) * (t^(1-m) - 1) ]

    Only appropriate where q/Gp plots as a straight line on log-log against
    time. It has no boundary-dominated limit, so it is a poor choice for a
    conventional condensate reservoir already in depletion.
    """
    name = "Duong"
    param_names = ("q1", "a", "m")
    default_bounds = {"q1": (1e-3, 1e9), "a": (1e-4, 10.0), "m": (0.5, 2.0)}

    def _tau_d(self, t):
        """Duong's time base: days since the reference, offset so tau >= 1."""
        return self._tau(t) + 1.0

    def _phi(self, tau):
        if abs(1.0 - self.m) < 1e-8:
            return np.log(tau) * self.a
        return self.a / (1.0 - self.m) * (tau ** (1.0 - self.m) - 1.0)

    def rate(self, t):
        tau = self._tau_d(t)
        return self.q1 * tau ** (-self.m) * np.exp(np.clip(self._phi(tau), -700, 700))

    def cum(self, t):
        """Cumulative measured from the reference time (Gp(tau=1) = 0)."""
        tau = self._tau_d(t)
        g = (self.q1 / self.a) * np.exp(np.clip(self._phi(tau), -700, 700))
        g0 = self.q1 / self.a
        return (g - g0) / MSCF_PER_MMSCF

    def gp_total(self, t):
        """Duong's own cumulative, Gp = (q1/a)*exp(phi), in MMscf."""
        tau = self._tau_d(t)
        return (self.q1 / self.a) * np.exp(np.clip(self._phi(tau), -700, 700)) / MSCF_PER_MMSCF

    @classmethod
    def initial_guess(cls, t, q):
        return {"q1": float(np.max(q)), "a": 0.5, "m": 1.1}

    @classmethod
    def bounds(cls, t, q):
        qmax = float(np.max(q))
        return {"q1": (1e-2 * qmax, 1e2 * qmax), "a": (1e-4, 20.0), "m": (0.51, 2.5)}


class PowerLawExponential(DeclineModel):
    """Ilk et al. power-law exponential: q = qi * exp(-D_inf*t - D1*t^n).

    Behaves like a hyperbolic early and like an exponential late without the
    unbounded-EUR problem, which makes it a useful cross-check on a modified
    hyperbolic fit.
    """
    name = "PLE"
    param_names = ("qi", "Dinf", "D1", "n")
    default_bounds = {"qi": (1e-3, 1e9), "Dinf": (0.0, 0.01),
                      "D1": (1e-6, 5.0), "n": (0.05, 1.0)}

    def rate(self, t):
        tau = self._tau(t)
        expo = -(self.Dinf * tau + self.D1 * tau ** self.n)
        return self.qi * np.exp(np.clip(expo, -700, 700))

    @classmethod
    def initial_guess(cls, t, q):
        return {"qi": float(np.max(q)),
                "Dinf": ModifiedHyperbolic.annual_effective_to_nominal(0.06),
                "D1": 0.05, "n": 0.4}

    @classmethod
    def bounds(cls, t, q):
        qmax = float(np.max(q))
        return {"qi": (0.2 * qmax, 50 * qmax), "Dinf": (0.0, 0.01),
                "D1": (1e-6, 5.0), "n": (0.05, 1.0)}


class StretchedExponential(DeclineModel):
    """Valko SEPD: q = q0 * exp(-(t/tau)^n). Bounded EUR by construction."""
    name = "SEPD"
    param_names = ("q0", "tau", "n")
    default_bounds = {"q0": (1e-3, 1e9), "tau": (1.0, 1e7), "n": (0.05, 1.5)}

    def rate(self, t):
        tt = self._tau(t)
        return self.q0 * np.exp(-np.clip((tt / self.tau) ** self.n, 0, 700))

    def cum(self, t):
        tt = self._tau(t)
        x = (tt / self.tau) ** self.n
        # int_0^t q dt = q0*tau/n * Gamma(1/n) * P(1/n, x)
        pref = self.q0 * self.tau / self.n * special.gamma(1.0 / self.n)
        return pref * special.gammainc(1.0 / self.n, x) / MSCF_PER_MMSCF

    def eur_infinite(self) -> float:
        """Total recoverable volume as t -> infinity, MMscf."""
        return (self.q0 * self.tau / self.n
                * special.gamma(1.0 / self.n)) / MSCF_PER_MMSCF

    @classmethod
    def initial_guess(cls, t, q):
        return {"q0": float(np.max(q)), "tau": float(max(t[-1], 30.0)), "n": 0.5}

    @classmethod
    def bounds(cls, t, q):
        qmax = float(np.max(q))
        return {"q0": (0.2 * qmax, 50 * qmax),
                "tau": (1.0, 5.0e6), "n": (0.05, 1.5)}


DECLINE_MODELS: Dict[str, type] = {
    "arps": Arps,
    "modified_hyperbolic": ModifiedHyperbolic,
    "duong": Duong,
    "ple": PowerLawExponential,
    "sepd": StretchedExponential,
}


# ==============================================================================
# SECTION 4 -- DIAGNOSTICS AND FITTING
# ==============================================================================

def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average that keeps the array length."""
    window = max(3, int(window) | 1)
    if len(y) < window:
        return y.copy()
    kernel = np.ones(window) / window
    pad = window // 2
    ypad = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(ypad, kernel, mode="valid")


def diagnose_b(t: np.ndarray, q: np.ndarray,
               smooth_window: int = 7) -> pd.DataFrame:
    """Loss-ratio diagnostics: D(t) and b(t) = d(1/D)/dt from the data.

    A genuinely hyperbolic segment shows b roughly constant. A b that keeps
    climbing means transient flow or a changing flowing pressure, and the Arps
    fit over that interval will not be predictive.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    ok = np.isfinite(t) & np.isfinite(q) & (q > 0)
    t, q = t[ok], q[ok]
    if len(t) < 5:
        return pd.DataFrame(columns=["t", "q", "D", "inv_D", "b"])

    lnq = _smooth(np.log(q), smooth_window)
    dlnq = np.gradient(lnq, t)
    D = -dlnq
    D_s = _smooth(D, smooth_window)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_D = np.where(D_s > 1e-12, 1.0 / D_s, np.nan)
    b = np.gradient(_smooth(np.nan_to_num(inv_D, nan=np.nanmean(inv_D)),
                            smooth_window), t)
    return pd.DataFrame({"t": t, "q": q, "D": D_s, "inv_D": inv_D, "b": b})


def detect_decline_start(t: np.ndarray, q: np.ndarray,
                         plateau_tol: float = 0.92,
                         smooth_window: int = 5,
                         max_discard_frac: float = 0.75
                         ) -> Tuple[Optional[int], Optional[float]]:
    """Find where free decline begins, i.e. where the plateau ends.

    Gas condensate wells are usually produced on a contracted or
    facility-limited plateau for years before deliverability falls below the
    target. Rate during that period is set by the contract, not the reservoir,
    and fitting Arps through it is meaningless - it drags b to zero and buries
    the real decline. This returns the first index after the smoothed rate has
    fallen to `plateau_tol` of its peak.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    n = len(t)
    if n < 6:
        return 0, float(t[0]) if n else None
    qs = _smooth(q, min(smooth_window, (n // 3) | 1))
    i_peak = int(np.argmax(qs))
    below = np.where(qs[i_peak:] < plateau_tol * qs[i_peak])[0]
    if below.size == 0:
        return i_peak, float(t[i_peak])
    idx = int(i_peak + below[0])
    idx = min(idx, int(max_discard_frac * n))
    idx = max(0, min(idx, n - 4))
    return idx, float(t[idx])


def detect_bdf_start(t: np.ndarray, q: np.ndarray,
                     min_points: int = 10,
                     r2_threshold: float = 0.80,
                     max_discard_frac: float = 0.40
                     ) -> Tuple[Optional[int], Optional[float]]:
    """Estimate where boundary-dominated flow begins.

    Walks a start index forward and keeps the earliest one whose 1/D-vs-t trend
    is well described by a straight line (constant b) over the remaining
    history. Returns (index into the arrays, time in days).

    `max_discard_frac` is the important guardrail: 1/D computed from noisy
    monthly rates is itself noisy, so an unconstrained search will happily
    throw away most of a perfectly good history and leave a handful of points
    from which b is unidentifiable. Never discard more than this fraction.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    n = len(t)
    if n < min_points + 3:
        return None, None
    diag = diagnose_b(t, q, smooth_window=max(5, min(13, n // 5 | 1)))
    if diag.empty:
        return None, None
    td = diag["t"].to_numpy()
    inv = diag["inv_D"].to_numpy()
    good = np.isfinite(inv) & (inv > 0)

    i_cap = int(max_discard_frac * len(td))
    i_cap = min(i_cap, max(0, len(td) - min_points))

    best = None
    best_r2 = -np.inf
    for i in range(0, i_cap + 1):
        sl = slice(i, len(td))
        x, y = td[sl][good[sl]], inv[sl][good[sl]]
        if len(x) < min_points:
            continue
        res = stats.linregress(x, y)
        if res.slope < -1e-6:                   # 1/D must not be shrinking
            continue
        if res.rvalue ** 2 >= r2_threshold:
            best = i
            break
        if res.rvalue ** 2 > best_r2:
            best_r2, best = res.rvalue ** 2, i
    if best is None:
        best = 0
    t_start = float(td[min(best, len(td) - 1)])
    idx = int(np.searchsorted(t, t_start))
    return idx, t_start


@dataclass
class FitResult:
    """Everything a fit produced, including how well it did and how uncertain."""
    model: DeclineModel
    model_name: str
    params: Dict[str, float]
    stderr: Dict[str, float]
    cov: np.ndarray
    n_points: int
    rmse_log: float
    r2: float
    aic: float
    bic: float
    t_fit: np.ndarray
    q_fit: np.ndarray
    t0: float = 0.0
    fixed: Dict[str, float] = field(default_factory=dict)
    at_bounds: List[str] = field(default_factory=list)
    converged: bool = True
    message: str = ""

    def predict(self, t: np.ndarray) -> np.ndarray:
        return self.model.rate(np.asarray(t, dtype=float))

    def summary(self) -> str:
        lines = [f"  model     : {self.model_name}",
                 f"  points    : {self.n_points}",
                 f"  t0 (ref)  : day {self.t0:.0f} "
                 f"({self.t0 / DAYS_PER_YEAR:.2f} yr on production)",
                 f"  R2 (log)  : {self.r2:.4f}",
                 f"  RMSE(log) : {self.rmse_log:.4f}",
                 f"  AIC / BIC : {self.aic:.1f} / {self.bic:.1f}"]
        for k, v in self.params.items():
            se = self.stderr.get(k, float("nan"))
            tag = " (fixed)" if k in self.fixed else ""
            if k in self.at_bounds:
                tag += "  <-- AT BOUND"
            lines.append(f"  {k:<10}: {v:.6g} +/- {se:.3g}{tag}")
        if "Di" in self.params:
            d_eff = 1.0 - math.exp(-self.params["Di"] * DAYS_PER_YEAR)
            lines.append(f"  Di (eff)  : {100 * d_eff:.1f} %/yr at t0")
        if "Dmin" in self.params:
            d_eff = 1.0 - math.exp(-self.params["Dmin"] * DAYS_PER_YEAR)
            lines.append(f"  Dmin(eff) : {100 * d_eff:.1f} %/yr")
        if self.at_bounds:
            if self.at_bounds == ["b"] and self.params.get("b", 1.0) < 1e-6:
                lines.append("  NOTE      : b sits at its lower bound, i.e. the data "
                             "are exponential. That is a legitimate\n              "
                             "  answer for a depleting gas well, not necessarily a "
                             "failed fit.")
            else:
                lines.append("  WARNING   : parameter(s) pinned to a bound - the fit "
                             "is not identifiable; narrow the\n                window, "
                             "fix a parameter, or use a different model.")
        return "\n".join(lines)


def _pack(params: Dict[str, float], free: Sequence[str],
          bounds: Dict[str, Tuple[float, float]],
          log_scale: Sequence[str]) -> np.ndarray:
    x = []
    for k in free:
        v = params[k]
        lo, hi = bounds[k]
        v = min(max(v, lo * (1 + 1e-9) if lo > 0 else lo + 1e-12), hi * (1 - 1e-9))
        x.append(math.log(max(v, 1e-300)) if k in log_scale else v)
    return np.asarray(x, dtype=float)


def _unpack(x: np.ndarray, free: Sequence[str],
            fixed: Dict[str, float],
            log_scale: Sequence[str]) -> Dict[str, float]:
    out = dict(fixed)
    for k, xi in zip(free, np.atleast_1d(x)):
        out[k] = math.exp(float(np.clip(xi, -700, 700))) if k in log_scale else float(xi)
    return out


def fit_decline(t: np.ndarray,
                q: np.ndarray,
                model: str | type = "modified_hyperbolic",
                fixed: Optional[Dict[str, float]] = None,
                bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                weights: Optional[np.ndarray] = None,
                loss: str = "soft_l1",
                f_scale: float = 0.25,
                multistart: bool = True,
                n_starts: int = 12,
                t0: Optional[float] = None,
                seed: int = 7) -> FitResult:
    """Fit a decline model to (t, q) robustly.

    Residuals are taken in log-rate space, which is the right error model for
    production data (multiplicative noise, orders of magnitude of range) and
    stops the early high-rate points from dominating the fit. A robust loss
    absorbs the odd workover spike. Parameters that span orders of magnitude
    are fitted in log space, and everything is bounded.

    Parameters
    ----------
    fixed : parameters to hold constant, e.g. {"Dmin": 0.00022} or {"b": 0.5}
    weights : per-point weights (same length as t); defaults to uniform. A
        common choice is to up-weight the recent history.
    t0 : time origin the parameters are referenced to. Defaults to the first
        point in the fitting window, which is what keeps qi and Di from being
        almost perfectly correlated when only the tail is fitted. Pass
        `t0=0.0` to anchor at first production instead.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    ok = np.isfinite(t) & np.isfinite(q) & (q > 0)
    t, q = t[ok], q[ok]
    if len(t) < 4:
        raise ValueError("Need at least 4 valid points to fit a decline.")
    order = np.argsort(t)
    t, q = t[order], q[order]

    t0 = float(t[0]) if t0 is None else float(t0)
    tau = t - t0

    cls = DECLINE_MODELS[model] if isinstance(model, str) else model
    fixed = dict(fixed or {})
    bnds = cls.bounds(tau, q)
    if bounds:
        bnds.update(bounds)

    guess = cls.initial_guess(tau, q)
    guess.update({k: v for k, v in fixed.items() if k in guess})
    free = [p for p in cls.param_names if p not in fixed]
    if not free:
        raise ValueError("All parameters are fixed; nothing to fit.")

    log_scale = [p for p in free if p in
                 ("qi", "q0", "q1", "Di", "D1", "Dmin", "tau", "a")]

    w = np.ones_like(q) if weights is None else np.asarray(weights, float)[ok][order]
    w = w / np.mean(w)
    lnq = np.log(q)

    def residual(x: np.ndarray) -> np.ndarray:
        pars = _unpack(x, free, fixed, log_scale)
        try:
            mdl = cls(t0=t0, **pars)
            qm = mdl.rate(t)
        except Exception:
            return np.full_like(lnq, 1e3)
        qm = np.where(np.isfinite(qm) & (qm > 0), qm, 1e-300)
        return w * (np.log(qm) - lnq)

    lo = np.array([math.log(max(bnds[k][0], 1e-300)) if k in log_scale else bnds[k][0]
                   for k in free])
    hi = np.array([math.log(max(bnds[k][1], 1e-299)) if k in log_scale else bnds[k][1]
                   for k in free])

    starts = [_pack(guess, free, bnds, log_scale)]
    if multistart:
        rng = np.random.default_rng(seed)
        # Deterministic sweep over the shape parameter, then random restarts.
        shape_key = next((k for k in ("b", "n", "m") if k in free), None)
        if shape_key is not None:
            klo, khi = bnds[shape_key]
            for v in np.linspace(klo + 0.05 * (khi - klo), khi - 0.05 * (khi - klo), 5):
                g = dict(guess); g[shape_key] = float(v)
                starts.append(_pack(g, free, bnds, log_scale))
        for _ in range(max(0, n_starts - len(starts))):
            x0 = lo + rng.random(len(free)) * (hi - lo)
            starts.append(x0)

    best, best_cost = None, np.inf
    for x0 in starts:
        x0 = np.clip(x0, lo + 1e-12, hi - 1e-12)
        try:
            sol = optimize.least_squares(
                residual, x0, bounds=(lo, hi), loss=loss, f_scale=f_scale,
                max_nfev=4000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        except Exception:
            continue
        if sol.cost < best_cost and np.all(np.isfinite(sol.x)):
            best, best_cost = sol, sol.cost
    if best is None:
        raise RuntimeError(f"{cls.__name__}: all optimisation starts failed.")

    pars = _unpack(best.x, free, fixed, log_scale)
    mdl = cls(t0=t0, **pars)
    qm = mdl.rate(t)

    at_bounds = []
    for p in free:
        lo_p, hi_p = bnds[p]
        span = hi_p - lo_p
        if span > 0 and (pars[p] - lo_p < 1e-4 * span or hi_p - pars[p] < 1e-4 * span):
            at_bounds.append(p)
    r = np.log(np.maximum(qm, 1e-300)) - lnq
    n, k = len(t), len(free)
    ssr = float(np.sum(r ** 2))
    rmse = math.sqrt(ssr / n)
    sst = float(np.sum((lnq - lnq.mean()) ** 2))
    r2 = 1.0 - ssr / sst if sst > 0 else float("nan")
    aic = n * math.log(max(ssr / n, 1e-300)) + 2 * k
    bic = n * math.log(max(ssr / n, 1e-300)) + k * math.log(n)

    # Covariance from the Jacobian, transformed back out of log space.
    stderr: Dict[str, float] = {p: float("nan") for p in cls.param_names}
    cov_full = np.full((len(cls.param_names), len(cls.param_names)), np.nan)
    try:
        J = best.jac
        dof = max(n - k, 1)
        s2 = ssr / dof
        JTJ = J.T @ J
        cov_x = np.linalg.pinv(JTJ) * s2
        # delta method: d(param)/d(x) = param for log-scaled entries, else 1
        scale = np.array([pars[p] if p in log_scale else 1.0 for p in free])
        cov_free = cov_x * np.outer(scale, scale)
        idx = {p: i for i, p in enumerate(cls.param_names)}
        cov_full = np.zeros_like(cov_full)
        for i, pi in enumerate(free):
            for j, pj in enumerate(free):
                cov_full[idx[pi], idx[pj]] = cov_free[i, j]
        for i, p in enumerate(free):
            stderr[p] = float(math.sqrt(max(cov_free[i, i], 0.0)))
        for p in fixed:
            stderr[p] = 0.0
    except Exception:
        pass

    return FitResult(model=mdl, model_name=cls.__name__,
                     params={p: pars[p] for p in cls.param_names},
                     stderr=stderr, cov=cov_full, n_points=n,
                     rmse_log=rmse, r2=r2, aic=aic, bic=bic,
                     t_fit=t, q_fit=q, t0=t0, fixed=fixed,
                     at_bounds=at_bounds,
                     converged=bool(best.success), message=str(best.message))


def rank_models(t: np.ndarray, q: np.ndarray,
                models: Sequence[str] = ("arps", "modified_hyperbolic",
                                         "ple", "sepd", "duong"),
                **kwargs) -> Tuple[pd.DataFrame, Dict[str, FitResult]]:
    """Fit several models to the same data and rank them by AIC.

    Lower AIC is better, but statistics is not physics: prefer the model whose
    late-time behaviour you can defend. Duong will often win on fit quality for
    a transient-dominated history and still give an indefensible EUR.
    """
    fits: Dict[str, FitResult] = {}
    rows = []
    for name in models:
        try:
            fr = fit_decline(t, q, model=name, **kwargs)
        except Exception as exc:                       # keep going on failures
            warnings.warn(f"{name} fit failed: {exc}")
            continue
        fits[name] = fr
        rows.append({"model": name, "R2_log": fr.r2, "RMSE_log": fr.rmse_log,
                     "AIC": fr.aic, "BIC": fr.bic, "converged": fr.converged})
    table = (pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
             if rows else pd.DataFrame())
    return table, fits


# ==============================================================================
# SECTION 5 -- CONDENSATE YIELD (CGR) MODEL
# ==============================================================================

@dataclass
class YieldModel:
    """Condensate-gas ratio as a function of cumulative wellstream gas.

        CGR(Gp) = cgr_min + (cgr_i - cgr_min) * exp(-k * max(Gp - Gp_dew, 0))

    Flat at the initial yield while the reservoir is above the dew point, then
    an exponential fall to a floor as liquid drops out. The floor matters: a
    pure exponential decays to zero, which is not what a real condensate
    reservoir does, and it will understate late-life liquid.

    Fit this against measured field yield and use the CVD liquid-dropout curve
    to sanity-check the shape, not to set the absolute level.
    """
    cgr_i: float
    cgr_min: float
    k: float
    Gp_dew: float = 0.0
    r2: float = float("nan")
    stderr: Dict[str, float] = field(default_factory=dict)

    def __call__(self, Gp: np.ndarray) -> np.ndarray:
        Gp = np.asarray(Gp, dtype=float)
        x = np.maximum(Gp - self.Gp_dew, 0.0)
        return self.cgr_min + (self.cgr_i - self.cgr_min) * np.exp(-self.k * x)

    def summary(self) -> str:
        return ("  CGR model : cgr_min + (cgr_i - cgr_min) * exp(-k*(Gp - Gp_dew))\n"
                f"  cgr_i     : {self.cgr_i:.2f} STB/MMscf\n"
                f"  cgr_min   : {self.cgr_min:.2f} STB/MMscf\n"
                f"  k         : {self.k:.5g} 1/MMscf\n"
                f"  Gp_dew    : {self.Gp_dew:.1f} MMscf\n"
                f"  R2        : {self.r2:.4f}")


def fit_yield_model(Gp: np.ndarray, cgr: np.ndarray,
                    Gp_dew: Optional[float] = None,
                    cgr_min_bounds: Tuple[float, float] = (0.0, None),
                    fit_dewpoint_break: bool = True,
                    seed: int = 3) -> YieldModel:
    """Fit CGR vs cumulative wellstream gas.

    Parameters
    ----------
    Gp : cumulative wellstream gas, MMscf
    cgr : measured condensate-gas ratio, STB/MMscf
    Gp_dew : cumulative at which the reservoir reached the dew point. Supply it
        from a pressure history and PVT when you have one; otherwise set
        `fit_dewpoint_break=True` and it is estimated.
    """
    Gp = np.asarray(Gp, dtype=float)
    cgr = np.asarray(cgr, dtype=float)
    ok = np.isfinite(Gp) & np.isfinite(cgr) & (cgr > 0)
    Gp, cgr = Gp[ok], cgr[ok]
    if len(Gp) < 4:
        raise ValueError("Need at least 4 valid CGR points.")

    cgr0 = float(np.median(cgr[:max(2, len(cgr) // 10)]))
    span = max(float(Gp[-1] - Gp[0]), 1.0)
    lo_min, hi_min = cgr_min_bounds
    hi_min = 0.95 * cgr0 if hi_min is None else hi_min

    fit_break = fit_dewpoint_break and Gp_dew is None
    p0 = [cgr0, max(0.2 * cgr0, lo_min + 1e-6), 2.0 / span]
    lo = [0.3 * cgr0, lo_min, 1e-9]
    hi = [3.0 * cgr0, max(hi_min, lo_min + 1e-6), 50.0 / span]
    if fit_break:
        p0.append(0.1 * span)
        lo.append(0.0)
        hi.append(0.8 * span + float(Gp[0]))

    def resid(x):
        cgr_i, cgr_mn, k = x[0], x[1], x[2]
        gdew = x[3] if fit_break else (Gp_dew or 0.0)
        model = YieldModel(cgr_i, cgr_mn, k, gdew)(Gp)
        return np.log(np.maximum(model, 1e-9)) - np.log(cgr)

    best, best_cost = None, np.inf
    rng = np.random.default_rng(seed)
    starts = [np.array(p0, dtype=float)]
    for _ in range(8):
        starts.append(np.array(lo) + rng.random(len(lo)) * (np.array(hi) - np.array(lo)))
    for x0 in starts:
        try:
            sol = optimize.least_squares(resid, np.clip(x0, lo, hi),
                                         bounds=(lo, hi), loss="soft_l1",
                                         f_scale=0.1, max_nfev=3000)
        except Exception:
            continue
        if sol.cost < best_cost:
            best, best_cost = sol, sol.cost
    if best is None:
        raise RuntimeError("CGR yield fit failed.")

    x = best.x
    gdew = float(x[3]) if fit_break else float(Gp_dew or 0.0)
    ym = YieldModel(float(x[0]), float(x[1]), float(x[2]), gdew)

    pred = ym(Gp)
    ssr = float(np.sum((np.log(pred) - np.log(cgr)) ** 2))
    sst = float(np.sum((np.log(cgr) - np.log(cgr).mean()) ** 2))
    ym.r2 = 1.0 - ssr / sst if sst > 0 else float("nan")
    try:
        dof = max(len(Gp) - len(x), 1)
        cov = np.linalg.pinv(best.jac.T @ best.jac) * (ssr / dof)
        names = ["cgr_i", "cgr_min", "k"] + (["Gp_dew"] if fit_break else [])
        ym.stderr = {n: float(math.sqrt(max(cov[i, i], 0.0)))
                     for i, n in enumerate(names)}
    except Exception:
        pass
    return ym


# ==============================================================================
# SECTION 6 -- MATERIAL BALANCE
# ==============================================================================

@dataclass
class MaterialBalanceResult:
    """p/z straight line plus the Havlena-Odeh diagnostics that police it.

    The p/z intercept on its own is not evidence of anything: a water-driven
    reservoir holds its pressure up, which flattens the trend and inflates the
    intercept, and the plot still looks perfectly straight while it happens.
    The fields below are what tell the two apart.

    ho_table            per-survey F, Eg, F/Eg, apparent G and the flags
    ho_rise             last reliable F/Eg over the first; 1.0 means no influx
    g_ceiling_mmscf     min(F/Eg). Since We >= 0, this is a hard upper bound
                        on G whatever the straight line says
    g_bound_mmscf       smallest apparent G, the same bound reached without
                        any PVT algebra
    drive               'volumetric' | 'borderline' | 'water drive'
    impossible          cumulative production already exceeds the ceiling, so
                        the surveys and the volumes cannot both be right
    """
    ogip_mmscf: float
    ogip_stderr: float
    r2: float
    pz_i: float
    method: str
    pressure: np.ndarray
    gp: np.ndarray
    pz: np.ndarray
    ogip_single_phase: Optional[float] = None
    drive_note: str = ""
    # -- Havlena-Odeh -----------------------------------------------------
    ho_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    ho_rise: float = float("nan")
    g_ceiling_mmscf: float = float("nan")
    g_bound_mmscf: float = float("nan")
    drive: str = "unknown"
    impossible: bool = False
    p_initial: float = float("nan")
    p_initial_known: bool = False
    gp_now: float = float("nan")
    n_surveys: int = 0
    n_skipped: int = 0
    fetkovich: Optional[Dict] = None
    pz_trend_ok: bool = True            # did the straight line decline at all
    pz_note: str = ""                   # why there is no intercept, when there isn't

    @property
    def volumetric(self) -> bool:
        return self.drive == "volumetric"

    @property
    def ogip_exceeds_ceiling(self) -> bool:
        """The straight line is reading through curvature it should have caught."""
        return bool(np.isfinite(self.g_ceiling_mmscf)
                    and np.isfinite(self.ogip_mmscf)
                    and self.ogip_mmscf > 1.10 * self.g_ceiling_mmscf
                    and not self.impossible)

    def summary(self) -> str:
        lines = [f"  method            : {self.method}",
                 f"  surveys used      : {self.n_surveys}"
                 + (f" ({self.n_skipped} earliest dropped)" if self.n_skipped else ""),
                 f"  p_initial         : {self.p_initial:,.0f} psia "
                 + ("(as entered)" if self.p_initial_known
                    else "(extrapolated to zero cumulative)"),
                 f"  (p/z)_i           : {self.pz_i:,.1f} psia",
                 (f"  OGIP (p/z line)   : {self.ogip_mmscf:,.0f} MMscf "
                  f"+/- {self.ogip_stderr:,.0f}" if self.pz_trend_ok else
                  "  OGIP (p/z line)   : not available - the trend does not "
                  "decline"),
                 f"  R2                : {self.r2:.4f}"]
        if self.pz_note:
            lines.append(f"  NOTE              : {self.pz_note}")
        if np.isfinite(self.ho_rise):
            lines.append(f"  F/Eg rise         : {self.ho_rise:.2f}x  -> {self.drive}")
        if np.isfinite(self.g_ceiling_mmscf):
            lines.append(f"  G ceiling, We>=0  : {self.g_ceiling_mmscf:,.0f} MMscf "
                         "= min(F/Eg)")
        if np.isfinite(self.g_bound_mmscf):
            lines.append(f"  smallest apparent G: {self.g_bound_mmscf:,.0f} MMscf")
        if self.ogip_single_phase is not None and np.isfinite(self.ogip_mmscf):
            diff = 100.0 * (self.ogip_single_phase / self.ogip_mmscf - 1.0)
            lines.append(f"  OGIP if single-phase z used : "
                         f"{self.ogip_single_phase:,.0f} MMscf ({diff:+.1f} %)")
            lines.append("  (that difference is the cost of ignoring the "
                         "retrograde liquid in the material balance)")
        if self.impossible:
            lines.append(f"  GUARD             : {self.gp_now:,.0f} MMscf already "
                         f"produced exceeds the {self.g_ceiling_mmscf:,.0f} MMscf "
                         "ceiling -\n                      the surveys and the "
                         "volumes cannot both be right.")
        elif self.ogip_exceeds_ceiling:
            lines.append(f"  WARNING           : the p/z intercept is "
                         f"{self.ogip_mmscf / self.g_ceiling_mmscf:.2f}x the "
                         "material-balance ceiling.")
        if self.fetkovich:
            f = self.fetkovich
            lines.append(f"  Fetkovich aquifer : G {f['G_mmscf']:,.0f} MMscf, "
                         f"Wei {f['Wei_mmbbl']:,.0f} MMbbl, "
                         f"J {f['J_bbl_d_psi']:,.2f} bbl/d/psi")
            lines.append(f"                      We {f['We_mmbbl']:,.1f} MMbbl "
                         f"to date, rms {f['rms_pct']:.2f} %")
            lines.append(f"                      G is only bounded to "
                         f"{f['g_range_mmscf'][0]:,.0f}-"
                         f"{f['g_range_mmscf'][1]:,.0f} MMscf by this fit")
        if self.drive_note:
            lines.append(f"  drive             : {self.drive_note}")
        return "\n".join(lines)


def gas_fvf_rb_per_scf(p: np.ndarray, T_R: float, z: np.ndarray) -> np.ndarray:
    """Gas formation volume factor, reservoir barrels per scf."""
    return 0.0050346 * np.asarray(z, float) * float(T_R) / np.asarray(p, float)


def havlena_odeh_gas(pressure: np.ndarray,
                     gp_mmscf: np.ndarray,
                     pvt: PVT,
                     p_initial: float,
                     water_mstb: Optional[np.ndarray] = None,
                     two_phase: bool = True,
                     method: str = "auto",
                     include_efw: bool = False,
                     sw: float = 0.25,
                     cf: float = 4.0e-6,
                     cw: float = 3.0e-6,
                     bw: float = 1.0,
                     min_depletion: float = 0.05) -> pd.DataFrame:
    """Havlena-Odeh material balance as a straight line, survey by survey.

        F  =  G * (Eg + Efw)  +  We

    with F the reservoir-volume withdrawal, Eg the gas expansion, Efw the rock
    and connate-water expansion, and We cumulative influx. Divide through:

        F / (Eg + Efw)  =  G  +  We / (Eg + Efw)

    so plotting F/Eg against cumulative production is the discriminator. **Flat
    means We is zero and the level is G itself.** Rising means something outside
    the gas is supplying energy, and the p/z intercept is then an artefact
    rather than a volume.

    Because We >= 0, every point gives G <= F/Eg, so **min(F/Eg) is a hard
    ceiling on gas in place** no matter how straight the p/z plot looks.

    Near the reference pressure Eg tends to zero and F/Eg explodes, so points
    shallower than `min_depletion` are computed but flagged unreliable rather
    than being allowed to set the ceiling.

    `include_efw` is **off by default**, which is the usual gas convention: gas
    compressibility dwarfs rock and connate water, so Efw is negligible except
    right at the reference - and right at the reference is precisely where Eg
    is small enough for it to distort the ratio. Left off, F/Eg equals G
    exactly for a closed tank at every depth of depletion, which is what makes
    the diagnostic readable. Turn it on for a hard-rock or very shallow case
    where you want the extra term.

    Returns one row per survey with F, Eg, Efw, F/Eg, apparent G and the flags.
    """
    p = np.asarray(pressure, dtype=float)
    g = np.asarray(gp_mmscf, dtype=float)
    wp = (np.zeros_like(p) if water_mstb is None
          else np.asarray(water_mstb, dtype=float))
    pi = float(p_initial)

    zf = (lambda x: pvt.z_two_phase(x, method=method)) if two_phase else pvt.z
    z = np.asarray(zf(p), dtype=float)
    zi = float(np.asarray(zf(np.array([pi])), dtype=float)[0])

    bg = gas_fvf_rb_per_scf(p, pvt.T_R, z)                 # rb/scf
    bgi = float(gas_fvf_rb_per_scf(np.array([pi]), pvt.T_R, np.array([zi]))[0])

    eg = bg - bgi                                          # rb/scf
    efw = (bgi * ((cw * sw + cf) / max(1.0 - sw, 1e-9)) * (pi - p)
           if include_efw else np.zeros_like(eg))
    et = eg + efw

    f = g * 1.0e6 * bg + wp * 1.0e3 * bw                   # rb

    with np.errstate(divide="ignore", invalid="ignore"):
        f_over_et = np.where(et > 0, f / et / 1.0e6, np.nan)     # MMscf
        pz = p / z
        depleted = 1.0 - pz / (pi / zi)
        apparent_g = np.where(depleted > 0, g / depleted, np.nan)

    reliable = np.isfinite(f_over_et) & (depleted > min_depletion) & (et > 0)
    g_usable = np.isfinite(apparent_g) & (depleted > min_depletion) & (g > 0)

    return pd.DataFrame({
        "p": p, "z": z, "pz": pz, "Gp_mmscf": g, "Wp_mstb": wp,
        "Bg_rb_per_scf": bg, "Eg_rb_per_scf": eg, "Efw_rb_per_scf": efw,
        "F_rb": f, "F_over_Eg_mmscf": f_over_et,
        "depleted_frac": depleted, "apparent_G_mmscf": apparent_g,
        "F_over_Eg_reliable": reliable, "apparent_G_usable": g_usable,
    })


def material_balance_pz(pressure: np.ndarray,
                        gp_mmscf: np.ndarray,
                        pvt: PVT,
                        two_phase: bool = True,
                        method: str = "auto",
                        compare_single_phase: bool = True,
                        water_mstb: Optional[np.ndarray] = None,
                        p_initial: Optional[float] = None,
                        skip_early: int = 0,
                        include_efw: bool = False,
                        sw: float = 0.25,
                        cf: float = 4.0e-6,
                        cw: float = 3.0e-6,
                        bw: float = 1.0,
                        min_depletion: float = 0.05) -> MaterialBalanceResult:
    """Volumetric gas material balance, policed by Havlena-Odeh.

        p/z = (p/z)_i * (1 - Gp/G)      ->      G = -intercept / slope

    The straight line is fitted as before, but the result now also carries the
    F/Eg drive diagnosis, the We >= 0 ceiling on G, and a per-survey apparent-G
    table. **A straight-looking p/z plot is not evidence of a volumetric
    reservoir** - pressure support flattens the trend and inflates the
    intercept while leaving it perfectly straight. Read `drive`, `ho_rise` and
    `g_ceiling_mmscf` before quoting `ogip_mmscf`.

    `p_initial` is used as the reference when supplied; otherwise the p/z line
    is extrapolated to zero cumulative, which is what makes the answer
    *original* gas in place rather than gas in place at the first survey.

    `skip_early` drops that many earliest surveys. Use it when the consistency
    guard fires: the usual cause is a reference pressure taken after first
    production, which makes every Eg downstream too small and F/Eg too large
    everywhere.

    For a retrograde condensate below the dew point you must use a **two-phase**
    z-factor. Below the dew point part of the hydrocarbon has condensed, so the
    two-phase z (which accounts for all remaining moles, liquid included) is
    lower than the single-phase gas z. Points plotted with the single-phase
    value therefore fall below the true line, the apparent p/z trend is too
    steep, and the straight-line extrapolation **understates** OGIP - commonly
    by 10-20%, and increasingly with depth of depletion. Set
    `compare_single_phase=True` to measure that error on your own fluid rather
    than assuming a rule of thumb: the sign and size depend on the CVD data.

    `gp_mmscf` must be the **wellstream** cumulative (gas plus condensate gas
    equivalent), consistent with the two-phase z definition.
    """
    p = np.asarray(pressure, dtype=float)
    g = np.asarray(gp_mmscf, dtype=float)
    w = (np.zeros_like(p) if water_mstb is None
         else np.asarray(water_mstb, dtype=float))
    ok = np.isfinite(p) & np.isfinite(g) & (p > 0)
    p, g, w = p[ok], g[ok], w[ok]
    order = np.argsort(g)                      # depletion order, not file order
    p, g, w = p[order], g[order], w[order]

    n_skipped = int(np.clip(skip_early, 0, max(len(p) - 3, 0)))
    if n_skipped:
        p, g, w = p[n_skipped:], g[n_skipped:], w[n_skipped:]
    if len(p) < 3:
        raise ValueError("Need at least 3 pressure/cumulative pairs.")

    z = pvt.z_two_phase(p, method=method) if two_phase else pvt.z(p)
    pz = p / z
    res = stats.linregress(g, pz)
    # A p/z trend that is flat or rising used to abort the whole calculation.
    # That is backwards. On a strongly supported reservoir the pressure really
    # does hold up or recover, so a non-declining trend is a RESULT - and the
    # parts of this function that matter most in exactly that case (Havlena-
    # Odeh, the We >= 0 ceiling, the apparent-G sequence, and the Fetkovich fit
    # downstream) need no straight line at all. Only the intercept-based OGIP
    # is unavailable, so only that is withheld.
    pz_trend_ok = bool(res.slope < 0)
    pz_note = ""
    if not pz_trend_ok:
        pz_note = (
            f"p/z does not decline with cumulative production: over "
            f"{float(g[-1] - g[0]):,.0f} MMscf the surveys move "
            f"{float(p[-1] - p[0]):+,.0f} psi "
            f"({float(pz[-1] - pz[0]):+,.0f} on p/z). No straight-line OGIP "
            "exists, so it is not reported. If the pressures and units are "
            "right, this is strong pressure support and the material balance "
            "has to be read from F/Eg, the ceiling and the aquifer fit "
            "instead.")
    ogip = -res.intercept / res.slope if pz_trend_ok else float("nan")
    # Propagate the two regression uncertainties (they are correlated; this is
    # the standard first-order approximation and is adequate for screening).
    rel = math.hypot(res.intercept_stderr / max(abs(res.intercept), 1e-12),
                     res.stderr / max(abs(res.slope), 1e-12))
    ogip_se = abs(ogip) * rel

    ogip_sp = None
    if compare_single_phase and two_phase:
        try:
            pz_sp = p / pvt.z(p)
            r2_sp = stats.linregress(g, pz_sp)
            if r2_sp.slope < 0:
                ogip_sp = -r2_sp.intercept / r2_sp.slope
        except Exception:
            pass

    # Curvature diagnostic: upward concavity on p/z vs Gp suggests water influx
    # or pressure support; downward suggests over-stated OGIP or a leaky datum.
    note = ""
    if len(g) >= 5:
        quad = np.polyfit(g, pz, 2)
        curvature = quad[0] * (g.max() - g.min()) ** 2 / max(abs(pz[0]), 1e-9)
        if curvature > 0.05:
            note = ("p/z is concave up - consider water influx or pressure "
                    "support; a straight-line OGIP will be optimistic.")
        elif curvature < -0.05:
            note = ("p/z is concave down - check the datum correction, "
                    "commingling, or a second compartment.")
        else:
            note = "p/z is essentially linear - consistent with volumetric depletion."

    # -- Havlena-Odeh ------------------------------------------------------
    pi_known = p_initial is not None and np.isfinite(p_initial) and p_initial > 0
    if pi_known:
        pi = float(p_initial)
    else:
        # With a flat or rising trend the intercept sits below the data, which
        # would put initial pressure under a pressure that was measured. The
        # reference then falls back to the highest survey - still wrong, but
        # wrong in the direction that understates G rather than inventing a
        # reservoir that was never at that pressure.
        pi = (float(pvt.pressure_from_pz(np.array([res.intercept]),
                                         two_phase=two_phase)[0])
              if pz_trend_ok else float("nan"))
        if not (np.isfinite(pi) and pi >= float(np.max(p))):
            pi = float(np.max(p))
            if not pz_trend_ok:
                pz_note += (" The reference pressure has been taken as the "
                            f"highest survey ({pi:,.0f} psia); supply a real "
                            "initial pressure to make F/Eg meaningful.")

    ho = havlena_odeh_gas(p, g, pvt, pi, water_mstb=w, two_phase=two_phase,
                          method=method, include_efw=include_efw, sw=sw,
                          cf=cf, cw=cw, bw=bw, min_depletion=min_depletion)

    rel = ho["F_over_Eg_reliable"].to_numpy()
    ho_rise, ceiling = float("nan"), float("nan")
    if rel.sum() >= 2:
        vals = ho.loc[rel, "F_over_Eg_mmscf"].to_numpy()
        ho_rise = float(vals[-1] / vals[0]) if vals[0] > 0 else float("nan")
        ceiling = float(np.min(vals))
    elif rel.sum() == 1:
        ceiling = float(ho.loc[rel, "F_over_Eg_mmscf"].iloc[0])

    usable = ho["apparent_G_usable"].to_numpy()
    g_bound = (float(np.nanmin(ho.loc[usable, "apparent_G_mmscf"]))
               if usable.any() else float("nan"))

    # Thresholds follow the convention that 1.10x is flat, 1.25x is influx, and
    # the band between them is reported as borderline rather than rounded into
    # whichever verdict it happens to be nearer.
    if not np.isfinite(ho_rise):
        drive = "unknown"
    elif ho_rise <= 1.10:
        drive = "volumetric"
    elif ho_rise <= 1.25:
        drive = "borderline"
    else:
        drive = "water drive"

    gp_now = float(g[-1])
    impossible = bool(np.isfinite(ceiling) and gp_now > ceiling)

    return MaterialBalanceResult(
        ogip_mmscf=float(ogip), ogip_stderr=float(ogip_se),
        r2=float(res.rvalue ** 2), pz_i=float(res.intercept),
        method=("two-phase z" if two_phase else "single-phase z"),
        pressure=p, gp=g, pz=pz, ogip_single_phase=ogip_sp, drive_note=note,
        pz_trend_ok=pz_trend_ok, pz_note=pz_note,
        ho_table=ho, ho_rise=ho_rise, g_ceiling_mmscf=ceiling,
        g_bound_mmscf=g_bound, drive=drive, impossible=impossible,
        p_initial=pi, p_initial_known=bool(pi_known), gp_now=gp_now,
        n_surveys=len(p), n_skipped=n_skipped)


# ------------------------------------------------------------------------------
# Initial reservoir pressure
# ------------------------------------------------------------------------------

@dataclass
class InitialPressureEstimate:
    """Initial reservoir pressure recovered from the early p/z trend.

    Almost no well is shut in and gauged before it produces. The first static
    survey lands months, sometimes years, into the life, by which time the
    reservoir has already lost pressure. Treating that survey as p_i is not a
    conservative approximation - it silently redefines G as the gas in place on
    the day of the survey, so everything produced before it goes missing from
    the answer, and it makes every Eg too small and every F/Eg too large, which
    is what trips the We>=0 consistency guard on otherwise sound data.
    """
    ok: bool
    p_initial: float = float("nan")     # psia at zero cumulative
    low: float = float("nan")           # psia, spread across accepted windows
    high: float = float("nan")
    method: str = ""
    reason: str = ""                    # why not, when ok is False
    n_surveys: int = 0                  # available
    n_used: int = 0                     # in the chosen early window
    r2: float = float("nan")
    pz_i: float = float("nan")
    ogip_window_mmscf: float = float("nan")
    gap_days: float = float("nan")      # first production to first survey
    gap_mmscf: float = float("nan")     # produced before the first survey
    gap_frac_ogip: float = float("nan")
    reach: float = float("nan")         # gap / cumulative span of the window
    p_first_survey: float = float("nan")
    floor_psia: float = float("nan")    # highest pressure ever measured
    rise_psi: float = float("nan")      # p_initial - p_first_survey
    candidates: Optional[pd.DataFrame] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def gap_months(self) -> float:
        return self.gap_days / 30.4375 if np.isfinite(self.gap_days) else float("nan")

    @property
    def spread_psi(self) -> float:
        return self.high - self.low

    @property
    def confident(self) -> bool:
        """Tight spread, a clean line and a short throw back to Gp = 0."""
        return bool(self.ok and np.isfinite(self.r2) and self.r2 >= 0.97
                    and self.spread_psi <= 0.03 * max(self.p_initial, 1.0)
                    and (not np.isfinite(self.reach) or self.reach <= 1.5))

    def summary(self) -> str:
        if not self.ok:
            return f"Initial pressure not estimated: {self.reason}"
        lines = [
            "Initial pressure estimate",
            "-" * 52,
            f"  p_i               : {self.p_initial:,.0f} psia "
            f"({self.low:,.0f} - {self.high:,.0f})",
            f"  method            : {self.method}",
            f"  surveys used      : {self.n_used} of {self.n_surveys} "
            f"(R2 = {self.r2:.4f})",
            f"  first survey      : {self.p_first_survey:,.0f} psia at "
            f"{self.gap_mmscf:,.0f} MMscf, {self.gap_months:,.1f} months in",
            f"  rise to p_i       : {self.rise_psi:,.0f} psi "
            f"({100 * self.rise_psi / max(self.p_first_survey, 1.0):.1f} %)",
            f"  OGIP of window    : {self.ogip_window_mmscf:,.0f} MMscf "
            f"({100 * self.gap_frac_ogip:.1f} % produced before survey 1)",
        ]
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


def estimate_initial_pressure(pressure: np.ndarray,
                              gp_mmscf: np.ndarray,
                              pvt: PVT,
                              *,
                              two_phase: bool = True,
                              method: str = "auto",
                              t_days: Optional[np.ndarray] = None,
                              p_wf_max: Optional[float] = None,
                              r2_min: float = 0.97,
                              stability_tol: float = 0.03,
                              min_window: int = 3) -> InitialPressureEstimate:
    """Back-extrapolate the EARLY p/z trend to zero cumulative.

    p/z is straight against Gp while depletion is volumetric, so the value at
    Gp = 0 is p_i/z_i and inverting it gives p_i. Two things make this more
    than a one-line regression.

    WHICH SURVEYS. Pressure support flattens the late trend, which rotates the
    fitted line about the data and pulls its intercept DOWN - so a line through
    every survey understates p_i at the same time as it overstates OGIP. The
    estimate therefore comes from the earliest surveys only. The window is
    chosen by growing it one survey at a time and stopping when the implied p_i
    moves away from the level set by the first few: the same test, in the units
    of the answer, that the apparent-G sequence applies to G.

    HOW FAR IT REACHES. Extrapolating back across a gap much larger than the
    span of the surveys used is a long throw on a line fitted to a short base,
    and the result deserves the caveat rather than a decimal place. `reach`
    reports the ratio and the warning fires above 1.5.

    The answer can never be below a pressure that was actually measured, in the
    well or in the reservoir, so both floors are enforced.
    """
    p = np.asarray(pressure, dtype=float)
    g = np.asarray(gp_mmscf, dtype=float)
    ok = np.isfinite(p) & np.isfinite(g) & (p > 0)
    p, g = p[ok], g[ok]
    t = None
    if t_days is not None:
        t = np.asarray(t_days, dtype=float)[ok]
    order = np.argsort(g)
    p, g = p[order], g[order]
    if t is not None:
        t = t[order]

    n = len(p)
    floor = float(np.max(p)) if n else float("nan")
    if p_wf_max is not None and np.isfinite(p_wf_max):
        floor = float(np.nanmax([floor, p_wf_max]))

    base = InitialPressureEstimate(
        ok=False, n_surveys=n, floor_psia=floor,
        p_first_survey=float(p[0]) if n else float("nan"),
        gap_mmscf=float(g[0]) if n else float("nan"),
        gap_days=float(t[0]) if (t is not None and n) else float("nan"))

    if n == 0:
        base.reason = ("there is no reservoir pressure in the data at all. "
                       "Initial pressure cannot be inferred from rates alone - "
                       "it needs at least two static surveys, a pre-production "
                       "RFT/DST, or a regional pressure gradient.")
        return base
    if n == 1:
        base.reason = (f"only one static survey ({p[0]:,.0f} psia) is "
                       "available, and a p/z line needs at least two. That "
                       "survey is a lower bound on p_i, not p_i itself.")
        return base

    # -- candidate windows -------------------------------------------------
    rows = []
    for k in range(2, n + 1):
        gk, pk = g[:k], p[:k]
        if np.ptp(gk) <= 0:
            continue
        zk = pvt.z_two_phase(pk, method=method) if two_phase else pvt.z(pk)
        pzk = pk / zk
        res = stats.linregress(gk, pzk)
        if res.slope >= 0:
            continue
        pz0 = float(res.intercept)
        pi_k = float(pvt.pressure_from_pz(np.array([pz0]),
                                          two_phase=two_phase)[0])
        r2k = float(res.rvalue ** 2) if k > 2 else 1.0
        rows.append({"n_surveys": k, "p_initial": pi_k, "pz_i": pz0,
                     "r2": r2k, "ogip_mmscf": -pz0 / res.slope,
                     "gp_span_mmscf": float(np.ptp(gk))})
    if not rows:
        base.reason = ("p/z does not decline with cumulative production over "
                       "any early window - check the pressure units, the "
                       "datum correction, or whether these are flowing "
                       "pressures rather than static ones.")
        return base

    cand = pd.DataFrame(rows)

    # -- choose the window -------------------------------------------------
    # The reference level is the smallest window that can actually be judged.
    # k = 2 has no residual and so no R2; k = 3 is the first fit with a degree
    # of freedom, which is why it anchors the stability test rather than k = 2.
    anchor_k = min(max(min_window, 3), int(cand["n_surveys"].max()))
    anchor = cand.loc[cand["n_surveys"] <= anchor_k, "p_initial"].median()
    acc = []
    for _, r in cand.iterrows():
        near = abs(r["p_initial"] / anchor - 1.0) <= stability_tol
        clean = (r["n_surveys"] <= 3) or (r["r2"] >= r2_min)
        if near and clean:
            acc.append(int(r["n_surveys"]))
        elif acc:
            break                       # stop at the first departure, not the last
    if not acc:
        acc = [int(cand["n_surveys"].iloc[0])]
    cand["used"] = cand["n_surveys"].isin(acc)

    chosen = cand[cand["n_surveys"] == max(acc)].iloc[0]
    accepted = cand[cand["used"]]
    pi = float(chosen["p_initial"])
    lo = float(accepted["p_initial"].min())
    hi = float(accepted["p_initial"].max())

    warns: List[str] = []
    if np.isfinite(floor) and pi < floor:
        warns.append(
            f"the extrapolated value ({pi:,.0f} psia) came out below a "
            f"pressure that was measured ({floor:,.0f} psia), which is "
            "impossible; the measured pressure has been used instead.")
        pi = floor
        hi = max(hi, floor)
        lo = max(lo, floor)
    lo, hi = min(lo, pi), max(hi, pi)

    span = float(chosen["gp_span_mmscf"])
    gap = float(g[0])
    reach = gap / span if span > 0 else float("inf")
    ogip_w = float(chosen["ogip_mmscf"])

    if reach > 1.5:
        warns.append(
            f"the extrapolation reaches back {gap:,.0f} MMscf from surveys "
            f"that span only {span:,.0f} MMscf ({reach:.1f}x). Treat p_i as "
            "an order of magnitude on the correction, not a measured number.")
    if len(acc) < max(2, n - 1) and max(acc) < n:
        warns.append(
            f"surveys after the first {max(acc)} were excluded: the implied "
            "p_i moves away from the early level there, which is the "
            "signature of pressure support. That is the right thing for this "
            "estimate, and it also means a straight-line OGIP through all "
            f"{n} surveys will be too high.")
    if np.isfinite(ogip_w) and ogip_w > 0 and gap / ogip_w > 0.15:
        warns.append(
            f"{100 * gap / ogip_w:.0f} % of the gas in place had already been "
            "produced before the first survey. The unmeasured early depletion "
            "is a large part of the answer.")
    if max(acc) == 2:
        warns.append("only two surveys support the line, so there is no "
                     "residual and no way to tell whether it is straight.")

    return InitialPressureEstimate(
        ok=True, p_initial=pi, low=lo, high=hi,
        method=(f"p/z back-extrapolation on the first {max(acc)} survey(s), "
                f"{'two-phase' if two_phase else 'single-phase'} z"),
        n_surveys=n, n_used=int(max(acc)), r2=float(chosen["r2"]),
        pz_i=float(chosen["pz_i"]), ogip_window_mmscf=ogip_w,
        gap_days=float(t[0]) if t is not None else float("nan"),
        gap_mmscf=gap,
        gap_frac_ogip=(gap / ogip_w if np.isfinite(ogip_w) and ogip_w > 0
                       else float("nan")),
        reach=reach, p_first_survey=float(p[0]), floor_psia=floor,
        rise_psi=pi - float(p[0]),
        candidates=cand.reset_index(drop=True), warnings=warns)


def field_survey_table(wells: Dict[str, "ProductionData"]) -> pd.DataFrame:
    """Pair every static survey in a field with the FIELD cumulative that day.

    A tank has one pressure and one cumulative. With several wells on it, the
    cumulative against which a survey must be read is the field's, not the
    gauged well's - reading it against that one well's own production is the
    commonest way a multi-well p/z plot ends up with a scatter of parallel
    lines instead of one trend.
    """
    frames = []
    for name, pdata in wells.items():
        s = pdata.surveys
        if len(s):
            s = s.copy()
            s["well"] = name
            frames.append(s)
    if not frames:
        return pd.DataFrame(columns=["date", "p_res", "Gp_ws", "Wp_water",
                                     "n_wells", "t"])
    surv = pd.concat(frames, ignore_index=True).sort_values("date")

    # Step-interpolate each well's cumulative onto the survey dates: a well
    # contributes nothing before its own first production and its latest
    # cumulative thereafter.
    dates = pd.to_datetime(surv["date"]).to_numpy()
    tot_g = np.zeros(len(surv))
    tot_w = np.zeros(len(surv))
    for _, pdata in wells.items():
        src = pdata.full_df if pdata.full_df is not None else pdata.df
        wd = pd.to_datetime(src["date"]).to_numpy()
        for col, acc in (("Gp_ws", tot_g), ("Wp_water", tot_w)):
            if col not in src.columns:
                continue
            v = pd.to_numeric(src[col], errors="coerce").ffill().fillna(0.0)
            idx = np.searchsorted(wd, dates, side="right") - 1
            acc += np.where(idx >= 0, v.to_numpy()[np.clip(idx, 0, None)], 0.0)

    t0 = min(pd.to_datetime(
        (p.full_df if p.full_df is not None else p.df)["date"]).min()
        for p in wells.values())
    out = pd.DataFrame({
        "date": pd.to_datetime(surv["date"]).to_numpy(),
        "well": surv["well"].to_numpy(),
        "p_res": pd.to_numeric(surv["p_res"], errors="coerce").to_numpy(),
        "p_wf": (pd.to_numeric(surv["p_wf"], errors="coerce").to_numpy()
                 if "p_wf" in surv.columns else np.nan),
        "Gp_ws": tot_g, "Wp_water": tot_w})
    out["t"] = (out["date"] - t0).dt.days.astype(float)
    # Two gauges run in the same month are two reads of one reservoir state.
    agg = (out.groupby("date", as_index=False)
              .agg(p_res=("p_res", "mean"), p_wf=("p_wf", "mean"),
                   Gp_ws=("Gp_ws", "mean"), Wp_water=("Wp_water", "mean"),
                   t=("t", "first"), n_wells=("well", "nunique")))
    return agg.sort_values("Gp_ws").reset_index(drop=True)


def estimate_initial_pressure_from_wells(
        wells: Dict[str, "ProductionData"],
        pvt: PVT, *, two_phase: bool = True,
        method: str = "auto", **kwargs) -> InitialPressureEstimate:
    """`estimate_initial_pressure` driven straight off prepared well data."""
    surv = field_survey_table(wells)
    p_wf_max = float("nan")
    for _, pdata in wells.items():
        src = pdata.full_df if pdata.full_df is not None else pdata.df
        if "p_wf" in src.columns:
            v = pd.to_numeric(src["p_wf"], errors="coerce")
            if np.isfinite(v).any():
                p_wf_max = float(np.nanmax([p_wf_max, np.nanmax(v)]))
    if not len(surv):
        return estimate_initial_pressure(
            np.array([]), np.array([]), pvt, two_phase=two_phase,
            method=method,
            p_wf_max=(p_wf_max if np.isfinite(p_wf_max) else None), **kwargs)
    return estimate_initial_pressure(
        surv["p_res"].to_numpy(float), surv["Gp_ws"].to_numpy(float), pvt,
        two_phase=two_phase, method=method, t_days=surv["t"].to_numpy(float),
        p_wf_max=(p_wf_max if np.isfinite(p_wf_max) else None), **kwargs)


# ------------------------------------------------------------------------------
# Fetkovich aquifer
# ------------------------------------------------------------------------------

CEILING_SLACK = 1.05        # how far above min(F/Eg) a cap may sit before clipping
PZ_MAX_REL_SE = 0.50        # above this the p/z intercept is too loose to cap with


@dataclass
class OGIPChoice:
    """Which gas in place the forecast is held to, and why that one."""
    value: float = float("nan")         # MMscf, or nan for no cap
    source: str = "none"                # 'p/z line' | 'Fetkovich' | 'ceiling'
    rel_sigma: float = 0.15             # spread for the Monte Carlo cap
    reason: str = ""
    candidates: Dict[str, float] = field(default_factory=dict)
    clipped_to_ceiling: bool = False

    def __repr__(self) -> str:
        if not np.isfinite(self.value):
            return f"<OGIPChoice none: {self.reason}>"
        return (f"<OGIPChoice {self.value:,.0f} MMscf from {self.source} "
                f"+/-{100 * self.rel_sigma:.0f}%>")


def select_ogip(matbal: Optional["MaterialBalanceResult"],
                fmb: Optional[Dict] = None,
                mode: str = "auto") -> OGIPChoice:
    """Pick the gas in place the forecast should be capped at.

    The p/z intercept is NOT the default, because it is only gas in place when
    the tank is closed. Under pressure support the influx holds p/z up, which
    flattens the trend and pushes the intercept out past anything the reservoir
    contains - the same number the drive diagnostics spend their time warning
    about. Capping a forecast with it in that case is worse than not capping at
    all: it puts a large, official-looking, wrong volume into the reserves.

    The order of preference, therefore, is by what each number means:

      volumetric      the p/z intercept IS G, and the ceiling agrees with it
      supported       Fetkovich G, which models the influx instead of absorbing
                      it into the intercept, when the fit is good and the locus
                      is not hopelessly wide
      otherwise       min(F/Eg), the We >= 0 ceiling - not an estimate but a
                      hard upper bound, which is the right shape for a cap

    Whatever is chosen is clipped to the ceiling, since no model output may
    exceed a bound that follows from We >= 0 alone.

    `rel_sigma` carries the spread into the Monte Carlo instead of the fixed
    15 % that used to be assumed for every well: the Fetkovich locus half-width
    where there is one, the regression standard error on the p/z intercept
    otherwise.
    """
    mode = (mode or "auto").lower()
    if mode in ("none", "off") or matbal is None:
        if matbal is None and fmb is not None and mode not in ("none", "off"):
            v = float(fmb.get("ogip_contacted_mmscf", float("nan")))
            if np.isfinite(v) and v > 0:
                return OGIPChoice(value=v, source="FMB (contacted)",
                                  rel_sigma=0.25,
                                  reason="No static surveys, so the cap comes "
                                         "from the flowing material balance. "
                                         "It measures contacted gas, which on "
                                         "a condensate well below the dew "
                                         "point reads low.",
                                  candidates={"FMB": v})
        return OGIPChoice(reason=("No material balance, so the forecast is "
                                  "not capped."), source="none")

    ceil = float(matbal.g_ceiling_mmscf)
    pz = float(matbal.ogip_mmscf)
    fk = matbal.fetkovich or None
    fk_g = float(fk["G_mmscf"]) if fk else float("nan")
    cands = {}
    if np.isfinite(pz):
        cands["p/z line"] = pz
    if np.isfinite(ceil):
        cands["We>=0 ceiling"] = ceil
    if np.isfinite(fk_g):
        cands["Fetkovich"] = fk_g

    # A p/z line so nearly flat that its intercept is meaningless still has a
    # finite intercept. Dividing a big number by a slope that is almost zero
    # gave a 3 Tcf "gas in place" on a well that has produced 19 Bscf - useless
    # as a cap and actively misleading as a headline. The regression's own
    # standard error says when that has happened.
    pz_rel_se = float("inf")
    if np.isfinite(matbal.ogip_stderr) and np.isfinite(pz) and pz > 0:
        pz_rel_se = float(matbal.ogip_stderr) / pz
    pz_usable = bool(np.isfinite(pz) and pz > 0 and pz_rel_se <= PZ_MAX_REL_SE)

    def _sigma_pz() -> float:
        if np.isfinite(pz_rel_se):
            return float(np.clip(pz_rel_se, 0.05, 0.50))
        return 0.15

    def _sigma_fk() -> float:
        if fk and np.isfinite(fk_g) and fk_g > 0:
            lo, hi = fk.get("g_range_mmscf", (np.nan, np.nan))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                return float(np.clip(0.5 * (hi - lo) / fk_g, 0.05, 0.60))
        return 0.25

    if mode in ("p/z", "pz", "p/z line"):
        if not np.isfinite(pz):
            return OGIPChoice(reason="The p/z line was requested but the trend "
                                     "does not decline, so it has no "
                                     "intercept; the forecast is not capped.",
                              candidates=cands)
        if not pz_usable:
            return OGIPChoice(
                reason=(f"The p/z line was requested but its intercept is "
                        f"{pz:,.0f} MMscf +/- {100 * pz_rel_se:.0f} %, which "
                        "is not a number anything should be held to; the "
                        "forecast is not capped."),
                candidates=cands)
        pick, src, sig = pz, "p/z line", _sigma_pz()
        why = "Chosen explicitly."
    elif mode in ("fetkovich", "aquifer"):
        if not np.isfinite(fk_g):
            return OGIPChoice(reason="Fetkovich was requested but no aquifer "
                                     "fit is available; the forecast is not "
                                     "capped.", candidates=cands)
        pick, src, sig = fk_g, "Fetkovich", _sigma_fk()
        why = "Chosen explicitly."
    elif mode in ("ceiling", "we", "bound"):
        if not np.isfinite(ceil):
            return OGIPChoice(reason="The ceiling was requested but F/Eg could "
                                     "not be evaluated; the forecast is not "
                                     "capped.", candidates=cands)
        pick, src, sig = ceil, "We>=0 ceiling", 0.10
        why = "Chosen explicitly."
    elif mode != "auto":
        raise ValueError(f"Unknown OGIP cap mode {mode!r}.")
    else:
        volumetric = (matbal.drive == "volumetric"
                      and not matbal.impossible
                      and not matbal.ogip_exceeds_ceiling
                      and pz_usable)
        fk_usable = bool(
            fk and np.isfinite(fk_g) and fk_g > 0
            and float(fk.get("rms_pct", 1e9)) < 5.0)
        if volumetric:
            pick, src, sig = pz, "p/z line", _sigma_pz()
            why = (f"F/Eg is flat ({matbal.ho_rise:.2f}x), so the tank is "
                   "closed and the p/z intercept is gas in place rather than "
                   "an artefact of pressure support.")
        elif fk_usable:
            pick, src, sig = fk_g, "Fetkovich", _sigma_fk()
            drive_bit = (f"The drive reads {matbal.drive} (F/Eg rises "
                         f"{matbal.ho_rise:.2f}x)"
                         if np.isfinite(matbal.ho_rise) else
                         "F/Eg could not be evaluated - no survey is far "
                         "enough into depletion for Eg to mean anything - so "
                         "the drive is undiagnosed")
            pz_bit = (f"the p/z intercept ({pz:,.0f} MMscf) cannot be trusted"
                      if np.isfinite(pz) else "there is no p/z intercept")
            why = (f"{drive_bit}, and {pz_bit}. The Fetkovich fit models the "
                   "influx explicitly and matches the pressure history to "
                   f"{float(fk['rms_pct']):.2f} %.")
        elif np.isfinite(ceil):
            pick, src, sig = ceil, "We>=0 ceiling", 0.10
            why = (f"The drive reads {matbal.drive} and no usable aquifer fit "
                   "is available, so the cap falls back to the hard bound "
                   "min(F/Eg), which holds whatever the influx turns out to "
                   "be.")
        elif pz_usable:
            pick, src, sig = pz, "p/z line", _sigma_pz()
            why = ("Neither the ceiling nor an aquifer fit could be evaluated, "
                   "so the p/z intercept is all there is. Treat the cap as "
                   "indicative.")
        else:
            return OGIPChoice(
                reason=(f"Nothing here is fit to cap a forecast: the drive "
                        f"reads {matbal.drive}, there is no usable aquifer "
                        "fit, F/Eg gives no ceiling, and the p/z intercept "
                        + (f"is {pz:,.0f} MMscf +/- {100 * pz_rel_se:.0f} %"
                           if np.isfinite(pz) else "does not exist")
                        + ". The forecast runs uncapped, which is the honest "
                        "outcome - supply an initial pressure, or more "
                        "surveys, to get a bound."),
                candidates=cands)

    # min(F/Eg) is a minimum over surveys, so scatter biases it low: on clean
    # synthetic data it lands within 2 % of the true G, and real surveys are
    # not clean. A cap that is slightly high merely fails to bite, while one
    # that is slightly low truncates reserves that are there - so the clip is
    # given a little room and only fires when the excess is real.
    clipped = False
    if np.isfinite(ceil) and np.isfinite(pick) and pick > ceil * CEILING_SLACK:
        why += (f" Clipped from {pick:,.0f} to the We >= 0 ceiling of "
                f"{ceil:,.0f} MMscf, which no gas in place may exceed.")
        pick, clipped = ceil, True

    if not (np.isfinite(pick) and pick > 0):
        return OGIPChoice(reason="No usable gas in place; the forecast is not "
                                 "capped.", candidates=cands)
    return OGIPChoice(value=float(pick), source=src, rel_sigma=float(sig),
                      reason=why, candidates=cands, clipped_to_ceiling=clipped)


def _fetkovich_march(g_scf: float, wei_bbl: float, tau_days: float,
                     t_days: np.ndarray, gp_scf: np.ndarray, wp_bbl: np.ndarray,
                     pvt: PVT, pi: float, bgi: float, bw: float,
                     two_phase: bool, n_inner: int = 12) -> np.ndarray:
    """March the Fetkovich tank forward and return the predicted pressures.

    Fetkovich treats the aquifer as a tank draining at pseudo-steady state:

        dWe/dt = J * (p_aq - p_res),      p_aq = p_i * (1 - We/Wei)

    which integrates over a step of constant reservoir pressure to

        dWe = (Wei/p_i) * (p_aq - p_res) * [1 - exp(-dt/tau)],   tau = Wei/(J*p_i)

    Pressure is not iterated against a residual here: the gas material balance
    gives it in closed form once We is known, because

        F = G*Eg + We    ->    Bg = (G*Bgi - We + Wp*Bw) / (G - Gp)

    and p/z follows from Bg directly. The only implicit part is that dWe wants
    the average reservoir pressure over the step, so the step is repeated a few
    times with the average updated - which converges in two or three passes.
    """
    n = len(t_days)
    p_pred = np.empty(n, dtype=float)
    p_pred[0] = pi
    we = 0.0
    c = 0.0050346 * pvt.T_R

    for i in range(1, n):
        dt = max(float(t_days[i] - t_days[i - 1]), 0.0)
        decay = 1.0 - math.exp(-dt / tau_days) if tau_days > 0 else 1.0
        p_prev = p_pred[i - 1]
        p_now = p_prev
        we_new = we
        for _ in range(n_inner):
            p_avg = 0.5 * (p_prev + p_now)
            p_aq = pi * (1.0 - we / wei_bbl)
            dwe = (wei_bbl / pi) * (p_aq - p_avg) * decay
            we_new = max(we + dwe, 0.0)
            denom = g_scf - gp_scf[i]
            if denom <= 0:
                return np.full(n, np.nan)
            bg = (g_scf * bgi - we_new + wp_bbl[i] * bw) / denom
            if not np.isfinite(bg) or bg <= 0:
                return np.full(n, np.nan)
            p_new = float(pvt.pressure_from_pz(np.array([c / bg]),
                                               two_phase=two_phase)[0])
            if abs(p_new - p_now) < 0.05:
                p_now = p_new
                break
            p_now = 0.5 * p_now + 0.5 * p_new
        we = we_new
        p_pred[i] = p_now
    return p_pred


def fetkovich_aquifer_fit(t_days: np.ndarray,
                          pressure: np.ndarray,
                          gp_mmscf: np.ndarray,
                          pvt: PVT,
                          p_initial: float,
                          water_mstb: Optional[np.ndarray] = None,
                          two_phase: bool = True,
                          method: str = "auto",
                          bw: float = 1.0,
                          g_grid: int = 10,
                          wei_grid: int = 11,
                          tau_grid: int = 11,
                          g_max_multiple: float = 6.0,
                          locus_tolerance: float = 1.08,
                          refine: bool = True) -> Optional[Dict]:
    """Fit a Fetkovich aquifer to a pressure history: returns G, Wei, J and We.

    The search is over (G, Wei, tau) rather than (G, Wei, J), because tau =
    Wei/(J*p_i) is the aquifer's response time in days and is the thing the data
    can actually constrain. J falls out as Wei/(p_i*tau).

    **The solution is not unique, and the locus is the honest output.** A large
    aquifer with a small J and a small aquifer with a large J deliver nearly the
    same influx over a finite record; only late depletion of the aquifer itself
    separates them. The returned `locus` holds every grid triplet within
    `locus_tolerance` of the best rms, and it is usually a long valley rather
    than a point. Quote a range from it, not the single best triplet.
    """
    t = np.asarray(t_days, dtype=float)
    p = np.asarray(pressure, dtype=float)
    g = np.asarray(gp_mmscf, dtype=float)
    w = (np.zeros_like(p) if water_mstb is None
         else np.asarray(water_mstb, dtype=float))
    ok = np.isfinite(t) & np.isfinite(p) & np.isfinite(g) & (p > 0)
    t, p, g, w = t[ok], p[ok], g[ok], w[ok]
    order = np.argsort(t)
    t, p, g, w = t[order], p[order], g[order], w[order]
    if len(t) < 4:
        return None

    pi = float(p_initial)
    zf = (lambda x: pvt.z_two_phase(x, method=method)) if two_phase else pvt.z
    zi = float(np.asarray(zf(np.array([pi])), dtype=float)[0])
    bgi = 0.0050346 * zi * pvt.T_R / pi              # rb/scf
    gp_scf = g * 1.0e6
    wp_bbl = w * 1.0e3

    gp_max = float(gp_scf.max())
    g_lo = 1.05 * gp_max
    g_hi = max(g_max_multiple * gp_max, g_lo * 1.5)
    span = max(float(t[-1] - t[0]), 1.0)

    # Bounds are enforced inside the objective so every caller - the grid, the
    # global refine and the profile - obeys them. Without this the optimiser
    # wanders off to an aquifer of 10^10 MMbbl with a 10^13-day time constant,
    # which is numerically an inert tank and physically nothing at all.
    hcpv0 = gp_max * bgi
    wei_lo, wei_hi = 1.0e-3 * hcpv0, 1.0e4 * hcpv0
    tau_lo, tau_hi = 1.0e-3 * span, 1.0e4 * span

    def rms(gv: float, weiv: float, tauv: float) -> float:
        if gv <= g_lo * 0.999 or weiv <= 0 or tauv <= 0:
            return np.inf
        if not (wei_lo <= weiv <= wei_hi) or not (tau_lo <= tauv <= tau_hi):
            return np.inf
        pp = _fetkovich_march(gv, weiv, tauv, t, gp_scf, wp_bbl, pvt, pi,
                              bgi, bw, two_phase)
        if not np.all(np.isfinite(pp)):
            return np.inf
        return float(np.sqrt(np.mean(((pp - p) / p) ** 2)) * 100.0)

    # Wei is scaled to the reservoir volume of the gas produced so far, so the
    # grid spans the same physical range whatever the size of the field.
    hcpv_bbl = hcpv0
    gs = np.geomspace(g_lo, g_hi, g_grid)
    weis = np.geomspace(0.05 * hcpv_bbl, 200.0 * hcpv_bbl, wei_grid)
    taus = np.geomspace(0.02 * span, 200.0 * span, tau_grid)

    rows = []
    best = (np.inf, None)
    for gv in gs:
        for wv in weis:
            for tv in taus:
                e = rms(gv, wv, tv)
                if np.isfinite(e):
                    rows.append((gv, wv, tv, e))
                    if e < best[0]:
                        best = (e, (gv, wv, tv))
    if best[1] is None:
        return None

    gb, wb, tb = best[1]
    if refine:
        def obj(x):
            return rms(math.exp(x[0]), math.exp(x[1]), math.exp(x[2]))
        try:
            sol = optimize.minimize(
                obj, np.log([gb, wb, tb]), method="Nelder-Mead",
                options=dict(maxiter=400, xatol=1e-3, fatol=1e-4))
            if np.isfinite(sol.fun) and sol.fun < best[0]:
                gb, wb, tb = (math.exp(v) for v in sol.x)
                best = (float(sol.fun), (gb, wb, tb))
        except Exception:
            pass

    p_pred = _fetkovich_march(gb, wb, tb, t, gp_scf, wp_bbl, pvt, pi, bgi,
                              bw, two_phase)
    # Replay the influx history at the fitted parameters.
    we_hist, we = np.zeros(len(t)), 0.0
    for i in range(1, len(t)):
        dt = max(float(t[i] - t[i - 1]), 0.0)
        decay = 1.0 - math.exp(-dt / tb) if tb > 0 else 1.0
        p_avg = 0.5 * (p_pred[i - 1] + p_pred[i])
        we = max(we + (wb / pi) * (pi * (1.0 - we / wb) - p_avg) * decay, 0.0)
        we_hist[i] = we

    # -- the locus -------------------------------------------------------
    # Profile along G: for each candidate gas in place, re-optimise the aquifer
    # and record the best fit it can manage. That traces the actual valley in
    # the objective, which a coarse grid cannot - and the valley, not the single
    # best triplet, is the honest answer. A big aquifer with a small J and a
    # small one with a big J bend the same pressure history; only late aquifer
    # depletion tells them apart, and a finite record rarely reaches it.
    best_rms = float(best[0])
    thresh = max(best_rms * locus_tolerance, best_rms + 0.05)
    prof = []
    for gv in np.geomspace(max(g_lo, 0.55 * gb), 2.2 * gb, 13):
        def obj_g(x, _g=gv):
            return rms(_g, math.exp(x[0]), math.exp(x[1]))
        try:
            sol = optimize.minimize(obj_g, np.log([wb, tb]), method="Nelder-Mead",
                                    options=dict(maxiter=160, xatol=1e-2,
                                                 fatol=1e-3))
            e, wv, tv = float(sol.fun), math.exp(sol.x[0]), math.exp(sol.x[1])
        except Exception:
            continue
        if np.isfinite(e):
            prof.append((gv / 1.0e6, wv / 1.0e6, tv, wv / (pi * tv), e))

    loc = pd.DataFrame(prof, columns=["G_mmscf", "Wei_mmbbl", "tau_days",
                                      "J_bbl_d_psi", "rms_pct"])
    inside = loc[loc["rms_pct"] <= thresh] if len(loc) else loc
    # Interpolate where the profile actually crosses the threshold rather than
    # quantising the range to whichever G values happened to be sampled - with
    # a tight fit that otherwise collapses to a single point and reports a
    # range of zero width.
    g_range = (gb / 1.0e6, gb / 1.0e6)
    if len(loc) >= 2:
        gx = loc["G_mmscf"].to_numpy()
        ex = loc["rms_pct"].to_numpy()
        i_min = int(np.argmin(ex))
        lo_g = gx[i_min]
        for i in range(i_min, 0, -1):
            if ex[i - 1] > thresh:
                frac = ((thresh - ex[i]) / (ex[i - 1] - ex[i])
                        if ex[i - 1] != ex[i] else 0.0)
                lo_g = gx[i] + frac * (gx[i - 1] - gx[i])
                break
            lo_g = gx[i - 1]
        hi_g = gx[i_min]
        for i in range(i_min, len(gx) - 1):
            if ex[i + 1] > thresh:
                frac = ((thresh - ex[i]) / (ex[i + 1] - ex[i])
                        if ex[i + 1] != ex[i] else 0.0)
                hi_g = gx[i] + frac * (gx[i + 1] - gx[i])
                break
            hi_g = gx[i + 1]
        g_range = (float(min(lo_g, hi_g)), float(max(lo_g, hi_g)))

    # We against the reservoir volume it is competing with, so "negligible"
    # has a denominator instead of being a judgement about a raw barrel count.
    hcpv_res_bbl = gb * bgi
    we_frac = float(we_hist[-1] / hcpv_res_bbl) if hcpv_res_bbl > 0 else np.nan

    return {
        "G_mmscf": gb / 1.0e6,
        "Wei_mmbbl": wb / 1.0e6,
        "tau_days": tb,
        "J_bbl_d_psi": wb / (pi * tb),
        "We_mmbbl": we_hist[-1] / 1.0e6,
        "rms_pct": float(best[0]),
        "p_initial": pi,
        "t_days": t,
        "p_observed": p,
        "p_predicted": p_pred,
        "we_bbl": we_hist,
        "gp_mmscf": g,
        "locus": loc.reset_index(drop=True),
        "locus_within": inside.reset_index(drop=True),
        "rms_threshold": thresh,
        "we_frac_hcpv": we_frac,
        "g_range_mmscf": g_range,
    }


def flowing_material_balance(t_days: np.ndarray,
                             q_ws: np.ndarray,
                             gp_mmscf: np.ndarray,
                             p_wf: np.ndarray,
                             pvt: PVT,
                             p_init: Optional[float] = None,
                             g_max_multiple: float = 30.0,
                             n_grid: int = 60,
                             min_r2: float = 0.60) -> Dict[str, float]:
    """Contacted gas in place from flowing data alone (Mattar-style FMB).

    Under boundary-dominated flow at (near) constant flowing pressure,

        q / [m(p_avg) - m(p_wf)]  =  (1/b_pss) * (1 - Gp/G)

    a straight line in Gp whose x-intercept is G. p_avg comes from the material
    balance, which itself needs G, so G is the fixed point of the map
    G -> x-intercept(G). That fixed point is found by bracketing and bisection
    rather than by damped substitution, which oscillates.

    **Read the answer with care on a condensate well.** Below the dew point the
    condensate bank steadily degrades near-wellbore mobility, so the normalised
    rate falls faster than depletion alone would explain, the line is too steep
    and the x-intercept lands *below* the true gas in place. FMB here estimates
    effective, currently-contacted gas under the prevailing skin - a useful
    lower bound and a good cross-check on the decline, not a replacement for a
    two-phase p/z material balance. `bias_warning` in the result flags this.
    """
    t = np.asarray(t_days, float)
    q = np.asarray(q_ws, float)
    g = np.asarray(gp_mmscf, float)
    pwf = np.asarray(p_wf, float)
    ok = (np.isfinite(t) & np.isfinite(q) & np.isfinite(g)
          & np.isfinite(pwf) & (q > 0))
    t, q, g, pwf = t[ok], q[ok], g[ok], pwf[ok]
    if len(t) < 6:
        raise ValueError("Need at least 6 points with flowing pressure for FMB.")

    pi = float(p_init if p_init is not None else (pvt.p_init or pwf.max() * 1.3))
    pz_i = pi / float(pvt.z_two_phase(np.array([pi]))[0])
    m_wf = pvt.m(pwf)
    g_lo = max(1.02 * float(g[-1]), 1.0)
    g_hi = g_max_multiple * max(float(g[-1]), 1.0)

    def x_intercept(G: float):
        """Regress normalised rate on cumulative for an assumed G."""
        p_avg = pvt.pressure_from_pz(
            np.array(pz_i * (1.0 - np.clip(g / G, 0.0, 0.999))))
        # An assumed G barely above the cumulative drives the implied average
        # pressure to the abandonment floor, which produces a beautifully
        # straight but entirely meaningless line. Reject those outright.
        if np.min(p_avg) <= np.max(pwf) + 100.0:
            return np.nan, None
        dm = pvt.m(p_avg) - m_wf
        valid = dm > 0
        if valid.sum() < 4:
            return np.nan, None
        res = stats.linregress(g[valid], q[valid] / dm[valid])
        if (res.slope >= 0 or not np.isfinite(res.intercept)
                or res.rvalue ** 2 < min_r2):
            # A line this poor carries no information about the x-intercept,
            # so it must not be allowed to anchor the fixed-point search.
            return np.nan, res
        return -res.intercept / res.slope, res

    grid = np.geomspace(g_lo, g_hi, n_grid)
    vals = np.array([x_intercept(G)[0] for G in grid])
    resid = vals - grid
    finite = np.isfinite(resid)
    if finite.sum() < 2:
        raise ValueError(
            "FMB: no assumed OGIP produces a declining normalised-rate line "
            f"with R2 >= {min_r2:.2f}. The window is probably not in "
            "boundary-dominated flow at constant flowing pressure.")

    converged = False
    G_star = float(grid[finite][np.argmin(np.abs(resid[finite]))])
    sign_change = (finite[:-1] & finite[1:]
                   & (np.sign(resid[:-1]) != np.sign(resid[1:])))
    idx = np.where(sign_change)[0]
    if idx.size:
        a, b_ = float(grid[idx[0]]), float(grid[idx[0] + 1])
        try:
            G_star = float(optimize.brentq(
                lambda G: x_intercept(G)[0] - G, a, b_, xtol=1e-3, maxiter=200))
            converged = True
        except Exception:
            pass

    _, res = x_intercept(G_star)
    if res is None or res.rvalue ** 2 < min_r2:
        converged = False
    p_avg = pvt.pressure_from_pz(
        np.array(pz_i * (1.0 - np.clip(g / G_star, 0.0, 0.999))))
    below_dew = bool(pvt.p_dew and np.nanmin(p_avg) < pvt.p_dew)

    return {
        "ogip_contacted_mmscf": float(G_star),
        "converged": bool(converged),
        "r2": float(res.rvalue ** 2) if res is not None else float("nan"),
        "b_pss": (float(1.0 / res.intercept)
                  if res is not None and res.intercept else float("nan")),
        "p_avg_last_psia": float(p_avg[-1]),
        "bias_warning": float(below_dew),
    }


# ==============================================================================
# SECTION 7 -- FORECASTING, PRODUCTS, UNCERTAINTY
# ==============================================================================

@dataclass
class ProductSplit:
    """Surface processing assumptions that turn wellstream gas into sales."""
    inert_fraction: float = 0.0        # CO2/N2/H2S removed at the plant
    fuel_flare_fraction: float = 0.02  # of separator gas
    ngl_yield_gal_per_mscf: float = 0.0   # plant liquids recovered
    ngl_shrinkage_fraction: float = 0.0   # volume shrinkage from NGL extraction

    def apply(self, q_ws: np.ndarray, q_cond: np.ndarray,
              v_eq: float) -> Dict[str, np.ndarray]:
        q_ws = np.asarray(q_ws, float)
        q_cond = np.asarray(q_cond, float)
        q_sep = np.maximum(q_ws - q_cond * v_eq / SCF_PER_MSCF, 0.0)
        q_sales = (q_sep * (1.0 - self.inert_fraction)
                   * (1.0 - self.fuel_flare_fraction)
                   * (1.0 - self.ngl_shrinkage_fraction))
        q_ngl = q_sep * self.ngl_yield_gal_per_mscf / 42.0     # STB/d
        return {"q_wellstream": q_ws, "q_sep_gas": q_sep,
                "q_sales_gas": q_sales, "q_condensate": q_cond, "q_ngl": q_ngl}


@dataclass
class Forecast:
    """A production forecast with products and cumulative volumes."""
    table: pd.DataFrame
    eur_wellstream_mmscf: float
    eur_sales_gas_mmscf: float
    eur_condensate_mstb: float
    eur_ngl_mstb: float
    economic_life_years: float
    remaining_wellstream_mmscf: float
    remaining_sales_gas_mmscf: float
    remaining_condensate_mstb: float
    remaining_ngl_mstb: float

    def summary(self) -> str:
        return "\n".join([
            f"  economic life      : {self.economic_life_years:.1f} yr "
            f"(from first production)",
            "  EUR (history + forecast)",
            f"    wellstream gas   : {self.eur_wellstream_mmscf:,.0f} MMscf",
            f"    sales gas        : {self.eur_sales_gas_mmscf:,.0f} MMscf",
            f"    condensate       : {self.eur_condensate_mstb:,.0f} Mstb",
            f"    plant NGL        : {self.eur_ngl_mstb:,.0f} Mstb",
            "  Remaining (forecast only)",
            f"    wellstream gas   : {self.remaining_wellstream_mmscf:,.0f} MMscf",
            f"    sales gas        : {self.remaining_sales_gas_mmscf:,.0f} MMscf",
            f"    condensate       : {self.remaining_condensate_mstb:,.0f} Mstb",
            f"    plant NGL        : {self.remaining_ngl_mstb:,.0f} Mstb",
        ])


def forecast_products(fit: FitResult,
                      yield_model: YieldModel,
                      pvt: PVT,
                      q_econ_mscfd: float,
                      gp_to_date_mmscf: float = 0.0,
                      np_to_date_mstb: float = 0.0,
                      sep_gas_to_date_mmscf: float = 0.0,
                      t_start_days: float = 0.0,
                      t_max_years: float = 40.0,
                      step_days: float = 30.4375,
                      products: Optional[ProductSplit] = None,
                      ogip_cap_mmscf: Optional[float] = None) -> Forecast:
    """Roll the decline forward, apply the yield model, and split into products.

    The gas forecast comes from the wellstream decline; condensate comes from
    CGR(Gp) evaluated on the forecast cumulative, so the two can never drift
    apart. `ogip_cap_mmscf` optionally truncates the forecast when cumulative
    wellstream gas reaches an independently-estimated OGIP - the cheapest way
    to stop a high-b hyperbolic producing more gas than the reservoir holds.

    EUR figures include history: pass `gp_to_date_mmscf`, `np_to_date_mstb`
    and `sep_gas_to_date_mmscf` from the production record. The processing
    split is applied to historical separator gas as well as to the forecast,
    so sales gas and NGL EURs are on the same footing as the gas EUR rather
    than being remaining-volumes-only.
    """
    products = products or ProductSplit()
    model = fit.model
    t_max = t_max_years * DAYS_PER_YEAR

    t_ab = model.time_to_rate(q_econ_mscfd, t_max=t_max)
    if not np.isfinite(t_ab):
        t_ab = t_max
    t_ab = float(min(max(t_ab, t_start_days + step_days), t_max))

    t = np.arange(t_start_days, t_ab + step_days, step_days)
    if t.size < 2:
        t = np.array([t_start_days, t_ab])
    q_ws = model.rate(t)
    cum_model = model.cum(t)                       # MMscf from model t=0
    cum_at_start = float(model.cum(np.array([t_start_days]))[0])
    gp = gp_to_date_mmscf + (cum_model - cum_at_start)

    if ogip_cap_mmscf is not None and np.isfinite(ogip_cap_mmscf):
        over = gp > ogip_cap_mmscf
        if np.any(over):
            keep = ~over
            if keep.sum() < 2:
                keep[:2] = True
            t, q_ws, gp = t[keep], q_ws[keep], gp[keep]
            t_ab = float(t[-1])

    cgr = yield_model(gp)                          # STB/MMscf
    q_cond = q_ws / MSCF_PER_MMSCF * cgr           # STB/d

    streams = products.apply(q_ws, q_cond, pvt.v_eq)

    cum_sales = np.concatenate([[0.0], cumulative_trapezoid(
        streams["q_sales_gas"], t)]) / MSCF_PER_MMSCF
    cum_cond = np.concatenate([[0.0], cumulative_trapezoid(q_cond, t)]) / 1e3
    cum_ngl = np.concatenate([[0.0], cumulative_trapezoid(
        streams["q_ngl"], t)]) / 1e3

    table = pd.DataFrame({
        "t_days": t,
        "t_years": t / DAYS_PER_YEAR,
        "q_wellstream_mscfd": q_ws,
        "q_sep_gas_mscfd": streams["q_sep_gas"],
        "q_sales_gas_mscfd": streams["q_sales_gas"],
        "cgr_stb_per_mmscf": cgr,
        "q_condensate_stbd": q_cond,
        "q_ngl_stbd": streams["q_ngl"],
        "Gp_wellstream_mmscf": gp,
        "Np_condensate_mstb": np_to_date_mstb + cum_cond,
        "cum_sales_gas_mmscf": cum_sales,
        "cum_ngl_mstb": cum_ngl,
    })

    # Apply the same processing split to the historical separator gas so the
    # sales-gas and NGL EURs include history, like the gas and condensate ones.
    sales_factor = ((1.0 - products.inert_fraction)
                    * (1.0 - products.fuel_flare_fraction)
                    * (1.0 - products.ngl_shrinkage_fraction))
    hist_sales = sep_gas_to_date_mmscf * sales_factor
    hist_ngl = sep_gas_to_date_mmscf * MSCF_PER_MMSCF \
        * products.ngl_yield_gal_per_mscf / 42.0 / 1.0e3

    return Forecast(
        table=table,
        eur_wellstream_mmscf=float(gp[-1]),
        eur_sales_gas_mmscf=float(hist_sales + cum_sales[-1]),
        eur_condensate_mstb=float(np_to_date_mstb + cum_cond[-1]),
        eur_ngl_mstb=float(hist_ngl + cum_ngl[-1]),
        economic_life_years=float(t[-1] / DAYS_PER_YEAR),
        remaining_wellstream_mmscf=float(gp[-1] - gp_to_date_mmscf),
        remaining_sales_gas_mmscf=float(cum_sales[-1]),
        remaining_condensate_mstb=float(cum_cond[-1]),
        remaining_ngl_mstb=float(cum_ngl[-1]),
    )


def monte_carlo_eur(fit: FitResult,
                    yield_model: YieldModel,
                    pvt: PVT,
                    q_econ_mscfd: float,
                    gp_to_date_mmscf: float = 0.0,
                    np_to_date_mstb: float = 0.0,
                    sep_gas_to_date_mmscf: float = 0.0,
                    products: Optional[ProductSplit] = None,
                    t_start_days: float = 0.0,
                    n_samples: int = 2000,
                    t_max_years: float = 40.0,
                    b_prior: Optional[Tuple[float, float]] = None,
                    dmin_prior_pct_yr: Optional[Tuple[float, float]] = None,
                    cgr_rel_sigma: float = 0.10,
                    ogip_cap_mmscf: Optional[float] = None,
                    ogip_cap_rel_sigma: float = 0.15,
                    max_rel_sd: float = 0.35,
                    seed: int = 11) -> pd.DataFrame:
    """Probabilistic EUR by sampling the fit covariance plus explicit priors.

    Two things drive condensate EUR uncertainty and neither is captured by the
    regression alone: the terminal decline you assume, and the yield you assume
    late in life. So on top of the parameter covariance you can impose

        b_prior             (mean, sigma) truncated to the model bounds
        dmin_prior_pct_yr   (mean, sigma) effective terminal decline, %/yr
        cgr_rel_sigma       lognormal scatter on the whole CGR curve
        ogip_cap_rel_sigma  scatter on the volumetric/material-balance cap
        max_rel_sd          caps each parameter's sampled standard deviation at
                            this fraction of its value. Regression covariances
                            on production data are routinely enormous because
                            the parameters trade off against each other; left
                            uncapped the sampler spends all its draws outside
                            the physical bounds and the EUR distribution turns
                            into noise. Set to None to use the raw covariance.

    Results are reported in the petroleum convention: P90 is the low case
    (90% chance of exceeding), P10 the high case.
    """
    rng = np.random.default_rng(seed)
    cls = type(fit.model)
    names = list(cls.param_names)
    mean = np.array([fit.params[p] for p in names], dtype=float)
    cov = np.array(fit.cov, dtype=float)
    if cov.shape != (len(names), len(names)) or not np.all(np.isfinite(cov)):
        cov = np.diag((0.10 * np.abs(mean)) ** 2)
    cov = 0.5 * (cov + cov.T)

    # Convert to correlation + sd so the sd can be capped without destroying
    # the parameter trade-offs the correlation encodes.
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    if max_rel_sd is not None:
        cap_sd = max_rel_sd * np.abs(mean)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(np.outer(sd, sd) > 0,
                            cov / np.outer(sd, sd), 0.0)
        np.fill_diagonal(corr, 1.0)
        sd = np.minimum(sd, cap_sd)
        cov = corr * np.outer(sd, sd)
    cov = 0.5 * (cov + cov.T)
    eig = np.linalg.eigvalsh(cov)
    if eig.min() < 0:
        cov = cov + np.eye(len(cov)) * (abs(eig.min()) + 1e-18)

    t_ref, q_ref = fit.t_fit, fit.q_fit
    bnds = cls.bounds(t_ref - fit.t0, q_ref)

    rows: List[Dict[str, float]] = []
    attempts = 0
    max_attempts = 40 * n_samples
    batch = max(256, n_samples // 2)

    while len(rows) < n_samples and attempts < max_attempts:
        draws = rng.multivariate_normal(mean, cov, size=batch)
        attempts += batch
        for x in draws:
            if len(rows) >= n_samples:
                break
            pars = dict(zip(names, x))

            if b_prior and "b" in pars:
                pars["b"] = float(rng.normal(*b_prior))
            if dmin_prior_pct_yr and "Dmin" in pars:
                d_eff = float(rng.normal(*dmin_prior_pct_yr)) / 100.0
                d_eff = float(np.clip(d_eff, 0.01, 0.45))
                pars["Dmin"] = ModifiedHyperbolic.annual_effective_to_nominal(d_eff)

            if any(not np.isfinite(v) or v < bnds.get(k, (-np.inf, np.inf))[0]
                   or v > bnds.get(k, (-np.inf, np.inf))[1]
                   for k, v in pars.items()):
                continue

            try:
                mdl = cls(t0=fit.t0, **pars)
                fr = FitResult(model=mdl, model_name=fit.model_name, params=pars,
                               stderr={}, cov=fit.cov, n_points=fit.n_points,
                               rmse_log=fit.rmse_log, r2=fit.r2, aic=fit.aic,
                               bic=fit.bic, t_fit=t_ref, q_fit=q_ref, t0=fit.t0)
                ym = YieldModel(
                    cgr_i=yield_model.cgr_i * float(rng.lognormal(0, cgr_rel_sigma)),
                    cgr_min=yield_model.cgr_min * float(rng.lognormal(0, cgr_rel_sigma)),
                    k=yield_model.k * float(rng.lognormal(0, 0.5 * cgr_rel_sigma)),
                    Gp_dew=yield_model.Gp_dew)
                cap = None
                if ogip_cap_mmscf is not None and np.isfinite(ogip_cap_mmscf):
                    cap = float(ogip_cap_mmscf * rng.lognormal(0, ogip_cap_rel_sigma))
                    cap = max(cap, gp_to_date_mmscf * 1.01)
                fc = forecast_products(fr, ym, pvt, q_econ_mscfd,
                                       gp_to_date_mmscf=gp_to_date_mmscf,
                                       np_to_date_mstb=np_to_date_mstb,
                                       sep_gas_to_date_mmscf=sep_gas_to_date_mmscf,
                                       t_start_days=t_start_days,
                                       t_max_years=t_max_years,
                                       products=products,
                                       ogip_cap_mmscf=cap)
            except Exception:
                continue

            if not (np.isfinite(fc.eur_wellstream_mmscf)
                    and np.isfinite(fc.eur_condensate_mstb)
                    and fc.eur_wellstream_mmscf >= gp_to_date_mmscf):
                continue

            rows.append({
                "eur_wellstream_mmscf": fc.eur_wellstream_mmscf,
                "eur_sales_gas_mmscf": fc.eur_sales_gas_mmscf,
                "eur_condensate_mstb": fc.eur_condensate_mstb,
                "life_years": fc.economic_life_years,
                **{f"p_{k}": v for k, v in pars.items()},
            })

    if len(rows) < max(20, 0.02 * n_samples):
        raise RuntimeError(
            f"Monte Carlo produced only {len(rows)} valid realisations out of "
            f"{attempts} draws; the fit covariance or the priors are "
            "inconsistent with the model bounds.")
    if len(rows) < n_samples:
        warnings.warn(f"Monte Carlo returned {len(rows)} of {n_samples} "
                      "requested realisations (bounds rejection).")
    return pd.DataFrame(rows)


def percentiles_petroleum(values: np.ndarray) -> Dict[str, float]:
    """P90/P50/P10 in the petroleum convention (P90 = low, 10th percentile)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return {"P90": float(np.percentile(v, 10)),
            "P50": float(np.percentile(v, 50)),
            "P10": float(np.percentile(v, 90)),
            "mean": float(np.mean(v)),
            "n": int(v.size)}


# ==============================================================================
# SECTION 8 -- ORCHESTRATION
# ==============================================================================

@dataclass
class WellResult:
    """Everything produced for one well, in one object."""
    well: str
    data: ProductionData
    pvt: PVT
    model_table: pd.DataFrame
    fits: Dict[str, FitResult]
    best_fit: FitResult
    yield_model: YieldModel
    forecast: Forecast
    mc: Optional[pd.DataFrame] = None
    mc_stats: Optional[Dict[str, Dict[str, float]]] = None
    matbal: Optional[MaterialBalanceResult] = None
    fmb: Optional[Dict[str, float]] = None
    ogip_choice: Optional[OGIPChoice] = None
    settings: Dict = field(default_factory=dict)

    def summary(self, stream=None) -> str:
        out = [f"{'=' * 78}",
               f" WELL {self.well}",
               f"{'=' * 78}",
               "",
               "-- Data QC " + "-" * 66,
               self.data.qc.summary(),
               "",
               "-- Fluid " + "-" * 68,
               "\n".join(f"  {k:<28}: {v}" for k, v in
                         self.pvt.describe().items()),
               "",
               "-- Model ranking (by AIC, lower is better) " + "-" * 35,
               self.model_table.to_string(index=False) if not self.model_table.empty
               else "  (none)",
               "",
               "-- Selected decline fit " + "-" * 54,
               self.best_fit.summary(),
               "",
               "-- Condensate yield " + "-" * 58,
               self.yield_model.summary(),
               ""]
        if self.matbal is not None:
            out += ["-- Material balance " + "-" * 58, self.matbal.summary(), ""]
        if self.fmb is not None:
            fmb_lines = [
                f"  contacted gas in place  : "
                f"{self.fmb['ogip_contacted_mmscf']:,.0f} MMscf",
                f"  converged               : {bool(self.fmb['converged'])}",
                f"  R2 of the FMB line      : {self.fmb['r2']:.4f}",
                f"  p_avg at last point     : {self.fmb['p_avg_last_psia']:,.0f} psia",
            ]
            if self.fmb.get("bias_warning"):
                fmb_lines.append(
                    "  NOTE: the reservoir is below the dew point over part of "
                    "this window, so\n        condensate banking biases this "
                    "figure LOW. Treat it as a lower bound.")
            out += ["-- Flowing material balance " + "-" * 50,
                    "\n".join(fmb_lines), ""]
        if self.ogip_choice is not None:
            oc = self.ogip_choice
            lines = []
            for k, v in (oc.candidates or {}).items():
                mark = "  <-- used" if k == oc.source else ""
                lines.append(f"    {k:<16}: {v:>12,.0f} MMscf{mark}")
            body = ("\n".join(lines) + "\n" if lines else "")
            if np.isfinite(oc.value):
                body += (f"  cap applied       : {oc.value:,.0f} MMscf "
                         f"({oc.source}, +/-{100 * oc.rel_sigma:.0f} % in the "
                         f"Monte Carlo)\n")
            else:
                body += "  cap applied       : none\n"
            body += f"  why               : {oc.reason}"
            out += ["-- Gas in place used for the cap " + "-" * 45, body, ""]
        out += ["-- Deterministic forecast " + "-" * 52, self.forecast.summary(), ""]
        if self.mc_stats:
            out.append("-- Probabilistic EUR (P90 = low case) " + "-" * 40)
            for key, st in self.mc_stats.items():
                out.append(f"  {key}")
                out.append(f"    P90 {st['P90']:>14,.0f} | P50 {st['P50']:>14,.0f} "
                           f"| P10 {st['P10']:>14,.0f}   (n={st['n']})")
            out.append("")
        text = "\n".join(out)
        print(text, file=stream) if stream is not None else print(text)
        return text

    def to_excel(self, path: str) -> str:
        """Write history, fits, forecast and Monte Carlo to one workbook."""
        with pd.ExcelWriter(path, engine="openpyxl") as xl:
            self.data.df.to_excel(xl, sheet_name="history", index=False)
            self.model_table.to_excel(xl, sheet_name="model_ranking", index=False)
            pd.DataFrame([{"parameter": k, "value": v,
                           "stderr": self.best_fit.stderr.get(k, np.nan)}
                          for k, v in self.best_fit.params.items()]
                         ).to_excel(xl, sheet_name="best_fit", index=False)
            pd.DataFrame([asdict(self.yield_model)]).to_excel(
                xl, sheet_name="yield_model", index=False)
            self.forecast.table.to_excel(xl, sheet_name="forecast", index=False)
            if self.mc is not None:
                self.mc.to_excel(xl, sheet_name="monte_carlo", index=False)
            if self.matbal is not None:
                pd.DataFrame({"pressure_psia": self.matbal.pressure,
                              "Gp_mmscf": self.matbal.gp,
                              "p_over_z": self.matbal.pz}).to_excel(
                    xl, sheet_name="material_balance", index=False)
        return path


def analyse_well(df: pd.DataFrame | ProductionData,
                 pvt: PVT,
                 well: str = "WELL",
                 q_econ_mscfd: float = 250.0,
                 models: Sequence[str] = ("arps", "modified_hyperbolic", "ple", "sepd"),
                 select: str = "modified_hyperbolic",
                 fit_from_bdf: bool = True,
                 fit_window_days: Optional[Tuple[float, float]] = None,
                 fixed_params: Optional[Dict[str, float]] = None,
                 terminal_decline_pct_yr: Optional[float] = 8.0,
                 products: Optional[ProductSplit] = None,
                 t_max_years: float = 40.0,
                 run_monte_carlo: bool = True,
                 n_mc: int = 1500,
                 b_prior: Optional[Tuple[float, float]] = None,
                 dmin_prior_pct_yr: Optional[Tuple[float, float]] = None,
                 use_material_balance: bool = True,
                 mb_p_initial: Optional[float] = None,
                 mb_skip_early: int = 0,
                 use_aquifer: bool = True,
                 use_fmb: bool = False,
                 apply_ogip_cap: bool = True,
                 ogip_cap_mode: str = "auto",
                 prepare_kwargs: Optional[Dict] = None,
                 verbose: bool = True) -> WellResult:
    """Run the full gas condensate DCA workflow on one well.

    Steps, in the order they must happen:
      1. QC and wellstream (gas-equivalent) conversion
      2. Flow-regime diagnosis; restrict the fit to boundary-dominated data
      3. Fit and rank candidate decline models on the wellstream rate
      4. Fit the CGR-vs-cumulative yield model
      5. Material balance (two-phase z) for an independent OGIP, used as a cap
      6. Deterministic forecast and product split
      7. Monte Carlo for P90/P50/P10

    `select` picks the model carried into the forecast: a name, or "auto" to
    take the lowest-AIC model. The default is the modified hyperbolic because
    it is the one with a defensible late-time limit.
    """
    data = (df if isinstance(df, ProductionData)
            else ProductionData.prepare(df, pvt, well=well, **(prepare_kwargs or {})))
    well = data.well

    # -- fitting window ---------------------------------------------------
    if fit_window_days is not None:
        t_lo, t_hi = fit_window_days
    elif fit_from_bdf and data.qc.fit_start_days is not None:
        t_lo, t_hi = data.qc.fit_start_days, None
    else:
        t_lo, t_hi = None, None
    t_w, q_w = data.window(t_lo, t_hi)
    if len(t_w) < 8:                                  # fall back to everything
        warnings.warn(f"[{well}] only {len(t_w)} points after the plateau; "
                      "fitting the full history instead.")
        t_w, q_w = data.t, data.q_ws
        t_lo = None

    # -- decline fits -----------------------------------------------------
    fixed = dict(fixed_params or {})
    if terminal_decline_pct_yr is not None and "Dmin" not in fixed:
        fixed_mh = dict(fixed)
        fixed_mh["Dmin"] = ModifiedHyperbolic.annual_effective_to_nominal(
            terminal_decline_pct_yr / 100.0)
    else:
        fixed_mh = fixed

    fits: Dict[str, FitResult] = {}
    rows = []
    for name in models:
        try:
            fr = fit_decline(t_w, q_w, model=name,
                             fixed=(fixed_mh if name == "modified_hyperbolic"
                                    else {k: v for k, v in fixed.items()
                                          if k in DECLINE_MODELS[name].param_names}))
        except Exception as exc:
            warnings.warn(f"[{well}] {name} failed: {exc}")
            continue
        fits[name] = fr
        rows.append({"model": name, "R2_log": fr.r2, "RMSE_log": fr.rmse_log,
                     "AIC": fr.aic, "BIC": fr.bic, "converged": fr.converged})
    if not fits:
        raise RuntimeError(f"[{well}] no decline model could be fitted.")
    table = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)

    key = table.iloc[0]["model"] if select == "auto" else select
    if key not in fits:
        key = table.iloc[0]["model"]
    best = fits[key]

    # -- yield model ------------------------------------------------------
    gp_dew = None
    if pvt.p_dew is not None and data.p_res is not None:
        pr = data.p_res
        okp = np.isfinite(pr)
        if okp.sum() >= 2 and np.nanmin(pr) < pvt.p_dew < np.nanmax(pr):
            gp_dew = float(np.interp(-pvt.p_dew, -pr[okp], data.Gp_ws[okp]))
    yield_model = fit_yield_model(data.Gp_ws, data.cgr, Gp_dew=gp_dew,
                                  fit_dewpoint_break=(gp_dew is None))

    # -- material balance -------------------------------------------------
    matbal, matbal_note = None, ""
    if not use_material_balance:
        matbal_note = "Material balance was switched off."
    else:
        # Read the surveys from the unfiltered record: a gauge is usually run
        # on a short or shut-in month, exactly the months the rate QC drops.
        surv = data.surveys
        if surv.empty:
            matbal_note = ("No reservoir pressure data. Supply a p_res column "
                           "to run the material balance.")
        elif len(surv) < 3:
            matbal_note = (f"Only {len(surv)} pressure survey(s) in the "
                           "record; at least 3 are needed to fit a p/z line.")
        else:
            try:
                wcol = ("Wp_water" if "Wp_water" in surv.columns else None)
                matbal = material_balance_pz(
                    surv["p_res"].to_numpy(float),
                    surv["Gp_ws"].to_numpy(float), pvt,
                    water_mstb=(surv[wcol].to_numpy(float) if wcol else None),
                    p_initial=mb_p_initial, skip_early=mb_skip_early)
                matbal_note = (f"{matbal.n_surveys} pressure surveys used"
                               + (f", {matbal.n_skipped} earliest dropped."
                                  if matbal.n_skipped else "."))
                if use_aquifer and len(surv) >= 4:
                    try:
                        matbal.fetkovich = fetkovich_aquifer_fit(
                            surv["t"].to_numpy(float)[mb_skip_early:],
                            surv["p_res"].to_numpy(float)[mb_skip_early:],
                            surv["Gp_ws"].to_numpy(float)[mb_skip_early:],
                            pvt, matbal.p_initial,
                            water_mstb=(surv[wcol].to_numpy(float)[mb_skip_early:]
                                        if wcol else None))
                    except Exception as exc:
                        warnings.warn(f"[{well}] aquifer fit failed: {exc}")
            except Exception as exc:
                matbal_note = f"Material balance failed: {exc}"
                warnings.warn(f"[{well}] material balance failed: {exc}")

    fmb = None
    if use_fmb and data.p_wf is not None:
        # FMB assumes boundary-dominated flow at roughly constant p_wf, so it
        # gets the same window as the decline fit - never the plateau, where
        # the rate is set by the choke and the normalised rate rises.
        m_fmb = data.t >= (t_lo if t_lo is not None else data.t[0])
        try:
            fmb = flowing_material_balance(data.t[m_fmb], data.q_ws[m_fmb],
                                           data.Gp_ws[m_fmb],
                                           data.p_wf[m_fmb], pvt)
        except Exception as exc:
            warnings.warn(f"[{well}] FMB failed: {exc}")

    # The cap is chosen, not assumed. Which of the three material-balance
    # numbers is gas in place depends on the drive, and the p/z intercept -
    # the one this used to take unconditionally - is the wrong one precisely
    # when the diagnostics are shouting loudest.
    ogip_choice = select_ogip(matbal, fmb,
                              mode=("none" if not apply_ogip_cap
                                    else ogip_cap_mode))
    ogip_cap = ogip_choice.value if np.isfinite(ogip_choice.value) else None

    # -- forecast ---------------------------------------------------------
    t_last = float(data.t[-1])
    fc = forecast_products(best, yield_model, pvt, q_econ_mscfd,
                           gp_to_date_mmscf=float(data.Gp_ws[-1]),
                           np_to_date_mstb=float(data.Np_cond[-1]),
                           sep_gas_to_date_mmscf=float(data.Gp_sep[-1]),
                           t_start_days=t_last,
                           t_max_years=t_max_years,
                           products=products,
                           ogip_cap_mmscf=ogip_cap)

    # -- uncertainty ------------------------------------------------------
    mc = mc_stats = None
    if run_monte_carlo:
        try:
            mc = monte_carlo_eur(best, yield_model, pvt, q_econ_mscfd,
                                 gp_to_date_mmscf=float(data.Gp_ws[-1]),
                                 np_to_date_mstb=float(data.Np_cond[-1]),
                                 sep_gas_to_date_mmscf=float(data.Gp_sep[-1]),
                                 products=products,
                                 t_start_days=t_last, n_samples=n_mc,
                                 t_max_years=t_max_years,
                                 b_prior=b_prior,
                                 dmin_prior_pct_yr=dmin_prior_pct_yr,
                                 ogip_cap_mmscf=ogip_cap,
                                 ogip_cap_rel_sigma=ogip_choice.rel_sigma)
            mc_stats = {
                "EUR wellstream gas (MMscf)":
                    percentiles_petroleum(mc["eur_wellstream_mmscf"]),
                "EUR sales gas (MMscf)":
                    percentiles_petroleum(mc["eur_sales_gas_mmscf"]),
                "EUR condensate (Mstb)":
                    percentiles_petroleum(mc["eur_condensate_mstb"]),
            }
        except Exception as exc:
            warnings.warn(f"[{well}] Monte Carlo failed: {exc}")

    res = WellResult(well=well, data=data, pvt=pvt, model_table=table, fits=fits,
                     best_fit=best, yield_model=yield_model, forecast=fc,
                     mc=mc, mc_stats=mc_stats, matbal=matbal, fmb=fmb,
                     ogip_choice=ogip_choice,
                     settings={"q_econ_mscfd": q_econ_mscfd,
                               "fit_window_days": (t_lo, t_hi),
                               "selected_model": key,
                               "ogip_cap_mmscf": ogip_cap,
                               "ogip_cap_source": ogip_choice.source,
                               "ogip_cap_reason": ogip_choice.reason,
                               "ogip_cap_rel_sigma": ogip_choice.rel_sigma,
                               "ogip_candidates": ogip_choice.candidates,
                               "matbal_note": matbal_note,
                               "terminal_decline_pct_yr": terminal_decline_pct_yr})
    if verbose:
        res.summary()
    return res


def analyse_field(wells: Dict[str, ProductionData] | pd.DataFrame,
                  pvt: PVT,
                  well_column: str = "well",
                  verbose: bool = False,
                  **kwargs) -> Tuple[Dict[str, WellResult], pd.DataFrame]:
    """Analyse every well and return the results plus an aggregated summary.

    Aggregation is done by summing well forecasts on a common time axis, which
    is the honest way to build a field profile: a single field-level decline
    hides well additions, workovers and liquid loading.
    """
    if isinstance(wells, pd.DataFrame):
        raw = map_columns(wells)
        groups = ({str(k): v for k, v in raw.groupby(well_column)}
                  if well_column in raw.columns else {"FIELD": raw})
        prepared = {}
        for name, grp in groups.items():
            try:
                prepared[name] = ProductionData.prepare(grp, pvt, well=name)
            except ValueError as exc:
                warnings.warn(f"Skipping {name}: {exc}")
        wells = prepared

    results: Dict[str, WellResult] = {}
    for name, pdata in wells.items():
        try:
            results[name] = analyse_well(pdata, pvt, well=name,
                                         verbose=verbose, **kwargs)
        except Exception as exc:
            warnings.warn(f"Well {name} failed: {exc}")

    rows = []
    for name, r in results.items():
        row = {
            "well": name,
            "model": r.settings["selected_model"],
            "n_points": r.best_fit.n_points,
            "R2_log": r.best_fit.r2,
            "qi_mscfd": r.best_fit.params.get("qi", np.nan),
            "b": r.best_fit.params.get("b", np.nan),
            "Di_pct_yr": 100 * (1 - math.exp(-r.best_fit.params["Di"] * DAYS_PER_YEAR))
                         if "Di" in r.best_fit.params else np.nan,
            "Gp_to_date_mmscf": float(r.data.Gp_ws[-1]),
            "Np_cond_to_date_mstb": float(r.data.Np_cond[-1]),
            "EUR_wellstream_mmscf": r.forecast.eur_wellstream_mmscf,
            "EUR_sales_gas_mmscf": r.forecast.eur_sales_gas_mmscf,
            "EUR_condensate_mstb": r.forecast.eur_condensate_mstb,
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
            for k, st in r.mc_stats.items():
                tag = k.split("(")[0].strip().replace(" ", "_")
                row[f"{tag}_P90"] = st["P90"]
                row[f"{tag}_P50"] = st["P50"]
                row[f"{tag}_P10"] = st["P10"]
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("EUR_wellstream_mmscf",
                                             ascending=False).reset_index(drop=True)
    return results, summary


def field_profile(results: Dict[str, WellResult],
                  step_days: float = 30.4375) -> pd.DataFrame:
    """Sum well forecasts onto one calendar-free time axis (days from t=0)."""
    if not results:
        return pd.DataFrame()
    t_end = max(float(r.forecast.table["t_days"].iloc[-1]) for r in results.values())
    grid = np.arange(0.0, t_end + step_days, step_days)
    gas = np.zeros_like(grid)
    cond = np.zeros_like(grid)
    sales = np.zeros_like(grid)
    for r in results.values():
        tb = r.forecast.table
        gas += np.interp(grid, tb["t_days"], tb["q_wellstream_mscfd"],
                         left=0.0, right=0.0)
        sales += np.interp(grid, tb["t_days"], tb["q_sales_gas_mscfd"],
                           left=0.0, right=0.0)
        cond += np.interp(grid, tb["t_days"], tb["q_condensate_stbd"],
                          left=0.0, right=0.0)
    return pd.DataFrame({"t_days": grid, "t_years": grid / DAYS_PER_YEAR,
                         "q_wellstream_mscfd": gas,
                         "q_sales_gas_mscfd": sales,
                         "q_condensate_stbd": cond})


# ==============================================================================
# SECTION 9 -- PLOTS
# ==============================================================================

def _style_axis(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(PALETTE["surface"])
    ax.grid(True, which="major", color=PALETTE["grid"], linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", color=PALETTE["grid"], linewidth=0.4, alpha=0.6,
            zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["axis"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=PALETTE["muted"], labelsize=8, length=3)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(PALETTE["ink2"])
    if title:
        ax.set_title(title, color=PALETTE["ink"], fontsize=10.5,
                     fontweight="600", loc="left", pad=8)
    ax.set_xlabel(xlabel, color=PALETTE["ink2"], fontsize=9)
    ax.set_ylabel(ylabel, color=PALETTE["ink2"], fontsize=9)
    return ax


def plot_diagnostics(res: WellResult, path: Optional[str] = None,
                     show: bool = False, dpi: int = 140):
    """Eight-panel diagnostic and forecast sheet for one well.

    Gas and liquid never share an axis - each measure gets its own panel, so
    nothing is distorted by a second y-scale.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = PALETTE["series"]
    d = res.data
    fc = res.forecast.table

    fig, axes = plt.subplots(4, 2, figsize=(13.5, 16.0))
    fig.patch.set_facecolor(PALETTE["surface"])
    ax = axes.ravel()

    # 1 -- rate vs time, semilog, with fit and forecast
    a = _style_axis(ax[0], "Wellstream rate history and forecast",
                    "Time on production (years)", "Gas rate (Mscf/d)")
    a.semilogy(d.t / DAYS_PER_YEAR, d.q_ws, "o", ms=4.5, color=C[0],
               mec=PALETTE["surface"], mew=0.8, label="Wellstream (measured)", zorder=3)
    a.semilogy(d.t / DAYS_PER_YEAR, d.q_gas, "o", ms=3.0, color=C[2],
               alpha=0.55, mec="none", label="Separator gas", zorder=2)
    tf = res.best_fit.t_fit
    a.semilogy(tf / DAYS_PER_YEAR, res.best_fit.predict(tf), "-", lw=2.0,
               color=C[1], label=f"{res.best_fit.model_name} fit", zorder=4)
    a.semilogy(fc["t_years"], fc["q_wellstream_mscfd"], "--", lw=2.0,
               color=C[1], alpha=0.75, label="Forecast", zorder=4)
    a.axhline(res.settings["q_econ_mscfd"], color=PALETTE["critical"], lw=1.2,
              ls=":", label="Economic limit", zorder=1)
    t_fit0 = res.settings["fit_window_days"][0]
    if t_fit0:
        a.axvspan(d.t[0] / DAYS_PER_YEAR, t_fit0 / DAYS_PER_YEAR,
                  color=PALETTE["grid"], alpha=0.55, lw=0, zorder=0,
                  label="Excluded (plateau / transient)")
    a.legend(frameon=False, fontsize=8, labelcolor=PALETTE["ink2"])

    # 2 -- rate vs cumulative
    a = _style_axis(ax[1], "Rate vs cumulative wellstream gas",
                    "Cumulative wellstream gas (MMscf)", "Gas rate (Mscf/d)")
    a.semilogy(d.Gp_ws, d.q_ws, "o", ms=4.5, color=C[0], mec=PALETTE["surface"],
               mew=0.8, label="Measured", zorder=3)
    a.semilogy(fc["Gp_wellstream_mmscf"], fc["q_wellstream_mscfd"], "-", lw=2.0,
               color=C[1], label="Forecast", zorder=4)
    if res.matbal is not None:
        if np.isfinite(res.matbal.ogip_mmscf):
            a.axvline(res.matbal.ogip_mmscf, color=C[3], lw=1.4, ls="--",
                      label="OGIP (material balance)", zorder=2)
    a.legend(frameon=False, fontsize=8, labelcolor=PALETTE["ink2"])

    # 3 -- loss-ratio diagnostic (declining period only: D is ~0 on plateau,
    #      so 1/D and its derivative are meaningless there)
    t_diag0 = t_fit0 if t_fit0 else d.t[0]
    m_diag = d.t >= t_diag0
    diag = (diagnose_b(d.t[m_diag], d.q_ws[m_diag]) if m_diag.sum() >= 6
            else diagnose_b(d.t, d.q_ws))
    a = _style_axis(ax[2], "Loss-ratio diagnostic: is b really constant?",
                    "Time on production (years)", "b  =  d(1/D)/dt")
    if not diag.empty:
        a.plot(diag["t"] / DAYS_PER_YEAR, diag["b"], "-", lw=1.6, color=C[0],
               label="b from data", zorder=3)
    b_fit = res.best_fit.params.get("b")
    if b_fit is not None:
        a.axhline(b_fit, color=C[1], lw=2.0, label=f"b fitted = {b_fit:.2f}", zorder=4)
    a.axhline(0.5, color=PALETTE["muted"], lw=1.0, ls=":",
              label="b = 0.5 (BDF gas theory)", zorder=1)
    a.set_ylim(-0.5, 2.5)
    a.legend(frameon=False, fontsize=8, labelcolor=PALETTE["ink2"])

    # 4 -- nominal decline, over the declining period and the forecast
    a = _style_axis(ax[3], "Nominal decline rate", "Time on production (years)",
                    "D (effective %/yr)")
    if not diag.empty:
        a.plot(diag["t"] / DAYS_PER_YEAR,
               100 * (1 - np.exp(-diag["D"] * DAYS_PER_YEAR)), "-", lw=1.6,
               color=C[0], label="From data", zorder=3)
    t_all = np.concatenate([d.t[m_diag], fc["t_days"].to_numpy()])
    t_all = t_all[t_all >= res.best_fit.t0]
    if t_all.size:
        a.plot(t_all / DAYS_PER_YEAR,
               100 * (1 - np.exp(-res.best_fit.model.D(t_all) * DAYS_PER_YEAR)),
               "-", lw=2.0, color=C[1], label="From model", zorder=4)
    a.set_ylim(0, 100)
    a.legend(frameon=False, fontsize=8, labelcolor=PALETTE["ink2"])

    # 5 -- CGR yield model
    a = _style_axis(ax[4], "Condensate yield vs cumulative gas",
                    "Cumulative wellstream gas (MMscf)", "CGR (STB/MMscf)")
    a.plot(d.Gp_ws, d.cgr, "o", ms=4.5, color=C[0], mec=PALETTE["surface"],
           mew=0.8, label="Measured CGR", zorder=3)
    gg = np.linspace(0, max(fc["Gp_wellstream_mmscf"].max(), d.Gp_ws.max()), 300)
    a.plot(gg, res.yield_model(gg), "-", lw=2.0, color=C[1],
           label="Yield model", zorder=4)
    if res.yield_model.Gp_dew > 0:
        a.axvline(res.yield_model.Gp_dew, color=C[3], lw=1.3, ls="--",
                  label="Dew-point break", zorder=2)
    a.legend(frameon=False, fontsize=8, labelcolor=PALETTE["ink2"])

    # 6 -- material balance
    a = _style_axis(ax[5], "Material balance: p/z vs cumulative gas",
                    "Cumulative wellstream gas (MMscf)", "p/z (psia)")
    if res.matbal is not None:
        mb = res.matbal
        a.plot(mb.gp, mb.pz, "o", ms=5, color=C[0], mec=PALETTE["surface"],
               mew=0.8, label="Two-phase z", zorder=3)
        if np.isfinite(mb.ogip_mmscf) and mb.ogip_mmscf > 0:
            xs = np.array([0.0, mb.ogip_mmscf])
            a.plot(xs, mb.pz_i * (1 - xs / mb.ogip_mmscf), "-", lw=2.0,
                   color=C[1], label=f"OGIP = {mb.ogip_mmscf:,.0f} MMscf",
                   zorder=4)
        if mb.ogip_single_phase:
            pz_sp = mb.pressure / res.pvt.z(mb.pressure)
            a.plot(mb.gp, pz_sp, "s", ms=4, color=C[4], mec="none", alpha=0.75,
                   label=f"Single-phase z ({mb.ogip_single_phase:,.0f} MMscf)",
                   zorder=2)
        a.set_ylim(bottom=0)
        a.legend(frameon=False, fontsize=8, labelcolor=PALETTE["ink2"])
    else:
        a.text(0.5, 0.5, "No reservoir pressure data", ha="center", va="center",
               transform=a.transAxes, color=PALETTE["muted"], fontsize=10)

    # 7 -- condensate forecast (its own axis; never share with gas)
    a = _style_axis(ax[6], "Condensate rate history and forecast",
                    "Time on production (years)", "Condensate rate (STB/d)")
    a.plot(d.t / DAYS_PER_YEAR, d.q_cond, "o", ms=4.5, color=C[2],
           mec=PALETTE["surface"], mew=0.8, label="Measured", zorder=3)
    a.plot(fc["t_years"], fc["q_condensate_stbd"], "-", lw=2.0, color=C[1],
           label="Forecast (gas x CGR model)", zorder=4)
    a.legend(frameon=False, fontsize=8, labelcolor=PALETTE["ink2"])

    # 8 -- probabilistic EUR
    a = _style_axis(ax[7], "Probabilistic EUR (wellstream gas)",
                    "EUR (MMscf)", "Probability of exceeding")
    if res.mc is not None and len(res.mc):
        v = np.sort(res.mc["eur_wellstream_mmscf"].to_numpy())
        prob = 1.0 - np.arange(1, len(v) + 1) / len(v)
        a.plot(v, prob, "-", lw=2.0, color=C[0], label="Monte Carlo", zorder=3)
        st = res.mc_stats["EUR wellstream gas (MMscf)"]
        for lbl, col, val in (("P90", C[3], st["P90"]), ("P50", C[1], st["P50"]),
                              ("P10", C[6], st["P10"])):
            a.axvline(val, color=col, lw=1.4, ls="--",
                      label=f"{lbl} = {val:,.0f}", zorder=2)
        a.axvline(res.forecast.eur_wellstream_mmscf, color=PALETTE["ink"], lw=1.4,
                  label=f"Deterministic = {res.forecast.eur_wellstream_mmscf:,.0f}",
                  zorder=4)
        a.set_ylim(0, 1)
        a.legend(frameon=False, fontsize=8, labelcolor=PALETTE["ink2"])
    else:
        a.text(0.5, 0.5, "Monte Carlo not run", ha="center", va="center",
               transform=a.transAxes, color=PALETTE["muted"], fontsize=10)

    fig.suptitle(f"Gas condensate decline curve analysis  -  {res.well}",
                 color=PALETTE["ink"], fontsize=14, fontweight="600",
                 x=0.012, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    if path:
        fig.savefig(path, dpi=dpi, facecolor=PALETTE["surface"],
                    bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


# ==============================================================================
# SECTION 10 -- SYNTHETIC DATA GENERATOR
# ==============================================================================

class _TankModel:
    """Single-tank gas condensate reservoir: deliverability + material balance.

    Minimal but physically coherent, so synthetic data can be used to
    *validate* the analysis rather than merely exercise it:

      deliverability   q = J * bank(p) * [m(p_avg) - m(p_wf)]  (pseudo-steady)
      material balance p/z_2ph = (p/z_2ph)_i * (1 - Gp/G)
      condensate bank  a mobility multiplier that falls below the dew point,
                       reproducing the productivity loss from liquid dropout
                       without changing the gas in place
      yield            CGR flat above the dew point, decaying to a floor below

    The plateau-then-decline shape this produces is what a real condensate
    well looks like, and the decline exponent emerges from the physics instead
    of being imposed - which is the point of testing against it.
    """

    def __init__(self, pvt: PVT, ogip_mmscf: float, p_init: float, p_wf: float,
                 q_plateau_mscfd: float, plateau_target_frac: float = 0.55,
                 cgr_i: float = 78.0, cgr_min: float = 22.0,
                 bank_min_factor: float = 0.55,
                 bank_pressure_scale: float = 900.0):
        self.pvt = pvt
        self.G = float(ogip_mmscf)
        self.p_wf = float(p_wf)
        self.q_plateau = float(q_plateau_mscfd)
        self.cgr_i, self.cgr_min = float(cgr_i), float(cgr_min)
        self.bank_min = float(bank_min_factor)
        self.bank_scale = float(bank_pressure_scale)
        self.p_dew = float(pvt.p_dew or 0.0)
        self.pz_i = float(p_init) / float(pvt.z_two_phase(np.array([p_init]))[0])
        self.m_wf = float(pvt.m(np.array([self.p_wf]))[0])
        p_target = float(pvt.pressure_from_pz(
            np.array([self.pz_i * (1.0 - plateau_target_frac)]))[0])
        dm_target = float(pvt.m(np.array([p_target]))[0]) - self.m_wf
        self.J = self.q_plateau / max(self.bank(p_target) * dm_target, 1e-9)

    def bank(self, p: float) -> float:
        if p >= self.p_dew:
            return 1.0
        return self.bank_min + (1.0 - self.bank_min) * math.exp(
            -(self.p_dew - p) / self.bank_scale)

    def pressure(self, gp_mmscf: float) -> float:
        frac = min(max(gp_mmscf / self.G, 0.0), 0.995)
        return float(self.pvt.pressure_from_pz(
            np.array([self.pz_i * (1.0 - frac)]))[0])

    def rate(self, p_avg: float) -> float:
        if p_avg <= self.p_wf + 25.0:
            return 0.0
        dm = float(self.pvt.m(np.array([p_avg]))[0]) - self.m_wf
        return min(self.J * self.bank(p_avg) * dm, self.q_plateau)

    def cgr(self, p_avg: float) -> float:
        if p_avg >= self.p_dew:
            return self.cgr_i
        return self.cgr_min + (self.cgr_i - self.cgr_min) * math.exp(
            -(self.p_dew - p_avg) / max(0.45 * self.p_dew, 1.0))


def simulate_tank(pvt: PVT, ogip_mmscf: float = 62000.0, p_init: float = 6400.0,
                  p_wf: float = 1400.0, q_plateau_mscfd: float = 40000.0,
                  plateau_target_frac: float = 0.55, cgr_i: float = 78.0,
                  cgr_min: float = 22.0, bank_min_factor: float = 0.55,
                  n_days: float = 4000.0, dt_days: float = 5.0) -> pd.DataFrame:
    """Run the tank model on a uniform time step. Returns the true solution."""
    tank = _TankModel(pvt, ogip_mmscf, p_init, p_wf, q_plateau_mscfd,
                      plateau_target_frac, cgr_i, cgr_min, bank_min_factor)
    t, gp, rows = 0.0, 0.0, []
    while t < n_days:
        p = tank.pressure(gp)
        q = tank.rate(p)
        if q <= 0:
            break
        rows.append({"t": t, "q_ws": q, "p_avg": p, "cgr": tank.cgr(p), "Gp": gp})
        gp += q * dt_days / MSCF_PER_MMSCF
        t += dt_days
    return pd.DataFrame(rows)


def make_synthetic_well(pvt: PVT,
                        well: str = "GC-1",
                        ogip_mmscf: float = 62000.0,
                        q_plateau_mscfd: float = 40000.0,
                        plateau_target_frac: float = 0.55,
                        cgr_i: float = 78.0,
                        cgr_min: float = 22.0,
                        p_init: float = 6400.0,
                        p_wf: float = 1400.0,
                        bank_min_factor: float = 0.55,
                        n_months: int = 84,
                        start: str = "2018-01-01",
                        noise_frac: float = 0.06,
                        downtime_prob: float = 0.10,
                        pressure_survey_every: int = 6,
                        stop_at_rate_mscfd: Optional[float] = None,
                        seed: int = 42) -> pd.DataFrame:
    """A realistic monthly production history for one gas condensate well.

    The tank model is stepped month by month with the operational noise and
    downtime applied *inside* the loop, so the reported rates, the cumulative
    they imply, and the reported reservoir pressures are all mutually
    consistent - which is what makes the material balance and FMB checks
    meaningful tests rather than circular ones. On top of that go the things
    that make real data awkward: partial-month downtime, a couple of
    post-workover spikes, infrequent static pressure surveys, and rising water.

    `stop_at_rate_mscfd` truncates the history once the rate falls below that
    value, leaving the well with genuine remaining life. Without it the
    generated well is produced to its economic limit, the forecast has nothing
    left to forecast, and any uncertainty analysis collapses to a point.
    """
    rng = np.random.default_rng(seed)
    tank = _TankModel(pvt, ogip_mmscf, p_init, p_wf, q_plateau_mscfd,
                      plateau_target_frac, cgr_i, cgr_min, bank_min_factor)

    dates = pd.date_range(start=start, periods=n_months, freq="MS")
    period = np.r_[31.0, np.diff((dates - dates[0]).days.to_numpy(float))]

    # Draw the operational pattern up front, then let the reservoir respond.
    noise = rng.lognormal(0.0, noise_frac, size=n_months)
    uptime = np.ones(n_months)
    down = rng.random(n_months) < downtime_prob
    uptime[down] = rng.uniform(0.25, 0.85, size=int(down.sum()))
    if n_months > 20:
        for idx in rng.choice(np.arange(6, n_months - 6), size=2, replace=False):
            noise[idx] *= rng.uniform(1.35, 1.7)        # post-workover uplift
            uptime[idx] = rng.uniform(0.3, 0.6)

    n_sub = 6
    gp = 0.0
    recs = []
    for i in range(n_months):
        p_month = tank.pressure(gp)
        if tank.rate(p_month) <= 0:
            break
        dt = period[i] * uptime[i] / n_sub
        qs, cgrs = [], []
        for _ in range(n_sub):
            p = tank.pressure(gp)
            q = tank.rate(p) * noise[i]
            if q <= 0:
                break
            qs.append(q)
            cgrs.append(tank.cgr(p))
            gp += q * dt / MSCF_PER_MMSCF
        if not qs:
            break
        q_ws = float(np.mean(qs))                        # stream-day rate
        if stop_at_rate_mscfd is not None and q_ws < stop_at_rate_mscfd:
            break
        cgr = float(np.mean(cgrs)) * float(rng.lognormal(0.0, 0.05))
        q_cond = q_ws / MSCF_PER_MMSCF * cgr
        q_sep = max(q_ws - q_cond * pvt.v_eq / SCF_PER_MSCF, 1.0)
        recs.append({
            "date": dates[i],
            "well": well,
            "days_on": round(period[i] * uptime[i], 1),
            "q_gas": round(q_sep, 1),                    # Mscf/d separator gas
            "q_cond": round(q_cond, 2),                  # STB/d
            "p_wf": round(min(p_wf * float(rng.lognormal(0.0, 0.02)),
                              p_month - 100.0), 0),
            "p_res": (round(p_month, 0)
                      if i % max(1, pressure_survey_every) == 0 else np.nan),
            "_gp_true": gp,
        })

    if len(recs) < 12:
        raise ValueError(f"[{well}] simulated life too short "
                         f"({len(recs)} months); check the tank inputs.")

    df = pd.DataFrame(recs)
    df["q_water"] = np.round(np.maximum(
        5.0 + 4.0e-5 * df["_gp_true"].to_numpy() * MSCF_PER_MMSCF / 30.0
        * rng.lognormal(0, 0.15, len(df)), 0.0), 1)
    df = df.drop(columns=["_gp_true"])
    return df[["date", "well", "days_on", "q_gas", "q_cond", "q_water",
               "p_wf", "p_res"]]


def make_synthetic_field(pvt: PVT, n_wells: int = 4, seed: int = 5,
                         **kwargs) -> pd.DataFrame:
    """A small multi-well synthetic field with well-to-well variability."""
    rng = np.random.default_rng(seed)
    frames = []
    truth = []
    for i in range(n_wells):
        ogip = float(rng.uniform(38000, 90000))
        frames.append(make_synthetic_well(
            pvt,
            well=f"GC-{i + 1}",
            ogip_mmscf=ogip,
            q_plateau_mscfd=float(rng.uniform(22000, 48000)),
            plateau_target_frac=float(rng.uniform(0.35, 0.60)),
            cgr_i=float(rng.uniform(62, 96)),
            cgr_min=float(rng.uniform(18, 32)),
            bank_min_factor=float(rng.uniform(0.45, 0.75)),
            stop_at_rate_mscfd=float(rng.uniform(3500, 6500)),
            n_months=int(rng.integers(64, 132)),
            start=f"20{16 + i}-0{int(rng.integers(1, 9))}-01",
            seed=seed + 17 * i,
            **kwargs))
        truth.append({"well": f"GC-{i + 1}", "true_ogip_mmscf": ogip})
    out = pd.concat(frames, ignore_index=True)
    # Kept as plain JSON-serialisable records: a DataFrame in .attrs breaks
    # Arrow serialisation wherever the frame is displayed or cached.
    out.attrs["truth"] = truth
    return out


# ==============================================================================
# SECTION 11 -- SELF TESTS
# ==============================================================================

def run_self_tests(verbose: bool = True) -> bool:
    """Internal consistency and parameter-recovery checks.

    These are not a substitute for benchmarking against your own field data,
    but they catch the errors that silently corrupt a DCA: a cumulative that
    does not integrate the rate, a model that is discontinuous at its switch
    point, a material balance that cannot recover a known OGIP.
    """
    results: List[Tuple[str, bool, str]] = []

    def check(name: str, cond: bool, detail: str = ""):
        results.append((name, bool(cond), detail))

    # 1 -- DAK z satisfies its own defining relation
    tpc, ppc = sutton_pseudocriticals(0.75, y_n2=0.02, y_co2=0.04)
    ok_all = True
    for ppr in (0.5, 2.0, 5.0, 10.0):
        for tpr in (1.1, 1.5, 2.0):
            z = _z_scalar(ppr, tpr)
            rho_r = 0.27 * ppr / (z * tpr)
            ok_all &= abs(_dak_z_from_rhor(rho_r, tpr) - z) < 1e-8
    check("DAK z-factor satisfies its implicit equation", ok_all)
    check("z -> 1 at low pressure", abs(_z_scalar(0.01, 1.5) - 1.0) < 0.02)
    check("z in a physical range at high pressure",
          0.8 < _z_scalar(10.0, 1.5) < 1.6, f"z={_z_scalar(10.0, 1.5):.3f}")

    # 2 -- pseudo-pressure is strictly increasing
    pvt = PVT(gas_gravity=0.72, temperature_F=248, condensate_api=52.0,
              p_dew=5100.0, p_init=6400.0, y_n2=0.01, y_co2=0.03,
              initial_cgr=78.0)
    pp = pvt.m(np.linspace(100, 10000, 200))
    check("pseudo-pressure m(p) is strictly increasing",
          bool(np.all(np.diff(pp) > 0)))

    # 3 -- Arps cumulative equals the integral of the rate
    for b in (0.0, 0.3, 0.5, 1.0, 1.4):
        a = Arps(qi=30000.0, Di=0.0015, b=b)
        t = np.linspace(0, 3650, 200000)
        num = trapezoid(a.rate(t), t) / MSCF_PER_MMSCF
        ana = float(a.cum(np.array([3650.0]))[0])
        ok = abs(num - ana) / max(ana, 1e-9) < 2e-5
        ok_all = ok if b == 0.0 else ok_all and ok
    check("Arps analytic cumulative matches numeric integral", ok_all)

    # 4 -- modified hyperbolic is continuous in rate and cumulative
    mh = ModifiedHyperbolic(qi=30000.0, Di=0.0018, b=1.1,
                            Dmin=ModifiedHyperbolic.annual_effective_to_nominal(0.08))
    ts = mh.t_switch
    q_lo = mh.rate(np.array([ts - 1e-4]))[0]
    q_hi = mh.rate(np.array([ts + 1e-4]))[0]
    c_lo = mh.cum(np.array([ts - 1e-4]))[0]
    c_hi = mh.cum(np.array([ts + 1e-4]))[0]
    dq = abs(q_hi - q_lo) / q_lo
    dc = abs(c_hi - c_lo) / c_lo
    check("modified hyperbolic continuous at the switch",
          dq < 1e-6 and dc < 1e-6, f"rel dq={dq:.2e}, rel dcum={dc:.2e}")
    t = np.linspace(0, 12000, 300000)
    num = trapezoid(mh.rate(t), t) / MSCF_PER_MMSCF
    ana = float(mh.cum(np.array([12000.0]))[0])
    check("modified hyperbolic cumulative matches integral",
          abs(num - ana) / ana < 1e-4, f"{num:.1f} vs {ana:.1f}")

    # 5 -- SEPD analytic cumulative
    se = StretchedExponential(q0=25000.0, tau=900.0, n=0.45)
    t = np.linspace(0, 20000, 400000)
    num = trapezoid(se.rate(t), t) / MSCF_PER_MMSCF
    ana = float(se.cum(np.array([20000.0]))[0])
    check("SEPD analytic cumulative matches integral",
          abs(num - ana) / ana < 1e-4, f"{num:.1f} vs {ana:.1f}")

    # 6 -- Duong satisfies q/Gp = a*t^-m
    du = Duong(q1=20000.0, a=0.8, m=1.15)
    t = np.array([0.0, 50.0, 500.0, 3000.0])
    tau = t + 1.0                       # Duong's day-1 reference
    ratio = du.rate(t) / (du.gp_total(t) * MSCF_PER_MMSCF)
    check("Duong satisfies q/Gp = a*t^-m",
          bool(np.allclose(ratio, du.a * tau ** (-du.m), rtol=1e-8)))
    tg = np.linspace(0, 4000, 200000)
    num = trapezoid(du.rate(tg), tg) / MSCF_PER_MMSCF
    check("Duong incremental cumulative matches integral",
          abs(num - float(du.cum(np.array([4000.0]))[0])) / num < 1e-4)

    # 7 -- decline parameter recovery on clean data
    truth = ModifiedHyperbolic(qi=40000.0, Di=0.0017, b=0.65,
                               Dmin=ModifiedHyperbolic.annual_effective_to_nominal(0.07))
    t = np.arange(15.0, 2600.0, 30.4375)
    q = truth.rate(t)
    fr = fit_decline(t, q, "modified_hyperbolic", t0=0.0,
                     fixed={"Dmin": truth.Dmin})
    err_b = abs(fr.params["b"] - 0.65)
    err_qi = abs(fr.params["qi"] / 40000.0 - 1)
    check("decline parameters recovered from clean data",
          err_b < 0.03 and err_qi < 0.02,
          f"b err={err_b:.4f}, qi err={100 * err_qi:.2f}%")

    # 7b -- fitting a late window with a shifted origin recovers the same curve
    late = t > 900.0
    fr_late = fit_decline(t[late], q[late], "modified_hyperbolic",
                          fixed={"Dmin": truth.Dmin})
    t_chk = np.linspace(1000.0, 5000.0, 50)
    rel = np.max(np.abs(fr_late.model.rate(t_chk) / truth.rate(t_chk) - 1))
    check("late-window fit with shifted origin reproduces the curve",
          rel < 0.02 and not fr_late.at_bounds,
          f"max rel err={100 * rel:.2f}%, at_bounds={fr_late.at_bounds}")

    # 8 -- decline parameter recovery with 8% noise
    rng = np.random.default_rng(1)
    qn = q * rng.lognormal(0, 0.08, size=q.shape)
    frn = fit_decline(t, qn, "modified_hyperbolic", t0=0.0,
                      fixed={"Dmin": truth.Dmin})
    eur_t, _ = truth.eur(300.0)
    eur_f, _ = frn.model.eur(300.0)
    check("EUR within 10% of truth with 8% noise",
          abs(eur_f / eur_t - 1) < 0.10,
          f"{eur_f:,.0f} vs {eur_t:,.0f} MMscf")

    # 9 -- material balance recovers a known OGIP, on a CVD-defined fluid
    cvd_t = CVDTable(
        pressure=np.array([5100, 4500, 3800, 3100, 2400, 1800, 1200, 700]),
        cum_produced_molfrac=np.array([0.0, 0.078, 0.176, 0.283, 0.402,
                                       0.515, 0.641, 0.757]))
    pvt_cvd = PVT(gas_gravity=0.72, temperature_F=248, condensate_api=52.0,
                  p_dew=5100.0, p_init=6400.0, y_n2=0.012, y_co2=0.031,
                  cvd=cvd_t, initial_cgr=78.0)

    G = 55000.0
    gp = np.linspace(0, 0.70 * G, 16)
    pi = 6400.0
    pz_i = pi / float(pvt_cvd.z_two_phase(np.array([pi]))[0])
    p_syn = pvt_cvd.pressure_from_pz(pz_i * (1 - gp / G))
    mb = material_balance_pz(p_syn, gp, pvt_cvd)
    check("material balance recovers known OGIP",
          abs(mb.ogip_mmscf / G - 1) < 0.02,
          f"{mb.ogip_mmscf:,.0f} vs {G:,.0f} MMscf")

    # Well below the dew point the retrograde liquid means z_2ph < z_1ph, so
    # p/z plotted with the single-phase value is too low, the trend too steep
    # and OGIP too small. Near the dew point the two can cross, which is why
    # the module measures the difference instead of assuming a rule of thumb.
    # z must be continuous at the dew point: no liquid has dropped out yet, so
    # the two-phase and single-phase values are the same number there.
    for name, pv in (("CVD table", pvt_cvd), ("Rayes correlation", pvt)):
        pd_ = float(pv.p_dew)
        above = float(pv.z_two_phase(np.array([pd_ * 1.0005]))[0])
        below = float(pv.z_two_phase(np.array([pd_ * 0.9995]))[0])
        check(f"two-phase z is continuous at the dew point ({name})",
              abs(below / above - 1.0) < 2e-3,
              f"{below:.4f} vs {above:.4f}")
    # and the p/z inverse must round-trip across it
    for pv in (pvt_cvd, pvt):
        for p_try in (0.98 * pv.p_dew, 0.7 * pv.p_dew, 0.4 * pv.p_dew):
            pz_try = p_try / float(pv.z_two_phase(np.array([p_try]))[0])
            back = float(pv.pressure_from_pz(np.array([pz_try]))[0])
            ok_all = abs(back / p_try - 1.0) < 2e-3
            if not ok_all:
                break
        if not ok_all:
            break
    check("p/z inverts back to the same pressure", ok_all,
          f"{back:,.0f} vs {p_try:,.0f} psia")

    p_deep = np.array([2400.0, 1800.0, 1200.0, 700.0])
    check("two-phase z is below single-phase z well under the dew point",
          bool(np.all(pvt_cvd.z_two_phase(p_deep) < pvt_cvd.z(p_deep)))
          and bool(np.all(pvt.z_two_phase(p_deep) < pvt.z(p_deep))),
          "checked for both the CVD table and the Rayes correlation")
    check("single-phase z understates OGIP once well depleted",
          mb.ogip_single_phase is not None and mb.ogip_single_phase < mb.ogip_mmscf,
          f"single-phase {mb.ogip_single_phase:,.0f} vs two-phase "
          f"{mb.ogip_mmscf:,.0f} MMscf "
          f"({100 * (mb.ogip_single_phase / mb.ogip_mmscf - 1):+.1f}%)"
          if mb.ogip_single_phase else "n/a")

    # 9c -- Havlena-Odeh: F/Eg IS G for a closed tank, and influx is caught
    ho = material_balance_pz(p_syn, gp, pvt_cvd, p_initial=pi)
    rel = ho.ho_table["F_over_Eg_reliable"].to_numpy()
    fe = ho.ho_table.loc[rel, "F_over_Eg_mmscf"].to_numpy()
    check("F/Eg equals G at every survey for a closed tank",
          bool(np.all(np.abs(fe / G - 1.0) < 0.01)),
          f"spread {100 * (fe.max() / fe.min() - 1):.3f} %")
    check("closed tank is diagnosed volumetric",
          ho.drive == "volumetric" and not ho.impossible,
          f"F/Eg rise {ho.ho_rise:.3f}x")
    check("the We>=0 ceiling brackets the true G on a closed tank",
          abs(ho.g_ceiling_mmscf / G - 1.0) < 0.02,
          f"ceiling {ho.g_ceiling_mmscf:,.0f} vs G {G:,.0f}")

    # Hold the pressure up artificially and the diagnosis must change.
    p_sup = p_syn + (gp / gp.max()) * 700.0
    sup = material_balance_pz(p_sup, gp, pvt_cvd, p_initial=pi)
    check("pressure support is caught by F/Eg",
          sup.drive in ("borderline", "water drive") and sup.ho_rise > 1.10,
          f"rise {sup.ho_rise:.2f}x -> {sup.drive}")
    check("support inflates the p/z intercept above the ceiling",
          sup.ogip_mmscf > sup.g_ceiling_mmscf > G * 0.95,
          f"line {sup.ogip_mmscf:,.0f} > ceiling {sup.g_ceiling_mmscf:,.0f}")

    # 9d -- Fetkovich aquifer: recover a known one, and find none when none
    pi_f = 6400.0
    zi_f = float(pvt_cvd.z_two_phase(np.array([pi_f]))[0])
    bgi_f = 0.0050346 * zi_f * pvt_cvd.T_R / pi_f
    G_t, Wei_t, tau_t = 80_000e6, 40e6, 1500.0
    t_f = np.linspace(0, 9 * 365.25, 12)
    gp_f = np.linspace(0, 0.42 * G_t, 12) / 1e6
    p_f = _fetkovich_march(G_t, Wei_t, tau_t, t_f, gp_f * 1e6, np.zeros(12),
                           pvt_cvd, pi_f, bgi_f, 1.0, True)
    fk = fetkovich_aquifer_fit(t_f, p_f, gp_f, pvt_cvd, pi_f)
    check("Fetkovich recovers a known aquifer",
          fk is not None and abs(fk["G_mmscf"] / (G_t / 1e6) - 1) < 0.02
          and fk["rms_pct"] < 0.05,
          f"G {fk['G_mmscf']:,.0f} vs {G_t / 1e6:,.0f} MMscf, "
          f"Wei {fk['Wei_mmbbl']:,.1f} vs {Wei_t / 1e6:,.1f} MMbbl, "
          f"rms {fk['rms_pct']:.3f} %")

    gpv = np.linspace(0, 0.65 * 55_000e6, 10) / 1e6
    pv = pvt_cvd.pressure_from_pz((pi_f / zi_f) * (1 - gpv * 1e6 / 55_000e6))
    fk0 = fetkovich_aquifer_fit(np.linspace(0, 8 * 365.25, 10), pv, gpv,
                                pvt_cvd, pi_f)
    check("Fetkovich finds no aquifer when there is none",
          fk0 is not None and fk0["We_mmbbl"] < 0.5
          and abs(fk0["G_mmscf"] / 55_000 - 1) < 0.02,
          f"We {fk0['We_mmbbl']:.3f} MMbbl, G {fk0['G_mmscf']:,.0f} MMscf")

    # 9c2 -- a p/z that does not decline is a result, not a crash
    # An aquifer that has caught up: the surveys start well into depletion and
    # then recover, ending ABOVE where they started while staying far below
    # p_i. Real wells do this, and it is precisely the case where the straight
    # line has nothing to offer and the ceiling has everything.
    gp_up = np.linspace(0.20 * G, 0.60 * G, 12)
    p_up = pvt_cvd.pressure_from_pz(pz_i * np.linspace(0.62, 0.70, 12))
    try:
        mb_flat = material_balance_pz(p_up, gp_up, pvt_cvd, p_initial=pi)
        flat_ok = (not mb_flat.pz_trend_ok) and bool(mb_flat.pz_note) \
            and not np.isfinite(mb_flat.ogip_mmscf) \
            and np.isfinite(mb_flat.g_ceiling_mmscf) \
            and mb_flat.drive == "water drive" and len(mb_flat.ho_table) >= 3
        detail = (f"trend_ok={mb_flat.pz_trend_ok}, OGIP="
                  f"{mb_flat.ogip_mmscf:,.0f}, drive={mb_flat.drive}, "
                  f"ceiling {mb_flat.g_ceiling_mmscf:,.0f} MMscf, "
                  f"{len(mb_flat.ho_table)} H-O rows")
    except Exception as exc:
        flat_ok, detail = False, f"raised {type(exc).__name__}: {exc}"
    check("a non-declining p/z is diagnosed, not raised", flat_ok, detail)
    check("no intercept means no cap rather than a nonsense cap",
          not np.isfinite(select_ogip(mb_flat, mode="p/z").value)
          and select_ogip(mb_flat).source in ("Fetkovich", "We>=0 ceiling"),
          f"auto falls back to {select_ogip(mb_flat).source}")

    # A nearly flat line still HAS an intercept - an enormous one, from a big
    # number over a slope near zero. It must not be allowed to cap anything.
    gp_loose = np.linspace(0.20 * G, 0.34 * G, 7)
    p_loose = pvt_cvd.pressure_from_pz(
        pz_i * (np.linspace(0.70, 0.699, 7) + np.array(
            [0, .004, -.004, .003, -.003, .002, -.002])))
    mb_loose = material_balance_pz(p_loose, gp_loose, pvt_cvd)
    rel_se = (mb_loose.ogip_stderr / mb_loose.ogip_mmscf
              if np.isfinite(mb_loose.ogip_mmscf) and mb_loose.ogip_mmscf > 0
              else float("nan"))
    check("an intercept from a near-flat line is refused as a cap",
          (not np.isfinite(mb_loose.ogip_mmscf))
          or rel_se > PZ_MAX_REL_SE
          and not np.isfinite(select_ogip(mb_loose, mode="p/z").value),
          f"OGIP {mb_loose.ogip_mmscf:,.0f} MMscf +/- {100 * rel_se:.0f} %, "
          f"cap = {select_ogip(mb_loose, mode='p/z').value}")

    # 9d1 -- the forecast cap is chosen by drive, not fixed on the p/z line
    ho_cap = material_balance_pz(p_syn, gp, pvt_cvd, p_initial=pi)
    pick_closed = select_ogip(ho_cap)
    check("a closed tank is capped on the p/z line",
          pick_closed.source == "p/z line"
          and abs(pick_closed.value / G - 1) < 0.02,
          f"{pick_closed.value:,.0f} MMscf from {pick_closed.source}")

    sup_cap = material_balance_pz(p_sup, gp, pvt_cvd, p_initial=pi)
    sup_cap.fetkovich = fetkovich_aquifer_fit(
        np.linspace(0, 10 * 365.25, len(gp)), p_sup, gp, pvt_cvd, pi)
    pick_sup = select_ogip(sup_cap)
    check("a supported tank is NOT capped on the p/z line",
          pick_sup.source != "p/z line"
          and pick_sup.value < sup_cap.ogip_mmscf,
          f"{pick_sup.value:,.0f} MMscf from {pick_sup.source}, vs the p/z "
          f"line at {sup_cap.ogip_mmscf:,.0f}")
    check("the cap never exceeds the We>=0 ceiling by more than the slack",
          all(select_ogip(m, mode=md).value
              <= m.g_ceiling_mmscf * CEILING_SLACK * 1.0001
              for m in (ho_cap, sup_cap)
              for md in ("auto", "p/z", "fetkovich", "ceiling")
              if np.isfinite(select_ogip(m, mode=md).value)
              and np.isfinite(m.g_ceiling_mmscf)),
          f"checked for the closed and supported tanks, every mode, at "
          f"{100 * (CEILING_SLACK - 1):.0f} % slack on a minimum statistic")
    check("a cap well above the ceiling is still clipped",
          select_ogip(sup_cap, mode="p/z").clipped_to_ceiling
          and select_ogip(sup_cap, mode="p/z").value
          <= sup_cap.g_ceiling_mmscf * 1.0001,
          f"p/z {sup_cap.ogip_mmscf:,.0f} clipped to "
          f"{select_ogip(sup_cap, mode='p/z').value:,.0f} MMscf")
    check("the Monte Carlo spread comes from the fit, not a fixed 15 %",
          0.0 < pick_sup.rel_sigma <= 0.60
          and abs(pick_sup.rel_sigma - 0.15) > 1e-9,
          f"+/-{100 * pick_sup.rel_sigma:.0f} % from the "
          f"{pick_sup.source} spread")
    check("switching the cap off leaves the forecast uncapped",
          not np.isfinite(select_ogip(sup_cap, mode="none").value),
          "mode='none' returns no cap")

    # 9d2 -- dates: ISO is accepted, ambiguity is refused rather than guessed
    iso = pd.Series(["2012-01-01", "2012-02-01", "2012-03-01", "2012-04-01"])
    got = parse_dates(iso)
    check("an ISO date column parses",
          bool(got.is_monotonic_increasing) and int(got.dt.day.nunique()) == 1
          and (got.max() - got.min()).days == 91,
          f"{got.iloc[0]:%Y-%m-%d} to {got.iloc[-1]:%Y-%m-%d}")
    check("ISO with a time stamp, and with slashes, both parse",
          bool(parse_dates(pd.Series(["2012-01-01 00:00:00",
                                      "2012-02-01 06:30"])).notna().all())
          and bool(parse_dates(pd.Series(["2012/01/01",
                                          "2012/02/01"])).notna().all()),
          "both accepted")
    check("dates already parsed pass straight through",
          bool(parse_dates(pd.to_datetime(iso)).equals(pd.to_datetime(iso))),
          "datetime64 input returned unchanged")

    def _refused(vals) -> Tuple[bool, str]:
        try:
            parse_dates(pd.Series(vals))
            return False, "ACCEPTED - it should not have been"
        except DateFormatError as exc:
            return True, str(exc)

    amb_ok, amb_msg = _refused(["01/05/2012", "01/06/2012", "01/07/2012"])
    check("an ambiguous slashed column is refused, not guessed",
          amb_ok and "either day-first or month-first" in amb_msg,
          amb_msg[:100] if amb_ok else amb_msg)
    # This is the case the old detector got wrong in silence: twelve months or
    # fewer of day-first data read as twelve consecutive days in January.
    short = [f"01/{m:02d}/2012" for m in range(1, 13)]
    short_ok, _ = _refused(short)
    check("a SHORT day-first history is refused rather than read as 12 days",
          short_ok, "twelve monthly rows, refused instead of misread")
    df_ok, df_msg = _refused(["13/05/2012", "14/06/2012"])
    check("a day-first column is refused but named as day-first",
          df_ok and "day-first" in df_msg, df_msg[:90])
    us_ok, us_msg = _refused(["03/15/2020", "04/15/2020"])
    check("a month-first column is refused but named as month-first",
          us_ok and "month-first" in us_msg, us_msg[:90])
    xl_ok, xl_msg = _refused([44227, 44255, 44286])
    check("Excel date serials are caught instead of becoming 1970",
          xl_ok and "Excel date serial" in xl_msg and "2021-01-31" in xl_msg,
          xl_msg[:120])
    ymd_ok, ymd_msg = _refused([20210131, 20210228])
    check("YYYYMMDD integers are caught instead of becoming 1970",
          ymd_ok and "YYYYMMDD" in ymd_msg, ymd_msg[:90])
    mn_ok, mn_msg = _refused(["Jan-2021", "Feb-2021"])
    check("month names are refused with a reason",
          mn_ok and "month names" in mn_msg, mn_msg[:90])
    bad_ok, bad_msg = _refused(["2021-13-01", "2021-02-30"])
    check("ISO-shaped impossible dates are caught",
          bad_ok and "not real dates" in bad_msg, bad_msg[:90])
    check("a blank column is empty, not an error",
          bool(parse_dates(pd.Series(["", None, "nan"])).isna().all()),
          "all NaT, no exception")

    # 9e -- initial pressure recovered when the first survey is months late
    # The surveys start at 12 % depletion, as they routinely do; nothing in
    # the data says what p_i was, and the estimator has to put it back.
    gp_late = np.linspace(0.12 * G, 0.55 * G, 7)
    p_late = pvt_cvd.pressure_from_pz(pz_i * (1 - gp_late / G))
    ip = estimate_initial_pressure(p_late, gp_late, pvt_cvd,
                                   t_days=np.linspace(400, 2600, 7))
    check("initial pressure recovered from late-starting surveys",
          ip.ok and abs(ip.p_initial / pi - 1) < 0.01,
          f"{ip.p_initial:,.0f} vs {pi:,.0f} psia, from {ip.n_used} of "
          f"{ip.n_surveys} surveys, first survey {p_late[0]:,.0f} psia")
    check("taking the first survey as p_i would have been far worse",
          abs(p_late[0] / pi - 1) > 6 * abs(ip.p_initial / pi - 1),
          f"first-survey error {100 * (p_late[0] / pi - 1):+.1f} % vs "
          f"estimate {100 * (ip.p_initial / pi - 1):+.2f} %")

    # Support on the late surveys must not drag the estimate down: the window
    # has to stop where the trend leaves the early line.
    gp_s = np.linspace(0.10 * G, 0.60 * G, 9)
    p_s = pvt_cvd.pressure_from_pz(pz_i * (1 - gp_s / G))
    p_s = p_s + np.where(gp_s > 0.32 * G,
                         (gp_s - 0.32 * G) / (0.28 * G) * 520.0, 0.0)
    ip_s = estimate_initial_pressure(p_s, gp_s, pvt_cvd)
    all_fit = stats.linregress(gp_s, p_s / pvt_cvd.z_two_phase(p_s))
    pi_all = float(pvt_cvd.pressure_from_pz(np.array([all_fit.intercept]))[0])
    check("pressure support does not drag the p_i estimate down",
          ip_s.ok and ip_s.n_used < len(gp_s)
          and abs(ip_s.p_initial / pi - 1) < abs(pi_all / pi - 1),
          f"early window {ip_s.p_initial:,.0f} ({ip_s.n_used} surveys) vs "
          f"all-survey line {pi_all:,.0f}, truth {pi:,.0f} psia")

    check("one survey is refused rather than mistaken for p_i",
          not estimate_initial_pressure(p_late[:1], gp_late[:1], pvt_cvd).ok
          and not estimate_initial_pressure(
              np.array([]), np.array([]), pvt_cvd).ok,
          "both the single-survey and the no-survey cases decline to answer")

    check("the estimate is never below a measured pressure",
          estimate_initial_pressure(
              np.array([3000.0, 2800.0, 2600.0]),
              np.array([0.0, 8000.0, 16000.0]), pvt_cvd
          ).p_initial >= 3000.0,
          "floored at the highest pressure on record")

    # A pre-production survey must survive the leading-zero filter, because it
    # is the one pressure paired with a cumulative of exactly zero.
    df_pre = make_synthetic_well(pvt_cvd, ogip_mmscf=52000.0, n_months=90,
                                 seed=12)
    df_pre = pd.concat([
        pd.DataFrame({"date": [df_pre["date"].iloc[0] - pd.Timedelta(days=1)],
                      "q_gas": [0.0], "q_cond": [0.0], "p_res": [6400.0]}),
        df_pre], ignore_index=True)
    d_pre = ProductionData.prepare(df_pre, pvt_cvd, well="PRE")
    s_pre = d_pre.surveys
    check("a pre-production survey survives to the material balance",
          len(s_pre) and abs(float(s_pre["p_res"].iloc[0]) - 6400.0) < 1e-6
          and float(s_pre["Gp_ws"].iloc[0]) == 0.0,
          f"{len(s_pre)} survey(s), first at Gp = "
          f"{float(s_pre['Gp_ws'].iloc[0]):.1f} MMscf")

    # 9b -- the tank model is recovered end to end by the material balance
    df_t = make_synthetic_well(pvt_cvd, ogip_mmscf=52000.0,
                               q_plateau_mscfd=32000.0,
                               plateau_target_frac=0.45, n_months=104, seed=77)
    d_t = ProductionData.prepare(df_t, pvt_cvd, well="TANK")
    pr = d_t.p_res
    okp = np.isfinite(pr)
    mb_t = material_balance_pz(pr[okp], d_t.Gp_ws[okp], pvt_cvd)
    check("material balance recovers the simulated tank OGIP",
          abs(mb_t.ogip_mmscf / 52000.0 - 1) < 0.05,
          f"{mb_t.ogip_mmscf:,.0f} vs 52,000 MMscf "
          f"({100 * (mb_t.ogip_mmscf / 52000.0 - 1):+.1f}%)")

    # 10 -- yield model recovery
    ym_true = YieldModel(cgr_i=80.0, cgr_min=25.0, k=1.2e-4, Gp_dew=4000.0)
    gpx = np.linspace(0, 30000, 60)
    cgrx = ym_true(gpx) * np.random.default_rng(2).lognormal(0, 0.04, 60)
    ymf = fit_yield_model(gpx, cgrx, Gp_dew=4000.0, fit_dewpoint_break=False)
    check("CGR yield model recovered",
          abs(ymf.cgr_i - 80) < 4 and abs(ymf.cgr_min - 25) < 4,
          f"cgr_i={ymf.cgr_i:.1f}, cgr_min={ymf.cgr_min:.1f}")

    # 11 -- gas equivalent sanity
    veq = gas_equivalent_of_condensate(api_to_sg(52.0),
                                       standing_condensate_mw(52.0))
    check("gas equivalent of condensate in the expected range",
          500 < veq < 1200, f"{veq:,.0f} scf/STB")

    # 12 -- end-to-end run on synthetic data
    try:
        df = make_synthetic_well(pvt, n_months=72, seed=9)
        res = analyse_well(df, pvt, well="TEST", verbose=False,
                           run_monte_carlo=True, n_mc=250)
        ok = (res.forecast.eur_wellstream_mmscf > float(res.data.Gp_ws[-1])
              and res.forecast.eur_condensate_mstb > float(res.data.Np_cond[-1])
              and res.mc_stats is not None
              and res.mc_stats["EUR wellstream gas (MMscf)"]["P90"]
              <= res.mc_stats["EUR wellstream gas (MMscf)"]["P10"])
        check("end-to-end analysis runs and is self-consistent", ok)
    except Exception as exc:
        check("end-to-end analysis runs and is self-consistent", False, str(exc))

    if verbose:
        width = 62
        print("\n" + "=" * 78)
        print(" SELF TESTS")
        print("=" * 78)
        for name, ok, detail in results:
            tag = "PASS" if ok else "FAIL"
            line = f"  [{tag}] {name:<{width}}"
            if detail:
                line += f"  {detail}"
            print(line)
        n_ok = sum(1 for _, ok, _ in results if ok)
        print(f"\n  {n_ok}/{len(results)} checks passed.\n")
    return all(ok for _, ok, _ in results)


# ==============================================================================
# SECTION 12 -- DEMO AND CLI
# ==============================================================================

def demo(outdir: str = "dca_output", n_wells: int = 4,
         make_plots: bool = True) -> Dict[str, WellResult]:
    """End-to-end worked example on a synthetic gas condensate field."""
    os.makedirs(outdir, exist_ok=True)

    # -- 1. Define the fluid ----------------------------------------------
    # A CVD table would normally come from the lab report. This one is a
    # plausible retrograde gas: cumulative wellstream moles produced against
    # pressure, plus the liquid dropout curve.
    cvd = CVDTable(
        pressure=np.array([5100, 4500, 3800, 3100, 2400, 1800, 1200, 700]),
        cum_produced_molfrac=np.array([0.000, 0.078, 0.176, 0.283, 0.402,
                                       0.515, 0.641, 0.757]),
        liquid_dropout=np.array([0.000, 0.081, 0.134, 0.152, 0.146, 0.131,
                                 0.108, 0.086]),
    )
    pvt = PVT(gas_gravity=0.72, temperature_F=248.0, condensate_api=52.0,
              p_dew=5100.0, p_init=6400.0, y_n2=0.012, y_co2=0.031,
              cvd=cvd, initial_cgr=78.0)

    print("\n" + "=" * 78)
    print(" GAS CONDENSATE DECLINE CURVE ANALYSIS  -  DEMONSTRATION")
    print("=" * 78)
    print("\nFluid definition:")
    for k, v in pvt.describe().items():
        print(f"  {k:<28}: {v}")

    # -- 2. Data -----------------------------------------------------------
    raw = make_synthetic_field(pvt, n_wells=n_wells)
    raw_path = os.path.join(outdir, "synthetic_production.csv")
    raw.to_csv(raw_path, index=False)
    print(f"\nSynthetic production written to {raw_path} "
          f"({len(raw)} rows, {raw['well'].nunique()} wells)")

    # -- 3. Analysis -------------------------------------------------------
    products = ProductSplit(inert_fraction=0.043, fuel_flare_fraction=0.02,
                            ngl_yield_gal_per_mscf=1.8,
                            ngl_shrinkage_fraction=0.03)
    results, summary = analyse_field(
        raw, pvt, q_econ_mscfd=400.0,
        select="modified_hyperbolic",
        terminal_decline_pct_yr=7.0,
        products=products,
        run_monte_carlo=True, n_mc=800,
        dmin_prior_pct_yr=(7.0, 2.0),
        use_fmb=True,
        verbose=False)

    for r in results.values():
        r.summary()

    print("=" * 78)
    print(" FIELD SUMMARY")
    print("=" * 78)
    cols = ["well", "model", "b", "Di_pct_yr", "Gp_to_date_mmscf",
            "EUR_wellstream_mmscf", "EUR_sales_gas_mmscf",
            "EUR_condensate_mstb", "OGIP_cap_mmscf",
            "OGIP_cap_source", "life_yr"]
    print(summary[[c for c in cols if c in summary.columns]]
          .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    tot = summary[["EUR_wellstream_mmscf", "EUR_sales_gas_mmscf",
                   "EUR_condensate_mstb"]].sum()
    print(f"\n  Field EUR wellstream gas : {tot['EUR_wellstream_mmscf']:,.0f} MMscf")
    print(f"  Field EUR sales gas      : {tot['EUR_sales_gas_mmscf']:,.0f} MMscf")
    print(f"  Field EUR condensate     : {tot['EUR_condensate_mstb']:,.0f} Mstb")

    profile = field_profile(results)
    profile.to_csv(os.path.join(outdir, "field_profile.csv"), index=False)
    summary.to_csv(os.path.join(outdir, "well_summary.csv"), index=False)

    for name, r in results.items():
        r.to_excel(os.path.join(outdir, f"{name}_dca.xlsx"))
        if make_plots:
            plot_diagnostics(r, path=os.path.join(outdir, f"{name}_diagnostics.png"))

    print(f"\n  Outputs written to ./{outdir}/")
    return results


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gas_condensate_dca",
        description="Decline curve analysis for gas condensate fields.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--demo", action="store_true",
                   help="run the worked example on synthetic data")
    p.add_argument("--selftest", action="store_true",
                   help="run internal consistency and recovery checks")
    p.add_argument("--input", help="production CSV/Excel file")
    p.add_argument("--outdir", default="dca_output", help="output directory")
    p.add_argument("--well-column", default="well")
    p.add_argument("--gas-gravity", type=float, default=0.70)
    p.add_argument("--temperature-f", type=float, default=220.0)
    p.add_argument("--condensate-api", type=float, default=52.0)
    p.add_argument("--condensate-mw", type=float, default=None)
    p.add_argument("--p-dew", type=float, default=None)
    p.add_argument("--p-init", type=float, default=None)
    p.add_argument("--initial-cgr", type=float, default=None,
                   help="initial CGR, STB/MMscf (used for wellstream gravity)")
    p.add_argument("--y-n2", type=float, default=0.0)
    p.add_argument("--y-co2", type=float, default=0.0)
    p.add_argument("--y-h2s", type=float, default=0.0)
    p.add_argument("--q-econ", type=float, default=250.0,
                   help="economic limit gas rate, Mscf/d")
    p.add_argument("--model", default="modified_hyperbolic",
                   choices=list(DECLINE_MODELS) + ["auto"])
    p.add_argument("--terminal-decline", type=float, default=8.0,
                   help="terminal effective decline, %%/yr")
    p.add_argument("--rate-basis", default="stream-day",
                   choices=["stream-day", "calendar-day"])
    p.add_argument("--t-max-years", type=float, default=40.0)
    p.add_argument("--n-mc", type=int, default=1500)
    p.add_argument("--no-mc", action="store_true", help="skip Monte Carlo")
    p.add_argument("--no-plots", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_cli().parse_args(argv)

    if args.selftest:
        return 0 if run_self_tests() else 1
    if args.demo or not args.input:
        if not args.input and not args.demo:
            print("No --input given; running the demonstration.\n")
        demo(outdir=args.outdir, make_plots=not args.no_plots)
        return 0

    pvt = PVT(gas_gravity=args.gas_gravity, temperature_F=args.temperature_f,
              condensate_api=args.condensate_api,
              condensate_mw=args.condensate_mw,
              p_dew=args.p_dew, p_init=args.p_init,
              y_n2=args.y_n2, y_co2=args.y_co2, y_h2s=args.y_h2s,
              initial_cgr=args.initial_cgr)

    wells = load_production(args.input, pvt, well_column=args.well_column,
                            rate_basis=args.rate_basis)
    results, summary = analyse_field(
        wells, pvt, q_econ_mscfd=args.q_econ, select=args.model,
        terminal_decline_pct_yr=args.terminal_decline,
        t_max_years=args.t_max_years,
        run_monte_carlo=not args.no_mc, n_mc=args.n_mc, verbose=True)

    os.makedirs(args.outdir, exist_ok=True)
    summary.to_csv(os.path.join(args.outdir, "well_summary.csv"), index=False)
    field_profile(results).to_csv(
        os.path.join(args.outdir, "field_profile.csv"), index=False)
    for name, r in results.items():
        r.to_excel(os.path.join(args.outdir, f"{name}_dca.xlsx"))
        if not args.no_plots:
            plot_diagnostics(r, path=os.path.join(args.outdir,
                                                  f"{name}_diagnostics.png"))
    print("\n" + summary.to_string(index=False))
    print(f"\nOutputs written to {args.outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
