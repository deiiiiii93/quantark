"""
Vanna-Volga core: the Omega market-price matrix and the simple/matrix
second-order volatility corrections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from quantark.util.exceptions import NumericalError
from quantark.util.numerical import Tolerance

from .bs_fx import GKInput, greeks_gk
from .market_conventions import (
    DeltaConvention,
    FXEnv,
    atm_strike,
    strike_for_delta,
)
from .smile_builder import SmileQuotes, rr_bf_costs


@dataclass(frozen=True)
class InstrumentGreeks:
    vega: float
    vanna: float
    volga: float


def _vanilla_greeks(env: FXEnv, strike: float, sigma: float, is_call: bool) -> Dict[str, float]:
    return greeks_gk(is_call, GKInput(env.spot, strike, env.rd, env.rf, sigma, env.tau))


def _instrument_greeks(
    env: FXEnv, sigma_atm: float, k_atm: float, kp: float, kc: float
) -> Dict[str, InstrumentGreeks]:
    # ATM = 1/2 * Straddle(K_atm)
    g_call_atm = _vanilla_greeks(env, k_atm, sigma_atm, True)
    g_put_atm = _vanilla_greeks(env, k_atm, sigma_atm, False)
    vega_atm = 0.5 * (g_call_atm["vega"] + g_put_atm["vega"])
    vanna_atm = 0.5 * (g_call_atm["vanna"] + g_put_atm["vanna"])
    volga_atm = 0.5 * (g_call_atm["volga"] + g_put_atm["volga"])

    # RR = Call(Kc) - Put(Kp)
    g_call_kc = _vanilla_greeks(env, kc, sigma_atm, True)
    g_put_kp = _vanilla_greeks(env, kp, sigma_atm, False)
    vega_rr = g_call_kc["vega"] - g_put_kp["vega"]
    vanna_rr = g_call_kc["vanna"] - g_put_kp["vanna"]
    volga_rr = g_call_kc["volga"] - g_put_kp["volga"]

    # BF = 1/2 * Strangle(Kc,Kp) - 1/2 * Straddle(K_atm)
    vega_bf = 0.5 * (g_call_kc["vega"] + g_put_kp["vega"]) - vega_atm
    vanna_bf = 0.5 * (g_call_kc["vanna"] + g_put_kp["vanna"]) - vanna_atm
    volga_bf = 0.5 * (g_call_kc["volga"] + g_put_kp["volga"]) - volga_atm

    return {
        "ATM": InstrumentGreeks(vega_atm, vanna_atm, volga_atm),
        "RR": InstrumentGreeks(vega_rr, vanna_rr, volga_rr),
        "BF": InstrumentGreeks(vega_bf, vanna_bf, volga_bf),
    }


def compute_omega(
    env: FXEnv,
    quotes: SmileQuotes,
    conv: DeltaConvention,
    premium_included_atm: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Omega (market price per unit Vega/Vanna/Volga).

    Returns (Omega[3], A[3x3]) with Omega ordered [Vega, Vanna, Volga].

    Raises:
        NumericalError: If the greeks matrix is singular (no VV weights exist).
    """
    conv = DeltaConvention(conv)
    # Each 25-delta wing strike must be solved with its OWN quoted wing vol, not
    # the ATM vol; otherwise Omega calibrates to the wrong options and the smile
    # fails to reprice market quotes at the true 25-delta strikes.
    sigma_25p, sigma_25c = quotes.sigma_25d()
    kp = strike_for_delta(-0.25, False, env, sigma_25p, conv)
    kc = strike_for_delta(+0.25, True, env, sigma_25c, conv)
    k_atm = atm_strike(quotes.sigma_atm, env, premium_included=premium_included_atm)

    ig = _instrument_greeks(env, quotes.sigma_atm, k_atm, kp, kc)
    A = np.array(
        [
            [ig["ATM"].vega, ig["ATM"].vanna, ig["ATM"].volga],
            [ig["RR"].vega, ig["RR"].vanna, ig["RR"].volga],
            [ig["BF"].vega, ig["BF"].vanna, ig["BF"].volga],
        ],
        dtype=float,
    )

    rr_cost, bf_cost = rr_bf_costs(env, quotes.sigma_atm, kc, kp, sigma_25c, sigma_25p)
    I = np.array([0.0, rr_cost, bf_cost], dtype=float)

    # A has instruments as rows and greeks as columns; I is indexed by
    # instrument. The calibration condition is therefore A @ Omega = I (each
    # instrument's greeks dotted with Omega reproduce its market cost).
    try:
        Omega = np.linalg.solve(A, I)
    except np.linalg.LinAlgError as exc:
        raise NumericalError(
            "Vanna-Volga greeks matrix is singular; cannot solve for Omega"
        ) from exc
    return Omega, A


def vv_adjustment_simple(
    x_vanna: float,
    x_volga: float,
    rr_cost: float,
    bf_cost: float,
    rr_vanna: float,
    bf_volga: float,
) -> float:
    """Simple VV correction. Vega term omitted by design.

    X_VV = w_RR * RR_cost + w_BF * BF_cost, with
    w_RR = Vanna(X)/Vanna(RR), w_BF = Volga(X)/Volga(BF).
    """
    w_rr = 0.0 if abs(rr_vanna) < Tolerance.LOG_MIN else x_vanna / rr_vanna
    w_bf = 0.0 if abs(bf_volga) < Tolerance.LOG_MIN else x_volga / bf_volga
    return w_rr * rr_cost + w_bf * bf_cost


def vv_adjustment_matrix(x_vega: float, x_vanna: float, x_volga: float, Omega: np.ndarray) -> float:
    """Matrix VV correction: x . Omega."""
    x = np.array([x_vega, x_vanna, x_volga], dtype=float)
    return float(x @ Omega)


__all__ = [
    "InstrumentGreeks",
    "compute_omega",
    "vv_adjustment_simple",
    "vv_adjustment_matrix",
]
