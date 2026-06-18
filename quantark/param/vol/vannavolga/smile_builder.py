"""
Vanna-Volga smile inputs: ATM / risk-reversal / butterfly quotes, RR/BF costs,
and the broker-strangle single-vol solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from quantark.util.exceptions import NumericalError, ValidationError
from quantark.util.numerical import Tolerance

from .bs_fx import GKInput, greeks_gk, price_gk
from .market_conventions import DeltaConvention, FXEnv, strikes_25d


@dataclass(frozen=True)
class SmileQuotes:
    """Market smile quotes: ATM vol, 25d risk-reversal, 25d butterfly (2-vol)."""

    sigma_atm: float
    rr25: float
    bf25_2vol: float

    def __post_init__(self) -> None:
        if self.sigma_atm <= 0.0:
            raise ValidationError(f"sigma_atm must be positive, got {self.sigma_atm}")
        sigma_25p, sigma_25c = self.sigma_25d()
        if sigma_25p <= 0.0 or sigma_25c <= 0.0:
            raise ValidationError(
                "Derived 25-delta wing vols must be positive "
                f"(got put={sigma_25p:.4f}, call={sigma_25c:.4f}); check RR/BF quotes"
            )

    def sigma_25d(self) -> Tuple[float, float]:
        # sigma_25C = sigma_ATM + BF25(2vol) + RR25/2
        # sigma_25P = sigma_ATM + BF25(2vol) - RR25/2
        sigma_25c = self.sigma_atm + self.bf25_2vol + 0.5 * self.rr25
        sigma_25p = self.sigma_atm + self.bf25_2vol - 0.5 * self.rr25
        return sigma_25p, sigma_25c


def rr_bf_costs(
    env: FXEnv,
    sigma_atm: float,
    kc: float,
    kp: float,
    sigma_kc: float,
    sigma_kp: float,
    notional: float = 1.0,
) -> Tuple[float, float]:
    """Risk-reversal and butterfly market costs (smile minus flat-ATM prices)."""
    call_mkt = price_gk(True, GKInput(env.spot, kc, env.rd, env.rf, sigma_kc, env.tau, notional))
    put_mkt = price_gk(False, GKInput(env.spot, kp, env.rd, env.rf, sigma_kp, env.tau, notional))

    call_bs = price_gk(True, GKInput(env.spot, kc, env.rd, env.rf, sigma_atm, env.tau, notional))
    put_bs = price_gk(False, GKInput(env.spot, kp, env.rd, env.rf, sigma_atm, env.tau, notional))

    rr_cost = (call_mkt - put_mkt) - (call_bs - put_bs)
    bf_cost = 0.5 * (call_mkt + put_mkt) - 0.5 * (call_bs + put_bs)
    return rr_cost, bf_cost


def broker_strangle_sigma_1vol(
    env: FXEnv,
    quotes: SmileQuotes,
    conv: DeltaConvention,
    tol: float = Tolerance.PRECISION * 1e-2,
    max_iter: int = 50,
) -> Tuple[float, float, float, float]:
    """Solve the single-vol broker strangle sigma_STG25(1vol).

    Returns (sigma_stg_1vol, sigma_bf25_1vol, K25P_star, K25C_star).

    Known approximation (faithful to the legacy source): the smile leg of the
    strangle equality is valued with the quoted wing vols ``sigma_25p/sigma_25c``
    at the iterated trial strikes ``K25P_star/K25C_star``. The rigorous
    smile-consistent form evaluates the *calibrated* smile vols
    ``sigma(K25P_star)/sigma(K25C_star)`` at those strikes, which differ for
    steep skews. This utility is not on the delivered pricing path
    (SABRVolSurface / VannaVolgaVolSurface / price_vv_one_touch do not call it).
    # TODO(smile-consistent-strangle): evaluate the calibrated smile at the
    # iterated trial strikes instead of reusing the quoted 25-delta wing vols.

    Raises:
        NumericalError: If the secant iteration fails to converge.
    """
    sigma_25p, sigma_25c = quotes.sigma_25d()

    # Initial guess: vega-weighted approximation at ATM strikes.
    k25p, k25c = strikes_25d(quotes.sigma_atm, env, conv)
    vega_p = greeks_gk(False, GKInput(env.spot, k25p, env.rd, env.rf, quotes.sigma_atm, env.tau))["vega"]
    vega_c = greeks_gk(True, GKInput(env.spot, k25c, env.rd, env.rf, quotes.sigma_atm, env.tau))["vega"]
    sigma_stg = (sigma_25p * vega_p + sigma_25c * vega_c) / max(
        vega_p + vega_c, Tolerance.DENOMINATOR_MIN
    )

    def diff(sigma_single: float) -> float:
        k25p_star, k25c_star = strikes_25d(sigma_single, env, conv)
        val_single = (
            price_gk(True, GKInput(env.spot, k25c_star, env.rd, env.rf, sigma_single, env.tau))
            + price_gk(False, GKInput(env.spot, k25p_star, env.rd, env.rf, sigma_single, env.tau))
        )
        val_smile = (
            price_gk(True, GKInput(env.spot, k25c_star, env.rd, env.rf, sigma_25c, env.tau))
            + price_gk(False, GKInput(env.spot, k25p_star, env.rd, env.rf, sigma_25p, env.tau))
        )
        return val_single - val_smile

    # Secant iteration with explicit (x0, x1) bracket points.
    x0, fx0 = sigma_stg, diff(sigma_stg)
    x1 = max(Tolerance.PRECISION, quotes.sigma_atm)
    if abs(x1 - x0) < Tolerance.PRECISION * 1e-3:
        x1 *= 1.05
    fx1 = diff(x1)

    sigma_solution = x1 if abs(fx1) <= abs(fx0) else x0
    converged = min(abs(fx0), abs(fx1)) < tol
    for _ in range(max_iter):
        if converged:
            break
        if abs(fx1 - fx0) < Tolerance.LOG_MIN:
            break  # secant denominator collapsed: stagnation
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        x2 = max(x2, Tolerance.LOG_MIN)
        fx2 = diff(x2)
        sigma_solution = x2
        if abs(fx2) < tol:
            converged = True
            break
        x0, fx0 = x1, fx1
        x1, fx1 = x2, fx2

    if not converged:
        raise NumericalError(
            "broker_strangle_sigma_1vol failed to converge for the supplied quotes"
        )

    sigma_stg = sigma_solution
    k25p_star, k25c_star = strikes_25d(sigma_stg, env, conv)
    sigma_bf25_1vol = sigma_stg - quotes.sigma_atm
    return sigma_stg, sigma_bf25_1vol, k25p_star, k25c_star


__all__ = ["SmileQuotes", "rr_bf_costs", "broker_strangle_sigma_1vol"]
