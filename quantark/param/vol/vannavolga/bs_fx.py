"""
Self-contained Garman-Kohlhagen FX pricing and greeks for Vanna-Volga.

This lives in the ``param`` layer, which must not import from the ``engine``
layer; it therefore keeps a small, dependency-light GK implementation rather
than reusing ``quantark.asset.fx.engine``. The engine-layer
``GarmanKohlhagenEngine`` remains the source of truth for option pricing; a
test cross-checks the two for numerical consistency.

Greeks here are the *raw* Black-Scholes/GK sensitivities (vega per unit vol,
not per 1%), which is what the Vanna-Volga construction consumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

from scipy.stats import norm

from quantark.util.numerical import is_zero


@dataclass(frozen=True)
class GKInput:
    spot: float
    strike: float
    rd: float
    rf: float
    vol: float
    tau: float
    notional: float = 1.0

    @property
    def df_dom(self) -> float:
        return math.exp(-self.rd * self.tau)

    @property
    def df_for(self) -> float:
        return math.exp(-self.rf * self.tau)

    @property
    def forward(self) -> float:
        return self.spot * math.exp((self.rd - self.rf) * self.tau)


def _d1_d2(forward: float, strike: float, vol: float, tau: float) -> Tuple[float, float, float]:
    sqrt_tau = math.sqrt(max(tau, 0.0))
    if is_zero(sqrt_tau) or vol <= 0.0:
        # Deterministic limit: the option is worth its forward intrinsic. Return
        # sign-aware infinities so N(d1)=N(d2) collapse to 1 (forward in-the-
        # money) or 0 (out-of-the-money), giving DF_dom*max(F-K, 0) etc.
        sign = float("inf") if forward >= strike else float("-inf")
        return (sign, sign, sqrt_tau)
    a1 = math.log(forward / strike) + 0.5 * (vol**2) * tau
    a2 = vol * sqrt_tau
    d1 = a1 / a2
    d2 = d1 - a2
    return d1, d2, sqrt_tau


def price_gk(is_call: bool, x: GKInput) -> float:
    """Garman-Kohlhagen price in domestic currency."""
    F = x.forward
    d1, d2, _ = _d1_d2(F, x.strike, x.vol, x.tau)
    if is_call:
        val = x.notional * (
            x.spot * x.df_for * norm.cdf(d1) - x.strike * x.df_dom * norm.cdf(d2)
        )
    else:
        val = x.notional * (
            x.strike * x.df_dom * norm.cdf(-d2) - x.spot * x.df_for * norm.cdf(-d1)
        )
    return float(val)


def greeks_gk(is_call: bool, x: GKInput) -> Dict[str, float]:
    """Raw GK greeks (vega/vanna/volga per unit vol) consumed by Vanna-Volga."""
    F = x.forward
    d1, d2, sqrt_tau = _d1_d2(F, x.strike, x.vol, x.tau)
    pdf_d1 = norm.pdf(d1)

    delta = x.notional * (
        x.df_for * norm.cdf(d1) if is_call else -x.df_for * norm.cdf(-d1)
    )

    degenerate = is_zero(sqrt_tau) or x.vol <= 0.0
    if degenerate:
        gamma = 0.0
    else:
        gamma = x.notional * (x.df_for * pdf_d1) / (x.spot * x.vol * sqrt_tau)

    vega = x.notional * (x.spot * x.df_for * pdf_d1 * sqrt_tau)

    # Theta diffusion term is undefined at sqrt(tau)=0; the deterministic carry
    # terms remain well defined.
    theta_diffusion = (
        0.0
        if degenerate
        else -0.5 * x.notional * x.spot * x.df_for * pdf_d1 * x.vol / sqrt_tau
    )
    if is_call:
        theta_carry = x.notional * (
            -x.rd * x.strike * x.df_dom * norm.cdf(d2)
            + x.rf * x.spot * x.df_for * norm.cdf(d1)
        )
    else:
        theta_carry = x.notional * (
            x.rd * x.strike * x.df_dom * norm.cdf(-d2)
            - x.rf * x.spot * x.df_for * norm.cdf(-d1)
        )
    theta = theta_diffusion + theta_carry

    # Domestic rho is positive for calls (higher rd raises the forward),
    # foreign rho is negative for calls; signs reverse for puts.
    rho_dom = x.notional * (
        x.tau * x.strike * x.df_dom * (norm.cdf(d2) if is_call else -norm.cdf(-d2))
    )
    rho_for = x.notional * (
        -x.tau * x.spot * x.df_for * (norm.cdf(d1) if is_call else -norm.cdf(-d1))
    )

    # Vanna = d(vega)/dS, Volga = d(vega)/dsigma.
    if x.vol <= 0.0:
        vanna = 0.0
        volga = 0.0
    else:
        vanna = x.notional * x.df_for * pdf_d1 * (sqrt_tau - d1 / x.vol)
        a = math.log(F / x.strike)
        volga = (
            x.notional
            * x.spot
            * x.df_for
            * pdf_d1
            * (d1 * (a / (x.vol**2) - 0.5 * x.tau))
        )

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho_dom": float(rho_dom),
        "rho_for": float(rho_for),
        "vanna": float(vanna),
        "volga": float(volga),
        "d1": float(d1),
        "d2": float(d2),
    }


__all__ = ["GKInput", "price_gk", "greeks_gk"]
