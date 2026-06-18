"""Shared model-appropriate Greeks for FX local-vol engines.

Exposes delta, gamma, theta, rho_dom, rho_for (NO IV-bump vega) holding the calibrated
LocalVolSurface fixed across bumps (sticky-local-vol Greeks).
"""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from typing import Callable, Dict

from quantark.param.rrf import ParallelShiftRateCurve
from quantark.util.numerical.constants import Tolerance


def fx_local_vol_model_greeks(
    price_with_surface: Callable,
    product,
    fx_env,
    lv_surface,
    spot_bump: float = Tolerance.BUMP_SPOT,
    rate_bump: float = Tolerance.BUMP_RATE,
    theta_days: int = 1,
) -> Dict[str, float]:
    base = price_with_surface(product, fx_env, lv_surface)
    greeks: Dict[str, float] = {"price": base}

    spot = fx_env.effective_spot()
    env_up = deepcopy(fx_env)
    env_up.spot_quote.spot = fx_env.spot_quote.spot * (1.0 + spot_bump)
    env_dn = deepcopy(fx_env)
    env_dn.spot_quote.spot = fx_env.spot_quote.spot * (1.0 - spot_bump)
    price_up = price_with_surface(product, env_up, lv_surface)
    price_dn = price_with_surface(product, env_dn, lv_surface)
    greeks["delta"] = (price_up - price_dn) / (2.0 * spot * spot_bump)
    greeks["gamma"] = (price_up - 2.0 * base + price_dn) / (spot * spot_bump) ** 2

    # Repository convention: rho reported per 1% rate move (dV/dr / 100).
    env_du = deepcopy(fx_env)
    env_du.domestic_curve = ParallelShiftRateCurve(fx_env.domestic_curve, rate_bump)
    env_dd = deepcopy(fx_env)
    env_dd.domestic_curve = ParallelShiftRateCurve(fx_env.domestic_curve, -rate_bump)
    greeks["rho_dom"] = (
        price_with_surface(product, env_du, lv_surface)
        - price_with_surface(product, env_dd, lv_surface)
    ) / (2.0 * rate_bump) / 100.0

    env_fu = deepcopy(fx_env)
    env_fu.foreign_curve = ParallelShiftRateCurve(fx_env.foreign_curve, rate_bump)
    env_fd = deepcopy(fx_env)
    env_fd.foreign_curve = ParallelShiftRateCurve(fx_env.foreign_curve, -rate_bump)
    greeks["rho_for"] = (
        price_with_surface(product, env_fu, lv_surface)
        - price_with_surface(product, env_fd, lv_surface)
    ) / (2.0 * rate_bump) / 100.0

    # Theta: daily decay. Shrink float maturity AND delivery (preserving expiry==delivery);
    # for date-based products advance the valuation date so theta is always provided.
    dt = theta_days / 365.0
    maturity = getattr(product, "maturity", None)
    if maturity is not None and maturity > 0:
        eff = min(dt, 0.5 * maturity)  # never shrink past expiry
        shifted = deepcopy(product)
        shifted.maturity = maturity - eff
        if getattr(shifted, "delivery", None) is not None:
            shifted.delivery = shifted.delivery - eff
        future = price_with_surface(shifted, fx_env, lv_surface)
        greeks["theta"] = (future - base) / (eff * 365.0)
    else:
        env_fut = deepcopy(fx_env)
        env_fut.valuation_date = fx_env.valuation_date + timedelta(days=theta_days)
        future = price_with_surface(product, env_fut, lv_surface)
        greeks["theta"] = (future - base) / theta_days

    return greeks
