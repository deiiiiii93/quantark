"""Shared model-appropriate Greeks for equity local-vol engines.

For a calibrated local-volatility model the IV-bump vega is meaningless, so only
delta, gamma, theta, and rho are exposed. The derived LocalVolSurface is held FIXED
across spot/rate/time bumps (sticky-local-vol Greeks): the surface is calibrated once
and reused, so the sensitivities exclude recalibration effects.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict

from quantark.param.rrf import ParallelShiftRateCurve
from quantark.util.numerical.constants import Tolerance


def local_vol_model_greeks(
    price_with_surface: Callable,
    product,
    pricing_env,
    lv_surface,
    spot_bump: float = Tolerance.BUMP_SPOT,
    rate_bump: float = Tolerance.BUMP_RATE,
    theta_days: int = 1,
) -> Dict[str, float]:
    """Delta, gamma, theta, rho with a fixed local-vol surface. No vega."""
    base = price_with_surface(product, pricing_env, lv_surface)
    greeks: Dict[str, float] = {"price": base}

    spot = pricing_env.spot
    env_up = deepcopy(pricing_env)
    env_up.spot_quote.spot = spot * (1.0 + spot_bump)
    env_dn = deepcopy(pricing_env)
    env_dn.spot_quote.spot = spot * (1.0 - spot_bump)
    price_up = price_with_surface(product, env_up, lv_surface)
    price_dn = price_with_surface(product, env_dn, lv_surface)
    greeks["delta"] = (price_up - price_dn) / (2.0 * spot * spot_bump)
    greeks["gamma"] = (price_up - 2.0 * base + price_dn) / (spot * spot_bump) ** 2

    env_ru = deepcopy(pricing_env)
    env_ru.rate_curve = ParallelShiftRateCurve(pricing_env.rate_curve, rate_bump)
    env_rd = deepcopy(pricing_env)
    env_rd.rate_curve = ParallelShiftRateCurve(pricing_env.rate_curve, -rate_bump)
    greeks["rho"] = (
        price_with_surface(product, env_ru, lv_surface)
        - price_with_surface(product, env_rd, lv_surface)
    ) / (2.0 * rate_bump)

    dt = theta_days / 365.0
    maturity = getattr(product, "maturity", None)
    if maturity is not None and maturity > dt:
        shifted = deepcopy(product)
        shifted.maturity = maturity - dt
        future = price_with_surface(shifted, pricing_env, lv_surface)
        greeks["theta"] = (future - base) / theta_days

    return greeks
