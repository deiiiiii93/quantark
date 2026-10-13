"""Shared model-appropriate Greeks for FX local-vol engines.

Exposes delta, gamma, theta, rho_dom, rho_for (NO IV-bump vega) holding the calibrated
LocalVolSurface fixed across bumps (sticky-local-vol Greeks).
"""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from typing import Callable, Dict

from quantark.param.rrf import ParallelShiftRateCurve
from quantark.util.calendar import calculate_year_fraction
from quantark.util.numerical import is_zero
from quantark.util.numerical.constants import Tolerance


def fx_numeric_theta(
    price_fn: Callable,
    product,
    fx_env,
    base: float,
    theta_days: int,
) -> float:
    """Theta (per calendar day) of a numeric (``maturity=``) FX product.

    The step is the environment's year fraction of ``theta_days`` calendar days (spec I8 / A10, user decision Q7), so
    it matches the step the dated twin's times take when the valuation date moves. Maturity and delivery shrink by
    ``eff = min(dt_env, 0.5 * maturity)`` and the difference is rescaled linearly to a per-calendar-day figure:
    ``(V(T - eff) - V) * dt_env / (eff * theta_days)``. A zero environment step (BUSINESS_DAYS weekend/holiday) gives
    0.0; BUSINESS_DAYS without a calendar raises ValidationError (from ``calculate_year_fraction``). Under
    CALENDAR_DAYS / ACT_365 ``dt_env == theta_days / 365.0`` and the historical expression is kept bitwise.
    """
    maturity = product.maturity
    future_date = fx_env.valuation_date + timedelta(days=theta_days)
    dt_env = calculate_year_fraction(
        fx_env.valuation_date, future_date, fx_env.day_count_convention, fx_env.bus_days_in_year,
        calendar=fx_env.calendar)
    if is_zero(dt_env):
        return 0.0
    eff = min(dt_env, 0.5 * float(maturity))  # never shrink past expiry
    shifted = deepcopy(product)
    shifted.maturity = float(maturity) - eff
    if getattr(shifted, "delivery", None) is not None:
        shifted.delivery = shifted.delivery - eff
    future = price_fn(shifted, fx_env)
    if dt_env == theta_days / 365.0:
        return (future - base) / (eff * 365.0)
    return (future - base) * dt_env / (eff * theta_days)


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
    maturity = getattr(product, "maturity", None)
    if maturity is not None and maturity > 0:
        greeks["theta"] = fx_numeric_theta(
            lambda p, e: price_with_surface(p, e, lv_surface), product, fx_env, base, theta_days)
    else:
        env_fut = deepcopy(fx_env)
        env_fut.valuation_date = fx_env.valuation_date + timedelta(days=theta_days)
        future = price_with_surface(product, env_fut, lv_surface)
        greeks["theta"] = (future - base) / theta_days

    return greeks
