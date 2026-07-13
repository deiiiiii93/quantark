"""Cash-Greeks report in model-validation conventions (spec WP2.1).

All FD bumps reuse the SAME engine instance (fixed seed -> common random
numbers). Conventions are pinned by spec: rho/rhoq are derivative-based
(central ±1bp, scaled ×100 to +1% absolute); theta_1d rolls the valuation
date one SSE trading day with contractual dates fixed (no schedule
time_shift); vega is a central ±1 vol pt bump reported per +1 vol pt.
"""
from __future__ import annotations

import dataclasses
from copy import deepcopy
from dataclasses import dataclass

from quantark.param.div.dividend_yield import DividendYield
from quantark.param.rrf import ParallelShiftRateCurve
from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class CashGreeksReport:
    pv: float
    delta_cash: float        # Delta * S
    gamma_cash_1pct: float   # Gamma * S^2 / 100
    vega_1pct: float         # dV per +1 vol pt (absolute)
    theta_1d: float          # V(t + 1 trading day) - V(t), contract dates fixed
    rho_1pct: float          # central ±1bp parallel zero-rate bump × 100
    rhoq_1pct: float         # central ±1bp q bump × 100, r held fixed
    bump_metadata: dict      # sizes, style, seed policy per greek

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class _ShiftedVolSurface:
    """get_vol-compatible view returning base vol + an absolute shift."""

    def __init__(self, base, shift: float):
        self._base = base
        self._shift = float(shift)

    def get_vol(self, strike, time_to_maturity, spot=None):
        return self._base.get_vol(strike, time_to_maturity, spot) + self._shift

    def __getattr__(self, name):
        return getattr(self._base, name)


class _ShiftedDividendYield(DividendYield):
    """Dividend yield shifted by a constant (r held fixed by construction)."""

    def __init__(self, base, shift: float):
        self._base = base
        self._shift = float(shift)

    def get_yield(self, time_to_maturity: float) -> float:
        base = 0.0 if self._base is None else self._base.get_yield(
            time_to_maturity
        )
        return base + self._shift


def _reprice_spot(product, env, engine, rel_bump):
    e = deepcopy(env)
    e.spot_quote = dataclasses.replace(
        e.spot_quote, spot=float(env.spot) * (1.0 + rel_bump)
    )
    return engine.price(product, e)


def _reprice_vol(product, env, engine, abs_bump):
    e = deepcopy(env)
    e.vol_surface = _ShiftedVolSurface(env.vol_surface, abs_bump)
    return engine.price(product, e)


def _reprice_rate(product, env, engine, abs_bump):
    e = deepcopy(env)
    e.rate_curve = ParallelShiftRateCurve(env.rate_curve, abs_bump)
    return engine.price(product, e)


def _reprice_div(product, env, engine, abs_bump):
    e = deepcopy(env)
    e.div_yield = _ShiftedDividendYield(env.div_yield, abs_bump)
    return engine.price(product, e)


def _date_rolled_theta(product, env, engine, calendar, base_pv):
    """V(t+1 SSE trading day) - V(t) with contractual dates held fixed.

    Supported for products whose schedule exposes a rebuildable spec
    (DCNSchedule); anything else must raise rather than approximate.
    """
    schedule = getattr(product, "schedule", None)
    spec = getattr(schedule, "spec", None)
    if spec is None or calendar is None:
        raise ValidationError(
            "date-rolled theta needs a product with a rebuildable schedule "
            "spec and an explicit calendar; refusing to approximate"
        )
    rolled_date = calendar.trading_days_after(schedule.valuation_date, 1)
    rolled_schedule = dataclasses.replace(
        spec, valuation_date=rolled_date
    ).build(schedule.calendar)
    rolled_product = deepcopy(product)
    rolled_product.schedule = rolled_schedule
    return engine.price(rolled_product, env) - base_pv


def build_cash_greeks_report(
    product,
    pricing_env,
    engine,
    calendar=None,
    spot_rel_bump: float = 0.01,
    vol_abs_bump: float = 0.01,
    rate_abs_bump: float = 1e-4,
) -> CashGreeksReport:
    s = float(pricing_env.spot)
    base = engine.price(product, pricing_env)

    up = _reprice_spot(product, pricing_env, engine, +spot_rel_bump)
    dn = _reprice_spot(product, pricing_env, engine, -spot_rel_bump)
    h = s * spot_rel_bump
    delta = (up - dn) / (2.0 * h)
    gamma = (up - 2.0 * base + dn) / (h * h)

    v_up = _reprice_vol(product, pricing_env, engine, +vol_abs_bump)
    v_dn = _reprice_vol(product, pricing_env, engine, -vol_abs_bump)
    vega_1pct = (
        (v_up - v_dn) / (2.0 * vol_abs_bump) * 0.01
    )

    r_up = _reprice_rate(product, pricing_env, engine, +rate_abs_bump)
    r_dn = _reprice_rate(product, pricing_env, engine, -rate_abs_bump)
    rho_1pct = (r_up - r_dn) / (2.0 * rate_abs_bump) * 0.01

    q_up = _reprice_div(product, pricing_env, engine, +rate_abs_bump)
    q_dn = _reprice_div(product, pricing_env, engine, -rate_abs_bump)
    rhoq_1pct = (q_up - q_dn) / (2.0 * rate_abs_bump) * 0.01

    theta_1d = _date_rolled_theta(product, pricing_env, engine, calendar, base)

    meta = {
        "spot": {"style": "central", "rel_bump": spot_rel_bump,
                 "seed_policy": "common_random_numbers"},
        "vol": {"style": "central", "abs_bump": vol_abs_bump,
                "unit": "per +1 vol pt"},
        "rho": {"style": "central", "abs_bump": rate_abs_bump,
                "scale": "x100 to +1% absolute"},
        "rhoq": {"style": "central", "abs_bump": rate_abs_bump,
                 "scale": "x100 to +1% absolute", "holds_fixed": "r"},
        "theta": {"style": "date_rolled", "mode": "business_days",
                  "contract_dates": "fixed"},
    }
    return CashGreeksReport(
        pv=base,
        delta_cash=delta * s,
        gamma_cash_1pct=gamma * s * s / 100.0,
        vega_1pct=vega_1pct,
        theta_1d=theta_1d,
        rho_1pct=rho_1pct,
        rhoq_1pct=rhoq_1pct,
        bump_metadata=meta,
    )
