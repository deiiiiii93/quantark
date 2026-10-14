"""A double knock-out's rebate is paid only to knocked-out paths, at the terminal payment time (T17 R30).

``DoubleBarrierOption`` has no hit-paid option: its rebate is paid at the terminal payment time, as the analytical
forward formula, the PDE's forward leg and its discrete records pay it. Two routes broke that:

* QUAD's forward lattice: ``DoubleBarrierQuadInputAdapter._terminal_payoff_coefficients`` moves the last record's
  ``k_minus`` to the strike for a call (``k_plus`` for a put) to carve the vanilla, so that record's rebate paid on
  (0, K] (put: [K, inf)) at maturity -- to paths that never knocked out;
* the PDE's continuous hit-now value paid the rebate at the hit instant (undiscounted).

Oracle (independent of the double-barrier adapter): the rebate leg ``V(R) - V(0) = R x DF(T) x P(knock-out)``, the
survival probability read from QUAD's double no-touch (paying 1 at the terminal when no record is breached); a
knock-in's rebate leg is ``R x DF(T) x P(survive)``.
"""
from __future__ import annotations

import math
import warnings
from datetime import datetime

import pytest

from quantark.asset.equity.engine.pde.double_barrier_pde_solver import DoubleBarrierPDESolver
from quantark.asset.equity.engine.quad import DiscreteQuadEngine
from quantark.asset.equity.param import PDEParams, QuadParams
from quantark.asset.equity.product.option import DoubleBarrierOption, DoubleOneTouchOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import DoubleBarrierType, ObservationType, OptionType, TouchType

VALUATION = datetime(2025, 1, 6)
RATE = 0.03
#: QUAD's rebate leg vs the oracle (one lattice, the same survival probabilities: measured <= 1.6e-15)
ORACLE_ABS = 1e-9
#: QUAD vs the Black-Scholes double-barrier PDE (grid error; measured <= 6.2e-4 on these cells, 0.75 before R30)
PDE_ABS = 2e-3
#: (aged years, spot)
CELLS = {"inception_100": (0.0, 100.0), "aged_0.30_129": (0.30, 129.0), "aged_0.30_80": (0.30, 80.0)}


def env(spot: float) -> PricingEnvironment:
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=0.20),
                              rate_curve=FlatRateCurve(rate=RATE), div_yield=ContinuousDividendYield(0.01),
                              valuation_date=VALUATION)


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _schedule():
    return ObservationSchedule(records=[ObservationRecord(observation_time=k / 12, upper_barrier=130.0,
                                                          lower_barrier=75.0) for k in range(1, 13)])


def _aged(product, years):
    if years:
        product.time_shift(years, VALUATION, env(100.0), keep_landing=False)
    return product


def _double(option_type, barrier_type, rebate, years, observation_type=ObservationType.DISCRETE):
    schedule = _schedule() if observation_type == ObservationType.DISCRETE else None
    return _aged(DoubleBarrierOption(strike=100.0, option_type=option_type, upper_barrier=130.0, lower_barrier=75.0,
                                     barrier_type=barrier_type, rebate=rebate, maturity=1.0,
                                     observation_type=observation_type, observation_schedule=schedule), years)


def _survival(years, spot) -> float:
    """P(no record breached), from QUAD's double no-touch paying 1 at the terminal."""
    dnt = _aged(DoubleOneTouchOption(upper_barrier=130.0, lower_barrier=75.0, maturity=1.0, rebate=1.0,
                                     payment_at_hit=False, touch_type=TouchType.DOUBLE_NO_TOUCH,
                                     observation_type=ObservationType.DISCRETE, observation_schedule=_schedule()),
                years)
    return _quiet(lambda: _quad().price(dnt, env(spot))) / math.exp(-RATE * (1.0 - years))


def _quad():
    return DiscreteQuadEngine(params=QuadParams(grid_points=401))


def _leg(engine, option_type, barrier_type, years, spot) -> float:
    with_rebate = _quiet(lambda: engine().price(_double(option_type, barrier_type, 2.0, years), env(spot)))
    without = _quiet(lambda: engine().price(_double(option_type, barrier_type, 0.0, years), env(spot)))
    return with_rebate - without


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT], ids=["call", "put"])
@pytest.mark.parametrize("cell", list(CELLS))
def test_quad_knock_out_rebate_leg_is_paid_only_on_knock_out(cell, option_type):
    years, spot = CELLS[cell]
    oracle = 2.0 * math.exp(-RATE * (1.0 - years)) * (1.0 - _survival(years, spot))
    assert _leg(_quad, option_type, DoubleBarrierType.KNOCK_OUT, years, spot) == pytest.approx(oracle, abs=ORACLE_ABS)


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT], ids=["call", "put"])
@pytest.mark.parametrize("cell", list(CELLS))
def test_quad_knock_in_rebate_leg_is_paid_only_on_survival(cell, option_type):
    years, spot = CELLS[cell]
    oracle = 2.0 * math.exp(-RATE * (1.0 - years)) * _survival(years, spot)
    assert _leg(_quad, option_type, DoubleBarrierType.KNOCK_IN, years, spot) == pytest.approx(oracle, abs=ORACLE_ABS)


@pytest.mark.parametrize("barrier_type", [DoubleBarrierType.KNOCK_OUT, DoubleBarrierType.KNOCK_IN], ids=["ko", "ki"])
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT], ids=["call", "put"])
@pytest.mark.parametrize("cell", list(CELLS))
def test_quad_double_barrier_equals_the_pde(cell, option_type, barrier_type):
    years, spot = CELLS[cell]
    quad = _quiet(lambda: _quad().price(_double(option_type, barrier_type, 2.0, years), env(spot)))
    pde = _quiet(lambda: DoubleBarrierPDESolver(PDEParams()).price(_double(option_type, barrier_type, 2.0, years),
                                                                    env(spot)))
    assert quad == pytest.approx(pde, abs=PDE_ABS)


@pytest.mark.parametrize("spot", [131.0, 74.0])
def test_pde_continuous_hit_now_pays_the_rebate_at_the_terminal_payment_time(spot):
    """Aged 0.30, the spot outside the corridor: knocked out now, the rebate paid at maturity (2 x DF(0.70))."""
    product = _double(OptionType.CALL, DoubleBarrierType.KNOCK_OUT, 2.0, 0.30, ObservationType.CONTINUOUS)
    value = _quiet(lambda: DoubleBarrierPDESolver(PDEParams()).price(product, env(spot)))
    assert value == pytest.approx(2.0 * math.exp(-RATE * 0.70), rel=1e-12)


def test_pde_continuous_price_is_continuous_at_the_barrier():
    """Just inside the corridor the forward leg tends to the terminal-paid rebate: no step at the barrier."""
    def price(spot):
        product = _double(OptionType.CALL, DoubleBarrierType.KNOCK_OUT, 2.0, 0.30, ObservationType.CONTINUOUS)
        return _quiet(lambda: DoubleBarrierPDESolver(PDEParams()).price(product, env(spot)))
    assert abs(price(129.999) - price(130.0)) < 5e-3
