"""An observation AT the valuation instant is decided on the known spot, on every engine.

The spot is known, so nothing about the observation is random: the contract is worth what it pays now plus the same
contract WITHOUT that record, carrying the state the observation leaves behind. That identity needs no second engine
and no reference value, so each engine is checked against itself:

* a coupon observed now is paid now (the price rises by exactly the coupon);
* a missed memory coupon observed now is owed (it equals the contract carrying one coupon of arrears);
* a knock-in observed now is carried (it equals the contract carrying the knock-in).

Before 2026-09-21 the Phoenix QUAD V1 engine skipped an observation now altogether (one coupon short, no memory, no
knock-in), the knock-out-reset QUAD V1 engine did the same, and the knock-out-reset Monte Carlo refused to price
(a zero time step). A knock-out observed now, tie included, is in ``test_autocallable_same_observation_tie.py``.
"""
from dataclasses import replace
from datetime import datetime

import pytest

from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde import PhoenixPDESolver, SnowballPDESolver
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.quad.v2 import KOResetSnowballQuadEngineV2, PhoenixQuadEngineV2, SnowballQuadEngineV2
from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.option_enums import ObservationType

KO, KI, POST_KO, COUPON_LEVEL, COUPON = 103.0, 75.0, 98.0, 80.0, 1.0
LATER = (0.5, 1.0)
NOW_AND_LATER = (0.0,) + LATER
PAYS, MISSES, KNOCKS_IN = 90.0, 78.0, 70.0       # between the coupon and knock-out levels; below the coupon level; below the knock-in level


def env(spot):
    return PricingEnvironment(rate_curve=FlatRateCurve(0.03), valuation_date=datetime(2026, 3, 16), spot_quote=SpotQuote(spot),
                              vol_surface=FlatVolSurface(0.25), div_yield=ContinuousDividendYield(0.01))


def _schedule(level, times):
    return ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=level) for t in times])


def _barrier(ko_rate, times):
    return BarrierConfig(ko_barrier=KO, ko_rate=ko_rate, ko_observation_type=ObservationType.DISCRETE,
                         ko_observation_schedule=_schedule(KO, times), ki_barrier=KI, ki_observation_type=ObservationType.DISCRETE,
                         ki_observation_schedule=_schedule(KI, times))


def snowball(times):
    return SnowballOption(initial_price=100.0, strike=100.0, maturity=1.0, barrier_config=_barrier(0.15, times),
                          accrual_config=AccrualConfig(is_annualized=False),
                          payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False))


def phoenix(times, memory=False, arrears=0.0):
    return PhoenixOption(initial_price=100.0, strike=100.0, maturity=1.0, barrier_config=_barrier(0.0, times),
                         # 2% a year at a fixed half-year fraction: one unit a period, a quotation both carriers of
                         # the memory arrears accept (the count QUAD V2 takes needs every period worth the same)
                         coupon_config=CouponBarrierConfig(coupon_barrier=COUPON_LEVEL, coupon_rate=0.02, memory_coupon=memory,
                                                           initial_coupon_arrears=arrears, fixed_coupon_year_fraction=0.5),
                         accrual_config=AccrualConfig(is_annualized=False, is_annualized_coupon=True),
                         payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True))


def ko_reset(times):
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0, barrier_config=_barrier(0.15, times),
        post_barrier_config=BarrierConfig(ko_barrier=POST_KO, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=_schedule(POST_KO, times)),
        accrual_config=AccrualConfig(is_annualized=False), payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False))


def carrying_a_knock_in(product):
    product._otc_lifecycle_knocked_in = True
    return product


_mc = lambda cls: (lambda: cls(params=MCParams(seed=7, num_paths=32768)))                                     # noqa: E731
_quad = lambda cls: (lambda: cls(params=QuadParams(grid_points=801)))                                         # noqa: E731
ENGINES = {
    "snowball": {"mc": _mc(SnowballMCEngine), "pde": lambda: SnowballPDESolver(PDEParams()),
                 "quad_v1": _quad(SnowballQuadEngine), "quad_v2": lambda: SnowballQuadEngineV2()},
    "phoenix": {"mc": _mc(PhoenixMCEngine), "pde": lambda: PhoenixPDESolver(PDEParams()),
                "quad_v1": _quad(PhoenixQuadEngine), "quad_v2": lambda: PhoenixQuadEngineV2()},
    "ko_reset": {"mc": _mc(SnowballMCEngine), "pde": lambda: KOResetSnowballPDESolver(PDEParams()),
                 "quad_v1": _quad(KOResetSnowballQuadEngine), "quad_v2": lambda: KOResetSnowballQuadEngineV2()},
}
BUILD = {"snowball": snowball, "phoenix": phoenix, "ko_reset": ko_reset}
# The observation now adds no simulation node and no lattice level, so each engine prices the two sides of an identity
# on the same paths or the same grid: they agree to rounding, not to a discretisation error.
SAME_GRID = 1e-8


def price(family, engine, product, spot, **state):
    return float(ENGINES[family][engine]().price(product, env(spot), **state))


def owing_one_coupon(engine):
    """The same memory Phoenix without the record observed now, carrying the coupon it missed.

    The arrears have two carriers, by design (see ``quantark/intraday/twin.py``): the stateless engines read the AMOUNT on
    the contract, QUAD V2 rebuilds it from the COUNT of missed periods on the lifecycle state. Exactly one is set per
    engine, so neither counts it twice."""
    if engine == "quad_v2":
        return phoenix(LATER, memory=True), {"lifecycle_state": AutocallableLifecycleState(coupon_memory_count=1)}
    return phoenix(LATER, memory=True, arrears=COUPON), {}


@pytest.mark.parametrize("engine", ["mc", "pde", "quad_v1", "quad_v2"])
@pytest.mark.parametrize("memory", [False, True], ids=["plain", "memory"])
def test_a_coupon_observed_now_is_paid_now(engine, memory):
    observed = price("phoenix", engine, phoenix(NOW_AND_LATER, memory), PAYS)
    without = price("phoenix", engine, phoenix(LATER, memory), PAYS)
    assert observed - without == pytest.approx(COUPON, abs=SAME_GRID)


@pytest.mark.parametrize("engine", ["mc", "pde", "quad_v1", "quad_v2"])
def test_a_coupon_missed_now_is_lost_without_memory(engine):
    observed = price("phoenix", engine, phoenix(NOW_AND_LATER), MISSES)
    assert observed == pytest.approx(price("phoenix", engine, phoenix(LATER), MISSES), abs=SAME_GRID)


@pytest.mark.parametrize("engine", ["mc", "pde", "quad_v1", "quad_v2"])
def test_a_memory_coupon_missed_now_is_owed(engine):
    observed = price("phoenix", engine, phoenix(NOW_AND_LATER, memory=True), MISSES)
    contract, state = owing_one_coupon(engine)
    owed = price("phoenix", engine, contract, MISSES, **state)
    forgotten = price("phoenix", engine, phoenix(LATER, memory=True), MISSES)
    assert observed == pytest.approx(owed, abs=SAME_GRID)
    assert owed - forgotten > 0.5                       # the arrears are worth having: the identity is not vacuous


@pytest.mark.parametrize("engine", ["mc", "pde", "quad_v1", "quad_v2"])
@pytest.mark.parametrize("family", sorted(BUILD))
def test_a_knock_in_observed_now_is_carried(family, engine):
    build = BUILD[family]
    observed = price(family, engine, build(NOW_AND_LATER), KNOCKS_IN)
    carried = price(family, engine, carrying_a_knock_in(build(LATER)), KNOCKS_IN)
    untouched = price(family, engine, build(LATER), KNOCKS_IN)
    assert observed == pytest.approx(carried, abs=SAME_GRID)
    assert abs(carried - untouched) > 0.5               # the knock-in changes the value: the identity is not vacuous
