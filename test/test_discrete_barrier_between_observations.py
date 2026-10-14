"""A DISCRETE barrier is decided only on an observation (A15, user decision 2026-10-14: "knock out should only happen
at the observation date, otherwise price it").

With NO record on the valuation instant nothing is decided now, whatever the spot: every route prices the contract
forward from today's spot and observes each future record with its own machinery (MC paths, QUAD lattice, PDE
events, the analytical engines' first-observation integral). A record ON the valuation instant (dated: the valuation
date; numeric: ``is_zero(observation_time)``) with the spot beyond its level still decides at once (W18b / W18c,
unchanged). This supersedes the W18b pre-check that read the NEXT record between observations (the Black-Scholes
barrier MC, the QUAD adapters; the single / double sharkfin MC read ``product.barrier``).

Contracts: maturity 1.0, twelve monthly records (``k / 12``), flat 20 % vol, r 3 %, q 1 %. "Between observations"
is aged 0.30 (the next record at 1/3, twelve days ahead); "record today" is aged 0.25 keeping the record on the
valuation instant. Knock-out cash is 0 (sharkfins 0.5), so an immediate knock-out is far from the forward value
and a jump at the level would show.

Reference and tolerances (measured 2026-10-14, see the T17 report):

* QUAD (401 points) is exact discrete monitoring up to its quadrature error: it is the reference where it prices
  the product. The Black-Scholes PDE agrees within ``PDE_ABS`` (its grid; measured <= 8e-4), the Black-Scholes
  barrier MC within 4 s.e.
* Beyond the level the analytical engines integrate the first observation exactly and continue with the
  BGK-shifted continuous formula: within ``ANALYTICAL_ABS`` of QUAD (measured <= 0.023). The sharkfins have no QUAD
  adapter, so their MC is checked against the analytical value within 4 s.e. + ``ANALYTICAL_ABS``.
* Continuity across the level (spot ``level -/+ STEP``): QUAD, the PDE and the MC (common random numbers) move by
  delta x 2 STEP (measured <= 0.026, ``CONTINUITY_ABS``). The analytical engines switch at the level from the
  BGK-shifted closed form (inside) to the first-observation integral (beyond); the BGK error next to the barrier
  makes a step of up to 0.152 here (``ANALYTICAL_STEP_ABS``, pre-existing, unchanged by A15). Every route's step
  stays far below the step a decision taken now would make (``kind.now``: >= 0.5 on these contracts).
"""
from __future__ import annotations

import math
import warnings
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine import barrier_schedule_rules
from quantark.asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine
from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.engine.analytical.double_barrier_option_engine import DoubleBarrierOptionAnalyticalEngine
from quantark.asset.equity.engine.analytical.double_sharkfin_option_analytical_engine import (
    DoubleSharkfinOptionAnalyticalEngine,
)
from quantark.asset.equity.engine.analytical.one_touch_analytical_engine import OneTouchAnalyticalEngine
from quantark.asset.equity.engine.analytical.single_sharkfin_option_analytical_engine import (
    SingleSharkfinOptionAnalyticalEngine,
)
from quantark.asset.equity.engine.mc import BarrierOptionMCEngine
from quantark.asset.equity.engine.mc.double_sharkfin_option_mc_engine import DoubleSharkfinOptionMCEngine
from quantark.asset.equity.engine.mc.single_sharkfin_option_mc_engine import SingleSharkfinOptionMCEngine
from quantark.asset.equity.engine.pde.barrier_pde_solver import BarrierPDESolver
from quantark.asset.equity.engine.pde.double_barrier_pde_solver import DoubleBarrierPDESolver
from quantark.asset.equity.engine.pde.double_one_touch_pde_solver import DoubleOneTouchPDESolver
from quantark.asset.equity.engine.pde.one_touch_pde_solver import OneTouchPDESolver
from quantark.asset.equity.engine.quad import DiscreteQuadEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option import (
    BarrierOption,
    DoubleBarrierOption,
    DoubleOneTouchOption,
    DoubleSharkfinOption,
    EuropeanVanillaOption,
    OneTouchOption,
    SingleSharkfinOption,
)
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import (
    BarrierDirection,
    BarrierType,
    DoubleBarrierType,
    ObservationType,
    OptionType,
    TouchType,
)

VALUATION = datetime(2025, 1, 6)
TIMES = [k / 12 for k in range(1, 13)]
BETWEEN = 0.30          # aged: no record on the valuation instant, the next at 1/3
RECORD_DAY = 0.25       # aged onto the 3-month record, kept on the valuation instant
#: the Black-Scholes PDE vs QUAD (grid error, measured <= 8e-4 on these contracts)
PDE_ABS = 2e-3
#: the analytical engines beyond the level vs exact discrete monitoring (measured <= 0.023 on these contracts)
ANALYTICAL_ABS = 0.05
#: continuity: spot level -/+ STEP; QUAD / PDE / MC move by delta x 2 STEP (measured <= 0.026)
STEP = 0.01
CONTINUITY_ABS = 0.04
#: the analytical engines' step at the level: the BGK error next to the barrier (measured <= 0.152, pre-existing)
ANALYTICAL_STEP_ABS = 0.25
#: a decision taken now would step by at least this much on every contract here (measured >= 0.5)
DECISION_STEP = 0.4


def env(spot: float, valuation: datetime = VALUATION) -> PricingEnvironment:
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=0.20),
                              rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(0.01),
                              valuation_date=valuation)


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _single_schedule(level):
    return ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=level) for t in TIMES])


def _double_schedule(upper, lower):
    return ObservationSchedule(records=[
        ObservationRecord(observation_time=t, upper_barrier=upper, lower_barrier=lower) for t in TIMES])


_DISCRETE = dict(maturity=1.0, observation_type=ObservationType.DISCRETE)


def _barrier(barrier_type, option_type, level):
    return lambda: BarrierOption(strike=100.0, option_type=option_type, barrier=level, barrier_type=barrier_type,
                                 rebate=0.0, observation_schedule=_single_schedule(level), **_DISCRETE)


def _touch(touch_type):
    return lambda: OneTouchOption(barrier=120.0, barrier_direction=BarrierDirection.UP, rebate=5.0,
                                  payment_at_hit=False, touch_type=touch_type,
                                  observation_schedule=_single_schedule(120.0), **_DISCRETE)


def _double(barrier_type):
    return lambda: DoubleBarrierOption(strike=100.0, option_type=OptionType.CALL, upper_barrier=130.0,
                                       lower_barrier=75.0, barrier_type=barrier_type, rebate=0.0,
                                       observation_schedule=_double_schedule(130.0, 75.0), **_DISCRETE)


def _double_touch(touch_type):
    return lambda: DoubleOneTouchOption(upper_barrier=130.0, lower_barrier=75.0, rebate=5.0, payment_at_hit=False,
                                        touch_type=touch_type, observation_schedule=_double_schedule(130.0, 75.0),
                                        **_DISCRETE)


def _single_sharkfin():
    return SingleSharkfinOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, knock_out_rebate=0.5,
                                observation_schedule=_single_schedule(120.0), **_DISCRETE)


def _double_sharkfin():
    return DoubleSharkfinOption(strike=100.0, option_type=OptionType.CALL, upper_barrier=130.0, lower_barrier=75.0,
                                knock_out_rebate=0.5, observation_schedule=_double_schedule(130.0, 75.0), **_DISCRETE)


def _mc():
    return MCParams(seed=11, num_paths=20000, time_steps=73)


def _quad():
    return DiscreteQuadEngine(params=QuadParams(grid_points=401))


ROUTES = {
    "analytical_barrier": BarrierAnalyticalEngine,
    "analytical_touch": OneTouchAnalyticalEngine,
    "analytical_double": DoubleBarrierOptionAnalyticalEngine,
    "analytical_single_sharkfin": SingleSharkfinOptionAnalyticalEngine,
    "analytical_double_sharkfin": DoubleSharkfinOptionAnalyticalEngine,
    "quad": _quad,
    "bs_mc": lambda: BarrierOptionMCEngine(params=_mc()),
    "single_sharkfin_mc": lambda: SingleSharkfinOptionMCEngine(params=_mc()),
    "double_sharkfin_mc": lambda: DoubleSharkfinOptionMCEngine(params=_mc()),
    "pde_barrier": lambda: BarrierPDESolver(PDEParams()),
    "pde_touch": lambda: OneTouchPDESolver(PDEParams()),
    "pde_double": lambda: DoubleBarrierPDESolver(PDEParams()),
    "pde_double_touch": lambda: DoubleOneTouchPDESolver(PDEParams()),
}


class Kind:
    """A contract kind: its builder, the monitored level the spot crosses (the upper one of a corridor, the barrier
    side a single barrier watches), the sign of "beyond" (+1 above), the routes and the reference route; ``now`` is
    the value decided at the valuation instant on a record day (spot beyond, 0.75 years left)."""

    def __init__(self, build, level, side, routes, reference, now):
        self.build, self.level, self.side = build, level, side
        self.routes, self.reference, self.now = routes, reference, now

    @property
    def beyond(self) -> float:
        return self.level + self.side * 1.0


_SINGLE = ["analytical_barrier", "quad", "bs_mc", "pde_barrier"]
_TOUCH = ["analytical_touch", "quad", "pde_touch"]
_DOUBLE = ["analytical_double", "quad", "pde_double"]
_DOUBLE_TOUCH = ["quad", "pde_double_touch"]


def _df(tau):
    return env(100.0).get_discount_factor(tau)


def _vanilla(option_type, spot):
    return BlackScholesEngine().price(EuropeanVanillaOption(strike=100.0, option_type=option_type, maturity=0.75),
                                      env(spot))


KINDS = {
    "up_out_call": Kind(_barrier(BarrierType.UP_OUT, OptionType.CALL, 120.0), 120.0, +1, _SINGLE, "quad",
                        lambda spot: 0.0),
    "up_in_call": Kind(_barrier(BarrierType.UP_IN, OptionType.CALL, 120.0), 120.0, +1, _SINGLE, "quad",
                       lambda spot: _vanilla(OptionType.CALL, spot)),
    "down_out_put": Kind(_barrier(BarrierType.DOWN_OUT, OptionType.PUT, 85.0), 85.0, -1, _SINGLE, "quad",
                         lambda spot: 0.0),
    "down_in_put": Kind(_barrier(BarrierType.DOWN_IN, OptionType.PUT, 85.0), 85.0, -1, _SINGLE, "quad",
                        lambda spot: _vanilla(OptionType.PUT, spot)),
    "one_touch": Kind(_touch(TouchType.ONE_TOUCH), 120.0, +1, _TOUCH, "quad", lambda spot: 5.0 * _df(0.75)),
    "no_touch": Kind(_touch(TouchType.NO_TOUCH), 120.0, +1, _TOUCH, "quad", lambda spot: 0.0),
    "double_knock_out": Kind(_double(DoubleBarrierType.KNOCK_OUT), 130.0, +1, _DOUBLE, "quad", lambda spot: 0.0),
    "double_knock_in": Kind(_double(DoubleBarrierType.KNOCK_IN), 130.0, +1, _DOUBLE, "quad",
                            lambda spot: _vanilla(OptionType.CALL, spot)),
    "double_one_touch": Kind(_double_touch(TouchType.DOUBLE_ONE_TOUCH), 130.0, +1, _DOUBLE_TOUCH, "quad",
                             lambda spot: 5.0 * _df(0.75)),
    "double_no_touch": Kind(_double_touch(TouchType.DOUBLE_NO_TOUCH), 130.0, +1, _DOUBLE_TOUCH, "quad",
                            lambda spot: 0.0),
    "single_sharkfin": Kind(_single_sharkfin, 120.0, +1, ["analytical_single_sharkfin", "single_sharkfin_mc"],
                            "analytical_single_sharkfin", lambda spot: 0.5 * _df(0.75)),
    "double_sharkfin": Kind(_double_sharkfin, 130.0, +1, ["analytical_double_sharkfin", "double_sharkfin_mc"],
                            "analytical_double_sharkfin", lambda spot: 0.5 * _df(0.75)),
}

CASES = [(name, route) for name, kind in KINDS.items() for route in kind.routes]
_IDS = [f"{n}-{r}" for n, r in CASES]


def _aged(product, years: float, keep_landing: bool):
    product.time_shift(years, VALUATION, env(100.0), keep_landing=keep_landing)
    return product


def _price(route: str, product, spot: float):
    engine = ROUTES[route]()
    value = _quiet(lambda: engine.price(product, env(spot)))
    se = engine.get_last_std_error() if hasattr(engine, "get_last_std_error") else None
    return float(value), se


def _between(kind: Kind, route: str, spot: float):
    return _price(route, _aged(kind.build(), BETWEEN, keep_landing=False), spot)


# ======================================================================================================================
# the shared rule (barrier_schedule_rules)
# ======================================================================================================================

def test_no_record_on_the_valuation_instant_decides_nothing():
    product = _aged(KINDS["up_out_call"].build(), BETWEEN, keep_landing=False)
    assert barrier_schedule_rules.valuation_instant_barrier_terms(product, env(121.0)) is None


def test_the_record_on_the_valuation_instant_decides_with_its_level_and_payoff():
    product = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT,
                            rebate=5.0, observation_schedule=ObservationSchedule(records=[
                                ObservationRecord(observation_time=t, barrier=115.0, payoff=4.5) for t in TIMES]),
                            **_DISCRETE)
    _aged(product, RECORD_DAY, keep_landing=True)
    assert barrier_schedule_rules.valuation_instant_barrier_terms(product, env(121.0)) == (115.0, 4.5)


def test_a_dated_record_on_the_valuation_date_decides_and_the_day_after_does_not():
    records = [ObservationRecord(observation_date=VALUATION + timedelta(days=30 * k), barrier=120.0)
               for k in range(1, 12)]
    product = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT,
                            rebate=5.0, exercise_date=VALUATION + timedelta(days=360),
                            observation_type=ObservationType.DISCRETE,
                            observation_schedule=ObservationSchedule(records=records))
    on_record = VALUATION + timedelta(days=30)
    assert barrier_schedule_rules.valuation_instant_barrier_terms(product, env(121.0, on_record)) == (120.0, 5.0)
    assert barrier_schedule_rules.valuation_instant_barrier_terms(
        product, env(121.0, on_record - timedelta(days=1))) is None


def test_the_double_rule_reads_the_record_on_the_valuation_instant_only():
    between = _aged(KINDS["double_knock_out"].build(), BETWEEN, keep_landing=False)
    on_record = _aged(KINDS["double_knock_out"].build(), RECORD_DAY, keep_landing=True)
    assert barrier_schedule_rules.valuation_instant_double_barrier_terms(between, env(131.0)) is None
    assert barrier_schedule_rules.valuation_instant_double_barrier_terms(on_record, env(131.0)) == (130.0, 75.0, 0.0)


def test_continuous_monitoring_keeps_the_product_terms():
    product = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT,
                            rebate=5.0, maturity=1.0, observation_type=ObservationType.CONTINUOUS)
    assert barrier_schedule_rules.valuation_instant_barrier_terms(product, env(121.0)) == (120.0, 5.0)


# ======================================================================================================================
# AC17a: no record today, spot beyond the next record's level -- every route prices forward
# ======================================================================================================================

@pytest.mark.parametrize("name,route", CASES, ids=_IDS)
def test_between_observations_beyond_the_level_every_route_prices_forward(name, route):
    """The route's forward value agrees with the reference (QUAD, the sharkfins' analytical value) and is not the
    value an immediate decision would give."""
    kind = KINDS[name]
    value, se = _between(kind, route, kind.beyond)
    reference, _ = _between(kind, kind.reference, kind.beyond)
    assert math.isfinite(value)
    assert abs(value - kind.now(kind.beyond)) > 0.1, "decided now, not priced forward"
    if route == kind.reference:
        return
    if se is not None:                                       # Monte Carlo
        slack = ANALYTICAL_ABS if kind.reference.startswith("analytical") else 0.0
        assert abs(value - reference) <= 4.0 * se + slack, (value, reference, se)
    elif route.startswith("pde"):
        assert value == pytest.approx(reference, abs=PDE_ABS)
    else:
        assert value == pytest.approx(reference, abs=ANALYTICAL_ABS)


@pytest.mark.parametrize("name,route", CASES, ids=_IDS)
def test_between_observations_the_price_is_continuous_across_the_level(name, route):
    """Just inside vs just beyond the level, no record today: the price moves by about delta x 2 STEP (the MC on
    common random numbers; the analytical engines by their BGK step, see the module docstring), never by the jump
    of a decision taken now."""
    kind = KINDS[name]
    spot_inside, spot_beyond = kind.level - kind.side * STEP, kind.level + kind.side * STEP
    inside, _ = _between(kind, route, spot_inside)
    beyond, _ = _between(kind, route, spot_beyond)
    assert abs(beyond - kind.now(spot_beyond)) >= DECISION_STEP      # the step a decision now would make
    bound = ANALYTICAL_STEP_ABS if route.startswith("analytical") else CONTINUITY_ABS
    assert abs(beyond - inside) <= bound, (inside, beyond)


# ======================================================================================================================
# AC17b: aged onto a record day, spot beyond -- decided at once on every route (unchanged)
# ======================================================================================================================

@pytest.mark.parametrize("name,route", CASES, ids=_IDS)
def test_on_a_record_day_beyond_the_level_every_route_decides_at_once(name, route):
    """A knock-out pays its cash (0; a sharkfin's 0.5) at the terminal payment time, a one-touch its 5, a no-touch
    nothing; a knock-in is the vanilla (the route's own: QUAD's quadrature, the MC's paths, the PDE's grid)."""
    kind = KINDS[name]
    value, se = _price(route, _aged(kind.build(), RECORD_DAY, keep_landing=True), kind.beyond)
    expected = kind.now(kind.beyond)
    if name.endswith("_in_call") or name.endswith("_in_put") or name == "double_knock_in":
        if se:
            assert abs(value - expected) <= 4.0 * se
        else:
            assert value == pytest.approx(expected, rel=2e-3)
    else:
        assert value == pytest.approx(expected, rel=1e-12, abs=1e-12)


# ======================================================================================================================
# the vol-model routes (Local Vol / Heston / SLV barrier MC and PDE): already forward (no pre-check), pinned here
# ======================================================================================================================

def _vol_routes():
    from test_vol_dated_schedules import BARRIER_ROUTES

    from quantark.asset.equity.engine.pde.barrier_vol_pde_solvers import LocalVolBarrierPDESolver

    def mc(factory):
        def build():
            engine = factory()
            engine.params.num_paths = 4000
            return engine
        return build

    from test_dated_expiry_settles import HESTON, SLV_LEVERAGE

    from quantark.asset.equity.engine.pde import HestonBarrierPDESolver, HestonSLVBarrierPDESolver

    routes = {name: (mc(factory), grid) for name, (factory, grid) in BARRIER_ROUTES.items() if name.endswith("_mc")}
    routes["lv_pde"] = (lambda: LocalVolBarrierPDESolver(PDEParams()), True)
    routes["heston_pde"] = (lambda: HestonBarrierPDESolver(HESTON, **_ADI_FINE), False)
    routes["slv_pde"] = (lambda: HestonSLVBarrierPDESolver(HESTON, SLV_LEVERAGE, **_ADI_FINE), False)
    return routes


VOL_ROUTES = ["lv_mc", "heston_mc", "slv_mc", "lv_pde", "heston_pde", "slv_pde"]
#: the 2D ADI grid of the continuity check: next to the pinned barrier node the coarse grids step across the level
#: (measured on the up-and-out call: 0.31 at 48x12x24, 0.13 at 96x24x96, 0.055 here -- O(h), a grid artefact, not a
#: decision), so the 2D routes get ``ADI_STEP_ABS`` on this grid
_ADI_FINE = dict(n_x=160, n_v=32, n_t=300)
ADI_STEP_ABS = 0.1
#: (barrier type, option type, level, the side of "beyond")
VOL_KINDS = {"up_out_call": (BarrierType.UP_OUT, OptionType.CALL, 120.0, +1),
             "down_in_put": (BarrierType.DOWN_IN, OptionType.PUT, 85.0, -1)}


def _vol_case(route: str, kind: str, day: int, spot: float, keep_landing: bool = False) -> float:
    from test_vol_dated_schedules import EXERCISE, INITIAL, _on, _records, env as vol_env

    barrier_type, option_type, level, _ = VOL_KINDS[kind]
    product = BarrierOption(strike=100.0, option_type=option_type, barrier=level, barrier_type=barrier_type,
                            rebate=0.0, exercise_date=EXERCISE, initial_date=INITIAL,
                            observation_type=ObservationType.DISCRETE,
                            observation_schedule=ObservationSchedule(records=_records("dated", level)))
    product.time_shift(day / 365.0, _on(day), vol_env(100.0, _on(day)), keep_landing=keep_landing)
    factory, grid = _vol_routes()[route]
    return float(_quiet(lambda: factory().price(product, vol_env(spot, _on(day), grid=grid))))


@pytest.mark.parametrize("route", VOL_ROUTES)
@pytest.mark.parametrize("kind", list(VOL_KINDS))
def test_vol_model_routes_price_forward_between_observations(route, kind):
    """Day 92, the next record on day 121: across the level the price moves by delta x 2 STEP, never by a decision
    taken now (a knock-out's 0, a knock-in's vanilla)."""
    _, _, level, side = VOL_KINDS[kind]
    inside = _vol_case(route, kind, 92, level - side * STEP)
    beyond = _vol_case(route, kind, 92, level + side * STEP)
    bound = ADI_STEP_ABS if route in ("heston_pde", "slv_pde") else CONTINUITY_ABS
    assert abs(beyond - inside) <= bound, (inside, beyond)
    if kind == "up_out_call":
        assert beyond >= DECISION_STEP           # a knock-out decided now pays its cash 0


@pytest.mark.parametrize("route", VOL_ROUTES)
def test_vol_model_routes_knock_out_at_once_on_a_record_day(route):
    """Day 91 holds a record: a spot beyond it knocks out at once (cash 0)."""
    assert _vol_case(route, "up_out_call", 91, 121.0, keep_landing=True) == pytest.approx(0.0, abs=1e-12)
