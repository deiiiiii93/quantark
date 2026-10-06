"""EXPIRY monitoring and the certain-knock-out far-field row (spec 3C decision 9; §7.2 rows 'EXPIRY',
'far-field row under bounds', 'term structure' (monthly barriers) and 'delayed payments on term curves')."""
import math

import pytest

from pde_routes.barrier_reference import barrier_reference, knock_out, knock_out_term
from pde_routes.cases import CONST, barrier, env_const, kinked, obs_dates
from quantark.asset.equity.engine.pde import BarrierPDESolver
from quantark.asset.equity.engine.pde.grid import GridConfig
from quantark.asset.equity.param import PDEParams

VOL, R, Q = CONST["vol"], CONST["rate"], CONST["div"]
M = dict(vol=VOL, r=R, q=Q)


@pytest.mark.parametrize("kind,call,level,rebate,spots", [("UO", True, 130.0, 0.0, (100.0, 126.1)),
                                                          ("DI", False, 85.0, 0.0, (100.0, 87.55)),
                                                          ("UO", True, 130.0, 5.0, (100.0, 135.0))])
def test_expiry_against_the_exact_recursion(kind, call, level, rebate, spots):
    p = barrier(kind, call, level, obs="expiry", rebate=rebate)
    exact = barrier_reference(kind=kind, call=call, strike=100.0, level=level, rebate=rebate, at_hit=False, obs=[1.0],
                              spots=list(spots), **M)
    for spot in spots:
        assert abs(BarrierPDESolver().price(p, env_const(spot)) - exact[spot][0]) <= 1e-3   # measured 2.8e-4


@pytest.mark.parametrize("bounds,spot", [((None, 140.0), 100.0), ((None, 140.0), 126.1)])
def test_far_field_row_under_explicit_bounds(bounds, spot):
    p = barrier("UO", True, 130.0, obs="monthly")
    exact = barrier_reference(kind="UO", call=True, strike=100.0, level=130.0, rebate=0.0, at_hit=False,
                              obs=obs_dates("monthly"), spots=[spot], **M)[spot][0]
    got = BarrierPDESolver(PDEParams(grid=GridConfig(bounds=bounds))).price(p, env_const(spot))
    assert abs(got - exact) <= 5e-2                              # measured 1.6e-2; today 9.2


def test_monthly_barriers_on_kinked_curves():
    """Decision 27: the term recursion, not self-convergence (measured 4.6e-4 on eight cells)."""
    obs = obs_dates("monthly")
    for kind, call, level, rebate in (("UO", True, 130.0, 5.0), ("DO", False, 85.0, 3.0)):
        for spot in (100.0, 120.0 if kind == "UO" else 90.0):
            env = kinked(spot)
            vol, df = VOL, (lambda t, e=env: 1.0 if t <= 0 else float(e.get_discount_factor(t)))
            growth = lambda t, e=env: 1.0 if t <= 0 else math.exp((e.get_rate(t) - e.get_div_yield(t)) * t)   # noqa: E731
            up = kind[0] == "U"
            exact = knock_out_term(payoff="call" if call else "put", strike=100.0, lower=None if up else level,
                                   upper=level if up else None, cash=rebate, at_hit=False, obs=obs, spots=[spot],
                                   vol=vol, df=df, growth=growth)[0][spot][0]
            assert abs(BarrierPDESolver().price(barrier(kind, call, level, obs="monthly", rebate=rebate), env) - exact) <= 1e-3


def test_delayed_payments_on_kinked_curves():
    """A pay-at-hit rebate and a terminal settlement lag against the term references (§7.2, 'new')."""
    from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
    obs = obs_dates("monthly")
    env = kinked(100.0)
    df = lambda t: 1.0 if t <= 0 else float(env.get_discount_factor(t))                       # noqa: E731
    growth = lambda t: 1.0 if t <= 0 else math.exp((env.get_rate(t) - env.get_div_yield(t)) * t)  # noqa: E731
    at_hit = barrier("UO", True, 130.0, obs="monthly", rebate=5.0, at_hit=True)
    exact = knock_out_term(payoff="call", strike=100.0, lower=None, upper=130.0, cash=5.0, at_hit=True, obs=obs,
                           spots=[100.0], vol=VOL, df=df, growth=growth)[0][100.0][0]
    assert abs(BarrierPDESolver().price(at_hit, env) - exact) <= 1e-3
    lagged = barrier("UO", True, 130.0, obs="monthly", rebate=5.0)
    lagged.settlement_convention = SettlementConvention(lag=0.25, lag_unit=SettlementLagUnit.YEAR_FRACTION)
    exact = knock_out_term(payoff="call", strike=100.0, lower=None, upper=130.0, cash=5.0, at_hit=False, obs=obs,
                           spots=[100.0], vol=VOL, df=df, growth=growth, terminal_pay=1.25)[0][100.0][0]
    assert abs(BarrierPDESolver().price(lagged, env) - exact) <= 1e-3


def test_accumulate_keeps_todays_row():
    """Under ACCUMULATE nothing knocks out: the row is the vanilla asymptote (decision 9)."""
    from pde_routes.cases import schedule
    from quantark.asset.equity.product.option import BarrierOption
    from quantark.util.enum import BarrierType, ObservationAggregation, ObservationType, OptionType
    p = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, barrier_type=BarrierType.UP_OUT,
                      maturity=1.0, rebate=5.0, observation_type=ObservationType.DISCRETE,
                      observation_schedule=schedule(obs_dates("monthly"), ObservationAggregation.ACCUMULATE))
    solver = BarrierPDESolver()
    solver.price(p, env_const(100.0))
    assert solver._next_observation(0.5) is not None
    assert solver._schedule_aggregation is ObservationAggregation.ACCUMULATE


# --- final review I2: EXPIRY on the double barrier and the touch products ---------------------------------------


def _expiry_products():
    from quantark.asset.equity.engine.pde import DoubleBarrierPDESolver, DoubleOneTouchPDESolver, OneTouchPDESolver
    from quantark.asset.equity.product.option import DoubleBarrierOption
    from quantark.asset.equity.product.option.double_one_touch_option import DoubleOneTouchOption
    from quantark.asset.equity.product.option.one_touch_option import OneTouchOption
    from quantark.util.enum import BarrierDirection, DoubleBarrierType, ObservationType, OptionType, TouchType
    x = ObservationType.EXPIRY
    return {
        "double_knock_out": (DoubleBarrierPDESolver, lambda: DoubleBarrierOption(
            100.0, OptionType.CALL, 130.0, 80.0, DoubleBarrierType.KNOCK_OUT, maturity=1.0, observation_type=x)),
        "double_knock_in": (DoubleBarrierPDESolver, lambda: DoubleBarrierOption(
            100.0, OptionType.CALL, 130.0, 80.0, DoubleBarrierType.KNOCK_IN, maturity=1.0, observation_type=x)),
        "one_touch": (OneTouchPDESolver, lambda: OneTouchOption(
            barrier=120.0, barrier_direction=BarrierDirection.UP, rebate=10.0, payment_at_hit=False,
            touch_type=TouchType.ONE_TOUCH, maturity=1.0, observation_type=x)),
        "double_one_touch": (DoubleOneTouchPDESolver, lambda: DoubleOneTouchOption(
            upper_barrier=120.0, lower_barrier=85.0, rebate=10.0, payment_at_hit=False,
            touch_type=TouchType.DOUBLE_ONE_TOUCH, maturity=1.0, observation_type=x)),
    }


@pytest.mark.parametrize("name", ["double_knock_out", "double_knock_in", "one_touch", "double_one_touch"])
def test_expiry_before_maturity_is_refused_where_no_expiry_far_field_row_exists(name):
    """These solvers' march applied the barrier at every step under EXPIRY (a double knock-out 2.66 against the exact
    4.64): only the single barrier has the EXPIRY far-field row (spec 3C §5.2 (b); §9 leaves the others unbuilt), so
    before maturity they refuse instead of pricing continuous monitoring. At maturity EXPIRY is decision 30's
    terminal value (test_maturity.py)."""
    from quantark.util.exceptions import ValidationError
    cls, make = _expiry_products()[name]
    with pytest.raises(ValidationError, match="EXPIRY"):
        cls().price(make(), env_const(100.0))
