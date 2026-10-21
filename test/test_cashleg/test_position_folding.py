"""EquityPosition with its cash legs folded into the product solve (ship spec 2026-10-21 D11, D12, AC7, AC8, AC10, AC12, AC15)."""
from copy import deepcopy
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.cashleg import DeterministicLeg, FoldMode, LegDirection
from quantark.execution.errors import CapabilityError
from quantark.portfolio.equity.position import EquityPosition
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import ValidationError

from intraday.conftest import dated_snowball, flat_env
from test_cashleg._autocallable_helpers import make_env, make_snowball
from test_cashleg.test_folding import UNIT, _accrual, _leg_set, _route_dist

GREEKS = ["delta", "gamma", "vega", "rho", "dividend_rho", "theta"]
CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
T0 = datetime(2026, 3, 16)


def _position(product, engine, legs, quantity=1.0, mode=None):
    pos = EquityPosition(product=product, quantity=quantity, entry_price=0.0, underlying="X",
                         engine=engine, entry_timestamp=datetime(2024, 1, 1), cash_legs=list(legs))
    pos.cash_leg_folding = mode
    return pos


def _snowball_case():
    env, product = make_env(), make_snowball(ko_dates=(0.25, 0.5, 0.75, 1.0))
    times = _route_dist(SnowballPDESolver(PDEParams()), product, env).event_times
    return product, env, _leg_set(times)


def _close(a, b, scale=UNIT, rel=1e-9):
    assert abs(a - b) <= rel * scale, (a, b, a - b)


@pytest.mark.parametrize("q", [1.0, -2.0])
def test_total_value_and_risk_match_the_event_route(q):
    product, env, legs = _snowball_case()
    gc = GreeksCalculator()
    route = _position(product, SnowballPDESolver(PDEParams()), legs, q)
    folded = _position(product, SnowballPDESolver(PDEParams()), legs, q, FoldMode.TOTAL)
    _close(folded.get_trade_value(env), route.get_trade_value(env))
    want, got = route.get_trade_risk(env, gc, GREEKS), folded.get_trade_risk(env, gc, GREEKS)
    assert got.product == {} and got.leg_pvs == {} and got.leg_greeks == {}
    assert set(got.total) == set(want.total)
    for name in want.total:
        _close(got.total[name], want.total[name], rel=1e-8)


def test_drilldown_reports_product_total_and_each_leg():
    product, env, legs = _snowball_case()
    gc, q = GreeksCalculator(), 2.0
    route = _position(product, SnowballPDESolver(PDEParams()), legs, q)
    folded = _position(product, SnowballPDESolver(PDEParams()), legs, q, FoldMode.DRILLDOWN)
    want, got = route.get_trade_risk(env, gc, GREEKS), folded.get_trade_risk(env, gc, GREEKS)
    for name in want.product:
        _close(got.product[name], want.product[name], rel=1e-8)
        _close(got.total[name], want.total[name], rel=1e-8)
    assert got.leg_pvs.keys() == want.leg_pvs.keys()
    for leg_id, pv in want.leg_pvs.items():
        assert got.leg_pvs[leg_id].name == pv.name and got.leg_pvs[leg_id].direction == pv.direction
        _close(got.leg_pvs[leg_id].pv, pv.pv)
    assert got.leg_greeks.keys() == want.leg_pvs.keys()
    for name in GREEKS:
        legs_sum = sum(g[name] for g in got.leg_greeks.values())
        _close(legs_sum, got.total[name] - got.product[name], rel=1e-9)
    breakdown = folded.get_trade_value_breakdown(env)
    _close(breakdown.total, route.get_trade_value_breakdown(env).total)


def test_total_mode_has_no_breakdown():
    product, env, legs = _snowball_case()
    with pytest.raises(ValidationError, match="DRILLDOWN"):
        _position(product, SnowballPDESolver(PDEParams()), legs, mode=FoldMode.TOTAL) \
            .get_trade_value_breakdown(env)


@pytest.mark.parametrize("mode", [FoldMode.TOTAL, FoldMode.DRILLDOWN])
def test_without_legs_the_flag_is_ignored(mode):
    product, env, _ = _snowball_case()
    gc = GreeksCalculator()
    plain = _position(product, SnowballPDESolver(PDEParams()), [])
    flagged = _position(product, SnowballPDESolver(PDEParams()), [], mode=mode)
    assert flagged.get_trade_value(env) == plain.get_trade_value(env)
    assert flagged.get_trade_risk(env, gc, GREEKS) == plain.get_trade_risk(env, gc, GREEKS)


def test_rho_and_delta_rebuild_the_overlay_in_every_bumped_market():
    product, env, _ = _snowball_case()
    legs = [_accrual(pay_shift=0.05), _accrual(pay_shift=0.11)]
    gc = GreeksCalculator()
    want = _position(product, SnowballPDESolver(PDEParams()), legs).get_trade_risk(env, gc, ["delta", "rho"])
    got = _position(product, SnowballPDESolver(PDEParams()), legs, mode=FoldMode.TOTAL) \
        .get_trade_risk(env, gc, ["delta", "rho"])
    for name in ("delta", "rho"):
        _close(got.total[name], want.total[name], rel=1e-8)


def _dated_case(env):
    product = dated_snowball(CAL, T0)
    times = _route_dist(SnowballPDESolver(PDEParams()), product, env).event_times
    legs = [leg for leg in _leg_set(times if len(times) else [1.0])]
    return product, legs


def _ko_dates(product):
    return [r.observation_date for r in product.barrier_config.ko_observation_schedule.records]


@pytest.mark.parametrize("case", ["day-before-observation", "ko-decided-at-valuation", "knocked-in"])
def test_theta_and_lifecycle_states_match_the_event_route(case):
    base = dated_snowball(CAL, T0)
    ko = _ko_dates(base)[2]
    if case == "day-before-observation":
        env = flat_env(CAL.add_business_days(ko, -1))
    elif case == "ko-decided-at-valuation":
        env = flat_env(ko, spot=110.0)
    else:
        env = flat_env(datetime(2026, 4, 1), spot=90.0)
    product, legs = _dated_case(env)
    if case == "knocked-in":
        product = deepcopy(product)
        product._otc_lifecycle_knocked_in = True
    legs = [leg for leg in legs if _leg_valid(leg, product, env)]
    gc = GreeksCalculator()
    for mode in (FoldMode.TOTAL, FoldMode.DRILLDOWN):
        want = _position(product, SnowballPDESolver(PDEParams()), legs).get_trade_risk(env, gc, GREEKS)
        got = _position(product, SnowballPDESolver(PDEParams()), legs, mode=mode).get_trade_risk(env, gc, GREEKS)
        for name in want.total:
            _close(got.total[name], want.total[name], rel=1e-8)


def _leg_valid(leg, product, env):
    try:
        leg.value(_route_dist(SnowballPDESolver(PDEParams()), product, env), env, 100.0)
        return True
    except Exception:
        return False


def test_an_engine_that_cannot_fold_refuses():
    option = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    leg = DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=1.0, payment_time=0.0)
    pos = _position(option, BlackScholesEngine(), [leg], mode=FoldMode.TOTAL)
    with pytest.raises(CapabilityError):
        pos.get_trade_value(make_env())


@pytest.mark.parametrize("engine", [
    lambda: SnowballMCEngine(params=MCParams(num_paths=2048, seed=3), method=MonteCarloMethod.RANDOMIZED_QUASI),
    lambda: SnowballQuadEngine(params=QuadParams(grid_points=301, auto_converge=True)),
])
def test_drilldown_refuses_where_two_solves_may_not_share_a_discretization(engine):
    product, env, legs = _snowball_case()
    pos = _position(product, engine(), legs, mode=FoldMode.DRILLDOWN)
    with pytest.raises(CapabilityError):
        pos.get_trade_risk(env, GreeksCalculator(), ["delta"])
