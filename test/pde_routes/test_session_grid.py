"""Session outputs and the public spot curve (spec 3C decision 26; §7.2 rows 'session grid and spot curve' (i)-(iv),
'valuation-date jump' on the grid, 'sparse schedule's grid', 'valuation-date record' (i) on the grid)."""
import math

import numpy as np
import pytest

from pde_routes.barrier_reference import knock_out
from pde_routes.cases import CONST, barrier, black_scholes, env_const, obs_dates, schedule
from pde_routes.sessions import GRID, PV, session
from quantark.asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine
from quantark.asset.equity.engine.analytical.double_barrier_option_engine import DoubleBarrierOptionAnalyticalEngine
from quantark.asset.equity.engine.pde import (BarrierPDESolver, DoubleBarrierPDESolver, EuropeanPDESolver,
                                              OneTouchPDESolver)
from quantark.asset.equity.product.option import DoubleBarrierOption
from quantark.asset.equity.product.option.one_touch_option import OneTouchOption
from quantark.execution.errors import CapabilityError
from quantark.util.enum import (BarrierDirection, DoubleBarrierType, ObservationAggregation, ObservationType,
                                OptionType, TouchType)

VOL, R, Q = CONST["vol"], CONST["rate"], CONST["div"]
M = dict(vol=VOL, r=R, q=Q)
BOUND = dict(price=1e-3, delta=2e-4, gamma=2e-4)


def one_sided(f, s, barriers, h=1e-4):
    """(f, f', f'') at s by a three-point stencil that never crosses a barrier (§4.10's trap)."""
    hs = h * s
    up_ok = all(not (s < b <= s + 2 * hs) for b in barriers)
    dn_ok = all(not (s - 2 * hs <= b < s) for b in barriers)
    if up_ok and dn_ok:
        a, b, c = f(s - hs), f(s), f(s + hs)
        return b, (c - a) / (2 * hs), (c - 2 * b + a) / hs ** 2
    if dn_ok:                                                  # backward
        a, b, c = f(s - 2 * hs), f(s - hs), f(s)
        return c, (3 * c - 4 * b + a) / (2 * hs), (c - 2 * b + a) / hs ** 2
    a, b, c = f(s), f(s + hs), f(s + 2 * hs)                  # forward
    return a, (-3 * a + 4 * b - c) / (2 * hs), (c - 2 * b + a) / hs ** 2


def reference(name, s):
    """(price, delta, gamma) of the six knock-ins of §4.10 without a rebate (Task 12 adds the rebate cells)."""
    if name == "cont_ui_call_130":
        f = lambda x: BarrierAnalyticalEngine().price(barrier("UI", True, 130.0), env_const(x))     # noqa: E731
        return one_sided(f, s, [130.0])
    if name == "cont_di_put_85":
        f = lambda x: BarrierAnalyticalEngine().price(barrier("DI", False, 85.0), env_const(x))     # noqa: E731
        return one_sided(f, s, [85.0])
    if name == "cont_dki_call":
        p = DoubleBarrierOption(100.0, OptionType.CALL, 130.0, 80.0, DoubleBarrierType.KNOCK_IN, maturity=1.0,
                                observation_type=ObservationType.CONTINUOUS)
        f = lambda x: DoubleBarrierOptionAnalyticalEngine().price(p, env_const(x))                    # noqa: E731
        return one_sided(f, s, [80.0, 130.0])
    up, call, level = {"monthly_ui_call_130": (True, True, 130.0), "monthly_di_put_85": (False, False, 85.0)}[name]
    ko = knock_out(payoff="call" if call else "put", strike=100.0, lower=None if up else level,
                   upper=level if up else None, cash=0.0, at_hit=False, obs=obs_dates("monthly"), spots=[s], **M)[0][s]
    bs = black_scholes(call, s, 100.0, 1.0)
    return bs[0] - ko[0], bs[1] - ko[1], bs[2] - ko[2]


PRODUCTS = {
    "cont_ui_call_130": (BarrierPDESolver, lambda: barrier("UI", True, 130.0), [80, 100, 120, 129, 131, 150]),
    "cont_di_put_85": (BarrierPDESolver, lambda: barrier("DI", False, 85.0), [70, 84, 86, 100, 120]),
    "monthly_ui_call_130": (BarrierPDESolver, lambda: barrier("UI", True, 130.0, obs="monthly"), [80, 100, 120, 135, 150]),
    "monthly_di_put_85": (BarrierPDESolver, lambda: barrier("DI", False, 85.0, obs="monthly"), [70, 80, 100, 120]),
    "cont_dki_call": (DoubleBarrierPDESolver,
                      lambda: DoubleBarrierOption(100.0, OptionType.CALL, 130.0, 80.0, DoubleBarrierType.KNOCK_IN,
                                                  maturity=1.0, observation_type=ObservationType.CONTINUOUS),
                      [70, 79, 82, 100, 129, 131]),
}


@pytest.mark.parametrize("name", sorted(PRODUCTS))
def test_knock_in_curve_is_the_composition(name):
    cls, make, levels = PRODUCTS[name]
    rows = cls().calculate_spot_greeks_curve(make(), env_const(100.0), [float(x) for x in levels])
    for row in rows:
        p, d, g = reference(name, row["spot"])
        assert abs(row["price"] - p) <= BOUND["price"], (name, row["spot"])        # today -4.2 at 100, -52 at 150
        assert abs(row["delta"] - d) <= BOUND["delta"], (name, row["spot"])
        assert abs(row["gamma"] - g) <= BOUND["gamma"], (name, row["spot"])


@pytest.mark.parametrize("name", ["cont_ui_call_130", "cont_dki_call"])
def test_default_levels_are_the_vanilla_parts_nodes(name):
    cls, make, _ = PRODUCTS[name]
    out = session(cls(), make(), env_const(100.0), {PV, GRID})
    nodes = [row["spot"] for row in out.grid]
    van = EuropeanPDESolver()._session_outputs(make_vanilla(make()), env_const(100.0), want_grid=True).solution
    assert nodes == [float(s) for s in van.s_vec]
    for row in out.grid[5:-5]:                                   # the domain's interior
        p, d, g = reference(name, row["spot"])
        assert abs(row["price"] - p) <= BOUND["price"] and abs(row["delta"] - d) <= BOUND["delta"]
        assert abs(row["gamma"] - g) <= BOUND["gamma"]


def make_vanilla(product):
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    return EuropeanVanillaOption(strike=product.strike, option_type=product.option_type, maturity=product.maturity)


@pytest.mark.parametrize("cls,product", [
    (EuropeanPDESolver, lambda: make_vanilla(barrier("UO", True, 130.0))),
    (BarrierPDESolver, lambda: barrier("UO", True, 130.0, rebate=5.0)),
    (BarrierPDESolver, lambda: barrier("UI", True, 130.0)),
])
def test_session_rows_equal_the_public_curve(cls, product):
    levels = [80.0, 100.0, 125.0, 135.0]
    env = env_const(100.0)
    out = session(cls(), product(), env, {PV, GRID}, levels=levels)
    assert out.pv == cls().price(product(), env)
    assert list(out.grid) == cls().calculate_spot_greeks_curve(product(), env, levels)


@pytest.mark.parametrize("cls,product,level", [
    (BarrierPDESolver, lambda: barrier("UO", True, 130.0, rebate=5.0), 135.0),
    (BarrierPDESolver, lambda: barrier("UO", True, 130.0), 135.0),
    (OneTouchPDESolver, lambda: OneTouchOption(barrier=130.0, barrier_direction=BarrierDirection.UP, maturity=1.0,
                                               rebate=1.0, payment_at_hit=False, touch_type=TouchType.ONE_TOUCH,
                                               observation_type=ObservationType.CONTINUOUS), 135.0),
])
def test_hit_levels_return_the_hit_outcome(cls, product, level):
    row = cls().calculate_spot_greeks_curve(product(), env_const(100.0), [level])[0]
    g = cls().calculate_greeks(product(), env_const(level))
    assert row["price"] == g["price"] and row["delta"] == 0.0 and row["gamma"] == 0.0   # today -0.150 in delta


def test_product_decided_at_the_base_spot():
    p = barrier("UO", True, 130.0, rebate=5.0)
    env = env_const(135.0)
    with pytest.raises(CapabilityError):
        session(BarrierPDESolver(), p, env, {PV, GRID})
    assert session(BarrierPDESolver(), p, env, {PV}) == BarrierPDESolver().price(p, env)
    rows = BarrierPDESolver().calculate_spot_greeks_curve(p, env, [120.0, 135.0])
    for row in rows:
        g = BarrierPDESolver().calculate_greeks(p, env_const(row["spot"]))
        assert row["price"] == g["price"] and row["delta"] == g["delta"]


def test_valuation_date_jump_on_the_grid():
    """Observations 0 / 0.5 / 1; the grid from base spot 100 read through the continuation (§4.11)."""
    from quantark.asset.equity.product.option import BarrierOption
    from quantark.util.enum import BarrierType
    p = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, barrier_type=BarrierType.UP_IN,
                      maturity=1.0, observation_type=ObservationType.DISCRETE,
                      observation_schedule=schedule([0.0, 0.5, 1.0], ObservationAggregation.STOP_FIRST_HIT, payoff=0.0))
    rows = BarrierPDESolver().calculate_spot_greeks_curve(p, env_const(100.0), [129.0, 129.9, 129.99, 130.0, 131.0])
    for row in rows:
        s = row["spot"]
        bs = black_scholes(True, s, 100.0, 1.0)
        if s >= 130.0:
            want = bs[:2]
        else:
            ko = knock_out(payoff="call", strike=100.0, lower=None, upper=130.0, cash=0.0, at_hit=False,
                           obs=[0.5, 1.0], spots=[s], **M)[0][s]
            want = (bs[0] - ko[0], bs[1] - ko[1])
        assert abs(row["price"] - want[0]) <= 1.5e-3 and abs(row["delta"] - want[1]) <= 2e-4   # today 32.25 for 29.00


@pytest.mark.parametrize("accuracy,bound", [("standard", 1.5e-3), ("high", 5e-4)])
def test_sparse_schedule_grid(accuracy, bound):
    """A semiannual knock-in from base spot 100: the composed grid needs repair (b)'s row (§4.11: -0.28 at 150)."""
    from quantark.asset.equity.param import PDEParams
    from quantark.asset.equity.product.option import BarrierOption
    from quantark.util.enum import BarrierType
    # built with its semiannual dates: assigning observation_dates afterwards leaves the monthly schedule the
    # constructor derived
    p = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, barrier_type=BarrierType.UP_IN,
                      maturity=1.0, observation_type=ObservationType.DISCRETE, observation_dates=[0.5, 1.0])
    rows = BarrierPDESolver(PDEParams(accuracy=accuracy)).calculate_spot_greeks_curve(
        p, env_const(100.0), [80.0, 100.0, 120.0, 129.99, 150.0])
    for row in rows:
        s = row["spot"]
        ko = knock_out(payoff="call", strike=100.0, lower=None, upper=130.0, cash=0.0, at_hit=False, obs=[0.5, 1.0],
                       spots=[s], **M)[0][s][0]
        assert abs(row["price"] - (black_scholes(True, s, 100.0, 1.0)[0] - ko)) <= bound


def test_valuation_record_on_the_grid_and_in_a_session():
    """Review Focus 1 and §7.2 'valuation-date record' (i): the record's cash at and beyond its level."""
    from pde_routes.cases import record_product
    env = env_const(100.0)
    rows = BarrierPDESolver().calculate_spot_greeks_curve(record_product(), env, [119.99, 120.0, 135.0])
    cash = 7.0 * math.exp(-R * 0.25)
    assert abs(rows[1]["price"] - cash) <= 1e-12 and abs(rows[2]["price"] - cash) <= 1e-12
    assert rows[1]["delta"] == rows[2]["delta"] == 0.0
    assert session(BarrierPDESolver(), record_product(), env_const(125.0), {PV}) == BarrierPDESolver().price(
        record_product(), env_const(125.0))
