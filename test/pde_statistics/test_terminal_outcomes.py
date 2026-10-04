"""Valuation-date outcomes (spec 3B §5.5): one rule set from every entry point."""
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest, prepare
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.util.calendar import CalendarType, create_calendar
from intraday.conftest import dated_phoenix, dated_snowball, flat_env

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
T0 = datetime(2026, 3, 16)


def _first_obs(p):
    return p.barrier_config.ko_observation_schedule.records[0].observation_date


@pytest.mark.parametrize("cls,make", [(SnowballPDESolver, dated_snowball), (PhoenixPDESolver, dated_phoenix)])
def test_knocked_out_event_stats_is_a_terminal_outcome(cls, make, monkeypatch):
    product = make(CAL, T0)
    env = flat_env(_first_obs(product), spot=130.0)
    solver = cls(PDEParams())
    p = prepare(solver, product, env, OutputRequest(Operation.EVENT_STATS, price=False, event_stats=True))
    assert p.terminal is not None and p.terminal.reason == "knocked_out_at_valuation"
    assert solver.explain(product, env).terminal["event_stats"] == "knocked_out_at_valuation"
    from quantark.asset.equity.engine.pde.grid.binder import GridBinder
    calls = []
    monkeypatch.setattr(GridBinder, "bind", lambda *a, **k: calls.append(1))
    stats = solver.calculate_event_stats(product, env)
    assert not calls                                      # no layout was bound
    assert stats.ko_probability[0] == 1.0 and not np.any(stats.ko_probability[1:])


def test_nodal_knocked_out_is_a_terminal_outcome_too():
    # spec 3B stage 2 (d): the decision, not the projection, makes the outcome terminal
    product = dated_snowball(CAL, T0)
    env = flat_env(_first_obs(product), spot=130.0)
    p = prepare(SnowballPDESolver(PDEParams(event_projection="nodal")), product, env,
                OutputRequest(Operation.EVENT_STATS, price=False, event_stats=True))
    assert p.terminal is not None and p.terminal.reason == "knocked_out_at_valuation"


@pytest.mark.parametrize("memory,pay", [(False, "instant"), (True, "instant"), (True, "expiry")])
def test_knocked_out_phoenix_pays_its_coupon_with_the_knock_out(memory, pay):
    """Spec 3B §5.3/§5.5: the coupon decided with a knock-out at valuation is cash in the ledger (memory
    included, its arrears released), the maturity leg is empty and the identity closes on the value's pv."""
    import dataclasses
    from quantark.util.enum.option_enums import CouponPayType
    product = dated_phoenix(CAL, T0, memory=memory)
    if pay == "expiry":
        product.coupon_config = dataclasses.replace(product.coupon_config, coupon_pay_type=CouponPayType.EXPIRY)
    env = flat_env(_first_obs(product), spot=130.0)
    solver = PhoenixPDESolver(PDEParams())
    stats = solver.calculate_event_stats(product, env)
    rec0 = product.resolve_ko_observations(env)[0]
    df = env.get_discount_factor(float(rec0.settlement_time)) if rec0.settlement_time else 1.0
    coupon = product.get_coupon_payoff(0) + (float(product.coupon_config.initial_coupon_arrears) if memory else 0.0)
    assert stats.expected_discounted_coupon_cashflow[0] == pytest.approx(coupon * df, abs=1e-14)
    assert not np.any(stats.expected_discounted_coupon_cashflow[1:])
    assert stats.coupon_probability[0] == 1.0 and not np.any(stats.coupon_probability[1:])
    assert stats.expected_discounted_maturity_cashflow == 0.0
    assert abs(stats.reconciliation_error) <= 1e-10
    assert bool(stats.coupon_payment_is_path_dependent) is (pay == "expiry")


# --- spec 3B stage 2 (d): the family's decision first (decisions 7, 15) ----------------------------
import dataclasses

from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.cashleg.event_distribution import EventDistribution
from intraday.conftest import dated_ko_reset
from pde_statistics.cases import CASES

_FAMILIES = {"snowball": (SnowballPDESolver, dated_snowball), "phoenix": (PhoenixPDESolver, dated_phoenix),
             "ko_reset": (KOResetSnowballPDESolver, dated_ko_reset)}


def _flag(product):
    setattr(product, "_otc_lifecycle_knocked_in", True)
    return product


def _dkak(product):
    product.barrier_config = dataclasses.replace(product.barrier_config, disable_ko_after_ki=True)
    return product


def _knocked_out(kind, projection):
    """A dated product valued on its first observation date with the spot (130) above every knock-out barrier."""
    cls, make = _FAMILIES[kind]
    product = make(CAL, T0)
    params = PDEParams() if projection is None else PDEParams(event_projection=projection)
    return product, cls(params), flat_env(_first_obs(product), spot=130.0)


def _same_distribution(a, b):
    """Field-by-field equality (the dataclass's own == is ambiguous on its array fields)."""
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, dict):
            assert x.keys() == y.keys(), f.name
            for k in x:
                assert np.array_equal(np.asarray(x[k]), np.asarray(y[k])), (f.name, k)
        elif isinstance(x, np.ndarray):
            assert np.array_equal(x, y), f.name
        else:
            assert x == y, f.name


@pytest.mark.parametrize("projection", [None, "nodal"])
@pytest.mark.parametrize("kind", ["snowball", "phoenix", "ko_reset"])
def test_knocked_out_at_valuation_is_one_answer(kind, projection):
    product, solver, env = _knocked_out(kind, projection)
    stats = solver.calculate_event_stats(product, env)
    dist = solver.price_with_events(product, env).event_distribution
    _same_distribution(dist, EventDistribution.from_autocallable_stats(stats))
    assert stats.ko_probability[0] == 1.0 and stats.expected_discounted_maturity_cashflow == 0.0
    assert abs(stats.reconciliation_error) <= 1e-12
    assert solver.explain(product, env).terminal["event_stats"] == "knocked_out_at_valuation"
    if kind == "phoenix":
        assert stats.expected_discounted_coupon_cashflow[0] > 0.0      # the coupon paid with the knock-out


def test_carried_knock_in_with_disabled_knock_out_does_not_knock_out():
    product = _flag(_dkak(dated_snowball(CAL, T0)))
    env = flat_env(_first_obs(product), spot=130.0)
    solver = SnowballPDESolver(PDEParams())
    assert solver.explain(product, env).terminal["price"] == "marches"
    stats = solver.calculate_event_stats(product, env)
    assert stats.ko_probability[0] == 0.0


def test_carried_knock_in_ko_reset_knocks_out_on_its_post_schedule():
    product = _flag(dated_ko_reset(CAL, T0))
    env = flat_env(_first_obs(product), spot=100.0)                 # pre 103, post 95
    solver = KOResetSnowballPDESolver(PDEParams())
    decision = product.decide_observations_at_valuation(100.0, env, knocked_in=True)
    assert decision.knocked_out and decision.ko_regime == "post"
    assert solver.price(product, env) == pytest.approx(
        float(decision.ko_record.payoff) * env.get_discount_factor(decision.ko_record.settlement_time), abs=1e-12)


# ki_survive = P(knocked in AND reaches maturity without a knock-out); a note already knocked in (the
# lifecycle flag) reports the marched KIS column of its knocked-in surface (spec 3B §5.5), which is the
# no-knock-out survival 1 - sum(KO) up to the march's discrete discounting: measured 0 (round-off) at
# r = 0 and 3.29e-9 at the case's r = 2.5 % (standard; it grows as r^2 and shrinks 4x at high, the
# user's ruling of 2026-10-04 amending spec §7.2's 1e-12). Bound: twice the measurement.
@pytest.mark.parametrize("rate,bound", [(0.0, 1e-12), (None, 7e-9)])
def test_lifecycle_flag_marches_ki_survive(rate, bound):
    from pde_3a.products import env as make_env
    product = _flag(CASES["sb_cont"][0]())
    env = CASES["sb_cont"][1]() if rate is None else make_env(rate=rate)
    stats = SnowballPDESolver(PDEParams()).calculate_event_stats(product, env)
    assert stats.ki_ever_probability == 1.0
    assert abs(stats.ki_survive_knocked_in_probability - (1.0 - stats.ko_probability.sum())) <= bound
    assert stats.ki_probability == stats.ki_survive_knocked_in_probability     # settles knocked in
