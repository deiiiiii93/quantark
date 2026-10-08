"""KOResetLifecycleTracker, one case per contract rule (spec 2026-10-08 KO-reset lifecycle, sections 2 and 4)."""
from dataclasses import replace

import pandas as pd
import pytest

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.v2 import KOResetSnowballQuadEngineV2
from quantark.asset.equity.lifecycle import KOResetLifecycleTracker, LifecycleEventType as E
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.accrual import KnockInAnchor
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.util.enum.option_enums import ObservationType, PostKOScheduleMode as M
from quantark.util.exceptions import ValidationError

from .conftest import START, env, schedule

D = 1 / 365.0


def kr(mode=M.ABSOLUTE, post=(273, 365), *, pre=(91, 182), ki=(91, 182), ki_level=75.0, post_level=95.0,
       annualized=False, disable=False, continuous=False):
    """Days on the numeric clock: schedule 1 at `pre` (103, 15%), KI at `ki`, schedule 2 at `post` (3%; offsets
    from the knock-in under REBASED); 5% rebate, no principal."""
    pre_t, ki_t, post_t = ([d * D for d in x] for x in (pre, ki, post))
    maturity = max(ki_t) + max(post_t) if mode == M.REBASED else max(post_t)
    barrier = BarrierConfig(
        ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
        ko_observation_schedule=schedule(pre_t, 103.0), ki_barrier=ki_level,
        ki_observation_type=ObservationType.CONTINUOUS if continuous else ObservationType.DISCRETE,
        ki_continuous=continuous, ki_observation_schedule=None if continuous else schedule(ki_t, ki_level),
        disable_ko_after_ki=disable)
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=maturity, tenor=maturity if annualized else None,
        barrier_config=barrier,
        post_barrier_config=BarrierConfig(ko_barrier=post_level, ko_rate=0.03,
                                          ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=schedule(post_t, post_level)),
        accrual_config=AccrualConfig(is_annualized=annualized, is_annualized_ki=annualized,
                                     is_annualized_rebate=annualized),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False), post_ko_mode=mode)


def day(n):
    return START + pd.Timedelta(days=n)


def run(tracker, closes):
    """closes: {day number: spot}, observed as PortfolioLifecycleManager does; [(day, event type, cashflow id)]."""
    out = []
    for n, spot in sorted(closes.items()):
        e = env(spot, day(n))
        product = tracker.product_for_lifecycle()
        events = tracker.observe(day(n), product, e, spot)
        maturity = tracker.settle_maturity_if_due(day(n), product, e, spot)
        events += [maturity] if maturity is not None else []
        out += [(n, ev.event_type, getattr(ev.realized_cashflow, "cashflow_id", None)) for ev in events]
        if not tracker.lifecycle.alive:
            break
    return out


def tracker(product, **kw):
    return KOResetLifecycleTracker(product=product, quantity=1.0, start_date=START, **kw)


def test_a_schedule_one_knock_out_ends_the_contract():
    t = tracker(kr())
    assert run(t, {91: 104.0}) == [(91, E.KNOCK_OUT, "knock-out:ko1:0")]
    assert not t.lifecycle.alive


def test_knock_out_wins_a_tie():
    t = tracker(kr(ki_level=105.0))                     # 104 breaches the knock-out AND the knock-in
    assert run(t, {91: 104.0}) == [(91, E.KNOCK_OUT, "knock-out:ko1:0")]
    assert not t.lifecycle.knocked_in


def test_a_knock_in_switches_and_tests_schedule_two_the_same_close():
    assert run(tracker(kr(post=(91, 365), post_level=70.0)), {91: 72.0}) == [
        (91, E.KNOCK_IN, None), (91, E.KNOCK_OUT, "knock-out:ko2:0")]


def test_a_knock_in_on_schedule_ones_last_observation_switches_instead_of_maturing():
    t = tracker(kr())
    assert run(t, {91: 90.0, 182: 70.0}) == [(182, E.KNOCK_IN, None)]
    assert t.lifecycle.alive and t.lifecycle.knocked_in


ENGINES = {
    "pde": lambda: KOResetSnowballPDESolver(PDEParams()),
    "quad_v1": lambda: KOResetSnowballQuadEngine(params=QuadParams(grid_points=2001)),
    "quad_v2": lambda: KOResetSnowballQuadEngineV2(),
    "mc": lambda: SnowballMCEngine(params=MCParams(seed=7, num_paths=4096)),
}


@pytest.mark.parametrize("engine", list(ENGINES.values()), ids=list(ENGINES))
def test_unknocked_at_schedule_ones_end_matures_with_v0_equal_to_the_engines(engine):
    t = tracker(kr(annualized=True))
    assert run(t, {91: 90.0, 182: 90.0}) == [(182, E.MATURITY, "maturity")]
    cash = t.lifecycle.ledger.cashflows[0].amount
    assert cash == pytest.approx(0.05 * 100.0 * 182 * D, rel=1e-12)
    fresh = tracker(kr(annualized=True))
    run(fresh, {91: 90.0})
    decided = engine().price(fresh.product_for_pricing(day(182), env(90.0, day(182)), initial=True),
                             env(90.0, day(182)))
    assert decided == pytest.approx(cash, rel=1e-9)


def test_a_schedule_one_maturity_settles_with_the_terminal_lag():
    from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
    product = kr()
    product.settlement_convention = SettlementConvention(lag=2 / 365.0, lag_unit=SettlementLagUnit.YEAR_FRACTION)
    product._validate_settlement_convention()
    t = tracker(product)
    run(t, {91: 90.0, 182: 90.0})
    cf = t.lifecycle.ledger.cashflows[0]
    assert cf.determination_time == pytest.approx(182 * D)
    assert cf.payment_time == pytest.approx(cf.determination_time + 2 / 365.0)


def test_continuous_knock_in_is_tested_on_schedule_ones_last_close():
    assert run(tracker(kr(continuous=True)), {182: 70.0}) == [(182, E.KNOCK_IN, None)]


def test_rebased_schedule_two_follows_the_knock_in_date():
    assert run(tracker(kr(M.REBASED, post=(91, 365))), {91: 70.0, 182: 96.0}) == [
        (91, E.KNOCK_IN, None), (182, E.KNOCK_OUT, "knock-out:ko2:0")]


@pytest.mark.parametrize("ki_day, maturity_day", [(91, 456), (182, 547)])
def test_a_knocked_in_rebased_contract_matures_on_its_own_tenor(ki_day, maturity_day):
    t = tracker(kr(M.REBASED, post=(91, 365), annualized=True))
    closes = {91: 90.0, ki_day: 70.0, maturity_day - 1: 80.0, maturity_day: 60.0}
    assert run(t, closes)[-1] == (maturity_day, E.MATURITY, "maturity")
    assert t.lifecycle.ledger.cashflows[-1].amount == pytest.approx(
        t.contract().get_maturity_payoff_v1(60.0, env(60.0), contract_tenor=maturity_day * D))


@pytest.mark.parametrize("carried", [False, True], ids=["fresh", "carried"])
def test_disable_ko_after_ki_tests_no_schedule_two_knock_out(carried):
    product = kr(post=(91, 273), post_level=70.0, disable=True)
    if carried:
        setattr(product, "_otc_lifecycle_knocked_in", True)
        assert run(tracker(product), {91: 99.0, 273: 99.0}) == [(273, E.MATURITY, "maturity")]
    else:
        assert run(tracker(product), {91: 72.0, 273: 99.0}) == [(91, E.KNOCK_IN, None),
                                                                  (273, E.MATURITY, "maturity")]


def test_a_due_day_on_a_holiday_fires_on_the_next_close():
    assert run(tracker(kr()), {90: 99.0, 93: 104.0}) == [(93, E.KNOCK_OUT, "knock-out:ko1:0")]


def test_holiday_last_observation_knock_in_switches():
    """Review focus 1: schedule 1's last record and a knock-in fall on a holiday: KI, not V0 maturity."""
    t = tracker(kr())
    assert run(t, {91: 90.0, 181: 90.0, 184: 70.0}) == [(184, E.KNOCK_IN, None)]
    assert t.lifecycle.alive


def test_first_close_consumes_nothing():
    """Review focus 2: a run whose first close comes after records fall due observes them all on it."""
    assert run(tracker(kr(post=(91, 365), post_level=70.0)), {95: 72.0}) == [
        (95, E.KNOCK_IN, None), (95, E.KNOCK_OUT, "knock-out:ko2:0")]


def test_schedule_two_records_before_the_knock_in_close_are_consumed():
    t = tracker(kr(post=(73, 365), post_level=70.0))          # 0.2 years
    assert run(t, {73: 90.0, 91: 90.0, 182: 72.0}) == [(182, E.KNOCK_IN, None)]
    assert 0 in t.lifecycle.observed_post_ko_indices


def test_a_schedule_two_record_delayed_by_a_missing_close_is_still_tested():
    assert run(tracker(kr()), {91: 70.0, 272: 99.0, 275: 96.0}) == [
        (91, E.KNOCK_IN, None), (275, E.KNOCK_OUT, "knock-out:ko2:0")]


def test_a_carried_knock_in_is_kept():
    product = kr()
    setattr(product, "_otc_lifecycle_knocked_in", True)
    t = tracker(product)
    assert t.lifecycle.knocked_in and t.product_for_lifecycle()._otc_lifecycle_knocked_in


def test_a_carried_rebased_knock_in_without_a_date_is_refused():
    product = kr(M.REBASED, post=(91, 365))
    setattr(product, "_otc_lifecycle_knocked_in", True)
    with pytest.raises(ValidationError, match="knock-in date"):
        tracker(product)


def test_carried_rebased_knock_in_before_start():
    """Review focus 5: knocked in 100 days before the run: schedule 2 at -100 + (91, 365) days; the record before
    the run is history (the resolution drops it, as the pricing copy does), never tested."""
    product = kr(M.REBASED, post=(91, 365))
    setattr(product, "_otc_lifecycle_knocked_in", True)
    setattr(product, "_otc_lifecycle_knock_in_date", (START - pd.Timedelta(days=100)).to_pydatetime())
    # a carried knock-in arrives with its anchor (accrual time spec §5.1): the record that fired, 100 days before
    # this product's numeric origin
    ki = product.barrier_config.ki_observation_schedule.records[0].accrual
    setattr(product, "_otc_lifecycle_knock_in_anchor", KnockInAnchor(0, -100 * D, ki.origin, ki.basis))
    t = tracker(product)
    copy = t.product_for_pricing(day(1), env(90.0, day(1)))
    assert [r.observation_time for r in copy.post_barrier_config.ko_observation_schedule.records] == [
        pytest.approx(264 * D)]
    assert run(t, {1: 99.0, 265: 80.0}) == [(265, E.MATURITY, "maturity")]


def test_a_knocked_in_rebased_pricing_copy_is_the_absolute_contract():
    t = tracker(kr(M.REBASED, post=(91, 365)))
    run(t, {91: 70.0})
    copy = t.product_for_pricing(day(100), env(90.0, day(100)))
    assert copy.post_ko_mode == M.ABSOLUTE and copy._otc_lifecycle_knocked_in
    assert copy.maturity == pytest.approx((91 + 365 - 100) * D)
    assert copy._otc_lifecycle_knock_in_date == day(91).to_pydatetime()


def test_an_absolute_pricing_copy_ages_schedule_two():
    t = tracker(kr())
    run(t, {91: 70.0})
    copy = t.product_for_pricing(day(100), env(90.0, day(100)))
    assert [r.observation_time for r in copy.post_barrier_config.ko_observation_schedule.records] == [
        pytest.approx(173 * D), pytest.approx(265 * D)]
