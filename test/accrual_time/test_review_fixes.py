"""Final-review findings on the accrual-time branch (I1-I6), each pinned by the case that exposed it."""
import time
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.v2.engine import SnowballQuadEngineV2
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine as _MC
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.accrual import (
    contract_tenor, coupon_period_fraction, knock_in_anchor_for, ko_accrual_factor,
)
from quantark.asset.equity.product.option.accrual_build import accrual_terms_key
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, PayoffConfig
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.day_counter import DayCountConvention as DC
from quantark.util.enum import PostKOScheduleMode as M
from quantark.util.exceptions import ValidationError

from .conftest import T0, dates, env, ko_reset, phoenix, rebuilt, snowball, times

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))


# -- I1: every KO-reset engine pays the explicit factor, annualized or not ------------------------------------------

def _factored_ko_reset():
    return ko_reset(accrual=AccrualConfig(is_annualized=False, accrual_factors=[0.4, 0.4],
                                          post_accrual_factors=[0.2, 0.2, 0.2]))


def test_mc_ko_reset_pays_the_explicit_factor_without_annualization():
    p, e = _factored_ko_reset(), env()
    profile = p.get_pre_ko_observation_profile(e)
    payoffs, _ = SnowballMCEngine(params=MCParams(seed=1, num_paths=64))._compute_ko_schedule_payoffs(
        p, np.array(profile["observation_times"]), np.array(profile["rates"]), profile["records"], e, 0.75)
    assert payoffs.tolist() == [100.0 + 100.0 * 0.12 * 0.4] * 2


@pytest.mark.parametrize("engine", [
    lambda: SnowballMCEngine(params=MCParams(seed=1, num_paths=512)),
    lambda: KOResetSnowballPDESolver(PDEParams()),
    lambda: KOResetSnowballQuadEngine(params=QuadParams(grid_points=401)),
], ids=["mc", "pde", "quad"])
def test_every_ko_reset_engine_pays_the_factor_at_the_valuation_instant(engine):
    """One answer everywhere (spec §8): an aged contract knocked out now pays principal + rate x factor."""
    p = ko_reset(pre=times(0.0, 0.25), accrual=AccrualConfig(is_annualized=False, accrual_factors=[0.4, 0.4],
                                                              accrued_offset=0.25), maturity=0.5)
    assert float(engine().price(p, env(spot=110.0))) == pytest.approx(100.0 + 100.0 * 0.12 * 0.4, abs=1e-12)


# -- I2: a re-validated KO-reset twin keeps its source contract's tenors -------------------------------------------

def test_a_revalidated_ko_reset_twin_keeps_the_contract_tenors():
    from intraday.conftest import dated_ko_reset, flat_env
    from quantark.intraday.events import EventKind, EventPhase, _schedule_env, resolve_timeline
    from quantark.intraday.fixings import Fixing
    from quantark.intraday.provisional import reconstruct_lifecycle
    from quantark.intraday.session import TradingSession, TradingSessionCalendar  # noqa: F401
    from quantark.intraday.twin import build_numerical_contract
    from intraday.conftest import sse_sessions as _sessions_fixture  # noqa: F401
    import inspect
    sessions = inspect.unwrap(_sessions_fixture)(CAL)
    prod = dated_ko_reset(CAL, datetime(2026, 3, 16))
    prod.tenor = 1.0
    prod.accrual_config = replace(prod.accrual_config, is_annualized_rebate=True, is_annualized_ki=True)
    prod.validate()
    want = (contract_tenor(prod, None, "pre"), contract_tenor(prod, None, "post"))
    events = resolve_timeline(prod, sessions, flat_env(datetime(2026, 4, 1))).events
    first_ko = min(ev.timestamp for ev in events if ev.kind is EventKind.KO)
    ts = first_ko - timedelta(hours=1)     # before any event: no fixing, no checkpoint needed
    e = flat_env(ts)
    tl = resolve_timeline(prod, sessions, e)
    fixings = []
    rec = reconstruct_lifecycle(prod, tl, None, fixings, valuation_timestamp=ts,
                                phase=EventPhase.AFTER, spot=100.0, spot_timestamp=ts,
                                schedule_env=_schedule_env(prod, e), session_calendar=sessions)
    twin = build_numerical_contract(prod, tl, rec, valuation_timestamp=ts, phase=EventPhase.AFTER,
                                    session_calendar=sessions, schedule_env=_schedule_env(prod, e)).product
    twin.validate()   # QUAD V2's compile_contract re-validates the product it is handed
    assert (contract_tenor(twin, None, "pre"), contract_tenor(twin, None, "post")) == want


# -- I3: a rebuild re-derives what its terms changed ----------------------------------------------------------------

def test_a_re_dated_record_re_derives_its_cash():
    p = snowball(ko=dates(30), initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=90))
    p.barrier_config.ko_observation_schedule.records[0].observation_date = T0 + timedelta(days=90)
    p.validate()
    assert ko_accrual_factor(p, p.barrier_config.ko_observation_schedule.records[0]) == 90 / 365


def test_a_rebuilt_numeric_terminal_follows_its_maturity():
    p = snowball(accrual=AccrualConfig(is_annualized_rebate=True), initial_date=T0)
    q = rebuilt(p, maturity=1.0)
    assert contract_tenor(q) == 1.0


def test_an_aged_terminal_stays_the_contracts():
    p = snowball(accrual=AccrualConfig(accrued_offset=0.0))
    p.time_shift(1 / 24, T0 + timedelta(days=15), env())
    q = rebuilt(p)
    assert contract_tenor(q) == pytest.approx(0.25, abs=1e-15)


def test_a_rebuilt_ko_reset_pre_terminal_follows_schedule_one():
    p = ko_reset()
    q = deepcopy(p)
    q.barrier_config = replace(q.barrier_config, ko_observation_schedule=times(0.25, 0.6))
    q.validate()
    assert contract_tenor(q, None, "pre") == 0.6


def test_re_deriving_an_aged_dated_phoenix_keeps_its_first_remaining_period():
    p = phoenix(ko=dates(30, 60, 90), initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=90))
    p.time_shift(35 / 365, T0 + timedelta(days=35), env())
    q = rebuilt(p, coupon_config=replace(p.coupon_config, day_count_convention=DC.ACT_360))
    assert coupon_period_fraction(q, q.barrier_config.ko_observation_schedule.records[0]) == 30 / 360


# -- I4: in-place changes to an override are refused, not paid stale -----------------------------------------------

def test_an_in_place_fixed_coupon_fraction_change_is_refused():
    p = phoenix(coupon=CouponBarrierConfig(coupon_barrier=80.0, coupon_rate=0.12, fixed_coupon_year_fraction=1 / 12))
    p.coupon_config = replace(p.coupon_config, fixed_coupon_year_fraction=0.5)
    with pytest.raises(ValidationError, match="validate"):
        p.get_coupon_payoff(0)


def test_an_in_place_factor_change_is_refused_and_ageing_is_not():
    p = snowball(accrual=AccrualConfig(accrual_factors=[0.1, 0.2, 0.3]))
    aged = deepcopy(p)
    aged.time_shift(1.5 / 12, T0 + timedelta(days=45), env())
    assert [ko_accrual_factor(aged, r) for r in aged.barrier_config.ko_observation_schedule.records] == [0.2, 0.3]
    p.accrual_config = replace(p.accrual_config, accrual_factors=[0.5, 0.5, 0.5])
    with pytest.raises(ValidationError, match="validate"):
        ko_accrual_factor(p, p.barrier_config.ko_observation_schedule.records[0])


def test_an_in_place_tenor_change_is_refused():
    p = snowball(tenor=1.0)
    p.tenor = 2.0
    with pytest.raises(ValidationError, match="validate"):
        contract_tenor(p)


# -- I5: the stale-terms check is cheap with a calendar ------------------------------------------------------------

def test_the_terms_key_is_cheap_with_a_calendar():
    p = snowball(ko=dates(30, 60), initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=60),
                 annualization_day_count=DC.BUSINESS_DAYS,
                 accrual=AccrualConfig(accrual_calendar=CAL, accrual_bus_days_in_year=244.0))
    accrual_terms_key(p)
    start = time.perf_counter()
    for _ in range(2000):
        accrual_terms_key(p)
    assert (time.perf_counter() - start) / 2000 < 20e-6


# -- I6: the spec's controls ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("shift_days", [1, 10, 31, 45, 70])
def test_phoenix_coupons_and_tenors_are_invariant_under_any_shift(shift_days):
    p = phoenix(accrual=AccrualConfig(is_annualized_rebate=True), initial_date=T0)
    coupons = [p.get_coupon_payoff(i) for i in range(3)]
    tenor = contract_tenor(p)
    aged = deepcopy(p)
    aged.time_shift(shift_days / 365, T0 + timedelta(days=shift_days), env())
    n = len(aged.barrier_config.ko_observation_schedule.records)
    assert [aged.get_coupon_payoff(i) for i in range(n)] == coupons[3 - n:]
    assert contract_tenor(aged) == pytest.approx(tenor, abs=1e-15)


@pytest.mark.parametrize("mode", [M.ABSOLUTE, M.REBASED])
def test_ko_reset_cash_and_tenors_are_invariant_under_ageing(mode):
    """The fix behind the 27 moved KO-reset bump thetas: ageing never grows schedule 2's cash or tenors."""
    p = ko_reset(mode)
    anchor = knock_in_anchor_for(p.barrier_config.ki_observation_schedule.records[0], 0)
    post = list(p.post_barrier_config.ko_observation_schedule.records)
    before = ([ko_accrual_factor(p, r, knock_in=anchor) for r in post], contract_tenor(p, None, "pre"),
              contract_tenor(p, None, "post", knock_in=anchor))
    p.time_shift(1 / 365, T0 + timedelta(days=1), env())
    post = list(p.post_barrier_config.ko_observation_schedule.records)
    after = ([ko_accrual_factor(p, r, knock_in=anchor) for r in post], contract_tenor(p, None, "pre"),
             contract_tenor(p, None, "post", knock_in=anchor))
    assert after == before


def test_quad_v2_pays_the_resolvers_cash_at_the_valuation_instant():
    p = snowball(ko=times(0.0, 0.25), maturity=0.25, accrual=AccrualConfig(accrued_offset=0.25))
    cash = 100.0 + 100.0 * 0.12 * ko_accrual_factor(p, p.barrier_config.ko_observation_schedule.records[0])
    assert float(SnowballQuadEngineV2().price(p, env(spot=110.0))) == pytest.approx(cash, abs=1e-12)


def test_a_dated_rebased_contract_without_initial_date_needs_no_anchor():
    """Review Focus 5 with NONE knock-in periods (dated, no initial_date, nothing annualized)."""
    from quantark.asset.equity.product.option.accrual import AccrualKind
    p = ko_reset(M.REBASED, pre=dates(91, 182), ki=dates(91, 182, level=75.0),
                 accrual=AccrualConfig(is_annualized=False), maturity=(182 + 270) / 365)
    assert {r.accrual.kind for r in p.barrier_config.ki_observation_schedule.records} == {AccrualKind.NONE}
    assert ko_accrual_factor(p, p.post_barrier_config.ko_observation_schedule.records[0]) == 1.0
    assert np.isfinite(float(SnowballMCEngine(params=MCParams(seed=3, num_paths=1024)).price(p, env())))


def test_mc_refuses_a_carried_rebased_knock_in():
    p = ko_reset(M.REBASED)
    setattr(p, "_otc_lifecycle_knocked_in", True)
    with pytest.raises(ValidationError, match="REBASED"):
        SnowballMCEngine(params=MCParams(seed=3, num_paths=256)).price(p, env())


def test_phoenix_factor_precedence():
    p = phoenix(ko=times(0.25), maturity=0.25, accrual=AccrualConfig(accrual_factors=[0.4], is_annualized_ko=False))
    assert ko_accrual_factor(p, p.barrier_config.ko_observation_schedule.records[0]) == 0.4
