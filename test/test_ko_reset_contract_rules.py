"""The knock-out-reset snowball's two contract rules (stated by the desk, 2026-09-21), on every day-level engine.

1. A knock-in REPLACES the original knock-out schedule by the second one (usually more dates at lower rates), and the
   knocked-in state stays to the end. So a knocked-in contract is a knocked-in snowball on the second schedule, and it
   can be priced at any time up to that schedule's end -- also after the first schedule has ended.
2. The second schedule only becomes effective through a knock-in DURING the first schedule. A contract not knocked in
   when the first schedule ends has matured there, so a knock-in is never tested after that.

Before this file: Monte Carlo kept testing discrete knock-in records after the first schedule had ended (rule 2) and
ignored a carried knock-in; QUAD V1 always priced the not-knocked-in state and refused a contract whose first schedule
had ended; and every engine priced a not-knocked-in contract past its first schedule as if that schedule ran on to
the final maturity.
"""
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.quad.v2 import KOResetSnowballQuadEngineV2, SnowballQuadEngineV2
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.option_enums import ObservationType
from quantark.util.exceptions import ValidationError

MONTHS = [i / 12.0 for i in range(1, 13)]
PRE_KO, POST_KO, KI = 103.0, 95.0, 75.0
#: (the knock-out-reset engine, the plain snowball engine of the same family)
ENGINES = {
    "mc": (lambda: SnowballMCEngine(params=MCParams(seed=7, num_paths=32768)),) * 2,
    "pde": (lambda: KOResetSnowballPDESolver(PDEParams()), lambda: SnowballPDESolver(PDEParams())),
    "quad_v1": (lambda: KOResetSnowballQuadEngine(params=QuadParams(grid_points=2001)),
                lambda: SnowballQuadEngine(params=QuadParams(grid_points=2001))),
    "quad_v2": (lambda: KOResetSnowballQuadEngineV2(), lambda: SnowballQuadEngineV2()),
}


def env(spot=90.0):
    return PricingEnvironment(rate_curve=FlatRateCurve(0.03), valuation_date=datetime(2026, 3, 16), spot_quote=SpotQuote(spot),
                              vol_surface=FlatVolSurface(0.25), div_yield=ContinuousDividendYield(0.01))


def _schedule(times, level):
    return ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=level) for t in times])


def ko_reset(pre, post, ki, maturity, knocked_in):
    """A time-based contract, aged as the library ages one: past records are dropped, so a first schedule that has
    ended is EMPTY. Cash is not annualized: 15 on the first schedule, 3 on the second, a 5 rebate."""
    product = KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0,
        barrier_config=BarrierConfig(ko_barrier=PRE_KO, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_schedule([0.5], PRE_KO), ki_barrier=KI,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=_schedule([0.5], KI)),
        post_barrier_config=BarrierConfig(ko_barrier=POST_KO, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=_schedule([1.0], POST_KO)),
        accrual_config=AccrualConfig(is_annualized=False), payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False))
    product.barrier_config = replace(product.barrier_config, ko_observation_schedule=_schedule(pre, PRE_KO),
                                     ki_observation_schedule=_schedule(ki, KI))
    product.post_barrier_config = replace(product.post_barrier_config, ko_observation_schedule=_schedule(post, POST_KO))
    product.maturity = maturity
    setattr(product, "_otc_lifecycle_knocked_in", knocked_in)
    return product


def knocked_in_snowball(ko_times, maturity):
    """Rule 1's other side: a knocked-in snowball whose knock-out schedule is the second one."""
    product = SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0,
        barrier_config=BarrierConfig(ko_barrier=POST_KO, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_schedule([1.0], POST_KO), ki_barrier=KI,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=_schedule([1.0], KI)),
        accrual_config=AccrualConfig(is_annualized=False), payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False))
    # already knocked in, so its knock-in records decide nothing; they sit on the knock-out dates so that they add
    # no node to a time grid (QUAD V1 will not take an empty knock-in schedule)
    product.barrier_config = replace(product.barrier_config, ko_observation_schedule=_schedule(ko_times, POST_KO),
                                     ki_observation_schedule=_schedule(ko_times, KI))
    product.maturity = maturity
    setattr(product, "_otc_lifecycle_knocked_in", True)
    return product


#: the first schedule ended 0.1y ago: what is left of the second, and of the contract
LATE = [t - 0.6 for t in MONTHS if t > 0.6]


@pytest.mark.parametrize("name", sorted(ENGINES))
def test_rule_2_a_knock_in_record_after_the_first_schedule_changes_nothing(name):
    """Knock-in records every month for a year against a six-month first schedule: months 7 to 12 are never tested,
    so trimming them leaves the price where it was. (Monte Carlo draws the same paths: the second schedule keeps the
    time grid identical.)"""
    make = ENGINES[name][0]
    full = float(make().price(ko_reset(MONTHS[:6], MONTHS, MONTHS, 1.0, False), env()))
    trimmed = float(make().price(ko_reset(MONTHS[:6], MONTHS, MONTHS[:6], 1.0, False), env()))
    assert full == pytest.approx(trimmed, abs=1e-10)


@pytest.mark.parametrize("name", sorted(ENGINES))
@pytest.mark.parametrize("first_schedule", ["live", "ended"])
def test_rule_1_a_knocked_in_contract_is_a_knocked_in_snowball_on_the_second_schedule(name, first_schedule):
    make_reset, make_plain = ENGINES[name]
    if first_schedule == "live":
        reset, plain = ko_reset(MONTHS[:6], MONTHS, MONTHS, 1.0, True), knocked_in_snowball(MONTHS, 1.0)
    else:
        reset, plain = ko_reset([], LATE, [], 0.4, True), knocked_in_snowball(LATE, 0.4)
    value, reference = float(make_reset().price(reset, env())), float(make_plain().price(plain, env()))
    # the same recursion on the same grid for the lattice engines; Monte Carlo shares its paths only when the first
    # schedule adds no dates to the time grid
    tolerance = 0.25 if (name, first_schedule) == ("mc", "live") else 2e-3
    assert value == pytest.approx(reference, abs=tolerance)
    assert value < -3.0                       # and nothing like the not-knocked-in price, which is positive here


@pytest.mark.parametrize("name", sorted(ENGINES))
def test_rule_1_the_engines_agree_on_a_knocked_in_contract_past_its_first_schedule(name):
    value = float(ENGINES[name][0]().price(ko_reset([], LATE, LATE, 0.4, True), env()))
    reference = float(KOResetSnowballPDESolver(PDEParams()).price(ko_reset([], LATE, LATE, 0.4, True), env()))
    assert value == pytest.approx(reference, abs=0.25 if name == "mc" else 5e-3)


@pytest.mark.parametrize("name", sorted(ENGINES))
def test_rule_2_a_contract_not_knocked_in_past_its_first_schedule_has_matured(name):
    """It is not alive, so there is nothing to price. Every engine used to price it as if the first schedule ran on to
    the final maturity, still testing knock-ins."""
    with pytest.raises(ValidationError, match="matured"):
        ENGINES[name][0]().price(ko_reset([], LATE, LATE, 0.4, False), env())


def test_monte_carlo_pays_the_rebate_on_the_spot_at_the_end_of_the_first_schedule():
    """A call-style rebate depends on where the spot is WHEN the contract matures: the end of the first schedule,
    not the end of the time grid."""
    def contract():
        product = ko_reset(MONTHS[:6], MONTHS, MONTHS[:6], 1.0, False)
        product.payoff_config = PayoffConfig(call_rebate_enabled=True, call_strike=100.0, call_participation_rate=1.0,
                                             include_principal=False)
        product.barrier_config = replace(product.barrier_config, ko_observation_schedule=_schedule(MONTHS[:6], 1e6))
        return product

    mc = SnowballMCEngine(params=MCParams(seed=7, num_paths=65536))
    value, error = float(mc.price(contract(), env(100.0))), float(mc.get_last_std_error())
    reference = float(KOResetSnowballQuadEngine(params=QuadParams(grid_points=2001)).price(contract(), env(100.0)))
    assert value == pytest.approx(reference, abs=4.0 * error + 5e-3)


# --- the knock-in instant: the second schedule is in force from the observation that knocks the contract in -------------
def _second_level_below_the_knock_in_level(pre, post, ki, maturity, level=60.0):
    """Only then can one spot both knock the contract in (S <= 75) and knock it out on the second schedule (S >= 60)."""
    product = ko_reset(pre, post, ki, maturity, False)
    product.post_barrier_config = replace(product.post_barrier_config, ko_barrier=level,
                                          ko_observation_schedule=_schedule(post, level))
    return product


@pytest.mark.parametrize("name", sorted(ENGINES))
def test_a_second_schedule_observation_at_the_knock_in_instant_counts(name):
    """The lattice engines hand a fresh knock-in to a knocked-in surface that has already applied this instant's
    second-schedule knock-out. Monte Carlo counted that schedule strictly AFTER the knock-in (4.60 against 4.75 here,
    seventeen standard errors), although its own REBASED branch tests an offset of zero at the knock-in instant."""
    make = ENGINES[name][0] if name != "mc" else (lambda: SnowballMCEngine(params=MCParams(seed=7, num_paths=131072)))
    engine = make()
    value = float(engine.price(_second_level_below_the_knock_in_level(MONTHS[:6], MONTHS, MONTHS[:6], 1.0), env(80.0)))
    reference = float(KOResetSnowballPDESolver(PDEParams()).price(
        _second_level_below_the_knock_in_level(MONTHS[:6], MONTHS, MONTHS[:6], 1.0), env(80.0)))
    tolerance = 4.0 * float(engine.get_last_std_error()) + 5e-3 if name == "mc" else 5e-3
    assert value == pytest.approx(reference, abs=tolerance)


@pytest.mark.parametrize("name", ["mc", "pde", "quad_v1"])
def test_the_same_rule_decides_a_contract_at_zero_time_to_maturity(name):
    """Both schedules and the knock-in observed now, spot 72: knocked in (<= 75) and out on the second schedule
    (>= 70) at once, which pays that schedule's 3. At 65 it is only knocked in, and redeems at the loss."""
    paid = _second_level_below_the_knock_in_level([0.0], [0.0], [0.0], 0.0, level=70.0)
    lost = _second_level_below_the_knock_in_level([0.0], [0.0], [0.0], 0.0, level=70.0)
    assert float(ENGINES[name][0]().price(paid, env(72.0))) == pytest.approx(3.0, abs=1e-12)
    assert float(ENGINES[name][0]().price(lost, env(65.0))) == pytest.approx(-35.0, abs=1e-12)


# --- the first schedule ends AT the valuation instant (review 2026-09-21 R8, R9) ---------------------------------------
def _first_schedule_ends_now(monitoring):
    """First schedule [0] (knock-out 103), second [0.5, 1] (knock-out 95), knock-in 75. Knocked in now it runs on the
    second schedule; not knocked in now it has matured here, and pays the 5 rebate now whatever the spot does next."""
    product = ko_reset([0.0], [0.5, 1.0], [0.0], 1.0, False)
    if monitoring == "continuous":
        product.barrier_config = replace(product.barrier_config, ki_observation_type=ObservationType.CONTINUOUS,
                                         ki_continuous=True, ki_observation_schedule=None)
    return product


@pytest.mark.parametrize("monitoring", ["discrete", "continuous"])
@pytest.mark.parametrize("name", ["mc", "pde", "quad_v1"])
def test_r9_r8_a_contract_not_knocked_in_when_its_first_schedule_ends_now_pays_its_rebate(name, monitoring):
    """QUAD V1 never seeded the not-knocked-in surface (0 for 5): its recursion has no step at time zero. Monte Carlo
    with a continuous knock-in watched the first FUTURE interval for it (-3.85 +/- 0.09): a later hit switched a claim
    that had already matured over to the second schedule."""
    assert float(ENGINES[name][0]().price(_first_schedule_ends_now(monitoring), env(90.0))) == 5.0


@pytest.mark.parametrize("monitoring", ["discrete", "continuous"])
@pytest.mark.parametrize("name", ["mc", "quad_v1"])
def test_r9_r8_a_knock_in_now_still_puts_the_second_schedule_in_force(name, monitoring):
    """At 70 the contract knocks in now and is a knocked-in snowball on the second schedule."""
    engine = ENGINES[name][0]()
    value = float(engine.price(_first_schedule_ends_now(monitoring), env(70.0)))
    reference = float(SnowballPDESolver(PDEParams()).price(knocked_in_snowball([0.5, 1.0], 1.0), env(70.0)))
    tolerance = 4.0 * float(engine.get_last_std_error()) if name == "mc" else 2e-3
    assert value == pytest.approx(reference, abs=tolerance)
    assert value < -20.0


def test_r8_a_path_after_the_first_schedule_ended_cannot_change_its_cash():
    """The payoff kernel on one fixed path, no sampling: 90 now, then 60 (below the knock-in level) and 60. The claim
    matured now at 90, so it pays the rebate now; it paid -40 at time 1."""
    engine = SnowballMCEngine(params=MCParams(seed=7, num_paths=16))
    product, market = _first_schedule_ends_now("continuous"), env(90.0)
    grid = engine._build_time_grid_ko_reset(product, market, 1.0)
    assert grid["ki_horizon_idx"] == -1 and list(grid["all_times"]) == [0.5, 1.0]
    cash, settlement, _, _ = engine._compute_payoffs_ko_reset(
        product, market, np.array([[90.0, 60.0, 60.0]]), grid, r=0.03, T=1.0, sigma=0.25, rng_seed=7)
    assert cash.tolist() == [5.0] and settlement.tolist() == [0.0]


# --- a continuous knock-in AT the valuation instant (review 2026-09-21 R10) ----------------------------------------------
def _knocked_in_now(monitoring, pre=(0.0,), pre_level=PRE_KO, post=(0.0, 0.5, 1.0), post_level=70.0):
    """First schedule ``pre`` (knock-out ``pre_level``), second ``post`` (knock-out ``post_level``, paying 3), knock-in
    75 observed now: discretely, continuously, or already carried in."""
    product = ko_reset(list(pre), list(post), [0.0], 1.0, monitoring == "carried")
    product.barrier_config = replace(product.barrier_config, ko_barrier=pre_level,
                                     ko_observation_schedule=_schedule(list(pre), pre_level))
    product.post_barrier_config = replace(product.post_barrier_config, ko_barrier=post_level,
                                          ko_observation_schedule=_schedule(list(post), post_level))
    if monitoring == "continuous":
        product.barrier_config = replace(product.barrier_config, ki_observation_type=ObservationType.CONTINUOUS,
                                         ki_continuous=True, ki_observation_schedule=None)
    return product


@pytest.mark.parametrize("monitoring", ["discrete", "continuous", "carried"])
@pytest.mark.parametrize("name", ["mc", "pde", "quad_v1"])
def test_r10_a_knock_in_now_puts_a_second_schedule_observation_now_in_force(name, monitoring):
    """Spot 72: the first schedule's 103 is not hit, the knock-in level 75 is, and the second schedule's 70 at the
    same instant is: 3 paid now, whatever the spot does next. Monte Carlo read a continuous knock-in now as the first
    future node, so the observation now was not in force (-12.32 +/- 0.12)."""
    assert float(ENGINES[name][0]().price(_knocked_in_now(monitoring), env(72.0))) == 3.0


@pytest.mark.parametrize("monitoring", ["discrete", "continuous", "carried"])
def test_r10_the_payoff_kernel_pays_a_knock_out_now_at_time_zero(monitoring):
    """One fixed path, no sampling: 72 now, then 60 and 60. It paid -40 at time 1 under a continuous knock-in."""
    engine = SnowballMCEngine(params=MCParams(seed=7, num_paths=16))
    product, market = _knocked_in_now(monitoring), env(72.0)
    grid = engine._build_time_grid_ko_reset(product, market, 1.0)
    cash, settlement, _, _ = engine._compute_payoffs_ko_reset(
        product, market, np.array([[72.0, 60.0, 60.0]]), grid, r=0.03, T=1.0, sigma=0.25, rng_seed=7)
    assert cash.tolist() == [3.0] and settlement.tolist() == [0.0]


@pytest.mark.parametrize("name", ["mc", "pde", "quad_v1"])
def test_r10_a_first_schedule_knock_out_now_still_wins_a_continuous_knock_in_now(name):
    """First-schedule level 60 under the knock-in level 75: spot 72 breaches both at the valuation instant. The
    knock-out wins, as the decision at valuation orders them (15 now); a continuous knock-in found INSIDE a step is
    the one that precedes the observation closing it."""
    product = _knocked_in_now("continuous", pre=(0.0, 0.5), pre_level=60.0)
    assert float(ENGINES[name][0]().price(product, env(72.0))) == 15.0


def test_r10_a_continuous_knock_in_now_continues_on_the_second_schedule():
    """Spot 58: knocked in now, no knock-out now; the claim runs on the second schedule on every engine."""
    product = lambda: _knocked_in_now("continuous", pre=(0.0, 0.5), pre_level=60.0)   # noqa: E731
    engine = SnowballMCEngine(params=MCParams(seed=7, num_paths=32768))
    value, error = float(engine.price(product(), env(58.0))), float(engine.get_last_std_error())
    reference = float(KOResetSnowballPDESolver(PDEParams()).price(product(), env(58.0)))
    assert reference < -30.0 and value == pytest.approx(reference, abs=4.0 * error)
