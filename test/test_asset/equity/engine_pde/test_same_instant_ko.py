"""Same-instant knock-out records on the PDE core (0.16.0 DCN/dated run, T2: spec D3/D4/D5/D8/D10).

A dated Snowball on a US BUSINESS_DAYS env with PRECEDING holiday adjustment: 2025-07-04 is a US holiday, so its
knock-out record sits on 2025-07-03's env instant. Owner ruling (D1): the records of one instant are tested at the
same spot in contract order; the first whose barrier is breached knocks out and pays its own payoff. The PDE core
(1D, local-vol and Heston/Heston-SLV, all through ``SnowballAdapter._event_schedule``) stages such a group as ONE
piecewise event (D3), identifies records by contract index (D4), replays the group in the forward adjoint from a
``"ko_group"`` evidence row (D5), attributes each knock-out to the record that fired it (D8) and refuses to fold cash
onto a group of two or more effective records (D10). Phoenix and KO-reset (T8) are covered in
``test_same_instant_phoenix_ko_reset.py``; their AC7 rows here are pinned after T8.
"""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest
from scipy.stats import norm

from quantark.asset.equity.engine.mc import HestonSnowballMCEngine, LocalVolSnowballMCEngine, SnowballMCEngine
from quantark.asset.equity.engine.pde import HestonSnowballPDESolver, LocalVolSnowballPDESolver, SnowballPDESolver
from quantark.asset.equity.engine.pde.core.march import march
from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest, market_scope, prepare
from quantark.asset.equity.engine.pde.core.statistics import adjoint
from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.cashleg.folding import FoldedEventCash, FoldOverlay
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.vol.vol_surface import GridVolSurface
from quantark.priceenv import CurveQuotingBasis, PricingEnvironment
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.enum.engine_enums import EventProjectionMode
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams

US = create_calendar(CalendarType.US, year_range=(2020, 2030))
INITIAL = datetime(2025, 1, 3)
JUL3, JUL4 = datetime(2025, 7, 3), datetime(2025, 7, 4)
EXERCISE = datetime(2026, 1, 5)
REPRO_DATES = [datetime(2025, 3, 3), datetime(2025, 5, 5), JUL3, JUL4, datetime(2025, 9, 3), datetime(2025, 11, 3)]
R, Q, SIG = 0.02, 0.0, 0.25
TQ = CurveQuotingBasis.TRADING_YEAR
HESTON = HestonParams(v0=0.0625, kappa=1.5, theta=0.0625, sigma=0.3, rho=-0.5)
#: the repro's Heston grid (``.ship/.../tmp/r4_repro.py``)
HESTON_GRID = dict(n_x=64, n_v=16, n_t=64)


def env(valuation=INITIAL, spot=100.0, grid=False):
    vol = FlatVolSurface(SIG)
    if grid:   # the local-vol engines build Dupire from a grid surface: a flat one is the Black-Scholes model
        strikes, mats = [50.0, 75.0, 100.0, 125.0, 150.0], [0.25, 0.5, 1.0, 1.5, 2.0]
        vol = GridVolSurface(strikes=strikes, maturities=mats, iv_grid=np.full((len(mats), len(strikes)), SIG))
    return PricingEnvironment(spot_quote=SpotQuote(spot), vol_surface=vol, rate_curve=FlatRateCurve(R),
                              div_yield=ContinuousDividendYield(Q), valuation_date=valuation,
                              day_count_convention=DayCountConvention.BUSINESS_DAYS, calendar=US,
                              rate_curve_quoting=TQ, div_yield_quoting=TQ)


def snowball(records, ki=75.0, exercise=EXERCISE, reverse=False):
    p = create_standard_snowball(100.0, 100.0, 1.0, ko_barrier=97.0 if reverse else 103.0, ki_barrier=75.0,
                                 ko_rate=0.15, is_reverse=reverse)
    p.barrier_config = replace(p.barrier_config, ki_barrier=ki, ko_observation_dates=None,
                               ko_observation_schedule=ObservationSchedule(records=records))
    p.maturity = None
    p.initial_date = INITIAL
    p.exercise_date = exercise
    p.validate()
    return p


def repro(jul4_barrier=100.0, jul4_rate=0.10, drop_jul4=False):
    """The repro product: KO 03-03/05-05/07-03(103)/07-04/09-03/11-03, continuous KI 75 (07-04 step-down by default)."""
    recs = []
    for d in REPRO_DATES:
        if d == JUL4:
            if not drop_jul4:
                recs.append(ObservationRecord(observation_date=d, barrier=jul4_barrier, return_rate=jul4_rate))
        else:
            recs.append(ObservationRecord(observation_date=d, barrier=103.0))
    return snowball(recs)


def pair(b1=103.0, b2=100.0, r2=0.10):
    return [ObservationRecord(observation_date=JUL3, barrier=b1),
            ObservationRecord(observation_date=JUL4, barrier=b2, return_rate=r2)]


def _p_breach(b, t, reverse=False):
    """Risk-neutral BS probability of the breach region at env time ``t`` (up: S_t >= b; reverse: S_t <= b)."""
    up = norm.cdf((math.log(100.0 / b) + (R - Q - 0.5 * SIG * SIG) * t) / (SIG * math.sqrt(t)))
    return 1.0 - up if reverse else up


def _maturity_branch(product, e):
    """M * DF(s_T): the fixed not-knocked-out redemption with no knock-in barrier (checked constant in spot)."""
    m = product.get_maturity_payoff_v0(100.0, e)
    assert product.get_maturity_payoff_v0(60.0, e) == m and product.get_maturity_payoff_v0(160.0, e) == m
    return m * e.get_discount_factor(resolve_terminal_timing(product, e).payment_time)


def _exact_pair_value(product, e, reverse=False):
    """AC2: V = P1 DF(s1) N(b1) + P2 DF(s2) [N(b2) - N(b1)] + M DF(s_T) [1 - N(b2)]."""
    r1, r2 = product.resolve_ko_observations(e)
    t1 = r1.observation_time
    assert t1 == r2.observation_time
    n1, n2 = _p_breach(r1.barrier, t1, reverse), _p_breach(r2.barrier, t1, reverse)
    df = e.get_discount_factor
    return (r1.payoff * df(r1.settlement_time) * n1 + r2.payoff * df(r2.settlement_time) * (n2 - n1)
            + _maturity_branch(product, e) * (1.0 - n2)), (n1, n2 - n1)


# --- AC2: the exact reference ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("reverse", [False, True], ids=["up", "reverse"])
def test_pair_matches_the_exact_reference_and_converges(reverse):
    """AC2 (1D PDE): the shared pair alone, no KI, fixed maturity redemption: rel 1e-3 at the default grid, and the
    error does not grow under grid refinement. The exact construction: ``ki_barrier=None`` (no knock-in), and the
    not-knocked-out maturity payoff is the fixed coupon (asserted spot-independent).

    Measured (rel error, fast / standard / high): up +5.2e-6 / +3.3e-8 / +9.4e-8, reverse +2.9e-6 / +5.2e-7 /
    +6.4e-8; the single 07-03 record alone +1.4e-6 / +7.9e-8 / -5.9e-7. From the default grid on, the error sits at the
    route's own floor (the single record's, ~1e-7, not monotone there), so refinement is checked from the coarse grid
    to the default and, beyond it, against that floor."""
    product = snowball(pair(97.0, 100.0) if reverse else pair(), ki=None, reverse=reverse)
    e = env()
    assert not product.has_ki_barrier
    ref, _ = _exact_pair_value(product, e, reverse)
    err = {acc: abs(SnowballPDESolver(PDEParams(accuracy=acc)).price(product, e) / ref - 1.0)
           for acc in ("fast", "standard", "high")}
    assert err["standard"] <= 1e-3, err
    assert err["standard"] <= err["fast"], err
    assert err["high"] <= max(err["standard"], 1e-6), err


def test_pair_nodal_projection_is_the_exact_contract_order_test():
    """D3 nodal mode: the group's masks applied in reverse contract order -- the reference within the nodal
    discretization's own first-order error."""
    product = snowball(pair(), ki=None)
    e = env()
    ref, _ = _exact_pair_value(product, e)
    nodal = SnowballPDESolver(PDEParams(event_projection=EventProjectionMode.NODAL)).price(product, e)
    assert abs(nodal / ref - 1.0) <= 1e-2, (nodal, ref)


def test_pair_at_maturity_matches_the_exact_reference():
    """AC5 maturity instant: the pair is the terminal knock-out (exercise on the 07-04 holiday, both records at tau);
    the same closed form with t1 = tau."""
    product = snowball(pair(), ki=None, exercise=JUL4)
    e = env()
    r1, r2 = product.resolve_ko_observations(e)
    assert r1.observation_time == r2.observation_time == product.get_maturity(e)
    ref, _ = _exact_pair_value(product, e)
    value = SnowballPDESolver(PDEParams()).price(product, e)
    assert abs(value / ref - 1.0) <= 1e-3, (value, ref)


# --- AC4: the duplicate invariant -------------------------------------------------------------------------------------

def _solvers(grid_env):
    return {
        "pde": (lambda: SnowballPDESolver(PDEParams()), env()),
        "lv_pde": (lambda: LocalVolSnowballPDESolver(PDEParams()), grid_env),
        "heston_pde": (lambda: HestonSnowballPDESolver(HESTON, **HESTON_GRID), env()),
    }


SOLVERS = _solvers(env(grid=True))


@pytest.mark.parametrize("name", list(SOLVERS))
@pytest.mark.parametrize("dup_rate", [0.15, 0.40], ids=["same_payoff", "other_payoff"])
def test_duplicate_record_prices_as_the_product_without_it(name, dup_rate):
    """AC4: a same-instant duplicate (same barrier, any payoff) is dominated and dropped: bitwise the product
    without it."""
    make, e = SOLVERS[name]
    assert make().price(repro(103.0, dup_rate), e) == make().price(repro(drop_jul4=True), e)


# Pinned BEFORE the T2 change (worktree 6ad50f8d): the products without a shared instant stay bitwise.
SINGLETON_PINS = {"pde": "-0x1.3db0bae502f1ap-3", "lv_pde": "-0x1.3db0bae50769bp-3",
                  "heston_pde": "-0x1.46767ce12deaep-2"}


@pytest.mark.parametrize("name", list(SOLVERS))
def test_product_without_a_shared_instant_is_bitwise(name):
    make, e = SOLVERS[name]
    assert make().price(repro(drop_jul4=True), e) == float.fromhex(SINGLETON_PINS[name])


# --- AC3: the step-down against MC ---------------------------------------------------------------------------------

MC_PATHS = 400_000
MC = {
    "pde": lambda: SnowballMCEngine(params=MCParams(num_paths=MC_PATHS, seed=7)),
    "lv_pde": lambda: LocalVolSnowballMCEngine(params=MCParams(num_paths=MC_PATHS, seed=7)),
    "heston_pde": lambda: HestonSnowballMCEngine(HESTON, params=MCParams(num_paths=MC_PATHS, seed=7)),
}


@pytest.mark.slow
@pytest.mark.parametrize("name", list(SOLVERS))
def test_step_down_agrees_with_mc(name):
    """AC3: the repro step-down (07-04 at 100 paying 10%) within 3 SE + 1e-3 relative of the same model's MC."""
    make, e = SOLVERS[name]
    value = make().price(repro(), e)
    mc = MC[name]()
    ref = mc.price(repro(), e)
    se = mc.get_last_std_error()
    assert abs(value - ref) <= 3.0 * se + 1e-3 * abs(ref), (value, ref, se)


# --- AC5: the valuation instant ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(SOLVERS))
@pytest.mark.parametrize("spot,deciding", [(105.0, 0), (101.0, 1)], ids=["first_band", "step_down_band"])
def test_valuation_instant_pays_the_deciding_record(name, spot, deciding):
    """AC5: valuation on 07-03 -- both records at t = 0 -- with the spot in a record's band: knocked out at
    valuation, paying THAT record's cash."""
    make, grid_env = SOLVERS[name]
    e = env(JUL3, spot=spot, grid=grid_env.vol_surface.__class__ is GridVolSurface)
    product = repro()
    records = [r for r in product.resolve_ko_observations(e) if r.observation_time == 0.0]
    assert len(records) == 2
    rec = records[deciding]
    expected = rec.payoff * e.get_discount_factor(rec.settlement_time)
    assert make().price(product, e) == pytest.approx(expected, rel=1e-14, abs=1e-14)


@pytest.mark.parametrize("spot,deciding", [(105.0, 0), (101.0, 1)], ids=["first_band", "step_down_band"])
def test_valuation_and_maturity_instant_pays_the_deciding_record(spot, deciding):
    """AC5: valuation on 07-03 with the exercise on the 07-04 holiday -- zero time to maturity, both records pending
    at the valuation instant: the deciding record's cash."""
    e = env(JUL3, spot=spot)
    product = snowball(pair(), ki=None, exercise=JUL4)
    rec = product.resolve_ko_observations(e)[deciding]
    expected = rec.payoff * e.get_discount_factor(rec.settlement_time)
    assert SnowballPDESolver(PDEParams()).price(product, e) == pytest.approx(expected, rel=1e-14, abs=1e-14)


# --- AC6: event statistics ------------------------------------------------------------------------------------------

def _observation_rows(stats):
    return list(stats.observations.probability)


def test_pair_statistics_attribute_each_record_exactly():
    """AC6 (exact): on the AC2 product each record's knock-out probability is its band's BS probability."""
    product = snowball(pair(), ki=None)
    e = env()
    _, (p1, p2) = _exact_pair_value(product, e)
    stats = SnowballPDESolver(PDEParams()).calculate_event_stats(product, e)
    probs = _observation_rows(stats)
    assert len(probs) == 2
    assert probs[0] == pytest.approx(p1, abs=1e-3)
    assert probs[1] == pytest.approx(p2, abs=1e-3)
    assert abs(stats.reconciliation_error) <= 1e-9 * abs(stats.pv)


@pytest.mark.parametrize("name", ["pde", "heston_pde"])
def test_dominated_duplicate_has_zero_probability(name):
    """AC6: the dominated record keeps its row with probability 0; the total equals the deduplicated product's."""
    make, e = SOLVERS[name]
    dup = make().calculate_event_stats(repro(103.0, 0.40), e)
    base = make().calculate_event_stats(repro(drop_jul4=True), e)
    probs, base_probs = _observation_rows(dup), _observation_rows(base)
    assert len(probs) == len(base_probs) + 1
    assert probs[3] == 0.0
    assert probs[:3] + probs[4:] == base_probs
    assert math.isclose(sum(probs), sum(base_probs), rel_tol=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(SOLVERS))
def test_step_down_statistics_agree_with_mc(name):
    """AC6: each record's knock-out probability and P(knock in before any knock-out) -- the knock-in path stream of
    the forward adjoint (1D) / the native ADI forward pass (Heston) -- within 3 SE of the same model's MC (binomial SE
    of a probability over the MC paths)."""
    make, e = SOLVERS[name]
    if name == "heston_pde":
        # the repro grid's 05-05 row (before the pair) sits 3.3 SE off Heston MC with or without the 07-04 record:
        # the route's own discretization there, not the group's; the doubled grid is within 3 SE on every row
        make = lambda: HestonSnowballPDESolver(HESTON, n_x=128, n_v=32, n_t=128)  # noqa: E731
    pde = make().calculate_event_stats(repro(), e)
    mc = MC[name]().calculate_event_stats(repro(), e)
    rows, mc_rows = _observation_rows(pde), _observation_rows(mc)
    assert len(rows) == len(mc_rows) == len(REPRO_DATES)
    for p, m in zip(rows, mc_rows):
        se = math.sqrt(m * (1.0 - m) / MC_PATHS)
        assert abs(p - m) <= 3.0 * se, (p, m, se)
    m = mc.knock_in.probability
    se = math.sqrt(m * (1.0 - m) / MC_PATHS)
    assert abs(pde.knock_in.probability - m) <= 3.0 * se, (pde.knock_in.probability, m, se)
    assert rows[3] > 0.01      # the step-down band fires (1D: 0.029)


@pytest.mark.parametrize("projection", [None, EventProjectionMode.NODAL], ids=["cell_average", "nodal"])
def test_forward_adjoint_reconstructs_every_column_through_the_group(projection):
    """D5: the forward pass replays the "ko_group" evidence row as the same map: the adjoint identity holds for every
    column of the step-down product (knock-out units, maturity units, KIS, maturity cash)."""
    solver = SnowballPDESolver(PDEParams(event_projection=projection))
    e = env()
    p = prepare(solver, repro(), e, OutputRequest(Operation.EVENT_STATS, event_stats=True))
    with market_scope(p.solver, p.env, p.market_fps), p.adapter.scope(p.product, p.env):
        result = march(p)
        forward = adjoint.forward_pass(p.statistics, result.states, result.captures, reconstruct=True)
        at = p.statistics._reader(result.states, result.captures)
    assert any(row[0] == "ko_group" for row in p.events.evidence())
    L = p.statistics.layout
    for c in [*L.ko, *L.maturity_unit, L.kis]:
        assert abs(forward.reconstructed[int(c)] - at(int(c))) <= 1e-10, (c,)
    for c in L.maturity:
        assert math.isclose(forward.reconstructed[int(c)], at(int(c)), rel_tol=1e-12, abs_tol=1e-12), (c,)


@pytest.mark.parametrize("reverse", [False, True], ids=["up", "reverse"])
def test_maturity_instant_pair_statistics_attribute_each_record_exactly(reverse):
    """CP1 follow-up (AC6 at the maturity instant): the pair is the terminal knock-out; each record's knock-out
    probability is its band's BS probability (the terminal group's statistics stage)."""
    records = pair(b1=97.0, b2=100.0) if reverse else pair()
    product = snowball(records, ki=None, exercise=JUL4, reverse=reverse)
    e = env()
    _, (p1, p2) = _exact_pair_value(product, e, reverse=reverse)
    stats = SnowballPDESolver(PDEParams()).calculate_event_stats(product, e)
    probs = _observation_rows(stats)
    assert len(probs) == 2
    assert probs[0] == pytest.approx(p1, abs=1e-5) and probs[1] == pytest.approx(p2, abs=1e-5)
    assert abs(stats.reconciliation_error) <= 1e-9 * max(1.0, abs(stats.pv))


@pytest.mark.parametrize("projection", [None, EventProjectionMode.NODAL], ids=["cell_average", "nodal"])
def test_forward_adjoint_reconstructs_through_a_terminal_group_with_knock_in(projection):
    """CP1 follow-up (D5 at the maturity instant): the forward pass replays the terminal group with a knock-in."""
    solver = SnowballPDESolver(PDEParams(event_projection=projection))
    e = env()
    product = snowball(pair(), ki=75.0, exercise=JUL4)
    p = prepare(solver, product, e, OutputRequest(Operation.EVENT_STATS, event_stats=True))
    with market_scope(p.solver, p.env, p.market_fps), p.adapter.scope(p.product, p.env):
        result = march(p)
        forward = adjoint.forward_pass(p.statistics, result.states, result.captures, reconstruct=True)
        at = p.statistics._reader(result.states, result.captures)
    L = p.statistics.layout
    for c in [*L.ko, *L.maturity_unit, L.kis]:
        assert abs(forward.reconstructed[int(c)] - at(int(c))) <= 1e-10, (c,)


@pytest.mark.parametrize("name", list(SOLVERS))
@pytest.mark.parametrize("spot,deciding", [(105.0, 0), (101.0, 1)], ids=["first_band", "step_down_band"])
def test_valuation_instant_statistics_put_probability_one_on_the_deciding_record(name, spot, deciding):
    """AC6: a knock-out decided at valuation puts probability 1 on the deciding record (by contract index; the rows
    are the records still ahead: 07-03, 07-04, 09-03, 11-03)."""
    make, grid_env = SOLVERS[name]
    e = env(JUL3, spot=spot, grid=grid_env.vol_surface.__class__ is GridVolSurface)
    stats = make().calculate_event_stats(repro(), e)
    probs = _observation_rows(stats)
    assert probs[deciding] == 1.0
    assert sum(probs) == 1.0


# --- AC7: Phoenix and KO-reset (moved by T8) -------------------------------------------------------------------------

def _phoenix():
    from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
    from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
    from quantark.util.enum.option_enums import ObservationType

    p = PhoenixOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0, initial_date=INITIAL,
        exercise_date=datetime(2025, 11, 3),
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=ObservationSchedule(records=repro().barrier_config.ko_observation_schedule.records),
            ki_barrier=75.0, ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=ObservationSchedule(
                records=[ObservationRecord(observation_date=d, barrier=75.0) for d in REPRO_DATES])),
        coupon_config=CouponBarrierConfig(coupon_barrier=80.0, coupon_rate=0.12, memory_coupon=True),
        payoff_config=PayoffConfig(include_principal=True, rebate_rate=0.02),
    )
    p.validate()
    return p


def _ko_reset():
    from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
    from quantark.util.enum.option_enums import ObservationType

    post = [datetime(2025, 8, 4), datetime(2025, 10, 3), datetime(2025, 12, 3), EXERCISE]
    p = KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0, initial_date=INITIAL, exercise_date=EXERCISE,
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=ObservationSchedule(records=repro().barrier_config.ko_observation_schedule.records),
            ki_barrier=75.0, ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=ObservationSchedule(
                records=[ObservationRecord(observation_date=d, barrier=75.0) for d in sorted(set(REPRO_DATES + post))])),
        post_barrier_config=BarrierConfig(
            ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=ObservationSchedule(
                records=[ObservationRecord(observation_date=d, barrier=95.0) for d in post])),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False),
    )
    p.validate()
    return p


def _ac7_routes():
    from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
    from quantark.asset.equity.engine.pde.ko_reset_vol_pde_solvers import HestonKOResetSnowballPDESolver
    from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
    from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import HestonPhoenixPDESolver, LocalVolPhoenixPDESolver

    return {
        # name: (solver, product, env, pinned price AFTER T8 (D14 zero-gap on the Phoenix/KO-reset PDE); before T8, at
        # 6ad50f8d..9fc56f6c: 0x1.91bfe86d1fac6p+6 / 0x1.91bfe86d1fbccp+6 / 0x1.901d7efae0b82p+6 / -0x1.042eb5ce01d20p-3 /
        # -0x1.b0b937aeea858p-2 (one record of each pair dropped))
        "phoenix_pde": (lambda: PhoenixPDESolver(PDEParams()), _phoenix, env, "0x1.938203ff9713ep+6"),
        "phoenix_lv_pde": (lambda: LocalVolPhoenixPDESolver(PDEParams()), _phoenix, lambda: env(grid=True),
                           "0x1.938203ff97246p+6"),
        "phoenix_heston_pde": (lambda: HestonPhoenixPDESolver(HESTON, n_x=48, n_v=12, n_t=48), _phoenix, env,
                               "0x1.91a5800c84806p+6"),
        "ko_reset_pde": (lambda: KOResetSnowballPDESolver(PDEParams()), _ko_reset, env, "0x1.932c2d753031ap-5"),
        "ko_reset_heston_pde": (lambda: HestonKOResetSnowballPDESolver(HESTON, n_x=96, n_v=16, n_t=64), _ko_reset,
                                env, "-0x1.f6cbcbce1d934p-3"),
    }


AC7 = _ac7_routes()


@pytest.mark.parametrize("name", list(AC7))
def test_phoenix_and_ko_reset_with_a_same_instant_pair_are_pinned(name):
    """AC7: T2 left Phoenix and KO-reset untouched; T8 (D14 zero-gap rule, ``test_same_instant_phoenix_ko_reset.py``)
    moved them -- pinned after T8 so a later silent move fails here."""
    make, product, make_env, pin = AC7[name]
    assert make().price(product(), make_env()) == float.fromhex(pin)


@pytest.mark.parametrize("name", ["phoenix_pde", "ko_reset_pde"])
def test_phoenix_and_ko_reset_knock_in_pair_statistics(name):
    """These products also observe a discrete knock-in on both 07-03 and 07-04: the knock-in table still refuses the
    repeated time (an open item, decided in T5); the knock-out (and coupon) rows are reported per record and reconcile."""
    from quantark.cashleg.event_distribution import EventType

    make, product, make_env, _ = AC7[name]
    with pytest.raises(ValidationError, match="KnockInTable.times has duplicate rows"):
        make().calculate_event_stats(product(), make_env())
    stats = make().calculate_event_stats(product(), make_env(), streams=frozenset({EventType.KO, EventType.COUPON}))
    assert abs(stats.reconciliation_error) <= 1e-9 * max(1.0, abs(stats.pv))
    assert stats.observations.dates is not None and JUL4 in stats.observations.dates


# --- AC8: cash-leg folding ------------------------------------------------------------------------------------------

def _overlay(product, e, ko_amounts):
    """A hand overlay: per-unit cash on each knock-out record (contract order) and on the maturity branches."""
    records = sorted(product.resolve_ko_observations(e), key=lambda r: r.observation_time)
    timing = resolve_terminal_timing(product, e)
    det, pay = float(timing.determination_time), float(timing.payment_time)
    return FoldOverlay(
        ko=tuple(FoldedEventCash("main", float(r.observation_time), float(r.settlement_time), x)
                 for r, x in zip(records, ko_amounts)),
        maturity_no_ki=FoldedEventCash("main", det, pay, 0.25),
        maturity_ki=FoldedEventCash("main", det, pay, -0.5),
        constant_pv=0.0,
    )


@pytest.mark.parametrize("name", ["pde", "heston_pde"])
def test_folding_onto_a_two_record_group_is_refused(name):
    """AC8 / D10: the binder is one-to-one by time; a group of two effective records is refused, naming it."""
    make, e = SOLVERS[name]
    product = repro()
    with pytest.raises(ValidationError, match="2025-07-03.*2025-07-04"):
        make().price_folded(product, e, _overlay(product, e, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


@pytest.mark.parametrize("name", ["pde", "heston_pde"])
def test_folding_onto_a_duplicate_only_group_binds_the_effective_record(name):
    """AC8: a duplicate-only group binds its effective record and prices as the deduplicated product."""
    make, e = SOLVERS[name]
    dup, dedup = repro(103.0, 0.40), repro(drop_jul4=True)
    folded = make().price_folded(dup, e, _overlay(dup, e, [1.0, 2.0, 3.0, 9.0, 5.0, 6.0]))
    assert folded == make().price_folded(dedup, e, _overlay(dedup, e, [1.0, 2.0, 3.0, 5.0, 6.0]))
    assert folded != make().price(dedup, e)
