"""Same-instant knock-out records on QUAD V1 (0.16.0 DCN/dated run, T3: spec D1/D2/D4/D6/D8/D14).

A dated note on a US BUSINESS_DAYS env with PRECEDING holiday adjustment: 2025-07-04 is a US holiday, so its record
sits on 2025-07-03's env instant. Owner ruling (D1, D14 zero-gap rule): the records of one instant are tested at the
same spot in contract order, as consecutive observations with zero elapsed time.

* Snowball (D6): the observation recursion applies the instant's effective records (``same_instant_ko_groups``) as one
  piecewise breach -- the first record (contract order) whose barrier the spot breaches pays its own cash; the exact
  first step carries every effective barrier as a level; the statistics (stacked and forward) attribute each
  knock-out to its record, a dominated record keeping its row at probability 0.
* Phoenix (D14): every record of the instant is its own coupon date (never dropped): record k, if the note is still
  alive, pays its coupon (memory: plus the arrears since the last paid coupon) and then tests its knock-out.

The exact references, the products and the env are the PDE task's (``engine_pde/test_same_instant_ko.py``).
"""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from test_asset.equity.engine_pde.test_same_instant_ko import (
    JUL3,
    JUL4,
    REPRO_DATES,
    _exact_pair_value,
    _p_breach,
    env,
    pair,
    repro,
    snowball,
)
from quantark.asset.equity.engine.mc import PhoenixMCEngine, SnowballMCEngine
from quantark.asset.equity.engine.quad import PhoenixQuadEngine, SnowballQuadEngine
from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
from quantark.asset.equity.param import MCParams, QuadParams
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix
from quantark.util.enum.engine_enums import EventProjectionMode

#: the V1 lattice the tests price on (``QuadParams()``'s default)
GRID_POINTS = 1001
#: the lattice of the exact-reference tests (AC2 rel 1e-6): at 1001 nodes the reverse pair is 2.1e-6 off (the up pair
#: 1.7e-7; one record alone -1.3e-5 / -1.1e-5); at 4001, 9.7e-9 / 1.4e-7 (the knock-out probabilities 4.7e-7 / 3.2e-7)
GRID_EXACT = 4001
ROUTES = ("forward", "stacked")


def v1(grid_points=GRID_POINTS, **kwargs):
    return SnowballQuadEngine(params=QuadParams(grid_points=grid_points, **kwargs))


def v1_route(route, engine_cls=SnowballQuadEngine, grid_points=GRID_POINTS, **kwargs):
    engine = engine_cls(params=QuadParams(grid_points=grid_points, **kwargs))
    engine._v1_statistics_route = route
    return engine


def _rows(stats):
    return list(stats.observations.probability)


# --- AC2: the exact reference ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("reverse", [False, True], ids=["up", "reverse"])
def test_snowball_pair_matches_the_exact_reference(reverse):
    """AC2 (QUAD V1, grid_points=4001): the shared pair alone, no KI, fixed maturity redemption, rel 1e-6. The exact
    construction is the PDE task's: ``ki_barrier=None`` and the not-knocked-out redemption asserted spot-independent.
    Measured rel error (1001 / 2001 / 4001 / 8001 nodes): up +1.7e-7 / -3.4e-7 / +9.7e-9 / -2.8e-8, reverse
    +2.1e-6 / +5.0e-7 / +1.4e-7 / +2.4e-8."""
    product = snowball(pair(97.0, 100.0) if reverse else pair(), ki=None, reverse=reverse)
    e = env()
    ref, _ = _exact_pair_value(product, e, reverse)
    value = v1(GRID_EXACT).price(product, e)
    assert abs(value / ref - 1.0) <= 1e-6, (value, ref)


@pytest.mark.parametrize("reverse", [False, True], ids=["up", "reverse"])
def test_snowball_pair_nodal_events_are_the_contract_order_test(reverse):
    """D6 nodal mode: the records' breach weights in reverse contract order -- the reference within the nodal
    events' own first-order error (measured rel, 1001 nodes: up +7.3e-5, reverse +6.4e-5; the lattice aligns one
    barrier, the other's breach is a node mask)."""
    product = snowball(pair(97.0, 100.0) if reverse else pair(), ki=None, reverse=reverse)
    e = env()
    ref, _ = _exact_pair_value(product, e, reverse)
    value = v1(event_projection=EventProjectionMode.NODAL).price(product, e)
    assert abs(value / ref - 1.0) <= 5e-4, (value, ref)


def test_snowball_pair_as_the_first_observation_matches_the_exact_reference():
    """AC5 / D6: valuation on 07-02, the pair one business day ahead -- the exact first step prices it, with every
    effective barrier a level of its quadrature: the AC2 closed form at t1 = one day, rel 1e-6."""
    product = snowball(pair(), ki=None)
    e = env(datetime(2025, 7, 2))
    ref, _ = _exact_pair_value(product, e)
    engine = v1()
    value = engine.price(product, e)
    assert engine._last_first_step is not None
    assert abs(value / ref - 1.0) <= 1e-12, (value, ref)     # measured 1.6e-14


def test_snowball_pair_at_maturity_matches_the_exact_reference():
    """AC5 maturity instant: the pair is the terminal knock-out (exercise on the 07-04 holiday); rel 1e-6 at 4001
    nodes (measured 1001 / 2001 / 4001: -7.4e-7 / +3.0e-7 / +7.0e-8)."""
    product = snowball(pair(), ki=None, exercise=JUL4)
    e = env()
    r1, r2 = product.resolve_ko_observations(e)
    assert r1.observation_time == r2.observation_time == product.get_maturity(e)
    ref, _ = _exact_pair_value(product, e)
    value = v1(GRID_EXACT).price(product, e)
    assert abs(value / ref - 1.0) <= 1e-6, (value, ref)


# --- AC4: the duplicate invariant, and products without a shared instant -------------------------------------------

@pytest.mark.parametrize("dup_rate", [0.15, 0.40], ids=["same_payoff", "other_payoff"])
def test_snowball_duplicate_record_prices_as_the_product_without_it(dup_rate):
    """AC4: a same-instant duplicate (same barrier, any payoff) is dominated and dropped: bitwise."""
    e = env()
    assert v1().price(repro(103.0, dup_rate), e) == v1().price(repro(drop_jul4=True), e)


# Pinned BEFORE the T3 change (worktree e4ffd608): the products without a shared instant stay bitwise.
SNOWBALL_PIN = "-0x1.38663ea4a5adap-3"
SNOWBALL_ROW_PINS = {
    "forward": ["0x1.80415fbd50683p-2", "0x1.24d63e97900dfp-3", "0x1.2966a7b49f16dp-4", "0x1.7e1e7e9f8583ap-5",
                "0x1.0fea41ee53762p-5"],
    "stacked": ["0x1.80415fbd50684p-2", "0x1.24d63e97900e2p-3", "0x1.2966a7b49f16ep-4", "0x1.7e1e7e9f85839p-5",
                "0x1.0fea41ee5375fp-5"],
}


def test_snowball_without_a_shared_instant_prices_bitwise():
    assert v1().price(repro(drop_jul4=True), env()) == float.fromhex(SNOWBALL_PIN)


@pytest.mark.parametrize("route", ROUTES)
def test_snowball_without_a_shared_instant_reports_bitwise(route):
    stats = v1_route(route).calculate_event_stats(repro(drop_jul4=True), env())
    assert stats.pv == float.fromhex(SNOWBALL_PIN)
    assert [float(p).hex() for p in _rows(stats)] == SNOWBALL_ROW_PINS[route]


# --- AC5: the valuation instant ------------------------------------------------------------------------------------

@pytest.mark.parametrize("spot,deciding", [(105.0, 0), (101.0, 1)], ids=["first_band", "step_down_band"])
def test_snowball_valuation_instant_pays_the_deciding_record(spot, deciding):
    """AC5: valuation on 07-03 -- both records at t = 0 -- with the spot in a record's band: knocked out at
    valuation, paying THAT record's cash (and probability 1 on its row)."""
    e = env(JUL3, spot=spot)
    product = repro()
    records = [r for r in product.resolve_ko_observations(e) if r.observation_time == 0.0]
    assert len(records) == 2
    rec = records[deciding]
    expected = rec.payoff * e.get_discount_factor(rec.settlement_time)
    assert v1().price(product, e) == pytest.approx(expected, rel=1e-14, abs=1e-14)
    for route in ROUTES:
        probs = _rows(v1_route(route).calculate_event_stats(product, e))
        assert probs[deciding] == 1.0 and sum(probs) == 1.0, (route, probs)


@pytest.mark.parametrize("spot,deciding", [(105.0, 0), (101.0, 1)], ids=["first_band", "step_down_band"])
def test_snowball_valuation_and_maturity_instant_pays_the_deciding_record(spot, deciding):
    """AC5: valuation on 07-03, exercise on the 07-04 holiday: zero time to maturity, both records pending."""
    e = env(JUL3, spot=spot)
    product = snowball(pair(), ki=None, exercise=JUL4)
    rec = product.resolve_ko_observations(e)[deciding]
    expected = rec.payoff * e.get_discount_factor(rec.settlement_time)
    assert v1().price(product, e) == pytest.approx(expected, rel=1e-14, abs=1e-14)


# --- AC6: event statistics -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("reverse", [False, True], ids=["up", "reverse"])
def test_snowball_pair_statistics_attribute_each_record_exactly(route, reverse):
    """AC6 (exact): on the AC2 product each record's knock-out probability is its band's BS probability, abs 1e-6 at
    4001 nodes (measured: up +4.7e-7 / -3.7e-7, reverse +2.1e-7 / -3.2e-7; 1001 nodes: up +7.6e-6 / -5.9e-6)."""
    product = snowball(pair(97.0, 100.0) if reverse else pair(), ki=None, reverse=reverse)
    e = env()
    _, (p1, p2) = _exact_pair_value(product, e, reverse)
    stats = v1_route(route, grid_points=GRID_EXACT).calculate_event_stats(product, e)
    probs = _rows(stats)
    assert len(probs) == 2
    assert probs[0] == pytest.approx(p1, abs=1e-6) and probs[1] == pytest.approx(p2, abs=1e-6), (probs, p1, p2)
    assert abs(stats.reconciliation_error) <= 1e-9 * abs(stats.pv)


@pytest.mark.parametrize("route", ROUTES)
def test_snowball_pair_at_maturity_statistics_attribute_each_record_exactly(route):
    """The terminal group's statistics, abs 1e-6 at 4001 nodes (measured +2.5e-7 / -2.1e-7)."""
    product = snowball(pair(), ki=None, exercise=JUL4)
    e = env()
    _, (p1, p2) = _exact_pair_value(product, e)
    probs = _rows(v1_route(route, grid_points=GRID_EXACT).calculate_event_stats(product, e))
    assert probs[0] == pytest.approx(p1, abs=1e-6) and probs[1] == pytest.approx(p2, abs=1e-6), (probs, p1, p2)


@pytest.mark.parametrize("route", ROUTES)
def test_snowball_pair_as_the_first_observation_statistics_attribute_each_record_exactly(route):
    """The exact first step's statistics event carries the group too (measured 5e-15)."""
    product = snowball(pair(), ki=None)
    e = env(datetime(2025, 7, 2))
    _, (p1, p2) = _exact_pair_value(product, e)
    probs = _rows(v1_route(route).calculate_event_stats(product, e))
    assert probs[0] == pytest.approx(p1, abs=1e-12) and probs[1] == pytest.approx(p2, abs=1e-12), (probs, p1, p2)


@pytest.mark.parametrize("route", ROUTES)
def test_snowball_dominated_duplicate_has_zero_probability(route):
    """AC6: the dominated record keeps its row with probability 0; the other rows and the total equal the
    deduplicated product's."""
    e = env()
    dup = v1_route(route).calculate_event_stats(repro(103.0, 0.40), e)
    base = v1_route(route).calculate_event_stats(repro(drop_jul4=True), e)
    probs, base_probs = _rows(dup), _rows(base)
    assert len(probs) == len(base_probs) + 1
    assert probs[3] == 0.0
    assert probs[:3] + probs[4:] == base_probs
    assert math.isclose(sum(probs), sum(base_probs), rel_tol=1e-12)
    assert dup.pv == base.pv


def test_snowball_step_down_routes_agree():
    """The forward pass attributes the group's sources as the stacked columns do (adjoint identity)."""
    e = env()
    fwd = _rows(v1_route("forward").calculate_event_stats(repro(), e))
    stk = _rows(v1_route("stacked").calculate_event_stats(repro(), e))
    assert np.allclose(fwd, stk, rtol=0.0, atol=1e-12), (fwd, stk)
    assert fwd[3] > 0.01


# --- AC3 / AC6 against MC ------------------------------------------------------------------------------------------

MC_PATHS = 400_000


@pytest.fixture(scope="module")
def snowball_mc():
    """The repro step-down on Black-Scholes MC (400k paths, seed 7): price, SE and statistics."""
    mc = SnowballMCEngine(params=MCParams(num_paths=MC_PATHS, seed=7))
    value = mc.price(repro(), env())
    se = mc.get_last_std_error()
    stats = SnowballMCEngine(params=MCParams(num_paths=MC_PATHS, seed=7)).calculate_event_stats(repro(), env())
    return value, se, stats


@pytest.mark.slow
def test_snowball_step_down_agrees_with_mc(snowball_mc):
    """AC3: the repro step-down (07-04 at 100 paying 10%) within 3 SE + 1e-3 relative of MC."""
    ref, se, _ = snowball_mc
    value = v1().price(repro(), env())
    assert abs(value - ref) <= 3.0 * se + 1e-3 * abs(ref), (value, ref, se)


@pytest.mark.slow
@pytest.mark.parametrize("route", ROUTES)
def test_snowball_step_down_statistics_agree_with_mc(snowball_mc, route):
    """AC6: each record's knock-out probability within 3 SE of MC (binomial SE over the paths)."""
    _, _, mc_stats = snowball_mc
    rows, mc_rows = _rows(v1_route(route).calculate_event_stats(repro(), env())), _rows(mc_stats)
    assert len(rows) == len(mc_rows) == len(REPRO_DATES)
    for p, m in zip(rows, mc_rows):
        se = math.sqrt(m * (1.0 - m) / MC_PATHS)
        assert abs(p - m) <= 3.0 * se, (p, m, se)
    assert rows[3] > 0.01


# === Phoenix (D14: every record of the instant is its own coupon date) ==============================================

PHX_DATES = [datetime(2025, 3, 3), datetime(2025, 5, 5), JUL3, JUL4, datetime(2025, 9, 3), datetime(2025, 11, 3)]
COUPON_BARRIER = 80.0
#: the Phoenix exact-reference lattice: V1's own floor on ONE record (coupon 80, KO 103) is +2.3e-6 at 4001 nodes and
#: +5.5e-7 at 8001; the pair at 8001: +2.4e-7 (b2 < b1) / +1.8e-9 (b2 > b1) (4001: -3.0e-6 / -1.5e-6)
GRID_PHX_EXACT = 8001


def phoenix(records, ki=70.0):
    """A dated memory Phoenix (1% per period, not annualized, coupon barrier 80, no principal) on ``records``."""
    p = create_standard_phoenix(100.0, 100.0, 1.0, ko_barrier=103.0, ki_barrier=70.0, coupon_barrier=COUPON_BARRIER,
                                coupon_rate=0.01, num_observations=12, memory_coupon=True, is_annualized=False)
    p.barrier_config = replace(p.barrier_config, ki_barrier=ki, ko_observation_dates=None,
                               ko_observation_schedule=ObservationSchedule(records=records))
    p.maturity = None
    p.initial_date = datetime(2025, 1, 3)
    p.exercise_date = datetime(2026, 1, 5)
    p.validate()
    return p


def phoenix_repro(variant):
    """The run's Phoenix repro (``.ship/.../r4_phx_kor.py``): KO 103 on every date, KI 70; ``single`` drops 07-04,
    ``duplicate`` keeps it at 103, ``stepdown`` puts it at 100."""
    records = []
    for d in PHX_DATES:
        if d == JUL4:
            if variant == "single":
                continue
            records.append(ObservationRecord(observation_date=d, barrier=100.0 if variant == "stepdown" else 103.0))
        else:
            records.append(ObservationRecord(observation_date=d, barrier=103.0))
    return phoenix(records)


def phoenix_pair(b1, b2):
    """AC7: the pair alone (07-03 at ``b1``, 07-04 at ``b2`` paying 10%), no KI, memory on."""
    return phoenix([ObservationRecord(observation_date=JUL3, barrier=b1),
                    ObservationRecord(observation_date=JUL4, barrier=b2, return_rate=0.10)], ki=None)


def _zero_gap_cash(product, e, s):
    """The zero-gap rule's cash on a spot ``s`` at the shared instant, each leg discounted at its own settlement:
    record k (alive) pays its coupon plus the arrears since the last paid coupon where ``s`` meets the coupon barrier,
    then knocks out paying its redemption (plus the arrears carried into it when its coupon missed); else the
    maturity redemption (the arrears are paid by a coupon or a knock-out only)."""
    records = product.resolve_ko_observations(e)
    fractions = product.get_coupon_period_year_fractions(e)
    df = e.get_discount_factor
    cash, arrears = 0.0, 0.0
    for k, rec in enumerate(records):
        c = product.get_coupon_payoff(k, year_fraction=fractions[k])
        hit = s >= COUPON_BARRIER
        if hit:
            cash += (c + arrears) * df(rec.settlement_time)
        if s >= rec.barrier:
            return cash + (rec.payoff + (0.0 if hit else arrears)) * df(rec.settlement_time)
        arrears = 0.0 if hit else arrears + c
    m = product.get_maturity_payoff_v0(s, pricing_env=e)
    assert product.get_maturity_payoff_v0(60.0, pricing_env=e) == m == product.get_maturity_payoff_v0(160.0, pricing_env=e)
    return cash + m * df(resolve_terminal_timing(product, e).payment_time)


def _exact_phoenix_pair(product, e):
    """AC7: ``V = sum over the bands of S_t1 of P(band) x zero-gap cash``; also each record's knock-out and coupon
    probability (coupon: alive before the record and the coupon barrier met)."""
    r1, r2 = product.resolve_ko_observations(e)
    t1 = r1.observation_time
    assert t1 == r2.observation_time
    levels = sorted({COUPON_BARRIER, float(r1.barrier), float(r2.barrier)})
    edges = [0.0] + levels + [math.inf]
    above = lambda b: 0.0 if b == math.inf else (1.0 if b == 0.0 else _p_breach(b, t1))   # noqa: E731
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sample = 0.5 * (lo + hi) if hi < math.inf else lo + 1.0
        value += (above(lo) - above(hi)) * _zero_gap_cash(product, e, sample)
    b1, b2 = float(r1.barrier), float(r2.barrier)
    ko = (above(b1), max(above(b2) - above(b1), 0.0))
    coupon = (above(COUPON_BARRIER), above(COUPON_BARRIER) - above(b1))
    return value, ko, coupon


def phx(grid_points=GRID_POINTS, **kwargs):
    return PhoenixQuadEngine(params=QuadParams(grid_points=grid_points, **kwargs))


def _coupon_rows(stats):
    return list(stats.coupons.probability)


@pytest.mark.parametrize("b1,b2", [(103.0, 100.0), (100.0, 103.0)], ids=["b2_below_b1", "b2_above_b1"])
def test_phoenix_pair_matches_the_exact_reference(b1, b2):
    """AC7 (QUAD V1, grid_points=8001): the pair alone, no KI, memory on, rel 1e-6. ``b2 > b1``: record 2's
    knock-out is inert (its region lies inside record 1's) but its coupon still counts. Before the change V1 dropped
    the 07-04 record: rel error -0.56 / -0.31."""
    product = phoenix_pair(b1, b2)
    e = env()
    ref, _, _ = _exact_phoenix_pair(product, e)
    value = phx(GRID_PHX_EXACT).price(product, e)
    assert abs(value / ref - 1.0) <= 1e-6, (value, ref)


def test_phoenix_pair_nodal_events_follow_the_zero_gap_rule():
    """The nodal events apply the records in reverse contract order (first-order: one barrier is aligned)."""
    product = phoenix_pair(103.0, 100.0)
    e = env()
    ref, _, _ = _exact_phoenix_pair(product, e)
    value = phx(event_projection=EventProjectionMode.NODAL).price(product, e)
    assert abs(value / ref - 1.0) <= 1e-3, (value, ref)


def test_phoenix_pair_as_the_first_observation_matches_the_exact_reference():
    """The exact first step: valuation on 07-02; every record's knock-out and coupon barrier is a level."""
    product = phoenix_pair(103.0, 100.0)
    e = env(datetime(2025, 7, 2))
    ref, _, _ = _exact_phoenix_pair(product, e)
    engine = phx()
    value = engine.price(product, e)
    assert engine._last_first_step is not None
    assert abs(value / ref - 1.0) <= 1e-12, (value, ref)


# Pinned BEFORE the T3 change (worktree e4ffd608).
PHOENIX_PIN = "-0x1.695e6b15e0dbdp+0"
PHOENIX_ROW_PINS = {
    "forward": (["0x1.80415fbd50682p-2", "0x1.24d64730c2803p-3", "0x1.2966df7925555p-4", "0x1.7e1f32a0c2d28p-5",
                 "0x1.0feb0162982e6p-5"],
                ["0x1.f9bb323cadd62p-1", "0x1.1f743539bf739p-1", "0x1.8579ece10550cp-2", "0x1.2248f09b18930p-2",
                 "0x1.c6d0a9eca64abp-3"]),
    "stacked": (["0x1.80415fbd50684p-2", "0x1.24d64730c2808p-3", "0x1.2966df7925553p-4", "0x1.7e1f32a0c2d29p-5",
                 "0x1.0feb0162982e7p-5"],
                ["0x1.f9bb323cadd61p-1", "0x1.1f743539bf738p-1", "0x1.8579ece10550cp-2", "0x1.2248f09b18930p-2",
                 "0x1.c6d0a9eca64adp-3"]),
}


def test_phoenix_without_a_shared_instant_prices_bitwise():
    assert phx().price(phoenix_repro("single"), env()) == float.fromhex(PHOENIX_PIN)


@pytest.mark.parametrize("route", ROUTES)
def test_phoenix_without_a_shared_instant_reports_bitwise(route):
    stats = v1_route(route, PhoenixQuadEngine).calculate_event_stats(phoenix_repro("single"), env())
    assert stats.pv == float.fromhex(PHOENIX_PIN)
    ko, coupon = PHOENIX_ROW_PINS[route]
    assert [float(p).hex() for p in _rows(stats)] == ko
    assert [float(p).hex() for p in _coupon_rows(stats)] == coupon


@pytest.mark.parametrize("spot,deciding", [(105.0, 0), (101.0, 1)], ids=["first_band", "step_down_band"])
def test_phoenix_valuation_instant_pays_the_deciding_record(spot, deciding):
    """AC5: valuation on 07-03 with the pair at t = 0: knocked out at valuation by the deciding record, paying its
    redemption and the coupons of the records up to it (the product's own decision, zero-gap)."""
    e = env(JUL3, spot=spot)
    product = phoenix_repro("stepdown")
    decision = product.decide_observations_at_valuation(spot, e)
    records = [r for r in product.resolve_ko_observations(e) if r.observation_time == 0.0]
    assert decision.knocked_out and decision.ko_record.observation_date == records[deciding].observation_date
    expected = (decision.ko_record.payoff + decision.coupon) * e.get_discount_factor(decision.ko_record.settlement_time)
    assert phx().price(product, e) == pytest.approx(expected, rel=1e-14, abs=1e-14)
    for route in ROUTES:
        probs = _rows(v1_route(route, PhoenixQuadEngine).calculate_event_stats(product, e))
        assert probs[deciding] == 1.0 and sum(probs) == 1.0, (route, probs)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("b1,b2", [(103.0, 100.0), (100.0, 103.0)], ids=["b2_below_b1", "b2_above_b1"])
def test_phoenix_pair_statistics_attribute_each_record_exactly(route, b1, b2):
    """AC7 statistics (exact): each record's knock-out probability and coupon probability, abs 1e-6 at 8001 nodes;
    the ledger reconciles with the price."""
    product = phoenix_pair(b1, b2)
    e = env()
    _, ko, coupon = _exact_phoenix_pair(product, e)
    stats = v1_route(route, PhoenixQuadEngine, grid_points=GRID_PHX_EXACT).calculate_event_stats(product, e)
    assert _rows(stats) == pytest.approx(list(ko), abs=1e-6)
    assert _coupon_rows(stats) == pytest.approx(list(coupon), abs=1e-6)
    assert abs(stats.reconciliation_error) <= 1e-9 * abs(stats.pv)


@pytest.fixture(scope="module")
def phoenix_mc():
    """The Phoenix repro on Black-Scholes MC (400k paths, seed 7): per variant the price, its SE and statistics."""
    out = {}
    for variant in ("single", "duplicate", "stepdown"):
        mc = PhoenixMCEngine(params=MCParams(num_paths=MC_PATHS, seed=7))
        value = mc.price(phoenix_repro(variant), env())
        se = mc.get_last_std_error()
        stats = PhoenixMCEngine(params=MCParams(num_paths=MC_PATHS, seed=7)).calculate_event_stats(
            phoenix_repro(variant), env())
        out[variant] = (value, se, stats)
    return out


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["single", "duplicate", "stepdown"])
def test_phoenix_repro_agrees_with_mc(phoenix_mc, variant):
    """AC7: the repro Phoenix (KI 70) within 3 SE + 1e-3 relative of MC, single / duplicate / step-down."""
    ref, se, _ = phoenix_mc[variant]
    value = phx().price(phoenix_repro(variant), env())
    assert abs(value - ref) <= 3.0 * se + 1e-3 * abs(ref), (value, ref, se)


@pytest.mark.slow
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("variant", ["duplicate", "stepdown"])
def test_phoenix_repro_statistics_agree_with_mc(phoenix_mc, route, variant):
    """AC7: each record's knock-out and coupon probability within 3 SE of MC (binomial SE over the paths)."""
    _, _, mc_stats = phoenix_mc[variant]
    stats = v1_route(route, PhoenixQuadEngine).calculate_event_stats(phoenix_repro(variant), env())
    for rows, mc_rows in ((_rows(stats), _rows(mc_stats)), (_coupon_rows(stats), _coupon_rows(mc_stats))):
        assert len(rows) == len(mc_rows) == len(PHX_DATES)
        for p, m in zip(rows, mc_rows):
            se = math.sqrt(m * (1.0 - m) / MC_PATHS)
            assert abs(p - m) <= 3.0 * se, (rows, mc_rows)
    assert abs(stats.reconciliation_error) <= 1e-9 * max(1.0, abs(stats.pv))


@pytest.mark.slow
@pytest.mark.parametrize("spot", [75.0, 90.0], ids=["both_missed", "both_paid"])
def test_phoenix_valuation_instant_pair_without_knock_out_agrees_with_mc(spot):
    """Valuation on 07-03, the step-down pair at t = 0 and no knock-out there: the coupons decided now are paid, and
    the memory state the instant leaves counts the periods missed there after the last paid one (both missed: two
    periods of arrears; before the change one: -19.485 against MC -19.139, z = -12.7)."""
    e = env(JUL3, spot=spot)
    product = phoenix_repro("stepdown")
    mc = PhoenixMCEngine(params=MCParams(num_paths=MC_PATHS, seed=7))
    ref = mc.price(product, e)
    se = mc.get_last_std_error()
    value = phx().price(product, e)
    assert abs(value - ref) <= 3.0 * se + 1e-3 * abs(ref), (value, ref, se)


def _phoenix_variant(variant, pay, memory, ki):
    from quantark.util.enum import ObservationType

    p = phoenix_repro(variant)
    p.coupon_config = replace(p.coupon_config, coupon_pay_type=pay, memory_coupon=memory)
    if ki == "discrete":   # a discrete knock-in observed at the shared instant (07-03)
        p.barrier_config = replace(
            p.barrier_config, ki_observation_type=ObservationType.DISCRETE, ki_continuous=False,
            ki_observation_schedule=ObservationSchedule(
                records=[ObservationRecord(observation_date=d, barrier=70.0) for d in PHX_DATES if d != JUL4]))
    p.validate()
    return p


@pytest.mark.parametrize("ki", ["continuous", "discrete"])
@pytest.mark.parametrize("memory", [True, False], ids=["memory", "no_memory"])
@pytest.mark.parametrize("pay", ["INSTANT", "EXPIRY"])
@pytest.mark.parametrize("variant", ["duplicate", "stepdown"])
def test_phoenix_forward_route_transposes_the_group(variant, pay, memory, ki):
    """The forward route reads every record's coupon columns off the adjoint pass (``CouponGroupStep``: the group's
    regions projected, the records' chain pointwise in each) -- the stacked columns to round-off (measured 2.2e-16),
    without falling back to them."""
    from quantark.util.enum import CouponPayType

    product = _phoenix_variant(variant, getattr(CouponPayType, pay), memory, ki)
    e = env()
    forward_engine = v1_route("forward", PhoenixQuadEngine)
    forward = forward_engine.calculate_event_stats(product, e)
    assert forward_engine._last_v1_march.extension is not None       # the forward pass ran: no stacked fallback
    stacked = v1_route("stacked", PhoenixQuadEngine).calculate_event_stats(product, e)
    for name in ("observations.probability", "coupons.probability", "coupons.discounted_cash"):
        table, field = name.split(".")
        f, s = getattr(getattr(forward, table), field), getattr(getattr(stacked, table), field)
        assert np.allclose(f, s, rtol=0.0, atol=1e-12), (name, f, s)
    if variant == "duplicate":
        # the dominated knock-out keeps its row at probability 0; its coupon still counts
        assert forward.observations.probability[3] == 0.0 and forward.coupons.probability[3] > 0.1


# --- D10: cash-leg folding onto a same-instant group ----------------------------------------------------------------

def test_snowball_folding_onto_a_two_record_group_is_refused():
    """D10: the fold binder is one to one by time; a group of two effective records is refused, naming it (before:
    V1 bound the two records' cash by time and priced silently)."""
    from test_asset.equity.engine_pde.test_same_instant_ko import _overlay
    from quantark.util.exceptions import ValidationError

    product, e = repro(), env()
    with pytest.raises(ValidationError, match="2025-07-03.*2025-07-04"):
        v1().price_folded(product, e, _overlay(product, e, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


def test_snowball_folding_onto_a_duplicate_only_group_binds_the_effective_record():
    from test_asset.equity.engine_pde.test_same_instant_ko import _overlay

    e = env()
    dup, dedup = repro(103.0, 0.40), repro(drop_jul4=True)
    folded = v1().price_folded(dup, e, _overlay(dup, e, [1.0, 2.0, 3.0, 9.0, 5.0, 6.0]))
    assert folded == v1().price_folded(dedup, e, _overlay(dedup, e, [1.0, 2.0, 3.0, 5.0, 6.0]))
    assert folded != v1().price(dedup, e)


def test_phoenix_folding_onto_a_same_instant_pair_is_refused():
    """D10/D14: every Phoenix record of the instant pays its own cash; the one-to-one-by-time binder is refused."""
    from test_asset.equity.engine_pde.test_same_instant_ko import _overlay
    from quantark.util.exceptions import ValidationError

    product, e = phoenix_repro("stepdown"), env()
    with pytest.raises(ValidationError, match="2025-07-03.*2025-07-04"):
        phx().price_folded(product, e, _overlay(product, e, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
