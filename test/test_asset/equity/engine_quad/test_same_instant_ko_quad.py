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

The exact references, the products and the env are the PDE task's (``engine_pde/test_same_instant_ko.py``). The QUAD V2
rows (T4, D7/D14) are at the end of the file.
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


# --- T3bc (b): coupons decided at the valuation instant by several records settle on their own dates ---------------

SETTLE_1, SETTLE_2 = datetime(2025, 7, 8), datetime(2025, 7, 15)


def phoenix_settled_pair(b1=103.0, b2=100.0, later=()):
    """The AC7 pair (07-03 at ``b1``; 07-04 at ``b2`` paying 10%) settling on DIFFERENT dates (07-08 / 07-15), then
    the ``later`` records at 103; no KI, memory on."""
    return phoenix([ObservationRecord(observation_date=JUL3, barrier=b1, settlement_date=SETTLE_1),
                    ObservationRecord(observation_date=JUL4, barrier=b2, return_rate=0.10, settlement_date=SETTLE_2)]
                   + [ObservationRecord(observation_date=d, barrier=103.0) for d in later], ki=None)


def _decided_legs(product, e, spot):
    """The zero-gap legs decided on ``spot`` at the valuation instant, by hand: per record of the instant (alive) its
    coupon (``spot`` above the coupon barrier; memory arrears zero here: no period is missed before it) discounted at
    ITS OWN settlement, then its knock-out redemption there; ``(legs, knocked_out, coupon rows)``."""
    records = [r for r in product.resolve_ko_observations(e) if r.observation_time == 0.0]
    fractions = product.get_coupon_period_year_fractions(e)
    df = e.get_discount_factor
    legs, rows = 0.0, {}
    for k, rec in enumerate(records):
        assert spot >= COUPON_BARRIER
        c = product.get_coupon_payoff(k, year_fraction=fractions[k])
        legs += c * df(rec.settlement_time)
        rows[k] = c * df(rec.settlement_time)
        if spot >= rec.barrier:
            return legs + rec.payoff * df(rec.settlement_time), k, rows
    return legs, None, rows


@pytest.mark.parametrize("spot", [90.0, 101.0], ids=["both_paid", "paid_then_step_down_knock_out"])
def test_phoenix_valuation_instant_coupons_settle_on_their_own_dates(spot):
    """T3bc (b), exact: valued ON 07-03 (07-03 and 07-04 at env time 0), the pair settling 07-08 / 07-15, the spot in
    the both-pay band: each record's coupon is discounted at its OWN settlement (and the step-down knock-out pays its
    redemption at record 2's). With no further record the rest is the closed form: the fixed maturity redemption
    (zero here) when not knocked out. Before the fix the whole decided coupon was discounted at the last paid
    record's settlement: -3.97e-4 on both bands."""
    e = env(JUL3, spot=spot)
    product = phoenix_settled_pair()
    r1, r2 = product.resolve_ko_observations(e)
    assert r1.observation_time == r2.observation_time == 0.0 and r1.settlement_time < r2.settlement_time
    legs, knocked_out, _ = _decided_legs(product, e, spot)
    expected = legs
    if knocked_out is None:
        m = product.get_maturity_payoff_v0(spot, pricing_env=e)
        expected += m * e.get_discount_factor(resolve_terminal_timing(product, e).payment_time)
    assert phx().price(product, e) == pytest.approx(expected, rel=1e-13, abs=1e-13)


def test_phoenix_valuation_instant_coupons_settle_on_their_own_dates_with_a_continuation():
    """T3bc (b), exact: with records after the pair, the price is the decided legs (each at its own settlement) plus
    the continuation, which does not depend on where the pair settles -- the same continuation on two settlement
    layouts of the pair."""
    later = (datetime(2025, 9, 3), datetime(2025, 11, 3))
    e = env(JUL3, spot=90.0)
    a = phoenix_settled_pair(later=later)
    b = phoenix([ObservationRecord(observation_date=JUL3, barrier=103.0, settlement_date=SETTLE_2),
                 ObservationRecord(observation_date=JUL4, barrier=100.0, return_rate=0.10, settlement_date=SETTLE_2)]
                + [ObservationRecord(observation_date=d, barrier=103.0) for d in later], ki=None)
    cont_a = phx().price(a, e) - _decided_legs(a, e, 90.0)[0]
    cont_b = phx().price(b, e) - _decided_legs(b, e, 90.0)[0]
    assert cont_a == pytest.approx(cont_b, rel=1e-12, abs=1e-12)
    assert cont_a > 0.1


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("spot", [90.0, 101.0], ids=["both_paid", "paid_then_step_down_knock_out"])
def test_phoenix_valuation_instant_coupons_are_attributed_per_record(route, spot):
    """T3bc (b): the statistics put each decided coupon on its own record's coupon row (probability 1, cash discounted
    at its own settlement), the knock-out on the deciding record; the ledger reconciles with the price."""
    later = (datetime(2025, 9, 3), datetime(2025, 11, 3))
    e = env(JUL3, spot=spot)
    product = phoenix_settled_pair(later=later)
    _, knocked_out, rows = _decided_legs(product, e, spot)
    stats = v1_route(route, PhoenixQuadEngine).calculate_event_stats(product, e)
    assert stats.pv == phx().price(product, e)
    for k, cash in rows.items():
        assert stats.coupons.probability[k] == 1.0, (k, list(stats.coupons.probability))
        assert stats.coupons.discounted_cash[k] == pytest.approx(cash, rel=1e-13), (k, list(stats.coupons.discounted_cash))
    if knocked_out is not None:
        assert _rows(stats)[knocked_out] == 1.0 and sum(_rows(stats)) == 1.0
    assert abs(stats.reconciliation_error) <= 1e-9 * max(1.0, abs(stats.pv))


# === KO-reset (T3bc (c), D14: schedule-1 KO, discrete KI and schedule-2 KO records of one instant, contract order) ===

from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine  # noqa: E402
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption  # noqa: E402
from quantark.asset.equity.product.option.snowball_config import (  # noqa: E402
    AccrualConfig,
    BarrierConfig,
    PayoffConfig,
)
from quantark.util.enum import ObservationType, PostKOScheduleMode  # noqa: E402

KOR_PRE = [datetime(2025, 3, 3), datetime(2025, 5, 5), JUL3]
KOR_POST = [datetime(2025, 9, 3), datetime(2025, 11, 3), datetime(2025, 12, 3)]
#: 2025-11-27 (Thanksgiving) is a US holiday: its record sits on 2025-11-26's env instant
NOV26, NOV27 = datetime(2025, 11, 26), datetime(2025, 11, 27)


def _kor_schedule(items):
    """``(date, level)`` or ``(date, level, flat rate)`` records; a float in place of the date is a REBASED offset
    (years after the knock-in)."""
    def record(it):
        when = (dict(observation_time=it[0]) if isinstance(it[0], float) else dict(observation_date=it[0]))
        return ObservationRecord(barrier=it[1], return_rate=it[2] if len(it) > 2 else None, **when)

    return ObservationSchedule(records=[record(it) for it in items])


def ko_reset(pre=None, ki=None, post=None, mode=PostKOScheduleMode.ABSOLUTE, ki_type=ObservationType.DISCRETE,
             reverse=False):
    """The run's KO-reset repro (``.ship/.../r4_phx_kor.py``): schedule 1 KO 103 (15%) with a discrete KI 75 on
    03-03 / 05-05 / 07-03, schedule 2 KO 95 (3%) on 09-03 / 11-03 / 12-03 (ABSOLUTE), 5% rebate, no principal.
    ``reverse``: the mirrored note (KO 97 / 105 below, KI 125 above) unless the records are given."""
    pre = [(d, 97.0 if reverse else 103.0) for d in KOR_PRE] if pre is None else pre
    ki = [(d, 125.0 if reverse else 75.0) for d in KOR_PRE] if ki is None else ki
    post = [(d, 105.0 if reverse else 95.0) for d in KOR_POST] if post is None else post
    discrete = ki_type == ObservationType.DISCRETE
    p = KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=None, initial_date=datetime(2025, 1, 3),
        exercise_date=datetime(2026, 1, 5), is_reverse=reverse,
        barrier_config=BarrierConfig(ko_barrier=pre[0][1], ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_kor_schedule(pre), ki_barrier=ki[0][1],
                                     ki_observation_type=ki_type,
                                     ki_observation_schedule=_kor_schedule(ki) if discrete else None),
        post_barrier_config=BarrierConfig(ko_barrier=post[0][1], ko_rate=0.03,
                                          ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=_kor_schedule(post)),
        accrual_config=AccrualConfig(is_annualized=False),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False), post_ko_mode=mode)
    p.validate()
    return p


def kor(grid_points=GRID_POINTS, **kwargs):
    return KOResetSnowballQuadEngine(params=QuadParams(grid_points=grid_points, **kwargs))


def kor_route(route, grid_points=GRID_POINTS, **kwargs):
    engine = kor(grid_points, **kwargs)
    engine._v1_statistics_route = route
    return engine


_PRE_103 = [(d, 103.0) for d in KOR_PRE]
_KI_75 = [(d, 75.0) for d in KOR_PRE]
KOR_CASES = {
    # a schedule-1 pair: 07-04 steps the knock-out down to 100 (10%); no knock-in record on 07-04 (KO-only)
    "schedule1_step_down": dict(pre=_PRE_103 + [(JUL4, 100.0, 0.10)]),
    # a knock-in pair: 07-03 at 75, 07-04 raised to 80 (not dominated: the 75-80 band knocks in on 07-04 only)
    "ki_pair": dict(ki=_KI_75 + [(JUL4, 80.0)]),
    # both: schedule-1 step-down and the knock-in pair on 07-03 / 07-04
    "schedule1_and_ki_pair": dict(pre=_PRE_103 + [(JUL4, 100.0, 0.10)], ki=_KI_75 + [(JUL4, 80.0)]),
    # a schedule-2 pair: 11-27 steps the knock-out down to 88 (5%); a knock-in at 85 puts mass on schedule 2
    "schedule2_step_down": dict(ki=[(d, 85.0) for d in KOR_PRE],
                                post=[(KOR_POST[0], 95.0), (NOV26, 95.0), (NOV27, 88.0, 0.05), (KOR_POST[2], 95.0)]),
}


@pytest.fixture(scope="module")
def ko_reset_mc():
    """Each case on Black-Scholes MC (400k paths, seed 7): price and SE."""
    out = {}
    for name, spec in KOR_CASES.items():
        mc = SnowballMCEngine(params=MCParams(num_paths=MC_PATHS, seed=7))
        value = mc.price(ko_reset(**spec), env())
        out[name] = (value, mc.get_last_std_error())
    return out


@pytest.mark.slow
@pytest.mark.parametrize("case", list(KOR_CASES))
def test_ko_reset_same_instant_records_agree_with_mc(ko_reset_mc, case):
    """D14 on the KO-reset V1: the instant's records at one spot as one unit (CP2 ruling, T5a) -- every schedule-1
    knock-out record in contract order, then the knock-in at the widest level (the knock-out wins the tie), then every
    schedule-2 record -- within 3 SE + 1e-3 relative of MC. Before T3bc: V1 kept the instant's first record of each
    kind (8.871475 on the schedule-1 and knock-in cases); overlapping levels, where the instant unit and a date-by-date
    order differ: ``engine_pde/test_same_instant_phoenix_ko_reset.py``."""
    ref, se = ko_reset_mc[case]
    value = kor().price(ko_reset(**KOR_CASES[case]), env())
    assert abs(value - ref) <= 3.0 * se + 1e-3 * abs(ref), (value, ref, se, (value - ref) / se)


@pytest.mark.slow
@pytest.mark.parametrize("case", ["schedule1_step_down", "schedule1_and_ki_pair"])
def test_ko_reset_same_instant_records_as_the_first_observation_agree_with_mc(case):
    """The exact first step: valuation on 07-02, the 07-03 / 07-04 instant one business day ahead -- every level of
    the instant is a level of the first step's quadrature."""
    e = env(datetime(2025, 7, 2))
    product = ko_reset(**KOR_CASES[case])
    mc = SnowballMCEngine(params=MCParams(num_paths=MC_PATHS, seed=7))
    ref = mc.price(product, e)
    se = mc.get_last_std_error()
    engine = kor()
    value = engine.price(product, e)
    assert engine._last_first_step is not None
    assert abs(value - ref) <= 3.0 * se + 1e-3 * abs(ref), (value, ref, se, (value - ref) / se)


@pytest.mark.slow
@pytest.mark.parametrize("case", ["ki_pair", "schedule1_and_ki_pair"])
def test_ko_reset_same_instant_records_nodal_events_agree_with_mc(ko_reset_mc, case):
    """Nodal events: the instant unit's maps on the nodes (the knock-in at the widest level, blended as one record)."""
    ref, se = ko_reset_mc[case]
    value = kor(event_projection=EventProjectionMode.NODAL).price(ko_reset(**KOR_CASES[case]), env())
    assert abs(value - ref) <= 3.0 * se + 1e-3 * abs(ref), (value, ref, se, (value - ref) / se)


@pytest.mark.parametrize("kind", ["schedule1", "schedule2", "knock_in"])
def test_ko_reset_duplicate_record_prices_as_the_product_without_it(kind):
    """A same-instant duplicate (same level) is dominated and dropped: bitwise the product without it."""
    if kind == "schedule1":
        dup, base = ko_reset(pre=_PRE_103 + [(JUL4, 103.0, 0.40)]), ko_reset()
    elif kind == "knock_in":
        dup, base = ko_reset(ki=_KI_75 + [(JUL4, 75.0)]), ko_reset()
    else:
        post = [(KOR_POST[0], 95.0), (NOV26, 95.0), (KOR_POST[2], 95.0)]
        dup = ko_reset(post=post[:2] + [(NOV27, 95.0, 0.40)] + post[2:])
        base = ko_reset(post=post)
    e = env()
    assert kor().price(dup, e) == kor().price(base, e)


# Pinned BEFORE the T3bc change (worktree d4677f84 = f21c7998 for every file the KO-reset V1 reads).
KOR_PINS = {
    "base": "0x1.1be31f43f709dp+3",
    "base_continuous_ki": "0x1.f98d386ea5480p+2",
    "base_first_step": "0x1.5337d6098b9fap+2",
    "schedule2_base": "0x1.1bcfc9bab5037p+3",
}
KOR_ROW_PINS = {
    "forward": (["0x1.80415fbd50683p-2", "0x1.2504dfab8b35dp-3", "0x1.296a191de821dp-4", "0x1.bdd25215ee2c2p-11",
                 "0x1.bd8d9d64c9235p-10", "0x1.428e553c01dc3p-10"], "0x1.6474e8987cde0p-2", "0x1.d46de38a3aa47p-5"),
    "stacked": (["0x1.80415fbd50684p-2", "0x1.2504dfab8b362p-3", "0x1.296a191de821fp-4", "0x1.bdd25215ee2a0p-11",
                 "0x1.bd8d9d64c9233p-10", "0x1.428e553c01de0p-10"], "0x1.6474e8987cde0p-2", "0x1.d46de38a3aa4ap-5"),
}


def _kor_pinned_products():
    return {
        "base": (ko_reset(), env()),
        "base_continuous_ki": (ko_reset(ki_type=ObservationType.CONTINUOUS), env()),
        "base_first_step": (ko_reset(), env(datetime(2025, 7, 2))),
        "schedule2_base": (ko_reset(post=[(KOR_POST[0], 95.0), (NOV26, 95.0), (KOR_POST[2], 95.0)]), env()),
    }


@pytest.mark.parametrize("name", list(KOR_PINS))
def test_ko_reset_without_a_shared_instant_prices_bitwise(name):
    product, e = _kor_pinned_products()[name]
    assert kor().price(product, e) == float.fromhex(KOR_PINS[name])


@pytest.mark.parametrize("route", ROUTES)
def test_ko_reset_without_a_shared_instant_reports_bitwise(route):
    stats = kor_route(route).calculate_event_stats(ko_reset(), env())
    rows, maturity, ki_settlement = KOR_ROW_PINS[route]
    assert stats.pv == float.fromhex(KOR_PINS["base"])
    assert [float(p).hex() for p in _rows(stats)] == rows
    assert float(stats.maturity.probability).hex() == maturity
    assert float(stats.ki_settlement.probability).hex() == ki_settlement


def test_ko_reset_rebased_stays_refused_on_v1():
    """The KO-reset V1 prices ABSOLUTE schedule 2 only; a REBASED contract is refused, pair or not (unchanged)."""
    from quantark.util.exceptions import ValidationError

    product = ko_reset(pre=_PRE_103 + [(JUL4, 100.0, 0.10)], mode=PostKOScheduleMode.REBASED,
                       post=[(k / 12.0, 95.0) for k in range(1, 7)])
    with pytest.raises(ValidationError, match="ABSOLUTE"):
        kor().price(product, env())


@pytest.mark.parametrize("grid_points", [GRID_POINTS, GRID_EXACT])
@pytest.mark.parametrize("reverse", [False, True], ids=["up", "reverse"])
def test_ko_reset_knock_in_pair_is_the_wider_knock_in(reverse, grid_points):
    """Exact equivalence (D14): with no knock-out on 07-04 and schedule 2 starting in September, a knock-in at 75 on
    07-03 then at 80 on 07-04 (zero gap, one spot) knocks in exactly where a single knock-in at 80 on 07-03 does
    (reverse: 125 then 120 against 120). The instant unit (T5a) applies ONE knock-in at the widest level -- bitwise
    the single record (T3bc's piecewise projection over both levels was rel 2.7e-7 / 4.3e-9 at 1001 / 4001 nodes)."""
    first, second = (125.0, 120.0) if reverse else (75.0, 80.0)
    others = [(d, first) for d in KOR_PRE[:2]]
    pair = ko_reset(ki=others + [(JUL3, first), (JUL4, second)], reverse=reverse)
    single = ko_reset(ki=others + [(JUL3, second)], reverse=reverse)
    e = env()
    a, b = kor(grid_points).price(pair, e), kor(grid_points).price(single, e)
    assert a == b, (a, b, a / b - 1.0)


@pytest.mark.slow
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("case", ["schedule1_step_down", "schedule2_step_down"])
def test_ko_reset_same_instant_statistics_agree_with_mc(route, case):
    """Statistics (D8 on the KO-reset V1): each knock-out record of a same-instant pair is its own row and adjoint
    source -- per-record knock-out probability within 3 SE of MC (binomial SE over the paths); the ledger reconciles.
    Before: the table refused the repeated (time, schedule) on both engines; the dated table keeps one row per record
    (``ObservationTable``: a repeated time and tag needs distinct dates)."""
    from quantark.util.exceptions import ValidationError  # noqa: F401  (the refusal named above)

    product, e = ko_reset(**KOR_CASES[case]), env()
    mc = SnowballMCEngine(params=MCParams(num_paths=MC_PATHS, seed=7)).calculate_event_stats(product, e)
    stats = kor_route(route).calculate_event_stats(product, e)
    rows, mc_rows = _rows(stats), _rows(mc)
    assert len(rows) == len(mc_rows) == 7
    assert list(stats.observations.dates) == list(mc.observations.dates)
    for p, m in zip(rows, mc_rows):
        se = math.sqrt(m * (1.0 - m) / MC_PATHS)
        assert abs(p - m) <= 3.0 * se, (rows, mc_rows)
    assert abs(stats.reconciliation_error) <= 1e-9 * max(1.0, abs(stats.pv))


@pytest.mark.slow
@pytest.mark.parametrize("route", ROUTES)
def test_ko_reset_knock_in_pair_statistics_agree_with_mc_on_the_wider_knock_in(route):
    """The knock-in pair's knock-out rows (the knock-in stream not requested), held to MC on the exactly equivalent
    single knock-in at 80 -- within 3 SE per record (the pair's own knock-in rows against MC:
    ``engine_pde/test_same_instant_ki.py``)."""
    from quantark.cashleg.event_distribution import EventType

    e = env()
    pair = ko_reset(**KOR_CASES["ki_pair"])
    single = ko_reset(ki=_KI_75[:2] + [(JUL3, 80.0)])
    mc = SnowballMCEngine(params=MCParams(num_paths=MC_PATHS, seed=7)).calculate_event_stats(single, e)
    rows = _rows(kor_route(route).calculate_event_stats(pair, e, streams=frozenset({EventType.KO})))
    for p, m in zip(rows, _rows(mc)):
        se = math.sqrt(m * (1.0 - m) / MC_PATHS)
        assert abs(p - m) <= 3.0 * se, (rows, _rows(mc))


def test_ko_reset_same_instant_statistics_routes_agree():
    """The forward pass attributes the instant's sources as the stacked columns do (adjoint identity), the knock-in
    pair's single projection included."""
    from quantark.cashleg.event_distribution import EventType

    product, e = ko_reset(**KOR_CASES["schedule1_and_ki_pair"]), env()
    streams = frozenset({EventType.KO})
    fwd = kor_route("forward").calculate_event_stats(product, e, streams=streams)
    stk = kor_route("stacked").calculate_event_stats(product, e, streams=streams)
    assert np.allclose(_rows(fwd), _rows(stk), rtol=0.0, atol=1e-12), (_rows(fwd), _rows(stk))
    assert _rows(fwd)[3] > 0.01                                      # the 07-04 step-down's own row
    assert abs(fwd.reconciliation_error) <= 1e-9 * abs(fwd.pv)


@pytest.mark.parametrize("route", ROUTES)
def test_ko_reset_knock_in_pair_knock_in_stream_is_reported_per_record(route):
    """T5a (zero-gap rule; before: ``KnockInTable.times has duplicate rows`` on V1 and MC alike): one knock-in row per
    record -- 07-03 (75) takes the mass below 75, 07-04 (80) the 75-80 band; the two sum to the single knock-in at 80's
    row, every other row is that product's (MC agreement: ``engine_pde/test_same_instant_ki.py``)."""
    e = env()
    pair = kor_route(route).calculate_event_stats(ko_reset(**KOR_CASES["ki_pair"]), e)
    single = kor_route(route).calculate_event_stats(ko_reset(ki=_KI_75[:2] + [(JUL3, 80.0)]), e)
    rows, ref = list(pair.ki_table.probability), list(single.ki_table.probability)
    assert list(pair.ki_table.dates) == KOR_PRE + [JUL4]
    assert rows[2] + rows[3] == pytest.approx(ref[2], rel=1e-12) and rows[3] > 0.01
    assert rows[:2] == pytest.approx(ref[:2], rel=1e-12)
    assert pair.knock_in.probability == pytest.approx(single.knock_in.probability, rel=1e-12)


@pytest.mark.parametrize("route", ROUTES)
def test_ko_reset_dominated_duplicate_has_zero_probability(route):
    """The dominated schedule-1 record keeps its row at probability 0; the other rows equal the product's without it."""
    e = env()
    dup = kor_route(route).calculate_event_stats(ko_reset(pre=_PRE_103 + [(JUL4, 103.0, 0.40)]), e)
    base = kor_route(route).calculate_event_stats(ko_reset(), e)
    rows, base_rows = _rows(dup), _rows(base)
    assert rows[3] == 0.0
    assert rows[:3] + rows[4:] == base_rows
    assert dup.pv == base.pv


def test_ko_reset_folding_onto_a_same_instant_pair_is_refused():
    """D10: two effective knock-out records of one schedule on one instant each pay their own cash; the binder is one
    to one by (schedule, time) -- refused, naming them."""
    from quantark.cashleg.folding import FoldedEventCash, FoldOverlay
    from quantark.util.exceptions import ValidationError

    product, e = ko_reset(**KOR_CASES["schedule1_step_down"]), env()
    pre, _, _ = product._resolve_ko_schedule(product.barrier_config, e)
    timing = resolve_terminal_timing(product, e)
    det, pay = float(timing.determination_time), float(timing.payment_time)
    overlay = FoldOverlay(
        ko=tuple(FoldedEventCash("pre", float(r.observation_time), float(r.settlement_time), float(k + 1))
                 for k, r in enumerate(pre)),
        maturity_no_ki=FoldedEventCash("main", det, pay, 0.25), maturity_ki=FoldedEventCash("main", det, pay, -0.5),
        constant_pv=0.0)
    with pytest.raises(ValidationError, match="2025-07-03.*2025-07-04"):
        kor().price_folded(product, e, overlay)


@pytest.mark.parametrize("spot,deciding", [(105.0, 0), (101.0, 1)], ids=["first_band", "step_down_band"])
def test_ko_reset_valuation_instant_pays_the_deciding_record(spot, deciding):
    """Valuation on 07-03 (07-03 and 07-04 at t = 0): the product's decision on the known spot knocks out by the first
    schedule-1 record (contract order) the spot breaches, paying ITS cash; the statistics put probability 1 on that
    record's own row (by index, not the instant's first record). Indices count the records still ahead (03-03 and
    05-05 have passed)."""
    e = env(JUL3, spot=spot)
    product = ko_reset(**KOR_CASES["schedule1_step_down"])
    decision = product.decide_observations_at_valuation(spot, e)
    assert decision.knocked_out and decision.ko_index == deciding
    expected = decision.ko_record.payoff * e.get_discount_factor(decision.ko_record.settlement_time)
    assert kor().price(product, e) == pytest.approx(expected, rel=1e-14, abs=1e-14)
    for route in ROUTES:
        rows = _rows(kor_route(route).calculate_event_stats(product, e))
        assert rows[deciding] == 1.0 and sum(rows) == 1.0, (route, rows)


# === QUAD V2 (T4: spec D2/D4/D7/D8/D10/D14) ===========================================================================
#
# QUAD V2 compiles the KO rows that share a canonical time into ONE event (the helper's same-instant group is the unit):
# its action at each spot region is the zero-gap chain of the instant's records in contract order -- a Snowball's
# effective records (``same_instant_ko_groups``; the first breached pays its own cash), a Phoenix's every record (its
# coupon with memory, then its knock-out). Each record has its own statistics channels.

from quantark.asset.equity.engine.quad.v2.engine import (  # noqa: E402
    KOResetSnowballQuadEngineV2,
    PhoenixQuadEngineV2,
    SnowballQuadEngineV2,
)
from quantark.asset.equity.param import QuadV2Params  # noqa: E402

#: the V2 parameters of every V2 test: the defaults (``QuadV2Params()``: order 8, 2 cells per SD, 11 SD domain)
V2_PARAMS = dict()


def v2(cls=SnowballQuadEngineV2, **kwargs):
    return cls(QuadV2Params(**{**V2_PARAMS, **kwargs}))


def v2p(**kwargs):
    return v2(PhoenixQuadEngineV2, **kwargs)


def phoenix_at_maturity(b1, b2):
    """The AC7 pair as the maturity instant: exercise on the 07-04 holiday (env time of 07-03)."""
    p = phoenix_pair(b1, b2)
    p.exercise_date = JUL4
    p.validate()
    return p


# --- AC2 / AC7: the exact references ---------------------------------------------------------------------------------
# V2 integrates the Gaussian law of the env-time-0.5 spot across the instant's breaks exactly, so it meets the closed
# forms to round-off (measured rel 2e-16 .. 3e-15; the knock-out and coupon probabilities 1e-15). The spec asks rel
# 1e-6; the tests hold 1e-10 (abs 1e-10 on probabilities).

@pytest.mark.parametrize("reverse", [False, True], ids=["up", "reverse"])
def test_v2_snowball_pair_matches_the_exact_reference(reverse):
    """AC2 (QUAD V2, default params): the shared pair alone, no KI, fixed maturity redemption (measured rel +2.2e-16
    up, -3.3e-16 reverse; before T4 V2 refused: "multiple observations of one kind share a canonical time")."""
    product = snowball(pair(97.0, 100.0) if reverse else pair(), ki=None, reverse=reverse)
    e = env()
    ref, _ = _exact_pair_value(product, e, reverse)
    value = v2().price(product, e)
    assert abs(value / ref - 1.0) <= 1e-10, (value, ref)


def test_v2_snowball_pair_as_the_first_observation_matches_the_exact_reference():
    product = snowball(pair(), ki=None)
    e = env(datetime(2025, 7, 2))
    ref, _ = _exact_pair_value(product, e)
    value = v2().price(product, e)
    assert abs(value / ref - 1.0) <= 1e-10, (value, ref)


def test_v2_snowball_pair_at_maturity_matches_the_exact_reference():
    """AC5 maturity instant: the pair is the terminal knock-out (exercise on the 07-04 holiday)."""
    product = snowball(pair(), ki=None, exercise=JUL4)
    e = env()
    ref, _ = _exact_pair_value(product, e)
    value = v2().price(product, e)
    assert abs(value / ref - 1.0) <= 1e-10, (value, ref)


@pytest.mark.parametrize("b1,b2", [(103.0, 100.0), (100.0, 103.0)], ids=["b2_below_b1", "b2_above_b1"])
def test_v2_phoenix_pair_matches_the_exact_reference(b1, b2):
    """AC7 (QUAD V2, default params): the pair alone, no KI, memory on; ``b2 > b1``: record 2's knock-out is inert but
    its coupon counts (measured rel -2.7e-15 / +1.3e-15)."""
    product = phoenix_pair(b1, b2)
    e = env()
    ref, _, _ = _exact_phoenix_pair(product, e)
    value = v2p().price(product, e)
    assert abs(value / ref - 1.0) <= 1e-10, (value, ref)


@pytest.mark.parametrize("b1,b2", [(103.0, 100.0), (100.0, 103.0)], ids=["b2_below_b1", "b2_above_b1"])
def test_v2_phoenix_pair_at_maturity_matches_the_exact_reference(b1, b2):
    """AC5 maturity instant on the Phoenix: the zero-gap chain, then the maturity redemption."""
    product = phoenix_at_maturity(b1, b2)
    e = env()
    r1, r2 = product.resolve_ko_observations(e)
    assert r1.observation_time == r2.observation_time == product.get_maturity(e)
    ref, _, _ = _exact_phoenix_pair(product, e)
    value = v2p().price(product, e)
    assert abs(value / ref - 1.0) <= 1e-10, (value, ref)


def _zero_gap_expiry_cash(product, e, s, memory):
    """EXPIRY pay: the zero-gap rule's coupons are paid at the note's end -- with the redemption at the settlement of
    the record that knocks out, else at the maturity payment; the arrears (memory on) as ``_zero_gap_cash``."""
    records = product.resolve_ko_observations(e)
    fractions = product.get_coupon_period_year_fractions(e)
    df = e.get_discount_factor
    earned, arrears = 0.0, 0.0
    for k, rec in enumerate(records):
        c = product.get_coupon_payoff(k, year_fraction=fractions[k])
        hit = s >= COUPON_BARRIER
        if hit:
            earned += c + arrears
        if s >= rec.barrier:
            return (earned + rec.payoff + (0.0 if hit else arrears)) * df(rec.settlement_time)
        arrears = 0.0 if hit or not memory else arrears + c
    m = product.get_maturity_payoff_v0(s, pricing_env=e)
    return (earned + m) * df(resolve_terminal_timing(product, e).payment_time)


@pytest.mark.parametrize("memory", [True, False], ids=["memory", "no_memory"])
@pytest.mark.parametrize("b1,b2", [(103.0, 100.0), (100.0, 103.0)], ids=["b2_below_b1", "b2_above_b1"])
def test_v2_phoenix_expiry_pair_matches_the_exact_reference(b1, b2, memory):
    """D14 under EXPIRY pay (exact): record 1's coupon is deferred to the note's end, which a knock-out by record 2 at
    the same instant makes record 2's settlement (07-15, record 1 settling 07-08) -- the deferred coupon goes through
    the termination state of the chain. The ledger reconciles."""
    from quantark.util.enum import CouponPayType

    product = phoenix_settled_pair(b1, b2)
    product.coupon_config = replace(product.coupon_config, coupon_pay_type=CouponPayType.EXPIRY,
                                    memory_coupon=memory)
    product.validate()
    e = env()
    r1, r2 = product.resolve_ko_observations(e)
    t1 = r1.observation_time
    assert t1 == r2.observation_time and r1.settlement_time < r2.settlement_time
    levels = sorted({COUPON_BARRIER, b1, b2})
    edges = [0.0] + levels + [math.inf]
    above = lambda b: 0.0 if b == math.inf else (1.0 if b == 0.0 else _p_breach(b, t1))   # noqa: E731
    ref = sum((above(lo) - above(hi)) * _zero_gap_expiry_cash(product, e, 0.5 * (lo + hi) if hi < math.inf
                                                              else lo + 1.0, memory)
              for lo, hi in zip(edges[:-1], edges[1:]))
    value = v2p().price(product, e)
    assert abs(value / ref - 1.0) <= 1e-10, (value, ref)
    stats = v2p().calculate_event_stats(product, e)
    assert stats.pv == value
    assert abs(stats.reconciliation_error) <= 1e-9 * abs(stats.pv)


# --- AC4: the duplicate invariant -------------------------------------------------------------------------------------

@pytest.mark.parametrize("dup_rate", [0.15, 0.40], ids=["same_payoff", "other_payoff"])
def test_v2_snowball_duplicate_record_prices_as_the_product_without_it(dup_rate):
    """AC4: a same-instant duplicate (same barrier, any payoff) is dominated and dropped: bitwise, price and
    statistics (the dominated record keeps its row at exactly 0)."""
    e = env()
    dup, base = repro(103.0, dup_rate), repro(drop_jul4=True)
    assert v2().price(dup, e) == v2().price(base, e)
    s_dup, s_base = v2().calculate_event_stats(dup, e), v2().calculate_event_stats(base, e)
    probs, base_probs = _rows(s_dup), _rows(s_base)
    assert probs[3] == 0.0
    assert probs[:3] + probs[4:] == base_probs
    assert math.isclose(sum(probs), sum(base_probs), rel_tol=1e-12)
    assert s_dup.pv == s_base.pv


# --- AC5: the valuation instant --------------------------------------------------------------------------------------

@pytest.mark.parametrize("spot,deciding", [(105.0, 0), (101.0, 1)], ids=["first_band", "step_down_band"])
def test_v2_snowball_valuation_instant_pays_the_deciding_record(spot, deciding):
    e = env(JUL3, spot=spot)
    product = repro()
    records = [r for r in product.resolve_ko_observations(e) if r.observation_time == 0.0]
    assert len(records) == 2
    rec = records[deciding]
    expected = rec.payoff * e.get_discount_factor(rec.settlement_time)
    assert v2().price(product, e) == pytest.approx(expected, rel=1e-14, abs=1e-14)
    probs = _rows(v2().calculate_event_stats(product, e))
    assert probs[deciding] == 1.0 and sum(probs) == 1.0, probs


@pytest.mark.parametrize("spot,deciding", [(105.0, 0), (101.0, 1)], ids=["first_band", "step_down_band"])
def test_v2_snowball_valuation_and_maturity_instant_pays_the_deciding_record(spot, deciding):
    e = env(JUL3, spot=spot)
    product = snowball(pair(), ki=None, exercise=JUL4)
    rec = product.resolve_ko_observations(e)[deciding]
    expected = rec.payoff * e.get_discount_factor(rec.settlement_time)
    assert v2().price(product, e) == pytest.approx(expected, rel=1e-14, abs=1e-14)
    probs = _rows(v2().calculate_event_stats(product, e))
    assert probs[deciding] == 1.0 and sum(probs) == 1.0, probs


def test_v2_snowball_valuation_instant_below_every_band_continues():
    """Valuation on 07-03, spot below both records: no knock-out, both rows 0, the note continues (V1 agrees to its
    lattice)."""
    e = env(JUL3, spot=97.0)
    product = repro()
    stats = v2().calculate_event_stats(product, e)
    assert _rows(stats)[:2] == [0.0, 0.0]
    value = v2().price(product, e)
    assert value == pytest.approx(v1().price(product, e), rel=2e-3)


@pytest.mark.parametrize("spot,deciding", [(105.0, 0), (101.0, 1)], ids=["first_band", "step_down_band"])
def test_v2_phoenix_valuation_instant_pays_the_deciding_record(spot, deciding):
    e = env(JUL3, spot=spot)
    product = phoenix_repro("stepdown")
    decision = product.decide_observations_at_valuation(spot, e)
    records = [r for r in product.resolve_ko_observations(e) if r.observation_time == 0.0]
    assert decision.knocked_out and decision.ko_record.observation_date == records[deciding].observation_date
    expected = (decision.ko_record.payoff + decision.coupon) * e.get_discount_factor(decision.ko_record.settlement_time)
    assert v2p().price(product, e) == pytest.approx(expected, rel=1e-14, abs=1e-14)
    probs = _rows(v2p().calculate_event_stats(product, e))
    assert probs[deciding] == 1.0 and sum(probs) == 1.0, probs


@pytest.mark.parametrize("spot", [90.0, 101.0], ids=["both_paid", "paid_then_step_down_knock_out"])
def test_v2_phoenix_valuation_instant_coupons_settle_on_their_own_dates(spot):
    """The pair settling 07-08 / 07-15, valued on 07-03 in the both-pay band: each record's coupon discounted at its
    OWN settlement (the step-down knock-out pays its redemption at record 2's), exact."""
    e = env(JUL3, spot=spot)
    product = phoenix_settled_pair()
    legs, knocked_out, _ = _decided_legs(product, e, spot)
    expected = legs
    if knocked_out is None:
        m = product.get_maturity_payoff_v0(spot, pricing_env=e)
        expected += m * e.get_discount_factor(resolve_terminal_timing(product, e).payment_time)
    assert v2p().price(product, e) == pytest.approx(expected, rel=1e-13, abs=1e-13)


@pytest.mark.parametrize("spot", [90.0, 101.0], ids=["both_paid", "paid_then_step_down_knock_out"])
def test_v2_phoenix_valuation_instant_coupons_are_attributed_per_record(spot):
    later = (datetime(2025, 9, 3), datetime(2025, 11, 3))
    e = env(JUL3, spot=spot)
    product = phoenix_settled_pair(later=later)
    _, knocked_out, rows = _decided_legs(product, e, spot)
    stats = v2p().calculate_event_stats(product, e)
    assert stats.pv == v2p().price(product, e)
    for k, cash in rows.items():
        assert stats.coupons.probability[k] == 1.0, (k, list(stats.coupons.probability))
        assert stats.coupons.discounted_cash[k] == pytest.approx(cash, rel=1e-13), (
            k, list(stats.coupons.discounted_cash))
    if knocked_out is not None:
        assert _rows(stats)[knocked_out] == 1.0 and sum(_rows(stats)) == 1.0
    assert abs(stats.reconciliation_error) <= 1e-9 * max(1.0, abs(stats.pv))


def test_v2_phoenix_valuation_instant_continuation_does_not_depend_on_where_the_pair_settles():
    later = (datetime(2025, 9, 3), datetime(2025, 11, 3))
    e = env(JUL3, spot=90.0)
    a = phoenix_settled_pair(later=later)
    b = phoenix([ObservationRecord(observation_date=JUL3, barrier=103.0, settlement_date=SETTLE_2),
                 ObservationRecord(observation_date=JUL4, barrier=100.0, return_rate=0.10, settlement_date=SETTLE_2)]
                + [ObservationRecord(observation_date=d, barrier=103.0) for d in later], ki=None)
    cont_a = v2p().price(a, e) - _decided_legs(a, e, 90.0)[0]
    cont_b = v2p().price(b, e) - _decided_legs(b, e, 90.0)[0]
    assert cont_a == pytest.approx(cont_b, rel=1e-12, abs=1e-12)
    assert cont_a > 0.1


# --- AC6: event statistics -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("reverse", [False, True], ids=["up", "reverse"])
def test_v2_snowball_pair_statistics_attribute_each_record_exactly(reverse):
    product = snowball(pair(97.0, 100.0) if reverse else pair(), ki=None, reverse=reverse)
    e = env()
    _, (p1, p2) = _exact_pair_value(product, e, reverse)
    stats = v2().calculate_event_stats(product, e)
    probs = _rows(stats)
    assert len(probs) == 2
    assert probs[0] == pytest.approx(p1, abs=1e-10) and probs[1] == pytest.approx(p2, abs=1e-10), (probs, p1, p2)
    assert list(stats.observations.dates) == [JUL3, JUL4]
    assert abs(stats.reconciliation_error) <= 1e-9 * abs(stats.pv)


def test_v2_snowball_pair_at_maturity_statistics_attribute_each_record_exactly():
    product = snowball(pair(), ki=None, exercise=JUL4)
    e = env()
    _, (p1, p2) = _exact_pair_value(product, e)
    probs = _rows(v2().calculate_event_stats(product, e))
    assert probs[0] == pytest.approx(p1, abs=1e-10) and probs[1] == pytest.approx(p2, abs=1e-10), (probs, p1, p2)


@pytest.mark.parametrize("b1,b2", [(103.0, 100.0), (100.0, 103.0)], ids=["b2_below_b1", "b2_above_b1"])
def test_v2_phoenix_pair_statistics_attribute_each_record_exactly(b1, b2):
    product = phoenix_pair(b1, b2)
    e = env()
    _, ko, coupon = _exact_phoenix_pair(product, e)
    stats = v2p().calculate_event_stats(product, e)
    assert _rows(stats) == pytest.approx(list(ko), abs=1e-10)
    assert _coupon_rows(stats) == pytest.approx(list(coupon), abs=1e-10)
    assert abs(stats.reconciliation_error) <= 1e-9 * abs(stats.pv)


def test_v2_snowball_step_down_total_is_not_double_counted():
    """Each record reads its own channel: the step-down's rows plus the maturity and knock-in settlement units sum
    to one (before the per-record keys the instant's channel was read twice)."""
    stats = v2().calculate_event_stats(repro(), env())
    total = sum(_rows(stats)) + stats.maturity.probability + stats.ki_settlement.probability
    assert total == pytest.approx(1.0, abs=1e-10)
    assert _rows(stats)[3] > 0.01 and _rows(stats)[2] != _rows(stats)[3]


@pytest.mark.slow
def test_v2_snowball_step_down_agrees_with_mc_and_v1(snowball_mc):
    """AC3: within 3 SE + 1e-3 relative of MC; V2 and V1 agree to V1's lattice error (measured V2 -0.264722, V1
    -0.265338, MC -0.257029 SE 0.021928: z -0.35; V2 - V1 +6.2e-4, the single-record product's own gap 2.5e-3)."""
    ref, se, _ = snowball_mc
    value = v2().price(repro(), env())
    assert abs(value - ref) <= 3.0 * se + 1e-3 * abs(ref), (value, ref, se)
    assert abs(value - v1().price(repro(), env())) <= 2.5e-3, value


@pytest.mark.slow
def test_v2_snowball_step_down_statistics_agree_with_mc(snowball_mc):
    _, _, mc_stats = snowball_mc
    rows, mc_rows = _rows(v2().calculate_event_stats(repro(), env())), _rows(mc_stats)
    assert len(rows) == len(mc_rows) == len(REPRO_DATES)
    for p, m in zip(rows, mc_rows):
        se = math.sqrt(m * (1.0 - m) / MC_PATHS)
        assert abs(p - m) <= 3.0 * se, (rows, mc_rows)
    assert rows[3] > 0.01


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["single", "duplicate", "stepdown"])
def test_v2_phoenix_repro_agrees_with_mc_and_v1(phoenix_mc, variant):
    """AC7: measured z(V2) -1.09 / -1.16 / -1.32 (single / duplicate / step-down); V2 - V1 +1.45e-3 / +1.51e-3 /
    +2.1e-4 -- the pair's gap is the single record's (V1's lattice)."""
    ref, se, _ = phoenix_mc[variant]
    value = v2p().price(phoenix_repro(variant), env())
    assert abs(value - ref) <= 3.0 * se + 1e-3 * abs(ref), (value, ref, se)
    assert abs(value - phx().price(phoenix_repro(variant), env())) <= 2.5e-3, value


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["duplicate", "stepdown"])
def test_v2_phoenix_repro_statistics_agree_with_mc(phoenix_mc, variant):
    _, _, mc_stats = phoenix_mc[variant]
    stats = v2p().calculate_event_stats(phoenix_repro(variant), env())
    for rows, mc_rows in ((_rows(stats), _rows(mc_stats)), (_coupon_rows(stats), _coupon_rows(mc_stats))):
        assert len(rows) == len(mc_rows) == len(PHX_DATES)
        for p, m in zip(rows, mc_rows):
            se = math.sqrt(m * (1.0 - m) / MC_PATHS)
            assert abs(p - m) <= 3.0 * se, (rows, mc_rows)
    if variant == "duplicate":
        assert _rows(stats)[3] == 0.0 and _coupon_rows(stats)[3] > 0.1
    assert abs(stats.reconciliation_error) <= 1e-9 * max(1.0, abs(stats.pv))


@pytest.mark.parametrize("ki", ["continuous", "discrete"])
@pytest.mark.parametrize("memory", [True, False], ids=["memory", "no_memory"])
@pytest.mark.parametrize("pay", ["INSTANT", "EXPIRY"])
@pytest.mark.parametrize("variant", ["duplicate", "stepdown"])
def test_v2_phoenix_variants_agree_with_v1(variant, pay, memory, ki):
    """Every coupon variant of the pair (pay INSTANT / EXPIRY, memory on / off, continuous / discrete KI): V2 and V1
    agree to their discretisations, per-record knock-out and coupon probabilities too; the V2 ledger reconciles.
    Measured: price gaps 2.1e-4 .. 1.9e-3 (the single-record variants' own gaps 1.5e-3 .. 1.8e-3), row gaps <= 1.7e-4,
    V2 reconciliation <= 5e-15."""
    from quantark.util.enum import CouponPayType

    product = _phoenix_variant(variant, getattr(CouponPayType, pay), memory, ki)
    e = env()
    a, b = v2p().price(product, e), phx().price(product, e)
    assert abs(a - b) <= 2.5e-3, (a, b)
    s2, s1 = v2p().calculate_event_stats(product, e), v1_route("stacked", PhoenixQuadEngine).calculate_event_stats(
        product, e)
    assert np.allclose(_rows(s2), _rows(s1), atol=3e-4), (_rows(s2), _rows(s1))
    assert np.allclose(_coupon_rows(s2), _coupon_rows(s1), atol=3e-4), (_coupon_rows(s2), _coupon_rows(s1))
    assert abs(s2.reconciliation_error) <= 1e-9 * max(1.0, abs(s2.pv))


# --- D10 / D2: folding, tolerances, refusals kept -------------------------------------------------------------------

def test_v2_snowball_folding_onto_a_two_record_group_is_refused():
    from test_asset.equity.engine_pde.test_same_instant_ko import _overlay
    from quantark.util.exceptions import ValidationError

    product, e = repro(), env()
    with pytest.raises(ValidationError, match="2025-07-03.*2025-07-04"):
        v2().price_folded(product, e, _overlay(product, e, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


def test_v2_snowball_folding_onto_a_duplicate_only_group_binds_the_effective_record():
    from test_asset.equity.engine_pde.test_same_instant_ko import _overlay

    e = env()
    dup, dedup = repro(103.0, 0.40), repro(drop_jul4=True)
    folded = v2().price_folded(dup, e, _overlay(dup, e, [1.0, 2.0, 3.0, 9.0, 5.0, 6.0]))
    assert folded == v2().price_folded(dedup, e, _overlay(dedup, e, [1.0, 2.0, 3.0, 5.0, 6.0]))
    assert folded != v2().price(dedup, e)


def test_v2_phoenix_folding_onto_a_same_instant_pair_is_refused():
    from test_asset.equity.engine_pde.test_same_instant_ko import _overlay
    from quantark.util.exceptions import ValidationError

    product, e = phoenix_repro("stepdown"), env()
    with pytest.raises(ValidationError, match="2025-07-03.*2025-07-04"):
        v2p().price_folded(product, e, _overlay(product, e, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


def test_v2_user_tolerance_merging_distinct_instants_still_refuses():
    """D2: the helper's group is the unit -- a user ``event_time_tolerance`` that merges two records the helper keeps
    apart (07-03 and 07-07, one business day) keeps V2's refusal."""
    from quantark.util.exceptions import ValidationError

    jul7 = datetime(2025, 7, 7)
    product = snowball([ObservationRecord(observation_date=JUL3, barrier=103.0),
                        ObservationRecord(observation_date=jul7, barrier=100.0, return_rate=0.10)], ki=None)
    e = env()
    r1, r2 = product.resolve_ko_observations(e)
    assert 0.0 < r2.observation_time - r1.observation_time < 0.01
    with pytest.raises(ValidationError, match="share a canonical time"):
        v2(event_time_tolerance=0.01).price(product, e)
    with pytest.raises(ValidationError, match="share a canonical time"):
        v2p(event_time_tolerance=0.01).price(phoenix([ObservationRecord(observation_date=JUL3, barrier=103.0),
                                                      ObservationRecord(observation_date=jul7, barrier=100.0)],
                                                     ki=None), e)


def test_v2_knock_in_records_sharing_an_instant_merge_into_one_knock_in():
    """T5a (zero-gap rule; before: refused "share a canonical time"): knock-in rows sharing an instant are one knock-in
    at the widest level -- a duplicate prices bitwise as the product without it (the pairs: ``engine_pde/
    test_same_instant_ki.py``); a user ``event_time_tolerance`` merging knock-in records of distinct instants keeps
    the refusal."""
    from quantark.util.enum import ObservationType
    from quantark.util.exceptions import ValidationError

    def with_ki(dates):
        p = repro(drop_jul4=True)
        p.barrier_config = replace(
            p.barrier_config, ki_observation_type=ObservationType.DISCRETE, ki_continuous=False,
            ki_observation_schedule=ObservationSchedule(
                records=[ObservationRecord(observation_date=d, barrier=75.0) for d in dates]))
        p.validate()
        return p

    e = env()
    assert v2().price(with_ki(REPRO_DATES), e) == v2().price(with_ki([d for d in REPRO_DATES if d != JUL4]), e)
    jul7 = datetime(2025, 7, 7)
    with pytest.raises(ValidationError, match="share a canonical time"):
        v2(event_time_tolerance=0.01).price(with_ki(sorted([d for d in REPRO_DATES if d != JUL4] + [jul7])), e)


def test_v2_ko_reset_pair_stays_refused():
    """KO-reset same-instant records are not supported by V2 (unchanged refusal)."""
    from quantark.util.exceptions import ValidationError

    with pytest.raises(ValidationError, match="share a canonical time"):
        v2(KOResetSnowballQuadEngineV2).price(ko_reset(**KOR_CASES["schedule1_step_down"]), env())


# --- products without a shared instant: bitwise against pins captured at d0ac7fb4, before T4 -----------------------

def _v2_pinned():
    from quantark.util.enum import CouponPayType

    e = env()
    single = [ObservationRecord(observation_date=JUL3, barrier=103.0)]
    out = {
        "snowball_repro_single": (SnowballQuadEngineV2, repro(drop_jul4=True), e),
        "snowball_one_record_no_ki": (SnowballQuadEngineV2, snowball(single, ki=None), e),
        "snowball_reverse_single": (SnowballQuadEngineV2, snowball(
            [ObservationRecord(observation_date=JUL3, barrier=97.0)], ki=None, reverse=True), e),
        "snowball_val_jul3_ko": (SnowballQuadEngineV2, repro(drop_jul4=True), env(JUL3, spot=105.0)),
        "snowball_val_jul3_alive": (SnowballQuadEngineV2, repro(drop_jul4=True), env(JUL3, spot=101.0)),
        "snowball_first_step": (SnowballQuadEngineV2, repro(drop_jul4=True), env(datetime(2025, 7, 2))),
        "phoenix_single": (PhoenixQuadEngineV2, phoenix_repro("single"), e),
        "phoenix_val_jul3": (PhoenixQuadEngineV2, phoenix_repro("single"), env(JUL3, spot=90.0)),
        "phoenix_val_jul3_ko": (PhoenixQuadEngineV2, phoenix_repro("single"), env(JUL3, spot=105.0)),
        "phoenix_one_record": (PhoenixQuadEngineV2, phoenix(single, ki=None), e),
        "ko_reset_base": (KOResetSnowballQuadEngineV2, ko_reset(), e),
    }
    for pay in ("INSTANT", "EXPIRY"):
        for memory in (True, False):
            for ki in ("continuous", "discrete"):
                out[f"phoenix_{pay}_{memory}_{ki}"] = (
                    PhoenixQuadEngineV2, _phoenix_variant("single", getattr(CouponPayType, pay), memory, ki), e)
    return out


# Pinned BEFORE the T4 change (worktree d0ac7fb4): price, pv, KO rows and discounted cash, coupon rows and discounted
# cash (None: no coupon table), maturity / knocked-in settlement / knock-in probabilities.
V2_PINS = {
    'ko_reset_base': dict(
        price='0x1.1be5a6a674620p+3', pv='0x1.1be5a6a674620p+3',
        ko=['0x1.80268d2a30d98p-2', '0x1.24f61998cb561p-3', '0x1.29640d787ce9dp-4', '0x1.bc6697ca056c6p-11',
            '0x1.bcbb33e26a3d6p-10', '0x1.41cce6cbff2c9p-10'],
        ko_cash=['0x1.670736d1b46cap+2', '0x1.10d38a23b978bp+1', '0x1.14079e3fd8ad3p+0', '0x1.48e2d461387b5p-9',
                 '0x1.48090e4ccf7ebp-8', '0x1.d9f825c451c88p-9'],
        coupon=None,
        coupon_cash=None,
        maturity='0x1.64a75ac99f7c8p-2', ki_settlement='0x1.d40a63d8bac41p-5', ki='0x1.f2f03f0d56208p-5'),
    'phoenix_EXPIRY_False_continuous': dict(
        price='-0x1.86b9c0248fabap+0', pv='-0x1.86b9c0248fabap+0',
        ko=['0x1.80268d2a30d98p-2', '0x1.24f6b675401c7p-3', '0x1.297c923798da7p-4', '0x1.7e337bafaeb13p-5',
            '0x1.0ff5c8d9205ddp-5'],
        ko_cash=['0x0.0p+0', '0x0.0p+0', '0x0.0p+0', '0x0.0p+0', '0x0.0p+0'],
        coupon=['0x1.f9c29f0040534p-1', '0x1.1f8ac010c2594p-1', '0x1.8592e8190b814p-2', '0x1.225938d8a2872p-2',
                '0x1.c6e799a2b0bd6p-3'],
        coupon_cash=['0x1.f481928abc9b2p-1', '0x1.1b64dff632acdp-1', '0x1.7f0791197d3b4p-2', '0x1.1d0a187af0d27p-2',
                     '0x1.be264b73542d8p-3'],
        maturity='0x1.8e200177e7d0bp-3', ki_settlement='0x1.14539380762d9p-3', ki='0x1.158a7b0b813ddp-3'),
    'phoenix_EXPIRY_False_discrete': dict(
        price='0x1.cf70403976140p-6', pv='0x1.cf70403976140p-6',
        ko=['0x1.80268d2a30d98p-2', '0x1.24f6b6753fb69p-3', '0x1.297c9237995dcp-4', '0x1.7e337bafaf540p-5',
            '0x1.0ff5c8d920a62p-5'],
        ko_cash=['0x0.0p+0', '0x0.0p+0', '0x0.0p+0', '0x0.0p+0', '0x0.0p+0'],
        coupon=['0x1.f9c29f0040534p-1', '0x1.1f8ac010c2545p-1', '0x1.8592e8190baeep-2', '0x1.225938d8a2983p-2',
                '0x1.c6e799a2b0bedp-3'],
        coupon_cash=['0x1.f481928abc9b3p-1', '0x1.1b64dff632a81p-1', '0x1.7f0791197d67dp-2', '0x1.1d0a187af0e38p-2',
                     '0x1.be264b73542f0p-3'],
        maturity='0x1.07147f7338b08p-2', ki_settlement='0x1.28952c23d90f8p-4', ki='0x1.28f3c52a4c115p-4'),
    'phoenix_EXPIRY_True_continuous': dict(
        price='-0x1.6d4c53a5a8236p+0', pv='-0x1.6d4c53a5a8236p+0',
        ko=['0x1.80268d2a30d98p-2', '0x1.24f6b675401c7p-3', '0x1.297c923798da7p-4', '0x1.7e337bafaeb13p-5',
            '0x1.0ff5c8d9205ddp-5'],
        ko_cash=['0x0.0p+0', '0x0.0p+0', '0x0.0p+0', '0x0.0p+0', '0x0.0p+0'],
        coupon=['0x1.f9c29f0040534p-1', '0x1.1f8ac010c2594p-1', '0x1.8592e8190b814p-2', '0x1.225938d8a2872p-2',
                '0x1.c6e799a2b0bd6p-3'],
        coupon_cash=['0x1.f481928abc9b2p-1', '0x1.1dae56df6ef54p-1', '0x1.9318f1b3383bdp-2', '0x1.3e9e5fbb983b3p-2',
                     '0x1.0a9042086d3edp-2'],
        maturity='0x1.8e200177e7d0bp-3', ki_settlement='0x1.14539380762d9p-3', ki='0x1.158a7b0b813e0p-3'),
    'phoenix_EXPIRY_True_discrete': dict(
        price='0x1.05596bfe6b212p-3', pv='0x1.05596bfe6b212p-3',
        ko=['0x1.80268d2a30d98p-2', '0x1.24f6b6753fb69p-3', '0x1.297c9237995dcp-4', '0x1.7e337bafaf540p-5',
            '0x1.0ff5c8d920a62p-5'],
        ko_cash=['0x0.0p+0', '0x0.0p+0', '0x0.0p+0', '0x0.0p+0', '0x0.0p+0'],
        coupon=['0x1.f9c29f0040534p-1', '0x1.1f8ac010c2545p-1', '0x1.8592e8190baeep-2', '0x1.225938d8a2983p-2',
                '0x1.c6e799a2b0bedp-3'],
        coupon_cash=['0x1.f481928abc9b3p-1', '0x1.1dae56df6eeedp-1', '0x1.9318f1b3386c2p-2', '0x1.3e9e5fbb98526p-2',
                     '0x1.0a9042086d45ap-2'],
        maturity='0x1.07147f7338b08p-2', ki_settlement='0x1.28952c23d90f8p-4', ki='0x1.28f3c52a4c116p-4'),
    'phoenix_INSTANT_False_continuous': dict(
        price='-0x1.8292f0683a35ep+0', pv='-0x1.8292f0683a35ep+0',
        ko=['0x1.80268d2a30d98p-2', '0x1.24f6b675401c7p-3', '0x1.297c923798da7p-4', '0x1.7e337bafaeb13p-5',
            '0x1.0ff5c8d9205ddp-5'],
        ko_cash=['0x0.0p+0', '0x0.0p+0', '0x0.0p+0', '0x0.0p+0', '0x0.0p+0'],
        coupon=['0x1.f9c29f0040534p-1', '0x1.1f8ac010c2594p-1', '0x1.8592e8190b814p-2', '0x1.225938d8a2872p-2',
                '0x1.c6e799a2b0bd6p-3'],
        coupon_cash=['0x1.f8327c6391e62p-1', '0x1.1da1a4e2d36d3p-1', '0x1.81b29149b3245p-2', '0x1.1e80be5867d65p-2',
                     '0x1.bf62c224cdd2cp-3'],
        maturity='0x1.8e200177e7d0bp-3', ki_settlement='0x1.14539380762d9p-3', ki='0x1.158a7b0b813ddp-3'),
    'phoenix_INSTANT_False_discrete': dict(
        price='0x1.6c9217a769bffp-5', pv='0x1.6c9217a769bffp-5',
        ko=['0x1.80268d2a30d98p-2', '0x1.24f6b6753fb69p-3', '0x1.297c9237995dcp-4', '0x1.7e337bafaf540p-5',
            '0x1.0ff5c8d920a62p-5'],
        ko_cash=['0x0.0p+0', '0x0.0p+0', '0x0.0p+0', '0x0.0p+0', '0x0.0p+0'],
        coupon=['0x1.f9c29f0040534p-1', '0x1.1f8ac010c2545p-1', '0x1.8592e8190baeep-2', '0x1.225938d8a2983p-2',
                '0x1.c6e799a2b0bedp-3'],
        coupon_cash=['0x1.f8327c6391e62p-1', '0x1.1da1a4e2d3683p-1', '0x1.81b29149b3512p-2', '0x1.1e80be5867e71p-2',
                     '0x1.bf62c224cdd48p-3'],
        maturity='0x1.07147f7338b08p-2', ki_settlement='0x1.28952c23d90f8p-4', ki='0x1.28f3c52a4c115p-4'),
    'phoenix_INSTANT_True_continuous': dict(
        price='-0x1.68ff567e9edfcp+0', pv='-0x1.68ff567e9edfcp+0',
        ko=['0x1.80268d2a30d98p-2', '0x1.24f6b675401c7p-3', '0x1.297c923798da7p-4', '0x1.7e337bafaeb13p-5',
            '0x1.0ff5c8d9205ddp-5'],
        ko_cash=['0x0.0p+0', '0x0.0p+0', '0x0.0p+0', '0x0.0p+0', '0x0.0p+0'],
        coupon=['0x1.f9c29f0040534p-1', '0x1.1f8ac010c2594p-1', '0x1.8592e8190b814p-2', '0x1.225938d8a2872p-2',
                '0x1.c6e799a2b0bd6p-3'],
        coupon_cash=['0x1.f8327c6391e62p-1', '0x1.1ff20c2158441p-1', '0x1.95f3b4fa02734p-2', '0x1.404c03c4da227p-2',
                     '0x1.0b52911f08fa0p-2'],
        maturity='0x1.8e200177e7d0bp-3', ki_settlement='0x1.14539380762d9p-3', ki='0x1.158a7b0b813e0p-3'),
    'phoenix_INSTANT_True_discrete': dict(
        price='0x1.27c15536b53aep-3', pv='0x1.27c15536b53aep-3',
        ko=['0x1.80268d2a30d98p-2', '0x1.24f6b6753fb69p-3', '0x1.297c9237995dcp-4', '0x1.7e337bafaf540p-5',
            '0x1.0ff5c8d920a62p-5'],
        ko_cash=['0x0.0p+0', '0x0.0p+0', '0x0.0p+0', '0x0.0p+0', '0x0.0p+0'],
        coupon=['0x1.f9c29f0040534p-1', '0x1.1f8ac010c2545p-1', '0x1.8592e8190baeep-2', '0x1.225938d8a2983p-2',
                '0x1.c6e799a2b0bedp-3'],
        coupon_cash=['0x1.f8327c6391e62p-1', '0x1.1ff20c21583d8p-1', '0x1.95f3b4fa02a39p-2', '0x1.404c03c4da3a2p-2',
                     '0x1.0b52911f0900ep-2'],
        maturity='0x1.07147f7338b08p-2', ki_settlement='0x1.28952c23d90f8p-4', ki='0x1.28f3c52a4c116p-4'),
    'phoenix_one_record': dict(
        price='0x1.c38552e3d9ed7p-1', pv='0x1.c38552e3d9ed7p-1',
        ko=['0x1.af3a05fa9a9f6p-2'],
        ko_cash=['0x0.0p+0'],
        coupon=['0x1.c80f040be46f2p-1'],
        coupon_cash=['0x1.c38552e3d9ed7p-1'],
        maturity='0x1.2862fd02b2b06p-1', ki_settlement='0x0.0p+0', ki='0x0.0p+0'),
    'phoenix_single': dict(
        price='-0x1.68ff567e9edfcp+0', pv='-0x1.68ff567e9edfcp+0',
        ko=['0x1.80268d2a30d98p-2', '0x1.24f6b675401c7p-3', '0x1.297c923798da7p-4', '0x1.7e337bafaeb13p-5',
            '0x1.0ff5c8d9205ddp-5'],
        ko_cash=['0x0.0p+0', '0x0.0p+0', '0x0.0p+0', '0x0.0p+0', '0x0.0p+0'],
        coupon=['0x1.f9c29f0040534p-1', '0x1.1f8ac010c2594p-1', '0x1.8592e8190b814p-2', '0x1.225938d8a2872p-2',
                '0x1.c6e799a2b0bd6p-3'],
        coupon_cash=['0x1.f8327c6391e62p-1', '0x1.1ff20c2158441p-1', '0x1.95f3b4fa02734p-2', '0x1.404c03c4da227p-2',
                     '0x1.0b52911f08fa0p-2'],
        maturity='0x1.8e200177e7d0bp-3', ki_settlement='0x1.14539380762d9p-3', ki='0x1.158a7b0b813e0p-3'),
    'phoenix_val_jul3': dict(
        price='-0x1.0f58ad2802ddcp+1', pv='-0x1.0f58ad2802ddcp+1',
        ko=['0x0.0p+0', '0x1.70f079929e079p-4', '0x1.bf12991551d1bp-4'],
        ko_cash=['0x0.0p+0', '0x0.0p+0', '0x0.0p+0'],
        coupon=['0x1.0000000000000p+0', '0x1.be705cf26220ap-1', '0x1.640f0f14e1848p-1'],
        coupon_cash=['0x1.0000000000000p+0', '0x1.bcf4093d6f8ccp-1', '0x1.7689018128794p-1'],
        maturity='0x1.47f958f576e75p-1', ki_settlement='0x1.481912d62c74cp-3', ki='0x1.481b9e6f0641ep-3'),
    'phoenix_val_jul3_ko': dict(
        price='0x1.0000000000000p+0', pv='0x1.0000000000000p+0',
        ko=['0x1.0000000000000p+0', '0x0.0p+0', '0x0.0p+0'],
        ko_cash=['0x0.0p+0', '0x0.0p+0', '0x0.0p+0'],
        coupon=['0x1.0000000000000p+0', '0x0.0p+0', '0x0.0p+0'],
        coupon_cash=['0x1.0000000000000p+0', '0x0.0p+0', '0x0.0p+0'],
        maturity='0x0.0p+0', ki_settlement='0x0.0p+0', ki='0x0.0p+0'),
    'snowball_first_step': dict(
        price='0x1.0feddc8968eeap+3', pv='0x1.0feddc8968eeap+3',
        ko=['0x1.ecac3f49fc36bp-6', '0x1.7579ba1a5f01ep-2', '0x1.0bfebdf00edaep-3'],
        ko_cash=['0x1.ca0c53c56fb5cp-3', '0x1.d09dd65ec0bb9p+1', '0x1.9fb2f685817afp+0'],
        coupon=None,
        coupon_cash=None,
        maturity='0x1.7c572c55dd7d5p-2', ki_settlement='0x1.a593da8c713ccp-4', ki='0x1.a5a3c213be7acp-4'),
    'snowball_one_record_no_ki': dict(
        price='0x1.751d806c444b3p+3', pv='0x1.751d806c444b3p+3',
        ko=['0x1.af3a05fa9a9f6p-2'],
        ko_cash=['0x1.8cf66f80dc1d6p+1'],
        coupon=None,
        coupon_cash=None,
        maturity='0x1.2862fd02b2b06p-1', ki_settlement='0x0.0p+0', ki='0x0.0p+0'),
    'snowball_repro_single': dict(
        price='-0x1.3355224f29360p-3', pv='-0x1.3355224f29360p-3',
        ko=['0x1.80268d2a30d98p-2', '0x1.24f6b6753fbccp-3', '0x1.297c923798b41p-4', '0x1.7e337bafaec5ep-5',
            '0x1.0ff5c8d9207c2p-5'],
        ko_cash=['0x1.d0470cacfcf45p-1', '0x1.6cc49e5d73896p-1', '0x1.11d964447039dp-1', '0x1.d6c721a2d331ap-2',
                 '0x1.a1af5fb8d7d97p-2'],
        coupon=None,
        coupon_cash=None,
        maturity='0x1.015de9d98cb2ep-3', ki_settlement='0x1.a115ab1ed1b24p-3', ki='0x1.a9edf8ed21a3bp-3'),
    'snowball_reverse_single': dict(
        price='0x1.6fa63755af80dp+3', pv='0x1.6fa63755af80dp+3',
        ko=['0x1.c6cc5f9052a3ap-2'],
        ko_cash=['0x1.a2a950493ab12p+1'],
        coupon=None,
        coupon_cash=None,
        maturity='0x1.1c99d037d6ae2p-1', ki_settlement='0x0.0p+0', ki='0x0.0p+0'),
    'snowball_val_jul3_alive': dict(
        price='0x1.1d45eb3cb019bp+3', pv='0x1.1d45eb3cb019bp+3',
        ko=['0x0.0p+0', '0x1.aaa3f038825b6p-2', '0x1.0f5bbda80d599p-3'],
        ko_cash=['0x0.0p+0', '0x1.09661986d5612p+2', '0x1.a4f30c830fd56p+0'],
        coupon=None,
        coupon_cash=None,
        maturity='0x1.7011bf7be31a3p-2', ki_settlement='0x1.7671c5de4f7aep-4', ki='0x1.767c646ddcd9cp-4'),
    'snowball_val_jul3_ko': dict(
        price='0x1.dc0e070381c0ep+2', pv='0x1.dc0e070381c0ep+2',
        ko=['0x1.0000000000000p+0', '0x0.0p+0', '0x0.0p+0'],
        ko_cash=['0x1.dc0e070381c0ep+2', '0x0.0p+0', '0x0.0p+0'],
        coupon=None,
        coupon_cash=None,
        maturity='0x0.0p+0', ki_settlement='0x0.0p+0', ki='0x0.0p+0'),
}
V2_FOLDED_PIN = '0x1.14b5db17456f1p+0'
V2_COMPONENT_PINS = {
    'coupon': '0x0.0p+0',
    'ko': '0x1.82c813ff2d8f7p+1',
    'pending': '0x0.0p+0',
    'price': '-0x1.3355224f29360p-3',
    'reconciliation_error': '0x1.0000000000000p-50',
    'terminal': '-0x1.95fd66242022fp+1',
}


def _hex(values):
    return None if values is None else [float(v).hex() for v in values]


@pytest.mark.parametrize("name", sorted(V2_PINS))
def test_v2_without_a_shared_instant_prices_and_reports_bitwise(name):
    """Every product without a shared instant (Snowball, Phoenix in every coupon variant, KO-reset, the valuation
    instant, the first step): price and statistics bitwise against the pins taken before T4."""
    cls, product, e = _v2_pinned()[name]
    pin = V2_PINS[name]
    assert cls(QuadV2Params()).price(product, e).hex() == pin["price"]
    stats = cls(QuadV2Params()).calculate_event_stats(product, e)
    assert float(stats.pv).hex() == pin["pv"]
    assert _hex(stats.observations.probability) == pin["ko"]
    assert _hex(stats.observations.discounted_cash) == pin["ko_cash"]
    assert _hex(None if stats.coupons is None else stats.coupons.probability) == pin["coupon"]
    assert _hex(None if stats.coupons is None else stats.coupons.discounted_cash) == pin["coupon_cash"]
    assert float(stats.maturity.probability).hex() == pin["maturity"]
    assert float(stats.ki_settlement.probability).hex() == pin["ki_settlement"]
    assert float(stats.knock_in.probability).hex() == pin["ki"]


def test_v2_without_a_shared_instant_folds_and_splits_bitwise():
    from test_asset.equity.engine_pde.test_same_instant_ko import _overlay

    product, e = repro(drop_jul4=True), env()
    folded = v2().price_folded(product, e, _overlay(product, e, [1.0, 2.0, 3.0, 5.0, 6.0]))
    assert folded.hex() == V2_FOLDED_PIN
    assert {k: float(v).hex() for k, v in v2().price_components(product, e).items()} == V2_COMPONENT_PINS
