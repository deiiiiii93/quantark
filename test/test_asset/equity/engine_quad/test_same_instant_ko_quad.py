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
