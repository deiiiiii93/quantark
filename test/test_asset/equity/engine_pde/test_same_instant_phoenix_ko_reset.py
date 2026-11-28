"""Same-instant records on the Phoenix and KO-reset PDE core (0.16.0 DCN/dated run, T8: spec D10/D14, AC7).

A dated note on a US BUSINESS_DAYS env with PRECEDING holiday adjustment: 2025-07-04 is a US holiday, so its record
sits on 2025-07-03's env instant (env time 0.5). D14 zero-gap rule: the records of one instant are consecutive
observations with zero elapsed time, at one spot, in contract order.

* Phoenix: every record of the instant is its own coupon date (never dropped). Record k, if the note is still alive,
  pays its coupon (memory: plus the arrears since the last paid coupon) and then tests its knock-out; a knock-out ends
  the note. The PDE stages the instant as ONE event on every memory state of the fan (``PhoenixPDESolver
  ._apply_phoenix_group``): one piecewise cell average whose region branches are that chain (nodal: the chain on the
  nodes' own indicators), the knock-in first (the knock-out wins a same-instant tie), the termination surfaces W with
  it (EXPIRY: a coupon earned by record 1 is paid when the note ends -- at record 2's settlement if record 2 knocks
  out at the same instant). The statistics apply the same chain column by column, so each record keeps its own
  knock-out, trigger and coupon-cash rows.
* KO-reset: PDE against MC on a schedule-1 pair, a knock-in pair, a schedule-2 pair; REBASED is refused by the PDE
  (pre-existing); a knock-out-only duplicate is bitwise harmless.

Products, env and exact references are the QUAD task's (``engine_quad/test_same_instant_ko_quad.py``).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from datetime import datetime

import pytest
from golden_compare import assert_pins_close

from test_asset.equity.engine_pde.test_same_instant_ko import HESTON, JUL3, JUL4, _overlay, _p_breach, env
from test_asset.equity.engine_quad.test_same_instant_ko_quad import (
    COUPON_BARRIER,
    KOR_CASES,
    PHX_DATES,
    _decided_legs,
    _exact_phoenix_pair,
    _phoenix_variant,
    _zero_gap_expiry_cash,
    ko_reset,
    phoenix,
    phoenix_at_maturity,
    phoenix_pair,
    phoenix_repro,
    phoenix_settled_pair,
)
from quantark.asset.equity.engine.mc import HestonPhoenixMCEngine, PhoenixMCEngine, SnowballMCEngine
from quantark.asset.equity.engine.pde import PhoenixPDESolver
from quantark.asset.equity.engine.pde.core.march import march
from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest, market_scope, prepare
from quantark.asset.equity.engine.pde.core.statistics import adjoint
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import HestonPhoenixPDESolver, LocalVolPhoenixPDESolver
from quantark.asset.equity.engine.quad import PhoenixQuadEngine
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord
from quantark.util.enum import CouponPayType, PostKOScheduleMode
from quantark.util.enum.engine_enums import EventProjectionMode
from quantark.util.exceptions import ValidationError

MC_PATHS = 400_000
HESTON_PHX_GRID = dict(n_x=128, n_v=12, n_t=48)


def pde(**kwargs):
    return PhoenixPDESolver(PDEParams(**kwargs))


def _ko_rows(stats):
    return list(stats.observations.probability)


def _coupon_rows(stats):
    return list(stats.coupons.probability)


# --- AC7: the exact reference ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("b1,b2", [(103.0, 100.0), (100.0, 103.0)], ids=["b2_below_b1", "b2_above_b1"])
def test_phoenix_pair_matches_the_exact_reference_and_converges(b1, b2):
    """AC7 (1D PDE): the pair alone (no KI, memory on): rel 1e-3 at the default grid; the error does not grow under
    refinement down to the route's own floor (the AC2 ruling: ~1e-7..1e-6, a single record's too)."""
    product, e = phoenix_pair(b1, b2), env()
    ref, _, _ = _exact_phoenix_pair(product, e)
    err = {acc: abs(pde(accuracy=acc).price(product, e) / ref - 1.0) for acc in ("fast", "standard", "high")}
    assert err["standard"] <= 1e-3, err
    assert err["standard"] <= max(err["fast"], 1e-6), err
    assert err["high"] <= max(err["standard"], 1e-6), err


def test_phoenix_pair_nodal_events_follow_the_zero_gap_rule():
    """Nodal mode: the chain on each node's own indicators -- the reference within the nodal first-order error."""
    product, e = phoenix_pair(103.0, 100.0), env()
    ref, _, _ = _exact_phoenix_pair(product, e)
    value = pde(event_projection=EventProjectionMode.NODAL).price(product, e)
    assert abs(value / ref - 1.0) <= 1e-2, (value, ref)


@pytest.mark.parametrize("b1,b2", [(103.0, 100.0), (100.0, 103.0)], ids=["b2_below_b1", "b2_above_b1"])
def test_phoenix_pair_at_maturity_matches_the_exact_reference(b1, b2):
    """AC5 maturity instant: the pair is the terminal observation (exercise on the 07-04 holiday)."""
    product, e = phoenix_at_maturity(b1, b2), env()
    r1, r2 = product.resolve_ko_observations(e)
    assert r1.observation_time == r2.observation_time == product.get_maturity(e)
    ref, _, _ = _exact_phoenix_pair(product, e)
    value = pde().price(product, e)
    assert abs(value / ref - 1.0) <= 1e-3, (value, ref)


def _expiry_pair(b1, b2, memory):
    product = phoenix_settled_pair(b1, b2)
    product.coupon_config = replace(product.coupon_config, coupon_pay_type=CouponPayType.EXPIRY, memory_coupon=memory)
    product.validate()
    return product


def _exact_expiry_pair(product, e, memory):
    r1, r2 = product.resolve_ko_observations(e)
    t1 = r1.observation_time
    assert t1 == r2.observation_time and r1.settlement_time < r2.settlement_time
    levels = sorted({COUPON_BARRIER, float(r1.barrier), float(r2.barrier)})
    edges = [0.0] + levels + [math.inf]
    above = lambda b: 0.0 if b == math.inf else (1.0 if b == 0.0 else _p_breach(b, t1))   # noqa: E731
    return sum((above(lo) - above(hi)) * _zero_gap_expiry_cash(product, e, 0.5 * (lo + hi) if hi < math.inf
                                                               else lo + 1.0, memory)
               for lo, hi in zip(edges[:-1], edges[1:]))


@pytest.mark.parametrize("memory", [True, False], ids=["memory", "no_memory"])
@pytest.mark.parametrize("b1,b2", [(103.0, 100.0), (100.0, 103.0)], ids=["b2_below_b1", "b2_above_b1"])
def test_phoenix_expiry_pair_matches_the_exact_reference(b1, b2, memory):
    """EXPIRY pay (V1/V2's rule, exact): record 1's coupon is deferred to the note's end, which a knock-out by record 2
    at the same instant makes record 2's settlement (07-15; record 1 settles 07-08) -- through the termination
    surfaces. The statistics reconcile."""
    product, e = _expiry_pair(b1, b2, memory), env()
    ref = _exact_expiry_pair(product, e, memory)
    value = pde().price(product, e)
    assert abs(value / ref - 1.0) <= 1e-3, (value, ref)
    stats = pde().calculate_event_stats(product, e)
    assert abs(stats.reconciliation_error) <= 1e-6 * max(1.0, abs(stats.pv))


@pytest.mark.parametrize("b1,b2", [(103.0, 100.0), (100.0, 103.0)], ids=["b2_below_b1", "b2_above_b1"])
@pytest.mark.parametrize("maturity", [False, True], ids=["interior", "maturity"])
def test_phoenix_pair_statistics_attribute_each_record_exactly(b1, b2, maturity):
    """Each record's knock-out and coupon probability (coupon: alive before the record and its barrier met) equals
    its band's BS probability; one row per contract record, dated 07-03 / 07-04."""
    product, e = (phoenix_at_maturity(b1, b2) if maturity else phoenix_pair(b1, b2)), env()
    _, ko, coupon = _exact_phoenix_pair(product, e)
    stats = pde().calculate_event_stats(product, e)
    assert stats.observations.dates == (JUL3, JUL4)
    assert _ko_rows(stats) == pytest.approx(list(ko), abs=1e-4)
    assert _coupon_rows(stats) == pytest.approx(list(coupon), abs=1e-4)
    assert abs(stats.reconciliation_error) <= 1e-6 * max(1.0, abs(stats.pv))


# --- AC7: the repro against MC ---------------------------------------------------------------------------------------

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
    """AC7: the repro Phoenix (KI 70) within 3 SE + 1e-3 relative of MC; within 2.5e-3 of QUAD V1 (the single
    record's own PDE-V1 gap). Before: the PDE dropped one record of the pair (duplicate == single, -1.41192)."""
    ref, se, _ = phoenix_mc[variant]
    value = pde().price(phoenix_repro(variant), env())
    assert abs(value - ref) <= 3.0 * se + 1e-3 * abs(ref), (value, ref, se)
    v1 = PhoenixQuadEngine(params=QuadParams(grid_points=1001)).price(phoenix_repro(variant), env())
    assert abs(value - v1) <= 2.5e-3, (value, v1)


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["duplicate", "stepdown"])
def test_phoenix_repro_statistics_agree_with_mc(phoenix_mc, variant):
    """AC7: each record's knock-out and coupon probability within 3 binomial SE of MC; the duplicate's 07-04
    knock-out row is exactly 0 (its region is inside 07-03's) while its coupon row counts."""
    _, _, mc_stats = phoenix_mc[variant]
    stats = pde().calculate_event_stats(phoenix_repro(variant), env())
    for rows, mc_rows in ((_ko_rows(stats), _ko_rows(mc_stats)), (_coupon_rows(stats), _coupon_rows(mc_stats))):
        assert len(rows) == len(mc_rows) == len(PHX_DATES)
        for p, m in zip(rows, mc_rows):
            se = math.sqrt(m * (1.0 - m) / MC_PATHS)
            assert abs(p - m) <= 3.0 * se, (rows, mc_rows)
    if variant == "duplicate":
        assert _ko_rows(stats)[3] == 0.0 and _coupon_rows(stats)[3] > 0.1
    assert abs(stats.reconciliation_error) <= 1e-6 * max(1.0, abs(stats.pv))


@pytest.mark.slow
def test_local_vol_phoenix_step_down_agrees_with_mc(phoenix_mc):
    """The local-vol Phoenix PDE (flat grid surface: the Black-Scholes model) shares the stage."""
    ref, se, _ = phoenix_mc["stepdown"]
    value = LocalVolPhoenixPDESolver(PDEParams()).price(phoenix_repro("stepdown"), env(grid=True))
    assert abs(value - ref) <= 3.0 * se + 1e-3 * abs(ref), (value, ref, se)


@pytest.mark.slow
def test_heston_phoenix_step_down_agrees_with_heston_mc():
    """The Heston Phoenix PDE (native ADI march through the same adapter stage) against Heston MC."""
    product, e = phoenix_repro("stepdown"), env()
    value = HestonPhoenixPDESolver(HESTON, **HESTON_PHX_GRID).price(product, e)
    mc = HestonPhoenixMCEngine(HESTON, params=MCParams(num_paths=MC_PATHS, seed=7))
    ref = mc.price(product, e)
    se = mc.get_last_std_error()
    assert abs(value - ref) <= 3.0 * se + 1e-3 * abs(ref), (value, ref, se)


# --- the valuation instant -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("spot", [90.0, 101.0], ids=["both_paid", "paid_then_step_down_knock_out"])
def test_phoenix_valuation_instant_coupons_settle_on_their_own_dates(spot):
    """Valued ON 07-03 (both records at env time 0), the pair settling 07-08 / 07-15: each decided coupon at its OWN
    settlement, the step-down knock-out's redemption at record 2's; nothing after the pair but the zero maturity
    redemption. The knock-out band is decided before the march (exact). The both-paid band marches the rest: its
    continuation carries the PDE's pre-existing far-field edge row (after the last observation the upper edge pays the
    last record's redemption, 10 here), measured +3.17e-5 -- the same as ONE record with that redemption valued on
    its own date (+3.17e-5 standard, +3.13e-5 high; 0.0 with a zero redemption)."""
    e = env(JUL3, spot=spot)
    product = phoenix_settled_pair()
    legs, knocked_out, _ = _decided_legs(product, e, spot)
    expected = legs
    if knocked_out is None:
        m = product.get_maturity_payoff_v0(spot, pricing_env=e)
        expected += m * e.get_discount_factor(resolve_terminal_timing(product, e).payment_time)
        assert pde().price(product, e) == pytest.approx(expected, abs=5e-5)
    else:
        assert pde().price(product, e) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_phoenix_valuation_instant_coupons_with_a_continuation():
    """With records after the pair the price is the decided legs (own settlements) plus a continuation that does not
    depend on where the pair settles."""
    later = (datetime(2025, 9, 3), datetime(2025, 11, 3))
    e = env(JUL3, spot=90.0)
    a = phoenix_settled_pair(later=later)
    b = phoenix([ObservationRecord(observation_date=JUL3, barrier=103.0, settlement_date=datetime(2025, 7, 15)),
                 ObservationRecord(observation_date=JUL4, barrier=100.0, return_rate=0.10,
                                   settlement_date=datetime(2025, 7, 15))]
                + [ObservationRecord(observation_date=d, barrier=103.0) for d in later], ki=None)
    cont_a = pde().price(a, e) - _decided_legs(a, e, 90.0)[0]
    cont_b = pde().price(b, e) - _decided_legs(b, e, 90.0)[0]
    assert cont_a == pytest.approx(cont_b, rel=1e-9, abs=1e-12)
    assert cont_a > 0.1


@pytest.mark.parametrize("spot", [90.0, 101.0], ids=["both_paid", "paid_then_step_down_knock_out"])
def test_phoenix_valuation_instant_coupons_are_attributed_per_record(spot):
    """Each decided coupon on its own record's row (probability 1, cash at its own settlement), the knock-out on the
    deciding record; the statistics reconcile with the price."""
    later = (datetime(2025, 9, 3), datetime(2025, 11, 3))
    e = env(JUL3, spot=spot)
    product = phoenix_settled_pair(later=later)
    _, knocked_out, rows = _decided_legs(product, e, spot)
    stats = pde().calculate_event_stats(product, e)
    assert stats.pv == pytest.approx(pde().price(product, e), rel=1e-12)
    for k, cash in rows.items():
        assert stats.coupons.probability[k] == 1.0, (k, list(stats.coupons.probability))
        assert stats.coupons.discounted_cash[k] == pytest.approx(cash, rel=1e-12), (k, list(stats.coupons.discounted_cash))
    if knocked_out is not None:
        assert _ko_rows(stats)[knocked_out] == 1.0 and sum(_ko_rows(stats)) == 1.0
    assert abs(stats.reconciliation_error) <= 1e-6 * max(1.0, abs(stats.pv))


@pytest.mark.parametrize("spot,deciding", [(105.0, 0), (101.0, 1)], ids=["first_band", "step_down_band"])
def test_phoenix_valuation_instant_knock_out_pays_the_deciding_record(spot, deciding):
    """The repro step-down valued on 07-03: the record whose barrier the spot breaches first (contract order) knocks
    out; every record before it pays its coupon at its own settlement (here all settle on their observation)."""
    e = env(JUL3, spot=spot)
    product = phoenix_repro("stepdown")
    records = [r for r in product.resolve_ko_observations(e) if r.observation_time == 0.0]
    fractions = product.get_coupon_period_year_fractions(e)
    full = product.resolve_ko_observations(e)
    first = full.index(records[0])
    coupons = sum(product.get_coupon_payoff(first + k, year_fraction=fractions[first + k])
                  for k in range(deciding + 1))
    expected = (records[deciding].payoff + coupons) * e.get_discount_factor(records[deciding].settlement_time)
    assert pde().price(product, e) == pytest.approx(expected, rel=1e-12)
    stats = pde().calculate_event_stats(product, e)
    assert _ko_rows(stats)[deciding] == 1.0 and sum(_ko_rows(stats)) == 1.0
    assert _coupon_rows(stats)[:deciding + 1] == [1.0] * (deciding + 1)


@pytest.mark.slow
@pytest.mark.parametrize("spot", [75.0, 90.0], ids=["both_missed", "both_paid"])
def test_phoenix_valuation_instant_pair_without_knock_out_agrees_with_mc(spot):
    """Valued on 07-03, no knock-out there: the decided coupons are paid and the memory state the instant leaves
    counts the periods missed after the last paid one (both missed: two periods of arrears)."""
    e = env(JUL3, spot=spot)
    product = phoenix_repro("stepdown")
    mc = PhoenixMCEngine(params=MCParams(num_paths=MC_PATHS, seed=7))
    ref = mc.price(product, e)
    se = mc.get_last_std_error()
    value = pde().price(product, e)
    assert abs(value - ref) <= 3.0 * se + 1e-3 * abs(ref), (value, ref, se)


def test_phoenix_valuation_instant_carried_knock_in_disables_the_pair_knock_outs():
    """A knock-in carried into the valuation instant under ``disable_ko_after_ki``: neither record of the pair can
    knock out, both coupons are decided at the spot (101 >= 80) and the knocked-in note runs on -- the knocked-in
    regime's readout, within the PDE-V1 gap of QUAD V1."""
    e = env(JUL3, spot=101.0)

    def product():
        p = phoenix_repro("stepdown")
        p.barrier_config = replace(p.barrier_config, disable_ko_after_ki=True)
        p.validate()
        p._otc_lifecycle_knocked_in = True
        return p

    value = pde().price(product(), e)
    v1 = PhoenixQuadEngine(params=QuadParams(grid_points=1001)).price(product(), e)
    assert abs(value - v1) <= 2.5e-3, (value, v1)


# --- the forward adjoint pass ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("pay", ["INSTANT", "EXPIRY"])
@pytest.mark.parametrize("projection", [None, EventProjectionMode.NODAL], ids=["cell_average", "nodal"])
def test_forward_adjoint_reconstructs_through_the_phoenix_group(projection, pay):
    """The forward pass replays the group as the march's own map (the knock-in first, then the chain): the adjoint
    identity holds for the knock-out units, the maturity units and KIS, with a discrete knock-in on 07-03."""
    solver = pde(event_projection=projection)
    product = _phoenix_variant("stepdown", getattr(CouponPayType, pay), True, "discrete")
    p = prepare(solver, product, env(), OutputRequest(Operation.EVENT_STATS, event_stats=True))
    with market_scope(p.solver, p.env, p.market_fps), p.adapter.scope(p.product, p.env):
        result = march(p)
        forward = adjoint.forward_pass(p.statistics, result.states, result.captures, reconstruct=True)
        at = p.statistics._reader(result.states, result.captures)
    assert p.solver._phx_groups
    L = p.statistics.layout
    for c in [*L.ko, *L.maturity_unit, L.kis]:
        assert abs(forward.reconstructed[int(c)] - at(int(c))) <= 1e-10, (c,)


@pytest.mark.parametrize("pay", ["INSTANT", "EXPIRY"])
@pytest.mark.parametrize("memory", [True, False], ids=["memory", "no_memory"])
@pytest.mark.parametrize("projection", [None, EventProjectionMode.NODAL], ids=["cell_average", "nodal"])
def test_phoenix_group_statistics_reconcile(projection, memory, pay):
    """Every coupon variant: the statistics' columns sum to the price (the chain applied column by column)."""
    product = _phoenix_variant("stepdown", getattr(CouponPayType, pay), memory, "discrete")
    stats = pde(event_projection=projection).calculate_event_stats(product, env())
    assert abs(stats.reconciliation_error) <= 1e-6 * max(1.0, abs(stats.pv))
    assert stats.observations.dates[2:4] == (JUL3, JUL4)


# --- D10: cash-leg folding --------------------------------------------------------------------------------------------

def test_phoenix_folding_onto_a_same_instant_pair_is_refused():
    product, e = phoenix_repro("stepdown"), env()
    with pytest.raises(ValidationError, match="2025-07-03.*2025-07-04"):
        pde().price_folded(product, e, _overlay(product, e, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


def test_phoenix_folding_onto_a_duplicate_only_instant_binds_the_effective_record():
    """A duplicate (07-04 at 07-03's level): its knock-out never fires, so the cash folded onto it is never paid."""
    product, e = phoenix_repro("duplicate"), env()
    base = pde().price_folded(product, e, _overlay(product, e, [1.0, 2.0, 3.0, 0.0, 5.0, 6.0]))
    dominated = pde().price_folded(product, e, _overlay(product, e, [1.0, 2.0, 3.0, 40.0, 5.0, 6.0]))
    assert dominated == base


# --- products without a shared instant: bitwise --------------------------------------------------------------------

_PINS_PATH = os.path.join(os.path.dirname(__file__), "same_instant_phoenix_ko_reset_pins.json")
with open(_PINS_PATH) as _f:
    #: float.hex pins captured at 9fc56f6c BEFORE the T8 change (scratchpad t8_pins.py)
    PINS = json.load(_f)


def _annualized():
    from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix

    p = phoenix_repro("single")
    p2 = create_standard_phoenix(100.0, 100.0, 1.0, ko_barrier=103.0, ki_barrier=70.0, coupon_barrier=80.0,
                                 coupon_rate=0.08, num_observations=12, memory_coupon=True, is_annualized=True)
    p2.barrier_config = replace(p2.barrier_config, ko_observation_dates=None,
                                ko_observation_schedule=p.barrier_config.ko_observation_schedule)
    p2.maturity = None
    p2.initial_date = p.initial_date
    p2.exercise_date = p.exercise_date
    p2.validate()
    return p2


def _pinned_cases():
    e = env()
    out = {"phoenix_single": (pde, phoenix_repro("single"), e)}
    for pay in ("INSTANT", "EXPIRY"):
        for memory in (True, False):
            for ki in ("continuous", "discrete"):
                out[f"phoenix_{pay}_{memory}_{ki}"] = (
                    pde, _phoenix_variant("single", getattr(CouponPayType, pay), memory, ki), e)
    nodal = lambda: pde(event_projection=EventProjectionMode.NODAL)  # noqa: E731
    out.update({
        "phoenix_annualized": (pde, _annualized(), e),
        "phoenix_nodal": (nodal, phoenix_repro("single"), e),
        "phoenix_nodal_expiry": (nodal, _phoenix_variant("single", CouponPayType.EXPIRY, True, "discrete"), e),
        "phoenix_val_jul3": (pde, phoenix_repro("single"), env(JUL3, spot=90.0)),
        "phoenix_val_jul3_miss": (pde, phoenix_repro("single"), env(JUL3, spot=75.0)),
        "phoenix_val_jul3_ko": (pde, phoenix_repro("single"), env(JUL3, spot=105.0)),
        "phoenix_first_step": (pde, phoenix_repro("single"), env(datetime(2025, 7, 2))),
        "phoenix_one_record": (pde, phoenix([ObservationRecord(observation_date=JUL3, barrier=103.0)], ki=None), e),
        "phoenix_maturity_record": (pde, phoenix([ObservationRecord(observation_date=JUL3, barrier=103.0),
                                                  ObservationRecord(observation_date=datetime(2026, 1, 5),
                                                                    barrier=103.0)]), e),
        "phoenix_lv": (lambda: LocalVolPhoenixPDESolver(PDEParams()), phoenix_repro("single"), env(grid=True)),
        "phoenix_heston": (lambda: HestonPhoenixPDESolver(HESTON, **HESTON_PHX_GRID), phoenix_repro("single"), e),
        "ko_reset_absolute": (lambda: KOResetSnowballPDESolver(PDEParams()), ko_reset(), e),
    })
    return out


def _stats_pins(s):
    h = lambda x: float(x).hex()  # noqa: E731
    out = {"pv": h(s.pv), "ko": [h(p) for p in s.observations.probability],
           "ko_cash": [h(p) for p in s.observations.discounted_cash],
           "maturity": h(s.maturity.probability), "ki_settlement": h(s.ki_settlement.probability),
           "lifetime": h(s.expected_lifetime), "recon": h(s.reconciliation_error)}
    if s.knock_in is not None and s.knock_in.probability is not None:
        out["ki"] = h(s.knock_in.probability)
    if s.coupons is not None:
        out["coupon_cash"] = [h(p) for p in s.coupons.discounted_cash]
        if s.coupons.probability is not None:
            out["coupon"] = [h(p) for p in s.coupons.probability]
    if getattr(s, "path_dependent_cash", None) is not None:
        out["ledger"] = [h(p) for p in s.path_dependent_cash.discounted_cash]
    return out


PINNED = _pinned_cases()


@pytest.mark.parametrize("name", list(PINNED))
def test_products_without_a_shared_instant_price_and_report_bitwise(name):
    make, product, e = PINNED[name]
    assert_pins_close(float(make().price(product, e)).hex(), PINS[name]["price"])
    assert_pins_close(_stats_pins(make().calculate_event_stats(product, e)), PINS[name]["stats"])


# === KO-reset (D14: schedule-1 knock-outs, the knock-in, schedule-2 knock-outs of one instant) =====================

from test_asset.equity.engine_quad.test_same_instant_ko_quad import KOR_POST, KOR_PRE, NOV26, NOV27  # noqa: E402

_PRE_103 = [(d, 103.0) for d in KOR_PRE]
#: KOR_CASES plus a knock-in pair in the other order (80 on 07-03, then 75 on 07-04: the 07-04 record is inert)
KOR_PDE_CASES = dict(KOR_CASES, ki_pair_reversed=dict(ki=[(d, 80.0) if d == JUL3 else (d, 75.0) for d in KOR_PRE]
                                                      + [(JUL4, 75.0)]))


def kor_pde():
    return KOResetSnowballPDESolver(PDEParams())


@pytest.fixture(scope="module")
def ko_reset_mc():
    """Each case on Black-Scholes MC (400k paths, seed 7): price, SE and (knock-out-only pairs) the statistics."""
    out = {}
    for name, spec in KOR_PDE_CASES.items():
        mc = SnowballMCEngine(params=MCParams(num_paths=MC_PATHS, seed=7))
        value = mc.price(ko_reset(**spec), env())
        se = mc.get_last_std_error()
        stats = None
        if name in ("schedule1_step_down", "schedule2_step_down"):
            stats = SnowballMCEngine(params=MCParams(num_paths=MC_PATHS, seed=7)).calculate_event_stats(
                ko_reset(**spec), env())
        out[name] = (value, se, stats)
    return out


@pytest.mark.slow
@pytest.mark.parametrize("case", list(KOR_PDE_CASES))
def test_ko_reset_same_instant_records_agree_with_mc(ko_reset_mc, case):
    """D14 on the KO-reset PDE: within 3 SE + 1e-3 relative of MC, and within 0.05 of QUAD V1 (the single record's
    own PDE-V1 gap is ~0.03)."""
    ref, se, _ = ko_reset_mc[case]
    product = ko_reset(**KOR_PDE_CASES[case])
    value = kor_pde().price(product, env())
    assert abs(value - ref) <= 3.0 * se + 1e-3 * abs(ref), (value, ref, se, (value - ref) / se)
    v1 = KOResetSnowballQuadEngine(params=QuadParams(grid_points=1001)).price(product, env())
    assert abs(value - v1) <= 0.05, (value, v1)


@pytest.mark.slow
@pytest.mark.parametrize("case", ["schedule1_step_down", "schedule2_step_down"])
def test_ko_reset_same_instant_statistics_agree_with_mc(ko_reset_mc, case):
    """Each knock-out record of a same-instant pair is its own row (dated, one per contract record): per-record
    knock-out probability within 3 binomial SE of MC; the statistics reconcile."""
    _, _, mc = ko_reset_mc[case]
    stats = kor_pde().calculate_event_stats(ko_reset(**KOR_PDE_CASES[case]), env())
    rows, mc_rows = _ko_rows(stats), _ko_rows(mc)
    assert len(rows) == len(mc_rows) == 7
    assert list(stats.observations.dates) == list(mc.observations.dates)
    for p, m in zip(rows, mc_rows):
        se = math.sqrt(m * (1.0 - m) / MC_PATHS)
        assert abs(p - m) <= 3.0 * se, (rows, mc_rows)
    assert abs(stats.reconciliation_error) <= 1e-6 * max(1.0, abs(stats.pv))


def _ko_reset_duplicate(kind, rate):
    """``(product with a same-instant knock-out duplicate, the product without it, the duplicate's row)``."""
    if kind == "schedule1":
        rec = (JUL4, 103.0) if rate is None else (JUL4, 103.0, rate)
        return ko_reset(pre=_PRE_103 + [rec]), ko_reset(), 3
    post = [(KOR_POST[0], 95.0), (NOV26, 95.0), (KOR_POST[2], 95.0)]
    rec = (NOV27, 95.0) if rate is None else (NOV27, 95.0, rate)
    return ko_reset(post=post[:2] + [rec] + post[2:]), ko_reset(post=post), 3 + 2


@pytest.mark.parametrize("kind", ["schedule1", "schedule2"])
@pytest.mark.parametrize("rate", [None, 0.40], ids=["same_payoff", "other_payoff"])
def test_ko_reset_knock_out_duplicate_prices_bitwise(kind, rate):
    """A same-instant knock-out duplicate (same level, any payoff) prices bitwise as the product without it."""
    dup, base, _ = _ko_reset_duplicate(kind, rate)
    assert kor_pde().price(dup, env()) == kor_pde().price(base, env())


@pytest.mark.parametrize("kind", ["schedule1", "schedule2"])
def test_ko_reset_knock_out_duplicate_row_is_zero(kind):
    """The duplicate's statistics row is exactly 0, the other rows the deduplicated product's."""
    dup, base, row = _ko_reset_duplicate(kind, 0.40)
    rows = _ko_rows(kor_pde().calculate_event_stats(dup, env()))
    base_rows = _ko_rows(kor_pde().calculate_event_stats(base, env()))
    assert rows[row] == 0.0
    assert rows[:row] + rows[row + 1:] == base_rows


def test_ko_reset_rebased_stays_refused_on_the_pde():
    """The KO-reset PDE prices ABSOLUTE schedule 2 only (pre-existing); a REBASED contract with a pair is refused."""
    product = ko_reset(pre=_PRE_103 + [(JUL4, 100.0, 0.10)], mode=PostKOScheduleMode.REBASED,
                       post=[(k / 12.0, 95.0) for k in range(1, 7)])
    with pytest.raises(ValidationError, match="ABSOLUTE"):
        kor_pde().price(product, env())


@pytest.mark.parametrize("case", ["schedule1_step_down", "schedule2_step_down", "schedule1_and_ki_pair"])
@pytest.mark.parametrize("projection", [None, EventProjectionMode.NODAL], ids=["cell_average", "nodal"])
def test_ko_reset_forward_adjoint_reconstructs_through_the_group(projection, case):
    """The forward pass replays each schedule's group (and the widest knock-in of a knock-in pair) as the march's own
    map: the adjoint identity holds for every knock-out unit and both maturity units."""
    solver = KOResetSnowballPDESolver(PDEParams(event_projection=projection))
    p = prepare(solver, ko_reset(**KOR_PDE_CASES[case]), env(),
                OutputRequest(Operation.EVENT_STATS, event_stats=True))
    with market_scope(p.solver, p.env, p.market_fps), p.adapter.scope(p.product, p.env):
        result = march(p)
        forward = adjoint.forward_pass(p.statistics, result.states, result.captures, reconstruct=True)
        at = p.statistics._reader(result.states, result.captures)
    assert p.solver._pre_ko_groups or p.solver._post_ko_groups
    L = p.statistics.layout
    for c in [*L.ko, *L.post_ko, *L.maturity_unit]:
        assert abs(forward.reconstructed[int(c)] - at(int(c))) <= 1e-10, (c,)


@pytest.mark.parametrize("spot,deciding", [(105.0, 0), (101.0, 1)], ids=["first_band", "step_down_band"])
def test_ko_reset_valuation_instant_knock_out_pays_the_deciding_record(spot, deciding):
    """The schedule-1 step-down valued on 07-03: the first record (contract order) whose barrier the spot breaches
    pays its own cash, and its own row takes probability 1 (rows: the pair, then schedule 2)."""
    e = env(JUL3, spot=spot)
    product = ko_reset(**KOR_PDE_CASES["schedule1_step_down"])
    pre = [r for r in KOResetSnowballPDESolver(PDEParams())._get_cached_pre_ko_records(e, product)
           if r.observation_time == 0.0]
    assert len(pre) == 2
    rec = pre[deciding]
    assert kor_pde().price(product, e) == pytest.approx(rec.payoff * e.get_discount_factor(rec.settlement_time),
                                                        rel=1e-12)
    rows = _ko_rows(kor_pde().calculate_event_stats(product, e))
    assert sum(rows) == 1.0 and rows[deciding] == 1.0, rows


# === KO-reset T5a: the instant unit, the Heston valuation instant, folding =========================================

#: overlapping knock-in / knock-out levels at the 07-03 / 07-04 instant, where the instant unit (CP2 ruling: every
#: schedule-1 knock-out record, then the knock-in, then every schedule-2 record) and a date-by-date order differ
KOR_OVERLAP = {
    # schedule-1 step-down at 90 on 07-04 under a 95 knock-in on 07-03: the 90-95 band knocks out on 07-04 (instant
    # unit; date by date it knocked in on 07-03 first)
    "pre_overlap": dict(pre=_PRE_103 + [(JUL4, 90.0, 0.10)], ki=[(d, 75.0) for d in KOR_PRE[:2]] + [(JUL3, 95.0)]),
    # a schedule-2 record on 07-03 (95) and a knock-in on 07-04 (97): the 95-97 band knocks in and then out on the
    # 07-03 schedule-2 record (instant unit; date by date that record had passed before the 07-04 knock-in)
    "post_overlap": dict(ki=[(d, 75.0) for d in KOR_PRE] + [(JUL4, 97.0)],
                         post=[(JUL3, 95.0)] + [(d, 95.0) for d in KOR_POST]),
}


@pytest.mark.slow
@pytest.mark.parametrize("case", list(KOR_OVERLAP))
def test_ko_reset_overlapping_levels_follow_the_instant_unit(case):
    """QUAD V1 orders the instant as one unit, as the PDE (T8) and MC do: each within 3 SE + 1e-3 relative of MC.
    Before T5a V1 ordered date by date (pre_overlap 4.871569, post_overlap 4.194909 against MC 6.0317 / 4.3333,
    SE 0.035 at 200k paths)."""
    product, e = ko_reset(**KOR_OVERLAP[case]), env()
    mc = SnowballMCEngine(params=MCParams(num_paths=MC_PATHS, seed=7))
    ref = mc.price(product, e)
    se = mc.get_last_std_error()
    for engine in (kor_pde(), KOResetSnowballQuadEngine(params=QuadParams(grid_points=1001)),
                   KOResetSnowballQuadEngine(params=QuadParams(grid_points=1001,
                                                               event_projection=EventProjectionMode.NODAL))):
        value = engine.price(product, e)
        assert abs(value - ref) <= 3.0 * se + 1e-3 * abs(ref), (type(engine).__name__, value, ref, se,
                                                                (value - ref) / se)


@pytest.mark.parametrize("case", list(KOR_OVERLAP))
def test_ko_reset_overlapping_levels_v1_agrees_with_the_pde(case):
    """The instant unit on QUAD V1 and the PDE: within the single record's own PDE-V1 gap (0.05); the statistics
    reconcile on both."""
    product, e = ko_reset(**KOR_OVERLAP[case]), env()
    v1 = KOResetSnowballQuadEngine(params=QuadParams(grid_points=1001))
    assert abs(v1.price(product, e) - kor_pde().price(product, e)) <= 0.05
    for engine in (v1, kor_pde()):
        stats = engine.calculate_event_stats(product, e)
        assert abs(stats.reconciliation_error) <= 1e-6 * max(1.0, abs(stats.pv))


@pytest.mark.parametrize("spot,deciding", [(105.0, 0), (101.0, 1)], ids=["first_band", "step_down_band"])
def test_heston_ko_reset_valuation_instant_knock_out_pays_the_deciding_record(spot, deciding):
    """The Heston (native ADI) KO-reset valued on 07-03: the knock-out decided at valuation pays the decision's record
    (``_knocked_out_value(decision)``, as the 1D route) -- before T5a it paid the instant's first record (15.0 where
    the step-down record decides, 10.0)."""
    from quantark.asset.equity.engine.pde.ko_reset_vol_pde_solvers import HestonKOResetSnowballPDESolver

    e = env(JUL3, spot=spot)
    product = ko_reset(**KOR_PDE_CASES["schedule1_step_down"])
    decision = product.decide_observations_at_valuation(spot, e)
    assert decision.knocked_out and decision.ko_index == deciding
    expected = kor_pde()._knocked_out_value(product, e, decision)
    heston = HestonKOResetSnowballPDESolver(HESTON, n_x=96, n_v=16, n_t=64)
    assert heston.price(product, e) == pytest.approx(expected, rel=1e-14, abs=1e-14)
    assert kor_pde().price(product, e) == pytest.approx(expected, rel=1e-14, abs=1e-14)
    rows = _ko_rows(heston.calculate_event_stats(product, e))
    assert rows[deciding] == 1.0 and sum(rows) == 1.0, rows


def _kor_overlay(product, e, pre_amounts, post_amounts):
    """Per-unit cash on every knock-out slot (schedule 1, then schedule 2, contract order) and both maturity branches
    (the folding module's own slots)."""
    from quantark.cashleg.folding import FoldedEventCash, FoldOverlay, _event_structure

    st = _event_structure(product, e)
    slot = lambda s, x: FoldedEventCash(s.schedule, s.determination_time, s.payment_time, float(x))  # noqa: E731
    pre = [s for s in st.ko if s.schedule == "pre"]
    post = [s for s in st.ko if s.schedule == "post"]
    assert len(pre) == len(pre_amounts) and len(post) == len(post_amounts)
    return FoldOverlay(ko=tuple(slot(s, x) for s, x in zip(pre, pre_amounts))
                       + tuple(slot(s, x) for s, x in zip(post, post_amounts)),
                       maturity_no_ki=slot(st.no_ki, 0.25), maturity_ki=slot(st.ki, -0.5), constant_pv=0.0)


def _kor_fold_engines():
    from quantark.asset.equity.engine.pde.ko_reset_vol_pde_solvers import HestonKOResetSnowballPDESolver

    return {"pde": kor_pde, "heston_pde": lambda: HestonKOResetSnowballPDESolver(HESTON, n_x=96, n_v=16, n_t=64),
            "v1": lambda: KOResetSnowballQuadEngine(params=QuadParams(grid_points=1001))}


@pytest.mark.parametrize("engine", ["pde", "heston_pde", "v1"])
@pytest.mark.parametrize("kind", ["schedule1", "schedule2"])
def test_ko_reset_folding_onto_a_same_instant_pair_is_refused(engine, kind):
    """D10 on the KO-reset: two effective knock-out records of one schedule on one instant each pay their own cash; the
    binder is one to one by (schedule, time) -- refused, naming them (before T5a the PDE raised the binder's generic
    "two knock-out records share ('pre', 0.5)")."""
    make = _kor_fold_engines()[engine]
    if kind == "schedule1":
        product, amounts, named = (ko_reset(**KOR_PDE_CASES["schedule1_step_down"]), ([1, 2, 3, 4], [5, 6, 7]),
                                   "2025-07-03, 2025-07-04")
    else:
        product, amounts, named = (ko_reset(**KOR_PDE_CASES["schedule2_step_down"]), ([1, 2, 3], [4, 5, 6, 7]),
                                   "2025-11-26, 2025-11-27")
    e = env()
    with pytest.raises(ValidationError, match=f"cash-leg folding cannot bind the knock-out records {named}"):
        make().price_folded(product, e, _kor_overlay(product, e, *amounts))


@pytest.mark.parametrize("engine", ["pde", "heston_pde", "v1"])
@pytest.mark.parametrize("kind", ["schedule1", "schedule2"])
def test_ko_reset_folding_onto_a_duplicate_binds_the_effective_record(engine, kind):
    """A same-level duplicate (another payoff) binds its effective record: the folded price is the deduplicated
    product's bitwise, and the cash folded onto the dominated record is never paid (before T5a the PDE refused the
    duplicate too)."""
    make = _kor_fold_engines()[engine]
    dup, base, row = _ko_reset_duplicate(kind, 0.40)
    e = env()
    pre_n = 4 if kind == "schedule1" else 3
    base_pre, base_post = [1.0, 2.0, 3.0], [5.0, 6.0, 7.0]

    def amounts(dominated):
        cash = base_pre + base_post
        cash.insert(row, dominated)
        return cash[:pre_n], cash[pre_n:]

    folded = make().price_folded(dup, e, _kor_overlay(dup, e, *amounts(9.0)))
    assert folded == make().price_folded(dup, e, _kor_overlay(dup, e, *amounts(40.0)))
    assert folded == make().price_folded(base, e, _kor_overlay(base, e, base_pre, base_post))
    assert folded != make().price(base, e)
