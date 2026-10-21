"""Cash-leg folding on QUAD v1 (ship spec 2026-10-21, task T4).

The overlay reaches the v1 lattice where the product's own cash does: each knock-out record's
payoff (before its settlement-delay factor), each maturity branch's terminal payoff (before the
terminal delay factor), and the valuation-instant short-circuits. One solve prices product and
legs together; the product twins (AC14) pin that the folded cash is weighted exactly as the
product's own cash.

v1's event route is not one discretization with the v1 price everywhere, and the tests say where:

* Snowball knock-out probabilities are read off the price lattice's own event weights, so
  knock-out cash folds to round-off against the route (AC4 / AC5 at the spec bar).
* The Snowball / Phoenix knock-in probability is a separate hard-mask estimator, first order in
  the spacing (the price recursion's own knock-in weight barely moves with the grid). Cash that
  splits the two maturity branches is checked against the route with its knock-in split replaced
  by the price recursion's own weight (round-off), and the raw gap is shown to halve per doubling.
* The Phoenix route runs the Snowball event recursion on the Snowball lattice alignment, not the
  Phoenix price's joint coupon/knock-out projection on its own alignment: the knock-out gap is
  third order in the spacing (shown) and the Phoenix parity bar is set by it.
* KO-reset v1 reports one maturity outcome, so KO-reset parity is checked against the PDE 1D event
  route at the engine-pair tolerance (spec AC6), with v1 at the 2001+ node grids those tests use.
"""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import PDEParams, QuadParams
from quantark.cashleg import EventType
from quantark.cashleg.autocallable_leg import AutocallableCashLeg
from quantark.cashleg.fixed_payoff_leg import FixedPayoffLeg, PaymentTrigger
from quantark.cashleg.folding import FoldedEventCash, FoldOverlay, _event_structure, fold_cash_legs
from quantark.execution.errors import CapabilityError
from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.exceptions import ValidationError

from intraday.conftest import dated_ko_reset, dated_phoenix, dated_snowball, flat_env
from pde_3a.products import KO_RESET, env as ko_reset_env
from test_cashleg._autocallable_helpers import make_env, make_phoenix, make_snowball
from test_cashleg.test_folding import UNIT, _leg_set

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
T0, LIVE = datetime(2026, 3, 16), datetime(2026, 4, 1)
CONVERGING = QuadParams(grid_points=501, auto_converge=True, convergence_rel_tol=1e-3)
#: KO-reset v1 against the PDE route: the 2001+ node grids of the existing engine-pair tests
KO_RESET_V1 = QuadParams(grid_points=4001)
#: the engine-pair tolerance of those tests (5e-3 on a 100 notional), relative to trade notional
KO_RESET_PAIR = 5e-5
#: the Phoenix route's lattice gap at 1001 nodes (third order in the spacing): on P_KO up to 1e-5
#: near the knock-in level (3e-7 at the money); on 1e6 of legs 2.8e-8 of notional at the money and
#: 2.3e-6 knocked in
PHOENIX_P_GAP = 2e-5
PHOENIX_LATTICE = 1e-7
PHOENIX_KNOCKED_IN = 5e-6


def _snowball(spot=100.0):
    return SnowballQuadEngine, make_snowball(ko_dates=(0.25, 0.5, 0.75, 1.0)), make_env(spot=spot)


def _converging_snowball(spot=100.0):
    return (lambda: SnowballQuadEngine(CONVERGING)), make_snowball(ko_dates=(0.25, 0.5, 0.75, 1.0)), \
        make_env(spot=spot)


def _phoenix(spot=100.0):
    product = make_phoenix(ko_dates=(0.25, 0.5, 0.75, 1.0), coupon_barrier=(80.0,) * 4, ki_barrier=70.0)
    return PhoenixQuadEngine, product, make_env(spot=spot)


def _dated_snowball(spot=100.0):
    return SnowballQuadEngine, dated_snowball(CAL, T0), flat_env(LIVE, spot=spot)


def _dated_phoenix(spot=100.0):
    return PhoenixQuadEngine, dated_phoenix(CAL, T0), flat_env(LIVE, spot=spot)


def _ko_reset(spot=100.0):
    return KOResetSnowballQuadEngine, make_ko_reset(KO_RESET), ko_reset_env(spot=spot)


def _dated_ko_reset(spot=100.0):
    return KOResetSnowballQuadEngine, dated_ko_reset(CAL, T0), flat_env(LIVE, spot=spot)


SNOWBALLS = {"snowball": _snowball, "dated_snowball": _dated_snowball}
PHOENIXES = {"phoenix": _phoenix, "dated_phoenix": _dated_phoenix}
AUTOCALLABLES = {**SNOWBALLS, **PHOENIXES}
KO_RESETS = {"ko_reset": _ko_reset, "dated_ko_reset": _dated_ko_reset}
FAMILIES = {**AUTOCALLABLES, **KO_RESETS, "converging_snowball": _converging_snowball}
# ATM, just under the knock-out level, just above the knock-in level
SPOTS = {"snowball": (100.0, 102.5, 76.0), "phoenix": (100.0, 104.5, 71.0),
         "dated_snowball": (100.0, 102.5, 76.0), "dated_phoenix": (100.0, 102.5, 76.0),
         "ko_reset": (100.0, 102.5, 81.0), "dated_ko_reset": (100.0, 102.5, 76.0)}
#: the route's hard-mask knock-in estimator gap at 1001 nodes: continuous vs discrete knock-in
KI_GAP = {"snowball": 2e-3, "phoenix": 2e-3, "dated_snowball": 1.5e-2, "dated_phoenix": 1.5e-2}


def _hand_overlay(product, env, *, ko=0.0, no_ki=0.0, ki=0.0):
    """``x/DF(pay)`` per unit on every slot of a kind: the folded PV rises by ``x·P``."""
    structure = _event_structure(product, env)

    def cash(slot, x):
        return FoldedEventCash(slot.schedule, slot.determination_time, slot.payment_time,
                               x / env.get_discount_factor(slot.payment_time))

    return FoldOverlay(
        ko=tuple(cash(s, ko) for s in structure.ko),
        maturity_no_ki=cash(structure.no_ki, no_ki),
        maturity_ki=None if structure.ki is None else cash(structure.ki, ki),
        constant_pv=0.0,
    )


def _branch_overlay(product, env, *, no_ki=0.0, ki=0.0):
    """Per-unit cash on the maturity branches only, paid at their own settlement."""
    structure = _event_structure(product, env)
    zero = tuple(FoldedEventCash(s.schedule, s.determination_time, s.payment_time, 0.0) for s in structure.ko)
    slot = lambda s, x: FoldedEventCash(s.schedule, s.determination_time, s.payment_time, x)  # noqa: E731
    return FoldOverlay(ko=zero, maturity_no_ki=slot(structure.no_ki, no_ki),
                       maturity_ki=slot(structure.ki, ki), constant_pv=0.0)


def _route(engine, product, env):
    return engine.price_with_events(product, env, emit_distribution=True)


def _p(dist, et):
    return float(np.sum(dist.probabilities.get(et, 0.0)))


def _splits_maturity_branches(leg):
    """Legs paying differently on the two maturity buckets (they fold cash onto the knock-in branch)."""
    if isinstance(leg, FixedPayoffLeg):
        return leg.trigger in (PaymentTrigger.AT_MATURITY_NO_KO, PaymentTrigger.AT_MATURITY_WITH_KI)
    if isinstance(leg, AutocallableCashLeg):
        return leg.terminal_events != frozenset({EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI})
    return False


def _legs(dist, env, *, splitting):
    """The shared leg set (every allowlisted type); ``splitting=False`` drops branch-splitting legs."""
    legs = []
    for leg in _leg_set(dist.event_times):
        if not splitting and _splits_maturity_branches(leg):
            continue
        try:
            leg.value(dist, env, 100.0)
        except Exception:          # legs the route itself refuses are out of scope
            continue
        legs.append(leg)
    return legs


def _raised_knock_in_redemption(product, c):
    twin = deepcopy(product)
    own = twin.get_maturity_payoff_v1
    twin.get_maturity_payoff_v1 = lambda s, pricing_env=None, _own=own: _own(s, pricing_env=pricing_env) + c
    return twin


def _ki_weight(engine_factory, product, env):
    """The price recursion's own knock-in weight: the knocked-in redemption raised by one."""
    pay_df = env.get_discount_factor(_event_structure(product, env).ki.payment_time)
    twin = _raised_knock_in_redemption(product, 1.0)
    return (engine_factory().price(twin, env) - engine_factory().price(product, env)) / pay_df


def _ki_corrected(dist, weight):
    """The route's distribution with its knock-in split replaced by the price recursion's weight."""
    maturity = _p(dist, EventType.MATURITY_NO_KO) + _p(dist, EventType.MATURITY_WITH_KI)
    return replace(dist, probabilities={**dist.probabilities, EventType.MATURITY_WITH_KI: weight,
                                        EventType.MATURITY_NO_KO: maturity - weight})


# --- contract ----------------------------------------------------------------------------------

def test_v1_engines_declare_folding():
    for engine in (SnowballQuadEngine, PhoenixQuadEngine, KOResetSnowballQuadEngine):
        assert engine.supports_cash_leg_folding is True


@pytest.mark.parametrize("family", list(FAMILIES))
def test_an_empty_overlay_is_the_plain_price_bitwise(family):
    engine_cls, product, env = FAMILIES[family]()
    overlay = _hand_overlay(product, env)
    assert overlay.is_empty
    assert engine_cls().price_folded(product, env, overlay) == engine_cls().price(product, env)


def test_a_lifecycle_state_is_refused_as_price_refuses_it():
    from quantark.asset.equity.lifecycle import AutocallableLifecycleState

    engine_cls, product, env = _snowball()
    state = AutocallableLifecycleState()
    with pytest.raises(CapabilityError, match="lifecycle_state"):
        engine_cls().price(product, env, lifecycle_state=state)
    with pytest.raises(CapabilityError, match="lifecycle_state"):
        engine_cls().price_folded(product, env, _hand_overlay(product, env, ko=1.0), lifecycle_state=state)


def test_auto_converge_folds_on_the_grid_it_adopts():
    """The overlay travels into every candidate engine the refinement builds."""
    _, product, env = _converging_snowball()
    overlay = _hand_overlay(product, env, ko=2.0, ki=1.0, no_ki=0.5)
    probe = SnowballQuadEngine(CONVERGING)
    folded = probe._folded_solve(product, env, overlay)          # keeps the folded solve's adoption
    adopted = probe._last_convergence_info["grid_points"]
    assert probe._last_convergence_info["converged"] and adopted > 501
    fixed = SnowballQuadEngine(replace(CONVERGING, auto_converge=False, grid_points=adopted))
    assert folded == fixed.price_folded(product, env, overlay)
    assert SnowballQuadEngine(CONVERGING).price_folded(product, env, overlay) == folded


# --- AC4: identity on the engine's own event weights -------------------------------------------

@pytest.mark.parametrize("family", list(SNOWBALLS) + list(KO_RESETS))
@pytest.mark.parametrize("spot_index", [0, 1, 2])
def test_knock_out_cash_adds_c_times_the_routes_knock_out_probability(family, spot_index):
    engine_cls, product, env = FAMILIES[family](SPOTS[family][spot_index])
    c = 2.5
    engine = engine_cls()
    dist = _route(engine, product, env).event_distribution
    npv = engine.price(product, env)
    folded = engine.price_folded(product, env, _hand_overlay(product, env, ko=c))
    assert folded - npv == pytest.approx(c * _p(dist, EventType.KO), rel=1e-10, abs=1e-10 * abs(npv))


@pytest.mark.parametrize("family", list(PHOENIXES))
@pytest.mark.parametrize("spot_index", [0, 1, 2])
def test_phoenix_knock_out_cash_matches_its_route_up_to_the_lattice_gap(family, spot_index):
    engine_cls, product, env = FAMILIES[family](SPOTS[family][spot_index])
    c = 2.5
    dist = _route(engine_cls(), product, env).event_distribution
    folded = engine_cls().price_folded(product, env, _hand_overlay(product, env, ko=c))
    assert abs(folded - engine_cls().price(product, env) - c * _p(dist, EventType.KO)) <= PHOENIX_P_GAP * c


@pytest.mark.parametrize("family, spot", [("phoenix", 100.0), ("dated_phoenix", 100.0), ("dated_phoenix", 76.0)])
def test_the_phoenix_route_gap_is_a_third_order_lattice_difference(family, spot):
    _, product, env = FAMILIES[family](spot)
    gaps = []
    for n in (1001, 2001, 4001):
        engine = lambda n=n: PhoenixQuadEngine(QuadParams(grid_points=n))
        dist = _route(engine(), product, env).event_distribution
        weight = engine().price_folded(product, env, _hand_overlay(product, env, ko=1.0)) - engine().price(product, env)
        gaps.append(weight - _p(dist, EventType.KO))
    assert gaps[0] / gaps[1] > 5.0 and gaps[1] / gaps[2] > 5.0


@pytest.mark.parametrize("family", list(AUTOCALLABLES))
@pytest.mark.parametrize("spot_index", [0, 1, 2])
def test_knock_in_branch_weight_is_the_routes_up_to_its_estimator_gap(family, spot_index):
    """Knock-in cash adds c × the price recursion's own knock-in weight, additively with knock-out cash."""
    engine_cls, product, env = FAMILIES[family](SPOTS[family][spot_index])
    c = 2.5
    weight = _ki_weight(engine_cls, product, env)
    ko_only = engine_cls().price_folded(product, env, _hand_overlay(product, env, ko=c))
    both = engine_cls().price_folded(product, env, _hand_overlay(product, env, ko=c, ki=c))
    assert both - ko_only == pytest.approx(c * weight, rel=1e-10, abs=1e-12)
    dist = _route(engine_cls(), product, env).event_distribution
    assert abs(weight - _p(dist, EventType.MATURITY_WITH_KI)) < KI_GAP[family]


@pytest.mark.parametrize("family", ["snowball", "dated_snowball"])
def test_the_route_knock_in_gap_is_its_own_first_order_error(family):
    """Doubling the lattice halves the gap while the price's own weight barely moves: it is the route's
    hard-mask estimator, not the fold."""
    engine_cls, product, env = FAMILIES[family]()
    gaps, weights = [], []
    for n in (1001, 2001):
        factory = lambda n=n: engine_cls(QuadParams(grid_points=n))
        dist = _route(factory(), product, env).event_distribution
        weights.append(_ki_weight(factory, product, env))
        gaps.append(weights[-1] - _p(dist, EventType.MATURITY_WITH_KI))
    assert 0.35 < gaps[1] / gaps[0] < 0.65
    assert abs(weights[1] - weights[0]) < 0.1 * abs(gaps[1])


# --- AC5: parity with v1's own event route (Snowball / Phoenix) ---------------------------------

def _parity_tolerance(family):
    return 1e-9 if family in SNOWBALLS else PHOENIX_LATTICE


@pytest.mark.parametrize("quantity", [1.0, -3.0])
@pytest.mark.parametrize("family", list(AUTOCALLABLES))
def test_folded_trade_matches_the_v1_event_route(family, quantity):
    """Every allowlisted leg type, legs that do not split the maturity branches."""
    engine_cls, product, env = FAMILIES[family]()
    route = _route(engine_cls(), product, env)
    dist = route.event_distribution
    legs = _legs(dist, env, splitting=False)
    assert {type(leg).__name__ for leg in legs} >= {
        "DeterministicLeg", "AccrualLeg", "FixedPayoffLeg", "AutocallableCashLeg"}
    overlay = fold_cash_legs(legs, product, env, quantity=quantity, unit_notional=100.0)
    assert overlay.maturity_ki.amount == 0.0
    folded = quantity * engine_cls().price_folded(product, env, overlay) + overlay.constant_pv
    expected = quantity * route.npv + sum(leg.value(dist, env, 100.0) for leg in legs)
    notional = max(100.0 * abs(quantity), UNIT)
    assert abs(folded - expected) <= _parity_tolerance(family) * notional


@pytest.mark.parametrize("family", list(AUTOCALLABLES))
def test_branch_splitting_legs_match_the_knock_in_corrected_route(family):
    """With the route's knock-in split replaced by the price recursion's own weight, every leg agrees;
    against the raw route the trade differs only by that estimator gap on the knock-in branch cash."""
    engine_cls, product, env = FAMILIES[family]()
    route = _route(engine_cls(), product, env)
    dist = route.event_distribution
    legs = _legs(dist, env, splitting=True)
    assert any(_splits_maturity_branches(leg) for leg in legs)
    weight = _ki_weight(engine_cls, product, env)
    overlay = fold_cash_legs(legs, product, env, quantity=1.0, unit_notional=100.0)
    folded = engine_cls().price_folded(product, env, overlay) + overlay.constant_pv
    exact = route.npv + sum(leg.value(_ki_corrected(dist, weight), env, 100.0) for leg in legs)
    assert abs(folded - exact) <= _parity_tolerance(family) * UNIT
    raw = route.npv + sum(leg.value(dist, env, 100.0) for leg in legs)
    ki_cash = overlay.maturity_ki.amount * env.get_discount_factor(overlay.maturity_ki.payment_time)
    gap = weight - _p(dist, EventType.MATURITY_WITH_KI)
    assert abs((folded - raw) - ki_cash * gap) <= _parity_tolerance(family) * UNIT


# --- AC6: KO-reset against the PDE 1D event route -----------------------------------------------

def _ko_reset_v1():
    return KOResetSnowballQuadEngine(KO_RESET_V1)


@pytest.mark.parametrize("quantity", [1.0, -3.0])
@pytest.mark.parametrize("family", list(KO_RESETS))
def test_folded_ko_reset_matches_the_pde_event_route(family, quantity):
    _, product, env = FAMILIES[family]()
    route = _route(KOResetSnowballPDESolver(PDEParams()), product, env)
    dist = route.event_distribution
    legs = _legs(dist, env, splitting=True)
    assert len(legs) >= 9
    overlay = fold_cash_legs(legs, product, env, quantity=quantity, unit_notional=100.0)
    folded = quantity * _ko_reset_v1().price_folded(product, env, overlay) + overlay.constant_pv
    expected = quantity * route.npv + sum(leg.value(dist, env, 100.0) for leg in legs)
    notional = max(100.0 * abs(quantity), UNIT)
    assert abs(folded - expected) <= KO_RESET_PAIR * notional


@pytest.mark.parametrize("family", list(KO_RESETS))
def test_ko_reset_no_ki_cash_lands_at_the_pre_maturity(family):
    """The alive branch ends at the first schedule's maturity: its cash is neither dropped nor deferred."""
    _, product, env = FAMILIES[family]()
    pde = _route(KOResetSnowballPDESolver(PDEParams()), product, env).event_distribution
    assert _p(pde, EventType.MATURITY_NO_KO) > 0.01
    c = 3.0
    folded = _ko_reset_v1().price_folded(product, env, _hand_overlay(product, env, no_ki=c))
    npv = _ko_reset_v1().price(product, env)
    assert folded - npv == pytest.approx(c * _p(pde, EventType.MATURITY_NO_KO), abs=KO_RESET_PAIR * 100.0 * c)


# --- AC8: lifecycle -----------------------------------------------------------------------------

def _knocked_in(product):
    product = deepcopy(product)
    product._otc_lifecycle_knocked_in = True
    return product


def _ko_dates(product):
    return [r.observation_date for r in product.barrier_config.ko_observation_schedule.records]


def _lifecycle_cases():
    sb, ph = dated_snowball(CAL, T0), dated_phoenix(CAL, T0)
    return {
        "snowball-knocked-in": (SnowballQuadEngine, _knocked_in(sb), flat_env(LIVE, spot=90.0)),
        "phoenix-knocked-in": (PhoenixQuadEngine, _knocked_in(ph), flat_env(LIVE, spot=90.0)),
        "snowball-ko-decided-at-valuation": (SnowballQuadEngine, sb, flat_env(_ko_dates(sb)[2], spot=110.0)),
        "phoenix-ko-decided-at-valuation": (PhoenixQuadEngine, ph, flat_env(_ko_dates(ph)[2], spot=110.0)),
    }


@pytest.mark.parametrize("case", list(_lifecycle_cases()))
def test_lifecycle_states_fold_like_the_event_route(case):
    """The legs' part of the folded value equals the legs on the route's distribution."""
    engine_cls, product, env = _lifecycle_cases()[case]
    route = _route(engine_cls(), product, env)
    dist = route.event_distribution
    legs = _legs(dist, env, splitting=True)
    overlay = fold_cash_legs(legs, product, env, quantity=1.0, unit_notional=100.0)
    npv = engine_cls().price(product, env)
    folded = engine_cls().price_folded(product, env, overlay) + overlay.constant_pv
    expected = npv + sum(leg.value(dist, env, 100.0) for leg in legs)
    tolerance = PHOENIX_KNOCKED_IN if case == "phoenix-knocked-in" else 1e-9
    assert abs(folded - expected) <= tolerance * UNIT
    if "decided" in case:
        assert _p(dist, EventType.KO) == pytest.approx(1.0, abs=1e-12)
    assert route.npv == npv


def _ko_reset_lifecycle_cases():
    kr = dated_ko_reset(CAL, T0)
    return {
        "ko-reset-knocked-in": (_knocked_in(kr), flat_env(LIVE, spot=90.0)),
        "ko-reset-ko-decided-at-valuation": (kr, flat_env(_ko_dates(kr)[2], spot=110.0)),
    }


@pytest.mark.parametrize("case", list(_ko_reset_lifecycle_cases()))
def test_ko_reset_lifecycle_states_fold_like_the_pde_event_route(case):
    product, env = _ko_reset_lifecycle_cases()[case]
    route = _route(KOResetSnowballPDESolver(PDEParams()), product, env)
    dist = route.event_distribution
    legs = _legs(dist, env, splitting=True)
    overlay = fold_cash_legs(legs, product, env, quantity=1.0, unit_notional=100.0)
    folded = _ko_reset_v1().price_folded(product, env, overlay) + overlay.constant_pv
    expected = route.npv + sum(leg.value(dist, env, 100.0) for leg in legs)
    assert abs(folded - expected) <= KO_RESET_PAIR * UNIT
    if "decided" in case:
        # the short-circuit pays the decided record's cash at its settlement, exactly
        structure = _event_structure(product, env)
        hit = next(i for i, s in enumerate(structure.ko) if s.determination_time == 0.0 and s.schedule == "pre")
        entry = overlay.ko[hit]
        assert folded - overlay.constant_pv - _ko_reset_v1().price(product, env) == pytest.approx(
            entry.amount * env.get_discount_factor(entry.payment_time), rel=1e-12)


@pytest.mark.parametrize("engine_cls, build", [(SnowballQuadEngine, dated_snowball),
                                               (PhoenixQuadEngine, dated_phoenix)])
def test_an_expired_product_refuses_a_non_empty_overlay(engine_cls, build):
    """Expired, the structure is trivial and leg overlays are empty; a hand-built one has no event to land on."""
    product = build(CAL, T0)
    env = flat_env(_ko_dates(product)[-1], spot=95.0)
    assert product.get_maturity(env) <= 0.0
    empty = _hand_overlay(product, env)
    assert empty.is_empty and engine_cls().price_folded(product, env, empty) == engine_cls().price(product, env)
    with pytest.raises(CapabilityError, match="expired"):
        engine_cls().price_folded(product, env, _hand_overlay(product, env, no_ki=1.0))


# --- fail closed --------------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["snowball", "phoenix"])
def test_an_overlay_built_for_another_schedule_is_refused(family):
    engine_cls, product, env = FAMILIES[family]()
    other = make_snowball(ko_dates=(0.5, 1.0))
    with pytest.raises(ValidationError):
        engine_cls().price_folded(product, env, _hand_overlay(other, env, ko=1.0))


def test_a_ko_reset_maturity_entry_at_the_wrong_branch_time_is_refused():
    engine_cls, product, env = _ko_reset()
    overlay = _hand_overlay(product, env, no_ki=1.0)
    moved = replace(overlay, maturity_no_ki=replace(overlay.maturity_no_ki,
                                                    determination_time=overlay.maturity_ki.determination_time,
                                                    payment_time=overlay.maturity_ki.payment_time))
    with pytest.raises(ValidationError, match="no_ki"):
        engine_cls().price_folded(product, env, moved)


def test_a_ko_reset_overlay_with_its_schedules_swapped_is_refused():
    engine_cls, product, env = _ko_reset()
    overlay = _hand_overlay(product, env, ko=1.0)
    swapped = replace(overlay, ko=tuple(replace(e, schedule="post" if e.schedule == "pre" else "pre")
                                        for e in overlay.ko))
    with pytest.raises(ValidationError):
        engine_cls().price_folded(product, env, swapped)


# --- AC9: no folded state survives a folded solve ------------------------------------------------

@pytest.mark.parametrize("family", ["snowball", "converging_snowball", "phoenix", "ko_reset"])
def test_a_shared_engine_never_mixes_overlays(family):
    engine_cls, product, env = FAMILIES[family]()
    a = _hand_overlay(product, env, ko=1.0, ki=2.0)
    b = _hand_overlay(product, env, ko=-0.5, no_ki=0.7)
    shared = engine_cls()
    seq = [shared.price_folded(product, env, a), shared.price(product, env),
           shared.price_folded(product, env, b), shared.price_folded(product, env, a)]
    fresh = [engine_cls().price_folded(product, env, a), engine_cls().price(product, env),
             engine_cls().price_folded(product, env, b), engine_cls().price_folded(product, env, a)]
    assert seq == fresh
    assert len({seq[0], seq[1], seq[2]}) == 3


@pytest.mark.parametrize("family", ["snowball", "converging_snowball", "phoenix", "ko_reset"])
def test_a_folded_solve_leaves_the_plain_solve_artefacts_untouched(family):
    engine_cls, product, env = FAMILIES[family]()
    engine = engine_cls()
    engine.price(product, env)
    grid_spots, grid_prices = (np.array(a, copy=True) for a in engine._last_spot_greeks_grid)
    info = getattr(engine, "_last_convergence_info", None)
    engine.price_folded(product, env, _hand_overlay(product, env, ko=5.0, ki=5.0, no_ki=5.0))
    np.testing.assert_array_equal(engine._last_spot_greeks_grid[0], grid_spots)
    np.testing.assert_array_equal(engine._last_spot_greeks_grid[1], grid_prices)
    assert getattr(engine, "_last_convergence_info", None) == info
    ladder = engine.calculate_spot_greeks_curve(product, env, [95.0, 100.0])
    assert ladder == engine_cls().calculate_spot_greeks_curve(product, env, [95.0, 100.0])


def test_recorded_backward_grids_are_the_plain_solves():
    engine_cls, product, env = _snowball()
    engine = engine_cls()
    engine.record_backward_grids = True
    engine.price(product, env)
    before = {t: tuple(np.array(a, copy=True) for a in v) for t, v in engine._backward_grids.items()}
    engine.price_folded(product, env, _hand_overlay(product, env, ko=5.0, ki=5.0))
    assert sorted(engine._backward_grids) == sorted(before)
    for t, arrays in before.items():
        for got, want in zip(engine._backward_grids[t], arrays):
            np.testing.assert_array_equal(got, want)


# --- AC14: product twins ------------------------------------------------------------------------

def _ko_cash_diff_overlay(product, bumped, env, ko_amounts):
    """Overlay whose KO amounts are the bumped product's record cash minus the base product's."""
    structure = _event_structure(product, env)
    base, moved = ko_amounts(product, env), ko_amounts(bumped, env)
    assert len(base) == len(structure.ko) and np.any(np.asarray(moved) != np.asarray(base))
    return FoldOverlay(
        ko=tuple(FoldedEventCash(s.schedule, s.determination_time, s.payment_time, m - b)
                 for s, b, m in zip(structure.ko, base, moved)),
        maturity_no_ki=FoldedEventCash(structure.no_ki.schedule, structure.no_ki.determination_time,
                                       structure.no_ki.payment_time, 0.0),
        maturity_ki=None,
        constant_pv=0.0,
    )


def _autocallable_ko_cash(product, env):
    return [float(r.payoff or 0.0) for r in product.resolve_ko_observations(env)
            if 0.0 <= r.observation_time <= product.get_maturity(env)]


def _ko_reset_ko_cash(product, env):
    """The v1 engine's own record cash, in the structure's merged order (pre first at a tie)."""
    engine = KOResetSnowballQuadEngine()
    tau = product.get_maturity(env)
    tagged = []
    for tag, config in ((0, product.barrier_config), (1, product.post_barrier_config)):
        tagged += [(r.observation_time, tag, float(r.payoff))
                   for r in engine._resolve_ko_records(product, env, config)
                   if 0.0 <= r.observation_time <= tau]
    return [cash for _, _, cash in sorted(tagged, key=lambda r: (r[0], r[1]))]


def _bumped_ko_rate(family, product):
    """The same family with a higher KO coupon, rebuilt by its own builder (records carry the cash)."""
    if family == "snowball":
        return make_snowball(ko_dates=(0.25, 0.5, 0.75, 1.0), ko_rate=0.17)
    if family == "dated_snowball":
        return dated_snowball(CAL, T0, ko_rate=0.14, rebate_rate=0.12)   # the rebate stays at the base 0.12
    if family == "dated_phoenix":
        bumped = deepcopy(product)
        bumped.barrier_config = replace(bumped.barrier_config, ko_rate=0.05)
        return bumped
    if family == "ko_reset":
        return make_ko_reset(dict(KO_RESET, pre_ko_rate=KO_RESET["pre_ko_rate"] + 0.02,
                                  post_ko_rate=KO_RESET["post_ko_rate"] + 0.02))
    return dated_ko_reset(CAL, T0, post_rate=0.05, rebate_rate=0.15)   # base: post 0.03, rebate = pre 0.15


@pytest.mark.parametrize("family, amounts", [
    ("snowball", _autocallable_ko_cash),
    ("dated_snowball", _autocallable_ko_cash),
    ("dated_phoenix", _autocallable_ko_cash),
    ("ko_reset", _ko_reset_ko_cash),
    ("dated_ko_reset", _ko_reset_ko_cash),
])
def test_ko_cash_twin_equals_the_product_with_a_higher_ko_rate(family, amounts):
    engine_cls, product, env = FAMILIES[family]()
    bumped = _bumped_ko_rate(family, product)
    overlay = _ko_cash_diff_overlay(product, bumped, env, amounts)
    folded = engine_cls().price_folded(product, env, overlay)
    assert folded == pytest.approx(engine_cls().price(bumped, env), rel=1e-10)


@pytest.mark.parametrize("family", ["snowball", "ko_reset"])
def test_maturity_cash_twin_equals_the_product_with_a_higher_rebate(family):
    engine_cls, product, env = FAMILIES[family]()
    bumped = deepcopy(product)
    bumped.payoff_config = replace(bumped.payoff_config,
                                   rebate_rate=bumped.payoff_config.rebate_rate + 0.03)
    diffs = {bumped.get_maturity_payoff_v0(s, pricing_env=env) - product.get_maturity_payoff_v0(s, pricing_env=env)
             for s in (80.0, 100.0, 102.0)}
    assert len(diffs) == 1 and next(iter(diffs)) > 0
    assert bumped.get_maturity_payoff_v1(90.0, pricing_env=env) == product.get_maturity_payoff_v1(90.0, pricing_env=env)
    structure = _event_structure(product, env)
    # no settlement lag on these fixtures: the alive branch's own delay factor is one either way
    assert structure.no_ki.payment_time == structure.no_ki.determination_time
    folded = engine_cls().price_folded(product, env, _branch_overlay(product, env, no_ki=next(iter(diffs))))
    assert folded == pytest.approx(engine_cls().price(bumped, env), rel=1e-10)


@pytest.mark.parametrize("family", ["dated_snowball"])
def test_no_ki_cash_twin_equals_the_product_with_a_raised_alive_redemption(family):
    engine_cls, product, env = FAMILIES[family]()
    c = 3.0
    twin = deepcopy(product)
    own = twin.get_maturity_payoff_v0
    twin.get_maturity_payoff_v0 = lambda s, pricing_env=None, _own=own: _own(s, pricing_env=pricing_env) + c
    folded = engine_cls().price_folded(product, env, _branch_overlay(product, env, no_ki=c))
    assert folded == pytest.approx(engine_cls().price(twin, env), rel=1e-10)


@pytest.mark.parametrize("family", ["snowball", "phoenix", "dated_snowball", "dated_phoenix", "ko_reset",
                                    "dated_ko_reset"])
def test_knock_in_cash_twin_equals_the_product_with_a_raised_knocked_in_redemption(family):
    engine_cls, product, env = FAMILIES[family]()
    c = 4.0
    folded = engine_cls().price_folded(product, env, _branch_overlay(product, env, ki=c))
    assert folded == pytest.approx(engine_cls().price(_raised_knock_in_redemption(product, c), env), rel=1e-10)
