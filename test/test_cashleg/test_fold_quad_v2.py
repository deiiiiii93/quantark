"""Cash-leg folding on QUAD V2 (ship spec 2026-10-21, task T3).

The overlay reaches V2 through the compiled contract: knock-out cash is added to each KO
record's settlement cash and maturity cash to each branch's terminal payoff (incl. the
KO-reset pre-maturity branch), so one prepared context prices product and legs together.
"""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

from quantark.asset.equity.engine.quad.v2.engine import (
    KOResetSnowballQuadEngineV2,
    PhoenixQuadEngineV2,
    SnowballQuadEngineV2,
)
from quantark.cashleg import EventType
from quantark.cashleg.folding import (
    FoldedEventCash,
    FoldOverlay,
    _event_structure,
    fold_cash_legs,
)
from quantark.execution.errors import CapabilityError
from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.exceptions import ValidationError

from intraday.conftest import dated_ko_reset, dated_snowball, flat_env
from pde_3a.products import KO_RESET, env as ko_reset_env
from test_cashleg._autocallable_helpers import make_env, make_phoenix, make_snowball
from test_cashleg.test_folding import UNIT, _leg_set

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
T0, LIVE = datetime(2026, 3, 16), datetime(2026, 4, 1)


def _snowball(spot=100.0):
    return SnowballQuadEngineV2, make_snowball(ko_dates=(0.25, 0.5, 0.75, 1.0)), make_env(spot=spot)


def _phoenix(spot=100.0):
    product = make_phoenix(ko_dates=(0.25, 0.5, 0.75, 1.0), coupon_barrier=(80.0,) * 4, ki_barrier=70.0)
    return PhoenixQuadEngineV2, product, make_env(spot=spot)


def _ko_reset(spot=100.0):
    # European knock-in keeps V2 fast; the pre (1y) and post (2y) maturities stay distinct.
    return (KOResetSnowballQuadEngineV2, make_ko_reset(dict(KO_RESET, ki_monitoring="european")),
            ko_reset_env(spot=spot))


def _dated_ko_reset(spot=100.0):
    return KOResetSnowballQuadEngineV2, dated_ko_reset(CAL, T0), flat_env(LIVE, spot=spot)


FAMILIES = {"snowball": _snowball, "phoenix": _phoenix, "ko_reset": _ko_reset,
            "dated_ko_reset": _dated_ko_reset}
# ATM, just under the KO barrier, just above the KI barrier
SPOTS = {"snowball": (100.0, 102.5, 76.0), "phoenix": (100.0, 104.5, 71.0),
         "ko_reset": (100.0, 102.5, 81.0), "dated_ko_reset": (100.0, 102.5, 76.0)}


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


def _dist(engine, product, env, **kw):
    return engine.price_with_events(product, env, emit_distribution=True, **kw).event_distribution


def _p(dist, et):
    return float(np.sum(dist.probabilities.get(et, 0.0)))


# --- contract ----------------------------------------------------------------------------------

def test_v2_engines_declare_folding_and_the_old_extras_are_gone():
    import inspect

    from quantark.asset.equity.engine.quad.v2.adapters import compile_contract

    for engine in (SnowballQuadEngineV2, PhoenixQuadEngineV2, KOResetSnowballQuadEngineV2):
        assert engine.supports_cash_leg_folding is True
    for fn in (compile_contract, SnowballQuadEngineV2.prepare, SnowballQuadEngineV2._compile):
        params = inspect.signature(fn).parameters
        assert "extra_ko_cash" not in params and "extra_terminal_cash" not in params
        assert "fold" in params


@pytest.mark.parametrize("family", ["snowball", "ko_reset"])
def test_an_empty_overlay_is_the_plain_price_bitwise(family):
    engine_cls, product, env = FAMILIES[family]()
    overlay = _hand_overlay(product, env)
    assert overlay.is_empty
    assert engine_cls().price_folded(product, env, overlay) == engine_cls().price(product, env)


# --- AC4: identity on the engine's own event weights -------------------------------------------

@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("spot_index", [0, 1, 2])
def test_folded_cash_on_every_non_anchor_event_adds_c_times_its_probability(family, spot_index):
    engine_cls, product, env = FAMILIES[family](SPOTS[family][spot_index])
    c = 2.5
    engine = engine_cls()
    dist = _dist(engine, product, env)
    npv = engine.price(product, env)
    folded = engine.price_folded(product, env, _hand_overlay(product, env, ko=c, ki=c))
    expected = c * (_p(dist, EventType.KO) + _p(dist, EventType.MATURITY_WITH_KI))
    assert folded - npv == pytest.approx(expected, rel=1e-10, abs=1e-10 * abs(npv))


@pytest.mark.parametrize("family", ["snowball", "ko_reset", "dated_ko_reset"])
def test_no_ki_branch_cash_lands_on_its_own_end_incl_ko_reset_pre_maturity(family):
    """The alive branch ends at the KO-reset PRE-maturity; its cash must not be dropped."""
    engine_cls, product, env = FAMILIES[family]()
    c = 3.0
    engine = engine_cls()
    dist = _dist(engine, product, env)
    assert _p(dist, EventType.MATURITY_NO_KO) > 0.01
    folded = engine.price_folded(product, env, _hand_overlay(product, env, no_ki=c))
    npv = engine.price(product, env)
    assert folded - npv == pytest.approx(c * _p(dist, EventType.MATURITY_NO_KO), rel=1e-10)


# --- AC5: parity with V2's own event route ------------------------------------------------------

@pytest.mark.parametrize("quantity", [1.0, -3.0])
@pytest.mark.parametrize("family", list(FAMILIES))
def test_folded_trade_matches_the_v2_event_route(family, quantity):
    engine_cls, product, env = FAMILIES[family]()
    route = engine_cls().price_with_events(product, env, emit_distribution=True)
    dist = route.event_distribution
    legs = []
    for leg in _leg_set(dist.stream_times[EventType.KO]):
        try:
            leg.value(dist, env, 100.0)
        except Exception:          # legs the route itself refuses are out of scope
            continue
        legs.append(leg)
    assert len(legs) >= 9
    overlay = fold_cash_legs(legs, product, env, quantity=quantity, unit_notional=100.0)
    folded = quantity * engine_cls().price_folded(product, env, overlay) + overlay.constant_pv
    expected = quantity * route.npv + sum(leg.value(dist, env, 100.0) for leg in legs)
    notional = max(100.0 * abs(quantity), UNIT)
    assert abs(folded - expected) <= 1e-9 * notional


# --- AC8: lifecycle -----------------------------------------------------------------------------

def _knocked_in(product):
    product = deepcopy(product)
    product._otc_lifecycle_knocked_in = True
    return product


def _lifecycle_cases():
    sb = dated_snowball(CAL, T0)
    ko_dates = [r.observation_date for r in sb.barrier_config.ko_observation_schedule.records]
    return {
        "snowball-knocked-in": (SnowballQuadEngineV2, _knocked_in(sb), flat_env(LIVE, spot=90.0)),
        "ko-reset-knocked-in": (KOResetSnowballQuadEngineV2, _knocked_in(dated_ko_reset(CAL, T0)),
                                flat_env(LIVE, spot=90.0)),
        "snowball-ko-decided-at-valuation": (SnowballQuadEngineV2, sb, flat_env(ko_dates[2], spot=110.0)),
        "snowball-live": (SnowballQuadEngineV2, sb, flat_env(LIVE)),
    }


@pytest.mark.parametrize("case", list(_lifecycle_cases()))
def test_lifecycle_states_fold_like_the_event_route(case):
    engine_cls, product, env = _lifecycle_cases()[case]
    route = engine_cls().price_with_events(product, env, emit_distribution=True)
    dist = route.event_distribution
    structure = _event_structure(product, env)
    legs = []
    for leg in _leg_set([s.determination_time for s in structure.ko] or [1.0]):
        try:
            leg.value(dist, env, 100.0)
        except Exception:
            continue
        legs.append(leg)
    overlay = fold_cash_legs(legs, product, env, quantity=1.0, unit_notional=100.0)
    folded = engine_cls().price_folded(product, env, overlay) + overlay.constant_pv
    expected = route.npv + sum(leg.value(dist, env, 100.0) for leg in legs)
    assert abs(folded - expected) <= 1e-9 * UNIT
    if case == "snowball-ko-decided-at-valuation":
        assert _p(dist, EventType.KO) == pytest.approx(1.0, abs=1e-12)


def test_a_terminated_lifecycle_refuses_a_non_empty_overlay():
    from quantark.asset.equity.lifecycle import AutocallableLifecycleState
    from quantark.asset.equity.lifecycle.cashflows import ValuationPoint

    engine_cls, product, env = _snowball()
    state = AutocallableLifecycleState()
    state.mark_ko(env.valuation_date, cashflow=105.0,
                  settlement_date=env.valuation_date + timedelta(days=10),
                  valuation_point=ValuationPoint(date=env.valuation_date))
    with pytest.raises(CapabilityError, match="terminated"):
        engine_cls().price_folded(product, env, _hand_overlay(product, env, ko=1.0),
                                  lifecycle_state=state)
    empty = _hand_overlay(product, env)
    assert engine_cls().price_folded(product, env, empty, lifecycle_state=state) == \
        engine_cls().price(product, env, lifecycle_state=state)


def test_event_phase_after_refuses_a_non_empty_overlay():
    engine_cls, product, env = _snowball()
    with pytest.raises(CapabilityError, match="event_phase"):
        engine_cls().price_folded(product, env, _hand_overlay(product, env, ko=1.0), event_phase="after")


# --- fail closed --------------------------------------------------------------------------------

def test_an_overlay_built_for_another_schedule_is_refused():
    engine_cls, product, env = _snowball()
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


# --- AC9: the single-slot prepared cache ----------------------------------------------------------

@pytest.mark.parametrize("family", ["snowball", "phoenix", "ko_reset"])
def test_the_prepared_cache_never_mixes_overlays(family):
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
    from quantark.asset.equity.engine.quad.v2.adapters import _reset_records

    pre = [(r[0], 0, r[2]) for r in _reset_records(product, env, product.barrier_config)]
    post = [(r[0], 1, r[2]) for r in _reset_records(product, env, product.post_barrier_config)]
    return [r[2] for r in sorted(pre + post, key=lambda r: (r[0], r[1]))]


def _bumped_ko_rate(family):
    """The same family with a higher KO coupon, rebuilt by its own builder (records carry the cash)."""
    if family == "snowball":
        return make_snowball(ko_dates=(0.25, 0.5, 0.75, 1.0), ko_rate=0.17)
    if family == "ko_reset":
        return make_ko_reset(dict(KO_RESET, ki_monitoring="european",
                                  pre_ko_rate=KO_RESET["pre_ko_rate"] + 0.02,
                                  post_ko_rate=KO_RESET["post_ko_rate"] + 0.02))
    return dated_ko_reset(CAL, T0, post_rate=0.05, rebate_rate=0.15)   # base: post 0.03, rebate = pre 0.15


@pytest.mark.parametrize("family, amounts", [
    ("snowball", _autocallable_ko_cash),
    ("ko_reset", _ko_reset_ko_cash),
    ("dated_ko_reset", _ko_reset_ko_cash),
])
def test_ko_cash_twin_equals_the_product_with_a_higher_ko_rate(family, amounts):
    engine_cls, product, env = FAMILIES[family]()
    bumped = _bumped_ko_rate(family)
    overlay = _ko_cash_diff_overlay(product, bumped, env, amounts)
    folded = engine_cls().price_folded(product, env, overlay)
    assert folded == pytest.approx(engine_cls().price(bumped, env), rel=1e-10)


def test_maturity_cash_twin_equals_the_snowball_with_a_higher_rebate():
    engine_cls, product, env = _snowball()
    bumped = deepcopy(product)
    bumped.payoff_config = replace(bumped.payoff_config,
                                   rebate_rate=bumped.payoff_config.rebate_rate + 0.03)
    diffs = {bumped.get_maturity_payoff_v0(s, pricing_env=env) - product.get_maturity_payoff_v0(s, pricing_env=env)
             for s in (80.0, 100.0, 102.0)}
    assert len(diffs) == 1 and next(iter(diffs)) > 0
    assert bumped.get_maturity_payoff_v1(90.0, pricing_env=env) == product.get_maturity_payoff_v1(90.0, pricing_env=env)
    structure = _event_structure(product, env)
    overlay = FoldOverlay(
        ko=tuple(FoldedEventCash(s.schedule, s.determination_time, s.payment_time, 0.0) for s in structure.ko),
        maturity_no_ki=FoldedEventCash(structure.no_ki.schedule, structure.no_ki.determination_time,
                                       structure.no_ki.payment_time, next(iter(diffs))),
        maturity_ki=None,
        constant_pv=0.0,
    )
    folded = engine_cls().price_folded(product, env, overlay)
    assert folded == pytest.approx(engine_cls().price(bumped, env), rel=1e-10)
