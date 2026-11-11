"""The shared cash-leg fold core (ship spec 2026-10-21 D1–D6, AC1–AC3, AC12)."""
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.cashleg import (
    AccrualBasis,
    AccrualLeg,
    AutocallableCashLeg,
    AutocallableLegType,
    BaseAmount,
    BaseAmountMode,
    DeterministicLeg,
    EventType,
    FixedPayoffLeg,
    KOBehavior,
    LegDirection,
    LegSchedule,
    PaymentConvention,
    PaymentTrigger,
    PvFormula,
    SurvivalBasis,
)
from quantark.cashleg.folding import (
    FoldedEventCash,
    FoldOverlay,
    ensure_foldable,
    fold_cash_legs,
    is_event_independent,
    is_foldable,
)
from quantark.execution.errors import CapabilityError
from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.exceptions import ValidationError

from test_cashleg._autocallable_helpers import make_engine, make_env, make_phoenix, make_snowball
from intraday.conftest import dated_ko_reset, flat_env
from pde_3a.products import KO_RESET, env as ko_reset_env

UNIT = 1_000_000.0


def _accrual(behavior=KOBehavior.TRUNCATE_AT_KO, basis=SurvivalBasis.ENTER_PERIOD,
             convention=PaymentConvention.AT_PERIOD_END, pay_shift=0.0):
    starts = np.array([0.0, 0.25, 0.5, 0.75])
    ends = starts + 0.25
    return AccrualLeg(
        direction=LegDirection.BUYER_RECEIVES, rate=0.04,
        base=BaseAmount(value=UNIT, mode=BaseAmountMode.ABSOLUTE),
        schedule=LegSchedule(period_starts=starts, period_ends=ends,
                             payment_times=ends + pay_shift),
        ko_behavior=behavior, survival_basis=basis, payment_convention=convention,
    )


def _autocallable_leg(obs, formula=PvFormula.NOTIONAL_MINUS_PAYOFF,
                      terminal=frozenset({EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI})):
    obs = tuple(float(t) for t in obs)
    return AutocallableCashLeg(
        direction=LegDirection.BUYER_PAYS, leg_type=AutocallableLegType.MARGIN,
        notional=UNIT, rate=0.05, observation_schedule=obs,
        accrual_factors=tuple(np.linspace(0.3, 1.0, len(obs))),
        settlement_schedule=tuple(t + 0.01 for t in obs),
        terminal_accrual_factor=1.1, terminal_settlement_time=obs[-1] + 0.02,
        pv_formula=formula, terminal_events=terminal,
    )


def _leg_set(event_times):
    return [
        DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=5_000.0, payment_time=0.1),
        _accrual(),
        _accrual(basis=SurvivalBasis.COMPLETE_PERIOD, pay_shift=0.02),
        _accrual(convention=PaymentConvention.AT_MATURITY),
        _accrual(behavior=KOBehavior.PAY_FULL_SCHEDULE),
        FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=7_000.0,
                       trigger=PaymentTrigger.AT_KO, payment_offset_days=2),
        FixedPayoffLeg(direction=LegDirection.BUYER_PAYS, amount=3_000.0,
                       trigger=PaymentTrigger.AT_MATURITY_ANY, payment_offset_days=3),
        FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=2_000.0,
                       trigger=PaymentTrigger.AT_MATURITY_NO_KO),
        FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=4_000.0,
                       trigger=PaymentTrigger.AT_MATURITY_WITH_KI),
        _autocallable_leg(event_times),
        _autocallable_leg(event_times, formula=PvFormula.NORMAL,
                          terminal=frozenset({EventType.MATURITY_NO_KO})),
    ]


def _route_dist(engine, product, env):
    return engine.price_with_events(product, env, emit_distribution=True).event_distribution


def _reassemble(overlay, dist, env, quantity):
    """Σ_e P_e · amount_e · DF(pay_e) · q + constant — the folded leg PV on a route's own weights."""
    p_ko = np.asarray(dist.probabilities.get(EventType.KO, np.zeros(0)), dtype=float)
    assert len(overlay.ko) == p_ko.size
    total = 0.0
    for p, entry in zip(p_ko, overlay.ko):
        total += p * entry.amount * env.get_discount_factor(entry.payment_time)
    for et, entry in ((EventType.MATURITY_NO_KO, overlay.maturity_no_ki),
                      (EventType.MATURITY_WITH_KI, overlay.maturity_ki)):
        p = float(dist.probabilities.get(et, 0.0))
        if entry is not None:
            total += p * entry.amount * env.get_discount_factor(entry.payment_time)
    return total * quantity + overlay.constant_pv


def _structure_equal(dist, overlay):
    ko_pay = np.asarray(dist.payment_times[EventType.KO], dtype=float)
    assert [e.determination_time for e in overlay.ko] == list(dist.stream_times[EventType.KO])
    assert [e.payment_time for e in overlay.ko] == list(ko_pay)
    assert overlay.maturity_no_ki.payment_time == dist.payment_times_for(EventType.MATURITY_NO_KO)
    assert overlay.maturity_ki.payment_time == dist.payment_times_for(EventType.MATURITY_WITH_KI)


# --- AC1 / AC12: foldability -------------------------------------------------------------

def test_allowlisted_legs_on_ko_and_maturity_streams_fold():
    legs = _leg_set([0.5, 1.0])
    assert all(is_foldable(leg) for leg in legs)
    ensure_foldable(legs)
    assert is_event_independent(legs[0]) and is_event_independent(legs[4])
    assert not is_event_independent(legs[1])


@pytest.mark.parametrize("leg", [
    FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=1.0, trigger=PaymentTrigger.AT_KI),
    AutocallableCashLeg(direction=LegDirection.BUYER_RECEIVES, leg_type=AutocallableLegType.MARGIN,
                        notional=1.0, rate=0.1, observation_schedule=(0.5, 1.0),
                        accrual_factors=(0.5, 1.0), settlement_schedule=(0.5, 1.0),
                        accrual_basis=AccrualBasis.COUPON),
])
def test_unfoldable_streams_are_refused(leg):
    assert not is_foldable(leg)
    with pytest.raises(CapabilityError):
        ensure_foldable([leg])


def test_subclasses_are_refused_even_when_their_streams_fold():
    @dataclass(frozen=True)
    class ScaledDeterministic(DeterministicLeg):
        def value(self, event_dist, env, position_notional):
            return 2.0 * super().value(event_dist, env, position_notional)

    leg = ScaledDeterministic(direction=LegDirection.BUYER_PAYS, amount=1.0, payment_time=0.0)
    assert not is_foldable(leg)
    with pytest.raises(CapabilityError, match="ScaledDeterministic"):
        ensure_foldable([leg])


def test_zero_quantity_cannot_fold():
    product, env = make_snowball(), make_env()
    with pytest.raises(ValidationError, match="quantity"):
        fold_cash_legs([_accrual()], product, env, quantity=0.0, unit_notional=100.0)


def test_rebased_ko_reset_is_refused():
    from copy import deepcopy
    from quantark.util.enum import PostKOScheduleMode
    rebased = deepcopy(make_ko_reset(KO_RESET))
    rebased.post_ko_mode = PostKOScheduleMode.REBASED
    with pytest.raises(CapabilityError, match="REBASED"):
        fold_cash_legs([_accrual()], rebased, ko_reset_env(), quantity=1.0, unit_notional=100.0)


# --- AC2: structure and the anchored decomposition, against the PDE event route -------------

@pytest.mark.parametrize("quantity", [1.0, -3.0])
@pytest.mark.parametrize("family", ["snowball", "phoenix"])
def test_fold_reproduces_the_pde_route_leg_value(family, quantity):
    env = make_env()
    product = make_snowball(ko_dates=(0.25, 0.5, 0.75, 1.0)) if family == "snowball" else \
        make_phoenix(ko_dates=(0.25, 0.5, 0.75, 1.0), coupon_barrier=(80.0,) * 4, ki_barrier=70.0)
    dist = _route_dist(make_engine("pde", family), product, env)
    legs = _leg_set(dist.stream_times[EventType.KO])
    overlay = fold_cash_legs(legs, product, env, quantity=quantity, unit_notional=100.0)
    _structure_equal(dist, overlay)
    assert overlay.maturity_no_ki.amount == 0.0                    # the anchor carries nothing
    expected = sum(leg.value(dist, env, 100.0) for leg in legs)
    assert _reassemble(overlay, dist, env, quantity) == pytest.approx(expected, rel=1e-12)


def test_event_independent_legs_put_nothing_into_the_solve():
    env, product = make_env(), make_snowball()
    legs = [DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=10.0, payment_time=0.3),
            _accrual(behavior=KOBehavior.PAY_FULL_SCHEDULE)]
    overlay = fold_cash_legs(legs, product, env, quantity=2.0, unit_notional=100.0)
    assert overlay.is_empty
    dist = _route_dist(make_engine("pde"), product, env)
    assert overlay.constant_pv == pytest.approx(sum(l.value(dist, env, 100.0) for l in legs), rel=1e-14)


@pytest.mark.parametrize("dated", [False, True])
def test_ko_reset_fold_reproduces_the_pde_route(dated):
    if dated:
        cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
        product, env = dated_ko_reset(cal, datetime(2026, 3, 16)), flat_env(datetime(2026, 4, 1))
    else:
        product, env = make_ko_reset(KO_RESET), ko_reset_env()
    stats = KOResetSnowballPDESolver(PDEParams()).calculate_event_stats(product, env)
    from quantark.cashleg import EventDistribution
    dist = EventDistribution.from_autocallable_stats(stats)
    legs = [leg for leg in _leg_set(dist.stream_times[EventType.KO])
            if not isinstance(leg, AccrualLeg) or leg.payment_convention is PaymentConvention.AT_PERIOD_END]
    overlay = fold_cash_legs(legs, product, env, quantity=1.0, unit_notional=100.0)
    _structure_equal(dist, overlay)
    assert [e.schedule for e in overlay.ko].count("post") == stats.post_ko_times.size
    assert overlay.maturity_no_ki.determination_time == stats.pre_maturity_time
    assert overlay.maturity_ki.determination_time == stats.post_maturity_time
    expected = sum(leg.value(dist, env, 100.0) for leg in legs)
    # The KO-reset route marches BOTH maturity buckets, so they miss 1 by the PDE's discount defect: the
    # distribution absorbs that residual into MATURITY_NO_KO (spec D16) while the F5 survival keeps the
    # marched bucket; any affine fold then differs from it by (v_anchor - c)·residual per leg, never more.
    total_p = sum(float(np.sum(dist.probabilities[e])) for e in
                  (EventType.KO, EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI))
    assert abs(total_p - 1.0) < 1e-12
    assert 0.0 < abs(dist.probability_residual) < 1e-7
    bound = abs(dist.probability_residual) * 2.0 * UNIT * len(legs) + 1e-9 * UNIT * 1e-3
    assert abs(_reassemble(overlay, dist, env, 1.0) - expected) <= bound
    # on weights that do sum to one the decomposition is exact
    p_ko = np.asarray(dist.probabilities[EventType.KO], dtype=float)
    p_ki = float(dist.probabilities[EventType.MATURITY_WITH_KI])
    from quantark.cashleg.folding import _event_structure
    exact = _event_structure(product, env).distribution(p_ko, 1.0 - p_ko.sum() - p_ki, p_ki)
    expected = sum(leg.value(exact, env, 100.0) for leg in legs)
    assert _reassemble(overlay, exact, env, 1.0) == pytest.approx(expected, rel=1e-12)


# --- AC3: binding ---------------------------------------------------------------------------

def _overlay():
    return FoldOverlay(
        ko=(FoldedEventCash("main", 0.5, 0.51, 1.0), FoldedEventCash("main", 1.0, 1.01, 2.0)),
        maturity_no_ki=FoldedEventCash("main", 1.0, 1.02, 0.0),
        maturity_ki=FoldedEventCash("main", 1.0, 1.02, 3.0),
        constant_pv=0.0,
    )


def test_bind_ko_aligns_amounts_to_the_engine_records():
    amounts = _overlay().bind_ko([("main", 1.0, 1.01), ("main", 0.5, 0.51)])
    assert list(amounts) == [2.0, 1.0]
    assert _overlay().bind_terminal("ki", 1.0, 1.02) == 3.0


@pytest.mark.parametrize("records, match", [
    ([("main", 0.5, 0.51)], "no engine record"),                         # an entry left unbound
    ([("main", 0.5, 0.51), ("main", 1.0, 1.01), ("main", 0.75, 0.76)], "no overlay entry"),
    ([("main", 0.5, 0.51), ("main", 0.5, 0.51)], "no overlay entry|twice"),
    ([("main", 0.5, 0.51), ("main", 1.0, 1.05)], "payment"),
    ([("pre", 0.5, 0.51), ("main", 1.0, 1.01)], "no overlay entry"),
])
def test_bind_ko_fails_closed(records, match):
    with pytest.raises(ValidationError, match=match):
        _overlay().bind_ko(records)


@pytest.mark.parametrize("args, match", [
    (("ki", 1.0, 1.03), "payment"),
    (("ki", 0.9, 1.02), "determination"),
    (("other", 1.0, 1.02), "branch"),
])
def test_bind_terminal_fails_closed(args, match):
    with pytest.raises(ValidationError, match=match):
        _overlay().bind_terminal(*args)


def test_an_overlay_without_branches_binds_zero_and_is_empty():
    overlay = FoldOverlay(ko=(), maturity_no_ki=None, maturity_ki=None, constant_pv=5.0)
    assert overlay.is_empty
    assert overlay.bind_ko([]).size == 0
    assert overlay.bind_terminal("ki", 1.0, 1.0) == 0.0


# --- AC2 on dated contracts and their edge states (CP1 finding 1, 5) -------------------------

def _dated_cases():
    from copy import deepcopy
    from intraday.conftest import dated_phoenix, dated_snowball
    from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
    from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver

    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    t0, live = datetime(2026, 3, 16), datetime(2026, 4, 1)
    sb = dated_snowball(cal, t0)
    ko_dates = [r.observation_date for r in sb.barrier_config.ko_observation_schedule.records]
    knocked_in = deepcopy(sb)
    knocked_in._otc_lifecycle_knocked_in = True
    return {
        "snowball-live": (SnowballPDESolver, sb, flat_env(live)),
        "phoenix-live": (PhoenixPDESolver, dated_phoenix(cal, t0), flat_env(live)),
        "snowball-ko-decided-at-valuation": (SnowballPDESolver, sb, flat_env(ko_dates[2], spot=110.0)),
        "snowball-knocked-in": (SnowballPDESolver, knocked_in, flat_env(live, spot=90.0)),
        "snowball-exercise-date": (SnowballPDESolver, sb, flat_env(ko_dates[-1], spot=95.0)),
    }


@pytest.mark.parametrize("case", list(_dated_cases()))
def test_dated_structure_and_value_match_the_pde_route(case):
    from quantark.cashleg.folding import _event_structure

    solver, product, env = _dated_cases()[case]
    dist = _route_dist(solver(PDEParams()), product, env)
    structure = _event_structure(product, env)
    n = len(structure.ko)
    basis = structure.distribution(np.zeros(n), 1.0, 0.0)
    if n:
        np.testing.assert_array_equal(basis.stream_times[EventType.KO], dist.stream_times[EventType.KO])
    np.testing.assert_array_equal(basis.survival_times, dist.survival_times)
    assert (basis.stream_dates or {}).get(EventType.KO) == (dist.stream_dates or {}).get(EventType.KO)
    assert basis.determination_times == dist.determination_times
    for et in (EventType.KO, EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI):
        np.testing.assert_array_equal(np.asarray(basis.payment_times.get(et, np.nan)),
                                      np.asarray(dist.payment_times.get(et, np.nan)))
        if dist.payment_dates is not None and et in dist.payment_dates:
            assert basis.payment_dates[et] == dist.payment_dates[et]
    legs = []
    for leg in _leg_set(basis.stream_times[EventType.KO] if n else [1.0]):
        try:
            leg.value(dist, env, 100.0)               # legs the route itself refuses are out of scope
        except Exception:
            continue
        legs.append(leg)
    overlay = fold_cash_legs(legs, product, env, quantity=1.0, unit_notional=100.0)
    if EventType.KO not in dist.probabilities:
        assert overlay.is_empty and overlay.ko == ()
    expected = sum(leg.value(dist, env, 100.0) for leg in legs)
    assert _reassemble(overlay, dist, env, 1.0) == pytest.approx(expected, rel=1e-12, abs=1e-6)
