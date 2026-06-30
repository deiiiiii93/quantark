"""Unit tests for AutocallableCashLeg against an independent R re-implementation."""

import numpy as np
import pytest

from quantark.cashleg.base import LegDirection
from quantark.cashleg.event_distribution import EventDistribution, EventType
from quantark.cashleg.autocallable_leg import (
    AutocallableCashLeg,
    AutocallableLegType,
    PvFormula,
    AccrualBasis,
)
from quantark.util.exceptions import ValidationError


def _leg(**kw):
    base = dict(
        direction=LegDirection.BUYER_RECEIVES,
        leg_type=AutocallableLegType.MARGIN,
        notional=1_000_000.0,
        rate=0.05,
        observation_schedule=(0.5, 1.0),
        accrual_factors=(0.5, 1.0),
        settlement_schedule=(0.51, 1.01),
        terminal_accrual_factor=1.0,
        terminal_settlement_time=1.01,
    )
    base.update(kw)
    return AutocallableCashLeg(**base)


# --- Task 1: construction & validation ---

def test_constructs_with_defaults():
    leg = _leg()
    assert leg.pv_formula is PvFormula.NORMAL
    assert leg.accrual_basis is AccrualBasis.KO_MATURITY
    assert leg.terminal_events == frozenset(
        {EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI}
    )
    assert leg.requires_event_distribution() is True
    assert leg.sign() == 1


def test_missing_leg_type_rejected():
    with pytest.raises(ValidationError):
        _leg(leg_type=None)


def test_unequal_schedule_lengths_rejected():
    with pytest.raises(ValidationError):
        _leg(accrual_factors=(0.5,))


def test_future_dated_notional_settlement_rejected_for_margin_formula():
    with pytest.raises(ValidationError):
        _leg(pv_formula=PvFormula.NOTIONAL_MINUS_PAYOFF, notional_settlement_time=0.25)


def test_terminal_events_must_be_maturity_buckets():
    with pytest.raises(ValidationError):
        _leg(terminal_events=frozenset({EventType.KO}))


def test_nan_accrual_factor_rejected():
    with pytest.raises(ValidationError):
        _leg(accrual_factors=(0.5, float("nan")))


def test_negative_observation_time_rejected():
    with pytest.raises(ValidationError):
        _leg(observation_schedule=(-0.1, 1.0), settlement_schedule=(-0.1, 1.0))


def test_negative_notional_rejected():
    with pytest.raises(ValidationError):
        _leg(notional=-1.0)


# --- Task 2: synthetic-distribution helper + R oracle ---

class FlatEnv:
    """Minimal pricing-env stub: continuously-compounded flat rate."""

    def __init__(self, rate: float = 0.02):
        self._r = rate

    def get_discount_factor(self, t: float) -> float:
        return float(np.exp(-self._r * float(t)))


def make_distribution(event_times, ko_probs, mat_no_ko, mat_with_ki, coupon_probs=None):
    event_times = np.asarray(event_times, dtype=float)
    ko_probs = np.asarray(ko_probs, dtype=float)
    survival = np.concatenate([[1.0], 1.0 - np.cumsum(ko_probs)])
    probs = {
        EventType.KO: ko_probs,
        EventType.MATURITY_NO_KO: float(mat_no_ko),
        EventType.MATURITY_WITH_KI: float(mat_with_ki),
    }
    if coupon_probs is not None:
        probs[EventType.COUPON] = np.asarray(coupon_probs, dtype=float)
    return EventDistribution(
        event_times=event_times,
        event_dates=None,
        probabilities=probs,
        survival_probability=survival,
    )


def expected_R(notional, rate, accrual_factors, settlement_schedule, event_probs,
               terminal_accrual_factor, p_term, terminal_settlement_time, env):
    af = np.asarray(accrual_factors, float)
    ss = np.asarray(settlement_schedule, float)
    ep = np.asarray(event_probs, float)
    df_obs = np.array([env.get_discount_factor(t) for t in ss])
    df_term = env.get_discount_factor(terminal_settlement_time)
    return notional * rate * (
        float(np.sum(af * ep * df_obs)) + terminal_accrual_factor * p_term * df_term
    )


def test_helper_builds_valid_distribution():
    ed = make_distribution([0.5, 1.0], [0.3, 0.2], 0.4, 0.1)
    assert ed.probabilities[EventType.KO].tolist() == [0.3, 0.2]
    assert abs(ed.survival_at(1.0) - 0.5) < 1e-12


# --- Task 3: KO_MATURITY valuation + guards ---

def test_normal_ko_maturity_value_matches_oracle():
    env = FlatEnv(0.02)
    ed = make_distribution([0.5, 1.0], [0.3, 0.2], 0.4, 0.1)
    leg = _leg(pv_formula=PvFormula.NORMAL, rate=0.05, notional=1_000_000.0,
               observation_schedule=(0.5, 1.0), accrual_factors=(0.5, 1.0),
               settlement_schedule=(0.5, 1.0), terminal_accrual_factor=1.0,
               terminal_settlement_time=1.0)
    R = expected_R(1_000_000.0, 0.05, (0.5, 1.0), (0.5, 1.0), [0.3, 0.2], 1.0, 0.5, 1.0, env)
    assert abs(leg.value(ed, env, 0.0) - R) < 1e-6


def test_margin_formula_is_notional_minus_R():
    env = FlatEnv(0.02)
    ed = make_distribution([0.5, 1.0], [0.3, 0.2], 0.4, 0.1)
    leg = _leg(pv_formula=PvFormula.NOTIONAL_MINUS_PAYOFF, rate=0.05, notional=1_000_000.0,
               observation_schedule=(0.5, 1.0), accrual_factors=(0.5, 1.0),
               settlement_schedule=(0.5, 1.0), terminal_accrual_factor=1.0,
               terminal_settlement_time=1.0)
    R = expected_R(1_000_000.0, 0.05, (0.5, 1.0), (0.5, 1.0), [0.3, 0.2], 1.0, 0.5, 1.0, env)
    assert abs(leg.value(ed, env, 0.0) - (1_000_000.0 - R)) < 1e-6


def test_buyer_pays_flips_sign():
    env = FlatEnv(0.02)
    ed = make_distribution([0.5, 1.0], [0.3, 0.2], 0.4, 0.1)
    recv = _leg(direction=LegDirection.BUYER_RECEIVES, observation_schedule=(0.5, 1.0),
                settlement_schedule=(0.5, 1.0), terminal_settlement_time=1.0)
    pays = _leg(direction=LegDirection.BUYER_PAYS, observation_schedule=(0.5, 1.0),
                settlement_schedule=(0.5, 1.0), terminal_settlement_time=1.0)
    assert abs(recv.value(ed, env, 0.0) + pays.value(ed, env, 0.0)) < 1e-9


def test_missing_ko_stream_raises():
    env = FlatEnv(0.02)
    trivial = EventDistribution.trivial(1.0)
    leg = _leg()
    with pytest.raises(ValidationError):
        leg.value(trivial, env, 0.0)


def test_shifted_observation_schedule_raises():
    env = FlatEnv(0.02)
    ed = make_distribution([0.5, 1.0], [0.3, 0.2], 0.4, 0.1)
    leg = _leg(observation_schedule=(0.49, 1.0), settlement_schedule=(0.49, 1.0),
               terminal_settlement_time=1.0)
    with pytest.raises(ValidationError):
        leg.value(ed, env, 0.0)


def test_missing_terminal_bucket_raises():
    env = FlatEnv(0.02)
    ed = EventDistribution(
        event_times=np.array([0.5, 1.0]),
        event_dates=None,
        probabilities={EventType.KO: np.array([0.3, 0.2]),
                       EventType.MATURITY_NO_KO: 0.5},
        survival_probability=np.array([1.0, 0.7, 0.5]),
    )
    leg = _leg(observation_schedule=(0.5, 1.0), settlement_schedule=(0.5, 1.0),
               terminal_settlement_time=1.0)
    with pytest.raises(ValidationError):
        leg.value(ed, env, 0.0)


@pytest.mark.parametrize("lt", list(AutocallableLegType))
def test_all_five_leg_types_value_finite(lt):
    env = FlatEnv(0.02)
    ed = make_distribution([0.5, 1.0], [0.3, 0.2], 0.4, 0.1)
    formula = (PvFormula.NOTIONAL_MINUS_PAYOFF
               if lt is AutocallableLegType.MARGIN else PvFormula.NORMAL)
    leg = _leg(leg_type=lt, pv_formula=formula,
               observation_schedule=(0.5, 1.0), settlement_schedule=(0.5, 1.0),
               terminal_settlement_time=1.0)
    assert np.isfinite(leg.value(ed, env, 0.0))


def test_value_ignores_position_notional_argument():
    env = FlatEnv(0.02)
    ed = make_distribution([0.5, 1.0], [0.3, 0.2], 0.4, 0.1)
    leg = _leg(observation_schedule=(0.5, 1.0), settlement_schedule=(0.5, 1.0),
               terminal_settlement_time=1.0)
    assert leg.value(ed, env, 0.0) == leg.value(ed, env, 999_999.0)


# --- Task 4: COUPON basis + value_standalone ---

def test_coupon_basis_uses_coupon_stream():
    env = FlatEnv(0.02)
    ed = make_distribution([0.5, 1.0], [0.1, 0.1], 0.4, 0.4, coupon_probs=[0.7, 0.5])
    leg = _leg(accrual_basis=AccrualBasis.COUPON, rate=0.03,
               observation_schedule=(0.5, 1.0), accrual_factors=(0.5, 0.5),
               settlement_schedule=(0.5, 1.0), terminal_accrual_factor=0.0,
               terminal_settlement_time=1.0)
    R = expected_R(1_000_000.0, 0.03, (0.5, 0.5), (0.5, 1.0), [0.7, 0.5], 0.0, 0.8, 1.0, env)
    assert abs(leg.value(ed, env, 0.0) - R) < 1e-6


def test_coupon_basis_missing_stream_raises():
    env = FlatEnv(0.02)
    ed = make_distribution([0.5, 1.0], [0.1, 0.1], 0.4, 0.4)  # no coupon stream
    leg = _leg(accrual_basis=AccrualBasis.COUPON, observation_schedule=(0.5, 1.0),
               settlement_schedule=(0.5, 1.0), terminal_settlement_time=1.0)
    with pytest.raises(ValidationError):
        leg.value(ed, env, 0.0)


class _StubEngine:
    def __init__(self, ed):
        self._ed = ed

    def price_with_events(self, product, env, emit_distribution=True):
        from quantark.cashleg.event_distribution import PricingResult
        return PricingResult(npv=0.0, event_distribution=self._ed)


def test_value_standalone_routes_through_price_with_events():
    env = FlatEnv(0.02)
    ed = make_distribution([0.5, 1.0], [0.3, 0.2], 0.4, 0.1)
    leg = _leg(observation_schedule=(0.5, 1.0), settlement_schedule=(0.5, 1.0),
               terminal_settlement_time=1.0)
    direct = leg.value(ed, env, 0.0)
    via = leg.value_standalone(parent_product=object(), engine=_StubEngine(ed), env=env)
    assert abs(direct - via) < 1e-12


# --- Task 5: exports ---

def test_public_exports():
    import quantark.cashleg as cl
    for name in ("AutocallableCashLeg", "AutocallableLegType", "PvFormula", "AccrualBasis"):
        assert hasattr(cl, name), name
