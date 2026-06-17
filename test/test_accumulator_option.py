"""
Tests for the AccumulatorOption product.

An accumulator is a call-only structured forward: on each observation date the
buyer accumulates shares at a strike set below spot, with a geared loss leg below
the strike and an upper knock-out barrier.
"""

import pytest

from quantark.asset.equity.product.option import AccumulatorOption
from quantark.util.enum import (
    AccumulatorKnockOutType,
    ObservationAggregation,
    ObservationFrequency,
    ObservationType,
    OptionType,
)
from quantark.util.exceptions import ValidationError


def _base_kwargs(**overrides):
    kwargs = dict(
        strike=95.0,
        knock_out_barrier=105.0,
        initial_price=100.0,
        maturity=1.0,
        option_type=OptionType.CALL,
        notional=1_000_000.0,
        observation_type=ObservationType.DISCRETE,
        observation_dates=[0.25, 0.5, 0.75, 1.0],
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------

def test_construct_valid_accumulator():
    acc = AccumulatorOption(**_base_kwargs())
    assert acc.strike == 95.0
    assert acc.knock_out_barrier == 105.0
    assert acc.is_call()
    assert acc.knock_out_type == AccumulatorKnockOutType.TERMINATION
    assert acc.gearing == 2.0


def test_put_is_rejected_with_decumulator_hint():
    with pytest.raises(ValidationError, match="decumulator"):
        AccumulatorOption(**_base_kwargs(option_type=OptionType.PUT))


def test_barrier_must_be_above_strike():
    with pytest.raises(ValidationError, match="barrier"):
        AccumulatorOption(**_base_kwargs(knock_out_barrier=90.0))


def test_gearing_must_be_non_negative():
    with pytest.raises(ValidationError, match="[Gg]earing"):
        AccumulatorOption(**_base_kwargs(gearing=-1.0))


def test_knock_out_barrier_must_be_positive():
    with pytest.raises(ValidationError):
        AccumulatorOption(**_base_kwargs(knock_out_barrier=0.0))


# ---------------------------------------------------------------------------
# Share derivation
# ---------------------------------------------------------------------------

def test_daily_shares_derived_from_notional():
    acc = AccumulatorOption(**_base_kwargs(notional=1_000_000.0, initial_price=100.0))
    assert acc.daily_share_accumulation == pytest.approx(10_000.0)


def test_explicit_daily_shares_take_precedence():
    acc = AccumulatorOption(
        **_base_kwargs(daily_share_accumulation=500.0, notional=1_000_000.0)
    )
    assert acc.daily_share_accumulation == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Per-observation settlement
# ---------------------------------------------------------------------------

def test_observation_payoff_gain_leg():
    acc = AccumulatorOption(**_base_kwargs(daily_share_accumulation=1.0))
    # spot above strike (and below barrier): linear gain (S - K) * shares
    assert acc.get_observation_payoff(100.0) == pytest.approx(5.0)


def test_observation_payoff_geared_loss_leg():
    acc = AccumulatorOption(
        **_base_kwargs(daily_share_accumulation=1.0, gearing=2.0)
    )
    # spot below strike: geared loss gearing * (S - K) * shares
    assert acc.get_observation_payoff(90.0) == pytest.approx(2.0 * (90.0 - 95.0))


def test_observation_payoff_scales_with_contract_multiplier():
    acc = AccumulatorOption(
        **_base_kwargs(daily_share_accumulation=1.0, contract_multiplier=10.0)
    )
    assert acc.get_observation_payoff(100.0) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Observation schedule
# ---------------------------------------------------------------------------

def test_observation_times_returned_sorted():
    acc = AccumulatorOption(**_base_kwargs(observation_dates=[0.25, 0.5, 0.75, 1.0]))
    times = acc.get_observation_times()
    assert times == sorted(times)
    assert times[-1] == pytest.approx(1.0)


def test_unsorted_observation_dates_rejected():
    with pytest.raises(ValidationError, match="ascending"):
        AccumulatorOption(**_base_kwargs(observation_dates=[1.0, 0.25, 0.5]))


# ---------------------------------------------------------------------------
# Review findings (Gate 1 zenmux): correctness of KO/schedule/derivation
# ---------------------------------------------------------------------------

def test_termination_realized_accrual_stops_at_first_ko():
    # Chronological past observations: gain, gain, KO breach, then a later gain.
    # Under TERMINATION the contract ends at the first breach, so the post-KO
    # observation must not accrue.
    acc = AccumulatorOption(
        **_base_kwargs(
            daily_share_accumulation=1.0,
            knock_out_type=AccumulatorKnockOutType.TERMINATION,
            past_observations=[(0.1, 100.0), (0.2, 110.0), (0.3, 102.0)],
        )
    )
    # Only the first observation (S=100 < KO=105) accrues; S=110 is the KO,
    # and S=102 is after termination.
    assert acc.get_realized_accrual() == pytest.approx(5.0)


def test_single_day_realized_accrual_skips_only_breached_day():
    acc = AccumulatorOption(
        **_base_kwargs(
            daily_share_accumulation=1.0,
            knock_out_type=AccumulatorKnockOutType.SINGLE_DAY,
            past_observations=[(0.1, 100.0), (0.2, 110.0), (0.3, 102.0)],
        )
    )
    # S=100 -> +5, S=110 -> skipped (>= KO), S=102 -> +7. Total = 12.
    assert acc.get_realized_accrual() == pytest.approx(12.0)


def test_single_day_schedule_uses_accumulate_aggregation():
    acc = AccumulatorOption(
        **_base_kwargs(knock_out_type=AccumulatorKnockOutType.SINGLE_DAY)
    )
    assert (
        acc.observation_schedule.aggregation_mode == ObservationAggregation.ACCUMULATE
    )


def test_termination_schedule_uses_stop_first_hit_aggregation():
    acc = AccumulatorOption(
        **_base_kwargs(knock_out_type=AccumulatorKnockOutType.TERMINATION)
    )
    assert (
        acc.observation_schedule.aggregation_mode
        == ObservationAggregation.STOP_FIRST_HIT
    )


def test_notional_without_initial_price_is_rejected():
    with pytest.raises(ValidationError, match="initial_price"):
        AccumulatorOption(
            **_base_kwargs(
                daily_share_accumulation=0.0, notional=1_000_000.0, initial_price=0.0
            )
        )


def test_observation_dates_after_maturity_rejected():
    with pytest.raises(ValidationError, match="maturity"):
        AccumulatorOption(
            **_base_kwargs(maturity=1.0, observation_dates=[0.5, 2.0])
        )


# ---------------------------------------------------------------------------
# Review findings (Gate 1 zenmux, iter 2): KO-aware payoff + rebate on schedule
# ---------------------------------------------------------------------------

def test_get_payoff_zero_on_knock_out_single_day():
    acc = AccumulatorOption(
        **_base_kwargs(
            daily_share_accumulation=1.0,
            knock_out_type=AccumulatorKnockOutType.SINGLE_DAY,
        )
    )
    # spot at/above barrier: that day's accrual is cancelled -> zero
    assert acc.get_payoff(105.0) == pytest.approx(0.0)
    assert acc.get_payoff(110.0) == pytest.approx(0.0)


def test_get_payoff_rebate_on_knock_out_termination():
    acc = AccumulatorOption(
        **_base_kwargs(
            daily_share_accumulation=1.0,
            notional=1_000_000.0,
            knock_out_type=AccumulatorKnockOutType.TERMINATION,
            knock_out_rebate_rate=0.02,
        )
    )
    # rebate cash = rate * notional = 20_000
    assert acc.get_payoff(110.0) == pytest.approx(20_000.0)


def test_get_payoff_zero_on_termination_ko_without_rebate():
    acc = AccumulatorOption(
        **_base_kwargs(
            daily_share_accumulation=1.0,
            knock_out_type=AccumulatorKnockOutType.TERMINATION,
            knock_out_rebate_rate=0.0,
        )
    )
    assert acc.get_payoff(110.0) == pytest.approx(0.0)


def test_get_payoff_below_barrier_unchanged():
    acc = AccumulatorOption(**_base_kwargs(daily_share_accumulation=1.0))
    assert acc.get_payoff(100.0) == pytest.approx(5.0)


def test_termination_schedule_carries_rebate_payoff():
    acc = AccumulatorOption(
        **_base_kwargs(
            notional=1_000_000.0,
            knock_out_type=AccumulatorKnockOutType.TERMINATION,
            knock_out_rebate_rate=0.02,
        )
    )
    assert all(
        rec.payoff == pytest.approx(20_000.0)
        for rec in acc.observation_schedule.records
    )


def test_single_day_schedule_has_zero_rebate_payoff():
    acc = AccumulatorOption(
        **_base_kwargs(
            notional=1_000_000.0,
            knock_out_type=AccumulatorKnockOutType.SINGLE_DAY,
            knock_out_rebate_rate=0.02,
        )
    )
    assert all(
        rec.payoff == pytest.approx(0.0)
        for rec in acc.observation_schedule.records
    )


def test_generated_daily_schedule():
    acc = AccumulatorOption(
        **_base_kwargs(
            observation_dates=None,
            observation_frequency=ObservationFrequency.DAILY,
            maturity=0.02,
        )
    )
    times = acc.get_observation_times()
    assert len(times) > 0
    assert times[-1] == pytest.approx(0.02)
