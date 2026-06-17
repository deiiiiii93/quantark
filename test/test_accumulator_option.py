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
