"""Weighted-averaging support for Asian options (sub-project D).

Covers the data model (AsianObservationRecord.weight), the resolve_observations
weight plumbing (normalized to sum to 1), and the weighted get_average primitive.
"""
import math
from datetime import datetime

import pytest

from quantark.asset.equity.product.option.asian_option import (
    AsianObservationRecord,
    AsianOption,
)
from quantark.priceenv import PricingEnvironment
from quantark.param.rrf import FlatRateCurve
from quantark.util.enum import AveragingType
from quantark.util.exceptions import ValidationError


def _env():
    return PricingEnvironment(
        rate_curve=FlatRateCurve(rate=0.05), valuation_date=datetime(2025, 1, 1)
    )


# --- Data model -------------------------------------------------------------

def test_observation_record_defaults_to_unweighted():
    rec = AsianObservationRecord(observation_time=0.5)
    assert rec.weight is None


def test_observation_record_stores_weight():
    rec = AsianObservationRecord(observation_time=0.5, weight=2.0)
    assert rec.weight == 2.0


def test_observation_record_rejects_nonpositive_weight():
    rec = AsianObservationRecord(observation_time=0.5, weight=0.0)
    with pytest.raises(ValidationError, match="weight must be positive"):
        rec.validate()
    rec_neg = AsianObservationRecord(observation_time=0.5, weight=-1.0)
    with pytest.raises(ValidationError, match="weight must be positive"):
        rec_neg.validate()


# --- resolve_observations weight plumbing -----------------------------------

def test_resolve_uniform_weights_sum_to_one():
    times = [0.25, 0.5, 0.75, 1.0]
    opt = AsianOption(
        strike=100.0,
        maturity=1.0,
        observation_records=[AsianObservationRecord(observation_time=t) for t in times],
    )
    past_prices, past_weights, future_times, future_weights, n = opt.resolve_observations(_env())
    assert past_prices == []
    assert past_weights == []
    assert n == 4
    assert future_weights == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert sum(future_weights) == pytest.approx(1.0)


def test_resolve_explicit_weights_normalized_across_all_observations():
    # two past (observed), two future; raw weights [1,1,2,2] -> normalized /6
    recs = [
        AsianObservationRecord(observation_time=-0.5, observed_price=90.0, weight=1.0),
        AsianObservationRecord(observation_time=-0.25, observed_price=110.0, weight=1.0),
        AsianObservationRecord(observation_time=0.25, weight=2.0),
        AsianObservationRecord(observation_time=0.5, weight=2.0),
    ]
    opt = AsianOption(strike=100.0, maturity=0.5, observation_records=recs)
    past_prices, past_weights, future_times, future_weights, n = opt.resolve_observations(_env())
    assert past_prices == [90.0, 110.0]
    assert n == 4
    assert past_weights == pytest.approx([1 / 6, 1 / 6])
    assert future_weights == pytest.approx([2 / 6, 2 / 6])
    assert sum(past_weights) + sum(future_weights) == pytest.approx(1.0)


def test_resolve_legacy_times_are_uniform():
    opt = AsianOption(strike=100.0, maturity=1.0, num_observations=4)
    _, past_weights, future_times, future_weights, n = opt.resolve_observations(_env())
    assert past_weights == []
    assert n == 4
    assert future_weights == pytest.approx([0.25, 0.25, 0.25, 0.25])


# --- weighted get_average primitive -----------------------------------------

def test_get_average_arithmetic_weighted():
    opt = AsianOption(strike=100.0, maturity=1.0, averaging_type=AveragingType.ARITHMETIC)
    # (100*1 + 200*3) / (1+3) = 175
    avg = opt.get_average([100.0, 200.0], weights=[1.0, 3.0])
    assert avg == pytest.approx(175.0)


def test_get_average_arithmetic_uniform_unchanged():
    opt = AsianOption(strike=100.0, maturity=1.0, averaging_type=AveragingType.ARITHMETIC)
    assert opt.get_average([100.0, 200.0]) == pytest.approx(150.0)


def test_get_average_rejects_nonpositive_weights():
    opt = AsianOption(strike=100.0, maturity=1.0, averaging_type=AveragingType.ARITHMETIC)
    with pytest.raises(ValidationError, match="weight"):
        opt.get_average([100.0, 200.0], weights=[-1.0, 2.0])
    with pytest.raises(ValidationError, match="weight"):
        opt.get_average([100.0, 200.0], weights=[0.0, 1.0])


def test_resolve_rejects_nonpositive_component_weights():
    # Build valid, then mutate to bypass construction-time validation, exercising
    # _normalize_weights' own defensive check (direct resolve_observations callers).
    recs = [
        AsianObservationRecord(observation_time=0.5, weight=1.0),
        AsianObservationRecord(observation_time=1.0, weight=2.0),
    ]
    opt = AsianOption(strike=100.0, maturity=1.0, observation_records=recs)
    recs[0].weight = -1.0
    with pytest.raises(ValidationError, match="weight"):
        opt.resolve_observations(_env())


def test_get_average_geometric_weighted():
    opt = AsianOption(strike=100.0, maturity=1.0, averaging_type=AveragingType.GEOMETRIC)
    # exp( (1*ln100 + 3*ln200) / 4 )
    expected = math.exp((1 * math.log(100.0) + 3 * math.log(200.0)) / 4)
    avg = opt.get_average([100.0, 200.0], weights=[1.0, 3.0])
    assert avg == pytest.approx(expected)
