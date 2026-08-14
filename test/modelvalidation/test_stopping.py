"""Tests for gate-driven sequential stopping."""

import math

import pytest

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.stopping import StopReason, should_stop
from quantark.modelvalidation.study import (
    GateBounds,
    HedgeContractScale,
    SamplingPolicy,
)


def _scale() -> HedgeContractScale:
    return HedgeContractScale(200.0, 100.0, 50_000_000.0)


def _bounds() -> GateBounds:
    # SE budget = 0.25 * 0.5 = 0.125 contracts
    return GateBounds(cell=0.5, mean_signed_bias=0.1)


def _policy(**overrides) -> SamplingPolicy:
    kwargs = dict(paths_per_batch=1024, min_batches=4, max_batches=16, seed=7)
    kwargs.update(overrides)
    return SamplingPolicy(**kwargs)


def _raw_se(contracts: float, quantity: str = "delta") -> float:
    """Invert the economic scale: how much raw SE is `contracts` worth?"""
    scale = _scale()
    unit = scale.to_economic(quantity, 1.0)
    return contracts / unit


def test_never_stops_below_min_batches():
    decision = should_stop(
        std_errors_raw={"delta": _raw_se(0.0001)},
        batches=3,
        scale=_scale(),
        bounds=_bounds(),
        policy=_policy(),
    )
    assert not decision.stop
    assert decision.reason == StopReason.BELOW_MIN_BATCHES
    assert decision.batches == 3


def test_stops_when_every_quantity_meets_budget():
    decision = should_stop(
        std_errors_raw={"delta": _raw_se(0.05), "pv": _raw_se(0.10, "pv")},
        batches=4,
        scale=_scale(),
        bounds=_bounds(),
        policy=_policy(),
    )
    assert decision.stop
    assert decision.reason == StopReason.SE_BUDGET_MET


def test_one_noisy_quantity_keeps_sampling():
    """Every quantity must be sharp -- the weakest one sets the pace."""
    decision = should_stop(
        std_errors_raw={"delta": _raw_se(0.01), "gamma": _raw_se(0.5, "gamma")},
        batches=8,
        scale=_scale(),
        bounds=_bounds(),
        policy=_policy(),
    )
    assert not decision.stop
    assert decision.reason == StopReason.CONTINUE


def test_max_batches_stops_even_when_noisy():
    decision = should_stop(
        std_errors_raw={"delta": _raw_se(5.0)},
        batches=16,
        scale=_scale(),
        bounds=_bounds(),
        policy=_policy(),
    )
    assert decision.stop
    assert decision.reason == StopReason.MAX_BATCHES


def test_max_batches_does_not_mask_a_met_budget():
    """At the cap with a sharp benchmark, the honest reason is the budget."""
    decision = should_stop(
        std_errors_raw={"delta": _raw_se(0.01)},
        batches=16,
        scale=_scale(),
        bounds=_bounds(),
        policy=_policy(),
    )
    assert decision.stop
    assert decision.reason == StopReason.SE_BUDGET_MET


def test_min_batches_outranks_max_when_policy_is_degenerate():
    policy = _policy(min_batches=4, max_batches=4)
    decision = should_stop(
        std_errors_raw={"delta": _raw_se(9.0)},
        batches=4,
        scale=_scale(),
        bounds=_bounds(),
        policy=policy,
    )
    assert decision.stop
    assert decision.reason == StopReason.MAX_BATCHES


def test_infinite_se_is_treated_as_not_met():
    decision = should_stop(
        std_errors_raw={"delta": math.inf},
        batches=8,
        scale=_scale(),
        bounds=_bounds(),
        policy=_policy(),
    )
    assert not decision.stop


def test_empty_std_errors_rejected():
    with pytest.raises(ValidationError):
        should_stop(
            std_errors_raw={},
            batches=8,
            scale=_scale(),
            bounds=_bounds(),
            policy=_policy(),
        )


def test_negative_batches_rejected():
    with pytest.raises(ValidationError):
        should_stop(
            std_errors_raw={"delta": 0.0},
            batches=-1,
            scale=_scale(),
            bounds=_bounds(),
            policy=_policy(),
        )
