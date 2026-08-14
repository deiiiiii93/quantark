"""Tests for the banked reference arm and its resume soundness rules."""

import math

import pytest

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.evidence import CheckpointStore
from quantark.modelvalidation.reference import (
    BatchResult,
    ReferenceEstimate,
    run_reference,
)
from quantark.modelvalidation.stopping import StopReason
from quantark.modelvalidation.study import (
    CaseSpec,
    GateBounds,
    HedgeContractScale,
    SamplingPolicy,
)

QUANTITIES = ("delta",)


def _scale() -> HedgeContractScale:
    return HedgeContractScale(200.0, 100.0, 50_000_000.0)


def _bounds() -> GateBounds:
    # SE budget = 0.25 * 0.5 = 0.125 contracts
    return GateBounds(cell=0.5, mean_signed_bias=0.1)


def _policy(**overrides) -> SamplingPolicy:
    kwargs = dict(paths_per_batch=1024, min_batches=2, max_batches=8, seed=100)
    kwargs.update(overrides)
    return SamplingPolicy(**kwargs)


def _raw(contracts: float, quantity: str = "delta") -> float:
    """Raw value worth `contracts` economic units."""
    return contracts / _scale().to_economic(quantity, 1.0)


class FakeBuilder:
    """Deterministic stand-in for an MC benchmark.

    ``series`` supplies each batch's values, so the standard-error trajectory is
    exactly known and stopping can be tested without any pricing.
    """

    def __init__(self, series, *, policy=None, tag="fake"):
        self.series = series
        self.policy = policy or _policy()
        self.tag = tag
        self.calls = 0

    def identity(self, case):
        return {
            "builder": self.tag,
            "case": case.name,
            "quantities": list(QUANTITIES),
            "sampling": {
                "paths_per_batch": self.policy.paths_per_batch,
                "min_batches": self.policy.min_batches,
                "max_batches": self.policy.max_batches,
                "seed": self.policy.seed,
                "bump": self.policy.bump,
            },
        }

    def run_batch(self, case, batch_index):
        self.calls += 1
        return BatchResult(
            index=batch_index,
            seed=self.policy.seed + batch_index,
            values=self.series[batch_index],
        )


def _tight_series(n=8, value_c=1.0):
    """Batches that agree closely -> small standard error."""
    return [{"delta": _raw(value_c + 0.001 * (i % 2))} for i in range(n)]


def _noisy_series(n=8):
    """Batches that disagree wildly -> standard error never meets budget."""
    return [{"delta": _raw(1.0 + 3.0 * (-1) ** i)} for i in range(n)]


def _run(builder, *, store=None, resume=False, policy=None, case=None):
    return run_reference(
        builder=builder,
        case=case or CaseSpec(name="ordinary"),
        quantities=QUANTITIES,
        scale=_scale(),
        bounds=_bounds(),
        policy=policy or builder.policy,
        store=store,
        resume=resume,
    )


def test_estimate_is_batch_mean_with_batch_mean_se():
    values = [1.0, 2.0, 3.0, 4.0]
    builder = FakeBuilder(
        [{"delta": v} for v in values], policy=_policy(min_batches=4, max_batches=4)
    )
    estimate = _run(builder)
    assert isinstance(estimate, ReferenceEstimate)
    assert estimate.batches == 4
    assert estimate.values["delta"] == pytest.approx(2.5)
    # SE of the mean of [1,2,3,4]: sample sd sqrt(5/3), divided by sqrt(4)
    assert estimate.std_errors["delta"] == pytest.approx(math.sqrt(5.0 / 3.0) / 2.0)


def test_std_error_is_infinite_below_two_batches():
    """One batch cannot express a spread, and inf can never meet a budget."""
    from quantark.modelvalidation.reference import _estimate

    values, std_errors = _estimate(
        [BatchResult(index=0, seed=100, values={"delta": 1.0})], ("delta",)
    )
    assert values["delta"] == 1.0
    assert math.isinf(std_errors["delta"])


def test_stops_once_the_budget_is_met():
    builder = FakeBuilder(_tight_series())
    estimate = _run(builder)
    assert estimate.stopped_reason == StopReason.SE_BUDGET_MET.value
    assert estimate.batches == 2  # min_batches; tight series meets budget at once
    assert builder.calls == 2


def test_noisy_series_runs_to_max_batches():
    builder = FakeBuilder(_noisy_series())
    estimate = _run(builder)
    assert estimate.stopped_reason == StopReason.MAX_BATCHES.value
    assert estimate.batches == 8
    assert estimate.std_errors["delta"] > 0.0


def test_seeds_are_recorded_and_derived_from_the_policy():
    builder = FakeBuilder(_tight_series())
    estimate = _run(builder)
    assert estimate.seeds == tuple(100 + i for i in range(estimate.batches))


def test_wrong_seed_from_builder_is_rejected():
    class BadSeedBuilder(FakeBuilder):
        def run_batch(self, case, batch_index):
            return BatchResult(index=batch_index, seed=999, values=self.series[batch_index])

    with pytest.raises(ValidationError):
        _run(BadSeedBuilder(_tight_series()))


def test_wrong_index_from_builder_is_rejected():
    class BadIndexBuilder(FakeBuilder):
        def run_batch(self, case, batch_index):
            return BatchResult(
                index=batch_index + 5,
                seed=self.policy.seed + batch_index,
                values=self.series[batch_index],
            )

    with pytest.raises(ValidationError):
        _run(BadIndexBuilder(_tight_series()))


def test_missing_quantity_from_builder_is_rejected():
    builder = FakeBuilder([{"gamma": 1.0}] * 4)
    with pytest.raises(ValidationError):
        _run(builder)


def test_non_finite_batch_value_is_rejected():
    builder = FakeBuilder([{"delta": math.nan}] * 4)
    with pytest.raises(ValidationError):
        _run(builder)


def test_resume_reuses_the_bank_without_re_running_batches(tmp_path):
    store = CheckpointStore(tmp_path)
    first = FakeBuilder(_noisy_series())
    estimate_a = _run(first, store=store)
    assert first.calls == 8

    second = FakeBuilder(_noisy_series())
    estimate_b = _run(second, store=store, resume=True)
    assert second.calls == 0
    assert estimate_b.values == estimate_a.values
    assert estimate_b.batches == estimate_a.batches
    assert estimate_b.stopped_reason == estimate_a.stopped_reason


def test_resume_ignores_a_bank_from_a_different_policy(tmp_path):
    """Identity covers the sampling policy, so a changed budget re-runs."""
    store = CheckpointStore(tmp_path)
    original = _policy(max_batches=8)
    _run(FakeBuilder(_noisy_series(), policy=original), store=store)

    changed = _policy(max_batches=6)
    rerun = FakeBuilder(_noisy_series(), policy=changed)
    estimate = _run(rerun, store=store, resume=True)
    assert rerun.calls == 6
    assert estimate.batches == 6


def test_resume_continues_an_interrupted_bank(tmp_path):
    """A bank that never reached a stop decision keeps sampling where it left off."""
    store = CheckpointStore(tmp_path)
    identity = FakeBuilder(_noisy_series()).identity(CaseSpec(name="ordinary"))
    store.save(
        "reference",
        "ordinary",
        identity,
        {
            "batches": [
                {"index": 0, "seed": 100, "values": {"delta": _raw(1.0)}},
                {"index": 1, "seed": 101, "values": {"delta": _raw(2.0)}},
            ],
            "stopped_reason": None,
        },
    )

    builder = FakeBuilder(_noisy_series())
    estimate = _run(builder, store=store, resume=True)
    assert builder.calls > 0  # continued rather than restarting
    assert estimate.batches == 8
    assert estimate.seeds[0] == 100


def test_bank_longer_than_the_policy_allows_is_rejected(tmp_path):
    """The stopping decision must be reproducible from the banked batches."""
    store = CheckpointStore(tmp_path)
    builder = FakeBuilder(_noisy_series(n=12), policy=_policy(max_batches=8))
    identity = builder.identity(CaseSpec(name="ordinary"))
    store.save(
        "reference",
        "ordinary",
        identity,
        {
            "batches": [
                {"index": i, "seed": 100 + i, "values": {"delta": _raw(1.0 + i)}}
                for i in range(11)
            ],
            "stopped_reason": StopReason.MAX_BATCHES.value,
        },
    )

    with pytest.raises(ValidationError):
        _run(builder, store=store, resume=True)


def test_bank_claiming_an_impossible_stop_reason_is_rejected(tmp_path):
    """A bank that says 'budget met' on noisy batches did not come from this policy."""
    store = CheckpointStore(tmp_path)
    builder = FakeBuilder(_noisy_series())
    identity = builder.identity(CaseSpec(name="ordinary"))
    store.save(
        "reference",
        "ordinary",
        identity,
        {
            "batches": [
                {"index": i, "seed": 100 + i, "values": {"delta": _raw(1.0 + 3.0 * (-1) ** i)}}
                for i in range(3)
            ],
            "stopped_reason": StopReason.SE_BUDGET_MET.value,
        },
    )

    with pytest.raises(ValidationError):
        _run(builder, store=store, resume=True)


def test_bank_is_saved_after_every_batch(tmp_path):
    """Durability is per batch: an interrupt loses at most one batch of work."""
    store = CheckpointStore(tmp_path)
    saved_lengths = []
    original_save = store.save

    def spy(kind, key, identity, payload):
        saved_lengths.append(len(payload["batches"]))
        original_save(kind, key, identity, payload)

    store.save = spy  # type: ignore[method-assign]
    _run(FakeBuilder(_noisy_series()), store=store)
    assert saved_lengths == [1, 2, 3, 4, 5, 6, 7, 8]


def test_no_store_still_runs(tmp_path):
    estimate = _run(FakeBuilder(_tight_series()), store=None)
    assert estimate.batches >= 2
