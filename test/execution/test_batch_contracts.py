"""BatchPlan/BatchTask/BatchOutcome contracts and plan fingerprints."""
import dataclasses

import pytest

from quantark.execution.cache.fingerprint import try_fingerprint
from quantark.execution.contracts import BatchOutcome, BatchPlan, BatchTask
from quantark.execution.policy import resolve_resource_budget


def make_plan(**overrides):
    tasks = tuple(
        BatchTask(plan_id="p1", batch_index=i, batch_id=i, n_paths=256)
        for i in range(4)
    )
    base = dict(
        plan_id="p1",
        engine_class_path="quantark.asset.equity.engine.mc.dcn_mc_engine.DCNMCEngine",
        num_batches=4, paths_per_batch=256, total_paths=1024,
        seed=42, stream_kind="sobol", stream_layout="batch-shifted-sobol/v1",
        time_steps=252, dimension=252, dtype="float64",
        scheme="gbm-term/v1", stderr_mode="scramble_means",
        reduction_order="batch_index/v1", tasks=tasks,
        est_task_peak_bytes=1_000_000, est_outcome_bytes=4_096,
        implementation_fingerprint=None,
    )
    base.update(overrides)
    return BatchPlan(**base)


def test_plan_is_frozen():
    plan = make_plan()
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.seed = 43


def test_plan_fingerprint_stable_and_sensitive():
    fp1 = try_fingerprint(make_plan())
    fp2 = try_fingerprint(make_plan())
    fp3 = try_fingerprint(make_plan(seed=43))
    assert fp1 is not None and fp1 == fp2 and fp1 != fp3


def test_outcome_carries_index_and_payload():
    out = BatchOutcome(batch_index=2, n_paths=256, payload=("stats",))
    assert out.batch_index == 2 and out.n_paths == 256


def test_budget_resolves_threads_and_draw_cache():
    budget, sources = resolve_resource_budget(environ={
        "QUANTARK_EXEC_MAX_THREADS": "8",
        "QUANTARK_EXEC_DRAW_CACHE_MB": "64",
    })
    assert budget.max_threads == 8
    assert budget.draw_cache_bytes == 64 * 2**20
    fields = dict(sources)
    assert fields["max_threads"] == "env"
    assert fields["draw_cache_bytes"] == "env"


def test_budget_default_threads_is_one_without_env():
    budget, _ = resolve_resource_budget(environ={})
    assert budget.max_threads == 1  # session upgrades to cpu_count for OWNED budgets
