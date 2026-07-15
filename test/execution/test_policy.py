"""Policy objects and field-by-field precedence resolution (spec section 17.2)."""
import dataclasses

import pytest

from quantark.execution.policy import (
    DeterminismPolicy,
    ExecutionPolicy,
    ExecutorSelection,
    ResourceBudget,
    resolve_execution_policy,
    resolve_resource_budget,
)


def test_defaults_are_serial_one_worker_fail_fast():
    policy = ExecutionPolicy()
    assert policy.batch.backend == "serial"
    assert policy.batch.workers == 1
    assert policy.scenario.backend == "serial"
    assert policy.nested_execution is False
    assert policy.fail_fast is True
    assert policy.retries == 0
    assert policy.batch.fallback_order == ()


def test_policy_objects_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        ExecutionPolicy().fail_fast = False
    with pytest.raises(dataclasses.FrozenInstanceError):
        DeterminismPolicy().require_manifest = False
    with pytest.raises(dataclasses.FrozenInstanceError):
        ResourceBudget().max_threads = 4


def test_env_alias_resolution():
    env = {
        "QUANTARK_EXEC_BATCH_BACKEND": "threads",
        "QUANTARK_EXEC_BATCH_WORKERS": "4",
    }
    policy, sources = resolve_execution_policy(explicit=None, environ=env)
    assert policy.batch.backend == "threads"
    assert policy.batch.workers == 4
    src = dict(sources)
    assert src["batch.backend"] == "env"
    assert src["batch.workers"] == "env"
    assert src["scenario.backend"] == "default"


def test_explicit_wins_over_env_field_by_field():
    env = {"QUANTARK_EXEC_BATCH_WORKERS": "4"}
    explicit = ExecutionPolicy(batch=ExecutorSelection(backend="serial", workers=2))
    policy, sources = resolve_execution_policy(explicit=explicit, environ=env)
    # Explicit object wins wholesale for fields it sets.
    assert policy.batch.workers == 2
    assert dict(sources)["batch.workers"] == "explicit"


def test_invalid_env_text_falls_back_to_default():
    env = {
        "QUANTARK_EXEC_BATCH_WORKERS": "not-a-number",
        "QUANTARK_EXEC_BATCH_BACKEND": "quantum",
    }
    policy, sources = resolve_execution_policy(explicit=None, environ=env)
    assert policy.batch.workers == 1
    assert policy.batch.backend == "serial"
    src = dict(sources)
    assert src["batch.workers"] == "env_invalid_default"
    assert src["batch.backend"] == "env_invalid_default"


def test_resource_budget_env_resolution():
    env = {"QUANTARK_EXEC_MEMORY_MB": "1024", "QUANTARK_EXEC_MAX_IN_FLIGHT": "2"}
    budget, sources = resolve_resource_budget(explicit=None, environ=env)
    assert budget.total_memory_bytes == 1024 * 2**20  # 1024 MiB in bytes
    assert budget.max_in_flight == 2
    assert dict(sources)["total_memory_bytes"] == "env"
