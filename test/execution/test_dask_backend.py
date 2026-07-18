"""Dask scenario backend: same plans, same reducer, no silent fallback
(spec section 12.4)."""
import dataclasses

import pytest

from quantark.execution.api import PricingSession
from quantark.execution.context import default_context
from quantark.execution.contracts import (
    PricingFailure,
    ScenarioOutcome,
    ScenarioSpec,
    economics_mapping,
)
from quantark.execution.errors import CapabilityError
from quantark.execution.policy import (
    ExecutionPolicy,
    ExecutorSelection,
    ResourceBudget,
)
from quantark.execution.scenario.contracts import BaseInputsRef

import execution.scenario_process_helpers  # noqa: F401 - registers toy fixtures


def _toy_base():
    return BaseInputsRef(
        factory_id="toy-inputs/v1",
        payload=(("spot", 100.0), ("vol", 0.25)),
    )


def _toy_spec(scenario_id, ds, runner="toy/v1"):
    return ScenarioSpec(
        scenario_id=scenario_id,
        transformer_id="toy-bump/v1",
        parameters=(("ds", ds),),
        mutation_tags=frozenset({"spot"}),
        required_capabilities=frozenset({f"runner:{runner}"}),
    )


def _dask_context(workers=2):
    return dataclasses.replace(
        default_context(),
        execution_policy=ExecutionPolicy(
            scenario=ExecutorSelection(backend="dask", workers=workers),
        ),
        resource_budget=ResourceBudget(max_processes=workers, max_threads=1),
    )


def test_explicit_dask_unavailable_raises_capability_error(monkeypatch):
    """No silent fallback when dask.distributed cannot import."""
    import builtins

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "distributed" or name.startswith("distributed."):
            raise ImportError("distributed blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)
    specs = [_toy_spec("a", 1.0)]
    with PricingSession(_dask_context()) as session:
        with pytest.raises(CapabilityError):
            session.run_scenarios(_toy_base(), specs, "toy-engine/v1")


distributed = pytest.importorskip("distributed")


@pytest.fixture(scope="module")
def dask_client():
    cluster = distributed.LocalCluster(
        n_workers=2, threads_per_worker=1, processes=True,
        dashboard_address=None,
    )
    client = distributed.Client(cluster)
    yield client
    client.close()
    cluster.close()


def test_dask_outcomes_exactly_equal_serial(dask_client):
    from quantark.execution.backends.dask_backend import run_plan_dask
    from quantark.execution.scenario.planner import plan_scenarios
    from quantark.execution.scenario.validate import compare_scenario_outcomes

    specs = [_toy_spec(f"s{i}", float(i)) for i in range(5)]
    with PricingSession() as session:
        serial = session.run_scenarios(_toy_base(), specs, "toy-engine/v1")

    plan = plan_scenarios(_toy_base(), specs, "toy-engine/v1")
    via_dask = run_plan_dask(
        plan, _toy_base(), "toy-engine/v1", _dask_context(),
        client=dask_client,
    )
    assert [o.scenario_id for o in via_dask] == [f"s{i}" for i in range(5)]
    for left, right in zip(serial, via_dask):
        assert isinstance(right, ScenarioOutcome)
        assert right.value == left.value
        assert economics_mapping(right) == economics_mapping(left)
    report = compare_scenario_outcomes(serial, via_dask)
    assert report.all_scenarios_match is True


def test_dask_collect_errors_isolates_failures(dask_client):
    from quantark.execution.backends.dask_backend import run_plan_dask
    from quantark.execution.scenario.planner import plan_scenarios

    specs = [
        _toy_spec("ok0", 0.0, runner="toy-failing/v1"),
        _toy_spec("bad", 1.0, runner="toy-failing/v1"),
        _toy_spec("ok1", -1.0, runner="toy-failing/v1"),
    ]
    plan = plan_scenarios(_toy_base(), specs, "toy-engine/v1")
    outcomes = run_plan_dask(
        plan, _toy_base(), "toy-engine/v1", _dask_context(),
        collect_errors=True, client=dask_client,
    )
    assert isinstance(outcomes[0], ScenarioOutcome)
    assert isinstance(outcomes[1], PricingFailure)
    assert outcomes[1].error.error_type == "ValueError"
    assert isinstance(outcomes[2], ScenarioOutcome)


def test_dask_rejects_native_value_runners(dask_client):
    specs = [_toy_spec("a", 1.0, runner="request/v1")]
    with PricingSession(_dask_context()) as session:
        with pytest.raises(CapabilityError):
            session.run_scenarios(_toy_base(), specs, "toy-engine/v1")


def test_legacy_use_dask_surface_preserved():
    """Spec 12.4/17.1: existing Snowball/Phoenix ``use_dask`` behavior is
    preserved. Through Phase 5 this was guarded by a no-diff-vs-main proxy;
    Phase 6 consolidated the triplicated batch loop into one legacy reducer,
    so the guard is now the preserved constructor surface here plus the
    bitwise behavioral gate in ``test_legacy_dask_goldens``."""
    import inspect

    from quantark.asset.equity.engine.mc import PhoenixMCEngine, SnowballMCEngine

    for cls in (SnowballMCEngine, PhoenixMCEngine):
        params = inspect.signature(cls.__init__).parameters
        assert "use_dask" in params, cls.__name__
        assert params["use_dask"].default is False, cls.__name__
        assert "num_batches" in params, cls.__name__
        assert params["num_batches"].default == 4, cls.__name__


def test_shared_client_concurrent_runs_do_not_collide(dask_client):
    """Same cells, same base, DIFFERENT engine factory on one borrowed
    client: dask must not deduplicate the tasks (code-gate finding
    2026-07-17 — explicit keys were task identity, so the second run could
    silently receive the first run's Futures)."""
    from quantark.execution.backends.dask_backend import run_plan_dask
    from quantark.execution.scenario.planner import plan_scenarios

    specs = [_toy_spec(f"s{i}", float(i), runner="toy-scaled/v1")
             for i in range(3)]
    plan_a = plan_scenarios(_toy_base(), specs, "toy-engine-a/v1")
    plan_b = plan_scenarios(_toy_base(), specs, "toy-engine-b/v1")
    # identical cell fingerprints: the engine factory is NOT part of them
    assert [c.cell_fingerprint for c in plan_a.cells] == [
        c.cell_fingerprint for c in plan_b.cells
    ]
    outcomes_a = run_plan_dask(
        plan_a, _toy_base(), "toy-engine-a/v1", _dask_context(),
        client=dask_client,
    )
    outcomes_b = run_plan_dask(
        plan_b, _toy_base(), "toy-engine-b/v1", _dask_context(),
        client=dask_client,
    )
    for left, right in zip(outcomes_a, outcomes_b):
        assert right.value == left.value * 1.5  # scale 3.0 vs 2.0
