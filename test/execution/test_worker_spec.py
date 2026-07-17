"""WorkerSpec construction, JSON round-trip, and environment verification
(spec section 12.3)."""
import json

import pytest

from quantark.execution.context import default_context
from quantark.execution.contracts import ScenarioSpec
from quantark.execution.errors import CapabilityError, DeterminismViolation
from quantark.execution.scenario.contracts import (
    SCENARIO_SCHEMA_VERSION,
    BaseInputsRef,
)
from quantark.execution.scenario.planner import plan_scenarios
from quantark.execution.scenario.worker import (
    build_worker_spec,
    payload_to_worker_spec,
    verify_worker_environment,
    worker_spec_to_payload,
)

import execution.scenario_process_helpers  # noqa: F401 - registers toy fixtures


def _toy_base():
    return BaseInputsRef(
        factory_id="toy-inputs/v1",
        payload=(("spot", 100.0), ("vol", 0.2)),
    )


def _toy_spec(scenario_id, ds, runner="toy/v1"):
    return ScenarioSpec(
        scenario_id=scenario_id,
        transformer_id="toy-bump/v1",
        parameters=(("ds", ds),),
        mutation_tags=frozenset({"spot"}),
        required_capabilities=frozenset({f"runner:{runner}"}),
    )


def _plan(n=2):
    return plan_scenarios(
        _toy_base(), [_toy_spec(f"s{i}", float(i)) for i in range(n)],
        "toy-engine/v1",
    )


def test_worker_spec_round_trips_json():
    spec = build_worker_spec(_plan(), _toy_base(), default_context(), workers=2)
    payload = worker_spec_to_payload(spec)
    assert json.loads(json.dumps(payload)) == payload
    assert payload_to_worker_spec(payload) == spec
    assert spec.schema_version == SCENARIO_SCHEMA_VERSION


def test_worker_spec_carries_callable_refs_for_every_registered_piece():
    spec = build_worker_spec(_plan(), _toy_base(), default_context(), workers=2)
    refs = {(r.kind, r.ref_id) for r in spec.callable_refs}
    assert ("factory", "toy-inputs/v1") in refs
    assert ("factory", "toy-engine/v1") in refs
    assert ("transformer", "toy-bump/v1") in refs
    assert ("runner", "toy/v1") in refs
    # every ref names a real module/qualname
    for ref in spec.callable_refs:
        assert ref.module and ref.qualname


def test_child_policy_is_inner_serial_and_nested_off():
    spec = build_worker_spec(_plan(), _toy_base(), default_context(), workers=2)
    policy = dict(spec.child_policy_values)
    assert policy["scenario.backend"] == "serial"
    assert policy["batch.backend"] == "serial"
    assert policy["nested_execution"] is False


def test_child_budget_divides_the_parent_budget():
    import dataclasses

    from quantark.execution.policy import ResourceBudget

    ctx = dataclasses.replace(
        default_context(),
        resource_budget=ResourceBudget(
            max_processes=4, max_threads=8,
            artifact_cache_bytes=400 * 2**20,
            draw_cache_bytes=200 * 2**20,
        ),
    )
    spec = build_worker_spec(_plan(), _toy_base(), ctx, workers=4)
    budget = dict(spec.child_budget_values)
    assert budget["max_processes"] == 1
    assert budget["max_threads"] == 1
    assert budget["max_in_flight"] == 1
    assert budget["artifact_cache_bytes"] == 100 * 2**20
    assert budget["draw_cache_bytes"] == 50 * 2**20


def test_verification_passes_in_process():
    spec = build_worker_spec(_plan(), _toy_base(), default_context(), workers=2)
    verify_worker_environment(spec)  # same interpreter: must not raise


def test_schema_mismatch_is_capability_error():
    import dataclasses

    spec = build_worker_spec(_plan(), _toy_base(), default_context(), workers=2)
    bad = dataclasses.replace(spec, schema_version="scenario/v999")
    with pytest.raises(CapabilityError):
        verify_worker_environment(bad)


def test_dependency_mismatch_is_determinism_violation():
    import dataclasses

    spec = build_worker_spec(_plan(), _toy_base(), default_context(), workers=2)
    expected = tuple(
        (k, "0.0.0-corrupted" if k == "numpy" else v) for k, v in spec.expected
    )
    bad = dataclasses.replace(spec, expected=expected)
    with pytest.raises(DeterminismViolation):
        verify_worker_environment(bad)
