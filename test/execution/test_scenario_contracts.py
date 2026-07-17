"""Scenario contracts and registries (spec sections 12.3, 13.1)."""
import json

import pytest

from quantark.execution.errors import CapabilityError, ValidationGateError
from quantark.execution.scenario import registries
from quantark.execution.scenario.contracts import (
    SCENARIO_SCHEMA_VERSION,
    BaseInputsRef,
    CallableRef,
    ScenarioCell,
    ScenarioPlan,
    WorkerSpec,
)


def _module_level_fn(base, parameters):
    return base


def _other_module_level_fn(base, parameters):
    return base


def test_schema_version_present():
    assert SCENARIO_SCHEMA_VERSION == "scenario/v1"


def test_worker_spec_is_json_serializable():
    worker = pytest.importorskip(
        "quantark.execution.scenario.worker"
    )  # arrives in Task 5
    payload_to_worker_spec = worker.payload_to_worker_spec
    worker_spec_to_payload = worker.worker_spec_to_payload

    spec = WorkerSpec(
        schema_version=SCENARIO_SCHEMA_VERSION,
        base_ref=BaseInputsRef(factory_id="f", payload=(("a", 1),)),
        callable_refs=(
            CallableRef("factory", "f", "some.module", "build", "1"),
            CallableRef("runner", "toy/v1", "some.module", "toy_runner", "1"),
        ),
        child_policy_values=(("scenario.backend", "serial"),),
        child_budget_values=(("max_threads", 1),),
        expected=(("numpy", "2.0"),),
    )
    payload = worker_spec_to_payload(spec)
    assert json.loads(json.dumps(payload)) == payload
    assert payload_to_worker_spec(payload) == spec


def test_register_transformer_requires_importable_function():
    with pytest.raises(ValidationGateError):
        registries.register_transformer(
            "t-lambda", lambda b, p: b, allowed_tags=frozenset(), components=()
        )

    def local_fn(base, parameters):
        return base

    with pytest.raises(ValidationGateError):
        registries.register_transformer(
            "t-local", local_fn, allowed_tags=frozenset(), components=()
        )


def test_register_transformer_and_lookup():
    registries.register_transformer(
        "t-test/v1", _module_level_fn,
        allowed_tags=frozenset({"vol_surface"}), components=(),
    )
    reg = registries.get_transformer("t-test/v1")
    assert reg.fn is _module_level_fn
    assert reg.allowed_tags == frozenset({"vol_surface"})
    # idempotent same-object re-registration
    registries.register_transformer(
        "t-test/v1", _module_level_fn,
        allowed_tags=frozenset({"vol_surface"}), components=(),
    )
    # a different object under the same id is rejected
    with pytest.raises(ValidationGateError):
        registries.register_transformer(
            "t-test/v1", _other_module_level_fn,
            allowed_tags=frozenset(), components=(),
        )


def test_runner_value_kind_is_validated_and_recorded():
    registries.register_runner("r-test/v1", _module_level_fn, value_kind="float")
    assert registries.get_runner("r-test/v1").value_kind == "float"
    with pytest.raises(ValidationGateError):
        registries.register_runner(
            "r-bad/v1", _other_module_level_fn, value_kind="tuple"
        )


def test_callable_ref_built_from_registration():
    registries.register_factory("f-test/v1", _module_level_fn)
    ref = registries.callable_ref("factory", "f-test/v1")
    assert ref == CallableRef(
        kind="factory",
        ref_id="f-test/v1",
        module=__name__,
        qualname="_module_level_fn",
        schema_version="1",
    )


def test_unknown_ids_raise_capability_error():
    with pytest.raises(CapabilityError):
        registries.get_transformer("nope")
    with pytest.raises(CapabilityError):
        registries.get_runner("nope")
    with pytest.raises(CapabilityError):
        registries.get_factory("nope")


def test_check_worker_payload_rejects_non_json():
    with pytest.raises(ValidationGateError):
        registries.check_worker_payload({"x": object()})
    registries.check_worker_payload({"x": [1, 2.5, "s", None, True]})


def test_contract_dataclasses_are_frozen():
    cell = ScenarioCell(
        scenario_id="s1", position=0, transformer_id="t", runner_id="r",
        parameters=(("ds", 1.0),), mutation_tags=frozenset({"spot"}),
        changed_tags=frozenset({"spot"}), invalidate_all=False,
        cell_fingerprint="fp", group_key=("r", "t"),
    )
    with pytest.raises(Exception):
        cell.position = 1
    plan = ScenarioPlan(
        plan_id="p", schema_version=SCENARIO_SCHEMA_VERSION,
        base_kind="request", base_fingerprint=None, engine_factory_id=None,
        cells=(cell,), groups=((("r", "t"), (0,)),),
    )
    with pytest.raises(Exception):
        plan.cells = ()
