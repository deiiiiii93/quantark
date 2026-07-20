"""run_scenario_plans: multi-base packing through one pool with a
per-plan error boundary (spec 2026-07-20)."""
import dataclasses

import pytest

from quantark.execution.api import PricingSession
from quantark.execution.context import default_context
from quantark.execution.contracts import (
    PricingFailure,
    ScenarioOutcome,
    ScenarioSpec,
)
from quantark.execution.errors import CapabilityError
from quantark.execution.policy import (
    ExecutionPolicy,
    ExecutorSelection,
    ResourceBudget,
)
from quantark.execution.scenario.contracts import BaseInputsRef

import execution.scenario_process_helpers  # noqa: F401 - registers toys


def _toy_base(vol=0.25):
    return BaseInputsRef(
        factory_id="toy-inputs/v1",
        payload=(("spot", 100.0), ("vol", vol)),
    )


def _toy_spec(scenario_id, ds, runner="toy/v1"):
    return ScenarioSpec(
        scenario_id=scenario_id,
        transformer_id="toy-bump/v1",
        parameters=(("ds", ds),),
        mutation_tags=frozenset({"spot"}),
        required_capabilities=frozenset({f"runner:{runner}"}),
    )


def _process_context(workers=2, window=None):
    window = window if window is not None else workers
    return dataclasses.replace(
        default_context(),
        execution_policy=ExecutionPolicy(
            scenario=ExecutorSelection(
                backend="processes", workers=workers, max_in_flight=window
            ),
        ),
        resource_budget=ResourceBudget(
            max_processes=workers, max_threads=1, max_in_flight=window,
        ),
        config_snapshot=(
            ("max_in_flight", "explicit"), ("max_processes", "explicit"),
        ),
    )


def _plans():
    return [
        (_toy_base(0.25), [_toy_spec(f"a{i}", float(i)) for i in range(3)]),
        (_toy_base(0.30), [_toy_spec(f"b{i}", -float(i)) for i in range(2)]),
    ]


def test_multi_plan_serial_matches_single_plan_runs():
    plans = _plans()
    with PricingSession() as session:
        grouped = session.run_scenario_plans(plans, "toy-engine/v1")
    with PricingSession() as session:
        first = session.run_scenarios(plans[0][0], plans[0][1], "toy-engine/v1")
        second = session.run_scenarios(plans[1][0], plans[1][1], "toy-engine/v1")
    assert [o.value for o in grouped[0]] == [o.value for o in first]
    assert [o.value for o in grouped[1]] == [o.value for o in second]
    assert [o.scenario_id for o in grouped[1]] == ["b0", "b1"]


def test_multi_plan_processes_bitwise_and_ordered():
    plans = _plans()
    with PricingSession() as session:
        serial = session.run_scenario_plans(plans, "toy-engine/v1")
    with PricingSession(_process_context()) as session:
        via_processes = session.run_scenario_plans(plans, "toy-engine/v1")
    for serial_plan, process_plan in zip(serial, via_processes):
        assert [o.scenario_id for o in process_plan] == [
            o.scenario_id for o in serial_plan
        ]
        for left, right in zip(serial_plan, process_plan):
            assert isinstance(right, ScenarioOutcome)
            assert right.value == left.value  # exact float equality


def test_multi_plan_duplicate_ids_allowed_across_plans_not_within():
    plans = [
        (_toy_base(), [_toy_spec("same", 1.0)]),
        (_toy_base(), [_toy_spec("same", 2.0)]),  # cross-plan dup: fine
    ]
    with PricingSession() as session:
        grouped = session.run_scenario_plans(plans, "toy-engine/v1")
    assert grouped[0][0].value != grouped[1][0].value

    from quantark.util.exceptions import ValidationError

    with PricingSession() as session:
        with pytest.raises(ValidationError):
            session.run_scenario_plans(
                [(_toy_base(), [_toy_spec("dup", 1.0), _toy_spec("dup", 2.0)])],
                "toy-engine/v1",
            )


@pytest.mark.parametrize("backend", ["serial", "processes"])
def test_per_plan_error_boundary_factory_and_transformer(backend):
    """One factory-raise plan + one bad-runner-capability plan + one healthy
    plan: the healthy plan completes, failures are aligned and typed."""
    plans = [
        (BaseInputsRef(factory_id="toy-file-inputs/v1",
                       payload=(("path", "/nonexistent/spot.txt"),
                                ("vol", 0.25))),
         [_toy_spec("dead0", 0.0), _toy_spec("dead1", 1.0)]),
        (_toy_base(), [_toy_spec("bad", 1.0, runner="no-such-runner/v1")]),
        (_toy_base(), [_toy_spec("ok", 2.0)]),
    ]
    context = (
        _process_context() if backend == "processes" else default_context()
    )
    with PricingSession(context) as session:
        grouped = session.run_scenario_plans(
            plans, "toy-engine/v1", collect_errors=True
        )
    assert len(grouped[0]) == 2
    assert all(isinstance(f, PricingFailure) for f in grouped[0])
    assert "FileNotFoundError" in grouped[0][0].error.error_type
    assert all(isinstance(f, PricingFailure) for f in grouped[1])
    assert isinstance(grouped[2][0], ScenarioOutcome)
    assert grouped[2][0].scenario_id == "ok"


def test_multi_plan_collect_errors_isolates_per_cell():
    plans = [
        (_toy_base(), [_toy_spec("ok0", 0.0, runner="toy-failing/v1"),
                       _toy_spec("boom", 1.0, runner="toy-failing/v1")]),
        (_toy_base(), [_toy_spec("clean", -1.0)]),
    ]
    with PricingSession(_process_context()) as session:
        grouped = session.run_scenario_plans(
            plans, "toy-engine/v1", collect_errors=True
        )
    assert isinstance(grouped[0][0], ScenarioOutcome)
    assert isinstance(grouped[0][1], PricingFailure)
    assert isinstance(grouped[1][0], ScenarioOutcome)


def test_multi_plan_threads_backend_rejected():
    context = dataclasses.replace(
        default_context(),
        execution_policy=ExecutionPolicy(
            scenario=ExecutorSelection(backend="threads", workers=2),
        ),
    )
    with PricingSession(context) as session:
        with pytest.raises(CapabilityError):
            session.run_scenario_plans(_plans(), "toy-engine/v1")


def test_empty_plan_list_returns_empty():
    with PricingSession() as session:
        assert session.run_scenario_plans([], "toy-engine/v1") == []
