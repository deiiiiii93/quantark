"""Spawn process backend: clean-worker reconstruction, fault isolation,
child budgets, and no silent fallback (spec sections 12.3, 12.5, 15)."""
import dataclasses

import pytest

from quantark.execution.api import PricingSession
from quantark.execution.context import default_context
from quantark.execution.contracts import (
    PricingFailure,
    PricingRequest,
    ScenarioOutcome,
    ScenarioSpec,
    economics_mapping,
)
from quantark.execution.errors import CapabilityError, TaskExecutionError
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
        payload=(("spot", 100.0), ("vol", 0.25)),  # non-default vol travels
    )


def _toy_spec(scenario_id, ds, runner="toy/v1"):
    return ScenarioSpec(
        scenario_id=scenario_id,
        transformer_id="toy-bump/v1",
        parameters=(("ds", ds),),
        mutation_tags=frozenset({"spot"}),
        required_capabilities=frozenset({f"runner:{runner}"}),
    )


def _process_context(workers=2):
    return dataclasses.replace(
        default_context(),
        execution_policy=ExecutionPolicy(
            scenario=ExecutorSelection(backend="processes", workers=workers),
        ),
        resource_budget=ResourceBudget(max_processes=workers, max_threads=1),
    )


def _serial_outcomes(specs, engine_factory="toy-engine/v1"):
    with PricingSession() as session:
        return session.run_scenarios(_toy_base(), specs, engine_factory)


def test_process_outcomes_bitwise_equal_serial_in_caller_order():
    from quantark.execution.scenario.validate import compare_scenario_outcomes

    specs = [_toy_spec(f"s{i}", float(i)) for i in range(6)]
    serial = _serial_outcomes(specs)
    with PricingSession(_process_context()) as session:
        via_processes = session.run_scenarios(
            _toy_base(), specs, "toy-engine/v1"
        )
    assert [o.scenario_id for o in via_processes] == [
        f"s{i}" for i in range(6)
    ]
    for left, right in zip(serial, via_processes):
        assert isinstance(right, ScenarioOutcome)
        assert right.value == left.value  # exact float equality
        assert economics_mapping(right) == economics_mapping(left)
    report = compare_scenario_outcomes(serial, via_processes)
    assert report.all_scenarios_match is True
    assert report.fields_compared > report.scenarios_compared


def test_collect_errors_isolates_one_poisoned_cell():
    specs = [
        _toy_spec("ok0", 0.0, runner="toy-failing/v1"),
        _toy_spec("bad", 1.0, runner="toy-failing/v1"),  # spot 101 -> boom
        _toy_spec("ok1", -1.0, runner="toy-failing/v1"),
        _toy_spec("ok2", -2.0, runner="toy-failing/v1"),
        _toy_spec("ok3", -3.0, runner="toy-failing/v1"),
        _toy_spec("ok4", -4.0, runner="toy-failing/v1"),
    ]
    with PricingSession(_process_context()) as session:
        outcomes = session.run_scenarios(
            _toy_base(), specs, "toy-engine/v1", collect_errors=True
        )
    assert isinstance(outcomes[1], PricingFailure)
    assert outcomes[1].item_id == "bad"
    assert outcomes[1].error.error_type == "ValueError"
    others = [o for i, o in enumerate(outcomes) if i != 1]
    assert all(isinstance(o, ScenarioOutcome) for o in others)


def test_fail_fast_raises_and_stops_submission_out_of_order():
    """The failing cell sits at a LATE position but finishes FIRST (earlier
    cells sleep); no new cells may be admitted after its failure lands
    (plan-gate finding 5)."""
    from quantark.execution.backends import processes as processes_backend

    specs = (
        [_toy_spec(f"slow{i}", -float(i), runner="toy-slow/v1")
         for i in range(2)]
        + [_toy_spec("late-bad", 5.0, runner="toy-slow/v1")]  # fails fast
        + [_toy_spec(f"never{i}", -10.0 - i, runner="toy-slow/v1")
           for i in range(4)]
    )
    submissions = []
    original = processes_backend.iter_ordered

    def observing(cells, spec_payload, workers, window, **kwargs):
        kwargs["observer"] = lambda submitted: submissions.append(submitted)
        return original(cells, spec_payload, workers, window, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "quantark.execution.scenario.worker.processes_iter_ordered",
            observing,
        )
        with PricingSession(_process_context(workers=3)) as session:
            with pytest.raises(TaskExecutionError) as excinfo:
                session.run_scenarios(_toy_base(), specs, "toy-engine/v1")
    assert "late-boom" in str(excinfo.value)
    # window is 3: the three in-flight cells were submitted, the failure
    # completed first, and nothing beyond the window was ever admitted.
    assert max(submissions) <= 3


def test_child_budget_and_nested_off_inside_the_worker():
    specs = [_toy_spec("probe", 0.0, runner="toy-budget/v1")]
    with PricingSession(_process_context()) as session:
        outcomes = session.run_scenarios(_toy_base(), specs, "toy-engine/v1")
    economics = economics_mapping(outcomes[0])
    assert economics["max_processes"] == 1.0
    assert economics["max_threads"] == 1.0
    assert economics["nested_execution"] is False
    assert economics["scenario_backend_is_serial"] is True
    assert economics["batch_backend_is_serial"] is True


def test_live_request_base_is_rejected_for_processes():
    specs = [_toy_spec("a", 1.0)]
    live_base = PricingRequest(product=object(), pricing_env=None)
    with PricingSession(_process_context()) as session:
        with pytest.raises(CapabilityError):
            session.run_scenarios(live_base, specs, "toy-engine/v1")


def test_native_value_runner_is_rejected_for_processes():
    specs = [_toy_spec("a", 1.0, runner="request/v1")]
    with PricingSession(_process_context()) as session:
        with pytest.raises(CapabilityError):
            session.run_scenarios(_toy_base(), specs, "toy-engine/v1")


def test_callable_engine_factory_is_rejected_for_processes():
    specs = [_toy_spec("a", 1.0)]
    with PricingSession(_process_context()) as session:
        with pytest.raises(CapabilityError):
            session.run_scenarios(_toy_base(), specs, lambda p: None)


def test_worker_verification_mismatch_fails_before_numerical_work():
    import quantark.execution.scenario.worker as worker_mod

    specs = [_toy_spec("a", 1.0)]
    original = worker_mod.build_worker_spec

    def corrupting(plan, base_ref, context, workers):
        spec = original(plan, base_ref, context, workers)
        expected = tuple(
            (k, "0.0.0-corrupted" if k == "scipy" else v)
            for k, v in spec.expected
        )
        return dataclasses.replace(spec, expected=expected)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(worker_mod, "build_worker_spec", corrupting)
        with PricingSession(_process_context()) as session:
            outcomes = session.run_scenarios(
                _toy_base(), specs, "toy-engine/v1", collect_errors=True
            )
    assert isinstance(outcomes[0], PricingFailure)
    assert outcomes[0].error.error_type == "DeterminismViolation"
