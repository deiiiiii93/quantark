"""Spawn process backend: clean-worker reconstruction, fault isolation,
child budgets, and no silent fallback (spec sections 12.3, 12.5, 15)."""
import dataclasses
import os
import subprocess
import sys
import types
from pathlib import Path

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
        resource_budget=ResourceBudget(
            max_processes=workers, max_threads=1, max_in_flight=workers,
        ),
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


# -------------------------------------- code-gate regressions (2026-07-17)
def test_drifting_factory_is_a_determinism_violation(tmp_path):
    """A factory reading mutable local state rebuilds a DIFFERENT base in
    the worker than the parent planned with; the worker verifies the
    rebuilt base against the parent's resolved-base fingerprint and fails
    closed."""
    state = tmp_path / "spot.txt"
    state.write_text("100.0")
    base = BaseInputsRef(
        factory_id="toy-file-inputs/v1",
        payload=(("path", str(state)), ("vol", 0.25)),
    )
    specs = [_toy_spec("a", 1.0)]
    with PricingSession(_process_context()) as session:
        # drift AFTER the parent resolved/planned, BEFORE workers rebuild
        plan_probe = session  # session resolves at run_scenarios time
        state_written = False

        import quantark.execution.scenario.worker as worker_mod

        original = worker_mod.build_worker_spec

        def drifting(plan, base_ref, context, workers):
            nonlocal state_written
            spec = original(plan, base_ref, context, workers)
            if not state_written:
                state.write_text("200.0")  # the drift
                state_written = True
            return spec

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(worker_mod, "build_worker_spec", drifting)
            outcomes = plan_probe.run_scenarios(
                base, specs, "toy-engine/v1", collect_errors=True
            )
    assert isinstance(outcomes[0], PricingFailure)
    assert outcomes[0].error.error_type == "DeterminismViolation"


def test_killed_worker_is_typed_infrastructure_error():
    """A dead worker process (os._exit) is NOT a collectable cell failure:
    it surfaces as a typed TaskExecutionError with identities even under
    collect_errors=True (retries default to zero)."""
    specs = [_toy_spec("a", 1.0, runner="toy-exit/v1")]
    with PricingSession(_process_context()) as session:
        with pytest.raises(TaskExecutionError) as excinfo:
            session.run_scenarios(
                _toy_base(), specs, "toy-engine/v1", collect_errors=True
            )
    message = str(excinfo.value)
    assert "worker process failure" in message
    assert "completed=" in message


def test_infrastructure_retry_resubmits_identical_payloads(tmp_path):
    """With retries=1, a worker death on the first attempt is retried with
    the identical serialized payloads and the run completes."""
    marker = tmp_path / "died.marker"
    specs = [
        ScenarioSpec(
            scenario_id="phoenix",
            transformer_id="toy-bump/v1",
            parameters=(("ds", 1.0), ("marker", str(marker))),
            mutation_tags=frozenset({"spot"}),
            required_capabilities=frozenset({"runner:toy-exit-once/v1"}),
        )
    ]
    ctx = _process_context()
    ctx = dataclasses.replace(
        ctx,
        execution_policy=dataclasses.replace(
            ctx.execution_policy, retries=1
        ),
    )
    with PricingSession(ctx) as session:
        outcomes = session.run_scenarios(_toy_base(), specs, "toy-engine/v1")
    assert isinstance(outcomes[0], ScenarioOutcome)
    assert marker.exists()  # first attempt really died
    assert economics_mapping(outcomes[0])["pv"] == (100.0 + 1.0) * 0.25


def test_window_is_clamped_by_max_in_flight_budget():
    """max_in_flight is a hard admission bound on concurrent cells; the
    clamp is recorded in diagnostics."""
    ctx = dataclasses.replace(
        default_context(),
        execution_policy=ExecutionPolicy(
            scenario=ExecutorSelection(backend="processes", workers=4),
        ),
        resource_budget=ResourceBudget(
            max_processes=4, max_threads=1, max_in_flight=1,
        ),
        # mark the budget operator-explicit so the session auto-budget
        # does not upgrade max_in_flight (spec 11.1)
        config_snapshot=(("max_in_flight", "explicit"),
                         ("max_processes", "explicit")),
    )
    specs = [_toy_spec(f"s{i}", float(i)) for i in range(3)]
    with PricingSession(ctx) as session:
        outcomes = session.run_scenarios(_toy_base(), specs, "toy-engine/v1")
        sink = session.context.diagnostics_sink
    assert all(isinstance(o, ScenarioOutcome) for o in outcomes)
    records = [
        r for d in sink.entries for r in getattr(d, "records", ())
    ]
    assert any(r.startswith("clamp:scenario.window=") for r in records)


# ---------------------------- spawn-safe __main__ preflight (2026-07-20)
def test_pseudo_file_main_is_rejected_before_pool_creation(monkeypatch):
    """A __main__ recorded as '<stdin>' (python - <<EOF heredoc) dooms
    EVERY spawn child at bootstrap (FileNotFoundError re-running the
    main), which the pool only surfaces as an opaque BrokenProcessPool
    with completed=0. The preflight converts that into a typed
    CapabilityError before any worker is spawned."""
    import quantark.execution.scenario.worker as worker_mod

    fake_main = types.ModuleType("__main__")
    fake_main.__file__ = "<stdin>"
    monkeypatch.setitem(sys.modules, "__main__", fake_main)
    with pytest.raises(CapabilityError, match="spawn"):
        worker_mod._verify_spawn_safe_main()


def test_deleted_file_main_is_rejected_before_pool_creation(monkeypatch):
    import quantark.execution.scenario.worker as worker_mod

    fake_main = types.ModuleType("__main__")
    fake_main.__file__ = "/nonexistent/deleted_script.py"
    monkeypatch.setitem(sys.modules, "__main__", fake_main)
    with pytest.raises(CapabilityError, match="spawn"):
        worker_mod._verify_spawn_safe_main()


@pytest.mark.parametrize("shape", ["script_file", "no_file", "module_spec"])
def test_spawn_safe_mains_pass_preflight(monkeypatch, shape):
    """Real script files, -c/REPL mains (no __file__), and -m/runpy mains
    (__spec__ set) are all left alone."""
    import importlib.machinery

    import quantark.execution.scenario.worker as worker_mod

    fake_main = types.ModuleType("__main__")
    if shape == "script_file":
        fake_main.__file__ = __file__
    elif shape == "module_spec":
        fake_main.__file__ = "<frozen imagined>"  # ignored: __spec__ wins
        fake_main.__spec__ = importlib.machinery.ModuleSpec(
            "pytest.__main__", loader=None
        )
    monkeypatch.setitem(sys.modules, "__main__", fake_main)
    worker_mod._verify_spawn_safe_main()  # must not raise


def test_stdin_heredoc_gets_typed_error_not_broken_pool():
    """End-to-end: a stdin-heredoc parent asking for the processes
    backend fails closed with CapabilityError BEFORE spawning workers
    (root-caused 2026-07-20: previously every child died at bootstrap
    and the parent reported BrokenProcessPool with completed=0)."""
    repo = Path(__file__).parents[2]
    script = """
import dataclasses
from quantark.execution.api import PricingSession
from quantark.execution.context import default_context
from quantark.execution.contracts import ScenarioSpec
from quantark.execution.errors import CapabilityError
from quantark.execution.policy import (
    ExecutionPolicy, ExecutorSelection, ResourceBudget,
)
from quantark.execution.scenario.contracts import BaseInputsRef
import execution.scenario_process_helpers  # registers toy fixtures

context = dataclasses.replace(
    default_context(),
    execution_policy=ExecutionPolicy(
        scenario=ExecutorSelection(backend="processes", workers=2),
    ),
    resource_budget=ResourceBudget(
        max_processes=2, max_threads=1, max_in_flight=2,
    ),
)
base = BaseInputsRef(
    factory_id="toy-inputs/v1", payload=(("spot", 100.0), ("vol", 0.25)),
)
specs = [
    ScenarioSpec(
        scenario_id=f"s{i}", transformer_id="toy-bump/v1",
        parameters=(("ds", float(i)),), mutation_tags=frozenset({"spot"}),
        required_capabilities=frozenset({"runner:toy/v1"}),
    )
    for i in range(3)
]
try:
    with PricingSession(context) as session:
        session.run_scenarios(base, specs, "toy-engine/v1")
except CapabilityError as exc:
    print(f"TYPED: CapabilityError: {exc}")
else:
    print("NO ERROR RAISED")
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo), str(repo / "test")]
        + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    result = subprocess.run(
        [sys.executable, "-"], input=script, capture_output=True,
        text=True, timeout=300, cwd=repo, env=env,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "TYPED: CapabilityError" in result.stdout
    assert "BrokenProcessPool" not in result.stdout + result.stderr


def test_nested_list_payload_prices_identically_serial_and_process():
    """Payload normalization is applied on BOTH sides (code-gate finding):
    a list-valued payload cannot produce backend-dependent behavior."""
    base = BaseInputsRef(
        factory_id="toy-inputs/v1",
        payload=(("spot", 100.0), ("vol", 0.25), ("extra", [1.0, 2.0])),
    )
    specs = [_toy_spec("a", 1.0)]
    with PricingSession() as session:
        serial = session.run_scenarios(base, specs, "toy-engine/v1")
    with PricingSession(_process_context()) as session:
        via_processes = session.run_scenarios(base, specs, "toy-engine/v1")
    assert economics_mapping(serial[0]) == economics_mapping(via_processes[0])
    assert serial[0].value == via_processes[0].value
