"""Typed surface-shock scenario port: menu, footprint, serial-vs-direct
equality, the serial-vs-process complete-payload gate, and fault
isolation (Phase 5 exit gates)."""
import dataclasses
import inspect

import pytest

from quantark.asset.equity.riskmeasures.surface_shock_scenarios import (
    SURFACE_SHOCK_RUNNER_ID,
    SurfaceShockCell,
    build_surface_shock_cells,
    cell_scenario_id,
    cells_to_scenario_specs,
    surface_shock_economics,
)
from quantark.execution.api import PricingSession
from quantark.execution.context import default_context
from quantark.execution.contracts import (
    PricingFailure,
    ScenarioOutcome,
    economics_mapping,
)
from quantark.execution.errors import ValidationGateError
from quantark.execution.policy import (
    ExecutionPolicy,
    ExecutorSelection,
    ResourceBudget,
)
from quantark.execution.scenario.contracts import BaseInputsRef
from quantark.execution.scenario.validate import compare_scenario_outcomes

from execution.surface_shock_process_helpers import (
    SURFACE_SHOCK_TEST_FACTORY_ID,
    build_surface_shock_test_inputs,
)

PATHS = 2 ** 12  # convention tests, not accuracy gates (matches the
#                  existing surface-shock pipeline tests)
SOLUTION_MONEYNESS_BUCKETS = ((-0.40, -0.10), (-0.10, 0.10), (0.10, 0.40))


def _base_ref(**overrides):
    payload = {"num_paths": PATHS, "seed": 42}
    payload.update(overrides)
    return BaseInputsRef(
        factory_id=SURFACE_SHOCK_TEST_FACTORY_ID,
        payload=tuple(sorted(payload.items())),
    )


def _fixture_tenors():
    return (91 / 365.0, 182 / 365.0)


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


# ------------------------------------------------------------------ menu
def test_solution_menu_shape_is_data_driven():
    # the solution's production data recorded 13 cells: 4 global +
    # 6 tenor + 3 moneyness
    cells = build_surface_shock_cells(
        tenors=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
        moneyness_buckets=SOLUTION_MONEYNESS_BUCKETS,
        dsigma=0.005,
    )
    assert len(cells) == 13
    ids = [cell_scenario_id(c) for c in cells]
    assert len(set(ids)) == 13
    # fixture-shaped menu: 4 + 2 + 3
    fixture_cells = build_surface_shock_cells(
        _fixture_tenors(), SOLUTION_MONEYNESS_BUCKETS, 0.005
    )
    assert len(fixture_cells) == 9


def test_no_name_parsing_in_the_port():
    """The typed port must never reconstruct cell semantics from ids: no
    .startswith()/.split() CALLS anywhere in the module (docstrings may
    describe the banned pattern)."""
    import ast

    import quantark.asset.equity.riskmeasures.surface_shock_scenarios as mod

    tree = ast.parse(inspect.getsource(mod))
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "startswith" not in calls
    assert "split" not in calls


def test_specs_plan_with_verified_vol_surface_footprint():
    from quantark.execution.scenario.planner import plan_scenarios

    cells = build_surface_shock_cells(
        _fixture_tenors(), SOLUTION_MONEYNESS_BUCKETS, 0.005
    )
    specs = cells_to_scenario_specs(cells)
    plan = plan_scenarios(
        build_surface_shock_test_inputs(
            {"num_paths": PATHS, "seed": 42}
        ),
        specs, None,
    )
    assert len(plan.cells) == 9
    for cell in plan.cells:
        assert cell.changed_tags == frozenset({"vol_surface"})
        assert cell.runner_id == SURFACE_SHOCK_RUNNER_ID


def spot_leaking_shock(base, parameters):
    import quantark.asset.equity.riskmeasures.surface_shock_scenarios as m

    shocked = m.surface_shock_transformer(base, parameters)
    return dataclasses.replace(shocked, spot=base.spot + 1.0)


def test_under_declared_surface_transformer_variant_raises():
    """A shock transformer that also moved spot would violate its
    footprint declaration."""
    from quantark.execution.contracts import ScenarioSpec
    from quantark.execution.scenario import registries
    from quantark.execution.scenario.planner import plan_scenarios

    registries.register_transformer(
        "surface-shock-leaky/v1", spot_leaking_shock,
        allowed_tags=frozenset({"vol_surface", "spot"}),
        components=(
            ("vol_surface", lambda s: s.spot * 0.0),  # ignores the surface
            ("spot", lambda s: s.spot),
        ),
    )
    spec = ScenarioSpec(
        scenario_id="leaky",
        transformer_id="surface-shock-leaky/v1",
        parameters=(
            ("dsigma", 0.005), ("moneyness_bucket", None),
            ("tenor_bucket", None),
        ),
        mutation_tags=frozenset({"vol_surface"}),  # under-declared: no spot
        required_capabilities=frozenset(
            {f"runner:{SURFACE_SHOCK_RUNNER_ID}"}
        ),
    )
    base = build_surface_shock_test_inputs({"num_paths": PATHS, "seed": 42})
    with pytest.raises(ValidationGateError):
        plan_scenarios(base, [spec], None)


# --------------------------------------------------- serial vs direct
def test_serial_cells_equal_direct_pipeline_calls():
    from quantark.asset.equity.riskmeasures.surface_shock_pipeline import (
        SurfaceShockMode,
        run_surface_shock_pipeline,
    )
    import quantark.asset.equity.riskmeasures.surface_shock_scenarios as mod

    cells = build_surface_shock_cells((), (), 0.005)  # the 4 global cells
    specs = cells_to_scenario_specs(cells)
    with PricingSession() as session:
        outcomes = session.run_scenarios(_base_ref(), specs, None)

    base = build_surface_shock_test_inputs({"num_paths": PATHS, "seed": 42})
    for cell, outcome in zip(cells, outcomes):
        assert isinstance(outcome, ScenarioOutcome)
        direct = run_surface_shock_pipeline(
            product=base.product,
            base_env_builder=mod._env_builder(base),
            cleaned=base.cleaned,
            spot=base.spot,
            rate_curve=base.rate_curve,
            carry_curve=base.carry_curve,
            model=cell.model,
            mode=SurfaceShockMode(cell.mode),
            engine_factory=mod._engine_factory_from_settings(
                dict(base.engine_settings)
            ),
            dsigma=cell.dsigma,
        )
        assert outcome.value == float(direct.pnl)
        assert economics_mapping(outcome) == dict(
            surface_shock_economics(direct)
        )


def test_cartesian_combination_the_solution_could_not_express():
    """heston x recalibrate x tenor bucket has no name in the solution's
    grammar; the typed cell prices it through the same runner."""
    tenor = _fixture_tenors()[0]
    cell = SurfaceShockCell(
        "heston", "recalibrate", 0.005,
        tenor_bucket=(tenor - 1e-6, tenor + 1e-6),
    )
    specs = cells_to_scenario_specs([cell])
    with PricingSession() as session:
        outcomes = session.run_scenarios(_base_ref(), specs, None)
    outcome = outcomes[0]
    assert isinstance(outcome, ScenarioOutcome)
    economics = economics_mapping(outcome)
    assert economics["model"] == "heston"
    assert economics["shock.tenor_bucket"] is not None
    assert economics["numerical.calibration.shocked.success"] is True


# ------------------------------------- the Phase 5 process equality gate
def test_process_cells_match_serial_complete_payload():
    """Reduced menu covering every LV transformer path (frozen,
    recalibrate, tenor bucket, moneyness bucket); the full 13-cell run
    lives in benchmark_phase5.py."""
    tenor = _fixture_tenors()[0]
    cells = (
        SurfaceShockCell("local_vol", "frozen", 0.005),
        SurfaceShockCell("local_vol", "recalibrate", 0.005),
        SurfaceShockCell(
            "local_vol", "recalibrate", 0.005,
            tenor_bucket=(tenor - 1e-6, tenor + 1e-6),
        ),
        SurfaceShockCell(
            "local_vol", "recalibrate", 0.005,
            moneyness_bucket=SOLUTION_MONEYNESS_BUCKETS[1],
        ),
    )
    specs = cells_to_scenario_specs(cells)
    with PricingSession() as session:
        serial = session.run_scenarios(_base_ref(), specs, None)
    with PricingSession(_process_context()) as session:
        via_processes = session.run_scenarios(_base_ref(), specs, None)

    report = compare_scenario_outcomes(serial, via_processes)
    assert report.all_scenarios_match is True, report.first_mismatch_path
    assert report.scenarios_compared == 4
    # field counts reported separately from scenario counts (spec 2)
    assert report.fields_compared > 4 * 10
    assert report.missing_fields == ()
    assert report.extra_fields == ()


def test_process_fault_isolation_on_real_cells():
    """A poisoned Heston gate fails ITS cell; LV cells still match their
    serial economics exactly."""
    cells = (
        SurfaceShockCell("local_vol", "frozen", 0.005),
        SurfaceShockCell("heston", "recalibrate", 0.005),
        SurfaceShockCell("local_vol", "recalibrate", 0.005),
    )
    specs = cells_to_scenario_specs(cells)
    poisoned = _base_ref(max_calibration_rmse_iv=0.0)
    with PricingSession() as session:
        serial = session.run_scenarios(
            poisoned, specs, None, collect_errors=True
        )
    with PricingSession(_process_context()) as session:
        via_processes = session.run_scenarios(
            poisoned, specs, None, collect_errors=True
        )
    assert isinstance(via_processes[1], PricingFailure)
    assert via_processes[1].error.error_type == "NumericalError"
    for index in (0, 2):
        assert isinstance(via_processes[index], ScenarioOutcome)
        assert economics_mapping(via_processes[index]) == economics_mapping(
            serial[index]
        )
