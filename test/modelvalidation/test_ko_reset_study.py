"""Smoke test for the KO-reset demo study over the real PDE and QUAD engines.

Asserts soundness, not outcome: at the tiny sampling budget used here the
benchmark cannot meet its standard-error budget, so INCONCLUSIVE is correct.
The real verdict comes from the offline acceptance run, whose evidence is
banked -- see docs/modelvalidation/RELEASE_PROCEDURE.md.
"""

import dataclasses
from pathlib import Path

import pytest

from quantark.modelvalidation.candidate import envelope_from_ladders
from quantark.modelvalidation.pipeline import certify
from quantark.modelvalidation.study import CaseSpec
from quantark.modelvalidation.yaml_loader import load_study

STUDY_PATH = Path("example/modelvalidation/ko_reset_flat_bsm.yaml")

CANDIDATES = ("equity.ko_reset_snowball.pde", "equity.ko_reset_snowball.quad")


@pytest.fixture(scope="module")
def study():
    return load_study(STUDY_PATH)


@pytest.fixture(scope="module")
def small_study(study):
    return dataclasses.replace(
        study,
        cases=study.cases[:2],
        sampling=dataclasses.replace(
            study.sampling, paths_per_batch=2048, min_batches=2, max_batches=3
        ),
    )


@pytest.fixture(scope="module")
def certified(tmp_path_factory, small_study):
    return certify(small_study, out_dir=tmp_path_factory.mktemp("ko_reset"))


def test_study_file_loads_with_both_engines(study):
    assert study.name == "ko-reset-flat-bsm"
    assert tuple(c.name() for c in study.candidates) == CANDIDATES
    assert [case.name for case in study.cases] == [
        # Market scenarios, from the original certification.
        "ordinary",
        "near_pre_ko",
        "near_ki",
        "below_ki",
        "low_vol",
        "near_expiry",
        "discrete_ki",
        # Barrier-shape and KI-monitoring variants (first amendment).
        "european_ki",
        "stepdown",
        "stepdown_near_last_pre_ko",
        "parachute",
        "parachute_near_ki",
        # The remaining product feature surface (second amendment).
        "ki_stepdown",
        "disable_ko_after_ki",
    ]


def test_both_engines_run_without_errors(certified):
    assert not [c for c in certified.payload["cells"] if c["verdict"] == "ERROR"]


def test_every_candidate_reaches_a_decision(certified):
    decisions = certified.payload["decisions"]
    assert set(decisions) == set(CANDIDATES)
    for candidate, decision in decisions.items():
        assert decision in ("ADMITTED", "REJECTED", "INCONCLUSIVE"), candidate


def test_cells_carry_finite_gate_numbers(certified):
    for cell in certified.payload["cells"]:
        gate = cell["gate"]
        assert gate is not None
        assert gate["interval_c"] == gate["interval_c"]
        assert gate["se_c"] >= 0.0
        assert cell["candidate_value"] is not None


def test_grid_engines_report_a_refinement_envelope(certified):
    for cell in certified.payload["cells"]:
        assert cell["gate"]["envelope_c"] is not None


def test_envelope_helper_agrees_with_the_banked_numbers(small_study):
    result = small_study.candidates[0].evaluate(CaseSpec(name="ordinary"))
    assert len(result.ladders) == 2
    envelope = envelope_from_ladders(result.ladders, "delta")
    assert envelope is not None and envelope >= 0.0


def test_the_below_ki_case_is_actually_knocked_in(study):
    """That case exists to exercise the post-KI regime; if spot sat above the
    KI barrier it would silently be a second `ordinary`."""
    from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset

    below = next(c for c in study.cases if c.name == "below_ki")
    product = make_ko_reset(study.candidates[0].product_params)
    assert below.environment_params["spot"] < product.barrier_config.ki_barrier


def test_the_discrete_ki_case_actually_flips_the_monitoring(study):
    """Continuous KI runs the Brownian-bridge path; discrete KI does not."""
    from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset

    base = make_ko_reset(study.candidates[0].product_params)
    assert base.barrier_config.ki_continuous is True

    case = next(c for c in study.cases if c.name == "discrete_ki")
    assert case.product_params == {"ki_continuous": False}
    flipped = make_ko_reset({**study.candidates[0].product_params, **case.product_params})
    assert flipped.barrier_config.ki_continuous is False


def test_report_is_written(certified):
    report = (certified.path.parent / "report.md").read_text(encoding="utf-8")
    assert "ko-reset-flat-bsm" in report
    for candidate in CANDIDATES:
        assert candidate in report


def test_pde_records_the_grid_its_profile_resolves_to(study):
    pde = next(c for c in study.candidates if c.name() == "equity.ko_reset_snowball.pde")
    params = pde.params()
    assert params["accuracy"] == "standard"
    assert params["grid"]["points"] == 400


def test_quad_records_its_full_numerical_configuration(study):
    grid = next(
        c for c in study.candidates if c.name() == "equity.ko_reset_snowball.quad"
    ).params()["grid"]
    assert grid["grid_points"] == 1001
    assert grid["integration_rule"] == "trapezoid"
    assert "auto_converge" not in grid
