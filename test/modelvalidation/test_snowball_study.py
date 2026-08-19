"""Smoke test for the snowball demo study over the real PDE and QUAD engines.

This asserts *soundness, not outcome*. At the tiny sampling budget used here the
benchmark cannot meet its standard-error budget, so INCONCLUSIVE is the correct
result; asserting ADMITTED would be asserting that noise agrees with us. What
must hold is that every arm runs, produces finite numbers, and gets gated.

The real verdict comes from the offline acceptance run at full sampling, whose
evidence is banked -- see docs/modelvalidation/RELEASE_PROCEDURE.md.
"""

import dataclasses
from pathlib import Path

import pytest

from quantark.modelvalidation.candidate import envelope_from_ladders
from quantark.modelvalidation.pipeline import certify
from quantark.modelvalidation.study import CaseSpec
from quantark.modelvalidation.yaml_loader import load_study

STUDY_PATH = Path("example/modelvalidation/snowball_flat_bsm.yaml")

CANDIDATES = ("equity.snowball.pde", "equity.snowball.quad")


@pytest.fixture(scope="module")
def study():
    return load_study(STUDY_PATH)


@pytest.fixture(scope="module")
def small_study(study):
    """Two cases, a few thousand paths: enough to prove the wiring."""
    return dataclasses.replace(
        study,
        cases=study.cases[:2],
        sampling=dataclasses.replace(
            study.sampling, paths_per_batch=2048, min_batches=2, max_batches=3
        ),
    )


@pytest.fixture(scope="module")
def certified(tmp_path_factory, small_study):
    out = tmp_path_factory.mktemp("snowball")
    return certify(small_study, out_dir=out)


def test_study_file_loads_with_both_engines(study):
    assert study.name == "snowball-flat-bsm"
    assert tuple(c.name() for c in study.candidates) == CANDIDATES
    assert [case.name for case in study.cases] == [
        # Market scenarios, from the original certification.
        "ordinary",
        "near_ko",
        "near_ki",
        "low_vol",
        "near_expiry",
        # Product variants, added by amendment. Each is a distinct engine code
        # path, so dropping one silently narrows what the study certifies.
        "discrete_ki",
        "european_ki",
        "stepdown_ko",
        "stepdown_near_last_ko",
        "parachute",
        "parachute_near_ki",
    ]


def test_both_engines_run_without_errors(certified):
    """The machinery must exercise the real engines cleanly."""
    assert not [cell for cell in certified.payload["cells"] if cell["verdict"] == "ERROR"]


def test_every_candidate_reaches_a_decision(certified):
    decisions = certified.payload["decisions"]
    assert set(decisions) == set(CANDIDATES)
    for candidate, decision in decisions.items():
        assert decision in ("ADMITTED", "REJECTED", "INCONCLUSIVE"), candidate


def test_cells_carry_finite_gate_numbers(certified):
    for cell in certified.payload["cells"]:
        gate = cell["gate"]
        assert gate is not None
        assert gate["interval_c"] == gate["interval_c"]  # not NaN
        assert gate["se_c"] >= 0.0
        assert cell["candidate_value"] is not None


def test_both_engines_share_one_benchmark(certified):
    """One reference bank serves both candidates -- that is why it excludes them."""
    references = certified.payload["references"]
    assert set(references) == {"ordinary", "near_ko"}
    for block in references.values():
        assert block["batches"] >= 2


def test_grid_engines_report_a_refinement_envelope(certified):
    """Unlike a closed-form engine, these must bound their own discretization."""
    for cell in certified.payload["cells"]:
        assert cell["gate"]["envelope_c"] is not None


def test_envelope_helper_agrees_with_the_banked_numbers(small_study):
    """The ladder machinery is wired to the engines, not just to the schema."""
    candidate = small_study.candidates[0]
    result = candidate.evaluate(CaseSpec(name="ordinary"))
    assert len(result.ladders) == 2
    envelope = envelope_from_ladders(result.ladders, "delta")
    assert envelope is not None and envelope >= 0.0


def test_report_is_written(certified):
    report = (certified.path.parent / "report.md").read_text(encoding="utf-8")
    assert "snowball-flat-bsm" in report
    for candidate in CANDIDATES:
        assert candidate in report


def test_pde_records_the_grid_its_profile_resolves_to(study):
    """`accuracy: standard` is an indirection; the evidence must record what it means."""
    pde = next(c for c in study.candidates if c.name() == "equity.snowball.pde")
    params = pde.params()
    assert params["accuracy"] == "standard"
    grid = params["grid"]
    assert grid["points"] == 400
    assert grid["steps_per_day"] == 4.0
    assert grid["max_steps"] == 5000
    assert grid["day_count"] == 252


def test_quad_records_its_full_numerical_configuration(study):
    quad = next(c for c in study.candidates if c.name() == "equity.snowball.quad")
    grid = quad.params()["grid"]
    assert grid["grid_points"] == 1001
    assert grid["num_std_devs"] == 10.0
    assert grid["integration_rule"] == "trapezoid"
    assert grid["event_projection"] == "cell_average"
    # Convergence knobs cannot move a fixed-grid answer, so they stay out.
    assert "auto_converge" not in grid


def test_grid_settings_reach_both_reports(certified):
    directory = certified.path.parent
    markdown = (directory / "report.md").read_text(encoding="utf-8")
    # The HTML marks soft break points inside long setting names; compare the
    # text a reader sees, not the markup.
    html = (directory / "report.html").read_text(encoding="utf-8").replace("<wbr>", "")
    for text in (markdown, html):
        assert "grid.points" in text
        assert "grid.grid_points" in text
        assert "400" in text and "1001" in text


def test_benchmark_configuration_is_recorded(certified):
    config = certified.payload["reference_config"]
    assert config["engine"] == "SnowballMCEngine"
    assert config["method"] == "randomized_quasi"
    assert "paths_per_batch" in config
    html = (certified.path.parent / "report.html").read_text(encoding="utf-8")
    assert "SnowballMCEngine" in html


def test_changing_the_grid_changes_the_candidate_identity(study):
    """Otherwise a re-tuned engine could silently reuse a stale checkpoint."""
    from quantark.modelvalidation.candidate import candidate_identity
    from quantark.modelvalidation.evidence import identity_hash
    from quantark.modelvalidation.registry import get_builder

    build = get_builder("equity.snowball.quad", kind="candidate")
    common = dict(
        environment_params={"spot": 100.0, "vol": 0.22, "rate": 0.025, "div_yield": 0.03},
        product_params={},
        quantities=("pv",),
    )
    coarse = build(params={"grid_points": 501}, **common)
    fine = build(params={"grid_points": 1001}, **common)
    case = CaseSpec(name="ordinary")
    assert identity_hash(candidate_identity(coarse, case)) != identity_hash(
        candidate_identity(fine, case)
    )
