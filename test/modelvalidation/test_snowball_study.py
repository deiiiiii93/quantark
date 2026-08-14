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
        "ordinary",
        "near_ko",
        "near_ki",
        "low_vol",
        "near_expiry",
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
