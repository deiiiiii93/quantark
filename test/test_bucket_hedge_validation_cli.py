"""The validation stage runs offline, reproduces itself, and refuses to lie.

The three properties under test: it needs no vendor history, a serialised
snapshot reproduces the same numbers, and a deliberately wrong bucket or a
too-coarse ladder produces a failure or an inconclusive rather than a
silently widened tolerance.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
STUDY = REPO / "example" / "snowball_q_term_structure"


def _load(name: str, filename: str):
    if str(STUDY) not in sys.path:
        sys.path.insert(0, str(STUDY))
    spec = importlib.util.spec_from_file_location(name, STUDY / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V = _load("validation_stage", "05_bucket_hedge_validation.py")
ORACLES = _load("_bucket_oracles", "_bucket_oracles.py")


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("validation")
    code = V.main(["--synthetic", "--out-dir", str(out)])
    return code, out


# ---------------------------------------------------------------------------
# It runs offline and writes everything
# ---------------------------------------------------------------------------


def test_the_synthetic_run_needs_no_vendor_history(synthetic_run):
    code, out = synthetic_run
    assert code == 0
    for name in (
        "validation_manifest.json",
        "input_snapshots.json",
        "price_ladder.csv",
        "greek_ladder.csv",
        "policy_holdings.csv",
        "direct_audits.csv",
        "stress_results.csv",
        "validation_summary.md",
    ):
        assert (out / name).exists(), name


def test_the_manifest_records_identity_coverage_and_the_exact_command(synthetic_run):
    _, out = synthetic_run
    manifest = json.loads((out / "validation_manifest.json").read_text())
    assert manifest["command"].startswith("python 05_bucket_hedge_validation.py")
    assert "--synthetic" in manifest["command"]
    assert len(manifest["source_digest"]) == 16
    assert manifest["coverage"]["cases"] == 7
    assert manifest["coverage"]["failed"] == 0
    assert manifest["coverage"]["inconclusive"] == 0
    assert manifest["tolerances"]["delta_hands"] == 0.01
    assert manifest["ladders"]["spot_bumps"]
    assert manifest["data_blocker"] is None


def test_every_required_case_reports_a_status(synthetic_run):
    _, out = synthetic_run
    manifest = json.loads((out / "validation_manifest.json").read_text())
    names = {c["case"] for c in manifest["status_by_case"]}
    assert names == {
        "linear_book",
        "forward_claim",
        "early_digital",
        "policy_residuals",
        "carry_curvature",
        "unquoted_scenarios",
        "spot_ladder",
    }
    assert all(c["status"] == "pass" for c in manifest["status_by_case"])


def test_the_summary_is_readable_and_names_its_reproduction_command(synthetic_run):
    _, out = synthetic_run
    text = (out / "validation_summary.md").read_text()
    assert "# Bucket futures hedge" in text
    assert "Reproduce with" in text
    assert "backed by a direct measurement" in text
    assert "7/7 required cases passed" in text


# ---------------------------------------------------------------------------
# The measurements themselves
# ---------------------------------------------------------------------------


def test_every_pass_is_backed_by_a_direct_measurement(synthetic_run):
    _, out = synthetic_run
    audits = pd.read_csv(out / "direct_audits.csv")
    assert not audits.empty
    assert set(audits["objective"]) == {"nodes", "spot_far", "spot_parallel"}
    assert set(audits["convention"]) == {"flat_q", "flat_forward_carry"}
    assert (audits["audit_status"] == "pass").all()
    assert audits["net_delta_audit_error_hands"].abs().max() < 0.01
    assert audits["parallel_rhoq_audit_error_bp"].abs().max() < 0.01
    # The residual table: nodes keeps D_F, the other two are spot neutral.
    nodes = audits[audits["objective"] == "nodes"]
    assert nodes["direct_net_delta_hands"].abs().min() > 0.0
    for objective in ("spot_far", "spot_parallel"):
        rows = audits[audits["objective"] == objective]
        assert rows["direct_net_delta_hands"].abs().max() < 1e-9


def test_the_tail_stress_reproduces_the_designs_49_8752_bp(synthetic_run):
    _, out = synthetic_run
    stresses = pd.read_csv(out / "stress_results.csv")
    tail_up = stresses[stresses["scenario_id"].str.startswith("tail+")].iloc[0]
    assert tail_up["product_pnl_bp_of_base"] == pytest.approx(-49.8752, abs=1e-3)
    # No futures hedge can respond to it, however the book is positioned.
    assert (stresses["hedge_pnl"] == 0.0).all()
    assert (stresses["book_pnl"] == stresses["product_pnl"]).all()


def test_the_shape_stress_moves_something(synthetic_run):
    _, out = synthetic_run
    stresses = pd.read_csv(out / "stress_results.csv")
    shape = stresses[stresses["scenario_id"].str.startswith("shape+")].iloc[0]
    # A claim at the peak of the displacement moves by exactly expm1(eps).
    assert shape["product_pnl_bp_of_base"] == pytest.approx(
        1e4 * math.expm1(0.01), rel=1e-9
    )
    assert shape["product_pnl"] != 0.0
    assert shape["hedge_pnl"] == 0.0


def test_the_29_30_fraction_is_measured_under_both_conventions(synthetic_run):
    _, out = synthetic_run
    ladder = pd.read_csv(out / "greek_ladder.csv")
    digital = ladder[ladder["case"] == "early_digital"]
    assert set(digital["convention"]) == {"flat_q", "flat_forward_carry"}
    assert digital["expected_fraction"].iloc[0] == pytest.approx(29.0 / 30.0)
    assert digital["abs_error"].max() < 1e-4


def test_the_ladder_records_effective_bumps_not_just_a_pass(synthetic_run):
    _, out = synthetic_run
    ladder = pd.read_csv(out / "greek_ladder.csv")
    assert "futures_bump_points" in ladder.columns
    spot = ladder[ladder["case"] == "spot_ladder"]
    assert "spot_step" in spot.columns
    assert len(spot) >= 3
    # Halving, level by level.
    steps = list(spot["spot_step"])
    assert all(
        steps[i + 1] == pytest.approx(steps[i] / 2.0) for i in range(len(steps) - 1)
    )


def test_the_policy_holdings_are_written_per_objective(synthetic_run):
    _, out = synthetic_run
    holdings = pd.read_csv(out / "policy_holdings.csv")
    assert set(holdings["objective"]) == {"nodes", "spot_far", "spot_parallel"}
    parallel = holdings[
        (holdings["objective"] == "spot_parallel")
        & (holdings["convention"] == "flat_q")
    ]
    # The two-tenor fold leaves +K and -K on the chosen pair and nothing else.
    assert len(parallel) == 3
    assert parallel["mapped_net_rhoq"].sum() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_a_snapshot_replays_to_the_same_numbers(tmp_path, synthetic_run):
    _, first = synthetic_run
    snapshot = first / "input_snapshots.json"
    second = tmp_path / "replayed"
    assert V.main(["--snapshot", str(snapshot), "--out-dir", str(second)]) == 0
    for name in ("greek_ladder.csv", "direct_audits.csv", "stress_results.csv"):
        original = pd.read_csv(first / name)
        replayed = pd.read_csv(second / name)
        pd.testing.assert_frame_equal(original, replayed)


def test_the_snapshot_carries_the_exact_inputs(synthetic_run):
    _, out = synthetic_run
    payload = json.loads((out / "input_snapshots.json").read_text())
    snapshot = payload["snapshots"][0]
    assert snapshot["mode"] == "synthetic"
    assert snapshot["valuation"] == "2025-03-03"
    assert snapshot["spot"] > 0
    assert len(snapshot["quotes"]) == 3
    assert {"contract", "tenor_years", "price", "multiplier"} <= set(
        snapshot["quotes"][0]
    )


# ---------------------------------------------------------------------------
# It refuses to pass what it cannot back
# ---------------------------------------------------------------------------


def test_a_deliberately_wrong_bucket_fails_the_case(monkeypatch, tmp_path):
    """Perturbing the sampler must break the audit, not widen a tolerance."""
    from dataclasses import replace as dc_replace

    original = V.buckets_of

    def perturbed(context, samples):
        buckets = list(original(context, samples))
        buckets[0] = dc_replace(
            buckets[0], bucket_currency=buckets[0].bucket_currency * 1.5
        )
        return tuple(buckets)

    monkeypatch.setattr(V, "buckets_of", perturbed)
    out = tmp_path / "wrong"
    code = V.main(["--synthetic", "--out-dir", str(out)])
    assert code == 1
    manifest = json.loads((out / "validation_manifest.json").read_text())
    assert manifest["coverage"]["failed"] > 0
    statuses = {c["case"]: c["status"] for c in manifest["status_by_case"]}
    assert statuses["policy_residuals"] == "fail"
    # The tolerances in the manifest are the DECLARED ones, untouched.
    assert manifest["tolerances"]["delta_hands"] == 0.01


def test_a_ladder_that_cannot_settle_is_inconclusive(monkeypatch, tmp_path):
    """A first-order estimator never stabilises; that is not a pass."""
    calls = {"n": 0}

    def drifting(price_at, context, spot_step):
        calls["n"] += 1
        return 1.0 + spot_step

    monkeypatch.setattr(V, "direct_pinned_delta", drifting)
    out = tmp_path / "unsettled"
    code = V.main(["--synthetic", "--out-dir", str(out)])
    assert code == 1
    manifest = json.loads((out / "validation_manifest.json").read_text())
    statuses = {c["case"]: c["status"] for c in manifest["status_by_case"]}
    assert statuses["spot_ladder"] == "inconclusive"
    assert manifest["coverage"]["inconclusive"] >= 1
    # Every sample it took is still on disk.
    ladder = pd.read_csv(out / "greek_ladder.csv")
    kept = ladder[ladder["case"] == "spot_ladder"]
    assert len(kept) == V.MAX_HALVINGS
    assert calls["n"] >= V.MAX_HALVINGS


def test_missing_history_is_a_data_blocker_not_a_failure(tmp_path):
    out = tmp_path / "blocked"
    code = V.main(
        [
            "--historical-dates",
            "2025-03-03",
            "--history-dir",
            str(tmp_path / "nowhere"),
            "--out-dir",
            str(out),
        ]
    )
    assert code == 2
    manifest = json.loads((out / "validation_manifest.json").read_text())
    assert manifest["data_blocker"]
    # No case is reported at all: it is neither a pass nor a numerical failure.
    assert manifest["status_by_case"] == []
    assert manifest["coverage"]["cases"] == 0
    assert "Data blocker" in (out / "validation_summary.md").read_text()


def test_the_cli_modes_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        V.parse_args(
            ["--synthetic", "--historical-dates", "2025-03-03", "--out-dir", str(tmp_path)]
        )


def test_the_stage_never_imports_the_test_package():
    source = (STUDY / "05_bucket_hedge_validation.py").read_text()
    assert "bucket_hedge_fixtures" not in source
    assert "model-validation-output" not in source
    assert "import test" not in source
    # It uses the shipped oracles, which import nothing from the library.
    oracle_source = (STUDY / "_bucket_oracles.py").read_text()
    imports = [
        line.strip()
        for line in oracle_source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    assert imports == [
        "from __future__ import annotations",
        "import math",
        "from dataclasses import dataclass",
        "from typing import Sequence, Tuple",
    ], imports
    assert "_bucket_oracles" in source


def test_the_oracles_and_the_test_fixtures_are_the_same_numbers():
    """The tests and the shipped stage check against ONE derivation.

    Object identity would depend on which module loaded the file first, so
    the assertion is on the source file and the numbers it produces.
    """
    import bucket_hedge_fixtures as fixtures

    assert Path(fixtures._ORACLES.__file__) == STUDY / "_bucket_oracles.py"

    def fields(quotes):
        return [(q.contract, q.tenor_years, q.price, q.multiplier) for q in quotes]

    # The dataclasses come from two module OBJECTS over the same file, so
    # compare the values they carry rather than the instances.
    assert fields(fixtures.LINEAR_QUOTES) == fields(ORACLES.LINEAR_QUOTES)
    from_fixtures = fixtures.linear_book_risk()
    from_stage = ORACLES.linear_book_risk()
    assert from_fixtures.delta_q == from_stage.delta_q
    assert from_fixtures.delta_f == from_stage.delta_f
    assert from_fixtures.buckets == from_stage.buckets
    assert from_fixtures.nodal_rhoq == from_stage.nodal_rhoq
    assert fixtures.early_digital_price(
        4700.0, 4650.0, 0.235, 1 / 365, 0.98, 5e7
    ) == ORACLES.early_digital_price(4700.0, 4650.0, 0.235, 1 / 365, 0.98, 5e7)
