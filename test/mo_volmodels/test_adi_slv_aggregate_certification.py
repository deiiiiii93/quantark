import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "example/mo_volmodels/17_adi_slv_aggregate_certification.py"
# The stage-16 parent this amendment was built on, and the amendment itself.
#
# These are the raw payloads: 14.4 MB of Monte-Carlo row dumps. They are
# artifacts rather than source, so this repository does not carry them -- it
# publishes code, tests and research docs. What IS published is their
# identity: the modelvalidation certificate that archives this certification
# records every digest under `imported.source_digests`, so a payload can be
# proved to be the one certified and cannot be silently substituted.
#
# The tests below therefore split in two. Everything that can be checked from
# the published record runs everywhere. Everything that needs the rows
# themselves is marked `requires_banked_evidence` and skips with a reason when
# they are absent -- which is the honest outcome, not a silent pass. Drop the
# payloads into the directory below to run the full set.
BANKED_EVIDENCE = (
    ROOT
    / "docs/modelvalidation/certificates/adi2d-snowball-greeks/2026-08-19/evidence"
)
PARENT_EVIDENCE_FILE = BANKED_EVIDENCE / "stage16_greek_certification.json"
PARENT_DECISION_FILE = BANKED_EVIDENCE / "stage16_decision.json"
#: The amendment this module produced, archived beside its parent.
BANKED_AMENDMENT_FILE = BANKED_EVIDENCE / "stage17_slv_aggregate_amendment.json"

requires_banked_evidence = pytest.mark.skipif(
    not (
        PARENT_EVIDENCE_FILE.is_file()
        and PARENT_DECISION_FILE.is_file()
        and BANKED_AMENDMENT_FILE.is_file()
    ),
    reason=(
        "raw stage-16/17 payloads are kept out of the repository as artifacts; "
        f"place them under {BANKED_EVIDENCE.relative_to(ROOT)} to run this. "
        "Their digests are published in the modelvalidation certificate."
    ),
)


def _load():
    name = "mo_adi_slv_aggregate_certification_17"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_schema12_parent_and_development_families_are_pinned():
    module = _load()

    assert module.SCHEMA_VERSION == 12
    assert module.PARENT_SCHEMA_VERSION == 13
    assert module.DEVELOPMENT_SEED == 20260806
    assert module.DEVELOPMENT_PATHS_PER_BATCH == 8192
    assert module.DEVELOPMENT_BATCHES == 16
    assert module.DEVELOPMENT_PRIMARY_BATCH_GRID == (512, 1024, 2048, 4096)
    assert module.DEVELOPMENT_MIDDLE_BATCH_GRID == (32, 64, 128, 256, 512, 1024)
    assert module.DEVELOPMENT_HESTON_WEIGHT_GRID == tuple(
        index / 10.0 for index in range(11)
    )
    assert module.DEVELOPMENT_NEW_FAMILY_GUARD_SIGMAS == 1.0
    assert module.DEVELOPMENT_VARIANCE_UPPER_CONFIDENCE == 0.95
    assert "near_ki" not in module.CONTROL_CASES
    assert set(module.CONTROL_CASES) == {
        "ordinary_full",
        "ordinary_decayed",
        "near_ko",
        "low_feller",
        "sigma_collapse",
        "near_expiry",
    }
    assert module.PRODUCTION_PRIMARY_SEED == 20260811
    assert module.PRODUCTION_MIDDLE_SEED == 20260812
    assert module.PRODUCTION_ALLOCATION_FROZEN is True
    # Re-frozen at 0f13d93 from fresh pilots against the strided parent. The
    # original freeze (4096 / 256 / weight 0.7 / 67,108,864 paths, designed at
    # b5a5243) was proven STALE: reconstruction against its own pinned hashes
    # showed it had measured pre-bridge8 estimators on ordinary_full,
    # ordinary_decayed and sigma_collapse, because the spot-bridge dimension
    # went 1 -> 8 after those pilots ran. low_feller, whose configuration did
    # not change, reproduced bit-for-bit -- which is what made the diagnosis
    # sound rather than a guess. The new plan is 8x smaller.
    assert module.PRODUCTION_PRIMARY_BATCHES == 512
    assert module.PRODUCTION_MIDDLE_BATCHES == 32
    assert module.PRODUCTION_PRIMARY_BATCH_WORKERS == 4
    assert module.PRODUCTION_MIDDLE_BATCH_WORKERS == 2
    assert module.PRODUCTION_MIDDLE_CELL_WORKERS == 2
    assert module.FROZEN_SMOOTH_HESTON_WEIGHT == 1.0
    assert module.FROZEN_ALLOCATION_TOTAL_UNIQUE_PATHS == 8_388_608
    assert module.FROZEN_ALLOCATION_PROJECTION_SHA256 == (
        "f2a37465317e114f93f6c6c2f1c6f3674275549597c3c8d74077bd8eb3c2733d"
    )
    assert module.FROZEN_ALLOCATION_PROJECTION_FILE_SHA256 == (
        "59b81f6ae0bd7c8d22e8709f4a6ab3c210e263fcbcc15597359bc80975e629b3"
    )
    assert module.FROZEN_ALLOCATION_GUARDED_INTERVAL == pytest.approx(
        (-0.09232497576315679, 0.0025153060199847935)
    )
    assert module.AGGREGATE_CONTROL_WEIGHTS["low_feller"]["heston"] == 0.0
    assert module.AGGREGATE_CONTROL_WEIGHTS["ordinary_full"]["heston"] == 1.0
    manifest = module.frozen_allocation_manifest()
    assert manifest["design_commit"] == (
        "acd9f2feb2d86bb8360fb799209c86a55d753280"
    )
    assert manifest["recommendation"]["primary_batches"] == 512
    assert manifest["recommendation"]["middle_batches"] == 32


def _accept_live_production_pde(module, monkeypatch):
    """Let a test past the parent's production-PDE compatibility guard.

    The guard hashes whole source trees (``quantark/asset/equity/engine/pde``,
    ``quantark/volmodels``, ``quantark/asset/equity/param``, ...), so ANY edit
    under them moves it -- including edits that provably cannot touch these
    cells. It has fired since main's continuous-KI work landed; see
    ``test_the_parent_no_longer_accepts_this_tree_for_new_pde_work``, which is
    where that fact is asserted. Tests about recomposition ARITHMETIC should
    not also be re-testing compatibility, so they opt past it explicitly.
    """
    monkeypatch.setattr(
        module,
        "PARENT_PRODUCTION_PDE_SHA256",
        module.stage16().production_pde_compatibility_sha256(),
    )


@requires_banked_evidence
def test_the_banked_parent_is_intact_and_every_cell_passed():
    """Pure evidence assertions: no live engine is consulted, so this keeps
    holding as the tree moves underneath it."""
    evidence = json.loads(PARENT_EVIDENCE_FILE.read_text())
    decision = json.loads(PARENT_DECISION_FILE.read_text())
    module = _load()

    assert evidence["evidence_sha256"] == module.PARENT_EVIDENCE_SHA256
    assert decision["decision_sha256"] == module.PARENT_DECISION_SHA256
    assert len(evidence["cells"]) == 14
    assert {cell["status"] for cell in evidence["cells"]} == {"PASS"}
    assert evidence["decisions"]["heston"]["route"] == "pde"
    # The parent left SLV unresolved on purpose -- that is the precondition the
    # stage-17 aggregate amendment exists to remove.
    assert evidence["decisions"]["heston_slv"]["route"] == (
        "excluded_greek_unresolved"
    )


@requires_banked_evidence
def test_the_parent_no_longer_accepts_this_tree_for_new_pde_work():
    """The parent's production-PDE guard fires, and that is correct.

    Main's continuous-KI work (62f72b6, c6b8401, 7baf042) edited files inside
    the guarded source trees, so the digest moved and the parent will not carry
    its completed PDE evidence into a NEW amendment from this tree. Nothing
    already published is affected: the amendment that used this parent is
    banked, and every one of the parent's fourteen PDE cells was re-measured on
    the merged tree and reproduced BIT-FOR-BIT
    (``docs/adi2d-greek-perf/probes/probe_merge_ki_invariance.py``).

    The guard is a source-set proxy for a behavioural question that has since
    been answered directly. Anyone re-pinning it must carry that measurement as
    the justification -- which is why this asserts the refusal rather than
    quietly widening it.
    """
    module = _load()
    live = module.stage16().production_pde_compatibility_sha256()
    assert live != module.PARENT_PRODUCTION_PDE_SHA256

    with pytest.raises(ValueError, match="production PDE differs"):
        module.load_and_validate_parent_certificate(
            PARENT_EVIDENCE_FILE,
            PARENT_DECISION_FILE,
        )


@requires_banked_evidence
def test_the_parent_loads_when_the_production_pde_matches(monkeypatch):
    module = _load()
    _accept_live_production_pde(module, monkeypatch)

    evidence, decision, manifest = module.load_and_validate_parent_certificate(
        PARENT_EVIDENCE_FILE,
        PARENT_DECISION_FILE,
    )

    assert manifest == module.parent_certificate_manifest()
    assert evidence["evidence_sha256"] == module.PARENT_EVIDENCE_SHA256
    assert decision["decision_sha256"] == module.PARENT_DECISION_SHA256


@requires_banked_evidence
def test_schema11_parent_loader_rejects_reformatted_bytes(tmp_path):
    module = _load()
    source = PARENT_EVIDENCE_FILE
    payload = json.loads(source.read_text())
    reformatted = tmp_path / source.name
    reformatted.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="evidence file hash mismatch"):
        module.load_and_validate_parent_certificate(
            reformatted,
            PARENT_DECISION_FILE,
        )


def test_schema12_checkpoint_round_trip_and_configuration_lock(tmp_path):
    module = _load()
    evidence = {"case": "ordinary_full", "rows": [1.0, 2.0]}
    module._write_checkpoint(
        tmp_path,
        "primary__ordinary_full",
        run_configuration_sha256="a" * 64,
        kind="primary_reference",
        evidence=evidence,
    )

    assert module._load_checkpoint(
        tmp_path,
        "primary__ordinary_full",
        run_configuration_sha256="a" * 64,
        kind="primary_reference",
    ) == evidence
    with pytest.raises(ValueError, match="provenance mismatch"):
        module._load_checkpoint(
            tmp_path,
            "primary__ordinary_full",
            run_configuration_sha256="b" * 64,
            kind="primary_reference",
        )


def test_schema12_checkpoint_rejects_resealed_evidence_tampering(tmp_path):
    module = _load()
    module._write_checkpoint(
        tmp_path,
        "middle__near_ko",
        run_configuration_sha256="c" * 64,
        kind="middle_control",
        evidence={"value": 1},
    )
    path = tmp_path / "checkpoints/middle__near_ko.json"
    record = json.loads(path.read_text())
    record["evidence"]["value"] = 2
    path.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="provenance mismatch"):
        module._load_checkpoint(
            tmp_path,
            "middle__near_ko",
            run_configuration_sha256="c" * 64,
            kind="middle_control",
        )


def test_level_checkpoints_merge_only_when_metadata_matches():
    module = _load()
    target = {"case": "ordinary_full", "seed": 7, "target": {"rows": [1]}}
    fine = {"case": "ordinary_full", "seed": 7, "fine": {"rows": [2]}}

    assert module.merge_added_reference_levels(target, fine) == {
        "case": "ordinary_full",
        "seed": 7,
        "target": {"rows": [1]},
        "fine": {"rows": [2]},
    }
    mismatched = copy.deepcopy(fine)
    mismatched["seed"] = 8
    with pytest.raises(ValueError, match="metadata do not match"):
        module.merge_added_reference_levels(target, mismatched)


def test_independent_cohort_summary_uses_sum_of_means_and_floored_welch_df():
    module = _load()
    left = np.array([1.0, 2.0, 3.0, 4.0])
    right = np.array([-2.0, 0.0, 2.0, 4.0, 6.0])

    summary = module.summarize_independent_cohorts(
        [left, right], confidence=0.975
    )

    left_se = np.std(left, ddof=1) / np.sqrt(left.size)
    right_se = np.std(right, ddof=1) / np.sqrt(right.size)
    total_variance = left_se**2 + right_se**2
    raw_df = total_variance**2 / (
        left_se**4 / (left.size - 1) + right_se**4 / (right.size - 1)
    )
    assert summary.estimate == pytest.approx(np.mean(left) + np.mean(right))
    assert summary.standard_error == pytest.approx(np.sqrt(total_variance))
    assert summary.degrees_of_freedom == int(np.floor(raw_df))
    assert summary.cohort_sizes == (4, 5)


def test_joint_endpoint_interval_preserves_delta_substep_covariance():
    module = _load()
    # Fine-reference noise is deliberately shared between D and S.  Forming
    # D +/- S before the outer standard error is the certification contract.
    fine_noise = np.array([-3.0, -1.0, 1.0, 3.0]) * 0.001
    delta = -0.04 + fine_noise
    substep = 0.005 + fine_noise

    result = module.certify_joint_bias_endpoints(
        [delta],
        [substep],
        economic_bound=0.1,
        pde_discretization_envelope=0.002,
        component_confidence=0.975,
        label="test",
    )

    plus = result["endpoints"]["delta_plus_substep"]
    minus = result["endpoints"]["delta_minus_substep"]
    assert plus["estimate"] == pytest.approx(-0.035)
    assert minus["estimate"] == pytest.approx(-0.045)
    assert minus["standard_error"] == pytest.approx(0.0, abs=1e-15)
    assert result["status"] == "PASS"
    assert result["simultaneous_coverage_lower_bound"] == pytest.approx(0.95)


@pytest.mark.parametrize(
    ("center", "expected"),
    [(-0.2, "FAIL"), (-0.09, "INCONCLUSIVE"), (0.0, "PASS")],
)
def test_joint_endpoint_tri_state(center, expected):
    module = _load()
    delta = np.array([center - 0.01, center, center + 0.01, center])
    substep = np.zeros_like(delta)

    result = module.certify_joint_bias_endpoints(
        [delta],
        [substep],
        economic_bound=0.1,
        pde_discretization_envelope=0.005,
        component_confidence=0.975,
        label="tri-state",
    )

    assert result["status"] == expected


def test_joint_endpoint_rejects_unpaired_rows():
    module = _load()

    with pytest.raises(ValueError, match="matching shapes"):
        module.certify_joint_bias_endpoints(
            [np.ones(3)],
            [np.ones(4)],
            economic_bound=0.1,
            pde_discretization_envelope=0.0,
            component_confidence=0.975,
            label="bad",
        )


def test_allocation_projection_adds_source_variances_and_new_family_guard():
    module = _load()
    pattern = np.linspace(-1.0, 1.0, module.AGGREGATE_OUTER_BATCHES)
    sources = {
        name: {
            endpoint: pattern * scale
            for endpoint, scale in (
                ("delta_plus_substep", 0.02),
                ("delta_minus_substep", 0.01),
            )
        }
        for name in module.AGGREGATE_COHORT_NAMES
    }
    candidate = module.project_allocation_candidate(
        sources,
        {
            "delta_plus_substep": -0.02,
            "delta_minus_substep": -0.03,
        },
        primary_batches=module.DEVELOPMENT_PRIMARY_BATCH_GRID[0],
        middle_batches=module.DEVELOPMENT_MIDDLE_BATCH_GRID[0],
        smooth_heston_weight=module.DEVELOPMENT_HESTON_WEIGHT_GRID[0],
        pde_envelope=0.005,
    )

    plus = candidate["endpoint_results"]["delta_plus_substep"]
    source_variance = sum(
        value**2 for value in plus["source_standard_errors"].values()
    )
    expected_paths = (
        2
        * module.PRODUCTION_PRIMARY_PATHS_PER_BATCH
        * module.DEVELOPMENT_PRIMARY_BATCH_GRID[0]
        * len(module.PRIMARY_REFRESH_CASES)
        + 2
        * module.PRODUCTION_MIDDLE_PATHS_PER_BATCH
        * module.DEVELOPMENT_MIDDLE_BATCH_GRID[0]
        * len(module.CONTROL_CASES)
    )
    assert plus["standard_error"] == pytest.approx(np.sqrt(source_variance))
    assert plus["new_family_center_guard"] > 0.0
    assert candidate["total_unique_paths"] == expected_paths
    assert candidate["guarded_interval"][0] < candidate["projected_interval"][0]
    assert candidate["guarded_interval"][1] > candidate["projected_interval"][1]


@requires_banked_evidence
def test_declared_allocation_grid_projects_from_complete_development_rows():
    module = _load()
    parent = json.loads(
        (PARENT_EVIDENCE_FILE).read_text()
    )
    slv = {
        cell["case"]["name"]: cell
        for cell in parent["cells"]
        if cell["variant"] == "heston_slv"
    }
    pilot = {}
    for case_name in module.CONTROL_CASES:
        reference = copy.deepcopy(slv[case_name]["reference"])
        for level in ("target", "fine"):
            row = reference[level]
            row["batches_used"] = module.DEVELOPMENT_BATCHES
            row["batch_estimates"] = row["batch_estimates"][
                : module.DEVELOPMENT_BATCHES
            ]
            row["control_batch_estimates"] = row["control_batch_estimates"][
                : module.DEVELOPMENT_BATCHES
            ]
        pilot[case_name] = reference

    selection = module.select_production_allocation(parent, pilot)

    assert len(selection["candidates"]) == (
        len(module.DEVELOPMENT_PRIMARY_BATCH_GRID)
        * len(module.DEVELOPMENT_MIDDLE_BATCH_GRID)
    )
    assert all(
        row["primary_batches"] in module.DEVELOPMENT_PRIMARY_BATCH_GRID
        and row["middle_batches"] in module.DEVELOPMENT_MIDDLE_BATCH_GRID
        and row["smooth_heston_weight"]
        in module.DEVELOPMENT_HESTON_WEIGHT_GRID
        for row in selection["candidates"]
    )


def test_controlled_case_rows_recompose_the_declared_unbiased_identity():
    module = _load()
    primary = {
        "target": _paired_payload(
            [2.0, 4.0, 6.0, 8.0], [1.0, 3.0, 5.0, 7.0]
        ),
        "fine": _paired_payload(
            [1.5, 3.5, 5.5, 7.5], [0.5, 2.5, 4.5, 6.5]
        ),
    }
    middle = {
        "target": _paired_payload(
            [1.2, 2.2, 3.2, 4.2], [0.9, 1.9, 2.9, 3.9]
        ),
        "fine": _paired_payload(
            [1.0, 2.0, 3.0, 4.0], [0.8, 1.8, 2.8, 3.8]
        ),
    }
    heston_high = {
        "target": _paired_payload(
            [0.7, 1.7, 2.7, 3.7], [0.0, 0.0, 0.0, 0.0]
        ),
        "fine": _paired_payload(
            [0.6, 1.6, 2.6, 3.6], [0.0, 0.0, 0.0, 0.0]
        ),
    }

    delta, substep = module.controlled_case_economic_rows(
        pde_delta=10.0,
        economic_factor=2.0,
        primary_reference=primary,
        middle_reference=middle,
        heston_high_reference=heston_high,
        frozen_weight=0.95,
        heston_weight=0.85,
        output_batches=2,
    )

    def grouped(values):
        # STRIDED, matching _group_delta_rows since ae460f8: output row j
        # averages scrambles {j, j+m, j+2m, ...}, so reshape is (n/m, m) and the
        # mean is over axis 0. The consecutive form this helper used to carry --
        # reshape(m, n/m).mean(axis=1) -- pushes same-scramble cross-case CRN
        # coupling across outer-row boundaries where the empirical standard
        # error cannot see it, which is the defect the strided alignment fixed.
        return np.asarray(values, dtype=float).reshape(2, 2).mean(axis=0)

    target = (
        grouped([2.0, 4.0, 6.0, 8.0])
        - 0.95 * grouped([1.0, 3.0, 5.0, 7.0])
        + 0.95 * grouped([1.2, 2.2, 3.2, 4.2])
        - 0.85 * grouped([0.9, 1.9, 2.9, 3.9])
        + 0.85 * grouped([0.7, 1.7, 2.7, 3.7])
    )
    fine = (
        grouped([1.5, 3.5, 5.5, 7.5])
        - 0.95 * grouped([0.5, 2.5, 4.5, 6.5])
        + 0.95 * grouped([1.0, 2.0, 3.0, 4.0])
        - 0.85 * grouped([0.8, 1.8, 2.8, 3.8])
        + 0.85 * grouped([0.6, 1.6, 2.6, 3.6])
    )
    assert np.allclose(delta, (10.0 - fine) * 2.0)
    assert np.allclose(substep, (target - fine) * 2.0)


def test_nonzero_heston_weight_requires_a_high_expectation():
    module = _load()
    paired = {
        "target": _paired_payload([1.0, 2.0], [0.9, 1.9]),
        "fine": _paired_payload([0.8, 1.8], [0.7, 1.7]),
    }

    with pytest.raises(ValueError, match="requires a high expectation"):
        module.controlled_case_economic_rows(
            pde_delta=1.0,
            economic_factor=1.0,
            primary_reference=paired,
            middle_reference=paired,
            heston_high_reference=None,
            frozen_weight=0.95,
            heston_weight=0.85,
            output_batches=2,
        )


@requires_banked_evidence
def test_parent_pde_envelope_and_saved_rows_recompose_without_new_work():
    module = _load()
    parent = json.loads(
        (PARENT_EVIDENCE_FILE).read_text()
    )
    slv = {
        cell["case"]["name"]: cell
        for cell in parent["cells"]
        if cell["variant"] == "heston_slv"
    }
    primary = {
        case_name: slv[case_name]["reference"]
        for case_name in module.PRIMARY_REFRESH_CASES
    }
    middle = {}
    for case_name in module.CONTROL_CASES:
        reference = copy.deepcopy(slv[case_name]["reference"])
        for level in ("target", "fine"):
            reference[level] = _reserialize_paired(
                reference[level],
                batches=module.AGGREGATE_OUTER_BATCHES,
                paths_per_batch=reference[level]["paths_per_batch"],
                label=f"test/common-middle/{case_name}/{level}",
            )
        middle[case_name] = reference

    delta_cohorts, substep_cohorts, hashes = (
        module.aggregate_reference_cohorts(
            parent,
            primary_by_case=primary,
            middle_by_case=middle,
        )
    )
    axes, envelope = module.aggregate_pde_refinement(parent)

    assert tuple(delta_cohorts) == module.AGGREGATE_COHORT_NAMES
    assert tuple(substep_cohorts) == module.AGGREGATE_COHORT_NAMES
    assert {name: rows.size for name, rows in delta_cohorts.items()} == {
        "primary_refresh": 128,
        "middle_control": 128,
        "parent_heston_high": 1024,
        "schema11_replacements": 128,
    }
    assert all(np.all(np.isfinite(rows)) for rows in delta_cohorts.values())
    assert all(np.all(np.isfinite(rows)) for rows in substep_cohorts.values())
    assert set(hashes) == set(module.CONTROL_CASES) | {"near_ki"}

    # Pinned against the PUBLISHED amendment rather than hand-copied literals.
    # The literals here were stale -- written against the pre-strided parent --
    # and a hand-copied number cannot notice when the artifact moves under it.
    published = json.loads(BANKED_AMENDMENT_FILE.read_text())
    assert axes == pytest.approx(
        published["aggregate_pde_signed_refinement_contracts"]
    )
    assert envelope == pytest.approx(
        published["aggregate_pde_discretization_envelope"]
    )


@pytest.mark.parametrize(
    ("center", "spread", "bias_status", "route"),
    [
        # Wholly inside the +/-0.1 bound.
        (0.0, 0.0, "PASS", "pde"),
        # Straddling it. This needs real scatter: with zero variance the
        # interval collapses to a point, so a centre of -0.09 is simply INSIDE
        # the bound and passes. The status is a property of the INTERVAL, not
        # of the estimate, and only a fixture with width can exercise the
        # straddle. (The old parametrization asked for -0.09 with no spread and
        # expected INCONCLUSIVE, which stopped being reachable once the
        # endpoints were built this way.)
        (-0.09, 0.2, "INCONCLUSIVE", "excluded_greek_unresolved"),
        # Wholly outside.
        (-0.2, 0.0, "FAIL", "excluded_greek_unresolved"),
    ],
)
@requires_banked_evidence
def test_aggregate_decision_carries_heston_and_routes_only_on_joint_pass(
    center, spread, bias_status, route
):
    module = _load()
    parent = json.loads(
        (PARENT_EVIDENCE_FILE).read_text()
    )
    delta = np.full(module.AGGREGATE_OUTER_BATCHES, center)
    if spread:
        delta = delta + spread * np.where(
            np.arange(module.AGGREGATE_OUTER_BATCHES) % 2 == 0, 1.0, -1.0
        )
    substep = np.zeros_like(delta)
    delta_cohorts = {
        name: (
            delta.copy()
            if name == module.AGGREGATE_COHORT_NAMES[0]
            else np.zeros_like(delta)
        )
        for name in module.AGGREGATE_COHORT_NAMES
    }
    substep_cohorts = {
        name: np.zeros_like(substep) for name in module.AGGREGATE_COHORT_NAMES
    }

    decisions = module.make_aggregate_decisions(
        parent, delta_cohorts, substep_cohorts
    )

    assert decisions["heston"]["route"] == "pde"
    assert decisions["heston"]["certification_source"] == "schema11_parent"
    assert decisions["heston_slv"]["delta_bias"]["status"] == bias_status
    assert decisions["heston_slv"]["route"] == route
    assert decisions["heston_slv"]["cell_status"] == "PASS"


def _paired_payload(frozen, heston):
    frozen = np.asarray(frozen, dtype=float)
    heston = np.asarray(heston, dtype=float)
    rows = np.zeros((frozen.size, 5), dtype=float)
    controls = np.zeros_like(rows)
    rows[:, 3] = frozen
    controls[:, 3] = heston
    return {
        "batches_used": frozen.size,
        "batch_estimates": rows.tolist(),
        "control_batch_estimates": controls.tolist(),
    }


def _reserialize_paired(source, *, batches, paths_per_batch, label):
    rows = np.asarray(source["batch_estimates"], dtype=float)
    controls = np.asarray(source["control_batch_estimates"], dtype=float)
    repeats = int(np.ceil(batches / rows.shape[0]))
    rows = np.tile(rows, (repeats, 1))[:batches]
    controls = np.tile(controls, (repeats, 1))[:batches]
    covariance = np.cov(rows, rowvar=False, ddof=1)
    means = np.mean(rows, axis=0)
    standard_errors = np.sqrt(np.diag(covariance) / batches)
    payload = copy.deepcopy(source)
    payload.update(
        {
            "batch_estimates": rows.tolist(),
            "control_batch_estimates": controls.tolist(),
            "batches_used": batches,
            "paths_per_batch": paths_per_batch,
            "price": float(means[1]),
            "price_std_error": float(standard_errors[1]),
            "delta": float(means[3]),
            "delta_std_error": float(standard_errors[3]),
            "gamma": float(means[4]),
            "gamma_std_error": float(standard_errors[4]),
            "covariance": covariance.tolist(),
            "randomization_key": label,
            "total_unique_paths": paths_per_batch * batches,
            "total_path_valuations": 3 * paths_per_batch * batches,
        }
    )
    return payload


def _synthetic_added_reference(
    module, cell, *, purpose, seed, paths, batches, workers, primary
):
    case_name = cell["case"]["name"]
    evidence = {
        "variant": "heston_slv",
        "case": cell["case"],
        "purpose": purpose,
        "seed": seed,
        "paths_per_batch": paths,
        "batches": batches,
        "batch_workers": workers,
        "target_substeps_per_interval": cell["reference"][
            "target_substeps_per_interval"
        ],
        "fine_substeps_per_interval": cell["reference"][
            "fine_substeps_per_interval"
        ],
        "slv_spot_bridge_profile": module.stage16().SLV_SPOT_BRIDGE_PROFILE_BY_CASE[
            case_name
        ],
        "target": _reserialize_paired(
            cell["reference"]["target"],
            batches=batches,
            paths_per_batch=paths,
            label=f"test/{purpose}/{case_name}/target",
        ),
        "fine": _reserialize_paired(
            cell["reference"]["fine"],
            batches=batches,
            paths_per_batch=paths,
            label=f"test/{purpose}/{case_name}/fine",
        ),
    }
    if primary:
        evidence.update(
            {
                "slv_spot_strata": module.stage16().SLV_SPOT_STRATA,
                "slv_spot_antithetic": module.stage16().SLV_SPOT_ANTITHETIC,
            }
        )
    return evidence


def test_control_summary_is_explicitly_non_admissive_and_keeps_coupling():
    module = _load()
    reference = {
        "target": _paired_payload([1.0, 2.0, 3.0], [0.8, 1.8, 2.8]),
        "fine": _paired_payload([0.9, 1.9, 2.9], [0.75, 1.75, 2.75]),
    }

    summary = module.summarize_control_reference(reference)

    assert summary["admissible"] is False
    assert summary["fine"]["correlation"] == pytest.approx(1.0)
    assert summary["fine"]["unit_difference_standard_error_contracts"] == (
        pytest.approx(0.0, abs=1e-14)
    )
    assert summary["substep"]["batches"] == 3


@requires_banked_evidence
def test_schema12_full_payload_recomposes_publishes_and_routes(tmp_path, monkeypatch):
    module = _load()
    # This test is about recomposition and publication, not about whether the
    # tree is still PDE-compatible with the parent -- which has its own test.
    _accept_live_production_pde(module, monkeypatch)
    module.PRODUCTION_ALLOCATION_FROZEN = True
    module.PRODUCTION_PRIMARY_BATCHES = module.AGGREGATE_OUTER_BATCHES
    module.PRODUCTION_MIDDLE_BATCHES = module.AGGREGATE_OUTER_BATCHES
    parent = json.loads(
        (PARENT_EVIDENCE_FILE).read_text()
    )
    parent_decision = json.loads(
        (PARENT_DECISION_FILE).read_text()
    )
    slv = {
        cell["case"]["name"]: cell
        for cell in parent["cells"]
        if cell["variant"] == "heston_slv"
    }
    primary = {
        case_name: _synthetic_added_reference(
            module,
            slv[case_name],
            purpose="aggregate_only_primary_reference_refresh",
            seed=module.PRODUCTION_PRIMARY_SEED,
            paths=module.PRODUCTION_PRIMARY_PATHS_PER_BATCH,
            batches=module.PRODUCTION_PRIMARY_BATCHES,
            workers=module.PRODUCTION_PRIMARY_BATCH_WORKERS,
            primary=True,
        )
        for case_name in module.PRIMARY_REFRESH_CASES
    }
    middle = {
        case_name: _synthetic_added_reference(
            module,
            slv[case_name],
            purpose="aggregate_only_frozen_slv_heston_middle_control",
            seed=module.PRODUCTION_MIDDLE_SEED,
            paths=module.PRODUCTION_MIDDLE_PATHS_PER_BATCH,
            batches=module.PRODUCTION_MIDDLE_BATCHES,
            workers=module.PRODUCTION_MIDDLE_BATCH_WORKERS,
            primary=False,
        )
        for case_name in module.CONTROL_CASES
    }
    delta_cohorts, substep_cohorts, component_hashes = (
        module.aggregate_reference_cohorts(
            parent,
            primary_by_case=primary,
            middle_by_case=middle,
        )
    )
    decisions = module.make_aggregate_decisions(
        parent, delta_cohorts, substep_cohorts
    )
    axes, envelope = module.aggregate_pde_refinement(parent)
    implementation_hash = module.implementation_sha256()
    runtime = module.runtime_environment()
    run_configuration = module.production_run_configuration(
        implementation_hash=implementation_hash,
        runtime=runtime,
    )
    payload = {
        "schema_version": module.SCHEMA_VERSION,
        "study": module.STUDY,
        "certification_mode": module.CERTIFICATION_MODE,
        "profile": "production aggregate-only amendment",
        "created_at": "test-clock",
        "quick": False,
        "parent_certificate": module.parent_certificate_manifest(),
        "production_pde_compatibility_sha256": module.PARENT_PRODUCTION_PDE_SHA256,
        "runtime_environment": runtime,
        "implementation_sha256": implementation_hash,
        "run_configuration_sha256": module._canonical_sha256(run_configuration),
        "run_configuration": run_configuration,
        "elapsed_seconds": 0.0,
        "production_engine_controls": module.stage16().PRODUCTION_ENGINE_CONTROLS,
        "reference_seeds": {
            "schema11_parent": parent["reference_seeds"],
            "aggregate_primary_refresh": module.PRODUCTION_PRIMARY_SEED,
            "aggregate_middle_control": module.PRODUCTION_MIDDLE_SEED,
        },
        "sampling_by_variant": parent["sampling_by_variant"],
        "aggregate_sampling": {
            "schema11_replacement_batches": module.AGGREGATE_OUTER_BATCHES,
            "primary_refresh": run_configuration["primary_refresh"],
            "middle_control": run_configuration["middle_control"],
        },
        "policy": {
            "economic_bound_contracts": module.stage16().DELTA_BIAS_BOUND_CONTRACTS,
            "component_confidence": (
                module.stage16().STOCHASTIC_COMPONENT_CONFIDENCE
            ),
            "endpoint_method": run_configuration["endpoint_method"],
            "weights_by_case": module.AGGREGATE_CONTROL_WEIGHTS,
            "no_optional_stopping": True,
            "individual_cell_authority": "schema11_parent",
            "production_engine_controls": module.stage16().PRODUCTION_ENGINE_CONTROLS,
        },
        "anchors": parent["anchors"],
        "cells": parent["cells"],
        # Renamed at ae460f8: per-cell reuse is gated on a per-cell IDENTITY
        # (numerical projection + declared plan + runtime + consumed cells),
        # not on a fleet-wide provenance record.
        "cell_identities": parent["cell_identities"],
        "aggregate_reference": {
            "primary_by_case": primary,
            "middle_by_case": middle,
            "component_hashes": component_hashes,
            "weights_by_case": module.AGGREGATE_CONTROL_WEIGHTS,
            "near_ki_source": "schema11_published_multilevel_reference",
            "heston_high_source": "schema11_parent_cells_no_rerun",
        },
        "aggregate_cohorts": {
            "order": list(module.AGGREGATE_COHORT_NAMES),
            "delta_contracts": {
                name: delta_cohorts[name].tolist()
                for name in module.AGGREGATE_COHORT_NAMES
            },
            "substep_contracts": {
                name: substep_cohorts[name].tolist()
                for name in module.AGGREGATE_COHORT_NAMES
            },
        },
        "aggregate_pde_signed_refinement_contracts": axes,
        "aggregate_pde_discretization_envelope": envelope,
        "added_work": module._added_work_counts(primary, middle),
        "decisions": decisions,
        "parent_decision_sha256": parent_decision["decision_sha256"],
    }

    module.publish_payload(payload, tmp_path)

    evidence = json.loads((tmp_path / "adi_greek_certification.json").read_text())
    decision = json.loads(
        (tmp_path / "adi_greek_certification_decision.json").read_text()
    )
    module.validate_payload(evidence)
    assert decision == module.build_decision_payload(evidence)
    assert decision["decisions"]["heston"]["route"] == "pde"
    assert decision["added_work"]["pde_solves"] == 0
    assert decision["added_work"]["near_ki_reruns"] == 0
    assert (tmp_path / "adi_greek_certification.md").exists()

    stage12_path = ROOT / "example/mo_volmodels/12_snowball_volmodel_backtest.py"
    spec = importlib.util.spec_from_file_location(
        "mo_stage12_schema12_integration", stage12_path
    )
    stage12 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = stage12
    assert spec.loader is not None
    spec.loader.exec_module(stage12)
    stage12._STAGE17 = module
    routing = stage12.load_adi_greek_routing(
        tmp_path / "adi_greek_certification_decision.json"
    )
    assert routing.routes == {
        variant: row["route"] for variant, row in decision["decisions"].items()
    }


@requires_banked_evidence
def test_production_invocation_fails_closed_before_allocation_is_frozen():
    module = _load()
    module.PRODUCTION_ALLOCATION_FROZEN = False

    with pytest.raises(ValueError, match="production allocation is not frozen"):
        module.main(
            [
                "--parent-evidence",
                str(PARENT_EVIDENCE_FILE),
                "--parent-decision",
                str(PARENT_DECISION_FILE),
            ]
        )


def test_admissive_validator_rejects_development_payload():
    module = _load()
    payload = {
        "schema_version": module.SCHEMA_VERSION,
        "study": module.STUDY,
        "certification_mode": "development_control_pilot",
    }

    with pytest.raises(ValueError, match="aggregate-only amendment"):
        module.validate_payload(copy.deepcopy(payload))


def test_adaptive_flag_default_off_preserves_frozen_allocation():
    """S-G4: without --adaptive nothing about the frozen allocation moves."""
    module = _load()
    config = module.production_run_configuration(
        implementation_hash=module.implementation_sha256(),
        runtime=module.runtime_environment(),
    )
    recommendation = module.frozen_allocation_manifest()["recommendation"]
    assert config["primary_refresh"]["batches"] == recommendation["primary_batches"]
    assert config["middle_control"]["batches"] == recommendation["middle_batches"]
    assert config["no_optional_stopping"] is True
    assert config["stopping_rule"] == "fixed_allocation"
    assert "adaptive_run" not in config


def test_adaptive_mode_declares_blind_stopping_rule():
    """Precision stopping is not optional stopping; both facts must be legible."""
    module = _load()
    config = module.production_run_configuration(
        implementation_hash=module.implementation_sha256(),
        runtime=module.runtime_environment(),
        adaptive=True,
        precision_target=0.02,
        budget_hours=12.0,
        pilot_batches=32,
    )
    assert config["no_optional_stopping"] is True
    assert config["stopping_rule"] == "precision_blind"
    assert config["stopping_rule_reads_estimate"] is False
    assert config["adaptive_run"]["precision_target"] == 0.02
    assert config["adaptive_run"]["budget_hours"] == 12.0
    assert config["adaptive_run"]["pilot_batches"] == 32


def test_adaptive_allocation_is_pilot_deterministic():
    """S-G3: the same pilot statistics must freeze to the same allocation."""
    module = _load()
    pilot = {
        "ordinary_full": {"batches": 32, "batch_sd": 1.07, "seconds_per_batch": 33.4},
        "ordinary_decayed": {"batches": 32, "batch_sd": 1.08, "seconds_per_batch": 17.4},
        "sigma_collapse": {"batches": 32, "batch_sd": 0.70, "seconds_per_batch": 33.3},
    }
    first = module.freeze_adaptive_allocation(pilot, budget_hours=12.0)
    second = module.freeze_adaptive_allocation(pilot, budget_hours=12.0)
    assert first == second
    assert first["allocation_sha256"] == second["allocation_sha256"]
    # A different budget must produce a different frozen decision.
    third = module.freeze_adaptive_allocation(pilot, budget_hours=6.0)
    assert third["allocation_sha256"] != first["allocation_sha256"]
    # Noisier-and-cheaper cells earn more batches than quieter-and-dearer ones.
    assert first["allocation"]["ordinary_decayed"] > first["allocation"]["sigma_collapse"]


def test_monitor_hook_is_estimate_blind():
    """S-G1 at the wiring level: only precision fields reach the stopping path."""
    module = _load()
    banked = {
        "ordinary_full": {
            "batch_deltas": np.array([0.1, 0.2, 0.15, 0.12]),
            "seconds_per_batch": 33.4,
        },
        "sigma_collapse": {
            "batch_deltas": np.array([0.30, 0.34, 0.31, 0.33]),
            "seconds_per_batch": 33.3,
        },
    }
    cells = module.monitor_cells_from_banked(banked)
    assert [c.name for c in cells] == ["ordinary_full", "sigma_collapse"]
    assert all(not hasattr(c, "estimate") for c in cells)
    assert cells[0].n_batches == 4
    assert cells[0].seconds_per_batch == 33.4
    # The decision must be reachable from these cells alone.
    decision = module.precision_stop(
        cells, target_halfwidth=10.0, elapsed_seconds=1.0, budget_seconds=100.0
    )
    assert decision.stop is True and decision.trigger == "target-reached"


def test_adaptive_allocation_never_reduces_evidence_below_the_frozen_floor():
    """An adaptive run may add batches to a cell, never take them away."""
    module = _load()
    pilot = {
        "ordinary_full": {"batches": 32, "batch_sd": 0.001, "seconds_per_batch": 33.4},
    }
    frozen = module.freeze_adaptive_allocation(pilot, budget_hours=0.1)
    batches = module.adaptive_batches_by_case(
        frozen, default_batches=module.PRODUCTION_PRIMARY_BATCHES
    )
    # Every control case is present, and the frozen count is the floor even
    # where the pilot says the cell is already quiet.
    assert set(batches) == set(module.CONTROL_CASES)
    assert min(batches.values()) >= module.PRODUCTION_PRIMARY_BATCHES
