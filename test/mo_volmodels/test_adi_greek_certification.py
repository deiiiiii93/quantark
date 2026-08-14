import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quantark.validation.cell_identity import project_source


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "example/mo_volmodels/16_adi_greek_certification.py"


def _load():
    name = "mo_adi_greek_certification_16"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_regime_matrix_covers_required_failure_modes():
    module = _load()
    cases = module.certification_cases(quick=False)
    tags = {tag for case in cases for tag in case.tags}

    assert {
        "full",
        "decayed",
        "near_ko",
        "near_ki",
        "low_feller",
        "sigma_collapse",
        "near_expiry",
    } <= tags
    assert {case.name for case in module.certification_cases(quick=True)} == {
        "ordinary_decayed",
        "near_ko",
        "sigma_collapse",
    }


def test_refinement_ladders_change_one_axis_at_a_time():
    module = _load()
    ladders = module.grid_ladders(3.0, quick=False)
    target = ladders["target"]

    assert (target.n_x, target.n_v, target.n_t) == (300, 135, 4800)
    assert (
        ladders["n_x"][0].n_x,
        ladders["n_v"][0].n_v,
        ladders["n_t"][0].n_t,
    ) == (200, 90, 2400)
    assert (
        ladders["n_x"][-1].n_x,
        ladders["n_v"][-1].n_v,
        ladders["n_t"][-1].n_t,
    ) == (450, 180, 7200)

    for point in ladders["n_x"]:
        assert point.n_v == target.n_v
        assert point.n_t == target.n_t
    for point in ladders["n_v"]:
        assert point.n_x == target.n_x
        assert point.n_t == target.n_t
    for point in ladders["n_t"]:
        assert point.n_x == target.n_x
        assert point.n_v == target.n_v
    assert [point.n_x for point in ladders["n_x"]] == sorted(
        point.n_x for point in ladders["n_x"]
    )
    assert [point.n_v for point in ladders["n_v"]] == sorted(
        point.n_v for point in ladders["n_v"]
    )
    assert [point.n_t for point in ladders["n_t"]] == sorted(
        point.n_t for point in ladders["n_t"]
    )


def test_dense_ki_ladder_matches_production_barrier_policy():
    module = _load()
    ladders = module.grid_ladders(
        1.0,
        quick=False,
        dense_ki_stencil=True,
    )

    assert ladders["target"].as_dict() == {
        "n_x": 600,
        "n_v": 135,
        "n_t": 16 * 252,
    }
    assert [point.n_v for point in ladders["n_v"]] == [90, 135, 180]


def test_schema11_amendment_profile_is_pinned():
    module = _load()

    assert module.SCHEMA_VERSION == 13
    assert module.PARENT_SCHEMA_VERSION == 9
    assert module.PARENT_SEED == 20260807
    assert module.AMENDMENT_REPLACEMENT_CASES == frozenset({"near_ki", "low_feller"})
    assert module.HESTON_REFERENCE_SEED == 20260808
    assert module.SLV_PRIMARY_SEED == 20260809
    assert module.SLV_MID_CONTROL_SEED == 20260810
    assert module.PRODUCTION_HESTON_BATCHES == 1024
    assert module.PRODUCTION_HESTON_BATCHES_BY_CASE["near_ki"] == 2048
    assert module.PRODUCTION_SLV_PRIMARY_BATCHES_BY_CASE["near_ki"] == 256
    assert module.PRODUCTION_SLV_PRIMARY_BATCHES_BY_CASE["low_feller"] == 512
    assert module.PRODUCTION_SLV_BATCHES_BY_CASE["near_ki"] == 128
    assert module.PRODUCTION_SLV_BATCHES_BY_CASE["low_feller"] == 512
    assert module.SLV_MULTILEVEL_CASES == frozenset({"near_ki"})
    assert module.PRODUCTION_CELL_WORKERS == 2
    # Batch workers are a scheduling choice, not part of the numerical profile:
    # the RQMC reduction is seed-keyed and ordered, so batch estimates are
    # bitwise identical across worker counts (probe P2).  This test used to
    # freeze the per-case downgrades (2 / 2 / 4) as though they were numerical,
    # which made a pure wall-clock decision look load-bearing.  The profile is
    # now uniform at the measured operating point, and
    # test_batch_worker_profile_is_uniform_at_the_measured_operating_point owns
    # that invariant across all fourteen cells rather than three.
    assert (
        module.PRODUCTION_RQMC_BATCH_WORKERS_BY_VARIANT_CASE["heston_slv"]["low_feller"]
        == module.PRODUCTION_RQMC_BATCH_WORKERS
    )


def test_schema9_parent_identity_is_pinned_and_its_pde_carry_path_is_closed():
    """The parent pin is history; WS-C legitimately closed the carry path.

    ``PARENT_PRODUCTION_PDE_SHA256`` records the production-PDE surface of the
    schema-9 parent.  It is a historical fact, never a value to re-pin: the
    guard in ``load_and_validate_parent_certificate`` exists precisely so that
    banked PDE cells cannot be carried once the live numerics move.  WS-C added
    the ``semi_lagrangian`` v-transport to ``quantark/volmodels/adi_core.py``,
    which sits inside ``PRODUCTION_PDE_INPUT_ROOTS``, so the surface diverged by
    design and P1.4 must run as a full recertification rather than an amendment.
    """
    module = _load()

    manifest = module.parent_certificate_manifest()
    assert manifest["schema_version"] == 9
    assert manifest["evidence_sha256"] == module.PARENT_EVIDENCE_SHA256
    assert manifest["carried_cell_sha256"] == (module.PARENT_CARRIED_CELL_SHA256)
    assert len(manifest["carried_cell_sha256"]) == 12
    assert manifest["production_pde_compatibility_sha256"] == (
        module.PARENT_PRODUCTION_PDE_SHA256
    )

    live = module.production_pde_compatibility_sha256()
    assert live != module.PARENT_PRODUCTION_PDE_SHA256

    # The divergence only means something if the digest genuinely tracks the
    # projected sources, so pin that too: deterministic for a fixed projection,
    # and different for a different one.
    assert live == module.production_pde_compatibility_sha256()
    original_roots = module.PRODUCTION_PDE_INPUT_ROOTS
    try:
        module.PRODUCTION_PDE_INPUT_ROOTS = original_roots[:-1]
        assert module.production_pde_compatibility_sha256() != live
    finally:
        module.PRODUCTION_PDE_INPUT_ROOTS = original_roots


def test_batch_worker_profile_is_uniform_at_the_measured_operating_point():
    """No cell may be downgraded below the measured safe worker count.

    Batch workers are a pure scheduling choice: the RQMC reduction is
    seed-keyed and ordered, so batch estimates are bitwise identical across
    worker counts (probe P2 measured 1/4/8 on the real paired reference,
    max_abs_batch_diff 0.0, 2.04x wall at four workers, saturating by eight).
    Peak RSS at four workers is 7.85 GB on the heaviest cell -- the three-year
    ``ordinary_full`` horizon -- so two cells in flight under
    ``PRODUCTION_CELL_WORKERS`` sit near 15.7 GB against a ~32 GB budget.

    Per-case downgrades therefore buy no accuracy and cost wall-clock, and a
    silently reintroduced one would be invisible in the evidence.  Pin the
    uniform profile so it has to be argued for, not merely edited.
    """
    module = _load()

    assert module.PRODUCTION_RQMC_BATCH_WORKERS == 4

    expected_cases = {case.name for case in module.certification_cases(quick=False)}
    profile = module.PRODUCTION_RQMC_BATCH_WORKERS_BY_VARIANT_CASE
    assert set(profile) == {"heston", "heston_slv"}
    for variant, by_case in profile.items():
        assert set(by_case) == expected_cases, variant
        for case_name, workers in by_case.items():
            assert workers == module.PRODUCTION_RQMC_BATCH_WORKERS, (
                variant,
                case_name,
            )

    # Two cells in flight at this worker count must stay inside the host budget.
    measured_peak_gb_heaviest_cell = 7.85
    assert module.PRODUCTION_CELL_WORKERS * measured_peak_gb_heaviest_cell < 32.0


def test_implementation_hash_covers_every_validation_module():
    """``quantark/validation`` IS the certification logic; none may sit outside.

    ``IMPLEMENTATION_INPUTS`` names files rather than directories, so a module
    added to the package is covered only if someone remembers to list it.  This
    closes that gap by construction instead of per file: any new certification
    primitive is inside the fail-closed digest the moment it lands.
    """
    module = _load()

    on_disk = {
        f"quantark/validation/{path.name}"
        for path in (ROOT / "quantark" / "validation").glob("*.py")
    }
    assert on_disk, "expected to find validation modules on disk"
    assert on_disk <= set(module.IMPLEMENTATION_INPUTS), sorted(
        on_disk - set(module.IMPLEMENTATION_INPUTS)
    )


def test_implementation_hash_covers_the_shared_qe_variance_kernel():
    """A shared kernel that sets reference values must sit inside the digest.

    The QE variance update was inline in ``snowball_vol_mc_engines.py`` -- which
    IS an implementation input -- until it was extracted to
    ``quantark/montecarlo/qe_kernels.py`` so one definition could carry the
    optional Numba backend.  That refactor preserved every value (the NumPy and
    Numba paths are asserted bitwise equal), but it moved the arithmetic out of
    the fail-closed projection, so editing the kernel would no longer invalidate
    a single banked checkpoint.  Extracting code must not shrink hash coverage.
    """
    module = _load()

    from quantark.asset.equity.engine.mc import snowball_vol_mc_engines
    from quantark.montecarlo import qe_kernels

    # The dependency is real, not hypothetical: the SLV reference engine calls
    # this exact function object for its variance step.
    assert snowball_vol_mc_engines.qe_variance_step is qe_kernels.qe_variance_step

    relative = "quantark/montecarlo/qe_kernels.py"
    assert relative in module.IMPLEMENTATION_INPUTS

    # And the entry is load-bearing rather than decorative: the digest moves
    # when the kernel leaves the projection.
    covered = module.implementation_sha256()
    original_inputs = module.IMPLEMENTATION_INPUTS
    try:
        module.IMPLEMENTATION_INPUTS = tuple(
            path for path in original_inputs if path != relative
        )
        assert module.implementation_sha256() != covered
    finally:
        module.IMPLEMENTATION_INPUTS = original_inputs


def test_production_run_cannot_implicitly_launch_all_14_cells(tmp_path):
    module = _load()

    with pytest.raises(ValueError, match="production certification is incremental"):
        module.main(["--output-dir", str(tmp_path)])


def _amendment_decision_fixture(module, *, replacement_status="PASS"):
    anchors = [
        {"name": name, "status": "PASS"} for name in module.REQUIRED_ANCHOR_NAMES
    ]
    cells = []
    for case in module.certification_cases(quick=False):
        cells.append(
            {
                "variant": "heston",
                "case": case.as_dict(),
                "status": "PASS",
            }
        )
    for case in module.certification_cases(quick=False):
        replacement = case.name in module.AMENDMENT_REPLACEMENT_CASES
        difference = -0.005 if replacement else 0.01
        cells.append(
            {
                "variant": "heston_slv",
                "case": case.as_dict(),
                "status": replacement_status if replacement else "PASS",
                "batch_difference_contracts": {
                    "delta": [difference] * module.AMENDMENT_AGGREGATE_BATCHES
                },
                "certifications": {
                    "delta": {
                        "pde_signed_refinement_contracts": {
                            "n_x": 0.0,
                            "n_v": 0.0,
                            "n_t": 0.0,
                        },
                        "reference_substep_batch_contracts": [0.0]
                        * module.AMENDMENT_AGGREGATE_BATCHES,
                    }
                },
            }
        )
    parent_decisions = {
        "heston": {
            "route": "pde",
            "evidence_complete": True,
            "aggregate_common_scrambles": module.PRODUCTION_HESTON_BATCHES,
        }
    }
    return cells, anchors, parent_decisions


def test_amendment_bias_sums_independent_cohort_means():
    module = _load()
    cells, anchors, parent_decisions = _amendment_decision_fixture(module)

    decisions, cohorts = module.make_amendment_decisions(
        cells,
        anchors,
        parent_decisions,
    )

    bias = decisions["heston_slv"]["delta_bias"]
    assert bias["estimate_difference"] == pytest.approx((5 * 0.01 - 2 * 0.005) / 7)
    assert bias["reference_standard_error"] == pytest.approx(0.0, abs=1e-18)
    assert bias["status"] == "PASS"
    assert decisions["heston"]["certification_source"] == "schema9_parent"
    assert decisions["heston_slv"]["route"] == "pde"
    assert cohorts["method"] == "sum_of_independent_cohort_means"
    assert [row["seed"] for row in cohorts["cohorts"]] == [
        module.PARENT_SEED,
        module.SLV_PRIMARY_SEED,
    ]


def test_amendment_keeps_slv_excluded_until_both_replacements_pass():
    module = _load()
    cells, anchors, parent_decisions = _amendment_decision_fixture(
        module,
        replacement_status="INCONCLUSIVE",
    )

    decisions, _ = module.make_amendment_decisions(
        cells,
        anchors,
        parent_decisions,
    )

    assert decisions["heston"]["route"] == "pde"
    assert decisions["heston_slv"]["route"] == "excluded_greek_unresolved"


def test_incremental_runner_computes_only_two_replacements_and_one_control(
    monkeypatch, tmp_path
):
    module = _load()
    cases = module.certification_cases(quick=False)
    parent_cells = [
        {
            "variant": variant,
            "case": case.as_dict(),
            "source_marker": "parent",
        }
        for variant in ("heston", "heston_slv")
        for case in cases
    ]
    parent = {"anchors": [], "cells": parent_cells}
    parent_decision = {"decisions": {"heston": {"route": "pde"}}}
    monkeypatch.setattr(
        module,
        "load_and_validate_parent_certificate",
        lambda *_: (parent, parent_decision, module.parent_certificate_manifest()),
    )
    monkeypatch.setattr(module, "_write_checkpoint", lambda *_, **__: None)
    calls = []

    def fake_control(case, **kwargs):
        calls.append(("control", case.name))
        return {
            "variant": "heston",
            "case": case.as_dict(),
            "purpose": "reference_only_slv_high_control",
        }

    def fake_cell(variant, case, **kwargs):
        calls.append(("replacement", case.name))
        return {
            "variant": variant,
            "case": case.as_dict(),
            "source_marker": "replacement",
        }

    monkeypatch.setattr(module, "build_heston_high_control_evidence", fake_control)
    monkeypatch.setattr(module, "certify_case", fake_cell)
    monkeypatch.setattr(
        module,
        "make_amendment_decisions",
        lambda *_: (
            {
                "heston": {"route": "pde"},
                "heston_slv": {"route": "excluded_greek_unresolved"},
            },
            {"method": "sum_of_independent_cohort_means"},
        ),
    )
    published = {}
    monkeypatch.setattr(
        module,
        "publish_payload",
        lambda payload, output_dir: published.update(
            {"payload": payload, "output_dir": output_dir}
        ),
    )
    args = SimpleNamespace(
        amend_parent_evidence=Path("parent-evidence.json"),
        amend_parent_decision=Path("parent-decision.json"),
        hedge_inception_spot=module.DEFAULT_HEDGE_INCEPTION_SPOT,
        output_dir=tmp_path,
        resume=False,
    )

    assert module.run_incremental_amendment(args) == 0

    assert sorted(calls) == [
        ("control", "near_ki"),
        ("replacement", "low_feller"),
        ("replacement", "near_ki"),
    ]
    payload = published["payload"]
    assert len(payload["cells"]) == 14
    assert all(
        cell["source_marker"] == "parent"
        for cell in payload["cells"]
        if cell["variant"] == "heston"
    )
    assert {
        cell["case"]["name"]
        for cell in payload["cells"]
        if cell.get("source_marker") == "replacement"
    } == {"near_ki", "low_feller"}
    assert module.PRODUCTION_HESTON_QE_SUBSTEPS_BY_CASE["near_ki"] == {
        "target": 8,
        "fine": 16,
    }
    assert module.PRODUCTION_HESTON_QE_SUBSTEPS_BY_CASE["low_feller"] == {
        "target": 4,
        "fine": 8,
    }
    assert module.PRODUCTION_SLV_QE_SUBSTEPS_BY_CASE["low_feller"] == {
        "target": 7,
        "fine": 14,
    }


def test_dense_ki_and_ko_share_one_exact_integer_clock():
    module = _load()
    case = next(
        case
        for case in module.certification_cases(quick=False)
        if case.name == "sigma_collapse"
    )
    product = module.make_snowball(case, dense_ki=True)
    ki = np.asarray(product.barrier_config.ki_observation_dates)
    ko = np.asarray(product.barrier_config.ko_observation_dates)

    assert np.all(np.diff(ki) > 0.0)
    assert np.all(np.diff(ko) > 0.0)
    assert ki[-1] == case.maturity
    assert ko[-1] == case.maturity
    assert set(ko).issubset(set(ki))


def test_pde_axis_refinement_rejects_material_divergence():
    module = _load()

    def row(delta, gamma):
        return {"delta": delta, "gamma": gamma}

    ladders = {
        "axes": {
            axis: [row(0.0, 0.0), row(0.2, 0.2), row(0.6, 0.6)]
            for axis in ("n_x", "n_v", "n_t")
        }
    }
    scale = module.EconomicGreekScale(
        model_spot=100.0,
        hedge_inception_spot=100.0,
        study_notional=100.0,
        hedge_multiplier=1.0,
    )

    diagnostics = module._pde_refinement_diagnostics(ladders, scale)

    assert diagnostics["status"] == "FAIL"
    assert diagnostics["delta"]["axes"]["n_x"]["status"] == "FAIL"


def _passing_cell(module, variant, batches=16, case_name="ordinary_full"):
    bridge_profile = module.HESTON_SPOT_BRIDGE_PROFILE_BY_CASE.get(
        case_name, {"strata": 1, "dimensions": 1}
    )
    slv_bridge_profile = module.SLV_SPOT_BRIDGE_PROFILE_BY_CASE.get(
        case_name, {"strata": 1, "dimensions": 1}
    )
    substeps = module.PRODUCTION_QE_SUBSTEPS_BY_VARIANT_CASE[variant][case_name]
    return {
        "variant": variant,
        "case": {"name": case_name},
        "status": "PASS",
        "reference": {
            "heston_spot_bridge_strata": (
                bridge_profile["strata"] if variant == "heston" else None
            ),
            "heston_spot_bridge_dimensions": (
                bridge_profile["dimensions"] if variant == "heston" else None
            ),
            "slv_spot_bridge_strata": (
                slv_bridge_profile["strata"] if variant == "heston_slv" else None
            ),
            "slv_spot_bridge_dimensions": (
                slv_bridge_profile["dimensions"] if variant == "heston_slv" else None
            ),
            "target_substeps_per_interval": substeps["target"],
            "fine_substeps_per_interval": substeps["fine"],
            "estimator": {
                "name": (
                    "three_level_frozen_slv_heston_control"
                    if variant == "heston_slv"
                    and case_name in module.SLV_MULTILEVEL_CASES
                    else "primary_conditional_rqmc"
                )
            },
        },
        "batch_difference_contracts": {
            "delta": [0.01 + 0.001 * ((i % 3) - 1) for i in range(batches)]
        },
        "certifications": {
            "delta": {
                "pde_envelope_contracts": {"total": 0.005},
                "pde_signed_refinement_contracts": {
                    "n_x": 0.001,
                    "n_v": -0.001,
                    "n_t": 0.0,
                },
                "reference_substep_envelope_contracts": 0.002,
                "reference_substep_batch_contracts": [
                    0.001 * ((i % 3) - 1) for i in range(batches)
                ],
            }
        },
    }


def test_decision_never_admits_a_quick_profile():
    module = _load()
    decisions = module.make_decisions(
        [_passing_cell(module, "heston", batches=4)],
        [{"status": "PASS"}],
        quick=True,
        variants=["heston"],
    )

    assert decisions["heston"]["route"] == "excluded_greek_unresolved"
    assert "quick profile" in decisions["heston"]["reason"]


def test_production_decision_admits_only_complete_pass_evidence():
    module = _load()
    rows = [
        _passing_cell(module, "heston", case_name=case.name)
        for case in module.certification_cases(quick=False)
    ]
    anchors = [
        {"name": name, "status": "PASS"} for name in module.REQUIRED_ANCHOR_NAMES
    ]
    decisions = module.make_decisions(
        rows,
        anchors,
        quick=False,
        variants=["heston"],
    )

    assert decisions["heston"]["route"] == "pde"
    assert decisions["heston"]["delta_bias"]["status"] == "PASS"


def test_production_decision_enforces_variant_sampling_profile():
    module = _load()
    anchors = [
        {"name": name, "status": "PASS"} for name in module.REQUIRED_ANCHOR_NAMES
    ]
    rows = [
        _passing_cell(
            module,
            "heston_slv",
            batches=module.PRODUCTION_SLV_BATCHES_BY_CASE[case.name],
            case_name=case.name,
        )
        for case in module.certification_cases(quick=False)
    ]
    sampling = {
        "heston_slv": {
            "paths_per_batch": module.PRODUCTION_SLV_PATHS_PER_BATCH,
            "batches": module.PRODUCTION_SLV_BATCHES,
            "batches_by_case": module.PRODUCTION_SLV_BATCHES_BY_CASE,
            "primary_batches_by_case": (module.PRODUCTION_SLV_PRIMARY_BATCHES_BY_CASE),
        }
    }

    admitted = module.make_decisions(
        rows,
        anchors,
        quick=False,
        variants=["heston_slv"],
        sampling_by_variant=sampling,
        slv_spot_strata=module.SLV_SPOT_STRATA,
        slv_spot_antithetic=module.SLV_SPOT_ANTITHETIC,
        slv_spot_bridge_strata=module.SLV_SPOT_BRIDGE_STRATA,
        slv_spot_bridge_profile_by_case=(module.SLV_SPOT_BRIDGE_PROFILE_BY_CASE),
    )
    stale = module.make_decisions(
        rows,
        anchors,
        quick=False,
        variants=["heston_slv"],
        sampling_by_variant=sampling,
        slv_spot_strata=module.SLV_SPOT_STRATA,
        slv_spot_antithetic=module.SLV_SPOT_ANTITHETIC,
        slv_spot_bridge_strata=1,
        slv_spot_bridge_profile_by_case=(module.SLV_SPOT_BRIDGE_PROFILE_BY_CASE),
    )

    assert admitted["heston_slv"]["route"] == "pde"
    assert stale["heston_slv"]["route"] == "excluded_greek_unresolved"
    assert stale["heston_slv"]["sampling_complete"] is False


def test_production_decision_enforces_heston_bridge_sampling_profile():
    module = _load()
    anchors = [
        {"name": name, "status": "PASS"} for name in module.REQUIRED_ANCHOR_NAMES
    ]
    rows = [
        _passing_cell(
            module,
            "heston",
            batches=module.PRODUCTION_HESTON_BATCHES_BY_CASE[case.name],
            case_name=case.name,
        )
        for case in module.certification_cases(quick=False)
    ]
    sampling = {
        "heston": {
            "paths_per_batch": module.PRODUCTION_HESTON_PATHS_PER_BATCH,
            "batches": module.PRODUCTION_HESTON_BATCHES,
            "batches_by_case": module.PRODUCTION_HESTON_BATCHES_BY_CASE,
        }
    }

    admitted = module.make_decisions(
        rows,
        anchors,
        quick=False,
        variants=["heston"],
        sampling_by_variant=sampling,
        heston_spot_bridge_profile_by_case=(module.HESTON_SPOT_BRIDGE_PROFILE_BY_CASE),
    )
    stale = module.make_decisions(
        rows,
        anchors,
        quick=False,
        variants=["heston"],
        sampling_by_variant=sampling,
        heston_spot_bridge_profile_by_case={
            name: {"strata": 1, "dimensions": 1}
            for name in module.HESTON_SPOT_BRIDGE_PROFILE_BY_CASE
        },
    )

    assert admitted["heston"]["route"] == "pde"
    assert stale["heston"]["route"] == "excluded_greek_unresolved"
    assert stale["heston"]["sampling_complete"] is False


def test_heston_bias_uses_common_scramble_prefix_and_exact_case_profile():
    module = _load()
    anchors = [
        {"name": name, "status": "PASS"} for name in module.REQUIRED_ANCHOR_NAMES
    ]
    rows = [
        _passing_cell(
            module,
            "heston",
            batches=module.PRODUCTION_HESTON_BATCHES_BY_CASE[case.name],
            case_name=case.name,
        )
        for case in module.certification_cases(quick=False)
    ]
    near_ki = next(row for row in rows if row["case"]["name"] == "near_ki")
    common = module.PRODUCTION_HESTON_BATCHES
    near_ki["batch_difference_contracts"]["delta"][common:] = [100.0] * (
        module.PRODUCTION_HESTON_BATCHES_BY_CASE["near_ki"] - common
    )
    near_ki["certifications"]["delta"]["reference_substep_batch_contracts"][common:] = [
        100.0
    ] * (module.PRODUCTION_HESTON_BATCHES_BY_CASE["near_ki"] - common)
    sampling = {
        "heston": {
            "paths_per_batch": module.PRODUCTION_HESTON_PATHS_PER_BATCH,
            "batches": common,
            "batches_by_case": module.PRODUCTION_HESTON_BATCHES_BY_CASE,
        }
    }

    admitted = module.make_decisions(
        rows,
        anchors,
        quick=False,
        variants=["heston"],
        sampling_by_variant=sampling,
        heston_spot_bridge_profile_by_case=(module.HESTON_SPOT_BRIDGE_PROFILE_BY_CASE),
    )
    stale_sampling = {
        "heston": {
            **sampling["heston"],
            "batches_by_case": {
                **module.PRODUCTION_HESTON_BATCHES_BY_CASE,
                "near_ki": common,
            },
        }
    }
    stale = module.make_decisions(
        rows,
        anchors,
        quick=False,
        variants=["heston"],
        sampling_by_variant=stale_sampling,
        heston_spot_bridge_profile_by_case=(module.HESTON_SPOT_BRIDGE_PROFILE_BY_CASE),
    )

    assert admitted["heston"]["route"] == "pde"
    assert admitted["heston"]["aggregate_common_scrambles"] == common
    assert admitted["heston"]["delta_bias"]["aggregate_common_scrambles"] == common
    assert stale["heston"]["route"] == "excluded_greek_unresolved"
    assert stale["heston"]["sampling_complete"] is False


def test_production_decision_rejects_an_incomplete_regime_matrix():
    module = _load()
    anchors = [
        {"name": name, "status": "PASS"} for name in module.REQUIRED_ANCHOR_NAMES
    ]
    decisions = module.make_decisions(
        [_passing_cell(module, "heston")],
        anchors,
        quick=False,
        variants=["heston"],
    )

    assert decisions["heston"]["route"] == "excluded_greek_unresolved"
    assert decisions["heston"]["evidence_complete"] is False
    assert decisions["heston"]["missing_cases"]


def test_checkpoint_resume_is_configuration_locked(tmp_path):
    module = _load()
    module._write_checkpoint(
        tmp_path,
        "heston__ordinary_full",
        run_configuration_sha256="a" * 64,
        kind="cell",
        evidence={"status": "PASS"},
    )

    assert module._load_checkpoint(
        tmp_path,
        "heston__ordinary_full",
        run_configuration_sha256="a" * 64,
        kind="cell",
    ) == {"status": "PASS"}
    try:
        module._load_checkpoint(
            tmp_path,
            "heston__ordinary_full",
            run_configuration_sha256="b" * 64,
            kind="cell",
        )
    except ValueError as exc:
        assert "configuration mismatch" in str(exc)
    else:
        raise AssertionError("mismatched checkpoint must fail closed")


def test_markdown_exposes_finite_bump_and_exclusion_semantics():
    module = _load()
    payload = {
        "quick": True,
        "seed": 1,
        "paths_per_batch": 8,
        "batches": 2,
        "policy": {"hedge_inception_spot": 4_532.52},
        "decisions": {
            "heston": {
                "route": "excluded_greek_unresolved",
                "cell_status": "INCONCLUSIVE",
                "delta_bias": {"status": "INCONCLUSIVE"},
                "reason": "uncertain",
            }
        },
        "anchors": [],
        "cells": [],
    }

    report = module.render_markdown(payload)

    assert "excluded_greek_unresolved" in report
    assert "finite-bump hedge exposures" in report
    assert "case-specific target→fine" in report
    assert "case-specific bridge profile" in report


def test_evidence_hash_projection_removes_only_run_clock_metadata():
    module = _load()
    payload = {
        "created_at": "now",
        "elapsed_seconds": 10.0,
        "cells": [{"seconds": 2.0, "delta": 0.1}],
        "seed": 7,
    }

    assert module.evidence_projection(payload) == {
        "cells": [{"delta": 0.1}],
        "seed": 7,
    }


def test_two_level_control_combines_inside_each_scramble():
    module = _load()

    def result(rows, key):
        rows = np.asarray(rows, dtype=float)
        covariance = np.cov(rows, rowvar=False, ddof=1)
        return module.PairedRQMCGreeksResult(
            price=float(rows[:, 1].mean()),
            price_std_error=0.0,
            delta=float(rows[:, 3].mean()),
            delta_std_error=0.0,
            gamma=float(rows[:, 4].mean()),
            gamma_std_error=0.0,
            spot=100.0,
            relative_bump=0.01,
            absolute_bump=1.0,
            paths_per_batch=8,
            batches_used=rows.shape[0],
            total_unique_paths=8 * rows.shape[0],
            total_path_valuations=24 * rows.shape[0],
            randomization_key=key,
            batch_estimates=rows,
            covariance=covariance,
        )

    slv = result([[1, 2, 3, 1, 2], [2, 3, 4, 2, 3]], "slv")
    low = result([[0, 1, 2, 0.5, 1], [1, 2, 3, 1, 1.5]], "low")
    high = result([[3, 4, 5, 0.2, 0.3], [4, 5, 6, 0.4, 0.5]], "high")

    combined = module.combine_two_level_control(slv, low, high)

    expected = slv.batch_estimates - low.batch_estimates + high.batch_estimates
    np.testing.assert_allclose(combined.batch_estimates, expected)
    assert combined.delta == np.mean(expected[:, 3])
    assert combined.gamma == np.mean(expected[:, 4])


def test_two_level_control_groups_disjoint_high_scrambles():
    module = _load()

    def result(rows, key):
        rows = np.asarray(rows, dtype=float)
        covariance = np.cov(rows, rowvar=False, ddof=1)
        return module.PairedRQMCGreeksResult(
            price=float(rows[:, 1].mean()),
            price_std_error=0.0,
            delta=float(rows[:, 3].mean()),
            delta_std_error=0.0,
            gamma=float(rows[:, 4].mean()),
            gamma_std_error=0.0,
            spot=100.0,
            relative_bump=0.01,
            absolute_bump=1.0,
            paths_per_batch=8,
            batches_used=rows.shape[0],
            total_unique_paths=8 * rows.shape[0],
            total_path_valuations=24 * rows.shape[0],
            randomization_key=key,
            batch_estimates=rows,
            covariance=covariance,
        )

    primary = result([[1, 2, 3, 1, 2], [2, 3, 4, 2, 3]], "primary")
    low = result([[0, 1, 2, 0.5, 1], [1, 2, 3, 1, 1.5]], "low")
    high = result(
        [
            [3, 4, 5, 0.2, 0.3],
            [5, 6, 7, 0.4, 0.5],
            [4, 5, 6, 0.6, 0.7],
            [6, 7, 8, 0.8, 0.9],
        ],
        "high",
    )

    combined = module.combine_two_level_control(
        primary, low, high, high_batches_per_low=2
    )

    grouped_high = high.batch_estimates.reshape(2, 2, 5).mean(axis=1)
    expected = primary.batch_estimates - low.batch_estimates + grouped_high
    np.testing.assert_allclose(combined.batch_estimates, expected)
    assert combined.batches_used == 2


def test_embedded_conditional_control_has_zero_incremental_work():
    module = _load()
    rows = np.asarray([[1, 2, 3, 1, 2], [2, 3, 4, 2, 3]], dtype=float)
    controls = 0.5 * rows
    result = module.PairedRQMCGreeksResult(
        price=float(rows[:, 1].mean()),
        price_std_error=0.0,
        delta=float(rows[:, 3].mean()),
        delta_std_error=0.0,
        gamma=float(rows[:, 4].mean()),
        gamma_std_error=0.0,
        spot=100.0,
        relative_bump=0.01,
        absolute_bump=1.0,
        paths_per_batch=8,
        batches_used=2,
        total_unique_paths=16,
        total_path_valuations=48,
        randomization_key="primary",
        batch_estimates=rows,
        covariance=np.cov(rows, rowvar=False, ddof=1),
        control_batch_estimates=controls,
    )

    embedded = module.extract_embedded_conditional_control(result)

    np.testing.assert_array_equal(embedded.batch_estimates, controls)
    assert embedded.total_unique_paths == 0
    assert embedded.total_path_valuations == 0
    assert embedded.control_batch_estimates is None


def test_serialized_paired_result_is_recomputed_and_tamper_checked():
    module = _load()
    rows = np.asarray(
        [[1, 2, 3, 1, 2], [2, 3, 4, 2, 3], [3, 4, 5, 3, 4]],
        dtype=float,
    )
    covariance = np.cov(rows, rowvar=False, ddof=1)
    errors = np.sqrt(np.diag(covariance) / rows.shape[0])
    payload = {
        "price": float(rows[:, 1].mean()),
        "price_std_error": float(errors[1]),
        "delta": float(rows[:, 3].mean()),
        "delta_std_error": float(errors[3]),
        "gamma": float(rows[:, 4].mean()),
        "gamma_std_error": float(errors[4]),
        "spot": 100.0,
        "relative_bump": 0.01,
        "absolute_bump": 1.0,
        "paths_per_batch": 8,
        "batches_used": 3,
        "total_unique_paths": 24,
        "total_path_valuations": 72,
        "randomization_key": "saved",
        "batch_estimates": rows.tolist(),
        "covariance": covariance.tolist(),
    }

    restored = module.paired_result_from_serialized(
        payload, randomization_label="checkpoint"
    )

    np.testing.assert_array_equal(restored.batch_estimates, rows)
    assert restored.randomization_key == "checkpoint(saved)"
    payload["gamma"] += 0.1
    with np.testing.assert_raises(ValueError):
        module.paired_result_from_serialized(payload, randomization_label="checkpoint")


def test_grouped_multilevel_components_preserve_pairing_and_disjoint_rows():
    module = _load()

    def result(rows, key, *, zero_work=False):
        rows = np.asarray(rows, dtype=float)
        covariance = np.cov(rows, rowvar=False, ddof=1)
        errors = np.sqrt(np.diag(covariance) / rows.shape[0])
        return module.PairedRQMCGreeksResult(
            price=float(rows[:, 1].mean()),
            price_std_error=float(errors[1]),
            delta=float(rows[:, 3].mean()),
            delta_std_error=float(errors[3]),
            gamma=float(rows[:, 4].mean()),
            gamma_std_error=float(errors[4]),
            spot=100.0,
            relative_bump=0.01,
            absolute_bump=1.0,
            paths_per_batch=8,
            batches_used=rows.shape[0],
            total_unique_paths=0 if zero_work else 8 * rows.shape[0],
            total_path_valuations=0 if zero_work else 24 * rows.shape[0],
            randomization_key=key,
            batch_estimates=rows,
            covariance=covariance,
        )

    low = result(
        [
            [1, 2, 3, 1, 2],
            [2, 3, 4, 2, 3],
            [3, 4, 5, 3, 4],
            [4, 5, 6, 4, 5],
        ],
        "low",
    )
    low_control = result(0.5 * low.batch_estimates, "low-control", zero_work=True)
    high = result([[10, 20, 30, 4, 6], [20, 30, 40, 6, 8]], "high")

    combined = module.combine_grouped_rqmc_components(
        ((1.0, low), (-1.0, low_control), (1.0, high)),
        output_batches=2,
        estimator_label="three-level",
    )

    expected = (
        low.batch_estimates.reshape(2, 2, 5).mean(axis=1)
        - low_control.batch_estimates.reshape(2, 2, 5).mean(axis=1)
        + high.batch_estimates
    )
    np.testing.assert_allclose(combined.batch_estimates, expected)
    assert combined.batches_used == 2
    assert combined.total_unique_paths == 48
    assert combined.total_path_valuations == 144


def test_stage16_parent_controls_match_stage11_and_schema12_router():
    module = _load()
    import importlib.util

    def load_path(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        assert spec.loader is not None
        spec.loader.exec_module(loaded)
        return loaded

    stage11 = load_path(
        "mo_pde_convergence_gate_11_controls",
        ROOT / "example/mo_volmodels/11_pde_convergence_gate.py",
    )
    stage12 = load_path(
        "mo_snowball_volmodel_backtest_12_controls",
        ROOT / "example/mo_volmodels/12_snowball_volmodel_backtest.py",
    )
    stage17 = load_path(
        "mo_adi_slv_aggregate_certification_17_controls",
        ROOT / "example/mo_volmodels/17_adi_slv_aggregate_certification.py",
    )

    assert module.PRODUCTION_ENGINE_CONTROLS == (
        stage11.ADI_2D_PRODUCTION_ENGINE_CONTROLS
    )
    assert module.PRODUCTION_ENGINE_CONTROLS == (
        stage12.ADI_2D_PRODUCTION_ENGINE_CONTROLS
    )
    assert module.SCHEMA_VERSION == stage17.PARENT_SCHEMA_VERSION
    assert stage17.SCHEMA_VERSION == stage12.ADI_GREEK_DECISION_SCHEMA_VERSION


def test_payload_validator_rejects_quick_pde_admission():
    module = _load()
    payload = {
        "schema_version": module.SCHEMA_VERSION,
        "study": "adi_2d_snowball_greek_certification",
        "certification_mode": module.CERTIFICATION_MODE_FULL,
        "batches": 4,
        "quick": True,
        "policy": {"hedge_inception_spot": 4_532.52},
        "cells": [],
        "decisions": {"heston": {"route": "pde"}},
    }

    try:
        module.validate_payload(payload)
    except ValueError as exc:
        assert "quick evidence" in str(exc)
    else:
        raise AssertionError("quick PDE admission must fail closed")


def test_payload_validator_binds_variant_before_qe_profile_check():
    module = _load()
    payload = {
        "schema_version": module.SCHEMA_VERSION,
        "study": "adi_2d_snowball_greek_certification",
        "certification_mode": module.CERTIFICATION_MODE_FULL,
        "batches": 4,
        "quick": False,
        "policy": {"hedge_inception_spot": 4_532.52},
        "cells": [
            {
                "variant": "heston_slv",
                "case": {"name": "low_feller"},
                "status": module.EquivalenceStatus.PASS.value,
                "variance_operator": {"monotone": True},
                "reference": {
                    "primary": "fine",
                    "target_substeps_per_interval": 6,
                    "fine_substeps_per_interval": 12,
                    "target": {"randomization_key": "test"},
                },
            }
        ],
        "decisions": {"heston_slv": {"route": "excluded_greek_unresolved"}},
    }

    with pytest.raises(ValueError, match="invalid QE-M refinement contract"):
        module.validate_payload(payload)


def test_reference_treatment_descriptor_matches_profiles():
    """Schema-13: the recorded treatment must match the profile that ran it."""
    module = _load()
    for variant, profiles in (
        ("heston_slv", module.SLV_SPOT_BRIDGE_PROFILE_BY_CASE),
        ("heston", module.HESTON_SPOT_BRIDGE_PROFILE_BY_CASE),
    ):
        for case_name, profile in profiles.items():
            descriptor = module.reference_treatment_descriptor(variant, case_name)
            assert descriptor["bridge_strata"] == profile["strata"]
            assert descriptor["bridge_dimensions"] == profile["dimensions"]
            assert descriptor["control"] in ("none", "cross_fitted")


def test_task7_treatments_are_applied_to_the_three_untreated_cells():
    """The 2026-08-10 decision matrix: bridge8 ships on exactly these cells."""
    module = _load()
    for case_name in ("ordinary_full", "ordinary_decayed", "sigma_collapse"):
        assert module.SLV_SPOT_BRIDGE_PROFILE_BY_CASE[case_name]["dimensions"] == 8
    # low_feller keeps its measured direct estimator; near_ki already treated.
    assert module.SLV_SPOT_BRIDGE_PROFILE_BY_CASE["low_feller"]["dimensions"] == 8
    assert module.SLV_SPOT_BRIDGE_PROFILE_BY_CASE["near_ki"]["dimensions"] == 8
    # Untouched cells stay on the single-factor profile.
    assert module.SLV_SPOT_BRIDGE_PROFILE_BY_CASE["near_ko"]["dimensions"] == 1
    assert module.SLV_SPOT_BRIDGE_PROFILE_BY_CASE["near_expiry"]["dimensions"] == 1


def test_gate_driven_levels_stop_early_and_bank_the_prefix_they_stopped_on():
    """The loop must stop on the gate and keep exactly the batches it judged.

    A chunked run may not bank more evidence than the decision rested on, nor
    less: the recorded reference has to be the one the verdict was computed
    from, or the certificate describes a run that never happened.
    """
    module = _load()

    calls = []

    class _Level:
        """Deterministic stand-in whose batch k depends only on k."""

        def __init__(self, first_batch, count, offset):
            rows = np.array(
                [
                    [0.0, 0.0, 0.0, 1.0 + offset, 2.0 + offset]
                    for _ in range(first_batch, first_batch + count)
                ],
                dtype=float,
            )
            self.batch_estimates = rows
            self.batch_delta = rows[:, 3]
            self.batch_gamma = rows[:, 4]
            self.batches_used = count
            self.spot = 100.0
            self.relative_bump = 0.01
            self.absolute_bump = 1.0
            self.paths_per_batch = 8
            self.randomization_key = "fake"
            self.control_batch_estimates = None
            self.covariance = np.zeros((5, 5))
            self.total_unique_paths = 8 * count
            self.total_path_valuations = 3 * 8 * count

    def run_level(level, first_batch, count):
        calls.append((level, first_batch, count))
        return _Level(first_batch, count, 0.0 if level == "fine" else 1e-9)

    scale = module.EconomicGreekScale(
        model_spot=100.0,
        hedge_inception_spot=4532.52,
        study_notional=50_000_000.0,
        hedge_multiplier=200.0,
    )
    policy = module.SequentialAdmissionPolicy(
        family_alpha=0.05,
        tests=28,
        min_batches=16,
        aggregate_floor_batches=32,
        planned_batches=32,
        max_batches=256,
    )
    target, fine, record = module.gate_driven_reference_levels(
        run_level=run_level,
        policy=policy,
        scale=scale,
        pde_target={"delta": 1.0, "gamma": 2.0},
        raw_pde_envelopes={"delta": {"total": 0.0}, "gamma": {"total": 0.0}},
        bounds={"delta": 0.5, "gamma": 0.5},
        chunk_batches=32,
        max_batches=256,
    )

    # A zero-dispersion stream sitting on the PDE decides at the first look.
    assert record["batches_banked"] == 32
    assert target.batches_used == fine.batches_used == 32
    assert calls == [("target", 0, 32), ("fine", 0, 32)]
    assert record["stopping_rule"] == "anytime_valid_sequential"
    assert record["policy_sha256"] == policy.sha256()
    assert set(record["decisions"]) == {"delta", "gamma"}
    assert all(d["status"] == "ADMIT" for d in record["decisions"].values())


def test_gate_driven_levels_extend_by_batch_range_without_recomputing():
    """An undecided chunk must extend the run, never restart it.

    Restarting would both waste the prefix and, without prefix invariance,
    silently rewrite banked batches. The call pattern is the observable proof
    that each chunk asks only for batches the run does not already have.
    """
    module = _load()
    from quantark.montecarlo import PairedRQMCGreeksResult

    calls = []

    def run_level(level, first_batch, count):
        calls.append((level, first_batch, count))
        # Dispersed enough that the gate cannot close before the cap.
        rows = np.array(
            [
                [0.0, 0.0, 0.0, 40.0 * ((k % 7) - 3), 40.0 * ((k % 5) - 2)]
                for k in range(first_batch, first_batch + count)
            ],
            dtype=float,
        )
        return PairedRQMCGreeksResult(
            price=0.0,
            price_std_error=0.0,
            delta=float(rows[:, 3].mean()),
            delta_std_error=0.0,
            gamma=float(rows[:, 4].mean()),
            gamma_std_error=0.0,
            spot=100.0,
            relative_bump=0.01,
            absolute_bump=1.0,
            paths_per_batch=8,
            batches_used=count,
            total_unique_paths=8 * count,
            total_path_valuations=3 * 8 * count,
            randomization_key="fake",
            batch_estimates=rows,
            covariance=np.zeros((5, 5)),
        )

    policy = module.SequentialAdmissionPolicy(
        family_alpha=0.05,
        tests=28,
        min_batches=16,
        aggregate_floor_batches=32,
        planned_batches=32,
        max_batches=96,
    )
    scale = module.EconomicGreekScale(
        model_spot=100.0,
        hedge_inception_spot=4532.52,
        study_notional=50_000_000.0,
        hedge_multiplier=200.0,
    )
    target, fine, record = module.gate_driven_reference_levels(
        run_level=run_level,
        policy=policy,
        scale=scale,
        pde_target={"delta": 0.0, "gamma": 0.0},
        raw_pde_envelopes={"delta": {"total": 0.0}, "gamma": {"total": 0.0}},
        bounds={"delta": 0.5, "gamma": 0.5},
        chunk_batches=32,
        max_batches=96,
    )

    # Three consecutive, non-overlapping chunks per level, then the cap.
    assert calls == [
        ("target", 0, 32),
        ("fine", 0, 32),
        ("target", 32, 32),
        ("fine", 32, 32),
        ("target", 64, 32),
        ("fine", 64, 32),
    ]
    assert record["batches_banked"] == 96
    assert target.batches_used == fine.batches_used == 96
    assert target.batch_estimates.shape == (96, 5)


def test_sequential_policy_is_declared_per_cell_and_capped_by_the_allocation():
    """The policy is built before any batch is priced, and cannot overspend.

    The cap is the cell's own frozen allocation, which is what makes gate-driven
    stopping strictly non-regressive on cost: worst case it spends exactly what
    the fixed run would have.
    """
    module = _load()
    args = SimpleNamespace(
        sequential=True,
        sequential_chunk_batches=128,
        sequential_margin=0.0,
        sequential_family_alpha=0.05,
    )

    policy = module.build_sequential_policy(args, "heston", "low_feller", cap=1024)
    assert policy is not None
    assert policy.max_batches == 1024
    assert policy.margin_fraction == 0.0
    # 7 regimes x 2 variants x 2 greeks, declared from the matrix.
    assert policy.tests == 28
    assert policy.aggregate_floor_batches == module.AMENDMENT_AGGREGATE_BATCHES
    assert policy.first_decidable_batch == module.AMENDMENT_AGGREGATE_BATCHES

    # A cell whose allocation is below the cohort floor cannot be asked for more
    # batches than it has.
    small = module.build_sequential_policy(args, "heston_slv", "near_ko", cap=64)
    assert small is not None
    assert small.max_batches == 64
    assert small.first_decidable_batch <= 64
    assert small.planned_batches <= 64


def test_every_exempt_symbol_still_exists():
    """A renamed non-numerical symbol must break loudly, not widen the digest."""
    module = _load()
    source = Path(module.__file__).read_text()
    top_level = {
        node.name
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    missing = sorted(set(module.NON_NUMERICAL_SYMBOLS) - top_level)
    assert not missing, f"exempt symbols no longer defined: {missing}"
    # And the digest itself must be computable, which is where a stale list bites.
    assert len(module.numerical_implementation_sha256()) == 64


def test_the_numerical_projection_ignores_validation_but_not_arithmetic():
    """The whole point: validation edits are free, arithmetic edits are not."""
    module = _load()
    source = Path(module.__file__).read_text()
    projected = project_source(source, exempt=module.NON_NUMERICAL_SYMBOLS)
    # Validation bodies are gone...
    assert "certification does not match the live numerical projection" not in projected
    assert "def validate_payload" not in projected
    assert "def main(" not in projected
    assert "def build_sequential_policy" not in projected
    # ...while the arithmetic that produces cell numbers remains.
    for kept in (
        "def certify_case",
        "def paired_mc_reference",
        "def build_slv_multilevel_reference",
        "def gate_driven_reference_levels",
        "def deterministic_anchors",
        "def make_pde_engine",
    ):
        assert kept in projected, f"{kept} must stay in the numerical projection"
    # Module-level plan constants stay, so a changed allocation invalidates.
    assert "PRODUCTION_HESTON_BATCHES_BY_CASE" in projected


def test_cell_plan_projection_isolates_one_cell():
    """A cell's plan must not carry other cells' plans or the fleet's selection."""
    module = _load()
    run_configuration = {
        "schema_version": module.SCHEMA_VERSION,
        "certification_mode": module.CERTIFICATION_MODE_FULL,
        "quick": False,
        "skip_anchors": False,
        "runtime_environment": {"numpy_version": "1.26.4"},
        "production_engine_controls": module.PRODUCTION_ENGINE_CONTROLS,
        "reference_seeds": {"heston": 11},
        "cases": [{"name": "near_ko", "spot": 100.0}, {"name": "low_feller"}],
        "sampling_by_variant": {
            "heston": {
                "paths_per_batch": 4096,
                "batches_by_case": {"near_ko": 1024, "low_feller": 1024},
                "primary_batches_by_case": {"near_ko": 1024, "low_feller": 1024},
            }
        },
        "spot_bump": module.SPOT_BUMP,
        "full_bump_ladder": list(module.FULL_BUMP_LADDER),
        "stochastic_component_confidence": module.STOCHASTIC_COMPONENT_CONFIDENCE,
        "hedge_inception_spot": 4532.52,
        "heston_spot_bridge_profile_by_case": {
            "near_ko": {"strata": 4, "dimensions": 8},
            "low_feller": {"strata": 1, "dimensions": 1},
        },
        "qe_substeps_by_variant_case": {"heston": {"near_ko": {"target": 4}}},
        "rqmc_batch_workers_by_variant_case": {"heston": {"near_ko": 4}},
        "sequential_stopping": {
            "enabled": True,
            "chunk_batches": 128,
            "margin_fraction": 0.0,
            "family_alpha": 0.05,
            "excluded_variant_cases": ["heston/near_ki", "heston_slv/near_ki"],
        },
        "cell_workers": 2,
    }
    plan = module.cell_plan_projection("heston", "near_ko", run_configuration)
    flat = json.dumps(plan, sort_keys=True, default=str)
    # Its own plan is present.
    assert plan["case"]["name"] == "near_ko"
    assert plan["batches_by_case"] == 1024
    assert plan["spot_bridge_profile"] == {"strata": 4, "dimensions": 8}
    assert plan["stopping"]["enabled"] is True
    # Another cell's plan is not, nor is fleet-wide scheduling.
    assert "low_feller" not in flat
    assert "cell_workers" not in plan

    # An excluded cell records that it may not stop, with no policy parameters.
    run_configuration["cases"].append({"name": "near_ki"})
    run_configuration["sampling_by_variant"]["heston"]["batches_by_case"][
        "near_ki"
    ] = 2048
    excluded = module.cell_plan_projection("heston", "near_ki", run_configuration)
    assert excluded["stopping"] == {"enabled": False}

    # A case absent from the configuration is refused, not silently defaulted.
    with pytest.raises(ValueError, match="not in the run configuration"):
        module.cell_plan_projection("heston", "absent_case", run_configuration)


def test_a_checkpoint_without_an_identity_cannot_be_reused(tmp_path):
    """Pre-identity checkpoints cannot state what they depended on."""
    module = _load()
    module._write_checkpoint(
        tmp_path,
        "heston__near_ko",
        run_configuration_sha256="cfg",
        kind="cell",
        evidence={"status": "PASS"},
    )
    # Fleet-wide artifacts (the shared anchors) still resume on the config hash.
    assert module._load_checkpoint(
        tmp_path, "heston__near_ko", run_configuration_sha256="cfg", kind="cell"
    ) == {"status": "PASS"}
    # ...but a cell asked for by identity is refused.
    with pytest.raises(ValueError, match="predates cell identities"):
        module._load_checkpoint(
            tmp_path,
            "heston__near_ko",
            run_configuration_sha256="cfg",
            kind="cell",
            identity_sha256="a" * 64,
        )


def test_a_cell_resumes_on_its_identity_not_the_fleet_configuration(tmp_path):
    """The whole point: an unrelated fleet change must not invalidate a cell."""
    module = _load()
    module._write_checkpoint(
        tmp_path,
        "heston__near_ko",
        run_configuration_sha256="original-fleet-hash",
        kind="cell",
        evidence={"status": "PASS"},
        identity_sha256="a" * 64,
    )
    # Different fleet hash, same cell identity -> reused.
    assert module._load_checkpoint(
        tmp_path,
        "heston__near_ko",
        run_configuration_sha256="a-totally-different-fleet-hash",
        kind="cell",
        identity_sha256="a" * 64,
    ) == {"status": "PASS"}
    # Same fleet hash, different identity -> refused.
    with pytest.raises(ValueError, match="cell identity mismatch"):
        module._load_checkpoint(
            tmp_path,
            "heston__near_ko",
            run_configuration_sha256="original-fleet-hash",
            kind="cell",
            identity_sha256="b" * 64,
        )


def test_a_multilevel_high_control_may_not_stop_early():
    """The Heston cell feeding the telescoping estimator is declared, not sized.

    The multilevel SLV cell is excluded from stopping because its telescoping
    weights require exactly its declared batch count. The SAME argument covers
    the Heston cell of that case, because it enters the very same estimator as
    ``heston_high_reference``: excluding only the consumer left its control free
    to stop, and the 35.5h fleet duly stopped ``heston/near_ki`` at 1664 of 2048
    and fed the truncated mean into the SLV estimator with weight 0.85.

    That truncation is not merely imprecise, it is selected: the cell stopped
    because ITS OWN delta and gamma gate closed, so the stop time is a function
    of the estimates that become the control. An anytime-valid sequence keeps
    that cell's own interval honest under optional stopping; it says nothing
    about a downstream estimator consuming the stopped mean at a fixed weight.
    """
    module = _load()
    args = SimpleNamespace(
        sequential=True,
        sequential_chunk_batches=128,
        sequential_margin=0.0,
        sequential_family_alpha=0.05,
    )
    for case_name in sorted(module.SLV_MULTILEVEL_CASES):
        for variant in ("heston", "heston_slv"):
            assert (
                module.build_sequential_policy(args, variant, case_name, cap=2048)
                is None
            ), f"{variant}/{case_name} must spend its declared allocation"

    # Non-multilevel cases are unaffected: the exclusion must be narrow, or
    # gate-driven stopping quietly stops being worth anything.
    assert module.build_sequential_policy(args, "heston", "near_ko", cap=1024)


def test_sequential_stopping_is_opt_in_and_never_touches_the_multilevel_cell():
    """The multilevel SLV cell is declared, not sized.

    ``build_slv_multilevel_reference`` requires exact equality with its declared
    batch count, so its telescoping level weights cannot survive an early stop.
    """
    module = _load()
    off = SimpleNamespace(
        sequential=False,
        sequential_chunk_batches=128,
        sequential_margin=0.0,
        sequential_family_alpha=0.05,
    )
    on = SimpleNamespace(
        sequential=True,
        sequential_chunk_batches=128,
        sequential_margin=0.05,
        sequential_family_alpha=0.05,
    )

    assert module.build_sequential_policy(off, "heston", "low_feller", cap=1024) is None
    for case_name in module.SLV_MULTILEVEL_CASES:
        assert (
            module.build_sequential_policy(on, "heston_slv", case_name, cap=256)
            is None
        )
    # This line previously asserted that the Heston cell of a multilevel case IS
    # eligible to stop. That was the defect, written down as if intended: the
    # cell feeds the telescoping estimator as heston_high_reference, so its count
    # belongs to the same contract. See
    # test_a_multilevel_high_control_may_not_stop_early.
    assert module.build_sequential_policy(on, "heston", "near_ki", cap=2048) is None
    # Ordinary cells are still eligible, so the exclusion stays narrow.
    assert module.build_sequential_policy(on, "heston", "near_ko", cap=1024)


def _stopping_record(*, banked, declared, chunk=128, margin=0.0, alpha=0.05,
                     status="ADMIT"):
    return {
        "batches_banked": banked,
        "chunk_batches": chunk,
        "policy": {"max_batches": declared, "margin_fraction": margin,
                   "family_alpha": alpha},
        "decisions": {
            greek: {"status": status, "batches_used": banked}
            for greek in ("delta", "gamma")
        },
    }


def _run_config(enabled=True, chunk=128, margin=0.0, alpha=0.05):
    return {
        "sequential_stopping": (
            {"enabled": True, "chunk_batches": chunk, "margin_fraction": margin,
             "family_alpha": alpha}
            if enabled
            else {"enabled": False}
        )
    }


def test_a_short_run_is_admissible_only_when_a_decision_explains_it():
    """Banking fewer batches than declared must be EXPLAINED, not merely allowed.

    The fixed-allocation equality is what proved nobody quietly ran short. Under
    gate-driven stopping the replacement has to be stronger, not weaker: the
    shortfall is legitimate only if every greek reached a decision under the
    policy the run declared, and the policy's cap equals the allocation so the
    stop could never overspend.
    """
    module = _load()
    common = dict(variant="heston", case_name="low_feller", declared_batches=1024)

    # Decided early: admissible, and the banked count governs the arrays.
    assert (
        module._admissible_batch_count(
            {"sequential_stopping": _stopping_record(banked=256, declared=1024)},
            _run_config(),
            **common,
        )
        == 256
    )
    # Ran short with nothing decided: refused.
    with pytest.raises(ValueError, match="undecided"):
        module._admissible_batch_count(
            {
                "sequential_stopping": _stopping_record(
                    banked=256, declared=1024, status="EXHAUSTED"
                )
            },
            _run_config(),
            **common,
        )
    # Exhausting the full allocation is fine whatever the statuses say.
    assert (
        module._admissible_batch_count(
            {
                "sequential_stopping": _stopping_record(
                    banked=1024, declared=1024, status="EXHAUSTED"
                )
            },
            _run_config(),
            **common,
        )
        == 1024
    )


@pytest.mark.parametrize(
    "record,config,match",
    [
        # Cap must equal the declared allocation: otherwise a cell could be
        # stopped under a policy allowed to spend more than the configuration.
        (_stopping_record(banked=256, declared=4096), _run_config(), "cap"),
        # The policy must be the declared one, not a looser variant.
        (_stopping_record(banked=256, declared=1024, margin=0.5), _run_config(),
         "margin_fraction"),
        (_stopping_record(banked=256, declared=1024, chunk=64), _run_config(),
         "chunk size"),
        # Never more than declared.
        (_stopping_record(banked=2048, declared=1024), _run_config(), "outside"),
        # A stopped cell in a run that never declared stopping.
        (_stopping_record(banked=256, declared=1024), _run_config(enabled=False),
         "does not declare"),
    ],
)
def test_stopping_records_that_do_not_match_the_declared_run_are_refused(
    record, config, match
):
    module = _load()
    with pytest.raises(ValueError, match=match):
        module._admissible_batch_count(
            {"sequential_stopping": record},
            config,
            variant="heston",
            case_name="low_feller",
            declared_batches=1024,
        )


def test_a_declared_sequential_run_cannot_bank_a_cell_without_a_decision():
    """Silence is not consent: a missing record in a sequential run is refused.

    Only the multilevel SLV cell may legitimately carry no record, because it is
    excluded from stopping by construction.
    """
    module = _load()
    with pytest.raises(ValueError, match="records no stopping decision"):
        module._admissible_batch_count(
            {}, _run_config(), variant="heston", case_name="low_feller",
            declared_batches=1024,
        )
    for case_name in module.SLV_MULTILEVEL_CASES:
        assert (
            module._admissible_batch_count(
                {}, _run_config(), variant="heston_slv", case_name=case_name,
                declared_batches=256,
            )
            == 256
        )
    # And a fixed-allocation run is unaffected.
    assert (
        module._admissible_batch_count(
            {}, _run_config(enabled=False), variant="heston",
            case_name="low_feller", declared_batches=1024,
        )
        == 1024
    )
