import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


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
