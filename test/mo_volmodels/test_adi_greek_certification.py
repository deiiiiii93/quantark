import importlib.util
import sys
from pathlib import Path

import numpy as np


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
    return {
        "variant": variant,
        "case": {"name": case_name},
        "status": "PASS",
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
        {"name": name, "status": "PASS"}
        for name in module.REQUIRED_ANCHOR_NAMES
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
        {"name": name, "status": "PASS"}
        for name in module.REQUIRED_ANCHOR_NAMES
    ]
    rows = [
        _passing_cell(module, "heston_slv", case_name=case.name)
        for case in module.certification_cases(quick=False)
    ]
    sampling = {
        "heston_slv": {
            "paths_per_batch": module.PRODUCTION_SLV_PATHS_PER_BATCH,
            "batches": module.PRODUCTION_SLV_BATCHES,
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
    )

    assert admitted["heston_slv"]["route"] == "pde"
    assert stale["heston_slv"]["route"] == "excluded_greek_unresolved"
    assert stale["heston_slv"]["sampling_complete"] is False


def test_production_decision_rejects_an_incomplete_regime_matrix():
    module = _load()
    anchors = [
        {"name": name, "status": "PASS"}
        for name in module.REQUIRED_ANCHOR_NAMES
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


def test_stage16_production_controls_match_stage11_and_stage12():
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

    assert module.PRODUCTION_ENGINE_CONTROLS == (
        stage11.ADI_2D_PRODUCTION_ENGINE_CONTROLS
    )
    assert module.PRODUCTION_ENGINE_CONTROLS == (
        stage12.ADI_2D_PRODUCTION_ENGINE_CONTROLS
    )


def test_payload_validator_rejects_quick_pde_admission():
    module = _load()
    payload = {
        "schema_version": module.SCHEMA_VERSION,
        "study": "adi_2d_snowball_greek_certification",
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
