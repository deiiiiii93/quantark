"""Focused contracts for MO Heston Jacobian/SVD and bootstrap helpers."""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "example" / "mo_volmodels"
sys.path.insert(0, str(EXAMPLE))

import _heston_diagnostics as diagnostics  # noqa: E402
import _mo_common as mo_common  # noqa: E402


def _load_stage04():
    spec = importlib.util.spec_from_file_location(
        "mo_heston_identification_contract", EXAMPLE / "04_heston_calibration.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bound_aware_jacobian_recovers_linear_map_and_reports_all_svd_policies():
    matrix = np.array(
        [
            [1.0, 0.2, 0.0, 0.1, -0.3],
            [0.0, 1.1, 0.3, -0.2, 0.0],
            [0.1, 0.0, 1.2, 0.4, 0.2],
            [0.2, -0.1, 0.0, 1.3, 0.1],
            [0.0, 0.3, -0.2, 0.1, 1.4],
            [0.4, 0.1, 0.2, 0.0, -0.1],
            [0.2, -0.3, 0.1, 0.2, 0.0],
        ],
        dtype=float,
    )
    parameters = np.array([0.08, 3.0, 0.07, 0.7, -0.05])
    lower = np.array([1e-6, 1e-3, 1e-4, 1e-3, -0.95])
    upper = np.array([0.5, 3.0, 0.5, 0.7, 0.0])

    result = diagnostics.finite_difference_model_jacobian(
        lambda values: matrix @ values + 0.25,
        parameters,
        lower,
        upper,
    )

    assert np.asarray(result["matrix"]) == pytest.approx(matrix, abs=2e-10)
    assert result["shape"] == [7, 5]
    assert result["difference_schemes"]["kappa"] == "backward_second_order"
    assert result["difference_schemes"]["sigma"] == "backward_second_order"
    assert result["difference_schemes"]["v0"] == "central_second_order"
    assert result["active_bounds"]["kappa"]["upper"] is True
    assert result["active_bounds"]["sigma"]["upper"] is True
    assert set(result["svd"]) == {
        "raw",
        "fit_relative",
        "fixed_economic",
        "bound_span",
    }
    assert result["scales"]["fixed_economic"] == {
        "v0": 0.01,
        "kappa": 1.0,
        "theta": 0.01,
        "sigma": 0.1,
        "rho": 0.1,
    }
    expected = np.linalg.svd(matrix, compute_uv=False)
    assert result["svd"]["raw"]["singular_values"] == pytest.approx(expected)
    assert result["svd"]["raw"]["numerical_rank"] == 5
    assert len(result["svd"]["raw"]["right_singular_vectors"]) == 5
    assert result["excludes_feller_penalty"] is True


def test_jacobian_fails_closed_on_nonfinite_perturbation():
    parameters = np.array([0.08, 2.0, 0.07, 0.6, -0.05])
    lower = np.array([1e-6, 1e-3, 1e-4, 1e-3, -0.95])
    upper = np.array([0.5, 3.0, 0.5, 0.7, 0.0])
    calls = 0

    def bad_model(values):
        nonlocal calls
        calls += 1
        output = np.arange(8, dtype=float) + np.sum(values)
        if calls == 2:
            output[0] = np.nan
        return output

    with pytest.raises(ValueError, match="finite"):
        diagnostics.finite_difference_model_jacobian(
            bad_model, parameters, lower, upper
        )


def test_latest_market_fixture_jacobian_is_finite_bound_aware_and_full_shape():
    """One real-data numerical pin beyond the synthetic linear-map contracts."""
    stage04 = _load_stage04()
    surface_path = EXAMPLE / "data" / "mo_iv_surface_latest.json"
    calibration_path = EXAMPLE / "data" / "mo_calib_heston_latest.json"
    raw_surface = json.loads(surface_path.read_text())
    surface = mo_common.prepare_model_surface(raw_surface, iv_smoothing="sabr")
    nodes = stage04._calibration_nodes(float(surface["s0"]), surface["per_expiry"])
    params = stage04.HestonParams(
        **json.loads(calibration_path.read_text())["params"]
    )

    result = diagnostics.finite_difference_model_jacobian(
        lambda values: stage04._model_ivs(
            float(surface["s0"]), nodes, stage04._params_from_vector(values)
        ),
        stage04._params_vector(params),
        stage04.HESTON_BOUNDS[0],
        stage04.HESTON_BOUNDS[1],
    )

    assert result["shape"] == [135, 5]
    assert result["active_bounds"]["kappa"]["upper"] is True
    assert result["active_bounds"]["sigma"]["upper"] is True
    fixed = result["svd"]["fixed_economic"]
    assert fixed["numerical_rank"] == 5
    assert fixed["condition_number"] == pytest.approx(20.17, rel=0.01)
    assert fixed["singular_values"][-1] > 0.0


def test_stratified_exponential_weights_are_seeded_positive_and_normalized():
    strata = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 1.0])
    first = diagnostics.stratified_exponential_weights(strata, seed=271828)
    again = diagnostics.stratified_exponential_weights(strata, seed=271828)
    other = diagnostics.stratified_exponential_weights(strata, seed=271829)

    assert first == pytest.approx(again)
    assert not np.allclose(first, other)
    assert np.all(first > 0.0)
    for maturity in np.unique(strata):
        mask = strata == maturity
        assert float(np.sum(first[mask])) == pytest.approx(float(np.sum(mask)))


def _bound_hits(*, kappa_upper: bool) -> dict:
    return {
        name: {
            "lower": False,
            "upper": bool(kappa_upper and name == "kappa"),
        }
        for name in diagnostics.PARAMETER_NAMES
    }


def test_bootstrap_summary_keeps_failures_quantiles_and_undefined_correlations_json_safe():
    base = {"v0": 0.08, "kappa": 3.0, "theta": 0.07, "sigma": 0.7, "rho": -0.05}
    replicates = [
        {
            "index": 0,
            "success": True,
            "params": dict(base),
            "bound_hits": _bound_hits(kappa_upper=True),
            "feller_satisfied": True,
            "full_sample_rmse_iv": 0.02,
            "bootstrap_weighted_rmse_iv": 0.018,
        },
        {
            "index": 1,
            "success": True,
            "params": {**base, "v0": 0.09, "theta": 0.08, "rho": -0.08},
            "bound_hits": _bound_hits(kappa_upper=True),
            "feller_satisfied": False,
            "full_sample_rmse_iv": 0.021,
            "bootstrap_weighted_rmse_iv": 0.019,
        },
        {
            "index": 2,
            "success": False,
            "failure_type": "RuntimeError",
            "message": "synthetic failure",
        },
    ]

    result = diagnostics.summarize_bootstrap_replicates(
        replicates, requested=3
    )

    assert result["status"] == "partial"
    assert result["successful_replicates"] == 2
    assert result["failed_replicates"] == 1
    assert result["parameter_quantiles"]["v0"]["q50"] == pytest.approx(0.085)
    assert result["bound_hit_rates"]["kappa"]["upper"] == 1.0
    assert result["feller_pass_fraction"] == 0.5
    assert result["sample_covariance"] is not None
    # Kappa and sigma are constant, so their correlation is deliberately undefined.
    assert result["sample_correlation"][1][1] is None
    assert result["failures"] == [replicates[-1]]
    json.dumps(result, allow_nan=False)
