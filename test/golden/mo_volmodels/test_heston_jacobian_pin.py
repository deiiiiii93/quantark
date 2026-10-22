"""Golden pin: the Heston model Jacobian on the committed latest MO market fixture.

Fails when the fixed-economic condition number (20.17) or the active-bound pattern moves,
i.e. when quantark's Heston implied vols move. The numbers come from the study's own
Jacobian helper and calibration stage, so this golden test loads them by path on purpose.
Cross-architecture: the 1% relative tolerance absorbs ARM64/x86_64 differences.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "example" / "case_studies" / "mo_volmodels"
sys.path.insert(0, str(EXAMPLE))

import _heston_diagnostics as diagnostics  # noqa: E402
import _mo_common as mo_common  # noqa: E402


def _load_stage04():
    spec = importlib.util.spec_from_file_location(
        "mo_heston_jacobian_pin", EXAMPLE / "04_heston_calibration.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
