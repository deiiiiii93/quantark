"""One-call Heston calibration tests (spec WP4.6)."""
import json

import pytest

from quantark.volmodels.heston import HestonParams
from quantark.volmodels.heston.from_quotes import calibrate_heston_from_quotes

from dcn_fixtures import synthetic_cleaned_set


@pytest.fixture(scope="module")
def calibration():
    cleaned, rate_curve, carry_curve, true_params = synthetic_cleaned_set()
    return calibrate_heston_from_quotes(cleaned, rate_curve, carry_curve), \
        true_params


def test_round_trip_recovers_own_model(calibration):
    calib, _ = calibration
    assert calib.residual_report.rmse_iv < 0.005


def test_config_records_setup(calibration):
    calib, _ = calibration
    for key in (
        "initial",
        "bounds",
        "target",
        "weights",
        "method",
        "feller_policy",
        "stopping",
    ):
        assert key in calib.config
    assert calib.config["method"] == "lewis"
    assert calib.config["feller_policy"] == {
        "enforce_feller": False,
        "regularize_feller": 1e-4,
    }


def test_to_dict_json_safe(calibration):
    calib, _ = calibration
    payload = calib.to_dict()
    json.dumps(payload)
    assert payload["objective"]["total_cost"] == pytest.approx(
        payload["objective"]["data_cost"]
        + payload["objective"]["feller_penalty_cost"]
    )
    assert payload["feller"]["margin"] == pytest.approx(
        payload["feller"]["two_kappa_theta"]
        - payload["feller"]["sigma_squared"]
    )


def test_deterministic():
    cleaned, rate_curve, carry_curve, _ = synthetic_cleaned_set()
    c1 = calibrate_heston_from_quotes(cleaned, rate_curve, carry_curve)
    c2 = calibrate_heston_from_quotes(cleaned, rate_curve, carry_curve)
    names = ("v0", "kappa", "theta", "sigma", "rho")
    assert tuple(getattr(c1.params, name) for name in names) == pytest.approx(
        tuple(getattr(c2.params, name) for name in names),
        rel=1e-9,
        abs=1e-9,
    )


def test_hard_feller_policy_and_diagnostics_are_serialized():
    true_params = HestonParams(
        v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5
    )
    cleaned, rate_curve, carry_curve, _ = synthetic_cleaned_set(true_params)
    calibration = calibrate_heston_from_quotes(
        cleaned,
        rate_curve,
        carry_curve,
        config={
            "enforce_feller": True,
            "regularize_feller": 0.0,
            "max_nfev": 400,
            "ftol": 1e-8,
        },
    )
    payload = calibration.to_dict()

    assert calibration.result.success is True
    assert calibration.params.feller_satisfied() is True
    assert payload["optimizer"] == "SLSQP"
    assert payload["config"]["feller_policy"] == {
        "enforce_feller": True,
        "regularize_feller": 0.0,
    }
    assert payload["feller"]["enforced"] is True
    assert payload["feller"]["satisfied"] is True
    assert payload["feller"]["margin"] >= 0.0
    assert payload["objective"]["feller_penalty_cost"] == 0.0
