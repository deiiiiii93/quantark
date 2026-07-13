"""One-call Heston calibration tests (spec WP4.6)."""
import json

import pytest

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
    for key in ("initial", "bounds", "target", "weights", "stopping"):
        assert key in calib.config


def test_to_dict_json_safe(calibration):
    calib, _ = calibration
    json.dumps(calib.to_dict())


def test_deterministic():
    cleaned, rate_curve, carry_curve, _ = synthetic_cleaned_set()
    c1 = calibrate_heston_from_quotes(cleaned, rate_curve, carry_curve)
    c2 = calibrate_heston_from_quotes(cleaned, rate_curve, carry_curve)
    assert c1.params == c2.params
