"""Cross-fitted control-variate weights (spec WS-V2)."""

import dataclasses

import numpy as np
import pytest

from quantark.montecarlo.control_weights import cross_fitted_control
from quantark.util.exceptions import ValidationError


def _correlated_batches(rho: float, n: int = 400, seed: int = 7):
    rng = np.random.default_rng(seed)
    control = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, np.sqrt(1.0 - rho * rho), n)
    primary = 5.0 + rho * control + noise
    return primary, control


def test_unbiased_mean_preserved():
    primary, control = _correlated_batches(rho=0.9)
    result = cross_fitted_control(primary, control, control_expectation=0.0)
    se = primary.std(ddof=1) / np.sqrt(primary.size)
    assert abs(result.adjusted.mean() - primary.mean()) < 3.0 * se


def test_variance_reduced_when_correlated():
    primary, control = _correlated_batches(rho=0.9)
    result = cross_fitted_control(primary, control, control_expectation=0.0)
    assert result.variance_ratio < 0.35  # 1 - rho^2 = 0.19 plus cross-fit slack


def test_no_gain_when_uncorrelated_but_still_unbiased():
    primary, control = _correlated_batches(rho=0.0)
    result = cross_fitted_control(primary, control, control_expectation=0.0)
    assert 0.8 < result.variance_ratio < 1.3
    se = primary.std(ddof=1) / np.sqrt(primary.size)
    assert abs(result.adjusted.mean() - primary.mean()) < 3.0 * se


def test_weights_are_out_of_fold():
    # A pathological fold-A-only outlier must not contaminate fold A's beta,
    # because fold A's beta is fitted on fold B.
    primary, control = _correlated_batches(rho=0.9, n=100)
    primary = primary.copy()
    primary[0] += 50.0  # index 0 -> fold A under the modulo assignment
    result = cross_fitted_control(primary, control, control_expectation=0.0, folds=2)
    clean_primary, clean_control = _correlated_batches(rho=0.9, n=100)
    clean = cross_fitted_control(
        clean_primary, clean_control, control_expectation=0.0, folds=2
    )
    assert result.weights[0] == pytest.approx(clean.weights[0])


def test_as_dict_is_json_ready():
    primary, control = _correlated_batches(rho=0.5, n=40)
    payload = cross_fitted_control(primary, control, control_expectation=0.0).as_dict()
    assert set(payload) == {"weights", "variance_ratio", "n_batches"}
    assert all(isinstance(w, float) for w in payload["weights"])
    assert payload["n_batches"] == 40
    assert not dataclasses.is_dataclass(payload)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValidationError):
        cross_fitted_control(np.ones(10), np.ones(9), control_expectation=0.0)


def test_rejects_too_few_batches_per_fold():
    with pytest.raises(ValidationError):
        cross_fitted_control(np.ones(3), np.ones(3), control_expectation=0.0, folds=2)
