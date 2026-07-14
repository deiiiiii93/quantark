from types import SimpleNamespace

import pytest

import quantark.volmodels.heston.calibration as calibration_module
from quantark.util.exceptions import NumericalError, ValidationError
from quantark.volmodels.heston import (
    HestonParams,
    MarketOption,
    calibrate_heston,
    heston_call_price,
)


def _options(params: HestonParams):
    spot, rate, carry = 100.0, 0.02, 0.0
    options = [
        MarketOption(
            K=strike,
            T=maturity,
            price=heston_call_price(
                spot, strike, maturity, params, rate, carry
            ),
        )
        for maturity in (0.5, 1.0)
        for strike in (90.0, 100.0, 110.0)
    ]
    return spot, rate, carry, options


def test_hard_feller_calibration_uses_real_constraint_and_preserves_cost():
    true = HestonParams(
        v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5
    )
    spot, rate, carry, options = _options(true)
    initial = HestonParams(
        v0=0.04, kappa=1.0, theta=0.04, sigma=0.9, rho=-0.2
    )
    assert initial.feller_satisfied() is False

    result = calibrate_heston(
        spot,
        options,
        rate,
        carry,
        initial,
        target="price",
        regularize_feller=0.0,
        enforce_feller=True,
        max_nfev=400,
        ftol=1e-9,
    )

    assert result.success is True
    assert result.optimizer == "SLSQP"
    assert result.enforce_feller is True
    assert result.params.feller_satisfied() is True
    assert result.feller_margin >= 0.0
    assert result.feller_penalty_cost == 0.0
    assert result.cost == pytest.approx(result.data_cost)


def test_hard_feller_rejects_constraint_infeasible_bounds():
    true = HestonParams(
        v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5
    )
    spot, rate, carry, options = _options(true)
    initial = HestonParams(
        v0=0.04, kappa=0.15, theta=0.015, sigma=1.5, rho=0.0
    )
    bounds = (
        (0.01, 0.10, 0.01, 1.0, -0.9),
        (0.10, 0.20, 0.02, 2.0, 0.9),
    )

    with pytest.raises(ValidationError, match="no parameters satisfying"):
        calibrate_heston(
            spot,
            options,
            rate,
            carry,
            initial,
            bounds=bounds,
            enforce_feller=True,
        )


def test_hard_feller_rejects_non_boolean_policy():
    true = HestonParams(
        v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5
    )
    spot, rate, carry, options = _options(true)

    with pytest.raises(ValidationError, match="enforce_feller must be a bool"):
        calibrate_heston(
            spot,
            options,
            rate,
            carry,
            true,
            enforce_feller=1,
        )


def test_hard_feller_constraint_violation_fails_closed(monkeypatch):
    true = HestonParams(
        v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5
    )
    spot, rate, carry, options = _options(true)
    violating = [0.04, 1.0, 0.04, 0.9, -0.2]

    monkeypatch.setattr(
        calibration_module,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(
            x=violating,
            success=True,
            message="incorrectly accepted constraint violation",
            nfev=1,
        ),
    )

    with pytest.raises(NumericalError, match="constraint-violating"):
        calibrate_heston(
            spot,
            options,
            rate,
            carry,
            true,
            enforce_feller=True,
        )
