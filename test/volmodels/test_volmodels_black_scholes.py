import math
import pytest
from quantark.util.exceptions import ValidationError, NumericalError
from quantark.volmodels.black_scholes import (
    bs_call_price, bs_put_price, bs_vega, implied_vol_call,
)


def test_atm_call_known_value():
    assert bs_call_price(100.0, 100.0, 1.0, 0.20, 0.0, 0.0) == pytest.approx(7.9656, abs=1e-3)


def test_put_call_parity():
    s, k, t, sig, r, q = 100.0, 110.0, 0.75, 0.25, 0.03, 0.01
    lhs = bs_call_price(s, k, t, sig, r, q) - bs_put_price(s, k, t, sig, r, q)
    rhs = s * math.exp(-q * t) - k * math.exp(-r * t)
    assert lhs == pytest.approx(rhs, abs=1e-10)


def test_zero_time_returns_intrinsic():
    assert bs_call_price(110.0, 100.0, 0.0, 0.2, 0.0, 0.0) == pytest.approx(10.0, abs=1e-12)


def test_negative_inputs_raise():
    with pytest.raises(ValidationError):
        bs_call_price(100.0, 100.0, -1.0, 0.2, 0.0, 0.0)
    with pytest.raises(ValidationError):
        bs_call_price(100.0, 100.0, 1.0, -0.2, 0.0, 0.0)


def test_implied_vol_round_trip():
    s, k, t, sig, r, q = 100.0, 95.0, 0.5, 0.27, 0.02, 0.0
    price = bs_call_price(s, k, t, sig, r, q)
    assert implied_vol_call(s, k, t, price, r, q) == pytest.approx(sig, abs=1e-6)


def test_implied_vol_out_of_bounds_raises():
    with pytest.raises(NumericalError):
        implied_vol_call(100.0, 100.0, 1.0, 200.0, 0.0, 0.0)  # price > spot


def test_vega_positive():
    assert bs_vega(100.0, 100.0, 1.0, 0.2, 0.0, 0.0) > 0


def test_implied_vol_at_intrinsic_lower_bound_returns_zero():
    s, k, t, r, q = 110.0, 100.0, 1.0, 0.0, 0.0
    intrinsic = s - k  # r=q=0
    assert implied_vol_call(s, k, t, intrinsic, r, q) == 0.0


def test_implied_vol_zero_maturity_raises():
    with pytest.raises(NumericalError):
        implied_vol_call(100.0, 100.0, 0.0, 5.0, 0.0, 0.0)


def test_implied_vol_recovers_high_vol():
    s, k, t, sig, r, q = 100.0, 100.0, 1.0, 1.5, 0.0, 0.0  # 150% vol > default bracket 10? no, but high
    price = bs_call_price(s, k, t, sig, r, q)
    assert implied_vol_call(s, k, t, price, r, q) == pytest.approx(sig, abs=1e-6)
