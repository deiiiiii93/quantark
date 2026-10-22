"""DCNOption product tests: derived levels, sign, validation."""
import pytest

from quantark.asset.equity.product.option.dcn_option import DCNDirection
from quantark.util.exceptions import ValidationError

from dcn_fixtures import DCN_A, make_dcn


def test_derived_levels():
    p = make_dcn(DCN_A)
    assert p.coupon_barrier == pytest.approx(0.80 * 6000.0)
    assert p.ko_barrier == pytest.approx(6000.0)
    assert p.ki_barrier == pytest.approx(0.75 * 6000.0)
    assert p.k_loss == pytest.approx(1.10 * 6000.0)  # > S0 must be allowed
    assert p.accrual_per_period == pytest.approx(30.0 / 360.0)
    assert p.direction_sign == +1.0


def test_seller_sign():
    p = make_dcn(DCN_A, direction=DCNDirection.SELLER)
    assert p.direction_sign == -1.0


@pytest.mark.parametrize("field,bad", [
    ("notional", 0.0), ("initial_price", -1.0), ("participation", 0.0),
    ("coupon_rate", -0.01), ("ki_barrier_ratio", 0.85),  # >= coupon barrier
])
def test_validation_rejects(field, bad):
    with pytest.raises(ValidationError):
        make_dcn(DCN_A, **{field: bad})


def test_ki_put_strike_above_one_allowed():
    p = make_dcn(DCN_A, ki_put_strike_ratio=1.5)
    assert p.k_loss == pytest.approx(1.5 * 6000.0)


def test_settlement_before_last_obs_rejected():
    from datetime import datetime
    with pytest.raises(ValidationError):
        make_dcn(DCN_A, settlement_date=datetime(2024, 1, 3))
