"""Unit tests for TRS params validation and accrual-calculator strategies."""

from datetime import datetime

import pandas as pd
import pytest

from quantark.util.calendar.business_calendar import Calendar
from quantark.util.exceptions import ValidationError
from quantark.asset.equity.product.swap.trs_params import (
    AccrualType,
    AccrualSide,
    SettleType,
    AssetParams,
    FixLegParams,
    PricingParams,
    TRSParams,
    FloatLegParams,
)
from quantark.asset.equity.engine.cashflow.accrual_calculator import (
    StandardAccrualCalculator,
    MarketValueAccrualCalculator,
    LastMarketValueAccrualCalculator,
    AccrualCalculatorFactory,
)


@pytest.fixture
def cal():
    return Calendar(name="Test")


# ---------------------------------------------------------------------------
# Accrual strategies
# ---------------------------------------------------------------------------
def test_standard_accrual_act_365(cal):
    # 100 days, 10% on 1,000,000 notional, act/365
    val = StandardAccrualCalculator().calculate_accrual(
        1_000_000, "2024-01-01", "2024-04-10", 0.10, cal,
        side="left", day_count_basis="act/365",
    )
    assert val == pytest.approx(1_000_000 * 0.10 * 100 / 365)


def test_standard_accrual_act_360(cal):
    val = StandardAccrualCalculator().calculate_accrual(
        1_000_000, "2024-01-01", "2024-04-10", 0.10, cal,
        side="left", day_count_basis="act/360",
    )
    assert val == pytest.approx(1_000_000 * 0.10 * 100 / 360)


def test_force_no_zero_gives_one_day(cal):
    val = StandardAccrualCalculator().calculate_accrual(
        1_000_000, "2024-01-01", "2024-01-01", 0.10, cal, force_no_zero=True,
    )
    assert val == pytest.approx(1_000_000 * 0.10 * 1 / 365)


def test_market_value_accrual_uses_quantity_times_price(cal):
    val = MarketValueAccrualCalculator().calculate_accrual(
        0, "2024-01-01", "2024-01-31", 0.10, cal, side="left",
        asset_quantity=100, asset_price=50,
    )
    assert val == pytest.approx(100 * 50 * 0.10 * 30 / 365)


def test_market_value_accrual_requires_inputs(cal):
    with pytest.raises(ValidationError):
        MarketValueAccrualCalculator().calculate_accrual(
            0, "2024-01-01", "2024-01-31", 0.10, cal,
        )


def test_last_market_value_accrual(cal):
    val = LastMarketValueAccrualCalculator().calculate_accrual(
        0, "2024-01-01", "2024-01-31", 0.10, cal, side="left",
        last_asset_quantity=100, last_asset_price=40,
    )
    assert val == pytest.approx(100 * 40 * 0.10 * 30 / 365)


def test_factory_dispatch(cal):
    assert isinstance(
        AccrualCalculatorFactory.create_calculator(AccrualType.NOTIONAL),
        StandardAccrualCalculator,
    ) or AccrualCalculatorFactory.create_calculator(
        AccrualType.NOTIONAL
    ).__class__.__name__ == "NotionalAccrualCalculator"
    assert isinstance(
        AccrualCalculatorFactory.create_calculator(AccrualType.MARKET_VALUE),
        MarketValueAccrualCalculator,
    )
    with pytest.raises(ValidationError):
        AccrualCalculatorFactory.create_calculator("nonsense")


# ---------------------------------------------------------------------------
# Params validation
# ---------------------------------------------------------------------------
def test_accrual_type_string_coercion():
    leg = FixLegParams(rate=0.05, accrual_type="marketvalue", accrual_side="right")
    assert leg.accrual_type == AccrualType.MARKET_VALUE
    assert leg.accrual_side == AccrualSide.RIGHT


def test_accrual_type_invalid_raises():
    with pytest.raises(ValidationError):
        FixLegParams(rate=0.05, accrual_type="bogus")


def test_asset_params_requires_series():
    with pytest.raises(ValidationError):
        AssetParams(asset_id="X", asset_initial_price=100, asset_prices=[1, 2, 3])


def test_pricing_output_mode_validation():
    with pytest.raises(ValidationError):
        PricingParams(valuation_date="2024-01-01", output_mode="weird")


def test_trsparams_requires_pricing():
    asset = AssetParams(
        asset_id="X", asset_initial_price=100,
        asset_prices=pd.Series([100.0], index=["2024-01-01"]),
    )
    with pytest.raises(ValidationError):
        TRSParams(
            contract_id="c1", asset=asset,
            fix_leg=FixLegParams(rate=0.05),
            float_leg=FloatLegParams(),
            pricing=None,
        )


def test_trsparams_insufficient_prices_for_last_mv():
    asset = AssetParams(
        asset_id="X", asset_initial_price=100,
        asset_prices=pd.Series([100.0], index=["2024-01-01"]),
    )
    with pytest.raises(ValidationError):
        TRSParams(
            contract_id="c1", asset=asset,
            fix_leg=FixLegParams(rate=0.05, accrual_type=AccrualType.LAST_MARKET_VALUE),
            float_leg=FloatLegParams(),
            pricing=PricingParams(valuation_date="2024-01-01"),
        )
