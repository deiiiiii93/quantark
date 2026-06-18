"""Tests for MultiAssetTRS and the dual-currency single-asset TRS."""

import pandas as pd
import pytest

from quantark.util.calendar.business_calendar import Calendar
from quantark.asset.equity.product.swap.trs_params import (
    AssetParams,
    FixLegParams,
    FloatLegParams,
    PricingParams,
    TRSParams,
)
from quantark.asset.equity.product.swap.multi_asset_trs import (
    MultiAssetTRS,
    MultiAssetInfo,
)
from quantark.asset.equity.product.swap.one_asset_trs_dual_ccy import (
    OneAssetTotalReturnSwapDualCcy,
)

START = "2024-01-01"
END = "2024-01-31"
VAL = "2024-01-29"  # before maturity: a live contract with non-zero value


@pytest.fixture
def cal():
    return Calendar(name="NoHoliday")


def _prices(level):
    days = pd.date_range(START, END, freq="D")
    idx = [d.strftime("%Y-%m-%d") for d in days]
    return pd.Series([level] * len(idx), index=idx)


def _base_params(cal, prices, initial=100.0):
    asset = AssetParams(asset_id="A", asset_initial_price=initial, asset_prices=prices)
    fix_leg = FixLegParams(
        rate=0.05, notional=1000, initial_notional=1000,
        start_date=START, end_date=END, payment_calendar=cal, direction=-1,
    )
    float_leg = FloatLegParams(
        notional=1000, initial_notional=1000,
        start_date=START, end_date=END, payment_calendar=cal, direction=1,
    )
    return TRSParams(
        contract_id="C", asset=asset, fix_leg=fix_leg, float_leg=float_leg,
        pricing=PricingParams(valuation_date=VAL, output_mode="spot"),
    )


def test_multi_asset_spot_total_is_sum(cal):
    params = _base_params(cal, _prices(110.0))
    assets = [
        MultiAssetInfo("A1", 100.0, _prices(110.0), notional=1000),
        MultiAssetInfo("A2", 50.0, _prices(55.0), notional=500),
    ]
    multi = MultiAssetTRS(params, assets)
    df = multi.price()
    # The total row's present value equals the sum of the per-asset rows.
    total_row = df[df["contract_id"] == "C"].iloc[0]
    asset_rows = df[df["contract_id"] != "C"]
    assert float(total_row["present_value"]) == pytest.approx(
        asset_rows["present_value"].astype(float).sum(), abs=1e-6
    )


def test_multi_asset_short_flips_float_direction(cal):
    params = _base_params(cal, _prices(110.0))
    # A short position on a rising asset produces a float-leg loss vs the long.
    long_assets = [MultiAssetInfo("A", 100.0, _prices(110.0), notional=1000, long_or_short=1)]
    short_assets = [MultiAssetInfo("A", 100.0, _prices(110.0), notional=1000, long_or_short=-1)]
    long_pv = float(MultiAssetTRS(params, long_assets).price()
                    .query("contract_id == 'C'").iloc[0]["present_value"])
    short_pv = float(MultiAssetTRS(params, short_assets).price()
                     .query("contract_id == 'C'").iloc[0]["present_value"])
    assert short_pv < long_pv


def test_dual_ccy_fixed_fx_converts_present_value(cal):
    params = _base_params(cal, _prices(110.0))
    trs = OneAssetTotalReturnSwapDualCcy(
        params, asset_ccy="CNY", settle_ccy="HKD",
        fx_rate_obs_type="fixed", fx_rate=0.9,
    )
    df = trs.price()  # spot mode -> single live valuation row
    row = df.iloc[-1]
    # present_value_HKD is rounded to the requested precision (default 2dp).
    assert float(row["present_value_HKD"]) == pytest.approx(
        float(row["present_value"]) / 0.9, abs=0.01
    )


def test_dual_ccy_same_currency_forces_unit_fx(cal):
    params = _base_params(cal, _prices(110.0))
    trs = OneAssetTotalReturnSwapDualCcy(
        params, asset_ccy="CNY", settle_ccy="CNY",
        fx_rate_obs_type="fixed", fx_rate=0.9,
    )
    assert trs.fx_rate == 1


def test_dual_ccy_invalid_obs_type_raises(cal):
    params = _base_params(cal, _prices(110.0))
    with pytest.raises(Exception):
        OneAssetTotalReturnSwapDualCcy(
            params, asset_ccy="CNY", settle_ccy="HKD", fx_rate_obs_type="bogus",
        )
