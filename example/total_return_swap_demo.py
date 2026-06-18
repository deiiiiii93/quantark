"""
Demonstration of equity Total Return Swap (TRS) pricing.

This script demonstrates:
1. Building a single-asset TRS from an observed asset price path
2. Pricing it to a per-period realized-cashflow table
3. A multi-asset (basket) TRS with a long/short mix
4. A dual-currency TRS reporting in a settlement currency

A TRS is a realized-cashflow instrument: given an observed price series it
accrues fixed-leg financing, float-leg total return, and a margin ledger over a
trading calendar. It does not use a risk-neutral PricingEnvironment.
"""

import pandas as pd

from quantark.util.calendar.business_calendar import Calendar
from quantark.asset.equity.product.swap import (
    AssetParams,
    FixLegParams,
    FloatLegParams,
    PricingParams,
    TRSParams,
    OneAssetTotalReturnSwap,
    MultiAssetTRS,
    MultiAssetInfo,
    OneAssetTotalReturnSwapDualCcy,
)

START = "2024-01-02"
END = "2024-01-31"
VALUATION = "2024-01-30"


def print_section(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def make_price_path(initial: float, final: float) -> pd.Series:
    """A linearly-interpolated daily price path between two levels."""
    days = pd.date_range(START, END, freq="D")
    idx = [d.strftime("%Y-%m-%d") for d in days]
    levels = [
        initial + (final - initial) * i / (len(idx) - 1) for i in range(len(idx))
    ]
    return pd.Series(levels, index=idx).round(4)


def build_params(calendar, asset_id, initial, final, notional, output_mode="spot"):
    asset = AssetParams(
        asset_id=asset_id,
        asset_initial_price=initial,
        asset_prices=make_price_path(initial, final),
    )
    fix_leg = FixLegParams(
        rate=0.048, notional=notional, initial_notional=notional,
        start_date=START, end_date=END, payment_calendar=calendar, direction=-1,
    )
    float_leg = FloatLegParams(
        notional=notional, initial_notional=notional,
        start_date=START, end_date=END, payment_calendar=calendar, direction=1,
    )
    return TRSParams(
        contract_id="TRS_DEMO", asset=asset, fix_leg=fix_leg, float_leg=float_leg,
        pricing=PricingParams(valuation_date=VALUATION, output_mode=output_mode),
    )


def main():
    calendar = Calendar(name="DemoCalendar")  # weekends only

    print_section("1. Single-asset TRS (full cashflow table)")
    params = build_params(calendar, "IDX", 3400.0, 3560.0, 50_000_000, "full")
    trs = OneAssetTotalReturnSwap(params)
    df = trs.price()
    print(f"State: {trs.state.value}")
    cols = ["period_start", "period_end", "accrual_interest_cum",
            "float_interest", "present_value", "margin_mtm"]
    print(df[cols].tail(5).to_string(index=False))

    print_section("2. Multi-asset TRS (long + short basket, spot view)")
    basket_params = build_params(calendar, "IDX", 3400.0, 3560.0, 50_000_000, "spot")
    assets = [
        MultiAssetInfo("IDX_A", 3400.0, make_price_path(3400.0, 3560.0),
                       notional=30_000_000, long_or_short=1),
        MultiAssetInfo("IDX_B", 120.0, make_price_path(120.0, 110.0),
                       notional=20_000_000, long_or_short=-1),
    ]
    multi = MultiAssetTRS(basket_params, assets)
    multi_df = multi.price()
    print(multi_df[["asset_id", "contract_id", "present_value"]].to_string(index=False))

    print_section("3. Dual-currency TRS (asset CNY, settle HKD, fixed FX)")
    dc_params = build_params(calendar, "IDX", 3400.0, 3560.0, 50_000_000, "spot")
    dual = OneAssetTotalReturnSwapDualCcy(
        dc_params, asset_ccy="CNY", settle_ccy="HKD",
        fx_rate_obs_type="fixed", fx_rate=0.92,
    )
    dc_df = dual.price()
    row = dc_df.iloc[-1]
    print(f"present_value (CNY): {row['present_value']}")
    print(f"present_value (HKD): {row['present_value_HKD']}")
    print(f"FX rate HKD/CNY:     {row['fx_rate_HKD/CNY']}")


if __name__ == "__main__":
    main()
