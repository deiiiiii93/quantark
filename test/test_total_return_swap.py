"""Behavioural tests for the single-asset Total Return Swap engine."""

import pandas as pd
import pytest

from quantark.util.calendar.business_calendar import Calendar
from quantark.asset.equity.product.swap.trs_params import (
    AccrualType,
    AssetParams,
    FixLegParams,
    FloatLegParams,
    PricingParams,
    EventParams,
    MarginParams,
    TRSParams,
    SwapState,
)
from quantark.asset.equity.product.swap.one_asset_trs import OneAssetTotalReturnSwap


START = "2024-01-01"  # Monday
END = "2024-01-31"
VAL = "2024-01-31"
INITIAL_PRICE = 100.0
NOTIONAL = 1000.0  # quantity = 10
RATE = 0.05


@pytest.fixture
def cal():
    return Calendar(name="NoHoliday")  # weekends only, no holidays


def _prices(level: float) -> pd.Series:
    days = pd.date_range(START, END, freq="D")
    idx = [d.strftime("%Y-%m-%d") for d in days]
    return pd.Series([level] * len(idx), index=idx)


def _build(cal, prices, *, output_mode="full", events=None,
           margin=None, fix_dir=-1, float_dir=1, accrual_type=AccrualType.NOTIONAL):
    asset = AssetParams(
        asset_id="TEST", asset_initial_price=INITIAL_PRICE, asset_prices=prices,
    )
    fix_leg = FixLegParams(
        rate=RATE, notional=NOTIONAL, initial_notional=NOTIONAL,
        start_date=START, end_date=END, payment_calendar=cal,
        accrual_type=accrual_type, direction=fix_dir,
    )
    float_leg = FloatLegParams(
        notional=NOTIONAL, initial_notional=NOTIONAL,
        start_date=START, end_date=END, payment_calendar=cal, direction=float_dir,
    )
    return OneAssetTotalReturnSwap(
        TRSParams(
            contract_id="c1", asset=asset, fix_leg=fix_leg, float_leg=float_leg,
            events=events or EventParams(),
            margin=margin or MarginParams(),
            pricing=PricingParams(valuation_date=VAL, output_mode=output_mode),
        )
    )


def test_flat_prices_give_zero_float_pnl(cal):
    trs = _build(cal, _prices(100.0))
    df = trs.price()
    # Flat at the initial price -> the float (total-return) leg has no P&L.
    assert df["float_interest"].astype(float).abs().max() == pytest.approx(0.0)


def _valuation_row(df):
    # The trailing row is the post-maturity settled period (all notional redeemed
    # -> zeros). The economically meaningful row accrues up to the valuation date.
    return df[df["period_end"] == VAL].iloc[-1]


def test_flat_prices_present_value_is_fixed_leg_only(cal):
    trs = _build(cal, _prices(100.0))
    row = _valuation_row(trs.price())
    # With no dividends and zero float P&L, PV == cumulative fixed interest.
    assert float(row["present_value"]) == pytest.approx(
        float(row["accrual_interest_cum"]), abs=1e-6
    )
    # Fixed leg is paid (direction -1) -> financing cost is negative.
    assert float(row["accrual_interest_cum"]) < 0


def test_price_appreciation_gives_known_float_pnl(cal):
    # Every observed price is 110 vs an initial of 100; quantity = 10, dir +1.
    trs = _build(cal, _prices(110.0))
    row = _valuation_row(trs.price())
    assert float(row["float_interest"]) == pytest.approx(100.0, abs=1e-6)
    assert float(row["present_value"]) == pytest.approx(
        float(row["accrual_interest_cum"]) + 100.0, abs=1e-6
    )


def test_spot_output_is_single_row(cal):
    trs = _build(cal, _prices(105.0), output_mode="spot")
    df = trs.price()
    assert len(df) == 1


def test_state_is_active_before_maturity(cal):
    trs = _build(cal, _prices(100.0))
    assert trs.state == SwapState.ACTIVE


def test_redemption_reduces_fixed_notional(cal):
    events = EventParams(redemption_events=[{
        "date": "2024-01-15",
        "redeem_notional": 400.0,
        "redeem_price": 100.0,
        "redeem_fee_rate": 0.0,
        "redeem_settle_option": ["asset"],
    }])
    trs = _build(cal, _prices(100.0), events=events)
    df = trs.price()
    # After a 400 redemption of a 1000 notional, the schedule's fix_notional
    # drops to 600 on/after the redemption date.
    schedule = trs.engine.create_notional_schedule(trs.params)
    by_date = {row["date"]: row for row in schedule}
    assert by_date["2024-01-16"]["fix_notional"] == pytest.approx(600.0)
    assert df is not None
