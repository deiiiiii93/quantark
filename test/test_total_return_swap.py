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


def test_matured_contract_prices_without_error(cal):
    # Valuation strictly after the contract end -> MATURED; pricing must not
    # index past the (maturity-capped) notional schedule.
    asset = AssetParams(
        asset_id="TEST", asset_initial_price=INITIAL_PRICE, asset_prices=_prices(100.0),
    )
    fix_leg = FixLegParams(
        rate=RATE, notional=NOTIONAL, initial_notional=NOTIONAL,
        start_date=START, end_date="2024-01-15", payment_calendar=cal, direction=-1,
    )
    float_leg = FloatLegParams(
        notional=NOTIONAL, initial_notional=NOTIONAL,
        start_date=START, end_date="2024-01-15", payment_calendar=cal, direction=1,
    )
    trs = OneAssetTotalReturnSwap(
        TRSParams(
            contract_id="m1", asset=asset, fix_leg=fix_leg, float_leg=float_leg,
            pricing=PricingParams(valuation_date="2024-01-31", output_mode="full"),
        )
    )
    assert trs.state == SwapState.MATURED
    df = trs.price()
    assert len(df) > 0


def test_engine_package_imports_first_without_circular_error():
    # Importing the engine package first must not trigger a product<->engine
    # circular import (regression guard).
    import importlib
    mod = importlib.import_module("quantark.asset.equity.engine.cashflow")
    assert hasattr(mod, "TotalReturnSwapEngine")


def test_dividend_without_deliver_ratio_defaults_to_one(cal):
    # Omitting deliver_ratio must default to 1.0 (not crash with TypeError).
    events = EventParams(dividend_events=[{
        "date": "2024-01-10", "cash_div_per_share": 1.0,
    }])
    trs = _build(cal, _prices(100.0), events=events)
    df = trs.price()  # must not raise
    assert df is not None


def test_dividend_missing_per_share_raises(cal):
    # Building the product constructs the notional schedule, where the missing
    # required field is detected.
    events = EventParams(dividend_events=[{"date": "2024-01-10"}])
    with pytest.raises(Exception):
        _build(cal, _prices(100.0), events=events).price()


def test_non_business_day_maturity_raises(cal):
    # 2024-01-06 is a Saturday -> must fail loudly rather than drop cashflows.
    asset = AssetParams(
        asset_id="TEST", asset_initial_price=INITIAL_PRICE, asset_prices=_prices(100.0),
    )
    fix_leg = FixLegParams(
        rate=RATE, notional=NOTIONAL, initial_notional=NOTIONAL,
        start_date=START, end_date="2024-01-06", payment_calendar=cal, direction=-1,
    )
    float_leg = FloatLegParams(
        notional=NOTIONAL, initial_notional=NOTIONAL,
        start_date=START, end_date="2024-01-06", payment_calendar=cal, direction=1,
    )
    trs = OneAssetTotalReturnSwap(
        TRSParams(
            contract_id="w1", asset=asset, fix_leg=fix_leg, float_leg=float_leg,
            pricing=PricingParams(valuation_date="2024-01-06", output_mode="full"),
        )
    )
    with pytest.raises(Exception):
        trs.price()


def test_maturity_redemption_handles_same_day_explicit_redemption(cal):
    # An explicit redemption ON the maturity date must not be double-counted by
    # the synthetic maturity redemption; final fixed notional settles to zero.
    events = EventParams(redemption_events=[{
        "date": END, "redeem_notional": 400.0, "redeem_price": 100.0,
        "redeem_fee_rate": 0.0, "redeem_settle_option": ["asset"],
    }])
    trs = _build(cal, _prices(100.0), events=events)
    schedule = trs.engine.create_notional_schedule(trs.params)
    last = schedule[-1]
    assert last["fix_notional"] == pytest.approx(0.0, abs=1e-6)
    assert trs.price() is not None


def test_same_day_redemptions_accumulate(cal):
    # Two redemptions on the same date must accumulate (not overwrite) their
    # notional and fees in the schedule row.
    events = EventParams(redemption_events=[
        {"date": "2024-01-15", "redeem_notional": 200.0, "redeem_price": 100.0,
         "redeem_fee_rate": 0.01, "redeem_settle_option": ["asset"]},
        {"date": "2024-01-15", "redeem_notional": 200.0, "redeem_price": 100.0,
         "redeem_fee_rate": 0.01, "redeem_settle_option": ["asset"]},
    ])
    trs = _build(cal, _prices(100.0), events=events)
    schedule = {r["date"]: r for r in trs.engine.create_notional_schedule(trs.params)}
    row = schedule["2024-01-15"]
    assert row["redeem_notional"] == pytest.approx(400.0)
    # qty per leg = 200/100 = 2; fee = 0.01*100*2 = 2 each -> 4 total.
    assert row["redeem_fee"] == pytest.approx(4.0)


def test_same_day_int_redemptions_do_not_over_realize(cal):
    # Two same-day redemptions settling 'int' must combine multiplicatively:
    # r1 = 3/10 = 0.3, then r2 = 3/7; combined opening fraction = 6/10 = 0.6.
    events = EventParams(redemption_events=[
        {"date": "2024-01-15", "redeem_notional": 300.0, "redeem_price": 100.0,
         "redeem_fee_rate": 0.0, "redeem_settle_option": ["int"]},
        {"date": "2024-01-15", "redeem_notional": 300.0, "redeem_price": 100.0,
         "redeem_fee_rate": 0.0, "redeem_settle_option": ["int"]},
    ])
    trs = _build(cal, _prices(100.0), events=events)
    schedule = {r["date"]: r for r in trs.engine.create_notional_schedule(trs.params)}
    realized = schedule["2024-01-15"]["fix_interest_realized"]
    assert realized == pytest.approx(0.6, abs=1e-9)
    assert realized <= 1.0


def test_non_business_day_redemption_raises(cal):
    # 2024-01-06 is a Saturday; an instantaneous redemption there would be
    # dropped by the working-day merge -> must fail loudly at pricing.
    events = EventParams(redemption_events=[
        {"date": "2024-01-06", "redeem_notional": 200.0, "redeem_price": 100.0,
         "redeem_fee_rate": 0.0, "redeem_settle_option": ["asset"]},
    ])
    trs = _build(cal, _prices(100.0), events=events)
    with pytest.raises(Exception):
        trs.price()


def test_share_dividend_adjusts_quantity(cal):
    events = EventParams(share_dividend_events=[
        {"date": "2024-01-10", "share_div_per_share": 0.1},
    ])
    trs = _build(cal, _prices(100.0), events=events)
    schedule = {r["date"]: r for r in trs.engine.create_notional_schedule(trs.params)}
    # Quantity steps from 10 to 10 * 1.1 = 11 on the share-dividend date.
    assert schedule["2024-01-10"]["asset_quantity"] == pytest.approx(11.0)


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
