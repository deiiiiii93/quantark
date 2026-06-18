"""Gate 6 tests: equity TRS in Basel SA-CCR.

A TRS is a delta-one equity trade. EquitySwapPosition.to_saccr_trade maps it to a
SACCRTrade (asset_class=EQUITY, adjusted notional = spot x shares, supervisory
delta +/-1), which flows through a SACCRNettingSet -> SACCRCalculator to an EAD.
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantark.util.calendar.business_calendar import Calendar
from quantark.asset.equity.product.swap import (
    AssetParams,
    FixLegParams,
    FloatLegParams,
    PricingParams,
    TRSParams,
    OneAssetTotalReturnSwap,
)
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.portfolio import EquitySwapPosition
from quantark.saccr import SACCRCalculator, SACCRNettingSet
from quantark.saccr.models.enums import AssetClass, Position
from quantark.util.exceptions import ValidationError

START, END, VALUATION = "2024-01-02", "2024-01-31", "2024-01-30"
INITIAL, VAL_SPOT, NOTIONAL = 100.0, 110.0, 1_000_000.0
SHARES = NOTIONAL / INITIAL


def _make_trs(long: bool = True) -> OneAssetTotalReturnSwap:
    days = pd.date_range(START, END, freq="D")
    idx = [d.strftime("%Y-%m-%d") for d in days]
    val_i = idx.index(VALUATION)
    path = pd.Series(
        [INITIAL + (VAL_SPOT - INITIAL) * min(i, val_i) / val_i for i in range(len(idx))],
        index=idx,
    ).round(4)
    calendar = Calendar(name="DemoCalendar")
    float_dir = 1 if long else -1
    return OneAssetTotalReturnSwap(
        TRSParams(
            contract_id="TRS_CCR",
            asset=AssetParams("IDX", INITIAL, path),
            fix_leg=FixLegParams(
                rate=0.048, notional=NOTIONAL, initial_notional=NOTIONAL,
                start_date=START, end_date=END, payment_calendar=calendar,
                direction=-float_dir,
            ),
            float_leg=FloatLegParams(
                notional=NOTIONAL, initial_notional=NOTIONAL,
                start_date=START, end_date=END, payment_calendar=calendar,
                direction=float_dir,
            ),
            pricing=PricingParams(valuation_date=VALUATION, output_mode="spot"),
        )
    )


@pytest.fixture
def env() -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=VAL_SPOT, asset_name="IDX"),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.048),
        valuation_date=datetime(2024, 1, 30),
    )


def test_long_swap_maps_to_equity_trade(env):
    pos = EquitySwapPosition(product=_make_trs(long=True), quantity=1.0)
    trade = pos.to_saccr_trade(env)
    assert trade.asset_class == AssetClass.EQUITY
    assert trade.reference_entity == "IDX"
    assert trade.position == Position.LONG
    assert trade.is_index is False
    assert trade.notional == pytest.approx(VAL_SPOT * SHARES)  # spot x shares


def test_short_swap_is_short_position(env):
    pos = EquitySwapPosition(product=_make_trs(long=False), quantity=1.0)
    assert pos.to_saccr_trade(env).position == Position.SHORT


def test_quantity_scales_adjusted_notional(env):
    pos = EquitySwapPosition(product=_make_trs(long=True), quantity=3.0)
    assert pos.to_saccr_trade(env).notional == pytest.approx(VAL_SPOT * SHARES * 3.0)


def test_saccr_ead_positive_for_swap(env):
    pos = EquitySwapPosition(product=_make_trs(long=True), quantity=1.0)
    trade = pos.to_saccr_trade(env)
    ns = SACCRNettingSet("NS_TRS", [trade], is_margined=False)
    result = SACCRCalculator().calculate(ns)

    assert result.ead > 0
    assert result.pfe > 0
    # Single-name equity addon = SF(32%) x |delta=1| x notional x MF, with the
    # unmargined maturity factor MF = sqrt(min(M, 1)). The near-expiry test swap's
    # maturity is floored to SA-CCR's 10-business-day minimum.
    mf = min(trade.maturity, 1.0) ** 0.5
    assert result.addon_aggregate == pytest.approx(
        0.32 * trade.notional * mf, rel=0.02
    )


def test_matured_swap_has_no_saccr_exposure(env):
    """A swap whose contract has ended carries no future counterparty exposure.

    Legs end 2024-01-10 but the valuation date is 2024-01-30 (state MATURED), so
    mapping to a live SA-CCR trade must raise rather than fabricate a floored
    10-business-day exposure.
    """
    days = pd.date_range(START, END, freq="D")
    idx = [d.strftime("%Y-%m-%d") for d in days]
    path = pd.Series([INITIAL] * len(idx), index=idx)
    calendar = Calendar(name="DemoCalendar")
    matured = OneAssetTotalReturnSwap(
        TRSParams(
            contract_id="TRS_MATURED",
            asset=AssetParams("IDX", INITIAL, path),
            fix_leg=FixLegParams(
                rate=0.048, notional=NOTIONAL, initial_notional=NOTIONAL,
                start_date=START, end_date="2024-01-10",
                payment_calendar=calendar, direction=-1,
            ),
            float_leg=FloatLegParams(
                notional=NOTIONAL, initial_notional=NOTIONAL,
                start_date=START, end_date="2024-01-10",
                payment_calendar=calendar, direction=1,
            ),
            pricing=PricingParams(valuation_date=VALUATION, output_mode="spot"),
        )
    )
    pos = EquitySwapPosition(product=matured, quantity=1.0)
    with pytest.raises(ValidationError):
        pos.to_saccr_trade(env)


def test_index_swap_uses_index_factor(env):
    pos = EquitySwapPosition(product=_make_trs(long=True), quantity=1.0)
    single = pos.to_saccr_trade(env, is_index=False)
    index = pos.to_saccr_trade(env, is_index=True)
    assert index.is_index is True

    ead_single = SACCRCalculator().calculate(
        SACCRNettingSet("S", [single], is_margined=False)
    ).ead
    ead_index = SACCRCalculator().calculate(
        SACCRNettingSet("I", [index], is_margined=False)
    ).ead
    # Index SF (20%) < single-name SF (32%), so index EAD is smaller.
    assert ead_index < ead_single
