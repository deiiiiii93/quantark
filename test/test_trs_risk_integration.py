"""Gate 1 tests: equity TRS risk-stack foundation.

Covers the cashflow-to-risk bridge (:class:`TRSValuationEngine`), the
:class:`EquitySwapPosition` (BasePosition interface) and its acceptance into an
:class:`EquityPortfolio` alongside payoff-on-spot option positions.
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
from quantark.asset.equity.engine.cashflow.trs_valuation import TRSValuationEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.portfolio import EquitySwapPosition, EquityPortfolio
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

START = "2024-01-02"
END = "2024-01-31"
VALUATION = "2024-01-30"
INITIAL = 100.0
VAL_SPOT = 110.0  # observed price on the valuation date
NOTIONAL = 1_000_000.0


def _price_path() -> pd.Series:
    """Daily linear path from INITIAL at START to VAL_SPOT at VALUATION."""
    days = pd.date_range(START, END, freq="D")
    idx = [d.strftime("%Y-%m-%d") for d in days]
    val_i = idx.index(VALUATION)
    levels = []
    for i in range(len(idx)):
        if i <= val_i:
            levels.append(INITIAL + (VAL_SPOT - INITIAL) * i / val_i)
        else:
            levels.append(VAL_SPOT)
    return pd.Series(levels, index=idx).round(4)


def _build_params(direction_long: bool = True) -> TRSParams:
    calendar = Calendar(name="DemoCalendar")
    asset = AssetParams(
        asset_id="IDX", asset_initial_price=INITIAL, asset_prices=_price_path()
    )
    float_dir = 1 if direction_long else -1
    fix_dir = -float_dir
    fix_leg = FixLegParams(
        rate=0.048, notional=NOTIONAL, initial_notional=NOTIONAL,
        start_date=START, end_date=END, payment_calendar=calendar, direction=fix_dir,
    )
    float_leg = FloatLegParams(
        notional=NOTIONAL, initial_notional=NOTIONAL,
        start_date=START, end_date=END, payment_calendar=calendar, direction=float_dir,
    )
    return TRSParams(
        contract_id="TRS_TEST", asset=asset, fix_leg=fix_leg, float_leg=float_leg,
        pricing=PricingParams(valuation_date=VALUATION, output_mode="spot"),
    )


@pytest.fixture
def long_trs() -> OneAssetTotalReturnSwap:
    return OneAssetTotalReturnSwap(_build_params(direction_long=True))


@pytest.fixture
def env() -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=VAL_SPOT, asset_name="IDX"),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.048),
        valuation_date=datetime(2024, 1, 30),
    )


# --------------------------------------------------------------------------- #
# TRSValuationEngine
# --------------------------------------------------------------------------- #
def test_mtm_matches_engine_present_value(long_trs, env):
    """Bridge MtM at the observed spot equals the product's own present value."""
    valuator = TRSValuationEngine(long_trs.params)
    df = long_trs.price()  # spot-mode: single valuation-date row
    engine_pv = float(df.iloc[-1]["present_value"])
    bridge_pv = valuator.mark_to_market(env)
    assert bridge_pv == pytest.approx(engine_pv, rel=1e-6, abs=1e-2)


def test_delta_is_delta_one(long_trs, env):
    """A long TRS float leg is delta-one: delta ~ notional/initial_price."""
    valuator = TRSValuationEngine(long_trs.params)
    greeks = valuator.greeks(env)
    expected_qty = NOTIONAL / INITIAL
    assert greeks["delta"] == pytest.approx(expected_qty, rel=1e-3)
    assert greeks["gamma"] == pytest.approx(0.0, abs=1e-4)
    assert greeks["vega"] == 0.0


def test_short_trs_has_negative_delta(env):
    short = OneAssetTotalReturnSwap(_build_params(direction_long=False))
    greeks = TRSValuationEngine(short.params).greeks(env)
    assert greeks["delta"] < 0


def test_higher_spot_raises_long_mtm(long_trs, env):
    """MtM increases monotonically with spot for a long swap."""
    valuator = TRSValuationEngine(long_trs.params)
    base = valuator.mark_to_market(env, spot=VAL_SPOT)
    higher = valuator.mark_to_market(env, spot=VAL_SPOT + 5.0)
    assert higher > base


def test_funding_flow_through_opt_in(long_trs):
    """With funding_rate_ref set, a higher env rate lowers a long swap MtM
    (financing cost rises); the default keeps MtM rate-insensitive."""
    base_rate, bumped_rate = 0.048, 0.088
    env_lo = PricingEnvironment(
        spot_quote=SpotQuote(spot=VAL_SPOT, asset_name="IDX"),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=base_rate),
        valuation_date=datetime(2024, 1, 30),
    )
    env_hi = PricingEnvironment(
        spot_quote=SpotQuote(spot=VAL_SPOT, asset_name="IDX"),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=bumped_rate),
        valuation_date=datetime(2024, 1, 30),
    )
    floating = TRSValuationEngine(long_trs.params, funding_rate_ref=base_rate)
    assert floating.mark_to_market(env_hi) < floating.mark_to_market(env_lo)

    fixed = TRSValuationEngine(long_trs.params)  # default: contractual fixed
    assert fixed.mark_to_market(env_hi) == pytest.approx(fixed.mark_to_market(env_lo))


# --------------------------------------------------------------------------- #
# EquitySwapPosition
# --------------------------------------------------------------------------- #
def test_position_market_value_and_pnl(long_trs, env):
    pos = EquitySwapPosition(product=long_trs, quantity=2.0, entry_price=1000.0)
    mtm = pos.get_current_price(env)
    assert pos.get_market_value(env) == pytest.approx(mtm * 2.0)
    assert pos.get_pnl(env) == pytest.approx((mtm - 1000.0) * 2.0)
    assert pos.underlying == "IDX"
    assert pos.is_long() and not pos.is_short()


def test_position_greeks_scale_with_quantity(long_trs, env):
    one = EquitySwapPosition(product=long_trs, quantity=1.0)
    three = EquitySwapPosition(product=long_trs, quantity=3.0)
    assert three.get_greeks(env)["delta"] == pytest.approx(
        3.0 * one.get_greeks(env)["delta"]
    )


def test_position_rejects_zero_quantity(long_trs):
    with pytest.raises(ValidationError):
        EquitySwapPosition(product=long_trs, quantity=0.0)


# --------------------------------------------------------------------------- #
# EquityPortfolio acceptance (mixed book)
# --------------------------------------------------------------------------- #
def test_portfolio_accepts_swap_alongside_option(long_trs, env):
    portfolio = EquityPortfolio(
        portfolio_name="MixedBook", pricing_environments={"IDX": env}
    )
    swap_pos = portfolio.add_swap_position(product=long_trs, quantity=1.0)
    option = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=0.5
    )
    option_pos = portfolio.add_position(
        product=option, quantity=10.0, entry_price=5.0,
        underlying="IDX", engine=BlackScholesEngine(),
    )
    assert len(portfolio) == 2

    # Portfolio value sums both position market values (duck-typed).
    expected = swap_pos.get_market_value(env) + option_pos.get_market_value(env)
    assert portfolio.get_portfolio_value() == pytest.approx(expected)

    # Aggregated greeks include the swap's large delta-one contribution.
    greeks = portfolio.get_portfolio_risk_measures()
    assert greeks["delta"] > NOTIONAL / INITIAL * 0.5


def test_portfolio_rejects_unknown_underlying(long_trs, env):
    portfolio = EquityPortfolio(
        portfolio_name="Empty", pricing_environments={"OTHER": env}
    )
    with pytest.raises(ValidationError):
        portfolio.add_swap_position(product=long_trs, quantity=1.0)
