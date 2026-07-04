"""Futures-tenor bucket Greeks (spec tests 3-8) + futures rhoq (5, 6)."""
import math
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical.deltaone_engine import DeltaOneEngine
from quantark.asset.equity.market import IndexFuturesCurve, IndexFuturesQuote
from quantark.asset.equity.product.deltaone.futures import Futures
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError


def _env(spot=5000.0, r=0.03, q=0.01, vol=0.20):
    return PricingEnvironment(
        rate_curve=FlatRateCurve(r),
        valuation_date=datetime(2026, 7, 3),
        spot_quote=SpotQuote(spot),
        vol_surface=FlatVolSurface(vol),
        div_yield=ContinuousDividendYield(q),
    )


# --- spec test 5: theoretical futures rhoq ---

def test_futures_theoretical_dividend_rho():
    env = _env()
    fut = Futures(underlying="IC", multiplier=1.0, maturity=0.5)
    greeks = DeltaOneEngine().calculate_greeks(fut, env)
    S, T, r, q = 5000.0, 0.5, 0.03, 0.01
    expected = -S * T * math.exp((r - q) * T) * 0.01
    assert greeks["dividend_rho"] == pytest.approx(expected, rel=1e-12)
    assert greeks["dividend_rho"] < 0.0  # long theoretical futures: rhoq < 0


# --- spec test 6: market-price mode keeps model rhoq at zero ---

def test_futures_market_price_dividend_rho_zero():
    env = _env()
    fut = Futures(underlying="IC", multiplier=1.0, maturity=0.5, market_price=5100.0)
    greeks = DeltaOneEngine(use_market_price=True).calculate_greeks(fut, env)
    assert greeks["dividend_rho"] == 0.0


from copy import deepcopy

from quantark.asset.equity.engine.analytical.black_scholes_engine import (
    BlackScholesEngine,
)
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.util.enum import FuturesCarryRiskMode, OptionType


def _ic_curve(spot=5000.0):
    return IndexFuturesCurve(
        underlying="IC",
        spot=spot,
        quotes=[
            IndexFuturesQuote("IC00", maturity=0.03, price=5008.0, multiplier=200.0),
            IndexFuturesQuote("IC01", maturity=0.10, price=5020.0, multiplier=200.0),
            IndexFuturesQuote("IC02", maturity=0.18, price=5036.0, multiplier=200.0),
            IndexFuturesQuote("IC03", maturity=0.32, price=5064.0, multiplier=200.0),
        ],
    )


# --- spec test 3: vanilla futures-delta bucket matches manual finite difference ---

def test_futures_delta_bucket_matches_manual_fd():
    env = _env()
    curve = _ic_curve()
    engine = BlackScholesEngine()
    option = EuropeanVanillaOption(5000.0, OptionType.CALL, maturity=0.10)

    rows = GreeksCalculator().calculate_futures_delta_buckets(
        option, env, engine, curve, price_bump=1.0
    )
    row = next(r for r in rows if r["contract"] == "IC01")

    base_env = deepcopy(env)
    base_env.div_yield = curve.to_dividend_yield_curve(env.rate_curve)
    bumped_env = deepcopy(env)
    bumped_env.div_yield = curve.bump_contract("IC01", 1.0).to_dividend_yield_curve(
        env.rate_curve
    )
    manual = engine.price(option, bumped_env) - engine.price(option, base_env)
    assert row["delta_bucket"] == pytest.approx(manual / 1.0, rel=1e-12)
    assert row["hedge_hands"] == pytest.approx(-row["delta_bucket"] / 200.0, rel=1e-12)
    assert row["delta_per_hand"] == 200.0
    assert row["extrapolated_tail"] is False


def test_futures_delta_buckets_row_shape_and_signs():
    env = _env()
    curve = _ic_curve()
    rows = GreeksCalculator().calculate_futures_delta_buckets(
        EuropeanVanillaOption(5000.0, OptionType.CALL, maturity=0.10),
        env,
        BlackScholesEngine(),
        curve,
    )
    assert [r["contract"] for r in rows] == ["IC00", "IC01", "IC02", "IC03"]
    # long call: positive futures exposure => short-futures hedge on active buckets
    active = [r for r in rows if abs(r["delta_bucket"]) > 1e-10]
    assert active and all(r["hedge_hands"] < 0 for r in active)


def test_futures_delta_buckets_mode_rejection():
    env = _env()
    curve = _ic_curve()
    calc = GreeksCalculator()
    option = EuropeanVanillaOption(5000.0, OptionType.CALL, maturity=0.10)
    with pytest.raises(ValidationError):
        calc.calculate_futures_delta_buckets(
            option, env, BlackScholesEngine(), curve,
            mode=FuturesCarryRiskMode.MARKET_PRICE,
        )
    with pytest.raises(ValidationError):
        calc.calculate_futures_delta_buckets(
            option, env, BlackScholesEngine(), curve,
            mode=FuturesCarryRiskMode.THEORETICAL_CARRY,
        )
    with pytest.raises(ValidationError):
        calc.calculate_futures_delta_buckets(
            option, env, BlackScholesEngine(), curve, price_bump=0.0
        )
