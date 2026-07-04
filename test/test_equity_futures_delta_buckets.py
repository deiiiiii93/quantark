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
