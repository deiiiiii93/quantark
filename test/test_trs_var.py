"""Gate 2 tests: equity TRS in the VaR engines.

Exercises parametric (sensitivity-based), historical and Monte Carlo VaR on a
swap portfolio, plus a mixed swap+option book, confirming the engines consume an
EquitySwapPosition through the BasePosition interface (no engine/product reach-in).
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
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
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.portfolio import EquityPortfolio
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.var import VaRConfig, EquityRiskFactorConfig
from quantark.var.engines import (
    ParametricVaREngine,
    HistoricalVaREngine,
    MonteCarloVaREngine,
)

START, END, VALUATION = "2024-01-02", "2024-01-31", "2024-01-30"
INITIAL, VAL_SPOT, NOTIONAL = 100.0, 110.0, 1_000_000.0


def _price_path() -> pd.Series:
    days = pd.date_range(START, END, freq="D")
    idx = [d.strftime("%Y-%m-%d") for d in days]
    val_i = idx.index(VALUATION)
    levels = [
        INITIAL + (VAL_SPOT - INITIAL) * min(i, val_i) / val_i for i in range(len(idx))
    ]
    return pd.Series(levels, index=idx).round(4)


def _make_trs() -> OneAssetTotalReturnSwap:
    calendar = Calendar(name="DemoCalendar")
    asset = AssetParams("IDX", INITIAL, _price_path())
    fix_leg = FixLegParams(
        rate=0.048, notional=NOTIONAL, initial_notional=NOTIONAL,
        start_date=START, end_date=END, payment_calendar=calendar, direction=-1,
    )
    float_leg = FloatLegParams(
        notional=NOTIONAL, initial_notional=NOTIONAL,
        start_date=START, end_date=END, payment_calendar=calendar, direction=1,
    )
    return OneAssetTotalReturnSwap(
        TRSParams(
            contract_id="TRS_VAR", asset=asset, fix_leg=fix_leg, float_leg=float_leg,
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


@pytest.fixture
def swap_portfolio(env) -> EquityPortfolio:
    pf = EquityPortfolio(portfolio_name="SwapBook", pricing_environments={"IDX": env})
    pf.add_swap_position(product=_make_trs(), quantity=1.0)
    return pf


@pytest.fixture
def hist_data() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 300
    return pd.DataFrame({
        "spot_return": rng.normal(0.0003, 0.02, n),
        "vol_change": rng.normal(0.0, 0.01, n),
        "rate_shift": rng.normal(0.0, 0.001, n),
    })


def _cfg() -> VaRConfig:
    return VaRConfig(
        confidence_level=0.99,
        lookback_days=252,
        equity_factors=EquityRiskFactorConfig(
            include_spot=True, include_vol=True, include_rate=True
        ),
    )


def test_parametric_var_on_swap(swap_portfolio, hist_data):
    result = ParametricVaREngine(_cfg()).calculate_var(swap_portfolio, hist_data)
    assert result.var > 0
    # Delta-one dollar spot sensitivity ~ (notional/initial) * spot drives VaR.
    assert result.var > 1_000.0


def test_historical_var_on_swap(swap_portfolio, hist_data):
    result = HistoricalVaREngine(_cfg()).calculate_var(swap_portfolio, hist_data)
    assert result.var > 0


def test_monte_carlo_var_on_swap(swap_portfolio, hist_data):
    cfg = _cfg()
    cfg.mc_num_simulations = 2_000
    result = MonteCarloVaREngine(cfg).calculate_var(swap_portfolio, hist_data)
    assert result.var > 0


def test_var_scales_with_notional(env, hist_data):
    """A larger swap (more quantity) carries proportionally larger VaR."""
    small = EquityPortfolio(portfolio_name="S", pricing_environments={"IDX": env})
    small.add_swap_position(product=_make_trs(), quantity=1.0)
    big = EquityPortfolio(portfolio_name="B", pricing_environments={"IDX": env})
    big.add_swap_position(product=_make_trs(), quantity=3.0)

    engine = ParametricVaREngine(_cfg())
    v_small = engine.calculate_var(small, hist_data).var
    v_big = engine.calculate_var(big, hist_data).var
    assert v_big == pytest.approx(3.0 * v_small, rel=1e-6)


def test_mixed_book_var_runs(env, hist_data):
    """Parametric VaR aggregates a swap and an option in one equity book."""
    pf = EquityPortfolio(portfolio_name="Mixed", pricing_environments={"IDX": env})
    pf.add_swap_position(product=_make_trs(), quantity=1.0)
    pf.add_position(
        product=EuropeanVanillaOption(
            strike=100.0, option_type=OptionType.CALL, maturity=0.5
        ),
        quantity=100.0, entry_price=5.0, underlying="IDX", engine=BlackScholesEngine(),
    )
    cfg = _cfg()
    cfg.calculate_component_var = True
    result = ParametricVaREngine(cfg).calculate_var(pf, hist_data)
    assert result.var > 0
    assert result.component_var is not None
    assert len(result.component_var) == 2
