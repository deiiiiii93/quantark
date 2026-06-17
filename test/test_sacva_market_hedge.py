"""SA-CVA market sensitivities (equity + FX) and eligible-hedge MV (S_k^Hdg).

FX spot is modelled as a GBM factor (rate=domestic, div=foreign) exactly like an
equity underlying, tagged by the foreign currency. Hedges are bumped on the SAME
factor and emitted with s_cva=0 so the SBA netting forms WS = RW*(s_cva - s_hdg).
"""

from datetime import datetime

import pytest

from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv.pricing_environment import PricingEnvironment
from quantark.asset.equity.product.option.european_vanilla_option import (
    EuropeanVanillaOption,
)
from quantark.asset.equity.engine.analytical.black_scholes_engine import (
    BlackScholesEngine,
)
from quantark.util.enum import OptionType
from quantark.util.calendar import DayCountConvention
from quantark.util.exceptions import ValidationError

from quantark.sacva.exposure.simulator import (
    MonteCarloExposureConfig,
    MonteCarloExposureEngine,
)
from quantark.sacva.models.enums import CreditQuality
from quantark.sacva.portfolio.counterparty import Counterparty
from quantark.sacva.portfolio.credit_curve import PillarHazardCurve
from quantark.sacva.portfolio.netting import NettingSet
from quantark.sacva.portfolio.trade import CVAHedge, CVATrade
from quantark.sacva.portfolio.trade_portfolio import CVATradePortfolio
from quantark.sacva.sacva_engine import SACVAEngine

VAL = datetime(2026, 6, 17)
EXP_3Y = datetime(2029, 6, 16)
REG_TENORS = [0.5, 1.0, 3.0, 5.0, 10.0]


def _env(spot, vol, rate, div, asset):
    return PricingEnvironment(
        rate_curve=FlatRateCurve(rate), valuation_date=VAL,
        spot_quote=SpotQuote(spot=spot, asset_name=asset),
        vol_surface=FlatVolSurface(vol), div_yield=ContinuousDividendYield(div),
        day_count_convention=DayCountConvention.CALENDAR_DAYS)


def _call(strike=100.0):
    return EuropeanVanillaOption(strike=strike, option_type=OptionType.CALL,
                                 exercise_date=EXP_3Y)


def _curve():
    return PillarHazardCurve(REG_TENORS, [0.02] * 5, recovery_rate=0.4)


def _engine(seed=11):
    return SACVAEngine(exposure_engine=MonteCarloExposureEngine(
        MonteCarloExposureConfig(num_paths=8000, n_steps=12, seed=seed)))


def test_fx_market_sensitivities_produced():
    # FX underlying: rate=domestic 3%, div=foreign 1%, foreign currency EUR
    env = _env(spot=1.10, vol=0.12, rate=0.03, div=0.01, asset="EURUSD")
    trade = CVATrade("fx1", _call(strike=1.10), BlackScholesEngine(), env,
                     quantity=1_000_000.0, trade_currency="USD", fx_currency="EUR")
    cp = Counterparty("FX_CP", [NettingSet("n", [trade])], _curve(), 2,
                      CreditQuality.IG)
    result = _engine().compute(CVATradePortfolio([cp], [], reporting_currency="USD"))
    # long FX call -> CVA rises with the FX rate -> positive FX delta + vega capital
    assert result.delta_capital > 0.0
    assert result.vega_capital > 0.0
    assert any(lbl.startswith("FX:") for lbl in result.by_risk_class)


def test_fx_currency_equal_reporting_raises():
    env = _env(spot=100.0, vol=0.2, rate=0.03, div=0.0, asset="X")
    trade = CVATrade("x", _call(), BlackScholesEngine(), env,
                     trade_currency="USD", fx_currency="USD")
    cp = Counterparty("CP", [NettingSet("n", [trade])], _curve(), 2, CreditQuality.IG)
    with pytest.raises(ValidationError):
        _engine().compute(CVATradePortfolio([cp], [], reporting_currency="USD"))


def _equity_cp(seed_bucket=5):
    env = _env(spot=100.0, vol=0.25, rate=0.03, div=0.0, asset="ACME")
    trade = CVATrade("eq", _call(), BlackScholesEngine(), env, quantity=50.0,
                     trade_currency="USD", equity_bucket=seed_bucket)
    return Counterparty("EQ_CP", [NettingSet("n", [trade])], _curve(), 2,
                        CreditQuality.IG)


def test_equity_hedge_reduces_delta_capital():
    cp = _equity_cp()
    no_hedge = _engine(seed=3).compute(
        CVATradePortfolio([cp], [], reporting_currency="USD"))

    # small long-call hedge in the same equity bucket: s_hdg same sign as s_cva,
    # modest magnitude -> net WS shrinks -> equity delta capital falls
    henv = _env(spot=100.0, vol=0.25, rate=0.03, div=0.0, asset="ACME")
    hedge = CVAHedge("h", _call(), BlackScholesEngine(), henv, quantity=0.01,
                     trade_currency="USD", equity_bucket=5)
    with_hedge = _engine(seed=3).compute(
        CVATradePortfolio([cp], [hedge], reporting_currency="USD"))

    assert no_hedge.delta_capital > 0.0
    assert with_hedge.delta_capital < no_hedge.delta_capital


def test_hedge_without_market_factor_raises():
    cp = _equity_cp()
    henv = _env(spot=100.0, vol=0.25, rate=0.03, div=0.0, asset="ACME")
    hedge = CVAHedge("h", _call(), BlackScholesEngine(), henv, quantity=1.0,
                     trade_currency="USD")  # no equity_bucket / fx_currency
    with pytest.raises(ValidationError):
        _engine().compute(CVATradePortfolio([cp], [hedge], reporting_currency="USD"))


def test_tagged_hedge_with_untagged_trades_raises():
    # untagged trade + tagged hedge: the trade's own market CVA would be silently
    # dropped while the hedge's s_hdg is capitalised -> require all-or-none.
    env = _env(spot=100.0, vol=0.25, rate=0.03, div=0.0, asset="ACME")
    trade = CVATrade("eq", _call(), BlackScholesEngine(), env, quantity=50.0,
                     trade_currency="USD")            # NO equity_bucket
    cp = Counterparty("EQ_CP", [NettingSet("n", [trade])], _curve(), 2,
                      CreditQuality.IG)
    hedge = CVAHedge("h", _call(), BlackScholesEngine(), env, quantity=1.0,
                     trade_currency="USD", equity_bucket=5)
    with pytest.raises(ValidationError):
        _engine().compute(CVATradePortfolio([cp], [hedge], reporting_currency="USD"))


def test_mixed_equity_and_fx_factors_both_emitted():
    eq = _equity_cp()
    fxenv = _env(spot=1.10, vol=0.12, rate=0.03, div=0.01, asset="EURUSD")
    fx_trade = CVATrade("fx", _call(strike=1.10), BlackScholesEngine(), fxenv,
                        quantity=1_000_000.0, trade_currency="USD", fx_currency="EUR")
    fx_cp = Counterparty("FX_CP", [NettingSet("n", [fx_trade])], _curve(), 3,
                         CreditQuality.HY_NR)
    result = _engine().compute(
        CVATradePortfolio([eq, fx_cp], [], reporting_currency="USD"))
    labels = set(result.by_risk_class)
    assert any(l.startswith("EQUITY:") for l in labels)
    assert any(l.startswith("FX:") for l in labels)
