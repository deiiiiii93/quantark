"""End-to-end SA-CVA: real trade portfolio -> regulatory capital (spec §3.5).

Exercises the full façade: priced EuropeanVanillaOption -> MC exposure -> regulatory
CVA -> counterparty credit-spread delta sensitivities -> SBA calculator -> capital.
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
from quantark.sacva.models.enums import CreditQuality, RiskClass, RiskType
from quantark.sacva.portfolio.counterparty import Counterparty
from quantark.sacva.portfolio.credit_curve import PillarHazardCurve
from quantark.sacva.portfolio.netting import NettingSet
from quantark.sacva.portfolio.trade import CVATrade
from quantark.sacva.portfolio.trade_portfolio import CVATradePortfolio
from quantark.sacva.sacva_engine import SACVAEngine
from quantark.sacva.sensitivities.engine import CVASensitivityEngine

VAL = datetime(2026, 6, 17)
EXP_5Y = datetime(2031, 6, 16)
REG_TENORS = [0.5, 1.0, 3.0, 5.0, 10.0]


def _env():
    return PricingEnvironment(
        rate_curve=FlatRateCurve(0.03), valuation_date=VAL,
        spot_quote=SpotQuote(spot=100.0, asset_name="ACME"),
        vol_surface=FlatVolSurface(0.25), div_yield=ContinuousDividendYield(0.0),
        day_count_convention=DayCountConvention.CALENDAR_DAYS)


def _counterparty(equity_bucket=None):
    opt = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL,
                                exercise_date=EXP_5Y)
    trade = CVATrade(trade_id="t1", product=opt, engine=BlackScholesEngine(),
                     env=_env(), quantity=10.0, trade_currency="USD",
                     equity_bucket=equity_bucket)
    curve = PillarHazardCurve(tenors=REG_TENORS, hazards=[0.02] * 5, recovery_rate=0.4)
    return Counterparty(name="ACME_CP", netting_sets=[NettingSet("n1", [trade])],
                        credit_curve=curve, bucket=2, credit_quality=CreditQuality.IG)


def test_portfolio_to_capital_end_to_end():
    portfolio = CVATradePortfolio(counterparties=[_counterparty()], hedges=[])
    eng = SACVAEngine(exposure_engine=MonteCarloExposureEngine(
        MonteCarloExposureConfig(num_paths=8000, n_steps=20, seed=11)),
        include_ir_delta=False)  # flat-curve portfolio: IR delta intentionally omitted
    result = eng.compute(portfolio)

    assert result.total_capital > 0.0
    assert result.delta_capital > 0.0          # counterparty credit-spread delta
    assert result.vega_capital == 0.0          # no vega risk factors emitted in v1
    assert result.counterparty_cva["ACME_CP"] > 0.0
    assert "ACME_CP" in result.exposure_profiles


def test_equity_market_sensitivities_produced():
    portfolio = CVATradePortfolio(counterparties=[_counterparty(equity_bucket=5)],
                                  hedges=[])
    eng = SACVAEngine(exposure_engine=MonteCarloExposureEngine(
        MonteCarloExposureConfig(num_paths=8000, n_steps=12, seed=21)),
        include_ir_delta=False)  # flat-curve portfolio: IR delta intentionally omitted
    result = eng.compute(portfolio)
    # long call CVA rises with both spot and vol -> positive equity delta + vega capital
    assert result.delta_capital > 0.0
    assert result.vega_capital > 0.0
    assert result.total_capital == pytest.approx(
        result.delta_capital + result.vega_capital)


def test_market_sensitivities_all_or_none():
    env = _env()
    opt = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL,
                                exercise_date=EXP_5Y)
    tagged = CVATrade("a", opt, BlackScholesEngine(), env, equity_bucket=5)
    untagged = CVATrade("b", opt, BlackScholesEngine(), env)   # no equity_bucket
    curve = PillarHazardCurve(REG_TENORS, [0.02] * 5, recovery_rate=0.4)
    cp = Counterparty("CP", [NettingSet("n1", [tagged, untagged])], curve, 2,
                      CreditQuality.IG)
    with pytest.raises(ValidationError):
        SACVAEngine(exposure_engine=MonteCarloExposureEngine(
            MonteCarloExposureConfig(num_paths=2000, n_steps=4)),
            include_ir_delta=False).compute(CVATradePortfolio([cp], []))


def test_credit_spread_deltas_localise_to_trade_maturity():
    cp = _counterparty()
    profile = MonteCarloExposureEngine(
        MonteCarloExposureConfig(num_paths=8000, n_steps=20, seed=5)).compute(cp)
    sens = CVASensitivityEngine().counterparty_spread_deltas(cp, profile)

    by_tenor = {s.tenor: s.s_cva for s in sens}
    assert set(by_tenor) == set(REG_TENORS)
    assert all(s.risk_class is RiskClass.COUNTERPARTY_CREDIT for s in sens)
    assert all(s.risk_type is RiskType.DELTA for s in sens)
    # a 5Y trade has credit-spread sensitivity at tenors <= 5Y, ~0 at the 10Y pillar
    assert by_tenor[0.5] > 0.0 and by_tenor[1.0] > 0.0 and by_tenor[5.0] > 0.0
    assert by_tenor[10.0] == pytest.approx(0.0, abs=1e-9)
    # increasing CVA with spread => positive credit-spread delta
    assert all(v >= -1e-9 for v in by_tenor.values())
