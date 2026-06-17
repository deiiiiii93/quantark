"""SA-CVA portfolio -> capital demo.

Builds a real equity-option counterparty portfolio, runs the SA-CVA engine
(MC exposure -> regulatory CVA -> credit-spread sensitivities -> SBA capital),
and prints the regulatory CVA and capital. Run:

    PYTHONPATH=. python example/sacva_portfolio_demo.py
"""

from datetime import datetime

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

from quantark.sacva import (
    Counterparty,
    CVATrade,
    CVATradePortfolio,
    MonteCarloExposureConfig,
    MonteCarloExposureEngine,
    NettingSet,
    PillarHazardCurve,
    SACVAEngine,
)
from quantark.sacva.models.enums import CreditQuality

VAL = datetime(2026, 6, 17)
REG_TENORS = [0.5, 1.0, 3.0, 5.0, 10.0]


def _env(spot, vol, rate, asset):
    return PricingEnvironment(
        rate_curve=FlatRateCurve(rate), valuation_date=VAL,
        spot_quote=SpotQuote(spot=spot, asset_name=asset),
        vol_surface=FlatVolSurface(vol), div_yield=ContinuousDividendYield(0.0),
        day_count_convention=DayCountConvention.CALENDAR_DAYS)


def _option(strike, exercise, opt_type=OptionType.CALL):
    return EuropeanVanillaOption(strike=strike, option_type=opt_type,
                                 exercise_date=exercise)


def build_portfolio():
    # counterparty A: long 3Y ACME call (uncollateralized, enforceable netting)
    env_a = _env(spot=100.0, vol=0.25, rate=0.03, asset="ACME")
    trade_a = CVATrade(
        trade_id="A-call-3y",
        product=_option(100.0, datetime(2029, 6, 16)),
        engine=BlackScholesEngine(), env=env_a, quantity=50.0, trade_currency="USD",
        equity_bucket=5)
    cp_a = Counterparty(
        name="ACME_BANK",
        netting_sets=[NettingSet("A-ns", [trade_a])],
        credit_curve=PillarHazardCurve(REG_TENORS, [0.015, 0.018, 0.022, 0.025, 0.028],
                                       recovery_rate=0.40),
        bucket=2, credit_quality=CreditQuality.IG)

    # counterparty B: long 5Y GLOBEX put, higher hazard (HY)
    env_b = _env(spot=80.0, vol=0.30, rate=0.03, asset="GLOBEX")
    trade_b = CVATrade(
        trade_id="B-put-5y",
        product=_option(80.0, datetime(2031, 6, 16), OptionType.PUT),
        engine=BlackScholesEngine(), env=env_b, quantity=80.0, trade_currency="USD",
        equity_bucket=6)
    cp_b = Counterparty(
        name="GLOBEX_LLC",
        netting_sets=[NettingSet("B-ns", [trade_b])],
        credit_curve=PillarHazardCurve(REG_TENORS, [0.04, 0.045, 0.05, 0.055, 0.06],
                                       recovery_rate=0.40),
        bucket=3, credit_quality=CreditQuality.HY_NR)

    return CVATradePortfolio(counterparties=[cp_a, cp_b], hedges=[],
                             reporting_currency="USD")


def main():
    portfolio = build_portfolio()
    engine = SACVAEngine(exposure_engine=MonteCarloExposureEngine(
        MonteCarloExposureConfig(num_paths=20000, n_steps=24, seed=2026)))
    result = engine.compute(portfolio)

    print("=" * 64)
    print("SA-CVA: portfolio -> capital")
    print("=" * 64)
    for name, cva in result.counterparty_cva.items():
        prof = result.exposure_profiles[name]
        print(f"  {name:12s}  regulatory CVA = {cva:12.2f}   "
              f"peak disc. EE = {prof.epe_discounted.max():12.2f}")
    print("-" * 64)
    print(f"  delta capital  = {result.delta_capital:12.2f}")
    print(f"  vega capital   = {result.vega_capital:12.2f}")
    print(f"  TOTAL CAPITAL  = {result.total_capital:12.2f}  (m_cva={result.m_cva})")
    print("=" * 64)


if __name__ == "__main__":
    main()
