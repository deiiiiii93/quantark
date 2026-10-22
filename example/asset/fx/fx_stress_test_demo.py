"""
FX Stress Test Demo
===================

Builds a small EUR/USD + GBP/USD FX book (vanilla option + forward) and runs it
through the FX stress engine, exercising spot, volatility and the two-rate
(domestic / foreign) carry shocks that are unique to FX.
"""
from datetime import datetime

from quantark.asset.fx.engine.analytical import FxDeltaOneEngine, GarmanKohlhagenEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.deltaone import FxForward
from quantark.asset.fx.product.option import FxVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.fx import FXPortfolio
from quantark.priceenv import FxPricingEnvironment
from quantark.stresstest import FXStressConfig, FXStressEngine
from quantark.stresstest.scenario.scenario_library import ScenarioLibrary
from quantark.stresstest.scenario.scenario import Scenario, Stress
from quantark.stresstest.stress.stress_types import StressLevel, StressType
from quantark.util.enum import OptionType


def build_book() -> FXPortfolio:
    eurusd = FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12),
        spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.05),  # USD
        foreign_curve=FlatRateCurve(rate=0.03),  # EUR
        vol_surface=FlatVolSurface(volatility=0.10),
    )
    gbpusd = FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12),
        spot_quote=SpotQuote(spot=1.30),
        domestic_curve=FlatRateCurve(rate=0.05),  # USD
        foreign_curve=FlatRateCurve(rate=0.045),  # GBP
    )
    pf = FXPortfolio(
        portfolio_name="FX Macro Book",
        pricing_environments={"EURUSD": eurusd, "GBPUSD": gbpusd},
    )
    pf.add_position(
        product=FxVanillaOption(
            currency_pair=CurrencyPair("EUR", "USD"), strike=1.25,
            option_type=OptionType.CALL, maturity=1.0, notional_foreign=1_000_000.0,
        ),
        quantity=1.0, entry_price=0.0, underlying="EURUSD",
        engine=GarmanKohlhagenEngine(),
    )
    pf.add_position(
        product=FxForward(
            currency_pair=CurrencyPair("GBP", "USD"), notional_base=1_000_000.0,
            contract_rate=1.3100, maturity_date=datetime(2027, 6, 14),
        ),
        quantity=1.0, entry_price=0.0, underlying="GBPUSD",
        engine=FxDeltaOneEngine(),
    )
    return pf


def main() -> None:
    pf = build_book()
    engine = FXStressEngine(FXStressConfig(calculate_greeks=True,
                                           save_detailed_results=True))

    scenarios = [
        ScenarioLibrary.market_crash(),  # -20% spot, +50% vol (spot/vol only)
        ScenarioLibrary.vol_spike(),  # +80% vol
        Scenario(name="USD Rate Hike +200bps", stresses=[
            Stress("domestic_rate", StressType.ABSOLUTE, 0.02, StressLevel.PORTFOLIO)
        ]),
        Scenario(name="EUR Rate Hike +200bps", stresses=[
            Stress("foreign_rate", StressType.ABSOLUTE, 0.02, StressLevel.UNDERLYING,
                   target="EURUSD")
        ]),
        Scenario(name="EUR Spot Crash -15%", stresses=[
            Stress("spot", StressType.PERCENTAGE, -0.15, StressLevel.UNDERLYING,
                   target="EURUSD")
        ]),
    ]

    results = engine.run_static_scenarios(pf, scenarios)
    print(results.get_summary())

    print("\nScenario P&L table:")
    print(results.to_summary_dataframe().to_string(index=False))


if __name__ == "__main__":
    main()
