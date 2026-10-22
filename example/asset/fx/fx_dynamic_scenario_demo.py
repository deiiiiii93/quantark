"""
FX Dynamic Scenario Demo
========================

Evolves a EUR/USD option + GBP/USD forward book day-by-day through FX paths
(spot trend, carry unwind, rate divergence), tracking value, P&L and the
two-rate FX greeks at each step.
"""
from datetime import datetime

from quantark.asset.fx.engine.analytical import FxDeltaOneEngine, GarmanKohlhagenEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.deltaone import FxForward
from quantark.asset.fx.product.option import FxVanillaOption
from quantark.dynamicscenario import (
    FXDynamicScenarioConfig,
    FXDynamicScenarioEngine,
    FXPathLibrary,
    get_engine_for_portfolio,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.fx import FXPortfolio
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import OptionType


def build_book() -> FXPortfolio:
    eurusd = FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12), spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.03),
        vol_surface=FlatVolSurface(volatility=0.10))
    gbpusd = FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12), spot_quote=SpotQuote(spot=1.30),
        domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.045))
    pf = FXPortfolio(portfolio_name="FX Dynamic Book",
                     pricing_environments={"EURUSD": eurusd, "GBPUSD": gbpusd})
    pf.add_position(
        product=FxVanillaOption(currency_pair=CurrencyPair("EUR", "USD"), strike=1.25,
                                option_type=OptionType.CALL, maturity=1.0,
                                notional_foreign=1_000_000.0),
        quantity=1.0, entry_price=0.0, underlying="EURUSD", engine=GarmanKohlhagenEngine())
    pf.add_position(
        product=FxForward(currency_pair=CurrencyPair("GBP", "USD"),
                          notional_base=1_000_000.0, contract_rate=1.31,
                          maturity_date=datetime(2027, 6, 14)),
        quantity=1.0, entry_price=0.0, underlying="GBPUSD", engine=FxDeltaOneEngine())
    return pf


def main() -> None:
    pf = build_book()
    engine = get_engine_for_portfolio(pf)  # factory auto-selects the FX engine
    assert isinstance(engine, FXDynamicScenarioEngine)

    for path in [
        FXPathLibrary.spot_trend(days=5, daily_pct=0.01),
        FXPathLibrary.carry_unwind(days=5),
        FXPathLibrary.rate_divergence(days=5, dom_bps_per_day=5.0, for_bps_per_day=-5.0),
    ]:
        results = FXDynamicScenarioEngine(
            FXDynamicScenarioConfig(calculate_greeks=True)
        ).run(pf, path)
        print(f"\n=== {path.name} ===")
        print(f"  Total P&L: ${results.total_pnl:,.2f} ({results.total_pnl_pct:+.2f}%)")
        worst = results.get_worst_day()
        print(f"  Worst day: Day {worst.day_index} "
              f"(daily P&L ${worst.daily_pnl:,.2f})")
        greeks = results.get_greeks_evolution()
        cols = [c for c in ["delta", "vega", "rho_dom", "rho_for"] if c in greeks.columns]
        print("  Greeks evolution (last day): "
              + ", ".join(f"{c}={greeks.iloc[-1][c]:,.1f}" for c in cols))


if __name__ == "__main__":
    main()
