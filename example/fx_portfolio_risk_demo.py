"""
FX Portfolio — Full Risk Stack Demo
===================================

Builds one FX book (EUR/USD vanilla + EUR/USD digital + GBP/USD forward) and
drives it end-to-end through all four portfolio-risk modules:

  1. Stress testing      (quantark.stresstest.FXStressEngine)
  2. Value-at-Risk       (quantark.var FX engines: parametric / historical / MC)
  3. Dynamic scenarios   (quantark.dynamicscenario.FXDynamicScenarioEngine)
  4. Backtest hedging    (quantark.backtest.FXBacktestEngine + FXDeltaNeutralStrategy)

It also demonstrates that the full FX instrument set (incl. quanto) flows through
the FX portfolio layer.
"""
from datetime import datetime

import numpy as np
import pandas as pd

from quantark.asset.fx.engine.analytical import (
    FxDeltaOneEngine,
    FxDigitalOptionAnalyticalEngine,
    GarmanKohlhagenEngine,
    GarmanKohlhagenQuantoEngine,
)
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.deltaone import FxForward
from quantark.asset.fx.product.option import (
    FxDigitalOption,
    FxQuantoVanillaOption,
    FxVanillaOption,
)
from quantark.backtest import FXBacktestConfig, FXBacktestEngine
from quantark.backtest.strategy import FXDeltaNeutralStrategy
from quantark.backtest.transaction_costs import ProportionalCostModel
from quantark.dynamicscenario import FXDynamicScenarioConfig, FXPathLibrary, get_engine_for_portfolio
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.fx import FXPortfolio, FXPosition
from quantark.priceenv import FxPricingEnvironment, FxQuantoMarketData
from quantark.stresstest import FXStressConfig, FXStressEngine
from quantark.stresstest.scenario.scenario import Scenario, Stress
from quantark.stresstest.scenario.scenario_library import ScenarioLibrary
from quantark.stresstest.stress.stress_types import StressLevel, StressType
from quantark.util.enum import FxPayoutCurrency, OptionType
from quantark.var import (
    FXHistoricalVaREngine,
    FXMonteCarloVaREngine,
    FXParametricVaREngine,
    VaRConfig,
)


def build_book() -> FXPortfolio:
    eurusd = FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12), spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.03),
        vol_surface=FlatVolSurface(volatility=0.10))
    gbpusd = FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12), spot_quote=SpotQuote(spot=1.30),
        domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.045))

    pf = FXPortfolio(portfolio_name="FX Multi-Instrument Book",
                     pricing_environments={"EURUSD": eurusd, "GBPUSD": gbpusd})
    pf.add_position(
        product=FxVanillaOption(currency_pair=CurrencyPair("EUR", "USD"), strike=1.25,
                                option_type=OptionType.CALL, maturity=1.0,
                                notional_foreign=2_000_000.0),
        quantity=1.0, entry_price=0.0, underlying="EURUSD", engine=GarmanKohlhagenEngine())
    pf.add_position(
        product=FxDigitalOption(currency_pair=CurrencyPair("EUR", "USD"), strike=1.28,
                                option_type=OptionType.CALL, maturity=1.0, payout=500_000.0,
                                payout_currency=FxPayoutCurrency.DOMESTIC),
        quantity=1.0, entry_price=0.0, underlying="EURUSD",
        engine=FxDigitalOptionAnalyticalEngine())
    pf.add_position(
        product=FxForward(currency_pair=CurrencyPair("GBP", "USD"), notional_base=1_000_000.0,
                          contract_rate=1.31, maturity_date=datetime(2027, 6, 14)),
        quantity=1.0, entry_price=0.0, underlying="GBPUSD", engine=FxDeltaOneEngine())
    return pf


def synthetic_history(days=300, seed=5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-04-01", periods=days, freq="B")
    data = {}
    for pair, s0, v0, rd, rf, svol in [
        ("EURUSD", 1.20, 0.10, 0.05, 0.03, 0.007),
        ("GBPUSD", 1.30, 0.09, 0.05, 0.045, 0.006),
    ]:
        data[f"{pair}_spot"] = s0 * np.exp(np.cumsum(rng.normal(0, svol, days)))
        data[f"{pair}_vol"] = np.clip(v0 + np.cumsum(rng.normal(0, 0.002, days)), 0.03, None)
        data[f"{pair}_dom_rate"] = rd + np.cumsum(rng.normal(0, 0.0003, days))
        data[f"{pair}_for_rate"] = rf + np.cumsum(rng.normal(0, 0.0003, days))
    return pd.DataFrame(data, index=idx)


def section(title: str) -> None:
    print("\n" + "=" * 64 + f"\n{title}\n" + "=" * 64)


def main() -> None:
    pf = build_book()
    print(f"FX book value: ${pf.get_portfolio_value():,.2f}")
    print(f"Portfolio greeks: {pf.get_portfolio_risk_measures()}")

    # 1) Stress -----------------------------------------------------------
    section("1) STRESS TEST")
    stress = FXStressEngine(FXStressConfig(calculate_greeks=False)).run_static_scenarios(
        pf, [
            ScenarioLibrary.market_crash(),
            Scenario(name="USD +200bps", stresses=[
                Stress("domestic_rate", StressType.ABSOLUTE, 0.02, StressLevel.PORTFOLIO)]),
            Scenario(name="EUR Spot -10%", stresses=[
                Stress("spot", StressType.PERCENTAGE, -0.10, StressLevel.UNDERLYING,
                       target="EURUSD")]),
        ])
    print(stress.to_summary_dataframe()[
        ["scenario_name", "portfolio_pnl", "portfolio_pnl_pct"]].to_string(index=False))

    # 2) VaR --------------------------------------------------------------
    section("2) VALUE-AT-RISK (99%, 1-day)")
    df = synthetic_history()
    cfg = VaRConfig(confidence_level=0.99, lookback_days=250, calculate_factor_var=True,
                    mc_num_simulations=10_000, mc_seed=1)
    for name, eng in [("Parametric", FXParametricVaREngine(cfg)),
                      ("Historical", FXHistoricalVaREngine(cfg)),
                      ("Monte Carlo", FXMonteCarloVaREngine(cfg))]:
        r = eng.calculate_var(pf, df)
        print(f"  {name:<12} VaR ${r.var:>12,.0f}   CVaR ${r.cvar:>12,.0f}")

    # 3) Dynamic scenario -------------------------------------------------
    section("3) DYNAMIC SCENARIO (5-day carry unwind)")
    engine = get_engine_for_portfolio(pf, FXDynamicScenarioConfig(calculate_greeks=True))
    dyn = engine.run(pf, FXPathLibrary.carry_unwind(days=5))
    print(f"  Total P&L over path: ${dyn.total_pnl:,.2f} ({dyn.total_pnl_pct:+.2f}%)")
    print(f"  Worst day P&L:       ${dyn.get_worst_day().daily_pnl:,.2f}")

    # 4) Backtest hedging -------------------------------------------------
    section("4) BACKTEST (delta-neutral hedge, 60 days)")
    bt_path = synthetic_history(days=60, seed=9)
    bt = FXBacktestEngine(FXBacktestConfig(
        portfolio=build_book(), market_path=bt_path,
        strategy=FXDeltaNeutralStrategy(delta_threshold=50_000.0,
                                        rebalance_frequency="on_threshold"),
        transaction_cost_model=ProportionalCostModel(commission_rate=0.00005))).run()
    eff = bt.get_hedge_effectiveness()
    print(f"  Hedges: {bt.num_hedges}   Net P&L ${bt.total_net_pnl:,.2f}   "
          f"vol reduction {eff['vol_reduction_pct']:.1f}%")

    # Instrument coverage: quanto flows through the FX portfolio layer ----
    section("FX INSTRUMENT COVERAGE (quanto, JPY-settled)")
    quanto_env = FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12), spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.03),
        vol_surface=FlatVolSurface(volatility=0.10),
        quanto=FxQuantoMarketData(settlement_curve=FlatRateCurve(rate=0.001),
                                  quanto_vol=0.12, correlation=0.3))
    quanto_pos = FXPosition(
        product=FxQuantoVanillaOption(
            currency_pair=CurrencyPair("EUR", "USD"), strike=1.25, option_type=OptionType.CALL,
            maturity=1.0, notional_foreign=1_000_000.0, quanto_fx_rate=150.0, settlement_ccy="JPY"),
        quantity=1.0, entry_price=0.0, underlying="EURUSD_JPY",
        engine=GarmanKohlhagenQuantoEngine(), entry_timestamp=datetime(2026, 6, 13))
    print(f"  Quanto vanilla value (JPY): {quanto_pos.get_market_value(quanto_env):,.0f}")
    print(f"  Quanto greeks: delta={quanto_pos.get_greeks(quanto_env)['delta']:,.1f}")


if __name__ == "__main__":
    main()
