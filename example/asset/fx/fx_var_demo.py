"""
FX Value-at-Risk Demo
=====================

Computes parametric, historical and Monte-Carlo VaR for a EUR/USD + GBP/USD FX
book using synthetic two-rate factor history (spot, vol, domestic & foreign
rates), and shows factor / component attribution.
"""
from datetime import datetime

import numpy as np
import pandas as pd

from quantark.asset.fx.engine.analytical import FxDeltaOneEngine, GarmanKohlhagenEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.deltaone import FxForward
from quantark.asset.fx.product.option import FxVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.fx import FXPortfolio
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import OptionType
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
        vol_surface=FlatVolSurface(volatility=0.10),
    )
    gbpusd = FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12), spot_quote=SpotQuote(spot=1.30),
        domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.045),
        vol_surface=FlatVolSurface(volatility=0.09),
    )
    pf = FXPortfolio(portfolio_name="FX VaR Book",
                     pricing_environments={"EURUSD": eurusd, "GBPUSD": gbpusd})
    pf.add_position(
        product=FxVanillaOption(currency_pair=CurrencyPair("EUR", "USD"), strike=1.25,
                                option_type=OptionType.CALL, maturity=1.0,
                                notional_foreign=1_000_000.0),
        quantity=1.0, entry_price=0.0, underlying="EURUSD", engine=GarmanKohlhagenEngine())
    pf.add_position(
        product=FxForward(currency_pair=CurrencyPair("GBP", "USD"),
                          notional_base=2_000_000.0, contract_rate=1.30,
                          maturity_date=datetime(2027, 6, 14)),
        quantity=1.0, entry_price=0.0, underlying="GBPUSD", engine=FxDeltaOneEngine())
    return pf


def synthetic_history(days=400, seed=11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-06-01", periods=days, freq="B")
    data = {}
    for pair, spot0, vol0, rd0, rf0, svol in [
        ("EURUSD", 1.20, 0.10, 0.05, 0.03, 0.007),
        ("GBPUSD", 1.30, 0.09, 0.05, 0.045, 0.006),
    ]:
        spot = spot0 * np.exp(np.cumsum(rng.normal(0, svol, days)))
        data[f"{pair}_spot"] = spot
        data[f"{pair}_vol"] = np.clip(vol0 + np.cumsum(rng.normal(0, 0.002, days)), 0.03, None)
        data[f"{pair}_dom_rate"] = rd0 + np.cumsum(rng.normal(0, 0.0003, days))
        data[f"{pair}_for_rate"] = rf0 + np.cumsum(rng.normal(0, 0.0003, days))
    return pd.DataFrame(data, index=idx)


def main() -> None:
    pf = build_book()
    df = synthetic_history()
    config = VaRConfig(confidence_level=0.99, lookback_days=250,
                       calculate_factor_var=True, calculate_component_var=True,
                       mc_num_simulations=20_000, mc_seed=7)

    print(f"Portfolio value: ${pf.get_portfolio_value():,.2f}\n")

    par = FXParametricVaREngine(config).calculate_var(pf, df)
    hist = FXHistoricalVaREngine(config).calculate_var(pf, df)
    mc = FXMonteCarloVaREngine(config).calculate_var(pf, df)

    print(f"{'Method':<14}{'VaR (99%, 1d)':>18}{'CVaR':>18}")
    print("-" * 50)
    for name, r in [("Parametric", par), ("Historical", hist), ("Monte Carlo", mc)]:
        print(f"{name:<14}{r.var:>18,.2f}{r.cvar:>18,.2f}")

    print("\nFactor VaR attribution (parametric):")
    for factor, val in par.factor_var.items():
        print(f"  {factor:<22}{val:>14,.2f}")

    print("\nComponent VaR by position (parametric):")
    for pid, val in par.component_var.items():
        print(f"  {pid[:8]}…{'':<6}{val:>14,.2f}")


if __name__ == "__main__":
    main()
