"""
FX Delta-One Demo
=================

Prices FX spot, forward and swap positions with FxDeltaOneEngine.
"""

from datetime import datetime

from quantark.asset.fx.engine.analytical import FxDeltaOneEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.deltaone import FxForward, FxSpot, FxSwap
from quantark.param import FlatRateCurve, SpotQuote
from quantark.priceenv import FxPricingEnvironment


def main():
    pair = CurrencyPair("EUR", "USD")
    env = FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12),
        spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.05),   # USD
        foreign_curve=FlatRateCurve(rate=0.03),    # EUR
    )
    engine = FxDeltaOneEngine()

    spot_trade = FxSpot(
        currency_pair=pair,
        notional_base=1_000_000.0,
        contract_rate=1.1990,
        value_date=datetime(2026, 6, 12),
    )
    print(f"{spot_trade}")
    print(f"  NPV: {engine.price(spot_trade, env):,.2f} USD\n")

    forward = FxForward(
        currency_pair=pair,
        notional_base=1_000_000.0,
        contract_rate=1.2100,
        maturity_date=datetime(2027, 6, 14),
    )
    details = engine.price_details(forward, env)
    print(f"{forward}")
    print(f"  Market forward:  {details['market_forward_rate']:.6f}")
    print(f"  Forward points:  {details['forward_points'] * 10_000:,.1f} pips")
    print(f"  NPV (USD):       {details['npv_quote_currency']:,.2f}")
    print(f"  NPV (EUR):       {details['npv_base_currency']:,.2f}\n")

    swap = FxSwap(
        currency_pair=pair,
        notional_base=1_000_000.0,
        near_rate=1.1990,
        far_rate=1.2050,
        near_date=datetime(2026, 6, 16),
        far_date=datetime(2026, 9, 16),
    )
    details = engine.price_details(swap, env)
    print(f"{swap}")
    print(f"  Swap points:     {swap.swap_points * 10_000:,.1f} pips")
    print(f"  Quote leg (USD): {details['npv_quote_currency']:,.2f}")
    print(f"  Base leg (EUR):  {details['npv_base_currency']:,.2f}")
    print(f"  Total NPV (USD): {details['npv']:,.2f}")

    # Spot-rate sensitivity of the forward via FDM Greeks
    greeks = engine.calculate_greeks(forward, env)
    print(f"\nForward FDM delta: {greeks['delta']:,.2f} EUR-equivalent")


if __name__ == "__main__":
    main()
