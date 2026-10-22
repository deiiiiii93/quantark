"""
FX Digital Option Demo
======================

Prices cash-or-nothing and asset-or-nothing EUR/USD digitals and shows the
digital call/put parity.
"""

from datetime import datetime

from quantark.asset.fx.engine.analytical import FxDigitalOptionAnalyticalEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxDigitalOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import FxPayoutCurrency, OptionType


def main():
    env = FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12),
        spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.05),
        foreign_curve=FlatRateCurve(rate=0.03),
        vol_surface=FlatVolSurface(volatility=0.10),
    )
    engine = FxDigitalOptionAnalyticalEngine()

    def digital(option_type, payout_currency):
        return FxDigitalOption(
            currency_pair=CurrencyPair("EUR", "USD"),
            strike=1.25,
            option_type=option_type,
            maturity=1.0,
            payout=100_000.0,
            payout_currency=payout_currency,
        )

    print("Cash-or-nothing (pays 100,000 USD if in the money):")
    call = engine.price(digital(OptionType.CALL, FxPayoutCurrency.DOMESTIC), env)
    put = engine.price(digital(OptionType.PUT, FxPayoutCurrency.DOMESTIC), env)
    print(f"  Call: {call:,.2f} USD")
    print(f"  Put:  {put:,.2f} USD")
    print(f"  Call + Put = {call + put:,.2f} "
          f"(= payout * df_dom = {100_000 * env.get_domestic_df(1.0):,.2f})")

    print("\nAsset-or-nothing (pays 100,000 EUR if in the money):")
    call_a = engine.price(digital(OptionType.CALL, FxPayoutCurrency.FOREIGN), env)
    put_a = engine.price(digital(OptionType.PUT, FxPayoutCurrency.FOREIGN), env)
    print(f"  Call: {call_a:,.2f} USD")
    print(f"  Put:  {put_a:,.2f} USD")

    print("\nAnalytical Greeks (cash-or-nothing call):")
    greeks = engine.calculate_greeks(
        digital(OptionType.CALL, FxPayoutCurrency.DOMESTIC), env
    )
    for key in ("delta", "gamma", "vega", "theta", "rho_dom", "rho_for"):
        print(f"  {key:10s} {greeks[key]:,.4f}")


if __name__ == "__main__":
    main()
