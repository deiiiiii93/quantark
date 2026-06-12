"""
FX European Vanilla Option Demo
================================

Prices a EUR/USD call with the Garman-Kohlhagen engine, shows analytical
Greeks (including premium-adjusted delta) and put-call parity.
"""

from datetime import datetime

from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxVanillaOption
from quantark.asset.fx.report import format_fx_option_report
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import OptionType


def main():
    env = FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12),
        spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.05),   # USD
        foreign_curve=FlatRateCurve(rate=0.03),    # EUR
        vol_surface=FlatVolSurface(volatility=0.10),
    )

    call = FxVanillaOption(
        currency_pair=CurrencyPair("EUR", "USD"),
        strike=1.25,
        option_type=OptionType.CALL,
        maturity=1.0,
        notional_foreign=1_000_000.0,
    )
    put = FxVanillaOption(
        currency_pair=CurrencyPair("EUR", "USD"),
        strike=1.25,
        option_type=OptionType.PUT,
        maturity=1.0,
        notional_foreign=1_000_000.0,
    )

    engine = GarmanKohlhagenEngine()
    print(format_fx_option_report(call, env, engine, display_mode="symbols"))

    call_price = engine.price(call, env)
    put_price = engine.price(put, env)
    forward = env.get_forward(1.0)
    parity = call.notional * (forward - call.strike) * env.get_domestic_df(1.0)
    print(f"\nPut price:                {put_price:,.2f} USD")
    print(f"Put-call parity check:    C - P = {call_price - put_price:,.2f} "
          f"vs N*(F-K)*df = {parity:,.2f}")

    # Premium-adjusted delta (premium paid in EUR)
    premium_option = FxVanillaOption(
        currency_pair=CurrencyPair("EUR", "USD"),
        strike=1.25,
        option_type=OptionType.CALL,
        maturity=1.0,
        notional_foreign=1_000_000.0,
        premium_currency="FOR",
        premium_amount=25_000.0,
    )
    greeks = engine.calculate_greeks(premium_option, env)
    print(f"\nSpot delta:               {greeks['delta']:,.2f} EUR")
    print(f"Premium-adjusted delta:   {greeks['delta_premium']:,.2f} EUR")
    print(f"Forward delta:            {greeks['fwd_delta']:,.2f} EUR")


if __name__ == "__main__":
    main()
