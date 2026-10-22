"""
FX Quanto Option Demo
=====================

Prices a EUR/USD option settled in JPY at a fixed conversion rate, showing
the effect of the quanto correlation on value.
"""

from datetime import datetime

from quantark.asset.fx.engine.analytical import (
    FxQuantoDigitalAnalyticalEngine,
    GarmanKohlhagenQuantoEngine,
)
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import (
    FxQuantoDigitalOption,
    FxQuantoVanillaOption,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import (
    FxPricingEnvironment,
    FxQuantoMarketData,
    QuantoConversionOrientation,
)
from quantark.util.enum import OptionType


def make_env(correlation):
    return FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12),
        spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.05),   # USD
        foreign_curve=FlatRateCurve(rate=0.03),    # EUR
        vol_surface=FlatVolSurface(volatility=0.10),
        quanto=FxQuantoMarketData(
            settlement_curve=FlatRateCurve(rate=0.001),  # JPY
            quanto_vol=0.12,                             # USD/JPY vol
            correlation=correlation,
            conversion_orientation=QuantoConversionOrientation.SETTLEMENT_PER_DOMESTIC,
        ),
    )


def main():
    vanilla = FxQuantoVanillaOption(
        currency_pair=CurrencyPair("EUR", "USD"),
        strike=1.25,
        option_type=OptionType.CALL,
        maturity=1.0,
        notional_foreign=1_000_000.0,
        quanto_fx_rate=150.0,   # fixed USD -> JPY conversion
        settlement_ccy="JPY",
    )
    engine = GarmanKohlhagenQuantoEngine()

    print("EUR/USD call settled in JPY (fixed rate 150.0):")
    print(f"{'correlation':>12s} {'price (JPY)':>18s}")
    for corr in (-0.5, -0.3, 0.0, 0.3, 0.5):
        price = engine.price(vanilla, make_env(corr))
        print(f"{corr:12.1f} {price:18,.0f}")

    digital = FxQuantoDigitalOption(
        currency_pair=CurrencyPair("EUR", "USD"),
        strike=1.25,
        option_type=OptionType.CALL,
        maturity=1.0,
        payout=100_000.0,       # USD amount converted at the fixed rate
        quanto_fx_rate=150.0,
        settlement_ccy="JPY",
    )
    digital_price = FxQuantoDigitalAnalyticalEngine().price(
        digital, make_env(-0.3)
    )
    print(f"\nQuanto digital call (corr -0.3): {digital_price:,.0f} JPY")


if __name__ == "__main__":
    main()
