"""
FX range accrual demo.

Prices a EUR/USD range accrual coupon four ways: the digital-combination and
call/put-spread methods, each under a flat Black-Scholes surface and under a
skewed Vanna-Volga smile. Under flat vol the two methods agree; under the smile
the call/put-spread method captures the skew the level-only digital omits.
"""

from datetime import datetime

from quantark.asset.fx.engine.analytical import (
    FxForeignRangeAccrualAnalyticalEngine,
    FxQuantoRangeAccrualAnalyticalEngine,
    FxRangeAccrualAnalyticalEngine,
)
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import (
    FxForeignRangeAccrualOption,
    FxQuantoRangeAccrualOption,
    FxRangeAccrualConfig,
    FxRangeAccrualOption,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv.fx_pricing_environment import (
    FxQuantoMarketData,
    QuantoConversionOrientation,
)
from quantark.param.vol.vannavolga import (
    DeltaConvention,
    FXEnv,
    SmileQuotes,
    TermStructureVannaVolgaVolSurface,
    VannaVolgaVolSurface,
)
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import FxRangeAccrualMethod

SPOT, RD, RF, VOL, T = 1.20, 0.05, 0.03, 0.10, 1.0


def base_env(vol_surface):
    return FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 15),
        spot_quote=SpotQuote(spot=SPOT),
        domestic_curve=FlatRateCurve(rate=RD),
        foreign_curve=FlatRateCurve(rate=RF),
        vol_surface=vol_surface,
    )


def main():
    option = FxRangeAccrualOption(
        notional=1_000_000.0,
        range_config=FxRangeAccrualConfig(
            upper_barrier=1.30, lower_barrier=1.10, accrual_rate=0.04
        ),
        currency_pair=CurrencyPair("EUR", "USD"),
        maturity=T,
        num_observations=252,  # daily fixings over one year
    )

    flat = base_env(FlatVolSurface(volatility=VOL))
    smile = base_env(
        VannaVolgaVolSurface(
            FXEnv(spot=SPOT, rd=RD, rf=RF, tau=T),
            SmileQuotes(sigma_atm=VOL, rr25=-0.02, bf25_2vol=0.004),
            DeltaConvention.SPOT,
        )
    )

    print(f"EUR/USD range accrual  spot={SPOT}  band=[1.10, 1.30]  T={T}y")
    print(f"{'surface':<14}{'digital combo':>16}{'call/put spread':>18}")
    for label, env in (("flat BS", flat), ("VV smile", smile)):
        dig = FxRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.DIGITAL_COMBINATION
        ).price(option, env)
        spr = FxRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.CALL_PUT_SPREAD
        ).price(option, env)
        print(f"{label:<14}{dig:>16,.2f}{spr:>18,.2f}")

    engine = FxRangeAccrualAnalyticalEngine(
        method=FxRangeAccrualMethod.CALL_PUT_SPREAD
    )
    engine.price(option, smile)
    result = engine.get_last_result()
    print(
        f"\nVV expected in-range fraction: {result.expected_accrual_ratio:.4f}"
        f"  ({result.num_future_observations} fixings)"
    )
    greeks = engine.calculate_greeks(option, smile)
    print(
        "Greeks (USD): "
        f"delta={greeks['delta']:,.0f}  vega={greeks['vega']:,.0f}  "
        f"theta={greeks['theta']:,.2f}  rho_dom={greeks['rho_dom']:,.0f}  "
        f"rho_for={greeks['rho_for']:,.0f}"
    )

    _mc_demo(option, flat)
    _quanto_demo(option)
    _foreign_demo(option, flat)
    _term_structure_demo()


def _mc_demo(option, flat_env):
    """Monte Carlo cross-check: under flat vol, MC must match the analytic PV."""
    from quantark.asset.fx.engine.mc import FxRangeAccrualMCEngine

    analytic = FxRangeAccrualAnalyticalEngine(
        method=FxRangeAccrualMethod.DIGITAL_COMBINATION
    ).price(option, flat_env)
    eng = FxRangeAccrualMCEngine(num_paths=200_000, seed=7)
    mc = eng.price(option, flat_env)
    se = eng.get_last_result().std_error
    print("\nMonte Carlo cross-check (flat BS, 200k paths):")
    print(f"  analytical : {analytic:,.2f}")
    print(f"  monte carlo: {mc:,.2f}  (se {se:,.2f}, {abs(mc - analytic) / se:.2f} sigma)")


def _term_structure_demo():
    """A 2y daily accrual sees a rising-ATM, flattening-skew smile per fixing."""
    ts = TermStructureVannaVolgaVolSurface(
        [
            VannaVolgaVolSurface(
                FXEnv(spot=SPOT, rd=RD, rf=RF, tau=tau),
                SmileQuotes(sigma_atm=atm, rr25=rr, bf25_2vol=bf),
                DeltaConvention.SPOT,
            )
            for tau, atm, rr, bf in [
                (0.25, 0.090, -0.030, 0.006),
                (1.00, 0.110, -0.020, 0.004),
                (2.00, 0.120, -0.010, 0.003),
            ]
        ]
    )
    # A flat 1y smile applied to every fixing, for contrast.
    single = VannaVolgaVolSurface(
        FXEnv(spot=SPOT, rd=RD, rf=RF, tau=1.0),
        SmileQuotes(sigma_atm=0.110, rr25=-0.020, bf25_2vol=0.004),
        DeltaConvention.SPOT,
    )
    option = FxRangeAccrualOption(
        notional=1_000_000.0,
        range_config=FxRangeAccrualConfig(
            upper_barrier=1.35, lower_barrier=1.05, accrual_rate=0.04
        ),
        currency_pair=CurrencyPair("EUR", "USD"),
        maturity=2.0,
        num_observations=504,
    )
    engine = FxRangeAccrualAnalyticalEngine(
        method=FxRangeAccrualMethod.CALL_PUT_SPREAD
    )
    print("\nSmile term structure (2y daily accrual, band [1.05, 1.35]):")
    print(f"  ATM(K=1.20) by tenor: {[round(s.get_vol(1.20, s.env.tau, SPOT), 4) for s in ts.slices]}")
    p_ts = engine.price(option, base_env(ts))
    p_single = engine.price(option, base_env(single))
    print(f"  term-structure smile : {p_ts:,.2f}")
    print(f"  single 1y smile (flat): {p_single:,.2f}")
    print(f"  difference           : {p_ts - p_single:,.2f}")


def _foreign_demo(domestic_option, flat_env):
    """Same band, but the coupon is paid in EUR (foreign measure -> N(d1))."""
    foreign = FxForeignRangeAccrualOption(
        notional=domestic_option.notional,
        range_config=domestic_option.range_config,
        currency_pair=domestic_option.currency_pair,
        maturity=T,
        num_observations=252,
    )
    print("\nForeign-currency coupon (paid in EUR, valued in USD):")
    print(f"{'method':<22}{'value (USD)':>16}")
    for method in (
        FxRangeAccrualMethod.DIGITAL_COMBINATION,
        FxRangeAccrualMethod.CALL_PUT_SPREAD,
    ):
        price = FxForeignRangeAccrualAnalyticalEngine(method=method).price(
            foreign, flat_env
        )
        print(f"{method.value:<22}{price:>16,.2f}")


def _quanto_demo(domestic_option):
    """Same coupon, but paid in JPY at a fixed quanto rate (settlement measure)."""
    quanto_rate = 110.0  # USD coupon -> JPY at a fixed rate
    quanto = FxQuantoRangeAccrualOption(
        notional=domestic_option.notional,
        range_config=domestic_option.range_config,
        quanto_fx_rate=quanto_rate,
        currency_pair=domestic_option.currency_pair,
        maturity=T,
        num_observations=252,
        settlement_ccy="JPY",
    )

    print(f"\nQuanto (coupon paid in JPY at {quanto_rate}, quanto vol 12%):")
    print(f"{'correlation':<14}{'spread method (JPY)':>22}")
    for rho in (-0.5, 0.0, 0.5):
        env = base_env(FlatVolSurface(volatility=VOL))
        env.quanto = FxQuantoMarketData(
            settlement_curve=FlatRateCurve(rate=RD),
            quanto_vol=0.12,
            correlation=rho,
            conversion_orientation=QuantoConversionOrientation.SETTLEMENT_PER_DOMESTIC,
        )
        price = FxQuantoRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.CALL_PUT_SPREAD
        ).price(quanto, env)
        print(f"{rho:<14.1f}{price:>22,.2f}")


if __name__ == "__main__":
    main()
