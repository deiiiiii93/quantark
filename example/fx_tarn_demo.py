"""FX Target Redemption (TARN) demo: TARF + digital-coupon note.

Shows both target-redemption products priced by Monte Carlo on the shared
constant-vol Garman-Kohlhagen path generator, each cross-checked against its
closed-form ``target=inf`` limit under a flat surface:

  - TARF: target=inf, gearing=1 -> plain forward strip.
  - TARN note: target=inf, no principal -> FX digital coupon strip.

Then shows the effect of a finite (binding) target.
"""

from datetime import datetime

from quantark.asset.fx.engine.analytical import FxDigitalOptionAnalyticalEngine
from quantark.asset.fx.engine.mc import (
    FxTarnForwardMCEngine,
    FxTargetRedemptionNoteMCEngine,
)
from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import (
    FxDigitalOption,
    FxTargetRedemptionForward,
    FxTargetRedemptionNote,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import FxPayoutCurrency, OptionType

PAIR = CurrencyPair("EUR", "USD")
FIXINGS = [0.25, 0.5, 0.75, 1.0]


def env(vol=0.12):
    return FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 15),
        spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.05),
        foreign_curve=FlatRateCurve(rate=0.03),
        vol_surface=FlatVolSurface(volatility=vol),
    )


def forward_strip(e, strike, fixings):
    return sum(
        (e.get_forward(t) - strike) * e.get_domestic_df(t) for t in fixings
    )


def digital_strip(e, note):
    eng = FxDigitalOptionAnalyticalEngine()
    total = 0.0
    for i, t in enumerate(note.fixing_times):
        total += eng.price(
            FxDigitalOption(
                strike=note.strike, option_type=OptionType.CALL,
                payout=note.full_coupon(i), maturity=t,
                payout_currency=FxPayoutCurrency.DOMESTIC, currency_pair=PAIR,
            ),
            e,
        )
    return total


def main():
    e = env()
    params = FxMCParams(num_paths=200_000, seed=7)

    print("=" * 66)
    print("FX Target Redemption Forward (TARF)")
    print("=" * 66)
    K = 1.20
    tarf_open = FxTargetRedemptionForward(
        strike=K, fixing_times=FIXINGS, target=float("inf"), currency_pair=PAIR)
    eng = FxTarnForwardMCEngine(params=params)
    mc = eng.price(tarf_open, e)
    se = eng.get_last_result().std_error
    strip = forward_strip(e, K, FIXINGS)
    print(f"  target=inf, gearing=1 (open):")
    print(f"    MC          = {mc:+.6f}  (se {se:.2e})")
    print(f"    forward strip = {strip:+.6f}  [closed form]")
    print(f"    diff        = {abs(mc - strip):.2e}")

    tarf_capped = FxTargetRedemptionForward(
        strike=K, fixing_times=FIXINGS, target=0.04, gearing=2.0,
        currency_pair=PAIR)
    capped = eng.price(tarf_capped, e)
    frac = eng.get_last_result().mean_redemption_fraction
    print(f"  target=0.04, gearing=2 (leveraged):")
    print(f"    MC          = {capped:+.6f}   early-redeem fraction {frac:.1%}")

    print()
    print("=" * 66)
    print("FX Target Redemption Note (digital coupon)")
    print("=" * 66)
    note_open = FxTargetRedemptionNote(
        fixing_times=FIXINGS, coupon_rate=0.08, strike=1.20,
        target=float("inf"), include_principal=False, currency_pair=PAIR)
    neng = FxTargetRedemptionNoteMCEngine(params=params)
    nmc = neng.price(note_open, e)
    nse = neng.get_last_result().std_error
    dstrip = digital_strip(e, note_open)
    print(f"  target=inf, no principal (coupon leg only):")
    print(f"    MC            = {nmc:.6f}  (se {nse:.2e})")
    print(f"    digital strip = {dstrip:.6f}  [closed form]")
    print(f"    diff          = {abs(nmc - dstrip):.2e}")

    note_full = FxTargetRedemptionNote(
        fixing_times=FIXINGS, coupon_rate=0.08, strike=1.20,
        target=0.10, include_principal=True, currency_pair=PAIR)
    full = neng.price(note_full, e)
    nfrac = neng.get_last_result().mean_redemption_fraction
    print(f"  target=0.10, with principal:")
    print(f"    MC (full note PV) = {full:.6f}   early-redeem fraction {nfrac:.1%}")


if __name__ == "__main__":
    main()
