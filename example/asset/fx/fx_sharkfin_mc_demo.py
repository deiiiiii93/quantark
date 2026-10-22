"""FX sharkfin MC demo.

Shows that a plain up-and-out sharkfin (cap = barrier) equals a knock-out call
under continuous monitoring, and how the knock-out rebate and no-hit bonus move
the value.
"""

from datetime import datetime

from quantark.asset.fx.engine.analytical import VannaVolgaBarrierEngine
from quantark.asset.fx.engine.mc import FxBarrierMCEngine, FxSharkfinMCEngine
from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxBarrierOption, FxSharkfinOption
from quantark.param import FlatRateCurve, SpotQuote
from quantark.param.vol.vannavolga import (
    DeltaConvention, FXEnv, SmileQuotes, VannaVolgaVolSurface,
)
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import FxBarrierType, OptionType

SPOT, RD, RF = 1.20, 0.05, 0.03


def main():
    surface = VannaVolgaVolSurface(
        FXEnv(spot=SPOT, rd=RD, rf=RF, tau=1.0),
        SmileQuotes(sigma_atm=0.10, rr25=0.0, bf25_2vol=0.0),
        DeltaConvention.SPOT,
    )
    env = FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 15), spot_quote=SpotQuote(spot=SPOT),
        domestic_curve=FlatRateCurve(rate=RD), foreign_curve=FlatRateCurve(rate=RF),
        vol_surface=surface,
    )
    params = FxMCParams(num_paths=200_000, time_steps=150, seed=7)

    def shark(**kw):
        return FxSharkfinOption(
            strike=1.20, barrier=1.35, is_up=True, option_type=OptionType.CALL,
            currency_pair=CurrencyPair("EUR", "USD"), maturity=1.0, **kw,
        )

    ko_call = FxBarrierOption(
        strike=1.20, barrier=1.35, is_up=True, knock_type=FxBarrierType.KNOCK_OUT,
        option_type=OptionType.CALL, currency_pair=CurrencyPair("EUR", "USD"),
        maturity=1.0,
    )

    analytic = VannaVolgaBarrierEngine().price(ko_call, env)
    p_barrier = FxBarrierMCEngine(params=params).price(ko_call, env)
    p_shark = FxSharkfinMCEngine(params=params).price(shark(), env)
    p_rebate = FxSharkfinMCEngine(params=params).price(shark(ko_rebate=0.02), env)
    p_bonus = FxSharkfinMCEngine(params=params).price(shark(no_hit_rebate=0.05), env)

    print("EUR/USD up-and-out sharkfin  K=1.20  H=1.35  T=1y (flat 10% vol)")
    print(f"  KO call analytic (VV/RR) : {analytic:.6f}")
    print(f"  KO call MC               : {p_barrier:.6f}")
    print(f"  sharkfin (plain)         : {p_shark:.6f}   (== KO call: cap=barrier)")
    print(f"  + KO rebate 0.02         : {p_rebate:.6f}")
    print(f"  + no-hit bonus 0.05      : {p_bonus:.6f}")


if __name__ == "__main__":
    main()
