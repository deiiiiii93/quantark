"""FX barrier MC demo: continuous BB MC vs the Vanna-Volga analytic under a
flat smile, plus a discrete-monitoring comparison."""

from datetime import datetime

from quantark.asset.fx.engine.analytical import VannaVolgaBarrierEngine
from quantark.asset.fx.engine.mc import FxBarrierMCEngine
from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxBarrierOption
from quantark.param import FlatRateCurve, SpotQuote
from quantark.param.vol.vannavolga import (
    DeltaConvention, FXEnv, SmileQuotes, VannaVolgaVolSurface,
)
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import FxBarrierType, ObservationType, OptionType

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
    opt = FxBarrierOption(
        strike=1.20, barrier=1.35, is_up=True, knock_type=FxBarrierType.KNOCK_OUT,
        option_type=OptionType.CALL, currency_pair=CurrencyPair("EUR", "USD"),
        maturity=1.0,
    )
    analytic = VannaVolgaBarrierEngine().price(opt, env)
    eng = FxBarrierMCEngine(params=FxMCParams(num_paths=200_000, time_steps=150, seed=7))
    mc = eng.price(opt, env)
    se = eng.get_last_result().std_error
    print("EUR/USD up-and-out call  K=1.20  H=1.35  T=1y (flat 10% vol)")
    print(f"  analytic (VV/RR) : {analytic:.6f}")
    print(f"  continuous BB MC : {mc:.6f}  (se {se:.6f}, {abs(mc-analytic)/se:.2f} sigma)")

    disc = FxBarrierOption(
        strike=1.20, barrier=1.35, is_up=True, knock_type=FxBarrierType.KNOCK_OUT,
        option_type=OptionType.CALL, currency_pair=CurrencyPair("EUR", "USD"),
        maturity=1.0, monitoring=ObservationType.DISCRETE,
        observation_times=[0.25, 0.5, 0.75, 1.0],
    )
    p_disc = FxBarrierMCEngine(params=FxMCParams(num_paths=200_000, seed=7)).price(disc, env)
    print(f"  discrete (4 fixings): {p_disc:.6f}  (>= continuous, fewer KO chances)")


if __name__ == "__main__":
    main()
