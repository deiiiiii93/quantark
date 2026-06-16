from datetime import datetime

import pytest

from quantark.param import SpotQuote, FlatRateCurve
from quantark.param.vol.vannavolga import FXEnv, SmileQuotes, VannaVolgaVolSurface, DeltaConvention
from quantark.priceenv import FxPricingEnvironment
from quantark.asset.fx.product.option import FxBarrierOption
from quantark.asset.fx.engine.analytical import VannaVolgaBarrierEngine
from quantark.util.enum import OptionType, FxBarrierType

VAL = datetime(2026, 6, 15)
TAU = 0.75
SMILE = SmileQuotes(sigma_atm=0.10, rr25=-0.01, bf25_2vol=0.003)


def _env():
    surface = VannaVolgaVolSurface(
        FXEnv(spot=1.20, rd=0.02, rf=0.01, tau=TAU), SMILE, DeltaConvention.SPOT
    )
    return FxPricingEnvironment(
        valuation_date=VAL,
        spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.02),
        foreign_curve=FlatRateCurve(rate=0.01),
        vol_surface=surface,
    )


def _barrier():
    return FxBarrierOption(
        strike=1.20, barrier=1.35, is_up=True,
        knock_type=FxBarrierType.KNOCK_OUT, option_type=OptionType.CALL,
        maturity=TAU,
    )


def test_vega_through_vv_surface_is_finite_and_nonzero():
    eng = VannaVolgaBarrierEngine()
    greeks = eng.calculate_greeks(_barrier(), _env())
    assert greeks["vega"] == greeks["vega"]      # not NaN
    assert abs(greeks["vega"]) > 0.0             # vol bump actually moved price


from quantark.var.fx.revaluation import bump_env


def test_var_bump_env_handles_vv_vol_shift():
    env = _env()
    bumped = bump_env(env, spot_return=0.0, vol_change=0.01)
    # The VV surface survives a vol shift and the ATM vol moved by +0.01.
    assert isinstance(bumped.vol_surface, VannaVolgaVolSurface)
    assert bumped.vol_surface.quotes.sigma_atm == pytest.approx(
        SMILE.sigma_atm + 0.01
    )
    # Original env is untouched.
    assert env.vol_surface.quotes.sigma_atm == pytest.approx(SMILE.sigma_atm)
