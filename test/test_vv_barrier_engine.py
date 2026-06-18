from datetime import datetime

import pytest

from quantark.param import SpotQuote, FlatRateCurve
from quantark.param.vol.vannavolga import FXEnv, SmileQuotes, VannaVolgaVolSurface, DeltaConvention
from quantark.priceenv import FxPricingEnvironment
from quantark.asset.fx.product.option import FxOneTouchOption, FxBarrierOption
from quantark.asset.fx.engine.analytical import VannaVolgaBarrierEngine
from quantark.asset.fx.engine.analytical.vannavolga import price_vv_one_touch
from quantark.util.enum import OptionType, FxBarrierType
from quantark.util.exceptions import MarketDataError

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


def test_engine_one_touch_matches_function():
    eng = VannaVolgaBarrierEngine()
    ot = FxOneTouchOption(barrier=1.35, is_up=True, payout=1.0, maturity=TAU)
    price = eng.price(ot, _env())
    ref = price_vv_one_touch(
        FXEnv(spot=1.20, rd=0.02, rf=0.01, tau=TAU), SMILE, 1.35, True,
        conv=DeltaConvention.SPOT,
    ).vv
    assert price == pytest.approx(ref, rel=1e-9)


def test_engine_barrier_prices_and_greeks():
    eng = VannaVolgaBarrierEngine()
    opt = FxBarrierOption(
        strike=1.20, barrier=1.35, is_up=True,
        knock_type=FxBarrierType.KNOCK_OUT, option_type=OptionType.CALL,
        maturity=TAU,
    )
    greeks = eng.calculate_greeks(opt, _env())
    assert greeks["price"] > 0.0
    assert "delta" in greeks and "vega" in greeks
    assert all(v == v for v in greeks.values())  # no NaNs


def test_engine_requires_vv_surface():
    from quantark.param import FlatVolSurface
    env = _env()
    env.vol_surface = FlatVolSurface(volatility=0.10)
    eng = VannaVolgaBarrierEngine()
    ot = FxOneTouchOption(barrier=1.35, is_up=True, maturity=TAU)
    with pytest.raises(MarketDataError):
        eng.price(ot, env)
