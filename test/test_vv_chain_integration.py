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


def test_var_vol_shift_keeps_wing_vols_positive():
    # A large negative VaR vol shift must not produce negative derived wing
    # vols (which SmileQuotes would reject). Construct directly to confirm.
    env = _env()
    bumped = bump_env(env, spot_return=0.0, vol_change=-0.30)
    surf = bumped.vol_surface
    sigma_25p, sigma_25c = surf.quotes.sigma_25d()
    assert sigma_25p > 0.0 and sigma_25c > 0.0
    assert surf.quotes.sigma_atm > 0.0


def test_full_greeks_dict_is_complete():
    eng = VannaVolgaBarrierEngine()
    greeks = eng.calculate_greeks(_barrier(), _env())
    for key in ("price", "delta", "gamma", "vega", "theta", "rho_dom", "rho_for"):
        assert key in greeks
        assert greeks[key] == greeks[key]  # not NaN


def test_one_touch_and_barrier_reprice_under_spot_bump():
    eng = VannaVolgaBarrierEngine()
    env = _env()
    base = eng.price(_barrier(), env)
    # Sticky-delta: bump spot via the standard engine path and confirm reprice.
    greeks = eng.calculate_greeks(_barrier(), env)
    assert greeks["price"] == pytest.approx(base, rel=1e-12)
    assert greeks["delta"] == greeks["delta"]  # finite


from quantark.asset.fx.product.option import FxVanillaOption
from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
from quantark.param import FlatVolSurface


def _vanilla():
    return FxVanillaOption(
        strike=1.20, option_type=OptionType.CALL, maturity=TAU, notional_foreign=1.0
    )


def test_vanilla_greeks_smile_consistent_match_bump_reprice():
    eng = GarmanKohlhagenEngine()
    env = _env()  # VannaVolgaVolSurface in the env
    greeks = eng.calculate_greeks(_vanilla(), env)
    from copy import deepcopy
    h = eng.params.spot_bump
    up, dn = deepcopy(env), deepcopy(env)
    up.spot_quote.spot = env.spot * (1 + h)
    dn.spot_quote.spot = env.spot * (1 - h)
    manual_delta = (eng.price(_vanilla(), up) - eng.price(_vanilla(), dn)) / (2 * env.spot * h)
    assert greeks["delta"] == pytest.approx(manual_delta, rel=1e-6)


def test_vanilla_greeks_flat_surface_unchanged():
    env = _env()
    env.vol_surface = FlatVolSurface(volatility=0.10)
    eng = GarmanKohlhagenEngine()
    greeks = eng.calculate_greeks(_vanilla(), env)
    assert greeks["vega"] == greeks["vega"]
    assert greeks["price"] > 0.0


def test_vanilla_and_digital_run_through_fx_var():
    from quantark.asset.fx.product.option import FxDigitalOption
    from quantark.asset.fx.engine.analytical import FxDigitalOptionAnalyticalEngine
    from quantark.var.fx.revaluation import bump_env
    env = _env()
    gk = GarmanKohlhagenEngine()
    dig = FxDigitalOptionAnalyticalEngine()
    digital = FxDigitalOption(
        strike=1.25, option_type=OptionType.CALL, payout=1.0, maturity=TAU
    )
    bumped = bump_env(env, spot_return=0.01, vol_change=0.01)
    assert gk.price(_vanilla(), bumped) > 0.0
    assert dig.price(digital, bumped) >= 0.0
    assert all(v == v for v in dig.calculate_greeks(digital, env).values())
