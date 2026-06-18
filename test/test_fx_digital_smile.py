from datetime import datetime

import pytest

from quantark.param import SpotQuote, FlatRateCurve, FlatVolSurface
from quantark.param.vol.vannavolga import FXEnv, SmileQuotes, VannaVolgaVolSurface, DeltaConvention
from quantark.priceenv import FxPricingEnvironment
from quantark.asset.fx.product.option import FxDigitalOption
from quantark.asset.fx.engine.analytical import FxDigitalOptionAnalyticalEngine
from quantark.util.enum import OptionType, FxPayoutCurrency

VAL = datetime(2026, 6, 15)
TAU = 0.75
SMILE = SmileQuotes(sigma_atm=0.10, rr25=-0.02, bf25_2vol=0.004)  # skewed
FLAT = SmileQuotes(sigma_atm=0.10, rr25=0.0, bf25_2vol=0.0)


def _env(quotes):
    surface = VannaVolgaVolSurface(
        FXEnv(spot=1.20, rd=0.02, rf=0.01, tau=TAU), quotes, DeltaConvention.SPOT
    )
    return FxPricingEnvironment(
        valuation_date=VAL,
        spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.02),
        foreign_curve=FlatRateCurve(rate=0.01),
        vol_surface=surface,
    )


def _flat_env():
    return FxPricingEnvironment(
        valuation_date=VAL,
        spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.02),
        foreign_curve=FlatRateCurve(rate=0.01),
        vol_surface=FlatVolSurface(volatility=0.10),
    )


def _digital(call=True, payout_ccy=FxPayoutCurrency.DOMESTIC):
    return FxDigitalOption(
        strike=1.25,
        option_type=OptionType.CALL if call else OptionType.PUT,
        payout=1.0,
        maturity=TAU,
        payout_currency=payout_ccy,
    )


def test_replication_reduces_to_closed_form_flat_vv_smile():
    eng = FxDigitalOptionAnalyticalEngine()
    vv_price = eng.price(_digital(), _env(FLAT))
    flat_price = eng.price(_digital(), _flat_env())
    assert vv_price == pytest.approx(flat_price, rel=1e-4)


def _level_only_nd2_digital(env):
    """Closed-form domestic cash digital at the smile vol AT THE STRIKE only
    (no skew term) -- this is exactly what the engine produced before
    replication. The replicated price must differ from this by the skew term."""
    import math
    from scipy.stats import norm
    K, tau = 1.25, TAU
    sigma_k = env.get_vol(K, tau)
    fwd = env.get_forward(tau)
    d2 = (math.log(fwd / K) - 0.5 * sigma_k * sigma_k * tau) / (sigma_k * math.sqrt(tau))
    return env.get_domestic_df(tau) * norm.cdf(d2)


def test_replication_captures_skew_beyond_level():
    # The smile-consistent digital must differ from the level-only N(d2) digital
    # (same sigma(K), no skew term). Pre-replication these were identical, so
    # this test is the true discriminator for the skew correction.
    eng = FxDigitalOptionAnalyticalEngine()
    env = _env(SMILE)
    engine_price = eng.price(_digital(), env)
    level_only = _level_only_nd2_digital(env)
    assert abs(engine_price - level_only) > 1e-3  # skew term is materially nonzero
    assert 0.0 < engine_price < 1.0


def test_cash_call_put_parity_under_vv_surface():
    eng = FxDigitalOptionAnalyticalEngine()
    env = _env(SMILE)
    c = eng.price(_digital(call=True), env)
    p = eng.price(_digital(call=False), env)
    df_dom = env.get_domestic_df(TAU)
    assert c + p == pytest.approx(df_dom, rel=1e-6)


def test_asset_or_nothing_parity_under_vv_surface():
    eng = FxDigitalOptionAnalyticalEngine()
    env = _env(SMILE)
    c = eng.price(_digital(call=True, payout_ccy=FxPayoutCurrency.FOREIGN), env)
    p = eng.price(_digital(call=False, payout_ccy=FxPayoutCurrency.FOREIGN), env)
    s_eff = env.effective_spot()
    df_for = env.get_foreign_df(TAU)
    assert c + p == pytest.approx(s_eff * df_for, rel=1e-6)


def test_digital_greeks_smile_consistent_finite():
    eng = FxDigitalOptionAnalyticalEngine()
    greeks = eng.calculate_greeks(_digital(), _env(SMILE))
    for k in ("price", "delta", "gamma", "vega", "theta", "rho_dom", "rho_for"):
        assert k in greeks and greeks[k] == greeks[k]


def test_digital_flat_surface_price_unchanged():
    eng = FxDigitalOptionAnalyticalEngine()
    env = _flat_env()
    import math
    from scipy.stats import norm
    tau = TAU
    fwd = env.get_forward(tau)
    sigma = 0.10
    d2 = (math.log(fwd / 1.25) - 0.5 * sigma * sigma * tau) / (sigma * math.sqrt(tau))
    ref = env.get_domestic_df(tau) * norm.cdf(d2)
    assert eng.price(_digital(), env) == pytest.approx(ref, rel=1e-10)
