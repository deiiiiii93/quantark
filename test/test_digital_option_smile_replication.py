import math
from datetime import datetime

from scipy.stats import norm

from quantark.asset.equity.product.option import CashOrNothingDigitalOption
from quantark.asset.equity.engine.analytical import DigitalOptionAnalyticalEngine
from quantark.param import SpotQuote, FlatRateCurve, FlatVolSurface
from quantark.param.vol import SABRVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType


def _env(surface):
    return PricingEnvironment(
        rate_curve=FlatRateCurve(rate=0.02),
        valuation_date=datetime(2026, 6, 24),
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=surface,
    )


def test_flat_surface_matches_closed_form_digital():
    # Under a flat surface, replication and N(d2) must agree (smile correction = 0).
    flat = FlatVolSurface(volatility=0.2)
    opt = CashOrNothingDigitalOption(
        strike=100.0, payout=1.0, option_type=OptionType.CALL, maturity=1.0
    )
    price = DigitalOptionAnalyticalEngine().price(opt, _env(flat))
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.02, 0.2
    d2 = (math.log(S / K) + (r - 0.0 - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    expected = math.exp(-r * T) * norm.cdf(d2)
    assert abs(price - expected) < 1e-4


def test_smile_digital_includes_skew_term():
    sabr = SABRVolSurface.from_params(alpha=0.2, beta=0.5, rho=-0.6, nu=0.6, maturity=1.0)
    opt = CashOrNothingDigitalOption(
        strike=110.0, payout=1.0, option_type=OptionType.CALL, maturity=1.0
    )
    env = _env(sabr)
    replicated = DigitalOptionAnalyticalEngine().price(opt, env)
    # Level-only reference: plug sigma(K,T) into N(d2) directly.
    S, K, T, r = 100.0, 110.0, 1.0, 0.02
    sigma = env.get_vol(K, T)
    d2 = (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    level_only = math.exp(-r * T) * norm.cdf(d2)
    assert abs(replicated - level_only) > 1e-4  # skew term is non-trivial
    assert 0.0 <= replicated <= math.exp(-r * T)  # arbitrage bounds


def test_call_put_digital_parity_under_smile():
    sabr = SABRVolSurface.from_params(alpha=0.2, beta=0.5, rho=-0.6, nu=0.6, maturity=1.0)
    env = _env(sabr)
    call = DigitalOptionAnalyticalEngine().price(
        CashOrNothingDigitalOption(
            strike=105.0, payout=1.0, option_type=OptionType.CALL, maturity=1.0
        ),
        env,
    )
    put = DigitalOptionAnalyticalEngine().price(
        CashOrNothingDigitalOption(
            strike=105.0, payout=1.0, option_type=OptionType.PUT, maturity=1.0
        ),
        env,
    )
    assert abs((call + put) - math.exp(-0.02 * 1.0)) < 1e-4
