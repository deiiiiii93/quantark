"""GreeksCalculator's closed forms under a terminal payment lag.

A European vanilla paid at T_pay > T is worth DF(T, T_pay) x its value at T (``BlackScholesEngine.price``); its
spot, vol, dividend and time greeks scale by DF(T, T_pay) and its rho takes the lag's own discounting,
-(T_pay - T) x value. The analytic route ignored the lag (price 96.0158 for 95.2983 at a quarter-year lag). The
oracle here is the numerical route, which bumps the engine price and so honours the lag: with a lag the analytic
route must agree with it as closely as it does without one.
"""
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType

#: the default set (greeks=None: the base dict, theta components and dividend_rho included) and the extended names
#: the analytical route takes in an explicit list
REQUESTS = (None, ["price", "delta", "gamma", "vega", "rho", "speed", "zomma", "charm", "color", "vega_theta",
                   "dividend_volga"])


def _env(spot):
    return PricingEnvironment(rate_curve=FlatRateCurve(0.03), valuation_date=datetime(2026, 1, 2),
                              spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=0.22),
                              div_yield=ContinuousDividendYield(0.01))


def _option(opt, lag, maturity=1.0):
    kw = {} if lag is None else {"settlement_convention": SettlementConvention(
        lag=lag, lag_unit=SettlementLagUnit.YEAR_FRACTION)}
    return EuropeanVanillaOption(strike=100.0, option_type=opt, maturity=maturity, contract_multiplier=10.0, **kw)


@pytest.mark.parametrize("spot", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("opt", [OptionType.CALL, OptionType.PUT])
def test_the_analytic_price_is_the_engines_under_a_lag(opt, spot):
    product, env = _option(opt, 0.25), _env(spot)
    got = GreeksCalculator().calculate(product, env, BlackScholesEngine(), method="analytical", greeks=["price"])
    assert got["price"] == pytest.approx(BlackScholesEngine().price(product, env), rel=1e-12)


@pytest.mark.parametrize("request_set", [0, 1], ids=["default", "extended"])
@pytest.mark.parametrize("spot", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("opt", [OptionType.CALL, OptionType.PUT])
def test_the_analytic_greeks_agree_with_the_bumped_engine_under_a_lag(opt, spot, request_set):
    calc, engine, env, greeks = GreeksCalculator(), BlackScholesEngine(), _env(spot), REQUESTS[request_set]
    gap = {}
    for lag in (None, 0.25):
        product = _option(opt, lag)
        a = calc.calculate(product, env, engine, method="analytical", greeks=greeks)
        n = calc.calculate(product, env, engine, method="numerical", greeks=greeks)
        keys = sorted(set(a) & set(n))
        gap[lag] = {k: abs(a[k] - n[k]) / max(1.0, abs(n[k])) for k in keys}
    assert len(gap[0.25]) >= 9
    if request_set == 0:
        a = calc.calculate(_option(opt, 0.25), env, engine, method="analytical")
        assert a["convexity_theta"] + a["r_theta"] + a["q_theta"] == pytest.approx(a["theta"], rel=1e-14)
    # convexity_theta / r_theta are a split convention, not a derivative: the numerical 'estimate' infers r_theta
    # from rho, which under a lag carries the payment's own discounting, shifting it by r x lag x V / 365 (ATM call
    # 0.00196 of -0.0390) though that discount does not decay with time on a flat curve; the analytic split is the
    # Black-Scholes PDE's own terms scaled by DF(T, T_pay). theta itself is compared.
    for key in sorted(set(gap[0.25]) - {"convexity_theta", "r_theta"}):
        # the bump error of the numerical route is the same with and without the lag (scaled by DF(T, T_pay) < 1)
        assert gap[0.25][key] <= 1.5 * gap[None][key] + 1e-9, (key, gap[0.25][key], gap[None][key])


@pytest.mark.parametrize("opt", [OptionType.CALL, OptionType.PUT])
def test_at_expiry_the_lagged_payoff_is_discounted(opt):
    """At expiry the value is the payoff discounted over the lag; delta scales alike, and rho is the lag's own."""
    product, env = _option(opt, 0.25, maturity=1e-12), _env(90.0 if opt is OptionType.PUT else 110.0)
    got = GreeksCalculator().calculate(product, env, BlackScholesEngine(), method="analytical",
                                       greeks=["price", "delta", "rho"])
    price = BlackScholesEngine().price(product, env)
    assert got["price"] == pytest.approx(price, rel=1e-12)
    assert abs(got["delta"]) == pytest.approx(10.0 * price / 100.0, rel=1e-12)      # |payoff| / 10 per unit spot
    assert got["rho"] == pytest.approx(-0.25 * price / 100.0, rel=1e-12)
