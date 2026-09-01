"""Analytical closed forms for higher-order greeks vs the FD oracle."""

from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical.black_scholes_engine import (
    BlackScholesEngine,
)
from quantark.asset.equity.product.option.european_vanilla_option import (
    EuropeanVanillaOption,
)
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

STRIKES = (80.0, 100.0, 120.0)
MATURITIES = (0.25, 1.0)
OPTION_TYPES = (OptionType.CALL, OptionType.PUT)

# (rel, abs) tolerances calibrated against the measured FD error on the
# grid above (max observed rel: vega_theta 7.3e-2 on the short-dated deep
# wing where the 1-day discrete step differs most from the instantaneous
# derivative; everything else <= 1.3e-2).
TOLERANCES = {
    "vanna": (2e-2, 1e-8),
    "volga": (1e-2, 1e-6),
    "speed": (2e-2, 1e-9),
    "zomma": (5e-2, 1e-4),
    "charm": (2e-2, 1e-7),
    "color": (2e-2, 1e-8),
    "vega_theta": (1e-1, 1e-6),
    "gamma_theta": (1e-2, 1e-8),
    "delta_q": (5e-2, 1e-3),
    "dividend_volga": (1e-3, 1e-6),
}


def _env() -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2024, 1, 2),
    )


def _numerical_reference(calc, product, env, engine, name):
    if name == "vanna":
        return calc.calculate_numerical_vanna(product, env, engine)
    if name == "volga":
        return calc.calculate_numerical_volga(product, env, engine)
    if name == "speed":
        return calc.calculate_numerical_speed(product, env, engine)
    if name == "zomma":
        return calc.calculate_numerical_zomma(product, env, engine)
    if name == "charm":
        return calc.calculate_numerical_charm(product, env, engine, clock="1d")
    if name == "color":
        return calc.calculate_numerical_color(product, env, engine, clock="1d")
    if name == "vega_theta":
        return calc.calculate_numerical_vega_theta(
            product, env, engine, clock="1d"
        )
    if name == "gamma_theta":
        return calc.calculate_gamma_theta(product, env, engine, clock="1d")
    if name == "delta_q":
        return calc.calculate_numerical_delta_q(product, env, engine)
    if name == "dividend_volga":
        return calc.calculate_numerical_dividend_volga(product, env, engine)
    raise AssertionError(name)


@pytest.mark.parametrize("name", sorted(TOLERANCES))
def test_analytical_vs_numerical_oracle(name):
    calc = GreeksCalculator()
    engine = BlackScholesEngine()
    rel, abs_tol = TOLERANCES[name]
    for strike in STRIKES:
        for maturity in MATURITIES:
            for option_type in OPTION_TYPES:
                product = EuropeanVanillaOption(
                    strike=strike, option_type=option_type, maturity=maturity
                )
                env = _env()
                ana = calc.calculate_analytical_greeks(
                    product, env, greeks=[name]
                )[name]
                num = _numerical_reference(calc, product, env, engine, name)
                assert ana == pytest.approx(num, rel=rel, abs=abs_tol), (
                    f"{name} K={strike} T={maturity} {option_type}: "
                    f"ana={ana} num={num}"
                )


def test_gamma_theta_equals_convexity_theta_for_vanillas():
    calc = GreeksCalculator()
    for strike in STRIKES:
        for option_type in OPTION_TYPES:
            product = EuropeanVanillaOption(
                strike=strike, option_type=option_type, maturity=1.0
            )
            env = _env()
            greeks = calc.calculate_analytical_greeks(
                product, env, greeks=["gamma_theta", "convexity_theta"]
            )
            # Same closed-form expression: exact equality, not approx.
            assert greeks["gamma_theta"] == greeks["convexity_theta"]


def test_analytical_default_keys_unchanged():
    calc = GreeksCalculator()
    product = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )
    greeks = calc.calculate_analytical_greeks(product, _env())
    assert set(greeks.keys()) == {
        "price",
        "delta",
        "gamma",
        "vega",
        "theta",
        "convexity_theta",
        "r_theta",
        "q_theta",
        "rho",
        "dividend_rho",
    }


def test_analytical_greeks_param_returns_requested_only():
    calc = GreeksCalculator()
    product = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )
    greeks = calc.calculate_analytical_greeks(
        product, _env(), greeks=["charm", "delta", "veta"]
    )
    assert set(greeks.keys()) == {"charm", "delta", "vega_theta"}


def test_auto_routes_new_names_to_analytical_for_vanilla():
    calc = GreeksCalculator()
    engine = BlackScholesEngine()
    product = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )
    env = _env()
    auto = calc.calculate(product, env, engine, greeks=["charm", "zomma"])
    explicit = calc.calculate_analytical_greeks(
        product, env, greeks=["charm", "zomma"]
    )
    assert auto == explicit


def test_auto_keeps_vanna_numerical_for_vanilla():
    calc = GreeksCalculator()
    engine = BlackScholesEngine()
    product = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )
    env = _env()
    auto = calc.calculate(product, env, engine, greeks=["vanna"])
    numerical = calc.calculate_numerical_vanna(product, env, engine)
    assert auto["vanna"] == numerical


def test_analytical_rejects_clock_qualified():
    calc = GreeksCalculator()
    product = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )
    with pytest.raises(ValidationError):
        calc.calculate_analytical_greeks(product, _env(), greeks=["theta_1td"])


def test_analytical_rejects_unsupported_method_analytical():
    # Incumbent contract: method="analytical" + vanna still errors (vanna is
    # not in the auto-routing analytical set even though a closed form now
    # exists behind the explicit greeks= parameter).
    calc = GreeksCalculator()
    engine = BlackScholesEngine()
    product = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )
    with pytest.raises(ValidationError):
        calc.calculate(
            product, _env(), engine, method="analytical", greeks=["vanna"]
        )
