"""
Cross-validation of analytical FX Greeks against finite-difference Greeks.

Ports the legacy test_fx_greeks_analytical_vs_fdm: for vanilla and digital
options across moneyness, the closed-form Greeks must agree with
bump-and-reprice within tight tolerances.
"""

from datetime import datetime

import pytest

from quantark.asset.fx.engine.analytical import (
    FxDigitalOptionAnalyticalEngine,
    GarmanKohlhagenEngine,
)
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxDigitalOption, FxVanillaOption
from quantark.asset.fx.riskmeasures import FxGreeksCalculator
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import GreeksCalculationMode, OptionType

SPOT = 1.20
R_DOM = 0.05
R_FOR = 0.03
VOL = 0.10
T = 1.0
NOTIONAL = 1_000_000.0
PAYOUT = 100_000.0

STRIKES = [1.08, 1.20, 1.32]  # ITM / ATM / OTM for a call


def make_env():
    return FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12),
        spot_quote=SpotQuote(spot=SPOT),
        domestic_curve=FlatRateCurve(rate=R_DOM),
        foreign_curve=FlatRateCurve(rate=R_FOR),
        vol_surface=FlatVolSurface(volatility=VOL),
    )


def vanilla(strike, option_type):
    return FxVanillaOption(
        currency_pair=CurrencyPair("EUR", "USD"),
        strike=strike,
        option_type=option_type,
        maturity=T,
        notional_foreign=NOTIONAL,
    )


def digital(strike, option_type):
    return FxDigitalOption(
        currency_pair=CurrencyPair("EUR", "USD"),
        strike=strike,
        option_type=option_type,
        maturity=T,
        payout=PAYOUT,
    )


# Tolerances: delta/gamma/vega central FDM is O(h^2); theta is a 1-day
# forward difference so analytical (instantaneous) theta differs slightly.
TOLERANCES = {
    "delta": 5e-4,
    "gamma": 5e-3,
    "vega": 5e-4,
    "theta": 2e-2,
    "rho_dom": 5e-4,
    "rho_for": 5e-4,
}


@pytest.mark.parametrize("strike", STRIKES)
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_vanilla_analytical_matches_fdm(strike, option_type):
    env = make_env()
    engine = GarmanKohlhagenEngine()
    option = vanilla(strike, option_type)
    calculator = FxGreeksCalculator()

    analytical = calculator.calculate(
        option, env, engine, mode=GreeksCalculationMode.ENGINE
    )
    numerical = calculator.calculate(
        option, env, engine, mode=GreeksCalculationMode.BUMP
    )

    for greek, rel_tol in TOLERANCES.items():
        a, n = analytical[greek], numerical[greek]
        scale = max(abs(a), abs(n), 1e-4 * NOTIONAL)
        assert abs(a - n) / scale < rel_tol, (
            f"{greek} mismatch for K={strike} {option_type}: "
            f"analytical={a:.6f}, fdm={n:.6f}"
        )


@pytest.mark.parametrize("strike", STRIKES)
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_digital_analytical_matches_fdm(strike, option_type):
    env = make_env()
    engine = FxDigitalOptionAnalyticalEngine()
    option = digital(strike, option_type)
    calculator = FxGreeksCalculator()

    analytical = calculator.calculate(
        option, env, engine, mode=GreeksCalculationMode.ENGINE
    )
    numerical = calculator.calculate(
        option, env, engine, mode=GreeksCalculationMode.BUMP
    )

    for greek, rel_tol in TOLERANCES.items():
        a, n = analytical[greek], numerical[greek]
        scale = max(abs(a), abs(n), 1e-4 * PAYOUT)
        assert abs(a - n) / scale < rel_tol, (
            f"{greek} mismatch for K={strike} {option_type}: "
            f"analytical={a:.6f}, fdm={n:.6f}"
        )


def test_auto_mode_uses_engine_greeks():
    env = make_env()
    engine = GarmanKohlhagenEngine()
    option = vanilla(1.20, OptionType.CALL)
    calculator = FxGreeksCalculator()

    auto = calculator.calculate(option, env, engine)
    analytical = engine.calculate_greeks(option, env)
    assert auto["delta"] == pytest.approx(analytical["delta"], rel=1e-12)
