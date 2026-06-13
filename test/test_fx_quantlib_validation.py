"""
Cross-validation of FX engines against QuantLib (optional).

Skipped automatically when QuantLib is not installed. Ports the legacy
test_fx_vanilla_quantlib / test_fx_digital_quantlib comparisons.
"""

from datetime import datetime

import pytest

ql = pytest.importorskip("QuantLib")

from quantark.asset.fx.engine.analytical import (  # noqa: E402
    FxDigitalOptionAnalyticalEngine,
    GarmanKohlhagenEngine,
)
from quantark.asset.fx.product import CurrencyPair  # noqa: E402
from quantark.asset.fx.product.option import (  # noqa: E402
    FxDigitalOption,
    FxVanillaOption,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote  # noqa: E402
from quantark.priceenv import FxPricingEnvironment  # noqa: E402
from quantark.util.enum import OptionType  # noqa: E402

SPOT = 1.20
R_DOM = 0.05
R_FOR = 0.03
VOL = 0.10
T_DAYS = 365
NOTIONAL = 1_000_000.0
PAYOUT = 100_000.0
VALUATION = datetime(2026, 6, 12)


def make_env():
    return FxPricingEnvironment(
        valuation_date=VALUATION,
        spot_quote=SpotQuote(spot=SPOT),
        domestic_curve=FlatRateCurve(rate=R_DOM),
        foreign_curve=FlatRateCurve(rate=R_FOR),
        vol_surface=FlatVolSurface(volatility=VOL),
    )


def quantlib_option(strike, is_call, payoff_kind="vanilla"):
    """Price with QuantLib's Garman-Kohlhagen setup (BSM with q = r_f)."""
    calc_date = ql.Date(VALUATION.day, VALUATION.month, VALUATION.year)
    ql.Settings.instance().evaluationDate = calc_date
    expiry = calc_date + ql.Period(T_DAYS, ql.Days)
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(SPOT))
    dom_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(calc_date, R_DOM, day_count)
    )
    for_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(calc_date, R_FOR, day_count)
    )
    vol_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(calc_date, calendar, VOL, day_count)
    )
    process = ql.GarmanKohlagenProcess(
        spot_handle, for_curve, dom_curve, vol_handle
    )

    option_type = ql.Option.Call if is_call else ql.Option.Put
    if payoff_kind == "vanilla":
        payoff = ql.PlainVanillaPayoff(option_type, strike)
    else:
        payoff = ql.CashOrNothingPayoff(option_type, strike, 1.0)
    exercise = ql.EuropeanExercise(expiry)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option


def quantlib_price(strike, is_call, payoff_kind="vanilla"):
    return quantlib_option(strike, is_call, payoff_kind).NPV()


@pytest.mark.parametrize("strike", [1.10, 1.20, 1.30])
@pytest.mark.parametrize("is_call", [True, False])
def test_vanilla_matches_quantlib(strike, is_call):
    option = FxVanillaOption(
        currency_pair=CurrencyPair("EUR", "USD"),
        strike=strike,
        option_type=OptionType.CALL if is_call else OptionType.PUT,
        maturity=T_DAYS / 365.0,
        notional_foreign=NOTIONAL,
    )
    ours = GarmanKohlhagenEngine().price(option, make_env())
    reference = quantlib_price(strike, is_call) * NOTIONAL
    assert ours == pytest.approx(reference, rel=1e-9)


@pytest.mark.parametrize("strike", [1.10, 1.20, 1.30])
@pytest.mark.parametrize("is_call", [True, False])
def test_digital_matches_quantlib(strike, is_call):
    option = FxDigitalOption(
        currency_pair=CurrencyPair("EUR", "USD"),
        strike=strike,
        option_type=OptionType.CALL if is_call else OptionType.PUT,
        maturity=T_DAYS / 365.0,
        payout=PAYOUT,
    )
    ours = FxDigitalOptionAnalyticalEngine().price(option, make_env())
    reference = quantlib_price(strike, is_call, payoff_kind="digital") * PAYOUT
    assert ours == pytest.approx(reference, rel=1e-9)


@pytest.mark.parametrize("strike", [1.10, 1.20, 1.30])
@pytest.mark.parametrize("is_call", [True, False])
def test_vanilla_greeks_match_quantlib(strike, is_call):
    option = FxVanillaOption(
        currency_pair=CurrencyPair("EUR", "USD"),
        strike=strike,
        option_type=OptionType.CALL if is_call else OptionType.PUT,
        maturity=1.0,
        notional_foreign=NOTIONAL,
    )
    ours = GarmanKohlhagenEngine().calculate_greeks(option, make_env())
    reference = quantlib_option(strike, is_call)
    assert ours["delta"] == pytest.approx(reference.delta() * NOTIONAL, rel=1e-8)
    assert ours["gamma"] == pytest.approx(reference.gamma() * NOTIONAL, rel=1e-8)
    assert ours["vega"] == pytest.approx(reference.vega() * NOTIONAL * 0.01, rel=1e-8)
    assert ours["theta"] == pytest.approx(reference.thetaPerDay() * NOTIONAL, rel=1e-8)
    assert ours["rho_dom"] == pytest.approx(reference.rho() * NOTIONAL / 100.0, rel=1e-8)
    assert ours["rho_for"] == pytest.approx(
        reference.dividendRho() * NOTIONAL / 100.0, rel=1e-8
    )
