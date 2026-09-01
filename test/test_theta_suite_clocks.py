"""Dual-clock (1D/1TD) theta suite: theta, components, gamma_theta, charm,
color, vega_theta; plus the exact theta decomposition mode."""

from datetime import datetime

import numpy as np
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
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

FRIDAY = datetime(2026, 6, 26)  # next CHINA_SSE trading day is Monday 06-29


def _calendar_env_with_trading_calendar() -> PricingEnvironment:
    """ACT/365-style default day count, but with a trading calendar attached
    so 1TD advances Friday -> Monday (3 calendar days of carry)."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=FRIDAY,
        calendar=create_calendar(CalendarType.CHINA_SSE),
    )


def _business_env() -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=FRIDAY,
        day_count_convention=DayCountConvention.BUSINESS_DAYS,
        bus_days_in_year=244,
        calendar=create_calendar(CalendarType.CHINA_SSE),
    )


def _product() -> EuropeanVanillaOption:
    return EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )


def test_theta_1d_and_1td_coexist_in_one_call():
    calc = GreeksCalculator()
    greeks = calc.calculate_numerical_greeks(
        _product(),
        _calendar_env_with_trading_calendar(),
        BlackScholesEngine(),
        greeks=["theta_1d", "theta_1td"],
    )
    assert set(greeks.keys()) == {"theta_1d", "theta_1td"}
    # Over a weekend the two clocks legitimately differ (3 days vs 1 day).
    assert greeks["theta_1td"] != pytest.approx(greeks["theta_1d"])


def test_theta_1td_matches_manual_reprice_at_monday():
    env = _calendar_env_with_trading_calendar()
    product = _product()
    engine = BlackScholesEngine()
    calc = GreeksCalculator()

    expected = calc.calculate_numerical_theta(
        product, env, engine, time_bump_days=1, time_bump_mode="business_days"
    )
    greeks = calc.calculate_numerical_greeks(
        product, env, engine, greeks=["theta_1td"]
    )
    assert greeks["theta_1td"] == pytest.approx(expected, rel=1e-12)


def test_theta_1d_matches_manual_calendar_reprice():
    env = _calendar_env_with_trading_calendar()
    product = _product()
    engine = BlackScholesEngine()
    calc = GreeksCalculator()

    expected = calc.calculate_numerical_theta(
        product, env, engine, time_bump_days=1, time_bump_mode="calendar_days"
    )
    greeks = calc.calculate_numerical_greeks(
        product, env, engine, greeks=["theta_1d"]
    )
    assert greeks["theta_1d"] == pytest.approx(expected, rel=1e-12)


def test_gamma_theta_clock_scaling_exact_ratio():
    env = _calendar_env_with_trading_calendar()
    calc = GreeksCalculator()
    greeks = calc.calculate_numerical_greeks(
        _product(),
        env,
        BlackScholesEngine(),
        greeks=["gamma", "gamma_theta_1d", "gamma_theta_1td"],
    )
    ratio = greeks["gamma_theta_1td"] / greeks["gamma_theta_1d"]
    assert ratio == pytest.approx(365.0 / env.bus_days_in_year, rel=1e-12)
    assert greeks["gamma_theta_1d"] < 0.0  # long-option gamma bleed


def test_gamma_theta_matches_identity_from_measured_gamma():
    env = _calendar_env_with_trading_calendar()
    product = _product()
    calc = GreeksCalculator()
    engine = BlackScholesEngine()
    greeks = calc.calculate_numerical_greeks(
        product, env, engine, greeks=["gamma", "gamma_theta_1d"]
    )
    T = product.get_maturity(env)
    sigma = env.get_vol(product.strike, T)
    expected = -0.5 * sigma**2 * env.spot**2 * greeks["gamma"] / 365.0
    assert greeks["gamma_theta_1d"] == pytest.approx(expected, rel=1e-12)


def test_r_theta_1td_scales_with_step_year_fraction():
    env = _calendar_env_with_trading_calendar()
    calc = GreeksCalculator()
    greeks = calc.calculate_numerical_greeks(
        _product(),
        env,
        BlackScholesEngine(),
        greeks=["r_theta_1d", "r_theta_1td", "q_theta_1d", "q_theta_1td"],
    )
    # Friday -> Monday carries 3 calendar days on an ACT/365-style env.
    assert greeks["r_theta_1td"] / greeks["r_theta_1d"] == pytest.approx(
        3.0, rel=1e-12
    )
    assert greeks["q_theta_1td"] / greeks["q_theta_1d"] == pytest.approx(
        3.0, rel=1e-12
    )


def test_business_env_1d_step_is_zero_time():
    # On a BUSINESS_DAYS day-count env, Fri -> Sat prices zero time passing:
    # theta_1d and its carry components are all zero, consistently.
    env = _business_env()
    calc = GreeksCalculator()
    greeks = calc.calculate_numerical_greeks(
        _product(),
        env,
        BlackScholesEngine(),
        greeks=["theta_1d", "r_theta_1d", "q_theta_1d", "convexity_theta_1d"],
    )
    assert greeks["theta_1d"] == 0.0
    assert greeks["r_theta_1d"] == 0.0
    assert greeks["q_theta_1d"] == 0.0
    assert greeks["convexity_theta_1d"] == 0.0


def test_qualified_components_reconcile_to_qualified_theta():
    env = _calendar_env_with_trading_calendar()
    calc = GreeksCalculator()
    greeks = calc.calculate_numerical_greeks(
        _product(),
        env,
        BlackScholesEngine(),
        greeks=["theta_1td", "r_theta_1td", "q_theta_1td", "convexity_theta_1td"],
    )
    assert greeks["convexity_theta_1td"] == pytest.approx(
        greeks["theta_1td"] - greeks["r_theta_1td"] - greeks["q_theta_1td"],
        rel=1e-12,
        abs=1e-15,
    )


def test_bare_theta_family_unchanged():
    env = _business_env()
    product = _product()
    engine = BlackScholesEngine()
    calc = GreeksCalculator()
    direct = calc.calculate_numerical_theta(product, env, engine)
    greeks = calc.calculate_numerical_greeks(product, env, engine)
    assert greeks["theta"] == pytest.approx(direct, rel=1e-12)


def test_clock_suffix_rejected_for_non_time_greek():
    calc = GreeksCalculator()
    with pytest.raises(ValidationError):
        calc.calculate_numerical_greeks(
            _product(),
            _calendar_env_with_trading_calendar(),
            BlackScholesEngine(),
            greeks=["vanna_1td"],
        )


def test_charm_color_now_computed():
    # The historical silent miss: charm/color validated but never computed.
    calc = GreeksCalculator()
    greeks = calc.calculate_numerical_greeks(
        _product(),
        _calendar_env_with_trading_calendar(),
        BlackScholesEngine(),
        greeks=["charm", "color"],
    )
    assert set(greeks.keys()) == {"charm", "color"}
    assert all(np.isfinite(v) for v in greeks.values())


def test_charm_matches_manual_delta_diff():
    env = _calendar_env_with_trading_calendar()
    product = _product()
    engine = BlackScholesEngine()
    calc = GreeksCalculator()

    base_delta = calc.calculate_numerical_delta(product, env, engine)
    adv_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2026, 6, 27),
        calendar=create_calendar(CalendarType.CHINA_SSE),
    )
    adv_product = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0 - 1.0 / 365.0,
    )
    adv_delta = calc.calculate_numerical_delta(adv_product, adv_env, engine)
    expected = adv_delta - base_delta

    greeks = calc.calculate_numerical_greeks(
        product, env, engine, greeks=["charm_1d"]
    )
    assert greeks["charm_1d"] == pytest.approx(expected, rel=1e-6, abs=1e-10)


def test_color_and_vega_theta_match_manual_diffs():
    env = _calendar_env_with_trading_calendar()
    product = _product()
    engine = BlackScholesEngine()
    calc = GreeksCalculator()

    base_gamma = calc.calculate_numerical_gamma(product, env, engine)
    base_vega = calc.calculate_numerical_vega(product, env, engine)
    adv_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2026, 6, 27),
        calendar=create_calendar(CalendarType.CHINA_SSE),
    )
    adv_product = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0 - 1.0 / 365.0,
    )
    adv_gamma = calc.calculate_numerical_gamma(adv_product, adv_env, engine)
    adv_vega = calc.calculate_numerical_vega(adv_product, adv_env, engine)

    greeks = calc.calculate_numerical_greeks(
        product, env, engine, greeks=["color_1d", "vega_theta_1d"]
    )
    assert greeks["color_1d"] == pytest.approx(
        adv_gamma - base_gamma, rel=1e-6, abs=1e-12
    )
    assert greeks["vega_theta_1d"] == pytest.approx(
        adv_vega - base_vega, rel=1e-6, abs=1e-12
    )


def test_veta_alias_resolves_to_vega_theta():
    calc = GreeksCalculator()
    greeks = calc.calculate_numerical_greeks(
        _product(),
        _calendar_env_with_trading_calendar(),
        BlackScholesEngine(),
        greeks=["veta"],
    )
    assert set(greeks.keys()) == {"vega_theta"}


def test_time_family_zero_when_step_passes_maturity():
    env = _calendar_env_with_trading_calendar()
    # Maturity smaller than the 1-day step: the whole time family is 0.0
    # (the incumbent theta guard, current_maturity <= time_bump).
    product = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1e-4
    )
    calc = GreeksCalculator()
    greeks = calc.calculate_numerical_greeks(
        product,
        env,
        BlackScholesEngine(),
        greeks=["charm_1d", "color_1d", "vega_theta_1d", "theta_1d"],
    )
    assert all(v == 0.0 for v in greeks.values())


def test_exact_mode_components_match_zeroed_env_definition():
    env = _calendar_env_with_trading_calendar()
    product = _product()
    engine = BlackScholesEngine()
    calc = GreeksCalculator()
    greeks = calc.calculate_numerical_greeks(
        product,
        env,
        engine,
        greeks=["theta", "convexity_theta", "r_theta", "q_theta"],
        theta_decomposition_mode="exact",
    )

    env_no_r = _calendar_env_with_trading_calendar()
    env_no_r.rate_curve = FlatRateCurve(0.0)
    env_no_q = _calendar_env_with_trading_calendar()
    env_no_q.div_yield = ContinuousDividendYield(0.0)
    env_no_rq = _calendar_env_with_trading_calendar()
    env_no_rq.rate_curve = FlatRateCurve(0.0)
    env_no_rq.div_yield = ContinuousDividendYield(0.0)
    theta_no_rq = calc.calculate_numerical_theta(product, env_no_rq, engine)
    theta_no_q = calc.calculate_numerical_theta(product, env_no_q, engine)
    theta_no_r = calc.calculate_numerical_theta(product, env_no_r, engine)

    assert greeks["convexity_theta"] == pytest.approx(theta_no_rq, rel=1e-12)
    assert greeks["r_theta"] == pytest.approx(
        theta_no_q - theta_no_rq, rel=1e-12
    )
    assert greeks["q_theta"] == pytest.approx(
        theta_no_r - theta_no_rq, rel=1e-12
    )
    # Reconciliation to total theta holds up to the r/q interaction term.
    total = greeks["convexity_theta"] + greeks["r_theta"] + greeks["q_theta"]
    assert total == pytest.approx(greeks["theta"], abs=5e-3)
    assert greeks["convexity_theta"] < 0.0


def test_exact_mode_differs_from_estimate():
    env = _calendar_env_with_trading_calendar()
    product = _product()
    engine = BlackScholesEngine()
    calc = GreeksCalculator()
    est = calc.calculate_numerical_greeks(
        product, env, engine, greeks=["r_theta"]
    )
    exact = calc.calculate_numerical_greeks(
        product, env, engine, greeks=["r_theta"], theta_decomposition_mode="exact"
    )
    assert exact["r_theta"] != est["r_theta"]


def test_invalid_theta_decomposition_mode_rejected():
    calc = GreeksCalculator()
    with pytest.raises(ValidationError):
        calc.calculate_numerical_greeks(
            _product(),
            _calendar_env_with_trading_calendar(),
            BlackScholesEngine(),
            greeks=["r_theta"],
            theta_decomposition_mode="bogus",
        )
