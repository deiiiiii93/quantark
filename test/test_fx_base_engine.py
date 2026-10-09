"""
Tests for the FX base product / base engine FDM Greeks contract.

Uses a deliberately simple linear instrument (PV = N * (S - K) * df_dom(T))
whose sensitivities are known in closed form, so the finite-difference
machinery of BaseFxEngine can be verified exactly.
"""

import math
from datetime import datetime, timedelta

import pytest

from quantark.asset.fx.engine import BaseFxEngine
from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
from quantark.asset.fx.product import BaseFxProduct, CurrencyPair, FxVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment
from quantark.util.calendar.business_calendar import CalendarType, create_calendar
from quantark.util.calendar.day_counter import DayCountConvention as DC
from quantark.util.enum import OptionType

SPOT = 1.20
R_DOM = 0.05
R_FOR = 0.03
MATURITY = 1.0
STRIKE = 1.25
NOTIONAL = 1_000_000.0


class _LinearFxProduct(BaseFxProduct):
    """PV = N * (S - K) * df_dom(T): linear in spot, no vol dependence."""

    def __init__(self):
        super().__init__(
            currency_pair=CurrencyPair("EUR", "USD"), maturity=MATURITY
        )
        self.strike = STRIKE
        self.notional = NOTIONAL

    def get_payoff(self, spot: float) -> float:
        return self.notional * (spot - self.strike)

    def validate(self) -> None:
        pass


class _LinearFxEngine(BaseFxEngine):
    def price(self, product, fx_env) -> float:
        t = product.get_maturity(fx_env)
        return (
            product.notional
            * (fx_env.spot - product.strike)
            * fx_env.get_domestic_df(t)
        )


@pytest.fixture
def env():
    return FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12),
        spot_quote=SpotQuote(spot=SPOT),
        domestic_curve=FlatRateCurve(rate=R_DOM),
        foreign_curve=FlatRateCurve(rate=R_FOR),
        vol_surface=FlatVolSurface(volatility=0.10),
    )


@pytest.fixture
def greeks(env):
    product = _LinearFxProduct()
    engine = _LinearFxEngine()
    return engine.calculate_greeks(product, env)


def test_price_included(greeks):
    expected = NOTIONAL * (SPOT - STRIKE) * math.exp(-R_DOM * MATURITY)
    assert greeks["price"] == pytest.approx(expected)


def test_fdm_delta_linear(greeks):
    # dV/dS = N * df_dom
    expected = NOTIONAL * math.exp(-R_DOM * MATURITY)
    assert greeks["delta"] == pytest.approx(expected, rel=1e-6)


def test_fdm_gamma_zero(greeks):
    assert greeks["gamma"] == pytest.approx(0.0, abs=1e-2)


def test_fdm_vega_zero(greeks):
    # No vol dependence
    assert greeks["vega"] == pytest.approx(0.0, abs=1e-8)


def test_fdm_rho_dom(greeks):
    # dV/dr_d = -T * V; reported as dV/dr / 100 (legacy convention)
    pv = NOTIONAL * (SPOT - STRIKE) * math.exp(-R_DOM * MATURITY)
    expected = -MATURITY * pv / 100.0
    assert greeks["rho_dom"] == pytest.approx(expected, rel=1e-4)


def test_fdm_rho_for_zero(greeks):
    # No foreign-rate dependence in this stub
    assert greeks["rho_for"] == pytest.approx(0.0, abs=1e-6)


def test_fdm_theta_daily(greeks):
    # V(T - 1d) - V(T): discount unwinds, value increases toward (negative) payoff?
    # Here S < K so PV < 0 and shrinking T makes df larger => value more negative.
    pv_now = NOTIONAL * (SPOT - STRIKE) * math.exp(-R_DOM * MATURITY)
    pv_next = NOTIONAL * (SPOT - STRIKE) * math.exp(-R_DOM * (MATURITY - 1 / 365))
    expected = pv_next - pv_now
    assert greeks["theta"] == pytest.approx(expected, rel=1e-6)


def test_get_maturity_from_dates(env):
    product = _LinearFxProduct()
    product.maturity = None
    product.expiry_date = datetime(2027, 6, 12)
    t = product.get_maturity(env)
    assert t == pytest.approx(1.0, abs=0.01)


def test_get_delivery_defaults_to_maturity(env):
    product = _LinearFxProduct()
    assert product.get_delivery(env) == pytest.approx(product.get_maturity(env))


def test_is_linear_default_false():
    assert _LinearFxProduct().is_linear is False


# --- Q7 (spec 2026-10-09-fx-tarn-lifecycle section 12): numeric theta steps by the env's year fraction --------------

_FRIDAY = datetime(2026, 6, 12)


def _q7_env(dc, cal=None):
    return FxPricingEnvironment(valuation_date=_FRIDAY, spot_quote=SpotQuote(spot=1.20),
                                domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.03),
                                vol_surface=FlatVolSurface(volatility=0.10), day_count_convention=dc, calendar=cal)


def _q7_pair(e):
    dated = FxVanillaOption(strike=1.2, option_type=OptionType.CALL, expiry_date=_FRIDAY + timedelta(days=365),
                            currency_pair=CurrencyPair("EUR", "USD"), notional_foreign=1.0)
    numeric = FxVanillaOption(strike=1.2, option_type=OptionType.CALL, maturity=dated.get_maturity(e),
                              currency_pair=CurrencyPair("EUR", "USD"), notional_foreign=1.0)
    return numeric, dated


def test_q7_business_days_numeric_theta_matches_dated():  # M9: was -9.135e-05 vs 0.0
    e = _q7_env(DC.BUSINESS_DAYS, create_calendar(CalendarType.TARGET, year_range=(2026, 2028)))
    numeric, dated = _q7_pair(e)
    eng = GarmanKohlhagenEngine()
    tn = eng._fdm_theta(numeric, e, eng.price(numeric, e))
    td = eng._fdm_theta(dated, e, eng.price(dated, e))
    assert tn == td == 0.0  # Friday -> Saturday adds no business day


def test_q7_calendar_days_step_is_unchanged_bitwise():
    e = _q7_env(DC.CALENDAR_DAYS)
    numeric, _ = _q7_pair(e)
    eng = GarmanKohlhagenEngine()
    base = eng.price(numeric, e)
    manual = FxVanillaOption(strike=1.2, option_type=OptionType.CALL, maturity=numeric.maturity - 1.0 / 365.0,
                             currency_pair=CurrencyPair("EUR", "USD"), notional_foreign=1.0)
    assert eng._fdm_theta(numeric, e, base) == (eng.price(manual, e) - base) / 1.0


def test_q7_act_360_steps_one_360th():
    e = _q7_env(DC.ACT_360)
    numeric, _ = _q7_pair(e)
    eng = GarmanKohlhagenEngine()
    base = eng.price(numeric, e)
    manual = FxVanillaOption(strike=1.2, option_type=OptionType.CALL, maturity=numeric.maturity - 1.0 / 360.0,
                             currency_pair=CurrencyPair("EUR", "USD"), notional_foreign=1.0)
    assert eng._fdm_theta(numeric, e, base) == (eng.price(manual, e) - base) / 1.0
