"""The two reports' ad hoc shifted-yield / shifted-vol wrappers delegate to the
``parallel_shifted`` primitives.

- ``greek_conventions_report``: its private ``_ShiftedVolSurface`` forwarded every
  unknown attribute to the base, so on a TradingClockVolSurface it advertised the
  exact total-variance protocol of the UNSHIFTED inner; the primitive fails closed
  there (patch spec 2026-09-03 §14) and keeps the shape everywhere else.
- ``autocallable_risk_report._shift_dividend_yield``: kept a ``max(0.0, .)`` zero
  floor on continuous/term yields that b410a30f removed library-wide (signed
  carry), while its own fallback and the report's three direct
  ``ShiftedDividendYield`` uses never floored. One bounded, signed path now.
"""
from datetime import datetime

import pytest

from quantark.asset.equity.report.autocallable_risk_report import _shift_dividend_yield
from quantark.asset.equity.report.term_structure import ShiftedDividendYield
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import (
    ContinuousDividendYield, ParallelShiftDividendYield, TermStructureDividendYield,
)
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
from quantark.util.exceptions import ValidationError

from dcn_fixtures import DCN_A, FLAT, SSE, flat_env, make_dcn, term_env


# --- autocallable risk report -------------------------------------------------------------

def test_risk_report_shift_is_signed_for_a_continuous_yield():
    q = _shift_dividend_yield(ContinuousDividendYield(0.01), -0.03)
    assert q.get_yield(1.0) == pytest.approx(-0.02)          # no zero floor (b410a30f)


def test_risk_report_shift_is_signed_for_a_term_yield():
    base = TermStructureDividendYield(times=[0.5, 1.0, 2.0], yields=[0.01, 0.015, 0.02])
    q = _shift_dividend_yield(base, -0.03)
    assert q.get_yield(1.0) == pytest.approx(-0.015)
    assert q.get_yield(0.5) == pytest.approx(-0.02)


def test_risk_report_zero_shift_returns_the_base_object():
    base = ContinuousDividendYield(0.01)
    assert _shift_dividend_yield(base, 0.0) is base


def test_report_shifted_yield_is_the_primitive_with_the_carry_bound():
    q = ShiftedDividendYield(base=ContinuousDividendYield(0.01), shift=0.02)
    assert isinstance(q, ParallelShiftDividendYield)
    assert q.get_yield(1.0) == pytest.approx(0.03)
    with pytest.raises(ValidationError):                      # |q| <= 1 policy kept
        ShiftedDividendYield(base=ContinuousDividendYield(0.99), shift=0.05).get_yield(1.0)
    with pytest.raises(ValidationError):                      # shift validation kept
        ShiftedDividendYield(base=ContinuousDividendYield(0.0), shift=-1.5)


# --- greek conventions report ------------------------------------------------------------

def test_cash_greeks_vega_on_a_clock_wrapped_surface_fails_closed():
    from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
    from quantark.asset.equity.riskmeasures.greek_conventions_report import _reprice_vol
    from quantark.param.vol.trading_clock_surface import TradingClockVolSurface

    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2023, 2026))
    m = BusinessTimeMap(TradingClock(cal, 244), datetime(2023, 1, 3), datetime(2025, 6, 30))
    env = PricingEnvironment(
        rate_curve=FlatRateCurve(FLAT["r"]), valuation_date=datetime(2023, 1, 3),
        spot_quote=SpotQuote(6000.0), vol_surface=TradingClockVolSurface(FlatVolSurface(0.2), m),
        div_yield=ContinuousDividendYield(FLAT["q"]),
        day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=244, calendar=cal,
    )
    with pytest.raises(ValidationError, match="unit"):
        _reprice_vol(make_dcn(DCN_A), env, DCNMCEngine(num_paths=2 ** 10, seed=42), 0.01)


def test_cash_greeks_rho_and_rhoq_on_term_structures_are_parallel_shifts():
    """Regression guard for the delegation: the report's rate/div reprices equal
    a manual parallel shift of the (term) curve and the (term) yield."""
    from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
    from quantark.asset.equity.riskmeasures.greek_conventions_report import (
        _reprice_div, _reprice_rate,
    )
    from copy import deepcopy

    env = term_env(**FLAT)
    env.rate_curve = LinearRateCurve([(0.5, 0.01), (1.0, 0.03), (2.0, 0.05)])
    p, e = make_dcn(DCN_A), DCNMCEngine(num_paths=2 ** 11, seed=42)
    up_r = deepcopy(env); up_r.rate_curve = env.rate_curve.parallel_shifted(1e-4)
    up_q = deepcopy(env); up_q.div_yield = env.div_yield.parallel_shifted(1e-4)
    assert _reprice_rate(p, env, e, 1e-4) == pytest.approx(e.price(p, up_r), rel=1e-12)
    assert _reprice_div(p, env, e, 1e-4) == pytest.approx(e.price(p, up_q), rel=1e-12)


def test_cash_greeks_report_still_builds_on_flat_env():
    from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
    from quantark.asset.equity.riskmeasures.greek_conventions_report import build_cash_greeks_report

    rep = build_cash_greeks_report(make_dcn(DCN_A), flat_env(**FLAT),
                                   DCNMCEngine(num_paths=2 ** 11, seed=42), calendar=SSE)
    assert rep.rhoq_1pct < 0.0 and rep.vega_1pct != 0.0
