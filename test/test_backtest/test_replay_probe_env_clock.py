"""The replay's maturity probe env carries the run's clock (0.16.0, DCN/dated run D8).

``ProductReplay._remaining_maturity_years`` measures the product's remaining maturity on a throwaway probe env; the
day's pricing env comes from ``ProductReplay.build_env``. Both pass the same explicit clock keywords (the calendar-day
defaults, or the run's business-day clock under ``AutocallableEngineConfig.trading_clock``, run 7 D4). The guard keeps
it that way: a clock field passed to one construction and not the other fails here. Run 7 AC3 widens it to
``_env_with``, the scenario / carry / surface / book env copy, which must keep every clock field (``replace``).
"""
import ast
import inspect
import textwrap

from quantark.backtest.replay.product_replay import ProductReplay

CLOCK_KEYWORDS = {"day_count_convention", "calendar", "bus_days_in_year", "holiday_time_adjustment",
                  "trading_clock_horizon", "rate_curve_quoting", "div_yield_quoting", "basis_yield_quoting"}


def _env_clock_keywords(method):
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) == "PricingEnvironment"]
    assert calls, f"{method.__name__} builds no PricingEnvironment"
    return [frozenset(k.arg for k in call.keywords if k.arg in CLOCK_KEYWORDS) for call in calls]


def test_probe_env_and_built_env_share_the_clock():
    built = set(_env_clock_keywords(ProductReplay.build_env))
    probe = set(_env_clock_keywords(ProductReplay._remaining_maturity_years))
    assert built == probe, f"build_env clock keywords {built} != probe {probe}"


# --------------------------------------------------------------------------- run 7 AC3: ``_env_with`` keeps the clock


def test_env_with_rebuilds_by_replace_not_field_by_field():
    """``_env_with`` (scenario / carry / surface / book callbacks) builds no literal ``PricingEnvironment``: a
    field-by-field rebuild drops every clock keyword it does not name; ``dataclasses.replace`` keeps them all."""
    from quantark.backtest.replay.product_replay import _env_with

    tree = ast.parse(textwrap.dedent(inspect.getsource(_env_with)))
    names = [getattr(node.func, "id", None) or getattr(node.func, "attr", None)
             for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert "PricingEnvironment" not in names
    assert "replace" in names


def test_env_with_keeps_every_clock_field_of_a_business_day_env():
    from datetime import datetime

    from quantark.backtest.replay.product_replay import _env_with
    from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
    from quantark.param.div.dividend_yield import ContinuousDividendYield
    from quantark.priceenv import CurveQuotingBasis, PricingEnvironment
    from quantark.util.calendar import BusinessDayConvention, CalendarType, DayCountConvention, create_calendar

    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2025, 2028))
    cq = CurveQuotingBasis.CALENDAR_YEAR
    env = PricingEnvironment(
        spot_quote=SpotQuote(100.0), vol_surface=FlatVolSurface(0.2), rate_curve=FlatRateCurve(0.02),
        div_yield=ContinuousDividendYield(0.01), valuation_date=datetime(2026, 1, 5),
        day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=244, calendar=cal,
        holiday_time_adjustment=BusinessDayConvention.FOLLOWING, rate_curve_quoting=cq, div_yield_quoting=cq,
        basis_yield_quoting=cq, trading_clock_horizon=datetime(2027, 6, 1))
    bumped = _env_with(env, spot=101.0, div_yield=ContinuousDividendYield(0.02), underlying="X")
    for name in CLOCK_KEYWORDS:
        assert getattr(bumped, name) == getattr(env, name), name
    assert bumped.calendar is cal
    assert bumped.rate_curve is env.rate_curve and bumped.vol_surface is env.vol_surface
    assert bumped.div_yield.clock.calendar is cal   # the native replacement is wrapped under the declaration
