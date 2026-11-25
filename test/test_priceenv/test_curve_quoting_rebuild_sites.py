"""AC10 / AC7c: every library site that rebuilds a PricingEnvironment keeps its cash-curve quoting (0.15.0 spec D12, D13).

A clone that drops the calendar or a declaration either refuses (an undeclared native curve) or carries a wrapper
the clone cannot re-anchor. The theta routes must see the same market as an environment built at the bumped date.
"""
from datetime import datetime

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.report.autocallable_risk_report import _clone_env
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.dynamicscenario.engine import DynamicScenarioEngine
from quantark.dynamicscenario.fi.engine import FIDynamicScenarioEngine
from quantark.param import SpotQuote
from quantark.param.basis import FlatBasisYield
from quantark.param.div.dividend_yield import ContinuousDividendYield
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.param.vol.vol_surface import FlatVolSurface
from quantark.portfolio import Portfolio
from quantark.portfolio.fi.portfolio import FIPortfolio
from quantark.priceenv import CurveQuotingBasis, PricingEnvironment
from quantark.rfq.builders import build_pricing_env_from_market_kwargs
from quantark.stresstest.stress.stress_applicator import StressApplicator
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.enum import OptionType
from quantark.util.marketdata.converter import MarketDataConverter
from quantark.util.marketdata.models import MarketDataPoint

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2025, 2028))
D0, D1 = datetime(2026, 2, 13), datetime(2026, 2, 24)  # Fri before CNY -> first trading day after
H = datetime(2027, 6, 1)
CQ = CurveQuotingBasis.CALENDAR_YEAR
WRAPPED = ("rate_curve", "div_yield", "basis_yield")


def _declared(d=D0, **kw):
    base = dict(rate_curve=FlatRateCurve(0.03), div_yield=ContinuousDividendYield(0.02),
                basis_yield=FlatBasisYield(0.01), valuation_date=d, spot_quote=SpotQuote(100.0),
                vol_surface=FlatVolSurface(0.2), day_count_convention=DayCountConvention.BUSINESS_DAYS,
                bus_days_in_year=244, calendar=CAL, rate_curve_quoting=CQ, div_yield_quoting=CQ,
                basis_yield_quoting=CQ, trading_clock_horizon=H)
    base.update(kw)
    return PricingEnvironment(**base)


def _assert_same_market(env, ref, fields=WRAPPED):
    """Wrapped on the clone's own calendar, anchored at its date, reading like ``ref`` bitwise."""
    for f in fields:
        w = getattr(env, f)
        assert w.clock.calendar is env.calendar, f
        assert w.time_map.anchor_date == env.valuation_date, f
    for u in (1 / 244.0, 0.3, 1.0):
        assert env.get_discount_factor(u) == ref.get_discount_factor(u)
        assert env.get_div_yield(u) == ref.get_div_yield(u)
        if "basis_yield" in fields:
            assert env.get_basis_yield(u) == ref.get_basis_yield(u)


def _assert_declarations_carried(clone, src):
    for name in ("rate_curve_quoting", "div_yield_quoting", "basis_yield_quoting", "trading_clock_horizon"):
        assert getattr(clone, name) == getattr(src, name), name


def test_stress_applicator_clone():
    src = _declared()
    clone = StressApplicator._clone_pricing_env(src)
    _assert_declarations_carried(clone, src)
    assert clone.calendar is not None
    _assert_same_market(clone, src)


def test_dynamic_scenario_clone():
    src = _declared()
    pf = Portfolio(portfolio_name="p", pricing_environments={"X": src})
    clone = DynamicScenarioEngine()._clone_portfolio(pf).pricing_environments["X"]
    _assert_declarations_carried(clone, src)
    _assert_same_market(clone, src)


def test_fi_dynamic_scenario_clone():
    src = _declared(basis_yield=None, basis_yield_quoting=None)
    pf = FIPortfolio(portfolio_name="p", pricing_environments={"X": src})
    clone = FIDynamicScenarioEngine()._clone_portfolio(pf).pricing_environments["X"]
    _assert_declarations_carried(clone, src)
    _assert_same_market(clone, src, fields=("rate_curve", "div_yield"))


def test_risk_report_clone_re_anchors_on_new_date():
    clone = _clone_env(_declared(), valuation_date=D1)
    assert clone.basis_yield is not None
    _assert_same_market(clone, _declared(D1))


def test_rfq_builder_passes_declarations():
    env = build_pricing_env_from_market_kwargs(dict(
        valuation_date=D0, rate=0.03, q=0.02, spot=100.0, volatility=0.2,
        day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=244, calendar=CAL,
        rate_curve_quoting=CQ, div_yield_quoting=CQ, trading_clock_horizon=H))
    _assert_same_market(env, _declared(basis_yield=None, basis_yield_quoting=None), fields=("rate_curve", "div_yield"))


def test_converter_passes_declarations():
    point = MarketDataPoint(timestamp=D0, asset_name="X", spot=100.0, volatility=0.2, rate=0.03, div_yield=0.02)
    env = MarketDataConverter.to_pricing_environment(
        point, DayCountConvention.BUSINESS_DAYS, 244, calendar=CAL, rate_curve_quoting=CQ, div_yield_quoting=CQ,
        trading_clock_horizon=H)
    _assert_same_market(env, _declared(basis_yield=None, basis_yield_quoting=None), fields=("rate_curve", "div_yield"))


def test_numerical_theta_equals_fresh_env_pnl_bitwise():
    """AC7c theta route: GreeksCalculator's bumped copy sees the market an environment built at D1 sees."""
    expiry = datetime(2026, 12, 15)

    def option():
        return EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, exercise_date=expiry)

    greeks = GreeksCalculator().calculate(product=option(), pricing_env=_declared(), engine=BlackScholesEngine(),
                                          greeks=["theta"], method="numerical")
    theta = greeks.theta if hasattr(greeks, "theta") else greeks["theta"]
    engine = BlackScholesEngine()
    assert theta == engine.price(option(), _declared(D1)) - engine.price(option(), _declared(D0))
