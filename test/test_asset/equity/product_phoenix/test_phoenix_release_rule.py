"""The release rule (spec 3B decision 4): at a knock-out the arrears are paid whatever the spot; the period's
coupon when its barrier is hit."""
import dataclasses
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.lifecycle import PortfolioLifecycleManager
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio import Portfolio
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import BusinessDayConvention, CalendarType, create_calendar
from intraday.conftest import dated_phoenix, flat_env

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
T0 = datetime(2026, 3, 16)


def _c105(arrears=2.0):
    p = dated_phoenix(CAL, T0, coupon_barrier=105.0)                  # knock-out 103 below the coupon barrier
    p.coupon_config = dataclasses.replace(p.coupon_config, initial_coupon_arrears=arrears)
    return p


def _first_obs(p):
    return p.barrier_config.ko_observation_schedule.records[0].observation_date


def test_decision_releases_arrears_below_the_coupon_barrier():
    p = _c105()
    env = flat_env(_first_obs(p), spot=104.0)                         # knocks out, misses the coupon
    d = p.decide_observations_at_valuation(104.0, env)
    assert d.knocked_out and d.coupon == pytest.approx(2.0, abs=1e-12)
    solver = PhoenixPDESolver(PDEParams())
    df = env.get_discount_factor(d.ko_record.settlement_time) if d.ko_record.settlement_time else 1.0
    assert solver._get_immediate_ko_payoff(p, env) == pytest.approx((d.ko_record.payoff + 2.0) * df, abs=1e-12)


class _ConstantEngine:
    def __init__(self, value):
        self.value = float(value)

    def price(self, product, pricing_env):
        return self.value


def _lifecycle_env(date, spot):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot, asset_name="IDX"),
                              vol_surface=FlatVolSurface(volatility=0.20),
                              rate_curve=FlatRateCurve(rate=0.03), valuation_date=date)


def test_lifecycle_books_released_arrears_at_the_knock_out():
    issue, d1, d2 = datetime(2026, 1, 5), datetime(2026, 2, 4), datetime(2026, 3, 4)
    p = create_standard_phoenix(initial_price=100.0, strike=100.0, maturity=1.0, ko_barrier=103.0,
                                ki_barrier=70.0, coupon_barrier=105.0, coupon_rate=0.01,
                                num_observations=2, memory_coupon=True)
    p.initial_date, p.exercise_date, p.maturity = issue, issue + timedelta(days=365), None
    p.settlement_convention = SettlementConvention(lag=2, lag_unit=SettlementLagUnit.CALENDAR_DAYS,
                                                   business_day_convention=BusinessDayConvention.UNADJUSTED)
    p.barrier_config = dataclasses.replace(p.barrier_config, ko_observation_dates=None,
        ko_observation_schedule=ObservationSchedule(records=[
            ObservationRecord(observation_date=d1, barrier=103.0),
            ObservationRecord(observation_date=d2, barrier=103.0)]))
    p.validate()   # terms set on a built product: derive its accrual periods
    portfolio = Portfolio(portfolio_name="release-rule",
                          pricing_environments={"IDX": _lifecycle_env(d1 - timedelta(days=1), 100.0)})
    portfolio.add_position(product=p, quantity=1.0, entry_price=0.0, underlying="IDX",
                           engine=_ConstantEngine(7.0), entry_timestamp=issue)
    manager = PortfolioLifecycleManager(base_date=issue)
    manager.register_positions(portfolio)

    def advance(date, spot):
        env = portfolio.pricing_environments["IDX"]
        env.valuation_date, env.spot_quote.spot = date, spot
        return manager.process_day(portfolio, day_index=0, day_date=date)

    first = advance(d1, 100.0)                                         # below 103 and below 105: missed
    assert "COUPON" not in [e.event.event_type.value for e in first]
    second = advance(d2, 104.0)                                        # knocks out, misses the coupon
    coupons = [e.event for e in second if e.event.event_type.value == "COUPON"]
    assert len(coupons) == 1
    assert coupons[0].cashflow == pytest.approx(p.get_coupon_payoff(0, year_fraction=p.coupon_period_fraction(0)),
                                                abs=1e-12)
    assert coupons[0].realized_cashflow.payment_date == d2 + timedelta(days=2)
