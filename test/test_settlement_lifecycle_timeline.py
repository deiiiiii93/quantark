"""Determination, pending-receivable, and paid-cash lifecycle timelines."""

from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.lifecycle import (
    BarrierLifecycleTracker,
    PortfolioLifecycleManager,
)
from quantark.asset.equity.product.option import BarrierOption
from quantark.asset.equity.product.option.observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
)
from quantark.asset.equity.product.option.phoenix_helpers import (
    create_standard_phoenix,
)
from quantark.asset.equity.settlement import (
    SettlementConvention,
    SettlementLagUnit,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio import Portfolio
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import BusinessDayConvention
from quantark.util.enum import BarrierType, OptionType


RATE = 0.03
ISSUE = datetime(2026, 1, 5)
DETERMINATION = datetime(2026, 2, 4)
PAYMENT = DETERMINATION + timedelta(days=2)


class ConstantEngine:
    """Minimal live-claim pricer used to isolate lifecycle accounting."""

    def __init__(self, value):
        self.value = float(value)

    def price(self, product, pricing_env):
        return self.value


def _env(date, spot):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot, asset_name="IDX"),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=RATE),
        valuation_date=date,
    )


def _portfolio(product, *, date, spot, live_value=7.0):
    portfolio = Portfolio(
        portfolio_name="settlement-timeline",
        pricing_environments={"IDX": _env(date, spot)},
    )
    portfolio.add_position(
        product=product,
        quantity=1.0,
        entry_price=0.0,
        underlying="IDX",
        engine=ConstantEngine(live_value),
        entry_timestamp=ISSUE,
    )
    return portfolio


def _advance(manager, portfolio, date, spot):
    env = portfolio.pricing_environments["IDX"]
    env.valuation_date = date
    env.spot_quote.spot = spot
    return manager.process_day(portfolio, day_index=0, day_date=date)


def _two_day_convention():
    return SettlementConvention(
        lag=2,
        lag_unit=SettlementLagUnit.CALENDAR_DAYS,
        business_day_convention=BusinessDayConvention.UNADJUSTED,
    )


def test_terminal_claim_moves_from_contingent_to_pending_to_paid_once():
    product = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        exercise_date=DETERMINATION + timedelta(days=30),
        rebate=2.0,
        pay_at_hit=True,
        settlement_convention=_two_day_convention(),
    )
    portfolio = _portfolio(
        product,
        date=DETERMINATION - timedelta(days=1),
        spot=100.0,
    )
    manager = PortfolioLifecycleManager(base_date=ISSUE)
    manager.register_positions(portfolio)

    assert portfolio.get_portfolio_value() == pytest.approx(7.0)
    assert manager.pending_receivable_pv == 0.0
    assert manager.paid_cash == 0.0

    events = _advance(manager, portfolio, DETERMINATION, 112.0)
    assert len(events) == 1
    assert events[0].event.realized_cashflow is not None
    assert events[0].event.realized_cashflow.payment_date == PAYMENT
    assert len(portfolio) == 0
    assert manager.pending_receivable_pv == pytest.approx(
        2.0
        * portfolio.pricing_environments["IDX"].get_discount_factor(2.0 / 365.0)
    )
    assert manager.paid_cash == 0.0

    _advance(
        manager,
        portfolio,
        DETERMINATION + timedelta(days=1),
        112.0,
    )
    assert manager.pending_receivable_pv == pytest.approx(
        2.0
        * portfolio.pricing_environments["IDX"].get_discount_factor(1.0 / 365.0)
    )
    assert manager.paid_cash == 0.0

    _advance(manager, portfolio, PAYMENT, 112.0)
    assert manager.pending_receivable_pv == 0.0
    assert manager.paid_cash == pytest.approx(2.0)

    _advance(manager, portfolio, PAYMENT + timedelta(days=1), 112.0)
    assert manager.pending_receivable_pv == 0.0
    assert manager.paid_cash == pytest.approx(2.0)
    assert len(manager.ledger.cashflows) == 1


def test_pending_phoenix_coupon_coexists_with_live_continuation_value():
    phoenix = create_standard_phoenix(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        ko_barrier=103.0,
        ki_barrier=70.0,
        coupon_barrier=85.0,
        coupon_rate=0.01,
        num_observations=1,
        memory_coupon=False,
    )
    phoenix.initial_date = ISSUE
    phoenix.exercise_date = ISSUE + timedelta(days=365)
    phoenix.maturity = None
    phoenix.settlement_convention = _two_day_convention()
    phoenix.barrier_config = replace(
        phoenix.barrier_config,
        ko_observation_dates=None,
        ko_observation_schedule=ObservationSchedule(
            records=[
                ObservationRecord(
                    observation_date=DETERMINATION,
                    barrier=103.0,
                )
            ]
        ),
    )
    portfolio = _portfolio(
        phoenix,
        date=DETERMINATION - timedelta(days=1),
        spot=100.0,
        live_value=7.0,
    )
    manager = PortfolioLifecycleManager(base_date=ISSUE)
    manager.register_positions(portfolio)

    events = _advance(manager, portfolio, DETERMINATION, 100.0)
    assert [item.event.event_type.value for item in events] == ["COUPON"]
    assert len(portfolio) == 1
    assert portfolio.get_portfolio_value() == pytest.approx(7.0)
    assert manager.pending_receivable_pv == pytest.approx(
        events[0].event.cashflow
        * portfolio.pricing_environments["IDX"].get_discount_factor(2.0 / 365.0)
    )
    assert manager.paid_cash == 0.0

    _advance(manager, portfolio, PAYMENT, 100.0)
    assert len(portfolio) == 1
    assert manager.pending_receivable_pv == 0.0
    assert manager.paid_cash == pytest.approx(events[0].event.cashflow)


def test_duplicate_terminal_observation_does_not_duplicate_cashflow():
    product = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=2.0,
        pay_at_hit=True,
        settlement_convention=_two_day_convention(),
    )
    tracker = BarrierLifecycleTracker(
        product=product,
        quantity=1.0,
        start_date=ISSUE,
    )
    env = _env(DETERMINATION, 112.0)

    assert len(tracker.observe(DETERMINATION, env, 112.0)) == 1
    assert tracker.observe(DETERMINATION, env, 112.0) == []
    assert len(tracker.state.ledger.cashflows) == 1
