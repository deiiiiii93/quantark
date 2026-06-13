"""Tests for exact key-rate IR DV01 on FI positions (SIMM provider)."""
import dataclasses
from datetime import datetime

import pytest

from quantark.asset.bond.engine.discount.bond_discount_engine import BondDiscountEngine
from quantark.asset.bond.product.couponbond.fixed_bond import FixedBond
from quantark.param import FlatRateCurve
from quantark.portfolio import FIPortfolio
from quantark.portfolio.fi.position import FIPosition
from quantark.priceenv import PricingEnvironment
from quantark.simm import SIMMConfig
from quantark.simm.engines.aggregation import SIMMCalculator
from quantark.simm.engines.base import SIMMSensitivityProvider
from quantark.simm.engines.portfolio_adapter import SIMMPortfolioAdapter
from quantark.simm.sensitivity import IRDeltaSensitivity
from quantark.simm.taxonomy import RiskClass
from quantark.util.calendar import (
    BusinessDayConvention,
    CalendarType,
    DayCountConvention,
    create_calendar,
)
from quantark.util.enum import PaymentFrequency

VALUATION_DATE = datetime(2026, 6, 13)


def _env(rate=0.04):
    return PricingEnvironment(rate_curve=FlatRateCurve(rate=rate),
                             valuation_date=VALUATION_DATE)


def _bond(maturity=datetime(2034, 6, 30), coupon=0.0425):
    return FixedBond(
        issue_date=datetime(2021, 6, 30), maturity_date=maturity, denominator=100.0,
        coupon_rate=coupon, payment_frequency=PaymentFrequency.SEMI_ANNUAL,
        day_count_convention=DayCountConvention.ACT_ACT_ISDA,
        calendar=create_calendar(CalendarType.NONE),
        business_day_convention=BusinessDayConvention.UNADJUSTED, settlement_days=0,
    )


def _position(quantity=200.0):
    env = _env()
    bond = _bond()
    return FIPosition(
        product=bond, quantity=quantity, entry_price=100.0, underlying="USD",
        engine=BondDiscountEngine(pricing_env=env), entry_timestamp=VALUATION_DATE,
        notional_per_unit=100.0,
    ), env


def test_fi_position_is_simm_provider():
    assert isinstance(_position()[0], SIMMSensitivityProvider)


def test_emits_ir_delta_at_multiple_vertices():
    pos, env = _position()
    config = SIMMConfig(calculation_currency="USD", calculate_delta=True)
    sens = pos.get_simm_sensitivities(config, {"USD": env})
    deltas = [s for s in sens.sensitivities if isinstance(s, IRDeltaSensitivity)]
    assert len(deltas) >= 2  # a 10y bond touches several curve vertices
    assert all(s.risk_class == RiskClass.INTEREST_RATE for s in deltas)
    assert all(s.currency == "USD" for s in deltas)


def test_keyrate_deltas_sum_to_parallel_dv01():
    """Sum of independently-shocked key-rate deltas equals the parallel delta."""
    pos, env = _position()
    config = SIMMConfig(calculation_currency="USD", calculate_delta=True)
    sens = pos.get_simm_sensitivities(config, {"USD": env})
    keyrate_sum = sum(s.amount for s in sens.sensitivities)

    # Independent parallel +/-1bp on the flat curve.
    h = 1e-4
    up = pos.get_market_value(
        dataclasses.replace(env, rate_curve=FlatRateCurve(rate=0.04 + h)))
    down = pos.get_market_value(
        dataclasses.replace(env, rate_curve=FlatRateCurve(rate=0.04 - h)))
    parallel = (up - down) / 2.0
    assert keyrate_sum == pytest.approx(parallel, rel=1e-3)


def test_long_bond_has_negative_ir_delta():
    pos, env = _position(quantity=200.0)
    config = SIMMConfig(calculation_currency="USD", calculate_delta=True)
    sens = pos.get_simm_sensitivities(config, {"USD": env})
    # Long a bond: +1bp rates -> price down -> negative IR delta.
    assert sum(s.amount for s in sens.sensitivities) < 0


def test_fi_portfolio_simm_margin_positive():
    pos, env = _position()
    pf = FIPortfolio(portfolio_name="fi", pricing_environments={"USD": env})
    pf.positions[pos.position_id] = pos
    config = SIMMConfig(calculation_currency="USD", calculate_delta=True)
    sens = SIMMPortfolioAdapter(config).portfolio_to_sensitivities(pf)
    result = SIMMCalculator(config).calculate(sens)
    assert result.total_margin > 0
