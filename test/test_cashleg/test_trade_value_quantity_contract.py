"""get_trade_value / get_trade_value_breakdown follow the §11.8 quantity contract.

Product PV scales by the signed quantity; cash legs are ABSOLUTE for the trade,
carry the position holder's direction, and are not scaled by the quantity or its
sign. That is get_trade_risk's contract (and the execution kernel's, which mirrors
it bitwise); the value-side methods used to multiply legs by quantity as well, so
a short position saw its legs flipped and a multi-lot one saw them multiplied.
"""
import math
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.cashleg import DeterministicLeg, FixedPayoffLeg, LegDirection, PaymentTrigger
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.equity.position import EquityPosition
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, OptionType

T0 = datetime(2026, 1, 1)


def _env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.05), div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=T0)


def _vanilla_position(quantity, legs):
    return EquityPosition(
        product=EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0),
        quantity=quantity, entry_price=0.0, underlying="SPX", engine=BlackScholesEngine(),
        entry_timestamp=T0, cash_legs=legs)


def _fee(amount=100.0, when=0.5):
    return DeterministicLeg(amount=amount, payment_time=when,
                            direction=LegDirection.BUYER_PAYS, name="fee")


@pytest.mark.parametrize("quantity", [3.0, -1.0, -2.5])
def test_trade_value_scales_the_product_only(quantity):
    env = _env()
    pos = _vanilla_position(quantity, [_fee()])
    unit_npv = BlackScholesEngine().price(pos.product, env)
    leg = -100.0 * math.exp(-0.05 * 0.5)          # absolute, holder pays, whatever the quantity
    assert pos.get_trade_value(env) == pytest.approx(quantity * unit_npv + leg, rel=1e-12)


def test_breakdown_legs_are_absolute_and_total_matches_trade_value():
    env = _env()
    pos = _vanilla_position(-2.5, [_fee(), DeterministicLeg(
        amount=40.0, payment_time=1.0, direction=LegDirection.BUYER_RECEIVES, name="rebate")])
    breakdown = pos.get_trade_value_breakdown(env)
    by_name = {v.name: v.pv for v in breakdown.leg_pvs.values()}
    assert by_name["fee"] == pytest.approx(-100.0 * math.exp(-0.05 * 0.5), rel=1e-12)
    assert by_name["rebate"] == pytest.approx(40.0 * math.exp(-0.05), rel=1e-12)
    assert breakdown.product_npv == pytest.approx(
        -2.5 * BlackScholesEngine().price(pos.product, env), rel=1e-12)
    assert breakdown.total == pytest.approx(pos.get_trade_value(env), rel=1e-12)


def test_value_side_matches_trade_risk_price_for_a_short_funded_note():
    # The desk case: a short note whose holder returns the client's deposit at
    # termination. Value and risk must agree on the same trade PV.
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2024, 1, 1))
    product = SnowballOption(
        initial_price=100.0, strike=100.0, contract_multiplier=10_000.0, maturity=1.0,
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[i / 12 for i in range(1, 13)], ki_barrier=75.0,
            ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True))
    deposit = 1_000_000.0
    legs = [FixedPayoffLeg(direction=LegDirection.BUYER_PAYS, amount=deposit,
                           trigger=PaymentTrigger.AT_KO, name="deposit at KO"),
            FixedPayoffLeg(direction=LegDirection.BUYER_PAYS, amount=deposit,
                           trigger=PaymentTrigger.AT_MATURITY_ANY, name="deposit at maturity")]
    engine = SnowballPDESolver(PDEParams())
    pos = EquityPosition(product=product, quantity=-1.0, entry_price=0.0, underlying="TEST",
                         engine=engine, entry_timestamp=datetime(2024, 1, 1), cash_legs=legs)
    risk = pos.get_trade_risk(env, GreeksCalculator(), ["delta"])
    value = pos.get_trade_value(env)
    breakdown = pos.get_trade_value_breakdown(env)
    assert value == pytest.approx(risk.total["price"], rel=1e-9)
    assert breakdown.total == pytest.approx(risk.total["price"], rel=1e-9)
    # A deposit the holder owes is a liability whatever the note's sign.
    assert sum(v.pv for v in breakdown.leg_pvs.values()) < 0.0
    assert sum(v.pv for v in breakdown.leg_pvs.values()) == pytest.approx(
        sum(v.pv for v in risk.leg_pvs.values()), rel=1e-9)
