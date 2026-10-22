"""Tests for the credit stress engine."""
from datetime import datetime

import pytest

from quantark.asset.credit.engine.analytical import CDSReducedFormEngine
from quantark.asset.credit.product import CDS, ProtectionSide
from quantark.param import FlatRateCurve
from quantark.param.credit import FlatHazardCurve
from quantark.portfolio import CreditPortfolio
from quantark.priceenv import CreditPricingEnvironment
from quantark.stresstest import CreditStressEngine
from quantark.stresstest.scenario.scenario import Scenario, Stress
from quantark.stresstest.stress.stress_types import StressLevel, StressType


def _portfolio():
    env = CreditPricingEnvironment(
        valuation_date=datetime(2026, 6, 13),
        discount_curve=FlatRateCurve(rate=0.03),
        hazard_curve=FlatHazardCurve(hazard_rate=0.02),
    )
    pf = CreditPortfolio(portfolio_name="cr", pricing_environments={"ACME": env})
    pf.add_position(
        product=CDS(notional=10_000_000, maturity=5.0, recovery_rate=0.4,
                    coupon_spread=0.01, side=ProtectionSide.BUY),
        quantity=1.0, entry_price=0.0, reference_entity="ACME",
        engine=CDSReducedFormEngine(),
    )
    return pf


def _spread_widening(pct=1.0):
    return Scenario(
        name="Spread doubling",
        stresses=[Stress(parameter="spread", stress_type=StressType.PERCENTAGE,
                         stress_value=pct, level=StressLevel.PORTFOLIO)],
    )


def test_spread_widening_helps_protection_buyer():
    engine = CreditStressEngine()
    results = engine.run_static_scenarios(_portfolio(), [_spread_widening(1.0)])
    res = results.scenario_results[0]
    # Protection buyer gains when the issuer's spread widens.
    assert res.portfolio_pnl > 0


def test_entity_level_rate_shock_runs():
    engine = CreditStressEngine()
    scenario = Scenario(
        name="Rates +200bp on ACME",
        stresses=[Stress(parameter="rate", stress_type=StressType.ABSOLUTE,
                         stress_value=0.02, target="ACME", level=StressLevel.UNDERLYING)],
    )
    results = engine.run_static_scenarios(_portfolio(), [scenario])
    assert results.scenario_results[0].portfolio_value != 0.0


def test_unknown_parameter_rejected():
    engine = CreditStressEngine()
    scenario = Scenario(
        name="bad",
        stresses=[Stress(parameter="vol", stress_type=StressType.ABSOLUTE,
                         stress_value=0.1, level=StressLevel.PORTFOLIO)],
    )
    with pytest.raises(Exception):
        engine.run_static_scenarios(_portfolio(), [scenario])
