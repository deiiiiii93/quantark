"""Tests for the credit dynamic scenario engine."""
from datetime import datetime

import pytest

from quantark.asset.credit.engine.analytical import CDSReducedFormEngine
from quantark.asset.credit.product import CDS, ProtectionSide
from quantark.dynamicscenario import (
    CreditDynamicScenarioEngine,
    CreditPathLibrary,
    get_engine_for_portfolio,
)
from quantark.param import FlatRateCurve
from quantark.param.credit import FlatHazardCurve
from quantark.portfolio import CreditPortfolio
from quantark.priceenv import CreditPricingEnvironment


def _portfolio(side=ProtectionSide.BUY):
    env = CreditPricingEnvironment(
        valuation_date=datetime(2026, 6, 13),
        discount_curve=FlatRateCurve(rate=0.03),
        hazard_curve=FlatHazardCurve(hazard_rate=0.02),
    )
    pf = CreditPortfolio(portfolio_name="cr", pricing_environments={"ACME": env})
    pf.add_position(
        product=CDS(notional=10_000_000, maturity=5.0, recovery_rate=0.4,
                    coupon_spread=0.01, side=side),
        quantity=1.0, entry_price=0.0, reference_entity="ACME",
        engine=CDSReducedFormEngine(),
    )
    return pf


def test_factory_dispatches_credit_engine():
    engine = get_engine_for_portfolio(_portfolio())
    assert isinstance(engine, CreditDynamicScenarioEngine)


def test_hazard_widening_path_profits_protection_buyer():
    engine = CreditDynamicScenarioEngine()
    path = CreditPathLibrary.hazard_widening(days=5, bps_per_day=10.0)
    results = engine.run(_portfolio(side=ProtectionSide.BUY), path)
    assert results.num_days == 5
    # Protection buyer's cumulative P&L is positive as hazard widens day over day.
    assert results.total_pnl > 0
    assert results.day_results[-1].greeks["cs01"] != 0.0
    assert results.day_results[-1].greeks["hazard01"] != 0.0


def test_credit_crisis_path_runs():
    engine = CreditDynamicScenarioEngine()
    path = CreditPathLibrary.credit_crisis(days=6)
    results = engine.run(_portfolio(), path)
    assert results.num_days == 6


def test_hedging_not_supported_inline():
    engine = CreditDynamicScenarioEngine()
    path = CreditPathLibrary.hazard_widening(days=2)
    with pytest.raises(NotImplementedError):
        engine.run(_portfolio(), path, hedge_strategy=object())
