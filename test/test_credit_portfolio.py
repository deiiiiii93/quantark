"""Tests for the credit portfolio layer and its SIMM integration."""
from datetime import datetime

import pytest

from quantark.asset.credit.engine.analytical import CDSReducedFormEngine
from quantark.asset.credit.product import CDS, ProtectionSide
from quantark.param import FlatRateCurve
from quantark.param.credit import FlatHazardCurve
from quantark.portfolio import BasePortfolio, CreditPosition, CreditPortfolio
from quantark.priceenv import CreditPricingEnvironment
from quantark.simm import SIMMConfig
from quantark.simm.engines.aggregation import SIMMCalculator
from quantark.simm.engines.base import SIMMSensitivityProvider
from quantark.simm.engines.portfolio_adapter import SIMMPortfolioAdapter
from quantark.simm.sensitivity import CreditDeltaSensitivity


def _env(rate=0.03, hazard=0.02):
    return CreditPricingEnvironment(
        valuation_date=datetime(2026, 6, 13),
        discount_curve=FlatRateCurve(rate=rate),
        hazard_curve=FlatHazardCurve(hazard_rate=hazard),
    )


def _cds(spread=0.01, side=ProtectionSide.BUY):
    return CDS(notional=10_000_000, maturity=5.0, recovery_rate=0.4,
               coupon_spread=spread, side=side)


def test_position_market_value_and_greeks_scale_with_quantity():
    env = _env()
    p1 = CreditPosition(product=_cds(), quantity=1.0, engine=CDSReducedFormEngine(),
                        reference_entity="ACME")
    p2 = CreditPosition(product=_cds(), quantity=3.0, engine=CDSReducedFormEngine(),
                        reference_entity="ACME")
    assert p2.get_market_value(env) == pytest.approx(3 * p1.get_market_value(env))
    assert p2.get_greeks(env)["cs01"] == pytest.approx(3 * p1.get_greeks(env)["cs01"])


def test_portfolio_is_base_portfolio():
    pf = CreditPortfolio(portfolio_name="cr", pricing_environments={"ACME": _env()})
    assert isinstance(pf, BasePortfolio)


def test_portfolio_value_and_greeks_aggregate():
    env = _env()
    pf = CreditPortfolio(portfolio_name="cr", pricing_environments={"ACME": env})
    pf.add_position(product=_cds(), quantity=1.0, entry_price=0.0,
                    reference_entity="ACME", engine=CDSReducedFormEngine())
    pf.add_position(product=_cds(side=ProtectionSide.SELL), quantity=1.0, entry_price=0.0,
                    reference_entity="ACME", engine=CDSReducedFormEngine())
    greeks = pf.get_portfolio_greeks()
    # Long + short protection on the same name net to ~0 CS01.
    assert greeks["cs01"] == pytest.approx(0.0, abs=1.0)


def test_position_is_simm_provider():
    assert issubclass(CreditPosition, SIMMSensitivityProvider) or hasattr(
        CreditPosition, "get_simm_sensitivities")


def test_position_emits_credit_delta_sensitivity():
    env = _env()
    pos = CreditPosition(product=_cds(), quantity=1.0, engine=CDSReducedFormEngine(),
                         reference_entity="JPMORGAN", is_qualifying=True,
                         payment_currency="USD")
    sens = pos.get_simm_sensitivities(
        SIMMConfig(calculation_currency="USD", calculate_delta=True), {"JPMORGAN": env}
    )
    deltas = [s for s in sens.sensitivities if isinstance(s, CreditDeltaSensitivity)]
    assert len(deltas) == 1
    assert deltas[0].issuer == "JPMORGAN"
    assert deltas[0].amount > 0          # protection buyer, positive spread delta
    assert deltas[0].bucket_number == 2  # JPMorgan -> Financials (IG)


def test_seasoned_cds_simm_tenor_is_remaining_maturity():
    # A dated 5y CDS valued two years after inception is a 3y exposure; the SIMM
    # credit-delta must be bucketed at the remaining ~3y vertex, not the original
    # 5y tenor, so seasoned-trade margin is consistent with its as-of pricing.
    from datetime import timedelta

    eff = datetime(2026, 6, 13)
    dated = CDS(notional=10_000_000, maturity=5.0, recovery_rate=0.4,
                coupon_spread=0.01, effective_date=eff)
    env = CreditPricingEnvironment(
        valuation_date=eff + timedelta(days=2 * 365),
        discount_curve=FlatRateCurve(rate=0.03),
        hazard_curve=FlatHazardCurve(hazard_rate=0.02),
    )
    pos = CreditPosition(product=dated, quantity=1.0, engine=CDSReducedFormEngine(),
                         reference_entity="JPMORGAN")
    sens = pos.get_simm_sensitivities(
        SIMMConfig(calculation_currency="USD", calculate_delta=True),
        {"JPMORGAN": env},
    )
    delta = next(s for s in sens.sensitivities
                 if isinstance(s, CreditDeltaSensitivity))
    assert delta.tenor == pytest.approx(3.0, abs=0.05)


def test_portfolio_simm_margin_positive():
    env = _env()
    pf = CreditPortfolio(portfolio_name="cr", pricing_environments={"JPMORGAN": env})
    pf.add_position(product=_cds(), quantity=1.0, entry_price=0.0,
                    reference_entity="JPMORGAN", engine=CDSReducedFormEngine())
    config = SIMMConfig(calculation_currency="USD", calculate_delta=True)
    sens = SIMMPortfolioAdapter(config).portfolio_to_sensitivities(pf)
    result = SIMMCalculator(config).calculate(sens)
    assert result.total_margin > 0
