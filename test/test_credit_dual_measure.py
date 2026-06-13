"""
Tests for the dual hazard/spread credit risk-measure convention.

* ``cs01 = hazard01 / (1 - R)`` at fixed recovery.
* A 1bp *hazard* scenario reprices to ``hazard01``.
* A 1bp *spread* scenario (R=40%) reprices to ``cs01``.
* Curve-level absolute spread shocks on mixed-recovery books are rejected
  unless a recovery convention is supplied.
* The legacy ``spread_*`` path factories are deprecated.
"""
from datetime import datetime

import pytest

from quantark.asset.credit.engine.analytical import CDSReducedFormEngine
from quantark.asset.credit.product import CDS, ProtectionSide
from quantark.asset.credit.riskmeasures import CreditGreeksCalculator
from quantark.dynamicscenario.credit import CreditPathLibrary
from quantark.param import FlatRateCurve
from quantark.param.credit import FlatHazardCurve
from quantark.portfolio import CreditPortfolio
from quantark.priceenv import CreditPricingEnvironment
from quantark.stresstest import CreditStressEngine
from quantark.stresstest.scenario.scenario import Scenario, Stress
from quantark.stresstest.stress.stress_types import StressLevel, StressType
from quantark.util.exceptions import ValidationError

_VAL_DATE = datetime(2026, 6, 13)


def _env(hazard=0.02, rate=0.03):
    return CreditPricingEnvironment(
        valuation_date=_VAL_DATE,
        discount_curve=FlatRateCurve(rate=rate),
        hazard_curve=FlatHazardCurve(hazard_rate=hazard),
    )


def _cds(recovery=0.4, side=ProtectionSide.BUY):
    return CDS(notional=10_000_000, maturity=5.0, recovery_rate=recovery,
               coupon_spread=0.01, side=side)


def _portfolio(recovery=0.4):
    pf = CreditPortfolio(portfolio_name="cr", pricing_environments={"ACME": _env()})
    pf.add_position(product=_cds(recovery=recovery), quantity=1.0, entry_price=0.0,
                    reference_entity="ACME", engine=CDSReducedFormEngine())
    return pf


def test_cs01_equals_hazard01_over_lgd():
    calc, eng = CreditGreeksCalculator(), CDSReducedFormEngine()
    R = 0.4
    cds, env = _cds(recovery=R), _env()
    hazard01 = calc.hazard01(cds, env, eng)
    cs01 = calc.cs01(cds, env, eng)
    assert cs01 == pytest.approx(hazard01 / (1.0 - R), rel=1e-3)


def test_hazard_scenario_reprices_to_hazard01():
    pf = _portfolio()
    hazard01 = pf.get_portfolio_greeks()["hazard01"]
    scenario = Scenario(
        name="hazard +1bp",
        stresses=[Stress(parameter="hazard", stress_type=StressType.ABSOLUTE,
                         stress_value=1e-4, level=StressLevel.PORTFOLIO)],
    )
    res = CreditStressEngine().run_static_scenarios(pf, [scenario]).scenario_results[0]
    # A +1bp hazard shock reprices to hazard01 (one-sided vs central diff: tiny gamma).
    assert res.portfolio_pnl == pytest.approx(hazard01, rel=0.05)


def test_spread_scenario_reprices_to_cs01():
    pf = _portfolio(recovery=0.4)
    cs01 = pf.get_portfolio_greeks()["cs01"]
    scenario = Scenario(
        name="spread +1bp",
        stresses=[Stress(parameter="spread", stress_type=StressType.ABSOLUTE,
                         stress_value=1e-4, level=StressLevel.PORTFOLIO)],
    )
    res = CreditStressEngine().run_static_scenarios(pf, [scenario]).scenario_results[0]
    assert res.portfolio_pnl == pytest.approx(cs01, rel=0.05)


def test_mixed_recovery_absolute_spread_shock_rejected():
    env_a = _env()
    env_b = _env()
    pf = CreditPortfolio(
        portfolio_name="mixed",
        pricing_environments={"ACME": env_a, "BETA": env_b},
    )
    pf.add_position(product=_cds(recovery=0.4), quantity=1.0, entry_price=0.0,
                    reference_entity="ACME", engine=CDSReducedFormEngine())
    pf.add_position(product=_cds(recovery=0.6), quantity=1.0, entry_price=0.0,
                    reference_entity="BETA", engine=CDSReducedFormEngine())
    scenario = Scenario(
        name="spread +1bp",
        stresses=[Stress(parameter="spread", stress_type=StressType.ABSOLUTE,
                         stress_value=1e-4, level=StressLevel.PORTFOLIO)],
    )
    with pytest.raises(ValidationError):
        CreditStressEngine().run_static_scenarios(pf, [scenario])


def test_mixed_recovery_accepts_explicit_recovery_convention():
    env_a, env_b = _env(), _env()
    pf = CreditPortfolio(
        portfolio_name="mixed",
        pricing_environments={"ACME": env_a, "BETA": env_b},
    )
    pf.add_position(product=_cds(recovery=0.4), quantity=1.0, entry_price=0.0,
                    reference_entity="ACME", engine=CDSReducedFormEngine())
    pf.add_position(product=_cds(recovery=0.6), quantity=1.0, entry_price=0.0,
                    reference_entity="BETA", engine=CDSReducedFormEngine())
    scenario = Scenario(
        name="spread +1bp",
        stresses=[Stress(parameter="spread", stress_type=StressType.ABSOLUTE,
                         stress_value=1e-4, level=StressLevel.PORTFOLIO,
                         metadata={"recovery_rate": 0.4})],
    )
    # An explicit recovery convention resolves the ambiguity; no error.
    res = CreditStressEngine().run_static_scenarios(pf, [scenario]).scenario_results[0]
    assert res.portfolio_pnl != 0.0


def test_percentage_spread_shock_needs_no_recovery():
    # Percentage spread == percentage hazard, so mixed recoveries are fine.
    env_a, env_b = _env(), _env()
    pf = CreditPortfolio(
        portfolio_name="mixed",
        pricing_environments={"ACME": env_a, "BETA": env_b},
    )
    pf.add_position(product=_cds(recovery=0.4), quantity=1.0, entry_price=0.0,
                    reference_entity="ACME", engine=CDSReducedFormEngine())
    pf.add_position(product=_cds(recovery=0.6), quantity=1.0, entry_price=0.0,
                    reference_entity="BETA", engine=CDSReducedFormEngine())
    scenario = Scenario(
        name="spread +50%",
        stresses=[Stress(parameter="spread", stress_type=StressType.PERCENTAGE,
                         stress_value=0.5, level=StressLevel.PORTFOLIO)],
    )
    res = CreditStressEngine().run_static_scenarios(pf, [scenario]).scenario_results[0]
    assert res.portfolio_pnl > 0  # protection buyers gain as hazard widens


def test_legacy_spread_path_factories_are_deprecated():
    with pytest.warns(DeprecationWarning):
        CreditPathLibrary.spread_widening(days=2)
    with pytest.warns(DeprecationWarning):
        CreditPathLibrary.spread_rally(days=2)
