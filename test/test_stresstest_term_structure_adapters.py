from datetime import datetime

import pytest

from param import FlatRateCurve, SpotQuote, TermStructureVolSurface
from param.div import TermStructureDividendYield
from portfolio import Portfolio
from priceenv import PricingEnvironment
from stresstest.scenario.scenario_builder import ScenarioBuilder
from stresstest.stress.stress_applicator import StressApplicator
from stresstest.stress.stress_types import StressType


def _build_env() -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=TermStructureVolSurface(times=[0.5, 1.0], vols=[0.2, 0.3]),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=TermStructureDividendYield(times=[0.5, 1.0], yields=[0.01, 0.02]),
        valuation_date=datetime(2024, 1, 1),
    )


def test_term_structure_vol_stress_absolute():
    env = _build_env()
    portfolio = Portfolio(
        portfolio_name="test",
        pricing_environments={"UNDERLYING": env},
    )
    scenario = (
        ScenarioBuilder()
        .name("Vol Up")
        .vol_stress(0.05, stress_type=StressType.ABSOLUTE)
        .build()
    )

    stressed_envs = StressApplicator.apply_scenario_to_portfolio(portfolio, scenario)
    stressed_env = stressed_envs["UNDERLYING"]

    assert isinstance(stressed_env.vol_surface, TermStructureVolSurface)
    assert stressed_env.vol_surface.vols == pytest.approx([0.25, 0.35])


def test_term_structure_dividend_stress_percentage():
    env = _build_env()
    portfolio = Portfolio(
        portfolio_name="test",
        pricing_environments={"UNDERLYING": env},
    )
    scenario = ScenarioBuilder().name("Div Down").div_yield_stress(-0.5).build()

    stressed_envs = StressApplicator.apply_scenario_to_portfolio(portfolio, scenario)
    stressed_env = stressed_envs["UNDERLYING"]

    assert isinstance(stressed_env.div_yield, TermStructureDividendYield)
    assert stressed_env.div_yield.yields == pytest.approx([0.005, 0.01])
