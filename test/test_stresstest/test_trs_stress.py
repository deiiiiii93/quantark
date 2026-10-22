"""Gate 3 tests: equity TRS in the stress-test engine.

The EquityStressEngine and StressApplicator are position-agnostic (they bump
pricing environments and value through the BasePosition interface), so a swap
position flows through with no engine changes. These tests confirm that.
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantark.util.calendar.business_calendar import Calendar
from quantark.asset.equity.product.swap import (
    AssetParams,
    FixLegParams,
    FloatLegParams,
    PricingParams,
    TRSParams,
    OneAssetTotalReturnSwap,
)
from quantark.portfolio import EquityPortfolio
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.stresstest.equity.engine import EquityStressEngine
from quantark.stresstest.equity.config import EquityStressConfig
from quantark.stresstest.scenario.scenario import Scenario, Stress
from quantark.stresstest.stress.stress_types import StressType, StressLevel
from quantark.dynamicscenario import (
    DynamicScenarioEngine,
    DynamicScenarioConfig,
    PathLibrary,
)

START, END, VALUATION = "2024-01-02", "2024-01-31", "2024-01-30"
INITIAL, VAL_SPOT, NOTIONAL = 100.0, 110.0, 1_000_000.0


def _make_trs() -> OneAssetTotalReturnSwap:
    days = pd.date_range(START, END, freq="D")
    idx = [d.strftime("%Y-%m-%d") for d in days]
    val_i = idx.index(VALUATION)
    path = pd.Series(
        [INITIAL + (VAL_SPOT - INITIAL) * min(i, val_i) / val_i for i in range(len(idx))],
        index=idx,
    ).round(4)
    calendar = Calendar(name="DemoCalendar")
    return OneAssetTotalReturnSwap(
        TRSParams(
            contract_id="TRS_STRESS",
            asset=AssetParams("IDX", INITIAL, path),
            fix_leg=FixLegParams(
                rate=0.048, notional=NOTIONAL, initial_notional=NOTIONAL,
                start_date=START, end_date=END, payment_calendar=calendar, direction=-1,
            ),
            float_leg=FloatLegParams(
                notional=NOTIONAL, initial_notional=NOTIONAL,
                start_date=START, end_date=END, payment_calendar=calendar, direction=1,
            ),
            pricing=PricingParams(valuation_date=VALUATION, output_mode="spot"),
        )
    )


@pytest.fixture
def swap_portfolio() -> EquityPortfolio:
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=VAL_SPOT, asset_name="IDX"),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.048),
        valuation_date=datetime(2024, 1, 30),
    )
    pf = EquityPortfolio(portfolio_name="SwapBook", pricing_environments={"IDX": env})
    pf.add_swap_position(product=_make_trs(), quantity=1.0)
    return pf


def _spot_scenario(name: str, shock_pct: float) -> Scenario:
    return Scenario(
        name=name,
        stresses=[
            Stress("spot", StressType.PERCENTAGE, shock_pct, StressLevel.PORTFOLIO)
        ],
    )


def test_spot_down_scenario_loses_value(swap_portfolio):
    """A long TRS loses ~delta-one value when spot drops."""
    engine = EquityStressEngine(EquityStressConfig(calculate_greeks=False))
    results = engine.run_static_scenarios(
        swap_portfolio, [_spot_scenario("Spot -10%", -0.10)]
    )
    baseline = results.baseline_value
    scenario_res = results.scenario_results[0]
    assert scenario_res.portfolio_pnl < 0
    # Delta-one: a 10% spot drop costs ~ qty_shares * 0.10 * spot.
    expected = -(NOTIONAL / INITIAL) * 0.10 * VAL_SPOT
    assert scenario_res.portfolio_pnl == pytest.approx(expected, rel=0.05)
    assert baseline == pytest.approx(swap_portfolio.get_portfolio_value())


def test_spot_up_and_down_symmetry(swap_portfolio):
    engine = EquityStressEngine(EquityStressConfig(calculate_greeks=False))
    results = engine.run_static_scenarios(
        swap_portfolio,
        [_spot_scenario("Up", 0.05), _spot_scenario("Down", -0.05)],
    )
    up, down = results.scenario_results
    assert up.portfolio_pnl > 0 > down.portfolio_pnl
    # Linear (delta-one) payoff: gains and losses are near-symmetric.
    assert up.portfolio_pnl == pytest.approx(-down.portfolio_pnl, rel=1e-6)


def test_stress_with_greeks(swap_portfolio):
    """Greeks aggregation runs for a swap book (delta-one, vega 0)."""
    engine = EquityStressEngine(
        EquityStressConfig(calculate_greeks=True, greeks_method="numerical")
    )
    results = engine.run_static_scenarios(
        swap_portfolio, [_spot_scenario("Spot -5%", -0.05)]
    )
    greeks = results.baseline_greeks
    assert greeks is not None
    assert greeks["delta"] == pytest.approx(NOTIONAL / INITIAL, rel=1e-3)
    assert greeks["vega"] == 0.0


# --------------------------------------------------------------------------- #
# Dynamic scenario (multi-day path)
# --------------------------------------------------------------------------- #
def test_dynamic_rally_gains(swap_portfolio):
    """A long TRS gains over a multi-day rally; PnL tracks the delta-one spot move.

    The TRS lifecycle is untracked by the dynamic-scenario LifecycleManager (it
    tracks only autocallable/barrier options), so the swap simply marks to market
    against each day's spot via the BasePosition interface.
    """
    config = DynamicScenarioConfig(calculate_greeks=False)
    path = PathLibrary.consecutive_rally(days=5, daily_pct=0.02, vol_change_pct=0.0)
    results = DynamicScenarioEngine(config).run(swap_portfolio, path)

    assert results.total_pnl > 0
    final_spot = VAL_SPOT * (1.02 ** 5)
    expected = (NOTIONAL / INITIAL) * (final_spot - VAL_SPOT)
    assert results.total_pnl == pytest.approx(expected, rel=0.05)
    assert len(results.day_results) == 5


def test_dynamic_decline_loses(swap_portfolio):
    config = DynamicScenarioConfig(calculate_greeks=False)
    path = PathLibrary.consecutive_decline(days=4, daily_pct=-0.03, vol_change_pct=0.0)
    results = DynamicScenarioEngine(config).run(swap_portfolio, path)
    assert results.total_pnl < 0
