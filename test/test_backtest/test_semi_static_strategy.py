"""
Unit tests for the semi-static (event-driven) hedging strategy.
"""

from datetime import datetime

import pytest

from quantark.backtest.strategy.hedge_optimizer import HedgeTarget
from quantark.backtest.strategy.semi_static_strategy import SemiStaticHedgeStrategy
from quantark.util.exceptions import ValidationError

TRADE_DATE = datetime(2026, 1, 5)
BETWEEN = datetime(2026, 1, 20)
OBS_1 = datetime(2026, 2, 5)
OBS_1_LATER = datetime(2026, 2, 5, 15, 0)
OBS_2 = datetime(2026, 3, 5)

DRIFTED = {"delta": 250.0}
MARKET = {"spot": 100.0}


def make_strategy(**kwargs):
    defaults = dict(rebalance_dates=[OBS_1, OBS_2])
    defaults.update(kwargs)
    strategy = SemiStaticHedgeStrategy(**defaults)
    strategy.on_step(TRADE_DATE, DRIFTED, MARKET)  # trade date seen first
    return strategy


class TestEventGating:
    def test_hedges_on_trade_date(self):
        strategy = make_strategy()
        assert strategy.should_hedge(TRADE_DATE, DRIFTED, MARKET)

    def test_skips_trade_date_when_disabled(self):
        strategy = make_strategy(hedge_at_start=False)
        assert not strategy.should_hedge(TRADE_DATE, DRIFTED, MARKET)

    def test_no_hedge_between_events_despite_drift(self):
        strategy = make_strategy()
        assert not strategy.should_hedge(BETWEEN, DRIFTED, MARKET)

    def test_hedges_on_observation_date_once(self):
        strategy = make_strategy()
        assert strategy.should_hedge(OBS_1, DRIFTED, MARKET)
        strategy.on_hedges_executed(OBS_1, {"spot": -250.0})
        # Same calendar date after executing: no second hedge
        assert not strategy.should_hedge(OBS_1_LATER, DRIFTED, MARKET)
        # Next scheduled date works again
        assert strategy.should_hedge(OBS_2, DRIFTED, MARKET)

    def test_no_hedge_at_event_without_deviation(self):
        strategy = make_strategy()
        assert not strategy.should_hedge(OBS_1, {"delta": 0.0}, MARKET)


class TestBarrierProximity:
    def test_near_barrier_forces_rebalance(self):
        strategy = make_strategy(barrier_level=80.0, barrier_proximity_band=0.05)
        # Spot 100 vs barrier 80: 25% away -> not an event
        assert not strategy.should_hedge(BETWEEN, DRIFTED, {"spot": 100.0})
        # Spot 82 vs barrier 80: 2.5% away -> event, every step
        assert strategy.should_hedge(BETWEEN, DRIFTED, {"spot": 82.0})
        strategy.on_hedges_executed(BETWEEN, {"spot": -250.0})
        later_same_day = datetime(2026, 1, 20, 15, 0)
        assert strategy.should_hedge(later_same_day, DRIFTED, {"spot": 81.0})

    def test_is_near_barrier_without_barrier(self):
        strategy = make_strategy()
        assert not strategy.is_near_barrier({"spot": 80.0})


class TestSizing:
    def test_default_is_pure_delta_solve(self):
        strategy = make_strategy()
        quantities = strategy.calculate_hedge_quantities(
            OBS_1, {"delta": 250.0}, MARKET, {"spot": {"delta": 1.0}}
        )
        assert quantities["spot"] == pytest.approx(-250.0)

    def test_custom_targets_compose(self):
        strategy = SemiStaticHedgeStrategy(
            targets=[HedgeTarget("delta"), HedgeTarget("gamma")],
            hedge_instruments=None,  # spot-only is now overdetermined
            rebalance_dates=[OBS_1],
        )
        assert [t.greek for t in strategy.targets] == ["delta", "gamma"]


class TestValidationAndReset:
    def test_invalid_parameters(self):
        with pytest.raises(ValidationError):
            SemiStaticHedgeStrategy(barrier_level=0.0)
        with pytest.raises(ValidationError):
            SemiStaticHedgeStrategy(barrier_proximity_band=0.0)

    def test_reset_clears_first_step(self):
        strategy = make_strategy()
        strategy.reset()
        # After reset, a new first step can hedge again
        strategy.on_step(OBS_2, DRIFTED, MARKET)
        assert strategy.should_hedge(OBS_2, DRIFTED, MARKET)
