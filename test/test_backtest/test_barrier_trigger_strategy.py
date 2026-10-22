"""
Unit tests for the barrier-trigger (zoned) hedging strategy.
"""

from datetime import datetime

import pytest

from quantark.backtest.strategy.barrier_trigger_strategy import (
    ZONE_FAR,
    ZONE_KNOCKED_IN,
    ZONE_NEAR,
    BarrierTriggerHedgeStrategy,
)
from quantark.util.exceptions import ValidationError

T0 = datetime(2026, 1, 5)
T1 = datetime(2026, 1, 6)


def make_strategy(**kwargs):
    defaults = dict(
        barrier_level=80.0,
        far_delta_threshold=100.0,
        near_delta_threshold=20.0,
        post_ki_delta_threshold=50.0,
    )
    defaults.update(kwargs)
    return BarrierTriggerHedgeStrategy(**defaults)


class TestZones:
    def test_far_zone(self):
        strategy = make_strategy()
        assert strategy.get_zone({"spot": 100.0}) == ZONE_FAR

    def test_near_zone(self):
        strategy = make_strategy(proximity_band=0.10)
        # 85/80 - 1 = 6.25% < 10%
        assert strategy.get_zone({"spot": 85.0}) == ZONE_NEAR

    def test_knock_in_latches(self):
        strategy = make_strategy()
        strategy.on_step(T0, {}, {"spot": 79.0})  # crossed (down)
        assert strategy.knocked_in
        # Spot recovers far above the barrier: regime stays knocked-in
        assert strategy.get_zone({"spot": 120.0}) == ZONE_KNOCKED_IN

    def test_up_barrier_direction(self):
        strategy = make_strategy(barrier_level=120.0, barrier_direction="up")
        strategy.on_step(T0, {}, {"spot": 100.0})
        assert not strategy.knocked_in
        strategy.on_step(T1, {}, {"spot": 121.0})
        assert strategy.knocked_in


class TestZonedThresholds:
    def test_light_hedging_far_from_barrier(self):
        strategy = make_strategy()
        market = {"spot": 100.0}
        assert not strategy.should_hedge(T0, {"delta": 60.0}, market)
        assert strategy.should_hedge(T0, {"delta": 150.0}, market)

    def test_aggressive_hedging_near_barrier(self):
        strategy = make_strategy()
        market = {"spot": 84.0}  # 5% from barrier, inside default 10% band
        # The same 60 delta that was fine far away now triggers
        assert strategy.should_hedge(T0, {"delta": 60.0}, market)
        assert not strategy.should_hedge(T0, {"delta": 15.0}, market)

    def test_post_ki_regime(self):
        strategy = make_strategy()
        strategy.on_step(T0, {}, {"spot": 78.0})  # knock-in
        market = {"spot": 100.0}  # recovered, but regime is post-KI
        assert strategy.should_hedge(T1, {"delta": 60.0}, market)
        assert not strategy.should_hedge(T1, {"delta": 40.0}, market)

    def test_hedge_size_full_to_target(self):
        strategy = make_strategy(target_delta=10.0)
        size = strategy.calculate_hedge_size(T0, {"delta": 150.0}, {"spot": 100.0})
        assert size == pytest.approx(-140.0)


class TestStatisticsAndReset:
    def test_zone_statistics(self):
        strategy = make_strategy()
        strategy.calculate_hedge_size(T0, {"delta": 150.0}, {"spot": 100.0})
        strategy.on_hedge_executed(T0, hedge_size=-150.0, hedge_price=100.0)
        strategy.calculate_hedge_size(T1, {"delta": 60.0}, {"spot": 84.0})
        strategy.on_hedge_executed(T1, hedge_size=-60.0, hedge_price=84.0)

        stats = strategy.get_statistics()
        assert stats["hedge_count_by_zone"][ZONE_FAR] == 1
        assert stats["hedge_count_by_zone"][ZONE_NEAR] == 1

    def test_reset_clears_knock_in(self):
        strategy = make_strategy()
        strategy.on_step(T0, {}, {"spot": 70.0})
        assert strategy.knocked_in
        strategy.reset()
        assert not strategy.knocked_in
        assert strategy.get_zone({"spot": 100.0}) == ZONE_FAR


class TestValidation:
    def test_invalid_parameters(self):
        with pytest.raises(ValidationError):
            BarrierTriggerHedgeStrategy(barrier_level=0.0)
        with pytest.raises(ValidationError):
            make_strategy(barrier_direction="sideways")
        with pytest.raises(ValidationError):
            make_strategy(proximity_band=0.0)
        with pytest.raises(ValidationError):
            make_strategy(near_delta_threshold=-1.0)
        with pytest.raises(ValidationError):
            make_strategy(hedge_instrument="swap")
