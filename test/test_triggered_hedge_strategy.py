"""
Unit tests for the trigger-armed (contingent) hedging strategy.
"""

from datetime import datetime

import pytest

from quantark.backtest.strategy.hedge_optimizer import HedgeTarget
from quantark.backtest.strategy.triggered_hedge_strategy import (
    HedgeTrigger,
    TriggeredHedgeStrategy,
)
from quantark.util.exceptions import ValidationError

T0 = datetime(2026, 1, 5)
T1 = datetime(2026, 1, 6)
T2 = datetime(2026, 1, 7)

START_MARKET = {"spot": 100.0, "volatility": 0.20}


class TestHedgeTrigger:
    def test_spot_drawdown(self):
        trigger = HedgeTrigger("crash", spot_drawdown=0.10)
        reference = {"spot": 100.0}
        assert not trigger.is_met(reference, {"spot": 91.0})
        assert trigger.is_met(reference, {"spot": 90.0})
        assert trigger.is_met(reference, {"spot": 85.0})

    def test_spot_rally(self):
        trigger = HedgeTrigger("melt_up", spot_rally=0.15)
        reference = {"spot": 100.0}
        assert not trigger.is_met(reference, {"spot": 114.0})
        assert trigger.is_met(reference, {"spot": 115.0})

    def test_vol_increase(self):
        trigger = HedgeTrigger("vol_spike", vol_increase=0.08)
        reference = {"volatility": 0.20}
        assert not trigger.is_met(reference, {"volatility": 0.27})
        assert trigger.is_met(reference, {"volatility": 0.28})

    def test_joint_conditions_require_all(self):
        trigger = HedgeTrigger("stress", spot_drawdown=0.10, vol_increase=0.05)
        reference = {"spot": 100.0, "volatility": 0.20}
        # Only spot condition met
        assert not trigger.is_met(reference, {"spot": 88.0, "volatility": 0.22})
        # Both met
        assert trigger.is_met(reference, {"spot": 88.0, "volatility": 0.26})

    def test_missing_market_data_raises(self):
        trigger = HedgeTrigger("vol_spike", vol_increase=0.05)
        with pytest.raises(ValidationError):
            trigger.is_met({"volatility": 0.20}, {"spot": 90.0})

    def test_validation(self):
        with pytest.raises(ValidationError):
            HedgeTrigger("empty")  # no conditions
        with pytest.raises(ValidationError):
            HedgeTrigger("", spot_drawdown=0.10)
        with pytest.raises(ValidationError):
            HedgeTrigger("bad", spot_drawdown=-0.10)
        with pytest.raises(ValidationError):
            HedgeTrigger("bad", spot_drawdown=1.5)


def make_strategy(**kwargs):
    defaults = dict(
        triggers=[HedgeTrigger("crash", spot_drawdown=0.10)],
        targets=[HedgeTarget("delta", threshold=50.0)],
    )
    defaults.update(kwargs)
    strategy = TriggeredHedgeStrategy(**defaults)
    # First step captures the reference market
    strategy.on_step(T0, {}, START_MARKET)
    return strategy


class TestArming:
    def test_inactive_before_trigger(self):
        strategy = make_strategy()
        drifted = {"delta": 500.0}
        # Spot down only 5%: trigger not fired, no hedging despite drift
        strategy.on_step(T1, drifted, {"spot": 95.0, "volatility": 0.2})
        assert not strategy.armed
        assert not strategy.should_hedge(T1, drifted, {"spot": 95.0})

    def test_arms_on_trigger_then_hedges_on_threshold(self):
        strategy = make_strategy()
        drifted = {"delta": 500.0}
        strategy.on_step(T1, drifted, {"spot": 89.0, "volatility": 0.2})
        assert strategy.armed
        assert strategy.should_hedge(T1, drifted, {"spot": 89.0})
        # Armed but delta inside threshold: no hedge
        assert not strategy.should_hedge(T1, {"delta": 30.0}, {"spot": 89.0})

    def test_latch_keeps_armed_after_recovery(self):
        strategy = make_strategy()  # latch=True default
        strategy.on_step(T1, {}, {"spot": 89.0, "volatility": 0.2})
        strategy.on_step(T2, {}, {"spot": 100.0, "volatility": 0.2})
        assert strategy.armed
        assert strategy.should_hedge(T2, {"delta": 500.0}, {"spot": 100.0})

    def test_unlatched_disarms_on_recovery(self):
        strategy = make_strategy(latch=False)
        strategy.on_step(T1, {}, {"spot": 89.0, "volatility": 0.2})
        assert strategy.armed
        strategy.on_step(T2, {}, {"spot": 100.0, "volatility": 0.2})
        assert not strategy.armed

    def test_reference_is_first_step(self):
        strategy = make_strategy()
        # Reference spot is 100 from T0; a later step at 95 then 86 fires
        # relative to 100, not relative to 95
        strategy.on_step(T1, {}, {"spot": 95.0, "volatility": 0.2})
        assert not strategy.armed
        strategy.on_step(T2, {}, {"spot": 86.0, "volatility": 0.2})
        assert strategy.armed

    def test_statistics_record_fired_trigger(self):
        strategy = make_strategy()
        strategy.on_step(T1, {}, {"spot": 89.0, "volatility": 0.2})
        stats = strategy.get_statistics()
        assert stats["armed"] is True
        assert stats["fired_triggers"] == {"crash": T1}
        assert stats["reference_market"]["spot"] == pytest.approx(100.0)

    def test_reset_clears_arming_and_reference(self):
        strategy = make_strategy()
        strategy.on_step(T1, {}, {"spot": 89.0, "volatility": 0.2})
        strategy.reset()
        assert not strategy.armed
        # New reference can be captured after reset
        strategy.on_step(T2, {}, {"spot": 89.0, "volatility": 0.2})
        assert not strategy.armed  # 89 is the new reference, no drawdown yet


class TestSizing:
    def test_default_delta_solve(self):
        strategy = make_strategy()
        strategy.on_step(T1, {}, {"spot": 89.0, "volatility": 0.2})
        quantities = strategy.calculate_hedge_quantities(
            T1, {"delta": 500.0}, {"spot": 89.0}, {"spot": {"delta": 1.0}}
        )
        assert quantities["spot"] == pytest.approx(-500.0)


class TestStrategyValidation:
    def test_requires_triggers(self):
        with pytest.raises(ValidationError):
            TriggeredHedgeStrategy(triggers=[])

    def test_duplicate_trigger_names_raise(self):
        with pytest.raises(ValidationError):
            TriggeredHedgeStrategy(
                triggers=[
                    HedgeTrigger("x", spot_drawdown=0.1),
                    HedgeTrigger("x", spot_rally=0.1),
                ]
            )
