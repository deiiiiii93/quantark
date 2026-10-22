"""
Unit tests for the Whalley-Wilmott utility-based hedging band strategy.
"""

import math
from datetime import datetime

import pytest

from quantark.backtest.strategy.whalley_wilmott_strategy import WhalleyWilmottStrategy
from quantark.util.exceptions import ValidationError

T0 = datetime(2026, 1, 5)


class TestBandWidth:
    def test_band_formula(self):
        # H = (1.5 * k * S * Gamma^2 / lambda)^(1/3)
        #   = (1.5 * 0.001 * 100 * 25 / 0.5)^(1/3) = 7.5^(1/3)
        strategy = WhalleyWilmottStrategy(risk_aversion=0.5, cost_rate=0.001)
        band = strategy.get_band_half_width(
            {"gamma": 5.0}, {"spot": 100.0}
        )
        assert band == pytest.approx(7.5 ** (1.0 / 3.0))

    def test_band_with_horizon_discount(self):
        strategy = WhalleyWilmottStrategy(
            risk_aversion=0.5, cost_rate=0.001, horizon=1.0
        )
        band = strategy.get_band_half_width(
            {"gamma": 5.0}, {"spot": 100.0, "rate": 0.05}
        )
        expected = (7.5 * math.exp(-0.05)) ** (1.0 / 3.0)
        assert band == pytest.approx(expected)

    def test_zero_gamma_collapses_band(self):
        strategy = WhalleyWilmottStrategy()
        assert strategy.get_band_half_width({"gamma": 0.0}, {"spot": 100.0}) == 0.0


class TestHedgeDecisions:
    def test_no_hedge_inside_band(self):
        strategy = WhalleyWilmottStrategy(risk_aversion=0.5, cost_rate=0.001)
        # Band ~1.957; delta deviation 1.5 is inside
        assert not strategy.should_hedge(
            T0, {"delta": 1.5, "gamma": 5.0}, {"spot": 100.0}
        )

    def test_hedge_outside_band(self):
        strategy = WhalleyWilmottStrategy(risk_aversion=0.5, cost_rate=0.001)
        assert strategy.should_hedge(
            T0, {"delta": 10.0, "gamma": 5.0}, {"spot": 100.0}
        )

    def test_rebalance_to_boundary(self):
        strategy = WhalleyWilmottStrategy(risk_aversion=0.5, cost_rate=0.001)
        band = strategy.get_band_half_width({"gamma": 5.0}, {"spot": 100.0})
        hedge = strategy.calculate_hedge_size(
            T0, {"delta": 10.0, "gamma": 5.0}, {"spot": 100.0}
        )
        # Trades back to the +band edge, not to zero
        assert hedge == pytest.approx(-(10.0 - band))
        # Symmetric on the short side
        hedge_short = strategy.calculate_hedge_size(
            T0, {"delta": -10.0, "gamma": 5.0}, {"spot": 100.0}
        )
        assert hedge_short == pytest.approx(10.0 - band)

    def test_rebalance_to_target(self):
        strategy = WhalleyWilmottStrategy(
            risk_aversion=0.5, cost_rate=0.001, rebalance_to="target"
        )
        hedge = strategy.calculate_hedge_size(
            T0, {"delta": 10.0, "gamma": 5.0}, {"spot": 100.0}
        )
        assert hedge == pytest.approx(-10.0)

    def test_inside_band_zero_hedge(self):
        strategy = WhalleyWilmottStrategy(risk_aversion=0.5, cost_rate=0.001)
        hedge = strategy.calculate_hedge_size(
            T0, {"delta": 1.0, "gamma": 5.0}, {"spot": 100.0}
        )
        assert hedge == 0.0


class TestValidation:
    def test_invalid_parameters(self):
        with pytest.raises(ValidationError):
            WhalleyWilmottStrategy(risk_aversion=0.0)
        with pytest.raises(ValidationError):
            WhalleyWilmottStrategy(cost_rate=-0.001)
        with pytest.raises(ValidationError):
            WhalleyWilmottStrategy(horizon=-1.0)
        with pytest.raises(ValidationError):
            WhalleyWilmottStrategy(rebalance_to="midpoint")
        with pytest.raises(ValidationError):
            WhalleyWilmottStrategy(hedge_instrument="swap")
