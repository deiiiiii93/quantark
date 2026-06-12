"""
Unit tests for the minimum-variance delta hedging strategy.
"""

from datetime import datetime

import pytest

from quantark.backtest.strategy.min_variance_delta_strategy import (
    MinimumVarianceDeltaStrategy,
)
from quantark.util.exceptions import ValidationError

T0 = datetime(2026, 1, 5)


class TestMinVarianceDelta:
    def test_constant_slope_adjustment(self):
        # Portfolio vega is per 1% vol; slope is per unit vol:
        # delta_MV = 200 + 50 * 100 * (-0.002) = 190
        strategy = MinimumVarianceDeltaStrategy(vol_spot_slope=-0.002)
        mv_delta = strategy.get_min_variance_delta(
            {"delta": 200.0, "vega": 50.0}, {}
        )
        assert mv_delta == pytest.approx(190.0)

    def test_zero_slope_recovers_bs_delta(self):
        strategy = MinimumVarianceDeltaStrategy(vol_spot_slope=0.0)
        mv_delta = strategy.get_min_variance_delta(
            {"delta": 200.0, "vega": 50.0}, {}
        )
        assert mv_delta == pytest.approx(200.0)

    def test_callable_slope_receives_market_data(self):
        def skew_slope(market_data):
            return -0.0001 * market_data["spot"]

        strategy = MinimumVarianceDeltaStrategy(vol_spot_slope=skew_slope)
        mv_delta = strategy.get_min_variance_delta(
            {"delta": 100.0, "vega": 10.0}, {"spot": 50.0}
        )
        # 100 + 10 * 100 * (-0.005) = 95
        assert mv_delta == pytest.approx(95.0)

    def test_should_hedge_uses_mv_delta(self):
        strategy = MinimumVarianceDeltaStrategy(
            vol_spot_slope=-0.002,
            delta_threshold=100.0,
            rebalance_frequency="continuous",
        )
        # Raw delta 105 breaches, but MV delta 105 - 100*100*0.002 = 85 does not
        greeks = {"delta": 105.0, "vega": 100.0}
        assert not strategy.should_hedge(T0, greeks, {})
        # Raw delta 90 is inside, but MV delta 90 + 100*100*0.002 = 110 breaches
        greeks = {"delta": 90.0, "vega": -100.0}
        assert strategy.should_hedge(T0, greeks, {})

    def test_hedge_size_targets_mv_delta(self):
        strategy = MinimumVarianceDeltaStrategy(vol_spot_slope=-0.002)
        hedge = strategy.calculate_hedge_size(
            T0, {"delta": 200.0, "vega": 50.0}, {}
        )
        assert hedge == pytest.approx(-190.0)

    def test_daily_frequency_gate(self):
        strategy = MinimumVarianceDeltaStrategy(
            vol_spot_slope=0.0, delta_threshold=10.0, rebalance_frequency="daily"
        )
        greeks = {"delta": 50.0, "vega": 0.0}
        assert strategy.should_hedge(T0, greeks, {})
        strategy.on_hedge_executed(T0, hedge_size=-50.0, hedge_price=100.0)
        same_day = datetime(2026, 1, 5, 15, 0)
        assert not strategy.should_hedge(same_day, greeks, {})

    def test_validation(self):
        with pytest.raises(ValidationError):
            MinimumVarianceDeltaStrategy(vol_spot_slope="steep")
        with pytest.raises(ValidationError):
            MinimumVarianceDeltaStrategy(vol_spot_slope=0.0, delta_threshold=-1.0)
        with pytest.raises(ValidationError):
            MinimumVarianceDeltaStrategy(vol_spot_slope=0.0, hedge_instrument="swap")
        with pytest.raises(ValidationError):
            MinimumVarianceDeltaStrategy(
                vol_spot_slope=0.0, rebalance_frequency="weekly"
            )
