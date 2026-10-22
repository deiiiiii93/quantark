"""
Unit tests for backtest module.
"""

import pytest
from datetime import datetime, timedelta
from quantark.backtest import (
    BacktestConfig,
    ZeroCostModel,
    CompleteCostModel,
    BacktestState,
    TradeRecord,
    StateTracker,
)
from quantark.backtest.strategy import DeltaNeutralStrategy
from quantark.portfolio import Position
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.util.enum import OptionType
from quantark.util.marketdata.adapter.mock_adapter import MockMarketDataAdapter


class TestDeltaNeutralStrategy:
    """Tests for DeltaNeutralStrategy."""

    def test_strategy_creation(self):
        """Test strategy can be created with valid parameters."""
        strategy = DeltaNeutralStrategy(
            name="Test",
            delta_threshold=100.0,
            rebalance_frequency="daily",
            hedge_instrument="spot",
        )

        assert strategy.name == "Test"
        assert strategy.delta_threshold == 100.0
        assert strategy.rebalance_frequency == "daily"
        assert strategy.hedge_instrument == "spot"

    def test_invalid_parameters(self):
        """Test strategy raises errors for invalid parameters."""
        with pytest.raises(Exception):
            DeltaNeutralStrategy(delta_threshold=-10.0)  # Should be non-negative

        with pytest.raises(Exception):
            DeltaNeutralStrategy(
                rebalance_frequency="invalid"  # Should be valid frequency
            )

        with pytest.raises(Exception):
            DeltaNeutralStrategy(hedge_ratio=1.5)  # Should be between 0 and 1

    def test_should_hedge_threshold(self):
        """Test hedge triggering based on threshold."""
        strategy = DeltaNeutralStrategy(
            delta_threshold=100.0, rebalance_frequency="on_threshold"
        )

        # Delta below threshold - no hedge
        greeks = {"delta": 50.0}
        assert not strategy.should_hedge(datetime.now(), greeks, {})

        # Delta exceeds threshold - hedge
        greeks = {"delta": 150.0}
        assert strategy.should_hedge(datetime.now(), greeks, {})

    def test_calculate_hedge_size(self):
        """Test hedge size calculation."""
        strategy = DeltaNeutralStrategy(
            delta_threshold=100.0, hedge_ratio=1.0, target_delta=0.0
        )

        greeks = {"delta": 150.0}
        hedge_size = strategy.calculate_hedge_size(datetime.now(), greeks, {})

        # Should hedge -150 to bring delta to 0
        assert hedge_size == pytest.approx(-150.0)

    def test_partial_hedging(self):
        """Test partial hedging with hedge_ratio < 1."""
        strategy = DeltaNeutralStrategy(
            delta_threshold=100.0, hedge_ratio=0.5, target_delta=0.0  # Only hedge 50%
        )

        greeks = {"delta": 200.0}
        hedge_size = strategy.calculate_hedge_size(datetime.now(), greeks, {})

        # Should hedge -100 (50% of 200)
        assert hedge_size == pytest.approx(-100.0)


class TestTransactionCosts:
    """Tests for transaction cost models."""

    def test_zero_cost(self):
        """Test zero cost model."""
        model = ZeroCostModel()
        cost = model.calculate_cost(
            quantity=100,
            price=50.0,
            notional=5000.0,
            instrument_type="spot",
            trade_type="hedge",
        )
        assert cost == 0.0

    def test_complete_cost_model(self):
        """Test complete cost model."""
        model = CompleteCostModel(
            fixed_commission=5.0,
            proportional_rate=0.001,  # 10 bps
            slippage_coefficient=0.0001,
            spread_bps=5.0,
        )

        cost = model.calculate_cost(
            quantity=100,
            price=50.0,
            notional=5000.0,
            instrument_type="spot",
            trade_type="hedge",
        )

        # Should have multiple components
        assert cost > 5.0  # At least the fixed commission

        # Test breakdown
        breakdown = model.get_cost_breakdown(
            quantity=100,
            price=50.0,
            notional=5000.0,
            instrument_type="spot",
            trade_type="hedge",
        )

        assert "fixed" in breakdown
        assert "proportional" in breakdown
        assert "slippage" in breakdown
        assert "spread" in breakdown
        assert breakdown["total"] == cost


class TestBacktestState:
    """Tests for BacktestState and StateTracker."""

    def test_state_creation(self):
        """Test state can be created."""
        state = BacktestState(
            timestamp=datetime.now(),
            portfolio_value=10000.0,
            pnl=500.0,
            greeks={"delta": 100.0},
            market_data={"spot": 100.0},
        )

        assert state.portfolio_value == 10000.0
        assert state.pnl == 500.0
        assert state.greeks["delta"] == 100.0

    def test_state_to_dict(self):
        """Test state conversion to dictionary."""
        state = BacktestState(
            timestamp=datetime.now(),
            portfolio_value=10000.0,
            pnl=500.0,
            greeks={"delta": 100.0, "gamma": 5.0},
            market_data={"spot": 100.0, "vol": 0.25},
        )

        data = state.to_dict()

        assert data["portfolio_value"] == 10000.0
        assert data["pnl"] == 500.0
        assert "greek_delta" in data
        assert "greek_gamma" in data
        assert "market_spot" in data
        assert "market_vol" in data

    def test_state_tracker(self):
        """Test StateTracker functionality."""
        tracker = StateTracker()

        assert len(tracker) == 0

        # Add states
        for i in range(5):
            state = BacktestState(
                timestamp=datetime.now() + timedelta(days=i),
                portfolio_value=10000.0 + i * 100,
                pnl=i * 50.0,
            )
            tracker.add_state(state)

        assert len(tracker) == 5

        # Test current state
        current = tracker.get_current_state()
        assert current.portfolio_value == 10400.0

        # Test to_dataframe
        df = tracker.to_dataframe()
        assert len(df) == 5
        assert "portfolio_value" in df.columns


class TestTradeRecord:
    """Tests for TradeRecord."""

    def test_trade_record_creation(self):
        """Test trade record creation."""
        trade = TradeRecord(
            timestamp=datetime.now(),
            trade_type="open",
            instrument_type="spot",
            underlying="AAPL",
            quantity=100,
            price=150.0,
            notional=15000.0,
            transaction_cost=10.0,
            reason="hedge",
        )

        assert trade.quantity == 100
        assert trade.price == 150.0
        assert trade.transaction_cost == 10.0

    def test_trade_record_to_dict(self):
        """Test trade record conversion to dictionary."""
        trade = TradeRecord(
            timestamp=datetime.now(),
            trade_type="open",
            instrument_type="spot",
            underlying="AAPL",
            quantity=100,
            price=150.0,
            notional=15000.0,
            transaction_cost=10.0,
            reason="hedge",
        )

        data = trade.to_dict()

        assert data["quantity"] == 100
        assert data["price"] == 150.0
        assert data["underlying"] == "AAPL"


class TestBacktestConfig:
    """Tests for BacktestConfig."""

    def test_config_creation(self):
        """Test backtest configuration creation."""
        strategy = DeltaNeutralStrategy()
        adapter = MockMarketDataAdapter()
        cost_model = ZeroCostModel()

        config = BacktestConfig(
            strategy=strategy,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            underlying="AAPL",
            initial_positions=[],
            market_data_adapter=adapter,
            transaction_cost_model=cost_model,
        )

        assert config.underlying == "AAPL"
        assert config.frequency == "D"
        assert config.calculate_greeks == True

    def test_invalid_config(self):
        """Test configuration validation."""
        strategy = DeltaNeutralStrategy()
        adapter = MockMarketDataAdapter()
        cost_model = ZeroCostModel()

        # End date before start date
        with pytest.raises(Exception):
            BacktestConfig(
                strategy=strategy,
                start_date=datetime(2024, 12, 31),
                end_date=datetime(2024, 1, 1),  # Before start
                underlying="AAPL",
                initial_positions=[],
                market_data_adapter=adapter,
                transaction_cost_model=cost_model,
            )

    def test_config_summary(self):
        """Test configuration summary."""
        strategy = DeltaNeutralStrategy()
        adapter = MockMarketDataAdapter()
        cost_model = ZeroCostModel()

        config = BacktestConfig(
            strategy=strategy,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            underlying="AAPL",
            initial_positions=[],
            market_data_adapter=adapter,
            transaction_cost_model=cost_model,
        )

        summary = config.get_summary()

        assert "strategy" in summary
        assert "underlying" in summary
        assert summary["underlying"] == "AAPL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
