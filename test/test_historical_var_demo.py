"""
Tests for historical_var_demo.py

Verifies that the historical VaR demonstration script executes correctly.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from example.historical_var_demo import (
    create_options_portfolio,
    generate_historical_data,
    main as historical_var_main,
)


class TestHistoricalVarDemo:
    """Test cases for historical VaR demonstration."""

    def test_create_options_portfolio(self):
        """Test options portfolio creation."""
        portfolio = create_options_portfolio()

        assert portfolio is not None
        assert portfolio.portfolio_name == "Historical VaR Demo - Options Portfolio"
        assert len(portfolio.positions) == 4
        assert portfolio.get_portfolio_value() > 0

    def test_generate_historical_data(self):
        """Test historical data generation."""
        historical_data = generate_historical_data()

        assert historical_data is not None
        assert len(historical_data) == 500
        assert 'spot_return' in historical_data.columns
        assert 'vol_change' in historical_data.columns
        assert 'rate_shift' in historical_data.columns

    def test_main_execution(self):
        """Test that main execution completes without errors."""
        try:
            historical_var_main()
            assert True
        except Exception as e:
            pytest.fail(f"Main execution raised exception: {e}")

    def test_historical_var_calculation(self):
        """Test that Historical VaR can be calculated."""
        from quantark.var import HistoricalVaREngine, VaRConfig, VaRMethod

        portfolio = create_options_portfolio()
        historical_data = generate_historical_data()

        config = VaRConfig(
            confidence_level=0.99,
            holding_period=1,
            var_method=VaRMethod.HISTORICAL,
        )

        engine = HistoricalVaREngine(config=config)
        result = engine.calculate_var(portfolio, historical_data)

        assert result is not None
        assert result.var > 0
        assert result.cvar > 0
        assert result.cvar >= result.var
        assert result.execution_time_seconds > 0
        assert result.execution_time_seconds < 15  # Historical is slower

    def test_non_linear_effects_captured(self):
        """Test that scenarios are available for non-linear analysis."""
        from quantark.var import HistoricalVaREngine, VaRConfig, VaRMethod

        portfolio = create_options_portfolio()
        historical_data = generate_historical_data()

        config = VaRConfig(
            confidence_level=0.99,
            holding_period=1,
            var_method=VaRMethod.HISTORICAL,
        )

        engine = HistoricalVaREngine(config=config)
        result = engine.calculate_var(portfolio, historical_data)

        # Historical VaR should provide scenarios for analysis
        assert result.scenarios is not None
        assert len(result.scenarios) > 0
        # Check for P&L column (could be 'pnl' or 'portfolio_pnl')
        pnl_columns = [col for col in result.scenarios.columns if 'pnl' in col.lower()]
        assert len(pnl_columns) > 0, f"No P&L column found in {result.scenarios.columns.tolist()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
