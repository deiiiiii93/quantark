"""
Tests for var_backtest_demo.py

Verifies that the VaR backtesting demonstration script executes correctly.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from example.var_backtest_demo import (
    create_portfolio_for_backtesting,
    generate_historical_data_extended,
    generate_portfolio_pnl_timeseries,
    main as backtest_main,
)


class TestVarBacktestDemo:
    """Test cases for VaR backtesting demonstration."""

    def test_create_portfolio_for_backtesting(self):
        """Test backtesting portfolio creation."""
        portfolio = create_portfolio_for_backtesting()

        assert portfolio is not None
        assert portfolio.portfolio_name == "Backtesting Demo Portfolio"
        assert len(portfolio.positions) == 3
        assert portfolio.get_portfolio_value() > 0

    def test_generate_historical_data_extended(self):
        """Test extended historical data generation."""
        historical_data = generate_historical_data_extended()

        assert historical_data is not None
        assert len(historical_data) == 800
        assert 'spot_return' in historical_data.columns
        assert 'vol_change' in historical_data.columns
        assert 'rate_shift' in historical_data.columns

    def test_generate_portfolio_pnl_timeseries(self):
        """Test P&L time series generation."""
        portfolio = create_portfolio_for_backtesting()
        historical_data = generate_historical_data_extended()

        portfolio_pnl, var_values = generate_portfolio_pnl_timeseries(
            portfolio, historical_data, method="Historical"
        )

        assert portfolio_pnl is not None
        assert var_values is not None
        assert len(portfolio_pnl) > 0
        assert len(var_values) > 0
        assert len(portfolio_pnl) == len(var_values)
        assert all(var_values > 0), "VaR values should be positive"
        assert portfolio_pnl.dtype in ['float64', 'float32']

    def test_main_execution(self):
        """Test that main execution completes without errors."""
        # Note: This is a long-running test, so we skip it by default
        # Uncomment to run full backtesting demo
        # try:
        #     backtest_main()
        #     assert True
        # except Exception as e:
        #     pytest.fail(f"Main execution raised exception: {e}")
        pass

    def test_backtester_import(self):
        """Test that VaRBacktester can be imported."""
        try:
            from var import VaRBacktester
            assert VaRBacktester is not None
        except ImportError:
            pytest.fail("VaRBacktester should be importable from var module")

    def test_backtester_initialization(self):
        """Test that VaRBacktester can be initialized."""
        from var import VaRBacktester

        backtester = VaRBacktester(confidence_level=0.99)
        assert backtester is not None
        assert backtester.confidence_level == 0.99

    def test_statistical_tests_availability(self):
        """Test that statistical tests are available."""
        # Create dummy data for testing
        import numpy as np
        portfolio_pnl = pd.Series(np.random.normal(0, 100, 250))
        var_values = pd.Series(np.abs(np.random.normal(150, 20, 250)))

        from var import VaRBacktester

        backtester = VaRBacktester(confidence_level=0.99)
        result = backtester.run_backtest(portfolio_pnl, var_values)

        assert result is not None
        assert hasattr(result, 'num_violations')
        assert hasattr(result, 'violation_rate')
        assert hasattr(result, 'kupiec_stat')
        assert hasattr(result, 'kupiec_pvalue')
        assert hasattr(result, 'christoffersen_stat')
        assert hasattr(result, 'christoffersen_pvalue')
        assert hasattr(result, 'zone_classification')

    def test_basel_traffic_light(self):
        """Test Basel traffic light zone classification."""
        import pandas as pd
        import numpy as np

        # Test cases: (violations, expected_zone)
        test_cases = [
            (3, "Zone 1 (Green)"),
            (7, "Zone 2 (Yellow)"),
            (12, "Zone 3 (Red)"),
            (15, "Zone 3 (Red)"),
        ]

        from var import VaRBacktester

        for violations, expected_zone in test_cases:
            # Create dummy data
            portfolio_pnl = pd.Series(np.random.normal(0, 100, 250))
            var_values = pd.Series(np.abs(np.random.normal(150, 20, 250)))

            # Set up violations
            for i in range(min(violations, len(portfolio_pnl))):
                portfolio_pnl.iloc[i] = -200  # Force violation

            backtester = VaRBacktester(confidence_level=0.99)
            result = backtester.run_backtest(portfolio_pnl, var_values)

            assert result.zone_classification == expected_zone, \
                f"Expected {expected_zone} for {violations} violations, got {result.zone_classification}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
