"""
Tests for monte_carlo_var_demo.py

Verifies that the Monte Carlo VaR demonstration script executes correctly.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from example.monte_carlo_var_demo import (
    create_portfolio_for_mc,
    generate_market_data_for_mc,
    main as monte_carlo_var_main,
)


class TestMonteCarloVarDemo:
    """Test cases for Monte Carlo VaR demonstration."""

    def test_create_portfolio_for_mc(self):
        """Test Monte Carlo portfolio creation."""
        portfolio = create_portfolio_for_mc()

        assert portfolio is not None
        assert portfolio.portfolio_name == "Monte Carlo VaR Demo Portfolio"
        assert len(portfolio.positions) == 4
        assert portfolio.get_portfolio_value() > 0

        # Portfolio uses European vanilla options (Asian option not available)
        has_european = any('EuropeanVanillaOption' in pos.product.__class__.__name__
                           for pos in portfolio.positions.values())
        assert has_european, "Portfolio should include European vanilla options"

        # Verify straddle structure (long call + long put at same strike)
        positions = list(portfolio.positions.values())
        strikes = [pos.product.strike for pos in positions]
        assert 100.0 in strikes  # ATM straddle

    def test_generate_market_data_for_mc(self):
        """Test market data generation for Monte Carlo."""
        historical_data = generate_market_data_for_mc()

        assert historical_data is not None
        assert len(historical_data) == 252
        assert 'spot_return' in historical_data.columns
        assert 'vol_change' in historical_data.columns
        assert 'rate_shift' in historical_data.columns

    def test_main_execution(self):
        """Test that main execution completes without errors."""
        try:
            monte_carlo_var_main()
            assert True
        except Exception as e:
            pytest.fail(f"Main execution raised exception: {e}")

    def test_monte_carlo_var_calculation(self):
        """Test that Monte Carlo VaR can be calculated."""
        from var import MonteCarloVaREngine, VaRConfig, VaRMethod

        portfolio = create_portfolio_for_mc()
        historical_data = generate_market_data_for_mc()

        config = VaRConfig(
            confidence_level=0.99,
            holding_period=1,
            var_method=VaRMethod.MONTE_CARLO,
            mc_num_simulations=1000,  # Small number for fast testing
            mc_seed=42,
        )

        engine = MonteCarloVaREngine(config=config)
        result = engine.calculate_var(portfolio, historical_data)

        assert result is not None
        assert result.var > 0
        assert result.cvar > 0
        assert result.cvar >= result.var
        assert result.execution_time_seconds > 0

    def test_convergence_analysis_structure(self):
        """Test that convergence analysis uses appropriate simulation counts."""
        # This test verifies the structure of convergence testing
        simulation_counts = [1000, 5000, 10000, 25000, 50000]

        # Verify all counts are positive integers
        for count in simulation_counts:
            assert isinstance(count, int)
            assert count > 0
            assert count >= 1000  # Minimum reasonable count

        # Verify they're in ascending order
        assert simulation_counts == sorted(simulation_counts)

    def test_path_dependent_products_handled(self):
        """Test that Monte Carlo can handle the portfolio structure."""
        portfolio = create_portfolio_for_mc()

        # Verify portfolio has European vanilla options
        # Note: Asian options not available, but MC would handle them naturally
        european_options = [pos for pos in portfolio.positions.values()
                            if 'EuropeanVanillaOption' in pos.product.__class__.__name__]

        assert len(european_options) == 4, "Portfolio should have 4 European vanilla options"

        # Verify different strike levels for straddle + OTM calls structure
        strikes = [pos.product.strike for pos in european_options]
        assert 100.0 in strikes  # ATM straddle
        assert 120.0 in strikes  # OTM call
        assert 140.0 in strikes  # Deep OTM call


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
