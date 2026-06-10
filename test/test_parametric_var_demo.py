"""
Tests for parametric_var_demo.py

Verifies that the parametric VaR demonstration script executes correctly.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from example.parametric_var_demo import (
    create_sample_portfolio,
    generate_historical_data,
    main as parametric_var_main,
)


class TestParametricVarDemo:
    """Test cases for parametric VaR demonstration."""

    def test_create_sample_portfolio(self):
        """Test portfolio creation."""
        portfolio = create_sample_portfolio()

        assert portfolio is not None
        assert portfolio.portfolio_name == "Parametric VaR Demo Portfolio"
        assert len(portfolio.positions) == 3
        assert portfolio.get_portfolio_value() > 0

    def test_generate_historical_data(self):
        """Test historical data generation."""
        historical_data = generate_historical_data()

        assert historical_data is not None
        assert len(historical_data) == 300
        assert 'spot_return' in historical_data.columns
        assert 'vol_change' in historical_data.columns
        assert 'rate_shift' in historical_data.columns
        assert historical_data['spot_return'].dtype in ['float64', 'float32']
        assert historical_data['vol_change'].dtype in ['float64', 'float32']
        assert historical_data['rate_shift'].dtype in ['float64', 'float32']

    def test_main_execution(self):
        """Test that main execution completes without errors."""
        # Capture output or just ensure no exceptions are raised
        try:
            parametric_var_main()
            assert True
        except Exception as e:
            pytest.fail(f"Main execution raised exception: {e}")

    def test_var_calculation(self):
        """Test that VaR can be calculated."""
        from quantark.var import ParametricVaREngine, VaRConfig, VaRMethod, EquityRiskFactorConfig

        portfolio = create_sample_portfolio()
        historical_data = generate_historical_data()

        equity_factors = EquityRiskFactorConfig(
            include_spot=True,
            include_vol=True,
            include_rate=True,
            include_div_yield=False
        )

        config = VaRConfig(
            confidence_level=0.99,
            holding_period=1,
            var_method=VaRMethod.PARAMETRIC,
            equity_factors=equity_factors,
        )

        engine = ParametricVaREngine(config=config)
        result = engine.calculate_var(portfolio, historical_data)

        assert result is not None
        assert result.var > 0
        assert result.cvar > 0
        assert result.cvar >= result.var
        assert result.execution_time_seconds > 0
        assert result.execution_time_seconds < 10  # Should complete in reasonable time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
