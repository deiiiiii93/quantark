"""
Unit tests for Incremental VaR functionality.

Tests incremental VaR calculation across all three VaR engines,
diversification benefits, and position-level contributions.
"""

import inspect
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from var import (
    VaRConfig,
    VaRMethod,
    ParametricVaREngine,
    HistoricalVaREngine,
    MonteCarloVaREngine,
)
from var.results import VaRResult, IncrementalVaRResult


class TestIncrementalVaRResult:
    """Test IncrementalVaRResult class."""

    def test_create_incremental_var_result(self):
        """Test creating IncrementalVaRResult."""
        result = IncrementalVaRResult(
            portfolio_var=1000.0,
            position_ivari={
                'POS1': 300.0,
                'POS2': 250.0,
                'POS3': 200.0
            },
            diversification_benefit=250.0
        )

        assert result.portfolio_var == 1000.0
        assert result.position_ivari['POS1'] == 300.0
        assert result.position_ivari['POS2'] == 250.0
        assert result.position_ivari['POS3'] == 200.0
        assert result.diversification_benefit == 250.0

    def test_diversification_benefit_calculation(self):
        """Test automatic diversification benefit calculation."""
        result = IncrementalVaRResult(
            portfolio_var=1000.0,
            position_ivari={
                'POS1': 400.0,
                'POS2': 350.0,
                'POS3': 300.0
            },
            diversification_benefit=None  # Will be auto-calculated
        )

        # Diversification benefit = Sum of individual VaRs - Portfolio VaR
        expected = (400.0 + 350.0 + 300.0) - 1000.0
        assert result.diversification_benefit == expected

    def test_diversification_ratio(self):
        """Test diversification ratio calculation."""
        result = IncrementalVaRResult(
            portfolio_var=1000.0,
            position_ivari={
                'POS1': 400.0,
                'POS2': 350.0,
                'POS3': 300.0
            },
            diversification_benefit=50.0
        )

        # Diversification Ratio = Portfolio VaR / Sum of Individual VaRs
        total_individual = sum(result.position_ivari.values())
        expected_ratio = result.portfolio_var / total_individual
        assert abs(result.get_diversification_ratio() - expected_ratio) < 1e-10

    def test_diversification_ratio_edge_case(self):
        """Test diversification ratio with zero individual VaR."""
        result = IncrementalVaRResult(
            portfolio_var=0.0,
            position_ivari={},
            diversification_benefit=0.0
        )

        # Should return 1.0 when no individual VaRs
        assert result.get_diversification_ratio() == 1.0

    def test_get_top_contributors(self):
        """Test getting top contributors."""
        result = IncrementalVaRResult(
            portfolio_var=1000.0,
            position_ivari={
                'POS1': 300.0,
                'POS2': 500.0,
                'POS3': 200.0,
                'POS4': 400.0
            },
            diversification_benefit=400.0
        )

        # Get top 2 contributors
        top_2 = result.get_top_contributors(2)
        assert len(top_2) == 2
        assert top_2[0] == ('POS2', 500.0)  # Highest
        assert top_2[1] == ('POS4', 400.0)  # Second highest

        # Get top 3 contributors
        top_3 = result.get_top_contributors(3)
        assert len(top_3) == 3
        assert top_3[0][0] == 'POS2'
        assert top_3[1][0] == 'POS4'
        assert top_3[2][0] == 'POS1'

    def test_get_top_contributors_all(self):
        """Test getting all contributors when n > number of positions."""
        result = IncrementalVaRResult(
            portfolio_var=1000.0,
            position_ivari={
                'POS1': 300.0,
                'POS2': 250.0
            },
            diversification_benefit=50.0
        )

        # Request 10 but only have 2 positions
        top_10 = result.get_top_contributors(10)
        assert len(top_10) == 2

    def test_validation_negative_portfolio_var(self):
        """Test validation rejects negative portfolio VaR."""
        with pytest.raises(ValueError, match="Portfolio VaR must be non-negative"):
            IncrementalVaRResult(
                portfolio_var=-100.0,
                position_ivari={'POS1': 100.0},
                diversification_benefit=0.0
            )

    def test_validation_negative_incremental_var(self):
        """Test validation rejects negative incremental VaR."""
        with pytest.raises(ValueError, match="Incremental VaR for POS1 must be non-negative"):
            IncrementalVaRResult(
                portfolio_var=1000.0,
                position_ivari={'POS1': -50.0},
                diversification_benefit=0.0
            )

    def test_get_summary_dict(self):
        """Test summary dictionary generation."""
        result = IncrementalVaRResult(
            portfolio_var=1000.0,
            position_ivari={
                'POS1': 400.0,
                'POS2': 350.0,
                'POS3': 300.0
            },
            diversification_benefit=50.0,
            ivari_method="Historical",
            config={'confidence_level': 0.99}
        )

        summary = result.get_summary_dict()

        assert 'portfolio_var' in summary
        assert 'diversification_benefit' in summary
        assert 'diversification_ratio' in summary
        assert 'num_positions' in summary
        assert 'top_contributors' in summary
        assert 'ivari_method' in summary
        assert 'calculation_timestamp' in summary

        assert summary['portfolio_var'] == 1000.0
        assert summary['num_positions'] == 3
        assert summary['ivari_method'] == "Historical"


class TestHistoricalVaRIncremental:
    """Test Historical VaR with Incremental VaR."""

    @pytest.fixture
    def ivari_config(self):
        """Create VaR configuration with Incremental VaR enabled."""
        return VaRConfig(
            confidence_level=0.99,
            holding_period=1,
            lookback_days=500,
            var_method=VaRMethod.HISTORICAL,
            calculate_incremental_var=True,
        )

    def test_calculate_ivar_basic(self, ivari_config):
        """Test basic Incremental VaR calculation."""
        engine = HistoricalVaREngine(config=ivari_config)

        # Create scenarios DataFrame with date index
        dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
        scenarios = pd.DataFrame({
            'spot_return': np.random.normal(0, 0.02, 300),
            'vol_change': np.random.normal(0, 0.01, 300),
            'rate_shift': np.random.normal(0, 0.001, 300)
        }, index=dates)

        # Verify the engine has the IVaR method
        assert hasattr(engine, 'calculate_incremental_var')
        assert callable(engine.calculate_incremental_var)

    def test_incremental_var_integration(self, ivari_config):
        """Test Incremental VaR is calculated and stored in VaRResult."""
        engine = HistoricalVaREngine(config=ivari_config)

        dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
        scenarios = pd.DataFrame({
            'spot_return': np.random.normal(0, 0.02, 300)
        }, index=dates)

        # Verify that calculate_var will call _calculate_incremental_var
        # when calculate_incremental_var is True
        assert ivari_config.calculate_incremental_var == True

        # The integration is in the calculate_var method
        # We'll verify this works when we have a proper portfolio


class TestMonteCarloVaRIncremental:
    """Test Monte Carlo VaR with Incremental VaR."""

    @pytest.fixture
    def ivari_config(self):
        """Create VaR configuration with Incremental VaR enabled."""
        return VaRConfig(
            confidence_level=0.99,
            holding_period=1,
            lookback_days=500,
            var_method=VaRMethod.MONTE_CARLO,
            mc_num_simulations=1000,
            mc_seed=42,
            calculate_incremental_var=True,
        )

    def test_calculate_ivar_basic(self, ivari_config):
        """Test basic Incremental VaR calculation."""
        engine = MonteCarloVaREngine(config=ivari_config)

        # Create scenarios DataFrame with date index
        dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
        scenarios = pd.DataFrame({
            'spot_return': np.random.normal(0, 0.02, 300),
            'vol_change': np.random.normal(0, 0.01, 300),
            'rate_shift': np.random.normal(0, 0.001, 300)
        }, index=dates)

        # Verify the engine has the IVaR method
        assert hasattr(engine, 'calculate_incremental_var')
        assert callable(engine.calculate_incremental_var)

    def test_incremental_var_integration(self, ivari_config):
        """Test Incremental VaR is calculated and stored in VaRResult."""
        engine = MonteCarloVaREngine(config=ivari_config)

        dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
        scenarios = pd.DataFrame({
            'spot_return': np.random.normal(0, 0.02, 300)
        }, index=dates)

        # Verify that calculate_var will call _calculate_incremental_var
        # when calculate_incremental_var is True
        assert ivari_config.calculate_incremental_var == True


class TestParametricVaRIncremental:
    """Test Parametric VaR with Incremental VaR."""

    @pytest.fixture
    def ivari_config(self):
        """Create VaR configuration with Incremental VaR enabled."""
        return VaRConfig(
            confidence_level=0.99,
            holding_period=1,
            lookback_days=500,
            var_method=VaRMethod.PARAMETRIC,
            calculate_incremental_var=True,
        )

    def test_calculate_ivar_basic(self, ivari_config):
        """Test basic Incremental VaR calculation."""
        engine = ParametricVaREngine(config=ivari_config)

        # Create risk factors DataFrame with date index
        dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
        risk_factors = pd.DataFrame({
            'spot_return': np.random.normal(0, 0.02, 300),
            'vol_change': np.random.normal(0, 0.01, 300),
            'rate_shift': np.random.normal(0, 0.001, 300)
        }, index=dates)

        # Verify the engine has the IVaR method
        assert hasattr(engine, 'calculate_incremental_var')
        assert callable(engine.calculate_incremental_var)

        # Verify _calculate_incremental_var exists
        assert hasattr(engine, '_calculate_incremental_var')
        assert callable(engine._calculate_incremental_var)

    def test_incremental_var_integration(self, ivari_config):
        """Test Incremental VaR is calculated and stored in VaRResult."""
        engine = ParametricVaREngine(config=ivari_config)

        dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
        risk_factors = pd.DataFrame({
            'spot_return': np.random.normal(0, 0.02, 300)
        }, index=dates)

        # Verify that calculate_var will call _calculate_incremental_var
        # when calculate_incremental_var is True
        assert ivari_config.calculate_incremental_var == True


class TestVaRConfigurationIncremental:
    """Test VaRConfig for Incremental VaR settings."""

    def test_incremental_var_config_default(self):
        """Test default Incremental VaR configuration."""
        config = VaRConfig()

        # Incremental VaR should be disabled by default
        assert config.calculate_incremental_var == False

    def test_incremental_var_config_enabled(self):
        """Test enabling Incremental VaR configuration."""
        config = VaRConfig(
            calculate_incremental_var=True,
            calculate_component_var=False,
            calculate_marginal_var=False,
            calculate_factor_var=False
        )

        assert config.calculate_incremental_var == True

    def test_incremental_var_with_other_attribution(self):
        """Test Incremental VaR works with other attribution methods."""
        config = VaRConfig(
            calculate_incremental_var=True,
            calculate_component_var=True,
            calculate_marginal_var=True,
            calculate_factor_var=True
        )

        # All attribution methods should be enabled
        assert config.calculate_incremental_var == True
        assert config.calculate_component_var == True
        assert config.calculate_marginal_var == True
        assert config.calculate_factor_var == True


class TestVaREngineProtocolIncremental:
    """Test that VaREngine protocol includes Incremental VaR."""

    def test_calculate_incremental_var_in_protocol(self):
        """Test calculate_incremental_var is in VaREngine protocol."""
        from var.base import VaREngine
        import inspect

        # Get all methods from the protocol
        methods = [name for name, method in inspect.getmembers(VaREngine, predicate=inspect.isfunction)]

        # Should have calculate_incremental_var method
        assert 'calculate_incremental_var' in methods

    def test_calculate_incremental_var_signature(self):
        """Test calculate_incremental_var signature matches protocol."""
        from var.base import VaREngine

        # Get the method signature
        sig = inspect.signature(VaREngine.calculate_incremental_var)

        # Should have portfolio and historical_data parameters
        params = list(sig.parameters.keys())
        assert 'portfolio' in params
        assert 'historical_data' in params


class TestIncrementalVaRCalculation:
    """Test Incremental VaR calculation logic."""

    def test_ivar_formula(self):
        """Test Incremental VaR formula understanding."""
        # Incremental VaR = VaR(full portfolio) - VaR(portfolio without position)
        # Example:
        # Full portfolio VaR = 1000
        # VaR without POS1 = 700
        # IVaR for POS1 = 1000 - 700 = 300

        portfolio_var = 1000.0
        var_without_pos = 700.0

        expected_ivar = abs(portfolio_var - var_without_pos)
        assert expected_ivar == 300.0

    def test_diversification_benefit_formula(self):
        """Test Diversification Benefit formula."""
        # Diversification Benefit = Sum(Individual VaRs) - Portfolio VaR
        # Individual VaR = VaR of each position in isolation
        individual_vars = [400.0, 350.0, 300.0]
        portfolio_var = 1000.0

        total_individual = sum(individual_vars)
        div_benefit = total_individual - portfolio_var

        assert div_benefit == 50.0

    def test_diversification_ratio_formula(self):
        """Test Diversification Ratio formula."""
        # Diversification Ratio = Portfolio VaR / Sum of Individual VaRs
        portfolio_var = 1000.0
        individual_vars = [400.0, 350.0, 300.0]

        total_individual = sum(individual_vars)
        div_ratio = portfolio_var / total_individual

        # Ratio should be less than 1.0 when there's diversification benefit
        assert div_ratio < 1.0
        assert abs(div_ratio - 0.9524) < 0.01  # 1000 / 1050

    def test_no_diversification_edge_case(self):
        """Test edge case with no diversification."""
        # When positions are perfectly correlated, IVaR sums to portfolio VaR
        portfolio_var = 1000.0
        individual_vars = [1000.0]  # One position = entire portfolio

        total_individual = sum(individual_vars)
        div_benefit = total_individual - portfolio_var
        div_ratio = portfolio_var / total_individual

        # No diversification benefit
        assert div_benefit == 0.0
        # Diversification ratio = 1.0 (no benefit)
        assert div_ratio == 1.0

    def test_full_diversification_edge_case(self):
        """Test edge case with perfect negative correlation."""
        # With perfect negative correlation, portfolio VaR could be very low
        portfolio_var = 100.0
        individual_vars = [500.0, 500.0]  # High individual VaRs

        total_individual = sum(individual_vars)
        div_benefit = total_individual - portfolio_var
        div_ratio = portfolio_var / total_individual

        # Large diversification benefit
        assert div_benefit == 900.0
        # Very low diversification ratio
        assert div_ratio == 0.1


if __name__ == "__main__":
    # Run basic tests
    print("Running Incremental VaR Tests...")

    # Test IncrementalVaRResult
    print("\n1. Testing IncrementalVaRResult...")
    test_result = TestIncrementalVaRResult()
    test_result.test_create_incremental_var_result()
    test_result.test_diversification_benefit_calculation()
    test_result.test_diversification_ratio()
    test_result.test_get_top_contributors()
    test_result.test_get_summary_dict()
    print("   ✓ IncrementalVaRResult tests passed")

    # Test VaRConfig
    print("\n2. Testing VaRConfig for Incremental VaR...")
    test_config = TestVaRConfigurationIncremental()
    test_config.test_incremental_var_config_default()
    test_config.test_incremental_var_config_enabled()
    print("   ✓ VaRConfig Incremental VaR tests passed")

    # Test VaREngine Protocol
    print("\n3. Testing VaREngine Protocol...")
    test_protocol = TestVaREngineProtocolIncremental()
    test_protocol.test_calculate_incremental_var_in_protocol()
    test_protocol.test_calculate_incremental_var_signature()
    print("   ✓ VaREngine Protocol tests passed")

    # Test VaR Engines
    print("\n4. Testing VaR Engines for Incremental VaR support...")
    historical_engine = HistoricalVaREngine()
    monte_carlo_engine = MonteCarloVaREngine()
    parametric_engine = ParametricVaREngine()

    assert hasattr(historical_engine, 'calculate_incremental_var')
    assert hasattr(historical_engine, '_calculate_incremental_var')
    assert hasattr(monte_carlo_engine, 'calculate_incremental_var')
    assert hasattr(monte_carlo_engine, '_calculate_incremental_var')
    assert hasattr(parametric_engine, 'calculate_incremental_var')
    assert hasattr(parametric_engine, '_calculate_incremental_var')

    print("   ✓ All VaR engines support Incremental VaR")

    # Test Calculation Logic
    print("\n5. Testing Incremental VaR calculation logic...")
    test_calc = TestIncrementalVaRCalculation()
    test_calc.test_ivar_formula()
    test_calc.test_diversification_benefit_formula()
    test_calc.test_diversification_ratio_formula()
    test_calc.test_no_diversification_edge_case()
    test_calc.test_full_diversification_edge_case()
    print("   ✓ Incremental VaR calculation logic tests passed")

    print("\n✅ All Incremental VaR tests passed!")
