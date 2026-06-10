"""
Tests for SIMM what-if analysis.
"""
from datetime import date
from unittest.mock import Mock, MagicMock

import pytest

from quantark.simm.taxonomy import ProductClass, RiskClass
from quantark.simm.results.simm_result import SIMMResult, RiskClassMargin
from quantark.simm.results.whatif import SIMMWhatIf, WhatIfResult
from quantark.simm.sensitivity import SensitivityCollection


class TestSIMMWhatIf:
    """Test SIMMWhatIf class."""

    def test_init(self):
        """Test initializing SIMMWhatIf."""
        base_result = Mock(spec=SIMMResult)
        calculator = Mock()

        what_if = SIMMWhatIf(base_result, calculator)

        assert what_if.base_result == base_result
        assert what_if.calculator == calculator

    def test_impact_of_adding(self):
        """Test impact of adding new sensitivities."""
        # Create base result
        base_result = SIMMResult(
            total_simm=1000.0,
            calculation_currency="USD",
            calculation_date=date(2024, 1, 15),
            simm_version="2.6",
            product_class_simm={pc: 250.0 for pc in ProductClass},
            risk_class_margin={},
            addon_amount=0.0,
        )

        # Create mock calculator
        calculator = Mock()
        calculator.calculate = Mock(return_value=SIMMResult(
            total_simm=1200.0,
            calculation_currency="USD",
            calculation_date=date(2024, 1, 15),
            simm_version="2.6",
            product_class_simm={pc: 300.0 for pc in ProductClass},
            risk_class_margin={},
            addon_amount=0.0,
        ))

        # Create what-if analyzer
        what_if = SIMMWhatIf(base_result, calculator)

        # Create new sensitivities
        new_sensitivities = SensitivityCollection([])

        # Get impact
        result = what_if.impact_of_adding(new_sensitivities)

        assert isinstance(result, WhatIfResult)
        assert result.base_simm == 1000.0
        assert result.new_simm == 1200.0
        assert result.delta_simm == 200.0
        assert result.delta_pct == 20.0

    def test_impact_of_removing(self):
        """Test impact of removing positions."""
        # Create base result
        base_result = SIMMResult(
            total_simm=1000.0,
            calculation_currency="USD",
            calculation_date=date(2024, 1, 15),
            simm_version="2.6",
            product_class_simm={pc: 250.0 for pc in ProductClass},
            risk_class_margin={},
            addon_amount=0.0,
        )

        # Create mock calculator
        calculator = Mock()
        calculator.calculate = Mock(return_value=SIMMResult(
            total_simm=800.0,
            calculation_currency="USD",
            calculation_date=date(2024, 1, 15),
            simm_version="2.6",
            product_class_simm={pc: 200.0 for pc in ProductClass},
            risk_class_margin={},
            addon_amount=0.0,
        ))

        # Create what-if analyzer
        what_if = SIMMWhatIf(base_result, calculator)

        # Get impact of removing positions
        result = what_if.impact_of_removing(["POS_001", "POS_002"])

        assert isinstance(result, WhatIfResult)
        assert result.base_simm == 1000.0
        assert result.new_simm == 800.0
        assert result.delta_simm == -200.0
        assert result.delta_pct == -20.0

    def test_marginal_simm(self):
        """Test marginal SIMM calculation."""
        # Create base result
        base_result = SIMMResult(
            total_simm=1000.0,
            calculation_currency="USD",
            calculation_date=date(2024, 1, 15),
            simm_version="2.6",
            product_class_simm={pc: 250.0 for pc in ProductClass},
            risk_class_margin={},
            addon_amount=0.0,
        )

        # Create mock calculator
        calculator = Mock()
        calculator.calculate = Mock(return_value=SIMMResult(
            total_simm=1010.0,
            calculation_currency="USD",
            calculation_date=date(2024, 1, 15),
            simm_version="2.6",
            product_class_simm={pc: 252.5 for pc in ProductClass},
            risk_class_margin={},
            addon_amount=0.0,
        ))

        # Create what-if analyzer
        what_if = SIMMWhatIf(base_result, calculator)

        # Create mock sensitivity
        sensitivity = Mock()
        sensitivity.position_id = "POS_001"

        # Get marginal SIMM
        marginal = what_if.marginal_simm(sensitivity)

        assert marginal == 10.0

    def test_sensitivity_optimization(self):
        """Test sensitivity optimization."""
        # Create base result
        base_result = SIMMResult(
            total_simm=1000.0,
            calculation_currency="USD",
            calculation_date=date(2024, 1, 15),
            simm_version="2.6",
            product_class_simm={pc: 250.0 for pc in ProductClass},
            risk_class_margin={},
            addon_amount=0.0,
        )

        # Create mock calculator
        calculator = Mock()
        calculator.calculate = Mock(return_value=SIMMResult(
            total_simm=900.0,
            calculation_currency="USD",
            calculation_date=date(2024, 1, 15),
            simm_version="2.6",
            product_class_simm={pc: 225.0 for pc in ProductClass},
            risk_class_margin={},
            addon_amount=0.0,
        ))

        # Create what-if analyzer
        what_if = SIMMWhatIf(base_result, calculator)

        # Create candidate sensitivities with different marginal impacts
        sensitivities = [Mock() for _ in range(5)]
        for i, sens in enumerate(sensitivities):
            sens.position_id = f"POS_{i:03d}"

        # Mock marginal_simm to return decreasing values
        what_if.marginal_simm = Mock(side_effect=[-50.0, -30.0, -20.0, -10.0, -5.0])

        # Run optimization
        result = what_if.sensitivity_optimization(sensitivities, target_reduction=80.0)

        assert "selected_sensitivities" in result
        assert "estimated_reduction" in result
        assert "target_reduction" in result
        assert "achievement_pct" in result
        assert result["target_reduction"] == 80.0

    def test_scenario_analysis(self):
        """Test multiple scenario analysis."""
        # Create base result
        base_result = SIMMResult(
            total_simm=1000.0,
            calculation_currency="USD",
            calculation_date=date(2024, 1, 15),
            simm_version="2.6",
            product_class_simm={pc: 250.0 for pc in ProductClass},
            risk_class_margin={},
            addon_amount=0.0,
        )

        # Create mock calculator
        calculator = Mock()
        calculator.calculate = Mock(return_value=SIMMResult(
            total_simm=1100.0,
            calculation_currency="USD",
            calculation_date=date(2024, 1, 15),
            simm_version="2.6",
            product_class_simm={pc: 275.0 for pc in ProductClass},
            risk_class_margin={},
            addon_amount=0.0,
        ))

        # Create what-if analyzer
        what_if = SIMMWhatIf(base_result, calculator)

        # Define scenarios
        scenarios = [
            {
                'name': 'Add Trade 1',
                'add_sensitivities': SensitivityCollection([]),
            },
            {
                'name': 'Remove Position 1',
                'remove_positions': ['POS_001'],
            },
        ]

        # Run scenario analysis
        results = what_if.scenario_analysis(scenarios)

        assert "Add Trade 1" in results
        assert "Remove Position 1" in results
        assert isinstance(results["Add Trade 1"], WhatIfResult)


class TestWhatIfResult:
    """Test WhatIfResult dataclass."""

    def test_create_whatif_result(self):
        """Test creating a WhatIfResult."""
        result = WhatIfResult(
            base_simm=1000.0,
            new_simm=1200.0,
            delta_simm=200.0,
            delta_pct=20.0,
            delta_by_product_class={pc: 50.0 for pc in ProductClass},
            delta_by_risk_class={rc: 40.0 for rc in RiskClass},
            affected_positions=["POS_001", "POS_002"],
        )

        assert result.base_simm == 1000.0
        assert result.new_simm == 1200.0
        assert result.delta_simm == 200.0
        assert result.delta_pct == 20.0
        assert result.affected_positions == ["POS_001", "POS_002"]

    def test_calculate_percentages(self):
        """Test percentage calculations."""
        result = WhatIfResult(
            base_simm=1000.0,
            new_simm=1100.0,
            delta_simm=100.0,
            delta_pct=10.0,
            delta_by_product_class={},
            delta_by_risk_class={},
        )

        assert result.delta_pct == 10.0


if __name__ == "__main__":
    pytest.main([__file__])
