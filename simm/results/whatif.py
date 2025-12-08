"""
SIMM What-If Analysis Module.

This module provides what-if analysis capabilities for SIMM calculations,
enabling users to assess the impact of adding or removing positions.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import date

from ..taxonomy import ProductClass, RiskClass
from ..sensitivity import SensitivityCollection, Sensitivity
from .simm_result import SIMMResult


@dataclass
class WhatIfResult:
    """Result of what-if analysis."""
    base_simm: float
    new_simm: float
    delta_simm: float
    delta_pct: float

    # Breakdown of change
    delta_by_product_class: Dict[ProductClass, float]
    delta_by_risk_class: Dict[RiskClass, float]

    # Optional: affected positions
    affected_positions: Optional[List[str]] = None


class SIMMWhatIf:
    """What-if analysis for SIMM impact."""

    def __init__(self, base_result: SIMMResult, calculator: Any):
        """Initialize what-if analysis.

        Args:
            base_result: Base SIMMResult to compare against.
            calculator: SIMMCalculator instance for recalculation.
        """
        self.base_result = base_result
        self.calculator = calculator

    def impact_of_adding(
        self,
        new_sensitivities: SensitivityCollection,
    ) -> WhatIfResult:
        """Calculate SIMM impact of adding new sensitivities.

        Args:
            new_sensitivities: New sensitivities to add.

        Returns:
            WhatIfResult showing the impact.
        """
        # Recalculate SIMM with new sensitivities
        # This is a simplified implementation - in practice, you'd want to
        # optimize by only recalculating affected risk classes

        # Get base sensitivities from the result (if available)
        # In practice, you'd store this in the result or retrieve from calculator
        base_sensitivities = getattr(
            self.base_result,
            '_base_sensitivities',
            SensitivityCollection()
        )

        # Combine sensitivities
        all_sens = base_sensitivities.sensitivities + new_sensitivities.sensitivities
        combined_sensitivities = SensitivityCollection(all_sens)

        # Recalculate
        new_result = self.calculator.calculate(
            combined_sensitivities,
            calculation_date=self.base_result.calculation_date,
            calculation_currency=self.base_result.calculation_currency
        )

        # Build what-if result
        return self._build_whatif_result(
            base=self.base_result,
            new=new_result,
            affected_positions=getattr(new_sensitivities, 'position_ids', set())
        )

    def impact_of_removing(
        self,
        position_ids: List[str],
    ) -> WhatIfResult:
        """Calculate SIMM impact of removing positions.

        Args:
            position_ids: List of position IDs to remove.

        Returns:
            WhatIfResult showing the impact.
        """
        # Get base sensitivities
        base_sensitivities = getattr(
            self.base_result,
            '_base_sensitivities',
            SensitivityCollection()
        )

        # Filter out sensitivities for removed positions
        remaining_sensitivities = SensitivityCollection([
            sens for sens in base_sensitivities.sensitivities
            if getattr(sens, 'position_id', '') not in position_ids
        ])

        # Recalculate
        new_result = self.calculator.calculate(
            remaining_sensitivities,
            calculation_date=self.base_result.calculation_date,
            calculation_currency=self.base_result.calculation_currency
        )

        # Build what-if result
        return self._build_whatif_result(
            base=self.base_result,
            new=new_result,
            affected_positions=position_ids
        )

    def marginal_simm(
        self,
        sensitivity: Sensitivity,
    ) -> float:
        """Calculate marginal SIMM for a single sensitivity.

        This calculates the change in SIMM from adding a single sensitivity,
        providing the gradient for optimization.

        Args:
            sensitivity: Single sensitivity to analyze.

        Returns:
            Marginal SIMM impact.
        """
        # Create a collection with just this sensitivity
        single_sens = SensitivityCollection([sensitivity])

        # Get impact
        impact = self.impact_of_adding(single_sens)

        return impact.delta_simm

    def _build_whatif_result(
        self,
        base: SIMMResult,
        new: SIMMResult,
        affected_positions: Optional[List[str]] = None
    ) -> WhatIfResult:
        """Build WhatIfResult from base and new results.

        Args:
            base: Base SIMMResult.
            new: New SIMMResult after change.
            affected_positions: List of affected position IDs.

        Returns:
            WhatIfResult instance.
        """
        base_simm = base.total_simm
        new_simm = new.total_simm
        delta_simm = new_simm - base_simm
        delta_pct = (delta_simm / base_simm * 100 if base_simm != 0 else 0)

        # Calculate deltas by product class
        delta_by_product_class = {}
        for pc in ProductClass:
            base_pc = base.product_class_simm.get(pc, 0)
            new_pc = new.product_class_simm.get(pc, 0)
            delta_by_product_class[pc] = new_pc - base_pc

        # Calculate deltas by risk class
        delta_by_risk_class = {}
        base_by_risk = base.get_margin_by_risk_class()
        new_by_risk = new.get_margin_by_risk_class()

        for rc in RiskClass:
            delta_by_risk_class[rc] = new_by_risk.get(rc, 0) - base_by_risk.get(rc, 0)

        return WhatIfResult(
            base_simm=base_simm,
            new_simm=new_simm,
            delta_simm=delta_simm,
            delta_pct=delta_pct,
            delta_by_product_class=delta_by_product_class,
            delta_by_risk_class=delta_by_risk_class,
            affected_positions=affected_positions
        )

    def sensitivity_optimization(
        self,
        candidate_sensitivities: List[Sensitivity],
        target_reduction: float,
    ) -> Dict[str, Any]:
        """Find optimal set of sensitivities to remove to achieve target reduction.

        Args:
            candidate_sensitivities: List of candidate sensitivities to consider.
            target_reduction: Desired SIMM reduction amount.

        Returns:
            Dictionary with optimization results.
        """
        results = []
        remaining_sensitivities = candidate_sensitivities.copy()

        # Sort by marginal impact (most negative first)
        marginal_impacts = []
        for sens in remaining_sensitivities:
            marginal = self.marginal_simm(sens)
            marginal_impacts.append((sens, marginal))

        marginal_impacts.sort(key=lambda x: x[1])  # Sort by marginal (most negative first)

        # Greedy selection
        selected = []
        total_reduction = 0

        for sens, marginal in marginal_impacts:
            if total_reduction >= target_reduction:
                break

            selected.append(sens)
            total_reduction += abs(marginal)

        return {
            "selected_sensitivities": selected,
            "estimated_reduction": total_reduction,
            "target_reduction": target_reduction,
            "achievement_pct": (total_reduction / target_reduction * 100
                              if target_reduction != 0 else 0)
        }

    def scenario_analysis(
        self,
        scenarios: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Perform multiple scenario analysis.

        Args:
            scenarios: List of scenario dictionaries, each containing:
                - 'name': Scenario name
                - 'add_sensitivities': Sensitivities to add (optional)
                - 'remove_positions': Positions to remove (optional)

        Returns:
            Dictionary mapping scenario names to WhatIfResult.
        """
        scenario_results = {}

        for scenario in scenarios:
            name = scenario['name']

            if 'add_sensitivities' in scenario:
                result = self.impact_of_adding(scenario['add_sensitivities'])
            elif 'remove_positions' in scenario:
                result = self.impact_of_removing(scenario['remove_positions'])
            else:
                continue

            scenario_results[name] = result

        return scenario_results
