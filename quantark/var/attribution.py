"""
VaR attribution module.

This module contains classes and functions for calculating VaR attribution
including component VaR (Euler allocation) and marginal VaR.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from quantark.util.exceptions import ValidationError
from quantark.var.results.var_result import VaRResult
from quantark.var.config import VaRConfig


class ComponentVaRCalculator:
    """
    Calculates component VaR using Euler decomposition.

    Component VaR allocates total portfolio VaR to individual positions
    using the Euler homogeneous property of VaR.
    """

    @staticmethod
    def calculate_from_sensitivities(
        position_values: Dict[str, float],
        sensitivities: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        confidence_level: float = 0.99,
    ) -> Dict[str, float]:
        """
        Calculate component VaR using first-order sensitivities.

        Uses the formula: Component VaR_i = ∂VaR/∂x_i * x_i
        where ∂VaR/∂x_i is the marginal VaR (calculated from sensitivity
        and covariance matrix).

        Args:
            position_values: Position market values by ID
            sensitivities: Position sensitivities (delta, DV01, etc.)
            covariance_matrix: Covariance matrix of risk factors
            confidence_level: VaR confidence level

        Returns:
            Dictionary mapping position ID to component VaR

        Raises:
            ValidationError: If inputs are invalid
            NumericalError: If calculation fails
        """
        if not position_values:
            return {}

        if not sensitivities:
            raise ValidationError("Sensitivities dictionary cannot be empty")

        # Calculate z-score for confidence level
        from scipy.stats import norm

        z_score = norm.ppf(confidence_level)

        # Extract sensitivity vector
        position_ids = list(position_values.keys())
        x_vector = np.array([position_values[pos_id] for pos_id in position_ids])

        # Handle different sensitivity types
        if isinstance(list(sensitivities.values())[0], dict):
            # Multi-factor sensitivities (e.g., {'delta': 0.5, 'vega': 100})
            # Need to aggregate across factors
            factor_names = set()
            for sens in sensitivities.values():
                factor_names.update(sens.keys())

            component_vars = {}
            for factor in factor_names:
                # Factor-specific sensitivity vector
                s_vector = np.array(
                    [sensitivities[pos_id].get(factor, 0.0) for pos_id in position_ids]
                )

                # Component VaR for this factor
                factor_cov = (
                    covariance_matrix.loc[factor, factor]
                    if factor in covariance_matrix.index
                    else 0.0
                )
                marginal_contrib = s_vector * x_vector
                factor_component_var = (
                    z_score
                    * np.sqrt(
                        marginal_contrib.T @ covariance_matrix.values @ marginal_contrib
                    )
                    if factor_cov > 0
                    else 0.0
                )

                # Distribute factor component VaR proportionally
                if factor_component_var > 0:
                    total_marginal = np.sum(np.abs(marginal_contrib))
                    if total_marginal > 0:
                        for i, pos_id in enumerate(position_ids):
                            allocation = np.abs(marginal_contrib[i]) / total_marginal
                            component_vars[pos_id] = component_vars.get(pos_id, 0.0) + (
                                allocation * factor_component_var
                            )
        else:
            # Single-factor sensitivities (scalar per position)
            # Use Euler decomposition: Component VaR_i = ∂VaR/∂x_i * x_i

            # For single-factor parametric VaR:
            # P&L_i = sensitivity_i * factor_return
            # P&L_portfolio = sum(sensitivities) * factor_return
            # Var(P&L_i) = sensitivity_i^2 * factor_variance
            # Cov(P&L_i, P&L_portfolio) = sensitivity_i * sum(sensitivities) * factor_variance
            # Component VaR_i = Cov(P&L_i, P&L_portfolio) / Var(P&L_portfolio) * Portfolio VaR

            component_vars = {}

            # Extract sensitivities and values
            s_vector = np.array([sensitivities[pos_id] for pos_id in position_ids])
            x_vector = np.array([position_values[pos_id] for pos_id in position_ids])

            # Calculate total sensitivity (for portfolio P&L)
            total_sensitivity = np.sum(s_vector)

            # Calculate portfolio variance
            primary_factor_var = covariance_matrix.iloc[0, 0] if len(covariance_matrix) > 0 else 1.0
            portfolio_variance = (total_sensitivity ** 2) * primary_factor_var

            # Calculate portfolio VaR
            portfolio_std = np.sqrt(portfolio_variance)
            portfolio_var = z_score * portfolio_std

            # Calculate Component VaR using Euler decomposition
            if portfolio_variance > 0 and abs(total_sensitivity) > 0:
                for i, pos_id in enumerate(position_ids):
                    # Component VaR_i = Cov(P&L_i, P&L_portfolio) / Var(P&L_portfolio) * Portfolio VaR
                    # Cov(P&L_i, P&L_portfolio) = sensitivity_i * total_sensitivity * factor_variance
                    covariance = s_vector[i] * total_sensitivity * primary_factor_var

                    # Allocation ratio
                    allocation_ratio = covariance / portfolio_variance

                    # Component VaR
                    component_var = abs(allocation_ratio * portfolio_var)
                    component_vars[pos_id] = component_var
            else:
                # No variance, distribute equally
                for pos_id in position_ids:
                    component_vars[pos_id] = 0.0

        return component_vars

    @staticmethod
    def calculate_from_delta_gamma(
        position_values: Dict[str, float],
        deltas: Dict[str, float],
        gammas: Dict[str, float],
        spot_volatility: float,
        confidence_level: float = 0.99,
    ) -> Dict[str, float]:
        """
        Calculate component VaR using delta-gamma approximation.

        Uses quadratic approximation for better accuracy with options.

        Args:
            position_values: Position market values by ID
            deltas: Position delta sensitivities
            gammas: Position gamma sensitivities
            spot_volatility: Spot price volatility
            confidence_level: VaR confidence level

        Returns:
            Dictionary mapping position ID to component VaR
        """
        from scipy.stats import norm

        z_score = norm.ppf(confidence_level)

        # Simplified delta-gamma component VaR
        # In practice, this requires more sophisticated calculation
        component_vars = {}

        for pos_id, value in position_values.items():
            delta = deltas.get(pos_id, 0.0)
            gamma = gammas.get(pos_id, 0.0)

            # Delta component (first order)
            delta_var = abs(delta * value * spot_volatility * z_score)

            # Gamma component (second order) - simplified
            # Full implementation would useCornish-Fisher expansion
            gamma_var = 0.5 * abs(gamma) * (spot_volatility * value) ** 2 * z_score

            component_vars[pos_id] = delta_var + gamma_var

        return component_vars


class MarginalVaRCalculator:
    """
    Calculates marginal VaR for position contributions.

    Marginal VaR measures the change in portfolio VaR from a small
    change in a position.
    """

    @staticmethod
    def calculate_incremental(
        portfolio_var: float,
        portfolio_value: float,
        position_value: float,
        position_var: float,
    ) -> float:
        """
        Calculate marginal VaR using incremental method.

        Marginal VaR = (VaR_with_position - VaR_without_position) / position_value

        Args:
            portfolio_var: VaR of full portfolio
            portfolio_value: Total portfolio market value
            position_value: Market value of the position
            position_var: Standalone VaR of the position

        Returns:
            Marginal VaR contribution

        Raises:
            ValidationError: If inputs are invalid
        """
        if position_value == 0:
            return 0.0

        # Simplified calculation: marginal VaR is approximately
        # the proportional contribution based on standalone risk
        position_weight = position_value / portfolio_value
        standalone_var = position_var

        # Marginal VaR is the increase in portfolio VaR from adding position
        marginal_var = (
            portfolio_var * position_weight * (standalone_var / portfolio_var)
            if portfolio_var > 0
            else 0.0
        )

        return marginal_var

    @staticmethod
    def calculate_from_sensitivity(
        position_value: float,
        sensitivity: float,
        portfolio_volatility: float,
        correlation: float = 1.0,
    ) -> float:
        """
        Calculate marginal VaR from position sensitivity.

        Args:
            position_value: Market value of position
            sensitivity: Position sensitivity (delta, DV01, etc.)
            portfolio_volatility: Portfolio return volatility
            correlation: Correlation between position and portfolio

        Returns:
            Marginal VaR
        """
        # Marginal VaR = sensitivity * position_value * portfolio_volatility
        return (
            abs(sensitivity) * abs(position_value) * portfolio_volatility * correlation
        )


class VaRAttributor:
    """
    High-level class for VaR attribution calculations.

    Combines component VaR and marginal VaR calculations.
    """

    def __init__(self, config: Optional[VaRConfig] = None):
        """
        Initialize VaR attributor.

        Args:
            config: VaR configuration (uses default if None)
        """
        self.config = config if config is not None else VaRConfig()
        self.component_calculator = ComponentVaRCalculator()
        self.marginal_calculator = MarginalVaRCalculator()

    def attribute_var(
        self,
        var_result: VaRResult,
        portfolio: Any,
        risk_factor_data: Dict[str, pd.Series],
    ) -> VaRResult:
        """
        Perform complete VaR attribution on a portfolio.

        Args:
            var_result: VaR result to enrich with attribution
            portfolio: Portfolio object
            risk_factor_data: Historical risk factor data

        Returns:
            VaRResult with attribution filled in

        Raises:
            ValidationError: If attribution fails
        """
        try:
            # Extract position data
            position_values = portfolio.get_position_values()
            position_ids = list(position_values.keys())

            if not position_ids:
                return var_result

            # Calculate component VaR
            component_var = self._calculate_component_var(
                position_values, portfolio, risk_factor_data
            )

            # Calculate marginal VaR
            marginal_var = self._calculate_marginal_var(
                position_values, component_var, var_result.var
            )

            # Update VaRResult
            var_result.component_var = component_var
            var_result.marginal_var = marginal_var

            # Calculate factor attribution
            var_result.factor_var = self._calculate_factor_attribution(
                risk_factor_data, var_result.var
            )

            return var_result

        except Exception as e:
            raise ValidationError(f"VaR attribution failed: {str(e)}")

    def _calculate_component_var(
        self,
        position_values: Dict[str, float],
        portfolio: Any,
        risk_factor_data: Dict[str, pd.Series],
    ) -> Dict[str, float]:
        """Calculate component VaR for positions."""
        # This would integrate with the portfolio to get sensitivities
        # For now, return empty dict - implementation depends on portfolio type
        return {}

    def _calculate_marginal_var(
        self,
        position_values: Dict[str, float],
        component_var: Dict[str, float],
        total_var: float,
    ) -> Dict[str, float]:
        """Calculate marginal VaR for positions."""
        if not component_var or total_var == 0:
            return {pos_id: 0.0 for pos_id in position_values.keys()}

        # Marginal VaR as proportion of component VaR
        marginal_var = {}
        for pos_id in position_values.keys():
            comp_var = component_var.get(pos_id, 0.0)
            # Simplified: marginal VaR ≈ component VaR / position weight
            marginal_var[pos_id] = comp_var

        return marginal_var

    def _calculate_factor_attribution(
        self,
        risk_factor_data: Dict[str, pd.Series],
        total_var: float,
    ) -> Dict[str, float]:
        """Calculate VaR attribution by risk factor."""
        factor_var = {}

        if not risk_factor_data:
            return factor_var

        # Calculate variance contribution of each factor
        for factor_name, factor_series in risk_factor_data.items():
            factor_vol = factor_series.std()
            factor_var[factor_name] = total_var * factor_vol

        # Normalize to total VaR
        total_attributed = sum(factor_var.values())
        if total_attributed > 0:
            for factor in factor_var:
                factor_var[factor] = (factor_var[factor] / total_attributed) * total_var

        return factor_var
