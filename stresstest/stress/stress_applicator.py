"""
Stress application logic for modifying pricing environments.
"""

from typing import Dict, Optional, Any
from copy import deepcopy
from priceenv import PricingEnvironment
from param import SpotQuote, VolatilitySurface, RateCurve, DividendYield, FlatVolSurface
from portfolio import Portfolio, Position
from stresstest.scenario.scenario import Scenario, Stress
from stresstest.stress.stress_types import StressType, StressLevel
from util.exceptions import ValidationError


class StressApplicator:
    """
    Applies stresses to pricing environments.
    
    This class handles the logic of taking a scenario and applying its stresses
    to portfolio pricing environments at different levels (portfolio, underlying, position).
    
    Key responsibilities:
    - Clone pricing environments for stress scenarios
    - Apply parameter stresses based on stress type
    - Handle portfolio/underlying/position level targeting
    """
    
    @staticmethod
    def apply_scenario_to_portfolio(
        portfolio: Portfolio,
        scenario: Scenario
    ) -> Dict[str, PricingEnvironment]:
        """
        Apply a scenario to portfolio and return stressed pricing environments.
        
        Creates a new set of pricing environments with stresses applied according
        to the scenario definition. Handles portfolio, underlying, and position-level
        stresses appropriately.
        
        Args:
            portfolio: Portfolio to stress
            scenario: Scenario defining the stresses
            
        Returns:
            Dictionary of stressed pricing environments by underlying
            
        Raises:
            ValidationError: If stress targets are invalid
        """
        # Clone all pricing environments
        stressed_envs = {}
        for underlying, env in portfolio.pricing_environments.items():
            stressed_envs[underlying] = StressApplicator._clone_pricing_env(env)
        
        # Apply stresses by level
        for stress in scenario.stresses:
            if stress.level == StressLevel.PORTFOLIO:
                # Apply to all underlyings
                for underlying in stressed_envs:
                    StressApplicator._apply_stress_to_env(
                        stressed_envs[underlying], stress
                    )
            
            elif stress.level == StressLevel.UNDERLYING:
                # Apply to specific underlying
                if stress.target not in stressed_envs:
                    raise ValidationError(
                        f"Stress target underlying '{stress.target}' not found in portfolio"
                    )
                StressApplicator._apply_stress_to_env(
                    stressed_envs[stress.target], stress
                )
            
            elif stress.level == StressLevel.POSITION:
                # Find position and apply to its underlying
                position = portfolio.get_position(stress.target)
                if position is None:
                    raise ValidationError(
                        f"Stress target position '{stress.target}' not found in portfolio"
                    )
                # For position-level, we only stress that position's environment
                # This is handled by creating position-specific environments if needed
                StressApplicator._apply_stress_to_env(
                    stressed_envs[position.underlying], stress
                )
        
        return stressed_envs
    
    @staticmethod
    def _clone_pricing_env(env: PricingEnvironment) -> PricingEnvironment:
        """
        Create a deep copy of a pricing environment.
        
        Args:
            env: Original pricing environment
            
        Returns:
            Cloned pricing environment
        """
        # Clone each component
        return PricingEnvironment(
            rate_curve=deepcopy(env.rate_curve),
            valuation_date=env.valuation_date,
            spot_quote=deepcopy(env.spot_quote) if env.spot_quote else None,
            vol_surface=deepcopy(env.vol_surface) if env.vol_surface else None,
            div_yield=deepcopy(env.div_yield) if env.div_yield else None,
            day_count_convention=env.day_count_convention,
            bus_days_in_year=env.bus_days_in_year,
        )
    
    @staticmethod
    def _apply_stress_to_env(env: PricingEnvironment, stress: Stress) -> None:
        """
        Apply a single stress to a pricing environment (in-place).
        
        Args:
            env: Pricing environment to modify
            stress: Stress to apply
            
        Raises:
            ValidationError: If parameter is not found or stress cannot be applied
        """
        param = stress.parameter.lower()
        
        if param == "spot":
            StressApplicator._stress_spot(env, stress)
        elif param in ["volatility", "vol"]:
            StressApplicator._stress_volatility(env, stress)
        elif param == "rate":
            StressApplicator._stress_rate(env, stress)
        elif param in ["dividend_yield", "div_yield", "dividend"]:
            StressApplicator._stress_dividend(env, stress)
        else:
            raise ValidationError(f"Unknown parameter to stress: {stress.parameter}")
    
    @staticmethod
    def _stress_spot(env: PricingEnvironment, stress: Stress) -> None:
        """Stress spot price."""
        if env.spot_quote is None:
            raise ValidationError("Cannot stress spot: no spot quote in environment")
        
        current_spot = env.spot_quote.spot
        new_spot = stress.stress_type.apply(current_spot, stress.stress_value)
        
        if new_spot <= 0:
            raise ValidationError(
                f"Stressed spot price must be positive, got {new_spot} "
                f"(original: {current_spot}, stress: {stress.stress_value})"
            )
        
        # Update spot quote
        env.spot_quote = SpotQuote(
            spot=new_spot,
            timestamp=env.spot_quote.timestamp,
            asset_name=env.spot_quote.asset_name,
        )
    
    @staticmethod
    def _stress_volatility(env: PricingEnvironment, stress: Stress) -> None:
        """Stress volatility surface."""
        if env.vol_surface is None:
            raise ValidationError("Cannot stress volatility: no vol surface in environment")
        
        # For flat vol surface, stress the constant volatility
        if isinstance(env.vol_surface, FlatVolSurface):
            current_vol = env.vol_surface.volatility
            new_vol = stress.stress_type.apply(current_vol, stress.stress_value)
            
            if new_vol <= 0:
                raise ValidationError(
                    f"Stressed volatility must be positive, got {new_vol} "
                    f"(original: {current_vol}, stress: {stress.stress_value})"
                )
            
            env.vol_surface = FlatVolSurface(volatility=new_vol)
        else:
            # For complex vol surfaces, would need more sophisticated handling
            raise ValidationError(
                f"Stressing non-flat volatility surfaces not yet supported. "
                f"Surface type: {type(env.vol_surface).__name__}"
            )
    
    @staticmethod
    def _stress_rate(env: PricingEnvironment, stress: Stress) -> None:
        """Stress interest rate curve."""
        if env.rate_curve is None:
            raise ValidationError("Cannot stress rate: no rate curve in environment")
        
        # Apply parallel shift to rate curve
        # Get the current rate (use 1-year as reference)
        current_rate = env.rate_curve.get_rate(1.0)
        new_rate = stress.stress_type.apply(current_rate, stress.stress_value)
        
        # Apply parallel shift to the curve
        shift = new_rate - current_rate
        
        # Create a new shifted rate curve
        from param.rrf.rate_curve import FlatRateCurve
        if isinstance(env.rate_curve, FlatRateCurve):
            env.rate_curve = FlatRateCurve(rate=new_rate)
        else:
            # For non-flat curves, apply parallel shift
            # This would need access to the underlying curve structure
            # For now, raise an error for complex curves
            raise ValidationError(
                f"Stressing non-flat rate curves requires parallel shift implementation. "
                f"Curve type: {type(env.rate_curve).__name__}"
            )
    
    @staticmethod
    def _stress_dividend(env: PricingEnvironment, stress: Stress) -> None:
        """Stress dividend yield."""
        # If no dividend yield, create one with zero base
        from param.div.dividend_yield import ContinuousDividendYield
        
        if env.div_yield is None:
            current_yield = 0.0
        else:
            # Assume flat dividend yield
            if isinstance(env.div_yield, ContinuousDividendYield):
                current_yield = env.div_yield.div_yield
            else:
                # For complex dividend structures, get 1-year rate
                current_yield = env.div_yield.get_yield(1.0)
        
        new_yield = stress.stress_type.apply(current_yield, stress.stress_value)
        
        if new_yield < 0:
            raise ValidationError(
                f"Stressed dividend yield cannot be negative, got {new_yield} "
                f"(original: {current_yield}, stress: {stress.stress_value})"
            )
        
        # Update dividend yield
        env.div_yield = ContinuousDividendYield(div_yield=new_yield)
    
    @staticmethod
    def get_stress_summary(
        original_env: PricingEnvironment,
        stressed_env: PricingEnvironment
    ) -> Dict[str, Any]:
        """
        Generate summary of changes between original and stressed environments.
        
        Args:
            original_env: Original pricing environment
            stressed_env: Stressed pricing environment
            
        Returns:
            Dictionary with before/after values and changes
        """
        summary = {}
        
        # Spot
        if original_env.spot_quote and stressed_env.spot_quote:
            orig_spot = original_env.spot_quote.spot
            new_spot = stressed_env.spot_quote.spot
            summary["spot"] = {
                "original": orig_spot,
                "stressed": new_spot,
                "change_abs": new_spot - orig_spot,
                "change_pct": (new_spot / orig_spot - 1.0) * 100,
            }
        
        # Volatility
        if original_env.vol_surface and stressed_env.vol_surface:
            if isinstance(original_env.vol_surface, FlatVolSurface) and \
               isinstance(stressed_env.vol_surface, FlatVolSurface):
                orig_vol = original_env.vol_surface.volatility
                new_vol = stressed_env.vol_surface.volatility
                summary["volatility"] = {
                    "original": orig_vol,
                    "stressed": new_vol,
                    "change_abs": new_vol - orig_vol,
                    "change_pct": (new_vol / orig_vol - 1.0) * 100,
                }
        
        # Rate
        if original_env.rate_curve and stressed_env.rate_curve:
            orig_rate = original_env.rate_curve.get_rate(1.0)
            new_rate = stressed_env.rate_curve.get_rate(1.0)
            summary["rate"] = {
                "original": orig_rate,
                "stressed": new_rate,
                "change_abs": new_rate - orig_rate,
                "change_bps": (new_rate - orig_rate) * 10000,
            }
        
        # Dividend yield
        if original_env.div_yield and stressed_env.div_yield:
            orig_div = original_env.div_yield.get_yield(1.0)
            new_div = stressed_env.div_yield.get_yield(1.0)
            summary["dividend_yield"] = {
                "original": orig_div,
                "stressed": new_div,
                "change_abs": new_div - orig_div,
                "change_bps": (new_div - orig_div) * 10000,
            }
        
        return summary

