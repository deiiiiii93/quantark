"""
Stress application logic for modifying pricing environments.
"""

import re
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from quantark.priceenv import PricingEnvironment
from quantark.param import (
    SpotQuote,
    VolatilitySurface,
    RateCurve,
    DividendYield,
    FlatVolSurface,
    TermStructureVolSurface,
    BasisYield,
    FlatBasisYield,
    TermStructureBasisYield,
)
from quantark.param.rrf.rate_curve import FlatRateCurve, InterpolatedRateCurve, LinearRateCurve
from quantark.portfolio import Portfolio, Position
from quantark.stresstest.stress.stress_types import (
    StressType,
    StressLevel,
    BasisDividendRelationshipMode,
)
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close

if TYPE_CHECKING:
    from quantark.stresstest.scenario.scenario import Scenario, Stress


class StressApplicator:
    _parameter_adapters: Dict[str, Callable[[PricingEnvironment, "Stress"], None]] = {}

    @classmethod
    def register_adapter(
        cls,
        parameter: str,
        handler: Callable[[PricingEnvironment, "Stress"], None],
    ) -> None:
        """
        Register a handler for a specific stress parameter.
        """
        cls._parameter_adapters[parameter.lower()] = handler

    @classmethod
    def unregister_adapter(cls, parameter: str) -> None:
        cls._parameter_adapters.pop(parameter.lower(), None)

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
        portfolio: Portfolio, scenario: "Scenario"
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
            basis_yield=deepcopy(env.basis_yield) if env.basis_yield else None,
            day_count_convention=env.day_count_convention,
            bus_days_in_year=env.bus_days_in_year,
        )

    @staticmethod
    def _apply_stress_to_env(env: PricingEnvironment, stress: "Stress") -> None:
        """
        Apply a single stress to a pricing environment (in-place).

        Args:
            env: Pricing environment to modify
            stress: Stress to apply

        Raises:
            ValidationError: If parameter is not found or stress cannot be applied
        """
        param = stress.parameter.lower()
        adapter = StressApplicator._parameter_adapters.get(param)
        if adapter:
            adapter(env, stress)
            return
        raise ValidationError(
            f"Unknown parameter to stress: {stress.parameter}. "
            "Register a custom adapter via StressApplicator.register_adapter()."
        )

    @staticmethod
    def _stress_spot(env: PricingEnvironment, stress: "Stress") -> None:
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
    def _stress_volatility(env: PricingEnvironment, stress: "Stress") -> None:
        """Stress volatility surface."""
        if env.vol_surface is None:
            raise ValidationError(
                "Cannot stress volatility: no vol surface in environment"
            )

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
            return

        if isinstance(env.vol_surface, TermStructureVolSurface):
            new_vols = [
                stress.stress_type.apply(float(v), stress.stress_value)
                for v in env.vol_surface.vols
            ]
            if any(v <= 0 for v in new_vols):
                raise ValidationError(
                    "Stressed term-structure vol must be positive for all tenors."
                )
            env.vol_surface = TermStructureVolSurface(
                times=list(env.vol_surface.times), vols=new_vols
            )
            return

        # For complex vol surfaces, would need more sophisticated handling
        raise ValidationError(
            f"Stressing non-flat volatility surfaces not yet supported. "
            f"Surface type: {type(env.vol_surface).__name__}"
        )

    @staticmethod
    def _stress_rate(env: PricingEnvironment, stress: "Stress") -> None:
        """Stress interest rate curve."""
        if env.rate_curve is None:
            raise ValidationError("Cannot stress rate: no rate curve in environment")

        # Apply parallel shift to rate curve
        # Get the current rate (use 1-year as reference)
        current_rate = env.rate_curve.get_rate(1.0)
        new_rate = stress.stress_type.apply(current_rate, stress.stress_value)

        # Apply parallel shift to the curve
        shift = new_rate - current_rate

        curve = env.rate_curve

        if isinstance(curve, FlatRateCurve):
            env.rate_curve = FlatRateCurve(rate=new_rate)
        elif isinstance(curve, InterpolatedRateCurve):
            delta = new_rate - current_rate
            updated = [(time, rate + delta) for time, rate in curve.pillars]
            env.rate_curve = curve.__class__(updated)
        else:
            raise ValidationError(
                f"Cannot stress curve type {type(curve).__name__}; "
                "only FlatRateCurve or interpolated curves are supported."
            )

    @staticmethod
    def _stress_key_rate(env: PricingEnvironment, stress: "Stress") -> None:
        curve = env.rate_curve
        bucket = StressApplicator._tenor_to_years(stress.metadata.get("tenor_bucket"))
        if bucket is None:
            raise ValidationError(
                "Key rate stress requires tenor_bucket metadata (e.g., '5Y')."
            )

        if isinstance(curve, InterpolatedRateCurve):
            new_pillars = []
            found = False
            for time, rate in curve.pillars:
                if is_close(time, bucket, rel_tol=1e-6):
                    found = True
                    new_rate = stress.stress_type.apply(rate, stress.stress_value)
                    new_pillars.append((time, new_rate))
                else:
                    new_pillars.append((time, rate))
            if not found:
                base = curve.get_rate(bucket)
                new_rate = stress.stress_type.apply(base, stress.stress_value)
                new_pillars.append((bucket, new_rate))
                new_pillars.sort(key=lambda x: x[0])
            env.rate_curve = curve.__class__(new_pillars)
        else:
            # Fallback to parallel shift for flat curves
            StressApplicator._stress_rate(env, stress)

    @staticmethod
    def _stress_spread(env: PricingEnvironment, stress: "Stress") -> None:
        StressApplicator._stress_rate(env, stress)

    @staticmethod
    def _stress_dividend(env: PricingEnvironment, stress: "Stress") -> None:
        """Stress dividend yield."""
        # If no dividend yield, create one with zero base
        from quantark.param.div.dividend_yield import (
            ContinuousDividendYield,
            TermStructureDividendYield,
        )
        from quantark.param.basis.basis_yield import (
            FlatBasisYield,
            calculate_basis_from_rate_dividend,
        )

        time_to_maturity = 1.0
        if stress.metadata and "time_to_maturity" in stress.metadata:
            time_to_maturity = float(stress.metadata.get("time_to_maturity", 1.0))
        if time_to_maturity <= 0:
            raise ValidationError(
                f"time_to_maturity must be positive, got {time_to_maturity}"
            )

        if isinstance(env.div_yield, TermStructureDividendYield):
            new_yields = [
                stress.stress_type.apply(float(y), stress.stress_value)
                for y in env.div_yield.yields
            ]
            new_yields = [max(0.0, y) for y in new_yields]
            env.div_yield = TermStructureDividendYield(
                times=list(env.div_yield.times), yields=new_yields
            )
        else:
            current_yield = 0.0
            if env.div_yield is not None:
                if isinstance(env.div_yield, ContinuousDividendYield):
                    current_yield = env.div_yield.div_yield
                else:
                    # For complex dividend structures, get representative rate
                    current_yield = env.div_yield.get_yield(time_to_maturity)

            new_yield = stress.stress_type.apply(current_yield, stress.stress_value)

            if new_yield < 0:
                new_yield = 0.0

            # Update dividend yield
            env.div_yield = ContinuousDividendYield(div_yield=new_yield)

        # Handle relationship modes where dividend stress drives basis
        relationship_mode = BasisDividendRelationshipMode.INDEPENDENT
        if stress.metadata:
            mode_str = stress.metadata.get("relationship_mode")
            if mode_str:
                try:
                    relationship_mode = BasisDividendRelationshipMode.from_string(mode_str)
                except ValueError:
                    relationship_mode = BasisDividendRelationshipMode.INDEPENDENT

        if relationship_mode in {
            BasisDividendRelationshipMode.AUTO_ADJUST_BASIS,
            BasisDividendRelationshipMode.SYNCHRONIZED,
        }:
            rate = env.rate_curve.get_rate(time_to_maturity)
            div_for_basis = env.div_yield.get_yield(time_to_maturity)
            if relationship_mode == BasisDividendRelationshipMode.AUTO_ADJUST_BASIS:
                new_basis = calculate_basis_from_rate_dividend(rate, div_for_basis)
            else:
                # SYNCHRONIZED: apply the same stress to basis
                current_basis = (
                    env.basis_yield.get_basis_yield(time_to_maturity)
                    if env.basis_yield is not None
                    else 0.0
                )
                new_basis = stress.stress_type.apply(current_basis, stress.stress_value)

            if isinstance(env.basis_yield, TermStructureBasisYield):
                current_basis = env.basis_yield.get_basis_yield(time_to_maturity)
                shift = new_basis - current_basis
                new_yields = [float(y) + shift for y in env.basis_yield.yields]
                if any(abs(y) > 0.50 for y in new_yields):
                    raise ValidationError(
                        "Stressed term-structure basis yields must be within +/-50%."
                    )
                env.basis_yield = TermStructureBasisYield(
                    times=list(env.basis_yield.times), yields=new_yields
                )
            else:
                env.basis_yield = FlatBasisYield(basis_yield=new_basis)

    @staticmethod
    def _stress_basis(env: PricingEnvironment, stress: "Stress") -> None:
        """
        Stress basis yield for futures contracts.

        This handles basis risk factor stress testing. When basis changes,
        it can automatically adjust the dividend yield (assuming rate stays constant)
        based on the configured relationship mode.

        For independent mode, basis is stressed directly.
        For auto-adjust modes, relationships between basis, dividend yield, and rate are maintained.
        """
        from quantark.param.basis.basis_yield import (
            FlatBasisYield,
            calculate_dividend_from_rate_basis,
        )
        from quantark.param.div.dividend_yield import (
            ContinuousDividendYield,
            TermStructureDividendYield,
        )

        # Check for configuration metadata about relationship mode
        relationship_mode = BasisDividendRelationshipMode.INDEPENDENT
        if stress.metadata:
            mode_str = stress.metadata.get("relationship_mode")
            if mode_str:
                try:
                    relationship_mode = BasisDividendRelationshipMode.from_string(mode_str)
                except ValueError:
                    relationship_mode = BasisDividendRelationshipMode.INDEPENDENT

        # Determine time to maturity for annualization
        # Check if T is provided in metadata (e.g., from a specific futures contract)
        time_to_maturity = 1.0
        if stress.metadata and "time_to_maturity" in stress.metadata:
            time_to_maturity = float(stress.metadata.get("time_to_maturity", 1.0))
        if time_to_maturity <= 0:
            raise ValidationError(
                f"time_to_maturity must be positive, got {time_to_maturity}"
            )

        # Get current basis yield - either existing or zero
        current_basis = 0.0
        if env.basis_yield is not None:
            current_basis = env.basis_yield.get_basis_yield(time_to_maturity)

        # Apply stress to basis yield
        new_basis = stress.stress_type.apply(current_basis, stress.stress_value)

        # Check if stress would create unrealistic scenarios
        # Update basis yield
        if isinstance(env.basis_yield, TermStructureBasisYield):
            new_yields = [
                stress.stress_type.apply(float(y), stress.stress_value)
                for y in env.basis_yield.yields
            ]
            if any(abs(y) > 0.50 for y in new_yields):
                raise ValidationError(
                    "Stressed term-structure basis yields must be within +/-50%."
                )
            env.basis_yield = TermStructureBasisYield(
                times=list(env.basis_yield.times), yields=new_yields
            )
        else:
            env.basis_yield = FlatBasisYield(basis_yield=new_basis)

        # Handle relationship modes
        if relationship_mode == BasisDividendRelationshipMode.AUTO_ADJUST_DIVIDEND:
            # When basis changes, adjust dividend yield to maintain r - q = b
            # Assuming rate stays constant: q_new = r - b_new
            rate = env.rate_curve.get_rate(time_to_maturity)
            new_div_yield = calculate_dividend_from_rate_basis(rate, new_basis)
            if isinstance(env.div_yield, TermStructureDividendYield):
                current_div = env.div_yield.get_yield(time_to_maturity)
                shift = new_div_yield - current_div
                new_yields = [max(0.0, float(y) + shift) for y in env.div_yield.yields]
                env.div_yield = TermStructureDividendYield(
                    times=list(env.div_yield.times), yields=new_yields
                )
            else:
                if new_div_yield < 0:
                    new_div_yield = 0.0
                env.div_yield = ContinuousDividendYield(div_yield=new_div_yield)

        elif relationship_mode == BasisDividendRelationshipMode.SYNCHRONIZED:
            # Apply the same stress to dividend yield
            current_div = (
                env.div_yield.get_yield(time_to_maturity)
                if env.div_yield is not None
                else 0.0
            )
            new_div_yield = stress.stress_type.apply(current_div, stress.stress_value)
            if isinstance(env.div_yield, TermStructureDividendYield):
                shift = new_div_yield - current_div
                new_yields = [max(0.0, float(y) + shift) for y in env.div_yield.yields]
                env.div_yield = TermStructureDividendYield(
                    times=list(env.div_yield.times), yields=new_yields
                )
            else:
                if new_div_yield < 0:
                    new_div_yield = 0.0
                env.div_yield = ContinuousDividendYield(div_yield=new_div_yield)

    @staticmethod
    def get_stress_summary(
        original_env: PricingEnvironment, stressed_env: PricingEnvironment
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
            if isinstance(original_env.vol_surface, FlatVolSurface) and isinstance(
                stressed_env.vol_surface, FlatVolSurface
            ):
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

        # Basis yield
        if original_env.basis_yield and stressed_env.basis_yield:
            orig_basis = original_env.basis_yield.get_basis_yield(1.0)
            new_basis = stressed_env.basis_yield.get_basis_yield(1.0)
            summary["basis_yield"] = {
                "original": orig_basis,
                "stressed": new_basis,
                "change_abs": new_basis - orig_basis,
                "change_bps": (new_basis - orig_basis) * 10000,
            }

        return summary

    @staticmethod
    def _tenor_to_years(bucket: Optional[str]) -> Optional[float]:
        if bucket is None:
            return None
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)([YyMm])\s*", bucket)
        if not match:
            raise ValidationError(
                f"Invalid tenor bucket '{bucket}'. Use formats like '2Y' or '6M'."
            )
        value = float(match.group(1))
        unit = match.group(2).upper()
        if unit == "Y":
            return value
        if unit == "M":
            return value / 12.0
        return None


# Register default adapters for equity workflows.
StressApplicator.register_adapter("spot", StressApplicator._stress_spot)
StressApplicator.register_adapter("volatility", StressApplicator._stress_volatility)
StressApplicator.register_adapter("vol", StressApplicator._stress_volatility)
StressApplicator.register_adapter("rate", StressApplicator._stress_rate)
StressApplicator.register_adapter("key_rate", StressApplicator._stress_key_rate)
StressApplicator.register_adapter("spread", StressApplicator._stress_spread)
StressApplicator.register_adapter("dividend_yield", StressApplicator._stress_dividend)
StressApplicator.register_adapter("div_yield", StressApplicator._stress_dividend)
StressApplicator.register_adapter("dividend", StressApplicator._stress_dividend)
StressApplicator.register_adapter("basis", StressApplicator._stress_basis)
