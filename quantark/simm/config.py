"""
SIMM Configuration Module.

This module provides configuration dataclasses for ISDA SIMM calculations.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from util.exceptions import ValidationError


class SIMMVersion(Enum):
    """ISDA SIMM version identifiers.
    
    Different SIMM versions may have different risk weights and correlations.
    """
    V2_5 = "2.5"
    V2_6 = "2.6"
    
    def __str__(self) -> str:
        return self.value


@dataclass
class SIMMConfig:
    """Configuration for SIMM calculation.
    
    This class encapsulates all configuration parameters for SIMM calculations,
    including version, calculation currency, component selection, and add-ons.
    
    Attributes:
        version: SIMM version (affects risk weights and correlations).
        calculation_currency: Base currency for SIMM calculation (typically USD).
        
        calculate_delta: Whether to calculate Delta margin.
        calculate_vega: Whether to calculate Vega margin.
        calculate_curvature: Whether to calculate Curvature margin.
        calculate_base_corr: Whether to calculate Base Correlation margin.
        
        ms_rates_fx: Product class multiplier for RatesFX (default 1.0).
        ms_credit: Product class multiplier for Credit (default 1.0).
        ms_equity: Product class multiplier for Equity (default 1.0).
        ms_commodity: Product class multiplier for Commodity (default 1.0).
        
        addon_fixed: Fixed add-on amount in calculation currency.
        addon_factors: Per-trade or per-product add-on factors.
        
        include_attribution: Include margin attribution in results.
        include_bucket_detail: Include bucket-level detail in results.
    
    Examples:
        Basic SIMM configuration:
        >>> config = SIMMConfig()
        
        SIMM with specific version and currency:
        >>> config = SIMMConfig(
        ...     version=SIMMVersion.V2_6,
        ...     calculation_currency="EUR"
        ... )
        
        SIMM with add-ons:
        >>> config = SIMMConfig(
        ...     addon_fixed=1_000_000,
        ...     addon_factors={"regulatory_addon": 0.05}
        ... )
        
        Delta-only SIMM:
        >>> config = SIMMConfig(
        ...     calculate_delta=True,
        ...     calculate_vega=False,
        ...     calculate_curvature=False,
        ...     calculate_base_corr=False
        ... )
    """
    # SIMM version
    version: SIMMVersion = SIMMVersion.V2_6
    
    # Calculation currency
    calculation_currency: str = "USD"
    
    # Which components to calculate
    calculate_delta: bool = True
    calculate_vega: bool = True
    calculate_curvature: bool = True
    calculate_base_corr: bool = True
    
    # Product class multipliers (default = 1.0)
    ms_rates_fx: float = 1.0
    ms_credit: float = 1.0
    ms_equity: float = 1.0
    ms_commodity: float = 1.0
    
    # Add-on configuration
    addon_fixed: float = 0.0
    addon_factors: Dict[str, float] = field(default_factory=dict)
    
    # Output options
    include_attribution: bool = True
    include_bucket_detail: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate configuration values."""
        # Validate calculation currency
        if not isinstance(self.calculation_currency, str) or len(self.calculation_currency) != 3:
            raise ValidationError(
                f"calculation_currency must be a 3-letter ISO currency code, "
                f"got {self.calculation_currency!r}"
            )
        
        # Validate multipliers are positive
        for name, value in [
            ("ms_rates_fx", self.ms_rates_fx),
            ("ms_credit", self.ms_credit),
            ("ms_equity", self.ms_equity),
            ("ms_commodity", self.ms_commodity),
        ]:
            if value <= 0:
                raise ValidationError(
                    f"{name} must be positive, got {value}"
                )
        
        # Validate addon_fixed is non-negative
        if self.addon_fixed < 0:
            raise ValidationError(
                f"addon_fixed must be non-negative, got {self.addon_fixed}"
            )
        
        # Validate addon_factors values are non-negative
        for key, value in self.addon_factors.items():
            if value < 0:
                raise ValidationError(
                    f"addon_factors['{key}'] must be non-negative, got {value}"
                )
    
    def get_product_class_multiplier(self, product_class: str) -> float:
        """Get the multiplier for a product class.
        
        Args:
            product_class: Product class name ("RatesFX", "Credit", "Equity", "Commodity").
            
        Returns:
            The multiplier for the specified product class.
            
        Raises:
            ValueError: If product_class is not recognized.
        """
        mapping = {
            "RatesFX": self.ms_rates_fx,
            "Credit": self.ms_credit,
            "Equity": self.ms_equity,
            "Commodity": self.ms_commodity,
        }
        if product_class not in mapping:
            raise ValueError(f"Unknown product class: {product_class}")
        return mapping[product_class]
    
    def with_version(self, version: SIMMVersion) -> "SIMMConfig":
        """Create a copy of this config with a different version.
        
        Args:
            version: The SIMM version to use.
            
        Returns:
            New SIMMConfig with the specified version.
        """
        return SIMMConfig(
            version=version,
            calculation_currency=self.calculation_currency,
            calculate_delta=self.calculate_delta,
            calculate_vega=self.calculate_vega,
            calculate_curvature=self.calculate_curvature,
            calculate_base_corr=self.calculate_base_corr,
            ms_rates_fx=self.ms_rates_fx,
            ms_credit=self.ms_credit,
            ms_equity=self.ms_equity,
            ms_commodity=self.ms_commodity,
            addon_fixed=self.addon_fixed,
            addon_factors=dict(self.addon_factors),
            include_attribution=self.include_attribution,
            include_bucket_detail=self.include_bucket_detail,
        )
