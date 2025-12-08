"""
Add-On Calculator for SIMM.

This module implements SIMM add-on calculations:
- Fixed add-on amounts
- Notional-based add-on factors
- Multiplicative scales (MS) per product class
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from simm.taxonomy import ProductClass
from simm.config import SIMMConfig


@dataclass
class AddOnResult:
    """Result of add-on calculation.
    
    Attributes:
        fixed_addon: Fixed add-on amount.
        factor_addon: Notional-based add-on amount.
        total_addon: Total add-on (fixed + factor).
        ms_adjustments: Per-product-class multiplicative scale adjustments.
    """
    fixed_addon: float = 0.0
    factor_addon: float = 0.0
    total_addon: float = 0.0
    ms_adjustments: Dict[ProductClass, float] = field(default_factory=dict)


class AddOnCalculator:
    """Calculator for SIMM add-ons.
    
    Add-ons are additional margin amounts that can be applied
    on top of the base SIMM calculation.
    """
    
    def __init__(self, config: SIMMConfig):
        """Initialize with configuration.
        
        Args:
            config: SIMM configuration with add-on settings.
        """
        self.config = config
    
    def calculate(
        self,
        notionals: Optional[Dict[str, float]] = None,
        product_class_margins: Optional[Dict[ProductClass, float]] = None,
    ) -> AddOnResult:
        """Calculate all add-ons.
        
        Args:
            notionals: Dict mapping identifier to notional amount for factor-based add-ons.
            product_class_margins: Dict mapping product class to margin (before MS).
            
        Returns:
            AddOnResult with all add-on components.
        """
        # Fixed add-on
        fixed = self.config.addon_fixed
        
        # Factor-based add-on (sum of factor × notional)
        factor_addon = 0.0
        if notionals and self.config.addon_factors:
            for key, factor in self.config.addon_factors.items():
                if key in notionals:
                    factor_addon += factor * notionals[key]
        
        total = fixed + factor_addon
        
        # Multiplicative scale adjustments (difference from base margin)
        ms_adjustments = {}
        if product_class_margins:
            ms_adjustments = self._calculate_ms_adjustments(product_class_margins)
        
        return AddOnResult(
            fixed_addon=fixed,
            factor_addon=factor_addon,
            total_addon=total,
            ms_adjustments=ms_adjustments,
        )
    
    def _calculate_ms_adjustments(
        self,
        product_class_margins: Dict[ProductClass, float],
    ) -> Dict[ProductClass, float]:
        """Calculate multiplicative scale adjustments.
        
        MS adjustment = (MS - 1) × base_margin
        
        Args:
            product_class_margins: Dict mapping product class to base margin.
            
        Returns:
            Dict mapping product class to MS adjustment amount.
        """
        adjustments = {}
        
        for pc, margin in product_class_margins.items():
            ms = self._get_multiplier(pc)
            if ms != 1.0:
                adjustments[pc] = (ms - 1) * margin
        
        return adjustments
    
    def _get_multiplier(self, product_class: ProductClass) -> float:
        """Get multiplicative scale for a product class.
        
        Args:
            product_class: The product class.
            
        Returns:
            The multiplicative scale (MS).
        """
        mapping = {
            ProductClass.RATES_FX: self.config.ms_rates_fx,
            ProductClass.CREDIT: self.config.ms_credit,
            ProductClass.EQUITY: self.config.ms_equity,
            ProductClass.COMMODITY: self.config.ms_commodity,
        }
        return mapping.get(product_class, 1.0)
    
    def apply_multipliers(
        self,
        product_class_margins: Dict[ProductClass, float],
    ) -> Dict[ProductClass, float]:
        """Apply multiplicative scales to product class margins.
        
        Args:
            product_class_margins: Dict mapping product class to base margin.
            
        Returns:
            Dict mapping product class to margin after MS.
        """
        return {
            pc: margin * self._get_multiplier(pc)
            for pc, margin in product_class_margins.items()
        }
