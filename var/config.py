"""
Configuration classes for Value-at-Risk (VaR) calculations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import List, Optional

from util.exceptions import ValidationError


class VaRMethod(Enum):
    """VaR calculation methods."""
    
    PARAMETRIC = auto()
    HISTORICAL = auto()
    MONTE_CARLO = auto()
    
    def __str__(self):
        return self.name.replace("_", " ").title()


@dataclass
class EquityRiskFactorConfig:
    """Configuration for equity risk factors."""
    
    include_spot: bool = True
    include_vol: bool = True
    include_rate: bool = True
    include_div_yield: bool = False


@dataclass
class FIRiskFactorConfig:
    """Configuration for fixed income risk factors."""
    
    include_parallel_shift: bool = True
    include_key_rates: bool = False
    key_rate_tenors: List[float] = field(default_factory=lambda: [2.0, 5.0, 10.0, 30.0])


@dataclass
class VaRConfig:
    """Configuration for VaR calculations."""
    
    confidence_level: float = 0.99
    holding_period: int = 1
    lookback_days: int = 252
    var_method: VaRMethod = VaRMethod.PARAMETRIC
    
    equity_factors: Optional[EquityRiskFactorConfig] = None
    fi_factors: Optional[FIRiskFactorConfig] = None
    
    scaling_method: str = "sqrt_t"
    
    mc_num_simulations: int = 10000
    mc_seed: Optional[int] = None
    
    calculate_component_var: bool = True
    calculate_marginal_var: bool = True
    calculate_factor_var: bool = True
    calculate_incremental_var: bool = False
    
    calculate_stressed_var: bool = False
    stressed_period_start: Optional[datetime] = None
    stressed_period_end: Optional[datetime] = None
    stressed_lookback_days: int = 252
    
    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate()
    
    def _validate(self):
        """Validate configuration values."""
        if not (0.0 < self.confidence_level < 1.0):
            raise ValidationError(
                f"confidence_level must be between 0 and 1, got {self.confidence_level}"
            )
        
        if self.holding_period < 1:
            raise ValidationError(
                f"holding_period must be >= 1, got {self.holding_period}"
            )
        
        if self.lookback_days < 1:
            raise ValidationError(
                f"lookback_days must be >= 1, got {self.lookback_days}"
            )
        
        if self.scaling_method not in ["sqrt_t", "overlapping"]:
            raise ValidationError(
                f"scaling_method must be 'sqrt_t' or 'overlapping', got {self.scaling_method}"
            )
        
        if self.mc_num_simulations < 100:
            raise ValidationError(
                f"mc_num_simulations must be >= 100, got {self.mc_num_simulations}"
            )
        
        if self.calculate_stressed_var:
            if self.stressed_period_start is not None and self.stressed_period_end is not None:
                if self.stressed_period_start >= self.stressed_period_end:
                    raise ValidationError(
                        "stressed_period_start must be before stressed_period_end"
                    )
