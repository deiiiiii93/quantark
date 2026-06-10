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
    """
    Configuration for Value-at-Risk (VaR) calculations.

    This class encapsulates all configuration parameters for VaR calculations,
    including core parameters (confidence level, holding period), risk factor
    configuration, attribution settings, and engine-specific options.

    Attributes:
        confidence_level: VaR confidence level (e.g., 0.99 for 99% VaR).
            Must be between 0 and 1. Typical values: 0.95 (internal), 0.99 (regulatory).

        holding_period: VaR holding period in days. Default is 1 for 1-day VaR.
            For multi-day VaR, can be 5, 10, etc. Note: longer holding periods
            require special scaling methods.

        lookback_days: Number of days of historical data to use for VaR calculation.
            Default is 252 (1 year of trading days). More data improves accuracy
            but increases calculation time.

        var_method: VaR calculation method (PARAMETRIC, HISTORICAL, MONTE_CARLO).
            Each method has different trade-offs in speed, accuracy, and flexibility.

        equity_factors: Configuration for equity risk factors (spot, vol, rate, div yield).
            Used by ParametricVaREngine for sensitivity-based calculations.
            If None, uses default configuration.

        fi_factors: Configuration for fixed income risk factors (parallel shift, key rates).
            Used by ParametricVaREngine for fixed income portfolios.
            If None, uses default configuration.

        scaling_method: Method for scaling VaR to longer holding periods.
            Options: "sqrt_t" (square root of time scaling), "overlapping" (overlapping returns).
            Default is "sqrt_t". "overlapping" is more accurate for short periods.

        mc_num_simulations: Number of simulations for Monte Carlo VaR.
            Default is 10000. More simulations improve accuracy but increase calculation time.
            Only used when var_method is MONTE_CARLO.

        mc_seed: Random seed for Monte Carlo simulations (for reproducibility).
            If None, uses system randomness. Set to an integer for reproducible results.

        calculate_component_var: Whether to calculate Component VaR (position-level risk).
            Component VaR decomposes portfolio VaR using Euler decomposition.
            Default is True. Adds some calculation overhead.

        calculate_marginal_var: Whether to calculate Marginal VaR (marginal risk impact).
            Marginal VaR measures the marginal contribution of each position.
            Default is True. Adds some calculation overhead.

        calculate_factor_var: Whether to calculate Factor VaR (risk factor attribution).
            Factor VaR decomposes VaR by underlying risk factors (spot, vol, rate, etc.).
            Default is True. Useful for factor-based risk management.

        calculate_incremental_var: Whether to calculate Incremental VaR (exclusion impact).
            Incremental VaR measures the change in VaR when excluding a position.
            Default is False (slower, requires multiple VaR calculations).
            Can also be calculated separately via calculate_incremental_var().

        calculate_stressed_var: Whether to calculate Stressed VaR (SVaR).
            SVaR measures VaR during crisis periods (Basel requirement).
            Default is False. Requires additional computation to identify crisis periods.

        stressed_period_start: Start date for stressed period (optional).
            If None, automatically detects the most volatile 12-month period.
            If specified, uses the provided date range.

        stressed_period_end: End date for stressed period (optional).
            Must be after stressed_period_start if both are specified.

        stressed_lookback_days: Number of days for stressed period window.
            Default is 252 (12 months of trading days). Typically kept at 252 days
            per Basel regulations.

    Examples:
        Basic 99% 1-day VaR:
        >>> config = VaRConfig(
        ...     confidence_level=0.99,
        ...     holding_period=1,
        ...     var_method=VaRMethod.HISTORICAL
        ... )

        VaR with attribution:
        >>> config = VaRConfig(
        ...     confidence_level=0.99,
        ...     calculate_component_var=True,
        ...     calculate_marginal_var=True,
        ...     calculate_factor_var=True
        ... )

        Fixed Income VaR:
        >>> from var.config import FIRiskFactorConfig
        >>> config = VaRConfig(
        ...     confidence_level=0.99,
        ...     fi_factors=FIRiskFactorConfig(
        ...         include_parallel_shift=True,
        ...         include_key_rates=True,
        ...         key_rate_tenors=[2.0, 5.0, 10.0, 30.0]
        ...     )
        ... )

        Monte Carlo VaR:
        >>> config = VaRConfig(
        ...     confidence_level=0.99,
        ...     var_method=VaRMethod.MONTE_CARLO,
        ...     mc_num_simulations=50000,
        ...     mc_seed=42
        ... )

        Stressed VaR (Basel):
        >>> config = VaRConfig(
        ...     confidence_level=0.99,
        ...     calculate_stressed_var=True,
        ...     stressed_lookback_days=252
        ... )
    """
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
