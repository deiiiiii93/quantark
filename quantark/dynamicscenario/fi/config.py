"""
Configuration for FI dynamic scenario analysis.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from quantark.util.exceptions import ValidationError


@dataclass
class FIDynamicScenarioConfig:
    """
    Configuration for FI dynamic scenario analysis execution.
    
    This class bundles all parameters needed to run an FI dynamic scenario,
    including risk measure calculation, hedging, and export settings.
    
    Attributes:
        calculate_dv01: Whether to calculate DV01 at each step
        calculate_convexity: Whether to calculate convexity at each step
        calculate_duration: Whether to calculate modified duration at each step
        calculate_key_rate_dv01: Whether to calculate key-rate DV01
        key_rate_tenors: List of tenors for key-rate DV01 (e.g., [1, 2, 5, 10, 30])
        dv01_alert_threshold: DV01 threshold for alerts (absolute value)
        hedge_enabled: Whether to enable hedging
        hedge_dv01_threshold: DV01 threshold for hedge trigger
        hedge_futures_spec: Specification for hedge futures (e.g., bond futures)
        futures_dv01_per_contract: DV01 per futures contract
        export_formats: List of export formats ('parquet', 'csv', 'json')
        output_dir: Directory for output files
        save_detailed_results: Whether to save position-level results
        save_intermediate_states: Whether to save state at each day
        generate_report: Whether to generate HTML report
        include_charts: Whether to include charts in report
        metadata: Additional metadata
        
    Example:
        >>> config = FIDynamicScenarioConfig(
        ...     calculate_dv01=True,
        ...     calculate_key_rate_dv01=True,
        ...     key_rate_tenors=[1, 2, 5, 10, 30],
        ...     hedge_enabled=True,
        ...     hedge_dv01_threshold=50000,
        ... )
    """
    # Risk measure calculation
    calculate_dv01: bool = True
    calculate_convexity: bool = True
    calculate_duration: bool = True
    calculate_key_rate_dv01: bool = False
    key_rate_tenors: List[float] = field(default_factory=lambda: [1, 2, 5, 10, 30])
    
    # Alert thresholds
    dv01_alert_threshold: float = 100000.0  # $100k DV01
    convexity_alert_threshold: float = 1000000.0
    
    # Hedging settings
    hedge_enabled: bool = False
    hedge_dv01_threshold: float = 50000.0  # Rebalance if DV01 exceeds threshold
    hedge_futures_spec: Optional[Dict[str, Any]] = None
    futures_dv01_per_contract: float = 80.0  # Default Treasury futures DV01
    
    # Export settings
    export_formats: List[str] = field(default_factory=lambda: ['parquet'])
    output_dir: str = './dynamic_results'
    save_detailed_results: bool = True
    save_intermediate_states: bool = True
    
    # Report settings
    generate_report: bool = True
    include_charts: bool = True
    
    # Additional settings
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration."""
        self._validate()
    
    def _validate(self):
        """
        Validate configuration parameters.
        
        Raises:
            ValidationError: If any parameters are invalid
        """
        if self.calculate_key_rate_dv01 and not self.key_rate_tenors:
            raise ValidationError(
                "key_rate_tenors must be provided when calculate_key_rate_dv01 is True"
            )
        
        valid_formats = ['parquet', 'csv', 'json', 'html']
        for fmt in self.export_formats:
            if fmt not in valid_formats:
                raise ValidationError(
                    f"Invalid export format '{fmt}'. "
                    f"Valid formats: {valid_formats}"
                )
        
        if self.dv01_alert_threshold <= 0:
            raise ValidationError("dv01_alert_threshold must be positive")
        
        if self.hedge_enabled and self.hedge_dv01_threshold <= 0:
            raise ValidationError("hedge_dv01_threshold must be positive when hedging is enabled")
    
    def validate(self) -> None:
        """
        Explicit validation method for external use.
        
        Raises:
            ValidationError: If configuration is invalid
        """
        self._validate()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of configuration.
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            'calculate_dv01': self.calculate_dv01,
            'calculate_convexity': self.calculate_convexity,
            'calculate_duration': self.calculate_duration,
            'calculate_key_rate_dv01': self.calculate_key_rate_dv01,
            'key_rate_tenors': self.key_rate_tenors,
            'dv01_alert_threshold': self.dv01_alert_threshold,
            'hedge_enabled': self.hedge_enabled,
            'hedge_dv01_threshold': self.hedge_dv01_threshold,
            'futures_dv01_per_contract': self.futures_dv01_per_contract,
            'export_formats': self.export_formats,
            'output_dir': self.output_dir,
            'save_detailed_results': self.save_detailed_results,
            'save_intermediate_states': self.save_intermediate_states,
            'generate_report': self.generate_report,
            'include_charts': self.include_charts,
            'metadata': self.metadata,
        }
    
    def __repr__(self) -> str:
        return (
            f"FIDynamicScenarioConfig("
            f"dv01={self.calculate_dv01}, "
            f"key_rate={self.calculate_key_rate_dv01}, "
            f"hedge={self.hedge_enabled}, "
            f"formats={self.export_formats})"
        )

