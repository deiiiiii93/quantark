"""
Configuration for dynamic scenario analysis.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from util.exceptions import ValidationError


@dataclass
class DynamicScenarioConfig:
    """
    Configuration for dynamic scenario analysis execution.
    
    This class bundles all parameters needed to run a dynamic scenario,
    including calculation settings, export formats, and hedging options.
    
    Attributes:
        calculate_greeks: Whether to calculate Greeks at each step
        greeks_method: Method for Greeks calculation ('analytical' or 'numerical')
        export_formats: List of export formats ('parquet', 'csv', 'json')
        output_dir: Directory for output files
        save_detailed_results: Whether to save position-level results
        save_intermediate_states: Whether to save state at each day
        generate_report: Whether to generate HTML report
        include_charts: Whether to include charts in report
        metadata: Additional metadata
        
    Example:
        >>> config = DynamicScenarioConfig(
        ...     calculate_greeks=True,
        ...     export_formats=['csv', 'json'],
        ...     output_dir='./dynamic_results'
        ... )
    """
    # Greeks calculation
    calculate_greeks: bool = True
    greeks_method: str = 'analytical'
    
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
        if self.greeks_method not in ['analytical', 'numerical']:
            raise ValidationError(
                f"Invalid greeks_method '{self.greeks_method}'. "
                "Must be 'analytical' or 'numerical'"
            )
        
        valid_formats = ['parquet', 'csv', 'json', 'html']
        for fmt in self.export_formats:
            if fmt not in valid_formats:
                raise ValidationError(
                    f"Invalid export format '{fmt}'. "
                    f"Valid formats: {valid_formats}"
                )
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of configuration.
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            'calculate_greeks': self.calculate_greeks,
            'greeks_method': self.greeks_method,
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
            f"DynamicScenarioConfig("
            f"greeks={self.calculate_greeks}, "
            f"formats={self.export_formats}, "
            f"output_dir='{self.output_dir}')"
        )

