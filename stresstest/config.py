"""
Configuration for stress test execution.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from util.exceptions import ValidationError


@dataclass
class StressTestConfig:
    """
    Configuration for stress test execution.
    
    This class bundles all parameters needed to run a stress test,
    including calculation settings, export formats, and reporting options.
    
    Attributes:
        calculate_greeks: Whether to calculate Greeks in stressed scenarios
        greeks_method: Method for Greeks calculation ('analytical' or 'numerical')
        export_formats: List of export formats ('parquet', 'csv', 'json')
        output_dir: Directory for output files (default: './stress_results')
        save_detailed_results: Whether to save position-level results
        parallel_execution: Whether to run scenarios in parallel (future)
        max_workers: Max worker threads for parallel execution (future)
        progress_callback: Optional callback for progress updates (future)
        metadata: Additional metadata for the stress test
        
    Example:
        >>> config = StressTestConfig(
        ...     calculate_greeks=True,
        ...     export_formats=['parquet', 'csv'],
        ...     output_dir='./my_stress_results'
        ... )
    """
    calculate_greeks: bool = True
    greeks_method: str = 'analytical'
    export_formats: List[str] = field(default_factory=lambda: ['parquet'])
    output_dir: str = './stress_results'
    save_detailed_results: bool = True
    
    # Future features for dynamic scenario analysis
    parallel_execution: bool = False
    max_workers: int = 4
    progress_callback: Optional[Any] = None
    
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
        
        if self.max_workers < 1:
            raise ValidationError(
                f"max_workers must be at least 1, got {self.max_workers}"
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
            'parallel_execution': self.parallel_execution,
            'metadata': self.metadata,
        }
    
    def __repr__(self) -> str:
        return (
            f"StressTestConfig("
            f"greeks={self.calculate_greeks}, "
            f"formats={self.export_formats}, "
            f"output_dir='{self.output_dir}')"
        )

