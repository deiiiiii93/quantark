# Code Style and Conventions

## Code Formatting Standards

### PEP 8 Compliance
All code must follow **PEP 8** Python style guide:
- Line length: 79 characters (hard limit: 99)
- Indentation: 4 spaces per level
- Blank lines: 2 blank lines between top-level classes/functions
- Imports: At top of file, grouped (standard library, third-party, local)
- Variable naming: snake_case
- Constants: UPPER_SNAKE_CASE

### Recommended Tools
- **Black**: Automatic code formatting
  ```bash
  black .  # Format entire project
  black file.py  # Format specific file
  ```
- **Flake8**: Linting and style checking
  ```bash
  flake8 .  # Lint entire project
  ```

## Naming Conventions

### Classes
- **PascalCase** (e.g., `EuropeanVanillaOption`, `BlackScholesEngine`)
- Nouns describing the entity
- Avoid abbreviations unless widely recognized (e.g., `VaR` for Value-at-Risk)

### Functions and Methods
- **snake_case** (e.g., `calculate_var`, `price_option`)
- Verb-noun naming pattern
- Descriptive names that indicate purpose

### Variables
- **snake_case** (e.g., `spot_price`, `volatility_surface`)
- Descriptive names
- Single-letter variables acceptable in narrow scope (e.g., `i` in loops)

### Constants
- **UPPER_SNAKE_CASE** (e.g., `MAX_ITERATIONS`, `DEFAULT_VOLATILITY`)

### Private Members
- Leading underscore: `_private_method`, `_internal_value`
- Double leading underscore for name mangling: `__very_private` (rarely used)

### Type Variables
- PascalCase: `T`, `K`, `VT_co` (per PEP 484)

## Type Hints

### Mandatory Type Annotations
All public APIs **must** include type hints:
```python
from typing import Dict, List, Optional, Union

def calculate_var(
    portfolio: Portfolio,
    market_data: pd.DataFrame,
    confidence_level: float = 0.99
) -> VaRResult:
    """Calculate Value-at-Risk for portfolio."""
    pass

class VaREngine:
    config: VaRConfig
    _internal_cache: Dict[str, float]

    def __init__(self, config: VaRConfig) -> None:
        self.config = config
```

### Common Type Patterns
```python
# Optional values
confidence_level: Optional[float] = None

# Union types
price: Union[int, float]

# Collections
positions: List[str]
risk_factors: Dict[str, float]

# Callable types
calculator: Callable[[float, float], float]

# Generic types
from typing import TypeVar, Generic

T = TypeVar('T')
class Container(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value
```

### Dataclasses
Extensive use of dataclasses for data containers:
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class VaRConfig:
    confidence_level: float
    holding_period: int = 1
    lookback_days: int = 252
    calculate_component_var: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not 0 < self.confidence_level < 1:
            raise ValidationError("Invalid confidence level")
```

## Docstring Standards

### Format
Use **Google-style** docstrings for all public classes and methods:
```python
def calculate_var(
    portfolio: Portfolio,
    market_data: pd.DataFrame,
    config: VaRConfig
) -> VaRResult:
    """
    Calculate Value-at-Risk for a portfolio.

    This method computes VaR using the configured engine (parametric,
    historical, or Monte Carlo) based on the provided market data
    and portfolio positions.

    Args:
        portfolio: Portfolio object containing positions
        market_data: DataFrame with historical market data
        config: VaR configuration parameters

    Returns:
        VaRResult object containing VaR value and metadata

    Raises:
        ValidationError: If portfolio is empty or invalid
        MarketDataError: If market data is insufficient

    Example:
        >>> config = VaRConfig(confidence_level=0.99)
        >>> engine = ParametricVaREngine(config)
        >>> result = engine.calculate_var(portfolio, data)
        >>> print(f"VaR: ${result.var:.2f}")

    Note:
        For large portfolios (>1000 positions), consider using
        the parametric engine for better performance.
    """
    pass
```

### Docstring Elements
1. **Short description**: One-line summary
2. **Detailed description**: Full explanation if needed
3. **Args**: Parameter descriptions
4. **Returns**: Return value description
5. **Raises**: Exceptions that may be raised
6. **Example**: Usage example (optional but recommended)
7. **Note**: Additional notes (optional)

### Classes
```python
class VaREngine:
    """
    Abstract base class for VaR calculation engines.

    This class defines the interface for all VaR calculation engines,
    including parametric, historical, and Monte Carlo methods.

    Attributes:
        config: VaR configuration parameters
        name: Engine name identifier

    See Also:
        ParametricVaREngine: Variance-covariance implementation
        HistoricalVaREngine: Historical simulation implementation
        MonteCarloVaREngine: Monte Carlo simulation implementation
    """
    pass
```

## Code Organization

### Import Organization
```python
"""Module docstring."""

# Standard library imports
from typing import Dict, List, Optional
from datetime import datetime
import logging

# Third-party imports
import numpy as np
import pandas as pd
from scipy import stats

# Local imports
from param import SpotQuote, VolatilitySurface
from priceenv import PricingEnvironment
from var.base import VaREngine
from util.exceptions import QuantArkException
```

### Class Organization
```python
class ExampleClass:
    """Class docstring."""

    # Class variables
    DEFAULT_VALUE: int = 100

    def __init__(self, value: int) -> None:
        """Initialize with value."""
        self.value = value
        self._logger = logging.getLogger(__name__)

    def public_method(self) -> None:
        """Public method documentation."""
        pass

    def _private_method(self) -> None:
        """Private method documentation."""
        pass

    @property
    def value_property(self) -> int:
        """Property documentation."""
        return self._value

    @classmethod
    def from_default(cls) -> 'ExampleClass':
        """Class method documentation."""
        return cls(cls.DEFAULT_VALUE)

    @staticmethod
    def static_method() -> None:
        """Static method documentation."""
        pass
```

## Design Patterns

### Engine Method Selection Pattern
For engines with multiple methods, use **two-level enum pattern**:
```python
# In util/enum/engine_enums.py
from enum import Enum

class EngineType(Enum):
    ANALYTICAL = "analytical"
    MONTE_CARLO = "monte_carlo"
    PDE = "pde"

class AmericanAnalyticalMethod(Enum):
    BAW = "baw"
    BSM = "bsm"

# Usage in engine
from util.enum.engine_enums import AmericanAnalyticalMethod, EngineType

engine = AmericanOptionAnalyticalEngine(
    method=EngineType.ANALYTICAL(AmericanAnalyticalMethod.BAW)
)
```

### Protocol Pattern
For abstract interfaces:
```python
from typing import Protocol

class VaREngine(Protocol):
    """Protocol for VaR calculation engines."""

    config: VaRConfig

    def calculate_var(
        self,
        portfolio: Portfolio,
        market_data: pd.DataFrame
    ) -> VaRResult:
        """Calculate VaR for portfolio."""
        ...
```

## Error Handling

### Exception Hierarchy
Always use the project's exception hierarchy:
```python
# Don't raise generic exceptions
raise ValueError("Invalid input")  # ❌ Wrong

# Use specific QuantArk exceptions
from util.exceptions import ValidationError, NumericalError

raise ValidationError("Confidence level must be between 0 and 1")  # ✅ Correct
raise NumericalError("Failed to converge after 100 iterations")  # ✅ Correct
```

### Exception Handling
```python
try:
    result = engine.calculate_var(portfolio, data)
except ValidationError as e:
    logger.error(f"Invalid configuration: {e}")
    raise
except NumericalError as e:
    logger.warning(f"Numerical issue: {e}")
    # Return default or handle gracefully
    return VaRResult(var=0.0, cvar=0.0)
```

## Performance Guidelines

### Vectorization
Use NumPy vectorized operations instead of loops:
```python
# ❌ Slow - Python loop
results = []
for i in range(len(data)):
    results.append(calculate_value(data[i]))

# ✅ Fast - NumPy vectorization
results = calculate_vectorized_values(data)
```

### Memory Efficiency
- Use views instead of copies when possible
- Avoid creating unnecessary intermediate arrays
- Use appropriate data types (float32 vs float64)

### Caching
```python
from functools import lru_cache

class VaREngine:
    @lru_cache(maxsize=128)
    def calculate_covariance(self, data: tuple) -> np.ndarray:
        """Cache covariance matrix calculations."""
        return np.cov(data)
```

## Testing Conventions

### Test Structure
```python
import pytest
from var import ParametricVaREngine, VaRConfig

class TestParametricVaREngine:
    """Test suite for ParametricVaREngine."""

    @pytest.fixture
    def sample_config(self) -> VaRConfig:
        """Create sample VaR configuration."""
        return VaRConfig(
            confidence_level=0.99,
            holding_period=1
        )

    def test_var_calculation_basic(self, sample_config) -> None:
        """Test basic VaR calculation."""
        engine = ParametricVaREngine(config=sample_config)
        # Test implementation
        assert result.var > 0

    def test_var_calculation_edge_case(self) -> None:
        """Test edge case handling."""
        with pytest.raises(ValidationError):
            # Test that invalid input raises error
            pass
```

## Documentation Standards

### Inline Comments
Use sparingly and write meaningful comments:
```python
# ✅ Good - explains WHY
# Use exponential moving average for faster volatility update
volatility = alpha * returns[-1] + (1 - alpha) * volatility

# ❌ Bad - states WHAT (obvious from code)
volatility = alpha * returns[-1] + (1 - alpha) * volatility  # Calculate volatility

# ❌ Bad - unnecessary comments
vol = 100.0  # Set vol to 100
```

### Module Docstring
Every module should have a docstring:
```python
"""
VaR (Value-at-Risk) calculation module.

This module provides functionality for calculating portfolio Value-at-Risk
using three different methodologies:
- Parametric (variance-covariance)
- Historical simulation
- Monte Carlo simulation

The module also includes risk attribution features (Component, Marginal,
Incremental, Factor VaR) and Stressed VaR for regulatory compliance.

Modules:
    engines: VaR calculation engines
    config: Configuration dataclasses
    attribution: Risk attribution calculators
    results: Result objects and report generators

Examples:
    Basic VaR calculation:
    >>> from var import VaRConfig, ParametricVaREngine
    >>> config = VaRConfig(confidence_level=0.99)
    >>> engine = ParametricVaREngine(config)
    >>> result = engine.calculate_var(portfolio, data)
"""

from .engines import ParametricVaREngine, HistoricalVaREngine
from .config import VaRConfig
# ...
```

## Configuration Files

### Python Style Config
Use dataclasses for configuration:
```python
@dataclass
class EngineConfig:
    """Configuration for engine parameters."""
    num_simulations: int = 10000
    confidence_level: float = 0.99
    method: str = "monte_carlo"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_simulations <= 0:
            raise ValidationError("num_simulations must be positive")
```

## Git Commit Messages

### Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation only changes
- **style**: Formatting, missing semi colons, etc
- **refactor**: Code restructuring without changing behavior
- **test**: Adding or modifying tests
- **chore**: Maintenance tasks

### Examples
```
feat(var): add Monte Carlo VaR engine implementation

Implement MonteCarloVaREngine with correlated scenario generation
and configurable simulation parameters.

Closes #42
```

## Best Practices Summary
1. ✅ Follow PEP 8 style guide
2. ✅ Use type hints on all public APIs
3. ✅ Write Google-style docstrings
4. ✅ Use dataclasses for data containers
5. ✅ Raise specific QuantArk exceptions
6. ✅ Use NumPy vectorization for performance
7. ✅ Write comprehensive tests
8. ✅ Keep functions small and focused
9. ✅ Use meaningful variable names
10. ✅ Document WHY, not WHAT
