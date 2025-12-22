# Product Implementation Patterns

Detailed patterns extracted from the QuantArk codebase.

## Class Structure Pattern

### Simple Product (Dataclass)
```python
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from util.enum import OptionType, ExerciseType
from util.exceptions import ValidationError
from .base_equity_option import BaseEquityOption


@dataclass
class MyOption(BaseEquityOption):
    """
    [Product description].

    Attributes:
        strike: Strike price
        option_type: CALL or PUT
        maturity: Time to maturity in years (mutually exclusive with exercise_date)
        exercise_date: Exercise date (mutually exclusive with maturity)
    """

    def __init__(
        self,
        strike: float,
        option_type: OptionType,
        maturity: Optional[float] = None,
        exercise_date: Optional[datetime] = None,
        # ... additional parameters
    ):
        # Handle maturity/date mutual exclusivity
        if maturity is None and exercise_date is None:
            maturity = 0.0  # Will trigger validation error

        super().__init__(
            strike=strike,
            maturity=maturity,
            option_type=option_type,
            exercise_type=ExerciseType.EUROPEAN,
            exercise_date=exercise_date,
        )

    def get_payoff(self, spot: float) -> float:
        """Calculate payoff at maturity."""
        if self.is_call():
            return max(spot - self.strike, 0.0)
        else:
            return max(self.strike - spot, 0.0)

    def validate(self) -> None:
        """Validate product parameters."""
        super().validate()
        # Add product-specific validation
```

### Complex Product (with Config Objects)
```python
from dataclasses import dataclass, field
from typing import Optional, Union, List

from .base_structured_product import BaseStructuredProduct
from .my_config import MyBarrierConfig, MyPayoffConfig


@dataclass
class MyStructuredProduct(BaseStructuredProduct):
    """Complex product with configuration objects."""

    barrier_config: MyBarrierConfig
    payoff_config: MyPayoffConfig = field(default_factory=MyPayoffConfig)

    def __init__(
        self,
        initial_price: float,
        strike: float,
        maturity: float,
        notional: float,
        barrier_config: MyBarrierConfig,
        payoff_config: Optional[MyPayoffConfig] = None,
        is_reverse: bool = False,
    ):
        self.barrier_config = barrier_config
        self.payoff_config = payoff_config or MyPayoffConfig()

        super().__init__(
            initial_price=initial_price,
            strike=strike,
            maturity=maturity,
            notional=notional,
            is_reverse=is_reverse,
        )
```

## Config Object Pattern

```python
from dataclasses import dataclass
from typing import Optional, Union, List

from util.enum import ObservationType


@dataclass(frozen=True)
class MyBarrierConfig:
    """
    Immutable configuration for barrier parameters.

    Attributes:
        barrier: Barrier level (scalar or array for time-varying)
        observation_type: DISCRETE or CONTINUOUS
    """

    barrier: Union[float, List[float]]
    observation_type: ObservationType = ObservationType.DISCRETE
    observation_dates: Optional[List[float]] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate barrier is positive
        self._validate_positive(self.barrier, "barrier")

        # Validate enum type
        if not isinstance(self.observation_type, ObservationType):
            raise ValueError(
                f"observation_type must be ObservationType, "
                f"got {type(self.observation_type)}"
            )

    @staticmethod
    def _validate_positive(value: Union[float, List[float]], name: str) -> None:
        """Validate that value(s) are positive."""
        if isinstance(value, list):
            if not value:
                raise ValueError(f"{name} list cannot be empty")
            for i, v in enumerate(value):
                if not isinstance(v, (int, float)) or v <= 0:
                    raise ValueError(f"{name}[{i}] must be positive, got {v}")
        else:
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
```

## Factory Function Pattern

```python
from typing import Optional, List
from dataclasses import fields

from util.exceptions import ValidationError
from .my_product import MyProduct
from .my_config import MyBarrierConfig, MyPayoffConfig


def _validate_core_params(
    initial_price: float,
    strike: float,
    maturity: float,
    func_name: str,
) -> None:
    """Validate core parameters common to all helpers."""
    if initial_price <= 0:
        raise ValidationError(f"{func_name}: initial_price must be positive")
    if strike <= 0:
        raise ValidationError(f"{func_name}: strike must be positive")
    if maturity <= 0:
        raise ValidationError(f"{func_name}: maturity must be positive")


def _extract_config_kwargs(kwargs: dict) -> tuple:
    """Extract kwargs for each config class using introspection."""
    barrier_fields = {f.name for f in fields(MyBarrierConfig)}
    payoff_fields = {f.name for f in fields(MyPayoffConfig)}

    barrier_kwargs = {}
    payoff_kwargs = {}
    unknown = []

    for key, value in kwargs.items():
        if key in barrier_fields:
            barrier_kwargs[key] = value
        elif key in payoff_fields:
            payoff_kwargs[key] = value
        else:
            unknown.append(key)

    if unknown:
        raise ValidationError(f"Unknown parameters: {', '.join(unknown)}")

    return barrier_kwargs, payoff_kwargs


def create_standard_product(
    initial_price: float,
    strike: float,
    maturity: float,
    notional: float = 1_000_000.0,
    barrier: Optional[float] = None,
    **kwargs,
) -> MyProduct:
    """
    Create standard product with sensible defaults.

    Args:
        initial_price: Reference price
        strike: Strike price
        maturity: Time to maturity in years
        notional: Notional principal (default: 1,000,000)
        barrier: Barrier level (default: 103% of initial_price)
        **kwargs: Additional config parameters

    Returns:
        Configured MyProduct instance

    Example:
        >>> product = create_standard_product(
        ...     initial_price=100.0,
        ...     strike=100.0,
        ...     maturity=1.0,
        ... )
    """
    _validate_core_params(initial_price, strike, maturity, "create_standard_product")

    # Apply sensible defaults
    if barrier is None:
        barrier = 1.03 * initial_price

    # Extract config kwargs
    barrier_kwargs, payoff_kwargs = _extract_config_kwargs(kwargs)

    # Build configs
    barrier_config = MyBarrierConfig(barrier=barrier, **barrier_kwargs)
    payoff_config = MyPayoffConfig(**payoff_kwargs)

    return MyProduct(
        initial_price=initial_price,
        strike=strike,
        maturity=maturity,
        notional=notional,
        barrier_config=barrier_config,
        payoff_config=payoff_config,
    )
```

## Validation Patterns

### Basic Validation
```python
def validate(self) -> None:
    """Validate product parameters."""
    # Call parent validation first
    super().validate()

    # Validate positive values
    if self.barrier <= 0:
        raise ValidationError(f"barrier must be positive, got {self.barrier}")

    # Validate enum types
    if not isinstance(self.barrier_type, BarrierType):
        raise ValidationError(
            f"barrier_type must be BarrierType enum, got {type(self.barrier_type)}"
        )

    # Validate relationships
    if self.barrier >= self.strike:
        raise ValidationError(
            f"barrier ({self.barrier}) must be less than strike ({self.strike})"
        )
```

### Maturity Validation (Dual Format)
```python
def _validate_maturity_dates(self) -> None:
    """Validate maturity specification."""
    has_dates = self.exercise_date is not None
    has_maturity = self.maturity is not None and self.maturity > 0

    if not has_dates and not has_maturity:
        raise ValidationError(
            "Either maturity or exercise_date must be provided"
        )

    if has_dates and has_maturity:
        raise ValidationError(
            "Cannot provide both maturity and exercise_date"
        )
```

## Maturity Resolution Pattern

```python
def get_maturity(self, pricing_env=None) -> float:
    """Get time to maturity, supporting both formats."""
    if self.exercise_date is not None:
        # Date-based: requires pricing environment
        if pricing_env is None:
            raise ValidationError(
                "PricingEnvironment required for date-based calculation"
            )

        if pricing_env.valuation_date >= self.exercise_date:
            raise ValidationError("Option has expired")

        return calculate_year_fraction(
            pricing_env.valuation_date,
            self.exercise_date,
            pricing_env.day_count_convention,
        )
    else:
        # Time-based: return stored maturity
        if self.maturity is None:
            raise ValidationError("Maturity is not set")
        return self.maturity
```

## Quantity/Notional Duality Pattern

```python
def get_quantity(self) -> float:
    """Get quantity (number of contracts)."""
    if self.quantity is not None and self.quantity > 0:
        return self.quantity
    if self.notional is not None and self.initial_price is not None:
        if self.initial_price <= 0:
            raise ValidationError("initial_price must be positive")
        return self.notional / self.initial_price
    raise ValidationError("Quantity cannot be determined")

def get_notional(self) -> float:
    """Get notional principal amount."""
    if self.notional is not None:
        return self.notional
    if self.quantity is not None and self.initial_price is not None:
        if self.initial_price <= 0:
            raise ValidationError("initial_price must be positive")
        return self.quantity * self.initial_price
    raise ValidationError("Notional cannot be determined")
```

## Numerical Safety Pattern

```python
from util.numerical import is_zero, safe_log, safe_sqrt, safe_exp

def calculate_something(self, spot: float, pricing_env) -> float:
    """Use safe numerical operations."""
    time_to_expiry = self.get_maturity(pricing_env)

    # Safe zero check for expiry
    if is_zero(time_to_expiry):
        return self.get_payoff(spot)

    # Safe log for moneyness
    log_moneyness = safe_log(spot / self.strike)

    # Safe sqrt for volatility scaling
    sigma_sqrt_t = safe_sqrt(variance * time_to_expiry)

    # Safe exp for discounting
    discount = safe_exp(-rate * time_to_expiry)

    return result
```

## Multi-State Payoff Pattern

```python
def get_payoff(self, spot: float, pricing_env=None, **kwargs) -> float:
    """Get appropriate payoff based on state."""
    knocked_in = kwargs.get("knocked_in", False)
    knocked_out = kwargs.get("knocked_out", False)

    if knocked_out:
        return self._get_knockout_payoff(spot, pricing_env)
    elif knocked_in:
        return self._get_knockin_payoff(spot, pricing_env)
    else:
        return self._get_vanilla_payoff(spot, pricing_env)

def _get_vanilla_payoff(self, spot: float, pricing_env=None) -> float:
    """Payoff when no barrier event occurred (V0 state)."""
    principal = self.notional if self.payoff_config.include_principal else 0.0
    rebate = self.payoff_config.rebate_rate * self.notional
    return principal + rebate

def _get_knockin_payoff(self, spot: float, pricing_env=None) -> float:
    """Payoff when knock-in occurred (V1 state)."""
    principal = self.notional if self.payoff_config.include_principal else 0.0

    # Calculate downside with participation
    raw_diff = spot - self.strike
    downside = self.payoff_config.participation_rate * min(raw_diff, 0.0)
    downside *= self.notional / self.initial_price

    return principal + downside
```

## Module Export Pattern

```python
# In __init__.py
from .my_product import MyProduct
from .my_config import MyBarrierConfig, MyPayoffConfig
from .my_helpers import (
    create_standard_product,
    create_variant_product,
)

__all__ = [
    # Main product
    "MyProduct",
    # Config classes
    "MyBarrierConfig",
    "MyPayoffConfig",
    # Factory functions
    "create_standard_product",
    "create_variant_product",
]
```
