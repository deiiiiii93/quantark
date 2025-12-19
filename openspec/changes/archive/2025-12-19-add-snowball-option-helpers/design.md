# Design: Snowball Option Helper Functions

## Overview

This document details the design for factory functions that simplify Snowball option creation. Each helper encapsulates structure-specific defaults while allowing full customization.

## Helper Function Signatures

### 1. Standard Snowball

```python
def create_standard_snowball(
    initial_price: float,
    strike: float,
    maturity: float,
    notional: float = 1_000_000.0,
    ko_barrier: float = None,      # Default: 1.03 * initial_price
    ko_rate: float = 0.15,         # 15% annualized
    ki_barrier: float = None,      # Default: 0.75 * initial_price
    num_observations: int = 12,    # Monthly by default
    is_reverse: bool = False,
    **kwargs,
) -> SnowballOption:
    """
    Create a standard snowball with flat KO barrier and continuous KI monitoring.
    
    Structure:
    - KO: Discrete, monthly observations, flat barrier
    - KI: Continuous monitoring
    - Payoff: Annualized coupon on KO, participation loss on KI
    """
```

### 2. Step-Down Snowball

```python
def create_stepdown_snowball(
    initial_price: float,
    strike: float,
    maturity: float,
    notional: float = 1_000_000.0,
    initial_ko_barrier: float = None,  # Default: 1.03 * initial_price
    stepdown_rate: float = 0.005,      # 0.5% per period
    ko_rate: float = 0.15,
    ki_barrier: float = None,          # Default: 0.75 * initial_price
    num_observations: int = 12,
    is_reverse: bool = False,
    **kwargs,
) -> SnowballOption:
    """
    Create a step-down snowball where KO barrier decreases each observation.
    
    Structure:
    - KO: Barrier starts at initial_ko_barrier, decreases by stepdown_rate each period
    - KI: Continuous monitoring
    - Common in: China structured products market
    """
```

### 3. European Knock-In Snowball

```python
def create_european_ki_snowball(
    initial_price: float,
    strike: float,
    maturity: float,
    notional: float = 1_000_000.0,
    ko_barrier: float = None,
    ko_rate: float = 0.15,
    ki_barrier: float = None,
    num_ko_observations: int = 12,
    is_reverse: bool = False,
    **kwargs,
) -> SnowballOption:
    """
    Create a snowball with European-style KI (only observed at maturity).
    
    Structure:
    - KO: Discrete observations
    - KI: Single observation at maturity only (European-style)
    - Benefit: Higher probability of V0 outcome
    """
```

### 4. Parachute Snowball

```python
def create_parachute_snowball(
    initial_price: float,
    strike: float,
    maturity: float,
    notional: float = 1_000_000.0,
    ko_barrier: float = None,      # Default: 1.03 * initial_price (for early observations)
    ko_rate: float = 0.15,
    ki_barrier: float = None,      # Default: 0.75 * initial_price
    num_observations: int = 12,
    is_reverse: bool = False,
    **kwargs,
) -> SnowballOption:
    """
    Create a parachute snowball where last KO barrier equals KI barrier.
    
    Structure:
    - KO barriers: [ko_barrier, ko_barrier, ..., ki_barrier] (last one = KI)
    - KI: Continuous monitoring at ki_barrier
    - Benefit: At final observation, if spot > KI barrier, product knocks out
              (guaranteed exit if not knocked in)
    - Common in: China market as "降落伞雪球"
    """
```

### 5. Phoenix Snowball

```python
def create_phoenix_snowball(
    initial_price: float,
    strike: float,
    maturity: float,
    notional: float = 1_000_000.0,
    ko_barrier: float = None,
    ko_rate: float = 0.15,
    ki_barrier: float = None,
    coupon_barrier: float = None,   # Default: 0.80 * initial_price
    coupon_rate: float = 0.01,      # 1% per period if above coupon_barrier
    num_observations: int = 12,
    is_reverse: bool = False,
    **kwargs,
) -> SnowballOption:
    """
    Create a phoenix snowball with periodic memory coupons.
    
    Structure:
    - Pays periodic coupon if spot > coupon_barrier (even without KO)
    - Memory feature: Missed coupons paid if recovered above barrier
    - Note: Full phoenix requires engine support; this creates the product structure
    """
```

### 6. Airbag Snowball

```python
def create_airbag_snowball(
    initial_price: float,
    strike: float,
    maturity: float,
    notional: float = 1_000_000.0,
    ko_barrier: float = None,
    ko_rate: float = 0.15,
    ki_barrier: float = None,
    airbag_barrier: float = None,   # Default: 0.60 * initial_price
    participation_rate: float = 0.5,  # 50% participation below airbag
    num_observations: int = 12,
    is_reverse: bool = False,
    **kwargs,
) -> SnowballOption:
    """
    Create an airbag snowball with reduced participation below airbag barrier.
    
    Structure:
    - Standard KO/KI structure
    - V1 payoff: Reduced participation rate below airbag_barrier
    - Benefit: Additional protection for extreme downside
    """
```

## Utility Functions

### Observation Date Generator

```python
def generate_ko_observation_dates(
    maturity: float,
    frequency: str = "monthly",  # "monthly", "quarterly", "weekly"
    skip_first: int = 0,         # Skip first N observations (lock-out period)
) -> List[float]:
    """Generate evenly spaced observation dates as year fractions."""
```

### Step-Down Barrier Generator

```python
def generate_stepdown_barriers(
    initial_barrier: float,
    stepdown_rate: float,
    num_observations: int,
    min_barrier: float = None,  # Optional floor
) -> List[float]:
    """Generate decreasing barrier levels for step-down structure."""
```

## Default Values

| Parameter | Standard | Step-Down | European KI | Parachute | Phoenix | Airbag |
|-----------|----------|-----------|-------------|-----------|---------|--------|
| KO barrier | 103% S0 | 103%→step | 103% S0 | 103%→KI | 103% S0 | 103% S0 |
| KO rate | 15% | 15% | 15% | 15% | 15% | 15% |
| KI barrier | 75% S0 | 75% S0 | 75% S0 | 75% S0 | 75% S0 | 75% S0 |
| KO obs type | Discrete | Discrete | Discrete | Discrete | Discrete | Discrete |
| KI obs type | Continuous | Continuous | Discrete | Continuous | Continuous | Continuous |
| Protection | None | None | None | None | None | None |
| Coupon timing | Instant | Instant | Instant | Instant | Instant | Instant |
| Include principal | False | False | False | False | False | False |

**Notes:**
- **Step-Down**: KO barrier decreases by stepdown_rate each observation
- **Parachute**: Last KO barrier equals KI barrier (guaranteed exit if not knocked in)

## Implementation Notes

### Parameter Override Pattern

All helpers accept `**kwargs` that are passed to underlying config objects:

```python
def create_standard_snowball(..., **kwargs):
    # Extract config-specific kwargs
    barrier_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) 
                      if k in BarrierConfig.__annotations__}
    payoff_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) 
                     if k in PayoffConfig.__annotations__}
    accrual_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) 
                      if k in AccrualConfig.__annotations__}
    
    # Remaining kwargs go to SnowballOption
    return SnowballOption(..., **kwargs)
```

### Validation

Helpers should validate that:
1. `initial_price > 0`, `strike > 0`, `maturity > 0`, `notional > 0`
2. Derived barriers are positive (e.g., if using multipliers)
3. Conflicting overrides are detected (e.g., both `ko_barrier` and `ko_barrier` in kwargs)

### Error Messages

Helpers should provide clear error messages:
```python
if maturity <= 0:
    raise ValidationError(
        f"create_standard_snowball: maturity must be positive, got {maturity}"
    )
```

## File Structure

```
asset/equity/product/option/
├── snowball_option.py      # Existing main class
├── snowball_config.py      # Existing config classes
├── snowball_helpers.py     # NEW: Factory functions
├── observation_schedule.py # Existing schedule class
└── __init__.py             # Updated to export helpers
```

## Testing Strategy

1. **Unit tests per helper**: Verify each creates valid SnowballOption
2. **Default value tests**: Ensure defaults produce expected configuration
3. **Override tests**: Verify kwargs properly override defaults
4. **Edge case tests**: Zero/negative values, extreme parameters
5. **Integration tests**: Price created options with SnowballMCEngine
