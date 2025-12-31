# Design: Factor-Specific Bump Sizes for Numerical Greeks

## Current State

The `GreeksCalculator` uses the following bump sizes:

| Greek   | Parameter        | Default | Type      |
|---------|------------------|---------|-----------|
| Delta   | `params.bump_size` | 1e-4   | Relative  |
| Gamma   | `params.bump_size` | 1e-4   | Relative  |
| Vega    | Hardcoded        | 0.01    | Absolute  |
| Theta   | Hardcoded        | 1 day   | Absolute  |
| Rho     | Hardcoded        | 0.01    | Absolute  |

Issues:
1. Only delta/gamma bump size is configurable
2. Different risk factors have different appropriate bump types (relative vs absolute)
3. No dividend_rho implementation
4. Hardcoded values cannot be adjusted for specific use cases

## Proposed Design

### 1. BumpConfig Dataclass

A new dataclass that encapsulates all bump sizes with appropriate types and validation:

```python
@dataclass
class BumpConfig:
    """
    Configuration for factor-specific bump sizes in numerical Greeks calculation.

    Each risk factor can have its own bump size and type (relative or absolute).
    Defaults follow industry conventions from Tolerance constants.

    Attributes:
        spot_bump: Relative bump for delta/gamma (default: 1% = 0.01)
        vol_bump: Absolute bump for vega (default: 1 vol point = 0.01)
        time_bump_days: Absolute bump in days for theta (default: 1)
        rate_bump: Absolute bump for rho (default: 1bp = 0.0001)
        div_bump: Absolute bump for dividend_rho (default: 1bp = 0.0001)
    """
    spot_bump: float = 0.01
    vol_bump: float = 0.01
    time_bump_days: int = 1
    rate_bump: float = 0.0001
    div_bump: float = 0.0001

    def __post_init__(self):
        # Validation: positive values, reasonable ranges
```

### 2. EngineParams Integration

Modify `EngineParams` to include `BumpConfig` while maintaining backward compatibility:

```python
@dataclass
class EngineParams:
    bump_size: float = 1e-4  # DEPRECATED - use bump_config instead
    bump_config: Optional[BumpConfig] = None
    bus_days_in_year: int = 252

    def __post_init__(self):
        # Validate bump_size (legacy)
        # Create bump_config from bump_size if None
        if self.bump_config is None:
            self.bump_config = BumpConfig(
                spot_bump=self.bump_size,
                # Use industry defaults for other factors
            )

    def get_effective_bump_config(self) -> BumpConfig:
        return self.bump_config if self.bump_config else BumpConfig(spot_bump=self.bump_size)
```

### 3. GreeksCalculator Updates

**Initialization:**
```python
def __init__(self, params: Optional[EngineParams] = None):
    self.params = params if params is not None else EngineParams()
    self._bump_config = self.params.get_effective_bump_config()
```

**Individual Methods:**
Each Greek calculation method accepts an optional bump parameter that overrides the config:

```python
def calculate_numerical_vega(
    self, product, pricing_env, engine,
    base_price: Optional[float] = None,
    vol_bump: Optional[float] = None,  # Override config
) -> float:
    vol_bump = vol_bump if vol_bump is not None else self._bump_config.vol_bump
    # ... implementation
```

### 4. New Dividend Rho Calculation

```python
def calculate_numerical_dividend_rho(
    self, product, pricing_env, engine,
    base_price: Optional[float] = None,
    div_bump: Optional[float] = None,
) -> float:
    """
    Numerical dividend_rho (psi) from dividend yield bump.

    Measures price sensitivity to dividend yield changes.
    Negative for calls (higher div = lower call price).
    Positive for puts (higher div = higher put price).
    """
    # Bump dividend yield and reprice
```

### 5. Bump Type Rationale

| Factor | Type      | Rationale                                      |
|--------|-----------|------------------------------------------------|
| Spot   | Relative  | Delta is unitless; relative bump scales with S |
| Vol    | Absolute  | Vega quoted per 1 vol point; industry standard |
| Time   | Absolute  | Theta quoted per day; natural time unit        |
| Rate   | Absolute  | Rho quoted per 1bp; industry standard           |
| Div    | Absolute  | Dividend rho quoted per 1bp; industry standard |

## Alternatives Considered

### 1. Single Bump Size with Type Flag
Rejected: Different factors inherently need different bump types (relative vs absolute).

### 2. Separate BumpConfig per Asset Class
Rejected: Over-engineering; bump sizes are universal across asset classes.

### 3. No Backward Compatibility (Breaking Change)
Rejected: Would break existing code using `EngineParams.bump_size`.

## Migration Path

1. **Phase 1**: Add new code without breaking changes
   - Add `BumpConfig`
   - Add `bump_config` field (default None)
   - Auto-create `bump_config` from `bump_size` in `__post_init__`

2. **Phase 2**: Update consumers
   - Update `GreeksCalculator` to use `BumpConfig`
   - Add `dividend_rho`

3. **Phase 3**: Deprecation (future)
   - Mark `bump_size` as deprecated
   - Update documentation

## Dependencies

- `util.numerical.constants.Tolerance` - For default bump size constants
- Existing `GreeksCalculator` infrastructure
- Existing pricing environment wrappers (`FlatVolSurface`, `FlatRateCurve`, `ContinuousDividendYield`)
