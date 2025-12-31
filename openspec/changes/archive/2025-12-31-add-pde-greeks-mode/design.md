# Design: PDE Greeks Calculation Mode

## Overview

This design adds a configurable mode to `GreeksCalculator` for controlling how delta and gamma are calculated when using PDE engines. The key insight is that PDE engines compute Greeks directly from the solution grid, which is more accurate and efficient than the bump method.

## Problem Analysis

### Current Behavior

```python
# GreeksCalculator always uses bump method
calc = GreeksCalculator()
greeks = calc.calculate_numerical_greeks(product, env, pde_engine)
# Delta/gamma computed via: (V(S+h) - V(S-h)) / (2h)
# Requires re-solving PDE 2 additional times
```

### Issues with Bump Method for PDE

1. **Numerical Instability**: Re-solving PDE with bumped spot causes:
   - Grid may not align with bumped spot values
   - Interpolation errors between grid points
   - Can produce negative gamma for call options (mathematically impossible)

2. **Inefficiency**: Requires 2 additional PDE solves:
   - Base price: 1 PDE solve
   - Bump method: +2 PDE solves (up and down)
   - Total: 3 PDE solves

3. **Inconsistency**: Greeks may not be consistent with the PDE solution:
   - Grid Greeks use exact finite differences on the solution surface
   - Bump Greeks use finite differences on re-solved PDEs

### PDE Grid-Based Greeks

PDE solvers compute the option value surface V(S,t) on a grid. Greeks are extracted via finite differences on this grid:

```python
# In log-price space: x = ln(S)
delta = (1/S) * dV/dx
gamma = (1/S²) * (d²V/dx² - dV/dx)
```

Advantages:
- **Single PDE solve**: No re-pricing needed
- **Exact consistency**: Greeks are from the same solution used for pricing
- **Better accuracy**: Finite differences on smooth solution surface

## Design

### 1. GreeksCalculationMode Enum

```python
class GreeksCalculationMode(Enum):
    """Mode for calculating delta/gamma in GreeksCalculator."""
    ENGINE = "engine"  # Use engine.calculate_greeks() if available
    BUMP = "bump"      # Use finite difference bump method
    AUTO = "auto"      # Use engine method for PDE, bump otherwise
```

**Rationale**:
- `BUMP`: Default for backward compatibility
- `ENGINE`: Explicit opt-in to engine method
- `AUTO`: Smart choice (PDE → engine, others → bump)

### 2. Engine Type Detection

```python
class BaseEngine(ABC):
    engine_type: EngineType = EngineType.ANALYTICAL
```

**Why class attribute instead of instance?**
- All engines of a class have the same type
- No need to store per-instance
- Enables `isinstance`-like checking without isinstance

### 3. Decision Logic

```python
def _should_use_engine_greeks(self, engine: BaseEngine) -> bool:
    """Check if engine's calculate_greeks() should be used."""
    if self.greeks_mode == GreeksCalculationMode.BUMP:
        return False
    if self.greeks_mode == GreeksCalculationMode.ENGINE:
        return True
    # AUTO mode: use for PDE engines
    return getattr(engine, 'engine_type', None) == EngineType.PDE
```

**Why `getattr` with default?**
- Defensive programming for engines without `engine_type`
- Allows graceful degradation
- Future-proof for external engine implementations

### 4. PDEEngine Delegation Fix

**Problem**: `PDEEngine` didn't override `calculate_greeks()`

**Before**:
```python
engine = PDEEngine()
engine.calculate_greeks(product, env)  # Used BaseEngine.bump method
```

**After**:
```python
def calculate_greeks(self, product, pricing_env):
    solver = self._get_solver(product)
    return solver.calculate_greeks(product, pricing_env)
```

## Accuracy Comparison

For ATM call option (S=100, K=100, T=1, σ=20%, r=5%):

| Method | Delta | Gamma | PDE Solves |
|--------|-------|-------|------------|
| Bump (h=1%) | 0.624 | -3.83* | 3 |
| PDE Grid | 0.637 | 0.019 | 1 |
| BS Analytical | 0.636 | 0.019 | 0 |

*Bump method produces negative gamma due to numerical issues

**Conclusion**: PDE grid Greeks are:
- More accurate (matches analytical)
- More efficient (1 solve vs 3)
- Numerically stable (positive gamma)

## Backward Compatibility

```python
# Old code continues to work
calc = GreeksCalculator()  # Defaults to BUMP mode
greeks = calc.calculate_numerical_greeks(product, env, engine)

# New code can opt in
calc = GreeksCalculator(greeks_mode=GreeksCalculationMode.AUTO)
greeks = calc.calculate_numerical_greeks(product, env, pde_engine)
```

## Future Enhancements

1. **Other Greeks**: Extend pattern to vega (volatility surface PDE)
2. **Engine Methods**: Add `calculate_greeks()` to other engines
3. **Performance**: Cache PDE solution for multiple Greeks calculations
4. **Adaptive**: Auto-detect best method based on product/engine combination
