# Design: Term Structure Support for Convertible Bond Engines

## Overview
This document describes the architectural approach for enabling time-dependent interest rates and volatility in Convertible Bond pricing engines.

## Current Implementation

### PDE Engines (Jump-Diffusion, TF)
```python
# Current: rate/vol queried once at maturity
T = bond.time_to_maturity(valuation_date)
vol = self.pricing_env.get_vol(spot, T)
r = self.pricing_env.rate_curve.get_rate(T)

# Time loop uses constant r, vol
for n in range(N_t - 1, -1, -1):
    A, b = self._build_matrices(S, V, r, q, vol, ...)
```

### Trinomial Engine
```python
# Current: parameters calculated once
T = bond.time_to_maturity(valuation_date)
vol = self.pricing_env.get_vol(spot, T)
r = self.pricing_env.rate_curve.get_rate(T)

tree_params = self._calculate_tree_params(vol, r, q, hazard_rate, dt)

# Backward induction uses constant tree_params
for i in range(n_steps - 1, -1, -1):
    # Uses fixed p_up, p_mid, p_down, discount
```

## Proposed Implementation

### PDE Engines: Time-Dependent Parameters

The PDE engines will query rates and volatility inside the time loop.

**Time convention:** all times `t` are measured in years from the valuation date (the same convention already used for coupon timing and `RateCurve` queries).

#### Rates
Use the curve’s continuously-compounded **forward rate** for the time step:
`r_step = rate_curve.get_forward_rate(t, t + dt)`

This is consistent with discounting between times `t` and `t+dt` using discount factor ratios.

#### Volatility
`PricingEnvironment.get_vol()` takes `(strike, time_to_maturity)` where `time_to_maturity` is also measured in years from valuation date. A PDE needs an effective volatility over each time step. To support piecewise/time-dependent term structure without requiring a new volatility object, derive a per-step effective volatility from implied vols using total variance differences (ATM by default):

- Choose a reference strike for the term structure lookup. For v1, use `strike_ref = spot` (ATM).
- Define total variance `w(τ) = σ_imp(spot, τ)^2 * τ`
- For step `[t, t+dt]`, define:
  `sigma_step(t,t+dt) = sqrt( max(0, w(t+dt) - w(t)) / dt )`
- Edge case: `w(0)=0` (do not call `get_vol(..., 0)` unless supported)

This yields a piecewise-constant instantaneous volatility per time step consistent (at the chosen strike) with the implied variance curve.

```python
# Build grid (unchanged)
dt = T / N_t
S = ...  # stock price grid

# Time stepping - backward from T to 0
for n in range(N_t - 1, -1, -1):
    t = n * dt
    t_next = t + dt  # time at end of step (closer to T)
    
    # Query forward rate from t to t+dt
    r_local = self.pricing_env.rate_curve.get_forward_rate(t, t_next)
    
    # Per-step effective volatility from implied variance (ATM by default)
    vol_local = sigma_step(t, t_next, strike_ref=self.pricing_env.spot)
    
    # Build matrices with time-local parameters
    A, b = self._build_matrices(S, V, r_local, q, vol_local, ...)
```

**Key Points:**
- Use `get_forward_rate(t, t+dt)` for the instantaneous rate appropriate to each step
- Derive `sigma_step(t, t+dt)` from implied vol term structure using total variance differences
- Keep the public engine API unchanged

### Trinomial Engine: Dynamic Probabilities

#### Grid Stability with Time-Varying Volatility

The trinomial tree grid spacing ($dx$) must accommodate the maximum volatility over the bond's life to prevent negative probabilities:

```python
def _calculate_max_vol_for_grid(self, bond: ConvertibleBond) -> float:
    """
    Find maximum volatility over bond life for stable grid construction.
    
    Returns:
        Maximum volatility to use for dx calculation
    """
    T = bond.time_to_maturity(self.pricing_env.valuation_date)
    spot = self.pricing_env.spot
    
    # Sample volatility at multiple time points
    num_samples = max(10, self.params.num_steps // 10)
    times = np.linspace(0.01, T, num_samples)  # Avoid t=0
    
    max_vol = 0.0
    for t in times:
        vol = self.pricing_env.get_vol(spot, t)
        max_vol = max(max_vol, vol)
    
    return max_vol
```

#### Time-Local Transition Probabilities

```python
def _backward_induction(self, bond, stock_tree, dt):
    n_steps = self.params.num_steps
    T = bond.time_to_maturity(...)
    
    # Use max_vol for grid spacing (calculated in price_with_details)
    # But use local parameters for probabilities
    
    for i in range(n_steps - 1, -1, -1):
        t = i * dt
        t_next = t + dt
        
        # Query local rate and volatility
        r_local = self.pricing_env.rate_curve.get_forward_rate(t, t_next)
        vol_local = sigma_step(t, t_next, strike_ref=self.pricing_env.spot)
        
        # Recalculate transition probabilities for this step
        tree_params_local = self._calculate_tree_params(vol_local, r_local, q, hazard_rate, dt)
        
        # Use tree_params_local for this time step
        for j in range(num_nodes):
            # Apply local probabilities
            ...
```

**Key Points:**
- Grid spacing ($dx$) uses `max_vol` for stability
- Transition probabilities recalculated per time step with local `vol` and `r`
- `u`, `d` factors remain constant (determined by `max_vol`) to maintain tree recombination

### Binomial Engine: Warning for Non-Flat Curves

The standard binomial tree cannot easily handle time-dependent volatility without breaking recombination. Instead of attempting a complex modification, we add a warning:

```python
import logging

logger = logging.getLogger(__name__)

def price_with_details(self, bond: ConvertibleBond) -> ConvertibleBondBinomialResult:
    # Check if curves are non-flat
    self._warn_if_non_flat_curves(bond)
    
    # Continue with existing implementation
    ...

def _warn_if_non_flat_curves(self, bond: ConvertibleBond):
    """Log warning if rate curve or vol surface is non-flat."""
    from param.rrf.rate_curve import FlatRateCurve
    from param.vol.vol_surface import FlatVolSurface
    
    rate_curve = self.pricing_env.rate_curve
    vol_surface = self.pricing_env.vol_surface
    
    is_non_flat = False
    if not isinstance(rate_curve, FlatRateCurve):
        is_non_flat = True
    if vol_surface is not None and not isinstance(vol_surface, FlatVolSurface):
        is_non_flat = True
    
    if is_non_flat:
        logger.warning(
            "Binomial GS engine approximates piecewise curves using a flat "
            "rate/vol to maturity. Use PDE or Trinomial engines for better accuracy."
        )
```

## Interface Changes

### `_build_matrices` Signature (No Change Needed)
The current signature already accepts `r` and `vol` as parameters. The change is in how these parameters are supplied (per-step vs once at init).

### New Helper Methods

#### `ConvertibleBondTrinomialEngine._calculate_max_vol_for_grid()`
```python
def _calculate_max_vol_for_grid(self, bond: ConvertibleBond) -> float:
    """Find maximum volatility over bond life for stable grid construction."""
```

#### `ConvertibleBondBinomialEngine._warn_if_non_flat_curves()`
```python
def _warn_if_non_flat_curves(self, bond: ConvertibleBond) -> None:
    """Log warning if rate curve or vol surface is non-flat."""
```

## Backward Compatibility

1. **Flat curves**: When `FlatRateCurve` and `FlatVolSurface` are used, the forward rate equals the spot rate and volatility is constant. Results will be identical to current implementation.

2. **No API changes**: The public interface (`price()`, `price_with_details()`) remains unchanged.

3. **No new dependencies**: Only uses existing `get_forward_rate()` and `get_vol()` methods.

## Testing Strategy

Create `test/test_convertible_bond_term_structure.py` with scenarios:

1. **Flat vs Stepped Rate Curve**:
   - Flat: 5% constant
   - Stepped: 1% for first half, 9% for second half (avg ~5%)
   - Expectation: Different prices for PDE/Trinomial; similar prices for Binomial

2. **Flat vs Stepped Volatility Surface** (if applicable):
   - Flat: 30% constant
   - Stepped: 40% short-term, 20% long-term
   - Expectation: Different prices showing vol term structure effect

3. **Binomial Warning**:
   - Use stepped curve
   - Assert warning is logged

## Diagram: Time-Stepping with Local Parameters

```
T=0         T=dt        T=2dt       ...        T=maturity
 |           |           |                        |
 +--- step 0 ---+ step 1 ---+ step 2 ---...---+ step N-1
 |           |           |                        |
 r(0,dt)     r(dt,2dt)   r(2dt,3dt)              r((N-1)dt,T)
 vol(0)      vol(dt)     vol(2dt)                vol((N-1)dt)
```

Each step uses:
- Forward rate from current time to next time step
- Local volatility at current time
