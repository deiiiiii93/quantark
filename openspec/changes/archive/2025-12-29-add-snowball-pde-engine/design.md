# Design: Snowball PDE Engine (Two-Surface Method)

## 1. Architectural Overview

The Snowball PDE engine uses a **Two-Surface** approach to handle the path-dependent knock-in feature while remaining within the PDE framework.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SnowballPDESolver                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────┐    ┌──────────────────────┐                   │
│  │   V0 Surface         │    │   V1 Surface         │                   │
│  │   (Not Knocked-In)   │    │   (Knocked-In)       │                   │
│  │                      │    │                      │                   │
│  │  • Standard rebate   │    │  • Downside payoff   │                   │
│  │  • KO coupon payoffs │    │  • Participation     │                   │
│  │  • V0 terminal       │    │  • Protection floor  │                   │
│  └──────────────────────┘    └──────────────────────┘                   │
│            │                          ▲                                  │
│            │   KI Barrier Hit         │                                  │
│            └──────────────────────────┘                                  │
│                 V0(S,t) = V1(S,t)                                        │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      Shared Infrastructure                        │   │
│  │  • Non-uniform spatial grid (spot, strike, barriers)             │   │
│  │  • Event-aligned time grid (KO and KI observation dates)         │   │
│  │  • Crank-Nicolson time stepping                                  │   │
│  │  • Rannacher smoothing at maturity and observation events        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. Mathematical Formulation

### 2.1 State Surfaces

Both surfaces satisfy the Black-Scholes PDE inside the domain:

$$\frac{\partial V}{\partial t} + (r - q) S \frac{\partial V}{\partial S} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} - r V = 0$$

Where:
- $r$: Risk-free rate
- $q$: Dividend yield
- $\sigma$: Volatility

### 2.2 Terminal Conditions (t = T)

**V1 Surface (Knocked-In):**
For standard snowball (short put exposure):
$$V_1(S, T) = \text{Principal} + \text{Participation} \times \min\left(\frac{S_T - K}{S_0}, 0\right) \times N$$

Subject to protection floor based on `protection_type`.

**V0 Surface (Not Knocked-In):**
$$V_0(S, T) = \text{Principal} + \text{Rebate}$$

Or call-style rebate if `call_rebate_enabled`.

### 2.3 Barrier Jump Conditions

**At KO observation times** (when $S \geq B_{KO}$ for standard snowball):
$$V_0(S, t) = V_1(S, t) = \text{KO\_Payoff}(t)$$

The KO payoff is calculated using `product.resolve_ko_observations()`.

**At KI observation times** (when $S \leq B_{KI}$ for standard snowball):
$$V_0(S, t) \leftarrow V_1(S, t)$$

The "Not Knocked-In" value becomes the "Knocked-In" value at that spot.

### 2.4 Continuous KI Handling

For continuous KI monitoring, the jump condition is applied at every time step within the barrier region, not just at discrete observation dates.

## 3. Class Design

### 3.1 SnowballPDESolver

```python
class SnowballPDESolver(BasePDESolver):
    """
    PDE solver for Snowball (autocallable) options using 2-surface method.
    
    Maintains two price grids:
    - grid_v0: Value surface for "not knocked-in" state
    - grid_v1: Value surface for "knocked-in" state
    """
    
    def __init__(self, params: Optional[PDEParams] = None):
        super().__init__(params)
        self._grid_v0: Optional[np.ndarray] = None
        self._grid_v1: Optional[np.ndarray] = None
        self._ko_observation_indices: Dict[int, ResolvedObservationRecord] = {}
        self._ki_observation_indices: Set[int] = set()
        self._ki_continuous: bool = False
        self._ki_barrier: float = 0.0
        
    def price(self, product: SnowballOption, pricing_env: PricingEnvironment) -> float:
        """Price snowball using 2-surface PDE method."""
        ...
        
    def set_terminal_condition_v0(self, ...):
        """Set terminal payoff for not-knocked-in state."""
        ...
        
    def set_terminal_condition_v1(self, ...):
        """Set terminal payoff for knocked-in state."""
        ...
        
    def _apply_ko_jump(self, t_idx: int, ...):
        """Apply KO payoff to both surfaces."""
        ...
        
    def _apply_ki_jump(self, t_idx: int, ...):
        """Apply KI jump: V0 = V1 in breached region."""
        ...
```

### 3.2 Integration with PDEEngine

The unified `PDEEngine` will be extended to dispatch `SnowballOption` to `SnowballPDESolver`:

```python
# In PDEEngine.price()
if isinstance(product, SnowballOption):
    solver = SnowballPDESolver(self.params)
    return solver.price(product, pricing_env)
```

## 4. Grid Construction

### 4.1 Spatial Grid

Non-uniform grid concentrated at critical points:
- Current spot price
- Strike price (K)
- KI barrier ($B_{KI}$)
- All KO barriers ($B_{KO,i}$)

Use Tavella-Randall or sinh transformation for smooth concentration.

### 4.2 Time Grid

- Uniform base stepping with $\Delta t$
- **Mandatory alignment** with all observation dates:
  - KO observation times from `resolve_ko_observations()`
  - KI observation times from `resolve_ki_observations()` (if discrete)
- Apply Rannacher smoothing (2-4 implicit Euler steps) after observation events

## 5. Algorithm: Backward Induction

```
1. Initialize grids:
   - Build spatial grid with critical points
   - Build time grid aligned with observations
   
2. Set terminal conditions (t = T):
   - grid_v0[:, -1] = V0 terminal payoff (rebate or call rebate)
   - grid_v1[:, -1] = V1 terminal payoff (downside with protection)

3. For t_idx from N_t-1 down to 0:
   a. Step V0 backward one time step (Crank-Nicolson)
   b. Step V1 backward one time step (Crank-Nicolson)
   
   c. If t_idx is KO observation:
      - Apply KO jump to both surfaces in breached region
      
   d. If t_idx is KI observation OR continuous KI:
      - Apply KI jump: V0 = V1 in breached region
      
   e. Apply boundary conditions

4. Interpolate final price:
   - If already knocked-in: return V1(spot, 0)
   - Otherwise: return V0(spot, 0)
```

## 6. Handling Snowball-Specific Features

### 6.1 `disable_ko_after_ki`

When `disable_ko_after_ki=True`, the KO barriers are ignored after KI occurs. In PDE terms:
- This is automatically handled because V1 surface does not check KO barriers
- Only V0 surface applies KO jumps
- After KI jump (V0 → V1), subsequent KO checks on V0 are irrelevant

### 6.2 INSTANT vs EXPIRY Coupon Payment

- **INSTANT**: KO payoff is valued at KO time (discounted from settlement_time if different)
- **EXPIRY**: KO payoff is discounted to maturity settlement

This is handled via `resolve_ko_observations()` which provides `settlement_time` per observation.

### 6.3 Airbag Structure

Airbag affects the V1 terminal condition only:
- For spot below airbag_barrier: use reduced participation rate and possibly different strike
- Delegate to `product.get_maturity_payoff_v1(spot)` which handles airbag logic

### 6.4 Annualized Accruals

The product's `resolve_ko_observations()` method handles annualized coupon calculations, returning pre-computed payoffs. The PDE solver uses these directly.

## 7. Boundary Conditions

### 7.1 Lower Boundary (S → 0)

**V1 Surface**: Deep ITM put value (discounted worst-case loss)
$$V_1(0, t) = \text{Principal} + \text{Participation} \times \left(-\frac{K}{S_0}\right) \times N \times e^{-r(T-t)}$$

**V0 Surface**: May transition to V1 if continuous KI, otherwise discounted principal

### 7.2 Upper Boundary (S → ∞)

**Both Surfaces**: If above all KO barriers, value is KO payoff at that time step.
Otherwise, deep OTM value (principal for V0, principal for V1 as downside is zero).

## 8. Performance Considerations

### 8.1 Default Grid Sizes

| Observation Count | Spatial Points | Time Steps per Interval |
|-------------------|----------------|-------------------------|
| ≤ 12 (monthly)    | 200            | 4 per interval          |
| 13-52 (weekly)    | 300            | 2 per interval          |
| > 52              | 400            | 1 per interval          |

### 8.2 Memory Usage

Two grids instead of one: approximately 2x memory of single-surface PDE.
For 400 × 200 grid: ~1.3 MB per grid × 2 = 2.6 MB total.

### 8.3 Computation Time

Expected: 200-500ms for typical monthly observation snowball.
Tridiagonal solver is O(n) per time step.

## 9. Error Handling

| Condition | Error |
|-----------|-------|
| Non-scalar KI barrier with continuous monitoring | `ValidationError` |
| Unsupported product type | `PricingError` |
| Missing pricing environment | `ValidationError` |
| Already expired product | Return intrinsic value |

## 10. Testing Strategy

1. **Unit Tests**: Terminal conditions, barrier jumps, boundary conditions
2. **Integration Tests**: Compare with MC for various snowball configurations
3. **Convergence Tests**: Grid refinement should reduce error monotonically
4. **Edge Cases**: Near-barrier spot, near-maturity, deep ITM/OTM
5. **Greeks Tests**: Delta/gamma smoothness, vega sensitivity
