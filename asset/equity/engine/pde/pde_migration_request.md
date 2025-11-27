# PDE Pricing Engine Migration Request

## Overview

This document provides a comprehensive technical specification for implementing a PDE (Partial Differential Equation) pricing engine for equity derivatives using finite difference methods. It is intended as a blueprint for migrating the PDE framework to a new project.

The PDE approach solves the Black-Scholes equation numerically, providing accurate pricing for complex derivatives that lack closed-form solutions, such as barrier options, American options, and autocallable structured products.

---

## Table of Contents

1. [Mathematical Foundation](#1-mathematical-foundation)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Components](#3-core-components)
4. [Specialized Solvers](#4-specialized-solvers)
5. [Numerical Techniques](#5-numerical-techniques)
6. [Implementation Guidelines](#6-implementation-guidelines)
7. [Structured Product Pricing Approaches](#7-structured-product-pricing-approaches)
8. [Product Interface Requirements](#8-product-interface-requirements)
9. [Testing and Validation](#9-testing-and-validation)
10. [References](#10-references)

---

## 1. Mathematical Foundation

### 1.1 Black-Scholes PDE

The Black-Scholes PDE for a derivative V(S, t) with underlying asset price S and time t:

```
∂V/∂t + (r - q)S·∂V/∂S + ½σ²S²·∂²V/∂S² - rV = 0
```

Where:
- `r` = risk-free rate
- `q` = dividend yield  
- `σ` = volatility
- `S` = spot price
- `t` = time

### 1.2 Log-Price Transformation

For numerical stability, transform to log-price space `x = ln(S)`:

```
∂V/∂t + (r - q - ½σ²)·∂V/∂x + ½σ²·∂²V/∂x² - rV = 0
```

This transformation:
- Converts multiplicative dynamics to additive
- Ensures uniform grid spacing in relative price terms
- Improves numerical stability for extreme price scenarios

### 1.3 Backward-in-Time Integration

The PDE is solved backward in time from maturity (T) to valuation date (t=0):
- Terminal condition: payoff function at maturity
- Boundary conditions: option behavior at extreme prices
- Each time step propagates information backward

---

## 2. Architecture Overview

### 2.1 Component Hierarchy

```
PDEConfig                    # Configuration container
    │
    ▼
BasePDESolver               # Abstract base with common infrastructure
    │
    ├── EuropeanOptionPDESolver
    ├── AmericanOptionPDESolver
    ├── BarrierOptionPDESolver
    ├── DoubleBarrierOptionPDESolver
    ├── OneTouchOptionPDESolver
    ├── DoubleOneTouchOptionPDESolver
    └── [Structured Product Solvers]*
    
TimeGrid                    # Time discretization utility
NonUniformGrid              # Spatial discretization utility
```

*Note: See Section 7 for approaches to pricing structured products (decomposition vs two-surface PDE).

### 2.2 Pricing Flow

1. **Initialize**: Create solver with product and configuration
2. **Build Grids**: Construct spatial and temporal discretization
3. **Set Terminal Condition**: Apply payoff at maturity
4. **Set Boundary Conditions**: Define behavior at domain edges
5. **Calculate Coefficients**: Build finite difference coefficients
6. **Setup Matrices**: Construct sparse matrices for time-stepping
7. **Time-Stepping**: Solve backward in time
8. **Interpolate**: Extract price at current spot
9. **Calculate Greeks**: Compute sensitivities

---

## 3. Core Components

### 3.1 PDEConfig

Configuration dataclass encapsulating all PDE solver parameters.

```python
@dataclass
class PDEConfig:
    # Spatial Grid Configuration
    x_grid_resolution: int = 400        # Number of spatial points
    Smin: float = 0.0                   # Lower bound (0 = auto-calculate)
    Smax: float = 0.0                   # Upper bound (0 = auto-calculate)
    use_adaptive_grid: bool = False     # Enable Tavella-Randall grid
    parent_grid: Optional[Tuple] = None # Reuse existing grid
    
    # Temporal Grid Configuration
    t_grid_resolution: Optional[int] = None  # Number of time steps
    t_grid_type: str = "uniform"        # "uniform", "graded", "event_clustered"
    event_times: Optional[List[float]] = None  # Observation times (year fractions)
    grade_exponent: float = 2.0         # Power for graded grid
    min_dt: Optional[float] = None      # Minimum time step
    max_dt: Optional[float] = None      # Maximum time step
    default_steps_per_day: int = 4      # Fallback steps per business day
    
    # Numerical Scheme Configuration
    use_rannacher: bool = True          # Enable Rannacher smoothing
    rannacher_steps: int = 1            # Number of smoothing steps
    theta: float = 0.5                  # 0.5=Crank-Nicolson, 1.0=Backward Euler
    
    # Spatial-Temporal Coupling
    use_cfl_coupling: bool = False      # Auto-adjust Ns based on min(dt)
    cfl_safety_factor: float = 0.5      # CFL stability factor
    max_cfl_adjusted_resolution: int = 1000  # Cap on adjusted Ns
```

### 3.2 TimeGrid

Static utility class for generating time discretizations.

#### 3.2.1 TimeGridParameters

```python
@dataclass
class TimeGridParameters:
    tau: float                          # Total time to maturity (years)
    num_steps: int                      # Target number of steps
    event_times: Optional[List[float]] = None  # Event times to align with
    method: str = "uniform"             # Grid generation method
    grade_exponent: float = 2.0         # For graded grids
    min_dt: Optional[float] = None      # Minimum step size
    max_dt: Optional[float] = None      # Maximum step size
```

#### 3.2.2 Grid Methods

**Uniform Grid**: Constant time steps
```
t_i = tau * (i / N)     for i = 0, 1, ..., N
dt = tau / N
```

**Graded Grid**: Power-law clustering near maturity
```
t_i = tau * (i / N)^p   where p = grade_exponent
```
- Higher exponent = more clustering near maturity
- Useful for European options with payoff discontinuities

**Event-Clustered Grid**: Exact alignment with observation times
- Ensures event times appear exactly in the grid
- Allocates steps proportionally between events
- Essential for discretely-observed barriers and autocallables

```python
@classmethod
def build(cls, params: TimeGridParameters) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        TVec: Time points from 0 to tau
        dt_vec: Time step sizes (length = len(TVec) - 1)
    """
```

### 3.3 NonUniformGrid

Spatial grid generation with concentration near critical prices.

#### 3.3.1 GridParameters

```python
@dataclass
class GridParameters:
    lower_bound: float                  # Log-price lower bound
    upper_bound: float                  # Log-price upper bound
    num_points: int                     # Number of grid points
    critical_points: Optional[List[float]] = None  # Strike prices, barriers
    concentration_parameter: Optional[float] = None  # Beta parameter
```

#### 3.3.2 Tavella-Randall Transformation

For a critical point K (e.g., strike), the transformation:

```
x_i = K + β * sinh(c₁(1 - i/N) + c₂(i/N))

where:
    a₁ = (L - K) / β
    a₂ = (H - K) / β
    c₁ = asinh(a₁)
    c₂ = asinh(a₂)
```

Key properties:
- Concentrates points near K
- Maintains smooth grid transitions
- β controls concentration strength (auto-calculated if not provided)

For multiple critical points, use piecewise segments with the `tavella_randall_multi` method.

### 3.4 BasePDESolver

Abstract base class providing common PDE solving infrastructure.

#### 3.4.1 Key Attributes

```python
class BasePDESolver:
    # From option product
    option: BaseOptionProduct
    rrf: float          # Risk-free rate
    div: float          # Dividend yield
    vol: float          # Volatility
    spot: float         # Current spot price
    tau: float          # Time to maturity (years)
    
    # Grid parameters
    Ns: int             # Spatial resolution
    Nt: int             # Temporal resolution
    grid_params: Dict   # Contains SVec, TVec, dt_vec, dS
    
    # Solution storage
    grid: np.ndarray    # Full solution grid [Ns+1, Nt+1]
    value: float        # Computed option price
    delta: float        # First-order Greek
    gamma: float        # Second-order Greek
    
    # Internal caches
    _A: sp.spmatrix     # Spatial operator matrix
    _matrix_cache: Dict # LU factorizations keyed by (dt, theta)
```

#### 3.4.2 Abstract Methods (Must Implement)

```python
def set_terminal_condition(self) -> None:
    """Set payoff at maturity in self.grid[:, -1]"""
    raise NotImplementedError

def set_boundary_conditions(self) -> None:
    """Set boundary values in self.grid[0, :] and self.grid[-1, :]"""
    raise NotImplementedError

def calculate_intrinsic_value(self) -> float:
    """Return intrinsic value when bus_days = 0"""
    raise NotImplementedError
```

#### 3.4.3 Key Methods

**Grid Construction**:
```python
def _build_solve_grid(self) -> Dict:
    """Build uniform spatial grid in log-price space"""
    
def _build_solve_adaptive_grid(self) -> Dict:
    """Build Tavella-Randall non-uniform grid"""
    
def _build_time_grid(self, grid_params: Dict) -> None:
    """Build temporal grid based on PDEConfig settings"""
```

**Coefficient Calculation**:
```python
def calculate_coefficients(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate finite difference coefficients for log-price PDE
    
    Returns:
        l: Lower diagonal (drift - diffusion)
        c: Center diagonal (-diffusion - discount)
        u: Upper diagonal (drift + diffusion)
    """
    drift = (self.rrf - self.div - 0.5 * self.vol**2) / dS
    diffusion_sq = (self.vol / dS) ** 2
    
    l = 0.5 * (diffusion_sq - drift)
    c = -diffusion_sq - self.rrf
    u = 0.5 * (diffusion_sq + drift)
    
    return l, c, u
```

**Matrix Setup**:
```python
def setup_matrices(self, l, c, u, theta=0.5) -> Tuple:
    """
    Build sparse matrices for Crank-Nicolson scheme
    
    Scheme: (I - θ·dt·A)·V^n = (I + (1-θ)·dt·A)·V^{n+1}
    
    Returns:
        M1: Right-hand side matrix
        M2: Left-hand side matrix  
        M2M: LU factorization of M2
    """
```

**Time-Stepping**:
```python
def time_stepping(self, M1, M2M, l, u, theta=0.5) -> None:
    """
    Backward-in-time integration with:
    - Variable time step support (uses dt_vec)
    - Rannacher smoothing (theta=1 for first steps)
    - Matrix caching for different (dt, theta) pairs
    """
```

**Greeks Calculation**:
```python
def calculate_greeks(self, surface, spot_log, h=0.0001) -> None:
    """
    Finite difference Greeks in log-space, converted to price-space
    
    delta = (1/S) * dV/dx
    gamma = (1/S²) * (d²V/dx² - dV/dx)
    """
```

---

## 4. Specialized Solvers

### 4.1 EuropeanOptionPDESolver

Simplest implementation for vanilla European calls/puts.

**Terminal Condition**:
```python
# Call: max(0, exp(x) - K)
# Put:  max(0, K - exp(x))
self.grid[:, -1] = np.maximum(0, np.exp(SVec) - K)  # for call
```

**Boundary Conditions**:
```python
# Call: V(0) = 0, V(Smax) = Smax - K*exp(-r*t)
# Put:  V(0) = K*exp(-r*t), V(Smax) = 0
```

### 4.2 AmericanOptionPDESolver

Adds early exercise constraint at each time step.

**Modification to time_stepping**:
```python
# After solving linear system:
self.grid[1:-1, j] = M2M.solve(U)

# Apply early exercise constraint:
intrinsic = np.maximum(0, np.exp(SVec) - K)  # for call
self.grid[:, j] = np.maximum(self.grid[:, j], intrinsic)
```

### 4.3 BarrierOptionPDESolver

Handles single barrier options (knock-in/knock-out, up/down).

**Key Features**:
- Barrier type detection (up/down, in/out)
- Continuous vs discrete observation
- Event time injection for discrete barriers

**Discrete Observation Handling**:
```python
def time_stepping(self, M1, M2M, l, u, theta=0.5):
    for j in reversed(range(self.Nt)):
        # Check if this is an observation time
        if j + 1 in obs_indices:
            barrier = self.obs_barriers[obs_index]
            spots = np.exp(SVec)
            
            if self.is_out:
                # Knock-out: zero where barrier hit
                hits = spots >= barrier if self.is_up else spots <= barrier
                self.grid[hits, j + 1] = 0
            else:
                # Knock-in: payoff where barrier hit
                self.grid[hits, j + 1] = payoff[hits]
        
        # Standard time step
        # ...
```

### 4.4 DoubleBarrierOptionPDESolver

Extends barrier solver for two barriers (upper and lower).

**Terminal Condition**: Zero if outside corridor
**Boundary Conditions**: Zero at both barriers
**Time-Stepping**: Check both barriers at observation times

### 4.5 OneTouchOptionPDESolver

Binary-style option that pays rebate on barrier touch.

**Key Differences**:
- Rebate array indexed by observation time
- No-touch rebate at maturity if never touched
- Two-surface method for knock-in style

---

## 5. Numerical Techniques

### 5.1 Rannacher Smoothing

**Problem**: Crank-Nicolson produces oscillations near payoff discontinuities.

**Solution**: Use backward Euler (theta=1.0) for first few time steps.

```python
for j in reversed(range(self.Nt)):
    steps_from_maturity = self.Nt - j
    
    if self.pde_config.use_rannacher and steps_from_maturity <= self.pde_config.rannacher_steps:
        step_theta = 1.0  # Backward Euler
    else:
        step_theta = theta  # Crank-Nicolson (0.5)
```

**Configuration**:
- `use_rannacher: bool = True` - Enable smoothing
- `rannacher_steps: int = 1` - Number of BE steps (1-3 typical)

### 5.2 CFL Spatial-Temporal Coupling

**CFL Condition** for numerical stability:
```
dt * σ² / (dS)² ≤ C
```

When enabled, automatically adjusts spatial resolution based on minimum time step:

```python
def _apply_cfl_coupling(self, grid_params):
    dt_min = np.min(dt_vec)
    
    # Required Ns for stability
    required_Ns = int(np.ceil(
        S_range_log / (sigma * np.sqrt(dt_min / safety_factor))
    ))
    
    if required_Ns > self.Ns:
        self.Ns = min(required_Ns, max_resolution)
        # Warn and rebuild spatial grid
```

### 5.3 Matrix Caching

Cache LU factorizations to avoid repeated decomposition:

```python
def _get_matrices_for_dt(self, dt: float, theta: float):
    key = (round(dt, 12), float(theta))
    
    if key in self._matrix_cache:
        return self._matrix_cache[key]
    
    # Build and cache
    M1 = I + (1 - theta) * dt * A
    M2 = I - theta * dt * A
    M2M = splu(M2)
    
    self._matrix_cache[key] = (M1, M2M)
    return M1, M2M
```

### 5.4 Event Time Injection

Automatically inject observation times into the time grid:

```python
def _collect_event_times_from_option(self) -> Optional[np.ndarray]:
    """Infer event times from product attributes"""
    candidates = []
    
    for attr in ('obs_bus_dates', 'knockout_obs_bus_dates', 'knockin_obs_dates'):
        if hasattr(self.option, attr):
            arr = np.asarray(getattr(self.option, attr))
            candidates.extend((arr / bus_days_in_year).tolist())
    
    return np.unique(candidates) if candidates else None
```

---

## 6. Implementation Guidelines

### 6.1 Grid Boundary Calculation

When boundaries not specified, use volatility-based expansion:

```python
zeta = 3.0  # Standard deviations
upper_limit = zeta * vol * sqrt(tau)
exp_change = (rrf - div) * tau

Smin = spot * exp(-upper_limit + exp_change)
Smax = spot * exp(+upper_limit + exp_change)

# Ensure strike is within bounds
Smin = min(Smin, strike)
Smax = max(Smax, strike)
```

### 6.2 Resolution Guidelines

| Product Type | Spatial (Ns) | Temporal (Nt) |
|--------------|--------------|---------------|
| European | 100-200 | 50-150 |
| American | 200-300 | 100-200 |
| Single Barrier | 200-400 | 150-300 |
| Double Barrier | 400-600 | 200-400 |
| Structured Products | 300-500 | per observation |

### 6.3 Sparse Matrix Usage

Use scipy.sparse for efficiency:

```python
import scipy.sparse as sp
import scipy.sparse.linalg as spl

# Build tridiagonal operator
A = sp.diags([l[1:], c, u[:-1]], [-1, 0, 1], format="csc")

# LU factorization
M2M = spl.splu(M2)

# Solve
V = M2M.solve(U)
```

---

## 7. Structured Product Pricing Approaches

Complex structured products like Phoenix and Snowball options can be priced using different approaches. This section presents two methods with their trade-offs.

### 7.1 Approach 1: Product Decomposition

**Concept**: Decompose a complex structured product into a portfolio of primitive options, price each component separately, and combine the results.

**Example - Phoenix Option Decomposition**:

A Phoenix option can be decomposed into:
1. **OneTouch** - captures knockout interest (rebate paid on knockout)
2. **BarrierOption** - captures knockin interest (put payoff if knocked in)
3. **DoubleBarrierOption** - compensates knockin value when knockout occurs

```
Price(Phoenix) = V_OneTouch - V_BarrierOption + V_DoubleBarrier
```

**Example - Snowball Option Decomposition**:

A Snowball option can be decomposed into:
1. **OneTouch** - knockout component with coupon rebates
2. **BarrierOption** - knockin put component
3. **DoubleBarrierOption** - compensation for knockin when knocked out
4. **DoubleNoTouch** - no-touch bonus at maturity

```
Price(Snowball) = V_OneTouch - V_BarrierOption + V_DoubleBarrier + V_DoubleNoTouch
```

### 7.2 Implementing Product Decomposition

**Architecture**:
```
StructuredProductDecomposer
    │
    ├── decompose_phoenix(phoenix: Phoenix) -> List[PrimitiveProduct]
    ├── decompose_snowball(snowball: Snowball) -> List[PrimitiveProduct]
    └── decompose(product: StructuredProduct) -> List[PrimitiveProduct]

PrimitiveProduct (Abstract)
    │
    ├── OneTouch
    ├── NoTouch
    ├── KnockOutOption
    ├── KnockInOption
    ├── DoubleKnockOutOption
    └── DigitalOption
```

**Decomposer Interface**:
```python
@dataclass
class DecomposedComponent:
    """A primitive product with its weight in the composite price"""
    product: BaseOptionProduct
    weight: float  # +1.0 or -1.0 for addition/subtraction
    description: str  # e.g., "knockout_interest", "knockin_compensation"


class StructuredProductDecomposer:
    """
    Decomposes complex structured products into primitive options.
    
    This separation allows:
    - PDE solvers to focus on numerical methods
    - Reuse of decomposition logic across pricing engines (PDE, MC, etc.)
    - Easier testing and validation of financial logic
    """
    
    @staticmethod
    def decompose_phoenix(phoenix: Phoenix) -> List[DecomposedComponent]:
        """
        Decompose Phoenix into:
        1. OneTouch - knockout interest (rebate on KO)
        2. BarrierOption - knockin interest (put payoff if KI)
        3. DoubleBarrierOption - compensation (subtract KI value when KO)
        
        Price = V_OneTouch - V_BarrierOption + V_DoubleBarrier
        """
        components = []
        
        # OneTouch for knockout interest
        components.append(DecomposedComponent(
            product=OneTouch(...),
            weight=+1.0,
            description="knockout_interest"
        ))
        
        # BarrierOption for knockin interest
        components.append(DecomposedComponent(
            product=KnockOutOption(...),
            weight=-1.0,
            description="knockin_interest"
        ))
        
        # DoubleBarrier for compensation
        components.append(DecomposedComponent(
            product=DoubleKnockOutOption(...),
            weight=+1.0,
            description="knockin_compensation"
        ))
        
        return components
    
    @staticmethod
    def decompose_snowball(snowball: Snowball) -> List[DecomposedComponent]:
        """Similar decomposition for Snowball products"""
        # ...
```

**Simplified PDE Solver**:
```python
class PhoenixPDESolver:
    """Clean solver that focuses on numerical methods"""
    
    def __init__(self, phoenix: Phoenix, pde_config: PDEConfig):
        self.phoenix = phoenix
        self.pde_config = pde_config
        
        # Decomposition happens OUTSIDE the solver
        self.components = StructuredProductDecomposer.decompose_phoenix(phoenix)
        
        # Create solvers for primitive components
        self.component_solvers = []
        for comp in self.components:
            solver = self._create_solver_for_product(comp.product)
            self.component_solvers.append((solver, comp.weight))
    
    def price(self) -> float:
        total = 0.0
        for solver, weight in self.component_solvers:
            total += weight * solver.price()
        return total
```

### 7.3 Advantages of Decomposition Approach

1. **Modularity**: Decomposer handles financial logic; solvers focus on numerics
2. **Testability**: Test decomposition independently of pricing
3. **Reusability**: Same decomposition works for PDE, Monte Carlo, tree methods
4. **Maintainability**: Clear separation of concerns
5. **Extensibility**: Add new structured products without modifying solvers
6. **Validation**: Each primitive component can be validated against analytical formulas

### 7.4 Approach 2: Two-Surface PDE Method

Instead of decomposing structured products into primitive options, you can directly solve a **multi-surface PDE** that tracks different states simultaneously.

**Concept**: For products with path-dependent states (e.g., knocked-in vs not-knocked-in), maintain multiple value surfaces that evolve together:

```
V0[S, t] = Value in "not-knocked-in" state
V1[S, t] = Value in "knocked-in" state
```

**Two-Surface PDE for Phoenix/Snowball**:

```python
class TwoSurfacePDESolver(BasePDESolver):
    """
    Maintains two value surfaces:
    - V0: not-knocked-in state
    - V1: knocked-in state
    """
    
    def price_pde(self):
        # Allocate two surfaces
        V0 = np.zeros((self.Ns + 1, self.Nt + 1))  # Not-KI state
        V1 = np.zeros((self.Ns + 1, self.Nt + 1))  # KI state
        
        # Set terminal conditions for both surfaces
        self.set_terminal_condition(V0, V1, S_levels)
        
        # Set boundary conditions for both surfaces
        self.set_boundary_conditions(V0, V1, dfVec, S_levels)
        
        # Backward induction with state transitions
        V0, V1 = self.time_stepping_two_surfaces(V0, V1, S_levels)
        
        # Return appropriate surface based on current state
        self.grid = V1 if self.is_knocked_in else V0
        return self.interpolate_to_spot()
    
    def time_stepping_two_surfaces(self, V0, V1, S_levels):
        """Advance both surfaces and apply state transitions at events."""
        for j in reversed(range(self.Nt)):
            # Standard CN step for both surfaces
            V0[1:-1, j] = self._cn_step(V0[:, j+1], V0[:, j])
            V1[1:-1, j] = self._cn_step(V1[:, j+1], V1[:, j])
            
            # At knockout observation times: apply KO jumps
            if j in self.KO_idx:
                self._apply_knockout_jump(V0, V1, j, S_levels)
            
            # At knockin observation times: transition V0 -> V1 where KI triggered
            if j in self.KI_idx:
                self._apply_knockin_transition(V0, V1, j, S_levels)
        
        return V0, V1
    
    def _apply_knockin_transition(self, V0, V1, j, S_levels):
        """Where S <= KI barrier, V0 transitions to V1."""
        KI_barrier = self.knockin_price_at(j)
        mask_ki = S_levels <= KI_barrier
        V0[mask_ki, j] = V1[mask_ki, j]  # Adopt knocked-in value
```

**Terminal Conditions**:
```python
def set_terminal_condition(self, V0, V1, S_levels):
    principal = self.notional
    
    # V0 at T: just principal (no knockin occurred)
    V0[:, -1] = principal if self.include_principal else 0.0
    
    # V1 at T: principal + downside payoff (knockin occurred)
    ki_downside = self.participation * np.minimum(S_levels - self.K, 0.0)
    V1[:, -1] = (principal + ki_downside) if self.include_principal else ki_downside
```

**Advantages of Two-Surface Method**:
1. **Single solver**: No need to create multiple product/solver instances
2. **State consistency**: State transitions happen at exact observation times
3. **Memory efficiency**: Two surfaces vs multiple decomposed solvers
4. **Natural for path-dependent**: Directly models the state machine

**Disadvantages**:
1. **Product-specific**: Each structured product needs custom logic
2. **More complex code**: Terminal/boundary conditions for multiple surfaces
3. **Harder to validate**: Cannot easily compare against analytical components

### 7.5 Choosing Between Approaches

| Criterion | Decomposition | Two-Surface PDE |
|-----------|---------------|-----------------|
| Code complexity | Lower per solver | Higher, but single solver |
| Reusability | High (same primitives for MC/tree) | Low (PDE-specific) |
| Performance | Multiple solver overhead | Single unified solve |
| Validation | Easy (test each component) | Harder (end-to-end only) |
| New products | Add decomposition logic | Implement new solver |

**Recommendation**: Choose based on your project's priorities:
- **Decomposition** if you value modularity, testability, and multi-engine support
- **Two-Surface PDE** if you prioritize performance and prefer self-contained solvers

---

## 8. Product Interface Requirements

The PDE solver framework requires product objects to provide certain information. **Adapt these requirements to your existing product class design** rather than creating new classes from scratch.

### 8.1 Required Product Attributes

The `BasePDESolver` expects any product object to expose the following attributes (via properties, methods, or direct fields):

**Market Parameters** (required):
- Risk-free rate (e.g., `rrf`, `r`, `rate`)
- Dividend/carry yield (e.g., `div`, `q`, `dividend_yield`)
- Volatility (e.g., `vol`, `sigma`, `volatility`)

**Contract Parameters** (required):
- Spot price of underlying
- Strike price (for options with strikes)
- Time to maturity in years (or days that can be converted)
- Discount factor or method to compute it

**Optional Parameters** (product-specific):
- Barrier levels (single or multiple)
- Observation schedules (for discrete monitoring)
- Rebate amounts
- Participation rates
- Barrier direction indicators

### 8.2 Adapting to Your Product Design

If your project already has product classes, create an **adapter layer** or **protocol/interface** that maps your attributes to what the solver expects:

```python
# Option 1: Protocol/Interface (Python 3.8+)
from typing import Protocol

class PDEPriceable(Protocol):
    """Interface that any product must satisfy for PDE pricing."""
    
    @property
    def spot(self) -> float: ...
    
    @property
    def rrf(self) -> float: ...
    
    @property
    def div(self) -> float: ...
    
    @property
    def vol(self) -> float: ...
    
    @property
    def tau(self) -> float:
        """Time to maturity in years."""
        ...
    
    @property
    def df(self) -> float:
        """Discount factor."""
        ...


# Option 2: Adapter pattern
class ProductAdapter:
    """Wraps your existing product to satisfy PDE solver requirements."""
    
    def __init__(self, your_product):
        self._product = your_product
    
    @property
    def rrf(self) -> float:
        return self._product.risk_free_rate  # Map to your field name
    
    @property
    def tau(self) -> float:
        return self._product.days_to_expiry / 252  # Convert as needed
```

### 8.3 Product Categories

The PDE framework supports these general product categories. Map your existing products accordingly:

| Category | Key Characteristics | Solver Type |
|----------|---------------------|-------------|
| Vanilla European | Strike, call/put flag | EuropeanOptionPDESolver |
| American | Strike, call/put, early exercise | AmericanOptionPDESolver |
| Single Barrier | Barrier level, up/down, in/out | BarrierOptionPDESolver |
| Double Barrier | Upper + lower barriers | DoubleBarrierOptionPDESolver |
| Touch/No-Touch | Barrier, rebate, pay timing | OneTouchOptionPDESolver |
| Structured (Phoenix/Snowball) | Multiple barriers, coupons, schedules | Decomposition or Two-Surface (see Section 7) |

### 8.4 Common Enumerations

You likely already have similar enumerations in your codebase. The solver logic needs to distinguish:

**Barrier Direction**:
- Up vs Down (barrier above or below current spot)
- In vs Out (knock-in or knock-out behavior)

**Observation Frequency**:
- Continuous (barrier monitored at all times)
- Discrete (barrier checked only at specific dates)

**Payment Timing** (for rebates/coupons):
- Immediate (pay when event occurs)
- At expiry (pay at maturity regardless of when event occurred)

Use your existing enums and map them in the solver initialization if names differ.

---

## 9. Testing and Validation

### 9.1 Benchmark Tests

1. **European Options**: Compare to Black-Scholes closed form
   - Expected relative error: < 0.01%

2. **Barrier Options**: Compare to analytical formulas (continuous/periodic discrete with barrier shift)
   - Expected relative error: < 0.1%

3. **American Options**: Compare to analytical formulas (BAW, BS93, BS02)
   - Expected relative error: < 0.1%

### 9.2 Convergence Tests

Verify second-order convergence in space and time:

```python
def test_convergence():
    errors = []
    resolutions = [50, 100, 200, 400]
    
    for N in resolutions:
        config = PDEConfig(x_grid_resolution=N, t_grid_resolution=N)
        solver = EuropeanOptionPDESolver(option, config)
        error = abs(solver.price() - analytical_price)
        errors.append(error)
    
    # Check O(h²) convergence
    for i in range(len(errors) - 1):
        ratio = errors[i] / errors[i + 1]
        assert ratio > 3.5, f"Expected ~4x error reduction, got {ratio}"
```

### 9.3 Numerical Stability Tests

1. High volatility (σ > 1.0)
2. Deep in/out of the money
3. Very short/long maturities
4. Near-barrier spot prices

---

## 10. References

### Books
1. Tavella, D., & Randall, C. (2000). *Pricing Financial Instruments: The Finite Difference Method*. Wiley.
2. Duffy, D. J. (2006). *Finite Difference Methods in Financial Engineering*. Wiley.
3. Wilmott, P. (1998). *Derivatives: The Theory and Practice of Financial Engineering*. Wiley.

### Papers
1. Rannacher, R. (1984). Finite element solution of diffusion problems with irregular data. *Numerische Mathematik*, 43.
2. Pooley, D. M., Vetzal, K. R., & Forsyth, P. A. (2003). Numerical convergence properties of option pricing PDEs. *IMA Journal of Numerical Analysis*, 23.

### Original Source Files
- `asset/equity/engine/bsm/pde/BasePDESolver.py`
- `asset/equity/engine/bsm/pde/PDEConfig.py`
- `asset/equity/engine/bsm/pde/TimeGrid.py`
- `asset/equity/engine/bsm/pde/NonUniformGrid.py`
- `asset/equity/engine/bsm/pde/EuropeanOptionPDESolver.py`
- `asset/equity/engine/bsm/pde/BarrierOptionPDESolver.py`
- `asset/equity/engine/bsm/pde/PhoenixPDESolver.py` (example of decomposition pattern to avoid)
- `asset/equity/engine/bsm/pde/SnowballPDESolver.py` (example of decomposition pattern to avoid)

---

*Document Version: 1.0*
*Created for: PDE Engine Migration Project*

