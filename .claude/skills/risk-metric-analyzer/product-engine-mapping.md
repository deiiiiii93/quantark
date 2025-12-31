# Product-Engine Mapping Reference

This document provides the complete mapping between QuantArk products and their available pricing engines.

## Engine Selection Priority

1. **Analytical** - Preferred when available (fastest, exact for vanilla products)
2. **PDE** - For path-dependent products requiring finite difference methods
3. **Monte Carlo** - For complex products or as validation

## Equity Products

### European Vanilla Option

**Product:** `asset.equity.product.option.EuropeanVanillaOption`

| Engine Type | Engine Class | Location | Notes |
|-------------|--------------|----------|-------|
| Analytical | `BlackScholesEngine` | `asset.equity.engine.analytical.black_scholes_engine` | **Default**, closed-form Black-Scholes |
| PDE | `EuropeanPDESolver` | `asset.equity.engine.pde.european_pde_solver` | Validation only |
| Monte Carlo | `EuropeanMCEngine` | `asset.equity.engine.mc.euro_mc_engine` | Supports Pseudo, Quasi, RQMC |

**Greeks Calculator:** `asset.equity.riskmeasures.GreeksCalculator`
- Analytical Greeks: `calculate_analytical_greeks()` - Full Black-Scholes formulas
- Numerical Greeks: `calculate_numerical_greeks()` - Central difference FDM

---

### American Option

**Product:** `asset.equity.product.option.AmericanOption`

| Engine Type | Engine Class | Location | Notes |
|-------------|--------------|----------|-------|
| Analytical | `AmericanOptionAnalyticalEngine` | `asset.equity.engine.analytical.american_option_engine` | **Default**, 3 methods: BS93, BS02, BAW |
| PDE | `AmericanPDESolver` | `asset.equity.engine.pde.american_pde_solver` | Projected SOR for early exercise |

**Analytical Methods:**
```python
from util.enum.engine_enums import EngineType, AmericanAnalyticalMethod

# Bjerksund & Stensland 1993 (fast, accurate for non-div)
engine = AmericanOptionAnalyticalEngine(method=EngineType.ANALYTICAL(AmericanAnalyticalMethod.BS93))

# Bjerksund & Stensland 2002 (better for dividends)
engine = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS02)

# Barone-Adesi & Whaley (quadratic approximation)
engine = AmericanOptionAnalyticalEngine(method="BAW")
```

**Greeks Calculator:** `asset.equity.riskmeasures.GreeksCalculator`
- Numerical Greeks only (no closed-form available)

---

### Barrier Option (Single Barrier)

**Product:** `asset.equity.product.option.BarrierOption`

| Engine Type | Engine Class | Location | Notes |
|-------------|--------------|----------|-------|
| Analytical | `BarrierAnalyticalEngine` | `asset.equity.engine.analytical` | Limited types only |
| PDE | `BarrierPDESolver` | `asset.equity.engine.pde.barrier_pde_solver` | **Default**, handles all barrier types |

**Barrier Types:**
- `DOWN_AND_OUT`, `DOWN_AND_IN`
- `UP_AND_OUT`, `UP_AND_IN`

**Greeks Calculator:** `asset.equity.riskmeasures.GreeksCalculator`
- Numerical Greeks: `calculate_numerical_greeks()` with PDE engine

---

### Double Barrier Option

**Product:** `asset.equity.product.option.DoubleBarrierOption`

| Engine Type | Engine Class | Location | Notes |
|-------------|--------------|----------|-------|
| PDE | `DoubleBarrierPDESolver` | `asset.equity.engine.pde.double_barrier_pde_solver` | **Default**, only engine available |

**Barrier Types:**
- `KNOCK_IN`, `KNOCK_OUT`

**Greeks Calculator:** `asset.equity.riskmeasures.GreeksCalculator`
- Numerical Greeks only

---

### One Touch Option

**Product:** `asset.equity.product.option.OneTouchOption`

| Engine Type | Engine Class | Location | Notes |
|-------------|--------------|----------|-------|
| Analytical | `OneTouchAnalyticalEngine` | `asset.equity.engine.analytical` | Limited |
| PDE | `OneTouchPDESolver` | `asset.equity.engine.pde.one_touch_pde_solver` | **Default** |

**Greeks Calculator:** `asset.equity.riskmeasures.GreeksCalculator`
- Numerical Greeks only

---

### Double One Touch Option

**Product:** `asset.equity.product.option.DoubleOneTouchOption`

| Engine Type | Engine Class | Location | Notes |
|-------------|--------------|----------|-------|
| PDE | `DoubleOneTouchPDESolver` | `asset.equity.engine.pde.double_one_touch_pde_solver` | **Default** |

**Greeks Calculator:** `asset.equity.riskmeasures.GreeksCalculator`
- Numerical Greeks only

---

### Snowball Option (Autocallable)

**Product:** `asset.equity.product.option.SnowballOption`

| Engine Type | Engine Class | Location | Notes |
|-------------|--------------|----------|-------|
| Monte Carlo | `SnowballMCEngine` | `asset.equity.engine.mc.snowball_mc_engine` | **Default**, Pseudo/Quasi/RQMC |
| PDE | `SnowballPDESolver` | `asset.equity.engine.pde.snowball_pde_solver` | Two-surface method |

**MC Methods:**
```python
from util.enum.engine_enums import MonteCarloMethod

engine = SnowballMCEngine(
    params=MCParams(num_paths=100000),
    method=MonteCarloMethod.QUASI  # or PSEUDO, RANDOMIZED_QUASI
)
```

**Greeks Calculator:** `asset.equity.riskmeasures.GreeksCalculator`
- Numerical Greeks with MC engine (higher variance)

---

### Delta-One Products (Spot, Futures)

**Products:**
- `asset.equity.product.deltaone.SpotInstrument`
- `asset.equity.product.deltaone.Futures`

| Engine Type | Engine Class | Location | Notes |
|-------------|--------------|----------|-------|
| Analytical | `DeltaOneEngine` | `asset.equity.engine.analytical.deltaone_engine` | **Default**, only engine |

**Greeks:**
- Delta = 1.0 (by definition)
- Gamma = 0.0
- Vega = 0.0
- Theta = 0.0 (Spot) / non-zero (Futures)
- Rho = 0.0

---

## Bond Products

### Fixed Bond

**Product:** `asset.bond.product.couponbond.FixedBond`

| Engine Type | Engine Class | Location | Notes |
|-------------|--------------|----------|-------|
| Discount | `BondDiscountEngine` | `asset.bond.engine.discount.bond_discount_engine` | **Default** |

**Risk Metrics:**
- Duration (Macaulay, Modified)
- Convexity
- DV01

---

### Floating Rate Note (FRN)

**Product:** `asset.bond.product.couponbond.FloatingRateNote`

| Engine Type | Engine Class | Location | Notes |
|-------------|--------------|----------|-------|
| Discount | `BondDiscountEngine` | `asset.bond.engine.discount.bond_discount_engine` | **Default** |

**Risk Metrics:**
- Duration (typically close to time to next reset)
- DV01

---

### Bond Option

**Product:** `asset.bond.product.option.EuroShortTermBondOption`

| Engine Type | Engine Class | Location | Notes |
|-------------|--------------|----------|-------|
| Analytical | `BlackBondOptionEngine` | `asset.bond.engine.analytical.black_engine` | **Default**, Black '76 model |

**Greeks Calculator:** `asset.bond.riskmeasures.BondGreeksCalculator`
- `calculate_analytical_greeks()` - Black '76 formulas
- `calculate_numerical_greeks()` - FDM
- `calculate_bond_sensitivities()` - DV01, Duration

**Bond-Specific Metrics:**
- Option DV01
- Option Duration
- Underlying DV01
- Underlying Duration
- Delta-equivalent DV01

---

## Rate Products

### Interest Rate Swap

**Product:** `asset.rate.product.irs.InterestRateSwap`

| Engine Type | Engine Class | Location | Notes |
|-------------|--------------|----------|-------|
| Discount | `IRSDiscountEngine` | `asset.rate.engine.discount` | **Default** |

**Risk Metrics:**
- PV01 (per leg)
- DV01
- Duration

---

## Engine Import Patterns

### Equity Engines

```python
# Analytical
from asset.equity.engine.analytical import (
    BlackScholesEngine,
    AmericanOptionAnalyticalEngine,
    DeltaOneEngine,
)

# Monte Carlo
from asset.equity.engine.mc import (
    EuropeanMCEngine,
    SnowballMCEngine,
)

# PDE (via facade)
from asset.equity.engine import PDEEngine  # Auto-dispatches to correct solver

# PDE (direct)
from asset.equity.engine.pde import (
    EuropeanPDESolver,
    AmericanPDESolver,
    BarrierPDESolver,
    DoubleBarrierPDESolver,
    OneTouchPDESolver,
    DoubleOneTouchPDESolver,
    SnowballPDESolver,
)
```

### Bond Engines

```python
from asset.bond.engine.discount import BondDiscountEngine
from asset.bond.engine.analytical import BlackBondOptionEngine
```

### Rate Engines

```python
from asset.rate.engine.discount import IRSDiscountEngine
```

---

## Risk Measures Import Patterns

### Equity Greeks

```python
from asset.equity.riskmeasures import GreeksCalculator
from asset.equity.param import EngineParams

# Create calculator
params = EngineParams(bump_size=0.01)  # 1% bump
calculator = GreeksCalculator(params)

# Analytical (European vanilla only)
greeks = calculator.calculate_analytical_greeks(product, pricing_env)

# Numerical (any product + engine)
greeks = calculator.calculate_numerical_greeks(product, pricing_env, engine)
```

### Bond Greeks

```python
from asset.bond.riskmeasures import BondGreeksCalculator

calculator = BondGreeksCalculator(bump_size=0.01)

# Analytical (bond options)
greeks = calculator.calculate_analytical_greeks(bond_option, pricing_env)

# Numerical
greeks = calculator.calculate_numerical_greeks(bond_option, pricing_env)

# Bond-specific
sensitivities = calculator.calculate_bond_sensitivities(bond_option, pricing_env)
```

---

## Quick Reference Table

| Product | Default Engine | Greeks Method | Risk Calculator |
|---------|----------------|---------------|-----------------|
| European Vanilla | BlackScholesEngine | Analytical | GreeksCalculator |
| American Option | AmericanOptionAnalyticalEngine | Numerical | GreeksCalculator |
| Barrier Option | BarrierPDESolver | Numerical | GreeksCalculator |
| Double Barrier | DoubleBarrierPDESolver | Numerical | GreeksCalculator |
| One Touch | OneTouchPDESolver | Numerical | GreeksCalculator |
| Double One Touch | DoubleOneTouchPDESolver | Numerical | GreeksCalculator |
| Snowball | SnowballMCEngine | Numerical | GreeksCalculator |
| Spot/Futures | DeltaOneEngine | Trivial | GreeksCalculator |
| Fixed Bond | BondDiscountEngine | Analytical | BondDiscountEngine |
| FRN | BondDiscountEngine | Analytical | BondDiscountEngine |
| Bond Option | BlackBondOptionEngine | Analytical/Numerical | BondGreeksCalculator |
| IRS | IRSDiscountEngine | Analytical | IRSDiscountEngine |
