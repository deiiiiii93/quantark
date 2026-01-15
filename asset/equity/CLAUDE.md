# Equity Derivatives Module - Developer Guide

## Overview

The Equity Derivatives module (`asset/equity/`) is a comprehensive framework for pricing and risk analysis of equity derivatives. It provides a **modular, engine-agnostic architecture** supporting multiple pricing methodologies (analytical, Monte Carlo, PDE) across various equity products.

## Architecture

### Core Design Pattern: Product-Engine Separation

```
┌────────────────────────────────────────────────────────────────┐
│                       Products Layer                           │
│  product/option/           product/deltaone/                   │
│  ├── EuropeanVanillaOption ├── SpotInstrument                  │
│  ├── AmericanOption        └── Futures                         │
│  ├── AsianOption                                               │
│  ├── BarrierOption, DoubleBarrierOption                        │
│  ├── OneTouchOption, DoubleOneTouchOption                      │
│  ├── CashOrNothingDigitalOption                                │
│  ├── SnowballOption                                            │
│  └── PhoenixOption                                             │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                        Engine Layer                            │
│  engine/analytical/    engine/mc/          engine/pde/         │
│  ├── BlackScholes      ├── EuropeanMC      ├── EuropeanPDE     │
│  ├── American (BS93,   ├── AmericanMC      ├── AmericanPDE     │
│  │   BS02, BAW)        ├── AsianMC         ├── BarrierPDE      │
│  ├── Asian             ├── DigitalMC       ├── DoubleBarrierPDE│
│  ├── Digital           ├── BarrierMC       ├── OneTouchPDE     │
│  ├── Barrier           └── SnowballMC      ├── DoubleOneTouchPDE
│  ├── OneTouch                              └── SnowballPDE     │
│  └── DeltaOne                                                  │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                       Process Layer                            │
│  process/bsm/                                                  │
│  ├── BSMProcess          (Geometric Brownian Motion)           │
│  ├── GBMPathGenerator    (QMC path generation)                 │
│  └── qmc_* utilities     (Sobol, Brownian bridge, variance red)│
└────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
asset/equity/
├── product/                       # Instrument definitions
│   ├── base_equity_product.py
│   ├── option/
│   │   ├── european_vanilla_option.py
│   │   ├── american_option.py
│   │   ├── asian_option.py
│   │   ├── digital_option.py
│   │   ├── barrier_option.py
│   │   ├── double_barrier_option.py
│   │   ├── one_touch_option.py
│   │   ├── double_one_touch_option.py
│   │   ├── snowball_option.py
│   │   ├── snowball_config.py
│   │   ├── snowball_helpers.py
│   │   ├── phoenix_option.py
│   │   ├── phoenix_config.py
│   │   ├── phoenix_helpers.py
│   │   └── observation_schedule.py
│   └── deltaone/
│       ├── spot_instrument.py
│       └── futures.py
├── engine/
│   ├── base_engine.py
│   ├── analytical/
│   │   ├── black_scholes_engine.py
│   │   ├── american_option_engine.py
│   │   ├── asian_option_analytical_engine.py
│   │   ├── digital_option_engine.py
│   │   ├── barrier_analytical_engine.py
│   │   ├── one_touch_analytical_engine.py
│   │   └── deltaone_engine.py
│   ├── mc/
│   │   ├── euro_mc_engine.py
│   │   ├── american_option_mc_engine.py
│   │   ├── asian_option_mc_engine.py
│   │   ├── digital_option_mc_engine.py
│   │   ├── barrier_option_mc_engine.py
│   │   └── snowball_mc_engine.py
│   └── pde/
│       ├── base_pde_solver.py
│       ├── european_pde_solver.py
│       ├── american_pde_solver.py
│       ├── barrier_pde_solver.py
│       ├── double_barrier_pde_solver.py
│       ├── one_touch_pde_solver.py
│       ├── double_one_touch_pde_solver.py
│       ├── snowball_pde_solver.py
│       ├── time_grid.py
│       └── spatial_grid.py
├── process/bsm/
│   ├── bsm_process.py
│   ├── qmc_path_generator.py
│   ├── qmc_sobol.py
│   ├── qmc_brownian_bridge.py
│   ├── qmc_rqmc_driver.py
│   └── qmc_variance_reduction.py
├── riskmeasures/
│   └── greeks_calculator.py
└── param/
    └── engine_params.py
```

## Products

### Options (`product/option/`)

| Product | Description | Supported Engines |
|---------|-------------|-------------------|
| `EuropeanVanillaOption` | Standard European call/put | Analytical, MC, PDE |
| `AmericanOption` | Early exercise options | Analytical (BS93/BS02/BAW), MC (LSM), PDE |
| `AsianOption` | Arithmetic/geometric averaging | Analytical, MC |
| `CashOrNothingDigitalOption` | Binary cash-or-nothing | Analytical, MC |
| `BarrierOption` | Single barrier (up/down, in/out) | Analytical, MC, PDE |
| `DoubleBarrierOption` | Two barriers | PDE |
| `OneTouchOption` | Pays if barrier touched | Analytical, PDE |
| `DoubleOneTouchOption` | Pays if either barrier touched | PDE |
| `SnowballOption` | Autocallable with KO/KI | MC, PDE |
| `PhoenixOption` | Autocallable with periodic coupons | MC, PDE |

### Delta-One Products (`product/deltaone/`)

| Product | Description |
|---------|-------------|
| `SpotInstrument` | Stocks, indices, ETFs (perpetual) |
| `Futures` | Futures contracts with multiplier |

## Usage Examples

### European Option Pricing

```python
from asset.equity.product.option import EuropeanVanillaOption
from asset.equity.engine.analytical import BlackScholesEngine
from priceenv import PricingEnvironment
from util.enum import OptionType

option = EuropeanVanillaOption(
    strike=100.0,
    option_type=OptionType.CALL,
    maturity=1.0
)

engine = BlackScholesEngine()
price = engine.price(option, pricing_env)
greeks = engine.calculate_greeks(option, pricing_env)
```

### American Option with Multiple Methods

```python
from asset.equity.product.option import AmericanOption
from asset.equity.engine.analytical import AmericanOptionAnalyticalEngine
from util.enum.engine_enums import EngineType, AmericanAnalyticalMethod

option = AmericanOption(
    strike=100.0,
    option_type=OptionType.PUT,
    maturity=0.5
)

# Preferred: Two-level enum pattern
engine = AmericanOptionAnalyticalEngine(
    method=EngineType.ANALYTICAL(AmericanAnalyticalMethod.BS93)
)
price = engine.price(option, pricing_env)
```

### Asian Option

```python
from asset.equity.product.option import AsianOption, AsianObservationRecord
from asset.equity.engine.analytical import AsianOptionAnalyticalEngine
from util.enum import AsianStrikeType, AveragingType

option = AsianOption(
    strike=100.0,
    option_type=OptionType.CALL,
    asian_strike_type=AsianStrikeType.FIXED,
    averaging_type=AveragingType.ARITHMETIC,
    num_observations=12,
    maturity=1.0
)
```

### Monte Carlo Pricing

```python
from asset.equity.engine.mc import EuropeanMCEngine
from asset.equity.param import MCParams
from util.enum.engine_enums import MonteCarloMethod

engine = EuropeanMCEngine(
    params=MCParams(num_paths=100000, time_steps=252, use_qmc=True),
    method=MonteCarloMethod.QUASI
)
price = engine.price(option, pricing_env)
```

### Snowball (Autocallable) Option

```python
from asset.equity.product.option import SnowballOption
from asset.equity.product.option.snowball_config import BarrierConfig
from asset.equity.engine.mc import SnowballMCEngine

barrier_config = BarrierConfig(
    ko_barrier=103.0,
    ko_rate=0.15,
    ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
    ki_barrier=75.0,
    ki_continuous=True,
)
snowball = SnowballOption(
    initial_price=100.0,
    strike=100.0,
    barrier_config=barrier_config,
    contract_multiplier=10_000.0,
    maturity=1.0,
)

engine = SnowballMCEngine(params=MCParams(num_paths=100000))
price = engine.price(snowball, pricing_env)
```

### Phoenix (Autocallable with Coupons) Option

```python
from asset.equity.product.option import (
    PhoenixOption,
    create_standard_phoenix,
)
from asset.equity.product.option.phoenix_config import CouponBarrierConfig
from asset.equity.engine.mc import SnowballMCEngine
from util.calendar.day_counter import DayCountConvention
from util.enum import CouponPayType

# Method 1: Use factory helper (recommended)
phoenix = create_standard_phoenix(
    initial_price=100.0,
    strike=100.0,
    maturity=1.0,
    ko_barrier=103.0,      # KO at 103%
    ki_barrier=75.0,       # KI at 75%
    coupon_barrier=85.0,   # Coupon paid when spot >= 85%
    coupon_rate=0.01,      # 1% per period
    num_observations=12,   # Monthly observations
    memory_coupon=True,    # Accumulate missed coupons
    day_count_convention=DayCountConvention.ACT_365,
    coupon_pay_type=CouponPayType.INSTANT,
)

# Method 2: Direct construction with custom configs
from asset.equity.product.option.snowball_config import BarrierConfig

barrier_config = BarrierConfig(
    ko_barrier=103.0,
    ko_rate=0.15,
    ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
    ki_barrier=75.0,
    ki_continuous=True,
)

coupon_config = CouponBarrierConfig(
    coupon_barrier=[85.0, 84.0, 83.0, 82.0],  # Step-down coupon barriers
    coupon_rate=0.01,
    memory_coupon=True,
    day_count_convention=DayCountConvention.ACT_365,
)

phoenix = PhoenixOption(
    initial_price=100.0,
    strike=100.0,
    maturity=1.0,
    barrier_config=barrier_config,
    coupon_config=coupon_config,
    contract_multiplier=10_000.0,
)

# Pricing (reuses SnowballMC engine)
engine = SnowballMCEngine(params=MCParams(num_paths=100000))
price = engine.price(phoenix, pricing_env)
```

**Phoenix vs Snowball:**
- **Snowball**: Coupon paid only on knock-out event
- **Phoenix**: Coupons paid at each observation where coupon barrier is hit
- **Memory Coupon**: Both support accumulating missed coupons when barrier is eventually hit

## Parameters

### EngineParams
```python
from asset.equity.param import EngineParams, MCParams, PDEParams, BumpConfig

# Base engine params
params = EngineParams(
    bump_size=1e-4,      # For finite difference Greeks
    bus_days_in_year=252
)

# Monte Carlo params
mc_params = MCParams(
    seed=42,
    num_paths=100000,
    time_steps=252,
    use_qmc=True,
    use_antithetic=True
)

# PDE params
pde_params = PDEParams(
    grid_size=400,
    time_steps=200,
    adaptive_grid=False,
    auto_grid=True,
    cache_enabled=True,
    grid_cache_max_entries=128,
    use_banded_solver=True,
    banded_cache_max_entries=512
)
```

## Risk Measures

### Greeks Calculator

```python
from asset.equity.riskmeasures import GreeksCalculator
from util.enum import GreeksCalculationMode

calculator = GreeksCalculator(
    params=EngineParams(),
    greeks_mode=GreeksCalculationMode.BUMP  # or ENGINE, AUTO
)

greeks = calculator.calculate_greeks(
    product=option,
    engine=engine,
    pricing_env=pricing_env
)
# Returns: delta, gamma, vega, theta, rho
```

**Greeks Calculation Modes:**
- `BUMP`: Always use finite difference bump-and-reprice
- `ENGINE`: Use engine's `calculate_greeks()` if implemented
- `AUTO`: Use engine method for PDE engines, bump otherwise

## Testing

```bash
# Run equity tests
python -m pytest test/test_european_option.py -v
python -m pytest test/test_american_option_analytical.py -v

# Run with coverage
python -m pytest test/test_european_option.py --cov=asset.equity
```

## Key Conventions

### Engine Method Selection (Two-Level Enum Pattern)

```python
from util.enum.engine_enums import EngineType, AmericanAnalyticalMethod, MonteCarloMethod

# Preferred: Two-level enum
engine = AmericanOptionAnalyticalEngine(
    method=EngineType.ANALYTICAL(AmericanAnalyticalMethod.BS93)
)

# Alternative: Direct method enum
engine = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93)

# Backward compatible: String
engine = AmericanOptionAnalyticalEngine(method="BS93")
```

### Product Validation
Products validate themselves on construction. Always call `validate()` explicitly if parameters are modified after construction.

### Numerical Stability
Use `util/numerical/` utilities for all numerical operations:
```python
from util.numerical import is_zero, safe_log, safe_exp

if is_zero(time_to_expiry):
    return intrinsic_value
```

## Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Engine only supports X" | Wrong product type | Use correct engine for product |
| "Option has expired" | Maturity ≤ 0 | Return payoff instead |
| "Barrier breached" | PDE grid doesn't cover barrier | Extend spatial grid |
| Slow MC convergence | Using pseudorandom | Enable `use_qmc=True` |

## Summary

- **Products**: 11 option types + 2 delta-one types
- **Engines**: 7 analytical, 6 Monte Carlo, 8 PDE solvers
- **Features**: 3 pricing methods, QMC variance reduction, American exercise (analytical + LSM), full Greeks suite
- **Autocallables**: Snowball (KO coupon only) and Phoenix (periodic coupons) with memory coupon support
