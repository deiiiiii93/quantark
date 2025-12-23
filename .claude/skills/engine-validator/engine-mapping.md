# Engine Mapping Reference

Quick reference for engine recognition from user prompts.

## Equity Analytical Engines

| User Prompt Keywords | Engine File | Product | Reference Doc |
|---------------------|-------------|---------|---------------|
| "european", "vanilla", "black scholes", "bs" | `black_scholes_engine.py` | `EuropeanVanillaOption` | — |
| "american", "early exercise" | `american_option_engine.py` | `AmericanOption` | `ameopt_analytical_engine.md` |
| "barrier", "knock out", "knock in", "ko", "ki" | `barrier_analytical_engine.py` | `BarrierOption` | `barrier_analytical_engine.md` |
| "asian", "average", "averaging" | `asian_option_analytical_engine.py` | `AsianOption` | `asian_option_analytical_engine.md` |
| "one touch", "no touch", "touch" | `one_touch_analytical_engine.py` | `OneTouchOption` | `onetouch_analytical_engine.md` |
| "digital", "binary", "cash or nothing" | `digital_option_engine.py` | `DigitalOption` | — |
| "delta one", "forward", "futures" | `deltaone_engine.py` | `DeltaOneProduct` | — |

## Equity Monte Carlo Engines

| User Prompt Keywords | Engine File | Product |
|---------------------|-------------|---------|
| "european mc", "vanilla monte carlo" | `euro_mc_engine.py` | `EuropeanVanillaOption` |
| "asian mc" | `asian_option_mc_engine.py` | `AsianOption` |
| "snowball", "autocallable" | `snowball_mc_engine.py` | `SnowballOption` |

## Equity PDE Engines

| User Prompt Keywords | Engine File | Product |
|---------------------|-------------|---------|
| "european pde" | `european_pde_solver.py` | `EuropeanVanillaOption` |
| "american pde" | `american_pde_solver.py` | `AmericanOption` |
| "barrier pde" | `barrier_pde_solver.py` | `BarrierOption` |
| "one touch pde" | `one_touch_pde_solver.py` | `OneTouchOption` |
| "double barrier" | `double_barrier_pde_solver.py` | `DoubleBarrierOption` |
| "double one touch" | `double_one_touch_pde_solver.py` | `DoubleOneTouchOption` |

## Bond Engines

| User Prompt Keywords | Engine File | Product |
|---------------------|-------------|---------|
| "fixed bond", "coupon bond" | `fixed_bond_engine.py` | `FixedBond` |
| "floating rate", "frn" | `frn_engine.py` | `FloatingRateNote` |
| "bond option" | `bond_option_engine.py` | `BondOption` |

## MC Benchmark Availability

| Analytical Engine | Has MC Benchmark? | MC Engine |
|-------------------|-------------------|-----------|
| `black_scholes_engine.py` | **YES** | `euro_mc_engine.py` |
| `asian_option_analytical_engine.py` | **YES** | `asian_option_mc_engine.py` |
| `american_option_engine.py` | NO | *Consider PDE as benchmark* |
| `barrier_analytical_engine.py` | NO | *Need to create* |
| `one_touch_analytical_engine.py` | NO | *Need to create* |
| `digital_option_engine.py` | NO | *Can use euro_mc with custom payoff* |

## MC Engine Validation (No Benchmark)

**IMPORTANT**: When validating MC engines, benchmark check is SKIPPED.

| MC Engine | Validation Focus |
|-----------|------------------|
| `euro_mc_engine.py` | GBM process + vanilla payoff |
| `asian_option_mc_engine.py` | GBM process + averaging logic + asian payoff |
| `snowball_mc_engine.py` | GBM process + barrier monitoring + autocall payoff |

**MC Engine Validation Checklist:**
1. **Process**: Verify SDE discretization (Euler/Milstein), drift/diffusion terms
2. **Payoff**: Verify payoff formula matches product specification
3. **Boundary Checks**: Same as analytical (extreme cases, theoretical relationships)
4. **Convergence**: Price should stabilize as n_paths increases

## File Path Patterns

### Analytical Engines
```
asset/equity/engine/analytical/<engine_name>.py
```

### Monte Carlo Engines
```
asset/equity/engine/mc/<engine_name>.py
```

### PDE Engines
```
asset/equity/engine/pde/<solver_name>.py
```

### Reference Docs
```
asset/equity/engine/docs/<engine_name>.md
```

### Validation Scripts
```
asset/equity/engine/validation/script/boundary_check_<engine_name>.py
asset/equity/engine/validation/script/benchmark_check_<engine_name>.py
```

### Validation Reports
```
asset/equity/engine/validation/report/<engine_name>_validation_report.md
```

## Product Import Patterns

```python
# European Vanilla Option
from asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption

# American Option
from asset.equity.product.option.american_option import AmericanOption

# Barrier Option
from asset.equity.product.option.barrier_option import BarrierOption

# Asian Option
from asset.equity.product.option.asian_option import AsianOption

# One Touch Option
from asset.equity.product.option.one_touch_option import OneTouchOption

# Digital Option
from asset.equity.product.option.digital_option import DigitalOption
```

## Engine Import Patterns

```python
# Analytical
from asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from asset.equity.engine.analytical.american_option_engine import AmericanOptionAnalyticalEngine
from asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine
from asset.equity.engine.analytical.asian_option_analytical_engine import AsianOptionAnalyticalEngine
from asset.equity.engine.analytical.one_touch_analytical_engine import OneTouchAnalyticalEngine
from asset.equity.engine.analytical.digital_option_engine import DigitalOptionEngine

# Monte Carlo
from asset.equity.engine.mc.euro_mc_engine import EuroMCEngine
from asset.equity.engine.mc.asian_option_mc_engine import AsianOptionMCEngine
from asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine

# PDE
from asset.equity.engine.pde.european_pde_solver import EuropeanPDESolver
from asset.equity.engine.pde.american_pde_solver import AmericanPDESolver
from asset.equity.engine.pde.barrier_pde_solver import BarrierPDESolver
```
