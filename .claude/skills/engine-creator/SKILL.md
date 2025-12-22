---
name: engine-creator
description: |
  Create new pricing engine scripts in the asset/ directory following QuantArk patterns.
  Use when the user asks to:
  - Create a new pricing engine (analytical, MC, PDE, quadrature, tree)
  - Add a new pricing method for a product
  - Implement a new solver (PDE, Monte Carlo)
  - Create a facade engine for multiple methods
  Triggers: "create engine", "new engine", "add pricing", "implement solver", "pricing method"
---

# Engine Creator Skill

Automatically scaffold new pricing engine scripts in `asset/` following established codebase patterns.

## When This Skill Activates

Claude should use this skill when:
- User asks to create/add/implement a new pricing engine
- User wants to add a new pricing method (analytical, MC, PDE, tree)
- User describes a pricing algorithm that needs implementation
- User mentions creating an engine class

## Workflow

### Step 1: Script Naming

**Mandatory Format**: `{product_name}_{engine_type}_engine.py`

| Product | Engine Type | Script Name |
|---------|-------------|-------------|
| European Vanilla Option | Analytical | `european_vanilla_option_analytical_engine.py` |
| American Option | PDE | `american_option_pde_engine.py` |
| Barrier Option | MC | `barrier_option_mc_engine.py` |
| Snowball Option | MC | `snowball_mc_engine.py` |
| One Touch Option | Analytical | `one_touch_analytical_engine.py` |

**Engine Type Suffixes**:
- `_analytical_engine.py` - Closed-form solutions
- `_mc_engine.py` - Monte Carlo simulation
- `_pde_engine.py` or `_pde_solver.py` - PDE/finite difference
- `_tree_engine.py` - Binomial/trinomial trees
- `_quad_engine.py` - Quadrature methods

### Step 2: Find Base Class

**Decision Matrix:**

| Engine Type | Base Class | Location |
|-------------|------------|----------|
| Analytical (equity) | `BaseEngine` | `asset/equity/engine/base_engine.py` |
| Monte Carlo (equity) | `BaseEngine` | Same |
| PDE Solver (equity) | `BasePDESolver` | `asset/equity/engine/pde/base_pde_solver.py` |
| Bond discount | `BaseEngine` | `asset/bond/engine/base_engine.py` |
| Tree (bond) | Custom | `asset/bond/engine/tree/` |

**Explore before deciding:**
```bash
rg "class Base.*Engine" asset/ --type py
rg "class Base.*Solver" asset/ --type py
```

**When to create NEW base class:**
- Engine has fundamentally different interface
- Multiple related engines will share the pattern
- Existing bases don't provide needed methods

### Step 3: Find Folder Location

**Folder Structure Pattern:**
```
asset/<asset_type>/engine/<engine_type>/
```

| Asset | Engine Type | Folder |
|-------|-------------|--------|
| Equity | Analytical | `asset/equity/engine/analytical/` |
| Equity | Monte Carlo | `asset/equity/engine/mc/` |
| Equity | PDE | `asset/equity/engine/pde/` |
| Bond | Discount | `asset/bond/engine/discount/` |
| Bond | Tree | `asset/bond/engine/tree/` |
| Bond | Convertible | `asset/bond/engine/convertible/` |
| Rate | Discount | `asset/rate/engine/discount/` |

### Step 4: Find Reference Documentation

**Reference Location Pattern:**
```
asset/<asset_type>/engine/docs/<script_name>.md
```

Example: For `one_touch_analytical_engine.py`:
- Check: `asset/equity/engine/docs/onetouch_analytical_engine.md`
- Also check: `docs/` in project root

**If no reference exists:**
1. Use web search: `"[product name] [method] pricing formula"`
2. Search for academic papers: `"[product] closed-form solution"`
3. Check standard references (Hull, Wilmott, Glasserman)
4. Document findings in a new `.md` file in `docs/`

**Create reference doc with:**
- Mathematical formulas
- Numerical considerations
- Edge cases
- References/citations

### Step 5: Assess Helper Requirements

**Complexity Indicators:**

| Indicator | Points |
|-----------|--------|
| Requires custom grid (PDE) | +2 |
| Multiple path generation methods (MC) | +2 |
| Time-varying parameters | +1 |
| Multiple solver variants | +1 |
| Complex boundary conditions | +1 |
| Matrix caching needed | +1 |

**Decision:**
- Score 0-2: Simple engine, no helpers
- Score 3-4: Add utility functions within engine
- Score 5+: Create separate helper modules

**PDE Helper Pattern:**
```
asset/equity/engine/pde/
├── base_pde_solver.py      # Abstract base
├── time_grid.py            # Time discretization helper
├── spatial_grid.py         # Spatial discretization helper
├── european_pde_solver.py  # Concrete solver
└── ...
```

### Step 6: Determine Facade Engine Need

**When to create a Facade:**
- Product has 2+ pricing methods of same type
- Want unified interface for method selection
- Client code shouldn't know about internal solvers

**Facade Pattern:**
```python
class PDEEngine(BaseEngine):
    """Unified PDE engine - dispatches to correct solver"""

    PRODUCT_SOLVER_MAP = {
        EuropeanVanillaOption: EuropeanPDESolver,
        AmericanOption: AmericanPDESolver,
        BarrierOption: BarrierPDESolver,
    }

    def price(self, product, pricing_env):
        solver_class = self.PRODUCT_SOLVER_MAP.get(type(product))
        if solver_class is None:
            raise PricingError(f"No solver for {type(product)}")
        solver = solver_class(params=self.params)
        return solver.price(product, pricing_env)
```

### Step 7: Register Engine Type

**Location:** `util/enum/engine_enums.py`

**Two-Level Enum Pattern:**
```python
class EngineType(Enum):
    ANALYTICAL = auto()
    MONTE_CARLO = auto()
    PDE = auto()
    QUADRATURE = auto()
    TREE = auto()

    def __call__(self, method=None):
        if method is not None:
            return (self, method)
        return self

# Method-specific enums
class AmericanAnalyticalMethod(Enum):
    BS93 = "BS93"
    BS02 = "BS02"
    BAW = "BAW"

class MonteCarloMethod(Enum):
    PSEUDO = "pseudo"
    QUASI = "quasi"
    RANDOMIZED_QUASI = "randomized_quasi"
```

**Usage in Engine:**
```python
class MyEngine(BaseEngine):
    def __init__(self, method=None, params=None):
        super().__init__(params)
        # Handle all three initialization patterns
        if isinstance(method, tuple):
            # EngineType.ANALYTICAL(Method.XYZ)
            _, self.method = method
        elif isinstance(method, MyMethodEnum):
            self.method = method
        elif isinstance(method, str):
            self.method = MyMethodEnum(method)
        else:
            self.method = MyMethodEnum.DEFAULT
```

### Step 8: Apply Codebase Patterns

See [patterns.md](patterns.md) for detailed patterns.

**Quick Checklist:**
- [ ] Inherit from appropriate base class
- [ ] Implement `price(product, pricing_env) -> float` (engine's ONLY responsibility)
- [ ] Validate product type at start of `price()`
- [ ] Handle edge cases (near expiry, deep ITM/OTM)
- [ ] **DO NOT** override `calculate_greeks()` unless analytical formulas exist
- [ ] For Greeks, use `GreeksCalculator` from `riskmeasures/` folder
- [ ] Use `util.numerical` for safe math operations
- [ ] Add to `util/enum/engine_enums.py` if new method type
- [ ] Update `__init__.py` exports
- [ ] Create reference doc in `docs/`

**Greeks Pattern:**
- Engines focus on `price()` only
- Numerical Greeks: Use `asset/<type>/riskmeasures/greeks_calculator.py`
- Analytical Greeks override: Only for closed-form formulas (e.g., Black-Scholes)

## Output Structure

**Simple Engine:**
```
asset/<type>/engine/<method>/
├── __init__.py (updated)
└── my_product_analytical_engine.py
```

**Complex Engine with Helpers:**
```
asset/<type>/engine/<method>/
├── __init__.py (updated)
├── my_product_pde_solver.py
├── my_grid_helper.py (if needed)
└── docs/
    └── my_product_pde_solver.md
```

**Facade Engine:**
```
asset/<type>/engine/
├── __init__.py (updated)
├── my_facade_engine.py          # Unified interface
└── <method>/
    ├── concrete_solver_1.py
    └── concrete_solver_2.py
```

## Reference Files

Study these for patterns:
- Base engine: `asset/equity/engine/base_engine.py`
- Analytical: `asset/equity/engine/analytical/black_scholes_engine.py`
- Multi-method: `asset/equity/engine/analytical/american_option_engine.py`
- Monte Carlo: `asset/equity/engine/mc/euro_mc_engine.py`
- PDE base: `asset/equity/engine/pde/base_pde_solver.py`
- PDE helpers: `asset/equity/engine/pde/time_grid.py`, `spatial_grid.py`
- Facade: `asset/equity/engine/pde_engine.py`
- Enum pattern: `util/enum/engine_enums.py`
