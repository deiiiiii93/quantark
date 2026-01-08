---
name: product-creator
description: |
  Create new financial product classes in the asset/ directory following QuantArk patterns.
  Use when the user asks to:
  - Create a new product (option, bond, swap, futures, forward)
  - Add a new derivative instrument
  - Scaffold a new asset class product
  - Implement a new payoff structure
  Triggers: "create product", "new product", "add option", "implement derivative", "scaffold product"
---

# Product Creator Skill

Automatically scaffold new financial product scripts in `asset/` following established codebase patterns.

## When This Skill Activates

Claude should use this skill when:
- User asks to create/add/implement a new financial product
- User wants to scaffold an option, bond, swap, futures, or other derivative
- User describes a payoff structure that needs implementation
- User mentions creating a product class

## Workflow

### Step 1: Script Naming
Convert product name to `snake_case.py`:
- "European Vanilla Option" → `european_vanilla_option.py`
- "Floating Rate Note" → `frn.py` (abbreviations OK if standard)
- "Asian Option" → `asian_option.py`

### Step 2: Find Base Class

**Decision Matrix:**

| Product Type | Base Class | Location |
|--------------|------------|----------|
| Simple option (call/put) | `BaseEquityOption` | `asset/equity/product/option/base_equity_option.py` |
| Barrier option | `BaseEquityOption` | Same |
| Autocallable/structured | `BaseStructuredProduct` | `asset/equity/product/option/base_structured_product.py` |
| Delta-one (stock, index) | `BaseDeltaOneProduct` | `asset/equity/product/deltaone/base_deltaone_product.py` |
| Fixed/floating bond | `BaseBondProduct` | `asset/bond/product/base_bond_product.py` |
| Bond forward/futures | `BaseBondForward` | `asset/bond/product/forward/base_bond_forward.py` |
| Interest rate swap | Custom or standalone | `asset/rate/product/` |

**Explore before deciding:**
```bash
rg "class Base" asset/ --type py
```

**When to create NEW base class:**
- Product has fundamentally different structure
- Multiple related products will share the pattern
- Existing bases don't provide needed interfaces

### Step 3: Payoff Analysis

Create comprehensive payoff specification:

```
## Payoff Analysis for [Product Name]

### States
1. Normal termination: [formula]
2. Early termination: [formula if applicable]
3. Contingent states: [barrier events, etc.]

### Path Dependence
- Discrete observations?
- Continuous monitoring?
- Averaging?

### Variables
- Spot, strike, barriers
- Participation rates
- Accrual factors
```

**If user's description is incomplete:**
- Use web search: `"[product name] payoff formula"`
- Ask clarifying questions before proceeding

### Step 4: Assess Helper Complexity

**Scoring System:**

| Factor | Points |
|--------|--------|
| >5 constructor parameters | +1 |
| Barrier configurations | +1 |
| Multiple payoff states | +1 |
| Time-varying parameters (arrays) | +1 |
| Observation schedules | +1 |
| Accrual/coupon conventions | +1 |
| Multiple product variants | +2 |

**Decision:**
- Score 0-2: Simple `@dataclass` product, no helpers
- Score 3-4: Add `@dataclass(frozen=True)` config classes
- Score 5+: Add helper module with factory functions

### Step 4.5: Define Pricing Scale Attributes

**IMPORTANT: Products must define appropriate scaling attributes for their asset class.**

QuantArk uses different scaling conventions by asset class:

**For Equity Derivatives:**

All equity products must include `contract_multiplier` attribute:

```python
@dataclass
class MyEquityProduct(BaseEquityOption):
    """
    My equity derivative product.

    Attributes:
        strike: Strike price
        option_type: CALL or PUT
        maturity: Time to maturity (years)
        contract_multiplier: Underlying units per contract (default: 1.0)
    """
    strike: float
    option_type: OptionType
    maturity: float
    contract_multiplier: float = 1.0  # REQUIRED for equity derivatives

    def __post_init__(self):
        # Validate contract_multiplier
        if self.contract_multiplier <= 0:
            raise ValidationError(
                f"Contract multiplier must be positive, got {self.contract_multiplier}"
            )
```

**Common contract_multipliers:**
- `1.0` - Single share (default)
- `100.0` - Standard option contract (100 shares)
- `10,000.0` - Large notional structured products

**For Fixed Income Products:**

All bond products must include `denominator` attribute:

```python
@dataclass
class MyBondProduct(BaseBondProduct):
    """
    My bond product.

    Attributes:
        coupon_rate: Annual coupon rate
        maturity: Maturity date
        denominator: Minimum tradable notional (default: 100.0)
    """
    coupon_rate: float
    maturity: float
    denominator: float = 100.0  # REQUIRED for bonds

    def get_denominator(self) -> float:
        """Get the minimum tradable notional (denominator) of the bond."""
        return self.denominator
```

**Common denominators:**
- `100.0` - Standard corporate/US Treasury bonds
- `1,000.0` - Municipal bonds, some institutional products
- `100,000.0` - Large notional bonds

**Two-Stage Scaling Model:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUANTARK SCALING MODEL                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: Product Definition                                     │
│  ┌─────────────────┬─────────────────────────────────────────┐  │
│  │ Equity Products │ contract_multiplier = shares per contract│  │
│  │ Bond Products    │ denominator = minimum tradable notional  │  │
│  └─────────────────┴─────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  Stage 2: Engine Output (per unit)                               │
│  ┌─────────────────┬─────────────────────────────────────────┐  │
│  │ Equity Engines  │ price = theoretical_value × multiplier   │  │
│  │ Bond Engines    │ price = PV including denominator         │  │
│  └─────────────────┴─────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  Stage 3: Position Scaling                                      │
│  ┌─────────────────┬─────────────────────────────────────────┐  │
│  │ All Positions   │ market_value = engine_price × quantity    │  │
│  └─────────────────┴─────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Product Payoff Methods Must Also Scale:**

```python
def get_payoff(self, spot: float) -> float:
    """
    Calculate payoff at maturity.

    IMPORTANT: Return payoff scaled by contract_multiplier or denominator.
    """
    intrinsic = max(spot - self.strike, 0)  # For calls
    return intrinsic * self.contract_multiplier  # Don't forget scaling!
```

**Reference Examples:**
- `asset/equity/product/option/base_equity_option.py:74` - contract_multiplier definition
- `asset/equity/product/option/european_vanilla_option.py:85` - get_payoff() with scaling
- `asset/bond/product/base_bond_product.py:53` - get_denominator() method
- `asset/bond/product/couponbond/fixed_bond.py:177` - Bond get_denominator()

### Step 5: Apply Codebase Patterns

See [patterns.md](patterns.md) for detailed patterns to apply.

**Quick Checklist:**
- [ ] Use `@dataclass` decorator
- [ ] Inherit from appropriate base class
- [ ] Implement: `get_payoff()`, `get_maturity()`, `validate()`
- [ ] Support dual maturity format (time-based and date-based)
- [ ] Use type-safe enums from `util/enum/`
- [ ] Use numerical utilities from `util/numerical/`
- [ ] **Equity: Add `contract_multiplier` attribute (default: 1.0)**
- [ ] **Bonds: Add `denominator` attribute (default: 100.0)**
- [ ] **Scale `get_payoff()` return value by contract_multiplier/denominator**
- [ ] Update `__init__.py` exports

## Output Structure

**Simple product (score 0-2):**
```
asset/<type>/product/<category>/
├── __init__.py (updated)
└── my_product.py
```

**Complex product (score 3+):**
```
asset/<type>/product/<category>/
├── __init__.py (updated)
├── my_product.py
├── my_product_config.py (if score >= 3)
└── my_product_helpers.py (if score >= 5)
```

## Reference Files

Study these for patterns:
- Simple option: `asset/equity/product/option/european_vanilla_option.py`
- Barrier option: `asset/equity/product/option/barrier_option.py`
- Complex structured: `asset/equity/product/option/snowball_option.py`
- Config objects: `asset/equity/product/option/snowball_config.py`
- Factory functions: `asset/equity/product/option/snowball_helpers.py`
- Bond: `asset/bond/product/couponbond/fixed_bond.py`
