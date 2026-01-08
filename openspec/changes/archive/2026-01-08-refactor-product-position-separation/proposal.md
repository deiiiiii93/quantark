# Change: Refactor Product-Position Separation

## Why

Products currently have both `notional` and `quantity` attributes with reconciliation logic (`NotionalQuantityPolicy`), creating confusion about the source of truth for position size. Positions also have their own `quantity` attribute. This violates the principle that Products should define instrument specifications ("what") while Positions should define holdings ("how much"). The `NotionalQuantityPolicy` with PREFER_NOTIONAL/PREFER_QUANTITY/STRICT is a code smell indicating two competing sources of truth.

## What Changes

**BREAKING CHANGE** - This refactoring fundamentally changes the product API:

- **REMOVE** `notional` and `quantity` attributes from all product classes
- **REMOVE** `NotionalQuantityPolicy` enum and reconciliation logic
- **REMOVE** `get_notional()`, `set_notional()`, `get_quantity()`, `set_quantity()` methods from `BaseEquityOption`
- **ADD** `contract_multiplier` to equity option products (instrument-level, defaults to 1.0)
- **ADD** `get_denominator()` abstract method to `BaseBondProduct` (minimum tradable notional)
- **MODIFY** all product classes to represent exactly 1 unit
- **MODIFY** all payoff calculations to return per-unit values
- **MODIFY** Position classes to be the single source of truth for quantity

### By Asset Class

| Asset Class | Product Change | Position Change |
|-------------|----------------|-----------------|
| Equity Options | Remove notional/quantity | Use position.quantity |
| Bonds | Replace notional → denominator | quantity × denominator = actual |
| IRS | NotionalSchedule → denominator concept | quantity × denominator |

### Unit Conventions (New)

- **Equity options**: A product represents **one contract**. `contract_multiplier` defines how many underlying units are delivered/represented by one contract (default: 1.0).
- **Snowball/Phoenix**: Per-contract principal/reference notional is `initial_price * contract_multiplier` (so payoffs previously written as `notional * ...` become `initial_price * contract_multiplier * ...`).

## Impact

- **Affected specs:** product-position-separation (new), equity-phoenix-option, snowball-option-helpers, equity-barrier-products, base-engine, greeks-calculator
- **Affected code:**
  - `asset/equity/product/option/base_equity_option.py` (~200 lines removed)
  - `asset/equity/product/option/snowball_option.py`
  - `asset/equity/product/option/phoenix_option.py`
  - `asset/bond/product/base_bond_product.py`
  - `asset/bond/product/couponbond/fixed_bond.py`
  - `asset/rate/product/irs.py`
  - `portfolio/equity/position.py`
  - `portfolio/fi/position.py`
  - `util/enum/option_enums.py` (remove NotionalQuantityPolicy)
  - 10+ test files
  - 8+ example files
- **Estimated changes:** ~900 lines across ~30 files
