# Design: Product-Position Separation

## Context

The current architecture has products and positions both managing quantity/notional:
- `BaseEquityOption` has `notional`, `quantity`, and `NotionalQuantityPolicy`
- `EquityPosition` also has `quantity` attribute
- Confusion about which is the source of truth

## Goals / Non-Goals

**Goals:**
- Products represent exactly 1 unit of an instrument
- Positions manage quantity as single source of truth
- Clear separation: Product = "what", Position = "how much"
- Eliminate `NotionalQuantityPolicy` reconciliation logic
- Preserve instrument-level scaling such as equity option contract multipliers

**Non-Goals:**
- Changing the pricing engine interface (still takes product + pricing_env)
- Changing how portfolios aggregate positions
- Supporting backward compatibility (breaking change)

## Decisions

### Decision 0: Add `contract_multiplier` for equity option products

**Rationale:** Product sizing (notional/quantity) belongs to positions, but an equity option contract may represent more than 1 underlying unit in some markets. The contract multiplier is an instrument specification (part of the "what") rather than a holding size (the "how much").

**Definition:**
- `contract_multiplier`: Underlying units represented by 1 contract (default: 1.0)
- `position.quantity`: Number of contracts held

**Examples:**
- If `contract_multiplier=1.0`: 1 contract represents 1 underlying unit
- If `contract_multiplier=100.0`: 1 contract represents 100 underlying units

### Decision 1: Remove notional/quantity from all equity option products

**Rationale:** Equity options should represent a contract specification, not position size. A call option with strike $100, expiry 1yr is the same regardless of whether you're buying 1 contract or 1000 contracts.

**Alternatives considered:**
- Keep notional, remove quantity only: Still violates single source of truth
- Keep both, add deprecation warnings: Adds technical debt
- **Chosen approach**: Remove both, clean break

### Decision 2: Bonds use "denominator" instead of "notional"

**Rationale:** Bond quotes are typically expressed as a percentage of par (e.g., 100). The "denominator" is the minimum tradable unit (e.g., 1000). Position quantity × denominator = actual notional.

**Naming:**
- `denominator`: Minimum tradable notional (e.g., $1000 for a bond)
- `quantity`: Number of contract units (at position level)
- `actual_notional`: quantity × denominator (derived)

### Decision 3: Engines return per-unit prices

**Rationale:** Engines price the instrument (product). Position layer applies quantity scaling. This maintains engine-agnostic products.

**Formula:**
```python
# Engine returns per-contract price (already includes contract_multiplier where applicable)
per_contract_price = engine.price(product, pricing_env)

# Position applies scaling
total_value = per_contract_price * position.quantity

# For bonds with denominator
total_value = per_contract_price * position.quantity * product.get_denominator()
```

### Decision 4: Exotic payoffs use per-unit calculations

**For Snowball/Phoenix:** Historically, structured equity products are often quoted in notional terms. Under the new convention, the per-contract "principal" is derived from the underlying reference level:

- Per-contract principal/reference notional: `initial_price * contract_multiplier`
- Position scaling: multiply by `position.quantity`

**Before:**
```python
payoff = self.notional * self.ko_rate * accrual_fraction  # Scaled
```

**After:**
```python
payoff = self.initial_price * self.contract_multiplier * self.ko_rate * accrual_fraction  # Per-contract
# Position layer: total = payoff * position.quantity
```

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Snowball/Phoenix payoff errors | Thorough testing of all KO/KI scenarios |
| Bond cashflow errors | Unit tests for cashflow generation |
| Position Greeks scaling | Compare before/after values |
| IRS amortization complexity | Test amortizing schedules separately |

## Migration Plan

**This is a breaking change with no backward compatibility.**

### Steps:
1. Implement all changes in a single PR
2. No gradual migration - all code must update together
3. Update all tests and examples before merging
4. No deprecation period

### Rollback:
- Revert the PR
- All existing code continues to work

## Open Questions

None - unit conventions and contract_multiplier are defined in this change.
