# Add Airbag Snowball Payoff Support

## Why

The `AirbagConfig` dataclass and `create_airbag_snowball()` helper function exist, but the actual airbag payoff logic is **not implemented** in `SnowballOption.get_maturity_payoff_v1()`. This means:

1. Users can create airbag snowball products with `create_airbag_snowball()`
2. The `airbag_config` is stored on the product
3. But when the MC engine prices the product, **the airbag logic is ignored**
4. The V1 payoff is calculated using the standard participation rate, not the reduced airbag participation rate

This is a critical bug/missing feature that renders airbag snowballs non-functional.

## What Changes

### SnowballOption Product (Primary)

Modify `get_maturity_payoff_v1()` to implement airbag payoff logic:

**Airbag Payoff Logic:**
- If `airbag_config.airbag_barrier` is None: use standard V1 payoff (current behavior)
- Standard snowball (`is_reverse=False`):
  - If spot >= airbag_barrier: use standard participation rate
  - If spot < airbag_barrier: use `airbag_participation_rate` for downside calculation
- Reverse snowball (`is_reverse=True`):
  - If spot <= airbag_barrier: use standard participation rate
  - If spot > airbag_barrier: use `airbag_participation_rate` for downside calculation
- Use `airbag_strike` if specified, otherwise use product strike

**Formula (standard snowball, V1 state, spot < airbag_barrier):**
```
downside = airbag_participation_rate × min(spot - airbag_strike, 0) × N / S0
```

**Formula (reverse snowball, V1 state, spot > airbag_barrier):**
```
downside = airbag_participation_rate × min(airbag_strike - spot, 0) × N / S0
```

### Monte Carlo Engine (No Changes Required)

The `SnowballMCEngine` calls `product.get_maturity_payoff_v1(spot, pricing_env)` for V1 paths. Once the product correctly implements airbag logic, the MC engine will automatically price airbag snowballs correctly.

## Affected Components

- `asset/equity/product/option/snowball_option.py` - Implement airbag logic in `get_maturity_payoff_v1()`
- `asset/equity/product/option/snowball_helpers.py` - Update reverse airbag validation (`airbag_barrier` relative to `ki_barrier`)
- `test/test_snowball_helpers.py` - Add tests for airbag payoff calculation
- `test/test_snowball_option.py` - Add tests for reverse airbag payoff calculation

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing V1 payoff for non-airbag products | Low | High | Only apply airbag logic when `airbag_barrier` is not None |
| Edge case: spot exactly at airbag barrier | Low | Medium | Use strict inequality (`spot < airbag_barrier` for standard; `spot > airbag_barrier` for reverse) |

## Success Criteria

1. `get_maturity_payoff_v1()` returns correct airbag payoff when `airbag_config` is configured
2. Standard snowballs (no airbag) produce identical results to before
3. MC engine correctly prices airbag snowballs without code changes
4. All existing tests pass
