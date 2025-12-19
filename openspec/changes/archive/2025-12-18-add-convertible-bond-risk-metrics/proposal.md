# Proposal: Add Convertible Bond Risk Metrics

## Summary
Add DV01, modified duration, convexity, and floor bond metrics to the `ConvertibleBondEngine`. The floor bond represents the straight bond component of a convertible bond without conversion and option features, providing a bond floor valuation and associated risk metrics.

## Motivation
Convertible bonds are hybrid securities requiring both equity and fixed-income risk measures. Currently, the `ConvertibleBondEngine` computes equity-like Greeks (delta, gamma) but lacks bond-specific interest rate sensitivities. Portfolio managers and risk systems need:

1. **DV01** - Dollar value of one basis point for interest rate hedging
2. **CS01** - Credit spread sensitivity (price change per 1bp credit spread move)
3. **Modified Duration** - Percentage price sensitivity to yield changes
4. **Convexity** - Second-order interest rate sensitivity
5. **Floor Bond Price** - The straight bond value assuming no conversion (investment value floor)
6. **Floor Bond DV01/CS01/Duration/Convexity** - Risk metrics for the bond floor component

These metrics enable proper interest rate risk management and help decompose the convertible's value into equity-sensitive and rate-sensitive components.

## Scope

### In Scope
- Add `floor_bond_price()` method to compute the straight bond value (no conversion option)
- Add `dv01()` method for convertible bond price sensitivity to rate changes
- Add `cs01()` method for convertible bond price sensitivity to credit spread changes
- Add `modified_duration()` method for the convertible bond
- Add `convexity()` method for the convertible bond
- Add floor bond risk metrics: `floor_bond_dv01()`, `floor_bond_cs01()`, `floor_bond_duration()`, `floor_bond_convexity()`
- Extend `ConvertibleBondResult` to include these new metrics
- Add unit tests for all new methods
- Update the example/demo file

### Out of Scope
- Key rate durations (partial DV01s by tenor)
- Yield-based duration calculations (already using discount curve approach)
- Changes to the underlying pricing engines (tree/PDE) - metrics computed via numerical bumping

## Approach

### Floor Bond Calculation
The floor bond (also called "investment value" or "straight bond value") is computed by:
1. Taking the convertible bond's coupon schedule and principal repayment
2. Discounting using the risky rate (risk-free rate + credit spread)
3. Ignoring all conversion, call, and put optionality

This gives the value if the holder never converts and the issuer never calls/puts.

### DV01, CS01, Duration, Convexity Calculation
For the full convertible bond, these are computed numerically:
- **DV01**: Bump risk-free rate by 1bp, reprice, measure change
- **CS01**: Bump credit spread by 1bp, reprice, measure change
- **Duration**: DV01 / (Price × 0.0001)
- **Convexity**: Computed via central difference with rate bumps

For the floor bond, these can be computed analytically (similar to `BondDiscountEngine`) since there's no optionality.

### Integration with ConvertibleBondResult
Extend the result dataclass to include:
- `floor_bond_price`: Straight bond value
- `dv01`: DV01 of the convertible
- `cs01`: CS01 of the convertible
- `modified_duration`: Duration of the convertible
- `convexity`: Convexity of the convertible
- `floor_bond_dv01`: DV01 of the floor bond
- `floor_bond_cs01`: CS01 of the floor bond
- `floor_bond_duration`: Duration of the floor bond
- `floor_bond_convexity`: Convexity of the floor bond

## Dependencies
- Existing `ConvertibleBondEngine` and `ConvertibleBondResult`
- `PricingEnvironment` with rate curve
- `ConvertibleBond` product class (already has `get_cashflows()`)

## Risks and Mitigations
| Risk | Mitigation |
|------|------------|
| Numerical noise in DV01 from rate bumping | Use small bump (1bp) and central difference |
| Performance overhead from re-pricing | Cache base price, compute metrics on demand |
| Credit spread handling ambiguity | Use product's credit_spread attribute consistently |

## Success Criteria
- All new methods return values consistent with plain bond analytics (for floor bond)
- DV01 × 100bp ≈ price change for 100bp parallel shift (within numerical tolerance)
- Duration × 0.01 ≈ percentage price change for 1% yield move
- Unit tests pass with known benchmark values
- Demo updated to show new metrics
