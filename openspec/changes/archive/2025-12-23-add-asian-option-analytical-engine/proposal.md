# Proposal: Add Asian Option Analytical Engine

## Summary

Add a new analytical pricing engine for Asian options (`AsianOptionAnalyticalEngine`) that provides closed-form approximation methods for pricing both geometric and arithmetic average-rate options. This complements the existing Monte Carlo engine (`AsianOptionMCEngine`) with faster, deterministic pricing methods.

## Motivation

1. **Performance**: Analytical approximations provide near-instantaneous pricing compared to Monte Carlo simulation (milliseconds vs seconds)
2. **Completeness**: The `AsianOption` product class exists with full observation schedule support, but lacks an analytical engine
3. **Validation**: Analytical prices serve as control variates and validation benchmarks for Monte Carlo
4. **Industry Standard**: The implemented methods (Kemna-Vorst, Turnbull-Wakeman, Levy, Curran) are widely used in production systems

## Scope

### In Scope
- **Geometric average options**: Exact closed-form pricing (Kemna-Vorst approach)
- **Arithmetic average options**: Multiple approximation methods
  - Turnbull-Wakeman (moment matching)
  - Levy approximation
  - Curran (geometric conditioning)
  - Discrete arithmetic (Haug-Haug-Margrabe)
- **Fixed strike** (average price) options
- **Floating strike** (average strike) options via symmetry transformation
- **In-the-averaging-period** pricing with realized average adjustment
- **Integration** with existing `AsianOption` product and `PricingEnvironment`
- **Greeks** calculation (delta, gamma, vega, theta, rho)

### Out of Scope
- Volatility term structure calibration (flat vol assumed for initial implementation)
- Asian options on futures (minor extension, can be added later)
- Path-dependent barrier features

## Approach

### Method Selection Pattern

Follow the established two-level enum pattern from `AmericanOptionAnalyticalEngine`:

```python
from util.enum.engine_enums import EngineType, AsianAnalyticalMethod

# Create enum in util/enum/engine_enums.py
class AsianAnalyticalMethod(Enum):
    KEMNA_VORST = "kemna_vorst"      # Geometric (exact)
    TURNBULL_WAKEMAN = "turnbull_wakeman"  # Arithmetic approx
    LEVY = "levy"                    # Arithmetic approx
    CURRAN = "curran"                # Geometric conditioning
    DISCRETE_HHM = "discrete_hhm"    # Discrete arithmetic (Haug-Haug-Margrabe)

# Usage
engine = AsianOptionAnalyticalEngine(
    method=EngineType.ANALYTICAL(AsianAnalyticalMethod.TURNBULL_WAKEMAN)
)
```

### Method Selection Logic

The engine will auto-select the appropriate method based on option characteristics:

| Option Type | Averaging | Default Method |
|-------------|-----------|----------------|
| Fixed Strike | Geometric | KEMNA_VORST |
| Fixed Strike | Arithmetic | TURNBULL_WAKEMAN |
| Floating Strike | Any | Transform + appropriate method |

Users can override with explicit method selection.

### Floating-Strike Symmetry

Floating-strike options are priced using Henderson-Wojakowski symmetry:
- Floating-strike call = Fixed-strike put with transformed parameters
- This avoids duplicating formulas

### In-Period Pricing

When valuation date is within the averaging period:
1. Extract realized average from `AsianOption.resolve_observations()`
2. Adjust strike: `X_adj = (n * X - m * S_A) / (n - m)`
3. Scale result by `(n - m) / n`

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Accuracy for high volatility (>30%) | Document limitation; fallback to MC for extreme cases |
| Numerical instability near boundaries | Use safe math utilities; handle edge cases explicitly |
| Inconsistency with MC engine | Add validation tests comparing analytical vs MC |

## Success Criteria

1. All geometric average options priced within 1e-6 of theoretical values
2. Arithmetic approximations within 0.5% of Monte Carlo (100k paths) for typical parameters
3. Greeks match finite-difference estimates within 1e-4
4. Full integration with existing `AsianOption` product
5. Performance: <1ms per price calculation

## References

- Haug, E.G. "The Complete Guide to Option Pricing Formulas" (Section 4.20)
- Kemna, A.G.Z. and Vorst, A.C.F. (1990) "A Pricing Method for Options Based on Average Asset Values"
- Turnbull, S.M. and Wakeman, L.M. (1991) "A Quick Algorithm for Pricing European Average Options"
- Levy, E. (1992) "Pricing European Average Rate Currency Options"
- Curran, M. (1992) "Valuing Asian and Portfolio Options by Conditioning on the Geometric Mean Price"
- Henderson, V. and Wojakowski, R. (2001) "On the Equivalence of Floating and Fixed Strike Asian Options"
