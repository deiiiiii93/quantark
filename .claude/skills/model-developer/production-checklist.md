# Model Developer Production Checklist

Use this checklist before considering a model implementation complete.

## Pre-Implementation

- [ ] Reference documentation exists or created
- [ ] Mathematical formulas verified against 2+ sources
- [ ] Edge cases identified from literature/experience
- [ ] Performance targets defined

## Implementation

### Code Structure
- [ ] Correct base class inheritance
- [ ] Proper file naming (`{product}_{engine_type}_engine.py`)
- [ ] Correct folder location (`asset/<type>/engine/<method>/`)
- [ ] All imports at top of file

### Documentation
- [ ] Class docstring with method description
- [ ] References cited
- [ ] All public methods have docstrings
- [ ] Type hints on all signatures
- [ ] Edge cases documented in price() docstring

### Input Validation
- [ ] Product type validated
- [ ] All numeric inputs: type check
- [ ] All numeric inputs: range check (positive, non-negative)
- [ ] All numeric inputs: NaN/Inf check
- [ ] Sanity bounds (vol < 10, T < 100, etc.)
- [ ] Meaningful ValidationError messages

### Edge Case Handling
- [ ] T = 0 (expired): returns intrinsic value
- [ ] σ = 0 (zero vol): returns deterministic price
- [ ] Deep ITM: stable calculation
- [ ] Deep OTM: stable calculation (no underflow)
- [ ] Near barrier (if applicable): appropriate handling
- [ ] Negative rates (if possible): handled correctly

### Numerical Stability
- [ ] Uses `util.numerical.safe_log()` for log operations
- [ ] Uses `util.numerical.safe_exp()` for exp operations
- [ ] Uses `util.numerical.safe_sqrt()` for sqrt operations
- [ ] Uses `util.numerical.safe_divide()` for divisions
- [ ] Uses `util.numerical.is_zero()` for near-zero checks
- [ ] No raw `math.log`, `math.sqrt`, `math.exp` on inputs

### Scaling & Contracts
- [ ] `product.contract_multiplier` applied (equity derivatives)
- [ ] `product.denominator` respected (fixed income)
- [ ] Price returned is per-contract (position scales separately)

### Performance
- [ ] Vectorized operations (no Python loops over arrays)
- [ ] No redundant calculations
- [ ] Appropriate caching (if applicable)
- [ ] Meets target: Analytical < 0.1ms, PDE < 100ms

## Integration

### Module Integration
- [ ] Added to `__init__.py` exports
- [ ] `__all__` list updated
- [ ] Engine type enum registered (if new)

### QuantArk Patterns
- [ ] `price()` returns float
- [ ] Does NOT override `calculate_greeks()` (unless analytical)
- [ ] Compatible with `GreeksCalculator`
- [ ] Works with `PricingEnvironment`

## Post-Implementation

### Development Report
- [ ] Implementation summary complete
- [ ] Files created/modified listed
- [ ] Input validation documented
- [ ] Edge cases documented
- [ ] Performance notes included
- [ ] Known limitations documented
- [ ] Items for Developer B listed

### Handoff Preparation
- [ ] Reference doc finalized
- [ ] Test scenarios documented
- [ ] Benchmark values provided (if available)
- [ ] No implementation details that would bias Developer B

---

## Quick Validation Commands

```bash
# Check file exists
ls -la asset/<type>/engine/<method>/<engine_name>.py

# Check syntax
python -m py_compile asset/<type>/engine/<method>/<engine_name>.py

# Quick import test
python -c "from asset.<type>.engine.<method>.<engine_name> import <ClassName>"

# Check __init__.py export
grep "<ClassName>" asset/<type>/engine/<method>/__init__.py
```

---

## Common Issues to Avoid

1. **Missing contract_multiplier**: Price is 100x too small
2. **Raw math.log()**: Crashes on S < K for puts
3. **No T=0 check**: Division by zero in sigma*sqrt(T)
4. **Hardcoded values**: Use constants or parameters
5. **Missing validation**: Crashes on invalid input
6. **Overriding calculate_greeks()**: Unless you have analytical formulas
7. **Not using util.numerical**: Numerical instability
