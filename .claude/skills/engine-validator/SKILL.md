---
name: engine-validator
description: |
  Validate pricing engine scripts by performing comprehensive checks and generating validation reports.
  Use when the user asks to:
  - Validate a pricing engine
  - Create a validation report for an engine
  - Perform boundary checks on an engine
  - Benchmark an engine against Monte Carlo
  - Check theoretical relationships for an engine
  Triggers: "validate engine", "validation report", "boundary check", "benchmark check", "verify engine"
---

# Engine Validator Skill

Automatically validate pricing engine scripts and generate comprehensive validation reports following established QuantArk patterns.

## When This Skill Activates

Claude should use this skill when:
- User asks to validate/verify a pricing engine
- User wants a validation report for an engine
- User requests boundary checks or benchmark comparisons
- User mentions "validate", "verify", "check", or "report" with engine context

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENGINE VALIDATION WORKFLOW                    │
├─────────────────────────────────────────────────────────────────┤
│ Step 1: Identify Engine          → Confirm with user            │
│ Step 2: Method Description       → Analytical: formulas         │
│                                  → MC: process + payoff         │
│ Step 3: Boundary Checks          → Extreme cases + relationships │
│ Step 4: Benchmark Check          → Compare with MC engine       │
│                                  → SKIP if validating MC engine │
│ Step 5: User Case Check          → Validate provided cases      │
│ Step 6: Generate Report          → Create validation report     │
└─────────────────────────────────────────────────────────────────┘
```

### Engine Type Detection

Determine engine type from file path or name:
- **Analytical**: `engine/analytical/*.py` or `*_analytical_engine.py`
- **Monte Carlo**: `engine/mc/*.py` or `*_mc_engine.py`
- **PDE**: `engine/pde/*.py` or `*_pde_solver.py`
- **Quadrature**: `engine/quad/*.py` or `*_quad_engine.py`

---

## Step 1: Identify Engine

### Engine Recognition

Parse user prompt to identify the target engine. Common patterns:

| User Input Pattern | Target Engine |
|-------------------|---------------|
| "european vanilla" / "black scholes" | `black_scholes_engine.py` |
| "american option" | `american_option_engine.py` |
| "barrier option" / "knock out" / "knock in" | `barrier_analytical_engine.py` |
| "asian option" | `asian_option_analytical_engine.py` |
| "one touch" / "no touch" | `one_touch_analytical_engine.py` |
| "digital option" / "binary" | `digital_option_engine.py` |

### Locate Engine File

```bash
# Search for engine in asset directories
find asset/ -name "*${engine_name}*engine*.py" -type f
```

### Confirm with User

**MANDATORY**: Before proceeding, present the recognized engine to the user for confirmation:

```
I identified the engine as: `asset/equity/engine/analytical/barrier_analytical_engine.py`

Is this the correct engine to validate? [Yes/No]
```

Wait for user confirmation before proceeding.

---

## Step 2: Method Description

### Locate Reference Documentation

**Reference Location Pattern:**
```
asset/<asset_type>/engine/docs/<script_name>.md
```

Example mappings:
- `barrier_analytical_engine.py` → `barrier_analytical_engine.md`
- `one_touch_analytical_engine.py` → `onetouch_analytical_engine.md`
- `asian_option_analytical_engine.py` → `asian_option_analytical_engine.md`
- `asian_option_mc_engine.py` → `asian_option_mc_engine.md`

Also check:
- `docs/` in project root
- Web search for academic references

---

### For ANALYTICAL Engines: Formula Comparison

Compare the engine implementation against reference documentation:

| Aspect | Check |
|--------|-------|
| **Core Formula** | Does the implementation match the mathematical formulas? |
| **Parameter Handling** | Are all parameters correctly defined and used? |
| **Edge Cases** | Are numerical edge cases handled (near-expiry, deep ITM/OTM)? |
| **Numerical Stability** | Are `util.numerical` utilities used correctly? |
| **Barrier Adjustment** | For barrier options: Is discrete monitoring shift applied? |
| **Greeks** | Are analytical Greeks provided where formulas exist? |
| **Scaling** | Is `contract_multiplier` applied for equity engines? |

### Pricing Scale Validation (CRITICAL)

**IMPORTANT: Verify that engines apply correct scaling for asset class.**

QuantArk uses a two-stage scaling model:
1. **Engines** return per-contract prices scaled by `contract_multiplier` (equity) or `denominator` (bonds)
2. **Positions** apply quantity scaling for total market value

**For Equity Derivatives:**

| Check | How to Verify |
|-------|---------------|
| contract_multiplier applied | Search for `* product.contract_multiplier` or `price *=" pattern |
| Scaling at correct location | Must be final step before return, NOT before Greeks calculation |
| MC scaling | Both price and std_error scaled (see `euro_mc_engine.py:161-162`) |

**Validation Test:**
```python
# Test: Verify contract_multiplier scaling
product = EuropeanVanillaOption(strike=100, contract_multiplier=100)  # 100 shares
engine = BlackScholesEngine()

# Price with multiplier=100 should be 100× price with multiplier=1
price_100 = engine.price(product, pricing_env)
product.contract_multiplier = 1
price_1 = engine.price(product, pricing_env)

assert abs(price_100 - price_1 * 100) < 1e-10, "Multiplier scaling incorrect"
```

**For Fixed Income:**

| Check | How to Verify |
|-------|---------------|
| denominator handling | Price should be for full denominator (e.g., $1000 notional) |
| Clean vs dirty | Clean price excludes accrued, dirty includes it |
| DV01 scaling | DV01 should be per bond, scaled by denominator |

**Validation Test:**
```python
# Test: Verify denominator scaling
bond = FixedBond(denominator=1000, coupon_rate=0.05)
engine = BondDiscountEngine()

dirty = engine.dirty_price(bond, valuation_date, valuation_date)

# Price should be in range that reflects $1000 notional
# (e.g., if bond is at par, dirty_price ≈ 1000 + accrued)
assert 0 < dirty < 5000, f"Price {dirty} outside reasonable range for denom=1000"
```

**Common Scaling Bugs to Check:**

| Bug | Symptom | Fix |
|-----|---------|-----|
| Missing multiplier | Prices too small (factor of 100-10000) | Add `* product.contract_multiplier` |
| Scaling before Greeks | Greeks also scaled incorrectly | Scale only final price, not intermediate values |
| Double scaling | Prices too large | Check not scaling by both multiplier AND quantity |
| Wrong denominator | Bond prices don't match market | Use `bond.get_denominator()` consistently |

---

### For MONTE CARLO Engines: Process & Payoff Validation

**Focus Areas for MC Engine Validation:**

#### 1. Stochastic Process Check

| Aspect | Check |
|--------|-------|
| **Process Type** | Is the correct process used (GBM, Heston, Local Vol, etc.)? |
| **SDE Implementation** | Does the discretization match reference (Euler, Milstein)? |
| **Drift Term** | Is drift correctly computed (risk-neutral: r - q)? |
| **Diffusion Term** | Is volatility applied correctly (σ × S × dW)? |
| **Time Discretization** | Is dt computed correctly (T / n_steps)? |
| **Random Number Generation** | Proper seeding, antithetic variates, quasi-random? |
| **Correlation** | For multi-asset: is Cholesky decomposition correct? |

**Process Validation Checklist:**
```python
# GBM Process: dS = (r-q)S dt + σS dW
# Euler discretization: S(t+dt) = S(t) * exp((r-q-0.5σ²)dt + σ√dt * Z)

# Check:
# 1. Is drift = (r - q - 0.5 * sigma^2)?
# 2. Is diffusion = sigma * sqrt(dt)?
# 3. Is the exponential form used (log-normal)?
```

#### 2. Payoff Implementation Check

| Aspect | Check |
|--------|-------|
| **Payoff Formula** | Does payoff match product specification? |
| **Path Dependency** | Are path-dependent features correctly tracked? |
| **Observation Dates** | Are discrete observations handled correctly? |
| **Averaging Method** | Arithmetic vs geometric, discrete vs continuous? |
| **Barrier Monitoring** | Continuous vs discrete, adjustment applied? |
| **Early Exercise** | For American: is exercise logic correct? |
| **Rebate/Coupon** | Are auxiliary payments included? |

**Payoff Validation Checklist:**
```python
# European Call: max(S_T - K, 0)
# Asian Call (arithmetic): max(A_T - K, 0) where A_T = mean(S_t1, ..., S_tn)
# Barrier KO Call: max(S_T - K, 0) if max(S_t) < H else rebate

# Check:
# 1. Is the payoff formula correct?
# 2. Are all path points correctly recorded?
# 3. Is discounting applied correctly?
```

#### 3. MC-Specific Implementation Check

| Aspect | Check |
|--------|-------|
| **Discounting** | Is payoff discounted by exp(-r*T)? |
| **Averaging** | Is mean computed over all paths? |
| **Standard Error** | Is SE = std / sqrt(n_paths) computed? |
| **Variance Reduction** | Antithetic, control variate, importance sampling? |
| **Convergence** | Does price stabilize with more paths? |

---

### Document Findings

Record any discrepancies between implementation and reference:
- Formula/process mismatches
- Missing edge case handling
- Numerical stability issues
- Payoff calculation errors
- Potential bugs

---

## Step 3: Boundary Checks

### Types of Boundary Checks

**1. Extreme Market Cases**

| Test Case | Expected Behavior |
|-----------|------------------|
| Very low volatility (σ → 0) | Option value converges to intrinsic value |
| Very high volatility (σ → ∞) | Call → S, Put → K×e^(-rT) for European |
| Near expiry (T → 0) | Value → max(payoff, 0) |
| Deep ITM | Delta → ±1, value → intrinsic + time value |
| Deep OTM | Delta → 0, value → 0 |
| Zero interest rate (r = 0) | No discounting effect |
| At barrier (S = H) | Barrier options: knockout value = rebate |
| Spot at strike (S = K) | ATM option behavior |

**2. Theoretical Relationship Checks**

| Relationship | Formula/Check |
|--------------|---------------|
| Put-Call Parity | C - P = S×e^(-qT) - K×e^(-rT) |
| Barrier Bounds | KO + KI = Vanilla (same barrier, no rebate) |
| American ≥ European | American option ≥ European option |
| Call Spread | C(K1) ≥ C(K2) when K1 < K2 |
| Butterfly Spread | C(K1) + C(K3) ≥ 2×C(K2) |
| Calendar Spread | Long-dated ≥ short-dated (American) |
| Monotonicity | Delta in [0,1] for calls, [-1,0] for puts |
| Convexity | Gamma ≥ 0 |
| Homogeneity | C(λS, λK) = λ×C(S, K) |

### Script Generation

Generate boundary check script at:
```
asset/<type>/engine/validation/script/boundary_check_<engine_name>.py
```

**Template Structure:**
```python
"""
Boundary Check Script for <Engine Name>
Generated: <date>
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from asset.equity.product.option.<product> import <Product>
from asset.equity.engine.analytical.<engine> import <Engine>
from priceenv.pricing_environment import PricingEnvironment
from param.spot_quote import SpotQuote
from param.rate_curve import FlatRateCurve
from param.vol_surface import FlatVolSurface
from param.dividend import ContinuousDividendYield

class BoundaryCheckResults:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []

    def add_result(self, test_name: str, passed: bool, message: str):
        if passed:
            self.passed.append((test_name, message))
        else:
            self.failed.append((test_name, message))

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*60}")
        print(f"BOUNDARY CHECK SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total}")
        print(f"Passed: {len(self.passed)} ({100*len(self.passed)/total:.1f}%)")
        print(f"Failed: {len(self.failed)} ({100*len(self.failed)/total:.1f}%)")
        if self.failed:
            print(f"\nFailed Tests:")
            for name, msg in self.failed:
                print(f"  - {name}: {msg}")
        return len(self.failed) == 0

def create_pricing_env(spot, rate, vol, div=0.0):
    """Helper to create pricing environment."""
    return PricingEnvironment(
        spot=SpotQuote(spot),
        rate_curve=FlatRateCurve(rate),
        vol_surface=FlatVolSurface(vol),
        dividend=ContinuousDividendYield(div)
    )

# ============================================================
# EXTREME MARKET CASE TESTS
# ============================================================

def test_low_volatility(results: BoundaryCheckResults):
    """Test: Low volatility → intrinsic value"""
    # Implementation here
    pass

def test_near_expiry(results: BoundaryCheckResults):
    """Test: Near expiry → payoff"""
    # Implementation here
    pass

def test_deep_itm(results: BoundaryCheckResults):
    """Test: Deep ITM behavior"""
    # Implementation here
    pass

def test_deep_otm(results: BoundaryCheckResults):
    """Test: Deep OTM behavior"""
    # Implementation here
    pass

# ============================================================
# THEORETICAL RELATIONSHIP TESTS
# ============================================================

def test_put_call_parity(results: BoundaryCheckResults):
    """Test: Put-Call Parity"""
    # Implementation here
    pass

def test_monotonicity(results: BoundaryCheckResults):
    """Test: Price monotonicity in parameters"""
    # Implementation here
    pass

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    results = BoundaryCheckResults()
    
    # Extreme market cases
    test_low_volatility(results)
    test_near_expiry(results)
    test_deep_itm(results)
    test_deep_otm(results)
    
    # Theoretical relationships
    test_put_call_parity(results)
    test_monotonicity(results)
    
    # Print summary
    success = results.summary()
    sys.exit(0 if success else 1)
```

### Product-Specific Boundary Checks

| Product Type | Additional Checks |
|--------------|------------------|
| **Barrier Options** | At-barrier value, barrier vs strike relationship, KO+KI=Vanilla |
| **Asian Options** | Geometric ≤ Arithmetic, averaging effect (lower vol) |
| **American Options** | American ≥ European, early exercise boundary |
| **Digital Options** | Value bounded by rebate × discount factor |
| **One-Touch Options** | Value bounded by [0, rebate × df] |

---

## Step 4: Benchmark Check

### IMPORTANT: Skip Benchmark for MC Engines

**If the validated engine IS a Monte Carlo engine, SKIP this entire step.**

Inform user:
```
The validated engine is a Monte Carlo engine.
Benchmark check is skipped (cannot benchmark MC against MC).
Validation relies on boundary checks and process/payoff verification.
```

---

### For ANALYTICAL, PDE, or QUADRATURE Engines: Identify MC Benchmark

**Available MC Engines:**
```
asset/equity/engine/mc/
├── euro_mc_engine.py         # European vanilla
├── asian_option_mc_engine.py # Asian options
├── snowball_mc_engine.py     # Snowball/autocallable
```

**Mapping:**
| Engine to Validate | MC Benchmark |
|-------------------|--------------|
| `black_scholes_engine.py` | `euro_mc_engine.py` |
| `asian_option_analytical_engine.py` | `asian_option_mc_engine.py` |
| `barrier_analytical_engine.py` | *(May need creation)* |
| `one_touch_analytical_engine.py` | *(May need creation)* |
| `snowball_pde_solver.py` | `snowball_mc_engine.py` |
| `snowball_quad_engine.py` | `snowball_mc_engine.py` |

### If No MC Engine Exists for Analytical Engine

Inform user:
```
No Monte Carlo engine found for <product>.
Benchmark check will be skipped.
Consider creating: asset/<type>/engine/mc/<product>_mc_engine.py
```

### Default Pass Criteria

- **Default tolerance**: 5% relative error
- **User override**: Accept criteria from prompt (e.g., "use 1% tolerance")

### IMPORTANT: Do NOT Adjust Tolerances

**CRITICAL RULE**: NEVER adjust tolerances to make tests pass.

```
❌ WRONG: Changing tolerance from 5% to 15% because some tests fail
❌ WRONG: Adding separate tolerance values for different methods to achieve "all pass"
❌ WRONG: Hiding failed tests to improve reported pass rate

✅ CORRECT: Report actual results with original tolerance
✅ CORRECT: Document failures in validation report with explanation
✅ CORRECT: Investigate and explain WHY tests fail (e.g., continuous vs discrete, approximation limits)
```

**When tests fail outside tolerance:**
1. Record actual error values in the report
2. Investigate the root cause (formula mismatch, MC convergence, model assumptions)
3. Document known limitations (e.g., "continuous vs discrete averaging creates known bias")
4. Provide recommendations for further investigation (e.g., "increase MC paths to 1M+")

**The validation report should reflect REALITY, not a manufactured "all pass" result.**

### Script Generation

Generate benchmark script at:
```
asset/<type>/engine/validation/script/benchmark_check_<engine_name>.py
```

**Template Structure:**
```python
"""
Benchmark Check Script for <Engine Name>
Benchmark: Monte Carlo Engine
Generated: <date>
Default Tolerance: 5%
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from asset.equity.product.option.<product> import <Product>
from asset.equity.engine.analytical.<engine> import <AnalyticalEngine>
from asset.equity.engine.mc.<mc_engine> import <MCEngine>
from priceenv.pricing_environment import PricingEnvironment
from param.spot_quote import SpotQuote
from param.rate_curve import FlatRateCurve
from param.vol_surface import FlatVolSurface
from param.dividend import ContinuousDividendYield

TOLERANCE = 0.05  # 5% relative error

class BenchmarkResults:
    def __init__(self, tolerance: float = TOLERANCE):
        self.tolerance = tolerance
        self.results = []

    def add_result(self, case_name: str, analytical: float, mc: float):
        if mc != 0:
            rel_error = abs(analytical - mc) / abs(mc)
        else:
            rel_error = abs(analytical - mc)
        passed = rel_error <= self.tolerance
        self.results.append({
            'case': case_name,
            'analytical': analytical,
            'mc': mc,
            'rel_error': rel_error,
            'passed': passed
        })

    def summary(self):
        passed = sum(1 for r in self.results if r['passed'])
        total = len(self.results)
        
        print(f"\n{'='*80}")
        print(f"BENCHMARK CHECK SUMMARY (Tolerance: {self.tolerance*100:.1f}%)")
        print(f"{'='*80}")
        print(f"{'Case':<30} {'Analytical':>12} {'MC':>12} {'Error':>10} {'Status':>8}")
        print(f"{'-'*80}")
        
        for r in self.results:
            status = "PASS" if r['passed'] else "FAIL"
            print(f"{r['case']:<30} {r['analytical']:>12.4f} {r['mc']:>12.4f} "
                  f"{r['rel_error']*100:>9.2f}% {status:>8}")
        
        print(f"{'-'*80}")
        print(f"Passed: {passed}/{total} ({100*passed/total:.1f}%)")
        
        return passed == total

def create_pricing_env(spot, rate, vol, div=0.0):
    """Helper to create pricing environment."""
    return PricingEnvironment(
        spot=SpotQuote(spot),
        rate_curve=FlatRateCurve(rate),
        vol_surface=FlatVolSurface(vol),
        dividend=ContinuousDividendYield(div)
    )

# ============================================================
# TEST CASES
# ============================================================

def run_benchmark_tests(results: BenchmarkResults):
    """Run all benchmark test cases."""
    
    # Base case
    env = create_pricing_env(spot=100, rate=0.05, vol=0.2)
    product = <Product>(strike=100, maturity=1.0, is_call=True)
    
    analytical_engine = <AnalyticalEngine>()
    mc_engine = <MCEngine>(n_paths=100000, n_steps=252)
    
    analytical_price = analytical_engine.price(product, env)
    mc_price = mc_engine.price(product, env)
    
    results.add_result("ATM Call T=1Y", analytical_price, mc_price)
    
    # Add more test cases...

if __name__ == "__main__":
    results = BenchmarkResults(tolerance=TOLERANCE)
    run_benchmark_tests(results)
    success = results.summary()
    sys.exit(0 if success else 1)
```

---

## Step 5: User Case Check

### Parse User Cases

Look for patterns in user prompt:
```
"S0=100, K=105, T=0.5, r=0.05, σ=0.2, value=5.23"
"spot 100, strike 100, maturity 1 year, expected price 10"
```

### If No Cases Provided

Skip this step with message:
```
No user test cases provided. Skipping user case validation.
To add test cases, provide them in format: "S=100, K=100, T=1, expected=10.5"
```

### Validation Logic

```python
def validate_user_case(case: dict, engine, product_class, pricing_env):
    product = product_class(**case['product_params'])
    calculated = engine.price(product, pricing_env)
    expected = case['expected_value']
    
    rel_error = abs(calculated - expected) / abs(expected)
    return {
        'calculated': calculated,
        'expected': expected,
        'rel_error': rel_error,
        'passed': rel_error < 0.01  # 1% tolerance for user cases
    }
```

---

## Step 6: Generate Final Report

### Report Location

```
asset/<type>/engine/validation/report/<engine_name>_validation_report.md
```

### Report Template

```markdown
# Validation Report: <Engine Name>

**Generated**: <date>
**Engine**: `<path/to/engine.py>`
**Reference**: `<path/to/reference.md>`

---

## Executive Summary

| Check Type | Status | Pass Rate |
|------------|--------|-----------|
| Method Implementation | ✅/⚠️/❌ | - |
| Boundary Checks | ✅/⚠️/❌ | XX% |
| Benchmark Checks | ✅/⚠️/❌ | XX% |
| User Cases | ✅/⚠️/❌ | XX% |

**Overall Status**: ✅ VALIDATED / ⚠️ WARNINGS / ❌ FAILED

---

## 1. Method Description

### 1.1 Pricing Method Summary

<Brief description of the pricing method>

### 1.2 Reference Comparison

| Aspect | Reference | Implementation | Match |
|--------|-----------|----------------|-------|
| Core Formula | ... | ... | ✅/❌ |
| Edge Cases | ... | ... | ✅/❌ |
| Numerical Stability | ... | ... | ✅/❌ |

### 1.3 Issues Found

<List any discrepancies or potential bugs>

---

## 2. Boundary Checks

**Script**: `asset/<type>/engine/validation/script/boundary_check_<engine>.py`

### 2.1 Extreme Market Cases

| Test | Status | Notes |
|------|--------|-------|
| Low volatility | ✅/❌ | ... |
| Near expiry | ✅/❌ | ... |
| Deep ITM | ✅/❌ | ... |
| Deep OTM | ✅/❌ | ... |

### 2.2 Theoretical Relationships

| Relationship | Status | Notes |
|--------------|--------|-------|
| Put-Call Parity | ✅/❌ | ... |
| Monotonicity | ✅/❌ | ... |
| ... | ... | ... |

---

## 3. Benchmark Comparison

**Benchmark Engine**: `<path/to/mc_engine.py>`
**Tolerance**: 5%
**Script**: `asset/<type>/engine/validation/script/benchmark_check_<engine>.py`

| Case | Analytical | MC | Error | Status |
|------|-----------|-----|-------|--------|
| ATM Call T=1Y | ... | ... | ...% | ✅/❌ |
| ... | ... | ... | ... | ... |

---

## 4. User Test Cases

| Case | Expected | Calculated | Error | Status |
|------|----------|------------|-------|--------|
| ... | ... | ... | ...% | ✅/❌ |

*(Skip if no user cases provided)*

---

## 5. Recommendations

1. <Recommendation 1>
2. <Recommendation 2>
...

---

## Appendix

### A. Test Environment

- Python version: X.X
- NumPy version: X.X
- SciPy version: X.X

### B. Script Execution Commands

```bash
# Run boundary checks
python asset/<type>/engine/validation/script/boundary_check_<engine>.py

# Run benchmark checks
python asset/<type>/engine/validation/script/benchmark_check_<engine>.py
```
```

---

## Output Structure

After validation, the following files should exist:

```
asset/<type>/engine/validation/
├── script/
│   ├── boundary_check_<engine_name>.py
│   └── benchmark_check_<engine_name>.py
└── report/
    └── <engine_name>_validation_report.md
```

---

## Reference Files

Study these for patterns:
- Reference doc example: `asset/equity/engine/docs/asian_option_analytical_engine.md`
- Analytical engine: `asset/equity/engine/analytical/black_scholes_engine.py`
- MC engine: `asset/equity/engine/mc/euro_mc_engine.py`
- Engine creator skill: `.claude/skills/engine-creator/SKILL.md`

---

## Common Theoretical Checks by Product Type

### European Vanilla Options
- Put-Call Parity: C - P = S×e^(-qT) - K×e^(-rT)
- Lower bounds: C ≥ max(0, S×e^(-qT) - K×e^(-rT)), P ≥ max(0, K×e^(-rT) - S×e^(-qT))
- Delta bounds: 0 ≤ Δ_call ≤ 1, -1 ≤ Δ_put ≤ 0
- Gamma positive: Γ ≥ 0
- Theta negative: Θ < 0 (for long positions)

### Barrier Options
- KO + KI = Vanilla (same barrier, no rebate)
- KO ≤ Vanilla
- Down-and-out call with H < K: value decreases as H increases
- Up-and-out put with H > K: value decreases as H decreases

### Asian Options
- Geometric average ≤ Arithmetic average (by Jensen's inequality)
- Asian option ≤ Vanilla option (averaging reduces volatility)
- As n → ∞, discrete → continuous

### American Options
- American ≥ European
- Early exercise boundary exists for puts (and dividend calls)
- At expiry: American = European = max(payoff, 0)

### Digital/Binary Options
- Digital call + Digital put = e^(-rT) (complementary payoffs)
- Value bounded by [0, rebate × e^(-rT)]

### One-Touch Options
- Up-and-in + Up-and-out (no rebate) = e^(-rT) for unit payoff
- Value bounded by [0, rebate × e^(-rT)]
