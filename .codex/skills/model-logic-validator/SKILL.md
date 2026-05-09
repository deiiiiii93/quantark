---
name: model-logic-validator
description: |
  Independent re-implementation for correctness verification as "Developer B" in model validation workflow.
  Prioritizes clarity over performance and does NOT follow QuantArk patterns to ensure independence.
  Use when the user asks to:
  - Independently verify a model implementation
  - Act as Developer B in model validation
  - Create a gate report for model correctness
  Triggers: "developer B", "logic validation", "independent verification", "gate check", "verify implementation"
---

# Model Logic Validator Skill (Developer B)

Independent re-implementation for correctness verification. This skill acts as "Developer B" in the two-developer model validation workflow, following SR 11-7 model risk management principles.

## Critical Principle: Independence

**CRITICAL**: Developer B MUST NOT look at Developer A's implementation code.

Developer B works ONLY from:
1. Reference documentation
2. Mathematical formulas
3. Research reports
4. Benchmark values

This ensures true independent verification of model logic.

---

## When This Skill Activates

Codex should use this skill when:
- Part of model validation workflow (invoked by model-orchestrator)
- User explicitly requests Developer B verification
- User asks for independent implementation verification
- Gate check is needed for model correctness

## Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                MODEL LOGIC VALIDATION (Developer B)              │
├─────────────────────────────────────────────────────────────────┤
│ Step 1: Read Reference ONLY     → NO access to Dev A code       │
│ Step 2: Independent Implement   → Clarity over performance      │
│ Step 3: Run Comparison          → Compare outputs               │
│ Step 4: Analyze Discrepancies   → Investigate differences       │
│ Step 5: Generate Gate Report    → Pass/Fail decision            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Read Reference Documentation ONLY

### Allowed Sources

| Source | Allowed | Notes |
|--------|---------|-------|
| Reference documentation | YES | Primary source |
| Research report | YES | From model-researcher |
| Benchmark values | YES | For comparison |
| Academic papers | YES | For formula verification |
| Developer A code | **NO** | NEVER read this |
| Developer A report | Partial | Only specs, not implementation details |

### Information to Extract

From reference docs, extract:
1. **Mathematical formulas** - Core pricing equation
2. **Parameter definitions** - What each variable means
3. **Edge case behavior** - Expected behavior at boundaries
4. **Benchmark values** - Known correct values for comparison

---

## Step 2: Independent Implementation

### Implementation Principles

**Prioritize clarity over performance:**

| Aspect | Developer A | Developer B |
|--------|-------------|-------------|
| Goal | Production code | Verification |
| Style | QuantArk patterns | Clear, readable |
| Performance | Optimized | Doesn't matter |
| Vectorization | Required | Optional |
| Caching | Use if helpful | Avoid (simplicity) |

### Implementation Guidelines

```python
"""
Developer B Independent Implementation
======================================
Purpose: Verify Developer A's model logic
Principle: Clarity over performance

DO:
- Write simple, straightforward code
- Use explicit loops (clarity > speed)
- Add verbose comments explaining each step
- Match exactly to mathematical formulas
- Implement edge cases as simple if-statements

DO NOT:
- Look at Developer A's code
- Optimize for performance
- Use complex abstractions
- Follow QuantArk patterns (intentionally different)
- Use utility functions (implement from scratch)
"""

import math
import numpy as np
from scipy.stats import norm

def price_option_independent(S, K, T, r, sigma, q=0.0, is_call=True):
    """
    Independent Black-Scholes implementation for verification.

    This implementation prioritizes clarity and direct formula matching
    over performance or code reuse.

    Formula (from Hull, 10th ed, Chapter 15):
    -----------------------------------------
    d1 = [ln(S/K) + (r - q + sigma^2/2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    Call: C = S * exp(-q*T) * N(d1) - K * exp(-r*T) * N(d2)
    Put:  P = K * exp(-r*T) * N(-d2) - S * exp(-q*T) * N(-d1)
    """
    # Edge case: expired option
    if T <= 0:
        if is_call:
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    # Edge case: zero volatility
    if sigma <= 0:
        forward = S * math.exp((r - q) * T)
        discount = math.exp(-r * T)
        if is_call:
            return discount * max(forward - K, 0)
        else:
            return discount * max(K - forward, 0)

    # Standard calculation - following formula exactly
    sqrt_T = math.sqrt(T)
    sigma_sqrt_T = sigma * sqrt_T

    # d1 calculation - broken down for clarity
    log_moneyness = math.log(S / K)
    drift_term = (r - q + 0.5 * sigma * sigma) * T
    d1 = (log_moneyness + drift_term) / sigma_sqrt_T

    # d2 calculation
    d2 = d1 - sigma_sqrt_T

    # Discount factors
    discount_rate = math.exp(-r * T)
    discount_div = math.exp(-q * T)

    # Option value
    if is_call:
        price = (S * discount_div * norm.cdf(d1)
                 - K * discount_rate * norm.cdf(d2))
    else:
        price = (K * discount_rate * norm.cdf(-d2)
                 - S * discount_div * norm.cdf(-d1))

    return price
```

### DO NOT Use QuantArk Utilities

```python
# Developer B should NOT use:
# from util.numerical import safe_log, is_zero  # NO!
# from asset.equity.engine.base_engine import BaseEngine  # NO!

# Instead, implement directly:
def is_near_zero(x, tol=1e-10):
    """Simple near-zero check for Developer B."""
    return abs(x) < tol

def safe_log_simple(x):
    """Simple safe log for Developer B."""
    if x <= 0:
        return float('-inf')
    return math.log(x)
```

---

## Step 3: Run Comparison

### Comparison Test Cases

Generate comprehensive test cases:

```python
# Test case matrix
test_cases = [
    # Base case (ATM)
    {"S": 100, "K": 100, "T": 1.0, "r": 0.05, "sigma": 0.20, "q": 0.0},

    # Moneyness variations
    {"S": 100, "K": 80,  "T": 1.0, "r": 0.05, "sigma": 0.20, "q": 0.0},  # ITM
    {"S": 100, "K": 120, "T": 1.0, "r": 0.05, "sigma": 0.20, "q": 0.0},  # OTM

    # Maturity variations
    {"S": 100, "K": 100, "T": 0.25, "r": 0.05, "sigma": 0.20, "q": 0.0},  # Short
    {"S": 100, "K": 100, "T": 5.0,  "r": 0.05, "sigma": 0.20, "q": 0.0},  # Long

    # Volatility variations
    {"S": 100, "K": 100, "T": 1.0, "r": 0.05, "sigma": 0.10, "q": 0.0},  # Low vol
    {"S": 100, "K": 100, "T": 1.0, "r": 0.05, "sigma": 0.50, "q": 0.0},  # High vol

    # With dividend
    {"S": 100, "K": 100, "T": 1.0, "r": 0.05, "sigma": 0.20, "q": 0.03},

    # Edge cases
    {"S": 100, "K": 100, "T": 0.001, "r": 0.05, "sigma": 0.20, "q": 0.0},  # Near expiry
    {"S": 100, "K": 100, "T": 1.0, "r": 0.05, "sigma": 0.001, "q": 0.0},  # Near zero vol
]
```

### Comparison Logic

```python
def compare_implementations(dev_a_engine, dev_b_func, test_cases, tolerance=0.001):
    """
    Compare Developer A and Developer B implementations.

    Args:
        dev_a_engine: Developer A's engine instance
        dev_b_func: Developer B's pricing function
        test_cases: List of test parameter dicts
        tolerance: Maximum allowed relative error (default: 0.1%)

    Returns:
        dict with 'passed', 'failed', 'results'
    """
    results = []

    for i, case in enumerate(test_cases):
        # Developer A price
        product = create_product_from_case(case)
        env = create_env_from_case(case)
        price_a = dev_a_engine.price(product, env)

        # Developer B price
        price_b = dev_b_func(
            S=case['S'], K=case['K'], T=case['T'],
            r=case['r'], sigma=case['sigma'], q=case.get('q', 0)
        )

        # Calculate relative error
        if price_b != 0:
            rel_error = abs(price_a - price_b) / abs(price_b)
        else:
            rel_error = abs(price_a - price_b)

        passed = rel_error <= tolerance

        results.append({
            'case_id': i + 1,
            'params': case,
            'price_a': price_a,
            'price_b': price_b,
            'rel_error': rel_error,
            'passed': passed
        })

    passed_count = sum(1 for r in results if r['passed'])
    failed_count = len(results) - passed_count

    return {
        'passed': passed_count,
        'failed': failed_count,
        'total': len(results),
        'results': results,
        'gate_passed': failed_count == 0
    }
```

---

## Step 4: Analyze Discrepancies

### When Results Don't Match

If any test case fails, investigate:

1. **Check formulas**: Are both implementations using same formula?
2. **Check edge cases**: Is one handling edge case differently?
3. **Check parameters**: Are parameters being interpreted the same?
4. **Check units**: Time in years vs days? Rate decimal vs percent?
5. **Check conventions**: Sign conventions, call/put definitions?

### Discrepancy Investigation Template

```markdown
## Discrepancy Analysis

### Test Case: #X

**Parameters**: S=100, K=105, T=0.5, r=0.05, sigma=0.20, q=0.02

| Implementation | Price |
|----------------|-------|
| Developer A | 5.1234 |
| Developer B | 5.2345 |
| Relative Error | 2.17% |

### Investigation

**Hypothesis 1**: Different d1 calculation
- Dev A d1 = 0.1234
- Dev B d1 = 0.1256
- Difference: 1.8%

**Root Cause**: Developer A uses (r - q - 0.5*sigma^2) while reference uses (r - q + 0.5*sigma^2)

**Recommendation**: Check Developer A formula at line XX
```

---

## Step 5: Generate Gate Report

### Gate Criteria

| Criterion | Threshold | Weight |
|-----------|-----------|--------|
| Price match | 0.1% relative error | Primary |
| Greeks match (if available) | 1.0% relative error | Secondary |
| Edge cases | Consistent behavior | Required |
| No unexplained divergences | 0 | Required |

### Gate Decision

| Scenario | Decision |
|----------|----------|
| All tests pass (< 0.1% error) | **PASS** |
| Minor discrepancies (0.1% - 1%) with explanation | **PASS with notes** |
| Unexplained discrepancies > 1% | **FAIL** |
| Edge case behavior differs | **FAIL** |

### Gate Report Template

```markdown
# Gate Report / 门禁报告 (Developer B)

**Model**: [Model Name]
**Date**: <date>
**Developer**: Developer B (Codex)
**Status**: PASS / FAIL

---

## 1. 验证摘要 / Validation Summary

| Metric | Result |
|--------|--------|
| Test Cases | XX |
| Passed | XX |
| Failed | XX |
| Pass Rate | XX% |
| Max Error | X.XX% |
| Gate Decision | **PASS / FAIL** |

---

## 2. 测试结果 / Test Results

### 2.1 Price Comparison

| Case | Parameters | Dev A | Dev B | Error | Status |
|------|------------|-------|-------|-------|--------|
| 1 | S=100, K=100, T=1 | 10.45 | 10.45 | 0.00% | PASS |
| 2 | S=100, K=80, T=1 | 21.23 | 21.24 | 0.05% | PASS |
| ... | ... | ... | ... | ... | ... |

### 2.2 Edge Case Behavior

| Edge Case | Dev A Behavior | Dev B Behavior | Match |
|-----------|----------------|----------------|-------|
| T = 0 | Returns intrinsic | Returns intrinsic | YES |
| sigma = 0 | Returns disc. fwd | Returns disc. fwd | YES |
| ... | ... | ... | ... |

---

## 3. 差异分析 / Discrepancy Analysis

### 3.1 Discrepancies Found

[List any discrepancies and their investigation]

### 3.2 Root Causes

[Identified root causes for any differences]

### 3.3 Resolutions

[How discrepancies were resolved or explained]

---

## 4. 独立实现 / Independent Implementation

### 4.1 Implementation Approach

[Brief description of Developer B's approach]

### 4.2 Formula Used

$$ V = ... $$

### 4.3 Code Location

Independent implementation: `model-validation-output/<model>/validation/independent-impl/`

---

## 5. 门禁决定 / Gate Decision

### Decision: PASS / FAIL

**Rationale**:
[Explain why the gate passed or failed]

### Conditions (if PASS with notes):
[Any conditions or caveats]

### Required Actions (if FAIL):
1. [Action 1 for Developer A]
2. [Action 2]
...

---

## 6. 建议 / Recommendations

1. [Recommendation 1]
2. [Recommendation 2]

---

## Appendix: Developer B Code

```python
[Include complete independent implementation]
```
```

---

## Integration with Model Validation Workflow

### Input from Orchestrator

- Reference documentation path
- Research report path
- Developer A output location (for comparison only, not code reading)
- Benchmark values

### Output to Orchestrator

- `gate-report.md` in designated output directory
- Independent implementation files
- Pass/Fail status
- If FAIL: specific issues for Developer A to address

### Rollback Trigger

If gate FAILS:
1. Orchestrator receives FAIL status
2. Orchestrator notifies user
3. Developer A issues documented
4. Workflow rolls back to development step
5. Developer A must address issues
6. Re-run Developer B validation

---

## Independence Verification

To ensure independence, Developer B should:

1. **Start fresh**: Don't reference any prior knowledge of Developer A's approach
2. **Use reference only**: Only read mathematical specifications
3. **Implement differently**: Intentionally use different variable names, structure
4. **Don't optimize**: Performance differences help identify implementation differences
5. **Document sources**: Cite where each formula comes from

---

## Principles

1. **Independence Above All**: Never look at Developer A code
2. **Clarity Over Performance**: Simple, readable implementation
3. **Formula Fidelity**: Match mathematical references exactly
4. **Thorough Comparison**: Test across parameter space
5. **Honest Reporting**: Report all discrepancies, even small ones
