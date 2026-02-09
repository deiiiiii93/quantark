---
name: model-developer
description: |
  Production-level model code development following QuantArk patterns and SR 11-7 standards.
  Acts as "Developer A" in the two-developer model validation workflow.
  Use when the user asks to:
  - Implement a new pricing model with production-level rigor
  - Create an engine following model validation standards
  - Develop code as part of a formal model validation workflow
  Triggers: "implement model", "developer A", "production model", "model development", "implement engine"
---

# Model Developer Skill (Developer A)

Production-level model code development following QuantArk patterns and regulatory best practices (SR 11-7). This skill acts as "Developer A" in the two-developer model validation workflow.

## When This Skill Activates

Claude should use this skill when:
- User asks to implement a new pricing model with production rigor
- Part of model validation workflow (invoked by model-orchestrator)
- User explicitly requests Developer A implementation
- User mentions SR 11-7 or model validation standards

## Relationship to Engine-Creator

This skill **extends** the `engine-creator` skill with:
- Mandatory reference documentation
- Stricter input validation requirements
- Explicit edge case documentation
- Performance considerations
- Model validation checklist

**First read**: `.claude/skills/engine-creator/SKILL.md` and `.claude/skills/engine-creator/patterns.md`

---

## Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                MODEL DEVELOPMENT WORKFLOW (Developer A)          │
├─────────────────────────────────────────────────────────────────┤
│ Step 1: Reference Documentation  → Verify/create model docs     │
│ Step 2: Apply Engine-Creator     → Follow standard workflow     │
│ Step 3: Enhanced Validation      → Stricter input checks        │
│ Step 4: Edge Case Documentation  → Explicit handling            │
│ Step 5: Performance Optimization → Speed considerations         │
│ Step 6: Integration Checklist    → Verify completeness          │
│ Step 7: Generate Dev Report      → Document implementation      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Reference Documentation

### Mandatory Reference Verification

**CRITICAL**: Every model implementation MUST have reference documentation.

**Check for existing docs:**
```
asset/<asset_type>/engine/docs/<script_name>.md
docs/<model_name>.md
```

**If no reference exists, CREATE one** following the template:

```markdown
# [Model Name] Reference Documentation

## 1. Model Overview

**Model Type**: [Analytical/MC/PDE/Quadrature]
**Product Supported**: [Product class name]
**Primary Use Case**: [Brief description]

## 2. Mathematical Formulation

### 2.1 Core Formula

$$ V = ... $$

**Variables**:
- $S$: Spot price
- $K$: Strike price
- ...

### 2.2 Greeks Formulas (if analytical)

| Greek | Formula |
|-------|---------|
| Delta | $\Delta = ...$ |
| Gamma | $\Gamma = ...$ |
| ... | ... |

## 3. Assumptions and Limitations

1. [Assumption 1]
2. [Assumption 2]
3. ...

## 4. Numerical Considerations

### 4.1 Edge Cases

| Condition | Expected Behavior | Implementation |
|-----------|-------------------|----------------|
| T → 0 | Intrinsic value | Check is_zero(T) |
| σ → 0 | Deterministic | Check is_zero(sigma) |
| ... | ... | ... |

### 4.2 Numerical Stability

- Use safe_log() for log calculations
- Use safe_exp() for exponentials
- ...

## 5. References

1. [Author, Year, Title, Publication]
2. ...
```

---

## Step 2: Apply Engine-Creator Workflow

Follow the complete `engine-creator` workflow:

1. Script Naming
2. Find Base Class
3. Find Folder Location
4. Find Reference Documentation (already done in Step 1)
5. Assess Helper Requirements
6. Determine Facade Engine Need
7. Register Engine Type
8. Apply Codebase Patterns

**Reference**: `.claude/skills/engine-creator/SKILL.md`

---

## Step 3: Enhanced Input Validation

### Validation Requirements Matrix

| Parameter | Type Check | Range Check | Edge Check | Sanity Check |
|-----------|------------|-------------|------------|--------------|
| strike | is numeric | > 0 | not NaN/Inf | < 1e10 |
| spot | is numeric | > 0 | not NaN/Inf | < 1e10 |
| volatility | is numeric | >= 0 | not NaN/Inf | < 10 (1000%) |
| rate | is numeric | any | not NaN/Inf | > -1 (> -100%) |
| maturity | is numeric | >= 0 | not NaN/Inf | < 100 years |
| dividend | is numeric | >= 0 | not NaN/Inf | < 1 (100%) |

### Validation Code Pattern

```python
from util.numerical import (
    validate_positive, validate_non_negative,
    validate_probability, is_valid_number, Tolerance
)
from util.exceptions import ValidationError

def _validate_inputs(self, product, pricing_env):
    """Comprehensive input validation."""
    # Type checks
    if not isinstance(product, SupportedProductType):
        raise ValidationError(
            f"Expected SupportedProductType, got {type(product).__name__}"
        )

    # Extract values
    S = pricing_env.spot
    K = product.strike
    T = product.get_maturity(pricing_env)
    sigma = pricing_env.volatility
    r = pricing_env.rate
    q = pricing_env.dividend_yield

    # Validate each parameter
    validate_positive(S, "spot")
    validate_positive(K, "strike")
    validate_non_negative(T, "maturity")
    validate_non_negative(sigma, "volatility")

    # Sanity checks
    if S > 1e10:
        raise ValidationError(f"spot {S} implausibly large")
    if K > 1e10:
        raise ValidationError(f"strike {K} implausibly large")
    if sigma > 10:
        raise ValidationError(f"volatility {sigma} (>{10*100}%) implausibly high")
    if T > 100:
        raise ValidationError(f"maturity {T} years implausibly long")
```

---

## Step 4: Edge Case Documentation

### Edge Case Handling Template

For each edge case, document:
1. **Condition**: When does this occur?
2. **Expected Behavior**: What should happen?
3. **Implementation**: How is it handled?
4. **Test Case**: How to verify?

```python
def price(self, product, pricing_env) -> float:
    """
    Price the product.

    Edge Cases Handled:
    -------------------
    1. T = 0: Returns intrinsic value (max(S-K, 0) for call)
    2. σ = 0: Returns discounted intrinsic (deterministic payoff)
    3. Deep ITM (S >> K): Uses asymptotic formula for stability
    4. Deep OTM (S << K): Returns near-zero with proper precision
    5. Near barrier: Applies discrete monitoring adjustment
    """
    # Validate inputs first
    self._validate_inputs(product, pricing_env)

    S = pricing_env.spot
    K = product.strike
    T = product.get_maturity(pricing_env)
    sigma = pricing_env.volatility

    # Edge case 1: Expired option
    if is_zero(T):
        return product.get_payoff(S) * product.contract_multiplier

    # Edge case 2: Zero volatility
    if is_zero(sigma):
        forward = S * safe_exp((r - q) * T)
        df = safe_exp(-r * T)
        return df * product.get_payoff(forward) * product.contract_multiplier

    # Edge case 3: Deep ITM/OTM
    moneyness = S / K
    if moneyness > 100 or moneyness < 0.01:
        return self._asymptotic_price(product, pricing_env)

    # Normal pricing path
    return self._standard_price(product, pricing_env)
```

---

## Step 5: Performance Optimization

### Performance Targets

| Engine Type | Price Target | Greeks Target |
|-------------|--------------|---------------|
| Analytical | < 0.1 ms | < 0.5 ms |
| Quadrature | < 10 ms | < 50 ms |
| PDE | < 100 ms | < 200 ms |
| Monte Carlo | < 1 s (100k paths) | < 5 s |

### Optimization Checklist

- [ ] Vectorized operations where possible
- [ ] No redundant calculations (cache intermediate results)
- [ ] Appropriate data types (float64 for precision, float32 for speed if acceptable)
- [ ] Grid/path generation is efficient
- [ ] No Python loops over array elements (use NumPy)

### Performance Notes Template

```python
"""
Performance Notes:
-----------------
- Time Complexity: O(N) where N is grid size
- Space Complexity: O(N) for solution array
- Bottleneck: Matrix solve in backward stepping
- Optimization: Uses banded matrix solver for tridiagonal system
- Benchmark: ~50ms for 1000-point grid (tested on i7-9700K)
"""
```

---

## Step 6: Integration Checklist

### Production Checklist

Before considering implementation complete:

**Code Quality**
- [ ] All public methods have docstrings
- [ ] Type hints on all function signatures
- [ ] Follows PEP 8 style guide
- [ ] No hardcoded values (use constants)
- [ ] Uses `util.numerical` for safe math

**Validation**
- [ ] All inputs validated at entry
- [ ] Meaningful error messages
- [ ] Edge cases handled explicitly
- [ ] Sanity checks on outputs

**Integration**
- [ ] Added to module `__init__.py` exports
- [ ] Registered in engine type enum (if new type)
- [ ] Contract multiplier applied for equity
- [ ] Compatible with GreeksCalculator

**Documentation**
- [ ] Reference documentation exists
- [ ] Edge cases documented in docstring
- [ ] Performance notes included
- [ ] Examples in docstring or docs/

**Testing** (to be done by validator)
- [ ] Unit tests planned
- [ ] Boundary test cases identified
- [ ] Benchmark cases documented

---

## Step 7: Generate Development Report

### Report Template

```markdown
# Model Development Report / 模型开发报告 (Developer A)

**Model**: [Model Name]
**Date**: <date>
**Developer**: Developer A (Claude)

---

## 1. 实现摘要 / Implementation Summary

### 1.1 Model Specification

| Attribute | Value |
|-----------|-------|
| Model Type | Analytical/MC/PDE/Quadrature |
| Product Supported | [Product class] |
| Base Class | [Base class name] |
| Engine Location | `asset/.../engine/.../` |

### 1.2 Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `path/to/engine.py` | Created | Main engine implementation |
| `path/to/docs/engine.md` | Created | Reference documentation |
| `path/to/__init__.py` | Modified | Added export |

---

## 2. 参考文档 / Reference Documentation

**Location**: `asset/<type>/engine/docs/<engine_name>.md`

### 2.1 Mathematical Basis

[Brief summary of the mathematical approach]

### 2.2 Key Formulas

| Formula | Source |
|---------|--------|
| Core pricing | [Citation] |
| Greeks | [Citation] |

---

## 3. 输入验证 / Input Validation

### 3.1 Validated Parameters

| Parameter | Type | Range | Edge Check | Sanity |
|-----------|------|-------|------------|--------|
| strike | float | > 0 | NaN/Inf | < 1e10 |
| ... | ... | ... | ... | ... |

### 3.2 Validation Code Location

`<file>:<line>` - `_validate_inputs()` method

---

## 4. 边界情况处理 / Edge Case Handling

| Condition | Handling | Code Location |
|-----------|----------|---------------|
| T = 0 | Return intrinsic | `price():XX` |
| σ = 0 | Deterministic price | `price():XX` |
| ... | ... | ... |

---

## 5. 性能考虑 / Performance Considerations

### 5.1 Complexity

- Time: O(...)
- Space: O(...)

### 5.2 Benchmark Results

| Operation | Time | Notes |
|-----------|------|-------|
| Single price | X.XX ms | Baseline case |
| Full Greeks | X.XX ms | Numerical |

### 5.3 Optimizations Applied

1. [Optimization 1]
2. [Optimization 2]

---

## 6. 集成清单 / Integration Checklist

- [x] Reference documentation created
- [x] Inputs validated
- [x] Edge cases handled
- [x] Added to __init__.py
- [x] Contract multiplier applied
- [ ] Tests created (Developer B)

---

## 7. 待验证项 / Items for Validation (Developer B)

1. Independent re-implementation of core formula
2. Boundary check tests
3. Monte Carlo benchmark comparison
4. Edge case test suite

---

## 8. 已知限制 / Known Limitations

1. [Limitation 1]
2. [Limitation 2]

---

## Appendix: Code Snippets

### A. Core Pricing Method

```python
[Include key pricing method code]
```

### B. Edge Case Handling

```python
[Include edge case handling code]
```
```

---

## Integration with Model Validation Workflow

When invoked by model-orchestrator:

### Input
- Model specification (from research phase or user)
- Reference materials
- Output directory path

### Output
- Engine implementation file(s)
- Reference documentation
- Development report (`dev-report.md`)
- List of files created/modified

### Handoff to Developer B

After Developer A completes:
1. All implementation files in specified location
2. Reference documentation complete
3. Development report generated
4. Edge case test scenarios documented

Developer B (model-logic-validator) will:
1. NOT read Developer A's implementation
2. Independently re-implement based on reference docs
3. Compare results against Developer A

---

## Principles

1. **Documentation First**: Reference doc must exist before coding
2. **Validate Everything**: Trust no input
3. **Handle All Edges**: Explicit handling, not crashes
4. **Performance Aware**: Meet targets for engine type
5. **Enable Validation**: Provide clear specs for Developer B
