---
name: model-cross-validator
description: |
  Cross-validate non-Monte Carlo models against MC implementations for consistency.
  Automatically finds corresponding MC engines and runs comprehensive comparisons.
  Use when the user asks to:
  - Cross-validate an analytical/PDE/quadrature engine against MC
  - Compare pricing methods for the same product
  - Verify model convergence
  Triggers: "cross validate", "MC comparison", "benchmark against monte carlo", "compare engines", "convergence test"
---

# Model Cross-Validator Skill

Cross-validate non-Monte Carlo models (analytical, PDE, quadrature) against Monte Carlo implementations to verify pricing consistency.

## When This Skill Activates

Codex should use this skill when:
- Part of model validation workflow (invoked by model-orchestrator)
- User asks to cross-validate against Monte Carlo
- User wants to compare pricing methods
- Verifying a new analytical/PDE/quadrature engine

## Relationship to Engine-Validator

This skill **extends** `engine-validator` benchmark patterns with:
- Automatic MC engine discovery
- Comprehensive parameter space coverage
- Convergence analysis
- Statistical comparison with confidence intervals

**Reference**: `.codex/skills/engine-validator/SKILL.md`

---

## Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                MC CROSS-VALIDATION WORKFLOW                      │
├─────────────────────────────────────────────────────────────────┤
│ Step 1: Identify Target Engine   → Confirm engine to validate   │
│ Step 2: Find MC Benchmark        → Locate corresponding MC      │
│ Step 3: Generate Test Matrix     → Comprehensive parameters     │
│ Step 4: Run Comparison           → Execute both engines         │
│ Step 5: Analyze Results          → Statistics and convergence   │
│ Step 6: Generate Report          → MC comparison report         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Identify Target Engine

### Engine Type Detection

| Engine Type | Skip MC Comparison? |
|-------------|---------------------|
| Analytical | No - compare |
| PDE | No - compare |
| Quadrature | No - compare |
| Tree | No - compare |
| **Monte Carlo** | **YES - skip** |

**If target is MC engine**: Skip cross-validation entirely.

```
The target engine is Monte Carlo.
MC cross-validation is skipped (cannot benchmark MC against MC).
```

---

## Step 2: Find MC Benchmark Engine

### MC Engine Mapping

| Product Type | MC Engine | Location |
|--------------|-----------|----------|
| European Vanilla | `EuroMCEngine` | `asset/equity/engine/mc/euro_mc_engine.py` |
| Asian Option | `AsianOptionMCEngine` | `asset/equity/engine/mc/asian_option_mc_engine.py` |
| Barrier Option | (may need creation) | - |
| Snowball/Autocallable | `SnowballMCEngine` | `asset/equity/engine/mc/snowball_mc_engine.py` |
| Phoenix | `PhoenixMCEngine` | `asset/equity/engine/mc/phoenix_mc_engine.py` |

### Discovery Process

```bash
# Search for MC engine for product type
rg "class.*MCEngine" asset/equity/engine/mc/ --type py

# Check what products the MC engine supports
rg "isinstance.*product" asset/equity/engine/mc/<engine>.py
```

### If No MC Engine Exists

```markdown
**MC Engine Not Found**

No Monte Carlo engine found for [Product Type].
Options:
1. Skip MC cross-validation (proceed without this check)
2. Create MC engine first using engine-creator skill
3. Use alternative benchmark (if available)

Recommendation: [Based on model complexity and risk]
```

---

## Step 3: Generate Test Matrix

### Standard Test Matrix

```python
# Parameter variations
TEST_MATRIX = {
    # Moneyness
    'moneyness': [
        ('ATM', 1.0),    # S/K = 1
        ('ITM', 0.8),    # S/K = 1.25
        ('OTM', 1.2),    # S/K = 0.83
        ('Deep ITM', 0.6),
        ('Deep OTM', 1.4),
    ],

    # Maturity
    'maturity': [
        ('Short', 0.25),   # 3 months
        ('Medium', 1.0),   # 1 year
        ('Long', 3.0),     # 3 years
    ],

    # Volatility
    'volatility': [
        ('Low', 0.10),     # 10%
        ('Medium', 0.25),  # 25%
        ('High', 0.50),    # 50%
    ],

    # Dividend yield
    'dividend': [
        ('None', 0.0),
        ('Low', 0.02),
        ('High', 0.05),
    ],

    # Rate
    'rate': [
        ('Low', 0.01),
        ('Medium', 0.05),
        ('High', 0.10),
    ],
}

# Generate all combinations or strategic subset
def generate_test_cases():
    """Generate comprehensive test cases."""
    cases = []

    # Base case
    cases.append({
        'name': 'Base Case',
        'S': 100, 'K': 100, 'T': 1.0,
        'r': 0.05, 'sigma': 0.20, 'q': 0.0
    })

    # Moneyness sweep
    for name, k_factor in TEST_MATRIX['moneyness']:
        cases.append({
            'name': f'Moneyness: {name}',
            'S': 100, 'K': 100 * k_factor, 'T': 1.0,
            'r': 0.05, 'sigma': 0.20, 'q': 0.0
        })

    # Maturity sweep
    for name, T in TEST_MATRIX['maturity']:
        cases.append({
            'name': f'Maturity: {name}',
            'S': 100, 'K': 100, 'T': T,
            'r': 0.05, 'sigma': 0.20, 'q': 0.0
        })

    # ... continue for other dimensions

    return cases
```

### Product-Specific Test Cases

| Product Type | Additional Test Cases |
|--------------|----------------------|
| Asian | Averaging frequencies, observation counts |
| Barrier | Barrier levels (near/far), rebates |
| Autocallable | Coupon levels, KO barriers, observation dates |
| American | Early exercise boundary proximity |

---

## Step 4: Run Comparison

### MC Configuration

```python
MC_CONFIG = {
    'n_paths_base': 100_000,    # Base number of paths
    'n_paths_high': 1_000_000,  # High precision for discrepancies
    'n_steps': 252,             # Daily steps for 1 year
    'seed': 42,                 # Reproducibility
    'use_antithetic': True,     # Variance reduction
    'confidence_level': 0.95,   # For confidence intervals
}
```

### Comparison Logic

```python
def run_comparison(target_engine, mc_engine, test_cases, mc_config):
    """
    Run comparison between target and MC engines.

    Returns:
        List of comparison results with statistics
    """
    results = []

    for case in test_cases:
        # Create product and environment
        product = create_product(case)
        env = create_environment(case)

        # Get target engine price
        target_price = target_engine.price(product, env)

        # Get MC price with statistics
        mc_result = mc_engine.price_with_stats(
            product, env,
            n_paths=mc_config['n_paths_base'],
            seed=mc_config['seed']
        )

        # Calculate comparison metrics
        mc_price = mc_result['price']
        mc_stderr = mc_result['std_error']

        # Relative error
        if mc_price != 0:
            rel_error = abs(target_price - mc_price) / abs(mc_price)
        else:
            rel_error = abs(target_price)

        # Is target within MC confidence interval?
        ci_lower = mc_price - 1.96 * mc_stderr
        ci_upper = mc_price + 1.96 * mc_stderr
        within_ci = ci_lower <= target_price <= ci_upper

        results.append({
            'case': case['name'],
            'params': case,
            'target_price': target_price,
            'mc_price': mc_price,
            'mc_stderr': mc_stderr,
            'mc_ci': (ci_lower, ci_upper),
            'rel_error': rel_error,
            'within_ci': within_ci,
            'passed': within_ci or rel_error < 0.05  # 5% tolerance OR within CI
        })

    return results
```

---

## Step 5: Analyze Results

### Acceptance Criteria

| Criterion | Threshold | Notes |
|-----------|-----------|-------|
| Relative error | < 5% | Default tolerance |
| Within MC CI | Yes | Target within 95% CI |
| Large error investigation | > 2% | Requires explanation |
| Systematic bias | None | Errors should be random |

### Statistical Analysis

```python
def analyze_results(results):
    """Analyze comparison results statistically."""
    errors = [r['rel_error'] for r in results]

    analysis = {
        'n_cases': len(results),
        'n_passed': sum(1 for r in results if r['passed']),
        'n_failed': sum(1 for r in results if not r['passed']),
        'pass_rate': sum(1 for r in results if r['passed']) / len(results),

        # Error statistics
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'max_error': np.max(errors),
        'std_error': np.std(errors),

        # Bias detection
        'signed_errors': [r['target_price'] - r['mc_price'] for r in results],
    }

    # Test for systematic bias
    signed = analysis['signed_errors']
    analysis['mean_signed_error'] = np.mean(signed)
    analysis['has_bias'] = abs(analysis['mean_signed_error']) > 2 * np.std(signed) / np.sqrt(len(signed))

    return analysis
```

### Convergence Check (for discrepancies)

```python
def check_convergence(target_engine, mc_engine, case, path_counts):
    """
    Check if MC converges to target with more paths.

    Args:
        path_counts: [10000, 50000, 100000, 500000, 1000000]

    Returns:
        Convergence data and assessment
    """
    product = create_product(case)
    env = create_environment(case)

    target_price = target_engine.price(product, env)

    convergence_data = []
    for n_paths in path_counts:
        mc_result = mc_engine.price_with_stats(product, env, n_paths=n_paths)
        error = abs(target_price - mc_result['price']) / abs(target_price)
        convergence_data.append({
            'n_paths': n_paths,
            'mc_price': mc_result['price'],
            'mc_stderr': mc_result['std_error'],
            'rel_error': error
        })

    # Check if error decreases with paths (proportional to 1/sqrt(N))
    errors = [d['rel_error'] for d in convergence_data]
    is_converging = errors[-1] < errors[0] * 0.5  # Should halve

    return {
        'data': convergence_data,
        'is_converging': is_converging,
        'final_error': errors[-1],
        'target_price': target_price
    }
```

---

## Step 6: Generate Report

### Report Template

```markdown
# MC Cross-Validation Report / MC交叉验证报告

**Target Engine 目标引擎**: [Engine Name]
**MC Benchmark MC基准**: [MC Engine Name]
**Date 日期**: <date>
**Status 状态**: PASS / FAIL / PASS_WITH_NOTES

---

## 1. 执行摘要 / Executive Summary

| Metric 指标 | Value 值 |
|------------|---------|
| Test Cases 测试用例 | XX |
| Passed 通过 | XX |
| Failed 失败 | XX |
| Pass Rate 通过率 | XX.X% |
| Mean Error 平均误差 | X.XX% |
| Max Error 最大误差 | X.XX% |
| Systematic Bias 系统偏差 | YES/NO |

---

## 2. 配置 / Configuration

### 2.1 Target Engine 目标引擎

- Type: Analytical/PDE/Quadrature
- Location: `asset/.../engine/.../`
- Parameters: [if any]

### 2.2 MC Benchmark MC基准

- Engine: [MC Engine Name]
- Paths: [n_paths]
- Steps: [n_steps]
- Seed: [seed]
- Variance Reduction: [methods]

---

## 3. 测试矩阵 / Test Matrix

### 3.1 Parameter Ranges

| Parameter | Values |
|-----------|--------|
| Moneyness (S/K) | 0.6, 0.8, 1.0, 1.2, 1.4 |
| Maturity (years) | 0.25, 1.0, 3.0 |
| Volatility | 10%, 25%, 50% |
| Dividend yield | 0%, 2%, 5% |
| Rate | 1%, 5%, 10% |

### 3.2 Total Cases

XX test cases generated

---

## 4. 对比结果 / Comparison Results

### 4.1 Summary by Category

| Category | Passed | Failed | Max Error |
|----------|--------|--------|-----------|
| Moneyness | X/X | X | X.XX% |
| Maturity | X/X | X | X.XX% |
| Volatility | X/X | X | X.XX% |
| Dividend | X/X | X | X.XX% |
| Rate | X/X | X | X.XX% |

### 4.2 Detailed Results

| Case | Target | MC | MC CI (95%) | Error | Status |
|------|--------|-----|-------------|-------|--------|
| Base | 10.45 | 10.47 | [10.42, 10.52] | 0.19% | PASS |
| ATM | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... |

### 4.3 Failed Cases (if any)

| Case | Target | MC | Error | Investigation |
|------|--------|-----|-------|---------------|
| ... | ... | ... | ... | [See Section 5] |

---

## 5. 差异分析 / Discrepancy Analysis

### 5.1 Error Distribution

- Mean Error: X.XX%
- Median Error: X.XX%
- Std Dev: X.XX%
- Max Error: X.XX% (Case: [name])

### 5.2 Bias Analysis

Mean Signed Error: +/- X.XX%
Systematic Bias Detected: YES/NO

[If bias detected, analyze pattern]

### 5.3 Convergence Analysis (for large discrepancies)

| Paths | MC Price | Std Error | Rel Error |
|-------|----------|-----------|-----------|
| 10,000 | ... | ... | ...% |
| 100,000 | ... | ... | ...% |
| 1,000,000 | ... | ... | ...% |

Convergence Assessment: [Is MC converging to target?]

---

## 6. 结论 / Conclusions

### 6.1 Overall Assessment

[Summary of findings]

### 6.2 Decision

**PASS** / **FAIL** / **PASS_WITH_NOTES**

Rationale: [Explain decision]

### 6.3 Notes/Caveats

1. [Note 1]
2. [Note 2]

---

## 7. 建议 / Recommendations

1. [Recommendation 1]
2. [Recommendation 2]

---

## Appendix: Test Code

```python
# Code used for cross-validation
...
```
```

---

## MC Engine Mapping Reference

### Existing MC Engines

| Product | MC Engine | Status |
|---------|-----------|--------|
| European Vanilla | EuroMCEngine | Available |
| Asian (Arithmetic) | AsianOptionMCEngine | Available |
| Asian (Geometric) | AsianOptionMCEngine | Available |
| Snowball | SnowballMCEngine | Available |
| Phoenix | PhoenixMCEngine | Available |
| Barrier | - | May need creation |
| One-Touch | - | May need creation |
| Digital | - | May need creation |

### Creating Missing MC Engine

If MC engine doesn't exist:

1. **Option A**: Skip MC validation with note
2. **Option B**: Create MC engine using `engine-creator`
3. **Option C**: Use existing engine with workaround

Recommendation depends on model complexity and validation requirements.

---

## Integration with Model Validation Workflow

### Input from Orchestrator

- Target engine path
- Product type
- MC configuration (optional)

### Output to Orchestrator

- `mc-comparison-report.md`
- Pass/Fail status
- List of any concerning discrepancies

### Quality Gate

| Criterion | Threshold | Required? |
|-----------|-----------|-----------|
| Pass rate | > 90% | Yes |
| Max error (no explanation) | < 10% | Yes |
| Systematic bias | None | Yes |

---

## Principles

1. **MC is Truth**: For well-configured MC, treat as ground truth
2. **Statistical Rigor**: Use confidence intervals, not just point estimates
3. **Convergence Matters**: Large discrepancies need convergence analysis
4. **Explain or Fix**: Every significant discrepancy needs explanation
5. **No Tolerance Adjustment**: Don't loosen tolerances to pass tests
