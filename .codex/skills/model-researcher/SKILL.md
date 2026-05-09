---
name: model-researcher
description: |
  Research and cross-check model implementation materials using internet sources.
  Verifies mathematical formulas, finds reference implementations, and identifies edge cases.
  Use when the user asks to:
  - Research a pricing model or algorithm
  - Verify mathematical formulas against authoritative sources
  - Find reference implementations or benchmarks
  - Identify numerical considerations and edge cases
  - Cross-check academic papers or documentation
  Triggers: "research model", "verify formulas", "cross-check implementation", "find references", "literature review"
---

# Model Researcher Skill

Research and cross-check model implementation materials against authoritative sources, academic papers, and reference implementations.

## When This Skill Activates

Codex should use this skill when:
- User asks to research a pricing model or algorithm
- User wants to verify mathematical formulas
- User needs reference implementations or benchmarks
- Part of a model validation workflow (invoked by orchestrator)
- User is preparing to implement a new quantitative model

## Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL RESEARCH WORKFLOW                       │
├─────────────────────────────────────────────────────────────────┤
│ Step 1: Gather Source Materials  → User docs + web search       │
│ Step 2: Cross-Check Formulas     → Verify against multiple src  │
│ Step 3: Find Implementations     → Reference code + benchmarks  │
│ Step 4: Identify Edge Cases      → Numerical considerations     │
│ Step 5: Generate Research Report → With confidence levels       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Gather Source Materials

### 1.1 User-Provided Materials

Collect any materials the user provides:
- Academic papers (PDF links, citations)
- Textbook references
- Internal documentation
- Existing code snippets

### 1.2 Web Search Strategy

**Search Queries by Model Type:**

| Model Category | Search Queries |
|---------------|----------------|
| Options Pricing | `"[model name]" pricing formula closed-form` |
| Interest Rates | `"[model name]" short rate model implementation` |
| Credit | `"[model name]" credit default swap pricing` |
| Volatility | `"[model name]" stochastic volatility calibration` |

**Authoritative Sources to Prioritize:**
1. Academic papers (arXiv, SSRN, published journals)
2. Standard textbooks (Hull, Wilmott, Glasserman, Brigo-Mercurio)
3. QuantLib documentation and source code
4. Official library docs (scipy, numpy)
5. Reputable quant finance blogs (e.g., Quantitative Research)

### 1.3 Source Categories

| Category | Trust Level | Usage |
|----------|-------------|-------|
| Peer-reviewed papers | HIGH | Primary source for formulas |
| Standard textbooks | HIGH | Secondary verification |
| QuantLib source | MEDIUM-HIGH | Implementation patterns |
| GitHub repos (well-starred) | MEDIUM | Reference implementations |
| Blog posts | LOW-MEDIUM | Intuition, explanations |
| Random code snippets | LOW | Avoid unless verified |

---

## Step 2: Cross-Check Mathematical Formulas

### 2.1 Formula Verification Matrix

For each key formula, verify across multiple sources:

| Formula | Source 1 | Source 2 | Source 3 | Consensus |
|---------|----------|----------|----------|-----------|
| Core pricing formula | | | | ✅/⚠️/❌ |
| Greeks formulas | | | | ✅/⚠️/❌ |
| Boundary conditions | | | | ✅/⚠️/❌ |
| Approximation terms | | | | ✅/⚠️/❌ |

### 2.2 Common Formula Discrepancies

**Watch for these variations:**
- Sign conventions (positive vs negative theta)
- Continuous vs discrete dividends
- Business days vs calendar days
- American vs European conventions
- Spot vs forward starting

**Document any discrepancies:**
```markdown
### Formula Discrepancy: [Formula Name]

**Source A (Hull, 9th ed, p.XXX)**:
$$ d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}} $$

**Source B (Paper XYZ)**:
$$ d_1 = \frac{\ln(F/K) + \sigma^2 T/2}{\sigma\sqrt{T}} $$

**Resolution**: Source B uses forward price $F = Se^{(r-q)T}$. Mathematically equivalent.
```

---

## Step 3: Find Reference Implementations

### 3.1 QuantLib Search

QuantLib is the gold standard for quantitative finance implementations.

**Search patterns:**
- GitHub: `site:github.com/lballabio/QuantLib "[model name]"`
- QuantLib docs: Search the class hierarchy

**Document findings:**
```markdown
### QuantLib Implementation

**Class**: `ql::AnalyticBarrierEngine`
**File**: `ql/pricingengines/barrier/analyticbarrierengine.cpp`
**Key Methods**:
- `calculate()`: Main pricing logic
- `helper()`: d1/d2 calculation helper

**Notable Implementation Details**:
- Uses Broadie-Glasserman discrete barrier adjustment
- Implements all barrier types (DI, DO, UI, UO)
```

### 3.2 Other Reference Sources

| Source | Pros | Cons |
|--------|------|------|
| scipy/numpy | Python, well-tested | May lack finance-specific |
| R packages (RQuantLib) | Direct QuantLib bindings | R syntax |
| Julia (QuantFinance.jl) | Modern, fast | Smaller community |
| MATLAB (Financial Toolbox) | Industry standard | Proprietary |

### 3.3 Benchmark Values

**Always try to find known benchmark values:**
- QuantLib test cases
- Textbook examples
- Published numerical results

```markdown
### Benchmark Values

**Source**: Hull, Table 15.2

| S | K | T | r | σ | q | Call Price |
|---|---|---|---|---|---|------------|
| 100 | 100 | 1 | 0.05 | 0.20 | 0 | 10.4506 |
| 100 | 110 | 0.5 | 0.05 | 0.30 | 0.02 | 6.7321 |
```

---

## Step 4: Identify Edge Cases and Numerical Considerations

### 4.1 Common Edge Cases

| Category | Edge Cases to Document |
|----------|----------------------|
| **Time** | Near expiry (T → 0), Very long dated |
| **Moneyness** | Deep ITM, Deep OTM, At-the-money |
| **Volatility** | Zero/near-zero vol, Very high vol |
| **Rates** | Zero rates, Negative rates |
| **Dividends** | Zero div, High div yield > r |
| **Barriers** | Spot near barrier, Spot = barrier |
| **Numerics** | log(0), exp(large), 0/0 |

### 4.2 Numerical Stability Analysis

**Identify potential numerical issues:**

```markdown
### Numerical Considerations

**Issue 1: Log of small numbers**
- When S << K (deep OTM call), log(S/K) can underflow
- Mitigation: Use log1p for small values or asymptotic approximation

**Issue 2: CDF tail precision**
- norm.cdf(d) loses precision for |d| > 8
- Mitigation: Use erfc/erfcx for tails

**Issue 3: Exponential overflow**
- exp(-rT) for large T and r
- Mitigation: Use log-space calculations
```

### 4.3 Convergence Properties (for numerical methods)

**For PDE/MC methods:**
- Grid/path convergence rates
- Stability conditions (CFL, etc.)
- Variance reduction techniques

---

## Step 5: Generate Research Report

### Report Template

```markdown
# Model Research Report / 模型研究报告

**Model**: [Model Name]
**Date**: <date>
**Confidence Level**: HIGH / MEDIUM / LOW

---

## 1. 摘要 / Executive Summary

Brief summary of research findings and recommendations.

---

## 2. 数学公式 / Mathematical Formulas

### 2.1 Core Pricing Formula

$$ V = ... $$

**Sources verified:**
- Hull (9th ed), Chapter X, p.XXX ✅
- Paper: [Author], [Year] ✅
- QuantLib implementation ✅

### 2.2 Greeks Formulas

| Greek | Formula | Verified |
|-------|---------|----------|
| Delta | $\partial V / \partial S$ | ✅/⚠️/❌ |
| Gamma | ... | ... |
| Vega | ... | ... |
| Theta | ... | ... |
| Rho | ... | ... |

### 2.3 Formula Discrepancies (if any)

[Document any discrepancies found between sources]

---

## 3. 参考实现 / Reference Implementations

### 3.1 QuantLib

- Class: `ql::XXXEngine`
- File: `path/to/file.cpp`
- Key implementation notes: ...

### 3.2 Other References

| Source | Language | Notes |
|--------|----------|-------|
| scipy.stats | Python | ... |
| ... | ... | ... |

---

## 4. 基准值 / Benchmark Values

| Parameters | Expected Value | Source |
|------------|----------------|--------|
| ... | ... | ... |

---

## 5. 边界情况 / Edge Cases

### 5.1 Extreme Market Conditions

| Condition | Expected Behavior | Notes |
|-----------|-------------------|-------|
| T → 0 | V → intrinsic | ... |
| σ → 0 | V → discounted intrinsic | ... |
| ... | ... | ... |

### 5.2 Numerical Considerations

1. **Issue**: [Description]
   - **Risk**: [What can go wrong]
   - **Mitigation**: [How to handle]

---

## 6. 实现建议 / Implementation Recommendations

1. Use `util.numerical.safe_log()` for log calculations
2. Handle T=0 case explicitly at start of `price()` method
3. Implement barrier adjustment for discrete monitoring
4. ...

---

## 7. 参考文献 / References

1. [Author, Year, Title, Publication]
2. ...

---

## Appendix: Confidence Assessment

| Aspect | Confidence | Notes |
|--------|------------|-------|
| Core formula | HIGH/MED/LOW | Verified in X sources |
| Greeks | HIGH/MED/LOW | ... |
| Edge cases | HIGH/MED/LOW | ... |
| Numerical stability | HIGH/MED/LOW | ... |
```

---

## Confidence Level Criteria

| Level | Criteria |
|-------|----------|
| HIGH | 3+ authoritative sources agree, QuantLib reference exists, benchmark values available |
| MEDIUM | 2 sources agree, some implementation reference, minor discrepancies resolved |
| LOW | Single source, no reference implementation, unresolved discrepancies |

---

## Integration with Model Validation Workflow

When invoked by model-orchestrator:

### Input
- Model name and type
- User-provided reference materials
- Specific focus areas (formulas, edge cases, benchmarks)

### Output
- `research-report.md` in designated output directory
- Confidence levels for each aspect
- Benchmark values for validation
- Edge case test scenarios

### Skip Conditions

Orchestrator can skip research phase if:
- User explicitly requests skip
- Model is well-known (e.g., Black-Scholes for European vanilla)
- Existing documentation is comprehensive

---

## Search Tools

This skill uses:
- **WebSearch**: For finding authoritative sources
- **WebFetch**: For retrieving specific pages
- **Read**: For reading user-provided local files

---

## Principles

1. **Verify, Don't Trust**: Always cross-check against multiple sources
2. **Document Discrepancies**: Note any formula variations between sources
3. **Prioritize Authoritative Sources**: Academic papers > textbooks > code > blogs
4. **Be Explicit About Confidence**: Use confidence levels honestly
5. **Focus on Implementation**: Research should enable correct implementation
