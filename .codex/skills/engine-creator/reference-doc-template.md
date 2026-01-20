# Reference Documentation Template

Template for creating engine reference documentation in `docs/` folders.

## File Naming

**Pattern**: `{script_name}.md` (matching the engine script name)

Examples:
- `american_option_analytical_engine.py` → `american_option_analytical_engine.md`
- `barrier_pde_solver.py` → `barrier_pde_solver.md`
- `snowball_mc_engine.py` → `snowball_mc_engine.md`

## Template Structure

```markdown
# [Product Name] [Engine Type] Engine

## Overview

Brief description of what this engine does and when to use it.

## Supported Products

- `ProductClass1`: Description
- `ProductClass2`: Description

## Methods / Algorithms

### Method 1: [Name]

**Description**: One-line description.

**Formula**:
$$
V = f(S, K, T, r, \sigma)
$$

**Parameters**:
| Parameter | Description | Range |
|-----------|-------------|-------|
| S | Spot price | > 0 |
| K | Strike price | > 0 |
| T | Time to maturity | > 0 |
| r | Risk-free rate | any |
| σ | Volatility | > 0 |

**Assumptions**:
- Assumption 1
- Assumption 2

**Accuracy**: O(ε) where ε is [description]

### Method 2: [Name]

[Same structure as Method 1]

## Numerical Considerations

### Edge Cases

| Condition | Handling |
|-----------|----------|
| T → 0 | Return intrinsic value |
| σ → 0 | Return discounted forward payoff |
| S >> K | Asymptotic approximation |
| S << K | Near-zero approximation |

### Stability

- Potential overflow/underflow points
- Recommended parameter ranges
- Known numerical issues

## Greeks

### Analytical Greeks (if available)

**Delta**:
$$
\Delta = \frac{\partial V}{\partial S}
$$

**Gamma**:
$$
\Gamma = \frac{\partial^2 V}{\partial S^2}
$$

[Continue for vega, theta, rho...]

### Numerical Greeks

- Method: Central difference
- Default bump size: 0.01%
- Accuracy: O(h²)

## Usage Example

```python
from asset.equity.engine.analytical import MyEngine
from asset.equity.product.option import MyProduct
from priceenv import PricingEnvironment

# Create product
product = MyProduct(strike=100.0, maturity=1.0, ...)

# Create pricing environment
pricing_env = PricingEnvironment(
    spot_quote=SpotQuote(spot=100.0),
    vol_surface=FlatVolSurface(volatility=0.20),
    rate_curve=FlatRateCurve(rate=0.05),
    valuation_date=datetime(2024, 1, 1),
)

# Create engine
engine = MyEngine(method="METHOD_A")

# Price
price = engine.price(product, pricing_env)
greeks = engine.calculate_greeks(product, pricing_env)
```

## Performance

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| price() | O(1) | < 1ms |
| calculate_greeks() | O(n) | < 5ms |

## Validation

### Test Cases

| Test | Input | Expected | Tolerance |
|------|-------|----------|-----------|
| ATM call | S=K=100, T=1, σ=0.2 | 10.45 | 0.01 |
| Deep ITM | S=150, K=100 | ~52.0 | 0.1 |
| Near expiry | T=1/365 | ~intrinsic | 0.001 |

### Comparison with Other Methods

- vs Black-Scholes: Max error < 0.01%
- vs Monte Carlo (10M paths): Max error < 0.1%

## References

1. Author, A. (Year). "Paper Title". Journal, Vol(Issue), pp-pp.
2. Book Author. (Year). *Book Title*. Publisher.
3. URL reference if applicable

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-01-01 | Initial implementation |
| 1.1.0 | 2024-06-01 | Added Method B |
```

## Example: American Option Analytical Engine Documentation

```markdown
# American Option Analytical Engine

## Overview

Analytical approximation engine for American vanilla options. Provides fast
pricing without numerical PDE/MC methods, suitable for real-time applications.

## Supported Products

- `AmericanOption`: American-style call and put options

## Methods / Algorithms

### Method 1: BS93 (Bjerksund-Stensland 1993)

**Description**: Single-barrier approximation using flat boundary.

**Formula**:
For American call:
$$
C_{Am} = \alpha_2 S^\beta - \alpha_2 \phi(S, T, \beta, I, I)
        + \phi(S, T, 1, I, I) - \phi(S, T, 1, K, I)
        - K \phi(S, T, 0, I, I) + K \phi(S, T, 0, K, I)
$$

where:
- $\alpha_2 = (I - K) I^{-\beta}$
- $\beta = \frac{1}{2} - \frac{b}{\sigma^2} + \sqrt{(\frac{b}{\sigma^2} - \frac{1}{2})^2 + \frac{2r}{\sigma^2}}$
- $I$ is the critical price (exercise boundary)
- $\phi$ is the bivariate normal distribution function

**Accuracy**: Typically < 0.1% error vs PDE benchmark

### Method 2: BS02 (Bjerksund-Stensland 2002)

**Description**: Two-barrier approximation with better accuracy for dividend-paying stocks.

**Accuracy**: Typically < 0.05% error vs PDE benchmark

### Method 3: BAW (Barone-Adesi-Whaley)

**Description**: Quadratic approximation with iterative critical price calculation.

**Formula**:
$$
C_{Am} = C_{Eu} + A_2 (S/S^*)^{q_2}
$$

**Accuracy**: Typically < 0.2% error, worse near dividend dates

## Numerical Considerations

### Edge Cases

| Condition | Handling |
|-----------|----------|
| T → 0 | Return max(intrinsic, 0) |
| σ → 0 | Return discounted intrinsic |
| q > r | European == American for calls |
| Deep ITM put | Early exercise likely |

### Critical Price Iteration (BAW)

- Newton-Raphson iteration
- Max iterations: 100
- Convergence tolerance: 1e-8
- Fallback: bisection if Newton fails

## References

1. Bjerksund, P. & Stensland, G. (1993). "Closed-Form Approximation of
   American Options". Scandinavian Journal of Management, 9, S87-S99.

2. Bjerksund, P. & Stensland, G. (2002). "Closed Form Valuation of American
   Options". Discussion paper 2002/09, Norwegian School of Economics.

3. Barone-Adesi, G. & Whaley, R.E. (1987). "Efficient Analytic Approximation
   of American Option Values". Journal of Finance, 42(2), 301-320.
```

## Web Search Queries for Missing References

When no documentation exists, use these search patterns:

**Analytical formulas:**
- `"[product] closed-form solution" pricing`
- `"[product] analytical approximation" formula`
- `"[author name]" "[year]" [product] option`

**PDE methods:**
- `"[product] finite difference" pricing`
- `"[product] Crank-Nicolson" boundary conditions`
- `"barrier option PDE" grid concentration`

**Monte Carlo:**
- `"[product] Monte Carlo" variance reduction`
- `"[product] path-dependent" simulation`
- `"quasi Monte Carlo" Sobol [product]`

**Academic sources:**
- `site:ssrn.com "[product] pricing"`
- `site:arxiv.org "[product] option"`
- `"Hull" "[product]" formula` (for standard reference)
- `"Wilmott" "[product]"` (for numerical methods)

## Documentation Checklist

- [ ] Overview explains purpose and use case
- [ ] All supported products listed
- [ ] Each method has formula and description
- [ ] Parameters documented with valid ranges
- [ ] Edge cases and their handling documented
- [ ] Numerical stability notes included
- [ ] Greeks formulas (if analytical)
- [ ] Working usage example
- [ ] Performance characteristics noted
- [ ] Test cases with expected values
- [ ] Academic references cited
- [ ] Changelog maintained
