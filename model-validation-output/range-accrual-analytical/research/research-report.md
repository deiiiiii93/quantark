# Research Report: Range Accrual Analytical Engine (Digital Decomposition)

## Key Finding

Under GBM, a Range Accrual's expected payoff decomposes exactly into a sum of digital option probabilities via linearity of expectation. Each observation contributes w_i * P(L_i <= S(t_i) <= U_i) independently.

## Core Formula

```
Price = exp(-r*T) * S_0 * M * c * tau * (1/W) * [past_in_range + sum_i w_i * P_i]

P_i = N(d2_L) - N(d2_U)   (standard mode)
P_i = 1 - [N(d2_L) - N(d2_U)]  (reverse mode)

d2(K, t_i) = [ln(S/K) + (r - q - sigma_i^2/2) * t_i] / (sigma_i * sqrt(t_i))
```

## Benchmark (Case 1): S=100, [90,110], sigma=0.2, 12 monthly, rate=5% annualized
- E[ratio] = 0.5548, Price = 2.6386

## Edge Cases
- Near-expiry (t<1e-10): deterministic check L<=S<=U
- Low vol (<1e-3): use forward S*exp((r-q)*t) deterministic check
- All-past: direct computation, no stochastic component
