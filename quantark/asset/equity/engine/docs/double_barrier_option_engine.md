# Double Barrier Option Analytical Engine Reference

## 1. Model Overview

**Model Type**: Analytical
**Product Supported**: `DoubleBarrierOption`
**Primary Use Case**: Price European double-barrier options (knock-out and knock-in) with continuous, discrete, or expiry-only observation.

## 2. Mathematical Formulation

### 2.1 Core Formula — Ikeda & Kuintomo (1992)

The engine implements the infinite-series closed-form solution for double knock-out options published by Ikeda & Kuintomo (1992). Knock-in options are priced via parity:

```
Knock-In Price = Vanilla Price - Knock-Out Price
```

#### Call Up-and-Out-Down-and-Out

```
c = S * e^{(b-r)T} * Σ_{n=-∞}^{∞} { w1 * [N(d1) - N(d2)] - w2 * [N(d3) - N(d4)] }
  - X * e^{-rT} * Σ_{n=-∞}^{∞} { w1_strike * [N(d1-σ√T) - N(d2-σ√T)] - w2_strike * [N(d3-σ√T) - N(d4-σ√T)] }
```

where

- `d1 = (ln(S*U^{2n} / (X*L^{2n})) + (b + σ²/2)*T) / (σ√T)`
- `d2 = (ln(S*U^{2n} / (F*L^{2n})) + (b + σ²/2)*T) / (σ√T)`
- `d3 = (ln(L^{2n+2} / (X*S*U^{2n})) + (b + σ²/2)*T) / (σ√T)`
- `d4 = (ln(L^{2n+2} / (F*S*U^{2n})) + (b + σ²/2)*T) / (σ√T)`
- `F = U * e^{δ1*T}`
- `E = L * e^{δ2*T}`

Weights:

- `w1 = (U^n / L^n)^{μ1} * (L/S)^{μ2}`
- `w2 = (L^{n+1} / (U^n * S))^{μ3}`
- `w1_strike = (U^n / L^n)^{μ1-2} * (L/S)^{μ2}`
- `w2_strike = (L^{n+1} / (U^n * S))^{μ3-2}`

Mus:

- `μ1 = 2 * [b - δ2 - n*(δ1 - δ2)] / σ² + 1`
- `μ2 = 2*n*(δ1 - δ2) / σ²`
- `μ3 = 2 * [b - δ2 + n*(δ1 - δ2)] / σ² + 1`

#### Put Up-and-Out-Down-and-Out

The put formula replaces the call `d`'s with `y`'s:

- `y1` uses `E` instead of `X` in the numerator (outer term)
- `y2` uses `X`
- `y3` uses `E` in the inner term
- `y4` uses `X` in the inner term

and reverses the asset/strike summand order:

```
p = X * e^{-rT} * strike_sum - S * e^{(b-r)T} * asset_sum
```

### 2.2 Discrete Monitoring Adjustment

For discrete observation (`ObservationType.DISCRETE`), the engine applies the Broadie-Glasserman-Kou barrier shift before calling the continuous formula:

- Lower barrier shifted down: `L' = L * exp(-β * σ * sqrt(dt))`
- Upper barrier shifted up: `U' = U * exp(β * σ * sqrt(dt))`

where `β ≈ 0.5826` and `dt` is the observation interval (in years) inferred from the `ObservationSchedule`.

### 2.3 Expiry-Only Observation

For expiry observation, the price is the discounted truncated-domain vanilla payoff:

- **Call KO**: `Call(K) - Call(U) - (U-K) * e^{-rT} * N(d2(U)) + rebate * discount * P(outside)`
- **Put KO**: `Put(K) - Put(L) - (K-L) * e^{-rT} * N(-d2(L)) + rebate * discount * P(outside)`

Knock-in uses parity on the truncated domain.

## 3. Assumptions and Limitations

1. Black-Scholes world: constant volatility, constant risk-free rate, log-normal spot.
2. The Ikeda-Kuintomo formula requires the strike to be **strictly between** the lower and upper barriers (`L < K < U`).
3. Continuous observation assumes perfect monitoring; discrete approximates this via barrier shift.
4. Rebate is assumed paid at maturity if the option knocks out (or does not knock in).
5. The infinite series is truncated to a finite number of terms (`max_terms`). Numerical studies suggest 2-3 terms are usually sufficient; the engine uses a default of 10 for safety.

## 4. Numerical Considerations

### 4.1 Edge Cases

| Condition | Expected Behavior | Implementation |
|-----------|-------------------|----------------|
| `T = 0` | Intrinsic payoff or rebate | `is_zero(T)` check at top of `price()` |
| `σ = 0` | Deterministic path | `is_zero(sigma)` check; forward compared to barriers |
| Spot outside barriers (KO) | Rebate paid immediately | `product.is_barrier_hit(S)` check |
| Spot outside barriers (KI) | Vanilla price (already activated) | `product.is_barrier_hit(S)` check |
| Strike at barrier | Rejected | `_validate_inputs()` raises `ValidationError` |
| Curved barriers (`δ1, δ2 ≠ 0`) | Non-flat boundary terms | Supported via internal `delta1`/`delta2` kwargs |

### 4.2 Numerical Stability

- All exponentials, logarithms, square roots, and powers use `util.numerical` safe wrappers (`safe_exp`, `safe_log`, `safe_sqrt`, `safe_power`).
- Series weights (`w1`, `w2`, etc.) are checked with `math.isfinite()` before contributing to the sum. This prevents `inf * 0 = NaN` when large `|n|` causes overflow in `math.pow` while the corresponding CDF difference is near zero.
- Results are cast to `float()` after safe-math operations to satisfy type checking.

## 5. Validation Baselines

Continuous observation benchmark cases are validated against **Haug, Table 4-15 (Ikeda & Kuintomo 1992)**:

- `S = 100`, `X = 100`, `r = 0.1`, `b = 0.1` (so `q = 0`)
- Test matrix: `L ∈ {50,60,70,80,90}`, `U ∈ {150,140,130,120,110}`, `T ∈ {0.25, 0.5}`, `σ ∈ {0.15, 0.25}`
- Additional curvature cases: `δ1 = -0.1, δ2 = 0.1` and `δ1 = 0.1, δ2 = -0.1`
- Absolute tolerance: `1e-3`

## 6. References

1. Ikeda, M. and Kuintomo, N. (1992). "Pricing Options with Curved Boundaries." *Mathematical Finance*, 2(4), 275–298.
2. Haug, E. G. (2007). *The Complete Guide to Option Pricing Formulas* (2nd ed.). McGraw-Hill. Table 4-15.
3. Broadie, M., Glasserman, P., and Kou, S. G. (1997). "A Continuity Correction for Discrete Barrier Options." *Mathematical Finance*, 7(4), 325–349.
