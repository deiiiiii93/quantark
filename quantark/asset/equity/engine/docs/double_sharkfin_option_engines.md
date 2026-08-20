# Double Sharkfin Option Engines

## 1. Model Overview

**Model Type**: Analytical and Monte Carlo  
**Product Supported**: `DoubleSharkfinOption`  
**Analytical Engine**: `DoubleSharkfinOptionAnalyticalEngine`  
**Monte Carlo Engine**: `DoubleSharkfinOptionMCEngine`

Double sharkfin options are double knock-out structures with a capped
vanilla-style call or put participation leg inside a lower/upper barrier
corridor. If either barrier is hit, the product pays a fixed knock-out rebate.
If neither barrier is hit, it pays a fixed no-hit rebate plus the capped
participation payoff.

## 2. Payoff Decomposition

For lower barrier `L`, upper barrier `U`, strike `K`, and terminal spot `S_T`:

```text
barrier hit:     knock_out_rebate
barrier not hit: no_hit_rebate + participation_rate * payoff_inside
```

For a call:

```text
payoff_inside = max(min(S_T, U) - K, 0)
```

For a put:

```text
payoff_inside = max(K - max(S_T, L), 0)
```

The analytical price is decomposed as:

```text
PV = participation_rate * DoubleBarrier_KO_Vanilla
   + PV(knock_out_rebate * double_touch)
   + PV(no_hit_rebate * double_no_touch)
```

The double knock-out vanilla leg is priced with
`DoubleBarrierOptionAnalyticalEngine` using zero rebate and unit contract
multiplier. Cash legs are priced from double-barrier survival probabilities.

## 3. Monitoring Modes

| Observation type | Analytical treatment | MC treatment |
|------------------|----------------------|--------------|
| `EXPIRY` | Truncated lognormal payoff and terminal inside/outside probabilities. | One terminal simulation step. |
| `CONTINUOUS` | Double-barrier closed form for the option leg and killed log-price density series for no-touch survival. | Simulated path grid, optionally with Brownian-bridge step crossing probabilities. |
| `DISCRETE` | BGK shifted barriers with regular schedule validation. | Observation-schedule-aligned path grid. |

## 4. Survival Probability

For continuous monitoring, the no-touch probability is computed from the
absorbing transition density of log spot inside `[log L, log U]`. If
`X_t = log S_t` and `mu = r - q - 0.5 sigma^2`, the survival probability is
evaluated as a sine-series integral over the killed density:

```text
P(tau > T) = integral_logL^logU p_abs(y, T | x0) dy
```

The pay-at-hit rebate factor uses the survival curve identity:

```text
E[e^{-r tau} 1_tau<=T]
  = 1 - e^{-rT} P(tau > T) - r * integral_0^T e^{-rt} P(tau > t) dt
```

The time integral is evaluated with fixed Gauss-Legendre quadrature.

## 5. Numerical Considerations

| Condition | Expected behavior |
|-----------|-------------------|
| `T -> 0` | Return product payoff at current spot. |
| Spot already outside barriers under continuous/discrete monitoring | Pay knock-out rebate immediately or discounted to expiry according to `pay_at_hit`. |
| Discrete schedule | Require regular observation spacing for analytical BGK shift. |
| Contract multiplier | Applied once at the top-level engine output. |
| MC standard error | Available through `get_last_std_error()`. |

## 6. Assumptions and Limitations

1. Analytical pricing assumes Black-Scholes dynamics with flat values supplied
   through `PricingEnvironment`.
2. Discrete analytical monitoring is an approximation using BGK barrier shifts.
3. Continuous MC monitoring is grid-based unless Brownian bridge is enabled.
4. Analytical pay-at-hit cash legs use a survival-curve quadrature rather than
   a separate closed-form first-exit density expression.

## 7. References

1. Ikeda, M. and Kunitomo, N. (1992), "Pricing Options with Curved Boundaries",
   Mathematical Finance.
2. Broadie, M., Glasserman, P. and Kou, S. (1997), "A Continuity Correction for
   Discrete Barrier Options", Mathematical Finance, 7(4), 325-348.
3. QuantArk `DoubleBarrierOptionAnalyticalEngine`,
   `SingleSharkfinOptionAnalyticalEngine`, and `SingleSharkfinOptionMCEngine`
   reference implementations.
