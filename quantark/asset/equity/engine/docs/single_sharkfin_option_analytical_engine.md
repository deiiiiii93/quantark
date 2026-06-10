# Single Sharkfin Option Analytical Engine

## 1. Model Overview

**Model Type**: Analytical  
**Product Supported**: `SingleSharkfinOption`  
**Engine**: `SingleSharkfinOptionAnalyticalEngine`  
**Location**: `asset/equity/engine/analytical/single_sharkfin_option_analytical_engine.py`

The engine prices capped single-barrier sharkfin options under Black-Scholes assumptions. A call sharkfin is an upper knock-out structure; a put sharkfin is a lower knock-out structure.

## 2. Payoff Decomposition

For a call sharkfin with upper barrier `B` and strike `K`, the terminal payoff is:

```text
barrier hit:     knock_out_rebate
barrier not hit: no_hit_rebate + participation_rate * max(S_T - K, 0)
```

For a put sharkfin with lower barrier `B`:

```text
barrier hit:     knock_out_rebate
barrier not hit: no_hit_rebate + participation_rate * max(K - S_T, 0)
```

The analytical price is decomposed as:

```text
PV = participation_rate * KO_vanilla_no_rebate
   + PV(knock_out_rebate * 1_touch)
   + PV(no_hit_rebate * no_touch)
```

`KO_vanilla_no_rebate` is priced by `BarrierAnalyticalEngine` with zero rebate and unit contract multiplier. The no-hit fixed cash leg always settles at maturity. The knock-out rebate leg uses `SingleSharkfinOption.pay_at_hit`: when `True`, it is valued as paid immediately on touch; when `False`, it is paid at expiry.

## 3. Monitoring Modes

| Observation type | Treatment |
|------------------|-----------|
| `EXPIRY` | Closed-form expiry-only barrier/digital decomposition through the composed engines. |
| `CONTINUOUS` | Continuous single-barrier closed forms through the composed engines. |
| `DISCRETE` | Regular discrete schedules are approximated with BGK barrier shift. |

## 4. Discrete Barrier Shift

Daily and other regular discrete observation schedules use the Broadie-Glasserman-Kou continuity correction already implemented in `util.barrier_shift.apply_barrier_shift`:

```text
upper barrier: B_shifted = B * exp(beta * sigma * sqrt(dt))
lower barrier: B_shifted = B * exp(-beta * sigma * sqrt(dt))
beta ~= 0.5826
```

This shifts the barrier away from spot before applying the continuous closed-form formula. It is most reliable for regular schedules and can be inaccurate when spot is very close to the barrier.

## 5. Numerical Considerations

| Condition | Expected Behavior |
|-----------|-------------------|
| `T -> 0` | Return product terminal payoff at current spot. |
| Spot already beyond a continuous/discrete barrier | One-touch leg pays the knock-out rebate immediately if `pay_at_hit=True`, otherwise discounted to expiry. No-touch and KO vanilla legs are zero. |
| Zero rebates | Corresponding touch/no-touch leg is skipped. |
| Contract multiplier | Applied once at the sharkfin engine level. |

## 6. References

1. Reiner, E. and Rubinstein, M. (1991), "Breaking Down the Barriers", Risk.
2. Broadie, M., Glasserman, P. and Kou, S. (1997), "A Continuity Correction for Discrete Barrier Options", Mathematical Finance, 7(4), 325-348.
3. QuantArk `BarrierAnalyticalEngine` and `OneTouchAnalyticalEngine` reference implementations.
