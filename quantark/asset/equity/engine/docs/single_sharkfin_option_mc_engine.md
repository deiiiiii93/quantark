# Single Sharkfin Option Monte Carlo Engine

## 1. Model Overview

**Model Type**: Monte Carlo  
**Product Supported**: `SingleSharkfinOption`  
**Engine**: `SingleSharkfinOptionMCEngine`  
**Location**: `asset/equity/engine/mc/single_sharkfin_option_mc_engine.py`

The engine simulates geometric Brownian motion under the risk-neutral measure and evaluates the single sharkfin payoff path by path.

## 2. Stochastic Process

The underlying follows Black-Scholes GBM:

```text
dS_t = (r - q) S_t dt + sigma S_t dW_t
```

The path generator uses the existing QuantArk BSM/QMC infrastructure.

## 3. Payoff Logic

For each path:

```text
if barrier hit:
    payoff = knock_out_rebate
else:
    payoff = no_hit_rebate + participation_rate * capped_vanilla_payoff
```

Call sharkfin:

```text
capped_vanilla_payoff = max(min(S_T, B) - K, 0)
```

Put sharkfin:

```text
capped_vanilla_payoff = max(K - max(S_T, B), 0)
```

## 4. Observation Modes

| Observation type | MC treatment |
|------------------|--------------|
| `EXPIRY` | Simulate terminal spot only and test terminal barrier hit. |
| `DISCRETE` | Simulate the observation grid plus maturity; use first observed hit. Custom schedules must use `STOP_FIRST_HIT` aggregation. |
| `CONTINUOUS` | Simulate a uniform grid; optional Brownian bridge crossing probabilities. |

## 5. Rebate Timing

`pay_at_hit=False` discounts knock-out rebate to expiry.  
`pay_at_hit=True` discounts knock-out rebate to the first hit observation time. With Brownian bridge enabled, the hit leg uses stepwise first-hit probabilities.

The no-hit rebate always pays at expiry.

## 6. Scaling and Outputs

The engine returns prices scaled by `product.contract_multiplier`. The last standard error is available through `get_last_std_error()` and is scaled by the same multiplier.
