# Model Development Report / 模型开发报告

**Model**: Single Sharkfin Monte Carlo Engine  
**Developer**: Developer A (Codex)  
**Status**: COMPLETE

---

## 1. 实现摘要 / Implementation Summary

Implemented `SingleSharkfinOptionMCEngine` for path-based pricing under the existing QuantArk GBM Monte Carlo infrastructure.

The engine supports:

- `PSEUDO`, `QUASI`, and `RANDOMIZED_QUASI` methods,
- expiry-only, discrete, and continuous monitoring,
- optional Brownian-bridge crossing probabilities for continuous monitoring,
- `pay_at_hit` and pay-at-expiry knock-out rebate timing,
- no-hit rebate paid at expiry,
- contract multiplier and standard-error scaling.

## 2. 核心逻辑 / Core Logic

For each path:

```text
if barrier hit:
    payoff = knock_out_rebate
else:
    payoff = no_hit_rebate + participation_rate * capped_vanilla_payoff
```

Discounting:

- no-hit payoff is discounted to expiry,
- knock-out payoff is discounted to the hit observation time when `pay_at_hit=True`,
- otherwise knock-out payoff is discounted to expiry.

## 3. 文件 / Files

| File | Description |
|------|-------------|
| `asset/equity/engine/mc/single_sharkfin_option_mc_engine.py` | Main MC engine |
| `asset/equity/engine/docs/single_sharkfin_option_mc_engine.md` | Reference documentation |
| `test/test_single_sharkfin_mc_engine.py` | MC engine tests |
| `asset/equity/engine/mc/__init__.py` | Engine export |

## 4. 边界情况 / Edge Cases

| Condition | Handling |
|-----------|----------|
| `T = 0` | Return product payoff at current spot |
| Spot already hit continuous/discrete barrier | Return knock-out rebate using requested timing |
| Zero rebates | Payoff naturally reduces to remaining components |
| Contract multiplier | Applied once to price and standard error |

