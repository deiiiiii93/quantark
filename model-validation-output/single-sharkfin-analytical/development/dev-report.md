# Model Development Report / 模型开发报告

**Model**: Single Sharkfin Analytical Engine  
**Developer**: Developer A (Codex)  
**Status**: COMPLETE

---

## 1. 实现摘要 / Implementation Summary

Implemented `SingleSharkfinOptionAnalyticalEngine` as a closed-form composition:

```text
PV = participation_rate * no_rebate_knock_out_option
   + knock_out_rebate_touch_leg
   + no_hit_rebate_no_touch_leg
```

The engine reuses the existing `BarrierAnalyticalEngine` for the capped knock-out vanilla payoff and `OneTouchAnalyticalEngine` for the knock-out and no-hit fixed cash legs.

## 2. 支持范围 / Scope

| Feature | Status |
|---------|--------|
| Call sharkfin, upper knock-out barrier | Supported |
| Put sharkfin, lower knock-out barrier | Supported |
| Expiry monitoring | Supported |
| Continuous monitoring | Supported |
| Regular discrete monitoring | Supported via BGK barrier shift |
| Daily observation variant | Supported via regular schedule and barrier shift |
| Knock-out rebate paid at expiry | Supported |
| Knock-out rebate paid at hit | Supported with `pay_at_hit=True` |

## 3. 文件 / Files

| File | Description |
|------|-------------|
| `asset/equity/engine/analytical/single_sharkfin_option_analytical_engine.py` | Main analytical engine |
| `asset/equity/engine/docs/single_sharkfin_option_analytical_engine.md` | Reference documentation |
| `test/test_single_sharkfin_analytical_engine.py` | Engine tests |
| `asset/equity/product/option/single_sharkfin_option.py` | Added `pay_at_hit` product field |

## 4. 边界情况 / Edge Cases

| Condition | Handling |
|-----------|----------|
| `T -> 0` | Return product terminal payoff |
| Already hit continuous/discrete barrier | Touch leg handles immediate vs expiry rebate timing |
| Zero rebate | Skip corresponding fixed cash leg |
| Discrete daily monitoring | Barrier shifted using BGK correction |

