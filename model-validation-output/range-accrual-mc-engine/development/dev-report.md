# Development Report: Range Accrual MC Engine
# 开发报告: Range Accrual 蒙特卡洛引擎

**Developer**: Claude (Developer A)
**Date**: 2026-02-05
**Status**: COMPLETED

---

## 1. 实现概述 / Implementation Overview

### Files Created / 创建的文件

| File | Description |
|------|-------------|
| `asset/equity/engine/mc/range_accrual_mc_engine.py` | Main MC engine implementation |
| `test/test_range_accrual_mc_engine.py` | Comprehensive test suite (17 tests) |

### Files Modified / 修改的文件

| File | Change |
|------|--------|
| `asset/equity/engine/mc/__init__.py` | Added exports for RangeAccrualMCEngine and RangeAccrualMCResult |

---

## 2. 设计决策 / Design Decisions

### 2.1 Engine Architecture / 引擎架构

The engine follows the established QuantArk MC engine pattern:

```python
class RangeAccrualMCEngine(BaseEngine):
    engine_type = EngineType.MONTE_CARLO
    DEFAULT_METHOD = MonteCarloMethod.PSEUDO
```

Key design choices:
1. **Inherits from BaseEngine** - Provides standard `price()` and `calculate_greeks()` interface
2. **Three MC Methods** - PSEUDO, QUASI (Sobol), RANDOMIZED_QUASI
3. **Vectorized NumPy operations** - Efficient batch processing of paths
4. **GBMPathGenerator integration** - Uses existing QMC infrastructure

### 2.2 Observation Grid Construction / 观测网格构建

The engine separates past and future observations using the product's `resolve_observations()` method:

```
Past Observations (t ≤ 0):
  - Use recorded observed_in_range values
  - Accumulate in_range weights

Future Observations (t > 0):
  - Build simulation grid
  - Check barriers at each observation time
```

### 2.3 Payoff Calculation / 收益计算

```
Payoff = initial_price × contract_multiplier × accrual_rate
         × (in_range_weights / total_weights) × year_fraction
```

The engine computes:
1. Past in-range weights from historical observations
2. Future in-range weights via MC simulation
3. Total weights sum across all observations
4. Final ratio and discounted payoff

---

## 3. 关键实现细节 / Key Implementation Details

### 3.1 In-Range Checking

```python
def _check_in_range(self, product, spots, future_obs):
    for i, (_, _, obs_idx) in enumerate(future_obs):
        upper = product.range_config.get_upper_barrier(obs_idx)
        lower = product.range_config.get_lower_barrier(obs_idx)
        in_range_col = (spots[:, i] >= lower) & (spots[:, i] <= upper)
        if product.range_config.is_reverse:
            in_range_col = ~in_range_col
        in_range[:, i] = in_range_col
    return in_range
```

### 3.2 Weighted Accrual

Weights are accumulated based on observation outcomes:
- In standard mode: weight added if `lower ≤ spot ≤ upper`
- In reverse mode: weight added if `spot < lower OR spot > upper`

### 3.3 Historical Observations

Past observations with `observed_in_range` set are handled separately:
- No simulation needed for past observations
- Weights accumulated from recorded outcomes
- Combined with simulated future observations

---

## 4. 测试覆盖 / Test Coverage

### 4.1 Test Categories

| Category | Tests | Status |
|----------|-------|--------|
| Basic Pricing | 6 | ✓ PASS |
| Range Effects | 2 | ✓ PASS |
| Reverse Mode | 1 | ✓ PASS |
| Historical Observations | 2 | ✓ PASS |
| Edge Cases | 4 | ✓ PASS |
| Convergence | 1 | ✓ PASS |
| Repr | 1 | ✓ PASS |
| **TOTAL** | **17** | **✓ ALL PASS** |

### 4.2 Key Test Results

```
Test: Reverse mode ratios sum to 1.0
  Normal in-range ratio: ~0.5059
  Reverse out-of-range ratio: ~0.4941
  Sum: 1.0000 ✓

Test: Wider range → higher in-range ratio
  Standard (90-110): ~0.5059
  Wide (70-130): ~0.9314 ✓

Test: Narrower range → lower in-range ratio
  Standard (90-110): ~0.5059
  Narrow (95-105): ~0.2717 ✓
```

---

## 5. 性能特征 / Performance Characteristics

### 5.1 Computational Complexity

- **Time**: O(num_paths × num_observations)
- **Memory**: O(num_paths × num_times) for path storage

### 5.2 Benchmark Results

| Configuration | Time (50K paths) | Std Error |
|---------------|------------------|-----------|
| 4 observations, QUASI | ~0.3s | ~69 |
| 4 observations, PSEUDO | ~0.3s | ~69 |

---

## 6. API Reference / API 参考

### 6.1 Engine Initialization

```python
engine = RangeAccrualMCEngine(
    params=MCParams(num_paths=100000, seed=42),
    method=MonteCarloMethod.QUASI,  # or 'quasi', or EngineType.MONTE_CARLO(MonteCarloMethod.QUASI)
)
```

### 6.2 Pricing

```python
price = engine.price(option, pricing_env)
result = engine.get_last_result()

# Access statistics
print(f"Price: {price}")
print(f"Std Error: {result.std_error}")
print(f"In-Range Ratio: {result.in_range_ratio_mean}")
print(f"Past Observations: {result.num_past_observations}")
print(f"Future Observations: {result.num_future_observations}")
```

### 6.3 Result Container

```python
@dataclass
class RangeAccrualMCResult:
    price: float
    std_error: float
    num_paths: int
    in_range_ratio_mean: float
    in_range_ratio_std: float
    num_past_observations: int = 0
    num_future_observations: int = 0
    past_in_range_weights: float = 0.0
    total_weights: float = 0.0
    batches_used: Optional[int] = None  # For RQMC
```

---

## 7. 与现有代码的一致性 / Consistency with Existing Code

The implementation follows patterns from:
- `AsianOptionMCEngine` - Observation-based MC structure
- `SnowballMCEngine` - Path generation and discounting
- `PhoenixMCEngine` - Method selection and validation

Key patterns followed:
- Two-level enum pattern for method selection
- MCParams for configuration
- GBMPathGenerator for path generation
- Vectorized NumPy operations
- Result dataclass pattern

---

## 8. 已知限制 / Known Limitations

1. **No Dask parallelization** - Unlike Snowball/Phoenix engines, Dask parallel batching is not implemented (can be added if needed)
2. **Continuous barrier monitoring** - Not supported (only discrete observations)
3. **Brownian bridge correction** - Not used for barrier checking (observations are discrete, not continuous barriers)

---

## 9. 下一步 / Next Steps

1. **Developer B Validation** - Independent re-implementation for gate check
2. **Code Reviews** - Performance, security, and code quality reviews
3. **Documentation Update** - Update CLAUDE.md with Range Accrual MC engine reference

---

## 10. 附录: 代码清单 / Appendix: Code Listing

See `asset/equity/engine/mc/range_accrual_mc_engine.py` for full implementation.
