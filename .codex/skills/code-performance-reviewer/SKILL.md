---
name: code-performance-reviewer
description: |
  Review code for performance optimization opportunities with focus on computational efficiency.
  Especially relevant for single-trade pricing and Greeks calculation speed.
  Use when the user asks to:
  - Review code for performance issues
  - Optimize computation speed
  - Profile code for bottlenecks
  - Improve vectorization or memory efficiency
  - Benchmark against existing implementations
  Triggers: "performance review", "optimize code", "speed up", "profiling", "benchmark performance", "vectorize"
---

# Code Performance Reviewer Skill

Review code for performance optimization opportunities with focus on computational efficiency, memory usage, and algorithmic improvements.

## When This Skill Activates

Codex should use this skill when:
- User asks to review code for performance
- User wants to optimize computation speed
- User needs profiling or benchmarking
- Part of a model validation workflow (invoked by orchestrator)
- User mentions "slow", "optimize", "speed", "vectorize", "memory"

## Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    PERFORMANCE REVIEW WORKFLOW                   │
├─────────────────────────────────────────────────────────────────┤
│ Step 1: Profile Code            → Identify hot spots            │
│ Step 2: Algorithm Analysis      → Complexity, patterns          │
│ Step 3: Memory Analysis         → Allocations, copies           │
│ Step 4: Vectorization Check     → NumPy/array operations        │
│ Step 5: Benchmark Comparison    → vs existing implementations   │
│ Step 6: Generate Report         → With specific recommendations │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Profile Code

### 1.1 Quick Profiling (Python)

```python
import cProfile
import pstats
import io

def profile_function(func, *args, **kwargs):
    """Profile a function and return stats."""
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    return result, s.getvalue()
```

### 1.2 Line-Level Profiling

```python
# Using line_profiler (if available)
# @profile decorator on functions of interest

# Or manual timing
import time

def timed_section(name):
    """Context manager for timing code sections."""
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            elapsed = time.perf_counter() - self.start
            print(f"{name}: {elapsed*1000:.3f} ms")
    return Timer()
```

### 1.3 Key Metrics to Collect

| Metric | Tool | Target |
|--------|------|--------|
| Single price time | timeit | < 1ms for analytical |
| Full Greeks time | timeit | < 5ms for analytical |
| Memory per call | tracemalloc | < 1MB typically |
| Function call count | cProfile | Minimize redundant calls |

---

## Step 2: Algorithm Analysis

### 2.1 Complexity Assessment

| Pattern | Complexity | Notes |
|---------|------------|-------|
| Single loop over N items | O(N) | Usually acceptable |
| Nested loops | O(N²) | Often optimizable |
| Matrix operations | O(N³) or O(N²) | Use optimized BLAS |
| Tree traversal | O(2^N) | May need memoization |

### 2.2 Common Performance Anti-Patterns

#### Python-Specific
```python
# ❌ BAD: Growing list in loop
result = []
for i in range(n):
    result.append(calculate(i))  # O(N) amortized, but allocations

# ✅ GOOD: List comprehension or pre-allocation
result = [calculate(i) for i in range(n)]
# or
result = np.zeros(n)
for i in range(n):
    result[i] = calculate(i)
```

```python
# ❌ BAD: Redundant calculations
for i in range(n):
    for j in range(m):
        value = expensive_function(i) * another_function(j)

# ✅ GOOD: Cache intermediate results
cached_i = [expensive_function(i) for i in range(n)]
for i in range(n):
    for j in range(m):
        value = cached_i[i] * another_function(j)
```

```python
# ❌ BAD: String concatenation in loop
result = ""
for item in items:
    result += str(item)

# ✅ GOOD: Join list
result = "".join(str(item) for item in items)
```

### 2.3 Quant Finance Specific Patterns

```python
# ❌ BAD: Repeated option pricing with same market data
for strike in strikes:
    product = Option(strike=strike, ...)
    prices.append(engine.price(product, env))  # env unchanged

# ✅ GOOD: Vectorized strike pricing (if engine supports)
prices = engine.price_strikes(strikes, base_product, env)

# ❌ BAD: Recalculating Greeks separately
delta = calc_delta(product, env)
gamma = calc_gamma(product, env)  # Repeats price calculation
vega = calc_vega(product, env)    # Repeats again

# ✅ GOOD: Calculate all Greeks together
greeks = calc_all_greeks(product, env)  # Reuses intermediate values
```

---

## Step 3: Memory Analysis

### 3.1 Memory Profiling

```python
import tracemalloc

tracemalloc.start()
# ... code to profile ...
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024**2:.2f} MB")
print(f"Peak: {peak / 1024**2:.2f} MB")
tracemalloc.stop()
```

### 3.2 Common Memory Issues

| Issue | Detection | Solution |
|-------|-----------|----------|
| Large array copies | Check for `.copy()`, slicing | Use views where possible |
| Intermediate arrays | Multiple array operations | Use `out=` parameter |
| Memory leaks | Growing memory over iterations | Clear caches, use generators |
| Object accumulation | List of objects in loop | Process in batches |

### 3.3 NumPy Memory Patterns

```python
# ❌ BAD: Creates intermediate arrays
result = np.exp(-r * T) * np.maximum(S - K, 0)  # 3 arrays

# ✅ GOOD: In-place operations (when possible)
result = S - K
np.maximum(result, 0, out=result)
result *= np.exp(-r * T)

# ✅ ALSO GOOD: Let NumPy optimize (modern versions are smart)
# For small arrays, readability > micro-optimization
```

---

## Step 4: Vectorization Check

### 4.1 Vectorization Opportunities

**Signs of non-vectorized code:**
- Python `for` loops over array elements
- `map()` or list comprehensions over arrays
- Element-wise `if` statements

**Vectorization pattern:**
```python
# ❌ BAD: Python loop
payoffs = []
for s in final_prices:
    if option_type == 'call':
        payoffs.append(max(s - strike, 0))
    else:
        payoffs.append(max(strike - s, 0))

# ✅ GOOD: Vectorized
if option_type == 'call':
    payoffs = np.maximum(final_prices - strike, 0)
else:
    payoffs = np.maximum(strike - final_prices, 0)
```

### 4.2 Vectorization Checklist

- [ ] All loops over array elements converted to NumPy operations
- [ ] Conditional logic uses `np.where()` or boolean indexing
- [ ] Mathematical functions use NumPy versions (`np.exp`, `np.log`, etc.)
- [ ] No Python `math` module calls on arrays
- [ ] Broadcasting used instead of explicit loops

### 4.3 Monte Carlo Specific

```python
# ❌ BAD: Loop over paths
prices = []
for i in range(n_paths):
    path = generate_path(...)
    prices.append(calculate_payoff(path))
mean_price = np.mean(prices)

# ✅ GOOD: Generate all paths at once
paths = generate_all_paths(n_paths, ...)  # Shape: (n_paths, n_steps)
payoffs = calculate_payoffs_vectorized(paths)  # Vectorized
mean_price = np.mean(payoffs)
```

---

## Step 5: Benchmark Comparison

### 5.1 Benchmark Framework

```python
import timeit

def benchmark_engine(engine, product, env, n_runs=1000):
    """Benchmark engine pricing speed."""

    # Warmup
    for _ in range(10):
        engine.price(product, env)

    # Benchmark price only
    price_time = timeit.timeit(
        lambda: engine.price(product, env),
        number=n_runs
    ) / n_runs * 1000  # ms

    # Benchmark full Greeks
    greeks_time = timeit.timeit(
        lambda: engine.calculate_greeks(product, env),
        number=n_runs
    ) / n_runs * 1000  # ms

    return {
        'price_ms': price_time,
        'greeks_ms': greeks_time,
        'price_per_sec': 1000 / price_time,
        'greeks_per_sec': 1000 / greeks_time,
    }
```

### 5.2 Performance Targets by Engine Type

| Engine Type | Price Target | Greeks Target | Notes |
|-------------|--------------|---------------|-------|
| Analytical | < 0.1 ms | < 0.5 ms | Should be very fast |
| Quadrature | < 10 ms | < 50 ms | Grid-based |
| PDE | < 100 ms | < 200 ms | Grid-based |
| Monte Carlo (10k paths) | < 100 ms | < 500 ms | Path-based |
| Monte Carlo (100k paths) | < 1 s | < 5 s | Path-based |

### 5.3 Comparison with Reference

```python
# Compare with existing implementation
def compare_performance(new_engine, reference_engine, product, env):
    new_stats = benchmark_engine(new_engine, product, env)
    ref_stats = benchmark_engine(reference_engine, product, env)

    speedup_price = ref_stats['price_ms'] / new_stats['price_ms']
    speedup_greeks = ref_stats['greeks_ms'] / new_stats['greeks_ms']

    return {
        'new': new_stats,
        'reference': ref_stats,
        'speedup_price': speedup_price,
        'speedup_greeks': speedup_greeks,
    }
```

---

## Step 6: Generate Performance Report

### Report Template

```markdown
# Performance Review Report / 性能审查报告

**Date**: <date>
**Code Reviewed**: <file paths>
**Benchmark Environment**: Python X.X, NumPy X.X, <CPU info>

---

## 1. 性能摘要 / Executive Summary

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Single Price | X.XX ms | < X ms | ✅/⚠️/❌ |
| Full Greeks | X.XX ms | < X ms | ✅/⚠️/❌ |
| Memory/Call | X.XX MB | < X MB | ✅/⚠️/❌ |
| vs Reference | X.XX× | ≥ 1× | ✅/⚠️/❌ |

---

## 2. 性能剖析 / Profile Results

### Hot Spots (Top 5 by cumulative time)

| Function | Calls | Time (ms) | % Total |
|----------|-------|-----------|---------|
| ... | ... | ... | ... |

### Memory Allocation

| Phase | Allocated | Peak |
|-------|-----------|------|
| Setup | X MB | X MB |
| Pricing | X MB | X MB |
| Greeks | X MB | X MB |

---

## 3. 发现的问题 / Issues Found

### High Priority

1. **[Issue Name]**
   - Location: `file.py:XX`
   - Impact: X% of execution time / X MB memory
   - Pattern: [Anti-pattern description]
   - Fix: [Recommended solution]

### Medium Priority

...

### Low Priority

...

---

## 4. 优化建议 / Optimization Recommendations

### Vectorization Opportunities

| Location | Current | Recommended | Expected Speedup |
|----------|---------|-------------|------------------|
| `file.py:XX` | Python loop | np.vectorize | 10-100× |
| ... | ... | ... | ... |

### Caching Opportunities

| Value | Recomputation Count | Cache Strategy |
|-------|---------------------|----------------|
| ... | X times | LRU cache / memoization |

### Algorithm Improvements

| Current | Improved | Complexity Change |
|---------|----------|-------------------|
| ... | ... | O(N²) → O(N log N) |

---

## 5. 基准比较 / Benchmark Comparison

### vs Reference Implementation

| Metric | New | Reference | Speedup |
|--------|-----|-----------|---------|
| Price | X.XX ms | X.XX ms | X.XX× |
| Greeks | X.XX ms | X.XX ms | X.XX× |

### Scaling Behavior

| Input Size | Time (ms) | Expected | Actual Complexity |
|------------|-----------|----------|-------------------|
| N=100 | ... | ... | ... |
| N=1000 | ... | ... | ... |
| N=10000 | ... | ... | ... |

---

## 6. 实施路线图 / Implementation Roadmap

### Quick Wins (< 1 hour)
- [ ] Fix 1: [description] - Expected: X% improvement
- [ ] Fix 2: [description] - Expected: X% improvement

### Medium Effort (1-4 hours)
- [ ] Fix 3: [description] - Expected: X% improvement

### Major Refactoring (> 4 hours)
- [ ] Fix 4: [description] - Expected: X% improvement

---

## Appendix: Benchmark Code

```python
# Code used for benchmarking
...
```
```

---

## Integration with Model Validation

When invoked by model-orchestrator:

### Input
- File paths of Developer A's implementation
- Reference engine for comparison (if available)
- Performance targets from spec

### Output
- `performance-report.md` in designated output directory
- Pass/Fail status based on performance targets
- Prioritized list of optimization opportunities

### Quality Gate Criteria

| Engine Type | Price Max | Greeks Max | Pass? |
|-------------|-----------|------------|-------|
| Analytical | 1 ms | 5 ms | Auto |
| PDE/Quad | 100 ms | 500 ms | Auto |
| Monte Carlo | 1 s | 5 s | Auto |

---

## Tool-Specific Optimizations

### NumPy

- Use `np.einsum` for complex contractions
- Use `np.dot` instead of `@` for 1D (can be faster)
- Set `dtype` explicitly to avoid conversions
- Use contiguous arrays (`.copy(order='C')`)

### SciPy

- Use sparse matrices for PDE operators
- Use `scipy.linalg` over `numpy.linalg` (often faster)
- Use `solve_banded` for tridiagonal systems

### Numba (if applicable)

- JIT compile hot loops with `@numba.jit(nopython=True)`
- Parallelize with `@numba.jit(parallel=True, nopython=True)`
- Use `@numba.vectorize` for element-wise operations

---

## Principles

1. **Measure First**: Profile before optimizing
2. **Focus on Hot Spots**: 80/20 rule applies
3. **Don't Sacrifice Readability**: Unless performance is critical
4. **Verify Correctness**: Optimizations must not change results
5. **Document Trade-offs**: Note any readability vs performance choices
