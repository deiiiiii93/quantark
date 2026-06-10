# Validation Report: American Option PDE Solver

**Generated**: 2025-02-14  
**Engine**: `asset/equity/engine/pde/american_pde_solver.py`  
**Reference**: `asset/equity/engine/pde/base_pde_solver.py`  
**Benchmark**: `asset/equity/engine/mc/american_option_mc_engine.py`

---

## Executive Summary

| Check Type | Status | Pass Rate |
|------------|--------|-----------|
| Method Implementation | ✅ PASS | - |
| Boundary Checks | ✅ PASS | 100% (6/6) |
| Benchmark Checks | ✅ PASS | 100% (4/4) |
| User Cases | ⚠️ NOT PROVIDED | - |

**Overall Status**: ✅ VALIDATED

---

## 1. Method Description

### 1.1 Pricing Method Summary

The solver uses finite-difference methods (via `BasePDESolver`) to solve the
Black-Scholes PDE in log-price space and applies an **early exercise
constraint** at each time step:

```
V(S, t) = max(continuation_value, intrinsic_value)
```

Boundary conditions follow American option logic (intrinsic at boundaries),
and terminal condition equals the payoff at maturity.

### 1.2 Reference Comparison

| Aspect | Reference | Implementation | Match |
|--------|-----------|----------------|-------|
| Terminal condition | Payoff at maturity | `set_terminal_condition()` | ✅ |
| Boundary conditions | American intrinsic bounds | `set_boundary_conditions()` | ✅ |
| Early exercise constraint | Max(intrinsic, continuation) | `_apply_step_modifications()` | ✅ |
| Grid concentration | Strike as critical point | `get_critical_points()` | ✅ |
| Time stepping | Crank-Nicolson w/ Rannacher | `BasePDESolver` | ✅ |

### 1.3 Implementation Quality

**Strengths:**
- Proper early exercise constraint at each time step
- Boundary conditions reflect immediate exercise value
- Reuses stable grid/time-step infrastructure from `BasePDESolver`

**Notes / Risks:**
- Accuracy is sensitive to `PDEParams` (grid size/time steps).  
  Ensure convergence checks for production use.

---

## 2. Boundary Checks

**Script**: `asset/equity/engine/validation/script/boundary_check_american_pde.py`

Planned tests:
- Near expiry (T→0) → payoff  
- Low volatility → intrinsic value  
- High volatility → finite, positive  
- Deep OTM call → near zero  
- American ≥ European (put)  
- American call equals European when q=0 (approximate)

**Result**: 6/6 passed (100%).

| Test | Status | Notes |
|------|:-------|-------|
| Near expiry (T→0) | ✅ | Price close to payoff |
| Low volatility (σ→0) | ✅ | Matches deterministic forward value |
| High volatility (σ=100%) | ✅ | Finite positive price |
| Deep OTM call | ✅ | Near zero |
| American ≥ European (put) | ✅ | Dominance holds |
| Call (q=0) equals European | ✅ | Early exercise not optimal |

---

## 3. Benchmark Comparison

**Benchmark Engine**: `asset/equity/engine/mc/american_option_mc_engine.py`  
**Tolerance**: 5%  
**Script**: `asset/equity/engine/validation/script/benchmark_check_american_pde.py`

Planned cases:
- ATM call/put, ITM put, OTM call  
- PDE grid: 250×250  
- MC benchmark: 100k paths (quasi-MC)

**Result**: 4/4 passed (100%).

| Case | PDE | MC | SE | Error | Status |
|------|----:|---:|---:|------:|:-------|
| ATM Call (q=2%) | 9.2273 | 9.0895 | 0.0382 | 1.52% | ✅ |
| ATM Put (q=2%) | 6.6590 | 6.6317 | 0.0238 | 0.41% | ✅ |
| ITM Put (q=2%) | 21.2385 | 21.1787 | 0.0328 | 0.28% | ✅ |
| OTM Call (q=0%) | 2.1372 | 2.0897 | 0.0185 | 2.27% | ✅ |

---

## 4. User Test Cases

No user test cases provided.  
Provide cases in the form: `S=100, K=100, T=1, r=0.05, σ=0.2, expected=10.5`

---

## 5. Recommendations

1. Re-run benchmark checks after changing `PDEParams` to ensure convergence.  
2. Add user cases if you need regression targets for specific market regimes.

---

## Appendix

### A. Script Execution Commands

```bash
python asset/equity/engine/validation/script/boundary_check_american_pde.py
python asset/equity/engine/validation/script/benchmark_check_american_pde.py
```
