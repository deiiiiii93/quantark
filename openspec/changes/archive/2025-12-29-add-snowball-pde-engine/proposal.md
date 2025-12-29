# Proposal: Add PDE Engine for Snowball Options

## Summary

Implement a PDE (Partial Differential Equation) pricing engine for Snowball (autocallable) options using the Two-Surface method. This provides an alternative to the existing Monte Carlo engine (`SnowballMCEngine`) with better accuracy for Greeks calculation and deterministic results.

## Why

Currently, Snowball options can only be priced via Monte Carlo simulation (`SnowballMCEngine`). While MC is flexible and handles path-dependence well, it has limitations:

1. **Stochastic Error**: MC results have standard error that requires many paths (100k+) for accuracy
2. **Greeks Instability**: Numerical Greeks via MC bump-and-reprice have high variance
3. **Speed vs Accuracy Tradeoff**: High accuracy requires slow simulation
4. **No Deterministic Baseline**: Difficult to validate MC results without an alternative method

A PDE-based approach provides deterministic pricing, smooth Greeks, and serves as a validation baseline for MC.

## What Changes

### New Files
- `asset/equity/engine/pde/snowball_pde_solver.py` - Two-Surface PDE solver for Snowball options
- `test/test_snowball_pde.py` - Unit tests for the PDE solver

### Modified Files
- `asset/equity/engine/pde/__init__.py` - Export `SnowballPDESolver`
- `asset/equity/engine/pde_engine.py` - Add `SnowballOption` to dispatcher map
- `asset/equity/product/option/snowball_option.py` - Bug fix: add missing `self.notional = notional`

## Problem Statement

Currently, Snowball options can only be priced via Monte Carlo simulation (`SnowballMCEngine`). While MC is flexible and handles path-dependence well, it has limitations:

1. **Stochastic Error**: MC results have standard error that requires many paths (100k+) for accuracy
2. **Greeks Instability**: Numerical Greeks via MC bump-and-reprice have high variance
3. **Speed vs Accuracy Tradeoff**: High accuracy requires slow simulation
4. **No Deterministic Baseline**: Difficult to validate MC results without an alternative method

## Proposed Solution

Implement `SnowballPDESolver` using the **Two-Surface PDE Method** described in `asset/equity/engine/docs/snowball_pde_engine.md`:

### Key Design Elements

1. **Two Coupled Surfaces**:
   - `V0(S, t)`: Value when KI has NOT occurred
   - `V1(S, t)`: Value when KI HAS occurred
   - Surfaces interact at barrier observation times

2. **Backward Induction**:
   - Solve from maturity to present
   - Apply KO payoffs when KO barriers are hit
   - Apply KI jumps: `V0 → V1` when KI barrier is hit

3. **Grid Design**:
   - Non-uniform spatial grid concentrated at spot, strike, KI barrier, KO barriers
   - Time grid aligned with observation dates (both KO and KI)

4. **Integration with `PDEEngine`**:
   - Extend the unified `PDEEngine` dispatcher to route `SnowballOption` to `SnowballPDESolver`

## Scope

### In Scope

- `SnowballPDESolver` class implementing the 2-surface method
- Support for standard and reverse snowball structures
- Discrete KO observation with time-varying barriers and rates
- Continuous and discrete KI monitoring
- Support for `disable_ko_after_ki` logic
- INSTANT and EXPIRY coupon payment timing
- Protection types (NONE, PARTIAL, FULL)
- Airbag payoff structures
- Integration with `PDEEngine` unified interface
- Integration with `GreeksCalculator` for numerical Greeks
- Analytical Greeks (Delta and Gamma) extraction from PDE grid, which means one single price gives value, delta and gamma

### Out of Scope

- Phoenix snowball periodic coupon feature (requires separate surfaces per coupon state)
- Step-down KI barriers (KI must be scalar for continuous monitoring)

## Dependencies

- `SnowballOption` product (exists)
- `BasePDESolver` base class (exists)
- `PDEEngine` unified interface (exists)
- `BarrierPDESolver` patterns for barrier handling (reference implementation)

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Numerical instability near barriers | Use Rannacher smoothing and concentrated grids |
| Grid resolution vs performance | Adaptive defaults based on observation count |
| Complex payoff logic | Delegate to product's `get_maturity_payoff_v0/v1` methods |

## Success Criteria

1. PDE prices agree with MC prices within 0.5% for standard configurations
2. Numerical Greeks via PDE have < 1% variance vs analytical approximations
3. Performance: < 1 second for typical snowball configurations
4. All existing snowball helper functions work with PDE engine
