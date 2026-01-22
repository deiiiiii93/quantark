# Proposal: Add Pricing Engines for Phoenix Options (MC, PDE, QUAD)

## Summary

Implement three pricing engines for Phoenix (autocallable with periodic coupons) options: Monte Carlo, PDE, and Quadrature. This mirrors the existing engine coverage for Snowball options and enables comprehensive pricing and risk analysis for Phoenix products.

## Why

The Phoenix option product (`PhoenixOption`) exists but currently has no dedicated pricing engines. While `SnowballMCEngine` can price Phoenix options (they share the autocallable structure), Phoenix has distinct coupon mechanics that warrant specialized engine implementations:

1. **Periodic Coupon Payments**: Unlike Snowball which pays coupons only on KO, Phoenix pays at each observation where the coupon barrier is hit
2. **Memory Coupon Tracking**: Phoenix accumulates missed coupons across observations (when `memory_coupon=True`)
3. **Day Count Convention**: Phoenix uses `DayCountConvention` for coupon accrual, requiring proper time fraction handling
4. **Coupon Pay Type**: Phoenix supports INSTANT vs EXPIRY coupon payment timing

The existing Snowball engines can be extended to handle these Phoenix-specific mechanics.

## What Changes

### New Files
- `asset/equity/engine/mc/phoenix_mc_engine.py` - Monte Carlo engine for Phoenix options (may extend SnowballMCEngine)
- `asset/equity/engine/pde/phoenix_pde_solver.py` - Two-Surface PDE solver with coupon tracking
- `asset/equity/engine/quad/phoenix_quad_engine.py` - Regime-switching quadrature with coupon states
- `test/test_phoenix_mc.py` - MC engine unit tests
- `test/test_phoenix_pde.py` - PDE solver unit tests
- `test/test_phoenix_quad.py` - QUAD engine unit tests

### Modified Files
- `asset/equity/engine/mc/__init__.py` - Export `PhoenixMCEngine`
- `asset/equity/engine/pde/__init__.py` - Export `PhoenixPDESolver`
- `asset/equity/engine/quad/__init__.py` - Export `PhoenixQuadEngine`
- `asset/equity/engine/pde_engine.py` - Add `PhoenixOption` to dispatcher map
- `asset/equity/engine/quad_engine.py` - Add `PhoenixOption` to dispatcher map
- `asset/equity/CLAUDE.md` - Document Phoenix engine usage

### New Specs
- `phoenix-mc-engine` - Monte Carlo engine specification
- `phoenix-pde-engine` - PDE solver specification
- `phoenix-quad-engine` - Quadrature engine specification

## Problem Statement

Currently, Phoenix options pricing relies on the generic Snowball MC engine, which does not fully leverage Phoenix's coupon structure. Key limitations:

1. **No Dedicated MC Engine**: No specialized event statistics or coupon tracking for Phoenix
2. **No PDE Support**: Cannot use deterministic PDE pricing for Phoenix
3. **No QUAD Support**: Cannot use fast quadrature pricing for Phoenix
4. **Missing Greeks Methods**: No engine-level Greeks for Phoenix products

## Proposed Solution

### Phase 1: Monte Carlo Engine (`PhoenixMCEngine`)

Extend the Snowball MC engine pattern with:
- Per-observation coupon barrier checks
- Memory coupon accumulation logic
- Coupon pay type handling (INSTANT vs EXPIRY discounting)
- Day count convention support for time fractions
- Detailed `PhoenixMCResult` with coupon breakdown

### Phase 2: PDE Engine (`PhoenixPDESolver`)

Extend the Two-Surface PDE method for Snowball with:
- Third surface dimension for memory coupon state (or encode in payoff)
- Coupon barrier checks at observation times
- Proper discounting for INSTANT coupon payments
- Integration with `PDEEngine` dispatcher

### Phase 3: Quadrature Engine (`PhoenixQuadEngine`)

Extend the regime-switching quadrature for Snowball with:
- Multiple value surfaces for coupon states
- Coupon barrier transitions at observation times
- Memory coupon accumulation in value propagation
- Brownian-bridge handling for continuous KI with coupons

## Scope

### In Scope

**All Engines:**
- Support for standard and reverse Phoenix structures
- Discrete KO observation with time-varying barriers
- Continuous and discrete KI monitoring
- Coupon barrier checks at each observation
- Memory coupon accumulation
- Day count convention support
- INSTANT and EXPIRY coupon payment timing
- Protection types (NONE, PARTIAL, FULL)
- Airbag payoff structures
- Integration with unified engine dispatchers
- Integration with `GreeksCalculator`

**MC Engine:**
- `PhoenixMCResult` dataclass with per-observation statistics
- Path-wise coupon tracking
- Event statistics API

**PDE Engine:**
- Analytical Delta and Gamma from grid
- Grid construction with coupon barrier concentration

**QUAD Engine:**
- Fast O(N) pricing via convolution
- Deterministic baseline for MC validation

### Out of Scope

- Path-dependent coupon rates (coupons are fixed per observation)
- Multiple underlying assets
- Stochastic volatility models

## Dependencies

- `PhoenixOption` product (exists)
- `CouponBarrierConfig` configuration (exists)
- `SnowballMCEngine` as reference implementation (exists)
- `SnowballPDESolver` as reference implementation (exists)
- `SnowballQuadEngine` as reference implementation (exists)
- `BasePDESolver` base class (exists)
- `QuadratureCore` quadrature infrastructure (exists)

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Memory coupon state explosion in PDE | Use payoff-encoding rather than extra surfaces |
| Complex coupon timing logic | Delegate to product's coupon calculation methods |
| MC convergence with many coupons | Use QMC with antithetic variates |
| QUAD complexity with coupon states | Start with discrete coupon barriers only |

## Success Criteria

1. All three engines produce consistent prices within 0.5% for standard configurations
2. MC engine handles all Phoenix helper configurations (`create_standard_phoenix`, etc.)
3. PDE and QUAD engines provide deterministic baselines for MC validation
4. Greeks are finite and stable across all engines
5. Performance: MC < 2s (100k paths), PDE < 1s, QUAD < 0.5s for typical configurations
6. All existing Phoenix helper functions work with new engines

## Implementation Order

1. **MC Engine First**: Most flexible, validates Phoenix mechanics
2. **PDE Engine Second**: Deterministic baseline, validates MC
3. **QUAD Engine Third**: Performance optimization, validates PDE
