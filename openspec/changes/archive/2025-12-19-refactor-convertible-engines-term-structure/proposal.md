# Refactor Convertible Bond Engines to Support Piecewise Rate and Volatility Curves

## Summary
Update the Convertible Bond PDE and Trinomial engines to dynamically query interest rates and volatility at each time step during backward induction, enabling correct pricing against non-flat yield curves and time-dependent volatility term structures.

## Motivation
The current Convertible Bond pricing engines (Jump-Diffusion PDE, TF PDE, and Trinomial) initialize interest rates and volatility by querying the curves only once at maturity ($T$). This effectively "flattens" the term structure, ignoring any piecewise or time-dependent behavior defined in the `PricingEnvironment`.

For realistic pricing:
- **Interest rates** should use forward rates appropriate to each time step during backward induction
- **Volatility** should respect the term structure by deriving a per-step effective (piecewise) volatility from the implied volatility term structure

This limitation becomes significant when pricing long-dated convertible bonds (5-10 years) where:
1. The yield curve may be steep or inverted
2. Volatility term structure may vary significantly (e.g., higher short-term vol, lower long-term vol)

## Scope

### In Scope
1. **PDE Engines** (`ConvertibleBondJumpDiffusionEngine`, `ConvertibleBondTFEngine`):
   - Query forward rate `r(t, t+dt)` at each time step via `rate_curve.get_forward_rate(t, t + dt)`
   - Derive a per-step effective volatility `σ_step(t, t+dt)` from implied vols using total variance differences
   - Pass time-dependent parameters to `_build_matrices`

2. **Trinomial Engine** (`ConvertibleBondTrinomialEngine`):
   - Use maximum volatility over bond life for grid spacing ($dx$) to ensure stability
   - Query forward rate and per-step effective volatility at each time step
   - Recalculate transition probabilities $(p_u, p_m, p_d)$ per time step

3. **Binomial Engine** (`ConvertibleBondBinomialEngine`):
   - Add warning when non-flat curves are detected
   - No mathematical changes (standard binomial trees cannot handle time-varying vol without breaking recombination)

4. **Testing**:
   - New test file `test/test_convertible_bond_term_structure.py`
   - Tests comparing flat vs stepped curves to verify engines respect term structure

### Out of Scope
- Changes to the rate curve or volatility surface classes themselves
- Changes to other asset class engines (equity, rate)
- Performance optimizations beyond what's necessary for correctness

## Affected Components
- `asset/bond/engine/pde/convertible/jump_diffusion_engine.py`
- `asset/bond/engine/pde/convertible/tf_engine.py`
- `asset/bond/engine/tree/convertible/trinomial_engine.py`
- `asset/bond/engine/tree/convertible/binomial_engine.py`
- `test/test_convertible_bond_term_structure.py` (new)

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Trinomial tree probability negativity with varying vol | Medium | High | Use max vol over bond life for grid spacing |
| Breaking backward compatibility for flat curve users | Low | Medium | Flat curves will produce identical results |
| Performance degradation from per-step queries | Low | Low | Rate/vol queries are O(1) for most implementations |

## Success Criteria
1. PDE and Trinomial engines produce different prices for flat curve vs stepped curve with same average rate
2. All existing tests pass without modification
3. Binomial engine logs appropriate warning for non-flat curves
