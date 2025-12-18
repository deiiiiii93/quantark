# Design: Convertible Bond Product and Pricing Engines

## Context
Convertible bonds are corporate debt securities that give the holder the right to convert into a predetermined number of shares of the issuer's common stock. They combine characteristics of:
- Fixed income (coupons, principal, credit risk)
- Equity derivatives (conversion option, American-style exercise)

The pricing involves modeling the interaction between stock price dynamics, interest rates, and credit risk.

## Goals
- Provide a comprehensive `ConvertibleBond` product class capturing all relevant contract terms
- Implement multiple pricing approaches (tree, PDE) to support different use cases
- Maintain API consistency with existing products for seamless integration
- Support credit risk modeling via hazard rates and credit spreads

## Non-Goals
- Two-factor models (stochastic rates) - single factor (stock) for initial implementation
- Exotic convertible features (contingent conversion, mandatory conversion, CoCos)
- Real-time market data integration

## Decisions

### Decision 1: Product Structure
**What**: `ConvertibleBond` extends `BaseBondProduct` with conversion-specific attributes
**Why**: Leverages existing bond infrastructure (cashflows, accrued interest) while adding conversion features
**Alternatives**:
- Separate class hierarchy: Would duplicate bond functionality
- Composition pattern: More complex, less intuitive

### Decision 2: Credit Risk Modeling
**What**: Support both credit-adjusted discounting (GS) and hazard rate (Bloomberg) approaches
**Why**: Different trading desks prefer different models; both are industry-standard
**Implementation**:
- GS approach: Credit-adjusted discount rate `y = p*r + (1-p)*d` where p = conversion probability
- Bloomberg approach: Jump-diffusion with hazard rate λ, stock jump on default

### Decision 3: Engine Method Pattern
**What**: Use two-level enum pattern consistent with `AmericanOptionAnalyticalEngine`
**Why**: Maintains codebase consistency and enables clear method selection
**Pattern**:
```python
class ConvertibleBondMethod(Enum):
    BINOMIAL_GS = "binomial_gs"        # Goldman Sachs credit-adjusted binomial
    TRINOMIAL_HW = "trinomial_hw"      # Hull-White trinomial with default
    JUMP_DIFFUSION = "jump_diffusion"  # Bloomberg OVCV model
    TF = "tf"                          # Tsiveriotis-Fernandes decomposition
```

**Two-level typing usage**:
```python
# Model selection (engine family + model)
engine = ConvertibleBondEngine(method=EngineType.TREE(ConvertibleBondMethod.BINOMIAL_GS))
engine = ConvertibleBondEngine(method=EngineType.PDE(ConvertibleBondMethod.JUMP_DIFFUSION))

# PDE numerical scheme selection remains separate (uses existing PDEMethod enum)
engine = ConvertibleBondEngine(
    method=EngineType.PDE(ConvertibleBondMethod.TF),
    scheme=PDEMethod.CRANK_NICOLSON,
)
```

### Decision 4: Facade Engine Design
**What**: `ConvertibleBondEngine` dispatches to specialized engines based on method
**Why**: Provides unified API while allowing method-specific optimizations
**Dispatch logic**:
- Tree methods → `ConvertibleBondBinomialEngine` or `ConvertibleBondTrinomialEngine`
- PDE methods → `ConvertibleBondJumpDiffusionEngine` or `ConvertibleBondTFEngine`

### Decision 5: PricingEnvironment Integration
**What**: Engines accept `PricingEnvironment` plus additional credit parameters
**Why**: Consistent with existing engine patterns; credit data may not be in standard PricingEnvironment
**Additional inputs**:
- `credit_spread`: Observable from issuer's straight bonds
- `hazard_rate`: For jump-diffusion model (can derive from spread)
- `recovery_rate`: Fraction recovered on default (typically 40%)
- `stock_jump_on_default`: η parameter (typically 40%)

### Decision 6: Add `EngineType.TREE`
**What**: Extend `EngineType` with a `TREE` member.
**Why**: Required for two-level typing of tree-based engines (`EngineType.TREE(ConvertibleBondMethod.BINOMIAL_GS)`), consistent with existing `EngineType.PDE(...)` usage.

## Risks / Trade-offs

### Risk: Numerical Stability Near Boundaries
**Issue**: PDE methods can be unstable near conversion, call, or put boundaries
**Mitigation**: Rannacher smoothing (implicit Euler steps near boundary conditions)

### Risk: Credit Model Calibration
**Issue**: Hazard rate calibration requires CDS or bond spread data
**Mitigation**: Accept hazard rate directly; provide utility function to derive from spread

### Risk: Performance for Large Trees
**Issue**: Tree methods can be slow for long-dated convertibles
**Mitigation**: Default to moderate grid size (200 steps); allow user override

## Migration Plan
Not applicable - all new functionality.

## Open Questions
1. Should we support discrete dividends in addition to continuous yield?
   - **Resolution**: Yes, discrete dividends are common; include in product spec
2. Should Greeks be computed analytically where possible?
   - **Resolution**: Numerical Greeks via bump-and-reprice initially; analytical later
3. Should the COCB (cash-only component) from TF model be exposed?
   - **Resolution**: Yes, as optional output for decomposition analysis
