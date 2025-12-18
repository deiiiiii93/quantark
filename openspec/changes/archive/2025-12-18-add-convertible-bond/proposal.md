# Change: Add Convertible Bond Product and Pricing Engines

## Why
Convertible bonds are hybrid securities combining debt and equity features. They are critical instruments in corporate finance and derivatives trading. The codebase currently lacks support for convertible bonds, which limits the library's applicability in fixed-income hybrid product pricing and risk management.

## What Changes
- **NEW** `ConvertibleBond` product class in `asset/bond/product/convertible/` with comprehensive contract terms (conversion ratio, call/put schedules, coupon payments, credit features)
- **NEW** Tree-based pricing engine (binomial/trinomial) following Goldman Sachs credit-adjusted model
- **NEW** PDE-based pricing engine implementing Bloomberg's Jump-Diffusion model and Tsiveriotis-Fernandes decomposition
- **NEW** Facade engine (`ConvertibleBondEngine`) that dispatches to appropriate underlying engine based on method selection
- Integration with existing `PricingEnvironment` for market data input
- API designed to be compatible with other modules (backtest/stresstest/portfolio) via explicit, stable engine/product interfaces

## Impact
- Affected specs: None (all new capabilities)
- Affected code:
  - `asset/bond/product/convertible/convertible_bond.py` (new)
  - `asset/bond/engine/tree/convertible/` (new tree engines)
  - `asset/bond/engine/pde/convertible/` (new PDE engines)
  - `asset/bond/engine/convertible/` (new facade engine)
  - `util/enum/engine_enums.py` (add `ConvertibleBondMethod`, add `EngineType.TREE`)
- New dependencies: None (uses existing NumPy/SciPy)

## Technical Approach
Based on the documentation reviewed:
1. **Bloomberg OVCV Model**: Jump-diffusion with hazard rate for credit risk, PDE solver
2. **Goldman Sachs Hydra**: Binomial tree with credit-adjusted discount rates
3. **Hull-White (from textbook)**: Trinomial tree with default probability
4. **Tsiveriotis-Fernandes**: Coupled PDE system separating cash and equity components

The facade engine will support method selection via the two-level enum pattern:
```python
engine = ConvertibleBondEngine(method=EngineType.TREE(ConvertibleBondMethod.BINOMIAL_GS))
engine = ConvertibleBondEngine(method=EngineType.PDE(ConvertibleBondMethod.JUMP_DIFFUSION))
```

## Spec Organization
Given the scope, this change is organized into 4 related specs:
1. `convertible-bond-product` - Product definition and contract terms
2. `convertible-bond-tree-engine` - Tree-based pricing (GS, HW approaches)
3. `convertible-bond-pde-engine` - PDE-based pricing (Bloomberg, TF)
4. `convertible-bond-facade-engine` - Unified dispatcher engine
