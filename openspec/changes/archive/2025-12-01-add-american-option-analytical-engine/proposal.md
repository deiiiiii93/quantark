# Change: Add Analytical Pricing Engine for American Options

## Why

American options are widely used derivatives that can be exercised at any time before maturity. Currently, QuantArk has the `AmericanOption` product class but lacks an analytical pricing engine. The existing codebase contains a specification document (`ameopt_analytical_engine.md`) describing three analytical approximation methods, but the implementation is missing.

Adding this engine will provide fast, accurate pricing for American vanilla options without requiring computationally expensive PDE or binomial tree methods. Multiple methods offer flexibility for different use cases and accuracy requirements.

## What Changes

- Create new `AmericanOptionAnalyticalEngine` class in `asset/equity/engine/analytical/`
- Implement three approximation methods:
  - **BS93** (Bjerksund-Stensland 1993): Single-barrier approximation with phi auxiliary function
  - **BS02** (Bjerksund-Stensland 2002): More accurate two-barrier approximation with bivariate normal CDF
  - **BAW** (Barone-Adesi-Whaley 1987): Quadratic approximation with iterative critical price search
- Implement put-call transformation for American put options (BS93, BS02)
- Implement direct put pricing for BAW method
- Add method selection parameter (configurable via `EngineParams`, default: BS93)
- Add numerical stability safeguards (safe log/sqrt/exp, underflow/overflow protection)
- Handle special cases where American options reduce to European options
- Follow existing QuantArk patterns: inherit from `BaseEngine`, use `PricingEnvironment`, raise appropriate exceptions
- Integrate with existing exception hierarchy (`ValidationError`, `NumericalError`, `PricingError`)

## Impact

- **Affected specs**: `equity-analytical-engine` (new capability)
- **Affected code**: 
  - `asset/equity/engine/analytical/` (new file: `american_option_engine.py`)
  - `asset/equity/engine/analytical/__init__.py` (export new engine)
  - `asset/equity/param/engine_params.py` (may need to add method selection parameter)
  - Integration with `AmericanOption` product class
- **Dependencies**: Uses existing `BlackScholesEngine` as fallback for edge cases, requires `scipy.stats.multivariate_normal` for BS02 bivariate CDF
- **Testing**: Requires comprehensive tests for all three methods, pricing accuracy comparisons, numerical stability, and edge cases