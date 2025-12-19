# Change: Add explicit trinomial volatility schemes for convertible bonds

## Why
The current convertible bond trinomial engine computes a step-local volatility but never applies it, effectively pricing with a max-volatility constant grid. We need explicit, user-selected schemes that correctly incorporate volatility term structure while preserving backward-compatible behavior for constant-vol trees.

## What Changes
- Add explicit trinomial scheme selection at engine initialization.
- Keep the existing constant-vol CRR-style trinomial as a supported scheme and document that it does not support term-structure volatility.
- Add a log-price trinomial with fixed `dx` and time-dependent probabilities that match step moments.
- Add a log-price trinomial with step-varying `dx` using re-gridding/interpolation to maintain recombination.
- Update tests and documentation to cover scheme selection and term-structure behavior.

## Impact
- Affected specs: convertible-bond-tree-engine
- Affected code: `asset/bond/engine/tree/convertible/trinomial_engine.py`, `asset/bond/engine/tree/convertible/tree_params.py`, `util/enum/engine_enums.py`, tests under `test/`
