# Change: Exact conversion probability for convertible bond PDE engines

## Why
Convertible bond PDE engines currently do not provide a mathematically correct “eventual conversion probability” consistent with the optimal exercise/call/put policy, which prevents accurate risk reporting and method parity with tree engines.

## What Changes
- Add an auxiliary PDE solve for the conversion event indicator to compute an exact (within the PDE discretization) risk-neutral probability of eventual conversion under the same optimal policy used for pricing.
- Return `conversion_probability` from both PDE engines (`ConvertibleBondJumpDiffusionEngine`, `ConvertibleBondTFEngine`) via `price_with_details()`.
- Add regression tests for boundary cases (always-convert vs never-convert) and ensure probabilities are in `[0, 1]`.

## Impact
- Affected specs: `convertible-bond-pde-engine`, `convertible-bond-facade-engine`
- Affected code:
  - `asset/bond/engine/pde/convertible/jump_diffusion_engine.py`
  - `asset/bond/engine/pde/convertible/tf_engine.py`
  - `test/test_convertible_bond_engines.py`
