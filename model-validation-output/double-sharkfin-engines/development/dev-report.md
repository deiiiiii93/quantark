# 开发总结 / Development Summary

## 1. Scope

Implemented pricing engines for `DoubleSharkfinOption`:

- `asset/equity/engine/analytical/double_sharkfin_option_analytical_engine.py`
- `asset/equity/engine/mc/double_sharkfin_option_mc_engine.py`

## 2. Analytical Model

The analytical engine decomposes price into:

- double knock-out vanilla participation leg
- knock-out cash leg
- no-hit cash leg

It reuses `DoubleBarrierOptionAnalyticalEngine` for the option leg, uses
closed-form terminal lognormal probabilities for expiry monitoring, uses a
double-barrier killed-density survival series for continuous cash legs, and
uses BGK shifted barriers for regular discrete schedules.

## 3. Monte Carlo Model

The MC engine supports:

- `MonteCarloMethod.PSEUDO`
- `MonteCarloMethod.QUASI`
- `MonteCarloMethod.RANDOMIZED_QUASI`
- two-level enum construction via `EngineType.MONTE_CARLO(method)`
- expiry, discrete, and continuous monitoring
- optional Brownian-bridge crossing probabilities for continuous monitoring

## 4. Integration

Updated package exports:

- `asset/equity/engine/analytical/__init__.py`
- `asset/equity/engine/mc/__init__.py`
- `asset/equity/engine/__init__.py`

Added reference documentation:

- `asset/equity/engine/docs/double_sharkfin_option_engines.md`

Added tests:

- `test/test_double_sharkfin_analytical_engine.py`
- `test/test_double_sharkfin_mc_engine.py`
