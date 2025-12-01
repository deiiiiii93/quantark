# Change: Implement Monte Carlo Engine for European Vanilla Options

## Why
The QuantArk library currently lacks Monte Carlo pricing capabilities for European options. While analytical Black-Scholes pricing is available, Monte Carlo methods are essential for:
- Validating analytical results through simulation
- Path-dependent pricing extensions
- Handling complex volatility surfaces through simulation
- Providing a foundation for exotic option pricing

## What Changes
- Implement `EuropeanMCEngine` in `asset/equity/engine/mc/euro_mc_engine.py`
- Support three Monte Carlo methods:
  - **Normal MC**: Pseudorandom Monte Carlo simulation
  - **QMC**: Quasi-Monte Carlo using Sobol sequences
  - **RQMC**: Randomized Quasi-Monte Carlo with adaptive batching
- Integration with existing path generators in `asset/equity/process/bsm/`
- Use the two-level enum pattern `EngineType.MONTE_CARLO(MonteCarloMethod.XXX)`
- Proper variance reduction support (antithetic variates, control variates)
- Standard error estimation for MC convergence diagnostics

## Impact
- Affected specs: equity-mc-pricing (new capability)
- Affected code:
  - New file: `asset/equity/engine/mc/euro_mc_engine.py`
  - Uses: `asset/equity/process/bsm/qmc_path_generator.py`
  - Uses: `asset/equity/process/bsm/qmc_rqmc_driver.py`
  - Follows: `asset/equity/engine/base_engine.py` interface
  - Pattern: `util/enum/engine_enums.py` (MonteCarloMethod, EngineType)
