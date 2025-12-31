# Change: Add Factor-Specific Bump Sizes for Numerical Greeks

## Why

Numerical Greeks calculation uses finite difference method (FDM) which requires "bumping" market parameters to compute sensitivities. Currently, `EngineParams` has a single `bump_size` parameter used for spot delta/gamma, while vega, theta, and rho use hardcoded bump sizes (0.01 for vol/rate, 1 day for time). This is inflexible because:

1. Different risk factors have different appropriate bump sizes (spot uses relative %, vol/rate use absolute)
2. Users may need different bump sizes for different use cases (precision testing, illiquid securities, regulatory reporting)
3. Dividend yield sensitivity (dividend_rho or "psi") is commonly needed but not implemented

## What Changes

- **NEW** `BumpConfig` dataclass with per-factor bump sizes:
  - `spot_bump`: Relative bump for delta/gamma (default: 1%)
  - `vol_bump`: Absolute bump for vega (default: 1 vol point)
  - `time_bump_days`: Absolute bump in days for theta (default: 1 day)
  - `rate_bump`: Absolute bump for rho (default: 1bp)
  - `div_bump`: Absolute bump for dividend_rho (default: 1bp)
- **MODIFIED** `EngineParams` to include optional `bump_config: BumpConfig` field
- **MODIFIED** `GreeksCalculator` to use `BumpConfig` for all numerical Greeks
- **NEW** `calculate_numerical_dividend_rho` method in `GreeksCalculator`
- **MODIFIED** Individual Greek calculation methods to accept optional bump parameters

## Impact

- Affected specs: equity-greeks (MODIFIED - add bump configuration and dividend_rho)
- Affected code:
  - `asset/equity/param/engine_params.py` - Add `BumpConfig`, modify `EngineParams`
  - `asset/equity/riskmeasures/greeks_calculator.py` - Use `BumpConfig`, add `dividend_rho`
  - `asset/equity/param/__init__.py` - Export `BumpConfig`
- New tests:
  - `test/test_bump_config.py` - Unit tests for `BumpConfig`
  - `test/test_greeks_bump_config.py` - Integration tests for Greeks with custom bumps

## Backward Compatibility

- `EngineParams.bump_size` is preserved for existing code
- When `bump_config` is None, it is auto-created from `bump_size` for spot, with defaults for other factors
- Individual Greek methods accept optional bump parameters, falling back to config values
- All existing tests continue to pass without modification
