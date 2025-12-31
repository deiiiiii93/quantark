# Tasks: Add Factor-Specific Bump Sizes for Numerical Greeks

## Implementation Tasks

### 1. Add BumpConfig Dataclass
- [x] Create `BumpConfig` dataclass in `asset/equity/param/engine_params.py`
- [x] Define all bump size fields with appropriate defaults
- [x] Add `__post_init__` validation for each field
- [x] Add `get_bump_for_factor(factor: str)` helper method
- [x] Add `from_tolerance()` class method to use Tolerance constants

### 2. Update EngineParams
- [x] Add `bump_config: Optional[BumpConfig] = None` field to `EngineParams`
- [x] Modify `__post_init__` to create `bump_config` from `bump_size` when None
- [x] Add `get_effective_bump_config()` method
- [x] Ensure existing `bump_size` validation still works
- [x] Verify `MCParams` and `PDEParams` inheritance works correctly

### 3. Update Exports
- [x] Add `BumpConfig` to `asset/equity/param/__init__.py`
- [x] Verify imports work from `asset.equity.param import BumpConfig`

### 4. Update GreeksCalculator Initialization
- [x] Add `self._bump_config` property in `__init__`
- [x] Initialize from `self.params.get_effective_bump_config()`

### 5. Update calculate_numerical_greeks
- [x] Use `self._bump_config.spot_bump` for delta/gamma
- [x] Use `self._bump_config.vol_bump` for vega
- [x] Use `self._bump_config.time_bump_days` for theta
- [x] Use `self._bump_config.rate_bump` for rho
- [x] Add `dividend_rho` to returned dictionary

### 6. Update Individual Greek Methods
- [x] Update `calculate_numerical_delta` with optional `bump` parameter
- [x] Update `calculate_numerical_gamma` with optional `bump` parameter
- [x] Update `calculate_numerical_vega` with optional `vol_bump` parameter
- [x] Update `calculate_numerical_theta` with optional `time_bump_days` parameter
- [x] Update `calculate_numerical_rho` with optional `rate_bump` parameter

### 7. Add calculate_numerical_dividend_rho
- [x] Create new method with signature matching other Greek methods
- [x] Implement dividend yield bump logic
- [x] Add docstring explaining sign convention (negative for calls, positive for puts)

### 8. Unit Tests for BumpConfig
- [x] Create `test/test_bump_config.py`
- [x] Test default values match expectations
- [x] Test `from_tolerance()` method
- [x] Test validation for each field (negative, too large, zero)
- [x] Test `get_bump_for_factor()` method

### 9. Backward Compatibility Tests
- [x] Test `EngineParams(bump_size=x)` creates valid `bump_config`
- [x] Test `EngineParams` with explicit `bump_config` uses it
- [x] Test `GreeksCalculator` with default params works
- [x] Test `GreeksCalculator` with legacy `bump_size` works

### 10. Integration Tests
- [x] Test delta with custom spot bump
- [x] Test vega with custom vol bump
- [x] Test theta with custom time bump
- [x] Test rho with custom rate bump
- [x] Test dividend_rho sign (negative for calls, positive for puts)
- [x] Test all Greeks with full custom BumpConfig

### 11. Edge Case Tests
- [x] Test dividend_rho with zero dividend yield
- [x] Test theta when time_bump exceeds maturity
- [x] Test vega with very low volatility

### 12. Documentation
- [x] Update docstrings in `GreeksCalculator` methods
- [x] Add usage examples in docstrings
- [ ] Update `CLAUDE.md` if needed

## Validation

- [x] Run `openspec validate add-factor-specific-bump-sizes --strict`
- [x] Run existing test suite: `python -m pytest test/test_european_option.py -v`
- [x] Run new tests: `python -m pytest test/test_bump_config.py test/test_greeks_bump_config.py -v`
- [x] Verify no breaking changes to existing code
