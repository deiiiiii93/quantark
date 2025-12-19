# Tasks: Add Snowball Option Helper Functions

## Implementation Tasks

### Phase 1: Core Helper Functions

- [x] **1.1** Create `asset/equity/product/option/snowball_helpers.py` with module docstring and imports
- [x] **1.2** Implement `create_standard_snowball()` - basic snowball with flat KO barrier, continuous KI
- [x] **1.3** Implement `create_stepdown_snowball()` - KO barrier decreases each observation period
- [x] **1.4** Implement `create_european_ki_snowball()` - KI only at maturity (discrete single observation)
- [x] **1.5** Implement `create_parachute_snowball()` - last KO barrier equals KI barrier
- [x] **1.6** Implement `create_phoenix_snowball()` - periodic coupons regardless of KO status
- [x] **1.7** Implement `create_airbag_snowball()` - participation-based KI protection structure

### Phase 2: Utilities and Export

- [x] **2.1** Add `generate_ko_observation_dates()` utility for common observation frequencies
- [x] **2.2** Add `generate_stepdown_barriers()` utility for step-down barrier generation
- [x] **2.3** Update `asset/equity/product/option/__init__.py` to export helper functions
- [ ] **2.4** Update `asset/equity/__init__.py` if needed for top-level exports

### Phase 3: Testing

- [x] **3.1** Create `test/test_snowball_helpers.py` with unit tests
- [x] **3.2** Test `create_standard_snowball()` - verify default configuration
- [x] **3.3** Test `create_stepdown_snowball()` - verify decreasing barriers
- [x] **3.4** Test `create_european_ki_snowball()` - verify single KI observation at maturity
- [x] **3.5** Test `create_parachute_snowball()` - verify last KO barrier equals KI barrier
- [x] **3.6** Test `create_phoenix_snowball()` - verify periodic coupon behavior
- [x] **3.7** Test `create_airbag_snowball()` - verify participation structure
- [x] **3.8** Test parameter override via `**kwargs` for all helpers
- [x] **3.9** Test validation errors for invalid parameter combinations

### Phase 4: Documentation and Examples

- [x] **4.1** Add docstrings with examples to each helper function
- [ ] **4.2** Update `example/snowball_mc_demo.py` to demonstrate helper usage
- [ ] **4.3** Update `asset/equity/CLAUDE.md` to document snowball helpers

## Dependencies

- Tasks 1.1-1.7 can be done in parallel after 1.1 creates the file
- Tasks 2.1-2.2 can be done in parallel with Phase 1
- Phase 3 depends on Phase 1 and 2
- Phase 4 depends on Phase 3

## Acceptance Criteria

1. All helper functions create valid `SnowballOption` instances
2. Default parameters produce sensible market-standard structures
3. All parameters can be overridden via `**kwargs`
4. Helpers reduce typical snowball creation from ~30 lines to ~5 lines
5. All tests pass with >90% coverage of helpers module
