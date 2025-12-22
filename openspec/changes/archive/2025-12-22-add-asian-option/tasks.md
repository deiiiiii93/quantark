# Tasks: Add Asian Option Product

## 1. Add Enum Types
- [x] 1.1 Add `AveragingType` enum (ARITHMETIC, GEOMETRIC) to `util/enum/option_enums.py`
- [x] 1.2 Add `AsianStrikeType` enum (FIXED, FLOATING) to `util/enum/option_enums.py`
- [x] 1.3 Export new enums from `util/enum/__init__.py`

## 2. Implement Asian Option Product
- [x] 2.1 Create `asset/equity/product/option/asian_option.py`
- [x] 2.2 Implement `AsianOption` class extending `BaseEquityOption`
- [x] 2.3 Add observation schedule support (list of observation times/dates)
- [x] 2.4 Implement `get_payoff()` for fixed strike (arithmetic average)
- [x] 2.5 Implement `get_payoff()` for floating strike (arithmetic average)
- [x] 2.6 Add `get_average()` method to compute average from price path
- [x] 2.7 Add proper validation for averaging parameters

## 3. Export and Integration
- [x] 3.1 Export `AsianOption` from `asset/equity/product/option/__init__.py`
- [x] 3.2 Export `AsianOption` from `asset/equity/product/__init__.py`

## 4. Testing
- [x] 4.1 Create `test/test_asian_option.py`
- [x] 4.2 Test fixed strike call/put payoffs
- [x] 4.3 Test floating strike call/put payoffs
- [x] 4.4 Test validation (invalid inputs)
- [x] 4.5 Test edge cases (single observation, at expiry)

## 5. Documentation
- [x] 5.1 Create demo script `example/asian_option_demo.py`
