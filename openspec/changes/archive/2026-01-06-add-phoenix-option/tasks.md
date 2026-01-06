## 1. Core Product Implementation

- [x] 1.1 Create `phoenix_config.py` with `CouponBarrierConfig` dataclass
  - Define coupon_barrier, coupon_rate, coupon_pay_type fields
  - Add day_count_convention field (DayCountConvention enum)
  - Add memory_coupon boolean field
  - Implement validation in `__post_init__`

- [x] 1.2 Create `phoenix_option.py` with `PhoenixOption` class
  - Inherit from `BaseEquityOption`
  - Add configuration attributes (barrier_config, coupon_config, payoff_config, accrual_config, airbag_config)
  - Add is_reverse field for standard/reverse variants
  - Implement `__post_init__` validation

- [x] 1.3 Implement coupon barrier methods
  - `is_coupon_triggered(spot, obs_idx)` - check if coupon barrier is hit
  - `get_coupon_payoff(obs_idx)` - calculate single coupon amount
  - `get_coupon_year_fraction(start, end)` - use day_count_convention

- [x] 1.4 Implement KO/KI barrier methods (similar to Snowball)
  - `is_ko_triggered(spot, obs_idx)` - check if KO barrier is hit
  - `is_ki_triggered(spot, obs_idx)` - check if KI barrier is hit
  - `get_ko_payoff(spot, obs_idx, accumulated_coupons)` - KO payoff with coupons

- [x] 1.5 Implement maturity payoff methods
  - `get_maturity_payoff_v0(spot, accumulated_coupons)` - not knocked-in payoff
  - `get_maturity_payoff_v1(spot)` - knocked-in payoff (participation in downside)
  - `get_payoff(spot, knocked_in, accumulated_coupons)` - router method

## 2. Helper Functions

- [x] 2.1 Create `phoenix_helpers.py` with factory functions
  - `create_standard_phoenix()` - basic phoenix with defaults
  - `create_stepdown_phoenix()` - step-down KO barriers
  - `create_reverse_phoenix()` - reverse direction (down KO, up KI)
  - `create_memory_phoenix()` - explicit memory coupon
  - `create_non_memory_phoenix()` - no memory coupon

- [x] 2.2 Add validation utilities
  - Parameter validation (positive prices, valid barriers)
  - Barrier ordering validation (coupon_barrier between KI and KO typically)

## 3. Module Exports

- [x] 3.1 Update `asset/equity/product/option/__init__.py`
  - Export `PhoenixOption`
  - Export `CouponBarrierConfig`
  - Export helper functions

## 4. Testing

- [x] 4.1 Create `test/test_phoenix_option.py`
  - Test product creation with various configurations
  - Test coupon barrier triggering logic
  - Test memory coupon accumulation
  - Test KO/KI triggering (standard and reverse)
  - Test maturity payoffs (V0 and V1)
  - Test day count fraction calculations
  - Test step-down barrier variants
  - Test validation errors for invalid inputs

## 5. Documentation and Examples

- [x] 5.1 Create `example/phoenix_option_demo.py`
  - Demonstrate standard Phoenix creation
  - Demonstrate step-down Phoenix creation
  - Demonstrate memory vs non-memory coupon
  - Show coupon payoff calculations
  - Show day count convention effects
