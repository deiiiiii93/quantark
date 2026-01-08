# Tasks: Refactor Product-Position Separation

## 1. Bond Product Foundation
- [x] 1.1 Add `get_denominator()` to `BaseBondProduct` (note: new abstract method is breaking unless a default is provided)
- [x] 1.2 Implement `get_denominator()` in `FixedBond` (returns `self.denominator`)
- [x] 1.3 Implement `get_denominator()` in `FRN`
- [x] 1.4 Implement `get_denominator()` in `BondForward`
- [x] 1.5 Implement `get_denominator()` in `BondFutures`
- [x] 1.6 Add `get_denominator()` to `InterestRateSwap` (time-based)

## 2. Equity Option Products (BREAKING)
- [x] 2.1 Remove `notional` and `quantity` attributes from `BaseEquityOption`
- [x] 2.2 Remove `reconciliation_policy` parameter from `BaseEquityOption.__init__`
- [x] 2.3 Remove `get_notional()`, `set_notional()`, `get_quantity()`, `set_quantity()` methods
- [x] 2.4 Remove `_validate_notional_quantity()` method
- [x] 2.5 Remove `_resolve_notional_quantity()` method (~100 lines)
- [x] 2.6 Add `contract_multiplier: float = 1.0` to `BaseEquityOption` (and all derived equity options)
- [x] 2.7 Ensure equity option payoffs/prices are per-contract and include `contract_multiplier`
- [x] 2.8 Update SnowballOption payoffs to use `self.initial_price * self.contract_multiplier` instead of `self.notional`
- [x] 2.9 Update PhoenixOption payoffs to use `self.initial_price * self.contract_multiplier` instead of `self.notional`
- [x] 2.10 Update `create_standard_snowball()` helper to remove notional parameter
- [x] 2.11 Update `create_standard_phoenix()` helper to remove notional parameter
- [x] 2.12 Update all snowball/phoenix helper functions

## 2A. Equity Engine & Greeks Scaling (BREAKING)
- [x] 2A.1 Ensure analytical engines (e.g., BlackScholesEngine) include `contract_multiplier` in per-contract price/greeks
- [x] 2A.2 Ensure numerical Greeks calculators respect `contract_multiplier` consistently (product-level)

## 3. Bond Products (BREAKING)
- [x] 3.1 Rename `FixedBond.notional` attribute to `denominator`
- [x] 3.2 Update `FixedBond.__post_init__` validation for `denominator`
- [x] 3.3 Update `FixedBond._generate_schedule()` to use `denominator`
- [x] 3.4 Update `FixedBond.get_cashflows()` to use `denominator`
- [x] 3.5 Update `FixedBond.calculate_accrued_interest()` to use `denominator`
- [x] 3.6 Update `FixedBond.get_coupon_payment()` to use `denominator`
- [x] 3.7 Update all bond subclasses similarly

## 4. IRS Products (BREAKING)
- [x] 4.1 Rename IRS factory function `notional` parameter to `denominator`
- [x] 4.2 Update IRS `get_denominator(as_of_date)` method implementation
- [x] 4.3 Document `NotionalSchedule` as denominator schedule

## 5. Position Classes
- [x] 5.1 Update `EquityPosition.get_market_value()` for per-contract pricing
- [x] 5.2 Update `EquityPosition.get_greeks()` for quantity scaling
- [x] 5.3 Add `EquityPosition.get_actual_notional()` convenience method
- [x] 5.4 Update `FIPosition.get_market_value()` for denominator handling
- [x] 5.5 Add `FIPosition.get_actual_notional()` method

## 6. Engine Verification
- [x] 6.1 Verify `SnowballMCEngine` returns per-contract prices
- [x] 6.2 Verify `bond_discount_engine` returns per-unit prices (consistent with bond denominator convention)
- [x] 6.3 Verify `irs_discount_engine` returns per-unit prices (consistent with IRS denominator convention)
- [x] 6.4 Update any engines that incorrectly scale by notional

## 7. Remove Enum
- [x] 7.1 Remove `NotionalQuantityPolicy` enum from `util/enum/option_enums.py`
- [x] 7.2 Remove `NotionalQuantityPolicy` from `util/enum/__init__.py`
- [x] 7.3 Remove imports from `base_equity_option.py`

## 8. Test Updates
- [x] 8.1 Update `test/test_snowball_option.py` (remove notional params)
- [x] 8.2 Update `test/test_snowball_pde.py`
- [x] 8.3 Update `test/test_snowball_mc_engine.py`
- [x] 8.4 Update `test/test_phoenix_option.py` (remove notional params)
- [x] 8.5 Update `test/test_fixed_bond.py` (notional → denominator)
- [x] 8.6 Update `test/test_frn.py`
- [x] 8.7 Update `test/test_irs.py`
- [x] 8.8 Update `test/test_portfolio.py`

## 9. Example Updates
- [x] 9.1 Update `example/snowball_mc_demo.py` (line 89)
- [x] 9.2 Update `example/phoenix_option_demo.py`
- [x] 9.3 Update `example/fixed_bond_demo.py`
- [x] 9.4 Update `example/irs_demo.py`
- [x] 9.5 Update `example/portfolio_demo.py`

## 10. Documentation
- [x] 10.1 Update `CLAUDE.md` with new pattern
- [x] 10.2 Update `asset/equity/CLAUDE.md`
- [x] 10.3 Update class docstrings

## 11. Validation
- [x] 11.1 Run all tests with `python -m pytest`
- [x] 11.2 Compare before/after pricing results
- [x] 11.3 Verify Greeks scaling is correct
