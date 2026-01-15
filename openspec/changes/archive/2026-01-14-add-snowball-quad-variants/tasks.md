## 1. Implementation
- [x] Update `SnowballQuadEngine` validation to allow airbag and call-rebate features.
- [x] Add `disable_ko_after_ki` logic to KO/KI handling in the quadrature recursion.
- [x] Update snowball quad documentation to list supported variants and KO/KI precedence rules.

## 2. Tests
- [x] Add tests for airbag V1 payoff handling.
- [x] Add tests for call-rebate V0 payoff handling.
- [x] Add tests for `disable_ko_after_ki` (KO suppressed after KI).

## 3. Validation
- [x] Run `python -m pytest test/test_snowball_quad_engine.py`.
