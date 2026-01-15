# Change: Add Snowball Quad support for airbag, call-rebate, and disable-KO-after-KI

## Why
Snowball quadrature currently rejects several common product variants that are already supported by the Monte Carlo engine. Adding these variants removes a coverage gap and enables faster pricing for real-world structures.

## What Changes
- Allow airbag features in `SnowballQuadEngine` by leveraging existing V1 payoff logic.
- Allow call-rebate V0 payoff features in `SnowballQuadEngine` by relying on product V0 payoff logic.
- Add `disable_ko_after_ki` handling in quadrature recursion (KO only applies before KI).
- Update snowball quad documentation and tests to cover these variants.

## Impact
- Affected specs: `snowball-quad-engine`
- Affected code: `asset/equity/engine/quad/snowball_quad_engine.py`, `asset/equity/engine/docs/snowball_quad_engine.md`, `test/test_snowball_quad_engine.py`
