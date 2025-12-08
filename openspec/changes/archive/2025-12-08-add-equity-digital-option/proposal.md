# Change: Add cash-or-nothing digital option

## Why
- Expand equity coverage to include cash-or-nothing digital options with closed-form pricing.
- Enable consistent validation and specification coverage for digital products alongside vanilla options.

## What Changes
- Add new capability spec for cash-or-nothing European digital option product.
- Add analytical pricing requirement for digital calls and puts via closed-form Black-Scholes.
- Implement product, analytical engine, and tests.

## Impact
- Affected specs: equity-digital-products, equity-analytical-engine
- Affected code: asset/equity/product/option, asset/equity/engine/analytical, tests

