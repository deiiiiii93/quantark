# Change: Add barrier analytical engine

## Why
- Provide closed-form pricing for single-barrier options to complement PDE and MC paths.
- Support expiry-only observation type and reusable barrier-shift logic for discrete monitoring.

## What Changes
- Add `BarrierAnalyticalEngine` with continuous, discrete (shifted barrier), and expiry decomposition methods.
- Extend observation types to include `EXPIRY` for barrier products.
- Introduce shared barrier-shift utility for discretely observed barriers.

## Impact
- Specs: `equity-analytical-engine`, `equity-barrier-products`
- Code: analytical engines, enums, barrier utility, tests

