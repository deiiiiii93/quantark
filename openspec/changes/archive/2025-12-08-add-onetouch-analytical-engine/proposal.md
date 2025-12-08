# Change: Add one-touch/no-touch analytical engine

## Why
- Enable closed-form pricing for one-touch and no-touch options to match existing product coverage.
- Provide analytical parity with barrier and digital engines for continuous, discrete (shifted), and expiry-only monitoring.

## What Changes
- Add analytical engine supporting one-touch and no-touch payoff styles with continuous, discrete (barrier-shifted), and expiry-only monitoring paths.
- Handle pay-at-hit vs pay-at-expiry for one-touch; treat no-touch as expiry-only payment.
- Integrate with existing analytical engine exports and add regression tests.

## Impact
- Specs: equity-analytical-engine
- Code: analytical engine module, exports, tests

