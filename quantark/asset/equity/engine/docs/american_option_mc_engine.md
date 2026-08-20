# American Option Monte Carlo Engine (LSM)

This document describes the Monte Carlo engine for American vanilla options
implemented in `asset/equity/engine/mc/american_option_mc_engine.py`.

## Method

The engine uses the Longstaff-Schwartz least squares Monte Carlo (LSM)
algorithm to estimate the optimal early exercise policy. The underlying
asset paths are simulated under the risk-neutral measure using a GBM model.

At each exercise step (working backward in time):
1. Compute intrinsic values for in-the-money paths.
2. Regress discounted continuation cashflows onto a polynomial basis in
   normalized spot (S / K).
3. Exercise when intrinsic value exceeds estimated continuation value.

## Regression Basis

The default basis is polynomial with degree 2:
- 1
- x
- x^2

where x = S / K. The degree can be adjusted via `regression_degree`.

## Discounting

Cashflows are discounted step-by-step using:

discount_factor = exp(-r * dt)

The final price is the average of discounted cashflows at time 0.

## Numerical Considerations

- Near expiry: if T is effectively zero, the payoff is returned directly.
- The engine enforces price >= intrinsic value within a small tolerance.
- Regression falls back to pathwise continuation values when too few
  in-the-money paths are available.

## References

- Longstaff, F. A., and Schwartz, E. S. (2001). Valuing American Options
  by Simulation: A Simple Least-Squares Approach. Review of Financial Studies.
