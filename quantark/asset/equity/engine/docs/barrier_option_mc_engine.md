# Barrier Option Monte Carlo Engine

This document describes the Monte Carlo engine for barrier options
implemented in `asset/equity/engine/mc/barrier_option_mc_engine.py`.

## Method

The engine simulates risk-neutral GBM paths and applies barrier logic based
on the monitoring type:

- **Continuous**: checks barrier crossings along the simulated path grid.
- **Discrete**: checks barrier hits at observation times from the
  `ObservationSchedule` (or legacy observation dates).
- **Expiry**: checks the barrier only at maturity.

Knock-out and knock-in payoffs follow standard definitions:
- Knock-out: rebate on hit (or zero if rebate is zero), vanilla payoff otherwise.
- Knock-in: vanilla payoff on hit, rebate otherwise.

## Brownian-Bridge Optionality

For continuous monitoring, the engine can optionally approximate barrier
crossings between time steps using a Brownian-bridge crossing probability.
When enabled, the engine computes an approximate survival probability per
path and uses it to weight the payoff (and rebate timing if `pay_at_hit=True`).

This is controlled by `use_brownian_bridge=True` at engine construction.

## Discounting

Discounting uses the risk-free rate from the pricing environment:

- Terminal payoffs are discounted by `exp(-r * T)`.
- If `pay_at_hit=True` and a hit time is identified, rebates are discounted
  to the hit time.

## Numerical Considerations

- Near expiry: if maturity is effectively zero, the payoff is returned
  directly.
- The engine enforces non-negative prices and validates input parameters.
- Antithetic variates are supported for pseudo-MC (not QMC).

## References

- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*.
- Broadie, M., & Glasserman, P. (1997). A stochastic mesh method for pricing
  high-dimensional American options (Brownian bridge barrier correction).
