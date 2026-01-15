# Snowball Quadrature Engine

This document describes the direct regime-switching quadrature engine implemented in
`asset/equity/engine/quad/snowball_quad_engine.py`. The method follows the discrete
quadrature framework in `docs/quad/quad.md` and maintains two value functions during
the backward recursion.

## Method Summary

The engine evolves two states on the same log-price grid:

1. **V_in(S, t)**: Value given knock-in has already occurred.
2. **V_out(S, t)**: Value given knock-in has not occurred.

### Terminal Conditions

- `V_in(S, T)` = `get_maturity_payoff_v1(S)`
- `V_out(S, T)` = `get_maturity_payoff_v0(S)`

### Backward Recursion

For each interval `(t_{m-1}, t_m)`:

1. **Diffusion step**: Apply FFT-based convolution to propagate `V_in` and `V_out`.
2. **Continuous KI (if enabled)**: Use Brownian-bridge hit probabilities to mix
   `V_out` into `V_in` inside the diffusion step.
3. **Observation updates**:
   - **KO**: At KO observation dates, both states jump to the KO payoff when the
     KO barrier is breached. If `disable_ko_after_ki=True`, only `V_out` is
     overwritten.
   - **Discrete KI**: At KI observation dates, `V_out` switches to `V_in` when the
     KI barrier is breached.
   - **KO precedence**: If KO and KI trigger together, KO overrides KI unless
     `disable_ko_after_ki=True`.

The price is `V_out(S0, t0)`.

## Supported Configuration

- Discrete KO observation schedules
- Discrete or continuous KI monitoring (continuous uses Brownian-bridge transitions)
- Fixed rebates or call-style rebates (via product V0 payoff)
- Airbag features (via product V1 payoff)
- Optional `disable_ko_after_ki` logic (KO only applies before KI)

## References

- Huang & Luo (2015): *A Simple and Efficient Numerical Method for Pricing Discretely
  Monitored Early-Exercise Options*  
- `docs/quad/quad.md`
