# Design: Regime-Switching Snowball Quadrature

## Summary
Implement a single quadrature recursion that propagates two state value arrays on a shared log-price grid:
- `V_in(S, t)`: value conditional on knock-in already occurred.
- `V_out(S, t)`: value conditional on knock-in not yet occurred.

The recursion uses the Huang-Luo discrete quadrature method (FFT-based convolution + Simpson integration) for diffusion, and applies KO/KI state transitions at observation dates. Continuous KI is handled via Brownian-bridge transition probabilities between observation dates.

## Quadrature Math Extraction
Create a reusable utility (e.g., `QuadratureMath` or `QuadGrid`) that owns:
- Log-price grid construction and spacing.
- Simpson integration weight preparation.
- FFT-based convolution helper.
- Final interpolation/extraction at spot.

`QuadratureCore` should delegate math operations to this utility so other engines can reuse the same primitives without duplicating numerical code.

## Snowball Recursion
### Terminal Condition (t = T)
- `V_in(S, T) = product.get_maturity_payoff_v1(S, pricing_env)`
- `V_out(S, T) = product.get_maturity_payoff_v0(S, pricing_env)`

### Backward Step (t_{m-1} <- t_m)
1. **Diffusion**: Independently convolve `V_in` and `V_out` using quadrature recursion to obtain values at t_{m-1}.
2. **Continuous KI (Brownian-bridge)**: If KI is continuous, mix the two states using the bridge probability `p_hit(S_{m-1}, S_m)`:
   - `V_out` uses expectation of `p_survive * V_out + (1 - p_survive) * V_in`.
   - `p_survive` is computed per (S_{m-1}, S_m) using log-price bridge for the KI barrier.
   - This step is applied on every time interval (not only observation dates).
3. **Discrete Observations** (when t_{m-1} is an observation time):
   - **KO**: If `S` crosses KO barrier, both states jump to KO payoff (coupon + principal) with settlement timing handled by coupon pay type.
   - **KI (discrete)**: If KI barrier is breached, set `V_out = V_in` on the KI region only at KI observation times.

### Result
Return `V_out(S0, t0)` as the snowball price (assuming not yet knocked in).

## Barrier Direction Handling
- Standard snowball: KO up, KI down.
- Reverse snowball: KO down, KI up.

The recursion applies region checks based on barrier direction; KO takes precedence if KO and KI occur at the same observation time.

## Brownian-Bridge Formula (Continuous KI)
For a down barrier `B` and log-price `X = ln S` with endpoints `X_{m-1}, X_m`, the bridge hit probability is:

`p_hit = exp(-2 * (X_{m-1} - ln B) * (X_m - ln B) / (sigma^2 * dt))`

This is applied to transition probability from `V_out` to `V_in` during the diffusion step.

## Limitations (Phase 1)
- Airbag and call-style rebate features are excluded for now (future extension).
- `disable_ko_after_ki` is not supported in this phase.
