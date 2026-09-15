# Autocallable Event Stats API (Engine-level, Optional)

## Goal
Provide a standard engine hook for per-observation event probabilities and expected discounted cashflows for autocallable products.

This enables risk reporting to:
- request event stats from QUAD/PDE engines (fast, deterministic) when implemented
- fall back to Monte Carlo analysis when not implemented

## API Surface
- `asset/equity/engine/event_stats.py`: `AutocallableEventStats` dataclass
- `asset/equity/engine/base_engine.py`: `BaseEngine.calculate_event_stats(...) -> Optional[AutocallableEventStats]`

## Semantics (Snowball-first)
The returned stats represent:
- `ko_times[i]`: KO observation time (year fractions)
- `ko_probability[i]`: `P(KO occurs at observation i)`
- `survival_probability[i]`: `P(no KO up to and including observation i)`
- `expected_discounted_ko_cashflow[i]`: `E[ DF(settlement_i) * KO_payoff_i * 1_{KO at i} ]`
- `expected_discounted_maturity_cashflow`: `E[ DF(T) * maturity_payoff * 1_{no KO} ]`
- `pv`: engine PV estimate for the same product/env
- `reconciliation_error`: `pv - (sum(ed_ko_cf) + ed_maturity_cf)`

## Implementations
- `SnowballMCEngine.calculate_event_stats()` provides a Monte Carlo implementation today.
- `SnowballQuadEngine.calculate_event_stats()` provides a quadrature implementation by propagating stacked indicator surfaces.
- `SnowballPDESolver.calculate_event_stats()` provides a native PDE implementation by propagating stacked indicator surfaces
  through the PDE time-stepping and applying KO/KI jumps at observation times.
- `PhoenixQuadEngineV2.calculate_event_stats()` returns `PhoenixEventStats`
  through the native V2 state/event recursion, including memory and deferred coupons.

## Phoenix QUAD V2

`coupon_probability[i]` is the unconditional probability of being alive just
before observation `ko_times[i]` and meeting the coupon barrier, including a
simultaneous KO. `expected_discounted_coupon_cashflow[i]` attributes coupon PV
to the observation that earns or releases it, including caught-up memory.
The existing V2 KO payoff also releases missed memory when the current coupon
barrier is not met; coupon cashflow and trigger probability are distinct.

Deferred (`EXPIRY`) coupons settle at actual KO or maturity, using the payment
curve at that settlement. The additive `determination_times`, `payment_times`,
`expected_undiscounted_cashflows` and `expected_discounted_cashflows` ledger
groups these amounts by payment event. It includes pending receivables and
deferred coupons in terminal payment rows. Its discounted sum reconciles to
`pv`; the coupon diagnostic array must not be added again to that ledger.

Deferred stats set `coupon_payment_is_path_dependent=True`. Conversion to
`EventDistribution` preserves coupon-trigger probabilities and marks the
coupon payment event as path-dependent. Asking that distribution for one
fixed coupon payment time raises; use the expected-cashflow ledger instead.
The direct API, `price_with_events()` and serial framework `EVENT_STATS` all
preserve lifecycle state and the specified before/after event phase.
