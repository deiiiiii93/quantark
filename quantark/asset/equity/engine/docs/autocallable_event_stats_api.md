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
