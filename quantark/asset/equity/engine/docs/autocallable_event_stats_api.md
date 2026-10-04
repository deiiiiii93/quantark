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
- `expected_discounted_maturity_cashflow`: `E[ DF(T_pay) * maturity_payoff * 1_{no KO} ]`
- `pv`: engine PV estimate for the same product/env
- `reconciliation_error`: `pv − Σ expected_discounted_cashflows`, i.e.
  `pv − Σ ed_ko_cf − Σ coupon cash − ed_maturity_cf` (every per-date coupon column or
  payment bucket, pruned or not)

### The reconciliation identity (PDE, 0.9.0)
The 1D PDE statistics march in the value's own groups: the value's boundary rows
allocated per column, its θ rule, and the adapter's event stages (knock-out, knock-in,
coupon, the Phoenix coupon-barrier smoothing) applied to every column. The maturity leg
is a **marched column** `M` (terminal: the value's terminal payoffs times the terminal
delay factor; zeroed where a knock-out absorbs; copied at a knock-in), not the residual
`pv − ΣKO cash`. So on every native 1D route `reconciliation_error` is round-off
(|·| ≤ 1e-10 in the suite). On the 2D surrogate (below) it is the model difference.
The cashflow ledger (`payment_times`, `expected_discounted_cashflows`) takes `M` as its
terminal row; its sum is `pv` up to the residual.

### Phoenix coupon cash
- **Memory, INSTANT pay:** `expected_discounted_coupon_cashflow[k]` is the expected
  discounted cash paid at observation `k`, including the released arrears, marched per
  missed-coupon state. A knock-out pays the memory arrears whatever the spot, plus the
  current coupon when its barrier is met (the release rule, as QUAD V2).
- **EXPIRY pay:** the earned coupon is paid at whichever termination ends the note, so
  the cash is reported per earning date *and* bucketed by payment event in the ledger
  (each knock-out's settlement, the terminal payment time).
  `coupon_payment_is_path_dependent = True`; an `EventDistribution` built from it marks
  the COUPON stream path-dependent and asking it for one fixed coupon payment time
  raises. No cash leg reads the buckets in 0.9.0.
- The coupon amounts are the contractual periods (`get_coupon_period_year_fractions(env)`:
  the original schedule index and the coupon day count, never truncated at the
  valuation date).

### KO-reset (native PDE)
`KOResetSnowballPDESolver.calculate_event_stats` returns a `KOResetEventStats` marched
on the PDE (no QUAD delegation). `ko_*` and `pre_ko_*` are the pre-knock-in schedule;
`post_ko_*` the post-knock-in schedule (fired on the knocked-in surface only, with
`disable_ko_after_ki` honoured). The maturity cash has **two buckets**:
`expected_discounted_pre_maturity_cashflow` (matures on the pre schedule, not knocked
in) and `…_post_maturity_cashflow` (matures knocked in), each with its probability,
determination time and payment time (`pre_maturity_*`, `post_maturity_*`), and
`expected_discounted_maturity_cashflow` is their sum. QUAD V2 fills the same fields;
legacy QUAD and Monte Carlo leave them `None`.
`EventDistribution.from_autocallable_stats` converts a `KOResetEventStats` with both
schedules as knock-out events, the two maturity outcomes at their own determination
times (`EventDistribution.determination_times`), and a survival that subtracts matured
mass. It raises `NotImplementedError` on `post_ko_times_are_offsets` (Monte Carlo under
`PostKOScheduleMode.REBASED`): a distribution needs absolute times.

### Valuation-date rules
One rule set for `calculate_event_stats`, `price_with_events` and a prepared session's
`EVENT_STATS`, decided by the product's `decide_observations_at_valuation` with the
carried knock-in state:
- **Knocked out at valuation:** knock-out probability 1 at the deciding observation (a
  pre- or post-schedule record on KO-reset), its cash `payoff · DF(0→settlement)`,
  survival 0, maturity leg 0, `reconciliation_error` 0; `price_with_events` returns that
  knock-out distribution. A Phoenix's coupon and arrears paid with it are coupon cash.
  A carried knock-in with `disable_ko_after_ki` does not knock out at valuation; a
  carried-knock-in KO-reset decides on its post schedule.
- **Knocked in by the lifecycle flag:** `ki_ever_probability = 1` with `ki_times = [0]`,
  `ki_event_probability = [1]`; `ki_survive_knocked_in_probability` (and the legacy
  `ki_probability`) is the marched value (the no-knock-out survival), not 1. With the
  knock-in streams pruned (`streams` without KI) it is reported 0.0, like every pruned
  knock-in field.
- A spot below a continuous knock-in barrier, or a discrete knock-in observed at
  valuation, takes the marched path.
- **Expired:** `calculate_event_stats` returns `None`; `price_with_events` the trivial
  distribution at maturity 0.

### Knock-in field definitions
- `ki_survive_knocked_in_probability` = P(knocked in **and** survives to maturity
  without knocking out): the note settles knocked in. It is never P(not knocked in).
- `ki_ever_probability` = P(the barrier is breached at any time in the contract's
  monitoring window). MC, QUAD and PDE all count a breach *after* a knock-out, which a
  paying note never experiences; this is a definition, consistent across engines.
- The legacy `ki_probability` is "settles knocked in" on PDE and QUAD and "ever
  breached" on MC; prefer the two explicit fields.

### Provenance and 2D solvers
Every PDE result carries `provenance = StatisticsProvenance(statistics_model, price_model)`
(`EventDistribution.provenance` copies it). It names the implementation, not an accuracy.
The 2D Heston / SLV solvers have no native statistics: by default
(`statistics="refuse"`) `calculate_event_stats` and `price_with_events` raise
`StatisticsRefused`. With `statistics="flat_vol_surrogate"` they return the flat-vol 1D
statistics at the environment's vol with the 2D price as `pv`, provenance
`("flat_vol_1d", "heston_adi_2d" | "slv_adi_2d")`, and `reconciliation_error` the model
difference.

## Implementations
- `SnowballMCEngine.calculate_event_stats()` provides a Monte Carlo implementation today.
- `SnowballQuadEngine.calculate_event_stats()` provides a quadrature implementation by propagating stacked indicator surfaces.
- `SnowballPDESolver.calculate_event_stats()` provides a native PDE implementation by propagating stacked indicator surfaces
  through the PDE time-stepping and applying KO/KI jumps at observation times, in the value's own groups (above).
- `PhoenixPDESolver` and `KOResetSnowballPDESolver` do the same for their families, and so do the local-vol 1D variants.
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
