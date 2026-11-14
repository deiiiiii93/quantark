# Event Statistics API (autocallables and DCN)

One result type, `EventStatistics`, describes how a Snowball, Phoenix, KO-reset Snowball or
DCN ends and what it pays. Every engine that prices these products fills it with one
definition per field: Monte Carlo, QUAD V1, QUAD V2, PDE 1D (flat and local vol), Heston/SLV
ADI and the DCN PDE engines. Cash legs, the risk report, replay and hedging read it.

Since 0.11.0. It replaces `AutocallableEventStats`, `PhoenixEventStats`, `KOResetEventStats`,
`DCNEventStats` and `RiskNeutralSnowballEventStats`, with no aliases. The field-by-field map from
the old types is in [`docs/event-statistics-migration.md`](../../../../../docs/event-statistics-migration.md).

## API surface

| Name | Where |
|---|---|
| `EventStatistics` and its tables (`ObservationTable`, `CouponTable`, `OutcomeRow`, `KnockInStatistics`, `KnockInTable`, `PostKIOffsetTable`, `TerminationDistribution`, `ReceivablesTable`, `CashLedger`) | `quantark/asset/equity/engine/event_stats.py` |
| `StatisticsProvenance`, `StatisticsRefused`, `StatisticUnavailable` | same module |
| `build_cash_ledger`, `build_rebased_cash_ledger` | same module (for producers) |
| `engine.calculate_event_stats(product, pricing_env, *, streams=None, ...) -> Optional[EventStatistics]` | every engine above |
| `engine.price_with_events(...) -> PricingResult(npv, event_distribution)` | every engine above |
| `EventDistribution.from_statistics(stats)` | `quantark/cashleg/event_distribution.py`, the one converter |

Every type is a frozen dataclass of plain data: 1-D NumPy arrays (copied and made read-only),
floats, tuples of `datetime` (or `None` on a numeric-clock contract). No pandas.

## Outcomes partition the paths

Every path ends in exactly one outcome:

| Outcome | Definition | Where |
|---|---|---|
| Knocked out | KO at some observation, whether or not it knocked in first | one row of `observations` |
| Matured, never knocked in | no KO and no KI | `maturity` |
| Matured knocked in | KI and no KO | `ki_settlement` |

Two rules decide which outcome a path has:

- **KI is never triggered after a KO.** A path that knocks out stops KI monitoring at that
  instant. A barrier crossed after the KO does not count anywhere.
- **KO wins a same-instant tie with a discrete KI.** A discrete KI observation and a KO
  observation at the same instant, both breached, is a KO. A continuous KI touched inside a
  step before the closing KO observation is a KI first. On KO-reset, a KI and a post-schedule
  KO at one instant count as the KO.

The two identities below hold on every engine. They are **reported**, never forced:

- `probability_residual = 1 − (Σ observations.probability + maturity.probability + ki_settlement.probability)`
- `reconciliation_error = pv − (Σ observations.discounted_cash + Σ coupons.discounted_cash + maturity.discounted_cash + ki_settlement.discounted_cash + Σ receivables.discounted_cash)`

Both are computed by the type in `__post_init__`; a producer cannot pass them in. On a REBASED
KO-reset both sums run over `termination` (plus receivables for the cash). On a terminated note
`probability_residual` is 0 and not checked. The type also checks that `Σ ledger.discounted_cash`
equals the same cash sum (relative tolerance `Tolerance.RELATIVE`) and that
`knock_in.probability_no_ko == ki_settlement.probability` exactly.

Validation never rescales or clips a number. It raises `ValidationError` on misaligned arrays,
non-finite values, negative probabilities (beyond `Tolerance.PROBABILITY`), non-monotone survival,
unordered rows, a payment before its determination, or a missing `provenance`.

How large the residuals are depends on the engine: round-off on MC and QUAD V2, round-off or
grid-dependent and shrinking under refinement on the PDE and QUAD V1 (see
[Per-engine notes](#per-engine-notes)).

## `EventStatistics`

| Field | Meaning |
|---|---|
| `product_family` | `"snowball"`, `"phoenix"`, `"ko_reset"` or `"dcn"` |
| `pv` | `engine.price(...)` for the same product and environment, bitwise. It is the price path's value, not a separately marched one; a marching difference shows in `reconciliation_error`. |
| `observations` | KO table on the shared KO/coupon observation grid |
| `coupons` | coupon table on the same grid; `None` exactly for snowball and ko_reset |
| `maturity` | matured, never KO, never KI |
| `ki_settlement` | matured knocked in |
| `knock_in` | KI path statistics |
| `ki_table` | per-date first-KI table; `None` only when pruned or unavailable (`unavailable` says why) |
| `post_ki_offsets`, `termination` | REBASED KO-reset only (both or neither) |
| `receivables` | cash decided at or before valuation and paid after it (possibly empty) |
| `ledger` | payment-date view of all cash |
| `expected_lifetime` | E[time of the observation that decides the note]: the KO row's time if it knocks out, otherwise the final observation. A KO at valuation gives 0. On the engine's pricing axis (DCN: its ACT/365F axis). |
| `total_discounted_cash`, `total_cash_to_pay` | the cash sums of the identity (computed) |
| `reconciliation_error`, `probability_residual` | the identities (computed) |
| `provenance` | `StatisticsProvenance(statistics_model, price_model)`; required |
| `unavailable` | field path → reason; required for each of `ki_table`, `knock_in.probability`, `knock_in.expected_time`, `knock_in.expected_loss`, `ledger.cash_to_pay` and (when present) `coupons.probability` that is `None`. Other `None` values (dates, MC `standard_errors` on a grid engine) need no entry |
| `standard_errors` | MC only: field path → standard error |

Properties: `rebased` (REBASED KO-reset), `terminated` (receivables only).

**Sign.** Cash carries the sign of `price()`. For autocallables that is the per-unit buyer
view; for a DCN it includes `direction_sign`, so a seller DCN reports negative cash with the
identities intact. Probabilities, times, `expected_lifetime` and `knock_in.expected_loss` are
unsigned. A position's signed quantity applies on top, as for the price.

### Observation table (`ObservationTable`)

One row per observation in the union of the KO and coupon observations, sorted by time.

| Column | Meaning |
|---|---|
| `times`, `dates` | when the row is decided |
| `payment_times`, `payment_dates` | when the KO cash settles. Under EXPIRY coupon pay: with the note's end -- a KO-reset `pre` row with the `maturity` row (the product's explicit `settlement_date` when booked, else the first schedule's maturity plus the terminal delay: a first-schedule KO ends a never-knocked-in note; `settlement_support.first_schedule_settlement`), a `post` row with the product's terminal. |
| `is_ko_obs`, `is_coupon_obs` | which kinds of observation the row is. A coupon-only row (DCN `ko_lock_months`, a Phoenix observation without a KO barrier) has KO probability and cash exactly 0. |
| `schedule` | `"pre"` / `"post"` per row on KO-reset, else `None`. Both schedules sit in one table; a time can repeat once per tag, `pre` first. |
| `probability` | P(KO at this row) |
| `survival` | P(alive after this row). KO-reset ABSOLUTE: also drops the never-knocked-in maturity mass at its determination time when that comes before the last row. |
| `discounted_cash` | E[DF(payment) × KO redemption × 1{KO here}] |
| `cash_to_pay` | E[KO redemption × 1{KO here}], undiscounted: the nominal amount to fund |
| `payment_is_path_dependent` | the row has no single payment time; its `payment_times` entry is a placeholder and the ledger holds the actual payments |

KO redemption **excludes coupons**. For Snowball and KO-reset it is principal plus the KO rate
times the accrued period. Phoenix's unconditional `ko_rate` and DCN's `ko_coupon_rate` amount
are KO redemption. Barrier-contingent coupons (Phoenix `coupon_rate`, released memory arrears,
DCN fixed coupons) are coupon cash, also when paid with a KO.

### Coupon table (`CouponTable`)

Same rows as `observations`. Rows that are not coupon observations carry zero probability and cash.

| Column | Meaning |
|---|---|
| `probability` | Phoenix: P(alive just before the row and the coupon barrier met), a simultaneous KO included. DCN: P(the row's fixed coupon is paid); a DCN pays no fixed coupon on a KO date or once knocked in. `None` when pruned. |
| `survival` | P(alive going into the row) |
| `payment_times`, `payment_dates` | the coupon's settlement (DCN: the coupon pay dates, distinct from the KO pay dates); a placeholder on path-dependent rows |
| `discounted_cash` | E[DF × coupon paid × 1{paid here}], the current coupon plus any memory arrears released at the row |
| `cash_to_pay` | the same, undiscounted, computed directly |
| `payment_is_path_dependent` | True on every row of an EXPIRY-paid coupon |

Memory coupons: arrears released at a row are that row's cash, so cash ≠ probability × coupon
there; the probability stays "barrier met at this row". A KO releases the arrears whatever the
current coupon barrier. `initial_coupon_arrears` owed at the pricing date are released at the
first remaining coupon observation by the same rule.

EXPIRY-paid coupons are paid at whichever KO or maturity ends the note. The row is the earning
observation; its payment is path-dependent; the ledger buckets the cash at the actual payment
events (each KO row's settlement and the terminal payment). Never add `coupons.discounted_cash`
to the ledger: the ledger already contains it.

### Maturity row and KI settlement row (`OutcomeRow`)

Columns: `time`, `payment_time`, `date`, `payment_date`, `probability`, `discounted_cash`,
`cash_to_pay`.

- `maturity`: matured, never knocked out and never knocked in. Its cash is the never-knocked-in
  maturity redemption, excluding anything the coupon table counts. DCN: cash 0.
- `ki_settlement`: matured knocked in. Its cash is the knocked-in maturity redemption. DCN: the
  loss leg.

Both are computed directly by every engine (a marched or integrated column, or a path sum), never
as `1 − ΣKO − …`. KO-reset ABSOLUTE: `maturity` is at the first schedule's end with its own
settlement; `ki_settlement` at the post-schedule maturity. REBASED: `ki_settlement` is the totals
row placed at the latest candidate knocked-in settlement; the actual dates are the
`ki_settlement` rows of `termination`.

### Knock-in path statistics (`KnockInStatistics`)

They describe the path, not how the note ends, so they may overlap KO outcomes.

| Field | Meaning |
|---|---|
| `probability` | P(KI before any KO). `None` when pruned. |
| `probability_no_ko` | P(KI and no KO); equal to `ki_settlement.probability` exactly. Never pruned. |
| `expected_time` | E[first KI time \| KI before any KO]. Continuous KI: the first crossing; discrete KI: the first breaching KI observation. `None` when pruned, and `None` with an `"undefined: ..."` reason when P(KI) = 0. |
| `expected_loss` | E[DF(maturity payment) × (principal − knocked-in redemption) × 1{matured knocked in}] ≥ 0 |

Expected loss uses `principal` = 0 when the principal is excluded, so it is −downside either way.
The downside already includes participation, the airbag, the protection floor and the
annualized-KI tenor, so a fully protected note reports 0. Coupons are not netted. KO-reset uses
its knocked-in schedule's downside; DCN the loss leg's magnitude.

### Per-date knock-in table (`KnockInTable`)

| Column | Meaning |
|---|---|
| `times`, `dates` | the KI observation |
| `payment_times`, `payment_dates` | equal to `times` / `dates`. A cash leg paid at KI adds its own settlement offset (`FixedPayoffLeg.payment_offset_days`). |
| `probability` | P(first KI at this row, before any KO) |
| `survival` | P(not knocked in and not knocked out after this row). A KO at the row's own instant is out (it wins the tie). |
| `interpolated` | True where the row boundary is not a time of the engine's grid |
| `interpolation_scheme` | the scheme name when any row is interpolated, else `None` |

- Rows follow the product's own KI schedule: a monthly KI gets monthly rows, a daily KI daily
  rows. Discrete rows are exact on every engine.
- Continuous KI: one row per trading day of the monitoring window, each holding the first-crossing
  mass within that day; the last row is the window end. A dated contract in an environment with a
  calendar gets business days with dates; a numeric-clock contract gets steps of
  `1 / bus_days_in_year` with no dates. This is a labelled approximation that disappears as the
  time grid is refined. The scheme and the `expected_time` convention per engine:

  | Engine | `interpolation_scheme` | continuous `expected_time` |
  |---|---|---|
  | MC | `"step_overlap"`: each step's crossing mass spread over the days it overlaps, in proportion to time | at the MC step end |
  | PDE 1D, ADI | `"step_overlap"` | at step midpoints |
  | QUAD V2, QUAD V1 | `"interval_first_passage"`: each interval's crossing mass split by the drifted Brownian first-passage law the price uses | at day-row midpoints |

- Identities: Σ `probability` = `knock_in.probability`, and Σ (`times` × `probability`) /
  `knock_in.probability` = `knock_in.expected_time`. Exact (round-off) for discrete KI; within one
  trading day for continuous KI on grid engines, one trading day plus one MC step on MC.
- Already knocked in (carried, or a KI decided at valuation): one row at time 0 with probability
  1. No reachable KI (no barrier, or knocked out at valuation without a carried KI): an empty table.

### REBASED KO-reset (`PostKIOffsetTable`, `TerminationDistribution`)

Under `PostKOScheduleMode.REBASED` the post-KI schedule starts at the knock-in, so its dates are
path-dependent. Supported on Monte Carlo only (the other KO-reset engines refuse REBASED, as
they refuse to price it).

- `observations` holds the `pre` rows only (absolute times, absolute survival).
- `post_ki_offsets`: one `"ko_post"` row per offset from the knock-in and one `"ki_settlement"`
  row; columns `offsets`, `kind`, `probability`, `discounted_cash`, `cash_to_pay`; no survival.
  For reading.
- `termination`: one row per (absolute determination time, event) for the events `"ko_pre"`,
  `"ko_post"`, `"maturity"`, `"ki_settlement"`, on the full deterministic candidate date set (every
  `pre` row, every reachable KI date plus each offset, calendar-rolled, every knocked-in end, and
  the maturity). Zero rows are kept, so the axis depends only on the contract and the valuation
  date, never on the seed or a bump. Columns: `times`, `payment_times`, `dates`, `payment_dates`,
  `event`, `probability`, `discounted_cash`, `cash_to_pay`, plus `survival_times` / `survival`.
  Its totals by event equal the offset table's totals. The residuals and
  `EventDistribution.from_statistics` read it.

### Receivables (`ReceivablesTable`)

Cash decided at or before valuation and paid after it: QUAD V2's pending receivables, a DCN's
pending ledger flows, a matured but unsettled note. Columns: `source` (`"ko"`, `"coupon"`,
`"maturity"`, `"ki_settlement"`, `"dcn_loss"`), `decision_times` (≤ 0), `decision_dates`,
`payment_times`, `payment_dates`, `discounted_cash`, `cash_to_pay`.

Receivables are inside the cash identity and outside the probability identity. They are not
events: `EventDistribution` never reads them. A KO decided at valuation (t = 0) is still a KO row
with probability 1, not a receivable.

### Cash ledger (`CashLedger`)

One row per (determination, payment) pair: `determination_times`, `payment_times`,
`determination_dates`, `payment_dates` (kept only when every row has one), `discounted_cash`,
`cash_to_pay`. Row order: KO rows and coupon rows with a deterministic payment time, then the
path-dependent payments at their actual events, then the maturity row, the KI settlement row and
the receivables. REBASED: the termination rows, then the receivables.

It is the only place where EXPIRY coupons and other path-dependent payments appear at their
actual payment events. Its discounted sum equals the tables' discounted cash.

**Cash to pay is computed directly.** On a row with a deterministic payment time,
`cash_to_pay == discounted_cash / DF(payment)`. On a path-dependent row (EXPIRY coupons, released
memory arrears, REBASED rows) E[C] ≠ E[DF(τ)·C] / DF(t) for any fixed t, so grid engines march a
separate undiscounted column and MC sums undiscounted cash per path.

## Valuation-date and terminated-state rules

One rule set for `calculate_event_stats`, `price_with_events` and a prepared session's
`EVENT_STATS`, decided by the product's `decide_observations_at_valuation` with the carried
knock-in state:

- **Knocked out at valuation:** the deciding observation is a KO row with probability 1, its cash
  `payoff · DF(0 → settlement)`; survival 0; maturity and KI settlement rows 0. A Phoenix's coupon
  and arrears paid with it are coupon cash. P(KI) = 1 if the note was already knocked in (one KI
  row at 0), else 0 with an empty KI table, `expected_time` undefined and expected loss 0. A
  carried KI with `disable_ko_after_ki` does not knock out at valuation; a carried-KI KO-reset
  decides on its post schedule.
- **Already knocked in (carried state):** P(KI) = 1, `expected_time` = 0, one KI row at 0.
- **Spot through a continuous KI barrier, or a discrete KI observed at valuation:** a KI at time 0
  (same as carried).
- **No KI barrier** (`has_ki_barrier == False`): P(KI) = 0, KI settlement probability and cash 0
  (computed zeros), expected loss 0, empty KI table, `expected_time` undefined.
- **Expired, nothing unpaid:** `calculate_event_stats` returns `None` on every engine.
  `price_with_events` returns the trivial maturity-only distribution, or
  `EventDistribution.ended()` on the engines that take a lifecycle state.
- **Terminated with unpaid cash** (engines accepting `lifecycle_state`: QUAD V2 and the DCN
  engines): statistics with only `receivables` (and the ledger) filled; every termination
  probability is 0 and `probability_residual` is 0. **Terminated and fully settled:** `None`.
  `price_with_events` returns `EventDistribution.ended()`, which every event-dependent leg refuses
  (`ensure_live`): a leg riding an ended note settles with it.

## Unavailable values and `require`

A number an engine cannot produce is `None`, and `unavailable[path]` says why. There are no
stand-in zeros. Keys are dotted field paths (`"knock_in.expected_time"`, `"ki_table"`,
`"coupons.probability"`) or `"standard_errors.<path>"`. Reasons:

- `"not requested"`: pruned by `streams=` (below).
- `"undefined: ..."`: mathematically undefined for this trade, e.g. `knock_in.expected_time`
  when P(KI) = 0 (`"undefined: P(knock in) = 0"`), or `standard_errors` on a single-batch MC run
  (`"undefined: single batch"`). Not an engine gap.
- Any other reason: the engine cannot produce the value on this route (listed per engine below).

A full request (`streams=None`) leaves `unavailable` empty on every engine's default
configuration, except for `"undefined:"` entries.

Read a value through `stats.require(path)` when it may be missing: it returns the value or raises
`StatisticUnavailable` (a `ValidationError`) with the reason. `stats.flag(path)` returns the
reason covering a path (the path itself or its component), or `None`. Both raise
`ValidationError` on a path that is not a field of the type.

## Requesting part of the statistics (`streams=`)

`calculate_event_stats(..., streams=...)` and `price_with_events(..., streams=...)` take a set of
`EventType` values (`quantark.cashleg.event_distribution`). Greek bumps and theta shifts use it
for speed.

- **Never pruned:** the KO table (probability, survival, cash), the maturity and KI settlement
  rows, `knock_in.probability_no_ko`, receivables, the ledger's discounted cash, the totals and
  both residuals. Coupon cash stays in the ledger when the coupon table is pruned.
- **Prunable:** `coupons.probability` (without `EventType.COUPON`); `knock_in.probability`,
  `knock_in.expected_time` and `ki_table` (without `EventType.KI`). Pruned values are `None` with
  reason `"not requested"`. A pruned request never runs the KI forward pass on PDE 1D, ADI or QUAD; the DCN PDE engines are the exception and always run their single forward pass.
- The type also allows `knock_in.expected_loss` and `ledger.cash_to_pay` to be pruned; no engine
  in this release prunes them.
- MC engines and the DCN PDE engines accept `streams` and prune nothing: every number comes from
  the same paths, or (DCN PDE) from one forward pass that also yields the outcomes.

## Provenance

`provenance = StatisticsProvenance(statistics_model, price_model)` is required on every result
and copied to `EventDistribution.provenance`. It names the implementation, not an accuracy. On
every native route the two names are equal (`provenance.surrogate` is False): the MC and QUAD V2
engine class names, `"quad_v1"`, `"flat_vol_1d"` / `"local_vol_1d"` (PDE 1D), `"heston_adi_2d"` /
`"slv_adi_2d"` (ADI). They differ only on an explicit approximation:
`("quad_v1_forward_density", "quad_v1")` and the ADI `statistics="flat_vol_surrogate"` route
(`"flat_vol_1d"` beside the 2D price model). The path analyzer reports
`("path_analyzer_mc", "SnowballMCEngine")`; fold structures `("fold_structure", "fold_structure")`.

## `EventDistribution` and cash legs

`EventDistribution.from_statistics(stats)` is the only converter. Each stream has its own axis
(`stream_times`, `stream_dates`, `payment_times`, `payment_dates` keyed by `EventType`):

- **KO** and **COUPON** share the observation grid. COUPON is absent when `coupons.probability` is
  `None`, and path-dependent where coupon rows are.
- **KI** comes from `ki_table` on its own axis, paid at the KI observation. An empty table gives an
  empty stream (an AT_KI leg is worth 0); `ki_table is None` gives no stream (an AT_KI leg raises).
- **MATURITY_NO_KO** = `maturity.probability + probability_residual`: the residual is absorbed here
  and carried visibly as `EventDistribution.probability_residual`.
- **MATURITY_WITH_KI** = `ki_settlement`, at its own determination and payment time.
- **REBASED:** KO = the `ko_pre` and `ko_post` termination rows; MATURITY_WITH_KI is an array stream
  on the same axis; survival is the termination distribution's absolute curve; `rebased=True`.
  Legs value each event at its own time; an `AutocallableCashLeg` on a REBASED note must list every
  candidate event time that carries mass (else `ValidationError`). Folding a REBASED note is refused
  (`CapabilityError`).
- **Receivables** are never events. A terminated note converts to `EventDistribution.ended()`.

`price_with_events(...).npv == price()` bitwise on every engine.

## Per-engine notes

### Monte Carlo (Snowball, Phoenix, KO-reset, DCN, and every vol-model subclass)

- Statistics are recorded on `price()`'s own paths: `calculate_event_stats` runs `price()` under a
  recorder, and `pv` is the value it returned. Every outcome is a direct path sum, so both
  residuals are round-off.
- `standard_errors` from independent batch means (ratio estimator) for every path-average field,
  `knock_in.expected_time` by the delta method. Undefined (`None` with a reason) on a single batch,
  on deterministic QMC Dask batches (`"undefined: deterministic QMC batches are not independent"`),
  and for `expected_time` with fewer than two KI paths. Grid engines report no standard errors.
- Exact-affine conditional routes (Heston QE `rqmc_affine_spot_factor`, SLV-QE control-only) report
  exact conditional outcome probabilities on the price batches, KI before KO included.
- Heston-SLV conditional control (`rqmc_heston_conditional_control`): each outcome uses the price's
  control-corrected estimator, so the residuals are round-off. A control-corrected probability
  below the bound raises `PricingError` (never clipped); seen only below about 256 paths per batch.
- Local-vol MC `estimator="one_step_survival"` (a Greek estimator) has no per-path outcomes and
  still refuses statistics.
- An expired note (T ≤ 0) returns `None`.
- KO-reset: payment dates are `None` (the settlement resolver works on times); determination dates
  are set.
- DCN: `calculate_event_stats(..., lifecycle_state=None)`, `price_detailed(...).event_stats` and
  `price_with_events` all return the same statistics. `price_with_events(emit_distribution=False)`
  returns the maturity-only distribution.

### PDE 1D (flat and local vol: Snowball, Phoenix, KO-reset)

- Every outcome is a marched backward column in the value's own groups (boundary rows, θ rule,
  event stages). Edge rows are labelled by the price's own edge-row rule: a far-edge row is a KI
  settlement only where the price row there settles knocked in.
- P(KI), `expected_time` and the KI table come from one forward pass, the exact discrete adjoint
  of the backward step; it reproduces the backward termination probabilities to round-off. A
  pruned request skips it.
- `probability_residual` is second order in the time step (≤ about 1e-8 on default grids, shrinking
  under refinement); `reconciliation_error` is round-off.
- A route the pass cannot transpose reports the KI path fields unavailable with the reason.

### Heston / SLV ADI (`statistics="native"`, the default)

- Columns use the native ADI operator, event stages, boundary allocation and readout of the price.
  `pv` uses the readout `price()` uses.
- The forward pass is the hand-derived transpose of the ADI step. The scalar backend, sparse and
  semi-Lagrangian variance transport (non-default opt-ins) report the KI path fields unavailable
  with a reason. Coarse meshes with an up barrier can produce a negative first-KI row below
  `−Tolerance.PROBABILITY`; the KI fields are then refused with a "refine the grid" reason.
  Default meshes are within tolerance.
- `statistics="refuse"` raises `StatisticsRefused`. `statistics="flat_vol_surrogate"` returns 1D
  flat-vol statistics beside the 2D price, labelled by `provenance`; its `reconciliation_error` is
  the 2D − 1D price gap.
- Price-only calls allocate no statistics. Refine `n_x`, `n_v`, `n_t` for accuracy.
- Absolute KO-reset classes: `HestonKOResetSnowballPDESolver`, `HestonSLVKOResetSnowballPDESolver`
  (and `LocalVolKOResetSnowballPDESolver` on the 1D core).

### DCN PDE (flat, local vol, Heston DCN ADI)

- Every outcome and the KI table come from one forward pass (about one price solve); `streams` is
  accepted and nothing is pruned.
- `probability_residual` on default grids is below 1e-7 (Heston below 1e-6), shrinking under time
  refinement.
- A `PricingSession` `EVENT_STATS` request is still refused for `DCNPDEEngine` and
  `LocalVolDCNPDEEngine` (the adapter does not advertise it); call `calculate_event_stats` directly.

### DCN time axis (all DCN engines)

A DCN prices on its own ACT/365F grid times and discounts them on the environment curve. Its
statistics use the same axis, so the identities and the folded route agree with `price()`; dates
are exact. Under a `BUSINESS_DAYS` or `ACT_360` environment those times are not the environment's
axis. Fixing that moves DCN prices and is left to a separate release. Receivable rows carry the
decision time on the DCN axis and the payment time `price()` discounted the flow at.

### QUAD V2

- Outcomes are backward channels of the V2 recursion; the forward pass (the hand transpose of each
  operator) fills only the KI path statistics. `pv` is the price channel, bitwise equal to
  `price()`.
- Continuous `expected_time` sits about half a trading day below MC (day-midpoint convention vs MC
  step bucketing), within the one-trading-day tolerance of the KI table identity.
- Non-default routes (zero-variance intervals, the pointwise continuous backend, |log weight| ≥ 500)
  report the KI path fields unavailable with a reason; outcomes and `pv` are produced.
- Accepts `lifecycle_state` (receivables, terminated states, see above).
  `calculate_event_stats_at_spots` reads the same statistics at several spots.
- Memory Phoenix `initial_coupon_arrears` are paid (since 0.11.0). With a lifecycle tracker count
  as well, the count carries the debt; a non-zero product amount beside it must state the same debt,
  else `ValidationError`.

### QUAD V1

- The maturity outcomes are direct columns; the KI path statistics come from the exact transpose of
  V1's own backward step. Ties follow the backward event: KO wins.
- The BGK continuous-KI approximation reports the KI path fields unavailable.
- Phoenix EXPIRY: `reconciliation_error` is about 7e-9 × pv at 401 nodes (1e-9 at 1001, 7e-11 at
  4001). The price knocks out the termination surface with nodal weights while the statistics use
  the joint projection; this is a pre-existing price-side difference, not hidden.
- Memory Phoenix with more than 50 observations is refused, as for pricing.
- Opt-in `event_stats_mode="forward_density"` keeps its own finite-grid numbers, provenance
  `("quad_v1_forward_density", "quad_v1")`: diagnostic approximations that need not equal folded-price
  increments. It reports no KI path statistics (the fields are unavailable with a reason) and
  raises `StatisticsRefused` for memory or EXPIRY Phoenix, whose coupon cash it cannot compute.
  See [`autocallable-ki-consistency`](../../../../../docs/plans/2026-11-10-autocallable-ki-consistency.md).

### Path analyzer (`AutocallablePathAnalyzer.analyze_snowball_risk_neutral`)

Returns `EventStatistics` (family `"snowball"`) from its own simulated paths, with `pv` from
`SnowballMCEngine.price()`. One path set is one batch: `standard_errors` is undefined. With a
continuous KI the analyzer knows only the monitoring step, so `ki_table` and
`knock_in.expected_time` are unavailable with that reason.

## Continuous KI: shared native 1D controls

MC, PDE, QUAD V1 and QUAD V2 keep their numerical methods but share the coefficient-change knots
and the controls `continuous_term_structure="exact" | "piecewise_constant"` and
`continuous_steps_per_year`. Exact mode rejects unrecognized smooth coefficient models; piecewise
mode supports them with time-refinement qualification. Native MC uses its path generator's
interval variance in the crossing bridge. V2 uses accelerated grid reflection with
`continuous_grid_backend="pointwise"` kept as a reference. Qualification:
[continuous KI](../../../../../docs/research/continuous-ki-consistency/README.md) (native 1D GBM)
and `docs/research/vol-model-ki-consistency/README.md` (local vol, Heston/SLV). Those reports
predate this API and measure the old KI-ever field; their pricing results stand.
