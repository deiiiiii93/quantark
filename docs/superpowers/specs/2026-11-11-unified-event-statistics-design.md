# Unified Event Statistics — Design

> **Historical design.** The final behaviour is the CHANGELOG 0.11.0 entry and
> `docs/event-statistics-migration.md`. Where this draft differs (for example "changing any engine's
> price" is a non-goal here, while 0.11.0 ships sanctioned price moves), those documents win.

Date: 2026-11-11
Status: draft, awaiting approval (review round 2 folded in)

Decisions marked **(user)** were made by the desk owner in the design conversation.
Decisions marked **(builder's default)** were filled in by the author and stand unless
the owner overrides them.

## 1. Intent

Every autocallable engine returns event statistics: how the note ends, when, with what
probability, and what cash moves. Three consumers depend on them **(user)**:

1. **Cash legs.** Margin, premium, rebate and interest legs are valued from the
   distribution of termination events (`EventDistribution`, folding).
2. **Risk reports and hedging.** The desk hedges from the KO profile, knock-in exposure
   and expected cash.
3. **Traders.** They read the statistics to understand the option: its expected
   lifetime, where its value comes from, and how much cash it will need.

Today the same field means different things on different engines and products.
`ki_ever_probability` comes back as `None`, 0.0, 0/1 or a forward probability
depending on the engine, and the legacy `ki_probability` means "ever breached" on MC
but "settles knocked in" on PDE/QUAD. Maturity cash is a computed quantity on some
engines and the residual `pv − KO cash` on others. A number therefore depends on
which engine priced the trade, which defeats all three consumers.

**Goal:** one set of statistics, with one definition each, filled by every engine for
every autocallable product.

## 2. Goals, non-goals, anti-goals

**Goals**

- One result type for Snowball, Phoenix, KO-reset Snowball and DCN.
- Every number below is filled by every engine in the first release **(user)**.
- Termination outcomes partition the paths: no outcome overlaps another **(user)**.
- Every outcome is computed directly, never as a residual of the PV **(user)**.
- Engines agree within tolerance, with Monte Carlo as the reference **(user)**.

**Non-goals**

- Rendering. The result is plain data; downstream apps render the tables **(user)**.
- Changing any engine's price. PVs stay bitwise where the pricing path is untouched.
- A finer knock-in time distribution than the product's KI schedule (or the
  daily buckets for continuous KI), see §4.5.
- New products.

**Anti-goals** (failures even if everything technically works)

- A stand-in 0 for a number the engine did not compute. 0 always means the computed
  answer is zero.
- A field that silently means different things on different engines or products.
- An outcome computed as "PV minus everything else", which hides error.
- Deprecated aliases that keep the old ambiguous fields alive.

## 3. Current state (as of commit 79938a1c)

Types in `quantark/asset/equity/engine/event_stats.py`: `AutocallableEventStats`,
`PhoenixEventStats`, `KOResetEventStats`, `DCNEventStats`; plus
`RiskNeutralSnowballEventStats` in `asset/equity/analysis/autocallable_path_analyzer.py`.
The field-by-field semantics are documented in
`quantark/asset/equity/engine/docs/autocallable_event_stats_api.md`.

Producers (`calculate_event_stats` / `price_with_events`):

| Family | Files |
|---|---|
| MC | `mc/snowball_mc_engine.py` (incl. KO-reset), `mc/phoenix_mc_engine.py`, `mc/snowball_vol_mc_engines.py`, `mc/dcn_mc_engine.py` |
| QUAD V1 | `quad/snowball_quad_engine.py`, `quad/phoenix_quad_engine.py`, `quad/ko_reset_snowball_quad_engine.py` |
| QUAD V2 | `quad/v2/engine.py` |
| PDE 1D (flat vol, local vol) | `pde/snowball_pde_solver.py` and `pde/core/statistics/` (snowball, phoenix, ko_reset, terminal, readout) |
| Heston/SLV ADI | `pde/vol_model_statistics.py`, `pde/*_vol_pde_solvers.py` |
| DCN PDE (flat vol, vol models) | no event statistics today (`pde/dcn_pde_solver.py`, `pde/dcn_vol_pde_solvers.py`); there is no DCN QUAD engine and none is built here |
| Also | `pde_engine.py` (`price_with_events`), `mc/phoenix_vol_mc_engines.py`, `mc/dcn_vol_mc_engines.py`, `quad/v2/engine.py` `calculate_event_stats_at_spots`, `pde/pde_execution_adapters.py`, `quad/v2/execution_adapter.py` |

None of the 8 DCN engines (`DCNMCEngine`, its 3 vol-MC subclasses, `DCNPDEEngine`,
`LocalVolDCNPDEEngine`, `HestonDCNPDESolver`, …) implements `calculate_event_stats`
today. DCN's statistics come from `price_with_events` / `DCNEventStats` on MC only.
`HestonDCNPDESolver` runs its own 2D ADI loop outside `vol_model_statistics.native_adi`.

Consumers: `cashleg/event_distribution.py`, `cashleg/folding.py`,
`cashleg/autocallable_leg.py`, `cashleg/fixed_payoff_leg.py`, `cashleg/accrual_leg.py`,
`asset/equity/engine/dcn_valuation.py` (`bind_dcn_fold`),
`asset/equity/report/autocallable_risk_report.py`,
`asset/equity/riskmeasures/greeks/theta_shift.py`, `backtest/replay/product_replay.py`,
`backtest/replay/schema.py`, `modelvalidation/builders/equity_snowball.py`,
`execution/greeks.py`, `execution/legacy_adapter.py`, `portfolio/equity/position.py`,
and about 64 test files.

Known defects this work must fix first (found in review, verified):

- `PhoenixMCEngine` coupon cash, the MC reference itself:
  - under EXPIRY pay it discounts every coupon at the terminal DF, even on paths
    that knock out early and pay at the KO settlement;
  - under memory it omits arrears released at a KO where the current coupon barrier
    is missed (`phoenix_mc_engine.py` ~1030-1070).

  Today the error is hidden because maturity cash is the leftover.
- MC `calculate_event_stats` uses its own bridge seed (`seed + 1337`) and skips
  `price()`'s adaptive RQMC / conditional-moment paths, so `stats.pv` can differ
  from `price()`.
- QUAD V1 `PhoenixQuadEngine` does not compute memory coupon cash, and it discounts
  non-memory EXPIRY coupons at the terminal payment time instead of the early KO
  settlement.
- `EventDistribution` silently drops the KI stream when its length differs from
  the KO grid (`event_distribution.py:175`).

Main inconsistencies being removed:

- `ki_ever_probability` counts breaches after a knock-out on every engine except
  DCN; a knock-out at valuation gives `None` / 0 / 0-or-1 / a forward probability.
- Legacy `ki_probability` has two meanings.
- MC and QUAD V1 compute maturity cash as `pv − KO cash`, so their
  `reconciliation_error` is 0 by construction.
- `provenance` is missing on MC and QUAD V2.
- KO-reset maturity buckets exist only on PDE and V2; `ko_times` means different
  things per producer; MC REBASED reports post-KO times as offsets.
- `coupon_payment_is_path_dependent` is set only by V2 and PDE.
- Phoenix coupon paid at a knock-out sits inside KO cash on some engines and in coupon
  cash on others.
- The first-hit KI arrays are filled only by MC (and V1 `forward_density`).

## 4. The design

### 4.1 Termination outcomes partition the paths (user)

Every path ends in exactly one outcome:

| Outcome | Definition | Table |
|---|---|---|
| Knocked out | KO at some observation (whether or not it knocked in first) | Knock-out table |
| Matured, never knocked in | No KO and no KI | Maturity table |
| Matured knocked in | KI and no KO | Knock-in table (settlement row) |

Identities that hold on every engine:

- Σ P(KO rows) + P(maturity) + P(KI settlement) = 1
- PV = Σ KO cash + Σ coupon cash + maturity cash + KI settlement cash

Both are reported as residuals (§4.7), never forced.

### 4.2 Knock-out table (user)

One row per KO observation:

| Column | Definition |
|---|---|
| observation time, observation date | when the KO is decided |
| payment time, payment date | when the KO cash settles |
| probability | P(KO at this observation) |
| survival | P(alive after this observation) |
| discounted cash | E[DF(payment) × KO redemption × 1{KO here}] |
| cash to pay | E[KO redemption × 1{KO here}], the nominal amount to fund at settlement |

KO redemption **excludes coupons** (§4.3) **(user)**. For Snowball and KO-reset the KO
rate *is* the KO payoff, so their KO row keeps principal + KO rate × accrued time
**(user)**. Treated the same way, as KO payoff **(builder's default)**:

- DCN's `ko_coupon_rate` amount;
- Phoenix's unconditional `ko_rate` (`ko_coupon`).

Only barrier-contingent coupons (Phoenix `coupon_rate`, released memory arrears, DCN
fixed coupons) move to the coupon table.

**One shared observation grid (builder's default, from review).** The KO and coupon
tables use the same rows: the union of KO and coupon observations, each row flagged
`is_ko_obs` / `is_coupon_obs`. A coupon-only observation (lock-up periods: DCN
`ko_lock_months`, Phoenix observations with no KO barrier) has KO probability 0 and
carries survival unchanged. This keeps today's DCN padding, which
`EventDistribution.from_dcn_stats`, `AutocallableCashLeg(AccrualBasis.COUPON)` and
`bind_dcn_fold` rely on.

KO-reset: rows from both schedules appear in one table, each tagged `pre` or `post`.
ABSOLUTE schedules use absolute times. **REBASED schedules (user):** the `post` rows
and the knocked-in settlement row (§4.5) are reported by **offset from the knock-in
time**, one row per offset. **They go in a separate post-KI offset table with no
survival column**: survival has no meaning across absolute and KI-relative times
mixed together. The `pre` rows stay in the KO table with absolute times and survival
(builder's default, from review). Their payment times are path-dependent, flagged as for
coupons paid at expiry (§4.3). The rows carry no single absolute date, and the cash
ledger buckets their cash at the actual per-path payment events.

**REBASED absolute-time termination distribution (builder's default, from review
P1).** The offset rows are for reading. They cannot value cash legs:
margin, premium, rebate and interest legs need, per termination event type,
probabilities and survival on *absolute* dates. The offset marginals plus the
marginal per-date KI table cannot reconstruct those, because that needs their joint
distribution. Today `EventDistribution._from_ko_reset` refuses offsets
(`cashleg/event_distribution.py:305-309`). So REBASED results also carry an
absolute-time termination distribution, aggregated directly from the MC paths:

- One row per (absolute determination date, event type), for event types KO-`pre`,
  KO-`post`, maturity and KI settlement. Columns: determination time/date,
  payment time/date, probability, discounted cash, cash to pay. Plus the
  absolute survival curve on the same dates.
- Rebased observations fall on trading days (knock-in date + offset period, rolled by
  the calendar), so bucketing by determination date is exact. Each date has one
  deterministic payment date under the record's settlement rule. Rows are sparse,
  and only dates with non-zero mass are stored.
- `EventDistribution` is built from this distribution for REBASED. The
  `NotImplementedError` refusal is removed. ABSOLUTE KO-reset and every other
  product build `EventDistribution` from the ordinary tables as before.
- Identities (§4.1, §4.7) hold on this distribution as on the tables. Its totals by
  event type equal the offset tables' totals.

Summary values: total KO probability, total discounted KO cash, total cash to pay
**(builder's default)**.

### 4.3 Coupon table (user)

The same columns as the KO table, one row per coupon observation:

| Column | Definition |
|---|---|
| observation time/date, payment time/date | the earning observation, and its payment |
| probability | P(alive just before this observation and the coupon barrier is met), including a simultaneous KO |
| survival | P(alive going into this observation) |
| discounted cash | E[DF × coupon paid × 1{paid here}] |
| cash to pay | the same, undiscounted |

Rules:

- **Coupon paid with a knock-out goes to the coupon table**, not KO cash: the current
  coupon and any released memory arrears **(user)**.
- **Snowball and KO-reset: the coupon table is `None`**, not an empty table
  **(user)**.
- **DCN** fixed coupons use this table; DCN already pays no fixed coupon on the KO
  date **(builder's default)**.
- **Memory coupons:** arrears released at an observation are counted in that
  observation's cash, so cash ≠ probability × coupon on those rows. Probability stays
  "barrier met at this observation" **(builder's default, matches today)**.
- **Coupons paid at expiry:** the row is the earning observation. Its payment time is
  path-dependent (the KO or maturity that ends the note), so the row carries a
  path-dependent flag instead of a single payment time. The cash to pay is bucketed by
  the actual payment event in the cash ledger (§4.6). Every engine must set the flag,
  not just V2 and PDE **(builder's default, generalises today's V2/PDE behaviour)**.

### 4.4 Maturity table (user)

One row: **matured, never knocked out and never knocked in**.

Columns: determination time/date, payment time/date, probability, discounted cash,
cash to pay. The cash is the maturity redemption of the never-knocked-in state
(principal if included, plus any maturity coupon or bonus defined by the product),
excluding anything the coupon table already counts.

Computed directly on every engine (§4.7). QUAD V1 needs a directly computed maturity
column like PDE's.

### 4.5 Knock-in (user)

**KI is not triggered after a knock-out (user).** A path that knocks out stops KI
monitoring at that moment. This matches DCN today and changes every other engine.

**Settlement row:** matured knocked in (KI and no KO). Columns: determination
time/date, payment time/date, probability, discounted cash, cash to pay, where the
cash is the knocked-in maturity redemption.

**Path statistics** (they describe the path, not how the note ends, so they may
overlap KO outcomes):

| Value | Definition |
|---|---|
| P(knock in) | P(barrier hit before any knock-out) |
| P(knock in, no knock-out) | equals the settlement row's probability; reported for readers |
| expected knock-in time | E[first-hit time \| knock in before any KO]. Continuous KI: first crossing time. Discrete KI: first breaching KI observation. Undefined when P(knock in) = 0 (§4.9) |
| expected loss | E[DF(maturity payment) × (−downside) × 1{matured knocked in}], a non-negative number |

**Expected loss (user):** principal minus the knocked-in redemption, discounted. For a
principal-excluded product it is the value of the knock-in barrier option. Both cases
use the same formula: the knocked-in redemption is `principal + downside`
(`SnowballOption.get_maturity_payoff_v1`, `snowball_option.py:810`), with `principal`
= 0 when `include_principal` is false, so the loss is `−downside` either way.
`downside` already includes participation, the airbag, the protection floor and the
annualised-KI scaling, so a fully protected note reports 0. Coupons are not netted
against the loss. KO-reset uses its knocked-in schedule's maturity downside.
Phoenix and DCN use their own knocked-in maturity downside term (DCN:
`expected_discounted_loss_leg`'s amount, sign flipped to non-negative).

**Valuation-date rules (user, plus builder's default for the rest):**

- Knocked out at valuation: P(knock in) = 1 if the note was already knocked in,
  else 0. No forward KI calculation.
- Already knocked in (carried state): P(knock in) = 1, expected knock-in time = 0.
- A spot through a continuous KI barrier, or a discrete KI observed at valuation, is
  a knock-in at time 0.
- Knocked out at valuation without a carried KI, or a note with no KI barrier
  (`has_ki_barrier == False`): P(knock in) = 0, the KI settlement row has
  probability 0 and cash 0 (computed zeros, not stand-ins), expected loss 0, the
  per-date KI table is empty, and expected knock-in time is undefined (§4.9)
  **(builder's default, from review)**.

**Per-date knock-in table (user).** Kept, because knock-in-triggered cash legs
(`FixedPayoffLeg` with `AT_KI`, the `EventType.KI` stream) are valued from it. One row
per KI observation date:

| Column | Definition |
|---|---|
| observation time, observation date | the KI observation |
| payment time, payment date | settlement of a leg paid at knock-in, from the KI date under the leg's settlement convention (added after review: `fixed_payoff_leg.py:95` needs it) |
| probability | P(first knock-in at this observation, before any KO) |
| survival | P(not yet knocked in and not knocked out after this observation) |

- The rows follow the product's own KI schedule **(user)**: a monthly-observed KI
  gets monthly rows, a daily-observed KI gets daily rows. No daily table is built
  for a schedule observed less often than daily.
- Continuous KI: one row per trading day of the monitoring window, each holding the
  first-crossing probability within that day. On a numeric-clock contract
  (`ProductClock.NUMERIC`, no calendar or dates), the rows are steps of
  `1 / bus_days_in_year` on the product's time axis, with no date columns
  **(builder's default, from review)**.
- Already knocked in: a single row at time 0 with probability 1.
- Consistency checks: Σ probabilities = P(knock in), and Σ (t × probability) /
  P(knock in) = expected knock-in time (the *conditional* mean, the same quantity
  as the path statistic), within the continuous-KI bucketing.
- **How the grid engines produce it: a forward solve (user, option A).** One
  backward claim per KI date was rejected, as was coarser rows on grid engines.
  - Each backward grid engine (PDE 1D flat and local vol, Heston/SLV ADI, QUAD V2)
    adds a forward march from today's spot using the **exact discrete adjoint** of
    its own backward step: the transpose of the same operators, event stages and
    boundary rows.
  - One forward pass yields the first-knock-in mass absorbed at every KI date, before
    any KO. KO absorption is applied in the same pass.
  - QUAD V1 extends its existing `forward_density` mode.
  - The **price grid is not changed**: no nodes are added for statistics, and
    `price()` stays bitwise.
  - Discrete KI dates are already price-grid event times, so their rows are exact on
    the engine's grid.
  - Continuous KI: the cumulative absorbed mass is read at each trading-day boundary.
    Where a boundary is not a grid time, the value is interpolated between the
    neighbouring grid times. Each result marks which rows are interpolated
    (`interpolated=True` per row) and the interpolation scheme. This is a labelled
    approximation, not a silent one, and it disappears as the time grid is refined.
  - Acceptance: the adjoint pass must reproduce the backward solve's own termination
    probabilities (Σ KO, maturity, KI settlement) to round-off. That is the check
    that it really is the adjoint. Its KI rows go into the cross-engine comparison
    against MC (§5).
  - Expected knock-in time and P(knock in) on grid engines come from the same
    forward pass, so they agree with the table by construction.

### 4.6 Cash ledger (builder's default)

Kept as the payment-date view across all tables: one row per (determination,
payment) pair with discounted and undiscounted cash and dates, built by
`payment_aware_cashflow_fields`. It is the only place where coupons paid at expiry
appear at their actual payment events. Its discounted sum equals the sum of the
tables' discounted cash.

**Cash to pay is computed directly, not recovered from discounted cash (from
review).** Today undiscounted cash = discounted / DF(payment)
(`event_stats.py:144`). That is wrong wherever the payment time is random:
- EXPIRY-paid coupons settle at the KO or maturity that ends the note;
- released memory arrears;
- REBASED rows.

There E[C] ≠ E[DF(τ) C] / DF(any fixed t). Grid engines march a separate
undiscounted column or state for those rows; MC sums undiscounted cash per path.
Where the payment time is deterministic, division by the payment DF stays exact
and may be kept.

**Time axis (from review).** All times in the statistics are on the pricing
environment's time axis, the one `pricing_env.get_discount_factor` reads. Today
DCN builds its times in ACT/365F (`dcn_grid.py:20`) and discounts them as if they
were on the environment's axis, which is wrong under BUSINESS_DAYS or ACT_360. DCN
converts through dates.

### 4.7 Residuals (user)

- **Every outcome is computed directly** on every engine. MC sums each outcome over
  its own paths; grid engines march or integrate a column per outcome.
- `reconciliation_error` = PV − (Σ KO cash + Σ coupon cash + maturity cash + KI
  settlement cash + Σ receivables (§4.9c)). Always reported, never absorbed into an
  outcome.
- `probability_residual` = 1 − (Σ P(KO) + P(maturity) + P(KI settlement))
  **(builder's default)**.

### 4.8 Summary values

- **Expected lifetime (user):** E[time to the deciding observation date]: the KO
  observation date if the note knocks out, otherwise the final observation date
  (year fractions on the product's time axis). A knock-out at valuation gives 0.
- Total discounted cash and total cash to pay across all outcomes and coupons
  **(builder's default)**.

### 4.9 Unavailable numbers and provenance (user)

- A number an engine cannot produce is `None`, and the result's `unavailable`
  mapping (field name → one-line reason) says why. Never a stand-in 0.
- The first release must leave `unavailable` empty on every engine and product
  (§2) for a full request, with one exception: a value that is **mathematically
  undefined** for the trade is `None` with a reason starting `"undefined:"`. Today
  the only such value is expected knock-in time when P(knock in) = 0. It never
  counts as an engine gap. The mechanism otherwise exists for later engines and
  for honest failure.
- Expiry: an expired note with nothing unpaid returns `None`. Expired with unpaid
  cash, it returns receivables only (§4.9c). Today PDE and QUAD V1 return `None` while
  DCN and V2 return settled statistics; all converge on this rule.
- `provenance = (statistics_model, price_model)` is required on every engine,
  including MC and QUAD V2.

### 4.9a Which configurations "every engine" covers (builder's default)

- Every engine class in its default configuration, and every native statistics mode,
  fills every number **for every product it can price**. A product an engine refuses
  to price stays refused for statistics, and the statistics work does not change
  any price (§2). Refusals that remain:
  - un-knocked REBASED KO-reset is MC-only (`lifecycle/providers.py:36`; the PDE,
    ADI, QUAD V1 and V2 KO-reset engines raise);
  - QUAD V1 refuses memory Phoenix with more than 50 observations.

  The REBASED acceptance tests (§5) therefore run on MC only.
- `lifecycle_state` is accepted only by QUAD V2 and the DCN engines (the others raise
  `CapabilityError`, `settlement_support.py:322`). Tests involving receivables and
  terminated states (§4.9c) run on the engines that accept it. Adding lifecycle
  support elsewhere is out of scope.
- Explicit opt-outs keep their current behaviour and are outside the identity
  guarantees and the acceptance matrix:
  - `statistics="refuse"` still raises `StatisticsRefused`.
  - `statistics="flat_vol_surrogate"` still returns 1D flat-vol statistics beside a 2D
    PV, labelled by `provenance`.
  - Local-vol MC `estimator="one_step_survival"` (a Greek estimator) still raises for
    statistics.
- No fallback is added for any of these.

### 4.9b Requesting part of the statistics (builder's default)

The `streams=` selection on `calculate_event_stats` / `price_with_events` stays,
because greek bumps and theta shifts rely on it for speed. Revised after review:

- **Never pruned:** the termination outcomes (KO table probability, survival and
  cash; maturity row; KI settlement row), the receivables table (§4.9c), the cash
  ledger, both residuals and the totals. These are what `EventDistribution`'s
  invariant (KO + maturity outcomes = 1) and the identities need. An
  `AccrualLeg(TRUNCATE_AT_KO)` asking for `{KO}` still gets a valid distribution.
- **Prunable:** the coupon table's probability column, the per-date KI table, the KI
  path statistics (P(knock in), expected knock-in time) and expected loss. Pruned
  values are `None` with `unavailable[field] = "not requested"`. Coupon *cash* stays
  in the ledger even when the coupon table is pruned (as PDE does today,
  `pde/core/statistics/phoenix.py:428`), so the identities still hold.
- The "`unavailable` empty" rule (§4.9) applies to a full request (`streams=None`).

### 4.9c Cash decided but not yet paid (builder's default, revised after review)

Cash decided at or before valuation but paid after it goes into a separate
**receivables table**, not the KO or coupon table. Examples are QUAD V2's
`pending_receivable_pv`, DCN's receivables, and a matured but unsettled note or DCN
loss leg. Putting it in the KO or coupon table would give negative observation
times, which break `AutocallableCashLeg` (rejects t < 0) and
`EventDistribution._survival_grid` (non-monotone grid).

- Columns: source (KO / coupon / maturity / KI settlement / DCN loss), decision date,
  payment time and date, discounted cash, cash to pay.
- The receivables are inside the cash identity (§4.1): PV = outcome cash + coupon
  cash + receivables. They are not in the probability identity, and they are not
  events in `EventDistribution`, because they are not part of the remaining life.
- A knock-out decided *at* valuation (t = 0) is still a KO row with probability 1,
  as today.
- A terminated note with unpaid cash returns statistics with only the receivables
  table filled. All termination probabilities are 0, and the probability identity
  is not checked. A terminated, fully settled note returns `None`. Today QUAD V2
  raises on terminated states and DCN returns `event_stats=None`; both change.

### 4.10 One result type, plain data (user)

- One frozen dataclass family replaces `AutocallableEventStats`,
  `PhoenixEventStats`, `KOResetEventStats`, `DCNEventStats` and
  `RiskNeutralSnowballEventStats`. Tables are plain arrays (or small frozen row
  dataclasses); no rendering, no pandas dependency.
- Optional parts are `None` by product: coupon table for Snowball and KO-reset;
  pre/post schedule tags only on KO-reset.
- MC additionally reports a standard error for each probability, cash and summary
  value, needed by the acceptance tests (§5) **(builder's default)**.
- **Sign convention (builder's default, revised after review):** statistics use
  **the same sign as the engine's `price()`**.
  - `pv` equals `price()`, and every cash amount carries the same sign, so the
    identities hold.
  - For autocallables that is the per-unit buyer view.
  - For DCN it includes `direction_sign`, as `price()` and the DCN lifecycle ledger
    already do. A seller DCN reports negative cash.
  - Probabilities, times and expected lifetime are unsigned.
  - Expected loss is a non-negative magnitude: the size of the note's knocked-in
    downside, whoever holds it.
  - Products refuse a non-positive notional, so the position's signed `quantity`
    applies the holder's sign on top, as today.
  - The earlier "divide out `direction_sign`" rule is dropped: with `pv = price()`
    it broke reconciliation for seller DCNs.

### 4.11 Migration (user)

- Replace outright in one breaking release, with **no deprecated aliases**.
- Every consumer listed in §3 moves in the same change: `EventDistribution` and its
  converters (`from_autocallable_stats`, `from_dcn_stats`), folding, the autocallable
  cash leg, the risk report, theta shift, product replay, execution adapters, the
  equity position, and all tests and golden files that read the old fields.
- **`EventDistribution` changes (builder's default, from review):**
  - Each stream gets its own time axis, replacing the single `event_times` axis.
    The KI stream is built from the per-date KI table, with its own times and
    payment times, so it is never dropped again for having a different length
    from the KO grid.
  - The KO and COUPON streams share the observation grid of §4.2.
  - REBASED KO-reset converts from the absolute-time termination distribution
    (§4.2), and the `NotImplementedError` in `_from_ko_reset` and in
    `folding.py:301` is removed. Folding a REBASED note stays out of scope unless
    the fold binder can take the absolute distribution; otherwise `folding.py`
    keeps refusing it, with that reason.
  - Receivables (§4.9c) are not events.
- Package version takes a minor bump with a migration note mapping each old field
  to its replacement or removal **(builder's default)**.
- **Prerequisite defect fixes** (§3, "Known defects"): the Phoenix MC coupon
  cash, the MC statistics path, and QUAD V1 Phoenix memory and EXPIRY coupon cash
  are fixed before any cross-engine comparison, since MC is the reference.
- `quantark/asset/equity/engine/docs/autocallable_event_stats_api.md` is rewritten to
  this design.
- **Model-validation certificates:** this change edits fingerprinted trees
  (`modelvalidation/builders/intraday_common.py`), so the affected certificates are
  re-run before release, as with earlier fingerprinted-tree edits. This is on the
  release's critical path and should be scheduled as a long unattended run
  **(builder's default, following project practice)**.

## 5. Acceptance (user: cross-engine tests within tolerance, MC as reference)

- **Cross-engine matrix.** For each product family, every number from every engine
  compared against MC. Cases cover at least: continuous and discrete KI; memory and
  non-memory Phoenix; INSTANT and EXPIRY coupon pay; principal included and
  excluded; protection none/partial/full; already knocked in; knocked out at
  valuation; KI observed at valuation; reverse (call-style) notes; KO-reset ABSOLUTE
  and REBASED (REBASED on MC only, §4.9a); DCN on a non-SSE run calendar and with
  `ko_lock_months > lock_months`; Phoenix with a non-zero `ko_rate`; no KI barrier;
  a numeric-clock contract; a seller DCN; aged trades with receivables and a
  terminated-unpaid note (on engines accepting `lifecycle_state`).
- **Tolerance** = k × MC standard error + a discretisation allowance per grid engine.
  The allowance is measured, by showing convergence under grid refinement, and banked;
  it is never a hand-picked constant **(builder's default)**.
- **The MC reference run (builder's default, from review):**
  - It uses PSEUDO random draws, or RQMC with `num_batches ≥ 2`. Standard errors
    come from independent batch means, never IID `std/√N` on a single Sobol batch.
    That rules out DCN's default single-batch Sobol as a reference.
  - Expected knock-in time is a ratio estimator, so its standard error uses the delta
    method. With fewer than two knock-in paths the standard error is `None`, with an
    `"undefined:"` reason, and that number is not gated.
  - MC statistics come from the **same paths as `price()`**: same seeds, bridge
    seed, adaptive RQMC and conditional-moment paths. `stats.pv == price()`
    bitwise.
- **Exact identities** on every engine and case: `reconciliation_error` and
  `probability_residual` within each engine's own residual bound.
- **No-KI-after-KO check:** a path that knocks out and later crosses the KI
  barrier contributes to neither P(knock in) nor expected loss, on every engine.
- **REBASED cash legs end to end (from review P1):** margin, premium, rebate,
  interest (`AccrualLeg`, `AutocallableCashLeg`) and `FixedPayoffLeg` on every trigger,
  valued through `price_with_events` on an un-knocked REBASED KO-reset. The reference
  values each leg directly on the same MC paths. Agreement must be within round-off,
  since both use the same paths. Also tested on an already-knocked-in REBASED note.
- **PV unchanged:** every engine's `price()` is bitwise identical before and after,
  unless a change to the pricing path is explicitly called out.

## 6. Left to the builder

- Dataclass names, module layout, and whether rows are parallel arrays or row
  objects.
- How each grid engine produces expected loss (for example a downside column). Expected
  knock-in time comes from the forward pass (§4.5).
- The interpolation scheme for continuous-KI day boundaries in the forward pass,
  provided it is monotone in cumulative mass and labelled.
- How QUAD V1 gets a direct maturity column, and whether V1 `forward_density` mode
  is kept, provided it fills every number.
- The order engines migrate in within the single release.
- The value of k and the measured discretisation allowances.
- Whether the QUAD V2 `underlying_ki_monitor` state and the ADI KI-ever claim are
  removed or turned into the before-KO monitor; the after-KO behaviour must go.
