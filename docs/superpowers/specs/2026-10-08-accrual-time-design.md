# Accrual time, separate from observation time — design

Status: approved in conversation 2026-10-08 (sections 1–3); revised for review rounds 1 and 2;
awaiting review.

## 1. Purpose

An autocallable's observation is a **trading-day** quantity: when the market is checked. Its accrual is a
**calendar-day** quantity in the contract's interest basis (ACT/365F, ACT/360, 30/360, ...): how much the contract
owes. On a numeric (year-fraction) schedule the library uses one number for both. The record's
`observation_time` is also its accrual (`accrued_offset + observation_time`). So anything that moves an
observation moves the cash.

The KO-reset lifecycle branch (`handoff-small-fixes`, spec `2026-10-08-ko-reset-lifecycle-design.md` §4a) snaps
numeric observation times to the day the tracker observes them. It measured the consequence:
- a `k / 12` Phoenix coupon at 12% pays 0.9863 or 1.0192 instead of 1.0;
- a `k / 12` snowball knock-out pays 202.466 instead of 202.5;
- a pinned standard-snowball backtest moved from +10,849 to −1,656. Prices moved by at most 0.4%, but one hedge round trip at multiplier 200 disappeared.

The user stopped that branch (2026-10-08) until accrual time is separated from observation time. **The KO-reset
branch waits for this spec.**

**Success.** A contract's cash — knock-out coupons, Phoenix coupons, annualized rebate (V0) and knock-in (V1)
amounts — depends only on its accrual terms. Ageing a contract (`time_shift`), snapping its observations, or
converting a REBASED schedule moves observation timing and never moves cash. Accrual is computed in one place;
every product, engine and intraday path asks that place.

## 2. User rulings (2026-10-08)

- **R1. Numeric accrual is the quoted fraction.** A numeric record's accrual is its original numeric time: "monthly" means 1/12 of a year by definition. Observation timing may move; the accrual does not.
- **R2. Accrual end type.** A contract-level option `ACCRUAL_END_TYPE ∈ {OBSERVATION, SETTLEMENT}` says whether a record accrues to its observation (fixing) or to its settlement (payment). Default `OBSERVATION`: existing contracts do not move.
- **R3. Scope.** One accrual resolver for snowball, Phoenix and KO-reset and every engine and intraday path they run on. The inconsistencies of §3 items 1–5 are fixed as a consequence. Items 6–10 are listed follow-ups, untouched.
- **R4. Approach A.** The accrual period lives on each observation record, not in positional lists (B), nor as a separate observation instant (C).
- **R5. A dated, annualized contract without `initial_date` is refused at construction.** Its accrual has no start but "today".

## 3. Where accrual is computed today (inventory, 2026-10-08)

Product side:
- `SnowballOption.resolve_ko_observations` and `PhoenixOption.resolve_ko_observations`: identical copies. Precedence: `accrual_factors[i]`; dated: YF(`initial_date` or the *valuation date* → `observation_date`); numeric without `initial_date`: `accrued_offset + observation_time`; numeric with `initial_date`: YF(`initial_date` → valuation) + raw `observation_time`.
- `KnockOutResetSnowballOption.compute_ko_accrual_factor` (both schedules): the same three branches, and it **ignores `accrual_factors`**.
- `PhoenixOption.coupon_period_fraction` / `get_coupon_period_year_fractions`: `accrual_factors`, `fixed_coupon_year_fraction`, dated DCF in the coupon day count from `[initial_date] + dates`, numeric `t[i] − t[i−1]` with `accrued_offset` / `first_period_start` for the first period.
- `PhoenixOption.get_ko_payoff`: an older copy that **omits `accrued_offset`**.
- Contract tenor (V0 rebate, V1 knock-in): `BaseEquityOption.get_tenor` (`tenor`, else `initial_date` → `tenor_end`, else `maturity + accrued_offset`); KO-reset `_resolve_pre_contract_tenor` / `_resolve_post_contract_tenor`.

Engine and intraday side:
- `SnowballMCEngine._compute_payoffs_ko_reset` (REBASED): re-implements the knock-out accrual (`initial_to_valuation + ki_time + offset`) and the V1 tenor.
- `quad/v2/adapters.py` `compile_contract` (Phoenix): its own coupon-fraction precedence; it ignores `is_annualized_coupon`.
- `intraday/events.py` `phoenix_coupon_fractions`: its own precedence, with a raw first period; `intraday/twin.py` back-solves `accrual_factors`.

Inconsistencies found there:
1. Dated records without `initial_date` accrue from the valuation date: coupons shrink as the contract ages.
2. Numeric records with `initial_date` add a basis year fraction to a raw observation time (two clocks).
3. Phoenix `get_ko_payoff` omits `accrued_offset`.
4. An aged dated Phoenix with `initial_date` may start its first remaining coupon period at `initial_date` (dropped records shift the `[initial_date] + dates` indexing). Inferred from code; the regression test confirms or refutes it.
5. Product, QUAD V2 and intraday disagree on the first coupon period.

Follow-ups, out of scope (R3):
6. `is_annualized_rebate` / `is_annualized_ki` are read raw (`None` is not annualized) while KO accrual uses `_effective_annualized_flag`.
7. The PDE V0 boundary rebate is never tenor-scaled.
8. `get_tenor` passes no calendar to a `BUSINESS_DAYS` year fraction.
9. Range accrual (equity and FX) annualizes by the remaining maturity.
10. DCN (fixed 30/360), FX TARN and `cashleg` keep their own accrual.

Dead field: `AccrualConfig.accrual_dates` has no reader.

## 4. Data model

**`AccrualEndType`** (`util/enum/option_enums.py`, beside `TenorEnd`): `OBSERVATION`, `SETTLEMENT`.
**`AccrualConfig.accrual_end: AccrualEndType = AccrualEndType.OBSERVATION`.** `AccrualConfig.accrual_dates` is
removed. **`AccrualConfig.post_accrual_factors: Optional[List[float]]`** is new; it is KO-reset schedule 2's explicit
factors (§4.3).

### 4.1 `AccrualPeriod`: immutable and self-contained

A frozen dataclass in a new module, `asset/equity/product/option/accrual.py`. It carries everything a resolver
needs, so a period read off a copy, a twin or an aged contract gives the same cash wherever it travels (review round
2, P1-1):

- **`kind`**: `TIME` (contract-clock years, the quoted fraction, R1) or `DATE` (calendar dates). Every endpoint below is of that kind.
- **`origin`**: the contract's accrual start. `0.0` for `TIME`; `initial_date` for `DATE`, copied in, so a twin that clears `initial_date` still resolves.
- **`start`**: the coupon period's start. It may be unknown (`None`).
- **`end`**: the accrual end. It may be absent only when `cumulative_fraction` is set.
- **`day_count`**: the cumulative leg's basis, the contract's `annualization_day_count`.
- **`period_day_count`**: the coupon leg's basis, Phoenix's coupon day count, else `day_count`.
- **`cumulative_fraction`**, **`period_fraction`**: explicit overrides, one per cash leg (review round 1, P1-1).
- **`relative`**: True for a REBASED schedule-2 offset record. Its `start` and `end` are quoted offsets from the knock-in anchor, and it has no `origin` (§4.3, §5.1).

The representations that are allowed (review round 2, P2-7), every other combination raising `ValidationError`:

| representation | cumulative leg | coupon leg |
|---|---|---|
| `cumulative_fraction` only (e.g. a terminal from `product.tenor`) | the fraction | `period_fraction`, else raises |
| `origin` + `end`, `start` unknown (an aged first period with no recorded start) | from `origin` to `end` | `period_fraction`, else raises "period start unknown" |
| `origin` + `start` + `end`, `start <= end` | from `origin` to `end` | from `start` to `end` |
| `relative`: `start` + `end` offsets, no `origin` | anchor + `end` (§5) | `end − start` |

An override always wins on its own leg; endpoints serve whichever leg has no override.

**`ObservationRecord.accrual: Optional[AccrualPeriod] = None`.**

### 4.2 Who fills it, and when (review round 2, P1-6)

**The product owns its periods.** At construction a product **detaches** its schedules: every `ObservationSchedule` it
holds (`barrier_config`'s knock-out and knock-in schedules, KO-reset's `post_barrier_config`) is replaced by a copy
whose records are fresh copies. It then fills the copies.
- Two products built on one shared `ObservationSchedule` derive their periods independently. With fixed fractions 1/12 and 0.1, their coupons pay 1.0 and 1.2, as today.
- Construction order can never leak one product's terms into another's.

**A record that arrives already carrying an `accrual` keeps it**, never recomputed. That covers a period the caller
set explicitly, a `deepcopy`, an ageing copy, a twin built from source records (§6), and a converted REBASED schedule.
Detaching copies the period along with the record.

**Covered records:**
- every schedule-1 knock-out record (Phoenix coupons share them);
- every knock-in record;
- KO-reset schedule 2.

### 4.3 The rules, by schedule (review round 2, P1-5)

**Overrides.** Each list belongs to one schedule only.
- `accrual_factors`: schedule 1's knock-out records, positionally. Its length must equal schedule 1's record count, or construction refuses.
  - Each factor sets `cumulative_fraction = period_fraction = accrual_factors[i]`. This keeps today's meaning: the knock-out factor, and Phoenix's coupon period.
- `post_accrual_factors`: KO-reset schedule 2, positionally. Its length must equal schedule 2's record count.
  - ABSOLUTE: each factor sets `cumulative_fraction`.
  - REBASED: each factor is the offset's accrual: `relative` with `end = factor`, and the cumulative leg is anchor + factor (§5).
- `fixed_coupon_year_fraction` (Phoenix): sets `period_fraction` only. The knock-out leg is still computed from the endpoints, so a monthly Phoenix at 12% keeps paying knock-out coupons 1, 2, 3 (review round 1, P1-1).
- Knock-in records take no factor: they carry no cash, only the anchor's fixing (below).
- The controls: two pre records, three post records, `accrual_factors` of length 2. Schedule 1 takes the factors and schedule 2 is computed. A `post_accrual_factors` of length 2 against three post records is refused, as is an `accrual_factors` of length 3.

**Endpoints.** These serve every leg without an override.
- **Dated** schedule-1 knock-out record: `end` is the observation date (`OBSERVATION`) or the authoritative settlement date (`SETTLEMENT`, §4.5). `start` is the previous record's `end`, or `initial_date` for the first record. `origin = initial_date`.
- **Numeric** schedule-1 knock-out record: `end` is `accrued_offset + observation_time` (`OBSERVATION`) or `accrued_offset` + the authoritative settlement time (`SETTLEMENT`). `start` is the previous record's `end`.
  - For the first record, `start` is `first_period_start` when the contract arrives aged (legacy `accrued_offset` / `first_period_start`). The control `accrued_offset = 0.3`, observation 0.2, `first_period_start = 0.25` gives 0.25, as today.
  - Otherwise `start` is `0.0`.
  - `first_period_start is None` leaves `start` unknown.
- **Knock-in records accrue to their contractual fixing, always** (review round 2, P1-3). `end` is the observation date or `accrued_offset + observation_time`, whatever `ACCRUAL_END_TYPE` says. A knock-in pays nothing, and its end serves only as the anchor of a REBASED reset. A settlement lag on the knock-in must never be counted into the reset.
- **ABSOLUTE schedule 2**: as schedule 1, on its own records.
- **REBASED schedule 2: its own rule** (review round 2, P1-2). Its times are offsets from the knock-in, so no `accrued_offset` is banked into them. The record is `relative`, with `end` = its quoted offset (`OBSERVATION`) or its offset settlement endpoint (`SETTLEMENT`: the explicit offset payment, else offset + the convention's `YEAR_FRACTION` lag). `start` is the previous offset's `end`, or 0 for the first.
  - The control: an aged contract with elapsed 0.3, remaining knock-in time 0.2 (anchor 0.5) and offset 0.25 accrues to 0.75, not 1.05.

Dated annualized contracts need `initial_date`: one without it raises at construction (R5).

### 4.4 Terminal periods

The contract gets a terminal period the same way:
- `product.tenor` set: `cumulative_fraction` only;
- dated: `origin = initial_date`, `end` = the `tenor_end` date;
- numeric: `origin = 0.0`, `end = accrued_offset + maturity`.

KO-reset has two:
- `pre`: schedule 1's last record's accrual end, from schedule 1's origin.
- `post`:
  - ABSOLUTE: schedule 2's last record's accrual end.
  - REBASED: a `relative` terminal whose `end` is the final offset. It resolves only with an anchor (§5).

### 4.5 The authoritative settlement endpoint (`SETTLEMENT`)

At construction the endpoint comes from the settlement resolver's own precedence, applied with the product's own
terms only (construction has no environment):
1. The record's explicit `settlement_date` / `settlement_time`.
2. Else the product's `SettlementConvention`:
   - no convention, or `lag == 0`: the observation itself. That is the contractual payment instant, not a fallback.
   - `YEAR_FRACTION` on a numeric record: `observation_time + lag`. An observation at 0.25 with lag 0.01 accrues to 0.26.
   - `BUSINESS_DAYS`, or `CALENDAR_DAYS` with an adjustment, on a dated record: the convention's own `calendar`. The environment's calendar is unavailable at construction, so a convention without one is refused.
   - Any other pairing is refused, never mapped to the observation.
3. A coupon paid at expiry (`coupon_pay_type = EXPIRY`) under `SETTLEMENT` is refused: its accrual end is a term-sheet choice the contract must state.

### 4.6 What moves and what does not

- These move `observation_time`, and settlement times, only: `ObservationSchedule.time_shift`, `BarrierConfig.time_shift`, `snap_contract` and `with_absolute_schedule_2`. Every record keeps its `accrual`.
- `accrued_offset` and `first_period_start` stay as fields, banked by `AccrualConfig.shifted`. They are read once, when legacy inputs are turned into periods, and never again.

## 5. The resolver (`accrual.py`)

The only functions that compute accrual. Each reads the period, not the product's dates (review round 2, P1-1), and
keeps today's precedence leg by leg (review round 1, P2-6).

- **`ko_accrual_factor(product, record, env, *, knock_in=None) -> float`.** The cumulative leg:
  1. `cumulative_fraction` when set, whatever the annualization flag. Explicit factors win today: notional 100, rate 12% and factor 0.4 pay 4.8.
  2. Otherwise 1.0 when knock-out accrual is not annualized (`_effective_annualized_flag(is_annualized_ko)`).
  3. Otherwise `DATE`: YF(`origin`, `end`, `day_count`, `env.bus_days_in_year`, `env.calendar`); `TIME`: `end − origin`.
  4. A `relative` record needs `knock_in`: the end is the anchor's fixing + `end` (`DATE`: + `end` whole days), measured from the anchor's `origin` in the anchor's `day_count`.
- **`coupon_period_fraction(product, record, env) -> float`.** The coupon leg, in today's order:
  1. 1.0 when the coupon rate is not annualized;
  2. else `period_fraction`;
  3. else an unknown `start` raises;
  4. else `DATE`: YF(`start`, `end`, `period_day_count`); `TIME`: `end − start`.
- **`contract_tenor(product, env, leg="contract", *, knock_in=None) -> float`.** The cumulative leg of a terminal period. A REBASED `post` leg needs `knock_in`: with a final offset 0.5, knock-ins at 0.25 and 0.5 give 0.75 and 1.0.

Each function raises `ValidationError` on a record it needs that carries no `accrual`, or on a `relative` period
without `knock_in`. None falls back to observation time.

### 5.1 The knock-in anchor (review round 1, P1-3; round 2, P1-3 and P1-4)

**`KnockInAnchor`** is a frozen dataclass:
- `source_index`: the knock-in record that fired, an index into the contract's original knock-in schedule;
- `fixing`: that record's accrual `end`, its contractual fixing (`TIME` or `DATE`);
- `origin` and `day_count`: copied from that record's period.

It is **resolved once, when the knock-in happens, and then persisted**, never looked up again:
- **The tracker** builds it from the record whose test fired. With knock-in records due on days 91 and 93 and the first close on day 95, that is index 0 with its contractual fixing at day 91. The close the knock-in was observed on, day 95, stays the timing of the reset (the KO-reset spec's `ki_date`), and is not the accrual anchor.
- **The state stores it:** `AutocallableLifecycleState.knock_in_anchor`. It travels with every checkpoint and every pricing copy (`_otc_lifecycle_knock_in_anchor`). Ageing that drops the source record cannot lose it, because the anchor is a value, not a reference.
- **A carried knock-in** must arrive with its anchor (`_otc_lifecycle_knock_in_anchor` on the product, or `knock_in_anchor` on a passed state). A REBASED contract carrying a knock-in date but no anchor is refused. No lookup by date is attempted: a date can match zero records or several.
- **Monte Carlo** builds one anchor per path from the path's first knock-in record.
- **Continuous monitoring** never needs an anchor: REBASED requires discrete knock-in monitoring, which the product already validates, and ABSOLUTE contracts never need one.

### 5.2 REBASED conversion

`KnockOutResetSnowballOption.with_absolute_schedule_2(knock_in_time, *, anchor)` places each converted record's
observation at `knock_in_time + offset`; this is timing, so snapping still applies to it. Each record's accrual comes
from its **preserved** `relative` period: `end = anchor.fixing + period.end`, `start` likewise, with `origin` and
`day_count` from the anchor. The result is an ordinary non-relative period.

Under `SETTLEMENT` the offset endpoint already holds the payment offset, and the anchor is the fixing, so the lag is
counted once. A knock-in at 0.25 with a knock-in lag of 0.01 and an explicit post payment offset of 0.52 accrues to
0.77, not 0.78.

The converted contract's `post` terminal becomes `anchor.fixing + final offset`. A knocked-in REBASED contract
therefore prices and books the same cash as the un-knocked contract's MC route on the same path.

## 6. Consumers rewired (each local copy deleted)

- `SnowballOption.resolve_ko_observations`, `PhoenixOption.resolve_ko_observations`: `ko_accrual_factor`.
- `PhoenixOption.coupon_period_fraction`, `get_coupon_period_year_fractions`, `get_ko_payoff`: `coupon_period_fraction` / `ko_accrual_factor` (fixes §3.3–4).
- `KnockOutResetSnowballOption.compute_ko_accrual_factor`, `_resolve_pre_contract_tenor`, `_resolve_post_contract_tenor`: the resolver (schedule-specific factors honoured, §4.3).
- `SnowballOption.get_contract_tenor`, `PhoenixOption.get_contract_tenor`: `contract_tenor`.
- `SnowballMCEngine._compute_payoffs_ko_reset`: `ko_accrual_factor(..., knock_in=anchor)` and `contract_tenor(leg="post", knock_in=anchor)` with each path's anchor.
- `quad/v2/adapters.py` Phoenix fractions: `coupon_period_fraction` (fixes §3.5).
- `intraday/events.py` `phoenix_coupon_fractions`: `coupon_period_fraction` (fixes §3.5).
- **The intraday twin changes** (review round 1, P1-4; round 2, P1-1). It builds new `ObservationRecord`s without running product construction, and it clears `initial_date`.
  - `intraday/twin.py` copies each source record's `accrual` onto the twin record built from it, matched by source index. Periods are self-contained (`origin` and both day counts travel inside them), so clearing `initial_date` on the twin no longer matters.
  - The snowball/Phoenix twin stops back-solving `accrual_factors`.
  - The KO-reset twin stops clearing `accrual_factors` and stops reconstructing cash through one `accrued_offset`. It carries a carried knock-in's `KnockInAnchor`.
  - Its existing cash and tenor checks become equalities against the source contract's periods.

## 7. Compatibility

| contract | expected |
|---|---|
| dated, `initial_date` set | bit-identical (the same YF call on the same dates) |
| numeric, never aged | bit-identical (the period end is today's `accrued_offset + t`) |
| numeric, aged | ulp-level: today `(offset + bump) + (t − bump)`, now the original `t`; every moved golden row listed with its magnitude |
| numeric with `initial_date` | the quoted fraction replaces YF(`initial_date` → valuation) + raw `t` (§3.2); can move when the basis is not ACT/365; measured and listed |
| dated, annualized, no `initial_date` | refused (R5); before landing, every test, example and golden that builds one is counted and given an `initial_date`, each listed |

Every product's fingerprint changes once (every record gains a field): simulation and replay caches cold-start; any
golden that pins a fingerprint is regenerated and listed.

**Where it is built.** On a branch `accrual-time` cut from `main` (this spec's commit cherry-picked onto it),
independent of the KO-reset work.

**After this spec merges.** `handoff-small-fixes` is rebased onto it. The KO-reset spec's "cash follows the snapped
time" (§4a) is withdrawn: snapping then moves observation timing only. The five tracker tests moved to snapped cash
return to their original values, and the vol-history expectations are re-checked. The `test_book_backtest` anchor is
re-measured; any move left is observation timing alone and goes to the user before it is re-pinned. The KO-reset tracker
adopts §5.1's persisted `KnockInAnchor` beside its `ki_date`. That branch's final-review findings (C1, C2, I1–I3) are
fixed after the rebase.

## 8. Tests and acceptance

- **Cash invariance.** The core property: for `k / 12`, `k / 52` and `k / 252` numeric schedules on snowball, Phoenix and KO-reset (ABSOLUTE and REBASED), every record's knock-out cash, every Phoenix coupon, and the V0/V1 tenors are unchanged by `time_shift` of any size and by snapping every observation time to a whole day.
- **`ACCRUAL_END_TYPE = SETTLEMENT`.** A dated contract with a 2-business-day settlement lag accrues each coupon to its payment date; a numeric record accrues to its `settlement_time`; `OBSERVATION` (the default) is bit-identical to today.
- **One answer everywhere.**
  - At an observation landing at the valuation instant, the knock-out cash MC, PDE, QUAD V1 and QUAD V2 decide equals `ko_accrual_factor`'s.
  - Phoenix coupon fractions agree across the product, QUAD V2 and intraday.
  - A knocked-in REBASED contract's converted copy books the cash the un-knocked MC route pays on the same path.
- **Regressions, §3.1–5.**
  - A dated annualized contract without `initial_date` is refused.
  - A numeric contract with `initial_date` accrues the quoted fraction.
  - Phoenix `get_ko_payoff` keeps `accrued_offset`.
  - An aged dated Phoenix's first remaining coupon period starts at the previous record.
  - The first coupon period is the same in the product, QUAD V2 and intraday.
- **Review round 1 controls.**
  - **Two legs:** a monthly Phoenix at 12% with `fixed_coupon_year_fraction = 1/12` pays knock-out coupons 1, 2, 3 on notional 100, and coupons 1, 1, 1.
  - **Settlement endpoint:** an observation at 0.25 with a `YEAR_FRACTION` lag of 0.01 accrues to 0.26 under `SETTLEMENT`. A business-day lag without the convention's calendar, `YEAR_FRACTION` on a dated record, and `EXPIRY` coupons under `SETTLEMENT` are each refused.
  - **Knock-in anchor:** REBASED with final offset 0.5 gives knock-ins at 0.25 and 0.5 tenors of 0.75 and 1.0, the same in the tracker's converted copy and per MC path. A carried knock-in date that matches no record is refused.
  - **Twin:** the intraday twin's records carry the source contract's periods, and the existing twin cash checks hold.
  - **Legacy first period:** `accrued_offset = 0.3`, first observation 0.2, `first_period_start = 0.25` gives a first period of 0.25. With `first_period_start = None`, an annualized coupon still raises.
  - **Factor precedence:** notional 100, rate 12%, `accrual_factors = [0.4]` with knock-out annualization off pays 4.8.
- **Review round 2 controls.**
  - **Dated twin:** a dated contract's twin, with `initial_date` cleared, resolves the source contract's knock-out cash and coupons exactly.
  - **REBASED offsets:** an aged contract (elapsed 0.3, remaining knock-in 0.2, offset 0.25) accrues the converted knock-out to 0.75.
  - **Anchor at the fixing:** knock-in at 0.25 with a lag of 0.01, explicit post payment offset 0.52, `SETTLEMENT`: the converted record accrues to 0.77.
  - **Persisted anchor:** knock-in records due on days 91 and 93, first close day 95. The state's anchor is index 0 with its fixing at 91, and it survives ageing past both records, a checkpoint round trip and the pricing copy. A REBASED product carrying a date but no anchor is refused.
  - **Schedule-specific factors:** 2 pre, 3 post, `accrual_factors` of length 2 is accepted; `post_accrual_factors` of length 2 against 3 post records is refused; a REBASED `post_accrual_factors` entry is the offset's accrual.
  - **Ownership:** two products on one shared `ObservationSchedule` with fixed fractions 1/12 and 0.1 pay 1.0 and 1.2, in either construction order.
  - **Representations:** a `cumulative_fraction`-only terminal and an end-only first period both serve the cumulative leg; the coupon leg on the end-only period raises "period start unknown".
- **Refusals.** Every `AccrualPeriod` combination outside §4.1's table raises, as do mixed `TIME`/`DATE` endpoints and `end < start`. The resolver raises on a record without `accrual`, and on a `relative` period without `knock_in`.
- **Goldens.** Dated and unaged numeric products are byte-identical. Aged numeric and numeric-with-`initial_date` rows are listed with magnitudes. Fingerprint-pinned goldens are regenerated and listed.
- **Full suite** with `-n 6` under the RAM guard.

## 9. Out of scope

§3 items 6–10 (annualization-flag defaults, the PDE V0 boundary rebate, `get_tenor`'s `BUSINESS_DAYS` calendar,
range accrual, DCN / FX TARN / cashleg on the resolver): each a follow-up of its own. The KO-reset lifecycle work
itself (spec `2026-10-08-ko-reset-lifecycle-design.md`) resumes after this spec merges (§7).
