# Accrual time, separate from observation time — design

Status: approved in conversation 2026-10-08 (sections 1–3); revised for review round 1 (P1-1..5, P2-6);
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
**`AccrualConfig.accrual_end: AccrualEndType = AccrualEndType.OBSERVATION`.** `AccrualConfig.accrual_dates` is removed.

**`AccrualPeriod`** (new module `asset/equity/product/option/accrual.py`), a frozen dataclass:
- `start_time`, `end_time`: contract-clock years (the quoted fraction, R1); or
- `start_date`, `end_date`: calendar dates; and two explicit overrides, one per cash leg (review round 1, P1-1):
  - `cumulative_fraction: Optional[float]`: the knock-out accrual from the contract start to this record;
  - `period_fraction: Optional[float]`: this record's coupon period (Phoenix).
Exactly one of the time pair or the date pair is set, or neither when both legs carry an override. `end >= start`.
Anything else raises `ValidationError`. A leg without its override is computed from the time or date pair.

**`ObservationRecord.accrual: Optional[AccrualPeriod] = None`.**

**Filled once, at product construction, never overwritten.** Each product fills its records after building its
schedules, in its validation step. A record that already carries an `accrual` keeps it: a copy, a twin or a converted
schedule passes its periods on (review round 1, P1-5). Covered records:
- every knock-out record (Phoenix coupons share them);
- every knock-in record (a REBASED schedule hangs on the knock-in's accrual end, §5);
- KO-reset schedule 2.

The overrides first, each leg on its own:
1. `accrual_factors[i]` given: `cumulative_fraction = period_fraction = accrual_factors[i]`. This keeps today's meaning: the snowball and Phoenix knock-out factor, and Phoenix's coupon period, are both read from it.
2. Phoenix `fixed_coupon_year_fraction` given: `period_fraction` only. The knock-out leg is still computed from the times or dates, so a monthly Phoenix at 12% keeps paying knock-out coupons 1, 2, 3 (review round 1, P1-1).

Then the endpoints, which every leg without an override reads:
3. Dated record: `end_date` = the observation date (`OBSERVATION`) or the authoritative settlement date (`SETTLEMENT`, below). `start_date` = the previous record's `end_date`, or `initial_date` for the first record.
4. Numeric record: `end_time` = `accrued_offset + observation_time` (`OBSERVATION`) or `accrued_offset` + the authoritative settlement time (`SETTLEMENT`). `start_time` = the previous record's `end_time`. For the first record it is `first_period_start` when the contract arrives aged (legacy `accrued_offset` / `first_period_start` inputs). A control: `accrued_offset = 0.3`, first observation `0.2`, `first_period_start = 0.25` gives a first period of 0.25, as today, not 0.5. Otherwise it is `0.0`. A first period whose start is unknown (`first_period_start is None`) stays unset; a coupon reading it raises, as today.

Rule 3 needs `initial_date` whenever the contract annualizes: an annualized dated contract without one raises at
construction (R5).

**The authoritative settlement endpoint (`SETTLEMENT`; review round 1, P1-2).** At construction it comes from the
settlement resolver's own precedence, applied with the product's own terms only (construction has no environment):
1. The record's explicit `settlement_date` / `settlement_time`.
2. Else the product's `SettlementConvention`:
   - no convention, or `lag == 0`: the observation itself. That is the contractual payment instant, not a fallback.
   - `YEAR_FRACTION` on a numeric record: `observation_time + lag`. An observation at 0.25 with lag 0.01 accrues to 0.26.
   - `BUSINESS_DAYS`, or `CALENDAR_DAYS` with an adjustment, on a dated record: the convention's own `calendar`. The environment's calendar is not available at construction, so a convention without one is refused ("SETTLEMENT accrual needs the settlement calendar on the contract").
   - Any other pairing is refused, never mapped to the observation: `YEAR_FRACTION` on a dated record, or a business-day lag on a numeric record.
3. A coupon paid at expiry (`coupon_pay_type = EXPIRY`) under `SETTLEMENT` is refused. Its accrual end, the coupon's own settlement or the deferred payment, is a term-sheet choice the contract must state.

**A terminal accrual period** is filled the same way for the contract:
- `product.tenor` set: `cumulative_fraction`;
- dated: `initial_date` to the `tenor_end` date;
- numeric: `0.0` to `accrued_offset + maturity`.

KO-reset has two:
- `pre` ends at schedule 1's last record's accrual end;
- `post`: for ABSOLUTE, at schedule 2's last accrual end. For REBASED there is none until a knock-in anchor exists (§5).

**What moves and what does not.**
- These move `observation_time`, and settlement times, only: `ObservationSchedule.time_shift`, `BarrierConfig.time_shift`, `snap_contract` and `with_absolute_schedule_2`. A copied record keeps its `accrual`.
- `accrued_offset` and `first_period_start` stay as fields, still banked by `AccrualConfig.shifted`. They are read once, when legacy inputs are turned into periods (rule 4), and never again once a record carries its period.

## 5. The resolver (`accrual.py`)

The only functions that compute accrual. Precedence is preserved leg by leg (review round 1, P2-6).

- **`ko_accrual_factor(product, record, env) -> float`.** The knock-out accrual from the contract start to the record's accrual end:
  1. `cumulative_fraction` when set, whatever the annualization flag. Explicit factors win today: notional 100, rate 12% and factor 0.4 pay 4.8.
  2. Otherwise 1.0 when knock-out accrual is not annualized (`_effective_annualized_flag(is_annualized_ko)`).
  3. Otherwise dated: YF(`initial_date`, `end_date`, `annualization_day_count`, `env.bus_days_in_year`, `env.calendar`); numeric: `end_time`.
- **`coupon_period_fraction(product, record, env) -> float`.** Today's order:
  1. 1.0 when the coupon rate is not annualized;
  2. else `period_fraction`;
  3. else dated: the coupon day count's fraction from `start_date` to `end_date`; numeric: `end_time − start_time`.
- **`contract_tenor(product, env, leg="contract", knock_in=None) -> float`.** V0/V1's tenor from the terminal accrual period (KO-reset: `leg` is `"pre"` or `"post"`). A REBASED `post` leg needs `knock_in`.
- **`rebased_ko_accrual(product, offset_record, env, knock_in) -> float`.** A REBASED schedule-2 knock-out accrues to the knock-in's accrual end plus the offset's quoted fraction. For a dated contract the end is the knock-in's accrual end date plus the offset's whole days, and the factor is the YF from `initial_date`.

Every function raises `ValidationError` when a record it needs carries no `accrual`, or a REBASED call has no
`knock_in`. None falls back to observation time.

**The knock-in anchor (review round 1, P1-3).** `KnockInAnchor` is a frozen dataclass: the knock-in's accrual end,
`end_time` or `end_date`. It is resolved once, when the knock-in happens, and passed through the resolver and the
conversion:
- **A knock-in on a record** takes that record's `accrual.end`. A REBASED contract requires discrete knock-in monitoring (the product's existing validation), so a REBASED knock-in is always on a record. ABSOLUTE contracts, continuous monitoring included, never need an anchor.
- **A carried knock-in** (`_otc_lifecycle_knock_in_date`) takes the knock-in record whose due day is that date, under the tracker's due rule. A date matching no record is refused.
- **Monte Carlo** builds one anchor per path from the path's first knock-in record.
- **REBASED post tenor.** With a final offset 0.5, knock-ins at 0.25 and 0.5 give tenors 0.75 and 1.0. So `contract_tenor(leg="post", knock_in=a)` = the anchor's end + the final offset's quoted fraction (dated: + its whole days).

**REBASED conversion.** `KnockOutResetSnowballOption.with_absolute_schedule_2(knock_in_time, *, anchor)` writes
each converted record's `AccrualPeriod`:
- numeric: `end_time = anchor.end_time + offset`, start = the previous converted end (the first starts at the anchor);
- dated: as in `rebased_ko_accrual`.

It also rebuilds the converted contract's `post` terminal period from the anchor and the final offset. A knocked-in
REBASED contract therefore prices and books the same cash as the un-knocked contract's MC route on the same path.

## 6. Consumers rewired (each local copy deleted)

- `SnowballOption.resolve_ko_observations`, `PhoenixOption.resolve_ko_observations`: `ko_accrual_factor`.
- `PhoenixOption.coupon_period_fraction`, `get_coupon_period_year_fractions`, `get_ko_payoff`: `coupon_period_fraction` / `ko_accrual_factor` (fixes §3.3–4).
- `KnockOutResetSnowballOption.compute_ko_accrual_factor`, `_resolve_pre_contract_tenor`, `_resolve_post_contract_tenor`: the resolver (`accrual_factors` now honoured on KO-reset).
- `SnowballOption.get_contract_tenor`, `PhoenixOption.get_contract_tenor`: `contract_tenor`.
- `SnowballMCEngine._compute_payoffs_ko_reset`: `rebased_ko_accrual` and `contract_tenor(leg="post", knock_in=...)` with each path's anchor.
- `quad/v2/adapters.py` Phoenix fractions: `coupon_period_fraction` (fixes §3.5).
- `intraday/events.py` `phoenix_coupon_fractions`: `coupon_period_fraction` (fixes §3.5).
- **The intraday twin changes** (review round 1, P1-4). It deep-copies the product and then builds new `ObservationRecord`s without running product construction, so those records would carry no `accrual` and the resolver would raise.
  - `intraday/twin.py` copies each source record's `accrual` onto the twin record built from it, matched by source index. The twin's records therefore accrue exactly as the contract's do.
  - The snowball/Phoenix twin stops back-solving `accrual_factors` from cash.
  - The KO-reset twin stops clearing `accrual_factors` and stops reconstructing cash through one `accrued_offset`.
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
re-measured; any move left is observation timing alone and goes to the user before it is re-pinned. That branch's
final-review findings (C1, C2, I1–I3) are fixed after the rebase.

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
- **Refusals.** An `AccrualPeriod` with both representations, or with neither and no overrides, or with `end < start`, raises. The resolver raises on a record without `accrual`, and on a REBASED call without `knock_in`.
- **Goldens.** Dated and unaged numeric products are byte-identical. Aged numeric and numeric-with-`initial_date` rows are listed with magnitudes. Fingerprint-pinned goldens are regenerated and listed.
- **Full suite** with `-n 6` under the RAM guard.

## 9. Out of scope

§3 items 6–10 (annualization-flag defaults, the PDE V0 boundary rebate, `get_tenor`'s `BUSINESS_DAYS` calendar,
range accrual, DCN / FX TARN / cashleg on the resolver): each a follow-up of its own. The KO-reset lifecycle work
itself (spec `2026-10-08-ko-reset-lifecycle-design.md`) resumes after this spec merges (§7).
