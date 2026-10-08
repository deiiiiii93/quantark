# Accrual time, separate from observation time — design

Status: approved in conversation 2026-10-08 (sections 1–3); awaiting review of this written spec.

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
- `start_date`, `end_date`: calendar dates; and
- `fraction: Optional[float]`: an explicit accrual that overrides both.
Exactly one of the time pair or the date pair is set (or neither, when `fraction` is set). `end >= start`. Anything
else raises `ValidationError`.

**`ObservationRecord.accrual: Optional[AccrualPeriod] = None`.**

**Filled once, at product construction.** Each product fills its records after building its schedules, in its
validation step. Covered records: every knock-out record (Phoenix coupons share them), every knock-in record (REBASED
needs the knock-in's accrual end, §5) and KO-reset schedule 2. The first rule that applies wins:
1. `accrual_factors[i]` given: `fraction = accrual_factors[i]` (unchanged meaning: the cumulative knock-out factor, also Phoenix's coupon period).
2. Phoenix `fixed_coupon_year_fraction` given: the coupon period's `fraction`.
3. Dated record: `end_date` = the observation date (`OBSERVATION`) or the record's settlement date (`SETTLEMENT`); `start_date` = the previous record's `end_date`, or `initial_date` for the first record.
4. Numeric record: `end_time` = `accrued_offset + observation_time` (`OBSERVATION`) or `accrued_offset + settlement_time` (`SETTLEMENT`, falling back to the observation when the record has no settlement time); `start_time` = the previous record's `end_time`, or `0.0` for the first record.

Rule 3 needs `initial_date` whenever the contract annualizes: an annualized dated contract without one raises at
construction (R5).

**A terminal accrual period** is filled the same way for the contract:
- `product.tenor` set: `fraction`;
- dated: `initial_date` to the `tenor_end` date;
- numeric: `0.0` to `accrued_offset + maturity`.

KO-reset has two: `pre` ends at schedule 1's last record's accrual end, `post` at schedule 2's last.

**What moves and what does not.**
- These move `observation_time`, and settlement times, only: `ObservationSchedule.time_shift`, `BarrierConfig.time_shift`, `snap_contract` and `with_absolute_schedule_2`. A copied record keeps its `accrual`.
- `accrued_offset` and `first_period_start` stay as fields, still banked by `AccrualConfig.shifted`, but no accrual reads them once a record carries its period.

## 5. The resolver (`accrual.py`)

The only functions that compute accrual:
- **`ko_accrual_factor(product, record, env) -> float`.** 1.0 when knock-out accrual is not annualized (`_effective_annualized_flag(is_annualized_ko)`). Otherwise the cumulative accrual from the contract start to the record's accrual end: `fraction`; dated: YF(`initial_date`, `end_date`, `annualization_day_count`, `env.bus_days_in_year`, `env.calendar`); numeric: `end_time`.
- **`coupon_period_fraction(product, record, env) -> float`.** 1.0 when the coupon rate is not annualized. Otherwise the period from `start` to `end`: `fraction`; dated: the coupon day count's fraction; numeric: `end_time − start_time`.
- **`contract_tenor(product, env, leg="contract") -> float`.** V0/V1's tenor from the terminal accrual period (KO-reset: `leg ∈ {"pre", "post"}`).
- **`rebased_ko_accrual(product, knock_in_record, offset_record, env) -> float`.** A REBASED schedule-2 knock-out accrues to the knock-in record's accrual end plus the offset's quoted fraction. For a dated contract the end is the knock-in's accrual end date plus the offset's whole days, and the factor is the YF from `initial_date`.

Every function raises `ValidationError` when a record carries no `accrual`. It never falls back to observation time.

**REBASED conversion.** `KnockOutResetSnowballOption.with_absolute_schedule_2` writes each converted record's
`AccrualPeriod`: numeric `end_time = knock-in end_time + offset`, start = the previous one; dated as above. A knocked-in
REBASED contract therefore prices and books the same cash as the un-knocked contract's MC route.

## 6. Consumers rewired (each local copy deleted)

- `SnowballOption.resolve_ko_observations`, `PhoenixOption.resolve_ko_observations`: `ko_accrual_factor`.
- `PhoenixOption.coupon_period_fraction`, `get_coupon_period_year_fractions`, `get_ko_payoff`: `coupon_period_fraction` / `ko_accrual_factor` (fixes §3.3–4).
- `KnockOutResetSnowballOption.compute_ko_accrual_factor`, `_resolve_pre_contract_tenor`, `_resolve_post_contract_tenor`: the resolver (`accrual_factors` now honoured on KO-reset).
- `SnowballOption.get_contract_tenor`, `PhoenixOption.get_contract_tenor`: `contract_tenor`.
- `SnowballMCEngine._compute_payoffs_ko_reset`: `rebased_ko_accrual` and `contract_tenor(leg="post")` per path.
- `quad/v2/adapters.py` Phoenix fractions: `coupon_period_fraction` (fixes §3.5).
- `intraday/events.py` `phoenix_coupon_fractions`: `coupon_period_fraction` (fixes §3.5). The intraday twin's back-solved `accrual_factors` become explicit `fraction`s by rule 1; the twin itself is unchanged.

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
- **Refusals.** An `AccrualPeriod` with both or neither representation, or with `end < start`, raises; the resolver raises on a record without `accrual`.
- **Goldens.** Dated and unaged numeric products are byte-identical. Aged numeric and numeric-with-`initial_date` rows are listed with magnitudes. Fingerprint-pinned goldens are regenerated and listed.
- **Full suite** with `-n 6` under the RAM guard.

## 9. Out of scope

§3 items 6–10 (annualization-flag defaults, the PDE V0 boundary rebate, `get_tenor`'s `BUSINESS_DAYS` calendar,
range accrual, DCN / FX TARN / cashleg on the resolver): each a follow-up of its own. The KO-reset lifecycle work
itself (spec `2026-10-08-ko-reset-lifecycle-design.md`) resumes after this spec merges (§7).
