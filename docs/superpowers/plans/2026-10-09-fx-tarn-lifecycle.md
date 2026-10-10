# FX TARN / TARF Lifecycle (W4b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Execution method (already chosen by the user): subagent-driven**, in a NEW worktree branched from `main` AFTER the
> prerequisites below have landed. Do not ask for an execution method.

**Goal:** Give the FX TARN and TARF a mandatory realized state, a dated form, correct ageing (`time_shift` /
`record_passed_fixings` / theta), terminal bookings, and day-by-day ageing inside the FX backtest and FX dynamic
scenario, with a numeric contract showing no difference from its dated twin.

**Architecture:** A new frozen state module (`TargetRedemptionState`, `PendingPayment`, events, `TARGET_HIT_RTOL`) and
a shared base class `TargetRedemptionProduct` own the two schedule forms, validation, banking and the resolved view;
`FxTargetRedemptionNote` / `FxTargetRedemptionForward` only supply their terms and per-fixing outcome. The two MC
engines read only `product.resolve(fx_env)`. TARN/TARF implement W6's run-fixings product protocol, so W6's
`RunFixingRecorder` (`quantark/util/run_fixings.py`, already wired into the FX backtest and FX dynamic scenario) ages,
records and settles them inside the two FX run engines, which book their cash from `RunFixingDay` (re-plan
2026-10-10; the planned `quantark/asset/fx/lifecycle/` hook is dropped, W6 spec §6); `BaseFxEngine._fdm_theta` steps
numeric products by the environment's year fraction (Q7).

**Tech Stack:** Python 3.10–3.13, NumPy, pandas (run engines only), pytest + xdist. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-09-fx-tarn-lifecycle-design.md` (rev 2 + Q7/Q8, user-approved 2026-10-09),
evidence in `docs/superpowers/specs/2026-10-09-fx-tarn-lifecycle-evidence/`. Its §1.2 decision table and §12
decisions are user rulings and bind every task. Executors read the spec section a task names before starting.

## Prerequisites (must be on `main` before the implementation branch is cut)

The implementation branch is cut from `main` **after W5 and W1 land**. Neither is on `main` at the time of writing
(`main` = `dcea77b6`).

| Branch | Head | What this plan imports (quoted exactly as it exists on the branch) |
|---|---|---|
| `w5-missing-fixings` | `bfb9b88f` | `quantark/util/fixings.py`: `class MissingFixingWarning(UserWarning)`; `def fixing_label(index: int, record) -> str` (returns `#<index> t=<time:.6g>` when `record.observation_time is not None`, else `#<index> <record.observation_date.date().isoformat()>`); `def warn_filled_fixings(product_name: str, field: str, labels: Sequence[str], *, stacklevel: int = 3) -> None` (one warning naming every label, nothing when empty). Precedent only: `FxRangeAccrualOption.resolve_observations(self, fx_env: "FxPricingEnvironment", *, warn_missing_fixings: bool = True) -> Tuple[List[Tuple[float, bool]], List[Tuple[float, float, int]], float]`. |
| `w1-tracker-clock` | `05ee1f9b` (the spec cites the fix commit `510bf6a7`; `05ee1f9b` is a docs commit on top) | `quantark/asset/equity/lifecycle/autocallable.py`: `class MaturedBeforeRunWarning(UserWarning)`. Also on the branch, **not used by this plan** (see Ruling R3): `def lifecycle_clock_origin(product: Any, start_date) -> Optional[pd.Timestamp]` (returns `product.initial_date` when set, else `start_date`). `MaturedBeforeRunWarning` is NOT re-exported from `quantark/asset/equity/lifecycle/__init__.py`; import it from the `autocallable` module. |
| `w2-rebate-flag` | in progress | Not imported. **Rebase risk only**: it may touch shared accrual code and the `CHANGELOG.md` `[Unreleased]` section. This plan touches no equity accrual code. |
| W6 run fixings (re-plan 2026-10-10) | `6f9c6e74`, on `main`; `w4b-tarn` is rebased onto it (Tasks 1–8 done) | `quantark/util/run_fixings.py`: `RunFixingRecorder` (`register(portfolio) -> RunFixingDay`, `advance(portfolio) -> RunFixingDay`, `days`, `recorded`), `RunFixingDay(date, fixings, product_events=(), pending_pv=0.0)`, `RecordedFixing(position_id, index, label, outcome, spot, gap)`, `participates`, `RUN_FIXINGS_PROTOCOL`; optional hooks `matured_before`, `lifecycle_events`, `pending_pv`; `quantark/util/fixings.py`: `is_history`, `snap_time`, `freeze_missing_fixings`; `FXBacktestEngine.run_fixings` / `FXDynamicScenarioEngine.run_fixings` (register / advance already wired). Tasks 9–12 build on it (section "Re-plan 2026-10-10 against W6"). |

Landing order on `main` is ff-only (memory: landing worktree branches); W5/W1 shas may change on rebase, so Task 1
verifies their **content**, not their shas.

## Global Constraints

- Canonical `quantark.*` imports only; never a flat import.
- Numerics via `quantark/util/numerical` (`is_zero`, `is_close`, `Tolerance`). The ONLY hardcoded tolerance is the
  spec ruling `TARGET_HIT_RTOL: Final[float] = 1e-12` (moved from the engines' `_HIT_RTOL`, value unchanged). Test
  parity bounds use `is_close(a, b, rel_tol=1e-12)` (spec §10).
- Errors are `ValidationError` (inputs), `PricingError` (pricing) from `quantark.util.exceptions`.
- Products stay engine-agnostic; parameter/state objects are immutable (frozen dataclasses; ageing replaces state).
- **Numeric == dated twin** (standing user rule 2026-10-09): identical events, ledger and PV; every parity test runs
  in both forms.
- **Never invent fallback semantics**: if something cannot be done correctly, write `# TODO(<why>)` and stop and
  report; a refusal (`ValidationError`) is allowed, a guessed value is not.
- No MC engine imported inside PDE code (none of these tasks touch PDE code; keep it that way).
- **Run semantics are W6's** (re-plan 2026-10-10; run-fixings spec §3.3 R1–R10, plan Rulings 2, 4, 8, 9, 10 and the
  user's weekend ruling): a fixing at t = 0 is live; a fixing is decided at the spot of the first instant it was live,
  a gap fixing at the previous instant's; an undated run never advances; time-of-day instants and a backwards clock
  are refused; a same-date advance only refreshes the spot. Tasks 9–11 add no second clock, loop or ordering rule;
  `lifecycle_events` stays append-only; nothing replaces a participant's product between `register` and `advance`
  except the recorder, with copies that keep their records.
- Test command for EVERY run (subagents at most `-n 2`; the controller runs the full suite at `-n 6`), with `<WT>`
  replaced by the implementation worktree's absolute path:
  `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest <paths> -n 2 -q`
  (a `.pth` imports the main checkout's `quantark` otherwise; scripts need the same `env PYTHONPATH=...`).
  A nonexistent test path voids the run ("no tests ran") — check the path exists.
- Existing expected values that move are listed old→new in
  `docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/moved_values.txt` for **user sign-off before
  landing**. Q7 is shared FX code: Task 8 measures every FX theta that moves under a non-calendar environment.
- The full suite rewrites two tracked sample files; restore them after every full run with
  `git checkout -- example/mo_volmodels/data/mo_barrier_sample.json example/mo_volmodels/data/mo_calibration_explainer_sample.html`.
  Never `git add example/` (add example files by explicit path only).
- `docs/` is git-excluded: commit plan evidence with `git add -f`.
- `CHANGELOG.md` `[Unreleased]` gets one block for this workstream (Task 2 creates it; later tasks append bullets).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Commit by pathspec. Never push,
  never merge (landing is the controller's job after sign-off).

## Spec drift found while writing this plan (code at `dcea77b6` vs the spec's citations)

- §2.2 M2 cites `fx_pricing_environment.py:187` for the DF error. Line 187 is the text of `get_forward`'s
  `MarketDataError`; the `ValidationError` M2 printed comes from `quantark/param/rrf/rate_curve.py:95` (FlatRateCurve
  DF) via `FxPricingEnvironment.get_domestic_df` (`fx_pricing_environment.py:138-140`). Evidence-only; no task depends on it.
- §1.3 cites W1 at `510bf6a7`; the branch head is `05ee1f9b` (docs on top). Same interface.
- §10 P1(b) says `accrual_start = 1/12 − τ` for both ages. That is right at τ = 0.125 (one fixing banked) but at
  τ = 91/365 two fixings are banked and the first listed period starts at the SECOND fixing, `2/12 − τ`. This plan
  uses `accrual_start = FIX[n_banked − 1] − τ` (Ruling R4). Fixed here, not in the spec.
- Every other cited file:line was checked and matches (`base_fx_engine.py:219-241`, `execution/inventory.py:438-439`,
  `backtest/fx/engine.py:31, 44-45, 57-58, 66, 99, 117, 138-157, 141`, `dynamicscenario/fx/engine.py:79-80, 84-89, 91,
  99-109, 112`, `day_path.py:298-310`, `dynamic_results.py:224-227`, `path_library.py:38, 59, 76, 90`,
  `float_roll.py:51`, `manager.py:168, 211-215`, `lifecycle_manager.py:11-13`, `test_fx_tarn_note_mc.py:68`,
  `example/fx_tarn_demo.py:74, 85, 97, 109`, `matrix_fixtures.py` `fx_tarf`/`fx_tarn` at 932/947).

## Rulings made in this plan (each: what — why — cost if wrong)

- **R1** `state` is keyword-only with a `None` sentinel that raises `ValidationError` — spec R2 requires a QuantArk
  exception for a missing state, which a no-default parameter (`TypeError`) cannot give — cost: none (still mandatory).
- **R2** `FxTargetRedemptionNote(fixing_times=None, coupon_rate=None, ...)`: both get `None` defaults and `coupon_rate is
  None` raises `ValidationError` — the dated form must be callable by keyword without a positional `fixing_times`, and
  Python forbids a required parameter after a defaulted one — cost: none (positional order unchanged).
- **R3** The FX hook's clock origin is the pair environment's `valuation_date` at registration (spec §7.2), NOT W1's
  `lifecycle_clock_origin` — W1's helper returns `product.initial_date` first, which on a dated TARN is the accrual
  start, not the run's clock — cost: none for numeric (no `initial_date`); dated products need no origin.
  *Re-plan 2026-10-10: superseded by W6 R6 (the same rule, held by `RunFixingRecorder`).*
- **R4** P1(b) uses `accrual_start = FIX[n_banked − 1] − τ` (spec erratum above) — the spec's literal value gives a
  first factor of 2/12 at 91/365 — cost: none.
- **R5** Theta parity (D4) is asserted as `|θ_numeric − θ_dated| <= 1e-12 × |price|` — θ is a difference of two prices
  that each match to relative 1e-12 (M10: times differ by ULPs), so a bound relative to θ (≈1e-4 of the price) would
  demand 1e-16 price agreement — cost: the check is 1e-12-of-price, not 1e-12-of-theta.
- **R6** Dynamic-scenario `LifecycleEventSnapshot.cashflow` for a FIXING = quantity × the PAYMENT events of THAT
  fixing booked the same day (an immediate PAYMENT carries the FIXING's `label`) — spec §7.3 says "the quantity-scaled
  amount paid on the day"; with a pay lag that is 0.0 and the cash shows in `paid_cash` on the pay day — cost: one
  result field; easy to change.
- **R7** A dated fixing strictly after valuation that resolves to `is_zero` time (e.g. a Saturday fixing under a
  `BUSINESS_DAYS` env valued on the Friday) is refused at `resolve` with `ValidationError` — the engines need
  t > 0 and treating it as passed or as future would be invented semantics — cost: such a booking cannot be priced
  on that day until a rule is chosen. *Superseded 2026-10-10: the user answered OQ-2 (b); Task 8b makes such a date
  live, like its numeric twin.*
- **R8** Dated `get_delivery` mirrors `get_maturity` (0.0 after `delivery_date`) — `BaseFxEngine.calculate_greeks`
  calls `get_delivery`, so Greeks on an aged dated contract past its last pay date would otherwise raise — cost: none.
- **R9** Q7 changes only `BaseFxEngine._fdm_theta` (spec §12). `localvol_greeks.py:60` and the Heston/SLV PDE thetas
  keep `days/365` — out of the spec's scope, listed as a follow-up in `moved_values.txt` — cost: those engines keep the
  M9 numeric/dated theta gap under non-calendar envs. A non-integral `theta_days` follows the dated branch's
  whole-day truncation (both branches then agree; recorded for sign-off) — cost: a fractional `theta_days` numeric
  theta becomes 0.0 under `CALENDAR_DAYS`, as the dated one already is.
- **R10** P2 adds τ = 0.1 — at τ ∈ {0.125, 91/365} the lag-0.02 coupon is already paid, so pending would be empty and
  P2 would test nothing — cost: none.
- **R11** `_numeric_twin` lives in `test/fx_target_redemption_fixtures.py` (shared by both new test modules) — spec
  §4.3 places it "in the new test module"; still a test helper, not public API — cost: none.
- **R12** Determined cash with `is_zero(amount)` is neither pending nor paid (spec "a non-zero c") — QuantArk numeric
  convention; both forms apply the same rule, so twin parity holds — cost: cash below 1e-10 currency units is dropped.

## Re-plan 2026-10-10 against W6 (Tasks 9–12)

**Context.** Tasks 1–8 are done on `w4b-tarn`, which was rebased onto W6 (`6f9c6e74`, run fixings). W6's spec §6
(user-approved) decides that the general run-fixings recorder **subsumes** the FX lifecycle hook this plan's old
Task 9 built; its plan's "Hand-off to W4b" fixes the interfaces. Main has since moved to `0d1e4133` (W8 / W9 /
carried-KI): `git diff 6f9c6e74 0d1e4133 -- quantark/asset/fx quantark/backtest/fx quantark/dynamicscenario
quantark/util` is empty (W8 touched only W6's tests and equity / replay / explain code), so nothing here depends on it
beyond the W8 rule below. Inputs read: TARN spec rev 2 (+ Q7/Q8), W6 spec rev 2 and plan (Rulings 1–13, hand-off),
`quantark/util/run_fixings.py`, the W6 wiring in `quantark/backtest/fx/engine.py` and
`quantark/dynamicscenario/fx/engine.py`, `test/test_run_fixings*.py`, `test/run_fixings_fixtures.py`, and the FX range
accrual's protocol (`fx_range_accrual_option.py:383-464`) as the worked example. Measured on this branch (scratchpad
probe): a dated zero-lag TARN prices 102.0922 the day before its first fixing, **101.1173 on the fixing day** (the
day's coupon 0.9863 already paid inside `resolve`) and 101.1319 the day after with the coupon booked — the OQ-1 dip.

### Changes

1. **Task 9 rewritten** — "TARN / TARF implement the run-fixings product protocol". No `quantark/asset/fx/lifecycle/`.
   `TargetRedemptionProduct` gains `has_run_fixings`, `open_fixings`, `with_time_elapsed`, `with_history_settled`,
   `matured_before`; `missing_fixing_fills` / `with_missing_fixings_filled` / `resolve`'s warning become strict
   (`is_history`; t = 0 live, W6 D1); protocol indices become booking-relative. `quantark/util/run_fixings.py` gains
   one generic optional hook (`settle_history` → `with_history_settled(env)`), called at registration after the W5b
   fill and after each clock move's recording. Tests: product-level protocol tests and D2 in
   `test_fx_target_redemption_lifecycle.py`; recorder-level tests (old Task 9 list, remapped) plus new W6-rule tests
   (R8, Ruling 9, append-only events, index stability) and the OQ-1 / OQ-2 tests (flipped to (b), item 6) in the new
   `test_fx_target_redemption_runs.py`; one generic stub test appended to `test/test_run_fixings.py`.
2. **Task 10 rewritten** — the FX backtest's wiring is W6's; this task adds `booked_cash(product_events)` (in
   `target_redemption_state.py`) and the `paid_cash` / `pending_pv` columns, value = MTM + cumulative paid cash. B1–B6
   kept on flat paths with the same day indices; the weekend test now expects Friday's spot with `gap=True` (W6 D5); new
   W8 twin test and an OQ-1 value test.
3. **Task 11 rewritten** — the FX dynamic scenario's wiring is W6's; this task adds the Q8 refusal (before cloning or
   registering), the cash fields and the FIXING snapshots, all read from `RunFixingDay`.
4. **Task 12 adjusted** — new Step 0 re-captures the B6 baseline and demo stdout on the W6 base (Task 1's captures
   predate the rebase and W6 moves numeric FX range accruals in the B6 book); Steps 1–2 compare against the new
   captures; Step 3 checks the W6 test files are untouched apart from the Task 9 append; Step 4's sign-off list gains
   the W6-applied moves and the OQ answers; landing note: rebase onto current main and re-run Steps 0–3.
5. **Header** — Architecture sentence, a W6 prerequisites row, a Global Constraints bullet (run semantics are W6's),
   File Structure rows, annotations on Rulings R3 and R7, Review Focus (items 1, 2, 4, 5 retargeted; items 6, 7 new),
   Self-review coverage and type-consistency lines.
6. **User answers 2026-10-10 (OQ-1 (b), OQ-2 (b))** — new Tasks 8a and 8b re-open the built Tasks 2, 5, 6, 7 (moved
   values, CHANGELOG, tests in the full task format); Task 9's `_record_dated` block, its terminal-booking / settle /
   `with_time_elapsed` tests and both pins follow the new rules; Task 10's B3 and W8 expectations move one instant
   (cash on a path-aligned pay date is booked the next instant), its OQ-1 pin becomes the no-dip test and it gains a
   business-day twin test; Task 11 gains a business-day twin test; Task 12's sign-off list, Review Focus 5 / 7 and the
   Self-review follow.
7. **User OQ-3 ruling 2026-10-10 (per-fixing banking)** — Task 8b rewritten (state `fixings_banked`, identity-based
   banked set, `as_of` never past the valuation date via `_state_date`, Friday + Saturday test, OQ-4 pin); Task 9's
   `_record_dated` block, settle test, `fixings_banked` property, D2 parity and a run-level Friday + Saturday test; R15
   and R22 superseded, R24 and R25 added; new open question OQ-4.
8. **User answers 2026-10-10: R24 approved; OQ-4 (b), generalised ("two period dates can share the same fixing
   date")** — new Task 8c (re-opens Tasks 3, 4, 6, 7: equal fixing dates / times accepted, engines simulate each
   distinct time once, periods in schedule order with the spec §6 dead-after-hit rule, R26 — since superseded,
   item 9); Task 8b's OQ-4 pin is
   dropped; Task 9 gains a run-level shared-date test; Re-plan section, Task 12 sign-off, Review Focus 9 and
   Self-review follow.
9. **User ruling 2026-10-10 on R26: option (b), replacing R26** — the TARN gets its own accrual period dates, separate
   from its fixing dates. New **Task 8d** (re-opens Tasks 3 and 7, and narrows Task 8c's refusal):
   `FxTargetRedemptionNote(..., accrual_end_dates=...)` (Rulings R27, R28); default = today's derivation, bitwise;
   periods sharing a fixing date price with no `accrual_factors`. Task 8c's refusal is reworded (default derivation
   only; its test matches "zero length", so 8d does not rewrite it); R26 superseded; new non-blocking OQ-5 / OQ-6;
   Task 9 consumes 8d; Task 12 sign-off, Review Focus 10, Self-review and File Structure follow.

### Old Tasks 9–11 requirements: where each lives now

| # | Old requirement (old task) | Status | Where now |
|---|---|---|---|
| 1 | `quantark/asset/fx/lifecycle/` (`FxTargetRedemptionLifecycle`, `FxLifecycleDay`, `FxLifecycleEntry`, `is_ageable`, `holds_ageable`) (T9) | Dropped | W6 spec §6 (user-approved): the recorder subsumes it. `FxLifecycleDay.events` → `RunFixingDay.product_events`; `.pending_pv` → `RunFixingDay.pending_pv`; `.paid_cash` → `booked_cash(...)` (T10); `holds_ageable` → the Q8 `isinstance` check (T11) |
| 2 | `TargetRedemptionProduct.matured_before(fx_env)` (T9) | Kept, time-based (R16) | T9 product method; called by W6 `_register_position` (W6 Task 5 hook) |
| 3 | Clock origin = pair env date at registration (old R3) | Satisfied by W6 | W6 R6: `_last[underlying]` set at `register` |
| 4 | Matured before the run: `MaturedBeforeRunWarning`, position removed, value 0, no cash (T9) | Satisfied by W6 + T9 `matured_before` | T9 `test_matured_before_the_run_is_removed_with_a_warning`, T10 B5 |
| 5 | Fixings before the run banked at the origin spot, one `MissingFixingWarning` per position (T9) | Satisfied by W6 R5 (fill) + T9 settle hook (banking) | T9 `test_register_fills_history_once_and_books_nothing`, T10 B5 |
| 6 | A fixing ON the origin date banks silently at registration (T9) | Changed by W6 D1 (user) | Live at registration; recorded on the next instant at the origin day's spot as refreshed by a same-date advance (W6 Ruling 9), silently: T9 `test_a_fixing_on_the_origin_date_is_live_and_records_the_next_day_silently` |
| 7 | Payments produced by registration are never booked (T9) | Kept | settle at registration + W6 Ruling 10 (register reports no events): T9 register test, T10 B5 |
| 8 | `advance` before the last date → `ValidationError` (T9) | Satisfied by W6 R9 | T9 `test_the_run_clock_cannot_go_back`, T10 `test_b5_backwards_path_raises`, T11 `test_start_date_before_environment_date_raises` |
| 9 | `advance(portfolio, date)` refuses an environment not on `date` (T9) | Dropped | The recorder takes the instant from the environment; there is no second date to disagree with |
| 10 | Numeric `time_shift` by the environment's year fraction since the last date (T9) | Satisfied by W6 R6 (`run_step_dt`) + T9 `with_time_elapsed` (snap) | T9 `test_with_time_elapsed_is_a_snapping_copy_that_banks_nothing`, D2 |
| 11 | `record_passed_fixings` at the day's spot for every fixing at or before the date (T9) | Changed by W6 D1/D2 + Ruling 4 | Strictly past fixings only, decided at the spot of the first instant they were live; banked by `with_history_settled`: T9 D2, gap, consecutive-fixing tests |
| 12 | Calendar gaps: banked on the next path day at that day's spot, no warning (T9, T10 weekend test) | Changed by W6 D5 (user) | Previous instant's spot, `gap=True`, no warning: T9 `test_a_calendar_gap_records_at_the_previous_instants_spot_without_a_warning`, T10 `test_business_day_path_records_a_saturday_fixing_at_fridays_spot` |
| 13 | A payment due between path days is booked, undiscounted, on the first path day on or after it (T9) | Kept, amended by OQ-1 (b) (Task 8a): booked on the first path instant at which its pay time is strictly past (a pay date ON a path instant: the next instant) | numeric `with_time_elapsed` (via `time_shift`), dated `with_history_settled`: T9 terminal-booking and short tests |
| 14 | Day's paid cash = Σ quantity × PAYMENT (T9) | Moved | T10 `booked_cash` (used by T10, T11) |
| 15 | Pending PV = Σ quantity × `pending_pv` (T9) | Satisfied by W6 (Task 5 hook) | `RunFixingDay.pending_pv` |
| 16 | Other products untouched (T9) | Satisfied by W6 (`participates`) | T9 `test_other_products_are_untouched` |
| 17 | Short position: signed cash and pending PV (T9, Review Focus 1) | Kept | T9 `test_a_short_position_carries_its_signed_quantity`, T10 `test_booked_cash_scales_by_the_signed_quantity_and_ignores_other_entries` |
| 18 | Two pairs age independently (T9) | Kept | T9 `test_two_pairs_record_at_their_own_previous_spots` (each pair's previous spot, W6 D5) |
| 19 | Terminal booking kept at PV(pending), paid on its day (T9) | Kept | T9 `test_a_terminal_booking_is_kept_at_its_pending_pv_and_paid_on_its_day` |
| 20 | D2 twins age identically (T9) | Kept, rewritten on the recorder | T9 `test_d2_twins_age_identically` (live on its day, recorded next day) |
| 21 | Fixtures `drive`, `book` (T9) | Kept | `drive` now replaces the env and calls `RunFixingRecorder.advance` |
| 22 | CHANGELOG `FxTargetRedemptionLifecycle` bullet (T9) | Replaced | T9 protocol / hook bullets |
| 23 | Backtest calls register / advance (T10) | Satisfied by W6 Task 8 | `backtest/fx/engine.py:50-51, 66` |
| 24 | Rows `paid_cash` (cumulative) and `pending_pv`; `portfolio_value`, `final_value` include paid cash (T10) | Kept | T10, fed from `RunFixingDay` |
| 25 | B1 numeric TARN ages; day 91 equals the rebuild (T10) | Kept, same values | T10 `test_b1_numeric_tarn_ages_in_the_backtest` (k/12 gap fixings on a flat path) |
| 26 | B2 redeeming path: cash on pay days, terminal = Σ DF × pending, 0.0 after, no warning (T10) | Kept, same day indices | T10 `test_b2_redeeming_path_books_cash_on_its_pay_days` |
| 27 | B3 quiet path: principal; negative TARF cash (T10) | Kept, same day indices | T10 B3 (the last fixing at t = 1.0 is live on day 365, recorded day 366; principal still day 367) |
| 28 | B4 twins in one backtest (T10) | Kept | T10 `test_b4_twins_in_one_backtest` (spy on `RunFixingRecorder.advance`; also compares `RecordedFixing` rows) |
| 29 | B5 registration rules; backwards path (T10) | Kept | T10 B5 tests (R9 message from W6) |
| 30 | B6 vanilla-only backtest books nothing (T10) | Kept | T10 `test_b6_book_without_target_redemption_books_nothing` + T12 byte gate on the W6 base |
| 31 | Backtest module docstring (T10) | Kept | T10 Step 3 |
| 32 | `test/test_fx_backtest.py` passes unmodified (T10) | Kept | T10 Step 4 (plus `test/test_run_fixings_runs.py` unmodified) |
| 33 | Q8: undated `DayPath` with a TARN / TARF refused before `register` (T11) | Kept | T11 (`isinstance(TargetRedemptionProduct)`, R18); W6 spec §6 keeps it |
| 34 | `DayPath.start_date` before the environment's date → `ValidationError` (T11, Review Focus 4) | Satisfied by W6 R9 | T11 `test_start_date_before_environment_date_raises` |
| 35 | Dynamic scenario calls register / advance (T11) | Satisfied by W6 Task 8 | `dynamicscenario/fx/engine.py:84-85, 98` |
| 36 | `DayResult.paid_cash` / `realized_cash` / `pending_receivable_pv`; value and final value include paid cash (T11) | Kept | T11 |
| 37 | FIXING snapshots: spot, payoff, cashflow (old R6), terminates_position (T11) | Kept | T11 `_fixing_snapshots` from `RunFixingDay.product_events` |
| 38 | Dynamic redeeming path books cash; event days `[31, 61]` (T11) | Kept | T11 `test_dynamic_tarn_redeems_and_books_cash` |
| 39 | Snapshot cashflow when paid on the fixing's banking day (T11) | Kept, renamed | T11 `test_dynamic_snapshot_cashflow_when_paid_in_the_recording_step` |
| 40 | B4 dynamic twins; B6 dynamic (dated and undated) (T11) | Kept | T11 |
| 41 | `LifecycleEventSnapshot` docstring gains `'FIXING'` (T11) | Kept | T11 |

New W6 requirements with no old counterpart: `has_run_fixings`, `open_fixings`, `with_time_elapsed` (T9); t = 0 live
and silent (T9); time-of-day instants refused (T9, W6 R8); same-date refresh (T9, W6 Ruling 9); append-only
`lifecycle_events` (T9); a numeric participant aged to its maturity is exactly 0.0 and equals its dated twin (T10,
W8); stable indices across numeric banking (T9).

### Rulings made in this re-plan (each: what — why — cost if wrong)

- **R13** A generic optional recorder hook `with_history_settled(env)` (`settle_history` in
  `quantark/util/run_fixings.py`) banks recorded history and pays due cash; `with_fixings_recorded` keeps the built
  rates-only semantics — `with_fixings_recorded(fills)` gets no environment, and a dated contract needs the date to bank
  (its `as_of`) and to pay a pending payment whose date has come (the old hook's daily `record_passed_fixings`); the
  W6 hand-off's "`with_fixings_recorded` banks … PAYMENT events" cannot be met for the dated form, and the built
  rates-only form is what keeps the Greek freeze value-neutral — cost: one optional hook in a shared W6 module (two
  call sites); no existing participant moves (they lack the hook); a reviewer may prefer the hook named differently.
- **R14** Protocol indices are booking-relative (`_index_offset` = numeric fixings banked and dropped since
  construction, dead ones included; dated = schedule position) — W6 keys each record's first-live spot by index across
  steps (Ruling 4) and assumes records are never dropped (`RecordedFixing.index` comment); a numeric TARN drops banked
  fixings (spec §6), so with listed positions a fixing that goes live on the instant an earlier one banks would inherit
  the wrong first-live spot (pinned by `test_consecutive_daily_fixings_keep_their_own_live_spots_across_banking`) —
  cost: numeric event labels after a banking read `#<booking index>`; informational only; inception twins share indices.
- **R15** *(SUPERSEDED 2026-10-10 by the user's OQ-3 ruling: banked fixings are tracked per fixing, `as_of` never
  passes the valuation date; see Task 8b and R24.)* ~~`with_history_settled` sets a dated `as_of` to the last banked
  fixing's date, used as a high-water mark of the banked set.~~
- **R16** `matured_before` and every live/history decision use the resolved TIME (`is_history`), not the date — W6
  Ruling 2 (time-based gap flag) and the twin rule (a numeric twin only has times) — cost: only a weekend/holiday
  last fixing on a `BUSINESS_DAYS` environment is affected (live on the run start instead of matured).
- **R17** Run value = Σ position MTM + cumulative `booked_cash` of the steps; the day's cash is summed from 0.0 in
  event order, then added to the running total — the TARN spec §7.2 ledger; fixes the exact float order the tests
  assert — cost: none.
- **R18** Q8 detects the book by `isinstance(product, TargetRedemptionProduct)` on the INPUT portfolio, before cloning —
  the refusal must come before `register` (spec §12 Q8) and needs no clone — cost: a future subclass is refused too
  (intended).
- **R19** A TARN / TARF stays in a run after its last fixing (PV(pending), then exactly 0.0 after its last pay date),
  whereas W6/W8 refuse a range accrual or Asian run past its maturity — TARN spec §4.2 (dated `get_maturity` 0.0 after
  expiry) and §7.2 (pending cash still pays after the last fixing); both twins continue identically (T10 W8 test) —
  cost: none against the TARN spec; differs from the other participants by design.
- **R20** Task 12's B6 and demo references are re-captured from an export of the W6 base (`git archive
  $(merge-base) quantark example`), not taken from Task 1 — Task 1 captured them before the rebase onto W6, and W6
  ages the numeric FX range accruals in the B6 book — cost: B6 then cannot see a W6 defect (W6's own gates cover W6).
- **R21** (Task 8a, OQ-1 b) One cash rule for both forms: booked once `is_history(pay time)`, otherwise pending at
  `snap_time(pay time)`; `PendingPayment` refuses only a non-finite or strictly past `pay_time` (an `is_zero` time,
  either sign, is "today"); a dated pending `pay_date` may equal `as_of` (refused only before it); the dated pay test
  uses the resolved time (R16) — the user's rule needs a boundary, and `is_history` is the one W6 uses for fixings —
  cost: none beyond the user-approved moves listed in Task 8a.
- **R22** *(SUPERSEDED 2026-10-10 by the user's OQ-3 ruling: double-banking is prevented by fixing identity, not by
  date; `as_of` never passes the valuation date; `_check_as_of` stays as built.)* ~~`as_of` = max(valuation date, last
  banked fixing date) and a time-based `_check_as_of`.~~
- **R23** The two follow-ups are numbered **Task 8a / Task 8b** and run before Task 9 (Tasks 9–12 keep their numbers
  and cross-references): Task 9 replaces `_record_dated` wholesale, so the amended cash and passed-by-time rules must
  already be in place; its replacement block carries them — cost: none.
- **R24** (Task 8b, OQ-3 ruling; **USER-APPROVED 2026-10-10**: under `BUSINESS_DAYS`, banking on a non-business
  day writes `as_of` = the preceding business day; never after the valuation date; on calendar clocks the valuation
  date) The `as_of` that banking writes is `_state_date(fx_env)`: the valuation date, moved
  back over the days at zero time from it — valued on a Saturday under `BUSINESS_DAYS` (a weekend row, W6 Ruling 4)
  it is the Friday, the same instant; under `CALENDAR_DAYS` / ACT conventions it is the valuation date itself. Why:
  the ruling keeps the "a pending `pay_date` must not precede `as_of`" check, and on a Saturday row a zero-lag Friday
  fixing is still live with its Friday coupon due at that instant (Task 8a); `as_of` = Saturday would make that state
  invalid and stop every business-day run with weekend rows, while the Friday is the instant the state describes and
  is never after the valuation date — cost: on a non-business valuation date a dated state reads an `as_of` up to a
  weekend or holiday earlier; no price, event or banked set depends on it (banked is by identity). Pinned by Task 8b
  `test_as_of_on_a_weekend_valuation_is_the_friday`.
- **R25** (Task 8b) A dated booking that states only `as_of` (no `fixings_banked`) banks the fixings dated on or before
  it — that is what such a booking meant under spec §4.2, so every existing booking and test keeps its banked set — and
  the "cash needs a banked fixing" rule is stated by count (`as_of is None or banked count == 0` refused), which equals
  the old "`as_of` before the first fixing" refusal for every such booking — cost: none.
- ~~**R26**~~ **Superseded 2026-10-10 by the user's ruling (b)** (section "R26 replaced" below): a dated TARN that
  shares a fixing date no longer needs explicit `accrual_factors`; it states its own accrual period dates
  (`accrual_end_dates`, Task 8d). What survives is user-directed, not a ruling of this plan: under the DEFAULT
  derivation (no period dates, no factors) a shared fixing date is still refused, because the later period would run
  from a fixing date to the same date (a zero-length derived period) — Task 8c builds the refusal, Task 8d narrows it to
  the default derivation.
- **R27** (Task 8d, user ruling (b) on R26) The TARN's accrual period dates are stated as one **accrual END date per
  period**, `accrual_end_dates`, 1:1 with `fixing_dates`; period `i` runs from the previous period's end
  (`initial_date` for `i = 0`) to `accrual_end_dates[i]`, and `initial_date` stays the first period's start. No
  separate start dates — this is the accrual-time spec's own dated rule (§4.3 Endpoints: "`end` is the observation
  date (`OBSERVATION`) or the authoritative settlement date (`SETTLEMENT`). `start` is the previous record's `end`, or
  `initial_date` for the first record") with the end freed from the fixing date, and it keeps the TARN spec's
  contiguity premise (§4.1 consistency rules: "periods are contiguous"); the default (`accrual_end_dates=None` ⇒ each
  end is its fixing date) is then exactly today's derivation, call for call — cost if wrong: a contract with gapped or
  overlapping periods cannot be booked by dates (it still books with `accrual_factors`); adding start dates later is
  additive (OQ-6).
- **R28** (Task 8d) Validation of `accrual_end_dates`, each rule taken from a spec: (i) 1:1 with `fixing_dates`, datetimes
  (as `pay_dates`); (ii) strictly increasing — a period accrues from the previous end, so an equal end is a zero-length
  period, which the TARN refuses everywhere today ("accrual_factors must be positive", both forms) and which the numeric
  twin could not carry; (iii) `fixing_dates[i] <= accrual_end_dates[i] <= pay_dates[i]` — the accrual-time spec defines
  exactly two accrual ends, the observation (`OBSERVATION`) and the settlement date (`SETTLEMENT`, §4.5), so an end in
  that window is defined and an end before its fixing or after its payment is not (OQ-5); (iv) `accrual_end_dates`
  requires `initial_date` (the first start) and excludes `accrual_factors` (exactly one period source, as the TARN spec
  §4.2 already rules for `initial_date` / `accrual_factors`); `day_count_convention` applies to it as to the default.
  The year fraction is the TARN spec §4.2 call unchanged — `calculate_year_fraction(start, end, day_count_convention,
  fx_env.bus_days_in_year, calendar=fx_env.calendar)` — so a `BUSINESS_DAYS` count still reads the environment's
  calendar (TARN spec §4.2; the accrual-time spec's frozen basis terms, §4.7, are out of its own scope for the FX TARN,
  §3 item 10 and §9) — cost if wrong: (ii)/(iii) refuse a booking that a future ruling could accept; nothing that
  constructs today changes.
- **Superseded** old R3 (clock origin): now W6 R6, same rule, held by the recorder. Old **R7** (refuse a dated future
  fixing or pay date at zero time): **superseded by the user's OQ-2 (b) answer, 2026-10-10 — reversed by Task 8b**.

### Conflicts between the TARN spec and W6

Resolved by user-approved W6 decisions (no question needed; listed in Task 12's sign-off text):
- **Gap fixings** (TARN §7.2: next instant's spot) → previous instant's spot, `gap=True` (W6 D5, spec §6/§9).
- **Origin-day fixing** (TARN §7.2 register step 3: banked silently at registration) → live, recorded on the next
  instant (W6 D1, spec §6).
- **Today's fixing under Greeks** (built Task 6/7 freeze recorded a t = 0 fixing at the base spot) → live under bumps,
  silent at pricing (W6 D1, hand-off note). Moves Greeks of a TARN / TARF valued ON a fixing date.
- **Time-of-day run instants** → refused for TARN / TARF too (W6 D6).
- **Undated paths** → W6 R7 never advances; TARN keeps its Q8 refusal (W6 spec §6).

Interface conflicts resolved by rulings above (no value moves): W6 hand-off "`with_fixings_recorded` banks" → R13;
"records are never dropped" → R14; W6/W8 refusal past maturity vs TARN continuing → R19.

### Answered 2026-10-10 by the user (both (b))

- **OQ-1 — cash due on the day of a live fixing: (b).** The conflict: TARN spec §3 / §5.1 booked cash whose pay time is
  at or before valuation (out of the value), while W6 records a fixing dated day d only on day d+1; a zero-lag dated
  TARN's run value dipped by one coupon on each path-aligned fixing day (measured: 101.1173 with 0.0 paid on the fixing
  day, 101.1319 + 0.9863 paid the day after), and a zero-lag fixing on the run's first day showed a false +coupon.
  **Decision:** cash due today stays in today's value and is booked once its pay time is strictly past, as W8 (a dated
  contract on its date prices its settled value) and W6 D1 (t = 0 is live). **Implemented by Task 8a** (re-opens Tasks
  2, 5, 6, 7; engines unchanged since DF(0) = 1). Moved values (Task 8a's pins): the price on a pay date gains the cash
  due that day (TARN zero-lag 101.115629 → 102.101931), theta the day before a pay date gains it (−0.977801 →
  0.008500), theta on the pay date loses it (0.015260 → −0.971042); full table in Task 8a. Pins flipped: T9
  `test_zero_lag_cash_of_a_live_fixing_stays_in_the_value_until_recorded`, T10
  `test_zero_lag_fixing_day_value_keeps_the_coupon_no_dip` (no dip); T9 terminal-booking and settle tests, T10 B3 and
  W8 now book a path-aligned payment on the instant after its pay date.
- **OQ-2 — a dated fixing or pay date at zero business time: (b).** The conflict: Ruling R7 (Task 7) refused to price a
  dated fixing or pay date after the valuation date that resolves to t = 0 (a Saturday under `BUSINESS_DAYS` on the
  Friday), while W6 Ruling 2 and the numeric twin treat it as live, so business-day runs stopped on the Friday.
  **Decision:** it is LIVE, like its numeric twin and the FX range accrual; R7 is reversed. **Implemented by Task 8b**
  (re-opens Task 7: the two `zero year fraction` tests and the CHANGELOG bullet flip; new twin test and a
  bank-once test). Pins flipped: T9 `test_a_weekend_fixing_on_a_business_day_clock_is_live_on_friday`; new run tests
  T10 `test_business_day_clock_weekend_fixing_twins_in_a_backtest` and T11
  `test_business_day_clock_weekend_fixing_twins_in_a_dynamic_scenario`.

**Task order now:** 1–8 (done) → **8a** (OQ-1 b) → **8b** (OQ-2 b) → 9 (run-fixings protocol) → 10 (FX backtest
cash) → 11 (FX dynamic cash, Q8) → 12 (final gate).

### OQ-3 answered by a user ruling (2026-10-10, replacing the options offered)

- **Ruling:** the TARN records banked fixings PER FIXING (by schedule index), as the FX range accrual does through W6's
  recorder (each fixing recorded once, at its first live instant's spot, W6 Ruling 4). A weekend fixing inherits
  Friday's spot (live on the Friday on a `BUSINESS_DAYS` clock; weekend rows carry Friday's spot on a calendar clock).
  `as_of` never moves past the valuation date (R15, R22 superseded). Double-banking is prevented by fixing identity.
  No validation check is dropped (the "pending `pay_date` must not precede `as_of`" check stays). A Monday / following
  convention is NOT adopted (separate future decision).
- **Implemented by Task 8b (rewritten):** `TargetRedemptionState.fixings_banked` (Task 2), `_banked_count` /
  `_is_unbanked` by identity / `is_terminal` by count / `_record_dated` writing the count and `as_of =
  _state_date(fx_env)` (Task 7); numeric form unchanged — its banked fixings are the dropped ones, counted by Task 9's
  `_index_offset` (R14), and Task 9's `fixings_banked` property reports both forms alike (asserted in D2). Tests: the
  Friday + Saturday case prices on the Friday in both forms (dated == numeric, state and payments; the numeric twin is
  booked on the Friday; Task 8c adds the Thursday) and Monday does not bank the Saturday fixing again (Task 8b
  `test_oq3_friday_and_saturday_fixings_price_on_friday_and_bank_once`; in a run, Task 9
  `test_oq3_friday_and_saturday_fixings_in_a_run_are_recorded_once_at_fridays_spot`). Moved values: none beyond Tasks
  8a / 8b's (a dated state now states its count — not a value).

**Task order now:** 1–8 (done) → **8a** (OQ-1 b) → **8b** (OQ-2 b + OQ-3 ruling) → **8c** (OQ-4 b) → 9 → 10 → 11
→ 12.

### OQ-4 answered 2026-10-10: (b), generalised by the user

- **Question:** two scheduled fixings at one resolved time (the OQ-3 Friday + Saturday pair valued before the Friday)
  could not be priced in either form (the path generator needs strictly increasing times; the numeric form refused
  equal times).
- **Decision:** "two period dates can share the same fixing date": equal fixing DATES and equal fixing TIMES are a
  legitimate schedule; both engines simulate each distinct time once and every period sharing it fixes at that spot;
  target accumulation and redemption run through the periods in schedule order; each period keeps its own pay date and
  cash; per-fixing banking banks every period sharing the date together.
- **Spec rule for a period after the hit on the same date (cited, not invented):** TARN spec §6,
  `record_passed_fixings`, TARN bullet: "On a hit (`is_redeemed`), the principal (with `include_principal`) at `pay_i`
  is handled the same way and every later fixing is dead", applied to the fixings "in order, stopping once the
  contract is terminal"; TARF bullet: "the hit rule as for the TARN. On a final hit the geared unfavourable leg is still
  paid" (the hitting period's own leg); spec §5.2: the engines' `alive` mask. A later period on the same date is a later
  fixing in schedule order, so it is dead: no coupon, no cash, no FIXING event; the principal pays at the hitting
  period's `pay_i`. No open question.
- **Implemented by Task 8c** (re-opens Tasks 3, 4, 6, 7). Moved values: contracts with equal fixing
  times / dates (and dated dates that resolve to one time, valued before it) go from `ValidationError` to a price equal
  to their twin; nothing else moves (a schedule of distinct times takes exactly the old engine path).
  `test_r2_dated_refusals[fixing_dates=FIX_DATES[:1] * 2]` still raises, now by the default derivation's zero-length
  refusal instead of uniqueness (it states no period dates and no factors).

### R26 replaced 2026-10-10: option (b), the TARN's own accrual period dates (user ruling)

- **Ruling (user):** the TARN gets its own accrual PERIOD dates, separate from its fixing dates, as the accrual-time
  spec (`2026-10-08-accrual-time-design.md`) separated accrual from observation for autocallables. Each period accrues
  over its own period dates in the coupon day count; the fixing date is a separate date; any number of periods may
  share one fixing date, and none needs explicit `accrual_factors`. Default (no period dates): today's derivation
  (`initial_date` / previous fixing date → this fixing date), so every existing booking keeps its exact meaning and
  value. Explicit `accrual_factors` stay an alternative. Pay dates stay per period. R26 is superseded.
- **Design chosen (Rulings R27, R28):** one new keyword, `accrual_end_dates: Optional[List[datetime]]` on
  `FxTargetRedemptionNote` (dated form only; the TARF has no accrual term, TARN spec §2.3). Naming follows the
  accrual-time spec ("`end`: the accrual end", `AccrualEndType`, `AccrualConfig.accrual_end`); `initial_date` keeps its
  meaning (the first period's start, the accrual-time spec's `origin`). Period `i` = [`E_{i−1}`, `E_i`] with
  `E_{−1} = initial_date` and `E_i = accrual_end_dates[i]`, or `fixing_dates[i]` when `accrual_end_dates is None`.
- **Precedence and exclusivity (dated TARN):** exactly one period SOURCE —
  1. `accrual_factors` (explicit): excludes `initial_date`, `accrual_end_dates` and `day_count_convention` (the first
     and last as today);
  2. `initial_date` + `accrual_end_dates`: each period over its own dates;
  3. `initial_date` alone: the default, ends = fixing dates (today's derivation, bitwise).
  `accrual_end_dates` without `initial_date` is refused ("needs initial_date"); with `accrual_factors` it is refused
  ("cannot be combined"); neither source is refused as today. The numeric form refuses `accrual_end_dates` ("belongs to
  the dated form"). There is no override layering: a contract states one source, and that source gives every factor.
- **Numeric form and the twin: nothing new.** The numeric TARN already takes per-period `accrual_factors`, and
  `_numeric_twin` builds it with `accrual_factors=list(view.accrual_factors)` from `dated.resolve(env)`, which reads
  `_dated_factor` — so a dated contract with period dates hands its twin exactly its per-period factors (pinned by
  Task 8d), with its per-period pay times.
- **Shared fixing date under the default derivation** stays refused (zero-length derived period) unless the contract
  states `accrual_end_dates` or `accrual_factors` (Task 8c builds the refusal; Task 8d narrows it and names both cures in
  its message).
- **Implemented by Task 8d.** Moved values: none for existing bookings (`accrual_end_dates=None` is today's call with
  today's dates; every existing test passes unmodified); new bookings (`accrual_end_dates` given, incl. periods sharing a
  fixing date without factors) go from `TypeError` (unknown keyword) to priced.

**Open questions from this ruling (non-blocking: Task 8d refuses both cases, so no value depends on the answer; a
"yes" is additive):**
- **OQ-5 — an accrual end outside [its fixing date, its pay date].** An end BEFORE its fixing (the coupon's period
  closes, the rate is fixed later — a fixing lag) or AFTER its pay date (paid before it finishes accruing) is defined by
  neither spec: the accrual-time spec's two ends are the observation and the settlement date, and the TARN spec's
  derived end is the fixing date. Task 8d refuses both (R28 iii). **Recommendation: keep the refusal** until a term
  sheet needs it — such a booking still books with explicit `accrual_factors`; lifting it later needs no engine change
  (the factor is known at construction either way).
- **OQ-6 — non-contiguous periods (gaps or overlaps, i.e. explicit start dates).** The TARN spec's premise is "periods
  are contiguous" (§4.1) and the accrual-time spec derives each start from the previous end; neither defines a gap or an
  overlap. Task 8d takes only end dates (R27). **Recommendation: no `accrual_start_dates` until a contract needs one**;
  `accrual_factors` books such a contract meanwhile.

**Task order now:** 1–8 (done) → **8a** (OQ-1 b) → **8b** (OQ-2 b + OQ-3 ruling) → **8c** (OQ-4 b) → **8d** (R26 (b):
accrual period dates) → 9 → 10 → 11 → 12.

No blocking open question remains from this re-plan (OQ-5 / OQ-6 above are non-blocking).

## Review Focus

Updated by the re-plan 2026-10-10 (items 1, 2, 4 retargeted to W6; 6 new) and by the user's OQ-1 / OQ-2 answers
(items 5, 7), the OQ-3 ruling (item 8), the OQ-4 (b) answer (item 9) and the R26 (b) ruling (item 10).

1. **A short position (negative quantity)** in a run: the recorder reports events and pending PV with the signed
   quantity, and paid cash scales by it — pinned by Task 9 `test_a_short_position_carries_its_signed_quantity` and
   Task 10 `test_booked_cash_scales_by_the_signed_quantity_and_ignores_other_entries`.
2. **A run calendar that skips a fixing date** (business-day backtest index, CALENDAR_DAYS environment, fixing on a
   Saturday): recorded on Monday at FRIDAY's spot, `gap=True`, no warning, numeric twin identical (W6 D5) — pinned by
   Task 10 `test_business_day_path_records_a_saturday_fixing_at_fridays_spot`.
3. **An environment with `spot_days > 0`**: banking uses the quoted `fx_env.spot`, not `effective_spot()` (Q3; W6 R1
   records the quoted spot too) — pinned by Task 5 `test_banking_uses_quoted_spot_not_effective_spot`.
4. **A dynamic scenario whose `DayPath.start_date` is before the environment's own date**: `ValidationError` from the
   recorder (W6 R9: the clock cannot go back), never a silent re-age — pinned by Task 11
   `test_start_date_before_environment_date_raises`.
5. **A dated future fixing or pay date that resolves to zero time** under `BUSINESS_DAYS` (a Saturday, valued on the
   Friday): live, priced like its numeric twin, banked once (by fixing identity), recorded on Monday at Friday's spot
   (OQ-2 b, Task 8b) — pinned by Task 8b `test_future_fixing_at_zero_business_time_is_live_like_its_twin` and
   `test_a_weekend_fixing_banked_on_friday_is_not_banked_again`, Task 9
   `test_a_weekend_fixing_on_a_business_day_clock_is_live_on_friday`, Tasks 10 / 11
   `test_business_day_clock_weekend_fixing_twins_in_a_backtest` / `..._in_a_dynamic_scenario`.
6. **A numeric fixing that goes live on the instant an earlier fixing banks** (fixings on consecutive path days): the
   later fixing keeps its own first-live spot and `gap=False` although the numeric contract dropped the earlier one
   (Ruling R14) — pinned by Task 9 `test_consecutive_daily_fixings_keep_their_own_live_spots_across_banking`.
7. **A pay date on a valuation instant** (a dated TARN's default zero-lag `pay_dates`, or a numeric pay time aged to
   exactly 0.0): the cash due that day stays in the value and is booked the next instant — no dip, no false first-day
   gain (OQ-1 b, Task 8a) — pinned by Task 8a `test_moved_values_around_a_pay_date` and
   `test_numeric_cash_due_today_stays_pending_until_its_pay_time_is_past`, Task 9
   `test_zero_lag_cash_of_a_live_fixing_stays_in_the_value_until_recorded`, Task 10
   `test_zero_lag_fixing_day_value_keeps_the_coupon_no_dip`.
8. **Fixings on a Friday AND the following Saturday (zero lag) under `BUSINESS_DAYS`** (OQ-3 ruling): on the Friday
   both price and bank in both forms with `as_of` = Friday, on Monday nothing is banked again; valued on a Saturday
   row, `as_of` is the Friday (R24) — pinned by Task 8b `test_oq3_friday_and_saturday_fixings_price_on_friday_and_bank_once`,
   `test_as_of_on_a_weekend_valuation_is_the_friday`, Task 9
   `test_oq3_friday_and_saturday_fixings_in_a_run_are_recorded_once_at_fridays_spot`.
9. **Periods sharing a fixing date or time** (OQ-4 b): one simulated spot per distinct time, periods in schedule order,
   a period after the hit on the same date dead (spec §6), each period its own pay date and cash, banked together once;
   the Friday + Saturday pair prices on the Thursday like its twin — pinned by Task 8c
   `test_engines_simulate_a_shared_time_once`, `test_two_periods_on_one_fixing_date_price_like_the_twin_and_each_is_paid`,
   `test_a_target_hit_by_the_first_of_two_same_date_periods_kills_the_second`,
   `test_friday_and_saturday_fixings_price_on_the_thursday_like_the_twin`, and Task 9
   `test_periods_sharing_a_fixing_date_are_recorded_and_banked_once_in_a_run`.
10. **A TARN whose accrual periods are not its fixing intervals** (R26 (b), `accrual_end_dates`): stating the fixing
   dates as the ends reproduces the default bitwise (factors and price, calendar and business-day counts); two periods
   sharing a fixing date with their own period dates accrue their own factors, price like the numeric twin (which
   carries those factors) and bank together with their own cash and pay dates; a shared fixing date under the default
   derivation is still refused; ends that are not strictly increasing, outside [fixing, pay], unaligned, without
   `initial_date`, with `accrual_factors` or on the numeric form are refused — pinned by Task 8d
   `test_accrual_end_dates_on_the_fixing_dates_reproduce_the_default_bitwise`,
   `test_two_periods_sharing_a_fixing_date_accrue_their_own_period_dates_like_the_twin`,
   `test_a_shared_fixing_date_needs_period_dates_or_factors`, `test_accrual_end_dates_refusals`,
   `test_accrual_end_dates_belong_to_the_dated_tarn_only`.
---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `quantark/asset/fx/product/option/target_redemption_state.py` | Create (T2), extend (T8a, T8b `fixings_banked`, T10) | `TARGET_HIT_RTOL`, `target_hit_tol`, `is_target_hit`, `PendingPayment`, `TargetRedemptionState`, `TerminalBookingWarning`, `TargetRedemptionEventType`, `TargetRedemptionEvent`, `ResolvedTargetRedemption`, `pending_present_value`; `booked_cash` (T10) |
| `quantark/asset/fx/product/option/target_redemption_base.py` | Create (T3), extend (T5, T6, T7, T8a, T8b, T8c, T9) | `TargetRedemptionProduct(BaseFxProduct)`: forms, validation, terminal warning, `time_shift`, `record_passed_fixings`, `resolve`, `pending_pv`, dated `get_maturity`/`get_delivery`; W5 fill protocol; run-fixings protocol (T9: `has_run_fixings`, `open_fixings`, `with_time_elapsed`, `with_history_settled`, `matured_before`, booking-relative indices); helpers `is_passed_time`, `event_label`, `signed_year_fraction` |
| `quantark/asset/fx/product/option/fx_target_redemption_note.py` | Modify (T3, T5, T7, T8c default-derivation refusal, T8d `accrual_end_dates`) | TARN terms, period sources (`accrual_factors` / `initial_date` [+ `accrual_end_dates`]), `coupon_amount`, `_fixing_outcome`, `_accrual_end`, `_dated_factor` |
| `quantark/asset/fx/product/option/fx_target_redemption_forward.py` | Modify (T4, T5, T7, T8c docstring) | TARF terms, `_fixing_outcome` |
| `quantark/asset/fx/product/option/__init__.py` | Modify (T2) | export the state names |
| `quantark/asset/fx/engine/mc/fx_tarn_note_mc_engine.py`, `fx_tarf_mc_engine.py` | Modify (T6, T8c: one spot per distinct time) | read `resolve`; terminal; start `acc`; pending PV; product hit tolerance |
| `quantark/asset/fx/engine/base_fx_engine.py` | Modify (T8) | Q7 numeric theta step |
| `quantark/util/run_fixings.py` | Modify (T9) | optional hook `with_history_settled(env)` via `settle_history`, at registration and after each clock move's recording (re-plan Ruling R13). (`quantark/asset/fx/lifecycle/` is NOT created: W6 spec §6) |
| `quantark/backtest/fx/engine.py` | Modify (T10) | paid cash ledger and `paid_cash` / `pending_pv` columns from `RunFixingDay` (register / advance are W6's) |
| `quantark/dynamicscenario/fx/engine.py`, `quantark/dynamicscenario/results/dynamic_results.py` | Modify (T11) | Q8 refusal, cash fields and FIXING snapshots from `RunFixingDay` (register / advance are W6's) |
| `test/test_fx_target_redemption_state.py` | Create (T2), modify (T8a, T8b) | state module; cash due today (T8a); `fixings_banked` (T8b) |
| `test/fx_target_redemption_fixtures.py` | Create (T3), extend (T7, T9) | shared env/product builders, `_numeric_twin`, `book`, `drive` |
| `test/test_fx_target_redemption_lifecycle.py` | Create (T3), extend (T4–T9, incl. T8a / T8b flips, T8c shared dates, T8d accrual period dates) | R1, R2, P1–P5, D1–D5, T1, T2, WARN1–3; shared fixing dates (T8c); `accrual_end_dates` (T8d); run-fixings protocol unit tests (T9) |
| `test/test_fx_target_redemption_runs.py` | Create (T9), extend (T10, T11) | recorder-level TARN/TARF rules, B1–B6, Q8, W8 twins, OQ-1 / OQ-2 (b) run tests |
| `test/test_run_fixings.py` | Modify (T9) | one appended generic test of the settle hook (W6's file, otherwise untouched) |
| `test/test_fx_tarn_note_mc.py`, `test/test_fx_tarf_mc.py`, `test/execution/matrix_fixtures.py`, `example/fx_tarn_demo.py` | Modify (T3, T4) | constructor arguments only (C1) |
| `docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/` | Create (T1, T8, T12) | `w4b_baseline.py`, `q7_theta_plugin.py`, `q7_summarize.py`, `moved_values.txt` |

---

### Task 1: Prerequisite check and main baselines (no library code)

**Files:**
- Create: `docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/w4b_baseline.py`
- Output (untracked, never committed): `<WT>/.superpowers/w4b/baseline.json`, `<WT>/.superpowers/w4b/demo_<name>.main.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: `baseline.json` (B6 rows on main) and the demo stdouts, read by Task 12; script `w4b_baseline.py capture|compare <json>`.

- [ ] **Step 1: Verify the prerequisites are on this branch's base**

Run:
```bash
git -C <WT> log --oneline -1
env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -c "
from quantark.util.fixings import MissingFixingWarning, fixing_label, warn_filled_fixings
from quantark.asset.equity.lifecycle.autocallable import MaturedBeforeRunWarning
print('W5 and W1 present')"
```
Expected: `W5 and W1 present`. If either import fails, STOP: the branch was cut before W5/W1 landed.

- [ ] **Step 2: Write the baseline script**

```python
"""W4b B6 byte gate (plan 2026-10-09-fx-tarn-lifecycle, Tasks 1 and 12).

capture <json>: run on the implementation branch BEFORE any library change (== main) and store the FX backtest and FX
dynamic-scenario rows of a book holding every M7 FX product except TARN/TARF.
compare <json>: run on the finished branch; every stored number must match to the bit, and the new cash columns must
be exactly 0.0.

Run: env PYTHONPATH=<WT>:<WT>/test <venv>/bin/python <this file> capture|compare <WT>/.superpowers/w4b/baseline.json
"""
import json
import math
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from quantark.asset.fx.engine.analytical import (
    FxDigitalOptionAnalyticalEngine, FxForeignRangeAccrualAnalyticalEngine, FxQuantoDigitalAnalyticalEngine,
    FxQuantoRangeAccrualAnalyticalEngine, GarmanKohlhagenEngine, GarmanKohlhagenQuantoEngine,
)
from quantark.asset.fx.engine.analytical.fx_range_accrual_engine import FxRangeAccrualAnalyticalEngine
from quantark.asset.fx.engine.mc import FxBarrierMCEngine, FxSharkfinMCEngine
from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import (
    FxBarrierOption, FxDigitalOption, FxForeignRangeAccrualOption, FxQuantoDigitalOption, FxQuantoRangeAccrualOption,
    FxQuantoVanillaOption, FxRangeAccrualConfig, FxRangeAccrualOption, FxSharkfinOption, FxVanillaOption,
)
from quantark.backtest.fx import FXBacktestConfig, FXBacktestEngine
from quantark.backtest.strategy.fx_delta_neutral_strategy import FXDeltaNeutralStrategy
from quantark.dynamicscenario import FXDynamicScenarioConfig, FXDynamicScenarioEngine
from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.fx import FXPortfolio
from quantark.priceenv import FxPricingEnvironment
from quantark.priceenv.fx_pricing_environment import FxQuantoMarketData, QuantoConversionOrientation
from quantark.stresstest.stress.stress_types import StressType
from quantark.util.enum import FxBarrierType, OptionType

warnings.simplefilter("ignore")
PAIR = CurrencyPair("EUR", "USD")
V = datetime(2026, 6, 15)
FIX = [k / 12 for k in range(1, 13)]
NEW_BACKTEST_KEYS = {"paid_cash", "pending_pv"}


def env():
    return FxPricingEnvironment(
        valuation_date=V, spot_quote=SpotQuote(spot=1.20), domestic_curve=FlatRateCurve(rate=0.05),
        foreign_curve=FlatRateCurve(rate=0.03), vol_surface=FlatVolSurface(volatility=0.10),
        quanto=FxQuantoMarketData(settlement_curve=FlatRateCurve(rate=0.02), quanto_vol=0.08, correlation=0.3,
                                  conversion_orientation=QuantoConversionOrientation.SETTLEMENT_PER_DOMESTIC))


def book():
    mc = FxMCParams(num_paths=5_000, seed=7, time_steps=24)
    cfg = FxRangeAccrualConfig(upper_barrier=1.30, lower_barrier=1.10, accrual_rate=0.04)
    rows = [
        (FxVanillaOption(strike=1.20, option_type=OptionType.CALL, maturity=1.0, notional_foreign=1.0,
                         currency_pair=PAIR), GarmanKohlhagenEngine()),
        (FxVanillaOption(strike=1.20, option_type=OptionType.CALL, expiry_date=V + timedelta(days=365),
                         notional_foreign=1.0, currency_pair=PAIR), GarmanKohlhagenEngine()),
        (FxDigitalOption(strike=1.20, option_type=OptionType.CALL, payout=1.0, maturity=1.0, currency_pair=PAIR),
         FxDigitalOptionAnalyticalEngine()),
        (FxQuantoVanillaOption(strike=1.20, option_type=OptionType.CALL, quanto_fx_rate=1.0, maturity=1.0,
                               currency_pair=PAIR, notional_foreign=1.0), GarmanKohlhagenQuantoEngine()),
        (FxBarrierOption(strike=1.20, barrier=1.35, is_up=True, knock_type=FxBarrierType.KNOCK_OUT,
                         option_type=OptionType.CALL, maturity=1.0, currency_pair=PAIR), FxBarrierMCEngine(mc)),
        (FxSharkfinOption(strike=1.20, barrier=1.35, is_up=True, option_type=OptionType.CALL, maturity=1.0,
                          currency_pair=PAIR), FxSharkfinMCEngine(mc)),
        (FxRangeAccrualOption(notional=1.0, range_config=cfg, maturity=1.0, observation_times=FIX,
                              currency_pair=PAIR), FxRangeAccrualAnalyticalEngine()),
        (FxQuantoDigitalOption(strike=1.20, option_type=OptionType.CALL, payout=1.0, quanto_fx_rate=1.0,
                               maturity=1.0, currency_pair=PAIR), FxQuantoDigitalAnalyticalEngine()),
        (FxQuantoRangeAccrualOption(notional=1.0, range_config=cfg, quanto_fx_rate=1.0, maturity=1.0,
                                    observation_times=FIX, currency_pair=PAIR),
         FxQuantoRangeAccrualAnalyticalEngine()),
        (FxForeignRangeAccrualOption(notional=1.0, range_config=cfg, maturity=1.0, observation_times=FIX,
                                     currency_pair=PAIR), FxForeignRangeAccrualAnalyticalEngine()),
    ]
    pf = FXPortfolio(portfolio_name="b6", pricing_environments={"EURUSD": env()},
                     creation_date=datetime(2026, 1, 1))
    for prod, eng in rows:
        pf.add_position(product=prod, quantity=1.0, entry_price=0.0, underlying="EURUSD", engine=eng,
                        entry_timestamp=datetime(2026, 1, 1))
    return pf


def _plain(v):
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    if isinstance(v, (np.floating, np.bool_)):
        return v.item()
    return v


def backtest(days, greeks):
    rng = np.random.default_rng(3)
    idx = pd.date_range(V, periods=days, freq="D")
    spot = 1.20 * np.exp(np.cumsum(rng.normal(0.0, 0.005, days)))
    path = pd.DataFrame({"EURUSD_spot": spot, "EURUSD_vol": 0.10, "EURUSD_dom_rate": 0.05,
                         "EURUSD_for_rate": 0.03}, index=idx)
    res = FXBacktestEngine(FXBacktestConfig(portfolio=book(), market_path=path, strategy=FXDeltaNeutralStrategy(),
                                            calculate_greeks=greeks)).run()
    return {"rows": [{k: _plain(v) for k, v in r.items()} for r in res.rows],
            "initial_value": res.initial_value, "final_value": res.final_value, "num_hedges": res.num_hedges,
            "total_hedge_pnl": res.total_hedge_pnl}


def dynamic(start):
    steps = [DayStep(day_index=i, changes=[ParameterChange("spot", StressType.PERCENTAGE, 0.001)])
             for i in range(30)]
    r = FXDynamicScenarioEngine(FXDynamicScenarioConfig(calculate_greeks=True)).run(
        book(), DayPath(name="b6", steps=steps, start_date=start))
    return {"baseline": r.baseline_value, "final": r.final_value, "total_pnl": r.total_pnl,
            "days": [{"portfolio_value": d.portfolio_value, "daily_pnl": d.daily_pnl,
                      "cumulative_pnl": d.cumulative_pnl, "net_pnl": d.net_pnl,
                      "greeks": {k: _plain(v) for k, v in d.greeks.items()},
                      "paid_cash": d.paid_cash, "realized_cash": d.realized_cash,
                      "pending_receivable_pv": d.pending_receivable_pv,
                      "n_lifecycle_events": len(d.lifecycle_events)} for d in r.day_results]}


CASES = {
    "backtest_92_no_greeks": lambda: backtest(92, False),
    "backtest_20_greeks": lambda: backtest(20, True),
    "dynamic_dated": lambda: dynamic(V),
    "dynamic_undated": lambda: dynamic(None),
}


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _diff(path, base, got, out):
    if isinstance(base, dict):
        for k, v in base.items():
            if k not in got:
                out.append(f"{path}.{k}: missing")
            else:
                _diff(f"{path}.{k}", v, got[k], out)
        for k in set(got) - set(base):
            if k in NEW_BACKTEST_KEYS and got[k] == 0.0:
                continue
            out.append(f"{path}.{k}: new key with value {got[k]!r}")
    elif isinstance(base, list):
        if len(base) != len(got):
            out.append(f"{path}: length {len(base)} -> {len(got)}")
        for i, (x, y) in enumerate(zip(base, got)):
            _diff(f"{path}[{i}]", x, y, out)
    elif not _same(base, got):
        out.append(f"{path}: {base!r} -> {got!r}")


def main():
    mode, target = sys.argv[1], sys.argv[2]
    got = json.loads(json.dumps({name: fn() for name, fn in CASES.items()}))
    if mode == "capture":
        with open(target, "w") as fh:
            json.dump(got, fh, indent=1)
        print(f"captured {len(got)} cases to {target}")
        return
    with open(target) as fh:
        base = json.load(fh)
    diffs = []
    _diff("", base, got, diffs)
    if diffs:
        print("B6 FAIL\n" + "\n".join(diffs[:200]))
        sys.exit(1)
    print("B6 OK: every stored number bitwise equal; new cash columns 0.0")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Capture the baselines on the untouched branch**

Run:
```bash
mkdir -p <WT>/.superpowers/w4b
env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python <WT>/docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/w4b_baseline.py capture <WT>/.superpowers/w4b/baseline.json
for f in fx_tarn_demo fx_backtest_demo fx_dynamic_scenario_demo fx_portfolio_risk_demo; do
  env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python <WT>/example/$f.py > <WT>/.superpowers/w4b/demo_$f.main.txt 2>&1; echo "$f exit $?"; done
env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python <WT>/docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/w4b_baseline.py compare <WT>/.superpowers/w4b/baseline.json
```
Expected: `captured 4 cases ...`, four `exit 0`, then `B6 OK ...` (the script is self-consistent before any change; a
FAIL here means a nondeterministic case — fix the script, not the library).

- [ ] **Step 4: Confirm the T1 reference values on this base**

Run (values were measured on `dcea77b6`; W5/W1 do not touch these engines, so they must be identical):
```bash
env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python - <<'EOF'
from datetime import datetime
from quantark.asset.fx.engine.mc import FxTargetRedemptionNoteMCEngine, FxTarnForwardMCEngine
from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxTargetRedemptionForward, FxTargetRedemptionNote
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment
P = CurrencyPair("EUR", "USD"); FIX = [2 / 365] + [k / 12 for k in range(2, 13)]
e = FxPricingEnvironment(valuation_date=datetime(2026, 6, 15), spot_quote=SpotQuote(spot=1.20), domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.03), vol_surface=FlatVolSurface(volatility=0.10))
gn = FxTargetRedemptionNoteMCEngine(FxMCParams(num_paths=20_000, seed=7)).calculate_greeks(FxTargetRedemptionNote(fixing_times=FIX, coupon_rate=0.12, notional=100.0, strike=1.10, target=3.0, currency_pair=P), e)
gf = FxTarnForwardMCEngine(FxMCParams(num_paths=20_000, seed=7)).calculate_greeks(FxTargetRedemptionForward(strike=1.15, fixing_times=FIX, target=0.05, currency_pair=P), e)
print(repr(gn["price"]), repr(gn["theta"]), repr(gf["price"]), repr(gf["theta"]))
EOF
```
Expected exactly: `101.63518163030555 0.01686139016959487 0.01925287164907336 0.003076911159477322`. If not, STOP and
report (Task 6's T1 test hard-codes these).

- [ ] **Step 5: Commit the script**

```bash
git -C <WT> add -f docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/w4b_baseline.py
git -C <WT> commit -m "test(fx-tarn): B6 byte-gate baseline script

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `TargetRedemptionState` module

Spec §3.

**Files:**
- Create: `quantark/asset/fx/product/option/target_redemption_state.py`
- Modify: `quantark/asset/fx/product/option/__init__.py`, `CHANGELOG.md`
- Test: `test/test_fx_target_redemption_state.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (exact):
  - `TARGET_HIT_RTOL: Final[float] = 1e-12`
  - `target_hit_tol(target: float) -> float` (0.0 for `inf`, else `abs(target) * TARGET_HIT_RTOL`)
  - `is_target_hit(accumulated: float, target: float) -> bool` (`accumulated >= target - target_hit_tol(target)`)
  - `PendingPayment(amount: float, pay_time: Optional[float] = None, pay_date: Optional[datetime] = None)` frozen
  - `TargetRedemptionState(accumulated: float, pending: Tuple[PendingPayment, ...] = (), as_of: Optional[datetime] = None)` frozen
  - `TerminalBookingWarning(UserWarning)`
  - `TargetRedemptionEventType` enum: `FIXING = "fixing"`, `PAYMENT = "payment"`
  - `TargetRedemptionEvent(event_type, amount, fixing=None, time=0.0, redeemed=False, label="")` frozen
  - `ResolvedTargetRedemption(fixing_times: Tuple[float, ...], pay_times: Tuple[float, ...], accrual_factors: Optional[Tuple[float, ...]], accumulated: float, pending: Tuple[Tuple[float, float], ...], redeemed: bool, terminal: bool)` frozen; `pending` holds `(pay_time, amount)`
  - `pending_present_value(pending: Sequence[Tuple[float, float]], fx_env) -> float` (`pv = 0.0; pv += amount * fx_env.get_domestic_df(t)` in order)

- [ ] **Step 1: Write the failing tests**

```python
"""TargetRedemptionState (spec 2026-10-09-fx-tarn-lifecycle-design.md section 3)."""
import math
from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from quantark.asset.fx.product.option.target_redemption_state import (
    TARGET_HIT_RTOL,
    PendingPayment,
    ResolvedTargetRedemption,
    TargetRedemptionEvent,
    TargetRedemptionEventType,
    TargetRedemptionState,
    TerminalBookingWarning,
    is_target_hit,
    pending_present_value,
    target_hit_tol,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment
from quantark.util.exceptions import ValidationError

D1, D2, D3 = datetime(2026, 7, 15), datetime(2026, 8, 15), datetime(2026, 9, 15)


def test_hit_rtol_keeps_the_engines_value():
    assert TARGET_HIT_RTOL == 1e-12


def test_hit_tolerance_is_the_engine_rule():
    assert target_hit_tol(math.inf) == 0.0
    assert target_hit_tol(3.0) == abs(3.0) * 1e-12
    assert is_target_hit(3.0 - 1e-13, 3.0)
    assert not is_target_hit(3.0 - 1e-11, 3.0)
    assert not is_target_hit(1e300, math.inf)


def test_inception_state():
    s = TargetRedemptionState(0.0)
    assert s.accumulated == 0.0 and s.pending == () and s.as_of is None


def test_pending_list_is_stored_as_a_tuple():
    s = TargetRedemptionState(1.0, pending=[PendingPayment(1.0, pay_time=0.1)])
    assert s.pending == (PendingPayment(1.0, pay_time=0.1),)


def test_state_is_immutable():
    s = TargetRedemptionState(0.0)
    with pytest.raises(FrozenInstanceError):
        s.accumulated = 1.0


@pytest.mark.parametrize("acc", [-1e-9, math.nan, math.inf])
def test_rejects_bad_accumulated(acc):
    with pytest.raises(ValidationError, match="accumulated"):
        TargetRedemptionState(acc)


@pytest.mark.parametrize("kw", [{}, {"pay_time": 0.1, "pay_date": D1}])
def test_pending_needs_exactly_one_of_time_or_date(kw):
    with pytest.raises(ValidationError, match="exactly one"):
        PendingPayment(1.0, **kw)


@pytest.mark.parametrize("t", [0.0, -0.1, 1e-12, math.nan, math.inf])
def test_numeric_pay_time_must_be_after_valuation(t):
    with pytest.raises(ValidationError, match="pay_time"):
        PendingPayment(1.0, pay_time=t)


@pytest.mark.parametrize("amount", [math.nan, math.inf])
def test_pending_amount_must_be_finite(amount):
    with pytest.raises(ValidationError, match="finite"):
        PendingPayment(amount, pay_time=0.1)


def test_rejects_mixed_pending_forms():
    with pytest.raises(ValidationError, match="mixes"):
        TargetRedemptionState(1.0, pending=(PendingPayment(1.0, pay_time=0.1), PendingPayment(1.0, pay_date=D3)),
                              as_of=D1)


def test_rejects_unsorted_pending():
    with pytest.raises(ValidationError, match="sorted"):
        TargetRedemptionState(1.0, pending=(PendingPayment(1.0, pay_time=0.2), PendingPayment(1.0, pay_time=0.1)))


def test_equal_pay_times_are_sorted():
    TargetRedemptionState(1.0, pending=(PendingPayment(0.5, pay_time=0.1), PendingPayment(100.0, pay_time=0.1)))


def test_dated_pending_needs_as_of():
    with pytest.raises(ValidationError, match="as_of"):
        TargetRedemptionState(1.0, pending=(PendingPayment(1.0, pay_date=D2),))


@pytest.mark.parametrize("pay", [D1, datetime(2026, 7, 1)])
def test_dated_pending_must_be_after_as_of(pay):
    with pytest.raises(ValidationError, match="after as_of"):
        TargetRedemptionState(1.0, pending=(PendingPayment(1.0, pay_date=pay),), as_of=D1)


def test_terminal_booking_warning_is_a_user_warning():
    assert issubclass(TerminalBookingWarning, UserWarning)


def test_event_defaults():
    e = TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, 1.0)
    assert (e.fixing, e.time, e.redeemed, e.label) == (None, 0.0, False, "")
    assert TargetRedemptionEventType.FIXING.value == "fixing"


def test_resolved_view_is_frozen():
    v = ResolvedTargetRedemption((0.5,), (0.5,), None, 0.0, (), False, False)
    with pytest.raises(FrozenInstanceError):
        v.accumulated = 1.0


def test_pending_present_value_sums_in_order():
    e = FxPricingEnvironment(valuation_date=D1, spot_quote=SpotQuote(spot=1.2),
                             domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.03),
                             vol_surface=FlatVolSurface(volatility=0.1))
    expected = 0.0 + 0.5 * e.get_domestic_df(0.1) + 100.0 * e.get_domestic_df(0.1)
    assert pending_present_value(((0.1, 0.5), (0.1, 100.0)), e) == expected
    assert pending_present_value((), e) == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_state.py -n 2 -q`
Expected: collection error `ModuleNotFoundError: No module named 'quantark.asset.fx.product.option.target_redemption_state'`.

- [ ] **Step 3: Implement the module**

```python
"""Realized state of an FX target-redemption contract (TARN / TARF).

Spec: docs/superpowers/specs/2026-10-09-fx-tarn-lifecycle-design.md, section 3. ``accumulated`` is the undiscounted
amount banked towards the target (TARN coupons, TARF favourable amounts, including those fixed but not yet paid).
``pending`` is determined cash whose pay time is after valuation. The state is immutable: ageing replaces it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Final, Optional, Sequence, Tuple

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_zero

#: Relative target-hit tolerance, moved from the two MC engines' ``_HIT_RTOL`` with its value unchanged (spec section
#: 3: ``Tolerance.RELATIVE`` would move redemption decisions and inception prices).
TARGET_HIT_RTOL: Final[float] = 1e-12


def target_hit_tol(target: float) -> float:
    """Target-relative hit tolerance; 0 for an unbounded target (the engines' rule, unchanged)."""
    return 0.0 if math.isinf(target) else abs(target) * TARGET_HIT_RTOL


def is_target_hit(accumulated: float, target: float) -> bool:
    """True once ``accumulated`` reaches ``target`` within :func:`target_hit_tol`."""
    return accumulated >= target - target_hit_tol(target)


@dataclass(frozen=True)
class PendingPayment:
    """Determined, unpaid domestic cash: ``pay_time`` (numeric form, years from valuation) or ``pay_date`` (dated)."""

    amount: float
    pay_time: Optional[float] = None
    pay_date: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.amount):
            raise ValidationError(f"pending amount must be finite, got {self.amount}")
        if (self.pay_time is None) == (self.pay_date is None):
            raise ValidationError(
                "a pending payment takes exactly one of pay_time (numeric form) or pay_date (dated form)"
            )
        if self.pay_time is not None:
            t = self.pay_time
            if not math.isfinite(t) or t <= 0.0 or is_zero(t):
                raise ValidationError(
                    f"a numeric pending pay_time must be after valuation (> 0), got {t}: cash paid at or before "
                    "valuation is booked, not pending"
                )


@dataclass(frozen=True)
class TargetRedemptionState:
    """What a target-redemption contract has banked (``accumulated``) and still has to pay (``pending``).

    ``as_of`` is used by the dated form only: every fixing dated on or before it is banked.
    """

    accumulated: float
    pending: Tuple[PendingPayment, ...] = ()
    as_of: Optional[datetime] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pending", tuple(self.pending))
        acc = self.accumulated
        if not math.isfinite(acc) or acc < 0.0:
            raise ValidationError(f"state.accumulated must be finite and >= 0, got {acc}")
        if any(not isinstance(p, PendingPayment) for p in self.pending):
            raise ValidationError("state.pending holds PendingPayment entries")
        numeric = [p.pay_time is not None for p in self.pending]
        if any(numeric) and not all(numeric):
            raise ValidationError("state.pending mixes pay_time and pay_date entries: one form per contract")
        keys = [p.pay_time if p.pay_time is not None else p.pay_date for p in self.pending]
        if keys != sorted(keys):
            raise ValidationError("state.pending must be sorted by ascending pay time / date")
        if self.pending and not any(numeric):
            if self.as_of is None:
                raise ValidationError(
                    "dated pending cash with as_of=None is refused: cash implies a banked fixing"
                )
            late = [p.pay_date for p in self.pending if p.pay_date <= self.as_of]
            if late:
                raise ValidationError(f"a dated pending pay_date must be after as_of {self.as_of}, got {late[0]}")


class TerminalBookingWarning(UserWarning):
    """A contract was BOOKED terminal (redeemed, or every fixing banked): it is valued at the PV of its pending cash,
    0.0 when none (spec section 5.3, user decision Q6). Raised once, by the constructor."""


class TargetRedemptionEventType(Enum):
    FIXING = "fixing"  # a fixing was banked
    PAYMENT = "payment"  # determined cash reached its pay time / date


@dataclass(frozen=True)
class TargetRedemptionEvent:
    """One lifecycle event, appended to a product's ``lifecycle_events`` by ``time_shift`` / ``record_passed_fixings``.

    ``amount``: FIXING - the cash it determined (coupon / net cashflow, principal excluded); PAYMENT - the cash paid.
    ``time``: FIXING - the fixing's signed time at banking (<= 0); PAYMENT - the pay time (<= 0).
    ``redeemed``: FIXING - this fixing ended the contract (target hit, or the last fixing).
    ``label``: W5 ``fixing_label`` text; an immediate PAYMENT carries its FIXING's label (informational).
    """

    event_type: TargetRedemptionEventType
    amount: float
    fixing: Optional[float] = None
    time: float = 0.0
    redeemed: bool = False
    label: str = ""


@dataclass(frozen=True)
class ResolvedTargetRedemption:
    """The one engine input (spec section 5.1): unbanked fixings still ahead (all > 0), their factors (TARN), the
    banked amount, pending ``(pay_time, amount)`` after valuation, and the derived flags."""

    fixing_times: Tuple[float, ...]
    pay_times: Tuple[float, ...]
    accrual_factors: Optional[Tuple[float, ...]]
    accumulated: float
    pending: Tuple[Tuple[float, float], ...]
    redeemed: bool
    terminal: bool


def pending_present_value(pending: Sequence[Tuple[float, float]], fx_env) -> float:
    """``sum(amount * DF_dom(pay_time))`` over ``(pay_time, amount)`` pairs, in order."""
    pv = 0.0
    for pay_time, amount in pending:
        pv += amount * fx_env.get_domestic_df(pay_time)
    return pv
```

Append to `quantark/asset/fx/product/option/__init__.py` (imports after the TARN import, names in `__all__`):

```python
from .target_redemption_state import (
    PendingPayment,
    TargetRedemptionEvent,
    TargetRedemptionEventType,
    TargetRedemptionState,
    TerminalBookingWarning,
)
```
and add `'PendingPayment', 'TargetRedemptionEvent', 'TargetRedemptionEventType', 'TargetRedemptionState',
'TerminalBookingWarning'` to `__all__`.

Insert at the top of `CHANGELOG.md` `## [Unreleased]` (right after that heading and its blank line):

```markdown
FX TARN / TARF mid-life booking, dated form and ageing (spec:
`docs/superpowers/specs/2026-10-09-fx-tarn-lifecycle-design.md`). A target-redemption contract carries its realized
state, can be booked by dates, ages through `time_shift` / `record_passed_fixings`, and is aged day by day in the FX
backtest and FX dynamic scenario.

### Added
- `quantark.asset.fx.product.option.target_redemption_state`: `TargetRedemptionState` (amount banked towards the
  target, determined `PendingPayment`s, dated `as_of`), `TargetRedemptionEvent` / `TargetRedemptionEventType`,
  `TerminalBookingWarning`, `ResolvedTargetRedemption`, and `TARGET_HIT_RTOL` (1e-12, moved from the two TARN/TARF
  engines with its value unchanged).

```

- [ ] **Step 4: Run to verify it passes**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_state.py -n 2 -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/asset/fx/product/option/target_redemption_state.py quantark/asset/fx/product/option/__init__.py test/test_fx_target_redemption_state.py CHANGELOG.md
git -C <WT> commit -m "feat(fx-tarn): TargetRedemptionState - realized state, pending cash, lifecycle events

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Shared base and numeric TARN construction (mandatory state, period start, terminal booking)

Spec §3, §4.1, §5.3, §5.4, §8 (construction sites), §10 R1/R2/WARN1 (construction part).

**Files:**
- Create: `quantark/asset/fx/product/option/target_redemption_base.py`, `test/fx_target_redemption_fixtures.py`, `test/test_fx_target_redemption_lifecycle.py`
- Modify: `quantark/asset/fx/product/option/fx_target_redemption_note.py` (whole `__init__`/validation; keep `coupon_condition`, `is_band`, `__repr__`), `test/test_fx_tarn_note_mc.py` (`_note` only), `test/execution/matrix_fixtures.py:947-958`, `example/fx_tarn_demo.py:97-99, 109-111`, `CHANGELOG.md`

**Interfaces:**
- Consumes: Task 2 names.
- Produces:
  - `quantark.asset.fx.product.option.target_redemption_base`: `is_passed_time(t: float) -> bool`, `event_label(index: int, *, time: Optional[float] = None, date: Optional[datetime] = None) -> str`, class `TargetRedemptionProduct(BaseFxProduct)` with properties `is_numeric`, `hit_tol`, `is_unbounded_target`, `is_redeemed`, `is_terminal`; attributes `fixing_times`, `pay_times`, `fixing_dates` (None), `pay_dates` (None), `state`, `lifecycle_events: Tuple[TargetRedemptionEvent, ...]`; subclass hooks `_validate_terms()`, `_fixing_outcome(factor, spot, accumulated) -> (banked, cash)`, `_principal_amount() -> float`, class attrs `_has_accrual`, `_pending_must_be_positive`.
  - `FxTargetRedemptionNote(fixing_times=None, coupon_rate=None, notional=1.0, target=math.inf, strike=None, is_above=True, lower=None, upper=None, accrual_factors=None, pay_times=None, include_principal=True, currency_pair=None, *, state=None, accrual_start=None)`; methods `coupon_amount(factor) -> float` (`self.notional * self.coupon_rate * factor`), `full_coupon(period) -> float`.
  - Test fixtures `env`, `mc`, `note`, `engine_for`, `PAIR`, `VAL`, `FIX`, `INCEPTION`.

- [ ] **Step 1: Write the fixtures module**

`test/fx_target_redemption_fixtures.py`:

```python
"""Shared builders for the FX TARN / TARF lifecycle tests (spec 2026-10-09-fx-tarn-lifecycle-design.md, section 2.2
fixture: twelve monthly fixings at k/12, TARN notional 100 / coupon 12% / strike 1.10, TARF strike 1.15 / notional 1,
EURUSD 1.20, r_d 5%, r_f 3%, vol 10%)."""
from datetime import datetime

from quantark.asset.fx.engine.mc import FxTargetRedemptionNoteMCEngine, FxTarnForwardMCEngine
from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxTargetRedemptionForward, FxTargetRedemptionNote
from quantark.asset.fx.product.option.target_redemption_state import TargetRedemptionState
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment
from quantark.util.calendar import DayCountConvention

PAIR = CurrencyPair("EUR", "USD")
VAL = datetime(2026, 6, 15)
FIX = [k / 12 for k in range(1, 13)]
INCEPTION = TargetRedemptionState(0.0)


def env(spot=1.20, val=VAL, vol=0.10, dc=DayCountConvention.CALENDAR_DAYS, calendar=None, spot_days=0):
    return FxPricingEnvironment(
        valuation_date=val, spot_quote=SpotQuote(spot=spot), domestic_curve=FlatRateCurve(rate=0.05),
        foreign_curve=FlatRateCurve(rate=0.03), vol_surface=FlatVolSurface(volatility=vol),
        day_count_convention=dc, calendar=calendar, spot_days=spot_days,
    )


def mc(n=20_000, seed=7, method=None):
    kw = dict(num_paths=n, seed=seed)
    if method is not None:
        kw["method"] = method
    return FxMCParams(**kw)


def note(fix=FIX, state=INCEPTION, **kw):
    """Numeric TARN; the first listed period starts at 0.0 unless the caller states accrual_factors or accrual_start."""
    base = dict(fixing_times=fix, coupon_rate=0.12, notional=100.0, strike=1.10, currency_pair=PAIR, state=state)
    if "accrual_factors" not in kw:
        base["accrual_start"] = 0.0
    base.update(kw)
    return FxTargetRedemptionNote(**base)


def engine_for(product, n=20_000, seed=7, method=None):
    params = mc(n=n, seed=seed, method=method)
    if isinstance(product, FxTargetRedemptionNote):
        return FxTargetRedemptionNoteMCEngine(params)
    return FxTarnForwardMCEngine(params)
```

- [ ] **Step 2: Write the failing tests**

`test/test_fx_target_redemption_lifecycle.py`:

```python
"""FX TARN / TARF lifecycle: construction, ageing, pricing, dated twins (spec 2026-10-09-fx-tarn-lifecycle-design.md,
section 10)."""
import copy
import math
import warnings

import pytest

from fx_target_redemption_fixtures import FIX, INCEPTION, PAIR, env, note
from quantark.asset.fx.product.option import FxTargetRedemptionNote
from quantark.asset.fx.product.option.target_redemption_state import (
    PendingPayment,
    TargetRedemptionState,
    TerminalBookingWarning,
)
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close


# --- R1 (findings section 10): the period start is a contract term -------------------------------------------------

def test_r1_rebuild_without_a_period_source_is_refused():
    remaining = [t - 0.125 for t in FIX[1:]]
    with pytest.raises(ValidationError, match="accrual_start"):
        FxTargetRedemptionNote(fixing_times=remaining, coupon_rate=0.12, notional=100.0, strike=1.10,
                               currency_pair=PAIR, state=INCEPTION)


def test_r1_accrual_start_restores_the_full_first_coupon():
    remaining = [t - 0.125 for t in FIX[1:]]
    rebuilt = note(fix=remaining, accrual_start=FIX[0] - 0.125)
    assert is_close(rebuilt.accrual_factors[0], 1 / 12, rel_tol=1e-12)
    assert is_close(rebuilt.full_coupon(0), 1.0, rel_tol=1e-12)


def test_accrual_start_zero_is_todays_default_bitwise():
    explicit = note(fix=[0.5, 1.0, 1.5])
    assert explicit.accrual_factors == [0.5 - 0.0, 1.0 - 0.5, 1.5 - 1.0]


def test_inception_horizon_is_unchanged():
    n = note(pay_times=[t + 0.02 for t in FIX])
    assert n.maturity == FIX[-1] and n.delivery == FIX[-1] + 0.02
    assert n.lifecycle_events == () and n.is_numeric and not n.is_terminal and not n.is_redeemed


def test_hit_tolerance_lives_on_the_product():
    assert note(target=3.0).hit_tol == 3.0 * 1e-12
    assert note().hit_tol == 0.0


# --- R2 (numeric TARN refusals) ------------------------------------------------------------------------------------

@pytest.mark.parametrize("kw, match", [
    (dict(accrual_factors=[1 / 12] * 12, accrual_start=0.0), "not both"),
    (dict(state=None), "state is required"),
    (dict(state=TargetRedemptionState(3.0 + 1e-9), target=3.0), "exceeds the target"),
    (dict(accrual_start=FIX[0]), "before the first listed fixing"),
    (dict(accrual_start=0.01, state=TargetRedemptionState(1.0), target=3.0), "contiguous"),
    (dict(state=TargetRedemptionState(1.0, pending=(PendingPayment(-1.0, pay_time=0.1),)), target=3.0),
     "positive"),
    (dict(state=TargetRedemptionState(3.0), target=3.0), "redeemed"),
    (dict(coupon_rate=None), "coupon_rate"),
])
def test_r2_numeric_tarn_refusals(kw, match):
    with pytest.raises(ValidationError, match=match):
        note(**kw)


def test_r2_numeric_state_refuses_as_of_and_pay_dates():
    from datetime import datetime
    with pytest.raises(ValidationError, match="as_of"):
        note(state=TargetRedemptionState(0.0, as_of=datetime(2026, 6, 1)))
    dated_pending = TargetRedemptionState(1.0, pending=(PendingPayment(1.0, pay_date=datetime(2026, 7, 1)),),
                                          as_of=datetime(2026, 6, 1))
    with pytest.raises(ValidationError):
        note(state=dated_pending, target=3.0)


def test_r2_missing_state_is_a_validation_error_not_a_type_error():
    with pytest.raises(ValidationError, match="state is required"):
        FxTargetRedemptionNote(fixing_times=FIX, coupon_rate=0.12, strike=1.10, accrual_start=0.0)


def test_past_fixing_times_are_accepted():
    n = note(fix=[-0.01] + FIX[1:], accrual_start=-0.05)
    assert n.fixing_times[0] == -0.01


# --- WARN1 (numeric construction part; values in Task 6) -----------------------------------------------------------

def test_warn1_empty_schedule_booking_warns_once():
    with pytest.warns(TerminalBookingWarning) as rec:
        n = note(fix=[], accrual_factors=[], target=3.0,
                 state=TargetRedemptionState(1.0, pending=(PendingPayment(100.0, pay_time=0.05),)))
    assert len([w for w in rec if issubclass(w.category, TerminalBookingWarning)]) == 1
    assert n.is_terminal and not n.is_redeemed
    assert n.maturity == 0.05 and n.delivery == 0.05


def test_warn1_redeemed_booking_warns_once():
    with pytest.warns(TerminalBookingWarning):
        n = note(fix=[], accrual_factors=[], target=1.5, state=TargetRedemptionState(1.5))
    assert n.is_redeemed and n.is_terminal and n.maturity == 0.0


def test_live_booking_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", TerminalBookingWarning)
        note()


def test_deepcopy_does_not_warn_and_keeps_the_state():
    with pytest.warns(TerminalBookingWarning):
        n = note(fix=[], accrual_factors=[], target=1.5, state=TargetRedemptionState(1.5))
    with warnings.catch_warnings():
        warnings.simplefilter("error", TerminalBookingWarning)
        c = copy.deepcopy(n)
    assert c.state == n.state and c.lifecycle_events == ()
```

- [ ] **Step 3: Run to verify it fails**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_lifecycle.py -n 2 -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'state'` (fixtures pass `state`).

- [ ] **Step 4: Implement the base class**

`quantark/asset/fx/product/option/target_redemption_base.py`:

```python
"""Shared machinery of the FX target-redemption products (TARN and TARF).

Spec: docs/superpowers/specs/2026-10-09-fx-tarn-lifecycle-design.md, sections 3-6. A contract carries its realized
state (``state``: the amount banked towards the target and the determined cash still to pay). In the NUMERIC form
``fixing_times`` / ``pay_times`` are years from valuation and list ONLY the fixings not yet banked; a time at or
before valuation is an unbanked past fixing. Only the two concrete products are instantiated.
"""
from __future__ import annotations

import math
import warnings
from datetime import datetime
from types import SimpleNamespace
from typing import List, Optional, Tuple

from quantark.util.exceptions import ValidationError
from quantark.util.fixings import fixing_label
from quantark.util.numerical import is_zero

from ..base_fx_product import BaseFxProduct
from .target_redemption_state import (
    TargetRedemptionEvent,
    TargetRedemptionState,
    TerminalBookingWarning,
    is_target_hit,
    target_hit_tol,
)


def is_passed_time(t: float) -> bool:
    """A numeric fixing or pay time at or before valuation; ``is_zero`` absorbs the ULPs daily ageing leaves."""
    return t <= 0.0 or is_zero(t)


def event_label(index: int, *, time: Optional[float] = None, date: Optional[datetime] = None) -> str:
    """W5 ``fixing_label`` text (``#i t=<time>`` numeric, ``#i <date>`` dated) for a TARN/TARF fixing."""
    return fixing_label(index, SimpleNamespace(observation_time=time, observation_date=date))


class TargetRedemptionProduct(BaseFxProduct):
    """Base of ``FxTargetRedemptionNote`` and ``FxTargetRedemptionForward``: schedule forms, realized state,
    validation, terminal bookings, ageing and the resolved view."""

    #: TARN: per-period accrual factors aligned with the listed fixings.
    _has_accrual: bool = False
    #: TARN: determined cash is a coupon or the principal, so a pending amount must be > 0.
    _pending_must_be_positive: bool = False

    fixing_times: Optional[List[float]]
    pay_times: Optional[List[float]]
    lifecycle_events: Tuple[TargetRedemptionEvent, ...]

    # -- subclass contract --------------------------------------------------------------------------------------
    def _validate_terms(self) -> None:
        raise NotImplementedError

    def _fixing_outcome(self, factor: Optional[float], spot: float, accumulated: float) -> Tuple[float, float]:
        """``(amount banked towards the target, cash determined)`` for one fixing at ``spot``."""
        raise NotImplementedError

    def _principal_amount(self) -> float:
        """Cash due when the contract ends (the TARN principal); 0.0 when none."""
        return 0.0

    # -- construction -------------------------------------------------------------------------------------------
    @staticmethod
    def _require_state(state) -> TargetRedemptionState:
        if state is None:
            raise ValidationError(
                "state is required: a contract booked at inception passes TargetRedemptionState(0.0); a mid-life "
                "booking states what it has banked and what is still to pay"
            )
        if not isinstance(state, TargetRedemptionState):
            raise ValidationError(f"state must be a TargetRedemptionState, got {type(state).__name__}")
        return state

    def _init_numeric(self, fixings: List[float], pay_times, currency_pair, state: TargetRedemptionState) -> None:
        BaseFxProduct.__init__(self, currency_pair=currency_pair)
        self.fixing_times = list(fixings)
        self.pay_times = [float(t) for t in pay_times] if pay_times is not None else list(fixings)
        self.fixing_dates = None
        self.pay_dates = None
        self.state = state
        self.lifecycle_events = ()

    def _refresh_numeric_horizon(self) -> None:
        """``maturity`` = last listed fixing, else the last pending pay time, else 0.0; ``delivery`` alike."""
        if self.fixing_times:
            self.maturity = self.fixing_times[-1]
            self.delivery = max(self.pay_times)
        else:
            pending = [p.pay_time for p in self.state.pending]
            self.maturity = self.delivery = max(pending) if pending else 0.0

    def _warn_if_terminal_booking(self) -> None:
        if self.is_terminal:
            why = "redeemed" if self.is_redeemed else "every fixing banked"
            warnings.warn(
                TerminalBookingWarning(
                    f"{type(self).__name__} booked terminal ({why}): it is valued at the PV of its "
                    f"{len(self.state.pending)} pending payment(s), 0.0 when none"
                ),
                stacklevel=3,
            )

    # -- derived flags ------------------------------------------------------------------------------------------
    @property
    def is_numeric(self) -> bool:
        return self.fixing_dates is None

    @property
    def hit_tol(self) -> float:
        """Target-hit tolerance shared with the engines (``target_hit_tol``)."""
        return target_hit_tol(self.target)

    @property
    def is_unbounded_target(self) -> bool:
        return math.isinf(self.target)

    @property
    def is_redeemed(self) -> bool:
        return is_target_hit(self.state.accumulated, self.target)

    @property
    def is_terminal(self) -> bool:
        return not self.fixing_times or self.is_redeemed

    # -- validation ---------------------------------------------------------------------------------------------
    def validate(self) -> None:
        self._validate_terms()
        self._validate_numeric_schedule()
        self._validate_state()

    def _validate_numeric_schedule(self) -> None:
        fixings, pays = self.fixing_times, self.pay_times
        if list(fixings) != sorted(fixings):
            raise ValidationError("fixing_times must be ascending")
        if len(set(fixings)) != len(fixings):
            raise ValidationError("fixing_times must be unique")
        if any(not math.isfinite(t) for t in fixings):
            raise ValidationError("fixing_times must be finite")
        if len(pays) != len(fixings):
            raise ValidationError("pay_times must align 1:1 with fixing_times")
        for tf, tp in zip(fixings, pays):
            if not math.isfinite(tp):
                raise ValidationError(f"pay time {tp} must be finite")
            if tp < tf:
                raise ValidationError(f"pay time {tp} precedes its fixing {tf}")
        if list(pays) != sorted(pays):
            raise ValidationError("pay_times must be non-decreasing")
        if self._has_accrual:
            factors = self.accrual_factors
            if len(factors) != len(fixings):
                raise ValidationError("accrual_factors must align 1:1 with fixing_times")
            if any((not math.isfinite(a)) or a <= 0.0 for a in factors):
                raise ValidationError("accrual_factors must be positive and finite")

    def _validate_state(self) -> None:
        st = self.state
        if st.accumulated > self.target + self.hit_tol:
            raise ValidationError(f"state.accumulated {st.accumulated} exceeds the target {self.target}")
        if self._pending_must_be_positive and any(p.amount <= 0.0 for p in st.pending):
            raise ValidationError(
                "pending amounts must be positive: a TARN's determined cash is a coupon or its principal"
            )
        if st.as_of is not None:
            raise ValidationError("state.as_of belongs to the dated form: a numeric state lists only unbanked fixings")
        if any(p.pay_date is not None for p in st.pending):
            raise ValidationError("a numeric contract's pending payments take pay_time, not pay_date")
        if self.is_redeemed and self.fixing_times:
            raise ValidationError("a redeemed state cannot list fixings: every fixing after the target hit is dead")
```

- [ ] **Step 5: Rewrite the TARN constructor on the base**

In `fx_target_redemption_note.py`: change the base class to `TargetRedemptionProduct`, set `_has_accrual = True`,
`_pending_must_be_positive = True`, delete `_validate_schedule`, `is_unbounded_target` (inherited), and update the
module/class docstrings (`accrual_factors`: "explicit per-listed-period factors; or `accrual_start`, the time the
first listed period began — exactly one is required"; `state`: mandatory). Replace `__init__`/`validate`/`full_coupon`/
`get_payoff` with:

```python
import math
from typing import List, Optional, Tuple

from quantark.util.exceptions import ValidationError
from ..currency_pair import CurrencyPair
from .target_redemption_base import TargetRedemptionProduct
from .target_redemption_state import TargetRedemptionState


class FxTargetRedemptionNote(TargetRedemptionProduct):
    _has_accrual = True
    _pending_must_be_positive = True

    def __init__(
        self,
        fixing_times: Optional[List[float]] = None,
        coupon_rate: Optional[float] = None,
        notional: float = 1.0,
        target: float = math.inf,
        strike: Optional[float] = None,
        is_above: bool = True,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        accrual_factors: Optional[List[float]] = None,
        pay_times: Optional[List[float]] = None,
        include_principal: bool = True,
        currency_pair: Optional[CurrencyPair] = None,
        *,
        state: Optional[TargetRedemptionState] = None,
        accrual_start: Optional[float] = None,
    ):
        self.coupon_rate = coupon_rate
        self.notional = notional
        self.target = float(target)
        self.strike = strike
        self.is_above = bool(is_above)
        self.lower = lower
        self.upper = upper
        self.include_principal = bool(include_principal)
        state = self._require_state(state)
        if fixing_times is None:
            raise ValidationError("fixing_times is required")
        fixings = [float(t) for t in fixing_times]
        self.accrual_factors = self._numeric_accruals(fixings, accrual_start, accrual_factors, state)
        self._init_numeric(fixings, pay_times, currency_pair, state)
        self.validate()
        self._refresh_numeric_horizon()
        self._warn_if_terminal_booking()

    @staticmethod
    def _numeric_accruals(fixings, accrual_start, accrual_factors, state) -> List[float]:
        """The listed periods' factors: explicit, or ``t_i - t_{i-1}`` with ``t_{-1} = accrual_start`` (spec Q2a)."""
        if accrual_start is None and accrual_factors is None:
            raise ValidationError(
                "a numeric TARN must state when its first listed period began: pass accrual_start (years from "
                "valuation, negative when it began before) or explicit accrual_factors - no start is assumed"
            )
        if accrual_start is not None and accrual_factors is not None:
            raise ValidationError("give accrual_start or accrual_factors, not both")
        if accrual_factors is not None:
            return [float(a) for a in accrual_factors]
        start = float(accrual_start)
        if not math.isfinite(start):
            raise ValidationError(f"accrual_start must be finite, got {start}")
        if fixings and not start < fixings[0]:
            raise ValidationError(f"accrual_start {start} must be before the first listed fixing {fixings[0]}")
        if start > 0.0 and state.accumulated > 0.0:
            raise ValidationError(
                "accrual_start > 0 with accumulated > 0: periods are contiguous, so a contract that has banked a "
                "fixing has a listed period that began at or before valuation"
            )
        prev = [start] + fixings[:-1]
        return [t - p for t, p in zip(fixings, prev)]

    def _validate_terms(self) -> None:
        if self.coupon_rate is None:
            raise ValidationError("coupon_rate is required")
        for name, v in (("notional", self.notional), ("coupon_rate", self.coupon_rate)):
            if not math.isfinite(v) or v <= 0:
                raise ValidationError(f"{name} must be positive and finite, got {v}")
        if not (self.target > 0):  # rejects 0, negatives, NaN; inf passes
            raise ValidationError(f"target must be positive, got {self.target}")
        self._validate_condition()

    def coupon_amount(self, factor: float) -> float:
        """Undiscounted coupon of a period with accrual ``factor`` when the condition holds."""
        return self.notional * self.coupon_rate * factor

    def full_coupon(self, period: int) -> float:
        """Undiscounted coupon of listed period ``period`` if the condition holds."""
        return self.coupon_amount(self.accrual_factors[period])

    def _fixing_outcome(self, factor, spot, accumulated) -> Tuple[float, float]:
        coupon = self.coupon_amount(factor) if self.coupon_condition(spot) else 0.0
        c = min(coupon, self.target - accumulated)
        return c, c

    def _principal_amount(self) -> float:
        return self.notional if self.include_principal else 0.0

    def get_payoff(self, spot: float) -> float:
        """First listed period's coupon at ``spot`` (path-independent utility)."""
        if not self.accrual_factors:
            raise ValidationError("a terminal TARN has no next coupon")
        return self.full_coupon(0) if self.coupon_condition(spot) else 0.0
```

Keep `_validate_condition`, `is_band`, `coupon_condition`, the existing `time_shift` (replaced in Task 5) and
`__repr__` as they are.

- [ ] **Step 6: Update the TARN construction sites (constructor arguments only, spec §8)**

`test/test_fx_tarn_note_mc.py` — add the import and replace `_note`:

```python
from quantark.asset.fx.product.option.target_redemption_state import TargetRedemptionState


def _note(**kw):
    base = dict(fixing_times=[0.25, 0.5, 0.75, 1.0], coupon_rate=0.08,
               notional=1.0, strike=1.20, currency_pair=_PAIR,
               state=TargetRedemptionState(0.0))
    if "accrual_factors" not in kw:
        base["accrual_start"] = 0.0
    base.update(kw)
    return FxTargetRedemptionNote(**base)
```

`test/execution/matrix_fixtures.py` `fx_tarn` — add `TargetRedemptionState` import inside the function and the two
arguments:

```python
        from quantark.asset.fx.product.option.target_redemption_state import TargetRedemptionState
        ...
            FxTargetRedemptionNote(
                fixing_times=[0.25, 0.5, 0.75, 1.0], coupon_rate=0.08,
                notional=1.0, strike=1.20, currency_pair=_pair(),
                state=TargetRedemptionState(0.0), accrual_start=0.0,
            ),
```

`example/fx_tarn_demo.py` — import `TargetRedemptionState` from `quantark.asset.fx.product.option` and add
`state=TargetRedemptionState(0.0), accrual_start=0.0` to `note_open` (line 97) and `note_full` (line 109).

Append to this workstream's CHANGELOG block (create `### Changed` after `### Added`):

```markdown
### Changed
- **Breaking:** `FxTargetRedemptionNote` and `FxTargetRedemptionForward` take a mandatory keyword `state`
  (`TargetRedemptionState(0.0)` for a contract booked at inception); omitting it raises `ValidationError`.
- **Breaking:** a numeric `FxTargetRedemptionNote` states when its first listed period began: `accrual_start` (years
  from valuation) or explicit `accrual_factors`, exactly one. The old default (`t_0 = 0`, the first period starting at
  the valuation instant) halved the first coupon of a contract rebuilt mid-life. `accrual_start=0.0` reproduces the old
  factors bitwise.
- A non-positive fixing time is accepted (an unbanked past fixing) and an empty `fixing_times` is a terminal booking
  (`TerminalBookingWarning`, valued at the PV of its pending cash).
```

- [ ] **Step 7: Run the gates**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_lifecycle.py test/test_fx_tarn_note_mc.py -n 2 -q` and
`env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/execution/test_matrix_parity.py -k "TargetRedemption" -n 2 -q`
Expected: all pass (27 TARN tests unchanged apart from `_note`; 1 matrix parity test).
Then: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python <WT>/example/fx_tarn_demo.py | cmp - <WT>/.superpowers/w4b/demo_fx_tarn_demo.main.txt && echo SAME`
Expected: `SAME`.

- [ ] **Step 8: Commit**

```bash
git -C <WT> add quantark/asset/fx/product/option/target_redemption_base.py quantark/asset/fx/product/option/fx_target_redemption_note.py test/fx_target_redemption_fixtures.py test/test_fx_target_redemption_lifecycle.py test/test_fx_tarn_note_mc.py test/execution/matrix_fixtures.py example/fx_tarn_demo.py CHANGELOG.md
git -C <WT> commit -m "feat(fx-tarn): numeric TARN states its realized state and period start; terminal bookings warn

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Numeric TARF on the shared base

Spec §2.3, §4.1.

**Files:**
- Modify: `quantark/asset/fx/product/option/fx_target_redemption_forward.py`, `test/test_fx_tarf_mc.py` (`_tarf` only), `test/execution/matrix_fixtures.py:932-943`, `example/fx_tarn_demo.py:74-75, 85-87`
- Test: `test/test_fx_target_redemption_lifecycle.py`, `test/fx_target_redemption_fixtures.py`

**Interfaces:**
- Consumes: `TargetRedemptionProduct` (Task 3).
- Produces: `FxTargetRedemptionForward(strike, fixing_times=None, notional=1.0, target=math.inf, gearing=1.0, is_buy_base=True, pay_times=None, currency_pair=None, *, state=None)`; fixture `tarf(fix=FIX, state=INCEPTION, **kw)`.

- [ ] **Step 1: Add the fixture and failing tests**

Append to `test/fx_target_redemption_fixtures.py`:

```python
def tarf(fix=FIX, state=INCEPTION, **kw):
    base = dict(strike=1.15, fixing_times=fix, notional=1.0, currency_pair=PAIR, state=state)
    base.update(kw)
    return FxTargetRedemptionForward(**base)
```

Append to `test/test_fx_target_redemption_lifecycle.py` (add `tarf` to the fixtures import and
`FxTargetRedemptionForward` to the product import):

```python
@pytest.mark.parametrize("kw, match", [
    (dict(state=None), "state is required"),
    (dict(state=TargetRedemptionState(0.2 + 1e-9), target=0.2), "exceeds the target"),
    (dict(state=TargetRedemptionState(0.2), target=0.2), "redeemed"),
])
def test_r2_numeric_tarf_refusals(kw, match):
    with pytest.raises(ValidationError, match=match):
        tarf(**kw)


def test_tarf_pending_may_be_negative():
    f = tarf(gearing=2.0, target=0.2, state=TargetRedemptionState(0.0, pending=(PendingPayment(-0.1, pay_time=0.01),)))
    assert f.state.pending[0].amount == -0.1


def test_tarf_terminal_booking_warns():
    with pytest.warns(TerminalBookingWarning):
        f = tarf(fix=[], target=0.2, state=TargetRedemptionState(0.05))
    assert f.is_terminal and not f.is_redeemed and f.maturity == 0.0


def test_tarf_inception_horizon_is_unchanged():
    f = tarf(pay_times=[t + 0.02 for t in FIX])
    assert f.maturity == FIX[-1] and f.delivery == FIX[-1] + 0.02 and f.hit_tol == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_lifecycle.py -k tarf -n 2 -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'state'`.

- [ ] **Step 3: Implement**

Rebase `FxTargetRedemptionForward` on `TargetRedemptionProduct`; delete `_validate_schedule` and
`is_unbounded_target`; keep `period_cashflow`, `get_payoff`, `__repr__`, and the old `time_shift` (Task 5 replaces):

```python
    def __init__(
        self,
        strike: float,
        fixing_times: Optional[List[float]] = None,
        notional: float = 1.0,
        target: float = math.inf,
        gearing: float = 1.0,
        is_buy_base: bool = True,
        pay_times: Optional[List[float]] = None,
        currency_pair: Optional[CurrencyPair] = None,
        *,
        state: Optional[TargetRedemptionState] = None,
    ):
        self.strike = strike
        self.notional = notional
        self.target = float(target)
        self.gearing = gearing
        self.is_buy_base = bool(is_buy_base)
        state = self._require_state(state)
        if fixing_times is None:
            raise ValidationError("fixing_times is required")
        self._init_numeric([float(t) for t in fixing_times], pay_times, currency_pair, state)
        self.validate()
        self._refresh_numeric_horizon()
        self._warn_if_terminal_booking()

    def _validate_terms(self) -> None:
        for name, v in (("strike", self.strike), ("notional", self.notional)):
            if not math.isfinite(v) or v <= 0:
                raise ValidationError(f"{name} must be positive and finite, got {v}")
        if not math.isfinite(self.gearing) or self.gearing < 0:
            raise ValidationError(f"gearing must be non-negative and finite, got {self.gearing}")
        if not (self.target > 0):  # rejects 0, negatives, and NaN; inf passes
            raise ValidationError(f"target must be positive, got {self.target}")

    def _fixing_outcome(self, factor, spot, accumulated) -> Tuple[float, float]:
        """The engine's arithmetic for one fixing: favourable amount clipped to the remaining target, net cash."""
        sign = 1.0 if self.is_buy_base else -1.0
        diff = sign * (spot - self.strike)
        fav = self.notional * max(diff, 0.0)
        unf = self.notional * max(-diff, 0.0)
        fav_eff = min(fav, self.target - accumulated)
        return fav_eff, fav_eff - self.gearing * unf
```

Update the docstring (`state` mandatory; a non-positive fixing time is an unbanked past fixing; empty
`fixing_times` is a terminal booking).

Construction sites: `test/test_fx_tarf_mc.py` `_tarf` gains `state=TargetRedemptionState(0.0)` in `base` (import from
`quantark.asset.fx.product.option.target_redemption_state`); `matrix_fixtures.py` `fx_tarf` gains
`state=TargetRedemptionState(0.0)`; `example/fx_tarn_demo.py` `tarf_open` / `tarf_capped` gain
`state=TargetRedemptionState(0.0)`.

- [ ] **Step 4: Run the gates**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_lifecycle.py test/test_fx_tarf_mc.py test/test_fx_tarn_note_mc.py -n 2 -q` and
`env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/execution/test_matrix_parity.py -k "Tarn or TargetRedemption" -n 2 -q`
Expected: all pass (52 + 2 C1 gate). Demo: `... example/fx_tarn_demo.py | cmp - <WT>/.superpowers/w4b/demo_fx_tarn_demo.main.txt && echo SAME` → `SAME`.

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/asset/fx/product/option/fx_target_redemption_forward.py test/fx_target_redemption_fixtures.py test/test_fx_target_redemption_lifecycle.py test/test_fx_tarf_mc.py test/execution/matrix_fixtures.py example/fx_tarn_demo.py
git -C <WT> commit -m "feat(fx-tarn): numeric TARF states its realized state; terminal bookings warn

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Ageing hooks, numeric form (`time_shift`, `record_passed_fixings`, events)

Spec §6 (numeric), A1.3, Q3.

**Files:**
- Modify: `quantark/asset/fx/product/option/target_redemption_base.py`; delete the `time_shift` overrides in `fx_target_redemption_note.py` and `fx_target_redemption_forward.py`
- Test: `test/test_fx_target_redemption_lifecycle.py`

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces: `TargetRedemptionProduct.time_shift(time_bump: float) -> None`; `TargetRedemptionProduct.record_passed_fixings(fx_env) -> None`; private `_record_numeric(fx_env)`, `_settle_numeric(amount, pay_time, label, pending, events)`. Events: per banked fixing one FIXING (`time` = its signed time, `redeemed` when it ends the contract), immediately-due cash as PAYMENT with the FIXING's `label`; PAYMENT events from `time_shift` have `label=""`.

- [ ] **Step 1: Write the failing tests**

Append (add `from quantark.asset.fx.product.option.target_redemption_state import TargetRedemptionEvent, TargetRedemptionEventType`
and `FIXING, PAYMENT = TargetRedemptionEventType.FIXING, TargetRedemptionEventType.PAYMENT` near the top):

```python
LAG = [t + 0.02 for t in FIX]


def _kinds(events):
    return [(e.event_type, e.amount, e.fixing, e.time, e.redeemed) for e in events]


def test_time_shift_keeps_a_passed_fixing_unbanked():
    n = note(target=3.0, pay_times=LAG)
    a = copy.deepcopy(n)
    a.time_shift(0.1)
    assert a.fixing_times == [t - 0.1 for t in FIX] and a.pay_times == [t - 0.1 for t in LAG]
    assert a.state == n.state and a.lifecycle_events == ()
    assert a.maturity == FIX[-1] - 0.1


def test_record_banks_the_passed_fixing_at_the_held_spot():
    n = note(target=3.0, pay_times=LAG)
    a = copy.deepcopy(n)
    a.time_shift(0.1)
    a.record_passed_fixings(env(spot=1.20))
    c = n.full_coupon(0)
    assert a.state == TargetRedemptionState(c, pending=(PendingPayment(c, pay_time=LAG[0] - 0.1),))
    assert a.fixing_times == [t - 0.1 for t in FIX[1:]] and a.accrual_factors == n.accrual_factors[1:]
    assert _kinds(a.lifecycle_events) == [(FIXING, c, 1.20, FIX[0] - 0.1, False)]
    assert a.lifecycle_events[0].label.startswith("#0 t=")


def test_pending_cash_leaves_when_a_later_shift_reaches_its_pay_time():
    n = note(target=3.0, pay_times=LAG)
    a = copy.deepcopy(n)
    a.time_shift(0.1)
    a.record_passed_fixings(env())
    a.time_shift(0.01)
    c = n.full_coupon(0)
    assert a.state.pending == ()
    assert _kinds(a.lifecycle_events[-1:]) == [(PAYMENT, c, None, (LAG[0] - 0.1) - 0.01, False)]


def test_redemption_clips_the_coupon_and_settles_the_principal():
    n = note(target=1.5, pay_times=LAG)
    a = copy.deepcopy(n)
    a.time_shift(0.175)
    a.record_passed_fixings(env(spot=1.20))
    c0 = n.full_coupon(0)
    c1 = min(n.full_coupon(1), 1.5 - c0)
    t_pay = LAG[1] - 0.175
    assert a.is_redeemed and a.is_terminal
    assert a.fixing_times == [] and a.pay_times == [] and a.accrual_factors == []
    assert a.state == TargetRedemptionState(c0 + c1, pending=(PendingPayment(c1, pay_time=t_pay),
                                                              PendingPayment(100.0, pay_time=t_pay)))
    assert _kinds(a.lifecycle_events) == [
        (FIXING, c0, 1.20, FIX[0] - 0.175, False),
        (PAYMENT, c0, None, LAG[0] - 0.175, False),
        (FIXING, c1, 1.20, FIX[1] - 0.175, True),
    ]
    assert a.lifecycle_events[1].label == a.lifecycle_events[0].label
    assert a.maturity == t_pay


def test_quiet_fixing_banks_zero_and_no_cash():
    a = note(target=3.0, pay_times=LAG)
    a.time_shift(0.1)
    a.record_passed_fixings(env(spot=1.05))
    assert a.state == TargetRedemptionState(0.0)
    assert _kinds(a.lifecycle_events) == [(FIXING, 0.0, 1.05, FIX[0] - 0.1, False)]


@pytest.mark.parametrize("principal", [True, False])
def test_last_fixing_without_a_hit_settles_the_principal(principal):
    a = note(fix=[0.05], pay_times=[0.07], include_principal=principal)
    a.time_shift(0.06)
    a.record_passed_fixings(env(spot=1.05))
    assert a.is_terminal and not a.is_redeemed
    assert a.state.pending == ((PendingPayment(100.0, pay_time=0.07 - 0.06),) if principal else ())
    assert _kinds(a.lifecycle_events) == [(FIXING, 0.0, 1.05, 0.05 - 0.06, True)]


def test_tarf_unfavourable_fixing_is_negative_pending_cash():
    a = tarf(gearing=2.0, target=0.2, pay_times=LAG)
    a.time_shift(0.1)
    a.record_passed_fixings(env(spot=1.10))
    diff = 1.0 * (1.10 - 1.15)
    cf = 0.0 - 2.0 * (1.0 * max(-diff, 0.0))
    assert cf < 0.0
    assert a.state == TargetRedemptionState(0.0, pending=(PendingPayment(cf, pay_time=LAG[0] - 0.1),))


def test_tarf_redeems_on_target():
    a = tarf(target=0.075, pay_times=LAG)
    a.time_shift(0.175)
    a.record_passed_fixings(env(spot=1.20))
    fav = 1.0 * max(1.0 * (1.20 - 1.15), 0.0)
    second = min(fav, 0.075 - fav)
    assert a.is_redeemed and a.fixing_times == []
    assert a.state.accumulated == fav + second
    assert a.state.pending == (PendingPayment(second, pay_time=LAG[1] - 0.175),)  # gearing 1, favourable fixing


def test_record_before_the_first_fixing_changes_nothing():
    a = note(target=3.0)
    before = copy.deepcopy(a)
    a.record_passed_fixings(env())
    assert a.fixing_times == before.fixing_times and a.state == before.state and a.lifecycle_events == ()


def test_lifecycle_events_survive_deepcopy():
    a = note(target=3.0)
    a.time_shift(0.1)
    a.record_passed_fixings(env())
    assert copy.deepcopy(a).lifecycle_events == a.lifecycle_events


def test_banking_uses_quoted_spot_not_effective_spot():
    e = env(spot=1.20, spot_days=2)
    assert e.effective_spot() != e.spot
    a = note(target=3.0)
    a.time_shift(0.1)
    a.record_passed_fixings(e)
    assert a.lifecycle_events[0].fixing == e.spot
```

- [ ] **Step 2: Run to verify it fails**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_lifecycle.py -n 2 -q`
Expected: the new tests FAIL (base `record_passed_fixings` is a no-op; `state` unchanged).

- [ ] **Step 3: Implement**

Delete the `time_shift` methods of both products. Add to `TargetRedemptionProduct` (imports:
`PendingPayment, TargetRedemptionEventType` from the state module):

```python
    # -- ageing (spec section 6) --------------------------------------------------------------------------------
    def time_shift(self, time_bump: float) -> None:
        """Numeric form: move every fixing, pay and pending time by ``time_bump``. Pending cash now at or before
        valuation was paid within the bump and leaves as a PAYMENT event; fixings stepped over stay unbanked until
        :meth:`record_passed_fixings`. The dated form ages by its environment's date, so this is a no-op for it."""
        if not self.is_numeric:
            return
        self.fixing_times = [t - time_bump for t in self.fixing_times]
        self.pay_times = [t - time_bump for t in self.pay_times]
        kept, events = [], []
        for p in self.state.pending:
            t = p.pay_time - time_bump
            if is_passed_time(t):
                events.append(TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, p.amount, time=t))
            else:
                kept.append(PendingPayment(p.amount, pay_time=t))
        self.state = TargetRedemptionState(self.state.accumulated, pending=tuple(kept))
        self.lifecycle_events = self.lifecycle_events + tuple(events)
        self._refresh_numeric_horizon()
        self.validate()

    def record_passed_fixings(self, fx_env) -> None:
        """Bank every unbanked fixing at or before valuation, in order, at the QUOTED ``fx_env.spot`` (spec Q3),
        stopping once the contract is terminal; append the events; re-validate."""
        self._record_numeric(fx_env)
        self.validate()

    def _record_numeric(self, fx_env) -> None:
        spot = fx_env.spot
        acc = self.state.accumulated
        pending = list(self.state.pending)
        events: List[TargetRedemptionEvent] = []
        fixings, pays = self.fixing_times, self.pay_times
        factors = self.accrual_factors if self._has_accrual else None
        banked = 0
        while banked < len(fixings) and is_passed_time(fixings[banked]):
            t, tp = fixings[banked], pays[banked]
            added, cash = self._fixing_outcome(factors[banked] if factors is not None else None, spot, acc)
            acc = acc + added
            hit = is_target_hit(acc, self.target)
            ends = hit or banked == len(fixings) - 1
            label = event_label(banked, time=t)
            events.append(TargetRedemptionEvent(TargetRedemptionEventType.FIXING, cash, fixing=spot, time=t,
                                                redeemed=ends, label=label))
            self._settle_numeric(cash, tp, label, pending, events)
            if ends:
                self._settle_numeric(self._principal_amount(), tp, label, pending, events)
            banked += 1
            if hit:
                banked = len(fixings)  # every later fixing is dead
                break
        self.fixing_times = fixings[banked:]
        self.pay_times = pays[banked:]
        if factors is not None:
            self.accrual_factors = factors[banked:]
        self.state = TargetRedemptionState(acc, pending=tuple(sorted(pending, key=lambda p: p.pay_time)))
        self.lifecycle_events = self.lifecycle_events + tuple(events)
        self._refresh_numeric_horizon()

    @staticmethod
    def _settle_numeric(amount: float, pay_time: float, label: str, pending: list, events: list) -> None:
        """Determined cash: paid now (PAYMENT, the fixing's label) when its pay time has passed, else pending."""
        if is_zero(amount):
            return
        if is_passed_time(pay_time):
            events.append(TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, amount, time=pay_time,
                                                label=label))
        else:
            pending.append(PendingPayment(amount, pay_time=pay_time))
```

- [ ] **Step 4: Run to verify it passes**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_lifecycle.py test/test_fx_tarn_note_mc.py test/test_fx_tarf_mc.py -n 2 -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/asset/fx/product/option/target_redemption_base.py quantark/asset/fx/product/option/fx_target_redemption_note.py quantark/asset/fx/product/option/fx_target_redemption_forward.py test/test_fx_target_redemption_lifecycle.py
git -C <WT> commit -m "feat(fx-tarn): time_shift and record_passed_fixings bank fixings into the realized state

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Resolved view and engines (P1–P5, T1, T2, WARN1–3 numeric)

Spec §5.1, §5.2, §5.3, §8, §10 P/T/WARN/C1.

**Files:**
- Modify: `target_redemption_base.py` (add `resolve`, `_has_unbanked_passed`, `_view`, `pending_pv`), `quantark/asset/fx/engine/mc/fx_tarn_note_mc_engine.py`, `quantark/asset/fx/engine/mc/fx_tarf_mc_engine.py`, `test/test_fx_tarn_note_mc.py` (`_det_pv` gains `acc0`), `CHANGELOG.md`
- Test: `test/test_fx_target_redemption_lifecycle.py`, `test/test_fx_tarn_note_mc.py`

**Interfaces:**
- Consumes: Tasks 2–5.
- Produces: `TargetRedemptionProduct.resolve(fx_env, *, warn_missing_fixings: bool = True) -> ResolvedTargetRedemption`; `TargetRedemptionProduct.pending_pv(fx_env) -> float`. Engines: `_accumulate(self, opt, view, spots, df_pay)`; `_HIT_RTOL` and `_hit_tol` removed from both engines.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_fx_target_redemption_lifecycle.py` (imports: `from fx_target_redemption_fixtures import engine_for, mc`,
`from quantark.asset.fx.engine.mc import FxTargetRedemptionNoteMCEngine, FxTarnForwardMCEngine, FxTarnMCResult`,
`from quantark.util.enum.engine_enums import MonteCarloMethod`, `from quantark.util.fixings import MissingFixingWarning`):

```python
PSEUDO, QUASI = MonteCarloMethod.PSEUDO, MonteCarloMethod.QUASI


def _age(product, tau, spot):
    aged = copy.deepcopy(product)
    aged.time_shift(tau)
    aged.record_passed_fixings(env(spot=spot))
    return aged


def _rebuild_note(aged, source, *, accrual_start=None):
    kw = dict(fixing_times=list(aged.fixing_times), pay_times=list(aged.pay_times), coupon_rate=source.coupon_rate,
              notional=source.notional, target=source.target, strike=source.strike,
              include_principal=source.include_principal, currency_pair=PAIR, state=aged.state)
    if accrual_start is None:
        kw["accrual_factors"] = list(aged.accrual_factors)  # C0's surviving factors
    else:
        kw["accrual_start"] = accrual_start
    return FxTargetRedemptionNote(**kw)


def _rebuild_tarf(aged, source):
    return FxTargetRedemptionForward(strike=source.strike, fixing_times=list(aged.fixing_times),
                                     pay_times=list(aged.pay_times), notional=source.notional, target=source.target,
                                     gearing=source.gearing, currency_pair=PAIR, state=aged.state)


def _price(product, method=PSEUDO, e=None, n=20_000):
    return engine_for(product, n=n, method=method).price(product, e or env())


# --- P1-P5: numeric-vs-aged parity (A1.1) --------------------------------------------------------------------------

@pytest.mark.parametrize("method", [PSEUDO, QUASI])
@pytest.mark.parametrize("spot_h", [1.20, 1.05])
@pytest.mark.parametrize("tau", [0.125, 91 / 365])
def test_p1a_tarn_aged_equals_rebuilt_from_surviving_factors(tau, spot_h, method):
    c0 = note(target=3.0)
    aged = _age(c0, tau, spot_h)
    assert _price(aged, method) == _price(_rebuild_note(aged, c0), method)


@pytest.mark.parametrize("tau, n_banked, bitwise", [(0.125, 1, True), (91 / 365, 2, False)])
def test_p1b_tarn_rebuilt_from_accrual_start(tau, n_banked, bitwise):
    c0 = note(target=3.0)
    aged = _age(c0, tau, 1.20)
    assert len(aged.fixing_times) == 12 - n_banked
    p_aged = _price(aged)
    p_rebuilt = _price(_rebuild_note(aged, c0, accrual_start=FIX[n_banked - 1] - tau))  # Ruling R4
    assert p_aged == p_rebuilt if bitwise else is_close(p_aged, p_rebuilt, rel_tol=1e-12)


@pytest.mark.parametrize("tau", [0.1, 0.125, 91 / 365])  # Ruling R10: tau = 0.1 leaves the coupon pending
def test_p2_pay_lag_pending_is_stated_and_prices_bitwise(tau):
    c0 = note(target=3.0, pay_times=LAG)
    aged = _age(c0, tau, 1.20)
    if tau == 0.1:
        c = c0.full_coupon(0)
        stated = TargetRedemptionState(c, pending=(PendingPayment(c, pay_time=LAG[0] - 0.1),))
        assert aged.state == stated
    rebuilt = _rebuild_note(aged, c0)
    assert _price(aged) == _price(rebuilt)


def test_p3_across_redemption_aged_and_rebuilt_terminal():
    c0 = note(target=1.5, pay_times=LAG)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TerminalBookingWarning)  # WARN2: an aged copy that redeems does not warn
        aged = _age(c0, 0.175, 1.20)
        e = env()
        p_aged = _price(aged, e=e)
    c1 = min(c0.full_coupon(1), 1.5 - c0.full_coupon(0))
    t_pay = LAG[1] - 0.175
    assert p_aged == 0.0 + c1 * e.get_domestic_df(t_pay) + 100.0 * e.get_domestic_df(t_pay)
    with pytest.warns(TerminalBookingWarning):  # WARN1
        rebuilt = note(fix=[], accrual_factors=[], target=1.5, state=aged.state)
    assert _price(rebuilt, e=e) == p_aged


@pytest.mark.parametrize("method", [PSEUDO, QUASI])
@pytest.mark.parametrize("spot_h", [1.20, 1.10])
@pytest.mark.parametrize("tau", [0.1, 0.125, 91 / 365])
def test_p4_tarf_aged_equals_rebuilt(tau, spot_h, method):
    c0 = tarf(target=0.20, gearing=2.0, pay_times=LAG)
    aged = _age(c0, tau, spot_h)
    if tau == 0.1 and spot_h == 1.10:
        assert aged.state.pending and aged.state.pending[0].amount < 0.0
    assert _price(aged, method) == _price(_rebuild_tarf(aged, c0), method)


def test_p4_tarf_across_redemption():
    c0 = tarf(target=0.075, pay_times=LAG)
    aged = _age(c0, 0.175, 1.20)
    e = env()
    fav = 1.0 * max(1.0 * (1.20 - 1.15), 0.0)
    second = min(fav, 0.075 - fav)
    assert aged.is_terminal and _price(aged, e=e) == 0.0 + second * e.get_domestic_df(LAG[1] - 0.175)
    with pytest.warns(TerminalBookingWarning):
        rebuilt = tarf(fix=[], target=0.075, state=aged.state)
    assert _price(rebuilt, e=e) == _price(aged, e=e)


def test_p5_daily_ageing_equals_one_rebuild():
    c0 = note(target=3.0)
    f0 = tarf(target=0.20)
    aged_n, aged_f = copy.deepcopy(c0), copy.deepcopy(f0)
    e = env()
    for _ in range(91):
        for a in (aged_n, aged_f):
            a.time_shift(1 / 365)
            a.record_passed_fixings(e)
    tau = 91 / 365
    rebuilt_n = note(fix=[t - tau for t in FIX[2:]], accrual_factors=c0.accrual_factors[2:], target=3.0,
                     state=TargetRedemptionState(c0.full_coupon(0) + c0.full_coupon(1)))
    fav = 1.0 * max(1.0 * (1.20 - 1.15), 0.0)
    rebuilt_f = tarf(fix=[t - tau for t in FIX[2:]], target=0.20, state=TargetRedemptionState(fav + fav))
    assert aged_n.state.accumulated == rebuilt_n.state.accumulated
    assert aged_f.state.accumulated == rebuilt_f.state.accumulated
    assert is_close(_price(aged_n), _price(rebuilt_n), rel_tol=1e-12)
    assert is_close(_price(aged_f), _price(rebuilt_f), rel_tol=1e-12)


# --- WARN1 values (numeric) and terminal engine result -------------------------------------------------------------

def test_warn1_terminal_values_and_no_repeat_warning():
    with pytest.warns(TerminalBookingWarning):
        empty = note(fix=[], accrual_factors=[], target=3.0, state=TargetRedemptionState(1.0))
        paid_out = tarf(fix=[], target=0.2, state=TargetRedemptionState(0.2))
        owed = note(fix=[], accrual_factors=[], target=1.5,
                    state=TargetRedemptionState(1.5, pending=(PendingPayment(100.5, pay_time=0.05),)))
    e = env()
    with warnings.catch_warnings():
        warnings.simplefilter("error", TerminalBookingWarning)
        assert _price(empty, e=e) == 0.0
        assert _price(paid_out, e=e) == 0.0
        assert _price(owed, e=e) == 0.0 + 100.5 * e.get_domestic_df(0.05)
        for product in (empty, paid_out, owed):
            engine_for(product).calculate_greeks(product, e)


def test_terminal_engine_result_fields():
    with pytest.warns(TerminalBookingWarning):
        redeemed = note(fix=[], accrual_factors=[], target=1.5, state=TargetRedemptionState(1.5))
    eng = FxTargetRedemptionNoteMCEngine(mc())
    assert eng.price(redeemed, env()) == 0.0
    assert eng.get_last_result() == FxTarnMCResult(0.0, 0.0, 0, 0.0, 1.0)


# --- WARN3 (numeric): an unbanked passed fixing is filled once, on a copy ------------------------------------------

@pytest.mark.parametrize("make", [lambda: note(target=3.0), lambda: tarf(target=0.20)])
def test_warn3_numeric_passed_fixing_is_filled_with_one_warning(make):
    bare = make()
    bare.time_shift(0.125)
    snapshot = copy.deepcopy(bare)
    e = env(spot=1.20)
    with pytest.warns(MissingFixingWarning) as rec:
        p = _price(bare, e=e)
    missing = [w for w in rec if issubclass(w.category, MissingFixingWarning)]
    assert len(missing) == 1 and "#0" in str(missing[0].message)
    recorded = copy.deepcopy(bare)
    recorded.record_passed_fixings(e)
    assert p == _price(recorded, e=e)
    assert (bare.fixing_times, bare.pay_times, bare.state, bare.lifecycle_events) == (
        snapshot.fixing_times, snapshot.pay_times, snapshot.state, snapshot.lifecycle_events)


# --- T1, T2: theta across a fixing / past the last fixing ----------------------------------------------------------

@pytest.mark.parametrize("first", [0.5 / 365, 1 / 365])
@pytest.mark.parametrize("make", [lambda f: note(fix=f, target=3.0), lambda f: tarf(fix=f, target=0.05)])
def test_t1_theta_steps_over_a_fixing(first, make):
    product = make([first] + FIX[1:])
    greeks = engine_for(product).calculate_greeks(product, env())
    assert math.isfinite(greeks["theta"])


def test_t1_two_days_away_is_mains_theta_to_the_bit():
    fix = [2 / 365] + FIX[1:]
    n, f = note(fix=fix, target=3.0), tarf(fix=fix, target=0.05)
    gn = engine_for(n).calculate_greeks(n, env())
    gf = engine_for(f).calculate_greeks(f, env())
    # main dcea77b6 (plan Task 1 Step 4): inception prices and theta unchanged to the bit
    assert (gn["price"], gn["theta"]) == (101.63518163030555, 0.01686139016959487)
    assert (gf["price"], gf["theta"]) == (0.01925287164907336, 0.003076911159477322)


def test_t2_bump_past_the_last_fixing_is_the_pending_carry():
    opt = note(fix=[0.5 / 365], pay_times=[30 / 365], target=3.0)
    e = env()
    g = engine_for(opt).calculate_greeks(opt, e)
    c = opt.full_coupon(0)  # spot 1.20 >= 1.10: the coupon pays
    t = 30 / 365 - 1 / 365
    future = 0.0 + c * e.get_domestic_df(t) + 100.0 * e.get_domestic_df(t)
    assert g["theta"] == (future - g["price"]) / 1.0


def test_t2_terminal_theta_is_pure_carry():
    with pytest.warns(TerminalBookingWarning):
        owed = note(fix=[], accrual_factors=[], target=1.5,
                    state=TargetRedemptionState(1.5, pending=(PendingPayment(100.5, pay_time=30 / 365),)))
    e = env()
    g = engine_for(owed).calculate_greeks(owed, e)
    base = 0.0 + 100.5 * e.get_domestic_df(30 / 365)
    future = 0.0 + 100.5 * e.get_domestic_df(30 / 365 - 1 / 365)
    assert g["price"] == base and g["theta"] == (future - base) / 1.0
```

In `test/test_fx_tarn_note_mc.py`, extend `_det_pv` (spec T1: "`_det_pv` ... extended with a start `acc`") and add the
small-vol landing-fixing check to `TestDeterministicLimit`:

```python
def _det_pv(forwards, df_pay, opt, acc0=0.0):
    acc, pv, alive = acc0, 0.0, True
    ...  # body unchanged


    def test_landing_fixing_theta_matches_forward_path(self):
        from copy import deepcopy
        env = _env(vol=1e-7)
        fix = [0.5 / 365, 0.25, 0.5, 0.75, 1.0]
        opt = _note(fixing_times=fix)
        theta = FxTargetRedemptionNoteMCEngine(params=_mc(num_paths=20_000)).calculate_greeks(opt, env)["theta"]
        fwd, df = _forwards_and_df(env, opt)
        base = _det_pv(fwd, df, opt)
        aged = deepcopy(opt)
        aged.time_shift(1 / 365)
        aged.record_passed_fixings(env)
        fwd_a, df_a = _forwards_and_df(env, aged)
        future = _det_pv(fwd_a, df_a, aged, acc0=aged.state.accumulated) + aged.pending_pv(env)
        assert theta == pytest.approx(future - base, abs=1e-6, rel=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_lifecycle.py test/test_fx_tarn_note_mc.py -n 2 -q`
Expected: FAIL — terminal products raise in the engine (`IndexError` / `ValidationError` on empty fixings), the P1
tests raise `ValidationError: Time to maturity must be non-negative` (engine prices passed pay times), `pending_pv`
missing.

- [ ] **Step 3: Implement the resolved view**

Add to `TargetRedemptionProduct` (imports: `from copy import deepcopy`, `from quantark.util.fixings import
warn_filled_fixings`, `ResolvedTargetRedemption, pending_present_value` from the state module):

```python
    # -- resolved view (spec section 5.1) ------------------------------------------------------------------------
    def resolve(self, fx_env, *, warn_missing_fixings: bool = True) -> ResolvedTargetRedemption:
        """The engines' only input. Unbanked fixings at or before valuation are banked on a COPY at the quoted spot
        (``record_passed_fixings``) with one ``MissingFixingWarning`` naming them (W5); the product is unchanged."""
        if self._has_unbanked_passed(fx_env):
            banked = deepcopy(self)
            start = len(banked.lifecycle_events)
            banked.record_passed_fixings(fx_env)
            if warn_missing_fixings:
                labels = [e.label for e in banked.lifecycle_events[start:]
                          if e.event_type is TargetRedemptionEventType.FIXING]
                warn_filled_fixings(type(self).__name__, "a banked outcome", labels)
            return banked._view(fx_env)
        return self._view(fx_env)

    def _has_unbanked_passed(self, fx_env) -> bool:
        return bool(self.fixing_times) and is_passed_time(self.fixing_times[0])

    def _view(self, fx_env) -> ResolvedTargetRedemption:
        st = self.state
        return ResolvedTargetRedemption(
            fixing_times=tuple(self.fixing_times),
            pay_times=tuple(self.pay_times),
            accrual_factors=tuple(self.accrual_factors) if self._has_accrual else None,
            accumulated=st.accumulated,
            pending=tuple((p.pay_time, p.amount) for p in st.pending),
            redeemed=self.is_redeemed,
            terminal=self.is_terminal,
        )

    def pending_pv(self, fx_env) -> float:
        """PV of the pending cash still due after valuation (no warning: callers have banked the passed fixings)."""
        return pending_present_value(self.resolve(fx_env, warn_missing_fixings=False).pending, fx_env)
```

- [ ] **Step 4: Make both engines read the view**

`fx_tarn_note_mc_engine.py` — import `pending_present_value` from
`quantark.asset.fx.product.option.target_redemption_state`; delete `_HIT_RTOL` and `_hit_tol`; replace the body of
`price` after the RANDOMIZED_QUASI guard up to `spots = ...` and the `_accumulate` call:

```python
        view = opt.resolve(fx_env)
        pv_pending = pending_present_value(view.pending, fx_env)
        if view.terminal:  # spec 5.2 step 2: no simulation, no vol lookup
            self._last_result = FxTarnMCResult(pv_pending, 0.0, 0, 0.0, 1.0 if view.redeemed else 0.0)
            return pv_pending

        spot = fx_env.effective_spot()
        if spot <= 0:
            raise ValidationError(f"Spot must be positive, got {spot}")
        T = float(view.fixing_times[-1])  # == opt.get_maturity at a numeric inception, bitwise
        sigma = float(fx_env.get_vol(fx_env.get_forward(T), T))
        if sigma <= 0:
            raise ValidationError(f"vol must be positive, got {sigma}")

        fixings = np.asarray(view.fixing_times, dtype=float)
        df_pay = np.array([fx_env.get_domestic_df(float(t)) for t in view.pay_times])
        spots = self._simulate(fx_env, spot, sigma, fixings)  # (P, M)

        pv, redeemed = self._accumulate(opt, view, spots, df_pay)
        if view.pending:
            pv = pv + pv_pending
```
(the rest of `price` — mean, std error, negative-price `PricingError`, `_last_result` — is unchanged). In
`_accumulate(self, opt, view, spots, df_pay)` change exactly three lines:

```python
        acc = np.full(n_paths, view.accumulated)   # was np.zeros(n_paths): identical at inception
        tol = opt.hit_tol                          # was self._hit_tol(target)
            full_c = opt.coupon_amount(view.accrual_factors[i])  # was opt.full_coupon(i)
```

`fx_tarf_mc_engine.py` — same edits (`view`, terminal branch, `T`, `fixings`, `df_pay` from the view,
`pv = pv + pv_pending` when pending, `acc = np.full(n_paths, view.accumulated)`, `tol = opt.hit_tol`, signature
`_accumulate(self, opt, view, spots, df_pay)`); delete `_HIT_RTOL` and `_hit_tol`. Update both module docstrings: "the
walk starts from the product's realized state (`resolve`); a terminal contract is the PV of its pending cash".

Append to the CHANGELOG block (create `### Fixed` after `### Changed`):

```markdown
### Fixed
- FX TARN / TARF theta within one day of a fixing raised (`Time to maturity must be non-negative` / `times must be
  strictly increasing and positive`); the stepped-over fixing is now banked at the held spot and theta is finite.
- Pricing a TARN / TARF after a bare `time_shift` past a fixing raised; the passed fixing is now filled at the quoted
  spot on a copy with one `MissingFixingWarning`.
- A rebuilt mid-life contract no longer prices with the full target: the engines start from `state.accumulated` and
  add the PV of `state.pending`. Inception prices and Greeks are unchanged to the bit.
```

- [ ] **Step 5: Run to verify it passes**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_lifecycle.py test/test_fx_tarn_note_mc.py test/test_fx_tarf_mc.py test/test_fx_target_redemption_state.py -n 2 -q` and the matrix parity `-k "Tarn or TargetRedemption"`.
Expected: all pass. Demo byte compare (Task 3 Step 7 command) → `SAME`.

- [ ] **Step 6: Commit**

```bash
git -C <WT> add quantark/asset/fx/product/option/target_redemption_base.py quantark/asset/fx/engine/mc/fx_tarn_note_mc_engine.py quantark/asset/fx/engine/mc/fx_tarf_mc_engine.py test/test_fx_target_redemption_lifecycle.py test/test_fx_tarn_note_mc.py CHANGELOG.md
git -C <WT> commit -m "feat(fx-tarn): engines price the resolved view - realized state, pending PV, terminal contracts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Dated form (D1, D3, D4 calendar, D5, R2 dated, WARN1/WARN3 dated)

Spec §4.2, §4.3, §5.1 (dated), §6 (dated), §10 D1/D3/D4/D5/R2/WARN1/WARN3.

**Files:**
- Modify: `target_redemption_base.py`, `fx_target_redemption_note.py`, `fx_target_redemption_forward.py`, `test/fx_target_redemption_fixtures.py`, `CHANGELOG.md`
- Test: `test/test_fx_target_redemption_lifecycle.py`

**Interfaces:**
- Consumes: Tasks 2–6.
- Produces:
  - TARN keywords `fixing_dates`, `pay_dates`, `initial_date`, `day_count_convention`; TARF keywords `fixing_dates`, `pay_dates`.
  - Base: `signed_year_fraction(date, fx_env) -> float`; `_schedule_form(fixing_times, fixing_dates) -> bool` (True = dated); `_init_dated`, `_refresh_dated_horizon`, `_validate_dated_schedule`, `_validate_dated_accrual` (hook, TARN overrides), `_record_dated`, `_dated_view`, `_is_unbanked(i)`, `_check_as_of(fx_env)`, `_dated_factor(index, fx_env) -> Optional[float]` (hook, TARN overrides); `get_maturity` / `get_delivery` return 0.0 past `expiry_date` / `delivery_date`.
  - Fixtures: `FIX_DATES`, `LAG2`, `TARGET_CAL`, `following(d)`, `dated_note(...)`, `dated_tarf(...)`, `_numeric_twin(dated, fx_env)`.

- [ ] **Step 1: Extend the fixtures**

Append to `test/fx_target_redemption_fixtures.py`:

```python
from datetime import timedelta

from quantark.asset.fx.product.option.target_redemption_state import PendingPayment
from quantark.util.calendar.business_calendar import CalendarType, create_calendar

FIX_DATES = [datetime(2026 + (6 + m) // 12, (6 + m) % 12 + 1, 15) for m in range(12)]  # 2026-07-15 .. 2027-06-15
LAG2 = [d + timedelta(days=2) for d in FIX_DATES]
TARGET_CAL = create_calendar(CalendarType.TARGET, year_range=(2026, 2028))


def following(d):
    """Test-side business-day adjustment (the product takes dates as stated)."""
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def dated_note(fixing_dates=FIX_DATES, state=INCEPTION, **kw):
    base = dict(fixing_dates=fixing_dates, coupon_rate=0.12, notional=100.0, strike=1.10, currency_pair=PAIR,
                state=state)
    if "accrual_factors" not in kw:
        base["initial_date"] = VAL
    base.update(kw)
    return FxTargetRedemptionNote(**base)


def dated_tarf(fixing_dates=FIX_DATES, state=INCEPTION, **kw):
    base = dict(strike=1.15, fixing_dates=fixing_dates, notional=1.0, currency_pair=PAIR, state=state)
    base.update(kw)
    return FxTargetRedemptionForward(**base)


def _numeric_twin(dated, fx_env):
    """Spec 4.3: the numeric contract with the dated contract's unbanked fixings, resolved factors and state."""
    view = dated.resolve(fx_env, warn_missing_fixings=False)
    state = TargetRedemptionState(view.accumulated,
                                  pending=tuple(PendingPayment(a, pay_time=t) for t, a in view.pending))
    common = dict(fixing_times=list(view.fixing_times), pay_times=list(view.pay_times), target=dated.target,
                  currency_pair=dated.currency_pair, state=state)
    if isinstance(dated, FxTargetRedemptionNote):
        return FxTargetRedemptionNote(coupon_rate=dated.coupon_rate, notional=dated.notional, strike=dated.strike,
                                      is_above=dated.is_above, lower=dated.lower, upper=dated.upper,
                                      include_principal=dated.include_principal,
                                      accrual_factors=list(view.accrual_factors), **common)
    return FxTargetRedemptionForward(strike=dated.strike, notional=dated.notional, gearing=dated.gearing,
                                     is_buy_base=dated.is_buy_base, **common)
```

- [ ] **Step 2: Write the failing tests**

Append to `test/test_fx_target_redemption_lifecycle.py` (imports: `dataclasses`, `from datetime import datetime,
timedelta`, the new fixtures, `from quantark.param import SpotQuote`,
`from quantark.util.calendar import DayCountConvention, calculate_year_fraction`):

```python
BUS = dict(dc=DayCountConvention.BUSINESS_DAYS, calendar=TARGET_CAL)
BUS_DATES = [following(d) for d in FIX_DATES]


def _dated(kind, dates=FIX_DATES, **kw):
    if kind == "tarf":
        return dated_tarf(fixing_dates=dates, target=kw.pop("target", 0.20), **kw)
    return dated_note(fixing_dates=dates, target=kw.pop("target", 3.0), include_principal=(kind == "tarn"), **kw)


def _at(e, days, spot=None):
    changes = {"valuation_date": e.valuation_date + timedelta(days=days)}
    if spot is not None:
        changes["spot_quote"] = SpotQuote(spot=spot)
    return dataclasses.replace(e, **changes)


# --- D1: twins bitwise at booking ----------------------------------------------------------------------------------

@pytest.mark.parametrize("calendar", ["calendar", "business"])
@pytest.mark.parametrize("method", [PSEUDO, QUASI])
@pytest.mark.parametrize("kind", ["tarn", "tarn_no_principal", "tarf"])
def test_d1_twin_prices_bitwise_at_booking(kind, method, calendar):
    e = env() if calendar == "calendar" else env(**BUS)
    dated = _dated(kind, dates=FIX_DATES if calendar == "calendar" else BUS_DATES)
    twin = _numeric_twin(dated, e)
    assert _price(dated, method, e) == _price(twin, method, e)


# --- D3: a dated booking rebuilt mid-life equals the aged contract -------------------------------------------------

def test_d3_dated_rebuild_mid_life_equals_aged_and_so_does_its_twin():
    e0 = env(spot=1.20)
    aged = dated_note(target=3.0, pay_dates=LAG2)
    twin = _numeric_twin(aged, e0)
    for k in range(1, 101):
        ek = _at(e0, k)
        aged.record_passed_fixings(ek)
        twin.time_shift(1 / 365)
        twin.record_passed_fixings(ek)
    mid = _at(e0, 100)
    rebuilt = dated_note(target=3.0, pay_dates=LAG2, state=aged.state)
    assert rebuilt.state == aged.state and aged.state.as_of == mid.valuation_date
    assert is_close(_price(rebuilt, e=mid), _price(aged, e=mid), rel_tol=1e-12)
    rebuilt_twin = _numeric_twin(rebuilt, mid)
    assert rebuilt_twin.state.accumulated == twin.state.accumulated
    assert [p.amount for p in rebuilt_twin.state.pending] == [p.amount for p in twin.state.pending]
    assert is_close(_price(rebuilt_twin, e=mid), _price(twin, e=mid), rel_tol=1e-12)


# --- D4 (CALENDAR_DAYS): theta twins agree, including a fixing inside the bump -------------------------------------

@pytest.mark.parametrize("inside_bump", [False, True])
@pytest.mark.parametrize("kind", ["tarn", "tarf"])
def test_d4_theta_twins_agree_calendar_days(kind, inside_bump):
    dates = ([VAL + timedelta(days=1)] + FIX_DATES[1:]) if inside_bump else FIX_DATES
    kw = {"initial_date": VAL - timedelta(days=29)} if (inside_bump and kind == "tarn") else {}
    dated = _dated(kind, dates=dates, **kw)
    e = env()
    twin = _numeric_twin(dated, e)
    gd = engine_for(dated).calculate_greeks(dated, e)
    gn = engine_for(twin).calculate_greeks(twin, e)
    assert gd["price"] == gn["price"]
    assert abs(gd["theta"] - gn["theta"]) <= 1e-12 * abs(gd["price"])  # Ruling R5


# --- D5: dated accrual factors -------------------------------------------------------------------------------------

def test_d5_initial_date_gives_act_365_factors():
    view = dated_note().resolve(env())
    starts = [VAL] + FIX_DATES[:-1]
    assert view.accrual_factors == tuple(
        calculate_year_fraction(s, d, DayCountConvention.ACT_365, 252) for s, d in zip(starts, FIX_DATES))


def test_d5_business_days_contract_count_reads_the_environment_calendar():
    dated = dated_note(fixing_dates=BUS_DATES, day_count_convention=DayCountConvention.BUSINESS_DAYS)
    view = dated.resolve(env(calendar=TARGET_CAL))
    starts = [VAL] + BUS_DATES[:-1]
    assert view.accrual_factors == tuple(
        calculate_year_fraction(s, d, DayCountConvention.BUSINESS_DAYS, 252, calendar=TARGET_CAL)
        for s, d in zip(starts, BUS_DATES))
    with pytest.raises(ValidationError, match="calendar"):
        dated.resolve(env())  # no calendar on the environment


def test_d5_both_period_sources_refused():
    with pytest.raises(ValidationError, match="not both"):
        dated_note(initial_date=VAL, accrual_factors=[1 / 12] * 12)


def test_d5_neither_period_source_refused():
    with pytest.raises(ValidationError, match="initial_date"):
        FxTargetRedemptionNote(fixing_dates=FIX_DATES, coupon_rate=0.12, strike=1.10, state=INCEPTION)


# --- R2 (dated and mixed forms) ------------------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [
    dict(fixing_times=FIX),                                   # both schedules
    dict(pay_times=FIX),                                      # numeric pay with dates
    dict(accrual_start=0.0),                                  # numeric start with dates
    dict(accrual_factors=[1 / 12] * 12, day_count_convention=DayCountConvention.ACT_365, initial_date=None),
    dict(initial_date=FIX_DATES[0]),                          # initial_date not before the first fixing
    dict(fixing_dates=[]),
    dict(fixing_dates=list(reversed(FIX_DATES))),
    dict(fixing_dates=FIX_DATES[:1] * 2),
    dict(pay_dates=FIX_DATES[:-1]),
    dict(pay_dates=[d - timedelta(days=1) for d in FIX_DATES]),
    dict(state=TargetRedemptionState(1.0)),                   # accumulated > 0 with as_of None
    dict(state=TargetRedemptionState(1.0, as_of=VAL)),        # as_of before the first fixing
])
def test_r2_dated_refusals(kw):
    with pytest.raises(ValidationError):
        dated_note(**kw)


@pytest.mark.parametrize("kw", [
    dict(pay_dates=FIX_DATES), dict(initial_date=VAL), dict(day_count_convention=DayCountConvention.ACT_365)])
def test_r2_dated_terms_refused_on_the_numeric_form(kw):
    with pytest.raises(ValidationError, match="dated form"):
        note(**kw)


def test_r2_neither_schedule_refused():
    with pytest.raises(ValidationError, match="schedule"):
        FxTargetRedemptionForward(strike=1.15, state=INCEPTION)


def test_r2_dated_pending_must_follow_as_of():
    with pytest.raises(ValidationError, match="after as_of"):
        dated_note(state=TargetRedemptionState(1.0, pending=(PendingPayment(1.0, pay_date=FIX_DATES[0]),),
                                               as_of=FIX_DATES[0]))


def test_r2_as_of_after_valuation_is_refused_at_pricing():
    booked = dated_note(state=TargetRedemptionState(0.0, as_of=FIX_DATES[1]))
    with pytest.raises(ValidationError, match="as_of"):
        _price(booked, e=env())


def test_future_fixing_with_zero_year_fraction_is_refused():
    friday = datetime(2026, 8, 14)
    saturday_fixing = dated_note(fixing_dates=[datetime(2026, 8, 15), datetime(2026, 9, 15)],
                                 initial_date=datetime(2026, 7, 15))
    with pytest.raises(ValidationError, match="zero year fraction"):
        saturday_fixing.resolve(env(val=friday, **BUS))


# --- dated record semantics, maturity ------------------------------------------------------------------------------

def test_dated_record_banks_through_the_valuation_date_and_sweeps_paid_cash():
    d = dated_note(target=3.0, pay_dates=LAG2)
    d.record_passed_fixings(env(val=FIX_DATES[0]))
    coupon = 100.0 * 0.12 * calculate_year_fraction(VAL, FIX_DATES[0], DayCountConvention.ACT_365, 252)
    assert d.state == TargetRedemptionState(coupon, pending=(PendingPayment(coupon, pay_date=LAG2[0]),),
                                            as_of=FIX_DATES[0])
    d.record_passed_fixings(env(val=LAG2[0]))
    assert d.state.pending == () and d.lifecycle_events[-1].event_type is PAYMENT
    assert d.state.as_of == LAG2[0]


def test_dated_time_shift_is_a_noop():
    d = dated_note()
    d.time_shift(0.5)
    assert d.state == INCEPTION and d.lifecycle_events == ()


def test_dated_maturity_and_delivery_after_the_end_are_zero():
    with pytest.warns(TerminalBookingWarning):
        done = dated_note(target=3.0, pay_dates=LAG2, state=TargetRedemptionState(1.0, as_of=FIX_DATES[-1]))
    late = env(val=LAG2[-1] + timedelta(days=5))
    assert done.get_maturity(late) == 0.0 and done.get_delivery(late) == 0.0
    assert done.get_maturity(env()) == calculate_year_fraction(VAL, FIX_DATES[-1],
                                                                DayCountConvention.CALENDAR_DAYS, 252)


# --- WARN1 / WARN3 (dated) -----------------------------------------------------------------------------------------

def test_warn1_dated_terminal_bookings():
    e = env(val=FIX_DATES[-1] + timedelta(days=1))
    with pytest.warns(TerminalBookingWarning):
        all_banked = dated_note(target=3.0, pay_dates=LAG2, state=TargetRedemptionState(
            1.0, pending=(PendingPayment(100.0, pay_date=LAG2[-1]),), as_of=FIX_DATES[-1]))
    with pytest.warns(TerminalBookingWarning):
        redeemed = dated_note(target=1.5, pay_dates=LAG2, state=TargetRedemptionState(1.5, as_of=FIX_DATES[1]))
    with warnings.catch_warnings():
        warnings.simplefilter("error", TerminalBookingWarning)
        t = calculate_year_fraction(e.valuation_date, LAG2[-1], DayCountConvention.CALENDAR_DAYS, 252)
        assert _price(all_banked, e=e) == 0.0 + 100.0 * e.get_domestic_df(t)
        assert _price(redeemed, e=e) == 0.0
        engine_for(all_banked).calculate_greeks(all_banked, e)


def test_warn3_dated_passed_fixing_is_filled_with_one_warning():
    d = dated_note(target=3.0)
    snapshot = copy.deepcopy(d)
    e = env(val=FIX_DATES[0] + timedelta(days=5), spot=1.20)
    with pytest.warns(MissingFixingWarning) as rec:
        p = _price(d, e=e)
    missing = [w for w in rec if issubclass(w.category, MissingFixingWarning)]
    assert len(missing) == 1 and FIX_DATES[0].date().isoformat() in str(missing[0].message)
    recorded = copy.deepcopy(d)
    recorded.record_passed_fixings(e)
    assert p == _price(recorded, e=e)
    assert d.state == snapshot.state and d.lifecycle_events == snapshot.lifecycle_events
```

- [ ] **Step 3: Run to verify it fails**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_lifecycle.py -n 2 -q`
Expected: the new tests FAIL (`TypeError: unexpected keyword argument 'fixing_dates'`); earlier tests still pass.

- [ ] **Step 4: Implement the dated form in the base**

Add to `target_redemption_base.py` (imports: `from quantark.util.calendar import calculate_year_fraction`):

```python
def signed_year_fraction(date: datetime, fx_env) -> float:
    """Signed year fraction from ``fx_env.valuation_date`` to ``date`` in the ENVIRONMENT's convention: 0.0 on the
    valuation date, negative before it (``FxRangeAccrualObservationRecord.resolve_time``'s rule)."""
    val = fx_env.valuation_date
    if date == val:
        return 0.0
    if date > val:
        return calculate_year_fraction(val, date, fx_env.day_count_convention, fx_env.bus_days_in_year,
                                       calendar=fx_env.calendar)
    return -calculate_year_fraction(date, val, fx_env.day_count_convention, fx_env.bus_days_in_year,
                                    calendar=fx_env.calendar)
```

Methods on `TargetRedemptionProduct` (replace `is_terminal`, `validate`, `_validate_state`, `record_passed_fixings`,
`_has_unbanked_passed`, `_view` with the dispatching versions below; add the rest):

```python
    @staticmethod
    def _schedule_form(fixing_times, fixing_dates) -> bool:
        """True for the dated form; exactly one schedule is required."""
        if fixing_times is not None and fixing_dates is not None:
            raise ValidationError("give fixing_times (numeric form) or fixing_dates (dated form), not both")
        if fixing_times is None and fixing_dates is None:
            raise ValidationError("a schedule is required: fixing_times (numeric form) or fixing_dates (dated form)")
        return fixing_dates is not None

    def _init_dated(self, fixing_dates, pay_dates, currency_pair, state: TargetRedemptionState) -> None:
        BaseFxProduct.__init__(self, currency_pair=currency_pair)
        self.fixing_dates = list(fixing_dates)
        self.pay_dates = list(pay_dates) if pay_dates is not None else list(self.fixing_dates)
        self.fixing_times = None
        self.pay_times = None
        self.state = state
        self.lifecycle_events = ()

    def _refresh_dated_horizon(self) -> None:
        self.expiry_date = self.fixing_dates[-1]
        self.delivery_date = max(self.pay_dates)

    @property
    def is_terminal(self) -> bool:
        if self.is_numeric:
            return not self.fixing_times or self.is_redeemed
        as_of = self.state.as_of
        return self.is_redeemed or (as_of is not None and as_of >= self.fixing_dates[-1])

    def validate(self) -> None:
        self._validate_terms()
        if self.is_numeric:
            self._validate_numeric_schedule()
        else:
            self._validate_dated_schedule()
        self._validate_state()

    def _validate_dated_accrual(self) -> None:
        """TARN hook: the dated period source."""

    def _validate_dated_schedule(self) -> None:
        dates, pays = self.fixing_dates, self.pay_dates
        if not dates:
            raise ValidationError("fixing_dates must be non-empty: the dated form lists the whole schedule")
        if any(not isinstance(d, datetime) for d in list(dates) + list(pays)):
            raise ValidationError("fixing_dates and pay_dates must be datetimes")
        if dates != sorted(dates):
            raise ValidationError("fixing_dates must be ascending")
        if len(set(dates)) != len(dates):
            raise ValidationError("fixing_dates must be unique")
        if len(pays) != len(dates):
            raise ValidationError("pay_dates must align 1:1 with fixing_dates")
        for f, p in zip(dates, pays):
            if p < f:
                raise ValidationError(f"pay date {p.date()} precedes its fixing {f.date()}")
        if pays != sorted(pays):
            raise ValidationError("pay_dates must be non-decreasing")
        self._validate_dated_accrual()

    def _validate_state(self) -> None:
        st = self.state
        if st.accumulated > self.target + self.hit_tol:
            raise ValidationError(f"state.accumulated {st.accumulated} exceeds the target {self.target}")
        if self._pending_must_be_positive and any(p.amount <= 0.0 for p in st.pending):
            raise ValidationError(
                "pending amounts must be positive: a TARN's determined cash is a coupon or its principal"
            )
        if self.is_numeric:
            if st.as_of is not None:
                raise ValidationError(
                    "state.as_of belongs to the dated form: a numeric state lists only unbanked fixings")
            if any(p.pay_date is not None for p in st.pending):
                raise ValidationError("a numeric contract's pending payments take pay_time, not pay_date")
            if self.is_redeemed and self.fixing_times:
                raise ValidationError(
                    "a redeemed state cannot list fixings: every fixing after the target hit is dead")
            return
        if any(p.pay_time is not None for p in st.pending):
            raise ValidationError("a dated contract's pending payments take pay_date, not pay_time")
        if (st.accumulated > 0.0 or st.pending) and (st.as_of is None or st.as_of < self.fixing_dates[0]):
            raise ValidationError(
                "accumulated or pending cash needs state.as_of on or after the first fixing date")

    def _dated_factor(self, index: int, fx_env) -> Optional[float]:
        """TARN hook: the accrual factor of scheduled period ``index``; None for the TARF."""
        return None

    def _is_unbanked(self, index: int) -> bool:
        as_of = self.state.as_of
        return as_of is None or self.fixing_dates[index] > as_of

    def _check_as_of(self, fx_env) -> None:
        as_of = self.state.as_of
        if as_of is not None and as_of > fx_env.valuation_date:
            raise ValidationError(
                f"state.as_of {as_of} is after the valuation date {fx_env.valuation_date}: the state claims "
                "fixings that have not happened")

    def get_maturity(self, fx_env=None) -> float:
        """Dated: 0.0 once valuation is past the last fixing date (an aged contract still answers FX consumers)."""
        if self.expiry_date is not None and fx_env is not None and fx_env.valuation_date > self.expiry_date:
            return 0.0
        return super().get_maturity(fx_env)

    def get_delivery(self, fx_env=None) -> float:
        """Dated: 0.0 once valuation is past the last pay date (Ruling R8: ``calculate_greeks`` reads it)."""
        if self.delivery_date is not None and fx_env is not None and fx_env.valuation_date > self.delivery_date:
            return 0.0
        return super().get_delivery(fx_env)

    def record_passed_fixings(self, fx_env) -> None:
        if self.is_numeric:
            self._record_numeric(fx_env)
        else:
            self._record_dated(fx_env)
        self.validate()

    def _record_dated(self, fx_env) -> None:
        self._check_as_of(fx_env)
        val, spot, st = fx_env.valuation_date, fx_env.spot, self.state
        events: List[TargetRedemptionEvent] = []
        pending: list = []
        for p in st.pending:  # cash due by the valuation date was paid
            if p.pay_date <= val:
                events.append(TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, p.amount,
                                                    time=signed_year_fraction(p.pay_date, fx_env)))
            else:
                pending.append(p)
        acc = st.accumulated
        if not self.is_terminal:
            last = len(self.fixing_dates) - 1
            for i, fd in enumerate(self.fixing_dates):
                if not self._is_unbanked(i):
                    continue
                if fd > val:
                    break
                t = signed_year_fraction(fd, fx_env)
                added, cash = self._fixing_outcome(self._dated_factor(i, fx_env), spot, acc)
                acc = acc + added
                hit = is_target_hit(acc, self.target)
                ends = hit or i == last
                label = event_label(i, date=fd)
                events.append(TargetRedemptionEvent(TargetRedemptionEventType.FIXING, cash, fixing=spot, time=t,
                                                    redeemed=ends, label=label))
                self._settle_dated(cash, self.pay_dates[i], label, fx_env, pending, events)
                if ends:
                    self._settle_dated(self._principal_amount(), self.pay_dates[i], label, fx_env, pending, events)
                    break  # every later fixing is dead (or there is none)
        self.state = TargetRedemptionState(acc, pending=tuple(sorted(pending, key=lambda p: p.pay_date)), as_of=val)
        self.lifecycle_events = self.lifecycle_events + tuple(events)

    @staticmethod
    def _settle_dated(amount, pay_date, label, fx_env, pending: list, events: list) -> None:
        if is_zero(amount):
            return
        if pay_date <= fx_env.valuation_date:
            events.append(TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, amount,
                                                time=signed_year_fraction(pay_date, fx_env), label=label))
        else:
            pending.append(PendingPayment(amount, pay_date=pay_date))

    def _has_unbanked_passed(self, fx_env) -> bool:
        if self.is_numeric:
            return bool(self.fixing_times) and is_passed_time(self.fixing_times[0])
        self._check_as_of(fx_env)
        if self.is_terminal:
            return False
        val = fx_env.valuation_date
        return any(self._is_unbanked(i) and d <= val for i, d in enumerate(self.fixing_dates))

    def _view(self, fx_env) -> ResolvedTargetRedemption:
        if self.is_numeric:
            st = self.state
            return ResolvedTargetRedemption(
                fixing_times=tuple(self.fixing_times), pay_times=tuple(self.pay_times),
                accrual_factors=tuple(self.accrual_factors) if self._has_accrual else None,
                accumulated=st.accumulated, pending=tuple((p.pay_time, p.amount) for p in st.pending),
                redeemed=self.is_redeemed, terminal=self.is_terminal)
        return self._dated_view(fx_env)

    def _dated_view(self, fx_env) -> ResolvedTargetRedemption:
        val, st = fx_env.valuation_date, self.state
        fix_t: List[float] = []
        pay_t: List[float] = []
        factors: List[float] = []
        if not self.is_terminal:
            for i, d in enumerate(self.fixing_dates):
                if not self._is_unbanked(i):
                    continue
                t = signed_year_fraction(d, fx_env)
                if is_passed_time(t):  # Ruling R7: after valuation by date, zero by the environment's day count
                    raise ValidationError(
                        f"fixing date {d.date()} is after the valuation date {val} but resolves to a zero year "
                        f"fraction under the environment's {fx_env.day_count_convention.name} day count")
                fix_t.append(t)
                pay_t.append(signed_year_fraction(self.pay_dates[i], fx_env))
                if self._has_accrual:
                    factors.append(self._dated_factor(i, fx_env))
        return ResolvedTargetRedemption(
            fixing_times=tuple(fix_t), pay_times=tuple(pay_t),
            accrual_factors=tuple(factors) if self._has_accrual else None,
            accumulated=st.accumulated,
            pending=tuple((signed_year_fraction(p.pay_date, fx_env), p.amount) for p in st.pending
                          if p.pay_date > val),
            redeemed=self.is_redeemed, terminal=self.is_terminal)
```

- [ ] **Step 5: Dated constructors**

TARN `__init__` gains keywords `fixing_dates: Optional[List[datetime]] = None, pay_dates: Optional[List[datetime]] =
None, initial_date: Optional[datetime] = None, day_count_convention: Optional[DayCountConvention] = None` (import
`DayCountConvention, calculate_year_fraction` from `quantark.util.calendar`) and replaces the part after
`state = self._require_state(state)` with:

```python
        if self._schedule_form(fixing_times, fixing_dates):
            for name, value in (("pay_times", pay_times), ("accrual_start", accrual_start)):
                if value is not None:
                    raise ValidationError(
                        f"{name} belongs to the numeric form; a dated TARN takes pay_dates and initial_date")
            if initial_date is None and accrual_factors is None:
                raise ValidationError(
                    "a dated TARN derives its periods from initial_date or states explicit accrual_factors")
            if initial_date is not None and accrual_factors is not None:
                raise ValidationError("give initial_date or accrual_factors, not both")
            if accrual_factors is not None and day_count_convention is not None:
                raise ValidationError(
                    "day_count_convention sets the derived accrual factors; it cannot be combined with explicit "
                    "accrual_factors")
            self.initial_date = initial_date
            self.day_count_convention = day_count_convention
            self.accrual_factors = [float(a) for a in accrual_factors] if accrual_factors is not None else None
            self._init_dated(fixing_dates, pay_dates, currency_pair, state)
            self.validate()
            self._refresh_dated_horizon()
        else:
            for name, value in (("pay_dates", pay_dates), ("initial_date", initial_date),
                                ("day_count_convention", day_count_convention)):
                if value is not None:
                    raise ValidationError(
                        f"{name} belongs to the dated form; a numeric TARN takes pay_times and accrual_start")
            self.initial_date = None
            self.day_count_convention = None
            fixings = [float(t) for t in fixing_times]
            self.accrual_factors = self._numeric_accruals(fixings, accrual_start, accrual_factors, state)
            self._init_numeric(fixings, pay_times, currency_pair, state)
            self.validate()
            self._refresh_numeric_horizon()
        self._warn_if_terminal_booking()
```

TARN hooks:

```python
    def _validate_dated_accrual(self) -> None:
        if self.initial_date is not None and not self.initial_date < self.fixing_dates[0]:
            raise ValidationError(
                f"initial_date {self.initial_date} must be before the first fixing date {self.fixing_dates[0]}")
        if self.accrual_factors is not None:
            if len(self.accrual_factors) != len(self.fixing_dates):
                raise ValidationError("accrual_factors must align 1:1 with fixing_dates")
            if any((not math.isfinite(a)) or a <= 0.0 for a in self.accrual_factors):
                raise ValidationError("accrual_factors must be positive and finite")

    def _dated_factor(self, index: int, fx_env) -> float:
        """Explicit factor, else ``initial_date`` / previous fixing date -> this fixing date in the contract day count
        (``None`` = ACT_365), resolved against the environment's calendar (a BUSINESS_DAYS count needs it)."""
        if self.accrual_factors is not None:
            return self.accrual_factors[index]
        start = self.initial_date if index == 0 else self.fixing_dates[index - 1]
        convention = DayCountConvention.ACT_365 if self.day_count_convention is None else self.day_count_convention
        return calculate_year_fraction(start, self.fixing_dates[index], convention, fx_env.bus_days_in_year,
                                       calendar=fx_env.calendar)
```

`full_coupon` on a dated TARN without explicit factors: raise `ValidationError("a dated TARN without explicit
accrual_factors resolves its factors against an environment: use resolve(fx_env).accrual_factors")` when
`self.accrual_factors is None`. `__repr__` uses `len(self.fixing_times if self.is_numeric else self.fixing_dates)`.

TARF `__init__` gains `fixing_dates=None, pay_dates=None` keywords; after `_require_state`:

```python
        if self._schedule_form(fixing_times, fixing_dates):
            if pay_times is not None:
                raise ValidationError("pay_times belongs to the numeric form; a dated TARF takes pay_dates")
            self._init_dated(fixing_dates, pay_dates, currency_pair, state)
            self.validate()
            self._refresh_dated_horizon()
        else:
            if pay_dates is not None:
                raise ValidationError("pay_dates belongs to the dated form; a numeric TARF takes pay_times")
            self._init_numeric([float(t) for t in fixing_times], pay_times, currency_pair, state)
            self.validate()
            self._refresh_numeric_horizon()
        self._warn_if_terminal_booking()
```
(`__repr__` as for the TARN.) Remove the Task 3/4 `"fixing_times is required"` checks (replaced by `_schedule_form`).

Append to the CHANGELOG `### Added`:

```markdown
- Dated `FxTargetRedemptionNote` / `FxTargetRedemptionForward`: `fixing_dates` (the whole contractual schedule),
  `pay_dates`, `state.as_of`; the TARN derives its periods from `initial_date` (or explicit `accrual_factors`) in its
  `day_count_convention` (default ACT/365). Times resolve in the environment's convention; a numeric contract and its
  dated twin price identically.
- `resolve(fx_env)` (the engines' one input), `lifecycle_events`, `is_redeemed`, `is_terminal`, `hit_tol`,
  `pending_pv(fx_env)` on both products.
```

- [ ] **Step 6: Run to verify it passes**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_lifecycle.py test/test_fx_target_redemption_state.py test/test_fx_tarn_note_mc.py test/test_fx_tarf_mc.py -n 2 -q`
Expected: all pass. If `test_d4_*` fails by more than R5's bound, STOP and report the measured gap (do not loosen it).

- [ ] **Step 7: Commit**

```bash
git -C <WT> add quantark/asset/fx/product/option/target_redemption_base.py quantark/asset/fx/product/option/fx_target_redemption_note.py quantark/asset/fx/product/option/fx_target_redemption_forward.py test/fx_target_redemption_fixtures.py test/test_fx_target_redemption_lifecycle.py CHANGELOG.md
git -C <WT> commit -m "feat(fx-tarn): dated TARN/TARF - whole schedule, as_of state, env-resolved times, twin parity

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Q7 — numeric FX theta steps by the environment's year fraction

Spec §6 (last bullet), §12 Q7, §10 D4.

**Files:**
- Create: `docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/q7_theta_plugin.py`, `q7_summarize.py`, `moved_values.txt`
- Modify: `quantark/asset/fx/engine/base_fx_engine.py:219-241`, `CHANGELOG.md`
- Test: `test/test_fx_target_redemption_lifecycle.py`, `test/test_fx_base_engine.py`

**Interfaces:**
- Consumes: Tasks 2–7 (dated twins for D4).
- Produces: `BaseFxEngine._fdm_theta` numeric step `calculate_year_fraction(V, V + timedelta(days=theta_days), env.day_count_convention, env.bus_days_in_year, calendar=env.calendar)`.

- [ ] **Step 1: Write the measurement plugin**

`q7_theta_plugin.py`:

```python
"""Q7 measurement (plan 2026-10-09-fx-tarn-lifecycle, Task 8): for every BaseFxEngine._fdm_theta call on a NUMERIC FX
product, price theta with the current step (theta_days / 365) and with the Q7 step (the environment's year fraction of
theta_days calendar days) and log every call where the two steps differ. Equal steps give bitwise-equal theta (same
code path), so only differing steps are logged.

Controller run (full suite, before or after the Q7 change - both thetas are computed explicitly):
  env PYTHONPATH=<WT>:<WT>/test:<WT>/docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence \
      Q7_LOG_DIR=<WT>/.superpowers/w4b/q7 /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test \
      -p q7_theta_plugin -n 6 -q
"""
import json
import os
from copy import deepcopy
from datetime import timedelta

import pytest

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine
from quantark.util.calendar import calculate_year_fraction

_LOG_DIR = os.environ.get("Q7_LOG_DIR")
_ORIG = BaseFxEngine._fdm_theta
_CURRENT = {"nodeid": "<collection>"}


def _theta_with_step(engine, product, fx_env, base_price, dt):
    shifted = deepcopy(product)
    shifted.time_shift(dt)
    shifted.record_passed_fixings(fx_env)
    return (engine.price(shifted, fx_env) - base_price) / engine.params.theta_days


def _spy(self, product, fx_env, base_price):
    result = _ORIG(self, product, fx_env, base_price)
    if _LOG_DIR and product.maturity is not None:
        days = self.params.theta_days
        val = fx_env.valuation_date
        old_dt = days / 365.0
        new_dt = calculate_year_fraction(val, val + timedelta(days=days), fx_env.day_count_convention,
                                         fx_env.bus_days_in_year, calendar=fx_env.calendar)
        if new_dt != old_dt:
            record = {
                "nodeid": _CURRENT["nodeid"], "engine": type(self).__name__, "product": type(product).__name__,
                "convention": fx_env.day_count_convention.name, "valuation_date": str(val),
                "theta_days": days, "old_dt": old_dt, "new_dt": new_dt,
                "old_theta": _theta_with_step(self, product, fx_env, base_price, old_dt),
                "new_theta": _theta_with_step(self, product, fx_env, base_price, new_dt),
            }
            os.makedirs(_LOG_DIR, exist_ok=True)
            with open(os.path.join(_LOG_DIR, f"{os.getpid()}.jsonl"), "a") as fh:
                fh.write(json.dumps(record) + "\n")
    return result


BaseFxEngine._fdm_theta = _spy


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    _CURRENT["nodeid"] = item.nodeid
```

`q7_summarize.py`:

```python
"""Summarize the Q7 plugin logs into moved_values.txt lines (one per distinct test / engine / product / step)."""
import glob
import json
import os
import sys

rows = {}
for path in glob.glob(os.path.join(sys.argv[1], "*.jsonl")):
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            key = (r["nodeid"], r["engine"], r["product"], r["convention"], r["valuation_date"], r["new_dt"])
            rows[key] = r
print(f"# Q7 numeric FX theta moves: {len(rows)} distinct calls (old step theta_days/365 -> env year fraction)")
for key in sorted(rows):
    r = rows[key]
    print(f"Q7 | {r['nodeid']} | {r['engine']} / {r['product']} | {r['convention']} @ {r['valuation_date']} "
          f"| dt {r['old_dt']!r} -> {r['new_dt']!r} | theta {r['old_theta']!r} -> {r['new_theta']!r}")
```

- [ ] **Step 2: Measure (controller step, full suite at `-n 6`)**

Run the plugin command from the docstring, then:
```bash
env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python <WT>/docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/q7_summarize.py <WT>/.superpowers/w4b/q7 > <WT>/.superpowers/w4b/q7_moves.txt
git -C <WT> checkout -- example/mo_volmodels/data/mo_barrier_sample.json example/mo_volmodels/data/mo_calibration_explainer_sample.html
```
Expected: a list (possibly empty) of numeric FX thetas under non-calendar environments. Create `moved_values.txt`:

```text
# W4b (FX TARN / TARF lifecycle) - existing values that move, OLD -> NEW, for user sign-off BEFORE landing.
# Format: source | site | quantity | old -> new | reason
#
# Q7 (spec section 12): numeric FX theta (BaseFxEngine._fdm_theta) steps by the environment's year fraction.
# Measured by q7_theta_plugin.py over the full suite on <branch sha>:
<paste q7_moves.txt here>
#
# Q7 scope note (Ruling R9): localvol_greeks.py:60 and the Heston / Heston-SLV PDE thetas keep theta_days/365
# (not BaseFxEngine._fdm_theta); their numeric/dated gap under non-calendar envs is a follow-up, not changed here.
# Non-integral theta_days: the numeric step becomes the env year fraction of a partial day (0.0 under CALENDAR_DAYS),
# as the dated branch already computes; no test in the suite sets theta_days (grep: none).
```

- [ ] **Step 3: Write the failing tests**

Append to `test/test_fx_base_engine.py` (imports: `from datetime import datetime, timedelta`,
`from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine`, `FxVanillaOption`, `CurrencyPair`,
`OptionType`, `FlatRateCurve`, `FlatVolSurface`, `SpotQuote`, `FxPricingEnvironment`,
`from quantark.util.calendar.day_counter import DayCountConvention as DC`,
`from quantark.util.calendar.business_calendar import CalendarType, create_calendar`):

```python
_FRIDAY = datetime(2026, 6, 12)


def _q7_env(dc, cal=None):
    return FxPricingEnvironment(valuation_date=_FRIDAY, spot_quote=SpotQuote(spot=1.20),
                                domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.03),
                                vol_surface=FlatVolSurface(volatility=0.10), day_count_convention=dc, calendar=cal)


def _q7_pair(e):
    dated = FxVanillaOption(strike=1.2, option_type=OptionType.CALL, expiry_date=_FRIDAY + timedelta(days=365),
                            currency_pair=CurrencyPair("EUR", "USD"), notional_foreign=1.0)
    numeric = FxVanillaOption(strike=1.2, option_type=OptionType.CALL, maturity=dated.get_maturity(e),
                              currency_pair=CurrencyPair("EUR", "USD"), notional_foreign=1.0)
    return numeric, dated


def test_q7_business_days_numeric_theta_matches_dated():  # M9: was -9.135e-05 vs 0.0
    e = _q7_env(DC.BUSINESS_DAYS, create_calendar(CalendarType.TARGET, year_range=(2026, 2028)))
    numeric, dated = _q7_pair(e)
    eng = GarmanKohlhagenEngine()
    tn = eng._fdm_theta(numeric, e, eng.price(numeric, e))
    td = eng._fdm_theta(dated, e, eng.price(dated, e))
    assert tn == td == 0.0  # Friday -> Saturday adds no business day


def test_q7_calendar_days_step_is_unchanged_bitwise():
    e = _q7_env(DC.CALENDAR_DAYS)
    numeric, _ = _q7_pair(e)
    eng = GarmanKohlhagenEngine()
    base = eng.price(numeric, e)
    manual = FxVanillaOption(strike=1.2, option_type=OptionType.CALL, maturity=numeric.maturity - 1.0 / 365.0,
                             currency_pair=CurrencyPair("EUR", "USD"), notional_foreign=1.0)
    assert eng._fdm_theta(numeric, e, base) == (eng.price(manual, e) - base) / 1.0


def test_q7_act_360_steps_one_360th():
    e = _q7_env(DC.ACT_360)
    numeric, _ = _q7_pair(e)
    eng = GarmanKohlhagenEngine()
    base = eng.price(numeric, e)
    manual = FxVanillaOption(strike=1.2, option_type=OptionType.CALL, maturity=numeric.maturity - 1.0 / 360.0,
                             currency_pair=CurrencyPair("EUR", "USD"), notional_foreign=1.0)
    assert eng._fdm_theta(numeric, e, base) == (eng.price(manual, e) - base) / 1.0
```

Append to `test/test_fx_target_redemption_lifecycle.py`:

```python
@pytest.mark.parametrize("kind", ["tarn", "tarf"])
@pytest.mark.parametrize("val", [datetime(2026, 6, 12), datetime(2026, 6, 15)])  # Friday, Monday
def test_d4_theta_twins_agree_business_days(kind, val):
    e = env(val=val, **BUS)
    dated = _dated(kind, dates=BUS_DATES, **({"initial_date": val} if kind != "tarf" else {}))
    twin = _numeric_twin(dated, e)
    gd = engine_for(dated).calculate_greeks(dated, e)
    gn = engine_for(twin).calculate_greeks(twin, e)
    assert abs(gd["theta"] - gn["theta"]) <= 1e-12 * abs(gd["price"])  # Ruling R5
```

- [ ] **Step 4: Run to verify they fail**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_base_engine.py test/test_fx_target_redemption_lifecycle.py -k "q7 or business_days" -n 2 -q`
Expected: `test_q7_business_days_*`, `test_q7_act_360_*` and `test_d4_*business_days[*Friday*]` FAIL (numeric
step is still 1/365); `test_q7_calendar_days_*` passes.

- [ ] **Step 5: Implement**

In `base_fx_engine.py` (imports: `from datetime import timedelta`, `from quantark.util.calendar import
calculate_year_fraction`; drop the local `from datetime import timedelta`):

```python
    def _fdm_theta(
        self,
        product: BaseFxProduct,
        fx_env: FxPricingEnvironment,
        base_price: float,
    ) -> float:
        days = self.params.theta_days
        future_date = fx_env.valuation_date + timedelta(days=days)

        shifted = deepcopy(product)
        if product.maturity is not None:
            # Q7 (user decision 2026-10-09): a numeric product ages by the environment's year fraction of the bump,
            # the step its dated twin's times take when the env date moves; days/365 bitwise under CALENDAR_DAYS /
            # ACT_365.
            shifted.time_shift(calculate_year_fraction(
                fx_env.valuation_date, future_date, fx_env.day_count_convention, fx_env.bus_days_in_year,
                calendar=fx_env.calendar))
            env_future = fx_env
        else:
            env_future = deepcopy(fx_env)
            env_future.valuation_date = future_date
        shifted.record_passed_fixings(env_future)
        future_price = self.price(shifted, env_future)

        # Forward difference: daily decay
        return (future_price - base_price) / days
```

Append to the CHANGELOG `### Changed`:

```markdown
- FX finite-difference theta (`BaseFxEngine._fdm_theta`) steps a numeric (`maturity=`) product by the environment's
  year fraction of `theta_days` calendar days instead of `theta_days / 365` (user decision Q7). Unchanged to the bit
  under `CALENDAR_DAYS` / `ACT_365`; under `BUSINESS_DAYS` a Friday or holiday theta is 0.0 and `ACT_360` steps 1/360,
  matching the dated product. Values that move are listed in
  `docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/moved_values.txt`.
```

- [ ] **Step 6: Run the FX gates and record moved expectations**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_base_engine.py test/test_fx_target_redemption_lifecycle.py test/test_fx_tarn_note_mc.py test/test_fx_tarf_mc.py test/test_fx_range_accrual.py test/test_fx_range_accrual_mc.py test/test_fx_greeks_analytical_vs_fdm.py test/test_range_accrual_contract_period.py -n 2 -q`
Expected: all pass. Any existing assertion that fails ONLY because of the Q7 step: do not change the code; update
the assertion to the new value with a comment `# Q7 (2026-10-09): env year-fraction theta step - pending sign-off`,
and add `Q7 | <nodeid> | <quantity> | <old> -> <new> | numeric theta step` to `moved_values.txt`. Report every such
edit to the controller.

- [ ] **Step 7: Commit**

```bash
git -C <WT> add quantark/asset/fx/engine/base_fx_engine.py test/test_fx_base_engine.py test/test_fx_target_redemption_lifecycle.py CHANGELOG.md
git -C <WT> add -f docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/q7_theta_plugin.py docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/q7_summarize.py docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/moved_values.txt
git -C <WT> commit -m "feat(fx): numeric FX theta steps by the environment's year fraction (Q7)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
(plus any test files edited in Step 6, by path).

---

### Task 8a: Cash due today stays in today's value (OQ-1 (b), user decision 2026-10-10)

Follow-up that re-opens built Tasks 2, 5, 6 and 7. User decision OQ-1 (b): cash whose pay time is TODAY (t = 0 after
`snap_time`) is still pending and priced at `DF(0) = 1`; it is booked (a PAYMENT event, out of the value) only once its
pay time is strictly past (`is_history`). This is W8's rule (a dated contract on its date prices its settled value)
and W6 D1's (t = 0 is live) applied to TARN / TARF cash. It amends TARN spec §3 ("pending: pay time after valuation"),
§5.1 ("pay time at or before valuation is booked") and §6 (`time_shift` / `record_passed_fixings` remove cash at or
before valuation). Fixings are unchanged: a fixing at t <= 0 is still banked by `record_passed_fixings` / `resolve`
(spec §6 landing rule); only the CASH it determines now stays in the value on its pay day.

**Files:**
- Modify: `quantark/asset/fx/product/option/target_redemption_state.py` (`PendingPayment`, `TargetRedemptionState`,
  docstrings), `quantark/asset/fx/product/option/target_redemption_base.py` (`time_shift`, `_settle_numeric`,
  `_record_dated` cash sweep, `_settle_dated`, `_dated_view` pending), `CHANGELOG.md`,
  `docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/moved_values.txt`
- Test: `test/test_fx_target_redemption_state.py` (two tests flip), `test/test_fx_target_redemption_lifecycle.py` (two
  tests flip; new tests)
- Engines unchanged: `FxTargetRedemptionNoteMCEngine` / `FxTarnForwardMCEngine` add `pending_present_value(view.pending)`
  and `fx_env.get_domestic_df(0.0) == 1.0` (measured on this branch), so a t = 0 entry needs no engine change.

**Interfaces:**
- Consumes: Tasks 2–8.
- Produces (one rule, both forms, Ruling R21):
  - `PendingPayment(amount, pay_time=t)`: refused only when `t` is not finite or `is_history(t)` (strictly past);
    `t = 0.0` (and an `is_zero` time) is cash due today. Error text names `pay_time`.
  - `TargetRedemptionState`: a dated pending `pay_date` must be ON OR after `as_of` (`pay_date < as_of` refused,
    message "... must be on or after as_of ..."): `record_passed_fixings` on a pay date sets `as_of` to that date and
    keeps the cash due that day pending.
  - Numeric `time_shift` and `_settle_numeric`: cash with `is_history(pay_time)` → PAYMENT event; otherwise pending at
    `snap_time(pay_time)`.
  - Dated `_record_dated` sweep and `_settle_dated`: cash with `is_history(signed_year_fraction(pay_date, fx_env))` →
    PAYMENT event (decided by TIME, Ruling R16); otherwise pending.
  - `_dated_view`: pending entries with `is_history(t)` are booked (skipped); the others enter the view at
    `snap_time(t)` (0.0 on the pay date). The R7 refusal of a pay date AFTER the valuation date that resolves to zero
    time stays until Task 8b.
  - `ResolvedTargetRedemption.pending`: `(pay_time, amount)` with `pay_time >= 0`.

- [ ] **Step 1: Flip the tests that pin the old rule and write the new ones**

In `test/test_fx_target_redemption_state.py` replace `test_numeric_pay_time_must_be_after_valuation` and
`test_dated_pending_must_be_after_as_of` with:

```python
@pytest.mark.parametrize("t", [-0.1, math.nan, math.inf])
def test_numeric_pay_time_must_not_be_past(t):
    with pytest.raises(ValidationError, match="pay_time"):
        PendingPayment(1.0, pay_time=t)


@pytest.mark.parametrize("t", [0.0, 1e-12, -1e-12])
def test_numeric_pay_time_due_today_is_pending(t):  # OQ-1 (b), user 2026-10-10
    assert PendingPayment(1.0, pay_time=t).pay_time == t


def test_dated_pending_must_not_precede_as_of():
    with pytest.raises(ValidationError, match="after as_of"):
        TargetRedemptionState(1.0, pending=(PendingPayment(1.0, pay_date=datetime(2026, 7, 1)),), as_of=D1)


def test_dated_pending_due_on_as_of_is_allowed():  # OQ-1 (b)
    s = TargetRedemptionState(1.0, pending=(PendingPayment(1.0, pay_date=D1),), as_of=D1)
    assert s.pending[0].pay_date == D1
```

In `test/test_fx_target_redemption_lifecycle.py` replace `test_r2_dated_pending_must_follow_as_of` and
`test_dated_record_banks_through_the_valuation_date_and_sweeps_paid_cash` with:

```python
def test_r2_dated_pending_must_not_precede_as_of():
    with pytest.raises(ValidationError, match="after as_of"):
        dated_note(state=TargetRedemptionState(1.0, pending=(PendingPayment(1.0, pay_date=FIX_DATES[0]),),
                                               as_of=FIX_DATES[1]))


def test_dated_record_banks_through_the_valuation_date_and_pays_cash_once_it_is_past():
    d = dated_note(target=3.0, pay_dates=LAG2)
    d.record_passed_fixings(env(val=FIX_DATES[0]))
    coupon = 100.0 * 0.12 * calculate_year_fraction(VAL, FIX_DATES[0], DayCountConvention.ACT_365, 252)
    assert d.state == TargetRedemptionState(coupon, pending=(PendingPayment(coupon, pay_date=LAG2[0]),),
                                            as_of=FIX_DATES[0])
    d.record_passed_fixings(env(val=LAG2[0]))                    # its pay date: due today, still in the value
    assert d.state.pending == (PendingPayment(coupon, pay_date=LAG2[0]),) and d.state.as_of == LAG2[0]
    assert [e.event_type for e in d.lifecycle_events] == [FIXING]
    d.record_passed_fixings(env(val=LAG2[0] + timedelta(days=1)))
    assert d.state.pending == () and d.lifecycle_events[-1].event_type is PAYMENT
    assert d.state.as_of == LAG2[0] + timedelta(days=1)
```

Append to `test/test_fx_target_redemption_lifecycle.py`:

```python
# --- OQ-1 (b), user decision 2026-10-10: cash due today stays in today's value --------------------------------------

def test_numeric_cash_due_today_stays_pending_until_its_pay_time_is_past():
    n = note(fix=[10 / 365] + FIX[1:], pay_times=[12 / 365] + [t + 2 / 365 for t in FIX[1:]], target=3.0)
    c = n.full_coupon(0)
    n.time_shift(11 / 365)
    n.record_passed_fixings(env())
    assert [p.amount for p in n.state.pending] == [c]
    n.time_shift(1 / 365)                                        # its pay day: snapped to exactly 0.0, still pending
    assert n.state.pending == (PendingPayment(c, pay_time=0.0),)
    assert [e.event_type for e in n.lifecycle_events] == [FIXING]
    assert n.pending_pv(env()) == c
    n.time_shift(1 / 365)                                        # strictly past: paid
    assert n.state.pending == () and [e.event_type for e in n.lifecycle_events] == [FIXING, PAYMENT]


def test_a_zero_lag_fixing_banked_today_leaves_its_cash_in_the_value():
    n = note(fix=[0.0] + FIX[1:], accrual_start=-30 / 365, target=3.0)
    c = n.full_coupon(0)
    n.record_passed_fixings(env())
    assert [e.event_type for e in n.lifecycle_events] == [FIXING]
    assert n.state.pending == (PendingPayment(c, pay_time=0.0),)


@pytest.mark.parametrize("kind", ["tarn", "tarf"])
def test_value_on_a_pay_date_holds_the_cash_due_that_day(kind):
    d = dated_note(target=3.0) if kind == "tarn" else dated_tarf(target=0.20)    # pay dates = fixing dates
    e0 = env(val=FIX_DATES[0])                                                    # live fixing, its cash due today
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFixingWarning)                     # Task 9 makes a t = 0 fill silent
        view = d.resolve(e0)
        banked = copy.deepcopy(d)
        banked.record_passed_fixings(e0)
        (cash,) = [x.amount for x in banked.lifecycle_events if x.event_type is FIXING]
        assert view.pending == ((0.0, cash),) and banked.state.pending[0].pay_date == FIX_DATES[0]
        without = copy.deepcopy(banked)
        without.state = TargetRedemptionState(banked.state.accumulated, as_of=banked.state.as_of)
        assert is_close(_price(d, e=e0), _price(without, e=e0) + cash, rel_tol=1e-12)
        twin = _numeric_twin(d, e0)
        assert twin.state.pending == (PendingPayment(cash, pay_time=0.0),)
        assert is_close(_price(twin, e=e0), _price(d, e=e0), rel_tol=1e-12)


# Moved values (sign-off): measured on 51186894 (before Task 8a), MC 20_000 paths, seed 7, flat EURUSD 1.20; (price,
# theta) the day before the pay date and on it. After Task 8a: the day-before price is unchanged; the day-before theta
# gains the cash (the bump lands on the pay date, where the cash is still in the value); the pay-date price gains the
# cash; the pay-date theta loses it (the cash leaves the value the next day).
_PAY_DAY_OLD = {
    "tarn_zero_lag": ((102.09343055572586, -0.9778012736891526), (101.1156292820367, 0.015259847019819972)),
    "tarf_zero_lag": ((0.09637222423452786, -0.05217110639974772), (0.04420111783478014, 0.0008503526677482928)),
    "tarn_lag2_pending": ((102.08935210063005, -0.9702789813483861), (101.11907311928167, 0.01592892165280091)),
}


def _pay_day_case(name):
    if name == "tarn_zero_lag":
        return dated_note(target=3.0), FIX_DATES[0]
    if name == "tarf_zero_lag":
        return dated_tarf(target=0.20), FIX_DATES[0]
    lagged = dated_note(target=3.0, pay_dates=LAG2)
    lagged.record_passed_fixings(env(val=FIX_DATES[0]))
    return lagged, LAG2[0]


@pytest.mark.parametrize("name", sorted(_PAY_DAY_OLD))
def test_moved_values_around_a_pay_date(name):
    product, pay = _pay_day_case(name)
    probe = copy.deepcopy(product)
    probe.record_passed_fixings(env(val=pay))
    cash = probe.state.pending[0].amount                         # due on the pay date, still pending
    (p_before, th_before), (p_on, th_on) = _PAY_DAY_OLD[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFixingWarning)
        g_before = engine_for(product).calculate_greeks(product, env(val=pay - timedelta(days=1)))
        g_on = engine_for(product).calculate_greeks(product, env(val=pay))
    # pinned numbers were measured on ARM64; CI runs x86_64 (memory: golden cross-arch), hence 1e-9, not bitwise
    assert is_close(g_before["price"], p_before, rel_tol=1e-9)
    assert is_close(g_before["theta"], th_before + cash, rel_tol=0.0, abs_tol=1e-9)
    assert is_close(g_on["price"], p_on + cash, rel_tol=1e-9)
    assert is_close(g_on["theta"], th_on - cash, rel_tol=0.0, abs_tol=1e-9)
```

- [ ] **Step 2: Run to verify they fail**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_state.py test/test_fx_target_redemption_lifecycle.py -k "pay_time or as_of or pays_cash_once or due_today or zero_lag_fixing_banked or holds_the_cash or moved_values_around" -n 2 -q`
Expected: FAIL — `test_numeric_pay_time_due_today_is_pending[0.0]` / `[1e-12]` / `[-1e-12]` (`ValidationError: ... pay_time
must be after valuation`), `test_dated_pending_due_on_as_of_is_allowed` (`... must be after as_of`), the flipped dated
record test (PAYMENT on the pay date), the three new behaviour tests (cash paid at t = 0), and
`test_moved_values_around_a_pay_date` on everything but the day-before price. `test_numeric_pay_time_must_not_be_past`,
`test_dated_pending_must_not_precede_as_of` and `test_r2_dated_pending_must_not_precede_as_of` already pass.

- [ ] **Step 3: Implement**

`target_redemption_state.py` (import `from quantark.util.fixings import is_history`):

```python
        if self.pay_time is not None:
            t = self.pay_time
            if not math.isfinite(t) or is_history(t):
                raise ValidationError(
                    f"a numeric pending pay_time must be at or after valuation (>= 0), got {t}: cash whose pay time is "
                    "strictly past is booked, not pending (cash due today, t = 0, is still pending)"
                )
```
and in `TargetRedemptionState.__post_init__`:
```python
            late = [p.pay_date for p in self.pending if p.pay_date < self.as_of]
            if late:
                raise ValidationError(
                    f"a dated pending pay_date must be on or after as_of {self.as_of}, got {late[0]}")
```
Docstrings: module ("``pending`` is determined cash whose pay time is at or after valuation; cash due today is still
pending"), `ResolvedTargetRedemption` ("pending ``(pay_time, amount)`` with ``pay_time >= 0``").

`target_redemption_base.py` (import `is_history`, `snap_time` from `quantark.util.fixings`):

```python
    # time_shift: the pending loop
        for p in self.state.pending:
            t = p.pay_time - time_bump
            if is_history(t):  # strictly past: paid within the bump (OQ-1 b: cash due today stays pending)
                events.append(TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, p.amount, time=t))
            else:
                kept.append(PendingPayment(p.amount, pay_time=snap_time(t)))

    @staticmethod
    def _settle_numeric(amount: float, pay_time: float, label: str, pending: list, events: list) -> None:
        """Determined cash: paid now (PAYMENT, the fixing's label) when its pay time is strictly past, else pending
        (cash due today included, at exactly 0.0)."""
        if is_zero(amount):
            return
        if is_history(pay_time):
            events.append(TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, amount, time=pay_time,
                                                label=label))
        else:
            pending.append(PendingPayment(amount, pay_time=snap_time(pay_time)))

    # _record_dated: the cash sweep at its top
        for p in st.pending:  # cash whose pay time is strictly past was paid; cash due today stays pending
            t_pay = signed_year_fraction(p.pay_date, fx_env)
            if is_history(t_pay):
                events.append(TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, p.amount, time=t_pay))
            else:
                pending.append(p)

    @staticmethod
    def _settle_dated(amount: float, pay_date: datetime, label: str, fx_env, pending: list, events: list) -> None:
        """Dated twin of ``_settle_numeric``: paid now when ``pay_date``'s time is strictly past, else pending."""
        if is_zero(amount):
            return
        t = signed_year_fraction(pay_date, fx_env)
        if is_history(t):
            events.append(TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, amount, time=t, label=label))
        else:
            pending.append(PendingPayment(amount, pay_date=pay_date))

    # _dated_view: the pending loop
        for p in st.pending:
            t = signed_year_fraction(p.pay_date, fx_env)
            if is_history(t):
                continue  # strictly past: paid, booked, not pending
            if p.pay_date > val and is_passed_time(t):  # Ruling R7 applied to cash - reversed by Task 8b
                raise ValidationError(
                    f"pending pay date {p.pay_date.date()} is after the valuation date {val} but resolves to a zero "
                    f"year fraction under the environment's {fx_env.day_count_convention.name} day count")
            pending.append((snap_time(t), p.amount))
```

Append to the CHANGELOG `### Changed`:

```markdown
- FX TARN / TARF cash due on the valuation date stays in the value (user decision OQ-1 b, 2026-10-10, as W8 values a
  dated contract on its date at its settled value): determined cash is booked (a `PAYMENT` event, out of the value)
  once its pay time is strictly past. A numeric pending `pay_time` of 0.0 and a dated pending `pay_date` equal to
  `as_of` are accepted. Moves: the price on a pay date gains the cash due that day, theta the day before a pay date
  gains it and theta on the pay date loses it (values in
  `docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/moved_values.txt`); inside runs, cash on a
  path-aligned pay date is booked on the next path instant.
```

Append to `moved_values.txt`:

```text
#
# Task 8a - OQ-1 (b), user decision 2026-10-10: cash due today stays in today's value.
# Pins: test/test_fx_target_redemption_lifecycle.py::test_moved_values_around_a_pay_date (MC 20_000, seed 7, flat 1.20)
# case               cash          price d-1 (unchanged)   theta d-1  old -> new         price on pay date  old -> new     theta on pay date old -> new
# tarn_zero_lag      0.98630137    102.093431              -0.977801 ->  0.008500        101.115629 -> 102.101931         0.015260 -> -0.971042
# tarf_zero_lag      0.05000000      0.096372              -0.052171 -> -0.002171          0.044201 ->   0.094201         0.000850 -> -0.049150
# tarn_lag2_pending  0.98630137    102.089352              -0.970279 ->  0.016022        101.119073 -> 102.105374         0.015929 -> -0.970372
# Rule: every TARN / TARF valued on a pay date: price + cash due that day (x DF(0) = 1); theta the day before a pay
# date: + cash; theta on the pay date: - cash; the terminal value on the last pay date is that cash, not 0.0.
# Validation: a numeric pending pay_time of 0.0 (or is_zero) refused -> accepted; a dated pending pay_date equal to
# as_of refused -> accepted. record_passed_fixings on a pay date: PAYMENT -> still pending (PAYMENT the day after).
# Flipped tests: test_numeric_pay_time_must_be_after_valuation[0.0, 1e-12] (refused -> accepted),
# test_dated_pending_must_be_after_as_of[D1], test_r2_dated_pending_must_follow_as_of (refused -> accepted; the
# replacement tests refuse a pay_date BEFORE as_of), test_dated_record_banks_through_the_valuation_date_and_sweeps_paid_cash
# (PAYMENT on LAG2[0] -> on LAG2[0] + 1 day).
```

- [ ] **Step 4: Run to verify they pass, and that nothing else moved**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_state.py test/test_fx_target_redemption_lifecycle.py test/test_fx_tarn_note_mc.py test/test_fx_tarf_mc.py test/test_fx_base_engine.py -n 2 -q`
Expected: all pass. No other existing TARN / TARF expectation moves (checked while writing: P1–P5, T1, T2, WARN1–3,
D1–D5 and the engine suites hold no pay time landing exactly on a valuation instant, except
`test_t1_theta_steps_over_a_fixing[first=1/365]`, which asserts only that theta is finite). Any other failure: STOP and
report it as an unlisted mover.

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/asset/fx/product/option/target_redemption_state.py quantark/asset/fx/product/option/target_redemption_base.py test/test_fx_target_redemption_state.py test/test_fx_target_redemption_lifecycle.py CHANGELOG.md
git -C <WT> add -f docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/moved_values.txt
git -C <WT> commit -m "feat(fx-tarn): cash due today stays in today's value; booked once its pay time is past (OQ-1 b)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8b: Dated fixings banked per fixing; a fixing or pay date at zero business time is live (OQ-2 (b), OQ-3 ruling, 2026-10-10)

Follow-up that re-opens built Tasks 2 and 7. Two user decisions of 2026-10-10:
- **OQ-2 (b):** a dated fixing or pay date AFTER the valuation date that resolves to t = 0 in the environment's day
  count (a Saturday under `BUSINESS_DAYS`, valued on the Friday) is LIVE, like its numeric twin and the FX range accrual
  (W6 Ruling 2): the fixing is decided at the quoted spot as on its own date, its cash is due today (Task 8a). Plan
  Ruling R7's refusal is reversed. Passed vs ahead is decided by TIME (Ruling R16).
- **OQ-3 ruling (replaces the options the re-plan offered):** a dated TARN / TARF records banked fixings PER FIXING
  (by schedule index), as the FX range accrual does through W6's recorder; double-banking is prevented by fixing
  identity, not by date; `as_of` NEVER moves past the valuation date; no validation check is dropped; a Monday /
  following convention is NOT adopted (separate future decision). A weekend fixing inherits Friday's spot: on a
  `BUSINESS_DAYS` clock it is live on the Friday (W6 Ruling 4), on a calendar clock the weekend rows carry Friday's
  spot (user weekend ruling).

**What changes in the built code (exactly):**
- Task 2, `TargetRedemptionState`: new field `fixings_banked: Optional[int] = None` (after `as_of`). Dated form: the
  number of leading scheduled fixings that are banked — fixing `i` is banked iff `i < fixings_banked` (fixings bank in
  schedule order, so a count is the index set `{0, …, n − 1}`). State validation: `None` or an `int` (not `bool`)
  `>= 0`. `as_of` keeps its own checks (pending needs `as_of`; a pending `pay_date` must not precede `as_of`, Task 8a);
  its meaning becomes "the date the state was brought up to date", never after the valuation date.
- Task 7, `TargetRedemptionProduct` (dated form):
  - `_banked_count()`: `state.fixings_banked` when stated; else, for a booking that states only `as_of`, the number of
    fixing dates on or before `as_of` (exactly the spec §4.2 reading of such a booking, so every existing booking
    keeps its meaning); else 0.
  - `_is_unbanked(i)`: `i >= _banked_count()` (was: `fixing_dates[i] > as_of`).
  - `is_terminal`: redeemed, or `_banked_count() >= len(fixing_dates)` (was: `as_of >= fixing_dates[-1]`).
  - `_record_dated`: banks the leading unbanked fixings whose resolved time is passed (`is_passed_time`, OQ-2 b),
    writes `fixings_banked` = index of the last banked fixing + 1, and sets `as_of = _state_date(fx_env)`: the
    valuation date, or — valued on a day at zero time from the previous one (a Saturday under `BUSINESS_DAYS`) — the
    first date of that zero-time stretch (Ruling R24). Never after the valuation date.
  - `_has_unbanked_passed`: unbanked by count and passed by time.
  - `_dated_view`: the two R7 refusals are removed (a passed fixing is banked by `resolve` first; a pay date at zero
    time is pending at 0.0, Task 8a).
  - `_validate_state` (dated): `fixings_banked <= len(fixing_dates)`; the "cash needs a banked fixing" rule is kept and
    stated by count: accumulated > 0 or pending cash with `as_of is None` or `_banked_count() == 0` is refused (for a
    booking stating only `as_of` this is exactly the old "`as_of` before the first fixing" refusal; the date form could
    not hold once a Saturday first fixing banks on the Friday with `as_of` = Friday). Numeric: `fixings_banked` must be
    `None` (form check, like `as_of`).
  - `_check_as_of` is unchanged (an `as_of` after the valuation date is refused): `as_of` never passes it now.
- **Numeric mapping:** the numeric state carries no count (`fixings_banked` is `None`; it lists only unbanked
  fixings). Its banked fixings are the ones dropped from the listed schedule; Task 9 counts them in `_index_offset`
  (Ruling R14, booking-relative indices) and exposes both forms through one product property `fixings_banked`
  (dated: `_banked_count()`; numeric: `_index_offset`), so a numeric twin booked with its dated contract reports the
  same count at every step (asserted in Task 9's D2 test).

**Files:**
- Modify: `quantark/asset/fx/product/option/target_redemption_state.py` (`fixings_banked`),
  `quantark/asset/fx/product/option/target_redemption_base.py` (`_banked_count`, `_is_unbanked`, `is_terminal`,
  `_record_dated`, `_has_unbanked_passed`, `_dated_view`, `_validate_state`), `CHANGELOG.md`,
  `docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/moved_values.txt`
- Test: `test/test_fx_target_redemption_state.py` (new field tests), `test/test_fx_target_redemption_lifecycle.py`
  (two R7 tests flip; Task 8a's dated record test states the count; new tests incl. the OQ-3 Friday + Saturday case)

**Interfaces:**
- Consumes: Task 8a.
- Produces: `TargetRedemptionState(accumulated, pending=(), as_of=None, fixings_banked=None)`; dated
  `_banked_count() -> int`; the rules above. Every booking written before this task constructs and prices unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_fx_target_redemption_state.py`:

```python
@pytest.mark.parametrize("n", [-1, 1.5, True])
def test_fixings_banked_must_be_a_non_negative_int(n):
    with pytest.raises(ValidationError, match="fixings_banked"):
        TargetRedemptionState(1.0, as_of=D1, fixings_banked=n)


def test_fixings_banked_defaults_to_none():
    assert TargetRedemptionState(0.0).fixings_banked is None
    assert TargetRedemptionState(1.0, as_of=D1, fixings_banked=1).fixings_banked == 1
```

In `test/test_fx_target_redemption_lifecycle.py`, in Task 8a's
`test_dated_record_banks_through_the_valuation_date_and_pays_cash_once_it_is_past`, the first state assertion states
the count:

```python
    assert d.state == TargetRedemptionState(coupon, pending=(PendingPayment(coupon, pay_date=LAG2[0]),),
                                            as_of=FIX_DATES[0], fixings_banked=1)
```

Replace `test_future_fixing_with_zero_year_fraction_is_refused` and
`test_pending_pay_date_with_zero_year_fraction_is_refused` with (and append the rest):

```python
THURSDAY, FRIDAY, SATURDAY, MONDAY = (datetime(2026, 8, 13), datetime(2026, 8, 14), datetime(2026, 8, 15),
                                      datetime(2026, 8, 17))


def _business_twin_on_friday(dated):
    """The numeric twin booked on the Thursday and aged to the Friday by the business-day step (both forms are then
    valued at the Friday with the same history)."""
    twin = _numeric_twin(dated, env(val=THURSDAY, **BUS))
    twin.time_shift(calculate_year_fraction(THURSDAY, FRIDAY, DayCountConvention.BUSINESS_DAYS, 252,
                                            calendar=TARGET_CAL))
    return twin


def _kinds_only(events):
    return [(e.event_type, e.amount, e.fixing, e.redeemed) for e in events]


def test_bookings_that_state_only_as_of_keep_their_meaning():
    """OQ-3 ruling: banked is tracked per fixing; a booking stating only as_of banks the fixings dated on or before it."""
    d = dated_note(target=3.0, state=TargetRedemptionState(1.0, as_of=FIX_DATES[0] + timedelta(days=1)))
    assert d._banked_count() == 1 and not d._is_unbanked(0) and d._is_unbanked(1)
    assert dated_note(target=3.0)._banked_count() == 0
    with pytest.warns(TerminalBookingWarning):
        done = dated_note(target=3.0, state=TargetRedemptionState(1.0, as_of=FIX_DATES[-1]))
    assert done._banked_count() == 12 and done.is_terminal


def test_record_counts_banked_fixings_and_as_of_is_the_valuation_date():  # CALENDAR_DAYS: R24 is the date itself
    d = dated_note(target=3.0, pay_dates=LAG2)
    e = env(val=FIX_DATES[1] + timedelta(days=1))
    d.record_passed_fixings(e)
    assert (d.state.fixings_banked, d.state.as_of) == (2, e.valuation_date)


@pytest.mark.parametrize("kw, match", [
    (dict(state=TargetRedemptionState(1.0, as_of=FIX_DATES[0], fixings_banked=13)), "fixings_banked"),
    (dict(state=TargetRedemptionState(1.0, as_of=FIX_DATES[0], fixings_banked=0)), "as_of"),  # cash, nothing banked
])
def test_r2_dated_banked_count_refusals(kw, match):
    with pytest.raises(ValidationError, match=match):
        dated_note(target=3.0, **kw)


def test_r2_numeric_state_refuses_a_banked_count():
    with pytest.raises(ValidationError, match="fixings_banked"):
        note(state=TargetRedemptionState(0.0, fixings_banked=0))


def test_future_fixing_at_zero_business_time_is_live_like_its_twin():  # OQ-2 (b) (was refused, Ruling R7)
    d = dated_note(fixing_dates=[SATURDAY, datetime(2026, 9, 15)], initial_date=datetime(2026, 7, 15), target=3.0)
    e_fri = env(val=FRIDAY, **BUS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFixingWarning)          # Task 9 makes a t = 0 fill silent
        view = d.resolve(e_fri)                                         # banked on a copy at the quoted spot
        banked = copy.deepcopy(d)
        banked.record_passed_fixings(e_fri)
        assert view == banked.resolve(e_fri)
        c = banked.lifecycle_events[0].amount
        assert view.pending == ((0.0, c),)                              # its cash is due "today"
        assert (banked.state.fixings_banked, banked.state.as_of) == (1, FRIDAY)   # never past the valuation date
        assert is_close(_price(_business_twin_on_friday(d), e=e_fri), _price(d, e=e_fri), rel_tol=1e-12)


def test_pending_pay_date_at_zero_business_time_is_due_today():  # OQ-2 (b) (was refused, Ruling R7)
    c = 1.0
    owed = dated_note(fixing_dates=[THURSDAY, datetime(2026, 9, 15)], pay_dates=[SATURDAY, datetime(2026, 9, 17)],
                      initial_date=datetime(2026, 7, 15), target=3.0,
                      state=TargetRedemptionState(c, pending=(PendingPayment(c, pay_date=SATURDAY),), as_of=THURSDAY))
    assert owed.resolve(env(val=FRIDAY, **BUS)).pending == ((0.0, c),)
    owed.record_passed_fixings(env(val=FRIDAY, **BUS))
    assert owed.state.pending == (PendingPayment(c, pay_date=SATURDAY),)            # not paid on the Friday
    owed.record_passed_fixings(env(val=MONDAY, **BUS))
    assert owed.state.pending == () and owed.lifecycle_events[-1].event_type is PAYMENT


def test_oq3_friday_and_saturday_fixings_price_on_friday_and_bank_once():
    """OQ-3 ruling: zero-lag fixings on a Friday and the Saturday after it, under BUSINESS_DAYS. On the Friday both are
    passed by time (the Saturday one live like the Friday one, at Friday's spot): the dated contract prices, equal to
    its numeric twin, and banks them once; as_of stays the Friday, so the Friday coupon due that day does not precede
    it. On Monday nothing is banked again (fixing identity) and both coupons are paid, as in the twin. The twin is
    booked on the Friday from the dated view, so FIXING events are compared from the Friday on (Task 8c adds the
    Thursday valuation, both fixings at one business time, and twin event parity for shared fixing dates)."""
    dated = dated_note(fixing_dates=[FRIDAY, SATURDAY, datetime(2026, 9, 15)], initial_date=datetime(2026, 7, 15),
                       target=50.0)
    e_fri, e_mon = env(val=FRIDAY, **BUS), env(val=MONDAY, **BUS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFixingWarning)
        twin = _numeric_twin(dated, e_fri)
        assert is_close(_price(twin, e=e_fri), _price(dated, e=e_fri), rel_tol=1e-12)
        dated.record_passed_fixings(e_fri)
    assert [e.event_type for e in dated.lifecycle_events] == [FIXING, FIXING]
    assert (dated.state.fixings_banked, dated.state.as_of) == (2, FRIDAY)
    assert [p.pay_date for p in dated.state.pending] == [FRIDAY, SATURDAY]            # both due today, in the value
    assert dated.state.accumulated == twin.state.accumulated
    assert [p.amount for p in dated.state.pending] == [p.amount for p in twin.state.pending]
    twin.time_shift(calculate_year_fraction(FRIDAY, MONDAY, DayCountConvention.BUSINESS_DAYS, 252,
                                            calendar=TARGET_CAL))
    twin.record_passed_fixings(e_mon)
    dated.record_passed_fixings(e_mon)
    assert [e.event_type for e in dated.lifecycle_events] == [FIXING, FIXING, PAYMENT, PAYMENT]   # banked once
    assert _kinds_only(dated.lifecycle_events[2:]) == _kinds_only(twin.lifecycle_events)          # same payments
    assert (dated.state.fixings_banked, dated.state.as_of) == (2, MONDAY)


def test_as_of_on_a_weekend_valuation_is_the_friday():
    """Ruling R24: valued on the Saturday under BUSINESS_DAYS (a weekend row), a zero-lag Friday fixing is still live
    (t = 0, W6 Ruling 4) and its Friday coupon is due at that instant; banking writes as_of = the Friday, the first date
    of the valuation's zero-time stretch (never after the valuation date), so the kept check "a pending pay_date must
    not precede as_of" holds and the contract prices like its numeric twin."""
    d = dated_note(fixing_dates=[FRIDAY, datetime(2026, 9, 15)], initial_date=datetime(2026, 7, 15), target=50.0)
    e_sat = env(val=SATURDAY, **BUS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFixingWarning)
        banked = copy.deepcopy(d)
        banked.record_passed_fixings(e_sat)
        assert (banked.state.as_of, banked.state.fixings_banked) == (FRIDAY, 1)
        assert banked.state.pending[0].pay_date == FRIDAY
        twin = _numeric_twin(d, env(val=THURSDAY, **BUS))
        twin.time_shift(calculate_year_fraction(THURSDAY, SATURDAY, DayCountConvention.BUSINESS_DAYS, 252,
                                                calendar=TARGET_CAL))
        assert is_close(_price(twin, e=e_sat), _price(d, e=e_sat), rel_tol=1e-12)
```

- [ ] **Step 2: Run to verify they fail**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_state.py test/test_fx_target_redemption_lifecycle.py -k "fixings_banked or banked_count or only_as_of or counts_banked or zero_business_time or oq3 or weekend_valuation or pays_cash_once" -n 2 -q`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'fixings_banked'` (state tests, the count refusals and the
8a record test), `AttributeError: ... '_banked_count'`, `ValidationError: ... resolves to a zero year fraction` (both
flipped R7 tests, the OQ-3 test and the weekend-valuation test). (The OQ-4 status-quo pin that an earlier
revision put here is not written: the user answered OQ-4 (b), implemented by Task 8c.)

- [ ] **Step 3: Implement**

`target_redemption_state.py` — the field and its check:

```python
@dataclass(frozen=True)
class TargetRedemptionState:
    """... ``as_of`` (dated form only): the date the state was brought up to date, never after the valuation date.
    ``fixings_banked`` (dated form only): how many leading scheduled fixings are banked (fixing ``i`` is banked iff
    ``i < fixings_banked``); ``None`` on a booking means "the fixings dated on or before ``as_of``"."""

    accumulated: float
    pending: Tuple[PendingPayment, ...] = ()
    as_of: Optional[datetime] = None
    fixings_banked: Optional[int] = None

    def __post_init__(self) -> None:
        ...  # existing checks unchanged
        n = self.fixings_banked
        if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n < 0):
            raise ValidationError(f"state.fixings_banked must be None or an int >= 0, got {n!r}")
```

`target_redemption_base.py` (add `timedelta` to the `datetime` import):

```python
    def _banked_count(self) -> int:
        """Dated: how many leading scheduled fixings are banked (OQ-3 ruling, per fixing): ``state.fixings_banked``,
        else - a booking that states only ``as_of`` - the fixings dated on or before ``as_of``, else 0."""
        st = self.state
        if st.fixings_banked is not None:
            return st.fixings_banked
        if st.as_of is None:
            return 0
        return sum(1 for d in self.fixing_dates if d <= st.as_of)

    @staticmethod
    def _state_date(fx_env) -> datetime:
        """The ``as_of`` banking writes (Ruling R24): the valuation date, moved back over the days at zero time from it
        (valued on a Saturday under BUSINESS_DAYS, the Friday: the same instant), so it never passes the valuation date
        and cash due at that instant never precedes it. Under CALENDAR_DAYS / ACT conventions: the valuation date."""
        val = fx_env.valuation_date
        d = val
        for _ in range(31):  # bounded: no zero-time stretch (weekend + holidays) is a month long
            prev = d - timedelta(days=1)
            if not is_zero(calculate_year_fraction(prev, val, fx_env.day_count_convention, fx_env.bus_days_in_year,
                                                   calendar=fx_env.calendar)):
                break
            d = prev
        return d

    def _is_unbanked(self, index: int) -> bool:
        """Dated: fixing ``index`` is banked iff it is one of the first ``_banked_count()`` (identity, not date)."""
        return index >= self._banked_count()

    @property
    def is_terminal(self) -> bool:
        """Redeemed, or no fixing left: numeric - an empty list; dated - every scheduled fixing banked."""
        if self.is_numeric:
            return not self.fixing_times or self.is_redeemed
        return self.is_redeemed or self._banked_count() >= len(self.fixing_dates)

    # _validate_state: numeric branch, next to the as_of form check
            if st.fixings_banked is not None:
                raise ValidationError(
                    "state.fixings_banked belongs to the dated form: a numeric state lists only unbanked fixings")
    # _validate_state: dated branch (replaces the as_of-before-the-first-fixing rule; same refusal for every
    # as_of-only booking)
        if st.fixings_banked is not None and st.fixings_banked > len(self.fixing_dates):
            raise ValidationError(
                f"state.fixings_banked {st.fixings_banked} exceeds the {len(self.fixing_dates)} scheduled fixings")
        if (st.accumulated > 0.0 or st.pending) and (st.as_of is None or self._banked_count() == 0):
            raise ValidationError(
                "accumulated or pending cash needs state.as_of and at least one banked fixing")

    def _has_unbanked_passed(self, fx_env) -> bool:
        if self.is_numeric:
            return bool(self.fixing_times) and is_passed_time(self.fixing_times[0])
        self._check_as_of(fx_env)
        if self.is_terminal:
            return False
        first = self._banked_count()
        return first < len(self.fixing_dates) and is_passed_time(
            signed_year_fraction(self.fixing_dates[first], fx_env))

    def _record_dated(self, fx_env) -> List[Tuple[int, str]]:
        self._check_as_of(fx_env)
        filled: List[Tuple[int, str]] = []
        val, st = fx_env.valuation_date, self.state
        events: List[TargetRedemptionEvent] = []
        pending: list = []
        for p in st.pending:  # cash whose pay time is strictly past was paid; cash due today stays (Task 8a)
            t_pay = signed_year_fraction(p.pay_date, fx_env)
            if is_history(t_pay):
                events.append(TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, p.amount, time=t_pay))
            else:
                pending.append(p)
        acc = st.accumulated
        banked = self._banked_count()
        if not self.is_terminal:
            last = len(self.fixing_dates) - 1
            for i in range(banked, len(self.fixing_dates)):
                fd = self.fixing_dates[i]
                t = signed_year_fraction(fd, fx_env)
                if not is_passed_time(t):  # ahead by TIME (OQ-2 b): a Saturday at zero business time is passed
                    break
                label = event_label(i, date=fd)
                spot = self._rate(i, fx_env, filled, label)
                added, cash = self._fixing_outcome(self._dated_factor(i, fx_env), spot, acc)
                acc = acc + added
                hit = is_target_hit(acc, self.target)
                ends = hit or i == last
                events.append(TargetRedemptionEvent(TargetRedemptionEventType.FIXING, cash, fixing=spot, time=t,
                                                    redeemed=ends, label=label))
                self._settle_dated(cash, self.pay_dates[i], label, fx_env, pending, events)
                banked = i + 1  # identity: fixings 0 .. i are banked, whatever their dates (OQ-3 ruling)
                if ends:
                    self._settle_dated(self._principal_amount(), self.pay_dates[i], label, fx_env, pending, events)
                    break  # every later fixing is dead (or there is none)
        self.state = TargetRedemptionState(acc, pending=tuple(sorted(pending, key=lambda p: p.pay_date)),
                                           as_of=self._state_date(fx_env), fixings_banked=banked)
        self.lifecycle_events = self.lifecycle_events + tuple(events)
        # a banked or dead fixing's recorded rate is spent
        self._recorded_rates = {i: r for i, r in self._recorded_rates.items()
                                if not self.is_terminal and self._is_unbanked(i)}
        return filled
```

In `_dated_view`: iterate `for i in range(self._banked_count(), len(self.fixing_dates))` (when not terminal), remove the
fixing refusal (`if is_passed_time(t): raise ValidationError(... zero year fraction ...)`) and the pending refusal Task
8a kept (`if p.pay_date > val and is_passed_time(t): raise ...`); the pending loop becomes
`t = signed_year_fraction(p.pay_date, fx_env); if is_history(t): continue; pending.append((snap_time(t), p.amount))`.
Any other reader of the banked set already goes through `_is_unbanked` / `is_terminal` (`_is_listed_unbanked`,
`with_fixings_recorded`, the recorded-rate cleanup).

CHANGELOG — in the `### Added` bullet "Dated `FxTargetRedemptionNote` / `FxTargetRedemptionForward`: ...", replace "the
TARN derives its periods" sentence's neighbour "A fixing or pending pay date after the valuation date that resolves to
a zero year fraction in the environment's day count (a weekend date under `BUSINESS_DAYS`) is refused with
`ValidationError`." with:

```markdown
A fixing or pay date that resolves to zero time in the environment's day count (a weekend date under `BUSINESS_DAYS`,
valued on the Friday) is live, as for its numeric twin (user decision OQ-2 b, 2026-10-10): the fixing is decided at
the quoted spot and its cash is due that day. A dated state counts its banked fixings
(`TargetRedemptionState.fixings_banked`, fixing identity in schedule order; a booking stating only `as_of` banks the
fixings dated on or before it); `as_of` never passes the valuation date (user ruling OQ-3, 2026-10-10).
```

Append to `moved_values.txt`:

```text
#
# Task 8b - OQ-2 (b) and the OQ-3 ruling, user 2026-10-10: a dated fixing / pay date at zero business time is live;
# banked fixings are tracked per fixing (TargetRedemptionState.fixings_banked), as_of never passes the valuation date.
# test_future_fixing_with_zero_year_fraction_is_refused: ValidationError -> priced (equal to its numeric twin, rel 1e-12).
# test_pending_pay_date_with_zero_year_fraction_is_refused: ValidationError -> pending at t = 0 (in the value), paid
# on the next business day. A Friday + Saturday fixing pair on BUSINESS_DAYS prices on the Friday (twins equal, events
# equal) and is banked once. Ruling R7 superseded.
# Not a value: a dated state after ageing now states fixings_banked (Task 8a's dated record test expects
# fixings_banked=1; no price, Greek or event moves). Every booking stating only as_of keeps its banked set.
# Not a value: banking on a non-business valuation date under BUSINESS_DAYS writes as_of = the first date of its
# zero-time stretch (Ruling R24: a Saturday row writes the Friday); under CALENDAR_DAYS / ACT as_of is the valuation
# date as before.
```

- [ ] **Step 4: Run to verify they pass**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_state.py test/test_fx_target_redemption_lifecycle.py test/test_fx_tarn_note_mc.py test/test_fx_tarf_mc.py test/test_fx_base_engine.py -n 2 -q`
Expected: all pass, and no other existing expectation moves (D1–D5, WARN1 dated, WARN3 dated, P-tests,
`test_r2_as_of_after_valuation_is_refused_at_pricing` and the R2 "accumulated with as_of before the first fixing"
refusal keep their outcomes: every booking there states only `as_of`, so its banked set is unchanged). Any other
failure: STOP and report it as an unlisted mover.

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/asset/fx/product/option/target_redemption_state.py quantark/asset/fx/product/option/target_redemption_base.py test/test_fx_target_redemption_state.py test/test_fx_target_redemption_lifecycle.py CHANGELOG.md
git -C <WT> add -f docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/moved_values.txt
git -C <WT> commit -m "feat(fx-tarn): dated fixings banked per fixing; a fixing or pay date at zero business time is live (OQ-2 b, OQ-3)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8c: Periods may share a fixing date or time (OQ-4 (b), user decision 2026-10-10)

Follow-up that re-opens built Tasks 3, 4, 6 and 7. User decision OQ-4 (b), generalised: "two period dates can share
the same fixing date". Equal fixing DATES (dated form) and equal fixing TIMES (numeric form, and dated dates that
resolve to one time, e.g. Friday + Saturday under `BUSINESS_DAYS`) are a legitimate schedule. This amends TARN spec
§4.1 / §4.2 ("ascending, unique" → ascending, i.e. non-decreasing).

**Rules (all from the user decision or the existing spec; nothing invented):**
- Both MC engines simulate each DISTINCT fixing time once; each period's fixing takes the spot at its time, so periods
  sharing a time share one spot.
- Target accumulation and redemption go through the periods in SCHEDULE order, deterministically. A period after the
  one that hits the target — on the same date or later — is dead, by the spec's existing rule: TARN spec §6
  (`record_passed_fixings`, TARN bullet) "On a hit (`is_redeemed`), the principal (with `include_principal`) at `pay_i`
  is handled the same way and **every later fixing is dead**", applied "for each unbanked fixing at or before
  valuation, **in order**, stopping once the contract is terminal"; TARF bullet: "the hit rule as for the TARN. On a
  final hit the geared unfavourable leg is still paid" (the hitting period's own leg); spec §5.2 / engines: the `alive`
  mask stops every later period's cash. The principal goes with the hitting period's `pay_i`.
- Each period keeps its own pay date / pay time and its own cash (its own accrual factor for the TARN).
- Per-fixing banking (OQ-3, Task 8b) banks every period sharing the date together, each once, by identity.
- **Default-derivation refusal (user ruling 2026-10-10 on R26, option (b); R26 itself is superseded):** a dated TARN
  on the DEFAULT derivation (`initial_date`, then each previous fixing date → this fixing date) cannot share a fixing
  date — the later period would run from a fixing date to the same date, a zero-length derived period — and is refused
  with `ValidationError` ("... would have zero length ..."). A contract that states its own accrual is NOT refused:
  explicit `accrual_factors` here, and from Task 8d its own period dates (`accrual_end_dates`). Task 8d narrows the
  condition to the default derivation and names both cures in the message; the test below matches "zero length", which
  both wordings carry, so Task 8d does not rewrite it. The numeric form already refuses the same case through its built
  "accrual_factors must be positive" check (factors `t_i − t_{i−1}` from `accrual_start`). Distinct dates that resolve
  to one time (Friday + Saturday) keep their positive derived factors (ACT/365 by default).

**Files:**
- Modify: `quantark/asset/fx/product/option/target_redemption_base.py` (`_validate_numeric_schedule`,
  `_validate_dated_schedule`: the uniqueness checks go), `quantark/asset/fx/product/option/fx_target_redemption_note.py`
  (`_validate_dated_accrual`: the default-derivation refusal; docstring),
  `quantark/asset/fx/product/option/fx_target_redemption_forward.py`
  (docstring), `quantark/asset/fx/engine/mc/fx_tarn_note_mc_engine.py` and `fx_tarf_mc_engine.py` (`_simulate`),
  `CHANGELOG.md`, `docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/moved_values.txt`
- Test: `test/test_fx_target_redemption_lifecycle.py` (new tests; Task 8b's OQ-4 pin is not written — see the note in
  Task 8b)

**Interfaces:**
- Consumes: Tasks 8a, 8b.
- Produces: `fixing_times` / `fixing_dates` non-decreasing (equal values allowed); both engines'
  `_simulate(fx_env, spot, sigma, fixings) -> np.ndarray (P, M)` with one column per PERIOD, columns of periods that
  share a time identical (one simulated spot); for a schedule of distinct times the generator input and the returned
  array are exactly today's (bitwise: no reindexing).

- [ ] **Step 1: Write the failing tests**

Append to `test/test_fx_target_redemption_lifecycle.py` (and add `import numpy as np` to its imports):

```python
# --- OQ-4 (b), user decision 2026-10-10: periods may share a fixing date / time ------------------------------------

SHARED_FACTORS = [1 / 12, 1 / 24] + [1 / 12] * 11


def _shared_dated(kind, target, **kw):
    """Thirteen periods: the first two share FIX_DATES[0]; the second pays two days later."""
    dates = [FIX_DATES[0]] + FIX_DATES
    pays = [FIX_DATES[0], LAG2[0]] + FIX_DATES[1:]
    if kind == "tarf":
        return dated_tarf(fixing_dates=dates, pay_dates=pays, target=target, **kw)
    return dated_note(fixing_dates=dates, pay_dates=pays, accrual_factors=SHARED_FACTORS, target=target, **kw)


def test_numeric_form_accepts_equal_fixing_times():
    n = note(fix=[FIX[0], FIX[0]] + FIX[1:], pay_times=[FIX[0], FIX[0] + 2 / 365] + FIX[1:],
             accrual_factors=SHARED_FACTORS, target=50.0)
    assert n.fixing_times[:2] == [FIX[0], FIX[0]]
    with pytest.raises(ValidationError, match="ascending"):
        note(fix=[FIX[1], FIX[0]] + FIX[2:], target=50.0)              # decreasing is still refused


def test_the_default_derivation_cannot_share_a_fixing_date():
    """User ruling 2026-10-10 on R26 (b): a zero-length DERIVED period is refused; a stated accrual is not."""
    with pytest.raises(ValidationError, match="zero length"):
        dated_note(fixing_dates=[FIX_DATES[0]] + FIX_DATES, target=50.0)  # initial_date=VAL: a zero-length period
    assert _shared_dated("tarn", target=50.0).fixing_dates[:2] == [FIX_DATES[0]] * 2  # explicit factors: accepted


def test_engines_simulate_a_shared_time_once():
    e = env()
    eng = engine_for(note(target=50.0), n=1_000)
    t0, t1 = FIX[0], FIX[1]
    shared = eng._simulate(e, 1.20, 0.10, np.array([t0, t0, t1]))
    distinct = eng._simulate(e, 1.20, 0.10, np.array([t0, t1]))
    assert np.array_equal(shared[:, 0], shared[:, 1]) and np.array_equal(shared[:, [0, 2]], distinct)
    tarf_eng = engine_for(tarf(target=0.2), n=1_000)
    assert np.array_equal(tarf_eng._simulate(e, 1.20, 0.10, np.array([t0, t0, t1]))[:, [0, 2]],
                          tarf_eng._simulate(e, 1.20, 0.10, np.array([t0, t1])))


@pytest.mark.parametrize("kind", ["tarn", "tarf"])
def test_two_periods_on_one_fixing_date_price_like_the_twin_and_each_is_paid(kind):
    dated = _shared_dated(kind, target=50.0)
    twin = _numeric_twin(dated, env())
    assert twin.fixing_times[0] == twin.fixing_times[1]
    assert is_close(_price(twin), _price(dated), rel_tol=1e-12)
    e0, e3 = env(val=FIX_DATES[0]), env(val=FIX_DATES[0] + timedelta(days=3))
    twin.time_shift(calculate_year_fraction(VAL, FIX_DATES[0], DayCountConvention.CALENDAR_DAYS, 252))
    for product in (dated, twin):
        product.record_passed_fixings(e0)                              # the shared date: both periods bank together
    fixings = [x for x in dated.lifecycle_events if x.event_type is FIXING]
    assert len(fixings) == 2 and _kinds_only(dated.lifecycle_events) == _kinds_only(twin.lifecycle_events)
    assert fixings[0].fixing == fixings[1].fixing == 1.20                # one spot for both
    assert dated.state.fixings_banked == 2
    assert [p.pay_date for p in dated.state.pending] == [FIX_DATES[0], LAG2[0]]      # each its own pay date
    assert [p.amount for p in dated.state.pending] == [x.amount for x in fixings]    # each its own cash
    twin.time_shift(3 / 365)
    for product, e in ((dated, e3), (twin, e3)):
        product.record_passed_fixings(e)
    assert _kinds_only(dated.lifecycle_events) == _kinds_only(twin.lifecycle_events)
    assert [x.event_type for x in dated.lifecycle_events] == [FIXING, FIXING, PAYMENT, PAYMENT]   # banked once
    assert dated.state.fixings_banked == 2 and dated.state.pending == ()


def test_a_target_hit_by_the_first_of_two_same_date_periods_kills_the_second():
    """TARN spec section 6: "On a hit ... every later fixing is dead" (periods in schedule order)."""
    dated = _shared_dated("tarn", target=0.9)                           # the first coupon (1.0) is clipped to 0.9
    twin = _numeric_twin(dated, env())
    dated.record_passed_fixings(env(val=FIX_DATES[0]))
    (fixing,) = [x for x in dated.lifecycle_events if x.event_type is FIXING]  # the second period never banks
    assert fixing.amount == 0.9 and fixing.redeemed and dated.is_redeemed and dated.is_terminal
    assert [(p.amount, p.pay_date) for p in dated.state.pending] == [(0.9, FIX_DATES[0]), (100.0, FIX_DATES[0])]
    e = env(vol=1e-7)                                                   # the engine walk applies the same rule
    t = calculate_year_fraction(VAL, FIX_DATES[0], DayCountConvention.CALENDAR_DAYS, 252)
    expected = (0.9 + 100.0) * e.get_domestic_df(t)                     # nothing from the second period
    assert is_close(_price(twin, e=e), expected, rel_tol=1e-6)
    assert is_close(_price(_shared_dated("tarn", target=0.9), e=e), expected, rel_tol=1e-6)


def test_friday_and_saturday_fixings_price_on_the_thursday_like_the_twin():
    """OQ-3 case valued before the shared instant (was refused: OQ-4 status quo): both fixings at one business time,
    one simulated spot; dated == numeric."""
    dated = dated_note(fixing_dates=[FRIDAY, SATURDAY, datetime(2026, 9, 15)], initial_date=datetime(2026, 7, 15),
                       target=50.0)
    e_thu = env(val=THURSDAY, **BUS)
    twin = _numeric_twin(dated, e_thu)
    assert twin.fixing_times[0] == twin.fixing_times[1]
    assert is_close(_price(twin, e=e_thu), _price(dated, e=e_thu), rel_tol=1e-12)
```

(`FRIDAY`, `SATURDAY`, `THURSDAY`, `BUS`, `_kinds_only` come from Task 8b's and Task 7's sections of the same module.)

- [ ] **Step 2: Run to verify they fail**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_lifecycle.py -k "equal_fixing_times or default_derivation_cannot or shared_time or one_fixing_date or same_date_periods or on_the_thursday" -n 2 -q`
Expected: FAIL — `ValidationError: fixing_times must be unique` / `fixing_dates must be unique` (construction and the
twins, and `_shared_dated` in the default-derivation test), `ValidationError: times must be strictly increasing and
positive` (the engine `_simulate` test); the default-derivation test's first call already raises (uniqueness) but on
the wrong message ("zero length" not matched).

- [ ] **Step 3: Implement**

`target_redemption_base.py`: delete the two uniqueness checks
(`if len(set(dates)) != len(dates): raise ValidationError("fixing_dates must be unique")` in `_validate_dated_schedule`
and `if len(set(fixings)) != len(fixings): raise ValidationError("fixing_times must be unique")` in
`_validate_numeric_schedule`); the "must be ascending" checks stay and now mean non-decreasing (`list(x) != sorted(x)`).

`fx_target_redemption_note.py`, `_validate_dated_accrual`, after the `initial_date` check:

```python
        if self.initial_date is not None and len(set(self.fixing_dates)) != len(self.fixing_dates):
            # user ruling 2026-10-10 on R26 (b): the DEFAULT derivation's later period on a shared fixing date runs
            # from that date to itself; a contract stating its own accrual is not refused (Task 8d narrows this
            # condition to the default derivation and adds accrual_end_dates to the message)
            raise ValidationError(
                "periods sharing a fixing date need their own accrual: under the default derivation (initial_date, "
                "then each previous fixing date) the later period would have zero length - state accrual_factors")
```
Docstrings (`fx_target_redemption_note.py:59`, `fx_target_redemption_forward.py:42`): "Ascending (non-decreasing:
periods may share a fixing time), finite year-fraction fixing times."

Both engines (`fx_tarn_note_mc_engine.py`, `fx_tarf_mc_engine.py`), `_simulate`:

```python
    def _simulate(self, fx_env, spot, sigma, fixings) -> np.ndarray:
        """Spots at each PERIOD's fixing, (P, M). Each distinct fixing time is simulated once; periods sharing a time
        share its spot (user decision OQ-4 b). A schedule of distinct times is simulated exactly as before."""
        times, period_of = np.unique(fixings, return_inverse=True)
        gen = self._make_generator(fx_env, spot, sigma, times)
        paths, _ = gen.generate_paths()
        spots = paths[:, 1:]  # drop S_0 column -> one column per distinct time
        return spots if times.size == fixings.size else spots[:, period_of]
```

The walks (`_accumulate`) need no change: they already go through the periods in schedule order with the `alive` mask
(a period after the hit is dead, spec §6 / §5.2) and each period's own `df_pay[i]`. `_record_numeric` /
`_record_dated` need no change: they bank in schedule order by time and stop on a hit (Task 8b's dated loop banks every
period whose time has passed, so periods sharing a date bank together and are counted once each).

Append to the CHANGELOG `### Changed`:

```markdown
- FX TARN / TARF periods may share a fixing date or time (user decision OQ-4 b, 2026-10-10): `fixing_dates` /
  `fixing_times` must be non-decreasing instead of strictly increasing. Both MC engines simulate each distinct fixing
  time once and every period sharing it fixes at that spot; periods are accumulated in schedule order (a period after
  the target hit is dead), each with its own pay date and cash. A dated TARN on the default period derivation
  (`initial_date`, then each previous fixing date) cannot share a fixing date (a zero-length period): it states its own
  accrual.
```

Append to `moved_values.txt`:

```text
#
# Task 8c - OQ-4 (b), user decision 2026-10-10: periods may share a fixing date / time.
# Contracts with equal fixing times (numeric) or equal fixing dates (dated, explicit accrual_factors / TARF), and dated
# contracts whose distinct dates resolve to one time (Friday + Saturday under BUSINESS_DAYS valued before that
# instant): ValidationError (unique / strictly increasing) -> priced, equal to the twin. Nothing else moves: a schedule
# of distinct times takes exactly the old engine path (no reindexing).
# Same refusal, new reason: test_r2_dated_refusals[fixing_dates=FIX_DATES[:1] * 2] (initial_date-derived) still raises
# ValidationError, now the default derivation's zero-length refusal (user ruling 2026-10-10 on R26 (b)) instead of
# "fixing_dates must be unique".
```

- [ ] **Step 4: Run to verify they pass**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_state.py test/test_fx_target_redemption_lifecycle.py test/test_fx_tarn_note_mc.py test/test_fx_tarf_mc.py test/test_fx_base_engine.py -n 2 -q`
Expected: all pass, unmodified apart from this task's additions (every existing schedule has distinct times, so every
existing price, Greek and event is bitwise unchanged). Any other failure: STOP and report it as an unlisted mover.

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/asset/fx/product/option/target_redemption_base.py quantark/asset/fx/product/option/fx_target_redemption_note.py quantark/asset/fx/product/option/fx_target_redemption_forward.py quantark/asset/fx/engine/mc/fx_tarn_note_mc_engine.py quantark/asset/fx/engine/mc/fx_tarf_mc_engine.py test/test_fx_target_redemption_lifecycle.py CHANGELOG.md
git -C <WT> add -f docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/moved_values.txt
git -C <WT> commit -m "feat(fx-tarn): periods may share a fixing date or time; one simulated spot per distinct time (OQ-4 b)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8d: The TARN's own accrual period dates, separate from its fixing dates (user ruling on R26: (b), 2026-10-10)

Follow-up that re-opens built Tasks 3 and 7 and narrows Task 8c's refusal. User ruling (b), replacing R26: the TARN
gets its own accrual PERIOD dates, separate from its fixing dates, as the accrual-time spec
(`docs/superpowers/specs/2026-10-08-accrual-time-design.md`) separated accrual from observation. This amends TARN spec
§4.2 (a new dated-form row; the "Accrual factors" bullet's end becomes the period's accrual end) and Q2 (a dated TARN
has three period sources, exactly one stated). The TARF has no accrual term (TARN spec §2.3) and gets no new field.

**Rules (user ruling; Rulings R27, R28; nothing invented — anything undefined is refused and listed as OQ-5 / OQ-6):**
- New keyword `accrual_end_dates: Optional[List[datetime]]`, dated TARN only, 1:1 with `fixing_dates`. Period `i` runs
  from `E_{i−1}` to `E_i`, with `E_{−1} = initial_date` and `E_i = accrual_end_dates[i]` — the accrual-time spec's
  dated rule (§4.3 Endpoints: "`start` is the previous record's `end`, or `initial_date` for the first record") with
  the end freed from the fixing date. The fixing date only fixes the spot; the pay date only pays.
- Default `accrual_end_dates=None`: `E_i = fixing_dates[i]`, exactly today's derivation (`initial_date` / previous
  fixing date → this fixing date): the same `calculate_year_fraction` call on the same date objects, so every existing
  booking keeps its factors and value bitwise.
- Factor: `calculate_year_fraction(E_{i−1}, E_i, day_count_convention or ACT_365, fx_env.bus_days_in_year,
  calendar=fx_env.calendar)` — TARN spec §4.2's call unchanged (a `BUSINESS_DAYS` count reads the environment's
  calendar, as today).
- Precedence / exclusivity — exactly one period source: (1) `accrual_factors` (excludes `initial_date`,
  `accrual_end_dates`, `day_count_convention`); (2) `initial_date` + `accrual_end_dates`; (3) `initial_date` alone (the
  default). `accrual_end_dates` without `initial_date` is refused; with `accrual_factors` refused; on the numeric form
  refused. No layering: the stated source gives every factor.
- Validation (R28): aligned 1:1 and datetimes; strictly increasing (an equal end is a zero-length period, refused like
  a non-positive factor in both forms today); `fixing_dates[i] <= accrual_end_dates[i] <= pay_dates[i]` (the
  accrual-time spec's two ends are the observation and the settlement date; an end outside that window is undefined:
  OQ-5). `initial_date < fixing_dates[0]` stays (so `initial_date < E_0`). No check is dropped.
- Shared fixing date: with `accrual_end_dates` (or `accrual_factors`) periods may share a fixing date; under the
  DEFAULT derivation a shared fixing date is still refused (zero-length derived period). This task narrows Task 8c's
  refusal to the default derivation and names both cures in its message.
- Numeric form and twin: nothing new. `_numeric_twin` builds the numeric TARN from `dated.resolve(env).accrual_factors`
  (per-period factors from `_dated_factor`) and per-period pay times, so the twin carries each period's own factor.

**Files:**
- Modify: `quantark/asset/fx/product/option/fx_target_redemption_note.py` (`__init__` keyword and source checks,
  `_validate_dated_accrual`, new `_validate_accrual_end_dates` and `_accrual_end`, `_dated_factor`, module and class
  docstrings), `CHANGELOG.md`, `docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/moved_values.txt`
- Test: `test/test_fx_target_redemption_lifecycle.py` (new tests; every existing test, including D5, R2 and Task 8c's,
  unmodified)

**Interfaces:**
- Consumes: Task 7 (`_dated_factor`, `_validate_dated_accrual`, `initial_date` / `accrual_factors` /
  `day_count_convention`), Task 8b (`fixings_banked`), Task 8c (shared fixing dates; the default-derivation refusal).
- Produces: `FxTargetRedemptionNote(..., *, accrual_end_dates: Optional[List[datetime]] = None)` (keyword-only, after
  `day_count_convention`); attribute `accrual_end_dates: Optional[List[datetime]]` (`None` on the numeric form and on
  the default); `_accrual_end(index) -> datetime`; `_dated_factor(index, fx_env) -> float` reading period ends.
  `ResolvedTargetRedemption`, the engines and the numeric form are unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_fx_target_redemption_lifecycle.py`:

```python
# --- R26 (b), user ruling 2026-10-10: the TARN's own accrual period dates (accrual_end_dates) ---------------------

def _by_period_dates(**kw):
    """Twelve periods accruing VAL -> FIX_DATES[0] -> ... -> FIX_DATES[11] (the default contract's periods); periods 0
    and 1 both FIX on FIX_DATES[0], period k >= 1 fixes on FIX_DATES[k - 1], and each pays at its accrual end."""
    return dated_note(fixing_dates=[FIX_DATES[0]] + FIX_DATES[:-1], accrual_end_dates=list(FIX_DATES),
                      pay_dates=list(FIX_DATES), target=kw.pop("target", 50.0), **kw)


@pytest.mark.parametrize("calendar", ["calendar", "business"])
def test_accrual_end_dates_on_the_fixing_dates_reproduce_the_default_bitwise(calendar):
    if calendar == "calendar":
        dates, e, kw, convention = FIX_DATES, env(), {}, DayCountConvention.ACT_365
    else:
        convention = DayCountConvention.BUSINESS_DAYS
        dates, e, kw = BUS_DATES, env(**BUS), dict(day_count_convention=convention)
    default = dated_note(fixing_dates=dates, **kw)
    stated = dated_note(fixing_dates=dates, accrual_end_dates=list(dates), **kw)
    assert default.accrual_end_dates is None
    today = tuple(calculate_year_fraction(s, d, convention, e.bus_days_in_year, calendar=e.calendar)
                  for s, d in zip([VAL] + dates[:-1], dates))
    assert default.resolve(e).accrual_factors == today       # the default is today's derivation, call for call
    assert stated.resolve(e).accrual_factors == today
    assert _price(stated, e=e) == _price(default, e=e)


def test_two_periods_sharing_a_fixing_date_accrue_their_own_period_dates_like_the_twin():
    dated = _by_period_dates()
    assert dated.fixing_dates[0] == dated.fixing_dates[1] and dated.accrual_factors is None   # no factors stated
    view = dated.resolve(env())
    own = tuple(calculate_year_fraction(s, d, DayCountConvention.ACT_365, 252)
                for s, d in zip([VAL] + FIX_DATES[:-1], FIX_DATES))
    assert view.accrual_factors == own                        # each period over its own dates, not its fixing interval
    assert view.accrual_factors[0] != view.accrual_factors[1]  # 30 and 31 days on one fixing date
    twin = _numeric_twin(dated, env())
    assert twin.fixing_times[0] == twin.fixing_times[1]
    assert twin.accrual_factors == list(own)                  # the twin carries the same per-period factors
    assert twin.pay_times == list(view.pay_times)             # and the same per-period pay times
    assert is_close(_price(twin), _price(dated), rel_tol=1e-12)
    dated.record_passed_fixings(env(val=FIX_DATES[0]))        # both periods fix on FIX_DATES[0], at one spot
    fixings = [x for x in dated.lifecycle_events if x.event_type is FIXING]
    assert len(fixings) == 2 and fixings[0].fixing == fixings[1].fixing == 1.20
    assert [x.amount for x in fixings] == [dated.coupon_amount(f) for f in own[:2]]   # each its own factor's coupon
    assert dated.state.fixings_banked == 2
    assert [(p.amount, p.pay_date) for p in dated.state.pending] == [
        (fixings[0].amount, FIX_DATES[0]), (fixings[1].amount, FIX_DATES[1])]          # each its own pay date


def test_a_shared_fixing_date_needs_period_dates_or_factors():
    shared, pays = [FIX_DATES[0]] + FIX_DATES[:-1], list(FIX_DATES)
    with pytest.raises(ValidationError, match="zero length - state accrual_end_dates or accrual_factors"):
        dated_note(fixing_dates=shared, pay_dates=pays)       # default derivation: period 1 runs FIX_DATES[0] -> itself
    assert _by_period_dates().accrual_end_dates == FIX_DATES  # its own period dates: accepted
    assert dated_note(fixing_dates=shared, pay_dates=pays, accrual_factors=[1 / 12] * 12).accrual_end_dates is None


_SHARED = [FIX_DATES[0]] + FIX_DATES[:-1]


@pytest.mark.parametrize("kw, match", [
    (dict(accrual_end_dates=FIX_DATES[:-1]), "align 1:1"),
    (dict(accrual_end_dates=[d.date() for d in FIX_DATES]), "datetimes"),
    (dict(fixing_dates=_SHARED, pay_dates=list(FIX_DATES), accrual_end_dates=list(_SHARED)), "strictly increasing"),
    (dict(pay_dates=LAG2, accrual_end_dates=[FIX_DATES[1], FIX_DATES[0]] + FIX_DATES[2:]), "strictly increasing"),
    (dict(accrual_end_dates=[FIX_DATES[0] - timedelta(days=1)] + FIX_DATES[1:]), "between its fixing date"),
    (dict(pay_dates=LAG2, accrual_end_dates=[LAG2[0] + timedelta(days=1)] + FIX_DATES[1:]), "between its fixing date"),
    (dict(accrual_end_dates=list(FIX_DATES), accrual_factors=[1 / 12] * 12), "cannot be combined"),
    (dict(accrual_end_dates=list(FIX_DATES), initial_date=None), "needs initial_date"),
], ids=["unaligned", "not_datetimes", "equal_end", "decreasing", "end_before_fixing", "end_after_pay",
        "with_accrual_factors", "without_initial_date"])
def test_accrual_end_dates_refusals(kw, match):
    with pytest.raises(ValidationError, match=match):
        dated_note(**kw)


def test_accrual_end_dates_belong_to_the_dated_tarn_only():
    with pytest.raises(ValidationError, match="belongs to the dated form"):
        note(accrual_end_dates=list(FIX_DATES))
    with pytest.raises(TypeError):                            # the TARF has no accrual term (TARN spec 2.3)
        dated_tarf(accrual_end_dates=list(FIX_DATES))
```

- [ ] **Step 2: Run to verify they fail**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_lifecycle.py -k "accrual_end_dates or own_period_dates or needs_period_dates" -n 2 -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'accrual_end_dates'` (every test that
passes the keyword to the TARN, including the numeric refusal, which expects `ValidationError`); the shared-date test's
first call raises `ValidationError` but on Task 8c's wording (`... zero length - state accrual_factors`, the pattern
not matched). The TARF `TypeError` assertion passes before and after (a guard, not a change).

- [ ] **Step 3: Implement**

`fx_target_redemption_note.py`, `__init__`: add the keyword after `day_count_convention`:

```python
        day_count_convention: Optional[DayCountConvention] = None,
        accrual_end_dates: Optional[List[datetime]] = None,
    ):
```

In the dated branch, after the numeric-keyword loop and BEFORE the existing "derives its periods from initial_date or
states explicit accrual_factors" check:

```python
            if accrual_end_dates is not None and accrual_factors is not None:
                raise ValidationError(
                    "accrual_end_dates sets the derived accrual factors; it cannot be combined with explicit "
                    "accrual_factors")
            if accrual_end_dates is not None and initial_date is None:
                raise ValidationError(
                    "accrual_end_dates needs initial_date: the first period accrues from initial_date to "
                    "accrual_end_dates[0]")
```
and, beside `self.day_count_convention = day_count_convention`:
```python
            self.accrual_end_dates = list(accrual_end_dates) if accrual_end_dates is not None else None
```
In the numeric branch, add `("accrual_end_dates", accrual_end_dates)` to the refused-keyword tuple (message: "...
belongs to the dated form; a numeric TARN takes pay_times and accrual_start") and set `self.accrual_end_dates = None`
beside `self.day_count_convention = None`.

`_validate_dated_accrual`: replace Task 8c's block (the `len(set(self.fixing_dates))` refusal) with:

```python
        if self.accrual_end_dates is not None:
            self._validate_accrual_end_dates()
        elif self.initial_date is not None and len(set(self.fixing_dates)) != len(self.fixing_dates):
            # user ruling 2026-10-10 on R26 (b): only the DEFAULT derivation's later period on a shared fixing date
            # runs from that date to itself
            raise ValidationError(
                "periods sharing a fixing date need their own accrual: under the default derivation (initial_date, "
                "then each previous fixing date) the later period would have zero length - state accrual_end_dates "
                "or accrual_factors")
```

New methods, beside `_dated_factor`, and `_dated_factor` reading period ends:

```python
    def _validate_accrual_end_dates(self) -> None:
        """One accrual end per period; period ``i`` runs from the previous end (``initial_date`` for the first). An
        end lies between its fixing date (the observation end) and its pay date (the settlement end)."""
        ends, dates, pays = self.accrual_end_dates, self.fixing_dates, self.pay_dates
        if len(ends) != len(dates):
            raise ValidationError("accrual_end_dates must align 1:1 with fixing_dates")
        if any(not isinstance(e, datetime) for e in ends):
            raise ValidationError("accrual_end_dates must be datetimes")
        if any(not prev < e for prev, e in zip(ends, ends[1:])):
            raise ValidationError(
                "accrual_end_dates must be strictly increasing: each period accrues from the previous period's end, "
                "so an equal end is a zero-length period")
        for i, (f, e, p) in enumerate(zip(dates, ends, pays)):
            if not f <= e <= p:
                raise ValidationError(
                    f"accrual end date {e.date()} of period {i} must lie between its fixing date {f.date()} and its "
                    f"pay date {p.date()}")

    def _accrual_end(self, index: int) -> datetime:
        """Period ``index``'s accrual end: ``accrual_end_dates[index]``, else its fixing date (the default)."""
        return self.fixing_dates[index] if self.accrual_end_dates is None else self.accrual_end_dates[index]

    def _dated_factor(self, index: int, fx_env) -> float:
        """Explicit factor, else ``initial_date`` / the previous period's accrual end -> this period's accrual end
        (``accrual_end_dates``; default: the fixing dates) in the contract day count (``None`` = ACT_365), resolved
        against the environment's calendar (a BUSINESS_DAYS count needs it)."""
        if self.accrual_factors is not None:
            return self.accrual_factors[index]
        start = self.initial_date if index == 0 else self._accrual_end(index - 1)
        convention = DayCountConvention.ACT_365 if self.day_count_convention is None else self.day_count_convention
        return calculate_year_fraction(start, self._accrual_end(index), convention, fx_env.bus_days_in_year,
                                       calendar=fx_env.calendar)
```

Docstrings: module "Accrual:" paragraph — "dated form - period ``i`` runs from ``initial_date`` / the previous
period's accrual end to its own accrual end (``accrual_end_dates``; by default the fixing dates, i.e. the previous
fixing date to this fixing date) in the contract ``day_count_convention`` (default ACT/365), or explicit factors."
Class Attributes, after `day_count_convention`: "accrual_end_dates: Dated form: each period's accrual end, 1:1 with
``fixing_dates``, strictly increasing, between the period's fixing and pay dates; requires ``initial_date``, excludes
``accrual_factors``; ``None`` = the fixing dates (periods may then not share a fixing date)."

Append to the CHANGELOG `### Added`:

```markdown
- `FxTargetRedemptionNote(accrual_end_dates=...)` (dated form; user ruling 2026-10-10 on R26, option (b)): each
  period's own accrual end date, separate from its fixing date. Period `i` accrues from `initial_date` (first) or the
  previous period's end to its own end in the contract day count, so periods may share a fixing date without explicit
  `accrual_factors`. One end per fixing, strictly increasing, each between its fixing and pay dates; it requires
  `initial_date` and excludes `accrual_factors`. Without it the ends are the fixing dates: every existing booking keeps
  its factors and value exactly; such a contract still cannot share a fixing date (a zero-length period).
```

Append to `moved_values.txt`:

```text
#
# Task 8d - user ruling 2026-10-10 on R26, option (b): the TARN's own accrual period dates (accrual_end_dates).
# Existing bookings: no value moves (accrual_end_dates=None calls calculate_year_fraction with today's dates; pinned
# bitwise by test_accrual_end_dates_on_the_fixing_dates_reproduce_the_default_bitwise, calendar and business-day).
# New bookings: accrual_end_dates given (incl. periods sharing a fixing date without accrual_factors): TypeError ->
# priced, equal to the numeric twin built from the per-period factors.
# Narrowed refusal (Task 8c): a shared fixing date is refused only under the default derivation; message now names
# accrual_end_dates or accrual_factors.
# Refused pending OQ-5 / OQ-6 (non-blocking): an accrual end before its fixing date or after its pay date; gapped or
# overlapping periods by dates. Both still book with accrual_factors.
```

- [ ] **Step 4: Run to verify they pass**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_state.py test/test_fx_target_redemption_lifecycle.py test/test_fx_tarn_note_mc.py test/test_fx_tarf_mc.py -n 2 -q`
Expected: all pass; every existing test unmodified — D5 (`test_d5_initial_date_gives_act_365_factors`,
`test_d5_business_days_contract_count_reads_the_environment_calendar`, `test_d5_both_period_sources_refused`,
`test_d5_neither_period_source_refused`), `test_r2_dated_refusals` (incl. `fixing_dates=FIX_DATES[:1] * 2`, now the
narrowed default-derivation refusal) and Task 8c's `test_the_default_derivation_cannot_share_a_fixing_date` ("zero
length" still matched). Any other failure: STOP and report it as an unlisted mover.

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/asset/fx/product/option/fx_target_redemption_note.py test/test_fx_target_redemption_lifecycle.py CHANGELOG.md
git -C <WT> add -f docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/moved_values.txt
git -C <WT> commit -m "feat(fx-tarn): the TARN's own accrual period dates, separate from its fixing dates (R26 b)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: TARN / TARF implement the run-fixings product protocol (W6; D2, B5 at recorder level)

Re-planned 2026-10-10 against W6 (see "Re-plan 2026-10-10 against W6"). Spec: TARN §6, §7.2 as amended by run-fixings
spec §6 (user-approved), §10 D2 / B5; run-fixings spec §3.2, §3.3 (R1–R10), §6; W6 plan "Hand-off to W4b" and its
Rulings 2, 4, 8, 9, 10, 11. **No `quantark/asset/fx/lifecycle/` is built**: W6's `RunFixingRecorder` is the loop.

**Files:**
- Modify: `quantark/asset/fx/product/option/target_redemption_base.py`, `quantark/util/run_fixings.py` (one optional
  hook, `settle_history`), `test/fx_target_redemption_fixtures.py` (add `book`, `drive`),
  `test/test_fx_target_redemption_lifecycle.py` (protocol unit tests, D2), `test/test_run_fixings.py` (append one
  generic hook test), `CHANGELOG.md`
- Create: `test/test_fx_target_redemption_runs.py`

**Interfaces:**
- Consumes: Tasks 2–8, 8a (cash due today stays pending until its pay time is strictly past), 8c (periods may share a
  fixing date / time), 8d (`accrual_end_dates`: factors only, nothing in the run protocol reads period dates) and 8b
  (dated fixings
  banked per fixing — `fixings_banked`, `_banked_count`, `_is_unbanked` by identity, `as_of` = `_state_date`; a fixing
  or pay date at zero business time is passed/due by time); W6 `quantark.util.run_fixings` (`RunFixingRecorder.register(portfolio) -> RunFixingDay`,
  `.advance(portfolio) -> RunFixingDay`, `.days`, `.recorded`; `RunFixingDay(date, fixings, product_events,
  pending_pv)`; `RecordedFixing(position_id, index, label, outcome, spot, gap)`; `participates(product)`); W5b
  `quantark.util.fixings` (`is_history`, `snap_time`, `freeze_missing_fixings`, `MissingFixingWarning`); W1
  `MaturedBeforeRunWarning` (raised by the recorder, imported by the tests from
  `quantark.asset.equity.lifecycle.autocallable`).
- Produces on `TargetRedemptionProduct` (TARN and TARF, both forms; W6 hand-off note 1):
  - `has_run_fixings` (property) -> `True`.
  - `open_fixings(fx_env) -> Dict[int, Tuple[float, str]]`: `{index: (time, label)}` of every fixing not banked and
    without a recorded rate; numeric times as listed, dated times `signed_year_fraction(date, fx_env)`; none once
    terminal.
  - `with_time_elapsed(dt) -> TargetRedemptionProduct`: numeric — a copy aged by `time_shift(dt)` (pending cash
    reaching its pay time leaves as a PAYMENT event, spec §6), fixing and pay times then `snap_time`d, maturity /
    delivery refreshed, nothing banked; dated, or `is_zero(dt)`: `self`.
  - `with_history_settled(fx_env) -> TargetRedemptionProduct` (new run hook, Ruling R13): a copy with every strictly
    past (`is_history`) unbanked fixing that HAS a recorded rate banked at that rate, in order, stopping at the first
    one that is live or unrecorded (clip, redemption, dead fixings, FIXING / PAYMENT events through the Task 5
    arithmetic), and — dated — every pending payment whose pay time is strictly past (`is_history`, Task 8a) paid;
    dated `fixings_banked` counts what it banked and `as_of` becomes `_state_date(fx_env)` (Task 8b; never after the
    valuation date — R15 superseded by the OQ-3 ruling). Nothing is ever filled at `fx_env`'s spot. `self` when
    nothing changes.
  - `fixings_banked` (property) -> `int`: fixings banked since booking, by fixing identity — dated: `_banked_count()`
    (Task 8b, schedule positions `0 .. n − 1`); numeric: `_index_offset` (R14, the fixings dropped from the listed
    schedule). A numeric twin booked with its dated contract reports the same count at every step (D2 test).
  - `matured_before(fx_env) -> bool`: not terminal and the last fixing's time is `is_history` (numeric: listed time;
    dated: `signed_year_fraction`) — Ruling R16.
  - Changed: `missing_fixing_fills` / `with_missing_fixings_filled` list only `is_history` fixings (a t = 0 fixing is
    live, W6 D1); `resolve` still banks a t = 0 fixing on its copy at the quoted (possibly bumped) spot but names only
    `is_history` fixings in its `MissingFixingWarning`; protocol indices are booking-relative (Ruling R14): dated =
    schedule position, numeric = `_index_offset` (fixings banked and dropped since construction, dead ones included) +
    listed position; `with_fixings_recorded` maps them back; numeric FIXING / PAYMENT labels use the same index. At
    construction `_index_offset == 0`, so every Task 2–8 index and label is unchanged.
  - `lifecycle_events` stays append-only (W6 hand-off note 3): every path (`time_shift`, banking, settle) builds
    `self.lifecycle_events + tuple(new)` on the product or its copy.
- Produces in `quantark.util.run_fixings`: `settle_history(product, env) -> product` — calls the optional hook
  `with_history_settled(env)` when the product has it, else returns the product. `RunFixingRecorder` calls it once at a
  position's registration (after W5b's freeze, so registration's banking and payments are never reported — W6
  Ruling 10, TARN §7.2 register step 4) and at every clock-moving `advance` right after `step_fixings` (before the
  `lifecycle_events` diff, so the step's FIXING and PAYMENT events reach `RunFixingDay.product_events`). A same-date
  `advance` (W6 Ruling 9) does not call it. Products without the hook (range accruals, Asians) are untouched.
- Fixtures (`test/fx_target_redemption_fixtures.py`): `book(*products, quantities=None, n=2_000, fx_env=None) ->
  (FXPortfolio, List[str])`; `drive(portfolio, recorder, date, spot, pair="EURUSD") -> RunFixingDay`.
- Recorder precondition (W6 hand-off note 4): nothing in Tasks 9–11 replaces a participant's product between
  `register` and `advance` except the recorder itself (with copies that keep their records).

- [ ] **Step 0: Confirm W6 is in this branch's base**

Run:
```bash
git -C <WT> merge-base --is-ancestor 6f9c6e74 HEAD && echo W6_IN_BASE
env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -c "
from quantark.util.run_fixings import RunFixingRecorder, RunFixingDay, participates
from quantark.util.fixings import is_history, snap_time
assert 'pending_pv' in RunFixingDay.__dataclass_fields__ and 'product_events' in RunFixingDay.__dataclass_fields__
print('W6 interfaces present')"
```
Expected: `W6_IN_BASE` and `W6 interfaces present`. Otherwise STOP and report.

- [ ] **Step 1: Fixtures and failing tests**

Append to `test/fx_target_redemption_fixtures.py` (move `import dataclasses` and the `FXPortfolio` import to the top
of the module with the other imports):

```python
import dataclasses

from quantark.portfolio.fx import FXPortfolio


def book(*products, quantities=None, n=2_000, fx_env=None):
    """An ``FXPortfolio`` on EURUSD holding ``products`` (quantity 1 each unless given), each with its MC engine; returns
    ``(portfolio, position_ids)``. The products themselves are held: the run engines clone the book, the recorder
    replaces a held product only with copies."""
    pf = FXPortfolio(portfolio_name="w4b", pricing_environments={"EURUSD": fx_env or env()})
    ids = []
    for i, prod in enumerate(products):
        q = 1.0 if quantities is None else quantities[i]
        ids.append(pf.add_position(product=prod, quantity=q, entry_price=0.0, underlying="EURUSD",
                                   engine=engine_for(prod, n=n)).position_id)
    return pf, ids


def drive(portfolio, recorder, date, spot, pair="EURUSD"):
    """Run instant ``date`` at ``spot`` on ``pair`` (a replaced environment, as the FX backtest's ``_apply_row``), then
    ``recorder.advance`` (W6 run fixings); returns the ``RunFixingDay``."""
    e = portfolio.pricing_environments[pair]
    portfolio.pricing_environments[pair] = dataclasses.replace(e, valuation_date=date,
                                                               spot_quote=SpotQuote(spot=spot))
    return recorder.advance(portfolio)
```

Append to `test/test_fx_target_redemption_lifecycle.py` (imports to add at the top: `book`, `drive` from
`fx_target_redemption_fixtures`; `from quantark.util.run_fixings import RunFixingRecorder, participates`):

```python
# --- Run-fixings protocol (W6 hand-off; plan re-plan 2026-10-10) --------------------------------------------------

DAY = 1 / 365


def _days(product, n):
    for _ in range(n):
        product = product.with_time_elapsed(DAY)
    return product


@pytest.mark.parametrize("make", [lambda: note(target=3.0), lambda: tarf(target=0.20),
                                  lambda: dated_note(target=3.0), lambda: dated_tarf(target=0.20)])
def test_target_redemption_products_take_part_in_run_fixings(make):
    product = make()
    assert product.has_run_fixings and participates(product)


def test_open_fixings_lists_the_unbanked_fixings_by_booking_index():
    n, d = note(target=3.0), dated_note(target=3.0)
    assert n.open_fixings(env())[0] == (FIX[0], "#0 t=0.0833333")
    assert d.open_fixings(env())[11] == (1.0, "#11 2027-06-15")
    assert sorted(n.open_fixings(env())) == sorted(d.open_fixings(env())) == list(range(12))
    e = env(val=FIX_DATES[1] + timedelta(days=1))
    banked = copy.deepcopy(d)
    banked.record_passed_fixings(e)
    assert sorted(banked.open_fixings(e)) == list(range(2, 12))


def test_numeric_indices_count_from_the_booking_across_banking():
    """Ruling R14: a numeric contract drops banked fixings, so its protocol index counts from the booking."""
    aged = _days(note(target=50.0), 32)                                  # fixing #0 (30.4 days) is history
    assert aged.missing_fixing_fills(env()) == {0: 1.20}
    settled = aged.with_fixings_recorded({0: 1.20}).with_history_settled(env())
    assert sorted(settled.open_fixings(env()))[:2] == [1, 2]             # #0 banked and dropped; #1 keeps its index
    assert settled.with_fixings_recorded({0: 1.05}) is settled           # #0 is banked: nothing left to record
    later = _days(settled, 30)                                           # #1 (60.8 days) is history on day 62
    assert later.missing_fixing_fills(env(spot=1.30)) == {1: 1.30}
    events = later.with_fixings_recorded({1: 1.30}).with_history_settled(env()).lifecycle_events
    assert [(x.event_type, x.label[:3]) for x in events] == [(FIXING, "#0 "), (PAYMENT, "#0 "),
                                                              (FIXING, "#1 "), (PAYMENT, "#1 ")]


def test_with_time_elapsed_is_a_snapping_copy_that_banks_nothing():
    n = note(fix=[31 / 365] + FIX[1:], target=3.0)
    aged = _days(n, 31)
    assert n.fixing_times[0] == 31 / 365 and aged is not n
    assert aged.fixing_times[0] == 0.0 and aged.open_fixings(env())[0] == (0.0, "#0 t=0")
    assert aged.state == INCEPTION and aged.lifecycle_events == ()
    assert is_close(aged.maturity, 1.0 - 31 / 365, rel_tol=0.0, abs_tol=1e-14)
    with pytest.warns(TerminalBookingWarning):
        owed = note(fix=[], accrual_factors=[], target=1.5,
                    state=TargetRedemptionState(1.5, pending=(PendingPayment(100.5, pay_time=10 / 365),)))
    due = _days(owed, 10)                                                # its pay day: due today (Task 8a)
    assert due.state.pending == (PendingPayment(100.5, pay_time=0.0),) and due.lifecycle_events == ()
    paid = _days(owed, 11)
    assert owed.state.pending and paid.state.pending == ()
    assert [(x.event_type, x.amount) for x in paid.lifecycle_events] == [(PAYMENT, 100.5)]
    d = dated_note(target=3.0)
    assert d.with_time_elapsed(0.5) is d and n.with_time_elapsed(0.0) is n


@pytest.mark.parametrize("form", ["numeric", "dated"])
def test_a_fixing_at_t0_is_live_silent_and_never_frozen(form):
    """W6 D1: today's fixing moves with the (bumped) spot; only history is frozen or settled."""
    if form == "numeric":
        product, e = note(fix=[0.0] + FIX[1:], accrual_start=-30 / 365, target=3.0), env()
    else:
        product, e = dated_note(target=3.0), env(val=FIX_DATES[0])
    assert product.missing_fixing_fills(e) == {}
    assert freeze_missing_fixings(product, e) is product
    assert product.with_history_settled(e) is product
    assert product.open_fixings(e)[0][0] == 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        view = product.resolve(e)                                          # decided at the quoted spot, silently
    banked = copy.deepcopy(product)
    banked.record_passed_fixings(e)
    assert view == banked.resolve(e)


@pytest.mark.parametrize("form", ["numeric", "dated"])
def test_with_history_settled_banks_only_recorded_history(form):
    if form == "numeric":
        product, e = _days(note(target=3.0, pay_times=[t + 2 / 365 for t in FIX]), 31), env()
    else:
        product, e = dated_note(target=3.0, pay_dates=LAG2), env(val=FIX_DATES[0] + timedelta(days=1))
    assert product.with_history_settled(e) is product                    # history without a rate: never filled here
    assert product.missing_fixing_fills(e) == {0: 1.20}
    settled = product.with_fixings_recorded({0: 1.25}).with_history_settled(e)
    assert [(x.event_type, x.fixing) for x in settled.lifecycle_events] == [(FIXING, 1.25)]
    assert settled.state.accumulated > 0.0 and len(settled.state.pending) == 1   # paid two days after the fixing
    assert product.state == INCEPTION and product.missing_fixing_fills(e) == {0: 1.20}  # the product is a copy
    if form == "dated":
        assert (settled.state.fixings_banked, settled.state.as_of) == (1, e.valuation_date)   # per fixing (Task 8b)
        assert settled.with_history_settled(env(val=LAG2[0])) is settled          # due today: pending (Task 8a)
        paid = settled.with_history_settled(env(val=LAG2[0] + timedelta(days=1)))
        assert [x.event_type for x in paid.lifecycle_events] == [FIXING, PAYMENT] and paid.state.pending == ()


def test_matured_before_is_strict_history_of_the_last_fixing():
    assert note(fix=[-60 / 365, -30 / 365], accrual_start=-90 / 365).matured_before(env())
    assert not note(fix=[-60 / 365, 0.0], accrual_start=-90 / 365).matured_before(env())   # live today
    two = dated_note(fixing_dates=FIX_DATES[:2])
    assert two.matured_before(env(val=FIX_DATES[1] + timedelta(days=1)))
    assert not two.matured_before(env(val=FIX_DATES[1]))
    with pytest.warns(TerminalBookingWarning):
        owed = note(fix=[], accrual_factors=[], target=1.5,
                    state=TargetRedemptionState(1.5, pending=(PendingPayment(100.5, pay_time=10 / 365),)))
    assert not owed.matured_before(env())                                # terminal: kept at PV(pending), spec 5.3


_D2_SPOT = {("tarn", "quiet"): 1.05, ("tarn", "redeeming"): 1.20, ("tarn_no_principal", "quiet"): 1.05,
            ("tarn_no_principal", "redeeming"): 1.20, ("tarf", "quiet"): 1.15, ("tarf", "redeeming"): 1.20}


@pytest.mark.parametrize("path", ["quiet", "redeeming"])
@pytest.mark.parametrize("kind", ["tarn", "tarn_no_principal", "tarf"])
def test_d2_twins_age_identically(kind, path):
    """D2 on the W6 recorder: a fixing is live on its own day in both forms and recorded the next day at that day's
    spot; events, recordings and state bitwise, prices relative 1e-12, on every checkpoint."""
    spot = _D2_SPOT[(kind, path)]
    if kind == "tarf":
        dated = dated_tarf(pay_dates=LAG2, target=0.075, gearing=2.0)
    else:
        dated = dated_note(pay_dates=LAG2, target=1.5, include_principal=(kind == "tarn"))
    twin = _numeric_twin(dated, env())
    pf, (pid_d, pid_n) = book(dated, twin, n=4_000, fx_env=env(spot=spot))
    rec = RunFixingRecorder()
    rec.register(pf)
    checkpoints = {FIX_DATES[1] - timedelta(days=1), FIX_DATES[1], FIX_DATES[1] + timedelta(days=1),
                   VAL + timedelta(days=91), LAG2[1] + timedelta(days=1), LAG2[-1] + timedelta(days=1)}
    events = {pid_d: [], pid_n: []}
    recorded = {pid_d: [], pid_n: []}
    day = VAL
    while day < LAG2[-1] + timedelta(days=2):
        day += timedelta(days=1)
        result = drive(pf, rec, day, spot)
        for pid, _q, ev in result.product_events:
            events[pid].append((day, ev.event_type, ev.amount, ev.fixing, ev.redeemed))
        for f in result.fixings:
            recorded[f.position_id].append((day, f.index, f.outcome, f.spot, f.gap))
        if day in checkpoints:
            e = pf.pricing_environments["EURUSD"]
            a, b = pf.positions[pid_d], pf.positions[pid_n]
            assert a.product.state.accumulated == b.product.state.accumulated
            assert a.product.fixings_banked == b.product.fixings_banked          # same fixing identity (OQ-3)
            assert [p.amount for p in a.product.state.pending] == [p.amount for p in b.product.state.pending]
            assert is_close(a.get_market_value(e), b.get_market_value(e), rel_tol=1e-12)
            if day == FIX_DATES[1]:                                    # live on its own day, in both forms (W6 D1)
                assert a.product.open_fixings(e)[1][0] == b.product.open_fixings(e)[1][0] == 0.0
    assert events[pid_d] and events[pid_d] == events[pid_n]
    assert recorded[pid_d] and recorded[pid_d] == recorded[pid_n]
```

Create `test/test_fx_target_redemption_runs.py`:

```python
"""FX TARN / TARF ageing inside runs (spec 2026-10-09-fx-tarn-lifecycle-design.md section 7, tests B1-B6, Q8) on the
W6 run-fixings recorder (spec 2026-10-09-run-fixings-design.md section 6; TARN plan re-plan 2026-10-10)."""
import dataclasses
import warnings
from datetime import datetime, timedelta

import pytest

from fx_target_redemption_fixtures import (
    FIX, FIX_DATES, INCEPTION, LAG2, PAIR, TARGET_CAL, VAL, _numeric_twin, book, dated_note, drive, engine_for, env,
    note, tarf,
)
from quantark.asset.equity.lifecycle.autocallable import MaturedBeforeRunWarning
from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxVanillaOption
from quantark.asset.fx.product.option.target_redemption_state import (
    PendingPayment, TargetRedemptionEventType, TargetRedemptionState, TerminalBookingWarning,
)
from quantark.param import SpotQuote
from quantark.util.calendar import DayCountConvention
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError
from quantark.util.fixings import MissingFixingWarning
from quantark.util.run_fixings import RunFixingRecorder

FIXING, PAYMENT = TargetRedemptionEventType.FIXING, TargetRedemptionEventType.PAYMENT
BUS = dict(dc=DayCountConvention.BUSINESS_DAYS, calendar=TARGET_CAL)


def _rows(day, pid=None):
    return [(f.index, f.outcome, f.spot, f.gap) for f in day.fixings if pid is None or f.position_id == pid]


def _registered(*products, **kw):
    pf, ids = book(*products, **kw)
    rec = RunFixingRecorder()
    rec.register(pf)
    return pf, ids, rec


# ================================================================ Task 9: the recorder drives TARN / TARF


def test_register_fills_history_once_and_books_nothing():
    prod = note(fix=[-10 / 365, -5 / 365] + FIX[1:], accrual_start=-40 / 365, target=50.0)
    expected = prod.full_coupon(0) + prod.full_coupon(1)
    pf, (pid,) = book(prod)
    rec = RunFixingRecorder()
    with pytest.warns(MissingFixingWarning) as caught:
        day0 = rec.register(pf)
    msgs = [str(w.message) for w in caught if issubclass(w.category, MissingFixingWarning)]
    assert len(msgs) == 1 and "#0" in msgs[0] and "#1" in msgs[0]                 # W6 R5: once per position
    held = pf.positions[pid].product
    assert held.state.accumulated == expected
    assert [x.event_type for x in held.lifecycle_events] == [FIXING, PAYMENT, FIXING, PAYMENT]
    assert day0.product_events == () and day0.fixings == ()                       # W6 Ruling 10: nothing reported
    assert rec.advance(pf).product_events == ()                                   # same date (Ruling 9): no cash
    assert prod.state == INCEPTION and prod.lifecycle_events == ()               # the booked product is untouched


@pytest.mark.parametrize("form", ["dated", "numeric"])
def test_a_fixing_on_the_origin_date_is_live_and_records_the_next_day_silently(form):
    """Run-fixings spec 6 (W6 D1): not banked at registration; a same-date advance refreshes its spot (Ruling 9); it
    is recorded on the first later instant at that refreshed spot, without a warning."""
    if form == "dated":
        prod = dated_note(fixing_dates=[VAL] + FIX_DATES[1:], pay_dates=[VAL + timedelta(days=2)] + LAG2[1:],
                          initial_date=VAL - timedelta(days=30), target=50.0)
    else:
        prod = note(fix=[0.0] + FIX[1:], pay_times=[2 / 365] + [t + 2 / 365 for t in FIX[1:]],
                    accrual_start=-30 / 365, target=50.0)
    pf, (pid,) = book(prod)
    rec = RunFixingRecorder()
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        rec.register(pf)
        assert pf.positions[pid].product.state.accumulated == 0.0
        drive(pf, rec, VAL, 1.05)
        day1 = drive(pf, rec, VAL + timedelta(days=1), 1.30)
    assert _rows(day1) == [(0, 1.05, 1.05, False)]
    assert [(ev.event_type, ev.fixing, ev.amount) for _, _, ev in day1.product_events] == [(FIXING, 1.05, 0.0)]


@pytest.mark.parametrize("make", [
    lambda: note(fix=[-60 / 365, -30 / 365], accrual_start=-90 / 365),
    lambda: dated_note(fixing_dates=[VAL - timedelta(days=60), VAL - timedelta(days=30)],
                       initial_date=VAL - timedelta(days=90)),
])
def test_matured_before_the_run_is_removed_with_a_warning(make):
    pf, (pid,) = book(make())
    rec = RunFixingRecorder()
    with pytest.warns(MaturedBeforeRunWarning):
        day0 = rec.register(pf)
    assert pid not in pf.positions and day0.pending_pv == 0.0
    day1 = drive(pf, rec, VAL + timedelta(days=1), 1.20)
    assert (day1.product_events, day1.fixings, day1.pending_pv) == ((), (), 0.0)


def test_a_terminal_booking_is_kept_at_its_pending_pv_and_paid_after_its_day():
    with pytest.warns(TerminalBookingWarning):
        owed = note(fix=[], accrual_factors=[], target=1.5,
                    state=TargetRedemptionState(1.5, pending=(PendingPayment(100.5, pay_time=10 / 365),)))
    pf, (pid,) = book(owed)
    rec = RunFixingRecorder()
    day0 = rec.register(pf)
    e = pf.pricing_environments["EURUSD"]
    assert pid in pf.positions and day0.pending_pv == 100.5 * e.get_domestic_df(10 / 365)
    days = {k: drive(pf, rec, VAL + timedelta(days=k), 1.20) for k in range(1, 13)}
    paid = {k: [ev.amount for _, _, ev in d.product_events if ev.event_type is PAYMENT] for k, d in days.items()}
    assert days[10].pending_pv == 100.5                                           # due today: in the value (8a)
    assert paid[11] == [100.5] and sum(1 for p in paid.values() if p) == 1       # booked once its pay day is past


def test_the_run_clock_cannot_go_back():
    pf, _, rec = _registered(note(target=3.0))
    drive(pf, rec, VAL + timedelta(days=2), 1.20)
    with pytest.raises(ValidationError, match="cannot go back"):                  # W6 R9
        drive(pf, rec, VAL + timedelta(days=1), 1.20)


def test_a_time_of_day_instant_is_refused():
    pf, _, rec = _registered(dated_note(target=3.0))
    with pytest.raises(ValidationError, match="time of day"):                     # W6 R8
        drive(pf, rec, VAL + timedelta(days=1, hours=15), 1.20)


def test_other_products_are_untouched():
    vanilla = FxVanillaOption(strike=1.2, option_type=OptionType.CALL, maturity=1.0, notional_foreign=1.0,
                              currency_pair=PAIR)
    pf, _ = book(note(target=3.0))
    pid = pf.add_position(product=vanilla, quantity=1.0, entry_price=0.0, underlying="EURUSD",
                          engine=GarmanKohlhagenEngine()).position_id
    rec = RunFixingRecorder()
    rec.register(pf)
    drive(pf, rec, VAL + timedelta(days=40), 1.20)
    assert pf.positions[pid].product is vanilla and vanilla.maturity == 1.0


def test_a_calendar_gap_records_at_the_previous_instants_spot_without_a_warning():
    """W6 D5: the k/12 fixings #0 and #1 fall between two path instants: recorded at the PREVIOUS instant's spot
    (1.20, coupons), not the new day's (1.05, below the strike), gap=True, no warning; banked in order."""
    prod = note(target=50.0)
    c = [prod.full_coupon(0), prod.full_coupon(1)]
    pf, _, rec = _registered(prod)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        day = drive(pf, rec, VAL + timedelta(days=70), 1.05)
    assert _rows(day) == [(0, 1.20, 1.20, True), (1, 1.20, 1.20, True)]
    kinds = [(ev.event_type, ev.amount) for _, _, ev in day.product_events]
    assert kinds == [(FIXING, c[0]), (PAYMENT, c[0]), (FIXING, c[1]), (PAYMENT, c[1])]


def test_a_short_position_carries_its_signed_quantity():
    prod = tarf(gearing=2.0, target=0.2, pay_times=[t + 0.02 for t in FIX])
    pf, (pid,) = book(prod, quantities=[-2.0])
    rec = RunFixingRecorder()
    rec.register(pf)
    drive(pf, rec, VAL + timedelta(days=30), 1.10)                               # fixing #0 (30.4 days) still ahead
    fixing_day = drive(pf, rec, VAL + timedelta(days=31), 1.10)                  # recorded at day 30's 1.10 (gap)
    (cash,) = pf.positions[pid].product.state.pending
    assert cash.amount < 0.0                                                     # below the 1.15 strike: a liability
    assert [(q, ev.event_type) for _, q, ev in fixing_day.product_events] == [(-2.0, FIXING)]
    e = pf.pricing_environments["EURUSD"]
    assert fixing_day.pending_pv == -2.0 * (cash.amount * e.get_domestic_df(cash.pay_time))
    pay_day = drive(pf, rec, VAL + timedelta(days=38), 1.10)
    assert [(q, ev.event_type, ev.amount) for _, q, ev in pay_day.product_events] == [(-2.0, PAYMENT, cash.amount)]


def test_two_pairs_record_at_their_own_previous_spots():
    jpy_tarf = tarf(strike=145.0, currency_pair=CurrencyPair("USD", "JPY"), target=1e9)
    pf, (pid_eur,) = book(note(target=50.0))
    pf.pricing_environments["USDJPY"] = env(spot=150.0)
    pid_jpy = pf.add_position(product=jpy_tarf, quantity=1.0, entry_price=0.0, underlying="USDJPY",
                              engine=engine_for(jpy_tarf, n=2_000)).position_id
    rec = RunFixingRecorder()
    rec.register(pf)

    def move(days, eur, jpy):
        for pair, spot in (("EURUSD", eur), ("USDJPY", jpy)):
            e = pf.pricing_environments[pair]
            pf.pricing_environments[pair] = dataclasses.replace(e, valuation_date=VAL + timedelta(days=days),
                                                                spot_quote=SpotQuote(spot=spot))
        return rec.advance(pf)

    move(30, 1.05, 151.0)
    day = move(31, 1.30, 140.0)                                                  # both fixings #0 pass in the gap
    by_pid = {pid: ev for pid, _, ev in day.product_events if ev.event_type is FIXING}
    assert by_pid[pid_eur].fixing == 1.05 and by_pid[pid_eur].amount == 0.0      # below the TARN strike
    assert by_pid[pid_jpy].fixing == 151.0 and by_pid[pid_jpy].amount == 1.0 * max(151.0 - 145.0, 0.0)


def test_consecutive_daily_fixings_keep_their_own_live_spots_across_banking():
    """Review Focus 6 / Ruling R14: on day 6 fixing #0 banks (the numeric twin drops it) while #1 goes live; #1 must
    keep ITS first-live spot and gap=False, as in the dated contract."""
    dates = [VAL + timedelta(days=k) for k in (5, 6, 7)]
    dated = dated_note(fixing_dates=dates, pay_dates=[d + timedelta(days=3) for d in dates], target=50.0)
    pf, (pid_d, pid_n), rec = _registered(dated, _numeric_twin(dated, env()))
    spot = lambda k: 1.00 + 0.02 * k                                              # noqa: E731
    for k in range(1, 10):
        drive(pf, rec, VAL + timedelta(days=k), spot(k))
    rows = {pid: [(f.index, f.outcome, f.spot, f.gap) for f in rec.recorded if f.position_id == pid]
            for pid in (pid_d, pid_n)}
    assert rows[pid_d] == rows[pid_n] == [(i, spot(5 + i), spot(5 + i), False) for i in range(3)]


def test_lifecycle_events_are_append_only_in_a_run():
    pf, (pid,), rec = _registered(dated_note(target=1.5, pay_dates=LAG2))
    before = pf.positions[pid].product.lifecycle_events
    for k in range(1, 70):
        drive(pf, rec, VAL + timedelta(days=k), 1.20)
        now = pf.positions[pid].product.lifecycle_events
        assert now[:len(before)] == before
        before = now
    assert [ev.event_type for ev in before] == [FIXING, PAYMENT, FIXING, PAYMENT, PAYMENT]


def test_zero_lag_cash_of_a_live_fixing_stays_in_the_value_until_recorded():
    """OQ-1 (b), user 2026-10-10 (Task 8a): the fixing dated day 30 (pay date = fixing date) is live on day 30 and its
    cash is due that day, so day 30 holds it as pending PV (in the value, nothing booked); on day 31 the fixing is
    recorded, the cash is past and leaves the value as a PAYMENT - no dip."""
    pf, _, rec = _registered(dated_note(target=3.0))
    days = {k: drive(pf, rec, VAL + timedelta(days=k), 1.20) for k in range(1, 32)}
    fixing, payment = [ev for _, _, ev in days[31].product_events]
    assert (fixing.event_type, payment.event_type) == (FIXING, PAYMENT) and payment.amount == fixing.amount
    assert days[30].product_events == () and days[30].pending_pv == fixing.amount
    assert days[31].pending_pv == 0.0


def test_a_weekend_fixing_on_a_business_day_clock_is_live_on_friday():
    """OQ-2 (b), user 2026-10-10 (Task 8b): under BUSINESS_DAYS a Saturday fixing is t = 0 on the Friday, live in both
    forms (W6 Ruling 2); the run does not stop; on Monday it is recorded at Friday's spot, gap=False, twins identical."""
    friday, saturday, monday = datetime(2026, 8, 14), datetime(2026, 8, 15), datetime(2026, 8, 17)
    dated = dated_note(fixing_dates=[datetime(2026, 7, 15), saturday, datetime(2026, 9, 15)], target=50.0)
    pf, (id_d, id_n), rec = _registered(dated, _numeric_twin(dated, env(**BUS)), fx_env=env(**BUS))
    drive(pf, rec, friday - timedelta(days=1), 1.20)
    fri = drive(pf, rec, friday, 1.21)
    e = pf.pricing_environments["EURUSD"]
    assert pf.positions[id_d].product.open_fixings(e)[1][0] == pf.positions[id_n].product.open_fixings(e)[1][0] == 0.0
    assert fri.pending_pv > 0.0                                   # the live fixing's cash is due "today" (Task 8a)
    mon = drive(pf, rec, monday, 1.30)
    assert [r for r in _rows(mon, id_d) if r[0] == 1] == [r for r in _rows(mon, id_n) if r[0] == 1] == [
        (1, 1.21, 1.21, False)]
    assert [ev.event_type for p, _, ev in mon.product_events if p == id_d] == [FIXING, PAYMENT]




def test_oq3_friday_and_saturday_fixings_in_a_run_are_recorded_once_at_fridays_spot():
    """OQ-3 ruling in a run (BUSINESS_DAYS, weekday instants): the Friday fixing and the Saturday fixing are both live
    on the Friday (W6 Ruling 4); on Monday both are recorded at Friday's spot, gap=False, banked by identity with
    their cash; nothing is banked again later. (Pricing either form before the Friday: Task 8c.)"""
    friday, saturday, monday = datetime(2026, 8, 14), datetime(2026, 8, 15), datetime(2026, 8, 17)
    dated = dated_note(fixing_dates=[datetime(2026, 7, 15), friday, saturday, datetime(2026, 9, 15)], target=50.0)
    pf, (pid,), rec = _registered(dated, fx_env=env(**BUS))
    drive(pf, rec, friday - timedelta(days=1), 1.20)
    fri = drive(pf, rec, friday, 1.21)
    assert fri.product_events == () and fri.pending_pv > 0.0                     # both live; their cash due today
    mon = drive(pf, rec, monday, 1.30)
    assert [r for r in _rows(mon) if r[0] in (1, 2)] == [(1, 1.21, 1.21, False), (2, 1.21, 1.21, False)]
    assert [ev.event_type for _, _, ev in mon.product_events] == [FIXING, PAYMENT, FIXING, PAYMENT]
    held = pf.positions[pid].product
    assert (held.fixings_banked, held.state.as_of) == (3, monday)
    tue = drive(pf, rec, monday + timedelta(days=1), 1.30)
    assert tue.product_events == () and pf.positions[pid].product.fixings_banked == 3


def test_periods_sharing_a_fixing_date_are_recorded_and_banked_once_in_a_run():
    """OQ-4 (b), Task 8c: two periods on one fixing date are both live on it, recorded the next day at its spot (one
    RecordedFixing each, gap=False) and banked together once; each keeps its own pay date (the second pays two days
    later)."""
    dates = [FIX_DATES[0]] + FIX_DATES
    dated = dated_note(fixing_dates=dates, pay_dates=[FIX_DATES[0], LAG2[0]] + FIX_DATES[1:],
                       accrual_factors=[1 / 12, 1 / 24] + [1 / 12] * 11, target=50.0)
    pf, (pid,), rec = _registered(dated)
    days = {k: drive(pf, rec, VAL + timedelta(days=k), 1.20 + 0.001 * k) for k in range(1, 34)}
    spot30 = 1.20 + 0.001 * 30                                                   # FIX_DATES[0] is day 30
    assert _rows(days[31]) == [(0, spot30, spot30, False), (1, spot30, spot30, False)]
    assert [ev.event_type for _, _, ev in days[31].product_events] == [FIXING, PAYMENT, FIXING]
    assert pf.positions[pid].product.fixings_banked == 2
    assert days[32].product_events == ()                                         # nothing banked again
    assert [ev.event_type for _, _, ev in days[33].product_events] == [PAYMENT]  # the second period's own pay date
```

Append to `test/test_run_fixings.py` (after `test_hooks_matured_events_pending_pv`):

```python
class _SettlingStub(_Stub):
    """A _Stub whose run state settles on the instant (``with_history_settled``), like W4b's TARN paying cash."""

    def with_history_settled(self, env):
        if ("SETTLED", env.valuation_date) in self.lifecycle_events:
            return self
        return _SettlingStub(self.matured, self.lifecycle_events + [("SETTLED", env.valuation_date)], self.recorded)

    def with_fixings_recorded(self, fills):
        out = super().with_fixings_recorded(fills)
        return out if out is self else _SettlingStub(out.matured, out.lifecycle_events, out.recorded)


def test_settle_hook_runs_at_register_and_on_every_clock_move():
    """W4b re-plan Ruling R13: ``with_history_settled`` runs after the first valuation's fill (its events never
    reported) and after each clock move's recording (its events reported); a same-date advance does not call it."""
    pf = _DuckBook(keep=(_SettlingStub(), 2.0))
    rec = RunFixingRecorder()
    day0 = rec.register(pf)
    assert day0.product_events == () and pf.positions["keep"].product.lifecycle_events == [("SETTLED", F.START)]
    assert rec.advance(pf).product_events == ()
    F.move(pf, 1)
    assert rec.advance(pf).product_events == (("keep", 2.0, ("SETTLED", F.START + timedelta(days=1))),)
```

- [ ] **Step 2: Run to verify they fail**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_runs.py test/test_fx_target_redemption_lifecycle.py test/test_run_fixings.py -k "run_fixings or open_fixings or booking_index or time_elapsed or t0 or history_settled or matured or d2 or register or origin or terminal_booking or clock or time_of_day or untouched or gap or short or pairs or consecutive or append or zero_lag or weekend_fixing or oq3 or periods_sharing or settle_hook" -n 2 -q`
Expected: FAIL — `AttributeError: ... object has no attribute 'has_run_fixings'` / `'open_fixings'` / `'with_time_elapsed'` /
`'with_history_settled'` / `'matured_before'`; the recorder-level tests fail because `participates()` is False (nothing
is banked: e.g. `assert 0.0 == ...`, `DID NOT WARN`); `test_a_fixing_at_t0_is_live_silent_and_never_frozen` fails on
`missing_fixing_fills(e) == {}` (today's fixing is listed); `test_settle_hook_runs_at_register_and_on_every_clock_move`
fails (the recorder never calls the hook). The other W6 tests in `test/test_run_fixings.py` still pass.

- [ ] **Step 3: Implement**

`quantark/asset/fx/product/option/target_redemption_base.py`:

1. Imports: `from quantark.util.fixings import fixing_label, is_history, snap_time, warn_filled_fixings` (Task 8a
   already added `is_history` and `snap_time`).
2. In BOTH `_init_dated` and `_init_numeric`, after `self._recorded_rates: Dict[int, float] = {}`:
   ```python
           #: Booking-relative index of the first listed fixing (numeric: fixings banked and dropped since construction,
           #: dead ones included; dated: always 0, the schedule is whole). Run-fixings protocol indices (Ruling R14).
           self._index_offset = 0
   ```
3. Replace `_bank_passed`, `_record_numeric` and `_record_dated` (the arithmetic of each banked fixing is unchanged;
   only the loop's stop rule in `history_only` mode and the numeric label/offset change; the dated block is Task 8b's
   per-fixing version with the `history_only` stop added):

```python
    def _bank_passed(self, fx_env, *, history_only: bool = False) -> List[Tuple[int, str]]:
        """``record_passed_fixings`` without the re-validation; returns ``(index, label)`` of every fixing banked at
        ``fx_env.spot`` because it had no recorded rate (the fills; index = listed position numeric, schedule position
        dated). ``history_only`` (the run's settle step, :meth:`with_history_settled`) banks only the leading strictly
        past fixings that have a recorded rate and never fills."""
        if self.is_numeric:
            return self._record_numeric(fx_env, history_only=history_only)
        return self._record_dated(fx_env, history_only=history_only)

    def _bankable_numeric(self, position: int, history_only: bool) -> bool:
        t = self.fixing_times[position]
        if history_only:
            return is_history(t) and position in self._recorded_rates
        return is_passed_time(t)

    def _record_numeric(self, fx_env, *, history_only: bool = False) -> List[Tuple[int, str]]:
        filled: List[Tuple[int, str]] = []
        acc = self.state.accumulated
        pending = list(self.state.pending)
        events: List[TargetRedemptionEvent] = []
        fixings, pays = self.fixing_times, self.pay_times
        factors = self.accrual_factors if self._has_accrual else None
        offset = self._index_offset
        banked = 0
        while banked < len(fixings) and self._bankable_numeric(banked, history_only):
            t, tp = fixings[banked], pays[banked]
            label = event_label(offset + banked, time=t)
            spot = self._rate(banked, fx_env, filled, label)
            added, cash = self._fixing_outcome(factors[banked] if factors is not None else None, spot, acc)
            acc = acc + added
            hit = is_target_hit(acc, self.target)
            ends = hit or banked == len(fixings) - 1
            events.append(TargetRedemptionEvent(TargetRedemptionEventType.FIXING, cash, fixing=spot, time=t,
                                                redeemed=ends, label=label))
            self._settle_numeric(cash, tp, label, pending, events)
            if ends:
                self._settle_numeric(self._principal_amount(), tp, label, pending, events)
            banked += 1
            if hit:
                banked = len(fixings)  # every later fixing is dead
                break
        self.fixing_times = fixings[banked:]
        self.pay_times = pays[banked:]
        if factors is not None:
            self.accrual_factors = factors[banked:]
        # the listed schedule lost its first ``banked`` fixings: recorded rates follow their fixings, and the
        # booking-relative index of the first listed fixing moves on by as many
        self._recorded_rates = {i - banked: r for i, r in self._recorded_rates.items() if i >= banked}
        self._index_offset = offset + banked
        self.state = TargetRedemptionState(acc, pending=tuple(sorted(pending, key=lambda p: p.pay_time)))
        self.lifecycle_events = self.lifecycle_events + tuple(events)
        self._refresh_numeric_horizon()
        return filled

    def _record_dated(self, fx_env, *, history_only: bool = False) -> List[Tuple[int, str]]:
        self._check_as_of(fx_env)
        filled: List[Tuple[int, str]] = []
        st = self.state
        events: List[TargetRedemptionEvent] = []
        pending: list = []
        for p in st.pending:  # cash whose pay time is strictly past was paid; cash due today stays (Task 8a)
            t_pay = signed_year_fraction(p.pay_date, fx_env)
            if is_history(t_pay):
                events.append(TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, p.amount, time=t_pay))
            else:
                pending.append(p)
        acc = st.accumulated
        banked = self._banked_count()
        if not self.is_terminal:
            last = len(self.fixing_dates) - 1
            for i in range(banked, len(self.fixing_dates)):
                fd = self.fixing_dates[i]
                t = signed_year_fraction(fd, fx_env)
                if not is_passed_time(t):  # ahead by TIME (Task 8b)
                    break
                if history_only and not (is_history(t) and i in self._recorded_rates):
                    break  # a live (t = 0) or unrecorded fixing stays open: the settle step never fills
                label = event_label(i, date=fd)
                spot = self._rate(i, fx_env, filled, label)
                added, cash = self._fixing_outcome(self._dated_factor(i, fx_env), spot, acc)
                acc = acc + added
                hit = is_target_hit(acc, self.target)
                ends = hit or i == last
                events.append(TargetRedemptionEvent(TargetRedemptionEventType.FIXING, cash, fixing=spot, time=t,
                                                    redeemed=ends, label=label))
                self._settle_dated(cash, self.pay_dates[i], label, fx_env, pending, events)
                banked = i + 1  # identity: fixings 0 .. i are banked (OQ-3 ruling), never re-banked by date
                if ends:
                    self._settle_dated(self._principal_amount(), self.pay_dates[i], label, fx_env, pending, events)
                    break  # every later fixing is dead (or there is none)
        # both modes: banked set by count, as_of never after the valuation date (Task 8b, R24; R15 superseded)
        self.state = TargetRedemptionState(acc, pending=tuple(sorted(pending, key=lambda p: p.pay_date)),
                                           as_of=self._state_date(fx_env), fixings_banked=banked)
        self.lifecycle_events = self.lifecycle_events + tuple(events)
        # a banked or dead fixing's recorded rate is spent
        self._recorded_rates = {i: r for i, r in self._recorded_rates.items()
                                if not self.is_terminal and self._is_unbanked(i)}
        return filled
```

4. In `resolve`, name only history in the warning (D1):

```python
            if warn_missing_fixings:
                warn_filled_fixings(type(self).__name__, "a banked outcome",
                                    [label for i, label in filled if self._is_history_index(i, fx_env)])
```

5. Replace the W5 fill protocol block (`missing_fixing_fills`, `_passed_fills`, `with_fixings_recorded`,
   `with_missing_fixings_filled`) and add the run-fixings members:

```python
    # -- W5 fill protocol (user decision 2026-10-09: a filled past fixing does not move) ------------------------
    def missing_fixing_fills(self, fx_env) -> Dict[int, float]:
        """``{index: fx_env.spot}`` for every STRICTLY past (``is_history``) unbanked fixing that banking would fill at
        the quoted spot (no recorded rate; fixings dead after a redemption inside the fill are not listed). A fixing at
        t = 0 is live (W6 D1) and is not listed. ``index`` is booking-relative (Ruling R14). No warning; the product
        is unchanged."""
        offset = self._index_offset
        return {offset + i: fx_env.spot for i, _ in self._history_fills(fx_env)}

    def _is_history_index(self, position: int, fx_env) -> bool:
        """Listed (numeric) or scheduled (dated) fixing ``position`` is strictly before valuation, by TIME (W6
        Ruling 2: on a business-day clock a weekend date can resolve to t = 0)."""
        if self.is_numeric:
            return is_history(self.fixing_times[position])
        return is_history(signed_year_fraction(self.fixing_dates[position], fx_env))

    def _history_fills(self, fx_env) -> List[Tuple[int, str]]:
        """``(position, label)`` of the strictly past fixings banking at ``fx_env`` would fill at its spot, found on a
        probe copy."""
        if not self._has_unbanked_passed(fx_env):
            return []
        return [(i, label) for i, label in deepcopy(self)._bank_passed(fx_env) if self._is_history_index(i, fx_env)]

    def with_fixings_recorded(self, fills: Mapping[int, float]) -> "TargetRedemptionProduct":
        """A COPY with each listed unbanked fixing's RATE recorded from ``fills`` (``{index: rate}``, booking-relative
        as in ``missing_fixing_fills``); banking (``record_passed_fixings``, ``resolve``, ``with_history_settled``)
        then uses that rate, so the outcome follows the contract's state at banking (a recorded fill can redeem it).
        A fixing that already has a recorded rate keeps it; a banked or out-of-range index is skipped. Returns
        ``self`` when nothing changes; the product is never changed. A rate recorded for a fixing still after
        valuation is refused at ``resolve``."""
        offset = self._index_offset
        rates = dict(self._recorded_rates)
        for index, rate in fills.items():
            position = index - offset
            if position in rates or not self._is_listed_unbanked(position):
                continue
            rate = float(rate)
            if not math.isfinite(rate) or rate <= 0.0:
                raise ValidationError(f"a recorded fixing rate must be positive and finite, got {rate}")
            rates[position] = rate
        if rates == self._recorded_rates:
            return self
        recorded = deepcopy(self)
        recorded._recorded_rates = rates
        return recorded

    def with_missing_fixings_filled(self, fx_env, *, warn: bool = True) -> "TargetRedemptionProduct":
        """This contract with every strictly past unbanked fixing's rate recorded at ``fx_env``'s QUOTED spot (Q3), on
        a COPY, so that bumped and shocked repricings bank it at the base spot; a fixing at t = 0 stays live (W6 D1).
        One ``MissingFixingWarning`` names the fixings filled (``warn=False`` skips it). Returns ``self`` when nothing
        is missing; never mutates."""
        filled = self._history_fills(fx_env)
        if not filled:
            return self
        if warn:
            warn_filled_fixings(type(self).__name__, "a banked outcome", [label for _, label in filled])
        offset = self._index_offset
        return self.with_fixings_recorded({offset + i: fx_env.spot for i, _ in filled})

    # -- run-fixings protocol (spec 2026-10-09 run fixings 3.2 and 6; TARN plan re-plan 2026-10-10) ---------------
    @property
    def has_run_fixings(self) -> bool:
        """Takes part in run fixings (``quantark.util.run_fixings``), both forms."""
        return True

    @property
    def fixings_banked(self) -> int:
        """Fixings banked since booking, by fixing identity (OQ-3 ruling): dated - ``_banked_count()`` (schedule
        positions ``0 .. n - 1``); numeric - ``_index_offset`` (the fixings dropped from the listed schedule, R14)."""
        return self._index_offset if self.is_numeric else self._banked_count()

    def open_fixings(self, fx_env) -> Dict[int, Tuple[float, str]]:
        """``{index: (time, label)}`` of every fixing not banked and without a recorded rate: numeric times as listed,
        dated times resolved against ``fx_env`` in its convention. ``index`` is booking-relative (Ruling R14). A
        terminal contract has none (dead fixings are not open)."""
        if self.is_terminal:
            return {}
        if self.is_numeric:
            offset = self._index_offset
            return {offset + i: (t, event_label(offset + i, time=t)) for i, t in enumerate(self.fixing_times)
                    if i not in self._recorded_rates}
        return {i: (signed_year_fraction(d, fx_env), event_label(i, date=d)) for i, d in enumerate(self.fixing_dates)
                if self._is_unbanked(i) and i not in self._recorded_rates}

    def with_time_elapsed(self, dt: float) -> "TargetRedemptionProduct":
        """The run clock's step (run-fixings R6): a COPY of a numeric contract aged by :meth:`time_shift` (pending cash
        reaching its pay time leaves as a PAYMENT event, spec 6) with every fixing and pay time within ``is_zero`` of
        valuation snapped to exactly 0.0 (``snap_time``: a twin's fixing lands on t = 0 on its day), nothing banked.
        A dated contract, or ``dt`` of zero, returns ``self``."""
        if not self.is_numeric or is_zero(dt):
            return self
        aged = deepcopy(self)
        aged.time_shift(dt)
        aged.fixing_times = [snap_time(t) for t in aged.fixing_times]
        aged.pay_times = [snap_time(t) for t in aged.pay_times]
        aged._refresh_numeric_horizon()
        aged.validate()
        return aged

    def with_history_settled(self, fx_env) -> "TargetRedemptionProduct":
        """Run hook (``quantark.util.run_fixings.settle_history``, Ruling R13): a COPY with every strictly past
        unbanked fixing that has a recorded rate banked at that rate, in order, stopping at the first live (t = 0) or
        unrecorded one (clip, redemption, dead fixings, FIXING / PAYMENT events), and every pending payment whose pay
        time is strictly past paid (dated; numeric cash is paid by :meth:`with_time_elapsed`; cash due today stays,
        Task 8a). Nothing is filled at ``fx_env``'s spot. ``self`` when nothing changes."""
        settled = deepcopy(self)
        settled._bank_passed(fx_env, history_only=True)
        if len(settled.lifecycle_events) == len(self.lifecycle_events):
            return self
        settled.validate()
        return settled

    def matured_before(self, fx_env) -> bool:
        """W1 rule at a run's registration (run-fixings recorder hook): not terminal, and the last fixing is strictly
        before ``fx_env``'s valuation by time (numeric: its listed time; dated: its resolved time, Ruling R16), so the
        state has not banked it. A terminal booking is kept at PV(pending) (spec 5.3)."""
        if self.is_terminal:
            return False
        if self.is_numeric:
            return is_history(self.fixing_times[-1])
        return is_history(signed_year_fraction(self.fixing_dates[-1], fx_env))
```

`quantark/util/run_fixings.py`:

1. Module docstring, append: "Optional product hooks for product-held run state (W4b TARN / TARF): ``matured_before(env)``,
   ``lifecycle_events`` (append-only), ``pending_pv(env)`` and ``with_history_settled(env)`` (the recorded history
   banked and the cash due paid, on a copy; ``settle_history``)."
2. After `at_spot`, add:

```python
def settle_history(product: Any, env: Any) -> Any:
    """The optional hook ``with_history_settled(env)`` of a product holding run state (W4b TARN / TARF): a copy with
    its recorded strictly past fixings banked and the cash due by ``env``'s date paid, or the product itself. A product
    without the hook is returned unchanged (TARN plan re-plan 2026-10-10, Ruling R13)."""
    hook = getattr(product, "with_history_settled", None)
    return hook(env) if callable(hook) else product
```

3. In `RunFixingRecorder._register_position`, replace `frozen = freeze_missing_fixings(position.product, env)` with:

```python
        # W5b's fill, then the product's own settle step (W4b): registration's banking and payments are before the
        # baseline and are never reported (Ruling 10)
        frozen = settle_history(freeze_missing_fixings(position.product, env), env)
```

4. In `RunFixingRecorder.advance`, right after the `recorded, new = step_fixings(...)` call, add:

```python
            recorded = settle_history(recorded, env)   # W4b: bank what was just recorded, pay what is due
```

   (before the `RecordedFixing` loop; the `prev_events` / `new_events` diff below then reports the step's FIXING and
   PAYMENT events.) Update the class docstring's hook sentence to name `with_history_settled`.

Append to the CHANGELOG `### Added`:

```markdown
- FX TARN / TARF take part in run fixings (`quantark.util.run_fixings`, W6): `has_run_fixings`,
  `open_fixings(fx_env)`, `with_time_elapsed(dt)` (a numeric copy aged by `time_shift`, times snapped to exactly 0.0,
  nothing banked), `with_history_settled(fx_env)` (recorded history banked and cash due paid, on a copy) and
  `matured_before(fx_env)`. In the FX backtest and FX dynamic scenario they age with the run's clock: a fixing is
  recorded at the spot of the first instant it was live (a fixing between path instants at the previous instant's,
  `gap=True`), banked in order with its cash; a contract whose last fixing is before the run start is removed with
  `MaturedBeforeRunWarning`.
- `RunFixingRecorder` calls an optional product hook `with_history_settled(env)` after the first valuation's fill and
  after each clock move's recording (`settle_history`); products without it are unchanged.
```

and to `### Changed`:

```markdown
- FX TARN / TARF: a fixing at exactly the valuation instant (t = 0) is live (run-fixings D1):
  `missing_fixing_fills` / `with_missing_fixings_filled` list only strictly past fixings, so the Greek freeze leaves
  today's fixing to move with the bumped spot, and pricing decides it at the quoted spot without a
  `MissingFixingWarning`. Numeric protocol indices and event labels count from the booking (fixings banked since
  construction + listed position).
```

- [ ] **Step 4: Run to verify they pass**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_runs.py test/test_fx_target_redemption_lifecycle.py test/test_fx_target_redemption_state.py test/test_fx_tarn_note_mc.py test/test_fx_tarf_mc.py test/test_fx_base_engine.py test/test_run_fixings.py test/test_run_fixings_runs.py -n 2 -q`
Expected: all pass. `test/test_run_fixings.py` changed only by the appended stub test; `test/test_run_fixings_runs.py`
unmodified (W6's range-accrual / Asian behaviour is unchanged: they have no `with_history_settled`).

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/asset/fx/product/option/target_redemption_base.py quantark/util/run_fixings.py test/fx_target_redemption_fixtures.py test/test_fx_target_redemption_runs.py test/test_fx_target_redemption_lifecycle.py test/test_run_fixings.py CHANGELOG.md
git -C <WT> commit -m "feat(fx-tarn): TARN/TARF implement the run-fixings protocol; the recorder settles product-held history

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: FX backtest books TARN / TARF cash on W6's recorder (B1–B5 backtest, B6 invariants)

Re-planned 2026-10-10 against W6. Spec §7.1, §7.3 (FX backtest), §10 B1–B6; run-fixings spec §3.5 (the FX backtest
already calls `register` after its first row and `advance` after every row: `quantark/backtest/fx/engine.py:50-51,
66`). This task adds ONLY the cash ledger and the two columns; it builds no wiring.

**Files:**
- Modify: `quantark/asset/fx/product/option/target_redemption_state.py` (add `booked_cash`),
  `quantark/backtest/fx/engine.py:1-130`, `CHANGELOG.md`
- Test: `test/test_fx_target_redemption_runs.py`

**Interfaces:**
- Consumes: Task 9 (TARN/TARF participate; `RunFixingDay.product_events` carries their FIXING / PAYMENT events with the
  position's signed quantity; `RunFixingDay.pending_pv` = Σ quantity × `pending_pv`); `FXBacktestEngine.run_fixings`
  (W6, the last run's recorder).
- Produces:
  - `quantark.asset.fx.product.option.target_redemption_state.booked_cash(product_events) -> float`: Σ quantity ×
    amount over the PAYMENT `TargetRedemptionEvent`s of one step's `(position_id, quantity, event)` entries, summed in
    order from 0.0; any other entry is ignored (used by Tasks 10 and 11).
  - Backtest rows gain `paid_cash` (cumulative, quantity-scaled) and `pending_pv` (that row's
    `RunFixingDay.pending_pv`); `portfolio_value` = Σ position MTM + cumulative paid cash; `final_value` likewise;
    `portfolio_pnl` / `net_pnl` follow `portfolio_value`. A book without TARN/TARF: `paid_cash` stays 0.0 and every
    stored number is unchanged (B6, Task 12).
- Does not replace any participant's product (W6 recorder precondition).

- [ ] **Step 1: Write the failing tests**

Append to `test/test_fx_target_redemption_runs.py` (imports to add at the top: `copy`, `pandas as pd`,
`from quantark.asset.fx.engine.mc import FxTargetRedemptionNoteMCEngine`,
`from quantark.asset.fx.product.option.target_redemption_state import TargetRedemptionEvent, booked_cash`,
`from quantark.backtest import FXBacktestConfig, FXBacktestEngine`,
`from quantark.backtest.strategy import FXDeltaNeutralStrategy`,
`from quantark.backtest.transaction_costs import ZeroCostModel`, `from quantark.util.numerical import is_close`; and
`following`, `dated_tarf`, `mc` from `fx_target_redemption_fixtures`):

```python
# ================================================================ Task 10: FX backtest


def _path(index, spots):
    return pd.DataFrame({"EURUSD_spot": spots, "EURUSD_vol": 0.10, "EURUSD_dom_rate": 0.05,
                         "EURUSD_for_rate": 0.03}, index=index)


def _flat_path(days, spot, start=VAL, freq="D"):
    return _path(pd.date_range(start, periods=days, freq=freq), spot)


def _run_backtest(pf, path):
    engine = FXBacktestEngine(FXBacktestConfig(
        portfolio=pf, market_path=path,
        strategy=FXDeltaNeutralStrategy(delta_threshold=1e12, rebalance_frequency="on_threshold"),
        transaction_cost_model=ZeroCostModel(), calculate_greeks=False))
    return engine, engine.run()


def _backtest(pf, path):
    return _run_backtest(pf, path)[1]


def _spy_advance(monkeypatch, keep=lambda day: True):
    """Capture ``(RunFixingDay, {position id: (product, market value)})`` after each recorder advance of the run."""
    seen = []
    original = RunFixingRecorder.advance

    def spy(self, portfolio):
        day = original(self, portfolio)
        if keep(day):
            e = portfolio.pricing_environments["EURUSD"]
            seen.append((day, {pid: (p.product, p.get_market_value(e)) for pid, p in portfolio.positions.items()}))
        return day

    monkeypatch.setattr(RunFixingRecorder, "advance", spy)
    return seen


LAG2D = [t + 2 / 365 for t in FIX]


def test_booked_cash_scales_by_the_signed_quantity_and_ignores_other_entries():
    pay = TargetRedemptionEvent(PAYMENT, -0.1, time=0.0)
    fix = TargetRedemptionEvent(FIXING, -0.1, fixing=1.10)
    assert booked_cash(()) == 0.0
    assert booked_cash((("a", -2.0, pay), ("a", -2.0, fix), ("b", 1.0, ("SETTLED", None)))) == 0.0 + -2.0 * -0.1


def test_b1_numeric_tarn_ages_in_the_backtest():
    c0 = note(target=3.0)
    pf, _ = book(c0)
    rows = _backtest(pf, _flat_path(92, 1.20)).rows
    mtm = [r["portfolio_value"] - r["paid_cash"] for r in rows]
    assert mtm[0] != mtm[91]
    tau = 91 / 365
    rebuilt = note(fix=[t - tau for t in FIX[2:]], accrual_factors=c0.accrual_factors[2:], target=3.0,
                   state=TargetRedemptionState(c0.full_coupon(0) + c0.full_coupon(1)))
    e91 = dataclasses.replace(env(), valuation_date=VAL + timedelta(days=91))
    assert is_close(mtm[91], FxTargetRedemptionNoteMCEngine(mc(n=2_000)).price(rebuilt, e91), rel_tol=1e-12)
    assert rows[91]["paid_cash"] == c0.full_coupon(0) + c0.full_coupon(1)        # gap fixings, flat path: 1.20


def test_b2_redeeming_path_books_cash_on_its_pay_days():
    c0 = note(target=1.5, pay_times=LAG2D)
    pf, _ = book(c0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TerminalBookingWarning)
        rows = _backtest(pf, _flat_path(70, 1.20)).rows
    c_first = c0.full_coupon(0)
    c_second = min(c0.full_coupon(1), 1.5 - c_first)
    paid = [r["paid_cash"] for r in rows]
    assert paid[32] == 0.0 and paid[33] == c_first and paid[62] == c_first
    assert paid[63] == c_first + (c_second + 100.0)
    e = env()
    for d in (61, 62):  # terminal from the second fixing's recording day: value = sum DF x pending, no simulation
        t = LAG2D[1] - d / 365
        expected = 0.0 + c_second * e.get_domestic_df(t) + 100.0 * e.get_domestic_df(t)
        assert is_close(rows[d]["pending_pv"], expected, rel_tol=1e-12)
        assert is_close(rows[d]["portfolio_value"] - paid[d], rows[d]["pending_pv"], rel_tol=1e-12)
    for d in range(63, 70):
        assert rows[d]["pending_pv"] == 0.0 and rows[d]["portfolio_value"] == paid[d]


def test_b3_quiet_path_books_the_principal_and_negative_tarf_cash():
    tarn = note(target=3.0, pay_times=LAG2D)
    pf, _ = book(tarn)
    rows = _backtest(pf, _flat_path(370, 1.05)).rows
    paid = [r["paid_cash"] for r in rows]
    # the last fixing (t = 1.0, day 365) is live on day 365 and recorded on day 366 at 1.05; its principal is due on
    # day 367 (1 + 2/365 years: still in the value that day, Task 8a) and booked on day 368
    assert paid[366] == 0.0 and paid[367] == 0.0 and rows[367]["pending_pv"] == 100.0
    assert paid[368] == 100.0 and rows[-1]["portfolio_value"] == 100.0
    f = tarf(target=0.20, gearing=2.0, pay_times=LAG2D)
    pf, _ = book(f)
    rows = _backtest(pf, _flat_path(370, 1.05)).rows
    paid = [r["paid_cash"] for r in rows]
    steps = [b - a for a, b in zip([0.0] + paid[:-1], paid)]
    assert sum(1 for s in steps if s < 0.0) == 12 and all(s <= 0.0 for s in steps)
    diff = 1.0 * (1.05 - 1.15)
    cf = 0.0 - 2.0 * (1.0 * max(-diff, 0.0))
    expected = 0.0
    for _ in range(12):
        expected = expected + (0.0 + 1.0 * cf)
    assert paid[-1] == expected


def test_b4_twins_in_one_backtest(monkeypatch):
    dn = dated_note(target=1.5, pay_dates=LAG2)
    dt = dated_tarf(target=0.075, gearing=2.0, pay_dates=LAG2)
    nn, nt = _numeric_twin(dn, env()), _numeric_twin(dt, env())
    pf, (id_dn, id_nn, id_dt, id_nt) = book(dn, nn, dt, nt)
    seen = _spy_advance(monkeypatch)
    _backtest(pf, _flat_path(100, 1.20))
    assert any(day.product_events for day, _ in seen)
    for day, values in seen:
        for a, b in ((id_dn, id_nn), (id_dt, id_nt)):
            ev = {pid: [(x.event_type, x.amount, x.fixing, x.redeemed) for p, _q, x in day.product_events if p == pid]
                  for pid in (a, b)}
            assert ev[a] == ev[b]
            assert _rows(day, a) == _rows(day, b)
            assert booked_cash([x for x in day.product_events if x[0] == a]) == booked_cash(
                [x for x in day.product_events if x[0] == b])
            assert is_close(values[a][1], values[b][1], rel_tol=1e-12)


def test_b5_backtest_registration_rules():
    early = note(fix=[-10 / 365] + FIX[1:], accrual_start=-40 / 365, target=50.0)
    pf, _ = book(early)
    with pytest.warns(MissingFixingWarning) as rec:
        rows = _backtest(pf, _flat_path(5, 1.20)).rows
    assert len([w for w in rec if issubclass(w.category, MissingFixingWarning)]) == 1
    assert all(r["paid_cash"] == 0.0 for r in rows)                             # paid before the baseline
    gone = note(fix=[-60 / 365, -30 / 365], accrual_start=-90 / 365)
    pf, _ = book(gone)
    with pytest.warns(MaturedBeforeRunWarning):
        rows = _backtest(pf, _flat_path(5, 1.20)).rows
    assert all(r["portfolio_value"] == 0.0 and r["paid_cash"] == 0.0 for r in rows)


def test_b5_backwards_path_raises():
    pf, _ = book(note(target=3.0))
    idx = pd.DatetimeIndex([VAL, VAL + timedelta(days=2), VAL + timedelta(days=1)])
    with pytest.raises(ValidationError, match="cannot go back"):                  # W6 R9
        _backtest(pf, _path(idx, 1.20))


def test_business_day_path_records_a_saturday_fixing_at_fridays_spot():
    """Review Focus 2 (W6 D5): a Saturday fixing on a weekday path under a CALENDAR_DAYS environment is never live
    (t = 1/365 on Friday); it is recorded on Monday at FRIDAY's spot, gap=True, no warning, in both forms."""
    saturday = datetime(2026, 8, 15)
    dn = dated_note(fixing_dates=[datetime(2026, 7, 15), saturday, datetime(2026, 9, 15)], target=50.0)
    nn = _numeric_twin(dn, env())
    pf, (id_d, id_n) = book(dn, nn)
    index = pd.bdate_range(VAL, periods=70)
    spots = [1.20 + 0.001 * k for k in range(len(index))]
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        engine, _ = _run_backtest(pf, _path(index, spots))
    friday_spot = spots[list(index).index(pd.Timestamp(2026, 8, 14))]
    monday = pd.Timestamp(following(saturday))
    got = [(day.date, f.position_id, f.outcome, f.spot, f.gap) for day in engine.run_fixings.days
           for f in day.fixings if f.index == 1]
    assert got == [(monday, id_d, friday_spot, friday_spot, True), (monday, id_n, friday_spot, friday_spot, True)]


def test_w8_twins_on_the_last_fixing_day_and_after_the_end(monkeypatch):
    """W8 (on main): a numeric participant aged to its maturity is exactly 0.0 and equals its dated twin. A TARN stays
    in the run after its end (Ruling R19): PV(pending) through its last pay date (the cash due that day included,
    Task 8a), then exactly 0.0."""
    dn = dated_note(target=3.0, pay_dates=LAG2)
    nn = _numeric_twin(dn, env())
    pf, (id_d, id_n) = book(dn, nn)
    last = pd.Timestamp(FIX_DATES[-1])                                            # day 365
    seen = _spy_advance(monkeypatch, keep=lambda day: day.date >= last - pd.Timedelta(days=1))
    rows = _backtest(pf, _flat_path(370, 1.05)).rows
    on = {day.date: values for day, values in seen}
    numeric_on_last, value_n = on[last][id_n]
    assert numeric_on_last.maturity == 0.0 and numeric_on_last.fixing_times == [0.0]
    assert is_close(on[last][id_d][1], value_n, rel_tol=1e-12)
    due = on[pd.Timestamp(LAG2[-1])]                                              # last pay date: cash due today
    assert due[id_d][1] == due[id_n][1] == 100.0
    after = on[pd.Timestamp(LAG2[-1]) + pd.Timedelta(days=1)]
    assert after[id_d][1] == after[id_n][1] == 0.0
    assert rows[-1]["portfolio_value"] == rows[-1]["paid_cash"] == 0.0 + 1.0 * 100.0 + 1.0 * 100.0


def test_zero_lag_fixing_day_value_keeps_the_coupon_no_dip():
    """OQ-1 (b), user 2026-10-10 (Task 8a): on its fixing day a zero-lag dated TARN keeps that day's coupon in its
    value (pending, due today); the run books it the next day, when the fixing is recorded. No dip."""
    dn = dated_note(target=3.0)                                                   # pay dates = fixing dates
    pf, _ = book(dn)
    rows = _backtest(pf, _flat_path(33, 1.20)).rows
    e30 = env(val=FIX_DATES[0])                                                   # day 30
    banked = copy.deepcopy(dn)
    banked.record_passed_fixings(e30)
    coupon = banked.state.pending[0].amount
    assert rows[30]["paid_cash"] == 0.0 and rows[30]["pending_pv"] == coupon
    assert rows[31]["paid_cash"] == coupon and rows[31]["pending_pv"] == 0.0
    assert is_close(rows[30]["portfolio_value"], engine_for(banked, n=2_000).price(banked, e30), rel_tol=1e-12)
    for d in (30, 31):                                     # flat market: carry only, never a coupon-sized step
        assert abs(rows[d]["portfolio_value"] - rows[d - 1]["portfolio_value"]) < 0.1 * coupon


def test_business_day_clock_weekend_fixing_twins_in_a_backtest(monkeypatch):
    """OQ-2 (b), user 2026-10-10 (Task 8b): on a BUSINESS_DAYS environment and a weekday path the Saturday fixing is
    live on the Friday in both forms (W6 Ruling 2), the run does not stop, Monday records it at Friday's spot with
    gap=False, and the twins agree every day."""
    saturday = datetime(2026, 8, 15)
    dn = dated_note(fixing_dates=[datetime(2026, 7, 15), saturday, datetime(2026, 9, 15)], target=50.0)
    nn = _numeric_twin(dn, env(**BUS))
    pf, (id_d, id_n) = book(dn, nn, fx_env=env(**BUS))
    seen = _spy_advance(monkeypatch)
    index = pd.bdate_range(VAL, periods=70)
    spots = [1.20 + 0.001 * k for k in range(len(index))]
    engine, _ = _run_backtest(pf, _path(index, spots))
    friday_spot = spots[list(index).index(pd.Timestamp(2026, 8, 14))]
    monday = pd.Timestamp(following(saturday))
    got = [(day.date, f.position_id, f.outcome, f.spot, f.gap) for day in engine.run_fixings.days
           for f in day.fixings if f.index == 1]
    assert got == [(monday, id_d, friday_spot, friday_spot, False), (monday, id_n, friday_spot, friday_spot, False)]
    for _day, values in seen:
        assert is_close(values[id_d][1], values[id_n][1], rel_tol=1e-12)


def test_b6_book_without_target_redemption_books_nothing():
    from quantark.portfolio.fx import FXPortfolio
    vanilla = FxVanillaOption(strike=1.2, option_type=OptionType.CALL, maturity=1.0, notional_foreign=1.0,
                              currency_pair=PAIR)
    pf = FXPortfolio(portfolio_name="b6", pricing_environments={"EURUSD": env()})
    pf.add_position(product=vanilla, quantity=1.0, entry_price=0.0, underlying="EURUSD",
                    engine=GarmanKohlhagenEngine())
    res = _backtest(pf, _flat_path(20, 1.20))
    assert all(r["paid_cash"] == 0.0 and r["pending_pv"] == 0.0 for r in res.rows)
```

- [ ] **Step 2: Run to verify they fail**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_runs.py -k "booked_cash or b1 or b2 or b3 or b4 or b5 or business_day or w8 or zero_lag or b6_book" -n 2 -q`
Expected: collection error `ImportError: cannot import name 'booked_cash'`; after adding only the helper, FAIL with
`KeyError: 'paid_cash'`.

- [ ] **Step 3: Implement**

Append to `quantark/asset/fx/product/option/target_redemption_state.py` (add `Any`, `Iterable` to its `typing` import):

```python
def booked_cash(product_events: Iterable[Tuple[str, float, Any]]) -> float:
    """Quantity-scaled cash of the PAYMENT events among one run step's ``(position_id, quantity, event)`` entries
    (``quantark.util.run_fixings.RunFixingDay.product_events``), summed in order from 0.0. Any other entry (a FIXING
    event, another product's event) is ignored."""
    total = 0.0
    for _position_id, quantity, event in product_events:
        if isinstance(event, TargetRedemptionEvent) and event.event_type is TargetRedemptionEventType.PAYMENT:
            total += quantity * event.amount
    return total
```

In `quantark/backtest/fx/engine.py` (import `from quantark.asset.fx.product.option.target_redemption_state import
booked_cash`):

```python
        cum_hedge_pnl = 0.0
        cum_costs = 0.0
        cum_paid = 0.0   # TARN / TARF cash paid inside the run (spec 2026-10-09 fx-tarn 7.2): quantity-scaled, cumulative
        num_hedges = 0
        rows = []

        for ts, row in path.iterrows():
            self._apply_row(portfolio, pairs, row, ts)
            day = self.run_fixings.advance(portfolio)
            cum_paid += booked_cash(day.product_events)
            ...                                                       # hedge marking unchanged
            value = portfolio.get_portfolio_value() + cum_paid
            portfolio_pnl = value - initial_value
            ...                                                       # Greeks / hedging unchanged
            rows.append({
                ...existing keys unchanged...,
                "paid_cash": cum_paid,
                "pending_pv": day.pending_pv,
            })

        return FXBacktestResults(
            rows=rows,
            initial_value=initial_value,
            final_value=portfolio.get_portfolio_value() + cum_paid,
            ...
        )
```
(`initial_value` stays `portfolio.get_portfolio_value()` right after `register`: cash paid by registration is before
the baseline and never booked.) Module docstring, append: "TARN / TARF positions age with the run on the run-fixings
recorder (their fixings recorded and banked, determined cash booked into `paid_cash` on its pay date);
`portfolio_value` includes cumulative paid cash and each row carries the pending cash's PV (`pending_pv`)."

Append to the CHANGELOG `### Added`:

```markdown
- `target_redemption_state.booked_cash(product_events)`: the quantity-scaled PAYMENT cash of one run step's
  `RunFixingDay.product_events`.
```

and to `### Changed`:

```markdown
- `FXBacktestEngine` books TARN / TARF cash: rows gain `paid_cash` (cumulative, quantity-scaled) and `pending_pv`;
  `portfolio_value` and `final_value` include paid cash. A book without TARN / TARF is unchanged (cash columns 0.0).
```

- [ ] **Step 4: Run to verify they pass**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_runs.py test/test_fx_backtest.py test/test_fx_integration_demo.py test/test_run_fixings_runs.py -n 2 -q`
Expected: all pass; `test/test_fx_backtest.py` and `test/test_run_fixings_runs.py` unmodified.

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/asset/fx/product/option/target_redemption_state.py quantark/backtest/fx/engine.py test/test_fx_target_redemption_runs.py CHANGELOG.md
git -C <WT> commit -m "feat(fx-backtest): TARN/TARF cash from the run-fixings recorder; paid cash and pending PV columns

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: FX dynamic scenario books TARN / TARF cash on W6's recorder (Q8, snapshots, B4 dynamic, B6 invariants)

Re-planned 2026-10-10 against W6. Spec §7.3 (dynamic scenario), §12 Q8, §10 B4/B6; run-fixings spec §3.3 R7 (an
undated run never advances), R9 (the clock cannot go back), §3.5 (the FX dynamic scenario already calls `register` on
the working copy and `advance` on every dated day: `quantark/dynamicscenario/fx/engine.py:84-85, 93-98`), §6 (TARN
keeps its Q8 refusal). This task adds the Q8 refusal, the cash fields and the FIXING snapshots; it builds no wiring.

**Files:**
- Modify: `quantark/dynamicscenario/fx/engine.py:1-132`, `quantark/dynamicscenario/results/dynamic_results.py`
  (`LifecycleEventSnapshot` docstring: add `'FIXING'`), `CHANGELOG.md`
- Test: `test/test_fx_target_redemption_runs.py`

**Interfaces:**
- Consumes: Task 9 (participation), Task 10 (`booked_cash`); `FXDynamicScenarioEngine.run_fixings` (W6).
- Produces: `FXDynamicScenarioEngine.run` raises `ValidationError` naming `DayPath.start_date` when `start_date is None`
  and the input book holds a `TargetRedemptionProduct` (Ruling R18), before cloning or registering; on dated days fills
  `DayResult.paid_cash` / `realized_cash` (cumulative), `pending_receivable_pv` (`RunFixingDay.pending_pv`) and
  `lifecycle_events` (one `LifecycleEventSnapshot(event_type="FIXING", ...)` per FIXING event of the step:
  `spot` = the fixing rate (the recorded spot), `payoff` = the cash it determined, `cashflow` = quantity × that fixing's
  cash paid in the same step (Ruling R6, matched by label), `terminates_position` = `redeemed`,
  `observation_index=None`); `portfolio_value` and `final_value` include cumulative paid cash. A start date before the
  environment's date is refused by the recorder (W6 R9). Does not replace any participant's product.

- [ ] **Step 1: Write the failing tests**

Append (imports: `from quantark.dynamicscenario import FXDynamicScenarioConfig, FXDynamicScenarioEngine`,
`from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange`,
`from quantark.stresstest.stress.stress_types import StressType`):

```python
# ================================================================ Task 11: FX dynamic scenario


def _held_path(days, spot, start=VAL):
    steps = [DayStep(day_index=i, changes=[ParameterChange("spot", StressType.VALUE, spot)]) for i in range(days)]
    return DayPath(name="held", steps=steps, start_date=start)


def _scenario(pf, path):
    return FXDynamicScenarioEngine(FXDynamicScenarioConfig(calculate_greeks=False)).run(pf, path)


def test_q8_undated_path_with_a_tarn_is_refused():
    pf, _ = book(note(target=3.0))
    with pytest.raises(ValidationError, match="DayPath.start_date"):
        _scenario(pf, _held_path(5, 1.20, start=None))


def test_start_date_before_environment_date_raises():
    pf, _ = book(note(target=3.0))
    with pytest.raises(ValidationError, match="cannot go back"):                  # W6 R9
        _scenario(pf, _held_path(5, 1.20, start=VAL - timedelta(days=1)))


def test_dynamic_tarn_redeems_and_books_cash():
    c0 = note(target=1.5, pay_times=LAG2D)
    pf, (pid,) = book(c0)
    res = _scenario(pf, _held_path(70, 1.20))
    days = res.day_results
    c_first = c0.full_coupon(0)
    c_second = min(c0.full_coupon(1), 1.5 - c_first)
    assert days[33].paid_cash == c_first and days[33].realized_cash == c_first
    assert days[63].paid_cash == c_first + (c_second + 100.0)
    fixings = [(i, s) for i, d in enumerate(days) for s in d.lifecycle_events]
    assert [i for i, _ in fixings] == [31, 61]                                    # k/12 fixings between days (D5)
    first, second = fixings[0][1], fixings[1][1]
    assert (first.event_type, first.spot, first.payoff, first.cashflow, first.terminates_position) == (
        "FIXING", 1.20, c_first, 0.0, False)  # pay lag: nothing paid in the recording step (Ruling R6)
    assert (second.payoff, second.terminates_position, second.position_id) == (c_second, True, pid)
    assert days[61].pending_receivable_pv > 0.0 and days[69].pending_receivable_pv == 0.0
    assert days[69].portfolio_value == days[69].paid_cash
    assert res.final_value == days[69].paid_cash


def test_dynamic_snapshot_cashflow_when_paid_in_the_recording_step():
    c0 = note(target=1.5)                                                         # pay time = fixing time
    pf, _ = book(c0, quantities=[3.0])
    days = _scenario(pf, _held_path(70, 1.20)).day_results
    c_first = c0.full_coupon(0)
    c_second = min(c0.full_coupon(1), 1.5 - c_first)
    (s1,), (s2,) = days[31].lifecycle_events, days[61].lifecycle_events
    assert s1.cashflow == 3.0 * c_first
    assert s2.cashflow == 3.0 * (c_second + 100.0)


def test_b4_twins_in_one_dynamic_scenario():
    dn = dated_note(target=1.5, pay_dates=LAG2)
    dt = dated_tarf(target=0.075, gearing=2.0, pay_dates=LAG2)
    nn, nt = _numeric_twin(dn, env()), _numeric_twin(dt, env())
    pf, (id_dn, id_nn, id_dt, id_nt) = book(dn, nn, dt, nt)
    days = _scenario(pf, _held_path(100, 1.20)).day_results
    assert any(d.lifecycle_events for d in days)
    for d in days:
        by = {}
        for s in d.lifecycle_events:
            by.setdefault(s.position_id, []).append((s.payoff, s.cashflow, s.terminates_position, s.spot))
        assert by.get(id_dn) == by.get(id_nn) and by.get(id_dt) == by.get(id_nt)


def test_business_day_clock_weekend_fixing_twins_in_a_dynamic_scenario():
    """OQ-2 (b), user 2026-10-10 (Task 8b) on a daily dated DayPath (weekend rows carry Friday's spot, user ruling)
    under BUSINESS_DAYS: the Saturday fixing is live Friday to Sunday and recorded on Monday in both forms; snapshots
    identical, and the run does not stop on the Friday."""
    saturday = datetime(2026, 8, 15)
    dn = dated_note(fixing_dates=[datetime(2026, 7, 15), saturday, datetime(2026, 9, 15)], target=50.0)
    nn = _numeric_twin(dn, env(**BUS))
    pf, (id_d, id_n) = book(dn, nn, fx_env=env(**BUS))
    days = _scenario(pf, _held_path(70, 1.20)).day_results
    snaps = {pid: [(d.date, s.payoff, s.cashflow, s.spot, s.terminates_position) for d in days
                   for s in d.lifecycle_events if s.position_id == pid] for pid in (id_d, id_n)}
    assert snaps[id_d] == snaps[id_n]
    assert [x[0] for x in snaps[id_d]] == [datetime(2026, 7, 16), datetime(2026, 8, 17)]


def test_b6_dynamic_book_without_target_redemption_books_nothing():
    from quantark.portfolio.fx import FXPortfolio
    pf = FXPortfolio(portfolio_name="b6", pricing_environments={"EURUSD": env()})
    pf.add_position(product=FxVanillaOption(strike=1.2, option_type=OptionType.CALL, maturity=1.0,
                                            notional_foreign=1.0, currency_pair=PAIR),
                    quantity=1.0, entry_price=0.0, underlying="EURUSD", engine=GarmanKohlhagenEngine())
    for start in (VAL, None):                                                    # Q8 refuses only a TARN / TARF book
        days = _scenario(pf, _held_path(10, 1.20, start=start)).day_results
        assert all(d.paid_cash == 0.0 and d.pending_receivable_pv == 0.0 and d.lifecycle_events == []
                   for d in days)
```

- [ ] **Step 2: Run to verify they fail**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_runs.py -k "q8 or start_date or dynamic" -n 2 -q`
Expected: FAIL — `DID NOT RAISE` on the undated path (W6 R7 runs it frozen); `paid_cash` stays 0.0 and
`lifecycle_events` stays empty. `test_start_date_before_environment_date_raises` already passes (W6 R9).

- [ ] **Step 3: Implement**

In `quantark/dynamicscenario/fx/engine.py` (imports: `from quantark.asset.fx.product.option.target_redemption_base
import TargetRedemptionProduct`, `from quantark.asset.fx.product.option.target_redemption_state import (
TargetRedemptionEvent, TargetRedemptionEventType, booked_cash)`, and `LifecycleEventSnapshot` from
`quantark.dynamicscenario.results.dynamic_results`):

```python
        start_time = time.time()
        if day_path.start_date is None and any(isinstance(p.product, TargetRedemptionProduct)
                                               for p in portfolio.positions.values()):
            raise ValidationError(  # TARN spec 12 Q8 (a): no clock is invented (an undated run never advances, W6 R7)
                "the book holds an FX TARN / TARF, which ages by date: set DayPath.start_date (an undated path "
                "cannot age it)")
        working = self._clone_portfolio(portfolio)
        # Run fixings (...existing comment...)
        self.run_fixings = RunFixingRecorder()
        self.run_fixings.register(working)
        baseline_value = working.get_portfolio_value()
        previous_value = baseline_value
        cum_paid = 0.0   # TARN / TARF cash paid inside the run (spec 2026-10-09 fx-tarn 7.3)
        day_results: List[DayResult] = []

        for day_step in day_path:
            day_date = day_path.get_date_for_day(day_step.day_index)
            self._apply_day_changes(working, day_step)
            snapshots: List[LifecycleEventSnapshot] = []
            pending_pv = 0.0
            if day_date is not None:
                for env in working.pricing_environments.values():
                    env.valuation_date = day_date
                # (existing comment)
                day = self.run_fixings.advance(working)
                cum_paid += booked_cash(day.product_events)
                pending_pv = day.pending_pv
                snapshots = self._fixing_snapshots(working, day, day_date)

            value = working.get_portfolio_value() + cum_paid
            daily_pnl = value - previous_value
            cumulative_pnl = value - baseline_value
            ...                                                    # Greeks unchanged
            day_results.append(DayResult(
                ...existing fields...,
                lifecycle_events=snapshots,
                pending_receivable_pv=pending_pv,
                paid_cash=cum_paid,
                realized_cash=cum_paid,
            ))
            previous_value = value

        final_value = working.get_portfolio_value() + cum_paid
```

and the helper:

```python
    @staticmethod
    def _fixing_snapshots(portfolio, day, day_date) -> List[LifecycleEventSnapshot]:
        """One FIXING snapshot per TARN / TARF fixing banked in the step (``RunFixingDay.product_events``);
        ``cashflow`` = quantity x that fixing's cash paid in the same step (an immediate PAYMENT carries its FIXING's
        label - Ruling R6 of the W4b plan)."""
        fixing, payment = TargetRedemptionEventType.FIXING, TargetRedemptionEventType.PAYMENT
        out: List[LifecycleEventSnapshot] = []
        for pid, quantity, ev in day.product_events:
            if not (isinstance(ev, TargetRedemptionEvent) and ev.event_type is fixing):
                continue
            paid = 0.0
            for other_pid, _q, other in day.product_events:
                if (other_pid == pid and isinstance(other, TargetRedemptionEvent) and other.event_type is payment
                        and other.label == ev.label):
                    paid += other.amount
            position = portfolio.positions[pid]
            out.append(LifecycleEventSnapshot(
                position_id=pid, underlying=position.underlying, product_type=type(position.product).__name__,
                event_type="FIXING", date=day_date, observation_index=None, spot=ev.fixing, barrier=None,
                payoff=ev.amount, cashflow=quantity * paid, terminates_position=ev.redeemed,
            ))
        return out
```

In `dynamic_results.py` `LifecycleEventSnapshot` docstring: `event_type: One of 'KO', 'KI', 'COUPON', 'MATURITY',
'EXPIRY', 'FIXING'` (FIXING: an FX TARN / TARF fixing banked; the position stays in the book, valued at its pending
cash, when `terminates_position`).

Append to the CHANGELOG `### Changed`:

```markdown
- `FXDynamicScenarioEngine` books TARN / TARF cash on a dated `DayPath` (determined cash moves to `paid_cash` /
  `realized_cash` on its pay date; one `FIXING` lifecycle snapshot per banked fixing; `pending_receivable_pv` filled).
  An undated `DayPath` with a TARN / TARF in the book raises `ValidationError` (user decision Q8 a). Books without
  TARN / TARF are unchanged.
```

- [ ] **Step 4: Run to verify they pass**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_runs.py test/test_fx_dynamic_scenario.py test/test_fx_integration_demo.py test/test_run_fixings_runs.py -n 2 -q`
Expected: all pass; `test/test_fx_dynamic_scenario.py` and `test/test_run_fixings_runs.py` unmodified.

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/dynamicscenario/fx/engine.py quantark/dynamicscenario/results/dynamic_results.py test/test_fx_target_redemption_runs.py CHANGELOG.md
git -C <WT> commit -m "feat(fx-dynamic): TARN/TARF cash and FIXING snapshots from the run-fixings recorder; undated paths with them are refused (Q8)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Final gate (B6 byte compare, demos, full suite, sign-off list)

Spec §8, §10 B6/C1. Re-planned 2026-10-10: Task 1's `baseline.json` and `demo_*.main.txt` were captured on
2026-10-09 BEFORE this branch was rebased onto W6 (`6f9c6e74`); W6 ages the numeric FX range accruals that the B6 book
holds, so those files no longer describe this branch's base. Step 0 re-captures both on the W6 base (Ruling R20).

**Files:**
- Modify: `docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/moved_values.txt`, `CHANGELOG.md` (review only)
- Output (untracked, never committed): `<WT>/.superpowers/w4b/w6base/` (an export of the base),
  `<WT>/.superpowers/w4b/baseline_w6.json`, `<WT>/.superpowers/w4b/demo_<name>.w6.txt`

**Interfaces:**
- Consumes: everything above; `w4b_baseline.py` (Task 1).
- Produces: a branch ready for the whole-branch review and the user's sign-off.

- [ ] **Step 0: Re-capture the B6 baseline and the demo stdout on the W6 base**

Run:
```bash
BASE=$(git -C <WT> merge-base HEAD main); echo "$BASE"
rm -rf <WT>/.superpowers/w4b/w6base && mkdir -p <WT>/.superpowers/w4b/w6base
git -C <WT> archive "$BASE" quantark example | tar -x -C <WT>/.superpowers/w4b/w6base
env PYTHONPATH=<WT>/.superpowers/w4b/w6base /Users/fuxinyao/quant-ark/.venv/bin/python <WT>/docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/w4b_baseline.py capture <WT>/.superpowers/w4b/baseline_w6.json
for f in fx_tarn_demo fx_backtest_demo fx_dynamic_scenario_demo fx_portfolio_risk_demo; do
  env PYTHONPATH=<WT>/.superpowers/w4b/w6base /Users/fuxinyao/quant-ark/.venv/bin/python <WT>/.superpowers/w4b/w6base/example/$f.py > <WT>/.superpowers/w4b/demo_$f.w6.txt 2>&1; done
```
Expected: `$BASE` is the W6 commit the branch sits on (`6f9c6e74` unless the branch was rebased since; after a
rebase onto a newer main, re-run this step); `captured 4 cases to .../baseline_w6.json`; four demo files written
(the base's `fx_tarn_demo` runs its own, pre-W4b constructors). Diff `baseline_w6.json` against Task 1's
`baseline.json` for the record (expected differences: only values of the dated runs, which hold numeric FX range
accruals that W6 now ages) and note the count in `moved_values.txt` (Step 4).

- [ ] **Step 1: B6 byte gate**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python <WT>/docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/w4b_baseline.py compare <WT>/.superpowers/w4b/baseline_w6.json`
Expected: `B6 OK: every stored number bitwise equal; new cash columns 0.0`. Any diff: STOP, report (a book without
TARN/TARF must not move).

- [ ] **Step 2: Demo stdout byte compare (C1)**

Run:
```bash
for f in fx_tarn_demo fx_backtest_demo fx_dynamic_scenario_demo fx_portfolio_risk_demo; do
  env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python <WT>/example/$f.py > <WT>/.superpowers/w4b/demo_$f.branch.txt 2>&1
  cmp <WT>/.superpowers/w4b/demo_$f.w6.txt <WT>/.superpowers/w4b/demo_$f.branch.txt && echo "$f SAME"; done
```
Expected: four `SAME`.

- [ ] **Step 3: Full suite (controller, `-n 6`)**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test -n 6 -q` then
`git -C <WT> checkout -- example/mo_volmodels/data/mo_barrier_sample.json example/mo_volmodels/data/mo_calibration_explainer_sample.html`.
Expected: pass except the known pre-existing `mo_volmodels` / `otc` failures on main (memory: 4 tests since 0.7.0) —
compare the failing node ids against a run of the W6 base; any other failure blocks. A `mo` smoke "fingerprint
drift" is a shared-artifact race: re-run that test with `-n0` before calling it a defect. `git -C <WT> diff
$(git -C <WT> merge-base HEAD main) --stat -- test/test_run_fixings.py test/test_run_fixings_runs.py` shows only the
Task 9 append to `test/test_run_fixings.py`.

- [ ] **Step 4: Complete the sign-off list**

Append to `moved_values.txt`:

```text
#
# Constructor-only edits (no expected value moved): test/test_fx_tarn_note_mc.py::_note,
# test/test_fx_tarf_mc.py::_tarf, test/execution/matrix_fixtures.py fx_tarn / fx_tarf, example/fx_tarn_demo.py
# (demo stdout byte-identical to main).
# test/test_fx_tarn_note_mc.py::_det_pv gained acc0=0.0 (default reproduces the old reference).
#
# Behaviour changes with no previously asserted value (spec section 8 "Expected to move"):
# theta within one day of a fixing: exception -> number (M3); construction without state / period start:
# accepted -> refused; pricing after a bare time_shift: exception -> MissingFixingWarning fill (M2); a backtest or
# dated dynamic scenario holding a TARN/TARF: frozen -> aged with paid cash (M7); an undated dynamic scenario holding
# a TARN/TARF: ran frozen -> ValidationError (Q8).
#
# Re-plan 2026-10-10 (W6), run-fixings decisions applied to TARN / TARF (user-approved in the W6 spec):
# - a fixing at t = 0 is live (D1): the Greek freeze no longer records it at the base spot, so delta / gamma of a
#   TARN / TARF valued ON a fixing date move with the coupon condition (previously frozen); no existing test asserted
#   them. MissingFixingWarning for a t = 0 fill: warned -> silent.
# - in runs a fixing is recorded at the spot of the instant it was live; a fixing between two path instants at the
#   PREVIOUS instant's spot, gap=True (D5) - the TARN spec (7.2) had the next instant's.
# - a fixing on the run's first date is live at registration and recorded on the next instant (TARN spec 7.2 banked it
#   at registration); a run instant with a time of day holding a TARN / TARF is refused (D6).
# - numeric FIXING / PAYMENT event labels count from the booking (#<fixings banked since construction + position>);
#   informational only.
# B6 / demo references re-captured on the W6 base (Task 12 Step 0): <N> rows of baseline.json -> baseline_w6.json
# differ, all numeric FX range accruals in the dated runs (W6's own moved values).
# User answers 2026-10-10 (plan section "Re-plan 2026-10-10 against W6"): OQ-1 (b) - cash due today stays in today's
# value (Task 8a block above: pay-date prices + cash, theta the day before + cash, theta on the pay date - cash, the
# terminal value on the last pay date = that cash, validation of pay_time 0.0 / pay_date == as_of); OQ-2 (b) - a dated
# fixing or pay date at zero business time is live (Task 8b block above: ValidationError -> priced like the twin;
# Ruling R7 superseded). In runs: a payment on a path-aligned pay date is booked on the next instant (B3 principal
# day 367 -> 368 against the pre-8a plan text; nothing on main asserted it), and a zero-lag fixing day keeps its
# coupon in the run value (no dip, no false first-day gain).
# OQ-3 (user ruling 2026-10-10): banked fixings tracked per fixing (TargetRedemptionState.fixings_banked), as_of never
# after the valuation date (R24: on a weekend row, the Friday), no check dropped. Moved values: none beyond Tasks 8a /
# 8b; not a value: aged dated states state their count, and a dated state's as_of on a non-business valuation date
# reads the preceding business date.
# R24 user-approved 2026-10-10.
# OQ-4 (b), user 2026-10-10 (Task 8c block above): periods may share a fixing date / time. Contracts with equal fixing
# times or dates, and dated dates resolving to one time valued before it: ValidationError -> priced (= twin). Nothing
# else moves. test_r2_dated_refusals[fixing_dates=FIX_DATES[:1] * 2] still refused (now the default derivation's
# zero-length period; R26 superseded by the user's ruling (b)).
# R26 (b), user 2026-10-10 (Task 8d block above): the TARN's own accrual period dates (accrual_end_dates). Existing
# bookings: none move (no accrual_end_dates = today's derivation, bitwise). New bookings with accrual_end_dates
# (including periods sharing a fixing date without accrual_factors): TypeError (unknown keyword) -> priced (= twin).
# Refused, pending OQ-5 / OQ-6 (non-blocking): an accrual end before its fixing or after its pay date; gapped or
# overlapping periods by dates (accrual_factors books both).
```
Verify the CHANGELOG block has `### Added`, `### Changed`, `### Fixed` with the bullets from Tasks 2–11 and no empty
heading.

- [ ] **Step 5: Commit**

```bash
git -C <WT> add -f docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/moved_values.txt
git -C <WT> add CHANGELOG.md
git -C <WT> commit -m "docs(fx-tarn): moved-values list for sign-off; final gates green

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Landing (controller, after the whole-branch review): the user signs off `moved_values.txt`; the branch is rebased
onto the current `main` (W8/W9 touched no FX or run-fixings library code at `0d1e4133`, but re-run Step 0–3 after
the rebase); only then ff-merge.

---

## Self-review (done while writing)

- **Spec coverage:** §3 → T2/T3; §4.1 → T3/T4; §4.2/§4.3 → T7 (§4.2 period sources amended by T8d, user ruling on R26
  (b); §4.1/§4.2 uniqueness amended by T8c); §5.1/§5.2 → T6/T7; §5.3 → T3/T4/T6/T7; §5.4 → T3/T6/T7;
  §6 → T5/T7/T8 (cash rule amended by T8a); §7.2 → T9 (on W6's recorder, as amended by run-fixings spec §6); §7.3 →
  T10/T11; user OQ-1 (b) → T8a, OQ-2 (b) → T8b; §8 → T3/T4/T12; §10 P1–P5 → T6; D1/D3/D4/D5 → T7, D2 → T9, D4 business → T8;
  B1–B5 → T9/T10/T11, B6 → T1/T10/T11/T12; WARN1 → T3/T4/T6/T7, WARN2 → T6 (P3) and T10 (B2), WARN3 → T6/T7; R1/R2 →
  T3/T4/T7; T1/T2 → T6; C1 → T3/T4/T12; §12 Q7 → T8, Q8 → T11.
- **Type consistency:** `resolve(fx_env, *, warn_missing_fixings=True)`, `pending_pv(fx_env)`, `matured_before(fx_env)`,
  `coupon_amount(factor)`, `_fixing_outcome(factor, spot, accumulated)`, `open_fixings(fx_env)`,
  `with_time_elapsed(dt)`, `with_history_settled(fx_env)`, `booked_cash(product_events)`, W6's
  `RunFixingDay(date, fixings, product_events, pending_pv)` and `RecordedFixing(position_id, index, label, outcome,
  spot, gap)` are used with the same names in every task. (Re-plan 2026-10-10: `FxLifecycleDay` / `FxLifecycleEntry`
  no longer exist.) Task 8d: `accrual_end_dates`, `_accrual_end(index)`, `_dated_factor(index, fx_env)` and
  `_validate_accrual_end_dates()` are named the same in its Rules, Interfaces, tests, CHANGELOG and Review Focus 10.
- **User answers (2026-10-10):** OQ-1 (b) → Task 8a (spec §3 / §5.1 / §6 cash rule amended; pins in 8a, 9, 10);
  OQ-2 (b) → Task 8b (Ruling R7 superseded; flips in 8b, twin pins in 9, 10, 11). OQ-3 ruling → Task 8b rewritten
  (per-fixing banked count; R15, R22 superseded; R24, R25 added) with the Friday + Saturday case in 8b and Task 9.
  Task 9's `_record_dated` replacement carries the 8a cash sweep and 8b's per-fixing loop, count and `_state_date`; no
  task relies on `as_of` as a banked high-water mark, and no task still pins an answered question's old behaviour.
  R24 user-approved. OQ-4 (b) → Task 8c (cites spec §6 "every later fixing is dead" for a period after the hit on the
  same date). R26 superseded by the user's ruling (b) → Task 8d (`accrual_end_dates`, R27 / R28; the default
  derivation's shared-date refusal built in 8c, narrowed in 8d; nothing existing moves). Open, non-blocking: OQ-5
  (an accrual end outside [fixing, pay]) and OQ-6 (non-contiguous periods), both refused by 8d with a recommendation
  to keep the refusal.
- **Re-plan self-check (2026-10-10):** every old Task 9–11 requirement is in the mapping table of "Re-plan 2026-10-10
  against W6"; W6 hand-off notes 1–5 map to Task 9 (protocol, hooks, append-only events, R4/R7/R8/R9/Ruling 9, W8),
  Task 10 (W8 twins) and Task 11 (R7 + Q8); W6 Ruling 2's time-based gap flag drives R16 and OQ-2; the user's weekend
  ruling (weekend rows carry Friday's spot) is why no guard is added for a business-day clock with weekend rows.
