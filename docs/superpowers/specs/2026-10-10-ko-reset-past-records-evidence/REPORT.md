# A dated autocallable with past knock-in records prices like its numeric twin (W11)

Branch `w11-ko-reset-past-records`, from `0d1e4133`. A dated `KnockOutResetSnowballOption` whose schedule-1 records
are in the past was refused on QUAD V2 at any date and on the PDE before maturity, while its numeric twin priced.

## 1. Cause: one shared step, the knock-in resolver

`SnowballOption.resolve_ki_observations` (`quantark/asset/equity/product/option/snowball_option.py`, inherited by
`KnockOutResetSnowballOption`) resolved EVERY knock-in record of the schedule through `ObservationSchedule.resolve`
-> `SettlementResolver.resolve_contingent` -> `_time_from_date` -> `calculate_year_fraction`
(`quantark/util/calendar/day_counter.py:86`), which refuses a record dated before the valuation date: "end_date
(2025-02-04) must be after start_date (...)". The knock-out resolvers (`resolve_ko_observations`, the KO-reset's
`_resolve_ko_schedule`, the Phoenix's `active_ko_source_indices`, spec 3B decision 13) already kept only the records
not before the valuation date, and the numeric twin's `time_shift` drops the passed records, so only the dated
form met it. `PhoenixOption.resolve_ki_observations` is a copy of the same code.

Shared by every engine (tracebacks on `0d1e4133`, KO-reset ABSOLUTE, 100 days in, not knocked in):

| Engine | Call into the resolver |
|---|---|
| QUAD V2 | `v2/adapters.py compile_contract` -> `resolve_ki_observations` (any date, knocked in or not: the compile reads the KI schedule unconditionally) |
| PDE | `KOResetSnowballPDESolver._valuation_decision` -> `decide_observations_at_valuation` (not knocked in); knocked in: `_knock_in_side_row` -> `_get_cached_ki_profile` -> `get_ki_observation_profile` |
| QUAD V1 | `KOResetSnowballQuadEngine.price` -> `decide_observations_at_valuation` (not knocked in only) |
| MC (BS, local vol, Heston, SLV) | `_build_time_grid_ko_reset` -> `get_ki_observation_profile` (not knocked in only) |

So the refusal was not QUAD V2 / PDE only: MC and QUAD V1 refused the dated KO-reset too whenever it was not knocked
in (the state the knocked-in fixtures never met), and QUAD V2 refused it on the exercise date. Nor was it KO-reset
specific: a dated snowball or Phoenix with a DISCRETE knock-in schedule and a past record was refused on every
route (measured on MC, PDE, QUAD V1, QUAD V2: same message). The snowball / Phoenix fixtures that "price fine" have a
continuous knock-in.

## 2. Fix

- `observation_schedule.active_record_indices(schedule, env)`: the positions of the records not at a negative time
  and not dated before the valuation date (the knock-out predicate, now shared). A record AT the valuation instant
  is pending (kept, time 0).
- `SnowballOption.active_ki_source_indices` / `PhoenixOption.active_ki_source_indices`; both
  `resolve_ki_observations` resolve only those records. A passed knock-in record is history, its outcome the carried
  knock-in state (`_otc_lifecycle_knocked_in` / `lifecycle_state.knocked_in`), never resolved forward.
- `SnowballMCEngine._build_time_grid_ko_reset`: the profile index of each kept record is mapped to its source
  position in the priced contract's schedule (`ki_record_indices`, the knock-in anchor's `source_index`); the code
  assumed the profile and the schedule had the same length.
- `quantark/intraday/events.py`: the two knock-in loops index the source records by position; they now refuse a
  schedule env that drops a knock-in record ("schedule env must keep every KI record active (anchor at
  initial_date)"), as the knock-out loops already did. Before the fix such an env raised the year-fraction error.

KO-reset contract rules unchanged: a knock-in replaces schedule 1 by schedule 2 for good; no knock-in test after
schedule 1 ends (its records are all past, nothing is resolved); an ended schedule contributes empty pre-records; a
contract past schedule 1 and not knocked in has matured and `require_alive` refuses it on both forms (pinned). No
new semantics, no TODO. No raw tolerance added; no MC in any PDE path.

## 3. Tests (TDD)

New `test/test_ko_reset_past_records.py` (214 cases). Twins: dated records on `INITIAL + d` for the snapped days
`d = round(k / 12 * 365)` (30, 61, ..., 182 | 213, ..., 365), numeric at `d / 365` aged by
`time_shift(keep_landing=True)`; price (`price`) and price / delta / gamma (`calculate_greeks`) of the dated form
equal the numeric twin's within 1e-11 (measured worst 7.0e-13, 293 of 591 pairs bitwise equal; the two forms reach
the engines as different floats, as `test_tracker_clock_parity.PV_ATOL`).

- KO-reset ABSOLUTE (schedule 1 KO 103 / discrete KI 75, schedule 2 KO 95) on MC, PDE, QUAD V1, QUAD V2, local-vol
  MC, Heston MC, SLV MC: not knocked in on days 30 (KO at 104 decided at the instant), 100 (spots 100, 80), 152
  (KI at 70 decided at the instant), 182 (schedule 1's last record); knocked in on days 100, 152, 182, 183 (the day
  after schedule 1 ends), 213 (a schedule-2 record), 292, 364 and 365 (the exercise date). Not knocked in on days
  183 and 292: both forms refuse with `require_alive`'s message.
- Snowball (KO 103 / discrete KI 75) and Phoenix (KO 103 / coupon 80 / discrete KI 70, no memory) on all twelve
  fixings: MC, PDE, QUAD V1, QUAD V2, local-vol MC and PDE, Heston MC (+ QE for the snowball), SLV MC; days 100,
  152, 292 alive, 292 and 365 knocked in.
- The resolver keeps exactly the records not before the valuation date, with their source positions.

Pinned splits replaced: `test_dated_expiry_settles.py` (`DATED_REFUSED_ANY_DATE` and
`test_numeric_twin_settles_where_the_dated_contract_is_refused_at_any_date` removed: the KO-reset QUAD V2 cells are
in `SETTLING`, dated == numeric == settled value) and `test_expiry_carried_knock_in.py` (`("ko_reset_absolute",
"quad_v2")` out of `AUTOCALL_REFUSED`, `("ko_reset_absolute", "pde")` out of `AUTOCALL_BEFORE_REFUSED`: both forms
pinned on the exercise date, with the flag and the lifecycle state, and the day before).

On `0d1e4133`'s library (git archive) these three files fail 130 cases: 118 in the new file (45 KO-reset twins, 70
snowball / Phoenix twins, 3 resolver), 2 in `test_dated_expiry_settles` (KO-reset QUAD V2 at 104 / 90), 10 in
`test_expiry_carried_knock_in` (3 exercise date flag, 3 lifecycle state, 4 day before). All pass on the branch.

## 4. Moved values

`moved_values.txt` (`measure.py` on both trees, `summarize.py`; raw rows `base_0d1e4133.txt`, `after_branch.txt`).
1332 rows measured; 969 unchanged bitwise; 363 moved, every one a DATED row "raised -> value" (345 section A, the
new test's cases; 6 section B, `test_dated_expiry_settles`' KO-reset on QUAD V2 / PDE at the exercise date and 292
days in; 12 section C, `test_expiry_carried_knock_in`'s KO-reset on QUAD V2 / PDE). No numeric row moved, no value
moved to another value, nothing newly raised. Six moved rows are measured only (the tracker-clock k/12 twin 292
days in is not the dated contract's snapped schedule, so it is pinned through the new file's snapped twin; the
day-before branch test takes spots below the 103 KO).

## 5. Runs (`-n 2`, worktree PYTHONPATH)

1. Every ko_reset / snowball / phoenix / quad / autocallable test file, `test_dated_expiry_settles`,
   `test_expiry_carried_knock_in`, `test_tracker_clock_*`, `test/lifecycle_ko_reset`, `test/accrual_time`: 2721
   passed, 6 skipped (the carried-KI file's `_BEFORE_REFUSED` cells); it ran the new file before its local-vol PDE, QE and
   SLV snowball / Phoenix routes were added (184 of its 214 cases); the final versions: run 4.
2. Lifecycle / replay / `test/replay_golden` / pnlexplain / otc autocallable / settlement timeline,
   `test/theta_boundary`, `test/modelvalidation/{test_anchors,test_banked_certificates}`, `test/pde_bitwise`:
   586 passed, 33 skipped (superseded certificates / empty golden frames, skipped by design, as in W9).
3. `test/intraday`, the SA-CVA snowball / Phoenix / KO-reset exposure files: 606 passed.
4. Final versions of the new file, `test_dated_expiry_settles`, `test_expiry_carried_knock_in`,
   `test_tracker_clock_parity`: 1404 passed, 6 skipped.

No certificate anchor, golden or pde_bitwise row was touched or moved.

## 6. Open questions

- A dated record on the valuation DAY but before a valuation timestamp later that day (intraday valuation) is
  dropped by the shared predicate (`observation_date < valuation_date`), as the knock-out records already were;
  `time_shift` keeps such a record when `keep_landing`. Before this change a knock-in record there raised. The
  intraday paths use numeric twins and an initial-date env, so nothing measured here meets it; worth a ruling if
  dated contracts are ever priced at a time of day on a record date.
- Unchanged and out of scope: the 2D Heston / SLV snowball and Phoenix PDE refuse a dated knock-out schedule before
  maturity (carried-knock-in report section 6), so their dated discrete-KI twins are not in the new test.
