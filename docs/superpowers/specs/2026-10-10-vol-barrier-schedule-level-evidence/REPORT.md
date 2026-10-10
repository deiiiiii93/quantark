# W18 — the vol-model barrier routes price a discrete schedule at its records' terms, first-hit only

Branch `w18-vol-barrier-schedule-level` from 468dc2df (W12). LV / Heston / SLV barrier MC
(`engine/mc/barrier_vol_mc_engines.py`) and Heston / SLV barrier PDE (`engine/pde/barrier_vol_pde_solvers.py`); user
rulings 2026-10-10 on W12's open questions 1 and 2. (Report text by the W18 agent, committed by the controller because
the agent could not write files here.)

## 1. Ruling 1 (a): the records' common level and a knock-out's common payoff

Symptom: with every record at a common level different from `product.barrier`, the kernels were still given
`product.barrier` (`_barrier_kwargs` MC, `_bkw` PDE), silently, in both forms. Measured on 468dc2df: up-out call booked
at 120 with every record at 115 priced exactly as the 120 contract (Heston MC 2.9469 vs 2.2992 for the contract booked
at 115). Same defect for the payoff: `product.rebate` was passed (`_barrier_kwargs`, PDE `_apply_participation`).
Knock-outs now take the records' common payoff, as BS barrier MC pays `payoffs[first_idx]`; knock-ins keep
`product.rebate` (paid when nothing hits — BS MC `_discounted_payoffs_discrete` pays `product.rebate*df_T`), unchanged
and pinned.

Fix: `vol_barrier_schedule.schedule_barrier_terms(product)` — level = records' common level (a record with no level
falls back to `product.barrier`, the product's own normalisation); rebate = knock-out: records' common payoff (unset ->
`product.rebate`), knock-in: `product.rebate`; no schedule: the product's own terms. Plain contracts: kernel inputs
byte-identical.

Paths that bypass the kernels: PDE deterministic limit (1D core reads each record) already right, pinned within 2e-3
of BS PDE; PDE exercise instant (`barrier_expiry_outcome`) already right, pinned; MC exercise instant
(`barrier_expiry_value`, BS MC, reads `product.barrier`) NOT changed — open question 1; LV MC ended monitoring /
carried knock-in: no barrier involved.

## 2. Ruling 2 (a): first-hit only

`aggregation_mode` was never read (468dc2df gives identical values for STOP_FIRST_HIT / ACCUMULATE / BEST / WORST).
`refuse_non_first_hit()` runs first on every path (price, greeks, lifecycle fixed PV, exercise instant, deterministic
limit), from `_check_barrier` (MC) and `_check` (PDE): `ValidationError("<engine>: the vol-model barrier kernels
monitor a discrete schedule first-hit (stop-first-hit) only; aggregation <mode> is refused")`,
`# TODO(vol-aggregation)`. CONTINUOUS / EXPIRY unaffected. In the deterministic limit ACCUMULATE used to price (11.9801136153051,
the 1D core supports it) and now raises; BEST / WORST were refused there by the core (`PricingError`) and now by the
route (`ValidationError`).

## 3. Tests

`test/test_vol_barrier_schedule_level.py`, 298 cases, both forms aged to days 10 and 92: up-out / up-in calls (120,
records at 115), down-out / down-in puts (85, records at 90), pay-at-hit up-out; rebate 3; 5 routes. Level / payoff /
greeks cells bitwise equal to the contract booked at the record terms (exact `==`); aggregation refusals on every route
and the deterministic limit; BSM limits (deterministic limit vs BS PDE; PDE exercise date vs BS PDE and hand values 5 and
12). 231 of 298 failed before the fix; 298 pass after.

## 4. Moved values (`moved_values.txt`: 446 measured, 276 moved, 0 unexpected)

| Group | Moved | Movement |
|---|---|---|
| level (records at L != product.barrier) | 100/100 | value -> value (LV MC up-out d10 2.5460128720856594 -> 2.078777615280778) |
| terms (records at L paying 4.5) | 50/50 | value -> value |
| payoff (records at product.barrier paying 4.5) | 30/50 | knock-outs value -> value; 20 knock-ins unmoved |
| greeks (up-out, level L, day 90) | 30/30 | value -> value |
| agg (ACCUMULATE / BEST / WORST) | 60/60 | value -> refused |
| agglimit (PDE deterministic limit) | 6/6 | ACCUMULATE value -> refused; BEST / WORST PricingError -> ValidationError |
| plain (records at product.barrier, STOP_FIRST_HIT) | 0/100 | bitwise |
| nolevel (records with no level) | 0/50 | bitwise |

## 5. Numeric schedules at product.barrier did not move

PDE byte book (`capture_book.py --workers 2`, 468dc2df export vs branch, back to back): 713 rows, 0 unexpected diffs,
0 manifest violations (10 rows error identically on both sides). Anchors / banked certificates / pde_bitwise: 39
passed, 29 skipped. W12's `measure.py`: 438 values byte-identical. W18 measure: 150/150 plain and no-level cells
unmoved.

## 6. Runs (`-n 2`)

Barrier / vol suites (barrier vol MC/PDE, participation-expiry, family settlement, `test_vol_dated_schedules`,
carried-KI, dated-expiry, vol term structure, theta_boundary monitoring_ended, `test/volmodels`, BS barrier MC,
`test/execution`): 2231 passed, 2 skipped (BS MC `down_in_put` day-before, skipped before W18). New file 298 passed.
modelvalidation + pde_bitwise 39 passed, 29 skipped.

## 7. Open questions

1. MC routes on the exercise date: at T = 0 the vol MC routes return BS MC's `barrier_expiry_value` (W9), which tests
   the spot against `product.barrier`, not the maturity record's level. Up-out call barrier 120, one record at 115 on
   the exercise date, spot 117: vol MC and BS MC 17.0; BS PDE, vol PDE and the contract booked at 115 give 5.0. The BS
   MC also decides "hit at valuation" from `product.barrier` (`is_barrier_hit(S)`) for a discrete schedule. The fix
   belongs in the BS MC (outside W18); not changed, MC value not pinned.
2. ACCUMULATE in the PDE deterministic limit priced correctly via the 1D core and is now refused route-wide: keep, or
   let the limit price it?
3. A knock-in whose records carry a payoff != product.rebate: vol routes and BS MC pay `product.rebate`; the BS / LV PDE
   core refuses it (spec 3C §5.2(a)), so Heston / SLV PDE refuses it in the deterministic limit but prices it in 2D
   (pre-existing). Refuse on the vol routes too?

# W18b (user approval / ruling 2026-10-10): open questions 1 and 3

Rebased first: `git tag pre-rebase-w18 HEAD && git rebase --onto w12-vol-dated-schedules 468dc2df` (W12 head
a514c88d on main c02296cb). No conflicts; `range-diff` shows the three W18 commits unchanged (fda931b9, 4acd2e6c,
ee008e9a). W18 tests re-run: 487 passed; W18's 446 measured values byte-identical to `after_branch.txt`.

## 8. The Black-Scholes barrier MC decides at the valuation instant on the record's level (question 1)

Two decisions of `BarrierOptionMCEngine` read `product.barrier` / `product.rebate` while its discrete path reads each
resolved record (`_build_discrete_grid`, `payoffs[first_idx]`): (a) the exercise-date value `barrier_expiry_value`,
also the LV / Heston / SLV barrier MC routes' value at T = 0 since W9; (b) the "hit at valuation" decision before
simulating. New `valuation_barrier_terms(product, env)`: for a DISCRETE schedule, the first record at or after the
valuation instant (the record on the valuation date; otherwise the next observation, the barrier PDE's
`_next_observation` rule) gives the level (no level: `product.barrier`) and the knock-out cash (no payoff:
`product.rebate`). A dated record behind the valuation date is skipped without being resolved (the year fraction
refuses it; a dated monthly schedule on its exercise date keeps pricing); one on the valuation instant is at 0.0, as
the settlement resolver does. CONTINUOUS / EXPIRY monitoring and a knock-in's no-hit rebate are unchanged.
`barrier_expiry_value` now takes the environment (both callers updated).

Tests (`test/test_barrier_mc_record_level_at_valuation.py`, 281; 144 failed before the fix):
- exercise date, BS MC + LV / Heston / SLV MC, both forms, one-record and monthly schedules, up/down x in/out, spots
  between the levels / beyond both / inside both: equal to the hand value and bitwise to the contract booked at the
  record level; the one-record schedule equals the BS PDE (the 120/115 example, spot 117: 17.0 -> 5.0 on every MC
  route = BS PDE = vol PDE = the contract booked at 115);
- a record on a mid-life valuation date (day 91), spot between the levels: knock-outs equal the BS PDE (1e-12),
  knock-ins the PDE within MC error and bitwise the contract booked at the level;
- day 92 (no record today): the next record decides, bitwise the contract booked at its level;
- a record without a level and CONTINUOUS monitoring read `product.barrier`.

## 9. A knock-in record paying other than its rebate is refused on every route (question 3)

The 1D PDE core's rule and message (`_KnockInAdapter.validate`, spec 3C §5.2 (a): "a knock-in has one rebate (R); a
schedule record pays P (spec 3C §5.2 (a))") move to the neutral `engine/barrier_schedule_rules.
refuse_knock_in_record_payoff`, called by the core (unchanged behaviour), the BS MC, the QUAD barrier
(`DiscreteQuadEngine.price`, `BarrierOption` only), the LV / Heston / SLV MC (`_check_barrier`) and the Heston / SLV
PDE (`_check`, before the 2D kernel and the deterministic limit). Both forms; every path of each route (the MC
exercise date included). Knock-ins whose records pay the rebate or nothing, and all knock-outs, are unaffected.

Tests (`test/test_knock_in_record_payoff_refused.py`, 141; 36 failed before: the 7 routes off the core, and the 4 MC
routes on the exercise date; the BS PDE, LV PDE and the Heston / SLV deterministic limit already refused). The W18 /
W18b knock-in record-payoff cells (`test_a_common_record_payoff_and_level_are_the_terms_priced`,
`test_the_black_scholes_mc_reads_the_record_terms_the_same_way`, `test_the_exercise_date_knock_out_pays_the_record_
payoff`) now expect the refusal.

## 10. W18b moved values (`moved_values.txt`, sections W18b)

Record level (`measure_w18b.py`, ee008e9a vs 822b9433): 600 values, 128 moved, 0 unexpected; 264 of 264 plain /
reference values unmoved.

| Group | Moved | Movement |
|---|---|---|
| expiry (exercise date, records at L, spot between the levels) | 64 | value -> value (e.g. bs_mc up-out 17.0 -> 5.0; up-in 5.0 -> 17.0) |
| payoff (exercise date, records at L paying 4.5) | 48 | value -> value (between: decision; knock-out beyond: 5.0 -> 4.5) |
| mid (day 91, record today, spot between) | 8 | raised -> value (the MC simulated and failed on the zero-length step; now KO 4.888655733104503 = PDE) |
| next (day 92, spot between) | 8 | value -> value (e.g. up-out 4.573221020284145 -> 4.8890575569381145) |
| expiry0 / mid0 / next0 / midpde (plain, BS PDE) | 0 of 264 | unmoved |

Knock-in refusal (`measure_w18b_ki.py`, plus the W18 and W18b measures re-run; 822b9433 vs d1e8ade3): 1222 values,
116 moved, 0 unexpected, all price -> "a knock-in has one rebate" refusal:

| Cells | Moved |
|---|---|
| kipay (KI, records pay 4.5, rebate 3, day 10): bs_mc, quad, lv_mc, heston_mc, slv_mc, heston_pde, slv_pde | 28 (4 each) |
| kipay on bs_pde, lv_pde, heston/slv deterministic limit | 0 (already refused, same message) |
| W18 `terms` / `payoff` KI cells (LV / Heston / SLV MC, Heston / SLV PDE) | 40 |
| W18b `payoff` KI cells (BS MC, LV / Heston / SLV MC, exercise date) | 48 |
| kireb / kinone / kopay (KI paying the rebate or nothing, KO paying 4.5) | 0 |

## 11. Nothing else moved

- PDE byte book (`capture_book.py --workers 2`), ee008e9a vs 822b9433 and 822b9433 vs d1e8ade3: 713 rows, 0
  unexpected diffs, 0 manifest violations each (10 rows error identically on both sides).
- modelvalidation `test_anchors` + `test_banked_certificates` + `test/pde_bitwise`: 39 passed, 29 skipped (after both
  W18b commits). No anchor or golden touched.
- Suites (`-n 2`): barrier vol MC / PDE, participation-expiry, family / numerical settlement, BS barrier MC, QUAD
  barrier / adapters / one-touch, double barrier, carried-KI, dated-expiry, W12 / W18 / W18b files, euro MC lower
  bound, MC term structure, PDE/QUAD audit regressions, theta_boundary, intraday MC / analytical barrier routes,
  `test/execution`, `test/volmodels`, `test/pde_core`: 2970 passed.

## 12. Notes

1. A record on the valuation instant that the spot has NOT crossed still fails in the BS MC path builder
   ("dt_array must have strictly positive entries": a zero-length first step), unmoved and pre-existing (the 'inside'
   mid cells); out of W18b.
2. Without a record today, the BS MC's hit-at-valuation shortcut (knock out / in at once) is its own pre-existing
   convention; W18b only changed the level it reads (the next record's). The PDE evolves to the next observation
   instead.
3. Other engines read `product.is_barrier_hit(S)` for discrete schedules (barrier analytical, QUAD adapters' expiry /
   early paths); not in W18b's scope.
