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
