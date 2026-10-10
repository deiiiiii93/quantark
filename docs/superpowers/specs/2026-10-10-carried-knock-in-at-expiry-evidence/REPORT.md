# A knock-in carried as triggered decides the exercise date

User ruling 2026-10-10 (W9 open question Q-W9-1): "fix it, should check knock in state first. autocallables with ki
term may have the same issue, please check". Branch `w9b-carried-ki`, from W9 07c799cf.

## 1. Cause

On the exercise date (T = 0: a dated contract on its exercise instant, or its numeric twin aged to
`maturity == 0.0`) the barrier engines skip the carried-knock-in branch, which every one of them gates on
`get_maturity(env) > 0.0` (analytical / QUAD: the remaining-contract replication or the vanilla; MC: the knocked-in
vanilla path; PDE: the knocked-in composite, built only before maturity). The T = 0 code that runs instead decided the
knock-in from the valuation instant alone (`hit` at the spot, or a schedule record at t = 0) and paid the rebate
otherwise. A contract carried as knocked in therefore priced the knocked-in vanilla the day before and the rebate on
its exercise date. Up-and-in call K100 B130 rebate 5 at spot 110: 10.0 (T = 1/365) -> 5.0 (T = 0), must be 10.0.

Carriers of a knock-in (searched in `quantark/`): `lifecycle_state.knocked_in` and the product attribute
`_otc_lifecycle_knocked_in`; `monitoring_ended.knock_in_carried(product, lifecycle_state)` reads both (only for a
product with `is_knock_in`). Barrier schedules record no observed hits (a past observation's outcome is the carried
state), so there is no third carrier. The DCN carries `knocked_in_at_valuation` (section 3).

## 2. The fix: "carried OR hit today", in the shared expiry code

The T = 0 knock-in decision is `knock_in_carried(...)` first, then today's observation, the helper the T > 0 path
already reads. No other semantics: knocked in pays the payoff (participation, settlement discount as before), a
contract never knocked in pays the rebate.

| Shared code | Routes that call it | Carrier read |
|---|---|---|
| `BarrierAnalyticalEngine._price_at_maturity(..., knocked_in=)` | BS barrier analytical | flag + `lifecycle_state` |
| `barrier_option_mc_engine.barrier_expiry_value(..., knocked_in=)` | BS barrier MC; local-vol / Heston / SLV barrier MC (`_settled_at_exercise`) | flag + `lifecycle_state` |
| `_KnockInAdapter.maturity_value(..., knocked_in=)`, passed by `_BarrierFamilyAdapter.terminal_outcome` | BS barrier PDE, local-vol barrier PDE, Heston / SLV barrier PDE (`barrier_expiry_outcome`, price and greeks), double barrier PDE | flag (+ `lifecycle_state`, which these solvers refuse) |
| `quad_adapters` `_barrier_expired_price` / `_double_barrier_expired_price` | QUAD barrier, QUAD double barrier | flag (QUAD takes no lifecycle state) |
| `DoubleBarrierOptionAnalyticalEngine._price_zero_maturity(..., knocked_in=)` | double barrier analytical | flag + `lifecycle_state` |

No new raw tolerance; no MC in any PDE path; certificate anchors and goldens untouched.

## 3. Every route checked

Measured on 07c799cf and the branch (`measure.py`; `today_07c799cf.txt`, `after_branch.txt`), both forms; pinned by
`test/test_expiry_carried_knock_in.py` (hand settled values; day-before continuity: within one day's spot standard
deviation of the knocked-in settled value, the never-knocked-in value lying further).

**Part 1, barriers.** Up-and-in call B130 (110, 90, 135), down-and-in put B80 (90, 110, 75), up-and-in call
discrete on an exercise-date record (110, 135), double knock-in call 80/130 (110, 90, 135); carried by the flag on
every route and by `lifecycle_state` where the engine takes one; not carried; PDE greeks.

| Route | Before | After |
|---|---|---|
| BS analytical, MC, PDE, QUAD; LV MC, LV PDE; Heston MC / PDE; SLV MC / PDE (single barrier) | carried: rebate | FIXED: payoff |
| Double barrier analytical, PDE, QUAD (knock-in) | carried: rebate | FIXED: payoff |
| Same routes, not carried | today's observation | unchanged |
| Single / double knock-out, single / double sharkfin, one-touch, double one-touch | carried knock-out / touch = terminal `lifecycle_state`, read by `terminal_lifecycle_pv` before maturity is read, at T = 0 and T > 0 alike (measured: the same branch on both days on every route taking a lifecycle state); no product-level knock-out / touch carrier exists | already correct, not the same defect |

**Part 2, autocallables with a knock-in term** (carried `_otc_lifecycle_knocked_in`; QUAD V2 also
`lifecycle_state`). Snowball (KO 103, continuous KI 75): carried 101 / 90 / 104 -> 0 / -10 / 15, plain 90 / 74 / 104
-> 15 / -26 / 15. Phoenix memory and not (KO 103, coupon 80, continuous KI 70): carried 101 / 98 / 90 / 104 -> 1 / -1
/ -9 / 1, plain 90 / 74 / 60 -> 1 / 0 / -40. KO-reset ABSOLUTE knocked in: 101 / 90 / 104 -> 3 / -10 / 3.

| Route | Snowball | Phoenix (memory) | Phoenix (no memory) | KO-reset knocked in |
|---|---|---|---|---|
| MC | correct | correct | correct | correct |
| PDE | correct | correct | correct | correct |
| QUAD V1 | correct | correct | correct | correct |
| QUAD V2 (flag and `lifecycle_state`) | correct | correct | correct | numeric correct; dated refused at any date (pre-existing, W8) |
| LV MC / LV PDE | correct | correct | correct | LV MC correct (no LV KO-reset PDE) |
| Heston MC, QE MC | correct | correct | correct | Heston MC correct |
| Heston PDE | correct | refuses memory coupons (pre-existing) | correct | not routed |
| SLV MC, SLV QE MC | correct | correct | correct | SLV MC correct |
| SLV PDE | correct | refuses memory coupons (pre-existing) | correct | not routed |

All of these already read the carried state at T = 0 (their knock-in flag enters the terminal payoff / event
statistics directly): nothing to fix, now pinned. A not-knocked-in KO-reset ABSOLUTE has no exercise-date state: it
matured when schedule 1 ended and every engine refuses it (`require_alive`, contract rule), so "not knocked in below
KI at T = 0" does not exist for it.

**DCN** (MC, PDE, local-vol MC / PDE, Heston MC / PDE; carrier `knocked_in_at_valuation`): refuses on its final
observation date in both states ("DCN grid has no steps", "tau must be positive", "t_grid must be a 1D array with
at least 2 points"). Not this defect (no T = 0 value at all); W8 open question Q3, unchanged.

## 4. Moved values

`moved_values.txt`. Section A (07c799cf -> branch): 222 of 1400 measured values, all carried-knock-in cells on the
exercise date, rebate -> payoff (5.0 -> 10.0 or 5.0 -> 0.0; PDE / vol-route greeks price 5.0 -> 10.0, delta 0.0 ->
1.0). Nothing else moved: not-carried cells, every day-before cell, every autocallable cell are identical. Section B
(main 6f9c6e74 -> branch, numeric form): 52 values the numeric twin got wrong on main too (the same rebate -> payoff
moves on the BS analytical / MC / PDE / QUAD / LV PDE routes and the double barrier routes), and 187 "raised ->
value" from W8 / W9 / Q1 (vol-model routes and QUAD V2 settling at T = 0; not this change).

## 5. Tests and runs (`-n 2`, worktree PYTHONPATH)

- New `test/test_expiry_carried_knock_in.py` (646 cases) and, in `test_dated_expiry_settles.py`,
  `test_vol_model_route_settles_a_carried_knock_in_at_its_payoff` (W9's `up_in_call_knocked_in` cells at the hand
  value). On the base library code: 92 failed (56 flag, 27 lifecycle state, 4 PDE greeks, 5 W9 hand value). After the
  fix: all pass.
- Suites run after the fix (four batches):
  1. every barrier / double barrier / sharkfin / one-touch / QUAD test file, `test_dated_expiry_settles`,
     `test_expiry_carried_knock_in`: 1599 passed, 6 skipped (the `_BEFORE_REFUSED` day-before cells).
  2. snowball / Phoenix / KO-reset / DCN product and engine files (incl. vol-model engines, `lifecycle_ko_reset`):
     801 passed.
  3. Heston / local-vol / SLV / Dupire engine and kernel files, `test/volmodels`, `test/fx_volmodels`: 330 passed.
  4. lifecycle, replay, `test/replay_golden`, `test/theta_boundary`, `test/modelvalidation/{test_anchors,
     test_banked_certificates}`, `test/pde_bitwise`: 495 passed, 33 skipped (superseded certificates / empty golden
     frames, skipped by design). No certificate anchor, golden or pde_bitwise row moved; none was touched.

## 6. Pre-existing refusals met on the way (T > 0, not the exercise date; left as is, listed in the tests)

- The 2D Heston / SLV snowball and Phoenix PDE routes refuse a DATED knock-out schedule before maturity ("a
  date-based knock-out observation schedule needs a pricing environment to resolve its observation times").
- The Heston / SLV barrier MC / PDE kernels refuse a discrete `observation_schedule` before maturity ("discrete
  monitoring requires observation_dates").
- The BS barrier MC prices a carried knock-in through `EuropeanMCEngine`, whose lower-bound sanity check refuses a
  one-day deep in-the-money put estimate (9.99399 vs bound 9.99430 with 256 antithetic paths; the antithetic estimate
  stays below the bound as paths grow, 9.99415 at 16384 QMC antithetic: a small systematic drift worth a look).
- The KO-reset PDE refuses a dated contract with schedule-1 records in the past before maturity (settlement
  resolution of a past record), as QUAD V2 does at any date (W8).

## 7. Open questions

- None on the fix. The four T > 0 refusals in section 6 are outside this item; the third (a sub-bound drift of
  `EuropeanMCEngine` at T = 1/365) may deserve its own look.
