# W8 — a dated option valued on its exercise date prices its settled value

User ruling 2026-10-09, option (a): a dated equity option valued ON its exercise date prices its settled value,
identical to its numeric twin aged to exactly maturity 0.0. Branch `w8-dated-expiry-settles` from main e4271b6d.
(Report text written by the W8 agent; committed by the controller because the agent could not write files here.)

## 1. Measurement

`measure.py` prices each family of `FAMILIES` (`test/test_dated_expiry_settles.py`) on every engine that routes it,
in two forms, at several spots:

- **Dated:** exercise date = valuation date = 2026-01-05, initial date 2025-01-05, fixings / observations on
  initial + round(k/12 × 365) days, the last one today.
- **Numeric twin:** aged to maturity 0.0 with `time_shift(1.0, keep_landing=True)`; the range accrual and Asian are
  built directly with fixings at k/12 − 1.
- **Settled value (by hand):** the payoff at the spot, past fixings as recorded, today's fixing live at the spot.

Market: flat 20% vol (a flat GridVolSurface for the local-vol routes), r 3%, q 1%. Heston: v0 = θ = 0.04, κ 1.5,
σ 0.3, ρ −0.5.

On main every dated cell raised. Of the 188 numeric cells, 154 priced and 8 of those were wrong (Asian MC 2,
accumulator 6). On the branch, 154 cells give dated == numeric == settled to rel/abs 1e-12, and 34 refuse in both
forms. Raw tables: `today_e4271b6d.txt`, `after_branch.txt` (`summarize.py`).

| Family (settled values) | Engines | Main: dated | Main: numeric at T = 0 | Branch |
|---|---|---|---|---|
| European call K100 (10, 0) | BS analytical, MC, PDE, QUAD, LV PDE | raises | correct | settles |
| European call K100 | Heston analytical, Heston MC, Heston PDE, LV MC | raises | raises | raises both (TODO) |
| American put K100 | analytical, MC, PDE | raises | correct | settles |
| Digital call, payout 10 | analytical, MC | raises | correct | settles |
| Asian fixed call, fixings 101–111 + today (6.3333, 4.6667) | analytical | raises | correct | settles |
| Asian fixed call | MC | raises | wrong (10, 0) | settles (fixed) |
| Up-and-out call K100 B130, rebate 5 | analytical, MC, PDE, QUAD, LV PDE | raises | correct | settles |
| Up-and-out call | LV MC, Heston MC, Heston PDE | raises | raises (t_grid) | raises both (TODO) |
| Double knock-out call 80/130 | analytical, PDE | raises | correct | settles |
| One-touch 130 | analytical, PDE, QUAD | raises (year-fraction message) | correct | settles |
| Double one-touch | PDE | raises | correct | settles |
| Single sharkfin, double sharkfin | analytical, MC | raises | correct | settles |
| Range accrual, 100 × 10%, 11 in range + today (10, 9.1667) | analytical, MC | raises | correct | settles |
| Accumulator (5, −25, 2) | analytical, MC | raises | wrong (0, −15, 0) | settles (fixed) |
| Snowball KO103, continuous KI75 (15, 15, −30) | MC, PDE, QUAD V1, LV MC, LV PDE, Heston MC, Heston PDE | raises | correct | settles |
| Snowball | QUAD V2 | raises | raises (`validate()`) | raises both (TODO) |
| Phoenix, memory and non-memory (1, 1, 0, −40) | MC, PDE, QUAD V1, LV MC, LV PDE, Heston MC, Heston PDE (non-memory only) | raises | correct | settles |
| Phoenix | QUAD V2 | raises | raises (`validate()`) | raises both (TODO) |
| KO-reset ABSOLUTE, knocked in (3, −10) | MC, PDE, QUAD V1 | raises | correct | settles |
| KO-reset | QUAD V2 | raises | raises (`validate()`) | raises both (TODO) |

Greeks on the exercise date (price, delta, gamma, theta, vega, rho) match between the two forms on 13 routes
(`greeks_on_exercise_date.txt`). Not measured: SLV routes, FX products, a settlement lag after the exercise date,
the DCN (Q3).

## 2. The change

- `time_to_exercise()` in `base_equity_option.py`: 0.0 when the valuation date equals the exercise date, raises when
  strictly after, otherwise the year fraction as before. `BaseEquityOption`, `OneTouchOption` and
  `DoubleOneTouchOption` use it in `get_maturity`.
- Asian MC expiry shortcut: averages the recorded fixings plus today's live one, as the analytical engine does.
- Accumulator analytical + MC expiry shortcut: new `settle_at_expiry` (calls `settle_valuation_instant`, refuses any
  observation after the maturity instant).
- QUAD V2: refuses a dated contract at T = 0 (TODO), matching its `validate()` refusal of the numeric maturity 0.0.
- Replay and ensemble readout: a dated maturity day keeps the 1e-8 dividend pillar (a carry-implied yield refuses
  T ≤ 0).
- `# TODO(dated-expiry)` at the eight vol-model refusal sites.

New test `test/test_dated_expiry_settles.py`: 196 tests, all passing on the branch; on main's tree 155 failed,
41 passed. CHANGELOG `[Unreleased]` "Changed" entry.

## 3. Moved values

See `moved_values.txt`. No certificate anchor, replay golden or pde_bitwise capture moved.

- Dated contracts on their exercise date: raised → settled value (154 of 188 cells).
- Asian MC, numeric twin at maturity 0.0 (ignored recorded fixings, bug): spot 110 10.0 → 6.3333; spot 90
  0.0 → 4.6667.
- Accumulator analytical + MC at maturity 0.0 (dropped the t = 0 observation, bug): spot 100 0 → 5; spot 90
  −15 → −25; spot 110 (knocked out today) 0 → 2 (the rebate).
- `test/test_accumulator_mc_engine.py`: 5.0 → 10.0, −5.0 → −15.0, −5.0 → −15.0.
- Error text strictly after the exercise date: now "must be on or before exercise date" (routes resolving
  settlement timing first keep their own message).

## 4. Runs (`-n 2`)

1. Products, engines, missing-fixings, theta_boundary, pde_routes: 1179 passed (3 accumulator expectations updated).
2. Replay golden, replay, backtest, lifecycle, tracker, matured-before-run, simulation: 758 passed.
3. The new test, intraday, accrual_time, all pnlexplain, theta, replay golden, replay/simulation re-run: 1521 passed.
4. Barrier, digital, sharkfin, American, range accrual, snowball, Phoenix, KO-reset, VaR, stress, portfolio,
   vol-model, execution, pde_core, futures carry: 1742 passed.

## 5. Routes left refusing (both forms, `# TODO(dated-expiry)`)

- Heston analytical / MC / PDE and local-vol MC European: "maturity must be positive".
- Local-vol, Heston and SLV barrier MC; Heston and SLV barrier PDE: "t_grid must be strictly increasing".
- QUAD V2 snowball / Phoenix / KO-reset. (Settles since ruling Q1, 2026-10-10: §7.)
- Unchanged, not expiry-caused: Heston Phoenix PDE with memory coupons; QUAD V2 on a dated KO-reset whose
  schedule-1 records are in the past.

## 6. Open questions

- **Q1 QUAD V2:** let V2 accept an aged copy with maturity == 0.0 and drop the dated refusal. A runtime patch
  (`proposal_quad_v2.py`) priced the settled value exactly in both forms on all 11 cells. Recommend yes.
  **Ruled yes 2026-10-10; implemented, §7.**
- **Q2 vol-model routes:** exact settlement at T = 0 (payoff at the spot × settlement discount factor) for the eight
  refusing routes, as the BSM engines have. Recommend yes, as a follow-up.
- **Q3 DCN on its final observation date:** both engines refuse ("DCN grid has no steps", "tau must be positive");
  no numeric twin exists. Recommend a separate item settling it at the final row (`dcn_final_observation.txt`).
- **Q4 same instant vs same day:** a timestamp later on the exercise day counts as after it and raises. Recommend
  keeping the same-instant rule.

Note: numerical theta returns 0.0 whenever maturity ≤ the time bump, so last-day theta is 0 in both forms on both
trees (`accumulator_theta.txt`), unchanged by this branch.

## 7. Rebase + Q1 + W6 items (2026-10-10)

**Rebase** onto main 6f9c6e74 (W6 run fixings; `pre-rebase-w8` tags the old head). One conflict, the
`product_replay.py` imports (W8's `is_zero` + W6's `run_fixings`); CHANGELOG merged cleanly (W6 and W8 entries both
kept). After the rebase `test_dated_expiry_settles`, `test_run_fixings`, `test_run_fixings_runs`: 330 passed.

**Q1 (user ruling 2026-10-10): QUAD V2 settles on the exercise date.** `compile_contract` re-validates its copy with
`validate(allow_zero_maturity=True)` (new keyword on `BaseEquityOption` / `SnowballOption` / `PhoenixOption` /
`KnockOutResetSnowballOption`; default False, so construction still refuses maturity 0.0, and a negative maturity is
refused either way); W8's dated T = 0 refusal is removed. Unlike the runtime patch (`proposal_quad_v2.py`, which
validated as maturity 1.0 and rebuilt the contract terminal at 2.0), the copy keeps its own terminal period
(`accrued_offset + 0.0`). All 11 snowball / Phoenix V2 cells: dated == numeric == settled to 1e-12. The knocked-in
KO-reset: the numeric twin now settles on V2 (3, -10); the dated one keeps its refusal, which is NOT the exercise-date
path -- a schedule-1 record in the past fails settlement resolution ("end_date ... must be after start_date") at any
valuation date (measured mid-life, 292 days in; there the numeric twin prices 2.95303 on V2, 2.95289 on V1). Left as is.

**W6 items.**
- a. The replay tracker ages a participant with `with_time_elapsed` and no 1e-8 floor (plan Ruling 12 superseded):
  0.0 on its maturity day, equal to the dated twin (numeric Asian copy on T 6.9999999965 -> 7.0); negative past it,
  refused as the dated twin is. The replay dividend-pillar readout maps 0.0 / negative to the same 1e-8 pillar.
- b. `check_contract_roll` expects a participant's `with_time_elapsed` maturity; an explain step ending on the
  maturity explains (it raised) with the dated twin's rows.
- c. `test_ageing_to_maturity_snaps_and_prices_as_settled` asserts the dated twin again.
- d. Backtest twins (range accrual, Asian) through T: both settle (10.0, 7.0) on T; a run to T + 1 refuses both
  (`ValidationError`), so `test_equity_backtest_twins_refuse_a_run_past_the_maturity` is kept. The replay cannot reach
  T with an Asian in either form, on main too: `AutocallableLifecycleTracker.settle_maturity_if_due` calls
  `get_payoff(spot, env, knocked_in=...)` (TypeError). Cash for range accruals / Asians at expiry inside runs is
  out of scope (run-fixings spec §10); the replay test therefore checks the pricing copy on T, not the settlement.

No certificate anchor, replay golden or pde_bitwise capture moved. Moved values: `moved_values.txt`, last section.

Runs (`-n 2`): QUAD (all `test_*quad*`) + snowball / Phoenix / KO-reset product and lifecycle: 637 passed; replay
golden + replay + pnlexplain + run fixings + this test file: 681 passed, 4 skipped (golden frames recorded empty);
backtest, lifecycle, trackers, missing fixings, simulation, FX backtest / dynamic: 794 passed; intraday + accrual
time + theta boundary: 754 passed.
