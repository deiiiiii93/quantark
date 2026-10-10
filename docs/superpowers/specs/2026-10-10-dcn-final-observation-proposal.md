# W10 proposal: the DCN on its final observation date

Status: PROPOSAL for user approval. No library code is changed by this branch (`w10-dcn-final-observation`, from main
0d1e4133). Evidence: `2026-10-10-dcn-final-observation-evidence/` (`measure.py` -> `measure_0d1e4133.txt`,
`measure_routes.py` -> `routes_0d1e4133.txt`, `measure_pending.py` -> `pending_0d1e4133.txt`; standalone scripts, runtime
patches only).

Today every DCN route refuses on the final observation date in both knocked-in states (W8 `dcn_final_observation.txt`,
re-measured here on all seven routes, `routes_0d1e4133.txt`):

| Route | Refusal on the final observation date |
|---|---|
| `DCNMCEngine`, `LocalVolDCNMCEngine`, `HestonDCNMCEngine`, `QEDCNMCEngine` (and `CoupledCoarseHestonDCNMCEngine`, same `price_detailed`) | `DCN grid has no steps (valuation at maturity?)` (`dcn_mc_engine.py:311-313`) |
| execution-session DCN MC adapters | same text (`dcn_execution_adapters.py:276-279`) |
| `DCNPDEEngine`, `LocalVolDCNPDEEngine` | `tau must be positive, got 0.0` (`grid/request.py:123`, via `DCNPDEEngine.grid_request`, `dcn_pde_solver.py:95-96`) |
| `HestonDCNPDESolver` | `t_grid must be a 1D array with at least 2 points` (`param/term_sampling.py:20`) |

The proposal: settle the final row at T = 0 from ONE shared helper that writes down what the engines' own final row
already pays (section 2), shown to be the limit of both engines (section 3).

## 1. What the code defines (file:line, all under `quantark/asset/equity/`)

### Product and schedule

- **Legs.** `product/option/dcn_option.py:3-7`: daily knock-in (sticky) gates all future fixed coupons; monthly knock-out
  has priority over the fixed coupon on its date; the loss leg pays `-(N/S0)·part·max(K_loss - S_T, 0)` at the
  independent `settlement_date`; PV sign = `direction_sign` (+1 BUYER, -1 SELLER) on every leg.
- **No principal leg.** The DCN PV is coupons + knock-out coupon + loss leg. There is no par redemption flow anywhere in
  the product or the engines; "redemption" on the final date is the loss leg only (`get_payoff`, `dcn_option.py:112-128`:
  "Final-observation-date cashflow EXCLUDING coupons", 0 when never knocked in; engines never call it).
- **Levels.** All derived from ratios × `initial_price` (`dcn_option.py:79-93`); `ki_barrier < coupon_barrier` and
  `ki_barrier < ko_barrier` are enforced (`dcn_option.py:149-154`), so a same-day knock-in can never coincide with a
  knock-out or a coupon decision on a NOT-carried contract. `K_loss` may lie above the knock-out barrier (fixtures:
  K_loss 6600 / 6900 > KO 6000).
- **Coupon accrual: no daily accrual.** The fixed coupon is a monthly digital: `accrual_per_period =
  coupon_counted_days / coupon_days_denom` (30/360 by contract, `dcn_option.py:95-98`), one per merged eligible month
  (`product/option/dcn_grid.py:56-65`). The daily schedule is the knock-in monitoring set only
  (`dcn_schedule.py:173-179`). So there is no "accrued so far" coupon on the final date: the final row's coupon is
  decided by the final close alone.
- **The final row.** `build_dcn_schedule` requires `0 < lock_months <= ko_lock_months <= tenor_months`
  (`dcn_schedule.py:118-122`) and the last generated observation to equal `maturity_date` (`:143-149`), so the final
  row (month `tenor_months`, possibly merged with earlier months that adjust onto the same date, accruals summed) is
  ALWAYS both coupon-eligible and knock-out-eligible (`:162-163`). Its payment dates: `coupon_payment_date =
  trading_days_after(O, coupon_settlement_offset)`, `ko_payment_date = trading_days_after(O, ko_settlement_offset)`
  (`:164-169`); loss leg at `product.settlement_date >= final observation` (`dcn_option.py:157-161`).
- **Valuation on the final date** builds: one monthly row, one daily date, `get_maturity() == 0.0`
  (`dcn_option.py:105-110`; measured: "grid times [0.0] get_maturity 0.0"). After it: `valuation_date after
  maturity_date` (`dcn_schedule.py:125-126`), kept.
- **Past rows are dropped.** Rows observed before the valuation date are not in the schedule (`dcn_schedule.py:153-155`),
  whatever their payment date (measured, `pending_0d1e4133.txt`: with a 25-trading-day coupon lag, the month-23
  coupon observed 2024-12-03 and paid 2025-01-08 is gone from the schedule valued on 2024-12-04). No DCN engine takes a
  `lifecycle_state` (`BaseEngine.supports_lifecycle_state = False`, `engine/base_engine.py:91`); the only carried state
  is `knocked_in_at_valuation` (`dcn_option.py:57,74`). No knock-out carrier exists.
- **Timing origin.** All times are ACT/365F from `schedule.valuation_date` (`dcn_grid.py:18-22,42,66-67,78`);
  discounting is `pricing_env.get_discount_factor(t)` (`priceenv/term_sampling.py:103-126`). `settlement_convention`
  is accepted and validated (`dcn_option.py:131-137`) but no DCN engine reads it for payment timing: payment dates come
  from the schedule and `settlement_date`.

### The MC's final row (`engine/mc/dcn_payoff.py`, shared by every MC route)

Forward order on the path, column 0 = the valuation-date close (`pricing_env.spot`):

1. `:64-65` knocked in = `knocked_in_at_valuation` OR `S[0] <= KI` (carried state FIRST, then the valuation-day
   observation; commit b08019be "Include valuation-date KI in MC/PDE").
2. `:80-82` daily knock-in on `(previous obs, this obs]` for alive paths (empty on a one-column grid).
3. `:85-92` knock-out: alive AND KO-eligible AND `S_obs >= KO` pays `part·ko_coupon_rate·ko_accrual·N·DF(ko_pay)`,
   in BOTH knocked-in states; the path is then dead.
4. `:93-100` fixed coupon: alive AND NOT knocked in AND coupon-eligible AND `S_obs >= coupon barrier` pays
   `part·coupon_rate·coupon_accrual·N·DF(coupon_pay)`.
5. `:103-110` loss: alive (never knocked out) AND knocked in pays `-(N/S0)·part·max(K_loss - S_T, 0)·DF(settlement)`.

Every DF is from the valuation date to the flow's own payment date (`:56-58`). The kernel is well defined on a
one-column grid; only the engine's guard (`dcn_mc_engine.py:311-313`) stops it.

### The PDE's terminal condition (`engine/pde/core/adapters/dcn.py`; Heston: `engine/pde/dcn_vol_pde_solvers.py`)

- `DCNAdapter.terminal` (`dcn.py:125-130`): V1 (knocked-in surface) = loss leg × `DF(settlement)/DF(T)`; V0 = 0; then
  the final row's events `ev.apply(v0, v1, last row, T)` (`:47-70`; legacy masks `dcn_pde_solver.py:46-65`): coupon on
  V0 above the coupon barrier, knock-out overwrite on BOTH surfaces above KO (coupon dropped: KO has priority), V0 :=
  V1 below KI. Amounts carry `DF(pay)/DF(T)` (`dcn.py:37-45`).
- Readout (`dcn.py:150-158`): column = V1 if `knocked_in_at_valuation` OR `spot <= KI`, else V0 (carried first, then
  today's close).
- Heston: identical terminal (`dcn_vol_pde_solvers.py:247-259`) and column switch (`:301-305`).
- Default projection is cell-average (`dcn.py:55-70`): the terminal row is smeared over the one cell that contains a
  barrier (measured below); nodes away from a barrier carry the exact pointwise value.
- `terminal_outcome` returns `None` (`dcn.py:106-107`: "today's engine had no expiry branch"), hence the refusal.

## 2. Proposed final-row cash flows (T = 0)

At the valuation instant = the final observation date, with `S` = `pricing_env.spot` taken as the final close (same
instant rule as W8), `DF(d)` = the curve DF from the final date to date `d`, sign = `direction_sign`, `a_c` / `a_k` =
the final row's coupon / knock-out accrual:

1. **Knock-in state.** `KI = knocked_in_at_valuation OR S <= KI_barrier` (carried state first, then today's
   observation: the 2026-10-10 ruling; the same as both engines).
2. **Knock-out** (final row is always KO-eligible): `KO = S >= KO_barrier`, in either knock-in state. Pays
   `part·ko_coupon_rate·a_k·N·DF(ko_payment_date)`. Ends the contract: no coupon, no loss.
3. **Fixed coupon** (final row is always coupon-eligible): paid iff `NOT KO AND NOT KI AND S >= coupon_barrier`:
   `part·coupon_rate·a_c·N·DF(coupon_payment_date)`.
4. **Loss leg** iff `KI AND NOT KO`: `-(N/S0)·part·max(K_loss - S, 0)·DF(settlement_date)`.
5. **Pending cash from earlier days:** none in the PV, at T = 0 as at T > 0 (no daily accrual exists; decided past
   coupons are not carried by the product, Q3).
6. **Settled value** = sign × (KO + coupon + loss), each at its own payment-date DF. A flow paid on the final date
   itself (offset 0) enters at DF = 1.

| State | `S >= KO` | `coupon <= S < KO` | `KI < S < coupon` | `S <= KI` |
|---|---|---|---|---|
| not carried | KO coupon | fixed coupon | 0 | loss leg (knocked in today) |
| carried knocked in | KO coupon | loss leg | loss leg | loss leg |

Hand values (flat r = 3.56 %, BUYER; ×(-1) for SELLER, measured):

| Contract | KO / coupon cash | Loss at S = 4000 | Loss at S = 5500 (carried) |
|---|---|---|---|
| DCN_A (lags 2 TD: pay 2025-01-07, 4 calendar days) | 10000 × 0.99960994 = 9996.099391 | -433164.306946 | -183261.822170 |
| DCN_B (offset 0: pay on the final date; loss 2026-01-07) | 10000.000000 | -483239.059424 | -233287.821791 |

## 3. Measurements (main 0d1e4133, `measure_0d1e4133.txt`, `routes_0d1e4133.txt`)

**A. The engines' own final row equals the statement.** DCN_A and DCN_B, both states, nine spots (7000, 6300, KO 6000,
5999, 5500, coupon barrier 4800, 4650, KI 4500, 4000), plus SELLER cells:

- MC kernel on the one-column final-date path (guard bypassed by calling `compute_dcn_cashflows` directly): equal to the
  hand value in all 40 cells (printed to 1e-6; the arithmetic is the same expression).
- PDE terminal column (captured from `DCNAdapter.terminal` during a day-before solve, read at the spot with the
  readout's column rule; on a flat curve `DF(pay)/DF(T)` from the day before is the final-date DF): equal to the hand
  value in every cell whose spot is not inside a barrier cell. At the barrier cells the row is the cell average, not
  the pointwise value: coupon barrier S = 4800 not carried 5368.81 (hand 9996.10); KO barrier in the carried state
  S = 6000: -45108.40 (hand +9996.10), S = 5999: -70613.68 (hand -100127.60). Proposed: the settled value is POINTWISE
  (the kernel's closed sides `>=` KO, `>=` coupon, `<=` KI), following 4c0f2185 ("valuation-date events apply the
  exact inclusive trigger, never a cell average") and 65004edb (t = 0 transitions pointwise at spot) (Q2).

**B. Day before (2025-01-02 / 2025-12-31) vs the settled value.** MC (2^16 Sobol) and PDE agree with each other; the
difference to the settled value is the last day only:

| DCN_A cell | MC | PDE | settled | gap |
|---|---|---|---|---|
| not carried, 5500 / 6300 / 7000 | 9995.124477 | 9995.124478 | 9996.099391 | -0.975 = one more day of discounting |
| not carried, 4650 | -129.71 | -222.57 | 0 | P(knock-in on the last day) × loss |
| not carried / carried, 4000 | -433313.69 | -433313.60 | -433164.31 | -149.4: one day of forward drift (q > r) + discounting |
| carried, 5500 | -183507.45 | -183507.43 | -183261.82 | -245.6: same |
| not carried, 4800 (on the coupon barrier) | 4859.23 | 4858.76 | 9996.10 | the barrier digital: about half |
| carried, 6000 (on KO) / 5999 | -50484.89 / -51327.02 | -50478.21 / -51277.49 | +9996.10 / -100127.60 | the KO digital in the knocked-in state: about the midpoint |

DCN_B shows the same pattern (gap -4.876 = five days of discounting; carried 6300: 7593.73 / 7588.99 vs 10000: 2.2
one-day sd above KO). Every route's day-before value is in `routes_0d1e4133.txt` (LV MC/PDE equal the BS routes on the
flat grid surface; Heston MC/QE/PDE differ by the model, as expected).

**C. The limit.** The day-before contract with its last step shrunk to h (payment gaps after the final date held fixed;
runtime patch of `build_dcn_grid_context`), spots off the barriers, both contracts, both states:

| h | DCN_A not carried 5500 (MC / PDE) | DCN_A carried 4000 (MC / PDE) | settled |
|---|---|---|---|
| 1 day | 9995.124477 / 9995.124478 | -433313.694754 / -433313.599082 | 9996.099391 / -433164.306946 |
| 1 hour | 9996.058768 / 9996.058768 | -433170.528966 / -433170.414212 | |
| 1 minute | 9996.098714 / 9996.098714 | -433164.410049 / -433164.290673 | |
| 1 second | 9996.099380 / 9996.099380 | -433164.308588 / -433164.188613 | |
| 1e-9 y | 9996.099391 / 9996.099391 | -433164.306985 / -433164.186938 | |

Both engines converge to the settled value (MC to 4e-5 at 1e-9 y; the PDE's residual 0.12 = 2.8e-7 relative is its
spatial readout, not a time effect). The not-carried 4650 cell goes -129.7 -> 0.0000 by h = 1 hour. No jump remains
other than the last day's diffusion; at a spot exactly on a barrier the day-before value is the diffusion average and
the settled value takes the closed side, as for any digital at expiry.

**D. Found on the way: an interior valuation date that is a monthly observation.** Valued ON 2024-12-03 (month 23),
the MC pays that day's observation (column 0, `dcn_payoff.py:73-100`), the PDE routes do not (`dcn.py:28-31,144`:
only `times[1:-1]` are event nodes and `j == 0` is skipped; Heston `dcn_vol_pde_solvers.py:288-289`: `k > 0`):

| Cell (DCN_A, 2024-12-03) | MC | PDE | MC - PDE |
|---|---|---|---|
| not carried 5500 (coupon today) | 19801.74 | 9779.74 | 10022.0 |
| carried 6300 (KO today) | 9998.05 | -23212.00 | 33210.0 |
| LV MC / LV PDE, 5500 | 19761.61 | 9779.74 | |
| Heston MC / Heston PDE, 5500 | 19128.99 | 9151.62 | |

The day before (2024-12-02) MC and PDE agree (19746.64 / 19739.93), so the PDE value drops today's coupon overnight.
This is the same "valuation-instant observation" the final row needs, at an interior date; it is NOT proposed for W10
(it would move values value -> value). Q5.

**E. Date-rolled theta** (`riskmeasures/greek_conventions_report.py:69-87`, rolls the schedule one SSE day): on the
penultimate day it raises on MC and PDE in both states (it prices the final date); two days before it is finite.

## 4. Where the T = 0 settlement lives

One shared helper in a neutral module (no Monte Carlo import, so the PDE may call it), next to
`engine/accumulator_valuation.py`:

`quantark/asset/equity/engine/dcn_valuation.py`

- `at_final_observation(product) -> bool`: `settlement_support.at_exercise_instant(product.get_maturity())`.
- `settle_final_observation(product, pricing_env) -> DCNSettledRow`: frozen dataclass with the signed legs `ko`,
  `coupon`, `loss`, `total`, and the decisions `knocked_in`, `knocked_out`, `coupon_paid`, computed from
  `build_dcn_grid_context(product)` (last row's accruals and payment times, `loss_pay_time`) and
  `make_df_fn(pricing_env)`, in the kernel's order and arithmetic (section 2), so the MC route is bit-identical to the
  kernel's one-column row (pinned by a test).
- Not `expiry_terminal_value`: the DCN has three legs with their own payment dates, which `get_payoff × delay_df`
  cannot express; `DCNOption.get_payoff` stays unchanged.

Callers (each before any model work, as W8/W9 did with `_settled_at_exercise` / `barrier_expiry_outcome`):

| Route | Entry | Result |
|---|---|---|
| `DCNMCEngine.price_detailed` (inherited by LV, Heston, QE, coupled-coarse MC) | after `build_dcn_grid_context`, before the `dt_array` guard, `term` and `_prepare_simulation` (no Dupire build) | `DCNMCResult`: pv = total, legs, `std_error = 0.0`, one-row by-period tuples, decision indicators as probabilities, `expected_life_years = 0.0`, event stats on the same payment fields (Q6) |
| execution-session adapters (`dcn_execution_adapters.py:265-279`) | at the final observation, execute natively instead of planning batches (Q8) | same as direct |
| `DCNAdapter.terminal_outcome` (`dcn.py:106-107`; `LocalVolDCNAdapter` inherits) | `TerminalOutcome("expired", ...)` for PRICE, GREEKS, SESSION_OUTPUTS | `DCNPDEEngine.price_detailed` (`dcn_pde_solver.py:129-136`) gets a terminal branch (no layout: 0 nodes, 0 steps) |
| `HestonDCNPDESolver.price_detailed` (`dcn_vol_pde_solvers.py:167-172`) | before `_fine_grid` | same settled pv |

Unchanged: every T > 0 path (bitwise), the schedule builder, the refusal after the final date.

## 5. Tests (proposed `test/test_dcn_final_observation.py`)

- Hand table: DCN_A (2-trading-day lags) and DCN_B (offset 0, loss 2 days later) × {not carried, carried} × spots
  {7000, 6300, KO 6000, 5999, 5500, coupon 4800, 4650, KI 4500, 4000} × {BUYER, SELLER}; hand value from section 2 with
  literal anchors (9996.099391..., -433164.306946..., 10000.0, -483239.059424...).
- Every route (BS / LV / Heston / QE / coupled MC, BS / LV / Heston PDE, the session adapters) equals the hand value
  (rel 1e-12) and the MC and PDE routes are equal to each other.
- `settle_final_observation` == `compute_dcn_cashflows` on the one-column path, bitwise (pins the kernel as the
  definition).
- Result fields: MC legs / std error / event stats; PDE `price_detailed`; PDE greeks per Q6.
- Day-before continuity: for spots at least 3 one-day standard deviations from every barrier, |V(day before) - settled|
  within the last day's carry (both engines), and the measured limit of section 3C at h = 1 minute.
- Date-rolled theta on the penultimate day = settled - V(day before).
- Kept refusal: valuation after the final date.

## 6. Expected moved values ("raised -> settled" only)

- All DCN routes on the final observation date, both knocked-in states, any spot: BS MC, LV MC, Heston MC, QE MC,
  coupled-coarse Heston MC, the execution-session DCN MC adapters, BS PDE, LV PDE, Heston PDE (price, `price_detailed`,
  PDE greeks / session outputs): RAISES -> the section 2 value.
- Consequence: the cash-Greeks report's `theta_1d` on the penultimate day: RAISES -> settled - V(day before) (the only
  derived quantity that moves; it is "raised -> value").
- Nothing else: every T > 0 value (including the day before) is untouched; no certificate anchor, golden or
  pde_bitwise row contains a final-date DCN (all raise today).

## 7. Open questions (each with a recommendation)

- **Q1 The final close is today's spot, and today's observation pays.** On the final date the value includes the
  final row decided on `pricing_env.spot`, every flow at its own payment-date DF, and a flow paid ON the final date at
  DF = 1. This is what the MC kernel already does at column 0. Recommend yes (same-instant rule of W8).
- **Q2 Barrier ties at T = 0: pointwise.** The PDE's own terminal row is a cell average at the coupon / KO / KI cell
  (section 3A). Recommend the pointwise kernel rule (`S >= KO`, `S >= coupon`, `S <= KI`, closed sides as the MC) for
  every route, so both engines give the same number at a tie (precedent: 4c0f2185, 65004edb for the PDE valuation-date
  events).
- **Q3 Pending cash.** A coupon or knock-out coupon decided before the valuation date and paid after it is not in the
  DCN PV at any date (schedule drops past rows; no ledger; engines take no `lifecycle_state`). The product defines no
  carrier. Recommend: out of W10 (no change at T = 0 either); a separate item if the desk wants a DCN ledger. With the
  fixture lags (0 / 2 trading days) nothing is pending on the final date.
- **Q4 Knocked out earlier.** No knock-out carrier exists; a DCN priced on its final date is alive by construction.
  Recommend: unchanged (the caller does not price a knocked-out DCN), not invented here.
- **Q5 Interior observation on the valuation date (section 3D).** The three PDE routes drop a monthly observation that
  falls on the valuation date (10,022 / 33,210 on DCN_A), the MC routes pay it. Recommend a follow-up item (W10b): the
  PDE routes apply the valuation-date row through the same helper's row logic at t = 0, with its own moved values
  (value -> value). Not in W10.
- **Q6 Greeks and result fields at T = 0.** Recommend: price = settled; PDE delta = slope of the settled row at the
  spot (sign·(N/S0)·part·DF(settlement) where knocked in, not knocked out and S < K_loss; 0 elsewhere and at a tie);
  gamma 0; MC `std_error = 0.0`, KI / KO / coupon "probabilities" = the 0/1 decisions, `expected_life_years = 0.0`.
  Bump-and-reprice Greeks (cash-Greeks report) then see the settled row's jumps when a bump crosses a barrier: the
  contract, not smoothed.
- **Q7 `settlement_convention` on a DCN.** Accepted and validated, never applied by any DCN engine (the schedule's dates
  rule). Recommend: the T = 0 helper ignores it exactly as T > 0 does; any change is a separate item.
- **Q8 Execution sessions.** Recommend the session adapters answer the final date natively from the helper (no
  zero-batch plan, no draws), equal to the direct route.
- **Q9 Penultimate-day theta.** The date-rolled theta becomes settled(final, today's spot) - V(day before): it carries
  the last day's diffusion and the final decision. Recommend accepting this as the report's definition.
