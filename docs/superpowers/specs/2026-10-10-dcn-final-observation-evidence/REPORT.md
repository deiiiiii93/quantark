# W10 / W10b report: the DCN on its final observation date and on an observation-date valuation

Branch `w10-dcn-final-observation` (from main 0d1e4133); proposal and user rulings Q1-Q9:
`../2026-10-10-dcn-final-observation-proposal.md`. Commits 0176b5c3 (proposal), d37b431f (W10), e97e9837 (W10b).
(Report text by the W10 agent, committed by the controller because the agent could not write files here.)

## 1. W10

New neutral `quantark/asset/equity/engine/dcn_valuation.py` (no MC import): `observe_valuation_row`,
`settle_final_observation`, `at_final_observation`, `settled_row_delta`.

- Callers: `DCNMCEngine.price_detailed` (inherited by every MC subclass; `settled_dcn_result`),
  `DCNAdapter.terminal_outcome` (GBM and LV PDE), `HestonDCNPDESolver`, `DCNPDEEngine.price_detailed` (0 nodes, 0 steps).
- Sessions (Q8): the batch adapters prepare `_DCNFinalObservation`; the kernel hook `answers_natively(prepared)`
  (`quantark/execution/kernel.py`) runs it natively (`plan_fingerprint` None); session == direct bitwise.
- The settled row is bitwise the MC kernel's one-column row; every route is bitwise equal. Both engines converge to it
  as the last step shrinks (pinned at one minute).
- Penultimate-day theta = settled - today (Q9). The last-day theta is unchanged and left for a future study.
- Unchanged: W14 (receivables / knocked-out state), W15 (`settlement_convention`), the refusal after the final date.

## 2. W10b

The PDE never applied a monthly observation row at t = 0.
- Knock-out today: settles at the KO coupon with no march (greeks 0, reason `knocked_out`).
- Coupon today: added to the PDE's value from just after today (delta and gamma unchanged).
- Knock-in today: the knocked-in surface, as before.
- The spot curve applies the same rule at each level through new optional `BranchSolution` `t0_*` fields.
- `obsdate_mc_vs_pde.txt`: knock-out-today cells equal the MC exactly; the coupon-today gap is closed (DCN_A
  2024-12-03, 5500: PDE 9779.74 -> 19777.79, MC 19767.09 at 4096 paths, day before 19739.93; carried KI, KO today at
  6300: -23212.00 -> 9998.05 = MC).

## 3. Moved values (`moved_values.txt`, 816 values measured per tree with `measure_moved.py`)

- W10 (0d1e4133 -> d37b431f): 492 moved — 468 raised -> settled on the final date (8 routes, MC `price_detailed`, PDE
  greeks, sessions), 24 raised -> value for the penultimate-day `theta_1d`. Unchanged: 108 day-before controls, 216
  observation-date values.
- W10b (d37b431f -> e97e9837): 90 moved — PDE value -> value on observation-date valuations (BS, LV, Heston). Unchanged:
  every final-date value, control, theta and MC value.
- No certificate anchor, golden or pde_bitwise row moved.

## 4. Tests

`test/test_dcn_final_observation.py` 117 tests (112 failed before W10); `test/test_dcn_valuation_date_observation.py`
35 tests (25 failed before W10b). After W10: 821 passed (DCN, execution, pde_routes DCN, pde_core prepare, cash-greeks,
greek conventions, settlement) and 1245 passed / 35 skipped (theta_boundary, dated-expiry, carried-KI, `test_anchors`,
`test_banked_certificates`, `test/pde_bitwise`). After W10b: 1149 passed (+ pde_core, pde_grid) and 1245 / 35.

## 5. Open

The grid spot-curve row and `price()` at the same spot differ by 6.6e-8 relative on the 2024-12-03 observation date
(5500): two readouts of one grid; the test allows rel 1e-6. Not checked on dates without an observation.
