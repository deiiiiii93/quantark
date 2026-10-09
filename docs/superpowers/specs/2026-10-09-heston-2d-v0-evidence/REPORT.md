# W7: two 2D Heston/SLV PDE defects left alone by W3 (2026-10-09)

Branch `w7-heston-2d-rows` from main e4271b6d. (Report text by the W7 agent, committed by the controller.) Scripts here
are standalone and runtime-patched; both arms of every comparison run back to back in one process; printed output in
`outputs/`. Run from the repo root with `env PYTHONPATH=$PWD:$PWD/test <venv>/bin/python <script>`. Prices per 100
notional. Heston (W3's): v0 = theta = 0.04, kappa 1.5, sigma 0.3, rho −0.5; r = 0.03, q = 0, T = 2.

| | Status | One line |
|---|---|---|
| A: continuous-KI copy row reads a never-set `_grid_v1` | FIXED (6e051d37) | 2D edge row 0.0 → the knocked-in row; desk prices move at round-off; no banked anchor moves |
| B: certain-V0 control bias | DIAGNOSED, NOT FIXED (gate) | ADI core `A1` skips the v = 0 column + damping by design + a real Heston KO premium (reverse); fix proposed as `adi_core_v0_column.patch`, it moves the adi2d certificate |
| Side: reverse continuous-KI Heston diverges at n_x = 384 | reported | `a02` |
| Side: 1D Phoenix has defect A too | reported (gate) | same fix moves `phoenix-flat-bsm` near_expiry by ~3e−16 |

## Defect A — the 2D continuous-KI copy row

**Cause.** Under continuous KI, `SnowballPDESolver._boundary_rows_v0` gives the V0 knock-in-side edge the knocked-in
row when the edge is at/beyond the KI level (`v0_{edge}_copy_ki`), reading `self._grid_v1[row, t_idx]` or 0.0 when
None. The 1D `SnowballAdapter` sets V1's rows first (`adapters/snowball.py:321`); the 2D `_boundary_hook` of the
snowball and Phoenix solvers solves V1 to completion and never sets `_grid_v1` → the row was 0.0. The 2D lower edge is
8 standard deviations out (S_min 10.2 at T = 2), so the branch is taken on essentially every 2D continuous-KI contract.

**Measured** (`a01`, `a03`, `a04`, `a07`):
- Rows at the valuation step: KO 103 / KI 75 standard lower row 0.0 → 10.200416; KO 97 / KI 125 reverse upper row
  0.0 → −791.999284.
- Desk-like prices: |after − before| ≤ 4e−13 at (96, 24, 48) and (192, 48, 192) for Heston/SLV snowballs and
  standard/reverse Heston Phoenix; delta/gamma 1.4e−14 / 2.8e−14. The step hook overwrites every node at/below the
  barrier with V1 after each step, so a wrong Dirichlet value decays to round-off before reaching the barrier.
- KI just inside the edge (Heston, (96, 24, 48)): KI 10.5 +8.09e−5, 11 +5.58e−5, 12 +2.55e−5, 15 +3.9e−6; BSM limit
  (sigma 1e−4) ≤ 7e−11. The fixed row equals what the solver's own step hook writes on that node each step.
- PDE byte book 2D rows: heston_snowball and slv_snowball 97.99960730539338 → 97.99960730539375; heston_phoenix
  −2.406498274944137 → −2.406498275044068.

**Fix.** `SnowballPDESolver._set_boundary_conditions_v0_beside_v1_rows`: only under continuous KI, computes
`_boundary_rows_v1` for the step, exposes them as `_grid_v1` for that call and restores the previous value; otherwise
exactly `_set_boundary_conditions_v0`. Both 2D hooks call it for V0. V1's 2D edge rows are uniform in v, so the
scalar copy equals the knocked-in row on every variance column (asserted).

**Tests** (`test/test_snowball_pde_v0_rows.py::TestHeston2DContinuousKICopyRow`, failed first with 0.0 vs 10.2004):
standard and reverse snowball (edge row == knocked-in row on all 24 variance columns, written across the v axis);
Phoenix (edge row == V1's row).

**1D Phoenix — same defect, not fixed (gate).** `PhoenixAdapter` never sets `_grid_v1` (`statistics/phoenix.py:62`).
`a05` patch: T = 2 bit-identical; T = 0.25 up to +1.737e−5 (reverse, KI 150), +1.35e−7 (standard, KI 60). `a06`
banked `phoenix-flat-bsm` 2026-10-06: 124/126 bit-identical; pde/near_expiry pv −2.8e−16, delta +6.9e−17 (fails the
exact anchor). Phoenix statistics would also need `_alive_ki_edge_copies_ki = True`.

## Defect B — the 2D Heston certain-V0 control

Control (W3's): KO 500 / KI 1 standard, KO 20 / KI 10000 reverse, flat rebate 10%, 24 monthly KO dates, discrete KI;
closed form 110·e^{−rT} = 103.594.

Reproduced (`b01`, PDE − cf):

| n | std flat | std ann | rev flat | rev ann |
|---|---|---|---|---|
| (96, 24, 48) | +3.827e−3 | +4.174e−3 | +8.742e−3 | +6.108e−3 |
| (192, 24, 192) | +2.096e−3 | +2.286e−3 | +6.931e−3 | +4.211e−3 |
| (192, 192, 192) | +2.37e−4 | +2.58e−4 | +5.03e−3 | +2.17e−3 |

Decision matrix (`b03`, `b02`; one knob from (96, 24, 48); std / rev):

| Knob | std | rev | Reads |
|---|---|---|---|
| r = q = 0 | −2.8e−8 | +7.08e−3 | standard: the whole gap is discounting |
| r = q = 0.03 | +3.826e−3 | — | carry not the cause |
| KO out of reach | +3.826e−3 | +3.826e−3 | standard gap not the KO region; the reverse excess is |
| event damping 0 | +2.452e−3 | +7.50e−3 | |
| event_theta 0.5 | +2.453e−3 | +7.25e−3 | damped steps' θ = 1 is the 1.4e−3 |
| BSM limit | +3.765e−3 | +3.762e−3 | reverse excess is Heston-only |
| BSM limit, damping 0 | +4.1e−5 | +4.0e−5 | the Rannacher start |
| n_t 96/192/384 | 2.19/2.10/2.23e−3 | 7.27/7.22/7.36e−3 | does NOT converge in n_t |
| n_v 48/96/192, damping 0 | 2.2e−4/5.6e−5/4.1e−5 | 5.23/5.06/5.04e−3 | n_v-dependent, n_t-independent part |
| n_x 192/384 | 3.827e−3/3.827e−3 | 8.47e−3/8.41e−3 | not n_x |
| Douglas vs Craig–Sneyd | +3.827e−3 | +8.74e−3 | |
| v-drift centered / semi-Lagrangian | 3.86e−3/3.97e−3 | 8.77e−3/9.49e−3 | not the v-drift scheme |
| v0_boundary neumann | +3.761e−3 | +8.68e−3 | |

**Root cause — three parts.**
1. **ADI-core bug (`HestonSLVADICore._A1`)**: fills `out[1:-1, 1:-1]` only, so on the degenerate v = 0 column
   A1·U = 0 while `_tri_S` applies S-drift and −rU on every column → the v = 0 row evolves under θ·A1 + A2,
   inconsistent on every θ = 1/2 step. Runtime patch (`b04`, adds `(r − q) U_x − r U` there): standard damping-0
   +2.4525e−3 → +3.95e−5; n_v 48/96/192: 2.2e−4/5.6e−5/4.1e−5 → 3.96/3.97/3.97e−5; (192, 24, 192) +2.096e−3 →
   +2.35e−4; (192, 192, 192) 2.37e−4 → 2.35e−4.
2. **Damping, by design**: `DAMPING_DEFAULTS["2D"] = ("full_step", 1, 2)`; 46 of 48 steps are θ = 1, each
   + c (r dt)²/2, Rannacher start + c (r dt)²/4. Patched (`b07`): n_t 48 +3.7613e−3 vs predicted +3.7634e−3
   (−2.1e−6); 96 +9.404e−4 vs +9.408e−4; 192 +2.351e−4 vs +2.352e−4. All damping off: −9.3e−7/−2.7e−7/−1.1e−7.
   Second order in dt.
3. **The reverse control is not certain under Heston** (`b05`, independent full-truncation log-Euler Heston MC,
   2,000,000 antithetic paths, no library MC engine): reverse KO 20 true − cf +4.902e−3 ± 2.0e−4 (P(KO) 3.1e−4);
   standard KO 500 exactly 0; reverse KO 15 +1.05e−3 ± 9.4e−5. Patched PDE → +5.03e−3, within 0.7 s.e.

**Proposal (not applied)**: `adi_core_v0_column.patch` (~10 lines in `_A1`, only under `degenerate_v0`; affects
every 2D Heston/SLV solve on the degenerate v = 0 boundary — snowball, Phoenix, DCN 2D defaults; `neumann` default
kernels unaffected). `b06`, banked `adi2d-snowball-greeks` 2026-10-06 (as-is == banked bitwise):

| Anchor | Move | abs error vs banked MC reference |
|---|---|---|
| low_feller delta Heston | +2.655e−4 | 1.93e−3 → 1.67e−3 |
| low_feller delta SLV | +2.568e−4 | 2.88e−3 → 2.63e−3 |
| low_feller gamma Heston | −4.89e−5 | 1.07e−3 → 1.02e−3 |
| low_feller gamma SLV | −4.85e−5 | 7.18e−4 → 6.70e−4 |
| sigma_collapse delta Heston | −1.23e−5 | 4.0e−5 → 2.8e−5 |
| sigma_collapse delta SLV | −1.25e−5 | 5.33e−4 → 5.46e−4 (s.e. 6.8e−4) |
| sigma_collapse gamma | +2.0e−7 / +2.1e−7 | |
| other 20 | ≤ 5.1e−10 | |

Regression test with the fix: the certain-V0 control at (96, 24, 48), event damping off, within 1e−6 of the closed
form (as-is +2.45e−3, patched −9.3e−7).

## Side finding — reverse continuous-KI divergence (`a02`)

KO 97 / continuous KI 125 reverse: finite at (192, 48, 192), (192, 48, 384), (288, 48, 288); (384, 48, 192)
−1.6e158; (384, 48, 384) −3.8e211; (384, 24 or 96, 384) NaN. Discrete twin (95.81), standard orientation and BSM limit
finite. Both arms (before/after A) diverge identically. Suspect: the 2D continuous-KI crossing correction on a fine
x-grid, reverse orientation. Not diagnosed further.

## Moved values (checked at -n 2 on the banking machine)

- Banked certificate anchors: all live files pass exact (14 passed, 29 skipped as superseded); `adi2d-snowball-greeks`
  monitors KI discretely and never reaches the new path.
- Replay goldens / replay event-stats: 84 passed, 4 skipped by design.
- `pde_statistics`, `pde_core`, 2D vol-engine suites, ADI core suites, `test_snowball_pde_v0_rows`: 1243 passed.
- Unbanked movers: byte book 2D rows (+3.7e−13, +3.7e−13, −1.0e−10); other 2D continuous-KI prices at round-off; up to
  8.1e−5 with an edge KI.

## Open questions

1. Apply `adi_core_v0_column.patch` and re-certify `adi2d-snowball-greeks` through an amendment? Recommended: yes.
2. Fix the 1D Phoenix copy row, with an amendment for the two `phoenix-flat-bsm` near_expiry ulp anchors?
   Recommended: yes, as a follow-up.
3. Diagnose the reverse continuous-KI divergence at n_x = 384? Recommended: yes — the production barrier-Greek floor
   `PRODUCTION_BARRIER_GREEK_MIN_N_X = 600` lands in that regime.
4. Change the 2D damping default? Recommended: no; it converges as dt².
5. Retire the reverse 2D certain-V0 control (use the standard orientation, or KO 2 / KI 10000)? Recommended: yes.
