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

---

## Follow-up 2026-10-10: user rulings Q1–Q5

Branch rebased onto main 0d1e4133 (tag `pre-rebase-w7` = a4b26aa5, the pre-rebase head; no conflicts; W7 tests
re-run: 37 passed). Commits: d3d5c3b0 (Q1), 96f654dc (Q2), e5bb31a6 (Q5), then the certificates, CHANGELOG and this
evidence. Every moved value vs main 0d1e4133 is in `moved_values.txt` (generated from the banked anchors and `c09` /
`c10`, main measured on a `git archive` export of 0d1e4133, the branch on the worktree, same machine).

### Q1 — `HestonSLVADICore._A1` on the degenerate v = 0 column: APPLIED, `adi2d-snowball-greeks` amended

- `adi_core_v0_column.patch` applied as is (d3d5c3b0).
- TDD regression (`TestHeston2DCertainV0Control`): the standard control KO 500 / discrete KI 1 at (96, 24, 48) within
  1e−6 of its closed form, failed first at +2.478e−3, passes at −9.31e−7. **Deviation from the ruling's wording**: with
  event damping off alone the patched value is +3.95e−5 — the Rannacher start's own θ = 1 term (`b07` predicts
  +4.05e−5), not a defect — so the test switches the terminal damping off too (both off: as-is +2.478e−3, patched
  −9.312e−7, the numbers the ruling quotes).
- Amendment `adi2d-snowball-greeks/2026-10-10` by the procedure's own run (`amend example/modelvalidation/
  adi2d_snowball_greeks.yaml --parent …/2026-10-06/certificate.json --reason …`, 1613 s on one process): the imported
  benchmark carried and never run, every candidate re-ran, 28 of 28 cells moved and were translated (shifts ≤ 0.0146
  contracts), none carried; both candidates ADMITTED; 2026-10-06 retired in place (`superseded_by`). The anchors moved
  exactly as `b06` predicted (low_feller delta +2.655e−4 / +2.568e−4, gamma −4.89e−5 / −4.85e−5; sigma_collapse delta
  −1.23e−5 / −1.25e−5; the other 20 ≤ 5.1e−10). Error against the banked reference, contracts: low_feller delta
  Heston −0.1066 → −0.0919 (gauge 0.211 → 0.196), SLV −0.1591 → −0.1449 (0.298 → 0.284); low_feller gamma
  0.0592 → 0.0565 / 0.0396 → 0.0369; sigma_collapse SLV delta −0.0294 → −0.0301 (within its s.e.).
- Other movers (`c09`, `c10`): the certain-V0 controls at default damping −6.56e−5 flat / −7.16e−5 annualized in
  every orientation (std +3.827e−3 → +3.761e−3; the rest is the damping, Q4); with damping off standard −9.31e−7,
  reverse KO 2 −7.30e−7. Desk-like 2D prices −1.6e−4 … +6.1e−4 except the reverse continuous-KI Heston Phoenix
  (KO 97 / KI 125): −5.76e−3 at (96, 24, 48), −0.1059 at (192, 48, 192). `c10` ladders it: main 95.806 / 95.702 /
  95.624 / 95.647 / 95.619 (non-monotone; n_v 48 → 96 moves it 0.078), branch 95.800 / 95.596 / 95.584 / 95.580 /
  95.573 (monotone; 0.012); library MC 95.567 ± 0.045 — main's (192, 48, 192) 3.0 s.e. off, the branch's 0.65. The
  Q3 floor is bit-identical on that whole ladder (the move is Q1, not Q3). The Feller-violating Heston DCN (DCN_A, per
  1,000,000) at (151, 41): −89,176.7 → −90,111.4, toward its refined values (main −93,527 / −93,543, branch −94,008 /
  −93,751 at (251, 81) / (501, 201)); the library QE-M MC (4× the test's paths) −93,203 ± 825 cannot rank main and
  branch at (501, 201) (208 apart = 0.25 s.e.). Byte book 2D rows: heston / slv snowball +1.18e−8, heston phoenix
  −1.68e−5.
- Tests at the Q1 state: ADI core, 2D vol-engine, Heston / SLV snowball / Phoenix / DCN PDE, `pde_statistics`,
  `pde_core`, `pde_bitwise`, `test/execution`, Phoenix PDE, `test_snowball_pde_v0_rows`: 2935 passed, 6 skipped;
  anchors + banked certificates + mo_volmodels ADI certifications + replay goldens: 151 passed, 45 skipped, 1 failed —
  `adi2d-snowball-greeks/2026-10-06` (28 anchors, as predicted), now superseded.

### Q2 — the 1D Phoenix continuous-KI copy row: FIXED, `phoenix-flat-bsm` amended

- `PhoenixAdapter.boundaries` (96f654dc): under continuous KI a `v0[k]` group first sets V1's rows of the step on
  `v1[k]` (the `v1[k]` group rewrites the same values), then exposes `v1[k]` as `_grid_v1` for the V0 call only, as
  `SnowballAdapter.boundaries` does; discrete KI takes the unchanged path. `PhoenixStatistics._alive_ki_edge_copies_ki`
  = True. It is bit-identical to `a05`'s helper patch on all 24 `a05` cases, so `a05`'s moves stand (T = 2 bit-
  identical; T = 0.25 up to +1.737e−5, reverse KI 150; +1.349e−7 standard KI 60).
- TDD: `TestPhoenix1DContinuousKICopyRow` — edge row == V1's row (standard / reverse × plain / memory) failed with 0.0
  first; the statistics reconcile within 1e−10 where the row matters (with the flag left False: +1.349e−7 / +1.737e−5).
  `pde_statistics/test_rows.py::test_phoenix_rows` pinned the old 0.0 row ("Phoenix never exposes the knocked-in
  grid") and now asserts the copied knocked-in row.
- Amendment `phoenix-flat-bsm/2026-10-10` (`amend … --rerun-candidate equity.phoenix.pde`, 5 s): 42 cells re-run,
  84 carried, all three candidates ADMITTED; 2 of 126 anchors moved, exactly the predicted near_expiry pv −2.776e−16,
  delta +6.939e−17; 2026-10-06 retired in place.
- Tests at the Q2 state: `pde_statistics`, `pde_core`, `pde_bitwise`, every `test_phoenix_*`, `test_snowball_pde_v0_rows`,
  `test_continuous_ki_scope`, replay goldens + event-stats refusal, anchors + banked certificates: 1105 passed, 33
  skipped, 3 failed — `test_phoenix_rows` (updated above) and the two superseded 2026-10-06 certificates.

### Q5 — the reverse 2D certain-V0 control: REPLACED by KO 2 / KI 10000

- `c01` runs `b05`'s own `premium` (independent full-truncation log-Euler Heston MC, no library engine) at KO 2: no
  knock-out on 1e8 paths at 10 and at 40 substeps, so |premium| ≤ max|KO cash · DF − cf| × 3 / N = 18.84 × 3e−8 =
  5.65e−7 at 95% (rule of three) — below the control test's 1e−6. Trend: KO 10 +1.31e−4 ± 1.1e−5, KO 5 +7.0e−6 ±
  2.5e−6 (KO 20 +4.90e−3, KO 15 +1.05e−3 from `b05`). Decision: replace (not drop).
- `HESTON_REVERSE_CONTROL_KO = 2.0` in `test_snowball_pde_v0_rows.py` (the 1D BSM controls keep 20); the 2D reverse
  rows test reads it (the 2D grid reaches past it, S_min 1.97, so the lower edge is the knock-out edge, branch
  `v0_lower_ko`), and the control price test runs both orientations: reverse KO 2 −7.30e−7 (KO 20 would be +4.95e−3).

### Q4 — the 2D damping default: unchanged (ruling).
### Q3 — the reverse continuous-KI Heston divergence: DIAGNOSED (no library change)

Contract: `a02`'s reverse snowball, KO 97 / continuous KI 125, W3's Heston. Standalone runtime-patched scripts
(`c02_q3_common.py` holds the patches; `c03`–`c08`), both arms back to back in one process.

**Two mechanisms, both in `_ContinuousKIFirstPassage._geometry` (the FIRST_PASSAGE crossing correction, shared with
the 1D solvers), both reverse-only, both on the near-zero-variance columns of the 2D surface.**

1. **The half-cell λ floor never acts on a reverse product.** λ = d[k*] / a[k*] is a slope, and the floor (added for
   the vol-model continuous-KI 1e13 blow-up on the ADI concentrated grid, `snowball_pde_solver.py` `_geometry`) keeps k* at least half a cell from
   the barrier. It measures the cell as `a[live[1]] - a[live[0]]`, the two live nodes of LOWEST INDEX. On a standard
   (down) product those are the two nearest the barrier. On a reverse (up) product the live side is below the
   barrier, so they are the two lowest-spot nodes, `a` decreases with the index, the "cell" is negative
   (−4.9e−2 … −7.5e−2 on every n_x scanned) and the floor removes nothing. On a column whose σ√dt is below one cell
   (the v = 0 column, σ² floored at 1e−10, and v = 3e−5, 1.9e−4), k* is then the live node nearest the barrier, and
   when the grid puts that node a few thousandths of a cell below 125, λ is a step height over a near-zero distance.
2. **The reflection factor overflows** (`c07`, `c08`): G = exp(−2 μ_e a / σ²) · [m Φ(m/s) + s φ(m/s)]. On an up barrier
   μ_e = −μ < 0 when r − q > σ²/2, so the factor is exp(+…); with σ² = 1e−10 a node within 10 s of the barrier
   (a < ~1e−5) enters the band, the factor is inf and the bracket 0: inf × 0 = NaN, written into V0. The product is
   bounded (exp(−2 μ_e a/σ²) φ(m/s) = φ((μ_e dt + a)/s)); only its evaluation overflows. The floor does not touch it.

**Decision matrix.**

| Probe (script) | Result | Reads |
|---|---|---|
| n_x scan 288 … 384 at (n_x, 48, 192), every n_x (`c03`) | diverges at 312 (−1.3e173), 317 (−1.1e30), 384 (−1.7e158): nearest live node 0.0039 / 0.0182 / 0.0046 cells below the barrier; finite but off by up to −6.6e−4 at 322 / 327 / 365 (0.027 … 0.037 cells); finite elsewhere (95.4333 … 95.4380) | alignment, not resolution: `a02`'s "finite at n_x ≤ 288" was a sample; 312 already diverges |
| crossing correction off (`c03`) | finite on all 97 n_x, +0.099 … +0.149 above the floored arm (the uncorrected O(√dt) bias) | the correction is the amplifier |
| half-cell floor measured at the barrier (`c03`, `c04`) | finite on all 97 n_x (95.4333 … 95.4381); it moves the 51 of 97 grids whose nearest node is within half a cell (finite ones by at most 6.6e−4); at 384 k* moves 0.0046 → 1.0046 cells, max λ at the first step 2.4e6 → 1.3e4 | cures mechanism 1 |
| per-step growth at the barrier (`c04`) | as-is 384: the live band's max \|V0\| grows ×4 … ×17 per backward step (×8 typical; λ 2.4e6 → 7.9e31 in 28 steps) on the v = 3e−5 / 1.9e−4 columns; 288 and the floored 384: growth ≤ 1.000 | an explicit feedback: λ ∝ d / a, d grows with V0 |
| BSM limit, standard orientation, discrete twin (`a02`, `c05`) | finite; the floor is bit-identical on each | the floor already measured the right cell there; BSM has no near-zero-variance column |
| n_x = `PRODUCTION_BARRIER_GREEK_MIN_N_X` = 600 (`c05`) | (600, 48, 192) 95.436519, (600, 60, 800) 95.424723; floor bit-identical (nearest node 0.546 cells) | safe for KI 125 by alignment |
| production path for this contract (`c05`) | `ADI_2D_PRODUCTION_ENGINE_CONTROLS`, PV on (200, 60, 800) 95.426394; `calculate_greeks` resolves (300, 135, 3200): price 95.424085, Δ −0.901871, Γ −0.057579. Under continuous KI `greek_time_grid_policy` returns before its dense-KI branch, so `barrier_greek_min_n_x` = 600 is never applied | safe for KI 125 by alignment |
| how often (`c06`, geometry only) | 24 of 1101 n_x in [100, 1200] put the node within 0.02 cells (54 within 0.05); production n_x 200 / 300 / 600 at 0.60 / 0.75 / 0.55 cells | ~2% of grids |
| other KI levels on the production grids (`c07`, KI 110 … 140 step 0.05) | 13 of 601 levels within 0.02 cells on n_x 200, 13 on 300. KI 139.80 (0.0010 cells on the Greek grid): production **Greeks raise PricingError (non-finite)**, floor → 98.313775 / −0.599596 / −0.072751. KI 114.45 (0.0009 cells on the PV grid): production **PV NaN**, floor alone still NaN | the production profile does land in it |
| guarded reflection (`c08`) | KI 114.45 PV: as-is NaN, floor NaN, floor + guarded reflection 94.678183; the six divergent cases above all finite with both; the guarded reflection alone is bit-identical on six finite checks (2D std / rev / discrete / BSM limit, 1D std / rev) | cures mechanism 2 without moving a finite value |

**PROPOSED fix (not applied; Q3 is diagnose-only).** In `_ContinuousKIFirstPassage._geometry`:

1. Measure the half-cell floor on the cell AT the barrier, on both orientations:
   `near = live[np.argsort(a[live], kind="stable")]`, `resolved = live[a[live] >= 0.5 * (a[near[1]] - a[near[0]])]`
   (one changed line; `c02_q3_common._geometry_barrier_cell`). On a standard product `near` is `live`, so it is
   bit-identical by construction; every 1D layout is barrier-aligned (`c05`: 1D bit-identical).
2. Evaluate G in one exponent on the band nodes where the naive exponent exceeds 700 (np.exp overflows past ~709):
   `m · exp(−2 μ_e a/σ² + log_ndtr(m/s)) + s · φ((μ_e dt + a)/s)` (`c08._g_guarded`); every value that is finite today
   keeps its bits. The same closed form, not a fallback.

What it would move: reverse 2D continuous-KI prices wherever the nearest live node sits within half a cell of the
barrier (`c03`: 51 of 97 grids, e.g. 288 +5.0e−5, 289 +9.8e−5, 322 +6.6e−4), plus the divergent ones. No
banked certificate prices reverse 2D continuous KI (`adi2d-snowball-greeks` monitors KI discretely), so no anchor is
expected to move; to be confirmed by the anchor guards and the byte book when applied. Suggested tests: the `c03`
divergent grids (312 / 317 / 384) finite and within 1e−3 of their neighbours, KI 139.80 production Greeks finite,
KI 114.45 production PV finite, and the standard-orientation / 1D bit-identity.

### Open questions (2026-10-10)

1. **Sign the moved anchors** before landing: `adi2d-snowball-greeks/2026-10-10` (28 values) and
   `phoenix-flat-bsm/2026-10-10` (2 values), both in `moved_values.txt`.
2. **Q3 fix**: apply the two-part proposal (barrier-cell floor + guarded reflection) as its own change? It moves the
   reverse 2D continuous-KI prices whose nearest live node is within half a cell of the barrier (51 of 97 grids in
   `c03`, finite ones by ≤ 6.6e−4) and removes the NaN / divergent / PricingError cases; it is bit-identical on every
   standard-orientation, discrete-KI, BSM-limit and 1D check run. Recommended: yes, with the regression tests listed.
3. `greek_time_grid_policy` applies `barrier_greek_min_n_x` (600) only to dense DISCRETE KI; a continuous-KI contract
   gets the generic Greek floor (300, 135, 1600/yr). Intended? (Not a cause of Q3: 600 is not safe by construction
   either, `c06`.)
4. Q1's regression test switches the terminal damping off as well as the event damping (above); confirm.
5. Q1 moves the Feller-violating 2D Heston DCN by about 1% of its PV on a coarse (151, 41) grid, toward the refined
   values; a sharper DCN reference would be needed to rank main and branch on the fine grid.

---

## W7b — the Q3 fix, applied (ruling 2026-10-10)

Commit hashes in this report are the landed ones (stacked on main a7ebefe4 as W17a -> W18 -> W7 -> TARN, 2026-10-12). The 2026-10-10 certificates record the
pre-rebase `quantark_git_sha` they were run at (8772c1fc for adi2d, 4bb994e0 for Phoenix; reachable through tag
`pre-rebase2-w7`); on the rebased branch every one of their anchors reproduces bit for bit (the guard run above).

Branch rebased onto main c02296cb (tag `pre-rebase2-w7` = ffb5db88; no conflicts). The two amended certificates'
guards and the W7 tests pass on the rebased branch (670 passed, 11 skipped): main's commits since 0d1e4133 move no
amended anchor.

**Change** (3b48a45d, `_ContinuousKIFirstPassage._geometry`, exactly `c02` / `c08`): the half-cell floor takes the two
live nodes nearest the barrier (`near = live[np.argsort(a[live], kind="stable")]`); the reflection profile is
`_reflection_profile`, the plain product where `-2 mu_e a / sig2 <= Tolerance.EXP_MAX` and the one-exponent form of the
same closed form above it.

**Tests** (`test/test_vol_pde_reverse_continuous_ki.py`; every one below failed first except the inertness set):
the floor and the reflection profile directly; KI 125 at n_x 312 / 317 / 384 finite, within 5e−3 of the nearest
well-aligned grid and within 4 s.e. + 0.02 of the library MC; production Greeks at KI 139.80 and production PV at KI
114.45 finite and within the same MC band; bit-identity against the pre-W7b `_geometry` in one process (standard
continuous, reverse discrete, BSM limit, reverse n_x 313, 1D standard / reverse / reverse KI 115). Slow (the
production Greek policy path, ruling 2026-10-10): reverse and standard orientation × Heston and SLV (unit leverage) ×
KI 114.45 / 125 / 139.80 (standard: mirrored through the spot, 87.37 / 80 / 71.53), PV and Greeks finite, the Greek
grid resolved to (300, 135, 3200), reverse within the MC band; PV n_x sweep 192 … 208 and reverse Greek n_x sweep 300 …
310, finite and within 0.02. MC references (`d01`, HestonSnowballMCEngine QE, continuous-KI bridge, 4 substeps,
200,000 antithetic × 8 batches): KI 125 95.4157 ± 0.0458, KI 139.80 98.2961 ± 0.0436, KI 114.45 94.6825 ± 0.0458.

**Results.** Fast set 14 passed; slow set 30 passed (21.6 min at -n 2); continuous-KI regression (168 files and
directories referencing continuous KI, plus anchors, banked certificates, replay goldens, pde_statistics / core /
bitwise / grid / routes, execution, mo_volmodels ADI certifications; `-m "not slow"`): 5196 passed, 138 skipped — every
certificate anchor and golden bit for bit. `d02` (the moved values, `moved_values.txt` "W7b"): 54 of 112 rows moved, 51
of the 97-grid scan (the 3 divergent ones, 48 finite by 9.2e−6 … 6.59e−4), the KI 139.80 Greeks (PricingError →
98.313775 / −0.599596 / −0.072751), the KI 114.45 PV (NaN → 94.678183) and the KI 114.45 production Greeks (+1.86e−4).
Run before the fix, `d02` moved 0 of 112 rows (the legacy copy in the test is faithful).

**Production matrix after the fix** (`d03`; SLV with unit leverage reproduces Heston to the last digit, as it must):

| orientation | KI | PV (200, 60, 800) | Greeks (300, 135, 3200): price / delta / gamma | PV n_x sweep spread | vs MC (s.e.) |
|---|---|---|---|---|---|
| reverse | 114.45 | 94.678183 | 94.675349 / −0.955185 / −0.047586 | 6.3e−4 | −0.09 / −0.16 |
| reverse | 125 | 95.426394 | 95.424085 / −0.901871 / −0.057579 | 2.6e−3 | +0.23 / +0.18 |
| reverse | 139.80 | 98.327205 | 98.313775 / −0.599596 / −0.072751 | 4.1e−3 | +0.71 / +0.41 |
| standard | 87.37 | 98.940533 | 98.939392 / +0.324837 / −0.038548 | 4.0e−4 | — |
| standard | 80 | 99.144314 | 99.143763 / +0.293119 / −0.035817 | 5.5e−4 | — |
| standard | 71.53 | 99.978517 | 99.977754 / +0.168865 / −0.026720 | 8.2e−4 | — |

No production continuous-KI configuration tested still misbehaves. Not covered: an SLV leverage surface that is not
flat (the matrix's SLV arm is the Heston dynamics through the SLV solver).

## DCN refinement ladder (ruling 2026-10-10, "(a) first")

`d04_dcn_refinement_ladder.py`, one process, no MC: DCN_A, Feller-violating Heston (ratio 0.36), HestonDCNPDESolver
from its default grid (n_x 301, n_v 101, 1 substep per daily interval); each axis doubled alone, then all together,
4 levels; patched core (W7 Q1) and the runtime-unpatched core (the v = 0 column of `A1` zeroed again — checked: it
reproduces main's (151, 41) −89,176.7328541755 bit for bit). Per 1,000,000 notional; d = successive difference, p =
observed order log2(|d_prev| / |d|).

| ladder | level | patched | d | p | unpatched | d | p |
|---|---|---|---|---|---|---|---|
| n_x | 301 | −97,286.01 | | | −96,889.48 | | |
| | 601 | −93,698.04 | +3,587.98 | | −93,312.21 | +3,577.27 | |
| | 1201 | −93,695.62 | +2.42 | 10.5 | −93,314.02 | −1.81 | 11.0 |
| | 2401 | −93,281.83 | +413.79 | −7.4 | −92,902.07 | +411.95 | −7.8 |
| n_v | 101 | −97,286.01 | | | −96,889.48 | | |
| | 201 | −97,286.14 | −0.13 | | −97,072.67 | −183.19 | |
| | 401 | −97,286.17 | −0.03 | 2.2 | −97,171.07 | −98.40 | 0.90 |
| | 801 | −97,286.19 | −0.02 | 0.8 | −97,224.11 | −53.04 | 0.89 |
| time (substeps) | 1 | −97,286.01 | | | −96,889.48 | | |
| | 2 | −97,081.43 | +204.58 | | −96,669.05 | +220.42 | |
| | 4 | −97,141.50 | −60.07 | 1.8 | −96,709.00 | −39.95 | 2.5 |
| | 8 | −97,148.25 | −6.75 | 3.2 | −96,705.71 | +3.29 | 3.6 |
| together | (301, 101, 1) | −97,286.01 | | | −96,889.48 | | |
| | (601, 201, 2) | −93,491.66 | +3,794.35 | | −93,275.82 | +3,613.66 | |
| | (1201, 401, 4) | −93,385.69 | +105.97 | 5.2 | −93,265.03 | +10.79 | 8.4 |
| | (2401, 801, 8) | −93,043.52 | +342.17 | −1.7 | −92,977.26 | +287.78 | −4.7 |

What it shows:
- **The variance axis is what the patch changes.** Patched, the price is converged in n_v already at 101 (the next
  three doublings move it 0.13, 0.03, 0.02). Unpatched it converges at first order (p ≈ 0.9: −183, −98, −53) toward the
  patched value (its differences shrink by 0.54 per doubling, so the geometric tail adds ≈ −62: −97,224 − 62 ≈ −97,286, the patched value); the patched−unpatched gap at n_v 101 (−397) is that
  O(h_v) error, and it shrinks with n_v along the "together" ladder (−216, −121, −66). On n_v the patch makes the
  ladder flat; it is the first-order v = 0 inconsistency of W7 defect B.
- **The spot axis is not converged, in either arm, and the patch does not change that.** n_x 301 → 601 moves the price
  +3,588; 601 → 1201 by 2; 1201 → 2401 by +414 again: non-monotone in n_x with the same steps in both arms (the "together"
  ladder inherits it: p = 5.2 then −1.7 patched, 8.4 then −4.7 unpatched). Neither arm is in its asymptotic range on the
  spot axis by n_x 2401, so neither ladder supports an extrapolated value; the spread of the last three spot levels is
  ~400 (0.04% of notional), larger than the patch's own move at the finer grids.
- **Time** is regular in both arms (p 1.8–3.6), the damped start's signature.
- The (151, 41) grid the W7 c09 row used (−90,111 patched, −89,177 unpatched) is far outside all of this; its −935 move
  is mostly the n_v = 41 first-order error the patch removes.

Not diagnosed (no library change asked): why the DCN's spot axis is non-monotone in n_x (a barrier / event-node
alignment effect on the concentrated grid would fit the step pattern). No sharper MC reference was run (ruling).
