# T14: DCN n_x ladder diagnosis (I10 / AC14)

Run 2026-10-13/14 in worktree `ship-followup-queue` (HEAD at start 7577d5cf, then e89ec8c3, which touches only `lifecycle/dcn.py` and the replay, not the PDE solvers). Values are per 1,000,000 notional. The scripts are in this directory and their logs in `outputs/`. Run any script from the worktree root with `env PYTHONPATH=<wt>:<wt>/test <venv>/bin/python <script>`.

## Question

W7b's d04 ladder (`docs/superpowers/specs/2026-10-09-heston-2d-v0-evidence/`, REPORT.md:325-368) priced `HestonDCNPDESolver` on DCN_A (`test/dcn_fixtures.py`, `flat_env(**FLAT)`) under a Feller-violating Heston model (v0 .0249, kappa 9.92, theta .045, sigma 1.583, rho -.025; Feller ratio 0.36), with n_v 101 and 1 substep. Along n_x the price was non-monotone: 301 → 601 → 1201 → 2401 moved it +3,588, +2.4, +414 in both core arms, while the n_v axis was flat and the time axis regular. Why?

## Answer

The cause is the event treatment, together with the level pinning it relied on. The 2D Heston DCN applied every coupon / KO / KI event as a whole-node mask (`s >= b` / `s <= b`) on a grid that moves one node onto each level. A nodal mask puts the event's step at a dual-cell edge, half a local cell to one side of the level. That is an O(h) trigger shift, and its size depends on which side owns the level's node and on the two cells next to it. On the pinned grid those two cells are set by the level's position inside its unpinned sinh cell. On the nested doubling ladder that position follows the level's binary digits (KO: 0.542 → 0.087 → 0.174 → 0.347), so the error jumps between grids instead of shrinking smoothly.

Node ownership was also decided by 1-ULP `exp(log b)` noise: the KO and KI nodes were not owned by their masks, although the contract compares inclusively.

The 1D DCN route has used the dual-cell projection on an unsnapped grid since 0.10.0, and so has every other 2D autocallable route; the 2D DCN was never ported. This is a contained defect, now fixed: by default the 2D DCN uses the 1D route's projection on the unpinned grid, and `event_projection="nodal"` keeps the old discretization bit for bit.

## Reproduction (e01)

| n_x | d04 (2026-10-10) | this worktree | difference |
|---|---|---|---|
| 301 | -97,286.0115 | -97,286.01153979279 | 0.0000 |
| 601 | -93,698.0363 | -93,698.03627772542 | 0.0000 |
| 1201 | -93,695.6189 | -93,695.6188573165 | 0.0000 |
| 2401 | -93,281.8252 | -93,281.82524896841 | 0.0000 |

- The numbers match d04 to 4 dp. The T11-T13 changes do not affect this inception valuation.
- The probe in `common.py` is a verbatim copy of `_price_contingent` with the event calls switchable. Its "nodal" arm is bit-identical to production (e01 at n_x 301; e08 at 301 and 601).

## Measurements

### Grid geometry (e02, no solves)

Per level: the pre-pin offset in its sinh cell, the ratio h_r/h_l of the two spacings next to the pinned node (log units), and the production nodal step position relative to ln b.

| n_x | KI 4500: offset, h_r/h_l, step | CPN 4800: offset, h_r/h_l, step | KO 6000: offset, h_r/h_l, step |
|---|---|---|---|
| 301 | 0.629, 2.18, -0.00395 | 0.751, 1.67, -0.00475 | 0.542, 2.71, +0.01047 |
| 601 | 0.258, 0.59, -0.00395 | 0.501, 2.99, -0.00159 | 0.087, 0.84, +0.00326 |
| 1201 | 0.515, 2.88, -0.00081 | 0.003, 0.99, -0.00159 | 0.174, 0.71, +0.00147 |
| 2401 | 0.031, 0.94, -0.00081 | 0.007, 0.99, -0.00080 | 0.347, 0.48, +0.00058 |
| 4801 | 0.061, 0.88, -0.00042 | 0.013, 0.97, -0.00040 | 0.695, 1.88, +0.00058 |

- Every level is pinned on a node at every n_x, so pinning works as designed.
- `exp(log b)` lands 1 ULP off b for KI (4500.000000000001) and KO (5999.999999999995). So the production masks `s <= KI` and `s >= KO` do not own their level's node. CPN (4800.000000000001) is owned. The contract is inclusive on all three.
- The nodal step does not shrink like h: KI's step is unchanged from 301 to 601 and from 1201 to 2401, while KO's drops 3.2x from 301 to 601. The cells next to each pinned node are uneven (ratios 0.35 to 2.99), and the ratios vary with n_x.

### Dense n_x ladder, three event treatments (e03; n_v 101, substeps 1)

| n_x | nodal (= production) | inclusive (nodal, ULP-robust ownership) | cell-average, pinned grid |
|---|---|---|---|
| 301 | -97,286.01 | -91,697.43 | -91,366.64 |
| 401 | -92,809.89 | -88,896.95 | -93,593.42 |
| 501 | -93,750.07 | -90,136.44 | -92,939.74 |
| 601 | -93,698.04 | -91,290.74 | -93,129.74 |
| 801 | -93,733.33 | -92,115.43 | -93,281.00 |
| 1001 | -93,120.81 | -91,503.62 | -93,371.12 |
| 1201 | -93,695.62 | -92,170.10 | -93,058.73 |
| 1601 | -93,905.54 | -92,981.62 | -93,149.02 |
| 2401 | -93,281.83 | -92,549.07 | -93,196.20 |
| 3201 | -93,266.32 | -92,774.99 | -93,195.94 |
| 4801 | -93,369.33 | -93,080.32 | -93,194.78 |

- **Nodal swings.** The nodal arm moves +4,476, -940, +613, -575, +624 along the dense ladder. d04's +2.4 step was a near-cancellation between neighbours of a ±600 oscillation.
- **Ownership flip.** The inclusive arm differs from nodal by 5,589 at n_x 301. The gap between the two arms is 5,589 / 2,407 / 1,525 / 733 / 289 at 301 / 601 / 1201 / 2401 / 4801, halving per doubling: a first-order half-cell shift. The two arms sit on opposite sides of the cell-average arm (at 4801: -175 and +114).
- **Pinned cell-average.** It converges from 1601 on (steps -47, +0.3, +1.2) but is erratic below that (301: -91,367; 401: -93,593; a +312 step at 1201).

### Unpinned grid (e04; n_v 101)

| n_x | cell-average, unpinned | d | nodal, unpinned | d |
|---|---|---|---|---|
| 301 | -92,971.87 | | -91,767.32 | |
| 401 | -93,040.10 | -68.2 | -95,002.10 | -3,234.8 |
| 501 | -93,002.76 | +37.3 | -94,179.11 | +823.0 |
| 601 | -93,028.00 | -25.2 | -95,082.29 | -903.2 |
| 801 | -93,103.71 | -75.7 | -92,735.46 | +2,346.8 |
| 1001 | -93,152.60 | -48.9 | -93,270.78 | -535.3 |
| 1201 | -93,114.74 | +37.9 | -94,025.84 | -755.1 |
| 1601 | -93,154.52 | -39.8 | -93,140.92 | +884.9 |
| 2401 | -93,168.04 | -13.5 | -93,620.75 | -479.8 |
| 3201 | -93,189.30 | -21.3 | -93,591.79 | +29.0 |
| 4801 | -93,186.21 | +3.1 | -93,232.05 | +359.7 |

- The cell-average projection needs no node on a level, and removing the pinning removes the pinned arm's coarse-grid error: at 301 it is about 220 unpinned against about 1,820 pinned, relative to the fine value of about -93,190.
- At 4801 the two cell-average arms agree within 9.
- Nodal on the unpinned grid (the textbook mid-cell case) stays erratic.

### 1D BSM DCN on the same contract (e05; sigma 0.184)

| points | default (cell-average) | "nodal" |
|---|---|---|
| 301 | -93,282.61 | -93,168.88 |
| 401 | -93,282.09 | -93,666.77 |
| 601 | -93,286.43 | -92,825.79 |
| 1001 | -93,286.42 | -94,219.90 |
| 1201 | -93,286.60 | -93,565.31 |
| 2401 | -93,289.44 | -93,514.67 |
| 4801 | -93,293.08 | -93,485.39 |

All rows are in `outputs/e05.txt`. Across the ladder the cell-average steps stay within ±10.5 and the nodal steps reach ±1,048. The erratic ladder therefore comes from nodal events in general, not from the 2D core, the Feller regime or the v = 0 boundary.

### Other axes and the MC reference

- **n_v (e07b):** cell-average, unpinned, n_x 1201: n_v 41 / 101 / 201 give -93,111.19 / -93,114.74 / -93,115.33. The axis is flat, so the test's n_v 41 is harmless.
- **Time (e09):** new default, n_x 2401, n_v 41: substeps 1 / 2 / 4 give -93,164.56 / -92,813.19 / -92,946.47 (steps +351, -133). This is d04's damped-restart pattern; an Aitken extrapolation of the three values (ratio -133.28 / +351.37 = -0.379) puts the limit at about -92,910 (-92,909.8). Three points with a sign-alternating step do not bound the limit; the extrapolation is indicative only.
- **MC (e06):** Heston QE-M, Sobol, 2^21 paths in 64 batches: -93,000.2 ± 106 at 2 substeps, -92,980.3 ± 100 at 4 substeps. Against the Aitken limit of about -92,910, the MC values lie 0.85 s.e. (substeps 2) and 0.70 s.e. (substeps 4) below it, so they are consistent with the fixed PDE's time-converged value. The MC cannot separate the old and new results at fine grids, where they differ by 100-300. At the default grid the old value is about 40 s.e. away. (`outputs/e06_2pow18.txt` is from an earlier 2^18-path / 8-batch version of the script; the committed `e06` runs 2^21 / 64.)
- e03 wall times are inflated because the machine slept during that run; the values are unaffected.

## Decision: fixed (contained)

**Why this is a contained defect:**
- The documented library default is the dual-cell projection. `PDEParams.event_projection` says None means CELL_AVERAGE "on every core route ... 2D Heston/SLV autocallables slice-wise ... the DCN's daily dates" since 0.10.0. `grid/space.py` deliberately does not snap levels for the same reason. The 2D Heston DCN was the one route still on the 0.7.0 masks.
- The nodal path also mis-assigned the inclusive KI / KO nodes through ULP noise, which the 2D snowball's `_event_nodal_mask` explicitly guards against.
- The change is confined to `HestonDCNPDESolver`, plus one opt-out keyword on the shared core whose default runs the existing code.

**Changes** (TDD: the regression test failed first, with a max step of 4,488 against the limit of 250):
- `HestonDCNPDESolver(event_projection=None | EventProjectionMode | str)`:
  - None, i.e. CELL_AVERAGE (the default), applies the 1D `DCNEvents.apply` projection on axis 0 of the (n_x, n_v) surfaces: `project_event_values` for V1's KO, `project_piecewise_event` for V0. The core is built with `pin_critical_levels=False`.
  - `"nodal"` keeps the pre-T14 discretization bit for bit.
- `HestonSLVADICore(pin_critical_levels=True)`: False skips `_pin_grid_targets`. The default is the identical code path, so every other route is bitwise unchanged.
- New tests in `test/test_dcn_vol_pde_solvers.py`:
  - `test_heston_pde_nx_ladder_is_smooth_by_default` (n_x 301 / 401 / 501 at n_v 41, max step under 2.5e-4 of notional, about 9 s)
  - `test_heston_pde_nodal_mode_keeps_the_legacy_discretization` (equals d04's (151, 41) value of -90,111.4177797157)
  - `test_heston_pde_rejects_unknown_event_projection`

### Ladder rerun after the fix (e08; production default, n_v 101)

The values are bit-identical to the e04 cell-average, unpinned arm: 301 -92,971.87; 401 -93,040.10; 501 -93,002.76; 601 -93,028.00; 801 -93,103.71; 1001 -93,152.60; 1201 -93,114.74; 1601 -93,154.52; 2401 -93,168.04; 3201 -93,189.30; 4801 -93,186.21.

- **Monotone on d04's nested ladder.** 301 / 601 / 1201 / 2401 / 4801 gives -92,972, -93,028, -93,115, -93,168, -93,186 (steps -56, -87, -53, -18). Before the fix: +3,588, +2.4, +414, then -88 (2401 → 4801: -93,281.8 → -93,369.3).
- **Non-nested levels wiggle within ±75 instead of ±4,500.** At 401, 501, 801, 1001, 1601 and 3201 each level lands at a different position in its cell, which leaves a second-order remainder under the projection. This attribution is not measured level by level.
- **Not yet asymptotic.** The observed order on the doubling ladder is p ≈ -0.6, 0.7, 1.5, so extrapolation is not supported.

### Moved values (Heston DCN PDE only; every other route bitwise)

| grid (DCN_A, Feller Heston) | before | after | move |
|---|---|---|---|
| (301, 101), the solver's default | -97,286.0115 | -92,971.8705 | +4,314.14 |
| (151, 41), the test / execution-matrix grid | -90,111.4178 | -92,001.9490 | -1,890.53 |

No modelvalidation anchor and no replay golden prices a Heston DCN PDE: neither `quantark/modelvalidation` nor `test/replay_golden` mentions a DCN. The existing Heston PDE vs QE-M MC gate (n_x 501, n_v 201) and the vol-of-vol → 0 match against the 1D route still pass.

## Verification

- `test/test_dcn_vol_pde_solvers.py`: 12 passed.
- The DCN lifecycle / settlement / final-observation / valuation-date / non-trading files plus `test/execution`: 889 passed.
- The ADI-core, Heston / SLV / barrier vol PDE, vol-PDE KI / v0-boundary and remaining DCN test files: 338 passed.
- The full suite was not run here (it runs at the run's final gate).

## Open questions

- The ±75 wiggle on non-nested grids is attributed to each level's cell position under the projection but not measured; a per-level position scan under the projection would confirm it.
- The time axis (+351, -133 at n_x 2401) is now the larger discretization error. It is the same damped-restart pattern d04 saw and was outside this task.
- The 2D snowball / Phoenix Heston and SLV routes still pin levels while using cell-average events. Whether pinning costs accuracy there was not measured, and changing it would move certified values.
- The default n_x 301 is still about 220 from the fine-grid value (0.02% of notional; it was 4,100 off). Whether to raise the default is a separate decision.
