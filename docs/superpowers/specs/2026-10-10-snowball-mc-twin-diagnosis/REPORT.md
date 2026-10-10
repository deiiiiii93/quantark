# W17 — BS snowball MC: dated vs aged numeric twin gap (diagnosis, 2026-10-10)

Branch `w17-snowball-mc-twin` from main 489cac69. Diagnosis only; no library change. Scripts and raw outputs in this
folder (run as files: `env PYTHONPATH=$WT:$WT/test <venv python> <script>.py`). `common.py` copies W12's contracts from
`test/test_vol_dated_schedules.py` (branch `w12-vol-dated-schedules`): snowball monthly KO 103 (flat 15%), continuous KI
75; Phoenix non-memory KO 103, 1% coupon at 80, continuous KI 70; records on contract days round(k/12·365) = 30, 61,
91, …, 365; numeric twin aged with `time_shift(day/365)` without `keep_landing` (as W12). Flat 20% vol, r 3%, q 1%,
spot 100. (Condensed by the controller from the W17 agent's report; the agent could not write files here.)

## Summary

- **Neither form is wrong.** Both MC forms converge to the PDE / QUAD V2 value; the gap is Monte Carlo noise. On some
  days one form's time grid gains one extra node 1e-16 years wide, which re-deals the draws, so the two forms get
  independent samples of the same price.
- **Root cause.** The MC time grid is a Python `set` of float times. A KO record time and a continuous-KI
  `np.linspace` node can be the same instant in exact arithmetic but one ulp apart (dated `(d_k - a)/365`, numeric
  `d_k/365 - a/365`, linspace a third spelling), so both survive. Grids: `snowball_mc_engine.py:1101-1102`
  (`_build_time_grid`), `:1182` / `:1247-1249` (`_build_time_grid_ko_reset`, incl. REBASED `t_ki + offset`),
  `phoenix_mc_engine.py:529-530`. Affected: BS snowball / Phoenix / KO-reset MC and LV / Heston / SLV snowball and
  Phoenix MC.
- **Proposed fix.** Merge grid nodes closer than `Tolerance.ZERO` (1e-10 y). Twins then agree to 8.9e-15 on all 364
  days. Only prices whose grid had such a step move (one noise re-draw each).

## 1. Reproduction (`w12_config.py`, `repro_256.py`, `mc_convergence.py`)

Seed 1 at 256 paths reproduces W12 on day 92: dated 12.081121, numeric 11.342308 (engine s.e. 0.643 / 0.693; gap 1.1
single-run s.e.; two independent draws differ with sd ≈ 0.95). Other seeds on day 92: +0.705 (7), +0.957 (11), +0.677
(42). Days 10 / 45 / 90 / 200 agree within 3.6e-15. Antithetic on/off irrelevant.

1M paths per form (pseudo: 16 seeds 1000 apart × 65,536; RQMC: 8 reps × 16 batches × 8192, base seeds 1000 apart):

| day | method | dated | numeric | paired gap | PDE | QUAD V2 |
|---|---|---|---|---|---|---|
| 92 | pseudo | 11.69242 ± 0.00642 | 11.68513 ± 0.00995 | +0.0073 ± 0.0074 | 11.67366 | 11.67941 |
| 92 | RQMC | 11.67983 ± 0.00450 | 11.67896 ± 0.00350 | +0.0009 ± 0.0057 | | |
| 14 | pseudo | 10.74989 ± 0.01520 | 10.76375 ± 0.01176 | −0.0139 ± 0.0158 | 10.75510 | 10.75539 |
| 14 | RQMC | 10.76189 ± 0.00469 | 10.75169 ± 0.00649 | +0.0102 ± 0.0097 | | |
| 45 | pseudo / RQMC | 11.13502 / 11.13695 | identical | ≤ 1.8e-15 | 11.13605 | 11.13608 |
| 200 | pseudo / RQMC | 13.71488 / 13.71355 | identical | ≤ 1.8e-15 | 13.71439 | 13.71435 |

The gap shrinks to within 1 s.e. as paths rise.

## 2. Which form is wrong: neither (`references.py`)

BS PDE and QUAD V2 give the two forms the same value within 6.7e-13 on every snowball day measured (10, 14, 41, 45, 53,
90–93, 105, 122, 170, 183, 200, 248). RQMC vs references: day 92 dated +0.0062 PDE / +0.0004 QUAD V2, numeric +0.0053 /
−0.0005; day 14 dated +0.0068 / +0.0065, numeric −0.0034 / −0.0037 — inside the PDE–QUAD V2 band (0.0057 on day 92).

**2.1 Observation-day twin construction (separate; `landing.py`).** Valued ON an observation day, the dated form keeps
the landing record; the twin aged without `keep_landing` drops it: snowball at 104 +1.730 / +1.660 / +1.660 (MC / PDE /
QUAD V2), Phoenix at 100 +1.000 everywhere, at 104 −1.323 / −1.331 / −1.331; with `keep_landing=True` the gap is 0
(≤ 8.9e-16). W12 never values on an observation day.

## 3. Where the forms diverge (`diff_inputs.py`, `grid_scan.py`)

Day 92: T 0.7479452054794521 vs 0.747945205479452; passed / active records, KO times, barriers, payoffs, payment times
equal to ≤ 1.1e-16; **grid 196 vs 197 nodes, smallest step 9.1e-4 vs 1.11e-16**; KO indices +1 after the fifth record.
Continuous-KI grid `np.linspace(0, T, n+1)[1:]`, n = int(252T)+1 = 189; the day-274 record is 182/273 = 2/3 = node
126/189; linspace node 0.49863013698630126 = dated 182/365 bit-identically; numeric 274/365 − 92/365 =
0.4986301369863014, one ulp away; `set(grid_times) | {T}` keeps both.

Spurious steps (all KO time vs KI linspace; maturity never collides): snowball and Phoenix — dated days 14, 53, 61,
105, 144, 209; numeric 41, 61, 92, 170, 183, 209, 248, 261. Continuous-KI KO-reset (days 1–181) — dated 26; numeric 26,
65, 78, 82.

Why a 1e-16 node changes the price: pseudo draws `standard_normal((n_paths, n_steps))` row-major
(`montecarlo/qmc_sobol.py:86`); the KI bridge draws `rng.random(idx.size)` per step (`snowball_mc_engine.py:1673`);
RQMC / QMC dimension = step count (`_rqmc_dimension`, `:487-492`). The law is unchanged (no variance, bridge
probability 0; KO/KI ordering preserved).

Runtime merge patch (`patch_grid.py`): day 92 seed 1 numeric → 12.081121 = dated; W12 snowball all 364 days twin gap
≤ 8.9e-15 (256 paths, seed 7) and ≤ 3.6e-15 (4096, seed 11), before up to 0.98 on 11 days; Phoenix remaining gaps are
only the §2.1 landing coupons.

## 4. Other engines (`other_engines.py`)

Days 14, 41, 45, 53, 92, 170, 200; gaps exactly on collision days, ≤ 3.8e-15 on 45 / 200. Largest gaps (seed 7 × 256
unless noted): BS MC (seed 11 × 4096) snowball 0.198 (92), Phoenix 0.104 (53); LV MC 0.298 (170) / 0.327 (92); Heston
MC 1.009 (53) / 0.730 (53); SLV MC 0.533 (170) / 0.365 (170). KO-reset BS MC: discrete KI none; continuous KI −0.027
(65), +0.017 (78), −0.236 (82). LV / Heston MC on a continuous-KI KO-reset raise on both forms every day ("recorded step
variance (256, 127) does not describe these paths (256, 121)") — a separate pre-existing bug.

## 5. Proposed fix and moved values

Helper `merge_time_nodes(times, tol=Tolerance.ZERO)`: sort the union, collapse runs closer than `tol` into the LAST
member (maturity stays the exact final node), map each observation time to the first merged node at or after
`t − tol`. Use at `snowball_mc_engine.py:1101-1102`, `:1182` / `:1247-1249`, `phoenix_mc_engine.py:529-530`. 1e-10 y
(~3 ms) is far below the smallest legitimate step measured (7.2e-5 y). Narrower alternative: snap KI linspace nodes onto
observation times (misses the REBASED collision).

Moves only where a grid had a sub-tolerance step; identity elsewhere (bitwise). W12 snowball BS MC: dated 6 days,
numeric 8 days, up to 1.873 at 256 paths / 0.338 at 4096 (day 61); Phoenix up to 1.155 / 0.115. Existing tests
(`w17_grid_probe.py`, `run_probe.sh detect`, 70 files, 2461 passed / 48 skipped): no snowball / Phoenix grid had such a
step; 2 REBASED KO-reset grids did (step 2.8e-17): `test_ko_reset_event_stats_shape[Rebased]` (shapes) and
`test_cashleg/test_ko_reset_distribution.py::test_mc_rebased_is_refused` (refusal) — no price pinned. Not probed:
`test/mo_volmodels`, `test/modelvalidation`, `test/pde_grid`.

## 6. Open questions (recommendations)

1. Merge in the engine (shared helper) rather than canonicalising time spellings upstream (cannot be guaranteed); add
   twin tests on collision days 14, 41, 53, 61, 92, 170, 209 for snowball, Phoenix, continuous-KI KO-reset MC.
2. MC twin tolerance after the fix: rel 1e-10, as PDE / QUAD.
3. Run the detect probe over modelvalidation / mo_volmodels / pde_grid before landing; any engaged cell needs
   re-certification, not a tolerance bump.
4. LV / Heston MC continuous-KI KO-reset variance-shape error (`snowball_mc_engine.py:1976-1979` vs
   `snowball_vol_mc_engines.py:326-331`): own ticket; slice the recorded variance to the KI horizon.
5. Twins valued on an observation day must age with `keep_landing=True` (W11 does; W12 never values there).
6. Same set-union in `range_accrual_mc_engine.py:325` and `asian_option_mc_engine.py:337`: not measured; use the helper
   if a twin test shows a collision.
