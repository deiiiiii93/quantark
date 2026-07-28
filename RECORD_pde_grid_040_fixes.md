# Record — three PDE grid-layer defects found and fixed (0.4.0)

**Status: all three fixed and verified, 2026-07-28.**
Regression guard: `repro_grid_040.py` (exits non-zero if any regresses).

## Background

Found while `otc-price-adapter` (the OTC/BCT book pricer) migrated from
`quantark==0.3.0` to `0.4.0`. 88 of its 99 positions price through
`SnowballPDESolver` / `PhoenixPDESolver`, so the declarative grid layer
(spec `docs/superpowers/specs/2026-07-27-pde-grid-redesign-design.md`) sits on
its critical path.

The redesign was never in question — it is a real accuracy win. Validated
against `PhoenixQuadEngine` / `SnowballQuadEngine` at high `grid_points`, the
0.3.0 solver carried a **+3.02%** bias on one Phoenix that 0.4.0 removes. The
three defects below were what blocked adoption, not the method.

## The defects

### 1. `accuracy="standard"` failed open

The **default** profile (`points=400`) returned a sign-flipped price on a 2.6y
部分保本 Phoenix (¥50m notional, 32 discrete KO observations):

| profile | points | PV | err vs converged quad |
|---|---|---|---|
| `fast` | 200 | −1,018,785 | −3.35% |
| **`standard`** | **400** | **+165,132** | **+116.75%** |
| `high` | 800 | −998,885 | −1.33% |

Converged reference: **−985,774**. The default reported a +165k asset where the
position was a −986k liability — ¥1.15m of error on ¥50m notional, gamma 19x
too large. Being *worse than a coarser profile* ruled out simple
under-resolution. `space.py` logged `achieved spacing 97.19705 exceeds 2x
target eps_crit 0.00300 (accuracy degradation, not an error)` at
`logger.warning` only, so a caller taking the documented default shipped a
wrong number silently.

### 2. `_auto_bounds` expanded for unreachable critical prices

Every critical price widened the domain unconditionally, with no reachability
test. The adapter pads already-passed KO observations with an unreachable
sentinel (`initial * 100.0` — "this date can never knock out"). Spot 8,601,
sentinel 832,341:

| critical prices | domain | log-span |
|---|---|---|
| real barriers only | [1001.8, **71,467.9**] | 4.27 |
| + sentinel 832341.0 | [1001.8, **844,901.3**] | 6.74 |

The 4σ envelope already reached 71,468 (8.3x spot); everything above was dead
domain, absorbing ~37% of the nodes. `_local_eps` then reported achieved
spacing *at the sentinel*, producing the misleading `97.19705` above and
masking the spacing at the barriers that mattered.

0.3.0 clamped bounds in `_resolve_spatial_bounds`, so this only became
reachable once that method was deleted in 0.4.0.

Independent of defect 1 — pinning the domain manually with
`GridConfig(bounds=(1001.8, 71467.9))` at `accuracy="standard"` still left the
error at +110%.

### 3. Solve cost was non-monotonic in `points`

Same product, same `steps_per_day=8`, varying only `points`:

| points | wall clock | err vs quad |
|---|---|---|
| 1001 | 8.95s | −1.47% |
| 2000 | 18.12s | −0.26% |
| 3000 | 26.71s | −0.09% |
| **3500** | **0.76s** | +0.06% |
| 4000 | 0.84s | +0.11% |

A **35x speedup from asking for more nodes**. In the slow band numpy emitted
`RuntimeWarning: overflow encountered in square` and `divide by zero` from
`_ode_f` (`space.py:52,54`), inside `_concentrated_mesh`'s beta bracket search.

## What changed

Read from the diff at the level of introduced/removed symbols:

- **`grid/space.py`** — `_reachable_critical_prices()` gates which criticals may
  move the bounds (defect 2); `_equidistributed_mesh()` and
  `_stable_concentrated_mesh()` replace the RK4 ODE bracket search on the
  primary path, with the old machinery retained as `_legacy_ode_*` /
  `_legacy_concentrated_mesh` (defects 1 and 3); supporting
  `_anchor_envelopes()`, `_integration_support()`, `_monitor_values()`,
  `_worst_spacing()`.
- **`grid/binder.py`, `base_pde_solver.py`** — `SpatialLayout` now carries
  `active_critical_prices` / `ignored_critical_prices`, and the frozen-layout
  coverage check skips deliberately-ignored out-of-reach markers so bump
  contexts do not fail closed on them.
- **Quad engines** — the same reachability and projection thinking applied:
  new `QuadParams.filter_unreachable_barriers` (default `True`),
  `event_projection` (default `CELL_AVERAGE`), `integration_rule` (default
  `"trapezoid"` — phase-stable after discontinuous autocallable events;
  `"simpson"` is the legacy weighting), plus `auto_converge` with
  `convergence_rel_tol` / `convergence_abs_tol` / `max_convergence_grid_points`.
- **Spec** amended (`2026-07-27-pde-grid-redesign-design.md`).

## Verification

`repro_grid_040.py` — PASSED, exit 0:

| check | before | after |
|---|---|---|
| `accuracy="standard"` | +116.75% (sign flip) | **+0.17%** |
| sentinel effect on domain | 71,468 → 844,901 | **no change** (4.27 both) |
| cost, points 1001→6000 | 8.95s → 0.76s (35x inversion) | **0.29s → 1.21s, monotone** |

No RuntimeWarnings. `standard` went 3.45s → 0.12s.

Four real book positions, vs a **re-derived** quad reference at
`grid_points=32001` (re-derived because the QuadParams default changes above
moved it):

| position | 0.3.0 | `fast` | `standard` | `high` | p4000/spd8 |
|---|---|---|---|---|---|
| Phoenix 部分保本 2.6y | +0.58% | +0.00% | +0.17% | +0.15% | +0.11% |
| Snowball 部分保本 2.7y | +0.59% | +0.12% | +0.56% | +0.63% | +0.48% |
| Phoenix 非保本 2.8y | **+3.02%** | −0.12% | +0.07% | +0.04% | +0.06% |
| Snowball 非保本 2.9y | +0.29% | +0.12% | +0.18% | +0.20% | +0.14% |
| **worst** | **3.02%** | 0.12% | **0.56%** | 0.63% | 0.48% |

Cost, full rows including greeks (4 positions): quantark 0.3.0 production
config **43.4s**; 0.4.0 `accuracy="standard"` **23.0s**; `accuracy="high"`
**70.3s**. `standard` is now both more accurate and ~1.9x faster than 0.3.0.

## Open notes

- **`fast` topping the table is cancellation, not quality.** A convergence
  sweep on the Snowball 部分保本 position gives points 400→+0.14%, 1000→+0.61%,
  2000→+0.69%, 4000→+0.55%, 8000→+0.41% — non-monotone, so `fast`'s number is
  a coincidental offset against a systematic positive bias. Do not select a
  profile on that evidence.
- **One position has a residual ~0.4–0.6% gap.** Every PDE setting on the
  Snowball 部分保本 position sits above quad and descends only slowly with
  resolution. Quad is itself only first-order there (128001 still moving
  −0.024%/step, converging from above), so part of the gap is reference error.
  Worth a look; it is 0.5%, not 3%.
- **The quad reference is not version-stable.** It was byte-identical between
  0.3.0 and 0.4.0, which is what made it a good arbiter — but the QuadParams
  default changes moved it from −985,774 to −985,949 on the repro position.
  Re-derive it whenever the quad engines change.

## Re-checking

```bash
.venv/bin/python repro_grid_040.py      # exits non-zero on regression
```

`ACC_TOL` in that file (currently 1%) is a policy call, not a fact — it is set
to survive legitimate numerical drift while still catching a blow-up.
