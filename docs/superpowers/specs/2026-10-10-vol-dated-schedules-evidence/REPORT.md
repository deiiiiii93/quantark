# The vol-model routes resolve DATED observation schedules (W12)

Two user-approved fixes, one theme: a vol-model route refused a contract written on dates because it never resolved
the dated schedule against the pricing environment. Branch `w12-vol-dated-schedules`, from main 0d1e4133.

## 1. Refusal 1: the 2D Heston / SLV snowball and Phoenix PDE

**Symptom.** A dated knock-out schedule was refused at any valuation date before maturity: "a date-based knock-out
observation schedule needs a pricing environment to resolve its observation times, but the grid was requested
without one" (`snowball_pde_solver._grid_request_environment`).

**Cause.** `grid_request` takes no environment; the 1D adapter's solve state stashes it first
(`solver._stash_grid_request_env(env)`, `core/adapters/snowball.py`). The 2D solvers call `grid_request` themselves
for their S-axis layout (`_layer_x_nodes`, from `_make_core` on every price) without stashing. Their bump context
(`create_bump_context`) already had the environment by inheritance: `LegacyBumpCloneMixin` runs
`_prepare_for_request` on the clone, which sets it. That is why the 2D snowball / Heston-Phoenix GREEKS of a dated
contract already priced on 0d1e4133 while their PRICE raised.

**Fix.** `_layer_x_nodes` and `create_bump_context` of `_Heston2DSnowballPDEBase` and `_Heston2DPhoenixPDEBase` call
`_stash_grid_request_env` before `grid_request`, exactly as the 1D adapter does (the bump context's is now explicit).

**Other grid_request callers checked.** In `quantark/`: the 1D adapters (stash in the solve state),
`core.prepare.request_geometry` (adapter solve state), `_legacy_bump.create_bump_context` (`_prepare_for_request`
stashes), and the two 2D `_layer_x_nodes` / `create_bump_context` pairs (fixed). The 2D event statistics
(`statistics="flat_vol_surrogate"`) run the 1D march through the adapter (stashes) with the 2D pv (now prices). The
2D greeks: the snowball's one-surface readout runs on the bump context and its `greek_time_grid_policy` already
resolves the schedules with the environment; the Phoenix bump-and-reprices over its bump context. There is no 2D
KO-reset solver. `HestonSLVPhoenixPDESolver._make_core` does not use the grid layer's S-axis (its own
`HestonSLVADICore` grid; a pre-existing difference from the Heston Phoenix and both 2D snowballs), so the SLV Phoenix
already priced a dated contract on 0d1e4133; its cells are unmoved and equal the numeric twin. The DCN 2D solvers do
not call `grid_request`.

## 2. Refusal 2: local-vol / Heston / SLV barrier MC and Heston / SLV barrier PDE

**Symptom.** A DISCRETE barrier on a dated schedule raised "discrete monitoring requires observation_dates".

**Cause.** `barrier_vol_mc_engines._monitoring` and `barrier_vol_pde_solvers._obs_dates` read only the legacy
`product.observation_dates`. `BarrierOption.validate` mirrors a NUMERIC schedule's times there; a dated schedule has
no times until resolved, so the field stays `None`.

**Fix.** New `quantark/asset/equity/engine/vol_barrier_schedule.discrete_observation_times(product, env, maturity)`
(a neutral module: the PDE solvers import no MC module). A schedule that `uses_dates()` is resolved exactly as the
Black-Scholes barrier MC resolves it (`schedule.resolve(env, default_barrier=product.barrier,
default_payoff=product.rebate, require_single=True, product=product)`, `barrier_option_mc_engine._build_discrete_grid`)
and its resolved observation times feed the kernels (MC: snapped to the time grid as before; PDE: `T - t` observe
taus as before). A numeric schedule is read from `observation_dates` verbatim, so a numeric contract's kernel inputs
are byte-identical. The checks after it are the numeric path's, verbatim: no observation -> "discrete monitoring
requires observation_dates"; an observation after maturity (`d > T + 1e-12`, the existing check kept exactly) ->
"observation date ... exceeds maturity". The Heston / SLV PDE's deterministic-variance limit (vol-of-vol below the
kernel floor) already priced through `LocalVolBarrierPDESolver`, which resolves schedules itself.

**Limits kept, identical for both forms.**

| Limit | Where | Dated form |
|---|---|---|
| per-observation varying barriers / payoffs | `_reject_nonuniform_schedule` (MC), `_check` (PDE), before monitoring | refused, same message (`test_vol_barrier_varying_schedule_stays_refused_in_both_forms`) |
| observation after maturity | the route's check on the (resolved) time, or the product's own | refused (`test_vol_barrier_observation_after_maturity_stays_refused_in_both_forms`) |
| a record behind the valuation date | `schedule.resolve` ("end_date must be after start_date") | refused, as on the Black-Scholes barrier MC / PDE and local-vol PDE (measured); a contract aged with `time_shift` has no such record |
| delayed pay-at-hit rebate | `_settlement_scale`, which already resolved the schedule with the environment for both forms | unchanged |
| ended monitoring | `_refuse_ended` / `ended_without_knock_in`, before monitoring | unchanged |
| memory Phoenix on the 2D PDE | `phoenix_vol_pde_solvers.price` | refused (`test_2d_memory_phoenix_stays_refused_in_both_forms`) |

**Settlement per record.** The resolved records carry a settlement time. The vol kernels pay a non-pay-at-hit rebate
at the terminal and never read a record's settlement, for a numeric schedule too (the Black-Scholes MC reads
`settlement_times` only under `pay_at_hit` as well). Under `pay_at_hit`, `_settlement_scale` resolves the same
schedule and refuses any record whose settlement differs from its observation (or a lagged convention), in both
forms. The dated form therefore carries nothing the kernel ignores that the numeric form does not: parity, nothing
silently dropped, no new refusal needed.

## 3. Tests (written first, watched fail, then fixed)

`test/test_vol_dated_schedules.py`, 152 cases. Each contract twice: dated (records on `INITIAL + d_k`,
`d_k = round(k / 12 * 365)`, exercise 365 days on) and numeric (records at `d_k / 365`, maturity 1.0), the numeric
twin aged to the valuation date with `time_shift`. Valuation days 10, 45, 90, 92 (right before / after the day-91
observation), 200. Dated == numeric to rel 1e-10.

- 2D PDE, Heston and SLV: snowball (KO 103 flat 15%, continuous KI 75), snowball with a monthly discrete KI,
  non-memory Phoenix. Price on every day; price / delta / gamma on days 45 and 92; the flat-vol-surrogate statistics
  (pv, KO times / probabilities / survival / discounted KO cash).
- Barrier: LV / Heston / SLV MC, Heston / SLV PDE; up-out call, up-in call (B120), down-out put, down-in put (B85),
  rebate 3, and a pay-at-hit up-out; days 10, 90, 92; price / delta / gamma on day 90. Both forms aged (a dated
  barrier record behind the valuation date is refused on every route, Black-Scholes included).
- A dated discrete-KI schedule with past records is refused on every route too (Black-Scholes PDE / MC / QUAD V2
  measured), so the discrete-KI snowball's dated form is aged as well; the continuous-KI snowball and Phoenix dated
  forms are priced as booked (their past knock-out records are filtered by the autocallable resolver).

Before the fix 127 of 152 failed with the two refusals (the 25 passing: the refusal pins, the SLV Phoenix prices and
the 2D greeks, section 1). After: 152 passed. Twin gaps: 153 of 201 measured values bitwise equal, worst 3.5e-12
relative (a 2D gamma; the last ulp of `d_k / 365 - a / 365` against `(d_k - a) / 365`).

Pins removed in `test/test_expiry_carried_knock_in.py`: `_BEFORE_REFUSED` loses the Heston / SLV MC / PDE
`up_in_call_discrete` cells; `AUTOCALL_BEFORE_REFUSED` loses the 2D snowball / Phoenix dated cells. Those day-before
cells now run and pass (within one day's spot standard deviation of the settled value). No pin in
`test/test_dated_expiry_settles.py` concerned these refusals.

## 4. Moved values (`moved_values.txt`)

`measure.py` on 0d1e4133 (a `git archive` export; `base_0d1e4133.txt`) and on the branch (`after_branch.txt`),
`summarize.py`: 438 values, 175 moved, 0 unexpected. Every moved value is a DATED form, raised -> price; all 219
numeric values are unmoved. The 44 unmoved dated values are those that already priced on 0d1e4133 (2D greeks, SLV
Phoenix price and its day-before cells), bitwise unchanged.

## 5. Numeric schedules do not move

- PDE byte book (`test/pde_bitwise/capture_book.py --workers 2`, 713 rows incl. the Heston / SLV snowball and Phoenix
  rows; base export vs branch, same machine, back to back): `compare.py` 0 unexpected diffs; with `--manifest
  manifest.json` 0 manifest violations. The 10 erroring rows error identically on both sides (manifest-declared).
- `test/modelvalidation/test_anchors.py` + `test_banked_certificates.py`: 30 passed, 29 skipped (superseded
  certificates, checked through their successors). No anchor or golden touched.
- mo_volmodels ADI certifications (`test_adi_greek_certification.py`, `test_adi_slv_aggregate_certification.py`):
  71 passed, 12 skipped (raw stage-16/17 payload artifacts absent from the worktree).

## 6. Test runs (`-n 2`, worktree PYTHONPATH)

| Area | Result |
|---|---|
| `test/test_vol_dated_schedules.py` | 152 passed |
| `test/test_expiry_carried_knock_in.py`, `test/test_dated_expiry_settles.py` | 1118 passed, 2 skipped (the pre-existing BS MC `down_in_put` day-before cells) |
| barrier vol MC / PDE, Heston / SLV / local-vol engines, snowball / Phoenix vol-model engines, vol PDE continuous KI / v0 boundary, terminal settlement, term structure, volmodel risk, snowball PDE date schedule, tracker-clock parity, `test/volmodels` | 355 passed |
| snowball / Phoenix / KO-reset PDE, ADI (8 files), DCN vol PDE, `test/pde_statistics`, continuous-KI scope, event projection, replay statistics refusal, execution PDE adapters / session seams, `test/pde_core`, PDE params surface, OTC vol calibrators, Heston PDE convergence, Heston / SLV / LV PDE kernels, bump-grid steadiness, `test/pde_bitwise` | 1271 passed, 42 skipped |
| modelvalidation anchors + banked certificates | 30 passed, 29 skipped |
| mo_volmodels ADI certifications | 71 passed, 12 skipped |

## 7. Notes and open questions

1. A uniform barrier schedule whose records all carry a barrier different from `product.barrier` is priced at
   `product.barrier` by the vol kernels (`_barrier_kwargs` / `_bkw`), numeric form included; the dated form keeps
   that parity. Pre-existing, not changed: refuse it, or honour the record level?
2. The vol barrier kernels ignore `aggregation_mode` (they price STOP_FIRST_HIT), numeric and dated alike.
   Pre-existing.
3. The Black-Scholes `SnowballMCEngine` gives different values for the dated snowball and its aged numeric twin on day
   92 (12.08 vs 11.34 with 256 paths; equal on day 45), while the BS PDE / QUAD V2 and every W12 route agree. Not a
   vol route, out of W12; measured only.
4. A dated schedule (a barrier's, or a snowball's discrete KI) holding a record behind the valuation date is refused
   on every route ("end_date must be after start_date") unless the contract is aged with `time_shift`; an
   autocallable's dated knock-out schedule is filtered instead. Pre-existing and uniform across routes.
