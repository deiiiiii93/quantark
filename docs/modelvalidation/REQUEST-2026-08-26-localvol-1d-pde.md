# Request: certify the 1-D local-vol snowball PDE engine

**Raised** 2026-08-26, from the snowball vol-model backtest study
(`example/mo_volmodels/`).
**Engine** `LocalVolSnowballPDESolver`
(`quantark/asset/equity/engine/pde/snowball_vol_pde_solvers.py:69`).
**Procedure** `docs/modelvalidation/RELEASE_PROCEDURE.md`.

---

## Why this is being asked for

Gate G2 routes `localvol` to **Monte Carlo**, alone among the study's six
variants. `flat_bsm`, `flat_bsm_quad`, `ts_bsm`, `heston` and `heston_slv` all
route to a deterministic engine. That leaves one MC-priced arm inside a
six-arm *model* comparison, so an engine difference is confounded with the
model difference the study exists to measure. It also costs more.

The route is not a preference. `decide_route` needs PV **and** delta to pass,
and localvol's delta does not:

```
PV     15/15 cells pass at coarse, medium and fine;  max residual 0.099%  (tol 0.25%)
delta  delta_pass: false
       3 of 8 cells exceed the 0.5-contract per-cell bound
       worst  -1.2726 contracts  at 2024-02-08
       mean signed  -0.0551  ->  INSIDE the 0.1-contract bias bound
```

So there is no accumulating bias — `delta_biased: false`. Individual dates
blow the per-cell bound. `delta_required: true` and
`delta_authority: "stage11"`, because unlike `heston` / `heston_slv` there is
no banked certificate for 1-D local vol to delegate to. Stage 11's own MC
reference is the sole authority, it failed, and the route falls back.

Source: `output/pde_convergence_gate/gate_decision.json`, key
`variants.localvol`.

## What the disagreement looks like

Every `full` cell is a freshly-struck 3-year snowball with
`valuation == inception`, so spot always sits at 100% of its own barriers —
nothing here is near a barrier. What varies date to date is the **local-vol
surface**.

```
date          s0    PV residual: coarse / medium / fine     PV sigma   delta contracts
2024-02-08  4993   -0.09907  -0.09902  -0.09945               2.88        -1.2726
2024-06-14  5181   -0.06423  -0.06465  -0.06528               1.82        -0.6006
2025-01-13  5559   +0.03553  +0.03463  +0.03341               1.46        +0.6998
2024-10-10  5655   -0.02065  -0.02136  -0.02153               0.77        -0.3079
2023-11-15  6207   -0.01619  -0.01754  -0.01801               0.65        +0.2614
2023-05-15  6586   +0.02218  +0.02042  +0.01933               0.91        +0.1337
2025-04-09  5652   +0.00213  +0.00166  -0.00034               0.05        +0.1826
2026-07-15  7818   -0.00322  -0.00336  -0.00353               0.08        +0.4623
```

Three things to take from this table:

1. **The residual is flat under 3x grid refinement, on every date.** Coarse,
   medium and fine agree to the fourth decimal. Discretization error shrinks;
   this does not. The PDE has converged.
2. **`|PV residual|` and `|delta contracts|` track each other** down the whole
   table. One cause, not two.
3. **The same disagreement passes one gate and fails the other.** The PV
   tolerance is 0.25% of notional and the residual maxes at 0.099%. Expressed
   as a hedge ratio, that identical disagreement is 1.27 futures contracts
   against a 0.5 bound. PV barely notices what delta calls fatal.

Worst cell is 2024-02-08 — the CSI1000 crash bottom, the most extreme
local-vol surface in the sample.

---

## Do this cheap test FIRST. It may make the certification unnecessary.

**The gate refines only one side of the comparison.**

The PDE ladder walks `accuracy` fast -> standard -> high. The MC reference
does not move: `MC_FULL = {"paths_per_batch": 8192, "batches": 16,
"substeps_per_interval": 4}` (`11_pde_convergence_gate.py:177`) is used at
every level. In the banked artifact, `mc_ref` is byte-identical across the
three grids — `-360.29042974897817` three times for 2024-02-08 — as is
`mc_se`.

So "flat under refinement" proves the **PDE** converged. It says nothing about
whether the **MC** converged, because the MC was never refined. And `mc_se` is
a *sampling* standard error: it measures path noise and is structurally blind
to the reference's own time-discretization bias at
`substeps_per_interval = 4`.

**There is direct precedent in this repository for the reference being the
guilty party.** A suspected 2-D PDE bias was exonerated when MC stride bias
was found — `time_steps` turned out to be inert in the vol MC engines — and
`substeps_per_interval` shipped as the fix in `8d9b987`. See the PDE auto-grid
root-cause work.

### The test

On 2024-02-08 `full` (and 2024-06-14 as a second point), hold the PDE fixed at
`accuracy: standard` and refine the reference instead:

```
substeps_per_interval:  4  ->  8  ->  16  ->  32
```

Then read which side moves.

- **`mc_ref` migrates toward the PDE and the residual collapses** -> the PDE
  was never wrong, the gate's reference was under-resolved, and the fix is to
  raise `MC_FULL["substeps_per_interval"]` and re-run G2. localvol routes to
  PDE and no certification is needed.
- **`mc_ref` is stable and the residual survives** -> the disagreement is
  real, the reference is trustworthy at this configuration, and the full
  certification below is warranted.

This is minutes-to-hours. The certification is hours-to-days. Do not spend the
second before knowing the answer to the first.

**A hypothesis already ruled out — do not re-chase it.** A discrete-vs-
continuous knock-in mismatch was the obvious first guess and it is wrong. The
study's product sets `ki_continuous=False`,
`ki_observation_type=ObservationType.DISCRETE` with 732 observation dates
(`11_pde_convergence_gate.py:500`), so `LocalVolBarrierCrossingMixin`'s
FIRST_PASSAGE correction is not active on these cells at all.

Other live candidates, if the reference proves clean: KI observation dates
projecting onto the PDE time grid versus the MC's exact dates; and Dupire
surface interpolation read on a PDE `(S,t)` mesh versus along MC paths, which
would diverge most where the surface is steepest — consistent with 2024-02-08
being the worst cell.

---

## If certification is warranted: what is missing

`quantark/modelvalidation/` has no local-vol arm. Concretely:

| Piece | Status |
|---|---|
| `equity.snowball.heston_pde`, `equity.snowball.heston_slv_pde` candidates | exist (`builders/equity_snowball_vol.py`) |
| **local-vol candidate builder** | **absent — must be written** |
| **local-vol environment builder** (a Dupire surface, the way `equity.snowball.heston_flat_market` supplies Heston params) | **absent — must be written** |
| `equity.snowball.vol_multilevel_rqmc` reference | exists but **refuses to run** |
| `LocalVolSnowballMCEngine` | exists (`engine/mc/snowball_vol_mc_engines.py:627`) |

The reference is the real obstacle. `VolSnowballExternalReference` is
"declared, not run" by design — it records the archived 28.6-hour multilevel
telescope and raises rather than standing in for it. A local-vol study
therefore needs a **runnable** reference builder wrapping
`LocalVolSnowballMCEngine`, in the shape of `equity.snowball.mc_rqmc`
(`builders/equity_snowball.py:544`).

Whatever that reference is, **its own convergence must be demonstrated, not
assumed** — refine `substeps_per_interval` until the estimate stops moving,
and record the level at which it stopped. That is the defect this whole
request traces back to.

### Study definition

Model `example/modelvalidation/snowball_flat_bsm.yaml`. Keep the desk bounds
the other snowball studies use, and which stage 11 uses:

```yaml
bounds:
  cell: 0.5              # rounding provably absorbs less than this
  mean_signed_bias: 0.1  # accumulates over ~700 rebalances
quantities: [pv, delta, gamma]
```

Cases must include the surfaces that actually fail, not only synthetic ones.
The four dates above at their real calibrated Dupire surfaces are the point of
the exercise; **2024-02-08 is mandatory**. A study that certifies local vol
only on benign surfaces would admit the engine without touching the regime
that put it on MC.

Note for scope: the certificate covers only the configurations its YAML names.
That lesson is banked from the 2026-08-19 variant-surface amendment, which
found three defects hiding in configurations the original certificate never
enumerated.

---

## What lands when this is resolved

- Either `MC_FULL` gains a resolved `substeps_per_interval` and G2 is re-run,
  or a `localvol-1d-pde` certificate is banked under
  `docs/modelvalidation/certificates/`.
- If certified, stage 11 gains `delta_authority: "stage16"`-style delegation
  for `localvol`, exactly as `heston` / `heston_slv` have.
- `localvol` routes to PDE, and the study's six arms are all deterministic.
- The fleet's localvol cost projection is re-measured; the 40.6 CPU-hour
  figure on record was measured under `route=pde` and does not describe the MC
  route it currently uses.

## Pointers

- Gate decision: `output/pde_convergence_gate/gate_decision.json`
  -> `variants.localvol`
- Full cells and delta rows: `output/pde_convergence_gate/pde_convergence_gate.json`
  -> `dates[].cases.full.{cells,deltas}`, filter `variant == "localvol"`
- Admission rule and bounds: `example/mo_volmodels/11_pde_convergence_gate.py`
  -> `DELTA_CELL_CONTRACTS`, `DELTA_BIAS_CONTRACTS`, `delta_cell_passed`,
  `detect_delta_bias`
- Engine pair under test: same file, `GATE_PAIRS["localvol"]` (line ~726)
- Procedure: `docs/modelvalidation/RELEASE_PROCEDURE.md`

---

## Resolution, 2026-08-28

**The cheap test was run first, as this request insisted, and it took the first
branch.** `FINDING-2026-08-26-localvol-1d-pde.md`: `mc_ref` migrated toward the
PDE and the residual collapsed. The reference had been under-resolved twice
over — `substeps=1` where 4 was declared, and still delta-unconverged at 4 —
and the delta admission rule carried no reference-uncertainty term. By this
request's own criterion, **no certification was warranted.**

**It was nevertheless carried out**, for the three things §"What lands" names
that a routing decision cannot deliver: banked schema-versioned evidence,
`delta_authority` delegation, and CI anchors. That is a different goal from
settling a correctness question, and the certificate says so plainly.

Banked at `docs/modelvalidation/certificates/snowball-localvol-1d/2026-08-28/`,
digest `931345b9ae5e684910b1e85be5f3376522af2b19ecc4548ee918764631eae06c`.
`LocalVolSnowballPDESolver` is **ADMITTED** on 48 gated cells across two real
calibrated Dupire surfaces, 2024-02-08 included as this request required.
Budget consumed: median 5.6%, max 40.8%.

Three of this request's instructions shaped the result:

- *"its own convergence must be demonstrated, not assumed"* — a substeps
  4/8/16 ladder was walked at this study's own scale rather than inherited from
  the FINDING's three-year trade. 8→16 shifts by 0.06σ and 0.55σ.
- *"2024-02-08 is mandatory"* — it is, at its real calibrated surface, paired
  with a calm contrast so the steepness mechanism is tested rather than assumed.
- *"the certificate covers only the configurations its YAML names"* — the scope
  exclusions are written into both the study file and the certificate README.

**Still outstanding, and not addressed by this certification** (both inherited
from FINDING §7): the fleet's existing localvol numbers were produced under the
biased `substeps=1` MC route and must be discarded, and the 40.6 CPU-hour cost
projection needs re-measuring.
