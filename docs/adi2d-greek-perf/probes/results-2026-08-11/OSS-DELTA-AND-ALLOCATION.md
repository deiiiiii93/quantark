# OSS helps delta too, and that decides the certification allocation

**Date** 2026-08-11 · Probe `probes/probe_oss_delta_gate.py` · Raw
`output/oss_delta_gate/oss_delta_gate.json` · Consumes the
mc2d-gamma-convergence prototype verbatim (no edits to that session's files).

## Why this measurement exists

The mc2d-gamma-convergence session established that gamma's non-convergence is
an estimator problem: FD-with-CRN on KO/KI indicators gives
`stderr(gamma) ~ J*h^(-3/2)/sqrt(N)`, so RMSE bottoms out at `N^(-2/7)` -- 11x
the paths to halve the error. Its one-step-survival (OSS) estimator removes that,
measured at 7.4x / 34x / 101x gamma variance at h = 1% / 0.3% / 0.1%.

But its gate 3 measured **gamma only** on the real fixture; the delta figure
(125x) came from a two-observation Black-Scholes reduction, the most favourable
geometry available. Delta is not a side concern for certification:

- Feeding the measured gamma gains into the stage-16 sizing collapses every
  gamma row to the batch floor and leaves **delta binding** (`heston/near_ki`
  256, `heston_slv/low_feller` 256).
- The aggregate gate is **delta-only** (`mean_signed_delta_bias`) -- the gate
  whose interval width made the SLV route INCONCLUSIVE.

So the certification's post-OSS cost is set by a number nobody had measured on a
daily-monitored snowball.

## Result: OSS wins on delta as well

10 seeds x 50k paths per point, matched bumps, delta and gamma read off the
**same** three prices per seed (so the comparison shares its randomness).

| bump | delta engine | delta OSS | delta var ratio | gamma var ratio |
|---|---|---|---|---|
| 1.0% | +0.1049 ± 0.0063 | +0.1115 ± 0.0037 | **2.85x** | 7.42x |
| 0.3% | +0.1089 ± 0.0074 | +0.1102 ± 0.0039 | **3.67x** | 33.32x |
| 0.1% | +0.1144 ± 0.0153 | +0.1119 ± 0.0042 | **13.33x** | 101.35x |

The gamma column reproduces that session's 7.4 / 34 / 101 to three digits, which
is the harness's own validation: same fixture, same pricer, independent driver.

Delta's gain is smaller than gamma's, as theory predicts -- plain-FD delta
suffers `J^2/h` rather than `J^2/h^3`, so there is less indicator noise to
remove -- and it grows as the bump shrinks, for the same reason. OSS does pay a
small tax on smooth functionals (that session measured PV stderr 1.0166x worse,
because survival weights carry their own variance), but on delta that tax is
swamped many times over by what the smoothing removes.

**Recorded so it is not repeated:** a first pass at 4 seeds x 15k produced delta
ratios of 0.48 and 0.35 and the confident but wrong conclusion that OSS *hurts*
delta. A variance ratio from 4 seeds has 3 degrees of freedom and is worthless;
the PV-tax argument then made the noise look mechanistic. Variance-ratio claims
need the full seed count before they mean anything.

## What it does to the certification allocation

Applying the measured h=0.3% gains (delta 3.67x, gamma 33.32x) to the 32-batch
pilot evidence, through the certification's own gate arithmetic
(`probes/size_allocation.py`):

| variant | frozen | derived | reduction |
|---|---|---|---|
| heston (7 cells) | 8192 | **288** | **28.4x** |
| heston_slv (6 cells measured) | 1152 | **384** | **3.0x** |

Per-cell, everything lands at 32-64 batches. `heston/near_ki` -- the cell whose
frozen 2048 batches projected to ~17 h of wall clock, and the one cell a deep
pilot proved could *not* be cut on the current estimator -- comes down to 64.

The aggregate gate needs 32 (heston) / 64 (SLV). At that point the binding
constraints are `MIN_PRODUCTION_RQMC_BATCHES = 16` and the deterministic PDE
envelope, **not MC noise** -- which is the right end state for a certification
whose subject is the PDE discretization.

## Caveats that bound the claim

1. **The gains are measured on one fixture** (2y snowball, monthly KO 103, daily
   KI 75, Feller-violated Heston v0=0.04 kappa=1.5 theta=0.05 sigma=0.6
   rho=-0.6). The certification spans 7 regimes x 2 variants. That session's
   §5.3 warns that v->0 / Feller-violated fits steepen the `Phi` arguments and
   weaken smoothing locally, so `sigma_collapse` and `low_feller` should be
   expected to gain less. The table above assumes uniform gains and is therefore
   optimistic per-cell; a post-port pilot must re-measure per cell.
2. **SLV is unvalidated.** §5.3 flags that stable second derivatives need C^1
   leverage interpolation in S, and `LeverageSurface` is bilinear -- kinks on
   grid lines would leak into gamma. The SLV column above inherits a
   Heston-measured gain.
3. **OSS is a prototype, not the engine.** It is a runtime-patched standalone
   pricer. Using it for certification references means porting it behind an
   opt-in flag, with the mirror-parity, price-parity and h-stability gates
   re-run in-engine. It is not bitwise by construction (it changes the estimator
   distribution while preserving the mean), so it needs a default-off proof.
4. **My pilot floors are noise-inflated.** `|diff|` and `|substep_mean|` at 32
   batches carry standard errors comparable to themselves, and `|E|` of a noisy
   estimate is upward-biased, so the derived counts are conservative in that
   direction.
5. **SLV `near_ki` is pinned at 256** by the multilevel estimator's own guard
   (primary batches must equal the declared profile and be a multiple of 128),
   regardless of any variance gain.

## Consequence for the "further convergence" question

The unbuilt lever that session flags -- monthly-window preintegration of the
KO-only run, up to ~96x on the KO-leg component -- targets **gamma**. After OSS,
gamma rows already sit at the batch floor, so that work would buy the
certification nothing. Same now applies to further delta work: at 3.67x, delta
also reaches the floor.

The remaining headroom is not in MC variance at all. It is in the deterministic
PDE envelope (0.0269 heston / 0.0230 SLV against a 0.1 aggregate bound) and in
the substep bias terms. That is where "improve convergence further" should point
next, and it is PDE-side work, not estimator work.
