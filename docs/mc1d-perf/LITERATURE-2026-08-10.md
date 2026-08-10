# Literature techniques worth borrowing for the 1D MC engines

**Date:** 2026-08-10. Companion to `DECISION-2026-08-10.md`. The decision
matrix covered techniques migrated from our own 2D program; this survey covers
what the Monte Carlo derivatives literature offers beyond them, classified by
how each item fits QuantArk's conventions (bitwise-provable implementation
change vs opt-in estimator change needing re-baseline vs strategic program).

Already in the stack, so NOT proposed: antithetic variates, GBM control
variate, constant-drift importance sampling shift, Owen-scrambled Sobol RQMC,
Brownian-bridge path construction, Brownian-bridge barrier survival weights
(= conditional MC smoothing of the barrier indicator), QE/QE-M, pilot-frozen
Neyman batch allocation, and an MLMC-style coupled coarse/fine ladder in the
DCN vol engines (shared-draw pairing, measured ~4x difference-variance
reduction under QE-M).

## Tier 1 — estimator-preserving implementation ideas (bitwise-gateable)

**T1a. Single-pass fused path+payoff kernels (cache blocking).** The C3 demo
failure showed the regime: at 100k x 252, any design that materializes
(n_paths, n_steps) matrices pays ~200 MB per temporary and loses to
cache-resident code. The standard HPC answer is to stream path *blocks*
through generation -> barrier scan -> payoff in one fused pass, never storing
the full path matrix (barrier/autocallable payoffs need only running state:
extrema, first-hit index, coupon accruals). Natural extension of accepted
candidate 2; same bit-identity contract (per-path sequential order preserved;
reductions stay in NumPy). Expected value: removes the ~38% path-build +
scan bandwidth from snowball MC; unmeasured — spike before deciding.

**T1b. Counter-based RNG substreams (Philox; Salmon et al., SC'11).**
`numpy.random.Philox` gives O(1) jump-ahead and worker-count-independent
reproducible parallel streams — the right foundation if the dask batch layer
ever parallelizes a *single* pricing. Switching an engine's default stream is
a re-baseline, so this is only for new parallel modes, not a retrofit.

## Tier 2 — unbiased estimator changes (opt-in mode + re-baseline)

**T2a. One-step survival conditioning (Glasserman–Staum 2001; Alm–Harrach;
Rakhmonov 2019). The headline item.** At each KO observation date, instead of
sampling the KO indicator, multiply the path weight by the one-step survival
probability and condition the next draw on survival (truncated normal /
inverse-CDF restriction). Unbiased; removes the Bernoulli noise of KO events
that dominates autocallable estimator variance. Published numbers for
autocallables: ~3x variance reduction on price and >500x on first-order
Greeks — precisely our pain point (bump-revaluation Greeks on snowball MC are
noise-limited). Interacts correctly with our continuous-KI bridge weights
(both are conditional-MC smoothings; the KI side already uses survival
*weights* in `bridge_survival`, while the snowball engine's KI check *samples*
hits — the sampled variant would need the weighted form for full effect).
Estimator changes => new goldens, opt-in flag, exact-semantics default
untouched.

**T2b. Terminal stratification / Latin hypercube (Glasserman 2004, Ch. 4).**
Stratify the terminal Brownian coordinate (the first bridge coordinate we
already generate) across paths. Unbiased, nearly free, complements antithetic;
strongest for payoffs with a dominant terminal component (euro legs, sharkfin,
DCN coupon legs), weaker for KO-dominated payoffs. Cheap spike.

**T2c. Optimized importance sampling for rare KI (Glasserman–Heidelberger–
Shahabuddin 1999).** We ship a *constant* user-chosen drift shift; the
literature chooses the tilt by variance minimization (saddlepoint / adaptive
pilot). For deep KI barriers (75% and below) the KI-conditional cells are
exactly the variance-dominant cells the reference program fights with brute
force. Rakhmonov combines this with T2a for worst-of autocallables. Opt-in.

**T2d. Multilevel Monte Carlo (Giles 2008; Giles–Debrabant–Rössler for
barrier/digital payoffs via bridge conditioning).** Cost to RMSE ε drops from
O(ε^-3) to ~O(ε^-2); published barrier-option results ~200x at ε = 0.01.
Only pays where discretization bias forces fine grids: the LV kernels
(log-Euler) and the MC reference stacks (overnight G1-style budgets), not the
exact-in-distribution BSM engines. We already own the hard part culturally:
the DCN coupled ladder IS an MLMC level pair (shared-draw coupling, measured
4x). Formalizing = level schedule + bias/variance estimator + aggregator.
Substantial but well-trodden; MLQMC variant composes with our RQMC layer.

**T2e. Milstein for the 1D LV kernel.** In 1D there is no Lévy-area
obstruction; the correction needs only dσ/dS off the LV grid. Main value is
NOT standalone bias (weak order stays 1) but strong-error decay: it lifts
MLMC level-variance decay from O(h) to O(h^2) (Giles 2013), making T2d
markedly cheaper. Bundle with T2d, not alone.

## Tier 3 — strategic Greeks program

**T3. Pathwise/adjoint Greeks (Giles–Glasserman "Smoking Adjoints", 2006) +
Vibrato MC (Giles 2008) for discontinuities, on top of T2a smoothing.**
Adjoint pathwise gives all first-order Greeks for ~O(1) extra cost per path,
vs one full re-pricing per bump today. Discontinuous autocallable payoffs
block naive pathwise; the literature's resolution is exactly T2a (the
conditional estimator is differentiable — Alm–Harrach) or Vibrato. This is a
program, not a task: it touches estimator, engines, and the BumpConfig
contract. Record as the long-run destination that T2a unlocks.

## Tier 4 — QMC construction experiments (cheap, measured)

**T4. PCA / linear-transform path constructions (Imai–Tan 2006; fast
orthogonal transforms, Irrgeher–Leobacher; Wang–Sloan on discontinuous
integrands).** Alternatives to the Brownian bridge ordering; payoff-adapted
rotations can cut effective dimension further, BUT the discontinuity
literature warns the bridge/PCA orderings can *hurt* barrier-style payoffs —
which matches our own finding that bridge8 helped only variance-dominant
cells (2.14/2.62/1.49x, not the hoped 8x). A half-day measured experiment on
the existing generator scaffolding; adopt only what the numbers support.

## Explicitly not recommended now

- **Broadie–Glasserman–Kou 0.5826-shift continuity correction**: we already
  price discrete/continuous monitoring exactly via bridge survival; the shift
  is an approximation with no place under exact-semantics-by-default (except
  as a cross-check oracle).
- **Neural control variates / learned pricers**: active 2024–26 literature,
  but a heavyweight dependency with no bitwise story — wrong fit for a
  certification-grade reference library today.
- **GPU ports**: platform-artifact and reproducibility questions dominate;
  revisit only after the numba/block-fusion ceiling is reached.

## Suggested order of attack

1. **T2a one-step survival** for snowball/phoenix (biggest measured literature
   payoff, directly attacks Greek noise; opt-in + re-baseline).
2. **T1a fused single-pass kernels** (extends accepted candidate 2; bitwise).
3. **T2b stratification + T4 QMC-ordering experiments** (cheap spikes on
   existing scaffolding).
4. **T2d(+T2e) MLMC** for the LV kernels and MC reference stacks (largest
   infrastructure, largest asymptotic payoff).
5. **T3 adjoint program** once T2a is in production.

## Sources

- Glasserman & Staum, "Conditioning on one-step survival for barrier option
  simulations" (Oper. Res., 2001).
- Alm & Harrach, "A Monte Carlo pricing algorithm for autocallables that
  allows for stable differentiation" (J. Comp. Finance;
  math.uni-frankfurt.de/~harrach/publications/StableDiffs.pdf).
- Rakhmonov, "Conditional Monte-Carlo scheme for stable Greeks of worst-of
  autocallable notes" (IJTAF, 2019, 10.1142/S0219024919500286).
- Giles, "Multilevel Monte Carlo path simulation" (Oper. Res., 2008) and the
  MLMC community page (people.maths.ox.ac.uk/gilesm/mlmc.html).
- Giles, Debrabant & Rössler, MLMC with Milstein discretisation
  (arXiv:1302.4676); Milstein Brownian-bridge convergence (arXiv:1906.11002).
- Giles & Glasserman, "Smoking Adjoints" (RISK, 2006); Giles, Vibrato MC
  (2008); Chan & Joshi, fast Greeks for discontinuous payoffs (Math. Finance,
  2013).
- Glasserman, Heidelberger & Shahabuddin, asymptotically optimal importance
  sampling (Math. Finance, 1999); Glasserman, *MC Methods in Financial
  Engineering* (2004), Ch. 4.
- Imai & Tan, linear-transform dimension reduction (2006); Irrgeher &
  Leobacher, fast orthogonal transforms (arXiv:1508.02160); Wang & Sloan,
  QMC with discontinuous functions (Mgmt. Sci., 2013).
- Salmon et al., "Parallel random numbers: as easy as 1, 2, 3" (SC'11) —
  counter-based RNGs.
