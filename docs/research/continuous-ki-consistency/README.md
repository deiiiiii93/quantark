# Continuous KI consistency — native 1D GBM

## Result

MC, native PDE, QUAD V1 and QUAD V2 now support the same two continuous-KI
probability definitions for Snowball, Phoenix, Memory Phoenix and absolute
schedule KO-reset:

- **KI-ever:** underlying barrier touch anywhere in the contractual monitoring
  window, including after autocall. KO-reset's window ends at pre maturity.
- **KI settlement:** the note ends knocked in without autocalling.

A carried KI has KI-ever = 1. An immediate KO has KI settlement = 0, while
underlying monitoring can still give positive KI-ever. Constant barriers are
required. Expired products keep their existing terminal APIs.

The engines retain their own numerical methods. Equal definitions and inputs
do not imply identical finite-grid estimates or identical MC samples.

## Changes

### V2 continuous monitoring

A separate one-channel no-touch claim computes KI-ever. It has no note or
coupon-memory states and does not run for price-only contexts. Constant drift
per unit variance permits an analytic time-change/reflection readout. Other
piecewise coefficient models use a native survival recursion on the prepared
clock and mesh. Both scalar and vector spot statistics use the prepared claim.
The vector API now prepares the requested spot range instead of failing outside
the default two-percent range.

### One coefficient-model contract

`continuous_ki_support.py` supplies the common model classification and required
coefficient-change knots. All four native routes preserve those knots and
sample their own stepping coefficients on the resulting clock. Forward-carry
adapters retain cumulative carry; no curve is replaced by a terminal flat yield.

All parameter classes accept:

```python
continuous_term_structure="exact"                 # reject unrecognized smooth models
continuous_term_structure="piecewise_constant"    # permit coefficient approximation
continuous_steps_per_year=96                       # refine smooth coefficient intervals
```

`exact` describes the coefficient model, not an exact numerical price. The
recognized exact inputs include flat/log-linear discount curves, constant or
forward-carry-implied yields, and flat/total-variance-interpolated volatility,
including supported parallel shifts. Smooth curves require time refinement.

Historical defaults remain: MC/PDE/V1 permit piecewise coefficients and use
native time density (`continuous_steps_per_year=0` adds knots only). V2 defaults
to exact-model validation; opt-in piecewise mode uses 48 subdivisions/year.
The same capabilities and controls are available; default densities remain
engine-specific. Resource limits remain enforced.

### MC variance consistency

The old native GBM bridge used maturity volatility even when the path generator
used interval volatilities. It now reads exactly the interval variances used by
the generator through a read-only broadcast view. Storage is one step vector,
not another paths-by-steps array. Vol-model overrides retain their own realized
variance path.

### V2 reflected-grid acceleration

A barrier-aligned Gaussian grid permits reflection by a node permutation. The
native direct/FFT operator supplies both reflection terms. Mirrored queries
outside the mesh, shifted source grids, unaligned meshes and extreme reflection
weights retain the pointwise integration path. The final point readout still
computes complete derivatives. Plain alive transitions that would be overwritten
by the survival kernel are no longer computed.

`QuadV2Params(continuous_grid_backend="pointwise")` retains the comparison
implementation. The default `"auto"` selects the accelerated compatible grid.

## Verification

**525 selected regression tests passed** in 67.82 seconds with four workers.
They include the new continuous controls, prior outcome-price increment tests,
MC/PDE statistics, lifecycle scenarios, folding, Greeks, memory/deferred coupons,
term inputs, pruning and batch spot queries. Python compilation and
`git diff --check` also passed.

The **60-case qualification matrix** uses actual engine outputs: four product
variants, three market/contract scenarios and five engine configurations (V1,
V2, finer V2, native PDE, seeded MC). Scenarios are overlapping KI/KO with KO
suppressed after KI, piecewise carry/volatility, and smooth carry approximation.

| Gate | Observed maximum | Result |
|---|---:|---|
| Deterministic PV difference from finer V2, per 100 | 0.00498156 | Pass: <= 0.01 |
| Deterministic KI-ever / KI settlement / KO difference | 0.000432957 | Pass: <= 0.0005 |
| MC probability difference, 100,000 paths, seed 917 | within allowance | Pass: 95% binomial sampling interval + 0.0005 |

These are qualification tolerances on this matrix, not global error bounds.
Finer V2 is an executable comparison, not a hand-asserted exact solution.
Independent flat-GBM first-passage formulas and the retained pointwise kernel
provide additional controls.

For smooth carry, increasing V2 density from 96 to 192/year changes PV by at
most **0.00001353 per 100**, KI-ever by **0.00000154**, and KI settlement by
**0.000000955** on the four checked notes. See `time_refinement.json`.

## Pricing speed at identical accuracy settings

Cold calls use a fresh engine each repetition. Values are medians of three
repetitions, one numerical-library thread, identical order 8 / cells-per-SD 2
and identical notes. Compare the retained pointwise implementation with the
accelerated implementation; no grid reduction is involved.

| V2 continuous note | Price before | Price after | Speedup | Stats before | Stats after |
|---|---:|---:|---:|---:|---:|
| Snowball | 0.6815 s | 0.00643 s | 106x | 2.4197 s | 0.01847 s |
| Memory Phoenix | 7.8591 s | 0.04315 s | 182x | 44.8445 s | 0.27419 s |
| KO-reset | 2.9606 s | 0.01519 s | 195x | 20.2809 s | 0.08785 s |

The paired price differences were below **4.3e-14**, KI settlement differences
below **3.0e-15**, and KI-ever values identical on these notes. These timing
results cover cold price/statistics calls, not a full finite-bump Greek timing
claim. Runtime varies with note states, time refinement, grid and hardware.

## Artifacts and reproduction

- `qualification.json`: all 60 actual-engine records.
- `gates.json`: qualification thresholds and observed maxima.
- `time_refinement.json`: smooth-curve time refinement.
- `reflection_timing.json`: matched-grid cold timing and result parity.
- `baseline.json`: historical warm-call baseline; not used for the speedup table.
- `qualification.pointwise.partial.json`: preserved partial run before acceleration.
- `verification_summary.json`: test result, source hashes and verification settings.

From the repository:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python docs/research/continuous-ki-consistency/qualify_engines.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python docs/research/continuous-ki-consistency/benchmark_reflection.py
```

The relevant new tests are
`test/test_asset/equity/engine_quad/test_continuous_ki_capability_consistency.py`.

## Scope

This release qualifies native 1D GBM. Local volatility and Heston/SLV probability
support are deferred as requested. Continuous KI on rebased KO-reset schedules
remains unsupported across these native routes. First-hit time distributions,
legacy V1 memory-coupon cash attribution and the opt-in forward-density
approximation are outside this continuous-KI probability repair. All changes
remain uncommitted.
