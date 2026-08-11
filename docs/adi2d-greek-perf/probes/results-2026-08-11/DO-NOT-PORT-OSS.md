# Do not port OSS: the certification's reference already beats it

**Date** 2026-08-11 · Probe `probes/probe_oss_vs_certification_stack.py` · Raw
`output/oss_vs_certification/`

## The error this corrects

Every OSS measurement in this program -- mine and the mc2d-gamma-convergence
session's -- benchmarked against `QESnowballMCEngine(method="pseudo")`: plain
Monte Carlo with a central FD bump. Stage 16 does not use that. Its Heston
reference is

    QESnowballMCEngine(
        method=MonteCarloMethod.RANDOMIZED_QUASI,
        rqmc_affine_spot_factor=True,
        rqmc_spot_bridge_strata=1, rqmc_spot_bridge_dimensions=1,
    )

and the comment on that flag says what it is for: "QE variance is independent of
the residual spot Brownian factor. Integrate that factor exactly so barrier
indicators do not dominate finite-bump delta/gamma uncertainty."

That is conditional smoothing of the same factor OSS smooths. Applying a
plain-MC-relative gain to standard deviations produced by it -- which is what my
28.4x allocation projection did -- overstates the benefit by whatever the
existing estimator already achieves. Which turns out to be more than OSS does.

## Measured, 12 seeds x 25k paths, same fixture, equal paths

Standard error of the FD Greek:

| h | plain delta | cert delta | OSS delta | plain gamma | cert gamma | OSS gamma |
|---|---|---|---|---|---|---|
| 1.0% | 0.00643 | **0.00182** | 0.00351 | 0.01039 | **0.00310** | 0.00420 |
| 0.3% | 0.01345 | **0.00252** | 0.00582 | 0.09041 | **0.00483** | 0.01402 |
| 0.1% | 0.02008 | **0.00262** | 0.00841 | 0.58433 | **0.00822** | 0.03002 |

OSS variance ratio vs the certification stack: delta 0.27x / 0.19x / 0.10x,
gamma 0.55x / 0.12x / 0.08x. **The certification's estimator wins at every bump,
by a margin that widens as the bump shrinks.**

## Why, and why it was never going to be close

Growth of gamma stderr from h=1% to h=0.1%:

| estimator | growth | reading |
|---|---|---|
| plain FD | **56.3x** | the h^(-3/2) blow-up, confirmed (law predicts 31.6x) |
| certification | **2.65x** | nearly h-independent |
| OSS | 7.15x | h-dependence reduced but not removed |

Both methods target the same quantity: the residual spot Brownian factor's
interaction with barrier indicators. OSS **samples from a truncated
distribution** of that factor. `rqmc_affine_spot_factor` **integrates it out
analytically**, which is available because QE variance is independent of it.
Exact integration strictly dominates truncated sampling -- the truncated draw
still carries indicator variance, which is exactly the residual 7.15x growth.

They are also **mutually exclusive, not composable.** RESEARCH.md §5.3 states
that OSS truncation makes the asset-factor map history-dependent and therefore
requires generating the asset factor forward. Exact integration requires the
opposite: that the factor be a known affine loading. Porting OSS means giving up
the mechanism that produces the certification's advantage.

## Cost-adjusted, for completeness

Variance x seconds, certification = 1.0 (lower is better):

| h | plain | certification | OSS |
|---|---|---|---|
| 1.0% | 3.10x | 1.00x | **0.79x** |
| 0.3% | 94.7x | 1.00x | 3.68x |
| 0.1% | 1355.7x | 1.00x | 5.77x |

At h=1% OSS is ~26% more efficient per second (cert costs 3.6x plain per
pricing, OSS 1.56x). That is the single number favouring OSS, it is marginal at
12 seeds, and it evaporates at any smaller bump. It does not justify a non-bitwise
engine port that would forfeit exact spot-factor integration.

## Verdict

**Do not port OSS.** Drop it as a certification-reference candidate. The existing
reference stack is the best of the three measured, is already nearly h-stable in
gamma, and its cost is bought back many times over in precision.

Corollary for the earlier findings in this program, all withdrawn:

- "delta 3.67x / gamma 33.32x from OSS" -- correct, but against plain MC only.
- "heston 8192 -> 288 batch-cells, 28.4x" -- withdrawn.
- "near_ki 2048 -> 64" -- withdrawn; its deep pilot's 2048 stands.
- "the certification is limited by the batch floor, not MC noise" -- withdrawn.

What survives is the pilot's direct measurement of the current estimator:
**heston 8192 -> 2816 batch-cells (2.9x)**, six cells at 128 with `near_ki`
pinned at 2048, and the SLV side keeping its frozen counts.

## Note on the bump semantics

Shrinking the reference bump to cut FD bias looks attractive given the 2.65x
variance cost, but it is not needed: the certification compares PDE-at-bump
against MC-at-bump, so the O(h^2) FD bias largely cancels between them. The
evidence says as much -- "barrier-adjacent values are h-width hedge exposures,
not a claim that a pointwise classical derivative exists". The bump is a shared
convention, not an error term to minimise.

## Lesson

A method's published speedup is relative to the baseline its authors chose. Before
carrying that ratio into another system, measure that system's baseline. One grep
of `make_mc_engine` would have caught this before the pathwise derivative was
written.
