# Smoothed QE-M bias gate: the bias does not replicate

**Date** 2026-08-12 · Probe `probes/probe_qem_bias_gate.py` · Raw
`output/qem_bias_gate/` · Gate 1 of the two blocking smoothed QE-M adoption.

## Headline

1. **On the certified quantities there is no detectable difference.** Delta and
   gamma agree between QE-M and gamma-QEM at every resolution from daily to
   8x daily, at paired SEs of ~0.008 (delta) and ~0.013-0.019 (gamma).
2. **P5's PV bias does not survive replication.** Its z = +3.11 reproduces
   *exactly* on its own 8-seed set, then collapses as seeds are added. Pooled
   over 48 seeds from two independent roots the difference is
   **+0.0135 +/- 0.0116 (z = +1.16)** — not significant.
3. So the gate does not fail gamma-QEM. It also does not clear it: a null result
   at this precision only *bounds* the bias.

## The refinement ladder (24 seeds x 16,384 paths, 1% bump, paired)

| substeps | PV diff | delta diff | gamma diff |
|---|---|---|---|
| 1 | −0.0281 ± 0.0269 | +0.0005 ± 0.0080 | +0.0228 ± 0.0126 |
| 2 | −0.0447 ± 0.0290 | −0.0130 ± 0.0074 | +0.0001 ± 0.0183 |
| 4 | +0.0002 ± 0.0202 | +0.0075 ± 0.0094 | −0.0183 ± 0.0189 |
| 8 | +0.0244 ± 0.0334 | +0.0035 ± 0.0073 | +0.0337 ± 0.0137 |

Every column changes sign along the ladder and |z| stays below ~2.5. This is
noise, not a trend.

**The convergence orders the probe prints (+0.83 / −0.77 / −0.90) are artifacts
and must not be quoted.** Fitting `log|diff|` against `log(substeps)` is
meaningless when `diff` crosses zero; the fit is reporting the shape of the noise.
The probe should refuse the fit when the sign is not constant — a defect in my
gate, recorded here rather than silently left in the output.

**One real signal, and it is reassuring.** Delta moves monotonically with dt for
*both* samplers — QE-M 0.1186 → 0.1169 → 0.1076 → 0.1012, about 0.017 against a
per-sampler SE of 0.005 — while the difference *between* samplers stays at zero.
Both schemes carry their own genuine O(dt) discretization error, and they agree
with each other to within noise at every level. That is exactly the picture you
expect if the higher-moment mismatch is small next to the shared discretization
error, and it also confirms the refinement is live.

## Why P5 saw a bias and this does not

Same instrument (the OSS estimator), same fixture, same daily grid, same code
path — only the seed set changes:

| configuration | diff | paired z |
|---|---|---|
| root 24680, **8** seeds x 32,000 (**P5's exact run**) | **+0.0873** | +4.95 |
| root 24680, **24** seeds x 32,000 (superset) | +0.0295 | +2.15 |
| root 20260811, 24 seeds x 32,000 | −0.0025 | −0.13 |
| root 20260811, 24 seeds x 16,384 | −0.0090 | −0.32 |
| **pooled, 48 seeds x 32,000** | **+0.0135 ± 0.0116** | **+1.16** |

The first row reproduces P5 to the digit (+0.08733, unpaired SE 0.02808,
z = +3.11), so nothing about their computation is wrong — the estimate is simply
unstable in the seed set. The per-seed spread is 0.067-0.092, so an SE of 0.005
would need roughly 324 seeds; eight could never resolve an effect of order 0.02.

Note P5's own z understated its case: the two samplers consume the *same* pseudo
draws, so the difference is paired, and the unpaired `sqrt(var_qe + var_ga)`
formula is the wrong denominator. Paired, their run reads z = +4.95.

**Checked for a seed-generation defect, and cleared it.** All nine leading seeds
of root 24680 are positive (p ~ 0.2% under a zero median), and the seeds are an
arithmetic progression `base + 1000*b` — the classic correlated-stream setup. But
the pattern does not recur: root 20260811 gives sign pattern
`-+++-+++---++-+--------+`, a leading same-sign run of 1, and a first-8 mean
that tracks the full set. So the run of nine was luck, and there is no systematic
early-index effect contaminating the mc2d measurements.

## Cost, remeasured

The cost ratio on this estimator is **x6.4 - x7.0**, worse than the x4.9 P5
recorded, so the quantile-table target moves further away than the earlier
costing assumed.

## What the gate actually establishes, and what it cannot

Established: no transition-law difference is visible in delta or gamma at
production resolution, and the PV bias that motivated the gate is not real at
the precision available. Bounds at 2 sigma, in fixture units:
|PV| <~ 0.037, |delta| <~ 0.016, |gamma| <~ 0.027-0.038.

Not established: whether those bounds are comfortable against the certification's
0.5-contract economic bound. This fixture's Greeks are per-unit; the mapping to
contracts needs the certification fixture and its `EconomicGreekScale`.

## The gate was run at the wrong end of the ladder

The deeper limitation: at daily dt the moment-matching error is *already* below
the noise floor, so there is no signal whose decay rate could be measured. A rate
gate needs a regime where the effect is first clearly visible, and refining from
daily only pushes it further down.

Two better instruments, in order of preference:

1. **Test the sampler against the exact CIR transition directly.** Draw v' from
   gamma-QEM and from the exact noncentral chi-square at matched dt and compare
   moments and CDFs. This isolates the *law* question with no payoff noise at all,
   runs at enormous sample sizes for pennies, and gives the h-dependence cleanly.
   A price-based gate is the wrong tool for a distributional claim.
2. **Coarsen rather than refine.** Establish a visible difference at coarse dt,
   then show it shrinks. Not available on this fixture, whose contractual grid is
   already daily — it would need a fixture with a coarser natural grid.

## Bearing on adoption

Gate 2 is no longer the blocker it appeared to be; there is no established bias
to explain away. That promotes the two open questions to the front:

- **does the variance win transfer to stage-16's estimator?** P5's 3.0x was
  measured against the OSS payoff estimator. The mechanism is *orthogonal* to
  `rqmc_affine_spot_factor` — affine integration handles the residual spot factor,
  QE smoothness the variance factor — so it plausibly survives, unlike OSS which
  was mutually exclusive with it. Unmeasured, and cheap to measure.
- **cost**, now x6.4-7.0 rather than x4.9, against a variance win that has not
  been confirmed on the deployment baseline.

## Note on testability

This gate had to reimplement the path loop because the Heston engine's QE block
is inline rather than routed through the shared `qe_variance_step`. Had it been
shared, swapping the variance sampler would be a one-line patch of `qe_kernels`
and the gate could have run against the real engine and the real certification
estimator. That is a second argument for the migration, independent of the 1.04x
performance case that did not justify it.
