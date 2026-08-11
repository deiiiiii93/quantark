# Pathwise delta for the OSS estimator — implemented, verified, NOT accepted

**Date** 2026-08-11 · Probe `probes/probe_pathwise_delta.py` · Raw
`output/pathwise_delta/` · Extends the mc2d-gamma-convergence prototype
(RESEARCH.md §5.4) without modifying that session's files.

## Verdict first

The derivative machinery is implemented and every component verifies to ~1e-9.
The assembled estimator nonetheless disagrees with the finite-difference limit of
the same estimator by **−4.3e-3 (t = −4.36) at h = 0.125%**, about 4% of delta,
and the discrepancy **grows** as h shrinks instead of vanishing.

For a *reference* estimator whose entire purpose is to remove finite-difference
bias, an unexplained 4% bias is disqualifying. **Do not use for certification
references until the h-dependence is explained.**

## Two pieces of mathematics that made it computable

**1. s_i cancels.** The naive chain rule divides by the conditional step stdev
s_i, which floors at 1e-300 whenever the variance path touches zero -- routine in
a Feller-violated fit -- sending z-scores to ~1e300 and the derivative to 1e35.
But the increment is `s_i * dz = s_i * ratio * (-d_log/s_i)`, so s_i cancels
exactly and the recursion is

    d_log <- d_log * (1 - ratio)

with no division anywhere.

**2. The hazard identity.** For a one-sided truncated draw
`z_a = Phi^-1(Phi(z_b) * u)`, the chain rule wants `u * phi(z_b)/phi(z_a)`, which
as `u -> 0` is `0 * exp(+huge)` and overflows. But `u = Phi(z_a)/Phi(z_b)` holds
*exactly* by construction, so the product collapses to

    hazard(z_b) / hazard(z_a),      hazard(z) = phi(z)/Phi(z)

a ratio of two quantities each behaving like |z| in the tail. The corridor's
upper-truncated case needs `phi(z)/(1-Phi(z))` via `log_ndtr(-z)`; the two-sided
case admits no such collapse but is safe, since `z_b` lies between the endpoints
and `phi(z_b)` is bounded below by the smaller endpoint density.

## What is verified

| check | result |
|---|---|
| PV parity vs published `oss_price` | **exact**, 0.00e+00 |
| per-step derivative, KO one-sided | 1e-9 vs FD |
| per-step derivative, KI one-sided (upper tail) | 1e-9 vs FD |
| per-step derivative, two-sided corridor | 1e-9 vs FD |
| terminal closed form, Run A | 1e-9 vs FD |
| terminal closed form, Run B | 1e-9 vs FD |
| path recursion `d_log_b`, weight-carrying paths | **O(h^2)**, ratios 3.90 / 3.52 |
| gamma as FD-of-pathwise vs OSS second difference | 4.0x variance (3 seeds, indicative only) |

## What is not explained

Paired per-path difference `E[FD] - E[pathwise]` on identical draws, 64 seeds x
25k paths:

| bump | mean diff | stderr | t |
|---|---|---|---|
| 0.25% | −2.53e-3 | 1.40e-3 | −1.81 |
| 0.125% | −4.33e-3 | 9.95e-4 | **−4.36** |

A correct pathwise estimator requires only `E[d val] = d/dS0 E[val]` -- the
interchange of derivative and expectation -- so the difference should fall like
h^2. It grows. Neither standard failure mode fits:

- a **jump** in the per-path value would give an h-INDEPENDENT offset (crossing
  probability ~h, FD amplification ~1/h);
- a **kink** would give an offset shrinking like h.

Growth as h shrinks matches neither. The mechanism is unidentified.

## Hypotheses tested and rejected

- **Numerical guards (`_UEPS`, the `alive` fallback) inject jumps.** Rejected:
  the bias is identical to four significant figures across `_UEPS` = 1e-14,
  1e-30, 1e-80. Sixty-six orders of magnitude move it not at all.
- **`d_log_b` is wrong** (its unweighted error is flat in h). Rejected: weighting
  by `w_b` shows O(h^2) convergence on the 12.4% of paths that carry weight. The
  flat error is entirely on dead paths, whose `log_b` is a fiction the `alive`
  fallback pins to `z_b = 0`.

## Two of my own diagnostics were wrong first

Recorded because both are easy traps:

1. **`mean|per-path error|` is the wrong test.** It demands per-path smoothness
   the method never claims; kinks on null sets are permitted. That test reported
   O(h) for a derivative whose components are each exact to 1e-9, and sent me
   looking for a bug in correct algebra. The right test is the signed mean across
   seeds.
2. **12 seeds could not resolve it.** The 12-seed run reported −6.5e-3 / −6.2e-3;
   64 seeds gives −2.5e-3 / −4.3e-3. The earlier pair was noise, and reading a
   plateau into it was over-interpretation -- the same error that produced the
   retracted "OSS hurts delta" claim earlier the same day. Variance-ratio and
   small-bias claims need the seeds before they mean anything.

## Bearing on the certification

Low, by the corrected OSS measurement earlier today: OSS alone (delta 3.67x,
gamma 33.32x) already drops every stage-16 gamma AND delta row to the 16-64 batch
floor, so pathwise delta buys no variance the allocation needs. Its value was
always bias removal -- and it currently has an unexplained bias of its own, so it
delivers the opposite.

**Recommendation:** park this behind the OSS port. The port is what the
allocation needs; pathwise delta is an accuracy refinement that should not go
near a reference until its h-dependence is understood. The machinery here is
sound and reusable -- the two identities above are the hard part, and both are
verified -- so resuming is cheap.
