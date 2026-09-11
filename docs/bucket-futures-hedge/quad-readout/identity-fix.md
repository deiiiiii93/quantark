# Matched carry identity: implementation and validation

Implemented in `worktree-bucket-futures-hedge` on 2026-09-10, as uncommitted
changes on top of `fd1efc03`. The 1% pricing/hedge delta, hedge strategy,
QUAD readout, tolerance and existing study outputs are unchanged.

The audit now recomputes both spot derivatives at each of
`identity_spot_bumps_rel = (0.001, 0.0005, 0.00025)`. The supplied bucket
vector is retained, so a wrong bucket still fails. A wrong reported hedge
delta still fails the separate reproduction check at its pricing bump.

Let R be the finest matched residual and E the absolute change from the
previous level. The identity passes only when `abs(R) + E <= 0.01` hands;
it fails when `abs(R) - E > 0.01`, and is inconclusive otherwise. E is an
observed spot-step refinement allowance, not a rigorous total error bound,
a Monte Carlo confidence interval, or an external delta-accuracy certificate.

The ladder adds 12 price calls per audited date. The default bump choices
do not affect prices used to size the hedge. They are configurable through
`CarryRiskSettings` and the fleet CLI's `--identity-spot-bumps-rel`; their
values enter the study fingerprint. No existing run was overwritten.

## Reported fields

| Field | Meaning |
|---|---|
| `identity_residual_hands` | Finest matched local identity; this is the gated residual |
| `identity_spot_refinement_error_hands` | Absolute last successive residual change |
| `identity_status` | Pass, fail, inconclusive, or not measured for the local identity |
| `identity_ladder` | JSON containing both spot derivatives and the residual at every bump |
| `finite_bump_identity_residual_hands` | Original pricing-delta minus original pinned-spot-delta minus scaled buckets |
| `finite_bump_identity_status`, `finite_bump_identity_reason` | Whether the original large-bump diagnostic could be measured |
| `pricing_delta_local_gap_hands` | Reported hedge delta minus the finest repriced local delta |

The existing `delta_f_derived_hands` and `delta_f_direct_hands` retain their
original conventions. Their difference now corresponds to
`finite_bump_identity_residual_hands`, not to the gated local residual.
Consumers that relied on the old equivalence should use the explicit new
diagnostic field.

A 1% pinned spot move can exceed the dividend object's supported yield
range even when local bumps remain valid. This optional large-bump
diagnostic then records `inconclusive` and NaN, with the actual exception
message. Required audit measurements still run; any failure to construct
one of those keeps the overall audit inconclusive.

## Historical validation

The production `audit_held_book` was run on every archived live state for
both curve conventions, using actual held quantities and saved pricing
Greeks. The product, coupon and market state were rebuilt first; every
base PV matched the archive within 0.0000016 CNY before any saved Greek
was reused. This validates the audit against the two archived front-control
books, without replaying or changing any hedges.

| Curve | Live dates | Overall passes | Maximum absolute R | Maximum abs(R)+E |
|---|---:|---:|---:|---:|
| Flat zero q | 242 | 242 | 0.002195980 | 0.004339868 |
| Flat forward carry | 242 | 242 | 0.001993829 | 0.003947494 |

The net-delta reproduction errors remained below 4.9e-11 hands. All nodal
and parallel rho-q errors remained below 0.000097 bp.

On 2024-02-05 the flat-q local audit now passes. Its original 1% pinned
scenario still records `inconclusive: dividend yield magnitude must be
<= 1.0.` This accounts for the original missing finite residual: the
previous 241-date convergence experiment excluded this inconclusive live
date as well as the unmeasured terminal date.

Evidence: `identity_fix_flat_q.csv`, `identity_fix_flat_forward.csv` and
the reproducible driver `verify_identity_fix.py` in this directory.

## Tests and reproduction

287 focused tests passed: carry-risk settings and algebra, nonlinear
analytic identity, intentionally wrong buckets and pricing delta,
unresolved refinement, the historical snowball under both QUAD readouts,
replay recorder/config/results, bucket replay, and study/validation CLI.
Two existing deprecation warnings concern the old OTC import aliases.

Run the archived-state validation from the worktree:

```sh
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/gate-d-mpl \
  /Users/fuxinyao/quant-ark/.venv/bin/python \
  docs/bucket-futures-hedge/quad-readout/verify_identity_fix.py \
  --run-dir example/snowball_q_term_structure/data/bucket_hedge_v2/subset/runs/20230504/term_flat_q__front \
  --history-dir /Users/fuxinyao/quant-ark/example/mo_volmodels/data/history \
  --output /private/tmp/identity_fix_flat_q.csv --workers 4
```

Use `term_flat_fwd__front` and a different output path for the other curve.
The frozen regression at `test/test_futures_carry_identity_quad.py` requires
no market-data cache and reproduces the original 0.548283-hand residual.

The next numerical task is external accuracy of the hedge deltas. A fresh
comparison should use a corrected PDE reference or an independently
validated Gaussian transition derivative, in actual CSI1000/IM units, with
genuine grid and bump refinement. Local identity closure alone cannot
certify that accuracy.

**Update 2026-09-11.** The sentence here used to say the PDE in this branch
predates its own near-barrier delta repair, which was the blocker. The
branch is now merged up to its base, so `25d4f7d6` is an ancestor and the
corrected reference is available. The comparison is work, not a blocker.
