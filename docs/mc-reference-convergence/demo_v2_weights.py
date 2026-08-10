"""V2 (cross-fitted control weights) -- DESIGN INVALIDATED 2026-08-10, NOT RUN.

The first version of this demo paired two separate harness runs by seed (an
SLV primary at 1024 paths/batch against a Heston "control" at 8192) and fed
them to cross_fitted_control. Reading the estimator it was supposed to model
showed that is not what stage-17 does, so the measurement would have been
meaningless. It is kept here as a corrected design note rather than deleted,
because the V2 lever itself is still open.

What stage-17 actually does
---------------------------
`controlled_case_economic_components` builds a three-level TELESCOPING
estimator, not a regression control across runs:

    primary[level]     = state_dependent - w_frozen * frozen_low
    middle[level]      = w_frozen * frozen_high - w_heston * heston_low
    heston_high[level] = w_heston * heston_high

Summing telescopes the control terms away in expectation. Crucially, each
low/high pair is evaluated on the SAME paths within one run: `frozen_low`
comes from the primary payload with `control=True` (the frozen-leverage
control that `rqmc_frozen_leverage_conditional_control=True` emits alongside
the SLV payoff), and `heston_low` likewise rides the middle run. The weights
`w_frozen` / `w_heston` are per-case constants in AGGREGATE_CONTROL_WEIGHTS
(e.g. ordinary_full heston 0.7, low_feller heston 0.0).

Correct V2 experiment (not yet run)
-----------------------------------
Cross-fitting must operate on the intra-run rows, not on fresh runs:

1. Build one primary reference per cell via `build_primary_reference` and one
   control-only reference via `build_control_only_reference` -- these already
   carry both the state-dependent rows and their matched `control=True` rows.
2. For each level, cross-fit w_frozen on (state_dependent, frozen_low) and
   w_heston on (frozen_high, heston_low) using
   `quantark.montecarlo.control_weights.cross_fitted_control`, with the
   out-of-fold expectation taken from the independent higher level -- which is
   exactly the role `heston_high_reference` already plays.
3. Compare the resulting per-case weights against the frozen constants, and
   the recomposed row variance against the frozen-weight recomposition, using
   `controlled_case_economic_rows` for both so the comparison is apples to
   apples.

Because step 1 reuses references the production run already builds, the
corrected V2 costs no extra Monte Carlo once a reference set exists -- it is
post-processing. That is why it is deferred to after the Phase-1 regeneration
rather than run as a standalone demo now.
"""

raise SystemExit(__doc__)
