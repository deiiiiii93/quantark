# Autocallable KI consistency

## Contract

`ki_survive_knocked_in_probability` is the probability that the note settles
knocked in without autocalling. `ki_ever_probability` monitors the underlying
through the contractual KI window, including after an autocall. KO-reset's
window ends on the pre schedule. A carried KI implies KI-ever = 1; its KI
settlement probability is the surviving mass, which can be zero.

Retain the historical meaning of `ki_probability` for compatibility. Consumers
should use the two explicit fields. The initial repair left continuous V2 KI-ever unavailable. The subsequent
[continuous KI repair](../research/continuous-ki-consistency/README.md) closes
that gap with a KO-independent survival claim.

## Implementation choices

Three approaches were considered:

1. Derive statistics from the pricing event operations (selected): share
   projection, event order, grid construction and readout; propagate separate
   indicator columns in one sweep.
2. Compute each probability by a separate folded-price perturbation: useful as
   independent evidence, but scales with the observation count and subtracts
   large nearly equal prices.
3. Switch to forward density: rejected as an exact repair because its finite
   grid discretization differs from price.

### QUAD V1 Snowball

Use `_apply_observation` for prices and KO/KI settlement indicators. Continuous
node transfers and discrete KI projection now act on those indicators exactly
as on prices. KI settlement is a column of the KO stack, including on carried
KI. KI-ever uses a separate, KO-independent KI pair. Auto-converged Snowball
and Phoenix statistics use the accepted price grid; KO-reset preserves its
existing fixed-grid policy.

### QUAD V1 Phoenix

Use the price's coupon-inclusive alignment and barrier breaks. Apply the price's
`_project_joint_phoenix_event` to every statistical column at a coupon
observation, including columns whose continuation does not jump at that
barrier. Generalize its spatial axis handling for indicator stacks. Memory
amounts do not change the KO/KI ending probabilities, so these columns need no
memory state fan. Preserve the existing nodal price event order separately.

### QUAD V1 KO-reset

Extract the price's event sequence into `_apply_reset_observation` and use it
for the indicator stack. Apply simultaneous discrete pre-KO last, exactly as
price does. Carried KI omits the future KI schedule from the lattice, as price
does. Exempt the underlying KI-ever column from KO operations.

### QUAD V2

Add one independent discrete first-hit monitoring state to event-statistics
contexts. Restrict note continuation to contractual channels; the KI-ever
channel follows the independent monitoring state even on terminating KO
actions. It shares the native operator, vector spot readout and resource gates.
Price-only contexts acquire no extra state.

### Forward density

Keep the opt-in diagnostic route. Expose different statistics/price models via
`StatisticsProvenance("quad_v1_forward_density", "quad_v1")`. Repair its carried
KI semantics but do not assert exact price consistency. Residual-based cashflow
reconciliation does not establish exact outcome probabilities.

## Evidence

Regression tests are in
`test/test_asset/equity/engine_quad/test_autocallable_probability_consistency.py`.
They add deterministic cash through the actual `price_folded` API to each KO or
terminal outcome and compare price increments with reported probabilities. They
also cover carried KI, continuous/discrete monitoring, overlapping barriers,
reverse contracts, disabled KO after KI, transition readout, deferred Phoenix
coupons, memory, and valuation-date KO.

Raw before/after results and the comparison against `HEAD` prices are preserved
in `docs/research/autocallable-ki-consistency/verification.json`.

The original audit scripts were re-run with their original inputs, 40,000 MC
paths and seed 917. Previous JSON outputs are preserved under
`/private/tmp/*before_consistency_fix.json`.

On the ordinary 16 family/engine cases, maximum KI/KO price increment gaps were
below 2.4e-13. The V1 overlapping KO-reset KI settlement probability is now
0.004035707146438, versus price increment 0.004035707146443. Its old report was
0.004907650239651. Carried Snowball KI settlement is 0.366087552728384, versus
price increment 0.366087552728345, rather than 1.

On the frozen 2Y base-curve/hold-forward note:

| Engine | KI settlement | Underlying KI-ever |
|---|---:|---:|
| V1 | 28.693071% | 62.654497% |
| V2 | 28.695053% | 62.653024% |
| PDE high | 28.695221% | 62.651208% |
| MC | 28.405000% | 62.667500% |

The 2Y unit-cash subtraction gaps are at most 5.2e-9 in probability because the
cash increment is small relative to the trade's value surfaces. MC sampling
and different deterministic discretizations remain; equality across engines
is not asserted. No market snapshot or study output was overwritten.

## Qualification limits

This repair targets the audit's native 1D GBM routes. It does not certify
stochastic/local volatility, parallel or RQMC estimators, or every lifecycle
variant. Legacy V1 maturity cash is still a residual, and its memory coupon
cash attribution is unavailable; a zero reconciliation residual is not proof
of a native, independently propagated cashflow decomposition. Forward-density
statistics remain an explicitly identified approximation. Continuous V2 underlying KI-ever is now available through the subsequent
continuous-KI repair linked above.

## Test results

All 40 new regressions pass on the final code. The expanded existing suite ran
482 selected tests: 481 passed and one new KO-reset auto-convergence test failed
on the source snapshot loaded before its correction. KO-reset does not support
auto-convergence in its existing price or folded-price route; the statistics
now retain that policy too. The corrected case passed in the subsequent
40-test run. The earlier six existing failures were assertions preserving the
removed probability gaps or carried KI=100% survival; these were replaced with
stronger direct price/statistics invariants and passed on re-run. Representative
V1 prices compared bit-for-bit equal to `HEAD` in eight ordinary/carried cases.
Python compilation and `git diff --check` passed.
