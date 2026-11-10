# Native local-volatility and Heston/SLV KI consistency

## Capabilities

The native MC and PDE model routes support the same KI definitions for
Snowball, Phoenix, memory Phoenix and absolute KO-reset:

- **KI settlement:** the note settles knocked in without KO.
- **KI-ever:** the underlying touches KI in its contractual monitoring window,
  including after KO. KO-reset monitoring ends at pre maturity.
- Carried KI has KI-ever = 1; its settlement probability still allows future KO.
- Immediate KO has KI settlement = 0 and can have positive underlying KI-ever.

Native Heston/SLV statistics are now the default (`statistics="native"`).
Explicit refusal and the labelled `flat_vol_surrogate` remain available.
Memory/deferred coupons and pre/post KO-reset schedules use native product
states and cashflow buckets. No flat-volatility substitution is used in native
statistics. Legacy `ki_probability` meanings remain unchanged.

Local-volatility statistics use the existing 1D product march. Its immediate
KO branch now retains independent underlying monitoring. Vol-model MC keeps
its realized interval-variance bridge; these routes are independently qualified
against the PDE results below.

## Implementation

The ADI backend uses the existing product-family terminal conditions, event
stages, boundary cash allocation and bilinear readout for prices and probability
columns. It retains rolling spot/variance claims and shares the directional
operator across columns. Memory needs two arrears states in each KI regime;
KO-reset uses the existing pre-maturity seed and pre/post event dispatch.

KI-ever has a separate one-channel underlying claim with its own damping
schedule, so note KO events cannot absorb it or alter its numerical damping.
Price-only calls allocate no statistics or KI-ever companion. Carried KI skips
unused alive-regime ADI sweeps.

`column_backend="auto"` batches claims. `"scalar"` retains the corresponding
native scalar sweeps for parity and timing controls. Sparse and characteristic
variance-transport configurations retain scalar directional sweeps. Native
characteristic variance transport uses fixed positive linear interpolation,
which preserves cashflow linearity. The old payoff-clipped cubic transport
mismatched a KI cash increment by up to 0.0000669191 on the checked spots.
The repaired discrepancies are below 2.1e-14. `transport.json` records the
control, repair and refinement: 192 variance nodes pass the audit price and
probability gates against 384-node adaptive transport. The 96-node price
comparison fails and is preserved in the ledger. This is a case-specific
qualification, not a universal grid prescription.
Deterministic discounting is integrated exponentially outside the spatial
ADI generator, with the same transformed boundary values for every claim.
This preserves sure-unit probability partitions and allows coarse-grid rich
outputs without weakening `EventDistribution` validation. Terminal payoffs
are initialized once across variance nodes. Spot operator/factor caches retain
two recent keys, and Snowball/reset event projections batch variance nodes.

The standard NumPy tridiagonal backend and the existing optional C backend
remain supported.

The ADI clock retains actual observation dates, staged maturity and coefficient
knots. The existing time builder allocates the configured `n_t` budget between
mandatory nodes; delivered counts may differ slightly. This avoids rounding an
interior observation onto the terminal node or missing its contractual ordering.
Rate/carry coefficients and barrier-local diffusion use that actual clock.
Constant leverage does not add redundant time knots; unit leverage reproduces
Heston on the same mesh. SLV Phoenix now uses the same declarative spot grid as
Heston Phoenix, repairing its previous bypass.

Refinement controls remain native: `n_x`, `n_v`, `n_t` for ADI; 1D PDE grid
controls for local volatility; MC path count and `ki_bridge_nodes_per_day`.
The local/frozen-variance crossing corrections are approximations within each
step and require refinement for non-flat models. Numerical probabilities are
reported without renormalization so partition error remains visible.

New public PDE routes:

- `LocalVolKOResetSnowballPDESolver`
- `HestonKOResetSnowballPDESolver`
- `HestonSLVKOResetSnowballPDESolver`

## Accuracy qualification

`qualification.json` contains 36 actual-engine records: three models, four
product variants and three configurations (coarse PDE, finer PDE, seeded MC).
Local volatility has a downside skew; Heston has genuine stochastic variance;
SLV has a spot-dependent leverage surface. All include piecewise term carry.
MC uses 100,000 independent paths, seed 917, and two bridge nodes/day. PDE ADI
settings are 160/32/192 and 240/48/384; local-volatility uses standard/high
profiles. These are qualification settings, not new engine defaults.

All **96 checks pass**. Maximum coarse/finer differences:

| Metric | Difference | Gate |
|---|---:|---:|
| PV per 100 | 0.00325701 | 0.01 |
| KI-ever | 0.000363880 | 0.0005 |
| KI settlement | 0.000122194 | 0.0005 |
| KO probability | 0.000169873 | 0.0005 |

MC checks use the same numerical allowance plus a 95% binomial allowance for
probabilities, and the actual MC price standard error for PV. The finer PDE is
an executable comparison, not an asserted exact solution. The matrix qualifies
these cases and parameters; it is not a global error bound.

Independent tests add cash through the actual `price_folded` API and compare
its PV increment with outcome probabilities. They also cover carried KI,
reverse and discrete/overlapping events, disabled KO after KI, memory and
expiry coupons, immediate KO, monitoring after KO, boundary cash allocation,
unit-leverage reduction, batching parity and existing Greek/bump routes.

## Verification and pricing speed

**237 regressions pass on both NumPy and the temporary C Thomas backend**
(26.23 s / 16.06 s). They cover the final native outcome, transport, folding,
calendar, memory/deferred coupon, immediate-KO, rich-output and batching paths.
Another 13 variance-policy checks pass. Earlier exact-discounting model/Greek
qualification passed 183 tests, including the production reverse-KI Greek
sweeps; see the logs and phase notes in `verification_summary.json`.
Compilation, public imports and `git diff --check` pass. Eight direct rich API
checks are recorded in `rich_api.json`.

Cold engines, medians of three repeats, one numerical-library thread, NumPy,
identical 120 spot / 24 variance / 96 configured time steps. The scalar and
batched native methods give the same numerical results; no grid reduction is
used. The prior production price is a separately loaded checkout-HEAD control
for the two previously supported products.

| Product | Prior production price | Native scalar price | Batched price | Scalar stats | Batched stats |
|---|---:|---:|---:|---:|---:|
| Snowball | 0.20078 s | 0.18394 s | 0.11528 s | 1.79881 s | 0.50410 s |
| Phoenix | 0.20274 s | 0.20683 s | 0.14054 s | 4.01111 s | 1.09035 s |
| Memory Phoenix | unavailable | 0.40690 s | 0.19325 s | 5.29610 s | 1.45814 s |
| KO-reset | unavailable | 0.17832 s | 0.11693 s | 4.17952 s | 1.15592 s |

The existing production Heston Snowball/Phoenix price calls improve by
**1.74x / 1.44x**. Native batching improves price by **1.47–2.11x** and
statistics by **3.57–3.68x** relative to the scalar control. Paired scalar/batch
PV differences are below **2.9e-14** and KI settlement differences below
**5.6e-17**. Prior-production PV differences are below **0.00006 per 100**,
including the discounting correction. These are cold price/statistics timings;
they do not claim a full finite-bump Greek speedup.

## Reproduction and evidence

Run from the repository with its virtual environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python docs/research/vol-model-ki-consistency/qualify_engines.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python docs/research/vol-model-ki-consistency/benchmark_columns.py
```

- `qualification.json`, `gates.json`: final calendar-aligned matrix and checks.
- `qualification.uniform_clock.json`: preserved preceding uniform-clock run.
- `qualification.scalar.partial.json`: preserved slow scalar partial run.
- `before.json`: initial price observations, not used for speedup claims.
- `broad_pre_alignment.partial.log`: preserved interrupted broad diagnostic.
- `verification_summary.json`: final tests, settings and source hashes.
- `transport.json`, `transport.initial.json`: characteristic transport qualification and preserved coarse-grid failure.
- `rich_api.json`: scalar/rich PV and provenance checks.
- `timing.numpy.json`: matched-grid cold engine timings and result differences.

## Scope

This extends native MC/PDE model routes. Production QUAD retains its GBM model
scope. Rebased continuous KO-reset, first-hit time distributions, arbitrary
model-parameter accuracy guarantees and the old approximate forward-density
route remain outside this repair. Changes are uncommitted.
