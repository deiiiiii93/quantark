# Study revision: a deterministic reference for `snowball-intraday-daily-ki-bsm`

Date: 2026-09-18. Status: **implemented on `worktree-intraday-plan1`; revised the same day after
[review](../reviews/intraday-deterministic-reference-2026-09-18/REVIEW.md) (four findings, all valid, all fixed:
[response](../reviews/intraday-deterministic-reference-2026-09-18/RESPONSE.md)).** It revises the
[intraday modelvalidation certification design](2026-09-18-intraday-modelvalidation-certification-design.md) for
one study, through that design's own clause (section 6.1): *"No bespoke reference is built unless a study revision
states that it cannot proceed without one; that revision then declares the reference kind, its error model and its
radius policy before any candidate is compared."* This is that revision. No budget changes. No candidate had been
compared with the new reference when the policy below was fixed, and none has been since the review changed it: the
production run started under the first policy was stopped, unread, when the review arrived.

**Reference selection belongs to each certification study.** Schema 2 supports stochastic and deterministic primary
references through the shared builder, evidence and gate protocols. This revision selects the Gaussian reference
only for `snowball-intraday-daily-ki-bsm`; other studies retain or declare their own references, and RQMC remains
the reference of every study it can resolve. A reference's error policy and any qualification policy are frozen
before candidate comparison and remain part of the certification contract. A deterministic method does not, by
itself, establish an error bound.

## 1. The study cannot proceed on RQMC

The Task 17 pilot ran the RQMC reference on all 23 cases with 32 replicates of 2,048 paths (seed 20260917, substream
seeds disjoint from production). Its standard errors, as multiples of what the gate allows (25% of each cell's budget):

| quantity | cells | median | max |
|---|---:|---:|---:|
| pv | 23 | 537 | 990 |
| desk_delta | 23 | 212 | 1,971 |
| desk_gamma | 23 | 603 | 12,412 |
| desk_theta | 22 | 743 | 314,377 |

Only the two exact cases (`maturity_day`, `terminated_pending_cash`) resolve. More paths do not close the gap. The
measured convergence of the 32-replicate standard error on `ordinary`:

| paths per replicate | pv | desk_delta | desk_gamma | desk_theta | seconds |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 1.50e-2 | 7.51e-3 | 1.74e-2 | 2.16e-2 | 52 |
| 8,192 | 5.49e-3 | 3.22e-3 | 7.67e-3 | 8.06e-3 | 56 |
| 32,768 | 3.38e-3 | 1.85e-3 | 3.98e-3 | 3.97e-3 | 74 |
| slope, last step | -0.35 | -0.40 | -0.47 | -0.51 | |

Scrambled Sobol keeps a rate near `N^-0.7` only until the knock-in and knock-out jumps dominate the variance; past
8,192 paths the estimator is at the plain Monte Carlo rate or worse. PV needs a standard error of 2.5e-5 (a quarter of
1e-6 of notional); from 3.4e-3 at `N^-0.5` that is about 18,000 times more paths, 6e8 per replicate, for this one
cell. A one-second theta roll divides price noise by 1/3600 h and is out of reach at any path count. The design's two
rulings, "RQMC is the only reference" and "the 7.1 budgets are never widened", are jointly infeasible for this
product. Widening is not the way out: at the frozen qualifier sampling the smallest PV budget RQMC could resolve is
about 1.4e-4 of notional, which is `ordinary`'s whole hourly theta, so a widened budget would exceed the intraday
moves the study exists to measure. This revision keeps the budgets and changes the reference.

## 2. The declared reference

**Kind.** `deterministic`. The design reserved this kind (section 5); it is now implemented for every schema-2 study.

**Implementation.** `quantark/modelvalidation/builders/intraday_gaussian.py` is the Gaussian-transition solver that Gate C
used as its control, moved out of `test/intraday/controls` (the closed forms stay there);
`intraday_gaussian_reference.py` is its builder, registered as `equity.snowball.intraday.gaussian`. Under deterministic
BSM with discrete monitoring the value is an iterated Gaussian convolution between event instants. The solver holds
each state's value as a piecewise-linear function of `ln S` with exact jumps on the barrier knots and takes every
expectation in closed form, so its discretization error is the `O(h^2)` representation of the smooth continuation.
The last step to the valuation spot is one exact pointwise expectation with exact first and second derivatives: the
point delta and gamma are analytic, not bumped. One change was made to the solver after the review: its local grid at
the first event has exactly `(points - 1) / 4 + 1` knots, nested with the global ladder. The Gate C rule,
`max(points // 4, 201)`, repeated one local grid across a ladder of small levels (review R1).

**Independence.** The solver imports nothing from the QUAD, PDE or MC engine packages. It shares with every arm the
resolved intraday context (events, clock, market) and the product's payoff functions, as the RQMC arm does. It shares
with QUAD V2 the backward-induction formulation, not code or lattice; it shares nothing with the PDE route's method.
A mistake in the shared context is the business of the typed economic controls (plan Task 13); a mistake in the
backward formulation is the business of the qualifier below, which simulates forward. The builder's identity hashes
the runtime trees and its own two files, never the candidates' file, and the reverse, so neither arm's edits
invalidate the other's banked work.

**What a radius is.** Every quantity's radius declares its kind, and the framework refuses one that does not:

- `analytical`: the solve names an **exactness basis** and the radius is that basis's own error. Three bases exist.
  `terminated`: no contingent claim remains, the value is the pending receivables discounted in closed form.
  `decided_at_valuation`: the only remaining instant is the valuation instant, decided at the known spot.
  `single_instant_quadrature`: one instant remains, so the value and its point derivatives are single Gaussian
  integrals; they are taken by adaptive Gauss-Kronrod quadrature split at every jump and kink, with no grid at all,
  and the radius is QUADPACK's error estimate (never credited below 1e-13 of the integral's scale) plus the bounded
  tail. An unconverged piece gives no basis.
- `calibrated_estimate`: everything else. The radius is the ladder policy's estimate plus the errors every level of a
  ladder shares. **It is a calibrated numerical estimate, not an analytical bound**: finitely many contracting
  extrapolants do not prove the next ones contract, and a safety factor on recent differences is an estimate too.
  It is accepted as the gate's allowance only with the calibration evidence below, recorded per quantity.

**The ladder policy** is one structured, versioned object (`LadderPolicy`, version 2). It executes and it serializes
itself into the contract's `reference_error_model`, so the certificate describes the policy that ran, and changing a
parameter (the order window, a safety or contraction factor, the stagnation floor) changes the contract and refuses
an amendment (review R4). A ladder has five nested levels, each halving **every** spacing in play (production:
2001, 4001, 8001, 16001, 32001 global knots; 501 to 8001 local knots). The builder records the grids each level
actually used and refuses a ladder that repeats or unevenly refines one (review R1). The four finest levels give the
estimate:

| rule | when | value | radius |
|---|---|---|---|
| `exact` | the last two differences are floating point **and** the solve names an exactness basis | finest level | the basis's error |
| `stagnant` | the last two differences are floating point and nothing explains why | finest level | infinite: equal values do not establish exactness |
| `geometric` | differences share a sign, observed order in [1.5, 2.5], successive Richardson extrapolants contract to half or less | extrapolant `E_n = (4 v_n - v_{n-1})/3` | the last spread `abs(E_n - E_{n-1})`, never below 1/16 of the previous one (Richardson gives `O(h^4)` at best) |
| `correction` | order confirmed, extrapolants do not contract (their spread sits on a non-asymptotic floor) | extrapolant | the whole correction `abs(E_n - v_n)` |
| `unextrapolated` | order not confirmed (outside the window, or alternating differences) and the last difference did not grow | finest level | 3 x the previous difference (the grid-convergence-index factor for an unconfirmed order) |
| `unbounded` | the last difference grew | finest level | infinite |
| `uncalibrated` | the same rule on the ladder **without its finest level** did not cover this value with its own radius | as estimated | infinite |

The last row is the in-ladder calibration: the coarsest level exists to apply the rule one level down and ask whether
its radius covered the better value. A rule that was too optimistic one level down is not trusted at this one, and an
infinite radius one level down is no evidence. The first radius tried during this work, the plain spread of two
extrapolants on three levels, fails exactly this test on 2 of 12 measured cells (by 1.6x and 3.6x near the barrier).

**Shared errors** are what a spacing ladder cannot see, because every level carries them. They are computed per case,
recorded by component, scaled to each quantity (pv 1; desk delta `1/(S h)`; desk gamma `4/(S h)^2`; desk theta
`2/step`; point delta `E|Z|/(S sd_1)`; point gamma `(E|Z^2-1|/sd_1^2 + E|Z|/sd_1)/S^2`) and added to the radius:
*truncation* (analytical bound: the value span times the reflection-principle mass beyond the flat-extended global
grid, from the spot bumped 1% either way and after allowing the whole drift, plus `2 Phi(-15)` for the local grid),
*saturation* (analytical bound: `4 Phi(-9)` of the span per instant, for ramps and jumps set to 0 or 1 beyond nine
standard deviations) and *roundoff* (an estimate: 16 eps times the span per instant). On `ordinary` they total
3.7e-11 on PV against an allowance of 2.5e-5; roundoff is nearly all of it (saturation 4.6e-15, truncation 1.7e-29).

**Validity domain.** Deterministic Black-Scholes coefficients, discrete monitoring, snowball payoffs without continuous
knock-in or Phoenix coupons (the solver refuses both), every level refining every discretization in play.

**Radius policy in the gate.** `R` in section 7.2 is the radius itself, so `interval_k` never multiplies it. The
reference is sharp enough when its radius is at most `bounds.radius_budget_fraction` (0.25) of the cell budget. This
is a new field; `se_budget_fraction` keeps its meaning and is not reinterpreted. In the aggregate gate the radius of
the mean is the **mean** of the cell radii: discretization errors may share a sign across cells, so they add
linearly, never in quadrature.

**Record.** The reference block is typed (`kind`, `values`, `radii`, `radius_basis`, `undefined`, `evidence`), with no
batches, seeds or standard errors. The evidence holds, per quantity, the ladder values, extrapolants, rule, observed
order, basis, ladder radius, shared error and calibration, and per case the grids used and the shared-error
components. A gate under a deterministic reference has `se_c: null` and a `radius_c`, so a consumer that forgets the
radius fails loudly instead of reading a zero standard error. A quantity with no value at a case (a derivative or a
roll on an event instant) is `undefined` with its reason; a numeric cell there is an `ERROR`, a declared semantic cell
is judged on the candidate's status as before.

## 3. Qualification by RQMC

The RQMC arm stays in the study as the qualifier (`reference_qualification`, builder
`equity.snowball.intraday.mc_rqmc`). It simulates every case under the study's frozen sampling (32 replicates of
32,768 paths). For every quantity both arms target and the reference defines, the deterministic value must sit within
`max_z = 4` qualifier standard errors plus the reference's radius.

Qualification is an **eligibility condition**. A case that fails, whose qualifying arm errored, or where nothing could
be compared, is not qualified: nobody knows which arm is wrong. Its cells are `UNRESOLVED`, their gates are kept only
as `diagnostic_gate` and are not decision-eligible, and they feed no aggregate, so an unqualified reference can
produce neither a confident pass nor a confident rejection (review R2). Failures in qualified cases still reject.

The whole qualification policy is in the contract and immutable under amendment: `max_z`, the targets, the effective
sampling (paths, replicates, seed, scheme, bump) and the arm's class and configuration. Fewer paths or replicates
widen `max_z x SE` and make qualification easier, so a weaker qualifier is a new study revision and a full
certification, never an amendment (review R3). Source fingerprints stay in the evidence identity, where they
invalidate reuse without being mistaken for numerical policy.

What it can see: an economic or formulation error of about 4 standard errors, 1.4e-2 on PV (1.4 bp of notional) at the
frozen sampling. What it cannot see: a numerical error at the 1e-6 level, and the point Greeks, for which RQMC has no
estimator. It therefore cannot vouch for a radius at the certification budget; the analytical bases, the in-ladder
calibration and the closed-form controls do that.

## 4. Evidence gathered before any candidate comparison

**The reviewed policy, all 23 cases through the production builder**, ladder 1001/2001/4001/8001/16001 (one level
coarser than production, so the table is pessimistic):

| quantity | numeric cells | max radius / allowance | cells over the allowance | calibration: max move / radius one level down | observed order |
|---|---:|---:|---:|---:|---|
| pv | 23 | 1.99 (`lunch_break`) | 2 | 0.42 | 1.57 to 2.36 |
| desk_delta | 23 | 0.89 | 0 | 0.46 | 1.99 to 2.01 |
| desk_gamma | 23 | 0.75 | 0 | 0.56 | 1.76 to 2.53 |
| desk_theta | 22 | 0.85 | 0 | 0.38 | 1.90 to 2.15 |
| point_delta | 22 | 1.00 | 1 | 0.48 | 1.99 to 2.01 |
| point_gamma | 22 | 0.52 | 0 | 0.44 | 1.81 to 2.60 |

- 135 numeric cells: 91 `geometric`, 30 `correction`, 2 `unextrapolated`, 12 `exact`; none `stagnant`, `uncalibrated` or
  `unbounded`, so no radius is infinite. 123 radii are calibrated estimates and 12 are analytical: `maturity_day` by
  `single_instant_quadrature` (and `decided_at_valuation` for its rolled price), `terminated_pending_cash` by
  `terminated`. The three undefined quantities are exactly the three semantic cells of `on_ki_barrier_at_close`.
- In-ladder calibration: the rule applied one level down covered every cell's value, the move never above 0.56 of
  that radius. It is not a formality: on the wiring ladder (101 to 1601) `ordinary`'s desk theta fails it and is
  `uncalibrated`.
- Out-of-ladder coverage, all 135 cells: the production reference (finest level 32001, policy already frozen) against
  this pre-flight (finest level 16001). No production value moved by more than 0.34 of the radius declared one level
  coarser: at most 0.34 on the `geometric` branch, 0.004 on `correction`, 0.02 on `unextrapolated`, and exactly 0 on
  the analytical cells. This is the out-of-sample check the in-ladder calibration cannot be, because it shares no
  level with the estimate it tests.
- Three cells exceed the allowance with 16001 as the finest level, all on the `correction` branch, whose radius
  quarters with each doubling. The production ladder therefore ends at 32001.
- Closed form (`test_intraday_snowball_gaussian_reference.py`): a single-event snowball at spots 70, 75, 78 and 101,
  the review's failing case. Price, the finite-move delta and gamma, and the analytic point delta and gamma all sit
  within the declared analytical radius (about 3e-11 on price) plus 1e-8 of stencil noise; the grid solver's finest
  level agrees with the quadrature to better than 1e-3 and is recorded as a cross-check. A two-instant case, which has
  no closed form, is covered against a ladder four times finer.
- Shared errors are negligible here and are recorded anyway: 3.7e-11 on `ordinary`'s PV.

**Against the pilot's RQMC values** (first policy, ladder 2001 to 16001; the values differ from the reviewed policy's
at the 1e-9 level): of 83 comparisons none is beyond 4 standard errors, 7 are beyond 2 and one beyond 3. That one,
`near_ko_on_ko_day_1s` desk delta at -3.20 with 2,048 paths, fell to -1.25 at 32,768 paths, and its PV from 2.00 to
1.00: the few-path standard error understates itself across the knock-out jump (plan Task 13); it was not a bias.
That is why the qualifier is frozen at 32,768 paths.

**Cost.** About 2 minutes per solved sweep at 16001 and 7 at 32001. A case needs up to three sweeps (base,
spot-bumped, theta-rolled); the solver's cache keys are deliberately conservative and were left untouched. The
production reference is about five hours on one worker, against roughly 2e10 paths per case for RQMC.

## 5. What this changes in the design

| design text | now, for this study |
|---|---|
| 5: "A deterministic kind is reserved for a future revision" | implemented: typed record with `radius_basis`, `radius_c`, linear aggregation |
| 6.1: "RQMC is the reference for every intraday study" | RQMC qualifies; the Gaussian-transition solver is the reference |
| 6.1: "The Gaussian-transition reference ... is not migrated as a certification reference" | migrated to `modelvalidation/builders/intraday_gaussian.py` (solver) and `intraday_gaussian_reference.py` (builder) |
| 6.1, 7.2: point cells are `UNRESOLVED` by construction and outside the decision | the reference targets point delta and gamma; they are certified and enter the decision |
| 6.3: the pilot chooses the reference's path budget | the pilot shows RQMC cannot be the reference, and chooses the qualifier's budget |
| 7.1: `se_budget_fraction` only | plus `radius_budget_fraction`, deterministic references only |
| contract | adds `reference_kind`, `reference_error_model` (the structured policy), `qualification` (threshold, targets, sampling, seed, method) |
| 7.2 decisions | a case whose reference is not qualified has no decision-eligible gate: `UNRESOLVED`, outside every aggregate |

Unchanged: the 7.1 budgets, the three-way verdicts, envelopes and convergence axes, per-case substream seeds,
schema-1 arithmetic and wire format (the golden tripwire passes), banking, anchors and amendments (an amendment
carries a solved reference and its qualification together, or re-gates the case).

## 6. Open items

1. The solver's sweep cache treats a spot-bumped market and a theta-rolled clock as new markets. Both are the same
   function of `ln S`; sharing them would cut the reference's cost about threefold. It was left alone because the
   control's keys are part of what Gate C validated.
2. The MC route ignores a knock-in decided exactly at the maturity close under BEFORE (MC 12.0 where PDE and the
   lifecycle give -30.0 at spot 70). No study case reaches that state; the qualifier would not see it.
3. A stochastic candidate (the MC route) against a deterministic reference needs the candidate's own sampling radius
   and any covariance in the gate. It is deferred, and nothing here claims to support it.
4. One certificate has one primary reference. Different methods per case or per quantity would need a composite
   reference with its own selection and applicability evidence; separate named studies already provide that choice.
5. The ladder radius stays an estimate. An a-posteriori bound is available in principle (the representation error of
   each step is measurable at the cell midpoints, and expectation is a sup-norm contraction), and would replace the
   calibration argument with a proof; it is a change to the solver and is not attempted here.
