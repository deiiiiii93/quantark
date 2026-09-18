# Study revision: a deterministic reference for `snowball-intraday-daily-ki-bsm`

Date: 2026-09-18. Status: **implemented on `worktree-intraday-plan1`; awaiting review.** It revises the
[intraday modelvalidation certification design](2026-09-18-intraday-modelvalidation-certification-design.md) for
one study, through that design's own clause (section 6.1): *"No bespoke reference is built unless a study revision
states that it cannot proceed without one; that revision then declares the reference kind, its error model and its
radius policy before any candidate is compared."* This is that revision. No budget changes. No candidate had been
compared with the new reference when the policy below was fixed.

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

**Implementation.** `quantark/modelvalidation/builders/intraday_gaussian.py`, the Gaussian-transition solver that Gate C
used as its control, moved byte-for-byte out of `test/intraday/controls` (the closed forms stay there). Under
deterministic BSM with discrete monitoring the value is an iterated Gaussian convolution between event instants. The
solver holds each state's value as a piecewise-linear function of `ln S` with exact jumps on the barrier knots and
takes every expectation in closed form, so its only error is the `O(h^2)` representation of the smooth continuation.
The last step to the valuation spot is one exact pointwise expectation with exact first and second derivatives: the
point delta and gamma are analytic, not bumped. Registered as `equity.snowball.intraday.gaussian`.

**Independence.** It imports nothing from the QUAD, PDE or MC engine packages. It shares with every arm the resolved
intraday context (events, clock, market) and the product's payoff functions, as the RQMC arm does. It shares with
QUAD V2 the backward-induction formulation, not code or lattice; it shares nothing with the PDE route's method. A
mistake in the shared context is the business of the typed economic controls (plan Task 13); a mistake in the
backward formulation is the business of the qualifier below, which simulates forward.

**Error model.** A nested ladder of four levels, each halving the spacing (production: 4001, 8001, 16001, 32001 knots).
`ladder_estimate` turns each quantity's ladder into a value and a radius. A refinement difference is an estimate, not
a bound, so every branch is declared and the branch taken is recorded per quantity in the evidence:

| rule | when | value | radius |
|---|---|---|---|
| `exact` | the last difference is floating point | finest level | that difference |
| `geometric` | differences share a sign, observed order in [1.5, 2.5], successive Richardson extrapolants contract by 2x or more | extrapolant `E_n = (4 v_n - v_{n-1})/3` | last spread `abs(E_n - E_{n-1})`, the sum of the geometric series it starts; never credited below 1/16 of the previous spread (Richardson gives `O(h^4)` at best) |
| `correction` | order confirmed, extrapolants do not contract (their spread sits on a non-asymptotic floor) | extrapolant | the whole correction `abs(E_n - v_n)`: only that extrapolating did not make the finest level worse, true for any order of 1.3 or more |
| `unextrapolated` | order not confirmed (outside the window, or alternating differences) but still converging | finest level | 3 x the largest movement over the last two doublings (the grid-convergence-index factor for an unconfirmed order) |
| `unbounded` | the last difference grew | finest level | infinite: the cell is `UNRESOLVED` |

The first radius tried, the plain spread of two extrapolants on three levels, under-covered 2 of 12 measured cells
(by 1.6x and 3.6x near the barrier), which is why the rule needs four levels and the `correction` branch.

**Radius policy in the gate.** `R` in section 7.2 is the radius itself: it is a bound, so `interval_k` never
multiplies it. The reference is sharp enough when its radius is at most `bounds.radius_budget_fraction` (0.25) of the
cell budget. This is a new field; `se_budget_fraction` keeps its meaning and is not reinterpreted. In the aggregate
gate the radius of the mean is the **mean** of the cell radii: discretization errors may share a sign across cells,
so they add linearly, never in quadrature.

**Record.** The reference block is typed (`kind`, `values`, `radii`, `undefined`, `evidence`), with no batches, seeds or
standard errors. A gate under a deterministic reference has `se_c: null` and a `radius_c`, so a consumer that forgets
the radius fails loudly instead of reading a zero standard error. A quantity with no value at a case (a derivative or
a roll on an event instant) is `undefined` with its reason; a numeric cell there is an `ERROR`, a declared semantic
cell is judged on the candidate's status as before.

## 3. Qualification by RQMC

The RQMC arm stays in the study as the qualifier (`reference_qualification`, builder
`equity.snowball.intraday.mc_rqmc`). It simulates every case under the study's frozen sampling (32 replicates of
32,768 paths). For every quantity both arms target and the reference defines, the deterministic value must sit within
`max_z = 4` qualifier standard errors plus the reference's radius. A case that fails is **not qualified**: nobody
knows which arm is wrong, so all of its cells are `UNRESOLVED` and the candidate is at best `INCONCLUSIVE`. The
checks, z-scores and the qualifier's bank are in the certificate; the policy is part of the contract, so an amendment
cannot loosen it.

What it can see: an economic or formulation error of about 4 standard errors, 1.4e-2 on PV (1.4 bp of notional) at the
frozen sampling. What it cannot see: a numerical error at the 1e-6 level, which is the ladder's and the closed-form
controls' job, and the point Greeks, for which RQMC has no estimator.

## 4. Evidence gathered before any candidate comparison

All 23 cases, ladder 2001/4001/8001/16001, the rule of section 2, against the pilot's RQMC values:

| quantity | numeric cells | max radius / allowance | cells over the allowance | max abs z | abs z over 2 | over 3 |
|---|---:|---:|---:|---:|---:|---:|
| pv | 23 | 1.99 (`lunch_break`) | 2 | 2.13 | 3 of 21 | 0 |
| desk_delta | 23 | 0.89 | 0 | 3.20 | 2 of 21 | 1 |
| desk_gamma | 23 | 0.75 | 0 | 2.05 | 1 of 21 | 0 |
| desk_theta | 22 | 0.85 | 0 | 2.10 | 1 of 20 | 0 |
| point_delta | 22 | 1.00 | 1 | | | |
| point_gamma | 22 | 0.52 | 0 | | | |

- Rules taken: 90 `geometric`, 31 `correction`, 2 `unextrapolated`, 12 `exact`, none `unbounded`. Observed orders
  cluster at 2.00. The three undefined quantities are exactly the three semantic cells of `on_ki_barrier_at_close`.
- Three cells exceed the allowance with 16001 as the finest level, all on the `correction` branch, whose radius
  quarters with each doubling. The production ladder therefore ends at 32001.
- Coverage: for the 18 cells solved one level further (32001), the next extrapolant moved by at most 0.18 of the
  declared radius.
- The one z beyond 3 (`near_ko_on_ko_day_1s` desk delta, -3.20 at 2,048 paths) fell to -1.25 at 32,768 paths, and PV
  from 2.00 to 1.00: it was the few-path standard error understating itself across the knock-out jump (plan Task 13),
  not a bias. That is why the qualifier is frozen at 32,768 paths.
- Closed form (`test_intraday_snowball_gaussian_reference.py`): a single-event snowball at spots 78 and 101. Price,
  the finite-move delta and gamma, and the analytic point delta and gamma all sit within the declared radius plus
  1e-8 of quadrature noise.
- Cost: about 2 minutes per solved sweep at 16001 and 7 at 32001. A case needs up to three sweeps (base, spot-bumped,
  theta-rolled); the solver's cache keys are deliberately conservative and were left untouched. The production
  reference is about five hours on one worker, against roughly 2e10 paths per case for RQMC.

## 5. What this changes in the design

| design text | now, for this study |
|---|---|
| 5: "A deterministic kind is reserved for a future revision" | implemented: typed record, `radius_c`, linear aggregation |
| 6.1: "RQMC is the reference for every intraday study" | RQMC qualifies; the Gaussian-transition solver is the reference |
| 6.1: "The Gaussian-transition reference ... is not migrated as a certification reference" | migrated to `modelvalidation/builders/intraday_gaussian.py` |
| 6.1, 7.2: point cells are `UNRESOLVED` by construction and outside the decision | the reference targets point delta and gamma; they are certified and enter the decision |
| 6.3: the pilot chooses the reference's path budget | the pilot shows RQMC cannot be the reference, and chooses the qualifier's budget |
| 7.1: `se_budget_fraction` only | plus `radius_budget_fraction`, deterministic references only |
| contract | adds `reference_kind`, `reference_error_model`, `qualification` |

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
   in the gate; nothing in this study needs it yet.
