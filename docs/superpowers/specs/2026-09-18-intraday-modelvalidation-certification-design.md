# Intraday engine certification through modelvalidation

Date: 2026-09-18. Status: **Specification for implementation.** The certification policy below was explicitly requested by the user; the detailed migration has not been implemented by this document.

Source baseline: `5bd9671d` on `worktree-intraday-plan1`. Existing daily-KI design and plan files, numerical evidence, and local example work remain intact.

## 1. Decision and precedence

**Certify engines offline by product and scenarios in `quantark.modelvalidation`; use those engines without additional certification checks during pricing.**

The release procedure is:

```text
Product + scenario study + candidate configurations
    -> independent reference and numerical convergence evidence
    -> shared modelvalidation gates and candidate decisions
    -> banked certificate, reports and CI anchors
    -> release and ordinary engine use
```

There is no certificate-to-pricing permission step. `value_intraday`, session pricing, batch pricing, spot curves, portfolio aggregation, and Greek calculation must not read a certificate, match a request against evidence, or run a certification gate. Neither a startup registration check nor a cached admission token may replace the removed per-request checks.

This decision supersedes the runtime qualification requirements in the [original intraday design](../plans/2026-09-15-intraday-pricing-design.md), the [daily-KI design](2026-09-17-intraday-daily-ki-certification-design.md), and the runtime applicability matcher recommended in the [2026-09-18 research review](../reviews/intraday-certification-2026-09-18/RESEARCH.md). Their economic conventions, numerical findings and historical evidence remain useful. They no longer prescribe runtime certificate matching.

This is an ownership and release-policy migration. Removing a gate does not fix a gamma error or turn a previous inconclusive measurement into a pass.

## 2. Scope and invariants

The migration covers all intraday certification: reference implementations, reference qualification, product fixtures, scenario generation, candidate ladders, error budgets, decisions, evidence schemas, reporting, banking, amendments and anchors. These belong to `modelvalidation`, its examples, its tests, and its documentation.

The following requirements are mandatory:

1. Intraday uses the same `CertificationStudy` pipeline, builder registry, `run`/`amend`/`anchors` commands, certificate format, reports and decision names as other engines. There is no replacement intraday certificate service, runner or registry.
2. Certification identifies a product, model, scenario suite, quantities, engine implementation and resolved configuration. It is release evidence for that declared study, not permission for one trade, one timestamp or one spot.
3. Runtime pricing has no dependency on banked evidence or on `quantark.modelvalidation`. Successful execution is possible when no certificates or certification tooling are installed or accessible.
4. Studies evaluate the same estimators used by normal pricing. Remove `certify=True/False` and equivalent bypasses; validation does not need a privileged execution path.
5. Keep input, model and algorithm validity checks. Missing fixings, future checkpoints, unsupported monitoring, non-finite calculations, failed solves and genuinely undefined derivatives still produce the appropriate error or result status.
6. Preserve the distinction between point derivatives and finite desk moves, including units, bump sizes, frozen-market rules and actual theta steps.
7. Preserve historical evidence and existing shared certificates. Do not relabel old Gate C results as `ADMITTED`, overwrite certificate digests, or modify old study semantics in place.
8. Numerical budgets are requirements fixed before candidate evaluation. Do not widen them, drop difficult cases, or change defaults solely to obtain admission.

New pricing models, new monitoring operators, payoff changes, and numerical fixes are outside this migration unless separately identified and certified. In particular, continuous-monitoring QUAD remains unsupported where its intraday interval crossing operator is not implemented.

## 3. Architecture and ownership

```mermaid
flowchart TB
    subgraph Offline[Offline validation and release]
        S[Product studies and scenarios] --> M[modelvalidation pipeline]
        R[Independent references and controls] --> M
        C[Candidate adapters and refinement ladders] --> M
        M --> E[Certificate and reports]
        E --> A[CI anchors and release review]
    end
    subgraph Runtime[Normal pricing]
        Q[Intraday request] --> X[Resolve clocks, events and state]
        X --> P[Engine and Greek estimators]
        P --> V[Values, conventions and numerical diagnostics]
    end
    C -->|ordinary execution API| P
```

There is deliberately no dependency from runtime to the offline block. The shared package may import pricing code to evaluate candidates; pricing code may not import the shared package to decide whether an output is allowed.

| Owner | Responsibilities after migration |
|---|---|
| `quantark.intraday` | Timestamp resolution, immutable pricing context, lifecycle and cash ledger, route dispatch, price and Greek algorithms, numerical diagnostics, functional capability descriptions |
| `quantark.modelvalidation` | Shared study definitions, reference and candidate protocols, uncertainty accounting, gates, decisions, provenance, checkpointing, reports, amendments and anchors |
| `quantark.modelvalidation.builders` | Product-specific intraday study adapters; construction of markets, schedules, states and candidate configurations |
| `quantark.modelvalidation.references` (new) | Reusable independent Gaussian and intraday RQMC reference implementations, with explicit numerical limitations |
| `example/modelvalidation` | Declarative product studies and documented commands |
| `test/modelvalidation` | Reference controls, study/runner tests, gate tests, banked evidence guards and deterministic anchor replay |
| `test/intraday` | Runtime behavior, clocks, fixings, state, cashflow, estimator semantics and integration regressions |
| `docs/modelvalidation` | Release procedure, current certificate bank and clearly separated historical evidence archive |

Do not keep production study builders importing `test.intraday`, and do not move the pricing estimators themselves into validation. Small fixtures needed by runtime semantic tests can remain test-local; expensive certification orchestration cannot.

## 4. Product studies and coverage

### 4.1 Unit of certification

A candidate decision belongs to `(study, candidate configuration)`. A study specifies product and monitoring variants, model assumptions, all required scenarios and quantities, reference policy and tolerances. The certificate records all of these explicitly.

Use product builders in the existing registry, for example `equity.snowball.intraday.pde` and `equity.snowball.intraday.quad_v2`. These names register offline builders only. Runtime engine registration must not consult them.

The first complete migration study is `snowball-intraday-daily-ki-bsm`, defined by `example/modelvalidation/snowball_intraday_daily_ki_bsm.yaml`. PDE and QUAD V2 are separate candidates; their default and refined configurations receive separate names and decisions. Passing at a refined configuration does not admit the default configuration.

Other product suites use the same pattern:

| Product family | Required study distinctions |
|---|---|
| Snowball | Daily discrete KI, other discrete schedules, continuous KI where implemented, long remaining gaps, already-KI and terminated states |
| Phoenix | Coupon and KI/KO schedules, coupon memory, missed coupons, pending and overdue contractual cashflows |
| KO-reset Snowball | Reset terms, absolute post-KI schedules, checkpoint/state handoff |
| Digital | Terminal fixing, positive and zero remaining variance, expiry phase and pending settlement |
| Barrier and one-touch | Discrete, terminal and supported continuous monitoring as distinct cases; analytical formula coefficient restrictions |

The migration inventory must account for every existing Gate C fixture and claimed output, including analytical and MC routes. Shared-kernel reuse is not evidence for an untested product. A study can bank an inconclusive result; the release report must distinguish that from an admitted claim. Functional runtime capability remains independent of this report.

### 4.2 Daily-KI scenario matrix

Carry forward the current daily-KI contract as a reproducible baseline: initial/strike 100, KI 75 at every SSE close, monthly KO 103, 12% coupon/rebate, no principal, flat volatility 20%, rate 3%, yield 1%, and the existing `desk` variance profile. Its historical 2026-09-10 valuation becomes one scenario cohort, not a licensed date.

Expand the study with a deterministic, versioned case generator. Bank the fully expanded case list; a generator name alone is insufficient provenance.

| Axis | Required cases |
|---|---|
| Time to event | Existing 6h, 1h, 15m, 5m, 1m, 10s and 1s daily-KI horizons; explicit event instants before/after; clock/session boundaries and zero-variance segments |
| Calendar | More than one ordinary KI day, a simultaneous KI/KO day, maturity, and weekend/holiday gaps with real contractual schedules |
| Spot | Both KI and KO neighborhoods; existing bp and diffusion-width offsets; equality; interior and far-field controls; gamma zero crossings and cancellation regions |
| Market | Baseline, low/high volatility, distinct rate and carry levels; deterministic term structures on routes supporting them |
| Clock | `desk`, `uniform`, `sessions_only`, and supported explicit fixing times, with their coefficients recorded |
| State/cash | Alive before KI, already KI, KO/terminated, pending settlement, and provisional versus confirmed fixing provenance |
| Product terms | Remaining tenor and event count, barrier schedules, principal/accrual/settlement variants supported by the product |
| Execution | Scalar and session execution, batch parity, optimized spot curves and portfolio composition |

Use targeted stress cohorts and predeclared interaction cases instead of an indiscriminate Cartesian product. Include holdout spots/times selected before results are inspected. Record any excluded combination with its mathematical or capability reason. Changes to coverage require a new study revision; a failed run cannot shrink its matrix retrospectively.

Every numeric claim includes PV as well as the requested Greeks. An error in PV that cancels under differentiation must remain visible.

At an event discontinuity, the expected point Greek can be undefined. Such cells belong to explicit semantic assertions with expected status and reason; do not insert zero, omit them after evaluation, or count them as numeric Greek passes. Unexpected undefined results in numeric cells block admission. Semantic assertion results are reported with the study and must pass for release.

Finite scenario coverage provides empirical evidence over the declared study. Neither the certificate nor runtime documentation may describe it as a proof of a uniform error bound for every possible contract and market.

## 5. Quantities and shared schema changes

### 5.1 Explicit estimands

The current shared schema accepts only `pv`, `delta` and `gamma`. Introduce a shared schema 2 with quantity definitions; do not put intraday interpretation into magic quantity strings in report code.

Each quantity definition contains a stable ID, output field, convention, raw unit, economic scaling rule, bump/step specification where relevant, and cashflow scope. Point and desk versions have distinct IDs, for example `point_gamma` and `desk_gamma`; `price` is mapped to the shared `pv` ID.

| Quantity | Target |
|---|---|
| `pv` | Total request PV with contingent claim and pending ledger reconciled |
| Point delta/gamma | First/second spot derivatives at the requested spot, holding the declared conditional state and other inputs fixed |
| Desk delta/gamma | The declared central finite differences at the recorded absolute/relative spot bumps |
| Point vega/rho/dividend rho | Local market sensitivities under the declared curve/volatility perturbation; a finite-difference estimator records its step and approximation semantics |
| Desk vega/rho/dividend rho | The existing finite move/PnL convention, preserving sign, direction and move size |
| Point theta | Local time derivative inside one event/clock/coefficient segment, with the declared side and unit |
| Desk theta | Frozen-market value roll over its recorded actual interval, with requested interval and event-boundary adjustment disclosed |

Certifying a desk gamma does not certify point gamma. A fixed 1% RQMC spot bump cannot be silently substituted for a point derivative. Near the one-second daily fixing, that bump can span more than one hundred diffusion widths.

### 5.2 Shared types and compatibility

Schema 2 extends the existing framework, with the following data carried through loading, checkpoints, decisions, reports, banking, amendment and anchor replay:

| Area | Required additions |
|---|---|
| Study | Quantity definitions, per-quantity bounds, reference/error policy, semantic assertions and declared scenario-generator version |
| Case | Serializable intraday context parameters: valuation timestamp/timezone, phase, fixing schedule overrides, calendar/profile identity, history/checkpoint, assumptions and pending cash |
| Reference estimate | Reference kind, estimator identity, value, sampling uncertainty, deterministic/bump bias evidence, uncertainty method and unresolved components per quantity |
| Candidate result | Target values, actual status/convention, resolved settings, refinement axes/rungs, achieved solver diagnostics; uncertainty for stochastic candidates |
| Evidence | Complete expanded cases, cashflow/model conventions, implementation provenance and source/configuration digests, reference controls and gate-policy version |

Use typed stochastic, deterministic and analytical reference records. Deterministic estimates must not be represented as identical MC batches with zero standard error. Analytically exact formulas still record their assumptions and relevant floating-point diagnostics.

Schema 1 studies and banked certificates retain their existing arithmetic, identity projections, reports and replay behavior. Version dispatch must support reading them; changing the `SCHEMA` constant alone is insufficient. A new schema-2 run creates new evidence. It does not rewrite old certificates or convert old `REJECTED`/`INCONCLUSIVE` decisions.

All schema-2 studies, including future day-level studies, use the same quantity/reference/gate implementation. This extension must not branch on “intraday” to select a separate policy.

A schema-2 amendment requires a schema-2 parent with compatible estimands and gate policy. The first migration from Gate C or schema 1 is a full run. Record implementation fingerprints or equivalent source provenance so interrupted runs cannot silently reuse checkpoints after a code change.

## 6. Reference and convergence policy

### 6.1 Independent reference hierarchy

RQMC remains the normal independent benchmark for PDE and QUAD product studies. Intraday builders must use the actual observation schedule and exact interval integrated rate, carry and variance where the deterministic BSM assumptions allow it:

```text
log(S_next/S) = integrated_rate - integrated_yield
               - 0.5 * integrated_variance
               + sqrt(integrated_variance) * Z
```

Daily KI is observed at daily fixings. Do not add continuous-barrier crossing corrections between those fixings. A continuous-monitoring study requires its own valid first-passage treatment.

Use independent RQMC scrambles for uncertainty, with common random numbers within a bump pair. Point-Greek references require a separately assessed derivative estimator: conditional smoothing, likelihood-ratio/hybrid differentiation, or a demonstrably local bump-limit sequence. Sampling standard error alone does not cover finite-bump bias.

Retain the independent Gaussian-transition reference under `modelvalidation.references`. It is valuable for dense, short-horizon gamma studies, but must satisfy explicit analytical and independent RQMC controls before acting as the primary reference in a declared cohort. Reference selection is fixed before candidate comparisons. If RQMC cannot resolve point gamma, record that limitation; a qualified deterministic reference may cover the cohort only with its own disclosed error evidence. Do not average disagreeing references or select the one nearest the candidate.

The reference plan must include analytical single-event price/delta/gamma controls, short multi-event examples and selected independent RQMC comparisons. Separately test clock, payoff, state and ledger semantics with independently specified examples: sharing a resolved context or payoff function can hide a common economic bug.

MC candidates need independent analytical/Gaussian controls or an independently constructed simulation reference. Comparing the same MC implementation with more paths does not establish product implementation correctness. Any stochastic candidate uncertainty enters the shared comparison explicitly; fixed-seed replay is only a regression anchor.

### 6.2 Numerical error evidence

For each reference quantity, record sampling error, representation/discretization error, truncation/tail error, bump/time-step bias and material floating-point limitations. Missing required error evidence makes the comparison unresolved; it must not be replaced by zero.

Use at least three levels to inspect convergence and additional refinement or independent controls where the order is unstable. `abs(fine-medium)/3` is permissible only with evidence supporting second-order asymptotic behavior for that quantity and scenario. A converged PV or an accurately differentiated interpolation polynomial does not prove converged gamma. Oscillation, barrier/grid phase sensitivity, and gamma cancellation require explicit controls.

Candidate ladders use the shipped target setting and declared coarser/finer probes on relevant axes: PDE space/time/readout placement and damping, QUAD density/order/domain, and finite-difference bump or theta step. Preserve actual settings and values at every rung. A finer validation rung must not silently replace the output of the configured candidate.

All certification ladders and their budget decisions execute offline. The runtime may retain adaptive work that is part of the declared numerical algorithm, but it must not run these studies, compare against a reference, or enforce the study's economic budget.

### 6.3 RQMC intervals and resources

Estimate uncertainty across independent randomizations, not individual quasi-random paths. New studies declare a replicate-level interval method and confidence level, with its assumptions. A Student-t interval is an empirical statistical construction, not an unconditional rigorous bound.

For the initial migration, use an independent pilot to choose a frozen production replication/path budget. Do not repeatedly inspect agreement and stop on a favorable pass. A future sequential policy must be implemented and assessed in the shared framework before use.

Assign deterministic, independent case substreams for new studies so aggregate standard errors do not silently assume independence while reusing the same scrambles across cases. If a study deliberately couples cases, retain replicate-level covariance and use it in aggregate uncertainty. Preserve schema-1 seed behavior for old evidence replay.

Use durable checkpoints, bounded reference caches and conservative worker limits. Default long runs to one worker and one BLAS/OpenMP/Numba thread; inspect available CPU/memory before increasing concurrency. Record resource exhaustion or incomplete work as unresolved/error evidence. Quick runs and partially completed checkpoints cannot be banked as admitted certification.

## 7. Shared gates and certificate decisions

### 7.1 Budgets

Express every gate in a stated economic or normalized unit. Keep the current shared hedge-contract scale available; add general per-quantity scaling for the intraday measures. Reusing the framework does not require pretending every sensitivity has the same unit or tolerance.

The initial intraday study preserves the existing numerical tolerance levels as an explicit migration baseline:

| Normalized quantity (`N` = initial notional, `S` = recorded spot scale) | Absolute floor | Relative term |
|---|---:|---:|
| PV / N | `1e-6` | none |
| Delta × S / N | `1e-5` | `1e-4` |
| Gamma × S² / N | `1e-4` | `1e-3` |
| Point market sensitivity × 0.01 / N | `1e-6` | `1e-4` |
| Desk market-move PnL / N | `1e-6` | `1e-4` |
| Theta in PnL/hour / N | `1e-6` | `1e-4` |

For a relative term, the cell budget is the maximum of the absolute floor and the relative term times the magnitude of the independently characterized reference quantity. Record the resolved budget and scaling inputs before candidate evaluation. These tolerances come from existing engineering requirements, not from a paper. A different economic requirement needs a separately explained study revision and full certification.

For the initial schema-2 policy, require total reference uncertainty radius at most 25% of the cell budget, and candidate refinement envelope at most 50%. The aggregate signed-error bound is 20% of the cell budget for constant-bound cohorts; with varying cell bounds, aggregate signed error normalized by each cell's budget and require an uncertainty-inclusive magnitude at most `0.2`. The aggregate ratio follows the existing Snowball study's `0.1 / 0.5` allocation. These allocations are proposed shared policy parameters, recorded in the study rather than hidden intraday constants. The radius fraction is distinct from schema 1's standard-error fraction; do not silently reinterpret the old field.

### 7.2 Conservative admission

Let `c` and `r` be candidate and reference values in the same units, `d = abs(c-r)`, `B` the cell budget, and `R` the justified comparison uncertainty radius. For a deterministic candidate, `R` includes the reference's sampling radius and assessed bias components. For a stochastic candidate, it also accounts for candidate sampling uncertainty, with independence or covariance handled explicitly.

```text
upper_error = d + R
lower_error = max(0, d - R)
```

Schema-2 rules are shared by all studies:

- `PASS`: reference precision and required convergence controls are adequate, the candidate refinement envelope meets its allocation, and `upper_error <= B`.
- `FAIL`: adequate evidence establishes `lower_error > B`, or a separately declared algorithm/convergence requirement is demonstrably violated. Record which requirement failed.
- `UNRESOLVED`: precision or convergence evidence is insufficient, or the disagreement interval straddles the budget.
- `ERROR`: construction or execution failed, with the affected case and diagnostics retained.

The candidate refinement envelope is a separate convergence requirement. Do not add it twice to a discrepancy that already compares the configured candidate with the reference. Do not describe an envelope failure as proof of a particular true Greek error.

Apply the same uncertainty-aware logic to the aggregate mean signed error, using the declared cohort weights/covariance and bias components. Aggregate signed errors only within compatible quantities/conventions. Per-cell gates remain mandatory so cancellation cannot hide local failures.

Decisions keep the existing names: `ADMITTED` requires every required numeric cell and aggregate to pass and every semantic assertion to hold; an established failure yields `REJECTED`; otherwise incomplete or unresolved evidence yields `INCONCLUSIVE`. Reports retain per-quantity results even when one quantity blocks the candidate decision.

The old intraday inequality `d <= B + R` is removed. Uncertainty consumes the admission budget; it does not enlarge it. Schema 1 keeps its historical verdict mapping; schema 2 explicitly distinguishes failure to demonstrate admission from evidence of an out-of-budget discrepancy.

## 8. Runtime cleanup and result semantics

### 8.1 Remove certificate coupling

Delete runtime evidence loading, `_QUALIFIED` promotions, `qualified_horizon`, economics/settings/profile matching, spot envelopes, `output_qualification_gap`, `point_output_qualified`, `qualification_scope`, `certificate_gap`, `point_certificate_gap`, and all certificate callbacks.

Move useful study identity/configuration serialization out of `capability.py` into modelvalidation adapters. Preserve runtime context identities needed for immutability, cache correctness or provenance; an identity is not inherently a licensing mechanism.

Remove the `certify` argument from route protocols and implementations. Scalar, session, batch and optimized spot-curve paths must use the same normal estimator behavior. `point_delta` must not be suppressed merely because gamma lacks historical evidence.

`IntradayCapability` describes implemented behavior: product/engine/monitoring support, available conventions and actual model restrictions. Remove qualification status and horizons. A support matrix must not read certificates to promote or demote rows. Release documentation can link to studies separately.

### 8.2 Runtime statuses

| Status | Meaning after migration |
|---|---|
| `ok` | A finite result was successfully computed by the declared algorithm and convention; this is not a per-request certification or error-bound claim |
| `undefined` | The requested mathematical derivative does not exist at the event/spot, with the reason recorded |
| `failed` | A numerical calculation could not produce the requested result, including non-finite output or a required solver step that failed |
| `not_requested` | The quantity was not requested |

Unsupported models/routes/conventions and invalid input continue to use existing typed capability/validation errors. Stop emitting `unqualified` for new runtime results. If compatibility requires accepting that string when reading historical serialized results, retain read compatibility only and document its legacy meaning. Do not silently turn historical missing values into successful results.

Finite-difference point estimators return their approximation with the actual bump/step and method disclosed. Their point-target accuracy is assessed offline. Desk outputs preserve the existing finite-move definition. Missing old certificate coverage is never a reason to change convention, suppress a number or reprice with another engine.

Preserve numerical diagnostics such as PDE achieved mesh/step sizes, actual resource exhaustion, MC standard errors, estimator steps and cancellation estimates. Distinguish hard algorithm failure from a warning or resolution indicator; a heuristic mesh diagnostic must not become a replacement certificate gate. Any hard failure condition retained must correspond to the algorithm's ability to perform the requested computation, with a documented reason.

### 8.3 Embedded analytical-theta gate

`analytical_theta_limit` currently performs a local stencil ladder and applies frozen Gate C theta tolerances during pricing. Removing JSON lookups alone therefore does not complete this migration.

Retain the numerical stencil/Richardson estimator and its segment-safe step selection as the declared theta algorithm. Move its Gate C budget constants, admission comparison and certification ladder assessment into modelvalidation. Runtime can report an available error estimate as diagnostic information; it must not set `ok` by comparing with an offline economic budget. Non-finite output or the absence of a representable valid stencil still produces the appropriate computational status.

`GreekValue.error_budget` must no longer carry a certification budget in new pricing results. It may remain optional for serialization compatibility, with `None` for this use. Offline reports own the tolerance and pass/fail fields. Removing the local budget gate changes output availability and requires explicit regression coverage and offline theta certification.

## 9. Evidence migration and ordinary release workflow

Before deleting `quantark/intraday/evidence`, archive its exact bytes outside the runtime package:

```text
docs/modelvalidation/legacy/intraday-gate-c/<snapshot>/
    README.md
    manifest.json
    gate_c_results.json
    gate_c_greeks.json
    source-records/                 # retained raw records supporting the snapshot
```

The manifest records original paths, SHA-256 hashes, original schema, source/run revision when available, commands/settings, included raw record files and any missing provenance. Label all archived decisions as historical Gate C decisions. These files are not shared certificates and must not be discovered by live certificate/anchor guards.

Do not invent missing raw observations or provenance. Preserve relevant daily-KI runs used to support conclusions, including inconclusive results, with their hashes. Scratch files need not all be copied; the manifest must distinguish included evidence from unretained scratch or unavailable source material.

Then remove the runtime evidence package and its packaging expectations. Historical evidence remains accessible through validation documentation. Old Gate C reports are not imported as admitted certificates; rerun references/candidates under the new policy or explicitly re-evaluate independently sufficient archived raw measurements with complete provenance. The initial intraday migration uses a full shared study run.

Use the existing command pattern:

```bash
python -m quantark.modelvalidation run \
  example/modelvalidation/snowball_intraday_daily_ki_bsm.yaml --quick

python -m quantark.modelvalidation run \
  example/modelvalidation/snowball_intraday_daily_ki_bsm.yaml \
  --out output/modelvalidation --resume

# After banking the full run's certificate and reports:
python -m quantark.modelvalidation anchors \
  docs/modelvalidation/certificates/snowball-intraday-daily-ki-bsm/<date>/certificate.json
```

Bank under the existing `docs/modelvalidation/certificates/<study>/<date>/` layout with `certificate.json`, `report.md`, `report.html`, and `anchors.json`. Keep checkpoints out of the bank. Every report is generated from the same validated payload. Release notes identify the admitted candidate configuration and certificate digest.

Follow the [shared release procedure](../../modelvalidation/RELEASE_PROCEDURE.md): new or changed numerical code needs a full certification; eligible configuration-only changes use amendments; a refactor demonstrated to preserve numerical behavior uses anchors. Retain old certificate files and use the existing explicit supersession mechanism where applicable. Do not manually update anchors to bless new outputs.

CI replays anchors and semantic tests offline. Pricing does not inspect CI results or the certificate bank. A release's numerical claim is controlled by this release process, not by a runtime entitlement system.

## 10. File-level migration plan

Paths below are relative to the repository. New paths are proposed destinations; related helpers may be grouped without changing ownership or contracts.

| Files | Required work |
|---|---|
| `quantark/modelvalidation/{study,yaml_loader,registry}.py` | Add shared schema-2 quantities, case context, per-quantity bounds and typed study policy; retain schema-1 loading |
| `quantark/modelvalidation/{reference,stopping,candidate,gates,decisions,pipeline}.py` | Add typed uncertainty and reference controls, stochastic candidate accounting, interval verdicts, semantic assertions and shared convergence requirements |
| `quantark/modelvalidation/{evidence,anchors,amendment,report,html_report,cli}.py` | Carry schema-2 fields through hashing, resume, reports, banking and replay; maintain old evidence compatibility and CLI pattern |
| `quantark/modelvalidation/builders/intraday_common.py` and product builder modules | Construct resolved case data, normal-API candidate evaluators and refinement configurations; register product-specific intraday builders |
| `quantark/modelvalidation/references/{gaussian_intraday,rqmc_intraday}.py` | Move and qualify the Gaussian reference; build independent schedule-aware RQMC references and analytical controls |
| `example/modelvalidation/*intraday*.yaml` | Add daily-KI and remaining product suites with frozen expanded scenarios and explicit estimands |
| `quantark/intraday/capability.py` | Retain functional support; remove evidence access, qualification promotions, matching and certification-only identity helpers |
| `quantark/intraday/greeks.py` | Remove certificate gap functions/callbacks and theta budget gate; preserve estimators, units, steps, ledger and event semantics |
| `quantark/intraday/engines/{base,pde,quad_v2,mc}.py` and any other route implementing the protocol | Remove `certify` path and evidence suppression; return normal estimator results and actual computational failures |
| `quantark/intraday/{service,batch,result}.py`, session integration and aggregation consumers | Remove scalar/batch licensing paths; update status semantics and compatibility handling |
| `quantark/intraday/evidence/`, `pyproject.toml` | Archive then delete runtime evidence; inspect built wheel/sdist for residual files and dependencies |
| `quantark/intraday/publish.py` | Remove certificate publication; retain only support documentation if needed; shared validation reports own all certification output |
| `test/intraday/{gate_c,reference}/` | Move certification logic/controls into `test/modelvalidation`; eliminate Gate C runners and special environment-variable workflow |
| `test/intraday/test_capability_evidence.py`, publication tests and qualification-based regressions | Replace packaged-evidence assumptions with runtime independence tests; preserve useful numerical and semantic regressions |
| `test/modelvalidation/` | Add intraday study, schema-compatibility, reference, gate, archive and banked-anchor tests |
| `quantark/intraday/README.md`, `docs/execution/intraday-capability-matrix.md`, `docs/modelvalidation/RELEASE_PROCEDURE.md`, example READMEs | Explain the single offline procedure and new `ok` semantics; separate functional support from published study outcomes |
| `example/intraday_benchmark/`, intraday demo scripts | Use ordinary pricing for demos and modelvalidation builders for study work; remove evidence imports and date/spot certificate restrictions |

Leave historical specifications/reviews intact and link to this superseding policy from active documentation. Preserve local user changes during implementation; in particular, do not overwrite the existing daily-KI plan or untracked near-KI example wholesale.

### Migration sequence and dependencies

1. Inventory all Gate C consumers, quantities, studies and artifacts; create and verify the historical archive manifest.
2. Add shared schema-2 contracts and compatibility tests. Establish reference controls and adapters before any admission run.
3. Build the daily-KI product study and migrate the remaining certification fixtures to shared ownership. Freeze matrices, budgets and reference policies.
4. Remove runtime certificate paths in one coherent change, including analytical theta. Update results, docs, demos and runtime tests. Candidate evaluators now call the normal APIs without bypass flags.
5. Run focused numerical/semantic regressions and shared quick studies. Resolve discrepancies in estimator or cashflow semantics before expensive runs.
6. Run full product studies under the resource policy, bank their evidence, and create anchors. Report admitted, rejected and inconclusive candidate configurations explicitly.
7. Remove obsolete Gate C entry points and evidence packaging; verify clean installed-package execution and release documentation.

Steps 2–4 may be developed before full evidence runs, but release claims require step 6. There is no transitional runtime fallback to the old certificate system. If a full study remains inconclusive, publish that validation outcome and continue the numerical investigation; do not restore per-request certification or claim that cleanup repaired the engine.

## 11. Acceptance and release gates

### Runtime independence

- Install the built package into an environment with no certificate bank. Price representative admitted-study products and evaluate all implemented Greek conventions through scalar, session, batch and curve APIs.
- In a separate process, make `quantark.modelvalidation` imports fail and prohibit certificate resource reads. Those pricing calls must still succeed.
- Change valid spot, valuation date, term curves, profile and engine settings beyond the former exact-economics/horizon/envelope matches. Results depend on the normal algorithm and functional support, never on whether a certificate row exists.
- Assert that no runtime route or wrapper accepts a certification flag, callback, token, global switch or implicit evidence cache.
- Retain genuine failure probes: unsupported continuous QUAD, invalid checkpoints/missing fixings, event discontinuities, non-finite computation and actual solver resource failure. These must not be bypassed by the cleanup.

### Numerical and semantic integrity

- Compare available pre-migration raw estimator values against post-migration outputs at identical settings. Any value change, beyond the intentional removal of suppression, requires explanation and the appropriate full numerical certification.
- Test point versus desk Greek units/bumps, theta segment boundaries, zero variance, provisional fixings, post-KI dispatch, pending cash and terminated claims.
- Test scalar/optimized-curve agreement and per-output status behavior, including a successful delta when another requested computation fails independently.
- Check reference gamma controls and convergence, not only PV. Include a deliberate reference/candidate mismatch and an uncertainty-overlap case so neither can become `ADMITTED`.
- Require all planned product/candidate/quantity cells to be represented in the report. Unexpected absence, exception or unresolved precision cannot disappear in aggregation.

### Shared-framework and release integrity

- Existing schema-1 tests, banked certificates, report rendering, hashes and anchors remain valid under their recorded policy.
- New schema-2 round-trip, resume invalidation, amendment, report and anchor tests cover conventions, case contexts, reference kinds and uncertainty components.
- Archived Gate C bytes match their manifest and are excluded from current certificate discovery. Wheel/sdist contain no intraday evidence JSON or legacy certificate package.
- Shared quick runs are visibly non-bankable. Full runs use the recorded source revision, resolved settings and frozen cases, and produce the usual certificate/report/anchor artifacts.
- Every advertised release certification has an `ADMITTED` decision for its actual product study and candidate setting. An incomplete migration inventory or unresolved candidate prevents that claim; it does not create a runtime check.

Run relevant `test/intraday` and `test/modelvalidation` suites serially for integration, then the existing banked-anchor and affected day-level product regressions. Long certification runs are separate from unit tests and resume through the shared CLI. Document commands, machine resources and results in the implementation report. This specification itself requires no pricing fleet and grants no numerical admission.

## 12. Architecture decisions and trade-offs

| Decision | Reason and consequence |
|---|---|
| One offline owner, no runtime certificate matcher | Implements the requested procedure and removes duplicate policies and request-dependent output suppression. Release discipline and tests carry the accuracy claim; callers receive normal engine behavior. |
| Extend shared schemas rather than wrap Gate C | Intraday needs richer estimands and error evidence. Reusing only the CLI around the old harness would preserve a second gate system and is rejected. |
| RQMC plus explicit qualified deterministic/analytical references | Independent simulation remains central; specialized references make short-horizon Greeks measurable. Every method's uncertainty and limitations must be visible in the same certificate. |
| Version gate semantics and preserve old evidence | Conservative admission and uncertainty-overlap handling improve new studies without rewriting historical decisions or breaking evidence chains. |
| Preserve numerical algorithms while removing licensing | Limits unrelated numerical change. Where code mixes an estimator with a certification budget, split them explicitly and certify the resulting normal runtime path. |

The principal risks are accidentally leaving a hidden gate in theta/batch code, changing an estimand while migrating its name, sharing an economic bug with the reference, treating a refinement difference as a proven bound, and corrupting old evidence compatibility. The acceptance checks above target each risk. The module move alone does not resolve the known near-KO small-gamma reference limitations.

## 13. Numerical literature and supporting material

The [research review](../reviews/intraday-certification-2026-09-18/RESEARCH.md) records source inspection limits and the detailed benchmark assessment. Relevant methodological support includes:

- Andricopoulos et al. (2003), [Universal option valuation using quadrature methods](https://doi.org/10.1016/S0304-405X(02)00257-X): transition-density integration for path-dependent claims.
- Pooley, Vetzal and Forsyth (2003), [Convergence remedies for non-smooth payoffs in option pricing](https://doi.org/10.21314/JCF.2003.101): nonsmooth data, Greek oscillations, and the need to measure convergence order.
- Broadie and Glasserman (1996), [Estimating Security Price Derivatives Using Simulation](https://doi.org/10.1287/mnsc.42.2.269): derivative estimators and finite-difference bias.
- Alm et al. (2013), [A Monte Carlo pricing algorithm for autocallables that allows for stable differentiation](https://doi.org/10.21314/JCF.2013.265): conditioned simulation for autocallables; its first-order results do not by themselves establish our gamma estimator.
- L'Ecuyer et al. (2023), [Confidence Intervals for Randomized Quasi-Monte Carlo Estimators](https://doi.org/10.1109/WSC60868.2023.10408613): uncertainty from independent RQMC randomizations and interval construction.

These sources motivate numerical methods and validation controls. The single-module architecture and absence of runtime certificate checks are the user's release-policy decision, not requirements asserted by the literature.
