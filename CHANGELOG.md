# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).
During 0.x the public API may still change between minor versions.

## [Unreleased]

## [0.4.7] - 2026-08-25

Settlement-date-aware payoff discounting across the equity option stack
(spec: `docs/superpowers/specs/2026-07-29-equity-option-settlement-date-support-design.md`,
plan: `docs/plans/2026-07-29-equity-option-settlement-date-support.md`).
Determination and payment are now two separate clocks: underlying dynamics,
monitoring and optimal exercise stop at determination; every cashflow
discounts at its own payment time with curve-exact discount factors.

Alongside it, autocallable batch pricing gets its performance pass — the PDE
solve loop no longer re-serializes the product on every time step, and the
QUAD event-stats recursion gains a fused KI walk plus an opt-in forward
transition-density mode (`docs/autocall-engine-perf/`).

### Added
- `quantark.asset.equity.settlement`: shared settlement timing kernel —
  `SettlementConvention` (business-day / calendar-day / year-fraction lag
  with business-day adjustment and calendar), `SettlementRequest`,
  `ResolvedPaymentTiming` (determination/payment times + curve-exact
  `determination_df`/`payment_df`/`delay_df`), `SettlementResolver`
  (`resolve_contingent` / `resolve_pending`). No fabricated dates: a
  date-based convention on a time-only determination raises.
- Product contract: `BaseEquityOption.settlement_convention`; the existing
  terminal `settlement_date` is now honored by engines as the
  terminal-payment override; per-record `ObservationRecord.settlement_date` /
  `settlement_time` drive event-cashflow payment (KO redemptions, coupons,
  hit-paid rebates) across analytical, MC, PDE and QUAD engines for the
  BSM / Local Vol / Heston / SLV families.
- American exercise obstacle reflects delayed cash receipt in PDE and LSMC
  (node-specific `delay_df`); analytical American approximations raise a
  capability error for settlement forms they cannot represent.
- Engines declare settlement/timing capabilities and raise precise
  capability errors before numerical work rather than silently pricing at
  determination.
- Lifecycle: pending-cashflow ledger
  (`quantark.asset.equity.lifecycle.cashflows`) carries
  determined-but-unpaid receivables at `amount * DF(valuation, payment)`
  through payment; `price(..., lifecycle_state=...)` propagates through
  engines, positions, portfolio valuation, Greeks and the consolidated
  replay backtest (settlement decomposition; KO termination waits for
  terminal cash to land).
- Payment-aware event statistics: determination/payment time arrays with
  undiscounted and discounted expected cashflows that reconcile to engine
  PV.
- QUAD autocallable engines: opt-in `QuadParams.event_stats_mode="forward_density"`
  computes the event distribution from a forward transition-density march
  (2–3 surfaces instead of one indicator row per KO observation). `npv` is
  unchanged (always the backward value solve); distribution fields differ
  within the banked validation tolerances
  (`docs/autocall-engine-perf/FORWARD-DENSITY-EVIDENCE-2026-08.md`).
  KO-reset ignores the flag. Default remains `"stacked"`.

### Changed
- Snowball/Phoenix PDE: the product cache token is now memoized per solve
  (`SnowballPDESolver._product_token_memo`, cleared at every solve entry).
  The boundary-condition path previously re-serialized the whole product —
  every KO/KI schedule record — once per time step, >90% of a snowball
  solve; bit-identical output, ~7× faster per solve
  (`docs/autocall-engine-perf/FINDINGS-2026-08-24.md`).
- QUAD stacked event stats: the KI-probability recursion now rides the main
  stacked recursion as fused surfaces (bit-identical output, one fewer full
  time loop per `price_with_events`); identically-zero indicator rows are
  skipped in the diffusion/bridge steps and the row-independent bridge
  kernel stack is cached per engine (64-entry bound). Bit-identical output.
- QUAD `price_with_events` now honors `streams` pruning [§11.1], mirroring
  the PDE contract: KI-probability recursion and Phoenix coupon rows are
  skipped when no requested stream reads them. Callers that pass `streams`
  (the execution framework's greek-bump runner does) get `0`/empty for
  pruned distribution fields where QUAD previously returned computed values;
  `ko_probability`/`survival`/`pv` and all requested fields are unchanged.
  `streams=None` keeps the old full computation.
- **The flat dividend/carry sanity bound moves from 20% to 100%**
  (`ContinuousDividendYield`, the BSM process gate, and the market-data
  point model), matching `TermStructureDividendYield`'s own magnitude
  bound. OTC borrow/lending carry (融券率) legitimately exceeds 20% on
  some trades; the 100% ceiling still rejects unit errors (entering
  `26.89` for `0.2689`). Values between 20% and 100% now price instead of
  raising `ValidationError`.

### Compatibility
- Zero-lag identity: with no settlement terms supplied, payment equals
  determination (`delay_df == 1`) and all prices, Greeks, event statistics
  and exercise boundaries are unchanged — verified by the existing suite
  plus the dedicated settlement test files.

### Also in this release
- Constructor normalization forwarding the shared lifecycle fields across
  the concrete equity option products (wip/equity-option-contract).
- Heston temporal regularization in the ADI core
  (wip/heston-temporal-regularization).
- PDE regression coverage ports, per-module reference docs, example suites
  (FX vol models, snowball backtest terms, bucketed-greeks scenarios), and
  modelvalidation certification archival.

## [0.4.6] - 2026-08-20

0.4.5's test suite passed on the ARM64 machine its goldens were frozen on and
failed 28 tests on x86_64 Linux. Chasing that down found one genuine pricing
defect and a set of comparisons that were architecture-dependent by accident.
The published 0.4.5 package is unaffected except as noted below.

### Fixed
- **Parallel-batch pricing of continuously monitored knock-in under Local Vol /
  Heston / SLV Monte Carlo.** The per-step variance the Brownian-bridge crossing
  estimate needs was held in a single engine attribute, written while simulating
  and read while computing payoffs. The dask path (`use_dask=True`) submits every
  batch through `delayed`/`compute` on the threaded scheduler, so batches sat
  between their own generation and their payoffs concurrently on one engine and
  overwrote each other — 20,000 paths over 3 batches is 6667/6667/6666. Now
  scoped per thread, which is where a batch begins and ends. A shape guard made
  this raise rather than price one batch against another's variance, so no wrong
  number could have been returned; serial pricing was never affected.
- `quantark.modelvalidation.anchors`: comparing a banked certificate off the
  banking machine now floors the tolerance at the suite's cross-architecture
  standard (`DEFAULT_REL_TOL` 1e-12 → 1e-9, `DEFAULT_ABS_TOL` 1e-15 → 1e-12).
  Measured across 326 anchors, x86_64-vs-ARM64 drift spans 1e-14 to 2.2e-11
  relative (median 2.0e-12, both signs, no outlier of a different order): last-ULP
  differences in FMA contraction, libm transcendentals and BLAS kernels, amplified
  by an autocallable PDE/QUAD solve marching hundreds of steps through barrier
  events. The bound keeps ~44x margin over that and stays six orders under any
  real numerics change. Applied as a floor on each banked file's own tolerances,
  so already-banked evidence is corrected without its anchored *values* being
  touched. Same-machine comparison remains exact.

### Changed (tests)
- The replay golden gate compares floats against each column's scale rather than
  bitwise. These frames carry bump derivatives, so the rounding noise is set by
  the magnitude of the price being differenced and is near-constant in absolute
  terms down a column: gamma drifts 4.6e-10 against a column max of 43, and the
  same quantity in cash drifts 3.8e-8 against 4,301 — one ratio, ~1e-11. Column
  names and order, row count, and integer, string and date columns still compare
  exactly. New tests pin the tolerance itself: it admits the measured drift,
  rejects 1e-6 of column scale, and every real golden frame is checked against a
  simulated drift locally, because bitwise-identical frames short-circuit before
  any tolerance is consulted and made earlier calibration bugs invisible here.
- Two cross-architecture anchor tests hardcoded the fingerprint
  `{"machine": "x86_64", "system": "Linux"}` — which is the CI machine, so there
  they took the same-machine *exact* branch, the opposite of what they verify.
  They now derive a fingerprint that cannot be the current machine.
- `test_event_stats_pruning` pinned a PV of ~9.9e5 with a `1e-6` absolute
  epsilon, i.e. 1e-12 relative — tighter than the measured drift, passing on
  margin rather than by design. Now on the documented cross-arch tolerance.
- A settlement fixture placed a strike exactly on the parity-implied forward.
  Stage 10 splits OTM puts from calls with `strike < forward` against an
  OLS-regressed forward, so that strike changed side with the last ULP of the
  regression. The grid now keeps every strike clear of the forward.

## [0.4.5] - 2026-08-19

Two programs land together: **model-release certification** — the first
module that decides whether an engine is fit to ship, and the numerics
fixes that decision surfaced — and the **backtest replay consolidation**.

### Model-release certification

#### Added
- `quantark.modelvalidation`: engine-release certification. A study YAML
  declares the product, the market environment, the candidate engines and
  the economic scale; the runner prices every (candidate, case) cell against
  a paired-RQMC benchmark, applies per-cell and aggregate-bias gates, and
  emits a signed certificate, a Markdown/HTML report, and an *anchor file*.
  Anchors are the cheap residue of an expensive run — the deterministic
  engines' own outputs at the certified configurations — so CI can re-check
  in seconds that a released engine still produces what its certificate
  claims (exact on the banking machine, tolerance-based elsewhere).
  `amend` extends a certification's scope while carrying unaffected cells
  forward on identity hashes; scope may grow but never shrink.
  Procedure: `docs/modelvalidation/RELEASE_PROCEDURE.md`.
- Banked flat-BSM certifications for snowball (23 cases), phoenix (14) and
  KO-reset snowball (14), PDE and QUAD both ADMITTED, under
  `docs/modelvalidation/certificates/`, covering the full product feature
  surface: discrete / European / stepping knock-in, step-down and parachute
  knock-out, reverse, airbag, protection types, participation, call rebate,
  `disable_ko_after_ki`, and EXPIRY coupons.

#### Fixed
- **Continuous knock-in was monitored at the time-step width** by the
  local-vol, Heston and SLV solvers, which were gated off the FIRST_PASSAGE
  correction. The correction is barrier-local, so it never needed globally
  constant coefficients — only the dynamics at the barrier. Each solver now
  reports its own: `sigma_loc(barrier, t)` for local vol, one
  `(mu, sigma^2)` per variance column for Heston, and `L(barrier,t)^2 * v`
  for SLV. On a flat surface the local-vol solver had been bitwise identical
  to the *uncorrected* flat-BSM solver (+0.099% of PV). Applies to the
  snowball and phoenix families alike; the phoenix 2-D solvers had never
  built the state at all.
- **The Monte Carlo continuous-KI Brownian bridge used the implied vol at
  the strike** as the variance of every path and every step. The vol-model
  engines now supply the log-variance their own scheme accumulated. Against
  the closed-form first-passage probability the error falls from -10.0 / -8.5
  / -9.2 standard errors (Heston / local vol / SLV) to inside 4.
- The first-passage correction estimated its slope from a node that could sit
  a fraction of a cell above the barrier, where the knock-in mask's own step
  — not the slope — dominates. Sampling is now floored at half a cell.
- `PhoenixOption` never built its knock-in observation schedule, so a
  discretely monitored phoenix raised instead of pricing.
- `disable_ko_after_ki` was ignored by the snowball and phoenix PDE solvers,
  which wrote the knock-out payoff to the knocked-in surface as well.
- Phoenix `coupon_pay_type=EXPIRY` coupons roll up and are paid at
  termination — the knock-out settlement when it knocks out, maturity
  otherwise — and are never forfeited. All three engine families agree;
  PDE and QUAD gained a termination-value surface to price it.

#### Changed
- Continuous knock-in prices move in the local-vol, Heston and SLV engines
  (PDE and Monte Carlo). Discretely monitored, European and no-knock-in
  products are bitwise unchanged, verified across 54 engine/product/
  monitoring combinations; the sole exception is the opt-in
  `KnockInMonitoringMode.BGK_APPROXIMATION`, which replaces a discrete
  schedule with continuous monitoring by design.

### Backtest replay consolidation

(spec: `docs/superpowers/specs/2026-07-30-backtest-replay-consolidation-design.md`).

#### Added
- `quantark.backtest.replay`: canonical home of the product-replay backtest —
  ONE multi-product daily loop (`ReplayBacktestEngine`/`ReplayBacktestConfig`/
  `ReplayProduct`/`ReplayBacktestResults`); the single-product
  `AutocallableBacktestEngine` is now a book-of-one wrapper with byte-identical
  output (golden-gated). Book runs gain per-day vol-model calibration.
- `quantark.backtest.futures_ledger`: shared `FuturesHedgePosition` +
  `FuturesRollPolicy`.
- `quantark.backtest.metrics.CorePerformanceMetrics`: cross-asset metrics over
  the `BaseBacktestResults` protocol; replay results expose `.metrics` and the
  full protocol; equity `PerformanceMetrics` now extends the core.
- `quantark.backtest.replay.schema`: typed row schemas — the single source of
  truth for every record column.
- Pending-settlement KO termination: `terminate_on_lifecycle_end=True`
  (default) ends a replay when terminal cash lands (observation vs settlement
  are separate moments; discounted receivable carried in portfolio value);
  summaries gain `termination_reason` / `days_replayed` / `days_in_contract`.
- `AutocallableEngineConfig.event_stats_fallback` (`"none"` default): the
  silent 5000-path MC event-stats fallback is now opt-in, logged, and recorded
  per row in the new `event_stats_engine` column.
- `BumpConfig.gamma_spot_bump`: engine-side asymmetric gamma bumping;
  replay-side manual bumping removed (greeks are always
  `engine.calculate_greeks`, failures raise — no silent zero-delta days).
- `AutocallableDeltaHedgeStrategy` joined the `BaseStrategy` hierarchy
  (`quantark.backtest.strategy`).

#### Changed
- Relocations (old paths remain as `DeprecationWarning` shims until 0.5.0):
  `quantark.backtest.otc.*` -> `quantark.backtest.replay.*`;
  `otc.vol_calibrators` -> `quantark.volmodels.calibration`;
  `otc.vol_history` -> `quantark.param.vol.surface_history`;
  `_atomic_write_json` -> `quantark.util.io.atomic_write_json`.
- Replay books with `vol_model != "bsm"` reject non-Snowball products at
  config construction (the vol-model engine factory is snowball-only).
- **Heston/Heston-SLV Snowball and Phoenix 2D PDE solvers default
  `v0_boundary="degenerate_pde"`** (was the ADI core's `"neumann"`), and accept
  it as a constructor argument. Smile-calibrated Heston sits near the Feller
  boundary, where the Neumann treatment of the `v = 0` edge mis-prices by
  −0.540% of notional against a QE-M MC reference while the degenerate
  treatment holds +0.156%. `"neumann"` remains selectable for cross-checks.
  **Prices change** for these four solvers at default settings.
- **Heston/Heston-SLV Snowball concentrated PDE grids default to a power-graded
  variance axis** (`v_grid_power=2.5`) to resolve the near-zero layer around the
  Feller boundary. `v_grid_power=0.0` retains the legacy concentrated mesh for
  explicit cross-checks; uniform grids remain ungraded. **Snowball Heston/SLV
  prices and finite-difference Greeks change** at default settings.
- **The `mo_frozen` Heston preset enforces the Feller condition**
  (`enforce_feller=True`; the soft `regularize_feller` penalty is retained).
  Deliberate divergence from the `mo_volmodels` diagnostics scripts, which keep
  the soft-only policy. Feasibility is bought with smile accuracy — measured
  across 762 CFFEX MO settlement surfaces: median 8.4 bp, p90 29.3 bp, max
  217 bp of IV. On 6.6% of dates the constraint is met by driving vol-of-vol to
  its lower bound, which degenerates Heston to deterministic variance; consumers
  should screen on the new `feller_ratio`. Heston and Heston-SLV calibration
  cache entries are invalidated automatically (the fingerprint embeds the
  preset); `localvol` entries are unaffected.

#### Fixed
- `BaseEngine.calculate_greeks` and Gate G2's deterministic delta helper bypassed
  `create_bump_context`, so Heston/SLV central spot bumps rebuilt their S grids even
  though the solvers already provided a frozen-layout context. Base, up, down, and
  separate gamma reprices now all use the resolved base-market context.
- `Heston2DAutocallableSessionAdapter` now preserves the Snowball solver's
  `v_grid_power`; an explicit grading or legacy opt-out no longer disappears when
  the execution session clones the engine. Gate metadata records the selected
  value and Stage 12 forwards it into the production engine options.
- `calibrate_heston(enforce_feller=True)` failed closed on any problem where the
  constraint was active at the optimum and `ftol` was loose. SLSQP satisfies
  constraints only to about its own accuracy tolerance, but the feasibility
  buffer was a constant `1e-8` — measured slack is ~1e-9 at `ftol=1e-8` but
  ~1.6e-7 at `ftol=1e-6`, so the strict post-check rejected every fit. The
  buffer now scales as `max(1e-8, 10 * ftol)`; the post-check is unchanged and
  still fails closed.
- `VolModelCalibrator` Heston records carry `feller_ratio` (`2*kappa*theta /
  sigma**2`) alongside `feller_margin`. The margin is a difference and so is not
  comparable across dates; the ratio is.
- `Heston2DAutocallableSessionAdapter._clone_engine` rebuilds the engine from an
  explicit kwargs list and dropped `v0_boundary`, so a session or prepared-adapter
  run silently priced with the default boundary while the direct call honoured the
  constructor. Any 2D autocallable constructor argument omitted from that list
  fails this way — silently, with no error.

## [0.4.0] - 2026-07-27

Clean rewrite of the PDE grid construction and event application layer
(spec: `docs/superpowers/specs/2026-07-27-pde-grid-redesign-design.md`).

### Added
- `quantark.asset.equity.engine.pde.grid`: declarative grid layer —
  `GridRequest` (frozen geometry), `GridConfig` + accuracy profiles
  (`fast`/`standard`/`high`), ONE time builder (event-aligned, extras-scaled
  cap, damping schedules), ONE spatial builder (multi-point sinh, pinned
  auto-bounds, no barrier snapping), four-stage `EventSchedule`
  (terminal/interior/continuous/valuation-readout, pure block transforms),
  engine-owned `GridBinder` (LRU, `bind_shared` for same-underlying books,
  `rebind_time` for calendar rolls) and `GridLayerMixin` for
  BaseEngine-derived adopters.
- `PDEParams(accuracy=..., grid=GridConfig(...))` tiered front door.
- Frozen-`Layout` bump contexts: spot/vol/rate/div bumps reuse the base
  layout by object identity; calendar rolls rebind time on the same spatial
  object.
- `HestonSLVADICore(x_nodes=...)`: 2D solvers take their S-axis from the
  shared spatial builder.
- Tier-2 anchor certification (`test/pde_grid/test_anchor_certification.py`):
  MC/QUAD/closed-form/smoothness anchors gate every accuracy profile.

### Removed (breaking)
- Legacy grid modules `time_grid.py`, `spatial_grid.py`,
  `event_projection.py` (projection math moved verbatim to `grid/events.py`).
- `PDEParams` knobs: `auto_grid`, `time_grid_type`, `grade_exponent`,
  `event_min_steps_per_interval`, `log_dx_target`,
  `include_spot_in_critical_points`, `frozen_critical_points`,
  `barrier_refine_log_width`, `barrier_refine_levels`,
  `barrier_domain_expand`, `adaptive_grid`, `event_steps_per_day`,
  `max_time_steps`, `max_grid_size` (the last three had no remaining
  consumers — `GridConfig.steps_per_day` / `max_steps` / `max_points` own
  those roles).
- The class-level shared grid cache (`clear_grid_cache` is now a no-op).

### Changed
- All PDE prices reprice slightly (re-certified against MC/QUAD/closed-form
  anchors, goldens re-frozen): the layer builds concentration-based grids
  without barrier snapping — cell-average event projection is placement-
  independent — and event-aligned time grids emit one exact dt per interval
  (operator caches are arithmetic-neutral: cache-on == cache-off bitwise).
- Grid-layer solvers REJECT non-default `grid_size`/`time_steps`/`s_min`/
  `s_max` with a `ValidationError` naming the `accuracy`/`GridConfig`
  replacement (previously the values would have been silently inert); the
  knobs remain live inputs for the standalone vol-model solvers only.
- `make_pde_params` profiles (`fast`/`accurate`/`barrier_sensitive`/
  `reverse_sensitive`) now emit `accuracy`/`grid=GridConfig(...)` instead of
  the legacy knobs (which also fixes a crash: the product-hint path injected
  deleted `barrier_refine_*` keys).
- Bump contexts (`create_bump_context`) clone via `deepcopy`, preserving
  full constructor state (Heston `model_params`, SLV leverage surfaces, ADI
  dimensions); `PDEEngine.create_bump_context` carries the frozen solver
  instance. Frozen layouts fail closed when reused across different hard
  (absorbing-barrier) bounds, and calendar-roll rebinds coverage-validate.
- 2D Heston/SLV snowball/phoenix bump contexts freeze the ADI S-axis at the
  solve configuration (`n_x`, `num_std=8`) and reuse it by identity under
  market bumps (`_layer_x_nodes` routes through `resolve_bound_layout`);
  previously the S-axis was rebuilt per bump, mixing grid movement into 2D
  numerical greeks.
- The scheme knobs are LIVE on the grid layer: `use_rannacher`/
  `rannacher_steps` drive terminal damping and `rannacher_at_events`/
  `event_rannacher_steps` drive event damping (the binder derives the
  damping schedule from them; explicit `GridConfig` fields still win). At
  default knob values the schedule is unchanged.
- Prepared/session PDE pricing resolves solve state
  (`_prepare_for_request`) before fingerprinting grid geometry — BGK
  products no longer trip `DeterminismViolation` against their own
  preparation when an artifact cache is active.

### Fixed
- Autocallable PDE grids ignore out-of-reach critical-price markers (such as
  disabled 100× KO observations) for bounds and concentration while retaining
  the observation date and any coupon event. Dense critical sets now use
  bounded cumulative inversion of the Tavella-Randall monitor, removing the
  overflow-prone shooting path and its non-monotonic construction-time cliff.
- Snowball/Phoenix QUAD event handling now defaults to cell-average projection
  and phase-stable trapezoidal integration. Unreachable KO markers are
  suppressed without deleting their dates, and the legacy nodal/Simpson
  behavior remains available explicitly.
- QUAD can optionally refine nested odd grids until consecutive PV estimates
  satisfy configured absolute/relative tolerances. Fixed
  `grid_points=1001` remains a speed-oriented estimate rather than a universal
  convergence guarantee.

### Deferred (recorded follow-ups)
- Event-aligned 2D ADI time axis (uniform march retained).
- Standalone vol engines (`LocalVolPDESolver`, `HestonPDESolver`,
  `HestonSLVPDESolver`, LV/Heston/SLV barrier engines, `HestonDCNPDESolver`)
  keep bespoke kernels and their `grid_size`/`time_steps`/`s_min`/`s_max`
  inputs; `event_projection` + Rannacher scheme knobs survive for the live
  characterization paths.
- Stencil-delta accuracy on concentrated grids (~2e-3 absolute) —
  provisional 5e-3 anchor band.

## [0.3.0] - 2026-07-20

First release of the `quantark.execution` framework kernel and the
`quantark.execution.greeks` scenario layer. The `test_snowball_quad_flat_identity_golden`
failure predates this program and remains quarantined (reproduces on
unmodified main). The spec §20 controlled-host performance gates in
`docs/execution/README.md` are documented as release-preparation evidence
and were not re-measured on the release host.

### Added
- **`quantark.execution.greeks` — greek bumps as scenario cells** (spec
  2026-07-20): `TradeState` per-trade base, same-type `greek-bump/v1`
  transformer with real spot/vol/rate/div/time mutation attribution,
  `greek-value/v1` float runner, and assemblers mirroring
  `EquityPosition.get_trade_risk` / `GreeksCalculator.
  calculate_numerical_greeks` operation-for-operation (bitwise unit gate).
- **`PricingSession.run_scenario_plans`** — many `(base, specs)` plans
  packed through ONE bounded-window process pool (per-cell worker-spec
  payloads; the portfolio × bumps shape), with a per-plan error boundary
  under `collect_errors=True` (base-resolution/planning failures become
  aligned typed `PricingFailure`s while other plans execute).
- **`quantark.execution` — composable execution kernel** (framework contract
  v1; spec `docs/superpowers/specs/2026-07-15-mc-pde-performance-generalization-design.md`).
  `PricingSession` wraps every exported MC/PDE engine without changing direct
  legacy behavior: adapter registry, resource budgets/leases, prepared-artifact
  and session draw caches, reproducibility manifests and immutable diagnostics.
  Capabilities: fixed-batch MC backends (serial/threads, bit-identical
  reductions in canonical order), adaptive RQMC session mode (bitwise vs
  direct across the 12 autocallable vol engines), PDE preparation artifacts
  (grids, step coefficients, factorization packs; one-solve PV/event/grid
  outputs across 18 prepared engines + FX LV), typed scenario execution with
  verified mutation footprints, spawn-safe `WorkerSpec` process workers, a
  Dask backend over the same plans, `price_many` grouping, and a
  complete-payload scenario validator. Published artifacts: generated
  [capability matrix](docs/execution/capability-matrix.md) (CI-fresh), policy
  guide, legacy-internals rationale, reproducibility JSON Schemas (validated
  against live payloads), and runnable migration examples. Adoption is
  inventory-driven and honest: `temporary_legacy` rows remain, and no
  universal framework support is claimed until the controlled-host gates pass.
- **DCN (Digital Coupon Note)**: product + observation-schedule generator,
  curve-aware MC engine with leg decomposition and event stats, two-surface
  PDE engine with MC↔PDE cross-validation gates, LocalVol/Heston/QE/SLV DCN
  MC engines sharing one payoff kernel, MLMC-style coupled timestep-ladder
  Heston pair, SVI/cleaned-quote calibration layers with no-arbitrage and
  repricing-residual reports, node-bump vega buckets and carry-invariant rate
  bumps.
- **Implied futures carry**: `IndexFuturesCurve` implied q(T), futures-tenor
  delta/rhoq buckets, delta-one futures rhoq, portfolio aggregation and hedge
  integration.
- **Engine term-structure upgrade**: MC, PDE, and quadrature autocallable
  engines consume full r/q/vol term structures with a cross-family term
  agreement gate.
- **Vol-model barrier exotics**: LV/Heston/SLV barrier MC pricers on a shared
  monitoring core, LV 1D barrier PDE, Heston/SLV 2D barrier ADI solvers.
- **Vol-model numerics program (phases 2–5)**: vectorized Lewis Heston
  calibration (~21x), unified Heston/SLV ADI core, opt-in Krylov and TR-BDF2
  Fokker–Planck marches, opt-in QE-M martingale correction, concentrated
  (x,v) grids, degenerate v=0 Feller boundary, LV Rannacher start-up
  (default on), opt-in QMC sampling for Heston/SLV/LV kernels.
- **PDE event-stats API** on Snowball/Phoenix vol solvers with event
  distributions surfaced through session outputs.

### Changed
- DCN MC engine performance: draw cache, batched threads, LV build hoist.
- The legacy autocallable Dask batch loop (Snowball vanilla/KO-reset,
  Phoenix) is consolidated into one shared reducer
  (`autocallable_dask_batch`) with byte-preserved behavior, gated by frozen
  bitwise goldens. The legacy `use_dask` route itself is unchanged and
  preserved (see `docs/execution/internals-and-legacy.md` for the §17.3
  removal preconditions).
- Dev extras now include `dask[distributed]` and `jsonschema`.

### Fixed
- `PhoenixMCEngine` silently disabled its Dask parallel path on modern dask
  (`from dask.compute import compute` no longer resolves); availability now
  matches `SnowballMCEngine`, with a regression test.
- Processes scenario backend: a parent whose `__main__` cannot be
  re-imported by spawn children (stdin/heredoc scripts record
  `__file__ = '<stdin>'`; deleted script files) now fails closed with a
  typed `CapabilityError` before any worker is spawned, instead of every
  child dying at bootstrap and surfacing as an opaque
  `BrokenProcessPool` with zero completed cells.
- Craig–Sneyd ADI corrector order restored (base `Y0` + implicit `-rU`),
  QE sampler sign bug, and the 2026-07 PDE/QUAD audit fixes (13 findings)
  are included via the programs above.

## [0.2.5] - 2026-07-03

### Fixed
- 2026-07 PDE/QUAD audit: 13 numerical findings fixed (discrete-monitoring
  boundary handling, Rannacher terminal off-by-one, BGK state resolution,
  smoothed KI for Phoenix/KO-reset, shared theta damping schedule) with
  solver-family consolidation and regression tests.

## [0.2.4] - 2026-07-02

### Added
- Opt-in BGK continuous-KI mode for Snowball and Phoenix PDE solvers
  (direction-aware shifted barrier, per-product flag).
- `EquityPosition.get_trade_risk` (product + total, frozen one-loop) and
  `AutocallableCashLeg.time_shift`; event streams selected by leg
  requirements.

### Fixed
- Portfolio vega/rho/dividend-rho aligned with the canonical greek
  convention; bump contexts freeze PDE critical points for grid steadiness.

## [0.2.3] - 2026-07-01

### Added
- Cross-engine-consistent `ki_ever` / `ki_survive` event-stat fields.

### Fixed
- Theta advances by business day; quadrature event-stats recursion uses
  smoothed barrier indicators.

## [0.2.2] - 2026-07-01

### Fixed
- PDE numerical Greeks: freeze the base spatial domain for finite-difference
  bump repricing so rho and dividend rho measure market sensitivity without
  contamination from auto-grid/domain movement under bumped rate, dividend,
  volatility, or time inputs.

## [0.2.1] - 2026-06-30

### Fixed
- `PhoenixPDESolver`: apply a KO observation scheduled exactly at maturity.
  The inherited grid builder stores the maturity KO in `_ko_terminal_record`
  (intentionally kept out of `_ko_observation_indices`), but the Phoenix
  `_solve` override looked the terminal KO up in `_ko_observation_indices`
  and therefore dropped it — mispricing products with a terminal KO date by
  several percent versus the quadrature and Monte Carlo engines. Terminal KO
  is now applied after the terminal coupon/KI jumps (matching
  `SnowballPDESolver._solve`), routed through `_apply_ko_jump_vector` so the
  same-date coupon-at-KO payoff is preserved.

## [0.1.2] - 2026-06-13

### Added
- Credit dual-measure framework: a recovery convention layer
  (`quantark.asset.credit.conventions`, `STANDARD_RECOVERY=0.40`) that
  separates the canonical shared-curve **hazard01** factor from the
  recovery-converted **CS01** used by products and SIMM. Curve shocks stay
  in hazard space; spread stresses convert through the recovery convention.
- Single-name CDS **roll-down / as-of pricing** via effective and maturity
  dates (seasoned and forward-start), with `schedule_asof` and a
  total-return coupon cash ledger threaded through the dynamic-scenario and
  backtest engines. SIMM buckets the remaining tenor. (Basket as-of is
  deferred.)

## [0.1.1] - 2026-06-11

### Added
- `SnowballQuadEngine`: explicit `ki_monitoring_mode` on `QuadParams`
  (`KnockInMonitoringMode`). `EXACT_DISCRETE` (default) prices every KI
  observation date exactly with adaptive spatial-grid refinement.
  `BGK_APPROXIMATION` is an opt-in performance mode that replaces a dense
  discrete KI schedule with continuous monitoring at a
  Broadie-Glasserman-Kou shifted barrier; the engine validates approximately
  regular spacing (median-band dispersion test), a constant resolved
  barrier, full-horizon coverage, stable volatility, and a minimum schedule
  density (`bgk_min_ki_observations`), raising `ValidationError` otherwise.
  Converted pricing matches the equivalent shifted-continuous product
  exactly (grid-aligned to the shifted barrier) while the valuation-time KI
  state keeps contractual discrete semantics. A first-order residual bias
  remains (a few bp of PV at daily spacing, growing with observation
  spacing and drift).

### Fixed
- `SnowballQuadEngine`: dense discrete KI schedules now retain their explicit
  observation dates instead of being delegated to continuous monitoring. The
  engine adaptively refines its internal spatial grid to resolve short
  diffusion intervals (accuracy-oriented default of 2.5 cells per interval
  diffusion stddev; lower to 1.25 or opt into BGK for speed), avoiding the
  material fair-KO-rate bias caused by the previous continuous-monitoring
  approximation.

## [0.1.0] - 2026-06-11

### Added
- First public release.
- Equity derivatives: European/American/Asian vanilla options, barrier,
  one-touch, digital, sharkfin, and autocallable products (snowball,
  phoenix, KO-reset snowball, range accrual) with analytical, Monte
  Carlo, PDE, quadrature, and tree engines.
- Fixed income: fixed bonds, FRNs, bond options, bond forwards/futures,
  convertible bonds, interest rate swaps.
- Market data layer (`quantark.param`, `quantark.priceenv`), Greeks
  calculators, portfolio VaR (parametric/historical/Monte Carlo),
  ISDA SIMM v2.6, stress testing, multi-day scenario simulation, and a
  hedging backtest framework.
- Legacy flat-import compatibility shim (`asset`, `util`, …) with
  `DeprecationWarning`; slated for removal in 1.0.
