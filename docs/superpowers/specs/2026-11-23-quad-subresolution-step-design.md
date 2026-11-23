# QUAD sub-resolution steps: price short gaps locally instead of refining everything

Status: DRAFT brief, owner direction 2026-11-23 ("treat a sub-resolution step locally ... we should do this").
Destination: docs/superpowers/specs/2026-11-23-quad-subresolution-step-design.md (parked in the contract-clock run
dir because that session was worktree-isolated). Follows the contract-clock run, which makes V1 REFUSE
under-resolved steps. Nothing is built.

## 1. Problem

Both deterministic QUAD engines price a diffusion step by convolving with the Gaussian transition kernel sampled on
a spatial lattice. When a step is short relative to the lattice cell (``s = sigma * sqrt(dt)`` against cell ``h``),
the sampled kernel aliases: its discrete mass is wrong by ``2 exp(-2 pi^2 (s/h)^2)`` (measured on V1's operator:
s/h 1.047 -> 8.0e-10, 0.722 -> 6.7e-5, 0.574 -> 3.0e-3, 0.331 -> 0.23).

- **QUAD V1** sizes its lattice from the knock-in spacing only (`_resolve_grid_points` is fed `ki_records`' times
  at all five call sites), so a short step created by a KO next to a KI, a settlement node or a continuous-KI knot is
  never resolved. Before the contract clock it silently MISPRICED them (KO 0.5 / KI 0.5+gap, S=100, vol 20%, true
  98.92: gap 53 min -> 109.33, 5 min -> 226.05, 35 s -> 573.39). The contract-clock run makes it refuse instead.
- **QUAD V2** refines GLOBALLY: the cell is set by the shortest step's variance, so every step runs at that
  resolution. It is exact but nodes grow like sqrt(T / dt_min): 5 min ~112k (1.2 s), 35 s ~339k (7.2 s), 3 s ~1.1M
  (refused: kernel workspace > ~390 MB). Default ``max_nodes=100_000`` refuses below ~53 min.
- **PDE** and **MC** price every gap correctly (PDE within 5e-4 of the coincident value at gaps 0.3 s .. 88 h; MC
  within 3 sigma).

Probes (copied next to this brief): `engines_gap_probe.py`, `v1_mass_probe.py`, `v2_nodes_probe.py`.

## 2. Goal

A step whose kernel the lattice cannot resolve is applied by an operator that is exact in the small-step limit, on
the EXISTING lattice, at the cost of an ordinary step. Resolved steps are untouched (bitwise). V1 and V2 then price
event gaps down to the contract clock's resolution (``Tolerance.ZERO``, ~3 ms) at normal cost.

In scope (owner 2026-11-23): the FIRST step from the valuation instant as well -- a valuation minutes or seconds
before an observation (the backtest's 14:55 proxy-close clock) must be priced exactly at V1's accuracy target, not
refused (measured on base: 5 min ahead 265.92 vs PDE 98.30; 60 min 99.44 vs 98.30). Review 2026-09-21 R1 still
owns the instant itself: an exact 0.0 is decided on the known spot, and a positive time below Tolerance.ZERO stays
refused by `_require_resolvable_time`.

Landing (owner 2026-11-23): the contract-clock branch is HELD and lands together with this fix.

Non-goals: changing how resolved steps are priced; FX.

## 3. Candidates (to be demoed standalone first, owner rule "standalone demo first")

A. **Exact Fourier multiplier.** Apply ``exp(i mu k dt - 0.5 sigma^2 k^2 dt)`` (times the discount) to the FFT of
   the Simpson-weighted values instead of the FFT of the sampled kernel. No aliasing for any ``dt``; tends to the
   identity on the grid. Open: domain boundary/padding (V1 already pads for FFT), Gibbs at the event discontinuity
   (the event mask is applied in physical space either side of the step), interaction with V1's tail correction
   and V2's order-8 basis.
B. **Cell-integrated kernel.** Weights ``Phi((z + h/2)/s) - Phi((z - h/2)/s)``: exact mass for any ``s``, O(h^2)
   smoothing. Simple; may lose V1's spectral accuracy on the step.
C. **Small-time expansion.** ``I + tau * L`` (+ ``tau^2 L^2 / 2``) with the generator on the grid: exact as
   ``dt -> 0``; needs a stability/accuracy floor on ``s/h``.
D. **Ordered event fusion.** For ``s/h`` below a floor, apply the two events' masks in time order on the same grid
   nodes with no diffusion between (the limit of A-C); error O(s/h). Cheapest; must be shown to keep KO-wins-tie
   and the D35 / KO-reset rules exact.

Switch rule (all candidates): a step uses the local operator only when its aliasing bound exceeds a tolerance
(V1's contract-clock guard: ``Tolerance.PRECISION``); every other step keeps today's kernel, so contracts priced
today are bitwise unchanged.

## 4. Demo plan and decision matrix

Runtime-patched scripts (no engine edits) on: the KO/KI gap ladder (0.3 s .. 88 h), intraday near-close contracts,
a KO-reset with schedule 1/2 a few minutes apart, a continuous-KI contract whose knots fall near observations, and
365/y KI with monthly KO. Reference: PDE (fine grid) and MC (RQMC, 3 sigma). Columns: max |price error|, delta
error, cost vs a resolved step, bitwise-unchanged share of the regression book, implementation risk on V1 and V2.

## 5. Acceptance (draft)

- V1 and V2 price the gap ladder within PDE's own refinement difference at default parameters, at most 1.2x the
  cost of the coincident contract.
- Every contract that prices today is bitwise unchanged (goldens, replay goldens, certificate anchors probed).
- V2's exact-semantics contract: the opt-in ``event_time_tolerance`` keeps its meaning; certificates that move are
  amended through the release procedure with the owner's signature.
- V1's refusal guard stays as the backstop for anything the local operator cannot price.

## 6. Open questions for the owner (before the run)

- Does V2 adopt the same local operator, or keep global refinement and only raise its default node budget?
- Is O(s/h) (candidate D) acceptable for sub-second gaps, or must every candidate be exact to ``Tolerance.PRECISION``?


---

# As built: run specification (decisions, owner rulings, acceptance criteria)

# QUAD sub-resolution steps (V1 + V2)

Brief with measurements: the first half of this document.
Stacked on the contract-clock change (`2026-11-23-contract-clock-design.md`), which was held and
lands together with this one.

## Goal

- **Outcome:** QUAD V1 and QUAD V2 price a diffusion step their lattice cannot resolve with a local small-step
  operator on the EXISTING lattice, instead of refusing it (V1, contract clock) or refining the whole grid (V2).
  Event-to-event gaps and the FIRST step from the valuation instant (the backtest's 14:55 proxy-close clock) down to
  Tolerance.ZERO (~3 ms) price correctly. V1's refusal guard stays only as a backstop.
- **Scope:** V1 (snowball, Phoenix, KO-reset; price and event stats; continuous-KI knots and settlement nodes) and V2
  (the same local operator on its mesh). Non-goals: changing how resolved steps are priced; FX; the exact-0.0
  valuation instant (review 2026-09-21 R1) and positive times below Tolerance.ZERO (still refused).
- **Acceptance:** every gap on the ladder 3 ms .. 88 h (two-event and first-step) prices within V1's
  convergence_rel_tol (1e-4, relative) of a fine PDE / MC reference on V1 and V2 at default params, deltas included;
  contracts with no unresolved step are bitwise and speed-unchanged; full suite green.
- **Constraints:** standalone runtime-patched demos + decision matrix FIRST (owner rule); exact semantics, no
  fallbacks; quantark.util.numerical tolerances; resolved steps bitwise; certificate anchors unchanged (or amended
  with owner signature, D4 b).
- **Cost:** a short-gap contract costs at most 1.2x the coincident booking on V1 and V2, measured interleaved in one
  window.
- **Decision gate:** the controller picks the candidate from the decision matrix against these criteria and proceeds
  (ruling); the matrix goes in the report.
- **Landing:** together with the contract-clock branch: ff local main, CHANGELOG 0.12.0 entry extended, no tag, no
  push.

## Scope

- In: QUAD V1 diffusion steps (`_diffuse_fft` / `_diffuse_with_bridge` / forward kernel for statistics) whose
  kernel the lattice does not resolve (`_require_resolved_kernels` would refuse them today), on every V1 route
  (snowball, Phoenix, KO-reset; price and event statistics; continuous-KI knots; settlement nodes; first step from
  the valuation instant). QUAD V2 kernel construction (`v2/operator.py` `Operator.kernel`, and the readout/adjoint
  paths that integrate a step) for intervals whose Gaussian its cell quadrature does not resolve, including the
  first interval; V2's `make_mesh` stops sizing the mesh from the shortest interval once that interval is priced
  locally (the node budget then follows the resolved steps only).
- Out: resolved steps (bitwise unchanged); FX; the exact-0.0 valuation instant and positive times below
  `Tolerance.ZERO` (still refused by `_require_resolvable_time`); V2's `event_time_tolerance` semantics.

## Decisions

- Owner (grill 2026-11-23): V2 gets the same local treatment as V1; accuracy = within the engine accuracy target
  (V1 `convergence_rel_tol` 1e-4 relative) incl. deltas; cost <= 1.2x; demos first; controller picks; first step in
  scope; land together with the held contract-clock branch.
- (controller, doc review P1/P2) BITWISE PRIORITY switch rule: the local treatment engages ONLY where the engine
  refuses today, so every contract that prices today is bit-identical:
  - V1: (i) `_resolve_grid_points` would raise (required > `max_adaptive_grid_points`): the steps that drive the
    requirement (the shortest ones, first step included) are excluded from the sizing until it fits the cap, and
    are priced locally; (ii) `_require_resolved_kernels` would refuse a step on the resulting lattice: that step is
    priced locally. A contract the sizing resolves today by growing the grid (e.g. 60 min first step -> 4571 points)
    keeps that grid (bitwise; its cost is today's and is exempt from AC7).
  - V2: an interval is excluded from `make_mesh` sizing only when including it would exceed `max_nodes` (today's
    refusal); excluded intervals are priced locally. The mesh-free form of the rule: exclude the smallest variances
    in turn until the required node count fits.
  - `auto_converge` (V1): the ladder keeps skipping unresolved rungs (contract-clock behaviour); the local treatment
    is used only if the cap rung is still unresolved.
  Continuity across each threshold is MEASURED (price, bump delta, theta just below/above it) and must lie within the
  accuracy target; it is reported, not assumed.
- (controller) The V1 refusal guard stays as the backstop: it refuses only what the local operator cannot price
  within the accuracy target (expected: nothing above `Tolerance.ZERO`).
- (controller, doc review P1) The FIRST step (valuation -> first event) needs a semi-analytic candidate: (F) apply
  the first event analytically at spot on the PRE-event continuation surfaces (smoothly interpolated), with exact
  Gaussian limits at the barrier -- the information a lattice-only operator lacks when spot is within a cell of a
  barrier. V2's `readout` already does this with piece limits (doc review P2: measure V2's first step on base before
  touching its readout).
- (controller) Candidate set for the demos. V1: (A) exact Fourier multiplier on the existing FFT grid; (B)
  cell-integrated kernel weights (Gaussian CDF differences); (B') exact Gaussian integration of the piecewise-linear
  (or cubic) interpolant of the grid values; (C) small-time expansion I + tau L (+ tau^2 L^2 / 2); (D) ordered event
  fusion (no diffusion between two events below a floor). V2: (E) exact truncated-Gaussian-moment integration of
  each cell's polynomial (generalising `affine_moments` to the basis degree); (A) on V2's mesh. The decision matrix
  ranks them on: max relative price error and delta error vs reference over the gap ladder (two-event and
  first-step), cost vs the coincident booking, bitwise-unchanged share of resolved contracts, implementation risk.
- (controller, doc review P1/P2) Error definition: the INCREMENT the local treatment adds -- the engine at default
  params against the SAME engine with the step resolved (V1 `grid_points` raised until s/h >= 1.05, feasible down to
  ~1 s; V2 `max_nodes` raised, down to ~35 s) and, below that, against a semi-analytic first-step reference (Gaussian
  x exact event x resolved continuation). Pass rule: |err| <= tol + reference half-width, with an absolute floor
  1e-8 * notional. PDE / MC are reported for context, not gated (their half-widths exceed 1e-4 at these gaps).
- (controller, doc review P2) Deltas: the central GreeksCalculator bump, the same bump on engine and reference,
  tolerance 1e-3 relative with an absolute floor 1e-6 * notional; points with |log(S/B)| < 3 * max(s, bump) are
  REPORTED, not gated (the pointwise delta diverges there).
- (controller, doc review P2) Rungs are year fractions on act/365 strictly above `Tolerance.ZERO`: 2e-10 (~6.3 ms),
  1 s, 35 s, 5 min, 53 min, 8.8 h, 88 h. AC3 states spot-to-barrier distances: on the barrier, 0.5 cell, 2 cells,
  1% and 5% away.
- (controller, doc review P2) AC7 baseline: two-event contracts against the coincident booking; first-step
  contracts against the same contract valued one resolved step earlier (one trading day before the observation).
- (controller, doc review P3) The local operator computes each stacked row independently (row-wise transforms, no
  cross-row BLAS), so `pv == price()` stays bitwise.

## Scope additions (doc review P2)

- V1 operators that sample the same aliased kernel and must get the same treatment: `v1_adjoint.py` (hand
  transposes of `_diffuse_fft` and `_readout_at_spot`, feeding the KI stream of `v1_statistics`),
  `_readout_at_spot` / `_tail_at_spot` (`readout='transition'`), `_forward_seed` / `_diffuse_density`
  (`event_stats_mode='forward_density'`).

## Decisions after S1 (controller)

- Two-event and every non-first local step: C1, the moment-matched 3-point stencil
  w = (s^2/(2h^2) - m/(2h), 1 - s^2/h^2, s^2/(2h^2) + m/(2h)) * exp(-r dt), row-independent; transposes for the
  adjoint/forward paths. First step (valuation -> first event) when it is unresolved: F, the first event applied
  exactly at spot on the pre-event surfaces (exact Gaussian limits at the barrier; closed forms or interior
  breakpoints, never plain Gauss-Legendre across the barrier), C1 on any other local step.
- Chain backstop (AC5d): per-step C1 error ~ (s^2 h^2 - 3 s^4) f''''/24, so the accumulated estimate is proportional
  to sum over local steps of s_i^2; refuse when it exceeds a threshold calibrated in S3 on the S1 chain probes so that
  the measured chain error is <= convergence_rel_tol / 2.
- `auto_converge`: non-cap rungs keep raising `UnderResolvedStepError` (skipped); the local treatment is used on the
  cap rung only.
- Delta tests gate against the increment reference (`fixture.json` `ref_delta`), not the refined-grid delta
  (the default grid's own bump-delta bias is up to 1.7e-3 at vol 5%, identical on both arms).

## Owner decision 2026-11-23 ("fix it") -- first steps V1 prices today

- F also replaces the first step on contracts V1 prices TODAY when today's operator misses the accuracy target there
  (S1: a 1-day lead near a barrier misses 1e-4 in 11/30 configurations, worst 1.1e-3; just above the guard threshold
  up to 1.1% at vol 60%, where the PDE agrees with F). Bitwise priority is lifted for those first steps only.
- (controller) F switch for priced contracts: the first step uses F when its s/h is below a threshold S_F MEASURED in
  S3b such that today's operator, at every S1 vol and spot-to-barrier distance, is within convergence_rel_tol / 2 of
  the F/semi-analytic reference for every first step with s/h >= S_F. Longer first steps keep today's operator
  bit for bit. S_F and the measurement table are recorded in the run state.
- AC3b: for first steps with s/h < S_F that V1 priced before (1-day / 1-week leads at the S1 distances), V1 is within
  the AC3 tolerance of the reference; the moved values (goldens, replay goldens, certificate anchors) are measured
  and listed; an anchor that moves goes to S7 (amendment prep, then stop for signature).
- AC5c then gates both sides of the first-step threshold (F on both sides of the guard threshold).

## Contracts

- No new public API is required. Internal: a V1 small-step operator `_diffuse_local(...)` with `_diffuse_fft`'s
  signature, chosen per step by the switch rule; a V2 kernel path selected in `Operator.kernel` by the switch rule.
  Names are the builder's.
- `_require_resolved_kernels` keeps its signature; it refuses only steps the local operator cannot price.
- V2: `make_mesh` sizes from intervals the mesh resolves; intervals priced locally do not raise the node count.

## Acceptance criteria

- AC1 (demos): runtime-patched demo scripts for every candidate, a decision matrix with the columns above, and a
  ruling naming the pick, all under `tmp/demos/` of this run; no engine edit before the ruling.
- AC2 (V1 ladder): KO 0.5 / discrete KI 0.5 + gap (S=100, vol 20%, r 3%, q 1%; plus vol 5% and 60%) for the gap
  rungs of §Decisions: V1 at default params differs from the resolved-reference (§Decisions error definition) by at
  most 1e-4 relative + reference half-width (floor 1e-8 notional), and its central bump delta by at most 1e-3
  relative (floor 1e-6 notional; near-barrier points reported, not gated).
- AC3 (V1 first step): valuation {2e-10 y, 1 s, 35 s, 5 min, 60 min} before the next KO observation and before a KI
  observation, at the spot-to-barrier distances of §Decisions: same tolerances and reference rule as AC2.
- AC4 (V2): AC2 and AC3 on V2 at default params, without raising `max_nodes`.
- AC5 (routes): Phoenix and KO-reset (two schedules a short gap apart), event statistics `pv == price()` bitwise and
  probabilities within 1e-4 of the reference, continuous-KI and settlement-node routes, the forward adjoint pass
  (`v1_adjoint`), `readout='transition'` and `event_stats_mode='forward_density'` covered.
- AC5c (continuity, amended after S1): price, bump delta and theta measured just below/above each switch threshold
  (V1 sizing cap, V1 kernel guard, V2 `max_nodes`). GATED: the locally priced side is within the AC2/AC3 tolerance of
  the reference. REPORTED: the jump itself, which on today's side carries today's V1 error (out of scope under
  bitwise priority; S1: up to 8.7e-4 rel on KO-first at vol 60%).
- AC5d (chain backstop, added after S1): local steps are priced only while their accumulated error estimate stays
  within `convergence_rel_tol`; a contract whose local steps chain beyond it (S1: T=5, vol 5%, daily KI -> 3760
  local steps, C1 -5.9e-4) is refused with `UnderResolvedStepError`, never mispriced. Test: one isolated-step contract
  prices, one chain contract refuses.
- AC6 (bitwise): every contract whose steps are all resolved prices bitwise as before on V1 and V2 (goldens, replay
  goldens, the six certificate anchors exact-equal; the T9 contract-clock acceptance suite green).
- AC7 (cost): each AC2/AC3 contract the local treatment prices costs <= 1.2x its baseline (§Decisions) on V1 and
  V2 (interleaved, idle machine); contracts the engine already resolves by growing its grid are exempt (bitwise).
- AC8: full suite green; CHANGELOG 0.12.0 entry extended (V1 no longer refuses short gaps / first steps; V2 prices
  short gaps at default params).

## Constraints

- Contract-clock spec constraints all apply (quantark.util.numerical; exact semantics; no fallbacks; dated ==
  numeric twin; moved values measured and signed; full suite green; never MC inside PDE; anchors never edited).
- Demos never edit engine files (runtime patching only), owner rule "standalone demo first".
- Timing: interleaved arms in one window on an idle machine (owner feedback "timing needs one window").

## Risks

- R1: Gibbs/ringing of a spectral operator at the event discontinuity right after a step (candidate A).
- R2: the bridge (continuous-KI) path and the forward statistics kernel need the same local operator, or the
  `pv == price()` and adjoint identities break.
- R3: V2's piecewise-polynomial jumps between cells: a local operator that ignores intra-step cell straddling is
  O(s/h) wrong; exact truncated moments must handle straddling cells.
- R4: certificate anchors may move if a certified contract has an unresolved step that previously priced (the V1
  guard would have refused it, so none expected; V2 intraday cert: verify).
