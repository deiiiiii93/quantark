# Investigation Report: PDE Auto-Grid Bias and Redesign

- **Date:** 2026-07-23
- **Status:** Investigation complete; root cause verified empirically (forced-ownership sign reversal, projection collapse, damping-only inertness). Phase-1 remediation SHIPPED 2026-07-23: `quantark/asset/equity/engine/pde/event_projection.py` dual-cell projection wired into the 1D BSM/LV autocallable solvers (price and event-stat sweeps) and slice-wise into the 2D Heston/SLV snowball/phoenix solvers; event-local damping decoupled from `auto_grid` and strengthened to two implicit solves per event. `PDEParams(event_projection="cell_average", event_rannacher_steps=2)` is the DEFAULT after repricing review (goldens moved toward QUAD/MC references, e.g. snowball event-stat PV −3.4e-4 relative with the PDE−QUAD gap halved); `"nodal"` + `event_rannacher_steps=1` remains the explicit legacy opt-out used by historical characterization oracles. Gates in `test/test_pde_event_projection.py`. Continuously monitored (and BGK) barriers keep nodal treatment by design. **External review 2026-07-23 hardening:** the straddling cell now averages the COMPLETE two-branch event function (envelope-preserving; the jump-only form could leave the branch envelope with a steep survive branch), observations falling on the valuation date apply the product's exact inclusive trigger pointwise (never cell-averaged; tolerance-inclusive at the barrier node since grid nodes are `exp(log(.))` round-trips), and the 2D ADI loop consumes `event_rannacher_steps` (per-event fully-implicit Douglas restarts; previously inert in 2D). **Scope honesty:** the original incident's instrument and Monte Carlo inputs were never stored (§2 note), so Phase 1 closes the verified grid-phase *mechanism* — reproduced and collapsed on the protected 24-obs Phoenix fixture — not the specific historical +8.5%/-2% incident numbers; capturing original cases as fixtures remains Phase 0 of any further redesign. Known open 2D limitation: event dates are rounded onto the uniform ADI step (`_integer_tau_key`), a placement error the projection does not address. **2D attribution study 2026-07-24** (external-reference evidence for the 2D solvers): on the Heston phoenix gate fixture the projected PDE Richardson limit is 1.9108 (internal steps −0.0094 → −0.0016 → −0.0003); the apparent −0.014 gap to the QE-M MC reference was the *MC's own* observation-stride bias — the Heston/SLV MC engines step only on observation dates (`MCParams.time_steps` is inert there), and sub-stepping the same engine (validated against the Lewis put identity on an always-knocked-in phoenix, all fine-stride runs within 1.3 se of analytic truth) moves the MC from 1.9238 ± 0.0017 to a fine-stride consensus of ~1.907 ± 0.003 (QE-M and log-Euler agree). PDE limit vs stride-converged MC: +0.004 ± 0.003 — statistically insignificant and inside the PDE's measured absolute-error envelope (phoenix-machinery put identity converges to Lewis, +0.004 at (256,80,192) with ATM-kink grid-phase noise of a few 1e-3 en route). The legacy nodal 2D solver, by contrast, plateaus 0.135 BELOW the stride-converged MC (~7% of premium) while accidentally looking accurate at coarse grids — the grid-phase lottery in 2D. Actionable follow-up outside this program: give the schedule-based vol MC engines optional sub-observation stepping (their quarterly QE strides bias sparse-schedule autocallables by O(1%% of premium)). Remaining literature-backed upgrades (Giles–Carter four half-step BE substepping, face-aligned meshes for Richardson, certified planner) stay open as §9 Phases 2+. **External review 2026-07-24 (round 2):** (1) *Valuation-date readout.* The exact t=0 nodal trigger still left a discontinuous t=0 surface that `price()` linearly interpolated ACROSS at off-node spots (uniform grids don't pin barriers onto nodes; a two-unit valuation-date coupon paid 0.09–1.8 units depending on N). The readout now interpolates the smooth 0+ branch column captured before the t=0 events land (`PDESolutionResult.readout_vec`) and applies today's deterministic transitions pointwise at the actual spot (`readout_override` for triggered phoenix coupons; KO/KI at valuation were already pointwise via the pre-PDE short-circuit and V1-surface selection); the event-stat sweep reads pre-event indicator columns at j==0 with the deterministic t=0 outcomes overlaid. En route a latent 2D bug surfaced: the ADI hooks compared `current_time = T − tau` against 0.0 with a RELATIVE `is_close`, and since `tau` accumulates FP step increments, `current_time` lands at ~1e-16 — the comparison could never fire, so 2D valuation-date events were silently cell-averaged; all sites now compare `tau` against `T`. (2) *2D damping semantics* now mirror `theta_by_step` exactly: `use_rannacher=False` is the master off-switch, damped restarts run at `params.event_theta` (was hard-coded 1.0), and maturity-node events are excluded (terminal Rannacher owns the payoff discontinuity; 1D `0 < idx < num_t − 1`). (3) *Coincident coupon+KO* project as ONE piecewise cell average (`project_piecewise_event`, K sorted thresholds / K+1 branches with exact sub-cell integration): sequential projection double-averaged any dual cell the two thresholds shared; well-separated barriers reproduce the sequential results. (4) *Default certification.* The reviewed bump-gamma instability (auto −1.19/−1.19/−0.62 at N=400/800/1600 vs uniform ≈−0.007) reproduces IDENTICALLY under legacy nodal projection — it is the deprecated `EngineParams.bump_size=1e-4` shim silently overriding `BumpConfig`'s documented 1% spot bump, so default BUMP-mode gamma second-differences the piecewise-LINEAR readout over 1bp (far below one cell) and measures the spot-node interpolation kink (auto grids pin spot onto a node; uniform grids leave it mid-cell where a line has zero curvature) — an estimator artifact orthogonal to event projection. Measured certification on the monthly synthetic snowball: engine grid-stencil greeks (the AUTO-mode public default for PDE engines) are N-stable to 1e-5 in gamma (−0.04335/−0.04336/−0.04337, dense-grid reference −0.04336), a 1% bump agrees to 5e-4, and PV is flat in N for `cell_average` (8e-4 over N=400→1600) while nodal drifts 0.055; pinned by `TestDefaultCertification`. Follow-up outside this branch: retire the silent `bump_size` shim (it makes `BumpConfig`'s documented 1% default unreachable for default-constructed params and invalidates BUMP-mode gamma for any C⁰ PDE readout). (5) Suite hygiene: NumPy≥1.24 floor restored (`np.trapezoid` shim) and a stray generated sample untracked.
- **Explainer:** `pde_event_projection_explainer.html` (same directory) — a self-contained visual walkthrough of why the nodal event masks failed and how the dual-cell projection fixes them, with an interactive barrier-through-the-cell demo. Open it directly in a browser.
- **Repository:** `quant-ark`
- **Branch / revision reviewed:** `main` / `b7f70b3`
- **Scope:** PDE spatial grids, time grids, discrete event treatment, damping, and per-position accuracy control for Snowball, Phoenix, and related barrier products

---

## 1. Problem statement

QuantArk's `PDEParams.auto_grid=True` is intended to choose a practical PDE grid automatically from product features. In the investigated autocallable cases it does not merely choose an inefficient grid: it changes the numerical representation of discrete contract events and introduces material, persistent pricing bias.

The observed behavior was:

| Configuration | Snowball result | Protected Phoenix result | Convergence behavior |
| --- | ---: | ---: | --- |
| `auto_grid=True` | Approximately 2% from the Monte Carlo reference | Approximately 2% from the Monte Carlo reference in the original comparison | Refinement did not remove the bias |
| Plain uniform grid (`auto_grid=False`) | Approximately 0.1%-0.2% from Monte Carlo | Approximately +8.5% from Monte Carlo | Phoenix oscillated materially across spatial grids |

The protected Snowball contains the same protected-loss-floor kink as the protected Phoenix and priced well on the uniform grid. The feature that distinguishes the unstable Phoenix case is the coupon-trigger state transition applied at each of 24 observation dates. That transition is a Heaviside value jump, not merely a continuous payoff kink.

The practical consequence is that neither current choice is safe:

- The feature-aware grid can produce a stable-looking but systematically biased value.
- The uniform grid can occasionally agree closely with a reference because of favorable grid phase, while another product or refinement level oscillates badly.
- Increasing `grid_size`, `time_steps`, or `event_steps_per_day` is not a reliable remedy when the discrete event itself is represented incorrectly.

### 1.1 Expected behavior

An automatic discretization facility should:

1. Preserve exact product semantics, including discrete monitoring dates and event inequalities.
2. Meet explicit tolerances for the requested quantities, such as PV, delta, gamma, or event probabilities.
3. Select the first certified plan from a deterministic ladder ordered by measured total cost.
4. Produce stable refinement behavior rather than depending on whether a trigger happens to fall on a node.
5. Reuse one frozen plan across pricing, event statistics, and risk bumps.
6. Report its actual grid and estimated errors, or fail explicitly when accuracy cannot be established.

The current boolean `auto_grid` does not provide such a contract. It bundles domain selection, spatial concentration, node placement, spatial size, time alignment, time density, and event damping into one heuristic switch.

### 1.2 Impact

The issue affects:

- Snowball, Phoenix, KO-reset, digital, and discretely monitored barrier values.
- Coupon-memory surfaces and other multi-state event transitions.
- Delta and gamma, which are more sensitive to ringing and grid phase than PV.
- Trade-level risk, where repeated solves amplify both grid noise and cost.
- Convergence tests, because two biased grids can agree without being close to the correct value.

Historical QuantArk 0.2.3 measurements also found event-aligned grids expanding to 4,164-7,280 time steps and one Phoenix trade requiring 14 price calls plus 7 event-stat calls. Those measurements are older than the current revision and were not rerun for this report, but they show why the planner's cost model must include state surfaces and requested scenarios, not only `N_x * N_t`.

---

## 2. Investigation scope and evidence

This report combines:

- The Snowball/Phoenix cross-engine and refinement observations summarized above.
- A node-equality perturbation diagnostic on a protected 24-observation coupon case.
- Inspection of the current PDE grid, event, and damping paths at revision `b7f70b3`.
- Primary literature on discontinuous payoffs, barrier placement, Rannacher damping, autocallable grid design, adaptive error control, and production mesher patterns.

The exact benchmark instrument inputs and Monte Carlo seed/path configuration used in the original comparison are not yet stored as a dedicated repository fixture. Capturing them is Phase 0 of the implementation plan; no redesign should proceed without that reproducible evidence bundle.

### 2.1 Equality-side diagnostic

The diagnostic changed only which side owns a trigger node. It did not change the PDE coefficients or product economics.

| Spatial size | Inclusive trigger | Exclusive trigger | Average of the two |
| ---: | ---: | ---: | ---: |
| 400 | -2.17% versus reference | +2.09% versus reference | -0.04% versus reference |
| 1,600 | -0.89% versus reference | +0.94% versus reference | +0.03% versus reference |

The near-symmetric sign reversal and cancellation are strong evidence of a one-sided event-projection error. The trigger is effectively displaced by roughly half a cell depending on the equality convention. Refinement reduces the displacement slowly but does not remove the phase sensitivity.

---

## 3. Current implementation findings

### 3.1 `auto_grid` controls unrelated numerical policies

`PDEParams` describes `auto_grid` as a feature-aware default and places spatial sizing, time density, and event damping controls next to it:

- `quantark/asset/equity/param/engine_params.py:394-406`
- `quantark/asset/equity/param/engine_params.py:425-467`

The base solver uses the same boolean to:

- Add spot, barriers, and refinement levels as spatial critical points.
- Activate the adaptive mesh.
- Increase spatial size using `log_dx_target`.
- Enable mandatory event-aligned time grids.
- Select day-based resolution heuristics.

See `quantark/asset/equity/engine/pde/base_pde_solver.py:923-959` and `:977-1048`.

These decisions have different mathematical purposes and must not share one on/off switch.

### 3.2 Discrete triggers are forced onto nodes and sampled one-sidedly

Phoenix explicitly adds each coupon barrier to the critical-point list (`quantark/asset/equity/engine/pde/phoenix_pde_solver.py:144-155`). The adaptive spatial grid is then constructed and every critical point is overwritten onto the nearest node before the result is sorted:

- `quantark/asset/equity/engine/pde/spatial_grid.py:521-543`
- `quantark/asset/equity/engine/pde/spatial_grid.py:633-641`

The generic Snowball/Phoenix barrier mask then applies inclusive comparisons such as `s_vec >= barrier` and `s_vec <= barrier`:

- `quantark/asset/equity/engine/pde/snowball_pde_solver.py:990-1024`

Phoenix coupon transitions use that Boolean mask to select entire nodal branches:

- `quantark/asset/equity/engine/pde/phoenix_pde_solver.py:608-659`

This is appropriate for some continuous boundary conditions, but not for a discretely observed value jump. For the latter it assigns the entire trigger cell to one branch based on an arbitrary equality convention.

### 3.3 The mesh is concentrated by density, not by event semantics

The code treats spot, strikes, continuous barriers, discrete KO/KI barriers, and coupon barriers as one list of "critical points." The literature distinguishes at least:

- A continuous absorbing boundary.
- A discrete state-transition threshold.
- A continuous payoff kink.
- A point where the final value is interpolated.

These features require different placement rules. More density near a discrete trigger cannot, by itself, correct a one-sided representation of that trigger.

There are also two secondary implementation inconsistencies:

1. `log_dx_target` influences the suggested point count in `base_pde_solver.py:950-957`, but it is not passed as `eps_crit` to either `SpatialGrid.build` call at `base_pde_solver.py:795-801` or `:833-839`. The mesh therefore falls back to `SpatialGrid.DEFAULT_EPSILON_CRIT` for its concentration parameter.
2. `_calculate_beta_for_multi_crit` calibrates against the minimum local spacing found across all critical points (`spatial_grid.py:492-506`). Meeting a target at the best-resolved critical point does not establish that every critical point meets it.

Fixing those two details would improve internal consistency, but would not solve the discrete-event phase problem.

The automatic domain is also a fixed four-standard-deviation lognormal heuristic with strike/barrier padding (`spatial_grid.py:645-706`); it has no position-level boundary-error check. Current PDE profiles remain fixed `N_x`/`N_t` presets, and their product hints only add generic barrier/reverse refinement (`engine_param_profiles.py:68-99`, `:280-295`). They are configuration presets, not accuracy-driven plans.

### 3.4 Time refinement cannot repair a spatial event projection

Mandatory event dates are useful and should remain exact. The problem is using a fixed `event_steps_per_day` as an accuracy certificate. The current time-grid path fills event intervals using that rule:

- `quantark/asset/equity/engine/pde/base_pde_solver.py:961-1020`

Increasing the number of propagation steps can reduce temporal error, but it cannot correct a Heaviside jump assigned to the wrong spatial measure. This explains why time/spatial refinement could appear stable while retaining approximately 2% bias.

The current convergence gate is correspondingly incomplete:

- `test/test_pde_grid_convergence_gate.py:1-19` explicitly limits strong checks for gamma and theta.
- `test/test_pde_grid_convergence_gate.py:68-87` compares production goldens and time densities at a fixed spatial/event representation.

Those tests remain valuable regression tests, but they are not an external accuracy or grid-phase certificate.

### 3.5 Current event damping is active, but is not classical Rannacher

An earlier inspection characterized `rannacher_at_events` and `event_theta` as vestigial. That is not true for the current checkout:

- `BackwardOperator.theta_by_step` actively constructs a terminal/event theta schedule at `quantark/asset/equity/engine/pde/backward_operator.py:95-140`.
- Phoenix consumes the schedule at `quantark/asset/equity/engine/pde/phoenix_pde_solver.py:479-489`.

However, the current implementation still has three problems:

1. Event damping is gated by `params.auto_grid` at `backward_operator.py:122`, although damping should depend on event regularity rather than mesh-selection mode.
2. The default treatment changes one complete propagation step to backward Euler.
3. It does not split nominal Crank-Nicolson intervals into the half-steps analyzed in the Rannacher literature.

Most importantly, damping suppresses time-stepping ringing but does not remove the spatial quantization error created by the raw coupon mask.

---

## 4. Root-cause conclusion

### 4.1 Confirmed facts

- The core diffusion path is not the dominant cause of the investigated discrepancy.
- `auto_grid=True` snaps all feature levels, including discrete coupon thresholds, to nodes.
- Discrete event branches are selected by inclusive Boolean nodal masks.
- Reversing the equality side reverses the error sign, while averaging the two nearly removes it.
- Uniform-grid Phoenix values change materially with grid phase.
- Current event damping is coupled to `auto_grid` and is not implemented as split-step Rannacher damping.

### 4.2 Primary root cause

The dominant error is the spatial representation of discrete event operators:

> A discretely observed coupon/KO/KI threshold is treated as if it were a continuous boundary that should be snapped to a node.

At an event time, the exact state transition has the form

\[
J[V](S)
=
\mathbf{1}_{S < B}V_{\mathrm{miss}}(S)
+
\mathbf{1}_{S \ge B}V_{\mathrm{pay}}(S).
\]

Sampling this discontinuity directly at a node creates a one-sided, phase-dependent approximation. The adaptive mesh makes the error systematic because it deliberately puts the threshold on a node. A uniform grid changes the phase as it is refined, producing oscillation and occasional accidental agreement.

### 4.3 Contributing causes

- Pure Crank-Nicolson propagation preserves high-frequency components introduced by repeated event jumps.
- The current one-step damping rule is weaker than the Rannacher construction supported by analysis.
- Spatial and temporal resolution are selected by fixed heuristics rather than estimated output error.
- Existing convergence gates do not independently test spatial, temporal, domain, and phase errors.

---

## 5. Literature review

### 5.1 Placement and projection of discontinuities

[Pooley, Vetzal, and Forsyth (2003), "Convergence Remedies for Non-Smooth Payoffs in Option Pricing"](https://cs.uwaterloo.ca/~paforsyt/report.pdf), DOI `10.21314/JCF.2003.101`, compares cell averaging, shifting a discontinuity midway between nodes, and true \(L^2\) projection. The paper finds that:

- Raw nodal sampling creates quantization error.
- Spatial/data treatment and special time stepping are both needed for robust second-order behavior.
- Projection and damping must be repeated whenever a monitoring event introduces a new discontinuity.
- A continuous payoff kink can be represented exactly by piecewise-linear basis functions when the kink is on a node; a value jump cannot.

This directly explains why the protected loss-floor kink can behave well while the Phoenix coupon machinery rings.

[Boyle and Tian (1998), "An Explicit Finite Difference Approach to the Pricing of Barrier Options"](https://doi.org/10.1080/135048698334718) gives a particularly useful semantic rule:

- A continuously monitored barrier should pass through a node layer.
- A discretely monitored barrier should lie halfway between node layers.
- Spot can be recovered by interpolation rather than forcing spot onto the grid.

[Cui, Li, and Zhang (2024), "Pricing and Hedging Autocallable Products by Markov Chain Approximation"](https://doi.org/10.1007/s11147-024-09206-z) applies the same distinction directly to autocallables:

- Discrete coupon/redemption/termination levels are placed midway between adjacent states.
- Continuous barriers are placed on states.
- Their phase-correct grids show approximately second-order, smooth convergence across tested Black-Scholes, CEV, Kou, regime-switching, time-dependent, SABR, and Heston cases; Variance Gamma is closer to order 1.5.
- A conventional grid is approximately first-order in the reported Black-Scholes comparison.
- If one level has conflicting continuous and discrete roles, the paper uses separate grids with controlled interpolation.

The Cui-Li-Zhang theorem is for continuous-time Markov-chain approximation rather than QuantArk's finite-difference operator. It is therefore strong design evidence, not a direct theorem for this code. Boyle-Tian and Pooley-Vetzal-Forsyth independently support the same placement/projection rule in finite-difference/finite-element settings.

### 5.2 Repeated event discontinuities

[Luo and Huang (2025), "An Analytically Modified Finite Difference Scheme for Pricing Discretely Monitored Options"](https://www.mdpi.com/2227-7390/13/2/241), DOI `10.3390/math13020241`, is the most directly relevant recent PDE study. It includes barrier, autocallable, and Snowball-type products and reports:

- Unmodified Crank-Nicolson is roughly first-order for repeated discontinuities and can fail to converge in the maximum norm.
- CN with Rannacher treatment and TR-BDF2 remain roughly first-order while the repeated discontinuities remain in the numerical data.
- Decomposing each event into a smooth continuation plus analytically valued cash/asset binary terms restores approximately second-order convergence in their experiments.
- Local refinement is effective after the discontinuity treatment is corrected.

Their analytic construction assumes Black-Scholes dynamics with deterministic time-dependent coefficients and a restricted payoff form. It is a promising BSM optimized path, not a generic Local Volatility, Heston, SLV, or Phoenix-memory solution.

### 5.3 Rannacher damping and space-time coupling

[Rannacher (1984), "Finite Element Solution of Diffusion Problems with Irregular Data"](https://eudml.org/doc/132904), DOI `10.1007/BF01390130`, establishes implicit startup treatment for irregular initial data.

[Giles and Carter (2006), "Convergence Analysis of Crank-Nicolson and Rannacher Time-Marching"](https://people.maths.ox.ac.uk/gilesm/files/giles_carter.pdf), DOI `10.21314/JCF.2006.152`, shows that:

- Replacing the first two Crank-Nicolson steps with four backward-Euler half-steps restores second-order convergence for value, delta, and gamma in their model problems.
- Very small clustered startup steps are not equivalent to proper damping and can reduce its effectiveness.
- Accuracy depends on a dimensionless space-time diffusion ratio; selecting time steps independently from the local spatial scale is not generally efficient.
- Richardson extrapolation requires phase-consistent refinement with a stable observed order.

[d'Halluin, Forsyth, and Vetzal (2004/2005), "Robust Numerical Methods for Contingent Claims under Jump Diffusion Processes"](https://cs.uwaterloo.ca/~paforsyt/jump.pdf), DOI `10.1093/imanum/drh011`, applies \(L^2\) projection and Rannacher treatment after every daily barrier observation and reports quadratic convergence in that experiment.

### 5.4 Automatic and goal-oriented grids

[Lötstedt, Persson, von Sydow, and Tysk (2007), "Space-Time Adaptive Finite Difference Method for European Multi-Asset Options"](https://doi.org/10.1016/j.camwa.2006.09.014) uses adjoint-weighted spatial and temporal truncation errors to refine a grid until a global output tolerance is met.

[Goll, Rannacher, and Wollner (2015), "The Damped Crank-Nicolson Time-Marching Scheme for Adaptive Solution of the Black-Scholes Equation"](https://www.math.uni-hamburg.de/home/wollner/preprints/JCF_18-4_2014_preprint.pdf), DOI `10.21314/JCF.2015.301`, develops goal-oriented Dual Weighted Residual estimators for prices and Greeks and demonstrates substantial savings over global refinement in its examples.

These papers establish the right objective: control error in the requested price or Greek, rather than target a fixed `dx`. Their demonstrations are substantially smoother than a multi-state Phoenix with repeated discontinuous event maps. Full adjoint/DWR adaptation should therefore be a later optimization, after event correctness and a simpler refinement controller are established.

[Lyu et al. (2021), "Optimal Non-Uniform Finite Difference Grids for the Black-Scholes Equations"](https://doi.org/10.1016/j.matcom.2020.12.002) constructs product-specific grids by repeatedly removing the least important node from a trusted fine solution. Tests include cash digitals and one-, two-, and three-dimensional ELS products. The optimal grid changes with product parameters, supporting cached family templates as initial guesses rather than one universal grid.

### 5.5 Smooth nonuniform meshes

[In 't Hout and Foulon (2010), "ADI Finite Difference Schemes for Option Pricing in the Heston Model with Correlation"](https://arxiv.org/abs/0811.3427) uses smooth sinh-type nonuniform mappings and states mesh regularity conditions in which adjacent spacing changes at the appropriate order. Post-constructing a grid and then independently snapping several nodes can violate those assumptions.

QuantLib provides a useful production pattern:

- [`FdmBlackScholesMesher`](https://github.com/lballabio/QuantLib/blob/master/ql/methods/finitedifferences/meshers/fdmblackscholesmesher.cpp) uses model-aware log-domain tails and an optional concentration point.
- [`Concentrating1dMesher`](https://github.com/lballabio/QuantLib/blob/master/ql/methods/finitedifferences/meshers/concentrating1dmesher.cpp) builds concentration through the mapping rather than overwriting and sorting finished nodes.
- [`FdmBackwardSolver`](https://github.com/lballabio/QuantLib/blob/master/ql/methods/finitedifferences/solvers/fdmbackwardsolver.cpp) separates stopping times, damping steps, and propagation scheme.

QuantLib is not evidence for a complete automatic accuracy controller: its grid sizes remain caller-selected.

### 5.6 Contract semantics versus numerical approximation

[Broadie, Glasserman, and Kou (1997), "A Continuity Correction for Discrete Barrier Options"](https://www.columbia.edu/~mnb2/broadie/Assets/bgk_mf.pdf), DOI `10.1111/1467-9965.00035`, approximates a discrete barrier by shifting a continuous barrier. That is a useful explicit performance mode, but it changes the representation of monitoring. An automatic grid planner must never select BGK silently when the requested engine mode is exact discrete monitoring.

---

## 6. Redesign objective

Replace `auto_grid: bool` internally with a per-position `PDEDiscretizationPlanner`.

The production controller should not claim to find a global mathematical optimum. It should walk a deterministic, offline-calibrated candidate ladder and return the first plan whose estimated error is certified. Candidate ordering should approximate the engineering objective:

\[
\min_{\pi}
C\left(
\pi;
N_{\mathrm{state\ surfaces}},
N_{\mathrm{requested\ outputs}},
N_{\mathrm{risk\ scenarios}}
\right)
\]

subject to:

\[
\widehat{E}_q(\pi) \le \tau_q
\quad
\text{for every requested quantity }q,
\]

plus exact contract semantics and mesh-quality invariants.

Measured cost must include planner pilots, state surfaces, event-stat streams, requested risk scenarios, factorization reuse, memory, and wall time. `N_x * N_t` alone is not a sufficient cost proxy. For a one-off PV, a conservative prevalidated plan can be cheaper than several adaptive pilots.

The error values are numerical estimates, not formal proofs. The planner must retain diagnostics and fail closed when the estimates are unreliable.

### 6.1 Inputs

- Product and current lifecycle state.
- Pricing environment and model family.
- Requested outputs: PV, delta, gamma, theta, event probabilities, cashflow streams, and so on.
- Risk-bump envelope.
- Explicit absolute/relative tolerances.
- Runtime, node, and memory caps.
- Exact versus approximate monitoring policy.

### 6.2 Output

An immutable, serializable `PDEDiscretizationPlan` containing:

- Domain and spatial nodes.
- Time nodes and mandatory event dates.
- Event projectors.
- Event-local damping schedule.
- Propagation scheme.
- Actual spatial/time sizes.
- Mesh-quality metrics.
- Estimated spatial, temporal, domain, and event-phase errors.
- Observed convergence orders where reliable.
- Plan hash, product/event-graph hash, market/model snapshot identifier, runtime estimate, and acceptance/failure reason.

The same plan must be reused by price, event-stat, cash-leg, and Greek calculations.

Certification must have an explicit status:

- `CERTIFIED`
- `BUDGET_EXCEEDED`
- `NONCONVERGENT`
- `UNSUPPORTED_FEATURE`
- `INVALID_MARKET_INPUT`

In certified mode, `price()` should raise a numerical-convergence error for every non-`CERTIFIED` result. A detailed or explicitly best-effort API may return the best estimate with its status and diagnostics. Alternate-engine routing belongs above the PDE solver and must preserve fallback provenance and uncertainty. An MC fallback must report both sampling uncertainty and monitoring/time-discretization uncertainty; it cannot be presented as automatically satisfying the original PDE tolerance.

---

## 7. Proposed architecture

### 7.1 `PDEAccuracySpec`

Profiles such as `fast`, `balanced`, and `accurate` may remain user-facing, but each profile must resolve to explicit tolerances and budgets:

```python
@dataclass(frozen=True)
class PDEAccuracySpec:
    pv_abs: float
    pv_rel: float
    delta_abs: float | None = None
    gamma_abs: float | None = None
    event_prob_abs: float | None = None
    max_spatial_nodes: int = 0
    max_time_nodes: int = 0
    max_runtime_ms: float | None = None
    on_failure: str = "raise"
```

Default numerical tolerances require desk calibration and should not be invented from the literature.

### 7.2 `PDENumericalFeatureCompiler`

Compile product structure into numerical roles rather than a flat critical-point list:

| Feature role | Spatial rule | Time/event rule |
| --- | --- | --- |
| Continuous absorbing barrier | Exact node or domain boundary | Enforced throughout propagation |
| Discrete KO/KI/coupon/redemption threshold | Cell face when compatible; otherwise conservative projection | Exact event time only |
| Continuous payoff kink/strike/floor | Node or exact piecewise-linear projection | Terminal/event treatment as applicable |
| Spot/evaluation point | Stable interpolation stencil | None |
| Dividend, reset, exercise, or coefficient knot | Model-dependent spatial mapping | Exact time node |
| Lifecycle-inactive event | Omit | Omit |

The compiler must distinguish pre-KI/post-KI states, reverse directions, memory-coupon state transitions, and coincident events.

### 7.3 `PDECandidatePlanner`

The first implementation should use a static spatial mesh for the whole solve. A moving mesh would complicate factorization reuse, event projection, Greeks, and validation.

Candidate construction:

1. Select model-aware log-domain bounds from the forward path, distribution tails, dividends, and product features.
2. Build a uniform-log, piecewise-uniform, or smooth constrained sinh/ODE mesh.
3. Enforce continuous-boundary node constraints and discrete-trigger face constraints during construction.
4. Do not overwrite and sort finished nodes.
5. Enforce positive spacing and a bounded adjacent-spacing ratio.
6. Add all true event and coefficient-knot times as mandatory nodes.
7. Use `event_steps_per_day` only as an initial time-density guess.
8. Couple the guess to local effective diffusivity and minimum spatial spacing.

For a new or difficult position, the planner runs pilots. For a previously validated normalized product/model signature, a cached template may seed the first candidate, but it cannot replace the position-level acceptance check.

### 7.4 `PDEEventProjector`

Provide three explicit event representations:

1. `FACE_ALIGNED`
   - Use when all relevant discrete thresholds can remain cell faces across the refinement family.
   - A Boolean branch is then phase-correct because the discontinuity lies between nodes.
   - Factor-two nodal refinement does not preserve this phase: a midpoint becomes a node. Use nonnested phase-preserving meshes, an odd refinement ratio, or switch the refinement family to conservative projection.

2. `CONSERVATIVE_PROJECTION`
   - Generic default for conflicting thresholds, nonuniform meshes, Local Volatility, Heston, SLV, and multi-state Phoenix surfaces.
   - Project the full left/right event operator using cell averages or an \(L^2\) projection onto the numerical basis.
   - Apply projection state-by-state for memory coupons.

3. `ANALYTIC_BINARY_SPLIT`
   - Later optimized Black-Scholes path based on Luo-Huang.
   - Requires explicit applicability checks.
   - Must fall back to conservative projection outside the proven payoff/model class.

A hard-coded weight of one-half at an equality node is not a general solution. On a nonuniform grid, projection weights must use the actual local cell geometry and the one-sided continuation values.

### 7.5 `PDEEventIntegrator`

Event treatment must be independent of mesh-selection mode:

1. Propagate to the exact event date.
2. Apply the product state transition.
3. Project or analytically regularize any newly introduced discontinuity.
4. Apply a precisely specified damping episode.
5. Resume the selected propagation scheme.

The first literature-backed damping candidate to test is the standard Rannacher construction: replace the first two nominal Crank-Nicolson steps after each nonsmooth event with four backward-Euler half-steps. The implementation should express the actual substeps, rather than only changing `theta` for a complete existing step.

This is a baseline experiment, not a universal final prescription. Giles-Carter also studies alternative substep placement, and dense event calendars can leave fewer than two nominal intervals between events. The event integrator must define an explicit overlap policy and benchmark projected Crank-Nicolson, fully implicit intervals, and L-stable alternatives rather than silently stacking damping windows.

TR-BDF2 or another L-stable scheme may later be benchmarked as a propagation default. It is not a substitute for correcting the event operator.

### 7.6 `PDEPlanVerifier`

For each requested quantity, run at least:

1. Baseline candidate.
2. Space-only refinement.
3. Time-only refinement.
4. Expanded-domain candidate.
5. Event-phase or projector perturbation.

All refinements must preserve:

- The exact event calendar.
- The same event representation family.
- Continuous-boundary node placement.
- Discrete-trigger face placement or the same conservative projector.
- The same requested-output definitions.

For face-aligned events, "preserve" means preserving the midpoint phase, not merely using a nested grid. Factor-two nested refinement is invalid for this check because it moves the trigger from a face to a node.

Use a third systematic level before estimating observed order or applying Richardson extrapolation. If order is unstable or oscillatory, use a conservative raw error envelope with a safety factor.

Choose the next refinement by measured requested-output error reduction per unit runtime. Any local spatial refinement must trigger a time/damping recheck, with a diagnostic such as:

\[
\max_{i,n}
\frac{\Delta t_n\,\sigma_{\mathrm{eff}}(S_i,t_n)^2}
     {\Delta x_i^2}.
\]

This is a diagnostic and candidate-selection aid, not a universal stability or accuracy theorem.

### 7.7 Risk-plan freezing

The planner must receive the full bump envelope and construct one plan valid for all non-theta bumps. Base, spot, volatility, rate, and dividend scenarios must use identical:

- Domain.
- Spatial nodes.
- Time nodes.
- Event projectors.
- Damping schedule.

Otherwise the reported Greek includes grid movement. The geometry is frozen, not the model coefficients: every scenario must rebuild the operator from its bumped market inputs. Theta may legitimately change maturity and the remaining event calendar. A theta bump crossing an observation date creates a new event graph and therefore requires a new compatible plan, while preserving the spatial design envelope where valid.

---

## 8. Certification algorithm

```text
compile active numerical features
        |
build phase-correct seed plan
        |
solve requested quantities on baseline plan
        |
run x-only, t-only, domain, and phase checks
        |
are all estimated errors within tolerance?
        | yes
freeze and return the first certified plan in the ordered candidate ladder
        |
        no
select refinement with best measured error reduction / runtime
        |
caps reached or convergence unreliable?
        | no -> repeat
        | yes
raise or route through an explicitly authorized fallback
```

Required diagnostics:

- Requested and actual `N_x` / `N_t`.
- Domain and boundary-expansion delta.
- Minimum/maximum spacing and adjacent-spacing ratio.
- Mandatory-event count.
- Projected-event and damping-event counts.
- Minimum/maximum time step.
- Maximum local diffusion ratio.
- Space-only, time-only, domain, and phase output deltas.
- Observed order and whether the asymptotic test passed.
- Runtime, caps, and acceptance/failure reason.
- Fallback engine, provenance, and uncertainty when an authorized fallback is used.

---

## 9. Final implementation plan

### Phase 0 - Lock the evidence

Deliverables:

- Add deterministic Snowball and protected-Phoenix fixtures matching the investigation.
- Store all product, market, lifecycle, grid, MC seed/path, and reference-result inputs.
- Reproduce the `auto_grid=True`, uniform-grid, and equality-side results.
- Add a machine-readable result table for PV and requested Greeks.

Exit gate:

- The material bias, Phoenix oscillation, and equality-side sign reversal are reproducible in CI or a dedicated numerical-validation job.

### Phase 1 - Correct discrete event representation

Deliverables:

- Add numerical feature-role classification.
- Implement phase-correct face placement for compatible discrete thresholds.
- Implement a conservative event projector for generic/multi-state cases.
- Remove the assumption that every barrier-like level must be a node.
- Implement explicit event-local damping, beginning with standard split-step Rannacher as the literature-backed baseline to benchmark.
- Decouple damping policy from `auto_grid`.

Exit gates:

- The event projector reproduces constant and affine branch functions to its designed order.
- The discrete event operator is linear and conservative, uses no negative projection weights, and introduces no spurious overshoot.
- Inclusive/exclusive equality choices no longer produce material PV differences.
- Phase-shift sweeps stay within the requested tolerance.
- Protected Phoenix converges smoothly against QUAD/MC references.
- Value, delta, and gamma contain no event-node ringing beyond their specified tolerances.

### Phase 2 - Introduce plan artifacts and separate policies, still manually selected

Deliverables:

- Add `PDEAccuracySpec`, `PDENumericalFeatureCompiler`, and immutable `PDEDiscretizationPlan`.
- Split existing controls into at least:
  - `domain_policy`
  - `spatial_policy`
  - `time_policy`
  - `event_projection_policy`
  - `damping_policy`
  - `propagation_scheme`
- Preserve the existing behavior as deprecated `grid_policy="legacy_auto"`.
- Add the new `grid_policy="certified"` as opt-in; reject conflicting legacy and new arguments.
- Make price, event stats, and prepared/session paths consume an explicit plan.

Exit gates:

- Existing manually configured grids remain reproducible.
- The plan is serializable, hashable, and visible in diagnostics.
- Grid geometry and event policy, not just critical points, are invariant across non-theta risk bumps within the certified envelope; scenario coefficients are rebuilt.

### Phase 3 - Add a certified online controller for one-dimensional BSM PV

Deliverables:

- Implement space-only, time-only, domain, and phase verification runs.
- Add three-level observed-order checks.
- Walk a deterministic, offline-calibrated candidate ladder and stop at the first certified plan.
- Implement the explicit certification statuses and cap/failure API.
- Calibrate the estimator on a held-out BSM product/lifecycle matrix so claimed error conservatively covers observed error.
- Keep the controller limited to PV until this calibration passes.

Exit gates:

- The planner returns the first certified BSM PV plan in its documented candidate order.
- It rejects the known oscillatory Phoenix sequences rather than extrapolating them.
- It never treats production-golden parity alone as an accuracy certificate.
- Certification status and observed reference error agree on the held-out validation matrix.

### Phase 4 - Extend BSM certification to Greeks, event outputs, and lifecycle bumps

Deliverables:

- Add delta, gamma, theta, and event-output tolerances.
- Plan over the non-theta bump envelope while rebuilding scenario coefficients.
- Recompile the event graph when theta crosses an observation or coupon date.
- Add Greek-specific candidate ladders and estimator calibration.
- Define desk-approved `fast`, `balanced`, and `accurate` tolerance profiles.

Exit gates:

- Claimed Greek/event-output error conservatively covers held-out observed error.
- Grid geometry is identical across compatible non-theta bumps.
- Theta event-graph changes are explicit and reproducible.

### Phase 5 - Recover performance

Deliverables:

- Cache normalized family/model candidate templates.
- Reuse one accepted plan and factorization schedule across price, event streams, and risk scenarios.
- Prototype the Luo-Huang analytic binary split for its supported BSM product class.
- Benchmark CN with projected events against TR-BDF2 and other L-stable alternatives.

Exit gates:

- Runtime improves without weakening any Phase 1-4 accuracy gate.
- Cache reuse cannot bypass per-position verification.
- The analytic fast path has explicit applicability and parity checks.

### Phase 6 - Add Local Volatility

Deliverables:

- Add Local Volatility domain/diffusivity adapters.
- Add coefficient-surface interpolation and domain-error checks.
- Derive or validate Local Volatility-specific residual/error indicators.

Exit gates:

- Local Volatility passes its own held-out price/Greek/reference matrix.
- BSM error calibration is not reused as proof of Local Volatility accuracy.

### Phase 7 - Design separate Heston/SLV certification

The current Heston/SLV Phoenix path is architecturally separate: it owns `N_x`, `N_v`, `N_t`, ADI scheme, grid style, variance grid, and variance-boundary choices (`phoenix_vol_pde_solvers.py:179-238`, `quantark/volmodels/adi_core.py:76-150`). It also currently maps real event dates to integer uniform-time keys (`phoenix_vol_pde_solvers.py:519-551`).

Deliverables:

- Add model-specific spot/variance/time grid planners and ADI-splitting checks.
- Apply the spot event projector slice-wise, while separately certifying variance-grid and splitting errors.
- Validate the cross derivative, \(v=0\) boundary, leverage-surface knots, and event-time representation.
- Consider adjoint/DWR localization only after the static two-dimensional verifier is stable.

Exit gates:

- Heston and SLV each pass their own price/Greek/reference matrix.
- No one-dimensional Black-Scholes tolerance or mesh theorem is claimed as a proof for Heston/SLV.

---

## 10. Validation matrix

The redesign must cover the cross-product below rather than a single Snowball row:

| Dimension | Required cases |
| --- | --- |
| Product | Vanilla, digital, continuous barrier, discrete barrier, Snowball, Phoenix, KO-reset |
| Protection | Protected, unprotected, mandatory-put where applicable |
| Direction | Standard and reverse |
| Coupon state | No-memory and memory |
| Observations | Single, sparse/monthly, 24-event, dense/daily |
| Lifecycle | Pre-KI, post-KI, near KO/coupon date, near maturity |
| Feature placement | Threshold near spot, far from spot, coincident continuous/discrete roles |
| Model | BSM first; Local Volatility next; Heston/SLV separately |
| Quantity of interest | PV, delta, gamma, theta, KO/KI/coupon probabilities, cashflow streams |
| Reference | Analytical where available; QUAD; high-accuracy MC with confidence intervals; fine projected-PDE reference |

Required numerical gates:

1. Event-projector unit properties: constant/affine reproduction, linearity, conservation, nonnegative weights, and no overshoot.
2. Exact event-date inclusion.
3. Spatial phase sweep.
4. Space-only and time-only refinement, with assertions that actual node arrays and hashes differ as intended.
5. Phase-preserving refinement for face-aligned triggers; no blind factor-two Richardson sequence.
6. Domain expansion.
7. Three-level observed order before Richardson.
8. Monotonicity/positivity and probability-mass invariants where applicable.
9. Frozen-plan Greek stability.
10. Held-out calibration showing the estimated error conservatively covers observed reference error.
11. Runtime, memory, and iteration caps.

Monte Carlo is a statistical reference, not an exact oracle. Comparisons must include confidence intervals and sufficient path/seed evidence.

Production-golden parity to a known biased value is characterization evidence, not a correctness gate. Any intentional repricing introduced by corrected event semantics must be versioned and explained.

---

## 11. Non-negotiable rules

- Do not snap every strike/barrier/coupon level to a node.
- Do not use `event_steps_per_day`, `log_dx_target`, or grid density as an error estimate.
- Do not claim convergence from two adjacent grids when node phase changes.
- Do not use factor-two nested refinement for a face-aligned trigger without projection; it moves the trigger onto a node.
- Do not use Richardson extrapolation without stable observed order.
- Do not certify Greeks from a PV-only check.
- Do not let `auto_grid` control event damping or contract semantics.
- Do not silently replace exact discrete monitoring with BGK or continuous monitoring.
- Do not move mandatory event dates to satisfy a time-step cap.
- Do not reuse a cached template as if it were a position-level certificate.
- Do not route to an alternate engine inside the PDE solver.
- Fail explicitly or use an explicitly authorized higher-level fallback with provenance and uncertainty when the budget is exhausted.

---

## 12. Relationship to the July 2026 event-distribution redesign

The earlier `docs/superpowers/specs/2026-07-01-pde-autocallable-event-distribution-redesign-design.md` correctly identified performance coupling between event alignment, dense KI schedules, event statistics, and repeated risk solves. Its time-grid decoupling and shared-solve architecture remain useful.

This investigation supersedes two accuracy assumptions in that document:

1. Event-wise damping is not sufficient while discrete coupon/KO/KI jumps are still sampled one-sidedly.
2. The current theta-switching implementation should not be described as complete Rannacher treatment.

The performance redesign and this accuracy redesign should be combined through the immutable `PDEDiscretizationPlan`; the time-grid performance work should not be reverted.

The older `docs/phoenix-engine-investigation/investigation-summary.md` is historical evidence from a different 12-observation case. Its statements that the coupon barrier had minimal impact and that PDE convergence was monotonic must not be generalized to the present protected 24-observation case; the newer phase/equality diagnostics supersede those conclusions for auto-grid design.

---

## 13. Open questions

1. Should the generic projector use finite-volume cell averages, consistent-mass \(L^2\) projection, or a product-specific exact integration of piecewise-linear continuation values?
2. When one level is both a continuous barrier and a discrete trigger, is one grid plus projection sufficient for all requested Greeks, or is a controlled two-grid transfer required?
3. How far can the Luo-Huang analytic split be extended to Phoenix memory states without losing its smoothness advantage?
4. What desk-level PV and Greek tolerances should define `fast`, `balanced`, and `accurate`?
5. Which alternate engine should be used when the PDE planner fails: QUAD for supported one-factor products, MC with confidence intervals, or an explicit error?
6. What model-specific domain and residual estimators are appropriate for Local Volatility, Heston, and SLV?

These questions affect implementation choices, but they do not change the first priority: correct and verify the discrete event operator before optimizing the mesh.

---

## 14. Final recommendation

Approve a redesign of the internal auto-grid feature around a per-position, requested-output accuracy controller.

The first implementation milestone is not a new nonuniform-grid formula. It is:

> phase-correct or conservatively projected discrete events, followed by genuine event-local damping and independent spatial/time/domain/phase verification.

Only after that foundation passes the protected-Phoenix regression should QuantArk begin certified online selection. Preserve the existing behavior as `legacy_auto`, introduce the certified policy as opt-in, shadow/dual-run it over the validation corpus, and switch defaults only after estimator calibration and latency review. The original `auto_grid` boolean can then be deprecated without silently changing both numerical semantics and runtime.

No production code was changed as part of this investigation report.

---

## 15. References

1. Boyle, P., and Tian, Y. (1998). "An Explicit Finite Difference Approach to the Pricing of Barrier Options." *Applied Mathematical Finance* 5(1), 17-43. <https://doi.org/10.1080/135048698334718>
2. Broadie, M., Glasserman, P., and Kou, S. (1997). "A Continuity Correction for Discrete Barrier Options." *Mathematical Finance* 7(4), 325-348. <https://doi.org/10.1111/1467-9965.00035>
3. Cui, Y., Li, L., and Zhang, G. (2024). "Pricing and Hedging Autocallable Products by Markov Chain Approximation." *Review of Derivatives Research* 27, 259-303. <https://doi.org/10.1007/s11147-024-09206-z>
4. d'Halluin, Y., Forsyth, P. A., and Vetzal, K. R. (2005). "Robust Numerical Methods for Contingent Claims under Jump Diffusion Processes." *IMA Journal of Numerical Analysis* 25(1), 87-112. <https://doi.org/10.1093/imanum/drh011>
5. Giles, M. B., and Carter, R. (2006). "Convergence Analysis of Crank-Nicolson and Rannacher Time-Marching." *Journal of Computational Finance* 9(4), 89-112. <https://doi.org/10.21314/JCF.2006.152>
6. Goll, C., Rannacher, R., and Wollner, W. (2015). "The Damped Crank-Nicolson Time-Marching Scheme for Adaptive Solution of the Black-Scholes Equation." *Journal of Computational Finance* 18(4). <https://doi.org/10.21314/JCF.2015.301>
7. Heston, S., and Zhou, G. (2000). "On the Rate of Convergence of Discrete-Time Contingent Claims." *Mathematical Finance* 10(1), 53-75. <https://doi.org/10.1111/1467-9965.00080>
8. In 't Hout, K. J., and Foulon, S. (2010). "ADI Finite Difference Schemes for Option Pricing in the Heston Model with Correlation." *International Journal of Numerical Analysis and Modeling* 7(2), 303-320. <https://arxiv.org/abs/0811.3427>
9. Lötstedt, P., Persson, J., von Sydow, L., and Tysk, J. (2007). "Space-Time Adaptive Finite Difference Method for European Multi-Asset Options." *Computers & Mathematics with Applications* 53(8), 1159-1180. <https://doi.org/10.1016/j.camwa.2006.09.014>
10. Luo, G., and Huang, M. (2025). "An Analytically Modified Finite Difference Scheme for Pricing Discretely Monitored Options." *Mathematics* 13(2), 241. <https://doi.org/10.3390/math13020241>
11. Lyu, J., Park, E., Kim, S., et al. (2021). "Optimal Non-Uniform Finite Difference Grids for the Black-Scholes Equations." *Mathematics and Computers in Simulation* 182, 690-704. <https://doi.org/10.1016/j.matcom.2020.12.002>
12. Pooley, D. M., Vetzal, K. R., and Forsyth, P. A. (2003). "Convergence Remedies for Non-Smooth Payoffs in Option Pricing." *Journal of Computational Finance* 6(4), 25-40. <https://doi.org/10.21314/JCF.2003.101>
13. Rannacher, R. (1984). "Finite Element Solution of Diffusion Problems with Irregular Data." *Numerische Mathematik* 43, 309-327. <https://doi.org/10.1007/BF01390130>
14. Zhang, G., and Li, L. (2019). "Analysis of Markov Chain Approximation for Option Pricing and Hedging: Grid Design and Convergence Behavior." *Operations Research* 67(2), 407-427. <https://doi.org/10.1287/opre.2018.1791>
