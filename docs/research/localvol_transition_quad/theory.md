# Local-vol transition QUAD: equations, literature and scope

Research implementation in `example/case_studies/localvol_transition_quad/`. It registers no production engine and changes no routing. Independent of `example/case_studies/localvol_quad/`.

## The problem

Between two monitoring dates a QUAD engine needs the transition law of the underlying. Under Black–Scholes it is Gaussian; under Heston the affine structure gives an exact partial Fourier kernel (`example/case_studies/heston_quad`, untracked). Under 1-D local volatility

\[ dX_t = (r-q-\tfrac12\sigma^2(e^{X_t},t))\,dt + \sigma(e^{X_t},t)\,dW_t,\qquad X=\ln S, \]

there is neither a closed form nor a characteristic function. The repository's `LocalVolSurface` is bilinear in (t, S) between a 6 × 33 Dupire node grid with flat extrapolation: σ is Lipschitz with kinks at every strike node and every time pillar. The choice of transition operator is therefore the whole design.

## Literature review

| Family | Key references | Kernel | Verdict for this surface |
|---|---|---|---|
| Classical QUAD | Andricopoulos, Widdicks, Duck, Newton (2003, JFE); Andricopoulos et al. (2007); Fusai & Recchioni (2007, JEDC); Sesana, Marazzina & Fusai (2014, EJOR, CEV) | closed-form density (BSM, Merton, CEV); node-on-discontinuity placement | Recursion and node placement carried over; the kernel is not available. |
| CF-inversion QUAD | Chen, Härkönen & Newton (2014, "Advancing the universality of quadrature methods to any underlying process"); Fang & Oosterlee COS (2008/2009, 2011 Heston Bermudan/barrier) | density from a characteristic function | No local-vol characteristic function. Excluded. |
| Markov-chain (CTMC) approximation | Mijatović & Pistorius (2013, Math. Finance, "Continuously monitored barrier options under Markov processes"); Lo & Skindilias (2014); Cui, Kirkby & Nguyen (2018, SIFIN, SABR/SLV); Zhang & Li (2019, Oper. Res., grid design and convergence) | generator matrix Q on a non-uniform grid; P = exp(hQ) | **Adopted (kernel A).** Robust to kinked σ; Zhang–Li's barrier placement result is the reason barriers sit midway between states here. |
| Analytic density expansions | Aït-Sahalia (2002 Econometrica; 2008 AoS) Hermite after Lamperti; Pagliarani & Pascucci (2012), Pagliarani, Pascucci & Riga (2013, SIFIN, adjoint expansions, time-dependent error bounds); Lorig, Pagliarani & Pascucci (2015); Gatheral, Hsu, Laurence, Ouyang & Wang (2012, heat kernel) | Gaussian × correction polynomials | Excluded as a default: the expansion coefficients differentiate σ, which is only piecewise linear here, and every truncation is an approximation (exact-semantics rule). |
| Short-step kernels / path integrals | Montagna, Nicrosini & Moreni (2002); Ruijter & Oosterlee (2016, Euler/Milstein/order-2 weak Taylor CF in COS); Arguin, Liu & Wang (2017, Brownian bridge, discretely monitored Asians under LV) | chained short Gaussian steps, Richardson | **Prototyped (kernel B)** as the second bake-off arm. |
| Matrix-exponential evaluation | Weideman (2006, "Optimizing Talbot's contours for the inversion of the Laplace transform"); Weideman & Trefethen (2007, Math. Comp., parabolic/hyperbolic contours); Trefethen, Weideman & Schmelzer (2006, BIT, Talbot quadratures and rational approximations) | contour quadrature of the resolvent | **Adopted backend for kernel A**: stiffness-independent cost. |

## Kernel A — CTMC with an exact exponential

States are the centres c_i of a finite-volume grid in X. Each interior row has an up rate u_i and a down rate d_i, two unknowns fixed by two conditions:

\[ u_i(e^{h_i^+}-1) + d_i(e^{-h_i^-}-1) = r-q, \qquad u_i (h_i^+)^2 + d_i (h_i^-)^2 = \sigma^2(e^{c_i}, t). \]

The first makes the discounted stock a martingale of the chain exactly (Q e^{x} = (r−q) e^{x} row by row); the second matches the instantaneous log variance. Rows conserve mass, the end rows reflect, and a negative rate (a drift-dominated cell) is refused with `UnresolvedGrid` — never clipped.

Between event dates the chain advances by exp(Δt Q(t_mid)), Δt the interval cut at surface time pillars so that σ² is smooth in t on each piece, and further into equal pieces no longer than `max_piece` = 1/252 (Magnus order 2; σ moves linearly across a slab, and one midpoint exponential over a 0.25y slab left a measured 1e-4 time error on a 1y European put, 2e-6 once pieces are daily). Backward induction applies the pieces **latest first**; the generators of different slabs do not commute, so the order matters (a reversed loop was invisible on daily-event Snowballs and on constant surfaces, and was caught by the European self-convergence check, AC4). Two exact-exponential backends, both validated against dense `scipy.linalg.expm` to 1e-12. The default `auto` picks per operator: Chebyshev while its a-priori series has at most 64 terms (measured per-operator crossover ≈ 60–70 terms for 200–1200 cells), the contour beyond.

* **Talbot contour.** \( e^{A}v = \frac{1}{2\pi i}\int_\Gamma e^{z}(zI-A)^{-1}v\,dz \), Γ Weideman's optimized cotangent contour z(θ) = N(−0.6122 + 0.5017 θ cot(0.6407 θ) + 0.2645 iθ); the trapezoid rule converges like 3.89^{−N} whenever the spectrum lies on (−∞, 0] — true here because Q is similar to a symmetric negative semidefinite tridiagonal matrix. Conjugate symmetry halves the nodes: N = 24 costs 12 complex tridiagonal solves (`zgtsv`) per piece, independent of the stiffness Δt‖Q‖.
* **Chebyshev–Bessel.** \( e^{\Delta t Q} = e^{-\nu/2} \sum_k \epsilon_k I_k(\nu/2) T_k(A) \), A = (2/ν)ΔtQ + I, ν a Gershgorin bound; needs O(√ν) terms, so its cost grows like N² per day. Kept as a validated alternative.

## Kernel B — moment-matched Euler Gaussian + Richardson

Row i over a substep δ is N(c_i + (r − q − σ_i²/2)δ, σ_i²δ), integrated over the cells (semi-infinite end cells), banded at ±8 sd. Landing points are rounded to centres, which adds ≈ w²/12 of variance per step and, where widths change, a mean bias; the Gaussian's mean and variance are therefore corrected (two fixed-point passes) so that the realized law on the centres has the target moments. Richardson: 2·P(δ/2)P(δ/2) − P(δ) per pillar-cut piece.

## Grid, events and readout

Finite-volume grid in ln S. KO and KI are cell **edges** with a symmetric pair of equal cells around each (for a CTMC the effective threshold is the midpoint between neighbouring states; unequal widths measured first order). ln s0 is a cell **centre** inside a block of 2k+1 uniform cells (k = 3): the readout needs no interpolation, and native Greeks come from the centred stencil, delta = ∂_x V, gamma = ∂_xx V − ∂_x V (per unit notional, ×S0 and ×S0² of the S-derivatives). A strike inside the readout block is not anchored (a kink costs O(w²) under the midpoint rule). Refinement for convergence studies splits every cell into an odd number of equal parts, which keeps the barrier edges and the s0 centre fixed (nested ladders; halving cannot).

Events act on whole cells: KI at a daily observation switches the never-KI channel to the KI channel below KI; KO at a monthly observation overwrites both channels with 1 + ko_rate·(t_k + accrued offset) above KO.

## Exact contract scope

Mirrors the production `SnowballOption` as the model-validation builders configure it, priced per unit notional S0: KO when S ≥ KO paying 1 + ko_rate·t_k at t_k (annualized, accrued from 0); KO stays live after KI; never-KI maturity pays the flat 1 + rebate_rate (`AccrualConfig(INSTANT, is_annualized=True, is_annualized_rebate=False)`); KI'd maturity pays 1 + min(S_T − K, 0)/S0; KI when S ≤ KI, daily at k/252; KO monthly at T·k/months. Vanilla (put, or call priced directly) and the discrete monthly up-and-out call (zero rebate) are the sanity products. Flat r and q. Valuation before the first remaining observation only.

Risks follow production semantics: vega = central ±0.01 absolute parallel shift of the local-vol grid (surface never rebuilt); rho/carry = central ±1bp on flat r/q (surface not rebuilt); theta = valuation +1 calendar day with all observation times reduced by 1/365, coupon amounts fixed and the surface **not** rolled (production `shift_for_theta` with `keep_landing=True`), so theta is a full re-solve in every engine.
