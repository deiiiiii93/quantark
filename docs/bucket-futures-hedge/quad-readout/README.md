# The QUAD snowball's spot derivative is a staircase

Root cause of the identity-residual failures recorded in `../gates.md`.
Everything here is a standalone runtime-patched script; **no engine code was
changed**. Run from this directory with the repo on the path:

```
PYTHONPATH=<repo>:. <repo>/.venv/bin/python proof.py
```

## The finding

`SnowballQuadEngine._price_once` ends at
`snowball_quad_engine.py:463`:

```python
return math_utils.interpolate(value_surface, x=0.0)
```

and `QuadratureMath.interpolate` (`quad_math.py:243`) is
`np.interp(x, self.grid, values)` — **linear**.

That would be harmless if the grid moved with spot. It does not. Barrier
alignment snaps a node onto the closest barrier, so the lattice in
log-moneyness translates with spot in exactly the way that leaves the
lattice in ABSOLUTE PRICE standing still. Node `k` sits at `B * exp(k*h)`
whatever the spot is.

So over any spot move short of the grid's extent, `values` is a fixed array
on fixed absolute prices, and the engine's price is the linear interpolant
read at a position that sweeps across cells.

`proof.py` states this as three falsifiable predictions and measures all
three at the study's worst audit date (2024-01-19, spot 5306.99 against a
5050.48 barrier, 0.296y remaining):

| Prediction | Measured |
|---|---|
| the absolute price nodes do not move with spot | max drift 9.1e-13 index points over a 1.8-point spot move |
| the kinks in `V(S)` sit on the nodes | largest kink at 5319.8700, nearest node 5319.8741 — 0.0002 cells |
| the price IS the linear interpolant of the node values | max relative difference 2.0e-14 over 97 probes |

The third is not an approximation. At machine precision, `V(S)` is a
piecewise-linear function of `log S`.

## What that does to delta

Delta is therefore a staircase: flat across a cell, stepping at every node.
At the worst date the cell is **19.7 index points wide (0.371% of spot)**
and the risers are about **0.25 reference hands**.

A finite difference narrower than a cell returns that cell's chord slope,
which is the true delta at the cell MIDPOINT. `repro.py --n 201
--half-width-rel 0.004` shows the staircase directly; the risers are visible
as a 0.285-hand drop across 0.1 index points at spot 5319.87, with a
perfectly smooth run either side.

**The amplitude is set in the LOG coordinate, not in spot.** The
interpolation is linear in `x = log S`, so the chord equals `dV/dx` at the
cell midpoint and the delta error is `(x_mid - x) * (S*Gamma + Delta)`,
giving a peak-to-peak sawtooth of

```
h * |S*Gamma + Delta| / m_ref
```

A `(cell/2) * Gamma` reading, which treats the interpolant as linear in
spot, is wrong and over-predicts by an inconsistent factor.
`amplitude.py 0.00005` checks both against the observed sawtooth:

| centre | measured ptp | `(cell/2)*Gamma` | `h*(S*Gamma+Delta)` | ratio |
|---:|---:|---:|---:|---:|
| 5310.5 | 0.2681 | 0.5609 | 0.2227 | 1.204 |
| 5450.5 | 1.1058 | 1.4238 | 1.1118 | 0.995 |
| 5650.5 | 1.7816 | 2.1324 | 1.8866 | 0.944 |
| 5950.5 | 1.8015 | 2.0401 | 1.9072 | 0.945 |

The log-coordinate formula lands within 6% at three of four states; the
outlier is the one nearest the barrier, where gamma varies fastest across
the cell. Measured with a finite bump the observed sawtooth is clipped by
roughly `1 - bump/cell`, which is why the same sweep at a 0.05% bump reads
about 0.70 of the prediction.

This explains every observation in the gate record:

- **it does not improve with more quadrature nodes.** `probe_grid.py` shows
  why the grid ladder measured nothing: `min_diffusion_stddev_cells=2.5`
  sets an adaptive floor, and for `ttm >= 0.6` that floor returns the SAME
  grid for `--quad-grid` 401, 801 and 1201. The failing dates are the
  long-dated ones.
- **it gets worse as the audit bump shrinks.** A narrower bump is more
  likely to sit inside one cell and return the chord slope, losing the
  curvature entirely.
- **it is confined to the pre-knock-in dates.** Once knocked in, the KI
  barrier is gone and the surface near the money is far smoother, so the
  same interpolation costs far less.

## It does not reach the hedge

`hedge_impact.py`. The replay sizes the hedge from `calculate_greeks`, a 1%
bump-and-reprice — **5.4 cells wide**, which averages the staircase away:

| | contracts |
|---|---:|
| position | 90.79 |
| sawtooth in the 1% delta, peak to peak | 0.0275 |
| worst deviation from the grid delta | 0.1446 |
| rounding granularity of a hedge trade | 1.0000 |

The worst error is seven times smaller than the smallest tradeable
increment, and hedge quantities round to whole contracts. The hedge never
sees it.

The audit saw it because decoupling the audit's spot bump took it BELOW one
cell, which is finer than the engine's own price grid.

## Candidate

`candidates.py` compares readouts on the same node values, measuring the
sub-cell finite-difference delta error against the engine's own
grid-gradient delta (`calculate_spot_greeks_curve`, which differentiates on
the grid and is staircase-free by construction).

| ttm | linear, ptp | cubic, ptp | price move |
|---:|---:|---:|---:|
| 0.85 | 0.0779 | 0.0007 | 0.033 bp |
| 0.60 | 0.2624 | 0.0058 | 0.155 bp |
| 0.42 | 0.1660 | 0.0106 | 0.112 bp |
| 0.30 | 0.2454 | 0.0165 | 0.205 bp |
| 0.15 | 1.7350 | 0.0240 | 2.250 bp |

`ptp` is the peak-to-peak sawtooth in reference hands; `price move` is the
largest relative price change against the current linear readout. The cubic
is a four-point centred Lagrange on the uniform log grid, so it costs
nothing measurable.

The 0.15 row keeps the KI barrier alive at a maturity where the real study
had already knocked in; it is a probe of the mechanism, not a state the
subset run visited.

**Disabling alignment is not the alternative.** With alignment off the
sawtooth is 18.0 ptp and prices move 14.9 bp: the barrier's own projection
error dominates, which is what alignment exists to prevent. Alignment is
correct; the readout simply stayed linear after alignment was introduced.

### There is no kink at t=0 — the transition already smoothed it

The obvious objection to a high-order readout is that a 4-point stencil
straddling the barrier node would span a kink. **It does not: by the time
the surface reaches t=0 there is no kink left.** The kink is created at an
observation date and then diffused away by the backward transition over the
next positive-variance interval.

`kink.py` inspects the retained t=0 surface directly. Through the barrier
node the first derivative rises smoothly (71.0M, 80.0M, 86.8M) and the
second peaks smoothly and declines; nothing is discontinuous. What is true
is that the THIRD derivative is a thousand times larger near the barrier:

| | mean `|d3V/dx3|` |
|---|---:|
| within 30 nodes of the barrier | 2.65e+10 |
| well above it | 2.17e+07 |

So the near-barrier regime is extreme but smooth curvature, not a kink.
That is exactly the regime where a linear interpolant's error explodes and a
high-order one stays bounded, and it is why the cubic does not overshoot —
it is interpolating a smooth function.

The caveat is narrow: a valuation date that IS an observation date, with the
event not yet applied, or a boundary splice, could still present a genuine
kink. Neither occurs here.

### The cubic helps most where the problem is worst

`near_barrier.py` measures it there:

| distance above the KI barrier | linear ptp | cubic ptp | linear mean | cubic mean |
|---|---:|---:|---:|---:|
| 0.1 – 2.1 cells | 5.2714 | **0.3428** | 1.2627 | 0.5758 |
| 3 – 5 cells | 1.1335 | **0.2951** | 0.2352 | 0.1777 |
| 12 – 14 cells (control) | 0.1992 | **0.0166** | 0.0429 | 0.0305 |

The staircase is an order of magnitude worse next to the barrier, where
gamma is largest, and the cubic improves consistency 15x precisely there.
It does not overshoot.

Read the mean column with care in the first two rows: the grid-gradient
delta used as the reference is itself taking `np.gradient` across a kink,
so part of what is charged to the cubic belongs to the reference. The `ptp`
column is the honest one — it measures whether neighbouring spots agree.

`interpolants.py` puts the obvious alternatives beside it, including PCHIP,
which cannot overshoot by construction:

| readout | ptp at the barrier | ptp in the control | mean at the barrier |
|---|---:|---:|---:|
| linear (current) | 5.2714 | 0.1992 | 1.2627 |
| 3-point Lagrange (quadratic) | 0.8812 | 0.0480 | 0.6008 |
| **4-point Lagrange (cubic)** | **0.3428** | **0.0166** | 0.5758 |
| PCHIP (shape preserving) | 0.8861 | 0.0402 | 0.5752 |

The cubic wins on consistency in both regimes. The three high-order options
agree on the mean to within 0.03, which is the signature of a shared offset
against the reference rather than a difference between them. Shape
preservation buys nothing here: there is no overshoot to prevent, and
PCHIP's limiter costs smoothness.

### The principled remedy: evaluate the final transition at spot

The engine does not have to interpolate at all. The last backward step is an
explicit smooth function of the readout coordinate:

```
V(x) = prefactor * scale * sum_j u_j * omega(x - x_j)  +  tail(x)
```

with `u_j` the already-weighted nodal values, `omega` the Gaussian kernel and
`tail` the closed-form `erfc` term. Evaluating that at `x = 0` is one
off-grid row of the operator the engine already applies — not a new
quadrature rule, and not interpolation.

**The product-agnostic core in the same module already works this way.**
`QuadratureCore._calculate_final_value` (`quad_core.py:591`) sums
`omega(0 - grid_j)` against the weighted values and adds closed-form boundary
slivers; it never interpolates. Only the bespoke snowball path FFT-diffuses
to the grid and then calls `np.interp`. So this is a restoration of
consistency, not a new method.

`transition_readout.py` implements it as a runtime patch and measures it
beside the interpolants:

| region | linear ptp | cubic ptp | transition-at-spot ptp |
|---|---:|---:|---:|
| 12–14 cells above the barrier | 0.1992 | 0.0166 | **0.0166** |
| 0.1–2.1 cells above | 5.2714 | 0.3428 | **0.3213** |

The cubic and the exact operator evaluation agree to four decimal places in
the control region and to within 6% at the barrier. That is worth stating
plainly: **the cubic is not an ad-hoc smoothing, it reproduces what the
principled method gives**, which is the strongest evidence available that
both are converging on the right answer rather than on a nicer-looking one.

Price moves are the same order either way: 0.205 bp in the control region,
2.801 bp within two cells of the barrier.

### Is the cubic more accurate, or only smoother? Both, depending on where

Smoothness alone would make this a matter of taste. `convergence.py` prices
the same state on a grid ladder under both readouts, in basis points of
notional against the finest grid.

Twelve cells above the barrier the two readouts are indistinguishable —
-0.6069 vs -0.5879 bp at the working grid, converging together. There the
readout is not the accuracy bottleneck, the scheme's own discretisation is,
and the cubic buys **smoothness only**.

Half a cell above the barrier they part company:

| n | linear err (bp) | cubic err (bp) |
|---:|---:|---:|
| 683 | 0.8292 | 0.0974 |
| 1001 | 0.2935 | 0.0216 |
| 1501 | 0.0619 | 0.0019 |
| 2001 | 0.0863 | -0.0006 |
| 3001 | 0.0261 | -0.0008 |
| 4001 | 0.0023 | -0.0003 |
| 5001 | 0.0115 | 0.0000 |

The cubic converges monotonically and is settled by n=1501. The linear does
not converge at all: it wanders 0.062 → 0.086 → 0.026 → 0.002 → 0.012 with
no trend, because refining the grid also moves the readout position relative
to the nodes. Its raw prices swing about 400 currency units across the top
of the ladder while the cubic's settle to within 2.

That non-monotonicity is visible in the linear sequence alone and does not
depend on the reference being the cubic's. **Near the barrier the linear
readout costs real accuracy — about 0.8 bp at the working grid — and no
amount of refinement removes it cleanly.** That is the regime a snowball
lives in and the regime the audit failed in.

### Putting spot on a node instead — measured, and it does not pay

The tempting alternative is to stop SHIFTING the grid onto the barrier and
instead STRETCH the envelope so the barrier lands on the lattice that
already contains spot: with an odd node count and no shift, log-moneyness 0
is the centre node exactly (verified: `linspace` gives 0.0, not 1e-17), so
the readout would need no interpolation at all. It only requires
`(n-1)*|log(B/S)| / (2*log_c)` to be an integer, and both `n` and the
envelope are free.

`option_d.py` shows it is constructible and cheap in envelope terms —
absorbing the residual moves `num_std_devs` from 10.0000 to at most 10.0012.
`option_d_sim.py` then runs it end to end, and it loses:

| variant | ptp | mean abs |
|---|---:|---:|
| current linear readout | 0.2454 | 0.0544 |
| cubic readout, alignment kept | 0.0165 | **0.0304** |
| spot+barrier on nodes, `n` fixed, envelope absorbs the residual | **0.0603** | 0.2923 |
| spot+barrier on nodes, `n` searched, envelope barely moves | 0.2243 | 0.1530 |
| alignment disabled | 18.0069 | 4.2206 |

Fixing `n` kills the sawtooth (ptp 0.060) but stretches the envelope by up
to `0.5/k` ≈ 3.8%, which coarsens the grid and biases the level. Searching
`n` keeps the envelope still but swings the node count from 691 to 1067
between neighbouring spots, so the scheme's own discretisation error becomes
the new noise.

The trade is real but small in the right direction: `grid_sensitivity.py`
measures the price step between adjacent grids at 8–22 currency units,
which if the grid changed once per cell would induce a delta jump of only
0.002–0.006 hands. The damage in the table comes from the envelope stretch
and the wide `n` swing, not from grid-dependence as such — so a more careful
selection rule might do better than these two. As measured, neither beats
the cubic readout.

## Decision matrix

None of these has been applied. Per the project's standing rule, a numerics
change goes to a decision before engine code moves.

| Option | Delta consistency | Accuracy | Price change | Blast radius |
|---|---|---|---|---|
| (a) leave as is; derive the audit budget from `h*|S*Gamma+Delta|` | staircase stays; sub-cell spot Greeks unreliable on this engine | ~0.8 bp lost near the barrier, non-convergent | none | audit only |
| (b) cubic readout as the default | sawtooth down 10–70x, and 15x right at the barrier | converges monotonically; settled by n=1501 near the barrier | 0.03–2.3 bp | every QUAD price; all QUAD goldens rebase |
| (c) cubic readout behind an opt-in param | same, where enabled | same, where enabled | none by default | new parameter; two paths to maintain |
| (d) put spot and the barrier both on nodes | measured worse than the cubic on mean error (0.15–0.29 vs 0.03) | not measured | large | grid-selection rule; cost varies with spot |
| **(e) evaluate the final transition at spot** | **matches the cubic: 0.0166 control, 0.3213 at the barrier** | inherits the scheme's own order; no interpolation at all | 0.205 bp control, 2.801 bp at the barrier | the snowball readout path; QUAD goldens rebase |

**Recommendation: (e), shipped as an explicit versioned readout mode with
the legacy default preserved — which is (c)'s release shape carrying (e)'s
method.** It is the only option that removes the error rather than bounding
it, it restores consistency with `QuadratureCore` instead of inventing a
rule, and it makes price and delta come from the same operator. Measured, it
delivers what the cubic delivers, so nothing is lost by preferring the
better justification.

(a) remains the right immediate move for the bucket-hedge audit, and it is
independent of the above: the budget should be derived rather than declared
whatever happens to the readout. It is not a fix, and it should not be
described as one.

Two things bear on the choice beyond the table.

The last row is cheap and honest: the sawtooth amplitude is not noise, it is
`h * |S*Gamma + Delta| / m_ref`, which the engine knows. A budget derived from it would
say what this engine can actually resolve, and needs no engine change. But
it accepts the ~0.8 bp of lost accuracy near the barrier rather than fixing
it.

Against that, the project's standing rule is to fix instability with
resolution or refinement rather than with smoothing — and refinement does
NOT fix this one. The linear readout's error near the barrier is
non-monotone in the grid size, so no amount of refinement converges it.
That is the argument for treating the readout as a defect rather than a
tolerance question.

## Review

Reviewed by `openai/gpt-6-astra` at `xhigh` reasoning via ZenMux, on the
evidence above. Its recommendation was option (e) — evaluate the final
transition operator at the requested spot, and take delta by differentiating
that same operator — shipped first as an explicit versioned readout mode
with the legacy default and its goldens preserved. It argued against
changing the shared `interpolate` utility to cubic as the primary fix, on
the grounds that the migration budget should be spent once, on the right
method.

Four of its points changed this document:

1. **The amplitude formula.** It caught that `(cell/2) * Gamma` treats the
   interpolant as linear in spot when it is linear in `log S`, and gave the
   correct leading order. Verified above: the corrected formula lands within
   6% at three of four states, the old one over-predicts by an inconsistent
   1.13x to 2.5x. It also noted that the next term is
   `[(m-x)^2/2 + h^2/24] * (Delta + 3*S*Gamma + S^2*V_SSS)`, and that my
   proposed explanation for the residual gap — endpoints not reaching the
   phase extrema — has the wrong sign, since under-sampling would reduce the
   measured range, not raise it. That gap is not certified.
2. **Where the kink lives.** It pointed out that a kink before a
   positive-variance backward transition is normally smoothed. Checked
   directly, and it is: see above. This removes the main objection to a
   high-order readout rather than answering it.
3. **The grid-gradient Greek path is not an unconditional substitute.** A
   centred stencil at a genuine kink averages the one-sided slopes rather
   than selecting a delta, and interpolating those derivatives spreads the
   averaging over neighbouring cells; the local error can stay O(1) while
   its region shrinks with `h`. It would sanction that path away from
   nonsmooth locations with convergence tests, not as a barrier-Greek
   standard. This also sharpens the caveat on using it as the reference
   here.
4. **Option (d)'s existence obstruction, withdrawn.** It began an argument
   that spot and barrier cannot generally share a uniform grid, then
   withdrew it once the envelope is adjustable: the construction measured
   above relaxes exactly the binding constraint. (d) is rejected on
   selection-rule stability, not on existence.

Its remaining challenges are recorded and not yet answered: separate readout
error from mesh-motion error, do not read ptp against the grid-gradient
delta as absolute accuracy, and core consistency is architectural evidence
rather than an accuracy proof — the bespoke path still needs its transition
conventions, boundary slivers and event ordering checked against the core's.

Where this document and the review differ: the cubic and the exact operator
evaluation were measured to agree to four decimal places in the control
region and within 6% at the barrier. That was not available when the review
was given, and it weakens its objection to (b) — the cubic is not an
arbitrary smoothing rule, it lands on what (e) computes. (e) remains the
better justification; (b) is now a measured approximation to it rather than
a different answer.
