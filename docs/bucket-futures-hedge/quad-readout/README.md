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
which is the true delta at the cell MIDPOINT. Its error is
`gamma * (S_mid - S)` — a sawtooth of amplitude `(cell/2) * gamma`, zero at
midpoints, worst at the nodes. `repro.py --n 201 --half-width-rel 0.004`
shows the staircase directly; the risers are visible as a 0.285-hand drop
across 0.1 index points at spot 5319.87, with a perfectly smooth run either
side.

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

### Does the cubic overshoot at the barrier? No — it helps most there

The obvious objection to a high-order readout is the kink: alignment puts a
node exactly ON the KI barrier, and a 4-point stencil straddling that node
spans the kink, which is where a Lagrange interpolant is supposed to
misbehave. `near_barrier.py` measures it there:

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
| leave as is | staircase stays; sub-cell spot Greeks unreliable on this engine | ~0.8 bp lost near the barrier, non-convergent | none | none |
| cubic readout as the default | sawtooth down 10–70x, and 15x right at the barrier | converges monotonically; settled by n=1501 near the barrier | 0.03–2.3 bp | every QUAD price; all QUAD goldens rebase |
| cubic readout behind an opt-in param | same, where enabled | same, where enabled | none by default | new parameter; two paths to maintain |
| put spot and the barrier both on nodes | measured worse than the cubic on mean error (0.15–0.29 vs 0.03) | not measured | large | grid-selection rule; cost varies with spot |
| leave the engine, derive the audit budget from the cell width | unchanged | unchanged | none | audit only |

Two things bear on the choice beyond the table.

The last row is cheap and honest: the sawtooth amplitude is not noise, it is
`(cell/2) * gamma`, which the engine knows. A budget derived from it would
say what this engine can actually resolve, and needs no engine change. But
it accepts the ~0.8 bp of lost accuracy near the barrier rather than fixing
it.

Against that, the project's standing rule is to fix instability with
resolution or refinement rather than with smoothing — and refinement does
NOT fix this one. The linear readout's error near the barrier is
non-monotone in the grid size, so no amount of refinement converges it.
That is the argument for treating the readout as a defect rather than a
tolerance question.
