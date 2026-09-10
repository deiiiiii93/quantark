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

## Decision matrix

None of these has been applied. Per the project's standing rule, a numerics
change goes to a decision before engine code moves.

| Option | Delta consistency | Price change | Blast radius |
|---|---|---|---|
| leave as is | staircase stays; sub-cell spot Greeks stay unreliable on this engine | none | none |
| cubic readout in `QuadratureMath.interpolate` | sawtooth down 10–70x | 0.03–2.3 bp of price | every QUAD price in the library; all QUAD goldens rebase |
| cubic readout behind an opt-in param | same, where enabled | none by default | new parameter; two paths to maintain |
| leave the engine, derive the audit budget from the cell width | no change to the engine | none | audit only; the budget becomes `(cell/2) * gamma`, a measured quantity rather than a declared one |

The last row is worth noting: the sawtooth amplitude is not noise, it is
`(cell/2) * gamma`, which the engine knows. An audit budget derived from it
would be honest about what this engine can resolve, and would not need the
engine to change.
