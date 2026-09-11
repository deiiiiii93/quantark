# The QUAD snowball's spot derivative is a staircase

Root cause of the identity-residual failures recorded in `../gates.md`.

> **Shipped.** `QuadParams.readout` now selects the rule:
> `"legacy_linear"` (the default, unchanged) or `"transition"`. The scripts
> below were written before that and monkeypatch their candidates; they are
> kept as the evidence the decision rested on. Measured through the shipped
> parameter on the state below, the detrended sub-cell delta spread falls
> from 0.1890 to 0.0000 reference hands twelve cells from the barrier, and
> from 4.6297 to 0.0168 within two cells of it.

Run from this directory with the repo on the path:

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
  curvature entirely. This is a statement about the AUDIT's ladder, and it
  holds only because shrinking that bump unmatches it from the engine's own;
  see the end-to-end section below.
- **it is confined to the pre-knock-in dates.**

> The pre-knock-in confinement was attributed here to the interpolation:
> once knocked in the barrier is gone, the surface near the money is
> smoother, and the same readout costs less. That attribution is wrong. The
> confinement is identical under `transition`, which has no interpolation
> error at all — 0.000808 hands after knock-in under both readouts, agreeing
> to six decimals. So the pre-knock-in residual is a property of the live
> barrier, not of how the price is read off the surface, and it is the one
> thing in this document the readout does not explain. See
> `residual_by_ki_state.py` and `../gates.md`.

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

### What this readout does NOT fix: a near-barrier delta lobe

Reported by a peer session working from an independently validated
Gaussian-transition reference. Their bump-free measurement is the load
bearing one, because delta there comes from differentiating the transition
density analytically:

| engine | gap from the reference, study hands |
|---|---:|
| PDE, 1601 points / 16 steps per day | 0.38 and 0.58 |
| QUAD, 401 points (the study default) | 16.73, worst |
| QUAD, 3201 points | 13.03, still |

PDE converges to the reference near the barrier and QUAD does not, on
evidence that cannot be a bump artefact. So the lobe is QUAD-specific.

> **My own QUAD-minus-PDE reproduction is retracted as a magnitude.** It
> reported +12.4 / −8.3 / −25.1 / −8.5 / −1.6 study hands at −1.50% /
> −0.50% / +0.84% / +2.00% / +5.00% versus the barrier, and I presented it
> as independent confirmation. It ran on this worktree's PDE, which predates
> `25d4f7d6` — a fix for three separate near-barrier readout defects in the
> PDE: a stencil snapped to the nearest node, a snowball greeks path reading
> a different vector from the one `price()` reads, and a life surface
> returning event-projected columns instead of branch columns. So the
> numbers are QUAD minus a PDE carrying its own near-barrier delta error,
> and the −25.1 worst case against the peer's 16.73 is consistent with that
> contamination. The lobe is real and QUAD-specific on the peer's evidence;
> the shape and magnitudes above are not to be quoted. Re-measuring needs
> this branch merged up to its base, which is pending.
>
> **No longer pending, 2026-09-11.** The branch is merged up; `25d4f7d6` is
> an ancestor and the corrected PDE is available here. The retraction
> stands — these magnitudes are still not to be quoted — but re-measuring is
> now work rather than a blocker. Note the merge also inherits 24 failing
> tests from the base, three of which are replay goldens moved by this very
> repair, and nothing was re-banked. See `../gates.md`.

**Both readouts show the lobe identically**, so it is in the surface, not
the readout. That comparison is QUAD against QUAD and is unaffected by the
PDE defect.

**My detrended metric cannot see this, by construction.** It fits and
removes a trend across one cell, so an error shared by every cell is removed
along with the trend. It measures the staircase and is silent on any smooth
bias. The convergence section below likewise measures PRICE, and near a
barrier the third derivative is a thousand times larger, so price
convergence says little about delta there. Both are real measurements of
what they measure and neither is evidence about near-barrier delta accuracy.

Two mechanisms ruled out, both against the pre-fix PDE and so worth only
what their internal comparison is worth:

- **event smoothing is not the cause.** `event_smoothing_cells=0` is
  bit-identical to the default 1, because that parameter is inert under the
  default `event_projection=CELL_AVERAGE`; only the NODAL path reads it.
  This one is a QUAD-against-QUAD bit comparison and survives the retraction
  intact.
- **NODAL projection is worse**, not better: −32.8 against CELL_AVERAGE's
  −25.1. Both legs share the same contaminated baseline, so the ordering
  stands while the two numbers do not.

The apparent halving of the gap when refining 401 to 1601 is not evidence of
convergence toward truth. The peer's ladder against a validated reference
shows 16.73 hands at 401 barely moving to 13.03 at 3201, which is the
statement to trust.

This is unresolved and is the dominant near-barrier delta error. The readout
claims that stand are narrow: it removes the staircase, and at a node it
reproduces the engine's own diffusion to 1e-14.

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
readout costs real PRICE accuracy — about 0.8 bp at the working grid — and
no amount of refinement removes it cleanly.**

Read that as a statement about the price and nothing more. It is not
evidence that the transition readout makes near-barrier DELTA right; the
section above shows a delta lobe an order of magnitude larger that both
readouts share.

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

**Chosen and shipped: (e), as an explicit versioned readout mode with the
legacy default preserved — (c)'s release shape carrying (e)'s method.** It
is the only option that removes the error rather than bounding it, it
restores consistency with `QuadratureCore` instead of inventing a rule, and
it makes price and delta come from the same operator.

(a) remains the right immediate move for the bucket-hedge audit, and it is
independent of the above: the budget should be derived rather than declared
whatever happens to the readout. It is not a fix, and it should not be
described as one.

## What shipped

`QuadParams.readout`, one of `QUAD_READOUT_MODES`. Default
`"legacy_linear"`; no existing price or golden moves.

Under `"transition"` the final backward transition is evaluated at the spot.
Three properties are worth recording:

**It reproduces the engine's own diffusion exactly.** At a grid node, where
the legacy interpolation is exact, the two agree to between 6e-15 and 2e-14
in all four combinations of spectral filter and FFT padding. That is the
correctness check that matters: the readout is the same operator, not a
better-behaved substitute for it.

**The spectral filter had to move, not be dropped.** `convolution_fft`
multiplies the transforms of the kernel, the weighted values and the filter,
so the filter can be carried by either factor. Carrying it on the kernel
leaves nothing analytic to evaluate off-lattice. `QuadratureMath
.filtered_weights` moves it onto the values instead, which is exact and
leaves the Gaussian kernel in closed form. Skipping the filter instead would
have left a 2.4e-7 discrepancy and a final step inconsistent with every step
before it.

**Everything the mode does not cover refuses it.** Continuous knock-in
monitoring takes the bridged transition, which has no pointwise form here,
and raises. `PhoenixQuadEngine` and `KOResetSnowballQuadEngine` read their
prices off their own surfaces and narrow `supported_readouts` to the legacy
rule, so constructing them with `readout="transition"` raises rather than
returning a legacy-readout price under another name.

The mode governs the event decomposition as well as the price, since the two
are reported beside each other and read off the same surfaces.

### Selecting it: which paths actually reach the engine

A parameter is only real where something threads it. Three paths, and one of
them was silently broken:

- **The fleet** (`02_backtest_fleet.py --quad-readout`) reaches both the
  replay's engine config and the fair-coupon solve, so a run prices one rule
  throughout. Verified live: the solved coupon moves 4.5276% to 4.5272% and
  the audit residuals move with it.
- **Study stages 01 and 04** construct `QuadParams(grid_points=...)` from
  defaults and take no readout argument, so they are always on the shipped
  default. Consistent, but they cannot exercise the mode.
- **The model-validation studies could NOT reach it, while appearing to.**
  A peer session found that the three quad candidates spread their YAML keys
  into `params()`, so `readout: transition` moved the identity hash, but
  `_greeks` built `QuadParams(grid_points=grid_points)` from defaults and
  priced the legacy path regardless. A study naming the mode would have
  banked a certificate claiming to cover it while measuring something else.
  Fixed on `feat/simulated-path-backtest`: a shared `_engine_params` builds
  the params both `params()` records and `_greeks` prices with, and a
  non-default readout takes its own candidate name so two candidates cannot
  overwrite each other's recorded decision.

The general shape of that bug is worth remembering: a config key that
reaches the RECORD but not the RUN is worse than one that reaches neither,
because it buys false confidence.

### End to end on the study: it fixes the defect, and the carry audit was never measuring that defect

The 14-cell subset was re-run entirely under `--quad-readout transition`
(`bucket_hedge_v2/subset_transition`). No cell's verdict changes. On
`term_flat_q__front` the two runs agree to the leg row — 66 pass, 175 fail,
1 not measured, 1 inconclusive — and the mean identity residual moves only
from 0.042500 to 0.042613 reference hands.

That is not the readout failing. The audit is blind to the staircase at its
own settings, by construction.

Its residual is `(delta_q - delta_f_direct - sum_i (F_i/S) B_i) / m_ref`.
Both delta terms are central differences of the same price surface at the
same spot, and at the default their steps are the same size, because
`audit_spot_bump_rel` resolves from the pricing bump. Any error that depends
only on the readout point and the step therefore enters both terms and
subtracts out. Measured over the same 243 dates, as a mean absolute change
in reference hands (`identity_ingredients.py`):

| what changes | `delta_f_derived` | `delta_f_direct` | identity residual |
|---|---:|---:|---:|
| readout, at the matched default bump | 0.006814 | 0.006503 | 0.000787 |
| bump 0.01 to 0.001, readout fixed | 0.000000 | 0.014838 | 0.014838 |

The first row is the cancellation. The readout moves both sides by nearly
the same amount, so the residual barely moves and no verdict flips.

The second row is why the bump ladder in `gates.md` ever showed anything.
`delta_f_derived` is EXACTLY invariant to the audit bump, to every digit,
because `delta_q` is the engine's own delta Greek at the engine's own
`BumpConfig` bump (`base_engine.py:245`) and never the audit's. Only
`delta_f_direct` follows `audit_spot_bump_rel`. Shrinking the audit bump
unmatches the two steps and exposes the staircase on one side alone.

The ladder under both readouts (`bump_ladder_table.py`):

| audit spot bump | legacy | transition |
|---:|---:|---:|
| 0.01 (default, matched) | 0.042500 | 0.042613 |
| 0.0025 | 0.051449 | 0.043244 |
| 0.001 | 0.084353 | 0.046811 |

So the mode does what it was built to do. It removes the bump-dependence of
the frozen-carry spot derivative: at the mismatched bump the residual falls
44% and the passing dates rise from 29 to 65. It changes nothing where the
bumps match.

**The ladder has to be read the other way round from how I read it.** I
built it to diagnose why the audit fails and treated its growth as evidence
about that failure. The growth was real, and following it did find a real
engine defect, now fixed and shipped. But the growth was an artefact of the
probe unmatching two bumps that the audit deliberately matches. At its own
settings the audit was never failing on the staircase, and my own earlier
`hedge_impact.py` measurement — 0.028 contracts peak to peak at the 1% bump
— already said so before I built the ladder.

What is left is a floor near 0.0425 hands, invariant to the readout and to
the spot bump, and it is **entirely the live knock-in barrier**
(`residual_by_ki_state.py`):

| | dates | mean abs | median | max |
|---|---:|---:|---:|---:|
| knock-in barrier live | 178 | 0.057257 | 0.044636 | 0.548283 |
| after knock-in | 63 | 0.000808 | 0.000578 | 0.002909 |

A 71x collapse on the same product, engine, curve and bumps; under
`transition` the same split gives 0.057410 and 0.000808. Once the barrier is
extinguished the identity holds an order of magnitude inside the budget on
63 consecutive dates, so the chain rule, the audit machinery, the bucket
deltas and the engine's delta Greek are all sound on this evidence. The
monthly knock-OUT barriers are live in both legs, so it is the
daily-monitored knock-IN barrier specifically — the same object as the
near-barrier lobe above, which makes it one open defect rather than two.

Two of my own hypotheses died on the way, and a third method failed:

- **the uncovered tail** — the identity sums over LISTED contracts and this
  product outlives the strip. Refuted by `tail_hypothesis.py`: correlation
  +0.16, and dates where the curve SPANS the product carry a higher mean
  residual than dates where it does not.
- **the bucket bump** — `B_i` is taken at a fixed 1 index point whatever the
  spot bump, which would reproduce the bump-invariance. Refuted by the
  knock-in split: a fixed bucket bump cannot switch off at knock-in. Its
  confirming 10-point run never completed, raising `dividend yield magnitude
  must be <= 1.0` for the reason Gate A already records.
- **`identity_vs_pde.py`**, which strips the identity to three price bumps
  through one flat yield to compare QUAD against PDE. Its own bump control
  condemns it: the residual there scales ~100x from a 1% to a 0.1% bump,
  which is O(h^2) truncation, while the study's is bump-invariant over the
  same three. Different error structure, different quantity. Kept with the
  control that condemns it so the construction is not tried again.

### It moves the model-validation certificate identities, and that is correct

`equity.snowball.quad` records "every numerically relevant knob, including
the ones taken from defaults: a default that changes in a later release is a
numerics change, and the identity hash has to notice"
(`builders/equity_snowball.py`). `readout` is such a knob, so adding the
field moves the hash and `test_banked_cells_keep_their_identity` fails for
the snowball certificates.

It must NOT be added to `_QUAD_NON_NUMERIC`. That list is for knobs which do
not move the certified numbers — `event_stats_mode` qualifies because its
npv is identical — and `readout` plainly does move them.

No priced number changed. The default reproduces the old behaviour and the
golden suites confirm it. What moved is the serialisation of the config
space, so the resolution is an amendment to the banked evidence, not a
regeneration, and that is a decision about banked certificates rather than
part of this change.

**Nothing was stale before.** Counting cells with `readout` excluded, which
is exactly the state before this change, all 678 match:

| candidate | cells matching their banked hash |
|---|---|
| `equity.snowball.pde` | 117 / 117 |
| `equity.phoenix.pde` | 123 / 123 |
| `equity.ko_reset_snowball.pde` | 99 / 99 |
| `equity.snowball.quad` | 117 / 117 |
| `equity.phoenix.quad` | 123 / 123 |
| `equity.ko_reset_snowball.quad` | 99 / 99 |

Adding the field moves exactly the 339 QUAD cells and leaves the 339 PDE
cells alone. The whole of it is this change's to answer.

> An earlier version of this section reported 222 cells as already stale.
> That was wrong, and the error was in the probe: `equity_ko_reset` and
> `equity_phoenix` do `from ...equity_snowball import _QUAD_NON_NUMERIC`,
> which binds the NAME at import time, so rebinding it in `equity_snowball`
> alone never reached them. They kept `readout` in their identity and
> mismatched, and I read that as pre-existing staleness. Caught by a peer
> session that audited the same question against the main checkout and got
> 726/726. `cert_probe.py` now patches all three builders.

Two things bear on the choice beyond the table.

The last row is cheap and honest: the sawtooth amplitude is not noise, it is
`h * |S*Gamma + Delta| / m_ref`, which the engine knows. A budget derived from it would
say what this engine can actually resolve, and needs no engine change. But
it accepts the ~0.8 bp of lost accuracy near the barrier rather than fixing
it.

> **That row would not have fixed the audit either**, which the decision
> matrix did not know when it was written. The audit's two spot derivatives
> use matched steps, so the sawtooth enters both and cancels; a tolerance
> sized to it would not admit one extra date. The amplitude formula is the
> right budget for a sub-cell delta probe, a different instrument. The
> matrix's "audit only" column for this row should be read as "neither".

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
