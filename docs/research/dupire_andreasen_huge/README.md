# Dupire repricing gap and the opt-in Andreasen–Huge construction

Code: `quantark/volmodels/localvol/andreasen_huge.py`. Tests: `test/volmodels/test_andreasen_huge*.py`.
Data next to this note:
- `t0_pde_ladder.json`: all 198 fixture puts on the production PDE ladder.
- `production_numbers.json`: AC2, AC7 and AC8 figures from the shipped module.

## 1. The problem

`build_dupire_local_vol` does not reprice its own input. On the 2024-02-08 CSI1000 fixture (r = 2 %, q parity-implied at the longest pillar), the production `LocalVolPDESolver` (3200 × 32) prices the quoted puts at these gaps to the input implied vols:

| Expiry | 8d | 36d | 71d | 134d | 225d | 316d |
|---|---|---|---|---|---|---|
| Mean gap, vol pts | +19.3 | +8.7 | +5.0 | +5.1 | +2.2 | +1.5 |
| Max gap, vol pts | 23.9 | 11.4 | 6.8 | 5.3 | 2.6 | 1.8 |

Every 4th strike was priced. The case-study QUAD gives the same gaps on all 198 quotes, within 0.2 vol pts. Every local-vol PDE, MC and QUAD price that uses that builder inherits the gap. The earlier `dupire_gap` table in `localvol_transition_quad` skipped the 8d expiry, which is the worst one.

**Root cause.** The builder computes local vol pointwise at the six quoted maturities only. Two things then go wrong.

1. **First segment.** Row 0's ∂w/∂T is a three-pillar quadratic through (8d, 36d, 71d). It ignores w(0) = 0, so it picks up the 36d hump of this term structure (ATM iv 0.284 → 0.429 → 0.387). That value (ATM 0.484, against a 0.284 implied vol) is then clamped flat back to t = 0.
2. **Between pillars.** `LocalVolSurface` interpolates linearly in vol. Nothing makes the variance delivered over a segment equal that segment's forward implied variance; for example, 71–134d delivers about 0.08 against 0.06.

Measured at fixed ATM spot, the surface's variance on [0, T] gives +20.0 / +3.1 / +2.5 / +3.3 / +1.1 / +0.9 vol pts over the input implied vol.

Flat-IV round trips cannot see this: with a flat term structure every interpolation scheme agrees.

## 2. Literature

| Method | First segment [0, T1] | Continuous-time engines reprice? |
|---|---|---|
| Andreasen & Huge (2011), *Volatility interpolation*, Risk; QuantLib `AndreasenHugeVolatilityInterpl` | One implicit forward-Dupire step from the payoff, like every segment | Yes, if they read Dupire on the AH price surface (QuantLib's `AndreasenHugeLocalVolAdapter`), not θ |
| Le Floc'h, *Don't stay flat* (blog); arXiv:2004.08650, 2305.13791 | — | Piecewise-constant θ explodes on fine grids, so use piecewise linear. Near-arbitrage quotes fit only approximately. Densities can be spiky. |
| Lipton & Sepp (2011), *Filling the gaps*; Itkin & Lipton (2018), *Filling the gaps smoothly* | Local variance constant in time on [0, T1], solved exactly in continuous time by Laplace transform; heat-kernel asymptotics (Gatheral et al. 2012) for short T | Yes; θ is the local vol |

AH with m implicit sub-steps per segment tends to the Lipton–Sepp model as m → ∞.

## 3. Standalone demo: decision matrix

The demo was runtime-patched, with no `quantark/` edits. Gaps are in vol pts on the production PDE, every 4th strike; the PDE grid is stated per row.

| First-segment sub-steps | Rows / segment | Cells / gap | PDE | 8d mean / max | 36d max |
|---|---|---|---|---|---|
| 1 (pure AH) | 16 | 4 | 3200 × 32 | −0.076 / 0.237 | 0.026 |
| 1 | 256 | 4 | 3200 × 32 | −0.003 / 0.136 | 0.007 |
| 1 | 64 | **8** | 3200 × 32 | **−0.003 / 0.043** | **0.009** |
| 1 | 256 | 16 | 3200 × 32 | −0.018 / 0.030 | 0.013 |
| 64 | 16 | 4 | 3200 × 32 | +0.311 / 0.906 | 0.011 |
| 64 | 256 | 4 | 1600 × 8 | +0.085 / 0.310 | 0.030 |
| 256 | 16 | 4 | 3200 × 32 | +0.068 / 0.223 | 0.019 |
| 256 | 16 | 8 / 16 | 3200 × 32 | +0.087 / 0.247, +0.088 / 0.248 (plateau) | 0.020 / 0.021 |

**Adopted: pure AH** (one step per expiry), with 64 rows per segment and 8 cells per gap.

Sub-stepping toward the Lipton–Sepp limit was measured worse. Each sub-step starts its own short-lived early-time wing spike, and the sampled rows alias them.

The 8d wings carry local vols above 1 in either model. The quoted 8d smile, 0.28 ATM to 0.53 in the wings, needs that; it is data, not an artefact.

Three findings shaped the design:

- **Call-only reading collapses the left tail.** Local vol read from deep in-the-money calls fell below 0.1 beyond y ≈ −1.4 (minimum 0.03), because the Dupire numerator 2(c_t + μ c_y) cancels an O(1) intrinsic drift. Reading each side from its out-of-the-money option (puts below F_t, calls above; QuantLib CallPut) removes this.
- **Unidentifiable quotes in thin-tailed smiles.** A flat 0.25 surface with ±7 sd 8d quotes cannot be fitted at any sub-step count: the wing quotes have vega ≈ 1e-10. With the opt-in |d| ≤ 4 quote rule it is exact (≤ 3e-12), and θ tends to 0.25 as sub-steps grow. On the fixture that rule excludes nothing.
- **Forward bow.** Interpolated-zero curves bow ln F between expiries. The engine-facing surface is therefore keyed through forward moneyness, σ(t, S) = σ_AH(t, S·F^AH_t / F^env_t).

## 4. T0: all 198 fixture puts on the PDE ladder (defaults 8 cells, 64 rows)

| PDE grid | 8d | 36d | 71d | 134d | 225d | 316d | Worst |
|---|---|---|---|---|---|---|---|
| 1600 × 8 | 0.098 | 0.026 | 0.019 | 0.011 | 0.007 | 0.005 | 0.098 |
| 3200 × 32 (105 s) | 0.043 | 0.010 | 0.002 | 0.005 | 0.003 | 0.002 | 0.0427 |
| 6400 × 64 (415 s) | 0.040 | 0.006 | 0.009 | 0.001 | 0.002 | 0.002 | 0.0398 |

Each cell is the max |gap| in vol pts.

- The largest change from 3200 × 32 to 6400 × 64 is 0.0074 (71d).
- Quote by quote, |gap| plus its refinement change is at most 0.0456.
- AC2 therefore runs at 3200 × 32 against the 0.05 bar.

## 5. Shipped module (production numbers)

**Calibration (AC1).** Every quote on both sides is repriced to ≤ 3.2e-12 in implied vol. The build, calibration plus 390 × 493 sampled surface, takes 0.11 s.

**PDE repricing (AC2), all 198 puts at 3200 × 32:**

| Expiry | Max \|gap\| | Mean gap |
|---|---|---|
| 8d | 0.0427 | +0.0000 |
| 36d | 0.0096 | −0.0079 |
| 71d | 0.0015 | −0.0003 |
| 134d | 0.0045 | −0.0036 |
| 225d | 0.0028 | −0.0008 |
| 316d | 0.0017 | +0.0006 |

These are identical to the demo at T0.

**Sampled local vol (AC7).** Values range from 0.198 to 1.67; the maximum is in the 8d wings. A −0.01 parallel local-vol bump stays admissible.

**Market vega (AC8).** For the 134d K = 5000 put, AH market vega (quotes ±0.01, recalibrated, priced on `LocalVolPDESolver`) gives a half-difference of 11.79253. The Black–Scholes half-difference at the quoted vol is 11.79193, a relative difference of 5e-5. AH reprices the bumped quotes too, so its market vega *is* the quoted vega.

**Fill rule.** `unreadable` reached 386 of 491 interior nodes on the side read. This is at the earliest sampled times, where the one-step density is concentrated near the forward, so most far-tail nodes take the flat fill. It does not affect any quote's price; see AC2.

**CP1 caught one bug.** The c_t solve counted the boundary column twice. On the fixture's short segments it moved values by about 1e-7; on a 2y → 5y segment it reached 1.3 vol pts. A unit test now checks the read-out against a finite difference of the partial step on that segment.

## 6. Conventions and limits

- **Beyond the last expiry**, local vol is flat in time at σ(T_last), matching `LocalVolSurface`'s clamp. QuantLib keeps stepping.
- **Unreadable nodes** (AH density below 1e-10 of its maximum, or σ² ≤ 0) are filled linearly in node index between readable nodes, and flat beyond them. The count is reported in `unreadable`.
- **The call/put seam at the forward.** The two sides are calibrated separately and differ at the forward by about 1e-7 (parity noise of the discrete schemes).
- **Exact by default.** A quote that is not fitted raises `NumericalError`. The |d| quote rule is opt-in and reported.
- **Not covered:** engines that build their own local vol keep Dupire (opt in by passing the AH `LocalVolSurface`); sub-stepped segments; regularisation; bucketed AH vega beyond `SurfaceBump`.

## 7. Reproduce

```bash
python -m pytest -n 4 test/volmodels/test_andreasen_huge.py test/volmodels/test_andreasen_huge_repricing.py test/volmodels/test_andreasen_huge_risk.py
```
