# Continuous-KI bridge refinement and SDE step cap (MC snowball / phoenix)

2026-10-29/30. Opt-in settings `ki_bridge_nodes_per_day` and `max_sde_step_days`; defaults
unchanged bit for bit. All errors are in bp of notional (the initial price), MC minus reference.

## 1. The question

The idea under test was to floor the time steps of the local-vol MC engines to help short-dated
positions. Measured on the AH local vol of the 2024-02-08 CSI1000 fixture, a uniform floor did
remove the gross one-step error of few-node grids, but refining the SDE made continuous-KI
contracts converge *away* from the PDE (Dupire, near-KI 2w snowball: +1.9 → +13.2 bp as the
substeps grow; `evidence/confirm_demo.json`). The gap was diagnosed before anything shipped.

## 2. Root cause

Controls with ~240–500 fine steps (`evidence/diagnose_gap.json`, 4 seeds × 2^18 antithetic):

| control | flat 25% | AH | Dupire |
|---|---|---|---|
| European puts 14d / 30d | ±0.03 | within ±0.3 (noise) | within ±0.5 (noise) |
| case B: KI at maturity only | +0.3 ± 0.5 | +1.4 ± 0.8 | +0.9 ± 1.0 |
| A1, DAILY DISCRETE KI | +0.1 ± 0.6 | +0.5 ± 0.8 | −0.6 ± 0.8 |
| **A1, CONTINUOUS KI** | +0.7 ± 0.6 | **+5.1 ± 0.7** | **+12.4 ± 0.6** |

Payoffs, conventions and the surface read-out agree (flat surface vs BSM engines and Black–Scholes;
puts on both skewed surfaces). The gap lives only in continuous knock-in on a skewed local vol.

Which side is wrong (`evidence/diagnose_ki.json`, 8 seeds × 2^17): MC and PDE agree on DENSE
discrete KI (240 dates: AH +36.3 PDE / +36.9 MC; 960 dates: +18.7 / +18.2), and bridge-free MC
— discrete KI at 240/480/960/1920 dates extrapolated linearly in √dt (Broadie–Glasserman–Kou) —
lands on the PDE: −0.9 (AH), +1.3 (Dupire), −0.2 (flat). The PDE is right; the MC bridge is biased.

Mechanism: `_check_ki_barriers_continuous_with_bridge` runs on the CONTRACTUAL nodes (one per
business day), with the path's own integrated variance over the day. Under a spot-dependent or
stochastic vol the bridge law is wrong inside the interval, and `substeps_per_interval` refines
only the SDE (the fine variance is folded back onto the daily interval), so the bias does not
shrink. At the default the Euler error of daily steps partly cancels it, which is why the default
looked better than refined runs.

## 3. Fix hypotheses (`evidence/fix_hypotheses.json`, A1, 8 seeds × 2^17)

| arm | Dupire | AH | flat |
|---|---|---|---|
| today's default | +3.9 | −6.9 | +0.4 |
| daily bridge, 24 SDE substeps | +11.1 | +4.7 | +0.4 |
| H1 bridge nodes ×4 per day | +1.6 | +6.6 | +0.6 |
| **H1 bridge nodes ×16** | **+1.1** | **+0.8** | +0.5 |
| H1 bridge nodes ×64 | +1.5 | +0.4 | +0.2 |
| H2 barrier-local variance σ_LV(B)², daily | −24.9 | −15.6 | +0.4 |
| H1 ×16 + H2 | −0.7 | +0.5 | +0.5 |

H1 (shrink the bridge interval) converges; H2 (barrier-local variance) over-corrects by 16–25 bp
and is rejected. ×4 is not enough on AH because the SDE is still coarse over the fast first segment.

## 4. Scope (`evidence/scope_h1.json`)

| family, near-KI 2w unless noted | today | H1 ×16 | cost of ×16 |
|---|---|---|---|
| LV phoenix, AH | −6.4 | +1.1 | 6.1× |
| LV phoenix, Dupire | +2.3 | +1.7 | 5.9× |
| Heston snowball (vs PDE; bridge-free ref +1.5) | +3.8 | +1.6 | 8.3× |
| SLV snowball (vs PDE; bridge-free ref +1.2) | +5.4 | +0.1 | 6.5× |
| LV snowball 1y fresh, AH (±2.2) | −1.0 | +0.8 | 16.9× |
| LV snowball 1y fresh, Dupire (±2.2) | −5.7 | +0.8 | 15.9× |

The LV phoenix on Dupire sits near +2 bp at every density (+2.3 / +3.0 / +1.7 / +2.5 at
×1/4/16/64): that offset is not bridge bias and stays open. The Heston/SLV 2D ADI references move
by ±1–2 bp between rungs, so their gates use the bridge-free reference instead (`reference.py`).

Rejected along the way (`evidence/confirm_demo.json`): a front-loaded fine window is non-monotone
in its parameters (AH A1: −0.96 / −0.56 / −4.17 at 32/64/128 steps in 10 days); a uniform floor
on total steps refines the SDE only. `time_sampling="integrated"` was no substitute on the sampled AH surface
(`evidence/kernel_time_sampling.json`, 8d puts): further off at coarse steps (−18.6 vs −6.1 vol
points at one step, 0.85 moneyness) and 62× (one step) to 2.5× (128 steps) slower.

## 5. What shipped

- `ki_bridge_nodes_per_day: int = 1` on `SnowballMCEngine` and `PhoenixMCEngine` (every subclass;
  KO-reset knock-in horizon included): the continuous-KI node count becomes
  `int(bus_days_in_year * n * horizon) + 1`. The nodes are contractual, so the SDE steps on them and
  the bridge runs between them. At n = 1 the historical expression runs unchanged.
- `max_sde_step_days: Optional[float] = None` on the LV / Heston / SLV snowball and phoenix engines:
  interval i is split into `max(substeps_per_interval, ceil(dt_i / cap))` steps; nodes, payoffs and
  the bridge variance (summed per interval) stay contractual. PSEUDO and the plain estimator only;
  QUASI / RANDOMIZED_QUASI, every RQMC conditioning flag, `one_step_survival` and
  `richardson_pair_price` refuse a cap (they assume a uniform substep factor).
- Opt-in labels `+kib:{n}` / `+cap:{days}` on the RQMC scheme label; default labels unchanged.

Acceptance (`test/test_asset/equity/engine_mc/test_ki_bridge_refinement_accuracy.py`, seeded):

| gate | today | refined |
|---|---|---|
| LV snowball A1, AH / Dupire, n = 16 vs PDE | −6.9 / +3.9 | +0.8 / +1.1 |
| LV phoenix, AH, n = 16 vs PDE | −6.4 | +1.1 |
| Heston / SLV A1, n = 16 vs bridge-free (112.6091 ± 0.0106 / 112.3610 ± 0.0108) | +2.4 / +4.2 | +0.2 / −1.1 |
| LV case B (1m, KI at maturity), AH: default / cap 1 / cap 0.1 | +105.2 | −2.0 / +0.1 |
| LV case B, Dupire | +24.2 | −3.0 / −0.1 |

Bitwise: `bitwise_check.py` prices an 18-cell default matrix (BSM / LV / Heston QE, Euler-log,
QE-M / SLV Euler and QE; snowball, discrete-KI snowball, KO-reset, phoenix; one QUASI and one RQMC
run). Base `88a44e81` and the branch give identical floats on ARM64 (`bitwise_base.json`); the unit
test pins them with the cross-architecture golden tolerance.

## 6. Open items

- LV phoenix on Dupire: a ~+2 bp offset flat in the node density (not bridge bias), unexplained.
- Cost: ×16 nodes is 6–8× on short trades and ~16× on one-year trades. Long trades may need less;
  not measured below ±2 bp noise.
- QMC with large n on long trades can exceed the Sobol 21201-dimension limit (existing error).
- Any edit under `quantark/asset/equity/engine/mc` changes the intraday MC certification's
  implementation fingerprint (MC_TREES); banked intraday references go stale at the next amend.

## Reproduction

The `evidence/` scripts are the standalone runs (no repo edits; runtime patches through engine
subclasses and an environment proxy). They import `test/volmodels/test_andreasen_huge_repricing.py`
helpers from an absolute main-checkout path; run them from the repo root with the venv.
`reference.py` rebuilds `references.json`:
`env PYTHONPATH=<tree>:<tree>/test python docs/research/mc_ki_bridge_refinement/reference.py out.json`.
