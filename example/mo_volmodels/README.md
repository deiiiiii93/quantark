# MO Options Vol-Model Calibration Suite

A hands-on, staged example that calibrates **Dupire Local Volatility**, **Heston**, and
**Heston Stochastic-Local Volatility (SLV)** to real **CSI 1000 index option (MO / 中证1000股指期权,
underlying `000852.SH`)** market data pulled via AKShare, and renders the study as both a
self-contained HTML lecture and an evidence-led calibration explainer.

The models are European index options — no early exercise — which is exactly the clean setting
Dupire / Heston / SLV assume.

## The interpreter split (important)

Two Python environments are involved and **no single interpreter has both** libraries:

| Stage | Interpreter | Why |
|-------|-------------|-----|
| `01` fetch | `/opt/anaconda3/bin/python` | has `akshare`; also hosts the network-boundary fetch scripts |
| `02`–`10` | `.venv/bin/python` | has `quantark`, no `akshare` |

The four `01_*` scripts are the only network boundaries: two AKShare fetchers
(`01_fetch_mo_snapshot.py`, `01_refresh_market_cache.py`) and two official CFFEX
HTTP fetchers (`01_fetch_mo_settlement_history.py`, `01_bulk_fetch_settlement_history.py`).
They write frozen JSON **snapshots** / CSV caches that later stages replay **offline**.
A committed synthetic `mo_snapshot_sample.json` (arbitrage-free by construction) drives
the automated tests and lets 02–10 run with no network at all.

## Running it

```bash
# 1) live fetch — AKShare interpreter (optional; the sample snapshot works offline)
/opt/anaconda3/bin/python example/mo_volmodels/01_fetch_mo_snapshot.py

# 2-17) replay and certify — quantark .venv
.venv/bin/python example/mo_volmodels/02_build_iv_surface.py  --snapshot latest
.venv/bin/python example/mo_volmodels/03_dupire_localvol.py   --tag latest
.venv/bin/python example/mo_volmodels/04_heston_calibration.py --tag latest \
  --bootstrap-reps 32 --bootstrap-seed 20260721
.venv/bin/python example/mo_volmodels/05_slv_calibration.py    --tag latest
.venv/bin/python example/mo_volmodels/07_barrier_exotic.py     --tag latest
.venv/bin/python example/mo_volmodels/08_snowball_exotic.py    --tag latest
.venv/bin/python example/mo_volmodels/09_delta_hedging.py      --tag latest
.venv/bin/python example/mo_volmodels/06_lecture.py           --tag latest
.venv/bin/python example/mo_volmodels/10_explainer.py         --tag latest

# Aggregate-only amendment of the immutable schema-11 ADI Greek certificate.
.venv/bin/python example/mo_volmodels/17_adi_slv_aggregate_certification.py \
  --parent-evidence output/adi_greek_certification_schema11/adi_greek_certification.json \
  --parent-decision output/adi_greek_certification_schema11/adi_greek_certification_decision.json \
  --resume \
  --output-dir output/adi_greek_certification
```

The schema-12 allocation was frozen before its held-out seeds were opened:
4,096 primary batches, 256 middle-control batches, and smooth-case Heston
coefficient 0.7. Its development-seed guarded interval was
`[-0.099272, -0.038637]` against the `±0.1` contract; production remains a
fixed-size run and never precision-stops.

The genuine cross-date study is a separate official-settlement cohort:

```bash
# Freeze CFFEX EOD settlement cross sections (or use --input-dir with downloaded CSVs).
/opt/anaconda3/bin/python example/mo_volmodels/01_fetch_mo_settlement_history.py \
  --dates 20260430 20260515 20260615 20260630 20260706 20260715 20260720

# Fit every admitted date under one frozen configuration and aggregate stability evidence.
.venv/bin/python example/mo_volmodels/10_calibration_diagnostics.py \
  --tags 20260430 20260515 20260615 20260630 20260706 20260715 20260720 \
  --output-tag latest

# Replay the frozen CSVs into a per-date SABR-smoothed IV-surface history
# (vol history for multi-year backtests; resumable, fail-closed, offline).
.venv/bin/python example/mo_volmodels/03_build_iv_surface_history.py --workers 4
```

Open the results:

- **`data/mo_volmodels_lecture_latest.html`** — long-form teaching material.
- **`data/mo_calibration_explainer_latest.html`** — decision-oriented calibration evidence,
  in the same visual/document system as the CFETS USD/CNY explainer.

Every artifact is keyed by a `--tag` (`latest` for live data, `sample` for the deterministic
fixture) so the test pipeline and the live pipeline never clobber each other's files.

## Stages

| Script | Does |
|--------|------|
| `01_fetch_mo_snapshot.py` | AKShare MO chain + CSI 1000 spot → `data/mo_snapshot_*.json` |
| `01_fetch_mo_settlement_history.py` | official CFFEX daily-statistics CSV → immutable settlement snapshots with source hashes |
| `01_refresh_market_cache.py` | AKShare refresh of the multi-year CSI 1000 spot + IM futures caches → `data/history/csi1000_spot.csv`, `data/history/im_futures.csv` (fail-closed, atomic writes) |
| `01_bulk_fetch_settlement_history.py` | bulk official CFFEX HTTP download of every trading day's raw settlement CSV → `data/history/settlement_csv/` + resumable fail-closed manifest |
| `02_build_iv_surface.py`  | **put-call parity** → r(T)/forward/carry; **OTM filter**; call-equivalent Black-IV inversion → `GridVolSurface` |
| `03_build_iv_surface_history.py` | replay every frozen settlement CSV offline → per-date SABR-smoothed IV surface + ATM pillars → `data/history/iv_surface/` + fail-closed `surface_manifest.json` (arb-validated via the Dupire input checks; gaps never filled) |
| `03_dupire_localvol.py`   | Dupire σ_LV(K,T); reprice via local-vol PDE; RMSE + surface plot |
| `04_heston_calibration.py`| calibrate (v0,κ,θ,σ,ρ); quote-only Jacobian/SVD; maturity-stratified multiplier bootstrap; Feller check; smile-fit plot + RMSE |
| `05_slv_calibration.py`   | Fokker-Planck leverage surface L(S,t); reprice via SLV PDE + plot |
| `07_barrier_exotic.py`    | up-and-out call priced **MC and PDE** under BSM/LV/Heston/SLV → model-divergence table + bar chart |
| `08_snowball_exotic.py`   | 2Y principal-excluded standard Snowball priced **MC and PDE** under BSM/LV/Heston QE/SLV plus standalone SLV QE MC → autocallable model-divergence table + bar chart |
| `09_delta_hedging.py`     | ATM European call delta-neutral hedge demo under BSM flat vol/LV/Heston/SLV → hedge inventory, turnover, residual PnL + chart |
| `06_lecture.py`           | weave everything into the HTML lecture + comparison CSV |
| `10_calibration_diagnostics.py` | normalized Heston fits across strictly comparable official CFFEX settlement dates → JSON/CSV/plot |
| `10_explainer.py`         | fail-closed, artifact-driven eight-section verdict with interactive raw-smile, tenor-error, and official-settlement stability explorers |
| `16_adi_greek_certification.py` | standalone Heston/Heston-SLV Snowball delta/gamma admission: deterministic anchors, separate ADI-axis ladders, paired QE-M RQMC uncertainty, resumable hash-locked cells, tri-state verdicts, and fail-closed routing |
| `17_adi_slv_aggregate_certification.py` | schema-12 aggregate-only SLV amendment: carries all 14 schema-11 PASS cells, adds fixed-size reference/control cohorts only, preserves paired fine/substep covariance through joint signed endpoints, and never reruns PDE, Heston cells, or Near-KI |
| `_heston_diagnostics.py`  | bound-aware finite-difference Jacobian, scaled SVD, and deterministic bootstrap summaries |
| `_mo_common.py`           | shared helpers (snapshot IO, parity, OTM filter, IV inversion, env build, leverage, plots) |

Stage 07 exercises the standalone barrier engines added to `quantark`
(`quantark/volmodels/barrier.py` + the `*BarrierMCEngine` / `*BarrierPDESolver` classes under
`quantark/asset/equity/engine/`), which price a single-barrier option under Local Vol, Heston,
and SLV by both Monte Carlo and 2-D ADI PDE. Run it after stage 05:
`.venv/bin/python example/mo_volmodels/07_barrier_exotic.py --tag latest`.

Stage 08 exercises the Snowball vol-model engines under the same MO surface with a 2Y,
principal-excluded standard Snowball. It reports BSM, Local Vol, Heston QE, and SLV with
MC/PDE cross-checks where available, plus the standalone SLV QE MC engine for the
QE-specific stochastic-local-vol path scheme:
`.venv/bin/python example/mo_volmodels/08_snowball_exotic.py --tag latest`.

Stage 09 isolates hedging behavior. It adds a BSM flat-vol baseline, then holds the
calibrated LV surface, Heston parameters, and SLV leverage surface fixed while walking
the same deterministic spot path for an ATM European call and rebalancing to delta
neutral under each model:
`.venv/bin/python example/mo_volmodels/09_delta_hedging.py --tag latest`.

Stage 16 answers a different question: whether the deterministic 2-D ADI
central-bump delta and gamma are accurate enough to drive the study's daily hedge.
The default run uses 8,192 paths × 256 common scrambles for Heston, extending
the near-KI cell alone to 2,048 scrambles, and 1,024 paths × 128 scrambles for
SLV, with a coupled QE-M 4→8 substep reference ladder. Heston integrates the
independent terminal spot factor exactly; near KI it also stratifies four inner
points across the eight leading residual bridge coordinates. Only the first 256
common scramble IDs enter the cross-regime signed-bias gate. SLV uses an unbiased
path-frozen-leverage affine control, four terminal strata plus their antithetic
shifts, and eight strata on the midpoint Brownian-bridge factor. The variance and
QE-branch Sobol streams are Brownian-bridge ordered as well. These production
sampling profiles, including the exact Heston case map, are checked again by
Stage 12; a smaller or stale profile cannot admit PDE routing. The fine-reference and substep
components each use 97.5% intervals, preserving at least 95% simultaneous
stochastic coverage. The 1% spot bump is the hedge exposure being certified;
smaller bumps are a semantic diagnostic rather than a numerical-error term. It writes
`output/adi_greek_certification/{adi_greek_certification.json,adi_greek_certification.md,adi_greek_certification_decision.json}`.
Long runs can use `--resume`; deterministic anchors and each regime cell are written
atomically and reused only when both the full run configuration and all numerical
source inputs have the same SHA-256 fingerprints. Independent RQMC scrambles can be
run concurrently with `--rqmc-batch-workers` without changing their reduction order
(the production default is four workers).
The schema-12 decision artifact is self-hashed, embeds the full run configuration
and Python/NumPy/SciPy/platform identity, and Stage 12 verifies it against the live
Stage-17 amendment and routing sources before admitting a 2-D variant.
Use `--quick` only for plumbing; quick evidence is marked non-production and can
never emit a PDE admission. A variant that is not proved is recorded as
`excluded_greek_unresolved`, not silently redirected to a noisy MC hedge.
Stage 12 reads this decision by default whenever `heston` or `heston_slv` is
requested and requires both the Stage-11 PV route and Stage-17 Greek route to
be `pde`; unresolved 2-D variants are excluded from the fleet and recorded in
its manifest. The Stage-11 decision must carry the certified 2-D controls
(`variance_grid_mode=auto`, adaptive-upwind V drift, degenerate V=0 boundary,
and the dense-KI Greek clock); Stage 12 rejects the older fixed-power-grid
profile instead of silently constructing a different engine.

## Identification and cross-date evidence

Stage 04 now saves the calibration-node Jacobian of model implied vols, four explicit parameter
scalings, SVD singular directions/ranks/condition numbers, and every successful or failed
multiplier-bootstrap replicate. The Feller penalty is excluded from the Jacobian. Bootstrap
quantiles are conditional node-influence evidence for the fixed prepared target, **not**
statistical confidence intervals.

The cross-date stage does not recycle the single Sina snapshot under different labels. It uses
independently dated official CFFEX end-of-day settlement files, infers each expiry's discount
factor and forward from put-call parity, and calibrates normalized `K/F` and `C/(DF·F)` nodes.
Settlement and intraday midpoint cohorts stay separate because their price and execution
semantics differ.

Admission is fail-closed: ACT/365 maturities must lie between 7 and 365 days; selected OTM
settlements need positive volume and open interest on both wings; full-wing parity pillars must
stay inside the broad ±10% annual-rate and 1%-of-forward RMSE gates; at least five expiries and
80 nodes must remain; and every included date must match the explicitly persisted required
calibration configuration, source class, and price field. Trade dates and source hashes must be
unique. The expiry calendar
uses third Friday plus frozen holiday adjustments. There is no interpolation, extrapolation, or
smile smoothing in this cohort. Its SVD uses the square-root equal-expiry objective weights;
the unweighted quote Jacobian is retained separately for audit.

The current six-date admitted panel (2026-04-30 through 2026-07-15) is a short, caller-selected
preliminary study, not a systematic sample of all MO regimes. Full-wing parity is the frozen
primary normalization; a near-ATM regression is reported only as a sensitivity check. Raw
settlement monotonicity/convexity violations are counted without repair or silent rejection.

The formulas, scale policies, artifact semantics, caveats, and official-source links are in
[`MODEL_DIAGNOSTICS.md`](MODEL_DIAGNOSTICS.md). In particular, only the fixed-economic SVD scale
is intended for condition-number comparison across dates, and even full local rank does not
prove global identification.

## What the real data teaches

- **Negative basis.** The MO forward sits well below spot and falls with maturity; the parity
  regression recovers a large implied carry `q` (10–20%) — not dividends, but the CSI 1000 futures
  discount driven by structured-product hedging flows.
- **Raw quotes are not arbitrage-free.** Building Dupire directly hits butterfly-arbitrage
  rejections at the wings/short end. The exact raw path *stops* (no fabricated local vol); the
  default model path uses SABR smoothing plus calendar projection. A small `--vol-floor` remains
  an explicit raw-surface diagnostic, not a production repair.
- **Parameter identification.** The latest Heston fit lands exactly on the configured κ and σ
  upper bounds. The saved quote-only SVD and multiplier replicates now make local weak directions,
  cap dependence, and conditional node sensitivity auditable rather than inferring identification
  from optimizer success alone.
- **Cross-date evidence remains negative.** Six of seven requested official settlement dates pass
  the frozen gates. Equal-expiry-weight Heston RMSE spans 0.71–19.05 vol points; the fixed-scale
  Jacobian condition reaches about 30,659 with minimum effective rank 1; κ hits its cap on half
  the dates; and ρ ranges from −0.482 to −0.038.
- **Settlement quality limits attribution.** The candidate cohort contains 90 convex-slope
  violations across 31 expiry cross sections, and full-wing versus near-ATM parity sensitivity
  reaches 0.694% in forward and 132 percentage points in annualized implied rate. The result is
  that public EOD settlement evidence does not establish a robust Heston calibration—not that
  every institutional MO feed must fail.
- **Objective ↔ numerics coupling.** Feller regularization prevents the most extreme variance
  process, but the latest Heston fit still misses raw quotes by 2.22 vol points. A Feller ratio near
  one is a numerical diagnostic, not proof of a good calibration.
- **Which model for which product.** Local-vol and SLV repricing misses raw quotes by roughly
  4.73 and 4.85 vol points. Their barrier, Snowball, and hedging examples remain useful model-risk
  demonstrations; the saved public snapshot does not qualify their dynamics for production.

## Tests

```bash
.venv/bin/python -m pytest test/mo_volmodels/ -v
```

All tests run on the deterministic `sample` snapshot (offline). See
`docs/superpowers/specs/2026-07-06-mo-volmodels-calibration-suite-design.md` and the matching
plan for the design rationale, and `quantark/volmodels/` for the model kernels.
