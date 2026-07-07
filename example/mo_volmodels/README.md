# MO Options Vol-Model Calibration Suite

A hands-on, staged example that calibrates **Dupire Local Volatility**, **Heston**, and
**Heston Stochastic-Local Volatility (SLV)** to real **CSI 1000 index option (MO / 中证1000股指期权,
underlying `000852.SH`)** market data pulled via AKShare, and renders the whole study as a
self-contained HTML lecture.

The models are European index options — no early exercise — which is exactly the clean setting
Dupire / Heston / SLV assume.

## The interpreter split (important)

Two Python environments are involved and **no single interpreter has both** libraries:

| Stage | Interpreter | Why |
|-------|-------------|-----|
| `01` fetch | `/opt/anaconda3/bin/python` | has `akshare`, no `quantark` |
| `02`–`09` | `.venv/bin/python` | has `quantark`, no `akshare` |

Stage 01 is the only script that touches the network; it writes a JSON **snapshot** that stages
02–09 replay **offline**. A committed synthetic `mo_snapshot_sample.json` (arbitrage-free by
construction) drives the automated tests and lets 02–09 run with no network at all.

## Running it

```bash
# 1) live fetch — AKShare interpreter (optional; the sample snapshot works offline)
/opt/anaconda3/bin/python example/mo_volmodels/01_fetch_mo_snapshot.py

# 2-9) replay the snapshot — quantark .venv
.venv/bin/python example/mo_volmodels/02_build_iv_surface.py  --snapshot latest
.venv/bin/python example/mo_volmodels/03_dupire_localvol.py   --tag latest --vol-floor 0.05
.venv/bin/python example/mo_volmodels/04_heston_calibration.py --tag latest
.venv/bin/python example/mo_volmodels/05_slv_calibration.py    --tag latest --vol-floor 0.05
.venv/bin/python example/mo_volmodels/07_barrier_exotic.py     --tag latest --vol-floor 0.05
.venv/bin/python example/mo_volmodels/08_snowball_exotic.py    --tag latest --vol-floor 0.05
.venv/bin/python example/mo_volmodels/09_delta_hedging.py      --tag latest --vol-floor 0.05
.venv/bin/python example/mo_volmodels/06_lecture.py           --tag latest
```

Open the result: **`data/mo_volmodels_lecture_latest.html`**.

Every artifact is keyed by a `--tag` (`latest` for live data, `sample` for the deterministic
fixture) so the test pipeline and the live pipeline never clobber each other's files.

## Stages

| Script | Does |
|--------|------|
| `01_fetch_mo_snapshot.py` | AKShare MO chain + CSI 1000 spot → `data/mo_snapshot_*.json` |
| `02_build_iv_surface.py`  | **put-call parity** → r(T)/forward/carry; **OTM filter**; call-equivalent Black-IV inversion → `GridVolSurface` |
| `03_dupire_localvol.py`   | Dupire σ_LV(K,T); reprice via local-vol PDE; RMSE + surface plot |
| `04_heston_calibration.py`| calibrate (v0,κ,θ,σ,ρ); Feller check; smile-fit plot + RMSE |
| `05_slv_calibration.py`   | Fokker-Planck leverage surface L(S,t); reprice via SLV PDE + plot |
| `07_barrier_exotic.py`    | up-and-out call priced **MC and PDE** under BSM/LV/Heston/SLV → model-divergence table + bar chart |
| `08_snowball_exotic.py`   | 2Y principal-excluded standard Snowball priced **MC and PDE** under BSM/LV/Heston QE/SLV plus standalone SLV QE MC → autocallable model-divergence table + bar chart |
| `09_delta_hedging.py`     | ATM European call delta-neutral hedge demo under BSM flat vol/LV/Heston/SLV → hedge inventory, turnover, residual PnL + chart |
| `06_lecture.py`           | weave everything into the HTML lecture + comparison CSV |
| `_mo_common.py`           | shared helpers (snapshot IO, parity, OTM filter, IV inversion, env build, leverage, plots) |

Stage 07 exercises the standalone barrier engines added to `quantark`
(`quantark/volmodels/barrier.py` + the `*BarrierMCEngine` / `*BarrierPDESolver` classes under
`quantark/asset/equity/engine/`), which price a single-barrier option under Local Vol, Heston,
and SLV by both Monte Carlo and 2-D ADI PDE. Run it after stage 05:
`.venv/bin/python example/mo_volmodels/07_barrier_exotic.py --tag latest --vol-floor 0.05`.

Stage 08 exercises the Snowball vol-model engines under the same MO surface with a 2Y,
principal-excluded standard Snowball. It reports BSM, Local Vol, Heston QE, and SLV with
MC/PDE cross-checks where available, plus the standalone SLV QE MC engine for the
QE-specific stochastic-local-vol path scheme:
`.venv/bin/python example/mo_volmodels/08_snowball_exotic.py --tag latest --vol-floor 0.05`.

Stage 09 isolates hedging behavior. It adds a BSM flat-vol baseline, then holds the
calibrated LV surface, Heston parameters, and SLV leverage surface fixed while walking
the same deterministic spot path for an ATM European call and rebalancing to delta
neutral under each model:
`.venv/bin/python example/mo_volmodels/09_delta_hedging.py --tag latest --vol-floor 0.05`.

## What the real data teaches

- **Negative basis.** The MO forward sits well below spot and falls with maturity; the parity
  regression recovers a large implied carry `q` (10–20%) — not dividends, but the CSI 1000 futures
  discount driven by structured-product hedging flows.
- **Raw quotes are not arbitrage-free.** Building Dupire directly hits butterfly-arbitrage
  rejections at the wings/short end. The exact path *stops* (no fabricated local vol); the pipeline
  opts into a small `--vol-floor`. A desk would arbitrage-free the smile (SVI/SABR) first.
- **Parameter identification.** The Heston smile only weakly pins κ vs σ with few maturities — we
  regularize (cap κ, Feller penalty) rather than overfit.
- **Objective ↔ numerics coupling.** An unconstrained Heston is deeply Feller-violated, and that
  same extreme σ makes the ADI PDE mis-price by ~2 vol-pts. A Feller-aware calibration keeps both
  the fit and the PDE trustworthy.
- **Which model for which product.** SLV does *not* out-reprice vanillas vs analytic Heston (the SLV
  PDE has an inherent discretization bias). SLV's value is smile-consistent *dynamics* for **exotics**
  (barriers, autocallables, forward-vol); the leverage surface is the reusable deliverable.

## Tests

```bash
.venv/bin/python -m pytest test/mo_volmodels/ -v
```

All tests run on the deterministic `sample` snapshot (offline). See
`docs/superpowers/specs/2026-07-06-mo-volmodels-calibration-suite-design.md` and the matching
plan for the design rationale, and `quantark/volmodels/` for the model kernels.
