# CFETS USD/CNY Vol-Model Calibration Suite

This seven-stage example asks a deliberately narrower question than “does China
have an FX-options market?”: **does the public CFETS USD/CNY volatility evidence
support a defensible Heston calibration?** It freezes the official five-delta
composite, reconstructs strikes under the published convention, calibrates
Heston to only those observations, and treats local-vol/SLV as separately
labelled diagnostics that require interpolation.

The practical conclusion is conditional. The public 1M–1Y (`core`) curve is
dense and consistent enough to obtain a low-error Heston fit and to compare
free versus hard-Feller calibrations. It is **not** an executable dealer surface:
the displayed public bid/mid/ask are composites, some displayed spreads are
zero, and the five-parameter solution can remain weakly identified or violate
Feller. This is enough for model research and public-benchmark calibration, not
enough by itself to certify production exotic pricing or execution marks.

## Data contract and quote convention

- Pair: USD/CNY, quoted as CNY per USD. CNY is domestic/quote currency and USD
  is foreign/base currency in Garman–Kohlhagen notation.
- Pillar order is fixed: `10P, 25P, ATM, 25C, 10C`.
- Delta is CFETS premium-excluded **spot delta**:
  `call = exp(-r_USD T) N(d1)` and
  `put = exp(-r_USD T) (N(d1)-1)`. Therefore call minus put equals the USD
  discount factor. ATM is ATMF, so `K = F`.
- A published mid outside the displayed public band is preserved and flagged;
  it is not “repaired” into a fictitious executable quote.
- Stage 01 is the only network boundary. Every later stage consumes tagged JSON
  offline, including the source hashes and limitations.

The historical CFETS reply for **2026-05-29 is excluded**. Its HTTP envelope
reported success, but the backend returned `data.error` and no complete 25P
records. The strict loader rejects that date rather than silently calibrating a
24-node surface. Cross-date diagnostics use only complete, like-for-like dates.

## The observed/prepared split

There are two data paths, and the artifacts never conflate them:

| Path | Input | Used by | Interpretation |
|---|---|---|---|
| Observed | Five direct CFETS nodes per expiry | Heston | Public composite evidence |
| Prepared | Per-expiry lognormal SABR, then fixed-strike calendar total-variance projection | Dupire and SLV | Differentiable interpolation, **not additional liquidity** |

Heston is fitted on normalized-forward raw nodes (`F=1`, `K/F`) so the objective
does not inherit SABR assumptions. Dupire needs strike and time derivatives;
SLV then needs the Dupire target, so both use the prepared rectangular grid and
report that smoothing dependency explicitly.
The adjacent 3W and 18M smiles are retained only as finite-difference boundary
support for the core Dupire grid; they are excluded from calibration scoring.

## Calibration universes

| Name | Tenors | Intended use |
|---|---|---|
| `core` | 1M, 2M, 3M, 6M, 9M, 1Y | Primary acceptance result; best-supported public liquidity |
| `liquid` | 1W–1Y, excluding 1D | Short-end sensitivity around the primary result |
| `full` | 1W–3Y, excluding 1D | Coverage stress; do not equate long-end publication with deep liquidity |

Use the same universe, weighting, node order, and optimizer settings when
comparing dates. Results from different objectives are sensitivity checks, not
strict time-series observations.

## Running the suite

All stages use the repository environment and add no dependency beyond
QuantArk's existing runtime.

```bash
# 01a) Live CFETS fetch (the only command that accesses the network)
.venv/bin/python example/fx_volmodels/01_fetch_cfets_snapshot.py \
  --date 2026-07-20 --time 16:00 --tag latest

# 01b) Or replay previously archived raw CFETS replies without network access
.venv/bin/python example/fx_volmodels/01_fetch_cfets_snapshot.py \
  --date 2026-07-20 --time 16:00 --raw-dir /tmp --tag latest

# 02-07) Offline modeling and report generation
.venv/bin/python example/fx_volmodels/02_build_fx_surface.py \
  --tag latest --tenor-set core
.venv/bin/python example/fx_volmodels/03_dupire_localvol.py --tag latest
.venv/bin/python example/fx_volmodels/04_heston_calibration.py \
  --tag latest --universe core --weight-mode equal
.venv/bin/python example/fx_volmodels/05_slv_calibration.py --tag latest
.venv/bin/python example/fx_volmodels/06_calibration_diagnostics.py \
  --tags 20260430 20260515 20260615 20260630 20260715 latest \
  --output-tag latest --universe core
.venv/bin/python example/fx_volmodels/07_explainer.py --tag latest
```

Or rerun stages 02–07 from an already frozen snapshot:

```bash
.venv/bin/python example/fx_volmodels/run_suite.py \
  --tag latest \
  --history-tags 20260430 20260515 20260615 20260630 20260715
```

Stage 03 is fail-closed on Dupire admissibility by default. If investigating a
known surface defect, `--vol-floor` is an explicit diagnostic override, not a
claim that the source surface was arbitrage-free. Use each script's `--help`
for resolution and multistart controls; `--fast` is intended for smoke tests,
not published calibration evidence.

Stage 05 uses the deterministic forward Fokker–Planck leverage calibration and
antithetic SLV Monte Carlo as its fit diagnostic. It also records a native
uniform-grid SLV-PDE resolution probe separately; that low-volatility numerical
probe is not allowed to contaminate the market-fit verdict.

## Stages and artifacts

| Stage | Script | Main tagged artifact |
|---|---|---|
| 01 | `01_fetch_cfets_snapshot.py` | `data/cfets_usdcny_snapshot_{tag}.json` plus dated source archive |
| 02 | `02_build_fx_surface.py` | `data/cfets_usdcny_surface_{tag}.json` |
| 03 | `03_dupire_localvol.py` | `data/cfets_usdcny_localvol_{tag}.json` |
| 04 | `04_heston_calibration.py` | `data/cfets_usdcny_heston_{tag}.json`, `data/cfets_usdcny_heston_residuals_{tag}.csv`, smile plots |
| 05 | `05_slv_calibration.py` | `data/cfets_usdcny_slv_{tag}.json` |
| 06 | `06_calibration_diagnostics.py` | `data/cfets_usdcny_diagnostics_{tag}.json` and `.csv` |
| 07 | `07_explainer.py` | `data/fx_calibration_explainer_{tag}.html` |
| runner | `run_suite.py` | fail-fast offline orchestration of stages 02–07 |

The HTML is self-contained (inline CSS and JavaScript), but its tables and
interactive figures are built from the tagged JSON evidence. Open
`data/fx_calibration_explainer_latest.html` after the pipeline finishes.

Stable `latest` JSON/CSV/HTML artifacts are intended to be reviewable and
committed with the example. Regenerable `sample` outputs are ignored. Plots are
supporting evidence and carry the same tag as their source run.

## Acceptance standard

A “developed enough” result requires all of the following, not merely one low
least-squares number:

1. the strict snapshot contains all five pillars for every selected expiry and
   round-trips the published spot-delta convention;
2. several deterministic starts converge to the same low-error raw-node fit;
3. core-universe RMSE is below **0.10 volatility point (10 vol bp)** and no
   single expiry dominates the error;
4. the conclusion survives the complete-date, same-objective cross-date panel;
5. free and hard-Feller fits are reported separately, with the fit cost of the
   constraint visible; and
6. Jacobian/SVD diagnostics disclose parameter identification rather than
   inferring it from price fit.

Passing items 1–4 supports the conclusion that CFETS's public core surface can
calibrate Heston **as a vanilla-smile benchmark**. Feller failure, unstable
parameter directions, a material hard-Feller fit penalty, or the absence of
executable spreads limits the conclusion: it does not validate Heston dynamics
for barriers, long-dated exotics, or production marks. Dupire/SLV repricing is
useful supporting evidence, but cannot upgrade five smoothed nodes into new
market observations.

## Frozen 2026-07-20 result

The checked `latest` artifacts give a deliberately split answer:

| Diagnostic | Result | Reading |
|---|---:|---|
| Core Heston free / hard-Feller RMSE | 0.0592 / 0.0786 vol points | both pass the 0.10-vol-point gate |
| Liquid-universe free RMSE | 0.0890 vol points | passes, but short-end sensitivity is larger |
| Full 1W–3Y free RMSE | 0.1178 vol points | fails; published long-end coverage is not equally calibratable |
| Six-date core free RMSE range | 0.0412–0.0592 vol points | consistently good cross-sectional fit |
| Six-date free Feller ratio | 0.529–0.632 | violated on every complete date |
| Six-date rho | −0.0596 to +0.0584 | sign changes; dynamics are not stable |
| Dupire prepared / raw in-domain RMSE | 0.1729 / 0.2132 vol points | smoothing and finite-difference sensitivity remain material |
| SLV prepared / raw in-domain RMSE | 0.2347 / 0.3122 vol points | the public surface does not validate production SLV dynamics |

So the comparison with MO is not a blanket approval. **CFETS USD/CNY is
developed enough for a good 1M–1Y Heston vanilla-smile calibration, and the raw
fit is materially cleaner and more repeatable. It is not, from public data
alone, developed enough to establish stable Heston parameters or a well-
calibrated production SLV model.** Executable option-specific history remains
the promotion gate.

## Tests

```bash
.venv/bin/python -m pytest test/fx_volmodels/ -v -o addopts=''
```

The contract tests use tiny in-memory fixtures and `tmp_path`; they do not call
CFETS and do not write into the shared `example/fx_volmodels/data` directory.
