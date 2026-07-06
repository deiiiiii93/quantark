# MO Options Vol-Model Calibration Suite — Design

**Date:** 2026-07-06
**Status:** Approved (pending spec review)
**Topic:** A staged example suite that calibrates Dupire Local Vol, Heston, and Heston-SLV to
real CSI 1000 index-option (MO / 中证1000股指期权, underlying `000852.SH`) market data pulled
via AKShare.

## 1. Goal & Scope

Provide a hands-on "practice" suite that walks from raw market data to three calibrated
volatility models and a repricing comparison, using **real MO option quotes**. It exercises the
existing `quantark.volmodels` kernels end-to-end on live-shaped data (not synthetic flat
surfaces), so a reader learns the full desk workflow: fetch → build IV surface → Dupire local
vol → Heston calibration → SLV leverage calibration → repricing comparison.

**In scope:** European index options (MO are European-style — no early-exercise complications).
**Out of scope:** American/autocallable products, portfolio/risk aggregation, new library kernels.
The suite *consumes* existing `volmodels` APIs; it does not add pricing math to the library.

## 2. Key Constraint: the AKShare ↔ quantark interpreter split

AKShare 1.18.64 is installed only in `/opt/anaconda3` (no `quantark`); `quantark` is installed
only in the project `.venv` (no `akshare`). No single interpreter has both. This is load-bearing
for the architecture:

- **Stage 01** is the *only* network/AKShare script. It runs under `/opt/anaconda3/bin/python`
  and writes a plain-JSON **snapshot** to `data/`.
- **Stages 02–06** are pure `quantark` and replay off that snapshot **offline** under
  `.venv/bin/python`. They never touch the network.

Stage 01 hard-checks `import akshare` at startup; if absent it prints the exact
`/opt/anaconda3/bin/python example/mo_volmodels/01_fetch_mo_snapshot.py` command and exits
non-zero rather than failing cryptically. The README documents the two interpreters explicitly.

## 3. Layout

```
example/mo_volmodels/
  README.md                  how to run, the interpreter split, theory pointers
  _mo_common.py              shared helpers (snapshot IO, PCP, OTM filter, Black-IV inversion, plots)
  01_fetch_mo_snapshot.py    /opt/anaconda3/bin/python  — AKShare fetch → data/mo_snapshot_*.json
  02_build_iv_surface.py     .venv/bin/python           — PCP + OTM filter + IV inversion → surface
  03_dupire_localvol.py      .venv/bin/python           — build_dupire_local_vol, reprice, plot
  04_heston_calibration.py   .venv/bin/python           — calibrate_heston, smile-fit plots
  05_slv_calibration.py      .venv/bin/python           — calibrate_leverage_surface, reprice
  06_compare_reprice.py      .venv/bin/python           — comparison table + HTML dashboard
  data/                      mo_snapshot_YYYYMMDD.json, mo_iv_surface_*.json, plots/*.png
```

## 4. Stage-by-stage specification

### Stage 01 — `01_fetch_mo_snapshot.py`  (AKShare interpreter)
- `option_cffex_zz1000_list_sina()` → available MO contract months / symbols.
- `option_cffex_zz1000_spot_sina(symbol=...)` → per-expiry call & put quotes (strike, last,
  bid/ask if present, volume, open interest).
- CSI 1000 index spot `S0` for `000852` (index spot; e.g. `stock_zh_index_spot_*` / index hist last close).
- Persist to `data/mo_snapshot_YYYYMMDD.json`: `{ fetched_at, underlying: {code, spot},
  expiries: [{ expiry_date, T_years, quotes: [{strike, type, last, bid, ask, volume, oi}] }] }`.
  Also update a stable `data/mo_snapshot_latest.json` pointer/copy for downstream stages.
- Records the fetch timestamp; if the market is closed (all volumes zero) it still snapshots and
  downstream stages fall back to last-price mids **with a printed warning** (not silent).

### Stage 02 — `02_build_iv_surface.py`
- Load snapshot. For each expiry, **put–call parity regression**: regress `(C−P)` on `K` across
  paired strikes → slope `= −DF(T)` gives discount factor and `r(T)`; intercept gives
  `DF·F` → forward `F(T)`; carry `q(T) = r(T) − ln(F(T)/S0)/T`.
- **OTM-only liquidity filter** (`select_otm`): OTM puts for `K < F`, OTM calls for `K ≥ F`;
  drop zero-volume, crossed, and static-arbitrage-violating quotes; drop expiries with
  `< MIN_STRIKES` survivors (logged with reason — no silent drops).
- Convert OTM puts to **call-equivalent** price `C = P + DF·(F − K)`, then invert every point
  with `volmodels.black_scholes.implied_vol_call` → Black IVs. This guarantees the put and call
  wings agree at the forward (the no-arbitrage property Dupire needs).
- Assemble a `GridVolSurface(strikes, maturities, iv_grid)` and persist processed data
  (`r(T)`, `F(T)`, `q(T)`, IV grid, surviving quotes) to `data/mo_iv_surface_*.json`.
- Output: a per-expiry market smile plot (`data/plots/02_smiles.png`).

### Stage 03 — `03_dupire_localvol.py`
- Rebuild `GridVolSurface` from the processed surface; wrap `r(T)` in a rate curve and `q(T)` in
  a `div_yield` callable.
- `lv = build_dupire_local_vol(iv_surface, spot=S0, rate_curve=..., div_yield=...)`.
- Reprice the OTM chain via `LocalVolPDESolver` (and optionally `LocalVolMCEngine` as a cross
  check); report per-expiry IV RMSE vs market.
- Output: local-vol surface plot `σ_LV(K,T)` (`data/plots/03_localvol_surface.png`).

### Stage 04 — `04_heston_calibration.py`
- Build `MarketOption(K, T, price=call_equiv_price)` list (or `iv=`) from the OTM chain.
- `calibrate_heston(s0=S0, options=..., r=r(T), carry=q(T), initial=..., method="lewis")`.
- Report calibrated `(v0, κ, θ, σ, ρ)`, Feller ratio `2κθ/σ²`, cost, success/message.
- Output: model-vs-market smile per expiry (`data/plots/04_heston_fit.png`), per-expiry IV RMSE.

### Stage 05 — `05_slv_calibration.py`
- Reuse the stage-04 Heston params + stage-03 Dupire LV surface.
- `calibrate_leverage_surface(...)` (Fokker–Planck default, `FpCalibrationConfig`).
- Reprice the chain via SLV (MC and/or PDE); show SLV tightens the fit vs pure Heston.
- Output: leverage surface `L(S,T)` plot + comparison note (`data/plots/05_slv_leverage.png`).

### Stage 06 — `06_compare_reprice.py`
- Consolidate per-expiry IV RMSE for **BS-flat / LocalVol / Heston / SLV**.
- Emit a self-contained **HTML dashboard** (`data/mo_volmodels_dashboard.html`, single file,
  inline CSS/JS, no external assets — matching `example/simm_portfolio_dashboard.html` style):
  header (underlying, spot, snapshot timestamp), a model-comparison RMSE table, per-expiry
  smile-fit charts (inline SVG or base64 PNGs), and the calibrated-parameter summary.
- Also write a `data/comparison_summary.csv` for scripting.

## 5. Shared `_mo_common.py`

One module keeps each stage focused on its concept:
- `load_snapshot(path)` / `save_snapshot(obj, path)` — JSON IO with schema check.
- `imply_forward_and_rate(expiry_quotes)` — PCP regression → `(r, F, DF, q)`. **[contribution point A]**
- `select_otm(quotes, forward)` — OTM + liquidity filter predicate. **[contribution point B]**
- `black_implied_vol(call_equiv_price, S0, K, T, r, q)` — thin wrapper over
  `volmodels.black_scholes.implied_vol_call` with graceful skip-on-uninvertible.
- Plotting wrappers (matplotlib) and a tiny HTML-dashboard renderer used by stage 06.

## 6. Error handling & fidelity (no fabricated math)

Per the project's standing rule ("no stupid fallbacks", "exact semantics by default"):
- If PCP regression is ill-posed for an expiry (too few paired strikes, singular fit) or IV
  inversion fails (price at/above the no-arb upper bound), that expiry/quote is **excluded with a
  logged reason** — never a fabricated forward, rate, or vol.
- Market-closed snapshots are usable but clearly flagged; downstream RMSE reports state whether
  they ran on live or stale mids.
- Thin expiries (`< MIN_STRIKES` after filtering) are dropped from calibration and named in the log.

## 7. Learning-mode contribution points

Three spots where the user writes the meaningful 5–10 lines (scaffolding + tests provided):
- **(A)** `imply_forward_and_rate` — the put–call-parity forward/rate regression.
- **(B)** `select_otm` — the OTM + liquidity + no-arb filter predicate.
- **(C)** the Heston initial guess + parameter bounds in stage 04.

These are the genuine modeling-judgment calls; everything else is scaffolding.

## 8. Dependencies & conventions

- Reuses only existing `quantark.volmodels`, `quantark.param`, `quantark.priceenv`,
  `quantark.asset.equity` engines — no new library code.
- matplotlib 3.10 and scipy 1.17 are present in `.venv`; PNGs saved under `data/plots/`
  (mirrors the existing `example/output/*.png` convention).
- AKShare functions verified present in `/opt/anaconda3` (v1.18.64):
  `option_cffex_zz1000_list_sina`, `option_cffex_zz1000_spot_sina`, `option_cffex_zz1000_daily_sina`.

## 9. Success criteria

1. `01` produces a valid snapshot JSON (or, offline, a documented sample snapshot is committed so
   `02–06` run without network).
2. `02` recovers a monotone-ish discount factor / sensible forward per expiry and a plausible
   OTM smile (skew present).
3. `03/04/05` run to completion, each reporting per-expiry IV RMSE; SLV RMSE ≤ Heston RMSE.
4. `06` opens as a standalone HTML dashboard summarizing all three models.
5. Each stage runs standalone from the committed snapshot with the documented interpreter.
```
