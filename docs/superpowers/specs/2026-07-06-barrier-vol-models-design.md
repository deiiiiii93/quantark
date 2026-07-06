# Barrier Option Pricing under Local-Vol / Heston / SLV — Design

**Date:** 2026-07-06
**Status:** feature-flow (autonomous); spec gate = Codex adversarial-review (1 iter)
**Topic:** Add standalone `quantark` engines that price single-barrier options under the three
`volmodels` processes (Dupire Local Vol, Heston, Heston-SLV) by **both Monte Carlo and PDE**,
then consume them in the MO options lecture suite to show model divergence on an exotic.

## 1. Problem

`quantark` can price single-barrier options only under **Black-Scholes** (`BarrierAnalyticalEngine`,
`BarrierPDESolver`, `BarrierOptionMCEngine` — all GBM/flat-vol). The `volmodels` MC/PDE kernels
price **European vanillas only**; none monitor a barrier along the path. Consequently there is no
way to price a path-dependent exotic under a smile-calibrated model, which is exactly where model
choice matters: LV, Heston, and SLV can all fit the same vanilla smile yet disagree materially on a
barrier because a barrier depends on **forward-vol dynamics** that vanillas do not see.

This feature adds barrier pricing under all three vol models (MC + PDE), validated four ways, and
wires an up-and-out-call case into the MO lecture (`example/mo_volmodels/`).

## 2. Decisions

- **Product:** reuse the existing `BarrierOption` (fields: `strike, maturity, option_type,
  barrier, barrier_type ∈ {UP_IN, UP_OUT, DOWN_IN, DOWN_OUT}, rebate, observation_type ∈
  {CONTINUOUS, DISCRETE}, observation_dates`). No product changes.
- **Barrier scope:** all four types; `rebate`; **continuous and discrete** monitoring. Single
  barrier only (double/window/partial barriers out of scope).
- **Rebate convention (honor `pay_at_hit`):** `BarrierOption` carries a `pay_at_hit: bool`. The
  analytical engine pays a knock-out rebate at the **hit time** when `pay_at_hit=True`, else
  discounts it to **maturity** (`rebate·e^{-rT}`). The new engines MUST reproduce this:
  - **MC:** on a knock-out breach at monitoring step `t_hit`, accrue `rebate` discounted to `t_hit`
    (`pay_at_hit=True`) or to `T` (`pay_at_hit=False`).
  - **PDE:** the knock-out Dirichlet boundary value at time `t` is `rebate` when `pay_at_hit=True`
    (rebate paid on contact, PV-at-contact = rebate) or `rebate·DF(t→T)` when `pay_at_hit=False`.
- **Knock-in pricing (rebate-correct, not naive parity):** `KI = Vanilla − KO` holds **only for
  zero rebate**. A knock-in rebate is a *no-touch* leg — paid iff the barrier is **not** breached by
  expiry — which naive parity drops. Therefore:
  - **MC:** price knock-in **directly** — pay the vanilla payoff if the barrier is ever breached,
    else pay `rebate` at expiry. This is exact for any rebate and is the reference.
  - **PDE:** `KI = Vanilla − KO(rebate=0) + NoTouchRebate`, where `NoTouchRebate = rebate ×`
    (no-touch value), obtained from the same barrier solver run with unit rebate and zero option
    payoff (a digital no-touch). Equivalently reject `rebate>0` on knock-in with a `ValidationError`
    if the no-touch leg is deferred — but v1 implements it so all four types support rebate.
  - **Parity test:** assert `KI + KO = Vanilla` **only at `rebate=0`**; test KI rebate separately
    against the flat-vol analytical no-touch price.
- **MC approach (Phase 1):** refactor each kernel's internal `_simulate_*` to optionally emit
  barrier-monitoring state — a **running extremum** (min and max spot over the path, O(1) memory,
  for CONTINUOUS) and **spot samples at monitoring indices** (for DISCRETE). Add
  `price_barrier_{lv,heston,slv}_mc(...)`. The European pricers keep calling the same simulation
  (DRY); a barrier pushed to ±∞ reproduces them.
- **PDE approach (Phase 2):**
  - **1D LV** (`price_barrier_lv_pde`): reuse `_solve_lv_pde`'s Crank–Nicolson machinery; impose a
    **Dirichlet knock-out boundary** `V = rebate · DF(t→T)` at the barrier. CONTINUOUS = enforce
    every step; DISCRETE = enforce only at observation steps (snap the barrier to the nearest grid
    time). Grid spans the barrier (barrier is an interior/edge node, not truncated away).
  - **2D Heston/SLV** (`price_barrier_{heston,slv}_pde`): reuse the validated `adi_core` operators
    on the full `(x=ln S, v)` grid; **inject the barrier condition between ADI steps** — at each
    monitoring time set `V(x beyond ln(barrier), all v) = rebate · DF` and keep the Dirichlet edge.
    This wraps barrier handling *around* the European ADI solve without rewriting its operators.
    CONTINUOUS = every step; DISCRETE = observation steps only.
- **Engines:** six thin equity wrappers mirroring the European ones, each pricing `BarrierOption`:
  `LocalVolBarrierMCEngine`, `HestonBarrierMCEngine`, `HestonSLVBarrierMCEngine`,
  `LocalVolBarrierPDESolver`, `HestonBarrierPDESolver`, `HestonSLVBarrierPDESolver`. They validate
  the product is a `BarrierOption`, thread per-step `r_fwd`/`carry_fwd` from the curves (as the
  European engines do), and delegate to the kernels. `HestonBarrier*` and `HestonSLVBarrier*` take
  `HestonParams` (+ `LeverageSurface` for SLV) exactly like their European counterparts.
- **BSM-flat validation anchor:** the existing `BarrierAnalyticalEngine._price_knock_out_closed_form`
  (Reiner–Rubinstein, continuous monitoring). A vol-model engine fed a **flat** IV surface / a
  degenerate Heston (`v0 = θ`, `σ → 0`) must reproduce it (continuous monitoring) within tolerance.
- **No MC inside PDE** (per repo rule): PDE kernels/engines never import or call MC.
- **Numerical hygiene:** `quantark.util.numerical` for all tolerances/protected math; two-level
  engine-enum pattern where a method choice exists (e.g. ADI scheme already exists as `ADIScheme`).

## 3. Architecture

```
quantark/volmodels/
  localvol/mc_kernel.py    + _simulate_lv_paths (extremum/samples) + price_barrier_lv_mc
  localvol/pde_kernel.py   + price_barrier_lv_pde  (1D CN, Dirichlet KO)
  heston/mc_kernel.py      + monitoring in _simulate_terminal_spot -> _simulate_heston + price_barrier_heston_mc
  heston/pde_kernel.py     + price_barrier_heston_pde (2D ADI + barrier injection)
  slv/slv_mc_kernel.py     + monitoring in _simulate_slv + price_barrier_slv_mc
  slv/slv_pde_kernel.py    + price_barrier_slv_pde (2D ADI + barrier injection)

quantark/asset/equity/engine/
  mc/{local_vol,heston,heston_slv}_barrier_mc_engine.py
  pde/{local_vol,heston,heston_slv}_barrier_pde_solver.py
  (register in the mc/ and pde/ __init__.py)

example/mo_volmodels/
  _mo_common.py            + build_barrier_models() helper (calibrated LV/Heston/SLV -> engines)
  07_barrier_exotic.py     up-and-out call priced MC+PDE under BSM/LV/Heston/SLV
  06_lecture.py            + "Exotics: where models diverge" section
```

Data flow: calibrated model objects (from stages 03–05: `LocalVolSurface`, `HestonParams`,
`LeverageSurface`) + `PricingEnvironment` → barrier engine → price. The engines own no calibration;
they consume already-calibrated inputs, mirroring the European engines.

## 4. Correctness backbone (4-way cross-check, as tests)

1. **European limit.** Barrier set far out-of-reach ⇒ `price_barrier_*` (KO, no rebate) matches the
   existing `price_european_*` to MC standard error / PDE discretization tolerance, per model.
2. **In–out parity.** `KI + KO = Vanilla` at **`rebate=0`** for each model × method (MC and PDE)
   within tolerance. The knock-in rebate (no-touch leg) is validated separately in (3).
3. **BSM-flat analytical.** LV with a flat surface, and Heston with `v0=θ, σ≈0`, both continuous
   monitoring, reproduce `BarrierAnalyticalEngine` (Reiner–Rubinstein) within tolerance; MC and PDE
   each checked against it. This covers **both** `pay_at_hit` settings for knock-out rebates and the
   **knock-in no-touch rebate** leg, since the analytical engine prices all of them.
4. **MC ↔ PDE agreement.** On a genuine smile, `price_barrier_*_mc` ≈ `price_barrier_*_pde` per
   model within combined tolerance. Discrete-vs-continuous monitoring gaps are reconciled with the
   Broadie–Glasserman–Kou continuity shift `B → B·exp(±0.5826·σ·√Δt)` (documented; used only to
   explain residuals, not to alter prices).

Additional guards: convergence sanity (finer grid/more paths ⇒ tighter MC↔PDE gap); reject a
barrier equal to spot or a non-positive barrier with a `ValidationError`.

## 5. Failure handling (no fabrication)

- Barrier already breached at t=0 (e.g. `UP_OUT` with `B ≤ S0`): a knock-out is worth only its
  rebate (paid at T); a knock-in is immediately alive (= vanilla). Handle explicitly, not by NaN.
- `DISCRETE` monitoring with `observation_dates` not aligned to the grid: snap each observation to
  the nearest grid step and record the snap (do not silently drop dates); if a date exceeds
  maturity, raise `ValidationError`.
- ADI barrier injection must not corrupt the `v`-direction: the override sets whole `x`-columns
  beyond the barrier, never partial cells that would break the tridiagonal solves.
- Any inability to price correctly (e.g. leverage surface missing for SLV) raises the appropriate
  `QuantArkException` subclass; no silent fallback.

## 6. Out of scope

- Double / window / partial-time / soft barriers; American exercise; discrete cash dividends.
- Per-date `ObservationSchedule` (varying barrier/payoff per observation): rejected with `ValidationError`
  in v1 (flat scalar barrier/rebate + `observation_dates` is supported). `participation_rate` **is**
  honored (applied once at each engine wrapper). Continuous monitoring uses a Brownian-bridge crossing
  correction (not grid-node monitoring).
- SR 11-7 two-developer validation package (the 4-way cross-check tests + Codex gate are the agreed
  rigor).
- Greeks for the barrier engines beyond what the shared greek helpers already provide (revaluation
  Greeks are acceptable; no bespoke barrier-adjoint work in v1).
- FX asset-class barrier-under-vol-models (equity only in v1).

## 7. Phasing (each phase independently testable; Phase N gates on N−1's cross-checks)

- **Phase 1 — Barrier MC.** Kernels + 3 MC engines. Tests: European limit, in–out parity,
  BSM-flat analytical (MC), convergence.
- **Phase 2 — Barrier PDE.** 1D LV + 2D Heston/SLV kernels + 3 PDE solvers. Tests: European limit,
  in–out parity, BSM-flat analytical (PDE), MC↔PDE agreement (Phase-1 engines as reference).
- **Phase 3 — MO consumer + lecture.** `07_barrier_exotic.py` (up-and-out call, MC+PDE, BSM/LV/
  Heston/SLV) + lecture section + comparison artifact. Test: runs on the sample tag; produces a
  finite divergence table; MC↔PDE agree on BSM and LV.

## 8. Success criteria

1. Six new engines price `BarrierOption` (all four types, rebate, both monitorings) under LV/Heston/SLV.
2. All four cross-check families pass as automated tests under `.venv/bin/python -m pytest`.
3. `example/mo_volmodels/07_barrier_exotic.py` prices an up-and-out call MC+PDE under all four
   models and shows a non-trivial spread across models, with the lecture section explaining it.
4. No regressions in the existing suite (`test/` and `test/mo_volmodels/`).
