# Framework-Native Greek-Bump Scenarios (quantark core) + otc-price-adapter v031

**Date:** 2026-07-20
**Repos:** `quant-ark` (core layer) and `otc-price-adapter` (v031 consumer)
**Kickoff decisions:** whole book; exact parity vs v023 with mandatory diagnosis on
any mismatch; greek-bump layer in quantark core; v031 beside v030; develop against
quant-ark source, vendor a fresh wheel only at the final gate.

## Problem

v030 proved `quantark.execution` matches a raw `ProcessPoolExecutor` at row
granularity (~3% overhead with the 2×workers window). But each row is an opaque
unit: the framework cannot see that `price_row` is internally **one base solve plus
six bump solves** (`EquityPosition.get_trade_risk` for cash-leg autocallables;
`GreeksCalculator.calculate_numerical_greeks` otherwise), every solve a full PDE/MC
pricing. Consequences:

1. **Load-balancing floor.** The parallel unit is the row (99 units, cost spread
   from <1s analytical to minutes of PDE). One snowball row pins a worker while
   others idle at the tail. At bump granularity the book is ~693 comparable cells.
2. **No reuse surface.** Prepared artifacts and draws cannot engage across bumps
   because the framework never sees the bumps; everything is rebuilt 7× per row
   inside the opaque runner.
3. **Greek bumping is not a framework citizen.** The bump loop is locked inside
   `GreeksCalculator`/`get_trade_risk`; no consumer can express "price these greeks
   as scenario cells" even though bumps are exactly the framework's
   shared-base + tagged-mutation shape.

## Decisions

- **D1 — Bump-granularity cells, whole book.** Every row (all structures) becomes
  `base` + per-greek bump cells in one scenario plan. Cheap analytical rows simply
  have cheap cells.
- **D2 — Exact parity, diagnose on mismatch.** v031 output must equal v023
  **bitwise** (floats `==`, strings `==`) row-for-row. Any mismatch is
  root-caused before proceeding — never widened to a tolerance, never silently
  fallen back. The design achieves this structurally: identical env mutations
  (the same `GreeksCalculator` helpers), identical frozen bump context per cell,
  identical assembly arithmetic (formulas mirrored verbatim), fresh deterministic
  rebuilds per cell.
- **D3 — Core layer in `quantark.execution.greeks` + one session API.** A
  reusable `TradeState` + bump-cell builder + same-type bump transformer +
  greek assembler, usable by any consumer (adapter, future VaR/backtest
  scenario work), plus `PricingSession.run_scenario_plans` (multi-base plan
  scheduling through one pool) so per-row plans still pack across rows.
  `GreeksCalculator` and `EquityPosition` are NOT modified; the new module
  calls their existing helpers so conventions stay single-sourced.
- **D4 — No cross-cell artifact/draw reuse in this phase.** The win claimed here
  is wall-clock load balancing + the decomposition foundation. Prepared-artifact
  and draw reuse across sibling cells requires fingerprint-verified invariance
  under each mutation and is a follow-up (see Out of scope). CPU total ≈ v023;
  wall-clock improves via packing and scales past 99 workers-equivalent.
- **D5 — v031 beside v030.** New `otc_quantark_pricer_v031.py`; v030 (row-opaque)
  and v023 (serial reference) remain untouched as baselines. Output suffix
  `_qa031_cells`.
- **D6 — Version/dependency mechanics.** quantark work lands on quant-ark main
  under the unreleased 0.3.0 line; version bumps `0.3.0rc1 → 0.3.0rc2`. Adapter
  develops against quant-ark source (PYTHONPATH override); at the final gate a
  fresh wheel is built, vendored, re-locked, and the full adapter suite reruns
  against the wheel.

## Architecture

### quantark core: `quantark/execution/greeks.py`

**Trade state.** `TradeState` (dataclass): `product`, `pricing_env`,
`cash_legs`, `engine`, `streams`, `quantity` — the resolved per-row base a
registered factory returns and the same-type unit bump transformers mutate
(deepcopy-based, never in place; the planner's purity check verifies this on
the declared components).

**Cell vocabulary.** `GreekBumpCell` (frozen dataclass): `bump_id` ∈
{`base`, `spot_up`, `spot_down`, `vol_up`, `rate_up`, `div_up`, `theta`},
`greeks_served` (e.g. `spot_up` serves delta+gamma), `mutation_tags`
(`spot`/`vol`/`rate`/`div`/`time`). Builder:

```python
def greek_bump_cells(greeks: Sequence[str]) -> tuple[GreekBumpCell, ...]
```

maps a requested greek set (v023 uses
`delta, gamma, vega, theta, rho, dividend_rho` (+`price`)) to the minimal cell
set with deterministic ordering (base first). Delta+gamma share `spot_up`/
`spot_down` exactly as both v023 paths do.

**Bump application.** One function, reusing the calculator's own env builders so
conventions can never fork:

```python
def apply_greek_bump(bump_id: str, state: TradeState, gc: GreeksCalculator)
    -> TradeState  # same type; theta degeneracy flagged on the returned state
```

- `spot_up/down`: deepcopy env, `spot_quote.spot *= 1 ± bc.spot_bump`
  (verbatim `get_trade_risk`).
- `vol_up`: `gc._build_vol_bumped_env(env, product, cur_vol, bc.vol_bump, +1)`
  with `cur_vol = env.get_vol(strike-or-spot, T)` — verbatim.
- `rate_up`: the exact env construction `calculate_numerical_rho` /
  `get_trade_risk` use (flat replacement at `cur_rate + bc.rate_bump`).
- `div_up`: `gc._build_div_bumped_env(...)` — verbatim.
- `theta`: `gc._advance_theta_bump(...)`, deepcopy product,
  `valuation_date` shift, `product.time_shift(...)`, per-leg `time_shift` —
  verbatim `_trade_theta`, including the degenerate early-outs
  (`time_bump <= 0`, maturity exhausted, `dropped_all`) reported as
  `theta_degenerate` so the assembler can return the same `0.0`.
- The bump engine for **all** cells of a row is
  `gc._resolve_bump_engine(product, base_env, engine)` resolved against the
  **base** env (the frozen-grid contract of `get_trade_risk` §11.4) — each cell
  re-resolves it deterministically from the same base inputs.
- Private-helper access happens inside quantark (same package family), so no
  consumer touches underscore APIs; the module re-exports what it needs.

**Cell execution.** For a cash-leg trade a cell returns
`(npv, leg_sum)` from ONE `bump_engine.price_with_events(product', env',
streams)` + `value_leg` sum (mirroring `get_trade_risk.reprice`); the `base`
cell additionally returns per-leg PVs. For a legless trade a cell returns
`engine.price(product', env')`. Helper:

```python
def run_greek_bump(bump_id, product, env, engine, cash_legs, streams, gc, unit_notional)
    -> GreekBumpValues  # floats only: npv/price, leg_sum, per-leg pvs (base), flags
```

**Assembly.** Pure arithmetic, mirrored verbatim from the two v023 paths:

```python
def assemble_trade_greeks(values_by_bump, quantity, spot, bc, requested)   # get_trade_risk conventions
def assemble_product_greeks(values_by_bump, bc, requested)                  # calculate_numerical_greeks conventions
```

producing the same dicts (`product`/`total`/`leg_pvs`, resp. flat greeks) as the
originals: central delta `(nu - nd)/(2h)` with `h = spot * bc.spot_bump`;
second-order gamma; one-sided vega raw P&L; one-sided rho/rhoQ with the single
`0.01/bump` rescale; theta as difference with degenerate-zero. A quantark unit
gate (`test/execution/test_greek_bump_cells.py`) asserts bitwise equality of
{`greek_bump_cells` + `apply_greek_bump` + `run_greek_bump` +
`assemble_*`} against `get_trade_risk` / `calculate_numerical_greeks` on fixture
trades (snowball with cash legs on PDE, vanilla/barrier/sharkfin on their
engines, MC phoenix with fixed seed) — this is the core exactness gate,
independent of the adapter.

### Adapter: `otc_quantark_pricer_v031.py`

- **Transport unchanged:** reuses `otc_execution_transport` (settings and row
  encodings).
- **Per-row base + same-type bump transformer (spec-gate finding 1, refined
  after contract analysis):** `plan_scenarios` executes transformers at
  planning time and attributes mutations by fingerprinting declared components
  on base vs transformed — the contract requires transformer input and output
  to be the same shape. A settings→bundle transformer would only pass
  vacuously (unfingerprintable whole ⇒ conservative invalidation, tags
  decorative). Therefore each ROW is its own scenario plan: base =
  `BaseInputsRef(factory_id="otc-trade-state/v1",
  payload=settings_payload + encoded row)`, whose registered factory performs
  the deterministic per-row build (`legacy.build_pricing_env`/`build_product`/
  `native_autocall_cash_legs` — the same calls v023 makes) and returns a
  `TradeState` (product, env, cash legs, engine, streams, quantity sign). The
  core transformer `greek-bump/v1` is then **same-type** (`TradeState →
  TradeState` via `apply_greek_bump`), with real `components` extracting the
  mutated coordinates (spot, vol reference, rate reference, div reference,
  valuation date) and real `allowed_tags`/`mutation_tags` — the framework's
  mutation-footprint validation genuinely engages, including transformer
  purity checks. The runner (`greek-value/v1`, value_kind float) only prices
  `resolved.transformed` via `run_greek_bump`.
- **Cross-row packing via a new core API (`run_scenario_plans`).** One plan per
  row must not serialize rows (99 barriers would forfeit the packing win). The
  session gains `run_scenario_plans(plan_inputs, engine_factory)` where
  `plan_inputs = [(base_ref, specs), ...]`: each entry is planned/validated
  independently (existing `plan_scenarios`), then ALL cells execute through
  ONE bounded-window pool (the processes backend gains per-cell worker-spec
  payloads — each cell already travels with its own base payload), and
  outcomes reassemble per plan in caller order. Serial backend = flat loop.
  This is the reusable "portfolio × bumps" primitive; v031 is its first
  consumer.
- **Row identity (spec-gate finding 4):** the loader does not enforce unique
  confirmation IDs, while v023 preserves every input row positionally. Rows
  are keyed by source ordinal (plan order = row-major ordinal); within a row,
  `scenario_id = f"r{ordinal:04d}::{bump_id}"` with `trade_id` carried as
  payload data only; assembly groups by ordinal. A duplicate-trade_id parity
  test is part of the release gates.
- **Two-phase execution (spec-gate finding 2):** phase A runs only the `base`
  cells of all rows; rows whose base cell errored become v023-identical error
  rows and emit NO bump cells. Phase B runs the bump cells of healthy rows.
  This mirrors v023's stop-at-first-exception semantics (a build or base-solve
  failure never triggers the six bump solves), prevents one bad row from
  burning ~a row of redundant solves under degradation, and costs one barrier
  between two plans (base solves are ~1/7 of total work; the barrier idle is
  bounded by the slowest single base solve). A bump-cell failure in phase B
  marks that row `error` with the failing cell's message (v023 would have
  failed at the same solve). Both phases use the v030 processes backend with
  the 2×workers default window; caller order is row-major.
- **Parent assembly:** group outcomes by ordinal, run `assemble_*`, then apply
  v023's row post-processing verbatim (delta×spot and gamma×spot²×0.01 scaling,
  leg-PV column mapping, `deterministic_adjustments` + `cash_leg_greeks` for the
  legless path — cheap parent-side calls, same code), producing the identical
  24-column row. Model tag `{model}:qa031_cells`; pricing warnings must
  byte-match v023's (assembled from the same parts).
- **CLI:** v030-compatible (`--workers`, `--retries`, `--max-in-flight`,
  `--limit`, `--trade-id`); output suffix `_qa031_cells`.

### Release gates (spec-gate finding 3 — adapter-level, not just core-level)

The core unit gate (bitwise helper equality) is necessary but NOT sufficient;
the release gates exercise every new boundary:

1. **Exact frame parity vs v023** (`check_exact` frame equality, all 24
   columns): structure-diverse subset by default; full 99-row book as a
   `@pytest.mark.slow` gate that must run green before merge.
2. **serial == processes** complete-payload validator + exact frame equality +
   manifest-fingerprint equality (as v030).
3. **Failure modes:** error row byte-parity (unsupported structure), a
   base-solve failure produces an error row WITHOUT running its bump cells
   (submission-count assertion), bump-cell failure marks only its row, empty
   book keeps the output schema, duplicate trade_id book prices row-for-row,
   `--retries` plumbing, worker-death typed error passthrough.
4. **Runtime duality:** the full default suite green against quant-ark SOURCE
   during development AND against the freshly vendored 0.3.0rc2 wheel at the
   final gate (rebuild, re-vendor, `uv sync --locked`, rerun).
5. **Benchmark:** subset + worker-scaling (4/8/14) v030 vs v031 published in
   the README with honest framing (packing win, no reuse claims).

### Exactness argument (what makes bitwise achievable)

v023 computes: build env/product once → resolve frozen bump engine → 7 solves on
mutated deepcopies → arithmetic. v031 computes the same solves in separate
processes: each cell deterministically rebuilds the same env/product from the
same settings+row bytes (v030's transport gates already prove rebuild-identity at
row level), applies the same mutation via the same helper code, resolves the same
frozen bump context from the same base inputs, and runs the same engine solve.
Float arithmetic in assembly is copied operation-for-operation. Fixed seeds make
MC deterministic. No shared mutable engine state crosses cells (fresh engine per
cell). Under D2, any observed mismatch is a bug to diagnose (likely suspects:
engine-internal caching keyed off mutated envs, non-deterministic dict ordering
in warnings, theta calendar-mode divergence) — fix or document root cause, never
tolerate.

### Performance model (honest)

- Same total CPU as v023 (+small per-cell rebuild overhead: env/product build,
  ~ms against solves of seconds-to-minutes; bump-context prep repeated per cell
  instead of once per row — measured and reported, expected small for PDE
  autocallables where time-stepping dominates).
- Wall-clock gain = packing: cells are the schedulable unit, so the book's
  critical path drops from `max(row cost)` to ≈ `max(cell cost)` and worker
  counts beyond 99 become meaningful. Benchmark gate: 16-row subset AND
  worker-scaling table (4/8/14 workers) v030 vs v031, plus full-book numbers in
  the README table; no specific speedup is promised in advance.

## Failure handling

- **Row errors mirror v023 via the two-phase design.** Phase A
  (`collect_errors=True`) surfaces build/base-solve failures as v023-identical
  error rows (same exception text — v023 fails at the same call) and suppresses
  their bump cells entirely. Phase B failures mark only the owning row `error`
  with the failing cell's message; v023 would have failed at the same solve, so
  the parity gate exposes any divergence and under D2 it gets diagnosed.
- **Worker deaths:** unchanged v030 semantics (`WorkerInfrastructureError`,
  `--retries` resubmits unfinished cells).
- **Version skew / registration:** same registries + defining-module rules as
  v030; v031 registers its runner under its own canonical module identity.
- **Degenerate theta** rows (matured products) return 0.0 exactly as v023.

## Out of scope

- Cross-cell prepared-artifact / draw-cache reuse (requires fingerprint-verified
  invariance of grids/factorizations under each mutation class; the cell
  decomposition built here is its prerequisite).
- Bucketed greeks, vanna/volga/charm/etc. (cell vocabulary covers the v023
  seven; extension is mechanical later).
- Dask backend for bump cells; VaR/backtest adoption of the layer.
- Any change to v023, v030, `GreeksCalculator`, or `EquityPosition` behavior.
- Retiring v030 (it remains the row-opaque baseline and fallback).
