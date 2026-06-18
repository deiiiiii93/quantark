# Equity Total Return Swap (TRS) Migration — Design Spec

**Date:** 2026-06-17
**Branch:** `worktree-worktree-equity-swap-migration`
**Source:** `/Users/fuxinyao/Documents/QuantArk/asset/equity/product/swap` (+ `engine/cashflow`)
**Target:** `quantark/asset/equity/product/swap` + `quantark/asset/equity/engine/cashflow`

## 1. Goal & Scope

Migrate the equity Total Return Swap product family and its cashflow pricing
engine from the legacy QuantArk repo into quant-ark, following quant-ark
conventions. All three product variants are in scope:

- `OneAssetTotalReturnSwap` — single-asset TRS (core).
- `MultiAssetTRS` — basket of single-asset TRS sharing one contract.
- `OneAssetTotalReturnSwapDualCcy` — single-asset TRS with an FX settlement overlay.

Plus the engine stack: `TotalReturnSwapEngine`, `accrual_calculator` strategies,
`ScheduleManager`/`MarginAccount`, and `TRSEventHandler`.

## 2. Nature of the Model (important)

This is a **realized-cashflow accounting engine**, not a risk-neutral derivative
pricer. It consumes an **observed** asset price series (`pd.Series` indexed by
`YYYY-MM-DD` strings) and walks a holiday-aware trading calendar from contract
start to valuation date, accruing:

- Fixed-leg interest (notional / market-value / last-market-value accrual bases).
- Float-leg mark-to-market (price appreciation × quantity × direction).
- Cash/share dividends, redemptions, upfront/unwind fees.
- A margin ledger (`MarginAccount`) producing a margin mark-to-market.

It does **not** use `PricingEnvironment`, stochastic processes, or
`get_payoff(spot)`. Per design decision, TRS products are **standalone cashflow
products** with their own small base interface (`BaseSwap`), not subclasses of
`BaseEquityProduct` (whose payoff-on-spot abstraction does not model a TRS).

## 3. Target Layout

```
quantark/asset/equity/product/swap/
  __init__.py                  # exports product + param classes
  base_swap.py                 # BaseSwap ABC: price(precision) + state property
  trs_params.py                # enums + dataclasses (English docstrings)
  trs_schedule.py              # ScheduleManager, MarginAccount
  trs_event_handler.py         # TRSEventHandler
  one_asset_trs.py             # OneAssetTotalReturnSwap(params)
  multi_asset_trs.py           # MultiAssetTRS(params, assets)
  one_asset_trs_dual_ccy.py    # OneAssetTotalReturnSwapDualCcy(params, fx_*)

quantark/asset/equity/engine/cashflow/
  __init__.py
  accrual_calculator.py        # AccrualCalculator strategies + factory
  total_return_swap_engine.py  # TotalReturnSwapEngine (no WindPy)

quantark/util/calendar/business_calendar.py
  # + four trading-day helper methods on Calendar (see §4)

test/
  test_trs_calendar_helpers.py
  test_total_return_swap.py
  test_multi_asset_trs.py
  test_trs_dual_ccy.py

example/
  total_return_swap_demo.py
```

## 4. Calendar Extension

quant-ark's `Calendar` (datetime-based, `holidays` set + `weekend_days`) lacks the
day-range helpers the engine needs. Add four methods to `Calendar`, porting the
legacy `XCalendarTools` semantics exactly but built on quant-ark's holiday set.
Inputs accept `str` (`YYYY-MM-DD`) or `datetime`.

- `get_calendar_days(start, end, side="both") -> list[datetime]`
  All calendar days in `[start, end]`; `side` trims endpoints
  (`left`=drop last, `right`=drop first, `both`=keep all, `neither`=drop both).
- `get_working_days(start, end, side="both") -> list[datetime]`
  Business days only (non-weekend, non-holiday); same `side` trimming.
- `get_next_trading_date(date, n=1, only_holidays=True, convention=MODIFIED_FOLLOWING) -> str`
  nth trading day from `date`. Preserves legacy edge behavior: when `n>0` and the
  base date is already a business day with `only_holidays=False`, advances a full
  `n`; when non-business, lands on the rolled date then offsets `n-1`. Returns
  `YYYY-MM-DD`.
- `get_num_of_calendar_days(start, end, side="left") -> int`
  Plain calendar-day difference; `both`=+1, `neither`=-1, else raw `(end-start).days`.

Day-range methods return `datetime` objects so engine code can call
`.strftime("%Y-%m-%d")` unchanged. These are general-purpose and reusable
library-wide; covered by `test_trs_calendar_helpers.py`.

## 5. Convention Upgrades During Port

- Canonical `quantark.*` imports throughout (no legacy flat names).
- Chinese docstrings/comments → English.
- `raise ValueError(...)` → `raise ValidationError(...)` (from
  `quantark.util.exceptions`).
- Raw float math → `quantark.util.numerical` helpers (`safe_divide` for
  `notional / price`, `is_zero` for guards) **only where it preserves exact
  numerical results** — accrual day-count arithmetic stays exact.
- `WindPy` import and the `get_close_price_by_date_range` Wind helper are
  **removed** (out of scope; data sourcing is the caller's responsibility).
- File names → snake_case; class names unchanged (PascalCase).
- `OneAssetTotalReturnSwapDualCcy` is **rebuilt** against the params-based API
  (the legacy version calls `super().__init__` positionally against the old
  signature and is broken).

## 6. Data Contract (preserved, validated explicitly)

The engine looks up `params.asset.asset_prices[pivot_date]` for arbitrary pivot
dates. The contract: `asset_prices` is a `pd.Series` indexed by `YYYY-MM-DD`
strings covering every trading pivot date from contract start to valuation date.
Rather than silently reindexing, the product validates on construction that the
series is non-empty and string-indexed, and the engine surfaces a clear
`MarketDataError` if a required pivot date is missing.

## 7. Dev Gates (each closed by `/zenmux-codex-review-loop`, ≤3 iterations)

**Gate A — Foundation**
- Calendar helpers (§4) + `trs_params.py` (enums + dataclasses) +
  `accrual_calculator.py`.
- Tests: `test_trs_calendar_helpers.py`, accrual-strategy unit tests.
- Review, fix, re-run until clean (max 3 loops).

**Gate B — Core engine + OneAssetTRS**
- `total_return_swap_engine.py`, `trs_schedule.py`, `trs_event_handler.py`,
  `base_swap.py`, `one_asset_trs.py`.
- Tests: `test_total_return_swap.py` — golden cashflow numbers, accrual-side
  variants, redemption/dividend/fee events, margin MtM, spot vs full output.
- Review loop (max 3).

**Gate C — Composites**
- `multi_asset_trs.py`, `one_asset_trs_dual_ccy.py`, `example/total_return_swap_demo.py`.
- Tests: `test_multi_asset_trs.py`, `test_trs_dual_ccy.py`.
- Review loop (max 3).

**Finish** — full suite green, `__init__` exports + equity CLAUDE.md updated,
finishing-a-development-branch.

## 8. Testing Strategy

- TDD per gate: characterize legacy numeric behavior with a constructed asset
  price path and assert exact cashflow/MtM outputs (the legacy engine is
  deterministic given inputs).
- Worktree shadowing: editable install resolves `quantark` to the main repo, so
  tests run with `PYTHONPATH=$PWD <main-venv>/bin/python -m pytest` to exercise
  worktree source.
- Edge cases: matured contract, accrual sides (left/right/both/neither),
  market-value vs last-market-value accrual, zero-notional redemption (interest /
  dividend-only settle), multi-asset long/short mix, dual-ccy fixed vs observed FX.

## 9. Out of Scope

- Wind/market-data sourcing (caller supplies `asset_prices`).
- Greeks / risk integration (VaR, stress, SIMM) — TRS is realized cashflow.
- Forward-looking / risk-neutral valuation.
```
