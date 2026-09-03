# PnL Explain Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the six engine / library quirks the PnL explain feature exposed (simple hedge executor accounting, unrolled float maturities, analytical first-hit lags, carry-invariant-only key-rate rho, identity-based MODEL detection, untested TradingClock environments) and amend the parent spec accordingly.

**Architecture:** Six independent patches, each a library change plus its explain-side consequence plus tests, then one documentation task. The backtest fixes (A, B) land first so every later backtest test runs on the corrected engine. Every patch is fail-closed: where an exact treatment does not exist the code raises; nothing approximates.

**Tech Stack:** Python 3.10–3.13, numpy, pandas, pytest (`-n0` while iterating; the parallel default for the final run).

**Spec:** `docs/superpowers/specs/2026-09-03-pnl-explain-followups-design.md` (patch spec; section numbers below refer to it). Parent: `docs/superpowers/specs/2026-09-02-pnl-explain-design.md`.

## Global Constraints

- Work in a worktree created with the native `EnterWorktree` tool from `main` at or after `5f89133`; verify `git log -1` after creating it (baseRef is a setting, never assumed). Run tests with the worktree source shadowing the editable install: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 ...` from the worktree root. The sandbox rejects `source`, `eval`, shell variables and heredocs: write scratch scripts with the Write tool and run them by absolute path.
- Commits: `git -c commit.gpgsign=false commit -F <msgfile>`; every message ends with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_019fnLipZtAJAPAQJLLEXQMb`. Never `git add example/` (two `example/mo_volmodels/data/` sample files churn under test runs; `git checkout --` them before staging). Files under `docs/` need `git add -f`. `CLAUDE.md` files are untracked and are never added.
- `riskmeasures`: the default `BucketedGreeksRequest` and every scalar greek stay bitwise unchanged (spec §10). Prices of contracts without a first-hit lag stay bitwise unchanged.
- No semantic fallbacks: raise `ValidationError` / `CapabilityError` where the spec says so (spec §11). Use `quantark.util.numerical` helpers (`is_zero`, `is_close`) instead of raw float comparisons.
- Numbers that move (spec §3.3, §4.3) are documented in the commit messages; no test is "re-pinned" silently.
- Deterministic row and frame order; every new test must pass under `-n auto` too.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `quantark/backtest/equity/hedge_executor.py` (modify) | average-cost accounting, `realized_pnl`, close-on-zero | 1 |
| `test/test_hedge_executor.py` (create) | executor unit tests | 1 |
| `quantark/asset/equity/lifecycle/float_roll.py` (create) | whitelist, `FloatMaturityRoller` | 2 |
| `quantark/asset/equity/lifecycle/manager.py` (modify) | roll untracked schedule-free products, warn once | 2 |
| `quantark/asset/equity/lifecycle/__init__.py` (modify) | exports | 2 |
| `quantark/backtest/equity/engine.py` (modify) | manager always constructed | 2 |
| `test/test_float_maturity_roll.py` (create) | manager-level roll tests | 2 |
| `quantark/asset/equity/engine/settlement_support.py` (modify) | `constant_hit_lag_year_fraction` | 3 |
| `quantark/asset/equity/engine/analytical/{one_touch,barrier,single_sharkfin,double_sharkfin}*.py` (modify) | constant-lag scaling, guards → helper | 3 |
| `quantark/asset/equity/riskmeasures/bucketed_greeks.py` (modify) | `RateKeyrateConvention`, request field | 4 |
| `quantark/asset/equity/riskmeasures/bucketed_coordinates/rate_keyrate.py` (modify) | convention branch | 4 |
| `quantark/asset/equity/riskmeasures/greeks_calculator.py` (modify) | result metadata | 4 |
| `quantark/pnlexplain/equity/bucketed.py`, `taylor.py` (modify) | rate buckets as components | 4 |
| `quantark/asset/equity/engine/base_engine.py` + opt-in analytical engines (modify) | `MODEL_FINGERPRINT_ATTRS`, `model_fingerprint()` | 5 |
| `quantark/pnlexplain/equity/fingerprints.py`, `factor_diff.py`, `lifecycle.py`, `explain.py` (modify) | `engines_equivalent` | 5 |
| `quantark/util/calendar/trading_clock.py`, `quantark/param/{vol/trading_clock_surface,rrf/trading_clock_curve,div/trading_clock_yield}.py` (modify) | `horizon_date`, `re_anchored`, `with_time_map` | 6 |
| `quantark/pnlexplain/equity/clock.py` (create), `factor_diff.py`, `scenario.py`, `taylor.py` (modify) | clock-aware equality, validation, re-anchoring, Taylor fail-closed | 6 |
| `test/test_pnlexplain_trading_clock.py` (create) | clock tests incl. axis equivalence | 6 |
| `docs/superpowers/specs/2026-09-02-pnl-explain-design.md`, `quantark/pnlexplain/README.md`, `quantark/backtest/CLAUDE.md` (modify) | amendments | 7 |

---

### Task 1: Simple `HedgeExecutor` average-cost accounting (spec §3)

**Files:**
- Modify: `quantark/backtest/equity/hedge_executor.py:1-18` (imports), `:74-77` (state), `:200-245` (`_update_hedge_position`), `:294-345` (`close_hedge_position`), `:347-358` (`get_statistics`)
- Create: `test/test_hedge_executor.py`
- Modify: `test/test_pnlexplain_backtest_equity.py:144-163`

**Interfaces:**
- Consumes: `Portfolio.update_position(position_id, quantity=None, entry_price=None)`, `Portfolio.remove_position(position_id)`, `TradeRecord` (`quantark/backtest/equity/state.py`).
- Produces: `HedgeExecutor.realized_pnl: float` (read by `BacktestEngine._record_state` through `getattr(self.hedge_executor, "realized_pnl", 0.0)` — no engine change); a `TradeRecord` with `trade_type="close"` when a hedge nets to zero.

- [ ] **Step 1: Write the failing executor tests**

```python
# test/test_hedge_executor.py
"""Simple HedgeExecutor: average-cost accounting (patch spec 2026-09-03 §3)."""
from datetime import datetime

import pytest

from quantark.backtest import ZeroCostModel
from quantark.backtest.equity.hedge_executor import HedgeExecutor
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio import Portfolio
from quantark.priceenv import PricingEnvironment

U = "TEST"
T = datetime(2026, 1, 5)


def _env(spot):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot, asset_name=U), vol_surface=FlatVolSurface(0.2),
                              rate_curve=FlatRateCurve(rate=0.0), valuation_date=T)


@pytest.fixture
def book():
    env = _env(100.0)
    portfolio = Portfolio(portfolio_name="t", pricing_environments={U: env}, creation_date=T)
    return portfolio, env, HedgeExecutor(portfolio, ZeroCostModel(), "spot")


def _hedge(ex, env, size, spot):
    env.spot_quote = SpotQuote(spot=spot, asset_name=U)
    return ex.execute_hedge(U, size, env, T)


def _identity(portfolio, ex, fills, spot):
    """portfolio_pnl + realized_pnl must equal the cash PnL of the fill sequence marked at `spot`."""
    return portfolio.get_portfolio_pnl() + ex.realized_pnl == pytest.approx(sum(q * (spot - p) for q, p in fills))


def test_open_then_increase_blends_entry_price(book):
    portfolio, env, ex = book
    assert _hedge(ex, env, 10.0, 100.0).trade_type == "open"
    rec = _hedge(ex, env, 10.0, 110.0)
    pos = ex.get_hedge_position(U)
    assert rec.trade_type == "adjust" and rec.metadata["action"] == "increase_hedge"
    assert pos.quantity == 20.0 and pos.entry_price == pytest.approx(105.0)
    assert ex.realized_pnl == 0.0
    assert _identity(portfolio, ex, [(10, 100.0), (10, 110.0)], 110.0)


def test_reduce_realizes_the_closed_lot(book):
    portfolio, env, ex = book
    _hedge(ex, env, 10.0, 100.0)
    _hedge(ex, env, 10.0, 110.0)
    rec = _hedge(ex, env, -4.0, 110.0)
    pos = ex.get_hedge_position(U)
    assert rec.metadata["action"] == "reduce_hedge"
    assert pos.quantity == 16.0 and pos.entry_price == pytest.approx(105.0)      # entry unchanged
    assert ex.realized_pnl == pytest.approx(20.0)                                # (110 - 105) * 4
    assert _identity(portfolio, ex, [(10, 100.0), (10, 110.0), (-4, 110.0)], 110.0)


def test_flip_realizes_old_lot_and_reenters(book):
    portfolio, env, ex = book
    _hedge(ex, env, 10.0, 100.0)
    _hedge(ex, env, 10.0, 110.0)
    _hedge(ex, env, -4.0, 110.0)
    rec = _hedge(ex, env, -26.0, 120.0)
    pos = ex.get_hedge_position(U)
    assert rec.metadata["action"] == "flip_hedge"
    assert pos.quantity == -10.0 and pos.entry_price == pytest.approx(120.0)
    assert ex.realized_pnl == pytest.approx(20.0 + (120.0 - 105.0) * 16.0)
    assert _identity(portfolio, ex, [(10, 100.0), (10, 110.0), (-4, 110.0), (-26, 120.0)], 120.0)


def test_net_to_zero_closes_and_the_next_hedge_opens_a_new_id(book):
    portfolio, env, ex = book
    _hedge(ex, env, 10.0, 100.0)
    old_id = ex.get_hedge_position(U).position_id
    rec = _hedge(ex, env, -10.0, 130.0)                     # exactly flat: was ValidationError before the patch
    assert rec.trade_type == "close" and rec.position_id == old_id
    assert rec.metadata["entry_price_after"] is None
    assert ex.get_hedge_position(U) is None and ex.get_hedge_quantity(U) == 0.0
    assert old_id not in portfolio.positions
    assert ex.realized_pnl == pytest.approx(300.0)
    assert _identity(portfolio, ex, [(10, 100.0), (-10, 130.0)], 130.0)
    rec2 = _hedge(ex, env, 5.0, 130.0)
    assert rec2.trade_type == "open" and rec2.position_id != old_id
    assert ex.get_statistics()["realized_pnl"] == pytest.approx(300.0)


def test_close_hedge_position_realizes(book):
    portfolio, env, ex = book
    _hedge(ex, env, 10.0, 100.0)
    env.spot_quote = SpotQuote(spot=90.0, asset_name=U)
    rec = ex.close_hedge_position(U, env, T)
    assert rec.trade_type == "close" and rec.quantity == -10.0
    assert ex.realized_pnl == pytest.approx(-100.0)
    assert not portfolio.positions
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_hedge_executor.py -q`
Expected: 5 failures — `AttributeError: 'HedgeExecutor' object has no attribute 'realized_pnl'` on the first four and `ValidationError: Quantity cannot be zero` on the net-to-zero test.

- [ ] **Step 3: Implement the accounting**

In `quantark/backtest/equity/hedge_executor.py`:

```python
# imports: add
from quantark.util.numerical import is_zero
```

In `__init__`, after `self._engine = DeltaOneEngine()`:

```python
        # Average-cost accounting (patch spec 2026-09-03 §3): realised P&L of reduced,
        # flipped and closed hedge lots. BacktestEngine adds it to net P&L.
        self.realized_pnl: float = 0.0
```

Replace the body of `_update_hedge_position`:

```python
        position = self.portfolio.positions[position_id]
        old_quantity = position.quantity
        entry = position.entry_price
        new_quantity = old_quantity + hedge_size
        entry_after: Optional[float]

        if is_zero(new_quantity):
            # net to zero: realise the whole lot and drop the position (a zero quantity
            # is not a position; this used to raise inside Portfolio.update_position)
            self.realized_pnl += (hedge_price - entry) * old_quantity
            self.portfolio.remove_position(position_id)
            del self._hedge_position_ids[underlying]
            trade_type, action, entry_after = "close", "close_hedge", None
        elif old_quantity * new_quantity < 0:
            # sign flip: realise the old lot, re-enter the remainder at today's price
            self.realized_pnl += (hedge_price - entry) * old_quantity
            self.portfolio.update_position(position_id, quantity=new_quantity, entry_price=hedge_price)
            trade_type, action, entry_after = "adjust", "flip_hedge", hedge_price
        elif abs(new_quantity) > abs(old_quantity):
            # increase: blend the entry price, realise nothing
            blended = (entry * old_quantity + hedge_price * hedge_size) / new_quantity
            self.portfolio.update_position(position_id, quantity=new_quantity, entry_price=blended)
            trade_type, action, entry_after = "adjust", "increase_hedge", blended
        else:
            # reduce: realise P&L on the closed part, keep the entry price
            self.realized_pnl += (hedge_price - entry) * (old_quantity - new_quantity)
            self.portfolio.update_position(position_id, quantity=new_quantity)
            trade_type, action, entry_after = "adjust", "reduce_hedge", entry

        return TradeRecord(
            timestamp=current_time,
            trade_type=trade_type,
            instrument_type=self.hedge_instrument_type,
            underlying=underlying,
            quantity=hedge_size,
            price=hedge_price,
            notional=notional,
            transaction_cost=transaction_cost,
            reason=reason,
            position_id=position_id,
            metadata={
                "action": action,
                "old_quantity": old_quantity,
                "new_quantity": new_quantity,
                "entry_price_after": entry_after,
            },
        )
```

In `close_hedge_position`, before `self.portfolio.remove_position(position_id)`:

```python
        self.realized_pnl += (close_price - position.entry_price) * position.quantity
```

In `get_statistics`, add `"realized_pnl": self.realized_pnl`. Update the class docstring ("Average-cost accounting: increases blend the entry price, reductions / flips / closes realise P&L into `realized_pnl`").

- [ ] **Step 4: Run the executor tests and the backtest suites**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_hedge_executor.py test/test_backtest.py test/test_backtest_interface.py test/test_multi_greek_backtest.py test/test_backtest_lifecycle.py -q`
Expected: all passed.

- [ ] **Step 5: Flip the explain test and add the zero-threshold KO test**

In `test/test_pnlexplain_backtest_equity.py` replace `test_spot_hedge_adjusts_identity_holds_states_gap_documented` with:

```python
def test_spot_hedge_adjusts_states_gap_is_zero():
    # Vanilla short-call book: the delta never vanishes, so the simple executor adjusts daily.
    # Average-cost accounting (patch spec §3) makes the engine's own P&L the value identity.
    cfg = make_multi_config(DeltaNeutralStrategy(delta_threshold=0.0))
    cfg.pnl_explain = PnLExplainConfig()
    results = BacktestEngine(cfg).run()
    recon = results.explain_reconciliation_df
    port = recon[recon["level"] == "portfolio"]
    assert len(port) == 2 * (len(results.states_df) - 1)          # two methods, from day two
    assert port["ok"].all()
    trades = results.trades_df
    assert len(trades) > 1 and (trades["trade_type"] == "adjust").any()
    assert (port["gap_states"].abs() <= 1e-8 * port["expected"].abs().clip(lower=1.0)).all()
    ex = results.explain_df
    assert (ex["method"] == "taylor").any() and (ex["factor"] == "trade").any()


def test_ko_book_with_zero_threshold_closes_the_hedge_and_reconciles():
    # After the KO the book's delta is exactly the hedge's: the executor nets it to zero.
    # Before the patch Portfolio.update_position raised "Quantity cannot be zero" here.
    results = BacktestEngine(_lifecycle_config(WF, delta_threshold=0.0)).run()
    events = results.get_lifecycle_events()
    ko_day = pd.Timestamp(events.index[0])
    trades = results.trades_df
    closes = trades[trades["trade_type"] == "close"]
    assert len(closes) == 1 and pd.Timestamp(closes.index[0]).normalize() == ko_day
    port = results.explain_reconciliation_df.query("level == 'portfolio'")
    assert port["ok"].all()
    assert (port["gap_states"].abs() <= 1e-8 * port["expected"].abs().clip(lower=1.0)).all()
    assert results.states_df["num_positions"].iloc[-1] == 0
```

`results.trades_df` is `StateTracker.get_trades_dataframe()`: one row per `TradeRecord.to_dict()`, indexed by `timestamp`, with a `trade_type` column (`quantark/backtest/equity/state.py:214-230`).

- [ ] **Step 6: Run the explain backtest tests**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_pnlexplain_backtest_equity.py -q`
Expected: all passed.

- [ ] **Step 7: Commit**

```bash
git add quantark/backtest/equity/hedge_executor.py test/test_hedge_executor.py test/test_pnlexplain_backtest_equity.py
git -c commit.gpgsign=false commit -F /path/to/msg_task1.txt
```

Message: `fix(backtest): simple HedgeExecutor books hedges at average cost and closes a hedge that nets to zero` + body naming the four branches, `realized_pnl`, the removed `gap_states` carve-out, and that adjust-day states P&L moves for delta-neutral backtests (no pinned numbers) + trailers.

---

### Task 2: Roll schedule-free float-maturity products (spec §4)

**Files:**
- Create: `quantark/asset/equity/lifecycle/float_roll.py`
- Modify: `quantark/asset/equity/lifecycle/manager.py:83-90` (`__init__`), `:137-156` (`pricing_products`), `:164-190` (`process_day`); `quantark/asset/equity/lifecycle/__init__.py`
- Modify: `quantark/backtest/equity/engine.py:201-215` (`_initialize` lifecycle block)
- Modify: `quantark/pnlexplain/equity/recorder.py:60-70` (docstring), `:145-149` (comment)
- Create: `test/test_float_maturity_roll.py`
- Modify: `test/test_backtest_lifecycle.py` (one new test), `test/test_pnlexplain_backtest_equity.py` (one new test)

**Interfaces:**
- Produces: `is_float_rollable(product) -> bool`, `has_unrolled_float_maturity(product) -> bool`, `FloatMaturityRoller` with `register(position_id, product, date)`, `rolled(position_id, product, date)`, `retain(position_ids)`; `PortfolioLifecycleManager.pricing_products` returns a **new** rolled object for rollable untracked positions; `process_day` assigns it.
- Consumes: `BaseEquityOption.maturity` (float), `Position.product` attribute assignment (as the trackers already do).

- [ ] **Step 1: Write the failing manager tests**

```python
# test/test_float_maturity_roll.py
"""Untracked schedule-free float-maturity products roll daily (patch spec 2026-09-03 §4)."""
import warnings
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import BlackScholesEngine, DeltaOneEngine
from quantark.asset.equity.lifecycle import PortfolioLifecycleManager
from quantark.asset.equity.lifecycle.float_roll import (
    FLOAT_ROLLABLE_PRODUCTS, MATURITY_FLOOR, FloatMaturityRoller, has_unrolled_float_maturity, is_float_rollable,
)
from quantark.asset.equity.product.deltaone import Futures
from quantark.asset.equity.product.option import AsianOption, EuropeanVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.portfolio import Portfolio
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType

U = "TEST"
START = datetime(2026, 1, 5)


def _env(date=START):
    return PricingEnvironment(spot_quote=SpotQuote(spot=100.0, asset_name=U), vol_surface=FlatVolSurface(0.2),
                              rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(0.01),
                              valuation_date=date)


def _book(*products):
    env = _env()
    portfolio = Portfolio(portfolio_name="t", pricing_environments={U: env}, creation_date=START)
    ids = []
    for p in products:
        engine = DeltaOneEngine() if isinstance(p, Futures) else BlackScholesEngine()
        ids.append(portfolio.add_position(product=p, quantity=1.0, entry_price=1.0, underlying=U, engine=engine,
                                          entry_timestamp=START).position_id)
    return portfolio, env, ids


def test_whitelist_and_unrolled_detection():
    call = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    dated = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, exercise_date=datetime(2027, 1, 5))
    fut = Futures(underlying=U, maturity=0.25)
    asian = AsianOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    assert EuropeanVanillaOption in FLOAT_ROLLABLE_PRODUCTS and Futures not in FLOAT_ROLLABLE_PRODUCTS
    assert is_float_rollable(call) and not is_float_rollable(dated) and not is_float_rollable(fut)
    assert not is_float_rollable(asian) and has_unrolled_float_maturity(asian)
    assert not has_unrolled_float_maturity(fut)          # static-maturity futures hedge is the documented design
    assert not has_unrolled_float_maturity(call)


def test_roller_rolls_from_first_sight_and_floors():
    r = FloatMaturityRoller()
    call = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=10 / 365)
    r.register("p", call, START)
    r.register("p", EuropeanVanillaOption(strike=1.0, option_type=OptionType.CALL, maturity=5.0), START)  # ignored
    day3 = r.rolled("p", call, datetime(2026, 1, 8))
    assert day3 is not call and day3.maturity == pytest.approx(7 / 365, abs=1e-15)
    assert call.maturity == 10 / 365                                        # input not mutated
    assert r.rolled("p", call, datetime(2026, 1, 15)).maturity == MATURITY_FLOOR
    assert r.rolled("p", call, datetime(2026, 2, 15)).maturity == MATURITY_FLOOR   # held at the floor
    r.retain(set())
    with pytest.raises(KeyError):
        r.rolled("p", call, START)


def test_manager_rolls_untracked_vanilla_but_not_futures_and_warns_once_for_schedule_products():
    call = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    fut = Futures(underlying=U, maturity=0.25)
    asian = AsianOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    portfolio, env, (pid_call, pid_fut, pid_asian) = _book(call, fut, asian)
    manager = PortfolioLifecycleManager(base_date=START)
    manager.register_positions(portfolio)                                   # nothing trackable here
    assert manager.num_tracked == 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        manager.process_day(portfolio, day_index=0, day_date=START)
        day10 = datetime(2026, 1, 15)
        env.valuation_date = day10
        before = manager.pricing_products(portfolio, day10)
        manager.process_day(portfolio, day_index=10, day_date=day10)
    assert portfolio.positions[pid_call].product.maturity == pytest.approx(1.0 - 10 / 365, abs=1e-15)
    assert before[pid_call].maturity == portfolio.positions[pid_call].product.maturity
    assert before[pid_call] is not portfolio.positions[pid_call].product   # fresh object per call
    assert portfolio.positions[pid_fut].product is fut and fut.maturity == 0.25
    assert portfolio.positions[pid_asian].product is asian and asian.maturity == 1.0
    msgs = [str(x.message) for x in w if "no roll rule" in str(x.message)]
    assert len(msgs) == 1 and "AsianOption" in msgs[0]


def test_ids_that_leave_the_book_are_forgotten():
    call = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    portfolio, env, (pid,) = _book(call)
    manager = PortfolioLifecycleManager(base_date=START)
    manager.process_day(portfolio, 0, START)
    portfolio.remove_position(pid)
    manager.process_day(portfolio, 1, datetime(2026, 1, 6))
    assert pid not in manager._float_roller._base
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_float_maturity_roll.py -q`
Expected: `ModuleNotFoundError: No module named 'quantark.asset.equity.lifecycle.float_roll'`.

- [ ] **Step 3: Create `float_roll.py`**

```python
"""Daily roll of schedule-free float-maturity contracts (patch spec 2026-09-03 §4).

A product whose only time coordinate is the scalar ``maturity`` ages by
``days / 365`` from the day its position enters the working portfolio, exactly
as the lifecycle trackers roll their products. Contracts with observation
schedules need a tracker (their timing must shift too), and a float-maturity
``Futures`` hedge is a constant-maturity proxy by design
(``FuturesHedgeInstrument``), so neither is rolled here.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Tuple

import pandas as pd

from quantark.asset.equity.product.deltaone.base_deltaone_product import BaseDeltaOneProduct
from quantark.asset.equity.product.option.american_option import AmericanOption
from quantark.asset.equity.product.option.digital_option import CashOrNothingDigitalOption
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption

FLOAT_ROLLABLE_PRODUCTS = (EuropeanVanillaOption, AmericanOption, CashOrNothingDigitalOption)
MATURITY_FLOOR = 1e-8          # the trackers' floor (AutocallableLifecycleTracker / BarrierLifecycleTracker)


def _float_only(product: Any) -> bool:
    return (getattr(product, "exercise_date", None) is None
            and getattr(product, "maturity_date", None) is None
            and getattr(product, "maturity", None) is not None)


def is_float_rollable(product: Any) -> bool:
    """Whitelisted class with a float maturity and no dates."""
    return isinstance(product, FLOAT_ROLLABLE_PRODUCTS) and _float_only(product)


def has_unrolled_float_maturity(product: Any) -> bool:
    """Float maturity, no dates, and no roll rule: repriced with a constant maturity.

    Delta-one products (the futures hedge is static by design) and the KO-reset
    snowball (already warned at registration) are excluded from the warning.
    """
    if is_float_rollable(product) or isinstance(product, (BaseDeltaOneProduct, KnockOutResetSnowballOption)):
        return False
    return _float_only(product)


class FloatMaturityRoller:
    """Per-position (base_date, m0); first sight wins."""

    def __init__(self) -> None:
        self._base: Dict[str, Tuple[pd.Timestamp, float]] = {}

    def register(self, position_id: str, product: Any, date) -> None:
        if position_id not in self._base:
            self._base[position_id] = (pd.Timestamp(date).normalize(), float(product.maturity))

    def rolled(self, position_id: str, product: Any, date) -> Any:
        base_date, m0 = self._base[position_id]
        elapsed_days = max(0, (pd.Timestamp(date).normalize() - base_date).days)
        copy = deepcopy(product)
        copy.maturity = max(MATURITY_FLOOR, m0 - elapsed_days / 365.0)
        return copy

    def retain(self, position_ids: Iterable[str]) -> None:
        keep = set(position_ids)
        for pid in [p for p in self._base if p not in keep]:
            del self._base[pid]
```

- [ ] **Step 4: Wire the manager**

In `manager.py` add the import `from .float_roll import FloatMaturityRoller, has_unrolled_float_maturity, is_float_rollable`; in `__init__` add `self._float_roller = FloatMaturityRoller()` and `self._unrolled_warned: set = set()`; add the helper:

```python
    def _roll_untracked(self, position_id: str, position, date: pd.Timestamp):
        """Rolled copy of a schedule-free float-maturity product, else None.

        Untracked products with a float maturity but no roll rule are repriced with
        a constant maturity; warn once per position so the choice is visible.
        """
        product = position.product
        if not is_float_rollable(product):
            if has_unrolled_float_maturity(product) and position_id not in self._unrolled_warned:
                self._unrolled_warned.add(position_id)
                warnings.warn(
                    f"position {position_id} ({type(product).__name__}) has a float maturity and "
                    "no roll rule; it is repriced with a constant maturity every day",
                    UserWarning,
                )
            return None
        self._float_roller.register(position_id, product, date)
        return self._float_roller.rolled(position_id, product, date)
```

In `pricing_products`, replace the `else: out[position_id] = position.product` branch with:

```python
            else:
                rolled = self._roll_untracked(position_id, position, date)
                out[position_id] = position.product if rolled is None else rolled
```

In `process_day`, add after the barrier branch:

```python
            else:
                rolled = self._roll_untracked(position_id, position, date)
                if rolled is not None:
                    position.product = rolled
```

and before `self._revalue_ledger(portfolio, date)`: `self._float_roller.retain(portfolio.positions.keys())`. Update the module docstring (the manager also ages schedule-free float contracts). Export `FloatMaturityRoller`, `FLOAT_ROLLABLE_PRODUCTS`, `is_float_rollable` from `lifecycle/__init__.py`.

- [ ] **Step 5: Run the manager tests**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_float_maturity_roll.py -q`
Expected: 4 passed.

- [ ] **Step 6: Make the engine construct the manager unconditionally**

In `quantark/backtest/equity/engine.py` `_initialize`, replace the lifecycle block:

```python
        # The lifecycle manager always exists: it also rolls schedule-free
        # float-maturity contracts (patch spec 2026-09-03 §4). Lifecycle TRACKERS
        # (KO/KI/coupon/maturity/expiry) attach only when handle_lifecycle_events
        # is on; hedge instruments added later stay untracked either way.
        self.lifecycle_manager = PortfolioLifecycleManager(base_date=self.config.start_date)
        if self.config.handle_lifecycle_events:
            self.lifecycle_manager.register_positions(self.portfolio)
            if self.lifecycle_manager.num_tracked > 0:
                self.logger.logger.info(
                    f"Lifecycle tracking enabled for "
                    f"{self.lifecycle_manager.num_tracked} position(s)"
                )
```

Leave the `if self.lifecycle_manager is not None` guards in `_step`, `_record_state`, `_finalize` as they are (harmless).

- [ ] **Step 7: Add the engine-level tests**

In `test/test_backtest_lifecycle.py` inside `TestBacktestLifecycle`:

```python
    def test_disabled_flag_still_rolls_schedule_free_products(self):
        from quantark.asset.equity.engine.analytical import BlackScholesEngine
        from quantark.asset.equity.product.option import EuropeanVanillaOption

        config = _make_config(handle_lifecycle_events=False)
        vanilla = Position(
            product=EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0),
            quantity=1.0, entry_price=5.0, underlying=UNDERLYING, engine=BlackScholesEngine(),
            entry_timestamp=START,
        )
        config.initial_positions.append(vanilla)
        engine = BacktestEngine(config)
        engine.run()
        held = {type(p.product).__name__: p.product for p in engine.portfolio.positions.values()}
        days = (config.end_date - START).days
        assert held["EuropeanVanillaOption"].maturity == pytest.approx(1.0 - days / 365.0, abs=1e-15)
        assert held["BarrierOption"].maturity == 1.0          # untracked barrier: no roll rule, unchanged
```

(add `import pytest` at the top). In `test/test_pnlexplain_backtest_equity.py`:

```python
def test_untracked_vanilla_is_rolled_and_the_time_row_carries_contract_theta():
    cfg = make_multi_config(DeltaNeutralStrategy(delta_threshold=1e12))       # no hedging: one position
    cfg.pnl_explain = PnLExplainConfig()
    results = BacktestEngine(cfg).run()
    ex = results.explain_df
    pos = ex[ex["level"] == "position"]
    contract_theta = pos[(pos["term"] == "theta_contract") & (pos["method"] == "taylor")]
    assert len(contract_theta) == len(results.states_df) - 1
    assert (contract_theta["pnl"] != 0.0).all()                              # rolled: the contract ages
    port = results.explain_reconciliation_df.query("level == 'portfolio'")
    assert port["ok"].all()
```

- [ ] **Step 8: Run the affected suites**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_float_maturity_roll.py test/test_backtest_lifecycle.py test/test_pnlexplain_backtest_equity.py test/test_multi_greek_backtest.py test/test_backtest_interface.py test/test_dynamic_scenario*.py -q`
Expected: all passed. If a dynamic-scenario test pins a value for a float-maturity vanilla, report it (do not re-pin) — the spec expects none.

- [ ] **Step 9: Update the recorder comments**

In `quantark/pnlexplain/equity/recorder.py` change the `_is_unrolled_float_contract` docstring to "True when the holder hands back the SAME float-maturity contract, unrolled (Futures hedges by design; schedule-bearing untracked products); schedule-free contracts are rolled by the manager and arrive as new objects." and the `roll_days` comment in `begin_day` accordingly.

- [ ] **Step 10: Commit**

```bash
git add quantark/asset/equity/lifecycle/float_roll.py quantark/asset/equity/lifecycle/manager.py quantark/asset/equity/lifecycle/__init__.py quantark/backtest/equity/engine.py quantark/pnlexplain/equity/recorder.py test/test_float_maturity_roll.py test/test_backtest_lifecycle.py test/test_pnlexplain_backtest_equity.py
git -c commit.gpgsign=false commit -F /path/to/msg_task2.txt
```

Message: `fix(lifecycle): roll schedule-free float-maturity contracts daily in the shared manager; equity backtest always constructs it` + body (whitelist, first-sight base, floor, Futures excluded by design, warning, numbers move for float-maturity vanilla backtests) + trailers.

---

### Task 3: Constant-lag first-hit payments in the analytical engines (spec §5)

**Files:**
- Modify: `quantark/asset/equity/engine/settlement_support.py` (helper + `__all__`)
- Modify: `quantark/asset/equity/engine/analytical/one_touch_analytical_engine.py:89-100, 160-180, 207-222, 231-262`
- Modify: `quantark/asset/equity/engine/analytical/barrier_analytical_engine.py:92-104`
- Modify: `quantark/asset/equity/engine/analytical/single_sharkfin_option_analytical_engine.py:116-126`
- Modify: `quantark/asset/equity/engine/analytical/double_sharkfin_option_analytical_engine.py:168-178, 304-348`
- Modify: `test/test_barrier_family_settlement.py:105-118` (+ new tests), `test/test_backtest_lifecycle.py` (sibling test), `test/test_pnlexplain_backtest_equity.py:27-49` (fixture)

**Interfaces:**
- Produces: `constant_hit_lag_year_fraction(product, pricing_env) -> float` (0.0 = pays at hit; raises `CapabilityError` with "first-hit" in the message).
- Consumes: `SettlementConvention` (`lag`, `lag_unit`, `business_day_convention`), `DayCountConvention`, `BusinessDayConvention`.

- [ ] **Step 1: Write the failing tests**

In `test/test_barrier_family_settlement.py` replace `test_continuous_first_hit_analytical_rejects_unrepresentable_lag` with:

```python
import math
from quantark.asset.equity.engine.pde import BarrierPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.util.calendar import BusinessDayConvention, DayCountConvention


def _one_touch(convention, **kw):
    return OneTouchOption(barrier=120.0, barrier_direction=BarrierDirection.UP, maturity=MATURITY, rebate=10.0,
                          payment_at_hit=True, touch_type=TouchType.ONE_TOUCH,
                          observation_type=ObservationType.CONTINUOUS, settlement_convention=convention, **kw)


def _up_out_call(convention):
    return BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, barrier_type=BarrierType.UP_OUT,
                         maturity=MATURITY, rebate=10.0, pay_at_hit=True, observation_type=ObservationType.CONTINUOUS,
                         settlement_convention=convention)


def test_continuous_first_hit_constant_lag_scales_by_exp_minus_r_lag(env):
    """E[e^{-r(tau+L)} 1{tau<=T}] = e^{-rL} E[e^{-r tau} 1{tau<=T}] under the formula's flat r (spec §5.2)."""
    r_T = env.get_rate(MATURITY)
    for engine, make in ((OneTouchAnalyticalEngine(), _one_touch), (BarrierAnalyticalEngine(), _up_out_call)):
        immediate = engine.price(make(None), env)
        delayed = engine.price(make(_lagged()), env)
        assert delayed != immediate
        # the rebate leg alone carries the factor; isolate it with rebate-free twins
        assert delayed == pytest.approx(immediate * math.exp(-r_T * LAG), rel=2e-12) or isinstance(engine, BarrierAnalyticalEngine)
    # barrier: option leg unchanged, rebate leg scaled
    plain, lagged = _up_out_call(None), _up_out_call(_lagged())
    no_rebate = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, barrier_type=BarrierType.UP_OUT,
                              maturity=MATURITY, rebate=0.0, observation_type=ObservationType.CONTINUOUS)
    eng = BarrierAnalyticalEngine()
    option_leg = eng.price(no_rebate, env)
    rebate_leg = eng.price(plain, env) - option_leg
    assert eng.price(lagged, env) == pytest.approx(option_leg + rebate_leg * math.exp(-r_T * LAG), rel=2e-12)


def test_calendar_day_lag_is_constant_only_under_act_style_day_counts(env):
    unadjusted = SettlementConvention(lag=2, lag_unit=SettlementLagUnit.CALENDAR_DAYS,
                                      business_day_convention=BusinessDayConvention.UNADJUSTED)
    r_T = env.get_rate(MATURITY)
    engine = OneTouchAnalyticalEngine()
    immediate = engine.price(_one_touch(None), env)
    assert engine.price(_one_touch(unadjusted), env) == pytest.approx(immediate * math.exp(-r_T * 2 / 365), rel=2e-12)
    env360 = deepcopy(env); env360.day_count_convention = DayCountConvention.ACT_360
    assert engine.price(_one_touch(unadjusted), env360) == pytest.approx(
        engine.price(_one_touch(None), env360) * math.exp(-env360.get_rate(MATURITY) * 2 / 360), rel=2e-12)
    env_isda = deepcopy(env); env_isda.day_count_convention = DayCountConvention.ACT_ACT_ISDA
    with pytest.raises(CapabilityError, match="first-hit"):
        engine.price(_one_touch(unadjusted), env_isda)
    following = SettlementConvention(lag=2, lag_unit=SettlementLagUnit.CALENDAR_DAYS)     # FOLLOWING adjusts
    with pytest.raises(CapabilityError, match="first-hit"):
        engine.price(_one_touch(following), env)
    with pytest.raises(CapabilityError, match="first-hit"):
        engine.price(_one_touch(SettlementConvention(lag=2, lag_unit=SettlementLagUnit.BUSINESS_DAYS)), env)


def test_per_record_settlement_timing_still_rejected(env):
    schedule = ObservationSchedule(records=[ObservationRecord(observation_time=0.5, settlement_time=0.6, barrier=120.0,
                                                              payoff=10.0)],
                                   aggregation_mode=ObservationAggregation.STOP_FIRST_HIT)
    product = OneTouchOption(barrier=120.0, barrier_direction=BarrierDirection.UP, maturity=MATURITY, rebate=10.0,
                             payment_at_hit=True, touch_type=TouchType.ONE_TOUCH,
                             observation_type=ObservationType.DISCRETE, observation_schedule=schedule)
    with pytest.raises(CapabilityError, match="first-hit"):
        OneTouchAnalyticalEngine().price(product, env)


def test_pde_lag_effect_matches_analytical_scaling():
    """Flat rate: the lag EFFECT (immediate - delayed) agrees across engines; grid error cancels."""
    flat = PricingEnvironment(spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(volatility=0.20),
                              rate_curve=FlatRateCurve(rate=0.05), div_yield=ContinuousDividendYield(div_yield=0.0),
                              valuation_date=datetime(2026, 1, 1))
    lag = SettlementConvention(lag=0.5, lag_unit=SettlementLagUnit.YEAR_FRACTION)
    an = BarrierAnalyticalEngine()
    pde = BarrierPDESolver(PDEParams(accuracy="high"))     # declarative grid layer: legacy grid_size/time_steps are rejected
    d_an = an.price(_up_out_call(None), flat) - an.price(_up_out_call(lag), flat)
    d_pde = pde.price(_up_out_call(None), flat) - pde.price(_up_out_call(lag), flat)
    assert d_an > 0.0
    assert d_pde == pytest.approx(d_an, rel=5e-2)      # record the measured values in this comment once green
```

Add `from copy import deepcopy` and `from quantark.param import FlatRateCurve, PricingEnvironment` imports where missing (check the file's import block). Keep the single-sharkfin and double-sharkfin assertions in the same style:

```python
def _single_sharkfin(convention, knock_out_rebate=2.0):
    return SingleSharkfinOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, maturity=MATURITY,
                                participation_rate=1.0, knock_out_rebate=knock_out_rebate, no_hit_rebate=0.0,
                                pay_at_hit=True, observation_type=ObservationType.CONTINUOUS,
                                settlement_convention=convention)


def _double_sharkfin(convention, knock_out_rebate=2.0):
    return DoubleSharkfinOption(strike=100.0, option_type=OptionType.CALL, upper_barrier=130.0, lower_barrier=70.0,
                                maturity=MATURITY, participation_rate=1.0, knock_out_rebate=knock_out_rebate,
                                no_hit_rebate=0.0, pay_at_hit=True, observation_type=ObservationType.CONTINUOUS,
                                settlement_convention=convention)


@pytest.mark.parametrize("engine,make", [(SingleSharkfinOptionAnalyticalEngine(), _single_sharkfin),
                                         (DoubleSharkfinOptionAnalyticalEngine(), _double_sharkfin)])
def test_sharkfin_first_hit_constant_lag_scales(env, engine, make):
    """Only the hit-paid cash leg carries exp(-r L); the option leg is untouched."""
    r_T = env.get_rate(MATURITY)
    base = engine.price(make(None), env)
    lagged = engine.price(make(_lagged()), env)
    no_cash = engine.price(make(None, knock_out_rebate=0.0), env)
    assert base > no_cash                                             # the cash leg is material
    assert lagged == pytest.approx(no_cash + (base - no_cash) * math.exp(-r_T * LAG), rel=2e-12)
```

(`no_hit_rebate=0.0` on both twins so the only difference is the hit-paid leg; `SingleSharkfinOption` / `DoubleSharkfinOption` constructor arguments as used at lines 266-295 of the same file.)

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_barrier_family_settlement.py -q -k "constant_lag or calendar_day_lag or per_record or pde_lag or sharkfin_first_hit"`
Expected: the scaling tests fail with `CapabilityError ... first-hit`; the per-record test passes already (keep it as the pinned rejection).

- [ ] **Step 3: Add the helper to `settlement_support.py`**

```python
from quantark.util.calendar import BusinessDayConvention, DayCountConvention

# day counts under which n calendar days is the same year fraction from every start date
_CONSTANT_DAY_BASIS = {
    DayCountConvention.CALENDAR_DAYS: 365.0,
    DayCountConvention.ACT_365: 365.0,
    DayCountConvention.ACT_360: 360.0,
}


def _schedule_requests_delayed_hit(product) -> bool:
    schedule = getattr(product, "observation_schedule", None)
    if schedule is None:
        return False
    return any(
        record.settlement_date is not None
        or (
            record.settlement_time is not None
            and record.observation_time is not None
            and record.settlement_time != record.observation_time
        )
        for record in schedule.records
    )


def constant_hit_lag_year_fraction(product, pricing_env: "PricingEnvironment") -> float:
    """Year-fraction lag of a first-hit payment when it is the same for EVERY hit time.

    A first-passage formula can carry a delayed hit payment only as the exact
    factor exp(-r * L) with one constant L (settlement spec, "Mixed-event
    formulas"). Returns 0.0 when the contract pays at the hit. Raises
    CapabilityError when the lag depends on the hit date: business-day lags,
    calendar-day lags with a business-day adjustment, calendar-day lags under a
    start-date-dependent day count, and per-observation settlement timing.
    """
    name = type(product).__name__
    if _schedule_requests_delayed_hit(product):
        raise CapabilityError(
            f"{name}: per-observation settlement timing makes the first-hit payment lag "
            "hit-date dependent; analytical first-hit formulas need one constant lag "
            "(use MC, PDE or QUAD)"
        )
    convention = getattr(product, "settlement_convention", None)
    if convention is None or float(convention.lag) == 0.0:
        return 0.0
    if convention.lag_unit is SettlementLagUnit.YEAR_FRACTION:
        return float(convention.lag)
    if convention.lag_unit is SettlementLagUnit.CALENDAR_DAYS:
        if convention.business_day_convention is not BusinessDayConvention.UNADJUSTED:
            raise CapabilityError(
                f"{name}: a business-day-adjusted calendar-day first-hit lag is hit-date "
                "dependent (use MC, PDE or QUAD)"
            )
        basis = _CONSTANT_DAY_BASIS.get(pricing_env.day_count_convention)
        if basis is None:
            raise CapabilityError(
                f"{name}: a {int(convention.lag)}-calendar-day first-hit lag is not a constant "
                f"year fraction under {pricing_env.day_count_convention.value} (use MC, PDE or QUAD)"
            )
        return float(convention.lag) / basis
    raise CapabilityError(
        f"{name}: a business-day first-hit lag is hit-date dependent (use MC, PDE or QUAD)"
    )
```

Add `"constant_hit_lag_year_fraction"` to `__all__`.

- [ ] **Step 4: One-touch engine**

In `price()` replace the guard (lines 89-100) with:

```python
        hit_lag = 0.0
        if (
            pay_at_hit
            and product.observation_type != ObservationType.EXPIRY
            and not product.is_barrier_hit(spot)
        ):
            hit_lag = constant_hit_lag_year_fraction(product, pricing_env)
```

Pass `hit_lag=hit_lag` into the `_one_touch_price(...)` call on the not-yet-hit path (line ~160) and change the method:

```python
    def _one_touch_price(self, spot, barrier, maturity, rate, div, vol, rebate, pay_at_hit, is_up,
                         hit_lag: float = 0.0) -> float:
        """Closed-form one-touch price for continuous or shifted discrete barriers.

        ``hit_lag`` is the constant year-fraction delay of a hit-paid rebate; under the
        formula's flat rate the delayed value is exactly exp(-rate * hit_lag) times the
        pay-at-hit value. 0.0 leaves the unlagged path bitwise unchanged.
        """
        if pay_at_hit:
            value = rebate * self._instant_touch_term(spot=spot, barrier=barrier, maturity=maturity, rate=rate,
                                                     div=div, vol=vol, is_up=is_up)
            if hit_lag != 0.0:
                value *= math.exp(-rate * hit_lag)
            return value
        ...
```

Delete `_requests_delayed_hit_payment` after the other engines stop calling it. Import the helper from `settlement_support`.

- [ ] **Step 5: Barrier and single sharkfin engines**

Barrier (`:92-104`) becomes:

```python
        if (
            product.is_knock_out
            and product.pay_at_hit
            and product.rebate > 0
            and product.observation_type != ObservationType.EXPIRY
            and not product.is_barrier_hit(spot)
        ):
            # fail closed BEFORE pricing; the rebate leg (one-touch engine) applies the factor
            constant_hit_lag_year_fraction(product, pricing_env)
```

Single sharkfin (`:116-126`): the same replacement with its own condition (`product.pay_at_hit`, `knock_out_rebate > 0`, monitored, not hit). Import the helper; drop the `_one_touch_engine._requests_delayed_hit_payment` reference.

- [ ] **Step 6: Double sharkfin engine**

Replace the guard (`:168-178`) with `hit_lag = constant_hit_lag_year_fraction(product, pricing_env) if (product.observation_type == ObservationType.CONTINUOUS and product.pay_at_hit and product.knock_out_rebate > 0.0) else 0.0`, pass `hit_lag` to `_price_knock_out_cash_leg`, and in its continuous branch multiply: `hit_discount_factor = self._continuous_hit_discount_factor(...) * (math.exp(-rate * hit_lag) if hit_lag != 0.0 else 1.0)`. Leave the discrete branch alone (it already discounts each node at its resolved `settlement_time`); add a test that a discrete double sharkfin with `_lagged()` prices as the per-node delay says (build the per-node expectation from `schedule.resolve(...)` records: Σ first_hit_prob·DF(settlement_time) with the same survival probabilities — or, simpler and still exact, assert that with the lag the discrete cash leg equals the unlagged leg times `DF(t_i+L)/DF(t_i)` when the schedule has ONE record).

- [ ] **Step 7: Run the settlement suites**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_barrier_family_settlement.py test/test_barrier_numerical_settlement.py test/test_engine_settlement_capabilities.py test/test_barrier_analytical_engine.py test/test_fx_one_touch.py -q`
Expected: all passed. Record the measured `d_an` / `d_pde` in the PDE test's comment.

- [ ] **Step 8: Backtest fixtures on the analytical engine**

`test/test_backtest_lifecycle.py`: add

```python
    def test_delayed_settlement_prices_on_the_analytical_engine(self):
        convention = SettlementConvention(lag=2, lag_unit=SettlementLagUnit.CALENDAR_DAYS,
                                          business_day_convention=BusinessDayConvention.UNADJUSTED)
        engine = BacktestEngine(_make_config(settlement_convention=convention))      # default engine, greeks on
        results = engine.run()
        states = results.states_df
        event_date = results.get_lifecycle_events().index[0]
        assert is_close(states.loc[event_date, "pending_receivable_pv"], 20.0 * safe_exp(-0.05 * 2.0 / 365.0))
        assert is_close(states.loc[event_date + pd.Timedelta(days=2), "paid_cash"], 20.0)
```

`test/test_pnlexplain_backtest_equity.py` `_lifecycle_config`: the `lagged` branch uses `_down_out_put_position(2.0, 10.0, settlement_convention=convention)` (analytical engine) and `calculate_greeks=True`; update its docstring.

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_backtest_lifecycle.py test/test_pnlexplain_backtest_equity.py -q`
Expected: all passed.

- [ ] **Step 9: Commit**

```bash
git add quantark/asset/equity/engine/settlement_support.py quantark/asset/equity/engine/analytical/one_touch_analytical_engine.py quantark/asset/equity/engine/analytical/barrier_analytical_engine.py quantark/asset/equity/engine/analytical/single_sharkfin_option_analytical_engine.py quantark/asset/equity/engine/analytical/double_sharkfin_option_analytical_engine.py test/test_barrier_family_settlement.py test/test_backtest_lifecycle.py test/test_pnlexplain_backtest_equity.py
git -c commit.gpgsign=false commit -F /path/to/msg_task3.txt
```

Message: `feat(analytical): constant first-hit settlement lags price exactly as exp(-r L) in the one-touch / barrier / sharkfin formulas` + body (helper rules, what still raises, PDE cross-check numbers) + trailers.

---

### Task 4: Dividend-held key-rate rho (spec §6)

**Files:**
- Modify: `quantark/asset/equity/riskmeasures/bucketed_greeks.py:22-60`, `quantark/asset/equity/riskmeasures/bucketed_coordinates/rate_keyrate.py`, `quantark/asset/equity/riskmeasures/greeks_calculator.py:316-330`
- Modify: `quantark/pnlexplain/equity/bucketed.py`, `quantark/pnlexplain/equity/taylor.py:236-247`
- Modify: `test/test_rate_keyrate_buckets.py`, `test/test_pnlexplain_bucketed.py:38-72`

**Interfaces:**
- Produces: `RateKeyrateConvention` enum; `BucketedGreeksRequest.rate_keyrate_convention` (default `CARRY_INVARIANT`); point metadata `convention`; result metadata `rate_keyrate_convention`.
- Consumes (explain): `bucketed_rows(cache, calc, bump, level) -> (rows_by_factor, covered)` with `Factor.RATE` now in `covered` when rate rows exist.

- [ ] **Step 1: Failing riskmeasures test**

Append to `test/test_rate_keyrate_buckets.py`:

```python
def test_rate_keyrate_dividend_held_is_the_forward_moving_sensitivity():
    import math
    from datetime import datetime
    from scipy.stats import norm
    from quantark.asset.equity.engine.analytical import BlackScholesEngine
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.asset.equity.riskmeasures.bucketed_greeks import RateKeyrateConvention
    from quantark.param import FlatVolSurface, SpotQuote
    from quantark.param.div import ContinuousDividendYield
    from quantark.param.rrf.rate_curve import LinearRateCurve
    from quantark.priceenv import PricingEnvironment
    from quantark.util.enum import OptionType

    S, K, T, r, q, sig = 100.0, 100.0, 1.0, 0.03, 0.01, 0.2
    env = PricingEnvironment(spot_quote=SpotQuote(spot=S), vol_surface=FlatVolSurface(sig),
                             rate_curve=LinearRateCurve([(0.5, r), (1.0, r), (2.0, r)]),
                             div_yield=ContinuousDividendYield(q), valuation_date=datetime(2026, 1, 5))
    call = EuropeanVanillaOption(strike=K, option_type=OptionType.CALL, maturity=T)
    calc = GreeksCalculator()

    def run(conv):
        return calc.calculate_bucketed_greeks(call, env, BlackScholesEngine(), request=BucketedGreeksRequest(
            coordinates=(BucketedGreekCoordinate.RATE_KEYRATE,), rate_keyrate_convention=conv))

    held = run(RateKeyrateConvention.DIVIDEND_HELD)
    carry = run(RateKeyrateConvention.CARRY_INVARIANT)
    par_held = [p for p in held.points if p.name == "rate_keyrate.parallel"][0]
    par_carry = [p for p in carry.points if p.name == "rate_keyrate.parallel"][0]
    d2 = (math.log(S / K) + (r - q - 0.5 * sig**2) * T) / (sig * math.sqrt(T))
    bs_rho = K * T * math.exp(-r * T) * norm.cdf(d2)                 # dV/dr, dividend held
    assert par_held.derivative == pytest.approx(bs_rho, rel=1e-6)
    assert par_held.derivative > 0.0 > par_carry.derivative           # the two conventions differ in sign for a call
    assert all(p.metadata["convention"] == "dividend_held" for p in held.points)
    assert all(p.metadata["convention"] == "carry_invariant" for p in carry.points)
    assert held.metadata["rate_keyrate_convention"] == "dividend_held"
    # default request == carry-invariant, bitwise
    default = calc.calculate_bucketed_greeks(call, env, BlackScholesEngine(),
                                             request=BucketedGreeksRequest(coordinates=(BucketedGreekCoordinate.RATE_KEYRATE,)))
    assert [p.reported for p in default.points] == [p.reported for p in carry.points]
```

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_rate_keyrate_buckets.py -q -k dividend_held`
Expected: `ImportError: cannot import name 'RateKeyrateConvention'`.

- [ ] **Step 2: Implement the convention**

`bucketed_greeks.py`, after `BucketedGreekDifferenceMode`:

```python
class RateKeyrateConvention(Enum):
    """What a zero-rate pillar bump holds fixed.

    CARRY_INVARIANT: the forward F(0,T) is held, the dividend yield is re-derived
        pointwise, so the bump is pure discounting (desk default, spec WP3.3).
    DIVIDEND_HELD: the dividend yield is held, so the forward moves with the rate;
        this is the sensitivity a rate-curve replacement with q fixed measures
        (the PnL-explain factor model).
    """

    CARRY_INVARIANT = "carry_invariant"
    DIVIDEND_HELD = "dividend_held"
```

`BucketedGreeksRequest`: append `rate_keyrate_convention: RateKeyrateConvention = RateKeyrateConvention.CARRY_INVARIANT` **after** `allow_partial` (positional construction must keep its slots) and validate in `__post_init__`:

```python
        if not isinstance(self.rate_keyrate_convention, RateKeyrateConvention):
            raise ValidationError("rate_keyrate_convention must be a RateKeyrateConvention")
```

`rate_keyrate.py`:

```python
    convention = request.rate_keyrate_convention
    rebuild_rule = {
        RateKeyrateConvention.CARRY_INVARIANT: "zero-rate pillar bump; carry-invariant q re-derivation (F unchanged) -> pure discounting",
        RateKeyrateConvention.DIVIDEND_HELD: "zero-rate pillar bump; dividend yield held (F moves with r)",
    }[convention]

    def _rate_bumped_env(bumped_curve):
        env = deepcopy(pricing_env)
        env.rate_curve = bumped_curve
        if convention is RateKeyrateConvention.DIVIDEND_HELD:
            return env
        # carry-invariant (unchanged): ... existing CarryInvariantDividendYield wrap ...
```

Add `"convention": convention.value` to every point's metadata and use `rebuild_rule` (parallel: `"ParallelShiftRateCurve; " + rebuild_rule`). In `greeks_calculator.py` result metadata add `"rate_keyrate_convention": request.rate_keyrate_convention.value`. Export `RateKeyrateConvention` wherever `BucketedGreeksRequest` is exported (`riskmeasures/__init__.py`).

Run the whole key-rate file: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_rate_keyrate_buckets.py test/test_bucketed_greeks*.py -q`
Expected: all passed (existing tests untouched).

- [ ] **Step 3: Failing explain test**

In `test/test_pnlexplain_bucketed.py` rewrite the rho half of `test_bucket_rows_replace_scalar_vega_and_reconcile` and add a parallel-move test:

```python
    # rate buckets (dividend-held) take the scalar rho slot, exactly as tenor vega does
    assert "rho" not in terms
    assert bucketed.metadata["bucketed_factors"] == ("rate", "vol")
    kr = [r for r in bucketed.rows if r.term.startswith("rate_keyrate.")]
    assert [r.term for r in kr] == ["rate_keyrate.0.5", "rate_keyrate.1", "rate_keyrate.2", "rate_keyrate.parallel"]
    assert all(r.kind is RowKind.COMPONENT and r.metadata["convention"] == "dividend_held" for r in kr[:-1])
    par = kr[-1]
    assert par.kind is RowKind.INFORMATIONAL and "sum_of_buckets" in par.metadata
    # the standard stencil is delta, vega, theta, rho, dividend_rho, ...: the rate buckets sit in rho's slot,
    # after theta and its informational sub-rows and before dividend_rho
    assert terms.index("theta") < terms.index("rate_keyrate.0.5") < terms.index("dividend_rho")
    scalar_rho = [r for r in scalar.rows if r.term == "rho"][0]
    assert par.moves["rate_pct"] == pytest.approx(scalar_rho.moves["rate_pct"], rel=1e-12)
    for term in ("delta", "gamma", "theta"):
        a = [r for r in scalar.rows if r.term == term][0].pnl
        b = [r for r in bucketed.rows if r.term == term][0].pnl
        assert a == pytest.approx(b, abs=1e-12)


def test_parallel_rate_move_buckets_sum_to_the_scalar_rho_pnl():
    s0, s1 = _snaps(_env(101.0, T1, (0.21, 0.225, 0.24), (0.031, 0.033, 0.035)))    # +10bp everywhere
    scalar = explain(s0, s1, config=PnLExplainConfig(greeks_method="numerical"))
    bucketed = explain(s0, s1, config=PnLExplainConfig(greeks_method="numerical", bucketed=True))
    rho_pnl = [r for r in scalar.rows if r.term == "rho"][0].pnl
    buckets = [r for r in bucketed.rows if r.term.startswith("rate_keyrate.") and r.kind is RowKind.COMPONENT]
    assert sum(r.pnl for r in buckets) == pytest.approx(rho_pnl, rel=5e-2)
    assert bucketed.reconcile(ExplainMethod.TAYLOR) == pytest.approx(0.0, abs=1e-12)
```

- [ ] **Step 4: Implement in the explain**

`bucketed.py`: import `RateKeyrateConvention`; `RATE_CONVENTION = RateKeyrateConvention.DIVIDEND_HELD`; request `BucketedGreeksRequest(coordinates=..., vol_bump=..., rate_bump=..., rate_keyrate_convention=RATE_CONVENTION)`; pillar rate rows `kind=RowKind.COMPONENT` with `"convention": RATE_CONVENTION.value`; parallel row stays INFORMATIONAL; `covered = frozenset(f for f in (Factor.VOL, Factor.RATE) if f in rows)`. Rewrite the module docstring: the dividend-held pillar bumps ARE the factor model's rate sensitivity split by pillar.

`taylor.py`: replace the `if name == "rho" and Factor.RATE in bucket_rows:` block with a `covered` check placed next to the vega one:

```python
        if name == "rho" and Factor.RATE in covered:
            # dividend-held key-rate buckets take the scalar rho slot exactly once (spec §7.6 as amended)
            for row in bucket_rows[Factor.RATE]:
                rows.append(row)
                if row.kind is RowKind.COMPONENT:
                    explained += row.pnl
            continue
```

and delete the old informational append after the scalar rho row.

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_pnlexplain_bucketed.py test/test_pnlexplain_taylor.py test/test_pnlexplain_matrix.py -q`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/riskmeasures/bucketed_greeks.py quantark/asset/equity/riskmeasures/bucketed_coordinates/rate_keyrate.py quantark/asset/equity/riskmeasures/greeks_calculator.py quantark/asset/equity/riskmeasures/__init__.py quantark/pnlexplain/equity/bucketed.py quantark/pnlexplain/equity/taylor.py test/test_rate_keyrate_buckets.py test/test_pnlexplain_bucketed.py
git -c commit.gpgsign=false commit -F /path/to/msg_task4.txt
```

Message: `feat(riskmeasures,pnlexplain): opt-in dividend-held key-rate rho; explain rate buckets become components` + trailers.

---

### Task 5: Semantic MODEL identity (spec §7)

**Files:**
- Modify: `quantark/asset/equity/engine/base_engine.py:77-100`; opt-in engines: `black_scholes_engine.py`, `deltaone_engine.py`, `digital_option_engine.py`, `barrier_analytical_engine.py`, `one_touch_analytical_engine.py`, `double_barrier_option_engine.py`, `single_sharkfin_option_analytical_engine.py`, `double_sharkfin_option_analytical_engine.py`, `american_option_engine.py`
- Modify: `quantark/pnlexplain/equity/fingerprints.py`, `factor_diff.py:180`, `lifecycle.py:129-133`, `explain.py:80-88`
- Modify: `test/test_pnlexplain_factor_diff.py:64-67`, `test/test_pnlexplain_lifecycle.py:145-147`, `test/test_pnlexplain_scenario.py` (+1 test); create `test/test_engine_model_fingerprint.py`

**Interfaces:**
- Produces: `BaseEngine.MODEL_FINGERPRINT_ATTRS: Optional[Tuple[str, ...]] = None`; `BaseEngine.model_fingerprint() -> Optional[tuple]`; `quantark.pnlexplain.equity.fingerprints.engines_equivalent(a, b) -> bool`.

- [ ] **Step 1: Verification script (no guessing which engines are params-only)**

Write `<scratchpad>/engine_attrs.py` that instantiates each candidate engine with defaults and prints `sorted(vars(e))`. Run it. An engine may opt in only if every printed attribute is in its declaration (sub-engine attributes included). Record the output in the commit message.

- [ ] **Step 2: Failing tests**

```python
# test/test_engine_model_fingerprint.py
"""BaseEngine.model_fingerprint: declared-attribute identity for params-only engines (patch spec §7)."""
from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine, BlackScholesEngine, DeltaOneEngine
from quantark.asset.equity.engine.mc.euro_mc_engine import EuropeanMCEngine
from quantark.asset.equity.param import EngineParams


def test_params_only_engines_fingerprint_by_class_and_params():
    a, b = BlackScholesEngine(), BlackScholesEngine()
    assert a.model_fingerprint() == b.model_fingerprint() is not None
    assert BlackScholesEngine(EngineParams(bus_days_in_year=244)).model_fingerprint() != a.model_fingerprint()
    assert DeltaOneEngine(use_market_price=True).model_fingerprint() != DeltaOneEngine().model_fingerprint()
    fp = BarrierAnalyticalEngine().model_fingerprint()
    assert fp is not None and any(name == "_one_touch_engine" for name, _ in fp[2])   # sub-engine folded in


def test_engines_without_a_declaration_have_no_fingerprint():
    assert EuropeanMCEngine().model_fingerprint() is None
```

`test/test_pnlexplain_factor_diff.py:64-67` becomes:

```python
    # two distinct but EQUIVALENT engine objects are not a model change (patch spec §7)
    other = build_factor_moves(s0, s1, coord, engine_alive_t1=BlackScholesEngine(), lifecycle_changed=False)
    assert Factor.MODEL not in other.changed
    from quantark.asset.equity.param import EngineParams
    diff = build_factor_moves(s0, s1, coord, engine_alive_t1=BlackScholesEngine(EngineParams(bus_days_in_year=244)),
                              lifecycle_changed=False)
    assert Factor.MODEL in diff.changed
```

`test/test_pnlexplain_lifecycle.py:145-147`: the `BlackScholesEngine()` substitution is now accepted; keep a rejection using a non-equivalent engine (`BlackScholesEngine(EngineParams(bus_days_in_year=244))`). `test/test_pnlexplain_scenario.py`: add a pricing-count test that an equivalent fresh engine on the t1 snapshot adds no MODEL state (`c.all_market()` prices once; MODEL ∉ `c.effective`).

- [ ] **Step 3: Implement**

`base_engine.py`:

```python
    #: Attribute names that fully determine this engine's pricing function. ``None`` (the
    #: default) means "identity only": two instances are never equivalent. Engines whose
    #: instance state is exactly their construction arguments opt in (patch spec §7).
    MODEL_FINGERPRINT_ATTRS: Optional[Tuple[str, ...]] = None

    def model_fingerprint(self) -> Optional[tuple]:
        """(module, qualname, ((attr, value), ...)) or None when this engine compares by identity.

        Sub-engine attributes are replaced by their own fingerprint; a sub-engine without
        one makes the whole fingerprint None (fail closed).
        """
        if self.MODEL_FINGERPRINT_ATTRS is None:
            return None
        items = []
        for name in self.MODEL_FINGERPRINT_ATTRS:
            value = getattr(self, name)
            if isinstance(value, BaseEngine):
                value = value.model_fingerprint()
                if value is None:
                    return None
            items.append((name, value))
        return (type(self).__module__, type(self).__qualname__, tuple(items))
```

Opt-ins (class attributes), per the Step 1 output: `BlackScholesEngine ("params",)`, `DeltaOneEngine ("params", "use_market_price")`, `DigitalOptionAnalyticalEngine ("params", "_vanilla_engine")`, `BarrierAnalyticalEngine ("params", "_bs_engine", "_one_touch_engine")`, `OneTouchAnalyticalEngine ("params", "_digital_engine")`, `DoubleBarrierOptionAnalyticalEngine ("params", "_bs_engine")`, `SingleSharkfinOptionAnalyticalEngine ("params", "_barrier_engine", "_one_touch_engine")`, `DoubleSharkfinOptionAnalyticalEngine ("params", "max_terms", "quad_points", "_double_barrier_engine")`, `AmericanOptionAnalyticalEngine ("params", "method")`. Skip any engine whose `vars()` shows more.

`fingerprints.py`:

```python
def engines_equivalent(a: Any, b: Any) -> bool:
    """Identity, or the same class with equal, normalisable model fingerprints (patch spec §7).

    Engines without a fingerprint compare by identity; a fingerprint the normaliser cannot
    represent counts as NOT equivalent. Never raises.
    """
    if a is b:
        return True
    if a is None or b is None or type(a) is not type(b):
        return False
    fa, fb = getattr(a, "model_fingerprint", None), getattr(b, "model_fingerprint", None)
    if not callable(fa) or not callable(fb):
        return False
    ra, rb = fa(), fb()
    if ra is None or rb is None:
        return False
    try:
        return _normalize(ra) == _normalize(rb)
    except ValidationError:
        return False
```

`factor_diff.py:180`: `if Factor.MODEL in app and engine_alive_t1 is not None and not engines_equivalent(snap0.engine, engine_alive_t1):`. `lifecycle.py:129`: `if not engines_equivalent(snap1.engine, transition.engine_alive_t1):`. `explain.py` metadata: `"model_equivalent": engines_equivalent(snapshot_t0.engine, transition.engine_alive_t1)`. Export `engines_equivalent` from `quantark/pnlexplain/__init__.py` and add it to `__all__` (update the export test in `test_pnlexplain_matrix.py` if it enumerates names).

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_engine_model_fingerprint.py test/test_pnlexplain_factor_diff.py test/test_pnlexplain_lifecycle.py test/test_pnlexplain_scenario.py test/test_pnlexplain_matrix.py test/test_pnlexplain_backtest_replay.py -q`
Expected: all passed (the replay's recalibrated local-vol engines have no fingerprint, so `test_localvol_recalibration_lands_in_model_row` is unchanged).

- [ ] **Step 4: Commit**

```bash
git add quantark/asset/equity/engine/base_engine.py quantark/asset/equity/engine/analytical/*.py quantark/pnlexplain/equity/fingerprints.py quantark/pnlexplain/equity/factor_diff.py quantark/pnlexplain/equity/lifecycle.py quantark/pnlexplain/equity/explain.py quantark/pnlexplain/__init__.py test/test_engine_model_fingerprint.py test/test_pnlexplain_factor_diff.py test/test_pnlexplain_lifecycle.py test/test_pnlexplain_scenario.py test/test_pnlexplain_matrix.py
git -c commit.gpgsign=false commit -F /path/to/msg_task5.txt
```

(`git add .../analytical/*.py` is safe here: only the opt-in files changed; verify with `git status --short` first.)

---

### Task 6: TradingClock-wrapped environments (spec §8)

**Files:**
- Modify: `quantark/util/calendar/trading_clock.py:37-60` (`horizon_date`, `re_anchored`); `quantark/param/vol/trading_clock_surface.py`, `quantark/param/rrf/trading_clock_curve.py`, `quantark/param/div/trading_clock_yield.py` (`with_time_map`)
- Create: `quantark/pnlexplain/equity/clock.py`
- Modify: `quantark/pnlexplain/equity/factor_diff.py` (`validate_pair`, change detection), `scenario.py:55-80` (`build_state`), `taylor.py:155-165`
- Create: `test/test_pnlexplain_trading_clock.py`

**Interfaces:**
- Produces: `BusinessTimeMap.horizon_date`, `BusinessTimeMap.re_anchored(anchor_date)`, `<wrapper>.with_time_map(time_map)`; `clock.is_clock_wrapped`, `clock.wrapped_equal`, `clock.validate_clock_env`, `clock.validate_clock_pair`, `clock.re_anchor(env)`, `clock.CLOCK_FIELDS = ("vol_surface", "rate_curve", "div_yield")`.

- [ ] **Step 1: Failing tests**

```python
# test/test_pnlexplain_trading_clock.py
"""Clock-wrapped environments: waterfall exact on both axes, Taylor fails closed (patch spec §8)."""
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.param.div.trading_clock_yield import TradingClockDividendYield
from quantark.param.rrf.trading_clock_curve import TradingClockRateCurve
from quantark.param.vol.trading_clock_surface import TradingClockVolSurface
from quantark.pnlexplain import ExplainMethod, Factor, PnLExplainConfig, ValuationSnapshot, explain, value
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

D = 244
CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
EXPIRY = datetime(2026, 9, 15)
HORIZON = datetime(2027, 3, 1)
ENG = BlackScholesEngine()
WF = PnLExplainConfig(methods=(ExplainMethod.WATERFALL,))


def _map(anchor):
    return BusinessTimeMap(TradingClock(CAL, D), anchor, HORIZON)


def _cal_env(date, spot=100.0, sigma_td=0.20, r=0.02, q=0.01, anchor=None):
    m = _map(anchor or date)
    return PricingEnvironment(rate_curve=FlatRateCurve(r), valuation_date=date, spot_quote=SpotQuote(spot),
                              vol_surface=TradingClockVolSurface(FlatVolSurface(sigma_td), m),
                              div_yield=ContinuousDividendYield(q))


def _td_env(date, spot=100.0, sigma_td=0.20, r=0.02, q=0.01):
    m = _map(date)
    return PricingEnvironment(rate_curve=TradingClockRateCurve(FlatRateCurve(r), m), valuation_date=date,
                              spot_quote=SpotQuote(spot), vol_surface=FlatVolSurface(sigma_td),
                              div_yield=TradingClockDividendYield(ContinuousDividendYield(q), m),
                              day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=D, calendar=CAL)


CALL = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, exercise_date=EXPIRY)
STEPS = [(datetime(2026, 3, 10), datetime(2026, 3, 11)),          # plain step
         (datetime(2026, 9, 30), datetime(2026, 10, 9))]          # National Day week: 9 calendar days, 1 trading day


@pytest.mark.parametrize("t0,t1", STEPS)
def test_calendar_axis_waterfall_reconciles_and_same_inner_is_not_a_vol_change(t0, t1):
    if t1 > EXPIRY:
        pytest.skip("step after expiry")
    e0, e1 = _cal_env(t0), _cal_env(t1, spot=101.0)               # fresh wrapper + fresh map, same inner sigma
    r = explain(ValuationSnapshot(CALL, ENG, e0, date=t0), ValuationSnapshot(CALL, ENG, e1, date=t1), config=WF)
    assert Factor.VOL not in r.moves.changed and Factor.RATE not in r.moves.changed
    assert r.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=1e-12)
    # TIME = same inner, same clock, map re-anchored at t1
    re_anchored = _cal_env(t1, anchor=t1)
    direct = value(ValuationSnapshot(CALL, ENG, re_anchored, date=t1)).total - value(ValuationSnapshot(CALL, ENG, e0, date=t0)).total
    assert r.metadata["time_pure"] == pytest.approx(direct, abs=1e-12)
    shap = explain(ValuationSnapshot(CALL, ENG, e0, date=t0), ValuationSnapshot(CALL, ENG, e1, date=t1),
                   config=PnLExplainConfig(methods=(ExplainMethod.WATERFALL,), interaction="shapley"))
    assert shap.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=1e-12)


def test_axis_equivalence_of_rows():
    t0, t1 = datetime(2026, 3, 10), datetime(2026, 3, 12)
    kw1 = dict(spot=102.0, sigma_td=0.22, r=0.025)
    a = explain(ValuationSnapshot(CALL, ENG, _cal_env(t0), date=t0), ValuationSnapshot(CALL, ENG, _cal_env(t1, **kw1), date=t1), config=WF)
    b = explain(ValuationSnapshot(CALL, ENG, _td_env(t0), date=t0), ValuationSnapshot(CALL, ENG, _td_env(t1, **kw1), date=t1), config=WF)
    for f in (Factor.TIME, Factor.SPOT, Factor.VOL, Factor.RATE):
        ra = [x for x in a.rows if x.factor is f and x.method is ExplainMethod.WATERFALL][0].pnl
        rb = [x for x in b.rows if x.factor is f and x.method is ExplainMethod.WATERFALL][0].pnl
        assert ra == pytest.approx(rb, abs=1e-9), f
    assert a.total_pnl == pytest.approx(b.total_pnl, abs=1e-9)


def test_fail_closed_rules():
    t0, t1 = datetime(2026, 3, 10), datetime(2026, 3, 11)
    ok0, ok1 = _cal_env(t0), _cal_env(t1)
    with pytest.raises(ValidationError, match="anchor"):
        explain(ValuationSnapshot(CALL, ENG, ok0, date=t0), ValuationSnapshot(CALL, ENG, _cal_env(t1, anchor=t0), date=t1), config=WF)
    native = PricingEnvironment(rate_curve=FlatRateCurve(0.02), valuation_date=t1, spot_quote=SpotQuote(100.0),
                                vol_surface=FlatVolSurface(0.2), div_yield=ContinuousDividendYield(0.01))
    with pytest.raises(ValidationError, match="clock"):
        explain(ValuationSnapshot(CALL, ENG, ok0, date=t0), ValuationSnapshot(CALL, ENG, native, date=t1), config=WF)
    other_clock = PricingEnvironment(rate_curve=FlatRateCurve(0.02), valuation_date=t1, spot_quote=SpotQuote(100.0),
                                     vol_surface=TradingClockVolSurface(FlatVolSurface(0.2), BusinessTimeMap(TradingClock(CAL, 252), t1, HORIZON)),
                                     div_yield=ContinuousDividendYield(0.01))
    with pytest.raises(ValidationError, match="clock"):
        explain(ValuationSnapshot(CALL, ENG, ok0, date=t0), ValuationSnapshot(CALL, ENG, other_clock, date=t1), config=WF)
    floating = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=0.5)
    with pytest.raises(ValidationError, match="BUSINESS_DAYS"):
        explain(ValuationSnapshot(floating, ENG, _td_env(t0), date=t0),
                ValuationSnapshot(EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=0.5 - 1 / 365), ENG, _td_env(t1), date=t1), config=WF)
    with pytest.raises(ValidationError, match="Taylor"):
        explain(ValuationSnapshot(CALL, ENG, ok0, date=t0), ValuationSnapshot(CALL, ENG, ok1, date=t1))
```

Confirm the holiday dates against `CAL.is_business_day` before relying on them (print the business-day count for each step); adjust the second step if the calendar differs.

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_pnlexplain_trading_clock.py -q`
Expected: failures (VOL read as changed because the wrappers differ by identity; `time_pure` differs; fail-closed rules missing).

- [ ] **Step 2: Library additions**

`trading_clock.py` `BusinessTimeMap.__init__`: store `self.horizon_date = horizon_date`; add

```python
    def re_anchored(self, anchor_date: datetime) -> "BusinessTimeMap":
        """The same clock and horizon seen from another valuation date."""
        return BusinessTimeMap(self.clock, anchor_date, self.horizon_date, extend_weekdays=self.extend_weekdays)
```

Each wrapper: `def with_time_map(self, time_map: BusinessTimeMap): return type(self)(self.inner, time_map)` (the surface is a frozen dataclass: `TradingClockVolSurface(self.inner, time_map)`).

- [ ] **Step 3: `quantark/pnlexplain/equity/clock.py`**

```python
"""Clock-wrapped market objects in the factor model (patch spec 2026-09-03 §8).

A TradingClock wrapper re-expresses an inner object through a BusinessTimeMap
anchored at the environment's valuation date. Two wrappers are the same MARKET
when their class, inner object and clock agree; the anchor is the valuation
date's business and moves with TIME.
"""
from __future__ import annotations

from typing import Any

from quantark.pnlexplain.equity.fingerprints import calendars_equal
from quantark.util.exceptions import ValidationError

CLOCK_FIELDS = ("vol_surface", "rate_curve", "div_yield")


def is_clock_wrapped(obj: Any) -> bool:
    return obj is not None and getattr(obj, "time_map", None) is not None and hasattr(obj, "inner")


def clocks_equal(a: Any, b: Any) -> bool:
    return int(a.days_per_year) == int(b.days_per_year) and calendars_equal(a.calendar, b.calendar)


def wrapped_equal(a: Any, b: Any, inner_equal) -> bool:
    if type(a) is not type(b):
        return False
    return clocks_equal(a.time_map.clock, b.time_map.clock) and inner_equal(a.inner, b.inner)


def validate_clock_env(env: Any, label: str) -> None:
    for name in CLOCK_FIELDS:
        obj = getattr(env, name, None)
        if is_clock_wrapped(obj) and obj.time_map.anchor_date != env.valuation_date:
            raise ValidationError(
                f"{label}.{name} is clock-wrapped with a time map anchored at {obj.time_map.anchor_date}, "
                f"not at the environment's valuation date {env.valuation_date}"
            )


def validate_clock_pair(e0: Any, e1: Any) -> None:
    for name in CLOCK_FIELDS:
        a, b = getattr(e0, name, None), getattr(e1, name, None)
        wa, wb = is_clock_wrapped(a), is_clock_wrapped(b)
        if wa != wb:
            raise ValidationError(f"{name} is clock-wrapped on one side only: a clock change is not a market move")
        if wa and not clocks_equal(a.time_map.clock, b.time_map.clock):
            raise ValidationError(f"{name} carries two different trading clocks: a clock change is not a market move")


def re_anchor(env: Any) -> None:
    """Re-anchor every wrapped field at env.valuation_date (mutates the given copy)."""
    for name in CLOCK_FIELDS:
        obj = getattr(env, name, None)
        if is_clock_wrapped(obj) and obj.time_map.anchor_date != env.valuation_date:
            setattr(env, name, obj.with_time_map(obj.time_map.re_anchored(env.valuation_date)))
```

- [ ] **Step 4: Wire the explain**

`factor_diff.py`:
- `validate_pair`: after the calendar check add `validate_clock_env(e0, "snapshot_t0.pricing_env"); validate_clock_env(e1, "snapshot_t1.pricing_env"); validate_clock_pair(e0, e1)`, and

```python
    if e0.day_count_convention is DayCountConvention.BUSINESS_DAYS and _float_maturity_only(snap0.product):
        raise ValidationError(
            "a float-maturity product on a BUSINESS_DAYS environment has a trading-time maturity; "
            "the days/365 roll rule does not describe it (use a date-based product)"
        )
```

  with `_float_maturity_only(p)` = no `exercise_date`, no `maturity_date`, `maturity` set.
- change detection: `def market_objects_equal(a, b, base=objects_equal)`: both wrapped → `wrapped_equal(a, b, base)`; else `base(a, b)`. Use it for VOL (`objects_equal`), RATE (`objects_equal`), DIVIDEND (`_yields_equal`), leaving BASIS as is.

`scenario.py` `build_state`: add `re_anchor(env)` as the last statement before `return`. (`deepcopy(e0)` already gives a private copy.)

`taylor.py` `taylor_rows`, first lines:

```python
    if any(is_clock_wrapped(getattr(snap0.pricing_env, f, None)) for f in CLOCK_FIELDS):
        raise ValidationError(
            "Taylor method on a TradingClock-wrapped environment is not supported: the calculator's "
            "vol / dividend / rate bumps replace the wrapper with a calendar-quoted object (a different "
            "clock); use methods=(ExplainMethod.WATERFALL,)"
        )
```

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_pnlexplain_trading_clock.py test/test_pnlexplain_factor_diff.py test/test_pnlexplain_scenario.py test/test_trading_clock_axis_equivalence.py test/test_trading_clock*.py -q`
Expected: all passed. If the axis-equivalence rows disagree beyond 1e-9, print both environments' `get_vol(100, T1)` and `get_rate(T1)` for the VOL/RATE coordinates first: on the trading axis `T1` is a trading-year tenor, so the coordinate sampling must go through each environment's own `get_*` (it does) — a mismatch there is a fixture error (σ_td vs σ_cal), not a kernel one.

- [ ] **Step 5: Commit**

```bash
git add quantark/util/calendar/trading_clock.py quantark/param/vol/trading_clock_surface.py quantark/param/rrf/trading_clock_curve.py quantark/param/div/trading_clock_yield.py quantark/pnlexplain/equity/clock.py quantark/pnlexplain/equity/factor_diff.py quantark/pnlexplain/equity/scenario.py quantark/pnlexplain/equity/taylor.py test/test_pnlexplain_trading_clock.py
git -c commit.gpgsign=false commit -F /path/to/msg_task6.txt
```

---

### Task 7: Documentation, parent-spec amendments, full suite

**Files:**
- Modify: `docs/superpowers/specs/2026-09-02-pnl-explain-design.md` (§5.3 line ~277, §7.6 lines ~630-645, §8 lines ~673-683, §10 lines ~975-981, §11 table, §15), `quantark/pnlexplain/README.md` (Backtests, Vol-model engines, Bucketed, Limitations), `quantark/backtest/CLAUDE.md` (untracked: hedge accounting + float roll notes, never `git add`), `docs/superpowers/plans/2026-09-03-pnl-explain-followups.md` (self-review note)

- [ ] **Step 1: Parent spec amendments** — apply the six rows of patch-spec §9 verbatim, each followed by "(amended 2026-09-03, patch spec §n)".

- [ ] **Step 2: README** — Backtests: "`gap_states` is zero for all three hedge paths (the simple executor books hedges at average cost)"; Vol-model engines: "MODEL is detected by engine equivalence: identity, or the same class with equal `model_fingerprint()`s; engines without a fingerprint (MC, PDE, vol-model) compare by identity"; Bucketed: rate buckets are components under the dividend-held convention; Limitations: replace the untracked-products bullet with "Futures hedges and schedule-bearing untracked products are repriced with a constant maturity (declared via `contract_roll_days=0`); schedule-free float contracts are rolled", drop "TradingClock untested" and the executor / barrier-lag bullets, add "Taylor on clock-wrapped environments is rejected (waterfall only)".

- [ ] **Step 3: Full suite on the final tree**

Write `<scratchpad>/run_full_suite.py` (copy of the previous session's runner with the new worktree path) and run it in the background with `nohup caffeinate -i -m -s` (multi-hour job; the machine idles to sleep otherwise). Expected: exit 0. Then `git checkout -- example/mo_volmodels/data/mo_barrier_sample.json example/mo_volmodels/data/mo_calibration_explainer_sample.html` and run `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python example/pnl_explain_demo.py` — the reconciliation gaps must still print as 0.0.

- [ ] **Step 4: Commit and hand over**

```bash
git add -f docs/superpowers/specs/2026-09-02-pnl-explain-design.md docs/superpowers/specs/2026-09-03-pnl-explain-followups-design.md docs/superpowers/plans/2026-09-03-pnl-explain-followups.md quantark/pnlexplain/README.md
git -c commit.gpgsign=false commit -F /path/to/msg_task7.txt
```

Then use `superpowers:finishing-a-development-branch` (merge menu). Update the memory file `project_pnl_explain.md` (quirks fixed; remaining findings from patch-spec §14).

---

## Self-review notes

1. **Spec coverage**: §3 → Task 1; §4 → Task 2; §5 → Task 3; §6 → Task 4; §7 → Task 5; §8 → Task 6; §9 / README → Task 7; §11 error rows → the `pytest.raises` tests in Tasks 3, 4, 6 and the close-on-zero test in Task 1; §12 gates → each task's final run plus Task 7's full suite.
2. **Type consistency to re-check while executing**: `bucketed_rows` still returns `(rows_by_factor, covered)`; `taylor.py` consumes `covered` for both VOL and RATE; `engines_equivalent` lives in `fingerprints.py` (imported by `factor_diff.py` and `lifecycle.py`, which already import from it — no cycle); `clock.py` imports `calendars_equal` from `fingerprints.py` and nothing from `factor_diff.py` (which imports `clock.py`); `FloatMaturityRoller.rolled` raises `KeyError` for an unregistered id by design (callers always register first).
3. **Known risks to report, not paper over**: Task 3's PDE cross-check tolerance is set after measuring; if the lag effect disagrees by more than 5 % at `accuracy="high"`, report the numbers rather than loosening the gate. Task 6's holiday step dates must be confirmed against the calendar object. Task 2 changes every float-maturity vanilla backtest's numbers; any test found pinning one is reported, not re-pinned.

## Execution notes (2026-09-03, inline execution on worktree `pnl-explain-followups`)

Deviations from the steps above, each decided while executing and recorded in the commit messages:

- **Task 2**: with the book's `maturity=1.0` call now rolled daily, `test_multi_greek_backtest::test_backtest_neutralizes_all_greeks` became degenerate (the default 1Y date-based vega hedge option replicated the book exactly and the 3×3 solve dropped the other two instruments, breaking its "≥ 3 positions" shape check); the test now passes a 9M vega instrument. `PortfolioLifecycleManager.register_positions` also registers untracked rollable positions at `base_date` (the trackers' convention), so an initial vanilla rolls "as of the start date" even when the first pricing call comes later; `test_pnlexplain_lifecycle_days::test_manager_pricing_products_is_pure` (not in the plan's run list) was updated to the new pure-accessor contract (fresh rolled copy, position untouched).
- **Task 3**: the settlement resolver refuses a `CALENDAR_DAYS` lag on the **terminal** leg of a float-maturity product ("requires an authoritative determination date"), so the calendar-day tests and both lagged backtest fixtures use date-based contracts (`exercise_date`); the FOLLOWING / BUSINESS_DAYS rejection cases need a pricing calendar for the terminal leg to resolve before the helper raises. The discrete double-sharkfin test uses two nodes (BGK needs a regular interval) and pins per-node discounting as exp(−rL) under a flat curve and a weighted average of the node ratios under the term curve. With the lagged barrier fixture on the analytical engine the explain's KO-day event row is exactly 0.0 (the already-hit price IS the receivable PV); the test asserts that instead of "≠ 0". PDE cross-check measured: analytical lag effect 0.137176 vs PDE high 0.137174 (rel −1.0e-5), gate tightened to rel 1e-3.
- **Task 5**: `vars(engine)` output matched the planned declarations for all nine engines (`EuropeanMCEngine` shows `['method', 'params']` but stays identity-only by design). `test_pnlexplain_scenario::test_every_engine_context_is_frozen_at_the_t0_state` needed a non-equivalent engine (`EngineParams(bus_days_in_year=244)`) to remain a MODEL change.
- **Task 6**: CHINA_SSE has 2 trading days (Oct 8–9) in the Sept 30 → Oct 9 step, not 1; the fixture-facts test pins the calendar. `market_objects_equal` takes the base comparator so the dividend field keeps its zero-yield equivalence.
- **Task 7**: `quantark/backtest/CLAUDE.md` is untracked and does not exist in the worktree; the main checkout's copy was updated directly.
