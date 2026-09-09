# Simulated-Path Backtest — Plan 4 of 4: Results, Distributions, Persistence and the Worked Study

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn an `EnsembleResults` into the answers the feature exists for — a per-path summary in the historical study's own measures, distributions with quantiles and expected shortfall, paired comparisons on matched paths, persistence — and ship the worked snowball study `example/snowball_simulated_paths/` that runs the q term-structure models and both hedge contracts over bootstrap and stress paths, checks the life-surface provider against QUAD repricing, locates the historical study's realised runs inside the simulated distribution, and writes a self-contained HTML report.

**Architecture:** The hedge measures move from the study into the library (`quantark/backtest/simulation/measures.py`) so the historical replay and the simulated ensemble report one implementation; the study's function delegates. `EnsembleResults` moves to its own module with `summary`, `distribution`, `paired`, `take`, `to_dir`/`from_dir`. One performance change in plan-3 code: the life-surface provider keeps only the columns a run reads (about four times less memory per surface, readouts bit-identical). The study reuses the q term-structure study's helpers (terms, fair-coupon solver, model catalogue, hedge policies, history loaders) by loading its `_common.py`, and every stage separates pure functions from the CLI so a synthetic test runs the whole pipeline in seconds.

**Tech Stack:** Python 3.10–3.13, numpy, pandas, pytest (`-n0` while iterating), matplotlib optional for report charts (as in the q study, charts are skipped when it is absent), `nohup caffeinate -i -m -s` for the real fleet.

**Spec:** `docs/superpowers/specs/2026-09-08-simulated-path-backtest-design.md`, sections 9 (results), 13 (results tests), 14 (example study), 15 (performance expectations) and items 5–6 of 16. Plans 1–3 are implemented on the branch (`docs/superpowers/plans/2026-09-08-simulated-paths-plan{1,2,3}-*.md`); the package README describes the engine, the providers, the gate and batching this plan builds on.

## Global Constraints

- Work on branch `feat/simulated-path-backtest`. Verify `git log --oneline -1` shows `6bfaeff1` or later before starting. Run tests as `/Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 <files>` (add `PYTHONPATH=$PWD` in a worktree so its source shadows the editable install).
- Commits: every message ends with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Never `git add example/` wholesale (two `example/mo_volmodels/data/` sample files churn under test runs; `git checkout -- example/mo_volmodels/data/` before staging). Stage the new study by path: `git add example/snowball_simulated_paths/`. Files under `docs/` need `git add -f`. `CLAUDE.md` files are never added. `/output` is excluded through `.git/info/exclude`; never stage anything under it.
- **No behaviour change for plan 1–3 callers.** `from quantark.backtest.simulation.engine import EnsembleResults, StateCube, FLOAT_COLUMNS, ...` must keep working after the move (Task 2 re-imports the names into `engine.py`). The exact-mode oracle tests keep reporting gaps of exactly `0.0`; `test/simulation`, `test/test_snowball_q_term_structure_study.py`, `test/replay_golden` stay green with no edits beyond those this plan names.
- Fail closed (spec §12): a `paired` on unmatched paths, a `from_dir` on a missing file or a foreign format, a `distribution` of an unknown measure, a study stage handed a missing history — each raises `ValidationError` (the study's loaders raise the q study's `StudyDataError`, as they do today) naming what is missing.
- Determinism: bootstrap and stress paths are reproducible from `(seed, history fingerprint, parameters)`; a cell's config fingerprint is `blake2b` over its stated settings, never a Python `hash`.
- Real market data (`example/mo_volmodels/data/history/`) is local and untracked; every study test runs on synthetic frames and never reads it. The real `--quick` run in Task 10 is verification, not a test.
- Use `quantark.util.numerical` helpers in library code. The measures copied from the study keep their arithmetic verbatim (they are the definition).
- Style: PEP 8, dataclasses with type hints, docstrings on all public APIs, `from __future__ import annotations` at the top of every new module.

---

## Scope of this plan

**In:** the library measures, `results.py` (`summary`, `distribution`, `paired`, `take`, `to_dir`, `from_dir`, manifest `library_version` and `book_notional`), the compact life surface, the study (`_common.py`, `01_build_paths.py`, `02_ensemble_fleet.py`, `03_report.py`, README), exports and the package README's "Results" section, the full regression, a real `--quick` run of the study.

**Out, by design:** the full 2,000-path fleet (hours; the user runs it, with the command this plan writes into the study README), any change to the providers' numerics, multi-cell scheduling beyond "one process pool per cell".

**Three facts that shape the study, established while writing this plan:**

1. **Under the approximate providers the term dividend models enter through the scalar `q_T` only.** A life surface is solved at a flat `q` (the bucket centre of the zero yield at the remaining maturity); a ladder node's environment carries a flat `SignedDividendYield`. Only exact repricing hands the engine the term object. The study therefore states what a cell measures: the *term model's `q_T` along the path* under the surface and the ladder, and the *term object itself* only in the optional exact-QUAD subset (`--exact-paths`). The gate reports the gap between the two on every cell.
2. **The q study's `build_product` sets `initial_date`, so its snowball is on the tracker's DATE clock**, while every plan-2/3 fixture was on the numeric clock. The vectorised lifecycle handles both (`CalendarSchedule.uses_date_timing`), and the fleet's oracle spot check (Task 8) is the first test of the date clock through the whole engine.
3. **A life surface at the plan-3 layout is ~19 MB** (`points=400`, `steps_per_day=4`, one year: six `(400, ~1000)` float64 slabs). The run reads ~260 daily columns; keeping only those cuts a surface to ~5 MB with bit-identical readouts, so a 2 GB surface budget holds ~400 buckets instead of ~100. Task 6 does that before the study needs it.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `quantark/backtest/simulation/measures.py` (create) | `path_measures`, `max_drawdown`, `bp_of`, `MEASURE_COLUMNS` | 1 |
| `example/snowball_q_term_structure/_common.py` (modify) | `hedge_measures` / `max_drawdown` / `_bp` delegate to the library | 1 |
| `quantark/backtest/simulation/results.py` (create) | `StateCube`, `EnsembleResults` (moved) + `summary`, `distribution`, `paired`, `take`, `to_dir`, `from_dir`, `PairedComparison`, `jsonable` | 2, 3, 4, 5 |
| `quantark/backtest/simulation/engine.py` (modify) | import the moved names; manifest `library_version`, `book_notional` | 2 |
| `quantark/backtest/simulation/pricing/surface.py` (modify) | `LifeSurface.select`, compact surfaces in the pricer | 6 |
| `example/snowball_simulated_paths/_common.py` (create) | study constants, q-study loader, engine configs, stress set | 7 |
| `example/snowball_simulated_paths/01_build_paths.py` (create) | history → bootstrap + stress batches on disk | 7 |
| `example/snowball_simulated_paths/02_ensemble_fleet.py` (create) | terms, fair coupon, cells, oracle spot check, resume | 8 |
| `example/snowball_simulated_paths/03_report.py` (create) | distributions, paired, stress, engine check, historical location, HTML | 9 |
| `example/snowball_simulated_paths/README.md` (create) | the study's guide and results | 10 |
| `quantark/backtest/simulation/__init__.py`, `README.md` (modify) | exports, "Results" section | 10 |
| `test/simulation/test_measures.py` (create) | Task 1 | 1 |
| `test/simulation/test_results.py` (create) | Tasks 2–5 | 2, 3, 4, 5 |
| `test/simulation/test_surface.py` (modify) | Task 6 | 6 |
| `test/test_snowball_simulated_paths_study.py` (create) | Tasks 7–9 | 7, 8, 9 |
| `test/simulation/test_market_path.py` (modify) | exports | 10 |

Conventions from plans 2–3 hold: the cube is `(n_paths, n_days)`; `i` paths, `d` days, `p` products.

---

### Task 1: The hedge measures move into the library

**Files:**
- Create: `quantark/backtest/simulation/measures.py`
- Modify: `example/snowball_q_term_structure/_common.py` (`_bp`, `max_drawdown`, `hedge_measures` at lines 874–946)
- Create: `test/simulation/test_measures.py`

**Interfaces:**
- `bp_of(value: float, notional: float) -> float`
- `max_drawdown(series: Sequence[float]) -> float`
- `path_measures(states: pd.DataFrame, trades: pd.DataFrame, *, notional: float) -> Dict[str, Any]` — the q study's `hedge_measures`, verbatim. `states` needs columns `total_pnl`, `product_pnl`, `product_mtm`, `pricing_q`, `active_contract`, `transaction_costs`; `trades` needs `trade_type`, `notional`, `quantity`. Both `EnsembleResults.path_states/path_trades` and the replay's `states_df()/trades_df()` carry them.
- `MEASURE_COLUMNS: Tuple[str, ...]` — the keys `path_measures` returns, in order.

- [ ] **Step 1: Write the failing tests**

`test/simulation/test_measures.py`:

```python
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.simulation.measures import MEASURE_COLUMNS, bp_of, max_drawdown, path_measures

NOTIONAL = 10_000.0


def _states(total, product, mtm, q, active, costs) -> pd.DataFrame:
    return pd.DataFrame({
        "total_pnl": total, "product_pnl": product, "product_mtm": mtm, "pricing_q": q,
        "active_contract": active, "transaction_costs": costs,
    })


def _hand_built():
    states = _states(
        total=[0.0, 100.0, 50.0, 200.0], product=[0.0, 300.0, -100.0, 400.0],
        mtm=[1000.0, 1100.0, 1000.0, 1300.0], q=[0.01, 0.01, 0.02, 0.02],
        active=["IM2401", "IM2401", "IM2402", "IM2402"], costs=[0.0, 1.0, 2.0, 3.0],
    )
    trades = pd.DataFrame({
        "trade_type": ["hedge_rebalance", "roll_close", "roll_open", "hedge_rebalance"],
        "notional": [1000.0, 500.0, 500.0, 2000.0], "quantity": [2.0, -1.0, 1.0, -3.0],
    })
    return states, trades


def test_every_measure_on_a_hand_built_path():
    states, trades = _hand_built()
    m = path_measures(states, trades, notional=NOTIONAL)
    assert tuple(m) == MEASURE_COLUMNS
    d_total = np.diff([0.0, 100.0, 50.0, 200.0])          # 100, -50, 150
    d_product = np.diff([0.0, 300.0, -100.0, 400.0])      # 300, -400, 500
    assert m["days"] == 4
    assert m["terminal_pnl_bp"] == 200.0                  # 200 / 10 000 * 1e4
    assert m["terminal_pnl_gross_bp"] == 203.0            # costs added back
    assert m["daily_pnl_std_bp"] == pytest.approx(np.std(d_total, ddof=1))
    assert m["variance_reduction_r2"] == pytest.approx(1.0 - np.var(d_total, ddof=1) / np.var(d_product, ddof=1))
    assert m["max_drawdown_bp"] == 50.0                   # peak 100 -> trough 50
    assert (m["turnover"], m["rebalance_turnover"], m["roll_turnover"]) == (0.4, 0.3, 0.1)
    assert m["cost_bp"] == 3.0
    assert m["roll_days"] == 1                            # the contract changes once, on day 2
    assert m["roll_day_mtm_jump_bp"] == 100.0             # |1000 - 1100| on the roll day
    assert m["other_day_mtm_jump_bp"] == 200.0            # mean(|1100-1000|, |1300-1000|)
    assert m["roll_day_q_jump"] == pytest.approx(0.01)
    assert m["delta_churn"] == 1.25                       # (2 + 3) rebalance contracts / 4 days


def test_short_paths_and_no_trades_report_nan_and_zero_not_errors():
    states = _states([0.0, 5.0], [0.0, 7.0], [10.0, 12.0], [0.0, 0.0], ["A", "A"], [0.0, 0.0])
    m = path_measures(states, pd.DataFrame(columns=["trade_type", "notional", "quantity"]), notional=NOTIONAL)
    assert np.isnan(m["daily_pnl_std_bp"]) and np.isnan(m["variance_reduction_r2"])
    assert m["turnover"] == 0.0 and m["delta_churn"] == 0.0 and m["roll_days"] == 0
    assert np.isnan(m["roll_day_mtm_jump_bp"])


def test_helpers():
    assert bp_of(25.0, NOTIONAL) == 25.0
    assert max_drawdown([0.0, 3.0, 1.0, 4.0, -2.0]) == 6.0
    assert max_drawdown([]) == 0.0


def test_the_q_study_delegates_to_the_library():
    path = Path(__file__).resolve().parents[2] / "example" / "snowball_q_term_structure" / "_common.py"
    spec = importlib.util.spec_from_file_location("q_term_structure_common_for_measures", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    states, trades = _hand_built()
    assert module.hedge_measures(states, trades, notional=NOTIONAL) == path_measures(states, trades, notional=NOTIONAL)
    assert module.max_drawdown is max_drawdown
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_measures.py -q`
Expected: `ModuleNotFoundError: No module named 'quantark.backtest.simulation.measures'`.

- [ ] **Step 3: Write the module**

`quantark/backtest/simulation/measures.py`:

```python
"""Hedge-quality measures of one path, in bp of notional (spec 9).

These are the q term-structure study's ``hedge_measures``, moved here so the
historical replay and the simulated ensemble report ONE implementation; the
study's function delegates to this one.  The arithmetic is the definition
and is kept verbatim.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd

#: The keys ``path_measures`` returns, in order.
MEASURE_COLUMNS: Tuple[str, ...] = (
    "days", "terminal_pnl_bp", "terminal_pnl_gross_bp", "daily_pnl_std_bp", "variance_reduction_r2",
    "max_drawdown_bp", "turnover", "rebalance_turnover", "roll_turnover", "cost_bp", "roll_days",
    "roll_day_mtm_jump_bp", "other_day_mtm_jump_bp", "roll_day_q_jump", "delta_churn",
)


def bp_of(value: float, notional: float) -> float:
    """``value`` as basis points of ``notional``."""
    return float(value) / float(notional) * 1e4


def max_drawdown(series: Sequence[float]) -> float:
    """Largest peak-to-trough fall of a cumulative P&L series (0 for an empty one)."""
    peak = -math.inf
    worst = 0.0
    for v in series:
        peak = max(peak, float(v))
        worst = max(worst, peak - float(v))
    return worst


def path_measures(states: pd.DataFrame, trades: pd.DataFrame, *, notional: float) -> Dict[str, Any]:
    """Hedge-quality measures of one replay or simulated path, in bp of notional.

    - ``terminal_pnl_bp``: final total P&L (product + hedge - costs);
      ``terminal_pnl_gross_bp`` adds the costs back.
    - ``daily_pnl_std_bp``: std of the DAILY hedged P&L increments -- the
      hedge error a desk lives with day to day.
    - ``variance_reduction_r2``: 1 - var(hedged daily P&L)/var(unhedged
      product daily P&L); how much of the product's daily variance the
      futures hedge removes.
    - ``max_drawdown_bp``: peak-to-trough of cumulative hedged P&L.
    - ``turnover``: sum of |trade notional| / notional, split into
      rebalance vs roll legs; ``cost_bp``: cumulative transaction cost.
    - ``roll_day_mtm_jump_bp`` vs ``other_day_mtm_jump_bp``: mean
      |delta product MTM| on days the active hedge contract changed vs all
      other days.  A pricing model tied to the active contract re-marks the
      book on every roll for no economic reason; a term model does not.
    - ``roll_day_q_jump``: mean |delta pricing_q| on roll days.
    - ``delta_churn``: mean |contracts traded| per day.

    ``states`` needs ``total_pnl``, ``product_pnl``, ``product_mtm``,
    ``pricing_q``, ``active_contract`` and ``transaction_costs``; ``trades``
    needs ``trade_type``, ``notional`` and ``quantity``.  A path shorter
    than three days reports NaN for the variance measures.
    """
    n = int(len(states))
    total = states["total_pnl"].to_numpy(dtype=float)
    product = states["product_pnl"].to_numpy(dtype=float)
    d_total = np.diff(total)
    d_product = np.diff(product)
    nan = float("nan")
    if n >= 3:
        std_total = float(np.std(d_total, ddof=1))
        var_total = float(np.var(d_total, ddof=1))
        var_product = float(np.var(d_product, ddof=1))
        r2 = 1.0 - var_total / var_product if var_product > 0.0 else nan
    else:
        std_total, r2 = nan, nan

    if len(trades):
        trade_notional = trades["notional"].to_numpy(dtype=float)
        is_roll = trades["trade_type"].astype(str).str.startswith("roll").to_numpy()
        turnover = float(trade_notional.sum()) / notional
        roll_turnover = float(trade_notional[is_roll].sum()) / notional
        rebalance_turnover = float(trade_notional[~is_roll].sum()) / notional
        contracts_traded = float(np.abs(trades["quantity"].to_numpy(dtype=float)[~is_roll]).sum())
    else:
        turnover = roll_turnover = rebalance_turnover = contracts_traded = 0.0

    active = states["active_contract"].astype(str).to_numpy()
    roll_mask = np.zeros(n, dtype=bool)
    if n >= 2:
        roll_mask[1:] = active[1:] != active[:-1]
    d_mtm = np.abs(np.diff(states["product_mtm"].to_numpy(dtype=float)))
    d_q = np.abs(np.diff(states["pricing_q"].to_numpy(dtype=float)))
    roll_idx = roll_mask[1:]
    roll_days = int(roll_idx.sum())
    roll_jump = float(d_mtm[roll_idx].mean()) if roll_days else nan
    other_jump = float(d_mtm[~roll_idx].mean()) if (~roll_idx).sum() else nan
    roll_q_jump = float(d_q[roll_idx].mean()) if roll_days else nan

    costs = float(states["transaction_costs"].iloc[-1]) if n else 0.0
    return {
        "days": n,
        "terminal_pnl_bp": bp_of(total[-1], notional) if n else 0.0,
        "terminal_pnl_gross_bp": bp_of(total[-1] + costs, notional) if n else 0.0,
        "daily_pnl_std_bp": bp_of(std_total, notional) if math.isfinite(std_total) else nan,
        "variance_reduction_r2": r2,
        "max_drawdown_bp": bp_of(max_drawdown(total), notional) if n else 0.0,
        "turnover": turnover,
        "rebalance_turnover": rebalance_turnover,
        "roll_turnover": roll_turnover,
        "cost_bp": bp_of(costs, notional),
        "roll_days": roll_days,
        "roll_day_mtm_jump_bp": bp_of(roll_jump, notional) if math.isfinite(roll_jump) else nan,
        "other_day_mtm_jump_bp": bp_of(other_jump, notional) if math.isfinite(other_jump) else nan,
        "roll_day_q_jump": roll_q_jump,
        "delta_churn": contracts_traded / n if n else 0.0,
    }
```

- [ ] **Step 4: Make the study delegate**

In `example/snowball_q_term_structure/_common.py` replace the three definitions (`_bp` at line 874, `max_drawdown` at 878, `hedge_measures` at 887 through its `return` block ending at line 946) with:

```python
from quantark.backtest.simulation.measures import bp_of as _bp  # noqa: E402  (the study's historical name)
from quantark.backtest.simulation.measures import max_drawdown, path_measures  # noqa: E402


def hedge_measures(states: pd.DataFrame, trades: pd.DataFrame, *, notional: float) -> Dict[str, Any]:
    """Hedge-quality measures of one replay run: ``quantark.backtest.simulation.measures.path_measures``.

    The definitions moved into the library so the simulated-path study
    reports the same numbers; see that module's docstring for each measure.
    """
    return path_measures(states, trades, notional=notional)
```

Check nothing else in the file defined or shadowed those names (`grep -n "_bp(\|max_drawdown(" example/snowball_q_term_structure/_common.py`); the other callers keep working because the names are still bound at module level.

- [ ] **Step 5: Run the tests and the study's own tests**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_measures.py test/test_snowball_q_term_structure_study.py -q`
Expected: all pass with no change to the study test file.

- [ ] **Step 6: Commit**

```bash
git add quantark/backtest/simulation/measures.py example/snowball_q_term_structure/_common.py test/simulation/test_measures.py
git commit -m "feat(backtest/simulation): hedge measures move into the library; the q study delegates

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `results.py` and the per-path summary

**Files:**
- Create: `quantark/backtest/simulation/results.py`
- Modify: `quantark/backtest/simulation/engine.py` (remove the moved code at lines 40–131; import it; manifest fields)
- Create: `test/simulation/test_results.py`

**Interfaces:**
- `results.py` owns `FLOAT_COLUMNS`, `BOOL_COLUMNS`, `TRADE_COLUMNS`, `EVENT_COLUMNS`, `StateCube`, `EnsembleResults` (moved unchanged, then extended). `engine.py` does `from .results import BOOL_COLUMNS, EVENT_COLUMNS, FLOAT_COLUMNS, TRADE_COLUMNS, EnsembleResults, StateCube` so every existing import from `.engine` still resolves.
- `EnsembleResults.notional -> float` (from `manifest["book_notional"]`); `EnsembleResults.summary -> pd.DataFrame` (memoised; columns `SUMMARY_COLUMNS = ("path", "days", "termination_reason", "knocked_in", "ko_observation_index", "terminal_pnl") + MEASURE_COLUMNS`).
- Manifest gains `"library_version"` (`LIBRARY_VERSION` from `pricing/cache.py`) and `"book_notional"` (`Σ |quantity| · initial_price · contract_multiplier` over the products).
- `termination_reason` is the path's LAST terminal event (`knock_out` or `maturity`) across the book's products, else `data_end`; `ko_observation_index` is that event's index (`-1` for maturity or data end); `knocked_in` is the book flag on the path's last day; `terminal_pnl` is `total_pnl` on that day in currency.

- [ ] **Step 1: Write the failing tests**

`test/simulation/test_results.py`:

```python
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

import quantark
from quantark.backtest.simulation.engine import EnsembleBacktestEngine
from quantark.backtest.simulation.measures import MEASURE_COLUMNS, path_measures
from quantark.backtest.simulation.paths.gbm import ConstantVol, GBMPaths
from quantark.backtest.simulation.paths.market_path import DEFAULT_TENOR_GRID, StartState, trading_calendar
from quantark.backtest.simulation.results import SUMMARY_COLUMNS, EnsembleResults
from quantark.backtest.simulation.runner import run_ensemble
from quantark.util.exceptions import ValidationError

from .conftest import RATE, SPOT, VOL, ensemble_config, flat_carry

START = date(2024, 1, 2)


def _paths(n_paths=8, n_days=8, sigma=1.2, seed=3):
    start = StartState(spot=SPOT, atm_vol=VOL, rate=RATE, carry=flat_carry(DEFAULT_TENOR_GRID))
    gen = GBMPaths(start=start, calendar=trading_calendar(START, n_days), mu=0.0, sigma=sigma,
                   vol_rule=ConstantVol(VOL), carry_schedule=None, rate=RATE, tenor_grid=DEFAULT_TENOR_GRID)
    return gen.generate(n_paths, n_days, seed=seed)


@pytest.fixture(scope="module")
def results() -> EnsembleResults:
    return EnsembleBacktestEngine(ensemble_config()).run(_paths())


def test_the_manifest_records_notional_and_library_version(results):
    assert results.manifest["book_notional"] == 1000.0 * SPOT * 1.0
    assert results.notional == results.manifest["book_notional"]
    assert results.manifest["library_version"] == quantark.__version__


def test_summary_has_one_row_per_path_with_the_library_measures(results):
    summary = results.summary
    assert list(summary.columns) == list(SUMMARY_COLUMNS)
    assert len(summary) == results.n_paths and list(summary["path"]) == list(range(results.n_paths))
    for i in range(results.n_paths):
        expected = path_measures(results.path_states(i), results.path_trades(i), notional=results.notional)
        row = summary.iloc[i]
        for name in MEASURE_COLUMNS:
            a, b = row[name], expected[name]
            assert (np.isnan(a) and np.isnan(b)) or a == b, (i, name, a, b)
        assert row["days"] == int(results.last_day[i]) + 1
        assert row["terminal_pnl"] == results.cube.total_pnl[i, int(results.last_day[i])]


def test_termination_reasons_and_flags_follow_the_events(results):
    summary = results.summary
    reasons = set(summary["termination_reason"])
    assert reasons <= {"knock_out", "maturity", "data_end"} and {"knock_out", "maturity"} <= reasons
    for i, row in summary.iterrows():
        last = int(results.last_day[i])
        assert bool(row["knocked_in"]) == bool(results.cube.knocked_in[i, last])
        mine = [e for e in results.events if e.path == i and e.event in ("knock_out", "maturity")]
        if row["termination_reason"] == "knock_out":
            assert bool(results.cube.knocked_out[i, last]) and row["ko_observation_index"] == mine[-1].index >= 0
        elif row["termination_reason"] == "maturity":
            assert bool(results.cube.matured[i, last]) and row["ko_observation_index"] == -1


def test_a_batched_run_has_the_same_summary(results):
    batched = run_ensemble(ensemble_config(batch_paths=3), _paths())
    pd.testing.assert_frame_equal(batched.summary, results.summary)


def test_summary_is_memoised(results):
    assert results.summary is results.summary
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_results.py -q`
Expected: `ModuleNotFoundError: No module named 'quantark.backtest.simulation.results'`.

- [ ] **Step 3: Move the results objects and add the summary**

Create `quantark/backtest/simulation/results.py` with the module docstring below, the constants and the two classes cut from `engine.py` lines 40–131 (`FLOAT_COLUMNS`, `BOOL_COLUMNS`, `TRADE_COLUMNS`, `StateCube`, `EnsembleResults`), then the additions:

```python
"""What a run returns: the state cube, the logs, the manifest, and the answers (spec 9).

``summary`` is one row per path in the historical study's measures;
``distribution`` and ``paired`` reduce it; ``to_dir`` / ``from_dir``
persist a run.  Nothing here prices anything.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from quantark.util.exceptions import ValidationError

from .lifecycle import LifecycleRecord
from .measures import MEASURE_COLUMNS, path_measures

# ... FLOAT_COLUMNS, BOOL_COLUMNS, TRADE_COLUMNS, StateCube exactly as in engine.py ...

EVENT_COLUMNS = ("product", "path", "day", "event", "index", "spot", "barrier", "cashflow")
SUMMARY_COLUMNS: Tuple[str, ...] = (
    "path", "days", "termination_reason", "knocked_in", "ko_observation_index", "terminal_pnl",
) + MEASURE_COLUMNS
```

`EnsembleResults` keeps its fields and the three `path_*` methods (use `EVENT_COLUMNS` in `path_events`), and gains:

```python
    _summary: Optional[pd.DataFrame] = field(default=None, init=False, repr=False, compare=False)

    @property
    def notional(self) -> float:
        """The book's unit notional, ``sum |quantity| * initial_price * contract_multiplier``."""
        value = self.manifest.get("book_notional")
        if value is None:
            raise ValidationError("the manifest carries no book_notional; these results predate plan 4")
        return float(value)

    @property
    def summary(self) -> pd.DataFrame:
        """One row per path: termination, flags, terminal P&L and the hedge measures (memoised)."""
        if self._summary is None:
            self._summary = self._build_summary()
        return self._summary

    def _build_summary(self) -> pd.DataFrame:
        notional = self.notional
        terminal: Dict[int, Tuple[str, int]] = {}
        for e in self.events:                      # day order: the LAST terminal event wins
            if e.event in ("knock_out", "maturity"):
                terminal[int(e.path)] = (e.event, int(e.index))
        rows = []
        for i in range(self.n_paths):
            last = int(self.last_day[i])
            reason, ko_index = terminal.get(i, ("data_end", -1))
            row = {
                "path": i, "days": last + 1, "termination_reason": reason,
                "knocked_in": bool(self.cube.knocked_in[i, last]),
                "ko_observation_index": int(ko_index) if reason == "knock_out" else -1,
                "terminal_pnl": float(self.cube.total_pnl[i, last]),
            }
            row.update(path_measures(self.path_states(i), self.path_trades(i), notional=notional))
            rows.append(row)
        return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
```

In `engine.py`: delete lines 40–131, add `from .pricing.cache import LIBRARY_VERSION` and `from .results import BOOL_COLUMNS, EVENT_COLUMNS, FLOAT_COLUMNS, TRADE_COLUMNS, EnsembleResults, StateCube  # noqa: F401  (re-exported)`, and in `_manifest` add:

```python
            "library_version": LIBRARY_VERSION,
            "book_notional": self._book_notional(),
```

with

```python
    def _book_notional(self) -> float:
        """``sum |quantity| * initial_price * contract_multiplier`` over the book (the summary's bp base)."""
        return float(sum(
            abs(float(bp.quantity)) * float(bp.product.initial_price)
            * float(getattr(bp.product, "contract_multiplier", 1.0))
            for bp in self.config.products
        ))
```

`runner._merge_manifests` spreads the head manifest, so both keys survive batching unchanged (the same book, the same library).

- [ ] **Step 4: Run the tests and the package suite**

Run: `.venv/bin/python -m pytest -n0 test/simulation -q`
Expected: all pass (the moved names are found through `engine` by `runner.py`, `conformance.py` and the tests).

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/results.py quantark/backtest/simulation/engine.py test/simulation/test_results.py
git commit -m "feat(backtest/simulation): results module with the per-path summary in the study's measures

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `distribution`

**Files:**
- Modify: `quantark/backtest/simulation/results.py`
- Modify: `test/simulation/test_results.py`

**Interfaces:**
- `EnsembleResults.distribution(measure: str, *, es_level: float, tail: Literal["lower", "upper"] = "lower") -> Dict[str, Any]` with keys `measure, n, n_paths, mean, std, quantiles {q01, q05, q25, q50, q75, q95, q99}, es_level, tail, expected_shortfall, share_positive, ko_frequency, ki_frequency, maturity_frequency, data_end_frequency`. `measure` is any of `MEASURE_COLUMNS` or `terminal_pnl`. NaNs are dropped and `n` counts what remains; `std` is `ddof=1` and `None` below two values. Expected shortfall is the mean of the values at or below the `es_level` quantile (`lower`, the loss tail of a P&L measure) or at or above the `1 - es_level` quantile (`upper`, for a cost-like measure). The frequencies are over ALL paths.
- `QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)`.

- [ ] **Step 1: Write the failing tests**

Append to `test/simulation/test_results.py`:

```python
def test_distribution_reports_moments_quantiles_and_the_loss_tail(results):
    d = results.distribution("terminal_pnl_bp", es_level=0.25)
    values = results.summary["terminal_pnl_bp"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    assert d["n"] == values.size and d["n_paths"] == results.n_paths
    assert d["mean"] == float(values.mean()) and d["std"] == float(values.std(ddof=1))
    assert d["quantiles"]["q50"] == pytest.approx(float(np.quantile(values, 0.5)), rel=1e-12)
    assert set(d["quantiles"]) == {"q01", "q05", "q25", "q50", "q75", "q95", "q99"}
    cut = np.quantile(values, 0.25)
    assert d["expected_shortfall"] == pytest.approx(float(values[values <= cut].mean()), rel=1e-12)
    assert d["share_positive"] == float((values > 0).mean())
    upper = results.distribution("cost_bp", es_level=0.25, tail="upper")
    costs = results.summary["cost_bp"].to_numpy(dtype=float)
    assert upper["expected_shortfall"] == pytest.approx(float(costs[costs >= np.quantile(costs, 0.75)].mean()), rel=1e-12)


def test_distribution_frequencies_cover_every_path(results):
    d = results.distribution("terminal_pnl", es_level=0.05)
    reasons = results.summary["termination_reason"]
    assert d["ko_frequency"] == float((reasons == "knock_out").mean())
    assert d["maturity_frequency"] == float((reasons == "maturity").mean())
    assert d["ki_frequency"] == float(results.summary["knocked_in"].mean())
    assert d["ko_frequency"] + d["maturity_frequency"] + d["data_end_frequency"] == pytest.approx(1.0)


def test_distribution_fails_closed_on_bad_arguments(results):
    with pytest.raises(ValidationError):
        results.distribution("sharpe", es_level=0.05)
    with pytest.raises(ValidationError):
        results.distribution("terminal_pnl_bp", es_level=0.0)
    with pytest.raises(ValidationError):
        results.distribution("terminal_pnl_bp", es_level=0.05, tail="middle")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_results.py -q -k distribution`
Expected: `AttributeError: 'EnsembleResults' object has no attribute 'distribution'`.

- [ ] **Step 3: Write the implementation**

In `results.py`:

```python
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
DISTRIBUTION_MEASURES = ("terminal_pnl",) + MEASURE_COLUMNS
```

on `EnsembleResults`:

```python
    def distribution(self, measure: str, *, es_level: float, tail: str = "lower") -> Dict[str, Any]:
        """Moments, quantiles, expected shortfall and event frequencies of one measure over the paths.

        Expected shortfall is the mean of the ``es_level`` tail: the values
        at or below that quantile (``lower``, the loss tail of a P&L measure)
        or at or above the ``1 - es_level`` quantile (``upper``, for a
        cost-like measure).  NaN values are dropped and ``n`` counts what
        remains; the frequencies are over every path.
        """
        if measure not in DISTRIBUTION_MEASURES:
            raise ValidationError(f"unknown measure {measure!r}; one of {DISTRIBUTION_MEASURES}")
        if not 0.0 < float(es_level) < 1.0:
            raise ValidationError("es_level must lie strictly between 0 and 1")
        if tail not in ("lower", "upper"):
            raise ValidationError("tail must be 'lower' or 'upper'")
        summary = self.summary
        values = summary[measure].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        reasons = summary["termination_reason"]
        out: Dict[str, Any] = {
            "measure": measure, "n": int(finite.size), "n_paths": int(self.n_paths),
            "es_level": float(es_level), "tail": tail,
            "ko_frequency": float((reasons == "knock_out").mean()),
            "ki_frequency": float(summary["knocked_in"].astype(bool).mean()),
            "maturity_frequency": float((reasons == "maturity").mean()),
            "data_end_frequency": float((reasons == "data_end").mean()),
        }
        if finite.size == 0:
            out.update(mean=None, std=None, quantiles={}, expected_shortfall=None, share_positive=None)
            return out
        if tail == "lower":
            cut = float(np.quantile(finite, float(es_level)))
            tail_values = finite[finite <= cut]
        else:
            cut = float(np.quantile(finite, 1.0 - float(es_level)))
            tail_values = finite[finite >= cut]
        out.update(
            mean=float(finite.mean()),
            std=float(finite.std(ddof=1)) if finite.size > 1 else None,
            quantiles={f"q{int(round(p * 100)):02d}": float(v) for p, v in zip(QUANTILES, np.quantile(finite, QUANTILES))},
            expected_shortfall=float(tail_values.mean()),
            share_positive=float((finite > 0.0).mean()),
        )
        return out
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_results.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/results.py test/simulation/test_results.py
git commit -m "feat(backtest/simulation): distribution of a hedge measure with quantiles, expected shortfall and event frequencies

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `paired` and `take`

**Files:**
- Modify: `quantark/backtest/simulation/results.py`
- Modify: `test/simulation/test_results.py`

**Design.** Matched paths are proved by the data, not by a label: two results are paired only if they have the same number of paths, the same calendar and identical `spot`, `volatility` and `rate` cubes (bit for bit). The fingerprint is recorded when both manifests carry one. `variant.paired(base)` returns `variant − base` per path for `terminal_pnl` and every measure, plus `same_termination`. `take(indices)` is the sub-run of the given paths (cube rows, filtered and renumbered logs, `manifest["path_indices"]`), which is how a 2,000-path surface run is matched against a 200-path QUAD check run.

**Interfaces:**
- `PairedComparison(differences: pd.DataFrame, n_paths: int, path_fingerprint: Optional[str])` frozen, with `describe(measure) -> Dict[str, Any]` (`measure, n, mean, median, std, share_positive, t_stat`; `t_stat = mean / (std / sqrt(n))`, `None` when `std` is `None` or `0`).
- `EnsembleResults.paired(base: EnsembleResults) -> PairedComparison`.
- `EnsembleResults.take(indices: Sequence[int]) -> EnsembleResults`.

- [ ] **Step 1: Write the failing tests**

Append to `test/simulation/test_results.py`:

```python
from quantark.backtest.strategy.futures_delta_strategy import AutocallableDeltaHedgeStrategy
from quantark.backtest.simulation.results import PairedComparison


def test_paired_with_itself_is_identically_zero(results):
    comparison = results.paired(results)
    assert isinstance(comparison, PairedComparison)
    assert comparison.n_paths == results.n_paths
    for name in MEASURE_COLUMNS + ("terminal_pnl",):
        diffs = comparison.differences[name].to_numpy(dtype=float)
        assert np.all((diffs == 0.0) | np.isnan(diffs)), name
    assert comparison.differences["same_termination"].all()
    d = comparison.describe("terminal_pnl_bp")
    assert d["n"] == results.n_paths and d["mean"] == 0.0 and d["share_positive"] == 0.0 and d["t_stat"] is None


def test_paired_measures_a_different_hedge_on_the_same_paths(results):
    half = EnsembleBacktestEngine(ensemble_config(
        strategy=AutocallableDeltaHedgeStrategy(delta_threshold=0.0, hedge_ratio=0.5, target_delta=0.0),
    )).run(_paths())
    comparison = half.paired(results)
    diffs = comparison.differences["terminal_pnl_bp"].to_numpy(dtype=float)
    assert np.any(diffs != 0.0)
    expected = half.summary["terminal_pnl_bp"].to_numpy() - results.summary["terminal_pnl_bp"].to_numpy()
    assert np.array_equal(diffs, expected)
    d = comparison.describe("terminal_pnl_bp")
    assert d["n"] == results.n_paths and d["t_stat"] is not None and 0.0 <= d["share_positive"] <= 1.0
    assert comparison.path_fingerprint == results.manifest["path_fingerprint"]


def test_paired_refuses_unmatched_paths(results):
    other = EnsembleBacktestEngine(ensemble_config()).run(_paths(seed=4))
    with pytest.raises(ValidationError):
        results.paired(other)
    with pytest.raises(ValidationError):
        results.paired(results.take([0, 1]))


def test_take_keeps_the_chosen_paths_and_renumbers_the_logs(results):
    sub = results.take([5, 2])
    assert sub.n_paths == 2 and sub.manifest["path_indices"] == [5, 2]
    assert np.array_equal(sub.cube.total_pnl[0], results.cube.total_pnl[5])
    assert np.array_equal(sub.last_day, results.last_day[[5, 2]])
    pd.testing.assert_frame_equal(sub.path_states(1), results.path_states(2))
    pd.testing.assert_frame_equal(sub.path_trades(0), results.path_trades(5).assign(path=0))
    assert {e.path for e in sub.events} <= {0, 1}
    assert sub.summary.drop(columns="path").iloc[1].equals(results.summary.drop(columns="path").iloc[2])
    with pytest.raises(ValidationError):
        results.take([results.n_paths])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_results.py -q -k "paired or take"`
Expected: `ImportError: cannot import name 'PairedComparison'`.

- [ ] **Step 3: Write the implementation**

In `results.py` (before `EnsembleResults`):

```python
@dataclass(frozen=True)
class PairedComparison:
    """``variant - base`` per matched path, for ``terminal_pnl`` and every measure."""

    differences: pd.DataFrame
    n_paths: int
    path_fingerprint: Optional[str]

    def describe(self, measure: str) -> Dict[str, Any]:
        """Mean, median, std, share positive and paired t-statistic of one measure's differences."""
        if measure not in self.differences.columns or measure in ("path", "same_termination"):
            raise ValidationError(f"unknown paired measure {measure!r}")
        values = self.differences[measure].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {"measure": measure, "n": 0, "mean": None, "median": None, "std": None,
                    "share_positive": None, "t_stat": None}
        std = float(finite.std(ddof=1)) if finite.size > 1 else None
        mean = float(finite.mean())
        t_stat = mean / (std / math.sqrt(finite.size)) if std not in (None, 0.0) else None
        return {
            "measure": measure, "n": int(finite.size), "mean": mean, "median": float(np.median(finite)),
            "std": std, "share_positive": float((finite > 0.0).mean()), "t_stat": t_stat,
        }
```

on `EnsembleResults`:

```python
    def _matches(self, other: "EnsembleResults") -> bool:
        """Same paths, proved by the data: the market columns are bit-identical up to each path's last day.

        Beyond a path's last day the cube repeats its final values, and a
        batch that settled early froze earlier than the whole run did, so
        only the days a path actually ran are compared; the last days
        themselves must agree (the lifecycle depends on the path alone).
        """
        if self.n_paths != other.n_paths or not self.cube.dates.equals(other.cube.dates):
            return False
        if not np.array_equal(self.last_day, other.last_day):
            return False
        for name in ("spot", "volatility", "rate"):
            mine, theirs = getattr(self.cube, name), getattr(other.cube, name)
            for i in range(self.n_paths):
                end = int(self.last_day[i]) + 1
                if not np.array_equal(mine[i, :end], theirs[i, :end]):
                    return False
        return True

    def paired(self, base: "EnsembleResults") -> PairedComparison:
        """``self - base`` per path; the two runs must be on the same paths."""
        if not self._matches(base):
            raise ValidationError(
                "paired comparison needs the same paths on both sides: same count, calendar and market columns"
            )
        columns = ["terminal_pnl", *MEASURE_COLUMNS]
        diff = self.summary[columns].to_numpy(dtype=float) - base.summary[columns].to_numpy(dtype=float)
        frame = pd.DataFrame(diff, columns=columns)
        frame.insert(0, "path", self.summary["path"].to_numpy())
        frame["same_termination"] = (
            self.summary["termination_reason"].to_numpy() == base.summary["termination_reason"].to_numpy()
        )
        mine, theirs = self.manifest.get("path_fingerprint"), base.manifest.get("path_fingerprint")
        return PairedComparison(differences=frame, n_paths=int(self.n_paths),
                                path_fingerprint=mine if mine == theirs else None)

    def take(self, indices: Sequence[int]) -> "EnsembleResults":
        """The sub-run of the given paths, in the given order; logs renumbered, manifest annotated."""
        idx = [int(i) for i in indices]
        if not idx or any(not 0 <= i < self.n_paths for i in idx):
            raise ValidationError(f"path indices {idx} out of range for {self.n_paths} paths")
        position = {old: new for new, old in enumerate(idx)}
        cube = StateCube(self.cube.dates, len(idx))
        cube.active_contract = list(self.cube.active_contract)
        for name in FLOAT_COLUMNS + BOOL_COLUMNS:
            setattr(cube, name, np.array(getattr(self.cube, name)[idx]))
        trades = [{**t, "path": position[int(t["path"])]} for t in self.trades if int(t["path"]) in position]
        events = [LifecycleRecord(**{**e.__dict__, "path": position[e.path]}) for e in self.events if e.path in position]
        manifest = {**self.manifest, "path_indices": idx}
        return EnsembleResults(cube=cube, trades=trades, events=events, manifest=manifest,
                               last_day=np.array(self.last_day[idx]),
                               initial_book_value=np.array(self.initial_book_value[idx]))
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_results.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/results.py test/simulation/test_results.py
git commit -m "feat(backtest/simulation): paired comparison on matched paths and sub-run selection

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: `to_dir` / `from_dir`

**Files:**
- Modify: `quantark/backtest/simulation/results.py`
- Modify: `test/simulation/test_results.py`

**Format** (`RESULTS_FORMAT = 1`): `cube.npz` (every float and bool column, `dates_ns` int64 nanoseconds, `active_contract` as a str array, `last_day`, `initial_book_value`), `trades.csv` (`TRADE_COLUMNS`, ISO dates), `events.csv` (`EVENT_COLUMNS`), `summary.csv` (a convenience export; `from_dir` recomputes the summary from the cube so it is exactly the original's), `manifest.json` (`{"results_format": 1, "manifest": jsonable(manifest)}`). `from_dir` refuses a missing file or another format with `ValidationError`.

**Interfaces:**
- `jsonable(value) -> Any` (dicts, lists, numpy scalars, NaN → `None`, timestamps → ISO) — also used by the study.
- `EnsembleResults.to_dir(path) -> Path`, `EnsembleResults.from_dir(path) -> EnsembleResults`.

- [ ] **Step 1: Write the failing tests**

Append to `test/simulation/test_results.py`:

```python
from quantark.backtest.simulation.results import EVENT_COLUMNS, jsonable


def _clean_events(events):
    out = []
    for e in events:
        d = dict(e.__dict__)
        d["barrier"] = None if np.isnan(d["barrier"]) else d["barrier"]
        out.append(d)
    return out


def test_results_round_trip_through_a_directory(results, tmp_path):
    out = results.to_dir(tmp_path / "run")
    assert {p.name for p in out.iterdir()} == {"cube.npz", "trades.csv", "events.csv", "summary.csv", "manifest.json"}
    loaded = EnsembleResults.from_dir(out)
    for name in ("total_pnl", "product_mtm", "delta", "spot", "pending_receivable_pv"):
        assert np.array_equal(getattr(loaded.cube, name), getattr(results.cube, name), equal_nan=True), name
    assert np.array_equal(loaded.cube.alive, results.cube.alive)
    assert loaded.cube.active_contract == results.cube.active_contract and loaded.cube.dates.equals(results.cube.dates)
    assert np.array_equal(loaded.last_day, results.last_day)
    assert np.array_equal(loaded.initial_book_value, results.initial_book_value)
    for i in range(results.n_paths):
        pd.testing.assert_frame_equal(loaded.path_states(i), results.path_states(i))
        pd.testing.assert_frame_equal(loaded.path_trades(i), results.path_trades(i))
    assert _clean_events(loaded.events) == _clean_events(results.events)
    assert loaded.manifest == jsonable(results.manifest)
    pd.testing.assert_frame_equal(loaded.summary, results.summary)
    assert loaded.paired(results).describe("terminal_pnl_bp")["mean"] == 0.0


def test_from_dir_fails_closed_on_a_missing_file_or_another_format(results, tmp_path):
    out = results.to_dir(tmp_path / "run")
    (out / "events.csv").unlink()
    with pytest.raises(ValidationError):
        EnsembleResults.from_dir(out)
    out = results.to_dir(tmp_path / "other")
    (out / "manifest.json").write_text('{"results_format": 99, "manifest": {}}')
    with pytest.raises(ValidationError):
        EnsembleResults.from_dir(out)


def test_jsonable_handles_numpy_nan_and_dates():
    value = {"a": np.float64(1.5), "b": float("nan"), "c": np.int64(3), "d": pd.Timestamp("2024-01-02"),
             "e": [np.bool_(True), {"f": np.array([1, 2])}]}
    assert jsonable(value) == {"a": 1.5, "b": None, "c": 3, "d": "2024-01-02T00:00:00", "e": [True, {"f": [1, 2]}]}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_results.py -q -k "round_trip or fails_closed or jsonable"`
Expected: `ImportError: cannot import name 'jsonable'`.

- [ ] **Step 3: Write the implementation**

In `results.py`:

```python
import datetime as _dt

RESULTS_FORMAT = 1
_RESULT_FILES = ("cube.npz", "trades.csv", "events.csv", "summary.csv", "manifest.json")


def jsonable(value: Any) -> Any:
    """A JSON-serialisable copy: numpy scalars to Python, NaN to None, timestamps to ISO strings."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (pd.Timestamp, _dt.datetime, _dt.date)):
        return value.isoformat()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    return value


def _dates_as_ns(dates: pd.DatetimeIndex) -> np.ndarray:
    """Pinned to nanoseconds; ``asi8`` counts in the index's own inferred unit (see MarketPath)."""
    return np.asarray(dates.values, dtype="datetime64[ns]").astype(np.int64)
```

on `EnsembleResults`:

```python
    def to_dir(self, path) -> Path:
        """Persist the cube (npz), the logs and the summary (csv) and the manifest (json)."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        cube = self.cube
        arrays = {name: np.asarray(getattr(cube, name)) for name in FLOAT_COLUMNS + BOOL_COLUMNS}
        np.savez_compressed(
            out / "cube.npz", dates_ns=_dates_as_ns(cube.dates), last_day=np.asarray(self.last_day),
            initial_book_value=np.asarray(self.initial_book_value),
            active_contract=np.array(cube.active_contract, dtype=str), **arrays,
        )
        pd.DataFrame(self.trades, columns=TRADE_COLUMNS).to_csv(out / "trades.csv", index=False)
        pd.DataFrame([e.__dict__ for e in self.events], columns=EVENT_COLUMNS).to_csv(out / "events.csv", index=False)
        self.summary.to_csv(out / "summary.csv", index=False)
        (out / "manifest.json").write_text(
            json.dumps({"results_format": RESULTS_FORMAT, "manifest": jsonable(self.manifest)}, indent=2, sort_keys=True)
        )
        return out

    @classmethod
    def from_dir(cls, path) -> "EnsembleResults":
        """Read a run written by ``to_dir``; the summary is recomputed from the cube."""
        src = Path(path)
        for name in _RESULT_FILES:
            if not (src / name).exists():
                raise ValidationError(f"results directory {src} is missing {name}")
        header = json.loads((src / "manifest.json").read_text())
        if header.get("results_format") != RESULTS_FORMAT:
            raise ValidationError(
                f"results at {src} are format {header.get('results_format')!r}; this library reads {RESULTS_FORMAT}"
            )
        with np.load(src / "cube.npz", allow_pickle=False) as z:
            dates = pd.DatetimeIndex(np.asarray(z["dates_ns"], dtype=np.int64).astype("datetime64[ns]"))
            last_day = np.array(z["last_day"], dtype=np.int64)
            cube = StateCube(dates, int(last_day.size))
            cube.active_contract = [str(c) for c in z["active_contract"]]
            for name in FLOAT_COLUMNS + BOOL_COLUMNS:
                setattr(cube, name, np.array(z[name]))
            initial = np.array(z["initial_book_value"], dtype=float)
        trades_frame = pd.read_csv(src / "trades.csv")
        trades = []
        for row in trades_frame.to_dict("records"):
            row["path"], row["day"] = int(row["path"]), int(row["day"])
            row["date"] = pd.Timestamp(row["date"])
            for key in ("quantity", "price", "multiplier", "notional", "transaction_cost"):
                row[key] = float(row[key])
            trades.append(row)
        events_frame = pd.read_csv(src / "events.csv")
        events = [
            LifecycleRecord(product=int(r["product"]), path=int(r["path"]), day=int(r["day"]), event=str(r["event"]),
                            index=int(r["index"]), spot=float(r["spot"]), barrier=float(r["barrier"]),
                            cashflow=float(r["cashflow"]))
            for r in events_frame.to_dict("records")
        ]
        return cls(cube=cube, trades=trades, events=events, manifest=header["manifest"],
                   last_day=last_day, initial_book_value=initial)
```

(`pd.read_csv` reads a header-only CSV as an empty frame, so a run without trades or events round-trips; an empty `barrier` cell reads as NaN, which is what the record held.)

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_results.py test/simulation/test_runner.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/results.py test/simulation/test_results.py
git commit -m "feat(backtest/simulation): persist and reload a run (cube npz, logs csv, manifest json)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: The life surface keeps only the columns a run reads

**Files:**
- Modify: `quantark/backtest/simulation/pricing/surface.py`
- Modify: `test/simulation/test_surface.py`

**Design.** After a solve the pricer knows every column the run can read: column 0, the terminal column, and the node of each calendar day up to maturity. `LifeSurface.select(columns)` returns the surface restricted to those columns with `step_of` remapped; a readout of a kept column is bit-identical because the column data is copied, not recomputed. `LifeSurfacePricer(..., compact=True)` applies it; `compact=False` keeps the whole slab (the test compares the two).

**Interfaces:**
- `LifeSurface.select(columns: Sequence[int]) -> LifeSurface`.
- `LifeSurfacePricer(..., compact: bool = True)`; `stats()["surface_cache"]["bytes_used"]` reflects the compact size.

- [ ] **Step 1: Write the failing tests**

Append to `test/simulation/test_surface.py`:

```python
def test_select_keeps_columns_bit_for_bit_and_remaps_the_nodes():
    pricer = _surface(vol_step=0.0, q_step=0.0, compact=False)
    pricer.price_day(_states(1, [SPOT]))
    (key, full), = pricer._surfaces._store.items()
    node = full.node(pricer.elapsed(2))
    sub = full.select([0, node, full.t.size - 1])
    assert sub.t.size == 3 and sub.nbytes < full.nbytes / 5      # 3 of ~25 columns on the six-day fixture
    assert sub.node(pricer.elapsed(2)) == 1 and sub.step_of[pricer.elapsed(2)] == 1
    x = np.log(SPOT * np.exp(np.array([-0.01, 0.0, 0.004])))
    for ki in (False, True):
        flags = np.full(3, ki)
        for a, b in zip(full.readout(x, node, flags), sub.readout(x, 1, flags)):
            assert np.array_equal(a, b)
    assert np.array_equal(sub.v1[:, -1], full.v1[:, -1])
    with pytest.raises(ValidationError):
        full.select([full.t.size])


def test_a_compact_pricer_prices_exactly_like_the_full_one_with_less_memory():
    full, compact = _surface(vol_step=0.0, q_step=0.0, compact=False), _surface(vol_step=0.0, q_step=0.0)
    for d in range(0, 5):
        states = _states(d, SPOT * np.exp(np.linspace(-0.02, 0.02, 5)), knocked_in=[False, True, False, True, False])
        for a, b in zip(full.price_day(states), compact.price_day(states)):
            assert np.array_equal(a, b)
    assert compact.stats()["surface_cache"]["bytes_used"] < full.stats()["surface_cache"]["bytes_used"] / 2
```

and give `_surface` the keyword: `def _surface(vol_step=0.01, q_step=0.0025, budget=200_000_000, compact=True)` passing `compact=compact` through.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_surface.py -q`
Expected: `TypeError: ... unexpected keyword argument 'compact'`.

- [ ] **Step 3: Write the implementation**

On `LifeSurface`:

```python
    def select(self, columns: Sequence[int]) -> "LifeSurface":
        """This surface restricted to ``columns`` (sorted, unique), ``step_of`` remapped.

        Column data is copied, so a readout of a kept column is bit-identical
        to the readout of the original.  Keep column 0 and the terminal
        column: ``readout`` reads ``t0_readout`` at column 0 and ``node``
        maps the maturity to the last column.
        """
        keep = np.array(sorted({int(k) for k in columns}), dtype=np.int64)
        if keep.size == 0 or keep[0] < 0 or keep[-1] >= self.t.size:
            raise ValidationError(f"columns must index the surface's {self.t.size} time nodes")
        position = {int(k): n for n, k in enumerate(keep)}
        keeps_zero = 0 in position
        return LifeSurface(
            t=np.array(self.t[keep]), x=self.x,
            v0=np.ascontiguousarray(self.v0[:, keep]), v1=np.ascontiguousarray(self.v1[:, keep]),
            d0=np.ascontiguousarray(self.d0[:, keep]), g0=np.ascontiguousarray(self.g0[:, keep]),
            d1=np.ascontiguousarray(self.d1[:, keep]), g1=np.ascontiguousarray(self.g1[:, keep]),
            step_of={t: position[k] for t, k in self.step_of.items() if k in position},
            t0_readout=self.t0_readout if keeps_zero else None,
            t0_delta=self.t0_delta if keeps_zero else None,
            t0_gamma=self.t0_gamma if keeps_zero else None,
        )
```

On `LifeSurfacePricer`: constructor keyword `compact: bool = True` stored as `self.compact`; in `_surface`, after building `surface` and before `self._surfaces.put(key, surface)`:

```python
        if self.compact:
            surface = surface.select(self._read_columns(surface))
```

with

```python
    def _read_columns(self, surface: LifeSurface) -> List[int]:
        """Column 0, the terminal column, and the node of every calendar day up to maturity."""
        columns = {0, int(surface.t.size - 1)}
        tau = float(surface.t[-1])
        for d in range(1, len(self.dates)):
            elapsed = self.elapsed(d)
            if elapsed > tau and not is_close(elapsed, tau):
                break
            columns.add(surface.node(elapsed))
        return sorted(columns)
```

- [ ] **Step 4: Run the surface, engine and conformance tests**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_surface.py test/simulation/test_engine.py test/simulation/test_conformance.py -q`
Expected: all pass; the surface oracle case reports the same gaps as before (the readouts are bit-identical).

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/pricing/surface.py test/simulation/test_surface.py
git commit -m "perf(backtest/simulation): a life surface keeps only the columns the run reads, readouts bit-identical

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: The study's shared module and `01_build_paths.py`

**Files:**
- Create: `example/snowball_simulated_paths/_common.py`
- Create: `example/snowball_simulated_paths/01_build_paths.py`
- Create: `test/test_snowball_simulated_paths_study.py`

**Design.** The study reuses the q term-structure study's `_common.py` (terms, product, fair-coupon solver, `Q_MODELS`, `HEDGE_POLICIES`, `load_history`, `surface_history`, `atm_vol_channel`, `TradingCalendar`) by loading it from its path under a fixed module name, as `test/test_snowball_q_term_structure_study.py` does. Stage 01 turns the history into a `PathHistory`, generates the bootstrap batch from the last day's snapshot on a forward weekday calendar, builds the stress set on the same calendar, and writes both batches plus a manifest of fingerprints and parameters. Every stage keeps its loaders in `main` and its work in functions that take frames or paths, so a test runs them on synthetic frames.

**Interfaces (`_common.py`):**
- `Q` — the loaded q-study module (`load_q_study()`); `STUDY_DIR`, `PROJECT_ROOT`, `DATA_DIR`, `DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "snowball_simulated_paths"`, `DEFAULT_HISTORICAL_DIR = PROJECT_ROOT / "output" / "snowball_q_term_structure"`.
- `MODELS = ("flat_active", "term_flat_q", "term_opt_tail")`, `HEDGES = ("front", "far")`, `BASELINE_CELL = "flat_active__front"`, `cell_name(model, hedge)`.
- Parameters: `N_PATHS = 2000`, `N_DAYS = 275` (a 12-month product matures about 261 weekdays after inception and the engine refuses a calendar that ends before the book settles), `QUICK_PATHS = 40`, `MEAN_BLOCK_DAYS = 20`, `ANNUAL_DRIFT = 0.0`, `VOL_FLOOR = 0.08`, `CARRY_MODE = "changes"`, `SEED = 1`, `COST_BP = 1.0`, `SPOT_STEP = 0.0025`, `VOL_STEP = 0.01`, `Q_STEP = 0.0025`, `SURFACE_CACHE_BYTES = 2_000_000_000`, `STATE_CACHE_BYTES = 500_000_000`, `GATE_SURFACE = dict(sample_states=64, pv_tolerance_bp=25.0, delta_tolerance_hands=2.0)`, `GATE_LADDER = dict(sample_states=64, pv_tolerance_bp=10.0, delta_tolerance_hands=2.0)`, `CHECK_PATHS = 200`, `ORACLE_PATHS = 3`.
- `engine_config(model: str, engine: str, *, quad_grid: int) -> AutocallableEngineConfig` — `engine` is `"pde"` (`PDEParams()`, for the life surface) or `"quad"` (`QuadParams(grid_points=quad_grid)`), with the model's `dividend_source`, `futures_curve_extrapolation` and `futures_curve_min_tenor_days` from `Q.Q_MODELS[model]`.
- `stress_paths(start: StartState, calendar, tenor_grid) -> MarketPath` — five scenarios of `len(calendar)` days: `crash_into_ki(0.30, 20, n)`, `v_shape(0.28, 20, 40, n)`, `vol_spike(0.15, 40, n)`, `basis_blowout(-0.05, 10, n)`, `grind_up_to_ko(0.002, 60, n)`.
- `write_json(path, payload)`, `read_json(path)`.

**Interfaces (`01_build_paths.py`):**
- `build_history(spot: pd.DataFrame, vol: pd.DataFrame, futures: pd.DataFrame, *, rate: float) -> PathHistory`.
- `load_real_history(history_dir, *, rate, vol_tenor) -> PathHistory` (the q study's loaders; raises the q study's `StudyDataError` when a cache is missing).
- `build_paths(history, *, n_paths, n_days, seed, mean_block_days, annual_drift, vol_floor, carry_mode) -> Tuple[MarketPath, MarketPath]` (bootstrap, stress), both on `trading_calendar(next weekday after history.dates[-1], n_days)` from `history.snapshot()`.
- `write_paths(out_dir, bootstrap, stress, *, history) -> Dict[str, Any]` writes `paths/bootstrap.npz`, `paths/stress.npz`, `paths/manifest.json` and returns the manifest (fingerprints, `history_fingerprint`, parameters, calendar bounds).
- `load_paths(out_dir) -> Tuple[MarketPath, MarketPath, Dict[str, Any]]`.
- CLI: `--history-dir`, `--out-dir`, `--n-paths`, `--n-days`, `--seed`, `--mean-block-days`, `--vol-floor`, `--rate`, `--vol-tenor`, `--quick` (`QUICK_PATHS`).

- [ ] **Step 1: Write the failing tests**

`test/test_snowball_simulated_paths_study.py`:

```python
"""Tests for ``example/snowball_simulated_paths``: every stage on synthetic frames.

The study's real inputs are local, untracked caches; these tests build a
synthetic history and a one-month snowball so the whole pipeline -- paths,
fair coupon, cells, oracle spot check, report -- runs in seconds.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.simulation.paths.market_path import DEFAULT_TENOR_GRID, MarketPath, trading_calendar

REPO = Path(__file__).resolve().parents[1]
STUDY_DIR = REPO / "example" / "snowball_simulated_paths"
RATE = 0.02


def _load(name: str):
    path = STUDY_DIR / name
    spec = importlib.util.spec_from_file_location(f"snowball_simulated_paths_{name.replace('.py', '')}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


C = _load("_common.py")
S01 = _load("01_build_paths.py")


def synthetic_frames(n_days: int = 80):
    """Spot, vol and a four-contract IM chain with a flat -10% carry, like the plan-1 tests."""
    dates = trading_calendar(date(2024, 1, 2), n_days)
    rng = np.random.default_rng(0)
    spot = 6000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, n_days)))
    spot_df = pd.DataFrame({"date": dates, "spot": spot})
    vol_df = pd.DataFrame({"date": dates, "volatility": 0.22 + 0.02 * np.sin(np.arange(n_days) / 9.0)})
    chain = [("IM2401", pd.Timestamp("2024-01-19")), ("IM2402", pd.Timestamp("2024-02-23")),
             ("IM2403", pd.Timestamp("2024-03-15")), ("IM2406", pd.Timestamp("2024-06-21")),
             ("IM2409", pd.Timestamp("2024-09-20"))]
    rows = []
    for k, d in enumerate(dates):
        for code, expiry in chain:
            t = (expiry - d).days / 365.0
            if t <= 0:
                continue
            rows.append({"date": d, "contract": code, "futures_price": float(spot[k]) * np.exp(-0.10 * t),
                         "expiry_date": expiry, "multiplier": 200.0})
    return spot_df, vol_df, pd.DataFrame(rows)


def test_build_history_and_paths_from_frames():
    spot, vol, futures = synthetic_frames()
    history = S01.build_history(spot, vol, futures, rate=RATE)
    assert history.n_days == 80 and history.tenor_grid.shape == DEFAULT_TENOR_GRID.shape
    bootstrap, stress = S01.build_paths(history, n_paths=4, n_days=30, seed=7, mean_block_days=5,
                                        annual_drift=0.0, vol_floor=0.08, carry_mode="changes")
    assert (bootstrap.n_paths, bootstrap.n_days) == (4, 30) and stress.n_days == 30 and stress.n_paths == 5
    assert bootstrap.dates[0] > history.dates[-1] and bootstrap.dates.equals(stress.dates)
    assert bootstrap.spot[:, 0] == pytest.approx(history.spot[-1])
    assert bootstrap.meta["seed"] == 7 and stress.meta["scenario_names"][0].startswith("crash_into_ki")
    again, _ = S01.build_paths(history, n_paths=4, n_days=30, seed=7, mean_block_days=5,
                               annual_drift=0.0, vol_floor=0.08, carry_mode="changes")
    assert again.fingerprint() == bootstrap.fingerprint()


def test_paths_are_written_with_a_manifest_and_read_back(tmp_path):
    spot, vol, futures = synthetic_frames()
    history = S01.build_history(spot, vol, futures, rate=RATE)
    bootstrap, stress = S01.build_paths(history, n_paths=3, n_days=20, seed=1, mean_block_days=5,
                                        annual_drift=0.0, vol_floor=0.08, carry_mode="changes")
    manifest = S01.write_paths(tmp_path, bootstrap, stress, history=history)
    assert manifest["bootstrap_fingerprint"] == bootstrap.fingerprint()
    assert manifest["history_fingerprint"] == history.source_fingerprint
    b, s, m = S01.load_paths(tmp_path)
    assert b.fingerprint() == bootstrap.fingerprint() and s.fingerprint() == stress.fingerprint() and m == manifest


def test_the_study_reuses_the_q_study_and_names_its_cells():
    assert C.Q.Q_MODELS["term_opt_tail"].dividend_source == "futures_curve"
    assert C.cell_name("term_flat_q", "far") == "term_flat_q__far"
    cfg = C.engine_config("term_opt_tail", "pde", quad_grid=101)
    assert cfg.dividend_source == "futures_curve" and cfg.futures_curve_extrapolation == "surface_forward_carry"
    assert cfg.futures_curve_min_tenor_days == 1
    assert C.engine_config("flat_active", "quad", quad_grid=101).dividend_source is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/test_snowball_simulated_paths_study.py -q`
Expected: `FileNotFoundError` loading `_common.py`.

- [ ] **Step 3: Write `_common.py`**

```python
"""Shared pieces of the simulated-path snowball study.

The product, the fair-coupon solver, the carry models and the hedge
policies are the q term-structure study's (``example/snowball_q_term_structure/_common.py``),
loaded here under a fixed module name so the two studies cannot drift.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from quantark.asset.equity.param import PDEParams, QuadParams
from quantark.backtest.replay import AutocallableEngineConfig
from quantark.backtest.simulation import MarketPath, SnowballStressLibrary, StartState, stress_set
from quantark.backtest.simulation.results import jsonable
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import ValidationError

STUDY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_DIR.parents[1]
Q_STUDY_DIR = PROJECT_ROOT / "example" / "snowball_q_term_structure"
DATA_DIR = STUDY_DIR / "data"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "snowball_simulated_paths"
DEFAULT_HISTORICAL_DIR = PROJECT_ROOT / "output" / "snowball_q_term_structure"


def load_q_study():
    """The q term-structure study's helpers, once per process."""
    name = "q_term_structure_common"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Q_STUDY_DIR / "_common.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


Q = load_q_study()

MODELS = ("flat_active", "term_flat_q", "term_opt_tail")
HEDGES = ("front", "far")
BASELINE_CELL = "flat_active__front"

N_PATHS = 2000
N_DAYS = 275          # a 12-month product matures ~261 weekdays out; the engine refuses a shorter calendar
QUICK_PATHS = 40
MEAN_BLOCK_DAYS = 20
ANNUAL_DRIFT = 0.0
VOL_FLOOR = 0.08
CARRY_MODE = "changes"
SEED = 1
COST_BP = 1.0
SPOT_STEP = 0.0025
VOL_STEP = 0.01
Q_STEP = 0.0025
SURFACE_CACHE_BYTES = 2_000_000_000
STATE_CACHE_BYTES = 500_000_000
GATE_SURFACE = dict(sample_states=64, pv_tolerance_bp=25.0, delta_tolerance_hands=2.0)
GATE_LADDER = dict(sample_states=64, pv_tolerance_bp=10.0, delta_tolerance_hands=2.0)
CHECK_PATHS = 200
ORACLE_PATHS = 3


def cell_name(model: str, hedge: str) -> str:
    return f"{model}__{hedge}"


def engine_config(model: str, engine: str, *, quad_grid: int) -> AutocallableEngineConfig:
    """The replay engine config of one carry model on the PDE (life surface) or QUAD (repricing) engine."""
    if model not in Q.Q_MODELS:
        raise ValidationError(f"unknown carry model {model!r}; one of {tuple(Q.Q_MODELS)}")
    q_model = Q.Q_MODELS[model]
    if engine == "pde":
        kwargs: Dict[str, Any] = dict(pricing_engine_type=EngineType.PDE, pde_params=PDEParams())
    elif engine == "quad":
        kwargs = dict(pricing_engine_type=EngineType.QUADRATURE, quad_params=QuadParams(grid_points=int(quad_grid)))
    else:
        raise ValidationError("engine must be 'pde' or 'quad'")
    return AutocallableEngineConfig(
        dividend_source=q_model.dividend_source, futures_curve_extrapolation=q_model.extrapolation,
        futures_curve_min_tenor_days=int(q_model.min_tenor_days), **kwargs,
    )


def stress_paths(start: StartState, calendar: pd.DatetimeIndex, tenor_grid: np.ndarray) -> MarketPath:
    """The five designed adverse paths on the run calendar."""
    n = len(calendar)
    lib = SnowballStressLibrary
    # The stated lengths on the full calendar; shortened on a short one so
    # every scenario fits (each library method checks its own bound).
    scenarios = [
        lib.crash_into_ki(0.30, min(20, n // 2), n),
        lib.v_shape(0.28, min(20, n // 3), min(40, n // 3), n),
        lib.vol_spike(0.15, min(40, n - 2), n),
        lib.basis_blowout(-0.05, min(10, n // 2), n),
        lib.grind_up_to_ko(0.002, min(60, n // 2), n),
    ]
    return stress_set(scenarios, start=start, calendar=calendar, tenor_grid=tenor_grid)


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())
```

(On the full 275-day calendar the `min` expressions give the stated 20 / 20+40 / 40 / 10 / 60 days; on the tests' 20- and 30-day calendars every scenario still fits its library bound.)

- [ ] **Step 4: Write `01_build_paths.py`**

```python
"""Stage 01: the realised history into simulated batches.

    .venv/bin/python example/snowball_simulated_paths/01_build_paths.py            # 2,000 x 275
    .venv/bin/python example/snowball_simulated_paths/01_build_paths.py --quick    # 40 paths

Reads the CSI 1000 spot, the IM chain and the admitted IV surfaces from
``example/mo_volmodels/data/history`` (local, untracked), builds the joint
daily history, bootstraps forward paths from its last day, adds the stress
set, and writes ``<out>/paths/{bootstrap,stress}.npz`` with a manifest.
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402

from quantark.backtest.simulation import (  # noqa: E402
    DEFAULT_TENOR_GRID, MarketPath, PathHistory, StationaryBlockBootstrap, trading_calendar,
)
from quantark.backtest.simulation.results import jsonable  # noqa: E402


def build_history(spot: pd.DataFrame, vol: pd.DataFrame, futures: pd.DataFrame, *, rate: float) -> PathHistory:
    """The joint daily state (ln S, vol, rate, carry curve) on the default tenor grid."""
    return PathHistory.from_frames(spot=spot, vol=vol, futures=futures, rate=float(rate), tenor_grid=DEFAULT_TENOR_GRID)


def load_real_history(history_dir, *, rate: float, vol_tenor: float) -> PathHistory:
    """The study's inputs through the q study's fail-closed loaders."""
    frames = C.Q.load_history(history_dir)
    surfaces = C.Q.surface_history(history_dir)
    vol = C.Q.atm_vol_channel(frames.dates, surfaces, vol_tenor)
    return build_history(frames.spot, vol, frames.futures, rate=rate)


def build_paths(
    history: PathHistory, *, n_paths: int, n_days: int, seed: int, mean_block_days: int,
    annual_drift: float, vol_floor: float, carry_mode: str,
) -> Tuple[MarketPath, MarketPath]:
    """The bootstrap batch and the stress set, both from the last day's state on a forward weekday calendar."""
    start = history.snapshot()
    first = (pd.Timestamp(history.dates[-1]) + timedelta(days=1)).date()
    calendar = trading_calendar(first, n_days)
    generator = StationaryBlockBootstrap(
        history, mean_block_days=mean_block_days, demean_returns=True, annual_drift=annual_drift,
        vol_floor=vol_floor, carry_mode=carry_mode, start=start, calendar=calendar,
    )
    bootstrap = generator.generate(n_paths, n_days, seed=seed)
    stress = C.stress_paths(start, calendar, history.tenor_grid)
    return bootstrap, stress


def write_paths(out_dir, bootstrap: MarketPath, stress: MarketPath, *, history: PathHistory) -> Dict[str, Any]:
    out = Path(out_dir) / "paths"
    out.mkdir(parents=True, exist_ok=True)
    bootstrap.to_npz(out / "bootstrap.npz")
    stress.to_npz(out / "stress.npz")
    manifest = jsonable({
        "bootstrap_fingerprint": bootstrap.fingerprint(), "stress_fingerprint": stress.fingerprint(),
        "history_fingerprint": history.source_fingerprint,
        "history_first_day": str(history.dates[0].date()), "history_last_day": str(history.dates[-1].date()),
        "n_paths": bootstrap.n_paths, "n_days": bootstrap.n_days,
        "calendar_first_day": str(bootstrap.dates[0].date()), "calendar_last_day": str(bootstrap.dates[-1].date()),
        "bootstrap_meta": bootstrap.meta, "stress_meta": stress.meta,
    })                                  # JSON-safe now, so it equals what load_paths reads back
    C.write_json(out / "manifest.json", manifest)
    return manifest


def load_paths(out_dir) -> Tuple[MarketPath, MarketPath, Dict[str, Any]]:
    out = Path(out_dir) / "paths"
    for name in ("bootstrap.npz", "stress.npz", "manifest.json"):
        if not (out / name).exists():
            raise C.Q.StudyDataError(f"missing {out / name}; run 01_build_paths.py first")
    return MarketPath.from_npz(out / "bootstrap.npz"), MarketPath.from_npz(out / "stress.npz"), C.read_json(out / "manifest.json")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history-dir", type=Path, default=C.Q.DEFAULT_HISTORY_DIR)
    parser.add_argument("--out-dir", type=Path, default=C.DEFAULT_OUT_DIR)
    parser.add_argument("--n-paths", type=int, default=C.N_PATHS)
    parser.add_argument("--n-days", type=int, default=C.N_DAYS)
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--mean-block-days", type=int, default=C.MEAN_BLOCK_DAYS)
    parser.add_argument("--vol-floor", type=float, default=C.VOL_FLOOR)
    parser.add_argument("--rate", type=float, default=C.Q.FLAT_RATE)
    parser.add_argument("--vol-tenor", type=float, default=C.Q.ATM_VOL_TENOR_YEARS)
    parser.add_argument("--quick", action="store_true", help=f"{C.QUICK_PATHS} paths")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    n_paths = C.QUICK_PATHS if args.quick else args.n_paths
    history = load_real_history(args.history_dir, rate=args.rate, vol_tenor=args.vol_tenor)
    bootstrap, stress = build_paths(
        history, n_paths=n_paths, n_days=args.n_days, seed=args.seed, mean_block_days=args.mean_block_days,
        annual_drift=C.ANNUAL_DRIFT, vol_floor=args.vol_floor, carry_mode=C.CARRY_MODE,
    )
    manifest = write_paths(args.out_dir, bootstrap, stress, history=history)
    print(f"history {manifest['history_first_day']}..{manifest['history_last_day']} ({history.n_days} days), "
          f"{bootstrap.n_paths} bootstrap paths x {bootstrap.n_days} days from {manifest['calendar_first_day']}, "
          f"{stress.n_paths} stress paths, vol floor hits {bootstrap.meta.get('vol_floor_hits')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest -n0 test/test_snowball_simulated_paths_study.py -q`
Expected: 3 passed. (If `stress_set` refuses the 30-day calendar, the `min` expressions in `stress_paths` are wrong — fix them there, not in the test.)

- [ ] **Step 6: Commit**

```bash
git add example/snowball_simulated_paths/_common.py example/snowball_simulated_paths/01_build_paths.py test/test_snowball_simulated_paths_study.py
git commit -m "feat(example): simulated-path snowball study, stage 01 — history to bootstrap and stress batches

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: `02_ensemble_fleet.py`

**Files:**
- Create: `example/snowball_simulated_paths/02_ensemble_fleet.py`
- Modify: `test/test_snowball_simulated_paths_study.py`

**Design.** One product for every cell: the q study's 1Y snowball built on the simulated calendar's first day at the start spot, its fair coupon solved ONCE under `term_flat_q` with the QUAD engine at the start state (so every cell starts from the traded price, `initial_price=0.0`, and paired terminal P&L differences are pure hedge P&L, as in the historical study). Cells are `MODELS × HEDGES`; each cell runs (a) the bootstrap batch with the life-surface provider on PDE, (b) the stress set with the same provider, (c) the first `CHECK_PATHS` bootstrap paths with the QUAD ladder (the engine-substitute check), and optionally (d) `--exact-paths N` with exact QUAD repricing (the only run in which the engine receives the term dividend object). Each run goes to `cells/<cell>[__stress|__ladder_quad|__exact_quad]/` via `to_dir`, with `config.json` (the config fingerprint and settings) and `oracle.json` (the spot check: `ORACLE_PATHS` single paths through `run_oracle` with the gate's tolerances). `--resume` skips a run whose `config.json` fingerprint matches.

**Interfaces:**
- `study_terms(calendar: pd.DatetimeIndex, *, maturity_months, lockout_months) -> Q.SnowballTerms`.
- `start_env(paths, model, *, quad_grid, multiplier) -> Tuple[PricingEnvironment, str]` (the day-0 environment of path 0 and the front contract).
- `fair_coupon(paths, terms, *, model, quad_grid) -> Q.CouponSolution`.
- `cell_config(product, model, hedge, *, provider: str, cost_bp, workers, batch_paths, quad_grid, disk_dir=None, gate_override=None) -> EnsembleConfig`; `provider` in `("life_surface", "ladder", "exact")`.
- `config_fingerprint(config: EnsembleConfig, paths: MarketPath) -> str`.
- `oracle_tolerances(config) -> Dict[str, float]` (`pv_tolerance`, `delta_tolerance`, `contracts_tolerance` from the gate: bp × 1e-4 × notional, hands × multiplier, hands).
- `run_cell(paths, config, out_dir, *, resume, oracle_paths: Sequence[int]) -> Dict[str, Any]` returning the run's summary (`cell`, `provider`, `seconds`, `gate`, `oracle` list, `skipped`).
- CLI: `--out-dir`, `--cells model:hedge ...`, `--provider {life_surface,ladder}` (the bootstrap and stress runs), `--check-paths`, `--exact-paths`, `--oracle-paths`, `--workers`, `--batch-paths`, `--quad-grid`, `--cost-bp`, `--maturity-months`, `--lockout-months`, `--disk-cache`, `--resume`, `--quick` (two cells, `flat_active:front term_flat_q:front`, `--check-paths 8 --oracle-paths 1`).

- [ ] **Step 1: Write the failing tests**

Append to `test/test_snowball_simulated_paths_study.py`:

```python
S02 = _load("02_ensemble_fleet.py")


@pytest.fixture(scope="module")
def tiny_fleet(tmp_path_factory):
    """A one-month snowball on four bootstrap paths: one cell on the life surface, a ladder check, an oracle."""
    out = tmp_path_factory.mktemp("fleet")
    spot, vol, futures = synthetic_frames()
    history = S01.build_history(spot, vol, futures, rate=RATE)
    bootstrap, stress = S01.build_paths(history, n_paths=4, n_days=40, seed=2, mean_block_days=5,
                                        annual_drift=0.0, vol_floor=0.08, carry_mode="changes")
    S01.write_paths(out, bootstrap, stress, history=history)
    terms = S02.study_terms(bootstrap.dates, maturity_months=1, lockout_months=1)
    coupon = S02.fair_coupon(bootstrap, terms, model="term_flat_q", quad_grid=101)
    product = C.Q.build_product(terms, float(bootstrap.spot[0, 0]), coupon.coupon)
    runs = {}
    for model, hedge in (("flat_active", "front"), ("term_flat_q", "front")):
        cell = C.cell_name(model, hedge)
        cfg = S02.cell_config(product, model, hedge, provider="life_surface", cost_bp=1.0, workers=1,
                              batch_paths=None, quad_grid=101)
        runs[cell] = S02.run_cell(bootstrap, cfg, out / "cells" / cell, resume=False, oracle_paths=[0])
        stress_cfg = S02.cell_config(product, model, hedge, provider="life_surface", cost_bp=1.0, workers=1,
                                     batch_paths=None, quad_grid=101)
        runs[cell + "__stress"] = S02.run_cell(stress, stress_cfg, out / "cells" / f"{cell}__stress",
                                               resume=False, oracle_paths=[])
        ladder_cfg = S02.cell_config(product, model, hedge, provider="ladder", cost_bp=1.0, workers=1,
                                     batch_paths=None, quad_grid=101)
        runs[cell + "__ladder_quad"] = S02.run_cell(bootstrap.take([0, 1]), ladder_cfg,
                                                    out / "cells" / f"{cell}__ladder_quad", resume=False, oracle_paths=[0])
    C.write_json(out / "fleet_manifest.json", {"coupon": coupon.summary(), "terms": terms.summary(), "runs": runs})
    return out, runs, coupon


def test_the_fair_coupon_prices_the_product_to_zero_at_the_start_state(tiny_fleet):
    _, _, coupon = tiny_fleet
    assert coupon.converged and abs(coupon.pv) <= coupon.tolerance and 0.0 < coupon.coupon < 2.0


def test_every_cell_run_is_persisted_gated_and_oracle_checked(tiny_fleet):
    out, runs, _ = tiny_fleet
    from quantark.backtest.simulation.results import EnsembleResults

    for name, run in runs.items():
        assert not run["skipped"] and run["gate"]["passed"], name
        results = EnsembleResults.from_dir(out / "cells" / name)
        assert results.manifest["mode"] == ("ladder" if name.endswith("ladder_quad") else "life_surface")
        assert (out / "cells" / name / "config.json").exists()
        for report in run["oracle"]:
            assert report["passed"] and report["exact_columns_match"], (name, report)
    assert len(runs["flat_active__front"]["oracle"]) == 1 and runs["flat_active__front__stress"]["oracle"] == []


def test_resume_skips_a_run_whose_config_matches(tiny_fleet):
    out, _, coupon = tiny_fleet
    bootstrap, _, _ = S01.load_paths(out)
    terms = S02.study_terms(bootstrap.dates, maturity_months=1, lockout_months=1)
    product = C.Q.build_product(terms, float(bootstrap.spot[0, 0]), coupon.coupon)
    cfg = S02.cell_config(product, "flat_active", "front", provider="life_surface", cost_bp=1.0, workers=1,
                          batch_paths=None, quad_grid=101)
    again = S02.run_cell(bootstrap, cfg, out / "cells" / "flat_active__front", resume=True, oracle_paths=[0])
    assert again["skipped"]
    other = S02.cell_config(product, "flat_active", "front", provider="life_surface", cost_bp=2.0, workers=1,
                            batch_paths=None, quad_grid=101)
    assert S02.config_fingerprint(other, bootstrap) != S02.config_fingerprint(cfg, bootstrap)


def test_the_cell_config_states_every_choice(tiny_fleet):
    out, _, coupon = tiny_fleet
    bootstrap, _, _ = S01.load_paths(out)
    terms = S02.study_terms(bootstrap.dates, maturity_months=1, lockout_months=1)
    product = C.Q.build_product(terms, float(bootstrap.spot[0, 0]), coupon.coupon)
    cfg = S02.cell_config(product, "term_opt_tail", "far", provider="ladder", cost_bp=1.0, workers=2,
                          batch_paths=100, quad_grid=201)
    assert cfg.pricing.mode == "ladder" and cfg.engine_config.futures_curve_extrapolation == "surface_forward_carry"
    assert cfg.products[0].initial_price == 0.0 and cfg.products[0].quantity == C.Q.PRODUCT_QUANTITY
    assert type(cfg.hedge.roll_policy).__name__ == "FarContractRollPolicy"
    assert (cfg.workers, cfg.batch_paths) == (2, 100) and cfg.metadata["model"] == "term_opt_tail"
    exact = S02.cell_config(product, "term_flat_q", "front", provider="exact", cost_bp=0.0, workers=1,
                            batch_paths=None, quad_grid=101)
    assert exact.pricing.mode == "exact" and exact.pricing.gate.sample_states == 0
    tol = S02.oracle_tolerances(cfg)
    assert tol["contracts_tolerance"] == C.GATE_LADDER["delta_tolerance_hands"]
    assert S02.oracle_tolerances(exact) == {"pv_tolerance": 0.0, "delta_tolerance": 0.0, "contracts_tolerance": 0.0}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/test_snowball_simulated_paths_study.py -q`
Expected: `FileNotFoundError` loading `02_ensemble_fleet.py`.

- [ ] **Step 3: Write `02_ensemble_fleet.py`**

```python
"""Stage 02: the cells over the simulated batches.

    .venv/bin/python example/snowball_simulated_paths/02_ensemble_fleet.py --quick
    nohup caffeinate -i -m -s .venv/bin/python example/snowball_simulated_paths/02_ensemble_fleet.py \
        --workers 4 --batch-paths 250 --disk-cache > output/snowball_simulated_paths/fleet.log 2>&1 &

Cells are {flat_active, term_flat_q, term_opt_tail} x {front, far}.  Each
cell runs the bootstrap batch and the stress set on the PDE life surface,
the first ``--check-paths`` bootstrap paths on the QUAD spot ladder (the
engine-substitute check), and optionally ``--exact-paths`` on exact QUAD
repricing -- the only run in which the engine receives the term dividend
OBJECT; the surface and the ladder read a flat q at the bucket centre and
the gate reports what that costs.  Every run is oracle-checked on
``--oracle-paths`` single paths against the replay engine.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402

S01 = __import__("importlib").import_module("01_build_paths") if False else None  # noqa: E402  (see below)

from quantark.backtest.futures_ledger import FuturesRollPolicy  # noqa: E402
from quantark.backtest.replay import HedgeSpec, ReplayProduct  # noqa: E402
from quantark.backtest.replay.engine_factory import create_pricing_engine  # noqa: E402
from quantark.backtest.simulation import (  # noqa: E402
    CacheConfig, EnsembleConfig, GateConfig, MarketPath, PricingProviderConfig, day_chain,
    dividend_yield_for_day, run_ensemble, run_oracle,
)
from quantark.backtest.simulation.hedge import day_active_contract  # noqa: E402
from quantark.backtest.simulation.results import EnsembleResults  # noqa: E402
from quantark.backtest.strategy.futures_delta_strategy import AutocallableDeltaHedgeStrategy  # noqa: E402
from quantark.backtest.transaction_costs import ProportionalCostModel, ZeroCostModel  # noqa: E402
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote  # noqa: E402
from quantark.priceenv import PricingEnvironment  # noqa: E402
from quantark.util.exceptions import ValidationError  # noqa: E402
```

Stage modules start with a digit, so load stage 01 the way the test does, right after `import _common as C`:

```python
import importlib.util  # noqa: E402


def _stage(name: str):
    spec = importlib.util.spec_from_file_location(f"snowball_simulated_paths_{name}", Path(__file__).resolve().parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


S01 = _stage("01_build_paths")
```

(delete the placeholder `S01 = ...` line above.) Then:

```python
PROVIDERS = ("life_surface", "ladder", "exact")


def study_terms(calendar: pd.DatetimeIndex, *, maturity_months: int, lockout_months: int):
    """The q study's term sheet on the simulated calendar, inception on its first day."""
    days = [pd.Timestamp(d).date() for d in calendar]
    return C.Q.build_terms(days[0], C.Q.TradingCalendar(days), maturity_months=maturity_months,
                           lockout_months=lockout_months)


def start_env(paths: MarketPath, model: str, *, quad_grid: int, multiplier: float) -> Tuple[PricingEnvironment, str]:
    """Path 0's day-0 pricing environment under ``model`` (the fair coupon's market)."""
    chain = day_chain(paths, 0, multiplier=multiplier)
    code, _ = day_active_contract(chain, FuturesRollPolicy(roll_days_before_expiry=C.Q.ROLL_DAYS_BEFORE_EXPIRY), None)
    spot, rate = float(paths.spot[0, 0]), float(paths.rate[0, 0])
    dividend = dividend_yield_for_day(
        chain, 0, spot=spot, rate=rate, engine_config=C.engine_config(model, "quad", quad_grid=quad_grid),
        active_contract=code, curve_tenors=paths.tenor_grid, curve_carry=paths.carry[0, 0],
    )
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=spot, asset_name=C.Q.UNDERLYING_NAME),
        vol_surface=FlatVolSurface(volatility=float(paths.atm_vol[0, 0])), rate_curve=FlatRateCurve(rate=rate),
        div_yield=dividend, valuation_date=pd.Timestamp(paths.dates[0]).to_pydatetime(),
    )
    return env, code


def fair_coupon(paths: MarketPath, terms, *, model: str, quad_grid: int):
    """The coupon that prices the product to zero at the start state under ``model`` (QUAD)."""
    env, _ = start_env(paths, model, quad_grid=quad_grid, multiplier=C.Q.FUTURES_MULTIPLIER)
    config = C.engine_config(model, "quad", quad_grid=quad_grid)
    s0 = float(paths.spot[0, 0])

    def pv_at(coupon: float) -> float:
        product = C.Q.build_product(terms, s0, coupon)
        return float(create_pricing_engine(product, config).price(product, env))

    return C.Q.solve_fair_coupon(pv_at, notional=terms.notional)


def _pricing(provider: str, *, disk_dir: Optional[str], gate_override: Optional[Dict[str, Any]]) -> PricingProviderConfig:
    cache = CacheConfig(memory_bytes=C.STATE_CACHE_BYTES, disk_dir=disk_dir)
    if provider == "life_surface":
        return PricingProviderConfig(
            provider="life_surface", cache=cache, gate=GateConfig(**(gate_override or C.GATE_SURFACE)),
            vol_step=C.VOL_STEP, q_step=C.Q_STEP, surface_cache_bytes=C.SURFACE_CACHE_BYTES,
        )
    if provider == "ladder":
        return PricingProviderConfig(
            provider="repricing", cache=cache, gate=GateConfig(**(gate_override or C.GATE_LADDER)),
            spot_step=C.SPOT_STEP, vol_step=C.VOL_STEP, q_step=C.Q_STEP,
        )
    if provider == "exact":
        return PricingProviderConfig(provider="repricing", cache=cache,
                                     gate=GateConfig(sample_states=0, pv_tolerance_bp=0.0, delta_tolerance_hands=0.0))
    raise ValidationError(f"provider must be one of {PROVIDERS}, got {provider!r}")


def cell_config(
    product, model: str, hedge: str, *, provider: str, cost_bp: float, workers: int, batch_paths: Optional[int],
    quad_grid: int, disk_dir: Optional[str] = None, gate_override: Optional[Dict[str, Any]] = None,
) -> EnsembleConfig:
    """One cell: the q study's product, model and hedge policy on the named provider."""
    if hedge not in C.Q.HEDGE_POLICIES:
        raise ValidationError(f"unknown hedge {hedge!r}; one of {tuple(C.Q.HEDGE_POLICIES)}")
    engine = "pde" if provider == "life_surface" else "quad"
    return EnsembleConfig(
        products=[ReplayProduct(product=product, quantity=C.Q.PRODUCT_QUANTITY, position_id=1,
                                has_lifecycle=True, initial_price=0.0)],
        engine_config=C.engine_config(model, engine, quad_grid=quad_grid),
        hedge=HedgeSpec(kind="futures", multiplier=C.Q.FUTURES_MULTIPLIER, roll_policy=C.Q.HEDGE_POLICIES[hedge]()),
        strategy=AutocallableDeltaHedgeStrategy(delta_threshold=0.0, hedge_ratio=1.0, target_delta=0.0),
        transaction_cost_model=ProportionalCostModel(commission_rate=float(cost_bp) * 1e-4) if cost_bp else ZeroCostModel(),
        pricing=_pricing(provider, disk_dir=disk_dir, gate_override=gate_override),
        underlying=C.Q.UNDERLYING_NAME, workers=int(workers), batch_paths=batch_paths,
        metadata={"study": "snowball_simulated_paths", "model": model, "hedge": hedge, "provider": provider,
                  "engine": engine, "cost_bp": float(cost_bp), "quad_grid": int(quad_grid),
                  "coupon": _coupon_of(product)},
    )


def _coupon_of(product) -> float:
    """The KO rate ``create_standard_snowball`` stored: ``BarrierConfig.ko_rate`` (a float, or one per observation)."""
    rate = product.barrier_config.ko_rate
    return float(rate[0] if isinstance(rate, (list, tuple)) else rate)
```

```python
def config_fingerprint(config: EnsembleConfig, paths: MarketPath) -> str:
    """blake2b over the stated settings and the batch; never Python's hash."""
    p = config.pricing
    parts = (
        config.metadata.get("model"), config.metadata.get("hedge"), config.metadata.get("provider"),
        config.metadata.get("cost_bp"), config.metadata.get("quad_grid"), config.metadata.get("coupon"),
        p.mode, p.spot_step, p.vol_step, p.q_step, p.surface_cache_bytes,
        p.gate.sample_states, p.gate.pv_tolerance_bp, p.gate.delta_tolerance_hands,
        repr(config.engine_config), paths.fingerprint(), paths.n_paths,
    )
    return hashlib.blake2b(repr(parts).encode(), digest_size=16).hexdigest()


def oracle_tolerances(config: EnsembleConfig) -> Dict[str, float]:
    """The gate's budget in the oracle's units: currency, delta units, hands."""
    gate = config.pricing.gate
    notional = sum(abs(float(bp.quantity)) * float(bp.product.initial_price) * float(bp.product.contract_multiplier)
                   for bp in config.products)
    if config.pricing.mode == "exact":
        return {"pv_tolerance": 0.0, "delta_tolerance": 0.0, "contracts_tolerance": 0.0}
    return {
        "pv_tolerance": float(gate.pv_tolerance_bp) * 1e-4 * notional,
        "delta_tolerance": float(gate.delta_tolerance_hands) * float(config.hedge.multiplier),
        "contracts_tolerance": float(gate.delta_tolerance_hands),
    }


def run_cell(paths: MarketPath, config: EnsembleConfig, out_dir, *, resume: bool,
             oracle_paths: Sequence[int]) -> Dict[str, Any]:
    """Run one cell over ``paths``, persist it, spot-check single paths against the replay engine."""
    out_dir = Path(out_dir)
    fingerprint = config_fingerprint(config, paths)
    config_path = out_dir / "config.json"
    if resume and config_path.exists() and C.read_json(config_path).get("fingerprint") == fingerprint:
        previous = C.read_json(out_dir / "run.json")
        return {**previous, "skipped": True}
    started = time.perf_counter()
    results = run_ensemble(config, paths)
    results.to_dir(out_dir)
    C.write_json(config_path, {
        "fingerprint": fingerprint, "metadata": config.metadata, "mode": config.pricing.mode,
        "gate": {"sample_states": config.pricing.gate.sample_states, "pv_tolerance_bp": config.pricing.gate.pv_tolerance_bp,
                 "delta_tolerance_hands": config.pricing.gate.delta_tolerance_hands},
        "steps": {"spot_step": config.pricing.spot_step, "vol_step": config.pricing.vol_step, "q_step": config.pricing.q_step},
        "workers": config.workers, "batch_paths": config.batch_paths, "n_paths": paths.n_paths,
        "path_fingerprint": paths.fingerprint(),
    })
    single = replace(config, workers=1, batch_paths=None)
    tolerances = oracle_tolerances(config)
    reports = []
    for i in oracle_paths:
        report = run_oracle(single, paths.take([int(i)]), 0, **tolerances)
        reports.append({"path": int(i), **report.as_dict()})
    run = {
        "cell": config.metadata.get("model") + "__" + config.metadata.get("hedge"), "provider": config.metadata.get("provider"),
        "n_paths": paths.n_paths, "seconds": time.perf_counter() - started, "gate": results.manifest["gate"],
        "engine_calls": results.manifest["engine_calls"], "solves": results.manifest.get("solves", 0),
        "oracle": reports, "oracle_tolerances": tolerances, "skipped": False,
    }
    C.write_json(out_dir / "run.json", run)
    return run
```

(`replace` on `EnsembleConfig` re-runs `__post_init__`, which is fine: the same validated settings.)

The CLI:

```python
def parse_cells(values: Optional[Sequence[str]]) -> List[Tuple[str, str]]:
    if not values:
        return [(m, h) for m in C.MODELS for h in C.HEDGES]
    out = []
    for value in values:
        model, _, hedge = value.partition(":")
        if model not in C.MODELS or hedge not in C.HEDGES:
            raise ValidationError(f"cell {value!r} must be model:hedge with model in {C.MODELS} and hedge in {C.HEDGES}")
        out.append((model, hedge))
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=C.DEFAULT_OUT_DIR)
    parser.add_argument("--cells", nargs="+", default=None, help="model:hedge, default = the full grid")
    parser.add_argument("--provider", choices=("life_surface", "ladder"), default="life_surface")
    parser.add_argument("--check-paths", type=int, default=C.CHECK_PATHS, help="QUAD ladder subset (0 = none)")
    parser.add_argument("--exact-paths", type=int, default=0, help="exact QUAD repricing subset (0 = none)")
    parser.add_argument("--oracle-paths", type=int, default=C.ORACLE_PATHS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-paths", type=int, default=None)
    parser.add_argument("--quad-grid", type=int, default=C.Q.DEFAULT_QUAD_GRID)
    parser.add_argument("--cost-bp", type=float, default=C.COST_BP)
    parser.add_argument("--maturity-months", type=int, default=C.Q.MATURITY_MONTHS)
    parser.add_argument("--lockout-months", type=int, default=C.Q.LOCKOUT_MONTHS)
    parser.add_argument("--disk-cache", action="store_true", help="share states through <out>/cache")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true", help="two cells, 8 check paths, 1 oracle path")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.cells = args.cells or ["flat_active:front", "term_flat_q:front"]
        args.check_paths, args.oracle_paths = min(args.check_paths, 8), min(args.oracle_paths, 1)
    cells = parse_cells(args.cells)
    bootstrap, stress, paths_manifest = S01.load_paths(args.out_dir)
    terms = study_terms(bootstrap.dates, maturity_months=args.maturity_months, lockout_months=args.lockout_months)
    coupon = fair_coupon(bootstrap, terms, model=C.Q.REFERENCE_MODEL, quad_grid=args.quad_grid)
    product = C.Q.build_product(terms, float(bootstrap.spot[0, 0]), coupon.coupon)
    C.write_json(args.out_dir / "coupon.json", {"coupon": coupon.summary(), "terms": terms.summary(),
                                                "reference_model": C.Q.REFERENCE_MODEL})
    print(f"fair coupon {coupon.coupon:.4%} under {C.Q.REFERENCE_MODEL} (|PV| {abs(coupon.pv):,.0f}); "
          f"{len(cells)} cells x {bootstrap.n_paths} paths, provider {args.provider}, workers {args.workers}")
    disk_dir = str(args.out_dir / "cache") if args.disk_cache else None
    runs: Dict[str, Any] = {}
    oracle = list(range(args.oracle_paths))
    common = dict(cost_bp=args.cost_bp, workers=args.workers, batch_paths=args.batch_paths, quad_grid=args.quad_grid,
                  disk_dir=disk_dir)
    for model, hedge in cells:
        cell = C.cell_name(model, hedge)
        plan = [(cell, bootstrap, args.provider, oracle),
                (f"{cell}__stress", stress, args.provider, [])]
        if args.check_paths:
            plan.append((f"{cell}__ladder_quad", bootstrap.take(range(min(args.check_paths, bootstrap.n_paths))), "ladder", oracle))
        if args.exact_paths:
            plan.append((f"{cell}__exact_quad", bootstrap.take(range(min(args.exact_paths, bootstrap.n_paths))), "exact", oracle))
        for name, batch, provider, checks in plan:
            config = cell_config(product, model, hedge, provider=provider, **common)
            run = run_cell(batch, config, args.out_dir / "cells" / name, resume=args.resume, oracle_paths=checks)
            runs[name] = run
            state = "skipped" if run["skipped"] else f"{run['seconds']:.0f}s"
            gate = run["gate"]
            print(f"  {name:32s} {state:>8s}  gate {gate['max_pv_gap_bp']:.2f} bp / {gate['max_delta_gap_hands']:.2f} hands "
                  f"({'ok' if gate['passed'] else 'FAIL'})  oracle {'ok' if all(r['passed'] for r in run['oracle']) else 'FAIL'}")
    C.write_json(args.out_dir / "fleet_manifest.json", {
        "coupon": coupon.summary(), "terms": terms.summary(), "paths": paths_manifest, "runs": runs,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    })
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -n0 test/test_snowball_simulated_paths_study.py -q`
Expected: all pass. The fixture is the first run of the q study's date-clock product through the ensemble engine and its oracle; if the oracle reports a gap in an EXACT column (a lifecycle flag, the contract), stop and investigate — that is a lifecycle bug, not a tolerance. If the life-surface gate fails on the one-month product, print the report and widen ONLY the test's `gate_override` with the measured number and the reason in the commit; the study's gates stay as `_common.py` states them.

- [ ] **Step 5: Commit**

```bash
git add example/snowball_simulated_paths/02_ensemble_fleet.py test/test_snowball_simulated_paths_study.py
git commit -m "feat(example): simulated-path snowball study, stage 02 — cells on the life surface, QUAD ladder check, oracle spot check

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: `03_report.py`

**Files:**
- Create: `example/snowball_simulated_paths/03_report.py`
- Modify: `test/test_snowball_simulated_paths_study.py`

**Design.** `aggregate(out_dir, *, es_level, historical_dir)` loads every cell directory and builds: per-cell distributions of the headline measures (`terminal_pnl_bp`, `daily_pnl_std_bp`, `variance_reduction_r2`, `max_drawdown_bp`, `turnover`, `cost_bp`, `roll_day_mtm_jump_bp`, `other_day_mtm_jump_bp`, `delta_churn`) with `ko/ki/maturity` frequencies; paired comparisons (every bootstrap cell against `BASELINE_CELL`; `far` against `front` per model) described per measure; the stress table (scenario × cell: terminal P&L, daily std, termination); the engine check (`<cell>.take(range(n))` paired against `<cell>__ladder_quad`, and against `__exact_quad` when present); the gates and oracle reports from `run.json`; and the historical location: for every run of the q study found under `historical_dir/runs/*/<cell>/` (via `Q.load_run`), its measures through the library's `path_measures` and the percentile rank of its `terminal_pnl_bp` and `daily_pnl_std_bp` within the simulated cell's summary. The historical section is skipped with a stated reason when the directory is absent. `write_tables` writes `fleet_cells.json`, `fleet_paired.csv`, `stress_table.csv`, `engine_check.csv`, `historical_location.csv`, `fleet_summary.json` to `DATA_DIR`; `build_report` writes a self-contained HTML (tables, matplotlib histograms of `terminal_pnl_bp` per cell when matplotlib imports, the study's caveats).

**Interfaces:**
- `HEADLINE_MEASURES` tuple as above; `load_cells(out_dir) -> Dict[str, EnsembleResults]`; `aggregate(...) -> Dict[str, Any]`; `write_tables(agg, data_dir) -> None`; `build_report(agg) -> str`; CLI `--out-dir`, `--data-dir`, `--es-level` (default `0.05`), `--historical-dir`, `--no-historical`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_snowball_simulated_paths_study.py`:

```python
S03 = _load("03_report.py")


def test_aggregate_reduces_every_cell_and_pairs_them(tiny_fleet):
    out, runs, _ = tiny_fleet
    agg = S03.aggregate(out, es_level=0.25, historical_dir=None)
    assert set(agg["cells"]) == {"flat_active__front", "term_flat_q__front"}
    cell = agg["cells"]["term_flat_q__front"]
    assert set(S03.HEADLINE_MEASURES) <= set(cell["distributions"])
    d = cell["distributions"]["terminal_pnl_bp"]
    assert d["n_paths"] == 4 and "q50" in d["quantiles"] and d["es_level"] == 0.25
    assert [p["variant"] for p in agg["paired"]] == ["term_flat_q__front"] and agg["paired"][0]["base"] == "flat_active__front"
    assert agg["paired"][0]["measures"]["terminal_pnl_bp"]["n"] == 4
    assert {row["cell"] for row in agg["stress"]} == {"flat_active__front", "term_flat_q__front"}
    assert len(agg["stress"]) == 10 and all(row["scenario"] for row in agg["stress"])
    check = {row["cell"]: row for row in agg["engine_check"]}
    assert check["term_flat_q__front"]["n"] == 2 and "terminal_pnl_bp" in check["term_flat_q__front"]["measures"]
    assert agg["gates"]["flat_active__front"]["passed"] and agg["historical"]["available"] is False


def test_tables_and_report_are_written(tiny_fleet, tmp_path):
    out, _, _ = tiny_fleet
    agg = S03.aggregate(out, es_level=0.25, historical_dir=None)
    S03.write_tables(agg, tmp_path)
    for name in ("fleet_cells.json", "fleet_paired.csv", "stress_table.csv", "engine_check.csv", "fleet_summary.json"):
        assert (tmp_path / name).exists(), name
    html = S03.build_report(agg)
    assert "<html" in html and "term_flat_q__front" in html and "flat_active__front" in html
    assert "expected shortfall" in html.lower() and "historical" in html.lower()
    (tmp_path / "report.html").write_text(html)


def test_the_historical_location_uses_the_library_measures(tiny_fleet, tmp_path):
    """A fake q-study run directory: one inception, one cell, frames in the study's layout."""
    out, _, _ = tiny_fleet
    from quantark.backtest.simulation.results import EnsembleResults

    results = EnsembleResults.from_dir(out / "cells" / "term_flat_q__front")
    run_dir = tmp_path / "runs" / "2024-01" / "term_flat_q__front"
    run_dir.mkdir(parents=True)
    states = results.path_states(0).set_index("date")
    states.to_csv(run_dir / "states.csv")
    trades = results.path_trades(0)
    (trades.set_index("date") if len(trades) else trades).to_csv(run_dir / "trades.csv")
    for name in ("greeks", "rebalances", "actions"):
        pd.DataFrame(index=pd.DatetimeIndex([], name="date")).to_csv(run_dir / f"{name}.csv")
    C.write_json(run_dir / "run_summary.json", {"inception": "2024-01-02", "model": "term_flat_q", "hedge": "front",
                                                "notional": results.notional})
    agg = S03.aggregate(out, es_level=0.25, historical_dir=tmp_path)
    hist = agg["historical"]
    assert hist["available"] and len(hist["rows"]) == 1
    row = hist["rows"][0]
    assert row["cell"] == "term_flat_q__front" and row["inception"] == "2024-01-02"
    assert row["terminal_pnl_bp"] == pytest.approx(results.summary["terminal_pnl_bp"].iloc[0])
    assert 0.0 <= row["terminal_pnl_percentile"] <= 100.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/test_snowball_simulated_paths_study.py -q`
Expected: `FileNotFoundError` loading `03_report.py`.

- [ ] **Step 3: Write `03_report.py`**

```python
"""Stage 03: distributions, paired differences, stress, the engine check, and the report.

    .venv/bin/python example/snowball_simulated_paths/03_report.py

Reads every cell under ``<out>/cells``, writes the tables to the study's
``data/`` directory and the self-contained ``simulated_paths_report.html``.
The historical study's realised runs (``output/snowball_q_term_structure``)
are located inside the simulated distribution when that directory exists.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402

from quantark.backtest.simulation.measures import path_measures  # noqa: E402
from quantark.backtest.simulation.results import EnsembleResults  # noqa: E402
from quantark.util.exceptions import ValidationError  # noqa: E402

HEADLINE_MEASURES = (
    "terminal_pnl_bp", "daily_pnl_std_bp", "variance_reduction_r2", "max_drawdown_bp", "turnover", "cost_bp",
    "roll_day_mtm_jump_bp", "other_day_mtm_jump_bp", "delta_churn",
)
SUFFIXES = ("__stress", "__ladder_quad", "__exact_quad")


def _is_bootstrap_cell(name: str) -> bool:
    return not any(name.endswith(s) for s in SUFFIXES)


def load_cells(out_dir) -> Dict[str, EnsembleResults]:
    cells_dir = Path(out_dir) / "cells"
    if not cells_dir.exists():
        raise ValidationError(f"no cells under {cells_dir}; run 02_ensemble_fleet.py first")
    out = {}
    for path in sorted(p for p in cells_dir.iterdir() if (p / "manifest.json").exists()):
        out[path.name] = EnsembleResults.from_dir(path)
    if not out:
        raise ValidationError(f"no persisted cell under {cells_dir}")
    return out


def _distributions(results: EnsembleResults, es_level: float) -> Dict[str, Any]:
    out = {}
    for measure in HEADLINE_MEASURES:
        tail = "upper" if measure in ("daily_pnl_std_bp", "max_drawdown_bp", "turnover", "cost_bp",
                                      "roll_day_mtm_jump_bp", "other_day_mtm_jump_bp", "delta_churn") else "lower"
        out[measure] = results.distribution(measure, es_level=es_level, tail=tail)
    return out


def _paired(variant: EnsembleResults, base: EnsembleResults) -> Dict[str, Any]:
    comparison = variant.paired(base)
    return {measure: comparison.describe(measure) for measure in HEADLINE_MEASURES}


def _stress_rows(name: str, results: EnsembleResults) -> List[Dict[str, Any]]:
    names = results.manifest.get("path_meta", {}).get("scenario_names") or [f"scenario_{i}" for i in range(results.n_paths)]
    rows = []
    for i, row in results.summary.iterrows():
        rows.append({"cell": name, "scenario": names[int(i)], "terminal_pnl_bp": row["terminal_pnl_bp"],
                     "daily_pnl_std_bp": row["daily_pnl_std_bp"], "max_drawdown_bp": row["max_drawdown_bp"],
                     "termination": row["termination_reason"], "knocked_in": bool(row["knocked_in"]), "days": int(row["days"])})
    return rows


def _historical(historical_dir, cells: Dict[str, EnsembleResults]) -> Dict[str, Any]:
    if historical_dir is None:
        return {"available": False, "reason": "no historical directory given", "rows": []}
    runs_dir = Path(historical_dir) / "runs"
    if not runs_dir.exists():
        return {"available": False, "reason": f"{runs_dir} does not exist (run the q term-structure study first)", "rows": []}
    rows = []
    for summary_path in sorted(runs_dir.glob("*/*/run_summary.json")):
        summary = C.read_json(summary_path)
        cell = C.cell_name(summary["model"], summary["hedge"])
        if cell not in cells:
            continue
        run = C.Q.load_run(summary_path.parent)
        measures = path_measures(run["states"], run["trades"], notional=float(summary["notional"]))
        simulated = cells[cell].summary
        row = {"cell": cell, "inception": summary["inception"], **{m: measures[m] for m in HEADLINE_MEASURES}}
        for measure in ("terminal_pnl_bp", "daily_pnl_std_bp"):
            values = simulated[measure].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"{measure.replace('_bp', '')}_percentile"] = (
                float((values <= measures[measure]).mean() * 100.0) if values.size else None
            )
        rows.append(row)
    if not rows:
        return {"available": False, "reason": f"no run under {runs_dir} matches a simulated cell", "rows": []}
    return {"available": True, "reason": None, "rows": rows}


def aggregate(out_dir, *, es_level: float, historical_dir) -> Dict[str, Any]:
    """Everything the report shows, as plain dicts and lists."""
    out_dir = Path(out_dir)
    cells = load_cells(out_dir)
    bootstrap = {name: r for name, r in cells.items() if _is_bootstrap_cell(name)}
    agg: Dict[str, Any] = {
        "out_dir": str(out_dir), "es_level": float(es_level),
        "fleet": C.read_json(out_dir / "fleet_manifest.json") if (out_dir / "fleet_manifest.json").exists() else {},
        "coupon": C.read_json(out_dir / "coupon.json") if (out_dir / "coupon.json").exists() else {},
        "cells": {name: {"n_paths": r.n_paths, "distributions": _distributions(r, es_level), "manifest_mode": r.manifest.get("mode"),
                         "gate": r.manifest.get("gate"), "solves": r.manifest.get("solves"), "engine_calls": r.manifest.get("engine_calls"),
                         "seconds": r.manifest.get("seconds")}
                  for name, r in bootstrap.items()},
        "paired": [], "stress": [], "engine_check": [], "gates": {}, "historical": None,
    }
    base = bootstrap.get(C.BASELINE_CELL)
    for name, r in bootstrap.items():
        if base is not None and name != C.BASELINE_CELL:
            agg["paired"].append({"variant": name, "base": C.BASELINE_CELL, "measures": _paired(r, base)})
    for model in C.MODELS:
        front, far = bootstrap.get(C.cell_name(model, "front")), bootstrap.get(C.cell_name(model, "far"))
        if front is not None and far is not None:
            agg["paired"].append({"variant": C.cell_name(model, "far"), "base": C.cell_name(model, "front"), "measures": _paired(far, front)})
    for name, r in cells.items():
        if name.endswith("__stress"):
            agg["stress"] += _stress_rows(name[: -len("__stress")], r)
    for name, r in cells.items():
        for suffix in ("__ladder_quad", "__exact_quad"):
            if name.endswith(suffix) and name[: -len(suffix)] in bootstrap:
                surface = bootstrap[name[: -len(suffix)]].take(range(r.n_paths))
                agg["engine_check"].append({"cell": name[: -len(suffix)], "check": suffix[2:], "n": r.n_paths,
                                            "measures": _paired(surface, r)})
    for name in cells:
        run_path = out_dir / "cells" / name / "run.json"
        if run_path.exists():
            run = C.read_json(run_path)
            agg["gates"][name] = {**run["gate"], "oracle": run.get("oracle", []), "seconds": run.get("seconds")}
    agg["historical"] = _historical(historical_dir, bootstrap)
    return agg
```

Tables and the report:

```python
def write_tables(agg: Dict[str, Any], data_dir) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    C.write_json(data_dir / "fleet_cells.json", {"cells": agg["cells"], "paired": agg["paired"]})
    paired_rows = [{"variant": p["variant"], "base": p["base"], "measure": m, **d}
                   for p in agg["paired"] for m, d in p["measures"].items()]
    pd.DataFrame(paired_rows).to_csv(data_dir / "fleet_paired.csv", index=False)
    pd.DataFrame(agg["stress"]).to_csv(data_dir / "stress_table.csv", index=False)
    check_rows = [{"cell": e["cell"], "check": e["check"], "n": e["n"], "measure": m, **d}
                  for e in agg["engine_check"] for m, d in e["measures"].items()]
    pd.DataFrame(check_rows).to_csv(data_dir / "engine_check.csv", index=False)
    if agg["historical"]["available"]:
        pd.DataFrame(agg["historical"]["rows"]).to_csv(data_dir / "historical_location.csv", index=False)
    C.write_json(data_dir / "fleet_summary.json", {k: v for k, v in agg.items() if k not in ("stress",)})


def _fmt(value: Any, digits: int = 1, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{value * 100:.{digits}f}%" if pct else f"{value:,.{digits}f}"
    return html.escape(str(value))


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], caption: str = "") -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    cap = f"<caption>{html.escape(caption)}</caption>" if caption else ""
    return f"<table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _histograms(agg: Dict[str, Any], cells: Dict[str, EnsembleResults]) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # matplotlib is optional here, as in the q study
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, r in cells.items():
        values = r.summary["terminal_pnl_bp"].to_numpy(dtype=float)
        ax.hist(values[np.isfinite(values)], bins=40, histtype="step", label=name)
    ax.set_xlabel("terminal hedged P&L (bp of notional)")
    ax.legend(fontsize=7)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
```

`build_report(agg, cells=None)` assembles the sections in this order, each a `<h2>` with one table: setup (paths manifest, coupon, terms), hedge-cost distributions per cell (mean, std, q05, q50, q95, expected shortfall, share positive, KO/KI frequency), paired differences (mean, median, share positive, t-stat per measure and pair), stress table, engine check (surface − ladder per measure: mean and worst absolute difference), gates and oracle spot checks, historical location (or the stated reason), and caveats. The caveats section states, in prose: one start state (the history's last day) for every simulated path, so a historical inception's percentile is indicative; under the surface and the ladder the term models enter through `q_T` only (the `__exact_quad` check is where the engine receives the term object); paired t-statistics on 2,000 simulated paths are independent draws, unlike the historical study's overlapping inceptions. The histogram, when `cells` is given and matplotlib imports, goes under the distributions table.

```python
def build_report(agg: Dict[str, Any], cells: Optional[Dict[str, EnsembleResults]] = None) -> str:
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Snowball hedging on simulated paths</title>",
             "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em}"
             "table{border-collapse:collapse;margin:1em 0;font-size:13px}th,td{border:1px solid #ccc;padding:3px 8px;text-align:right}"
             "th:first-child,td:first-child{text-align:left}caption{text-align:left;font-weight:600}</style></head><body>",
             "<h1>Snowball hedging on simulated paths</h1>"]
    # ... one section per key of agg, using _table and _fmt; see the section list above ...
    parts.append("</body></html>")
    return "\n".join(parts)
```

Write each section out in full (the test checks the cell names, the phrase "expected shortfall" and the word "historical" appear); do not leave the section list as a comment.

```python
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=C.DEFAULT_OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=C.DATA_DIR)
    parser.add_argument("--es-level", type=float, default=0.05)
    parser.add_argument("--historical-dir", type=Path, default=C.DEFAULT_HISTORICAL_DIR)
    parser.add_argument("--no-historical", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cells = load_cells(args.out_dir)
    agg = aggregate(args.out_dir, es_level=args.es_level, historical_dir=None if args.no_historical else args.historical_dir)
    write_tables(agg, args.data_dir)
    report = build_report(agg, {n: r for n, r in cells.items() if _is_bootstrap_cell(n)})
    (Path(args.data_dir) / "simulated_paths_report.html").write_text(report)
    print(f"{len(agg['cells'])} cells, {len(agg['paired'])} paired comparisons, {len(agg['stress'])} stress rows, "
          f"historical {'located' if agg['historical']['available'] else 'not available: ' + str(agg['historical']['reason'])}; "
          f"report at {Path(args.data_dir) / 'simulated_paths_report.html'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -n0 test/test_snowball_simulated_paths_study.py -q`
Expected: all pass. Open the `report.html` the second test wrote (its path is under the pytest tmp dir) once, by eye, to check the tables render.

- [ ] **Step 5: Commit**

```bash
git add example/snowball_simulated_paths/03_report.py test/test_snowball_simulated_paths_study.py
git commit -m "feat(example): simulated-path snowball study, stage 03 — distributions, paired, stress, engine check, report

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Exports, guides, regression, and the real quick run

**Files:**
- Modify: `quantark/backtest/simulation/__init__.py`, `quantark/backtest/simulation/README.md`
- Create: `example/snowball_simulated_paths/README.md`
- Modify: `test/simulation/test_market_path.py`

- [ ] **Step 1: Extend the export test and see it fail**

Add to the name list in `test_public_api_is_exported`: `"PairedComparison", "path_measures", "max_drawdown", "MEASURE_COLUMNS", "SUMMARY_COLUMNS", "jsonable"`.

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_market_path.py -k public_api -q`
Expected: `AssertionError: PairedComparison`.

- [ ] **Step 2: Export and document the library**

`__init__.py`: import `PairedComparison, SUMMARY_COLUMNS, jsonable` from `.results`, `MEASURE_COLUMNS, max_drawdown, path_measures` from `.measures`; keep `EnsembleResults, StateCube` coming from `.engine` (unchanged names); add them to `__all__`; extend the docstring with a plan-4 sentence.

`README.md`: add a section **"Results"** before "Why you can trust it": `summary` (one row per path, the q study's measures moved into `measures.py` so the two studies report one implementation, plus termination, KI flag, KO index, terminal P&L in currency; `book_notional` in the manifest is the bp base), `distribution` (quantiles, expected shortfall as the mean of the stated tail, event frequencies), `paired` (`variant.paired(base)`; matched paths proved by identical market columns; `describe` with the paired t-statistic), `take`, `to_dir`/`from_dir` (the format, the summary recomputed on load). One paragraph under "The life surface" on compaction (only the columns a run reads are kept; readouts bit-identical; ~4× less memory). Replace "What comes next" with a pointer to `example/snowball_simulated_paths/README.md`.

- [ ] **Step 3: Write the study README**

`example/snowball_simulated_paths/README.md` with these sections, each written out in full:

- **What it asks.** The historical study answered on 29 realised inceptions; this one runs the same product, models and hedges over 2,000 bootstrapped futures and five designed stresses from one start state, and reports distributions, paired differences on matched paths, and where the realised runs sit inside the simulated distribution.
- **Data.** The same local, untracked caches as the q study; the history's fingerprint is recorded in `paths/manifest.json`.
- **Paths.** Stationary block bootstrap (`MEAN_BLOCK_DAYS`, demeaned, drift 0, vol floor 0.08, carry changes), 275 weekdays from the day after the history's last day; the five stress scenarios and their parameters.
- **Product and cells.** The q study's 1Y snowball at the start spot, fair coupon solved once under `term_flat_q` with QUAD 401 at the start state (`coupon.json`), every cell starting from the traded price; `MODELS × HEDGES`; 1 bp per side.
- **What a cell measures under each provider.** The paragraph from "Scope of this plan" fact 1: on the life surface and the ladder the term models enter through `q_T` (flat at the bucket centre); the gate reports the gap to exact term repricing; `--exact-paths` is the run where the engine receives the term object.
- **Gates and checks.** The surface and ladder gates, the oracle spot check on single paths, the engine check (surface vs QUAD ladder on the first 200 paths).
- **Running it.** The three commands (`01`, `02 --quick`, the full `02` under `nohup caffeinate -i -m -s ... --workers 4 --batch-paths 250 --disk-cache --resume`, `03`), where outputs go (`output/snowball_simulated_paths/`, tables and the HTML under `data/`), and the measured `--quick` timings from Step 5.
- **Results.** After Step 5: the quick run's cell table (mean / q05 / q50 / q95 / ES of terminal P&L, daily std, R², cost) and its gate and oracle lines, stated as "40 paths, two cells" — and a sentence that the 2,000-path fleet has not been run yet and its numbers will replace this section.
- **Caveats.** One start state; `q_T`-only under the approximate providers; independent draws vs overlapping inceptions; the stress paths are designed, not sampled.

- [ ] **Step 4: Full regression**

Run: `.venv/bin/python -m pytest -n0 test/simulation test/test_snowball_simulated_paths_study.py test/test_snowball_q_term_structure_study.py test/test_snowball_life_surface.py test/replay_golden -q`
Then the whole suite: `.venv/bin/python -m pytest -q`
Expected: green apart from the two pre-existing `test/mo_volmodels/test_dashboard.py` failures. Before staging, `git checkout -- example/mo_volmodels/data/`.

- [ ] **Step 5: The real quick run (verification, not a test)**

If `example/mo_volmodels/data/history/csi1000_spot.csv` exists:

```bash
mkdir -p output/snowball_simulated_paths
.venv/bin/python example/snowball_simulated_paths/01_build_paths.py --quick
nohup caffeinate -i -m -s .venv/bin/python example/snowball_simulated_paths/02_ensemble_fleet.py --quick \
    > output/snowball_simulated_paths/quick.log 2>&1 &
```

Wait for it (`tail -f output/snowball_simulated_paths/quick.log`; a 1Y product on the life surface over 40 paths is a few hundred PDE solves per cell — expect tens of minutes; kill the process group, not `pkill -f`, if it must be stopped), then `.venv/bin/python example/snowball_simulated_paths/03_report.py`. Put the printed timings, the gate lines and the quick cell table into the README's "Running it" and "Results" sections. Stage `example/snowball_simulated_paths/data/` (the tables and the HTML the quick run wrote) with the README. If the history is absent, say so in the README's "Results" section instead and stage no `data/`.

- [ ] **Step 6: Commit**

```bash
git add quantark/backtest/simulation/__init__.py quantark/backtest/simulation/README.md \
        example/snowball_simulated_paths/ test/simulation/test_market_path.py
git commit -m "feat(backtest/simulation),example: results API exports, module guide, simulated-path study README and quick-run results

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review against the spec

- **§9 results** → Tasks 1–5: `summary` with every listed column (`terminal_pnl`, `terminal_pnl_bp`, `daily_pnl_std_bp`, `variance_reduction_r2`, `max_drawdown_bp`, `turnover`, `cost_bp`, `roll_day_mtm_jump_bp`, `other_day_mtm_jump_bp`, `delta_churn`, `termination_reason`, `days`, `knocked_in`, `ko_observation_index`), the measures moved into the library with the study delegating; `distribution` with mean, std, the seven quantiles, expected shortfall at a required `es_level`, share positive, KO and KI frequencies; `paired` on matched paths with mean, share positive and t-stat; `path_states/trades/events` unchanged; the manifest gains `library_version` (the generator meta, seed, fingerprints, cache statistics, gate reports and timing were already there); `to_dir`/`from_dir` persist the cube as npz, the summary and the manifest.
- **§13 results tests** → measures equal the study's on a known path (Task 1), distribution quantiles (Task 3), paired differences on matched paths (Task 4).
- **§14 example study** → Task 7 (`01_build_paths.py`: 2,000 × 275 bootstrap, block 20, demeaned, drift 0, plus the stress library, persisted with fingerprints), Task 8 (`02_ensemble_fleet.py`: the three models × two hedges, surface provider on PDE, QUAD repricing on a 200-path subset, gates recorded, three-path oracle spot check per cell), Task 9 (`03_report.py`: distributions with ES, paired differences, the stress table, the historical study's runs located in the simulated distribution, the HTML report). `N_DAYS = 275` rather than the spec's 260 because the engine refuses a calendar that ends before a 12-month product settles; recorded in the README.
- **§15 performance** → Task 6 makes the surface budget hold four times more buckets; the measured quick-run timings go into the README as the spec asks ("estimates to be measured in the example").
- **§16 items 5–6** → this plan.
- **Placeholder scan:** every step has its code or its exact instruction; the two "see the section list above" notes in Task 9 name the sections and the test pins their content; the README's "Results" section is filled by Task 10 Step 5 with measured numbers or an explicit statement that the data is absent.
- **Type consistency:** `path_measures(states, trades, *, notional)` is the one signature used by the summary, the study's delegate and the historical location; `distribution(measure, *, es_level, tail)`, `paired(base) -> PairedComparison`, `describe(measure)`, `take(indices)`, `to_dir(path)`, `from_dir(path)` are used with those names in Tasks 8–9; `cell_config(product, model, hedge, *, provider, cost_bp, workers, batch_paths, quad_grid, disk_dir, gate_override)` and `run_cell(paths, config, out_dir, *, resume, oracle_paths)` match between Task 8's tests and its CLI; `EnsembleResults` and `StateCube` remain importable from `.engine`.
