# PnL Explain (equity) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `quantark/pnlexplain/` — a two-snapshot PnL explain for equity option products with a full-revaluation waterfall and a greeks-based Taylor explain on one factor model, lifecycle cash flows as an explicit event term, position/portfolio aggregation with trade and cost rows, and a daily explain series from both backtest engines.

**Architecture:** An immutable `ValuationSnapshot` and one value identity (contingent MTM + pending receivable PV + paid cash); a `FactorDiff` that reads scalar moves at the product coordinate and detects which market objects changed; a `ScenarioCache` that builds the state for any *set* of applied factors and memoises its value, from which the sequential/Shapley waterfall, the canonical `time_pure`, and the endpoints are all read; a Taylor explainer that requests greeks from the existing `GreeksCalculator` on an explicit route and normalises units; a `LifecycleTransition` carrying the alive-at-t1 contract; position/portfolio layers; recorders hooked into both backtest engines behind `pnl_explain=None` defaults.

**Tech Stack:** Python 3.10+, dataclasses, numpy, pandas; existing `quantark.asset.equity.riskmeasures`, `quantark.asset.equity.lifecycle`, `quantark.backtest`.

**Spec:** `docs/superpowers/specs/2026-09-02-pnl-explain-design.md` (commit 12177bb on `worktree-pnl-explain`). The plan argues from the spec; read both.

## Global Constraints

- Work in the worktree `/Users/fuxinyao/quant-ark/.claude/worktrees/pnl-explain` (branch `worktree-pnl-explain`, base main @ 7528618). Never `cd` out of it.
- Run every test with the worktree source shadowing the editable install:
  `PYTHONPATH=/Users/fuxinyao/quant-ark/.claude/worktrees/pnl-explain /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 <files>` (abbreviated below as `PYTEST <files>`).
- Canonical imports only (`quantark.*`); no flat legacy imports.
- Numerics: `quantark.util.numerical` (`is_close`, `is_zero`, `safe_divide`), never raw float comparisons with hard-coded tolerances in library code; tests may use `pytest.approx` with explicit `abs`/`rel`.
- Exceptions: `ValidationError` for bad inputs, `NumericalError` for non-finite results (`quantark.util.exceptions`).
- No numeric change to `riskmeasures`: the only addition there is the pure `GreeksCalculator.resolve_route`. `PortfolioLifecycleManager` gains only the pure `pricing_products` accessor.
- Backtests with `pnl_explain=None` must be byte-identical in behaviour and output (gate in Task 11/12).
- No fallbacks that invent semantics: where the alive product, trade price, valuation point or transition is unknown, raise `ValidationError`.
- Result dict/row/frame order is deterministic and never derived from iterating a `set`.
- Every commit message ends with:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_019fnLipZtAJAPAQJLLEXQMb
  ```
- Commit only files you created/changed for the task (`git add <paths>`); never `git add example/` wholesale (mo sample data churn).

## File Structure

| Path | Responsibility |
|---|---|
| `quantark/pnlexplain/__init__.py` | public re-exports (spec §5.6) |
| `quantark/pnlexplain/base.py` | `Factor`, `MARKET_FACTORS`, `ExplainMethod`, `RowKind`, `ExplainRow`, `ValueBreakdown`, `PnLExplainResult`, frame helpers |
| `quantark/pnlexplain/config.py` | `PnLExplainConfig` + validation, stencil tables, term→factor map |
| `quantark/pnlexplain/equity/__init__.py` | equity subpackage exports |
| `quantark/pnlexplain/equity/snapshot.py` | `ValuationSnapshot`, `is_terminal`, `value()` |
| `quantark/pnlexplain/equity/fingerprints.py` | `calendars_equal`, `contract_fingerprint`, `check_contract_roll`, `lifecycle_fingerprint` |
| `quantark/pnlexplain/equity/coordinate.py` | `FactorCoordinate`, `resolve_coordinate` |
| `quantark/pnlexplain/equity/factor_diff.py` | `FactorMoves`, `build_factor_moves`, `validate_pair` |
| `quantark/pnlexplain/equity/lifecycle.py` | `LifecycleTransition`, `resolve_transition`, `event_row` |
| `quantark/pnlexplain/equity/scenario.py` | `ScenarioCache` (state-set construction + value memo, endpoints, `time_pure`) |
| `quantark/pnlexplain/equity/waterfall.py` | `sequential_rows`, `shapley_rows` |
| `quantark/pnlexplain/equity/taylor.py` | greek route/units/cash table/terms/time term, `taylor_rows` |
| `quantark/pnlexplain/equity/explain.py` | `explain()` orchestrator |
| `quantark/pnlexplain/equity/trades.py` | `ExplainTrade` |
| `quantark/pnlexplain/equity/portfolio.py` | `PositionSnapshot`, `QuotedLegSnapshot`, `BookSnapshot`, `explain_position`, `explain_portfolio`, results |
| `quantark/pnlexplain/equity/recorder.py` | `PnLExplainRecorder` (equity backtest), `ReplayPnLExplainRecorder`, frames |
| `quantark/pnlexplain/equity/bucketed.py` | P5 bucketed vega/rho rows |
| `quantark/pnlexplain/README.md` | module guide |
| `quantark/asset/equity/riskmeasures/greeks_calculator.py` | + `resolve_route` |
| `quantark/asset/equity/lifecycle/manager.py` | + `pricing_products` |
| `quantark/backtest/equity/{config,engine,results}.py` | `pnl_explain` config, hook, frames |
| `quantark/backtest/replay/{config,engine,single,results}.py` | `pnl_explain` config, hook, frames |
| `example/pnl_explain_demo.py` | demo |
| `test/test_pnlexplain_*.py` | tests (flat files, one per task area) |

---

### Task 1: Row schema and result container (`base.py`)

**Files:**
- Create: `quantark/pnlexplain/__init__.py`, `quantark/pnlexplain/base.py`
- Test: `test/test_pnlexplain_base.py`

**Interfaces:**
- Produces: `Factor`, `MARKET_FACTORS`, `ExplainMethod`, `RowKind`, `MOVE_KEYS`, `FRAME_COLUMNS`, `ExplainRow(factor, term, method, kind, level, pnl, moves, greek, cash_greek, step, metadata)`, `ExplainRow.relabel(level)`, `ExplainRow.to_dict()`, `component_sum(rows, method) -> float`, `make_total_row(level, pnl)`, `rows_to_frame(rows, *, date=None, position_id="", underlying="") -> pd.DataFrame`, `ValueBreakdown(contingent_mtm, pending_receivable_pv, paid_cash) .total`, `PnLExplainResult(...)` with `rows_for`, `by_factor`, `reconcile`, `to_frame`, `to_dict`.

- [ ] **Step 1: Write the failing tests**

```python
# test/test_pnlexplain_base.py
"""Row schema, additivity contract, frame schema (spec §5.4)."""
import math
from datetime import datetime

import pandas as pd
import pytest

from quantark.pnlexplain.base import (
    FRAME_COLUMNS, MOVE_KEYS, ExplainMethod, ExplainRow, Factor, RowKind,
    ValueBreakdown, PnLExplainResult, component_sum, make_total_row, rows_to_frame,
)
from quantark.util.exceptions import NumericalError, ValidationError


def _row(factor, term, method, kind, pnl, level="instrument", **kw):
    return ExplainRow(factor=factor, term=term, method=method, kind=kind, level=level,
                      pnl=pnl, **kw)


def _result(rows, total):
    vb0 = ValueBreakdown(10.0, 0.0, 0.0)
    vb1 = ValueBreakdown(10.0 + total, 0.0, 0.0)
    return PnLExplainResult(
        date_t0=datetime(2026, 6, 26), date_t1=datetime(2026, 6, 29),
        pv_t0=vb0, pv_alive_t1=vb1, pv_t1=vb1, total_pnl=total, moves=None,
        rows=tuple(rows), unexplained=None, metadata={},
    )


def test_value_breakdown_total_is_sum():
    vb = ValueBreakdown(contingent_mtm=1.0, pending_receivable_pv=2.0, paid_cash=3.0)
    assert vb.total == 6.0


def test_row_rejects_unknown_level_and_move_key_and_nonfinite():
    with pytest.raises(ValidationError):
        _row(Factor.SPOT, "spot", ExplainMethod.WATERFALL, RowKind.COMPONENT, 1.0, level="book")
    with pytest.raises(ValidationError):
        _row(Factor.SPOT, "spot", ExplainMethod.WATERFALL, RowKind.COMPONENT, 1.0,
             moves={"bogus": 1.0})
    with pytest.raises(NumericalError):
        _row(Factor.SPOT, "spot", ExplainMethod.WATERFALL, RowKind.COMPONENT, math.nan)


def test_component_sum_excludes_informational_and_summary_and_other_method():
    rows = [
        _row(Factor.TIME, "time", ExplainMethod.WATERFALL, RowKind.COMPONENT, 1.0),
        _row(Factor.SPOT, "spot", ExplainMethod.WATERFALL, RowKind.COMPONENT, 2.0),
        _row(Factor.LIFECYCLE_EVENT, "lifecycle_event", ExplainMethod.SHARED, RowKind.COMPONENT, 0.5),
        _row(Factor.TIME, "theta", ExplainMethod.TAYLOR, RowKind.COMPONENT, 0.9),
        _row(Factor.TIME, "r_theta", ExplainMethod.TAYLOR, RowKind.INFORMATIONAL, 0.4),
        _row(Factor.UNEXPLAINED, "unexplained", ExplainMethod.TAYLOR, RowKind.COMPONENT, 2.1),
        make_total_row("instrument", 3.5),
    ]
    assert component_sum(rows, ExplainMethod.WATERFALL) == pytest.approx(3.5)
    assert component_sum(rows, ExplainMethod.TAYLOR) == pytest.approx(3.5)
    # Summing every row is NOT the total (the additivity contract is per method).
    assert sum(r.pnl for r in rows) != pytest.approx(3.5)
    res = _result(rows, 3.5)
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=1e-12)
    assert res.reconcile(ExplainMethod.TAYLOR) == pytest.approx(0.0, abs=1e-12)
    assert res.by_factor(ExplainMethod.WATERFALL) == {
        "time": 1.0, "spot": 2.0, "lifecycle_event": 0.5}
    assert [r.term for r in res.rows_for(ExplainMethod.TAYLOR, kind=RowKind.INFORMATIONAL)] == ["r_theta"]


def test_value_breakdown_and_result_reject_non_finite():
    with pytest.raises(NumericalError):
        ValueBreakdown(math.nan, 0.0, 0.0)
    with pytest.raises(NumericalError):
        _result([], math.inf)


def test_frame_schema_and_empty_frame():
    rows = [
        _row(Factor.SPOT, "spot", ExplainMethod.WATERFALL, RowKind.COMPONENT, 2.0,
             moves={"spot_return": 0.01}, step=2),
    ]
    df = rows_to_frame(rows, date=datetime(2026, 6, 29), position_id="p1", underlying="X")
    assert list(df.columns) == ["date", *FRAME_COLUMNS]
    assert df.loc[0, "factor"] == "spot" and df.loc[0, "method"] == "waterfall"
    assert df.loc[0, "spot_return"] == pytest.approx(0.01)
    assert math.isnan(df.loc[0, "vol_pts"])
    assert str(df["date"].dtype) == "datetime64[ns]"
    empty = rows_to_frame([])
    assert list(empty.columns) == FRAME_COLUMNS and len(empty) == 0


def test_relabel_and_to_dict_keep_order():
    r = _row(Factor.VOL, "vega", ExplainMethod.TAYLOR, RowKind.COMPONENT, 1.5,
             greek=3.0, cash_greek=0.03, moves={"vol_pts": 0.5})
    p = r.relabel("position")
    assert p.level == "position" and p.pnl == 1.5 and r.level == "instrument"
    d = p.to_dict()
    assert list(d.keys()) == FRAME_COLUMNS
    assert d["term"] == "vega" and d["kind"] == "component"
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST test/test_pnlexplain_base.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'quantark.pnlexplain'`

- [ ] **Step 3: Implement `base.py` and an empty package**

```python
# quantark/pnlexplain/__init__.py
"""PnL explain: waterfall + Taylor attribution of PV changes (spec 2026-09-02)."""
from quantark.pnlexplain.base import (  # noqa: F401
    FRAME_COLUMNS, MARKET_FACTORS, MOVE_KEYS, ExplainMethod, ExplainRow, Factor,
    PnLExplainResult, RowKind, ValueBreakdown, component_sum, make_total_row,
    rows_to_frame,
)

__all__ = [
    "FRAME_COLUMNS", "MARKET_FACTORS", "MOVE_KEYS", "ExplainMethod", "ExplainRow",
    "Factor", "PnLExplainResult", "RowKind", "ValueBreakdown", "component_sum",
    "make_total_row", "rows_to_frame",
]
```

```python
# quantark/pnlexplain/base.py
"""Row schema and result container shared by every explain method.

Additivity contract (spec §5.4): for a method M, the COMPONENT rows with
method in {M, SHARED} sum to total_pnl. INFORMATIONAL rows (sub-decompositions)
and SUMMARY rows (total) are never summed.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd

from quantark.util.exceptions import NumericalError, ValidationError


class Factor(Enum):
    TIME = "time"
    SPOT = "spot"
    VOL = "vol"
    RATE = "rate"
    DIVIDEND = "dividend"
    BASIS = "basis"
    MODEL = "model"
    LIFECYCLE_EVENT = "lifecycle_event"
    TRADE = "trade"
    TRANSACTION_COST = "transaction_cost"
    UNEXPLAINED = "unexplained"
    TOTAL = "total"


MARKET_FACTORS: Tuple[Factor, ...] = (
    Factor.TIME, Factor.SPOT, Factor.VOL, Factor.RATE, Factor.DIVIDEND,
    Factor.BASIS, Factor.MODEL,
)


class ExplainMethod(Enum):
    WATERFALL = "waterfall"
    TAYLOR = "taylor"
    SHARED = "shared"


class RowKind(Enum):
    COMPONENT = "component"
    INFORMATIONAL = "informational"
    SUMMARY = "summary"


LEVELS: Tuple[str, ...] = ("instrument", "position", "portfolio")
MOVE_KEYS: Tuple[str, ...] = (
    "spot_return", "vol_pts", "rate_pct", "div_pct", "basis_pct", "days",
    "trading_days", "tenor",
)
FRAME_COLUMNS = [
    "level", "position_id", "underlying", "method", "kind", "factor", "term",
    "step", "pnl", "greek", "cash_greek", *MOVE_KEYS,
]

_EMPTY: Mapping[str, Any] = MappingProxyType({})


def _frozen(mapping: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if mapping is None:
        return _EMPTY
    if isinstance(mapping, MappingProxyType):
        return mapping
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True)
class ExplainRow:
    """One attribution row (money amounts for the whole position)."""

    factor: Factor
    term: str
    method: ExplainMethod
    kind: RowKind
    level: str
    pnl: float
    moves: Mapping[str, float] = field(default_factory=lambda: _EMPTY)
    greek: Optional[float] = None
    cash_greek: Optional[float] = None
    step: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValidationError(f"unknown row level {self.level!r}")
        if not isinstance(self.factor, Factor) or not isinstance(self.method, ExplainMethod) \
                or not isinstance(self.kind, RowKind):
            raise ValidationError("factor/method/kind must be the pnlexplain enums")
        pnl = float(self.pnl)
        if not math.isfinite(pnl):
            raise NumericalError(
                f"non-finite PnL in row {self.method.value}/{self.term}: {self.pnl!r}"
            )
        object.__setattr__(self, "pnl", pnl)
        for name in ("greek", "cash_greek"):
            val = getattr(self, name)
            if val is not None and not math.isfinite(float(val)):
                raise NumericalError(f"non-finite {name} in row {self.term}")
        moves = dict(self.moves or {})
        unknown = sorted(set(moves) - set(MOVE_KEYS))
        if unknown:
            raise ValidationError(f"unknown move keys {unknown}; allowed {MOVE_KEYS}")
        object.__setattr__(self, "moves", MappingProxyType({k: float(v) for k, v in moves.items()}))
        object.__setattr__(self, "metadata", _frozen(self.metadata))

    def relabel(self, level: str) -> "ExplainRow":
        return dataclasses.replace(self, level=level)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "level": self.level, "position_id": "", "underlying": "",
            "method": self.method.value, "kind": self.kind.value,
            "factor": self.factor.value, "term": self.term, "step": self.step,
            "pnl": self.pnl, "greek": self.greek, "cash_greek": self.cash_greek,
        }
        for key in MOVE_KEYS:
            out[key] = self.moves.get(key, math.nan)
        return out


def component_sum(rows: Iterable[ExplainRow], method: ExplainMethod) -> float:
    total = 0.0
    for row in rows:
        if row.kind is RowKind.COMPONENT and row.method in (method, ExplainMethod.SHARED):
            total += row.pnl
    return total


def make_total_row(level: str, pnl: float, **metadata: Any) -> ExplainRow:
    return ExplainRow(
        factor=Factor.TOTAL, term="total", method=ExplainMethod.SHARED,
        kind=RowKind.SUMMARY, level=level, pnl=pnl, metadata=metadata,
    )


def rows_to_frame(
    rows: Iterable[ExplainRow],
    *,
    date: Optional[datetime] = None,
    position_id: str = "",
    underlying: str = "",
) -> pd.DataFrame:
    records = []
    for row in rows:
        rec = row.to_dict()
        rec["position_id"] = position_id
        rec["underlying"] = underlying
        if date is not None:
            rec = {"date": pd.Timestamp(date), **rec}
        records.append(rec)
    columns = (["date"] if date is not None else []) + FRAME_COLUMNS
    frame = pd.DataFrame.from_records(records, columns=columns)
    if date is not None:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


@dataclass(frozen=True)
class ValueBreakdown:
    """Position value = contingent MTM + pending receivable PV + paid cash."""

    contingent_mtm: float
    pending_receivable_pv: float
    paid_cash: float

    def __post_init__(self) -> None:
        for name in ("contingent_mtm", "pending_receivable_pv", "paid_cash"):
            val = float(getattr(self, name))
            if not math.isfinite(val):
                raise NumericalError(f"non-finite {name}: {getattr(self, name)!r}")
            object.__setattr__(self, name, val)

    @property
    def total(self) -> float:
        return self.contingent_mtm + self.pending_receivable_pv + self.paid_cash

    def to_dict(self) -> Dict[str, float]:
        return {
            "contingent_mtm": self.contingent_mtm,
            "pending_receivable_pv": self.pending_receivable_pv,
            "paid_cash": self.paid_cash, "total": self.total,
        }


@dataclass(frozen=True)
class PnLExplainResult:
    date_t0: datetime
    date_t1: datetime
    pv_t0: ValueBreakdown
    pv_alive_t1: ValueBreakdown
    pv_t1: ValueBreakdown
    total_pnl: float
    moves: Any                      # FactorMoves (equity); typed loosely to avoid a cycle
    rows: Tuple[ExplainRow, ...]
    unexplained: Optional[float]
    metadata: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.total_pnl)):
            raise NumericalError(f"non-finite total_pnl: {self.total_pnl!r}")
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "metadata", _frozen(self.metadata))

    def rows_for(self, method: ExplainMethod, *, kind: Optional[RowKind] = None
                 ) -> Tuple[ExplainRow, ...]:
        return tuple(
            r for r in self.rows
            if r.method in (method, ExplainMethod.SHARED) and (kind is None or r.kind is kind)
        )

    def by_factor(self, method: ExplainMethod) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for r in self.rows_for(method, kind=RowKind.COMPONENT):
            out[r.factor.value] = out.get(r.factor.value, 0.0) + r.pnl
        return out

    def reconcile(self, method: ExplainMethod) -> float:
        return self.total_pnl - component_sum(self.rows, method)

    def to_frame(self) -> pd.DataFrame:
        return rows_to_frame(self.rows)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date_t0": self.date_t0.isoformat(), "date_t1": self.date_t1.isoformat(),
            "pv_t0": self.pv_t0.to_dict(), "pv_alive_t1": self.pv_alive_t1.to_dict(),
            "pv_t1": self.pv_t1.to_dict(), "total_pnl": self.total_pnl,
            "unexplained": self.unexplained, "rows": [r.to_dict() for r in self.rows],
            "metadata": dict(self.metadata),
        }
```

- [ ] **Step 4: Run tests**

Run: `PYTEST test/test_pnlexplain_base.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add quantark/pnlexplain/__init__.py quantark/pnlexplain/base.py test/test_pnlexplain_base.py
git commit -m "feat(pnlexplain): row schema, additivity contract, result container" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_019fnLipZtAJAPAQJLLEXQMb"
```

---

### Task 2: Configuration (`config.py`)

**Files:**
- Create: `quantark/pnlexplain/config.py`
- Test: `test/test_pnlexplain_config.py`

**Interfaces:**
- Produces: `PnLExplainConfig` (frozen dataclass, spec §5.5), `STENCILS`, `THETA_SUBROWS`, `TERM_FACTOR`, `resolve_stencil(config) -> tuple[str, ...]` (component terms in stencil order, aliases resolved), `resolved_subrows(config) -> tuple[str, ...]`.

- [ ] **Step 1: Write the failing tests**

```python
# test/test_pnlexplain_config.py
import pytest

from quantark.pnlexplain.base import ExplainMethod, Factor, MARKET_FACTORS
from quantark.pnlexplain.config import (
    PnLExplainConfig, STENCILS, TERM_FACTOR, resolve_stencil, resolved_subrows,
)
from quantark.util.exceptions import ValidationError


def test_defaults():
    cfg = PnLExplainConfig()
    assert cfg.methods == (ExplainMethod.WATERFALL, ExplainMethod.TAYLOR)
    assert cfg.waterfall_order == MARKET_FACTORS
    assert resolve_stencil(cfg) == STENCILS["standard"]
    assert resolved_subrows(cfg) == ("r_theta", "q_theta", "convexity_theta")


@pytest.mark.parametrize("kw", [
    dict(methods=()),
    dict(methods=(ExplainMethod.SHARED,)),
    dict(methods=(ExplainMethod.TAYLOR, ExplainMethod.TAYLOR)),
    dict(waterfall_order=(Factor.TIME, Factor.SPOT)),
    dict(waterfall_order=MARKET_FACTORS + (Factor.TIME,)),
    dict(waterfall_order=(Factor.LIFECYCLE_EVENT,) + MARKET_FACTORS[1:]),
    dict(interaction="random"),
    dict(time_term="gap"),
    dict(theta_decomposition_mode="approx"),
    dict(greeks_method="closed_form"),
    dict(clock="1td"),                      # clock with exact_gap
    dict(time_term="per_step", clock="2d"),
    dict(stencil="huge"),
    dict(stencil=["delta", "delta"]),
    dict(stencil=["delta", "theta_1td"]),   # clock qualifier rejected
    dict(stencil=["r_theta"]),              # sub-row without theta
    dict(stencil=["bogus"]),
])
def test_invalid_configs_raise(kw):
    with pytest.raises(ValidationError):
        PnLExplainConfig(**kw)


def test_explicit_stencil_resolves_aliases_and_keeps_order():
    cfg = PnLExplainConfig(stencil=["veta", "delta", "rhoq", "theta", "gamma_theta"])
    assert resolve_stencil(cfg) == ("vega_theta", "delta", "dividend_rho", "theta")
    assert resolved_subrows(cfg) == ("gamma_theta",)
    assert PnLExplainConfig(waterfall_order=tuple(reversed(MARKET_FACTORS))).waterfall_order[0] is Factor.MODEL


def test_term_factor_table_covers_every_stencil_term():
    for name in STENCILS["extended"]:
        assert name in TERM_FACTOR
    assert TERM_FACTOR["vanna"] is Factor.VOL and TERM_FACTOR["charm"] is Factor.TIME
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST test/test_pnlexplain_config.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'quantark.pnlexplain.config'`

- [ ] **Step 3: Implement `config.py`**

```python
# quantark/pnlexplain/config.py
"""PnLExplainConfig and the stencil / term tables (spec §5.5, §7.3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Union

from quantark.asset.equity.param import EngineParams
from quantark.asset.equity.riskmeasures.greeks import registry
from quantark.pnlexplain.base import MARKET_FACTORS, ExplainMethod, Factor
from quantark.util.enum.engine_enums import GreeksCalculationMode
from quantark.util.enum.greek_conventions import GreekConvention
from quantark.util.exceptions import ValidationError

STENCILS = {
    "first_order": ("delta", "vega", "theta", "rho", "dividend_rho"),
    "standard": ("delta", "vega", "theta", "rho", "dividend_rho", "gamma", "volga", "vanna"),
    "extended": (
        "delta", "vega", "theta", "rho", "dividend_rho", "gamma", "volga", "vanna",
        "speed", "zomma", "charm", "color", "vega_theta", "dividend_volga", "delta_q",
    ),
}
THETA_SUBROWS: Tuple[str, ...] = ("r_theta", "q_theta", "convexity_theta", "gamma_theta")
TERM_FACTOR = {
    "delta": Factor.SPOT, "gamma": Factor.SPOT, "speed": Factor.SPOT,
    "vega": Factor.VOL, "volga": Factor.VOL, "vanna": Factor.VOL, "zomma": Factor.VOL,
    "theta": Factor.TIME, "charm": Factor.TIME, "color": Factor.TIME, "vega_theta": Factor.TIME,
    "r_theta": Factor.TIME, "q_theta": Factor.TIME, "convexity_theta": Factor.TIME,
    "gamma_theta": Factor.TIME, "theta_contract": Factor.TIME, "ledger_carry": Factor.TIME,
    "rho": Factor.RATE,
    "dividend_rho": Factor.DIVIDEND, "dividend_volga": Factor.DIVIDEND, "delta_q": Factor.DIVIDEND,
}
_INTERACTIONS = ("sequential", "shapley")
_TIME_TERMS = ("exact_gap", "per_step")
_THETA_MODES = ("estimate", "exact")
_GREEK_METHODS = ("auto", "analytical", "numerical")
_CLOCKS = (None, "1d", "1td")


def _canonical(name: object) -> str:
    """One stencil entry -> canonical registry name; clock qualifiers rejected."""
    requests = registry.normalize_greeks([name])
    (req,) = tuple(requests)
    if req.clock is not None:
        raise ValidationError(
            f"stencil entries carry no clock qualifier ({name!r}); use PnLExplainConfig.clock"
        )
    return req.canonical


@dataclass(frozen=True)
class PnLExplainConfig:
    methods: Tuple[ExplainMethod, ...] = (ExplainMethod.WATERFALL, ExplainMethod.TAYLOR)
    waterfall_order: Tuple[Factor, ...] = MARKET_FACTORS
    interaction: str = "sequential"
    spot_convention: GreekConvention = GreekConvention.STICKY_STRIKE
    stencil: Union[str, Sequence[str]] = "standard"
    time_term: str = "exact_gap"
    clock: Optional[str] = None
    theta_decomposition_mode: str = "estimate"
    bucketed: bool = False
    greeks_method: str = "auto"
    greeks_mode: GreeksCalculationMode = GreeksCalculationMode.BUMP
    params: Optional[EngineParams] = None
    _terms: Tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)
    _subrows: Tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        methods = tuple(self.methods)
        if not methods or len(set(methods)) != len(methods) or any(
            m not in (ExplainMethod.WATERFALL, ExplainMethod.TAYLOR) for m in methods
        ):
            raise ValidationError(
                "methods must be a non-empty, duplicate-free subset of {WATERFALL, TAYLOR}"
            )
        object.__setattr__(self, "methods", methods)
        order = tuple(self.waterfall_order)
        if sorted(f.value for f in order) != sorted(f.value for f in MARKET_FACTORS) \
                or len(order) != len(MARKET_FACTORS):
            raise ValidationError(
                "waterfall_order must be a permutation of the seven market factors"
            )
        object.__setattr__(self, "waterfall_order", order)
        if self.interaction not in _INTERACTIONS:
            raise ValidationError(f"interaction must be one of {_INTERACTIONS}")
        if self.time_term not in _TIME_TERMS:
            raise ValidationError(f"time_term must be one of {_TIME_TERMS}")
        if self.theta_decomposition_mode not in _THETA_MODES:
            raise ValidationError(f"theta_decomposition_mode must be one of {_THETA_MODES}")
        if self.greeks_method not in _GREEK_METHODS:
            raise ValidationError(f"greeks_method must be one of {_GREEK_METHODS}")
        if self.clock not in _CLOCKS:
            raise ValidationError("clock must be None, '1d' or '1td'")
        if self.clock is not None and self.time_term != "per_step":
            raise ValidationError("clock is only meaningful with time_term='per_step'")
        if not isinstance(self.spot_convention, GreekConvention):
            raise ValidationError("spot_convention must be a GreekConvention")
        if self.spot_convention not in (GreekConvention.STICKY_STRIKE, GreekConvention.STICKY_MONEYNESS):
            raise ValidationError("spot_convention must be STICKY_STRIKE or STICKY_MONEYNESS")
        if not isinstance(self.greeks_mode, GreeksCalculationMode):
            raise ValidationError("greeks_mode must be a GreeksCalculationMode")
        terms, subrows = self._resolve_stencil()
        object.__setattr__(self, "_terms", terms)
        object.__setattr__(self, "_subrows", subrows)

    def _resolve_stencil(self) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        if isinstance(self.stencil, str):
            if self.stencil not in STENCILS:
                raise ValidationError(f"unknown stencil {self.stencil!r}; known {tuple(STENCILS)}")
            terms = STENCILS[self.stencil]
            subrows = ("r_theta", "q_theta", "convexity_theta") + (
                ("gamma_theta",) if self.stencil == "extended" else ()
            )
            return terms, subrows
        names = [_canonical(n) for n in self.stencil]
        if len(set(names)) != len(names):
            raise ValidationError(f"duplicate stencil entries after alias resolution: {names}")
        unknown = [n for n in names if n not in TERM_FACTOR or n in ("theta_contract", "ledger_carry")]
        if unknown:
            raise ValidationError(f"stencil entries are not Taylor terms: {unknown}")
        subrows = tuple(n for n in names if n in THETA_SUBROWS)
        terms = tuple(n for n in names if n not in THETA_SUBROWS)
        if subrows and "theta" not in terms:
            raise ValidationError("theta sub-rows require 'theta' in the stencil")
        return terms, subrows


def resolve_stencil(config: PnLExplainConfig) -> Tuple[str, ...]:
    return config._terms


def resolved_subrows(config: PnLExplainConfig) -> Tuple[str, ...]:
    return config._subrows
```

- [ ] **Step 4: Run tests**

Run: `PYTEST test/test_pnlexplain_config.py -q`
Expected: all passed (the parametrised invalid-config test has 17 cases)

- [ ] **Step 5: Commit**

```bash
git add quantark/pnlexplain/config.py test/test_pnlexplain_config.py
git commit -m "feat(pnlexplain): PnLExplainConfig with full validation and stencil tables" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_019fnLipZtAJAPAQJLLEXQMb"
```

---

### Task 3: Snapshot, value identity, fingerprints

**Files:**
- Create: `quantark/pnlexplain/equity/__init__.py`, `quantark/pnlexplain/equity/snapshot.py`, `quantark/pnlexplain/equity/fingerprints.py`
- Test: `test/test_pnlexplain_snapshot.py`

**Interfaces:**
- Produces: `ValuationSnapshot(product, engine, pricing_env, date, quantity=1.0, lifecycle_state=None, valuation_point=None, currency=None, label="")` with `.point`; `is_terminal(state) -> bool`; `value(snapshot, *, engine=None, product=None, pricing_env=None, valuation_point=None, lifecycle_state=<snapshot's>) -> ValueBreakdown`; `calendars_equal(a, b) -> bool`; `contract_fingerprint(product) -> tuple`; `check_contract_roll(product_t0, product_alive_t1, calendar_days)`; `lifecycle_fingerprint(state) -> tuple`; `LIFECYCLE_FINGERPRINT_VERSION = "v1"`.

- [ ] **Step 1: Write the failing tests**

```python
# test/test_pnlexplain_snapshot.py
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.lifecycle.cashflows import RealizedCashflow, ValuationPoint
from quantark.asset.equity.lifecycle.events import LifecycleEventType
from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState, BarrierLifecycleState
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.pnlexplain.equity.fingerprints import (
    calendars_equal, check_contract_roll, contract_fingerprint, lifecycle_fingerprint,
)
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot, is_terminal, value
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.enum import OptionType
from quantark.util.exceptions import NumericalError, ValidationError

D0 = datetime(2026, 6, 26)


def _env(spot=100.0, date=D0, calendar=None):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=date, calendar=calendar,
    )


def _call(maturity=1.0):
    return EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=maturity)


def test_snapshot_validation():
    env = _env()
    ok = ValuationSnapshot(product=_call(), engine=BlackScholesEngine(), pricing_env=env, date=D0)
    assert ok.point == ValuationPoint(date=D0)
    with pytest.raises(ValidationError):  # date != env date
        ValuationSnapshot(_call(), BlackScholesEngine(), env, date=D0 + timedelta(days=1))
    with pytest.raises(ValidationError):  # intraday
        env2 = _env(date=D0 + timedelta(hours=3))
        ValuationSnapshot(_call(), BlackScholesEngine(), env2, date=D0 + timedelta(hours=3))
    with pytest.raises(ValidationError):  # aware
        env3 = _env(date=D0.replace(tzinfo=timezone.utc))
        ValuationSnapshot(_call(), BlackScholesEngine(), env3, date=D0.replace(tzinfo=timezone.utc))
    with pytest.raises(ValidationError):  # zero quantity
        ValuationSnapshot(_call(), BlackScholesEngine(), env, date=D0, quantity=0.0)
    with pytest.raises(ValidationError):  # date-based point must equal date
        ValuationSnapshot(_call(), BlackScholesEngine(), env, date=D0,
                          valuation_point=ValuationPoint(date=D0 + timedelta(days=1)))


def test_value_identity_contingent_plus_ledger():
    env = _env()
    engine = BlackScholesEngine()
    snap = ValuationSnapshot(_call(), engine, env, date=D0, quantity=3.0)
    vb = value(snap)
    assert vb.contingent_mtm == pytest.approx(3.0 * engine.price(_call(), env))
    assert vb.pending_receivable_pv == 0.0 and vb.paid_cash == 0.0

    state = AutocallableLifecycleState()
    state.mark_ko(D0 - timedelta(days=5), cashflow=30.0,
                  settlement_date=D0 + timedelta(days=2))
    term = ValuationSnapshot(_call(), engine, env, date=D0, quantity=3.0, lifecycle_state=state)
    assert is_terminal(state)
    vb2 = value(term)
    assert vb2.contingent_mtm == 0.0
    df = env.get_discount_factor(2.0 / 365.0)
    assert vb2.pending_receivable_pv == pytest.approx(30.0 * df, rel=1e-9)
    # after payment the same ledger is paid cash
    env_later = _env(date=D0 + timedelta(days=3))
    vb3 = value(term, pricing_env=env_later, valuation_point=ValuationPoint(date=D0 + timedelta(days=3)))
    assert vb3.pending_receivable_pv == 0.0 and vb3.paid_cash == pytest.approx(30.0)


def test_calendar_semantic_equality():
    a = create_calendar(CalendarType.CHINA_SSE)
    assert calendars_equal(a, deepcopy(a))
    assert calendars_equal(None, None)
    assert not calendars_equal(a, None)
    assert not calendars_equal(a, create_calendar(CalendarType.US))


def test_contract_fingerprint_and_roll():
    p0 = _call(maturity=1.0)
    p1 = _call(maturity=1.0 - 3 / 365)
    assert contract_fingerprint(p0) == contract_fingerprint(p1)          # maturity is rolled
    check_contract_roll(p0, p1, calendar_days=3)                          # ok
    with pytest.raises(ValidationError):
        check_contract_roll(p0, _call(maturity=1.0 - 2 / 365), calendar_days=3)
    with pytest.raises(ValidationError):                                  # different strike
        check_contract_roll(p0, EuropeanVanillaOption(strike=105.0, option_type=OptionType.CALL,
                                                      maturity=1.0 - 3 / 365), calendar_days=3)
    # a product carrying the tracker's KI flag differs only by a rolled field
    p2 = deepcopy(p1)
    setattr(p2, "_otc_lifecycle_knocked_in", True)
    assert contract_fingerprint(p2) == contract_fingerprint(p1)


def test_value_rejects_non_finite_price():
    class NanEngine:
        def price(self, product, env):
            return float("nan")

    snap = ValuationSnapshot(_call(), NanEngine(), _env(), date=D0)
    with pytest.raises(NumericalError):
        value(snap)


def test_lifecycle_fingerprint_is_computed_and_sensitive():
    assert lifecycle_fingerprint(None) == ("v1", None)
    s = AutocallableLifecycleState()
    f0 = lifecycle_fingerprint(s)
    assert f0[0] == "v1" and f0[1] == "AutocallableLifecycleState"
    assert lifecycle_fingerprint(deepcopy(s)) == f0
    s.mark_ki(D0)
    assert lifecycle_fingerprint(s) != f0
    s2 = AutocallableLifecycleState()
    s2.coupon_memory_count = 2                    # a plain int field also counts
    assert lifecycle_fingerprint(s2) != f0
    b = BarrierLifecycleState()
    b.ledger.register(RealizedCashflow(
        cashflow_id="x", event_type=LifecycleEventType.COUPON, amount=1.0,
        determination_date=D0, payment_date=D0))
    assert lifecycle_fingerprint(b) != lifecycle_fingerprint(BarrierLifecycleState())
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST test/test_pnlexplain_snapshot.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `fingerprints.py`**

```python
# quantark/pnlexplain/equity/fingerprints.py
"""Semantic identities: calendars, contracts (rolled in time), lifecycle states.

contract_fingerprint excludes the fields the lifecycle trackers change day to
day (spec §5.3); lifecycle_fingerprint is generic over every dataclass field
of a state so a new pricing-relevant field can never be missed (spec §8).
"""
from __future__ import annotations

import dataclasses
from datetime import date, datetime
from enum import Enum
from typing import Any, Mapping, Tuple

import numpy as np

from quantark.asset.equity.lifecycle.cashflows import LifecycleCashflowLedger, ValuationPoint
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close

LIFECYCLE_FINGERPRINT_VERSION = "v1"
ROLL_TOL = 1e-12        # spec §5.3 roll equation tolerance
ROLLED_FIELDS = frozenset({"maturity", "_otc_lifecycle_knocked_in"})
SCHEDULE_FIELDS = frozenset({
    "barrier_config", "post_barrier_config", "observation_schedule",
    "coupon_schedule", "ko_observation_schedule", "ki_observation_schedule",
})
_TIMING_TOKENS = ("date", "time", "schedule", "record")
MATURITY_FLOOR = 1e-8   # the trackers clamp a rolled float maturity here


def calendars_equal(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if type(a) is not type(b):
        return False
    if getattr(a, "name", None) != getattr(b, "name", None):
        return False
    return set(getattr(a, "holidays", ())) == set(getattr(b, "holidays", ()))


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return ("enum", type(value).__name__, value.value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, ValuationPoint):
        return ("date", value.date.isoformat()) if value.date is not None else ("time", float(value.time))
    if isinstance(value, LifecycleCashflowLedger):
        return ("ledger", tuple(
            (cf.cashflow_id, cf.event_type.value, float(cf.amount),
             cf.determination_date.isoformat() if cf.determination_date is not None else cf.determination_time,
             cf.payment_date.isoformat() if cf.payment_date is not None else cf.payment_time,
             tuple(sorted((k, _normalize(v)) for k, v in cf.metadata.items())))
            for cf in value.cashflows
        ))
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_normalize(v) for v in value))
    if isinstance(value, (list, tuple)):
        return tuple(_normalize(v) for v in value)
    if isinstance(value, np.ndarray):
        return ("ndarray", value.shape, tuple(value.ravel().tolist()))
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _normalize(v)) for k, v in value.items()))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (type(value).__name__, tuple(
            (f.name, _normalize(getattr(value, f.name))) for f in dataclasses.fields(value)
        ))
    return ("repr", type(value).__name__, repr(value))


def _public_fields(obj: Any) -> Tuple[Tuple[str, Any], ...]:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        names = [f.name for f in dataclasses.fields(obj)]
    else:
        names = [k for k in vars(obj) if not k.startswith("_")]
    names += [k for k in vars(obj) if k in ROLLED_FIELDS and k not in names]
    return tuple((n, getattr(obj, n)) for n in names)


def _static_terms(value: Any) -> Any:
    """Schedule-bearing config -> its static terms (levels, rates, counts, indices)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (type(value).__name__, tuple(
            (f.name, _normalize(getattr(value, f.name)))
            for f in dataclasses.fields(value)
            if not any(tok in f.name for tok in _TIMING_TOKENS)
        ))
    return _normalize(value)


def contract_fingerprint(product: Any) -> tuple:
    items = []
    for name, val in _public_fields(product):
        if name in ROLLED_FIELDS:
            continue
        items.append((name, _static_terms(val) if name in SCHEDULE_FIELDS else _normalize(val)))
    return (type(product).__name__, tuple(items))


def check_contract_roll(product_t0: Any, product_alive_t1: Any, calendar_days: int) -> None:
    """Raise unless product_alive_t1 is product_t0 rolled forward calendar_days (spec §5.3)."""
    if contract_fingerprint(product_t0) != contract_fingerprint(product_alive_t1):
        raise ValidationError("contract replacement is not a time step")
    m0 = getattr(product_t0, "maturity", None)
    m1 = getattr(product_alive_t1, "maturity", None)
    if m0 is None or m1 is None:
        return
    date_based = getattr(product_t0, "exercise_date", None) is not None \
        or getattr(product_t0, "maturity_date", None) is not None
    if date_based:
        return                      # dates are in the fingerprint; the float is metadata
    m0, m1 = float(m0), float(m1)
    if m1 <= MATURITY_FLOOR + ROLL_TOL:
        return                      # clamped at the trackers' floor
    if not is_close(m0 - m1, calendar_days / 365.0, rel_tol=0.0, abs_tol=ROLL_TOL):
        raise ValidationError(
            f"alive product maturity {m1} is not {m0} rolled by {calendar_days} days "
            "(a float-maturity contract must be supplied rolled by calendar_days/365)"
        )


BOOKKEEPING_FIELDS = frozenset({
    "observed_ko_indices", "observed_ki_indices", "observed_coupon_indices",
})   # grow on every observation date without an event; pricing-neutral (spec §8)


def lifecycle_fingerprint(state: Any) -> tuple:
    if state is None:
        return (LIFECYCLE_FINGERPRINT_VERSION, None)
    if not dataclasses.is_dataclass(state):
        raise ValidationError(f"lifecycle state must be a dataclass, got {type(state).__name__}")
    fields = tuple(
        (f.name, _normalize(getattr(state, f.name)))
        for f in dataclasses.fields(state) if f.name not in BOOKKEEPING_FIELDS
    )
    return (LIFECYCLE_FINGERPRINT_VERSION, type(state).__name__, fields)
```

- [ ] **Step 4: Implement `snapshot.py` and the subpackage init**

```python
# quantark/pnlexplain/equity/__init__.py
"""Equity PnL explain."""
```

```python
# quantark/pnlexplain/equity/snapshot.py
"""ValuationSnapshot and the value identity (spec §5.1-5.2)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from quantark.asset.equity.lifecycle.cashflows import ValuationPoint
from quantark.pnlexplain.base import ValueBreakdown
from quantark.util.exceptions import NumericalError, ValidationError

_MISSING = object()


def is_terminal(state: Any) -> bool:
    """Same predicate as settlement_support.terminal_lifecycle_pv."""
    if state is None:
        return False
    return (
        getattr(state, "alive", None) is False
        or bool(getattr(state, "matured", False))
        or bool(getattr(state, "expired", False))
        or bool(getattr(state, "knocked_out", False))
    )


@dataclass(frozen=True)
class ValuationSnapshot:
    product: Any
    engine: Any
    pricing_env: Any
    date: datetime
    quantity: float = 1.0
    lifecycle_state: Any = None
    valuation_point: Optional[ValuationPoint] = None
    currency: Optional[str] = None
    label: str = ""

    def __post_init__(self) -> None:
        d = self.date
        if not isinstance(d, datetime) or d.tzinfo is not None:
            raise ValidationError("snapshot date must be a naive datetime")
        if (d.hour, d.minute, d.second, d.microsecond) != (0, 0, 0, 0):
            raise ValidationError("snapshot date must be midnight (date-only snapshots)")
        env_date = self.pricing_env.valuation_date
        if env_date != d:
            raise ValidationError(
                f"snapshot date {d} must equal pricing_env.valuation_date {env_date}"
            )
        spot = float(self.pricing_env.spot)
        if not math.isfinite(spot) or spot <= 0.0:
            raise ValidationError(f"spot must be positive and finite, got {spot}")
        q = float(self.quantity)
        if not math.isfinite(q) or q == 0.0:
            raise ValidationError(f"quantity must be non-zero and finite, got {self.quantity}")
        object.__setattr__(self, "quantity", q)
        vp = self.valuation_point
        if vp is not None and vp.date is not None and vp.date != d:
            raise ValidationError("a date-based valuation_point must equal the snapshot date")

    @property
    def point(self) -> ValuationPoint:
        return self.valuation_point if self.valuation_point is not None else ValuationPoint(date=self.date)


def value(
    snapshot: ValuationSnapshot,
    *,
    engine: Any = None,
    product: Any = None,
    pricing_env: Any = None,
    valuation_point: Optional[ValuationPoint] = None,
    lifecycle_state: Any = _MISSING,
) -> ValueBreakdown:
    """contingent MTM (quantity x engine price; 0 once terminal) + ledger PV + paid cash.

    The engine is NOT handed the lifecycle state: receivables are valued here
    from the ledger exactly as the backtests do, so nothing is counted twice.
    """
    engine = snapshot.engine if engine is None else engine
    product = snapshot.product if product is None else product
    env = snapshot.pricing_env if pricing_env is None else pricing_env
    point = snapshot.point if valuation_point is None else valuation_point
    state = snapshot.lifecycle_state if lifecycle_state is _MISSING else lifecycle_state

    if is_terminal(state):
        contingent = 0.0
    else:
        price = float(engine.price(product, env))
        if not math.isfinite(price):
            raise NumericalError(
                f"engine {type(engine).__name__} returned a non-finite price {price!r}"
            )
        contingent = snapshot.quantity * price
    pending = paid = 0.0
    if state is not None:
        ledger = getattr(state, "ledger", None)
        if ledger is None:
            raise ValidationError("lifecycle_state requires a cashflow ledger")
        if ledger.cashflows:
            pending = float(ledger.pending_pv(point, env))
            paid = float(ledger.paid_total(point))
    return ValueBreakdown(contingent_mtm=contingent, pending_receivable_pv=pending, paid_cash=paid)
```

- [ ] **Step 5: Run tests**

Run: `PYTEST test/test_pnlexplain_snapshot.py -q`
Expected: 6 passed. The ledger discounts a pending receivable through `SettlementResolver.resolve_pending`, which converts the payment date with the environment's own day count (`calculate_year_fraction`; 2/365 on a CALENDAR_DAYS env), so `30 × env.get_discount_factor(2/365)` is the exact expectation.

- [ ] **Step 6: Commit**

```bash
git add quantark/pnlexplain/equity/__init__.py quantark/pnlexplain/equity/snapshot.py quantark/pnlexplain/equity/fingerprints.py test/test_pnlexplain_snapshot.py
git commit -m "feat(pnlexplain): ValuationSnapshot, value identity, contract/lifecycle fingerprints" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_019fnLipZtAJAPAQJLLEXQMb"
```

---

### Task 4: Factor coordinate and factor moves

**Files:**
- Create: `quantark/pnlexplain/equity/coordinate.py`, `quantark/pnlexplain/equity/factor_diff.py`
- Test: `test/test_pnlexplain_factor_diff.py`

**Interfaces:**
- Produces: `FactorCoordinate(reference_strike, tenor_t1, applicable)`; `resolve_coordinate(product_t0, spot_t0, product_alive_t1, env_t1) -> FactorCoordinate`; `FactorMoves` (spec §5.3) with `.display(factor) -> dict`; `build_factor_moves(snap0, snap1, coordinate, *, engine_alive_t1, lifecycle_changed) -> FactorMoves`; `validate_pair(snap0, snap1)`; `objects_equal(a, b) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
# test/test_pnlexplain_factor_diff.py
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.analytical import DeltaOneEngine
from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.product.deltaone import Futures, SpotInstrument
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote, TermStructureVolSurface
from quantark.param.div import ContinuousDividendYield
from quantark.pnlexplain.base import Factor, MARKET_FACTORS
from quantark.pnlexplain.equity.coordinate import resolve_coordinate
from quantark.pnlexplain.equity.factor_diff import build_factor_moves, validate_pair
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.enum import OptionType
from quantark.util.enum.deltaone_enums import DeltaOneType
from quantark.util.exceptions import NumericalError, ValidationError

FRI = datetime(2026, 6, 26)
MON = datetime(2026, 6, 29)
# MODEL is detected by engine identity (spec §5.1): both snapshots share one engine object.
ENG = BlackScholesEngine()
D1 = DeltaOneEngine()


def _env(spot, vol, rate, div, date, calendar=None, **kw):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=vol, rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div), valuation_date=date, calendar=calendar, **kw)


def _call(m):
    return EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=m)


def test_moves_read_at_t1_coordinate_and_changed_set():
    cal = create_calendar(CalendarType.CHINA_SSE)
    e0 = _env(100.0, TermStructureVolSurface(times=[0.5, 1.0], vols=[0.20, 0.25]), 0.03, 0.01, FRI, cal)
    e1 = _env(102.0, TermStructureVolSurface(times=[0.5, 1.0], vols=[0.21, 0.26]), 0.03, 0.02, MON, cal)
    p0, p1 = _call(1.0), _call(1.0 - 3 / 365)
    s0 = ValuationSnapshot(p0, ENG, e0, date=FRI)
    s1 = ValuationSnapshot(p1, ENG, e1, date=MON)
    coord = resolve_coordinate(p0, 100.0, p1, e1)
    assert coord.reference_strike == 100.0
    assert coord.tenor_t1 == pytest.approx(1.0 - 3 / 365)
    assert coord.applicable == frozenset(MARKET_FACTORS)
    mv = build_factor_moves(s0, s1, coord, engine_alive_t1=s1.engine, lifecycle_changed=False)
    t1 = coord.tenor_t1
    assert mv.vol_t0 == pytest.approx(e0.get_vol(100.0, t1))
    assert mv.vol_t1 == pytest.approx(e1.get_vol(100.0, t1))
    assert mv.d_vol == pytest.approx(mv.vol_t1 - mv.vol_t0)
    assert mv.d_rate == 0.0 and mv.d_div == pytest.approx(0.01)
    assert mv.calendar_days == 3 and mv.trading_days == 1
    assert Factor.RATE not in mv.changed and Factor.DIVIDEND in mv.changed
    assert Factor.TIME in mv.changed and Factor.MODEL not in mv.changed
    assert mv.display(Factor.SPOT) == {"spot_return": pytest.approx(0.02)}
    assert mv.display(Factor.TIME) == {"days": 3.0, "trading_days": 1.0}
    assert "trading_days" not in build_factor_moves(
        ValuationSnapshot(p0, ENG, _env(100.0, FlatVolSurface(0.2), 0.03, 0.01, FRI), date=FRI),
        ValuationSnapshot(p1, ENG, _env(100.0, FlatVolSurface(0.2), 0.03, 0.01, MON), date=MON),
        coord, engine_alive_t1=None, lifecycle_changed=False).display(Factor.TIME)
    # two distinct but equivalent engine objects ARE a model change (identity rule)
    other = build_factor_moves(s0, s1, coord, engine_alive_t1=BlackScholesEngine(), lifecycle_changed=False)
    assert Factor.MODEL in other.changed


def test_delta_one_coordinates():
    e0 = _env(100.0, FlatVolSurface(0.2), 0.03, 0.01, FRI)
    e1 = _env(101.0, FlatVolSurface(0.2), 0.03, 0.01, MON)
    spot = SpotInstrument(underlying="X", deltaone_type=DeltaOneType.STOCK)
    c = resolve_coordinate(spot, 100.0, spot, e1)
    assert c.applicable == frozenset({Factor.SPOT, Factor.MODEL}) and c.tenor_t1 is None
    fut = Futures(underlying="X", multiplier=300.0, maturity=0.5)
    fut1 = Futures(underlying="X", multiplier=300.0, maturity=0.5 - 3 / 365)
    c2 = resolve_coordinate(fut, 100.0, fut1, e1)
    assert Factor.VOL not in c2.applicable and Factor.BASIS in c2.applicable
    s0 = ValuationSnapshot(spot, D1, e0, date=FRI)
    s1 = ValuationSnapshot(spot, D1, e1, date=MON)
    mv = build_factor_moves(s0, s1, c, engine_alive_t1=s1.engine, lifecycle_changed=False)
    assert mv.vol_t0 is None and mv.d_vol is None and mv.display(Factor.VOL) == {}
    assert Factor.VOL not in mv.changed


def test_expiry_day_drops_market_factors():
    e1 = _env(100.0, FlatVolSurface(0.2), 0.03, 0.01, MON)
    c = resolve_coordinate(_call(3 / 365), 100.0, _call(1e-8), e1)
    assert c.applicable == frozenset({Factor.TIME, Factor.SPOT, Factor.MODEL})


def test_validate_pair_rejects_order_clock_quantity_currency():
    cal = create_calendar(CalendarType.CHINA_SSE)
    e0 = _env(100.0, FlatVolSurface(0.2), 0.03, 0.01, FRI, cal)
    e1 = _env(100.0, FlatVolSurface(0.2), 0.03, 0.01, MON, cal)
    s0 = ValuationSnapshot(_call(1.0), BlackScholesEngine(), e0, date=FRI)
    s1 = ValuationSnapshot(_call(1.0 - 3 / 365), BlackScholesEngine(), e1, date=MON)
    validate_pair(s0, s1)
    with pytest.raises(ValidationError):
        validate_pair(s1, s0)
    e_us = _env(100.0, FlatVolSurface(0.2), 0.03, 0.01, MON, create_calendar(CalendarType.US))
    with pytest.raises(ValidationError):
        validate_pair(s0, ValuationSnapshot(_call(1.0 - 3 / 365), BlackScholesEngine(), e_us, date=MON))
    e_bd = _env(100.0, FlatVolSurface(0.2), 0.03, 0.01, MON, cal,
                day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=244)
    with pytest.raises(ValidationError):
        validate_pair(s0, ValuationSnapshot(_call(1.0 - 3 / 365), BlackScholesEngine(), e_bd, date=MON))
    with pytest.raises(ValidationError):
        validate_pair(s0, ValuationSnapshot(_call(1.0 - 3 / 365), BlackScholesEngine(), e1, date=MON, quantity=2.0))
    with pytest.raises(ValidationError):
        validate_pair(ValuationSnapshot(_call(1.0), BlackScholesEngine(), e0, date=FRI, currency="CNY"),
                      ValuationSnapshot(_call(1.0 - 3 / 365), BlackScholesEngine(), e1, date=MON, currency="USD"))


def test_non_finite_sample_raises():
    class InfSurface(FlatVolSurface):
        def get_vol(self, strike, time_to_maturity, spot=None):
            return float("inf")

    e0 = _env(100.0, FlatVolSurface(0.2), 0.03, 0.01, FRI)
    e1 = _env(100.0, InfSurface(0.2), 0.03, 0.01, MON)
    p0, p1 = _call(1.0), _call(1.0 - 3 / 365)
    coord = resolve_coordinate(p0, 100.0, p1, e1)
    with pytest.raises(NumericalError):
        build_factor_moves(ValuationSnapshot(p0, ENG, e0, date=FRI), ValuationSnapshot(p1, ENG, e1, date=MON),
                           coord, engine_alive_t1=ENG, lifecycle_changed=False)


def test_tenor_detects_date_expiry_and_reraises_other_validation_errors():
    e1 = _env(100.0, FlatVolSurface(0.2), 0.03, 0.01, MON)
    expired = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, exercise_date=FRI)
    c = resolve_coordinate(expired, 100.0, expired, e1)          # valued on/after expiry
    assert c.tenor_t1 == 0.0 and Factor.VOL not in c.applicable and Factor.SPOT in c.applicable

    class Broken(EuropeanVanillaOption):
        def get_maturity(self, pricing_env=None):
            raise ValidationError("malformed maturity")

    with pytest.raises(ValidationError, match="malformed"):
        resolve_coordinate(_call(1.0), 100.0,
                           Broken(strike=100.0, option_type=OptionType.CALL, maturity=1.0), e1)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST test/test_pnlexplain_factor_diff.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `coordinate.py`**

```python
# quantark/pnlexplain/equity/coordinate.py
"""Per-product factor coordinate (spec §5.3)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet, Optional

from quantark.asset.equity.product.deltaone import Futures, SpotInstrument
from quantark.pnlexplain.base import MARKET_FACTORS, Factor
from quantark.pnlexplain.equity.fingerprints import MATURITY_FLOOR

_TERM_FACTORS = frozenset({Factor.VOL, Factor.RATE, Factor.DIVIDEND, Factor.BASIS})


@dataclass(frozen=True)
class FactorCoordinate:
    reference_strike: Optional[float]
    tenor_t1: Optional[float]
    applicable: FrozenSet[Factor]


def _positive(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f > 0.0 else None


def _tenor(product: Any, env: Any) -> Optional[float]:
    """Remaining tenor of the alive-at-t1 product, or None when it has no expiry.

    The one known non-error case in which ``get_maturity`` raises is a
    date-based product valued on or after its expiry; that is detected here
    explicitly and read as tenor 0. Every other ``ValidationError`` (a
    malformed product or environment) propagates: no invented expiry.
    """
    expiry = getattr(product, "exercise_date", None)
    if expiry is None:
        expiry = getattr(product, "maturity_date", None)
    if getattr(product, "maturity", None) is None and expiry is None:
        return None
    if expiry is not None and env is not None and env.valuation_date >= expiry:
        return 0.0
    return float(product.get_maturity(env))


def resolve_coordinate(product_t0: Any, spot_t0: float, product_alive_t1: Any, env_t1: Any
                       ) -> FactorCoordinate:
    if isinstance(product_t0, SpotInstrument):
        return FactorCoordinate(reference_strike=float(spot_t0), tenor_t1=None,
                                applicable=frozenset({Factor.SPOT, Factor.MODEL}))
    if isinstance(product_t0, Futures):
        applicable = frozenset({Factor.TIME, Factor.SPOT, Factor.RATE, Factor.DIVIDEND,
                                Factor.BASIS, Factor.MODEL})
        strike = float(spot_t0)
    else:
        applicable = frozenset(MARKET_FACTORS)
        strike = _positive(getattr(product_t0, "strike", None)) \
            or _positive(getattr(product_t0, "initial_price", None)) or float(spot_t0)
    tenor = _tenor(product_alive_t1, env_t1)
    if tenor is not None and tenor <= MATURITY_FLOOR:     # at or below the trackers' 1e-8 floor = expired
        applicable = applicable - _TERM_FACTORS
    return FactorCoordinate(reference_strike=strike, tenor_t1=tenor, applicable=applicable)
```

- [ ] **Step 4: Implement `factor_diff.py`**

```python
# quantark/pnlexplain/equity/factor_diff.py
"""FactorMoves: scalar moves at the product coordinate + change detection (spec §5.3)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional

from quantark.param.div import ContinuousDividendYield, NoDividend
from quantark.pnlexplain.base import Factor
from quantark.pnlexplain.equity.coordinate import FactorCoordinate
from quantark.pnlexplain.equity.fingerprints import calendars_equal
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot
from quantark.util.calendar import calculate_year_fraction
from quantark.util.exceptions import NumericalError, ValidationError
from quantark.util.numerical import is_close, safe_divide

POINT_TOL = 1e-12       # spec §5.2: numeric valuation points advance by calendar_days/365


def objects_equal(a: Any, b: Any) -> bool:
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 - a comparison that raises counts as changed
        return False


def _is_zero_yield(obj: Any) -> bool:
    if obj is None or isinstance(obj, NoDividend):
        return True
    return isinstance(obj, ContinuousDividendYield) and float(getattr(obj, "div_yield", 1.0)) == 0.0


def _yields_equal(a: Any, b: Any) -> bool:
    if _is_zero_yield(a) and _is_zero_yield(b):
        return True
    return objects_equal(a, b)


@dataclass(frozen=True)
class FactorMoves:
    coordinate: FactorCoordinate
    spot_t0: float
    spot_t1: float
    d_spot: float
    spot_return: float
    vol_t0: Optional[float]
    vol_t1: Optional[float]
    d_vol: Optional[float]
    rate_t0: Optional[float]
    rate_t1: Optional[float]
    d_rate: Optional[float]
    div_t0: Optional[float]
    div_t1: Optional[float]
    d_div: Optional[float]
    basis_t0: Optional[float]
    basis_t1: Optional[float]
    d_basis: Optional[float]
    calendar_days: int
    trading_days: Optional[int]
    year_fraction: float
    changed: FrozenSet[Factor]

    def display(self, factor: Factor) -> Dict[str, float]:
        """Display-unit moves for one factor; unavailable keys are omitted."""
        if factor is Factor.SPOT:
            return {"spot_return": self.spot_return}
        if factor is Factor.VOL:
            return {} if self.d_vol is None else {"vol_pts": self.d_vol * 100.0}
        if factor is Factor.RATE:
            return {} if self.d_rate is None else {"rate_pct": self.d_rate * 100.0}
        if factor is Factor.DIVIDEND:
            return {} if self.d_div is None else {"div_pct": self.d_div * 100.0}
        if factor is Factor.BASIS:
            return {} if self.d_basis is None else {"basis_pct": self.d_basis * 100.0}
        if factor is Factor.TIME:
            out = {"days": float(self.calendar_days)}
            if self.trading_days is not None:
                out["trading_days"] = float(self.trading_days)
            return out
        return {}


def validate_pair(snap0: ValuationSnapshot, snap1: ValuationSnapshot) -> None:
    if snap1.date <= snap0.date:
        raise ValidationError(f"date_t1 {snap1.date} must be after date_t0 {snap0.date}")
    e0, e1 = snap0.pricing_env, snap1.pricing_env
    if e0.day_count_convention != e1.day_count_convention or e0.bus_days_in_year != e1.bus_days_in_year:
        raise ValidationError("day count convention / bus_days_in_year differ between snapshots")
    if not calendars_equal(getattr(e0, "calendar", None), getattr(e1, "calendar", None)):
        raise ValidationError("calendars differ between snapshots (semantic comparison)")
    if snap0.quantity != snap1.quantity:
        raise ValidationError(
            "quantities differ between snapshots; a quantity change is a trade (use explain_position)"
        )
    if snap0.currency is not None and snap1.currency is not None and snap0.currency != snap1.currency:
        raise ValidationError("currency labels differ between snapshots")
    p0, p1 = snap0.point, snap1.point
    if (p0.date is None) != (p1.date is None):
        raise ValidationError("valuation points must share one representation (date or time)")
    if p0.date is None:
        days = (snap1.date - snap0.date).days
        if not is_close(p1.time - p0.time, days / 365.0, rel_tol=0.0, abs_tol=POINT_TOL):
            raise ValidationError("numeric valuation points must advance by calendar_days/365")


def _sample(env: Any, coordinate: FactorCoordinate, factor: Factor) -> Optional[float]:
    tenor = coordinate.tenor_t1
    if factor not in coordinate.applicable or tenor is None:
        return None
    if factor is Factor.VOL:
        return float(env.get_vol(coordinate.reference_strike, tenor))
    if factor is Factor.RATE:
        return float(env.get_rate(tenor))
    if factor is Factor.DIVIDEND:
        return float(env.get_div_yield(tenor))
    if factor is Factor.BASIS:
        return float(env.get_basis_yield(tenor))
    return None


def build_factor_moves(
    snap0: ValuationSnapshot,
    snap1: ValuationSnapshot,
    coordinate: FactorCoordinate,
    *,
    engine_alive_t1: Any,
    lifecycle_changed: bool,
) -> FactorMoves:
    e0, e1 = snap0.pricing_env, snap1.pricing_env
    s0, s1 = float(e0.spot), float(e1.spot)
    days = (snap1.date - snap0.date).days
    cal = getattr(e0, "calendar", None)
    trading = None
    if cal is not None and hasattr(cal, "count_business_days"):
        trading = int(cal.count_business_days(snap0.date, snap1.date, include_start=False, include_end=True))
    yf = float(calculate_year_fraction(snap0.date, snap1.date, e0.day_count_convention,
                                       e0.bus_days_in_year, calendar=cal))

    def pair(factor):
        a, b = _sample(e0, coordinate, factor), _sample(e1, coordinate, factor)
        return a, b, (None if a is None or b is None else b - a)

    vol = pair(Factor.VOL)
    rate = pair(Factor.RATE)
    div = pair(Factor.DIVIDEND)
    basis = pair(Factor.BASIS)
    for label, triple in (("vol", vol), ("rate", rate), ("dividend", div), ("basis", basis)):
        for v in triple:
            if v is not None and not math.isfinite(v):
                raise NumericalError(f"non-finite {label} sample at the product coordinate: {v!r}")

    changed = {Factor.TIME}
    app = coordinate.applicable
    if Factor.SPOT in app and s0 != s1:
        changed.add(Factor.SPOT)
    if Factor.VOL in app and not objects_equal(e0.vol_surface, e1.vol_surface):
        changed.add(Factor.VOL)
    if Factor.RATE in app and not objects_equal(e0.rate_curve, e1.rate_curve):
        changed.add(Factor.RATE)
    if Factor.DIVIDEND in app and not _yields_equal(e0.div_yield, e1.div_yield):
        changed.add(Factor.DIVIDEND)
    if Factor.BASIS in app and not (e0.basis_yield is None and e1.basis_yield is None) \
            and not objects_equal(e0.basis_yield, e1.basis_yield):
        changed.add(Factor.BASIS)
    if Factor.MODEL in app and engine_alive_t1 is not None and engine_alive_t1 is not snap0.engine:
        changed.add(Factor.MODEL)
    if lifecycle_changed:
        changed.add(Factor.LIFECYCLE_EVENT)

    return FactorMoves(
        coordinate=coordinate, spot_t0=s0, spot_t1=s1, d_spot=s1 - s0, spot_return=safe_divide(s1 - s0, s0),
        vol_t0=vol[0], vol_t1=vol[1], d_vol=vol[2],
        rate_t0=rate[0], rate_t1=rate[1], d_rate=rate[2],
        div_t0=div[0], div_t1=div[1], d_div=div[2],
        basis_t0=basis[0], basis_t1=basis[1], d_basis=basis[2],
        calendar_days=days, trading_days=trading, year_fraction=yf, changed=frozenset(changed),
    )
```

- [ ] **Step 5: Run tests**

Run: `PYTEST test/test_pnlexplain_factor_diff.py -q`
Expected: 6 passed. (`PricingEnvironment.get_basis_yield` returns `0.0` when no basis object is set, so `basis_t0` / `basis_t1` are `0.0` and BASIS is unchanged in these fixtures.)

- [ ] **Step 6: Commit**

```bash
git add quantark/pnlexplain/equity/coordinate.py quantark/pnlexplain/equity/factor_diff.py test/test_pnlexplain_factor_diff.py
git commit -m "feat(pnlexplain): factor coordinate resolver and FactorMoves with change detection" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_019fnLipZtAJAPAQJLLEXQMb"
```

---

### Task 5: Lifecycle transition, scenario cache, sequential waterfall, `explain()`

**Review amendment (plan review 2):** land this task as THREE commits so each piece is
reviewable on its own: (a) `lifecycle.py` + the transition tests (`-k "transition or event_row"`),
(b) `scenario.py` + the cache tests (`-k "scenario or time_pure or endpoints"`), (c) `waterfall.py` +
`explain.py` + the remaining tests. The code below is unchanged; only the commit boundaries move, and the
test module grows with each commit (a commit's test file must import only what that commit provides —
put the transition tests in `test/test_pnlexplain_lifecycle.py`, the cache tests in
`test/test_pnlexplain_scenario.py`, and the rest in `test/test_pnlexplain_waterfall.py`).

**Files:**
- Create: `quantark/pnlexplain/equity/lifecycle.py`, `quantark/pnlexplain/equity/scenario.py`, `quantark/pnlexplain/equity/waterfall.py`, `quantark/pnlexplain/equity/explain.py`
- Test: `test/test_pnlexplain_waterfall.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: `LifecycleTransition(product_alive_t1, engine_alive_t1, state_before, state_after, events=())` with `.changed`; `resolve_transition(snap0, snap1, transition, *, calendar_days) -> LifecycleTransition`; `event_row(cache, transition, level) -> ExplainRow`; `ScenarioCache(snap0, snap1, transition, moves, config)` with `.effective`, `.value_for(applied) -> ValueBreakdown`, `.all_market()`, `.value_t1()`, `.time_pure()`, `.bump_engine_t0`, `.moves`; `sequential_rows(cache, order, level) -> tuple[ExplainRow, ...]`; `explain(snapshot_t0, snapshot_t1, *, config=None, transition=None) -> PnLExplainResult`.

- [ ] **Step 1: Write the failing tests**

```python
# test/test_pnlexplain_waterfall.py
"""Spec tests 1 (waterfall part), 2 (orders, sticky), 3 (time-step equivalence, contract identity)."""
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.param.vol.vol_surface import BlackImpliedVolSurface
from quantark.pnlexplain import explain
from quantark.pnlexplain.base import ExplainMethod, Factor, MARKET_FACTORS, RowKind
from quantark.pnlexplain.config import PnLExplainConfig
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.greek_conventions import GreekConvention
from quantark.util.exceptions import ValidationError

FRI = datetime(2026, 6, 26)
MON = datetime(2026, 6, 29)
WF = PnLExplainConfig(methods=(ExplainMethod.WATERFALL,))
ENG = BlackScholesEngine()      # one engine object on both sides: MODEL is identity-detected


class SkewSurface(BlackImpliedVolSurface):
    """Linear skew in absolute strike: exposes sticky-strike vs sticky-moneyness."""
    is_smile = True

    def __init__(self, atm, slope):
        self.atm, self.slope = atm, slope

    def get_vol(self, strike, time_to_maturity, spot=None):
        return self.atm + self.slope * (float(strike) - 100.0)


def _env(spot, vol, rate, div, date):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=vol, rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div), valuation_date=date)


def _call(m=None, exercise_date=None):
    return EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=m,
                                 exercise_date=exercise_date)


def _pair(vol0=FlatVolSurface(0.20), vol1=FlatVolSurface(0.22), spot1=103.0, rate1=0.032, div1=0.012):
    e0 = _env(100.0, vol0, 0.03, 0.01, FRI)
    e1 = _env(spot1, vol1, rate1, div1, MON)
    s0 = ValuationSnapshot(_call(1.0), ENG, e0, date=FRI, quantity=2.0)
    s1 = ValuationSnapshot(_call(1.0 - 3 / 365), ENG, e1, date=MON, quantity=2.0)
    return s0, s1


def _tol(total):
    return 1e-10 * max(1.0, abs(total))


def test_taylor_and_shapley_are_gated_until_their_tasks():
    s0, s1 = _pair()
    with pytest.raises(NotImplementedError):
        explain(s0, s1)                                     # the default config requests Taylor
    with pytest.raises(NotImplementedError):
        explain(s0, s1, config=PnLExplainConfig(methods=(ExplainMethod.WATERFALL,), interaction="shapley"))


def test_waterfall_is_exact_and_time_row_is_time_pure():
    s0, s1 = _pair()
    res = explain(s0, s1, config=WF)
    eng = BlackScholesEngine()
    assert res.pv_t0.total == pytest.approx(2.0 * eng.price(s0.product, s0.pricing_env))
    assert res.pv_t1.total == pytest.approx(2.0 * eng.price(s1.product, s1.pricing_env))
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(res.total_pnl))
    rows = res.rows_for(ExplainMethod.WATERFALL, kind=RowKind.COMPONENT)
    assert [r.factor for r in rows[:7]] == list(MARKET_FACTORS)
    assert [r.step for r in rows[:7]] == list(range(1, 8))
    time_row = rows[0]
    expected_time = 2.0 * (eng.price(s1.product, s0.pricing_env) - eng.price(s0.product, s0.pricing_env))
    assert time_row.pnl == pytest.approx(expected_time, abs=1e-12)
    assert res.metadata["time_pure"] == pytest.approx(expected_time, abs=1e-12)
    assert time_row.moves == {"days": 3.0}
    basis_row = [r for r in rows if r.factor is Factor.BASIS][0]
    model_row = [r for r in rows if r.factor is Factor.MODEL][0]
    assert basis_row.pnl == 0.0 and model_row.pnl == 0.0
    assert basis_row.metadata["changed"] is False
    event = [r for r in res.rows if r.factor is Factor.LIFECYCLE_EVENT][0]
    assert event.method is ExplainMethod.SHARED and event.pnl == 0.0
    total = [r for r in res.rows if r.kind is RowKind.SUMMARY]
    assert len(total) == 1 and total[0].pnl == pytest.approx(res.total_pnl)
    assert res.unexplained is None


def test_reordered_waterfall_sums_exactly_but_time_row_differs():
    s0, s1 = _pair()
    order = (Factor.SPOT, Factor.VOL, Factor.RATE, Factor.DIVIDEND, Factor.BASIS, Factor.MODEL, Factor.TIME)
    res = explain(s0, s1, config=PnLExplainConfig(methods=(ExplainMethod.WATERFALL,), waterfall_order=order))
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(res.total_pnl))
    time_row = [r for r in res.rows if r.factor is Factor.TIME and r.method is ExplainMethod.WATERFALL][0]
    assert time_row.step == 7
    assert time_row.pnl != pytest.approx(res.metadata["time_pure"], abs=1e-9)


@pytest.mark.parametrize("order", [
    MARKET_FACTORS,
    (Factor.TIME, Factor.VOL, Factor.SPOT, Factor.RATE, Factor.DIVIDEND, Factor.BASIS, Factor.MODEL),
])
def test_sticky_moneyness_is_order_independent(order):
    s0, s1 = _pair(vol0=SkewSurface(0.20, 0.002), vol1=SkewSurface(0.21, 0.002))
    cfg = PnLExplainConfig(methods=(ExplainMethod.WATERFALL,), waterfall_order=order,
                           spot_convention=GreekConvention.STICKY_MONEYNESS)
    res = explain(s0, s1, config=cfg)
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(res.total_pnl))
    res_strike = explain(s0, s1, config=PnLExplainConfig(methods=(ExplainMethod.WATERFALL,), waterfall_order=order))
    assert res.total_pnl == pytest.approx(res_strike.total_pnl)
    spot_m = res.by_factor(ExplainMethod.WATERFALL)["spot"]
    spot_k = res_strike.by_factor(ExplainMethod.WATERFALL)["spot"]
    if order.index(Factor.SPOT) < order.index(Factor.VOL):
        # the convention decides how the t0 surface travels with spot: SPOT and VOL split differently
        assert spot_m != pytest.approx(spot_k, abs=1e-9)
    else:
        # once the t1 surface is applied, spot moves under it: the convention is moot by design
        assert spot_m == pytest.approx(spot_k, abs=1e-12)


def test_date_based_and_float_rolls_give_the_same_time_row():
    s0, s1 = _pair()
    dated0 = ValuationSnapshot(_call(exercise_date=FRI + timedelta(days=365)), ENG,
                               s0.pricing_env, date=FRI, quantity=2.0)
    dated1 = ValuationSnapshot(dated0.product, ENG, s1.pricing_env, date=MON, quantity=2.0)
    a = explain(s0, s1, config=WF).by_factor(ExplainMethod.WATERFALL)
    b = explain(dated0, dated1, config=WF).by_factor(ExplainMethod.WATERFALL)
    for k in a:
        assert a[k] == pytest.approx(b[k], abs=1e-9), k


def test_contract_replacement_and_unrolled_float_are_rejected():
    s0, s1 = _pair()
    bad = ValuationSnapshot(EuropeanVanillaOption(strike=105.0, option_type=OptionType.CALL, maturity=1.0 - 3 / 365),
                            ENG, s1.pricing_env, date=MON, quantity=2.0)
    with pytest.raises(ValidationError, match="contract replacement"):
        explain(s0, bad, config=WF)
    unrolled = ValuationSnapshot(_call(1.0), ENG, s1.pricing_env, date=MON, quantity=2.0)
    with pytest.raises(ValidationError):
        explain(s0, unrolled, config=WF)


def test_unchanged_factors_are_not_priced(monkeypatch):
    s0, s1 = _pair(vol1=FlatVolSurface(0.20), rate1=0.03, div1=0.01)   # only time + spot move
    calls = []
    real = BlackScholesEngine.price

    def counting(self, product, env, **kw):
        calls.append(env.valuation_date)
        return real(self, product, env, **kw)

    monkeypatch.setattr(BlackScholesEngine, "price", counting)
    res = explain(s0, s1, config=WF)
    # base, time, spot, and the t1 endpoint: four pricings, no more
    assert len(calls) == 4
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(res.total_pnl))
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST test/test_pnlexplain_waterfall.py -q`
Expected: FAIL with `ImportError: cannot import name 'explain' from 'quantark.pnlexplain'`

- [ ] **Step 3: Implement `lifecycle.py`**

```python
# quantark/pnlexplain/equity/lifecycle.py
"""LifecycleTransition: the alive-at-t1 contract and the event row (spec §8)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from quantark.pnlexplain.base import ExplainMethod, ExplainRow, Factor, RowKind
from quantark.pnlexplain.equity.fingerprints import check_contract_roll, lifecycle_fingerprint
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot, is_terminal
from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class LifecycleTransition:
    product_alive_t1: Any
    engine_alive_t1: Any
    state_before: Any
    state_after: Any
    events: Tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))

    @property
    def changed(self) -> bool:
        return lifecycle_fingerprint(self.state_before) != lifecycle_fingerprint(self.state_after)


def _event_date(event: Any) -> datetime:
    return pd.Timestamp(event.date).normalize().to_pydatetime()


def resolve_transition(
    snap0: ValuationSnapshot,
    snap1: ValuationSnapshot,
    transition: Optional[LifecycleTransition],
    *,
    calendar_days: int,
) -> LifecycleTransition:
    fp0 = lifecycle_fingerprint(snap0.lifecycle_state)
    fp1 = lifecycle_fingerprint(snap1.lifecycle_state)
    if transition is None:
        if fp0 != fp1:
            raise ValidationError(
                "lifecycle state changed between the snapshots: supply a LifecycleTransition "
                "with the alive-at-t1 product (guessing it would mislabel the event row)"
            )
        if not is_terminal(snap0.lifecycle_state):    # a terminal position has no contract to roll (spec §8)
            check_contract_roll(snap0.product, snap1.product, calendar_days)
        return LifecycleTransition(
            product_alive_t1=snap1.product, engine_alive_t1=snap1.engine,
            state_before=snap0.lifecycle_state, state_after=snap1.lifecycle_state, events=(),
        )
    if lifecycle_fingerprint(transition.state_before) != fp0:
        raise ValidationError("transition.state_before does not match snapshot_t0.lifecycle_state")
    if lifecycle_fingerprint(transition.state_after) != fp1:
        raise ValidationError("transition.state_after does not match snapshot_t1.lifecycle_state")
    terminal_t0 = is_terminal(snap0.lifecycle_state)
    if not terminal_t0:
        check_contract_roll(snap0.product, transition.product_alive_t1, calendar_days)
    if fp0 == fp1:
        if transition.events:
            raise ValidationError("transition carries events but the lifecycle state is unchanged")
        if snap1.engine is not transition.engine_alive_t1:
            raise ValidationError(
                "engine substitution without a lifecycle event is a MODEL change: "
                "set engine_alive_t1 to snapshot_t1.engine"
            )
        if not terminal_t0:
            check_contract_roll(transition.product_alive_t1, snap1.product, 0)
        return transition
    if not transition.events:
        raise ValidationError("lifecycle state changed but the transition carries no events")
    last = None
    for ev in transition.events:
        d = _event_date(ev)
        if not (snap0.date < d <= snap1.date):
            raise ValidationError(f"lifecycle event dated {d} is outside ({snap0.date}, {snap1.date}]")
        if last is not None and d < last:
            raise ValidationError("lifecycle events must be chronological")
        last = d
    return transition


def event_summary(transition: LifecycleTransition) -> Tuple[Dict[str, Any], ...]:
    return tuple(
        {
            "event_type": ev.event_type.value, "date": _event_date(ev).isoformat(),
            "payoff": float(getattr(ev, "payoff", 0.0)), "cashflow": float(getattr(ev, "cashflow", 0.0)),
            "terminates_position": bool(getattr(ev, "terminates_position", False)),
        }
        for ev in transition.events
    )


def event_row(cache: Any, transition: LifecycleTransition, level: str = "instrument") -> ExplainRow:
    pnl = cache.value_t1().total - cache.all_market().total
    return ExplainRow(
        factor=Factor.LIFECYCLE_EVENT, term=Factor.LIFECYCLE_EVENT.value,
        method=ExplainMethod.SHARED, kind=RowKind.COMPONENT, level=level, pnl=pnl,
        metadata={"changed": transition.changed, "events": event_summary(transition)},
    )
```

- [ ] **Step 4: Implement `scenario.py`**

```python
# quantark/pnlexplain/equity/scenario.py
"""ScenarioCache: state(S) for a SET of applied factors + memoised values (spec §6)."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, FrozenSet, Iterable, Optional, Tuple

from quantark.asset.equity.riskmeasures.greeks.bump_envs import resolve_bump_engine
from quantark.param.vol.sticky import shocked_surface
from quantark.pnlexplain.base import MARKET_FACTORS, Factor, ValueBreakdown
from quantark.pnlexplain.config import PnLExplainConfig
from quantark.pnlexplain.equity.factor_diff import FactorMoves
from quantark.pnlexplain.equity.lifecycle import LifecycleTransition
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot, value
from quantark.util.enum.greek_conventions import GreekConvention


class ScenarioCache:
    """All scenario values of one explain call come from this one object."""

    def __init__(self, snap0: ValuationSnapshot, snap1: ValuationSnapshot,
                 transition: LifecycleTransition, moves: FactorMoves, config: PnLExplainConfig):
        self.snap0, self.snap1, self.transition, self.moves, self.config = snap0, snap1, transition, moves, config
        self.effective: FrozenSet[Factor] = frozenset(
            f for f in MARKET_FACTORS if f in moves.changed and f in moves.coordinate.applicable
        )
        self.bump_engine_t0 = resolve_bump_engine(snap0.product, snap0.pricing_env, snap0.engine)
        alive = transition.engine_alive_t1
        self.bump_engine_alive = (
            self.bump_engine_t0 if alive is snap0.engine
            else resolve_bump_engine(transition.product_alive_t1, snap1.pricing_env, alive)
        )
        e1 = snap1.engine
        if e1 is alive:
            self.bump_engine_t1 = self.bump_engine_alive
        elif e1 is snap0.engine:
            self.bump_engine_t1 = self.bump_engine_t0
        else:
            self.bump_engine_t1 = resolve_bump_engine(snap1.product, snap1.pricing_env, e1)
        self._memo: Dict[FrozenSet[Factor], ValueBreakdown] = {}
        self._t1: Optional[ValueBreakdown] = None

    def normalize(self, applied: Iterable[Factor]) -> FrozenSet[Factor]:
        return frozenset(applied) & self.effective

    def build_state(self, applied: Iterable[Factor]) -> Tuple[Any, Any, Any, Any]:
        """(product, engine, env, valuation_point) for the state after applying `applied`."""
        s = self.normalize(applied)
        e0, e1 = self.snap0.pricing_env, self.snap1.pricing_env
        env = deepcopy(e0)
        product, engine, point = self.snap0.product, self.bump_engine_t0, self.snap0.point
        if Factor.TIME in s:
            env.valuation_date = self.snap1.date
            product, point = self.transition.product_alive_t1, self.snap1.point
        if Factor.SPOT in s:
            env.spot_quote = deepcopy(e1.spot_quote)
        if Factor.VOL in s:
            env.vol_surface = e1.vol_surface
        elif Factor.SPOT in s and self.config.spot_convention is GreekConvention.STICKY_MONEYNESS:
            env.vol_surface = shocked_surface(e0.vol_surface, float(e0.spot), float(e1.spot),
                                              GreekConvention.STICKY_MONEYNESS)
        if Factor.RATE in s:
            env.rate_curve = e1.rate_curve
        if Factor.DIVIDEND in s:
            env.div_yield = e1.div_yield
        if Factor.BASIS in s:
            env.basis_yield = e1.basis_yield
        if Factor.MODEL in s:
            engine = self.bump_engine_alive
        return product, engine, env, point

    def value_for(self, applied: Iterable[Factor]) -> ValueBreakdown:
        key = self.normalize(applied)
        if key not in self._memo:
            product, engine, env, point = self.build_state(key)
            self._memo[key] = value(self.snap0, engine=engine, product=product,
                                    pricing_env=env, valuation_point=point)
        return self._memo[key]

    def all_market(self) -> ValueBreakdown:
        return self.value_for(MARKET_FACTORS)

    def value_t1(self) -> ValueBreakdown:
        if self._t1 is None:
            self._t1 = value(self.snap1, engine=self.bump_engine_t1)
        return self._t1

    def time_pure(self) -> float:
        return self.value_for((Factor.TIME,)).total - self.value_for(()).total
```

- [ ] **Step 5: Implement `waterfall.py` (sequential only; Shapley in Task 6)**

```python
# quantark/pnlexplain/equity/waterfall.py
"""Sequential full-revaluation waterfall (spec §6)."""
from __future__ import annotations

from typing import Sequence, Tuple

from quantark.pnlexplain.base import ExplainMethod, ExplainRow, Factor, RowKind
from quantark.pnlexplain.equity.scenario import ScenarioCache


def _row(cache: ScenarioCache, factor: Factor, pnl: float, level: str, step) -> ExplainRow:
    """Build one waterfall row; never prices."""
    metadata = {
        "changed": factor in cache.effective,
        "applicable": factor in cache.moves.coordinate.applicable,
    }
    if factor is Factor.MODEL:
        metadata["engine"] = type(cache.bump_engine_alive).__name__
    return ExplainRow(
        factor=factor, term=factor.value, method=ExplainMethod.WATERFALL, kind=RowKind.COMPONENT,
        level=level, pnl=pnl, moves=cache.moves.display(factor), step=step, metadata=metadata,
    )


def sequential_rows(cache: ScenarioCache, order: Sequence[Factor], level: str = "instrument"
                    ) -> Tuple[ExplainRow, ...]:
    applied = set()
    previous = cache.value_for(()).total
    rows = []
    for step, factor in enumerate(order, start=1):
        applied.add(factor)
        if factor in cache.effective:
            current = cache.value_for(applied).total
            pnl, previous = current - previous, current
        else:
            pnl = 0.0
        rows.append(_row(cache, factor, pnl, level, step))
    return tuple(rows)
```

- [ ] **Step 6: Implement `explain.py` and export it**

```python
# quantark/pnlexplain/equity/explain.py
"""Two-snapshot explain orchestrator (spec §5.6)."""
from __future__ import annotations

from typing import List, Optional

from quantark.pnlexplain.base import MARKET_FACTORS, ExplainMethod, ExplainRow, PnLExplainResult, make_total_row
from quantark.pnlexplain.config import PnLExplainConfig
from quantark.pnlexplain.equity.coordinate import resolve_coordinate
from quantark.pnlexplain.equity.factor_diff import build_factor_moves, validate_pair
from quantark.pnlexplain.equity.lifecycle import LifecycleTransition, event_row, resolve_transition
from quantark.pnlexplain.equity.scenario import ScenarioCache
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot
from quantark.pnlexplain.equity.waterfall import sequential_rows
from quantark.util.exceptions import NumericalError
from quantark.util.numerical import is_close

LEVEL = "instrument"


def explain(
    snapshot_t0: ValuationSnapshot,
    snapshot_t1: ValuationSnapshot,
    *,
    config: Optional[PnLExplainConfig] = None,
    transition: Optional[LifecycleTransition] = None,
) -> PnLExplainResult:
    config = config if config is not None else PnLExplainConfig()
    validate_pair(snapshot_t0, snapshot_t1)
    days = (snapshot_t1.date - snapshot_t0.date).days
    transition = resolve_transition(snapshot_t0, snapshot_t1, transition, calendar_days=days)
    coordinate = resolve_coordinate(
        snapshot_t0.product, float(snapshot_t0.pricing_env.spot),
        transition.product_alive_t1, snapshot_t1.pricing_env,
    )
    moves = build_factor_moves(
        snapshot_t0, snapshot_t1, coordinate,
        engine_alive_t1=transition.engine_alive_t1, lifecycle_changed=transition.changed,
    )
    cache = ScenarioCache(snapshot_t0, snapshot_t1, transition, moves, config)
    pv_t0 = cache.value_for(())
    pv_alive_t1 = cache.all_market()
    pv_t1 = cache.value_t1()
    total_pnl = pv_t1.total - pv_t0.total

    # Explicit gates for modes built by later tasks (each is removed by the task that
    # implements the mode; a gate is never a fallback, it refuses).
    if ExplainMethod.TAYLOR in config.methods:
        raise NotImplementedError("the Taylor explainer lands in Task 8")
    if config.interaction == "shapley":
        raise NotImplementedError("interaction='shapley' lands in Task 6")
    rows: List[ExplainRow] = []
    if ExplainMethod.WATERFALL in config.methods:
        rows.extend(sequential_rows(cache, config.waterfall_order, LEVEL))
    rows.append(event_row(cache, transition, LEVEL))
    unexplained = None
    rows.append(make_total_row(LEVEL, total_pnl))

    metadata = {
        "time_pure": cache.time_pure(),
        "effective_factors": tuple(f.value for f in MARKET_FACTORS if f in cache.effective),
        "coordinate": (coordinate.reference_strike, coordinate.tenor_t1),
        "transition_changed": transition.changed,
        "interaction": config.interaction,
    }
    result = PnLExplainResult(
        date_t0=snapshot_t0.date, date_t1=snapshot_t1.date, pv_t0=pv_t0,
        pv_alive_t1=pv_alive_t1, pv_t1=pv_t1, total_pnl=total_pnl, moves=moves,
        rows=tuple(rows), unexplained=unexplained, metadata=metadata,
    )
    if ExplainMethod.WATERFALL in config.methods:
        gap = result.reconcile(ExplainMethod.WATERFALL)
        if not is_close(gap, 0.0, rel_tol=0.0, abs_tol=1e-10 * max(1.0, abs(total_pnl))):
            raise NumericalError(f"waterfall does not reconcile: gap {gap}")
    return result
```

Add to `quantark/pnlexplain/__init__.py`:

```python
from quantark.pnlexplain.config import PnLExplainConfig  # noqa: F401
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot, value  # noqa: F401
from quantark.pnlexplain.equity.coordinate import FactorCoordinate  # noqa: F401
from quantark.pnlexplain.equity.factor_diff import FactorMoves  # noqa: F401
from quantark.pnlexplain.equity.fingerprints import contract_fingerprint, lifecycle_fingerprint  # noqa: F401
from quantark.pnlexplain.equity.lifecycle import LifecycleTransition  # noqa: F401
from quantark.pnlexplain.equity.explain import explain  # noqa: F401
```
and extend `__all__` with `"PnLExplainConfig", "ValuationSnapshot", "value", "FactorCoordinate", "FactorMoves", "contract_fingerprint", "lifecycle_fingerprint", "LifecycleTransition", "explain"`.

- [ ] **Step 7: Run tests**

Run: `PYTEST test/test_pnlexplain_waterfall.py -q`
Expected: 8 passed (`test_unchanged_factors_are_not_priced` counts exactly 4 `price` calls: `()`, `{TIME}`, `{TIME,SPOT}`, and the t1 endpoint; the `all_market()` call reuses the `{TIME, SPOT}` memo key because VOL/RATE/DIVIDEND/BASIS/MODEL are not effective, and MODEL is unchanged because both snapshots hold the same `ENG` object).

- [ ] **Step 8: Commit**

```bash
git add quantark/pnlexplain/__init__.py quantark/pnlexplain/equity/lifecycle.py quantark/pnlexplain/equity/scenario.py quantark/pnlexplain/equity/waterfall.py quantark/pnlexplain/equity/explain.py test/test_pnlexplain_waterfall.py
git commit -m "feat(pnlexplain): scenario cache, sequential waterfall, lifecycle transition, explain()" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_019fnLipZtAJAPAQJLLEXQMb"
```

---

### Task 6: Shapley interaction

**Files:**
- Modify: `quantark/pnlexplain/equity/waterfall.py`
- Test: `test/test_pnlexplain_waterfall.py` (append)

**Interfaces:**
- Produces: `shapley_rows(cache, level) -> tuple[ExplainRow, ...]` (rows for all seven market factors in canonical order, `step=None`).

- [ ] **Step 1: Append the failing test**

```python
# append to test/test_pnlexplain_waterfall.py
import itertools


def test_shapley_matches_brute_force_average_and_sums_to_total():
    from quantark.pnlexplain.equity.coordinate import resolve_coordinate
    from quantark.pnlexplain.equity.factor_diff import build_factor_moves
    from quantark.pnlexplain.equity.lifecycle import resolve_transition
    from quantark.pnlexplain.equity.scenario import ScenarioCache
    from quantark.pnlexplain.equity.waterfall import sequential_rows

    s0, s1 = _pair(rate1=0.03, div1=0.01)          # TIME, SPOT, VOL change (3 effective factors)
    cfg = PnLExplainConfig(methods=(ExplainMethod.WATERFALL,), interaction="shapley")
    res = explain(s0, s1, config=cfg)
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(res.total_pnl))
    sh = {r.factor: r.pnl for r in res.rows_for(ExplainMethod.WATERFALL, kind=RowKind.COMPONENT)
          if r.method is ExplainMethod.WATERFALL}
    assert all(r.step is None for r in res.rows if r.method is ExplainMethod.WATERFALL)

    tr = resolve_transition(s0, s1, None, calendar_days=3)
    coord = resolve_coordinate(s0.product, 100.0, tr.product_alive_t1, s1.pricing_env)
    mv = build_factor_moves(s0, s1, coord, engine_alive_t1=tr.engine_alive_t1, lifecycle_changed=False)
    cache = ScenarioCache(s0, s1, tr, mv, cfg)
    eff = [f for f in MARKET_FACTORS if f in cache.effective]
    assert len(eff) == 3
    acc = {f: 0.0 for f in eff}
    perms = list(itertools.permutations(eff))
    for perm in perms:
        order = tuple(perm) + tuple(f for f in MARKET_FACTORS if f not in eff)
        for row in sequential_rows(cache, order):
            if row.factor in acc:
                acc[row.factor] += row.pnl / len(perms)
    for f in eff:
        assert sh[f] == pytest.approx(acc[f], abs=1e-9)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST test/test_pnlexplain_waterfall.py -q -k shapley`
Expected: FAIL with `ImportError: cannot import name 'shapley_rows'`

- [ ] **Step 3: Implement**

In `quantark/pnlexplain/equity/explain.py` replace the gate
`raise NotImplementedError("interaction='shapley' lands in Task 6")` with the real dispatch:

```python
    if ExplainMethod.WATERFALL in config.methods:
        if config.interaction == "sequential":
            rows.extend(sequential_rows(cache, config.waterfall_order, LEVEL))
        else:
            rows.extend(shapley_rows(cache, LEVEL))
```
(import `shapley_rows` next to `sequential_rows`), and in `test/test_pnlexplain_waterfall.py` delete the
shapley half of `test_taylor_and_shapley_are_gated_until_their_tasks` (keep the Taylor assertion).

```python
# append to quantark/pnlexplain/equity/waterfall.py
import itertools
from math import factorial

from quantark.pnlexplain.base import MARKET_FACTORS


def shapley_rows(cache: ScenarioCache, level: str = "instrument") -> Tuple[ExplainRow, ...]:
    """Order-independent allocation over the effective factors; 2^n pricings."""
    effective = [f for f in MARKET_FACTORS if f in cache.effective]
    n = len(effective)
    phi = {f: 0.0 for f in MARKET_FACTORS}
    for f in effective:
        others = [g for g in effective if g is not f]
        for r in range(len(others) + 1):
            for combo in itertools.combinations(others, r):
                s = frozenset(combo)
                weight = factorial(len(s)) * factorial(n - len(s) - 1) / factorial(n)
                phi[f] += weight * (cache.value_for(s | {f}).total - cache.value_for(s).total)
    return tuple(_row(cache, f, phi[f], level, None) for f in MARKET_FACTORS)
```

- [ ] **Step 4: Run tests**

Run: `PYTEST test/test_pnlexplain_waterfall.py -q`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add quantark/pnlexplain/equity/waterfall.py quantark/pnlexplain/equity/explain.py test/test_pnlexplain_waterfall.py
git commit -m "feat(pnlexplain): Shapley allocation over the effective market factors" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_019fnLipZtAJAPAQJLLEXQMb"
```

---

### Task 7: `GreeksCalculator.resolve_route` (pure helper)

**Files:**
- Modify: `quantark/asset/equity/riskmeasures/greeks_calculator.py` (after `calculate`, before `_normalize_greeks`)
- Test: `test/test_greeks_registry.py` (append)

**Interfaces:**
- Produces: `GreeksCalculator.resolve_route(product, greeks=None) -> str` returning `"analytical"` or `"numerical"`, exactly the route `calculate(method="auto")` takes.

- [ ] **Step 1: Append failing tests**

```python
# append to test/test_greeks_registry.py
def test_resolve_route_mirrors_auto_routing():
    from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
    from quantark.asset.equity.product.option.barrier_option import BarrierOption
    from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
    from quantark.util.enum import OptionType
    from quantark.util.enum.option_enums import BarrierType
    from quantark.util.exceptions import ValidationError

    calc = GreeksCalculator()
    vanilla = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    barrier = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0,
                            barrier_type=BarrierType.UP_OUT, maturity=1.0)
    assert calc.resolve_route(vanilla) == "analytical"
    assert calc.resolve_route(vanilla, ["delta", "gamma", "charm"]) == "analytical"
    assert calc.resolve_route(vanilla, ["delta", "vanna"]) == "numerical"
    assert calc.resolve_route(vanilla, ["theta_1td"]) == "numerical"
    assert calc.resolve_route(vanilla, ["veta"]) == "analytical"
    assert calc.resolve_route(barrier) == "numerical"
    with pytest.raises(ValidationError):
        calc.resolve_route(vanilla, ["bogus"])
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST test/test_greeks_registry.py -q -k resolve_route`
Expected: FAIL with `AttributeError: 'GreeksCalculator' object has no attribute 'resolve_route'`

- [ ] **Step 3: Implement (insert after the `calculate` method body)**

```python
    def resolve_route(
        self,
        product: BaseEquityProduct,
        greeks: Optional[Sequence[object]] = None,
    ) -> str:
        """The route ``calculate(method="auto")`` takes for this request.

        Pure (no pricing): "analytical" for a European vanilla when every
        requested name (default set when None) has a closed form under the
        auto-routing rule, "numerical" otherwise. Lets callers that must know
        the unit convention of the returned greeks (PnL explain) ask instead
        of re-deriving the rule.
        """
        requests = registry.normalize_greeks(greeks)
        if isinstance(product, EuropeanVanillaOption):
            if requests is None or all(analytical.supports_request(r) for r in requests):
                return "analytical"
        return "numerical"
```

- [ ] **Step 4: Run the riskmeasures safety net**

Run: `PYTEST test/test_greeks_registry.py test/test_point_greeks.py test/test_theta_suite_clocks.py test/test_greeks_key_order.py test/test_analytical_higher_order.py test/test_higher_order_greeks.py test/test_greeks_pricing_reuse.py -q`
Expected: all passed (no numeric path touched).

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/riskmeasures/greeks_calculator.py test/test_greeks_registry.py
git commit -m "feat(riskmeasures): GreeksCalculator.resolve_route pure routing helper" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_019fnLipZtAJAPAQJLLEXQMb"
```

---

### Task 8: Taylor explainer

**Review amendment (plan review 2):** implement Steps 3, 4 AND 6 before the first test run and
make ONE commit (the Step 5 "part 1" checkpoint cannot be green on its own: the committed test file
already contains the sub-row and clock tests). Run the complete `test/test_pnlexplain_taylor.py` and
`test/test_pnlexplain_waterfall.py` at that single checkpoint.

**Files:**
- Create: `quantark/pnlexplain/equity/taylor.py`
- Modify: `quantark/pnlexplain/equity/explain.py` (replace the Taylor gate with the branch + route metadata), `test/test_pnlexplain_waterfall.py` (delete the Taylor gate test)
- Test: `test/test_pnlexplain_taylor.py`

**Interfaces:**
- Consumes: `ScenarioCache` (Task 5), `resolve_stencil` / `resolved_subrows` / `TERM_FACTOR` (Task 2), `GreeksCalculator.resolve_route` (Task 7).
- Produces: `taylor_rows(cache, config, level="instrument") -> tuple[tuple[ExplainRow, ...], float, dict]` (rows, unexplained, metadata with `route`, `n_steps`, `clock`, `vega_scale`, `gap_scale`); `cash_greek(name, raw_position_greek, spot_t0, per_day_divisor) -> float`; `TERM_SPEC`; `TIME_GREEKS`.

Routing facts this task is built on (verified in the registry): `vanna`, `volga`, `delta_q`, `dividend_rho` and the theta sub-rows `r_theta` / `q_theta` / `convexity_theta` are numerical-only, so every built-in stencil routes a vanilla **numerically**; the analytical route is reached only with an explicit closed-form stencil such as `["delta", "gamma", "vega", "theta", "rho"]`. On the analytical route time greeks are per-day rates (scaled once by `calendar_days` under `exact_gap`); on the numerical route they are measured over the gap step (`n = 1`).

- [ ] **Step 1: Write the failing tests**

```python
# test/test_pnlexplain_taylor.py
"""Spec tests 1 (Taylor part), 4 (route/units/cash), 7 (clocks)."""
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.param import BumpConfig, EngineParams
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.pnlexplain import explain
from quantark.pnlexplain.base import ExplainMethod, Factor, RowKind
from quantark.pnlexplain.config import PnLExplainConfig
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

FRI = datetime(2026, 6, 26)
SAT = datetime(2026, 6, 27)
MON = datetime(2026, 6, 29)
Q = 5.0
K = 105.0                                                       # off the d2 = 0 point: vanna/volga != 0
CLOSED_FORM = ["delta", "gamma", "vega", "theta", "rho"]      # every name auto-routes analytical
FD = dict(rel=2e-2, abs=1e-6)                                   # spec §12 "FD tolerance"


def _env(spot, vol, rate, div, date, calendar=None):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate), div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=date, calendar=calendar)


def _snaps(ds=0.5, dvol=0.002, dr=0.0005, dq=0.0, d1=SAT, calendar=None, engine=None):
    days = (d1 - FRI).days
    e0 = _env(100.0, 0.20, 0.03, 0.01, FRI, calendar)
    e1 = _env(100.0 + ds, 0.20 + dvol, 0.03 + dr, 0.01 + dq, d1, calendar)
    eng = engine or BlackScholesEngine()
    s0 = ValuationSnapshot(EuropeanVanillaOption(strike=K, option_type=OptionType.CALL, maturity=1.0),
                           eng, e0, date=FRI, quantity=Q)
    s1 = ValuationSnapshot(EuropeanVanillaOption(strike=K, option_type=OptionType.CALL, maturity=1.0 - days / 365),
                           eng, e1, date=d1, quantity=Q)
    return s0, s1


def _rows(res, method=ExplainMethod.TAYLOR):
    return {r.term: r for r in res.rows if r.method is method}


def _analytical(s0, names):
    return GreeksCalculator().calculate_analytical_greeks(s0.product, s0.pricing_env, greeks=names)


def test_default_stencil_routes_numerical_and_matches_closed_form_to_fd_tolerance():
    s0, s1 = _snaps()
    g = _analytical(s0, ["delta", "gamma", "vega", "vanna", "volga", "rho", "dividend_rho"])
    res = explain(s0, s1)
    rows = _rows(res)
    assert res.metadata["route"] == "numerical"        # vanna/volga + theta sub-rows are numerical-only
    assert rows["delta"].pnl == pytest.approx(Q * g["delta"] * 0.5, **FD)
    assert rows["gamma"].pnl == pytest.approx(Q * 0.5 * g["gamma"] * 0.5 ** 2, **FD)
    assert rows["vega"].pnl == pytest.approx(Q * (g["vega"] / 0.01) * 0.002, **FD)
    assert rows["vanna"].pnl == pytest.approx(Q * g["vanna"] * 0.5 * 0.002, **FD)
    assert rows["volga"].pnl == pytest.approx(Q * 0.5 * g["volga"] * 0.002 ** 2, **FD)
    assert rows["rho"].pnl == pytest.approx(Q * (g["rho"] / 0.01) * 0.0005, **FD)
    assert rows["theta"].pnl == pytest.approx(res.metadata["time_pure"], abs=1e-12)
    assert rows["theta"].kind is RowKind.COMPONENT and rows["r_theta"].kind is RowKind.INFORMATIONAL
    assert rows["convexity_theta"].pnl == pytest.approx(
        rows["theta_contract"].pnl - rows["r_theta"].pnl - rows["q_theta"].pnl, abs=1e-12)
    assert rows["ledger_carry"].pnl == 0.0
    assert res.reconcile(ExplainMethod.TAYLOR) == pytest.approx(0.0, abs=1e-12)
    components = [r.pnl for r in res.rows_for(ExplainMethod.TAYLOR, kind=RowKind.COMPONENT)
                  if r.method is ExplainMethod.TAYLOR and r.factor is not Factor.UNEXPLAINED]
    assert res.unexplained == pytest.approx((res.pv_alive_t1.total - res.pv_t0.total) - sum(components), abs=1e-12)
    # The extended stencil adds the time-cross and third-order terms in (S, sigma, t); no stencil
    # carries a delta-rate cross term, so the comparison is made without a rate move.
    s0r, s1r = _snaps(dr=0.0)
    std_r = explain(s0r, s1r)
    ext = explain(s0r, s1r, config=PnLExplainConfig(stencil="extended"))
    assert abs(ext.unexplained) < abs(std_r.unexplained)
    assert "gamma_theta" in _rows(ext) and "speed" in _rows(ext)
    assert rows["vanna"].factor is Factor.VOL and rows["delta"].factor is Factor.SPOT
    assert rows["vanna"].moves == {"spot_return": pytest.approx(0.005), "vol_pts": pytest.approx(0.2)}


def test_closed_form_stencil_routes_analytical_and_is_exact():
    s0, s1 = _snaps()
    res = explain(s0, s1, config=PnLExplainConfig(stencil=CLOSED_FORM))
    rows = _rows(res)
    assert res.metadata["route"] == "analytical" and res.metadata["vega_scale"] == pytest.approx(0.01)
    g = _analytical(s0, ["delta", "gamma", "vega", "theta", "rho"])
    assert rows["delta"].pnl == pytest.approx(Q * g["delta"] * 0.5, rel=1e-12)
    assert rows["gamma"].pnl == pytest.approx(Q * 0.5 * g["gamma"] * 0.5 ** 2, rel=1e-12)
    assert rows["vega"].pnl == pytest.approx(Q * (g["vega"] / 0.01) * 0.002, rel=1e-12)
    assert rows["rho"].pnl == pytest.approx(Q * (g["rho"] / 0.01) * 0.0005, rel=1e-12)
    assert rows["theta"].pnl == pytest.approx(res.metadata["time_pure"], abs=1e-12)
    assert "r_theta" not in rows                      # sub-rows only when requested
    assert rows["theta_contract"].pnl == pytest.approx(Q * g["theta"] * 1, rel=1e-12)   # per day x 1-day gap
    assert res.reconcile(ExplainMethod.TAYLOR) == pytest.approx(0.0, abs=1e-12)


def test_numerical_route_with_custom_vol_bump_agrees_with_dv_dsigma():
    s0, s1 = _snaps()
    params = EngineParams(bump_config=BumpConfig(vol_bump=0.02))
    res = explain(s0, s1, config=PnLExplainConfig(greeks_method="numerical", params=params))
    assert res.metadata["route"] == "numerical" and res.metadata["vega_scale"] == pytest.approx(0.02)
    g = _analytical(s0, ["vega"])
    vega_row = _rows(res)["vega"]
    assert vega_row.pnl == pytest.approx(Q * (g["vega"] / 0.01) * 0.002, **FD)
    assert vega_row.greek == pytest.approx(Q * g["vega"] / 0.01, **FD)        # position-level dV/dsigma
    assert vega_row.cash_greek == pytest.approx(vega_row.greek * 0.01, rel=1e-12)
    assert vega_row.moves == {"vol_pts": pytest.approx(0.2)}


def test_cash_greek_columns_follow_the_desk_table():
    s0, s1 = _snaps()
    res = explain(s0, s1, config=PnLExplainConfig(stencil=CLOSED_FORM))
    rows = _rows(res)
    g = _analytical(s0, ["delta", "gamma", "vega", "rho"])
    S = 100.0
    assert rows["delta"].cash_greek == pytest.approx(Q * g["delta"] * S, rel=1e-12)
    assert rows["gamma"].cash_greek == pytest.approx(Q * g["gamma"] * S * S / 100.0, rel=1e-12)
    assert rows["vega"].cash_greek == pytest.approx(Q * g["vega"], rel=1e-12)            # per 1 vol pt
    assert rows["rho"].cash_greek == pytest.approx(Q * g["rho"], rel=1e-12)              # per 1%
    assert rows["theta"].cash_greek == pytest.approx(res.metadata["time_pure"] / 1.0)   # 1-day gap
    assert rows["delta"].moves == {"spot_return": pytest.approx(0.005)}
    vanna = _rows(explain(s0, s1))["vanna"]
    assert vanna.cash_greek == pytest.approx(Q * _analytical(s0, ["vanna"])["vanna"] * S * 0.01, **FD)


def test_bucketed_is_gated_until_task_13():
    s0, s1 = _snaps()
    with pytest.raises(NotImplementedError):
        explain(s0, s1, config=PnLExplainConfig(bucketed=True))


def test_clocks_exact_gap_versus_per_step():
    cal = create_calendar(CalendarType.CHINA_SSE)
    s0, s1 = _snaps(d1=MON, calendar=cal)
    exact = explain(s0, s1)
    r = _rows(exact)
    assert r["theta"].pnl == pytest.approx(exact.metadata["time_pure"], abs=1e-12)
    assert r["theta"].moves == {"days": 3.0, "trading_days": 1.0}
    assert r["theta"].greek == pytest.approx(exact.metadata["time_pure"] / 3.0)
    assert exact.metadata["n_steps"] == 1 and exact.metadata["gap_scale"] == 1.0

    calc = GreeksCalculator()
    theta_1d = calc.calculate_numerical_greeks(s0.product, s0.pricing_env, s0.engine, greeks=["theta_1d"])["theta_1d"]
    theta_1td = calc.calculate_numerical_greeks(s0.product, s0.pricing_env, s0.engine, greeks=["theta_1td"])["theta_1td"]
    per_1d = explain(s0, s1, config=PnLExplainConfig(time_term="per_step", clock="1d"))
    assert _rows(per_1d)["theta"].pnl == pytest.approx(Q * theta_1d * 3, rel=1e-9)
    assert per_1d.metadata["n_steps"] == 3
    per_1td = explain(s0, s1, config=PnLExplainConfig(time_term="per_step", clock="1td"))
    assert _rows(per_1td)["theta"].pnl == pytest.approx(Q * theta_1td * 1, rel=1e-9)
    assert per_1td.metadata["n_steps"] == 1
    # the calculator rolls a float-maturity product by one day per step (product.time_shift),
    # so three one-day steps are not the exact three-day revaluation: per_step misexplains the weekend
    assert _rows(per_1d)["theta"].pnl != pytest.approx(r["theta"].pnl, abs=1e-9)

    s0n, s1n = _snaps(d1=MON)                     # no calendar
    with pytest.raises(ValidationError):
        explain(s0n, s1n, config=PnLExplainConfig(time_term="per_step", clock="1td"))


def test_terminal_position_has_only_the_time_row():
    from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
    s0, s1 = _snaps(d1=MON)
    st = AutocallableLifecycleState()
    st.mark_ko(FRI - timedelta(days=10), cashflow=50.0, settlement_date=MON + timedelta(days=5))
    t0 = ValuationSnapshot(s0.product, s0.engine, s0.pricing_env, date=FRI, quantity=Q, lifecycle_state=st)
    t1 = ValuationSnapshot(s1.product, s1.engine, s1.pricing_env, date=MON, quantity=Q, lifecycle_state=st)
    res = explain(t0, t1)
    rows = _rows(res)
    assert res.pv_t0.contingent_mtm == 0.0
    assert rows["theta"].pnl == pytest.approx(res.metadata["time_pure"], abs=1e-12)
    assert all(r.pnl == 0.0 for k, r in rows.items() if k not in ("theta", "unexplained", "ledger_carry"))
    assert rows["delta"].greek is None
    assert res.reconcile(ExplainMethod.TAYLOR) == pytest.approx(0.0, abs=1e-12)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST test/test_pnlexplain_taylor.py -q`
Expected: every test FAILS with `NotImplementedError: the Taylor explainer lands in Task 8` (the Task 5 gate).

- [ ] **Step 3: Implement the Taylor core (`taylor.py`, part 1: route, units, cash table, non-time terms, theta component, residual)**

```python
# quantark/pnlexplain/equity/taylor.py
"""Greeks-based Taylor explain (spec §7)."""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

from quantark.asset.equity.riskmeasures.greeks.bump_envs import resolve_theta_bump_mode
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.pnlexplain.base import ExplainMethod, ExplainRow, Factor, RowKind
from quantark.pnlexplain.config import PnLExplainConfig, TERM_FACTOR, resolve_stencil, resolved_subrows
from quantark.pnlexplain.equity.scenario import ScenarioCache
from quantark.pnlexplain.equity.snapshot import is_terminal
from quantark.util.exceptions import ValidationError

# coefficient, exponents (spot, vol, rate, div, time)
TERM_SPEC: Dict[str, Tuple[float, Tuple[int, int, int, int, int]]] = {
    "delta": (1.0, (1, 0, 0, 0, 0)), "gamma": (0.5, (2, 0, 0, 0, 0)), "speed": (1.0 / 6.0, (3, 0, 0, 0, 0)),
    "vega": (1.0, (0, 1, 0, 0, 0)), "volga": (0.5, (0, 2, 0, 0, 0)), "vanna": (1.0, (1, 1, 0, 0, 0)),
    "zomma": (0.5, (2, 1, 0, 0, 0)),
    "theta": (1.0, (0, 0, 0, 0, 1)), "charm": (1.0, (1, 0, 0, 0, 1)), "color": (0.5, (2, 0, 0, 0, 1)),
    "vega_theta": (1.0, (0, 1, 0, 0, 1)),
    "rho": (1.0, (0, 0, 1, 0, 0)), "dividend_rho": (1.0, (0, 0, 0, 1, 0)),
    "dividend_volga": (0.5, (0, 0, 0, 2, 0)), "delta_q": (1.0, (1, 0, 0, 1, 0)),
}
VEGA_SCALED = ("vega", "vega_theta")
PER_PCT = ("rho", "dividend_rho")
TIME_GREEKS = ("theta", "r_theta", "q_theta", "convexity_theta", "gamma_theta", "charm", "color", "vega_theta")
SUBROW_NAMES = ("r_theta", "q_theta", "convexity_theta", "gamma_theta", "theta_contract", "ledger_carry")


def cash_greek(name: str, raw_position_greek: float, spot_t0: float, per_day_divisor: float) -> float:
    """Desk cash convention (spec §7.5): raw x S^a / 100^max(a-1,0) x 0.01^(b+c+d); time greeks per day."""
    if name in SUBROW_NAMES:
        return raw_position_greek / per_day_divisor
    _, (a, b, c, d, e) = TERM_SPEC[name]
    scale = spot_t0 ** a / (100.0 ** max(a - 1, 0)) * (0.01 ** (b + c + d))
    if e:
        scale /= per_day_divisor
    return raw_position_greek * scale


def display_moves(exponents, moves) -> Dict[str, float]:
    out: Dict[str, float] = {}
    a, b, c, d, e = exponents
    if a:
        out["spot_return"] = moves.spot_return
    if b and moves.d_vol is not None:
        out["vol_pts"] = moves.d_vol * 100.0
    if c and moves.d_rate is not None:
        out["rate_pct"] = moves.d_rate * 100.0
    if d and moves.d_div is not None:
        out["div_pct"] = moves.d_div * 100.0
    if e:
        out.update(moves.display(Factor.TIME))
    return out


def _resolve_steps(cache: ScenarioCache, config: PnLExplainConfig, bump) -> Tuple[int, str]:
    """(n, clock_label): exact_gap -> (1, "gap"); per_step is added in Step 6."""
    if config.time_term == "exact_gap":
        return 1, "gap"
    raise NotImplementedError("time_term='per_step' lands in Task 8 Step 6")


def _clock_suffix(config: PnLExplainConfig) -> str:
    return "" if config.time_term == "exact_gap" or config.clock is None else f"_{config.clock}"


def _calculator(params, bump, config: PnLExplainConfig, days: int) -> GreeksCalculator:
    if config.time_term == "exact_gap":
        bump = dataclasses.replace(bump, time_bump_days=days, time_bump_mode="calendar_days")
        params = dataclasses.replace(params, bump_config=bump)
    return GreeksCalculator(params=params, greeks_mode=config.greeks_mode)


def _row(level, name, factor, pnl, greek=None, cash=None, kind=RowKind.COMPONENT, extra=None, mv=None):
    return ExplainRow(factor=factor, term=name, method=ExplainMethod.TAYLOR, kind=kind, level=level,
                      pnl=pnl, moves=mv if mv is not None else {}, greek=greek, cash_greek=cash,
                      metadata=extra or {})


def _info(level, term, pnl, per_day, moves, formula, extra=None):
    return ExplainRow(factor=Factor.TIME, term=term, method=ExplainMethod.TAYLOR, kind=RowKind.INFORMATIONAL,
                      level=level, pnl=pnl, greek=pnl / per_day, cash_greek=pnl / per_day,
                      moves=moves.display(Factor.TIME), metadata={"formula": formula, **(extra or {})})


def _theta_subrows(level, greeks, q, n, days, per_day, gap_scale, config, subrows, time_pure,
                   theta_pnl, terminal, moves) -> List[ExplainRow]:
    """Step 3 version: theta_contract and ledger_carry only; Step 6 adds r/q/convexity/gamma_theta."""
    if terminal:
        return [_info(level, "theta_contract", 0.0, per_day, moves, "no contingent leg"),
                _info(level, "ledger_carry", time_pure, per_day, moves, "ledger_carry = time_pure - theta_contract")]
    theta_contract = q * greeks["theta"] * gap_scale
    return [_info(level, "theta_contract", theta_contract, per_day, moves,
                  "calculator theta over the calendar gap x quantity (analytical: per day x days)"),
            _info(level, "ledger_carry", time_pure - theta_contract, per_day, moves,
                  "ledger_carry = time_pure - theta_contract")]


def taylor_rows(cache: ScenarioCache, config: PnLExplainConfig, level: str = "instrument"
                ) -> Tuple[Tuple[ExplainRow, ...], float, Dict[str, Any]]:
    if config.bucketed:
        raise NotImplementedError("bucketed rows land in Task 13")     # removed in Task 13
    snap0, moves = cache.snap0, cache.moves
    q, days, S0 = snap0.quantity, moves.calendar_days, moves.spot_t0
    terms, subrows = resolve_stencil(config), resolved_subrows(config)
    alive_move = cache.all_market().total - cache.value_for(()).total
    time_pure = cache.time_pure()
    params = config.params if config.params is not None else snap0.engine.params
    bump = params.get_effective_bump_config()
    n, clock_label = _resolve_steps(cache, config, bump)
    per_day = float(days) if config.time_term == "exact_gap" else 1.0
    terminal = is_terminal(snap0.lifecycle_state)

    greeks: Dict[str, float] = {}
    route: Optional[str] = None
    vega_scale: Optional[float] = None
    gap_scale = 1.0
    if not terminal:
        calc = _calculator(params, bump, config, days)
        wanted = list(terms)
        if "theta" in terms and subrows:                      # sub-rows are numerical-only in the calculator
            wanted += ["r_theta", "q_theta"] + (["gamma_theta"] if "gamma_theta" in subrows else [])
        suffix = _clock_suffix(config)
        request = [f"{name}{suffix}" if name in TIME_GREEKS else name for name in wanted]
        route = config.greeks_method if config.greeks_method != "auto" \
            else calc.resolve_route(snap0.product, request)
        raw = calc.calculate(snap0.product, snap0.pricing_env, cache.bump_engine_t0, method=route,
                             greeks=request, theta_decomposition_mode=config.theta_decomposition_mode)
        greeks = {name: float(raw[f"{name}{suffix}" if name in TIME_GREEKS else name]) for name in wanted}
        vega_scale = 0.01 if route == "analytical" else float(bump.vol_bump)
        # analytical time greeks are per-day rates; numerical ones are gap-valued under exact_gap
        gap_scale = float(days) if (route == "analytical" and config.time_term == "exact_gap") else 1.0
    meta = {"route": route, "vega_scale": vega_scale, "n_steps": n, "clock": clock_label, "gap_scale": gap_scale}

    def derivative(name: str) -> float:
        g = greeks[name]
        if name in VEGA_SCALED:
            return g / vega_scale
        if name in PER_PCT:
            return g / 0.01
        return g

    raw_moves = (moves.d_spot, moves.d_vol, moves.d_rate, moves.d_div, float(n))
    rows: List[ExplainRow] = []
    explained = 0.0
    for name in terms:
        factor = TERM_FACTOR[name]
        coeff, exps = TERM_SPEC[name]
        applicable = factor in moves.coordinate.applicable and all(
            raw_moves[i] is not None for i, e in enumerate(exps[:4]) if e)
        if name == "theta":
            pnl, greek = time_pure, time_pure / per_day                     # exact_gap; per_step in Step 6
            extra = {"basis": "revaluation", "formula": "time_pure = V(alive@t1, t0 market) - V(t0)"}
            rows.append(_row(level, name, factor, pnl, greek=greek, cash=greek, extra=extra,
                             mv=moves.display(Factor.TIME)))
            explained += pnl
            rows.extend(_theta_subrows(level, greeks, q, n, days, per_day, gap_scale, config, subrows,
                                       time_pure, pnl, terminal, moves))
            continue
        if terminal or not applicable:
            rows.append(_row(level, name, factor, 0.0, extra={"applicable": applicable}))
            continue
        g_pos = q * derivative(name) * (gap_scale if exps[4] else 1.0)
        move_product = 1.0
        for i, e in enumerate(exps):
            if e:
                move_product *= raw_moves[i] ** e
        pnl = coeff * g_pos * move_product
        greek_disp = g_pos / per_day if exps[4] else g_pos
        rows.append(_row(level, name, factor, pnl, greek=greek_disp,
                         cash=cash_greek(name, g_pos, S0, per_day), mv=display_moves(exps, moves)))
        explained += pnl
    unexplained = alive_move - explained
    rows.append(_row(level, "unexplained", Factor.UNEXPLAINED, unexplained,
                     extra={"basis_and_model_effects_included": True}))
    return tuple(rows), unexplained, meta
```

- [ ] **Step 4: Wire the Taylor branch into `explain.py`**

Replace the gate `raise NotImplementedError("the Taylor explainer lands in Task 8")` with:

```python
    taylor_meta: Dict[str, Any] = {"route": None, "vega_scale": None, "n_steps": None, "clock": None, "gap_scale": None}
    if ExplainMethod.TAYLOR in config.methods:
        from quantark.pnlexplain.equity.taylor import taylor_rows
```
and replace the existing `unexplained = None` line that follows the event row with this final block
(the residual is assigned exactly once — initialised before the conditional, replaced by the Taylor
branch when requested, and never reset before `PnLExplainResult` is built):
```python
    unexplained: Optional[float] = None
    if ExplainMethod.TAYLOR in config.methods:
        trows, unexplained, taylor_meta = taylor_rows(cache, config, LEVEL)
        rows.extend(trows)
```
merge `**taylor_meta` into `metadata` (add `from typing import Any, Dict, Optional` to the imports).
Delete `test_taylor_and_shapley_are_gated_until_their_tasks` from `test/test_pnlexplain_waterfall.py`.

- [ ] **Step 5: (superseded — see the review amendment at the top of this task)**

Do NOT commit here. Continue with Step 6, then run the COMPLETE `test/test_pnlexplain_taylor.py` and
`test/test_pnlexplain_waterfall.py` and make the single commit of Step 7 (use the Step 7 message; the
command below is kept only for reference).

```bash
git add quantark/pnlexplain/equity/taylor.py quantark/pnlexplain/equity/explain.py test/test_pnlexplain_taylor.py test/test_pnlexplain_waterfall.py
git commit -m "feat(pnlexplain): Taylor explainer core (route, units, cash columns, non-time terms, residual)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_019fnLipZtAJAPAQJLLEXQMb"
```

- [ ] **Step 6: Add the theta sub-rows and the per-step clock (`taylor.py`, part 2)**

Replace `_resolve_steps` and `_theta_subrows` with:

```python
def _resolve_steps(cache: ScenarioCache, config: PnLExplainConfig, bump) -> Tuple[int, str]:
    """(n, clock_label) for the Taylor time terms."""
    mv = cache.moves
    if config.time_term == "exact_gap":
        return 1, "gap"
    clock = config.clock
    if clock is None:
        mode = resolve_theta_bump_mode(cache.snap0.pricing_env, bump.time_bump_mode)
        clock = "1td" if mode == "business_days" else "1d"
    if clock == "1td":
        if mv.trading_days is None:
            raise ValidationError("time_term='per_step' with the '1td' clock requires pricing_env.calendar")
        return mv.trading_days, "1td"
    return mv.calendar_days, "1d"


def _theta_subrows(level, greeks, q, n, days, per_day, gap_scale, config, subrows, time_pure,
                   theta_pnl, terminal, moves) -> List[ExplainRow]:
    if terminal:
        out = [_info(level, "ledger_carry", time_pure, per_day, moves, "ledger_carry = time_pure - theta_contract")]
        if config.time_term == "exact_gap":
            out.insert(0, _info(level, "theta_contract", 0.0, per_day, moves, "no contingent leg"))
        return out
    out: List[ExplainRow] = []
    exact_gap = config.time_term == "exact_gap"
    if exact_gap:
        theta_contract = q * greeks["theta"] * gap_scale
        out.append(_info(level, "theta_contract", theta_contract, per_day, moves,
                         "calculator theta over the calendar gap x quantity (analytical: per day x days)"))
        out.append(_info(level, "ledger_carry", time_pure - theta_contract, per_day, moves,
                         "ledger_carry = time_pure - theta_contract"))
        base = theta_contract
    else:
        base = theta_pnl
    if not subrows:
        return out
    # r_theta / q_theta are numerical-only: estimate mode returns per-day values (x days under
    # exact_gap), exact mode reprices over the configured step (x1 under exact_gap); per_step x n.
    estimate = config.theta_decomposition_mode == "estimate"
    step_scale = (float(days) if estimate else 1.0) if exact_gap else float(n)
    r_theta = q * greeks["r_theta"] * step_scale
    q_theta = q * greeks["q_theta"] * step_scale
    if "r_theta" in subrows:
        out.append(_info(level, "r_theta", r_theta, per_day, moves, "calculator r_theta x steps"))
    if "q_theta" in subrows:
        out.append(_info(level, "q_theta", q_theta, per_day, moves, "calculator q_theta x steps"))
    convexity = base - r_theta - q_theta
    if "convexity_theta" in subrows:
        out.append(_info(level, "convexity_theta", convexity, per_day, moves, "theta_contract - r_theta - q_theta"))
    if "gamma_theta" in subrows:
        gamma_theta = q * greeks["gamma_theta"] * (float(days) if exact_gap else float(n))
        out.append(_info(level, "gamma_theta", gamma_theta, per_day, moves,
                         "-1/2 sigma^2 S^2 Gamma per day x days", {"theta_residual": convexity - gamma_theta}))
    return out
```
and in `taylor_rows` replace the theta branch's first line with the per-step-aware version:

```python
        if name == "theta":
            if config.time_term == "exact_gap":
                pnl, greek = time_pure, time_pure / per_day
                extra = {"basis": "revaluation", "formula": "time_pure = V(alive@t1, t0 market) - V(t0)"}
            else:
                greek = q * greeks["theta"] if not terminal else 0.0
                pnl = greek * n
                extra = {"basis": "greek", "formula": "theta_per_step x n"}
```

- [ ] **Step 7: Run all Taylor tests and commit part 2**

Run: `PYTEST test/test_pnlexplain_taylor.py test/test_pnlexplain_waterfall.py -q`
Expected: all passed (7 Taylor + 8 waterfall).

```bash
git add quantark/pnlexplain/equity/taylor.py
git commit -m "feat(pnlexplain): theta sub-rows (contract/ledger carry/r/q/convexity/gamma_theta) and per-step clocks" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_019fnLipZtAJAPAQJLLEXQMb"
```
---

### Task 9: Lifecycle days and the manager accessor

**Files:**
- Modify: `quantark/asset/equity/lifecycle/manager.py` (add `pricing_products` after `register_positions`)
- Test: `test/test_pnlexplain_lifecycle_days.py` (the transition unit tests of Task 5 already own `test_pnlexplain_lifecycle.py`)

**Interfaces:**
- Consumes: `explain`, `LifecycleTransition` (Task 5), trackers `AutocallableLifecycleTracker` / `BarrierLifecycleTracker`.
- Produces: `PortfolioLifecycleManager.pricing_products(portfolio, date) -> Dict[str, Any]` (pure: tracked positions → `tracker.product_for_pricing(date, env)`, untracked → the position's current product).

- [ ] **Step 1: Write the failing tests**

```python
# test/test_pnlexplain_lifecycle_days.py
"""Spec test 5: KO day, coupon day, terminal carry, receivable payment, barrier KI substitution."""
from copy import deepcopy
from datetime import datetime, timedelta

import pandas as pd
import pytest

from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine
from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.lifecycle.autocallable import AutocallableLifecycleTracker
from quantark.asset.equity.lifecycle.barrier import BarrierLifecycleTracker
from quantark.asset.equity.lifecycle.events import LifecycleEventType
from quantark.asset.equity.lifecycle.manager import PortfolioLifecycleManager
from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
from quantark.asset.equity.param import PDEParams, QuadParams
from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.pnlexplain import explain
from quantark.pnlexplain.base import ExplainMethod, Factor, RowKind
from quantark.pnlexplain.config import PnLExplainConfig
from quantark.pnlexplain.equity.lifecycle import LifecycleTransition
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot
from quantark.portfolio import Portfolio, Position
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, OptionType
from quantark.util.enum.option_enums import BarrierType
from quantark.util.exceptions import ValidationError

START = datetime(2024, 1, 1)
Q = -3.0            # short the note, as a desk would be
KO_DATES = [0.25, 0.5, 0.75, 1.0]
WF = PnLExplainConfig(methods=(ExplainMethod.WATERFALL,))


def _env(date, spot):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=0.22),
        rate_curve=FlatRateCurve(rate=0.02), div_yield=ContinuousDividendYield(div_yield=0.03),
        valuation_date=date)


def _snowball():
    return SnowballOption(
        initial_price=100.0, strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=[105.0, 104.0, 103.0, 102.0], ko_rate=0.10,
            ko_observation_type=ObservationType.DISCRETE, ko_observation_dates=KO_DATES,
            ki_barrier=75.0, ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=KO_DATES, ki_continuous=False),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
        accrual_config=AccrualConfig(is_annualized=True),
        contract_multiplier=1.0, maturity=1.0, is_reverse=False)


def _find_event_day(tracker, event_type, spot_quiet, spot_trigger, max_days=400):
    """Advance the tracker with a quiet spot until the day BEFORE `event_type` would fire
    at `spot_trigger`; return (t0, t1)."""
    d = START
    for _ in range(max_days):
        d = d + timedelta(days=1)
        probe = deepcopy(tracker)
        ev = probe.observe(pd.Timestamp(d), probe.product_for_lifecycle(), _env(d, spot_trigger), spot_trigger)
        if any(e.event_type is event_type for e in ev):
            return d - timedelta(days=1), d
        tracker.observe(pd.Timestamp(d), tracker.product_for_lifecycle(), _env(d, spot_quiet), spot_quiet)
    raise AssertionError(f"no {event_type} within {max_days} days")


def _transition_pair(tracker, engine, t0, t1, spot0, spot1):
    e0, e1 = _env(t0, spot0), _env(t1, spot1)
    state_before = deepcopy(tracker.lifecycle if hasattr(tracker, "lifecycle") else tracker.state)
    product_t0 = tracker.product_for_pricing(pd.Timestamp(t0), e0)
    product_alive = tracker.product_for_pricing(pd.Timestamp(t1), e1)
    if hasattr(tracker, "lifecycle"):
        events = tracker.observe(pd.Timestamp(t1), tracker.product_for_lifecycle(), e1, spot1)
    else:
        events = tracker.observe(pd.Timestamp(t1), e1, spot1)
    state_after = deepcopy(tracker.lifecycle if hasattr(tracker, "lifecycle") else tracker.state)
    product_t1 = tracker.product_for_pricing(pd.Timestamp(t1), e1)
    engine_t1 = engine
    override = getattr(tracker, "engine_for_pricing", None)
    if override is not None and override() is not None:
        engine_t1 = override()
    # Float-schedule products keep their ledger in contract time: the tracker stamps a
    # numeric ValuationPoint on the state at every observation, and the snapshot carries it.
    s0 = ValuationSnapshot(product_t0, engine, e0, date=t0, quantity=Q, lifecycle_state=state_before,
                           valuation_point=getattr(state_before, "valuation_point", None))
    s1 = ValuationSnapshot(product_t1, engine_t1, e1, date=t1, quantity=Q, lifecycle_state=state_after,
                           valuation_point=getattr(state_after, "valuation_point", None))
    tr = LifecycleTransition(product_alive_t1=product_alive, engine_alive_t1=engine,
                             state_before=state_before, state_after=state_after, events=tuple(events))
    return s0, s1, tr


def _tol(x):
    return 1e-10 * max(1.0, abs(x))


def test_snowball_ko_day_event_row_and_exact_waterfall():
    tracker = AutocallableLifecycleTracker(product=_snowball(), quantity=Q, start_date=pd.Timestamp(START))
    t0, t1 = _find_event_day(tracker, LifecycleEventType.KNOCK_OUT, spot_quiet=100.0, spot_trigger=110.0)
    engine = SnowballPDESolver(PDEParams(accuracy="fast"))
    s0, s1, tr = _transition_pair(tracker, engine, t0, t1, 100.0, 110.0)
    assert tr.changed and tr.events[0].event_type is LifecycleEventType.KNOCK_OUT
    res = explain(s0, s1, transition=tr)
    assert res.pv_t1.contingent_mtm == 0.0
    assert res.pv_t1.paid_cash + res.pv_t1.pending_receivable_pv == pytest.approx(
        s1.lifecycle_state.ledger.cashflows[0].amount, rel=1e-12)
    event = [r for r in res.rows if r.factor is Factor.LIFECYCLE_EVENT][0]
    assert event.pnl == pytest.approx(res.pv_t1.total - res.pv_alive_t1.total, abs=_tol(res.total_pnl))
    assert event.metadata["events"][0]["event_type"] == "KO"
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(res.total_pnl))
    assert res.reconcile(ExplainMethod.TAYLOR) == pytest.approx(0.0, abs=_tol(res.total_pnl))
    with pytest.raises(ValidationError):           # state changed, no transition
        explain(s0, s1)


def test_phoenix_coupon_day_books_cash_and_reconciles():
    phoenix = create_standard_phoenix(initial_price=100.0, strike=100.0, maturity=1.0, ko_barrier=130.0,
                                      ki_barrier=70.0, coupon_barrier=90.0, coupon_rate=0.01,
                                      num_observations=12)
    tracker = AutocallableLifecycleTracker(product=phoenix, quantity=Q, start_date=pd.Timestamp(START))
    t0, t1 = _find_event_day(tracker, LifecycleEventType.COUPON, spot_quiet=80.0, spot_trigger=100.0)
    engine = PhoenixQuadEngine(QuadParams())
    s0, s1, tr = _transition_pair(tracker, engine, t0, t1, 80.0, 100.0)
    assert any(e.event_type is LifecycleEventType.COUPON for e in tr.events)
    res = explain(s0, s1, transition=tr, config=WF)
    assert res.pv_t1.paid_cash + res.pv_t1.pending_receivable_pv != 0.0
    assert res.pv_t0.paid_cash == 0.0
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(res.total_pnl))


def test_terminal_position_carries_receivable_then_pays_inside_time():
    st = AutocallableLifecycleState()
    st.mark_ko(START + timedelta(days=90), cashflow=-300.0, settlement_date=START + timedelta(days=95))
    prod = _snowball()
    eng = SnowballPDESolver(PDEParams(accuracy="fast"))
    d0, d1, d2 = START + timedelta(days=91), START + timedelta(days=92), START + timedelta(days=96)
    # the same (unrolled) product on every date: a terminal position is never priced, so the
    # contract-roll check is skipped for it (spec §8)
    s0 = ValuationSnapshot(prod, eng, _env(d0, 100.0), date=d0, quantity=Q, lifecycle_state=st)
    s1 = ValuationSnapshot(prod, eng, _env(d1, 101.0), date=d1, quantity=Q, lifecycle_state=st)
    res = explain(s0, s1, config=WF)
    by = res.by_factor(ExplainMethod.WATERFALL)
    assert res.pv_t0.contingent_mtm == 0.0 and res.pv_t0.pending_receivable_pv < 0.0
    assert by["spot"] == 0.0 and by["lifecycle_event"] == 0.0 and by["time"] != 0.0
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(res.total_pnl))
    s2 = ValuationSnapshot(prod, eng, _env(d2, 101.0), date=d2, quantity=Q, lifecycle_state=st)
    res2 = explain(s1, s2, config=WF)
    assert res2.pv_t1.pending_receivable_pv == 0.0 and res2.pv_t1.paid_cash == pytest.approx(-300.0)
    assert res2.by_factor(ExplainMethod.WATERFALL)["lifecycle_event"] == 0.0
    assert res2.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(res2.total_pnl))


def test_barrier_ki_substitution_lands_in_event_row_not_model():
    dip = BarrierOption(strike=100.0, option_type=OptionType.PUT, barrier=85.0,
                        barrier_type=BarrierType.DOWN_IN, maturity=1.0)
    tracker = BarrierLifecycleTracker(product=dip, quantity=Q, start_date=pd.Timestamp(START))
    t0 = START + timedelta(days=10)
    t1 = START + timedelta(days=11)
    tracker.observe(pd.Timestamp(t0), _env(t0, 95.0), 95.0)
    engine = BarrierAnalyticalEngine()
    s0, s1, tr = _transition_pair(tracker, engine, t0, t1, 95.0, 80.0)
    assert tr.changed and s1.lifecycle_state.knocked_in
    assert isinstance(s1.product, EuropeanVanillaOption) and isinstance(s1.engine, BlackScholesEngine)
    res = explain(s0, s1, transition=tr, config=WF)
    by = res.by_factor(ExplainMethod.WATERFALL)
    assert by["model"] == 0.0
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(res.total_pnl))
    # the wrong transition (engine swap without an event) is rejected
    bad = LifecycleTransition(product_alive_t1=tr.product_alive_t1, engine_alive_t1=engine,
                              state_before=tr.state_before, state_after=tr.state_before, events=())
    with pytest.raises(ValidationError):
        explain(s0, ValuationSnapshot(s1.product, s1.engine, s1.pricing_env, date=t1, quantity=Q,
                                      lifecycle_state=tr.state_before), transition=bad, config=WF)


def test_manager_pricing_products_is_pure():
    env = _env(START, 100.0)
    portfolio = Portfolio(portfolio_name="book", pricing_environments={"IDX": env})
    note = portfolio.add_position(product=_snowball(), quantity=Q, entry_price=0.0, underlying="IDX",
                                  engine=SnowballPDESolver(PDEParams(accuracy="fast")), entry_timestamp=START)
    vanilla = portfolio.add_position(
        product=EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0),
        quantity=1.0, entry_price=5.0, underlying="IDX", engine=BlackScholesEngine(), entry_timestamp=START)
    manager = PortfolioLifecycleManager(base_date=START)
    manager.register_positions(portfolio)
    day = START + timedelta(days=30)
    before = {pid: p.product for pid, p in portfolio.positions.items()}
    products = manager.pricing_products(portfolio, pd.Timestamp(day))
    assert set(products) == {note.position_id, vanilla.position_id}
    assert products[vanilla.position_id] is vanilla.product
    assert products[note.position_id] is not note.product
    assert products[note.position_id].maturity == pytest.approx(1.0 - 30 / 365)
    assert {pid: p.product for pid, p in portfolio.positions.items()} == before   # no mutation
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST test/test_pnlexplain_lifecycle_days.py -q`
Expected: `test_manager_pricing_products_is_pure` FAILS with `AttributeError: ... has no attribute 'pricing_products'`; the other four pass (they exercise Task 5–8 code). `PhoenixQuadEngine` accepts the float-based schedule `create_standard_phoenix` builds (its validation covers vol, dividend and time-step bounds only); `PhoenixQuadEngine(QuadParams())` is the pinned engine.

- [ ] **Step 3: Implement `pricing_products`**

```python
    def pricing_products(self, portfolio, date) -> Dict[str, Any]:
        """Per-position pricing product for ``date`` under the CURRENT lifecycle state.

        Pure accessor for the PnL explain recorder: tracked positions return
        ``tracker.product_for_pricing(date, env)`` (the alive contract rolled
        to ``date``; call it before ``process_day`` to get the pre-event
        contract), untracked positions return their current product. Nothing
        is mutated.
        """
        date = pd.Timestamp(date).normalize()
        out: Dict[str, Any] = {}
        for position_id, position in portfolio.positions.items():
            env = portfolio.pricing_environments[position.underlying]
            if position_id in self._autocallable:
                out[position_id] = self._autocallable[position_id].product_for_pricing(date, env)
            elif position_id in self._barrier:
                out[position_id] = self._barrier[position_id].product_for_pricing(date, env)
            else:
                out[position_id] = position.product
        return out
```
(`Dict`, `Any` are already imported in `manager.py`.)

- [ ] **Step 4: Run tests**

Run: `PYTEST test/test_pnlexplain_lifecycle_days.py test/test_backtest_lifecycle.py test/test_equity_lifecycle_trackers.py -q`
Expected: all passed (the two existing suites prove `process_day` is untouched).

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/lifecycle/manager.py test/test_pnlexplain_lifecycle_days.py
git commit -m "feat(pnlexplain): lifecycle-day tests (KO, coupon, terminal carry, KI substitution) + manager.pricing_products" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_019fnLipZtAJAPAQJLLEXQMb"
```

---

### Task 10: Trades, position and portfolio layers

**Review amendment (plan review 2):** the test module must import only what exists at each
commit. Put the `ExplainTrade` tests in `test/test_pnlexplain_trades.py` (commit 1 = `trades.py` + that
file) and everything else in `test/test_pnlexplain_portfolio.py` (commit 2 = `portfolio.py` +
`quantark/pnlexplain/__init__.py` exports + that file). Never run a half-implemented module under `-k`:
a collection-time `ImportError` cannot be deselected.

**Files:**
- Create: `quantark/pnlexplain/equity/trades.py`, `quantark/pnlexplain/equity/portfolio.py`
- Modify: `quantark/pnlexplain/__init__.py` (exports — staged in the SECOND commit of this task, once `explain_portfolio` exists; `"coordinate"` is already in the explain metadata since Task 5)
- Test: `test/test_pnlexplain_portfolio.py`

**Interfaces:**
- Produces: `ExplainTrade(position_id, quantity, price, transaction_cost=0.0, timestamp=None, kind="adjust", instrument_type="", metadata={})`, `ExplainTrade.from_contracts(position_id, contracts, price, multiplier, **kw)`, `ExplainTrade.cash` property (= −quantity·price); `PositionSnapshot(position_id, underlying, snapshot, tombstone=False)`; `QuotedLegSnapshot(position_id, underlying, units, price, spot, date, tombstone=False)`; `BookSnapshot(date, positions, environments, quoted_legs={}, currency=None)` + `BookSnapshot.from_portfolio(portfolio, date, *, tombstones=None, currency=None)` (no `lifecycle_manager` argument, by design — spec §9: a book snapshot is a pure read of the products the portfolio holds on that date; the alive-at-t1 products come from `PortfolioLifecycleManager.pricing_products` and are consumed by the recorders of Tasks 11/12, which build `LifecycleTransition`s, not by the book); `PositionExplainResult`; `PortfolioExplainResult` with `reconcile(method)`, `by_underlying()`, `to_frame()`; `explain_position(pos_t0, pos_t1, *, trades=(), transition=None, config=None)`; `explain_quoted_leg(leg_t0, leg_t1, *, trades=())`; `explain_portfolio(book_t0, book_t1, *, trades=(), transaction_costs=0.0, transitions={}, config=None)`.

- [ ] **Step 1: Write the failing tests**

```python
# test/test_pnlexplain_portfolio.py
"""Spec test 5b: the §9 case table."""
from datetime import datetime, timedelta
from types import MappingProxyType

import pytest

from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.pnlexplain import explain
from quantark.pnlexplain.base import ExplainMethod, Factor, RowKind
from quantark.pnlexplain.equity.portfolio import (
    BookSnapshot, PositionSnapshot, QuotedLegSnapshot, explain_portfolio, explain_position, explain_quoted_leg,
)
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot, value
from quantark.pnlexplain.equity.trades import ExplainTrade
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

T0 = datetime(2026, 6, 26)
T1 = datetime(2026, 6, 29)
ENG = BlackScholesEngine()


def _env(spot, date, vol=0.2):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=vol),
                              rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.01),
                              valuation_date=date)


E0, E1 = _env(100.0, T0), _env(103.0, T1, vol=0.22)


def _call(strike=100.0, days=0):
    return EuropeanVanillaOption(strike=strike, option_type=OptionType.CALL, maturity=1.0 - days / 365)


def _pos(pid, product_t0, q, env=E0, date=T0, state=None, tombstone=False, product_t1=None):
    prod = product_t0 if date == T0 else (product_t1 or _call(product_t0.strike, days=3))
    snap = ValuationSnapshot(prod, ENG, env, date=date, quantity=q, lifecycle_state=state)
    return PositionSnapshot(position_id=pid, underlying="IDX", snapshot=snap, tombstone=tombstone)


def _unit_t1(product_t1):
    return ENG.price(product_t1, E1)


def _tol(x):
    return 1e-10 * max(1.0, abs(x))


def test_unchanged_position_promotes_instrument_rows():
    p0 = _pos("a", _call(), 2.0)
    p1 = _pos("a", _call(), 2.0, env=E1, date=T1)
    res = explain_position(p0, p1)
    inst = explain(p0.snapshot, p1.snapshot)
    assert res.instrument.total_pnl == pytest.approx(inst.total_pnl)
    assert all(r.level == "position" for r in res.rows)
    assert sum(1 for r in res.rows if r.kind is RowKind.SUMMARY) == 1
    assert res.total_pnl == pytest.approx(inst.total_pnl)
    assert not res.trade_rows


def test_quantity_change_with_trade_reconciles():
    p0 = _pos("h", _call(), 2.0)
    p1 = _pos("h", _call(), 3.0, env=E1, date=T1)
    u1 = _unit_t1(p1.snapshot.product)
    trade = ExplainTrade(position_id="h", quantity=1.0, price=u1 - 0.1, kind="adjust", timestamp=T1)
    res = explain_position(p0, p1, trades=[trade])
    assert len(res.trade_rows) == 1 and res.trade_rows[0].pnl == pytest.approx(0.1)
    expected = value(p1.snapshot).total - value(p0.snapshot).total + trade.cash
    assert res.total_pnl == pytest.approx(expected)
    for m in (ExplainMethod.WATERFALL, ExplainMethod.TAYLOR):
        assert res.reconcile(m) == pytest.approx(0.0, abs=_tol(expected))
    with pytest.raises(ValidationError):            # trades do not reconcile
        explain_position(p0, p1, trades=[ExplainTrade("h", 2.0, u1)])


def test_opened_today_and_closed_by_trading_and_terminal_tombstone():
    p1 = _pos("n", _call(), 4.0, env=E1, date=T1)
    u1 = _unit_t1(p1.snapshot.product)
    new = explain_position(None, p1, trades=[ExplainTrade("n", 4.0, u1 + 0.05, kind="open")])
    assert new.instrument is None
    assert new.total_pnl == pytest.approx(4.0 * u1 - 4.0 * (u1 + 0.05))
    with pytest.raises(ValidationError):
        explain_position(None, p1)

    p0 = _pos("c", _call(), 4.0)
    tomb = _pos("c", _call(), 4.0, env=E1, date=T1, tombstone=True)
    closed = explain_position(p0, tomb, trades=[ExplainTrade("c", -4.0, u1 - 0.2, kind="close")])
    v0 = value(p0.snapshot).total
    assert closed.total_pnl == pytest.approx(-v0 + 4.0 * (u1 - 0.2))
    assert closed.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(v0))
    assert closed.trade_rows[0].pnl == pytest.approx(-4.0 * 0.2)
    with pytest.raises(ValidationError):            # non-terminal tombstone without trades
        explain_position(p0, tomb)

    st = AutocallableLifecycleState()
    st.mark_ko(T1, cashflow=12.0)
    from quantark.pnlexplain.equity.lifecycle import LifecycleTransition
    before = AutocallableLifecycleState()
    t0 = _pos("k", _call(), 4.0, state=before)
    t1 = _pos("k", _call(), 4.0, env=E1, date=T1, state=st, tombstone=True)
    from quantark.asset.equity.lifecycle.events import LifecycleEvent, LifecycleEventType
    ev = LifecycleEvent(event_type=LifecycleEventType.KNOCK_OUT, date=T1, spot=103.0, cashflow=12.0,
                        terminates_position=True)
    tr = LifecycleTransition(product_alive_t1=t1.snapshot.product, engine_alive_t1=ENG,
                             state_before=before, state_after=st, events=(ev,))
    term = explain_position(t0, t1, transition=tr)
    assert term.total_pnl == pytest.approx(12.0 - value(t0.snapshot).total)
    with pytest.raises(ValidationError):            # trades on a terminal tombstone
        explain_position(t0, t1, transition=tr, trades=[ExplainTrade("k", -4.0, 1.0)])


def test_quoted_leg_rolls_with_contract_specific_ids():
    a0 = QuotedLegSnapshot("hedge:IF2401", "IDX", units=300.0, price=100.4, spot=100.0, date=T0)
    a1 = QuotedLegSnapshot("hedge:IF2401", "IDX", units=300.0, price=103.3, spot=103.0, date=T1, tombstone=True)
    b1 = QuotedLegSnapshot("hedge:IF2402", "IDX", units=300.0, price=104.0, spot=103.0, date=T1)
    close = ExplainTrade.from_contracts("hedge:IF2401", contracts=-1.0, price=103.3, multiplier=300.0, kind="roll_close")
    open_ = ExplainTrade.from_contracts("hedge:IF2402", contracts=1.0, price=104.0, multiplier=300.0, kind="roll_open")
    assert close.quantity == -300.0 and close.cash == pytest.approx(300.0 * 103.3)
    old = explain_quoted_leg(a0, a1, trades=[close])
    assert old.total_pnl == pytest.approx(300.0 * (103.3 - 100.4))
    rows = {r.term: r for r in old.rows if r.kind is RowKind.COMPONENT}
    assert rows["spot"].pnl == pytest.approx(300.0 * 3.0)
    assert rows["basis"].pnl == pytest.approx(300.0 * (2.9 - 3.0))
    assert rows["trade:roll_close"].pnl == pytest.approx(0.0)
    new = explain_quoted_leg(None, b1, trades=[open_])
    assert new.total_pnl == pytest.approx(0.0)


def test_portfolio_aggregation_costs_and_reconciliation():
    pos0 = {"a": _pos("a", _call(100.0), 2.0), "b": _pos("b", _call(110.0), -1.0), "h": _pos("h", _call(), 2.0)}
    pos1 = {"a": _pos("a", _call(100.0), 2.0, env=E1, date=T1),
            "b": _pos("b", _call(110.0), -1.0, env=E1, date=T1),
            "h": _pos("h", _call(), 3.0, env=E1, date=T1)}
    b0 = BookSnapshot(date=T0, positions=pos0, environments={"IDX": E0})
    b1 = BookSnapshot(date=T1, positions=pos1, environments={"IDX": E1})
    u1 = _unit_t1(pos1["h"].snapshot.product)
    trades = [ExplainTrade("h", 1.0, u1, transaction_cost=0.5, timestamp=T1)]
    res = explain_portfolio(b0, b1, trades=trades, transaction_costs=0.25)
    cost = [r for r in res.rows if r.factor is Factor.TRANSACTION_COST][0]
    assert cost.pnl == pytest.approx(-0.75) and cost.method is ExplainMethod.SHARED
    expected = sum(res.positions[k].total_pnl for k in res.positions) - 0.75
    assert res.total_pnl == pytest.approx(expected)
    for m in (ExplainMethod.WATERFALL, ExplainMethod.TAYLOR):
        assert res.reconcile(m) == pytest.approx(0.0, abs=_tol(expected))
        assert res.metadata["reconciliation"][m.value]["ok"] is True
    # two strikes under one underlying: aggregated display columns are null
    book_rows = {(r.factor, r.term): r for r in res.rows if r.method is ExplainMethod.TAYLOR}
    assert book_rows[(Factor.VOL, "vega")].greek is None and book_rows[(Factor.VOL, "vega")].moves == {}
    per_u = res.by_underlying()["IDX"]
    assert all(r.level == "portfolio" for r in per_u)
    frame = res.to_frame()
    assert set(frame["level"]) == {"position", "portfolio"}
    assert {"a", "b", "h", "portfolio"} <= set(frame["position_id"])
    # single-position book keeps the display columns
    single = explain_portfolio(BookSnapshot(T0, {"a": pos0["a"]}, {"IDX": E0}),
                               BookSnapshot(T1, {"a": pos1["a"]}, {"IDX": E1}))
    srow = [r for r in single.rows if r.method is ExplainMethod.TAYLOR and r.term == "vega"][0]
    assert srow.greek is not None and "vol_pts" in srow.moves


def test_from_portfolio_snapshots_rolled_products_without_mutation():
    from quantark.asset.equity.engine.pde import SnowballPDESolver
    from quantark.asset.equity.lifecycle.manager import PortfolioLifecycleManager
    from quantark.asset.equity.param import PDEParams
    from quantark.portfolio import Portfolio
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from test_pnlexplain_lifecycle_days import _snowball, _env as _lc_env

    start = datetime(2024, 1, 1)
    day = datetime(2024, 1, 31)
    portfolio = Portfolio(portfolio_name="book", pricing_environments={"IDX": _lc_env(start, 100.0)})
    note = portfolio.add_position(product=_snowball(), quantity=-3.0, entry_price=0.0, underlying="IDX",
                                  engine=SnowballPDESolver(PDEParams(accuracy="fast")), entry_timestamp=start)
    manager = PortfolioLifecycleManager(base_date=start)
    manager.register_positions(portfolio)
    portfolio.pricing_environments["IDX"] = _lc_env(day, 101.0)
    manager.process_day(portfolio, day_index=30, day_date=day)          # substitutes the rolled product
    before = {pid: (p.product, p.engine, p.quantity) for pid, p in portfolio.positions.items()}
    book = BookSnapshot.from_portfolio(portfolio, day)
    snap = book.positions[note.position_id].snapshot
    assert snap.product is portfolio.positions[note.position_id].product     # the rolled product
    assert snap.product.maturity == pytest.approx(1.0 - 30 / 365)
    assert snap.pricing_env is not portfolio.pricing_environments["IDX"]     # a copy, same date
    assert snap.pricing_env.valuation_date == day and snap.lifecycle_state is not note.lifecycle_state
    assert {pid: (p.product, p.engine, p.quantity) for pid, p in portfolio.positions.items()} == before
    with pytest.raises(ValidationError):                                      # env dated differently
        BookSnapshot.from_portfolio(portfolio, day + timedelta(days=1))


def test_portfolio_errors():
    pos0 = {"a": _pos("a", _call(), 2.0)}
    b0 = BookSnapshot(date=T0, positions=pos0, environments={"IDX": E0})
    with pytest.raises(ValidationError):            # absent at t1, no tombstone
        explain_portfolio(b0, BookSnapshot(date=T1, positions={}, environments={"IDX": E1}))
    with pytest.raises(ValidationError):            # new at t1 without trades
        explain_portfolio(BookSnapshot(date=T0, positions={}, environments={"IDX": E0}),
                          BookSnapshot(date=T1, positions={"a": _pos("a", _call(), 2.0, env=E1, date=T1)},
                                       environments={"IDX": E1}))
    with pytest.raises(ValidationError):            # currency mismatch
        explain_portfolio(BookSnapshot(date=T0, positions=pos0, environments={"IDX": E0}, currency="CNY"),
                          BookSnapshot(date=T1, positions={"a": _pos("a", _call(), 2.0, env=E1, date=T1)},
                                       environments={"IDX": E1}, currency="USD"))
    with pytest.raises(ValidationError):            # invalid trade fields
        ExplainTrade("a", 0.0, 1.0)
    with pytest.raises(ValidationError):
        ExplainTrade("a", 1.0, 1.0, transaction_cost=-1.0)
    with pytest.raises(ValidationError):
        ExplainTrade("a", 1.0, 1.0, kind="swap")
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST test/test_pnlexplain_portfolio.py -q`
Expected: FAIL with `ModuleNotFoundError: ... trades`

- [ ] **Step 3: Implement `trades.py`**

```python
# quantark/pnlexplain/equity/trades.py
"""Normalised trade schema (spec §9)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping, Optional

from quantark.util.exceptions import ValidationError

TRADE_KINDS = ("open", "adjust", "close", "roll_close", "roll_open")


@dataclass(frozen=True)
class ExplainTrade:
    position_id: str
    quantity: float                      # signed position UNITS: buy > 0, sell < 0
    price: float                         # per unit, same convention as engine.price
    transaction_cost: float = 0.0
    timestamp: Optional[datetime] = None
    kind: str = "adjust"
    instrument_type: str = ""
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.position_id:
            raise ValidationError("trade requires a position_id")
        q, p, c = float(self.quantity), float(self.price), float(self.transaction_cost)
        if not math.isfinite(q) or q == 0.0:
            raise ValidationError(f"trade quantity must be non-zero and finite, got {self.quantity}")
        if not math.isfinite(p):
            raise ValidationError(f"trade price must be finite, got {self.price}")
        if not math.isfinite(c) or c < 0.0:
            raise ValidationError(f"transaction_cost must be >= 0, got {self.transaction_cost}")
        if self.kind not in TRADE_KINDS:
            raise ValidationError(f"trade kind must be one of {TRADE_KINDS}, got {self.kind!r}")
        object.__setattr__(self, "quantity", q)
        object.__setattr__(self, "price", p)
        object.__setattr__(self, "transaction_cost", c)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def cash(self) -> float:
        """Cash flow of the trade: buying costs cash."""
        return -self.quantity * self.price

    @classmethod
    def from_contracts(cls, position_id: str, contracts: float, price: float, multiplier: float,
                       **kw: Any) -> "ExplainTrade":
        if not math.isfinite(float(multiplier)) or float(multiplier) <= 0.0:
            raise ValidationError(f"multiplier must be positive, got {multiplier}")
        return cls(position_id=position_id, quantity=float(contracts) * float(multiplier),
                   price=float(price), **kw)
```

- [ ] **Step 4: Implement `portfolio.py`**

```python
# quantark/pnlexplain/equity/portfolio.py
"""Position and portfolio layers (spec §9)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from quantark.pnlexplain.base import (
    ExplainMethod, ExplainRow, Factor, PnLExplainResult, RowKind, component_sum, make_total_row,
    rows_to_frame,
)
from quantark.pnlexplain.config import PnLExplainConfig
from quantark.pnlexplain.equity.explain import explain
from quantark.pnlexplain.equity.lifecycle import LifecycleTransition
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot, is_terminal, value
from quantark.pnlexplain.equity.trades import ExplainTrade
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close

_EMPTY: Mapping = MappingProxyType({})


@dataclass(frozen=True)
class PositionSnapshot:
    position_id: str
    underlying: str
    snapshot: ValuationSnapshot
    tombstone: bool = False


@dataclass(frozen=True)
class QuotedLegSnapshot:
    """A hedge leg valued at a quoted price (units x price); no engine."""
    position_id: str
    underlying: str
    units: float
    price: float
    spot: float
    date: datetime
    tombstone: bool = False

    @property
    def total(self) -> float:
        return self.units * self.price


@dataclass(frozen=True)
class BookSnapshot:
    date: datetime
    positions: Mapping[str, PositionSnapshot]
    environments: Mapping[str, Any]
    quoted_legs: Mapping[str, QuotedLegSnapshot] = field(default_factory=lambda: _EMPTY)
    currency: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", MappingProxyType(dict(self.positions)))
        object.__setattr__(self, "environments", MappingProxyType(dict(self.environments)))
        object.__setattr__(self, "quoted_legs", MappingProxyType(dict(self.quoted_legs)))
        for pid, ps in self.positions.items():
            if ps.position_id != pid:
                raise ValidationError(f"position key {pid!r} != position_id {ps.position_id!r}")
            if ps.snapshot.date != self.date:
                raise ValidationError(f"position {pid} snapshot date differs from the book date")
            if self.currency is not None and ps.snapshot.currency not in (None, self.currency):
                raise ValidationError(f"position {pid} currency {ps.snapshot.currency} != book {self.currency}")
        if set(self.positions) & set(self.quoted_legs):
            raise ValidationError("a position id cannot be both a priced position and a quoted leg")

    @classmethod
    def from_portfolio(cls, portfolio, date: datetime, *,
                       tombstones: Optional[Mapping[str, PositionSnapshot]] = None,
                       currency: Optional[str] = None) -> "BookSnapshot":
        """Snapshot the portfolio as it stands on `date` (post-event products; the alive
        contracts travel in transitions). Environments must already be dated `date`."""
        from copy import deepcopy
        envs = {u: deepcopy(env) for u, env in portfolio.pricing_environments.items()}
        for u, env in envs.items():
            if env.valuation_date != date:
                raise ValidationError(f"environment for {u} is dated {env.valuation_date}, book date is {date}")
        positions: Dict[str, PositionSnapshot] = {}
        for pid, pos in portfolio.positions.items():
            state = deepcopy(getattr(pos, "lifecycle_state", None))
            snap = ValuationSnapshot(product=pos.product, engine=pos.engine, pricing_env=envs[pos.underlying],
                                     date=date, quantity=float(pos.quantity), lifecycle_state=state,
                                     currency=currency, label=pid)
            positions[pid] = PositionSnapshot(position_id=pid, underlying=pos.underlying, snapshot=snap)
        for pid, tomb in (tombstones or {}).items():
            if pid in positions:
                raise ValidationError(f"tombstone {pid} collides with a live position")
            positions[pid] = tomb
        return cls(date=date, positions=positions, environments=envs, currency=currency)


@dataclass(frozen=True)
class PositionExplainResult:
    position_id: str
    underlying: str
    instrument: Optional[PnLExplainResult]
    trade_rows: Tuple[ExplainRow, ...]
    total_pnl: float
    rows: Tuple[ExplainRow, ...]
    coordinate: Optional[Tuple[Optional[float], Optional[float]]] = None

    def reconcile(self, method: ExplainMethod) -> float:
        return self.total_pnl - component_sum(self.rows, method)


@dataclass(frozen=True)
class PortfolioExplainResult:
    date_t0: datetime
    date_t1: datetime
    positions: Mapping[str, PositionExplainResult]
    rows: Tuple[ExplainRow, ...]
    total_pnl: float
    metadata: Mapping[str, Any]
    _by_underlying: Mapping[str, Tuple[ExplainRow, ...]] = field(default_factory=lambda: _EMPTY, repr=False)

    def reconcile(self, method: ExplainMethod) -> float:
        return self.total_pnl - component_sum(self.rows, method)

    def by_underlying(self) -> Mapping[str, Tuple[ExplainRow, ...]]:
        return self._by_underlying

    def to_frame(self) -> pd.DataFrame:
        frames = []
        for pid in self.positions:
            pr = self.positions[pid]
            frames.append(rows_to_frame(pr.rows, date=self.date_t1, position_id=pid, underlying=pr.underlying))
        for underlying, rows in self._by_underlying.items():
            frames.append(rows_to_frame(rows, date=self.date_t1, position_id="portfolio", underlying=underlying))
        frames.append(rows_to_frame(self.rows, date=self.date_t1, position_id="portfolio", underlying="*"))
        return pd.concat(frames, ignore_index=True) if frames else rows_to_frame([], date=self.date_t1)


def _validate_trades(position_id: str, trades: Sequence[ExplainTrade], t0: datetime, t1: datetime) -> None:
    for t in trades:
        if t.position_id != position_id:
            raise ValidationError(f"trade for {t.position_id} passed to position {position_id}")
        if t.timestamp is not None and not (t0 < t.timestamp <= t1):
            raise ValidationError(f"trade timestamp {t.timestamp} outside ({t0}, {t1}]")


def _check_sum(trades: Sequence[ExplainTrade], expected: float, what: str) -> None:
    traded = sum(t.quantity for t in trades)
    if not is_close(traded, expected, rel_tol=0.0, abs_tol=1e-9 * max(1.0, abs(expected), abs(traded))):
        raise ValidationError(f"trades sum to {traded}, expected {expected} ({what})")


def _trade_rows(trades: Sequence[ExplainTrade], unit_pv_t1: float) -> Tuple[ExplainRow, ...]:
    rows = []
    for t in trades:
        rows.append(ExplainRow(
            factor=Factor.TRADE, term=f"trade:{t.kind}", method=ExplainMethod.SHARED, kind=RowKind.COMPONENT,
            level="position", pnl=t.quantity * (unit_pv_t1 - t.price),
            metadata={"kind": t.kind, "quantity": t.quantity, "price": t.price, "unit_pv_t1": unit_pv_t1,
                      "timestamp": t.timestamp.isoformat() if t.timestamp else None},
        ))
    return tuple(rows)


def _promote(instrument: PnLExplainResult) -> List[ExplainRow]:
    return [r.relabel("position") for r in instrument.rows if r.kind is not RowKind.SUMMARY]


def explain_position(
    pos_t0: Optional[PositionSnapshot],
    pos_t1: PositionSnapshot,
    *,
    trades: Sequence[ExplainTrade] = (),
    transition: Optional[LifecycleTransition] = None,
    config: Optional[PnLExplainConfig] = None,
) -> PositionExplainResult:
    trades = tuple(trades)
    pid, underlying = pos_t1.position_id, pos_t1.underlying
    s1 = pos_t1.snapshot
    if pos_t0 is None:
        if pos_t1.tombstone:
            raise ValidationError(f"position {pid}: a tombstone needs a t0 side")
        if not trades:
            raise ValidationError(f"position {pid} is new at t1 but no trades were supplied")
        _validate_trades(pid, trades, datetime.min, s1.date)
        _check_sum(trades, s1.quantity, "opened today")
        v1 = value(s1).total
        unit = v1 / s1.quantity
        trade_rows = _trade_rows(trades, unit)
        total = v1 + sum(t.cash for t in trades)
        rows = list(trade_rows) + [make_total_row("position", total)]
        return PositionExplainResult(pid, underlying, None, trade_rows, total, tuple(rows))

    s0 = pos_t0.snapshot
    if pos_t0.position_id != pid:
        raise ValidationError("pos_t0 and pos_t1 must share a position_id")
    _validate_trades(pid, trades, s0.date, s1.date)
    q0, q1 = s0.quantity, s1.quantity
    has_lifecycle = s0.lifecycle_state is not None or s1.lifecycle_state is not None
    terminal_t1 = is_terminal(s1.lifecycle_state)
    if has_lifecycle and trades:
        raise ValidationError(f"position {pid} is lifecycle-tracked; trades are not allowed")
    if has_lifecycle and q0 != q1:
        raise ValidationError(f"position {pid} is lifecycle-tracked; its quantity cannot change")
    if pos_t1.tombstone and not terminal_t1 and not trades:
        raise ValidationError(f"position {pid} disappeared without closing trades or a terminal state")
    if pos_t1.tombstone and terminal_t1 and trades:
        raise ValidationError(f"position {pid} terminated by lifecycle; trades are not allowed")

    if q1 != q0:
        _check_sum(trades, q1 - q0, "quantity change")
        s1_at_q0 = dataclasses.replace(s1, quantity=q0)
    elif pos_t1.tombstone and not terminal_t1:
        _check_sum(trades, -q0, "closed by trading")
        s1_at_q0 = s1
    else:
        if trades:
            _check_sum(trades, 0.0, "unchanged quantity")
        s1_at_q0 = s1
    instrument = explain(s0, s1_at_q0, config=config, transition=transition)
    unit = instrument.pv_t1.total / q0
    trade_rows = _trade_rows(trades, unit) if trades else ()
    if pos_t1.tombstone and not terminal_t1:
        v1 = 0.0
    elif q1 != q0:
        v1 = instrument.pv_t1.total * (q1 / q0)
    else:
        v1 = instrument.pv_t1.total
    total = v1 - instrument.pv_t0.total + sum(t.cash for t in trades)
    rows = _promote(instrument) + list(trade_rows) + [make_total_row("position", total)]
    coord = instrument.metadata.get("coordinate")
    return PositionExplainResult(pid, underlying, instrument, trade_rows, total, tuple(rows), coord)


def explain_quoted_leg(leg_t0: Optional[QuotedLegSnapshot], leg_t1: QuotedLegSnapshot, *,
                       trades: Sequence[ExplainTrade] = ()) -> PositionExplainResult:
    trades = tuple(trades)
    pid = leg_t1.position_id
    rows: List[ExplainRow] = []

    def shared(factor, term, pnl, moves=None):
        return ExplainRow(factor=factor, term=term, method=ExplainMethod.SHARED, kind=RowKind.COMPONENT,
                          level="position", pnl=pnl, moves=moves or {})

    if leg_t0 is None:
        if not trades:
            raise ValidationError(f"quoted leg {pid} is new at t1 but no trades were supplied")
        _check_sum(trades, leg_t1.units, "leg opened today")
        trade_rows = _trade_rows(trades, leg_t1.price)
        total = leg_t1.total + sum(t.cash for t in trades)
        rows = list(trade_rows) + [make_total_row("position", total)]
        return PositionExplainResult(pid, leg_t1.underlying, None, trade_rows, total, tuple(rows))
    if leg_t0.position_id != pid:
        raise ValidationError("quoted legs must share a position_id (rolls use contract-specific ids)")
    _validate_trades(pid, trades, leg_t0.date, leg_t1.date)
    if leg_t1.tombstone:
        _check_sum(trades, -leg_t0.units, "leg closed")
    else:
        _check_sum(trades, leg_t1.units - leg_t0.units, "leg quantity change")
    d_spot = leg_t1.spot - leg_t0.spot
    d_price = leg_t1.price - leg_t0.price
    rows.append(shared(Factor.SPOT, "spot", leg_t0.units * d_spot,
                       {"spot_return": d_spot / leg_t0.spot}))
    rows.append(shared(Factor.BASIS, "basis", leg_t0.units * (d_price - d_spot)))
    trade_rows = _trade_rows(trades, leg_t1.price)
    rows.extend(trade_rows)
    v1 = 0.0 if leg_t1.tombstone else leg_t1.total
    total = v1 - leg_t0.total + sum(t.cash for t in trades)
    rows.append(make_total_row("position", total))
    return PositionExplainResult(pid, leg_t1.underlying, None, trade_rows, total, tuple(rows))


def _aggregate(results: Sequence[PositionExplainResult], level_underlying: Optional[str]
               ) -> Tuple[ExplainRow, ...]:
    """Aggregate COMPONENT/INFORMATIONAL rows by (method, kind, factor, term); display
    columns survive only when every constituent shares one coordinate and step."""
    groups: Dict[Tuple, List[Tuple[ExplainRow, PositionExplainResult]]] = {}
    order: List[Tuple] = []
    for pr in results:
        for r in pr.rows:
            if r.kind is RowKind.SUMMARY:
                continue
            key = (r.method, r.kind, r.factor, r.term)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append((r, pr))
    out = []
    for key in order:
        items = groups[key]
        method, kind, factor, term = key
        pnl = sum(r.pnl for r, _ in items)
        coords = {(pr.underlying, pr.coordinate) for _, pr in items}
        steps = {r.step for r, _ in items}
        same = len(coords) == 1 and len(steps) == 1     # one underlying + one coordinate + one step
        greek = cash = None
        moves: Dict[str, float] = {}
        if same:
            if all(r.greek is not None for r, _ in items):
                greek = sum(r.greek for r, _ in items)
            if all(r.cash_greek is not None for r, _ in items):
                cash = sum(r.cash_greek for r, _ in items)
            moves = dict(items[0][0].moves)
        out.append(ExplainRow(
            factor=factor, term=term, method=method, kind=kind, level="portfolio", pnl=pnl, moves=moves,
            greek=greek, cash_greek=cash, step=steps.pop() if same else None,
            metadata={"positions": tuple(pr.position_id for _, pr in items),
                      "underlying": level_underlying or "*"},
        ))
    return tuple(out)


def explain_portfolio(
    book_t0: BookSnapshot,
    book_t1: BookSnapshot,
    *,
    trades: Sequence[ExplainTrade] = (),
    transaction_costs: float = 0.0,
    transitions: Mapping[str, LifecycleTransition] = _EMPTY,
    config: Optional[PnLExplainConfig] = None,
) -> PortfolioExplainResult:
    if book_t1.date <= book_t0.date:
        raise ValidationError("book_t1 must be dated after book_t0")
    if book_t0.currency is not None and book_t1.currency is not None and book_t0.currency != book_t1.currency:
        raise ValidationError("book currencies differ")
    by_pid: Dict[str, List[ExplainTrade]] = {}
    for t in trades:
        by_pid.setdefault(t.position_id, []).append(t)
    results: Dict[str, PositionExplainResult] = {}
    ids = sorted(set(book_t0.positions) | set(book_t1.positions))
    for pid in ids:
        p0, p1 = book_t0.positions.get(pid), book_t1.positions.get(pid)
        if p1 is None:
            raise ValidationError(f"position {pid} is absent at t1 with no tombstone")
        results[pid] = explain_position(p0, p1, trades=by_pid.pop(pid, ()),
                                        transition=transitions.get(pid), config=config)
    leg_ids = sorted(set(book_t0.quoted_legs) | set(book_t1.quoted_legs))
    for pid in leg_ids:
        l0, l1 = book_t0.quoted_legs.get(pid), book_t1.quoted_legs.get(pid)
        if l1 is None:
            raise ValidationError(f"quoted leg {pid} is absent at t1 with no tombstone")
        results[pid] = explain_quoted_leg(l0, l1, trades=by_pid.pop(pid, ()))
    if by_pid:
        raise ValidationError(f"trades for unknown position ids: {sorted(by_pid)}")

    cost_total = sum(t.transaction_cost for t in trades) + float(transaction_costs)
    ordered = [results[pid] for pid in sorted(results)]
    per_underlying: Dict[str, Tuple[ExplainRow, ...]] = {}
    for u in sorted({pr.underlying for pr in ordered}):
        per_underlying[u] = _aggregate([pr for pr in ordered if pr.underlying == u], u)
    book_rows = list(_aggregate(ordered, None))
    book_rows.append(ExplainRow(factor=Factor.TRANSACTION_COST, term="transaction_cost", method=ExplainMethod.SHARED,
                                kind=RowKind.COMPONENT, level="portfolio", pnl=-cost_total))
    total = sum(pr.total_pnl for pr in ordered) - cost_total
    book_rows.append(make_total_row("portfolio", total))
    recon: Dict[str, Dict[str, Any]] = {}
    methods = (config or PnLExplainConfig()).methods
    for m in methods:
        explained = component_sum(book_rows, m)
        gap = total - explained
        recon[m.value] = {"expected": total, "explained": explained, "gap": gap,
                          "ok": bool(is_close(gap, 0.0, rel_tol=0.0, abs_tol=1e-8 * max(1.0, abs(total))))}
    return PortfolioExplainResult(
        date_t0=book_t0.date, date_t1=book_t1.date, positions=MappingProxyType(results),
        rows=tuple(book_rows), total_pnl=total,
        metadata={"reconciliation": recon, "transaction_costs": cost_total},
        _by_underlying=MappingProxyType(per_underlying),
    )
```

(`explain()` already records `"coordinate"` in its metadata since Task 5.) Export from
`quantark/pnlexplain/__init__.py`: `ExplainTrade`, `PositionSnapshot`, `QuotedLegSnapshot`, `BookSnapshot`,
`PositionExplainResult`, `PortfolioExplainResult`, `explain_position`, `explain_quoted_leg`, `explain_portfolio`
(and add them to `__all__`).

This task is reviewed in two halves. First land `trades.py` plus the snapshot types, `from_portfolio` and
`explain_position` (everything above `explain_quoted_leg`), run the position-level tests, and commit:

- [ ] **Step 5: Run the position-level tests and commit part 1**

Run: `PYTEST test/test_pnlexplain_portfolio.py -q -k "unchanged or quantity_change or opened_today or from_portfolio"`
Expected: 4 passed

```bash
git add quantark/pnlexplain/__init__.py quantark/pnlexplain/equity/trades.py quantark/pnlexplain/equity/portfolio.py test/test_pnlexplain_portfolio.py
git commit -m "feat(pnlexplain): ExplainTrade, PositionSnapshot/BookSnapshot, explain_position with tombstones" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_019fnLipZtAJAPAQJLLEXQMb"
```

- [ ] **Step 6: Add the quoted leg, the portfolio layer and aggregation; run everything; commit part 2**

Append `explain_quoted_leg`, `_aggregate`, `explain_portfolio` (from the listing above), then:

Run: `PYTEST test/test_pnlexplain_portfolio.py test/test_pnlexplain_waterfall.py test/test_pnlexplain_taylor.py -q`
Expected: all passed (7 portfolio tests)

```bash
git add quantark/pnlexplain/equity/portfolio.py test/test_pnlexplain_portfolio.py
git commit -m "feat(pnlexplain): quoted futures legs, explain_portfolio with aggregation tiers and reconciliation" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_019fnLipZtAJAPAQJLLEXQMb"
```

---

### Task 11: Equity `BacktestEngine` recorder

**Review amendment (plan review 2):** `_trade_from_record` must fail closed. Skip a record ONLY
when it is an explicitly recognised zero-quantity "no trade" row (quantity is exactly zero). For every
non-zero record require: a non-empty position/contract id, a finite price, a finite quantity, and the
fields the recorder reads (multiplier / trade_type where applicable); a missing or non-finite value
raises `ValidationError` (never `TypeError`/`KeyError`, and never a silent `None`). Add
`test_nonzero_trade_without_position_id_is_rejected` and `test_missing_or_non_finite_trade_price_is_rejected`
to this task's test file (each feeds one malformed record through the recorder and expects
`ValidationError`).

**Files:**
- Create: `quantark/pnlexplain/equity/recorder.py`
- Modify: `quantark/backtest/equity/config.py` (field), `quantark/backtest/equity/engine.py` (`__init__`, `_step`, `_record_state`, `_finalize`), `quantark/backtest/equity/results.py` (frames)
- Test: `test/test_pnlexplain_backtest_equity.py`

**Interfaces:**
- Produces: `PnLExplainRecorder(config)` with `begin_day(engine, timestamp)`, `end_day(engine, timestamp, trade_records, cumulative_costs, lifecycle_events, state_pnl)`, `frames() -> (explain_df, reconciliation_df)`; `RECON_COLUMNS`; `BacktestConfig.pnl_explain: Optional[Any] = None`; `BacktestResults.explain_df`, `BacktestResults.explain_reconciliation_df` (properties; empty frames with the fixed columns when the explain is off).

- [ ] **Step 1: Write the failing tests**

```python
# test/test_pnlexplain_backtest_equity.py
"""Spec test 6 (equity engine): reconciliation, tombstone across a settlement lag, explain-on isolation."""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_backtest_lifecycle import RampAdapter, _down_out_put_position, UNDERLYING, START  # noqa: E402
from test_multi_greek_backtest import make_config as make_multi_config  # noqa: E402

from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit  # noqa: E402
from quantark.backtest import BacktestConfig, BacktestEngine, ZeroCostModel  # noqa: E402
from quantark.backtest.strategy import DeltaNeutralStrategy  # noqa: E402
from quantark.backtest.strategy.multi_greek_strategy import DeltaGammaNeutralStrategy  # noqa: E402
from quantark.pnlexplain import PnLExplainConfig, FRAME_COLUMNS  # noqa: E402
from quantark.pnlexplain.equity.recorder import RECON_COLUMNS  # noqa: E402
from quantark.util.calendar import BusinessDayConvention  # noqa: E402


def _lifecycle_config(pnl_explain=None, delta_threshold=1e12):
    adapter = RampAdapter(start_spot=100.0, end_spot=70.0, num_days=40)
    convention = SettlementConvention(lag=2, lag_unit=SettlementLagUnit.CALENDAR_DAYS,
                                      business_day_convention=BusinessDayConvention.UNADJUSTED)
    return BacktestConfig(
        strategy=DeltaNeutralStrategy(delta_threshold=delta_threshold),
        start_date=START, end_date=adapter.dates[-1].to_pydatetime(), underlying=UNDERLYING,
        initial_positions=[_down_out_put_position(2.0, 10.0, settlement_convention=convention)],
        market_data_adapter=adapter, transaction_cost_model=ZeroCostModel(),
        handle_lifecycle_events=True, pnl_explain=pnl_explain,
    )


def _frames(results):
    return results.states_df, results.trades_df, results.explain_df, results.explain_reconciliation_df


def _normalized(frame):
    """Object columns (lifecycle event records) compared by their string form; numbers exactly."""
    out = frame.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str)
    return out


def test_explain_off_is_byte_identical_and_frames_are_empty():
    off = BacktestEngine(_lifecycle_config()).run()
    states, trades, explain_df, recon = _frames(off)
    assert list(explain_df.columns) == ["date", *FRAME_COLUMNS] and explain_df.empty
    assert list(recon.columns) == RECON_COLUMNS and recon.empty
    on = BacktestEngine(_lifecycle_config(PnLExplainConfig())).run()
    pd.testing.assert_frame_equal(_normalized(on.states_df), _normalized(states))     # every column
    pd.testing.assert_frame_equal(on.trades_df, trades)
    pd.testing.assert_frame_equal(_normalized(on.get_lifecycle_events()), _normalized(off.get_lifecycle_events()))


def test_unknown_trade_type_is_rejected():
    from quantark.backtest.equity.state import TradeRecord
    from quantark.pnlexplain.equity.recorder import _trade_from_record
    from quantark.util.exceptions import ValidationError
    rec = TradeRecord(timestamp=START, trade_type="bogus", instrument_type="spot", underlying=UNDERLYING,
                      quantity=1.0, price=100.0, notional=100.0, transaction_cost=0.0, reason="x", position_id="p")
    with pytest.raises(ValidationError, match="unknown backtest trade type"):
        _trade_from_record(rec)


def test_lifecycle_ko_with_settlement_lag_reconciles_every_day():
    results = BacktestEngine(_lifecycle_config(PnLExplainConfig())).run()
    recon = results.explain_reconciliation_df
    port = recon[recon["level"] == "portfolio"]
    assert len(port) == 2 * (len(results.states_df) - 1)          # two methods, from day two
    assert port["ok"].all()
    assert (port["gap_states"].abs() <= 1e-8 * port["expected"].abs().clip(lower=1.0)).all()
    ex = results.explain_df
    events = results.get_lifecycle_events()
    ko_day = pd.Timestamp(events.iloc[0]["date"])
    ko_rows = ex[(ex["date"] == ko_day) & (ex["factor"] == "lifecycle_event") & (ex["level"] == "position")]
    assert len(ko_rows) == 1 and ko_rows.iloc[0]["pnl"] != 0.0
    # the tombstone lives on for the settlement lag: time rows exist after the KO day, event rows are zero
    later = ex[(ex["date"] > ko_day) & (ex["level"] == "position")]
    assert (later[later["factor"] == "lifecycle_event"]["pnl"] == 0.0).all()
    assert set(later["term"]) >= {"time", "rate"}
    assert later["date"].nunique() >= 1


def test_spot_hedge_adjusts_identity_holds_states_gap_documented():
    results = BacktestEngine(_lifecycle_config(PnLExplainConfig(), delta_threshold=0.0)).run()
    recon = results.explain_reconciliation_df
    port = recon[recon["level"] == "portfolio"]
    assert port["ok"].all()
    trades = results.trades_df
    assert len(trades) > 1
    # the simple executor keeps the original entry price on adjusts: gap_states is reported, not zero
    assert port["gap_states"].notna().all()


def test_multi_instrument_hedge_states_gap_is_zero():
    cfg = make_multi_config(DeltaGammaNeutralStrategy(delta_threshold=1.0, gamma_threshold=0.5,
                                                      rebalance_frequency="continuous"))
    cfg.pnl_explain = PnLExplainConfig(methods=(__import__("quantark.pnlexplain", fromlist=["ExplainMethod"]).ExplainMethod.WATERFALL,))
    results = BacktestEngine(cfg).run()
    port = results.explain_reconciliation_df.query("level == 'portfolio'")
    assert port["ok"].all()
    assert (port["gap_states"].abs() <= 1e-8 * port["expected"].abs().clip(lower=1.0)).all()
    assert (results.explain_df["level"] == "position").any()
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST test/test_pnlexplain_backtest_equity.py -q`
Expected: FAIL with `TypeError: BacktestConfig.__init__() got an unexpected keyword argument 'pnl_explain'`

- [ ] **Step 3: Config and results plumbing**

`quantark/backtest/equity/config.py`: add after `handle_lifecycle_events`:
```python
    pnl_explain: Optional[Any] = None   # PnLExplainConfig; None = no explain, no behaviour change
```
and a docstring line `pnl_explain: Optional daily PnL explain configuration (quantark.pnlexplain)`.

`quantark/backtest/equity/results.py`: extend `__init__` with `explain_frames=None` and add:
```python
    @property
    def explain_df(self) -> pd.DataFrame:
        if self._explain_frames is None:
            from quantark.pnlexplain.base import rows_to_frame
            return rows_to_frame([], date=self.config.start_date).iloc[0:0]
        return self._explain_frames[0]

    @property
    def explain_reconciliation_df(self) -> pd.DataFrame:
        if self._explain_frames is None:
            from quantark.pnlexplain.equity.recorder import RECON_COLUMNS
            return pd.DataFrame(columns=RECON_COLUMNS)
        return self._explain_frames[1]
```
(store `self._explain_frames = explain_frames` in `__init__`).

- [ ] **Step 4: Implement `recorder.py` (equity part)**

```python
# quantark/pnlexplain/equity/recorder.py
"""Backtest recorders: one explain row set per day (spec §10)."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from quantark.pnlexplain.base import ExplainMethod, component_sum, rows_to_frame
from quantark.pnlexplain.config import PnLExplainConfig
from quantark.pnlexplain.equity.lifecycle import LifecycleTransition
from quantark.pnlexplain.equity.portfolio import (
    BookSnapshot, PortfolioExplainResult, PositionSnapshot, QuotedLegSnapshot, explain_portfolio,
)
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot, is_terminal
from quantark.pnlexplain.equity.trades import ExplainTrade
from quantark.asset.equity.lifecycle.cashflows import ValuationPoint
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close, is_zero

RECON_COLUMNS = ["date", "method", "level", "position_id", "expected", "explained", "gap", "ok",
                 "expected_states", "gap_states"]
# every trade type the two executors and the replay engine emit; anything else is an error
_KIND_MAP = {"open": "open", "adjust": "adjust", "close": "close", "roll_close": "roll_close",
             "roll_open": "roll_open", "hedge": "adjust", "hedge_rebalance": "adjust", "hedge_close": "close"}


def trade_kind(trade_type: str) -> str:
    try:
        return _KIND_MAP[trade_type]
    except KeyError:
        raise ValidationError(
            f"unknown backtest trade type {trade_type!r}; known {sorted(_KIND_MAP)}"
        ) from None


def _midnight(ts) -> datetime:
    return pd.Timestamp(ts).normalize().to_pydatetime()


class _RowSink:
    def __init__(self) -> None:
        self.frames: List[pd.DataFrame] = []
        self.recon: List[Dict[str, Any]] = []

    def add(self, date: datetime, result: PortfolioExplainResult, methods: Sequence[ExplainMethod],
            expected_states: float) -> None:
        self.frames.append(result.to_frame())
        for m in methods:
            explained = component_sum(result.rows, m)
            gap = result.total_pnl - explained
            self.recon.append({
                "date": pd.Timestamp(date), "method": m.value, "level": "portfolio", "position_id": "portfolio",
                "expected": result.total_pnl, "explained": explained, "gap": gap,
                "ok": bool(is_close(gap, 0.0, rel_tol=0.0, abs_tol=1e-8 * max(1.0, abs(result.total_pnl)))),
                "expected_states": expected_states, "gap_states": expected_states - explained,
            })
            for pid, pr in result.positions.items():
                pexp = component_sum(pr.rows, m)
                pgap = pr.total_pnl - pexp
                self.recon.append({
                    "date": pd.Timestamp(date), "method": m.value, "level": "position", "position_id": pid,
                    "expected": pr.total_pnl, "explained": pexp, "gap": pgap,
                    "ok": bool(is_close(pgap, 0.0, rel_tol=0.0, abs_tol=1e-8 * max(1.0, abs(pr.total_pnl)))),
                    "expected_states": float("nan"), "gap_states": float("nan"),
                })

    def result_frames(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        explain_df = pd.concat(self.frames, ignore_index=True) if self.frames else rows_to_frame([], date=datetime(1970, 1, 1)).iloc[0:0]
        recon = pd.DataFrame(self.recon, columns=RECON_COLUMNS)
        return explain_df, recon


def _trade_from_record(rec: Any) -> Optional[ExplainTrade]:
    if is_zero(float(rec.quantity)) or not rec.position_id:
        return None                       # the executors' zero-quantity "no trade" records
    return ExplainTrade(position_id=rec.position_id, quantity=float(rec.quantity), price=float(rec.price),
                        transaction_cost=float(rec.transaction_cost), timestamp=_midnight(rec.timestamp),
                        kind=trade_kind(rec.trade_type), instrument_type=rec.instrument_type)


class PnLExplainRecorder:
    """Hook for quantark.backtest.equity.BacktestEngine."""

    def __init__(self, config: PnLExplainConfig) -> None:
        self.config = config
        self._sink = _RowSink()
        self._prev_book: Optional[BookSnapshot] = None
        self._prev_costs = 0.0
        self._prev_state_pnl = 0.0
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._carried: Dict[str, PositionSnapshot] = {}

    # -- day boundaries -------------------------------------------------
    def begin_day(self, engine: Any, timestamp: datetime) -> None:
        """After _update_pricing_environment, BEFORE lifecycle processing."""
        self._pending = {}
        if self._prev_book is None:
            return
        portfolio = engine.portfolio
        manager = engine.lifecycle_manager
        products = (manager.pricing_products(portfolio, timestamp) if manager is not None
                    else {pid: p.product for pid, p in portfolio.positions.items()})
        for pid, pos in portfolio.positions.items():
            self._pending[pid] = {
                "product_alive": products[pid], "engine_alive": pos.engine, "underlying": pos.underlying,
                "quantity": float(pos.quantity), "state_ref": getattr(pos, "lifecycle_state", None),
                "state_before": deepcopy(getattr(pos, "lifecycle_state", None)),
            }

    def end_day(self, engine: Any, timestamp: datetime, trade_records: Sequence[Any],
                cumulative_costs: float, lifecycle_events: Sequence[Any], state_pnl: float) -> None:
        """After _record_state."""
        ts = _midnight(timestamp)
        portfolio = engine.portfolio
        envs = {u: deepcopy(env) for u, env in portfolio.pricing_environments.items()}
        live: Dict[str, PositionSnapshot] = {}
        for pid, pos in portfolio.positions.items():
            snap = ValuationSnapshot(product=pos.product, engine=pos.engine, pricing_env=envs[pos.underlying],
                                     date=ts, quantity=float(pos.quantity),
                                     lifecycle_state=deepcopy(getattr(pos, "lifecycle_state", None)), label=pid)
            live[pid] = PositionSnapshot(pid, pos.underlying, snap)
        if self._prev_book is None:
            self._prev_book = BookSnapshot(date=ts, positions=live, environments=envs)
            self._prev_costs, self._prev_state_pnl = float(cumulative_costs), float(state_pnl)
            return

        trades = [t for t in (_trade_from_record(r) for r in trade_records) if t is not None]
        traded_ids = {t.position_id for t in trades}
        events_by_pid: Dict[str, List[Any]] = {}
        for item in lifecycle_events:
            events_by_pid.setdefault(item.position_id, []).append(item.event)

        transitions: Dict[str, LifecycleTransition] = {}
        tombstones: Dict[str, PositionSnapshot] = {}
        for pid, pend in self._pending.items():
            state_after = deepcopy(pend["state_ref"]) if pend["state_ref"] is not None else None
            transitions[pid] = LifecycleTransition(
                product_alive_t1=pend["product_alive"], engine_alive_t1=pend["engine_alive"],
                state_before=pend["state_before"], state_after=state_after,
                events=tuple(events_by_pid.get(pid, ())))
            if pid in live:
                continue
            env = envs[pend["underlying"]]
            if is_terminal(state_after):
                snap = ValuationSnapshot(pend["product_alive"], pend["engine_alive"], env, date=ts,
                                         quantity=pend["quantity"], lifecycle_state=state_after, label=pid)
            elif pid in traded_ids:
                snap = ValuationSnapshot(pend["product_alive"], pend["engine_alive"], env, date=ts,
                                         quantity=pend["quantity"], lifecycle_state=None, label=pid)
            else:
                raise ValidationError(f"position {pid} left the book without a terminal state or closing trades")
            tombstones[pid] = PositionSnapshot(pid, pend["underlying"], snap, tombstone=True)
        for pid, old in self._carried.items():          # terminal tombstones still awaiting cash
            if pid in tombstones or pid in live:
                continue
            snap = old.snapshot
            tombstones[pid] = PositionSnapshot(pid, old.underlying, ValuationSnapshot(
                snap.product, snap.engine, envs[old.underlying], date=ts, quantity=snap.quantity,
                lifecycle_state=deepcopy(snap.lifecycle_state), label=pid), tombstone=True)

        book_t1 = BookSnapshot(date=ts, positions={**live, **tombstones}, environments=envs)
        day_costs = float(cumulative_costs) - self._prev_costs
        extra_costs = day_costs - sum(t.transaction_cost for t in trades)
        if is_zero(extra_costs):
            extra_costs = 0.0
        result = explain_portfolio(self._prev_book, book_t1, trades=trades, transaction_costs=extra_costs,
                                   transitions=transitions, config=self.config)
        self._sink.add(ts, result, self.config.methods, float(state_pnl) - self._prev_state_pnl)

        # tomorrow's t0: live positions + terminal tombstones with pending cash
        carried: Dict[str, PositionSnapshot] = {}
        for pid, tomb in tombstones.items():
            st = tomb.snapshot.lifecycle_state
            if st is not None and is_terminal(st) and st.ledger.pending(ValuationPoint(date=ts)):
                carried[pid] = tomb
        self._carried = carried
        self._prev_book = BookSnapshot(date=ts, positions={**live, **carried}, environments=envs)
        self._prev_costs, self._prev_state_pnl = float(cumulative_costs), float(state_pnl)

    def frames(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self._sink.result_frames()
```

- [ ] **Step 5: Hook the engine**

`quantark/backtest/equity/engine.py`:
- `__init__` (after `self._num_hedges_executed = 0`):
```python
        self._explain_recorder = None
        self._last_net_pnl: float = 0.0
        if getattr(config, "pnl_explain", None) is not None:
            from quantark.pnlexplain.equity.recorder import PnLExplainRecorder
            self._explain_recorder = PnLExplainRecorder(config.pnl_explain)
```
- `_step`: right after `self._update_pricing_environment(timestamp)`:
```python
        if self._explain_recorder is not None:
            self._explain_recorder.begin_day(self, timestamp)
```
  and after the `self._record_state(...)` call:
```python
        if self._explain_recorder is not None:
            self._explain_recorder.end_day(
                self, timestamp, trade_records, self._cumulative_transaction_costs,
                self._lifecycle_events_today, self._last_net_pnl,
            )
```
- `_record_state`: after computing `net_pnl`, add
  ```python
        if self._explain_recorder is not None:
            self._last_net_pnl = net_pnl
  ```
  (no state is written when the explain is off).
- `_finalize`: pass `explain_frames=self._explain_recorder.frames() if self._explain_recorder is not None else None` to `BacktestResults(...)`.

- [ ] **Step 6: Run tests**

Run: `PYTEST test/test_pnlexplain_backtest_equity.py test/test_backtest_lifecycle.py test/test_multi_greek_backtest.py test/test_backtest_engine.py -q`
Expected: all passed. Debug notes: (a) if `test_lifecycle_ko_with_settlement_lag_reconciles_every_day` shows a non-zero `gap_states` on the KO day only, the states frame's `pending_receivable_pv` uses the lifecycle manager's own discounting; compare `manager.pending_receivable_pv` with `ledger.pending_pv` at the same date and align `value()` to whichever the resolver produces — the ledger is the spec's authority, so if the manager differs, report it as a finding rather than bending `value()`. **Acceptance criterion (fixed, not contingent):** `gap` (value identity minus explained rows) is zero within `1e-8 × max(1, |expected|)` on every day for every executor. `gap_states` (states-frame PnL minus explained rows) is zero within the same tolerance for the average-cost executors (multi-instrument equity, replay) and is REPORTED — finite, uncapped — for the equity simple `HedgeExecutor`, whose adjust-day entry-price behaviour is a pre-existing accounting quirk outside this feature's scope. Neither `value()` nor the states accounting is bent to close a gap; a ledger-vs-engine discounting difference found under (a) leaves `gap_states` reported and goes into the final report. (b) The `states_df` comparison drops `lifecycle*` columns only because they hold event objects; every numeric column must match exactly.

- [ ] **Step 7: Commit**

```bash
git add quantark/pnlexplain/equity/recorder.py quantark/backtest/equity/config.py quantark/backtest/equity/engine.py quantark/backtest/equity/results.py test/test_pnlexplain_backtest_equity.py
git commit -m "feat(pnlexplain): equity BacktestEngine recorder with tombstones, explain_df and reconciliation frames" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_019fnLipZtAJAPAQJLLEXQMb"
```

---

### Task 12: Replay engine recorder

**Review amendment (plan review 2):** `_trade_from_record` must fail closed. Skip a record ONLY
when it is an explicitly recognised zero-quantity "no trade" row (quantity is exactly zero). For every
non-zero record require: a non-empty position/contract id, a finite price, a finite quantity, and the
fields the recorder reads (multiplier / trade_type where applicable); a missing or non-finite value
raises `ValidationError` (never `TypeError`/`KeyError`, and never a silent `None`). Add
`test_nonzero_trade_without_position_id_is_rejected` and `test_missing_or_non_finite_trade_price_is_rejected`
to this task's test file (each feeds one malformed record through the recorder and expects
`ValidationError`).

**Files:**
- Modify: `quantark/pnlexplain/equity/recorder.py` (add `ReplayPnLExplainRecorder`), `quantark/backtest/replay/config.py` (both configs), `quantark/backtest/replay/product_replay.py` (`events_today`), `quantark/backtest/replay/engine.py` (hooks), `quantark/backtest/replay/single.py` (pass-through), `quantark/backtest/replay/results.py` (frames)
- Test: `test/test_pnlexplain_backtest_replay.py`

**Interfaces:**
- Produces: `ReplayPnLExplainRecorder(config)` with `begin_day(engine, date, env)`, `end_day(engine, date, env, market, selected, trades_today, state_row)`, `frames()`; `ReplayBacktestConfig.pnl_explain` / `AutocallableBacktestConfig.pnl_explain`; `ProductReplay.events_today: list`; `BookBacktestResults.explain_df()` / `.explain_reconciliation_df()` (methods, like its other frames) and `AutocallableBacktestResults.explain_df` / `.explain_reconciliation_df` (properties).

- [ ] **Step 1: Write the failing tests**

```python
# test/test_pnlexplain_backtest_replay.py
"""Spec tests 6 (replay: futures hedge with a roll), 8 (LV model row)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from replay_golden import fixtures  # noqa: E402

from quantark.backtest.replay.engine import ReplayBacktestEngine  # noqa: E402
from quantark.backtest.replay.single import AutocallableBacktestEngine  # noqa: E402
from quantark.pnlexplain import ExplainMethod, PnLExplainConfig  # noqa: E402

WF = PnLExplainConfig(methods=(ExplainMethod.WATERFALL,))


def test_replay_book_explain_off_identical_and_on_reconciles():
    off = ReplayBacktestEngine(fixtures.make_book_config()).run()
    cfg = fixtures.make_book_config()
    cfg.pnl_explain = WF
    on = ReplayBacktestEngine(cfg).run()
    pd.testing.assert_frame_equal(on.states_df(), off.states_df())
    pd.testing.assert_frame_equal(on.greeks_df(), off.greeks_df())
    pd.testing.assert_frame_equal(on.trades_df(), off.trades_df())
    assert off.explain_df().empty
    recon = on.explain_reconciliation_df()
    port = recon[recon["level"] == "portfolio"]
    assert len(port) == len(on.states_df()) - 1
    assert port["ok"].all()
    assert (port["gap_states"].abs() <= 1e-8 * port["expected"].abs().clip(lower=1.0)).all()
    ex = on.explain_df()
    hedge_rows = ex[ex["position_id"].str.startswith("hedge:")]
    assert set(hedge_rows["term"]) >= {"spot", "basis"}
    assert (ex[ex["level"] == "position"]["position_id"].str.startswith("hedge:") | ex[ex["level"] == "position"]["position_id"].isin(["1", "2"])).all()


def test_replay_roll_produces_close_and_open_legs():
    cfg = fixtures.make_book_config()
    cfg.pnl_explain = WF
    results = ReplayBacktestEngine(cfg).run()
    trades = results.trades_df()
    assert set(trades["trade_type"]) >= {"roll_close", "roll_open"}    # IF2401 expires 2024-01-07 inside the window
    ex = results.explain_df()
    assert (ex["term"] == "trade:roll_close").any() and (ex["term"] == "trade:roll_open").any()
    port = results.explain_reconciliation_df().query("level == 'portfolio'")
    assert port["ok"].all()


def test_single_engine_passthrough():
    cfg = fixtures.make_scalar_bsm_config()
    cfg.pnl_explain = WF
    results = AutocallableBacktestEngine(cfg).run()
    assert not results.explain_df.empty
    assert results.explain_reconciliation_df.query("level == 'portfolio'")["ok"].all()


def test_unknown_replay_trade_type_is_rejected():
    from quantark.pnlexplain.equity.recorder import trade_kind
    from quantark.util.exceptions import ValidationError
    with pytest.raises(ValidationError, match="unknown backtest trade type"):
        trade_kind("hedge_swap")


def test_localvol_recalibration_lands_in_model_row(tmp_path):
    root = fixtures.write_localvol_history(tmp_path / "lv")
    cfg = fixtures.make_localvol_config(root)
    cfg.pnl_explain = WF
    results = AutocallableBacktestEngine(cfg).run()
    ex = results.explain_df
    pos = ex[(ex["level"] == "position") & (ex["position_id"] == "0")]
    model = pos[pos["term"] == "model"]["pnl"].abs().sum()
    vol = pos[pos["term"] == "vol"]["pnl"].abs().sum()
    total = pos[pos["term"] == "total"]["pnl"].abs().sum()
    assert vol <= 1e-8 * max(1.0, total)
    assert model > 0.0
    assert results.explain_reconciliation_df.query("level == 'portfolio'")["ok"].all()
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST test/test_pnlexplain_backtest_replay.py -q`
Expected: FAIL with `AttributeError: ... has no attribute 'explain_df'` / unexpected `pnl_explain`

- [ ] **Step 3: Config, replay events, results plumbing**

`quantark/backtest/replay/config.py`: add `pnl_explain: Optional[Any] = None` to both `AutocallableBacktestConfig` (after `terminate_on_lifecycle_end`) and `ReplayBacktestConfig` (same place). `single.py`: pass `pnl_explain=config.pnl_explain` into the `ReplayBacktestConfig(...)` it builds.

`quantark/backtest/replay/product_replay.py`: in `__init__` add `self.record_events: bool = False` and `self.events_today: list = []`; in `apply_lifecycle_events` start with `if self.record_events: self.events_today = []` and append each event before `self.actions_sink.append(...)` only when `self.record_events`; in `settle_maturity_if_due` append `event` when not None and `self.record_events`. The recorder-owning engine sets `replay.record_events = True` for every replay when `pnl_explain` is configured, so an explain-off run writes no new state.

`quantark/backtest/replay/results.py`: both result classes take `explain_frames=None`; `BookBacktestResults` gets methods `explain_df(self)` / `explain_reconciliation_df(self)`; `AutocallableBacktestResults` gets the same as `@property`. Empty defaults exactly as in Task 11 Step 3. `single.py` forwards `explain_frames=inner._explain_frames` (set in `run`).

- [ ] **Step 4: Implement `ReplayPnLExplainRecorder`**

```python
# append to quantark/pnlexplain/equity/recorder.py
class ReplayPnLExplainRecorder:
    """Hook for quantark.backtest.replay.ReplayBacktestEngine."""

    def __init__(self, config: PnLExplainConfig) -> None:
        self.config = config
        self._sink = _RowSink()
        self._prev_book: Optional[BookSnapshot] = None
        self._prev_costs = 0.0
        self._prev_state_pnl = 0.0
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._prev_engines: List[Any] = []

    @staticmethod
    def _pid(replay: Any) -> str:
        return str(replay.position_id)

    def begin_day(self, engine: Any, date, env: Any) -> None:
        """After the day's env is built and engines calibrated, BEFORE apply_lifecycle_events."""
        self._pending = {}
        if self._prev_book is None:
            return
        for replay, day_engine, qty in zip(engine._replays, engine._pricing_engines, engine._quantities):
            self._pending[self._pid(replay)] = {
                "product_alive": replay.product_for_date(date, env), "engine_alive": day_engine,
                "state_ref": replay.lifecycle, "state_before": deepcopy(replay.lifecycle), "quantity": float(qty),
            }

    def _leg(self, engine: Any, date: datetime, spot: float, price: float, contract: str,
             units: float, tombstone: bool) -> QuotedLegSnapshot:
        return QuotedLegSnapshot(position_id=f"hedge:{contract}", underlying=engine.config.underlying,
                                 units=units, price=price, spot=spot, date=date, tombstone=tombstone)

    def end_day(self, engine: Any, date, env: Any, market: Dict[str, float], selected: Any,
                trades_today: Sequence[Dict[str, Any]], state_row: Dict[str, Any]) -> None:
        ts = _midnight(date)
        env_copy = deepcopy(env)
        underlying = engine.config.underlying
        spot = float(market["spot"])
        # priced positions (post-event)
        live: Dict[str, PositionSnapshot] = {}
        for replay, day_engine, qty in zip(engine._replays, engine._pricing_engines, engine._quantities):
            pid = self._pid(replay)
            snap = ValuationSnapshot(replay.product_for_date(date, env), day_engine, env_copy, date=ts,
                                     quantity=float(qty), lifecycle_state=deepcopy(replay.lifecycle), label=pid)
            live[pid] = PositionSnapshot(pid, underlying, snap)
        # hedge leg(s)
        hp = engine.hedge_position
        legs: Dict[str, QuotedLegSnapshot] = {}
        trades: List[ExplainTrade] = []
        closed_contracts: Dict[str, float] = {}
        for row in trades_today:
            kind = trade_kind(str(row["trade_type"]))
            trades.append(ExplainTrade.from_contracts(
                f"hedge:{row['contract']}", row["quantity"], row["price"], row["multiplier"],
                transaction_cost=float(row["transaction_cost"]), timestamp=ts, kind=kind,
                instrument_type=row["instrument_type"]))
            if kind in ("close", "roll_close"):
                closed_contracts[str(row["contract"])] = float(row["price"])
        if hp.contract is not None and not is_zero(hp.quantity):
            legs[f"hedge:{hp.contract}"] = self._leg(engine, ts, spot, float(selected["futures_price"]),
                                                     str(hp.contract), hp.quantity * hp.multiplier, False)
        prev_legs = self._prev_book.quoted_legs if self._prev_book is not None else {}
        for pid, old in prev_legs.items():
            if pid in legs:
                continue
            contract = pid.split(":", 1)[1]
            if contract not in closed_contracts:
                raise ValidationError(f"hedge leg {pid} disappeared without a closing trade")
            legs[pid] = self._leg(engine, ts, spot, closed_contracts[contract], contract, old.units, True)

        if self._prev_book is None:
            self._prev_book = BookSnapshot(date=ts, positions=live, environments={underlying: env_copy}, quoted_legs=legs)
            self._prev_costs, self._prev_state_pnl = float(engine._transaction_costs), float(state_row["total_pnl"])
            return

        transitions = {}
        for pid, pend in self._pending.items():
            transitions[pid] = LifecycleTransition(
                product_alive_t1=pend["product_alive"], engine_alive_t1=pend["engine_alive"],
                state_before=pend["state_before"], state_after=deepcopy(pend["state_ref"]),
                events=tuple(next(r for r in engine._replays if self._pid(r) == pid).events_today))
        book_t1 = BookSnapshot(date=ts, positions=live, environments={underlying: env_copy}, quoted_legs=legs)
        day_costs = float(engine._transaction_costs) - self._prev_costs
        extra = day_costs - sum(t.transaction_cost for t in trades)
        if is_zero(extra):
            extra = 0.0
        result = explain_portfolio(self._prev_book, book_t1, trades=trades, transaction_costs=extra,
                                   transitions=transitions, config=self.config)
        self._sink.add(ts, result, self.config.methods, float(state_row["total_pnl"]) - self._prev_state_pnl)
        live_legs = {pid: leg for pid, leg in legs.items() if not leg.tombstone}
        self._prev_book = BookSnapshot(date=ts, positions=live, environments={underlying: env_copy}, quoted_legs=live_legs)
        self._prev_costs, self._prev_state_pnl = float(engine._transaction_costs), float(state_row["total_pnl"])

    def frames(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self._sink.result_frames()
```

Note on terminated replay products: a knocked-out snowball stays in `engine._replays` (the replay keeps pricing 0 and tracking the receivable), so `live` always contains it with its terminal state — no tombstones are needed for the replay engine; `value()` prices the contingent leg as 0 once terminal.

- [ ] **Step 5: Hook `ReplayBacktestEngine.run`**

In `__init__` (after `self._pricing_engines` is built):
```python
        self._explain_recorder = None
        self._explain_frames = None
        if getattr(config, "pnl_explain", None) is not None:
            from quantark.pnlexplain.equity.recorder import ReplayPnLExplainRecorder
            self._explain_recorder = ReplayPnLExplainRecorder(config.pnl_explain)
            for replay in self._replays:
                replay.record_events = True
```
In `run()` the per-day sequence is exactly this (NEW lines marked; everything else already exists):

1. `date = ...normalize()`
2. NEW — first statement after the normalisation, BEFORE the roll block, so roll-close / roll-open
   trades are captured: `trades_before = len(self._trades)`
3. the roll block
4. `env = build_env(...)` and `_calibrate_day` (fresh engines)
5. NEW — right after `pricing_started = time.perf_counter()`:
```python
            if self._explain_recorder is not None:
                self._explain_recorder.begin_day(self, date, env)
```
6. lifecycle events, pricing, `self._record_day(...)`
7. NEW — right after the `self._record_day(...)` call:
```python
            if self._explain_recorder is not None:
                self._explain_recorder.end_day(
                    self, date, env, market, selected, self._trades[trades_before:], self._states[-1],
                )
```
At the end of `run()` set `self._explain_frames = self._explain_recorder.frames() if self._explain_recorder is not None else None` and pass `explain_frames=self._explain_frames` to `BookBacktestResults(...)`. In `single.py` pass `explain_frames=inner._explain_frames` to `AutocallableBacktestResults(...)`.

- [ ] **Step 6: Run tests**

Run: `PYTEST test/test_pnlexplain_backtest_replay.py test/test_replay_greeks_failclosed.py test/replay_golden -q`
Expected: all passed. Debug notes: (a) if `gap_states` is non-zero on days with a pending KO receivable, the replay values it as `pending × DF((settlement − date).days/365)` while the ledger resolves through `SettlementResolver`; print both on that date and report the difference as a finding (the ledger is the spec's authority). **Acceptance criterion (fixed, not contingent):** `gap` (value identity minus explained rows) is zero within `1e-8 × max(1, |expected|)` on every day for every executor. `gap_states` (states-frame PnL minus explained rows) is zero within the same tolerance for the average-cost executors (multi-instrument equity, replay) and is REPORTED — finite, uncapped — for the equity simple `HedgeExecutor`, whose adjust-day entry-price behaviour is a pre-existing accounting quirk outside this feature's scope. Neither `value()` nor the states accounting is bent to close a gap; a ledger-vs-engine discounting difference found under (a) leaves `gap_states` reported and goes into the final report. (b) The two fixture products share `position_id` 1 and 2; the recorder keys by `str(position_id)`.

- [ ] **Step 7: Commit**

```bash
git add quantark/pnlexplain/equity/recorder.py quantark/backtest/replay/config.py quantark/backtest/replay/product_replay.py quantark/backtest/replay/engine.py quantark/backtest/replay/single.py quantark/backtest/replay/results.py test/test_pnlexplain_backtest_replay.py
git commit -m "feat(pnlexplain): replay engine recorder with quoted futures legs, rolls and model rows" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_019fnLipZtAJAPAQJLLEXQMb"
```

---

### Task 13: Bucketed vega / rho rows (P5, opt-in)

**Review amendment (plan review 2):** structure compatibility is a RULE, not a hint: both
snapshots must carry the same term-structure class on the SAME pillar grid (vol `times`, rate pillar
tenors) or bucketed mode raises `ValidationError`. Add
`test_bucketed_rejects_mismatched_pillars` (a `TermStructureVolSurface(times=[0.5, 1.0], ...)` at t0
against `TermStructureVolSurface(times=[0.25, 0.5, 1.0], ...)` at t1 with `bucketed=True` →
`ValidationError`), next to the existing term-structure-vs-flat rejection test. `_pillars` needs
`Mapping` from `typing` in `bucketed.py`.

**Files:**
- Create: `quantark/pnlexplain/equity/bucketed.py`
- Modify: `quantark/pnlexplain/equity/taylor.py` (replace scalar vega / rho rows when `config.bucketed`)
- Test: `test/test_pnlexplain_bucketed.py`

**Interfaces:**
- Consumes: `GreeksCalculator.calculate_bucketed_greeks(product, env, engine, BucketedGreeksRequest(coordinates=..., vol_bump=..., rate_bump=...))` returning `BucketedGreeksResult.points` with `coordinate`, `name`, `derivative`, `maturity`, `bump_size`, `metadata`.
- Produces: `bucketed_rows(cache, calc, bump, level) -> tuple[Mapping[Factor, tuple[ExplainRow, ...]], frozenset[Factor]]` — rows partitioned by factor (`Factor.VOL` → the `vega.<τ>` COMPONENT rows; `Factor.RATE` → the `rho.<τ>` COMPONENT rows followed by the informational `rate_keyrate.parallel` row) and the set of factors covered; `taylor_rows` inserts `rows[Factor.VOL]` where the scalar `vega` row would have been and `rows[Factor.RATE]` where `rho` would have been, emitting no scalar row for a covered factor. This task also removes the `bucketed` gate in `taylor_rows` and `test_bucketed_is_gated_until_task_13`.

- [ ] **Step 1: Write the failing tests**

```python
# test/test_pnlexplain_bucketed.py
"""Spec test 9: bucket rows replace the scalar rows and sum to them within FD tolerance."""
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.param import FlatRateCurve, LinearRateCurve, SpotQuote, TermStructureVolSurface
from quantark.param.div import ContinuousDividendYield
from quantark.pnlexplain import ExplainMethod, PnLExplainConfig, RowKind, ValuationSnapshot, explain
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

T0, T1 = datetime(2026, 6, 26), datetime(2026, 6, 29)


ENG = BlackScholesEngine()


def _env(spot, date, vols, rates):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=TermStructureVolSurface(times=[0.5, 1.0, 2.0], vols=list(vols)),
        rate_curve=LinearRateCurve(pillars=list(zip([0.5, 1.0, 2.0], rates))),
        div_yield=ContinuousDividendYield(div_yield=0.01), valuation_date=date)


def _snaps():
    s0 = ValuationSnapshot(EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0),
                           ENG, _env(100.0, T0, (0.20, 0.22, 0.24), (0.030, 0.032, 0.034)), date=T0)
    s1 = ValuationSnapshot(EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0 - 3 / 365),
                           ENG, _env(101.0, T1, (0.21, 0.225, 0.24), (0.031, 0.032, 0.035)), date=T1)
    return s0, s1


def test_bucket_rows_replace_scalar_rows_and_reconcile():
    s0, s1 = _snaps()
    scalar = explain(s0, s1, config=PnLExplainConfig(greeks_method="numerical"))
    bucketed = explain(s0, s1, config=PnLExplainConfig(greeks_method="numerical", bucketed=True))
    terms = [r.term for r in bucketed.rows if r.method is ExplainMethod.TAYLOR]
    assert "vega" not in terms and "rho" not in terms
    vega_rows = [r for r in bucketed.rows if r.term.startswith("vega.") and r.kind is RowKind.COMPONENT]
    rho_rows = [r for r in bucketed.rows if r.term.startswith("rho.") and r.kind is RowKind.COMPONENT]
    assert [r.term for r in vega_rows] == ["vega.0.5", "vega.1", "vega.2"]        # each pillar exactly once
    assert [r.term for r in rho_rows] == ["rho.0.5", "rho.1", "rho.2"]
    assert len(terms) == len(set(terms))
    assert all("tenor" in r.moves for r in vega_rows + rho_rows)
    scalar_vega = [r for r in scalar.rows if r.term == "vega"][0].pnl
    assert sum(r.pnl for r in vega_rows) == pytest.approx(scalar_vega, rel=5e-2, abs=1e-6)
    par = [r for r in bucketed.rows if r.term == "rate_keyrate.parallel"][0]
    assert par.kind is RowKind.INFORMATIONAL and "sum_of_buckets" in par.metadata
    assert bucketed.reconcile(ExplainMethod.TAYLOR) == pytest.approx(0.0, abs=1e-12)
    # the vega row order: bucket rows sit where the scalar vega row would have been (after delta)
    assert terms.index("vega.0.5") == terms.index("delta") + 1


def test_bucketed_honours_configured_bumps():
    from quantark.asset.equity.param import BumpConfig, EngineParams
    s0, s1 = _snaps()
    params = EngineParams(bump_config=BumpConfig(vol_bump=0.02, rate_bump=0.0005))
    res = explain(s0, s1, config=PnLExplainConfig(greeks_method="numerical", bucketed=True, params=params))
    vega_rows = [r for r in res.rows if r.term.startswith("vega.")]
    rho_rows = [r for r in res.rows if r.term.startswith("rho.") and r.kind is RowKind.COMPONENT]
    assert all(r.metadata["bump_size"] == pytest.approx(0.02) for r in vega_rows)
    assert all(r.metadata["bump_size"] == pytest.approx(0.0005) for r in rho_rows)


def test_bucketed_rejects_mismatched_structures():
    s0, s1 = _snaps()
    flat = ValuationSnapshot(s1.product, BlackScholesEngine(), PricingEnvironment(
        spot_quote=SpotQuote(spot=101.0), vol_surface=__import__("quantark.param", fromlist=["FlatVolSurface"]).FlatVolSurface(0.22),
        rate_curve=FlatRateCurve(rate=0.032), div_yield=ContinuousDividendYield(div_yield=0.01), valuation_date=T1), date=T1)
    with pytest.raises(ValidationError):
        explain(s0, flat, config=PnLExplainConfig(bucketed=True))
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST test/test_pnlexplain_bucketed.py -q`
Expected: FAIL (`vega` still present; no `ValidationError`)

- [ ] **Step 3: Implement `bucketed.py` and wire it**

```python
# quantark/pnlexplain/equity/bucketed.py
"""Opt-in tenor-vega / key-rate rows (spec §7.6)."""
from __future__ import annotations

from typing import Any, List, Tuple

from quantark.asset.equity.riskmeasures.bucketed_greeks import BucketedGreekCoordinate, BucketedGreeksRequest
from quantark.param import TermStructureVolSurface
from quantark.param.rrf.rate_curve import InterpolatedRateCurve
from quantark.pnlexplain.base import ExplainMethod, ExplainRow, Factor, RowKind
from quantark.pnlexplain.equity.scenario import ScenarioCache
from quantark.util.exceptions import ValidationError


def _pillars(obj: Any) -> Tuple[float, ...]:
    """The pillar grid of a term structure (vol: times; rate: tenors), as floats."""
    for attr in ("times", "tenors", "pillars"):
        grid = getattr(obj, attr, None)
        if grid is not None:
            return tuple(float(t) for t in (grid.keys() if isinstance(grid, Mapping) else grid))
    raise ValidationError(f"bucketed mode: cannot read the pillar grid of {type(obj).__name__}")


def _check_pair(name: str, a: Any, b: Any, cls) -> bool:
    """True when both snapshots carry a `cls` term structure ON THE SAME PILLAR GRID.

    Rule (spec §7.6): bucketed rows are read on the t0 grid and must sum to the
    scalar row; no interpolation between grids preserves that identity, so a
    class mismatch or a pillar mismatch between t0 and t1 is rejected.
    """
    is_a, is_b = isinstance(a, cls), isinstance(b, cls)
    if is_a != is_b:
        raise ValidationError(f"bucketed mode: {name} is a term structure on one side only")
    if is_a and _pillars(a) != _pillars(b):
        raise ValidationError(
            f"bucketed mode: {name} pillar grids differ between t0 and t1 "
            f"({_pillars(a)} vs {_pillars(b)}); re-mark both snapshots on one grid"
        )
    return is_a


def bucketed_rows(cache: ScenarioCache, calc: Any, bump: Any, level: str
                  ) -> Tuple[Mapping[Factor, Tuple[ExplainRow, ...]], FrozenSet[Factor]]:
    """Rows partitioned by factor + the covered factors (⊆ {VOL, RATE})."""
    snap0, snap1, moves = cache.snap0, cache.snap1, cache.moves
    e0, e1 = snap0.pricing_env, snap1.pricing_env
    q, K, T = snap0.quantity, moves.coordinate.reference_strike, moves.coordinate.tenor_t1
    coords = []
    if _check_pair("vol_surface", e0.vol_surface, e1.vol_surface, TermStructureVolSurface) \
            and Factor.VOL in moves.coordinate.applicable:
        coords.append(BucketedGreekCoordinate.VOL_TENOR_VEGA)
    if _check_pair("rate_curve", e0.rate_curve, e1.rate_curve, InterpolatedRateCurve) \
            and Factor.RATE in moves.coordinate.applicable:
        coords.append(BucketedGreekCoordinate.RATE_KEYRATE)
    if not coords:
        return {}, frozenset()
    request = BucketedGreeksRequest(coordinates=tuple(coords), vol_bump=float(bump.vol_bump),
                                    rate_bump=float(bump.rate_bump))
    result = calc.calculate_bucketed_greeks(snap0.product, e0, cache.bump_engine_t0, request)
    by_factor: Dict[Factor, List[ExplainRow]] = {Factor.VOL: [], Factor.RATE: []}
    for pt in result.points:
        tau = float(pt.maturity)
        if pt.coordinate is BucketedGreekCoordinate.VOL_TENOR_VEGA:
            d = float(e1.get_vol(K, tau)) - float(e0.get_vol(K, tau))
            g = q * float(pt.derivative)
            by_factor[Factor.VOL].append(ExplainRow(
                factor=Factor.VOL, term=f"vega.{tau:g}", method=ExplainMethod.TAYLOR, kind=RowKind.COMPONENT,
                level=level, pnl=g * d, greek=g, cash_greek=g * 0.01,
                moves={"vol_pts": d * 100.0, "tenor": tau},
                metadata={"pillar": tau, "bump_size": float(pt.bump_size), "difference_mode": pt.difference_mode}))
        elif pt.coordinate is BucketedGreekCoordinate.RATE_KEYRATE:
            if pt.name == "rate_keyrate.parallel":
                d = float(e1.get_rate(T)) - float(e0.get_rate(T)) if T is not None else 0.0
                g = q * float(pt.derivative)
                by_factor[Factor.RATE].append(ExplainRow(
                    factor=Factor.RATE, term="rate_keyrate.parallel", method=ExplainMethod.TAYLOR,
                    kind=RowKind.INFORMATIONAL, level=level, pnl=g * d, greek=g, cash_greek=g * 0.01,
                    moves={"rate_pct": d * 100.0}, metadata=dict(pt.metadata)))
                continue
            d = float(e1.get_rate(tau)) - float(e0.get_rate(tau))
            g = q * float(pt.derivative)
            by_factor[Factor.RATE].append(ExplainRow(
                factor=Factor.RATE, term=f"rho.{tau:g}", method=ExplainMethod.TAYLOR, kind=RowKind.COMPONENT,
                level=level, pnl=g * d, greek=g, cash_greek=g * 0.01,
                moves={"rate_pct": d * 100.0, "tenor": tau},
                metadata={"pillar": tau, "bump_size": float(pt.bump_size), "difference_mode": pt.difference_mode}))
    rows = {f: tuple(r) for f, r in by_factor.items() if r}
    return rows, frozenset(rows)
```
(`Dict`, `FrozenSet`, `List`, `Mapping` from `typing`.) The key-rate parallel row is emitted after the
`rho.<τ>` rows because `calculate_bucketed_greeks` appends it last.

Wiring in `taylor_rows` (Task 8): delete the gate `if config.bucketed: raise NotImplementedError(...)`;
right after the greeks are computed (inside `if not terminal:`) add
```python
        if config.bucketed:
            from quantark.pnlexplain.equity.bucketed import bucketed_rows
            bucket_rows, covered = bucketed_rows(cache, calc, bump, level)
```
(initialise `bucket_rows, covered = {}, frozenset()` before the `if not terminal:` block), and at the top
of the `for name in terms:` loop body add
```python
        if (name == "vega" and Factor.VOL in covered) or (name == "rho" and Factor.RATE in covered):
            for row in bucket_rows[TERM_FACTOR[name]]:
                rows.append(row)
                if row.kind is RowKind.COMPONENT:
                    explained += row.pnl
            continue
```
so the bucket rows take the scalar row's slot exactly once. Delete `test_bucketed_is_gated_until_task_13`
from `test/test_pnlexplain_taylor.py`.

- [ ] **Step 4: Run tests**

Run: `PYTEST test/test_pnlexplain_bucketed.py test/test_pnlexplain_taylor.py -q`
Expected: all passed (3 bucketed + 6 Taylor; `LinearRateCurve(pillars=[(tenor, rate), ...])` is the real constructor).

- [ ] **Step 5: Commit**

```bash
git add quantark/pnlexplain/equity/bucketed.py quantark/pnlexplain/equity/taylor.py test/test_pnlexplain_bucketed.py test/test_pnlexplain_taylor.py
git commit -m "feat(pnlexplain): opt-in tenor-vega / key-rate bucket rows (P5)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_019fnLipZtAJAPAQJLLEXQMb"
```

---

### Task 14: Support-matrix acceptance cases, exports, README, demo

**Files:**
- Create: `quantark/pnlexplain/README.md`, `example/pnl_explain_demo.py`, `test/test_pnlexplain_matrix.py`
- Modify: `quantark/pnlexplain/__init__.py` (final export list), `CLAUDE.md` (root, local-only: one row in the Supporting Modules table)

**Interfaces:**
- Consumes everything above. Produces the public import surface of spec §5.6 verbatim.

- [ ] **Step 1: Write the failing tests**

```python
# test/test_pnlexplain_matrix.py
"""Spec §12 support matrix + test 6b (MC common random numbers) + test 10 (frame determinism)."""
import subprocess
import sys
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import (
    AmericanOptionAnalyticalEngine, DeltaOneEngine, DigitalOptionAnalyticalEngine,
)
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.deltaone import Futures, SpotInstrument
from quantark.asset.equity.product.option.american_option import AmericanOption
from quantark.asset.equity.product.option.digital_option import CashOrNothingDigitalOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
import quantark.pnlexplain as pnlexplain_pkg
from quantark.pnlexplain import (
    BookSnapshot, ExplainMethod, ExplainRow, ExplainTrade, Factor, FactorCoordinate, FactorMoves,
    LifecycleTransition, MARKET_FACTORS, PnLExplainConfig, PnLExplainResult,
    PortfolioExplainResult, PositionExplainResult, PositionSnapshot, RowKind, ValuationSnapshot,
    ValueBreakdown, contract_fingerprint, explain, explain_portfolio, explain_position,
    lifecycle_fingerprint, value,
)
# `PnLExplainRecorder` and the equity re-exports are looked up INSIDE the export tests via
# `pnlexplain_pkg` so this module still collects before Step 3 lands them.
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot as VS
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.deltaone_enums import DeltaOneType
from quantark.util.enum.engine_enums import AmericanAnalyticalMethod, EngineType

sys.path.insert(0, __import__("os").path.dirname(__file__))
from test_pnlexplain_lifecycle import _snowball  # noqa: E402

T0, T1 = datetime(2026, 6, 26), datetime(2026, 6, 29)


def _env(spot, date, vol=0.2):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=vol),
                              rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.01),
                              valuation_date=date)


def _tol(x):
    return 1e-10 * max(1.0, abs(x))


@pytest.mark.parametrize("family,make", [
    ("american", lambda m: (AmericanOption(strike=100.0, option_type=OptionType.PUT, maturity=m),
                            AmericanOptionAnalyticalEngine(method=EngineType.ANALYTICAL(AmericanAnalyticalMethod.BS93)))),
    ("digital", lambda m: (CashOrNothingDigitalOption(strike=100.0, option_type=OptionType.CALL, maturity=m, payout=10.0),
                           DigitalOptionAnalyticalEngine())),
    ("futures", lambda m: (Futures(underlying="X", multiplier=300.0, maturity=m), DeltaOneEngine())),
])
def test_family_reconciles_both_methods(family, make):
    p0, eng = make(1.0)
    p1, _ = make(1.0 - 3 / 365)
    s0 = ValuationSnapshot(p0, eng, _env(100.0, T0), date=T0)
    s1 = ValuationSnapshot(p1, eng, _env(102.0, T1, vol=0.21), date=T1)
    res = explain(s0, s1)
    for m in (ExplainMethod.WATERFALL, ExplainMethod.TAYLOR):
        assert res.reconcile(m) == pytest.approx(0.0, abs=_tol(res.total_pnl)), family
    if family == "futures":
        assert res.by_factor(ExplainMethod.WATERFALL)["vol"] == 0.0
        assert [r for r in res.rows if r.term == "vega"][0].greek is None


def test_spot_instrument_is_delta_only():
    spot = SpotInstrument(underlying="X", deltaone_type=DeltaOneType.STOCK)
    s0 = ValuationSnapshot(spot, DeltaOneEngine(), _env(100.0, T0), date=T0, quantity=7.0)
    s1 = ValuationSnapshot(spot, DeltaOneEngine(), _env(102.0, T1), date=T1, quantity=7.0)
    res = explain(s0, s1)
    assert res.by_factor(ExplainMethod.WATERFALL)["spot"] == pytest.approx(14.0)
    assert res.by_factor(ExplainMethod.TAYLOR)["spot"] == pytest.approx(14.0)
    assert res.unexplained == pytest.approx(0.0, abs=1e-12)


def test_mc_common_random_numbers_make_the_waterfall_exact():
    eng = SnowballMCEngine(MCParams(num_paths=4000, time_steps=32, seed=7))
    p0 = _snowball()
    p1 = _snowball()
    p1.maturity = 1.0 - 3 / 365
    s0 = ValuationSnapshot(p0, eng, _env(100.0, T0, vol=0.22), date=T0, quantity=-2.0)
    s1 = ValuationSnapshot(p1, eng, _env(101.0, T1, vol=0.23), date=T1, quantity=-2.0)
    cfg = PnLExplainConfig(methods=(ExplainMethod.WATERFALL, ExplainMethod.TAYLOR), stencil="first_order")
    res = explain(s0, s1, config=cfg)
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(res.total_pnl))
    # No lifecycle event: the t1 endpoint and state(all) are the same market priced on the same
    # seed-frozen context, so the event row is zero. Independent MC noise would show up here.
    event = [r for r in res.rows if r.factor is Factor.LIFECYCLE_EVENT][0]
    assert event.pnl == pytest.approx(0.0, abs=_tol(res.total_pnl))
    assert res.unexplained is not None          # reported, not asserted small
    again = explain(s0, s1, config=cfg)          # seeded: row for row identical
    assert [(r.term, r.pnl) for r in again.rows] == [(r.term, r.pnl) for r in res.rows]


def test_frame_order_is_hash_seed_independent(tmp_path):
    script = tmp_path / "order.py"
    script.write_text(
        "import json\n"
        "from datetime import datetime\n"
        "from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine\n"
        "from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption\n"
        "from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote\n"
        "from quantark.param.div import ContinuousDividendYield\n"
        "from quantark.pnlexplain import ValuationSnapshot, explain\n"
        "from quantark.priceenv import PricingEnvironment\n"
        "from quantark.util.enum import OptionType\n"
        "def env(s, d, v=0.2):\n"
        "    return PricingEnvironment(spot_quote=SpotQuote(spot=s), vol_surface=FlatVolSurface(volatility=v),\n"
        "        rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.01), valuation_date=d)\n"
        "eng = BlackScholesEngine()\n"
        "s0 = ValuationSnapshot(EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0), eng, env(100.0, datetime(2026,6,26)), date=datetime(2026,6,26))\n"
        "s1 = ValuationSnapshot(EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0-3/365), eng, env(103.0, datetime(2026,6,29), 0.22), date=datetime(2026,6,29))\n"
        "res = explain(s0, s1)\n"
        "print(json.dumps(res.to_dict(), default=str))\n"
        "from quantark.pnlexplain import BookSnapshot, PositionSnapshot, explain_portfolio\n"
        "b0 = BookSnapshot(datetime(2026,6,26), {'a': PositionSnapshot('a', 'X', s0)}, {'X': s0.pricing_env})\n"
        "b1 = BookSnapshot(datetime(2026,6,29), {'a': PositionSnapshot('a', 'X', s1)}, {'X': s1.pricing_env})\n"
        "print(explain_portfolio(b0, b1).to_frame().to_csv(index=False))\n"
    )
    outs = set()
    for seed in ("0", "1", "12345"):
        out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=True,
                             env={**__import__('os').environ, "PYTHONHASHSEED": seed})
        outs.add(out.stdout.strip())
    assert len(outs) == 1


EXPECTED_ALL = [
    "BookSnapshot", "ExplainMethod", "ExplainRow", "ExplainTrade", "FRAME_COLUMNS", "Factor",
    "FactorCoordinate", "FactorMoves", "LifecycleTransition", "MARKET_FACTORS", "MOVE_KEYS",
    "PnLExplainConfig", "PnLExplainRecorder", "PnLExplainResult", "PortfolioExplainResult",
    "PositionExplainResult", "PositionSnapshot", "QuotedLegSnapshot", "RECON_COLUMNS",
    "ReplayPnLExplainRecorder", "RowKind", "ValuationSnapshot", "ValueBreakdown", "component_sum",
    "contract_fingerprint", "explain", "explain_portfolio", "explain_position", "explain_quoted_leg",
    "lifecycle_fingerprint", "make_total_row", "rows_to_frame", "value",
]


def test_public_exports_match_spec():
    assert list(pnlexplain_pkg.__all__) == EXPECTED_ALL          # exact, ordered surface
    for name in EXPECTED_ALL:
        assert getattr(pnlexplain_pkg, name) is not None, name


def test_equity_subpackage_exports():
    import importlib
    eq = importlib.import_module("quantark.pnlexplain.equity")
    for name in ("BookSnapshot", "ExplainTrade", "LifecycleTransition", "PnLExplainRecorder",
                 "PositionSnapshot", "QuotedLegSnapshot", "ReplayPnLExplainRecorder", "ValuationSnapshot",
                 "explain", "explain_portfolio", "explain_position", "explain_quoted_leg", "value"):
        assert hasattr(eq, name), name
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST test/test_pnlexplain_matrix.py -q`
Expected: the module COLLECTS (nothing missing is imported at module scope); `test_public_exports_match_spec` and `test_equity_subpackage_exports` FAIL on their assertions (`PnLExplainRecorder` and the equity re-exports are not exported yet); every other test passes. The constructors used are the real ones: `AmericanOption(strike=, option_type=, maturity=)`, `CashOrNothingDigitalOption(strike=, option_type=, maturity=, payout=)`, `Futures(underlying=, multiplier=, maturity=)`, `SpotInstrument(underlying=, deltaone_type=)`.

- [ ] **Step 3: Final exports**

`quantark/pnlexplain/__init__.py` `__all__` must be EXACTLY `EXPECTED_ALL` from the test above (the spec §5.6 names plus `QuotedLegSnapshot`, `explain_quoted_leg`, `ReplayPnLExplainRecorder`, `FRAME_COLUMNS`, `MOVE_KEYS`, `component_sum`, `make_total_row`, `rows_to_frame`, `RECON_COLUMNS`, in that sorted order). Import `PnLExplainRecorder`, `ReplayPnLExplainRecorder`, `RECON_COLUMNS` from `quantark.pnlexplain.equity.recorder`.

`quantark/pnlexplain/equity/__init__.py` (created empty in Task 3) re-exports the equity surface:

```python
"""Equity PnL explain."""
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot, is_terminal, value  # noqa: F401
from quantark.pnlexplain.equity.coordinate import FactorCoordinate, resolve_coordinate  # noqa: F401
from quantark.pnlexplain.equity.factor_diff import FactorMoves, build_factor_moves, validate_pair  # noqa: F401
from quantark.pnlexplain.equity.fingerprints import (  # noqa: F401
    calendars_equal, check_contract_roll, contract_fingerprint, lifecycle_fingerprint,
)
from quantark.pnlexplain.equity.lifecycle import LifecycleTransition, resolve_transition  # noqa: F401
from quantark.pnlexplain.equity.explain import explain  # noqa: F401
from quantark.pnlexplain.equity.trades import ExplainTrade  # noqa: F401
from quantark.pnlexplain.equity.portfolio import (  # noqa: F401
    BookSnapshot, PortfolioExplainResult, PositionExplainResult, PositionSnapshot, QuotedLegSnapshot,
    explain_portfolio, explain_position, explain_quoted_leg,
)
from quantark.pnlexplain.equity.recorder import (  # noqa: F401
    RECON_COLUMNS, PnLExplainRecorder, ReplayPnLExplainRecorder, trade_kind,
)
```
with a matching `__all__`.

- [ ] **Step 4: README and demo**

`quantark/pnlexplain/README.md` sections: What it does (two methods, one factor model); Quick start (the two-snapshot call from the demo, 15 lines); Reading a result (row kinds and the additivity contract, cash columns, `moves` keys); Lifecycle days (`LifecycleTransition`, tombstones, what the event row means, receivable payments accrue in TIME); Position/portfolio (unit contract, `ExplainTrade` sign convention, contract-specific ids for rolls, single-currency books); Backtests (`pnl_explain=` on both configs, `explain_df` / `explain_reconciliation_df`, `expected` vs `expected_states` and the simple-executor entry-price quirk); Vol-model engines (surface swap ~0, `model` row); Conventions table (copy spec §7.5); Limitations (untracked products have no cashflow ledger; engines mutated in place; TradingClock-wrapped environments untested; bucketed is experimental).

`example/pnl_explain_demo.py`: build a vanilla and a barrier (analytical engines) and a snowball (PDE fast) across a Friday→Monday gap with spot/vol/rate moves; print `res.to_frame()[["method","kind","factor","term","pnl","greek","cash_greek","spot_return","vol_pts","days"]]` for each; then a KO day using the `_find_event_day` pattern from `test_pnlexplain_lifecycle.py` (copy the helper, do not import tests) showing the event row; finish with `explain_portfolio` over the three positions and print the reconciliation dict.

Root `CLAUDE.md` Supporting Modules table (local, untracked): add
`| PnL explain | \`quantark/pnlexplain/\` | Two-snapshot PnL explain: full-revaluation waterfall + greeks-based Taylor on one factor model, lifecycle event term, portfolio aggregation, daily series from both backtests; see \`quantark/pnlexplain/README.md\` |`.

- [ ] **Step 5: Run everything touched**

Run: `PYTEST test/test_pnlexplain_*.py test/test_greeks_registry.py test/test_backtest_lifecycle.py test/test_multi_greek_backtest.py test/replay_golden test/test_replay_greeks_failclosed.py -q`
Expected: all passed.
Then the full suite: `PYTHONPATH=/Users/fuxinyao/quant-ark/.claude/worktrees/pnl-explain /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 -q -x --ignore=test/mo_volmodels` (the mo suite rewrites sample data and is slow; run it separately only if a touched file is imported there). Expected: green. Run `PYTHONPATH=/Users/fuxinyao/quant-ark/.claude/worktrees/pnl-explain /Users/fuxinyao/quant-ark/.venv/bin/python example/pnl_explain_demo.py > /private/tmp/claude-501/-Users-fuxinyao-quant-ark/8cbdbf24-eb5a-4fed-b544-bc4f1df15336/scratchpad/demo_output.txt` once; the commit below embeds its first 20 lines.

- [ ] **Step 6: Commit (demo output in the body)**

```bash
git add quantark/pnlexplain/__init__.py quantark/pnlexplain/equity/__init__.py quantark/pnlexplain/README.md example/pnl_explain_demo.py test/test_pnlexplain_matrix.py
git commit -m "docs(pnlexplain): README, demo, public exports; support-matrix acceptance tests" -m "Demo output (first 20 lines of example/pnl_explain_demo.py):" -m "$(head -20 /private/tmp/claude-501/-Users-fuxinyao-quant-ark/8cbdbf24-eb5a-4fed-b544-bc4f1df15336/scratchpad/demo_output.txt)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_019fnLipZtAJAPAQJLLEXQMb"
```
(`CLAUDE.md` is untracked in this repo; edit it locally, never `git add` it. If the sandbox refuses the `$(...)` substitution, paste the 20 lines literally into the third `-m` argument.)

---

## Self-review notes (run after Task 14)

1. **Spec coverage**: §5.1–5.6 → Tasks 1–5, 10; §6 → Tasks 5–6; §7.1 → Task 7; §7.2–7.5 → Task 8; §7.6 → Task 13; §8 → Tasks 5, 9; §9 → Task 10; §10 → Tasks 11–12; §11 error table → the `ValidationError` tests spread across Tasks 2, 3, 4, 5, 9, 10, 11, 12, 13; §12 tests 1–10 → Tasks 5, 6, 5, 8, 9, 11/12, 8, 12, 13, 14; §13 phases → task order (bucketed P5 is Task 13, the final acceptance pass is Task 14); §14 → Tasks 7, 9, 11, 12 gates.
2. **Type consistency** to re-check while executing: `ExplainRow.moves` keys ⊆ `MOVE_KEYS` (the bucket rows use `tenor`); `PositionExplainResult.coordinate` filled from `PnLExplainResult.metadata["coordinate"]` (set in `explain.py` since Task 5); `ScenarioCache.value_for` normalises keys so the pricing-count test in Task 5 holds; `trade_kind` must cover every `trade_type` both executors and the replay emit (unknown types raise); gates (`NotImplementedError`) exist only between the task that adds a config option and the task that implements it, and each implementing task deletes its gate and gate test.
3. **Known risks to report, not paper over**: the equity simple `HedgeExecutor` entry-price quirk (reported via `gap_states`); ledger vs replay receivable discounting (Task 12 debug note); TradingClock-wrapped environments are untested; MODEL is identity-detected, so two equivalent engine instances read as a model change (documented in the README).



