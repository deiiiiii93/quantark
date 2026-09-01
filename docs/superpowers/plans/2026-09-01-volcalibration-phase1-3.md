# quantark.volcalibration — Phases 1–3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the importable half of `quantark.volcalibration` — normalized quote snapshot → `QuoteSet` → SABR-smoothed, admission-checked IV surface artifact → calibrated vol model — with the existing `example/mo_volmodels/` suite still green through a re-export shim.

**Architecture:** Pure typed stage functions with one pluggable point (the quote normalizer). `snapshot.py` validates the canonical envelope and lifts today's two legacy MO shapes into it. `normalize/listed.py` turns strike-quoted call/put prices into `QuoteSet` via put-call parity, an OTM wing filter and call-equivalent Black-IV inversion. `surface.py` assembles the rectangular grid and applies SABR smoothing with calendar projection. `admission.py` runs the static-arbitrage gate. `store.py` serializes artifacts and manifest records byte-compatibly with today's output. `calibrate.py` is `quantark/volmodels/calibration.py` relocated. No runner, no CLI, no `CalibrationSet` — those are Phases 4–7 in a follow-on plan.

**Tech Stack:** Python 3.10+, NumPy, SciPy (via existing SABR/Heston calibrators), pytest with `-n auto --dist worksteal`. Library code only — no network, no vendor formats.

**Spec:** `docs/superpowers/specs/2026-09-01-volcalibration-module-design.md`

## Global Constraints

Every task's requirements implicitly include all of these.

- **Artifact bytes must not change.** The artifact sha256 is over raw file bytes and feeds the calibration cache key; 766 admitted MO surfaces and a warm cache depend on it. New provenance goes in manifest records, never the artifact body. (Spec §5.3)
- **Fail-closed, no fallbacks.** A failure raises `ValidationError` or `NumericalError` naming the date and the reason. Never substitute a flat vol, a floored value, or a guessed field. If the maths cannot be implemented correctly, write a `# TODO` and stop.
- **Numerical helpers only.** Use `quantark.util.numerical` (`is_zero`, `is_close`, `safe_log`, `safe_exp`, `safe_divide`, `Tolerance`). Never a raw float comparison or a hardcoded tolerance.
- **Canonical imports only.** `quantark.*`. Never the deprecated flat names (`asset`, `util`, `param`).
- **Dependency direction.** Nothing under `quantark/volcalibration/` may import `quantark.backtest` or `quantark.asset`. Allowed: `quantark.volmodels`, `quantark.param`, `quantark.priceenv`, `quantark.util`.
- **Never `git add example/`.** The mo_volmodels test runs rewrite two tracked sample files with float churn. `git checkout -- example/` before staging.
- **Python is `/Users/fuxinyao/quant-ark/.venv/bin/python`** — the venv lives in the *main repo*, not the worktree; there is no `.venv/` here. Because that editable install resolves `quantark` to the main repo, every test command is prefixed `PYTHONPATH=$PWD` to shadow it with worktree source. Run all commands from the worktree root.
- **Pre-resolved facts** (verified against the tree while writing this plan — do not re-derive):
  - `EXTRAPOLATION_POLICY = "flat_total_variance"`
  - `fd1_nonuniform`, `fd2_nonuniform`, `is_positive`, `safe_log` import from `quantark.util.numerical`; `Tolerance` from `quantark.util.numerical.constants`
  - `ValidationError` is **not** a subclass of `ValueError` (it derives from `QuantArkException`), so the Task 5 shim's re-raise is required
  - `test/replay_golden/fixtures.py` provides `DATE_A = date(2024, 1, 2)` and `write_localvol_history(root)`
- **Open interest is `oi`** in both existing MO snapshot shapes. The canonical schema keeps that name.

---

## File Structure

**Created:**

| File | Responsibility |
|---|---|
| `quantark/volcalibration/__init__.py` | Public re-exports |
| `quantark/volcalibration/quotes.py` | `IvNode`, `ExpiryQuotes`, `QuoteSet` — the convention-neutral convergence point |
| `quantark/volcalibration/snapshot.py` | `QuoteSnapshot` schema, validation, legacy lifters, price-field resolution |
| `quantark/volcalibration/normalize/__init__.py` | `QuoteNormalizer` protocol, normalizer registry |
| `quantark/volcalibration/normalize/listed.py` | Parity, OTM filter, IV inversion for strike-quoted books |
| `quantark/volcalibration/surface.py` | Grid assembly, SABR slice fit, calendar projection |
| `quantark/volcalibration/admission.py` | `AdmissionReason` enum, `AdmissionError`, static-arbitrage gate |
| `quantark/volcalibration/store.py` | Artifact serialization, manifest records, legacy migration |
| `quantark/volcalibration/calibrate.py` | Relocated `VolModelCalibrator` |
| `quantark/volcalibration/config.py` | Relocated `VolModelCalibrationConfig`, `HESTON_PRESETS` |
| `test/volcalibration/*` | Library-level tests |

**Modified:**

| File | Change |
|---|---|
| `example/mo_volmodels/_mo_common.py` | Becomes a re-export shim over the library |
| `quantark/volmodels/calibration.py` | Becomes a deprecation shim re-exporting from `volcalibration.calibrate` |
| `quantark/backtest/replay/config.py` | Imports `VolModelCalibrationConfig` from `volcalibration.config` |

**Note on `quotes.py`:** spec §4.1 does not name a file for the `QuoteSet` dataclasses. They go in `quotes.py` rather than inside `normalize/__init__.py` so that `surface.py` and `store.py` can import them without importing any normalizer.

---

## Phase 1 — Contracts and the listed normalizer

### Task 1: Canonical `QuoteSnapshot` with legacy lifters

**Files:**
- Create: `quantark/volcalibration/__init__.py`, `quantark/volcalibration/snapshot.py`
- Create: `test/volcalibration/__init__.py`, `test/volcalibration/test_snapshot.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `PRICE_FIELD_SETTLEMENT = "settlement"`, `PRICE_FIELD_MID_OR_LAST = "mid_or_last"`
  - `class QuoteSnapshot` with attributes `schema_version: int`, `convention: str`, `trade_date: date`, `spot: float`, `symbol: str`, `price_field: str`, `source: dict`, `expiries: tuple[dict, ...]`
  - `QuoteSnapshot.from_payload(payload: dict) -> QuoteSnapshot`
  - `QuoteSnapshot.from_legacy_live(payload: dict, *, trade_date, symbol, source_sha256=None) -> QuoteSnapshot`
  - `QuoteSnapshot.from_legacy_settlement(payload: dict, *, trade_date, spot, symbol, source_sha256=None) -> QuoteSnapshot`
  - `QuoteSnapshot.quote_price(quote: dict) -> float`

**Context you need:** Two incompatible legacy shapes exist. The live snapshot (`example/mo_volmodels/data/mo_snapshot_sample.json`) is `{fetched_at, market_open, underlying:{spot}, expiries:[{expiry_date, T_years, quotes:[{strike,type,last,bid,ask,volume,oi}]}]}` — no trade date, no schema version, no price field. The settlement snapshot (`example/mo_volmodels/data/mo_settlement_snapshot_20260430.json`) is `{schema_version, source_class, price_field, expiries:[{contract_month, expiry_date, T_years, calendar_days, quotes:[{contract,strike,type,settlement,close,volume,oi,exchange_delta}]}], …}` — **and carries no spot at all**. Spot comes from a separate CSV. That is why the lifters take `spot` and `trade_date` as required keyword arguments rather than inferring them.

- [ ] **Step 1: Write the failing test**

Create `test/volcalibration/__init__.py` (empty) and `test/volcalibration/test_snapshot.py`:

```python
import json
from datetime import date
from pathlib import Path

import pytest

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.snapshot import (
    PRICE_FIELD_MID_OR_LAST,
    PRICE_FIELD_SETTLEMENT,
    QuoteSnapshot,
)

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "example/mo_volmodels/data/mo_snapshot_sample.json"
SETTLE = ROOT / "example/mo_volmodels/data/mo_settlement_snapshot_20260430.json"


def test_lift_live_snapshot():
    snap = QuoteSnapshot.from_legacy_live(
        json.loads(LIVE.read_text()),
        trade_date=date(2026, 7, 6),
        symbol="000852.SH",
    )
    assert snap.convention == "listed_strike"
    assert snap.price_field == PRICE_FIELD_MID_OR_LAST
    assert snap.spot == 6000.0
    assert len(snap.expiries) == 4
    assert all(e["T_years"] > 0.0 for e in snap.expiries)


def test_lift_settlement_snapshot_requires_spot():
    payload = json.loads(SETTLE.read_text())
    snap = QuoteSnapshot.from_legacy_settlement(
        payload, trade_date=date(2026, 4, 30), spot=6000.0, symbol="000852.SH"
    )
    assert snap.price_field == PRICE_FIELD_SETTLEMENT
    assert snap.spot == 6000.0
    with pytest.raises(TypeError):
        QuoteSnapshot.from_legacy_settlement(  # spot is not optional
            payload, trade_date=date(2026, 4, 30), symbol="000852.SH"
        )


def test_price_field_resolution():
    live = QuoteSnapshot.from_legacy_live(
        json.loads(LIVE.read_text()), trade_date=date(2026, 7, 6), symbol="X"
    )
    # mid when both sides live
    assert live.quote_price({"bid": 2.0, "ask": 4.0, "last": 99.0}) == 3.0
    # falls back to last only when a side is missing or non-positive
    assert live.quote_price({"bid": 0.0, "ask": 4.0, "last": 99.0}) == 99.0

    settle = QuoteSnapshot.from_legacy_settlement(
        json.loads(SETTLE.read_text()),
        trade_date=date(2026, 4, 30),
        spot=6000.0,
        symbol="X",
    )
    assert settle.quote_price({"settlement": 12.5, "close": 99.0}) == 12.5


def test_declared_price_field_absent_is_rejected():
    snap = QuoteSnapshot.from_legacy_settlement(
        json.loads(SETTLE.read_text()),
        trade_date=date(2026, 4, 30),
        spot=6000.0,
        symbol="X",
    )
    with pytest.raises(ValidationError, match="price_field_mismatch"):
        snap.quote_price({"last": 1.0, "bid": 0.9, "ask": 1.1})


def test_from_payload_rejects_unknown_convention():
    with pytest.raises(ValidationError, match="convention"):
        QuoteSnapshot.from_payload(
            {
                "schema_version": 1,
                "convention": "nonsense",
                "trade_date": "2026-07-06",
                "underlying": {"symbol": "X", "spot": 1.0},
                "source": {"price_field": PRICE_FIELD_MID_OR_LAST},
                "expiries": [],
            }
        )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/test_snapshot.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'quantark.volcalibration'`

- [ ] **Step 3: Write the implementation**

Create `quantark/volcalibration/__init__.py`:

```python
"""Market-data-aware vol-model calibration: quotes -> IV surface -> LV/Heston/SLV.

Composes the asset-neutral kernels in ``quantark.volmodels`` with market data
from ``quantark.param``.  Nothing here imports ``quantark.backtest`` or
``quantark.asset``.
"""

from quantark.volcalibration.snapshot import (
    PRICE_FIELD_MID_OR_LAST,
    PRICE_FIELD_SETTLEMENT,
    QuoteSnapshot,
)

__all__ = [
    "QuoteSnapshot",
    "PRICE_FIELD_SETTLEMENT",
    "PRICE_FIELD_MID_OR_LAST",
]
```

Create `quantark/volcalibration/snapshot.py`:

```python
"""Canonical quote-snapshot envelope and lifters for the legacy MO shapes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Mapping, Optional, Tuple

from quantark.util.exceptions import ValidationError

SCHEMA_VERSION = 1

CONVENTION_LISTED = "listed_strike"
CONVENTION_FX_DELTA = "fx_delta"
CONVENTIONS = (CONVENTION_LISTED, CONVENTION_FX_DELTA)

PRICE_FIELD_SETTLEMENT = "settlement"
PRICE_FIELD_MID_OR_LAST = "mid_or_last"
PRICE_FIELDS = (PRICE_FIELD_SETTLEMENT, PRICE_FIELD_MID_OR_LAST)


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
    raise ValidationError(f"Cannot interpret {value!r} as a trade date")


def _positive(value, what: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{what} must be numeric, got {value!r}") from exc
    if not math.isfinite(out) or out <= 0.0:
        raise ValidationError(f"{what} must be positive and finite, got {out}")
    return out


@dataclass(frozen=True)
class QuoteSnapshot:
    """One trading date's option quotes in the canonical envelope."""

    schema_version: int
    convention: str
    trade_date: date
    symbol: str
    spot: float
    price_field: str
    source: Mapping[str, Any]
    expiries: Tuple[Dict[str, Any], ...]

    # ---------------------------------------------------------------- build
    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "QuoteSnapshot":
        """Validate a canonical snapshot payload; fail closed on any violation."""
        if not isinstance(payload, Mapping):
            raise ValidationError("snapshot payload must be a mapping")
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValidationError(
                f"snapshot schema_version must be {SCHEMA_VERSION}, got {version!r}"
            )
        convention = payload.get("convention")
        if convention not in CONVENTIONS:
            raise ValidationError(
                f"snapshot convention must be one of {CONVENTIONS}, got {convention!r}"
            )
        underlying = payload.get("underlying")
        if not isinstance(underlying, Mapping) or "spot" not in underlying:
            raise ValidationError("snapshot underlying must define 'spot'")
        source = payload.get("source")
        if not isinstance(source, Mapping):
            raise ValidationError("snapshot must define a 'source' object")
        price_field = source.get("price_field")
        if price_field not in PRICE_FIELDS:
            raise ValidationError(
                f"source.price_field must be one of {PRICE_FIELDS}, got {price_field!r}"
            )
        expiries = payload.get("expiries")
        if not isinstance(expiries, list):
            raise ValidationError("snapshot expiries must be a list")
        return cls(
            schema_version=SCHEMA_VERSION,
            convention=str(convention),
            trade_date=_as_date(payload.get("trade_date")),
            symbol=str(underlying.get("symbol", "")),
            spot=_positive(underlying["spot"], "underlying.spot"),
            price_field=str(price_field),
            source=dict(source),
            expiries=tuple(dict(e) for e in expiries),
        )

    @classmethod
    def from_legacy_live(
        cls,
        payload: Mapping[str, Any],
        *,
        trade_date,
        symbol: str,
        source_sha256: Optional[str] = None,
    ) -> "QuoteSnapshot":
        """Lift a live ``01_fetch_mo_snapshot`` payload into the canonical envelope.

        The live shape carries no trade date and no price-field declaration, so
        both are supplied by the caller rather than inferred.
        """
        underlying = payload.get("underlying")
        if not isinstance(underlying, Mapping) or "spot" not in underlying:
            raise ValidationError("live snapshot underlying missing 'spot'")
        expiries = payload.get("expiries")
        if not isinstance(expiries, list) or not expiries:
            raise ValidationError("live snapshot requires a non-empty 'expiries' list")
        return cls(
            schema_version=SCHEMA_VERSION,
            convention=CONVENTION_LISTED,
            trade_date=_as_date(trade_date),
            symbol=str(symbol),
            spot=_positive(underlying["spot"], "underlying.spot"),
            price_field=PRICE_FIELD_MID_OR_LAST,
            source={
                "vendor": "mo_live_snapshot",
                "price_field": PRICE_FIELD_MID_OR_LAST,
                "fetched_at": payload.get("fetched_at"),
                "market_open": payload.get("market_open"),
                "sha256": source_sha256,
            },
            expiries=tuple(dict(e) for e in expiries),
        )

    @classmethod
    def from_legacy_settlement(
        cls,
        payload: Mapping[str, Any],
        *,
        trade_date,
        spot,
        symbol: str,
        source_sha256: Optional[str] = None,
    ) -> "QuoteSnapshot":
        """Lift a CFFEX settlement payload into the canonical envelope.

        The settlement shape carries no spot — it comes from the separate spot
        CSV — so ``spot`` is a required argument. Nothing is inferred.
        """
        declared = payload.get("price_field")
        if declared != PRICE_FIELD_SETTLEMENT:
            raise ValidationError(
                "settlement snapshot must declare price_field "
                f"{PRICE_FIELD_SETTLEMENT!r}, got {declared!r}"
            )
        expiries = payload.get("expiries")
        if not isinstance(expiries, list) or not expiries:
            raise ValidationError(
                "settlement snapshot requires a non-empty 'expiries' list"
            )
        return cls(
            schema_version=SCHEMA_VERSION,
            convention=CONVENTION_LISTED,
            trade_date=_as_date(trade_date),
            symbol=str(symbol),
            spot=_positive(spot, "spot"),
            price_field=PRICE_FIELD_SETTLEMENT,
            source={
                "vendor": payload.get("source_class", "official_cffex_eod_settlement"),
                "price_field": PRICE_FIELD_SETTLEMENT,
                "record_count": payload.get("record_count"),
                "sha256": source_sha256,
            },
            expiries=tuple(dict(e) for e in expiries),
        )

    # ---------------------------------------------------------------- price
    def quote_price(self, quote: Mapping[str, Any]) -> float:
        """Resolve one quote's price through the declared ``price_field``.

        There is exactly one rule per price field and no cross-field fallback:
        a quote that cannot satisfy the declared field is a
        ``price_field_mismatch``, never silently priced off another key.
        """
        if self.price_field == PRICE_FIELD_SETTLEMENT:
            value = quote.get("settlement")
            if value is None:
                raise ValidationError(
                    "price_field_mismatch: snapshot declares "
                    f"{PRICE_FIELD_SETTLEMENT!r} but quote has no 'settlement' key"
                )
            return float(value)
        if self.price_field == PRICE_FIELD_MID_OR_LAST:
            bid, ask = quote.get("bid"), quote.get("ask")
            if bid is not None and ask is not None:
                bid_f, ask_f = float(bid), float(ask)
                if bid_f > 0.0 and ask_f > 0.0:
                    return 0.5 * (bid_f + ask_f)
            last = quote.get("last")
            if last is None:
                raise ValidationError(
                    "price_field_mismatch: snapshot declares "
                    f"{PRICE_FIELD_MID_OR_LAST!r} but quote has no usable "
                    "bid/ask pair and no 'last' key"
                )
            return float(last)
        raise ValidationError(f"unsupported price_field {self.price_field!r}")
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/test_snapshot.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add quantark/volcalibration/__init__.py quantark/volcalibration/snapshot.py \
        test/volcalibration/__init__.py test/volcalibration/test_snapshot.py
git commit -m "feat(volcalibration): canonical QuoteSnapshot with legacy lifters"
```

---

### Task 2: The `QuoteSet` convergence point

**Files:**
- Create: `quantark/volcalibration/quotes.py`
- Create: `test/volcalibration/test_quotes.py`
- Modify: `quantark/volcalibration/__init__.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `IvNode(strike: float, iv: float, weight_hint: float)`
  - `ExpiryQuotes(expiry_label: str, expiry_date: str | None, T: float, forward: float, discount_factor: float, r: float, q: float, nodes: tuple[IvNode, ...], diagnostics: Mapping[str, float])`
  - `QuoteSet(trade_date: date, spot: float, convention: str, expiries: tuple[ExpiryQuotes, ...])`

**Context you need:** `expiry_label` is mandatory and `expiry_date` optional because CFETS FX snapshots identify a slice by tenor (`3M`) and carry no calendar date; deriving one would need an FX expiry calendar this module does not own (spec §5.2). Validation is in `__post_init__` so an invalid `QuoteSet` cannot exist.

- [ ] **Step 1: Write the failing test**

Create `test/volcalibration/test_quotes.py`:

```python
from datetime import date

import pytest

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.quotes import ExpiryQuotes, IvNode, QuoteSet


def _expiry(T, label="2026-07-31", expiry_date="2026-07-31"):
    return ExpiryQuotes(
        expiry_label=label,
        expiry_date=expiry_date,
        T=T,
        forward=6000.0,
        discount_factor=0.99,
        r=0.02,
        q=0.01,
        nodes=(IvNode(5800.0, 0.21, 1.0), IvNode(6000.0, 0.20, 1.0)),
        diagnostics={"n_pairs": 7.0},
    )


def test_quoteset_orders_and_validates():
    qs = QuoteSet(
        trade_date=date(2026, 7, 6),
        spot=6000.0,
        convention="listed_strike",
        expiries=(_expiry(0.1), _expiry(0.35)),
    )
    assert [e.T for e in qs.expiries] == [0.1, 0.35]
    assert qs.expiries[0].nodes[0].strike == 5800.0


def test_quoteset_rejects_non_increasing_maturities():
    with pytest.raises(ValidationError, match="strictly increasing"):
        QuoteSet(
            trade_date=date(2026, 7, 6),
            spot=6000.0,
            convention="listed_strike",
            expiries=(_expiry(0.35), _expiry(0.1)),
        )


def test_expiry_date_is_optional_but_label_is_not():
    e = ExpiryQuotes(
        expiry_label="3M",
        expiry_date=None,
        T=0.25,
        forward=7.1,
        discount_factor=0.995,
        r=0.02,
        q=0.03,
        nodes=(IvNode(7.0, 0.08, 1.0),),
        diagnostics={},
    )
    assert e.expiry_label == "3M"
    assert e.expiry_date is None
    with pytest.raises(ValidationError, match="expiry_label"):
        ExpiryQuotes(
            expiry_label="",
            expiry_date=None,
            T=0.25,
            forward=7.1,
            discount_factor=0.995,
            r=0.02,
            q=0.03,
            nodes=(IvNode(7.0, 0.08, 1.0),),
            diagnostics={},
        )


def test_expiry_rejects_non_positive_iv():
    with pytest.raises(ValidationError, match="iv"):
        ExpiryQuotes(
            expiry_label="x",
            expiry_date=None,
            T=0.25,
            forward=7.1,
            discount_factor=0.995,
            r=0.02,
            q=0.03,
            nodes=(IvNode(7.0, 0.0, 1.0),),
            diagnostics={},
        )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/test_quotes.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'quantark.volcalibration.quotes'`

- [ ] **Step 3: Write the implementation**

Create `quantark/volcalibration/quotes.py`:

```python
"""Convention-neutral quote container: the point where every normalizer converges.

``listed.py`` (strike-quoted books) and ``fxdelta.py`` (delta-quoted books) do
different work but produce the same :class:`QuoteSet`, so everything downstream
-- smoothing, admission, model calibration -- never branches on convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Mapping, Optional, Tuple

from quantark.util.exceptions import ValidationError


def _finite_positive(value: float, what: str) -> float:
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValidationError(f"{what} must be positive and finite, got {out}")
    return out


@dataclass(frozen=True)
class IvNode:
    """One (strike, implied vol) observation plus its fit weight."""

    strike: float
    iv: float
    weight_hint: float

    def __post_init__(self) -> None:
        _finite_positive(self.strike, "IvNode.strike")
        _finite_positive(self.iv, "IvNode.iv")
        weight = float(self.weight_hint)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValidationError(
                f"IvNode.weight_hint must be non-negative and finite, got {weight}"
            )


@dataclass(frozen=True)
class ExpiryQuotes:
    """One expiry's carry pillars and IV nodes.

    ``expiry_label`` always identifies the slice (an expiry date for listed
    books, a tenor such as ``"3M"`` for FX).  ``expiry_date`` is None when the
    snapshot carries no calendar date; deriving one would require an expiry
    calendar and adjustment convention this module does not own.
    """

    expiry_label: str
    expiry_date: Optional[str]
    T: float
    forward: float
    discount_factor: float
    r: float
    q: float
    nodes: Tuple[IvNode, ...]
    diagnostics: Mapping[str, float]

    def __post_init__(self) -> None:
        if not str(self.expiry_label).strip():
            raise ValidationError("ExpiryQuotes.expiry_label must be non-empty")
        _finite_positive(self.T, "ExpiryQuotes.T")
        _finite_positive(self.forward, "ExpiryQuotes.forward")
        _finite_positive(self.discount_factor, "ExpiryQuotes.discount_factor")
        for name in ("r", "q"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValidationError(f"ExpiryQuotes.{name} must be finite, got {value}")
        if not self.nodes:
            raise ValidationError(
                f"ExpiryQuotes {self.expiry_label}: at least one IV node required"
            )
        strikes = [n.strike for n in self.nodes]
        if any(a >= b for a, b in zip(strikes, strikes[1:])):
            raise ValidationError(
                f"ExpiryQuotes {self.expiry_label}: nodes must be strike-ordered "
                "and unique"
            )


@dataclass(frozen=True)
class QuoteSet:
    """One trading date's expiries, normalized out of any quoting convention."""

    trade_date: date
    spot: float
    convention: str
    expiries: Tuple[ExpiryQuotes, ...]

    def __post_init__(self) -> None:
        _finite_positive(self.spot, "QuoteSet.spot")
        if not self.expiries:
            raise ValidationError("QuoteSet requires at least one expiry")
        times = [e.T for e in self.expiries]
        if any(a >= b for a, b in zip(times, times[1:])):
            raise ValidationError(
                "QuoteSet expiry maturities must be strictly increasing; got "
                f"{times}"
            )
```

Append to `quantark/volcalibration/__init__.py`:

```python
from quantark.volcalibration.quotes import ExpiryQuotes, IvNode, QuoteSet  # noqa: E402

__all__ += ["QuoteSet", "ExpiryQuotes", "IvNode"]
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/test_quotes.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add quantark/volcalibration/quotes.py quantark/volcalibration/__init__.py \
        test/volcalibration/test_quotes.py
git commit -m "feat(volcalibration): QuoteSet convergence contract"
```

---

### Task 3: Put-call parity recovery of `(DF, F, r, q)`

**Files:**
- Create: `quantark/volcalibration/normalize/__init__.py`, `quantark/volcalibration/normalize/listed.py`
- Create: `test/volcalibration/test_listed_parity.py`

**Interfaces:**
- Consumes: `QuoteSnapshot` (Task 1).
- Produces:
  - `ExpirySlice(expiry_label, expiry_date, T, calls: dict[float, float], puts: dict[float, float], volume: dict[tuple[float, str], int])`
  - `ParityResult(r, forward, discount_factor, q, n_pairs, rmse_over_forward)`
  - `iter_expiries(snapshot: QuoteSnapshot) -> list[ExpirySlice]`
  - `imply_forward_and_rate(sl: ExpirySlice, s0: float) -> ParityResult`
  - `MAX_ABS_PARITY_IMPLIED_RATE = 0.10`, `MAX_PARITY_RMSE_FORWARD_RATIO = 0.01`

**Context you need:** Put-call parity is model-free: `C(K) - P(K) = DF * (F - K)` is a straight line in `K` with slope `-DF` and intercept `DF*F`. One OLS fit yields both the market discount factor and the forward. The quality gates (`|r| <= 10%`, `RMSE/forward <= 1%`) come from `example/mo_volmodels/10_calibration_diagnostics.py` and are being promoted here. `ParityResult` gains `rmse_over_forward`, which `_mo_common.ParityResult` does not have — the shim in Task 5 keeps the old five fields available.

- [ ] **Step 1: Write the failing test**

Create `test/volcalibration/test_listed_parity.py`:

```python
import json
import math
from datetime import date
from pathlib import Path

import pytest

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.normalize.listed import (
    ExpirySlice,
    imply_forward_and_rate,
    iter_expiries,
)
from quantark.volcalibration.snapshot import QuoteSnapshot

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "example/mo_volmodels/data/mo_snapshot_sample.json"


def _snapshot():
    return QuoteSnapshot.from_legacy_live(
        json.loads(LIVE.read_text()), trade_date=date(2026, 7, 6), symbol="000852.SH"
    )


def test_parity_recovers_rate_and_forward():
    # sample built with r=0.02, q=0.01, S0=6000 -> F = S0*exp((r-q)*T)
    s0 = 6000.0
    snap = _snapshot()
    slices = iter_expiries(snap)
    assert len(slices) == 4
    for sl in slices:
        res = imply_forward_and_rate(sl, s0)
        assert res.r == pytest.approx(0.02, abs=1e-3)
        assert res.q == pytest.approx(0.01, abs=1e-3)
        assert res.forward == pytest.approx(
            s0 * math.exp((0.02 - 0.01) * sl.T), rel=1e-3
        )
        assert res.n_pairs >= 5
        assert res.rmse_over_forward < 0.01


def test_parity_too_few_pairs():
    sl = ExpirySlice("x", "x", 0.2, {6000.0: 10.0}, {6000.0: 9.0}, {})
    with pytest.raises(ValidationError, match="paired strikes"):
        imply_forward_and_rate(sl, 6000.0)


def test_parity_rejects_non_positive_discount_factor():
    # C - P increasing in K implies a negative discount factor: arbitrage.
    calls = {5000.0: 1.0, 6000.0: 5.0, 7000.0: 9.0}
    puts = {5000.0: 9.0, 6000.0: 5.0, 7000.0: 1.0}
    sl = ExpirySlice("x", "x", 0.5, calls, puts, {})
    with pytest.raises(ValidationError, match="discount factor"):
        imply_forward_and_rate(sl, 6000.0)


def test_parity_rate_gate_rejects_absurd_rate():
    # DF far below 1 at short T => |implied rate| >> 10%
    calls = {5000.0: 600.0, 6000.0: 100.0, 7000.0: 10.0}
    puts = {5000.0: 10.0, 6000.0: 100.0, 7000.0: 600.0}
    sl = ExpirySlice("x", "x", 0.02, calls, puts, {})
    with pytest.raises(ValidationError, match="implied rate"):
        imply_forward_and_rate(sl, 6000.0)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/test_listed_parity.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'quantark.volcalibration.normalize'`

- [ ] **Step 3: Write the implementation**

Create `quantark/volcalibration/normalize/__init__.py`:

```python
"""Quote normalizers: the one place a quoting convention is interpreted.

Each normalizer consumes a :class:`~quantark.volcalibration.snapshot.QuoteSnapshot`
and produces a :class:`~quantark.volcalibration.quotes.QuoteSet`.
"""

from typing import Protocol

from quantark.volcalibration.quotes import QuoteSet
from quantark.volcalibration.snapshot import QuoteSnapshot


class QuoteNormalizer(Protocol):
    """Turn one convention's quotes into the convention-neutral QuoteSet."""

    convention: str

    def normalize(self, snapshot: QuoteSnapshot) -> QuoteSet:
        ...


__all__ = ["QuoteNormalizer"]
```

Create `quantark/volcalibration/normalize/listed.py`:

```python
"""Strike-quoted listed-option normalizer: parity, OTM filter, IV inversion."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import safe_log
from quantark.volcalibration.snapshot import QuoteSnapshot

# Parity quality gates, promoted from
# example/mo_volmodels/10_calibration_diagnostics.py.
MAX_ABS_PARITY_IMPLIED_RATE = 0.10
MAX_PARITY_RMSE_FORWARD_RATIO = 0.01

MIN_PARITY_PAIRS = 3


@dataclass
class ExpirySlice:
    """One expiry's quotes, indexed by strike for calls and puts separately."""

    expiry_label: str
    expiry_date: str | None
    T: float
    calls: Dict[float, float]
    puts: Dict[float, float]
    volume: Dict[Tuple[float, str], int] = field(default_factory=dict)


@dataclass(frozen=True)
class ParityResult:
    """Carry recovered from one expiry via put-call parity."""

    r: float
    forward: float
    discount_factor: float
    q: float
    n_pairs: int
    rmse_over_forward: float


def iter_expiries(snapshot: QuoteSnapshot) -> List[ExpirySlice]:
    """Reshape each expiry's flat quote list into strike-indexed maps."""
    out: List[ExpirySlice] = []
    for exp in snapshot.expiries:
        calls: Dict[float, float] = {}
        puts: Dict[float, float] = {}
        volume: Dict[Tuple[float, str], int] = {}
        for quote in exp["quotes"]:
            strike = float(quote["strike"])
            kind = quote["type"]
            price = snapshot.quote_price(quote)
            (calls if kind == "C" else puts)[strike] = price
            volume[(strike, kind)] = int(quote.get("volume", 0) or 0)
        label = str(exp.get("expiry_date") or exp.get("contract_month"))
        out.append(
            ExpirySlice(
                expiry_label=label,
                expiry_date=exp.get("expiry_date"),
                T=float(exp["T_years"]),
                calls=calls,
                puts=puts,
                volume=volume,
            )
        )
    return out


def imply_forward_and_rate(sl: ExpirySlice, s0: float) -> ParityResult:
    """Recover ``(r, F, DF, q)`` for one expiry from put-call parity.

    Parity is model-free: ``C(K) - P(K) = DF * (F - K)`` is a straight line in
    ``K`` with slope ``-DF`` and intercept ``DF * F``, so one OLS fit yields the
    market discount factor and the forward together.  A slice that violates the
    quality gates is rejected, never repaired.
    """
    pairs = sorted(set(sl.calls) & set(sl.puts))
    if len(pairs) < MIN_PARITY_PAIRS:
        raise ValidationError(
            f"expiry {sl.expiry_label}: only {len(pairs)} paired strikes "
            f"(< {MIN_PARITY_PAIRS})"
        )
    K = np.asarray(pairs, dtype=float)
    y = np.asarray([sl.calls[k] - sl.puts[k] for k in pairs], dtype=float)
    slope, intercept = np.polyfit(K, y, 1)
    df = float(-slope)
    if df <= 0.0:
        raise ValidationError(
            f"expiry {sl.expiry_label}: non-positive discount factor {df:.4g} "
            "(arbitrage-violating quotes) — excluded, not fabricated"
        )
    forward = float(intercept / df)
    if not math.isfinite(forward) or forward <= 0.0:
        raise ValidationError(
            f"expiry {sl.expiry_label}: non-positive implied forward {forward:.4g}"
        )
    residual = y - (slope * K + intercept)
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    rmse_over_forward = rmse / forward
    if rmse_over_forward > MAX_PARITY_RMSE_FORWARD_RATIO:
        raise ValidationError(
            f"expiry {sl.expiry_label}: parity RMSE/forward "
            f"{rmse_over_forward:.4g} exceeds {MAX_PARITY_RMSE_FORWARD_RATIO}"
        )
    r = float(-safe_log(df) / sl.T)
    if abs(r) > MAX_ABS_PARITY_IMPLIED_RATE:
        raise ValidationError(
            f"expiry {sl.expiry_label}: implied rate {r:.4g} exceeds "
            f"+/-{MAX_ABS_PARITY_IMPLIED_RATE}"
        )
    q = float(r - safe_log(forward / s0) / sl.T)
    if not math.isfinite(q):
        raise ValidationError(f"expiry {sl.expiry_label}: non-finite carry q={q}")
    return ParityResult(
        r=r,
        forward=forward,
        discount_factor=df,
        q=q,
        n_pairs=len(pairs),
        rmse_over_forward=rmse_over_forward,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/test_listed_parity.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add quantark/volcalibration/normalize/ test/volcalibration/test_listed_parity.py
git commit -m "feat(volcalibration): put-call parity recovery with quality gates"
```

---

### Task 4: OTM filter, IV inversion, and the full listed normalizer

**Files:**
- Modify: `quantark/volcalibration/normalize/listed.py`
- Create: `test/volcalibration/test_listed_normalize.py`

**Interfaces:**
- Consumes: `ExpirySlice`, `ParityResult`, `imply_forward_and_rate`, `iter_expiries` (Task 3); `QuoteSet`, `ExpiryQuotes`, `IvNode` (Task 2).
- Produces:
  - `OtmQuote(strike: float, kind: str, price: float)`
  - `select_otm(sl, forward, min_volume=1) -> list[OtmQuote]`
  - `otm_implied_vol(oq, s0, r, q_carry, forward, discount_factor, T) -> float | None`
  - `class ListedNormalizer` with `convention = "listed_strike"`, `__init__(self, *, min_volume=1, min_strikes_per_expiry=5, moneyness_width=0.35)` and `normalize(snapshot) -> QuoteSet`

**Context you need:** Only OTM options carry clean vol information — deep-ITM quotes are dominated by intrinsic value and are typically stale and wide, so a small price error becomes a large IV error. The convention is: puts strictly below the forward, calls at or above it. Each OTM put is converted to its call-equivalent price via parity (`C = P + DF*(F - K)`) so a single call inverter handles both wings and the two wings agree at the forward by construction — the no-arbitrage property the Dupire builder requires. `weight_hint` is a Gaussian in log-moneyness with width 0.35, matching `_mo_common.sabr_smoothed_surface`.

- [ ] **Step 1: Write the failing test**

Create `test/volcalibration/test_listed_normalize.py`:

```python
import json
import math
from datetime import date
from pathlib import Path

import pytest

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.normalize.listed import (
    ListedNormalizer,
    OtmQuote,
    otm_implied_vol,
    select_otm,
)
from quantark.volcalibration.normalize.listed import ExpirySlice
from quantark.volcalibration.snapshot import QuoteSnapshot

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "example/mo_volmodels/data/mo_snapshot_sample.json"


def _snapshot():
    return QuoteSnapshot.from_legacy_live(
        json.loads(LIVE.read_text()), trade_date=date(2026, 7, 6), symbol="000852.SH"
    )


def test_select_otm_splits_at_the_forward():
    sl = ExpirySlice(
        "x",
        "x",
        0.25,
        calls={5000.0: 1050.0, 6000.0: 200.0, 7000.0: 5.0},
        puts={5000.0: 5.0, 6000.0: 200.0, 7000.0: 1050.0},
        volume={(k, s): 10 for k in (5000.0, 6000.0, 7000.0) for s in ("C", "P")},
    )
    picked = select_otm(sl, forward=6000.0)
    by_strike = {q.strike: q.kind for q in picked}
    assert by_strike == {5000.0: "P", 6000.0: "C", 7000.0: "C"}


def test_select_otm_drops_illiquid_and_non_positive():
    sl = ExpirySlice(
        "x",
        "x",
        0.25,
        calls={6000.0: 200.0, 7000.0: 0.0},
        puts={5000.0: 5.0},
        volume={(5000.0, "P"): 0, (6000.0, "C"): 3, (7000.0, "C"): 3},
    )
    picked = select_otm(sl, forward=6000.0, min_volume=1)
    assert [q.strike for q in picked] == [6000.0]  # 5000 illiquid, 7000 zero-priced


def test_otm_put_inverts_through_its_call_equivalent():
    s0, r, q, T = 6000.0, 0.02, 0.01, 0.25
    fwd = s0 * math.exp((r - q) * T)
    df = math.exp(-r * T)
    # Price a put via parity from a known-vol call, then invert it back.
    from quantark.volmodels.black_scholes import bs_call_price

    K = 5600.0
    call = bs_call_price(s0, K, T, 0.23, r, q)
    put = call - df * (fwd - K)
    iv = otm_implied_vol(OtmQuote(K, "P", put), s0, r, q, fwd, df, T)
    assert iv == pytest.approx(0.23, abs=1e-6)


def test_uninvertible_quote_returns_none_not_a_fabricated_vol():
    s0, r, q, T = 6000.0, 0.02, 0.01, 0.25
    fwd = s0 * math.exp((r - q) * T)
    df = math.exp(-r * T)
    # A call priced above the spot bound is outside the no-arb band.
    assert otm_implied_vol(OtmQuote(6000.0, "C", 9e9), s0, r, q, fwd, df, T) is None


def test_normalizer_builds_a_quoteset_from_the_sample():
    qs = ListedNormalizer().normalize(_snapshot())
    assert qs.convention == "listed_strike"
    assert qs.spot == 6000.0
    assert len(qs.expiries) >= 2
    times = [e.T for e in qs.expiries]
    assert times == sorted(times)
    first = qs.expiries[0]
    assert len(first.nodes) >= 5
    assert all(0.0 < n.iv < 2.0 for n in first.nodes)
    assert first.diagnostics["n_pairs"] >= 5
    # weight peaks at the forward
    atm = min(first.nodes, key=lambda n: abs(n.strike - first.forward))
    assert atm.weight_hint == max(n.weight_hint for n in first.nodes)


def test_normalizer_rejects_a_snapshot_with_too_few_usable_expiries():
    payload = json.loads(LIVE.read_text())
    payload["expiries"] = payload["expiries"][:1]
    snap = QuoteSnapshot.from_legacy_live(
        payload, trade_date=date(2026, 7, 6), symbol="X"
    )
    with pytest.raises(ValidationError, match="usable expiries"):
        ListedNormalizer().normalize(snap)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/test_listed_normalize.py -v
```

Expected: FAIL — `ImportError: cannot import name 'ListedNormalizer'`

- [ ] **Step 3: Write the implementation**

Append to `quantark/volcalibration/normalize/listed.py` (and add the imports shown to its header):

```python
# add to the imports at the top of the file:
from quantark.util.exceptions import NumericalError
from quantark.volcalibration.quotes import ExpiryQuotes, IvNode, QuoteSet
from quantark.volmodels.black_scholes import implied_vol_call

MIN_STRIKES_PER_EXPIRY = 5
MIN_USABLE_EXPIRIES = 2
DEFAULT_MONEYNESS_WIDTH = 0.35


@dataclass(frozen=True)
class OtmQuote:
    """A single out-of-the-money quote surviving the liquidity filter."""

    strike: float
    kind: str  # "C" or "P"
    price: float


def select_otm(sl: ExpirySlice, forward: float, min_volume: int = 1) -> List[OtmQuote]:
    """Keep only OTM options: puts below the forward, calls at/above it.

    Only OTM options carry clean volatility information.  Deep-ITM quotes are
    dominated by intrinsic value and are typically stale and wide, so a small
    price error there becomes a large IV error.
    """
    out: List[OtmQuote] = []
    for k in sorted(set(sl.calls) | set(sl.puts)):
        kind = "P" if k < forward else "C"
        book = sl.puts if kind == "P" else sl.calls
        if k not in book:
            continue
        if sl.volume.get((k, kind), 0) < min_volume:
            continue
        price = book[k]
        if price <= 0.0:
            continue
        out.append(OtmQuote(strike=k, kind=kind, price=price))
    return out


def otm_implied_vol(oq, s0, r, q_carry, forward, discount_factor, T):
    """Invert an OTM quote to Black IV via its call-equivalent price.

    An OTM put becomes the call at the same strike through parity,
    ``C = P + DF*(F - K)``, so one call inverter serves both wings and the two
    wings agree at the forward by construction -- the no-arbitrage property the
    Dupire builder needs.  A quote outside the no-arb band yields ``None``:
    excluded, never fabricated.
    """
    if oq.kind == "P":
        call_equiv = oq.price + discount_factor * (forward - oq.strike)
    else:
        call_equiv = oq.price
    try:
        return implied_vol_call(s0, oq.strike, T, call_equiv, r, q_carry)
    except NumericalError:
        return None


class ListedNormalizer:
    """Normalize a strike-quoted listed-option snapshot into a QuoteSet."""

    convention = "listed_strike"

    def __init__(
        self,
        *,
        min_volume: int = 1,
        min_strikes_per_expiry: int = MIN_STRIKES_PER_EXPIRY,
        moneyness_width: float = DEFAULT_MONEYNESS_WIDTH,
    ) -> None:
        if min_strikes_per_expiry < 3:
            raise ValidationError("min_strikes_per_expiry must be >= 3")
        if moneyness_width <= 0.0:
            raise ValidationError("moneyness_width must be positive")
        self._min_volume = int(min_volume)
        self._min_strikes = int(min_strikes_per_expiry)
        self._width = float(moneyness_width)

    def normalize(self, snapshot: QuoteSnapshot) -> QuoteSet:
        """Parity -> OTM filter -> IV inversion, one ExpiryQuotes per usable expiry.

        An expiry that fails parity or falls under the strike floor is dropped,
        not repaired; if fewer than two survive, the whole snapshot is rejected.
        """
        s0 = snapshot.spot
        expiries: List[ExpiryQuotes] = []
        for sl in iter_expiries(snapshot):
            try:
                par = imply_forward_and_rate(sl, s0)
            except ValidationError:
                continue  # a rejected expiry is dropped; the reason is re-raised only
                          # when too few survive (below)
            nodes: List[IvNode] = []
            for oq in select_otm(sl, par.forward, min_volume=self._min_volume):
                iv = otm_implied_vol(
                    oq, s0, par.r, par.q, par.forward, par.discount_factor, sl.T
                )
                if iv is None or not (0.0 < iv < 2.0):
                    continue
                weight = math.exp(
                    -0.5 * (math.log(oq.strike / par.forward) / self._width) ** 2
                )
                nodes.append(IvNode(strike=oq.strike, iv=float(iv), weight_hint=weight))
            if len(nodes) < self._min_strikes:
                continue
            nodes.sort(key=lambda n: n.strike)
            expiries.append(
                ExpiryQuotes(
                    expiry_label=sl.expiry_label,
                    expiry_date=sl.expiry_date,
                    T=sl.T,
                    forward=par.forward,
                    discount_factor=par.discount_factor,
                    r=par.r,
                    q=par.q,
                    nodes=tuple(nodes),
                    diagnostics={
                        "n_pairs": float(par.n_pairs),
                        "parity_rmse_over_forward": par.rmse_over_forward,
                        "n_nodes": float(len(nodes)),
                    },
                )
            )
        if len(expiries) < MIN_USABLE_EXPIRIES:
            raise ValidationError(
                f"{snapshot.trade_date.isoformat()}: only {len(expiries)} usable "
                f"expiries (need >= {MIN_USABLE_EXPIRIES}) after parity and the "
                f"{self._min_strikes}-strike floor"
            )
        expiries.sort(key=lambda e: e.T)
        return QuoteSet(
            trade_date=snapshot.trade_date,
            spot=s0,
            convention=self.convention,
            expiries=tuple(expiries),
        )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/test_listed_normalize.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add quantark/volcalibration/normalize/listed.py \
        test/volcalibration/test_listed_normalize.py
git commit -m "feat(volcalibration): OTM filter, IV inversion, listed normalizer"
```

---

### Task 5: `_mo_common` becomes a re-export shim

**Files:**
- Modify: `example/mo_volmodels/_mo_common.py`
- Create: `test/volcalibration/test_mo_common_shim.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: `_mo_common` keeps its existing public names (`load_snapshot`, `ExpirySlice`, `iter_expiries`, `ParityResult`, `imply_forward_and_rate`, `OtmQuote`, `select_otm`, `otm_implied_vol`) with unchanged call signatures and return shapes.

**Context you need:** This task is the proof that the promotion preserved behaviour. 14 Python modules import `_mo_common` (9 example scripts, 5 test modules), and 35 test modules live in `test/mo_volmodels/`. They must all pass **unchanged**.

The shim cannot be a bare re-export, because the legacy signatures differ:
- `load_snapshot(path)` takes a path and returns a **dict**; the library takes a payload and returns a `QuoteSnapshot`.
- `iter_expiries(snapshot_dict)` takes the raw dict.
- `ExpirySlice(expiry_date, T, calls, puts, volume)` has five positional fields; the library's has six (`expiry_label` first).
- `ParityResult` has five fields; the library's has six.

So the shim keeps thin legacy adapters that delegate the maths. The functions that already match (`select_otm`, `otm_implied_vol`) are re-exported directly.

- [ ] **Step 1: Write the failing test**

Create `test/volcalibration/test_mo_common_shim.py`:

```python
"""The shim must preserve _mo_common's exact legacy surface."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "example/mo_volmodels"))
import _mo_common as mc  # noqa: E402

SNAP = ROOT / "example/mo_volmodels/data/mo_snapshot_sample.json"


def test_load_snapshot_still_returns_a_dict():
    snap = mc.load_snapshot(SNAP)
    assert isinstance(snap, dict)
    assert snap["underlying"]["spot"] == 6000.0


def test_iter_expiries_still_takes_a_dict_and_yields_legacy_slices():
    slices = mc.iter_expiries(mc.load_snapshot(SNAP))
    assert len(slices) == 4
    sl = slices[0]
    assert sl.expiry_date  # legacy attribute name preserved
    assert sl.T > 0.0
    assert len(set(sl.calls) & set(sl.puts)) >= 5


def test_parity_delegates_to_the_library():
    from quantark.volcalibration.normalize import listed

    sl = mc.iter_expiries(mc.load_snapshot(SNAP))[0]
    legacy = mc.imply_forward_and_rate(sl, 6000.0)
    lib = listed.imply_forward_and_rate(
        listed.ExpirySlice(sl.expiry_date, sl.expiry_date, sl.T, sl.calls, sl.puts, sl.volume),
        6000.0,
    )
    assert legacy.r == lib.r
    assert legacy.forward == lib.forward
    assert legacy.discount_factor == lib.discount_factor
    assert legacy.q == lib.q
    assert legacy.n_pairs == lib.n_pairs


def test_legacy_five_field_constructor_still_works():
    sl = mc.ExpirySlice("x", 0.2, {6000.0: 10.0}, {6000.0: 9.0})
    with pytest.raises(ValueError):
        mc.imply_forward_and_rate(sl, 6000.0)


def test_otm_helpers_are_the_library_objects():
    from quantark.volcalibration.normalize import listed

    assert mc.select_otm is listed.select_otm
    assert mc.otm_implied_vol is listed.otm_implied_vol
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/test_mo_common_shim.py -v
```

Expected: FAIL — `test_otm_helpers_are_the_library_objects` fails (`mc.select_otm` is the local definition, not the library's).

- [ ] **Step 3: Write the implementation**

In `example/mo_volmodels/_mo_common.py`, replace the bodies of `load_snapshot`, `ExpirySlice`, `iter_expiries`, `ParityResult`, `imply_forward_and_rate`, `OtmQuote`, `select_otm` and `otm_implied_vol` with this shim. Leave `build_env`, `sabr_smoothed_surface`, `prepare_model_surface`, `calibrate_leverage_for` and `plot_smiles` untouched — later tasks handle them.

```python
"""Shared helpers for the MO vol-model suite.

DEPRECATED: the calibration procedure now lives in ``quantark.volcalibration``.
This module is a thin compatibility shim preserving the suite's legacy
signatures; new code should import from the library directly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

from quantark.volcalibration.normalize.listed import (  # noqa: F401
    OtmQuote,
    otm_implied_vol,
    select_otm,
)
from quantark.volcalibration.normalize import listed as _listed
from quantark.volcalibration.snapshot import QuoteSnapshot

_REQUIRED = {"fetched_at", "underlying", "expiries"}
_LEGACY_TRADE_DATE = date(2026, 7, 6)  # the suite's fixed valuation date


def load_snapshot(path) -> dict:
    """Load and lightly validate an MO snapshot JSON (legacy: returns the dict)."""
    snap = json.loads(Path(path).read_text())
    missing = _REQUIRED - set(snap)
    if missing:
        raise ValueError(f"snapshot missing keys: {sorted(missing)}")
    if "spot" not in snap.get("underlying", {}):
        raise ValueError("snapshot underlying missing 'spot'")
    return snap


@dataclass
class ExpirySlice:
    """Legacy five-field slice; delegates the maths to the library."""

    expiry_date: str
    T: float
    calls: Dict[float, float]
    puts: Dict[float, float]
    volume: Dict[Tuple[float, str], int] = field(default_factory=dict)

    def to_library(self) -> "_listed.ExpirySlice":
        return _listed.ExpirySlice(
            expiry_label=self.expiry_date,
            expiry_date=self.expiry_date,
            T=self.T,
            calls=self.calls,
            puts=self.puts,
            volume=self.volume,
        )


@dataclass
class ParityResult:
    """Legacy five-field parity result."""

    r: float
    forward: float
    discount_factor: float
    q: float
    n_pairs: int


def iter_expiries(snapshot: dict) -> List[ExpirySlice]:
    """Reshape a legacy snapshot dict into legacy slices via the library."""
    lifted = QuoteSnapshot.from_legacy_live(
        snapshot, trade_date=_LEGACY_TRADE_DATE, symbol="000852.SH"
    )
    return [
        ExpirySlice(
            expiry_date=sl.expiry_date,
            T=sl.T,
            calls=sl.calls,
            puts=sl.puts,
            volume=sl.volume,
        )
        for sl in _listed.iter_expiries(lifted)
    ]


def imply_forward_and_rate(sl: ExpirySlice, s0: float) -> ParityResult:
    """Recover (r, F, DF, q) for one expiry (legacy wrapper; raises ValueError).

    The library raises ``ValidationError``; the suite's tests expect
    ``ValueError``.  ``ValidationError`` subclasses ``QuantArkException``, not
    ``ValueError``, so the shim re-raises to preserve the legacy contract.
    """
    from quantark.util.exceptions import ValidationError

    try:
        res = _listed.imply_forward_and_rate(sl.to_library(), s0)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
    return ParityResult(
        r=res.r,
        forward=res.forward,
        discount_factor=res.discount_factor,
        q=res.q,
        n_pairs=res.n_pairs,
    )
```

**The re-raise is required, not defensive.** `ValidationError` derives from `QuantArkException`, not `ValueError` — verified:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -c "
from quantark.util.exceptions import ValidationError
print('ValidationError is a ValueError:', issubclass(ValidationError, ValueError))
"   # prints False
```

So `test/mo_volmodels/test_parity.py::test_parity_too_few_pairs`, which asserts `pytest.raises(ValueError)`, only keeps passing because of the wrapper. Do not remove it.

- [ ] **Step 4: Run the shim test, then the whole mo suite unchanged**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/ -v
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/mo_volmodels/ -x -q
```

Expected: `test/volcalibration/` all pass; `test/mo_volmodels/` shows the same pass/skip counts as on `main`. Capture the baseline first if you have not:

```bash
git stash && PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/mo_volmodels/ -q | tail -3 && git stash pop
```

- [ ] **Step 5: Commit — and do NOT stage `example/` data**

The mo test runs rewrite two tracked sample files with float churn. Revert them, then stage only the shim:

```bash
git checkout -- example/mo_volmodels/data/ 2>/dev/null || true
git status --short   # confirm no example/*.json is staged or modified
git add example/mo_volmodels/_mo_common.py test/volcalibration/test_mo_common_shim.py
git commit -m "refactor(mo): _mo_common delegates parity and IV inversion to volcalibration"
```

---

## Phase 2 — Surface, admission, store

### Task 6: Rectangular grid assembly from a `QuoteSet`

**Files:**
- Create: `quantark/volcalibration/surface.py`
- Create: `test/volcalibration/test_surface_grid.py`

**Interfaces:**
- Consumes: `QuoteSet` (Task 2).
- Produces:
  - `MIN_COMMON_STRIKES = 5`
  - `build_raw_surface(quotes: QuoteSet) -> dict` returning the stage-02 surface dict: `{"s0", "strikes", "maturities", "iv_grid", "per_expiry"}` where each `per_expiry` entry is `{"expiry_date", "T", "r", "q", "forward", "df", "points", "pair_count", "parity_rmse_points"}`.

**Context you need:** The output dict is deliberately the *existing* stage-02 schema, because `sabr_smoothed_surface` (Task 7) and the artifact writer (Task 9) already consume exactly that shape, and the artifact bytes must not change. The common strike grid holds strikes present in at least 2 expiries; each expiry's row is filled by linear interpolation of its own smile, flat past its wings.

- [ ] **Step 1: Write the failing test**

Create `test/volcalibration/test_surface_grid.py`:

```python
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.normalize.listed import ListedNormalizer
from quantark.volcalibration.snapshot import QuoteSnapshot
from quantark.volcalibration.surface import build_raw_surface

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "example/mo_volmodels/data/mo_snapshot_sample.json"


def _quotes():
    snap = QuoteSnapshot.from_legacy_live(
        json.loads(LIVE.read_text()), trade_date=date(2026, 7, 6), symbol="000852.SH"
    )
    return ListedNormalizer().normalize(snap)


def test_grid_shape_and_schema():
    surface = build_raw_surface(_quotes())
    grid = np.asarray(surface["iv_grid"])
    assert grid.shape == (len(surface["maturities"]), len(surface["strikes"]))
    assert np.all(np.isfinite(grid)) and np.all(grid > 0.0)
    assert surface["maturities"] == sorted(surface["maturities"])
    assert surface["strikes"] == sorted(surface["strikes"])
    pe = surface["per_expiry"][0]
    assert set(pe) >= {
        "expiry_date", "T", "r", "q", "forward", "df", "points",
        "pair_count", "parity_rmse_points",
    }
    assert pe["points"] == sorted(pe["points"])


def test_grid_rows_interpolate_each_expiry_own_smile():
    quotes = _quotes()
    surface = build_raw_surface(quotes)
    strikes = np.asarray(surface["strikes"])
    row0 = np.asarray(surface["iv_grid"][0])
    own = {n.strike: n.iv for n in quotes.expiries[0].nodes}
    for k, v in own.items():
        if strikes.min() <= k <= strikes.max():
            assert row0[np.argmin(np.abs(strikes - k))] == pytest.approx(v, abs=1e-9)


def test_too_few_common_strikes_is_rejected():
    quotes = _quotes()
    # Shift the second expiry's strikes so nothing is shared.
    from dataclasses import replace

    from quantark.volcalibration.quotes import IvNode

    shifted = replace(
        quotes.expiries[1],
        nodes=tuple(IvNode(n.strike + 3.7, n.iv, n.weight_hint)
                    for n in quotes.expiries[1].nodes),
    )
    broken = replace(quotes, expiries=(quotes.expiries[0], shifted))
    with pytest.raises(ValidationError, match="common strikes"):
        build_raw_surface(broken)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/test_surface_grid.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'quantark.volcalibration.surface'`

- [ ] **Step 3: Write the implementation**

Create `quantark/volcalibration/surface.py`:

```python
"""IV-surface construction: rectangular grid assembly and SABR smoothing.

The dict this module produces and consumes is the stage-02 surface schema the
artifact writer already serializes.  It is kept verbatim because artifact bytes
must not change (spec 5.3).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.quotes import QuoteSet

MIN_COMMON_STRIKES = 5
_STRIKE_MATCH_ATOL = 1e-6


def build_raw_surface(quotes: QuoteSet) -> Dict[str, Any]:
    """Assemble a rectangular strike x maturity IV grid from a QuoteSet.

    The common strike grid holds strikes quoted in at least two expiries; each
    expiry's row is linear interpolation of its own smile, flat past its wings.
    """
    expiries = list(quotes.expiries)
    if len(expiries) < 2:
        raise ValidationError("a surface needs at least 2 expiries")

    all_strikes = sorted({n.strike for e in expiries for n in e.nodes})

    def _count(k: float) -> int:
        return sum(
            any(abs(k - n.strike) < _STRIKE_MATCH_ATOL for n in e.nodes)
            for e in expiries
        )

    strikes = [k for k in all_strikes if _count(k) >= 2]
    if len(strikes) < MIN_COMMON_STRIKES:
        raise ValidationError(
            f"only {len(strikes)} common strikes across expiries "
            f"(need >= {MIN_COMMON_STRIKES})"
        )

    maturities = [e.T for e in expiries]
    grid = np.empty((len(maturities), len(strikes)), dtype=float)
    per_expiry: List[Dict[str, Any]] = []
    for i, e in enumerate(expiries):
        ks = np.asarray([n.strike for n in e.nodes], dtype=float)
        vs = np.asarray([n.iv for n in e.nodes], dtype=float)
        grid[i] = np.interp(strikes, ks, vs)
        rmse = e.diagnostics.get("parity_rmse_over_forward")
        per_expiry.append(
            {
                "expiry_date": e.expiry_date if e.expiry_date else e.expiry_label,
                "T": e.T,
                "r": e.r,
                "q": e.q,
                "forward": e.forward,
                "df": e.discount_factor,
                "pair_count": int(e.diagnostics.get("n_pairs", 0)),
                "parity_rmse_points": (
                    float(rmse) * e.forward if rmse is not None else None
                ),
                "points": [(n.strike, n.iv) for n in e.nodes],
            }
        )

    if not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
        raise ValidationError("grid assembly produced non-positive or non-finite IVs")

    return {
        "s0": quotes.spot,
        "strikes": strikes,
        "maturities": maturities,
        "iv_grid": grid.tolist(),
        "per_expiry": per_expiry,
    }
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/test_surface_grid.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add quantark/volcalibration/surface.py test/volcalibration/test_surface_grid.py
git commit -m "feat(volcalibration): rectangular IV grid assembly from QuoteSet"
```

---

### Task 7: SABR smoothing with calendar projection

**Files:**
- Modify: `quantark/volcalibration/surface.py`
- Create: `test/volcalibration/test_surface_smoothing.py`

**Interfaces:**
- Consumes: `build_raw_surface` output (Task 6).
- Produces:
  - `DEFAULT_SABR_BETA = 1.0`
  - `sabr_smoothed_surface(surface_json, *, beta=1.0, moneyness_width=0.35, grid_size=25, min_calendar_slope=1e-8) -> dict`
  - `prepare_model_surface(surface_json, *, iv_smoothing="sabr", sabr_beta=1.0) -> dict`

**Context you need:** Dupire differentiates total variance twice in strike and once in maturity, so it needs a smooth, static-arbitrage-controlled target. This fits one Hagan SABR slice per expiry, evaluates it on the rectangular grid, then projects total variance to be non-decreasing in maturity at each strike. The returned dict keeps `raw_points` beside each expiry's model `points` so later stages can fit the same smooth target without hiding the raw-market discrepancy.

**This is a straight move of `_mo_common.sabr_smoothed_surface` and `prepare_model_surface`.** Copy them verbatim from `example/mo_volmodels/_mo_common.py` into `surface.py`, with exactly two changes: raise `ValidationError` instead of bare `ValueError`, and import `calibrate_sabr_slice` / `sabr_implied_vol_black` at module top rather than inside the function. Do not restructure the maths — its output feeds the byte-frozen artifact.

- [ ] **Step 1: Write the failing test**

Create `test/volcalibration/test_surface_smoothing.py`:

```python
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.normalize.listed import ListedNormalizer
from quantark.volcalibration.snapshot import QuoteSnapshot
from quantark.volcalibration.surface import (
    build_raw_surface,
    prepare_model_surface,
    sabr_smoothed_surface,
)

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "example/mo_volmodels/data/mo_snapshot_sample.json"


def _raw():
    snap = QuoteSnapshot.from_legacy_live(
        json.loads(LIVE.read_text()), trade_date=date(2026, 7, 6), symbol="000852.SH"
    )
    return build_raw_surface(ListedNormalizer().normalize(snap))


def test_smoothing_matches_the_legacy_helper_bit_for_bit():
    """The promoted function must be numerically identical to the suite's."""
    import sys

    sys.path.insert(0, str(ROOT / "example/mo_volmodels"))
    import _mo_common as mc

    raw = _raw()
    lib = sabr_smoothed_surface(raw, beta=1.0)
    legacy = mc.sabr_smoothed_surface(raw, beta=1.0)
    assert np.array_equal(np.asarray(lib["iv_grid"]), np.asarray(legacy["iv_grid"]))


def test_total_variance_is_non_decreasing_in_maturity():
    out = sabr_smoothed_surface(_raw(), beta=1.0)
    grid = np.asarray(out["iv_grid"])
    T = np.asarray(out["maturities"])
    w = grid * grid * T[:, None]
    assert np.all(np.diff(w, axis=0) >= -1e-12)


def test_raw_points_are_retained_beside_the_model_points():
    out = sabr_smoothed_surface(_raw(), beta=1.0)
    pe = out["per_expiry"][0]
    assert len(pe["raw_points"]) == len(pe["points"])
    assert pe["raw_points"] != pe["points"]  # smoothing actually moved them
    assert out["target_smoothing"]["method"] == "sabr_calendar_projected"
    assert "sabr_params" in pe


def test_prepare_model_surface_none_is_a_passthrough():
    raw = _raw()
    out = prepare_model_surface(raw, iv_smoothing="none")
    assert out["target_smoothing"] == {"method": "none"}
    assert out["iv_grid"] == raw["iv_grid"]


def test_prepare_model_surface_rejects_unknown_mode():
    with pytest.raises(ValidationError, match="iv_smoothing"):
        prepare_model_surface(_raw(), iv_smoothing="bogus")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/test_surface_smoothing.py -v
```

Expected: FAIL — `ImportError: cannot import name 'sabr_smoothed_surface'`

- [ ] **Step 3: Write the implementation**

Add to the top of `quantark/volcalibration/surface.py`:

```python
import copy

from quantark.param.vol.sabr.calibration import calibrate_sabr_slice
from quantark.param.vol.sabr.hagan import sabr_implied_vol_black

DEFAULT_SABR_BETA = 1.0
DEFAULT_MONEYNESS_WIDTH = 0.35
```

Then copy `sabr_smoothed_surface` and `prepare_model_surface` verbatim from `example/mo_volmodels/_mo_common.py` (lines beginning `def sabr_smoothed_surface(` through the end of `prepare_model_surface`), applying only these edits:

1. Delete the two function-local imports (`from quantark.param.vol.sabr.calibration import calibrate_sabr_slice` and `... hagan import sabr_implied_vol_black`) — they are now at module top.
2. Replace every `raise ValueError(` with `raise ValidationError(`.
3. In `prepare_model_surface`, the final `raise ValueError("iv_smoothing must be one of: sabr, none")` becomes `raise ValidationError("iv_smoothing must be one of: sabr, none")`.

Change nothing else — not the weighting, not the bounds, not `min_calendar_slope`, not the order of operations. The bit-for-bit test in Step 1 is what enforces this.

- [ ] **Step 4: Run the test to verify it passes**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/test_surface_smoothing.py -v
```

Expected: 5 passed. If `test_smoothing_matches_the_legacy_helper_bit_for_bit` fails, you changed the maths — diff your copy against the original and revert the difference.

- [ ] **Step 5: Commit**

```bash
git add quantark/volcalibration/surface.py test/volcalibration/test_surface_smoothing.py
git commit -m "feat(volcalibration): SABR smoothing with calendar projection"
```

---

### Task 8: Static-arbitrage admission

**Files:**
- Create: `quantark/volcalibration/admission.py`
- Create: `test/volcalibration/test_admission.py`

**Interfaces:**
- Consumes: smoothed surface dict (Task 7).
- Produces:
  - `class AdmissionReason(str, Enum)` with members `MISSING_SPOT`, `INVALID_SPOT`, `MISSING_SOURCE`, `PARSE_FAILED`, `INVALID_SNAPSHOT`, `PARITY_GATING_FAILED`, `INSUFFICIENT_EXPIRIES`, `DUPLICATE_MATURITY`, `NON_FINITE_CARRY`, `SABR_SMOOTHING_FAILED`, `STATIC_ARBITRAGE`, `INVALID_ATM_PILLAR`, `PRICE_FIELD_MISMATCH`
  - `class AdmissionError(QuantArkException)` with `.reason: AdmissionReason` and `.detail: str`
  - `validate_static_arbitrage(surface: dict) -> str` returning the validation-method label

**Context you need:** With ≥3 maturities this is exactly `build_dupire_local_vol`'s default `validate_arbitrage=True` path (calendar `dw/dT|_y >= 0` plus butterfly denominator `> 0`). With exactly 2 maturities the Dupire builder refuses — it needs ≥3 to form `dw/dT` — so the same two checks are evaluated in reduced form with the same `quantark` finite differences and `Tolerance` thresholds, using the two-point one-sided stencil.

**This is a straight move of `_validate_static_arbitrage`** from `example/mo_volmodels/03_build_iv_surface_history.py:335-419`. Copy it verbatim, changing only the imports (module-top instead of function-local) and wrapping `NumericalError` into `AdmissionError(AdmissionReason.STATIC_ARBITRAGE, ...)` at the caller in Task 9 — **not** inside this function, which keeps raising `NumericalError` so its behaviour is unchanged.

- [ ] **Step 1: Write the failing test**

Create `test/volcalibration/test_admission.py`:

```python
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from quantark.util.exceptions import NumericalError
from quantark.volcalibration.admission import (
    AdmissionError,
    AdmissionReason,
    validate_static_arbitrage,
)
from quantark.volcalibration.normalize.listed import ListedNormalizer
from quantark.volcalibration.snapshot import QuoteSnapshot
from quantark.volcalibration.surface import build_raw_surface, sabr_smoothed_surface

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "example/mo_volmodels/data/mo_snapshot_sample.json"


def _smoothed():
    snap = QuoteSnapshot.from_legacy_live(
        json.loads(LIVE.read_text()), trade_date=date(2026, 7, 6), symbol="000852.SH"
    )
    return sabr_smoothed_surface(build_raw_surface(ListedNormalizer().normalize(snap)))


def test_clean_surface_admits_via_the_dupire_path():
    label = validate_static_arbitrage(_smoothed())
    assert label == "build_dupire_local_vol(validate_arbitrage=True)"


def test_two_maturity_surface_uses_the_reduced_form_checks():
    s = _smoothed()
    s["maturities"] = s["maturities"][:2]
    s["iv_grid"] = s["iv_grid"][:2]
    s["per_expiry"] = s["per_expiry"][:2]
    assert validate_static_arbitrage(s) == "reduced_form_dupire_checks_2_maturities"


def test_calendar_arbitrage_is_rejected_not_repaired():
    s = _smoothed()
    grid = np.asarray(s["iv_grid"])
    grid[-1] *= 0.3  # collapse the far maturity: total variance now falls with T
    s["iv_grid"] = grid.tolist()
    with pytest.raises(NumericalError):
        validate_static_arbitrage(s)


def test_admission_error_carries_a_machine_readable_reason():
    err = AdmissionError(AdmissionReason.STATIC_ARBITRAGE, "calendar arbitrage at T=0.5")
    assert err.reason is AdmissionReason.STATIC_ARBITRAGE
    assert err.reason.value == "static_arbitrage"
    assert "calendar arbitrage" in err.detail
    assert "static_arbitrage" in str(err)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/test_admission.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'quantark.volcalibration.admission'`

- [ ] **Step 3: Write the implementation**

Create `quantark/volcalibration/admission.py`. Start with this header and the enum/error, then append `_validate_static_arbitrage` copied verbatim from `example/mo_volmodels/03_build_iv_surface_history.py` (renamed to `validate_static_arbitrage`, with the function-local imports hoisted to module top):

```python
"""Surface admission: the static-arbitrage gate and its machine-readable reasons.

A date that fails admission gets NO artifact and a manifest record naming the
reason.  Nothing is floored, filled or repaired -- the input is fixed upstream
or the date is excluded.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np

from quantark.param import GridVolSurface
from quantark.param.div import TermStructureDividendYield
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.util.exceptions import NumericalError, QuantArkException
from quantark.util.numerical import fd1_nonuniform, fd2_nonuniform
from quantark.util.numerical.constants import Tolerance
from quantark.volmodels.localvol import build_dupire_local_vol


class AdmissionReason(str, Enum):
    """Stable, machine-readable admission-failure vocabulary."""

    MISSING_SPOT = "missing_spot"
    INVALID_SPOT = "invalid_spot"
    MISSING_SOURCE = "missing_source"
    PARSE_FAILED = "parse_failed"
    INVALID_SNAPSHOT = "invalid_snapshot"
    PARITY_GATING_FAILED = "parity_gating_failed"
    INSUFFICIENT_EXPIRIES = "insufficient_expiries"
    DUPLICATE_MATURITY = "duplicate_maturity"
    NON_FINITE_CARRY = "non_finite_carry"
    SABR_SMOOTHING_FAILED = "sabr_smoothing_failed"
    STATIC_ARBITRAGE = "static_arbitrage"
    INVALID_ATM_PILLAR = "invalid_atm_pillar"
    PRICE_FIELD_MISMATCH = "price_field_mismatch"


class AdmissionError(QuantArkException):
    """One date failed admission, with a stable reason code and free-text detail."""

    def __init__(self, reason: AdmissionReason, detail: str) -> None:
        self.reason = AdmissionReason(reason)
        self.detail = str(detail)
        super().__init__(f"{self.reason.value}: {self.detail}")
```

Then paste `_validate_static_arbitrage`'s body (from `"""Run the suite's LV-input arbitrage validation..."""` onward) under the signature `def validate_static_arbitrage(surface: dict) -> str:`, deleting the five function-local imports at its top since they are now at module level.

The import paths above are already verified: `fd1_nonuniform` / `fd2_nonuniform` come from `quantark.util.numerical`, and `Tolerance` from `quantark.util.numerical.constants`. Confirm with:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -c "
from quantark.util.numerical import fd1_nonuniform, fd2_nonuniform
from quantark.util.numerical.constants import Tolerance
print('imports ok')
"
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/test_admission.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add quantark/volcalibration/admission.py test/volcalibration/test_admission.py
git commit -m "feat(volcalibration): static-arbitrage admission with reason enum"
```

---

### Task 9: Artifact serialization, manifest records, legacy migration

**Files:**
- Create: `quantark/volcalibration/store.py`
- Create: `test/volcalibration/test_store.py`

**Interfaces:**
- Consumes: smoothed surface dict (Task 7), `AdmissionReason`/`AdmissionError` (Task 8).
- Produces:
  - `BUILDER_SCHEMA_VERSION = 1`
  - `EXTRAPOLATION_POLICY = "flat_total_variance"`
  - `serialize_artifact(artifact: dict) -> bytes`
  - `artifact_path(output_dir, trade_date: str) -> Path`
  - `builder_fingerprint(surface_config: Mapping) -> str`
  - `surface_record(trade_date, *, status, reason=None, detail=None, n_expiries=0, artifact_sha256=None, snapshot_sha256=None, price_field=None, fingerprint=None, provenance="verified") -> dict`
  - `migrate_manifest(manifest: dict) -> dict`

**Context you need (spec §5.4):** every existing manifest record predates `snapshot_sha256` / `price_field` / `builder_fingerprint` / `builder_schema_version` / `provenance`. A naive "any mismatch rebuilds" would rebuild all 766 artifacts on the first resumed run, changing their bytes on a different architecture and destroying the warm cache and cohort pins. So the migration copies the manifest's existing **top-level** `price_field` and `config` block down into each record, marks anything unrecoverable `grandfathered`, and **never rebuilds a grandfathered record**.

- [ ] **Step 1: Write the failing test**

Create `test/volcalibration/test_store.py`:

```python
import hashlib
import json

import pytest

from quantark.volcalibration.store import (
    BUILDER_SCHEMA_VERSION,
    builder_fingerprint,
    migrate_manifest,
    serialize_artifact,
    surface_record,
)

LEGACY_MANIFEST = {
    "schema_version": 1,
    "source": "official_cffex_eod_settlement",
    "price_field": "settlement",
    "gap_policy": "consumers carry forward previous admitted surface",
    "config": {
        "sabr_beta": 1.0,
        "min_expiries": 2,
        "min_strikes_per_expiry": 5,
        "min_common_strikes": 5,
        "artifact_schema_version": 1,
    },
    "records": [
        {"date": "20240930", "status": "excluded", "reason": "insufficient_expiries",
         "detail": "1 expiry", "n_expiries": 1, "artifact_sha256": None},
        {"date": "20260430", "status": "ok", "reason": None, "detail": None,
         "n_expiries": 4, "artifact_sha256": "abc123"},
    ],
}


def test_serialize_is_deterministic_and_nan_free():
    art = {"b": 1, "a": [1.0, 2.0]}
    first, second = serialize_artifact(art), serialize_artifact(art)
    assert first == second
    assert first.endswith(b"\n")
    assert b'"a"' in first and first.index(b'"a"') < first.index(b'"b"')  # sorted keys
    with pytest.raises(ValueError):
        serialize_artifact({"x": float("nan")})


def test_fingerprint_is_order_insensitive_but_value_sensitive():
    a = builder_fingerprint({"sabr_beta": 1.0, "min_expiries": 2})
    b = builder_fingerprint({"min_expiries": 2, "sabr_beta": 1.0})
    c = builder_fingerprint({"sabr_beta": 0.5, "min_expiries": 2})
    assert a == b
    assert a != c
    assert len(a) == 64  # sha256 hex


def test_migration_populates_records_without_touching_artifacts():
    migrated = migrate_manifest(LEGACY_MANIFEST)
    ok = next(r for r in migrated["records"] if r["date"] == "20260430")
    assert ok["price_field"] == "settlement"
    assert ok["builder_fingerprint"] == builder_fingerprint(LEGACY_MANIFEST["config"])
    assert ok["builder_schema_version"] == BUILDER_SCHEMA_VERSION
    assert ok["artifact_sha256"] == "abc123"  # untouched
    # no snapshot sha was recoverable -> grandfathered, not verified
    assert ok["provenance"] == "grandfathered"
    assert ok["snapshot_sha256"] is None


def test_migration_is_idempotent():
    once = migrate_manifest(LEGACY_MANIFEST)
    twice = migrate_manifest(once)
    assert once == twice


def test_new_records_are_verified():
    rec = surface_record(
        "20260901",
        status="ok",
        n_expiries=5,
        artifact_sha256="deadbeef",
        snapshot_sha256="feedface",
        price_field="settlement",
        fingerprint=builder_fingerprint({"sabr_beta": 1.0}),
    )
    assert rec["provenance"] == "verified"
    assert rec["builder_schema_version"] == BUILDER_SCHEMA_VERSION
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/test_store.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'quantark.volcalibration.store'`

- [ ] **Step 3: Write the implementation**

First confirm the extrapolation-policy constant's real value so the artifact stays byte-identical:

```bash
grep -n "EXTRAPOLATION_POLICY\s*=" example/mo_volmodels/03_build_iv_surface_history.py
```

Use that exact string. Then create `quantark/volcalibration/store.py`:

```python
"""On-disk contracts: artifact serialization and surface-manifest records.

Artifact bytes are frozen (spec 5.3): their sha256 feeds the calibration cache
key, so provenance lives in manifest records and never in the artifact body.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

# Bump whenever a change to normalization, smoothing or admission CODE alters
# builder output.  Config changes are covered by the fingerprint; code changes
# are not, so this constant is the only mechanism that invalidates artifacts
# after a builder upgrade.  Same obligation as _CACHE_SCHEMA_VERSION in
# quantark/volmodels/calibration.py.
BUILDER_SCHEMA_VERSION = 1

PROVENANCE_VERIFIED = "verified"
PROVENANCE_GRANDFATHERED = "grandfathered"


def serialize_artifact(artifact: Mapping[str, Any]) -> bytes:
    """Deterministic artifact bytes: sorted keys, no NaN, trailing newline."""
    return (
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def artifact_path(output_dir, trade_date: str) -> Path:
    """Canonical artifact filename for one trade date."""
    return Path(output_dir) / f"mo_iv_surface_{trade_date}.json"


def builder_fingerprint(surface_config: Mapping[str, Any]) -> str:
    """sha256 over canonical JSON of the resolved surface-builder config."""
    canonical = json.dumps(
        dict(surface_config), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def surface_record(
    trade_date: str,
    *,
    status: str,
    reason: Optional[str] = None,
    detail: Optional[str] = None,
    n_expiries: int = 0,
    artifact_sha256: Optional[str] = None,
    snapshot_sha256: Optional[str] = None,
    price_field: Optional[str] = None,
    fingerprint: Optional[str] = None,
    provenance: str = PROVENANCE_VERIFIED,
) -> Dict[str, Any]:
    """Build one surface-manifest record."""
    return {
        "date": str(trade_date),
        "status": str(status),
        "reason": reason,
        "detail": detail,
        "n_expiries": int(n_expiries),
        "artifact_sha256": artifact_sha256,
        "snapshot_sha256": snapshot_sha256,
        "price_field": price_field,
        "builder_fingerprint": fingerprint,
        "builder_schema_version": BUILDER_SCHEMA_VERSION,
        "provenance": provenance,
    }


def migrate_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """One-time, no-rebuild migration of legacy surface-manifest records.

    Most of the missing metadata is already in the manifest, one level up: the
    top-level ``price_field`` and ``config`` block ARE the settings those
    artifacts were built with.  Copying them down is lossless and touches no
    artifact.  Anything not recoverable leaves the record ``grandfathered``,
    which is never treated as a mismatch and never triggers a rebuild --
    rebuilding would destroy the very bytes the pins depend on.
    """
    out = dict(manifest)
    config = dict(out.get("config", {}))
    fingerprint = builder_fingerprint(config) if config else None
    top_price_field = out.get("price_field")

    migrated = []
    for raw in out.get("records", []):
        rec = dict(raw)
        rec.setdefault("snapshot_sha256", None)
        if rec.get("price_field") is None:
            rec["price_field"] = top_price_field
        if rec.get("builder_fingerprint") is None:
            rec["builder_fingerprint"] = fingerprint
        rec.setdefault("builder_schema_version", BUILDER_SCHEMA_VERSION)
        if rec.get("provenance") is None:
            recoverable = all(
                rec.get(key) is not None
                for key in ("snapshot_sha256", "price_field", "builder_fingerprint")
            )
            rec["provenance"] = (
                PROVENANCE_VERIFIED if recoverable else PROVENANCE_GRANDFATHERED
            )
        migrated.append(rec)

    out["records"] = migrated
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/test_store.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add quantark/volcalibration/store.py test/volcalibration/test_store.py
git commit -m "feat(volcalibration): artifact serialization and no-rebuild manifest migration"
```

---

### Task 10: End-to-end artifact reproduction against the real history

**Files:**
- Create: `test/volcalibration/test_artifact_reproduction.py`
- Modify: `quantark/volcalibration/surface.py` (add `build_artifact`)

**Interfaces:**
- Consumes: Tasks 6–9.
- Produces: `build_artifact(quotes: QuoteSet, *, sabr_beta=1.0) -> dict` — the complete artifact dict including `atm_pillars`, `extrapolation_policy` and `admission`.

**Context you need (spec §8):** byte comparison is **not CI-safe**. CI is x86_64 Linux, the artifacts were frozen on ARM64, and SABR runs through `scipy.optimize`, so bitwise equality across architectures is not a property this project has (see `reference_cross_arch_goldens`). The test therefore always compares with tolerance and byte-compares only when pointed at a local baseline directory via an environment variable.

- [ ] **Step 1: Write the failing test**

Create `test/volcalibration/test_artifact_reproduction.py`:

```python
"""Rebuild real historical surfaces and compare against the artifacts on disk.

Byte equality is a LOCAL property only: CI is x86_64 while the artifacts were
frozen on ARM64, and SABR goes through scipy.optimize.  So the tolerance
comparison always runs, and the byte comparison runs only when
QUANTARK_VOLCALIB_BASELINE points at a local artifact directory.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from quantark.volcalibration.normalize.listed import ListedNormalizer
from quantark.volcalibration.snapshot import QuoteSnapshot
from quantark.volcalibration.store import serialize_artifact
from quantark.volcalibration.surface import build_artifact

ROOT = Path(__file__).resolve().parents[2]
SETTLE_DIR = ROOT / "example/mo_volmodels/data"
IV_TOL = 5e-9


def _settlement_cases():
    """(settlement snapshot, matching artifact) pairs available in the repo."""
    cases = []
    for snap_path in sorted(SETTLE_DIR.glob("mo_settlement_snapshot_*.json")):
        tag = snap_path.stem.rsplit("_", 1)[-1]
        art = SETTLE_DIR / f"history/iv_surface/mo_iv_surface_{tag}.json"
        if art.is_file():
            cases.append((tag, snap_path, art))
    return cases


@pytest.mark.skipif(not _settlement_cases(), reason="no local settlement history")
@pytest.mark.parametrize("tag,snap_path,art_path", _settlement_cases())
def test_rebuilt_surface_matches_the_stored_artifact(tag, snap_path, art_path):
    stored = json.loads(art_path.read_text())
    snap = QuoteSnapshot.from_legacy_settlement(
        json.loads(snap_path.read_text()),
        trade_date=tag,
        spot=float(stored["s0"]),
        symbol="000852.SH",
    )
    rebuilt = build_artifact(ListedNormalizer().normalize(snap))

    assert rebuilt["strikes"] == stored["strikes"]
    assert rebuilt["maturities"] == pytest.approx(stored["maturities"], rel=1e-12)
    np.testing.assert_allclose(
        np.asarray(rebuilt["iv_grid"]), np.asarray(stored["iv_grid"]), atol=IV_TOL
    )

    baseline = os.environ.get("QUANTARK_VOLCALIB_BASELINE")
    if baseline:
        expected = (Path(baseline) / art_path.name).read_bytes()
        assert serialize_artifact(rebuilt) == expected, (
            f"{tag}: artifact bytes changed — the calibration cache and cohort "
            "pins key off this sha (spec 5.3)"
        )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/test_artifact_reproduction.py -v
```

Expected: FAIL — `ImportError: cannot import name 'build_artifact'`. If instead every case is **skipped**, the local history is absent; check whether `example/mo_volmodels/data/history/iv_surface/` exists and, if not, note it in the commit message — the test then guards future runs rather than this one.

- [ ] **Step 3: Write the implementation**

Append to `quantark/volcalibration/surface.py`:

```python
def build_artifact(quotes: QuoteSet, *, sabr_beta: float = DEFAULT_SABR_BETA) -> Dict[str, Any]:
    """QuoteSet -> complete, admission-checked IV-surface artifact dict.

    The key order and field set are those the existing artifacts already use;
    changing them changes the artifact sha and invalidates the calibration cache.
    """
    from quantark.volcalibration.admission import (
        AdmissionError,
        AdmissionReason,
        validate_static_arbitrage,
    )
    from quantark.util.exceptions import NumericalError

    raw = build_raw_surface(quotes)
    try:
        smoothed = sabr_smoothed_surface(raw, beta=sabr_beta)
    except (ValidationError, NumericalError) as exc:
        raise AdmissionError(
            AdmissionReason.SABR_SMOOTHING_FAILED, f"{type(exc).__name__}: {exc}"
        ) from exc

    atm_pillars = []
    for pe in smoothed["per_expiry"]:
        params = pe["sabr_params"]
        atm_vol = float(
            sabr_implied_vol_black(
                float(pe["forward"]),
                [float(pe["forward"])],
                [float(pe["T"])],
                params["alpha"],
                params["beta"],
                params["rho"],
                params["nu"],
                shift=params["shift"],
            )[0]
        )
        if not (np.isfinite(atm_vol) and atm_vol > 0.0):
            raise AdmissionError(
                AdmissionReason.INVALID_ATM_PILLAR,
                f"expiry {pe['expiry_date']}: SABR ATM vol {atm_vol}",
            )
        atm_pillars.append(
            {"T": float(pe["T"]), "expiry_date": pe["expiry_date"], "atm_vol": atm_vol}
        )

    try:
        validation_method = validate_static_arbitrage(smoothed)
    except (NumericalError, ValidationError) as exc:
        raise AdmissionError(
            AdmissionReason.STATIC_ARBITRAGE, f"{type(exc).__name__}: {exc}"
        ) from exc

    smoothed["atm_pillars"] = atm_pillars
    smoothed["extrapolation_policy"] = {
        "beyond_last_listed_expiry": EXTRAPOLATION_POLICY,
        "max_listed_T": max(float(t) for t in smoothed["maturities"]),
    }
    smoothed["admission"] = {
        "min_expiries": 2,
        "min_strikes_per_expiry": 5,
        "min_common_strikes": MIN_COMMON_STRIKES,
        "sabr_beta": float(sabr_beta),
        "static_arbitrage_validation": validation_method,
    }
    return smoothed
```

**Then reconcile the `admission` block against a stored artifact** — it must match key-for-key or the bytes change:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -c "
import json,glob
p=sorted(glob.glob('example/mo_volmodels/data/history/iv_surface/*.json'))[:1]
print(json.dumps(json.load(open(p[0]))['admission'], indent=2, sort_keys=True) if p else 'no local history')
"
```

Add every key it shows (including `parity_quality_gate` and `strike_grid` / `sabr_fit_domain` labels) with the same values. Also add `trade_date` to the artifact if the stored one has it.

- [ ] **Step 4: Run the test to verify it passes**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/test_artifact_reproduction.py -v
QUANTARK_VOLCALIB_BASELINE=example/mo_volmodels/data/history/iv_surface \
  PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/test_artifact_reproduction.py -v
```

Expected: tolerance run passes. If the byte run fails, diff the rebuilt JSON against the stored one and fix the *key set or ordering*, never the maths.

- [ ] **Step 5: Commit**

```bash
git checkout -- example/ 2>/dev/null || true
git add quantark/volcalibration/surface.py test/volcalibration/test_artifact_reproduction.py
git commit -m "feat(volcalibration): end-to-end artifact build with reproduction test"
```

---

## Phase 3 — Calibration relocation

### Task 11: Relocate `VolModelCalibrationConfig`

**Files:**
- Create: `quantark/volcalibration/config.py`
- Modify: `quantark/backtest/replay/config.py`
- Create: `test/volcalibration/test_config_relocation.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `quantark.volcalibration.config.VolModelCalibrationConfig` and `HESTON_PRESETS`, with the identical field set and `__post_init__` validation.

**Context you need:** `VolModelCalibrationConfig` currently lives in `quantark/backtest/replay/config.py:26`, and `quantark/volmodels/calibration.py` duck-types it as `config: Any` — a library engine depending on the backtest package's config shape (spec §2.3). Moving it reverses that. `quantark/backtest/otc/__init__.py` and `quantark/backtest/replay/__init__.py` both re-export the name; both must keep working.

- [ ] **Step 1: Write the failing test**

Create `test/volcalibration/test_config_relocation.py`:

```python
import pytest

from quantark.util.exceptions import ValidationError


def test_config_is_importable_from_the_new_home():
    from quantark.volcalibration.config import HESTON_PRESETS, VolModelCalibrationConfig

    cfg = VolModelCalibrationConfig()
    assert cfg.heston_preset == "mo_frozen"
    assert "mo_frozen" in HESTON_PRESETS
    assert cfg.slv_n_x == 161


def test_all_legacy_import_paths_are_the_same_class():
    from quantark.backtest.otc import VolModelCalibrationConfig as FromOtc
    from quantark.backtest.replay import VolModelCalibrationConfig as FromReplay
    from quantark.backtest.replay.config import VolModelCalibrationConfig as FromModule
    from quantark.volcalibration.config import VolModelCalibrationConfig as Canonical

    assert FromOtc is Canonical
    assert FromReplay is Canonical
    assert FromModule is Canonical


def test_validation_still_fails_closed():
    from quantark.volcalibration.config import VolModelCalibrationConfig

    with pytest.raises(ValidationError):
        VolModelCalibrationConfig(heston_preset="nope")
    with pytest.raises(ValidationError):
        VolModelCalibrationConfig(slv_n_x=2)
    with pytest.raises(ValidationError):
        VolModelCalibrationConfig(heston_temporal_regularization=1.0)  # no reference
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/test_config_relocation.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'quantark.volcalibration.config'`

- [ ] **Step 3: Write the implementation**

1. Move the `VolModelCalibrationConfig` dataclass (and its `_validate_heston_vector` helper) out of `quantark/backtest/replay/config.py` into a new `quantark/volcalibration/config.py`, unchanged.
2. Move `HESTON_PRESETS` out of `quantark/volmodels/calibration.py` into `quantark/volcalibration/config.py`, unchanged — including its full provenance comment block about the deliberate `enforce_feller=True` divergence.
3. In `quantark/backtest/replay/config.py`, replace the class definition with a re-export:

```python
# VolModelCalibrationConfig moved to quantark.volcalibration.config in 0.4.0:
# the vol-calibration engine must not depend on the backtest package's config
# shape.  Re-exported here until 0.5.0 for existing importers.
from quantark.volcalibration.config import (  # noqa: F401
    HESTON_PRESETS,
    VolModelCalibrationConfig,
)
```

4. In `quantark/volmodels/calibration.py`, import `HESTON_PRESETS` from the new home instead of defining it.

**Check for an import cycle before running the tests** — `backtest.replay.config` now imports `volcalibration.config`, so `volcalibration` must not import `backtest`:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -c "
import quantark.volcalibration.config, quantark.backtest.replay.config
print('no cycle')
"
grep -rn "quantark.backtest\|quantark.asset" quantark/volcalibration/ || echo "dependency direction clean"
```

- [ ] **Step 4: Run the tests**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/test_config_relocation.py -v
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_otc_vol_calibrators.py test/test_otc_vol_history_env.py -q
```

Expected: relocation tests pass; the two existing calibrator test modules unchanged.

- [ ] **Step 5: Commit**

```bash
git add quantark/volcalibration/config.py quantark/backtest/replay/config.py \
        quantark/volmodels/calibration.py test/volcalibration/test_config_relocation.py
git commit -m "refactor(volcalibration): own VolModelCalibrationConfig, reversing the layering inversion"
```

---

### Task 12: Relocate `VolModelCalibrator` with a cache-key invariant

**Files:**
- Create: `quantark/volcalibration/calibrate.py`
- Modify: `quantark/volmodels/calibration.py`, `quantark/volcalibration/__init__.py`
- Create: `test/volcalibration/test_calibrate_relocation.py`

**Interfaces:**
- Consumes: `VolModelCalibrationConfig` (Task 11).
- Produces: `quantark.volcalibration.calibrate.VolModelCalibrator`, `CalibratedVolModel`, `VOL_MODEL_VARIANTS`, `_CACHE_SCHEMA_VERSION`.

**Context you need:** the disk cache key is `sha256(f"{surface_sha}|{variant}|{fingerprint}")` and contains **no module paths**, so entries written before the move must be cache *hits* after it. `test/test_calibration_relocation.py` already asserts this for the previous `backtest.otc → volmodels` move; this task adds the same invariant for `volmodels → volcalibration`. Reuse its fixture helper `test/replay_golden/fixtures.py::write_localvol_history`, which builds a committed synthetic surface history that runs anywhere.

- [ ] **Step 1: Write the failing test**

Create `test/volcalibration/test_calibrate_relocation.py`:

```python
"""The relocation must not change the disk cache key."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from replay_golden import fixtures  # noqa: E402

from quantark.param.vol.surface_history import VolSurfaceHistory  # noqa: E402


def _artifact(tmp_path):
    history_dir = fixtures.write_localvol_history(tmp_path)
    return VolSurfaceHistory(history_dir).surface_for(fixtures.DATE_A)


def test_entry_written_via_the_old_path_is_a_hit_on_the_new_one(tmp_path):
    from quantark.volcalibration.calibrate import VolModelCalibrator as New
    from quantark.volcalibration.config import VolModelCalibrationConfig
    from quantark.volmodels.calibration import VolModelCalibrator as Old

    artifact = _artifact(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    first = Old(VolModelCalibrationConfig(cache_dir=str(cache_dir))).calibrate(
        "localvol", artifact
    )
    assert first.record["cache_hit"] is False

    second = New(VolModelCalibrationConfig(cache_dir=str(cache_dir))).calibrate(
        "localvol", artifact
    )
    assert second.record["cache_hit"] is True
    assert second.surface_sha == first.surface_sha


def test_old_module_re_exports_the_new_class():
    from quantark.volcalibration.calibrate import VolModelCalibrator as New
    from quantark.volmodels.calibration import VolModelCalibrator as Old

    assert Old is New
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/test_calibrate_relocation.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'quantark.volcalibration.calibrate'`

- [ ] **Step 3: Write the implementation**

1. `git mv quantark/volmodels/calibration.py quantark/volcalibration/calibrate.py`
2. In `calibrate.py`, change the `HESTON_PRESETS` definition to an import from `quantark.volcalibration.config` (Task 11 already moved it). Change nothing else — not `_CACHE_SCHEMA_VERSION`, not the key construction, not any default.
3. Create a new `quantark/volmodels/calibration.py` containing only:

```python
"""DEPRECATED: moved to :mod:`quantark.volcalibration.calibrate` in 0.4.0.

``quantark.volmodels`` is asset-neutral and imports no pricing environments;
per-day calibration needs market data, so it moved one layer up.  This shim is
kept until 0.5.0.  The disk cache key contains no module paths, so entries
written through either path interoperate.
"""

from quantark.volcalibration.calibrate import (  # noqa: F401
    VOL_MODEL_HESTON,
    VOL_MODEL_HESTON_SLV,
    VOL_MODEL_LOCALVOL,
    VOL_MODEL_VARIANTS,
    CalibratedVolModel,
    VolModelCalibrator,
    _CACHE_SCHEMA_VERSION,
)

__all__ = [
    "VolModelCalibrator",
    "CalibratedVolModel",
    "VOL_MODEL_VARIANTS",
    "VOL_MODEL_LOCALVOL",
    "VOL_MODEL_HESTON",
    "VOL_MODEL_HESTON_SLV",
]
```

4. Add to `quantark/volcalibration/__init__.py`:

```python
from quantark.volcalibration.calibrate import (  # noqa: E402
    CalibratedVolModel,
    VolModelCalibrator,
)
from quantark.volcalibration.config import VolModelCalibrationConfig  # noqa: E402

__all__ += ["VolModelCalibrator", "CalibratedVolModel", "VolModelCalibrationConfig"]
```

**Check whether the old module exported names this shim omits:**

```bash
git show HEAD:quantark/volmodels/calibration.py | grep -nE "^[A-Z_]+ =|^class |^def " | grep -v "^.*_[a-z]"
```

Add any public name that appears there to the shim's import list.

- [ ] **Step 4: Run the tests**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/volcalibration/ -v
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_calibration_relocation.py \
       test/test_otc_vol_calibrators.py test/test_otc_vol_history_env.py -q
```

Expected: all pass, including the pre-existing `test_calibration_relocation.py`.

- [ ] **Step 5: Full-suite gate and commit**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/ -q 2>&1 | tail -15
git checkout -- example/ 2>/dev/null || true
git add quantark/volcalibration/ quantark/volmodels/calibration.py \
        test/volcalibration/test_calibrate_relocation.py
git commit -m "refactor(volcalibration): relocate VolModelCalibrator, shim at the old path"
```

The full suite must show the same failures as the `main` baseline — no new ones. Capture that baseline before starting Task 12 if you have not already.

---

## Self-Review

**Spec coverage (phases 1–3):**

| Spec section | Task |
|---|---|
| §5.1 `QuoteSnapshot`, price-field rule, no mixed cohorts, legacy lifters | 1 |
| §5.2 `QuoteSet` / `ExpiryQuotes` / `IvNode`, `expiry_label` split | 2 |
| §4.4 convergence point; §7.1 parity, OTM filter, IV inversion | 3, 4 |
| §7.2 `_mo_common` shim, existing suite green | 5 |
| §7.1 `_surface_base` grid assembly | 6 |
| §7.1 `sabr_smoothed_surface`, `prepare_model_surface` | 7 |
| §7.1 `_validate_static_arbitrage`, reason enum | 8 |
| §5.3 byte constraint, §5.4 records + migration + `builder_schema_version` | 9, 10 |
| §8 artifact reproduction with the cross-arch caveat | 10 |
| §2.3 layering inversion; §7.1 config move | 11 |
| §7.1 `volmodels/calibration.py` relocation, cache-key invariant | 12 |

**Deferred to the phase 4–7 plan:** `runner.py`, `cli.py`, `__main__.py`, `yaml_loader.py`, `RunConfig`, exit codes, `CalibrationSet`, `environment_for`, the backtest config field, `normalize/fxdelta.py`, and the module `CLAUDE.md`.

**Resolved while writing this plan** (answers are in Global Constraints, not left as homework): the `EXTRAPOLATION_POLICY` string, the finite-difference and `Tolerance` import paths, the `ValidationError`/`ValueError` question, the `replay_golden` fixture names, and the venv's real location.

**Two gaps genuinely need the executor's environment**, each with a verification command in its step: the full `admission` key set, which must be read off a stored artifact so the rebuilt bytes match (Task 10 Step 3), and any public name the old `calibration.py` exported beyond the shim's import list (Task 12 Step 3). Both depend on local files this plan cannot inline.

**Type consistency:** `QuoteSet`/`ExpiryQuotes`/`IvNode` field names are identical in Tasks 2, 4, 6 and 10. `ExpirySlice` has six fields in the library (Task 3) and five in the shim (Task 5) — deliberate, and `to_library()` bridges them. `AdmissionReason.PRICE_FIELD_MISMATCH` (Task 8) matches the string `QuoteSnapshot.quote_price` raises (Task 1).
