# Equity Implied Futures Carry Risk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Listed index futures tenors (`IC00`..`IC03`) become both calibration
instruments for the option carry curve (implied `q(T)` from marks) and hedge
buckets (`delta_bucket_i = dPV/dF_i` → fractional hedge hands).

**Architecture:** A frozen `IndexFuturesCurve` market object converts futures
marks to `TermStructureDividendYield` via the continuous-compounding inversion
`q_i = r_i − ln(F_i/S)/T_i` and supports single-contract bumping. Two new
`GreeksCalculator` methods reprice under rebuilt implied curves (delta buckets)
or node/interval dividend bumps (rhoq buckets). `DeltaOneEngine` gains futures
`dividend_rho`. Portfolio helpers aggregate position-level rows by contract.

**Tech Stack:** Python dataclasses, numpy, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-03-equity-implied-futures-carry-risk-design.md`

## Global Constraints

- Continuous compounding throughout: `q(T_i) = r(T_i) − ln(F_i / S) / T_i`.
- `beta` must be exactly `1.0` (ValidationError otherwise); `delta_per_hand = multiplier`.
- `hedge_hands = -delta_bucket / delta_per_hand`, fractional, **never rounded** in the API.
- `len(quotes) >= 2` (conversion target `TermStructureDividendYield` requires ≥2 nodes).
- Delta buckets: one-sided **up** price bump, default `1.0` index point.
- Rhoq buckets: one-sided **up** dividend bump (matches scalar `calculate_numerical_dividend_rho`), scaled per 1% via `* (0.01 / div_bump)`.
- `calculate_futures_delta_buckets` requires resolved mode `IMPLIED_FUTURES_CARRY`; `calculate_futures_rhoq_buckets` rejects `MARKET_PRICE`.
- Flat extrapolation of nodal `q` outside `[T_first, T_last]`; rows flagged `extrapolated_tail` when option maturity is outside the quoted range.
- MC finite differences need common random numbers: reuse the caller's engine/params unchanged for base and bumped legs; tests pin `MCParams.seed`.
- Signs unflipped: theoretical long futures `rhoq < 0`.
- Only `linear_q` interpolation; unknown values raise `ValidationError` (`quantark.util.exceptions`).
- Never claim single-node PV locality under `linear_q` (interpolation spillover is intended).
- Docs commits need `git add -f` (docs/ is in `.git/info/exclude`).
- Test runner (from a worktree): `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest <file> -q`.

## File Structure

- Modify `quantark/util/enum/greeks_enums.py` — add `EquityDividendInputMode`, `FuturesCarryRiskMode`.
- Modify `quantark/util/enum/__init__.py:50` — export the two new enums.
- Create `quantark/asset/equity/market/__init__.py` + `quantark/asset/equity/market/index_futures_curve.py` — `IndexFuturesQuote`, `IndexFuturesCurve`, `hedge_hands`, `bump_term_yield_node`.
- Modify `quantark/asset/equity/engine/analytical/deltaone_engine.py:237-304` (`_calculate_futures_greeks`) — futures `dividend_rho`.
- Modify `quantark/asset/equity/riskmeasures/greeks_calculator.py` — `calculate_futures_delta_buckets`, `calculate_futures_rhoq_buckets`, `_rhoq_bucket_row` (insert after `calculate_numerical_delta_q`, which starts at line 920).
- Create `quantark/portfolio/equity/futures_buckets.py` — `aggregate_futures_delta_buckets`, `aggregate_futures_rhoq_buckets`; export from `quantark/portfolio/equity/__init__.py`.
- Create `example/equity_futures_delta_buckets_demo.py`.
- Test: `test/test_index_futures_implied_carry.py` (curve/enum/helpers), `test/test_equity_futures_delta_buckets.py` (calculator, engines, portfolio, demo-level flows).

---

### Task 1: Carry-mode enums

**Files:**
- Modify: `quantark/util/enum/greeks_enums.py` (append at end)
- Modify: `quantark/util/enum/__init__.py:50`
- Test: `test/test_index_futures_implied_carry.py` (new file)

**Interfaces:**
- Produces: `FuturesCarryRiskMode.{MARKET_PRICE, THEORETICAL_CARRY, IMPLIED_FUTURES_CARRY}` and `EquityDividendInputMode.{FLAT_DIVIDEND, TERM_DIVIDEND}`, importable from `quantark.util.enum`.

- [ ] **Step 1: Write the failing test**

Create `test/test_index_futures_implied_carry.py`:

```python
"""IndexFuturesCurve: implied carry from futures marks (spec tests 1, 2, 9)."""
import math

import pytest

from quantark.util.enum import EquityDividendInputMode, FuturesCarryRiskMode
from quantark.util.exceptions import ValidationError


def test_futures_carry_risk_mode_values():
    assert FuturesCarryRiskMode.MARKET_PRICE.value == "market_price"
    assert FuturesCarryRiskMode.THEORETICAL_CARRY.value == "theoretical_carry"
    assert FuturesCarryRiskMode.IMPLIED_FUTURES_CARRY.value == "implied_futures_carry"


def test_equity_dividend_input_mode_values():
    assert EquityDividendInputMode.FLAT_DIVIDEND.value == "flat_dividend"
    assert EquityDividendInputMode.TERM_DIVIDEND.value == "term_dividend"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_index_futures_implied_carry.py -q`
Expected: FAIL — `ImportError: cannot import name 'EquityDividendInputMode'`

- [ ] **Step 3: Implement**

Append to `quantark/util/enum/greeks_enums.py`:

```python
class EquityDividendInputMode(Enum):
    """How the option-pricing dividend/carry input is supplied."""

    FLAT_DIVIDEND = "flat_dividend"
    TERM_DIVIDEND = "term_dividend"


class FuturesCarryRiskMode(Enum):
    """Interpretation of index futures marks for pricing and carry risk.

    MARKET_PRICE: futures mark is exogenous; model rhoq = 0 by convention.
    THEORETICAL_CARRY: futures generated from S, r, q(T); rhoq non-zero.
    IMPLIED_FUTURES_CARRY: marks imply q(T) for option pricing; futures/rhoq
        buckets are portfolio risk coordinates.
    """

    MARKET_PRICE = "market_price"
    THEORETICAL_CARRY = "theoretical_carry"
    IMPLIED_FUTURES_CARRY = "implied_futures_carry"
```

In `quantark/util/enum/__init__.py` change line 50 to:

```python
from .greeks_enums import (
    CommonGreek,
    EquityGreek,
    EquityDividendInputMode,
    FuturesCarryRiskMode,
)
```

and add `'EquityDividendInputMode', 'FuturesCarryRiskMode',` entries to the
`__all__` list (keep alphabetical placement consistent with neighbors).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_index_futures_implied_carry.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add quantark/util/enum/greeks_enums.py quantark/util/enum/__init__.py test/test_index_futures_implied_carry.py
git commit -m "feat(enum): EquityDividendInputMode + FuturesCarryRiskMode"
```

---

### Task 2: IndexFuturesQuote / IndexFuturesCurve market objects

**Files:**
- Create: `quantark/asset/equity/market/__init__.py`
- Create: `quantark/asset/equity/market/index_futures_curve.py`
- Test: `test/test_index_futures_implied_carry.py` (append)

**Interfaces:**
- Consumes: `FuturesCarryRiskMode` (Task 1), `TermStructureDividendYield` (`quantark/param/div/dividend_yield.py`, requires ≥2 strictly-increasing times, `|y| ≤ 1.0`), `RateCurve.get_rate(t)`.
- Produces (used by Tasks 4, 5, 7, 9):
  - `IndexFuturesQuote(contract: str, maturity: float, price: float, multiplier: float, beta: float = 1.0, expiry_date: Optional[datetime] = None)` — frozen dataclass; `beta != 1.0` raises.
  - `IndexFuturesCurve(underlying: str, spot: float, quotes: Sequence[IndexFuturesQuote], mode: FuturesCarryRiskMode = IMPLIED_FUTURES_CARRY, interpolation: str = "linear_q")` — frozen dataclass with methods `get_quote(contract) -> IndexFuturesQuote`, `implied_yields(rate_curve) -> list[float]`, `to_dividend_yield_curve(rate_curve) -> TermStructureDividendYield`, `bump_contract(contract, price_bump) -> IndexFuturesCurve`, `delta_per_hand(contract) -> float`.
  - `hedge_hands(delta_bucket: float, delta_per_hand: float) -> float`.
  - `bump_term_yield_node(term_div: TermStructureDividendYield, node_index: int, bump: float) -> TermStructureDividendYield`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_index_futures_implied_carry.py`:

```python
from quantark.asset.equity.market import (
    IndexFuturesCurve,
    IndexFuturesQuote,
    bump_term_yield_node,
    hedge_hands,
)
from quantark.param import FlatRateCurve
from quantark.param.div.dividend_yield import TermStructureDividendYield


def _quotes_from_q(spot, r, times, qs, multiplier=200.0):
    """Generate marks from known q(T): F = S * exp((r - q) * T)."""
    return [
        IndexFuturesQuote(
            contract=f"IC{i:02d}",
            maturity=t,
            price=spot * math.exp((r - q) * t),
            multiplier=multiplier,
        )
        for i, (t, q) in enumerate(zip(times, qs))
    ]


def _curve(spot=100.0, r=0.03, times=(0.25, 0.5, 1.0), qs=(0.01, 0.02, 0.015)):
    return IndexFuturesCurve(
        underlying="IC", spot=spot, quotes=_quotes_from_q(spot, r, list(times), list(qs))
    )


# --- spec test 1: implied carry recovers known q(T) ---

def test_implied_yields_recover_known_carry():
    times, qs = [0.25, 0.5, 1.0], [0.01, 0.02, 0.015]
    curve = _curve(times=times, qs=qs)
    implied = curve.implied_yields(FlatRateCurve(0.03))
    assert implied == pytest.approx(qs, abs=1e-12)


def test_to_dividend_yield_curve_nodes():
    curve = _curve()
    term = curve.to_dividend_yield_curve(FlatRateCurve(0.03))
    assert isinstance(term, TermStructureDividendYield)
    assert term.times == [0.25, 0.5, 1.0]


def test_negative_implied_carry_contango_marks():
    # spec demo marks: contango => negative implied q, must construct fine
    quotes = [
        IndexFuturesQuote("IC00", maturity=0.03, price=5008.0, multiplier=200.0),
        IndexFuturesQuote("IC01", maturity=0.10, price=5020.0, multiplier=200.0),
        IndexFuturesQuote("IC02", maturity=0.18, price=5036.0, multiplier=200.0),
        IndexFuturesQuote("IC03", maturity=0.32, price=5064.0, multiplier=200.0),
    ]
    curve = IndexFuturesCurve(underlying="IC", spot=5000.0, quotes=quotes)
    term = curve.to_dividend_yield_curve(FlatRateCurve(0.03))
    assert all(y < 0.0 for y in term.yields)


# --- spec test 2: bumping one contract changes only that node ---

def test_bump_contract_changes_only_one_node():
    curve = _curve()
    rate_curve = FlatRateCurve(0.03)
    base = curve.to_dividend_yield_curve(rate_curve).yields
    bumped = curve.bump_contract("IC01", 1.0).to_dividend_yield_curve(rate_curve).yields
    q1, f1, t1 = base[1], curve.quotes[1].price, curve.quotes[1].maturity
    expected_q1 = 0.03 - math.log((f1 + 1.0) / 100.0) / t1
    assert bumped[1] == pytest.approx(expected_q1, abs=1e-14)
    assert bumped[0] == base[0] and bumped[2] == base[2]
    # dq ~= -dF / (T * F) for small bumps
    assert bumped[1] - q1 == pytest.approx(-1.0 / (t1 * f1), rel=1e-2)


def test_bump_contract_validation():
    curve = _curve()
    with pytest.raises(ValidationError):
        curve.bump_contract("XX99", 1.0)
    with pytest.raises(ValidationError):
        curve.bump_contract("IC00", -1e9)  # non-positive bumped price


# --- construction validation (incl. spec test 9 beta + >=2 quotes) ---

def test_quote_validation():
    with pytest.raises(ValidationError):
        IndexFuturesQuote("", maturity=0.25, price=100.0, multiplier=200.0)
    with pytest.raises(ValidationError):
        IndexFuturesQuote("IC00", maturity=-0.1, price=100.0, multiplier=200.0)
    with pytest.raises(ValidationError):
        IndexFuturesQuote("IC00", maturity=0.25, price=0.0, multiplier=200.0)
    with pytest.raises(ValidationError):
        IndexFuturesQuote("IC00", maturity=0.25, price=100.0, multiplier=0.0)
    with pytest.raises(ValidationError):
        IndexFuturesQuote("IC00", maturity=0.25, price=100.0, multiplier=200.0, beta=1.5)


def test_curve_validation():
    q = _quotes_from_q(100.0, 0.03, [0.25, 0.5], [0.01, 0.02])
    with pytest.raises(ValidationError):
        IndexFuturesCurve(underlying="", spot=100.0, quotes=q)
    with pytest.raises(ValidationError):
        IndexFuturesCurve(underlying="IC", spot=0.0, quotes=q)
    with pytest.raises(ValidationError):
        IndexFuturesCurve(underlying="IC", spot=100.0, quotes=q[:1])  # < 2 quotes
    dup = [q[0], IndexFuturesQuote("IC00", maturity=0.5, price=101.0, multiplier=200.0)]
    with pytest.raises(ValidationError):
        IndexFuturesCurve(underlying="IC", spot=100.0, quotes=dup)
    unsorted = [q[1], q[0]]
    with pytest.raises(ValidationError):
        IndexFuturesCurve(underlying="IC", spot=100.0, quotes=unsorted)
    with pytest.raises(ValidationError):
        IndexFuturesCurve(underlying="IC", spot=100.0, quotes=q, interpolation="cubic")


# --- spec test 4: hedge hands conversion ---

def test_hedge_hands_synthetic():
    assert [hedge_hands(d, 10.0) for d in (10.0, 20.0, 30.0, 40.0)] == [
        -1.0, -2.0, -3.0, -4.0,
    ]
    assert hedge_hands(600.0, 300.0) == -2.0


def test_delta_per_hand_is_multiplier():
    curve = _curve()
    assert curve.delta_per_hand("IC00") == 200.0


def test_curve_quotes_snapshot_immune_to_caller_mutation():
    quotes = _quotes_from_q(100.0, 0.03, [0.25, 0.5], [0.01, 0.02])
    curve = IndexFuturesCurve(underlying="IC", spot=100.0, quotes=quotes)
    base = curve.to_dividend_yield_curve(FlatRateCurve(0.03)).yields
    quotes.append(
        IndexFuturesQuote("IC99", maturity=2.0, price=110.0, multiplier=200.0)
    )
    quotes[0] = IndexFuturesQuote("IC00", maturity=0.25, price=99.0, multiplier=200.0)
    assert isinstance(curve.quotes, tuple)
    assert len(curve.quotes) == 2
    assert curve.to_dividend_yield_curve(FlatRateCurve(0.03)).yields == base


def test_bump_term_yield_node():
    term = TermStructureDividendYield(times=[0.25, 0.5], yields=[0.01, 0.02])
    bumped = bump_term_yield_node(term, 1, 0.0001)
    assert bumped.yields == pytest.approx([0.01, 0.0201])
    assert term.yields == pytest.approx([0.01, 0.02])  # original untouched
    with pytest.raises(ValidationError):
        bump_term_yield_node(term, 2, 0.0001)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_index_futures_implied_carry.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quantark.asset.equity.market'`

- [ ] **Step 3: Implement**

Create `quantark/asset/equity/market/index_futures_curve.py`:

```python
"""
Index futures marks as an implied dividend/carry curve.

Continuous-compounding inversion of F = S * exp((r - q) * T):

    q(T_i) = r(T_i) - ln(F_i / S) / T_i

Observed basis is intentionally folded into implied carry (tradable futures
risk). Interpolation between nodes is linear in nodal q ("linear_q");
extrapolation outside [T_first, T_last] is flat in nodal q (the
TermStructureDividendYield convention).
"""
import math
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Optional, Sequence

from quantark.param.div.dividend_yield import TermStructureDividendYield
from quantark.param.rrf.rate_curve import RateCurve
from quantark.util.enum import FuturesCarryRiskMode
from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class IndexFuturesQuote:
    contract: str
    maturity: float
    price: float
    multiplier: float
    beta: float = 1.0
    expiry_date: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not self.contract:
            raise ValidationError("contract must be non-empty")
        if self.maturity <= 0.0:
            raise ValidationError("maturity must be positive")
        if self.price <= 0.0:
            raise ValidationError("price must be positive")
        if self.multiplier <= 0.0:
            raise ValidationError("multiplier must be positive")
        if self.beta != 1.0:
            raise ValidationError(
                "beta must be 1.0 (cross-hedge beta is not supported in v1; "
                "dPV/dF_i already carries the hedge relationship)"
            )


def hedge_hands(delta_bucket: float, delta_per_hand: float) -> float:
    """Fractional hedge hands: -delta_bucket / delta_per_hand. No rounding."""
    if delta_per_hand <= 0.0:
        raise ValidationError("delta_per_hand must be positive")
    return -delta_bucket / delta_per_hand


def bump_term_yield_node(
    term_div: TermStructureDividendYield, node_index: int, bump: float
) -> TermStructureDividendYield:
    """Return a copy of a term dividend curve with one node bumped."""
    if not 0 <= node_index < len(term_div.times):
        raise ValidationError(
            f"node_index {node_index} out of range for {len(term_div.times)} nodes"
        )
    yields = [float(y) for y in term_div.yields]
    yields[node_index] += float(bump)
    return TermStructureDividendYield(times=list(term_div.times), yields=yields)


@dataclass(frozen=True)
class IndexFuturesCurve:
    underlying: str
    spot: float
    quotes: Sequence[IndexFuturesQuote]
    mode: FuturesCarryRiskMode = FuturesCarryRiskMode.IMPLIED_FUTURES_CARRY
    interpolation: str = "linear_q"

    def __post_init__(self) -> None:
        # snapshot the quotes: a frozen curve must not be mutable through the
        # caller's original list after validation
        object.__setattr__(self, "quotes", tuple(self.quotes))
        if not self.underlying:
            raise ValidationError("underlying must be non-empty")
        if self.spot <= 0.0:
            raise ValidationError("spot must be positive")
        if len(self.quotes) < 2:
            raise ValidationError(
                "at least 2 futures quotes are required "
                "(TermStructureDividendYield needs >= 2 nodes; single-tenor "
                "markets are out of scope in v1)"
            )
        contracts = [q.contract for q in self.quotes]
        if len(set(contracts)) != len(contracts):
            raise ValidationError("futures contracts must be unique")
        maturities = [q.maturity for q in self.quotes]
        if any(
            maturities[i] >= maturities[i + 1] for i in range(len(maturities) - 1)
        ):
            raise ValidationError("quotes must have strictly increasing maturities")
        if self.interpolation != "linear_q":
            raise ValidationError(
                f"unsupported interpolation: {self.interpolation!r} "
                "(only 'linear_q' in v1)"
            )
        if not isinstance(self.mode, FuturesCarryRiskMode):
            raise ValidationError("mode must be a FuturesCarryRiskMode")

    def get_quote(self, contract: str) -> IndexFuturesQuote:
        for quote in self.quotes:
            if quote.contract == contract:
                return quote
        raise ValidationError(f"unknown futures contract: {contract!r}")

    def implied_yields(self, rate_curve: RateCurve) -> list:
        """q(T_i) = r(T_i) - ln(F_i / S) / T_i for each quote, in order."""
        return [
            rate_curve.get_rate(q.maturity)
            - math.log(q.price / self.spot) / q.maturity
            for q in self.quotes
        ]

    def to_dividend_yield_curve(
        self, rate_curve: RateCurve
    ) -> TermStructureDividendYield:
        return TermStructureDividendYield(
            times=[q.maturity for q in self.quotes],
            yields=self.implied_yields(rate_curve),
        )

    def bump_contract(self, contract: str, price_bump: float) -> "IndexFuturesCurve":
        quote = self.get_quote(contract)
        bumped_price = quote.price + price_bump
        if bumped_price <= 0.0:
            raise ValidationError(
                f"bumped price must be positive for {contract}: "
                f"{quote.price} + {price_bump} = {bumped_price}"
            )
        new_quotes = tuple(
            replace(q, price=bumped_price) if q.contract == contract else q
            for q in self.quotes
        )
        return replace(self, quotes=new_quotes)

    def delta_per_hand(self, contract: str) -> float:
        """PnL per hand per futures index point = multiplier (beta == 1.0)."""
        return self.get_quote(contract).multiplier
```

Create `quantark/asset/equity/market/__init__.py`:

```python
"""Equity market objects (futures curves for implied carry)."""
from .index_futures_curve import (
    IndexFuturesCurve,
    IndexFuturesQuote,
    bump_term_yield_node,
    hedge_hands,
)

__all__ = [
    "IndexFuturesCurve",
    "IndexFuturesQuote",
    "bump_term_yield_node",
    "hedge_hands",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_index_futures_implied_carry.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/market/ test/test_index_futures_implied_carry.py
git commit -m "feat(market): IndexFuturesCurve implied-carry curve with single-contract bump"
```

---

### Task 3: DeltaOneEngine futures dividend_rho

**Files:**
- Modify: `quantark/asset/equity/engine/analytical/deltaone_engine.py:237-304` (`_calculate_futures_greeks`)
- Test: `test/test_equity_futures_delta_buckets.py` (new file)

**Interfaces:**
- Consumes: `Futures(underlying, multiplier, maturity, basis, market_price)` product (`quantark/asset/equity/product/deltaone/futures.py`); `DeltaOneEngine(use_market_price=...)` whose `calculate_greeks(product, pricing_env)` dispatches to `_calculate_futures_greeks(product, pricing_env, S, T, r, q)`.
- Produces: `calculate_greeks(...)["dividend_rho"]` for futures — `0.0` in market-price mode, `−S·T·e^{(r−q)T}·0.01` (per 1% q change, per unit contract) in theoretical mode. Used by Task 8's portfolio test and Task 9's demo.

- [ ] **Step 1: Write the failing test**

Create `test/test_equity_futures_delta_buckets.py`:

```python
"""Futures-tenor bucket Greeks (spec tests 3-8) + futures rhoq (5, 6)."""
import math
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical.deltaone_engine import DeltaOneEngine
from quantark.asset.equity.market import IndexFuturesCurve, IndexFuturesQuote
from quantark.asset.equity.product.deltaone.futures import Futures
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError


def _env(spot=5000.0, r=0.03, q=0.01, vol=0.20):
    return PricingEnvironment(
        rate_curve=FlatRateCurve(r),
        valuation_date=datetime(2026, 7, 3),
        spot_quote=SpotQuote(spot),
        vol_surface=FlatVolSurface(vol),
        div_yield=ContinuousDividendYield(q),
    )


# --- spec test 5: theoretical futures rhoq ---

def test_futures_theoretical_dividend_rho():
    env = _env()
    fut = Futures(underlying="IC", multiplier=1.0, maturity=0.5)
    greeks = DeltaOneEngine().calculate_greeks(fut, env)
    S, T, r, q = 5000.0, 0.5, 0.03, 0.01
    expected = -S * T * math.exp((r - q) * T) * 0.01
    assert greeks["dividend_rho"] == pytest.approx(expected, rel=1e-12)
    assert greeks["dividend_rho"] < 0.0  # long theoretical futures: rhoq < 0


# --- spec test 6: market-price mode keeps model rhoq at zero ---

def test_futures_market_price_dividend_rho_zero():
    env = _env()
    fut = Futures(underlying="IC", multiplier=1.0, maturity=0.5, market_price=5100.0)
    greeks = DeltaOneEngine(use_market_price=True).calculate_greeks(fut, env)
    assert greeks["dividend_rho"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_equity_futures_delta_buckets.py -q`
Expected: FAIL — `KeyError: 'dividend_rho'`

- [ ] **Step 3: Implement**

In `_calculate_futures_greeks` (`deltaone_engine.py`), inside the
market-price branch, add before `return greeks`:

```python
            greeks["dividend_rho"] = 0.0  # market price independent of model carry
```

In the theoretical branch, after `greeks["rho"] = S * T * math.exp(carry_cost)`
add:

```python
        # Dividend rho: dF/dq = -S*T*exp((r-q)*T); basis term is q-independent.
        # Per 1% q change; negative for long futures (higher carry lowers F).
        greeks["dividend_rho"] = -S * T * math.exp(carry_cost) * 0.01
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_equity_futures_delta_buckets.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/analytical/deltaone_engine.py test/test_equity_futures_delta_buckets.py
git commit -m "feat(deltaone): futures dividend_rho (theoretical carry; zero in market-price mode)"
```

---

### Task 4: GreeksCalculator.calculate_futures_delta_buckets

**Files:**
- Modify: `quantark/asset/equity/riskmeasures/greeks_calculator.py` (insert new methods after `calculate_numerical_delta_q`, which starts at line 920)
- Test: `test/test_equity_futures_delta_buckets.py` (append)

**Interfaces:**
- Consumes: `IndexFuturesCurve.{to_dividend_yield_curve, bump_contract, delta_per_hand, quotes}`, `hedge_hands` (Task 2); existing calculator helpers `self._resolve_bump_engine(product, pricing_env, engine)`, `product.get_maturity(pricing_env)`; `deepcopy` (already imported in the module).
- Produces: `calculate_futures_delta_buckets(product, pricing_env, engine, futures_curve, *, mode=None, price_bump=1.0) -> list[dict]` (no `base_price` — the implied-carry base PV is always computed internally) with row keys `contract, maturity, future_price, price_bump, delta_bucket, delta_per_hand, hedge_hands, extrapolated_tail`. Used by Tasks 6, 8, 9.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_equity_futures_delta_buckets.py`:

```python
from copy import deepcopy

from quantark.asset.equity.engine.analytical.black_scholes_engine import (
    BlackScholesEngine,
)
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.util.enum import FuturesCarryRiskMode, OptionType


def _ic_curve(spot=5000.0):
    return IndexFuturesCurve(
        underlying="IC",
        spot=spot,
        quotes=[
            IndexFuturesQuote("IC00", maturity=0.03, price=5008.0, multiplier=200.0),
            IndexFuturesQuote("IC01", maturity=0.10, price=5020.0, multiplier=200.0),
            IndexFuturesQuote("IC02", maturity=0.18, price=5036.0, multiplier=200.0),
            IndexFuturesQuote("IC03", maturity=0.32, price=5064.0, multiplier=200.0),
        ],
    )


# --- spec test 3: vanilla futures-delta bucket matches manual finite difference ---

def test_futures_delta_bucket_matches_manual_fd():
    env = _env()
    curve = _ic_curve()
    engine = BlackScholesEngine()
    option = EuropeanVanillaOption(5000.0, OptionType.CALL, maturity=0.10)

    rows = GreeksCalculator().calculate_futures_delta_buckets(
        option, env, engine, curve, price_bump=1.0
    )
    row = next(r for r in rows if r["contract"] == "IC01")

    base_env = deepcopy(env)
    base_env.div_yield = curve.to_dividend_yield_curve(env.rate_curve)
    bumped_env = deepcopy(env)
    bumped_env.div_yield = curve.bump_contract("IC01", 1.0).to_dividend_yield_curve(
        env.rate_curve
    )
    manual = engine.price(option, bumped_env) - engine.price(option, base_env)
    assert row["delta_bucket"] == pytest.approx(manual / 1.0, rel=1e-12)
    assert row["hedge_hands"] == pytest.approx(-row["delta_bucket"] / 200.0, rel=1e-12)
    assert row["delta_per_hand"] == 200.0
    assert row["extrapolated_tail"] is False


def test_futures_delta_buckets_row_shape_and_signs():
    env = _env()
    curve = _ic_curve()
    rows = GreeksCalculator().calculate_futures_delta_buckets(
        EuropeanVanillaOption(5000.0, OptionType.CALL, maturity=0.10),
        env,
        BlackScholesEngine(),
        curve,
    )
    assert [r["contract"] for r in rows] == ["IC00", "IC01", "IC02", "IC03"]
    # long call: positive futures exposure => short-futures hedge on active buckets
    active = [r for r in rows if abs(r["delta_bucket"]) > 1e-10]
    assert active and all(r["hedge_hands"] < 0 for r in active)


def test_futures_delta_buckets_mode_rejection():
    env = _env()
    curve = _ic_curve()
    calc = GreeksCalculator()
    option = EuropeanVanillaOption(5000.0, OptionType.CALL, maturity=0.10)
    with pytest.raises(ValidationError):
        calc.calculate_futures_delta_buckets(
            option, env, BlackScholesEngine(), curve,
            mode=FuturesCarryRiskMode.MARKET_PRICE,
        )
    with pytest.raises(ValidationError):
        calc.calculate_futures_delta_buckets(
            option, env, BlackScholesEngine(), curve,
            mode=FuturesCarryRiskMode.THEORETICAL_CARRY,
        )
    with pytest.raises(ValidationError):
        calc.calculate_futures_delta_buckets(
            option, env, BlackScholesEngine(), curve, price_bump=0.0
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_equity_futures_delta_buckets.py -q`
Expected: FAIL — `AttributeError: 'GreeksCalculator' object has no attribute 'calculate_futures_delta_buckets'`

- [ ] **Step 3: Implement**

Insert into `greeks_calculator.py` after the `calculate_numerical_delta_q`
method body:

```python
    def calculate_futures_delta_buckets(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        futures_curve,
        *,
        mode=None,
        price_bump: float = 1.0,
    ) -> List[Dict[str, object]]:
        """
        Futures-tenor bucket deltas: delta_bucket_i = dPV / dF_i.

        Bumps one futures mark at a time, rebuilds the implied q(T) curve,
        and reprices (one-sided up bump). The base and bumped legs reuse the
        same engine/params unchanged, so MC engines with a fixed seed price
        with common random numbers. The base PV is always computed internally
        under the implied-carry environment (div_yield rebuilt from
        ``futures_curve``); no ``base_price`` parameter is accepted because a
        caller's PV under ``pricing_env.div_yield`` would silently shift every
        bucket.

        hedge_hands = -delta_bucket / delta_per_hand (fractional, unrounded).
        Rows are flagged ``extrapolated_tail`` when the product maturity lies
        outside the quoted node range (flat-extrapolated carry).
        """
        from quantark.asset.equity.market import hedge_hands as _hedge_hands
        from quantark.util.enum import FuturesCarryRiskMode

        resolved_mode = mode if mode is not None else futures_curve.mode
        if resolved_mode is not FuturesCarryRiskMode.IMPLIED_FUTURES_CARRY:
            raise ValidationError(
                "calculate_futures_delta_buckets requires IMPLIED_FUTURES_CARRY "
                f"mode, got {resolved_mode}"
            )
        if price_bump <= 0.0:
            raise ValidationError("price_bump must be positive")

        engine = self._resolve_bump_engine(product, pricing_env, engine)
        base_env = deepcopy(pricing_env)
        base_env.div_yield = futures_curve.to_dividend_yield_curve(
            pricing_env.rate_curve
        )
        base_price = engine.price(product, base_env)

        maturity = product.get_maturity(pricing_env)
        last_index = len(futures_curve.quotes) - 1
        rows: List[Dict[str, object]] = []
        for i, quote in enumerate(futures_curve.quotes):
            bumped_curve = futures_curve.bump_contract(quote.contract, price_bump)
            bumped_env = deepcopy(pricing_env)
            bumped_env.div_yield = bumped_curve.to_dividend_yield_curve(
                pricing_env.rate_curve
            )
            bumped_price = engine.price(product, bumped_env)
            delta_bucket = (bumped_price - base_price) / price_bump
            per_hand = futures_curve.delta_per_hand(quote.contract)
            extrapolated_tail = (
                i == last_index and maturity > quote.maturity
            ) or (i == 0 and maturity < quote.maturity)
            rows.append(
                {
                    "contract": quote.contract,
                    "maturity": quote.maturity,
                    "future_price": quote.price,
                    "price_bump": price_bump,
                    "delta_bucket": delta_bucket,
                    "delta_per_hand": per_hand,
                    "hedge_hands": _hedge_hands(delta_bucket, per_hand),
                    "extrapolated_tail": extrapolated_tail,
                }
            )
        return rows
```

(`List`, `Dict`, `Optional`, `deepcopy`, `ValidationError`, `BaseEngine`,
`BaseEquityProduct`, `PricingEnvironment` are already imported at module top —
verify and add any that are missing.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_equity_futures_delta_buckets.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/riskmeasures/greeks_calculator.py test/test_equity_futures_delta_buckets.py
git commit -m "feat(greeks): futures-tenor delta buckets from implied carry rebuild"
```

---

### Task 5: GreeksCalculator.calculate_futures_rhoq_buckets

**Files:**
- Modify: `quantark/asset/equity/riskmeasures/greeks_calculator.py` (insert after `calculate_futures_delta_buckets`)
- Test: `test/test_equity_futures_delta_buckets.py` (append)

**Interfaces:**
- Consumes: `bump_term_yield_node` (Task 2); `BucketedDividendYield(base, bucket_start, bucket_end, bump)` from `quantark/asset/equity/report/term_structure.py` (bumps spot yield on `bucket_start < t <= bucket_end`); `self._bump_config.div_bump`.
- Produces: `calculate_futures_rhoq_buckets(product, pricing_env, engine, futures_curve, *, mode=None, div_bump=None) -> list[dict]` (no `base_price`) with row keys `contract, maturity, future_price, div_bump, rhoq_bucket`. One-sided up bump scaled per 1% (`* (0.01/div_bump)`). Used by Tasks 8, 9.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_equity_futures_delta_buckets.py`:

```python
# --- rhoq buckets: implied node bump / theoretical BucketedDividendYield ---

def test_rhoq_buckets_implied_mode_matches_manual_node_bump():
    from quantark.asset.equity.market import bump_term_yield_node

    env = _env()
    curve = _ic_curve()
    engine = BlackScholesEngine()
    option = EuropeanVanillaOption(5000.0, OptionType.CALL, maturity=0.10)
    calc = GreeksCalculator()

    rows = calc.calculate_futures_rhoq_buckets(
        option, env, engine, curve, div_bump=0.0001
    )
    row = next(r for r in rows if r["contract"] == "IC01")

    base_div = curve.to_dividend_yield_curve(env.rate_curve)
    base_env = deepcopy(env)
    base_env.div_yield = base_div
    bumped_env = deepcopy(env)
    bumped_env.div_yield = bump_term_yield_node(base_div, 1, 0.0001)
    manual = (
        engine.price(option, bumped_env) - engine.price(option, base_env)
    ) * (0.01 / 0.0001)
    assert row["rhoq_bucket"] == pytest.approx(manual, rel=1e-12)
    assert row["rhoq_bucket"] < 0.0  # call: higher carry lowers forward


def test_rhoq_buckets_theoretical_mode_uses_bucketed_dividend():
    env = _env(q=0.01)
    curve = _ic_curve()
    engine = BlackScholesEngine()
    option = EuropeanVanillaOption(5000.0, OptionType.CALL, maturity=0.10)
    calc = GreeksCalculator()

    rows = calc.calculate_futures_rhoq_buckets(
        option, env, engine, curve,
        mode=FuturesCarryRiskMode.THEORETICAL_CARRY, div_bump=0.0001,
    )
    # option matures at 0.10 = IC01 node: spot-yield q(0.10) sits in the
    # (0.03, 0.10] bucket, so only IC01's interval bump moves the PV
    by_contract = {r["contract"]: r["rhoq_bucket"] for r in rows}
    assert by_contract["IC01"] != pytest.approx(0.0, abs=1e-9)
    assert by_contract["IC00"] == pytest.approx(0.0, abs=1e-9)
    assert by_contract["IC02"] == pytest.approx(0.0, abs=1e-9)
    assert by_contract["IC03"] == pytest.approx(0.0, abs=1e-9)
    # bucket rows decompose the scalar rhoq: sum == scalar dividend_rho
    scalar = calc.calculate_numerical_dividend_rho(
        option, env, engine, div_bump=0.0001
    )
    assert sum(by_contract.values()) == pytest.approx(scalar, rel=1e-6)


def test_rhoq_buckets_market_price_mode_rejected():
    env = _env()
    curve = _ic_curve()
    with pytest.raises(ValidationError):
        GreeksCalculator().calculate_futures_rhoq_buckets(
            EuropeanVanillaOption(5000.0, OptionType.CALL, maturity=0.10),
            env, BlackScholesEngine(), curve,
            mode=FuturesCarryRiskMode.MARKET_PRICE,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_equity_futures_delta_buckets.py -q`
Expected: FAIL — `AttributeError: ... no attribute 'calculate_futures_rhoq_buckets'`

- [ ] **Step 3: Implement**

Insert after `calculate_futures_delta_buckets`:

```python
    def calculate_futures_rhoq_buckets(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        futures_curve,
        *,
        mode=None,
        div_bump: Optional[float] = None,
    ) -> List[Dict[str, object]]:
        """
        Bucketed rhoq diagnostics per futures tenor (carry coordinate).
        The base PV is always computed internally (per-mode base environment);
        no ``base_price`` parameter — see calculate_futures_delta_buckets.

        One-sided **up** dividend bump, matching the scalar
        ``calculate_numerical_dividend_rho`` convention; output scaled to
        per-1% yield change via ``* (0.01 / div_bump)``.

        IMPLIED_FUTURES_CARRY: bumps one implied q(T_i) node at a time on the
        curve rebuilt from ``futures_curve``. THEORETICAL_CARRY: bumps
        ``pricing_env.div_yield`` on the interval (T_{i-1}, T_i] via
        ``BucketedDividendYield`` (the futures curve supplies metadata only).
        MARKET_PRICE is rejected: a zero bucket table can look like a real
        hedge result.
        """
        from quantark.asset.equity.market import bump_term_yield_node
        from quantark.asset.equity.report.term_structure import (
            BucketedDividendYield,
        )
        from quantark.param.div import ContinuousDividendYield
        from quantark.util.enum import FuturesCarryRiskMode

        resolved_mode = mode if mode is not None else futures_curve.mode
        if resolved_mode is FuturesCarryRiskMode.MARKET_PRICE:
            raise ValidationError(
                "calculate_futures_rhoq_buckets does not support MARKET_PRICE "
                "mode (model carry rhoq is zero by convention there)"
            )
        div_bump = div_bump if div_bump is not None else self._bump_config.div_bump
        if div_bump <= 0.0:
            raise ValidationError("div_bump must be positive")
        engine = self._resolve_bump_engine(product, pricing_env, engine)

        rows: List[Dict[str, object]] = []
        if resolved_mode is FuturesCarryRiskMode.IMPLIED_FUTURES_CARRY:
            base_div = futures_curve.to_dividend_yield_curve(pricing_env.rate_curve)
            base_env = deepcopy(pricing_env)
            base_env.div_yield = base_div
            base_price = engine.price(product, base_env)
            for i, quote in enumerate(futures_curve.quotes):
                bumped_env = deepcopy(pricing_env)
                bumped_env.div_yield = bump_term_yield_node(base_div, i, div_bump)
                bumped_price = engine.price(product, bumped_env)
                rows.append(
                    self._rhoq_bucket_row(quote, div_bump, base_price, bumped_price)
                )
        else:  # THEORETICAL_CARRY: pricing_env.div_yield is the carry source
            base_price = engine.price(product, pricing_env)
            base_div = pricing_env.div_yield
            if base_div is None:
                base_div = ContinuousDividendYield(0.0)
            edges = [0.0] + [q.maturity for q in futures_curve.quotes]
            for i, quote in enumerate(futures_curve.quotes):
                bumped_env = deepcopy(pricing_env)
                bumped_env.div_yield = BucketedDividendYield(
                    base=base_div,
                    bucket_start=edges[i],
                    bucket_end=edges[i + 1],
                    bump=div_bump,
                )
                bumped_price = engine.price(product, bumped_env)
                rows.append(
                    self._rhoq_bucket_row(quote, div_bump, base_price, bumped_price)
                )
        return rows

    @staticmethod
    def _rhoq_bucket_row(quote, div_bump, base_price, bumped_price):
        return {
            "contract": quote.contract,
            "maturity": quote.maturity,
            "future_price": quote.price,
            "div_bump": div_bump,
            "rhoq_bucket": (bumped_price - base_price) * (0.01 / div_bump),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_equity_futures_delta_buckets.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/riskmeasures/greeks_calculator.py test/test_equity_futures_delta_buckets.py
git commit -m "feat(greeks): bucketed rhoq — implied node bump + theoretical BucketedDividendYield"
```

---

### Task 6: Extrapolated tail, snowball interior sensitivity, MC common random numbers

**Files:**
- Test: `test/test_equity_futures_delta_buckets.py` (append; no production code expected — these pin behavior delivered by Tasks 2/4)

**Interfaces:**
- Consumes: `calculate_futures_delta_buckets` (Task 4); `SnowballOption`/`BarrierConfig` (`quantark/asset/equity/product/option/snowball_option.py`, `snowball_config.py`), `SnowballQuadEngine` (`quantark/asset/equity/engine/quad`), `EuropeanMCEngine` (`quantark/asset/equity/engine/mc`), `MCParams(seed, num_paths)` (`quantark/asset/equity/param/engine_params.py:234`).

- [ ] **Step 1: Write the tests (spec test 8 + CRN)**

Append to `test/test_equity_futures_delta_buckets.py`:

```python
# --- spec test 8: extrapolated tail concentrates in the last contract ---

def _short_curve(spot=100.0, r=0.03):
    times, qs = [0.1, 0.3, 0.6], [0.01, 0.015, 0.012]
    return IndexFuturesCurve(
        underlying="IC",
        spot=spot,
        quotes=[
            IndexFuturesQuote(
                f"IC{i:02d}", maturity=t, price=spot * math.exp((r - q) * t),
                multiplier=200.0,
            )
            for i, (t, q) in enumerate(zip(times, qs))
        ],
    )


def test_european_beyond_last_node_all_delta_in_last_bucket():
    env = _env(spot=100.0)
    curve = _short_curve()
    option = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.5)
    rows = GreeksCalculator().calculate_futures_delta_buckets(
        option, env, BlackScholesEngine(), curve
    )
    assert rows[0]["delta_bucket"] == pytest.approx(0.0, abs=1e-12)
    assert rows[1]["delta_bucket"] == pytest.approx(0.0, abs=1e-12)
    assert abs(rows[2]["delta_bucket"]) > 1e-4
    assert [r["extrapolated_tail"] for r in rows] == [False, False, True]


def test_snowball_keeps_interior_node_sensitivity_beyond_last_node():
    from quantark.asset.equity.engine.quad import SnowballQuadEngine
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig
    from quantark.asset.equity.product.option.snowball_option import SnowballOption
    from quantark.util.enum import ObservationType

    snowball = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=1.03,
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=0.75,
            ki_observation_type=ObservationType.CONTINUOUS,
        ),
        payoff_config=None,
        contract_multiplier=1.0,
        maturity=1.0,
        is_reverse=False,
    )
    env = _env(spot=100.0)
    curve = _short_curve()  # T_last = 0.6 < snowball maturity 1.0
    rows = GreeksCalculator().calculate_futures_delta_buckets(
        snowball, env, SnowballQuadEngine(), curve
    )
    # KO observations at 0.25/0.5 sit inside [0.1, 0.6]: interior nodes
    # carry genuine sensitivity through the term-aware engine
    interior = [r for r in rows if not r["extrapolated_tail"]]
    assert any(abs(r["delta_bucket"]) > 1e-6 for r in interior)
    assert rows[-1]["extrapolated_tail"] is True


def test_first_bucket_flagged_when_maturity_before_first_node():
    env = _env(spot=100.0)
    curve = _short_curve()
    option = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=0.05)
    rows = GreeksCalculator().calculate_futures_delta_buckets(
        option, env, BlackScholesEngine(), curve
    )
    assert rows[0]["extrapolated_tail"] is True
    assert rows[1]["extrapolated_tail"] is False


# --- MC common random numbers ---

def test_mc_delta_buckets_deterministic_and_near_analytic():
    from quantark.asset.equity.engine.mc import EuropeanMCEngine
    from quantark.asset.equity.param import MCParams

    env = _env(spot=100.0)
    curve = _short_curve()
    option = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=0.3)
    calc = GreeksCalculator()

    mc_engine = EuropeanMCEngine(MCParams(seed=42, num_paths=100_000))
    rows_a = calc.calculate_futures_delta_buckets(option, env, mc_engine, curve)
    rows_b = calc.calculate_futures_delta_buckets(option, env, mc_engine, curve)
    # fixed seed => common random numbers => bit-identical reruns
    assert [r["delta_bucket"] for r in rows_a] == [
        r["delta_bucket"] for r in rows_b
    ]

    analytic = calc.calculate_futures_delta_buckets(
        option, env, BlackScholesEngine(), curve
    )
    mc_mid, bs_mid = rows_a[1]["delta_bucket"], analytic[1]["delta_bucket"]
    assert mc_mid == pytest.approx(bs_mid, rel=0.05)
```

- [ ] **Step 2: Run the tests**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_equity_futures_delta_buckets.py -q`
Expected: all pass. If the MC tolerance test fails on `rel=0.05`, raise
`num_paths` to `200_000` — do not loosen the tolerance beyond 0.05.

- [ ] **Step 3: Commit**

```bash
git add test/test_equity_futures_delta_buckets.py
git commit -m "test(greeks): extrapolated tail, snowball interior nodes, MC common random numbers"
```

---

### Task 7: Portfolio aggregation helpers

**Files:**
- Create: `quantark/portfolio/equity/futures_buckets.py`
- Modify: `quantark/portfolio/equity/__init__.py` (add imports + `__all__` entries)
- Test: `test/test_equity_futures_delta_buckets.py` (append)

**Interfaces:**
- Produces: `aggregate_futures_delta_buckets(rows_by_position: Mapping[str, Sequence[dict]]) -> list[dict]` and `aggregate_futures_rhoq_buckets(...)` — same aggregation core: group by `contract` (first-appearance order), sum additive keys (`delta_bucket, hedge_hands, rhoq_bucket, futures_rhoq_bucket, net_delta_bucket, net_rhoq_bucket`) where present, carry non-additive keys (`maturity, future_price, delta_per_hand, extrapolated_tail`) from the first row and raise `ValidationError` on mismatch. Input rows are already position-level PV sensitivities (quantity applied by the caller).

- [ ] **Step 1: Write the failing tests**

Append to `test/test_equity_futures_delta_buckets.py`:

```python
# --- portfolio aggregation ---

def test_aggregate_futures_delta_buckets():
    from quantark.portfolio.equity import aggregate_futures_delta_buckets

    row = {
        "contract": "IC01", "maturity": 0.10, "future_price": 5020.0,
        "price_bump": 1.0, "delta_bucket": 600.0, "delta_per_hand": 300.0,
        "hedge_hands": -2.0, "extrapolated_tail": False,
    }
    other = dict(row, delta_bucket=300.0, hedge_hands=-1.0)
    out = aggregate_futures_delta_buckets({"pos_a": [row], "pos_b": [other]})
    assert len(out) == 1
    agg = out[0]
    assert agg["contract"] == "IC01"
    assert agg["delta_bucket"] == pytest.approx(900.0)
    assert agg["hedge_hands"] == pytest.approx(-3.0)
    assert agg["delta_per_hand"] == 300.0
    assert agg["maturity"] == 0.10


def test_aggregate_rejects_incompatible_metadata():
    from quantark.portfolio.equity import aggregate_futures_delta_buckets

    row = {
        "contract": "IC01", "maturity": 0.10, "future_price": 5020.0,
        "delta_bucket": 600.0, "delta_per_hand": 300.0, "hedge_hands": -2.0,
    }
    clash = dict(row, future_price=5021.0)  # different curve mark
    with pytest.raises(ValidationError):
        aggregate_futures_delta_buckets({"a": [row], "b": [clash]})


def test_aggregate_rhoq_buckets_multiple_contracts_keeps_order():
    from quantark.portfolio.equity import aggregate_futures_rhoq_buckets

    r1 = {"contract": "IC00", "maturity": 0.03, "future_price": 5008.0,
          "div_bump": 1e-4, "rhoq_bucket": -1.0}
    r2 = {"contract": "IC01", "maturity": 0.10, "future_price": 5020.0,
          "div_bump": 1e-4, "rhoq_bucket": -2.0}
    out = aggregate_futures_rhoq_buckets({"a": [r1, r2], "b": [dict(r2, rhoq_bucket=-3.0)]})
    assert [r["contract"] for r in out] == ["IC00", "IC01"]
    assert out[1]["rhoq_bucket"] == pytest.approx(-5.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_equity_futures_delta_buckets.py -q`
Expected: FAIL — `ImportError: cannot import name 'aggregate_futures_delta_buckets'`

- [ ] **Step 3: Implement**

Create `quantark/portfolio/equity/futures_buckets.py`:

```python
"""
Aggregate position-level futures-tenor bucket rows by contract.

Input rows are already position-level PV sensitivities (position quantity
applied before aggregation). Additive fields are summed; non-additive fields
(maturity, future_price, delta_per_hand, extrapolated_tail) must agree across
rows for a contract — a mismatch means incompatible futures curves and raises
ValidationError instead of silently aggregating.
"""
from typing import Dict, List, Mapping, Sequence

from quantark.util.exceptions import ValidationError

FuturesBucketRow = Dict[str, object]

_ADDITIVE_KEYS = (
    "delta_bucket",
    "hedge_hands",
    "rhoq_bucket",
    "futures_rhoq_bucket",
    "net_delta_bucket",
    "net_rhoq_bucket",
)
_MATCH_KEYS = ("maturity", "future_price", "delta_per_hand", "extrapolated_tail")
_MATCH_TOL = 1e-9


def _values_match(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if b is None:
        return False
    fa, fb = float(a), float(b)
    return abs(fa - fb) <= _MATCH_TOL * max(1.0, abs(fa))


def _aggregate(
    rows_by_position: Mapping[str, Sequence[FuturesBucketRow]],
) -> List[FuturesBucketRow]:
    order: List[str] = []
    grouped: Dict[str, List[FuturesBucketRow]] = {}
    for rows in rows_by_position.values():
        for row in rows:
            contract = row["contract"]
            if contract not in grouped:
                grouped[contract] = []
                order.append(contract)
            grouped[contract].append(row)

    out: List[FuturesBucketRow] = []
    for contract in order:
        rows = grouped[contract]
        first = rows[0]
        agg: FuturesBucketRow = {"contract": contract}
        for key in _MATCH_KEYS:
            if key not in first:
                continue
            for row in rows[1:]:
                if key in row and not _values_match(first[key], row[key]):
                    raise ValidationError(
                        f"incompatible futures curves for contract {contract}: "
                        f"{key} mismatch ({first[key]} vs {row[key]})"
                    )
            agg[key] = first[key]
        for key in _ADDITIVE_KEYS:
            if any(key in row for row in rows):
                agg[key] = sum(float(row[key]) for row in rows if key in row)
        out.append(agg)
    return out


def aggregate_futures_delta_buckets(
    rows_by_position: Mapping[str, Sequence[FuturesBucketRow]],
) -> List[FuturesBucketRow]:
    return _aggregate(rows_by_position)


def aggregate_futures_rhoq_buckets(
    rows_by_position: Mapping[str, Sequence[FuturesBucketRow]],
) -> List[FuturesBucketRow]:
    return _aggregate(rows_by_position)
```

In `quantark/portfolio/equity/__init__.py` add:

```python
from .futures_buckets import (
    aggregate_futures_delta_buckets,
    aggregate_futures_rhoq_buckets,
)
```

and append `'aggregate_futures_delta_buckets', 'aggregate_futures_rhoq_buckets',`
to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_equity_futures_delta_buckets.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add quantark/portfolio/equity/futures_buckets.py quantark/portfolio/equity/__init__.py test/test_equity_futures_delta_buckets.py
git commit -m "feat(portfolio): aggregate futures delta/rhoq buckets by contract"
```

---

### Task 8: Portfolio net rhoq offset by futures hedge (spec test 7)

**Files:**
- Test: `test/test_equity_futures_delta_buckets.py` (append; no production code)

**Interfaces:**
- Consumes: everything above. Futures hedge rhoq per hand uses the diagonal
  convention `futures_rhoq_per_hand_i = multiplier_i * dF_i/dq_i * 0.01`,
  realized via `DeltaOneEngine().calculate_greeks(...)["dividend_rho"] * multiplier`
  on an env whose `div_yield` is the implied curve.

- [ ] **Step 1: Write the test**

Append to `test/test_equity_futures_delta_buckets.py`:

```python
# --- spec test 7: portfolio net rhoq offset by futures hedge ---

def test_portfolio_net_rhoq_offset_by_futures_hedge():
    env = _env()  # spot 5000, r 3%
    curve = _ic_curve()
    engine = BlackScholesEngine()
    calc = GreeksCalculator()
    # maturity 0.18 == IC02 node: risk concentrates on one bucket
    option = EuropeanVanillaOption(5000.0, OptionType.CALL, maturity=0.18)

    delta_rows = calc.calculate_futures_delta_buckets(option, env, engine, curve)
    rhoq_rows = calc.calculate_futures_rhoq_buckets(
        option, env, engine, curve, div_bump=0.0001
    )
    row_d = next(r for r in delta_rows if r["contract"] == "IC02")
    row_q = next(r for r in rhoq_rows if r["contract"] == "IC02")
    assert abs(row_q["rhoq_bucket"]) > 1e-6  # standalone option rhoq non-zero

    # hedge futures: theoretical carry under the implied curve
    hedge_env = deepcopy(env)
    hedge_env.div_yield = curve.to_dividend_yield_curve(env.rate_curve)
    fut = Futures(underlying="IC", multiplier=1.0, maturity=0.18)
    fut_rhoq_per_hand = (
        DeltaOneEngine().calculate_greeks(fut, hedge_env)["dividend_rho"] * 200.0
    )
    assert fut_rhoq_per_hand < 0.0  # long futures rhoq negative

    hands = row_d["hedge_hands"]
    # net bucket delta ~ 0 by construction of hedge_hands
    net_delta = row_d["delta_bucket"] + hands * row_d["delta_per_hand"]
    assert net_delta == pytest.approx(0.0, abs=1e-9)
    # net rhoq reduced relative to standalone option rhoq
    net_rhoq = row_q["rhoq_bucket"] + hands * fut_rhoq_per_hand
    assert abs(net_rhoq) < abs(row_q["rhoq_bucket"])
```

- [ ] **Step 2: Run the test**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_equity_futures_delta_buckets.py -q`
Expected: all pass. Note: option `rhoq < 0` (call) and hedge is short
(`hands < 0`) with futures rhoq per hand `< 0`, so the hedge contribution
`hands * fut_rhoq_per_hand > 0` offsets the option's negative rhoq.

- [ ] **Step 3: Commit**

```bash
git add test/test_equity_futures_delta_buckets.py
git commit -m "test(portfolio): futures hedge offsets bucket delta exactly and reduces net rhoq"
```

---

### Task 9: End-to-end demo + full suite

**Files:**
- Create: `example/equity_futures_delta_buckets_demo.py`

**Interfaces:**
- Consumes: all public API from Tasks 1-7; `SnowballQuadEngine` for a
  deterministic structured-product pricing.

- [ ] **Step 1: Write the demo**

Create `example/equity_futures_delta_buckets_demo.py`:

```python
"""
Implied futures carry demo: futures marks -> implied q(T) -> bucket deltas ->
hedge hands -> post-hedge diagnostics. Deterministic synthetic data.

Run: python example/equity_futures_delta_buckets_demo.py
"""
from copy import deepcopy
from datetime import datetime

from quantark.asset.equity.engine.analytical.deltaone_engine import DeltaOneEngine
from quantark.asset.equity.engine.quad import SnowballQuadEngine
from quantark.asset.equity.market import IndexFuturesCurve, IndexFuturesQuote
from quantark.asset.equity.product.deltaone.futures import Futures
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.equity import (
    aggregate_futures_delta_buckets,
    aggregate_futures_rhoq_buckets,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType


def main() -> None:
    spot = 5000.0
    # 1. index futures curve (000905 / IC contracts, synthetic marks)
    curve = IndexFuturesCurve(
        underlying="IC",
        spot=spot,
        quotes=[
            IndexFuturesQuote("IC00", maturity=0.03, price=5008.0, multiplier=200.0),
            IndexFuturesQuote("IC01", maturity=0.10, price=5020.0, multiplier=200.0),
            IndexFuturesQuote("IC02", maturity=0.18, price=5036.0, multiplier=200.0),
            IndexFuturesQuote("IC03", maturity=0.32, price=5064.0, multiplier=200.0),
        ],
    )

    env = PricingEnvironment(
        rate_curve=FlatRateCurve(0.03),
        valuation_date=datetime(2026, 7, 3),
        spot_quote=SpotQuote(spot),
        vol_surface=FlatVolSurface(0.20),
    )
    # 2./3. implied dividend curve attached to the environment
    implied_div = curve.to_dividend_yield_curve(env.rate_curve)
    env.div_yield = implied_div
    print("implied q(T) nodes:")
    for t, y in zip(implied_div.times, implied_div.yields):
        print(f"  T={t:5.2f}  q={y:+.4%}")

    # 4. price a 000905 snowball on the implied-carry environment
    snowball = SnowballOption(
        initial_price=spot,
        strike=spot,
        barrier_config=BarrierConfig(
            ko_barrier=1.03,
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=0.75,
            ki_observation_type=ObservationType.CONTINUOUS,
        ),
        payoff_config=None,
        contract_multiplier=1.0,
        maturity=1.0,
        is_reverse=False,
    )
    engine = SnowballQuadEngine()
    calc = GreeksCalculator()
    pv = engine.price(snowball, env)
    print(f"\nsnowball PV (implied carry): {pv:.6f}")

    # 5. futures-tenor delta buckets -> hedge hands
    delta_rows = calc.calculate_futures_delta_buckets(snowball, env, engine, curve)
    print("\ncontract  F        dPV/dF_i     per-hand  hedge hands   extrapolated")
    print("          (hedge_hands_i = -delta_bucket_i / multiplier_i)")
    for r in delta_rows:
        print(
            f"{r['contract']}   {r['future_price']:8.1f} {r['delta_bucket']:+11.6f}"
            f"  {r['delta_per_hand']:8.1f}  {r['hedge_hands']:+11.6f}"
            f"   {r['extrapolated_tail']}"
        )

    # 6. bucketed rhoq diagnostics
    rhoq_rows = calc.calculate_futures_rhoq_buckets(snowball, env, engine, curve)
    print("\ncontract  rhoq_bucket (per +1% carry)")
    for r in rhoq_rows:
        print(f"{r['contract']}   {r['rhoq_bucket']:+.6f}")

    # 7./8. option + futures hedge portfolio: post-hedge diagnostics
    deltaone = DeltaOneEngine()
    print(
        "\ncontract  net delta bucket  option rhoq  hedge rhoq   net rhoq"
    )
    option_rows = {"snowball": delta_rows}
    hedge_rows = []
    for rd, rq in zip(delta_rows, rhoq_rows):
        hands = rd["hedge_hands"]
        fut = Futures(
            underlying="IC",
            multiplier=1.0,
            maturity=rd["maturity"],
        )
        per_hand_rhoq = (
            deltaone.calculate_greeks(fut, env)["dividend_rho"]
            * curve.get_quote(rd["contract"]).multiplier
        )
        hedge_rows.append(
            {
                "contract": rd["contract"],
                "maturity": rd["maturity"],
                "future_price": rd["future_price"],
                "delta_per_hand": rd["delta_per_hand"],
                "delta_bucket": hands * rd["delta_per_hand"],
                "hedge_hands": 0.0,
                "rhoq_bucket": hands * per_hand_rhoq,
            }
        )
    net_delta = aggregate_futures_delta_buckets(
        {"snowball": delta_rows, "hedge": hedge_rows}
    )
    net_rhoq = aggregate_futures_rhoq_buckets(
        {"snowball": rhoq_rows, "hedge": hedge_rows}
    )
    for nd, nq, rq in zip(net_delta, net_rhoq, rhoq_rows):
        hedge_rhoq = nq["rhoq_bucket"] - rq["rhoq_bucket"]
        print(
            f"{nd['contract']}   {nd['delta_bucket']:+16.10f}"
            f"  {rq['rhoq_bucket']:+11.6f}  {hedge_rhoq:+11.6f}"
            f"  {nq['rhoq_bucket']:+11.6f}"
        )
    print(
        "\nnote: snowball maturity 1.0 > last futures node 0.32 — the IC03 "
        "bucket includes flat-extrapolated tail carry (extrapolated_tail=True)."
    )


if __name__ == "__main__":
    main()
```

Note for the implementer: `aggregate_futures_delta_buckets` sums `delta_bucket`
across the option and hedge rows, where each hedge row's `delta_bucket` is
`hedge_hands * delta_per_hand` (the hedge's own PV sensitivity per futures
point); the aggregated `delta_bucket` is therefore the **net** bucket and
should print as ~0. The rhoq aggregation mixes rhoq rows (option) with hedge
rows carrying `rhoq_bucket`; both aggregators share the same core so the
metadata-consistency check applies. If the metadata check trips because rhoq
rows lack `delta_per_hand`, that is fine — keys absent from the first row are
skipped by design (rhoq rows come first in the mapping).

- [ ] **Step 2: Run the demo**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python example/equity_futures_delta_buckets_demo.py`
Expected: prints implied q nodes (all negative — contango marks), snowball PV,
four bucket rows with fractional hands, IC03 flagged `True`, and net delta
buckets ~0 with net rhoq smaller in magnitude than option rhoq per bucket.

- [ ] **Step 3: Run the full test suite**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/ -q -x --timeout=1200`
Expected: no regressions (baseline: 3681 passed, 17 skipped; this feature adds ~25 tests).

- [ ] **Step 4: Commit**

```bash
git add example/equity_futures_delta_buckets_demo.py
git commit -m "docs(example): implied futures carry -> bucket deltas -> hedge hands demo"
```

---

## Self-Review Notes

- Spec coverage: tests 1-9 all mapped (1, 2, 9 → Task 2; 3, 4 → Tasks 2/4; 5, 6 → Task 3 + mode rejections in Tasks 4/5; 7 → Task 8; 8 → Task 6). Demo steps 1-8 → Task 9. Enums/mode propagation → Tasks 1/4/5. Aggregation → Task 7. Extrapolation flags → Tasks 4/6. CRN → Tasks 4 (docstring contract) / 6 (test). `THEORETICAL_CARRY` rhoq via `BucketedDividendYield` → Task 5.
- The reporting-layer "rounded hands + residual" belongs to execution/reporting per the spec's non-goals; the demo prints fractional hands only — deliberate.
- `EquityDividendInputMode` is defined and exported but not consumed by v1 code paths — the spec defines it as a descriptive enum; do not invent uses.
