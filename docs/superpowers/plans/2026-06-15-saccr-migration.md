# SA-CCR Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the standalone `sa_ccr_demo` Basel SA-CCR EAD calculator into a self-contained QuantArk supporting module `quantark/saccr/`, applying QuantArk conventions, validated against the 5 Basel Annex 4a worked examples plus component tests.

**Architecture:** Self-contained module mirroring the SIMM precedent: `models/` (trade, netting set, enums), `parameters/` (immutable Basel Table 2 data), `engines/` (RC, PFE, supervisory maths, per-asset-class add-ons), `results/` (decomposed result), `calculator.py` orchestrator. Internal logic is ported from the validated source; raw `math.*`→`quantark.util.numerical.safe_*`, Abramowitz–Stegun CDF→`scipy.stats.norm.cdf`, `ValueError`→`quantark.util.exceptions.ValidationError`/`NumericalError`, with a domain-validate-before-safe-math policy.

**Tech Stack:** Python 3.10+, NumPy, SciPy (`scipy.stats.norm`), pytest (+xdist), `quantark.util.numerical`, `quantark.util.exceptions`, `quantark.util.enum.option_enums`.

**Source reference:** `/Users/fuxinyao/Documents/sa_ccr_demo/src/saccr/` (read each source file before porting the corresponding task).

**Test command (worktree shadows the editable install):**
`PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_saccr*.py -p no:cacheprovider -n0 -q`
(If `/Users/fuxinyao/quant-ark/.venv` is absent, use whichever venv has quantark installed; `PYTHONPATH=$PWD` forces the worktree source to win over the editable install.)

---

## Field mapping (spec-gate clarification #1): source → migrated

`SACCRTrade` (was `Trade`) — identical fields, renamed class. Required: `trade_id: str`,
`asset_class: AssetClass`, `notional: float`, `market_value: float`. Optional with defaults:
`maturity=1.0`, `start_date=0.0`, `end_date=1.0`, `currency=None`, `currency_pair=None`,
`reference_entity=None`, `is_index=False`, `credit_rating=None`, `index_grade=None`,
`commodity_hedging_set=None`, `commodity_type=None`, `is_option=False`, `option_type=None`,
`underlying_price=None`, `strike_price=None`, `exercise_date=None`, `is_cdo_tranche=False`,
`attachment_point=None`, `detachment_point=None`, `position=Position.LONG`,
`transaction_type=TransactionType.REGULAR`.
Per-asset-class required attrs (validated in `__post_init__`): IR→`currency`; FX→`currency_pair`;
Credit→`reference_entity` (+`index_grade` if index else `credit_rating`); Equity→`reference_entity`;
Commodity→`commodity_type`. Options→all option params. CDO→attach+detach.

`SACCRNettingSet` (was `NettingSet`) — identical fields, renamed class: `netting_set_id: str`,
`trades: list[SACCRTrade]`, `is_margined=False`, `threshold=0.0`, `minimum_transfer_amount=0.0`,
`variation_margin=0.0`, `independent_collateral_received=0.0`, `independent_collateral_posted=0.0`,
`independent_collateral_posted_segregated=0.0`, `mpor_days=10`, `net_collateral=None`.
Properties: `market_value`, `collateral` (=`C`), `nica`, `mpor_years`.

**Explicit collateral formulas (clarification #3):**
- `C = net_collateral` if provided, else `variation_margin + independent_collateral_received - independent_collateral_posted` (segregated posted excluded, para 143).
- `NICA = independent_collateral_received - independent_collateral_posted` (unsegregated only).
- `mpor_years = mpor_days / 250`.

**Margined EAD cap (clarification #2):** `EAD = min(EAD_margined, EAD_unmargined)` where
`EAD_unmargined` recomputes RC via `max(V-C,0)` and add-ons using the **unmargined maturity
factor** (`maturity_factor_unmargined(trade.maturity)`), same collateral `C`/`V`, same
multiplier formula. Implemented by building a throwaway unmargined `SACCRNettingSet` copy.

**Time/MPOR floors (clarification #5):** trade maturity floored at `10/250` in
`SACCRTrade.__post_init__`; `supervisory_duration` floors `E` at `10/250` and ensures `E>S`;
`maturity_factor_unmargined` floors `M` at `10/250` then caps at 1.0; `maturity_factor_margined`
uses `mpor_years` directly (NO 10-business-day maturity floor — MPOR floor is the regulator's
3/5/10/20-day rule, left to the caller via `mpor_days`). Tests pin both independently.

**Currency canonicalization (clarification #4):** `currency_pair` must match
`^[A-Z]{3}/[A-Z]{3}$`; else `ValidationError`. `EUR/USD` and `USD/EUR` are **distinct**
hedging-set keys (no alias normalization). `currency` (IR) must be non-empty `[A-Z]{3}`-ish
(non-empty string required, as in source).

**Immutability (clarification #6):** supervisory dict tables wrapped in
`types.MappingProxyType` at class level; scalar params are plain class constants.

---

## Phase 0 — Scaffold & enums

### Task 0: Package skeleton

**Files:**
- Create: `quantark/saccr/__init__.py` (temporary minimal; finalized in Task 12)
- Create: `quantark/saccr/models/__init__.py`, `quantark/saccr/parameters/__init__.py`,
  `quantark/saccr/engines/__init__.py`, `quantark/saccr/engines/addons/__init__.py`,
  `quantark/saccr/results/__init__.py`

- [ ] **Step 1:** Create the six `__init__.py` files. The package `__init__.py` starts empty
  (a comment only); sub-package `__init__.py` files start empty. Final exports added in Task 12.
- [ ] **Step 2: Commit**
```bash
git add quantark/saccr
git commit -m "feat(saccr): scaffold package skeleton"
```

### Task 1: Enums (`models/enums.py`)

**Files:**
- Create: `quantark/saccr/models/enums.py`
- Read first: source `models/enums.py`

- [ ] **Step 1:** Port `AssetClass`, `Position`, `CreditRating`, `IndexGrade`,
  `CommodityHedgingSet`, `CommodityType`, `TransactionType`, and the
  `COMMODITY_TYPE_TO_HEDGING_SET` dict **verbatim** (these are pure enums, no math).
  Do **not** define an `OptionType` here — it is reused from `quantark.util.enum.option_enums`.
- [ ] **Step 2: Write the failing test** `test/test_saccr_components.py`:
```python
import re
from quantark.saccr.models.enums import (
    AssetClass, Position, CreditRating, IndexGrade,
    CommodityHedgingSet, CommodityType, TransactionType,
    COMMODITY_TYPE_TO_HEDGING_SET,
)
from quantark.util.enum.option_enums import OptionType

def test_enums_and_commodity_map():
    assert {a.name for a in AssetClass} == {
        "INTEREST_RATE", "FX", "CREDIT", "EQUITY", "COMMODITY"}
    assert COMMODITY_TYPE_TO_HEDGING_SET[CommodityType.CRUDE_OIL] is CommodityHedgingSet.ENERGY
    assert COMMODITY_TYPE_TO_HEDGING_SET[CommodityType.GOLD] is CommodityHedgingSet.METALS
    assert OptionType.CALL is not OptionType.PUT
```
- [ ] **Step 3:** Run `... -k test_enums_and_commodity_map`; expect PASS.
- [ ] **Step 4: Commit** `git add -A && git commit -m "feat(saccr): port enums; reuse util OptionType"`

---

## Phase 1 — Parameters

### Task 2: Supervisory parameters (`parameters/supervisory.py`)

**Files:**
- Create: `quantark/saccr/parameters/supervisory.py`
- Read first: source `components/supervisory.py`

- [ ] **Step 1:** Port `SupervisoryParameters` with these changes:
  - Add `SACCR_VERSION = "Basel SA-CCR (BCBS d291, Mar 2014, rev Apr 2014)"`.
  - Wrap every dict table (`SF_CREDIT_SINGLE`, `SF_CREDIT_INDEX`, `SF_COMMODITY`,
    `VOLATILITY_COMMODITY`) in `types.MappingProxyType(...)`.
  - Keep paragraph/Table-2 references in docstrings.
  - **Getters raise `ValidationError`** (`from quantark.util.exceptions import ValidationError`)
    for an unknown asset class / commodity type / credit grade / index grade instead of
    returning a silent default. (Source used `.get(..., default)` — replace with explicit
    membership check + raise.)
- [ ] **Step 2: Write failing tests** (append to `test/test_saccr_components.py`):
```python
import pytest
from quantark.saccr.parameters.supervisory import SupervisoryParameters as SP
from quantark.saccr.models.enums import AssetClass, CreditRating, CommodityType
from quantark.util.exceptions import ValidationError

def test_supervisory_factors_values():
    assert SP.ALPHA == 1.4
    assert SP.FLOOR == 0.05
    assert SP.get_supervisory_factor(AssetClass.INTEREST_RATE) == 0.005
    assert SP.get_supervisory_factor(AssetClass.FX) == 0.04
    assert SP.get_supervisory_factor(
        AssetClass.CREDIT, credit_rating=CreditRating.AA) == 0.0038
    assert SP.get_supervisory_factor(
        AssetClass.COMMODITY, commodity_type=CommodityType.ELECTRICITY) == 0.40

def test_supervisory_tables_immutable():
    with pytest.raises(TypeError):
        SP.SF_CREDIT_SINGLE[CreditRating.AA] = 0.0  # MappingProxyType is read-only
```
- [ ] **Step 3:** Run those tests; expect PASS.
- [ ] **Step 4: Commit** `git add -A && git commit -m "feat(saccr): immutable, versioned supervisory params"`

---

## Phase 2 — Supervisory maths (engine helpers)

### Task 3: `engines/maths.py`

**Files:**
- Create: `quantark/saccr/engines/maths.py`
- Read first: source `utils/math_utils.py`

- [ ] **Step 1:** Port `supervisory_duration`, `maturity_factor_unmargined`,
  `maturity_factor_margined`, `supervisory_delta`, `supervisory_delta_option` with changes:
  - Delete `_normal_cdf` (A&S). Use `from scipy.stats import norm` → `norm.cdf(x)`.
  - Replace `math.exp/log/sqrt` with `safe_exp/safe_log/safe_sqrt` from
    `quantark.util.numerical`.
  - **Domain-validate before safe-math** (raise `ValidationError`):
    - `supervisory_duration` (explicit order, flooring must NOT mask bad inputs):
      (1) validate raw `start_date >= 0` and raw `end_date >= 0` → else `ValidationError`;
      (2) validate **raw** `end_date >= start_date` → else `ValidationError`
      (this catches e.g. `(0.02, 0.01)` BEFORE flooring can hide it);
      (3) set `e = max(end_date, 10/250)`; (4) require `e > start_date` → else
      `ValidationError`.
    - `maturity_factor_unmargined`: require `maturity` not None.
    - `supervisory_delta` (option branch): require `underlying_price > 0`,
      `strike_price > 0`, `supervisory_volatility > 0`, `exercise_date > 0`
      (use `validate_positive`); CDO branch: require attach/detach not None.
  - `d1 = safe_divide(safe_log(P/K) + 0.5*sigma**2*T, sigma*safe_sqrt(T))`.
- [ ] **Step 2: Write failing tests** (component values are hand-computable):
```python
import math
from scipy.stats import norm
from quantark.util.numerical import is_close
from quantark.saccr.engines.maths import (
    supervisory_duration, maturity_factor_unmargined, maturity_factor_margined,
    supervisory_delta,
)
from quantark.saccr.models.enums import Position
from quantark.util.enum.option_enums import OptionType

def test_supervisory_duration_10y():
    # (exp(0)-exp(-0.5))/0.05 = 7.8694
    assert is_close(supervisory_duration(0, 10), (1 - math.exp(-0.5)) / 0.05)

def test_maturity_factors():
    assert is_close(maturity_factor_unmargined(0.75), math.sqrt(0.75))
    assert is_close(maturity_factor_unmargined(5.0), 1.0)          # capped at 1y
    assert is_close(maturity_factor_margined(14/250), 1.5*math.sqrt(14/250))

def test_supervisory_delta_linear_and_option():
    assert supervisory_delta(Position.LONG) == 1.0
    assert supervisory_delta(Position.SHORT) == -1.0
    # bought put, P=0.06,K=0.05,T=1,vol=0.5 -> -N(-d1)
    d1 = (math.log(0.06/0.05) + 0.5*0.5**2*1) / (0.5*1)
    got = supervisory_delta(Position.LONG, is_option=True, option_type=OptionType.PUT,
        underlying_price=0.06, strike_price=0.05, exercise_date=1.0,
        supervisory_volatility=0.5)
    assert is_close(got, -float(norm.cdf(-d1)))

def test_supervisory_delta_rejects_bad_domain():
    import pytest
    from quantark.util.exceptions import ValidationError
    with pytest.raises(ValidationError):
        supervisory_delta(Position.LONG, is_option=True, option_type=OptionType.CALL,
            underlying_price=-1, strike_price=0.05, exercise_date=1.0,
            supervisory_volatility=0.5)

def test_supervisory_duration_rejects_inverted_dates_even_with_floor():
    import pytest
    from quantark.util.exceptions import ValidationError
    # raw end < start; flooring e to 10/250 would otherwise hide it -> must still raise
    with pytest.raises(ValidationError):
        supervisory_duration(0.02, 0.01)
    with pytest.raises(ValidationError):
        supervisory_duration(-1, 5)
```
- [ ] **Step 3:** Run; expect PASS. (The option test above already pins the delta to the
  EXACT `scipy.stats.norm.cdf`, which is the whole point of the CDF change.)
- [ ] **Step 4: CDF-correction documentation test** (documents WHY the legacy CDF was
  replaced — it was materially inaccurate, NOT a negligible refinement):
```python
def test_legacy_cdf_was_materially_inaccurate():
    """The source's hand-rolled _normal_cdf mixed erf coefficients without the /sqrt(2)
    scaling, giving errors up to ~3.7e-2 vs the exact normal CDF. This test documents
    that switching to scipy.stats.norm.cdf is a CORRECTION (so we must NOT assert the two
    agree). At the Example-1 swaption d1, legacy~0.2324 vs exact~0.2694."""
    def _legacy_cdf(x):  # verbatim copy of the removed source approximation
        a1,a2,a3,a4,a5,p = 0.254829592,-0.284496736,1.421413741,-1.453152027,1.061405429,0.3275911
        s = 1 if x>=0 else -1; x=abs(x); t=1/(1+p*x)
        y=1-(((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*math.exp(-x*x/2)
        return 0.5*(1+s*y)
    d1 = (math.log(0.06/0.05) + 0.5*0.5**2*1) / (0.5*1)
    assert abs(float(norm.cdf(-d1)) - _legacy_cdf(-d1)) > 1e-2   # materially different
    assert is_close(float(norm.cdf(-d1)), 0.269395, abs_tol=1e-5)  # exact value used
```
- [ ] **Step 5:** Run; expect PASS. Add a docstring note in `engines/maths.py` recording the
  correction (legacy max error ~3.7e-2). **Commit**
  `git add -A && git commit -m "feat(saccr): supervisory maths with exact scipy CDF + domain validation"`

---

## Phase 3 — Models

### Task 4: `models/trade.py` (`SACCRTrade`)

**Files:**
- Create: `quantark/saccr/models/trade.py`
- Read first: source `models/trade.py`

- [ ] **Step 1:** Port `Trade`→`SACCRTrade` dataclass (field list above). Changes:
  - `__post_init__` raises `ValidationError` (not `ValueError`) for every check.
  - Add `validate_positive(self.notional, "notional")` (reject non-positive notional).
  - Add `currency_pair` canonical-format check (`re.fullmatch(r"[A-Z]{3}/[A-Z]{3}", ...)`)
    when `asset_class == FX`.
  - Keep the 10-business-day maturity floor (`self.maturity = max(self.maturity, 10/250)`).
  - Add `end_date >= start_date >= 0` validation.
- [ ] **Step 2: Failing tests:**
```python
import pytest
from quantark.saccr.models.trade import SACCRTrade
from quantark.saccr.models.enums import AssetClass, Position
from quantark.util.exceptions import ValidationError

def test_trade_maturity_floor():
    t = SACCRTrade("T", AssetClass.INTEREST_RATE, 1e6, 0.0, currency="USD",
                   start_date=0, end_date=0.01, maturity=0.001)
    assert t.maturity == pytest.approx(10/250)

def test_trade_requires_currency_for_ir():
    with pytest.raises(ValidationError):
        SACCRTrade("T", AssetClass.INTEREST_RATE, 1e6, 0.0)

def test_trade_rejects_bad_ccy_pair_and_notional():
    with pytest.raises(ValidationError):
        SACCRTrade("T", AssetClass.FX, 1e6, 0.0, currency_pair="EURUSD")
    with pytest.raises(ValidationError):
        SACCRTrade("T", AssetClass.INTEREST_RATE, -1, 0.0, currency="USD")
```
- [ ] **Step 3:** Run; expect PASS. **Commit** `git add -A && git commit -m "feat(saccr): SACCRTrade model"`

### Task 5: `models/netting_set.py` (`SACCRNettingSet`)

**Files:**
- Create: `quantark/saccr/models/netting_set.py`
- Read first: source `models/netting_set.py`

- [ ] **Step 1:** Port `NettingSet`→`SACCRNettingSet`. Changes: `__post_init__` raises
  `ValidationError` for empty trades. Keep `market_value`/`collateral`/`nica`/`mpor_years`
  properties exactly per the explicit formulas above.
- [ ] **Step 2: Failing tests:**
```python
import pytest
from quantark.saccr.models.netting_set import SACCRNettingSet
from quantark.saccr.models.trade import SACCRTrade
from quantark.saccr.models.enums import AssetClass
from quantark.util.exceptions import ValidationError

def _t(mv): return SACCRTrade("T", AssetClass.INTEREST_RATE, 1e6, mv, currency="USD",
                              start_date=0, end_date=5, maturity=5)

def test_netting_set_aggregates_and_collateral():
    ns = SACCRNettingSet("NS", [_t(30), _t(-20)], is_margined=True,
        independent_collateral_received=150, independent_collateral_posted=0,
        net_collateral=200, mpor_days=14)
    assert ns.market_value == pytest.approx(10)
    assert ns.collateral == pytest.approx(200)        # net_collateral override
    assert ns.nica == pytest.approx(150)
    assert ns.mpor_years == pytest.approx(14/250)

def test_netting_set_rejects_empty():
    with pytest.raises(ValidationError):
        SACCRNettingSet("NS", [])
```
- [ ] **Step 3:** Run; expect PASS. **Commit**
  `git add -A && git commit -m "feat(saccr): SACCRNettingSet model"`

---

## Phase 4 — Components (RC, PFE) and add-ons

### Task 6: `engines/replacement_cost.py`

**Files:**
- Create: `quantark/saccr/engines/replacement_cost.py`
- Read first: source `components/replacement_cost.py`

- [ ] **Step 1:** Port `calculate_rc_unmargined`, `calculate_rc_margined` verbatim (only
  `max(...)` — no raw transcendental math). Keep para references.
- [ ] **Step 2: Failing tests:**
```python
from quantark.util.numerical import is_close
from quantark.saccr.engines.replacement_cost import (
    calculate_rc_unmargined, calculate_rc_margined)

def test_rc_unmargined(): assert is_close(calculate_rc_unmargined(60, 0), 60)
def test_rc_unmargined_floored(): assert is_close(calculate_rc_unmargined(-20, 0), 0)
def test_rc_margined(): assert is_close(calculate_rc_margined(80, 79.5, 0, 1, 0), 1)
```
- [ ] **Step 3:** Run; PASS. **Commit** `git add -A && git commit -m "feat(saccr): replacement cost"`

### Task 7: `engines/pfe.py`

**Files:**
- Create: `quantark/saccr/engines/pfe.py`
- Read first: source `components/pfe.py`

- [ ] **Step 1:** Port `calculate_multiplier`, `calculate_pfe`. Changes: `math.exp`→`safe_exp`;
  use `safe_divide` for the exponent denominator; keep `addon_aggregate<=0 → 1.0` and
  `v-c>=0 → 1.0` guards (these are regulatory, not fallbacks). `FLOOR` default 0.05.
- [ ] **Step 2: Failing tests:**
```python
from quantark.util.numerical import is_close
from quantark.saccr.engines.pfe import calculate_multiplier, calculate_pfe

def test_multiplier_undercollateralized_is_one():
    assert is_close(calculate_multiplier(10, 0, 100), 1.0)

def test_multiplier_overcollateralized():
    import math
    m = calculate_multiplier(-20, 0, 282)
    assert is_close(m, 0.05 + 0.95*math.exp(-20/(2*0.95*282)))

def test_pfe(): assert is_close(calculate_pfe(282, 0.965), 282*0.965)
```
- [ ] **Step 3:** Run; PASS. **Commit** `git add -A && git commit -m "feat(saccr): PFE multiplier"`

### Task 8: `engines/addons/base.py`

**Files:**
- Create: `quantark/saccr/engines/addons/base.py`
- Read first: source `addons/base.py`

- [ ] **Step 1:** Port `BaseAddOn` (ABC). `get_maturity_factor` imports
  `maturity_factor_unmargined/margined` from `quantark.saccr.engines.maths`. Add an abstract
  or default `hedging_set_breakdown()` contract note in the docstring: each add-on's
  `calculate` must also be able to report per-hedging-set contributions for `SACCRResult`
  (implemented in Task 11 via a `calculate_with_breakdown` returning `(total, {label: addon})`).
  Define `calculate_with_breakdown(trades, ns) -> tuple[float, dict[str,float]]` as the new
  contract; `calculate` returns just the total (calls `calculate_with_breakdown`).
- [ ] **Step 2: Commit** (no standalone test; covered via add-on tasks)
  `git add -A && git commit -m "feat(saccr): add-on base contract with hedging-set breakdown"`

### Task 9: Per-asset-class add-ons

**Files (port each; read source first):**
- Create: `quantark/saccr/engines/addons/interest_rate.py` (source `addons/interest_rate.py`)
- Create: `quantark/saccr/engines/addons/fx.py` (source `addons/fx.py`)
- Create: `quantark/saccr/engines/addons/credit.py` (source `addons/credit.py`)
- Create: `quantark/saccr/engines/addons/equity.py` (source `addons/equity.py`)
- Create: `quantark/saccr/engines/addons/commodity.py` (source `addons/commodity.py`)

- [ ] **Step 1:** Port each add-on. Common changes:
  - `math.sqrt`→`safe_sqrt`, `math.log`→`safe_log` (none expected outside maths), no `math.*`.
  - Imports from `quantark.saccr.*`.
  - Each implements `calculate_with_breakdown(trades, ns) -> (total, {hedging_set_label: addon})`
    and `calculate(...)` delegating to it. Hedging-set labels:
    IR→`f"IR:{currency}"`; FX→`f"FX:{currency_pair}"`; Credit→`"CREDIT"`;
    Equity→`"EQUITY"`; Commodity→`f"COMMODITY:{hedging_set.name}"`.
  - Commodity: unknown `commodity_type` (not in `COMMODITY_TYPE_TO_HEDGING_SET`) →
    `ValidationError` (no silent OTHER bucket). [If source silently defaulted, this is the
    intentional upgrade per spec §5.5.]
- [ ] **Step 2: Failing tests** — at least one MEANINGFUL NUMERIC assertion per asset class,
  anchored to the Basel Annex 4a sub-add-on figures (the most error-prone ported logic):
  IR Example 1 → AddOn_IR ≈ 347; Credit Example 2 → AddOn_Credit ≈ 282; Commodity Example 3
  → Energy ≈ 2041, Metals ≈ 1800 (total ≈ 3841); Equity → single-name SF=0.32 sanity.
```python
import pytest
from quantark.util.numerical import is_close
from quantark.util.exceptions import ValidationError
from quantark.saccr.engines.addons.interest_rate import InterestRateAddOn
from quantark.saccr.engines.addons.credit import CreditAddOn
from quantark.saccr.engines.addons.commodity import CommodityAddOn
from quantark.saccr.engines.addons.equity import EquityAddOn
from quantark.saccr.engines.addons.fx import FXAddOn
from quantark.saccr.parameters.supervisory import SupervisoryParameters as SP
from quantark.saccr.models.trade import SACCRTrade
from quantark.saccr.models.netting_set import SACCRNettingSet
from quantark.saccr.models.enums import (
    AssetClass, Position, CreditRating, IndexGrade, CommodityType)
from quantark.util.enum.option_enums import OptionType

def _ns(trades): return SACCRNettingSet("NS", trades, is_margined=False)

def test_ir_addon_matches_basel_example1():
    trades = [
        SACCRTrade("IR1", AssetClass.INTEREST_RATE, 10_000, 30, currency="USD",
                   start_date=0, end_date=10, maturity=10, position=Position.LONG),
        SACCRTrade("IR2", AssetClass.INTEREST_RATE, 10_000, -20, currency="USD",
                   start_date=0, end_date=4, maturity=4, position=Position.SHORT),
        SACCRTrade("IR3", AssetClass.INTEREST_RATE, 5_000, 50, currency="EUR",
                   start_date=1, end_date=11, maturity=5.5, position=Position.LONG,
                   is_option=True, option_type=OptionType.PUT, underlying_price=0.06,
                   strike_price=0.05, exercise_date=1.0),
    ]
    total, bd = InterestRateAddOn().calculate_with_breakdown(trades, _ns(trades))
    assert total == pytest.approx(347, rel=0.05)
    assert {"IR:USD", "IR:EUR"} <= set(bd) and is_close(sum(bd.values()), total)

def test_credit_addon_matches_basel_example2():
    trades = [
        SACCRTrade("CR1", AssetClass.CREDIT, 10_000, 20, reference_entity="A",
                   credit_rating=CreditRating.AA, start_date=0, end_date=3, maturity=3,
                   position=Position.LONG),
        SACCRTrade("CR2", AssetClass.CREDIT, 10_000, -40, reference_entity="B",
                   credit_rating=CreditRating.BBB, start_date=0, end_date=6, maturity=6,
                   position=Position.SHORT),
        SACCRTrade("CR3", AssetClass.CREDIT, 10_000, 0, reference_entity="CDX.IG",
                   is_index=True, index_grade=IndexGrade.IG, start_date=0, end_date=5,
                   maturity=5, position=Position.LONG),
    ]
    total, _ = CreditAddOn().calculate_with_breakdown(trades, _ns(trades))
    assert total == pytest.approx(282, rel=0.05)

def test_commodity_addon_matches_basel_example3():
    trades = [
        SACCRTrade("C1", AssetClass.COMMODITY, 10_000, -50, commodity_type=CommodityType.CRUDE_OIL,
                   start_date=0, end_date=0.75, maturity=0.75, position=Position.LONG),
        SACCRTrade("C2", AssetClass.COMMODITY, 20_000, -30, commodity_type=CommodityType.CRUDE_OIL,
                   start_date=0, end_date=2, maturity=2, position=Position.SHORT),
        SACCRTrade("C3", AssetClass.COMMODITY, 10_000, 100, commodity_type=CommodityType.SILVER,
                   start_date=0, end_date=5, maturity=5, position=Position.LONG),
    ]
    total, bd = CommodityAddOn().calculate_with_breakdown(trades, _ns(trades))
    assert total == pytest.approx(3841, rel=0.05)
    assert bd["COMMODITY:ENERGY"] == pytest.approx(2041, rel=0.05)
    assert bd["COMMODITY:METALS"] == pytest.approx(1800, rel=0.05)

def test_equity_single_name_addon_sign_and_sf():
    t = SACCRTrade("EQ", AssetClass.EQUITY, 10_000, 0, reference_entity="AAPL",
                   start_date=0, end_date=1, maturity=1, position=Position.LONG)
    total, bd = EquityAddOn().calculate_with_breakdown([t], _ns([t]))
    # single-name SF=0.32, MF=1 (1y), delta=+1, adj notional=10_000 -> addon ~ 0.32*10_000
    assert total == pytest.approx(0.32 * 10_000, rel=0.05) and "EQUITY" in bd

def test_fx_addon_value_and_distinct_pair_labels():
    # FX add-on: per-currency-pair hedging set, SF_FX=0.04, MF=1 for 1y, delta=+1.
    # A single long 1y FX trade -> addon ~ 0.04 * adjusted_notional (= notional).
    one = SACCRTrade("FX1", AssetClass.FX, 10_000, 0, currency_pair="EUR/USD",
                     start_date=0, end_date=1, maturity=1, position=Position.LONG)
    total1, bd1 = FXAddOn().calculate_with_breakdown([one], _ns([one]))
    assert total1 == pytest.approx(0.04 * 10_000, rel=0.05)
    assert "FX:EUR/USD" in bd1
    # EUR/USD and USD/EUR must NOT net into one hedging set (distinct keys, no offset)
    two = [one, SACCRTrade("FX2", AssetClass.FX, 10_000, 0, currency_pair="USD/EUR",
                           start_date=0, end_date=1, maturity=1, position=Position.SHORT)]
    total2, bd2 = FXAddOn().calculate_with_breakdown(two, _ns(two))
    assert {"FX:EUR/USD", "FX:USD/EUR"} <= set(bd2)
    assert total2 == pytest.approx(2 * 0.04 * 10_000, rel=0.05)   # no cross-pair offset

def test_commodity_unknown_type_rejected_via_supervisory_lookup():
    # CommodityType is a closed, fully-mapped enum, so an "unknown type" cannot be built
    # through SACCRTrade. Test the upgrade at the parameter boundary instead: the getter
    # must raise on a missing commodity_type rather than silently bucketing to OTHER.
    with pytest.raises(ValidationError):
        SP.get_supervisory_factor(AssetClass.COMMODITY, commodity_type=None)
```
- [ ] **Step 3:** Run; PASS. **Commit** `git add -A && git commit -m "feat(saccr): per-asset-class add-ons"`

---

## Phase 5 — Result, calculator, public API

### Task 10: `results/result.py` (`SACCRResult`)

**Files:**
- Create: `quantark/saccr/results/result.py`

- [ ] **Step 1:** Define the `SACCRResult` dataclass with fields: `ead`, `ead_uncapped`,
  `ead_capped: bool`, `rc`, `pfe`, `multiplier`, `alpha`, `addon_aggregate`,
  `addon_by_asset_class: dict[str,float]`, `addon_by_hedging_set: dict[str,float]`,
  `v`, `c`, `nica`, `is_margined: bool`.
- [ ] **Step 2: Commit** `git add -A && git commit -m "feat(saccr): SACCRResult decomposition"`

### Task 11: `calculator.py` (`SACCRCalculator`)

**Files:**
- Create: `quantark/saccr/calculator.py`
- Read first: source `calculator.py`

- [ ] **Step 1:** Port `SACCRCalculator`. Changes:
  - Import `SACCRResult` from `quantark.saccr.results.result`.
  - `addon_calculators` keyed by `AssetClass` → the new add-on classes.
  - Build `addon_by_asset_class` and `addon_by_hedging_set` via
    `calculate_with_breakdown`; aggregate.
  - **Explicit cap/result decomposition (no ambiguity):** the result's `rc`, `pfe`,
    `multiplier`, `addon_aggregate`, `addon_by_asset_class`, `addon_by_hedging_set` ALWAYS
    correspond to the **primary path** (margined path for margined sets; unmargined path
    otherwise). The cap only affects the final scalar `ead`. Concretely:
    ```python
    ead_primary = alpha * (rc + pfe)                 # rc/pfe from the primary path
    if netting_set.is_margined:
        # recompute an unmargined twin (RC=max(V-C,0); add-ons with
        # maturity_factor_unmargined(trade.maturity)); same V, C, multiplier formula)
        ead_unmargined_cap = alpha * (rc_unmargined + pfe_unmargined)
        ead = min(ead_primary, ead_unmargined_cap)
        ead_uncapped = ead_primary                    # margined EAD BEFORE the cap
        ead_capped = ead_unmargined_cap < ead_primary
    else:
        ead = ead_uncapped = ead_primary
        ead_capped = False
    ```
  - Populate all `SACCRResult` fields including `alpha=SupervisoryParameters.ALPHA`,
    `nica=netting_set.nica`.
- [ ] **Step 2: Failing test** (Example 1 smoke through the orchestrator). NOTE: imports come
  from INTERNAL modules so this task does not depend on Task 12's public exports:
```python
from quantark.saccr.calculator import SACCRCalculator
from quantark.saccr.models.trade import SACCRTrade
from quantark.saccr.models.netting_set import SACCRNettingSet
from quantark.saccr.models.enums import AssetClass, Position
def test_calculator_example1_smoke():
    ns = SACCRNettingSet("E1", [
        SACCRTrade("IR1", AssetClass.INTEREST_RATE, 10_000, 30, currency="USD",
                   start_date=0, end_date=10, maturity=10, position=Position.LONG),
    ], is_margined=False)
    r = SACCRCalculator().calculate(ns)
    assert r.alpha == 1.4 and r.is_margined is False and r.ead_capped is False
    assert "IR:USD" in r.addon_by_hedging_set
    assert r.ead_uncapped == r.ead     # unmargined: no cap applied
```
- [ ] **Step 3:** Run; PASS (uses internal imports only — independent of Task 12). **Commit**
  `git add -A && git commit -m "feat(saccr): SACCRCalculator orchestrator with cap + breakdowns"`

### Task 12: Public API (`__init__.py`)

**Files:**
- Modify: `quantark/saccr/__init__.py`

- [ ] **Step 1:** Export and `__all__`-list: `SACCRCalculator`, `SACCRResult`, `SACCRTrade`,
  `SACCRNettingSet`, `AssetClass`, `Position`, `CreditRating`, `IndexGrade`,
  `CommodityHedgingSet`, `CommodityType`, `TransactionType`, and re-export `OptionType`
  from `quantark.util.enum.option_enums`. Add `__version__` mirroring `SACCR_VERSION`.
- [ ] **Step 2: Failing test:**
```python
def test_public_api_surface():
    import quantark.saccr as m
    for name in ["SACCRCalculator","SACCRResult","SACCRTrade","SACCRNettingSet",
                 "AssetClass","Position","CreditRating","IndexGrade",
                 "CommodityHedgingSet","CommodityType","TransactionType","OptionType"]:
        assert hasattr(m, name), name
    assert set(m.__all__) >= {"SACCRCalculator","SACCRTrade","SACCRNettingSet"}
```
- [ ] **Step 3:** Run; PASS. **Commit** `git add -A && git commit -m "feat(saccr): public API + __all__"`

---

## Phase 6 — Basel examples & docs

### Task 13: Basel Annex 4a examples (`test/test_saccr.py`)

**Files:**
- Create: `test/test_saccr.py`
- Read first: source `tests/test_examples.py`

- [ ] **Step 1:** Port all 5 example classes, **replacing** `from saccr import (...)` with
  `from quantark.saccr import (...)`, `Trade`→`SACCRTrade`, `NettingSet`→`SACCRNettingSet`,
  and **deleting** the `sys.path.insert` lines. Keep the documented `pytest.approx`
  tolerances (rel=0.01/0.05/0.10 as in source).
- [ ] **Step 2:** Run `... test/test_saccr.py -q`; expect all 5 examples' assertions PASS
  (EAD ≈ 569 / 381 / 5406 / 936 / 1879). If the scipy-CDF change shifts a value beyond
  tolerance, investigate (must not happen given tolerances) — do NOT loosen tolerances
  without justification.
- [ ] **Step 3: Commit** `git add -A && git commit -m "test(saccr): port 5 Basel Annex 4a examples"`

### Task 14: Methodology doc + module CLAUDE.md + root registration

**Files:**
- Create: `quantark/saccr/doc/saccr_basel.md` (copy source `reference/full.md`)
- Create: `quantark/saccr/CLAUDE.md`
- Modify: `CLAUDE.md` (root) — add SA-CCR row to the supporting-modules table

- [ ] **Step 1:** Copy `reference/full.md` → `quantark/saccr/doc/saccr_basel.md`.
- [ ] **Step 2:** Write `quantark/saccr/CLAUDE.md`: overview, `EAD=α(RC+PFE)`, module
  structure, the time/currency/collateral/sign conventions (from spec §6), supervisory
  versioning, the scipy-CDF note, deferred portfolio-adapter note, and the test command.
- [ ] **Step 3:** Add to root `CLAUDE.md` supporting-modules table:
  `| SA-CCR | quantark/saccr/ | Basel SA-CCR counterparty EAD (RC+PFE); see saccr/CLAUDE.md |`
- [ ] **Step 4: Commit** `git add -A && git commit -m "docs(saccr): methodology doc, module + root CLAUDE.md"`

### Task 15: Example script

**Files:**
- Create: `example/saccr_demo.py`

- [ ] **Step 1:** Write a runnable demo (public API only) building Example 1's netting set,
  calling `SACCRCalculator().calculate(ns)`, and printing `ead/rc/pfe/multiplier` plus
  `addon_by_asset_class` and `addon_by_hedging_set` using
  `quantark.util.numerical.format_currency` where appropriate.
- [ ] **Step 2:** Run `PYTHONPATH=$PWD <venv>/bin/python example/saccr_demo.py`; expect
  it prints EAD ≈ 569 and a hedging-set breakdown, exit 0.
- [ ] **Step 3: Commit** `git add -A && git commit -m "docs(saccr): runnable example/saccr_demo.py"`

---

## Phase 7 — Full verification

### Task 16: Full test run + final sweep

- [ ] **Step 1:** Run the full SA-CCR suite:
  `PYTHONPATH=$PWD <venv>/bin/python -m pytest test/test_saccr.py test/test_saccr_components.py -q`
  Expect: all PASS.
- [ ] **Step 2:** Grep for convention violations in `quantark/saccr/`:
  `grep -rnE "\bmath\.(log|exp|sqrt)\b|raise ValueError|from saccr|import saccr" quantark/saccr` → expect **no matches**.
- [ ] **Step 3:** Confirm `example/saccr_demo.py` runs clean.
- [ ] **Step 4: Commit** any cleanup `git add -A && git commit -m "chore(saccr): final verification sweep"`

---

## Self-Review (completed by author)

- **Spec coverage:** module structure (Tasks 0–12), parameters immutability+version (Task 2),
  numerical+domain policy (Tasks 3,7,9), scipy CDF + A&S regression (Task 3), exceptions
  (Tasks 2–5,9), enums/OptionType reuse (Tasks 1,12), naming (Tasks 4,5), time/MPOR floors
  (Tasks 3,4), currency canonicalization (Task 4), collateral/NICA formulas (Task 5),
  result decomposition (Tasks 10,11), margined cap (Task 11), Basel tests (Task 13),
  component/edge tests (Tasks 1–9,12), docs+example+root registration (Tasks 14,15),
  deferred adapter (Task 14 CLAUDE.md). All spec sections mapped.
- **Spec-gate clarifications #1–#6:** addressed in the "Field mapping" preamble and Tasks
  2,4,5,11.
- **Placeholder scan:** none.
- **Type consistency:** `calculate_with_breakdown(trades, ns) -> (float, dict[str,float])`
  used consistently in Tasks 8,9,11; `SACCRResult` field names consistent Tasks 10,11.
