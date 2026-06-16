# Vanna-Volga Chain Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the Vanna-Volga FX engine into the standard `BaseFxEngine` / `FxPricingEnvironment` convention so VV one-touch and vanilla single-barrier (KO/KI + rebate) products flow through Greeks, VaR, and backtest.

**Architecture:** Approach A — the VV smile becomes env-anchored (intrinsic data = quotes + convention; spot/rates/tau sourced from the live environment) so sticky-delta is structural. A new `VannaVolgaBarrierEngine(BaseFxEngine)` adapts the existing functional pricers and a new Reiner-Rubinstein baseline + Castagna-Mercurio VV correction for vanilla barriers. Three `isinstance` vol-bump sites are taught about the VV surface.

**Tech Stack:** Python, NumPy, SciPy (`scipy.stats.norm`), pytest. Reuses existing `quantark.param.vol.vannavolga` (`compute_omega`, `SmileQuotes`, `FXEnv`) and `quantark.asset.fx.engine.analytical.vannavolga` (`price_vv_one_touch`, `attenuation`, `arbitrage`, `barrier_bs`).

**Spec:** `docs/superpowers/specs/2026-06-16-vannavolga-chain-integration-design.md`

**Conventions (from CLAUDE.md):** use `quantark.util.numerical` helpers (`is_zero`, `safe_*`) over raw float comparisons; no silent fallbacks (raise instead); exact semantics by default; canonical `quantark.*` imports. Run tests serial with `-n0` when debugging.

---

## File Structure

**New files:**
- `quantark/util/enum/fx_enums.py` — add `FxBarrierType` (modify existing file)
- `quantark/asset/fx/product/option/fx_one_touch_option.py` — `FxOneTouchOption`
- `quantark/asset/fx/product/option/fx_barrier_option.py` — `FxBarrierOption`
- `quantark/asset/fx/engine/analytical/vannavolga/vv_vanilla_barrier.py` — `price_vv_barrier`, `numeric_greeks_barrier`
- `quantark/asset/fx/engine/analytical/vannavolga/vv_barrier_engine.py` — `VannaVolgaBarrierEngine`
- `test/test_fx_one_touch.py`, `test/test_fx_barrier_option.py`, `test/test_reiner_rubinstein.py`, `test/test_vv_vanilla_barrier.py`, `test/test_vv_barrier_engine.py`, `test/test_vv_chain_integration.py`

**Modified files:**
- `quantark/param/vol/vannavolga/vv_surface.py` — add `rebound`, `with_quotes`
- `quantark/asset/fx/engine/analytical/vannavolga/barrier_bs.py` — add `reiner_rubinstein_barrier`
- `quantark/asset/fx/engine/base_fx_engine.py` — `_vol_scaled_env` VV branch
- `quantark/var/fx/revaluation.py` — `_shift_vol` VV branch
- `quantark/util/enum/__init__.py`, `quantark/asset/fx/product/option/__init__.py`, `quantark/asset/fx/engine/analytical/__init__.py`, `quantark/asset/fx/engine/analytical/vannavolga/__init__.py` — exports

---

## Phase 1 — Env-anchored smile (foundation)

### Task 1: Add `rebound` and `with_quotes` to `VannaVolgaVolSurface`

**Files:**
- Modify: `quantark/param/vol/vannavolga/vv_surface.py`
- Test: `test/test_vanna_volga_smile.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `test/test_vanna_volga_smile.py`:

```python
def test_rebound_reanchors_spot_and_rates():
    surf = VannaVolgaVolSurface(ENV, QUOTES, DeltaConvention.SPOT)
    bumped = surf.rebound(spot=1.25, rd=0.025, rf=0.012, tau=1.0)
    assert bumped is not surf
    assert bumped.env.spot == 1.25
    assert bumped.env.rd == 0.025
    assert bumped.env.rf == 0.012
    # quotes + convention are preserved (intrinsic smile data)
    assert bumped.quotes == QUOTES
    assert bumped.conv == surf.conv
    # original is untouched (immutable market data)
    assert surf.env.spot == 1.20


def test_with_quotes_shifts_all_three_quotes():
    surf = VannaVolgaVolSurface(ENV, QUOTES, DeltaConvention.SPOT)
    shifted = SmileQuotes(
        sigma_atm=QUOTES.sigma_atm + 0.01,
        rr25=QUOTES.rr25 + 0.01,
        bf25_2vol=QUOTES.bf25_2vol + 0.01,
    )
    bumped = surf.with_quotes(shifted)
    assert bumped is not surf
    assert bumped.quotes == shifted
    assert bumped.env.spot == surf.env.spot  # anchor unchanged
    assert surf.quotes == QUOTES               # original untouched
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -n0 test/test_vanna_volga_smile.py::test_rebound_reanchors_spot_and_rates test/test_vanna_volga_smile.py::test_with_quotes_shifts_all_three_quotes -v`
Expected: FAIL with `AttributeError: 'VannaVolgaVolSurface' object has no attribute 'rebound'`

- [ ] **Step 3: Implement the two methods**

In `quantark/param/vol/vannavolga/vv_surface.py`, add inside the `VannaVolgaVolSurface` class (after `__post_init__`):

```python
    def rebound(
        self, spot: float, rd: float, rf: float, tau: float
    ) -> "VannaVolgaVolSurface":
        """Return a new surface re-anchored to a different market snapshot.

        The intrinsic smile data (quotes, delta convention, premium flag) is
        preserved; only the FXEnv anchor changes. Immutable: the original
        surface is left untouched. This is how the risk chain re-anchors the
        smile under spot/rate bumps (sticky-delta).
        """
        return VannaVolgaVolSurface(
            env=FXEnv(spot=spot, rd=rd, rf=rf, tau=tau),
            quotes=self.quotes,
            conv=self.conv,
            premium_included_atm=self.premium_included_atm,
        )

    def with_quotes(self, quotes: SmileQuotes) -> "VannaVolgaVolSurface":
        """Return a new surface with shifted ATM/RR/BF quotes (vega bump).

        The market anchor is preserved; only the three quotes change. This is
        the full-quote vega path used by the chain's vol-bump helpers.
        """
        return VannaVolgaVolSurface(
            env=self.env,
            quotes=quotes,
            conv=self.conv,
            premium_included_atm=self.premium_included_atm,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest -n0 test/test_vanna_volga_smile.py -v`
Expected: PASS (new tests + all existing tests stay green)

- [ ] **Step 5: Commit**

```bash
git add quantark/param/vol/vannavolga/vv_surface.py test/test_vanna_volga_smile.py
git commit -m "feat(fx): env-anchored VannaVolgaVolSurface rebound/with_quotes

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 2 — Products

### Task 2: Add `FxBarrierType` enum

**Files:**
- Modify: `quantark/util/enum/fx_enums.py`
- Modify: `quantark/util/enum/__init__.py`
- Test: `test/test_fx_barrier_option.py` (created here, first assertion)

- [ ] **Step 1: Write the failing test**

Create `test/test_fx_barrier_option.py`:

```python
import pytest

from quantark.util.enum import FxBarrierType


def test_fx_barrier_type_values():
    assert FxBarrierType.KNOCK_OUT.value == "knock_out"
    assert FxBarrierType.KNOCK_IN.value == "knock_in"
    assert str(FxBarrierType.KNOCK_OUT) == "knock_out"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -n0 test/test_fx_barrier_option.py::test_fx_barrier_type_values -v`
Expected: FAIL with `ImportError: cannot import name 'FxBarrierType'`

- [ ] **Step 3: Implement the enum**

In `quantark/util/enum/fx_enums.py`, append:

```python
class FxBarrierType(Enum):
    """Knock direction for a single-barrier FX option.

    - KNOCK_OUT: the option ceases to exist if the barrier is touched.
    - KNOCK_IN: the option only comes into existence if the barrier is touched.
    """

    KNOCK_OUT = "knock_out"
    KNOCK_IN = "knock_in"

    def __str__(self):
        return self.value
```

In `quantark/util/enum/__init__.py`, change the fx_enums import line to include the new name:

```python
from .fx_enums import FxPayoutCurrency, FxBarrierType
```

And add `"FxBarrierType"` to that file's `__all__` list.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -n0 test/test_fx_barrier_option.py::test_fx_barrier_type_values -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add quantark/util/enum/fx_enums.py quantark/util/enum/__init__.py test/test_fx_barrier_option.py
git commit -m "feat(fx): add FxBarrierType enum

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `FxOneTouchOption` product

**Files:**
- Create: `quantark/asset/fx/product/option/fx_one_touch_option.py`
- Modify: `quantark/asset/fx/product/option/__init__.py`
- Test: `test/test_fx_one_touch.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_fx_one_touch.py`:

```python
import pytest

from quantark.asset.fx.product.option import FxOneTouchOption
from quantark.util.exceptions import ValidationError


def test_one_touch_construction_and_payoff():
    ot = FxOneTouchOption(barrier=1.30, is_up=True, payout=1.0, maturity=0.5)
    assert ot.barrier == 1.30
    assert ot.is_up is True
    assert ot.payout == 1.0
    # up-touch terminal payoff: pays at/above the barrier, else 0
    assert ot.get_payoff(1.31) == 1.0
    assert ot.get_payoff(1.29) == 0.0


def test_one_touch_rejects_bad_inputs():
    with pytest.raises(ValidationError):
        FxOneTouchOption(barrier=-1.0, is_up=True, maturity=0.5)
    with pytest.raises(ValidationError):
        FxOneTouchOption(barrier=1.30, is_up=True, payout=0.0, maturity=0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -n0 test/test_fx_one_touch.py -v`
Expected: FAIL with `ImportError: cannot import name 'FxOneTouchOption'`

- [ ] **Step 3: Implement the product**

Create `quantark/asset/fx/product/option/fx_one_touch_option.py`:

```python
"""
FX one-touch option (expiry-pay).
"""

from datetime import datetime
from typing import Optional

from quantark.util.exceptions import ValidationError
from ..base_fx_product import BaseFxProduct
from ..currency_pair import CurrencyPair


class FxOneTouchOption(BaseFxProduct):
    """
    FX one-touch: pays a fixed domestic amount if the barrier is touched
    over the option's life (settled at expiry).

    Attributes:
        barrier: Barrier level (quote per base).
        is_up: True for an up-barrier (touched from below), False for a
            down-barrier (touched from above).
        payout: Domestic-currency amount paid on touch.
    """

    def __init__(
        self,
        barrier: float,
        is_up: bool,
        payout: float = 1.0,
        currency_pair: Optional[CurrencyPair] = None,
        maturity: Optional[float] = None,
        expiry_date: Optional[datetime] = None,
        delivery: Optional[float] = None,
        delivery_date: Optional[datetime] = None,
    ):
        super().__init__(
            currency_pair=currency_pair,
            maturity=maturity,
            expiry_date=expiry_date,
            delivery=delivery,
            delivery_date=delivery_date,
        )
        self.barrier = barrier
        self.is_up = is_up
        self.payout = payout
        self.validate()

    def validate(self) -> None:
        if self.barrier <= 0:
            raise ValidationError(f"Barrier must be positive, got {self.barrier}")
        if self.payout <= 0:
            raise ValidationError(f"Payout must be positive, got {self.payout}")
        self._validate_maturity_inputs()

    def get_payoff(self, spot: float) -> float:
        """Terminal payoff for a spot at/beyond the barrier (MC cross-check).

        The analytic engine handles path-touch directly; this terminal form
        is used only by Monte-Carlo validation.
        """
        if spot < 0:
            raise ValidationError(f"Spot must be non-negative, got {spot}")
        touched = spot >= self.barrier if self.is_up else spot <= self.barrier
        return self.payout if touched else 0.0

    def __repr__(self):
        side = "up" if self.is_up else "down"
        return (
            f"FxOneTouchOption({self.currency_pair}, {side}, "
            f"H={self.barrier:.6f}, payout={self.payout:g})"
        )
```

In `quantark/asset/fx/product/option/__init__.py`, add the import and `__all__` entry:

```python
from .fx_one_touch_option import FxOneTouchOption
```
(add `'FxOneTouchOption'` to `__all__`)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -n0 test/test_fx_one_touch.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/fx/product/option/fx_one_touch_option.py quantark/asset/fx/product/option/__init__.py test/test_fx_one_touch.py
git commit -m "feat(fx): add FxOneTouchOption product

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `FxBarrierOption` product

**Files:**
- Create: `quantark/asset/fx/product/option/fx_barrier_option.py`
- Modify: `quantark/asset/fx/product/option/__init__.py`
- Test: `test/test_fx_barrier_option.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `test/test_fx_barrier_option.py`:

```python
from quantark.asset.fx.product.option import FxBarrierOption
from quantark.util.enum import OptionType, FxBarrierType
from quantark.util.exceptions import ValidationError


def test_barrier_option_construction():
    opt = FxBarrierOption(
        strike=1.20, barrier=1.30, is_up=True,
        knock_type=FxBarrierType.KNOCK_OUT, option_type=OptionType.CALL,
        maturity=0.5,
    )
    assert opt.strike == 1.20
    assert opt.barrier == 1.30
    assert opt.knock_type == FxBarrierType.KNOCK_OUT
    # unconditional vanilla terminal payoff (barrier handled by engine)
    assert opt.get_payoff(1.25) == pytest.approx(0.05)
    assert opt.get_payoff(1.15) == 0.0


def test_barrier_rejects_rebate_at_hit_for_knock_in():
    with pytest.raises(ValidationError):
        FxBarrierOption(
            strike=1.20, barrier=1.30, is_up=True,
            knock_type=FxBarrierType.KNOCK_IN, option_type=OptionType.CALL,
            maturity=0.5, rebate=0.01, rebate_at_hit=True,
        )


def test_barrier_rejects_bad_inputs():
    with pytest.raises(ValidationError):
        FxBarrierOption(
            strike=-1.0, barrier=1.30, is_up=True,
            knock_type=FxBarrierType.KNOCK_OUT, option_type=OptionType.CALL,
            maturity=0.5,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -n0 test/test_fx_barrier_option.py -v`
Expected: FAIL with `ImportError: cannot import name 'FxBarrierOption'`

- [ ] **Step 3: Implement the product**

Create `quantark/asset/fx/product/option/fx_barrier_option.py`:

```python
"""
FX single-barrier vanilla option (knock-out / knock-in, with optional rebate).
"""

from datetime import datetime
from typing import Optional

from quantark.util.enum import OptionType, FxBarrierType
from quantark.util.exceptions import ValidationError
from ..base_fx_product import BaseFxProduct
from ..currency_pair import CurrencyPair


class FxBarrierOption(BaseFxProduct):
    """
    FX single-barrier vanilla option.

    Attributes:
        strike: Strike rate (quote per base).
        barrier: Barrier level (quote per base).
        is_up: True for an up-barrier, False for a down-barrier.
        knock_type: KNOCK_OUT (dies on touch) or KNOCK_IN (born on touch).
        option_type: CALL or PUT of the underlying vanilla.
        rebate: Cash rebate amount in domestic currency. For KNOCK_OUT it is
            paid when the barrier is touched; for KNOCK_IN it is paid at expiry
            only if the barrier is never touched.
        rebate_at_hit: KNOCK_OUT only — pay the rebate at hit (True) vs at
            expiry (False). Must be False for KNOCK_IN.
    """

    def __init__(
        self,
        strike: float,
        barrier: float,
        is_up: bool,
        knock_type: FxBarrierType,
        option_type: OptionType,
        rebate: float = 0.0,
        rebate_at_hit: bool = False,
        currency_pair: Optional[CurrencyPair] = None,
        maturity: Optional[float] = None,
        expiry_date: Optional[datetime] = None,
        delivery: Optional[float] = None,
        delivery_date: Optional[datetime] = None,
    ):
        super().__init__(
            currency_pair=currency_pair,
            maturity=maturity,
            expiry_date=expiry_date,
            delivery=delivery,
            delivery_date=delivery_date,
        )
        self.strike = strike
        self.barrier = barrier
        self.is_up = is_up
        self.knock_type = knock_type
        self.option_type = option_type
        self.rebate = rebate
        self.rebate_at_hit = rebate_at_hit
        self.validate()

    def validate(self) -> None:
        if self.strike <= 0:
            raise ValidationError(f"Strike must be positive, got {self.strike}")
        if self.barrier <= 0:
            raise ValidationError(f"Barrier must be positive, got {self.barrier}")
        if not isinstance(self.knock_type, FxBarrierType):
            raise ValidationError(f"Invalid knock_type: {self.knock_type}")
        if not isinstance(self.option_type, OptionType):
            raise ValidationError(f"Invalid option_type: {self.option_type}")
        if self.rebate < 0:
            raise ValidationError(f"Rebate must be non-negative, got {self.rebate}")
        if self.knock_type == FxBarrierType.KNOCK_IN and self.rebate_at_hit:
            raise ValidationError(
                "rebate_at_hit is only valid for KNOCK_OUT; a knock-in rebate "
                "is paid at expiry when the barrier is never touched."
            )
        self._validate_maturity_inputs()

    def is_call(self) -> bool:
        return self.option_type == OptionType.CALL

    def get_payoff(self, spot: float) -> float:
        """Unconditional vanilla terminal payoff (barrier handled by engine)."""
        if spot < 0:
            raise ValidationError(f"Spot must be non-negative, got {spot}")
        if self.is_call():
            return max(spot - self.strike, 0.0)
        return max(self.strike - spot, 0.0)

    def __repr__(self):
        side = "up" if self.is_up else "down"
        return (
            f"FxBarrierOption({self.currency_pair}, {self.knock_type}, {side}, "
            f"{self.option_type}, K={self.strike:.6f}, H={self.barrier:.6f})"
        )
```

In `quantark/asset/fx/product/option/__init__.py`, add the import and `__all__` entry:

```python
from .fx_barrier_option import FxBarrierOption
```
(add `'FxBarrierOption'` to `__all__`)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -n0 test/test_fx_barrier_option.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/fx/product/option/fx_barrier_option.py quantark/asset/fx/product/option/__init__.py test/test_fx_barrier_option.py
git commit -m "feat(fx): add FxBarrierOption product

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 3 — Math: RR baseline + VV vanilla-barrier correction

### Task 5: Reiner-Rubinstein single-barrier pricer

**Files:**
- Modify: `quantark/asset/fx/engine/analytical/vannavolga/barrier_bs.py`
- Modify: `quantark/asset/fx/engine/analytical/vannavolga/__init__.py`
- Test: `test/test_reiner_rubinstein.py`

The correctness oracle is **in-out parity** (`KI + KO = vanilla`, rebate=0) plus
no-arbitrage bounds and barrier-far limits — these need no hand-transcribed
reference decimals. An optional QuantLib cross-check is added in Step 6.

- [ ] **Step 1: Write the failing test**

Create `test/test_reiner_rubinstein.py`:

```python
import math
import pytest

from quantark.asset.fx.engine.analytical.vannavolga.barrier_bs import (
    reiner_rubinstein_barrier,
)
from quantark.param.vol.vannavolga import GKInput, price_gk

S, RD, RF, VOL, TAU = 1.20, 0.02, 0.01, 0.10, 0.75


def _vanilla(strike, is_call):
    return price_gk(is_call, GKInput(S, strike, RD, RF, VOL, TAU))


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("is_up,barrier", [(True, 1.35), (False, 1.05)])
def test_in_out_parity(is_call, is_up, barrier):
    ko = reiner_rubinstein_barrier(
        S, 1.20, barrier, VOL, TAU, RD, RF,
        is_up=is_up, is_call=is_call, knock_in=False, rebate=0.0,
    )
    ki = reiner_rubinstein_barrier(
        S, 1.20, barrier, VOL, TAU, RD, RF,
        is_up=is_up, is_call=is_call, knock_in=True, rebate=0.0,
    )
    assert ko + ki == pytest.approx(_vanilla(1.20, is_call), rel=1e-9, abs=1e-9)


def test_ko_bounded_by_vanilla_and_nonneg():
    ko = reiner_rubinstein_barrier(
        S, 1.20, 1.35, VOL, TAU, RD, RF,
        is_up=True, is_call=True, knock_in=False, rebate=0.0,
    )
    assert 0.0 <= ko <= _vanilla(1.20, True) + 1e-12


def test_far_up_barrier_call_approaches_vanilla():
    # Up-and-out call with the barrier very far above spot -> ~ vanilla.
    ko = reiner_rubinstein_barrier(
        S, 1.20, 5.0, VOL, TAU, RD, RF,
        is_up=True, is_call=True, knock_in=False, rebate=0.0,
    )
    assert ko == pytest.approx(_vanilla(1.20, True), rel=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -n0 test/test_reiner_rubinstein.py -v`
Expected: FAIL with `ImportError: cannot import name 'reiner_rubinstein_barrier'`

- [ ] **Step 3: Implement the pricer**

Append to `quantark/asset/fx/engine/analytical/vannavolga/barrier_bs.py`:

```python
def reiner_rubinstein_barrier(
    spot: float,
    strike: float,
    barrier: float,
    vol: float,
    tau: float,
    rd: float,
    rf: float,
    is_up: bool,
    is_call: bool,
    knock_in: bool,
    rebate: float = 0.0,
    rebate_at_hit: bool = False,
) -> float:
    """Reiner-Rubinstein continuously-monitored single-barrier option value.

    Black-Scholes/Garman-Kohlhagen baseline (cost of carry b = rd - rf,
    domestic discounting r = rd). Covers all 8 KO/KI types via the standard
    A-F term decomposition with sign parameters phi (call/put) and eta
    (barrier side). Rebate: for KO paid at hit (rebate_at_hit) or at expiry;
    for KI paid at expiry if never knocked in.

    Reference: Haug, The Complete Guide to Option Pricing Formulas, 2nd ed.,
    single-barrier chapter.
    """
    if tau <= 0.0:
        # No remaining time: knock-in cannot trigger; knock-out is the vanilla
        # unless already breached. Handle terminal value directly.
        intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        breached = (is_up and spot >= barrier) or ((not is_up) and spot <= barrier)
        if knock_in:
            # Never knocked in over [0, T]: pay the expiry rebate; else the
            # option is alive and worth its intrinsic value.
            return intrinsic if breached else rebate
        # knock-out: if breached it is dead (rebate already due, at expiry now);
        # otherwise it survived and pays intrinsic.
        return rebate if breached else intrinsic
    if vol <= 0.0:
        raise ValueError(
            "reiner_rubinstein_barrier requires vol > 0; the zero-vol "
            "deterministic limit is not implemented (would need a separate "
            "monotonic-path treatment)."
        )

    phi = 1.0 if is_call else -1.0
    eta = 1.0 if not is_up else -1.0  # +1 down-barrier, -1 up-barrier

    b = rd - rf
    r = rd
    sqrt_t = math.sqrt(tau)
    vst = vol * sqrt_t
    mu = (b - 0.5 * vol * vol) / (vol * vol)
    lam = math.sqrt(mu * mu + 2.0 * r / (vol * vol))

    S, X, H, K = spot, strike, barrier, rebate
    carry_df = math.exp((b - r) * tau)  # e^{(b-r)T}
    dom_df = math.exp(-r * tau)

    x1 = math.log(S / X) / vst + (1.0 + mu) * vst
    x2 = math.log(S / H) / vst + (1.0 + mu) * vst
    y1 = math.log(H * H / (S * X)) / vst + (1.0 + mu) * vst
    y2 = math.log(H / S) / vst + (1.0 + mu) * vst
    z = math.log(H / S) / vst + lam * vst

    HS = H / S

    A = phi * S * carry_df * norm.cdf(phi * x1) - phi * X * dom_df * norm.cdf(phi * x1 - phi * vst)
    B = phi * S * carry_df * norm.cdf(phi * x2) - phi * X * dom_df * norm.cdf(phi * x2 - phi * vst)
    C = (
        phi * S * carry_df * (HS ** (2.0 * (mu + 1.0))) * norm.cdf(eta * y1)
        - phi * X * dom_df * (HS ** (2.0 * mu)) * norm.cdf(eta * y1 - eta * vst)
    )
    D = (
        phi * S * carry_df * (HS ** (2.0 * (mu + 1.0))) * norm.cdf(eta * y2)
        - phi * X * dom_df * (HS ** (2.0 * mu)) * norm.cdf(eta * y2 - eta * vst)
    )
    # Rebate paid at expiry (used by KI, and KO-at-expiry):
    E = K * dom_df * (
        norm.cdf(eta * x2 - eta * vst) - (HS ** (2.0 * mu)) * norm.cdf(eta * y2 - eta * vst)
    )
    # Rebate paid at hit (KO only):
    F = K * (
        (HS ** (mu + lam)) * norm.cdf(eta * z)
        + (HS ** (mu - lam)) * norm.cdf(eta * z - 2.0 * eta * lam * vst)
    )

    strike_above_barrier = X >= H

    if knock_in:
        # In-options: rebate E (paid at expiry if not knocked in).
        if is_call and not is_up:        # down-and-in call
            val = (C + E) if strike_above_barrier else (A - B + D + E)
        elif is_call and is_up:          # up-and-in call
            val = (A + E) if strike_above_barrier else (B - C + D + E)
        elif (not is_call) and not is_up:  # down-and-in put
            val = (B - C + D + E) if strike_above_barrier else (A + E)
        else:                            # up-and-in put
            val = (A - B + D + E) if strike_above_barrier else (C + E)
    else:
        # Out-options: the rebate is paid because the barrier IS knocked out.
        # At hit -> F (the touch term with lambda). At expiry -> the rebate
        # discounted times the touch probability. Do NOT reuse E here: E is the
        # knock-in "never touched" term and pays on the opposite states.
        if rebate_at_hit:
            reb = F
        else:
            p_hit = one_touch_hit_prob(S, H, vol, tau, b, is_up)
            reb = K * dom_df * p_hit
        if is_call and not is_up:        # down-and-out call
            val = (A - C + reb) if strike_above_barrier else (B - D + reb)
        elif is_call and is_up:          # up-and-out call
            val = reb if strike_above_barrier else (A - B + C - D + reb)
        elif (not is_call) and not is_up:  # down-and-out put
            val = (A - B + C - D + reb) if strike_above_barrier else reb
        else:                            # up-and-out put
            val = (B - D + reb) if strike_above_barrier else (A - C + reb)

    return float(val)
```

Add `"reiner_rubinstein_barrier"` to the `__all__` list in `barrier_bs.py`, and
re-export it from `quantark/asset/fx/engine/analytical/vannavolga/__init__.py`
(add to both the `from .barrier_bs import (...)` block and `__all__`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest -n0 test/test_reiner_rubinstein.py -v`
Expected: PASS (in-out parity holds to 1e-9; bounds and far-barrier limit hold)

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/fx/engine/analytical/vannavolga/barrier_bs.py quantark/asset/fx/engine/analytical/vannavolga/__init__.py test/test_reiner_rubinstein.py
git commit -m "feat(fx): Reiner-Rubinstein single-barrier BS pricer

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6: Add optional QuantLib cross-check**

Append to `test/test_reiner_rubinstein.py`:

```python
def test_rr_matches_quantlib_down_out_call():
    pytest.importorskip("QuantLib")
    import QuantLib as q
    today = q.Date(15, 6, 2026)
    q.Settings.instance().evaluationDate = today
    dc = q.Actual365Fixed()
    spot_h = q.QuoteHandle(q.SimpleQuote(S))
    r_ts = q.YieldTermStructureHandle(q.FlatForward(today, RD, dc))
    q_ts = q.YieldTermStructureHandle(q.FlatForward(today, RF, dc))
    vol_ts = q.BlackVolTermStructureHandle(
        q.BlackConstantVol(today, q.NullCalendar(), VOL, dc)
    )
    process = q.BlackScholesMertonProcess(spot_h, q_ts, r_ts, vol_ts)
    exercise = q.EuropeanExercise(today + q.Period(int(round(TAU * 365)), q.Days))
    payoff = q.PlainVanillaPayoff(q.Option.Call, 1.20)
    opt = q.BarrierOption(q.Barrier.DownOut, 1.05, 0.0, payoff, exercise)
    opt.setPricingEngine(q.AnalyticBarrierEngine(process))
    ql_price = opt.NPV()
    ours = reiner_rubinstein_barrier(
        S, 1.20, 1.05, VOL, TAU, RD, RF,
        is_up=False, is_call=True, knock_in=False, rebate=0.0,
    )
    assert ours == pytest.approx(ql_price, rel=2e-3)
```

Run: `python -m pytest -n0 test/test_reiner_rubinstein.py -v`
Expected: PASS, or the QL test SKIPPED if QuantLib is not installed.

- [ ] **Step 7: Commit**

```bash
git add test/test_reiner_rubinstein.py
git commit -m "test(fx): optional QuantLib cross-check for RR barrier

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: `price_vv_barrier` — Castagna-Mercurio VV correction

**Files:**
- Create: `quantark/asset/fx/engine/analytical/vannavolga/vv_vanilla_barrier.py`
- Modify: `quantark/asset/fx/engine/analytical/vannavolga/__init__.py`
- Test: `test/test_vv_vanilla_barrier.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_vv_vanilla_barrier.py`:

```python
import pytest

from quantark.param.vol.vannavolga import FXEnv, SmileQuotes, DeltaConvention
from quantark.asset.fx.engine.analytical.vannavolga.vv_vanilla_barrier import (
    price_vv_barrier,
)
from quantark.asset.fx.engine.analytical.vannavolga.barrier_bs import (
    reiner_rubinstein_barrier,
)

ENV = FXEnv(spot=1.20, rd=0.02, rf=0.01, tau=0.75)
SMILE = SmileQuotes(sigma_atm=0.10, rr25=-0.01, bf25_2vol=0.003)
FLAT = SmileQuotes(sigma_atm=0.10, rr25=0.0, bf25_2vol=0.0)


def test_vv_reduces_to_bs_when_smile_flat():
    res = price_vv_barrier(
        ENV, FLAT, strike=1.20, barrier=1.35, is_up=True,
        is_call=True, knock_in=False, conv=DeltaConvention.SPOT,
    )
    bs = reiner_rubinstein_barrier(
        ENV.spot, 1.20, 1.35, FLAT.sigma_atm, ENV.tau, ENV.rd, ENV.rf,
        is_up=True, is_call=True, knock_in=False,
    )
    assert res.vv == pytest.approx(bs, abs=1e-9)


def test_vv_ko_bounded_by_vanilla_and_nonneg():
    res = price_vv_barrier(
        ENV, SMILE, strike=1.20, barrier=1.35, is_up=True,
        is_call=True, knock_in=False, conv=DeltaConvention.SPOT,
    )
    assert 0.0 <= res.vv <= res.vanilla + 1e-12
    assert res.bstv == pytest.approx(
        reiner_rubinstein_barrier(
            ENV.spot, 1.20, 1.35, SMILE.sigma_atm, ENV.tau, ENV.rd, ENV.rf,
            is_up=True, is_call=True, knock_in=False,
        )
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -n0 test/test_vv_vanilla_barrier.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` for `price_vv_barrier`

- [ ] **Step 3: Implement the correction**

Create `quantark/asset/fx/engine/analytical/vannavolga/vv_vanilla_barrier.py`:

```python
"""
Vanna-Volga correction for vanilla single-barrier FX options.

VV price = BS_RR + p_vanna * vanna * Omega[vanna] + p_volga * volga * Omega[volga]
(vega term dropped by construction), survival-attenuated and clamped to the
no-arbitrage range [0, VV-vanilla] via enforce_single_barrier_arbitrage.

Reference: Castagna & Mercurio, "The Vanna-Volga method for implied
volatilities," Risk (2007); Wystup, FX Options and Structured Products.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from quantark.param.vol.vannavolga import (
    DeltaConvention,
    FXEnv,
    GKInput,
    SmileQuotes,
    choose_delta_convention,
    compute_omega,
    greeks_gk,
    price_gk,
    vv_adjustment_matrix,
)
from quantark.util.exceptions import ValidationError

from .arbitrage import BarrierPrices, enforce_single_barrier_arbitrage
from .attenuation import gamma_surv, p_vanna_p_volga_from_gamma
from .barrier_bs import reiner_rubinstein_barrier
from .vv_barrier import BarrierGamma, VVBarrierResult, _GAMMA_PRESETS

# FD bump sizes, consistent with the one-touch numeric greeks.
_H_SIGMA = 5e-4
_H_SPOT_REL = 1e-4


def numeric_greeks_barrier(
    env: FXEnv,
    sigma: float,
    strike: float,
    barrier: float,
    is_up: bool,
    is_call: bool,
    knock_in: bool,
    rebate: float,
    rebate_at_hit: bool,
) -> Dict[str, float]:
    """Numeric vega/vanna/volga of the RR barrier price via finite differences."""
    h_sig = min(_H_SIGMA, 0.5 * sigma) if sigma > 0.0 else _H_SIGMA

    def f(spot_: float, s_: float) -> float:
        return reiner_rubinstein_barrier(
            spot_, strike, barrier, s_, env.tau, env.rd, env.rf,
            is_up=is_up, is_call=is_call, knock_in=knock_in,
            rebate=rebate, rebate_at_hit=rebate_at_hit,
        )

    vega = (f(env.spot, sigma + h_sig) - f(env.spot, sigma - h_sig)) / (2.0 * h_sig)

    h_S = max(1e-6, env.spot * _H_SPOT_REL)

    def vega_wrt_S(spot_: float) -> float:
        return (f(spot_, sigma + h_sig) - f(spot_, sigma - h_sig)) / (2.0 * h_sig)

    vanna = (vega_wrt_S(env.spot + h_S) - vega_wrt_S(env.spot - h_S)) / (2.0 * h_S)
    volga = (
        f(env.spot, sigma + h_sig) - 2.0 * f(env.spot, sigma) + f(env.spot, sigma - h_sig)
    ) / (h_sig ** 2)
    return {"vega": float(vega), "vanna": float(vanna), "volga": float(volga)}


def _vv_vanilla(env: FXEnv, quotes: SmileQuotes, strike: float, is_call: bool,
                omega: np.ndarray) -> float:
    """Smile-consistent (VV-adjusted) vanilla price — the KO upper bound."""
    sigma = quotes.sigma_atm
    g = greeks_gk(is_call, GKInput(env.spot, strike, env.rd, env.rf, sigma, env.tau))
    base = price_gk(is_call, GKInput(env.spot, strike, env.rd, env.rf, sigma, env.tau))
    return base + vv_adjustment_matrix(g["vega"], g["vanna"], g["volga"], omega)


def price_vv_barrier(
    env: FXEnv,
    quotes: SmileQuotes,
    strike: float,
    barrier: float,
    is_up: bool,
    is_call: bool,
    knock_in: bool,
    rebate: float = 0.0,
    rebate_at_hit: bool = False,
    conv: Optional[DeltaConvention] = None,
    gamma_type: BarrierGamma = BarrierGamma.SURV,
    gamma_star: float = 0.95,
    premium_included_atm: bool = False,
) -> VVBarrierResult:
    """Vanna-Volga corrected vanilla single-barrier price."""
    if strike <= 0.0:
        raise ValidationError(f"strike must be positive, got {strike}")
    if barrier <= 0.0:
        raise ValidationError(f"barrier must be positive, got {barrier}")
    if env.tau < 0.0:
        raise ValidationError(f"time to expiry must be non-negative, got {env.tau}")
    gamma_type = BarrierGamma(gamma_type)
    sigma = quotes.sigma_atm

    bstv = reiner_rubinstein_barrier(
        env.spot, strike, barrier, sigma, env.tau, env.rd, env.rf,
        is_up=is_up, is_call=is_call, knock_in=knock_in,
        rebate=rebate, rebate_at_hit=rebate_at_hit,
    )

    if env.tau == 0.0:
        return VVBarrierResult(
            bstv=bstv, vv=bstv, gamma=0.0, p_vanna=0.0, p_volga=0.0,
            omega=np.zeros(3), greeks={"vega": 0.0, "vanna": 0.0, "volga": 0.0},
        )

    conv = DeltaConvention(conv) if conv is not None else choose_delta_convention(env.tau)
    gx = numeric_greeks_barrier(
        env, sigma, strike, barrier, is_up, is_call, knock_in, rebate, rebate_at_hit
    )
    omega, _ = compute_omega(env, quotes, conv, premium_included_atm=premium_included_atm)

    barrier_low = None if is_up else barrier
    barrier_high = barrier if is_up else None
    g = gamma_surv(env, barrier_low, barrier_high, sigma)
    a, b, c = _GAMMA_PRESETS[gamma_type]
    p_vanna, p_volga = p_vanna_p_volga_from_gamma(g, a, b, c, gamma_star)

    adj = p_vanna * gx["vanna"] * float(omega[1]) + p_volga * gx["volga"] * float(omega[2])
    raw = bstv + adj

    vanilla_vv = _vv_vanilla(env, quotes, strike, is_call, omega)
    if rebate > 0.0:
        # A rebate is an extra cash leg on top of the option, so a barrier with
        # a rebate can legitimately be worth MORE than the plain vanilla. The
        # vanilla upper-bound clamp would underprice it, so enforce only the
        # non-negativity floor for rebate structures.
        vv = max(raw, 0.0)
    else:
        clamped = enforce_single_barrier_arbitrage(
            BarrierPrices(vanilla=vanilla_vv, ko=raw)
        )
        vv = clamped.ko if clamped.ko is not None else max(raw, 0.0)

    result = VVBarrierResult(
        bstv=bstv, vv=float(vv), gamma=g, p_vanna=p_vanna, p_volga=p_volga,
        omega=omega, greeks=gx,
    )
    # Attach the vanilla bound for downstream tests/diagnostics.
    object.__setattr__(result, "vanilla", float(vanilla_vv))
    return result
```

Note: `VVBarrierResult` is a frozen dataclass, so the `vanilla` diagnostic is
attached via `object.__setattr__`. In the test it is read as `res.vanilla`.

Re-export `price_vv_barrier` and `numeric_greeks_barrier` from
`quantark/asset/fx/engine/analytical/vannavolga/__init__.py` (add to the
`from .vv_vanilla_barrier import (...)` block and `__all__`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest -n0 test/test_vv_vanilla_barrier.py -v`
Expected: PASS (flat-smile reduction exact; KO within [0, vanilla])

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/fx/engine/analytical/vannavolga/vv_vanilla_barrier.py quantark/asset/fx/engine/analytical/vannavolga/__init__.py test/test_vv_vanilla_barrier.py
git commit -m "feat(fx): Castagna-Mercurio VV correction for vanilla barriers

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 4 — Engine

### Task 7: `VannaVolgaBarrierEngine(BaseFxEngine)`

**Files:**
- Create: `quantark/asset/fx/engine/analytical/vannavolga/vv_barrier_engine.py`
- Modify: `quantark/asset/fx/engine/analytical/vannavolga/__init__.py`
- Modify: `quantark/asset/fx/engine/analytical/__init__.py`
- Test: `test/test_vv_barrier_engine.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_vv_barrier_engine.py`:

```python
from datetime import datetime

import pytest

from quantark.param import SpotQuote, FlatRateCurve
from quantark.param.vol.vannavolga import FXEnv, SmileQuotes, VannaVolgaVolSurface, DeltaConvention
from quantark.priceenv import FxPricingEnvironment
from quantark.asset.fx.product.option import FxOneTouchOption, FxBarrierOption
from quantark.asset.fx.engine.analytical import VannaVolgaBarrierEngine
from quantark.asset.fx.engine.analytical.vannavolga import price_vv_one_touch
from quantark.util.enum import OptionType, FxBarrierType
from quantark.util.exceptions import MarketDataError

VAL = datetime(2026, 6, 15)
TAU = 0.75
SMILE = SmileQuotes(sigma_atm=0.10, rr25=-0.01, bf25_2vol=0.003)


def _env():
    surface = VannaVolgaVolSurface(
        FXEnv(spot=1.20, rd=0.02, rf=0.01, tau=TAU), SMILE, DeltaConvention.SPOT
    )
    return FxPricingEnvironment(
        valuation_date=VAL,
        spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.02),
        foreign_curve=FlatRateCurve(rate=0.01),
        vol_surface=surface,
    )


def test_engine_one_touch_matches_function():
    eng = VannaVolgaBarrierEngine()
    ot = FxOneTouchOption(barrier=1.35, is_up=True, payout=1.0, maturity=TAU)
    price = eng.price(ot, _env())
    ref = price_vv_one_touch(
        FXEnv(spot=1.20, rd=0.02, rf=0.01, tau=TAU), SMILE, 1.35, True,
        conv=DeltaConvention.SPOT,
    ).vv
    assert price == pytest.approx(ref, rel=1e-9)


def test_engine_barrier_prices_and_greeks():
    eng = VannaVolgaBarrierEngine()
    opt = FxBarrierOption(
        strike=1.20, barrier=1.35, is_up=True,
        knock_type=FxBarrierType.KNOCK_OUT, option_type=OptionType.CALL,
        maturity=TAU,
    )
    greeks = eng.calculate_greeks(opt, _env())
    assert greeks["price"] > 0.0
    assert "delta" in greeks and "vega" in greeks
    assert all(v == v for v in greeks.values())  # no NaNs


def test_engine_requires_vv_surface():
    from quantark.param import FlatVolSurface
    env = _env()
    env.vol_surface = FlatVolSurface(volatility=0.10)
    eng = VannaVolgaBarrierEngine()
    ot = FxOneTouchOption(barrier=1.35, is_up=True, maturity=TAU)
    with pytest.raises(MarketDataError):
        eng.price(ot, env)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -n0 test/test_vv_barrier_engine.py -v`
Expected: FAIL with `ImportError: cannot import name 'VannaVolgaBarrierEngine'`

- [ ] **Step 3: Implement the engine**

Create `quantark/asset/fx/engine/analytical/vannavolga/vv_barrier_engine.py`:

```python
"""
Vanna-Volga FX barrier engine adapting the functional VV pricers to the
standard BaseFxEngine contract (price + bump-and-reprice Greeks).
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine, FxEngineParams
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.asset.fx.product.option import FxOneTouchOption, FxBarrierOption
from quantark.param.vol.vannavolga import FXEnv, VannaVolgaVolSurface
from quantark.priceenv import FxPricingEnvironment
from quantark.util.exceptions import MarketDataError, PricingError

from .vv_barrier import VVBarrierResult, price_vv_one_touch
from .vv_vanilla_barrier import price_vv_barrier


class VannaVolgaBarrierEngine(BaseFxEngine):
    """Prices FxOneTouchOption and FxBarrierOption via Vanna-Volga.

    Greeks are inherited from BaseFxEngine (bump-and-reprice); they work
    because the VannaVolgaVolSurface re-anchors sticky-delta under spot/rate
    bumps and shifts all quotes under vega bumps (see the chain wiring).
    """

    def __init__(self, params: Optional[FxEngineParams] = None):
        super().__init__(params=params)

    def _build_fx_env(self, product: BaseFxProduct, fx_env: FxPricingEnvironment) -> FXEnv:
        tau = product.get_maturity(fx_env)
        return FXEnv(
            spot=fx_env.spot,
            rd=fx_env.get_domestic_rate(tau),
            rf=fx_env.get_foreign_rate(tau),
            tau=tau,
        )

    def _surface(self, fx_env: FxPricingEnvironment) -> VannaVolgaVolSurface:
        surface = fx_env.vol_surface
        if not isinstance(surface, VannaVolgaVolSurface):
            raise MarketDataError(
                "VannaVolgaBarrierEngine requires a VannaVolgaVolSurface "
                f"vol surface; got {type(surface).__name__}."
            )
        return surface

    def price_details(
        self, product: BaseFxProduct, fx_env: FxPricingEnvironment
    ) -> VVBarrierResult:
        env = self._build_fx_env(product, fx_env)
        surface = self._surface(fx_env)
        if isinstance(product, FxOneTouchOption):
            result = price_vv_one_touch(
                env, surface.quotes, product.barrier, product.is_up,
                conv=surface.conv, premium_included_atm=surface.premium_included_atm,
            )
            # price_vv_one_touch returns a UNIT one-touch; scale by the
            # product payout so price and bump-and-reprice Greeks are correct.
            if product.payout != 1.0:
                result = dataclasses.replace(
                    result,
                    bstv=result.bstv * product.payout,
                    vv=result.vv * product.payout,
                )
            return result
        if isinstance(product, FxBarrierOption):
            return price_vv_barrier(
                env, surface.quotes, product.strike, product.barrier,
                product.is_up, product.is_call(),
                knock_in=(product.knock_type.value == "knock_in"),
                rebate=product.rebate, rebate_at_hit=product.rebate_at_hit,
                conv=surface.conv, premium_included_atm=surface.premium_included_atm,
            )
        raise PricingError(
            f"VannaVolgaBarrierEngine cannot price {type(product).__name__}; "
            "expected FxOneTouchOption or FxBarrierOption."
        )

    def price(self, product: BaseFxProduct, fx_env: FxPricingEnvironment) -> float:
        result = self.price_details(product, fx_env)
        return result.vv
```

Re-export from `quantark/asset/fx/engine/analytical/vannavolga/__init__.py`
(add `from .vv_barrier_engine import VannaVolgaBarrierEngine` and `__all__`),
and from `quantark/asset/fx/engine/analytical/__init__.py` (add the import and
`__all__` entry alongside the existing `vannavolga` exports).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest -n0 test/test_vv_barrier_engine.py -v`
Expected: PASS — note `test_engine_barrier_prices_and_greeks` exercises
bump-and-reprice Greeks, which depends on Phase 5 vol bumping for `vega`. If
`vega` raises here, proceed to Task 8/9 first, then re-run. (Delta/gamma/rho
work without Phase 5 because they bump spot/rates, not the surface.)

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/fx/engine/analytical/vannavolga/vv_barrier_engine.py quantark/asset/fx/engine/analytical/vannavolga/__init__.py quantark/asset/fx/engine/analytical/__init__.py test/test_vv_barrier_engine.py
git commit -m "feat(fx): VannaVolgaBarrierEngine (BaseFxEngine adapter)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 5 — Chain wiring (Greeks vol bump + VaR)

### Task 8: Teach `_vol_scaled_env` about the VV surface

**Files:**
- Modify: `quantark/asset/fx/engine/base_fx_engine.py:152-169`
- Test: `test/test_vv_chain_integration.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_vv_chain_integration.py`:

```python
from datetime import datetime

import pytest

from quantark.param import SpotQuote, FlatRateCurve
from quantark.param.vol.vannavolga import FXEnv, SmileQuotes, VannaVolgaVolSurface, DeltaConvention
from quantark.priceenv import FxPricingEnvironment
from quantark.asset.fx.product.option import FxBarrierOption
from quantark.asset.fx.engine.analytical import VannaVolgaBarrierEngine
from quantark.util.enum import OptionType, FxBarrierType

VAL = datetime(2026, 6, 15)
TAU = 0.75
SMILE = SmileQuotes(sigma_atm=0.10, rr25=-0.01, bf25_2vol=0.003)


def _env():
    surface = VannaVolgaVolSurface(
        FXEnv(spot=1.20, rd=0.02, rf=0.01, tau=TAU), SMILE, DeltaConvention.SPOT
    )
    return FxPricingEnvironment(
        valuation_date=VAL,
        spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.02),
        foreign_curve=FlatRateCurve(rate=0.01),
        vol_surface=surface,
    )


def _barrier():
    return FxBarrierOption(
        strike=1.20, barrier=1.35, is_up=True,
        knock_type=FxBarrierType.KNOCK_OUT, option_type=OptionType.CALL,
        maturity=TAU,
    )


def test_vega_through_vv_surface_is_finite_and_nonzero():
    eng = VannaVolgaBarrierEngine()
    greeks = eng.calculate_greeks(_barrier(), _env())
    assert greeks["vega"] == greeks["vega"]      # not NaN
    assert abs(greeks["vega"]) > 0.0             # vol bump actually moved price
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -n0 test/test_vv_chain_integration.py::test_vega_through_vv_surface_is_finite_and_nonzero -v`
Expected: FAIL with `PricingError: Vol bumping not supported for surface type VannaVolgaVolSurface`

- [ ] **Step 3: Add the VV branch**

In `quantark/asset/fx/engine/base_fx_engine.py`, edit `_vol_scaled_env` (currently lines 152-169). Add a VV branch before the `else` that raises. The full updated method:

```python
    def _vol_scaled_env(
        self, fx_env: FxPricingEnvironment, factor: float
    ) -> FxPricingEnvironment:
        env = deepcopy(fx_env)
        surface = fx_env.vol_surface
        if isinstance(surface, FlatVolSurface):
            env.vol_surface = FlatVolSurface(volatility=surface.volatility * factor)
        elif isinstance(surface, TermStructureVolSurface):
            env.vol_surface = TermStructureVolSurface(
                times=list(surface.times),
                vols=[float(v) * factor for v in surface.vols],
            )
        elif isinstance(surface, VannaVolgaVolSurface):
            # Full-quote vega bump: scale ATM/RR/BF together so the whole smile
            # shifts (sticky-delta). RR/BF can be zero or negative, so scaling
            # by `factor` is the right multiplicative bump for all three.
            q = surface.quotes
            env.vol_surface = surface.with_quotes(
                SmileQuotes(
                    sigma_atm=q.sigma_atm * factor,
                    rr25=q.rr25 * factor,
                    bf25_2vol=q.bf25_2vol * factor,
                )
            )
        else:
            raise PricingError(
                f"Vol bumping not supported for surface type "
                f"{type(surface).__name__}"
            )
        return env
```

Add the imports near the top of `base_fx_engine.py` (with the existing
`from quantark.param import FlatVolSurface, TermStructureVolSurface`):

```python
from quantark.param.vol.vannavolga import VannaVolgaVolSurface, SmileQuotes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -n0 test/test_vv_chain_integration.py::test_vega_through_vv_surface_is_finite_and_nonzero -v`
Expected: PASS

Then re-run the engine test that needs vega:
Run: `python -m pytest -n0 test/test_vv_barrier_engine.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/fx/engine/base_fx_engine.py test/test_vv_chain_integration.py
git commit -m "feat(fx): VV-surface vega bumping in BaseFxEngine

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: Teach FX VaR `_shift_vol` about the VV surface

**Files:**
- Modify: `quantark/var/fx/revaluation.py` (`_shift_vol`)
- Test: `test/test_vv_chain_integration.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `test/test_vv_chain_integration.py`:

```python
from quantark.var.fx.revaluation import bump_env


def test_var_bump_env_handles_vv_vol_shift():
    env = _env()
    bumped = bump_env(env, spot_return=0.0, vol_change=0.01)
    # The VV surface survives a vol shift and the ATM vol moved by +0.01.
    assert isinstance(bumped.vol_surface, VannaVolgaVolSurface)
    assert bumped.vol_surface.quotes.sigma_atm == pytest.approx(
        SMILE.sigma_atm + 0.01
    )
    # Original env is untouched.
    assert env.vol_surface.quotes.sigma_atm == pytest.approx(SMILE.sigma_atm)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -n0 test/test_vv_chain_integration.py::test_var_bump_env_handles_vv_vol_shift -v`
Expected: FAIL with `ValidationError: Cannot bump FX vol surface type VannaVolgaVolSurface for VaR.`

- [ ] **Step 3: Add the VV branch**

In `quantark/var/fx/revaluation.py`, edit `_shift_vol`. VaR vol shifts are
**additive** (in vol points). Shift the ATM level; keep RR/BF (the skew/smile
shape) fixed — an additive ATM shift is the standard parallel vol move. The
updated function:

```python
def _shift_vol(surface, change: float):
    if is_zero(change) or surface is None:
        return surface
    if isinstance(surface, FlatVolSurface):
        return FlatVolSurface(volatility=max(_MIN_VOL, surface.volatility + change))
    if isinstance(surface, TermStructureVolSurface):
        return TermStructureVolSurface(
            times=list(surface.times),
            vols=[max(_MIN_VOL, float(v) + change) for v in surface.vols],
        )
    if isinstance(surface, VannaVolgaVolSurface):
        from quantark.param.vol.vannavolga import SmileQuotes
        q = surface.quotes
        return surface.with_quotes(
            SmileQuotes(
                sigma_atm=max(_MIN_VOL, q.sigma_atm + change),
                rr25=q.rr25,
                bf25_2vol=q.bf25_2vol,
            )
        )
    raise ValidationError(
        f"Cannot bump FX vol surface type {type(surface).__name__} for VaR."
    )
```

Add the import at the top of `revaluation.py` (alongside the other param
imports):

```python
from quantark.param.vol.vannavolga import VannaVolgaVolSurface
```

Both bumps are "full-quote" (spec decision), just in different conventions:
- Engine vega (Task 8) is *multiplicative* (`factor`), under which RR/BF scale
  with the vol level, so all three quotes are scaled.
- VaR (here) is *additive* (`change` in vol points). Under a parallel additive
  shift, RR (= σ25c − σ25p) and BF (= ½(σ25c+σ25p) − σ_atm) are invariant, so
  only σ_atm moves. Shifting ATM with RR/BF fixed *is* the parallel full-quote
  move in additive terms. The two conventions are intentionally not unified.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -n0 test/test_vv_chain_integration.py::test_var_bump_env_handles_vv_vol_shift -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add quantark/var/fx/revaluation.py test/test_vv_chain_integration.py
git commit -m "feat(var): VV-surface vol shift in FX VaR revaluation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: End-to-end chain smoke (Greeks + VaR + backtest)

**Files:**
- Test: `test/test_vv_chain_integration.py` (append)

- [ ] **Step 1: Write the failing/again-green smoke test**

Append to `test/test_vv_chain_integration.py`:

```python
def test_full_greeks_dict_is_complete():
    eng = VannaVolgaBarrierEngine()
    greeks = eng.calculate_greeks(_barrier(), _env())
    for key in ("price", "delta", "gamma", "vega", "theta", "rho_dom", "rho_for"):
        assert key in greeks
        assert greeks[key] == greeks[key]  # not NaN


def test_one_touch_and_barrier_reprice_under_spot_bump():
    eng = VannaVolgaBarrierEngine()
    env = _env()
    base = eng.price(_barrier(), env)
    # Sticky-delta: bump spot via the standard engine path and confirm reprice.
    greeks = eng.calculate_greeks(_barrier(), env)
    assert greeks["price"] == pytest.approx(base, rel=1e-12)
    assert greeks["delta"] == greeks["delta"]  # finite
```

- [ ] **Step 2: Run the smoke tests**

Run: `python -m pytest -n0 test/test_vv_chain_integration.py -v`
Expected: PASS (all)

- [ ] **Step 3: Run the full VV + FX suite to confirm no regressions**

Run: `python -m pytest test/test_vanna_volga_smile.py test/test_vv_barrier.py test/test_reiner_rubinstein.py test/test_vv_vanilla_barrier.py test/test_vv_barrier_engine.py test/test_vv_chain_integration.py test/test_fx_one_touch.py test/test_fx_barrier_option.py -v`
Expected: PASS (QuantLib cross-check SKIPPED if QL absent)

- [ ] **Step 4: Run the entire test suite**

Run: `python -m pytest`
Expected: PASS — confirm the surface refactor and enum/export changes broke nothing across the repo.

- [ ] **Step 5: Remove the obsolete TODO and commit**

In `quantark/asset/fx/engine/analytical/vannavolga/vv_barrier.py`, delete the
now-resolved `# TODO(vanilla-KO)` block from the module docstring (lines 14-19),
since vanilla single-barrier KO/KI pricing now exists. Update the scope note to
point at double-barrier as the remaining deferred work.

```bash
git add quantark/asset/fx/engine/analytical/vannavolga/vv_barrier.py test/test_vv_chain_integration.py
git commit -m "test(fx): end-to-end VV chain smoke; drop resolved vanilla-KO TODO

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Done-When

- [ ] `VannaVolgaBarrierEngine` prices both `FxOneTouchOption` and `FxBarrierOption` through `BaseFxEngine.price` / `calculate_greeks`.
- [ ] In-out parity (`KI + KO = vanilla`) holds for all 8 RR types.
- [ ] VV correction reduces to the BS barrier when the smile is flat.
- [ ] Greek `vega` and FX VaR both revalue under a VV surface without raising.
- [ ] Full `python -m pytest` passes; existing VV smile/barrier tests stay green.
- [ ] The resolved `# TODO(vanilla-KO)` is removed; double-barrier remains the only documented deferral.
