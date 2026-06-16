# Smile-Consistent FX Vanillas & Digitals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make FX vanilla and European digital options fully smile-consistent (price + Greeks) under a `VannaVolgaVolSurface`: the digital price gains the skew term via static replication, and both products' Greeks route through bump-and-reprice (sticky-delta) instead of sticky-strike closed forms.

**Architecture:** Upgrade the two existing analytical engines in place. Each branches on `isinstance(fx_env.vol_surface, VannaVolgaVolSurface)`: VV surface → smile-consistent path; `FlatVolSurface`/`TermStructureVolSurface` → existing closed forms (unchanged). The digital VV price is the call-spread limit `-dC/dK` computed from two smile-consistent vanillas priced by `GarmanKohlhagenEngine`.

**Tech Stack:** Python, NumPy/SciPy, pytest. Reuses `GarmanKohlhagenEngine`, `FxVanillaOption`, `VannaVolgaVolSurface`, and the `BaseFxEngine` bump-and-reprice machinery (already VV-aware via `_vol_scaled_env`).

**Spec:** `docs/superpowers/specs/2026-06-16-smile-consistent-vanilla-digital-design.md`

**Conventions (CLAUDE.md):** `quantark.util.numerical` helpers over raw float ops; no silent fallbacks; canonical `quantark.*` imports; debug with `-n0`.

---

## File Structure

**Modified:**
- `quantark/asset/fx/engine/analytical/garman_kohlhagen_engine.py` — smile-aware `calculate_greeks` branch.
- `quantark/asset/fx/engine/analytical/fx_digital_engine.py` — replication price under VV surface + smile-aware `calculate_greeks` branch.

**New tests:**
- `test/test_fx_digital_smile.py`
- additions to `test/test_vv_chain_integration.py`

---

## Key facts (verified against current source)

- `GarmanKohlhagenEngine.price` (line 62) and `FxDigitalOptionAnalyticalEngine._d1_d2` (line 173) already read `sigma = fx_env.get_vol(strike, tau)` → vanilla price is already smile-consistent; digital uses smile *level* only.
- Both engines' `calculate_greeks` already delegate to `super().calculate_greeks()` in some sub-cases (GK line 104; digital line 85/91). The VV branch is an additional early delegation.
- `BaseFxEngine.calculate_greeks` bump-and-reprice already handles a `VannaVolgaVolSurface` (vega via `_vol_scaled_env` VV branch; spot/rate bumps re-anchor sticky-delta).
- Digital identities (per-unit, discounted, domestic value):
  - `cash_call = -dC/dK` (cash-or-nothing call paying 1 if S_T > K)
  - `cash_put = DF_dom(T_del) - cash_call`
  - `asset_call = C(K) + K*cash_call`
  - `asset_put = S_eff*DF_for(T_del) - asset_call`  (since `asset_call + asset_put = PV[S_T] = S_eff*DF_for`)

---

## Task 1: Smile-aware Greeks for `GarmanKohlhagenEngine`

**Files:**
- Modify: `quantark/asset/fx/engine/analytical/garman_kohlhagen_engine.py`
- Test: `test/test_vv_chain_integration.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `test/test_vv_chain_integration.py`:

```python
from quantark.asset.fx.product.option import FxVanillaOption
from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
from quantark.param import FlatVolSurface


def _vanilla():
    return FxVanillaOption(
        strike=1.20, option_type=OptionType.CALL, maturity=TAU, notional_foreign=1.0
    )


def test_vanilla_greeks_smile_consistent_match_bump_reprice():
    eng = GarmanKohlhagenEngine()
    env = _env()  # VannaVolgaVolSurface in the env
    greeks = eng.calculate_greeks(_vanilla(), env)
    # Under a VV surface the engine must use bump-and-reprice (sticky-delta),
    # i.e. delegate to BaseFxEngine. Verify by matching a manual central bump.
    from copy import deepcopy
    h = eng.params.spot_bump
    up, dn = deepcopy(env), deepcopy(env)
    up.spot_quote.spot = env.spot * (1 + h)
    dn.spot_quote.spot = env.spot * (1 - h)
    manual_delta = (eng.price(_vanilla(), up) - eng.price(_vanilla(), dn)) / (2 * env.spot * h)
    assert greeks["delta"] == pytest.approx(manual_delta, rel=1e-6)


def test_vanilla_greeks_flat_surface_unchanged():
    # Flat surface must still use the closed-form path (not bump-reprice).
    env = _env()
    env.vol_surface = FlatVolSurface(volatility=0.10)
    eng = GarmanKohlhagenEngine()
    greeks = eng.calculate_greeks(_vanilla(), env)
    assert greeks["vega"] == greeks["vega"]  # finite, closed-form path
    assert greeks["price"] > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -n0 test/test_vv_chain_integration.py::test_vanilla_greeks_smile_consistent_match_bump_reprice -v`
Expected: FAIL — the closed-form delta at σ(K) does not equal the bump-reprice delta (smile moves with spot).

- [ ] **Step 3: Add the VV branch**

In `garman_kohlhagen_engine.py`, add the import near the other param imports:

```python
from quantark.param.vol.vannavolga import VannaVolgaVolSurface
```

At the very start of `calculate_greeks` (right after `option = self._check_product(product)` on line 100), insert:

```python
        # Under a Vanna-Volga smile the closed-form Greeks (which hold sigma(K)
        # fixed) are not smile-consistent: they ignore how the smile re-anchors
        # with spot and shifts with the quotes. Route through bump-and-reprice,
        # which re-anchors the VV surface sticky-delta and shifts all quotes for
        # vega. Flat/term surfaces keep the exact closed forms below.
        if isinstance(fx_env.vol_surface, VannaVolgaVolSurface):
            return super().calculate_greeks(option, fx_env)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest -n0 test/test_vv_chain_integration.py -k "vanilla_greeks" -v`
Expected: PASS (both)

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/fx/engine/analytical/garman_kohlhagen_engine.py test/test_vv_chain_integration.py
git commit -m "feat(fx): smile-consistent vanilla Greeks under VV surface

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Replication price + smile-aware Greeks for `FxDigitalOptionAnalyticalEngine`

**Files:**
- Modify: `quantark/asset/fx/engine/analytical/fx_digital_engine.py`
- Test: `test/test_fx_digital_smile.py`

- [ ] **Step 1: Write the failing tests**

Create `test/test_fx_digital_smile.py`:

```python
from datetime import datetime

import pytest

from quantark.param import SpotQuote, FlatRateCurve, FlatVolSurface
from quantark.param.vol.vannavolga import FXEnv, SmileQuotes, VannaVolgaVolSurface, DeltaConvention
from quantark.priceenv import FxPricingEnvironment
from quantark.asset.fx.product.option import FxDigitalOption
from quantark.asset.fx.engine.analytical import FxDigitalOptionAnalyticalEngine
from quantark.util.enum import OptionType, FxPayoutCurrency

VAL = datetime(2026, 6, 15)
TAU = 0.75
SMILE = SmileQuotes(sigma_atm=0.10, rr25=-0.02, bf25_2vol=0.004)  # skewed
FLAT = SmileQuotes(sigma_atm=0.10, rr25=0.0, bf25_2vol=0.0)


def _env(quotes):
    surface = VannaVolgaVolSurface(
        FXEnv(spot=1.20, rd=0.02, rf=0.01, tau=TAU), quotes, DeltaConvention.SPOT
    )
    return FxPricingEnvironment(
        valuation_date=VAL,
        spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.02),
        foreign_curve=FlatRateCurve(rate=0.01),
        vol_surface=surface,
    )


def _flat_env():
    return FxPricingEnvironment(
        valuation_date=VAL,
        spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.02),
        foreign_curve=FlatRateCurve(rate=0.01),
        vol_surface=FlatVolSurface(volatility=0.10),
    )


def _digital(call=True, payout_ccy=FxPayoutCurrency.DOMESTIC):
    return FxDigitalOption(
        strike=1.25,
        option_type=OptionType.CALL if call else OptionType.PUT,
        payout=1.0,
        maturity=TAU,
        payout_currency=payout_ccy,
    )


def test_replication_reduces_to_closed_form_flat_vv_smile():
    # A flat VV smile (RR=BF=0) has no skew, so replication must equal the
    # closed-form N(d2) digital (priced here under an equivalent flat surface).
    eng = FxDigitalOptionAnalyticalEngine()
    vv_price = eng.price(_digital(), _env(FLAT))
    flat_price = eng.price(_digital(), _flat_env())
    assert vv_price == pytest.approx(flat_price, rel=1e-4)


def test_skew_moves_digital_in_correct_direction():
    # Closed-form (level-only) digital vs smile-consistent (replication) digital.
    eng = FxDigitalOptionAnalyticalEngine()
    smile_price = eng.price(_digital(), _env(SMILE))
    level_only = eng.price(_digital(), _flat_env())  # same ATM, no skew
    assert smile_price != pytest.approx(level_only, rel=1e-6)  # skew matters
    assert 0.0 < smile_price < 1.0  # bounded by undiscounted unit payout


def test_cash_call_put_parity_under_vv_surface():
    eng = FxDigitalOptionAnalyticalEngine()
    env = _env(SMILE)
    c = eng.price(_digital(call=True), env)
    p = eng.price(_digital(call=False), env)
    df_dom = env.get_domestic_df(TAU)
    assert c + p == pytest.approx(df_dom, rel=1e-6)


def test_asset_or_nothing_parity_under_vv_surface():
    eng = FxDigitalOptionAnalyticalEngine()
    env = _env(SMILE)
    c = eng.price(_digital(call=True, payout_ccy=FxPayoutCurrency.FOREIGN), env)
    p = eng.price(_digital(call=False, payout_ccy=FxPayoutCurrency.FOREIGN), env)
    s_eff = env.effective_spot()
    df_for = env.get_foreign_df(TAU)
    assert c + p == pytest.approx(s_eff * df_for, rel=1e-6)


def test_digital_greeks_smile_consistent_finite():
    eng = FxDigitalOptionAnalyticalEngine()
    greeks = eng.calculate_greeks(_digital(), _env(SMILE))
    for k in ("price", "delta", "gamma", "vega", "theta", "rho_dom", "rho_for"):
        assert k in greeks and greeks[k] == greeks[k]


def test_digital_flat_surface_price_unchanged():
    # Flat surface keeps the existing closed-form N(d2) path.
    eng = FxDigitalOptionAnalyticalEngine()
    env = _flat_env()
    import math
    from scipy.stats import norm
    # closed-form reference: payout * DF_dom * N(d2)
    tau = TAU
    fwd = env.get_forward(tau)
    sigma = 0.10
    d2 = (math.log(fwd / 1.25) - 0.5 * sigma * sigma * tau) / (sigma * math.sqrt(tau))
    ref = env.get_domestic_df(tau) * norm.cdf(d2)
    assert eng.price(_digital(), env) == pytest.approx(ref, rel=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest -n0 test/test_fx_digital_smile.py -v`
Expected: FAIL — `test_skew_moves_digital_in_correct_direction` and the parity tests fail because the current engine uses level-only `N(d2)` (no skew); flat-surface tests pass.

- [ ] **Step 3: Implement replication + greeks branch**

In `fx_digital_engine.py`, add imports near the existing ones:

```python
from quantark.asset.fx.engine.analytical.garman_kohlhagen_engine import GarmanKohlhagenEngine
from quantark.asset.fx.product.option.fx_vanilla_option import FxVanillaOption
from quantark.param.vol.vannavolga import VannaVolgaVolSurface
from quantark.util.enum import OptionType
```

Add replication bump constants and a vanilla engine just below the class docstring constants (top of class body, before `__init__`):

```python
    # Strike bump for the call-spread replication of the digital (-dC/dK).
    _H_REL = 1e-4
    _H_FLOOR = 1e-6
```

Update `__init__` to hold a reusable vanilla engine:

```python
    def __init__(self, params: Optional[FxEngineParams] = None):
        super().__init__(params)
        self._vanilla_engine = GarmanKohlhagenEngine()
```

Add a private replication helper:

```python
    def _vanilla_call_price(
        self, option: FxDigitalOption, fx_env: FxPricingEnvironment, strike: float
    ) -> float:
        """Smile-consistent unit-notional vanilla CALL price at `strike`."""
        vanilla = FxVanillaOption(
            strike=strike,
            option_type=OptionType.CALL,
            currency_pair=option.currency_pair,
            maturity=option.maturity,
            expiry_date=option.expiry_date,
            delivery=option.delivery,
            delivery_date=option.delivery_date,
            notional_foreign=1.0,
        )
        return self._vanilla_engine.price(vanilla, fx_env)

    def _replicated_digital(
        self, option: FxDigitalOption, fx_env: FxPricingEnvironment, tau: float
    ) -> float:
        """Smile-consistent digital via static replication (-dC/dK call spread).

        Captures the smile skew because the two vanilla legs are priced at
        sigma(K +/- h) through the surface. Derives all four digital variants
        from the cash-or-nothing call atom by no-arbitrage identities.
        """
        K = option.strike
        h = max(self._H_FLOOR, self._H_REL * K)
        if K - h <= 0.0:
            raise PricingError(
                f"Replication strike bump h={h} too large for strike {K}"
            )
        c_up = self._vanilla_call_price(option, fx_env, K + h)
        c_dn = self._vanilla_call_price(option, fx_env, K - h)
        cash_call = (c_dn - c_up) / (2.0 * h)  # = -dC/dK

        tau_delivery = option.get_delivery(fx_env)
        df_dom = fx_env.get_domestic_df(tau_delivery)

        if option.payout_currency == FxPayoutCurrency.DOMESTIC:
            base = cash_call if option.is_call() else (df_dom - cash_call)
        else:
            c_k = self._vanilla_call_price(option, fx_env, K)
            asset_call = c_k + K * cash_call
            if option.is_call():
                base = asset_call
            else:
                s_eff = fx_env.effective_spot()
                df_for = fx_env.get_foreign_df(tau_delivery)
                base = s_eff * df_for - asset_call

        return option.payout * base * option.participation_rate
```

In `price`, route the VV case through replication. Replace the body after the `is_zero(tau)` early-return (currently lines 59-72) so that the VV surface uses replication and everything else keeps the closed form:

```python
        if isinstance(fx_env.vol_surface, VannaVolgaVolSurface):
            return self._replicated_digital(option, fx_env, tau)

        tau_delivery = option.get_delivery(fx_env)
        d1, d2 = self._d1_d2(option, fx_env, tau)

        if option.payout_currency == FxPayoutCurrency.DOMESTIC:
            df_dom = fx_env.get_domestic_df(tau_delivery)
            prob = norm.cdf(d2) if option.is_call() else norm.cdf(-d2)
            value = option.payout * df_dom * prob
        else:
            df_dom = fx_env.get_domestic_df(tau_delivery)
            fwd = fx_env.get_forward(tau)
            prob = norm.cdf(d1) if option.is_call() else norm.cdf(-d1)
            value = option.payout * fwd * df_dom * prob

        return value * option.participation_rate
```

At the start of `calculate_greeks` (right after `option = self._check_product(product)` on line 84), insert the VV branch:

```python
        # Under a VV smile, route Greeks through bump-and-reprice (the price()
        # method then uses replication, capturing skew); the closed forms below
        # are sticky-strike at sigma(K) and miss the smile dynamics.
        if isinstance(fx_env.vol_surface, VannaVolgaVolSurface):
            return super().calculate_greeks(option, fx_env)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest -n0 test/test_fx_digital_smile.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/fx/engine/analytical/fx_digital_engine.py test/test_fx_digital_smile.py
git commit -m "feat(fx): smile-consistent digital via static replication under VV surface

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Chain smoke + full regression

**Files:**
- Test: `test/test_vv_chain_integration.py` (append)

- [ ] **Step 1: Write the chain smoke test**

Append to `test/test_vv_chain_integration.py`:

```python
def test_vanilla_and_digital_run_through_fx_var():
    from quantark.asset.fx.product.option import FxDigitalOption
    from quantark.asset.fx.engine.analytical import FxDigitalOptionAnalyticalEngine
    from quantark.var.fx.revaluation import bump_env
    env = _env()
    # vanilla + digital both price and bump cleanly under the VV surface
    gk = GarmanKohlhagenEngine()
    dig = FxDigitalOptionAnalyticalEngine()
    digital = FxDigitalOption(
        strike=1.25, option_type=OptionType.CALL, payout=1.0, maturity=TAU
    )
    bumped = bump_env(env, spot_return=0.01, vol_change=0.01)
    assert gk.price(_vanilla(), bumped) > 0.0
    assert dig.price(digital, bumped) >= 0.0
    assert all(v == v for v in dig.calculate_greeks(digital, env).values())
```

- [ ] **Step 2: Run the smoke test**

Run: `python -m pytest -n0 test/test_vv_chain_integration.py::test_vanilla_and_digital_run_through_fx_var -v`
Expected: PASS

- [ ] **Step 3: Run the FX + VV suites**

Run: `python -m pytest test/test_fx_digital_smile.py test/test_vv_chain_integration.py test/test_fx_vanilla_option.py test/test_fx_digital_option.py -q`
Expected: PASS (existing FX vanilla/digital suites unchanged because flat/term paths are untouched).

- [ ] **Step 4: Run the entire suite**

Run: `python -m pytest`
Expected: PASS (modulo the known-flaky `test_phoenix_engine_comparison` equity MC test; re-run it with `-n0` in isolation if it fails under xdist).

- [ ] **Step 5: Commit**

```bash
git add test/test_vv_chain_integration.py
git commit -m "test(fx): chain smoke for smile-consistent vanilla + digital

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Done-When

- [ ] Digital price under a VV surface uses replication and captures the skew (differs from level-only `N(d2)` in the correct direction).
- [ ] Cash/asset digital parities hold under the VV surface.
- [ ] Replication reduces to the closed-form digital when the smile is flat.
- [ ] Vanilla and digital Greeks under a VV surface come from bump-and-reprice (match a manual bump); flat/term-surface price and Greeks are unchanged.
- [ ] Full suite green (modulo the pre-existing flaky Phoenix MC test).
