# FX TARN / TARF Lifecycle (W4b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Execution method (already chosen by the user): subagent-driven**, in a NEW worktree branched from `main` AFTER the
> prerequisites below have landed. Do not ask for an execution method.

**Goal:** Give the FX TARN and TARF a mandatory realized state, a dated form, correct ageing (`time_shift` /
`record_passed_fixings` / theta), terminal bookings, and day-by-day ageing inside the FX backtest and FX dynamic
scenario, with a numeric contract showing no difference from its dated twin.

**Architecture:** A new frozen state module (`TargetRedemptionState`, `PendingPayment`, events, `TARGET_HIT_RTOL`) and
a shared base class `TargetRedemptionProduct` own the two schedule forms, validation, banking and the resolved view;
`FxTargetRedemptionNote` / `FxTargetRedemptionForward` only supply their terms and per-fixing outcome. The two MC
engines read only `product.resolve(fx_env)`. A new `quantark/asset/fx/lifecycle/` hook (`FxTargetRedemptionLifecycle`)
registers and advances TARN/TARF positions inside the two FX run engines; `BaseFxEngine._fdm_theta` steps numeric
products by the environment's year fraction (Q7).

**Tech Stack:** Python 3.10–3.13, NumPy, pandas (run engines only), pytest + xdist. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-09-fx-tarn-lifecycle-design.md` (rev 2 + Q7/Q8, user-approved 2026-10-09),
evidence in `docs/superpowers/specs/2026-10-09-fx-tarn-lifecycle-evidence/`. Its §1.2 decision table and §12
decisions are user rulings and bind every task. Executors read the spec section a task names before starting.

## Prerequisites (must be on `main` before the implementation branch is cut)

The implementation branch is cut from `main` **after W5 and W1 land**. Neither is on `main` at the time of writing
(`main` = `dcea77b6`).

| Branch | Head | What this plan imports (quoted exactly as it exists on the branch) |
|---|---|---|
| `w5-missing-fixings` | `bfb9b88f` | `quantark/util/fixings.py`: `class MissingFixingWarning(UserWarning)`; `def fixing_label(index: int, record) -> str` (returns `#<index> t=<time:.6g>` when `record.observation_time is not None`, else `#<index> <record.observation_date.date().isoformat()>`); `def warn_filled_fixings(product_name: str, field: str, labels: Sequence[str], *, stacklevel: int = 3) -> None` (one warning naming every label, nothing when empty). Precedent only: `FxRangeAccrualOption.resolve_observations(self, fx_env: "FxPricingEnvironment", *, warn_missing_fixings: bool = True) -> Tuple[List[Tuple[float, bool]], List[Tuple[float, float, int]], float]`. |
| `w1-tracker-clock` | `05ee1f9b` (the spec cites the fix commit `510bf6a7`; `05ee1f9b` is a docs commit on top) | `quantark/asset/equity/lifecycle/autocallable.py`: `class MaturedBeforeRunWarning(UserWarning)`. Also on the branch, **not used by this plan** (see Ruling R3): `def lifecycle_clock_origin(product: Any, start_date) -> Optional[pd.Timestamp]` (returns `product.initial_date` when set, else `start_date`). `MaturedBeforeRunWarning` is NOT re-exported from `quantark/asset/equity/lifecycle/__init__.py`; import it from the `autocallable` module. |
| `w2-rebate-flag` | in progress | Not imported. **Rebase risk only**: it may touch shared accrual code and the `CHANGELOG.md` `[Unreleased]` section. This plan touches no equity accrual code. |

Landing order on `main` is ff-only (memory: landing worktree branches); W5/W1 shas may change on rebase, so Task 1
verifies their **content**, not their shas.

## Global Constraints

- Canonical `quantark.*` imports only; never a flat import.
- Numerics via `quantark/util/numerical` (`is_zero`, `is_close`, `Tolerance`). The ONLY hardcoded tolerance is the
  spec ruling `TARGET_HIT_RTOL: Final[float] = 1e-12` (moved from the engines' `_HIT_RTOL`, value unchanged). Test
  parity bounds use `is_close(a, b, rel_tol=1e-12)` (spec §10).
- Errors are `ValidationError` (inputs), `PricingError` (pricing) from `quantark.util.exceptions`.
- Products stay engine-agnostic; parameter/state objects are immutable (frozen dataclasses; ageing replaces state).
- **Numeric == dated twin** (standing user rule 2026-10-09): identical events, ledger and PV; every parity test runs
  in both forms.
- **Never invent fallback semantics**: if something cannot be done correctly, write `# TODO(<why>)` and stop and
  report; a refusal (`ValidationError`) is allowed, a guessed value is not.
- No MC engine imported inside PDE code (none of these tasks touch PDE code; keep it that way).
- Test command for EVERY run (subagents at most `-n 2`; the controller runs the full suite at `-n 6`), with `<WT>`
  replaced by the implementation worktree's absolute path:
  `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest <paths> -n 2 -q`
  (a `.pth` imports the main checkout's `quantark` otherwise; scripts need the same `env PYTHONPATH=...`).
  A nonexistent test path voids the run ("no tests ran") — check the path exists.
- Existing expected values that move are listed old→new in
  `docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/moved_values.txt` for **user sign-off before
  landing**. Q7 is shared FX code: Task 8 measures every FX theta that moves under a non-calendar environment.
- The full suite rewrites two tracked sample files; restore them after every full run with
  `git checkout -- example/mo_volmodels/data/mo_barrier_sample.json example/mo_volmodels/data/mo_calibration_explainer_sample.html`.
  Never `git add example/` (add example files by explicit path only).
- `docs/` is git-excluded: commit plan evidence with `git add -f`.
- `CHANGELOG.md` `[Unreleased]` gets one block for this workstream (Task 2 creates it; later tasks append bullets).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Commit by pathspec. Never push,
  never merge (landing is the controller's job after sign-off).

## Spec drift found while writing this plan (code at `dcea77b6` vs the spec's citations)

- §2.2 M2 cites `fx_pricing_environment.py:187` for the DF error. Line 187 is the text of `get_forward`'s
  `MarketDataError`; the `ValidationError` M2 printed comes from `quantark/param/rrf/rate_curve.py:95` (FlatRateCurve
  DF) via `FxPricingEnvironment.get_domestic_df` (`fx_pricing_environment.py:138-140`). Evidence-only; no task depends on it.
- §1.3 cites W1 at `510bf6a7`; the branch head is `05ee1f9b` (docs on top). Same interface.
- §10 P1(b) says `accrual_start = 1/12 − τ` for both ages. That is right at τ = 0.125 (one fixing banked) but at
  τ = 91/365 two fixings are banked and the first listed period starts at the SECOND fixing, `2/12 − τ`. This plan
  uses `accrual_start = FIX[n_banked − 1] − τ` (Ruling R4). Fixed here, not in the spec.
- Every other cited file:line was checked and matches (`base_fx_engine.py:219-241`, `execution/inventory.py:438-439`,
  `backtest/fx/engine.py:31, 44-45, 57-58, 66, 99, 117, 138-157, 141`, `dynamicscenario/fx/engine.py:79-80, 84-89, 91,
  99-109, 112`, `day_path.py:298-310`, `dynamic_results.py:224-227`, `path_library.py:38, 59, 76, 90`,
  `float_roll.py:51`, `manager.py:168, 211-215`, `lifecycle_manager.py:11-13`, `test_fx_tarn_note_mc.py:68`,
  `example/fx_tarn_demo.py:74, 85, 97, 109`, `matrix_fixtures.py` `fx_tarf`/`fx_tarn` at 932/947).

## Rulings made in this plan (each: what — why — cost if wrong)

- **R1** `state` is keyword-only with a `None` sentinel that raises `ValidationError` — spec R2 requires a QuantArk
  exception for a missing state, which a no-default parameter (`TypeError`) cannot give — cost: none (still mandatory).
- **R2** `FxTargetRedemptionNote(fixing_times=None, coupon_rate=None, ...)`: both get `None` defaults and `coupon_rate is
  None` raises `ValidationError` — the dated form must be callable by keyword without a positional `fixing_times`, and
  Python forbids a required parameter after a defaulted one — cost: none (positional order unchanged).
- **R3** The FX hook's clock origin is the pair environment's `valuation_date` at registration (spec §7.2), NOT W1's
  `lifecycle_clock_origin` — W1's helper returns `product.initial_date` first, which on a dated TARN is the accrual
  start, not the run's clock — cost: none for numeric (no `initial_date`); dated products need no origin.
- **R4** P1(b) uses `accrual_start = FIX[n_banked − 1] − τ` (spec erratum above) — the spec's literal value gives a
  first factor of 2/12 at 91/365 — cost: none.
- **R5** Theta parity (D4) is asserted as `|θ_numeric − θ_dated| <= 1e-12 × |price|` — θ is a difference of two prices
  that each match to relative 1e-12 (M10: times differ by ULPs), so a bound relative to θ (≈1e-4 of the price) would
  demand 1e-16 price agreement — cost: the check is 1e-12-of-price, not 1e-12-of-theta.
- **R6** Dynamic-scenario `LifecycleEventSnapshot.cashflow` for a FIXING = quantity × the PAYMENT events of THAT
  fixing booked the same day (an immediate PAYMENT carries the FIXING's `label`) — spec §7.3 says "the quantity-scaled
  amount paid on the day"; with a pay lag that is 0.0 and the cash shows in `paid_cash` on the pay day — cost: one
  result field; easy to change.
- **R7** A dated fixing strictly after valuation that resolves to `is_zero` time (e.g. a Saturday fixing under a
  `BUSINESS_DAYS` env valued on the Friday) is refused at `resolve` with `ValidationError` — the engines need
  t > 0 and treating it as passed or as future would be invented semantics — cost: such a booking cannot be priced
  on that day until a rule is chosen.
- **R8** Dated `get_delivery` mirrors `get_maturity` (0.0 after `delivery_date`) — `BaseFxEngine.calculate_greeks`
  calls `get_delivery`, so Greeks on an aged dated contract past its last pay date would otherwise raise — cost: none.
- **R9** Q7 changes only `BaseFxEngine._fdm_theta` (spec §12). `localvol_greeks.py:60` and the Heston/SLV PDE thetas
  keep `days/365` — out of the spec's scope, listed as a follow-up in `moved_values.txt` — cost: those engines keep the
  M9 numeric/dated theta gap under non-calendar envs. A non-integral `theta_days` follows the dated branch's
  whole-day truncation (both branches then agree; recorded for sign-off) — cost: a fractional `theta_days` numeric
  theta becomes 0.0 under `CALENDAR_DAYS`, as the dated one already is.
- **R10** P2 adds τ = 0.1 — at τ ∈ {0.125, 91/365} the lag-0.02 coupon is already paid, so pending would be empty and
  P2 would test nothing — cost: none.
- **R11** `_numeric_twin` lives in `test/fx_target_redemption_fixtures.py` (shared by both new test modules) — spec
  §4.3 places it "in the new test module"; still a test helper, not public API — cost: none.
- **R12** Determined cash with `is_zero(amount)` is neither pending nor paid (spec "a non-zero c") — QuantArk numeric
  convention; both forms apply the same rule, so twin parity holds — cost: cash below 1e-10 currency units is dropped.

## Review Focus

1. **A short position (negative quantity)** in a run: paid cash and pending PV scale by the signed quantity —
   pinned by Task 9 `test_short_position_books_signed_cash`.
2. **A run calendar that skips a fixing date** (business-day backtest index, fixing on a Saturday): banked on the
   next path day at that day's spot, no warning, numeric twin banks the same day — pinned by Task 10
   `test_business_day_path_banks_weekend_fixing_next_day`.
3. **An environment with `spot_days > 0`**: banking uses the quoted `fx_env.spot`, not `effective_spot()` (Q3) —
   pinned by Task 5 `test_banking_uses_quoted_spot_not_effective_spot`.
4. **A dynamic scenario whose `DayPath.start_date` is before the environment's own date**: `ValidationError` (the
   clock cannot go back), never a silent re-age — pinned by Task 11 `test_start_date_before_environment_date_raises`.
5. **A dated future fixing that resolves to a zero year fraction** under `BUSINESS_DAYS`: `ValidationError` naming the
   date, never a t = 0 path — pinned by Task 7 `test_future_fixing_with_zero_year_fraction_is_refused`.

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `quantark/asset/fx/product/option/target_redemption_state.py` | Create (T2) | `TARGET_HIT_RTOL`, `target_hit_tol`, `is_target_hit`, `PendingPayment`, `TargetRedemptionState`, `TerminalBookingWarning`, `TargetRedemptionEventType`, `TargetRedemptionEvent`, `ResolvedTargetRedemption`, `pending_present_value` |
| `quantark/asset/fx/product/option/target_redemption_base.py` | Create (T3), extend (T5, T6, T7, T9) | `TargetRedemptionProduct(BaseFxProduct)`: forms, validation, terminal warning, `time_shift`, `record_passed_fixings`, `resolve`, `pending_pv`, `matured_before`, dated `get_maturity`/`get_delivery`; helpers `is_passed_time`, `event_label`, `signed_year_fraction` |
| `quantark/asset/fx/product/option/fx_target_redemption_note.py` | Modify (T3, T5, T7) | TARN terms, period sources, `coupon_amount`, `_fixing_outcome`, `_dated_factor` |
| `quantark/asset/fx/product/option/fx_target_redemption_forward.py` | Modify (T4, T5, T7) | TARF terms, `_fixing_outcome` |
| `quantark/asset/fx/product/option/__init__.py` | Modify (T2) | export the state names |
| `quantark/asset/fx/engine/mc/fx_tarn_note_mc_engine.py`, `fx_tarf_mc_engine.py` | Modify (T6) | read `resolve`; terminal; start `acc`; pending PV; product hit tolerance |
| `quantark/asset/fx/engine/base_fx_engine.py` | Modify (T8) | Q7 numeric theta step |
| `quantark/asset/fx/lifecycle/__init__.py`, `target_redemption.py` | Create (T9) | `FxTargetRedemptionLifecycle`, `FxLifecycleDay`, `FxLifecycleEntry`, `is_ageable` |
| `quantark/backtest/fx/engine.py` | Modify (T10) | register / advance / paid cash columns |
| `quantark/dynamicscenario/fx/engine.py`, `quantark/dynamicscenario/results/dynamic_results.py` | Modify (T11) | Q8 refusal, register / advance, cash fields, FIXING snapshots |
| `test/test_fx_target_redemption_state.py` | Create (T2) | state module |
| `test/fx_target_redemption_fixtures.py` | Create (T3), extend (T7, T9) | shared env/product builders, `_numeric_twin`, `drive` |
| `test/test_fx_target_redemption_lifecycle.py` | Create (T3), extend (T4–T9) | R1, R2, P1–P5, D1–D5, T1, T2, WARN1–3 |
| `test/test_fx_target_redemption_runs.py` | Create (T9), extend (T10, T11) | hook, B1–B6, Q8 |
| `test/test_fx_tarn_note_mc.py`, `test/test_fx_tarf_mc.py`, `test/execution/matrix_fixtures.py`, `example/fx_tarn_demo.py` | Modify (T3, T4) | constructor arguments only (C1) |
| `docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/` | Create (T1, T8, T12) | `w4b_baseline.py`, `q7_theta_plugin.py`, `q7_summarize.py`, `moved_values.txt` |

---

### Task 1: Prerequisite check and main baselines (no library code)

**Files:**
- Create: `docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/w4b_baseline.py`
- Output (untracked, never committed): `<WT>/.superpowers/w4b/baseline.json`, `<WT>/.superpowers/w4b/demo_<name>.main.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: `baseline.json` (B6 rows on main) and the demo stdouts, read by Task 12; script `w4b_baseline.py capture|compare <json>`.

- [ ] **Step 1: Verify the prerequisites are on this branch's base**

Run:
```bash
git -C <WT> log --oneline -1
env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -c "
from quantark.util.fixings import MissingFixingWarning, fixing_label, warn_filled_fixings
from quantark.asset.equity.lifecycle.autocallable import MaturedBeforeRunWarning
print('W5 and W1 present')"
```
Expected: `W5 and W1 present`. If either import fails, STOP: the branch was cut before W5/W1 landed.

- [ ] **Step 2: Write the baseline script**

```python
"""W4b B6 byte gate (plan 2026-10-09-fx-tarn-lifecycle, Tasks 1 and 12).

capture <json>: run on the implementation branch BEFORE any library change (== main) and store the FX backtest and FX
dynamic-scenario rows of a book holding every M7 FX product except TARN/TARF.
compare <json>: run on the finished branch; every stored number must match to the bit, and the new cash columns must
be exactly 0.0.

Run: env PYTHONPATH=<WT>:<WT>/test <venv>/bin/python <this file> capture|compare <WT>/.superpowers/w4b/baseline.json
"""
import json
import math
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from quantark.asset.fx.engine.analytical import (
    FxDigitalOptionAnalyticalEngine, FxForeignRangeAccrualAnalyticalEngine, FxQuantoDigitalAnalyticalEngine,
    FxQuantoRangeAccrualAnalyticalEngine, GarmanKohlhagenEngine, GarmanKohlhagenQuantoEngine,
)
from quantark.asset.fx.engine.analytical.fx_range_accrual_engine import FxRangeAccrualAnalyticalEngine
from quantark.asset.fx.engine.mc import FxBarrierMCEngine, FxSharkfinMCEngine
from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import (
    FxBarrierOption, FxDigitalOption, FxForeignRangeAccrualOption, FxQuantoDigitalOption, FxQuantoRangeAccrualOption,
    FxQuantoVanillaOption, FxRangeAccrualConfig, FxRangeAccrualOption, FxSharkfinOption, FxVanillaOption,
)
from quantark.backtest.fx import FXBacktestConfig, FXBacktestEngine
from quantark.backtest.strategy.fx_delta_neutral_strategy import FXDeltaNeutralStrategy
from quantark.dynamicscenario import FXDynamicScenarioConfig, FXDynamicScenarioEngine
from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.fx import FXPortfolio
from quantark.priceenv import FxPricingEnvironment
from quantark.priceenv.fx_pricing_environment import FxQuantoMarketData, QuantoConversionOrientation
from quantark.stresstest.stress.stress_types import StressType
from quantark.util.enum import FxBarrierType, OptionType

warnings.simplefilter("ignore")
PAIR = CurrencyPair("EUR", "USD")
V = datetime(2026, 6, 15)
FIX = [k / 12 for k in range(1, 13)]
NEW_BACKTEST_KEYS = {"paid_cash", "pending_pv"}


def env():
    return FxPricingEnvironment(
        valuation_date=V, spot_quote=SpotQuote(spot=1.20), domestic_curve=FlatRateCurve(rate=0.05),
        foreign_curve=FlatRateCurve(rate=0.03), vol_surface=FlatVolSurface(volatility=0.10),
        quanto=FxQuantoMarketData(settlement_curve=FlatRateCurve(rate=0.02), quanto_vol=0.08, correlation=0.3,
                                  conversion_orientation=QuantoConversionOrientation.SETTLEMENT_PER_DOMESTIC))


def book():
    mc = FxMCParams(num_paths=5_000, seed=7, time_steps=24)
    cfg = FxRangeAccrualConfig(upper_barrier=1.30, lower_barrier=1.10, accrual_rate=0.04)
    rows = [
        (FxVanillaOption(strike=1.20, option_type=OptionType.CALL, maturity=1.0, notional_foreign=1.0,
                         currency_pair=PAIR), GarmanKohlhagenEngine()),
        (FxVanillaOption(strike=1.20, option_type=OptionType.CALL, expiry_date=V + timedelta(days=365),
                         notional_foreign=1.0, currency_pair=PAIR), GarmanKohlhagenEngine()),
        (FxDigitalOption(strike=1.20, option_type=OptionType.CALL, payout=1.0, maturity=1.0, currency_pair=PAIR),
         FxDigitalOptionAnalyticalEngine()),
        (FxQuantoVanillaOption(strike=1.20, option_type=OptionType.CALL, quanto_fx_rate=1.0, maturity=1.0,
                               currency_pair=PAIR, notional_foreign=1.0), GarmanKohlhagenQuantoEngine()),
        (FxBarrierOption(strike=1.20, barrier=1.35, is_up=True, knock_type=FxBarrierType.KNOCK_OUT,
                         option_type=OptionType.CALL, maturity=1.0, currency_pair=PAIR), FxBarrierMCEngine(mc)),
        (FxSharkfinOption(strike=1.20, barrier=1.35, is_up=True, option_type=OptionType.CALL, maturity=1.0,
                          currency_pair=PAIR), FxSharkfinMCEngine(mc)),
        (FxRangeAccrualOption(notional=1.0, range_config=cfg, maturity=1.0, observation_times=FIX,
                              currency_pair=PAIR), FxRangeAccrualAnalyticalEngine()),
        (FxQuantoDigitalOption(strike=1.20, option_type=OptionType.CALL, payout=1.0, quanto_fx_rate=1.0,
                               maturity=1.0, currency_pair=PAIR), FxQuantoDigitalAnalyticalEngine()),
        (FxQuantoRangeAccrualOption(notional=1.0, range_config=cfg, quanto_fx_rate=1.0, maturity=1.0,
                                    observation_times=FIX, currency_pair=PAIR),
         FxQuantoRangeAccrualAnalyticalEngine()),
        (FxForeignRangeAccrualOption(notional=1.0, range_config=cfg, maturity=1.0, observation_times=FIX,
                                     currency_pair=PAIR), FxForeignRangeAccrualAnalyticalEngine()),
    ]
    pf = FXPortfolio(portfolio_name="b6", pricing_environments={"EURUSD": env()},
                     creation_date=datetime(2026, 1, 1))
    for prod, eng in rows:
        pf.add_position(product=prod, quantity=1.0, entry_price=0.0, underlying="EURUSD", engine=eng,
                        entry_timestamp=datetime(2026, 1, 1))
    return pf


def _plain(v):
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    if isinstance(v, (np.floating, np.bool_)):
        return v.item()
    return v


def backtest(days, greeks):
    rng = np.random.default_rng(3)
    idx = pd.date_range(V, periods=days, freq="D")
    spot = 1.20 * np.exp(np.cumsum(rng.normal(0.0, 0.005, days)))
    path = pd.DataFrame({"EURUSD_spot": spot, "EURUSD_vol": 0.10, "EURUSD_dom_rate": 0.05,
                         "EURUSD_for_rate": 0.03}, index=idx)
    res = FXBacktestEngine(FXBacktestConfig(portfolio=book(), market_path=path, strategy=FXDeltaNeutralStrategy(),
                                            calculate_greeks=greeks)).run()
    return {"rows": [{k: _plain(v) for k, v in r.items()} for r in res.rows],
            "initial_value": res.initial_value, "final_value": res.final_value, "num_hedges": res.num_hedges,
            "total_hedge_pnl": res.total_hedge_pnl}


def dynamic(start):
    steps = [DayStep(day_index=i, changes=[ParameterChange("spot", StressType.PERCENTAGE, 0.001)])
             for i in range(30)]
    r = FXDynamicScenarioEngine(FXDynamicScenarioConfig(calculate_greeks=True)).run(
        book(), DayPath(name="b6", steps=steps, start_date=start))
    return {"baseline": r.baseline_value, "final": r.final_value, "total_pnl": r.total_pnl,
            "days": [{"portfolio_value": d.portfolio_value, "daily_pnl": d.daily_pnl,
                      "cumulative_pnl": d.cumulative_pnl, "net_pnl": d.net_pnl,
                      "greeks": {k: _plain(v) for k, v in d.greeks.items()},
                      "paid_cash": d.paid_cash, "realized_cash": d.realized_cash,
                      "pending_receivable_pv": d.pending_receivable_pv,
                      "n_lifecycle_events": len(d.lifecycle_events)} for d in r.day_results]}


CASES = {
    "backtest_92_no_greeks": lambda: backtest(92, False),
    "backtest_20_greeks": lambda: backtest(20, True),
    "dynamic_dated": lambda: dynamic(V),
    "dynamic_undated": lambda: dynamic(None),
}


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _diff(path, base, got, out):
    if isinstance(base, dict):
        for k, v in base.items():
            if k not in got:
                out.append(f"{path}.{k}: missing")
            else:
                _diff(f"{path}.{k}", v, got[k], out)
        for k in set(got) - set(base):
            if k in NEW_BACKTEST_KEYS and got[k] == 0.0:
                continue
            out.append(f"{path}.{k}: new key with value {got[k]!r}")
    elif isinstance(base, list):
        if len(base) != len(got):
            out.append(f"{path}: length {len(base)} -> {len(got)}")
        for i, (x, y) in enumerate(zip(base, got)):
            _diff(f"{path}[{i}]", x, y, out)
    elif not _same(base, got):
        out.append(f"{path}: {base!r} -> {got!r}")


def main():
    mode, target = sys.argv[1], sys.argv[2]
    got = json.loads(json.dumps({name: fn() for name, fn in CASES.items()}))
    if mode == "capture":
        with open(target, "w") as fh:
            json.dump(got, fh, indent=1)
        print(f"captured {len(got)} cases to {target}")
        return
    with open(target) as fh:
        base = json.load(fh)
    diffs = []
    _diff("", base, got, diffs)
    if diffs:
        print("B6 FAIL\n" + "\n".join(diffs[:200]))
        sys.exit(1)
    print("B6 OK: every stored number bitwise equal; new cash columns 0.0")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Capture the baselines on the untouched branch**

Run:
```bash
mkdir -p <WT>/.superpowers/w4b
env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python <WT>/docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/w4b_baseline.py capture <WT>/.superpowers/w4b/baseline.json
for f in fx_tarn_demo fx_backtest_demo fx_dynamic_scenario_demo fx_portfolio_risk_demo; do
  env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python <WT>/example/$f.py > <WT>/.superpowers/w4b/demo_$f.main.txt 2>&1; echo "$f exit $?"; done
env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python <WT>/docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/w4b_baseline.py compare <WT>/.superpowers/w4b/baseline.json
```
Expected: `captured 4 cases ...`, four `exit 0`, then `B6 OK ...` (the script is self-consistent before any change; a
FAIL here means a nondeterministic case — fix the script, not the library).

- [ ] **Step 4: Confirm the T1 reference values on this base**

Run (values were measured on `dcea77b6`; W5/W1 do not touch these engines, so they must be identical):
```bash
env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python - <<'EOF'
from datetime import datetime
from quantark.asset.fx.engine.mc import FxTargetRedemptionNoteMCEngine, FxTarnForwardMCEngine
from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxTargetRedemptionForward, FxTargetRedemptionNote
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment
P = CurrencyPair("EUR", "USD"); FIX = [2 / 365] + [k / 12 for k in range(2, 13)]
e = FxPricingEnvironment(valuation_date=datetime(2026, 6, 15), spot_quote=SpotQuote(spot=1.20), domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.03), vol_surface=FlatVolSurface(volatility=0.10))
gn = FxTargetRedemptionNoteMCEngine(FxMCParams(num_paths=20_000, seed=7)).calculate_greeks(FxTargetRedemptionNote(fixing_times=FIX, coupon_rate=0.12, notional=100.0, strike=1.10, target=3.0, currency_pair=P), e)
gf = FxTarnForwardMCEngine(FxMCParams(num_paths=20_000, seed=7)).calculate_greeks(FxTargetRedemptionForward(strike=1.15, fixing_times=FIX, target=0.05, currency_pair=P), e)
print(repr(gn["price"]), repr(gn["theta"]), repr(gf["price"]), repr(gf["theta"]))
EOF
```
Expected exactly: `101.63518163030555 0.01686139016959487 0.01925287164907336 0.003076911159477322`. If not, STOP and
report (Task 6's T1 test hard-codes these).

- [ ] **Step 5: Commit the script**

```bash
git -C <WT> add -f docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/w4b_baseline.py
git -C <WT> commit -m "test(fx-tarn): B6 byte-gate baseline script

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `TargetRedemptionState` module

Spec §3.

**Files:**
- Create: `quantark/asset/fx/product/option/target_redemption_state.py`
- Modify: `quantark/asset/fx/product/option/__init__.py`, `CHANGELOG.md`
- Test: `test/test_fx_target_redemption_state.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (exact):
  - `TARGET_HIT_RTOL: Final[float] = 1e-12`
  - `target_hit_tol(target: float) -> float` (0.0 for `inf`, else `abs(target) * TARGET_HIT_RTOL`)
  - `is_target_hit(accumulated: float, target: float) -> bool` (`accumulated >= target - target_hit_tol(target)`)
  - `PendingPayment(amount: float, pay_time: Optional[float] = None, pay_date: Optional[datetime] = None)` frozen
  - `TargetRedemptionState(accumulated: float, pending: Tuple[PendingPayment, ...] = (), as_of: Optional[datetime] = None)` frozen
  - `TerminalBookingWarning(UserWarning)`
  - `TargetRedemptionEventType` enum: `FIXING = "fixing"`, `PAYMENT = "payment"`
  - `TargetRedemptionEvent(event_type, amount, fixing=None, time=0.0, redeemed=False, label="")` frozen
  - `ResolvedTargetRedemption(fixing_times: Tuple[float, ...], pay_times: Tuple[float, ...], accrual_factors: Optional[Tuple[float, ...]], accumulated: float, pending: Tuple[Tuple[float, float], ...], redeemed: bool, terminal: bool)` frozen; `pending` holds `(pay_time, amount)`
  - `pending_present_value(pending: Sequence[Tuple[float, float]], fx_env) -> float` (`pv = 0.0; pv += amount * fx_env.get_domestic_df(t)` in order)

- [ ] **Step 1: Write the failing tests**

```python
"""TargetRedemptionState (spec 2026-10-09-fx-tarn-lifecycle-design.md section 3)."""
import math
from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from quantark.asset.fx.product.option.target_redemption_state import (
    TARGET_HIT_RTOL,
    PendingPayment,
    ResolvedTargetRedemption,
    TargetRedemptionEvent,
    TargetRedemptionEventType,
    TargetRedemptionState,
    TerminalBookingWarning,
    is_target_hit,
    pending_present_value,
    target_hit_tol,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment
from quantark.util.exceptions import ValidationError

D1, D2, D3 = datetime(2026, 7, 15), datetime(2026, 8, 15), datetime(2026, 9, 15)


def test_hit_rtol_keeps_the_engines_value():
    assert TARGET_HIT_RTOL == 1e-12


def test_hit_tolerance_is_the_engine_rule():
    assert target_hit_tol(math.inf) == 0.0
    assert target_hit_tol(3.0) == abs(3.0) * 1e-12
    assert is_target_hit(3.0 - 1e-13, 3.0)
    assert not is_target_hit(3.0 - 1e-11, 3.0)
    assert not is_target_hit(1e300, math.inf)


def test_inception_state():
    s = TargetRedemptionState(0.0)
    assert s.accumulated == 0.0 and s.pending == () and s.as_of is None


def test_pending_list_is_stored_as_a_tuple():
    s = TargetRedemptionState(1.0, pending=[PendingPayment(1.0, pay_time=0.1)])
    assert s.pending == (PendingPayment(1.0, pay_time=0.1),)


def test_state_is_immutable():
    s = TargetRedemptionState(0.0)
    with pytest.raises(FrozenInstanceError):
        s.accumulated = 1.0


@pytest.mark.parametrize("acc", [-1e-9, math.nan, math.inf])
def test_rejects_bad_accumulated(acc):
    with pytest.raises(ValidationError, match="accumulated"):
        TargetRedemptionState(acc)


@pytest.mark.parametrize("kw", [{}, {"pay_time": 0.1, "pay_date": D1}])
def test_pending_needs_exactly_one_of_time_or_date(kw):
    with pytest.raises(ValidationError, match="exactly one"):
        PendingPayment(1.0, **kw)


@pytest.mark.parametrize("t", [0.0, -0.1, 1e-12, math.nan, math.inf])
def test_numeric_pay_time_must_be_after_valuation(t):
    with pytest.raises(ValidationError, match="pay_time"):
        PendingPayment(1.0, pay_time=t)


@pytest.mark.parametrize("amount", [math.nan, math.inf])
def test_pending_amount_must_be_finite(amount):
    with pytest.raises(ValidationError, match="finite"):
        PendingPayment(amount, pay_time=0.1)


def test_rejects_mixed_pending_forms():
    with pytest.raises(ValidationError, match="mixes"):
        TargetRedemptionState(1.0, pending=(PendingPayment(1.0, pay_time=0.1), PendingPayment(1.0, pay_date=D3)),
                              as_of=D1)


def test_rejects_unsorted_pending():
    with pytest.raises(ValidationError, match="sorted"):
        TargetRedemptionState(1.0, pending=(PendingPayment(1.0, pay_time=0.2), PendingPayment(1.0, pay_time=0.1)))


def test_equal_pay_times_are_sorted():
    TargetRedemptionState(1.0, pending=(PendingPayment(0.5, pay_time=0.1), PendingPayment(100.0, pay_time=0.1)))


def test_dated_pending_needs_as_of():
    with pytest.raises(ValidationError, match="as_of"):
        TargetRedemptionState(1.0, pending=(PendingPayment(1.0, pay_date=D2),))


@pytest.mark.parametrize("pay", [D1, datetime(2026, 7, 1)])
def test_dated_pending_must_be_after_as_of(pay):
    with pytest.raises(ValidationError, match="after as_of"):
        TargetRedemptionState(1.0, pending=(PendingPayment(1.0, pay_date=pay),), as_of=D1)


def test_terminal_booking_warning_is_a_user_warning():
    assert issubclass(TerminalBookingWarning, UserWarning)


def test_event_defaults():
    e = TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, 1.0)
    assert (e.fixing, e.time, e.redeemed, e.label) == (None, 0.0, False, "")
    assert TargetRedemptionEventType.FIXING.value == "fixing"


def test_resolved_view_is_frozen():
    v = ResolvedTargetRedemption((0.5,), (0.5,), None, 0.0, (), False, False)
    with pytest.raises(FrozenInstanceError):
        v.accumulated = 1.0


def test_pending_present_value_sums_in_order():
    e = FxPricingEnvironment(valuation_date=D1, spot_quote=SpotQuote(spot=1.2),
                             domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.03),
                             vol_surface=FlatVolSurface(volatility=0.1))
    expected = 0.0 + 0.5 * e.get_domestic_df(0.1) + 100.0 * e.get_domestic_df(0.1)
    assert pending_present_value(((0.1, 0.5), (0.1, 100.0)), e) == expected
    assert pending_present_value((), e) == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_state.py -n 2 -q`
Expected: collection error `ModuleNotFoundError: No module named 'quantark.asset.fx.product.option.target_redemption_state'`.

- [ ] **Step 3: Implement the module**

```python
"""Realized state of an FX target-redemption contract (TARN / TARF).

Spec: docs/superpowers/specs/2026-10-09-fx-tarn-lifecycle-design.md, section 3. ``accumulated`` is the undiscounted
amount banked towards the target (TARN coupons, TARF favourable amounts, including those fixed but not yet paid).
``pending`` is determined cash whose pay time is after valuation. The state is immutable: ageing replaces it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Final, Optional, Sequence, Tuple

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_zero

#: Relative target-hit tolerance, moved from the two MC engines' ``_HIT_RTOL`` with its value unchanged (spec section
#: 3: ``Tolerance.RELATIVE`` would move redemption decisions and inception prices).
TARGET_HIT_RTOL: Final[float] = 1e-12


def target_hit_tol(target: float) -> float:
    """Target-relative hit tolerance; 0 for an unbounded target (the engines' rule, unchanged)."""
    return 0.0 if math.isinf(target) else abs(target) * TARGET_HIT_RTOL


def is_target_hit(accumulated: float, target: float) -> bool:
    """True once ``accumulated`` reaches ``target`` within :func:`target_hit_tol`."""
    return accumulated >= target - target_hit_tol(target)


@dataclass(frozen=True)
class PendingPayment:
    """Determined, unpaid domestic cash: ``pay_time`` (numeric form, years from valuation) or ``pay_date`` (dated)."""

    amount: float
    pay_time: Optional[float] = None
    pay_date: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.amount):
            raise ValidationError(f"pending amount must be finite, got {self.amount}")
        if (self.pay_time is None) == (self.pay_date is None):
            raise ValidationError(
                "a pending payment takes exactly one of pay_time (numeric form) or pay_date (dated form)"
            )
        if self.pay_time is not None:
            t = self.pay_time
            if not math.isfinite(t) or t <= 0.0 or is_zero(t):
                raise ValidationError(
                    f"a numeric pending pay_time must be after valuation (> 0), got {t}: cash paid at or before "
                    "valuation is booked, not pending"
                )


@dataclass(frozen=True)
class TargetRedemptionState:
    """What a target-redemption contract has banked (``accumulated``) and still has to pay (``pending``).

    ``as_of`` is used by the dated form only: every fixing dated on or before it is banked.
    """

    accumulated: float
    pending: Tuple[PendingPayment, ...] = ()
    as_of: Optional[datetime] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pending", tuple(self.pending))
        acc = self.accumulated
        if not math.isfinite(acc) or acc < 0.0:
            raise ValidationError(f"state.accumulated must be finite and >= 0, got {acc}")
        if any(not isinstance(p, PendingPayment) for p in self.pending):
            raise ValidationError("state.pending holds PendingPayment entries")
        numeric = [p.pay_time is not None for p in self.pending]
        if any(numeric) and not all(numeric):
            raise ValidationError("state.pending mixes pay_time and pay_date entries: one form per contract")
        keys = [p.pay_time if p.pay_time is not None else p.pay_date for p in self.pending]
        if keys != sorted(keys):
            raise ValidationError("state.pending must be sorted by ascending pay time / date")
        if self.pending and not any(numeric):
            if self.as_of is None:
                raise ValidationError(
                    "dated pending cash with as_of=None is refused: cash implies a banked fixing"
                )
            late = [p.pay_date for p in self.pending if p.pay_date <= self.as_of]
            if late:
                raise ValidationError(f"a dated pending pay_date must be after as_of {self.as_of}, got {late[0]}")


class TerminalBookingWarning(UserWarning):
    """A contract was BOOKED terminal (redeemed, or every fixing banked): it is valued at the PV of its pending cash,
    0.0 when none (spec section 5.3, user decision Q6). Raised once, by the constructor."""


class TargetRedemptionEventType(Enum):
    FIXING = "fixing"  # a fixing was banked
    PAYMENT = "payment"  # determined cash reached its pay time / date


@dataclass(frozen=True)
class TargetRedemptionEvent:
    """One lifecycle event, appended to a product's ``lifecycle_events`` by ``time_shift`` / ``record_passed_fixings``.

    ``amount``: FIXING - the cash it determined (coupon / net cashflow, principal excluded); PAYMENT - the cash paid.
    ``time``: FIXING - the fixing's signed time at banking (<= 0); PAYMENT - the pay time (<= 0).
    ``redeemed``: FIXING - this fixing ended the contract (target hit, or the last fixing).
    ``label``: W5 ``fixing_label`` text; an immediate PAYMENT carries its FIXING's label (informational).
    """

    event_type: TargetRedemptionEventType
    amount: float
    fixing: Optional[float] = None
    time: float = 0.0
    redeemed: bool = False
    label: str = ""


@dataclass(frozen=True)
class ResolvedTargetRedemption:
    """The one engine input (spec section 5.1): unbanked fixings still ahead (all > 0), their factors (TARN), the
    banked amount, pending ``(pay_time, amount)`` after valuation, and the derived flags."""

    fixing_times: Tuple[float, ...]
    pay_times: Tuple[float, ...]
    accrual_factors: Optional[Tuple[float, ...]]
    accumulated: float
    pending: Tuple[Tuple[float, float], ...]
    redeemed: bool
    terminal: bool


def pending_present_value(pending: Sequence[Tuple[float, float]], fx_env) -> float:
    """``sum(amount * DF_dom(pay_time))`` over ``(pay_time, amount)`` pairs, in order."""
    pv = 0.0
    for pay_time, amount in pending:
        pv += amount * fx_env.get_domestic_df(pay_time)
    return pv
```

Append to `quantark/asset/fx/product/option/__init__.py` (imports after the TARN import, names in `__all__`):

```python
from .target_redemption_state import (
    PendingPayment,
    TargetRedemptionEvent,
    TargetRedemptionEventType,
    TargetRedemptionState,
    TerminalBookingWarning,
)
```
and add `'PendingPayment', 'TargetRedemptionEvent', 'TargetRedemptionEventType', 'TargetRedemptionState',
'TerminalBookingWarning'` to `__all__`.

Insert at the top of `CHANGELOG.md` `## [Unreleased]` (right after that heading and its blank line):

```markdown
FX TARN / TARF mid-life booking, dated form and ageing (spec:
`docs/superpowers/specs/2026-10-09-fx-tarn-lifecycle-design.md`). A target-redemption contract carries its realized
state, can be booked by dates, ages through `time_shift` / `record_passed_fixings`, and is aged day by day in the FX
backtest and FX dynamic scenario.

### Added
- `quantark.asset.fx.product.option.target_redemption_state`: `TargetRedemptionState` (amount banked towards the
  target, determined `PendingPayment`s, dated `as_of`), `TargetRedemptionEvent` / `TargetRedemptionEventType`,
  `TerminalBookingWarning`, `ResolvedTargetRedemption`, and `TARGET_HIT_RTOL` (1e-12, moved from the two TARN/TARF
  engines with its value unchanged).

```

- [ ] **Step 4: Run to verify it passes**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_state.py -n 2 -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/asset/fx/product/option/target_redemption_state.py quantark/asset/fx/product/option/__init__.py test/test_fx_target_redemption_state.py CHANGELOG.md
git -C <WT> commit -m "feat(fx-tarn): TargetRedemptionState - realized state, pending cash, lifecycle events

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Shared base and numeric TARN construction (mandatory state, period start, terminal booking)

Spec §3, §4.1, §5.3, §5.4, §8 (construction sites), §10 R1/R2/WARN1 (construction part).

**Files:**
- Create: `quantark/asset/fx/product/option/target_redemption_base.py`, `test/fx_target_redemption_fixtures.py`, `test/test_fx_target_redemption_lifecycle.py`
- Modify: `quantark/asset/fx/product/option/fx_target_redemption_note.py` (whole `__init__`/validation; keep `coupon_condition`, `is_band`, `__repr__`), `test/test_fx_tarn_note_mc.py` (`_note` only), `test/execution/matrix_fixtures.py:947-958`, `example/fx_tarn_demo.py:97-99, 109-111`, `CHANGELOG.md`

**Interfaces:**
- Consumes: Task 2 names.
- Produces:
  - `quantark.asset.fx.product.option.target_redemption_base`: `is_passed_time(t: float) -> bool`, `event_label(index: int, *, time: Optional[float] = None, date: Optional[datetime] = None) -> str`, class `TargetRedemptionProduct(BaseFxProduct)` with properties `is_numeric`, `hit_tol`, `is_unbounded_target`, `is_redeemed`, `is_terminal`; attributes `fixing_times`, `pay_times`, `fixing_dates` (None), `pay_dates` (None), `state`, `lifecycle_events: Tuple[TargetRedemptionEvent, ...]`; subclass hooks `_validate_terms()`, `_fixing_outcome(factor, spot, accumulated) -> (banked, cash)`, `_principal_amount() -> float`, class attrs `_has_accrual`, `_pending_must_be_positive`.
  - `FxTargetRedemptionNote(fixing_times=None, coupon_rate=None, notional=1.0, target=math.inf, strike=None, is_above=True, lower=None, upper=None, accrual_factors=None, pay_times=None, include_principal=True, currency_pair=None, *, state=None, accrual_start=None)`; methods `coupon_amount(factor) -> float` (`self.notional * self.coupon_rate * factor`), `full_coupon(period) -> float`.
  - Test fixtures `env`, `mc`, `note`, `engine_for`, `PAIR`, `VAL`, `FIX`, `INCEPTION`.

- [ ] **Step 1: Write the fixtures module**

`test/fx_target_redemption_fixtures.py`:

```python
"""Shared builders for the FX TARN / TARF lifecycle tests (spec 2026-10-09-fx-tarn-lifecycle-design.md, section 2.2
fixture: twelve monthly fixings at k/12, TARN notional 100 / coupon 12% / strike 1.10, TARF strike 1.15 / notional 1,
EURUSD 1.20, r_d 5%, r_f 3%, vol 10%)."""
from datetime import datetime

from quantark.asset.fx.engine.mc import FxTargetRedemptionNoteMCEngine, FxTarnForwardMCEngine
from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxTargetRedemptionForward, FxTargetRedemptionNote
from quantark.asset.fx.product.option.target_redemption_state import TargetRedemptionState
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment
from quantark.util.calendar import DayCountConvention

PAIR = CurrencyPair("EUR", "USD")
VAL = datetime(2026, 6, 15)
FIX = [k / 12 for k in range(1, 13)]
INCEPTION = TargetRedemptionState(0.0)


def env(spot=1.20, val=VAL, vol=0.10, dc=DayCountConvention.CALENDAR_DAYS, calendar=None, spot_days=0):
    return FxPricingEnvironment(
        valuation_date=val, spot_quote=SpotQuote(spot=spot), domestic_curve=FlatRateCurve(rate=0.05),
        foreign_curve=FlatRateCurve(rate=0.03), vol_surface=FlatVolSurface(volatility=vol),
        day_count_convention=dc, calendar=calendar, spot_days=spot_days,
    )


def mc(n=20_000, seed=7, method=None):
    kw = dict(num_paths=n, seed=seed)
    if method is not None:
        kw["method"] = method
    return FxMCParams(**kw)


def note(fix=FIX, state=INCEPTION, **kw):
    """Numeric TARN; the first listed period starts at 0.0 unless the caller states accrual_factors or accrual_start."""
    base = dict(fixing_times=fix, coupon_rate=0.12, notional=100.0, strike=1.10, currency_pair=PAIR, state=state)
    if "accrual_factors" not in kw:
        base["accrual_start"] = 0.0
    base.update(kw)
    return FxTargetRedemptionNote(**base)


def engine_for(product, n=20_000, seed=7, method=None):
    params = mc(n=n, seed=seed, method=method)
    if isinstance(product, FxTargetRedemptionNote):
        return FxTargetRedemptionNoteMCEngine(params)
    return FxTarnForwardMCEngine(params)
```

- [ ] **Step 2: Write the failing tests**

`test/test_fx_target_redemption_lifecycle.py`:

```python
"""FX TARN / TARF lifecycle: construction, ageing, pricing, dated twins (spec 2026-10-09-fx-tarn-lifecycle-design.md,
section 10)."""
import copy
import math
import warnings

import pytest

from fx_target_redemption_fixtures import FIX, INCEPTION, PAIR, env, note
from quantark.asset.fx.product.option import FxTargetRedemptionNote
from quantark.asset.fx.product.option.target_redemption_state import (
    PendingPayment,
    TargetRedemptionState,
    TerminalBookingWarning,
)
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close


# --- R1 (findings section 10): the period start is a contract term -------------------------------------------------

def test_r1_rebuild_without_a_period_source_is_refused():
    remaining = [t - 0.125 for t in FIX[1:]]
    with pytest.raises(ValidationError, match="accrual_start"):
        FxTargetRedemptionNote(fixing_times=remaining, coupon_rate=0.12, notional=100.0, strike=1.10,
                               currency_pair=PAIR, state=INCEPTION)


def test_r1_accrual_start_restores_the_full_first_coupon():
    remaining = [t - 0.125 for t in FIX[1:]]
    rebuilt = note(fix=remaining, accrual_start=FIX[0] - 0.125)
    assert is_close(rebuilt.accrual_factors[0], 1 / 12, rel_tol=1e-12)
    assert is_close(rebuilt.full_coupon(0), 1.0, rel_tol=1e-12)


def test_accrual_start_zero_is_todays_default_bitwise():
    explicit = note(fix=[0.5, 1.0, 1.5])
    assert explicit.accrual_factors == [0.5 - 0.0, 1.0 - 0.5, 1.5 - 1.0]


def test_inception_horizon_is_unchanged():
    n = note(pay_times=[t + 0.02 for t in FIX])
    assert n.maturity == FIX[-1] and n.delivery == FIX[-1] + 0.02
    assert n.lifecycle_events == () and n.is_numeric and not n.is_terminal and not n.is_redeemed


def test_hit_tolerance_lives_on_the_product():
    assert note(target=3.0).hit_tol == 3.0 * 1e-12
    assert note().hit_tol == 0.0


# --- R2 (numeric TARN refusals) ------------------------------------------------------------------------------------

@pytest.mark.parametrize("kw, match", [
    (dict(accrual_factors=[1 / 12] * 12, accrual_start=0.0), "not both"),
    (dict(state=None), "state is required"),
    (dict(state=TargetRedemptionState(3.0 + 1e-9), target=3.0), "exceeds the target"),
    (dict(accrual_start=FIX[0]), "before the first listed fixing"),
    (dict(accrual_start=0.01, state=TargetRedemptionState(1.0), target=3.0), "contiguous"),
    (dict(state=TargetRedemptionState(1.0, pending=(PendingPayment(-1.0, pay_time=0.1),)), target=3.0),
     "positive"),
    (dict(state=TargetRedemptionState(3.0), target=3.0), "redeemed"),
    (dict(coupon_rate=None), "coupon_rate"),
])
def test_r2_numeric_tarn_refusals(kw, match):
    with pytest.raises(ValidationError, match=match):
        note(**kw)


def test_r2_numeric_state_refuses_as_of_and_pay_dates():
    from datetime import datetime
    with pytest.raises(ValidationError, match="as_of"):
        note(state=TargetRedemptionState(0.0, as_of=datetime(2026, 6, 1)))
    dated_pending = TargetRedemptionState(1.0, pending=(PendingPayment(1.0, pay_date=datetime(2026, 7, 1)),),
                                          as_of=datetime(2026, 6, 1))
    with pytest.raises(ValidationError):
        note(state=dated_pending, target=3.0)


def test_r2_missing_state_is_a_validation_error_not_a_type_error():
    with pytest.raises(ValidationError, match="state is required"):
        FxTargetRedemptionNote(fixing_times=FIX, coupon_rate=0.12, strike=1.10, accrual_start=0.0)


def test_past_fixing_times_are_accepted():
    n = note(fix=[-0.01] + FIX[1:], accrual_start=-0.05)
    assert n.fixing_times[0] == -0.01


# --- WARN1 (numeric construction part; values in Task 6) -----------------------------------------------------------

def test_warn1_empty_schedule_booking_warns_once():
    with pytest.warns(TerminalBookingWarning) as rec:
        n = note(fix=[], accrual_factors=[], target=3.0,
                 state=TargetRedemptionState(1.0, pending=(PendingPayment(100.0, pay_time=0.05),)))
    assert len([w for w in rec if issubclass(w.category, TerminalBookingWarning)]) == 1
    assert n.is_terminal and not n.is_redeemed
    assert n.maturity == 0.05 and n.delivery == 0.05


def test_warn1_redeemed_booking_warns_once():
    with pytest.warns(TerminalBookingWarning):
        n = note(fix=[], accrual_factors=[], target=1.5, state=TargetRedemptionState(1.5))
    assert n.is_redeemed and n.is_terminal and n.maturity == 0.0


def test_live_booking_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", TerminalBookingWarning)
        note()


def test_deepcopy_does_not_warn_and_keeps_the_state():
    with pytest.warns(TerminalBookingWarning):
        n = note(fix=[], accrual_factors=[], target=1.5, state=TargetRedemptionState(1.5))
    with warnings.catch_warnings():
        warnings.simplefilter("error", TerminalBookingWarning)
        c = copy.deepcopy(n)
    assert c.state == n.state and c.lifecycle_events == ()
```

- [ ] **Step 3: Run to verify it fails**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_lifecycle.py -n 2 -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'state'` (fixtures pass `state`).

- [ ] **Step 4: Implement the base class**

`quantark/asset/fx/product/option/target_redemption_base.py`:

```python
"""Shared machinery of the FX target-redemption products (TARN and TARF).

Spec: docs/superpowers/specs/2026-10-09-fx-tarn-lifecycle-design.md, sections 3-6. A contract carries its realized
state (``state``: the amount banked towards the target and the determined cash still to pay). In the NUMERIC form
``fixing_times`` / ``pay_times`` are years from valuation and list ONLY the fixings not yet banked; a time at or
before valuation is an unbanked past fixing. Only the two concrete products are instantiated.
"""
from __future__ import annotations

import math
import warnings
from datetime import datetime
from types import SimpleNamespace
from typing import List, Optional, Tuple

from quantark.util.exceptions import ValidationError
from quantark.util.fixings import fixing_label
from quantark.util.numerical import is_zero

from ..base_fx_product import BaseFxProduct
from .target_redemption_state import (
    TargetRedemptionEvent,
    TargetRedemptionState,
    TerminalBookingWarning,
    is_target_hit,
    target_hit_tol,
)


def is_passed_time(t: float) -> bool:
    """A numeric fixing or pay time at or before valuation; ``is_zero`` absorbs the ULPs daily ageing leaves."""
    return t <= 0.0 or is_zero(t)


def event_label(index: int, *, time: Optional[float] = None, date: Optional[datetime] = None) -> str:
    """W5 ``fixing_label`` text (``#i t=<time>`` numeric, ``#i <date>`` dated) for a TARN/TARF fixing."""
    return fixing_label(index, SimpleNamespace(observation_time=time, observation_date=date))


class TargetRedemptionProduct(BaseFxProduct):
    """Base of ``FxTargetRedemptionNote`` and ``FxTargetRedemptionForward``: schedule forms, realized state,
    validation, terminal bookings, ageing and the resolved view."""

    #: TARN: per-period accrual factors aligned with the listed fixings.
    _has_accrual: bool = False
    #: TARN: determined cash is a coupon or the principal, so a pending amount must be > 0.
    _pending_must_be_positive: bool = False

    fixing_times: Optional[List[float]]
    pay_times: Optional[List[float]]
    lifecycle_events: Tuple[TargetRedemptionEvent, ...]

    # -- subclass contract --------------------------------------------------------------------------------------
    def _validate_terms(self) -> None:
        raise NotImplementedError

    def _fixing_outcome(self, factor: Optional[float], spot: float, accumulated: float) -> Tuple[float, float]:
        """``(amount banked towards the target, cash determined)`` for one fixing at ``spot``."""
        raise NotImplementedError

    def _principal_amount(self) -> float:
        """Cash due when the contract ends (the TARN principal); 0.0 when none."""
        return 0.0

    # -- construction -------------------------------------------------------------------------------------------
    @staticmethod
    def _require_state(state) -> TargetRedemptionState:
        if state is None:
            raise ValidationError(
                "state is required: a contract booked at inception passes TargetRedemptionState(0.0); a mid-life "
                "booking states what it has banked and what is still to pay"
            )
        if not isinstance(state, TargetRedemptionState):
            raise ValidationError(f"state must be a TargetRedemptionState, got {type(state).__name__}")
        return state

    def _init_numeric(self, fixings: List[float], pay_times, currency_pair, state: TargetRedemptionState) -> None:
        BaseFxProduct.__init__(self, currency_pair=currency_pair)
        self.fixing_times = list(fixings)
        self.pay_times = [float(t) for t in pay_times] if pay_times is not None else list(fixings)
        self.fixing_dates = None
        self.pay_dates = None
        self.state = state
        self.lifecycle_events = ()

    def _refresh_numeric_horizon(self) -> None:
        """``maturity`` = last listed fixing, else the last pending pay time, else 0.0; ``delivery`` alike."""
        if self.fixing_times:
            self.maturity = self.fixing_times[-1]
            self.delivery = max(self.pay_times)
        else:
            pending = [p.pay_time for p in self.state.pending]
            self.maturity = self.delivery = max(pending) if pending else 0.0

    def _warn_if_terminal_booking(self) -> None:
        if self.is_terminal:
            why = "redeemed" if self.is_redeemed else "every fixing banked"
            warnings.warn(
                TerminalBookingWarning(
                    f"{type(self).__name__} booked terminal ({why}): it is valued at the PV of its "
                    f"{len(self.state.pending)} pending payment(s), 0.0 when none"
                ),
                stacklevel=3,
            )

    # -- derived flags ------------------------------------------------------------------------------------------
    @property
    def is_numeric(self) -> bool:
        return self.fixing_dates is None

    @property
    def hit_tol(self) -> float:
        """Target-hit tolerance shared with the engines (``target_hit_tol``)."""
        return target_hit_tol(self.target)

    @property
    def is_unbounded_target(self) -> bool:
        return math.isinf(self.target)

    @property
    def is_redeemed(self) -> bool:
        return is_target_hit(self.state.accumulated, self.target)

    @property
    def is_terminal(self) -> bool:
        return not self.fixing_times or self.is_redeemed

    # -- validation ---------------------------------------------------------------------------------------------
    def validate(self) -> None:
        self._validate_terms()
        self._validate_numeric_schedule()
        self._validate_state()

    def _validate_numeric_schedule(self) -> None:
        fixings, pays = self.fixing_times, self.pay_times
        if list(fixings) != sorted(fixings):
            raise ValidationError("fixing_times must be ascending")
        if len(set(fixings)) != len(fixings):
            raise ValidationError("fixing_times must be unique")
        if any(not math.isfinite(t) for t in fixings):
            raise ValidationError("fixing_times must be finite")
        if len(pays) != len(fixings):
            raise ValidationError("pay_times must align 1:1 with fixing_times")
        for tf, tp in zip(fixings, pays):
            if not math.isfinite(tp):
                raise ValidationError(f"pay time {tp} must be finite")
            if tp < tf:
                raise ValidationError(f"pay time {tp} precedes its fixing {tf}")
        if list(pays) != sorted(pays):
            raise ValidationError("pay_times must be non-decreasing")
        if self._has_accrual:
            factors = self.accrual_factors
            if len(factors) != len(fixings):
                raise ValidationError("accrual_factors must align 1:1 with fixing_times")
            if any((not math.isfinite(a)) or a <= 0.0 for a in factors):
                raise ValidationError("accrual_factors must be positive and finite")

    def _validate_state(self) -> None:
        st = self.state
        if st.accumulated > self.target + self.hit_tol:
            raise ValidationError(f"state.accumulated {st.accumulated} exceeds the target {self.target}")
        if self._pending_must_be_positive and any(p.amount <= 0.0 for p in st.pending):
            raise ValidationError(
                "pending amounts must be positive: a TARN's determined cash is a coupon or its principal"
            )
        if st.as_of is not None:
            raise ValidationError("state.as_of belongs to the dated form: a numeric state lists only unbanked fixings")
        if any(p.pay_date is not None for p in st.pending):
            raise ValidationError("a numeric contract's pending payments take pay_time, not pay_date")
        if self.is_redeemed and self.fixing_times:
            raise ValidationError("a redeemed state cannot list fixings: every fixing after the target hit is dead")
```

- [ ] **Step 5: Rewrite the TARN constructor on the base**

In `fx_target_redemption_note.py`: change the base class to `TargetRedemptionProduct`, set `_has_accrual = True`,
`_pending_must_be_positive = True`, delete `_validate_schedule`, `is_unbounded_target` (inherited), and update the
module/class docstrings (`accrual_factors`: "explicit per-listed-period factors; or `accrual_start`, the time the
first listed period began — exactly one is required"; `state`: mandatory). Replace `__init__`/`validate`/`full_coupon`/
`get_payoff` with:

```python
import math
from typing import List, Optional, Tuple

from quantark.util.exceptions import ValidationError
from ..currency_pair import CurrencyPair
from .target_redemption_base import TargetRedemptionProduct
from .target_redemption_state import TargetRedemptionState


class FxTargetRedemptionNote(TargetRedemptionProduct):
    _has_accrual = True
    _pending_must_be_positive = True

    def __init__(
        self,
        fixing_times: Optional[List[float]] = None,
        coupon_rate: Optional[float] = None,
        notional: float = 1.0,
        target: float = math.inf,
        strike: Optional[float] = None,
        is_above: bool = True,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        accrual_factors: Optional[List[float]] = None,
        pay_times: Optional[List[float]] = None,
        include_principal: bool = True,
        currency_pair: Optional[CurrencyPair] = None,
        *,
        state: Optional[TargetRedemptionState] = None,
        accrual_start: Optional[float] = None,
    ):
        self.coupon_rate = coupon_rate
        self.notional = notional
        self.target = float(target)
        self.strike = strike
        self.is_above = bool(is_above)
        self.lower = lower
        self.upper = upper
        self.include_principal = bool(include_principal)
        state = self._require_state(state)
        if fixing_times is None:
            raise ValidationError("fixing_times is required")
        fixings = [float(t) for t in fixing_times]
        self.accrual_factors = self._numeric_accruals(fixings, accrual_start, accrual_factors, state)
        self._init_numeric(fixings, pay_times, currency_pair, state)
        self.validate()
        self._refresh_numeric_horizon()
        self._warn_if_terminal_booking()

    @staticmethod
    def _numeric_accruals(fixings, accrual_start, accrual_factors, state) -> List[float]:
        """The listed periods' factors: explicit, or ``t_i - t_{i-1}`` with ``t_{-1} = accrual_start`` (spec Q2a)."""
        if accrual_start is None and accrual_factors is None:
            raise ValidationError(
                "a numeric TARN must state when its first listed period began: pass accrual_start (years from "
                "valuation, negative when it began before) or explicit accrual_factors - no start is assumed"
            )
        if accrual_start is not None and accrual_factors is not None:
            raise ValidationError("give accrual_start or accrual_factors, not both")
        if accrual_factors is not None:
            return [float(a) for a in accrual_factors]
        start = float(accrual_start)
        if not math.isfinite(start):
            raise ValidationError(f"accrual_start must be finite, got {start}")
        if fixings and not start < fixings[0]:
            raise ValidationError(f"accrual_start {start} must be before the first listed fixing {fixings[0]}")
        if start > 0.0 and state.accumulated > 0.0:
            raise ValidationError(
                "accrual_start > 0 with accumulated > 0: periods are contiguous, so a contract that has banked a "
                "fixing has a listed period that began at or before valuation"
            )
        prev = [start] + fixings[:-1]
        return [t - p for t, p in zip(fixings, prev)]

    def _validate_terms(self) -> None:
        if self.coupon_rate is None:
            raise ValidationError("coupon_rate is required")
        for name, v in (("notional", self.notional), ("coupon_rate", self.coupon_rate)):
            if not math.isfinite(v) or v <= 0:
                raise ValidationError(f"{name} must be positive and finite, got {v}")
        if not (self.target > 0):  # rejects 0, negatives, NaN; inf passes
            raise ValidationError(f"target must be positive, got {self.target}")
        self._validate_condition()

    def coupon_amount(self, factor: float) -> float:
        """Undiscounted coupon of a period with accrual ``factor`` when the condition holds."""
        return self.notional * self.coupon_rate * factor

    def full_coupon(self, period: int) -> float:
        """Undiscounted coupon of listed period ``period`` if the condition holds."""
        return self.coupon_amount(self.accrual_factors[period])

    def _fixing_outcome(self, factor, spot, accumulated) -> Tuple[float, float]:
        coupon = self.coupon_amount(factor) if self.coupon_condition(spot) else 0.0
        c = min(coupon, self.target - accumulated)
        return c, c

    def _principal_amount(self) -> float:
        return self.notional if self.include_principal else 0.0

    def get_payoff(self, spot: float) -> float:
        """First listed period's coupon at ``spot`` (path-independent utility)."""
        if not self.accrual_factors:
            raise ValidationError("a terminal TARN has no next coupon")
        return self.full_coupon(0) if self.coupon_condition(spot) else 0.0
```

Keep `_validate_condition`, `is_band`, `coupon_condition`, the existing `time_shift` (replaced in Task 5) and
`__repr__` as they are.

- [ ] **Step 6: Update the TARN construction sites (constructor arguments only, spec §8)**

`test/test_fx_tarn_note_mc.py` — add the import and replace `_note`:

```python
from quantark.asset.fx.product.option.target_redemption_state import TargetRedemptionState


def _note(**kw):
    base = dict(fixing_times=[0.25, 0.5, 0.75, 1.0], coupon_rate=0.08,
               notional=1.0, strike=1.20, currency_pair=_PAIR,
               state=TargetRedemptionState(0.0))
    if "accrual_factors" not in kw:
        base["accrual_start"] = 0.0
    base.update(kw)
    return FxTargetRedemptionNote(**base)
```

`test/execution/matrix_fixtures.py` `fx_tarn` — add `TargetRedemptionState` import inside the function and the two
arguments:

```python
        from quantark.asset.fx.product.option.target_redemption_state import TargetRedemptionState
        ...
            FxTargetRedemptionNote(
                fixing_times=[0.25, 0.5, 0.75, 1.0], coupon_rate=0.08,
                notional=1.0, strike=1.20, currency_pair=_pair(),
                state=TargetRedemptionState(0.0), accrual_start=0.0,
            ),
```

`example/fx_tarn_demo.py` — import `TargetRedemptionState` from `quantark.asset.fx.product.option` and add
`state=TargetRedemptionState(0.0), accrual_start=0.0` to `note_open` (line 97) and `note_full` (line 109).

Append to this workstream's CHANGELOG block (create `### Changed` after `### Added`):

```markdown
### Changed
- **Breaking:** `FxTargetRedemptionNote` and `FxTargetRedemptionForward` take a mandatory keyword `state`
  (`TargetRedemptionState(0.0)` for a contract booked at inception); omitting it raises `ValidationError`.
- **Breaking:** a numeric `FxTargetRedemptionNote` states when its first listed period began: `accrual_start` (years
  from valuation) or explicit `accrual_factors`, exactly one. The old default (`t_0 = 0`, the first period starting at
  the valuation instant) halved the first coupon of a contract rebuilt mid-life. `accrual_start=0.0` reproduces the old
  factors bitwise.
- A non-positive fixing time is accepted (an unbanked past fixing) and an empty `fixing_times` is a terminal booking
  (`TerminalBookingWarning`, valued at the PV of its pending cash).
```

- [ ] **Step 7: Run the gates**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_lifecycle.py test/test_fx_tarn_note_mc.py -n 2 -q` and
`env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/execution/test_matrix_parity.py -k "TargetRedemption" -n 2 -q`
Expected: all pass (27 TARN tests unchanged apart from `_note`; 1 matrix parity test).
Then: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python <WT>/example/fx_tarn_demo.py | cmp - <WT>/.superpowers/w4b/demo_fx_tarn_demo.main.txt && echo SAME`
Expected: `SAME`.

- [ ] **Step 8: Commit**

```bash
git -C <WT> add quantark/asset/fx/product/option/target_redemption_base.py quantark/asset/fx/product/option/fx_target_redemption_note.py test/fx_target_redemption_fixtures.py test/test_fx_target_redemption_lifecycle.py test/test_fx_tarn_note_mc.py test/execution/matrix_fixtures.py example/fx_tarn_demo.py CHANGELOG.md
git -C <WT> commit -m "feat(fx-tarn): numeric TARN states its realized state and period start; terminal bookings warn

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Numeric TARF on the shared base

Spec §2.3, §4.1.

**Files:**
- Modify: `quantark/asset/fx/product/option/fx_target_redemption_forward.py`, `test/test_fx_tarf_mc.py` (`_tarf` only), `test/execution/matrix_fixtures.py:932-943`, `example/fx_tarn_demo.py:74-75, 85-87`
- Test: `test/test_fx_target_redemption_lifecycle.py`, `test/fx_target_redemption_fixtures.py`

**Interfaces:**
- Consumes: `TargetRedemptionProduct` (Task 3).
- Produces: `FxTargetRedemptionForward(strike, fixing_times=None, notional=1.0, target=math.inf, gearing=1.0, is_buy_base=True, pay_times=None, currency_pair=None, *, state=None)`; fixture `tarf(fix=FIX, state=INCEPTION, **kw)`.

- [ ] **Step 1: Add the fixture and failing tests**

Append to `test/fx_target_redemption_fixtures.py`:

```python
def tarf(fix=FIX, state=INCEPTION, **kw):
    base = dict(strike=1.15, fixing_times=fix, notional=1.0, currency_pair=PAIR, state=state)
    base.update(kw)
    return FxTargetRedemptionForward(**base)
```

Append to `test/test_fx_target_redemption_lifecycle.py` (add `tarf` to the fixtures import and
`FxTargetRedemptionForward` to the product import):

```python
@pytest.mark.parametrize("kw, match", [
    (dict(state=None), "state is required"),
    (dict(state=TargetRedemptionState(0.2 + 1e-9), target=0.2), "exceeds the target"),
    (dict(state=TargetRedemptionState(0.2), target=0.2), "redeemed"),
])
def test_r2_numeric_tarf_refusals(kw, match):
    with pytest.raises(ValidationError, match=match):
        tarf(**kw)


def test_tarf_pending_may_be_negative():
    f = tarf(gearing=2.0, target=0.2, state=TargetRedemptionState(0.0, pending=(PendingPayment(-0.1, pay_time=0.01),)))
    assert f.state.pending[0].amount == -0.1


def test_tarf_terminal_booking_warns():
    with pytest.warns(TerminalBookingWarning):
        f = tarf(fix=[], target=0.2, state=TargetRedemptionState(0.05))
    assert f.is_terminal and not f.is_redeemed and f.maturity == 0.0


def test_tarf_inception_horizon_is_unchanged():
    f = tarf(pay_times=[t + 0.02 for t in FIX])
    assert f.maturity == FIX[-1] and f.delivery == FIX[-1] + 0.02 and f.hit_tol == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_lifecycle.py -k tarf -n 2 -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'state'`.

- [ ] **Step 3: Implement**

Rebase `FxTargetRedemptionForward` on `TargetRedemptionProduct`; delete `_validate_schedule` and
`is_unbounded_target`; keep `period_cashflow`, `get_payoff`, `__repr__`, and the old `time_shift` (Task 5 replaces):

```python
    def __init__(
        self,
        strike: float,
        fixing_times: Optional[List[float]] = None,
        notional: float = 1.0,
        target: float = math.inf,
        gearing: float = 1.0,
        is_buy_base: bool = True,
        pay_times: Optional[List[float]] = None,
        currency_pair: Optional[CurrencyPair] = None,
        *,
        state: Optional[TargetRedemptionState] = None,
    ):
        self.strike = strike
        self.notional = notional
        self.target = float(target)
        self.gearing = gearing
        self.is_buy_base = bool(is_buy_base)
        state = self._require_state(state)
        if fixing_times is None:
            raise ValidationError("fixing_times is required")
        self._init_numeric([float(t) for t in fixing_times], pay_times, currency_pair, state)
        self.validate()
        self._refresh_numeric_horizon()
        self._warn_if_terminal_booking()

    def _validate_terms(self) -> None:
        for name, v in (("strike", self.strike), ("notional", self.notional)):
            if not math.isfinite(v) or v <= 0:
                raise ValidationError(f"{name} must be positive and finite, got {v}")
        if not math.isfinite(self.gearing) or self.gearing < 0:
            raise ValidationError(f"gearing must be non-negative and finite, got {self.gearing}")
        if not (self.target > 0):  # rejects 0, negatives, and NaN; inf passes
            raise ValidationError(f"target must be positive, got {self.target}")

    def _fixing_outcome(self, factor, spot, accumulated) -> Tuple[float, float]:
        """The engine's arithmetic for one fixing: favourable amount clipped to the remaining target, net cash."""
        sign = 1.0 if self.is_buy_base else -1.0
        diff = sign * (spot - self.strike)
        fav = self.notional * max(diff, 0.0)
        unf = self.notional * max(-diff, 0.0)
        fav_eff = min(fav, self.target - accumulated)
        return fav_eff, fav_eff - self.gearing * unf
```

Update the docstring (`state` mandatory; a non-positive fixing time is an unbanked past fixing; empty
`fixing_times` is a terminal booking).

Construction sites: `test/test_fx_tarf_mc.py` `_tarf` gains `state=TargetRedemptionState(0.0)` in `base` (import from
`quantark.asset.fx.product.option.target_redemption_state`); `matrix_fixtures.py` `fx_tarf` gains
`state=TargetRedemptionState(0.0)`; `example/fx_tarn_demo.py` `tarf_open` / `tarf_capped` gain
`state=TargetRedemptionState(0.0)`.

- [ ] **Step 4: Run the gates**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_lifecycle.py test/test_fx_tarf_mc.py test/test_fx_tarn_note_mc.py -n 2 -q` and
`env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/execution/test_matrix_parity.py -k "Tarn or TargetRedemption" -n 2 -q`
Expected: all pass (52 + 2 C1 gate). Demo: `... example/fx_tarn_demo.py | cmp - <WT>/.superpowers/w4b/demo_fx_tarn_demo.main.txt && echo SAME` → `SAME`.

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/asset/fx/product/option/fx_target_redemption_forward.py test/fx_target_redemption_fixtures.py test/test_fx_target_redemption_lifecycle.py test/test_fx_tarf_mc.py test/execution/matrix_fixtures.py example/fx_tarn_demo.py
git -C <WT> commit -m "feat(fx-tarn): numeric TARF states its realized state; terminal bookings warn

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Ageing hooks, numeric form (`time_shift`, `record_passed_fixings`, events)

Spec §6 (numeric), A1.3, Q3.

**Files:**
- Modify: `quantark/asset/fx/product/option/target_redemption_base.py`; delete the `time_shift` overrides in `fx_target_redemption_note.py` and `fx_target_redemption_forward.py`
- Test: `test/test_fx_target_redemption_lifecycle.py`

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces: `TargetRedemptionProduct.time_shift(time_bump: float) -> None`; `TargetRedemptionProduct.record_passed_fixings(fx_env) -> None`; private `_record_numeric(fx_env)`, `_settle_numeric(amount, pay_time, label, pending, events)`. Events: per banked fixing one FIXING (`time` = its signed time, `redeemed` when it ends the contract), immediately-due cash as PAYMENT with the FIXING's `label`; PAYMENT events from `time_shift` have `label=""`.

- [ ] **Step 1: Write the failing tests**

Append (add `from quantark.asset.fx.product.option.target_redemption_state import TargetRedemptionEvent, TargetRedemptionEventType`
and `FIXING, PAYMENT = TargetRedemptionEventType.FIXING, TargetRedemptionEventType.PAYMENT` near the top):

```python
LAG = [t + 0.02 for t in FIX]


def _kinds(events):
    return [(e.event_type, e.amount, e.fixing, e.time, e.redeemed) for e in events]


def test_time_shift_keeps_a_passed_fixing_unbanked():
    n = note(target=3.0, pay_times=LAG)
    a = copy.deepcopy(n)
    a.time_shift(0.1)
    assert a.fixing_times == [t - 0.1 for t in FIX] and a.pay_times == [t - 0.1 for t in LAG]
    assert a.state == n.state and a.lifecycle_events == ()
    assert a.maturity == FIX[-1] - 0.1


def test_record_banks_the_passed_fixing_at_the_held_spot():
    n = note(target=3.0, pay_times=LAG)
    a = copy.deepcopy(n)
    a.time_shift(0.1)
    a.record_passed_fixings(env(spot=1.20))
    c = n.full_coupon(0)
    assert a.state == TargetRedemptionState(c, pending=(PendingPayment(c, pay_time=LAG[0] - 0.1),))
    assert a.fixing_times == [t - 0.1 for t in FIX[1:]] and a.accrual_factors == n.accrual_factors[1:]
    assert _kinds(a.lifecycle_events) == [(FIXING, c, 1.20, FIX[0] - 0.1, False)]
    assert a.lifecycle_events[0].label.startswith("#0 t=")


def test_pending_cash_leaves_when_a_later_shift_reaches_its_pay_time():
    n = note(target=3.0, pay_times=LAG)
    a = copy.deepcopy(n)
    a.time_shift(0.1)
    a.record_passed_fixings(env())
    a.time_shift(0.01)
    c = n.full_coupon(0)
    assert a.state.pending == ()
    assert _kinds(a.lifecycle_events[-1:]) == [(PAYMENT, c, None, (LAG[0] - 0.1) - 0.01, False)]


def test_redemption_clips_the_coupon_and_settles_the_principal():
    n = note(target=1.5, pay_times=LAG)
    a = copy.deepcopy(n)
    a.time_shift(0.175)
    a.record_passed_fixings(env(spot=1.20))
    c0 = n.full_coupon(0)
    c1 = min(n.full_coupon(1), 1.5 - c0)
    t_pay = LAG[1] - 0.175
    assert a.is_redeemed and a.is_terminal
    assert a.fixing_times == [] and a.pay_times == [] and a.accrual_factors == []
    assert a.state == TargetRedemptionState(c0 + c1, pending=(PendingPayment(c1, pay_time=t_pay),
                                                              PendingPayment(100.0, pay_time=t_pay)))
    assert _kinds(a.lifecycle_events) == [
        (FIXING, c0, 1.20, FIX[0] - 0.175, False),
        (PAYMENT, c0, None, LAG[0] - 0.175, False),
        (FIXING, c1, 1.20, FIX[1] - 0.175, True),
    ]
    assert a.lifecycle_events[1].label == a.lifecycle_events[0].label
    assert a.maturity == t_pay


def test_quiet_fixing_banks_zero_and_no_cash():
    a = note(target=3.0, pay_times=LAG)
    a.time_shift(0.1)
    a.record_passed_fixings(env(spot=1.05))
    assert a.state == TargetRedemptionState(0.0)
    assert _kinds(a.lifecycle_events) == [(FIXING, 0.0, 1.05, FIX[0] - 0.1, False)]


@pytest.mark.parametrize("principal", [True, False])
def test_last_fixing_without_a_hit_settles_the_principal(principal):
    a = note(fix=[0.05], pay_times=[0.07], include_principal=principal)
    a.time_shift(0.06)
    a.record_passed_fixings(env(spot=1.05))
    assert a.is_terminal and not a.is_redeemed
    assert a.state.pending == ((PendingPayment(100.0, pay_time=0.07 - 0.06),) if principal else ())
    assert _kinds(a.lifecycle_events) == [(FIXING, 0.0, 1.05, 0.05 - 0.06, True)]


def test_tarf_unfavourable_fixing_is_negative_pending_cash():
    a = tarf(gearing=2.0, target=0.2, pay_times=LAG)
    a.time_shift(0.1)
    a.record_passed_fixings(env(spot=1.10))
    diff = 1.0 * (1.10 - 1.15)
    cf = 0.0 - 2.0 * (1.0 * max(-diff, 0.0))
    assert cf < 0.0
    assert a.state == TargetRedemptionState(0.0, pending=(PendingPayment(cf, pay_time=LAG[0] - 0.1),))


def test_tarf_redeems_on_target():
    a = tarf(target=0.075, pay_times=LAG)
    a.time_shift(0.175)
    a.record_passed_fixings(env(spot=1.20))
    fav = 1.0 * max(1.0 * (1.20 - 1.15), 0.0)
    second = min(fav, 0.075 - fav)
    assert a.is_redeemed and a.fixing_times == []
    assert a.state.accumulated == fav + second
    assert a.state.pending == (PendingPayment(second, pay_time=LAG[1] - 0.175),)  # gearing 1, favourable fixing


def test_record_before_the_first_fixing_changes_nothing():
    a = note(target=3.0)
    before = copy.deepcopy(a)
    a.record_passed_fixings(env())
    assert a.fixing_times == before.fixing_times and a.state == before.state and a.lifecycle_events == ()


def test_lifecycle_events_survive_deepcopy():
    a = note(target=3.0)
    a.time_shift(0.1)
    a.record_passed_fixings(env())
    assert copy.deepcopy(a).lifecycle_events == a.lifecycle_events


def test_banking_uses_quoted_spot_not_effective_spot():
    e = env(spot=1.20, spot_days=2)
    assert e.effective_spot() != e.spot
    a = note(target=3.0)
    a.time_shift(0.1)
    a.record_passed_fixings(e)
    assert a.lifecycle_events[0].fixing == e.spot
```

- [ ] **Step 2: Run to verify it fails**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_lifecycle.py -n 2 -q`
Expected: the new tests FAIL (base `record_passed_fixings` is a no-op; `state` unchanged).

- [ ] **Step 3: Implement**

Delete the `time_shift` methods of both products. Add to `TargetRedemptionProduct` (imports:
`PendingPayment, TargetRedemptionEventType` from the state module):

```python
    # -- ageing (spec section 6) --------------------------------------------------------------------------------
    def time_shift(self, time_bump: float) -> None:
        """Numeric form: move every fixing, pay and pending time by ``time_bump``. Pending cash now at or before
        valuation was paid within the bump and leaves as a PAYMENT event; fixings stepped over stay unbanked until
        :meth:`record_passed_fixings`. The dated form ages by its environment's date, so this is a no-op for it."""
        if not self.is_numeric:
            return
        self.fixing_times = [t - time_bump for t in self.fixing_times]
        self.pay_times = [t - time_bump for t in self.pay_times]
        kept, events = [], []
        for p in self.state.pending:
            t = p.pay_time - time_bump
            if is_passed_time(t):
                events.append(TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, p.amount, time=t))
            else:
                kept.append(PendingPayment(p.amount, pay_time=t))
        self.state = TargetRedemptionState(self.state.accumulated, pending=tuple(kept))
        self.lifecycle_events = self.lifecycle_events + tuple(events)
        self._refresh_numeric_horizon()
        self.validate()

    def record_passed_fixings(self, fx_env) -> None:
        """Bank every unbanked fixing at or before valuation, in order, at the QUOTED ``fx_env.spot`` (spec Q3),
        stopping once the contract is terminal; append the events; re-validate."""
        self._record_numeric(fx_env)
        self.validate()

    def _record_numeric(self, fx_env) -> None:
        spot = fx_env.spot
        acc = self.state.accumulated
        pending = list(self.state.pending)
        events: List[TargetRedemptionEvent] = []
        fixings, pays = self.fixing_times, self.pay_times
        factors = self.accrual_factors if self._has_accrual else None
        banked = 0
        while banked < len(fixings) and is_passed_time(fixings[banked]):
            t, tp = fixings[banked], pays[banked]
            added, cash = self._fixing_outcome(factors[banked] if factors is not None else None, spot, acc)
            acc = acc + added
            hit = is_target_hit(acc, self.target)
            ends = hit or banked == len(fixings) - 1
            label = event_label(banked, time=t)
            events.append(TargetRedemptionEvent(TargetRedemptionEventType.FIXING, cash, fixing=spot, time=t,
                                                redeemed=ends, label=label))
            self._settle_numeric(cash, tp, label, pending, events)
            if ends:
                self._settle_numeric(self._principal_amount(), tp, label, pending, events)
            banked += 1
            if hit:
                banked = len(fixings)  # every later fixing is dead
                break
        self.fixing_times = fixings[banked:]
        self.pay_times = pays[banked:]
        if factors is not None:
            self.accrual_factors = factors[banked:]
        self.state = TargetRedemptionState(acc, pending=tuple(sorted(pending, key=lambda p: p.pay_time)))
        self.lifecycle_events = self.lifecycle_events + tuple(events)
        self._refresh_numeric_horizon()

    @staticmethod
    def _settle_numeric(amount: float, pay_time: float, label: str, pending: list, events: list) -> None:
        """Determined cash: paid now (PAYMENT, the fixing's label) when its pay time has passed, else pending."""
        if is_zero(amount):
            return
        if is_passed_time(pay_time):
            events.append(TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, amount, time=pay_time,
                                                label=label))
        else:
            pending.append(PendingPayment(amount, pay_time=pay_time))
```

- [ ] **Step 4: Run to verify it passes**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_lifecycle.py test/test_fx_tarn_note_mc.py test/test_fx_tarf_mc.py -n 2 -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/asset/fx/product/option/target_redemption_base.py quantark/asset/fx/product/option/fx_target_redemption_note.py quantark/asset/fx/product/option/fx_target_redemption_forward.py test/test_fx_target_redemption_lifecycle.py
git -C <WT> commit -m "feat(fx-tarn): time_shift and record_passed_fixings bank fixings into the realized state

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Resolved view and engines (P1–P5, T1, T2, WARN1–3 numeric)

Spec §5.1, §5.2, §5.3, §8, §10 P/T/WARN/C1.

**Files:**
- Modify: `target_redemption_base.py` (add `resolve`, `_has_unbanked_passed`, `_view`, `pending_pv`), `quantark/asset/fx/engine/mc/fx_tarn_note_mc_engine.py`, `quantark/asset/fx/engine/mc/fx_tarf_mc_engine.py`, `test/test_fx_tarn_note_mc.py` (`_det_pv` gains `acc0`), `CHANGELOG.md`
- Test: `test/test_fx_target_redemption_lifecycle.py`, `test/test_fx_tarn_note_mc.py`

**Interfaces:**
- Consumes: Tasks 2–5.
- Produces: `TargetRedemptionProduct.resolve(fx_env, *, warn_missing_fixings: bool = True) -> ResolvedTargetRedemption`; `TargetRedemptionProduct.pending_pv(fx_env) -> float`. Engines: `_accumulate(self, opt, view, spots, df_pay)`; `_HIT_RTOL` and `_hit_tol` removed from both engines.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_fx_target_redemption_lifecycle.py` (imports: `from fx_target_redemption_fixtures import engine_for, mc`,
`from quantark.asset.fx.engine.mc import FxTargetRedemptionNoteMCEngine, FxTarnForwardMCEngine, FxTarnMCResult`,
`from quantark.util.enum.engine_enums import MonteCarloMethod`, `from quantark.util.fixings import MissingFixingWarning`):

```python
PSEUDO, QUASI = MonteCarloMethod.PSEUDO, MonteCarloMethod.QUASI


def _age(product, tau, spot):
    aged = copy.deepcopy(product)
    aged.time_shift(tau)
    aged.record_passed_fixings(env(spot=spot))
    return aged


def _rebuild_note(aged, source, *, accrual_start=None):
    kw = dict(fixing_times=list(aged.fixing_times), pay_times=list(aged.pay_times), coupon_rate=source.coupon_rate,
              notional=source.notional, target=source.target, strike=source.strike,
              include_principal=source.include_principal, currency_pair=PAIR, state=aged.state)
    if accrual_start is None:
        kw["accrual_factors"] = list(aged.accrual_factors)  # C0's surviving factors
    else:
        kw["accrual_start"] = accrual_start
    return FxTargetRedemptionNote(**kw)


def _rebuild_tarf(aged, source):
    return FxTargetRedemptionForward(strike=source.strike, fixing_times=list(aged.fixing_times),
                                     pay_times=list(aged.pay_times), notional=source.notional, target=source.target,
                                     gearing=source.gearing, currency_pair=PAIR, state=aged.state)


def _price(product, method=PSEUDO, e=None, n=20_000):
    return engine_for(product, n=n, method=method).price(product, e or env())


# --- P1-P5: numeric-vs-aged parity (A1.1) --------------------------------------------------------------------------

@pytest.mark.parametrize("method", [PSEUDO, QUASI])
@pytest.mark.parametrize("spot_h", [1.20, 1.05])
@pytest.mark.parametrize("tau", [0.125, 91 / 365])
def test_p1a_tarn_aged_equals_rebuilt_from_surviving_factors(tau, spot_h, method):
    c0 = note(target=3.0)
    aged = _age(c0, tau, spot_h)
    assert _price(aged, method) == _price(_rebuild_note(aged, c0), method)


@pytest.mark.parametrize("tau, n_banked, bitwise", [(0.125, 1, True), (91 / 365, 2, False)])
def test_p1b_tarn_rebuilt_from_accrual_start(tau, n_banked, bitwise):
    c0 = note(target=3.0)
    aged = _age(c0, tau, 1.20)
    assert len(aged.fixing_times) == 12 - n_banked
    p_aged = _price(aged)
    p_rebuilt = _price(_rebuild_note(aged, c0, accrual_start=FIX[n_banked - 1] - tau))  # Ruling R4
    assert p_aged == p_rebuilt if bitwise else is_close(p_aged, p_rebuilt, rel_tol=1e-12)


@pytest.mark.parametrize("tau", [0.1, 0.125, 91 / 365])  # Ruling R10: tau = 0.1 leaves the coupon pending
def test_p2_pay_lag_pending_is_stated_and_prices_bitwise(tau):
    c0 = note(target=3.0, pay_times=LAG)
    aged = _age(c0, tau, 1.20)
    if tau == 0.1:
        c = c0.full_coupon(0)
        stated = TargetRedemptionState(c, pending=(PendingPayment(c, pay_time=LAG[0] - 0.1),))
        assert aged.state == stated
    rebuilt = _rebuild_note(aged, c0)
    assert _price(aged) == _price(rebuilt)


def test_p3_across_redemption_aged_and_rebuilt_terminal():
    c0 = note(target=1.5, pay_times=LAG)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TerminalBookingWarning)  # WARN2: an aged copy that redeems does not warn
        aged = _age(c0, 0.175, 1.20)
        e = env()
        p_aged = _price(aged, e=e)
    c1 = min(c0.full_coupon(1), 1.5 - c0.full_coupon(0))
    t_pay = LAG[1] - 0.175
    assert p_aged == 0.0 + c1 * e.get_domestic_df(t_pay) + 100.0 * e.get_domestic_df(t_pay)
    with pytest.warns(TerminalBookingWarning):  # WARN1
        rebuilt = note(fix=[], accrual_factors=[], target=1.5, state=aged.state)
    assert _price(rebuilt, e=e) == p_aged


@pytest.mark.parametrize("method", [PSEUDO, QUASI])
@pytest.mark.parametrize("spot_h", [1.20, 1.10])
@pytest.mark.parametrize("tau", [0.1, 0.125, 91 / 365])
def test_p4_tarf_aged_equals_rebuilt(tau, spot_h, method):
    c0 = tarf(target=0.20, gearing=2.0, pay_times=LAG)
    aged = _age(c0, tau, spot_h)
    if tau == 0.1 and spot_h == 1.10:
        assert aged.state.pending and aged.state.pending[0].amount < 0.0
    assert _price(aged, method) == _price(_rebuild_tarf(aged, c0), method)


def test_p4_tarf_across_redemption():
    c0 = tarf(target=0.075, pay_times=LAG)
    aged = _age(c0, 0.175, 1.20)
    e = env()
    fav = 1.0 * max(1.0 * (1.20 - 1.15), 0.0)
    second = min(fav, 0.075 - fav)
    assert aged.is_terminal and _price(aged, e=e) == 0.0 + second * e.get_domestic_df(LAG[1] - 0.175)
    with pytest.warns(TerminalBookingWarning):
        rebuilt = tarf(fix=[], target=0.075, state=aged.state)
    assert _price(rebuilt, e=e) == _price(aged, e=e)


def test_p5_daily_ageing_equals_one_rebuild():
    c0 = note(target=3.0)
    f0 = tarf(target=0.20)
    aged_n, aged_f = copy.deepcopy(c0), copy.deepcopy(f0)
    e = env()
    for _ in range(91):
        for a in (aged_n, aged_f):
            a.time_shift(1 / 365)
            a.record_passed_fixings(e)
    tau = 91 / 365
    rebuilt_n = note(fix=[t - tau for t in FIX[2:]], accrual_factors=c0.accrual_factors[2:], target=3.0,
                     state=TargetRedemptionState(c0.full_coupon(0) + c0.full_coupon(1)))
    fav = 1.0 * max(1.0 * (1.20 - 1.15), 0.0)
    rebuilt_f = tarf(fix=[t - tau for t in FIX[2:]], target=0.20, state=TargetRedemptionState(fav + fav))
    assert aged_n.state.accumulated == rebuilt_n.state.accumulated
    assert aged_f.state.accumulated == rebuilt_f.state.accumulated
    assert is_close(_price(aged_n), _price(rebuilt_n), rel_tol=1e-12)
    assert is_close(_price(aged_f), _price(rebuilt_f), rel_tol=1e-12)


# --- WARN1 values (numeric) and terminal engine result -------------------------------------------------------------

def test_warn1_terminal_values_and_no_repeat_warning():
    with pytest.warns(TerminalBookingWarning):
        empty = note(fix=[], accrual_factors=[], target=3.0, state=TargetRedemptionState(1.0))
        paid_out = tarf(fix=[], target=0.2, state=TargetRedemptionState(0.2))
        owed = note(fix=[], accrual_factors=[], target=1.5,
                    state=TargetRedemptionState(1.5, pending=(PendingPayment(100.5, pay_time=0.05),)))
    e = env()
    with warnings.catch_warnings():
        warnings.simplefilter("error", TerminalBookingWarning)
        assert _price(empty, e=e) == 0.0
        assert _price(paid_out, e=e) == 0.0
        assert _price(owed, e=e) == 0.0 + 100.5 * e.get_domestic_df(0.05)
        for product in (empty, paid_out, owed):
            engine_for(product).calculate_greeks(product, e)


def test_terminal_engine_result_fields():
    with pytest.warns(TerminalBookingWarning):
        redeemed = note(fix=[], accrual_factors=[], target=1.5, state=TargetRedemptionState(1.5))
    eng = FxTargetRedemptionNoteMCEngine(mc())
    assert eng.price(redeemed, env()) == 0.0
    assert eng.get_last_result() == FxTarnMCResult(0.0, 0.0, 0, 0.0, 1.0)


# --- WARN3 (numeric): an unbanked passed fixing is filled once, on a copy ------------------------------------------

@pytest.mark.parametrize("make", [lambda: note(target=3.0), lambda: tarf(target=0.20)])
def test_warn3_numeric_passed_fixing_is_filled_with_one_warning(make):
    bare = make()
    bare.time_shift(0.125)
    snapshot = copy.deepcopy(bare)
    e = env(spot=1.20)
    with pytest.warns(MissingFixingWarning) as rec:
        p = _price(bare, e=e)
    missing = [w for w in rec if issubclass(w.category, MissingFixingWarning)]
    assert len(missing) == 1 and "#0" in str(missing[0].message)
    recorded = copy.deepcopy(bare)
    recorded.record_passed_fixings(e)
    assert p == _price(recorded, e=e)
    assert (bare.fixing_times, bare.pay_times, bare.state, bare.lifecycle_events) == (
        snapshot.fixing_times, snapshot.pay_times, snapshot.state, snapshot.lifecycle_events)


# --- T1, T2: theta across a fixing / past the last fixing ----------------------------------------------------------

@pytest.mark.parametrize("first", [0.5 / 365, 1 / 365])
@pytest.mark.parametrize("make", [lambda f: note(fix=f, target=3.0), lambda f: tarf(fix=f, target=0.05)])
def test_t1_theta_steps_over_a_fixing(first, make):
    product = make([first] + FIX[1:])
    greeks = engine_for(product).calculate_greeks(product, env())
    assert math.isfinite(greeks["theta"])


def test_t1_two_days_away_is_mains_theta_to_the_bit():
    fix = [2 / 365] + FIX[1:]
    n, f = note(fix=fix, target=3.0), tarf(fix=fix, target=0.05)
    gn = engine_for(n).calculate_greeks(n, env())
    gf = engine_for(f).calculate_greeks(f, env())
    # main dcea77b6 (plan Task 1 Step 4): inception prices and theta unchanged to the bit
    assert (gn["price"], gn["theta"]) == (101.63518163030555, 0.01686139016959487)
    assert (gf["price"], gf["theta"]) == (0.01925287164907336, 0.003076911159477322)


def test_t2_bump_past_the_last_fixing_is_the_pending_carry():
    opt = note(fix=[0.5 / 365], pay_times=[30 / 365], target=3.0)
    e = env()
    g = engine_for(opt).calculate_greeks(opt, e)
    c = opt.full_coupon(0)  # spot 1.20 >= 1.10: the coupon pays
    t = 30 / 365 - 1 / 365
    future = 0.0 + c * e.get_domestic_df(t) + 100.0 * e.get_domestic_df(t)
    assert g["theta"] == (future - g["price"]) / 1.0


def test_t2_terminal_theta_is_pure_carry():
    with pytest.warns(TerminalBookingWarning):
        owed = note(fix=[], accrual_factors=[], target=1.5,
                    state=TargetRedemptionState(1.5, pending=(PendingPayment(100.5, pay_time=30 / 365),)))
    e = env()
    g = engine_for(owed).calculate_greeks(owed, e)
    base = 0.0 + 100.5 * e.get_domestic_df(30 / 365)
    future = 0.0 + 100.5 * e.get_domestic_df(30 / 365 - 1 / 365)
    assert g["price"] == base and g["theta"] == (future - base) / 1.0
```

In `test/test_fx_tarn_note_mc.py`, extend `_det_pv` (spec T1: "`_det_pv` ... extended with a start `acc`") and add the
small-vol landing-fixing check to `TestDeterministicLimit`:

```python
def _det_pv(forwards, df_pay, opt, acc0=0.0):
    acc, pv, alive = acc0, 0.0, True
    ...  # body unchanged


    def test_landing_fixing_theta_matches_forward_path(self):
        from copy import deepcopy
        env = _env(vol=1e-7)
        fix = [0.5 / 365, 0.25, 0.5, 0.75, 1.0]
        opt = _note(fixing_times=fix)
        theta = FxTargetRedemptionNoteMCEngine(params=_mc(num_paths=20_000)).calculate_greeks(opt, env)["theta"]
        fwd, df = _forwards_and_df(env, opt)
        base = _det_pv(fwd, df, opt)
        aged = deepcopy(opt)
        aged.time_shift(1 / 365)
        aged.record_passed_fixings(env)
        fwd_a, df_a = _forwards_and_df(env, aged)
        future = _det_pv(fwd_a, df_a, aged, acc0=aged.state.accumulated) + aged.pending_pv(env)
        assert theta == pytest.approx(future - base, abs=1e-6, rel=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_lifecycle.py test/test_fx_tarn_note_mc.py -n 2 -q`
Expected: FAIL — terminal products raise in the engine (`IndexError` / `ValidationError` on empty fixings), the P1
tests raise `ValidationError: Time to maturity must be non-negative` (engine prices passed pay times), `pending_pv`
missing.

- [ ] **Step 3: Implement the resolved view**

Add to `TargetRedemptionProduct` (imports: `from copy import deepcopy`, `from quantark.util.fixings import
warn_filled_fixings`, `ResolvedTargetRedemption, pending_present_value` from the state module):

```python
    # -- resolved view (spec section 5.1) ------------------------------------------------------------------------
    def resolve(self, fx_env, *, warn_missing_fixings: bool = True) -> ResolvedTargetRedemption:
        """The engines' only input. Unbanked fixings at or before valuation are banked on a COPY at the quoted spot
        (``record_passed_fixings``) with one ``MissingFixingWarning`` naming them (W5); the product is unchanged."""
        if self._has_unbanked_passed(fx_env):
            banked = deepcopy(self)
            start = len(banked.lifecycle_events)
            banked.record_passed_fixings(fx_env)
            if warn_missing_fixings:
                labels = [e.label for e in banked.lifecycle_events[start:]
                          if e.event_type is TargetRedemptionEventType.FIXING]
                warn_filled_fixings(type(self).__name__, "a banked outcome", labels)
            return banked._view(fx_env)
        return self._view(fx_env)

    def _has_unbanked_passed(self, fx_env) -> bool:
        return bool(self.fixing_times) and is_passed_time(self.fixing_times[0])

    def _view(self, fx_env) -> ResolvedTargetRedemption:
        st = self.state
        return ResolvedTargetRedemption(
            fixing_times=tuple(self.fixing_times),
            pay_times=tuple(self.pay_times),
            accrual_factors=tuple(self.accrual_factors) if self._has_accrual else None,
            accumulated=st.accumulated,
            pending=tuple((p.pay_time, p.amount) for p in st.pending),
            redeemed=self.is_redeemed,
            terminal=self.is_terminal,
        )

    def pending_pv(self, fx_env) -> float:
        """PV of the pending cash still due after valuation (no warning: callers have banked the passed fixings)."""
        return pending_present_value(self.resolve(fx_env, warn_missing_fixings=False).pending, fx_env)
```

- [ ] **Step 4: Make both engines read the view**

`fx_tarn_note_mc_engine.py` — import `pending_present_value` from
`quantark.asset.fx.product.option.target_redemption_state`; delete `_HIT_RTOL` and `_hit_tol`; replace the body of
`price` after the RANDOMIZED_QUASI guard up to `spots = ...` and the `_accumulate` call:

```python
        view = opt.resolve(fx_env)
        pv_pending = pending_present_value(view.pending, fx_env)
        if view.terminal:  # spec 5.2 step 2: no simulation, no vol lookup
            self._last_result = FxTarnMCResult(pv_pending, 0.0, 0, 0.0, 1.0 if view.redeemed else 0.0)
            return pv_pending

        spot = fx_env.effective_spot()
        if spot <= 0:
            raise ValidationError(f"Spot must be positive, got {spot}")
        T = float(view.fixing_times[-1])  # == opt.get_maturity at a numeric inception, bitwise
        sigma = float(fx_env.get_vol(fx_env.get_forward(T), T))
        if sigma <= 0:
            raise ValidationError(f"vol must be positive, got {sigma}")

        fixings = np.asarray(view.fixing_times, dtype=float)
        df_pay = np.array([fx_env.get_domestic_df(float(t)) for t in view.pay_times])
        spots = self._simulate(fx_env, spot, sigma, fixings)  # (P, M)

        pv, redeemed = self._accumulate(opt, view, spots, df_pay)
        if view.pending:
            pv = pv + pv_pending
```
(the rest of `price` — mean, std error, negative-price `PricingError`, `_last_result` — is unchanged). In
`_accumulate(self, opt, view, spots, df_pay)` change exactly three lines:

```python
        acc = np.full(n_paths, view.accumulated)   # was np.zeros(n_paths): identical at inception
        tol = opt.hit_tol                          # was self._hit_tol(target)
            full_c = opt.coupon_amount(view.accrual_factors[i])  # was opt.full_coupon(i)
```

`fx_tarf_mc_engine.py` — same edits (`view`, terminal branch, `T`, `fixings`, `df_pay` from the view,
`pv = pv + pv_pending` when pending, `acc = np.full(n_paths, view.accumulated)`, `tol = opt.hit_tol`, signature
`_accumulate(self, opt, view, spots, df_pay)`); delete `_HIT_RTOL` and `_hit_tol`. Update both module docstrings: "the
walk starts from the product's realized state (`resolve`); a terminal contract is the PV of its pending cash".

Append to the CHANGELOG block (create `### Fixed` after `### Changed`):

```markdown
### Fixed
- FX TARN / TARF theta within one day of a fixing raised (`Time to maturity must be non-negative` / `times must be
  strictly increasing and positive`); the stepped-over fixing is now banked at the held spot and theta is finite.
- Pricing a TARN / TARF after a bare `time_shift` past a fixing raised; the passed fixing is now filled at the quoted
  spot on a copy with one `MissingFixingWarning`.
- A rebuilt mid-life contract no longer prices with the full target: the engines start from `state.accumulated` and
  add the PV of `state.pending`. Inception prices and Greeks are unchanged to the bit.
```

- [ ] **Step 5: Run to verify it passes**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_lifecycle.py test/test_fx_tarn_note_mc.py test/test_fx_tarf_mc.py test/test_fx_target_redemption_state.py -n 2 -q` and the matrix parity `-k "Tarn or TargetRedemption"`.
Expected: all pass. Demo byte compare (Task 3 Step 7 command) → `SAME`.

- [ ] **Step 6: Commit**

```bash
git -C <WT> add quantark/asset/fx/product/option/target_redemption_base.py quantark/asset/fx/engine/mc/fx_tarn_note_mc_engine.py quantark/asset/fx/engine/mc/fx_tarf_mc_engine.py test/test_fx_target_redemption_lifecycle.py test/test_fx_tarn_note_mc.py CHANGELOG.md
git -C <WT> commit -m "feat(fx-tarn): engines price the resolved view - realized state, pending PV, terminal contracts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Dated form (D1, D3, D4 calendar, D5, R2 dated, WARN1/WARN3 dated)

Spec §4.2, §4.3, §5.1 (dated), §6 (dated), §10 D1/D3/D4/D5/R2/WARN1/WARN3.

**Files:**
- Modify: `target_redemption_base.py`, `fx_target_redemption_note.py`, `fx_target_redemption_forward.py`, `test/fx_target_redemption_fixtures.py`, `CHANGELOG.md`
- Test: `test/test_fx_target_redemption_lifecycle.py`

**Interfaces:**
- Consumes: Tasks 2–6.
- Produces:
  - TARN keywords `fixing_dates`, `pay_dates`, `initial_date`, `day_count_convention`; TARF keywords `fixing_dates`, `pay_dates`.
  - Base: `signed_year_fraction(date, fx_env) -> float`; `_schedule_form(fixing_times, fixing_dates) -> bool` (True = dated); `_init_dated`, `_refresh_dated_horizon`, `_validate_dated_schedule`, `_validate_dated_accrual` (hook, TARN overrides), `_record_dated`, `_dated_view`, `_is_unbanked(i)`, `_check_as_of(fx_env)`, `_dated_factor(index, fx_env) -> Optional[float]` (hook, TARN overrides); `get_maturity` / `get_delivery` return 0.0 past `expiry_date` / `delivery_date`.
  - Fixtures: `FIX_DATES`, `LAG2`, `TARGET_CAL`, `following(d)`, `dated_note(...)`, `dated_tarf(...)`, `_numeric_twin(dated, fx_env)`.

- [ ] **Step 1: Extend the fixtures**

Append to `test/fx_target_redemption_fixtures.py`:

```python
from datetime import timedelta

from quantark.asset.fx.product.option.target_redemption_state import PendingPayment
from quantark.util.calendar.business_calendar import CalendarType, create_calendar

FIX_DATES = [datetime(2026 + (6 + m) // 12, (6 + m) % 12 + 1, 15) for m in range(12)]  # 2026-07-15 .. 2027-06-15
LAG2 = [d + timedelta(days=2) for d in FIX_DATES]
TARGET_CAL = create_calendar(CalendarType.TARGET, year_range=(2026, 2028))


def following(d):
    """Test-side business-day adjustment (the product takes dates as stated)."""
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def dated_note(fixing_dates=FIX_DATES, state=INCEPTION, **kw):
    base = dict(fixing_dates=fixing_dates, coupon_rate=0.12, notional=100.0, strike=1.10, currency_pair=PAIR,
                state=state)
    if "accrual_factors" not in kw:
        base["initial_date"] = VAL
    base.update(kw)
    return FxTargetRedemptionNote(**base)


def dated_tarf(fixing_dates=FIX_DATES, state=INCEPTION, **kw):
    base = dict(strike=1.15, fixing_dates=fixing_dates, notional=1.0, currency_pair=PAIR, state=state)
    base.update(kw)
    return FxTargetRedemptionForward(**base)


def _numeric_twin(dated, fx_env):
    """Spec 4.3: the numeric contract with the dated contract's unbanked fixings, resolved factors and state."""
    view = dated.resolve(fx_env, warn_missing_fixings=False)
    state = TargetRedemptionState(view.accumulated,
                                  pending=tuple(PendingPayment(a, pay_time=t) for t, a in view.pending))
    common = dict(fixing_times=list(view.fixing_times), pay_times=list(view.pay_times), target=dated.target,
                  currency_pair=dated.currency_pair, state=state)
    if isinstance(dated, FxTargetRedemptionNote):
        return FxTargetRedemptionNote(coupon_rate=dated.coupon_rate, notional=dated.notional, strike=dated.strike,
                                      is_above=dated.is_above, lower=dated.lower, upper=dated.upper,
                                      include_principal=dated.include_principal,
                                      accrual_factors=list(view.accrual_factors), **common)
    return FxTargetRedemptionForward(strike=dated.strike, notional=dated.notional, gearing=dated.gearing,
                                     is_buy_base=dated.is_buy_base, **common)
```

- [ ] **Step 2: Write the failing tests**

Append to `test/test_fx_target_redemption_lifecycle.py` (imports: `dataclasses`, `from datetime import datetime,
timedelta`, the new fixtures, `from quantark.param import SpotQuote`,
`from quantark.util.calendar import DayCountConvention, calculate_year_fraction`):

```python
BUS = dict(dc=DayCountConvention.BUSINESS_DAYS, calendar=TARGET_CAL)
BUS_DATES = [following(d) for d in FIX_DATES]


def _dated(kind, dates=FIX_DATES, **kw):
    if kind == "tarf":
        return dated_tarf(fixing_dates=dates, target=kw.pop("target", 0.20), **kw)
    return dated_note(fixing_dates=dates, target=kw.pop("target", 3.0), include_principal=(kind == "tarn"), **kw)


def _at(e, days, spot=None):
    changes = {"valuation_date": e.valuation_date + timedelta(days=days)}
    if spot is not None:
        changes["spot_quote"] = SpotQuote(spot=spot)
    return dataclasses.replace(e, **changes)


# --- D1: twins bitwise at booking ----------------------------------------------------------------------------------

@pytest.mark.parametrize("calendar", ["calendar", "business"])
@pytest.mark.parametrize("method", [PSEUDO, QUASI])
@pytest.mark.parametrize("kind", ["tarn", "tarn_no_principal", "tarf"])
def test_d1_twin_prices_bitwise_at_booking(kind, method, calendar):
    e = env() if calendar == "calendar" else env(**BUS)
    dated = _dated(kind, dates=FIX_DATES if calendar == "calendar" else BUS_DATES)
    twin = _numeric_twin(dated, e)
    assert _price(dated, method, e) == _price(twin, method, e)


# --- D3: a dated booking rebuilt mid-life equals the aged contract -------------------------------------------------

def test_d3_dated_rebuild_mid_life_equals_aged_and_so_does_its_twin():
    e0 = env(spot=1.20)
    aged = dated_note(target=3.0, pay_dates=LAG2)
    twin = _numeric_twin(aged, e0)
    for k in range(1, 101):
        ek = _at(e0, k)
        aged.record_passed_fixings(ek)
        twin.time_shift(1 / 365)
        twin.record_passed_fixings(ek)
    mid = _at(e0, 100)
    rebuilt = dated_note(target=3.0, pay_dates=LAG2, state=aged.state)
    assert rebuilt.state == aged.state and aged.state.as_of == mid.valuation_date
    assert is_close(_price(rebuilt, e=mid), _price(aged, e=mid), rel_tol=1e-12)
    rebuilt_twin = _numeric_twin(rebuilt, mid)
    assert rebuilt_twin.state.accumulated == twin.state.accumulated
    assert [p.amount for p in rebuilt_twin.state.pending] == [p.amount for p in twin.state.pending]
    assert is_close(_price(rebuilt_twin, e=mid), _price(twin, e=mid), rel_tol=1e-12)


# --- D4 (CALENDAR_DAYS): theta twins agree, including a fixing inside the bump -------------------------------------

@pytest.mark.parametrize("inside_bump", [False, True])
@pytest.mark.parametrize("kind", ["tarn", "tarf"])
def test_d4_theta_twins_agree_calendar_days(kind, inside_bump):
    dates = ([VAL + timedelta(days=1)] + FIX_DATES[1:]) if inside_bump else FIX_DATES
    kw = {"initial_date": VAL - timedelta(days=29)} if (inside_bump and kind == "tarn") else {}
    dated = _dated(kind, dates=dates, **kw)
    e = env()
    twin = _numeric_twin(dated, e)
    gd = engine_for(dated).calculate_greeks(dated, e)
    gn = engine_for(twin).calculate_greeks(twin, e)
    assert gd["price"] == gn["price"]
    assert abs(gd["theta"] - gn["theta"]) <= 1e-12 * abs(gd["price"])  # Ruling R5


# --- D5: dated accrual factors -------------------------------------------------------------------------------------

def test_d5_initial_date_gives_act_365_factors():
    view = dated_note().resolve(env())
    starts = [VAL] + FIX_DATES[:-1]
    assert view.accrual_factors == tuple(
        calculate_year_fraction(s, d, DayCountConvention.ACT_365, 252) for s, d in zip(starts, FIX_DATES))


def test_d5_business_days_contract_count_reads_the_environment_calendar():
    dated = dated_note(fixing_dates=BUS_DATES, day_count_convention=DayCountConvention.BUSINESS_DAYS)
    view = dated.resolve(env(calendar=TARGET_CAL))
    starts = [VAL] + BUS_DATES[:-1]
    assert view.accrual_factors == tuple(
        calculate_year_fraction(s, d, DayCountConvention.BUSINESS_DAYS, 252, calendar=TARGET_CAL)
        for s, d in zip(starts, BUS_DATES))
    with pytest.raises(ValidationError, match="calendar"):
        dated.resolve(env())  # no calendar on the environment


def test_d5_both_period_sources_refused():
    with pytest.raises(ValidationError, match="not both"):
        dated_note(initial_date=VAL, accrual_factors=[1 / 12] * 12)


def test_d5_neither_period_source_refused():
    with pytest.raises(ValidationError, match="initial_date"):
        FxTargetRedemptionNote(fixing_dates=FIX_DATES, coupon_rate=0.12, strike=1.10, state=INCEPTION)


# --- R2 (dated and mixed forms) ------------------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [
    dict(fixing_times=FIX),                                   # both schedules
    dict(pay_times=FIX),                                      # numeric pay with dates
    dict(accrual_start=0.0),                                  # numeric start with dates
    dict(accrual_factors=[1 / 12] * 12, day_count_convention=DayCountConvention.ACT_365, initial_date=None),
    dict(initial_date=FIX_DATES[0]),                          # initial_date not before the first fixing
    dict(fixing_dates=[]),
    dict(fixing_dates=list(reversed(FIX_DATES))),
    dict(fixing_dates=FIX_DATES[:1] * 2),
    dict(pay_dates=FIX_DATES[:-1]),
    dict(pay_dates=[d - timedelta(days=1) for d in FIX_DATES]),
    dict(state=TargetRedemptionState(1.0)),                   # accumulated > 0 with as_of None
    dict(state=TargetRedemptionState(1.0, as_of=VAL)),        # as_of before the first fixing
])
def test_r2_dated_refusals(kw):
    with pytest.raises(ValidationError):
        dated_note(**kw)


@pytest.mark.parametrize("kw", [
    dict(pay_dates=FIX_DATES), dict(initial_date=VAL), dict(day_count_convention=DayCountConvention.ACT_365)])
def test_r2_dated_terms_refused_on_the_numeric_form(kw):
    with pytest.raises(ValidationError, match="dated form"):
        note(**kw)


def test_r2_neither_schedule_refused():
    with pytest.raises(ValidationError, match="schedule"):
        FxTargetRedemptionForward(strike=1.15, state=INCEPTION)


def test_r2_dated_pending_must_follow_as_of():
    with pytest.raises(ValidationError, match="after as_of"):
        dated_note(state=TargetRedemptionState(1.0, pending=(PendingPayment(1.0, pay_date=FIX_DATES[0]),),
                                               as_of=FIX_DATES[0]))


def test_r2_as_of_after_valuation_is_refused_at_pricing():
    booked = dated_note(state=TargetRedemptionState(0.0, as_of=FIX_DATES[1]))
    with pytest.raises(ValidationError, match="as_of"):
        _price(booked, e=env())


def test_future_fixing_with_zero_year_fraction_is_refused():
    friday = datetime(2026, 8, 14)
    saturday_fixing = dated_note(fixing_dates=[datetime(2026, 8, 15), datetime(2026, 9, 15)],
                                 initial_date=datetime(2026, 7, 15))
    with pytest.raises(ValidationError, match="zero year fraction"):
        saturday_fixing.resolve(env(val=friday, **BUS))


# --- dated record semantics, maturity ------------------------------------------------------------------------------

def test_dated_record_banks_through_the_valuation_date_and_sweeps_paid_cash():
    d = dated_note(target=3.0, pay_dates=LAG2)
    d.record_passed_fixings(env(val=FIX_DATES[0]))
    coupon = 100.0 * 0.12 * calculate_year_fraction(VAL, FIX_DATES[0], DayCountConvention.ACT_365, 252)
    assert d.state == TargetRedemptionState(coupon, pending=(PendingPayment(coupon, pay_date=LAG2[0]),),
                                            as_of=FIX_DATES[0])
    d.record_passed_fixings(env(val=LAG2[0]))
    assert d.state.pending == () and d.lifecycle_events[-1].event_type is PAYMENT
    assert d.state.as_of == LAG2[0]


def test_dated_time_shift_is_a_noop():
    d = dated_note()
    d.time_shift(0.5)
    assert d.state == INCEPTION and d.lifecycle_events == ()


def test_dated_maturity_and_delivery_after_the_end_are_zero():
    with pytest.warns(TerminalBookingWarning):
        done = dated_note(target=3.0, pay_dates=LAG2, state=TargetRedemptionState(1.0, as_of=FIX_DATES[-1]))
    late = env(val=LAG2[-1] + timedelta(days=5))
    assert done.get_maturity(late) == 0.0 and done.get_delivery(late) == 0.0
    assert done.get_maturity(env()) == calculate_year_fraction(VAL, FIX_DATES[-1],
                                                                DayCountConvention.CALENDAR_DAYS, 252)


# --- WARN1 / WARN3 (dated) -----------------------------------------------------------------------------------------

def test_warn1_dated_terminal_bookings():
    e = env(val=FIX_DATES[-1] + timedelta(days=1))
    with pytest.warns(TerminalBookingWarning):
        all_banked = dated_note(target=3.0, pay_dates=LAG2, state=TargetRedemptionState(
            1.0, pending=(PendingPayment(100.0, pay_date=LAG2[-1]),), as_of=FIX_DATES[-1]))
    with pytest.warns(TerminalBookingWarning):
        redeemed = dated_note(target=1.5, pay_dates=LAG2, state=TargetRedemptionState(1.5, as_of=FIX_DATES[1]))
    with warnings.catch_warnings():
        warnings.simplefilter("error", TerminalBookingWarning)
        t = calculate_year_fraction(e.valuation_date, LAG2[-1], DayCountConvention.CALENDAR_DAYS, 252)
        assert _price(all_banked, e=e) == 0.0 + 100.0 * e.get_domestic_df(t)
        assert _price(redeemed, e=e) == 0.0
        engine_for(all_banked).calculate_greeks(all_banked, e)


def test_warn3_dated_passed_fixing_is_filled_with_one_warning():
    d = dated_note(target=3.0)
    snapshot = copy.deepcopy(d)
    e = env(val=FIX_DATES[0] + timedelta(days=5), spot=1.20)
    with pytest.warns(MissingFixingWarning) as rec:
        p = _price(d, e=e)
    missing = [w for w in rec if issubclass(w.category, MissingFixingWarning)]
    assert len(missing) == 1 and FIX_DATES[0].date().isoformat() in str(missing[0].message)
    recorded = copy.deepcopy(d)
    recorded.record_passed_fixings(e)
    assert p == _price(recorded, e=e)
    assert d.state == snapshot.state and d.lifecycle_events == snapshot.lifecycle_events
```

- [ ] **Step 3: Run to verify it fails**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_lifecycle.py -n 2 -q`
Expected: the new tests FAIL (`TypeError: unexpected keyword argument 'fixing_dates'`); earlier tests still pass.

- [ ] **Step 4: Implement the dated form in the base**

Add to `target_redemption_base.py` (imports: `from quantark.util.calendar import calculate_year_fraction`):

```python
def signed_year_fraction(date: datetime, fx_env) -> float:
    """Signed year fraction from ``fx_env.valuation_date`` to ``date`` in the ENVIRONMENT's convention: 0.0 on the
    valuation date, negative before it (``FxRangeAccrualObservationRecord.resolve_time``'s rule)."""
    val = fx_env.valuation_date
    if date == val:
        return 0.0
    if date > val:
        return calculate_year_fraction(val, date, fx_env.day_count_convention, fx_env.bus_days_in_year,
                                       calendar=fx_env.calendar)
    return -calculate_year_fraction(date, val, fx_env.day_count_convention, fx_env.bus_days_in_year,
                                    calendar=fx_env.calendar)
```

Methods on `TargetRedemptionProduct` (replace `is_terminal`, `validate`, `_validate_state`, `record_passed_fixings`,
`_has_unbanked_passed`, `_view` with the dispatching versions below; add the rest):

```python
    @staticmethod
    def _schedule_form(fixing_times, fixing_dates) -> bool:
        """True for the dated form; exactly one schedule is required."""
        if fixing_times is not None and fixing_dates is not None:
            raise ValidationError("give fixing_times (numeric form) or fixing_dates (dated form), not both")
        if fixing_times is None and fixing_dates is None:
            raise ValidationError("a schedule is required: fixing_times (numeric form) or fixing_dates (dated form)")
        return fixing_dates is not None

    def _init_dated(self, fixing_dates, pay_dates, currency_pair, state: TargetRedemptionState) -> None:
        BaseFxProduct.__init__(self, currency_pair=currency_pair)
        self.fixing_dates = list(fixing_dates)
        self.pay_dates = list(pay_dates) if pay_dates is not None else list(self.fixing_dates)
        self.fixing_times = None
        self.pay_times = None
        self.state = state
        self.lifecycle_events = ()

    def _refresh_dated_horizon(self) -> None:
        self.expiry_date = self.fixing_dates[-1]
        self.delivery_date = max(self.pay_dates)

    @property
    def is_terminal(self) -> bool:
        if self.is_numeric:
            return not self.fixing_times or self.is_redeemed
        as_of = self.state.as_of
        return self.is_redeemed or (as_of is not None and as_of >= self.fixing_dates[-1])

    def validate(self) -> None:
        self._validate_terms()
        if self.is_numeric:
            self._validate_numeric_schedule()
        else:
            self._validate_dated_schedule()
        self._validate_state()

    def _validate_dated_accrual(self) -> None:
        """TARN hook: the dated period source."""

    def _validate_dated_schedule(self) -> None:
        dates, pays = self.fixing_dates, self.pay_dates
        if not dates:
            raise ValidationError("fixing_dates must be non-empty: the dated form lists the whole schedule")
        if any(not isinstance(d, datetime) for d in list(dates) + list(pays)):
            raise ValidationError("fixing_dates and pay_dates must be datetimes")
        if dates != sorted(dates):
            raise ValidationError("fixing_dates must be ascending")
        if len(set(dates)) != len(dates):
            raise ValidationError("fixing_dates must be unique")
        if len(pays) != len(dates):
            raise ValidationError("pay_dates must align 1:1 with fixing_dates")
        for f, p in zip(dates, pays):
            if p < f:
                raise ValidationError(f"pay date {p.date()} precedes its fixing {f.date()}")
        if pays != sorted(pays):
            raise ValidationError("pay_dates must be non-decreasing")
        self._validate_dated_accrual()

    def _validate_state(self) -> None:
        st = self.state
        if st.accumulated > self.target + self.hit_tol:
            raise ValidationError(f"state.accumulated {st.accumulated} exceeds the target {self.target}")
        if self._pending_must_be_positive and any(p.amount <= 0.0 for p in st.pending):
            raise ValidationError(
                "pending amounts must be positive: a TARN's determined cash is a coupon or its principal"
            )
        if self.is_numeric:
            if st.as_of is not None:
                raise ValidationError(
                    "state.as_of belongs to the dated form: a numeric state lists only unbanked fixings")
            if any(p.pay_date is not None for p in st.pending):
                raise ValidationError("a numeric contract's pending payments take pay_time, not pay_date")
            if self.is_redeemed and self.fixing_times:
                raise ValidationError(
                    "a redeemed state cannot list fixings: every fixing after the target hit is dead")
            return
        if any(p.pay_time is not None for p in st.pending):
            raise ValidationError("a dated contract's pending payments take pay_date, not pay_time")
        if (st.accumulated > 0.0 or st.pending) and (st.as_of is None or st.as_of < self.fixing_dates[0]):
            raise ValidationError(
                "accumulated or pending cash needs state.as_of on or after the first fixing date")

    def _dated_factor(self, index: int, fx_env) -> Optional[float]:
        """TARN hook: the accrual factor of scheduled period ``index``; None for the TARF."""
        return None

    def _is_unbanked(self, index: int) -> bool:
        as_of = self.state.as_of
        return as_of is None or self.fixing_dates[index] > as_of

    def _check_as_of(self, fx_env) -> None:
        as_of = self.state.as_of
        if as_of is not None and as_of > fx_env.valuation_date:
            raise ValidationError(
                f"state.as_of {as_of} is after the valuation date {fx_env.valuation_date}: the state claims "
                "fixings that have not happened")

    def get_maturity(self, fx_env=None) -> float:
        """Dated: 0.0 once valuation is past the last fixing date (an aged contract still answers FX consumers)."""
        if self.expiry_date is not None and fx_env is not None and fx_env.valuation_date > self.expiry_date:
            return 0.0
        return super().get_maturity(fx_env)

    def get_delivery(self, fx_env=None) -> float:
        """Dated: 0.0 once valuation is past the last pay date (Ruling R8: ``calculate_greeks`` reads it)."""
        if self.delivery_date is not None and fx_env is not None and fx_env.valuation_date > self.delivery_date:
            return 0.0
        return super().get_delivery(fx_env)

    def record_passed_fixings(self, fx_env) -> None:
        if self.is_numeric:
            self._record_numeric(fx_env)
        else:
            self._record_dated(fx_env)
        self.validate()

    def _record_dated(self, fx_env) -> None:
        self._check_as_of(fx_env)
        val, spot, st = fx_env.valuation_date, fx_env.spot, self.state
        events: List[TargetRedemptionEvent] = []
        pending: list = []
        for p in st.pending:  # cash due by the valuation date was paid
            if p.pay_date <= val:
                events.append(TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, p.amount,
                                                    time=signed_year_fraction(p.pay_date, fx_env)))
            else:
                pending.append(p)
        acc = st.accumulated
        if not self.is_terminal:
            last = len(self.fixing_dates) - 1
            for i, fd in enumerate(self.fixing_dates):
                if not self._is_unbanked(i):
                    continue
                if fd > val:
                    break
                t = signed_year_fraction(fd, fx_env)
                added, cash = self._fixing_outcome(self._dated_factor(i, fx_env), spot, acc)
                acc = acc + added
                hit = is_target_hit(acc, self.target)
                ends = hit or i == last
                label = event_label(i, date=fd)
                events.append(TargetRedemptionEvent(TargetRedemptionEventType.FIXING, cash, fixing=spot, time=t,
                                                    redeemed=ends, label=label))
                self._settle_dated(cash, self.pay_dates[i], label, fx_env, pending, events)
                if ends:
                    self._settle_dated(self._principal_amount(), self.pay_dates[i], label, fx_env, pending, events)
                    break  # every later fixing is dead (or there is none)
        self.state = TargetRedemptionState(acc, pending=tuple(sorted(pending, key=lambda p: p.pay_date)), as_of=val)
        self.lifecycle_events = self.lifecycle_events + tuple(events)

    @staticmethod
    def _settle_dated(amount, pay_date, label, fx_env, pending: list, events: list) -> None:
        if is_zero(amount):
            return
        if pay_date <= fx_env.valuation_date:
            events.append(TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, amount,
                                                time=signed_year_fraction(pay_date, fx_env), label=label))
        else:
            pending.append(PendingPayment(amount, pay_date=pay_date))

    def _has_unbanked_passed(self, fx_env) -> bool:
        if self.is_numeric:
            return bool(self.fixing_times) and is_passed_time(self.fixing_times[0])
        self._check_as_of(fx_env)
        if self.is_terminal:
            return False
        val = fx_env.valuation_date
        return any(self._is_unbanked(i) and d <= val for i, d in enumerate(self.fixing_dates))

    def _view(self, fx_env) -> ResolvedTargetRedemption:
        if self.is_numeric:
            st = self.state
            return ResolvedTargetRedemption(
                fixing_times=tuple(self.fixing_times), pay_times=tuple(self.pay_times),
                accrual_factors=tuple(self.accrual_factors) if self._has_accrual else None,
                accumulated=st.accumulated, pending=tuple((p.pay_time, p.amount) for p in st.pending),
                redeemed=self.is_redeemed, terminal=self.is_terminal)
        return self._dated_view(fx_env)

    def _dated_view(self, fx_env) -> ResolvedTargetRedemption:
        val, st = fx_env.valuation_date, self.state
        fix_t: List[float] = []
        pay_t: List[float] = []
        factors: List[float] = []
        if not self.is_terminal:
            for i, d in enumerate(self.fixing_dates):
                if not self._is_unbanked(i):
                    continue
                t = signed_year_fraction(d, fx_env)
                if is_passed_time(t):  # Ruling R7: after valuation by date, zero by the environment's day count
                    raise ValidationError(
                        f"fixing date {d.date()} is after the valuation date {val} but resolves to a zero year "
                        f"fraction under the environment's {fx_env.day_count_convention.name} day count")
                fix_t.append(t)
                pay_t.append(signed_year_fraction(self.pay_dates[i], fx_env))
                if self._has_accrual:
                    factors.append(self._dated_factor(i, fx_env))
        return ResolvedTargetRedemption(
            fixing_times=tuple(fix_t), pay_times=tuple(pay_t),
            accrual_factors=tuple(factors) if self._has_accrual else None,
            accumulated=st.accumulated,
            pending=tuple((signed_year_fraction(p.pay_date, fx_env), p.amount) for p in st.pending
                          if p.pay_date > val),
            redeemed=self.is_redeemed, terminal=self.is_terminal)
```

- [ ] **Step 5: Dated constructors**

TARN `__init__` gains keywords `fixing_dates: Optional[List[datetime]] = None, pay_dates: Optional[List[datetime]] =
None, initial_date: Optional[datetime] = None, day_count_convention: Optional[DayCountConvention] = None` (import
`DayCountConvention, calculate_year_fraction` from `quantark.util.calendar`) and replaces the part after
`state = self._require_state(state)` with:

```python
        if self._schedule_form(fixing_times, fixing_dates):
            for name, value in (("pay_times", pay_times), ("accrual_start", accrual_start)):
                if value is not None:
                    raise ValidationError(
                        f"{name} belongs to the numeric form; a dated TARN takes pay_dates and initial_date")
            if initial_date is None and accrual_factors is None:
                raise ValidationError(
                    "a dated TARN derives its periods from initial_date or states explicit accrual_factors")
            if initial_date is not None and accrual_factors is not None:
                raise ValidationError("give initial_date or accrual_factors, not both")
            if accrual_factors is not None and day_count_convention is not None:
                raise ValidationError(
                    "day_count_convention sets the derived accrual factors; it cannot be combined with explicit "
                    "accrual_factors")
            self.initial_date = initial_date
            self.day_count_convention = day_count_convention
            self.accrual_factors = [float(a) for a in accrual_factors] if accrual_factors is not None else None
            self._init_dated(fixing_dates, pay_dates, currency_pair, state)
            self.validate()
            self._refresh_dated_horizon()
        else:
            for name, value in (("pay_dates", pay_dates), ("initial_date", initial_date),
                                ("day_count_convention", day_count_convention)):
                if value is not None:
                    raise ValidationError(
                        f"{name} belongs to the dated form; a numeric TARN takes pay_times and accrual_start")
            self.initial_date = None
            self.day_count_convention = None
            fixings = [float(t) for t in fixing_times]
            self.accrual_factors = self._numeric_accruals(fixings, accrual_start, accrual_factors, state)
            self._init_numeric(fixings, pay_times, currency_pair, state)
            self.validate()
            self._refresh_numeric_horizon()
        self._warn_if_terminal_booking()
```

TARN hooks:

```python
    def _validate_dated_accrual(self) -> None:
        if self.initial_date is not None and not self.initial_date < self.fixing_dates[0]:
            raise ValidationError(
                f"initial_date {self.initial_date} must be before the first fixing date {self.fixing_dates[0]}")
        if self.accrual_factors is not None:
            if len(self.accrual_factors) != len(self.fixing_dates):
                raise ValidationError("accrual_factors must align 1:1 with fixing_dates")
            if any((not math.isfinite(a)) or a <= 0.0 for a in self.accrual_factors):
                raise ValidationError("accrual_factors must be positive and finite")

    def _dated_factor(self, index: int, fx_env) -> float:
        """Explicit factor, else ``initial_date`` / previous fixing date -> this fixing date in the contract day count
        (``None`` = ACT_365), resolved against the environment's calendar (a BUSINESS_DAYS count needs it)."""
        if self.accrual_factors is not None:
            return self.accrual_factors[index]
        start = self.initial_date if index == 0 else self.fixing_dates[index - 1]
        convention = DayCountConvention.ACT_365 if self.day_count_convention is None else self.day_count_convention
        return calculate_year_fraction(start, self.fixing_dates[index], convention, fx_env.bus_days_in_year,
                                       calendar=fx_env.calendar)
```

`full_coupon` on a dated TARN without explicit factors: raise `ValidationError("a dated TARN without explicit
accrual_factors resolves its factors against an environment: use resolve(fx_env).accrual_factors")` when
`self.accrual_factors is None`. `__repr__` uses `len(self.fixing_times if self.is_numeric else self.fixing_dates)`.

TARF `__init__` gains `fixing_dates=None, pay_dates=None` keywords; after `_require_state`:

```python
        if self._schedule_form(fixing_times, fixing_dates):
            if pay_times is not None:
                raise ValidationError("pay_times belongs to the numeric form; a dated TARF takes pay_dates")
            self._init_dated(fixing_dates, pay_dates, currency_pair, state)
            self.validate()
            self._refresh_dated_horizon()
        else:
            if pay_dates is not None:
                raise ValidationError("pay_dates belongs to the dated form; a numeric TARF takes pay_times")
            self._init_numeric([float(t) for t in fixing_times], pay_times, currency_pair, state)
            self.validate()
            self._refresh_numeric_horizon()
        self._warn_if_terminal_booking()
```
(`__repr__` as for the TARN.) Remove the Task 3/4 `"fixing_times is required"` checks (replaced by `_schedule_form`).

Append to the CHANGELOG `### Added`:

```markdown
- Dated `FxTargetRedemptionNote` / `FxTargetRedemptionForward`: `fixing_dates` (the whole contractual schedule),
  `pay_dates`, `state.as_of`; the TARN derives its periods from `initial_date` (or explicit `accrual_factors`) in its
  `day_count_convention` (default ACT/365). Times resolve in the environment's convention; a numeric contract and its
  dated twin price identically.
- `resolve(fx_env)` (the engines' one input), `lifecycle_events`, `is_redeemed`, `is_terminal`, `hit_tol`,
  `pending_pv(fx_env)` on both products.
```

- [ ] **Step 6: Run to verify it passes**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_lifecycle.py test/test_fx_target_redemption_state.py test/test_fx_tarn_note_mc.py test/test_fx_tarf_mc.py -n 2 -q`
Expected: all pass. If `test_d4_*` fails by more than R5's bound, STOP and report the measured gap (do not loosen it).

- [ ] **Step 7: Commit**

```bash
git -C <WT> add quantark/asset/fx/product/option/target_redemption_base.py quantark/asset/fx/product/option/fx_target_redemption_note.py quantark/asset/fx/product/option/fx_target_redemption_forward.py test/fx_target_redemption_fixtures.py test/test_fx_target_redemption_lifecycle.py CHANGELOG.md
git -C <WT> commit -m "feat(fx-tarn): dated TARN/TARF - whole schedule, as_of state, env-resolved times, twin parity

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Q7 — numeric FX theta steps by the environment's year fraction

Spec §6 (last bullet), §12 Q7, §10 D4.

**Files:**
- Create: `docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/q7_theta_plugin.py`, `q7_summarize.py`, `moved_values.txt`
- Modify: `quantark/asset/fx/engine/base_fx_engine.py:219-241`, `CHANGELOG.md`
- Test: `test/test_fx_target_redemption_lifecycle.py`, `test/test_fx_base_engine.py`

**Interfaces:**
- Consumes: Tasks 2–7 (dated twins for D4).
- Produces: `BaseFxEngine._fdm_theta` numeric step `calculate_year_fraction(V, V + timedelta(days=theta_days), env.day_count_convention, env.bus_days_in_year, calendar=env.calendar)`.

- [ ] **Step 1: Write the measurement plugin**

`q7_theta_plugin.py`:

```python
"""Q7 measurement (plan 2026-10-09-fx-tarn-lifecycle, Task 8): for every BaseFxEngine._fdm_theta call on a NUMERIC FX
product, price theta with the current step (theta_days / 365) and with the Q7 step (the environment's year fraction of
theta_days calendar days) and log every call where the two steps differ. Equal steps give bitwise-equal theta (same
code path), so only differing steps are logged.

Controller run (full suite, before or after the Q7 change - both thetas are computed explicitly):
  env PYTHONPATH=<WT>:<WT>/test:<WT>/docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence \
      Q7_LOG_DIR=<WT>/.superpowers/w4b/q7 /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test \
      -p q7_theta_plugin -n 6 -q
"""
import json
import os
from copy import deepcopy
from datetime import timedelta

import pytest

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine
from quantark.util.calendar import calculate_year_fraction

_LOG_DIR = os.environ.get("Q7_LOG_DIR")
_ORIG = BaseFxEngine._fdm_theta
_CURRENT = {"nodeid": "<collection>"}


def _theta_with_step(engine, product, fx_env, base_price, dt):
    shifted = deepcopy(product)
    shifted.time_shift(dt)
    shifted.record_passed_fixings(fx_env)
    return (engine.price(shifted, fx_env) - base_price) / engine.params.theta_days


def _spy(self, product, fx_env, base_price):
    result = _ORIG(self, product, fx_env, base_price)
    if _LOG_DIR and product.maturity is not None:
        days = self.params.theta_days
        val = fx_env.valuation_date
        old_dt = days / 365.0
        new_dt = calculate_year_fraction(val, val + timedelta(days=days), fx_env.day_count_convention,
                                         fx_env.bus_days_in_year, calendar=fx_env.calendar)
        if new_dt != old_dt:
            record = {
                "nodeid": _CURRENT["nodeid"], "engine": type(self).__name__, "product": type(product).__name__,
                "convention": fx_env.day_count_convention.name, "valuation_date": str(val),
                "theta_days": days, "old_dt": old_dt, "new_dt": new_dt,
                "old_theta": _theta_with_step(self, product, fx_env, base_price, old_dt),
                "new_theta": _theta_with_step(self, product, fx_env, base_price, new_dt),
            }
            os.makedirs(_LOG_DIR, exist_ok=True)
            with open(os.path.join(_LOG_DIR, f"{os.getpid()}.jsonl"), "a") as fh:
                fh.write(json.dumps(record) + "\n")
    return result


BaseFxEngine._fdm_theta = _spy


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    _CURRENT["nodeid"] = item.nodeid
```

`q7_summarize.py`:

```python
"""Summarize the Q7 plugin logs into moved_values.txt lines (one per distinct test / engine / product / step)."""
import glob
import json
import os
import sys

rows = {}
for path in glob.glob(os.path.join(sys.argv[1], "*.jsonl")):
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            key = (r["nodeid"], r["engine"], r["product"], r["convention"], r["valuation_date"], r["new_dt"])
            rows[key] = r
print(f"# Q7 numeric FX theta moves: {len(rows)} distinct calls (old step theta_days/365 -> env year fraction)")
for key in sorted(rows):
    r = rows[key]
    print(f"Q7 | {r['nodeid']} | {r['engine']} / {r['product']} | {r['convention']} @ {r['valuation_date']} "
          f"| dt {r['old_dt']!r} -> {r['new_dt']!r} | theta {r['old_theta']!r} -> {r['new_theta']!r}")
```

- [ ] **Step 2: Measure (controller step, full suite at `-n 6`)**

Run the plugin command from the docstring, then:
```bash
env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python <WT>/docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/q7_summarize.py <WT>/.superpowers/w4b/q7 > <WT>/.superpowers/w4b/q7_moves.txt
git -C <WT> checkout -- example/mo_volmodels/data/mo_barrier_sample.json example/mo_volmodels/data/mo_calibration_explainer_sample.html
```
Expected: a list (possibly empty) of numeric FX thetas under non-calendar environments. Create `moved_values.txt`:

```text
# W4b (FX TARN / TARF lifecycle) - existing values that move, OLD -> NEW, for user sign-off BEFORE landing.
# Format: source | site | quantity | old -> new | reason
#
# Q7 (spec section 12): numeric FX theta (BaseFxEngine._fdm_theta) steps by the environment's year fraction.
# Measured by q7_theta_plugin.py over the full suite on <branch sha>:
<paste q7_moves.txt here>
#
# Q7 scope note (Ruling R9): localvol_greeks.py:60 and the Heston / Heston-SLV PDE thetas keep theta_days/365
# (not BaseFxEngine._fdm_theta); their numeric/dated gap under non-calendar envs is a follow-up, not changed here.
# Non-integral theta_days: the numeric step becomes the env year fraction of a partial day (0.0 under CALENDAR_DAYS),
# as the dated branch already computes; no test in the suite sets theta_days (grep: none).
```

- [ ] **Step 3: Write the failing tests**

Append to `test/test_fx_base_engine.py` (imports: `from datetime import datetime, timedelta`,
`from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine`, `FxVanillaOption`, `CurrencyPair`,
`OptionType`, `FlatRateCurve`, `FlatVolSurface`, `SpotQuote`, `FxPricingEnvironment`,
`from quantark.util.calendar.day_counter import DayCountConvention as DC`,
`from quantark.util.calendar.business_calendar import CalendarType, create_calendar`):

```python
_FRIDAY = datetime(2026, 6, 12)


def _q7_env(dc, cal=None):
    return FxPricingEnvironment(valuation_date=_FRIDAY, spot_quote=SpotQuote(spot=1.20),
                                domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.03),
                                vol_surface=FlatVolSurface(volatility=0.10), day_count_convention=dc, calendar=cal)


def _q7_pair(e):
    dated = FxVanillaOption(strike=1.2, option_type=OptionType.CALL, expiry_date=_FRIDAY + timedelta(days=365),
                            currency_pair=CurrencyPair("EUR", "USD"), notional_foreign=1.0)
    numeric = FxVanillaOption(strike=1.2, option_type=OptionType.CALL, maturity=dated.get_maturity(e),
                              currency_pair=CurrencyPair("EUR", "USD"), notional_foreign=1.0)
    return numeric, dated


def test_q7_business_days_numeric_theta_matches_dated():  # M9: was -9.135e-05 vs 0.0
    e = _q7_env(DC.BUSINESS_DAYS, create_calendar(CalendarType.TARGET, year_range=(2026, 2028)))
    numeric, dated = _q7_pair(e)
    eng = GarmanKohlhagenEngine()
    tn = eng._fdm_theta(numeric, e, eng.price(numeric, e))
    td = eng._fdm_theta(dated, e, eng.price(dated, e))
    assert tn == td == 0.0  # Friday -> Saturday adds no business day


def test_q7_calendar_days_step_is_unchanged_bitwise():
    e = _q7_env(DC.CALENDAR_DAYS)
    numeric, _ = _q7_pair(e)
    eng = GarmanKohlhagenEngine()
    base = eng.price(numeric, e)
    manual = FxVanillaOption(strike=1.2, option_type=OptionType.CALL, maturity=numeric.maturity - 1.0 / 365.0,
                             currency_pair=CurrencyPair("EUR", "USD"), notional_foreign=1.0)
    assert eng._fdm_theta(numeric, e, base) == (eng.price(manual, e) - base) / 1.0


def test_q7_act_360_steps_one_360th():
    e = _q7_env(DC.ACT_360)
    numeric, _ = _q7_pair(e)
    eng = GarmanKohlhagenEngine()
    base = eng.price(numeric, e)
    manual = FxVanillaOption(strike=1.2, option_type=OptionType.CALL, maturity=numeric.maturity - 1.0 / 360.0,
                             currency_pair=CurrencyPair("EUR", "USD"), notional_foreign=1.0)
    assert eng._fdm_theta(numeric, e, base) == (eng.price(manual, e) - base) / 1.0
```

Append to `test/test_fx_target_redemption_lifecycle.py`:

```python
@pytest.mark.parametrize("kind", ["tarn", "tarf"])
@pytest.mark.parametrize("val", [datetime(2026, 6, 12), datetime(2026, 6, 15)])  # Friday, Monday
def test_d4_theta_twins_agree_business_days(kind, val):
    e = env(val=val, **BUS)
    dated = _dated(kind, dates=BUS_DATES, **({"initial_date": val} if kind != "tarf" else {}))
    twin = _numeric_twin(dated, e)
    gd = engine_for(dated).calculate_greeks(dated, e)
    gn = engine_for(twin).calculate_greeks(twin, e)
    assert abs(gd["theta"] - gn["theta"]) <= 1e-12 * abs(gd["price"])  # Ruling R5
```

- [ ] **Step 4: Run to verify they fail**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_base_engine.py test/test_fx_target_redemption_lifecycle.py -k "q7 or business_days" -n 2 -q`
Expected: `test_q7_business_days_*`, `test_q7_act_360_*` and `test_d4_*business_days[*Friday*]` FAIL (numeric
step is still 1/365); `test_q7_calendar_days_*` passes.

- [ ] **Step 5: Implement**

In `base_fx_engine.py` (imports: `from datetime import timedelta`, `from quantark.util.calendar import
calculate_year_fraction`; drop the local `from datetime import timedelta`):

```python
    def _fdm_theta(
        self,
        product: BaseFxProduct,
        fx_env: FxPricingEnvironment,
        base_price: float,
    ) -> float:
        days = self.params.theta_days
        future_date = fx_env.valuation_date + timedelta(days=days)

        shifted = deepcopy(product)
        if product.maturity is not None:
            # Q7 (user decision 2026-10-09): a numeric product ages by the environment's year fraction of the bump,
            # the step its dated twin's times take when the env date moves; days/365 bitwise under CALENDAR_DAYS /
            # ACT_365.
            shifted.time_shift(calculate_year_fraction(
                fx_env.valuation_date, future_date, fx_env.day_count_convention, fx_env.bus_days_in_year,
                calendar=fx_env.calendar))
            env_future = fx_env
        else:
            env_future = deepcopy(fx_env)
            env_future.valuation_date = future_date
        shifted.record_passed_fixings(env_future)
        future_price = self.price(shifted, env_future)

        # Forward difference: daily decay
        return (future_price - base_price) / days
```

Append to the CHANGELOG `### Changed`:

```markdown
- FX finite-difference theta (`BaseFxEngine._fdm_theta`) steps a numeric (`maturity=`) product by the environment's
  year fraction of `theta_days` calendar days instead of `theta_days / 365` (user decision Q7). Unchanged to the bit
  under `CALENDAR_DAYS` / `ACT_365`; under `BUSINESS_DAYS` a Friday or holiday theta is 0.0 and `ACT_360` steps 1/360,
  matching the dated product. Values that move are listed in
  `docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/moved_values.txt`.
```

- [ ] **Step 6: Run the FX gates and record moved expectations**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_base_engine.py test/test_fx_target_redemption_lifecycle.py test/test_fx_tarn_note_mc.py test/test_fx_tarf_mc.py test/test_fx_range_accrual.py test/test_fx_range_accrual_mc.py test/test_fx_greeks_analytical_vs_fdm.py test/test_range_accrual_contract_period.py -n 2 -q`
Expected: all pass. Any existing assertion that fails ONLY because of the Q7 step: do not change the code; update
the assertion to the new value with a comment `# Q7 (2026-10-09): env year-fraction theta step - pending sign-off`,
and add `Q7 | <nodeid> | <quantity> | <old> -> <new> | numeric theta step` to `moved_values.txt`. Report every such
edit to the controller.

- [ ] **Step 7: Commit**

```bash
git -C <WT> add quantark/asset/fx/engine/base_fx_engine.py test/test_fx_base_engine.py test/test_fx_target_redemption_lifecycle.py CHANGELOG.md
git -C <WT> add -f docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/q7_theta_plugin.py docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/q7_summarize.py docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/moved_values.txt
git -C <WT> commit -m "feat(fx): numeric FX theta steps by the environment's year fraction (Q7)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
(plus any test files edited in Step 6, by path).

---

### Task 9: FX lifecycle hook — `register` / `advance` (D2, B5 hook-level)

Spec §7.2, §10 D2, B5.

**Files:**
- Create: `quantark/asset/fx/lifecycle/__init__.py`, `quantark/asset/fx/lifecycle/target_redemption.py`, `test/test_fx_target_redemption_runs.py`
- Modify: `target_redemption_base.py` (add `matured_before`), `test/fx_target_redemption_fixtures.py` (add `drive`, `book`), `test/test_fx_target_redemption_lifecycle.py` (D2), `CHANGELOG.md`

**Interfaces:**
- Consumes: Tasks 2–8; W1 `MaturedBeforeRunWarning`; W5 `warn_filled_fixings`.
- Produces:
  - `TargetRedemptionProduct.matured_before(fx_env) -> bool` (not terminal and the last fixing is strictly before valuation: numeric `t < 0 and not is_zero(t)`, dated `date < valuation_date`).
  - `quantark.asset.fx.lifecycle`: `is_ageable(product) -> bool`; `FxLifecycleEntry(position_id: str, quantity: float, event: TargetRedemptionEvent)`; `FxLifecycleDay(date, paid_cash: float, events: Tuple[FxLifecycleEntry, ...], pending_pv: float)`; `FxTargetRedemptionLifecycle()` with `holds_ageable(portfolio) -> bool` (static), `register(portfolio) -> None`, `advance(portfolio, date) -> FxLifecycleDay`.
  - Fixtures: `drive(portfolio, lifecycle, date, spot, pair="EURUSD") -> FxLifecycleDay`; `book(*products, quantities=None, n=2_000) -> (FXPortfolio, List[str])`.

- [ ] **Step 1: Fixtures and failing tests**

Append to `test/fx_target_redemption_fixtures.py`:

```python
import dataclasses  # (move the appended imports to the top of the module)

from quantark.portfolio.fx import FXPortfolio


def drive(portfolio, lifecycle, date, spot, pair="EURUSD"):
    """Move ``pair``'s environment to ``date`` at ``spot`` and age the book (spec 7.2 ``advance``)."""
    e = portfolio.pricing_environments[pair]
    portfolio.pricing_environments[pair] = dataclasses.replace(e, valuation_date=date,
                                                               spot_quote=SpotQuote(spot=spot))
    return lifecycle.advance(portfolio, date)


def book(*products, quantities=None, n=2_000, fx_env=None):
    pf = FXPortfolio(portfolio_name="w4b", pricing_environments={"EURUSD": fx_env or env()})
    ids = []
    for i, prod in enumerate(products):
        q = 1.0 if quantities is None else quantities[i]
        ids.append(pf.add_position(product=prod, quantity=q, entry_price=0.0, underlying="EURUSD",
                                   engine=engine_for(prod, n=n)).position_id)
    return pf, ids
```

`test/test_fx_target_redemption_runs.py`:

```python
"""FX TARN / TARF ageing inside runs: the lifecycle hook, the FX backtest, the FX dynamic scenario (spec
2026-10-09-fx-tarn-lifecycle-design.md section 7, tests B1-B6, Q8)."""
import dataclasses
import warnings
from datetime import datetime, timedelta

import pytest

from fx_target_redemption_fixtures import (
    FIX, FIX_DATES, INCEPTION, LAG2, PAIR, VAL, book, dated_note, drive, engine_for, env, note, tarf,
)
from quantark.asset.equity.lifecycle.autocallable import MaturedBeforeRunWarning
from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
from quantark.asset.fx.lifecycle import FxLifecycleDay, FxTargetRedemptionLifecycle
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxVanillaOption
from quantark.asset.fx.product.option.target_redemption_state import (
    PendingPayment, TargetRedemptionEventType, TargetRedemptionState, TerminalBookingWarning,
)
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError
from quantark.util.fixings import MissingFixingWarning

FIXING, PAYMENT = TargetRedemptionEventType.FIXING, TargetRedemptionEventType.PAYMENT


def test_register_banks_fixings_before_the_run_with_one_warning():
    prod = note(fix=[-10 / 365, -5 / 365] + FIX[1:], accrual_start=-40 / 365, target=50.0)
    expected = prod.full_coupon(0) + prod.full_coupon(1)
    pf, (pid,) = book(prod)
    life = FxTargetRedemptionLifecycle()
    with pytest.warns(MissingFixingWarning) as rec:
        life.register(pf)
    msgs = [str(w.message) for w in rec if issubclass(w.category, MissingFixingWarning)]
    assert len(msgs) == 1 and "#0" in msgs[0] and "#1" in msgs[0]
    assert pf.positions[pid].product.state.accumulated == expected
    day = life.advance(pf, VAL)
    assert day.paid_cash == 0.0 and day.events == ()  # their payments were before the run: never booked


def test_fixing_on_the_origin_date_banks_without_a_warning():
    prod = dated_note(fixing_dates=[VAL] + FIX_DATES[1:], initial_date=VAL - timedelta(days=30), target=50.0)
    pf, (pid,) = book(prod)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        FxTargetRedemptionLifecycle().register(pf)
    st = pf.positions[pid].product.state
    assert st.as_of == VAL and st.accumulated > 0.0


@pytest.mark.parametrize("make", [
    lambda: note(fix=[-60 / 365, -30 / 365], accrual_start=-90 / 365),
    lambda: dated_note(fixing_dates=[VAL - timedelta(days=60), VAL - timedelta(days=30)],
                       initial_date=VAL - timedelta(days=90)),
])
def test_matured_before_the_run_is_removed_with_a_warning(make):
    pf, (pid,) = book(make())
    life = FxTargetRedemptionLifecycle()
    with pytest.warns(MaturedBeforeRunWarning):
        life.register(pf)
    assert pid not in pf.positions
    assert life.advance(pf, VAL) == FxLifecycleDay(date=VAL, paid_cash=0.0, events=(), pending_pv=0.0)


def test_terminal_booking_is_kept_at_its_pending_pv():
    with pytest.warns(TerminalBookingWarning):
        owed = note(fix=[], accrual_factors=[], target=1.5,
                    state=TargetRedemptionState(1.5, pending=(PendingPayment(100.5, pay_time=10 / 365),)))
    pf, (pid,) = book(owed)
    life = FxTargetRedemptionLifecycle()
    life.register(pf)
    e = pf.pricing_environments["EURUSD"]
    assert life.advance(pf, VAL).pending_pv == 0.0 + 100.5 * e.get_domestic_df(10 / 365)
    paid = [drive(pf, life, VAL + timedelta(days=k), 1.20).paid_cash for k in range(1, 12)]
    assert paid[9] == 100.5 and sum(p != 0.0 for p in paid) == 1  # booked on day 10 only


def test_clock_cannot_go_back():
    pf, _ = book(note(target=3.0))
    life = FxTargetRedemptionLifecycle()
    life.register(pf)
    drive(pf, life, VAL + timedelta(days=2), 1.20)
    with pytest.raises(ValidationError, match="cannot go back"):
        drive(pf, life, VAL + timedelta(days=1), 1.20)


def test_advance_needs_the_environment_on_that_date():
    pf, _ = book(note(target=3.0))
    life = FxTargetRedemptionLifecycle()
    life.register(pf)
    with pytest.raises(ValidationError, match="environment"):
        life.advance(pf, VAL + timedelta(days=1))


def test_other_products_are_untouched():
    vanilla = FxVanillaOption(strike=1.2, option_type=OptionType.CALL, maturity=1.0, notional_foreign=1.0,
                              currency_pair=PAIR)
    pf, _ = book(note(target=3.0))
    pid = pf.add_position(product=vanilla, quantity=1.0, entry_price=0.0, underlying="EURUSD",
                          engine=GarmanKohlhagenEngine()).position_id
    life = FxTargetRedemptionLifecycle()
    life.register(pf)
    drive(pf, life, VAL + timedelta(days=40), 1.20)
    assert pf.positions[pid].product.maturity == 1.0


def test_calendar_gap_banks_in_order_without_a_warning():
    prod = note(target=50.0)
    c = [prod.full_coupon(0), prod.full_coupon(1)]
    pf, (pid,) = book(prod)
    life = FxTargetRedemptionLifecycle()
    life.register(pf)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        day = drive(pf, life, VAL + timedelta(days=70), 1.20)  # k/12 fixings 1 and 2 fall inside the gap
    kinds = [(e.event.event_type, e.event.amount) for e in day.events]
    assert kinds == [(FIXING, c[0]), (PAYMENT, c[0]), (FIXING, c[1]), (PAYMENT, c[1])]
    assert day.paid_cash == (0.0 + 1.0 * c[0]) + 1.0 * c[1]


def test_short_position_books_signed_cash():
    lag = [t + 0.02 for t in FIX]
    prod = tarf(gearing=2.0, target=0.2, pay_times=lag)
    pf, (pid,) = book(prod, quantities=[-2.0])
    life = FxTargetRedemptionLifecycle()
    life.register(pf)
    fixing_day = drive(pf, life, VAL + timedelta(days=31), 1.10)  # fixing 1 banks; cash pending
    cf = pf.positions[pid].product.state.pending[0].amount
    assert cf < 0.0
    e = pf.pricing_environments["EURUSD"]
    assert fixing_day.pending_pv == 0.0 + -2.0 * (0.0 + cf * e.get_domestic_df(
        pf.positions[pid].product.state.pending[0].pay_time))
    pay_day = drive(pf, life, VAL + timedelta(days=38), 1.10)
    assert pay_day.paid_cash == 0.0 + -2.0 * cf and pay_day.paid_cash > 0.0


def test_two_pairs_age_independently():
    jpy_tarf = tarf(strike=145.0, currency_pair=CurrencyPair("USD", "JPY"), target=1e9)
    pf, (pid_eur,) = book(note(target=50.0))
    pf.pricing_environments["USDJPY"] = env(spot=150.0)
    pid_jpy = pf.add_position(product=jpy_tarf, quantity=1.0, entry_price=0.0, underlying="USDJPY",
                              engine=engine_for(jpy_tarf, n=2_000)).position_id
    life = FxTargetRedemptionLifecycle()
    life.register(pf)
    pf.pricing_environments["USDJPY"] = dataclasses.replace(pf.pricing_environments["USDJPY"],
                                                            valuation_date=VAL + timedelta(days=31))
    day = drive(pf, life, VAL + timedelta(days=31), 1.05)  # EUR below the TARN strike: no coupon
    by_pid = {e.position_id: e.event for e in day.events if e.event.event_type is FIXING}
    assert by_pid[pid_eur].fixing == 1.05 and by_pid[pid_eur].amount == 0.0
    assert by_pid[pid_jpy].fixing == 150.0 and by_pid[pid_jpy].amount == 1.0 * max(150.0 - 145.0, 0.0)
```

Append the D2 test to `test/test_fx_target_redemption_lifecycle.py` (imports: `from fx_target_redemption_fixtures
import book, drive`, `from quantark.asset.fx.lifecycle import FxTargetRedemptionLifecycle`):

```python
_D2_SPOT = {("tarn", "quiet"): 1.05, ("tarn", "redeeming"): 1.20, ("tarn_no_principal", "quiet"): 1.05,
            ("tarn_no_principal", "redeeming"): 1.20, ("tarf", "quiet"): 1.15, ("tarf", "redeeming"): 1.20}


@pytest.mark.parametrize("path", ["quiet", "redeeming"])
@pytest.mark.parametrize("kind", ["tarn", "tarn_no_principal", "tarf"])
def test_d2_twins_age_identically(kind, path):
    spot = _D2_SPOT[(kind, path)]
    if kind == "tarf":
        dated = dated_tarf(pay_dates=LAG2, target=0.075, gearing=2.0)
    else:
        dated = dated_note(pay_dates=LAG2, target=1.5, include_principal=(kind == "tarn"))
    twin = _numeric_twin(dated, env())
    pf, (pid_d, pid_n) = book(dated, twin, n=4_000, fx_env=env(spot=spot))
    life = FxTargetRedemptionLifecycle()
    life.register(pf)
    checkpoints = {FIX_DATES[1] - timedelta(days=1), FIX_DATES[1], FIX_DATES[1] + timedelta(days=1),
                   VAL + timedelta(days=91), LAG2[1] + timedelta(days=1), LAG2[-1] + timedelta(days=1)}
    log = {pid_d: [], pid_n: []}
    day = VAL
    while day < LAG2[-1] + timedelta(days=2):
        day += timedelta(days=1)
        result = drive(pf, life, day, spot)
        for pid in log:
            log[pid].extend((day, x.event.event_type, x.event.amount, x.event.fixing, x.event.redeemed)
                            for x in result.events if x.position_id == pid)
        if day in checkpoints:
            e = pf.pricing_environments["EURUSD"]
            a, b = pf.positions[pid_d], pf.positions[pid_n]
            assert a.product.state.accumulated == b.product.state.accumulated
            assert [p.amount for p in a.product.state.pending] == [p.amount for p in b.product.state.pending]
            assert is_close(a.get_market_value(e), b.get_market_value(e), rel_tol=1e-12)
    assert log[pid_d] and log[pid_d] == log[pid_n]
```

- [ ] **Step 2: Run to verify they fail**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_runs.py test/test_fx_target_redemption_lifecycle.py -k "d2 or register or matured or terminal_booking_is_kept or clock or advance or untouched or gap or short or pairs" -n 2 -q`
Expected: collection error `ModuleNotFoundError: No module named 'quantark.asset.fx.lifecycle'`.

- [ ] **Step 3: Implement**

Add to `TargetRedemptionProduct`:

```python
    def matured_before(self, fx_env) -> bool:
        """W1 rule: the contract is not terminal and its last fixing is strictly before ``fx_env``'s valuation, so
        the state has not banked it (numeric: last listed time < 0 and not zero; dated: last fixing date before)."""
        if self.is_terminal:
            return False
        if self.is_numeric:
            last = self.fixing_times[-1]
            return last < 0.0 and not is_zero(last)
        return self.fixing_dates[-1] < fx_env.valuation_date
```

`quantark/asset/fx/lifecycle/__init__.py`:

```python
"""Lifecycle of FX products inside runs (FX backtest, FX dynamic scenario)."""
from .target_redemption import FxLifecycleDay, FxLifecycleEntry, FxTargetRedemptionLifecycle, is_ageable

__all__ = ["FxLifecycleDay", "FxLifecycleEntry", "FxTargetRedemptionLifecycle", "is_ageable"]
```

`quantark/asset/fx/lifecycle/target_redemption.py`:

```python
"""Day-by-day ageing of FX TARN / TARF positions inside the FX backtest and FX dynamic scenario.

Spec: docs/superpowers/specs/2026-10-09-fx-tarn-lifecycle-design.md, section 7.2. ``register`` runs once on the run's
working copy before the baseline is valued; ``advance`` runs every simulated day after the day's market is applied
and the environment carries the day's date. Every other position is left exactly as it is.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

from quantark.asset.equity.lifecycle.autocallable import MaturedBeforeRunWarning
from quantark.asset.fx.product.option.fx_target_redemption_forward import FxTargetRedemptionForward
from quantark.asset.fx.product.option.fx_target_redemption_note import FxTargetRedemptionNote
from quantark.asset.fx.product.option.target_redemption_state import (
    TargetRedemptionEvent,
    TargetRedemptionEventType,
)
from quantark.util.calendar import calculate_year_fraction
from quantark.util.exceptions import ValidationError
from quantark.util.fixings import warn_filled_fixings
from quantark.util.numerical import is_zero

_AGEABLE = (FxTargetRedemptionNote, FxTargetRedemptionForward)


def is_ageable(product) -> bool:
    """True for the products this hook ages (FX TARN / TARF)."""
    return isinstance(product, _AGEABLE)


@dataclass(frozen=True)
class FxLifecycleEntry:
    """One product event of one position on one day."""

    position_id: str
    quantity: float
    event: TargetRedemptionEvent


@dataclass(frozen=True)
class FxLifecycleDay:
    """What ageing did on ``date``: quantity-scaled cash paid today, the events, and the pending cash's PV."""

    date: datetime
    paid_cash: float
    events: Tuple[FxLifecycleEntry, ...]
    pending_pv: float


class FxTargetRedemptionLifecycle:
    """Registers and ages the TARN / TARF positions of an FX portfolio (spec 7.2)."""

    def __init__(self) -> None:
        self._clock: Dict[str, datetime] = {}

    @staticmethod
    def holds_ageable(portfolio) -> bool:
        return any(is_ageable(p.product) for p in portfolio.positions.values())

    def register(self, portfolio) -> None:
        """Clock origin = the pair environment's valuation date now (Ruling R3). A contract whose last fixing is
        before it and unbanked warns ``MaturedBeforeRunWarning`` and leaves the working copy (W1). Fixings strictly
        before it are banked at its spot with one ``MissingFixingWarning`` per position (W5); a fixing ON it is the
        run's own fixing and banks silently. Payments this produces were before the baseline: never booked."""
        for pid, position in list(portfolio.positions.items()):
            product = position.product
            if not is_ageable(product):
                continue
            fx_env = portfolio.pricing_environments[position.underlying]
            origin = fx_env.valuation_date
            if product.matured_before(fx_env):
                warnings.warn(
                    MaturedBeforeRunWarning(
                        f"position {pid} ({type(product).__name__}): its last fixing is before the run start "
                        f"{origin} and is not banked; it is not settled in the run, books no cash and is valued at "
                        "zero"),
                    stacklevel=2,
                )
                portfolio.remove_position(pid)
                continue
            start = len(product.lifecycle_events)
            product.record_passed_fixings(fx_env)
            before_run = [e.label for e in product.lifecycle_events[start:]
                          if e.event_type is TargetRedemptionEventType.FIXING
                          and e.time < 0.0 and not is_zero(e.time)]
            warn_filled_fixings(type(product).__name__, "a banked outcome", before_run, stacklevel=3)
            self._clock[pid] = origin

    def advance(self, portfolio, date) -> FxLifecycleDay:
        """Age every registered position to ``date``: numeric ``time_shift`` by the environment's year fraction since
        its last date, then ``record_passed_fixings`` at the day's spot; book the day's PAYMENT events."""
        paid = 0.0
        pending_pv = 0.0
        entries: List[FxLifecycleEntry] = []
        for pid, prev in list(self._clock.items()):
            position = portfolio.positions.get(pid)
            if position is None:
                raise ValidationError(f"position {pid} left the portfolio after registration")
            fx_env = portfolio.pricing_environments[position.underlying]
            if date < prev:
                raise ValidationError(
                    f"the lifecycle clock cannot go back: position {pid} was aged to {prev}, asked for {date}")
            if fx_env.valuation_date != date:
                raise ValidationError(
                    f"advance({date}) needs the {position.underlying} environment on that date, it is on "
                    f"{fx_env.valuation_date}")
            product = position.product
            start = len(product.lifecycle_events)
            if product.is_numeric and date != prev:
                dt = calculate_year_fraction(prev, date, fx_env.day_count_convention, fx_env.bus_days_in_year,
                                             calendar=fx_env.calendar)
                if not is_zero(dt):
                    product.time_shift(dt)
            product.record_passed_fixings(fx_env)
            for event in product.lifecycle_events[start:]:
                entries.append(FxLifecycleEntry(pid, position.quantity, event))
                if event.event_type is TargetRedemptionEventType.PAYMENT:
                    paid += position.quantity * event.amount
            pending_pv += position.quantity * product.pending_pv(fx_env)
            self._clock[pid] = date
        return FxLifecycleDay(date=date, paid_cash=paid, events=tuple(entries), pending_pv=pending_pv)
```

Append to the CHANGELOG `### Added`:

```markdown
- `quantark.asset.fx.lifecycle.FxTargetRedemptionLifecycle`: registers TARN / TARF positions at a run's start
  (W1 matured-before-run rule, W5 missing-fixing rule) and ages them each simulated day (numeric `time_shift` by the
  environment's year fraction, then `record_passed_fixings` at the day's spot), returning the day's paid cash, events
  and pending PV.
```

- [ ] **Step 4: Run to verify they pass**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_runs.py test/test_fx_target_redemption_lifecycle.py -n 2 -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/asset/fx/lifecycle/__init__.py quantark/asset/fx/lifecycle/target_redemption.py quantark/asset/fx/product/option/target_redemption_base.py test/fx_target_redemption_fixtures.py test/test_fx_target_redemption_runs.py test/test_fx_target_redemption_lifecycle.py CHANGELOG.md
git -C <WT> commit -m "feat(fx-tarn): FxTargetRedemptionLifecycle registers and ages TARN/TARF positions day by day

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: FX backtest ages TARN / TARF (B1–B5 backtest, B6 invariants)

Spec §7.1, §7.3 (FX backtest), §10 B1–B6.

**Files:**
- Modify: `quantark/backtest/fx/engine.py:29-122`, `CHANGELOG.md`
- Test: `test/test_fx_target_redemption_runs.py`

**Interfaces:**
- Consumes: `FxTargetRedemptionLifecycle` (Task 9).
- Produces: backtest rows gain `paid_cash` (cumulative, quantity-scaled) and `pending_pv`; `portfolio_value` = Σ position MTM + cumulative paid cash; `final_value` likewise.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_fx_target_redemption_runs.py` (imports: `dataclasses`, `pandas as pd`,
`from quantark.backtest.fx import FXBacktestConfig, FXBacktestEngine`,
`from quantark.backtest.strategy.fx_delta_neutral_strategy import FXDeltaNeutralStrategy`,
`from quantark.asset.fx.engine.mc import FxTargetRedemptionNoteMCEngine`, `from quantark.util.numerical import is_close`,
`from fx_target_redemption_fixtures import _numeric_twin, dated_tarf, engine_for, following, mc`):

```python
def _flat_path(days, spot, start=VAL, freq="D"):
    idx = pd.date_range(start, periods=days, freq=freq)
    return pd.DataFrame({"EURUSD_spot": spot, "EURUSD_vol": 0.10, "EURUSD_dom_rate": 0.05,
                         "EURUSD_for_rate": 0.03}, index=idx)


def _backtest(pf, path):
    return FXBacktestEngine(FXBacktestConfig(portfolio=pf, market_path=path, strategy=FXDeltaNeutralStrategy(),
                                             calculate_greeks=False)).run()


LAG2D = [t + 2 / 365 for t in FIX]


def test_b1_numeric_tarn_ages_in_the_backtest():
    c0 = note(target=3.0)
    pf, _ = book(c0)
    rows = _backtest(pf, _flat_path(92, 1.20)).rows
    mtm = [r["portfolio_value"] - r["paid_cash"] for r in rows]
    assert mtm[0] != mtm[91]
    tau = 91 / 365
    rebuilt = note(fix=[t - tau for t in FIX[2:]], accrual_factors=c0.accrual_factors[2:], target=3.0,
                   state=TargetRedemptionState(c0.full_coupon(0) + c0.full_coupon(1)))
    e91 = dataclasses.replace(env(), valuation_date=VAL + timedelta(days=91))
    assert is_close(mtm[91], FxTargetRedemptionNoteMCEngine(mc(n=2_000)).price(rebuilt, e91), rel_tol=1e-12)
    assert rows[91]["paid_cash"] == c0.full_coupon(0) + c0.full_coupon(1)


def test_b2_redeeming_path_books_cash_on_its_pay_days():
    c0 = note(target=1.5, pay_times=LAG2D)
    pf, _ = book(c0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TerminalBookingWarning)
        rows = _backtest(pf, _flat_path(70, 1.20)).rows
    c_first = c0.full_coupon(0)
    c_second = min(c0.full_coupon(1), 1.5 - c_first)
    paid = [r["paid_cash"] for r in rows]
    assert paid[32] == 0.0 and paid[33] == c_first and paid[62] == c_first
    assert paid[63] == c_first + (c_second + 100.0)
    e = env()
    for d in (61, 62):  # terminal from the second fixing day: value = sum DF x pending, no simulation
        t = LAG2D[1] - d / 365
        expected = 0.0 + c_second * e.get_domestic_df(t) + 100.0 * e.get_domestic_df(t)
        assert is_close(rows[d]["pending_pv"], expected, rel_tol=1e-12)
        assert is_close(rows[d]["portfolio_value"] - paid[d], rows[d]["pending_pv"], rel_tol=1e-12)
    for d in range(63, 70):
        assert rows[d]["pending_pv"] == 0.0 and rows[d]["portfolio_value"] == paid[d]


def test_b3_quiet_path_books_the_principal_and_negative_tarf_cash():
    tarn = note(target=3.0, pay_times=LAG2D)
    pf, _ = book(tarn)
    rows = _backtest(pf, _flat_path(370, 1.05)).rows
    paid = [r["paid_cash"] for r in rows]
    assert paid[366] == 0.0 and paid[367] == 100.0 and rows[-1]["portfolio_value"] == 100.0
    f = tarf(target=0.20, gearing=2.0, pay_times=LAG2D)
    pf, _ = book(f)
    rows = _backtest(pf, _flat_path(370, 1.05)).rows
    paid = [r["paid_cash"] for r in rows]
    steps = [b - a for a, b in zip([0.0] + paid[:-1], paid)]
    assert sum(1 for s in steps if s < 0.0) == 12 and all(s <= 0.0 for s in steps)
    diff = 1.0 * (1.05 - 1.15)
    cf = 0.0 - 2.0 * (1.0 * max(-diff, 0.0))
    expected = 0.0
    for _ in range(12):
        expected = expected + (0.0 + 1.0 * cf)
    assert paid[-1] == expected


def test_b4_twins_in_one_backtest(monkeypatch):
    dn = dated_note(target=1.5, pay_dates=LAG2)
    dt = dated_tarf(target=0.075, gearing=2.0, pay_dates=LAG2)
    nn, nt = _numeric_twin(dn, env()), _numeric_twin(dt, env())
    pf, (id_dn, id_nn, id_dt, id_nt) = book(dn, nn, dt, nt)
    captured = []
    orig = FxTargetRedemptionLifecycle.advance

    def spy(self, portfolio, date):
        day = orig(self, portfolio, date)
        e = portfolio.pricing_environments["EURUSD"]
        captured.append((day, {pid: p.get_market_value(e) for pid, p in portfolio.positions.items()}))
        return day

    monkeypatch.setattr(FxTargetRedemptionLifecycle, "advance", spy)
    _backtest(pf, _flat_path(100, 1.20))
    for day, values in captured:
        for a, b in ((id_dn, id_nn), (id_dt, id_nt)):
            ev = {pid: [(x.event.event_type, x.event.amount, x.event.fixing, x.event.redeemed)
                        for x in day.events if x.position_id == pid] for pid in (a, b)}
            assert ev[a] == ev[b]
            paid = {pid: sum(x.quantity * x.event.amount for x in day.events
                             if x.position_id == pid and x.event.event_type is PAYMENT) for pid in (a, b)}
            assert paid[a] == paid[b]
            assert is_close(values[a], values[b], rel_tol=1e-12)


def test_b5_backtest_registration_rules():
    early = note(fix=[-10 / 365] + FIX[1:], accrual_start=-40 / 365, target=50.0)
    pf, _ = book(early)
    with pytest.warns(MissingFixingWarning) as rec:
        rows = _backtest(pf, _flat_path(5, 1.20)).rows
    assert len([w for w in rec if issubclass(w.category, MissingFixingWarning)]) == 1
    assert all(r["paid_cash"] == 0.0 for r in rows)
    gone = note(fix=[-60 / 365, -30 / 365], accrual_start=-90 / 365)
    pf, _ = book(gone)
    with pytest.warns(MaturedBeforeRunWarning):
        rows = _backtest(pf, _flat_path(5, 1.20)).rows
    assert all(r["portfolio_value"] == 0.0 and r["paid_cash"] == 0.0 for r in rows)


def test_b5_backwards_path_raises():
    pf, _ = book(note(target=3.0))
    idx = pd.DatetimeIndex([VAL, VAL + timedelta(days=2), VAL + timedelta(days=1)])
    path = pd.DataFrame({"EURUSD_spot": 1.20, "EURUSD_vol": 0.10, "EURUSD_dom_rate": 0.05,
                         "EURUSD_for_rate": 0.03}, index=idx)
    with pytest.raises(ValidationError, match="cannot go back"):
        _backtest(pf, path)


def test_business_day_path_banks_weekend_fixing_next_day(monkeypatch):
    saturday = datetime(2026, 8, 15)
    dn = dated_note(fixing_dates=[datetime(2026, 7, 15), saturday, datetime(2026, 9, 15)], target=50.0)
    nn = _numeric_twin(dn, env())
    pf, (id_d, id_n) = book(dn, nn)
    seen = []
    orig = FxTargetRedemptionLifecycle.advance

    def spy(self, portfolio, date):
        day = orig(self, portfolio, date)
        seen.extend((date, x.position_id) for x in day.events if x.event.event_type is FIXING)
        return day

    monkeypatch.setattr(FxTargetRedemptionLifecycle, "advance", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        _backtest(pf, _flat_path(70, 1.20, freq="B"))
    monday = pd.Timestamp(following(saturday))
    assert (monday, id_d) in seen and (monday, id_n) in seen


def test_b6_book_without_target_redemption_books_nothing():
    vanilla = FxVanillaOption(strike=1.2, option_type=OptionType.CALL, maturity=1.0, notional_foreign=1.0,
                              currency_pair=PAIR)
    from quantark.portfolio.fx import FXPortfolio
    pf = FXPortfolio(portfolio_name="b6", pricing_environments={"EURUSD": env()})
    pf.add_position(product=vanilla, quantity=1.0, entry_price=0.0, underlying="EURUSD",
                    engine=GarmanKohlhagenEngine())
    res = _backtest(pf, _flat_path(20, 1.20))
    assert all(r["paid_cash"] == 0.0 and r["pending_pv"] == 0.0 for r in res.rows)
```

- [ ] **Step 2: Run to verify they fail**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_runs.py -k "b1 or b2 or b3 or b4 or b5 or b6 or business_day" -n 2 -q`
Expected: FAIL — `KeyError: 'paid_cash'` (rows have no cash column) / frozen values.

- [ ] **Step 3: Implement**

In `quantark/backtest/fx/engine.py` (import `from quantark.asset.fx.lifecycle import FxTargetRedemptionLifecycle`):

```python
        # Apply first row, establish baseline.
        first_ts = path.index[0]
        self._apply_row(portfolio, pairs, path.iloc[0], first_ts)
        lifecycle = FxTargetRedemptionLifecycle()   # spec 7.3: TARN/TARF age with the run's clock
        lifecycle.register(portfolio)
        initial_value = portfolio.get_portfolio_value()
        ...
        cum_paid = 0.0
        rows = []

        for ts, row in path.iterrows():
            self._apply_row(portfolio, pairs, row, ts)
            day = lifecycle.advance(portfolio, ts)   # a no-op on the registration date
            cum_paid += day.paid_cash
            ...
            value = portfolio.get_portfolio_value() + cum_paid
            ...
            rows.append({
                ...existing keys unchanged...,
                "paid_cash": cum_paid,
                "pending_pv": day.pending_pv,
            })

        return FXBacktestResults(
            rows=rows,
            initial_value=initial_value,
            final_value=portfolio.get_portfolio_value() + cum_paid,
            ...
        )
```
Update the module docstring: "TARN / TARF positions age with the run (fixings bank at each row's spot, determined
cash moves to `paid_cash` on its pay date); `portfolio_value` includes cumulative paid cash."

Append to the CHANGELOG `### Changed`:

```markdown
- `FXBacktestEngine` ages TARN / TARF positions with the run: each row banks the fixings due at its spot and books
  determined cash on its pay date. Rows gain `paid_cash` (cumulative) and `pending_pv`; `portfolio_value` and
  `final_value` include paid cash. A book without TARN / TARF is unchanged to the bit (cash columns 0.0).
```

- [ ] **Step 4: Run to verify they pass**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_runs.py test/test_fx_backtest.py test/test_fx_integration_demo.py -n 2 -q`
Expected: all pass; `test/test_fx_backtest.py` unmodified.

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/backtest/fx/engine.py test/test_fx_target_redemption_runs.py CHANGELOG.md
git -C <WT> commit -m "feat(fx-backtest): TARN/TARF age with the run; paid cash and pending PV columns

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: FX dynamic scenario ages TARN / TARF (Q8, snapshots, B4 dynamic, B6 invariants)

Spec §7.3 (dynamic scenario), §12 Q8, §10 B4/B6.

**Files:**
- Modify: `quantark/dynamicscenario/fx/engine.py:59-123`, `quantark/dynamicscenario/results/dynamic_results.py:97` (docstring: add `'FIXING'`), `CHANGELOG.md`
- Test: `test/test_fx_target_redemption_runs.py`

**Interfaces:**
- Consumes: Task 9 hook.
- Produces: `FXDynamicScenarioEngine.run` raises `ValidationError` naming `DayPath.start_date` for an undated path with a TARN/TARF; fills `DayResult.paid_cash` / `realized_cash` (cumulative), `pending_receivable_pv`, `lifecycle_events` (one `LifecycleEventSnapshot(event_type="FIXING", ...)` per FIXING; `cashflow` per Ruling R6); `portfolio_value`, `final_value` include cumulative paid cash.

- [ ] **Step 1: Write the failing tests**

Append (imports: `from quantark.dynamicscenario import FXDynamicScenarioConfig, FXDynamicScenarioEngine`,
`from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange`,
`from quantark.stresstest.stress.stress_types import StressType`):

```python
def _held_path(days, spot, start=VAL):
    steps = [DayStep(day_index=i, changes=[ParameterChange("spot", StressType.VALUE, spot)]) for i in range(days)]
    return DayPath(name="held", steps=steps, start_date=start)


def _scenario(pf, path):
    return FXDynamicScenarioEngine(FXDynamicScenarioConfig(calculate_greeks=False)).run(pf, path)


def test_q8_undated_path_with_a_tarn_is_refused():
    pf, _ = book(note(target=3.0))
    with pytest.raises(ValidationError, match="DayPath.start_date"):
        _scenario(pf, _held_path(5, 1.20, start=None))


def test_start_date_before_environment_date_raises():
    pf, _ = book(note(target=3.0))
    with pytest.raises(ValidationError, match="cannot go back"):
        _scenario(pf, _held_path(5, 1.20, start=VAL - timedelta(days=1)))


def test_dynamic_tarn_redeems_and_books_cash():
    c0 = note(target=1.5, pay_times=LAG2D)
    pf, (pid,) = book(c0)
    res = _scenario(pf, _held_path(70, 1.20))
    days = res.day_results
    c_first = c0.full_coupon(0)
    c_second = min(c0.full_coupon(1), 1.5 - c_first)
    assert days[33].paid_cash == c_first and days[33].realized_cash == c_first
    assert days[63].paid_cash == c_first + (c_second + 100.0)
    fixings = [(i, s) for i, d in enumerate(days) for s in d.lifecycle_events]
    assert [i for i, _ in fixings] == [31, 61]
    first, second = fixings[0][1], fixings[1][1]
    assert (first.event_type, first.spot, first.payoff, first.cashflow, first.terminates_position) == (
        "FIXING", 1.20, c_first, 0.0, False)  # pay lag: nothing paid on the fixing day (Ruling R6)
    assert (second.payoff, second.terminates_position, second.position_id) == (c_second, True, pid)
    assert days[61].pending_receivable_pv > 0.0 and days[69].pending_receivable_pv == 0.0
    assert days[69].portfolio_value == days[69].paid_cash
    assert res.final_value == days[69].paid_cash


def test_dynamic_snapshot_cashflow_when_paid_on_the_fixing_day():
    c0 = note(target=1.5)
    pf, _ = book(c0, quantities=[3.0])
    days = _scenario(pf, _held_path(70, 1.20)).day_results
    c_first = c0.full_coupon(0)
    c_second = min(c0.full_coupon(1), 1.5 - c_first)
    (s1,), (s2,) = days[31].lifecycle_events, days[61].lifecycle_events
    assert s1.cashflow == 3.0 * c_first
    assert s2.cashflow == 3.0 * (c_second + 100.0)


def test_b4_twins_in_one_dynamic_scenario():
    dn = dated_note(target=1.5, pay_dates=LAG2)
    dt = dated_tarf(target=0.075, gearing=2.0, pay_dates=LAG2)
    nn, nt = _numeric_twin(dn, env()), _numeric_twin(dt, env())
    pf, (id_dn, id_nn, id_dt, id_nt) = book(dn, nn, dt, nt)
    days = _scenario(pf, _held_path(100, 1.20)).day_results
    for d in days:
        by = {}
        for s in d.lifecycle_events:
            by.setdefault(s.position_id, []).append((s.payoff, s.cashflow, s.terminates_position, s.spot))
        assert by.get(id_dn) == by.get(id_nn) and by.get(id_dt) == by.get(id_nt)


def test_b6_dynamic_book_without_target_redemption_books_nothing():
    from quantark.portfolio.fx import FXPortfolio
    pf = FXPortfolio(portfolio_name="b6", pricing_environments={"EURUSD": env()})
    pf.add_position(product=FxVanillaOption(strike=1.2, option_type=OptionType.CALL, maturity=1.0,
                                            notional_foreign=1.0, currency_pair=PAIR),
                    quantity=1.0, entry_price=0.0, underlying="EURUSD", engine=GarmanKohlhagenEngine())
    for start in (VAL, None):
        days = _scenario(pf, _held_path(10, 1.20, start=start)).day_results
        assert all(d.paid_cash == 0.0 and d.pending_receivable_pv == 0.0 and d.lifecycle_events == []
                   for d in days)
```

- [ ] **Step 2: Run to verify they fail**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_runs.py -k "q8 or start_date or dynamic" -n 2 -q`
Expected: FAIL — no refusal on the undated path; `paid_cash` stays 0.0.

- [ ] **Step 3: Implement**

In `quantark/dynamicscenario/fx/engine.py` (imports: `from quantark.asset.fx.lifecycle import
FxTargetRedemptionLifecycle`, `from quantark.asset.fx.product.option.target_redemption_state import
TargetRedemptionEventType`, `LifecycleEventSnapshot` from the results module):

```python
        start_time = time.time()
        working = self._clone_portfolio(portfolio)
        lifecycle = FxTargetRedemptionLifecycle()
        if day_path.start_date is None and lifecycle.holds_ageable(working):
            raise ValidationError(  # spec 12 Q8 (a): no clock is invented
                "the book holds an FX TARN / TARF, which ages by date: set DayPath.start_date (an undated path "
                "cannot age it)")
        lifecycle.register(working)
        baseline_value = working.get_portfolio_value()
        previous_value = baseline_value
        cum_paid = 0.0
        day_results: List[DayResult] = []

        for day_step in day_path:
            day_date = day_path.get_date_for_day(day_step.day_index)
            self._apply_day_changes(working, day_step)
            snapshots: List[LifecycleEventSnapshot] = []
            pending_pv = 0.0
            if day_date is not None:
                for env in working.pricing_environments.values():
                    env.valuation_date = day_date
                day = lifecycle.advance(working, day_date)
                cum_paid += day.paid_cash
                pending_pv = day.pending_pv
                snapshots = self._fixing_snapshots(working, day, day_date)

            value = working.get_portfolio_value() + cum_paid
            ...
            day_results.append(DayResult(
                ...existing fields...,
                lifecycle_events=snapshots,
                pending_receivable_pv=pending_pv,
                paid_cash=cum_paid,
                realized_cash=cum_paid,
            ))
            previous_value = value

        final_value = working.get_portfolio_value() + cum_paid
```

and the helper:

```python
    @staticmethod
    def _fixing_snapshots(portfolio, day, day_date) -> List[LifecycleEventSnapshot]:
        """One FIXING snapshot per banked fixing; ``cashflow`` = quantity x that fixing's cash paid the same day
        (an immediate PAYMENT carries its FIXING's label - Ruling R6 of the W4b plan)."""
        out: List[LifecycleEventSnapshot] = []
        for entry in day.events:
            ev = entry.event
            if ev.event_type is not TargetRedemptionEventType.FIXING:
                continue
            position = portfolio.positions[entry.position_id]
            paid = sum(x.event.amount for x in day.events
                       if x.position_id == entry.position_id
                       and x.event.event_type is TargetRedemptionEventType.PAYMENT
                       and x.event.label == ev.label)
            out.append(LifecycleEventSnapshot(
                position_id=entry.position_id, underlying=position.underlying,
                product_type=type(position.product).__name__, event_type="FIXING", date=day_date,
                observation_index=None, spot=ev.fixing, barrier=None, payoff=ev.amount,
                cashflow=entry.quantity * paid, terminates_position=ev.redeemed,
            ))
        return out
```

In `dynamic_results.py` `LifecycleEventSnapshot` docstring: `event_type: One of 'KO', 'KI', 'COUPON', 'MATURITY',
'EXPIRY', 'FIXING'`.

Append to the CHANGELOG `### Changed`:

```markdown
- `FXDynamicScenarioEngine` ages TARN / TARF positions on a dated `DayPath` (fixings bank at the day's spot;
  determined cash moves to `paid_cash` / `realized_cash` on its pay date; one `FIXING` lifecycle snapshot per banked
  fixing; `pending_receivable_pv` filled). An undated `DayPath` with a TARN / TARF in the book raises
  `ValidationError` (user decision Q8 a). Books without TARN / TARF are unchanged to the bit.
```

- [ ] **Step 4: Run to verify they pass**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_fx_target_redemption_runs.py test/test_fx_dynamic_scenario.py test/test_fx_integration_demo.py -n 2 -q`
Expected: all pass; `test/test_fx_dynamic_scenario.py` unmodified.

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/dynamicscenario/fx/engine.py quantark/dynamicscenario/results/dynamic_results.py test/test_fx_target_redemption_runs.py CHANGELOG.md
git -C <WT> commit -m "feat(fx-dynamic): TARN/TARF age on dated paths; undated paths with them are refused (Q8)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Final gate (B6 byte compare, demos, full suite, sign-off list)

Spec §8, §10 B6/C1.

**Files:**
- Modify: `docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/moved_values.txt`, `CHANGELOG.md` (review only)

**Interfaces:**
- Consumes: everything above; Task 1 baselines.
- Produces: a branch ready for the whole-branch review and the user's sign-off.

- [ ] **Step 1: B6 byte gate**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python <WT>/docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/w4b_baseline.py compare <WT>/.superpowers/w4b/baseline.json`
Expected: `B6 OK: every stored number bitwise equal; new cash columns 0.0`. Any diff: STOP, report (a book without
TARN/TARF must not move).

- [ ] **Step 2: Demo stdout byte compare (C1)**

Run:
```bash
for f in fx_tarn_demo fx_backtest_demo fx_dynamic_scenario_demo fx_portfolio_risk_demo; do
  env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python <WT>/example/$f.py > <WT>/.superpowers/w4b/demo_$f.branch.txt 2>&1
  cmp <WT>/.superpowers/w4b/demo_$f.main.txt <WT>/.superpowers/w4b/demo_$f.branch.txt && echo "$f SAME"; done
```
Expected: four `SAME`.

- [ ] **Step 3: Full suite (controller, `-n 6`)**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test -n 6 -q` then
`git -C <WT> checkout -- example/mo_volmodels/data/mo_barrier_sample.json example/mo_volmodels/data/mo_calibration_explainer_sample.html`.
Expected: pass except the known pre-existing `mo_volmodels` / `otc` failures on main (memory: 4 tests since 0.7.0) —
compare the failing node ids against a `main` run; any other failure blocks. A `mo` smoke "fingerprint drift" is a
shared-artifact race: re-run that test with `-n0` before calling it a defect.

- [ ] **Step 4: Complete the sign-off list**

Append to `moved_values.txt`:

```text
#
# Constructor-only edits (no expected value moved): test/test_fx_tarn_note_mc.py::_note,
# test/test_fx_tarf_mc.py::_tarf, test/execution/matrix_fixtures.py fx_tarn / fx_tarf, example/fx_tarn_demo.py
# (demo stdout byte-identical to main).
# test/test_fx_tarn_note_mc.py::_det_pv gained acc0=0.0 (default reproduces the old reference).
#
# Behaviour changes with no previously asserted value (spec section 8 "Expected to move"):
# theta within one day of a fixing: exception -> number (M3); construction without state / period start:
# accepted -> refused; pricing after a bare time_shift: exception -> MissingFixingWarning fill (M2); a backtest or
# dated dynamic scenario holding a TARN/TARF: frozen -> aged with paid cash (M7); an undated dynamic scenario holding
# a TARN/TARF: ran frozen -> ValidationError (Q8).
```
Verify the CHANGELOG block has `### Added`, `### Changed`, `### Fixed` with the bullets from Tasks 2–11 and no empty
heading.

- [ ] **Step 5: Commit**

```bash
git -C <WT> add -f docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/moved_values.txt
git -C <WT> add CHANGELOG.md
git -C <WT> commit -m "docs(fx-tarn): moved-values list for sign-off; final gates green

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Landing (controller, after the whole-branch review): the user signs off `moved_values.txt`; only then ff-merge.

---

## Self-review (done while writing)

- **Spec coverage:** §3 → T2/T3; §4.1 → T3/T4; §4.2/§4.3 → T7; §5.1/§5.2 → T6/T7; §5.3 → T3/T4/T6/T7; §5.4 → T3/T6/T7;
  §6 → T5/T7/T8; §7.2 → T9; §7.3 → T10/T11; §8 → T3/T4/T12; §10 P1–P5 → T6; D1/D3/D4/D5 → T7, D2 → T9, D4 business → T8;
  B1–B5 → T9/T10/T11, B6 → T1/T10/T11/T12; WARN1 → T3/T4/T6/T7, WARN2 → T6 (P3) and T10 (B2), WARN3 → T6/T7; R1/R2 →
  T3/T4/T7; T1/T2 → T6; C1 → T3/T4/T12; §12 Q7 → T8, Q8 → T11.
- **Type consistency:** `resolve(fx_env, *, warn_missing_fixings=True)`, `pending_pv(fx_env)`, `matured_before(fx_env)`,
  `coupon_amount(factor)`, `_fixing_outcome(factor, spot, accumulated)`, `FxLifecycleDay(date, paid_cash, events,
  pending_pv)`, `FxLifecycleEntry(position_id, quantity, event)` are used with the same names in every task.
