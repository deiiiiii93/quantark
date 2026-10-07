# Theta at an Observation Boundary — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Theta keeps an observation that lands on the bumped date, carries the outcome decided at the as-of date, and prices a contract whose monitoring has ended instead of returning theta 0.

**Architecture:** `ObservationSchedule.time_shift` gains `keep_landing` (the theta paths pass `True`; lifecycle aging keeps `False`) and returns an empty schedule marked `monitoring_ended` instead of `None`. A contract whose monitoring has ended is priced by every engine as a static replication on the analytic engines (vanillas plus cash digitals, the 3C `ExpiryReplication` machinery), which also gives its Greeks through `BaseEngine.static_replication`. One function, `shift_for_theta`, used by every theta caller, decides the as-of observation on the spot, carries its outcome (lifecycle state, accrued fixings, a fixed receivable) and shifts the product.

**Tech Stack:** Python 3.11, NumPy/SciPy, pytest (`-n` at most 6, RAM guard active).

**Spec:** `docs/superpowers/specs/2026-10-07-theta-observation-boundary-design.md`

## Global Constraints

- Lifecycle aging (`lifecycle/autocallable.py`, `backtest/replay/*`, `backtest/simulation/pricing/repricing.py`, the dashboard) is byte-identical: it never passes `keep_landing=True` and gets no outcome carry (R2).
- A record ON the bumped instant is kept at exactly `observation_time = 0.0` only under `keep_landing=True`; records before it are dropped (R1).
- An empty schedule is legal only with `monitoring_ended=True`, set only by `time_shift` (spec 3.3).
- Cash paid at or before the bumped instant leaves the value; a receivable paid after it stays (R4).
- No invented fallbacks: an engine that cannot carry a required lifecycle state raises (`CapabilityError`), never prices without it.
- Use `quantark.util.numerical` (`is_zero`, `Tolerance`) for every comparison.
- Full suite: `.venv/bin/python -m pytest -q -n 6`; afterwards `git checkout -- example/mo_volmodels/data/mo_barrier_sample.json example/mo_volmodels/data/mo_calibration_explainer_sample.html`. Known main-checkout failures (4): `test/mo_volmodels/test_grid_preflight.py::test_one_real_operating_point_builds_its_grid`, two in `test/mo_volmodels/test_adi_slv_aggregate_certification.py`, `test/test_otc_vol_calibrators.py::TestVolModelMcMethodSlot::test_putting_the_mc_method_in_method_still_breaks_pde`.

## Review Focus

1. A Friday-to-Monday business-day bump: Saturday/Sunday records (if any) are past, Monday's is kept — Task 1 test `test_multi_day_bump_drops_between_keeps_landing`.
2. A dated record at midnight against a bumped date carrying a 15:00 time of day is the landing record — Task 1 test `test_dated_landing_ignores_time_of_day`.
3. Float residue: `0.1 + 0.2 - 0.3`-style residue (5.6e-17) after the shift becomes exactly 0.0, never a record "1e-17 ahead" (QUAD raises on `0 < t <= 1e-10`) — Task 1 test `test_landing_residue_snaps_to_zero`.
4. A schedule emptied by hand (not by `time_shift`) is still refused at construction — Task 2 test `test_hand_emptied_schedule_is_refused`.
5. A theta at the as-of date that knocks the contract out: the bumped value is the receivable if it settles after the bump, 0 otherwise, never the alive contract — Task 5 test `test_t0_knock_out_carries_the_receivable`.

## File Structure

- Modify `quantark/asset/equity/product/option/observation_schedule.py` — `monitoring_ended`, `time_shift(keep_landing)`, `validate`.
- Create `quantark/asset/equity/engine/monitoring_ended.py` — `monitoring_ended(product)`, `remaining_legs(product, knocked_in)`, `remaining_replication(product, knocked_in)`.
- Modify engines (Task 3/4): analytic barrier, one-touch, double barrier, sharkfins, accumulator; MC barrier, sharkfins, accumulator; PDE base solver + barrier family adapters; QUAD discrete adapters; vol barrier MC/PDE.
- Modify products' `time_shift`: `base_equity_option.py`, `one_touch_option.py`, `double_one_touch_option.py`, `single_sharkfin_option.py`, `double_sharkfin_option.py`, `accumulator_option.py`, `range_accrual_option.py` (new override), `snowball_config.py`, `snowball_option.py`, `phoenix_option.py`; `cashleg/autocallable_leg.py`.
- Create `quantark/asset/equity/riskmeasures/greeks/theta_shift.py` — `ThetaShift`, `shift_for_theta(...)`.
- Modify theta callers: `riskmeasures/greeks/numerical.py`, `greeks_calculator.py` (zero branch), `execution/greeks.py`, `portfolio/equity/position.py`, `report/autocallable_risk_report.py`.
- Modify legacy autocallable QUAD engines (Task 6).
- Tests: `test/theta_boundary/` (new package: `test_schedule_shift.py`, `test_monitoring_ended.py`, `test_product_shift.py`, `test_theta_shift.py`, `test_quad_empty_side.py`), restated existing tests (Task 7).

---

### Task 1: The schedule's boundary rule

**Files:**
- Modify: `quantark/asset/equity/product/option/observation_schedule.py` (class `ObservationSchedule`: fields, `validate` ~:313, `time_shift` :337-375)
- Test: `test/theta_boundary/__init__.py` (empty), `test/theta_boundary/test_schedule_shift.py`

**Interfaces:**
- Produces: `ObservationSchedule.monitoring_ended: bool = False` (dataclass field, last); `ObservationSchedule.time_shift(time_bump: float, bumped_date: Optional[datetime] = None, *, keep_landing: bool = False) -> ObservationSchedule` (never `None`; an emptied result has `records=[]`, `monitoring_ended=True`).

- [ ] **Step 1: Write the failing tests**

```python
"""ObservationSchedule.time_shift at the bumped instant (spec 2026-10-07 theta boundary, R1, 3.1, 3.3)."""
from datetime import datetime

import pytest

from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.util.exceptions import ValidationError

D = 1 / 252


def numeric(times):
    return ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=120.0) for t in times])


def dated(dates):
    return ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=120.0) for d in dates])


def test_landing_record_kept_at_zero_only_when_asked():
    sch = numeric([D, 2 * D])
    kept = sch.time_shift(D, keep_landing=True)
    assert [r.observation_time for r in kept.records] == [0.0, pytest.approx(D)]
    assert kept.records[0].observation_time == 0.0
    dropped = sch.time_shift(D)
    assert [r.observation_time for r in dropped.records] == [pytest.approx(D)]


def test_landing_residue_snaps_to_zero():
    sch = numeric([0.1 + 0.2])
    out = sch.time_shift(0.3, keep_landing=True)
    assert out.records[0].observation_time == 0.0          # 0.30000000000000004 - 0.3 = 5.6e-17


def test_past_records_dropped_and_t0_gone():
    sch = numeric([0.0, D, 2 * D])                          # 0.0 is the as-of date's own record
    out = sch.time_shift(D, keep_landing=True)
    assert [r.observation_time for r in out.records] == [0.0, pytest.approx(D)]


def test_multi_day_bump_drops_between_keeps_landing():
    sch = numeric([D, 2 * D, 3 * D, 4 * D])                 # a three-day bump lands on 3D
    out = sch.time_shift(3 * D, keep_landing=True)
    assert [r.observation_time for r in out.records] == [0.0, pytest.approx(D)]


def test_dated_landing_ignores_time_of_day():
    sch = dated([datetime(2026, 1, 2), datetime(2026, 1, 5)])
    out = sch.time_shift(D, datetime(2026, 1, 2, 15, 0), keep_landing=True)
    assert out.records[0].observation_date == datetime(2026, 1, 2)
    assert out.records[0].observation_time == 0.0
    assert out.records[1].observation_date == datetime(2026, 1, 5)
    assert sch.time_shift(D, datetime(2026, 1, 2, 15, 0)).records[0].observation_date == datetime(2026, 1, 5)


def test_emptied_schedule_is_marked_not_none():
    out = numeric([D]).time_shift(2 * D, keep_landing=True)
    assert out is not None and out.records == [] and out.monitoring_ended
    out.validate()                                          # legal: monitoring has ended


def test_hand_emptied_schedule_is_refused():
    with pytest.raises(ValidationError):
        ObservationSchedule(records=[]).validate()
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest -q -n0 test/theta_boundary/test_schedule_shift.py`
Expected: FAIL (`keep_landing` unexpected keyword; `None` returned for the emptied schedule).

- [ ] **Step 3: Implement**

In `ObservationSchedule` add the field `monitoring_ended: bool = False` after `frequency`. In `validate`, replace the empty check with:

```python
        if len(self.records) == 0:
            if self.monitoring_ended:
                return
            raise ValidationError("ObservationSchedule must contain at least one record for discrete monitoring.")
```

Replace `time_shift` with:

```python
    def time_shift(
        self,
        time_bump: float,
        bumped_date: Optional[datetime] = None,
        *,
        keep_landing: bool = False,
    ) -> "ObservationSchedule":
        """Shift the schedule by ``time_bump`` (to ``bumped_date``) and drop what is past.

        A record before the bumped instant is dropped. A record ON it (numeric: ``is_zero(t - time_bump)``; dated:
        the same calendar date, whatever the time of day) is kept at exactly ``observation_time = 0.0`` when
        ``keep_landing`` (theta: it is observed at the new as-of instant, spec 2026-10-07 R1) and dropped otherwise
        (lifecycle aging has already observed it, R2). A schedule left with no record is returned empty with
        ``monitoring_ended`` set, never ``None``.
        """
        if not self.records:
            return self
        updated_records = []
        for rec in self.records:
            rec_copy = deepcopy(rec)
            landing = False
            if rec_copy.observation_date is not None and bumped_date is not None:
                day, bumped_day = rec_copy.observation_date.date(), bumped_date.date()
                if day < bumped_day:
                    continue
                landing = day == bumped_day
            if rec_copy.observation_time is not None:
                adjusted = rec_copy.observation_time - time_bump
                if is_zero(adjusted):
                    landing = True
                elif adjusted < 0.0:
                    continue
                else:
                    rec_copy.observation_time = adjusted
            if landing:
                if not keep_landing:
                    continue
                rec_copy.observation_time = 0.0
            if rec_copy.settlement_time is not None:
                rec_copy.settlement_time -= time_bump
            updated_records.append(rec_copy)
        return ObservationSchedule(
            records=updated_records,
            aggregation_mode=self.aggregation_mode,
            frequency=self.frequency,
            monitoring_ended=not updated_records,
        )
```

Add `from quantark.util.numerical import is_zero` to the imports. A dated record with `observation_time = 0.0` must resolve to time 0: check `SettlementResolver.resolve_contingent` (`settlement.py`) uses `determination_time` when both are set; if it prefers the date, make `ObservationRecord.resolve_time` and the `resolve` request pass `determination_time=0.0` with `determination_date=None` for such a record (test in Step 1's dated case through `resolve`, add `assert out.resolve(env)[0].observation_time == 0.0` with `env = PricingEnvironment(..., valuation_date=datetime(2026, 1, 2, 15, 0))` built as in `test/pde_routes/cases.py::env_const`).

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest -q -n0 test/theta_boundary/test_schedule_shift.py test/test_observation_schedule.py`
Expected: PASS (existing schedule tests unchanged: `keep_landing` defaults to `False`).

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/product/option/observation_schedule.py test/theta_boundary/
git commit -m "feat(schedule): time_shift keeps the record on the bumped instant when asked; an emptied schedule is marked monitoring_ended"
```

---

### Task 2: Products age always and accept an ended schedule

**Files:**
- Modify: `base_equity_option.py:645-669`, `one_touch_option.py:226-245`, `double_one_touch_option.py:240-259`, `single_sharkfin_option.py:409-424` and `:248-261`, `double_sharkfin_option.py:454-469` and `:268-281`, `accumulator_option.py:436-451`, the products' discrete-monitoring construction checks (`barrier_option.py:154-161`, `double_barrier_option.py:166-171`, `one_touch_option.py:149-154`, `double_one_touch_option.py:160-165`), `snowball_config.py:76-168`, `snowball_option.py:588-607`, `phoenix_option.py:490-509`
- Test: `test/theta_boundary/test_product_shift.py`

**Interfaces:**
- Consumes: Task 1 `time_shift(..., keep_landing=...)`, `monitoring_ended`.
- Produces: every product `time_shift(time_bump, bumped_date, pricing_env, *, keep_landing: bool = False) -> None` (no flag returned); `BarrierConfig.time_shift(time_bump, bumped_date, pricing_env, *, keep_landing=False) -> tuple[BarrierConfig, bool]` (signature kept for the lifecycle callers; the bool stays "both sides spent").

- [ ] **Step 1: Write the failing tests**

```python
"""Products' time_shift (spec 3.1, 3.3): always aged, landing kept by flag, an ended schedule accepted."""
import pytest

from quantark.asset.equity.product.option import BarrierOption, OneTouchOption, SingleSharkfinOption
from quantark.asset.equity.product.option.observation_schedule import ObservationSchedule
from quantark.util.enum import BarrierDirection, BarrierType, ObservationType, OptionType, TouchType
from quantark.util.exceptions import ValidationError

D = 1 / 252


def env():
    from pde_routes.cases import env_const
    return env_const()


def barrier(dates, T=1.0):
    return BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT,
                         maturity=T, rebate=5.0, observation_type=ObservationType.DISCRETE, observation_dates=dates)


def test_shift_past_last_observation_ages_and_ends_monitoring():
    p = barrier([D])
    assert p.time_shift(2 * D, None, env(), keep_landing=True) is None
    assert p.maturity == pytest.approx(1.0 - 2 * D)
    assert p.observation_schedule.records == [] and p.observation_schedule.monitoring_ended
    assert p.observation_dates == []


def test_landing_kept_for_theta_dropped_for_aging():
    p, q = barrier([D, 2 * D]), barrier([D, 2 * D])
    p.time_shift(D, None, env(), keep_landing=True)
    q.time_shift(D, None, env())
    assert p.observation_dates == [0.0, pytest.approx(D)]
    assert q.observation_dates == [pytest.approx(D)]


def test_sharkfin_does_not_regenerate_past_dates():
    p = SingleSharkfinOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, maturity=1.0,
                             participation_rate=0.8, knock_out_rebate=5.0, no_hit_rebate=3.0,
                             observation_type=ObservationType.DISCRETE, observation_dates=[D])
    p.time_shift(2 * D, None, env(), keep_landing=True)
    p.validate()
    assert p.observation_schedule.records == []


def test_hand_emptied_schedule_is_refused():
    with pytest.raises(ValidationError):
        BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT,
                      maturity=1.0, observation_type=ObservationType.DISCRETE,
                      observation_schedule=ObservationSchedule(records=[]))


def test_autocallable_config_keeps_landing_by_flag():
    from quantark.asset.equity.product.option import SnowballOption
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig

    s = SnowballOption(initial_price=100.0, strike=100.0, maturity=1.0,
                       barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_dates=[D, 0.5, 1.0],
                                                    ki_barrier=75.0, ki_continuous=True))
    s.time_shift(D, None, env(), keep_landing=True)
    assert s.barrier_config.ko_observation_dates[0] == 0.0
```

(`pde_routes.cases` is importable from `test/` as the other `test/pde_routes` tests do; if `test/theta_boundary` cannot import it, build the environment inline with `PricingEnvironment(rate_curve=FlatRateCurve(0.03), valuation_date=datetime(2026, 1, 1), spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(volatility=0.22), div_yield=ContinuousDividendYield(0.01))`.)

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest -q -n0 test/theta_boundary/test_product_shift.py`
Expected: FAIL (unexpected keyword `keep_landing`; `True` returned; maturity not aged).

- [ ] **Step 3: Implement**

`BaseEquityOption.time_shift` becomes:

```python
    def time_shift(self, time_bump: float, bumped_date: datetime, pricing_env, *, keep_landing: bool = False) -> None:
        """Age the option by ``time_bump`` (to ``bumped_date``): the schedule drops what is past and keeps the record
        on the bumped instant only under ``keep_landing`` (theta, spec 2026-10-07 R1); the maturity or valuation date
        always moves, an emptied schedule is kept marked ``monitoring_ended`` (R3)."""
        schedule = getattr(self, "observation_schedule", None)
        if schedule is not None:
            if schedule.uses_dates():
                pricing_env.valuation_date = bumped_date
            self.observation_schedule = schedule.time_shift(time_bump, bumped_date, keep_landing=keep_landing)
            if hasattr(self, "observation_dates"):
                self.observation_dates = self.observation_schedule.times
        if getattr(self, "exercise_date", None) is None:
            if getattr(self, "maturity", None) is not None:
                self.maturity -= time_bump
        else:
            pricing_env.valuation_date = bumped_date
```

Apply the same body to the overrides in `one_touch_option.py`, `double_one_touch_option.py`, `single_sharkfin_option.py`, `double_sharkfin_option.py`, `accumulator_option.py` (each keeps its own maturity line as it is today, plus the `exercise_date` branch where present). `observation_dates = schedule.times` gives `[]` for an ended schedule (fixes the stale-dates gap).

Construction checks: each of `barrier_option.py`, `double_barrier_option.py`, `one_touch_option.py`, `double_one_touch_option.py` raises when `observation_schedule is None and not observation_dates`; leave that, and route an ended schedule through `normalized_schedule = ObservationSchedule(records=[...], aggregation_mode=..., frequency=..., monitoring_ended=self.observation_schedule.monitoring_ended)` so the normalization keeps the flag. In the sharkfins' `_normalize_observation_schedule`, keep the flag the same way and skip `_generate_observation_dates` when `observation_schedule is not None and observation_schedule.monitoring_ended`.

`BarrierConfig.time_shift` and `_shift_side` take `keep_landing` and pass it to `schedule.time_shift`; the legacy-dates branch keeps `t - time_bump > 0` when not `keep_landing` and, under it, maps `is_zero(t - time_bump)` to `0.0` and keeps `> 0`. `_shift_side` no longer needs the `None` conversion (Task 1 never returns `None`); keep its empty-schedule path. `SnowballOption.time_shift` / `PhoenixOption.time_shift` take `keep_landing`, pass it to `super().time_shift(...)` and `self.barrier_config.time_shift(...)`, and return `None` (the `accrual_config.shifted(...)` count of dropped KO records is unchanged in meaning: count only records dropped, not the kept landing record).

- [ ] **Step 4: Run**

Run: `.venv/bin/python -m pytest -q -n 6 test/theta_boundary test/test_observation_schedule.py test/test_barrier_option.py test/test_snowball_option.py test/test_phoenix_option.py -k "not slow"` then the theta tests `test/test_greeks_theta_schedule.py`.
Expected: Task 2 tests PASS. `test_greeks_theta_schedule.py` cases that asserted theta 0 on an emptied schedule now fail or raise (the engines cannot price an ended schedule until Tasks 3-4); note them, they are restated in Task 7.

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/product/option/ test/theta_boundary/test_product_shift.py
git commit -m "feat(products): time_shift always ages, keeps the landing record by flag and leaves an ended schedule"
```

---

### Task 3: The remaining contract once monitoring has ended

**Files:**
- Create: `quantark/asset/equity/engine/monitoring_ended.py`
- Modify: analytic `barrier_analytical_engine.py`, `one_touch_analytical_engine.py`, `double_barrier_option_engine.py`, `single_sharkfin_option_analytical_engine.py`, `double_sharkfin_option_analytical_engine.py`, `accumulator_analytical_engine.py`; MC `barrier_option_mc_engine.py`, `single_sharkfin_option_mc_engine.py`, `double_sharkfin_option_mc_engine.py`, `accumulator_mc_engine.py`
- Test: `test/theta_boundary/test_monitoring_ended.py`

**Interfaces:**
- Consumes: `pde/core/adapters/expiry_replication.py`: `Leg(weight, kind, strike, option_type)`, `ExpiryReplication(legs, product, multiplier)` with `.price(env)` and `.parts`.
- Produces: `monitoring_ended(product) -> bool`; `remaining_legs(product, knocked_in: bool = False) -> list[Leg]`; `remaining_replication(product, knocked_in: bool = False) -> ExpiryReplication`.

- [ ] **Step 1: Write the failing tests**

```python
"""A contract whose monitoring has ended (spec 3.3): every engine prices the remaining contract."""
import pytest

from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.product.option import (BarrierOption, DoubleBarrierOption, DoubleSharkfinOption,
                                                  EuropeanVanillaOption, OneTouchOption, SingleSharkfinOption)
from quantark.util.enum import (BarrierDirection, BarrierType, DoubleBarrierType, ObservationType, OptionType,
                                TouchType)

D = 1 / 252
T = 0.5


def env():
    from pde_routes.cases import env_const
    return env_const()


def ended(p):
    p.time_shift(2 * D, None, env(), keep_landing=True)     # the only record was at D: monitoring has ended
    return p


def vanilla(strike, call, T_):
    return float(BlackScholesEngine().price(
        EuropeanVanillaOption(strike=strike, option_type=OptionType.CALL if call else OptionType.PUT, maturity=T_),
        env()))


def df(t):
    return float(env().get_discount_factor(t))


TB = T - 2 * D
CASES = {
    "uo_call": (lambda: ended(BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0,
                                            barrier_type=BarrierType.UP_OUT, maturity=T, rebate=5.0,
                                            participation_rate=0.8, observation_type=ObservationType.DISCRETE,
                                            observation_dates=[D])),
                lambda: 0.8 * vanilla(100.0, True, TB)),
    "ui_call": (lambda: ended(BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0,
                                            barrier_type=BarrierType.UP_IN, maturity=T, rebate=5.0,
                                            observation_type=ObservationType.DISCRETE, observation_dates=[D])),
                lambda: 5.0 * df(TB)),
    "dko_put": (lambda: ended(DoubleBarrierOption(100.0, OptionType.PUT, 130.0, 80.0, DoubleBarrierType.KNOCK_OUT,
                                                  maturity=T, rebate=5.0, observation_type=ObservationType.DISCRETE,
                                                  observation_dates=[D])),
                lambda: vanilla(100.0, False, TB)),
    "one_touch": (lambda: ended(OneTouchOption(barrier=120.0, barrier_direction=BarrierDirection.UP, maturity=T,
                                               rebate=5.0, touch_type=TouchType.ONE_TOUCH,
                                               observation_type=ObservationType.DISCRETE, observation_dates=[D])),
                  lambda: 0.0),
    "no_touch": (lambda: ended(OneTouchOption(barrier=120.0, barrier_direction=BarrierDirection.UP, maturity=T,
                                              rebate=5.0, touch_type=TouchType.NO_TOUCH,
                                              observation_type=ObservationType.DISCRETE, observation_dates=[D])),
                 lambda: 5.0 * df(TB)),
    "sharkfin_call": (lambda: ended(SingleSharkfinOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0,
                                                         maturity=T, participation_rate=0.8, knock_out_rebate=5.0,
                                                         no_hit_rebate=3.0, observation_type=ObservationType.DISCRETE,
                                                         observation_dates=[D])),
                      lambda: 0.8 * (vanilla(100.0, True, TB) - vanilla(130.0, True, TB)) + 3.0 * df(TB)),
    "double_sharkfin_put": (lambda: ended(DoubleSharkfinOption(strike=100.0, option_type=OptionType.PUT,
                                                               lower_barrier=80.0, upper_barrier=130.0, maturity=T,
                                                               participation_rate=0.8, knock_out_rebate=5.0,
                                                               no_hit_rebate=3.0,
                                                               observation_type=ObservationType.DISCRETE,
                                                               observation_dates=[D])),
                            lambda: 0.8 * (vanilla(100.0, False, TB) - vanilla(80.0, False, TB)) + 3.0 * df(TB)),
}


def engines_for(name):
    from quantark.asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine
    from quantark.asset.equity.engine.analytical.double_barrier_option_engine import DoubleBarrierOptionAnalyticalEngine
    from quantark.asset.equity.engine.analytical.double_sharkfin_option_analytical_engine import (
        DoubleSharkfinOptionAnalyticalEngine)
    from quantark.asset.equity.engine.analytical.one_touch_analytical_engine import OneTouchAnalyticalEngine
    from quantark.asset.equity.engine.analytical.single_sharkfin_option_analytical_engine import (
        SingleSharkfinOptionAnalyticalEngine)
    from quantark.asset.equity.engine.mc import BarrierOptionMCEngine
    from quantark.asset.equity.param import MCParams
    mc = MCParams(num_paths=2048, time_steps=4, seed=7)
    return {
        "uo_call": [BarrierAnalyticalEngine(), BarrierOptionMCEngine(params=mc)],
        "ui_call": [BarrierAnalyticalEngine(), BarrierOptionMCEngine(params=mc)],
        "dko_put": [DoubleBarrierOptionAnalyticalEngine()],
        "one_touch": [OneTouchAnalyticalEngine()],
        "no_touch": [OneTouchAnalyticalEngine()],
        "sharkfin_call": [SingleSharkfinOptionAnalyticalEngine()],
        "double_sharkfin_put": [DoubleSharkfinOptionAnalyticalEngine()],
    }[name]


@pytest.mark.parametrize("name", sorted(CASES))
def test_ended_monitoring_prices_the_remaining_contract(name):
    make, truth = CASES[name]
    for engine in engines_for(name):
        assert float(engine.price(make(), env())) == pytest.approx(truth(), abs=1e-10), type(engine).__name__
```

(Extend `engines_for` in Task 4 with the PDE, QUAD and vol engines and the sharkfin/accumulator MC engines; names of the MC engine classes: confirm with `grep -n "^class" quantark/asset/equity/engine/mc/*.py` before writing them.)

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest -q -n0 test/theta_boundary/test_monitoring_ended.py`
Expected: FAIL (PricingError / ValidationError on the empty schedule).

- [ ] **Step 3: Implement `monitoring_ended.py`**

```python
"""A discretely monitored contract whose monitoring has ended (spec 2026-10-07 theta boundary, 3.3).

``time_shift`` leaves such a product with an empty schedule marked ``monitoring_ended``: no barrier can be hit any
more, so what remains is a function of S_T alone, priced as a static replication on the analytic engines (the 3C
``ExpiryReplication``: European vanillas and cash digitals on the product's own expiry and settlement). A fixed
cash amount R is R x (digital call + digital put at one strike), exact since P(S_T = strike) = 0. Greeks come from
``BaseEngine.static_replication``.
"""
from __future__ import annotations

from quantark.util.enum import ObservationType, OptionType

from .pde.core.adapters.expiry_replication import ExpiryReplication, Leg

__all__ = ["monitoring_ended", "remaining_legs", "remaining_replication"]

_C, _P = OptionType.CALL, OptionType.PUT


def monitoring_ended(product) -> bool:
    schedule = getattr(product, "observation_schedule", None)
    return (getattr(product, "observation_type", None) == ObservationType.DISCRETE and schedule is not None
            and schedule.monitoring_ended and not schedule.records)


def _cash(amount: float, strike: float) -> list:
    return [] if amount == 0.0 else [Leg(amount, "digital", strike, _C), Leg(amount, "digital", strike, _P)]


def remaining_legs(product, knocked_in: bool = False) -> list:
    """The remaining contract's legs (``knocked_in``: a knock-in already triggered)."""
    from quantark.asset.equity.product.option import (BarrierOption, DoubleBarrierOption, DoubleOneTouchOption,
                                                      DoubleSharkfinOption, OneTouchOption, SingleSharkfinOption)

    if isinstance(product, BarrierOption):
        if product.is_knock_out or knocked_in:
            return [Leg(product.participation_rate, "vanilla", product.strike, product.option_type)]
        return _cash(product.rebate, product.strike)
    if isinstance(product, DoubleBarrierOption):
        if product.is_knock_out or knocked_in:
            return [Leg(1.0, "vanilla", product.strike, product.option_type)]
        return _cash(product.rebate, product.strike)
    if isinstance(product, OneTouchOption):
        return [] if product.is_one_touch else _cash(product.rebate, product.barrier)
    if isinstance(product, DoubleOneTouchOption):
        return [] if product.is_double_one_touch else _cash(product.rebate, product.upper_barrier)
    if isinstance(product, SingleSharkfinOption):
        cap = product.barrier
        spread = ([Leg(product.participation_rate, "vanilla", product.strike, _C),
                   Leg(-product.participation_rate, "vanilla", cap, _C)] if product.is_call() else
                  [Leg(product.participation_rate, "vanilla", product.strike, _P),
                   Leg(-product.participation_rate, "vanilla", cap, _P)])
        return spread + _cash(product.no_hit_rebate, product.strike)
    if isinstance(product, DoubleSharkfinOption):
        cap = product.upper_barrier if product.is_call() else product.lower_barrier
        kind = _C if product.is_call() else _P
        return ([Leg(product.participation_rate, "vanilla", product.strike, kind),
                 Leg(-product.participation_rate, "vanilla", cap, kind)]
                + _cash(product.no_hit_rebate, product.strike))
    raise TypeError(f"no remaining contract defined for {type(product).__name__}")


def remaining_replication(product, knocked_in: bool = False) -> ExpiryReplication:
    """The legs as an ``ExpiryReplication``: multiplier 1 for the sharkfins (their engines scale by
    ``contract_multiplier`` after the legs? check: they multiply the total), the product's own otherwise."""
    return ExpiryReplication(remaining_legs(product, knocked_in), product, product.contract_multiplier)
```

Check the multiplier convention against each engine before wiring: `ExpiryReplication` passes `contract_multiplier=multiplier` to every leg product, and the European/digital engines scale by it; the barrier, double-barrier and sharkfin engines return values that include the multiplier, the one-touch engine ignores the multiplier (`OneTouchOption` has none; pass `1.0` for touch products). Fix `remaining_replication` to pass `getattr(product, "contract_multiplier", 1.0)` and delete the question from its docstring.

- [ ] **Step 4: Wire the analytic and MC engines**

In each engine's `price`, right after the lifecycle terminal check (`fixed_pv = terminal_lifecycle_pv(...)` / `pending_pv = ...`) and before any schedule access, add:

```python
        if monitoring_ended(product):
            knocked_in = bool(getattr(lifecycle_state, "knocked_in", False))
            return remaining_replication(product, knocked_in).price(pricing_env) + pending_pv
```

(`pending_pv` named as in that engine; the double-barrier engine computes it in `price` — add the branch there, before `_price_immediate`.) Add to each of these engines:

```python
    def static_replication(self, product, pricing_env):
        if monitoring_ended(product):
            return remaining_replication(product)
        return super().static_replication(product, pricing_env)
```

Accumulator analytic/MC: guard `schedule.resolve` (`accumulator_analytical_engine.py:224`, `accumulator_mc_engine.py:229`) with `if not schedule.records: legs = []` so the realized accrual plus the terminal legs price (their existing empty-times path); add a test row to Step 1: an ended daily accumulator with `past_observations` equals `get_realized_accrual() × DF(T_pay)` plus its terminal extra-shares leg as each engine computes it with no future observation (assert analytic == MC to 1e-10 since no path is simulated, and analytic == `get_realized_accrual() * df(TB)` when the product has no terminal extra-shares leg).

- [ ] **Step 5: Run**

Run: `.venv/bin/python -m pytest -q -n0 test/theta_boundary/test_monitoring_ended.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add quantark/asset/equity/engine/monitoring_ended.py quantark/asset/equity/engine/analytical quantark/asset/equity/engine/mc test/theta_boundary/test_monitoring_ended.py
git commit -m "feat(engines): a contract whose monitoring has ended is priced as its remaining static replication (analytic, MC)"
```

---

### Task 4: PDE, QUAD and vol engines on an ended schedule

**Files:**
- Modify: `quantark/asset/equity/engine/pde/base_pde_solver.py` (`_setup_observation_indices` ~:365, `_valuation_instant_records` :419-436), the barrier-family PDE solvers' `price`/`calculate_greeks` entry or the core adapters (`pde/core/adapters/barrier.py:80-94` where `ExpiryReplication` is chosen for EXPIRY), `quad/quad_adapters.py` (`early_price` hooks :357, :681, :889, :1208), `barrier_vol_mc_engines.py:73`, `barrier_vol_pde_solvers.py:66`
- Test: extend `test/theta_boundary/test_monitoring_ended.py`

**Interfaces:**
- Consumes: Task 3's `monitoring_ended`, `remaining_replication`.

- [ ] **Step 1: Extend the test**

Add to `engines_for`: `BarrierPDESolver()`, `DoubleBarrierPDESolver()`, `OneTouchPDESolver()`, `DoubleOneTouchPDESolver()` (from `quantark.asset.equity.engine.pde`), the discrete QUAD engines for barrier / one-touch / double barrier / double one-touch (find with `grep -n "^class" quantark/asset/equity/engine/quad/*.py`), and the vol barrier MC/PDE engines with a flat `LocalVolSurface` (`test/pde_routes/cases.py::surface_const`). Add a double one-touch / double no-touch case to `CASES`. Add a Greeks test:

```python
def test_ended_monitoring_greeks_are_the_legs():
    from quantark.asset.equity.engine.pde import BarrierPDESolver
    from quantark.asset.equity.riskmeasures import GreeksCalculator
    make, _ = CASES["uo_call"]
    g = GreeksCalculator().calculate(product=make(), pricing_env=env(), engine=BarrierPDESolver(),
                                     greeks=["price", "delta", "gamma"])
    v = GreeksCalculator().calculate(product=EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL,
                                                                   maturity=TB), pricing_env=env(),
                                     engine=BlackScholesEngine(), greeks=["price", "delta", "gamma"])
    for k in ("price", "delta", "gamma"):
        assert g[k] == pytest.approx(0.8 * v[k], rel=1e-9)
```

- [ ] **Step 2: Run to verify they fail**, then **Step 3: Implement**: PDE — in the barrier-family adapter where EXPIRY selects `ExpiryReplication`, also select it when `monitoring_ended(product)` with `remaining_legs(product, knocked_in)` (knocked-in from the lifecycle state the adapter already receives); `_valuation_instant_records` and `_setup_observation_indices` return early on `not schedule.records`. QUAD — each `early_price` hook returns `remaining_replication(product, knocked_in).price(env)` when `monitoring_ended(product)`. Vol MC/PDE — `_monitoring` returns the ended state before reading `observation_dates`, and `price` returns the replication value. Add `static_replication` to each of these engines as in Task 3 Step 4.

- [ ] **Step 4: Run** `test/theta_boundary/test_monitoring_ended.py` and the PDE/QUAD barrier suites: `.venv/bin/python -m pytest -q -n 6 test/theta_boundary test/pde_routes test/test_pde_engine.py -k "barrier or touch or ended"`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/pde quantark/asset/equity/engine/quad quantark/asset/equity/engine/*vol* test/theta_boundary/test_monitoring_ended.py
git commit -m "feat(engines): PDE, QUAD and local-vol barrier engines price an ended schedule as its remaining replication"
```

---

### Task 5: `shift_for_theta` — the outcome at the as-of date, and the theta callers

**Files:**
- Create: `quantark/asset/equity/riskmeasures/greeks/theta_shift.py`
- Modify: `riskmeasures/greeks/numerical.py:203-262` and `:553-605`, `greeks_calculator.py:836-845`, `execution/greeks.py:150-185` and `:300-301`, `portfolio/equity/position.py:359-395`, `report/autocallable_risk_report.py:1046-1060`, `:1566-1580` and `_compute_pv_surface_sq` / `_compute_delta_surface_sq`, `range_accrual_option.py` (new `time_shift`), `accumulator_option.py` (outcome append), `cashleg/autocallable_leg.py:207`
- Test: `test/theta_boundary/test_theta_shift.py`

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces:

```python
@dataclass(frozen=True)
class ThetaShift:
    product: object                       # the bumped product (a copy)
    env: object                           # the bumped environment (a copy, valuation_date = bumped_date)
    lifecycle_state: object = None        # carried state for the bumped price, or None
    fixed_pv: float = 0.0                 # receivable decided at t0 and paid after the bump (bumped money)
    terminated: bool = False              # t0 ended the contract: the bumped value is fixed_pv alone

def shift_for_theta(product, pricing_env, time_bump: float, bumped_date, *, lifecycle_state=None) -> ThetaShift: ...
def bumped_value(engine, shift: ThetaShift) -> float: ...   # fixed_pv if terminated else engine price (+ fixed_pv)
```

- [ ] **Step 1: Write the failing tests** (`test/theta_boundary/test_theta_shift.py`)

```python
"""Theta: the landing record kept, the as-of outcome carried, an ended schedule priced (R1, R3, R4)."""
import pytest

from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.asset.equity.riskmeasures.greeks.theta_shift import bumped_value, shift_for_theta
from quantark.util.enum import BarrierType, ObservationType, OptionType

D = 1 / 252


def env(spot=100.0):
    from pde_routes.cases import env_const
    return env_const(spot)


def accumulator(dates):
    from quantark.asset.equity.product.option import AccumulatorOption
    return AccumulatorOption(strike=95.0, knock_out_barrier=105.0, initial_price=100.0, maturity=dates[-1],
                             option_type=OptionType.CALL, notional=1_000_000.0,
                             observation_type=ObservationType.DISCRETE, observation_dates=list(dates))


def test_accumulator_theta_keeps_the_landing_observation():
    """Measured before the change: theta +55,402 on V0 -154,177 (one observation dropped)."""
    from quantark.asset.equity.engine.analytical.accumulator_analytical_engine import AccumulatorAnalyticalEngine
    p, e, eng = accumulator([D * k for k in range(1, 64)]), env(), AccumulatorAnalyticalEngine()
    theta = GreeksCalculator().calculate(product=p, pricing_env=e, engine=eng, greeks=["theta"])["theta"]
    shifted = shift_for_theta(p, e, D, e.valuation_date)
    assert len(shifted.product.observation_schedule.records) == 63
    assert shifted.product.observation_schedule.records[0].observation_time == 0.0
    assert abs(theta) < 5_000.0                             # carry, not one observation's value


def test_t0_knock_in_is_carried():
    from quantark.asset.equity.engine.pde import SnowballPDESolver
    from quantark.asset.equity.product.option import SnowballOption
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig
    s = SnowballOption(initial_price=100.0, strike=100.0, maturity=1.0,
                       barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_dates=[0.5, 1.0],
                                                    ki_barrier=75.0, ki_observation_dates=[0.0, 0.25]))
    shifted = shift_for_theta(s, env(70.0), D, env(70.0).valuation_date)
    assert shifted.lifecycle_state is not None and shifted.lifecycle_state.knocked_in
    knocked = bumped_value(SnowballPDESolver(), shifted)
    from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
    direct = float(SnowballPDESolver().price(shifted.product, shifted.env,
                                             lifecycle_state=AutocallableLifecycleState(knocked_in=True)))
    assert knocked == pytest.approx(direct, abs=1e-12)


def test_t0_knock_out_carries_the_receivable():
    from quantark.asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine
    from quantark.asset.equity.product.option import BarrierOption
    p = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT,
                      maturity=1.0, rebate=5.0, observation_type=ObservationType.DISCRETE,
                      observation_dates=[0.0, 0.5, 1.0])
    shifted = shift_for_theta(p, env(125.0), D, env(125.0).valuation_date)
    assert shifted.terminated
    # rebate paid at the terminal payment time (no pay_at_hit): a receivable after the bump
    assert bumped_value(BarrierAnalyticalEngine(), shifted) == pytest.approx(
        5.0 * float(env().get_discount_factor(1.0 - D)), rel=1e-12)


def test_ended_schedule_theta_is_carry_not_zero():
    from quantark.asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine
    from quantark.asset.equity.product.option import BarrierOption
    p = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT,
                      maturity=0.5, observation_type=ObservationType.DISCRETE, observation_dates=[D / 2])
    theta = GreeksCalculator().calculate(product=p, pricing_env=env(), engine=BarrierAnalyticalEngine(),
                                         greeks=["theta"])["theta"]
    assert theta != 0.0
```

Add one test per remaining caller asserting it equals `numerical_theta` on the accumulator case: `execution/greeks` (through its public greeks entry used in `test/execution/` — find with `grep -rn "theta_degenerate" test/execution`), `portfolio/equity/position.py` (`_trade_theta` through the position's public greeks method), and the risk report's `theta_base` on a Snowball whose next KO date is `t1`.

- [ ] **Step 2: Run to verify they fail** (`ModuleNotFoundError: theta_shift`).

- [ ] **Step 3: Implement `theta_shift.py`**

```python
"""One theta step for every theta caller (spec 2026-10-07 theta boundary, R1-R4).

Decide the as-of date's own observation on the spot (the products' shared decision), carry its outcome, then move
the product with ``time_shift(..., keep_landing=True)``: the record on the bumped date is observed at the new as-of
instant; everything before it is gone.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

__all__ = ["ThetaShift", "shift_for_theta", "bumped_value"]


@dataclass(frozen=True)
class ThetaShift:
    product: object
    env: object
    lifecycle_state: object = None
    fixed_pv: float = 0.0
    terminated: bool = False


def _autocallable_outcome(product, env, state, time_bump):
    """(lifecycle_state, fixed_pv, terminated) from ``decide_observations_at_valuation``."""
    from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
    knocked_in = bool(getattr(state, "knocked_in", False))
    decision = product.decide_observations_at_valuation(env.spot, env, knocked_in=knocked_in)
    if decision.knocked_out:
        rec = decision.ko_record
        pay = float(rec.settlement_time if rec.settlement_time is not None else rec.observation_time)
        amount = product.ko_payment(decision, env)      # the KO cash the engines pay for this record; see note
        return None, (amount * float(env.get_discount_factor(pay)) / float(env.get_discount_factor(time_bump))
                      if pay > time_bump else 0.0), True
    if decision.knocked_in and not knocked_in:
        return AutocallableLifecycleState(knocked_in=True), 0.0, False
    return state, 0.0, False


def _barrier_outcome(product, env, state, time_bump):
    """The barrier family's records at the as-of instant, decided on the spot."""
    from quantark.asset.equity.engine.analytical.first_observation import observed_at_valuation, resolve_current_records
    from quantark.asset.equity.lifecycle.state import BarrierLifecycleState
    from quantark.util.enum import ObservationType
    if product.observation_type == ObservationType.CONTINUOUS:
        hit = product.is_barrier_hit(env.spot)
        records = []
    elif product.observation_type == ObservationType.DISCRETE:
        records = [r for r in observed_at_valuation(resolve_current_records(product.observation_schedule, env,
                                                                            product=product))
                   if product.is_barrier_hit(env.spot)]
        hit = bool(records)
    else:
        return state, 0.0, False
    if not hit:
        return state, 0.0, False
    knocks_in = getattr(product, "is_knock_in", False)
    if knocks_in:
        return BarrierLifecycleState(knocked_in=True), 0.0, False
    # knock-out / touch / sharkfin breach: the contract ends; its cash is a receivable if paid after the bump
    cash, pay = product.terminal_cash_on_hit(records[0] if records else None, env)
    return None, (cash * float(env.get_discount_factor(pay)) / float(env.get_discount_factor(time_bump))
                  if pay > time_bump else 0.0), True


def shift_for_theta(product, pricing_env, time_bump, bumped_date, *, lifecycle_state=None) -> ThetaShift:
    from quantark.asset.equity.product.option import (AccumulatorOption, KnockOutResetSnowballOption,
                                                      PhoenixOption, SnowballOption)
    if isinstance(product, (SnowballOption, PhoenixOption, KnockOutResetSnowballOption)):
        state, fixed_pv, terminated = _autocallable_outcome(product, pricing_env, lifecycle_state, time_bump)
    elif isinstance(product, AccumulatorOption):
        state, fixed_pv, terminated = lifecycle_state, 0.0, False
    else:
        state, fixed_pv, terminated = _barrier_outcome(product, pricing_env, lifecycle_state, time_bump)
    shifted, env = deepcopy(product), deepcopy(pricing_env)
    env.valuation_date = bumped_date
    if isinstance(shifted, AccumulatorOption):
        shifted.record_valuation_fixing(pricing_env.spot, pricing_env, time_bump)   # t0 shares into past_observations
    shifted.time_shift(time_bump, bumped_date, env, keep_landing=True)
    return ThetaShift(shifted, env, state, fixed_pv, terminated)


def bumped_value(engine, shift: ThetaShift) -> float:
    if shift.terminated:
        return shift.fixed_pv
    if shift.lifecycle_state is None:
        return float(engine.price(shift.product, shift.env)) + shift.fixed_pv
    return float(engine.price(shift.product, shift.env, lifecycle_state=shift.lifecycle_state)) + shift.fixed_pv
```

Product hooks this needs (add with their own unit tests in `test/theta_boundary/test_product_shift.py`):
- `ko_payment(decision, env) -> float` on Snowball/Phoenix/KO-reset: the amount the engines pay for `decision.ko_record` (read how `pde/core/adapters/snowball.py:84-91` computes the immediate KO payoff and move that computation to the product method, then call it from the adapter so both share it).
- `terminal_cash_on_hit(record, env) -> (amount, payment_time)` on the barrier family: knock-out rebate (record payoff for a discrete record) paid at the hit settlement (`pay_at_hit`) or the terminal payment time; one-touch rebate likewise; no-touch / double no-touch 0; sharkfins their knock-out cash. Multiplier included (the touch products: none).
- `AccumulatorOption.record_valuation_fixing(spot, env, time_bump)`: if the schedule has a record at the valuation instant (`observed_at_valuation`), append `(-time_bump, spot)` to `past_observations` (`get_realized_accrual` only sorts by time).
- `RangeAccrualOption.time_shift(time_bump, bumped_date, pricing_env, *, keep_landing=False)`: numeric `observation_time` records shift by `time_bump`; a record landing on the bumped instant (`is_zero`) under `keep_landing` gets `observation_time = 0.0` and, when its `observed_in_range is None`, `observed_in_range` decided on `pricing_env.spot` with the product's own range test; records crossing below zero without a fixing get it decided on the spot as well (spot is held in theta); dated records rely on the bumped `valuation_date` as today, with the landing record's fixing set the same way.

Callers: replace each `deepcopy` + `time_shift` + `if dropped_all` block with `shift = shift_for_theta(product, pricing_env, time_bump, bumped_date, lifecycle_state=<the caller's state or None>)` and `bumped_value(engine, shift)`:
- `numerical_theta`: `return bumped_value(engine, shift) - base_price`.
- `time_scenario`: store `shift` in the scenario dict (`"product": shift.product, "env": shift.env, "shift": shift`), drop the `dropped_all_observations` zero; `advanced_price` uses `bumped_value(engine, scenario["shift"])`.
- `greeks_calculator.py:836-845`: the comment drops "a time_shift that dropped every observation".
- `execution/greeks.py`: `prod2, env2` from `shift_for_theta(..., lifecycle_state=state.lifecycle_state if present)`; no `theta_degenerate` for the dropped-all case; legs shift with `leg.time_shift(time_bump, keep_landing=True)`.
- `portfolio/equity/position.py::_trade_theta`: same, with `self.lifecycle_state`.
- `autocallable_risk_report.py`: `theta_base` from `bumped_value`; `_compute_pv_surface_sq` / `_compute_delta_surface_sq` gain an optional `shift_at_spot` callable used to build each grid point's bumped product and state (`lambda env_s: shift_for_theta(product, env_s_at_t0, time_bump, bumped_date)` with `env_s_at_t0` the t0 env at that spot), so the carried outcome follows the grid spot.
- `cashleg/autocallable_leg.py::time_shift(dt, *, keep_landing=False)`: keep `obs - dt > Tolerance.ZERO`, and under `keep_landing` keep `is_zero(obs - dt)` at 0.0.

- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q -n 6 test/theta_boundary test/test_greeks_theta_schedule.py test/execution test/test_autocallable_leg_time_shift.py test/portfolio -k "theta or shift"`. Expected: Task 5 tests PASS; the pinned old-rule cases fail (restated in Task 7).

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/riskmeasures quantark/execution/greeks.py quantark/portfolio/equity/position.py quantark/asset/equity/report/autocallable_risk_report.py quantark/cashleg/autocallable_leg.py quantark/asset/equity/product/option quantark/asset/equity/engine/pde/core/adapters/snowball.py test/theta_boundary
git commit -m "feat(theta): one theta step - the landing observation kept, the as-of outcome carried, an ended schedule priced"
```

---

### Task 6: Legacy autocallable QUAD engines on an empty side

**Files:**
- Modify: `quad/snowball_quad_engine.py:204-205`, `:226-227`; `quad/phoenix_quad_engine.py:106-107`, `:125-126`; `quad/ko_reset_snowball_quad_engine.py:146`, `:158`, `:181`, `:566`
- Test: `test/theta_boundary/test_quad_empty_side.py`

- [ ] **Step 1: Failing test** — a Snowball whose knock-in dates have all passed (KI schedule shifted to empty by `time_shift(..., keep_landing=True)` past its last KI date) prices on `SnowballQuadEngine` (or the legacy class name found by `grep -n "^class" quantark/asset/equity/engine/quad/snowball_quad_engine.py`) equal to the same product on `SnowballPDESolver` within the two engines' measured agreement on the alive product (measure both on the unshifted product first; tolerance = 2 × that gap), and likewise Phoenix and KO-reset.
- [ ] **Step 2: Run, verify FAIL** (PricingError on the empty side).
- [ ] **Step 3: Implement** — where each engine raises on an empty KO or KI side, fold the empty side into its time set as QUAD V2 does (`quad/v2/adapters.py:160-193`): no KI side ⇒ no KI monitoring; no KO side ⇒ no autocall before maturity.
- [ ] **Step 4: Run PASS**, plus `test/quad` subset `-k "snowball or phoenix or reset"`.
- [ ] **Step 5: Commit** `fix(quad): legacy autocallable QUAD engines price an empty knock-in or knock-out side`.

---

### Task 7: Restated tests, goldens and the byte book

**Files:** `test/test_greeks_theta_schedule.py`, `test/test_autocallable_leg_time_shift.py`, and every failing test from the full run; `test/replay_golden/data/scalar_bsm_greeks.csv`, `localvol_greeks.csv`; `test/pde_bitwise` comparison (no manifest change expected).

- [ ] **Step 1:** Run the full suite (`-n 6`, RAM guard). List every failure.
- [ ] **Step 2:** For each failure decide, in writing in the commit message: (a) it pinned the old boundary rule (record on the bumped date dropped / theta 0 on an emptied schedule / landing leg dropped) → restate to the new rule with before → after values; (b) anything else → a regression: fix the code, not the test.
- [ ] **Step 3:** Replay goldens: regenerate only the theta columns with the golden capture tool (`test/replay_golden/capture.py`, loaded under its own module name — the documented `from test.replay_golden...` import fails because the stdlib `test` wins); diff old vs new; every moved row listed in the commit message with old/new theta; `book_greeks.csv` and book state rows must be byte-identical (lifecycle aging unchanged) — if not, stop and investigate.
- [ ] **Step 4:** PDE byte book: capture base (branch tip before Task 1) vs head with `test/pde_bitwise/capture_book.py` + `compare.py`; only `bump_theta` fields of rows whose next observation is one bump away may move; list them.
- [ ] **Step 5:** Commit (`test(theta): restate the boundary-pinned theta tests and regenerate the replay theta goldens`), with the lists.

---

### Task 8: Docs, changelog, full suite

- [ ] **Step 1:** `CHANGELOG.md` `[Unreleased]`: Changed — theta keeps the observation on the bumped date (daily accumulator +55,402 → its new value, measured), carries the as-of outcome, prices an ended schedule (no theta 0); `time_shift(..., keep_landing=)` and `ObservationSchedule.monitoring_ended` added; product `time_shift` returns `None`; Fixed — legacy autocallable QUAD engines on an empty side; range accrual numeric records now shift.
- [ ] **Step 2:** `docs/engine_param_guide.md` / `docs/` theta text: `grep -rn "time_shift\|theta" docs/*.md | grep -i "observ"` and restate any that describe the old rule (`git add -f` for docs/).
- [ ] **Step 3:** Full suite `-n 6` under the RAM guard; only the 4 known failures allowed; restore the two mo sample files.
- [ ] **Step 4:** Commit `docs(theta): changelog and guide for the observation boundary`.
