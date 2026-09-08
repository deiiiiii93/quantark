# Simulated-Path Backtest — Plan 3 of 4: Ladder and Life-Surface Providers, Disk Cache, Gate, Batching

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the ensemble engine fast enough for thousands of paths without giving up the exact mode as the reference: a spot-ladder mode for the repricing provider, a PDE life-surface provider that solves once per bucket and reads along the path, an on-disk cache tier, the sampling accuracy gate that measures every approximation against direct repricing, and path batching over a spawn process pool.

**Architecture:** Every approximation is opt-in by a config field that is absent until this plan adds it, and every approximation is measured: the provider keeps a deterministic reservoir of the states it priced, and at the end of the run reprices them exactly and reports the worst gap in bp of notional and hands. The life surface needs one change outside the package — `GridRequest.extra_times`, time nodes that are exact grid nodes but carry no event and no damping — plus a public seam on `SnowballPDESolver` that runs one solve and hands back the two value slabs. Batching splits a `MarketPath` into index ranges, runs each through the plan-2 engine (in process or in a `spawn` pool) and concatenates the results; the split is bit-inert, and a test pins that.

**Tech Stack:** Python 3.10–3.13, numpy, pandas, pytest (`-n0` while iterating; parallel default for the final run), `concurrent.futures.ProcessPoolExecutor` with the `spawn` context, `fcntl` advisory locks for shard writes (POSIX; CI is Linux, dev is macOS).

**Spec:** `docs/superpowers/specs/2026-09-08-simulated-path-backtest-design.md`, sections 3 (the grid-layer change), 7.2, 7.3 (ladder mode), 7.4 (disk tier), 7.5, 11 (`workers`, `batch_paths`, `spot_step`, `vol_step`, `q_step`), 12, 13 (life surface, repricing ladder and disk items) and the plan-3/4 lines of 16. Plans 1 and 2 are implemented on the branch (`docs/superpowers/plans/2026-09-08-simulated-paths-plan{1,2}-*.md`); plan 2's README section "How a state gets priced" describes the exact provider this plan extends.

## Global Constraints

- Work on branch `feat/simulated-path-backtest`. Verify `git log --oneline -1` shows `7cdfbd37` or later before starting. Run tests as `/Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 <files>` (add `PYTHONPATH=$PWD` in a worktree so its source shadows the editable install).
- Commits: every message ends with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Never `git add example/` wholesale (two `example/mo_volmodels/data/` sample files churn under test runs; `git checkout -- example/mo_volmodels/data/` before staging). Files under `docs/` need `git add -f`. `CLAUDE.md` files are never added.
- **No behaviour change for existing callers of the PDE grid layer or `SnowballPDESolver`.** Task 6 adds a defaulted field; Task 7 adds a method and a class attribute whose default reproduces today's `grid_request` byte for byte. `test/pde_grid`, `test/test_snowball_pde*.py`, `test/replay_golden`, `test/simulation` and `test/test_lifecycle_calendar_schedule.py` must stay green; the plan-2 exact-mode oracle tests must keep reporting gaps of exactly `0.0`.
- Fail closed (spec §12): a spot outside a surface grid, a gate over tolerance, a foreign cache shard, a worker failure — each raises `ValidationError` (or the `GateFailure` subclass) with the offending state in the message. Never substitute a provider, engine, seed or tolerance.
- Exact semantics stay the default: `spot_step=None`, `provider="repricing"` and `workers=1` reproduce plan 2 bit for bit. Every approximation is an explicit, validated setting.
- Determinism does not depend on `PYTHONHASHSEED` or on iteration order: the gate's reservoir sampler is seeded from `blake2b` of the provider's fingerprints; bucket indices are `np.round(value / step)` (half-to-even, the same on every machine).
- Use `quantark.util.numerical` helpers (`is_zero`, `is_close`) in library code, EXCEPT where a rule is copied verbatim from the replay or the settlement resolver for bit-identity (`POSITION_EPS`, `_TIME_TOLERANCE`); those are named and commented.
- Style: PEP 8, dataclasses with type hints, docstrings on all public APIs, `from __future__ import annotations` at the top of every new module.

---

## Scope of this plan

**In:** numeric-clock settlement fidelity (the plan-2 TODO), config fields for the approximate modes and batching, ladder mode, the disk cache tier, the sampling gate, `GridRequest.extra_times`, `SnowballPDESolver.solve_life_surface`, `LifeSurfacePricer`, engine and oracle wiring for approximate providers, batching over a spawn pool, exports and README.

**Out, by design:** `EnsembleResults.summary` / `distribution` / `paired` / `to_dir` / `from_dir` and the example study (plan 4). Multi-cell scheduling (`for path batch: for cell:` in spec §7.4) is a plan-4 runner concern; this plan's runner batches one cell, and the disk tier is what lets a second cell reuse the first cell's states.

**Three facts established while writing this plan, each pinned by a test:**

1. **The PDE engine refuses settlement conventions** (`PDEEngine settlement support is none; request requires event_and_terminal`). So no oracle test can exercise a delayed-settlement snowball; Task 1 pins the numeric-clock rule against the tracker's ledger directly, with no pricing engine in the loop.
2. **No equity engine or product reads `PricingEnvironment.basis_yield`** (`grep get_basis_yield quantark/asset/equity` is empty). A ladder node's environment therefore carries no basis, and the state key omits it; if an engine ever starts reading it, `verify` will show the gap.
3. **`SnowballPDESolver._solve` fills the whole `(n_x, n_t)` slab for both surfaces** (`_time_stepping_two_surface` writes column `j` of `grid_v0`/`grid_v1` at every step, and applies the event transforms to that column). A life surface is one solve plus two array copies; no re-march is needed.

One decision the spec leaves open: the life surface's solve needs a flat rate, and the `MarketPath` rate is per path per day. Task 8 buckets the rate with `q_step` (both are yields) and records that in the README.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `quantark/asset/equity/lifecycle/autocallable.py` (modify) | `CalendarSchedule` gains the numeric-clock fields | 1 |
| `quantark/backtest/simulation/lifecycle.py` (modify) | `paid`/`payment_time` arrays, both-clock settlement, `receivable_pv` | 1 |
| `quantark/backtest/simulation/engine.py` (modify) | delegate `_receivable_pv`; providers; gate wiring; flush; manifest | 1, 5, 9 |
| `quantark/backtest/simulation/config.py` (modify) | `spot_step`, `vol_step`, `q_step`, `surface_cache_bytes`, `disk_dir`, `workers`, `batch_paths` | 2 |
| `quantark/backtest/simulation/pricing/base.py` (modify) | `bucket_key`, `bucket_centre`, `GateScale`, `GateFailure`, `state_row`, protocol update | 3, 5 |
| `quantark/backtest/simulation/pricing/repricing.py` (modify) | ladder mode; reservoir; `verify` | 3, 5 |
| `quantark/backtest/simulation/pricing/cache.py` (modify) | `DiskTier`, flush, promotion | 4 |
| `quantark/asset/equity/engine/pde/grid/request.py` (modify) | `GridRequest.extra_times` | 6 |
| `quantark/asset/equity/engine/pde/grid/time.py` (modify) | extra nodes without damping | 6 |
| `quantark/asset/equity/engine/pde/base_pde_solver.py` (modify) | `LifeSurfaceSolution` | 7 |
| `quantark/asset/equity/engine/pde/snowball_pde_solver.py` (modify) | `_extra_time_nodes`, `solve_life_surface` | 7 |
| `quantark/asset/equity/engine/pde_engine.py` (modify) | `solve_life_surface` facade | 7 |
| `quantark/backtest/simulation/pricing/surface.py` (create) | `LifeSurface`, `SurfaceCache`, `LifeSurfacePricer` | 8 |
| `quantark/backtest/simulation/conformance.py` (modify) | `contracts_tolerance` | 9 |
| `quantark/backtest/simulation/paths/market_path.py` (modify) | `MarketPath.take` | 10 |
| `quantark/backtest/simulation/runner.py` (create) | `run_ensemble`, `concat_results`, spawn pool | 10 |
| `quantark/backtest/simulation/__init__.py`, `README.md` (modify) | exports, guide | 11 |
| `test/simulation/conftest.py` (modify) | `settlement_lag_days`, ladder/surface config helpers | 1, 2 |
| `test/simulation/test_lifecycle.py` (modify) | Task 1 | 1 |
| `test/simulation/test_config.py` (modify) | Task 2 | 2 |
| `test/simulation/test_ladder.py` (create) | Task 3 | 3 |
| `test/simulation/test_cache.py` (modify) | Task 4 | 4 |
| `test/simulation/test_gate.py` (create) | Task 5 | 5 |
| `test/pde_grid/test_time_builder.py` (modify) | Task 6 | 6 |
| `test/test_snowball_life_surface.py` (create) | Task 7 | 7 |
| `test/simulation/test_surface.py` (create) | Task 8 | 8 |
| `test/simulation/test_engine.py`, `test_conformance.py` (modify) | Task 9 | 9 |
| `test/simulation/test_runner.py` (create) | Task 10 | 10 |
| `test/simulation/test_market_path.py` (modify) | Task 10, 11 | 10, 11 |

Conventions from plan 2 hold: per-path arrays `(n_paths,)`, per-product-per-path `(n_products, n_paths)`, the cube `(n_paths, n_days)`; `p` products, `i` paths, `d` days, `j` ladder nodes, `k` time nodes.

---

### Task 1: Numeric-clock settlement fidelity

**Files:**
- Modify: `quantark/asset/equity/lifecycle/autocallable.py` (three fields on `CalendarSchedule`, four lines in `resolve_calendar_schedule`)
- Modify: `quantark/backtest/simulation/lifecycle.py`
- Modify: `quantark/backtest/simulation/engine.py` (`_receivable_pv` delegates)
- Modify: `test/simulation/conftest.py`, `test/simulation/test_lifecycle.py`

**The two rules the replay applies to a delayed terminal cashflow**, read off `AutocallableLifecycleState`, `LifecycleCashflowLedger` and `ProductReplay`:

- `realized_cashflows` is `ledger.paid_total(valuation_point)`; a flow is *paid* once it is not pending, and `_is_pending` is `payment_date > point.date` on the date clock or `payment_time > point.time` on the numeric clock. `observe` sets `valuation_point` to today at the top of every day, so on the numeric clock `point.time = max(0, elapsed_days / 365)` and `payment_time = determination_time + (settlement_time − observation_time)` with `determination_time` the point's time on the event day (`_record_cashflow`, `_timing_cashflow`).
- `settled` is date-keyed on both clocks: `settle_pending_if_due` fires when `date >= settlement_date`, and `settle()` then advances the numeric point past every pending payment time (so a flow one ulp short of paid becomes paid on the settlement day) and zeroes `pending_settlement_cashflow`. A flow that is not delayed (`payment > determination` false) is settled and paid on its event day.
- `pending_receivable_pv` discounts the date clock's scalar with `(settlement − date).days / 365` and the numeric clock's ledger flows with `payment_time − point.time`, refusing a flow whose remaining time is at or below `_TIME_TOLERANCE` (`SettlementResolver.resolve_pending`).

Plan 2 implemented only the date-clock reading and flagged this. Every fixture there settles at determination, so it was invisible; a year-fraction settlement lag makes it visible, and the tracker walk is the oracle.

**Interfaces:**
- `CalendarSchedule` gains `uses_date_timing: bool`, `ko_settlement_delay: np.ndarray` (`(n_ko,)` years), `terminal_settlement_delay: float`.
- `VectorLifecycle` gains arrays `cashflow`, `paid`, `determination_time`, `payment_time` (all `(n_products, n_paths)`); `realized` keeps its meaning (paid total); `pending` mirrors `pending_settlement_cashflow`; `settled` mirrors `settled`.
- `receivable_pv(lifecycle: VectorLifecycle, rate: np.ndarray, day_index: int) -> np.ndarray` (module-level in `lifecycle.py`), the replay's `pending_receivable_pv` per path.

- [ ] **Step 1: Write the failing tests**

Append a lag option to `short_snowball` in `test/simulation/conftest.py` (replace the function):

```python
def short_snowball(maturity_days: int = 6, *, ko_days=(2, 5), ki_days=(1, 3), continuous_ki: bool = False,
                   settlement_lag_days: int = 0):
    """A snowball short enough to run a whole life inside a test.

    ``settlement_lag_days`` puts a YEAR_FRACTION settlement lag on the
    contract (the numeric clock: the schedule is in year fractions, so the
    tracker keeps a time-based valuation point).  The factory does not take
    a convention, so it is set after construction and re-validated.
    """
    from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit

    product = create_standard_snowball(
        initial_price=SPOT, strike=SPOT, maturity=maturity_days / 365.0, contract_multiplier=1.0,
        ko_barrier=KO_BARRIER, ki_barrier=KI_BARRIER, ko_rate=0.20, num_observations=len(ko_days),
        ko_observation_dates=[d / 365.0 for d in ko_days],
        ki_observation_type=ObservationType.CONTINUOUS if continuous_ki else ObservationType.DISCRETE,
        ki_continuous=continuous_ki,
        ki_observation_dates=None if continuous_ki else [d / 365.0 for d in ki_days],
        include_principal=True,
    )
    if settlement_lag_days:
        product.settlement_convention = SettlementConvention(
            lag=settlement_lag_days / 365.0, lag_unit=SettlementLagUnit.YEAR_FRACTION,
        )
        product._validate_settlement_convention()
    return product
```

Append to `test/simulation/test_lifecycle.py`:

```python
from quantark.backtest.simulation.lifecycle import receivable_pv


@pytest.mark.parametrize("name, spots", [
    ("ko_then_two_days_of_receivable", [SPOT, SPOT, KO + 1.0, SPOT, SPOT, SPOT, SPOT, SPOT]),
    ("late_ko_settles_after_maturity", [SPOT] * 5 + [KO + 1.0, SPOT, SPOT]),
    ("maturity_with_a_lag", [SPOT] * 8),
    ("ki_then_maturity_with_a_lag", [SPOT, KI - 1.0, SPOT, SPOT, SPOT, SPOT, SPOT * 0.8, SPOT * 0.8]),
])
def test_a_settlement_lag_matches_the_tracker_day_by_day(name, spots):
    product = short_snowball(settlement_lag_days=2)
    expected = _walk_tracker(product, spots)
    actual = _walk_vector(product, spots)
    assert any(day["pending"] != 0.0 for day in expected), "the fixture must actually park a receivable"
    for d, (e, a) in enumerate(zip(expected, actual)):
        assert a == pytest.approx(e), f"{name} day {d}: {a} != {e}"


def test_the_receivable_pv_matches_the_ledger_on_the_numeric_clock():
    product = short_snowball(settlement_lag_days=2)
    spots = [SPOT, SPOT, KO + 1.0, SPOT, SPOT, SPOT, SPOT, SPOT]
    tracker = _tracker(product)
    lifecycle_product = tracker.product_for_lifecycle()
    vec = _vector(product)
    rate = np.array([0.02])
    seen_pending = False
    for d, (day, spot) in enumerate(zip(DATES, spots)):
        env = _env(day)
        tracker.observe(day, lifecycle_product, env, float(spot))
        tracker.settle_maturity_if_due(day, lifecycle_product, env, float(spot))
        if not tracker.lifecycle.settled and tracker.lifecycle.settlement_date is not None:
            if pd.Timestamp(day).normalize() >= pd.Timestamp(tracker.lifecycle.settlement_date).normalize():
                tracker.lifecycle.settle()
        vec.step(d, np.array([float(spot)]), lambda p, paths, ki: np.zeros(paths.size))
        point = tracker.lifecycle.valuation_point
        expected = float(tracker.lifecycle.ledger.pending_pv(point, env)) if point is not None else 0.0
        got = receivable_pv(vec, rate, d)
        assert got[0] == expected, f"day {d}: {got[0]!r} != {expected!r}"
        seen_pending |= expected != 0.0
    assert seen_pending


def test_the_schedule_carries_the_delays():
    product = short_snowball(settlement_lag_days=2)
    schedule = _tracker(product).resolve_calendar_schedule(DATES, _env(DATES[0]))
    assert schedule.uses_date_timing is False
    assert schedule.ko_settlement_delay == pytest.approx([2.0 / 365.0] * 2)
    assert schedule.terminal_settlement_delay == pytest.approx(2.0 / 365.0)
    plain = _tracker(short_snowball()).resolve_calendar_schedule(DATES, _env(DATES[0]))
    assert plain.ko_settlement_delay == pytest.approx([0.0, 0.0])
    assert plain.terminal_settlement_delay == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_lifecycle.py -q`
Expected: `ImportError: cannot import name 'receivable_pv'`.

- [ ] **Step 3: Extend `CalendarSchedule` and its resolver**

In `quantark/asset/equity/lifecycle/autocallable.py`, append three fields to `CalendarSchedule` (after `disable_ko_after_ki`):

```python
    uses_date_timing: bool
    ko_settlement_delay: np.ndarray
    terminal_settlement_delay: float
```

and extend the docstring with one paragraph:

```
    ``uses_date_timing`` says which clock the tracker keeps for this
    product (``_uses_date_timing``).  On the numeric clock a terminal
    cashflow's payment time is its determination time plus the record's
    ``settlement_time - time`` (``ko_settlement_delay`` per observation,
    ``terminal_settlement_delay`` for maturity), and that is what the ledger
    compares the valuation point against.
```

In `resolve_calendar_schedule`, change the KO loop and the return:

```python
        ko_due, ko_barrier, ko_payoff, ko_settle, ko_delay = [], [], [], [], []
        for rec in ko_records:
            due = first_due(lambda day, vp, rec=rec: self._record_is_due(day, vp, rec))
            ko_due.append(due)
            ko_barrier.append(float(rec["barrier"]) if rec["barrier"] is not None else np.nan)
            ko_payoff.append(float(rec["payoff"]))
            ko_settle.append(settlement_position(rec["settlement_date"], due))
            ko_delay.append(float(rec["settlement_time"]) - float(rec["time"]))
```

```python
        terminal_delay = float(timing.payment_time) - float(timing.determination_time)
        return CalendarSchedule(
            ...  # existing fields unchanged
            disable_ko_after_ki=bool(
                getattr(product.barrier_config, "disable_ko_after_ki", False)
            ),
            uses_date_timing=bool(self._uses_date_timing(product)),
            ko_settlement_delay=np.array(ko_delay, dtype=float),
            terminal_settlement_delay=terminal_delay,
        )
```

- [ ] **Step 4: Rewrite the settlement side of `VectorLifecycle`**

In `quantark/backtest/simulation/lifecycle.py`, add the import and the tolerance:

```python
from quantark.asset.equity.settlement import _TIME_TOLERANCE  # the resolver's own bound, for bit-identity
from quantark.param import FlatRateCurve
```

Add arrays in `__init__` (after `self.ko_index`):

```python
        self.cashflow = np.zeros(shape)                  # the terminal amount, quantity-signed
        self.paid = np.zeros(shape, dtype=bool)           # the ledger has paid it
        self.determination_time = np.full(shape, np.nan)  # numeric clock, years from day 0
        self.payment_time = np.full(shape, np.nan)
```

Add a helper:

```python
    @staticmethod
    def elapsed_time(schedule: CalendarSchedule, d: int) -> float:
        """``AutocallableLifecycleTracker._valuation_point``'s time for day ``d``."""
        days = (schedule.dates[d].normalize() - schedule.dates[0].normalize()).days
        return max(0.0, days / 365.0)
```

Replace the body of `_knock_out` from `cashflow = ...` to the end of the `hit` block with:

```python
            cashflow = float(self.quantities[p]) * float(schedule.ko_payoff[idx])
            self._terminate(p, schedule, d, hit, cashflow,
                            settle_day=int(schedule.ko_settlement_day[idx]),
                            delay=float(schedule.ko_settlement_delay[idx]), knocked_out=True)
            self.ko_index[p][hit] = idx
            today_ko |= hit
```

Replace the body of `_maturity` after `payoffs = ...` with:

```python
        cashflows = float(self.quantities[p]) * payoffs
        for i, c in zip(due, cashflows):
            one = np.zeros(self.n_paths, dtype=bool)
            one[int(i)] = True
            self._terminate(p, schedule, d, one, float(c),
                            settle_day=int(schedule.terminal_settlement_day),
                            delay=float(schedule.terminal_settlement_delay), knocked_out=False)
        return [
            LifecycleRecord(p, int(i), d, "maturity", -1, float(spot[i]), float("nan"), float(c))
            for i, c in zip(due, cashflows)
        ]
```

Add `_terminate` and replace `_settle`:

```python
    def _terminate(self, p, schedule, d, mask, cashflow: float, *, settle_day: int,
                   delay: float, knocked_out: bool) -> None:
        """Book a terminal cashflow the way ``mark_ko`` / ``mark_maturity`` do.

        A flow whose payment is not after its determination is paid and
        settled on the event day; a delayed one is parked in ``pending``
        until the settlement day, and on the numeric clock the ledger pays
        it as soon as the valuation point reaches its payment time.
        """
        if knocked_out:
            self.knocked_out[p][mask] = True
        else:
            self.matured[p][mask] = True
        self.alive[p][mask] = False
        self.terminal_day[p][mask] = d
        self.settlement_day[p][mask] = settle_day
        self.cashflow[p][mask] = cashflow
        det = self.elapsed_time(schedule, d)
        self.determination_time[p][mask] = det
        self.payment_time[p][mask] = det + delay        # _record_cashflow: determination + delay
        if schedule.uses_date_timing:
            delayed = settle_day > d                    # payment_date > determination_date
        else:
            delayed = (det + delay) > det               # payment_time > determination_time
        if delayed:
            self.pending[p][mask] = cashflow
        else:
            self.paid[p][mask] = True
            self.settled[p][mask] = True
            self.realized[p][mask] += cashflow

    def _settle(self, p, schedule, d) -> List[LifecycleRecord]:
        live = self.terminal_day[p] >= 0
        by_date = live & (self.settlement_day[p] >= 0) & (self.settlement_day[p] <= d)
        # The ledger's paid test: on the date clock payment_date <= today,
        # on the numeric clock NOT (payment_time > point.time).
        if schedule.uses_date_timing:
            paid_now = by_date
        else:
            cur = self.elapsed_time(schedule, d)
            paid_now = live & ~(self.payment_time[p] > cur)
        newly = paid_now & ~self.paid[p]
        self.paid[p][newly] = True
        self.realized[p][newly] += self.cashflow[p][newly]
        # settle_pending_if_due is date-keyed on both clocks; settle() then
        # moves the numeric point past the payment time, so a flow one ulp
        # short of paid is paid on its settlement day.
        landing = ~self.settled[p] & by_date
        if not landing.any():
            return []
        late = landing & ~self.paid[p]
        self.paid[p][late] = True
        self.realized[p][late] += self.cashflow[p][late]
        amounts = self.pending[p][landing]
        self.pending[p][landing] = 0.0
        self.settled[p][landing] = True
        return [
            LifecycleRecord(p, int(i), d, "settlement", -1, float("nan"), float("nan"), float(a))
            for i, a in zip(np.flatnonzero(landing), amounts)
        ]
```

Update `step` to pass the schedule: `records += self._settle(p, schedule, day_index)`.

Add the module-level function (after the class):

```python
def receivable_pv(lifecycle: VectorLifecycle, rate: np.ndarray, day_index: int) -> np.ndarray:
    """Determined-but-unpaid cash discounted the replay's way, per path.

    Date clock: the parked scalar through ``(settlement - date).days / 365``
    on the day's flat curve (``ProductReplay.pending_receivable_pv``).
    Numeric clock: every unpaid ledger flow through ``payment_time -
    point.time`` (``SettlementResolver.resolve_pending``), which refuses a
    remaining time at or below ``_TIME_TOLERANCE`` exactly as the resolver
    does -- a state the replay would abort on is not silently zeroed here.
    """
    out = np.zeros(rate.size)
    for p, schedule in enumerate(lifecycle.schedules):
        if schedule.uses_date_timing:
            live = np.flatnonzero((lifecycle.pending[p] != 0.0) & ~lifecycle.settled[p])
            for i in live:
                i = int(i)
                settle_day = int(lifecycle.settlement_day[p][i])
                if settle_day < 0:
                    continue
                tau = max((schedule.dates[settle_day] - schedule.dates[day_index]).days / 365.0, 0.0)
                out[i] += float(lifecycle.pending[p][i]) * float(
                    FlatRateCurve(rate=float(rate[i])).get_discount_factor(tau)
                )
            continue
        cur = lifecycle.elapsed_time(schedule, day_index)
        live = np.flatnonzero((lifecycle.terminal_day[p] >= 0) & ~lifecycle.paid[p])
        for i in live:
            i = int(i)
            tau = float(lifecycle.payment_time[p][i]) - cur
            if not np.isfinite(tau) or tau <= _TIME_TOLERANCE:
                raise ValidationError(
                    f"pending cashflow of product {p} on path {i} must pay strictly after "
                    f"day {day_index} (remaining time {tau!r})"
                )
            out[i] += float(lifecycle.cashflow[p][i]) * float(
                FlatRateCurve(rate=float(rate[i])).get_discount_factor(tau)
            )
    return out
```

In `quantark/backtest/simulation/engine.py`, replace the body of `_receivable_pv` with a delegation and drop its TODO docstring:

```python
    def _receivable_pv(self, lifecycle, rate, d, schedules) -> np.ndarray:
        """``ProductReplay.pending_receivable_pv`` per path, both clocks (see ``receivable_pv``)."""
        return receivable_pv(lifecycle, rate, d)
```

and import it: `from .lifecycle import LifecycleRecord, VectorLifecycle, receivable_pv`.

- [ ] **Step 5: Run the tests and the regressions**

Run: `.venv/bin/python -m pytest -n0 test/simulation test/test_lifecycle_calendar_schedule.py test/replay_golden -q`
Expected: all pass, and the plan-2 oracle tests still report `0.0` gaps (the undelayed fixture takes the "paid at determination" branch, which is what plan 2 did implicitly).

- [ ] **Step 6: Commit**

```bash
git add quantark/asset/equity/lifecycle/autocallable.py quantark/backtest/simulation/lifecycle.py \
        quantark/backtest/simulation/engine.py test/simulation/conftest.py test/simulation/test_lifecycle.py
git commit -m "feat(backtest/simulation): delayed settlement on both lifecycle clocks, pinned to the ledger

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Configuration for the approximate modes and batching

**Files:**
- Modify: `quantark/backtest/simulation/config.py`
- Modify: `test/simulation/conftest.py`, `test/simulation/test_config.py`

**Interfaces:**
- `CacheConfig(memory_bytes: int, disk_dir: Optional[str] = None)`.
- `PricingProviderConfig(provider: Literal["repricing", "life_surface"], cache, gate, spot_step=None, vol_step=None, q_step=None, surface_cache_bytes=None)`; property `mode -> Literal["exact", "ladder", "life_surface"]`.
- `EnsembleConfig` gains `workers: int = 1`, `batch_paths: Optional[int] = None`.
- Validation matrix (every violation a `ValidationError`):

| provider | spot_step | vol_step / q_step | surface_cache_bytes | engine |
|---|---|---|---|---|
| repricing | None | must be None | must be None | any |
| repricing | > 0 | both required, ≥ 0 (0 = exact) | must be None | any |
| repricing | ≤ 0 | — | — | rejected |
| life_surface | must be None | both required, ≥ 0 | required, > 0 | `EngineType.PDE` only |

- [ ] **Step 1: Write the failing tests**

Add two helpers to `test/simulation/conftest.py` (after `ensemble_config`):

```python
def ladder_pricing(spot_step: float = 0.002, vol_step: float = 0.0, q_step: float = 0.0, **cache):
    from quantark.backtest.simulation.config import CacheConfig, GateConfig, PricingProviderConfig

    return PricingProviderConfig(
        provider="repricing", cache=CacheConfig(memory_bytes=8_000_000, **cache),
        gate=GateConfig(sample_states=8, pv_tolerance_bp=5.0, delta_tolerance_hands=1.0),
        spot_step=spot_step, vol_step=vol_step, q_step=q_step,
    )


def surface_pricing(vol_step: float = 0.01, q_step: float = 0.0025, **cache):
    from quantark.backtest.simulation.config import CacheConfig, GateConfig, PricingProviderConfig

    return PricingProviderConfig(
        provider="life_surface", cache=CacheConfig(memory_bytes=8_000_000, **cache),
        gate=GateConfig(sample_states=8, pv_tolerance_bp=50.0, delta_tolerance_hands=3.0),
        vol_step=vol_step, q_step=q_step, surface_cache_bytes=200_000_000,
    )
```

Append to `test/simulation/test_config.py`:

```python
from quantark.util.enum.engine_enums import EngineType

from .conftest import ladder_pricing, pde_engine_config, surface_pricing


def _gate():
    return GateConfig(sample_states=0, pv_tolerance_bp=0.0, delta_tolerance_hands=0.0)


def test_exact_mode_is_the_default_and_rejects_bucket_steps():
    exact = PricingProviderConfig(provider="repricing", cache=CacheConfig(memory_bytes=1024), gate=_gate())
    assert exact.mode == "exact"
    with pytest.raises(ValidationError):
        PricingProviderConfig(provider="repricing", cache=CacheConfig(memory_bytes=1024), gate=_gate(), vol_step=0.01)


def test_ladder_mode_requires_both_bucket_steps_and_a_positive_spot_step():
    assert ladder_pricing().mode == "ladder"
    with pytest.raises(ValidationError):
        PricingProviderConfig(provider="repricing", cache=CacheConfig(memory_bytes=1024), gate=_gate(), spot_step=0.002)
    with pytest.raises(ValidationError):
        ladder_pricing(spot_step=0.0)
    with pytest.raises(ValidationError):
        ladder_pricing(vol_step=-0.01)
    with pytest.raises(ValidationError):
        PricingProviderConfig(provider="repricing", cache=CacheConfig(memory_bytes=1024), gate=_gate(),
                              spot_step=0.002, vol_step=0.0, q_step=0.0, surface_cache_bytes=10)


def test_life_surface_requires_its_budget_and_a_pde_engine():
    assert surface_pricing().mode == "life_surface"
    with pytest.raises(ValidationError):
        PricingProviderConfig(provider="life_surface", cache=CacheConfig(memory_bytes=1024), gate=_gate(),
                              vol_step=0.01, q_step=0.0025)
    with pytest.raises(ValidationError):
        PricingProviderConfig(provider="life_surface", cache=CacheConfig(memory_bytes=1024), gate=_gate(),
                              vol_step=0.01, q_step=0.0025, surface_cache_bytes=1024, spot_step=0.002)
    with pytest.raises(ValidationError):
        ensemble_config(pricing=surface_pricing(),
                        engine_config=pde_engine_config(pricing_engine_type=EngineType.QUADRATURE))


def test_batching_fields_are_validated():
    cfg = ensemble_config(workers=2, batch_paths=8)
    assert (cfg.workers, cfg.batch_paths) == (2, 8)
    with pytest.raises(ValidationError):
        ensemble_config(workers=0)
    with pytest.raises(ValidationError):
        ensemble_config(batch_paths=0)


def test_a_disk_dir_is_optional_and_kept(tmp_path):
    cache = CacheConfig(memory_bytes=1024, disk_dir=str(tmp_path))
    assert cache.disk_dir == str(tmp_path)
    assert CacheConfig(memory_bytes=1024).disk_dir is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_config.py -q`
Expected: `TypeError: ... unexpected keyword argument 'vol_step'` (and `disk_dir`, `workers`).

- [ ] **Step 3: Write the implementation**

Replace the three provider-side classes in `quantark/backtest/simulation/config.py` (keep `GateConfig` as is) and extend `EnsembleConfig`:

```python
PROVIDERS = ("repricing", "life_surface")
MODES = ("exact", "ladder", "life_surface")


@dataclass(frozen=True)
class CacheConfig:
    """State-cache budget (spec 7.4).

    ``disk_dir`` names the on-disk tier; ``None`` means memory only.  A
    shard written by another library version or engine is a miss, never
    reinterpreted (see ``pricing.cache.DiskTier``).
    """

    memory_bytes: int
    disk_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if int(self.memory_bytes) <= 0:
            raise ValidationError("CacheConfig.memory_bytes must be positive")
        if self.disk_dir is not None and not str(self.disk_dir):
            raise ValidationError("CacheConfig.disk_dir must be a directory path or None")


@dataclass(frozen=True)
class PricingProviderConfig:
    """Which pricer runs, how coarsely, and how much it may spend.

    ``spot_step`` (log-spot) switches the repricing provider from exact to
    ladder mode; ``vol_step`` / ``q_step`` bucket vol and the flat dividend
    yield in either approximate mode (0 keeps them exact); the life
    surface additionally needs ``surface_cache_bytes``.  A step that a
    mode cannot honour is rejected, not ignored.
    """

    provider: Literal["repricing", "life_surface"]
    cache: CacheConfig
    gate: GateConfig
    spot_step: Optional[float] = None
    vol_step: Optional[float] = None
    q_step: Optional[float] = None
    surface_cache_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValidationError(f"provider must be one of {PROVIDERS}, got {self.provider!r}")
        steps_given = self.vol_step is not None or self.q_step is not None
        if self.provider == "repricing":
            if self.surface_cache_bytes is not None:
                raise ValidationError("surface_cache_bytes belongs to the life_surface provider")
            if self.spot_step is None:
                if steps_given:
                    raise ValidationError(
                        "vol_step / q_step have no meaning in exact repricing mode; set spot_step for the ladder"
                    )
                return
            if float(self.spot_step) <= 0.0:
                raise ValidationError("spot_step must be positive (log-spot) or None for exact mode")
            self._require_bucket_steps("ladder")
            return
        if self.spot_step is not None:
            raise ValidationError("spot_step has no meaning for the life_surface provider")
        self._require_bucket_steps("life_surface")
        if self.surface_cache_bytes is None or int(self.surface_cache_bytes) <= 0:
            raise ValidationError("life_surface requires a positive surface_cache_bytes")

    def _require_bucket_steps(self, mode: str) -> None:
        if self.vol_step is None or self.q_step is None:
            raise ValidationError(f"{mode} mode requires vol_step and q_step (0 means exact)")
        if float(self.vol_step) < 0.0 or float(self.q_step) < 0.0:
            raise ValidationError("vol_step and q_step must be non-negative")

    @property
    def mode(self) -> str:
        if self.provider == "life_surface":
            return "life_surface"
        return "exact" if self.spot_step is None else "ladder"
```

In `EnsembleConfig`, add two fields after `allow_data_end`:

```python
    workers: int = 1
    batch_paths: Optional[int] = None
```

and append to `__post_init__`:

```python
        if int(self.workers) < 1:
            raise ValidationError("workers must be at least 1")
        if self.batch_paths is not None and int(self.batch_paths) < 1:
            raise ValidationError("batch_paths must be at least 1 or None (one batch)")
        if self.pricing.provider == "life_surface":
            from quantark.util.enum.engine_enums import EngineType

            if self.engine_config.pricing_engine_type != EngineType.PDE:
                raise ValidationError(
                    "the life_surface provider needs a PDE engine; use provider='repricing' for "
                    f"{self.engine_config.pricing_engine_type!r}"
                )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_config.py test/simulation/test_engine.py -q`
Expected: all pass (the plan-2 engine still constructs the exact provider).

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/config.py test/simulation/conftest.py test/simulation/test_config.py
git commit -m "feat(backtest/simulation): config fields for ladder, life surface, disk cache and batching

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Ladder mode in `RepricingPricer`

**Files:**
- Modify: `quantark/backtest/simulation/pricing/base.py` (`bucket_key`, `bucket_centre`)
- Modify: `quantark/backtest/simulation/pricing/repricing.py`
- Create: `test/simulation/test_ladder.py`

**The mode.** `S_ref = product.initial_price` and `x_ref = ln S_ref`, so ladder node `j` is `S_j = S_ref · exp(j · spot_step)` — the same node on every day and in every cell, which is what lets the cache serve across cells. A state at `x = ln S` needs nodes `j = ⌊(x − x_ref)/step⌋` and `j + 1` with weight `w = (x − x_ref)/step − j`; PV, delta and gamma are linear in `x` between them. Vol and `q` are replaced by their bucket centres (`np.round(v / step) · step`, exact when the step is 0). The node's environment is a flat vol, the path's exact rate, `SignedDividendYield(q_centre)` and no basis (nothing under `asset/equity` reads it). Only the two nodes each state needs are priced — a subset of the spec's "one node beyond each end" span with identical results.

**Interfaces:**
- `bucket_key(values: np.ndarray, step: Optional[float]) -> np.ndarray` (int64: bucket indices, or `float_key` when the step is `None`/0).
- `bucket_centre(values, step) -> np.ndarray` (the bucketed values, or `values` unchanged).
- `RepricingPricer(..., spot_step=None, vol_step=None, q_step=None)`; `mode` property; `price_day` in either mode; `ladder_nodes(states) -> (j, w)`; `fingerprint()` moves with the steps.

- [ ] **Step 1: Write the failing tests**

`test/simulation/test_ladder.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.replay.market import SignedDividendYield
from quantark.backtest.simulation.config import CacheConfig
from quantark.backtest.simulation.pricing.base import DayStates, bucket_centre, bucket_key, float_key
from quantark.backtest.simulation.pricing.cache import StateCache
from quantark.backtest.simulation.pricing.repricing import RepricingPricer

from .conftest import RATE, SPOT, pde_engine_config, short_snowball

START = pd.Timestamp("2024-01-02")


def _pricer(**steps) -> RepricingPricer:
    return RepricingPricer(
        short_snowball(), engine_config=pde_engine_config(), start_date=START, underlying="CSI1000",
        cache=StateCache(CacheConfig(memory_bytes=8_000_000)), **steps,
    )


def _states(day_index, spots, *, vol=0.22, q=0.05, knocked_in=None) -> DayStates:
    spots = np.asarray(spots, dtype=float)
    m = spots.size
    vols = np.full(m, vol) if np.isscalar(vol) else np.asarray(vol, dtype=float)
    qs = np.full(m, q) if np.isscalar(q) else np.asarray(q, dtype=float)
    return DayStates(
        day_index=day_index, date=START + pd.Timedelta(days=day_index), path_index=np.arange(m),
        spot=spots, vol=vols, rate=np.full(m, RATE), q_T=qs,
        div_yield=tuple(SignedDividendYield(float(x)) for x in qs), basis_yield=np.full(m, -0.03),
        env_key=float_key(np.full(m, RATE)),
        knocked_in=np.zeros(m, dtype=bool) if knocked_in is None else np.asarray(knocked_in, dtype=bool),
    )


def test_bucket_keys_and_centres():
    values = np.array([0.2149, 0.2151, 0.22, 0.0])
    assert list(bucket_key(values, 0.01)) == [21, 22, 22, 0]
    assert bucket_centre(values, 0.01) == pytest.approx([0.21, 0.22, 0.22, 0.0])
    assert list(bucket_key(values, 0.0)) == list(float_key(values))
    assert bucket_centre(values, None) is values


def test_ladder_nodes_bracket_each_spot():
    pricer = _pricer(spot_step=0.01, vol_step=0.0, q_step=0.0)
    states = _states(3, [SPOT, SPOT * np.exp(0.015), SPOT * np.exp(-0.004)])
    j, w = pricer.ladder_nodes(states)
    assert list(j) == [0, 1, -1]
    assert w == pytest.approx([0.0, 0.5, 0.6])


def test_ladder_prices_interpolate_between_the_exact_nodes():
    exact = _pricer()
    ladder = _pricer(spot_step=0.01, vol_step=0.0, q_step=0.0)
    spots = SPOT * np.exp(np.array([0.0, 0.004, 0.01, 0.0135, -0.006]))
    pv_l, delta_l, gamma_l = ladder.price_day(_states(3, spots))
    pv_e, delta_e, _ = exact.price_day(_states(3, spots))
    assert pv_l[0] == pv_e[0] and pv_l[2] == pv_e[2]          # on a node: the node's own price
    assert pv_l == pytest.approx(pv_e, rel=2e-3)              # off a node: linear in x
    assert delta_l == pytest.approx(delta_e, abs=0.02)
    assert ladder.stats()["engine_calls"] == 5                # nodes -1..3, each once
    assert exact.stats()["engine_calls"] == 5


def test_the_same_node_is_one_cache_key_across_days_and_paths():
    ladder = _pricer(spot_step=0.01, vol_step=0.0, q_step=0.0)
    a = ladder.state_keys(_states(3, [SPOT * np.exp(0.003)]))
    b = ladder.state_keys(_states(3, [SPOT * np.exp(0.007)]))
    assert a == b                                              # both need nodes 0 and 1
    c = ladder.state_keys(_states(4, [SPOT * np.exp(0.003)]))
    assert a != c                                              # a different day is a different key


def test_vol_and_q_buckets_share_a_node_price():
    ladder = _pricer(spot_step=0.01, vol_step=0.01, q_step=0.005)
    states = _states(3, [SPOT, SPOT], vol=[0.2149, 0.2151], q=[0.0501, 0.0499])
    pv, _, _ = ladder.price_day(states)
    assert pv[0] != pv[1]                                      # 0.2149 -> 0.21, 0.2151 -> 0.22
    states = _states(3, [SPOT, SPOT], vol=[0.2226, 0.2174], q=[0.0501, 0.0499])
    pv, _, _ = ladder.price_day(states)
    assert pv[0] == pv[1]                                      # both -> vol 0.22, q 0.05
    assert ladder.stats()["engine_calls"] == 3


def test_a_zero_step_keeps_vol_and_q_exact():
    ladder = _pricer(spot_step=0.01, vol_step=0.0, q_step=0.0)
    states = _states(3, [SPOT, SPOT], vol=[0.2200, np.nextafter(0.22, 1.0)])
    pv, _, _ = ladder.price_day(states)
    assert ladder.stats()["engine_calls"] == 2


def test_the_ki_flag_selects_the_product_in_ladder_mode():
    ladder = _pricer(spot_step=0.01, vol_step=0.0, q_step=0.0)
    pv, _, _ = ladder.price_day(_states(3, [SPOT * 0.7, SPOT * 0.7], knocked_in=[False, True]))
    assert pv[0] != pv[1]


def test_the_fingerprint_and_mode_move_with_the_steps():
    assert _pricer().mode == "exact"
    ladder = _pricer(spot_step=0.01, vol_step=0.0, q_step=0.0)
    assert ladder.mode == "ladder"
    assert ladder.fingerprint() != _pricer().fingerprint()
    assert ladder.fingerprint() != _pricer(spot_step=0.02, vol_step=0.0, q_step=0.0).fingerprint()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_ladder.py -q`
Expected: `ImportError: cannot import name 'bucket_centre'`.

- [ ] **Step 3: Add the bucket helpers to `pricing/base.py`**

After `float_key`:

```python
def bucket_key(values: np.ndarray, step: Optional[float]) -> np.ndarray:
    """Integer bucket per value: ``round(v / step)``, or the float bits when the step is None/0.

    ``np.round`` is half-to-even on every platform, so a value on a bucket
    edge lands in the same bucket on every machine.
    """
    arr = np.asarray(values, dtype=np.float64)
    if step is None or float(step) == 0.0:
        return float_key(arr)
    return np.round(arr / float(step)).astype(np.int64)


def bucket_centre(values: np.ndarray, step: Optional[float]) -> np.ndarray:
    """The bucket centre each value prices at; the values themselves when the step is None/0."""
    if step is None or float(step) == 0.0:
        return values
    return np.round(np.asarray(values, dtype=np.float64) / float(step)) * float(step)
```

(add `Optional` to the `typing` import.)

- [ ] **Step 4: Add ladder mode to `RepricingPricer`**

In `quantark/backtest/simulation/pricing/repricing.py`:

Change `engine_fingerprint` to take the steps:

```python
def engine_fingerprint(
    engine_config: Any, delta_bump_size: Optional[float], gamma_bump_size: Optional[float],
    *, spot_step: Optional[float] = None, vol_step: Optional[float] = None, q_step: Optional[float] = None,
) -> str:
    """Identity of everything that can change a price for a given state, the mode included."""
    h = hashlib.blake2b(digest_size=16)
    for name in _ENGINE_FIELDS:
        h.update(repr(getattr(engine_config, name, None)).encode())
        h.update(b"\x1f")
    h.update(repr((delta_bump_size, gamma_bump_size)).encode())
    h.update(repr(("ladder" if spot_step is not None else "exact", spot_step, vol_step, q_step)).encode())
    return h.hexdigest()
```

Extend the constructor (new keyword arguments and attributes):

```python
        spot_step: Optional[float] = None,
        vol_step: Optional[float] = None,
        q_step: Optional[float] = None,
    ) -> None:
        ...
        self.spot_step = None if spot_step is None else float(spot_step)
        self.vol_step = vol_step
        self.q_step = q_step
        if self.spot_step is not None and self.spot_step <= 0.0:
            raise ValidationError("spot_step must be positive or None")
        self._x_ref = math.log(float(product.initial_price))
        self._engine_fp = engine_fingerprint(engine_config, delta_bump_size, gamma_bump_size,
                                             spot_step=self.spot_step, vol_step=vol_step, q_step=q_step)
```

(add `import math`; `initial_price` is a term of every `SnowballOption`.)

Add the mode and the node arithmetic:

```python
    @property
    def mode(self) -> str:
        return "exact" if self.spot_step is None else "ladder"

    def ladder_nodes(self, states: DayStates) -> Tuple[np.ndarray, np.ndarray]:
        """Lower node index and interpolation weight per state (ladder mode)."""
        if self.spot_step is None:
            raise ValidationError("ladder_nodes is only defined in ladder mode")
        u = (np.log(np.asarray(states.spot, dtype=float)) - self._x_ref) / self.spot_step
        j = np.floor(u).astype(np.int64)
        return j, u - j

    def node_spot(self, j: np.ndarray) -> np.ndarray:
        return np.exp(self._x_ref + np.asarray(j, dtype=float) * self.spot_step)
```

Split `price_day` by mode. Keep the plan-2 body as `_price_exact` and add:

```python
    def price_day(self, states: DayStates) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(pv, delta, gamma)`` per alive state, per unit product."""
        if len(states) == 0:
            empty = np.empty(0)
            return empty, empty.copy(), empty.copy()
        if self.spot_step is None:
            return self._price_exact(states)
        return self._price_ladder(states)

    def _price_ladder(self, states: DayStates) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Price the two bracketing nodes of every state, then interpolate in log-spot."""
        date = pd.Timestamp(states.date).normalize()
        j, w = self.ladder_nodes(states)
        vol_k, vol_c = bucket_key(states.vol, self.vol_step), bucket_centre(states.vol, self.vol_step)
        q_k, q_c = bucket_key(states.q_T, self.q_step), bucket_centre(states.q_T, self.q_step)
        env_key = row_keys(np.column_stack([states.rate, np.asarray(q_c, dtype=float)]))
        m = len(states)
        # Rows: 2m node states, row 2n is the lower node of state n, 2n+1 the upper.
        node_j = np.repeat(j, 2) + np.tile([0, 1], m)
        rows = np.repeat(np.arange(m), 2)
        keys = [
            StateKey(
                product_fingerprint=self._product_fp, day_index=int(states.day_index),
                knocked_in=bool(states.knocked_in[n]), spot_key=int(node_j[r]), vol_key=int(vol_k[n]),
                env_key=int(env_key[n]), engine_fingerprint=self._engine_fp,
            )
            for r, n in enumerate(rows)
        ]
        hit, c_pv, c_delta, c_gamma = self.cache.get_many(keys)
        pv_n, delta_n, gamma_n = c_pv, c_delta, c_gamma
        pending: Dict[StateKey, List[int]] = {}
        for r in np.flatnonzero(~hit):
            pending.setdefault(keys[int(r)], []).append(int(r))
        node_spot = self.node_spot(node_j)
        for key, positions in pending.items():
            r = positions[0]
            n = int(rows[r])
            values = self._price_env(
                date, knocked_in=bool(states.knocked_in[n]), key=key, spot=float(node_spot[r]),
                vol=float(vol_c[n]), rate=float(states.rate[n]),
                div_yield=SignedDividendYield(float(q_c[n])), basis_yield=None,
                label=f"ladder node {int(node_j[r])}",
            )
            self.cache.put(key, *values)
            for pos in positions:
                pv_n[pos], delta_n[pos], gamma_n[pos] = values
        self._sample(states)
        lo, hi = slice(0, None, 2), slice(1, None, 2)
        pv = (1.0 - w) * pv_n[lo] + w * pv_n[hi]
        delta = (1.0 - w) * delta_n[lo] + w * delta_n[hi]
        gamma = (1.0 - w) * gamma_n[lo] + w * gamma_n[hi]
        return pv, delta, gamma
```

Refactor `_price_one` into `_price_env` so both modes share the engine call:

```python
    def _price_one(self, states: DayStates, n: int, date: pd.Timestamp, key: StateKey):
        return self._price_env(
            date, knocked_in=bool(states.knocked_in[n]), key=key, spot=float(states.spot[n]),
            vol=float(states.vol[n]), rate=float(states.rate[n]), div_yield=states.div_yield[n],
            basis_yield=ImpliedBasisYield(float(states.basis_yield[n])),
            label=f"day {states.day_index} path {int(states.path_index[n])}",
        )

    def _price_env(self, date, *, knocked_in, key, spot, vol, rate, div_yield, basis_yield, label):
        product = self.aged_product(date, knocked_in=knocked_in)
        self._seed_engine(self._engine, key)
        env = PricingEnvironment(
            spot_quote=SpotQuote(spot=spot, asset_name=self.underlying),
            vol_surface=FlatVolSurface(volatility=vol), rate_curve=FlatRateCurve(rate=rate),
            div_yield=div_yield, basis_yield=basis_yield,
            valuation_date=pd.Timestamp(date).to_pydatetime(),
        )
        try:
            price = float(self._engine.price(product, env))
            greeks = self._engine.calculate_greeks(product, env)
        except Exception as exc:  # fail closed with the state in the message
            raise ValidationError(
                f"pricing failed at {label}: spot={spot!r}, vol={vol!r}, knocked_in={knocked_in}: {exc}"
            ) from exc
        self._engine_calls += 1
        return price, float(greeks["delta"]), float(greeks["gamma"])
```

`_sample` is a no-op until Task 5 adds the reservoir; add it now as `def _sample(self, states): return None` so both modes call it. Add `SignedDividendYield` and `bucket_key`, `bucket_centre`, `row_keys` to the imports; extend `stats()` with `"mode": self.mode` (replacing the fixed `"exact"`).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_ladder.py test/simulation/test_repricing.py test/simulation/test_conformance.py -q`
Expected: all pass; the exact-mode oracle gaps stay `0.0`. If `test_ladder_prices_interpolate_between_the_exact_nodes` fails on its tolerance, record the measured gap in the commit and tighten or loosen with the number in the message — never silently.

- [ ] **Step 6: Commit**

```bash
git add quantark/backtest/simulation/pricing/base.py quantark/backtest/simulation/pricing/repricing.py \
        test/simulation/test_ladder.py
git commit -m "feat(backtest/simulation): spot-ladder mode with vol and q buckets in the repricing provider

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: The on-disk cache tier

**Files:**
- Modify: `quantark/backtest/simulation/pricing/cache.py`
- Modify: `test/simulation/test_cache.py`

**Design.** One shard per `(product_fingerprint, engine_fingerprint, library version)`, named `{product_fp}-{engine_fp}-{version}.npz`, holding `keys` (int64 `(n, 5)`: `day_index, knocked_in, spot_key, vol_key, env_key`), `values` (float64 `(n, 3)`) and three string headers. Loaded lazily on the first miss for that pair; a shard whose headers do not match its name is a miss and is never written to. `flush()` re-reads the shard under an `fcntl` advisory lock, merges, and replaces atomically — so two batch workers flushing the same shard lose nothing. A disk hit is promoted to the memory tier.

**Interfaces:**
- `DiskTier(directory: Path)`: `get(key) -> Optional[tuple]`, `record(key, pv, delta, gamma)`, `flush() -> int` (entries written), `stats() -> dict` (`disk_hits`, `disk_misses`, `shards_loaded`, `foreign_shards`, `flushed`).
- `StateCache(config)` consults the disk tier on a memory miss and records every `put`; `flush()` delegates; `CacheStats` gains `disk: Optional[dict]`.
- `shard_name(product_fp, engine_fp) -> str`; `LIBRARY_VERSION` from `quantark.__version__`.

- [ ] **Step 1: Write the failing tests**

Append to `test/simulation/test_cache.py`:

```python
import numpy as np

from quantark.backtest.simulation.pricing.cache import DiskTier, shard_name


def test_the_disk_tier_round_trips_between_two_caches(tmp_path):
    first = StateCache(CacheConfig(memory_bytes=10 * ENTRY_BYTES, disk_dir=str(tmp_path)))
    first.put(_key(1), 1.5, -0.2, 0.01)
    first.put(_key(2), 2.5, -0.3, 0.02)
    assert first.flush() == 2
    second = StateCache(CacheConfig(memory_bytes=10 * ENTRY_BYTES, disk_dir=str(tmp_path)))
    assert second.get(_key(1)) == (1.5, -0.2, 0.01)
    assert second.get(_key(3)) is None
    stats = second.stats()
    assert stats.disk["disk_hits"] == 1 and stats.disk["shards_loaded"] == 1
    assert stats.entries == 1                       # the hit was promoted to memory


def test_flush_merges_with_what_another_process_wrote(tmp_path):
    a = StateCache(CacheConfig(memory_bytes=10 * ENTRY_BYTES, disk_dir=str(tmp_path)))
    b = StateCache(CacheConfig(memory_bytes=10 * ENTRY_BYTES, disk_dir=str(tmp_path)))
    a.put(_key(1), 1.0, 0.0, 0.0)
    b.put(_key(2), 2.0, 0.0, 0.0)
    a.flush()
    b.flush()
    c = StateCache(CacheConfig(memory_bytes=10 * ENTRY_BYTES, disk_dir=str(tmp_path)))
    assert c.get(_key(1)) == (1.0, 0.0, 0.0) and c.get(_key(2)) == (2.0, 0.0, 0.0)


def test_a_foreign_shard_is_a_miss_and_is_left_alone(tmp_path):
    path = tmp_path / shard_name("prod", "eng")
    np.savez(path, keys=np.zeros((1, 5), dtype=np.int64), values=np.ones((1, 3)),
             library_version=np.array("0.0.0"), engine_fingerprint=np.array("eng"),
             product_fingerprint=np.array("prod"))
    before = path.read_bytes()
    cache = StateCache(CacheConfig(memory_bytes=10 * ENTRY_BYTES, disk_dir=str(tmp_path)))
    assert cache.get(StateKey("prod", 0, False, 0, 0, 0, "eng")) is None
    cache.put(StateKey("prod", 0, False, 0, 0, 0, "eng"), 5.0, 0.0, 0.0)
    cache.flush()
    assert path.read_bytes() == before
    assert cache.stats().disk["foreign_shards"] == 1


def test_a_memory_only_cache_has_no_disk_stats():
    cache = StateCache(CacheConfig(memory_bytes=10 * ENTRY_BYTES))
    assert cache.stats().disk is None
    assert cache.flush() == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_cache.py -q`
Expected: `ImportError: cannot import name 'DiskTier'`.

- [ ] **Step 3: Write the implementation**

Add to `quantark/backtest/simulation/pricing/cache.py`:

```python
import fcntl
import os
import tempfile
from pathlib import Path

import quantark

LIBRARY_VERSION = str(quantark.__version__)
_KEY_FIELDS = ("day_index", "knocked_in", "spot_key", "vol_key", "env_key")


def shard_name(product_fp: str, engine_fp: str) -> str:
    """The shard file for one product and engine at this library version."""
    return f"{product_fp}-{engine_fp}-{LIBRARY_VERSION.replace('.', '_')}.npz"


def _key_row(key: StateKey) -> Tuple[int, int, int, int, int]:
    return (int(key.day_index), int(key.knocked_in), int(key.spot_key), int(key.vol_key), int(key.env_key))


class DiskTier:
    """Append-only npz shards, one per (product, engine, library version).

    A shard is read once, lazily, on the first miss for its pair.  Its
    headers must match its name; otherwise it is foreign -- a miss, never
    reinterpreted, never overwritten.  ``flush`` merges under an advisory
    lock so concurrent writers (batch workers) cannot lose each other's
    entries, and replaces the file atomically.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._loaded: Dict[Tuple[str, str], Dict[tuple, Tuple[float, float, float]]] = {}
        self._foreign: set = set()
        self._pending: Dict[Tuple[str, str], Dict[tuple, Tuple[float, float, float]]] = {}
        self._hits = self._misses = self._loads = self._flushed = 0

    def _pair(self, key: StateKey) -> Tuple[str, str]:
        return (key.product_fingerprint, key.engine_fingerprint)

    def _path(self, pair: Tuple[str, str]) -> Path:
        return self.directory / shard_name(*pair)

    def _read(self, pair: Tuple[str, str]) -> Dict[tuple, Tuple[float, float, float]]:
        path = self._path(pair)
        if not path.exists():
            return {}
        with np.load(path, allow_pickle=False) as z:
            headers = (str(z["library_version"]), str(z["product_fingerprint"]), str(z["engine_fingerprint"]))
            if headers != (LIBRARY_VERSION, pair[0], pair[1]):
                self._foreign.add(pair)
                return {}
            keys = np.asarray(z["keys"], dtype=np.int64)
            values = np.asarray(z["values"], dtype=np.float64)
        return {tuple(int(x) for x in k): (float(v[0]), float(v[1]), float(v[2])) for k, v in zip(keys, values)}

    def _table(self, pair: Tuple[str, str]) -> Dict[tuple, Tuple[float, float, float]]:
        table = self._loaded.get(pair)
        if table is None:
            table = self._read(pair)
            self._loaded[pair] = table
            self._loads += 1
        return table

    def get(self, key: StateKey) -> Optional[Tuple[float, float, float]]:
        value = self._table(self._pair(key)).get(_key_row(key))
        if value is None:
            self._misses += 1
        else:
            self._hits += 1
        return value

    def record(self, key: StateKey, pv: float, delta: float, gamma: float) -> None:
        pair = self._pair(key)
        self._pending.setdefault(pair, {})[_key_row(key)] = (float(pv), float(delta), float(gamma))
        self._table(pair)[_key_row(key)] = (float(pv), float(delta), float(gamma))

    def flush(self) -> int:
        """Merge the pending entries into their shards; returns the count written."""
        written = 0
        for pair, entries in list(self._pending.items()):
            if not entries:
                continue
            if pair in self._foreign:
                continue                      # never overwrite a shard we could not read
            path = self._path(pair)
            lock = path.with_suffix(".lock")
            with open(lock, "w") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    merged = self._read(pair)
                    if pair in self._foreign:
                        continue
                    merged.update(entries)
                    keys = np.array(list(merged.keys()), dtype=np.int64).reshape(-1, 5)
                    values = np.array(list(merged.values()), dtype=np.float64).reshape(-1, 3)
                    fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".npz.tmp")
                    os.close(fd)
                    np.savez(tmp, keys=keys, values=values,
                             library_version=np.array(LIBRARY_VERSION),
                             product_fingerprint=np.array(pair[0]), engine_fingerprint=np.array(pair[1]))
                    os.replace(tmp, path)
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)
            written += len(entries)
            self._pending[pair] = {}
        self._flushed += written
        return written

    def stats(self) -> Dict[str, Any]:
        return {
            "disk_hits": self._hits, "disk_misses": self._misses, "shards_loaded": self._loads,
            "foreign_shards": len(self._foreign), "flushed": self._flushed,
            "pending": sum(len(v) for v in self._pending.values()),
        }
```

(`np.savez` appends `.npz` when the name lacks it; `tmp` ends in `.npz.tmp`, so pass `tmp` and then `os.replace(tmp + ".npz" if not tmp.endswith(".npz") else tmp, path)` — check what `np.savez` actually wrote with `Path(tmp).exists()`; the simplest robust form is `with open(tmp, "wb") as fh: np.savez(fh, ...)`, which writes exactly to `tmp`. Use that form.)

Extend `CacheStats` with `disk: Optional[Dict[str, Any]] = None` (and include it in `as_dict`), and `StateCache`:

```python
    def __init__(self, config: CacheConfig) -> None:
        ...
        self._disk = DiskTier(Path(config.disk_dir)) if config.disk_dir is not None else None

    def get(self, key):
        value = self._store.get(key)
        if value is None:
            if self._disk is not None:
                value = self._disk.get(key)
                if value is not None:
                    self._store[key] = value          # promote, counting as a hit below
                    self._evict()
            if value is None:
                self._misses += 1
                return None
        self._store.move_to_end(key)
        self._hits += 1
        return value

    def put(self, key, pv, delta, gamma):
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (float(pv), float(delta), float(gamma))
        if self._disk is not None:
            self._disk.record(key, pv, delta, gamma)
        self._evict()

    def _evict(self) -> None:
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)
            self._evictions += 1

    def flush(self) -> int:
        """Write pending entries to the disk tier; 0 without one."""
        return 0 if self._disk is None else self._disk.flush()

    def stats(self):
        return CacheStats(..., disk=None if self._disk is None else self._disk.stats())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_cache.py test/simulation/test_repricing.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/pricing/cache.py test/simulation/test_cache.py
git commit -m "feat(backtest/simulation): on-disk state-cache tier with fingerprinted, lock-merged shards

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: The sampling gate

**Files:**
- Modify: `quantark/backtest/simulation/pricing/base.py` (`GateScale`, `GateFailure`, `state_row`, protocol)
- Modify: `quantark/backtest/simulation/pricing/repricing.py` (reservoir, `verify`, `sample_visited`)
- Modify: `quantark/backtest/simulation/engine.py` (wiring)
- Create: `test/simulation/test_gate.py`

**Design.** The provider owns its gate. In an approximate mode `price_day` feeds every state it priced into a reservoir sampler (Algorithm R, capacity `gate.sample_states`, RNG seeded from `blake2b(product_fp, engine_fp)`), so the sample is a uniform draw over the run's visited states and the same on every machine. `verify(samples, gate, scale)` prices each sample twice — exactly, through the engine at the exact spot, vol and dividend object, and through the provider's own approximation — and reports the largest PV gap in bp of unit notional and the largest delta gap in hands. The engine calls it twice: on day 0's states before pricing the run, and on the reservoir after the loop; either failure raises `GateFailure` carrying the report, and the cell produces no results. In exact mode the reservoir is never filled and the report is the zero report.

**Interfaces:**
- `GateScale(unit_notional: float, hands_per_unit_delta: float)` frozen: `unit_notional = initial_price · contract_multiplier`, `hands_per_unit_delta = |quantity| / hedge multiplier`.
- `GateFailure(ValidationError)` with `.report: GateReport`.
- `state_row(states: DayStates, n: int) -> DayStates` (one-row view).
- `PathPricer.verify(samples: Sequence[DayStates], gate: GateConfig, scale: GateScale) -> GateReport`; `sample_visited() -> List[DayStates]`.
- `GateReport.combine(reports) -> GateReport` (max gaps, summed sample count, all passed).
- `RepricingPricer(..., gate: Optional[GateConfig] = None)`: the reservoir capacity comes from `gate.sample_states`; `None` disables sampling (exact mode never samples).

- [ ] **Step 1: Write the failing tests**

`test/simulation/test_gate.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.replay.market import SignedDividendYield
from quantark.backtest.simulation.config import CacheConfig, GateConfig
from quantark.backtest.simulation.pricing.base import DayStates, GateFailure, GateReport, GateScale, float_key
from quantark.backtest.simulation.pricing.cache import StateCache
from quantark.backtest.simulation.pricing.repricing import RepricingPricer

from .conftest import RATE, SPOT, pde_engine_config, short_snowball

START = pd.Timestamp("2024-01-02")
SCALE = GateScale(unit_notional=SPOT, hands_per_unit_delta=1000.0 / 200.0)


def _pricer(gate: GateConfig, **steps) -> RepricingPricer:
    return RepricingPricer(
        short_snowball(), engine_config=pde_engine_config(), start_date=START, underlying="CSI1000",
        cache=StateCache(CacheConfig(memory_bytes=8_000_000)), gate=gate, **steps,
    )


def _states(day_index, spots) -> DayStates:
    spots = np.asarray(spots, dtype=float)
    m = spots.size
    q = np.full(m, 0.05)
    return DayStates(
        day_index=day_index, date=START + pd.Timedelta(days=day_index), path_index=np.arange(m),
        spot=spots, vol=np.full(m, 0.22), rate=np.full(m, RATE), q_T=q,
        div_yield=tuple(SignedDividendYield(float(x)) for x in q), basis_yield=np.full(m, -0.03),
        env_key=float_key(np.full(m, RATE)), knocked_in=np.zeros(m, dtype=bool),
    )


def test_exact_mode_never_samples_and_reports_zero():
    pricer = _pricer(GateConfig(sample_states=4, pv_tolerance_bp=0.1, delta_tolerance_hands=0.1))
    pricer.price_day(_states(3, SPOT * np.exp(np.linspace(-0.02, 0.02, 6))))
    assert pricer.sample_visited() == []
    report = pricer.verify([], pricer.gate, SCALE)
    assert report.mode == "exact" and report.passed and report.max_pv_gap_bp == 0.0


def test_the_reservoir_is_a_deterministic_sample_of_visited_states():
    gate = GateConfig(sample_states=3, pv_tolerance_bp=100.0, delta_tolerance_hands=10.0)
    spots = SPOT * np.exp(np.linspace(-0.03, 0.03, 10))
    a, b = _pricer(gate, spot_step=0.01, vol_step=0.0, q_step=0.0), _pricer(gate, spot_step=0.01, vol_step=0.0, q_step=0.0)
    for day in (3, 4):
        a.price_day(_states(day, spots))
        b.price_day(_states(day, spots))
    sa, sb = a.sample_visited(), b.sample_visited()
    assert len(sa) == 3 and [(s.day_index, float(s.spot[0])) for s in sa] == [(s.day_index, float(s.spot[0])) for s in sb]
    assert len({float(s.spot[0]) for s in sa}) == 3


def test_a_coarse_ladder_fails_a_tight_gate_and_passes_a_loose_one():
    tight = GateConfig(sample_states=6, pv_tolerance_bp=0.01, delta_tolerance_hands=0.01)
    loose = GateConfig(sample_states=6, pv_tolerance_bp=200.0, delta_tolerance_hands=20.0)
    spots = SPOT * np.exp(np.linspace(-0.045, 0.045, 7))       # nothing on a node
    coarse = _pricer(tight, spot_step=0.03, vol_step=0.0, q_step=0.0)
    coarse.price_day(_states(3, spots))
    samples = coarse.sample_visited()
    failing = coarse.verify(samples, tight, SCALE)
    assert failing.mode == "ladder" and not failing.passed and failing.sampled == 6
    assert failing.max_pv_gap_bp > 0.01
    passing = coarse.verify(samples, loose, SCALE)
    assert passing.passed and passing.max_pv_gap_bp == failing.max_pv_gap_bp


def test_a_gap_is_measured_in_bp_of_notional_and_in_hands():
    gate = GateConfig(sample_states=2, pv_tolerance_bp=1e9, delta_tolerance_hands=1e9)
    pricer = _pricer(gate, spot_step=0.03, vol_step=0.0, q_step=0.0)
    states = _states(3, [SPOT * np.exp(0.015)])
    pricer.price_day(states)
    one = pricer.verify(pricer.sample_visited(), gate, GateScale(unit_notional=SPOT, hands_per_unit_delta=1.0))
    ten = pricer.verify(pricer.sample_visited(), gate, GateScale(unit_notional=10 * SPOT, hands_per_unit_delta=10.0))
    assert one.max_pv_gap_bp == pytest.approx(10.0 * ten.max_pv_gap_bp)
    assert ten.max_delta_gap_hands == pytest.approx(10.0 * one.max_delta_gap_hands)


def test_reports_combine_to_the_worst_case():
    a = GateReport(mode="ladder", sampled=2, max_pv_gap_bp=1.0, max_delta_gap_hands=0.5, passed=True)
    b = GateReport(mode="ladder", sampled=3, max_pv_gap_bp=0.2, max_delta_gap_hands=0.9, passed=False)
    c = GateReport.combine([a, b])
    assert (c.sampled, c.max_pv_gap_bp, c.max_delta_gap_hands, c.passed) == (5, 1.0, 0.9, False)


def test_gate_failure_carries_the_report():
    report = GateReport(mode="ladder", sampled=1, max_pv_gap_bp=9.0, max_delta_gap_hands=0.0, passed=False)
    with pytest.raises(GateFailure) as excinfo:
        raise GateFailure(report)
    assert excinfo.value.report is report and "9" in str(excinfo.value)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_gate.py -q`
Expected: `ImportError: cannot import name 'GateFailure'`.

- [ ] **Step 3: Extend `pricing/base.py`**

```python
from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class GateScale:
    """How a per-unit gap is expressed: bp of unit notional, hands of the hedge."""

    unit_notional: float
    hands_per_unit_delta: float

    def __post_init__(self) -> None:
        if float(self.unit_notional) <= 0.0 or float(self.hands_per_unit_delta) < 0.0:
            raise ValidationError("GateScale needs a positive notional and a non-negative hands ratio")


def state_row(states: DayStates, n: int) -> DayStates:
    """A one-row ``DayStates`` for state ``n`` (the reservoir stores these)."""
    sl = slice(n, n + 1)
    return DayStates(
        day_index=states.day_index, date=states.date, path_index=states.path_index[sl].copy(),
        spot=states.spot[sl].copy(), vol=states.vol[sl].copy(), rate=states.rate[sl].copy(),
        q_T=states.q_T[sl].copy(), div_yield=(states.div_yield[n],), basis_yield=states.basis_yield[sl].copy(),
        env_key=states.env_key[sl].copy(), knocked_in=states.knocked_in[sl].copy(),
    )
```

On `GateReport` add:

```python
    @staticmethod
    def combine(reports: Sequence["GateReport"]) -> "GateReport":
        """The worst case over several reports (the day-0 gate and the run's reservoir)."""
        reports = list(reports)
        if not reports:
            raise ValidationError("combine needs at least one report")
        return GateReport(
            mode=reports[0].mode, sampled=sum(r.sampled for r in reports),
            max_pv_gap_bp=max(r.max_pv_gap_bp for r in reports),
            max_delta_gap_hands=max(r.max_delta_gap_hands for r in reports),
            passed=all(r.passed for r in reports),
        )


class GateFailure(ValidationError):
    """An approximate provider missed its accuracy budget; the cell produced nothing."""

    def __init__(self, report: GateReport) -> None:
        self.report = report
        super().__init__(
            f"pricing gate failed ({report.mode}): max PV gap {report.max_pv_gap_bp:.4g} bp, "
            f"max delta gap {report.max_delta_gap_hands:.4g} hands over {report.sampled} sampled states"
        )
```

Update the protocol: `def verify(self, samples: Sequence[DayStates], gate: Any, scale: GateScale) -> GateReport: ...` and `def sample_visited(self) -> List[DayStates]: ...`.

- [ ] **Step 4: Add the reservoir and `verify` to `RepricingPricer`**

Constructor: new keyword `gate: Optional[GateConfig] = None`; attributes:

```python
        self.gate = gate
        self._reservoir: List[DayStates] = []
        self._seen = 0
        seed = int.from_bytes(hashlib.blake2b((self._product_fp + self._engine_fp).encode(), digest_size=4).digest(), "big")
        self._rng = np.random.default_rng(seed)
```

Replace the `_sample` stub:

```python
    def _sample(self, states: DayStates) -> None:
        """Algorithm R over every approximately priced state (ladder mode only)."""
        if self.spot_step is None or self.gate is None or int(self.gate.sample_states) <= 0:
            return
        capacity = int(self.gate.sample_states)
        for n in range(len(states)):
            self._seen += 1
            if len(self._reservoir) < capacity:
                self._reservoir.append(state_row(states, n))
                continue
            slot = int(self._rng.integers(0, self._seen))
            if slot < capacity:
                self._reservoir[slot] = state_row(states, n)

    def sample_visited(self) -> List[DayStates]:
        return list(self._reservoir)

    def verify(self, samples: Sequence[DayStates], gate: GateConfig, scale: GateScale) -> GateReport:
        """Exact mode: the zero report.  Ladder mode: reprice each sample exactly and compare."""
        if self.spot_step is None:
            return GateReport(mode="exact", sampled=0, max_pv_gap_bp=0.0, max_delta_gap_hands=0.0, passed=True)
        worst_pv = worst_delta = 0.0
        count = 0
        for row in samples:
            if len(row) != 1:
                raise ValidationError("verify takes one-row DayStates (see state_row)")
            date = pd.Timestamp(row.date).normalize()
            exact = self._price_env(
                date, knocked_in=bool(row.knocked_in[0]), key=self.state_keys(row)[0],
                spot=float(row.spot[0]), vol=float(row.vol[0]), rate=float(row.rate[0]),
                div_yield=row.div_yield[0], basis_yield=ImpliedBasisYield(float(row.basis_yield[0])),
                label=f"gate sample day {row.day_index} path {int(row.path_index[0])}",
            )
            pv, delta, _ = self._price_ladder(row)
            worst_pv = max(worst_pv, abs(float(pv[0]) - exact[0]) / float(scale.unit_notional) * 1e4)
            worst_delta = max(worst_delta, abs(float(delta[0]) - exact[1]) * float(scale.hands_per_unit_delta))
            count += 1
        return GateReport(
            mode="ladder", sampled=count, max_pv_gap_bp=worst_pv, max_delta_gap_hands=worst_delta,
            passed=worst_pv <= float(gate.pv_tolerance_bp) and worst_delta <= float(gate.delta_tolerance_hands),
        )
```

`state_keys` in ladder mode: keep the plan-2 exact keys for `state_keys` (the exact side of the gate uses the exact key so its seed is the exact state's seed); `_price_ladder` builds its own node keys. `_price_ladder` must not re-sample when called from `verify`: give it a keyword `sample: bool = True` and pass `False` there. (`_price_env` counts the exact call in `engine_calls`; report it as such.)

- [ ] **Step 5: Wire the gate into the engine**

In `engine.py`:

```python
from .pricing.base import DayStates, GateFailure, GateReport, GateScale, row_keys, state_row
```

Construct pricers with `gate=cfg.pricing.gate` and the steps:

```python
            RepricingPricer(
                bp.product, engine_config=cfg.engine_config, start_date=dates[0], underlying=cfg.underlying,
                cache=cache, delta_bump_size=cfg.delta_bump_size, gamma_bump_size=cfg.gamma_bump_size,
                spot_step=cfg.pricing.spot_step, vol_step=cfg.pricing.vol_step, q_step=cfg.pricing.q_step,
                gate=cfg.pricing.gate,
            )
```

Add the scale helper and the two gate points:

```python
    def _gate_scale(self, bp) -> GateScale:
        product = bp.product
        return GateScale(
            unit_notional=float(product.initial_price) * float(getattr(product, "contract_multiplier", 1.0)),
            hands_per_unit_delta=abs(float(bp.quantity)) / float(self.config.hedge.multiplier),
        )
```

In `run`, right after `_day_market` on day 0 (before `_initial_book_value`):

```python
            if d == 0:
                gate_reports = [
                    pricer.verify(
                        [state_row(states, n) for n in range(len(states))],
                        cfg.pricing.gate, self._gate_scale(bp),
                    )
                    for bp, pricer, states in (
                        (bp, pricer, self._states_for(day, d, np.arange(n_paths), np.zeros(n_paths, dtype=bool),
                                                      spot, vol, rate, q_T, div_yield, basis, env_key))
                        for bp, pricer in zip(cfg.products, pricers)
                    )
                ]
                if not all(r.passed for r in gate_reports):
                    raise GateFailure(GateReport.combine(gate_reports))
```

After the loop (before `_manifest`):

```python
        gate_reports += [
            pricer.verify(pricer.sample_visited(), cfg.pricing.gate, self._gate_scale(bp))
            for bp, pricer in zip(cfg.products, pricers)
        ]
        gate = GateReport.combine(gate_reports)
        if not gate.passed:
            raise GateFailure(gate)
        cache.flush()
```

and pass `gate` into `_manifest` (replace the `pricers[0].verify(empty_states(...))` line with `"gate": gate.as_dict()`; delete `empty_states` if nothing else uses it). Set `"mode": cfg.pricing.mode` in the manifest.

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest -n0 test/simulation -q`
Expected: all pass; `test_the_manifest_records_the_run` still sees `gate.mode == "exact"`.

- [ ] **Step 7: Commit**

```bash
git add quantark/backtest/simulation/pricing/base.py quantark/backtest/simulation/pricing/repricing.py \
        quantark/backtest/simulation/engine.py test/simulation/test_gate.py
git commit -m "feat(backtest/simulation): sampling accuracy gate with a deterministic reservoir of visited states

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: `GridRequest.extra_times` — time nodes without events

**Files:**
- Modify: `quantark/asset/equity/engine/pde/grid/request.py`
- Modify: `quantark/asset/equity/engine/pde/grid/time.py`
- Modify: `test/pde_grid/test_time_builder.py`

**Why a new field and not more `event_times`.** `build_time` gives every `event_times` entry an exact node AND `event_damping_steps` Rannacher restarts after it (`base_pde_solver.py:666`), and `EventSchedule.apply` fires on those steps. A daily node that carried damping would turn Crank–Nicolson into something close to backward Euler over the whole life. `extra_times` are nodes only: they enter the boundaries and `step_of`, never the damping sets, and no schedule transform is keyed on them. The field defaults to `()`, so every existing request and every `GridBinder` cache key is unchanged; a time within `is_close` of an event time is folded into the event (one node, damped, in `step_of` under the event's key and reachable through `step_at`).

- [ ] **Step 1: Write the failing tests**

Append to `test/pde_grid/test_time_builder.py`:

```python
def test_the_default_request_is_unchanged_by_the_new_field():
    plain, explicit = req(), GridRequest(
        tau=1.0, bound_anchors=(100.0,), critical_prices=(100.0,), hard_lower=None, hard_upper=None,
        event_times=(0.25, 0.5, 0.75), extra_times=(),
    )
    assert plain == explicit and hash(plain) == hash(explicit)
    a, b = build_time(plain, CFG()), build_time(explicit, CFG())
    assert np.array_equal(a.t, b.t) and np.array_equal(a.dt, b.dt)
    assert a.event_damping_steps == b.event_damping_steps


def test_extra_times_are_exact_nodes_without_damping():
    extras = tuple(d / 365.0 for d in range(1, 365))
    request = GridRequest(tau=1.0, bound_anchors=(100.0,), critical_prices=(100.0,), hard_lower=None,
                          hard_upper=None, event_times=(0.25, 0.5, 0.75), extra_times=extras)
    tl = build_time(request, CFG())
    for t in extras:
        assert tl.t[tl.step_at(t)] == pytest.approx(t, abs=1e-15)
    plain = build_time(req(), CFG())
    assert tl.event_damping_steps.issubset(set(range(tl.actual_steps)))
    assert len(tl.event_damping_steps) == len(plain.event_damping_steps)    # three events, damped alike
    assert tl.terminal_damping_steps == {tl.actual_steps - 1}


def test_an_extra_time_on_an_event_is_folded_into_it():
    request = GridRequest(tau=1.0, bound_anchors=(100.0,), critical_prices=(100.0,), hard_lower=None,
                          hard_upper=None, event_times=(0.5,), extra_times=(0.5 + 1e-13, 0.25))
    assert request.extra_times == (0.25,)
    tl = build_time(request, CFG())
    assert tl.step_at(0.5) == tl.step_of[0.5]


def test_extra_times_at_the_endpoints_are_rejected():
    from quantark.util.exceptions import ValidationError

    with pytest.raises(ValidationError):
        GridRequest(tau=1.0, bound_anchors=(100.0,), critical_prices=(100.0,), hard_lower=None,
                    hard_upper=None, event_times=(), extra_times=(1.0,))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/pde_grid/test_time_builder.py -q`
Expected: `TypeError: ... unexpected keyword argument 'extra_times'`.

- [ ] **Step 3: Write the implementation**

`request.py` — add the field (last, with a default) and fold it in `__post_init__`:

```python
    extra_times: Tuple[float, ...] = ()
```

docstring line: `extra_times: Additional interior times that must be exact grid nodes but carry NO event and NO damping (a life-surface readout along a path). One within is_close of an event time is dropped: the event's node serves it.`

```python
        events = _dedup_sorted_interior(self.event_times, self.tau)
        object.__setattr__(self, "event_times", events)
        extras = _dedup_sorted_interior(self.extra_times, self.tau)
        object.__setattr__(
            self, "extra_times",
            tuple(t for t in extras if not any(is_close(t, e) for e in events)),
        )
```

`time.py` — in `build_time`, build the boundaries from both sets and key `step_of` on both, damping on events only:

```python
    tau, events = request.tau, request.event_times
    nodes = tuple(sorted(set(events) | set(request.extra_times)))
    boundaries = np.array([0.0, *nodes, tau], dtype=float)
    ...
    step_of = {}
    for e in nodes:
        k = int(np.searchsorted(t, e))
        assert t[k] == e
        step_of[e] = k

    event_damp = frozenset(
        step_of[e] - j
        for e in events
        for j in range(1, int(config.event_damping_steps) + 1)
        if step_of[e] - j >= 0
    )
```

Update the module docstring's first sentence to mention extra nodes.

- [ ] **Step 4: Run the grid-layer regression**

Run: `.venv/bin/python -m pytest -n0 test/pde_grid test/test_pde_time_grid_decoupling.py test/test_snowball_pde_knocked_in_grid.py test/replay_golden -q`
Expected: all pass with no edits to existing tests.

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/pde/grid/request.py quantark/asset/equity/engine/pde/grid/time.py \
        test/pde_grid/test_time_builder.py
git commit -m "feat(pde/grid): extra time nodes that carry no event and no damping

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: `SnowballPDESolver.solve_life_surface`

**Files:**
- Modify: `quantark/asset/equity/engine/pde/base_pde_solver.py` (`LifeSurfaceSolution`)
- Modify: `quantark/asset/equity/engine/pde/snowball_pde_solver.py` (`_extra_time_nodes`, `grid_request`, `solve_life_surface`)
- Modify: `quantark/asset/equity/engine/pde_engine.py` (facade)
- Create: `test/test_snowball_life_surface.py`

**What the seam returns.** After `_solve`, `self._grid_v0` and `self._grid_v1` hold both surfaces on every time node (column `k` after the schedule's transforms at that node — the value just before the node's event in forward time). The seam runs one solve with the requested extra nodes, copies the slabs, the grid vectors, `step_of`, and the valuation-date readout column, and resets the solver so a later `price()` is byte-identical to one on a fresh solver.

**Interfaces:**
- `LifeSurfaceSolution(NamedTuple)`: `t (n_t,)`, `x (n_x,)`, `s (n_x,)`, `v0 (n_x, n_t)`, `v1 (n_x, n_t)`, `step_of: Dict[float, int]`, `t0_readout: Optional[np.ndarray]` (the smooth valuation-date column for the alive surface when the valuation date carries events, else `None`), `knocked_in_at_valuation: bool`.
- `SnowballPDESolver.solve_life_surface(product, pricing_env, *, extra_times: Sequence[float]) -> LifeSurfaceSolution`.
- `PDEEngine.solve_life_surface(product, pricing_env, *, extra_times)` dispatching to the solver; `ValidationError` for a product whose solver lacks the seam.

- [ ] **Step 1: Write the failing tests**

`test/test_snowball_life_surface.py`:

```python
"""One PDE solve with a node on every simulation day, read along the life."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.pde_engine import PDEEngine
from quantark.asset.equity.engine.pde.base_pde_solver import LifeSurfaceSolution
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option import create_standard_snowball
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.util.exceptions import ValidationError

SPOT = 100.0


def _product(maturity_days=60, ko_days=(20, 40, 60), ki_days=(10, 30, 50)):
    return create_standard_snowball(
        initial_price=SPOT, strike=SPOT, maturity=maturity_days / 365.0, contract_multiplier=1.0,
        ko_barrier=103.0, ki_barrier=75.0, ko_rate=0.2, num_observations=len(ko_days),
        ko_observation_dates=[d / 365.0 for d in ko_days], ki_observation_type=ObservationType.DISCRETE,
        ki_continuous=False, ki_observation_dates=[d / 365.0 for d in ki_days], include_principal=True,
    )


def _env(spot=SPOT, day=0, vol=0.25):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot, asset_name="X"), vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=0.02), valuation_date=datetime(2024, 1, 2 + day),
    )


def _daily(maturity_days=60):
    return tuple(d / 365.0 for d in range(1, maturity_days))


def test_the_surface_has_a_node_on_every_day_and_reads_the_engine_price_at_day_zero():
    engine = PDEEngine(params=PDEParams())
    product = _product()
    surface = engine.solve_life_surface(product, _env(), extra_times=_daily())
    assert isinstance(surface, LifeSurfaceSolution)
    assert surface.v0.shape == (surface.x.size, surface.t.size) == surface.v1.shape
    for t in _daily():
        assert surface.t[surface.step_of[t]] == pytest.approx(t, abs=1e-15)
    column = surface.t0_readout if surface.t0_readout is not None else surface.v0[:, 0]
    assert float(np.interp(np.log(SPOT), surface.x, column)) == engine.price(product, _env())


def test_the_solver_is_left_as_it_was_found():
    engine = PDEEngine(params=PDEParams())
    product = _product()
    before = engine.price(product, _env(spot=97.0))
    engine.solve_life_surface(product, _env(), extra_times=_daily())
    solver = engine._get_solver(product)
    assert solver._extra_time_nodes == ()
    assert engine.price(product, _env(spot=97.0)) == before
    assert PDEEngine(params=PDEParams()).price(product, _env(spot=97.0)) == before


def test_an_interior_node_agrees_with_a_fresh_aged_solve_within_a_stated_bound():
    # The surface reads day 25 off the start-date grid; a fresh solve on day
    # 25 builds its own grid around the new spot.  They differ by
    # discretisation only; the bound below is the measured gap with margin.
    engine = PDEEngine(params=PDEParams())
    product = _product()
    surface = engine.solve_life_surface(product, _env(), extra_times=_daily())
    day = 25
    node = surface.step_of[day / 365.0]
    aged = _product()
    aged.maturity = (60 - day) / 365.0
    aged.barrier_config = aged.barrier_config.time_shift(day / 365.0, datetime(2024, 1, 2 + day), _env(day=day))[0]
    for spot in (90.0, 100.0, 102.0):
        read = float(np.interp(np.log(spot), surface.x, surface.v0[:, node]))
        fresh = engine.price(aged, _env(spot=spot, day=day))
        assert read == pytest.approx(fresh, rel=1e-2), (spot, read, fresh)


def test_the_knocked_in_surface_is_the_v1_slab():
    engine = PDEEngine(params=PDEParams())
    product = _product()
    surface = engine.solve_life_surface(product, _env(), extra_times=_daily())
    node = surface.step_of[25 / 365.0]
    assert float(np.interp(np.log(80.0), surface.x, surface.v1[:, node])) < float(
        np.interp(np.log(80.0), surface.x, surface.v0[:, node])
    )


def test_a_product_without_the_seam_is_refused():
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.util.enum import OptionType

    vanilla = EuropeanVanillaOption(strike=SPOT, option_type=OptionType.CALL, maturity=0.5)
    with pytest.raises(ValidationError):
        PDEEngine(params=PDEParams()).solve_life_surface(vanilla, _env(), extra_times=(0.1,))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/test_snowball_life_surface.py -q`
Expected: `ImportError: cannot import name 'LifeSurfaceSolution'`.

- [ ] **Step 3: Write the implementation**

`base_pde_solver.py`, after `PDESolutionResult`:

```python
class LifeSurfaceSolution(NamedTuple):
    """Both value surfaces of one solve on every time node (life-surface readout).

    ``v0`` / ``v1`` are the alive and knocked-in surfaces, shape
    ``(n_x, n_t)``; column ``k`` is the value at ``t[k]`` after that node's
    event transforms.  ``step_of`` maps every event and extra time to its
    column.  ``t0_readout`` is the smooth valuation-date column for the
    alive surface when the valuation date itself carries events (the same
    column ``calculate_greeks`` reads), else ``None``.
    """

    t: np.ndarray
    x: np.ndarray
    s: np.ndarray
    v0: np.ndarray
    v1: np.ndarray
    step_of: Dict[float, int]
    t0_readout: Optional[np.ndarray]
    knocked_in_at_valuation: bool
```

`snowball_pde_solver.py` — a class attribute on `SnowballPDESolver` (right after the docstring), one line in `grid_request`, and the method (place it after `calculate_greeks`):

```python
    #: Interior times that must be grid nodes for a life-surface readout
    #: (no event, no damping).  Empty for every ordinary solve.
    _extra_time_nodes: Tuple[float, ...] = ()
```

```python
        return GridRequest(
            tau=tau,
            bound_anchors=(market.spot, strike),
            critical_prices=tuple(p for p in criticals if p and p > 0),
            hard_lower=None,
            hard_upper=None,
            event_times=tuple(sorted(set(align) | set(monitor))),
            extra_times=self._extra_time_nodes,
        )
```

```python
    def solve_life_surface(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment, *, extra_times: Sequence[float]
    ) -> LifeSurfaceSolution:
        """One two-surface solve with ``extra_times`` as grid nodes; both slabs returned.

        The solve is the ordinary one (same grid layer, same schedule, same
        stepping); the extra nodes carry no event and no damping.  The
        solver's request state is restored afterwards, so a later ``price``
        is byte-identical to one on a fresh solver.
        """
        self._product_token_memo.clear()
        self._check_product_type(product)
        if pricing_env is None:
            raise ValidationError(f"PricingEnvironment is required for {self._solver_name}")
        self._validate_product(product)
        tau = product.get_maturity(pricing_env)
        if tau <= 0 or is_zero(tau):
            raise ValidationError("a life surface needs a product with time to maturity")
        if self._is_knocked_out_at_valuation(product, pricing_env.spot, pricing_env):
            raise ValidationError("a life surface needs a product that is alive at valuation")
        nodes = tuple(sorted(float(t) for t in extra_times))
        if any(t <= 0.0 or t >= tau for t in nodes):
            raise ValidationError(f"extra_times must lie strictly inside (0, {tau})")
        self._extra_time_nodes = nodes
        try:
            result = self._solve(product, pricing_env)
            layout = self._active_layout
            if layout is None:
                raise PricingError("the life surface needs the declarative grid layer")
            return LifeSurfaceSolution(
                t=np.array(layout.time.t, dtype=float), x=np.array(result.x_vec, dtype=float),
                s=np.array(result.s_vec, dtype=float), v0=np.array(self._grid_v0, dtype=float),
                v1=np.array(self._grid_v1, dtype=float), step_of=dict(layout.time.step_of),
                t0_readout=None if result.readout_vec is None else np.array(result.readout_vec, dtype=float),
                knocked_in_at_valuation=bool(self._knocked_in_at_valuation),
            )
        finally:
            self._extra_time_nodes = ()
```

(add `LifeSurfaceSolution` and `Sequence` to the imports; check `is_zero` and `PricingError` are already imported in that module — `grep -n "is_zero\|PricingError" quantark/asset/equity/engine/pde/snowball_pde_solver.py | head -3`.)

`pde_engine.py`, after `calculate_spot_greeks_curve`:

```python
    def solve_life_surface(self, product, pricing_env, *, extra_times):
        """One solve with extra time nodes, both value slabs (life-surface providers)."""
        solver = self._get_solver(product)
        seam = getattr(solver, "solve_life_surface", None)
        if seam is None:
            raise ValidationError(f"{type(solver).__name__} does not expose a life surface")
        return seam(product, pricing_env, extra_times=extra_times)
```

- [ ] **Step 4: Run the tests and the PDE regression**

Run: `.venv/bin/python -m pytest -n0 test/test_snowball_life_surface.py test/pde_grid test/replay_golden -q` then `.venv/bin/python -m pytest test/ -q -k "snowball and pde"`.
Expected: all pass. If the interior-node bound (`rel=1e-2`) fails, print the three gaps, put the measured worst case with margin in the assertion and state it in the commit message.

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/pde/base_pde_solver.py quantark/asset/equity/engine/pde/snowball_pde_solver.py \
        quantark/asset/equity/engine/pde_engine.py test/test_snowball_life_surface.py
git commit -m "feat(pde/snowball): solve_life_surface returns both value slabs on every simulation day

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: `LifeSurfacePricer`

**Files:**
- Create: `quantark/backtest/simulation/pricing/surface.py`
- Create: `test/simulation/test_surface.py`

**Design.** One surface per `(product, vol bucket, q bucket, rate bucket)` — solved at the start date for the product's whole life with an extra node on every simulation day inside `(0, maturity)`, a flat vol, a flat `SignedDividendYield(q)` and a flat rate, all at their bucket centres (`rate` bucketed with `q_step`, both being yields). A day's readout picks node `k = step_of[t_d]` with `t_d = (dates[d] − dates[0]).days / 365` (the tracker's elapsed-time convention, so the float matches the request's node exactly; `step_at` covers the `is_close` case), the `v1` slab for a knocked-in path else `v0` (day 0 alive uses `t0_readout` when present), and interpolates PV, delta and gamma linearly in `x = ln S`. Delta and gamma per node come from the solver's own non-uniform three-point stencil applied to every interior node of the column (`_calculate_delta_gamma`'s formulas, vectorised), edges copied from their neighbours. A spot outside `[x[0], x[-1]]` raises. Surfaces live in `SurfaceCache`, an LRU by bytes. The gate reprices samples exactly through an internal exact `RepricingPricer`.

**Interfaces:**
- `LifeSurface(t, x, v0, v1, d0, g0, d1, g1, step_of, t0_readout, nbytes)`; `column(k, knocked_in) -> (v, d, g)`; `readout(x_spot, k, knocked_in) -> (pv, delta, gamma)` arrays.
- `SurfaceCache(max_bytes)`: `get`, `put`, `stats()`.
- `LifeSurfacePricer(product, *, engine_config, start_date, dates, underlying, vol_step, q_step, surface_cache_bytes, gate, delta_bump_size=None, gamma_bump_size=None)`; `price_day`, `verify`, `sample_visited`, `fingerprint`, `stats`, `mode == "life_surface"`, `aged_product` (delegates to its exact pricer, for the engine's payoff and `pricing_q` helpers).

- [ ] **Step 1: Write the failing tests**

`test/simulation/test_surface.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.replay.market import SignedDividendYield
from quantark.backtest.simulation.config import GateConfig
from quantark.backtest.simulation.pricing.base import DayStates, GateScale, float_key
from quantark.backtest.simulation.pricing.repricing import RepricingPricer
from quantark.backtest.simulation.pricing.cache import StateCache
from quantark.backtest.simulation.config import CacheConfig
from quantark.backtest.simulation.pricing.surface import LifeSurfacePricer
from quantark.backtest.simulation.paths.market_path import trading_calendar
from quantark.util.exceptions import ValidationError

from .conftest import RATE, SPOT, pde_engine_config, short_snowball

DATES = trading_calendar(pd.Timestamp("2024-01-02").date(), 8)
GATE = GateConfig(sample_states=6, pv_tolerance_bp=50.0, delta_tolerance_hands=3.0)
SCALE = GateScale(unit_notional=SPOT, hands_per_unit_delta=5.0)


def _surface(vol_step=0.01, q_step=0.0025, budget=200_000_000):
    return LifeSurfacePricer(
        short_snowball(), engine_config=pde_engine_config(), start_date=DATES[0], dates=DATES,
        underlying="CSI1000", vol_step=vol_step, q_step=q_step, surface_cache_bytes=budget, gate=GATE,
    )


def _exact():
    return RepricingPricer(short_snowball(), engine_config=pde_engine_config(), start_date=DATES[0],
                           underlying="CSI1000", cache=StateCache(CacheConfig(memory_bytes=8_000_000)))


def _states(day_index, spots, *, vol=0.22, q=0.05, knocked_in=None) -> DayStates:
    spots = np.asarray(spots, dtype=float)
    m = spots.size
    qs = np.full(m, q)
    return DayStates(
        day_index=day_index, date=DATES[day_index], path_index=np.arange(m), spot=spots,
        vol=np.full(m, vol), rate=np.full(m, RATE), q_T=qs,
        div_yield=tuple(SignedDividendYield(float(x)) for x in qs), basis_yield=np.full(m, -0.03),
        env_key=float_key(np.full(m, RATE)),
        knocked_in=np.zeros(m, dtype=bool) if knocked_in is None else np.asarray(knocked_in, dtype=bool),
    )


def test_one_solve_serves_every_day_of_a_bucket():
    pricer = _surface()
    for d in range(0, 5):
        pricer.price_day(_states(d, SPOT * np.exp(np.linspace(-0.02, 0.02, 5))))
    stats = pricer.stats()
    assert stats["solves"] == 1 and stats["surface_cache"]["hits"] == 4


def test_the_readout_tracks_the_exact_engine_within_the_gate():
    pricer, exact = _surface(vol_step=0.0, q_step=0.0), _exact()
    worst = 0.0
    for d in (0, 1, 3, 4):
        states = _states(d, SPOT * np.exp(np.linspace(-0.03, 0.02, 6)))
        pv_s, delta_s, _ = pricer.price_day(states)
        pv_e, delta_e, _ = exact.price_day(states)
        worst = max(worst, float(np.max(np.abs(pv_s - pv_e))) / SPOT * 1e4)
        assert delta_s == pytest.approx(delta_e, abs=0.05)
    assert worst <= GATE.pv_tolerance_bp, worst


def test_day_zero_reads_the_engine_price_exactly_on_the_start_state():
    pricer, exact = _surface(vol_step=0.0, q_step=0.0), _exact()
    pv_s, _, _ = pricer.price_day(_states(0, [SPOT]))
    pv_e, _, _ = exact.price_day(_states(0, [SPOT]))
    assert pv_s[0] == pytest.approx(pv_e[0], rel=1e-6)


def test_the_ki_flag_reads_the_other_slab():
    pricer = _surface()
    pv, _, _ = pricer.price_day(_states(3, [SPOT * 0.8, SPOT * 0.8], knocked_in=[False, True]))
    assert pv[1] < pv[0]


def test_a_spot_outside_the_grid_fails_closed():
    pricer = _surface()
    with pytest.raises(ValidationError):
        pricer.price_day(_states(3, [SPOT * 50.0]))


def test_buckets_key_the_solve_and_the_budget_evicts():
    small = _surface(budget=1)          # below one surface: every solve evicts the last
    with pytest.raises(ValidationError):
        small.price_day(_states(1, [SPOT]))
    pricer = _surface()
    pricer.price_day(_states(1, [SPOT], vol=0.2149))
    pricer.price_day(_states(1, [SPOT], vol=0.2151))
    pricer.price_day(_states(1, [SPOT], vol=0.2226))
    assert pricer.stats()["solves"] == 2


def test_the_gate_reprices_the_reservoir_exactly():
    pricer = _surface(vol_step=0.0, q_step=0.0)
    for d in (1, 2, 3):
        pricer.price_day(_states(d, SPOT * np.exp(np.linspace(-0.03, 0.02, 4))))
    samples = pricer.sample_visited()
    assert len(samples) == 6
    report = pricer.verify(samples, GATE, SCALE)
    assert report.mode == "life_surface" and report.sampled == 6
    assert report.passed, report
    tight = GateConfig(sample_states=6, pv_tolerance_bp=1e-9, delta_tolerance_hands=1e-9)
    assert not pricer.verify(samples, tight, SCALE).passed


def test_the_fingerprint_names_the_provider_and_its_steps():
    assert _surface().fingerprint() != _exact().fingerprint()
    assert _surface(vol_step=0.02).fingerprint() != _surface().fingerprint()
    assert _surface().mode == "life_surface"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_surface.py -q`
Expected: `ModuleNotFoundError: No module named 'quantark.backtest.simulation.pricing.surface'`.

- [ ] **Step 3: Write the implementation**

`quantark/backtest/simulation/pricing/surface.py`:

```python
"""The PDE life surface: one solve per bucket, read along every path (spec 7.2).

A snowball's remaining life on day ``d`` is the start-date solve's slab at
the node for ``d``.  Solving once with a node on every simulation day and
interpolating in log-spot replaces one engine call per path per day with
one solve per (vol, q, rate) bucket; the gate measures what that costs.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from quantark.backtest.replay.engine_factory import create_pricing_engine
from quantark.backtest.replay.market import ImpliedBasisYield, SignedDividendYield
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError

from ..config import CacheConfig, GateConfig
from .base import DayStates, GateReport, GateScale, bucket_centre, bucket_key, state_row
from .cache import StateCache
from .repricing import RepricingPricer, engine_fingerprint


def _node_greeks(x: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Delta and gamma on every node of every column.

    The solver's non-uniform three-point stencil (``_calculate_delta_gamma``)
    at each interior node, converted at that node's spot; the two edge
    nodes copy their neighbours.  ``v`` is ``(n_x, n_t)``.
    """
    h_m = (x[1:-1] - x[:-2])[:, None]
    h_p = (x[2:] - x[1:-1])[:, None]
    h_sum = h_m + h_p
    v_m, v_0, v_p = v[:-2], v[1:-1], v[2:]
    dv = -h_p / (h_m * h_sum) * v_m + (h_p - h_m) / (h_m * h_p) * v_0 + h_m / (h_p * h_sum) * v_p
    d2v = 2.0 * (v_m / (h_m * h_sum) - v_0 / (h_m * h_p) + v_p / (h_p * h_sum))
    s = np.exp(x[1:-1])[:, None]
    delta = np.empty_like(v)
    gamma = np.empty_like(v)
    delta[1:-1] = dv / s
    gamma[1:-1] = (d2v - dv) / (s * s)
    delta[0], delta[-1] = delta[1], delta[-2]
    gamma[0], gamma[-1] = gamma[1], gamma[-2]
    return delta, gamma


@dataclass(frozen=True)
class LifeSurface:
    """Both slabs of one solve with their node greeks."""

    t: np.ndarray
    x: np.ndarray
    v0: np.ndarray
    v1: np.ndarray
    d0: np.ndarray
    g0: np.ndarray
    d1: np.ndarray
    g1: np.ndarray
    step_of: Dict[float, int]
    t0_readout: Optional[np.ndarray]
    t0_delta: Optional[np.ndarray]
    t0_gamma: Optional[np.ndarray]

    @property
    def nbytes(self) -> int:
        arrays = (self.t, self.x, self.v0, self.v1, self.d0, self.g0, self.d1, self.g1)
        return int(sum(a.nbytes for a in arrays))

    def node(self, elapsed: float) -> int:
        hit = self.step_of.get(elapsed)
        if hit is not None:
            return hit
        from quantark.util.numerical import is_close

        for key, k in self.step_of.items():
            if is_close(key, elapsed):
                return k
        raise ValidationError(f"no surface node at elapsed time {elapsed!r}")

    def readout(self, x_spot: np.ndarray, k: int, knocked_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_spot = np.asarray(x_spot, dtype=float)
        if np.any(x_spot < self.x[0]) or np.any(x_spot > self.x[-1]):
            bad = x_spot[(x_spot < self.x[0]) | (x_spot > self.x[-1])]
            raise ValidationError(
                f"spot {np.exp(bad[0])!r} lies outside the life-surface grid "
                f"[{np.exp(self.x[0])!r}, {np.exp(self.x[-1])!r}]; widen the grid or use repricing"
            )
        ki = np.asarray(knocked_in, dtype=bool)
        if k == 0 and self.t0_readout is not None:
            v_alive, d_alive, g_alive = self.t0_readout, self.t0_delta, self.t0_gamma
        else:
            v_alive, d_alive, g_alive = self.v0[:, k], self.d0[:, k], self.g0[:, k]
        pv = np.where(ki, np.interp(x_spot, self.x, self.v1[:, k]), np.interp(x_spot, self.x, v_alive))
        delta = np.where(ki, np.interp(x_spot, self.x, self.d1[:, k]), np.interp(x_spot, self.x, d_alive))
        gamma = np.where(ki, np.interp(x_spot, self.x, self.g1[:, k]), np.interp(x_spot, self.x, g_alive))
        return pv, delta, gamma


class SurfaceCache:
    """LRU of ``LifeSurface`` objects under a byte budget."""

    def __init__(self, max_bytes: int) -> None:
        if int(max_bytes) <= 0:
            raise ValidationError("surface_cache_bytes must be positive")
        self.max_bytes = int(max_bytes)
        self._store: "OrderedDict[tuple, LifeSurface]" = OrderedDict()
        self._bytes = 0
        self._hits = self._misses = self._evictions = 0

    def get(self, key: tuple) -> Optional[LifeSurface]:
        hit = self._store.get(key)
        if hit is None:
            self._misses += 1
            return None
        self._store.move_to_end(key)
        self._hits += 1
        return hit

    def put(self, key: tuple, surface: LifeSurface) -> None:
        if surface.nbytes > self.max_bytes:
            raise ValidationError(
                f"one life surface needs {surface.nbytes} bytes, more than surface_cache_bytes={self.max_bytes}"
            )
        self._store[key] = surface
        self._bytes += surface.nbytes
        while self._bytes > self.max_bytes:
            _, old = self._store.popitem(last=False)
            self._bytes -= old.nbytes
            self._evictions += 1

    def stats(self) -> Dict[str, Any]:
        return {"hits": self._hits, "misses": self._misses, "evictions": self._evictions,
                "entries": len(self._store), "bytes_used": self._bytes}


class LifeSurfacePricer:
    """One PDE solve per (vol, q, rate) bucket; readout on every day and path."""

    def __init__(
        self, product: Any, *, engine_config: Any, start_date: pd.Timestamp, dates: pd.DatetimeIndex,
        underlying: str, vol_step: float, q_step: float, surface_cache_bytes: int, gate: GateConfig,
        delta_bump_size: Optional[float] = None, gamma_bump_size: Optional[float] = None,
    ) -> None:
        self.product = product
        self.engine_config = engine_config
        self.start_date = pd.Timestamp(start_date).normalize()
        self.dates = pd.DatetimeIndex(dates)
        self.underlying = underlying
        self.vol_step = float(vol_step)
        self.q_step = float(q_step)
        self.gate = gate
        self._engine = create_pricing_engine(product, engine_config, delta_bump_size=delta_bump_size,
                                             gamma_bump_size=gamma_bump_size)
        if not hasattr(self._engine, "solve_life_surface"):
            raise ValidationError("the life_surface provider needs a PDE engine")
        # The exact side of the gate, and the aged-product memo the engine's
        # helpers use; its own small cache never meets the run's states.
        self._exact = RepricingPricer(
            product, engine_config=engine_config, start_date=start_date, underlying=underlying,
            cache=StateCache(CacheConfig(memory_bytes=8_000_000)),
            delta_bump_size=delta_bump_size, gamma_bump_size=gamma_bump_size,
        )
        self._surfaces = SurfaceCache(surface_cache_bytes)
        self._fp = hashlib.blake2b(
            (engine_fingerprint(engine_config, delta_bump_size, gamma_bump_size)
             + repr(("life_surface", self.vol_step, self.q_step))).encode(), digest_size=16,
        ).hexdigest()
        self._maturity = float(product.maturity)
        self._extra_times = tuple(
            t for t in (self.elapsed(d) for d in range(1, len(self.dates)))
            if 0.0 < t < self._maturity
        )
        self._solves = 0
        self._readouts = 0
        self._reservoir: List[DayStates] = []
        self._seen = 0
        seed = int.from_bytes(hashlib.blake2b((self._exact.product_fingerprint + self._fp).encode(),
                                              digest_size=4).digest(), "big")
        self._rng = np.random.default_rng(seed)

    # -- identity -------------------------------------------------------

    @property
    def mode(self) -> str:
        return "life_surface"

    @property
    def product_fingerprint(self) -> str:
        return self._exact.product_fingerprint

    def fingerprint(self) -> str:
        return self._fp

    def aged_product(self, date, *, knocked_in: bool):
        return self._exact.aged_product(date, knocked_in=knocked_in)

    def elapsed(self, d: int) -> float:
        """The tracker's elapsed-time convention: ``(date - start).days / 365``."""
        return max(0.0, (self.dates[d].normalize() - self.start_date).days / 365.0)

    # -- surfaces -------------------------------------------------------

    def _surface(self, vol: float, q: float, rate: float) -> LifeSurface:
        key = (float(vol), float(q), float(rate))
        hit = self._surfaces.get(key)
        if hit is not None:
            return hit
        env = PricingEnvironment(
            spot_quote=SpotQuote(spot=float(self.product.initial_price), asset_name=self.underlying),
            vol_surface=FlatVolSurface(volatility=float(vol)), rate_curve=FlatRateCurve(rate=float(rate)),
            div_yield=SignedDividendYield(float(q)), valuation_date=self.start_date.to_pydatetime(),
        )
        try:
            sol = self._engine.solve_life_surface(self.product, env, extra_times=self._extra_times)
        except Exception as exc:
            raise ValidationError(f"life-surface solve failed at vol={vol!r}, q={q!r}, rate={rate!r}: {exc}") from exc
        self._solves += 1
        d0, g0 = _node_greeks(sol.x, sol.v0)
        d1, g1 = _node_greeks(sol.x, sol.v1)
        t0_delta = t0_gamma = None
        if sol.t0_readout is not None:
            t0_delta, t0_gamma = (a[:, 0] for a in _node_greeks(sol.x, sol.t0_readout[:, None]))
        surface = LifeSurface(t=sol.t, x=sol.x, v0=sol.v0, v1=sol.v1, d0=d0, g0=g0, d1=d1, g1=g1,
                              step_of=sol.step_of, t0_readout=sol.t0_readout, t0_delta=t0_delta, t0_gamma=t0_gamma)
        self._surfaces.put(key, surface)
        return surface

    # -- pricing --------------------------------------------------------

    def price_day(self, states: DayStates) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = len(states)
        pv, delta, gamma = np.empty(m), np.empty(m), np.empty(m)
        if m == 0:
            return pv, delta, gamma
        elapsed = self.elapsed(int(states.day_index))
        vol_c = bucket_centre(states.vol, self.vol_step)
        q_c = bucket_centre(states.q_T, self.q_step)
        rate_c = bucket_centre(states.rate, self.q_step)
        buckets = np.column_stack([bucket_key(states.vol, self.vol_step), bucket_key(states.q_T, self.q_step),
                                   bucket_key(states.rate, self.q_step)])
        x_spot = np.log(np.asarray(states.spot, dtype=float))
        for first in np.unique(buckets, axis=0, return_index=True)[1]:
            i = int(first)
            same = np.all(buckets == buckets[i], axis=1)
            surface = self._surface(float(vol_c[i]), float(q_c[i]), float(rate_c[i]))
            k = 0 if elapsed == 0.0 else surface.node(elapsed)
            pv[same], delta[same], gamma[same] = surface.readout(x_spot[same], k, states.knocked_in[same])
        self._readouts += m
        self._sample(states)
        return pv, delta, gamma

    def _sample(self, states: DayStates) -> None:
        capacity = int(self.gate.sample_states)
        if capacity <= 0:
            return
        for n in range(len(states)):
            self._seen += 1
            if len(self._reservoir) < capacity:
                self._reservoir.append(state_row(states, n))
                continue
            slot = int(self._rng.integers(0, self._seen))
            if slot < capacity:
                self._reservoir[slot] = state_row(states, n)

    def sample_visited(self) -> List[DayStates]:
        return list(self._reservoir)

    def verify(self, samples: Sequence[DayStates], gate: GateConfig, scale: GateScale) -> GateReport:
        """Each sample exactly through the engine at its own spot, vol and dividend object, then the readout."""
        worst_pv = worst_delta = 0.0
        count = 0
        for row in samples:
            if len(row) != 1:
                raise ValidationError("verify takes one-row DayStates (see state_row)")
            pv_e, delta_e, _ = self._exact.price_day(row)
            pv_s, delta_s, _ = self._readout_only(row)
            worst_pv = max(worst_pv, abs(float(pv_s[0]) - float(pv_e[0])) / float(scale.unit_notional) * 1e4)
            worst_delta = max(worst_delta, abs(float(delta_s[0]) - float(delta_e[0])) * float(scale.hands_per_unit_delta))
            count += 1
        return GateReport(
            mode="life_surface", sampled=count, max_pv_gap_bp=worst_pv, max_delta_gap_hands=worst_delta,
            passed=worst_pv <= float(gate.pv_tolerance_bp) and worst_delta <= float(gate.delta_tolerance_hands),
        )

    def _readout_only(self, states: DayStates):
        """``price_day`` without feeding the reservoir (the gate must not sample itself)."""
        seen, reservoir = self._seen, list(self._reservoir)
        try:
            return self.price_day(states)
        finally:
            self._seen, self._reservoir = seen, reservoir

    def stats(self) -> Dict[str, Any]:
        return {
            "provider": "life_surface", "mode": "life_surface", "solves": self._solves,
            "readouts": self._readouts, "engine_calls": self._exact.stats()["engine_calls"],
            "surface_cache": self._surfaces.stats(), "cache": self._exact.stats()["cache"],
        }
```

`ImpliedBasisYield` is unused here; drop it from the import if the linter objects.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_surface.py -q`
Expected: all pass. Two assertions carry stated bounds (`test_the_readout_tracks_the_exact_engine_within_the_gate`, `test_day_zero_reads...`); if one fails, print the measured gap and put it, with margin, in the assertion and the commit message. `test_buckets_key_the_solve_and_the_budget_evicts` expects `budget=1` to raise on the first solve: one surface is larger than one byte.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/pricing/surface.py test/simulation/test_surface.py
git commit -m "feat(backtest/simulation): PDE life-surface provider, one solve per bucket read along the path

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Providers in the engine and tolerances in the oracle

**Files:**
- Modify: `quantark/backtest/simulation/engine.py`
- Modify: `quantark/backtest/simulation/conformance.py`
- Modify: `test/simulation/test_engine.py`, `test/simulation/test_conformance.py`

**Interfaces:**
- `EnsembleBacktestEngine._make_pricers(cache, dates) -> list` choosing `RepricingPricer` or `LifeSurfacePricer` from `config.pricing`.
- `run_oracle(..., pv_tolerance=0.0, delta_tolerance=0.0, contracts_tolerance=0.0)`: lifecycle flags and the active contract always exact; `futures_contracts` exact when `contracts_tolerance == 0`, else compared numerically (an approximate delta can round to a different hand); `OracleReport` gains `max_contracts_gap`.
- The manifest records `provider`, `mode`, `gate`, and for the surface `solves` and `surface_cache`.

- [ ] **Step 1: Write the failing tests**

Append to `test/simulation/test_engine.py`:

```python
from .conftest import ladder_pricing, surface_pricing


def test_ladder_and_surface_runs_keep_the_accounting_identity():
    for pricing in (ladder_pricing(), surface_pricing()):
        results = EnsembleBacktestEngine(ensemble_config(pricing=pricing)).run(_paths(n_paths=6))
        cube = results.cube
        assert cube.total_pnl == pytest.approx(cube.portfolio_value - results.initial_book_value[:, None], abs=1e-6)
        assert results.manifest["mode"] == pricing.mode
        assert results.manifest["gate"]["mode"] == pricing.mode and results.manifest["gate"]["passed"]


def test_the_surface_run_makes_far_fewer_engine_calls():
    exact = EnsembleBacktestEngine(ensemble_config()).run(_paths(n_paths=6))
    surface = EnsembleBacktestEngine(ensemble_config(pricing=surface_pricing())).run(_paths(n_paths=6))
    assert surface.manifest["solves"] >= 1
    assert surface.manifest["solves"] < exact.manifest["engine_calls"] / 4


def test_a_failed_gate_aborts_the_cell_with_its_report():
    from quantark.backtest.simulation.config import GateConfig
    from quantark.backtest.simulation.pricing.base import GateFailure

    coarse = ladder_pricing(spot_step=0.05)
    strict = PricingProviderConfig(provider="repricing", cache=coarse.cache, spot_step=0.05, vol_step=0.0, q_step=0.0,
                                   gate=GateConfig(sample_states=8, pv_tolerance_bp=1e-6, delta_tolerance_hands=1e-6))
    with pytest.raises(GateFailure) as excinfo:
        EnsembleBacktestEngine(ensemble_config(pricing=strict)).run(_paths(n_paths=4))
    assert excinfo.value.report.mode == "ladder" and not excinfo.value.report.passed


def test_a_disk_cache_serves_a_second_run(tmp_path):
    pricing = ladder_pricing(disk_dir=str(tmp_path))
    first = EnsembleBacktestEngine(ensemble_config(pricing=pricing)).run(_paths(n_paths=4))
    second = EnsembleBacktestEngine(ensemble_config(pricing=pricing)).run(_paths(n_paths=4))
    assert second.manifest["engine_calls"] == 0
    assert second.manifest["cache"]["disk"]["disk_hits"] > 0
    assert second.cube.total_pnl == pytest.approx(first.cube.total_pnl)
```

(add `from quantark.backtest.simulation.config import PricingProviderConfig` to the imports.)

Append to `test/simulation/test_conformance.py`:

```python
from .conftest import ladder_pricing, surface_pricing


@pytest.mark.parametrize("pricing, pv_bp, delta_hands", [
    (ladder_pricing(spot_step=0.002), 2.0, 0.5),
    (surface_pricing(vol_step=0.0, q_step=0.0), 50.0, 3.0),
])
def test_approximate_providers_match_the_replay_within_their_gate(pricing, pv_bp, delta_hands):
    cfg = ensemble_config(pricing=pricing)
    book_notional = 1000.0 * SPOT
    report = run_oracle(cfg, _paths(n_paths=3), 1,
                        pv_tolerance=pv_bp * 1e-4 * book_notional,
                        delta_tolerance=delta_hands * 200.0, contracts_tolerance=delta_hands)
    assert report.passed, report.summary()
    assert report.exact_columns_match                      # lifecycle flags and the contract never differ
    assert report.max_pv_gap > 0.0                         # it is an approximation, and the report says so


def test_an_approximate_provider_is_not_bit_exact_by_default():
    report = run_oracle(ensemble_config(pricing=ladder_pricing(spot_step=0.002)), _paths(n_paths=2), 0)
    assert not report.passed and report.max_pv_gap > 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_engine.py test/simulation/test_conformance.py -q`
Expected: failures on `manifest["mode"]`, `contracts_tolerance` and the missing `solves` key (the engine still always builds the exact provider).

- [ ] **Step 3: Wire the providers**

In `engine.py`:

```python
from .pricing.surface import LifeSurfacePricer

    def _make_pricers(self, cache: StateCache, dates: pd.DatetimeIndex) -> List[Any]:
        cfg = self.config
        out: List[Any] = []
        for bp in cfg.products:
            if cfg.pricing.provider == "life_surface":
                out.append(LifeSurfacePricer(
                    bp.product, engine_config=cfg.engine_config, start_date=dates[0], dates=dates,
                    underlying=cfg.underlying, vol_step=cfg.pricing.vol_step, q_step=cfg.pricing.q_step,
                    surface_cache_bytes=cfg.pricing.surface_cache_bytes, gate=cfg.pricing.gate,
                    delta_bump_size=cfg.delta_bump_size, gamma_bump_size=cfg.gamma_bump_size,
                ))
            else:
                out.append(RepricingPricer(
                    bp.product, engine_config=cfg.engine_config, start_date=dates[0], underlying=cfg.underlying,
                    cache=cache, delta_bump_size=cfg.delta_bump_size, gamma_bump_size=cfg.gamma_bump_size,
                    spot_step=cfg.pricing.spot_step, vol_step=cfg.pricing.vol_step, q_step=cfg.pricing.q_step,
                    gate=cfg.pricing.gate,
                ))
        return out
```

Replace the `pricers = [...]` block in `run` with `pricers = self._make_pricers(cache, dates)`. In `_manifest`, set `"provider": cfg.pricing.provider`, `"mode": cfg.pricing.mode`, and add:

```python
            "solves": sum(int(p.stats().get("solves", 0)) for p in pricers),
            "surface_cache": [p.stats()["surface_cache"] for p in pricers if "surface_cache" in p.stats()],
```

`_pricing_q` and `_payoffs` call `pricer.aged_product`, which both providers expose.

In `conformance.py`, add `contracts_tolerance: float = 0.0` to `run_oracle` and `max_contracts_gap: float` to `OracleReport` (and `as_dict`/`summary`); split the columns:

```python
LIFECYCLE_COLUMNS = ("alive", "knocked_in", "knocked_out", "matured", "active_contract")
```

compare `LIFECYCLE_COLUMNS` exactly always; compare `futures_contracts` exactly when `contracts_tolerance == 0.0` (as today, into `exact_ok`) and otherwise as `max_contracts_gap = gap("futures_contracts", expected)` with `passed` requiring `max_contracts_gap <= contracts_tolerance`. `_compare_trades` keeps comparing trades exactly when `contracts_tolerance == 0.0`; with a tolerance, compare only the trade count and dates, and count size mismatches beyond the tolerance (a rounding difference in hands is a size difference, not a missing trade):

```python
def _compare_trades(simulated, expected, mismatches, *, contracts_tolerance: float = 0.0) -> int:
    ...
    for n, (a, b) in enumerate(zip(left, right)):
        if contracts_tolerance == 0.0:
            same = a == b
        else:
            same = a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and abs(a[3] - b[3]) <= contracts_tolerance
        if not same:
            ...
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -n0 test/simulation -q`
Expected: all pass. The two approximate oracle cases carry stated bounds; if one fails, print `report.summary()`, set the bound to the measured gap with margin, and say so in the commit. The exact-mode oracle cases must still report `0.0`.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/engine.py quantark/backtest/simulation/conformance.py \
        test/simulation/test_engine.py test/simulation/test_conformance.py
git commit -m "feat(backtest/simulation): ladder and life-surface providers in the engine, gated and oracle-checked

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Batching over a spawn pool

**Files:**
- Modify: `quantark/backtest/simulation/paths/market_path.py` (`take`)
- Create: `quantark/backtest/simulation/runner.py`
- Modify: `quantark/backtest/simulation/engine.py` (`EnsembleResults.concat` helper lives in `runner.py`; nothing in the engine changes)
- Create: `test/simulation/test_runner.py`; modify `test/simulation/test_market_path.py`

**Design.** `run_ensemble(config, paths)` splits the path indices into consecutive ranges of `batch_paths` (one range when `None`), runs `EnsembleBacktestEngine(config).run(paths.take(range))` for each — in process when `workers == 1`, otherwise in a `ProcessPoolExecutor` with the `spawn` context and a module-level worker function — and concatenates. The split is bit-inert: the plan-2 oracle showed a path's numbers do not depend on which other paths share its engine, and `test_batching_is_bit_inert` pins that on the cube, the trades and the events. Batches keep their own state caches; with a `disk_dir` they share through the shard files (Task 4's lock-merged flush). Manifests are merged: counters summed, gate reports combined, the whole batch's fingerprint kept, per-batch manifests listed.

**Interfaces:**
- `MarketPath.take(indices: Sequence[int]) -> MarketPath` (meta gains `path_indices`).
- `concat_results(parts: Sequence[EnsembleResults], *, path_fingerprint: str) -> EnsembleResults`.
- `run_ensemble(config: EnsembleConfig, paths: MarketPath) -> EnsembleResults`.
- `batch_ranges(n_paths: int, batch_paths: Optional[int]) -> List[range]`.

- [ ] **Step 1: Write the failing tests**

Append to `test/simulation/test_market_path.py`:

```python
def test_take_selects_paths_and_records_them():
    mp = make_market_path(n_paths=5)
    sub = mp.take([4, 1])
    assert sub.n_paths == 2 and sub.n_days == mp.n_days
    assert np.array_equal(sub.spot[0], mp.spot[4]) and np.array_equal(sub.carry[1], mp.carry[1])
    assert sub.meta["path_indices"] == [4, 1]
    with pytest.raises(ValidationError):
        mp.take([5])
    with pytest.raises(ValidationError):
        mp.take([])
```

`test/simulation/test_runner.py`:

```python
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quantark.backtest.simulation.engine import EnsembleBacktestEngine
from quantark.backtest.simulation.paths.gbm import ConstantVol, GBMPaths
from quantark.backtest.simulation.paths.market_path import DEFAULT_TENOR_GRID, StartState, trading_calendar
from quantark.backtest.simulation.runner import batch_ranges, concat_results, run_ensemble

from .conftest import RATE, SPOT, VOL, ensemble_config, flat_carry, ladder_pricing

START = date(2024, 1, 2)
FLOAT_COLUMNS = ("total_pnl", "product_mtm", "delta", "futures_contracts", "cash", "pending_receivable_pv")


def _paths(n_paths=6, n_days=8, sigma=0.6, seed=5):
    start = StartState(spot=SPOT, atm_vol=VOL, rate=RATE, carry=flat_carry(DEFAULT_TENOR_GRID))
    gen = GBMPaths(start=start, calendar=trading_calendar(START, n_days), mu=0.0, sigma=sigma,
                   vol_rule=ConstantVol(VOL), carry_schedule=None, rate=RATE, tenor_grid=DEFAULT_TENOR_GRID)
    return gen.generate(n_paths, n_days, seed=seed)


def _same(a, b):
    for name in FLOAT_COLUMNS:
        assert np.array_equal(getattr(a.cube, name), getattr(b.cube, name)), name
    assert a.cube.active_contract == b.cube.active_contract
    assert np.array_equal(a.last_day, b.last_day) and np.array_equal(a.initial_book_value, b.initial_book_value)
    strip = lambda t: {k: v for k, v in t.items() if k != "date"}
    assert [strip(t) for t in a.trades] == [strip(t) for t in b.trades]
    assert a.events == b.events


def test_batch_ranges():
    assert batch_ranges(7, 3) == [range(0, 3), range(3, 6), range(6, 7)]
    assert batch_ranges(7, None) == [range(0, 7)]
    assert batch_ranges(2, 5) == [range(0, 2)]


def test_batching_is_bit_inert_in_process():
    paths = _paths()
    whole = EnsembleBacktestEngine(ensemble_config()).run(paths)
    batched = run_ensemble(ensemble_config(batch_paths=2), paths)
    _same(whole, batched)
    assert batched.manifest["path_fingerprint"] == paths.fingerprint()
    assert len(batched.manifest["batches"]) == 3
    assert batched.manifest["engine_calls"] == sum(b["engine_calls"] for b in batched.manifest["batches"])


def test_a_spawn_pool_gives_the_same_numbers():
    paths = _paths()
    whole = EnsembleBacktestEngine(ensemble_config()).run(paths)
    pooled = run_ensemble(ensemble_config(workers=2, batch_paths=2), paths)
    _same(whole, pooled)


def test_the_ladder_batches_share_states_through_the_disk_tier(tmp_path):
    paths = _paths()
    cfg = ensemble_config(pricing=ladder_pricing(disk_dir=str(tmp_path)), batch_paths=3)
    first = run_ensemble(cfg, paths)
    second = run_ensemble(cfg, paths)
    assert second.manifest["engine_calls"] == 0
    _same(first, second)


def test_concat_remaps_path_indices():
    paths = _paths(n_paths=4)
    a = EnsembleBacktestEngine(ensemble_config()).run(paths.take([0, 1]))
    b = EnsembleBacktestEngine(ensemble_config()).run(paths.take([2, 3]))
    whole = concat_results([a, b], path_fingerprint=paths.fingerprint())
    assert whole.n_paths == 4
    assert {t["path"] for t in whole.trades} <= {0, 1, 2, 3}
    assert any(t["path"] >= 2 for t in whole.trades) or not b.trades
    assert whole.path_states(3).equals(b.path_states(1))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_market_path.py test/simulation/test_runner.py -q`
Expected: `AttributeError: 'MarketPath' object has no attribute 'take'` and `ModuleNotFoundError: ... runner`.

- [ ] **Step 3: Add `MarketPath.take`**

In `market_path.py`, after `path`:

```python
    def take(self, indices: Sequence[int]) -> "MarketPath":
        """The sub-batch of the given paths, in the given order (meta carries ``path_indices``)."""
        idx = [int(i) for i in indices]
        if not idx:
            raise ValidationError("take needs at least one path index")
        if any(not 0 <= i < self.n_paths for i in idx):
            raise ValidationError(f"path indices {idx} out of range for {self.n_paths} paths")
        return MarketPath(
            dates=self.dates, spot=self.spot[idx], atm_vol=self.atm_vol[idx], rate=self.rate[idx],
            carry=self.carry[idx], tenor_grid=self.tenor_grid, meta={**self.meta, "path_indices": idx},
        )
```

(add `Sequence` to the `typing` import.)

- [ ] **Step 4: Write `runner.py`**

```python
"""Batching the ensemble over path ranges, in process or in a spawn pool (spec 11).

The split is bit-inert -- a path's numbers do not depend on which other
paths share its engine (the plan-2 oracle pinned that) -- so batching is a
memory and parallelism knob, not a numerical one, and the runner test
compares batched and unbatched cubes with ``array_equal``.
"""
from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from quantark.util.exceptions import ValidationError

from .config import EnsembleConfig
from .engine import BOOL_COLUMNS, FLOAT_COLUMNS, EnsembleBacktestEngine, EnsembleResults, StateCube
from .paths.market_path import MarketPath
from .pricing.base import GateReport


def batch_ranges(n_paths: int, batch_paths: Optional[int]) -> List[range]:
    """Consecutive index ranges of at most ``batch_paths`` paths (one range when None)."""
    if n_paths < 1:
        raise ValidationError("n_paths must be positive")
    size = n_paths if batch_paths is None else int(batch_paths)
    if size < 1:
        raise ValidationError("batch_paths must be at least 1")
    return [range(start, min(start + size, n_paths)) for start in range(0, n_paths, size)]


def _run_batch(config: EnsembleConfig, batch: MarketPath) -> EnsembleResults:
    """Module-level so a spawn worker can import it by reference."""
    return EnsembleBacktestEngine(config).run(batch)


def concat_results(parts: Sequence[EnsembleResults], *, path_fingerprint: str) -> EnsembleResults:
    """Stack per-batch results into one, paths renumbered in batch order."""
    parts = list(parts)
    if not parts:
        raise ValidationError("concat_results needs at least one batch")
    first = parts[0].cube
    for other in parts[1:]:
        if other.cube.active_contract != first.active_contract or not other.cube.dates.equals(first.dates):
            raise ValidationError("batches disagree on the calendar or the active contract; they are not one run")
    n_paths = sum(p.n_paths for p in parts)
    cube = StateCube(first.dates, n_paths)
    cube.active_contract = list(first.active_contract)
    for name in FLOAT_COLUMNS + BOOL_COLUMNS:
        setattr(cube, name, np.concatenate([getattr(p.cube, name) for p in parts], axis=0))
    trades: List[Dict[str, Any]] = []
    events = []
    offset = 0
    for part in parts:
        trades += [{**t, "path": int(t["path"]) + offset} for t in part.trades]
        events += [replace(e, path=e.path + offset) for e in part.events]
        offset += part.n_paths
    manifest = _merge_manifests([p.manifest for p in parts], path_fingerprint)
    return EnsembleResults(
        cube=cube, trades=trades, events=events, manifest=manifest,
        last_day=np.concatenate([p.last_day for p in parts]),
        initial_book_value=np.concatenate([p.initial_book_value for p in parts]),
    )


def _merge_manifests(manifests: Sequence[Dict[str, Any]], path_fingerprint: str) -> Dict[str, Any]:
    head = dict(manifests[0])
    merged = {
        **head,
        "path_fingerprint": path_fingerprint,
        "days_run": max(int(m["days_run"]) for m in manifests),
        "engine_calls": sum(int(m["engine_calls"]) for m in manifests),
        "solves": sum(int(m.get("solves", 0)) for m in manifests),
        "dividend_builds": sum(int(m["dividend_builds"]) for m in manifests),
        "data_end_paths": sum(int(m["data_end_paths"]) for m in manifests),
        "seconds": sum(float(m["seconds"]) for m in manifests),
        "gate": GateReport.combine([GateReport(**m["gate"]) for m in manifests]).as_dict(),
        "cache": _sum_counters([m["cache"] for m in manifests]),
        "batches": [dict(m) for m in manifests],
    }
    return merged


def _sum_counters(dicts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for d in dicts:
        for key, value in d.items():
            if isinstance(value, dict):
                out[key] = _sum_counters([out.get(key, {}), value]) if key in out else dict(value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                out[key] = out.get(key, 0) + value
            else:
                out[key] = value
    return out


def run_ensemble(config: EnsembleConfig, paths: MarketPath) -> EnsembleResults:
    """Run one cell over ``paths`` in batches of ``config.batch_paths`` on ``config.workers`` processes."""
    ranges = batch_ranges(paths.n_paths, config.batch_paths)
    batches = [paths.take(list(r)) for r in ranges]
    if int(config.workers) == 1 or len(batches) == 1:
        parts = [_run_batch(config, batch) for batch in batches]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=int(config.workers), mp_context=context) as pool:
            futures = [pool.submit(_run_batch, config, batch) for batch in batches]
            parts = []
            for future in futures:
                try:
                    parts.append(future.result())
                except Exception as exc:
                    for other in futures:
                        other.cancel()
                    raise ValidationError(f"a batch worker failed: {type(exc).__name__}: {exc}") from exc
    if len(parts) == 1:
        part = parts[0]
        part.manifest["path_fingerprint"] = paths.fingerprint()
        part.manifest.setdefault("batches", [dict(part.manifest)])
        return part
    return concat_results(parts, path_fingerprint=paths.fingerprint())
```

`GateReport(**m["gate"])` needs `as_dict` keys to match the constructor (`mode, sampled, max_pv_gap_bp, max_delta_gap_hands, passed`) — they do.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_market_path.py test/simulation/test_runner.py -q`
Expected: all pass. The spawn test starts two interpreters that import `quantark` (a few seconds). If it hangs, check the memory note in `reference_stdin_spawn_trap`: pytest test modules are importable files, so the worker can unpickle `_run_batch`; a hang means a non-picklable object in the config — print `pickle.dumps(cfg)` to find it.

- [ ] **Step 6: Commit**

```bash
git add quantark/backtest/simulation/paths/market_path.py quantark/backtest/simulation/runner.py \
        test/simulation/test_market_path.py test/simulation/test_runner.py
git commit -m "feat(backtest/simulation): path batching over a spawn pool, bit-inert and manifest-merged

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: Exports, module guide, full regression

**Files:**
- Modify: `quantark/backtest/simulation/__init__.py`, `quantark/backtest/simulation/README.md`
- Modify: `test/simulation/test_market_path.py`

- [ ] **Step 1: Extend the failing export test**

Add to the name list in `test_public_api_is_exported`:

```python
                 "LifeSurfacePricer", "SurfaceCache", "GateScale", "GateFailure", "run_ensemble",
                 "concat_results", "batch_ranges", "DiskTier", "bucket_key", "bucket_centre",
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_market_path.py -k public_api -q`
Expected: `AssertionError: LifeSurfacePricer`.

- [ ] **Step 3: Export and document**

`__init__.py`: import and list `LifeSurfacePricer`, `SurfaceCache` (from `.pricing.surface`), `GateScale`, `GateFailure`, `bucket_key`, `bucket_centre` (from `.pricing.base`), `DiskTier` (from `.pricing.cache`), `run_ensemble`, `concat_results`, `batch_ranges` (from `.runner`); extend the module docstring with a plan-3 sentence.

`README.md`: extend "How a state gets priced" with two paragraphs (ladder mode: `S_ref`, the two bracketing nodes, linear in log-spot, bucket centres, why the same node keys across cells; the disk tier: shard naming, lock-merged flush, foreign shards) and add three sections before "Why you can trust it":

- **The life surface** — one solve per (vol, q, rate) bucket at the start date with a node on every simulation day; readout by slab and node; delta and gamma by the solver's stencil per node; spot outside the grid fails; the rate is bucketed with `q_step`; the surface's `q` is flat, so under a term dividend source the gate is what tells you whether the flat reading was good enough.
- **The gate** — reservoir of visited states, seeded from the fingerprints; `verify` on day 0 and on the reservoir at the end; bp of unit notional and hands; a failure aborts the cell and nothing is returned; exact mode reports zero.
- **Batching** — `run_ensemble`, `workers`, `batch_paths`, the spawn context, bit-inert split, manifests merged, the disk tier as the way batches (and later cells) share states.

Update "Why you can trust it" for the tolerance parameters and `contracts_tolerance`, and "Notes on the design" with the numeric-clock settlement rule (Task 1) and the `extra_times` change outside the package. Replace "What comes next" with plan 4 only.

- [ ] **Step 4: Run the full regression**

Run: `.venv/bin/python -m pytest test/simulation test/test_lifecycle_calendar_schedule.py test/test_snowball_life_surface.py test/pde_grid test/replay_golden test/test_replay_dividend_source.py test/test_snowball_q_term_structure_study.py -q`
Then the whole suite: `.venv/bin/python -m pytest -q`
Expected: green apart from the two pre-existing `test/mo_volmodels/test_dashboard.py` failures. Before staging, `git checkout -- example/mo_volmodels/data/`.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation test/simulation
git commit -m "feat(backtest/simulation): public API and module guide for the approximate providers and batching

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review against the spec

- **§3 the one change outside the package** → Task 6 (`GridRequest.extra_times`, nodes without events or damping; default unchanged and pinned) plus the solver seam in Task 7, which the spec's "extra time nodes" implies but does not name.
- **§7.2 `LifeSurfacePricer`** → Tasks 7–8: one solve per bucket, `V0`/`V1` slabs and grid vectors copied, the solver's stencil for delta (converted by `1/S` at the node) and gamma, linear interpolation in `x`, KI selects the slab, spot outside the grid aborts, buckets by `vol_step`/`q_step` at their centres, flat `q` at the remaining maturity, `LifeSurface` LRU by bytes with hit/miss counts.
- **§7.3 ladder mode** → Task 3: `S_j = S_ref · exp(j · spot_step)`, each bracketing node priced once, PV and delta linear between them, vol and `q` bucketed with zero meaning exact.
- **§7.4 disk tier** → Task 4: one npz shard per (product, engine), lazily loaded, foreign header a miss never reinterpreted; the cross-cell reuse rule is served by the shards.
- **§7.5 gate** → Task 5 (and the surface's `verify` in Task 8): sample before the cell runs and on visited states; exact spot, vol and `q` on the exact side; bp of notional and hands; failure aborts the cell; exact mode identically zero.
- **§8 settlement** → Task 1 closes the plan-2 receivable TODO on the numeric clock.
- **§11 configuration** → Task 2 adds `spot_step`, `vol_step`, `q_step`, `disk_dir`, `workers`, `batch_paths` with the validation matrix (`life_surface` with a non-PDE engine; `spot_step` with `life_surface`; `workers < 1`).
- **§12 error handling** → Tasks 1, 4, 5, 8, 10: each abort is a `ValidationError` (or `GateFailure`) naming the state.
- **§13 testing** → life surface (readout vs fresh solve, KI slab, delta, grid coverage), repricing (ladder within tolerance, disk round trip), lifecycle (settlement lag booking and receivable PV, cross-checked against the tracker), conformance (surface and ladder within the gate, exact still bit-for-bit).
- **§16 delivery order** → this plan is items 3 and 4; results and the study are plan 4.
- **Type consistency:** `DayStates` is unchanged; `StateKey` is unchanged (ladder keys use integer bucket indices in `spot_key`/`vol_key` and a `(rate, q_centre)` row key in `env_key`); `verify(samples, gate, scale)` and `sample_visited()` have the same signature on both providers and the protocol; `aged_product`, `product_fingerprint`, `fingerprint`, `stats`, `mode` exist on both providers; `receivable_pv(lifecycle, rate, day_index)` is the only signature used in Task 1's test and the engine; `GateReport.combine` consumes the dicts `as_dict` writes.
- **Deliberately not in this plan:** results distributions, persistence and the example study (plan 4); multi-cell scheduling; a CALENDAR_DAYS settlement fixture (the factory has no dated schedule for it) and a delayed-settlement oracle (the PDE engine refuses conventions).
