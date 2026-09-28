# Intraday Daily-KI Greek Certification and Near-KI Example Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **RETIRED 2026-09-18 from Task 3 onward.** Certification moved to `quantark.modelvalidation` under
> [the unified spec](../specs/2026-09-18-intraday-modelvalidation-certification-design.md): the pilot, sweep, evidence
> packaging, matrix regeneration and certificate documentation below build artifacts that migration deletes, and the
> Gate C records lack the schema-2 fields. Launch no further Gate C run. Tasks 1 and 2 landed (1831fe73, 5bd9671d);
> the daily-KI contract, fixture and near-KI example carry into the unified study.

**Goal:** Certify QUAD V2 intraday Greeks for a daily-observed-KI snowball on one trading day, then show, with an
example and an HTML report artifact, how its PV and Greeks move while the spot streams around the KI level.

**Architecture:** A new Gate C fixture (`snowball_daily_ki`) goes through the existing Greek harness on its own
horizon ladder and profile. A pilot of 6 groups gates a 154-group sweep. The sweep's rows merge into the packaged
evidence under a guard that refuses to change any of the 50 existing certificates. The example builds the certified
contract itself, streams ticks through `value_intraday` and `spot_curve`, and writes a JSON run record. A generator
under `tmp/` turns that record into the report page.

**Tech Stack:** Python 3.11, pytest + xdist, NumPy/SciPy (independent reference), `quantark.intraday`,
`tmp/rss_guard.py`, and the Artifact tool for the report.

**Spec:** `docs/superpowers/specs/2026-09-17-intraday-daily-ki-certification-design.md`

## Global Constraints

- Worktree: `/Users/fuxinyao/quant-ark/.claude/worktrees/intraday-plan1`, branch `worktree-intraday-plan1`. Run every command from it. Python: `/Users/fuxinyao/quant-ark/.venv/bin/python` (written `$PY` below). Set `WT=/Users/fuxinyao/quant-ark/.claude/worktrees/intraday-plan1`.
- Worktree source goes on the path: `PYTHONPATH=$WT` for pytest. Standalone scripts that import `intraday.*` also need `$WT/test`.
- `pytest.ini` defaults to `-n auto`. **Always pass `-n 4` or fewer.** A wide pool once used 75 GB on this 48 GB machine.
- Gate C runs only under `tmp/rss_guard.py`, with at most 4 workers and `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. Long runs are wrapped in `nohup caffeinate -i -m -s`. A killed run is stopped by killing its **process group**, never with `pkill -f`.
- Other sessions share this worktree's index, so **commit by pathspec** (`git commit -- <paths>`). `docs/` is excluded by `.git/info/exclude`, so a new file there needs `git add -f`. Tracked docs files commit through the pathspec.
- Never `git add example/` wholesale. After any full-suite run, `git checkout -- example/mo_volmodels/data/mo_barrier_sample.json example/mo_volmodels/data/mo_calibration_explainer_sample.html`.
- The frozen budgets in `test/intraday/reference/budgets.py` (including `GATE_C_POINTS`) are **never edited**.
- Certified contract (verbatim from the spec): initial 100, strike 100, contract multiplier 1.0, `initial_date` 2026-03-16, `exercise_date` 2027-03-16, KO 103 on the 12 monthly dates, KI 75 at every SSE business day `2026-03-16 < d <= 2027-03-16`, `ko_rate` = `rebate_rate` = 0.12, `include_principal` False. Market: `FlatRateCurve(0.03)`, `ContinuousDividendYield(0.01)`, `FlatVolSurface(0.20)`. Calendar: `create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))`, +08:00, sessions 09:30–11:30 and 13:00–15:00. Profile: `VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))`. Engine: `SnowballQuadEngineV2()` (defaults `order=8, cells_per_sd=2.0`).
- Certified day: the KI close at 2026-09-10 15:00 +08:00. Horizons: 6 h, 1 h, 15 min, 5 min, 1 min, 10 s, 1 s. That's 154 groups (7 × 11 offsets × `ko`/`ki`).
- Commit trailer: `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- Merge blocker outside this plan: the 7 PDE golden / banked-certificate failures introduced by `e0ba6ba7`. This plan never touches a PDE route. Don't try to fix them here.

---

### Task 1: The `snowball_daily_ki` fixture

**Files:**
- Modify: `test/intraday/gate_c/cells.py` (constants block near line 18; `BARRIERS`/`MONITORING` near line 30; `product`, `notional`, `fixing_and_history`)
- Create: `test/intraday/gate_c/test_daily_ki_fixture.py`

**Interfaces:**
- Consumes: `intraday.conftest.dated_snowball`, `harness.build_context(cell) -> (ctx, sw)`, `quantark.intraday.capability.economic_identity(ctx) -> str`
- Produces: `cells.DAILY_KI_DAY: datetime`, `cells.DAILY_KI_HORIZONS: tuple[timedelta, ...]`. `cells.product("snowball_daily_ki")`, `cells.fixing_and_history("snowball_daily_ki") -> (datetime, tuple[Fixing, ...])`, and entries in `BARRIERS`, `MONITORING` and `notional`.

- [ ] **Step 1: Write the failing tests**

Create `test/intraday/gate_c/test_daily_ki_fixture.py`:

```python
"""The daily-KI Gate C fixture: KI at every SSE close, certified on the 2026-09-10 close (design 2026-09-17)."""
from datetime import datetime, timedelta

from quantark.intraday.capability import economic_identity

from intraday.conftest import SHANGHAI, dated_snowball
from intraday.gate_c import cells as C
from intraday.gate_c.harness import build_context


def test_daily_ki_fixture_observes_ki_at_every_close_and_keeps_the_monthly_terms():
    daily, monthly = C.product("snowball_daily_ki"), dated_snowball(C.sse().calendar, C.T0)
    records = daily.barrier_config.ki_observation_schedule.records
    dates = [r.observation_date for r in records]
    assert len(records) == 243 and {r.barrier for r in records} == {75.0}
    assert dates[0] == datetime(2026, 3, 17) and dates[-1] == daily.exercise_date == datetime(2027, 3, 16)
    assert dates == sorted(set(dates)) and all(C.sse().calendar.is_business_day(d) for d in dates)
    ko_daily = [(r.observation_date, r.barrier) for r in daily.barrier_config.ko_observation_schedule.records]
    ko_monthly = [(r.observation_date, r.barrier) for r in monthly.barrier_config.ko_observation_schedule.records]
    assert ko_daily == ko_monthly and len(ko_daily) == 12
    assert (C.BARRIERS["snowball_daily_ki"], C.MONITORING["snowball_daily_ki"], C.notional("snowball_daily_ki")) == \
        (("ko", "ki"), "discrete", 100.0)


def test_daily_ki_fixing_is_the_2026_09_10_close_after_122_confirmed_closes():
    fixing, history = C.fixing_and_history("snowball_daily_ki")
    assert fixing == datetime(2026, 9, 10, 15, 0, tzinfo=SHANGHAI)
    assert len(history) == 122 and {f.value for f in history} == {100.0}
    assert all(f.timestamp < fixing for f in history)


def test_every_daily_ki_horizon_resolves_the_same_alive_claim_inside_one_gap():
    assert max(C.DAILY_KI_HORIZONS) < timedelta(days=1)      # a one-day rung would sit on the previous close
    identities, remaining = set(), set()
    for horizon in C.DAILY_KI_HORIZONS:
        ctx, _ = build_context(C.Cell("snowball_daily_ki", "quad_v2", "desk", horizon, "eq", "ki"))
        assert not ctx.numerical.knocked_in and not ctx.provisional
        identities.add(economic_identity(ctx))
        remaining.add(len(ctx.numerical.remaining_events))
    assert len(identities) == 1 and remaining == {129}
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `PYTHONPATH=$WT $PY -m pytest -n 0 -q -p no:cacheprovider test/intraday/gate_c/test_daily_ki_fixture.py`
Expected: 3 FAIL. The first fails with `ValueError: snowball_daily_ki` from `product`, and the others with `KeyError`/`ValueError`.

- [ ] **Step 3: Implement the fixture**

In `test/intraday/gate_c/cells.py`, add these imports at the top:

```python
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord
```

After `HORIZONS = (...)`, add:

```python
#: The daily-KI fixture's certified close and its ladder. Every rung sits inside the 24 h gap after the previous
#: close (2026-09-09 15:00): a one-day rung would value exactly at that close, itself an event.
DAILY_KI_DAY = datetime(2026, 9, 10)
DAILY_KI_HORIZONS = (timedelta(hours=6), timedelta(hours=1), timedelta(minutes=15), timedelta(minutes=5),
                     timedelta(minutes=1), timedelta(seconds=10), timedelta(seconds=1))
```

Extend the two maps (keep the existing entries):

```python
BARRIERS = {"snowball_discrete_ki": ("ko", "ki"), "digital": ("strike",), "barrier_uo_zero_carry": ("ko",),
            "one_touch_zero_carry": ("ko",), "snowball_long_gap": ("ko", "ki"), "snowball_daily_ki": ("ko", "ki")}
MONITORING = {"snowball_discrete_ki": "discrete", "digital": "terminal",
              "barrier_uo_zero_carry": "continuous", "one_touch_zero_carry": "continuous",
              "snowball_long_gap": "discrete", "snowball_daily_ki": "discrete"}
```

In `product(name)`, add before `if name == "digital":`:

```python
    if name == "snowball_daily_ki":
        # The monthly fixture's terms with KI 75 observed at EVERY SSE close after the trade date: the contract
        # desks book. Its remaining events differ from the monthly fixture's, so it earns its own certificate.
        prod = dated_snowball(sse().calendar, T0)
        cal, maturity = sse().calendar, prod.exercise_date
        closes = [d for d in (T0 + timedelta(days=k) for k in range(1, (maturity - T0).days + 1)) if cal.is_business_day(d)]
        prod.barrier_config.ki_observation_schedule.records = [ObservationRecord(observation_date=d, barrier=75.0)
                                                               for d in closes]
        return prod
```

In `notional(name)`, add `"snowball_daily_ki": 100.0` to the dict.

In `fixing_and_history(name)`, add before the existing `if name in ("snowball_discrete_ki", "snowball_long_gap"):` block:

```python
    if name == "snowball_daily_ki":
        fixing = next(e.timestamp for e in probe.timeline.events
                      if e.kind is EventKind.KI and e.timestamp.date() == DAILY_KI_DAY.date())
        history = sorted({e.timestamp for e in probe.timeline.events if e.timestamp < fixing})
        return fixing, tuple(Fixing(t, 100.0) for t in history)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `PYTHONPATH=$WT $PY -m pytest -n 0 -q -p no:cacheprovider test/intraday/gate_c/test_daily_ki_fixture.py test/intraday/gate_c/test_gate_c_cells.py`
Expected: all PASS. (`test_gate_c_cells.py` is unchanged: `PRODUCTS` does not list the new fixture, so the 3432 price cells stay the same.)

If `remaining == {129}` fails, print the event kinds of `ctx.numerical.remaining_events` and stop. The spec's probe counted 121 KI + 7 KO + 1 terminal. Don't change the assertion to fit.

No commit yet. Task 2 lands together with this in commit (1).

---

### Task 2: Harness ladders, certificate labels, evidence-test filter, matrix column (commit 1)

**Files:**
- Modify: `test/intraday/gate_c/greek_harness.py` (constants lines 50–71, `greek_groups` lines 90–94, `demonstrated` lines 538–599)
- Modify: `test/intraday/test_capability_evidence.py` (the Greek test near line 83)
- Modify: `quantark/intraday/publish.py` (`_greek_section` lines 39–72)
- Create: `test/intraday/gate_c/test_daily_ki_harness.py`, `test/intraday/test_publish_greek_rows.py`
- Regenerate: `docs/execution/intraday-capability-matrix.md`

**Interfaces:**
- Consumes: Task 1's `C.DAILY_KI_HORIZONS`, `C.BARRIERS["snowball_daily_ki"]`, `C.MONITORING`
- Produces: `greek_harness.FIXTURE_PROFILES: dict[str, tuple[str, ...]]` and `greek_harness.fixture_horizons(fixture: str) -> tuple[timedelta, ...]`. Every `demonstrated()` row gains `"fixtures": list[str]`. `publish._greek_sources(evidence: dict) -> str`. `test_capability_evidence._backing_cells(payload: dict, row: dict) -> list[dict]`.

- [ ] **Step 1: Write the failing harness tests**

Create `test/intraday/gate_c/test_daily_ki_harness.py`:

```python
"""The Greek harness sweeps the daily-KI fixture on its own ladder and profile and labels its certificates."""
from collections import Counter

from intraday.gate_c import cells as C
from intraday.gate_c.greek_harness import demonstrated, fixture_horizons, greek_groups


def test_daily_ki_groups_are_desk_only_on_their_own_ladder():
    groups = [g for g in greek_groups() if g.product == "snowball_daily_ki"]
    assert len(groups) == 154
    assert {g.profile for g in groups} == {"desk"}
    assert {g.horizon for g in groups} == set(C.DAILY_KI_HORIZONS)
    assert {g.barrier for g in groups} == {"ko", "ki"} and {g.offset for g in groups} == set(C.SPOT_OFFSETS)


def test_existing_fixture_groups_are_unchanged():
    counts = Counter(g.product for g in greek_groups())
    assert (counts["snowball_discrete_ki"], counts["snowball_long_gap"], counts["digital"]) == (528, 220, 352)


def _row(fixture, horizon_s, offset, barrier, status, identity):
    cell = {"product": fixture, "engine": "quad_v2", "profile": "desk", "horizon": horizon_s, "offset": offset,
            "barrier": barrier, "id": f"{fixture}-quad_v2-desk-{horizon_s}s-{offset}-{barrier}"}
    return {"cell": cell, "route": "QuadV2Route", "product": "SnowballOption", "settings": {"engine": "E"},
            "market": {}, "economic_identity": identity,
            "measures": [{"measure": "point_delta", "status": status, "measure_settings": {}}]}


def _sweep(fixture, identity, fail=None):
    return [_row(fixture, int(h.total_seconds()), o, b,
                 "failed" if (int(h.total_seconds()), o, b) == fail else "passed", identity)
            for h in fixture_horizons(fixture) for o in C.SPOT_OFFSETS for b in C.BARRIERS[fixture]]


def test_demonstrated_labels_rows_with_their_fixtures_and_keeps_families_apart():
    rows = demonstrated(_sweep("snowball_daily_ki", "daily") + _sweep("snowball_discrete_ki", "monthly"))
    by_identity = {r["economic_identity"]: r for r in rows}
    assert by_identity["daily"]["fixtures"] == ["snowball_daily_ki"]
    assert (by_identity["daily"]["horizon_s"], by_identity["daily"]["horizon_max_s"]) == (1, 21600)
    assert by_identity["monthly"]["fixtures"] == ["snowball_discrete_ki"]
    assert by_identity["monthly"]["horizon_max_s"] == 29 * 86400


def test_a_daily_ki_miss_splits_only_the_daily_window():
    rows = demonstrated(_sweep("snowball_daily_ki", "daily", fail=(300, "eq", "ki"))
                        + _sweep("snowball_discrete_ki", "monthly"))
    daily = sorted((r["horizon_s"], r["horizon_max_s"]) for r in rows if r["economic_identity"] == "daily")
    assert daily == [(1, 60), (900, 21600)]
    assert [(r["horizon_s"], r["horizon_max_s"]) for r in rows if r["economic_identity"] == "monthly"] == [(1, 29 * 86400)]
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `PYTHONPATH=$WT $PY -m pytest -n 0 -q -p no:cacheprovider test/intraday/gate_c/test_daily_ki_harness.py`
Expected: collection ERROR, `ImportError: cannot import name 'fixture_horizons'`.

- [ ] **Step 3: Implement the harness changes**

In `test/intraday/gate_c/greek_harness.py`:

Replace the `GREEK_ENGINES`, `ROUTE_NAMES` and `PRODUCT_NAMES` definitions with:

```python
GREEK_ENGINES = {"snowball_discrete_ki": ("quad_v2", "pde", "mc_rqmc"), "digital": ("analytical", "mc_rqmc"),
                 "snowball_long_gap": ("quad_v2",), "snowball_daily_ki": ("quad_v2",)}
```

```python
ROUTE_NAMES = {("snowball_discrete_ki", "quad_v2"): "QuadV2Route", ("snowball_discrete_ki", "pde"): "PDERoute",
               ("snowball_discrete_ki", "mc_rqmc"): "MCRoute", ("digital", "analytical"): "AnalyticalDigitalRoute",
               ("digital", "mc_rqmc"): "MCRoute", ("snowball_long_gap", "quad_v2"): "QuadV2Route",
               ("snowball_daily_ki", "quad_v2"): "QuadV2Route"}
PRODUCT_NAMES = {"snowball_discrete_ki": "SnowballOption", "snowball_long_gap": "SnowballOption",
                 "snowball_daily_ki": "SnowballOption", "digital": "CashOrNothingDigitalOption"}
#: Profiles swept per fixture where they differ from GREEK_PROFILES: the daily-KI certificate is desk only.
FIXTURE_PROFILES = {"snowball_daily_ki": ("desk",)}


def fixture_horizons(fixture: str):
    """The horizon ladder a fixture is swept on, which is also the rungs a window of its certificates must cover."""
    if fixture == "snowball_long_gap":
        return LONG_GAP_HORIZONS
    if fixture == "snowball_daily_ki":
        return C.DAILY_KI_HORIZONS
    if fixture == "digital":
        return tuple(sorted(set(GREEK_HORIZONS + LONG_GAP_HORIZONS)))
    return GREEK_HORIZONS
```

Replace `greek_groups`:

```python
def greek_groups():
    return [GreekGroup(p, prof, h, o, b) for p in GREEK_ENGINES for prof in FIXTURE_PROFILES.get(p, GREEK_PROFILES)
            for h in fixture_horizons(p) for o in C.SPOT_OFFSETS for b in C.BARRIERS[p]]
```

In `demonstrated`, replace the `expected = {...}` statement with:

```python
        expected = {int(h.total_seconds()) for fixture in fixtures for h in fixture_horizons(fixture)}
```

and add the fixture label to the emitted row dict (next to `"economic_identity": economics,`):

```python
                        "fixtures": sorted(fixtures),
```

- [ ] **Step 4: Run the harness tests and confirm they pass**

Run: `PYTHONPATH=$WT $PY -m pytest -n 0 -q -p no:cacheprovider test/intraday/gate_c/test_daily_ki_harness.py`
Expected: 4 PASS.

- [ ] **Step 5: Write the failing evidence-filter and publish tests**

Append to `test/intraday/test_capability_evidence.py`:

```python
def test_backing_cells_never_mix_economic_families():
    row = {"product": "SnowballOption", "route": "QuadV2Route", "profile": "desk", "settings": {"engine": "E"},
           "monitoring": "discrete", "economic_identity": "monthly", "horizon_s": 1, "horizon_max_s": 3600}

    def cell(fixture, identity, status):
        return {"product": "SnowballOption", "route": "QuadV2Route", "settings": {"engine": "E"},
                "economic_identity": identity, "cell": {"product": fixture, "profile": "desk", "horizon": 60},
                "measures": [{"measure": "point_delta", "status": status}]}

    payload = {"cells": [cell("snowball_discrete_ki", "monthly", "passed"),
                         cell("snowball_daily_ki", "daily", "unqualified")]}
    assert [c["economic_identity"] for c in _backing_cells(payload, row)] == ["monthly"]
```

Create `test/intraday/test_publish_greek_rows.py`:

```python
"""The generated matrix names each certificate's fixture and every evidence source."""
from quantark.intraday import publish


def test_greek_rows_name_their_fixtures_and_every_evidence_source(monkeypatch):
    evidence = {"git_sha": "bbbb", "sources": [{"git_sha": "aaaa", "cells": 10}, {"git_sha": "bbbb", "cells": 154}],
                "demonstrated": [{"product": "SnowballOption", "route": "QuadV2Route", "measure": "point_delta",
                                  "monitoring": "discrete", "profile": "desk", "settings": {"engine": "x.SnowballQuadEngineV2"},
                                  "measure_settings": {}, "horizon_s": 1, "horizon_max_s": 21600, "offsets": ["eq"],
                                  "barriers": ["ki", "ko"], "fixtures": ["snowball_daily_ki"]}]}
    monkeypatch.setattr(publish, "greek_evidence", lambda: evidence)
    text = publish._greek_section()
    assert "(git aaaa, 10 cells; git bbbb, 154 cells)" in text
    assert "| SnowballOption | snowball_daily_ki | QuadV2Route | point_delta |" in text


def test_rows_without_a_fixture_label_render_a_dash(monkeypatch):
    evidence = {"git_sha": "cccc", "demonstrated": [{"product": "P", "route": "R", "measure": "m", "settings": {},
                                                     "horizon_s": 1, "horizon_max_s": 2}]}
    monkeypatch.setattr(publish, "greek_evidence", lambda: evidence)
    text = publish._greek_section()
    assert "(git cccc)" in text and "| P | — | R | m |" in text
```

- [ ] **Step 6: Run them and confirm they fail**

Run: `PYTHONPATH=$WT $PY -m pytest -n 0 -q -p no:cacheprovider test/intraday/test_capability_evidence.py::test_backing_cells_never_mix_economic_families test/intraday/test_publish_greek_rows.py`
Expected: 3 FAIL: `NameError: name '_backing_cells' is not defined`, and two assertion failures on the missing Fixture column and sources.

- [ ] **Step 7: Implement the filter and the matrix column**

In `test/intraday/test_capability_evidence.py`, add this helper above `test_greek_evidence_parses_and_every_demonstration_is_backed_by_passing_cells`:

```python
def _backing_cells(payload, row):
    """The cells a demonstrated row rests on: its whole family, economic identity and monitoring included, in its window.

    Product, route, profile and settings alone would let the daily-KI cells back (and break) the monthly rows.
    """
    from intraday.gate_c.cells import MONITORING
    return [c for c in payload["cells"] if (c["product"], c["route"]) == (row["product"], row["route"])
            and c["cell"]["profile"] == row["profile"] and c.get("settings") == row["settings"]
            and MONITORING[c["cell"]["product"]] == row["monitoring"]
            and c.get("economic_identity", "") == row.get("economic_identity", "")
            and row["horizon_s"] <= c["cell"]["horizon"] <= row["horizon_max_s"]]
```

In that test, replace the `cells = [c for c in payload["cells"] if ...]` statement (all five lines) with:

```python
        cells = _backing_cells(payload, row)
```

In `quantark/intraday/publish.py`, add above `_greek_section`:

```python
def _greek_sources(evidence: dict) -> str:
    """Every measured revision the packaged Greek evidence rests on, with its cell count."""
    sources = evidence.get("sources")
    if not sources:
        return f"git {evidence.get('git_sha', 'n/a')}"
    return "; ".join(f"git {s['git_sha']}, {s['cells']} cells" for s in sources)
```

In `_greek_section`, change the provenance opening from
`f"Greek evidence: `quantark/intraday/evidence/gate_c_greeks.json` (git {evidence.get('git_sha', 'n/a')}). "`
to
`f"Greek evidence: `quantark/intraday/evidence/gate_c_greeks.json` ({_greek_sources(evidence)}). "`.
Then replace the table header and row lines with:

```python
              "| Product | Fixture | Route | Measure | Monitoring | Profile | Engine settings | Measure settings | "
              "Horizon window (s) | Spot offsets | Barriers |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        knobs = ", ".join(f"{k}={v}" for k, v in sorted((r.get("measure_settings") or {}).items())) or "—"
        fixtures = ", ".join(r.get("fixtures") or ()) or "—"
        lines.append(f"| {r['product']} | {fixtures} | {r['route']} | {r['measure']} | {r.get('monitoring', '?')} | "
                     f"{r.get('profile', '?')} | {_settings_label(r.get('settings') or {})} | {knobs} | "
                     f"{r['horizon_s']} – {r.get('horizon_max_s', '?')} | "
                     f"{', '.join(r.get('offsets', ()))} | {', '.join(r.get('barriers', ()))} |")
```

- [ ] **Step 8: Regenerate the matrix, then run the intraday suite**

```bash
PYTHONPATH=$WT $PY -m quantark.intraday.publish docs/execution/intraday-capability-matrix.md
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=$WT $PY tmp/rss_guard.py 24 tmp/near_ki_suite_guard.log -- \
  $PY -m pytest test/intraday -n 4 -m "not slow" -q -p no:cacheprovider
```

Expected: `wrote docs/execution/intraday-capability-matrix.md`, then all intraday tests pass. At `e0ba6ba7` that was 323, and this task adds 9. The packaged rows don't have `fixtures` yet, so the matrix shows `—` in the new column until Task 5.

- [ ] **Step 9: Commit (1)**

```bash
git add test/intraday/gate_c/test_daily_ki_fixture.py test/intraday/gate_c/test_daily_ki_harness.py test/intraday/test_publish_greek_rows.py
git commit -F - -- test/intraday/gate_c/cells.py test/intraday/gate_c/greek_harness.py \
  test/intraday/gate_c/test_daily_ki_fixture.py test/intraday/gate_c/test_daily_ki_harness.py \
  test/intraday/test_capability_evidence.py test/intraday/test_publish_greek_rows.py \
  quantark/intraday/publish.py docs/execution/intraday-capability-matrix.md <<'EOF'
test(intraday): daily-KI Gate C fixture; per-fixture Greek ladders and certificate labels

snowball_daily_ki is the monthly fixture with KI 75 at every SSE close, certified
on the 2026-09-10 close: 154 desk groups on a 1 s - 6 h ladder inside one gap
between closes, QUAD V2 only. demonstrated() labels every certificate with its
fixtures and the matrix renders them, with every evidence source. The evidence
test's backing-cell selection now matches economic identity and monitoring, so
a daily-KI miss can never fail the untouched monthly certificates.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
git log --oneline -1
```

Record the SHA. It's the revision the sweep runs on.

---

### Task 3: Runner and pilot gate

**Files:**
- Create (untracked): `tmp/near_ki/run_greeks.py`, `tmp/near_ki/pilot_report.py`

**Interfaces:**
- Consumes: `greek_harness.greek_groups()`, `GreekGroup.id`, and `test_greek_ladders.py::test_greek_group[<id>]` (which writes `gate_c_greeks.jsonl` under `QUANTARK_WRITE_GATE_C`)
- Produces: `tmp/near_ki/gate_c_greeks.jsonl` (one row per group), and `tmp/near_ki/wall.jsonl` with rows `{"groups": int, "pilot": bool, "git_sha": str, "start": float, "end": float, "exit": int}`

- [ ] **Step 1: Write the runner**

Create `tmp/near_ki/run_greeks.py`:

```python
"""Gate C daily-KI Greek sweep on the committed worktree source; resumable (recorded groups are skipped).

    python tmp/near_ki/run_greeks.py --pilot [--workers=4]    # the six pilot groups of the design
    python tmp/near_ki/run_greeks.py [--workers=4]            # every remaining snowball_daily_ki group

Rows append to tmp/near_ki/gate_c_greeks.jsonl; each invocation appends its revision and wall time to
tmp/near_ki/wall.jsonl. Launch under tmp/rss_guard.py.
"""
import json
import os
import subprocess
import sys
import time

WT = "/Users/fuxinyao/quant-ark/.claude/worktrees/intraday-plan1"
PY = "/Users/fuxinyao/quant-ark/.venv/bin/python"
OUT = os.path.join(WT, "tmp", "near_ki")
TARGET = "test/intraday/gate_c/test_greek_ladders.py"
PILOT = ("greek-snowball_daily_ki-desk-1s-bp-1-ki", "greek-snowball_daily_ki-desk-1s-eq-ki",
         "greek-snowball_daily_ki-desk-60s-sd-1-ki", "greek-snowball_daily_ki-desk-3600s-sd+0.5-ki",
         "greek-snowball_daily_ki-desk-21600s-sd-2-ki", "greek-snowball_daily_ki-desk-3600s-sd+1-ko")


def recorded():
    path = os.path.join(OUT, "gate_c_greeks.jsonl")
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as handle:
        cells = [json.loads(line)["cell"] for line in handle if line.strip()]
    return {f"greek-{c['product']}-{c['profile']}-{c['horizon']}s-{c['offset']}-{c['barrier']}" for c in cells}


def committed_revision():
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "quantark", "test/intraday"], cwd=WT,
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        sys.exit(f"refusing to sweep uncommitted source:\n{dirty}")
    return subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], cwd=WT, capture_output=True, text=True,
                          check=True).stdout.strip()


def main():
    os.makedirs(OUT, exist_ok=True)
    sha = committed_revision()
    sys.path[:0] = [WT, os.path.join(WT, "test")]
    from intraday.gate_c.greek_harness import greek_groups
    pilot = "--pilot" in sys.argv
    wanted = PILOT if pilot else tuple(g.id for g in greek_groups() if g.product == "snowball_daily_ki")
    assert set(PILOT) <= {g.id for g in greek_groups()}, "a pilot id is not a harness group"
    done = recorded()
    todo = [gid for gid in wanted if gid not in done]
    workers = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--workers=")), "4")
    env = dict(os.environ, PYTHONPATH=WT, QUANTARK_GATE_C_GREEKS="1", QUANTARK_WRITE_GATE_C=OUT,
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    args = [PY, "-m", "pytest", "-n", workers, "-q", "-p", "no:cacheprovider", "-m", "slow", "--durations=0"]
    args += [f"{TARGET}::test_greek_group[{gid}]" for gid in todo]
    started = time.time()
    code = subprocess.call(args, cwd=WT, env=env) if todo else 0
    with open(os.path.join(OUT, "wall.jsonl"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"groups": len(todo), "pilot": pilot, "git_sha": sha, "start": started,
                                 "end": time.time(), "exit": code}) + "\n")
    sys.exit(code)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the pilot classifier**

Create `tmp/near_ki/pilot_report.py`:

```python
"""Classify every recorded daily-KI measure by the design's no-go table; exit 0 only when every measure passes.

    python tmp/near_ki/pilot_report.py [tmp/near_ki/pilot.log]    # the log adds per-group wall times
"""
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent


def cause(m):
    reason = m.get("reason") or ""
    if m["status"] in ("passed", "undefined"):
        return "pass"
    if m["status"] == "inconclusive":
        return "reference uncertainty exceeds the budget"
    if m["status"] == "unqualified" and "ladder converges" in reason:
        return "QUAD ladder converges above budget"
    if m["status"] == "unqualified" and ("quadrature error" in reason or "stencil on the quadrature" in reason):
        return "QUAD above budget on a finite move or stencil"
    if m["status"] == "unqualified" and "bump ladder does not converge" in reason:
        return "point proxy bump ladder does not converge"
    if m["status"] == "failed" and "does not approach the reference" in reason:
        return "QUAD ladder does not approach the reference (KI-lobe candidate)"
    return f"other {m['status']}: {reason[:80]}"


rows = [json.loads(line) for line in (OUT / "gate_c_greeks.jsonl").read_text().splitlines() if line.strip()]
counts = Counter()
for row in sorted(rows, key=lambda r: r["cell"]["id"]):
    for m in row["measures"]:
        c = cause(m)
        counts[c] += 1
        if c != "pass":
            print(f"{row['cell']['id']:<58}{m['measure']:<20}{c}\n    route={m.get('route')} ref={m.get('ref')} "
                  f"unc={m.get('ref_uncertainty')} budget={m.get('budget')} ladder={m.get('ladder')}")
print(f"groups {len(rows)}; measures by cause: {dict(counts)}")
if len(sys.argv) > 1:
    walls = [float(s) for s in re.findall(r"^([\d.]+)s call .*test_greek_group\[", Path(sys.argv[1]).read_text(), re.M)]
    if walls:
        median = statistics.median(walls)
        print(f"per-group wall: median {median:.0f} s, max {max(walls):.0f} s; "
              f"154 groups at 4 workers ~ {154 * median / 4 / 3600:.1f} h (plus per-worker reference warm-up)")
sys.exit(0 if set(counts) == {"pass"} else 1)
```

- [ ] **Step 3: Run the pilot (background, guarded)**

```bash
cd $WT && PYTHONPATH=$WT:$WT/test nohup caffeinate -i -m -s $PY tmp/rss_guard.py 30 tmp/near_ki/pilot_guard.log -- \
  $PY tmp/near_ki/run_greeks.py --pilot --workers=4 > tmp/near_ki/pilot.log 2>&1
```

Run it with `run_in_background`. Expected wall time is about 20–40 minutes, most of it the reference warming up in each worker. Expected exit is 0 (`6 passed`). The test asserts only that statuses are in `ACCEPTED`, so a non-passing status still exits 0. A `failed` status exits 1.

- [ ] **Step 4: Classify**

Run: `$PY tmp/near_ki/pilot_report.py tmp/near_ki/pilot.log; echo "gate=$?"; sort -t= -k2 -n tmp/near_ki/pilot_guard.log | tail -1`

- [ ] **Step 5: GATE**

- `gate=0` (every measure `pass`): report the per-group wall time, the estimated full-sweep time and the peak memory to the user in one message. Then continue to Task 4.
- `gate=1`: **STOP.** Report the classifier's output (every non-pass measure with route, ref, unc, budget and ladder) and the matching remedy from the spec's no-go table. Wait for the user's decision. Don't run Task 4, don't change engine settings or reference points, and don't edit budgets.

---

### Task 4: Full sweep

**Files:** none tracked. Appends to `tmp/near_ki/gate_c_greeks.jsonl` and `tmp/near_ki/wall.jsonl`.

**Interfaces:**
- Consumes: Task 3's runner (the pilot rows count as recorded and are skipped)
- Produces: 154 unique `snowball_daily_ki` rows in `tmp/near_ki/gate_c_greeks.jsonl`

- [ ] **Step 1: Launch (background, guarded, unattended)**

```bash
cd $WT && PYTHONPATH=$WT:$WT/test nohup caffeinate -i -m -s $PY tmp/rss_guard.py 30 tmp/near_ki/sweep_guard.log -- \
  $PY tmp/near_ki/run_greeks.py --workers=4 > tmp/near_ki/sweep.log 2>&1
```

Run it with `run_in_background` and wait for the completion notification; don't poll. If the guard kills the run (exit 137) or the machine sleeps (`pmset -g log`), relaunch the same command. The runner skips recorded groups. If memory is the cause, relaunch with `--workers=3`.

- [ ] **Step 2: Verify completeness and summarize statuses**

```bash
$PY - <<'EOF'
import json, sys
sys.path[:0] = ["/Users/fuxinyao/quant-ark/.claude/worktrees/intraday-plan1", "/Users/fuxinyao/quant-ark/.claude/worktrees/intraday-plan1/test"]
from collections import Counter
from intraday.gate_c.greek_harness import greek_groups
rows = {json.loads(l)["cell"]["id"]: json.loads(l) for l in open("tmp/near_ki/gate_c_greeks.jsonl") if l.strip()}
expected = {g.cell("quad_v2").id for g in greek_groups() if g.product == "snowball_daily_ki"}
print("groups", len(set(rows) & expected), "/", len(expected), "unexpected", len(set(rows) - expected))
print(Counter(m["status"] for r in rows.values() for m in r["measures"]))
EOF
$PY tmp/near_ki/pilot_report.py tmp/near_ki/sweep.log; echo "all-pass=$?"
```

Expected: `groups 154 / 154 unexpected 0`.

- [ ] **Step 3: GATE**

- `all-pass=0`: continue to Task 5.
- `all-pass=1`: **STOP** and report the non-pass measures by cause. A miss narrows or splits the certificate windows, so some example ticks would print `unqualified`. The user decides whether to package as-is or remediate.

---

### Task 5: Package the evidence and regenerate the matrix (commit 2)

**Files:**
- Create (untracked): `tmp/near_ki/package_evidence.py`
- Modify: `quantark/intraday/evidence/gate_c_greeks.json`, `docs/execution/intraday-capability-matrix.md`

**Interfaces:**
- Consumes: Task 4's rows, `tmp/near_ki/wall.jsonl`, `greek_harness.demonstrated`, `greek_groups`
- Produces: packaged evidence with 2574 cells, top-level `sources`, `supplement` and `economic_identity_reconstruction` retained, and a `fixtures` label on every certificate

- [ ] **Step 1: Write the packaging script**

Create `tmp/near_ki/package_evidence.py`:

```python
"""Merge the daily-KI Greek sweep into the packaged Gate C evidence; refuse anything incomplete or disturbing.

    PYTHONPATH=$WT:$WT/test python tmp/near_ki/package_evidence.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WT), str(WT / "test")]
from intraday.gate_c.greek_harness import demonstrated, greek_groups  # noqa: E402

PACKAGE = WT / "quantark/intraday/evidence/gate_c_greeks.json"
OUT = WT / "tmp/near_ki"
FIXTURE = "snowball_daily_ki"


def canonical(row):
    """A certificate's pre-existing fields only: the new ``fixtures`` label is not a change to it."""
    return json.dumps({k: v for k, v in row.items() if k != "fixtures"}, sort_keys=True)


def main():
    payload = json.loads(PACKAGE.read_text())
    old_cells = {r["cell"]["id"]: r for r in payload["cells"]}
    old_certificates = sorted(canonical(r) for r in payload["demonstrated"])
    runs = [json.loads(line) for line in (OUT / "wall.jsonl").read_text().splitlines() if line.strip()]
    shas = {r["git_sha"] for r in runs if r["groups"]}
    if len(shas) != 1:
        raise SystemExit(f"the sweep ran on {sorted(shas)}; expected exactly one revision")
    sweep_sha = shas.pop()
    wall = round(sum(r["end"] - r["start"] for r in runs if r["groups"]))
    new = {}
    for line in (OUT / "gate_c_greeks.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            new[row["cell"]["id"]] = row
    expected = {g.cell("quad_v2").id for g in greek_groups() if g.product == FIXTURE}
    if len(expected) != 154 or set(new) != expected:
        raise SystemExit(f"incomplete daily-KI sweep: {len(set(new) & expected)} / {len(expected)} groups, "
                         f"{len(set(new) - expected)} unexpected")
    if set(new) & set(old_cells):
        raise SystemExit("daily-KI cell ids collide with packaged cells")
    if len({r["economic_identity"] for r in new.values()}) != 1:
        raise SystemExit("the daily-KI cells span more than one economic identity")
    for row in new.values():
        row["source_revision"] = sweep_sha
    cells = {**old_cells, **new}
    rows = [cells[k] for k in sorted(cells)]
    certificates = demonstrated(rows)
    retained = sorted(canonical(r) for r in certificates if FIXTURE not in r["fixtures"])
    if retained != old_certificates:
        raise SystemExit("a pre-existing certificate would change; refusing to write")
    added = [r for r in certificates if FIXTURE in r["fixtures"]]
    payload.update(
        cells=rows, demonstrated=certificates, horizons_s=sorted({r["cell"]["horizon"] for r in rows}),
        git_sha=sweep_sha, wall_time_s=wall,
        sources=[{"git_sha": payload["git_sha"], "cells": len(old_cells), "wall_time_s": payload["wall_time_s"],
                  "description": "the monthly, long-gap and digital fixtures, with the retained supplement"},
                 {"git_sha": sweep_sha, "cells": len(new), "wall_time_s": wall,
                  "description": "snowball_daily_ki: the 2026-09-10 close, desk profile, QUAD V2"}])
    PACKAGE.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    statuses = Counter(m["status"] for r in new.values() for m in r["measures"])
    print({"cells": len(rows), "certificates": len(certificates), "added": len(added), "retained": len(retained),
           "daily_statuses": dict(statuses), "sweep_sha": sweep_sha, "wall_s": wall})
    for r in sorted(added, key=lambda r: (r["measure"], r["horizon_s"])):
        print(f"  {r['measure']:<20} {r['horizon_s']} - {r['horizon_max_s']} s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Package**

Run: `PYTHONPATH=$WT:$WT/test $PY tmp/near_ki/package_evidence.py`
Expected: `cells 2574`, `retained 50`, and `added 12` with each window `1 - 21600 s` when Task 4 passed everywhere. Save the printed block; Task 6's baseline note quotes it. If the script exits with a refusal, **STOP** and report it. Don't bypass the check.

- [ ] **Step 3: Regenerate the matrix and run the intraday suite**

```bash
PYTHONPATH=$WT $PY -m quantark.intraday.publish docs/execution/intraday-capability-matrix.md
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=$WT $PY tmp/rss_guard.py 24 tmp/near_ki_suite_guard.log -- \
  $PY -m pytest test/intraday -n 4 -m "not slow" -q -p no:cacheprovider
```

Expected: all pass, including `test_greek_evidence_parses_and_every_demonstration_is_backed_by_passing_cells` over 62 certificates, and matrix freshness.

- [ ] **Step 4: Confirm a daily-KI point Greek is now published**

```bash
PYTHONPATH=$WT:$WT/test $PY - <<'EOF'
from datetime import timedelta
from intraday.gate_c import cells as C
from intraday.gate_c.harness import build_context
from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
from quantark.intraday import value_intraday
from dataclasses import replace
ctx, _ = build_context(C.Cell("snowball_daily_ki", "quad_v2", "desk", timedelta(hours=1), "sd+0.5", "ki"))
res = value_intraday(SnowballQuadEngineV2(), replace(ctx.request, greeks=("delta", "gamma"), greek_convention="point"))
print([(g.name, g.status, g.value) for g in res.greeks])
EOF
```

Expected: `delta` and `gamma` both `ok` with finite values.

- [ ] **Step 5: Commit (2)**

Fill in `<...>` from Step 2's printed block before committing:

```bash
git commit -F - -- quantark/intraday/evidence/gate_c_greeks.json docs/execution/intraday-capability-matrix.md <<'EOF'
evidence(intraday): daily-KI Greek certificate for the 2026-09-10 close

The Gate C Greek sweep of snowball_daily_ki (KI 75 at every SSE close, KO 103
monthly) on <sweep_sha>: 154 desk groups, 1 s - 6 h before the close, 11 spot
offsets around KI and KO, QUAD V2 order=8 cells_per_sd=2.0. Statuses:
<daily_statuses>. <wall_s> s at 4 workers under a 30 GiB guard.

<added> certificates added, windows <windows>. The 50 earlier certificates are
retained unchanged (packaged, not re-measured); packaging refuses any change to
them. Top-level sources list both measured revisions; supplement and
economic_identity_reconstruction are carried forward.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 6: The near-KI example, its test and the docs (commit 3)

**Files:**
- Create: `example/intraday_snowball_near_ki_demo.py`, `test/intraday/test_near_ki_example.py`
- Modify: `quantark/intraday/README.md` (Greeks → "What a status does and does not claim", after the "Swept today" bullet); `docs/superpowers/plans/2026-09-15-intraday-baseline.md` (after the "2026-09-17 follow-up" section)

**Interfaces:**
- Consumes: the packaged evidence from Task 5; the public `quantark.intraday` API
- Produces: `example.intraday_snowball_near_ki_demo` with `at(hour, minute, second=0) -> datetime`, `sse_sessions() -> TradingSessionCalendar`, `desk_profile() -> VarianceProfile`, `build_snowball(calendar) -> SnowballOption`, `close_history(product, sessions, profile) -> tuple[Fixing, ...]`, `request(product, sessions, profile, history, ts, spot, *, spot_ts=None, fixings=None, **options) -> IntradayValuationRequest`, `TICKS`, and `main(argv=None)`. The `--json` run record has top-level keys `contract, market, day, profile, evidence_git_sha, history_last_close, ticks, point_vs_desk, curves, close`.

- [ ] **Step 1: Write the failing test**

Create `test/intraday/test_near_ki_example.py`:

```python
"""The near-KI example streams the certified daily-KI contract: same economics as the Gate C fixture, and the
envelope decides which ticks publish Greeks."""
import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from example.intraday_snowball_near_ki_demo import (at, build_snowball, close_history, desk_profile,  # noqa: E402
                                                    request, sse_sessions)
from intraday.gate_c import cells as C  # noqa: E402
from intraday.gate_c.harness import build_context  # noqa: E402
from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2  # noqa: E402
from quantark.intraday import resolve_context, value_intraday  # noqa: E402
from quantark.intraday.capability import economic_identity  # noqa: E402


@pytest.fixture(scope="module")
def demo():
    sessions, profile = sse_sessions(), desk_profile()
    product = build_snowball(sessions.calendar)
    history = close_history(product, sessions, profile)
    return {"history": history,
            "build": lambda ts, spot, **kw: request(product, sessions, profile, history, ts, spot, **kw)}


def test_the_example_contract_has_the_certified_fixture_economics(demo):
    ctx = resolve_context(demo["build"](at(14, 0), 75.20))
    fixture, _ = build_context(C.Cell("snowball_daily_ki", "quad_v2", "desk", timedelta(hours=1), "eq", "ki"))
    assert economic_identity(ctx) == economic_identity(fixture)
    assert ctx.request.variance_profile.identity() == C.profile("desk").identity()


def test_confirmed_closes_stay_strictly_between_the_barriers(demo):
    history = demo["history"]
    assert len(history) == 122 and all(75.0 < f.value < 103.0 for f in history)
    assert history[-1].value == pytest.approx(75.9)


def test_point_delta_and_gamma_are_published_inside_the_envelope(demo):
    res = value_intraday(SnowballQuadEngineV2(), demo["build"](at(14, 0), 75.20, greeks=("delta", "gamma"),
                                                               greek_convention="point"))
    assert [res.greek(n).status for n in ("delta", "gamma")] == ["ok", "ok"]


def test_the_tick_below_the_ten_second_envelope_floor_publishes_no_greek(demo):
    res = value_intraday(SnowballQuadEngineV2(), demo["build"](at(14, 59, 50), 74.90, greeks=("delta", "gamma"),
                                                               greek_convention="point"))
    for name in ("delta", "gamma"):
        g = res.greek(name)
        assert g.status == "unqualified" and g.value is None and "spot envelope" in g.reason
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `PYTHONPATH=$WT $PY -m pytest -n 0 -q -p no:cacheprovider test/intraday/test_near_ki_example.py`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'example.intraday_snowball_near_ki_demo'`.

- [ ] **Step 3: Write the example**

Create `example/intraday_snowball_near_ki_demo.py`:

```python
"""
Intraday PV and Greeks of a daily-KI Snowball while the spot streams around its knock-in level.

The contract observes KI 75 at every SSE close and KO 103 monthly. On Thu 2026-09-10 the spot crosses 75 in the
morning and after lunch, then hovers within 0.1 of it before the 15:00 close, which is itself a KI observation.
This script demonstrates:
1. A tick stream: PV, point delta/gamma/vega/theta, and the distance to KI in remaining standard deviations
2. Point versus desk_bump Greeks at 14:00, 14:55 and 14:59:59
3. spot_curve snapshots at 10:00, 14:00, 14:55 and 14:59:59: the PV step and the delta spike forming at 75
4. The close: 15:00 "before" with the spot on the barrier, "after" with fixings either side of it, and a
   provisional row thirty seconds later with no fixing supplied

A numerical Greek is published only inside its Gate C certificate: this contract, this day, the desk profile, the
engine's certified settings and the swept spot envelope (quantark/intraday/evidence/gate_c_greeks.json).
Elsewhere it prints its status and reason, never a number.

Usage:
    python example/intraday_snowball_near_ki_demo.py [--json run.json]
"""

import argparse
import json
from datetime import datetime, time, timedelta, timezone
from math import exp, log, sin, sqrt

from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.intraday import (Fixing, TradingSession, TradingSessionCalendar, VarianceProfile, resolve_context,
                               spot_curve, value_intraday)
from quantark.intraday.capability import greek_evidence
from quantark.intraday.request import IntradayValuationRequest
from quantark.intraday.timestamp import SECONDS_PER_YEAR
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.enum.option_enums import ObservationType

SHANGHAI = timezone(timedelta(hours=8))
T0 = datetime(2026, 3, 16)
DAY = datetime(2026, 9, 10)
KO, KI, COUPON = 103.0, 75.0, 0.12
RATE, DIVIDEND, VOL = 0.03, 0.01, 0.20
STREAM_GREEKS = ("delta", "gamma", "vega", "theta")
#: (hour, minute, second, spot), hand-shaped. 14:59:50 is below the ten-second envelope floor 75 x 0.999 = 74.925.
TICKS = (
    (9, 30, 0, 75.90), (10, 30, 0, 74.85), (11, 30, 0, 75.10), (13, 0, 0, 75.30), (13, 30, 0, 74.90),
    (14, 0, 0, 75.20), (14, 30, 0, 74.80), (14, 45, 0, 75.15), (14, 50, 0, 74.90), (14, 55, 0, 75.05),
    (14, 58, 0, 74.95), (14, 59, 0, 75.02), (14, 59, 30, 74.97), (14, 59, 50, 74.90), (14, 59, 59, 75.01),
)
DESK_MOMENTS = ((14, 0, 0), (14, 55, 0), (14, 59, 59))
CURVE_MOMENTS = ((10, 0, 0), (14, 0, 0), (14, 55, 0), (14, 59, 59))
CURVE_SPOTS = tuple(round(74.5 + 0.1 * k, 1) for k in range(11))
CLOSE_FIXINGS = (74.98, 75.02)


def at(hour, minute, second=0):
    """A wall-clock instant on the streamed day, Shanghai time."""
    return datetime(DAY.year, DAY.month, DAY.day, hour, minute, second, tzinfo=SHANGHAI)


def sse_sessions():
    calendar = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    return TradingSessionCalendar(name="SSE", tz=SHANGHAI, calendar=calendar,
                                  sessions=(TradingSession(time(9, 30), time(11, 30)),
                                            TradingSession(time(13, 0), time(15, 0))))


def desk_profile():
    return VarianceProfile("desk", "1", days_per_year=244, overnight_weight=0.25,
                           session_weights=(0.35, 0.35), break_weights=(0.05,))


def monthly_trading_dates(calendar, start, months):
    dates = []
    for k in range(1, months + 1):
        month = (start.month - 1 + k) % 12 + 1
        year = start.year + (start.month - 1 + k) // 12
        d = datetime(year, month, min(start.day, 28))
        while not calendar.is_business_day(d):
            d += timedelta(days=1)
        dates.append(d)
    return dates


def build_snowball(calendar):
    """KO 103 on twelve monthly dates and KI 75 at every SSE close after the trade date: the certified terms."""
    ko_dates = monthly_trading_dates(calendar, T0, 12)
    maturity = ko_dates[-1]
    ki_dates = [d for d in (T0 + timedelta(days=k) for k in range(1, (maturity - T0).days + 1))
                if calendar.is_business_day(d)]
    return SnowballOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0, initial_date=T0, exercise_date=maturity,
        barrier_config=BarrierConfig(
            ko_barrier=KO, ko_rate=COUPON, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=KO)
                                                                 for d in ko_dates]),
            ki_barrier=KI, ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=KI)
                                                                 for d in ki_dates])),
        payoff_config=PayoffConfig(rebate_rate=COUPON, include_principal=False),
    )


def market(ts, spot, spot_ts=None):
    return PricingEnvironment(rate_curve=FlatRateCurve(RATE), valuation_date=ts,
                              spot_quote=SpotQuote(spot, timestamp=spot_ts if spot_ts is not None else ts),
                              vol_surface=FlatVolSurface(VOL), div_yield=ContinuousDividendYield(DIVIDEND))


def close_history(product, sessions, profile):
    """Confirmed closes before the streamed day's close: a deterministic slide from 100 to 75.9 inside (75, 103)."""
    probe = resolve_context(IntradayValuationRequest(
        product=product, pricing_env=market(datetime(2026, 3, 17, 9, 0, tzinfo=SHANGHAI), 100.0),
        session_calendar=sessions, variance_profile=profile))
    instants = sorted({e.timestamp for e in probe.timeline.events if e.timestamp < at(15, 0)})
    last = len(instants) - 1
    return tuple(Fixing(t, round(100.0 - 24.1 * k / last + 0.4 * sin(k / 2.0) * (1.0 - k / last), 4))
                 for k, t in enumerate(instants))


def request(product, sessions, profile, history, ts, spot, *, spot_ts=None, fixings=None, **options):
    return IntradayValuationRequest(product=product, pricing_env=market(ts, spot, spot_ts), session_calendar=sessions,
                                    variance_profile=profile, fixings=history if fixings is None else fixings,
                                    **options)


def spot_at(ts):
    """The latest streamed spot at or before ``ts``."""
    return [spot for hour, minute, second, spot in TICKS if at(hour, minute, second) <= ts][-1]


def ki_distance(ctx, spot):
    """(ln(S / KI) in standard deviations of the variance left to the close, the runtime envelope floor)."""
    tau = (at(15, 0) - ctx.valuation_timestamp).total_seconds() / SECONDS_PER_YEAR
    sd = sqrt(max(float(ctx.pricing_env.vol_surface.total_variance(100.0, tau, spot)), 0.0))
    return (log(spot / KI) / sd if sd > 0.0 else float("inf")), KI * min(exp(-2.0 * sd), 0.999)


def short(g):
    """A Greek's value, or a dash and a short cause when it has none."""
    if g.status == "ok":
        return f"{g.value:.4f}"
    return "- envelope" if "spot envelope" in (g.reason or "") else f"- {g.status}"


def greek_record(g):
    return {"name": g.name, "convention": g.convention, "status": g.status, "value": g.value, "unit": g.unit,
            "reason": g.reason}


def stream(engine, build):
    print("\n1. Tick stream: point Greeks, theta per minute")
    print(f"{'time':<9}{'spot':>7}{'KI sd':>8}{'floor':>9}{'PV':>11}" + "".join(f"{n:>13}" for n in STREAM_GREEKS))
    rows, reasons = [], {}
    for hour, minute, second, spot in TICKS:
        ts = at(hour, minute, second)
        req = build(ts, spot, greeks=STREAM_GREEKS, greek_convention="point", theta_unit="minute")
        distance, floor = ki_distance(resolve_context(req), spot)
        res = value_intraday(engine, req)
        print(f"{ts:%H:%M:%S}{spot:>7.2f}{distance:>8.2f}{floor:>9.3f}{res.price:>11.4f}"
              + "".join(f"{short(res.greek(n)):>13}" for n in STREAM_GREEKS))
        for g in res.greeks:
            if g.status != "ok":
                reasons.setdefault(g.reason, []).append(f"{ts:%H:%M:%S} {g.name}")
        rows.append({"time": ts.isoformat(), "spot": spot, "ki_distance_sd": distance, "envelope_floor": floor,
                     "price": res.price, "greeks": [greek_record(g) for g in res.greeks]})
    for reason, where in reasons.items():
        print(f"  not published ({', '.join(where)}): {reason}")
    return rows


def point_vs_desk(engine, build):
    print("\n2. Point derivative versus desk bump (desk theta: the certified one-hour roll, clamped to the close)")
    rows = []
    for hour, minute, second in DESK_MOMENTS:
        ts = at(hour, minute, second)
        spot = spot_at(ts)
        results = {c: value_intraday(engine, build(ts, spot, greeks=STREAM_GREEKS, greek_convention=c,
                                                   theta_unit="minute"))
                   for c in ("point", "desk_bump")}
        print(f"\n{ts:%H:%M:%S}  spot {spot:.2f}  PV {results['point'].price:.4f}")
        print(f"  {'greek':<7}{'point':>14}{'desk_bump':>14}  desk unit")
        for name in STREAM_GREEKS:
            desk = results["desk_bump"].greek(name)
            print(f"  {name:<7}{short(results['point'].greek(name)):>14}{short(desk):>14}  {desk.unit}")
        rows.append({"time": ts.isoformat(), "spot": spot, "price": results["point"].price,
                     **{c: [greek_record(g) for g in r.greeks] for c, r in results.items()}})
    return rows


def curves(engine, build):
    print("\n3. spot_curve snapshots: PV and point delta across the barrier ('-' = not published)")
    snapshots = []
    for hour, minute, second in CURVE_MOMENTS:
        ts = at(hour, minute, second)
        curve = spot_curve(engine, build(ts, spot_at(ts)), CURVE_SPOTS)
        snapshots.append({"time": ts.isoformat(), "points": [
            {"spot": p.spot, "price": p.price, "delta": p.delta, "gamma": p.gamma, "status": p.status,
             "reason": p.reason} for p in curve]})
    print(f"{'spot':>6}" + "".join(f"{t[11:19] + ' PV':>15}{'delta':>10}" for t in (s["time"] for s in snapshots)))
    for i, spot in enumerate(CURVE_SPOTS):
        cells = [(s["points"][i]["price"], s["points"][i]) for s in snapshots]
        print(f"{spot:>6.1f}" + "".join(f"{price:>15.4f}{(format(p['delta'], '.3f') if p['status'] == 'ok' else '-'):>10}"
                                         for price, p in cells))
    return snapshots


def close_rows(engine, build, history):
    print("\n4. The 15:00 close, a KI observation")
    close = at(15, 0)
    cases = [("15:00:00 before, spot 75.00 on the barrier",
              build(close, KI, event_phase="before", greeks=("delta", "gamma"), greek_convention="point"))]
    cases += [(f"15:00:00 after, fixing {f:.2f}", build(close, f, event_phase="after", fixings=history + (Fixing(close, f),)))
              for f in CLOSE_FIXINGS]
    cases += [("15:00:30 after, no fixing, close quote 74.98",
               build(close + timedelta(seconds=30), 74.98, spot_ts=close, event_phase="after"))]
    rows = []
    for label, req in cases:
        res = value_intraday(engine, req)
        state = "knocked in" if res.lifecycle["knocked_in"] else "not knocked in"
        assumed = [{"scheduled_at": a.scheduled_at.isoformat(), "assumed_value": a.assumed_value} for a in res.assumptions]
        depends = sorted({d for cf in res.cashflows if cf.provenance == "provisional" for d in cf.depends_on})
        print(f"{label:<45} PV {res.price:>9.4f}  {state:<15} provisional={res.provisional}"
              + (f"  assumed {', '.join(str(a['assumed_value']) for a in assumed)}" if assumed else ""))
        for g in res.greeks:
            print(f"{'':<47}{g.name}: {short(g)}  {g.reason or ''}")
        if depends:
            print(f"{'':<47}provisional flows depend on {', '.join(depends)}")
        rows.append({"label": label, "time": req.pricing_env.valuation_date.isoformat(), "spot": req.pricing_env.spot,
                     "phase": res.phase.value, "price": res.price, "contingent_pv": res.contingent_pv,
                     "paid_cash": res.paid_cash, "provisional": res.provisional,
                     "knocked_in": bool(res.lifecycle["knocked_in"]), "assumptions": assumed, "depends_on": depends,
                     "greeks": [greek_record(g) for g in res.greeks]})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Intraday PV and Greeks of a daily-KI snowball near its knock-in level")
    parser.add_argument("--json", dest="json_path", help="write the run record to this path")
    args = parser.parse_args(argv)
    sessions, profile = sse_sessions(), desk_profile()
    product = build_snowball(sessions.calendar)
    history = close_history(product, sessions, profile)
    engine = SnowballQuadEngineV2()

    def build(ts, spot, **options):
        return request(product, sessions, profile, history, ts, spot, **options)

    evidence = greek_evidence()
    print(f"Daily-KI snowball: KI {KI:g} at every SSE close, KO {KO:g} monthly, {COUPON:.0%} coupon; "
          f"{len(history)} confirmed closes, the last {history[-1].value:.2f}. Greek evidence git {evidence.get('git_sha')}.")
    record = {
        "contract": {"initial_date": T0.date().isoformat(), "exercise_date": product.exercise_date.date().isoformat(),
                     "ko": KO, "ki": KI, "coupon": COUPON,
                     "ki_observations": len(product.barrier_config.ki_observation_schedule.records)},
        "market": {"rate": RATE, "dividend": DIVIDEND, "vol": VOL},
        "day": DAY.date().isoformat(), "profile": list(profile.identity()),
        "evidence_git_sha": evidence.get("git_sha"), "history_last_close": history[-1].value,
    }
    record["ticks"] = stream(engine, build)
    record["point_vs_desk"] = point_vs_desk(engine, build)
    record["curves"] = curves(engine, build)
    record["close"] = close_rows(engine, build, history)
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=1, default=str)
        print(f"\nwrote {args.json_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `PYTHONPATH=$WT $PY -m pytest -n 0 -q -p no:cacheprovider test/intraday/test_near_ki_example.py`
Expected: 4 PASS. If `test_the_example_contract_has_the_certified_fixture_economics` fails, diff the two contexts' terms and events and fix the **example**. The fixture is what the evidence measured.

- [ ] **Step 5: Document the certificate**

In `quantark/intraday/README.md`, insert after the bullet that begins `- Swept today (see the packaged evidence and generated matrix`:

```markdown
- Daily-KI snowball (KI 75 at every SSE close, KO 103 monthly; the Gate C fixture `snowball_daily_ki`): QUAD V2
  point and desk measures under the desk profile, 1 s to 6 h before the 2026-09-10 close. Its economic identity
  holds that day's remaining events, so every other day of the same contract reports `unqualified`.
  `example/intraday_snowball_near_ki_demo.py` streams that day with the spot crossing 75, prints where the spot
  envelope stops publishing, and writes a JSON run record (`--json`).
```

If Task 5 printed windows other than `1 - 21600 s` for any measure, replace `1 s to 6 h` with the printed windows per measure.

In `docs/superpowers/plans/2026-09-15-intraday-baseline.md`, add after the `## 2026-09-17 follow-up` section. Fill in `<...>` from Task 5 Step 2's printed block and `sort -t= -k2 -n tmp/near_ki/sweep_guard.log | tail -1`:

```markdown
## 2026-09-17 daily-KI Greek certificate

Design `docs/superpowers/specs/2026-09-17-intraday-daily-ki-certification-design.md`. `snowball_daily_ki` on the
2026-09-10 close, desk profile, QUAD V2 `order=8, cells_per_sd=2.0`, swept on <sweep_sha>: 154 groups, statuses
<daily_statuses>, <wall_s> s at 4 workers under a 30 GiB guard, peak <peak> GiB. <added> certificates, windows
<windows>. The 50 earlier certificates are retained unchanged (packaged, not re-measured).
```

- [ ] **Step 6: Run the intraday suite**

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=$WT $PY tmp/rss_guard.py 24 tmp/near_ki_suite_guard.log -- \
  $PY -m pytest test/intraday -n 4 -m "not slow" -q -p no:cacheprovider
```

Expected: all pass.

- [ ] **Step 7: Commit (3)**

```bash
git add example/intraday_snowball_near_ki_demo.py test/intraday/test_near_ki_example.py
git commit -F - -- example/intraday_snowball_near_ki_demo.py test/intraday/test_near_ki_example.py \
  quantark/intraday/README.md docs/superpowers/plans/2026-09-15-intraday-baseline.md <<'EOF'
docs(intraday): near-KI example on the certified daily-KI snowball

example/intraday_snowball_near_ki_demo.py streams 2026-09-10 with the spot
crossing KI 75: point Greeks per tick with the KI distance in remaining
standard deviations, point vs desk at 14:00/14:55/14:59:59, spot_curve
snapshots, and the 15:00 close (before on the barrier, after either side,
provisional). --json writes the run record. Its test pins the example's
economic identity to the Gate C fixture's and the envelope's one refusal.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 7: Run the example and publish the report artifact

**Files:**
- Create (untracked): `tmp/near_ki_report/run.json`, `tmp/near_ki_report/run.txt`, `tmp/near_ki_report/build_report.py`, `tmp/near_ki_report/report.html`

**Interfaces:**
- Consumes: the example's `--json` record (keys listed in Task 6) and `quantark/intraday/evidence/gate_c_greeks.json` (`sources`, `demonstrated` rows whose `fixtures` include `snowball_daily_ki`, and the daily cells' measure statuses)
- Produces: a private artifact URL, reported to the user

- [ ] **Step 1: Run the example**

```bash
mkdir -p tmp/near_ki_report && OPENBLAS_NUM_THREADS=1 PYTHONPATH=$WT /usr/bin/time -l $PY \
  example/intraday_snowball_near_ki_demo.py --json tmp/near_ki_report/run.json > tmp/near_ki_report/run.txt 2>&1; tail -5 tmp/near_ki_report/run.txt
```

Expected: about 2–3 min, ending with `wrote tmp/near_ki_report/run.json`.

- [ ] **Step 2: Check the run against the design's expectations**

```bash
$PY - <<'EOF'
import json
r = json.load(open("tmp/near_ki_report/run.json"))
bad = [(t["time"][11:19], g["name"], g["status"]) for t in r["ticks"] for g in t["greeks"]
       if g["status"] != "ok" and t["time"][11:19] != "14:59:50"]
refusal = [g["status"] for t in r["ticks"] if t["time"][11:19] == "14:59:50" for g in t["greeks"]]
close = {row["label"]: row for row in r["close"]}
print("unexpected non-ok ticks:", bad)
print("14:59:50 statuses:", refusal)
print("close:", [(k, round(v["price"], 4), v["knocked_in"], v["provisional"], [g["status"] for g in v["greeks"]]) for k, v in close.items()])
EOF
```

Expected: `unexpected non-ok ticks: []`; the 14:59:50 statuses all `unqualified`; the close rows show `before` with delta/gamma `undefined`, `after` 74.98 knocked in, `after` 75.02 not knocked in, and the 15:00:30 row knocked in with `provisional` True. Any other outcome: **STOP** and report it before building the page. The report must describe what the run did, not what the design expected.

- [ ] **Step 3: Load the page contract**

Call the Artifact tool with `action: "quickstart"` and `intent: "other"` (a report page with charts), then load the `dataviz` skill. The page's tokens, dark mode, fonts, allowed script CDNs and layout rules come from those results. Follow them in Step 4.

- [ ] **Step 4: Write the generator**

Create `tmp/near_ki_report/build_report.py`. It reads `run.json` and the packaged evidence and computes the summary below in Python. It then writes `report.html` with the run record embedded as `<script type="application/json" id="run">` and the evidence summary as `<script type="application/json" id="evidence">`. Every chart and table is drawn from those two blocks; no number is typed into the HTML.

```python
"""Build tmp/near_ki_report/report.html from the example's run record and the packaged Gate C evidence."""
import json
from collections import Counter
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
run = json.loads((HERE / "run.json").read_text())
evidence = json.loads((WT / "quantark/intraday/evidence/gate_c_greeks.json").read_text())
daily_rows = [r for r in evidence["demonstrated"] if "snowball_daily_ki" in (r.get("fixtures") or ())]
daily_cells = [c for c in evidence["cells"] if c["cell"]["product"] == "snowball_daily_ki"]
summary = {
    "sources": evidence.get("sources", []),
    "certificates_total": len(evidence["demonstrated"]),
    "certificates_daily": [{"measure": r["measure"], "horizon_s": r["horizon_s"], "horizon_max_s": r["horizon_max_s"],
                            "settings": r["settings"].get("params", {})} for r in daily_rows],
    "groups": len(daily_cells),
    "measure_statuses": dict(Counter(m["status"] for c in daily_cells for m in c["measures"])),
    "retained": len(evidence["demonstrated"]) - len(daily_rows),
}


def embedded(element_id, payload):
    # JSON inside <script> must not close the tag early (built outside the f-string: Python 3.11 rejects a
    # backslash inside an f-string expression)
    body = json.dumps(payload).replace("</", "<\\/")
    return f'<script type="application/json" id="{element_id}">{body}</script>'


PAGE = (HERE / "page_template.html").read_text()
(HERE / "report.html").write_text(PAGE.replace("<!--RUN-->", embedded("run", run))
                                  .replace("<!--EVIDENCE-->", embedded("evidence", summary)))
print("wrote", HERE / "report.html", {k: v for k, v in summary.items() if k != "certificates_daily"})
```

Then write `tmp/near_ki_report/page_template.html` following Step 3's contract, with the markers `<!--RUN-->` and `<!--EVIDENCE-->` placed before the page script. Its sections and encodings:

1. **Header:** the title "Snowball near knock-in, intraday". A one-paragraph summary of the contract, day and engine, filled from `run.contract`, `run.market` and `run.day`.
2. **The day** (four charts sharing an x axis of session time with the lunch break compressed):
   - (a) spot as a line and markers, KI = 75 as a reference line, and the area below `envelope_floor` shaded as "Greeks not published";
   - (b) PV;
   - (c) point delta;
   - (d) point gamma.
   A tick whose Greek status is not `ok` is a **gap** in (c)/(d) with an annotated marker naming the status. Never interpolate across it. A theta chart goes under (d) with the same gap rule.
3. **Crossing the barrier:** PV vs spot and delta vs spot, with the four `run.curves` snapshots overlaid, one colour per time. Points with `status != "ok"` are omitted from the delta chart and listed underneath with their reason.
4. **Point vs desk:** a table per `run.point_vs_desk` moment (Greek, point, desk_bump, unit), plus two sentences on why a 1% desk bump across the barrier (0.75) differs from the derivative at the query spot.
5. **The close:** a table of `run.close` (label, PV, knocked in, provisional, assumed value, `depends_on`, Greek statuses), and the PV difference between the 74.98 and 75.02 `after` rows computed in the page script.
6. **What the Greeks rest on:** from `evidence`. Groups and measure statuses; every daily certificate's measure and window (seconds formatted as `1 s – 6 h`); engine settings; each source's git SHA and cell count; and the sentence "The {retained} earlier certificates were retained unchanged (packaged, not re-measured)."

- [ ] **Step 5: Build and check the page locally**

Run: `$PY tmp/near_ki_report/build_report.py && open tmp/near_ki_report/report.html`
Check it at phone width (the `resize_page` browser tool at 390 px) and in dark mode. Charts must render from the embedded data, with no console errors (`list_console_messages`), and gaps must show at 14:59:50.

- [ ] **Step 6: Publish**

Call the Artifact tool: `action: "publish"`, `file_path: tmp/near_ki_report/report.html`, `favicon: "❄️"`, `description: "How a daily-KI snowball's PV and certified Greeks move intraday as the spot streams around its knock-in level."`. Report the returned URL to the user.

---

## Execution notes

- Order: Tasks 1–2 (commit 1) → Task 3 pilot **gate** → Task 4 sweep **gate** → Task 5 (commit 2) → Task 6 (commit 3) → Task 7. Tasks 3–4 run in the background for hours, so don't start Task 6 before Task 5 packages the evidence (its test needs it).
- Keep other heavy jobs off the machine while the pilot and sweep run, so the wall-time estimates hold.
