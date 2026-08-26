# Snowball Re-baseline Gates (G1 / G4 / G2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run the snowball vol-model study's admission gates on the 0.4.0 engines with the §7A.4 fixes in place, and re-scope Gate G2 from a 2-variant PDE-vs-MC convergence check into a 6-variant engine-admission gate that tests delta as well as PV.

**Architecture:** Four phases against the existing `example/mo_volmodels/` stage scripts. Phase 0 (Tasks 0, 0B) freezes the inputs — a live daily scheduler now extends the surface history, so the cohort is pinned at `COHORT_ASOF` before anything measures against it, and the pipeline's 720 already-computed calibrations are seeded into the fleet cache. Phase A (Tasks 1–2) re-establishes the study's inputs — surface admission is *verified* against the pinned cohort (never rebuilt), and the fair coupon is re-solved because 0.4.0 repriced the 1D PDE. Phase B (Tasks 3–7) rewrites Gate G2's scope inside `11_pde_convergence_gate.py`: a production-vs-reference pair table replacing the hard-coded 2-variant assumption, maturity-bucketed bias detection, delta admission expressed in IM contracts, and Feller-regime conditioning. Phase C (Task 8) runs the gate and emits a fresh `gate_decision.json` that stage 12 consumes.

**Amended 2026-08-01** for a concurrently-landed daily calibration pipeline (spec §7A.12). Three changes carry real risk if skipped: the cohort pin (Task 0), the corrected G1 admission source (Task 1 — the original draft read a verdict key that does not exist in the artifacts), and the `--data-end` pin on G4 (Task 2).

**Tech Stack:** Python 3.11, numpy, pytest (`-n auto` by default; use `-n0` for serial debugging), `quantark.*` canonical imports, the project venv at `.venv/`.

## Global Constraints

- Source spec: `docs/superpowers/specs/2026-07-30-snowball-volmodel-backtest-040-rebaseline-design.md`. Every section reference below (§5, §7A.10, §9, §11) points there.
- Run everything with `.venv/bin/python` or after `source .venv/bin/activate`.
- Canonical imports only: `quantark.*`. Never `backtest.otc.*` (0.5.0 shim) or bare flat names.
- Numerics: use `quantark.util.numerical` (`is_zero`, `is_close`, `safe_divide`, …). Never a raw float `==` or a hardcoded tolerance.
- Fail closed. No silent fallbacks, no default-on approximations, no repaired data. A gate that cannot evaluate a cell records an error and counts that cell as failed.
- Gate tolerance floor: `TOL_ABS_PCT = 0.25` (% of notional); per-cell tolerance is `max(2 * mc_se_pct, 0.25)`.
- Study term sheet: `NOTIONAL = 50_000_000.0`, `FUTURES_MULTIPLIER = 200.0`, 3Y maturity, `KO_PCT = 1.03`, `KI_PCT = 0.75`, 3-month lockout.
- The gate prices a **1-unit** product (`contract_multiplier = 1.0`, `notional = s0`). The backtest prices `contract_multiplier = NOTIONAL / s0`. Any tolerance derived from hedge instrument size must convert between the two — see Task 6.
- **`COHORT_ASOF = "20260731"`.** A live launchd job (`com.quantark.mo-daily-calibration`, 18:30 + 20:30 Asia/Shanghai, Mon–Fri) extends the surface history every weekday. Every gate filters to `date <= COHORT_ASOF` and passes `data_end = COHORT_ASOF` explicitly — never "the last row of the spot CSV". Task 0 makes this concrete; §7A.12 explains why.
- The **inception fleet is 27 only while `data_end < 2026-08-01`.** `schedule_inceptions` admits a monthly start when `inception + 12 months <= data_end`; 2025-08-01 needs `data_end >= 2026-08-01`. Measured: 27 at both 2026-07-24 and 2026-07-31, **28 at 2026-08-03**. Any task that hardcodes 27 must instead assert against `len(schedule_inceptions(..., data_end=COHORT_ASOF))`.
- Do **not** regenerate `example/mo_volmodels/data/history/` as part of this plan — not because a rebuild is unrecoverable (the builder now preserves top-level `study_admission`, derives the manifest window from all records, and `exclude_thin_surfaces.py` is idempotent), but because it would move the pinned cohort mid-flight. If a rebuild ever happens, re-run `exclude_thin_surfaces.py` to restore the per-record exclusions, which the builder *does* overwrite.
- Surface admission verdicts live in `surface_manifest.json`, **not** in the artifacts. An artifact's `admission` block records the criteria used (`min_expiries: 2`, `sabr_beta`, …) and carries no per-surface verdict. Never infer admission from an artifact.
- Commit after every task. Branch: `fix/snowball-rebaseline-7a4-engine-fixes` (or a descendant).
- This branch's working tree carries an unrelated workstream's **uncommitted** changes (`quantark/backtest/replay/config.py`, `quantark/volmodels/calibration.py`, `quantark/volmodels/heston/calibration.py`, stages 14/15). Stage them deliberately or leave them alone — never `git add -A`.
- `docs/` is in `.gitignore` but tracked on `main`. Committing a docs file needs `git add -f`; a plain `git add` fails with "paths are ignored".
- **Do not work in a git worktree for this plan.** It depends on untracked data that exists only in this checkout: `example/mo_volmodels/data/history/` (766 surfaces, 768 artifacts) and `output/mo_daily_calibration/calibration_cache/` (720 entries). A fresh worktree has none of it, and Task 0's real-history test fails immediately.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `example/mo_volmodels/cohort.py` | The frozen cohort: `COHORT_ASOF` + manifest-driven admitted-date list | **create** |
| `test/mo_volmodels/test_cohort.py` | Cohort pin unit tests | **create** |
| `example/mo_volmodels/seed_calibration_cache.py` | Copy the daily pipeline's reusable cache entries into a fleet cache dir | **create** |
| `example/mo_volmodels/13_gate_g1_surface_admission.py` | G1: verify the manifest's admitted set against the artifacts on disk | **create** |
| `example/mo_volmodels/12_snowball_volmodel_backtest.py` | G4 coupon solve; adds `flat_bsm_quad` variant; consumes 6-variant routing | modify |
| `example/mo_volmodels/11_pde_convergence_gate.py` | G2: variant pair table, bucketed bias, delta admission, Feller conditioning | modify |
| `test/mo_volmodels/test_gate_g1_admission.py` | G1 verifier unit tests | **create** |
| `test/mo_volmodels/test_gate_scope.py` | G2 re-scope unit tests (pure functions only) | **create** |

`11_pde_convergence_gate.py` is already ~1600 lines. Do not restructure it wholesale — this plan adds pure functions near their existing neighbours and rewires `process_date`. If a task's diff exceeds ~150 lines in that file, stop and flag it rather than improvising a split.

---

## Task 0: Freeze the cohort

**Files:**
- Create: `example/mo_volmodels/cohort.py`
- Create: `test/mo_volmodels/test_cohort.py`
- Modify: `example/mo_volmodels/12_snowball_volmodel_backtest.py` (add `--data-end`; `:1243` argparse block, `:1300` derivation)

**Interfaces:**
- Consumes: `example/mo_volmodels/data/history/surface_manifest.json` (read-only)
- Produces: `COHORT_ASOF: date`; `admitted_dates(history_dir: Path | None = None) -> list[date]`; `excluded_records(history_dir: Path | None = None) -> list[dict]` — each `{date, reason, n_expiries}`, sorted by date.

**Why this exists:** the history is under a live scheduler (Global Constraints). Without a pin, G1 counts a different number of surfaces on Monday than on Friday, and — the sharp edge — `data_end` crossing 2026-08-01 admits a 28th inception, which would silently re-open G4 and shift §8's concentration statistics. Everything downstream reads its date list from here.

`cohort.py` deliberately imports nothing from the numbered stages, so stages 11, 12 and 13 can all import it without a cycle.

- [ ] **Step 1: Write the failing test**

```python
# test/mo_volmodels/test_cohort.py
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE = PROJECT_ROOT / "example/mo_volmodels/cohort.py"


def _load():
    spec = importlib.util.spec_from_file_location("mo_cohort", MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mo_cohort"] = module
    spec.loader.exec_module(module)
    return module


def _write_manifest(tmp_path: Path, records, study_admission=None) -> Path:
    history = tmp_path / "history"
    history.mkdir()
    payload = {"records": records}
    if study_admission is not None:
        payload["study_admission"] = study_admission
    (history / "surface_manifest.json").write_text(json.dumps(payload))
    return history


def test_asof_is_the_frozen_pin():
    assert _load().COHORT_ASOF == date(2026, 7, 31)


def test_admitted_dates_drop_records_after_the_asof(tmp_path):
    mod = _load()
    history = _write_manifest(
        tmp_path,
        [
            {"date": "20260730", "status": "ok"},
            {"date": "20260731", "status": "ok"},
            {"date": "20260803", "status": "ok"},   # next scheduler tick
        ],
    )
    assert mod.admitted_dates(history) == [date(2026, 7, 30), date(2026, 7, 31)]


def test_admitted_dates_drop_excluded_records(tmp_path):
    mod = _load()
    history = _write_manifest(
        tmp_path,
        [
            {"date": "20240930", "status": "excluded",
             "reason": "insufficient_expiries_for_dupire", "n_expiries": 2},
            {"date": "20241008", "status": "ok"},
        ],
    )
    assert mod.admitted_dates(history) == [date(2024, 10, 8)]


def test_study_admission_exclusions_are_enforced_even_if_status_says_ok(tmp_path):
    """A rebuild resets per-record status but preserves study_admission."""
    mod = _load()
    history = _write_manifest(
        tmp_path,
        [
            {"date": "20240930", "status": "ok", "n_expiries": 2},
            {"date": "20241008", "status": "ok"},
        ],
        study_admission={
            "vol_model_backtest": {"excluded_dates": ["20240930"], "min_expiries": 3}
        },
    )
    assert mod.admitted_dates(history) == [date(2024, 10, 8)]


def test_excluded_records_are_reported_with_reasons(tmp_path):
    mod = _load()
    history = _write_manifest(
        tmp_path,
        [
            {"date": "20240930", "status": "excluded",
             "reason": "insufficient_expiries_for_dupire", "n_expiries": 2},
            {"date": "20241008", "status": "ok"},
        ],
    )
    assert mod.excluded_records(history) == [
        {"date": date(2024, 9, 30),
         "reason": "insufficient_expiries_for_dupire", "n_expiries": 2}
    ]


def test_real_history_matches_the_pinned_counts():
    """Regression pin: the numbers §7A.12 froze."""
    mod = _load()
    admitted = mod.admitted_dates()
    assert len(admitted) == 766
    assert admitted[0] == date(2023, 5, 4)
    assert admitted[-1] == date(2026, 7, 31)
    assert date(2024, 9, 30) not in admitted
    assert date(2025, 4, 8) not in admitted
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_cohort.py -n0 -q`
Expected: FAIL — `spec_from_file_location` returns a spec whose loader raises `FileNotFoundError` for the missing `cohort.py`.

- [ ] **Step 3: Write the module**

```python
# example/mo_volmodels/cohort.py
"""The frozen surface cohort for the 0.4.0 re-baseline gates.

A launchd job (``com.quantark.mo-daily-calibration``) extends
``data/history/`` every weekday, so "the cohort" is a moving target unless it
is pinned.  Crossing 2026-08-01 also admits a 28th snowball inception, which
would re-open Gate G4.  Every gate reads its date list from here.

Admission verdicts come from ``surface_manifest.json``.  The artifacts
themselves carry only the *criteria* used to build them (``min_expiries: 2``),
never a per-surface verdict, so an artifact can never answer "was this
admitted?".
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_DIR = PROJECT_ROOT / "example/mo_volmodels/data/history"

# Frozen 2026-08-01 (spec §7A.12).  Raising this is a deliberate re-baseline:
# it changes the G1 count, and past 2026-08-01 it changes the inception fleet
# from 27 to 28 and therefore re-opens G4.
COHORT_ASOF = date(2026, 7, 31)

_STUDY_KEY = "vol_model_backtest"


def _parse(tag: str) -> date:
    return date(int(tag[:4]), int(tag[4:6]), int(tag[6:8]))


def _manifest(history_dir: Optional[Path]) -> Dict[str, Any]:
    path = Path(history_dir or DEFAULT_HISTORY_DIR) / "surface_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _study_exclusions(manifest: Dict[str, Any]) -> set:
    study = (manifest.get("study_admission") or {}).get(_STUDY_KEY) or {}
    return {_parse(tag) for tag in study.get("excluded_dates", [])}


def admitted_dates(history_dir: Optional[Path] = None) -> List[date]:
    """Admitted surface dates at or before ``COHORT_ASOF``, ascending.

    A date is admitted when the manifest record says ``status == "ok"`` AND
    the date is not in ``study_admission.vol_model_backtest.excluded_dates``.
    The second check matters because a history rebuild rewrites per-record
    status from the builder's own ``min_expiries=2`` while preserving the
    top-level study policy — so status alone would re-admit the thin surfaces.
    """
    manifest = _manifest(history_dir)
    excluded = _study_exclusions(manifest)
    out = []
    for record in manifest.get("records", []):
        day = _parse(str(record["date"]))
        if day > COHORT_ASOF or day in excluded:
            continue
        if record.get("status") != "ok":
            continue
        out.append(day)
    return sorted(out)


def excluded_records(history_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Non-admitted records at or before ``COHORT_ASOF``, with their reasons."""
    manifest = _manifest(history_dir)
    excluded = _study_exclusions(manifest)
    out = []
    for record in manifest.get("records", []):
        day = _parse(str(record["date"]))
        if day > COHORT_ASOF:
            continue
        if record.get("status") == "ok" and day not in excluded:
            continue
        out.append(
            {
                "date": day,
                "reason": record.get("reason"),
                "n_expiries": record.get("n_expiries"),
            }
        )
    return sorted(out, key=lambda item: item["date"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_cohort.py -n0 -q`
Expected: PASS. If `test_real_history_matches_the_pinned_counts` reports a count other than 766, **stop** — the scheduler has moved the history past the pin and the whole plan's cell counts need re-deriving before anything else runs.

- [ ] **Step 5: Confirm the inception fleet is still 27 at the pin**

```bash
.venv/bin/python - <<'PY'
import importlib.util, sys
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/fuxinyao/quant-ark")
spec = importlib.util.spec_from_file_location(
    "s12", ROOT / "example/mo_volmodels/12_snowball_volmodel_backtest.py"
)
s12 = importlib.util.module_from_spec(spec); sys.modules["s12"] = s12
spec.loader.exec_module(s12)

cohort_spec = importlib.util.spec_from_file_location(
    "mo_cohort", ROOT / "example/mo_volmodels/cohort.py"
)
cohort = importlib.util.module_from_spec(cohort_spec)
cohort_spec.loader.exec_module(cohort)

history = ROOT / "example/mo_volmodels/data/history"
spot = pd.read_csv(history / "csi1000_spot.csv")
admitted = cohort.admitted_dates()
inceptions = s12.schedule_inceptions(
    calendar=s12.stage11().TradingCalendar.from_spot_csv(history / "csi1000_spot.csv"),
    data_start=pd.Timestamp(spot["date"].iloc[0]).date(),
    data_end=cohort.COHORT_ASOF,
    first_admitted_surface=admitted[0],
)
print(f"admitted={len(admitted)} inceptions={len(inceptions)} last={inceptions[-1]}")
assert len(inceptions) == 27, f"fleet moved to {len(inceptions)} — re-derive the plan"
print("OK")
PY
```

Expected: `admitted=766 inceptions=27 last=2025-07-01` then `OK`.

- [ ] **Step 6: Give stage 12 a `--data-end` so the pin is reachable from the CLI**

`run_fleet` currently derives `data_end` from the last row of the spot CSV (`12_snowball_volmodel_backtest.py:1300`), which is exactly the value the scheduler moves. Tasks 2 and 8 both need to override it. Add the flag next to `--min-observable-months`:

```python
    parser.add_argument(
        "--data-end",
        default=None,
        help=(
            "ISO date pinning the replay window end (default: last spot row). "
            "The daily calibration pipeline extends the spot cache every "
            "weekday, and data_end crossing 2026-08-01 admits a 28th "
            "inception, so the gates pin this explicitly."
        ),
    )
```

and replace the derivation in `run_fleet`:

```python
    data_end = pd.Timestamp(spot["date"].iloc[-1]).date()
    if args.data_end:
        pinned = date.fromisoformat(str(args.data_end))
        if pinned > data_end:
            raise ValidationError(
                f"--data-end {pinned} is beyond the spot cache ({data_end})"
            )
        data_end = pinned
```

Fail closed on a pin beyond the data: silently truncating to the cache would make a typo look like a successful pinned run.

**`run_fleet` is not the only derivation.** `prepare_inceptions` (`:981`) independently recomputes `data_end = pd.Timestamp(spot["date"].iloc[-1]).date()` and hands *that* to `schedule_inceptions`. `run_fleet`'s pinned value only reaches `build_tasks`, which uses it for per-trade window censoring. Patching `run_fleet` alone therefore pins the *windows* and leaves the *fleet size* floating — exactly the failure this task exists to prevent, and invisible today because the pin and the cache end on the same date. Thread the pin through:

```python
def prepare_inceptions(
    *,
    history: VolSurfaceHistory,
    spot: pd.DataFrame,
    futures: pd.DataFrame,
    calendar,
    rate: float,
    notional: float,
    min_observable_months: int,
    data_end: Optional[date] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
```

and inside it:

```python
    data_start = pd.Timestamp(spot["date"].iloc[0]).date()
    if data_end is None:
        data_end = pd.Timestamp(spot["date"].iloc[-1]).date()
```

then pass `data_end=data_end` at the `run_fleet` call site (`:1319`), below the pin resolution so it receives the pinned value.

Keeping the parameter optional preserves every other caller's behaviour; only the CLI path pins it.

- [ ] **Step 7: Verify the flag pins the fleet**

Two things to prove: that an over-reaching pin fails closed, and that the pin actually governs the inception count.

```bash
.venv/bin/python example/mo_volmodels/12_snowball_volmodel_backtest.py \
  --gate-g3 --data-end 2026-08-15 2>&1 | tail -3
```
Expected: a `ValidationError` naming the spot-cache end — the pin refuses to invent data.

Then prove the pin reaches `schedule_inceptions`, by pinning to a date that *changes* the answer. `MIN_OBSERVABLE_MONTHS = 12`, so `--data-end 2026-06-30` must drop the 2025-07-01 inception and yield 26:

```python
# append to test/mo_volmodels/test_cohort.py
def test_data_end_pin_governs_the_inception_count():
    """The pin must reach schedule_inceptions, not only the task windows.

    prepare_inceptions used to re-derive data_end from the spot CSV, so a pin
    that only touched run_fleet left the fleet size floating -- invisible while
    the pin and the cache end coincide, and wrong the next weekday.
    """
    import pandas as pd
    s12 = _load_stage12_for_cohort()
    history_dir = PROJECT_ROOT / "example/mo_volmodels/data/history"
    spot = pd.read_csv(history_dir / "csi1000_spot.csv")
    calendar = s12.stage11().TradingCalendar.from_spot_csv(
        history_dir / "csi1000_spot.csv"
    )
    kwargs = dict(
        calendar=calendar,
        data_start=pd.Timestamp(spot["date"].iloc[0]).date(),
        first_admitted_surface=date(2023, 5, 4),
    )
    assert len(s12.schedule_inceptions(data_end=date(2026, 7, 31), **kwargs)) == 27
    assert len(s12.schedule_inceptions(data_end=date(2026, 6, 30), **kwargs)) == 26
```

Add the loader alongside the others in that file:

```python
def _load_stage12_for_cohort():
    path = PROJECT_ROOT / "example/mo_volmodels/12_snowball_volmodel_backtest.py"
    spec = importlib.util.spec_from_file_location("s12_cohort", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["s12_cohort"] = module
    spec.loader.exec_module(module)
    return module
```

- [ ] **Step 8: Commit**

```bash
git add example/mo_volmodels/cohort.py test/mo_volmodels/test_cohort.py \
        example/mo_volmodels/12_snowball_volmodel_backtest.py
git commit -m "feat(mo): pin the re-baseline surface cohort at 2026-07-31"
```

---

## Task 0B: Seed the fleet calibration cache from the daily pipeline

**Files:**
- Create: `example/mo_volmodels/seed_calibration_cache.py`
- Modify: `test/mo_volmodels/test_cohort.py` (append the seed tests)

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: `seed(src: Path, dst: Path, *, dry_run: bool = False) -> dict` with keys `n_source`, `n_copied`, `n_skipped_existing`, `by_variant` (dict), `fingerprints` (sorted list of distinct `config_fingerprint` values seen).

**Why:** `output/mo_daily_calibration/calibration_cache/` already holds 240 dates × `{localvol, heston, heston_slv}` = 720 entries covering 2025-07-31 → 2026-07-31. Their stored `config_fingerprint` is byte-identical to what stage 12's full-quality `VolModelCalibrationConfig(slv_n_steps=40, slv_n_x=161, slv_n_z=81)` computes — verified `240/240` on every variant. Since the cache key is `sha256(surface_sha | variant | fingerprint)` and the filename is `{variant}-{key}.json`, a plain file copy is a guaranteed hit. This removes the SLV leverage solves — the dominant per-day cost — for the final year of the replay window in Task 8.

This is a copy, not a merge: entries already present in `dst` are left alone, because the fleet's own writes are atomic and authoritative.

**Only seed full-quality runs.** Both consumers build the same config at full quality — stage 12 at `12_snowball_volmodel_backtest.py:684` and the gate's `make_calibrator` at `11_pde_convergence_gate.py:444` — but both drop to `slv_n_steps=12, slv_n_x=61, slv_n_z=31` under `--quick`. That is a different fingerprint, so seeding a quick run copies 720 files that every lookup misses. Harmless, but do not read the copy count as a speedup there.

- [ ] **Step 1: Write the failing test**

```python
# append to test/mo_volmodels/test_cohort.py
import importlib.util as _ilu

SEED_MODULE = PROJECT_ROOT / "example/mo_volmodels/seed_calibration_cache.py"


def _load_seed():
    spec = _ilu.spec_from_file_location("mo_seed", SEED_MODULE)
    module = _ilu.module_from_spec(spec)
    sys.modules["mo_seed"] = module
    spec.loader.exec_module(module)
    return module


def _entry(path: Path, variant: str, key: str, fingerprint: str) -> None:
    (path / f"{variant}-{key}.json").write_text(
        json.dumps({"variant": variant, "config_fingerprint": fingerprint,
                    "surface_date": "2026-07-31", "schema_version": 1})
    )


def test_seed_copies_entries_and_reports_by_variant(tmp_path):
    mod = _load_seed()
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    _entry(src, "heston", "aaa", "fp1")
    _entry(src, "localvol", "bbb", "fp2")
    summary = mod.seed(src, dst)
    assert summary["n_source"] == 2
    assert summary["n_copied"] == 2
    assert summary["by_variant"] == {"heston": 1, "localvol": 1}
    assert sorted(summary["fingerprints"]) == ["fp1", "fp2"]
    assert (dst / "heston-aaa.json").is_file()


def test_seed_never_overwrites_an_existing_entry(tmp_path):
    mod = _load_seed()
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    _entry(src, "heston", "aaa", "fp1")
    (dst / "heston-aaa.json").write_text('{"mine": true}')
    summary = mod.seed(src, dst)
    assert summary["n_copied"] == 0
    assert summary["n_skipped_existing"] == 1
    assert json.loads((dst / "heston-aaa.json").read_text()) == {"mine": True}


def test_seed_dry_run_writes_nothing(tmp_path):
    mod = _load_seed()
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    _entry(src, "heston", "aaa", "fp1")
    summary = mod.seed(src, dst, dry_run=True)
    assert summary["n_copied"] == 1
    assert not any(dst.iterdir())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_cohort.py -n0 -q -k seed`
Expected: FAIL — `seed_calibration_cache.py` does not exist.

- [ ] **Step 3: Write the script**

```python
# example/mo_volmodels/seed_calibration_cache.py
"""Seed a fleet calibration cache from the daily pipeline's cache.

The cache key is ``sha256(surface_sha | variant | config_fingerprint)`` and the
filename is ``{variant}-{key}.json``, so an entry written under one config is
reusable by any run that computes the same fingerprint — verified 240/240 per
variant against stage 12's full-quality config.  Copying is therefore sound and
needs no re-validation here; the calibrator re-checks schema version and
surface sha on read and treats a mismatch as a miss.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = PROJECT_ROOT / "output/mo_daily_calibration/calibration_cache"


def seed(src: Path, dst: Path, *, dry_run: bool = False) -> Dict[str, Any]:
    """Copy every cache entry from ``src`` into ``dst`` without overwriting."""
    src, dst = Path(src), Path(dst)
    if not src.is_dir():
        raise FileNotFoundError(f"source cache directory not found: {src}")
    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    by_variant: Counter = Counter()
    fingerprints = set()
    n_source = n_copied = n_skipped = 0
    for path in sorted(src.glob("*.json")):
        n_source += 1
        payload = json.loads(path.read_text(encoding="utf-8"))
        fingerprints.add(payload.get("config_fingerprint"))
        target = dst / path.name
        if target.exists():
            n_skipped += 1
            continue
        n_copied += 1
        by_variant[str(payload.get("variant"))] += 1
        if not dry_run:
            shutil.copy2(path, target)
    return {
        "n_source": n_source,
        "n_copied": n_copied,
        "n_skipped_existing": n_skipped,
        "by_variant": dict(by_variant),
        "fingerprints": sorted(f for f in fingerprints if f is not None),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default=str(DEFAULT_SRC))
    parser.add_argument("--dst", required=True, help="<out-dir>/calibration_cache")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    summary = seed(Path(args.src), Path(args.dst), dry_run=args.dry_run)
    print(
        f"[seed] source {summary['n_source']}, copied {summary['n_copied']}, "
        f"skipped {summary['n_skipped_existing']} existing"
    )
    print(f"[seed] by variant: {summary['by_variant']}")
    print(f"[seed] distinct fingerprints: {len(summary['fingerprints'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_cohort.py -n0 -q`
Expected: PASS, all tests from Tasks 0 and 0B.

- [ ] **Step 5: Dry-run against the real cache**

Run:
```bash
.venv/bin/python example/mo_volmodels/seed_calibration_cache.py \
  --dst output/volmodel_backtest/calibration_cache --dry-run
```
Expected: `source 720, copied 720, skipped 0 existing`, `by variant: {'heston': 240, 'heston_slv': 240, 'localvol': 240}`, and **`distinct fingerprints: 3`** — one per variant. More than 3 means the daily pipeline's config drifted from stage 12's and the entries will miss; investigate before relying on them.

- [ ] **Step 6: Commit**

```bash
git add example/mo_volmodels/seed_calibration_cache.py test/mo_volmodels/test_cohort.py
git commit -m "feat(mo): seed the fleet calibration cache from the daily pipeline"
```

---

## Task 1: G1 — verify surface admission over the existing artifacts

**Files:**
- Create: `example/mo_volmodels/13_gate_g1_surface_admission.py`
- Create: `test/mo_volmodels/test_gate_g1_admission.py`

**Interfaces:**
- Consumes: `cohort.admitted_dates` and `cohort.COHORT_ASOF` from Task 0
- Produces: `verify_surface(day: date, iv_dir: Path) -> tuple[bool, str]` — `(ok, reason)`, empty reason iff ok; `scan_cohort(iv_dir: Path | None = None, history_dir: Path | None = None) -> dict` with keys `n_admitted`, `n_verified`, `failures` (list of `{date, reason}`), `min_expiries_seen`, `asof`.

**Why this is a verifier and not a rebuild:** the §7A.4 fixes touched calibration and the 2D PDE, neither of which builds surfaces. G1's job is to confirm every surface the fleet will consume exists on disk and carries at least 3 expiries (Dupire's requirement). See Global Constraints for why the history is not regenerated.

**Two corrections to the original draft of this task, both load-bearing:**

1. **Admission cannot be read from an artifact.** The draft checked `payload["admission"]["admitted"] is True`. That key does not exist — an artifact's `admission` block records the *criteria* the builder used (`min_expiries: 2`, `sabr_beta: 1.0`, `static_arbitrage_validation: …`) and carries no verdict. Every one of the 768 artifacts would have failed. The verdict lives in `surface_manifest.json`, which is what Task 0 reads.

2. **The artifact directory is not the cohort.** `iv_surface/` holds **768** files; only **766** are admitted. The two thin surfaces (2024-09-30, 2025-04-08) still have artifacts on disk — they were excluded in the manifest, not deleted. A directory glob would scan them, fail them on the 3-expiry rule, and halt Phase A on a false alarm. G1 therefore iterates the *admitted date list* and looks up each artifact, rather than walking the directory.

The glob is still worth one assertion in the other direction: every admitted date must have a file. That catches a deleted or unwritten artifact, which is the failure the gate actually exists to find.

- [ ] **Step 1: Write the failing test**

```python
# test/mo_volmodels/test_gate_g1_admission.py
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load():
    path = REPO / "example" / "mo_volmodels" / "13_gate_g1_surface_admission.py"
    spec = importlib.util.spec_from_file_location("g1", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["g1"] = mod          # @dataclass resolves cls.__module__ here
    spec.loader.exec_module(mod)
    return mod


def _artifact(iv_dir: Path, day: date, n_maturities: int) -> None:
    iv_dir.mkdir(parents=True, exist_ok=True)
    tag = day.strftime("%Y%m%d")
    (iv_dir / f"mo_iv_surface_{tag}.json").write_text(
        json.dumps(
            {
                "trade_date": day.isoformat(),
                "maturities": [0.1 * (i + 1) for i in range(n_maturities)],
                # The real block: criteria, not a verdict.  Present here so the
                # test proves G1 does not read a verdict out of it.
                "admission": {"min_expiries": 2, "sabr_beta": 1.0},
            }
        )
    )


def test_surface_with_three_expiries_passes(tmp_path):
    g1 = _load()
    _artifact(tmp_path, date(2026, 7, 31), 3)
    ok, reason = g1.verify_surface(date(2026, 7, 31), tmp_path)
    assert ok is True
    assert reason == ""


def test_thin_surface_fails_the_dupire_rule(tmp_path):
    """The Phase-1 builder admits 2-expiry surfaces; Dupire needs 3."""
    g1 = _load()
    _artifact(tmp_path, date(2026, 7, 31), 2)
    ok, reason = g1.verify_surface(date(2026, 7, 31), tmp_path)
    assert ok is False
    assert "expiries" in reason


def test_missing_artifact_for_an_admitted_date_fails_closed(tmp_path):
    """The failure G1 exists to catch: the manifest admits it, disk lacks it."""
    g1 = _load()
    tmp_path.mkdir(parents=True, exist_ok=True)
    ok, reason = g1.verify_surface(date(2026, 7, 31), tmp_path)
    assert ok is False
    assert "no artifact" in reason


def test_scan_never_walks_the_directory(tmp_path, monkeypatch):
    """768 artifacts on disk, 766 admitted — the two extra must not be scanned.

    This is the regression that the first draft of this gate would have hit:
    a glob over iv_surface/ picks up the excluded thin surfaces, fails them on
    the 3-expiry rule, and halts Phase A on a false alarm.
    """
    g1 = _load()
    admitted = date(2026, 7, 30)
    excluded = date(2026, 7, 31)
    _artifact(tmp_path, admitted, 3)
    _artifact(tmp_path, excluded, 2)      # on disk, but NOT admitted
    monkeypatch.setattr(g1.cohort, "admitted_dates", lambda *a, **k: [admitted])

    summary = g1.scan_cohort(iv_dir=tmp_path)
    assert summary["n_admitted"] == 1
    assert summary["n_verified"] == 1
    assert summary["failures"] == []
    assert summary["min_expiries_seen"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_g1_admission.py -n0 -q`
Expected: FAIL — `FileNotFoundError` / `spec_from_file_location` returns None for the missing script.

- [ ] **Step 3: Write the script**

```python
# example/mo_volmodels/13_gate_g1_surface_admission.py
"""Gate G1: verify the pinned surface cohort against the artifacts on disk.

This is a verifier, not a builder.  It answers one question per admitted date:
"is there an artifact the fleet can actually price against?"

Two things it deliberately does NOT do:

* It does not read admission out of an artifact.  An artifact's ``admission``
  block records the *criteria* the builder used (``min_expiries: 2``,
  ``sabr_beta``, ...) and carries no per-surface verdict.  The verdict is in
  ``surface_manifest.json``, which ``cohort.admitted_dates`` reads.
* It does not walk ``iv_surface/``.  That directory holds 768 files while 766
  are admitted: the two thin surfaces (2024-09-30, 2025-04-08) were excluded in
  the manifest, not deleted.  Globbing would fail them on the 3-expiry rule and
  halt the study on a false alarm.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IV_DIR = PROJECT_ROOT / "example/mo_volmodels/data/history/iv_surface"

# Dupire local vol needs at least three expiries to form dw/dT; the Phase-1
# builder admits two, so G1 re-checks rather than trusting the manifest alone.
MIN_EXPIRIES = 3


def _load_cohort():
    """Import the sibling cohort module (the stages are not a package)."""
    path = Path(__file__).resolve().parent / "cohort.py"
    spec = importlib.util.spec_from_file_location("mo_cohort", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mo_cohort"] = module
    spec.loader.exec_module(module)
    return module


cohort = _load_cohort()


def artifact_path(day: date, iv_dir: Path) -> Path:
    return Path(iv_dir) / f"mo_iv_surface_{day.strftime('%Y%m%d')}.json"


def verify_surface(day: date, iv_dir: Path) -> Tuple[bool, str]:
    """Return (ok, reason) for one admitted date.  Empty reason iff ok."""
    path = artifact_path(day, iv_dir)
    if not path.is_file():
        return False, f"no artifact at {path.name}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"artifact unreadable: {exc}"
    n_expiries = len(payload.get("maturities") or [])
    if n_expiries < MIN_EXPIRIES:
        return False, (
            f"{n_expiries} expiries < {MIN_EXPIRIES} required by Dupire"
        )
    return True, ""


def scan_cohort(
    iv_dir: Optional[Path] = None,
    history_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Verify every admitted date at or before the pin; JSON-safe summary."""
    iv_dir = Path(iv_dir or DEFAULT_IV_DIR)
    admitted = cohort.admitted_dates(history_dir)

    failures = []
    min_expiries_seen = None
    for day in admitted:
        ok, reason = verify_surface(day, iv_dir)
        if not ok:
            failures.append({"date": day.isoformat(), "reason": reason})
            continue
        payload = json.loads(artifact_path(day, iv_dir).read_text(encoding="utf-8"))
        n_exp = len(payload.get("maturities") or [])
        min_expiries_seen = (
            n_exp if min_expiries_seen is None else min(min_expiries_seen, n_exp)
        )
    return {
        "gate": "G1",
        "asof": cohort.COHORT_ASOF.isoformat(),
        "iv_dir": str(iv_dir),
        "n_admitted": len(admitted),
        "n_verified": len(admitted) - len(failures),
        "failures": failures,
        "min_expiries_seen": min_expiries_seen,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iv-dir", default=str(DEFAULT_IV_DIR))
    parser.add_argument("--history-dir", default=None)
    parser.add_argument("--out", default="output/gate_g1_admission.json")
    args = parser.parse_args(argv)

    summary = scan_cohort(
        Path(args.iv_dir),
        Path(args.history_dir) if args.history_dir else None,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))

    print(f"[G1] asof {summary['asof']}, "
          f"admitted {summary['n_admitted']}, "
          f"verified {summary['n_verified']}, "
          f"min expiries {summary['min_expiries_seen']}")
    for f in summary["failures"][:20]:
        print(f"  FAIL {f['date']}: {f['reason']}")
    if summary["failures"]:
        print(f"[G1] FAILED — {len(summary['failures'])} surface(s) unusable")
        return 1
    print("[G1] PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_g1_admission.py -n0 -q`
Expected: PASS, 4 tests.

- [ ] **Step 5: Run G1 for real**

Run: `.venv/bin/python example/mo_volmodels/13_gate_g1_surface_admission.py`
Expected: `[G1] asof 2026-07-31, admitted 766, verified 766, min expiries 3` then `[G1] PASSED`, exit 0.

Interpreting a non-match:

| symptom | meaning | action |
|---|---|---|
| `admitted` > 766 | the scheduler advanced the history past the pin, or `COHORT_ASOF` was raised | **stop.** Re-derive the plan's cell counts and check whether the fleet is still 27 (Task 0 Step 5) |
| `admitted` < 766 | a surface lost its `ok` status, or `study_admission` grew | **stop and report.** Do not edit the history |
| a `no artifact` failure | the manifest admits a date whose file is missing | **stop and report.** This is the failure G1 exists to catch |
| a `< 3 expiries` failure | a thin surface leaked into the admitted set | re-run `exclude_thin_surfaces.py`, then re-run G1 |

- [ ] **Step 6: Commit**

```bash
git add example/mo_volmodels/13_gate_g1_surface_admission.py test/mo_volmodels/test_gate_g1_admission.py
git commit -m "feat(mo): add Gate G1 surface-admission verifier over existing artifacts"
```

---

## Task 2: G4 — re-solve the fair coupon on the 0.4.0 engines

**Files:**
- Modify: none (uses `12_snowball_volmodel_backtest.py` with Task 0's `--data-end`)

**Interfaces:**
- Consumes: `solve_fair_coupon` (`12_snowball_volmodel_backtest.py:524`); `--data-end` from Task 0
- Produces: `output/volmodel_backtest/inception_coupons.json` (or whatever the runner's prepare step writes) — the per-inception coupon table every later phase depends on.

**Why re-run:** the coupon is solved on `flat_bsm` — a **1D BSM PDE** (`12_snowball_volmodel_backtest.py:1001`). The §7A.4 fixes touch the Heston preset and the 2D `v0_boundary`, so they do **not** move these roots. 0.4.0's PDE grid rewrite does: the spec records 15.0975% → 15.0707% on the first inception. This is a spec correction — §10 implies G4 re-runs *because of* §7A.4; it re-runs because of 0.4.0, and it has not been run since.

- [ ] **Step 1: Find the coupon-only entry point**

Run: `.venv/bin/python example/mo_volmodels/12_snowball_volmodel_backtest.py --help`

Identify the flag that stops after the prepare/coupon phase. If none exists, run with `--variants flat_bsm` and take the coupon table from the run manifest.

- [ ] **Step 2: Solve coupons for the pinned fleet**

`--data-end` is **mandatory here**, not cosmetic: without it the fleet is whatever the scheduler last wrote, and a run started after 2026-08-01 silently solves a 28th coupon that no other gate covers.

Run (background; ~3 PDE prices × 27 inceptions, expect 20–40 min):

```bash
.venv/bin/python example/mo_volmodels/12_snowball_volmodel_backtest.py \
  --variants flat_bsm --data-end 2026-07-31 --workers 4 2>&1 \
  | tee output/g4_coupons.log
```

Expected per line: `[k/27] YYYY-MM-DD s0=… coupon=15.xxxx% |PV|=… (N iters, …s)`.

- [ ] **Step 3: Verify G4's fail-closed contract held**

Every inception must appear with a converged coupon. `solve_fair_coupon` raises rather than returning an unconverged or boundary value, so a completed run **is** the gate passing. Confirm explicitly:

```bash
grep -c "coupon=" output/g4_coupons.log     # expect 27
grep -E "coupon=(0\.0000|80\.0000)%" output/g4_coupons.log   # expect no matches (bounds)
```

A count of 28 means the pin did not take — check the `--data-end` argument reached `run_fleet` rather than accepting the extra row.

- [ ] **Step 4: Record the coupon table in the spec**

Add a short table to §2 of the spec: inception, `s0`, solved coupon, `|PV|`. State the first inception's coupon against the recorded 0.3.0 value (15.0975%) and 0.4.0 value (15.0707%) so the re-baseline is auditable.

- [ ] **Step 5: Commit**

```bash
git add -f docs/superpowers/specs/2026-07-30-snowball-volmodel-backtest-040-rebaseline-design.md
git commit -m "docs(mo): record the G4 fair-coupon re-solve on 0.4.0"
```

---

## Task 2B: `PDEEngine` must forward event stats to its solver

**Files:**
- Modify: `quantark/asset/equity/engine/pde_engine.py` (add a `calculate_event_stats` override)
- Create: `test/test_pde_engine_event_stats.py`

**Interfaces:**
- Produces: `PDEEngine.calculate_event_stats(product, pricing_env) -> Optional[AutocallableEventStats]` — delegates to the product's solver.

**Found by running Task 2.** The G4 coupon solve succeeded (27/27), but the `flat_bsm` fleet replay that followed failed **27/27** with:

```
event-stats engine PDEEngine returned no stats while event probabilities were
requested; set event_stats_fallback='mc' to opt into the MC fallback
```

**Root cause, traced:** `PDEEngine` (`pde_engine.py:43`) is a dispatch facade — `price()` routes to a product-specific solver via `_get_solver`, and `PRODUCT_SOLVER_MAP` sends `SnowballOption` to `SnowballPDESolver`. That solver **does** implement `calculate_event_stats` (`pde/snowball_pde_solver.py:445`), returning exact PDE event statistics. But the facade never overrides `calculate_event_stats`, so it inherits `BaseEngine`'s default (`base_engine.py:232`), which returns `None` meaning "not supported".

That gap was silent until 0.4.0. The replay consolidation changed `event_stats_fallback` to default `"none"` (`replay/config.py:196`), which correctly treats a `None` return as a failure rather than a silent no-op. So a pre-existing plumbing gap became a hard stop for every PDE-priced replay.

**Why the fix is delegation, not `event_stats_fallback="mc"`.** The exact statistics already exist and are cheap — they come out of the same PDE sweep. Switching on an MC fallback would substitute a sampled approximation for an available exact result, in a variant whose entire purpose is to be the deterministic control. Turning off `--event-probabilities` would hide the defect rather than fix it.

- [ ] **Step 1: Write the failing test**

```python
# test/test_pde_engine_event_stats.py
"""PDEEngine must expose the event stats its solvers already compute.

The facade dispatches price() to a product-specific solver but inherited
BaseEngine.calculate_event_stats, which returns None for "unsupported".
SnowballPDESolver implements it, so the facade was hiding a working result --
invisible until the replay layer's fail-closed default turned None into an
error.
"""
from datetime import datetime

import pytest

from quantark.asset.equity.engine.pde_engine import PDEEngine
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.util.exceptions import ValidationError


def test_pde_engine_returns_the_solver_event_stats(snowball_product, snowball_env):
    stats = PDEEngine().calculate_event_stats(snowball_product, snowball_env)
    assert stats is not None, "facade swallowed the solver's event stats"
    assert len(stats.observation_dates) > 0
    total_ko = sum(stats.ko_probabilities)
    assert 0.0 <= total_ko <= 1.0 + 1e-9


def test_pde_engine_event_stats_match_the_solver_directly(
    snowball_product, snowball_env
):
    """Delegation must not transform the result."""
    from quantark.asset.equity.engine.pde.snowball_pde_solver import (
        SnowballPDESolver,
    )

    facade = PDEEngine().calculate_event_stats(snowball_product, snowball_env)
    direct = SnowballPDESolver().calculate_event_stats(
        snowball_product, snowball_env
    )
    assert facade.ko_probabilities == pytest.approx(direct.ko_probabilities)
```

Build `snowball_product` / `snowball_env` fixtures from the study's own term sheet so the test exercises a realistic instrument, not a degenerate one. `example/mo_volmodels/11_pde_convergence_gate.py` constructs exactly such a product — read how it does it and follow that shape rather than inventing parameters. Keep the grid coarse enough that the test runs in a couple of seconds.

Add a third test for the unsupported-product path, asserting whichever behaviour you implement in Step 3 — and state your choice in the report.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/test_pde_engine_event_stats.py -n0 -q`
Expected: FAIL on `assert stats is not None` — the facade returns `None` today.

- [ ] **Step 3: Implement the delegation**

Add to `PDEEngine`, near `price()`:

```python
    def calculate_event_stats(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Optional[AutocallableEventStats]:
        """Forward to the product's solver, which owns the computation.

        The base class returns None for "unsupported"; that default silently
        discarded working solver results, and the replay layer's fail-closed
        event-stats policy turned the discard into a hard failure.
        """
        return self._get_solver(product).calculate_event_stats(product, pricing_env)
```

One judgement call is yours: `_get_solver` raises `ValidationError` for a product type not in `PRODUCT_SOLVER_MAP`. Decide whether `calculate_event_stats` should propagate that (fail closed — the caller asked for something this engine cannot do) or catch it and return `None` (preserve the base class's "unsupported means None" contract). Both are defensible. Pick one, make the third test assert it, and justify the choice in one sentence in your report.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest test/test_pde_engine_event_stats.py -n0 -q`
Expected: PASS.

- [ ] **Step 5: Check for blast radius — this is the risky part**

`calculate_event_stats` now returns real data where it returned `None`. Anything that branched on `None` changes behaviour. Run the full suite:

```bash
.venv/bin/python -m pytest -q 2>&1 | tail -30
```

Known pre-existing failures, NOT yours: 3 in `test_adi_core_tau_exactness.py`, 1 in `test_snowball_pde_knocked_in_grid.py`. **Any other failure is caused by this change** — investigate it rather than accepting it. Pay particular attention to golden/oracle tests and to anything asserting `calculate_event_stats(...) is None`.

- [ ] **Step 6: Re-run the fleet phase that exposed this**

```bash
.venv/bin/python example/mo_volmodels/12_snowball_volmodel_backtest.py \
  --variants flat_bsm --data-end 2026-07-31 --workers 4 2>&1 \
  | tee output/g4_fleet_recheck.log
```

Expected: `[summary] 27/27 runs completed` — previously `0/27 runs completed, 27 failed`. The coupon solve is cached in `inceptions.json`, so this is the replay phase only.

Report the KO / matured / censored split. Per the spec, all 27 realized paths knock out.

- [ ] **Step 7: Commit**

```bash
git add quantark/asset/equity/engine/pde_engine.py test/test_pde_engine_event_stats.py
git commit -m "fix(equity): PDEEngine forwards event stats to its product solver"
```

---

## Task 3: Add the `flat_bsm_quad` engine-control variant

**Files:**
- Modify: `example/mo_volmodels/12_snowball_volmodel_backtest.py:135-141` (`VARIANTS`), `:144-153` (`VariantSpec`), `:158-194` (`VARIANT_SPECS`), `:669-705` (`make_engine_config`)
- Create: `test/mo_volmodels/test_gate_scope.py` — **this task creates the file**, including the module-loader helpers Tasks 4–7 then reuse.

**Interfaces:**
- Produces: `"flat_bsm_quad"` in `VARIANTS`; a new `VariantSpec.pricing_engine_type: EngineType = EngineType.PDE` field; and `VARIANT_SPECS["flat_bsm_quad"]` identical to `flat_bsm` except `pricing_engine_type=EngineType.QUADRATURE`.

**The distinguishing field does not exist yet.** `VariantSpec` (`:144`) carries only `name`, `vol_source`, `surface_vol_mode`, `vol_model`, `description` — the solver is picked by `GateRouting.solver_for(spec.vol_model)`, which routes *vol-model* engines only. `make_engine_config` hardcodes `pricing_engine_type=EngineType.PDE` for every variant (`:691`). So `flat_bsm_quad` needs a new field; it cannot be expressed with the current dataclass. `EngineType.QUADRATURE` and `AutocallableEngineConfig.quad_params` both already exist.

**Why this comes before G2:** the spec's §5.1 gates all six variants, but §10 adds `flat_bsm_quad` at step 3 — *after* G2 at step 2. That ordering is impossible: you cannot gate a variant that does not exist. This plan pulls the addition forward. Note the spec correction in the commit message.

`flat_bsm` and `flat_bsm_quad` reference each other in §5.1. That is not circular — it is one PDE-vs-QUAD comparison serving as the admission test for both routes and as the study's engine control.

- [ ] **Step 1: Write the failing test**

```python
# test/mo_volmodels/test_gate_scope.py  (NEW FILE — Tasks 4-7 append to it)
import importlib.util
import sys
from pathlib import Path

import pytest

from quantark.util.enum.engine_enums import EngineType

REPO = Path(__file__).resolve().parents[2]


def _load(stem):
    """Import a numbered stage script (the stages are not a package)."""
    path = REPO / "example" / "mo_volmodels" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem.split("_")[0], path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod        # @dataclass resolves cls.__module__ here
    spec.loader.exec_module(mod)
    return mod


def _load_gate():
    return _load("11_pde_convergence_gate")


def _load_stage12():
    return _load("12_snowball_volmodel_backtest")


def test_flat_bsm_quad_differs_from_flat_bsm_in_engine_only():
    """The engine control is only a control if the market data is identical."""
    s12 = _load_stage12()
    assert "flat_bsm_quad" in s12.VARIANTS
    bsm = s12.VARIANT_SPECS["flat_bsm"]
    quad = s12.VARIANT_SPECS["flat_bsm_quad"]
    assert quad.vol_source == bsm.vol_source
    assert quad.surface_vol_mode == bsm.surface_vol_mode
    assert quad.vol_model == bsm.vol_model == "bsm"
    assert bsm.pricing_engine_type == EngineType.PDE
    assert quad.pricing_engine_type == EngineType.QUADRATURE


def test_engine_config_honours_the_variant_pricing_engine_type():
    """A new VariantSpec field is inert until make_engine_config reads it."""
    s12 = _load_stage12()
    routing = s12.GateRouting("p", None, {}, {})
    cfg = s12.make_engine_config("flat_bsm_quad", routing=routing)
    assert cfg.pricing_engine_type == EngineType.QUADRATURE
    assert s12.make_engine_config(
        "flat_bsm", routing=routing
    ).pricing_engine_type == EngineType.PDE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_scope.py -n0 -q -k flat_bsm_quad`
Expected: FAIL — `"flat_bsm_quad" not in VARIANTS` (and `AttributeError` on `pricing_engine_type`).

- [ ] **Step 3: Add the field, the variant, and the wiring**

Add the field to `VariantSpec` (`:144`), defaulted so the five existing entries are unchanged:

```python
@dataclass(frozen=True)
class VariantSpec:
    """How one model variant configures the daily pricing engine."""

    name: str
    vol_source: str
    surface_vol_mode: str
    vol_model: str
    description: str
    # Engine family for the DAILY PRICING engine.  Defaults to PDE so the
    # five original variants are untouched; flat_bsm_quad overrides it to
    # make an engine control that differs from flat_bsm in engine only.
    # The surface and event-stats engines stay 1D PDE for every variant.
    pricing_engine_type: EngineType = EngineType.PDE

    def uses_calibration(self) -> bool:
        return self.vol_model != "bsm"
```

Add to `VARIANTS` (after `"flat_bsm"`) and to `VARIANT_SPECS`:

```python
    "flat_bsm_quad": VariantSpec(
        name="flat_bsm_quad",
        vol_source="surface",
        surface_vol_mode="flat_atm_remaining",
        vol_model="bsm",
        description=(
            "Engine control: flat_bsm's market data priced by FFT "
            "regime-switching quadrature instead of the 1D PDE"
        ),
        pricing_engine_type=EngineType.QUADRATURE,
    ),
```

In `make_engine_config` (`:691`), replace the hardcoded `pricing_engine_type=EngineType.PDE` with `pricing_engine_type=spec.pricing_engine_type`, and add `quad_params=QuadParams()` alongside `pde_params=PDEParams()` so the QUAD route has its settings. Import `QuadParams` from wherever `PDEParams` comes from and confirm the name with:

`.venv/bin/python -c "import dataclasses; from quantark.backtest.replay.config import AutocallableEngineConfig as C; print([(f.name, f.type) for f in dataclasses.fields(C) if 'param' in f.name])"`

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_scope.py -n0 -q -k flat_bsm_quad`
Expected: PASS.

- [ ] **Step 5: Smoke-price one inception on the new variant**

Run: `.venv/bin/python example/mo_volmodels/12_snowball_volmodel_backtest.py --variants flat_bsm_quad --limit 1`
Expected: completes without error. A QUAD failure here (unreachable barrier, dense KI) is a §9/G5 finding — record it, do not paper over it.

- [ ] **Step 6: Commit**

```bash
git add example/mo_volmodels/12_snowball_volmodel_backtest.py test/mo_volmodels/test_gate_scope.py
git commit -m "feat(mo): add flat_bsm_quad engine-control variant

Spec §5.1 gates six variants but §10 added this one after G2; a variant
cannot be gated before it exists, so the addition moves ahead of the gate."
```

---

## Task 4: Generalise G2 from 2 hard-coded variants to a production/reference pair table

**Files:**
- Modify: `example/mo_volmodels/11_pde_convergence_gate.py:88` (`VARIANTS`), `:460-494` (`_make_mc_engine` / `_make_pde_engine`), `:776` (`validate_decision_payload`), `:844` (the `for variant in VARIANTS` loop)
- Test: `test/mo_volmodels/test_gate_scope.py`

**Interfaces:**
- Produces: `GATE_PAIRS: dict[str, GatePair]` where `GatePair` is a frozen dataclass with fields `production: str`, `reference: str` (free-text labels for the record) and two builders `build_production(model, case) -> engine`, `build_reference(model, case) -> engine`. `process_date` consumes `GATE_PAIRS` instead of branching on `variant == VOL_MODEL_HESTON`.

The §5.1 table this encodes:

| variant | production | reference |
|---|---|---|
| `flat_bsm` | PDE 1D | QUAD (high `grid_points`) |
| `flat_bsm_quad` | QUAD | PDE 1D `accuracy="high"` |
| `ts_bsm` | QUAD | PDE 1D `accuracy="high"` |
| `localvol` | PDE 1D | `LocalVolSnowballMCEngine` (RQMC) |
| `heston` | gate-decided | `QESnowballMCEngine` (RQMC QE-M) |
| `heston_slv` | gate-decided | `HestonSLVQESnowballMCEngine` |

- [ ] **Step 1: Write the failing test**

```python
# append to test/mo_volmodels/test_gate_scope.py
# (_load / _load_gate / _load_stage12 already exist — Task 3 created them)

EXPECTED = {
    "flat_bsm", "flat_bsm_quad", "ts_bsm", "localvol", "heston", "heston_slv",
}


def test_gate_covers_every_study_variant():
    gate = _load_gate()
    assert set(gate.GATE_PAIRS) == EXPECTED
    assert set(gate.VARIANTS) == EXPECTED


def test_gate_variants_match_the_backtest_variants():
    """A variant the fleet runs but the gate never admitted is unrouted."""
    assert set(_load_gate().VARIANTS) == set(_load_stage12().VARIANTS)


def test_every_pair_uses_two_distinct_numerical_methods():
    gate = _load_gate()
    for name, pair in gate.GATE_PAIRS.items():
        assert pair.production != pair.reference, name


def test_mc_referenced_variants_are_exactly_the_ones_needing_std_error():
    """Only these three get a 2*mc_se tolerance term; the rest get the floor."""
    gate = _load_gate()
    mc_refs = {n for n, p in gate.GATE_PAIRS.items() if p.reference_is_mc}
    assert mc_refs == {"localvol", "heston", "heston_slv"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_scope.py -n0 -q`
Expected: FAIL — `AttributeError: module has no attribute 'GATE_PAIRS'`.

- [ ] **Step 3: Add the pair table**

Insert near the existing engine factories (`~:445`):

```python
@dataclass(frozen=True)
class GatePair:
    """One variant's production engine and its independent reference.

    ``production`` / ``reference`` are labels recorded in the evidence; the
    builders are what actually run.  The two must be different numerical
    methods, or the cell is a common-mode comparison and proves nothing.

    Builder signature: ``(model, grid) -> engine``.  ``model`` is the
    CalibratedVolModel for calibrated variants and ``None`` for the BSM ones;
    ``grid`` is the ladder level's knob tuple (see _production_grid below).
    ``reference_is_mc`` drives whether a std error is expected: a
    deterministic reference has ``mc_se = None`` and the flat TOL_ABS floor.
    """

    production: str
    reference: str
    build_production: Callable[..., Any]
    build_reference: Callable[..., Any]
    reference_is_mc: bool


def _bsm_pde(accuracy: str = "standard"):
    def build(model, grid):
        return SnowballPDESolver(PDEParams(accuracy=accuracy))
    return build


def _bsm_quad(grid_points: Optional[int] = None):
    def build(model, grid):
        params = QuadParams() if grid_points is None else QuadParams(grid_points=grid_points)
        return SnowballQuadEngine(params=params)
    return build


GATE_PAIRS: Dict[str, GatePair] = {
    "flat_bsm": GatePair(
        production="pde_1d", reference="quad_high",
        build_production=_bsm_pde("standard"),
        build_reference=_bsm_quad(grid_points=QUAD_REFERENCE_POINTS),
        reference_is_mc=False,
    ),
    "flat_bsm_quad": GatePair(
        production="quad", reference="pde_1d_high",
        build_production=_bsm_quad(),
        build_reference=_bsm_pde("high"),
        reference_is_mc=False,
    ),
    "ts_bsm": GatePair(
        production="quad", reference="pde_1d_high",
        build_production=_bsm_quad(),
        build_reference=_bsm_pde("high"),
        reference_is_mc=False,
    ),
    "localvol": GatePair(
        production="pde_1d_lv", reference="lv_mc_rqmc",
        build_production=lambda model, grid: LocalVolSnowballPDESolver(
            params=PDEParams(), local_vol_surface=model.local_vol_surface),
        build_reference=lambda model, grid: LocalVolSnowballMCEngine(
            local_vol_surface=model.local_vol_surface,
            params=_make_mc_params(MC_FULL, SEED),
            method=MonteCarloMethod.RANDOMIZED_QUASI),
        reference_is_mc=True,
    ),
    "heston": GatePair(
        production="pde_2d_adi", reference="qe_m_rqmc",
        build_production=lambda model, grid: _make_pde_engine(
            VOL_MODEL_HESTON, model, PDEParams(), grid),
        build_reference=lambda model, grid: _make_mc_engine(
            VOL_MODEL_HESTON, model, _make_mc_params(MC_FULL, SEED),
            MC_FULL["substeps_per_interval"]),
        reference_is_mc=True,
    ),
    "heston_slv": GatePair(
        production="pde_2d_adi_slv", reference="slv_qe_m_rqmc",
        build_production=lambda model, grid: _make_pde_engine(
            VOL_MODEL_HESTON_SLV, model, PDEParams(), grid),
        build_reference=lambda model, grid: _make_mc_engine(
            VOL_MODEL_HESTON_SLV, model, _make_mc_params(MC_FULL, SEED),
            MC_FULL["substeps_per_interval"]),
        reference_is_mc=True,
    ),
}

VARIANTS = tuple(GATE_PAIRS)
```

`QUAD_REFERENCE_POINTS` is a new module constant — set it to at least 4× the QUAD production default so the reference is genuinely finer, and record it in the decision payload's `config` block.

**Do not import stage 12 to reuse its engine factory.** Stage 12 already imports stage 11 (`stage11()`, `:196-218`) because stage 11 owns the certified term sheet; importing back would be a cycle. Stage 11 builds its own engines from `quantark.*`, exactly as it does today for PDE and MC. The cost is a drift risk between the two files' engine settings — mitigate it with the test below rather than with an import.

- [ ] **Step 3b: Guard against gate/fleet engine drift**

```python
def test_gate_prices_the_same_engine_family_the_fleet_will_run():
    """Stage 11 cannot import stage 12 (cycle), so assert the pairing instead."""
    gate, s12 = _load_gate(), _load_stage12()
    for name, spec in s12.VARIANT_SPECS.items():
        pair = gate.GATE_PAIRS[name]
        if spec.vol_model == "bsm":
            expected = "quad" if spec.pricing_engine_type.name == "QUADRATURE" else "pde_1d"
            assert pair.production.startswith(expected), name
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_scope.py -n0 -q`
Expected: PASS, 4 tests.

- [ ] **Step 5: Rewire `process_date` and the payload validator**

Replace the `variant == VOL_MODEL_HESTON` branches in `process_date` with `GATE_PAIRS[variant]` lookups. Keep the existing per-cell error handling verbatim — a reference failure must still mark every cell for that variant failed.

Two behaviours must be preserved for MC-referenced variants (`pair.reference_is_mc`) and *skipped* for deterministic ones:

- `mc_se` and `mc_se_pct` are `None` for a deterministic reference; the tolerance is then the flat `TOL_ABS_PCT`. Extend `gate_tolerance_pct` to accept `None` explicitly rather than passing `0.0`, so the record says "no MC error" instead of "zero MC error":

```python
def gate_tolerance_pct(mc_se_pct: Optional[float], tol_abs_pct: float = TOL_ABS_PCT) -> float:
    """Per-cell tolerance in % of notional: max(2 x MC SE, TOL_ABS).

    ``mc_se_pct=None`` means the reference is DETERMINISTIC (QUAD or a finer
    PDE), not that its error is zero -- the tolerance is then the flat floor.
    """
    if mc_se_pct is None:
        return float(tol_abs_pct)
    return max(MC_SE_FACTOR * float(mc_se_pct), float(tol_abs_pct))
```

  Also relax `validate_gate_payload`'s finiteness check (`:795`): it currently demands finite `mc_se` on any cell without an `error`, which every deterministic-reference cell would now trip. Require finite `mc_se` only when `reference_is_mc`.

- The refinement ladder applies to the *production* engine, and its knob differs by family. Add:

```python
# QUAD ladder over grid_points, and the 1D-PDE ladder over accuracy
# profiles.  Both mirror the 2D ladder's shape: coarse -> medium -> fine,
# with 'medium' the level the fleet would actually run.
QUAD_LADDER = {"coarse": 2048, "medium": 4096, "fine": 8192}
QUAD_REFERENCE_POINTS = 16384          # >= 4x the QUAD production default
PDE1D_LADDER = {"coarse": "fast", "medium": "standard", "fine": "high"}


def _production_grid(variant: str, level: str, T: float, quick: bool):
    """Ladder knob for the PRODUCTION engine of ``variant`` at ``level``.

    2D ADI ladders over (n_x, n_v, n_t); QUAD over grid_points; the 1D PDE
    over its accuracy profile.  Returns a JSON-safe dict stored unchanged in
    ``cell["grid"]``, so the evidence keeps ONE schema across families and a
    reader can always see which knob moved.
    """
    pair = GATE_PAIRS[variant]
    if pair.production.startswith("pde_2d"):
        entry = next(g for g in pde_ladder(T, quick) if g[0] == level)
        return {"kind": "adi_2d", "n_x": entry[1], "n_v": entry[2], "n_t": entry[3]}
    if pair.production.startswith("quad"):
        return {"kind": "quad", "grid_points": QUAD_LADDER[level]}
    return {"kind": "pde_1d", "accuracy": PDE1D_LADDER[level]}
```

  The builders in Step 3 must consume this dict rather than the bare tuple — update `_bsm_pde` / `_bsm_quad` to read `grid["accuracy"]` / `grid["grid_points"]`, and the 2D lambdas to pass `(grid["n_x"], grid["n_v"], grid["n_t"])` to `_make_pde_engine`. Verify the ladder is monotone before running anything expensive:

```python
def test_every_production_family_ladders_monotonically():
    gate = _load_gate()
    for variant in gate.VARIANTS:
        grids = [gate._production_grid(variant, lvl, 3.0, False)
                 for lvl in ("coarse", "medium", "fine")]
        kinds = {g["kind"] for g in grids}
        assert len(kinds) == 1, variant
        if grids[0]["kind"] == "quad":
            pts = [g["grid_points"] for g in grids]
            assert pts == sorted(pts) and len(set(pts)) == 3, variant
        elif grids[0]["kind"] == "adi_2d":
            assert [g["n_x"] for g in grids] == sorted(g["n_x"] for g in grids), variant
```

- [ ] **Step 6: Run the gate in quick mode**

Run: `.venv/bin/python example/mo_volmodels/11_pde_convergence_gate.py --quick --out-dir output/gate_smoke`
Expected: completes, writes `pde_convergence_gate.json` and `gate_decision.json` with six variant entries. Quick mode's verdict is explicitly non-production-valid — this step checks plumbing only.

- [ ] **Step 7: Commit**

```bash
git add example/mo_volmodels/11_pde_convergence_gate.py test/mo_volmodels/test_gate_scope.py
git commit -m "feat(mo): gate all six study variants via a production/reference pair table"
```

---

## Task 5: Evaluate the bias detector within maturity buckets

**Files:**
- Modify: `example/mo_volmodels/11_pde_convergence_gate.py:552` (`detect_systematic_bias`), `:590` (`decide_route`)
- Test: `test/mo_volmodels/test_gate_scope.py`

**Interfaces:**
- Produces: `detect_systematic_bias_bucketed(cells, tol_abs_pct=TOL_ABS_PCT) -> tuple[bool, dict]`. Input cells need `signed_diff_pct` and `case`. Returns biased-if-**any**-bucket-is-biased, with per-bucket detail under key `buckets`.

**Why:** §5.2. The PDE error changes sign with remaining maturity (positive at T=0.25, negative at T≥1). Pooled across maturities that scores a 0.75 sign fraction — under the 0.9 threshold — which is how the original G2 recorded `biased: false` at 0.533 while a systematic per-maturity bias was present. Within any single maturity the sign is unanimous.

The existing `case` field (`"full"` ≈ 3Y, `"decayed"` ≈ 1Y remaining) is the bucket key. Do not invent maturity bins.

- [ ] **Step 1: Write the failing test**

```python
def test_pooled_bias_hides_a_sign_flip_that_buckets_expose():
    """The exact failure mode from the original G2 run."""
    gate = _load_gate()
    cells = (
        [{"case": "full", "signed_diff_pct": +0.20} for _ in range(6)]
        + [{"case": "decayed", "signed_diff_pct": -0.20} for _ in range(6)]
    )
    pooled, _ = gate.detect_systematic_bias([c["signed_diff_pct"] for c in cells])
    bucketed, info = gate.detect_systematic_bias_bucketed(cells)

    assert pooled is False        # 0.5 sign fraction: reads as unbiased
    assert bucketed is True       # each bucket is unanimous
    assert set(info["buckets"]) == {"full", "decayed"}


def test_unbiased_cells_stay_unbiased_under_bucketing():
    gate = _load_gate()
    cells = [
        {"case": "full", "signed_diff_pct": v}
        for v in (+0.20, -0.18, +0.02, -0.21, +0.19, -0.05)
    ]
    biased, _ = gate.detect_systematic_bias_bucketed(cells)
    assert biased is False


def test_a_bucket_below_the_minimum_cell_count_cannot_flag_bias():
    """detect_systematic_bias needs >= 4 cells; a thin bucket must not vote."""
    gate = _load_gate()
    cells = (
        [{"case": "full", "signed_diff_pct": +0.02} for _ in range(6)]
        + [{"case": "decayed", "signed_diff_pct": +0.30} for _ in range(2)]
    )
    biased, info = gate.detect_systematic_bias_bucketed(cells)
    assert biased is False
    assert info["buckets"]["decayed"]["skipped"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_scope.py -n0 -q -k bias`
Expected: FAIL — `AttributeError: ... 'detect_systematic_bias_bucketed'`.

- [ ] **Step 3: Implement**

```python
def detect_systematic_bias_bucketed(
    cells: Sequence[Dict[str, Any]],
    tol_abs_pct: float = TOL_ABS_PCT,
) -> Tuple[bool, Dict[str, Any]]:
    """Bias detection WITHIN maturity buckets, never pooled across them.

    The 2D PDE error changes sign with remaining maturity (spec §5.2), so a
    pooled sign fraction averages the two regimes and reads as unbiased while
    each bucket is unanimous.  A variant is biased if ANY bucket with enough
    cells is biased.  Buckets below detect_systematic_bias's 4-cell minimum
    are recorded and skipped -- they cannot flag, and they cannot mask.
    """
    buckets: Dict[str, Any] = {}
    any_biased = False
    for case in CASES:
        rows = [c for c in cells if c.get("case") == case]
        diffs = [c.get("signed_diff_pct") for c in rows]
        usable = [d for d in diffs if d is not None and math.isfinite(float(d))]
        if len(usable) < 4:
            buckets[case] = {"n_cells": len(usable), "skipped": True}
            continue
        biased, info = detect_systematic_bias(usable, tol_abs_pct)
        buckets[case] = {**info, "skipped": False, "biased": bool(biased)}
        any_biased = any_biased or bool(biased)
    return any_biased, {"buckets": buckets, "pooled_not_used": True}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_scope.py -n0 -q -k bias`
Expected: PASS, 3 tests.

- [ ] **Step 5: Wire it into `decide_route`**

In `decide_route`, replace

```python
biased, bias_info = detect_systematic_bias(
    [c.get("signed_diff_pct") for c in medium_cells], tol_abs_pct
)
```

with `biased, bias_info = detect_systematic_bias_bucketed(medium_cells, tol_abs_pct)`, and update the rationale string to name the offending bucket(s).

- [ ] **Step 6: Run the gate's existing unit tests**

Run: `.venv/bin/python -m pytest test/mo_volmodels -n0 -q`
Expected: PASS. If a pre-existing test asserts the pooled behaviour, update it — the pooled path is now a helper, not the gate criterion.

- [ ] **Step 7: Commit**

```bash
git add example/mo_volmodels/11_pde_convergence_gate.py test/mo_volmodels/test_gate_scope.py
git commit -m "fix(mo): evaluate G2 bias within maturity buckets, not pooled"
```

---

## Task 6: Promote delta from secondary evidence to a gate criterion

**Files:**
- Modify: `example/mo_volmodels/11_pde_convergence_gate.py:950-995` (delta rows), `:590` (`decide_route`)
- Test: `test/mo_volmodels/test_gate_scope.py`

**Interfaces:**
- Produces:
  - `delta_quantum_per_unit(s0: float, notional: float = NOTIONAL, multiplier: float = FUTURES_MULTIPLIER) -> float` — per-unit delta moved by **one** IM futures contract.
  - `delta_cell_passed(abs_diff: float, s0: float) -> bool` — `abs_diff <= 0.5 * quantum`.
  - `detect_delta_bias(rows) -> tuple[bool, dict]` — mean **signed** delta difference, in contracts, against a 0.1-contract bound.

**Why:** §5.3. This is a hedging study; P&L is driven by model-consistent delta, not PV. The original G2 gated PV only. The tolerance is derived from what the hedge can physically express rather than picked as a percentage.

**The unit conversion is the whole trick — get it wrong and the threshold is meaningless.** The gate prices a 1-unit product, so its delta is per index unit. The backtest holds `contract_multiplier = NOTIONAL / s0` index units, and one IM contract covers `FUTURES_MULTIPLIER = 200` index points. So one contract moves

```
per-unit delta quantum = 200 / contract_multiplier = 200 * s0 / NOTIONAL
```

Worked for 2023-05-04 (`s0 = 6733.97`, `contract_multiplier = 7425.5`): `200 / 7425.5 = 0.02694` per-unit delta, i.e. 13,468 CNY per 1% spot move; half a contract is `0.01347`.

`s0` differs per inception — **4,532.52** (2024-09-02) to **6,733.97** (2023-05-04) — so the quantum spans `0.01813` to `0.02694`, a factor of 1.49. The threshold is therefore computed **per date** and never hardcoded.

- [ ] **Step 1: Write the failing test**

```python
def test_delta_quantum_matches_the_worked_example():
    """Spec §5.3, 2023-05-04: s0=6733.97, multiplier=7425.5 -> 0.02694."""
    gate = _load_gate()
    q = gate.delta_quantum_per_unit(6733.97, notional=50_000_000.0)
    assert q == pytest.approx(0.026936, abs=1e-6)


def test_delta_quantum_scales_with_inception_spot():
    """A fixed threshold would be 1.49x wrong across the 27 inceptions."""
    gate = _load_gate()
    lo = gate.delta_quantum_per_unit(4532.52, notional=50_000_000.0)
    hi = gate.delta_quantum_per_unit(6733.97, notional=50_000_000.0)
    assert hi / lo == pytest.approx(6733.97 / 4532.52, rel=1e-9)


def test_disagreement_under_half_a_contract_passes():
    gate = _load_gate()
    q = gate.delta_quantum_per_unit(6733.97)
    assert gate.delta_cell_passed(0.49 * q, 6733.97) is True
    assert gate.delta_cell_passed(0.51 * q, 6733.97) is False


def test_small_one_sided_delta_bias_is_caught_even_though_each_cell_passes():
    """Rounding absorbs a single cell; 700 rebalances accumulate the mean."""
    gate = _load_gate()
    s0 = 6733.97
    q = gate.delta_quantum_per_unit(s0)
    rows = [{"s0": s0, "signed_diff": 0.2 * q} for _ in range(8)]
    assert all(gate.delta_cell_passed(abs(r["signed_diff"]), s0) for r in rows)
    biased, info = gate.detect_delta_bias(rows)
    assert biased is True
    assert info["mean_signed_contracts"] == pytest.approx(0.2, rel=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_scope.py -n0 -q -k delta`
Expected: FAIL — `AttributeError: ... 'delta_quantum_per_unit'`.

- [ ] **Step 3: Implement**

```python
# Hedge-derived delta tolerances (spec §5.3).  The gate prices a 1-unit
# product; the backtest holds NOTIONAL/s0 index units.  One IM contract
# covers FUTURES_MULTIPLIER index points, so it moves
# FUTURES_MULTIPLIER * s0 / NOTIONAL of per-unit delta.  s0 varies 1.49x
# across the 27 inceptions, so this is computed per date, never fixed.
FUTURES_MULTIPLIER = 200.0
STUDY_NOTIONAL = 50_000_000.0
DELTA_CELL_CONTRACTS = 0.5    # rounding provably absorbs less than this
DELTA_BIAS_CONTRACTS = 0.1    # accumulates over ~700 rebalances


def delta_quantum_per_unit(
    s0: float,
    notional: float = STUDY_NOTIONAL,
    multiplier: float = FUTURES_MULTIPLIER,
) -> float:
    """Per-unit delta moved by exactly one IM futures contract."""
    return float(multiplier) * float(s0) / float(notional)


def delta_cell_passed(abs_diff: float, s0: float, **kw) -> bool:
    return abs(float(abs_diff)) <= DELTA_CELL_CONTRACTS * delta_quantum_per_unit(s0, **kw)


def detect_delta_bias(rows: Sequence[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    """Mean SIGNED delta difference in contracts, against the bias bound.

    Signed, not absolute: contract rounding kills symmetric noise but lets a
    one-sided bias accumulate over the rebalance schedule.
    """
    contracts = [
        float(r["signed_diff"]) / delta_quantum_per_unit(float(r["s0"]))
        for r in rows
        if r.get("signed_diff") is not None and r.get("s0")
    ]
    if not contracts:
        return False, {"n_rows": 0, "mean_signed_contracts": None}
    mean_signed = float(np.mean(contracts))
    return abs(mean_signed) > DELTA_BIAS_CONTRACTS, {
        "n_rows": len(contracts),
        "mean_signed_contracts": mean_signed,
        "max_abs_contracts": max(abs(c) for c in contracts),
        "bound_contracts": DELTA_BIAS_CONTRACTS,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_scope.py -n0 -q -k delta`
Expected: PASS, 4 tests.

- [ ] **Step 5: Record `signed_diff`, `s0` and the verdict on every delta row**

The existing row (`:955`) stores only `abs_diff`. `detect_delta_bias` needs the **sign**, and the tolerance needs `s0`.

First rename the row's keys `delta_pde` → `delta_production` and `delta_mc` → `delta_reference`: after Task 4 the production engine is not always a PDE, and the reference not always MC. Update every reader (`grep -n "delta_pde\|delta_mc" example/mo_volmodels/`), including the report/printing code. Then extend the row:

```python
delta_row["s0"] = float(s0_inception)
if delta_row["delta_production"] is not None and delta_row["delta_reference"] is not None:
    signed = delta_row["delta_production"] - delta_row["delta_reference"]
    delta_row["signed_diff"] = signed
    delta_row["abs_diff"] = abs(signed)
    delta_row["diff_contracts"] = signed / delta_quantum_per_unit(float(s0_inception))
    delta_row["passed"] = delta_cell_passed(abs(signed), float(s0_inception))
else:
    delta_row["signed_diff"] = None
    delta_row["diff_contracts"] = None
    delta_row["passed"] = False        # fail closed
```

The `_bumped_pde_delta` / `_bumped_mc_delta` helpers (`:895`, and its MC twin) must likewise be driven off `GATE_PAIRS[variant]` rather than assuming PDE-vs-MC. A deterministic-reference variant uses the same central bump on both engines; there is no CRN to arrange.

- [ ] **Step 6: Add delta admission to `decide_route`**

`decide_route` currently takes only PV cells. Add a `delta_rows` parameter and two more conjuncts to the route decision:

```python
delta_pass = all(r.get("passed") is True for r in delta_rows) if delta_rows else False
delta_biased, delta_info = detect_delta_bias(delta_rows)
route = "pde" if (medium_pass and not biased and fine_pass and drift_ok
                  and delta_pass and not delta_biased) else "mc"
```

Add a `reasons.append(...)` for each new failure mode naming the offending dates, and surface `delta_pass` / `delta_biased` / `delta_info` in the returned dict and in `gate_decision.json`. Empty `delta_rows` must not admit — mirror the existing "refusing to admit PDE on empty evidence" behaviour.

- [ ] **Step 7: Run the full mo test suite**

Run: `.venv/bin/python -m pytest test/mo_volmodels -n0 -q`
Expected: PASS. Update any test that constructs `decide_route` positionally.

- [ ] **Step 8: Commit**

```bash
git add example/mo_volmodels/11_pde_convergence_gate.py test/mo_volmodels/test_gate_scope.py
git commit -m "feat(mo): gate G2 on delta agreement in IM contracts, not PV alone"
```

---

## Task 7: Condition the G2 verdict on the Feller regime

**Files:**
- Modify: `example/mo_volmodels/11_pde_convergence_gate.py:1132` (`_calibration_record`), `:590` (`decide_route`)
- Test: `test/mo_volmodels/test_gate_scope.py`

**Interfaces:**
- Produces: `feller_bucket(ratio: float | None) -> str` returning `"violated"` (< 0.5), `"boundary"` (0.5 … 10), `"degenerate"` (> 10) or `"unknown"`; and per-bucket route detail in `decide_route`'s output under `feller_buckets`.

**Why, and why these thresholds:** §7A.4(3) and §7A.11. A uniform verdict would average a 0.03% regime with a 2.5% one. The cut points come from the measurements, not from taste:

- **0.5** — §7A.11's measured failure boundary. Unconstrained fits fail the gate at ratio ≤ 0.50 and pass above it; 16.4% of the cohort sits below 0.50.
- **10** — the σ-collapse marker from §7A.10(3). 6.6% of enforced fits satisfy Feller by driving σ to its lower bound (ratios up to 1.7e5), giving a deterministic-variance model that fails the gate under *both* calibration policies.

With `enforce_feller=True` shipped, **80% of dates land in `boundary` at ratio ≈ 1.0**. `violated` should be empty in the re-run; if it is not, the enforcement did not take and that is a finding.

`feller_ratio` is already in the calibration record (landed with §7A.4) and `_calibration_record` passes the record through verbatim, so no plumbing is needed to obtain it — only to key on it.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("ratio,expected", [
    (0.197, "violated"), (0.493, "violated"), (0.5, "boundary"),
    (1.0001, "boundary"), (7.1945, "boundary"), (10.0, "boundary"),
    (172224.1, "degenerate"), (None, "unknown"),
])
def test_feller_buckets_use_the_measured_cut_points(ratio, expected):
    assert _load_gate().feller_bucket(ratio) == expected


def test_route_records_a_verdict_per_feller_bucket():
    """A pooled verdict averages a 0.03% regime with a 2.5% one (§7A.3)."""
    gate = _load_gate()
    cells = (
        [{"date": "2024-01-12", "case": "full", "signed_diff_pct": +0.58,
          "passed": False, "feller_ratio": 0.197, "pde_price": 1.0, "notional": 1.0}]
        + [{"date": f"d{i}", "case": "full", "signed_diff_pct": +0.10,
            "passed": True, "feller_ratio": 1.0001, "pde_price": 1.0, "notional": 1.0}
           for i in range(6)]
    )
    out = gate.decide_route(cells, cells, delta_rows=[])
    assert out["feller_buckets"]["violated"]["n_cells"] == 1
    assert out["feller_buckets"]["violated"]["n_passed"] == 0
    assert out["feller_buckets"]["boundary"]["n_passed"] == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_scope.py -n0 -q -k feller`
Expected: FAIL — `AttributeError: ... 'feller_bucket'`.

- [ ] **Step 3: Implement**

```python
# Feller-regime buckets.  Cut points are measured, not chosen:
#   0.5  -- §7A.11's failure boundary: unconstrained fits fail the gate at
#           ratio <= 0.50 and pass above it (16.4% of the cohort below).
#   10   -- §7A.10(3)'s sigma-collapse marker: 6.6% of enforced fits satisfy
#           Feller by driving sigma to its bound, giving a
#           deterministic-variance model that fails under BOTH policies.
FELLER_VIOLATED_BELOW = 0.5
FELLER_DEGENERATE_ABOVE = 10.0


def feller_bucket(ratio: Optional[float]) -> str:
    if ratio is None or not math.isfinite(float(ratio)):
        return "unknown"
    ratio = float(ratio)
    if ratio < FELLER_VIOLATED_BELOW:
        return "violated"
    if ratio > FELLER_DEGENERATE_ABOVE:
        return "degenerate"
    return "boundary"
```

In `process_date`, copy `feller_ratio` from the Heston calibration record onto every `heston` / `heston_slv` cell and delta row. Non-Heston variants carry `None` → `"unknown"`.

In `decide_route`, group the medium cells by `feller_bucket(c.get("feller_ratio"))` and emit per-bucket `n_cells` / `n_passed` / `max_abs_diff_pct` under `feller_buckets`. **The route stays all-cells-must-pass** — bucketing reports *where* a variant fails, it does not license failure in an unpopular regime.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_scope.py -n0 -q -k feller`
Expected: PASS, 9 tests.

- [ ] **Step 5: Run the full mo test suite**

Run: `.venv/bin/python -m pytest test/mo_volmodels -n0 -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add example/mo_volmodels/11_pde_convergence_gate.py test/mo_volmodels/test_gate_scope.py
git commit -m "feat(mo): report the G2 verdict per Feller regime, on measured cut points"
```

---

## Task 8: Run G2 and emit the decision

**Files:**
- Modify: `docs/superpowers/specs/2026-07-30-snowball-volmodel-backtest-040-rebaseline-design.md` (§5 results)

**Interfaces:**
- Consumes: everything above.
- Produces: `output/pde_convergence_gate/gate_decision.json` with six variant entries and a fresh `evidence_sha256`; stage 12 reads it via `load_gate_routing`.

- [ ] **Step 1: Confirm the whole suite is green before spending compute**

Run: `.venv/bin/python -m pytest -q`
Expected: the four known pre-existing failures only — `test_adi_core_tau_exactness.py` (×3) and `test_snowball_pde_knocked_in_grid.py` (×1). Both files are another session's untracked WIP and reproduce against a pristine `git archive HEAD` tree. **Any other failure blocks this task.**

- [ ] **Step 1B: Quiesce the scheduler and record the tree state**

G2 itself is already pinned by construction — it runs the eight fixed `DEFAULT_SAMPLE_DATES` (`11_pde_convergence_gate.py:94`), all at or before 2026-07-15, so a scheduler tick cannot change its cell set. The reason to unload the job is **CPU contention**: this task spends hours across 4 workers, and an 18:30 or 20:30 tick launches its own LV + Heston + SLV calibrations on the same machine, which distorts every timing figure §7.2 depends on and slows the run.

```bash
launchctl bootout gui/$(id -u)/com.quantark.mo-daily-calibration 2>/dev/null \
  || echo "already unloaded"
launchctl list | grep mo-daily-calibration || echo "confirmed: not loaded"
```

```bash
launchctl bootout gui/$(id -u)/com.quantark.mo-daily-calibration 2>/dev/null \
  || echo "already unloaded"
launchctl list | grep mo-daily-calibration || echo "confirmed: not loaded"
```

Re-enable afterwards with `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.quantark.mo-daily-calibration.plist` — or `install_daily_scheduler.py install`. Note that a missed weekday is recoverable: the pipeline is resumable and its caches are persistent.

Then record what the evidence is keyed to, because §7A.12's third risk row applies — the daily-pipeline workstream is uncommitted:

```bash
git rev-parse HEAD > output/pde_convergence_gate/tree_state.txt
git status --porcelain >> output/pde_convergence_gate/tree_state.txt
```

If that second command prints modifications under `quantark/`, either commit them first or state in §5 of the spec that the G2 evidence was produced against an uncommitted tree. Do not leave it implicit.

- [ ] **Step 1C: Seed the calibration cache**

```bash
.venv/bin/python example/mo_volmodels/seed_calibration_cache.py \
  --dst output/pde_convergence_gate/calibration_cache
```
Expected: `copied 720` on a cold cache. This is free time back on every gate cell whose date falls in 2025-07-31 → 2026-07-31 (Task 0B).

- [ ] **Step 2: Update `GateRouting` for six variants**

`GateRouting.solver_for` (`12_snowball_volmodel_backtest.py:238`) short-circuits `bsm` and `localvol` to `"pde"` as "outside the 2D-ADI gate's scope". After Task 4 they are inside it. Remove the short-circuit and let every variant read its route from the decision, keeping the fail-closed raise for a missing or unusable route.

Add a test in `test/mo_volmodels/test_gate_scope.py`:

```python
def test_routing_no_longer_short_circuits_one_d_variants():
    s12 = _load_stage12()
    routing = s12.GateRouting("p", None, {"localvol": "mc"}, {})
    assert routing.solver_for("localvol") == "mc"
```

Run it (expect FAIL: returns `"pde"`), then remove the short-circuit, then re-run (expect PASS). Commit.

- [ ] **Step 3: Run the gate for real**

Background; the QE-M reference at 4 substeps quadruples the reference MC time grid, and §7A.6 measured MC at 40–50 s/case against PDE at ~9 s, so **budget several hours**:

```bash
.venv/bin/python example/mo_volmodels/11_pde_convergence_gate.py \
  --workers 4 --out-dir output/pde_convergence_gate 2>&1 | tee output/g2_run.log
```

- [ ] **Step 4: Read the decision**

```bash
.venv/bin/python -c "
import json
d = json.load(open('output/pde_convergence_gate/gate_decision.json'))
print('evidence', d['evidence_sha256'][:16])
print('mc_reference', d['mc_reference']['scheme'])
for v, row in sorted(d['variants'].items()):
    print(f\"{v:16} route={row['route']:4} {row['rationale'][:90]}\")
"
```

Check three things specifically:
- `mc_reference.scheme == "QUADEXP_M"` — confirms the §7A.4(4) upgrade is in the evidence.
- `feller_buckets.violated` is empty for `heston` / `heston_slv`. If it is not, `enforce_feller` did not take; **stop and investigate** rather than accepting the verdict.
- Whether `biased` is `false` by a *narrow* median. §7A.11 measured sign fraction 1.0 with the enforced median at 0.113% against a 0.125% threshold — a 0.012-point margin. Record the actual margin; a narrow pass is "biased but small", not "clean".

- [ ] **Step 5: Stamp the calibration policy the evidence covers**

The config now carries a temporal-smoothing option (spec §7A.12) that no gate has evaluated. The decision must say which policy it certifies, so a future reader cannot mistake λ=0 evidence for coverage of λ>0. Add to the decision payload where `mc_reference` is assembled:

```python
        "calibration_policy": {
            "heston_preset": "mo_frozen",
            "enforce_feller": True,
            "heston_temporal_regularization": 0.0,
            "slv_heston_override": None,
            "note": (
                "Independent daily calibration. This gate does NOT cover "
                "--temporal-smoothing (heston_temporal_regularization > 0) "
                "or an explicit slv_heston_override; enabling either is a "
                "re-gate trigger (spec 7A.12)."
            ),
        },
```

Assert it round-trips:

```bash
.venv/bin/python -c "
import json
d = json.load(open('output/pde_convergence_gate/gate_decision.json'))
p = d['calibration_policy']
assert p['heston_temporal_regularization'] == 0.0, p
assert p['enforce_feller'] is True, p
print('calibration policy recorded:', p['heston_preset'])
"
```

- [ ] **Step 6: Record the outcome in the spec**

Add a §5.5 with: the per-variant routes, the per-Feller-bucket table, the delta admission results in contracts, the bias margin from Step 4, and `evidence_sha256`. State plainly which variants were admitted to PDE and which fell back to MC — including whether §7A.8's recorded *prediction* (both 2D variants admitted at 200×60×`ceil(400·T)`, on delta stability more than speed) held or was falsified.

State the cohort pin (`COHORT_ASOF = 2026-07-31`, 766 admitted surfaces, 27 inceptions) and the tree state from Step 1B alongside it, so the run is reproducible.

- [ ] **Step 7: Restore the scheduler**

```bash
.venv/bin/python example/mo_volmodels/install_daily_scheduler.py status \
  || launchctl bootstrap gui/$(id -u) \
       ~/Library/LaunchAgents/com.quantark.mo-daily-calibration.plist
launchctl list | grep mo-daily-calibration
```

Then run the pipeline once by hand to clear whatever backlog accrued while it was unloaded, and confirm it lands on `current`:

```bash
.venv/bin/python example/mo_volmodels/14_daily_calibration_pipeline.py run
.venv/bin/python example/mo_volmodels/14_daily_calibration_pipeline.py status --json \
  | .venv/bin/python -c "import json,sys; print(json.load(sys.stdin)['overall_status'])"
```

Expected: `current`. A `source_pending` is fine (CFFEX has not published yet) — anything else needs the recovery table in `DAILY_PIPELINE.md`.

- [ ] **Step 8: Commit**

```bash
git add -f docs/superpowers/specs/2026-07-30-snowball-volmodel-backtest-040-rebaseline-design.md
git commit -m "docs(mo): record the re-scoped G2 outcome and per-regime evidence"
```

---

## Task 9: Gate G2 must price each variant's own `surface_vol_mode`

*Added 2026-08-03, after Task 8's G2 run, with the owner's approval at the
reassessment point. Not part of the original eight.*

**Why this exists.** Reading the Task 8 evidence turned up that `ts_bsm` and
`flat_bsm` were **bitwise identical** — all 15 PV cells and every delta agreed
to 16 significant digits. `build_pricing_env` handed every variant
`artifact.grid_vol_surface()` and never read `surface_vol_mode`, so the gate ran
four distinct computations and reported six rows. `ts_bsm`'s `route=pde` rested
on no independent evidence. `test_gate_covers_every_study_variant` asserts set
equality of variant *names*, so it could not see this.

Full detail, including why the direction was conservative, is in spec §5.5.

**Files:**
- Modify: `example/mo_volmodels/11_pde_convergence_gate.py`
- Test: `test/mo_volmodels/test_gate_scope.py`,
  `test/mo_volmodels/test_pde_convergence_gate.py` (the new required
  `GatePair` field and the `_evaluate_case` `env`→`envs` rename break four
  pre-existing tests there)

**Interfaces:**
- Consumes: `GATE_PAIRS` (Task 4), `VariantSpec.surface_vol_mode` (stage 12)
- Produces: `GatePair.surface_vol_mode: str` (no default);
  `build_pricing_env(artifact, rate, *, surface_vol_mode, remaining_maturity_years=None)`;
  `_envs_by_mode(artifact, rate, remaining_maturity_years) -> Dict[str, PricingEnvironment]`

**The contract to mirror** is `ProductReplay._vol_and_dividend`
(`quantark/backtest/replay/product_replay.py:225-238`) — `term_structure` →
`artifact.term_structure_vol_surface()`; `full_grid` →
`artifact.grid_vol_surface()`; `flat_atm_remaining` → the ATM term structure
sampled at the remaining maturity, wrapped in `FlatVolSurface`. `div_yield` is
mode-independent.

Build **one env per distinct mode** (three), not one per variant (six) — the
three `full_grid` variants share a single env. `flat_atm_remaining` needs the
remaining maturity: `terms.maturity_years` for the full case, the decayed
terms' own already-adjusted `maturity_years` for the decayed case.

Both error paths fail closed with `ValidationError` — unknown mode, and
`flat_atm_remaining` with no maturity. No default for `surface_vol_mode`
anywhere: a default is how the next variant silently gets the wrong surface.
This follows the archetype Tasks 4–7 each hit, where an error path landed on a
value that read as success.

**Status: DONE** — commit `7109868`. 38/38 in `test_gate_scope.py`, 30/30 in
`test_pde_convergence_gate.py`. Mutation-checked: forcing `grid_vol_surface()`
for every mode is caught by four behavioural tests.

**Measured consequence.** `flat_bsm` now gets a `FlatVolSurface` at 0.264476;
`ts_bsm` a `TermStructureVolSurface` reading 0.279979 at T=0.5. They are no
longer the same computation. Note the two still agree from T=1 onward — CSI
1000 options are short-dated and the artifact clamps flat total variance beyond
the last listed expiry, so a 3-year snowball reads a term structure the market
barely prices. That is a study finding about `ts_bsm`'s discriminating power,
not a defect.

---

## Out of scope for this plan

Deliberately deferred; each needs its own plan.

- **G5 pre-flight grid sweep** (§9) — build grids only, no solve, for every operating point before the fleet. `fdf3a70` made under-resolution a fail-closed `ValidationError`, and `test_adi_core_tau_exactness.py`'s failures show `n_x=60` at T≥2 already trips it.
  - **DELIVERED 2026-08-26** as `example/mo_volmodels/11c_grid_preflight.py`: 14,084 operating points, 0 under-resolved. QUAD routes and the 2-D variance axis remain uncovered and are declared in the artifact's `scope.not_covered`.
- **Task 6.1 timing run** (§7.3) — the fleet total is set by measurement, never extrapolated from single solves.
- **G3 accounting sanity** (§11) on one inception.
- **Stage 13 σ-collapse handling** — §7A.10(3)'s 50 dates must be flagged or excluded, never averaged into a `heston` result.
  - **DELIVERED 2026-08-26.** `calibration_quality` screened `feller_satisfied`,
    which `enforce_feller=True` makes True by construction — 257 of 257 fits in
    the daily pool — so the metric could only ever report "clean". It now ranks
    the Feller *ratio* on Gate G2's measured cut points and emits
    `feller_buckets` / `n_sigma_collapse` / `sigma_collapse_fraction`, plus
    `n_enforcement_breaches` so the enforcement premise is itself checkable.
    Against the real pool this flags 3 dates (1.17%) — 2026-01-12, 2026-01-23,
    2026-07-16 — at ratios 2.3e5 / 7.9e3 / 1.7e5 with σ pinned at its 0.001
    lower bound. `heston_slv` records nest their Heston fit without a ratio, so
    the screen derives it from the nested parameters; reading only the top level
    left σ-collapse invisible for one of the two certified 2-D variants.
    Posture is **flag**, not exclude: §5.9 records that exclusion was the interim
    mitigation for a defect believed unfixable, and the monotone v-transport
    stencil (`v_drift_scheme: "auto"`, shipped `908588c`) fixed it.
- **Deriving `_clone_engine` from the constructor signature** (§7A.10) — the hand-transcribed kwargs list is a standing silent-mispricing hazard.
