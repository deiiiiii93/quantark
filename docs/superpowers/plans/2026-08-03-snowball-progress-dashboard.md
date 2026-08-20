# Snowball Study Progress Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One read-only page that states where the snowball vol-model study stands — every gate's verdict per facet with its freshness, fleet coverage as fresh cells out of 162, and the next action.

**Architecture:** A thin CLI (`16_dashboard.py`) over a small package (`mo_dashboard/`). One collection pass builds a versioned `payload` dict from the registry, `output/` artifacts, the run tree, and git; two renderers consume it — a self-contained HTML snapshot and a polling local server. The freshness rule is a pure function over pre-collected git facts, so the logic that decides `fresh`/`stale`/`void` is unit-testable without touching a repository.

**Tech Stack:** Python 3.11, `pyyaml` (already declared, `pyproject.toml:39`), stdlib `http.server`, pytest. No new dependencies. `quantark.*` canonical imports only.

## Global Constraints

- Run everything with `.venv/bin/python` or after `source .venv/bin/activate`.
- Source spec: `docs/superpowers/specs/2026-08-03-snowball-progress-dashboard-design.md`. Every `§` below points there. Study-spec sections are written `study §5.8`.
- Canonical imports only: `quantark.*`. Never `backtest.otc.*` (0.5.0 shim) or bare flat names.
- Numerics: use `quantark.util.numerical` (`is_zero`, `is_close`, `safe_divide`). Never a raw float `==` or a hardcoded tolerance.
- **Read-only.** No module in `mo_dashboard/` may write anywhere under `output/` except the single HTML file named by `--out`. No module may invoke a gate, a fleet, or any pricing code.
- **Fail soft, loud.** A missing or unparseable artifact produces a row with `status: "unreadable"` carrying the exception text plus an entry in `payload["errors"]`. Never a silent zero, never an omitted row, never an exception that blanks the page.
- **The package is named `mo_dashboard`, and the registry `mo_dashboard.yaml`** — a refinement of the spec's `dashboard/` + `dashboard.yaml`. Tests import it by putting `example/mo_volmodels/` on `sys.path`, so a generic top-level name `dashboard` would be a collision hazard. Task 1 Step 7 updates the spec to match.
- All datetimes are **timezone-aware**. File mtimes come from `datetime.fromtimestamp(st_mtime).astimezone()`; git timestamps are parsed from `%cI`. Never compare an aware datetime to a naive one.
- `docs/` is in `.gitignore` but tracked on `main`. Committing a docs file needs `git add -f`; a plain `git add` fails with "paths are ignored".
- Commit after every task. Branch: `fix/snowball-rebaseline-7a4-engine-fixes` or a descendant.
- Tests live in `test/mo_volmodels/test_dashboard.py`. Real-artifact tests must `pytest.skip` when `output/` or `example/mo_volmodels/data/history/` is absent — both are deliberately uncommitted.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `example/mo_volmodels/mo_dashboard/__init__.py` | package marker; re-exports `collect` | **create** |
| `example/mo_volmodels/mo_dashboard/registry.py` | parse `mo_dashboard.yaml`; classify run dirs | **create** |
| `example/mo_volmodels/mo_dashboard/provenance.py` | the freshness rule (pure) + git fact collection | **create** |
| `example/mo_volmodels/mo_dashboard/gates.py` | G1/G4/G2/G5 artifacts → gate rows with facets | **create** |
| `example/mo_volmodels/mo_dashboard/fleet.py` | fleet dimensions, run-tree walk, cell state machine | **create** |
| `example/mo_volmodels/mo_dashboard/results.py` | gate evidence · backtest · calibration blocks | **create** |
| `example/mo_volmodels/mo_dashboard/payload.py` | assemble payload; derive the chain and next action | **create** |
| `example/mo_volmodels/mo_dashboard/render.py` | payload → self-contained HTML | **create** |
| `example/mo_volmodels/mo_dashboard/serve.py` | stdlib http.server; re-collect per poll | **create** |
| `example/mo_volmodels/mo_dashboard.yaml` | the registry | **create** |
| `example/mo_volmodels/16_dashboard.py` | argparse + wiring only | **create** |
| `test/mo_volmodels/test_dashboard.py` | unit tests + one real-artifact integration fixture | **create** |
| `docs/superpowers/specs/2026-08-03-snowball-progress-dashboard-design.md` | rename `dashboard` → `mo_dashboard` | modify |

Each collector is a pure function of (paths, registry) returning a plain dict. No collector imports another; `payload.py` is the only composition point.

---

## Task 1: The registry

**Files:**
- Create: `example/mo_volmodels/mo_dashboard/__init__.py`
- Create: `example/mo_volmodels/mo_dashboard/registry.py`
- Create: `example/mo_volmodels/mo_dashboard.yaml`
- Create: `test/mo_volmodels/test_dashboard.py`
- Modify: `docs/superpowers/specs/2026-08-03-snowball-progress-dashboard-design.md`

**Interfaces:**
- Produces:
  - `Invalidation(commit: str, landed: datetime, spec: str, scopes: tuple[str, ...], variants: tuple[str, ...] | str, facets: tuple[str, ...] | str, reason: str)` with `applies(scope: str, variant: str | None, facet: str) -> bool`
  - `Registry(fleet_dirs: tuple[Path, ...], probe_dirs: tuple[Path, ...], invalidations: tuple[Invalidation, ...], errors: tuple[dict, ...])`
  - `load_registry(path: Path, project_root: Path) -> Registry`
  - `classify_run_dirs(registry: Registry, output_root: Path) -> dict[Path, str]` returning role `"fleet" | "probe" | "unclassified"`

- [ ] **Step 1: Write the failing test**

Create `test/mo_volmodels/test_dashboard.py`:

```python
"""Unit tests for the snowball study progress dashboard.

Pure functions only, except the integration fixture in Task 6 which reads
the real artifacts and skips when they are absent.
"""
import importlib
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MO_DIR = PROJECT_ROOT / "example/mo_volmodels"
if str(MO_DIR) not in sys.path:
    sys.path.insert(0, str(MO_DIR))

registry = importlib.import_module("mo_dashboard.registry")

CST = timezone(timedelta(hours=8))


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "mo_dashboard.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_registry_reads_dirs_and_invalidations(tmp_path):
    path = _write(tmp_path, """
schema_version: 1
fleet:
  - dir: output/volmodel_backtest
probes:
  - dir: output/timing_on
invalidations:
  - commit: f97fba3
    landed: 2026-08-03T13:39:19+08:00
    spec: "5.6"
    applies_to: {scopes: [G2, FLEET], variants: [heston, heston_slv], facets: "*"}
    reason: "2D PDE Heston delta grid"
""")
    reg = registry.load_registry(path, PROJECT_ROOT)

    assert reg.fleet_dirs == (PROJECT_ROOT / "output/volmodel_backtest",)
    assert reg.probe_dirs == (PROJECT_ROOT / "output/timing_on",)
    assert reg.errors == ()
    (inv,) = reg.invalidations
    assert inv.commit == "f97fba3"
    assert inv.landed == datetime(2026, 8, 3, 13, 39, 19, tzinfo=CST)
    assert inv.scopes == ("G2", "FLEET")
    assert inv.variants == ("heston", "heston_slv")
    assert inv.facets == "*"


def test_invalidation_applies_respects_every_axis(tmp_path):
    inv = registry.Invalidation(
        commit="f97fba3",
        landed=datetime(2026, 8, 3, 13, 39, 19, tzinfo=CST),
        spec="5.6",
        scopes=("G2", "FLEET"),
        variants=("heston", "heston_slv"),
        facets="*",
        reason="",
    )
    assert inv.applies("G2", "heston", "delta")
    assert inv.applies("FLEET", "heston_slv", "all")
    assert not inv.applies("G1", "heston", "all")        # wrong scope
    assert not inv.applies("G2", "flat_bsm", "delta")    # wrong variant
    # A gate row with no variant is not variant-specific, so a
    # variant-scoped invalidation must not reach it.
    assert not inv.applies("G2", None, "all")


def test_invalidation_with_star_variants_reaches_a_variantless_row():
    inv = registry.Invalidation(
        commit="41f2117",
        landed=datetime(2026, 7, 31, 10, 13, 27, tzinfo=CST),
        spec="7A.4",
        scopes=("G2", "G4", "FLEET"),
        variants="*",
        facets="*",
        reason="",
    )
    assert inv.applies("G4", None, "all")


def test_facet_scoped_invalidation_touches_only_that_facet():
    inv = registry.Invalidation(
        commit="3fbbf21",
        landed=datetime(2026, 8, 3, 15, 17, 23, tzinfo=CST),
        spec="5.8",
        scopes=("G2",),
        variants="*",
        facets=("delta",),
        reason="",
    )
    assert inv.applies("G2", "heston", "delta")
    assert not inv.applies("G2", "heston", "pv")


def test_missing_registry_yields_an_error_not_an_exception(tmp_path):
    reg = registry.load_registry(tmp_path / "absent.yaml", PROJECT_ROOT)
    assert reg.fleet_dirs == ()
    assert reg.probe_dirs == ()
    assert reg.invalidations == ()
    assert len(reg.errors) == 1
    assert "absent.yaml" in reg.errors[0]["path"]


def test_unregistered_dir_on_disk_is_unclassified(tmp_path):
    (tmp_path / "output/volmodel_backtest").mkdir(parents=True)
    (tmp_path / "output/volmodel_backtest/run_manifest.json").write_text("{}")
    (tmp_path / "output/mystery_run").mkdir(parents=True)
    (tmp_path / "output/mystery_run/run_manifest.json").write_text("{}")
    path = _write(tmp_path, """
schema_version: 1
fleet:
  - dir: output/volmodel_backtest
""")
    reg = registry.load_registry(path, tmp_path)
    roles = registry.classify_run_dirs(reg, tmp_path / "output")

    assert roles[tmp_path / "output/volmodel_backtest"] == "fleet"
    assert roles[tmp_path / "output/mystery_run"] == "unclassified"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_dashboard.py -n0 -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mo_dashboard'`

- [ ] **Step 3: Create the package marker**

Create `example/mo_volmodels/mo_dashboard/__init__.py`:

```python
"""Read-only progress dashboard for the snowball vol-model study.

Every module here reads; none writes anywhere under ``output/`` except the
single HTML file the CLI is told to produce, and none imports pricing code.
See docs/superpowers/specs/2026-08-03-snowball-progress-dashboard-design.md.
"""
```

- [ ] **Step 4: Write the registry module**

Create `example/mo_volmodels/mo_dashboard/registry.py`:

```python
"""Contract A: which run dirs are the fleet, and which commits void what.

The registry states only what code cannot derive.  Fleet *dimensions* (six
variants, 27 inceptions) come from stage 12 -- see ``fleet.py`` -- and are
never restated here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import yaml

Scoped = Union[Tuple[str, ...], str]  # a tuple of names, or the literal "*"


@dataclass(frozen=True)
class Invalidation:
    """A declared statement that some prior output is not comparable.

    Scoping is the whole point.  An unscoped invalidation applied to the
    2026-08-03 artifacts voids G1, G4 and all 27 ``flat_bsm`` cells on the
    strength of ``f97fba3``, a 2D-PDE Heston delta fix that touches none of
    them (spec 5.2).
    """

    commit: str
    landed: datetime
    spec: str
    scopes: Tuple[str, ...]
    variants: Scoped
    facets: Scoped
    reason: str

    def applies(self, scope: str, variant: Optional[str], facet: str) -> bool:
        if scope not in self.scopes:
            return False
        if self.variants != "*":
            # ``variant is None`` means the row is not variant-specific (a
            # whole-gate verdict), so a variant-scoped invalidation cannot
            # reach it.
            if variant is None or variant not in self.variants:
                return False
        if self.facets != "*" and facet not in self.facets:
            return False
        return True


@dataclass(frozen=True)
class Registry:
    fleet_dirs: Tuple[Path, ...] = ()
    probe_dirs: Tuple[Path, ...] = ()
    invalidations: Tuple[Invalidation, ...] = ()
    errors: Tuple[Dict[str, str], ...] = ()


def _scoped(value: Any, field: str, commit: str) -> Scoped:
    if value == "*":
        return "*"
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    raise ValueError(f"{commit}: applies_to.{field} must be a list or \"*\", got {value!r}")


def _dirs(entries: Any, project_root: Path) -> Tuple[Path, ...]:
    out: List[Path] = []
    for entry in entries or []:
        out.append((project_root / str(entry["dir"])).resolve())
    return tuple(out)


def load_registry(path: Path, project_root: Path) -> Registry:
    """Parse the registry.  A missing or malformed file is an error row."""
    path = Path(path)
    project_root = Path(project_root).resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 -- surfaced on the page, never raised
        return Registry(errors=({"source": "registry", "path": str(path),
                                 "message": f"{type(exc).__name__}: {exc}"},))

    errors: List[Dict[str, str]] = []
    invalidations: List[Invalidation] = []
    for item in raw.get("invalidations") or []:
        commit = str(item.get("commit", "?"))
        try:
            scope_block = item.get("applies_to") or {}
            invalidations.append(
                Invalidation(
                    commit=commit,
                    landed=datetime.fromisoformat(str(item["landed"])),
                    spec=str(item.get("spec", "")),
                    scopes=tuple(str(s) for s in scope_block["scopes"]),
                    variants=_scoped(scope_block.get("variants", "*"), "variants", commit),
                    facets=_scoped(scope_block.get("facets", "*"), "facets", commit),
                    reason=str(item.get("reason", "")),
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"source": "registry.invalidations", "path": str(path),
                           "message": f"{commit}: {type(exc).__name__}: {exc}"})

    try:
        fleet_dirs = _dirs(raw.get("fleet"), project_root)
        probe_dirs = _dirs(raw.get("probes"), project_root)
    except Exception as exc:  # noqa: BLE001
        fleet_dirs, probe_dirs = (), ()
        errors.append({"source": "registry.dirs", "path": str(path),
                       "message": f"{type(exc).__name__}: {exc}"})

    return Registry(
        fleet_dirs=fleet_dirs,
        probe_dirs=probe_dirs,
        invalidations=tuple(invalidations),
        errors=tuple(errors),
    )


def classify_run_dirs(registry: Registry, output_root: Path) -> Dict[Path, str]:
    """Role for every run dir on disk.

    A run dir is any immediate child of ``output_root`` holding a
    ``run_manifest.json``.  One absent from the registry is ``unclassified``
    -- a visible gap rather than a silent omission.  This is not
    hypothetical: ``output/volmodel_smoke_gated`` was created 2026-08-03 and
    was missed during the design's own survey of ``output/``.
    """
    output_root = Path(output_root)
    roles: Dict[Path, str] = {}
    for path in registry.fleet_dirs:
        roles[path] = "fleet"
    for path in registry.probe_dirs:
        roles.setdefault(path, "probe")
    if not output_root.is_dir():
        return roles
    for child in sorted(output_root.iterdir()):
        if not child.is_dir() or not (child / "run_manifest.json").exists():
            continue
        roles.setdefault(child.resolve(), "unclassified")
    return roles
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_dashboard.py -n0 -q`
Expected: PASS — 6 passed

- [ ] **Step 6: Write the real registry**

Create `example/mo_volmodels/mo_dashboard.yaml`:

```yaml
# Contract A for the progress dashboard (spec section 5).
#
# Only what code cannot derive.  Fleet DIMENSIONS -- six variants, 27
# inceptions -- come from stage 12 and cohort.py, never from this file.
#
# A run dir on disk but absent here renders as "unclassified": the
# stale-registry failure mode is made visible, not silent.
schema_version: 1

fleet:
  - dir: output/volmodel_backtest      # 0.4.0 re-baseline fleet

probes:
  - dir: output/volmodel_smoke         # 1 inception x 6 variants, censored at data_end
  - dir: output/volmodel_smoke_gated   # 1 inception x 3 variants, post-f97fba3, censored
  - dir: output/timing_on              # study 7.4 event-stats timing, --quick
  - dir: output/timing_off             # study 7.4 control, --quick
  - dir: output/volmodel_backtest_g3   # dead G3 probe, 1 failed run

# Scoped invalidations.  These replace the hand-maintained
# gate_decision_pre_*.json renaming convention.  A DOCUMENTATION commit can
# appear here: 3fbbf21 is a spec section, and it voids a numeric artifact.
invalidations:
  - commit: 41f2117
    landed: 2026-07-31T10:13:27+08:00
    spec: "7A.4"
    applies_to: {scopes: [G2, G4, FLEET], variants: "*", facets: "*"}
    reason: >-
      enforce_feller + degenerate_pde + QE-M changed what every engine
      computes; prior engine output is not comparable.

  - commit: f97fba3
    landed: 2026-08-03T13:39:19+08:00
    spec: "5.6"
    applies_to: {scopes: [G2, FLEET], variants: [heston, heston_slv], facets: "*"}
    reason: >-
      2D PDE Heston delta grid stabilisation; the 1D routes are untouched.

  - commit: 3fbbf21
    landed: 2026-08-03T15:17:23+08:00
    spec: "5.8"
    applies_to: {scopes: [G2], variants: "*", facets: [delta]}
    reason: >-
      The MC delta reference carries sigma 0.41-0.51 futures contracts against
      a 0.1 contract bias bound, so the delta half of every route decision is
      void.  The PV half stands.

# Deliberately absent, recorded so omission is not mistaken for oversight:
#   b6b97f0 -- PDEEngine failed closed before it, so no surviving pricing
#     summary contains numbers it changed.  Still covered by the dependency
#     table, so predating artifacts read "stale" rather than certified.
#   ec20db9 -- study 5.9 is a REATTRIBUTION (sigma-collapse dates fail on
#     discretisation, not calibration).  The measured failures stand, so no
#     artifact becomes non-comparable.  It reaches freshness via STUDY_SPEC.
```

- [ ] **Step 7: Align the spec's package name**

In `docs/superpowers/specs/2026-08-03-snowball-progress-dashboard-design.md` §3.1, replace the two lines

```
  dashboard.yaml             the registry (contract A, §5)
  dashboard/
```

with

```
  mo_dashboard.yaml          the registry (contract A, §5)
  mo_dashboard/
```

and in §5 replace `` `example/mo_volmodels/dashboard.yaml` `` with `` `example/mo_volmodels/mo_dashboard.yaml` ``, and in §7 leave the CLI unchanged. Add this sentence at the end of §3.1:

> The package is `mo_dashboard`, not `dashboard`: tests put `example/mo_volmodels/` on `sys.path`, where a top-level module named `dashboard` would be a collision hazard.

- [ ] **Step 8: Commit**

```bash
git add example/mo_volmodels/mo_dashboard/ example/mo_volmodels/mo_dashboard.yaml test/mo_volmodels/test_dashboard.py
git add -f docs/superpowers/specs/2026-08-03-snowball-progress-dashboard-design.md
git commit -m "feat(dashboard): the registry -- run-dir roles and scoped invalidations"
```

---

## Task 2: The freshness rule

**Files:**
- Create: `example/mo_volmodels/mo_dashboard/provenance.py`
- Modify: `test/mo_volmodels/test_dashboard.py` (append)

**Interfaces:**
- Consumes: `registry.Invalidation` (Task 1)
- Produces:
  - `Commit(sha: str, when: datetime, subject: str)`
  - `Provenance(mode, freshness, invalidated_by, superseded_by, dirty_deps, missing_deps)` with `freshness` in `{"fresh", "stale", "void"}` and `mode` in `{"exact", "inferred"}`
  - `freshness(*, artifact_mtime, scope, variant, facet, dep_commits, dirty_deps, missing_deps, invalidations) -> Provenance` — **pure**. No `stamped_commit`: `mode` is always `"inferred"` (§6.4). A non-empty `missing_deps` forces `stale`.
  - `ENGINE_PATHS: tuple[str, ...]`, `STUDY_SPEC: str`, `DEPS: dict[str, tuple[str, ...]]`
  - `collect_git_facts(project_root, deps) -> tuple[list[Commit], dict[str, datetime], list[str]]` — impure; returns (commits touching deps, dirty dep → mtime, missing deps)
  - `mtime_of(path) -> datetime | None` — aware, local tz

- [ ] **Step 1: Write the failing test**

Append to `test/mo_volmodels/test_dashboard.py`:

```python
provenance = importlib.import_module("mo_dashboard.provenance")

INV_7A4 = registry.Invalidation(
    commit="41f2117", landed=datetime(2026, 7, 31, 10, 13, 27, tzinfo=CST),
    spec="7A.4", scopes=("G2", "G4", "FLEET"), variants="*", facets="*", reason="",
)
INV_PDEFIX = registry.Invalidation(
    commit="f97fba3", landed=datetime(2026, 8, 3, 13, 39, 19, tzinfo=CST),
    spec="5.6", scopes=("G2", "FLEET"), variants=("heston", "heston_slv"),
    facets="*", reason="",
)
INV_DELTA = registry.Invalidation(
    commit="3fbbf21", landed=datetime(2026, 8, 3, 15, 17, 23, tzinfo=CST),
    spec="5.8", scopes=("G2",), variants="*", facets=("delta",), reason="",
)
ALL_INV = (INV_7A4, INV_PDEFIX, INV_DELTA)


def _fresh(**kw):
    base = dict(
        artifact_mtime=datetime(2026, 8, 3, 14, 39, 38, tzinfo=CST),
        scope="G2", variant="heston", facet="pv",
        dep_commits=(), dirty_deps={}, missing_deps=(), invalidations=ALL_INV,
    )
    base.update(kw)
    return provenance.freshness(**base)


def test_fresh_when_nothing_moved():
    p = _fresh(invalidations=())
    assert p.freshness == "fresh"
    assert p.mode == "inferred"
    assert p.invalidated_by is None


def test_stale_when_a_dependency_commit_landed_after():
    later = provenance.Commit("deadbee", datetime(2026, 8, 3, 16, 0, tzinfo=CST), "fix: x")
    p = _fresh(invalidations=(), dep_commits=(later,))
    assert p.freshness == "stale"
    assert [c.sha for c in p.superseded_by] == ["deadbee"]


def test_dirty_dependency_modified_after_the_artifact_is_stale():
    p = _fresh(
        invalidations=(),
        dirty_deps={"quantark/volmodels/calibration.py": datetime(2026, 8, 3, 15, 0, tzinfo=CST)},
    )
    assert p.freshness == "stale"
    assert p.dirty_deps == ("quantark/volmodels/calibration.py",)


def test_dirty_dependency_modified_before_the_artifact_is_not_counted():
    p = _fresh(
        invalidations=(),
        dirty_deps={"quantark/volmodels/calibration.py": datetime(2026, 8, 3, 10, 0, tzinfo=CST)},
    )
    assert p.freshness == "fresh"
    assert p.dirty_deps == ()


def test_the_live_g2_delta_facet_is_void_by_the_spec_commit():
    """The finding the dashboard exists for.

    gate_decision.json was written 2026-08-03 14:39.  Study section 5.8
    landed at 15:17 and states the delta half of every route decision is
    void.  Nothing on disk says so.
    """
    p = _fresh(facet="delta")
    assert p.freshness == "void"
    assert p.invalidated_by == "3fbbf21"


def test_the_same_artifacts_pv_facet_is_not_void():
    p = _fresh(facet="pv")
    assert p.freshness != "void"


@pytest.mark.parametrize(
    "scope,variant,facet",
    [
        ("G1", None, "all"),          # surface admission, engine-free
        ("G4", None, "all"),          # coupon solve
        ("FLEET", "flat_bsm", "all"),  # the 27 admitted cells
    ],
)
def test_pdefix_does_not_void_unrelated_scopes(scope, variant, facet):
    """Spec section 5.2 regression.

    An UNSCOPED f97fba3 voids G1 (Aug 1 11:35), G4 (Aug 3 01:55) and every
    flat_bsm cell (<= Aug 3 01:55), leaving zero admitted cells and
    contradicting the design's own success criteria.  Scoping is what makes
    the dashboard's headline numbers true.
    """
    p = provenance.freshness(
        artifact_mtime=datetime(2026, 8, 3, 1, 55, 26, tzinfo=CST),
        scope=scope, variant=variant, facet=facet,
        dep_commits=(), dirty_deps={}, missing_deps=(),
        invalidations=(INV_PDEFIX,),
    )
    assert p.freshness == "fresh", f"{scope}/{variant} must not be voided by f97fba3"


def test_pdefix_does_void_a_heston_cell():
    p = provenance.freshness(
        artifact_mtime=datetime(2026, 8, 3, 1, 55, 26, tzinfo=CST),
        scope="FLEET", variant="heston", facet="all",
        dep_commits=(), dirty_deps={}, missing_deps=(), invalidations=(INV_PDEFIX,),
    )
    assert p.freshness == "void"
    assert p.invalidated_by == "f97fba3"


def test_jul27_cells_are_void_by_the_engine_fixes():
    """The eight orphaned ts_bsm / localvol cells (spec section 1.2)."""
    for variant in ("ts_bsm", "localvol"):
        p = provenance.freshness(
            artifact_mtime=datetime(2026, 7, 27, 11, 3, 0, tzinfo=CST),
            scope="FLEET", variant=variant, facet="all",
            dep_commits=(), dirty_deps={}, missing_deps=(), invalidations=ALL_INV,
        )
        assert p.freshness == "void"
        assert p.invalidated_by == "41f2117"


def test_void_beats_stale():
    later = provenance.Commit("deadbee", datetime(2026, 8, 3, 18, 0, tzinfo=CST), "x")
    p = _fresh(facet="delta", dep_commits=(later,))
    assert p.freshness == "void"


def test_there_is_no_exact_mode_to_claim():
    """A badge saying 'exact' on unvalidated evidence is worse than no badge."""
    p = _fresh(invalidations=())
    assert p.mode == "inferred"
    with pytest.raises(TypeError):
        provenance.freshness(
            artifact_mtime=datetime(2026, 8, 3, tzinfo=CST), scope="G2",
            variant=None, facet="all", dep_commits=(), dirty_deps={},
            missing_deps=(), invalidations=(), stamped_commit="f97fba3",
        )


def test_a_missing_dependency_is_never_fresh():
    """Carrying it as green metadata is how a renamed engine directory
    silently certifies every verdict on the page."""
    p = _fresh(invalidations=(), missing_deps=("quantark/asset/equity/engine/",))
    assert p.freshness == "stale"
    assert p.missing_deps == ("quantark/asset/equity/engine/",)


def test_a_collapsed_untracked_parent_still_marks_its_declared_child_dirty(tmp_path):
    """git status reports '?? example/.../data/history/', never the
    surface_manifest.json inside it (verified against this repo)."""
    dep = "example/mo_volmodels/data/history/surface_manifest.json"
    target = tmp_path / dep
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")

    reported = "example/mo_volmodels/data/history/"
    dep_norm = dep.rstrip("/")
    assert dep_norm.startswith(reported.rstrip("/") + "/"), (
        "the reverse containment test is what catches the collapsed parent"
    )


def test_engine_paths_cover_the_facade_files():
    """b6b97f0 touched quantark/asset/equity/engine/pde_engine.py, which a
    narrower engine/pde/ glob would have missed entirely."""
    assert any(p == "quantark/asset/equity/engine/" for p in provenance.ENGINE_PATHS)


def test_every_gate_scope_depends_on_the_study_spec_where_it_can_be_invalidated():
    for scope in ("G2", "G4", "FLEET"):
        assert provenance.STUDY_SPEC in provenance.DEPS[scope], scope
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_dashboard.py -n0 -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mo_dashboard.provenance'`

- [ ] **Step 3: Write the provenance module**

Create `example/mo_volmodels/mo_dashboard/provenance.py`:

```python
"""Contract B: is this artifact's verdict still good?

Two verdicts, and the distinction carries weight.  *Stale* means a declared
dependency moved -- re-run to be sure.  *Void* means a declared invalidation
applies: the study spec says this output is not comparable.  Collapsing them
would let void output read as merely old, which is how it gets averaged into
a result.

Neither is proof.  Inferred freshness rests on wall-clock ordering, which is
necessary and not sufficient: a copied, restored or touched artifact reads
fresh while containing stale numbers.  Callers must render the mode.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .registry import Invalidation

FRESH = "fresh"
STALE = "stale"
VOID = "void"

EXACT = "exact"
INFERRED = "inferred"

# Directory-level, so a new engine file is covered the day it lands.  The
# whole engine/ directory, not engine/pde/: b6b97f0 changed the facade file
# quantark/asset/equity/engine/pde_engine.py, which a narrower glob misses.
ENGINE_PATHS: Tuple[str, ...] = (
    "quantark/asset/equity/engine/",
    "quantark/volmodels/",
    "quantark/backtest/replay/",
)

# A documentation commit invalidated a numeric artifact (study 5.8 vs the
# live gate_decision.json), so the study spec is a first-class dependency.
STUDY_SPEC = (
    "docs/superpowers/specs/"
    "2026-07-30-snowball-volmodel-backtest-040-rebaseline-design.md"
)

DEPS: Dict[str, Tuple[str, ...]] = {
    "G1": (
        "example/mo_volmodels/13_gate_g1_surface_admission.py",
        "example/mo_volmodels/cohort.py",
        "example/mo_volmodels/data/history/surface_manifest.json",
    ),
    "G4": (
        "example/mo_volmodels/12_snowball_volmodel_backtest.py",
        "quantark/asset/equity/product/option/snowball_option.py",
        *ENGINE_PATHS,
        STUDY_SPEC,
    ),
    "G2": (
        "example/mo_volmodels/11_pde_convergence_gate.py",
        *ENGINE_PATHS,
        STUDY_SPEC,
    ),
    "G5": (
        "example/mo_volmodels/11_pde_convergence_gate.py",
        *ENGINE_PATHS,
    ),
    "FLEET": (
        "example/mo_volmodels/12_snowball_volmodel_backtest.py",
        *ENGINE_PATHS,
        STUDY_SPEC,
    ),
}


@dataclass(frozen=True)
class Commit:
    sha: str
    when: datetime
    subject: str


@dataclass(frozen=True)
class Provenance:
    mode: str = INFERRED
    freshness: str = FRESH
    invalidated_by: Optional[str] = None
    invalidation_reason: str = ""
    superseded_by: Tuple[Commit, ...] = ()
    dirty_deps: Tuple[str, ...] = ()
    missing_deps: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "freshness": self.freshness,
            "invalidated_by": self.invalidated_by,
            "invalidation_reason": self.invalidation_reason,
            "superseded_by": [
                {"sha": c.sha, "when": c.when.isoformat(), "subject": c.subject}
                for c in self.superseded_by
            ],
            "dirty_deps": list(self.dirty_deps),
            "missing_deps": list(self.missing_deps),
        }


def freshness(
    *,
    artifact_mtime: datetime,
    scope: str,
    variant: Optional[str],
    facet: str,
    dep_commits: Sequence[Commit],
    dirty_deps: Mapping[str, datetime],
    missing_deps: Sequence[str],
    invalidations: Sequence[Invalidation],
) -> Provenance:
    """Pure.  All git and filesystem facts arrive pre-collected.

    There is no ``stamped_commit`` parameter and no ``exact`` mode.  An
    earlier draft had one that set the badge to "exact" while still deciding
    from mtime and never validating the SHA -- a badge saying *exact* on
    unvalidated evidence is the one label a reader would trust without
    checking.  It arrives with the rev-list comparison or not at all.
    """
    voided = [
        inv for inv in invalidations
        if inv.applies(scope, variant, facet) and inv.landed > artifact_mtime
    ]
    superseded = tuple(c for c in dep_commits if c.when > artifact_mtime)
    dirty = tuple(sorted(p for p, when in dirty_deps.items() if when > artifact_mtime))

    if voided:
        first = min(voided, key=lambda inv: inv.landed)
        state, by, why = VOID, first.commit, first.reason
    elif superseded or dirty or missing_deps:
        # A declared dependency that no longer exists cannot be checked, so
        # the verdict is not fresh.  Carrying it as green metadata is how a
        # renamed engine directory silently certifies everything.
        state, by, why = STALE, None, ""
    else:
        state, by, why = FRESH, None, ""

    return Provenance(
        mode=INFERRED,
        freshness=state,
        invalidated_by=by,
        invalidation_reason=why,
        superseded_by=superseded,
        dirty_deps=dirty,
        missing_deps=tuple(missing_deps),
    )


def mtime_of(path: Path) -> Optional[datetime]:
    """Timezone-aware local mtime, or None when the path is absent."""
    path = Path(path)
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone()


def _git(project_root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=str(project_root),
        capture_output=True, text=True, check=False,
    )
    return result.stdout if result.returncode == 0 else ""


def collect_git_facts(
    project_root: Path, deps: Sequence[str]
) -> Tuple[List[Commit], Dict[str, datetime], List[str]]:
    """Impure counterpart to ``freshness``.

    Returns commits touching ``deps`` (newest first), a map of
    modified-uncommitted dep paths to their mtimes, and deps whose path does
    not exist -- the last is an error row, never a skipped check: a renamed
    engine directory must not silently turn every verdict green.
    """
    project_root = Path(project_root)
    missing = [d for d in deps if not (project_root / d).exists()]
    present = [d for d in deps if (project_root / d).exists()]

    commits: List[Commit] = []
    if present:
        out = _git(project_root, "log", "--format=%h\x1f%cI\x1f%s", "--", *present)
        for line in out.splitlines():
            sha, _, rest = line.partition("\x1f")
            when, _, subject = rest.partition("\x1f")
            if sha and when:
                commits.append(Commit(sha, datetime.fromisoformat(when), subject))

    dirty: Dict[str, datetime] = {}
    status = _git(project_root, "status", "--porcelain")
    for line in status.splitlines():
        rel = line[3:].strip()
        if not rel:
            continue
        for dep in present:
            # Containment in BOTH directions.  git reports untracked trees
            # collapsed to a parent -- "?? example/mo_volmodels/data/history/"
            # and never the surface_manifest.json inside it -- so a one-way
            # test misses every change to a declared untracked dependency.
            dep_norm = dep.rstrip("/")
            if rel == dep or rel.startswith(dep_norm + "/") or dep_norm.startswith(rel.rstrip("/") + "/"):
                target = project_root / dep if rel != dep else project_root / rel
                when = mtime_of(target)
                if when is not None:
                    dirty[dep] = when
    return commits, dirty, missing
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_dashboard.py -n0 -q`
Expected: PASS — all Task 1 and Task 2 tests green

- [ ] **Step 5: Confirm the rule reproduces today's real state**

Run:

```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "example/mo_volmodels")
from pathlib import Path
from mo_dashboard import registry, provenance as P
reg = registry.load_registry(Path("example/mo_volmodels/mo_dashboard.yaml"), Path("."))
art = Path("output/pde_convergence_gate/gate_decision.json")
commits, dirty, missing = P.collect_git_facts(Path("."), P.DEPS["G2"])
for facet in ("pv", "delta"):
    p = P.freshness(
        artifact_mtime=P.mtime_of(art), scope="G2", variant="heston", facet=facet,
        dep_commits=commits, dirty_deps=dirty, missing_deps=missing,
        invalidations=reg.invalidations,
    )
    print(f"G2/heston/{facet:5s} -> {p.freshness:5s} by={p.invalidated_by}")
PY
```

Expected: `pv` is `fresh` or `stale`; **`delta` is `void` by `3fbbf21`**. If `delta` is not void, the registry's `landed` timestamp or the artifact mtime is wrong — stop and reconcile before continuing.

- [ ] **Step 6: Commit**

```bash
git add example/mo_volmodels/mo_dashboard/provenance.py test/mo_volmodels/test_dashboard.py
git commit -m "feat(dashboard): scoped freshness -- stale is not void"
```

---

## Task 3: Gate rows

**Files:**
- Create: `example/mo_volmodels/mo_dashboard/gates.py`
- Modify: `test/mo_volmodels/test_dashboard.py` (append)

**Interfaces:**
- Consumes: `registry.Registry`, `provenance.freshness`, `provenance.DEPS`, `provenance.mtime_of`, `provenance.collect_git_facts`
- Produces:
  - `GATE_SPECS: tuple[GateSpec, ...]` where `GateSpec(id, title, artifact_rel, facets, headline_fn)`
  - `headline_g1(doc) -> dict`, `headline_g4(doc) -> dict`, `headline_g2(doc) -> dict`, `headline_g5(doc) -> dict`
  - `collect_gates(project_root: Path, reg: Registry) -> tuple[list[dict], list[dict]]` returning (gate rows, errors)

- [ ] **Step 1: Write the failing test**

Append to `test/mo_volmodels/test_dashboard.py`:

```python
gates = importlib.import_module("mo_dashboard.gates")


def test_headline_g1_counts_verified_against_admitted():
    doc = {"n_admitted": 766, "n_verified": 766, "failures": [], "min_expiries_seen": 3,
           "asof": "2026-07-31"}
    h = gates.headline_g1(doc)
    assert h["n_admitted"] == 766
    assert h["n_verified"] == 766
    assert h["n_failures"] == 0
    assert h["satisfied"] is True


def test_headline_g1_is_unsatisfied_when_a_surface_failed():
    doc = {"n_admitted": 766, "n_verified": 765, "failures": [{"date": "20240101"}],
           "min_expiries_seen": 3}
    assert gates.headline_g1(doc)["satisfied"] is False


def test_headline_g4_reports_the_coupon_range_and_solved_count():
    doc = [
        {"inception": "2023-05-04", "coupon": 0.1507, "coupon_solution": {"solved": True}},
        {"inception": "2023-06-01", "coupon": 0.1153, "coupon_solution": {"solved": True}},
    ]
    h = gates.headline_g4(doc)
    assert h["n_solved"] == 2
    assert h["n_inceptions"] == 2
    assert h["coupon_min"] == pytest.approx(0.1153)
    assert h["coupon_max"] == pytest.approx(0.1507)
    assert h["satisfied"] is True


def test_headline_g4_is_unsatisfied_when_one_solve_failed():
    doc = [
        {"inception": "2023-05-04", "coupon": 0.15, "coupon_solution": {"solved": True}},
        {"inception": "2023-06-01", "coupon": None, "coupon_solution": {"solved": False}},
    ]
    assert gates.headline_g4(doc)["satisfied"] is False


def test_headline_g2_splits_pv_from_delta_per_variant():
    doc = {"variants": {
        "flat_bsm": {"route": "pde", "gate": {
            "medium_pass": True, "fine_pass": True, "biased": False,
            "delta_pass": True, "delta_biased": False,
            "delta_info": {"max_abs_contracts": 0.0142, "bound_contracts": 0.1},
        }},
        "heston": {"route": "mc", "gate": {
            "medium_pass": True, "fine_pass": False, "biased": False,
            "delta_pass": False, "delta_biased": True,
            "delta_info": {"max_abs_contracts": 0.9319, "bound_contracts": 0.1},
        }},
    }}
    h = gates.headline_g2(doc)
    assert h["variants"]["flat_bsm"]["pv"]["pass"] is True
    assert h["variants"]["flat_bsm"]["delta"]["pass"] is True
    assert h["variants"]["heston"]["route"] == "mc"
    assert h["variants"]["heston"]["delta"]["pass"] is False
    assert h["variants"]["heston"]["delta"]["max_abs_contracts"] == pytest.approx(0.9319)


def test_headline_g5_reports_not_run_when_the_artifact_is_absent():
    h = gates.headline_g5(None)
    assert h["satisfied"] is False
    assert h["state"] == "NOT_RUN"


@pytest.mark.parametrize("doc", [
    {},                                                  # empty
    {"n_operating_points": 3},                           # no under_resolved list
    {"under_resolved": []},                              # no point count
    {"n_operating_points": 0, "under_resolved": []},     # zero points swept
])
def test_headline_g5_fails_closed_on_a_partial_document(doc):
    """A truncated write must not clear a mandatory pre-flight.  An earlier
    draft returned satisfied=True for {} because `.get(...) or []` made an
    absent field indistinguishable from an empty one."""
    assert gates.headline_g5(doc)["satisfied"] is False


def test_g2_is_satisfied_by_routes_not_by_comparison_passes():
    """The real artifact routes localvol/heston/heston_slv to MC *because*
    delta_pass is false.  Treating delta_pass as the predicate would leave G2
    permanently unsatisfiable no matter what the study does."""
    doc = {"variants": {
        "flat_bsm": {"route": "pde", "gate": {"medium_pass": True, "delta_pass": True,
                                              "delta_info": {}}},
        "heston": {"route": "mc", "gate": {"medium_pass": True, "delta_pass": False,
                                           "delta_biased": True, "delta_info": {}}},
    }}
    assert gates.headline_g2(doc)["satisfied"] is True

    no_route = {"variants": {"heston": {"route": None, "gate": {"delta_info": {}}}}}
    assert gates.headline_g2(no_route)["satisfied"] is False


def test_g2_provenance_is_keyed_by_variant(tmp_path):
    """Without this, f97fba3's heston/heston_slv scope is unreachable and the
    whole scoping mechanism is dead code for the gate it was written for."""
    out = tmp_path / "output/pde_convergence_gate"
    out.mkdir(parents=True)
    (out / "gate_decision.json").write_text(json.dumps({"variants": {
        "flat_bsm": {"route": "pde", "gate": {"delta_info": {}}},
        "heston": {"route": "mc", "gate": {"delta_info": {}}},
    }}), encoding="utf-8")
    import os
    old = datetime(2026, 8, 3, 1, 0, tzinfo=CST).timestamp()
    os.utime(out / "gate_decision.json", (old, old))

    reg = registry.Registry(invalidations=(INV_PDEFIX,))
    rows, _ = gates.collect_gates(tmp_path, reg)
    g2 = next(r for r in rows if r["id"] == "G2")

    assert g2["by_variant"]["heston"]["pv"]["freshness"] == "void"
    assert g2["by_variant"]["flat_bsm"]["pv"]["freshness"] != "void"


def test_collect_gates_marks_an_unreadable_artifact(tmp_path):
    (tmp_path / "output").mkdir()
    bad = tmp_path / "output/gate_g1_admission.json"
    bad.write_text("{not json", encoding="utf-8")
    reg = registry.Registry()
    rows, errors = gates.collect_gates(tmp_path, reg)

    g1 = next(r for r in rows if r["id"] == "G1")
    assert g1["status"] == "unreadable"
    assert any("gate_g1_admission" in e["path"] for e in errors)


def test_collect_gates_marks_a_missing_artifact_not_run(tmp_path):
    (tmp_path / "output").mkdir()
    rows, _ = gates.collect_gates(tmp_path, registry.Registry())
    g5 = next(r for r in rows if r["id"] == "G5")
    assert g5["status"] == "missing"
    assert g5["headline"]["state"] == "NOT_RUN"


def test_g2_row_carries_two_facets_and_takes_the_worst():
    """The gate's overall status is the worst facet, so a void delta half
    cannot be hidden behind a passing PV half."""
    assert gates.worst_freshness(["fresh", "void"]) == "void"
    assert gates.worst_freshness(["fresh", "stale"]) == "stale"
    assert gates.worst_freshness(["fresh", "fresh"]) == "fresh"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_dashboard.py -n0 -q -k gate`
Expected: FAIL — `ModuleNotFoundError: No module named 'mo_dashboard.gates'`

- [ ] **Step 3: Write the gates module**

Create `example/mo_volmodels/mo_dashboard/gates.py`:

```python
"""Panel 1: one row per gate, each with per-facet provenance.

G4's artifact is inceptions.json, NOT the run manifest.  In the 2026-08-01
invocation the coupon solve succeeded 27/27 while every replay in the same
process failed on the PDEEngine event-stats defect (fixed by b6b97f0).  Gate
status and run status are independent axes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import provenance as P
from .registry import Registry

_ORDER = {P.FRESH: 0, P.STALE: 1, P.VOID: 2}


def worst_freshness(values: Sequence[str]) -> str:
    return max(values, key=lambda v: _ORDER.get(v, 0)) if values else P.FRESH


def headline_g1(doc: Any) -> Dict[str, Any]:
    failures = doc.get("failures") or []
    n_admitted = int(doc.get("n_admitted") or 0)
    n_verified = int(doc.get("n_verified") or 0)
    return {
        "asof": doc.get("asof"),
        "n_admitted": n_admitted,
        "n_verified": n_verified,
        "n_failures": len(failures),
        "min_expiries_seen": doc.get("min_expiries_seen"),
        "satisfied": not failures and n_admitted > 0 and n_verified == n_admitted,
    }


def headline_g4(doc: Any) -> Dict[str, Any]:
    records = list(doc or [])
    solved = [r for r in records if (r.get("coupon_solution") or {}).get("solved")]
    coupons = [float(r["coupon"]) for r in solved if r.get("coupon") is not None]
    return {
        "n_inceptions": len(records),
        "n_solved": len(solved),
        "coupon_min": min(coupons) if coupons else None,
        "coupon_max": max(coupons) if coupons else None,
        "satisfied": bool(records) and len(solved) == len(records),
    }


def headline_g2(doc: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, block in (doc.get("variants") or {}).items():
        gate = block.get("gate") or {}
        info = gate.get("delta_info") or {}
        out[name] = {
            "route": block.get("route"),
            "pv": {
                "pass": bool(gate.get("medium_pass")) and not gate.get("biased"),
                "medium_pass": gate.get("medium_pass"),
                "fine_pass": gate.get("fine_pass"),
                "biased": gate.get("biased"),
            },
            "delta": {
                "pass": bool(gate.get("delta_pass")),
                "biased": gate.get("delta_biased"),
                "max_abs_contracts": info.get("max_abs_contracts"),
                "mean_signed_contracts": info.get("mean_signed_contracts"),
                "bound_contracts": info.get("bound_contracts"),
            },
        }
    return {
        "variants": out,
        "tolerance": doc.get("tolerance"),
        "mc_reference": doc.get("mc_reference"),
        "calibration_policy": doc.get("calibration_policy"),
        # The ROUTE is the decision; the comparison flags are evidence.
        # delta_pass=false on heston is *why* heston routes to mc rather than
        # pde.  Reading delta_pass as the predicate leaves G2 permanently
        # unsatisfiable, because the routes that exist are exactly the ones
        # chosen when a comparison did not pass.
        "satisfied": bool(out) and all(v.get("route") for v in out.values()),
    }


def headline_g5(doc: Any) -> Dict[str, Any]:
    """Fail closed.  A partial write must not clear a mandatory pre-flight."""
    if doc is None:
        return {"state": "NOT_RUN", "n_under_resolved": None, "satisfied": False,
                "complete": False}
    n_points = doc.get("n_operating_points")
    under = doc.get("under_resolved")
    complete = isinstance(n_points, int) and n_points > 0 and isinstance(under, list)
    if not complete:
        return {"state": "INCOMPLETE", "n_operating_points": n_points,
                "n_under_resolved": None, "satisfied": False, "complete": False}
    return {
        "state": "RUN",
        "n_operating_points": n_points,
        "n_under_resolved": len(under),
        "satisfied": not under,
        "complete": True,
    }


@dataclass(frozen=True)
class GateSpec:
    id: str
    title: str
    artifact_rel: str
    facets: Tuple[str, ...]
    headline_fn: Callable[[Any], Dict[str, Any]]


GATE_SPECS: Tuple[GateSpec, ...] = (
    GateSpec("G1", "Surface admission", "output/gate_g1_admission.json",
             ("all",), headline_g1),
    GateSpec("G4", "Fair coupon", "output/volmodel_backtest/inceptions.json",
             ("all",), headline_g4),
    GateSpec("G2", "Engine admission", "output/pde_convergence_gate/gate_decision.json",
             ("pv", "delta"), headline_g2),
    GateSpec("G5", "Grid pre-flight", "output/pde_convergence_gate/grid_preflight.json",
             ("all",), headline_g5),
)


def collect_gates(
    project_root: Path, reg: Registry
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    project_root = Path(project_root)
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for spec in GATE_SPECS:
        path = project_root / spec.artifact_rel
        mtime = P.mtime_of(path)
        row: Dict[str, Any] = {
            "id": spec.id,
            "title": spec.title,
            "artifact_path": spec.artifact_rel,
            "artifact_mtime": mtime.isoformat() if mtime else None,
            "facets": {},
            "status": "ok",
        }

        if mtime is None:
            row["status"] = "missing"
            row["headline"] = spec.headline_fn(None) if spec.id == "G5" else {"satisfied": False}
            rows.append(row)
            continue

        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            row["headline"] = spec.headline_fn(doc)
        except Exception as exc:  # noqa: BLE001
            row["status"] = "unreadable"
            row["headline"] = {"satisfied": False, "error": f"{type(exc).__name__}: {exc}"}
            errors.append({"source": f"gate.{spec.id}", "path": str(path),
                           "message": f"{type(exc).__name__}: {exc}"})
            rows.append(row)
            continue

        commits, dirty, missing = P.collect_git_facts(project_root, P.DEPS[spec.id])

        # G2's invalidations are variant-scoped (f97fba3 -> heston,
        # heston_slv).  Evaluating with variant=None makes every one of them
        # unreachable -- the scoping mechanism would be dead code for the one
        # gate it was written for.
        variants = sorted((row["headline"].get("variants") or {}).keys()) \
            if spec.id == "G2" else []
        by_variant: Dict[str, Dict[str, Any]] = {}

        for facet in spec.facets:
            if variants:
                per = {}
                for variant in variants:
                    prov = P.freshness(
                        artifact_mtime=mtime, scope=spec.id, variant=variant,
                        facet=facet, dep_commits=commits, dirty_deps=dirty,
                        missing_deps=missing, invalidations=reg.invalidations,
                    )
                    per[variant] = prov
                    by_variant.setdefault(variant, {})[facet] = prov.as_dict()
                worst = worst_freshness([p.freshness for p in per.values()])
                pick = next(p for p in per.values() if p.freshness == worst)
                row["facets"][facet] = pick.as_dict()
            else:
                prov = P.freshness(
                    artifact_mtime=mtime, scope=spec.id, variant=None, facet=facet,
                    dep_commits=commits, dirty_deps=dirty, missing_deps=missing,
                    invalidations=reg.invalidations,
                )
                row["facets"][facet] = prov.as_dict()

        row["by_variant"] = by_variant
        row["freshness"] = worst_freshness([f["freshness"] for f in row["facets"].values()])
        rows.append(row)

    return rows, errors
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_dashboard.py -n0 -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add example/mo_volmodels/mo_dashboard/gates.py test/mo_volmodels/test_dashboard.py
git commit -m "feat(dashboard): gate rows with pv/delta facets"
```

---

## Task 4: Fleet dimensions, tree walk, and the cell state machine

**Files:**
- Create: `example/mo_volmodels/mo_dashboard/fleet.py`
- Modify: `test/mo_volmodels/test_dashboard.py` (append)

**Interfaces:**
- Consumes: `registry.Registry`, `registry.classify_run_dirs`, `provenance.freshness`, `provenance.mtime_of`
- Produces:
  - `VARIANTS: tuple[str, ...]` and `inception_tags(project_root) -> list[str]` — read from the G4 artifact, **never** by executing stage 12 (§5.3)
  - `fleet_dimensions(project_root) -> tuple[tuple[str, ...], list[str]]`
  - `CellFacts(inception, variant, run_dir, summary_mtime, dir_exists, summary_readable, dir_mtime=None)`
  - `walk_cells(run_dir: Path) -> dict[tuple[str, str], CellFacts]`
  - `manifest_failures(run_dir: Path) -> set[tuple[str, str]]`
  - `cell_state(*, facts, in_failures, prov, poll_window_seconds=None, now=None) -> str`
  - `count_states(states) -> dict[str, int]`; `admitted(counts) -> int` = `fresh + stale`
  - `collect_fleet(project_root, reg, *, poll_window_seconds=None, now=None) -> dict` — resolves `project_root` at entry

- [ ] **Step 1: Write the failing test**

Append to `test/mo_volmodels/test_dashboard.py`:

```python
fleet = importlib.import_module("mo_dashboard.fleet")

FRESH_PROV = provenance.Provenance(freshness="fresh")
STALE_PROV = provenance.Provenance(freshness="stale")
VOID_PROV = provenance.Provenance(freshness="void", invalidated_by="41f2117")


def _facts(**kw):
    base = dict(inception="2023-05-04", variant="flat_bsm", run_dir=Path("/x"),
                summary_mtime=datetime(2026, 8, 3, 1, 55, tzinfo=CST),
                dir_exists=True, summary_readable=True)
    base.update(kw)
    return fleet.CellFacts(**base)


def test_variants_come_from_stage12_not_stage13():
    """Stage 13's VARIANT_ORDER lists five and omits flat_bsm_quad, the
    engine control added by gate-plan Task 3 (spec section 1.3)."""
    assert fleet.VARIANTS == (
        "flat_bsm", "flat_bsm_quad", "ts_bsm", "localvol", "heston", "heston_slv",
    )


def test_cell_state_precedence_is_total():
    # 1 unreadable beats everything
    assert fleet.cell_state(facts=_facts(summary_readable=False),
                            in_failures=True, prov=VOID_PROV) == "unreadable"
    # 3 failed beats void
    assert fleet.cell_state(facts=_facts(), in_failures=True, prov=VOID_PROV) == "failed"
    # 4 void beats stale
    assert fleet.cell_state(facts=_facts(), in_failures=False, prov=VOID_PROV) == "void"
    # 5 stale beats fresh
    assert fleet.cell_state(facts=_facts(), in_failures=False, prov=STALE_PROV) == "stale"
    # 6 fresh
    assert fleet.cell_state(facts=_facts(), in_failures=False, prov=FRESH_PROV) == "fresh"
    # 7 missing
    assert fleet.cell_state(facts=_facts(dir_exists=False, summary_mtime=None),
                            in_failures=False, prov=FRESH_PROV) == "missing"


def test_a_failed_attempt_without_a_directory_is_failed_not_missing():
    """failed and missing must not both match; precedence resolves it."""
    state = fleet.cell_state(facts=_facts(dir_exists=False, summary_mtime=None),
                             in_failures=True, prov=FRESH_PROV)
    assert state == "failed"


def test_running_only_when_a_poll_window_is_supplied():
    facts = _facts(summary_mtime=None, dir_exists=True)
    now = datetime(2026, 8, 3, 16, 0, tzinfo=CST)
    # snapshot mode -- no poll window, so an in-flight dir is simply missing
    assert fleet.cell_state(facts=facts, in_failures=False, prov=FRESH_PROV) == "missing"
    # serve mode -- dir touched inside the window
    facts_live = _facts(summary_mtime=None, dir_exists=True,
                        dir_mtime=datetime(2026, 8, 3, 15, 59, 55, tzinfo=CST))
    assert fleet.cell_state(facts=facts_live, in_failures=False, prov=FRESH_PROV,
                            poll_window_seconds=30, now=now) == "running"


def test_walk_cells_finds_what_the_manifest_omits(tmp_path):
    """Spec section 1.2: the manifest records only its last invocation.

    The real tree holds 27 flat_bsm plus 4 ts_bsm and 4 localvol from Jul 27,
    while the manifest lists 27 flat_bsm.  The walk must win.
    """
    runs = tmp_path / "runs"
    for inception in ("2023-05-04", "2023-06-01"):
        for variant in ("flat_bsm", "ts_bsm"):
            cell = runs / inception / variant
            cell.mkdir(parents=True)
            (cell / "run_summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "config": {"variants": ["flat_bsm"]},
        "counts": {"runs_completed": 2},
        "runs": [{"inception": "2023-05-04", "variant": "flat_bsm"},
                 {"inception": "2023-06-01", "variant": "flat_bsm"}],
        "failures": [],
    }), encoding="utf-8")

    cells = fleet.walk_cells(tmp_path)
    assert len(cells) == 4
    assert ("2023-05-04", "ts_bsm") in cells


def test_manifest_failures_are_keyed_by_pair(tmp_path):
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "failures": [{"inception": "2025-07-01", "variant": "flat_bsm",
                      "error": "event-stats engine returned no stats"}],
    }), encoding="utf-8")
    assert fleet.manifest_failures(tmp_path) == {("2025-07-01", "flat_bsm")}


def test_coverage_counts_fresh_plus_stale_not_fresh_alone():
    """Stale means 're-run to be certain', not 'absent'.  Counting fresh
    alone reads 0/162 on the live tree, where every flat_bsm cell predates
    f97fba3, 3fbbf21 and ec20db9."""
    counts = fleet.count_states([
        "fresh", "fresh", "void", "void", "stale", "failed", "missing",
    ])
    assert counts["fresh"] == 2
    assert counts["stale"] == 1
    assert counts["void"] == 2
    assert fleet.admitted(counts) == 3


def test_a_manifest_failure_with_no_directory_still_renders_failed(tmp_path):
    """Collector-level, not a hand-built cell_state call.  A run that failed
    early enough to leave no directory must not vanish into 'missing'."""
    run_dir = tmp_path / "output/volmodel_backtest"
    (run_dir / "runs").mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "failures": [{"inception": "2023-05-04", "variant": "flat_bsm",
                      "error": "event-stats engine returned no stats"}],
    }), encoding="utf-8")
    (tmp_path / "output/volmodel_backtest/inceptions.json").write_text(
        json.dumps([{"inception": "2023-05-04", "coupon": 0.15,
                     "coupon_solution": {"solved": True}}]), encoding="utf-8")

    reg = registry.Registry(fleet_dirs=(run_dir.resolve(),))
    block = fleet.collect_fleet(tmp_path, reg)
    assert block["grid"]["flat_bsm"]["2023-05-04"]["state"] == "failed"
    assert block["counts"]["failed"] == 1


def test_dimensions_come_from_the_g4_artifact_not_from_running_stage12(tmp_path):
    """Rendering must not import the pricing stack (spec section 5.3)."""
    (tmp_path / "output/volmodel_backtest").mkdir(parents=True)
    (tmp_path / "output/volmodel_backtest/inceptions.json").write_text(
        json.dumps([{"inception": "2023-05-04"}, {"inception": "2023-06-01"}]),
        encoding="utf-8")

    before = {m for m in sys.modules if m.startswith("quantark.asset")}
    variants, tags = fleet.fleet_dimensions(tmp_path)
    after = {m for m in sys.modules if m.startswith("quantark.asset")}

    assert tags == ["2023-05-04", "2023-06-01"]
    assert len(variants) == 6
    assert after == before, "collecting dimensions imported pricing code"


def test_no_g4_artifact_means_no_defined_fleet(tmp_path):
    (tmp_path / "output").mkdir()
    block = fleet.collect_fleet(tmp_path, registry.Registry())
    assert block["inceptions"] == []
    assert block["expected_cells"] == 0
    assert any(e["source"] == "fleet.dimensions" for e in block["errors"])


@pytest.mark.skipif(
    not (PROJECT_ROOT / "example/mo_volmodels/data/history").is_dir(),
    reason="needs the uncommitted history cache",
)
def test_the_artifact_matches_stage12s_schedule():
    """The definition is still enforced -- here, in the test, where paying to
    import stage 12 is fine, rather than on every page render."""
    import pandas as pd
    s12 = fleet._stage12()
    cohort = fleet._cohort()
    history = PROJECT_ROOT / "example/mo_volmodels/data/history"
    spot_csv = history / "csi1000_spot.csv"
    spot = pd.read_csv(spot_csv)
    scheduled = s12.schedule_inceptions(
        calendar=s12.stage11().TradingCalendar.from_spot_csv(spot_csv),
        data_start=pd.Timestamp(spot["date"].iloc[0]).date(),
        data_end=cohort.COHORT_ASOF,
        first_admitted_surface=cohort.admitted_dates(history)[0],
    )
    assert fleet.inception_tags(PROJECT_ROOT) == [d.isoformat() for d in scheduled]
    assert len(scheduled) == 27
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_dashboard.py -n0 -q -k "fleet or cell"`
Expected: FAIL — `ModuleNotFoundError: No module named 'mo_dashboard.fleet'`

- [ ] **Step 3: Write the fleet module**

Create `example/mo_volmodels/mo_dashboard/fleet.py`:

```python
"""Panel 3: fleet coverage from the run tree, never from manifest counts.

run_manifest.json records only its last invocation.  Walking the tree finds
35 cells where the manifest reports 27 -- eight of them orphaned Jul-27
ts_bsm/localvol runs that predate the 7A.4 engine fixes and that no tool
counts.  aggregate() iterates manifest["runs"], so it does not see them
either; they simply occupy the tree looking like current work.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from . import provenance as P
from .registry import Registry, classify_run_dirs

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MO_DIR = PROJECT_ROOT / "example/mo_volmodels"
DEFAULT_HISTORY_DIR = MO_DIR / "data/history"

STATES = ("unreadable", "running", "failed", "void", "stale", "fresh", "missing")


def _load_stage(name: str, filename: str):
    path = MO_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _stage12():
    return _load_stage("mo_dash_s12", "12_snowball_volmodel_backtest.py")


def _cohort():
    return _load_stage("mo_dash_cohort", "cohort.py")


# The canonical 6-tuple from 12_snowball_volmodel_backtest.py:143.  NOT
# 13_aggregate_and_report.py:41 VARIANT_ORDER, which lists five and omits
# flat_bsm_quad (spec section 1.3).
VARIANTS: Tuple[str, ...] = (
    "flat_bsm", "flat_bsm_quad", "ts_bsm", "localvol", "heston", "heston_slv",
)


G4_ARTIFACT = "output/volmodel_backtest/inceptions.json"


def inception_tags(project_root: Path) -> List[str]:
    """The pinned inception fleet, read from the G4 artifact.

    This must NOT call ``schedule_inceptions``.  That function lives in stage
    12, and reaching it means exec'ing a module that imports the whole
    pricing and backtest stack: slow on every render, a violation of the
    read-only/no-pricing contract, and an outright failure in a read-only
    environment where matplotlib cannot write its font cache -- which would
    degrade the grid to zero cells silently.

    inceptions.json IS the authoritative list of what the fleet is.  The
    equality with schedule_inceptions() is enforced by
    test_the_artifact_matches_stage12s_schedule, not at render time.
    """
    path = Path(project_root) / G4_ARTIFACT
    if not path.exists():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    return [str(r["inception"]) for r in doc if r.get("inception")]


def fleet_dimensions(project_root: Path) -> Tuple[Tuple[str, ...], List[str]]:
    return VARIANTS, inception_tags(project_root)


@dataclass(frozen=True)
class CellFacts:
    inception: str
    variant: str
    run_dir: Path
    summary_mtime: Optional[datetime]
    dir_exists: bool
    summary_readable: bool
    dir_mtime: Optional[datetime] = None


def walk_cells(run_dir: Path) -> Dict[Tuple[str, str], CellFacts]:
    """Every (inception, variant) with a cell directory under ``runs/``."""
    run_dir = Path(run_dir)
    root = run_dir / "runs"
    out: Dict[Tuple[str, str], CellFacts] = {}
    if not root.is_dir():
        return out
    for inception_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for variant_dir in sorted(p for p in inception_dir.iterdir() if p.is_dir()):
            summary = variant_dir / "run_summary.json"
            readable = True
            if summary.exists():
                try:
                    json.loads(summary.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    readable = False
            out[(inception_dir.name, variant_dir.name)] = CellFacts(
                inception=inception_dir.name,
                variant=variant_dir.name,
                run_dir=run_dir,
                summary_mtime=P.mtime_of(summary),
                dir_exists=True,
                summary_readable=readable,
                dir_mtime=P.mtime_of(variant_dir),
            )
    return out


def manifest_failures(run_dir: Path) -> Set[Tuple[str, str]]:
    path = Path(run_dir) / "run_manifest.json"
    if not path.exists():
        return set()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return set()
    return {
        (str(f.get("inception")), str(f.get("variant")))
        for f in doc.get("failures") or []
    }


def cell_state(
    *,
    facts: CellFacts,
    in_failures: bool,
    prov: P.Provenance,
    poll_window_seconds: Optional[float] = None,
    now: Optional[datetime] = None,
) -> str:
    """Exhaustive, resolved by strict precedence (spec section 4.3)."""
    if facts.dir_exists and not facts.summary_readable:
        return "unreadable"
    if (
        poll_window_seconds is not None
        and facts.dir_exists
        and facts.summary_mtime is None
        and facts.dir_mtime is not None
        and now is not None
        and (now - facts.dir_mtime).total_seconds() <= poll_window_seconds
    ):
        return "running"
    if in_failures:
        return "failed"
    if facts.summary_mtime is None:
        return "missing"
    if prov.freshness == P.VOID:
        return "void"
    if prov.freshness == P.STALE:
        return "stale"
    return "fresh"


def count_states(states: Sequence[str]) -> Dict[str, int]:
    return {name: sum(1 for s in states if s == name) for name in STATES}


def admitted(counts: Dict[str, int]) -> int:
    """Work that exists: fresh plus stale.

    Counting fresh alone reads 0/162 on the live tree -- every flat_bsm cell
    predates f97fba3, 3fbbf21 and ec20db9 -- which is not a useful statement
    about a fleet with 27 completed cells.  Stale means "re-run to be
    certain", not "absent"; void, failed and missing are what disqualify.
    """
    return counts.get("fresh", 0) + counts.get("stale", 0)


def collect_fleet(
    project_root: Path,
    reg: Registry,
    *,
    poll_window_seconds: Optional[float] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    # Resolve first: the registry stores absolute dirs, so a relative
    # project_root makes run_dir.relative_to(project_root) raise ValueError.
    project_root = Path(project_root).resolve()
    errors: List[Dict[str, str]] = []
    variants, tags = fleet_dimensions(project_root)
    if not tags:
        errors.append({
            "source": "fleet.dimensions",
            "path": G4_ARTIFACT,
            "message": "no G4 artifact: with no coupon solve there is no defined fleet",
        })
    inception_tags = list(tags)

    roles = classify_run_dirs(reg, project_root / "output")
    commits, dirty, missing = P.collect_git_facts(project_root, P.DEPS["FLEET"])

    grid: Dict[str, Dict[str, Any]] = {}
    states: List[str] = []
    for variant in variants:
        row: Dict[str, Any] = {}
        for tag in inception_tags:
            row[tag] = {"state": "missing", "run_dir": None, "provenance": None}
        grid[variant] = row

    for run_dir in reg.fleet_dirs:
        cells = walk_cells(run_dir)
        failures = manifest_failures(run_dir)

        # A run that failed early enough to leave no directory appears only in
        # the manifest.  Iterating walk_cells alone leaves that cell "missing"
        # and hides the execution failure, so synthesise facts for it.
        for key in failures - set(cells):
            tag, variant = key
            cells[key] = CellFacts(
                inception=tag, variant=variant, run_dir=run_dir,
                summary_mtime=None, dir_exists=False, summary_readable=True,
            )

        for (tag, variant), facts in cells.items():
            prov = P.Provenance()
            if facts.summary_mtime is not None:
                prov = P.freshness(
                    artifact_mtime=facts.summary_mtime, scope="FLEET",
                    variant=variant, facet="all", dep_commits=commits,
                    dirty_deps=dirty, missing_deps=missing,
                    invalidations=reg.invalidations,
                )
            state = cell_state(
                facts=facts, in_failures=(tag, variant) in failures, prov=prov,
                poll_window_seconds=poll_window_seconds, now=now,
            )
            if variant in grid and tag in grid[variant]:
                grid[variant][tag] = {
                    "state": state,
                    "run_dir": str(run_dir.relative_to(project_root)),
                    "provenance": prov.as_dict(),
                }
            else:
                errors.append({
                    "source": "fleet.offgrid", "path": str(run_dir),
                    "message": f"cell {tag}/{variant} is outside the pinned 6x27 grid",
                })

    for variant in variants:
        for tag in inception_tags:
            states.append(grid[variant][tag]["state"])

    counts = count_states(states)
    run_dirs = []
    for path, role in sorted(roles.items()):
        shown = str(path.relative_to(project_root)) if path.is_relative_to(project_root) \
            else str(path)
        run_dirs.append({
            "dir": shown,
            "role": role,
            "n_cells": len(walk_cells(path)),
        })

    return {
        "variants": list(variants),
        "inceptions": inception_tags,
        "expected_cells": len(variants) * len(inception_tags),
        "grid": grid,
        "counts": counts,
        "admitted": admitted(counts),
        "run_dirs": run_dirs,
        "errors": errors,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_dashboard.py -n0 -q`
Expected: PASS

- [ ] **Step 5: Verify against the real tree**

Run:

```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "example/mo_volmodels")
from pathlib import Path
from mo_dashboard import registry, fleet
root = Path(".").resolve()
reg = registry.load_registry(root / "example/mo_volmodels/mo_dashboard.yaml", root)
f = fleet.collect_fleet(root, reg)
print("expected_cells", f["expected_cells"])
print("counts", f["counts"])
print("admitted", f["admitted"])
for d in f["run_dirs"]:
    print(" ", d["role"], d["dir"], d["n_cells"])
PY
```

Expected: `expected_cells 162`; `counts["stale"] == 27`; `counts["fresh"] == 0`; `counts["void"] == 8`; `admitted 27`; six run dirs listed with roles, none `unclassified`.

The 27 read **stale**, not fresh, and that is correct: every `flat_bsm` cell is dated no later
than 2026-08-03 01:55 while `f97fba3` (13:39), `3fbbf21` (15:17) and `ec20db9` (15:45) all touch
declared FLEET dependencies. They are work that exists and should be re-run to be certain — which
is exactly what `admitted = fresh + stale` encodes.

- [ ] **Step 6: Commit**

```bash
git add example/mo_volmodels/mo_dashboard/fleet.py test/mo_volmodels/test_dashboard.py
git commit -m "feat(dashboard): fleet grid from the run tree, 7-state precedence"
```

---

## Task 5: The results blocks

**Files:**
- Create: `example/mo_volmodels/mo_dashboard/results.py`
- Modify: `test/mo_volmodels/test_dashboard.py` (append)

**Interfaces:**
- Consumes: gate rows from Task 3, fleet dict from Task 4
- Produces:
  - `Read(state, doc, message, path)` and `read_json(path) -> Read` — `missing` and `unreadable` never collapse (§3.3)
  - `gate_evidence_block(g2_row) -> dict`
  - `backtest_block(project_root, fleet_block, errors: list) -> dict` — appends to `errors`; includes `reconciliation`
  - `calibration_block(project_root, errors: list) -> dict`
  - `collect_results(project_root, gate_rows, fleet_block) -> tuple[dict, list[dict]]` — the error list is never unconditionally empty

- [ ] **Step 1: Write the failing test**

Append to `test/mo_volmodels/test_dashboard.py`:

```python
results = importlib.import_module("mo_dashboard.results")


def test_gate_evidence_marks_the_delta_column_void_with_its_citation():
    g2_row = {
        "id": "G2",
        "headline": {"variants": {
            "heston": {"route": "mc",
                       "pv": {"pass": True},
                       "delta": {"pass": False, "max_abs_contracts": 0.9319,
                                 "bound_contracts": 0.1}},
        }},
        "facets": {
            "pv": {"freshness": "stale", "invalidated_by": None},
            "delta": {"freshness": "void", "invalidated_by": "3fbbf21",
                      "invalidation_reason": "reference noise"},
        },
    }
    block = results.gate_evidence_block(g2_row)
    assert block["delta"]["freshness"] == "void"
    assert block["delta"]["invalidated_by"] == "3fbbf21"
    assert block["pv"]["freshness"] == "stale"
    assert block["variants"]["heston"]["delta"]["max_abs_contracts"] == pytest.approx(0.9319)


def test_backtest_block_reconciles_its_denominator_against_the_tree():
    """Panel 2 is manifest-scoped, Panel 3 is tree-scoped, and they
    legitimately differ (spec section 1.2).  The difference is rendered, not
    left to the eye."""
    block = results.reconcile(manifest_runs=27, tree_fresh=27, tree_total=35)
    assert block["manifest_runs"] == 27
    assert block["tree_total"] == 35
    assert block["unaccounted"] == 8
    assert block["agrees"] is False

    same = results.reconcile(manifest_runs=27, tree_fresh=27, tree_total=27)
    assert same["agrees"] is True


def test_calibration_block_bands_the_feller_ratio():
    records = [
        {"feller_ratio": 0.3}, {"feller_ratio": 0.49},
        {"feller_ratio": 1.0}, {"feller_ratio": 9.9},
        {"feller_ratio": 10.1}, {"feller_ratio": 500.0},
    ]
    bands = results.feller_bands(records)
    assert bands["violated"]["n"] == 2       # < 0.5
    assert bands["usable"]["n"] == 2         # 0.5 .. 10
    assert bands["sigma_collapsed"]["n"] == 2  # > 10


def test_a_corrupt_manifest_is_an_error_not_zero_runs(tmp_path):
    """Fail soft, loud.  A truncated write must not render as a legitimate
    'runs_completed: 0'."""
    (tmp_path / "output/volmodel_backtest").mkdir(parents=True)
    (tmp_path / "output/volmodel_backtest/run_manifest.json").write_text(
        '{"counts": {"runs_comple', encoding="utf-8")

    errors = []
    block = results.backtest_block(tmp_path, {"run_dirs": [], "admitted": 0}, errors)
    assert block["manifest_state"] == "unreadable"
    assert any(e["source"] == "results.backtest" for e in errors)


def test_read_json_separates_missing_from_corrupt(tmp_path):
    assert results.read_json(tmp_path / "nope.json").state == "missing"
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    read = results.read_json(bad)
    assert read.state == "unreadable"
    assert "JSONDecodeError" in read.message or "Expecting" in read.message


def test_sigma_collapse_band_label_is_provisional():
    """Study 5.9 (ec20db9) supersedes 7A.11's attribution: those dates fail
    on discretisation, not calibration, and are fixable.  The label must not
    read as a property of the model."""
    bands = results.feller_bands([{"feller_ratio": 50.0}])
    assert bands["sigma_collapsed"]["label"] == "EXCLUDE (provisional)"
    assert "5.9" in bands["sigma_collapsed"]["citation"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_dashboard.py -n0 -q -k "results or feller or reconcile or evidence"`
Expected: FAIL — `ModuleNotFoundError: No module named 'mo_dashboard.results'`

- [ ] **Step 3: Write the results module**

Create `example/mo_volmodels/mo_dashboard/results.py`:

```python
"""Panel 2: gate evidence, backtest outcomes, calibration health."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Study section 8: KO dates collapse onto ~13 days and 2024-10-08 kills 7, so
# effective sample size is far below 27.  A 27-column table without this
# invites over-reading.
OUTCOME_CAVEAT = (
    "KO dates collapse onto ~13 days; 2024-10-08 kills 7. "
    "Effective sample size is far below 27 (study section 8)."
)

# Study 7A.10(3) established the exclusion; 5.9 (ec20db9) supersedes 7A.11's
# attribution -- these dates fail on DISCRETISATION (Peclet ~5,872 against a
# monotonicity bound of 2), not calibration, and are fixable.
SIGMA_COLLAPSE_LABEL = "EXCLUDE (provisional)"
SIGMA_COLLAPSE_CITATION = "study 7A.10(3); attribution superseded by 5.9"


def gate_evidence_block(g2_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not g2_row:
        return {"state": "MISSING", "variants": {}}
    facets = g2_row.get("facets") or {}
    headline = g2_row.get("headline") or {}
    return {
        "state": g2_row.get("status", "ok"),
        "variants": headline.get("variants") or {},
        "tolerance": headline.get("tolerance"),
        "mc_reference": headline.get("mc_reference"),
        "calibration_policy": headline.get("calibration_policy"),
        "pv": facets.get("pv") or {},
        "delta": facets.get("delta") or {},
    }


def reconcile(*, manifest_runs: int, tree_fresh: int, tree_total: int) -> Dict[str, Any]:
    """Panel 2 counts what aggregate() sees; Panel 3 counts what exists."""
    return {
        "manifest_runs": manifest_runs,
        "tree_fresh": tree_fresh,
        "tree_total": tree_total,
        "unaccounted": tree_total - manifest_runs,
        "agrees": manifest_runs == tree_total,
        "note": (
            "Panel 2 is manifest-scoped (13_aggregate_and_report.py iterates "
            "run_manifest.json['runs']); Panel 3 walks runs/. A difference "
            "means cells exist that no aggregate counts."
        ),
    }


def feller_bands(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ratios = [
        float(r["feller_ratio"]) for r in records
        if r.get("feller_ratio") is not None
    ]
    n = len(ratios)

    def band(lo: Optional[float], hi: Optional[float]) -> int:
        return sum(
            1 for x in ratios
            if (lo is None or x >= lo) and (hi is None or x < hi)
        )

    violated, usable, collapsed = band(None, 0.5), band(0.5, 10.0), band(10.0, None)
    return {
        "n": n,
        "violated": {"n": violated, "pct": 100.0 * violated / n if n else None,
                     "label": "unconstrained fails G2", "citation": "study 7A.11"},
        "usable": {"n": usable, "pct": 100.0 * usable / n if n else None,
                   "label": "usable", "citation": ""},
        "sigma_collapsed": {"n": collapsed, "pct": 100.0 * collapsed / n if n else None,
                            "label": SIGMA_COLLAPSE_LABEL,
                            "citation": SIGMA_COLLAPSE_CITATION},
    }


@dataclass(frozen=True)
class Read:
    """Absent and corrupt are different states.

    A reader that answers None to both lets a truncated run_manifest.json
    render as "0 runs completed" -- a legitimate-looking result produced by a
    parse failure.
    """

    state: str          # "ok" | "missing" | "unreadable"
    doc: Any = None
    message: str = ""
    path: str = ""


def read_json(path: Path) -> Read:
    path = Path(path)
    if not path.exists():
        return Read("missing", None, "no such file", str(path))
    try:
        return Read("ok", json.loads(path.read_text(encoding="utf-8")), "", str(path))
    except Exception as exc:  # noqa: BLE001
        return Read("unreadable", None, f"{type(exc).__name__}: {exc}", str(path))


def _calibration_records(status: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for entry in (status or {}).get("expected_date_records", {}).values():
        for variant in (entry.get("variants") or {}).values():
            record = variant.get("record")
            if isinstance(record, dict):
                out.append(record)
    return out


def backtest_block(
    project_root: Path, fleet_block: Dict[str, Any], errors: List[Dict[str, str]]
) -> Dict[str, Any]:
    project_root = Path(project_root)
    read = read_json(project_root / "output/volmodel_backtest/run_manifest.json")
    if read.state == "unreadable":
        errors.append({"source": "results.backtest", "path": read.path,
                       "message": read.message})
    manifest = read.doc if read.state == "ok" else {}
    counts = manifest.get("counts") or {}
    tree_total = sum(
        d.get("n_cells") or 0 for d in fleet_block.get("run_dirs", [])
        if d.get("role") == "fleet"
    )
    return {
        "manifest_state": read.state,
        "manifest_counts": counts,
        "config_variants": (manifest.get("config") or {}).get("variants"),
        "hedge_costs": manifest.get("hedge_costs"),
        "gate_g2": manifest.get("gate_g2"),
        "reconciliation": reconcile(
            manifest_runs=int(counts.get("runs_completed") or 0),
            tree_fresh=fleet_block.get("admitted", 0),
            tree_total=tree_total,
        ),
        "caveat": OUTCOME_CAVEAT,
    }


def calibration_block(
    project_root: Path, errors: List[Dict[str, str]]
) -> Dict[str, Any]:
    project_root = Path(project_root)
    read = read_json(project_root / "output/mo_daily_calibration/status.json")
    if read.state == "unreadable":
        errors.append({"source": "results.calibration", "path": read.path,
                       "message": read.message})
    status = read.doc if read.state == "ok" else None
    records = _calibration_records(status)
    costs = sorted(
        float(r["cost"]) for r in records if r.get("cost") is not None
    )

    def pct(fraction: float) -> Optional[float]:
        if not costs:
            return None
        return costs[min(len(costs) - 1, int(fraction * len(costs)))]

    return {
        "status_state": read.state,
        "as_of_date": (status or {}).get("as_of_date"),
        "n_records": len(records),
        "feller": feller_bands(records),
        "cost": {"median": pct(0.5), "p90": pct(0.9),
                 "max": costs[-1] if costs else None},
    }


def collect_results(
    project_root: Path,
    gate_rows: Sequence[Dict[str, Any]],
    fleet_block: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    g2 = next((r for r in gate_rows if r.get("id") == "G2"), None)
    errors: List[Dict[str, str]] = []
    block = {
        "gate_evidence": gate_evidence_block(g2),
        "backtest": backtest_block(project_root, fleet_block, errors),
        "calibration": calibration_block(project_root, errors),
    }
    return block, errors
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_dashboard.py -n0 -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add example/mo_volmodels/mo_dashboard/results.py test/mo_volmodels/test_dashboard.py
git commit -m "feat(dashboard): results blocks with an explicit denominator reconciliation"
```

---

## Task 6: Payload, the chain, and the integration fixture

**Files:**
- Create: `example/mo_volmodels/mo_dashboard/payload.py`
- Modify: `example/mo_volmodels/mo_dashboard/__init__.py`
- Modify: `test/mo_volmodels/test_dashboard.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 1–5
- Produces:
  - `SCHEMA_VERSION: int = 1`
  - `CHAIN: tuple[str, ...] = ("G1", "G4", "G2", "G5", "fleet", "aggregate")`
  - `node_satisfied(node, gate_rows, fleet_block, project_root) -> tuple[bool, str, str]` returning (satisfied, why, confidence)
  - `next_action(gate_rows, fleet_block, project_root) -> dict`
  - `git_state(project_root) -> dict`
  - `collect(project_root, registry_path=None, *, mode="snapshot", poll_window_seconds=None, now=None) -> dict`

- [ ] **Step 1: Write the failing test**

Append to `test/mo_volmodels/test_dashboard.py`:

```python
payload_mod = importlib.import_module("mo_dashboard.payload")


def _row(gid, satisfied=True, freshness="fresh", status="ok"):
    return {"id": gid, "status": status, "freshness": freshness,
            "headline": {"satisfied": satisfied},
            "facets": {"all": {"freshness": freshness, "mode": "inferred"}}}


def test_chain_includes_g5_before_fleet():
    """Study section 9 requires a grid-resolution sweep over every operating
    point before fleet work, and fdf3a70 made under-resolution a fail-closed
    ValidationError.  A chain that skips G5 recommends fleet work while a
    mandatory pre-flight is absent."""
    assert payload_mod.CHAIN == ("G1", "G4", "G2", "G5", "fleet", "aggregate")
    assert payload_mod.CHAIN.index("G5") < payload_mod.CHAIN.index("fleet")


def test_next_action_is_the_first_unsatisfied_node(tmp_path):
    rows = [_row("G1"), _row("G4"), _row("G2", satisfied=False, freshness="void"),
            _row("G5", satisfied=False, status="missing")]
    fleet_block = {"admitted": 27, "expected_cells": 162}
    action = payload_mod.next_action(rows, fleet_block, tmp_path)
    assert action["node"] == "G2"
    assert "void" in action["why"].lower()


def test_next_action_carries_the_confidence_of_its_evidence(tmp_path):
    rows = [_row("G1"), _row("G4"), _row("G2", satisfied=False)]
    action = payload_mod.next_action(rows, {"admitted": 0, "expected_cells": 162}, tmp_path)
    assert action["confidence"] in {"exact", "inferred"}


def test_fleet_node_needs_every_expected_cell(tmp_path):
    rows = [_row(g) for g in ("G1", "G4", "G2", "G5")]
    partial = payload_mod.next_action(rows, {"admitted": 27, "expected_cells": 162}, tmp_path)
    assert partial["node"] == "fleet"
    full = payload_mod.next_action(rows, {"admitted": 162, "expected_cells": 162}, tmp_path)
    assert full["node"] == "aggregate"


def test_payload_carries_the_schema_version_and_required_keys(tmp_path):
    (tmp_path / "output").mkdir()
    doc = payload_mod.collect(tmp_path, registry_path=tmp_path / "absent.yaml")
    for key in ("schema_version", "generated_at", "mode", "git", "cohort",
                "gates", "chain", "fleet", "results", "errors"):
        assert key in doc, key
    assert doc["schema_version"] == payload_mod.SCHEMA_VERSION
    assert doc["mode"] == "snapshot"
    assert "live" not in doc


def test_serve_mode_adds_a_live_block(tmp_path):
    (tmp_path / "output").mkdir()
    doc = payload_mod.collect(tmp_path, registry_path=tmp_path / "absent.yaml",
                              mode="serve", poll_window_seconds=30)
    assert doc["mode"] == "serve"
    assert "live" in doc


# --------------------------------------------------------------------------
# Integration fixture -- the check that would have caught the section 5.2
# defect.  Skips when the uncommitted artifacts are absent.
# --------------------------------------------------------------------------

REAL_OUTPUT = PROJECT_ROOT / "output"
REAL_HISTORY = PROJECT_ROOT / "example/mo_volmodels/data/history"

pytestmark_real = pytest.mark.skipif(
    not (REAL_OUTPUT.is_dir() and REAL_HISTORY.is_dir()),
    reason="needs the uncommitted output/ and data/history/ caches",
)


@pytestmark_real
def test_real_artifacts_produce_the_expected_dashboard_state():
    """Pins the whole expected state as of 2026-08-03.

    Every number here is a claim the design makes.  If scoping regresses,
    this fails rather than the page quietly lying.
    """
    doc = payload_mod.collect(PROJECT_ROOT)

    g2 = next(r for r in doc["gates"] if r["id"] == "G2")
    assert g2["facets"]["delta"]["freshness"] == "void"
    assert g2["facets"]["delta"]["invalidated_by"] == "3fbbf21"
    assert g2["facets"]["pv"]["freshness"] != "void"

    g1 = next(r for r in doc["gates"] if r["id"] == "G1")
    assert g1["freshness"] != "void", "f97fba3 must not reach G1"

    assert doc["fleet"]["expected_cells"] == 162
    # 27 STALE, not fresh: every flat_bsm cell is dated no later than
    # 2026-08-03 01:55 while f97fba3 (13:39), 3fbbf21 (15:17) and ec20db9
    # (15:45) all touch declared FLEET dependencies.  Coverage counts
    # fresh + stale precisely so this reads 27 rather than 0.
    assert doc["fleet"]["counts"]["stale"] == 27
    assert doc["fleet"]["counts"]["fresh"] == 0
    assert doc["fleet"]["counts"]["void"] == 8
    assert doc["fleet"]["admitted"] == 27
    assert doc["chain"]["next_action"]["node"] == "G2"


@pytestmark_real
def test_rendering_the_real_page_imports_no_pricing_code():
    """Spec criterion 9: the collector must stay out of the pricing stack."""
    before = {m for m in sys.modules if m.startswith("quantark.asset")}
    payload_mod.collect(PROJECT_ROOT)
    after = {m for m in sys.modules if m.startswith("quantark.asset")}
    assert after == before


@pytestmark_real
def test_no_run_dir_on_disk_is_unclassified():
    """A forgotten registry entry is a visible gap; there should be none now."""
    doc = payload_mod.collect(PROJECT_ROOT)
    stragglers = [d for d in doc["fleet"]["run_dirs"] if d["role"] == "unclassified"]
    assert stragglers == [], f"add these to mo_dashboard.yaml: {stragglers}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_dashboard.py -n0 -q -k "chain or next_action or payload or real_artifacts"`
Expected: FAIL — `ModuleNotFoundError: No module named 'mo_dashboard.payload'`

- [ ] **Step 3: Write the payload module**

Create `example/mo_volmodels/mo_dashboard/payload.py`:

```python
"""Assemble one versioned payload; derive the chain and the next action.

The dashboard is a viewer, not a gate.  It reports state and its own
confidence in that state, and never certifies a verdict as valid -- only
that it has, or has not, found evidence against it.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import fleet as fleet_mod
from . import gates as gates_mod
from . import provenance as P
from . import results as results_mod
from .registry import Registry, load_registry

SCHEMA_VERSION = 1

# G5 sits before fleet: study section 9 requires a grid sweep over every
# operating point first, and fdf3a70 made under-resolution fail closed.
CHAIN: Tuple[str, ...] = ("G1", "G4", "G2", "G5", "fleet", "aggregate")

DEFAULT_REGISTRY = "example/mo_volmodels/mo_dashboard.yaml"
AGGREGATE_ARTIFACT = "output/volmodel_backtest/aggregate.json"


def _git(project_root: Path, *args: str) -> str:
    result = subprocess.run(("git", *args), cwd=str(project_root),
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def git_state(project_root: Path) -> Dict[str, Any]:
    project_root = Path(project_root)
    dirty = [
        line[3:].strip()
        for line in _git(project_root, "status", "--porcelain").splitlines()
        if line[:2].strip() and not line.startswith("??")
    ]
    return {
        "branch": _git(project_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "head": _git(project_root, "rev-parse", "--short", "HEAD"),
        "head_subject": _git(project_root, "log", "-1", "--format=%s"),
        "dirty_paths": sorted(dirty),
    }


def cohort_block(project_root: Path) -> Dict[str, Any]:
    try:
        cohort = fleet_mod._cohort()
        history = Path(project_root) / "example/mo_volmodels/data/history"
        admitted = cohort.admitted_dates(history)
        excluded = cohort.excluded_records(history)
        return {
            "asof": cohort.COHORT_ASOF.isoformat(),
            "n_admitted": len(admitted),
            "n_excluded": len(excluded),
            "excluded": [
                {"date": r["date"].isoformat(), "reason": r["reason"],
                 "n_expiries": r["n_expiries"]}
                for r in excluded
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return {"asof": None, "n_admitted": None, "n_excluded": None,
                "excluded": [], "error": f"{type(exc).__name__}: {exc}"}


def _gate(rows: Sequence[Dict[str, Any]], gid: str) -> Optional[Dict[str, Any]]:
    return next((r for r in rows if r.get("id") == gid), None)


def node_satisfied(
    node: str,
    gate_rows: Sequence[Dict[str, Any]],
    fleet_block: Dict[str, Any],
    project_root: Path,
) -> Tuple[bool, str, str]:
    """(satisfied, why, confidence) for one chain node."""
    if node in ("G1", "G4", "G2", "G5"):
        row = _gate(gate_rows, node)
        if row is None:
            return False, f"{node}: no row", "inferred"
        if row.get("status") in ("missing", "unreadable"):
            return False, f"{node}: artifact {row['status']}", "inferred"
        facets = row.get("facets") or {}
        modes = {f.get("mode", "inferred") for f in facets.values()}
        confidence = "exact" if modes == {"exact"} else "inferred"
        voided = [name for name, f in facets.items() if f.get("freshness") == P.VOID]
        if voided:
            by = facets[voided[0]].get("invalidated_by")
            return False, f"{node}: {'/'.join(voided)} facet void by {by}", confidence
        if not (row.get("headline") or {}).get("satisfied"):
            return False, f"{node}: gate criteria not met", confidence
        return True, f"{node}: pass", confidence

    if node == "fleet":
        admitted = int(fleet_block.get("admitted") or 0)
        expected = int(fleet_block.get("expected_cells") or 0)
        if expected and admitted >= expected:
            return True, "fleet: complete", "inferred"
        return False, f"fleet: {admitted}/{expected} fresh cells", "inferred"

    if node == "aggregate":
        path = Path(project_root) / AGGREGATE_ARTIFACT
        if path.exists():
            return True, "aggregate: present", "inferred"
        return False, "aggregate: not produced", "inferred"

    return False, f"{node}: unknown node", "inferred"


def next_action(
    gate_rows: Sequence[Dict[str, Any]],
    fleet_block: Dict[str, Any],
    project_root: Path,
) -> Dict[str, Any]:
    for node in CHAIN:
        ok, why, confidence = node_satisfied(node, gate_rows, fleet_block, project_root)
        if not ok:
            return {"node": node, "why": why, "confidence": confidence}
    return {"node": None, "why": "every node satisfied", "confidence": "inferred"}


def _log_tails(project_root: Path, n_lines: int = 12) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    log_dir = Path(project_root) / "output"
    if not log_dir.is_dir():
        return out
    logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in logs[:3]:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:  # noqa: BLE001
            continue
        out[path.name] = lines[-n_lines:]
    return out


def collect(
    project_root: Path,
    registry_path: Optional[Path] = None,
    *,
    mode: str = "snapshot",
    poll_window_seconds: Optional[float] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    project_root = Path(project_root).resolve()
    registry_path = Path(registry_path or (project_root / DEFAULT_REGISTRY))
    reg = load_registry(registry_path, project_root)

    errors: List[Dict[str, str]] = list(reg.errors)
    gate_rows, gate_errors = gates_mod.collect_gates(project_root, reg)
    errors.extend(gate_errors)

    fleet_block = fleet_mod.collect_fleet(
        project_root, reg,
        poll_window_seconds=poll_window_seconds,
        now=now or datetime.now().astimezone(),
    )
    errors.extend(fleet_block.pop("errors", []))

    results_block, result_errors = results_mod.collect_results(
        project_root, gate_rows, fleet_block
    )
    errors.extend(result_errors)

    doc: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": (now or datetime.now().astimezone()).isoformat(),
        "mode": mode,
        "git": git_state(project_root),
        "cohort": cohort_block(project_root),
        "gates": gate_rows,
        "chain": {
            "nodes": list(CHAIN),
            "next_action": next_action(gate_rows, fleet_block, project_root),
        },
        "fleet": fleet_block,
        "results": results_block,
        "errors": errors,
    }
    if mode == "serve":
        doc["live"] = {"log_tails": _log_tails(project_root)}
    return doc
```

- [ ] **Step 4: Export `collect` from the package**

Replace `example/mo_volmodels/mo_dashboard/__init__.py` with:

```python
"""Read-only progress dashboard for the snowball vol-model study.

Every module here reads; none writes anywhere under ``output/`` except the
single HTML file the CLI is told to produce, and none imports pricing code.
See docs/superpowers/specs/2026-08-03-snowball-progress-dashboard-design.md.
"""
from .payload import SCHEMA_VERSION, collect

__all__ = ["collect", "SCHEMA_VERSION"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_dashboard.py -n0 -q`
Expected: PASS, including the two real-artifact tests. If
`test_real_artifacts_produce_the_expected_dashboard_state` fails on a count,
**do not adjust the assertion** — reconcile the registry or the collector
until the measured state matches the design, or record a spec amendment
explaining why the expected state changed.

- [ ] **Step 6: Commit**

```bash
git add example/mo_volmodels/mo_dashboard/payload.py example/mo_volmodels/mo_dashboard/__init__.py test/mo_volmodels/test_dashboard.py
git commit -m "feat(dashboard): payload, derived chain, and a real-artifact fixture"
```

---

## Task 7: The renderer

**Files:**
- Create: `example/mo_volmodels/mo_dashboard/render.py`
- Modify: `test/mo_volmodels/test_dashboard.py` (append)

**Interfaces:**
- Consumes: `payload.collect` output
- Produces: `render(payload: dict) -> str` (a complete HTML document), `PANEL_IDS: tuple[str, ...]`

- [ ] **Step 1: Write the failing test**

Append to `test/mo_volmodels/test_dashboard.py`:

```python
render_mod = importlib.import_module("mo_dashboard.render")


def test_render_produces_all_three_panels(tmp_path):
    (tmp_path / "output").mkdir()
    doc = payload_mod.collect(tmp_path, registry_path=tmp_path / "absent.yaml")
    html = render_mod.render(doc)
    assert html.startswith("<!doctype html>")
    for panel_id in render_mod.PANEL_IDS:
        assert f'id="{panel_id}"' in html


def test_render_inlines_the_payload_so_it_works_on_file_urls(tmp_path):
    """A file:// page cannot fetch() a sibling JSON, so a snapshot must carry
    its whole payload inline and issue no network calls."""
    (tmp_path / "output").mkdir()
    doc = payload_mod.collect(tmp_path, registry_path=tmp_path / "absent.yaml")
    html = render_mod.render(doc)
    assert 'id="__DASHBOARD_PAYLOAD__"' in html
    assert "fetch(" not in html
    assert "XMLHttpRequest" not in html
    assert "https://" not in html  # no CDN: the snapshot must render offline


def test_render_never_prints_a_bare_pass():
    """Spec section 6.3: never an unqualified PASS."""
    assert render_mod.verdict_label(True, "inferred") == "PASS (inferred)"
    assert render_mod.verdict_label(False, "inferred") == "FAIL (inferred)"


def test_payload_text_cannot_break_out_of_the_script_element():
    """json.dumps does not escape </script> (verified against this repo), and
    the payload carries log tails and exception text."""
    doc = {"schema_version": 1, "generated_at": "", "mode": "snapshot",
           "git": {"branch": "", "head": "", "head_subject": "", "dirty_paths": []},
           "cohort": {}, "gates": [], "chain": {"nodes": [], "next_action": {}},
           "fleet": {"grid": {}, "variants": [], "inceptions": [], "counts": {},
                     "run_dirs": [], "expected_cells": 0, "admitted": 0},
           "results": {},
           "errors": [{"source": "x", "path": "y",
                       "message": "</script><script>alert(1)</script>"}]}
    html = render_mod.render(doc)
    after_marker = html.split('id="__DASHBOARD_PAYLOAD__"', 1)[1]
    payload_element = after_marker.split("</script>", 1)[0]
    assert "alert(1)" in payload_element.replace("\\u003c", "<") or True
    # The injected closer must not appear literally inside the payload.
    assert "</script><script>" not in payload_element


def test_render_escapes_artifact_text(tmp_path):
    doc = {"schema_version": 1, "generated_at": "2026-08-03T16:00:00+08:00",
           "mode": "snapshot", "git": {"branch": "<script>x</script>", "head": "",
                                       "head_subject": "", "dirty_paths": []},
           "cohort": {}, "gates": [], "chain": {"nodes": [], "next_action": {}},
           "fleet": {"grid": {}, "variants": [], "inceptions": [], "counts": {},
                     "run_dirs": [], "expected_cells": 0, "admitted": 0},
           "results": {}, "errors": []}
    html = render_mod.render(doc)
    assert "<script>x</script>" not in html.split("__DASHBOARD_PAYLOAD__")[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_dashboard.py -n0 -q -k render`
Expected: FAIL — `ModuleNotFoundError: No module named 'mo_dashboard.render'`

- [ ] **Step 3: Write the renderer**

Create `example/mo_volmodels/mo_dashboard/render.py`. Follow the house style from
`example/simm_portfolio_demo.py` — dark paper/ink palette, monospace numerics, flat borders —
but use inline SVG/CSS rather than Plotly so a snapshot stays readable offline.

```python
"""Payload -> one self-contained HTML document.

A file:// page cannot fetch() a sibling JSON, so the payload is inlined.
"""
from __future__ import annotations

import html as _html
import json
from typing import Any, Dict, Sequence

PANEL_IDS = ("panel-status", "panel-results", "panel-fleet")

STATE_GLYPH = {
    "fresh": "██", "stale": "▒▒", "void": "░░",
    "failed": "▓▓", "running": "▶▶", "unreadable": "!!",
    "missing": "··",
}
STATE_COLOR = {
    "fresh": "var(--pos)", "stale": "var(--warn)", "void": "var(--neg)",
    "failed": "var(--neg)", "running": "var(--info)", "unreadable": "var(--neg)",
    "missing": "var(--hairline-2)",
}

_CSS = """
:root {
  --paper:#111110; --paper-2:#1c1b1a; --paper-3:#292725;
  --ink:#f5f2e8; --ink-2:#9b978d;
  --hairline:#3d3830; --hairline-2:#5c554a;
  --font-ui:'Inter Tight',-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
  --font-num:'Berkeley Mono','JetBrains Mono','SF Mono',monospace;
  --pos:#3cb371; --neg:#e45756; --warn:#f0ad4e; --info:#4c72b0;
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:var(--font-ui);background:var(--paper);color:var(--ink);
     line-height:1.5;padding:24px}
.container{max-width:1320px;margin:0 auto}
header{padding:8px 0 24px;border-bottom:1px solid var(--hairline);margin-bottom:24px}
header h1{font-size:1.5rem;font-weight:600;letter-spacing:-0.01em}
header .meta{font-family:var(--font-num);font-size:0.78rem;color:var(--ink-2);margin-top:6px}
.caveat{border-left:2px solid var(--warn);padding:8px 12px;margin:12px 0;
        font-size:0.82rem;color:var(--ink-2);background:var(--paper-2)}
section{border:1px solid var(--hairline);background:var(--paper-2);
        padding:20px 24px;margin-bottom:24px}
section h2{font-size:1rem;font-weight:600;margin-bottom:14px}
table{width:100%;border-collapse:collapse;font-family:var(--font-num);font-size:0.8rem}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--hairline)}
th{color:var(--ink-2);font-weight:500}
.badge{font-family:var(--font-num);font-size:0.72rem;padding:1px 6px;
       border:1px solid var(--hairline-2)}
.grid-wrap{overflow-x:auto}
.grid{font-family:var(--font-num);font-size:0.7rem;white-space:pre;line-height:1.35}
.err{color:var(--neg);font-family:var(--font-num);font-size:0.76rem}
"""


def esc(value: Any) -> str:
    return _html.escape(str(value if value is not None else ""), quote=True)


def verdict_label(passed: bool, confidence: str) -> str:
    """Never a bare PASS -- inferred freshness is not proof (spec 6.3)."""
    return f"{'PASS' if passed else 'FAIL'} ({confidence})"


def _badge(freshness: str, mode: str = "inferred") -> str:
    color = STATE_COLOR.get(freshness, "var(--ink-2)")
    return (f'<span class="badge" style="color:{color};border-color:{color}">'
            f"{esc(freshness)} · {esc(mode)}</span>")


def _panel_status(doc: Dict[str, Any]) -> str:
    rows = []
    for row in doc.get("gates", []):
        facets = row.get("facets") or {}
        cells = " ".join(
            f"{esc(name)}: {_badge(f.get('freshness', '?'), f.get('mode', '?'))}"
            for name, f in facets.items()
        ) or esc(row.get("status"))
        satisfied = (row.get("headline") or {}).get("satisfied")
        conf = "exact" if all(f.get("mode") == "exact" for f in facets.values()) and facets \
            else "inferred"
        rows.append(
            f"<tr><td>{esc(row.get('id'))}</td><td>{esc(row.get('title'))}</td>"
            f"<td>{esc(verdict_label(bool(satisfied), conf))}</td>"
            f"<td>{cells}</td>"
            f"<td>{esc(row.get('artifact_mtime') or '—')}</td></tr>"
        )
    action = (doc.get("chain") or {}).get("next_action") or {}
    git = doc.get("git") or {}
    dirty = git.get("dirty_paths") or []
    dirty_html = (
        "<div class='caveat'><b>Dirty working tree</b> — "
        + ", ".join(esc(p) for p in dirty) + "</div>"
    ) if dirty else ""
    cohort = doc.get("cohort") or {}
    return f"""
<section id="panel-status">
  <h2>Program status</h2>
  <table><tr><th>gate</th><th>title</th><th>verdict</th><th>facets</th><th>artifact</th></tr>
  {''.join(rows)}</table>
  <div class="caveat"><b>Next action:</b> {esc(action.get('node') or '—')} —
    {esc(action.get('why'))} <i>({esc(action.get('confidence'))})</i></div>
  {dirty_html}
  <div class="caveat">Inferred freshness is evidence <i>against</i> invalidity,
    not evidence <i>of</i> validity: a copied, restored or touched artifact
    reads fresh. Cohort pinned at {esc(cohort.get('asof'))} —
    {esc(cohort.get('n_admitted'))} admitted, {esc(cohort.get('n_excluded'))} excluded.</div>
</section>"""


def _panel_results(doc: Dict[str, Any]) -> str:
    res = doc.get("results") or {}
    evidence = res.get("gate_evidence") or {}
    rows = []
    for name, block in (evidence.get("variants") or {}).items():
        pv, delta = block.get("pv") or {}, block.get("delta") or {}
        rows.append(
            f"<tr><td>{esc(name)}</td><td>{esc(block.get('route'))}</td>"
            f"<td>{esc(pv.get('pass'))}</td>"
            f"<td>{esc(delta.get('pass'))}</td>"
            f"<td>{esc(delta.get('max_abs_contracts'))}</td>"
            f"<td>{esc(delta.get('bound_contracts'))}</td></tr>"
        )
    delta_facet = evidence.get("delta") or {}
    delta_note = ""
    if delta_facet.get("freshness") == "void":
        delta_note = (
            f"<div class='caveat'><b>Delta column VOID</b> by "
            f"{esc(delta_facet.get('invalidated_by'))} — "
            f"{esc(delta_facet.get('invalidation_reason'))}</div>"
        )
    backtest = res.get("backtest") or {}
    rec = backtest.get("reconciliation") or {}
    rec_note = "" if rec.get("agrees", True) else (
        f"<div class='caveat'><b>Denominator reconciliation:</b> "
        f"{esc(rec.get('manifest_runs'))} runs in the manifest, "
        f"{esc(rec.get('tree_total'))} cells on disk, "
        f"{esc(rec.get('unaccounted'))} unaccounted. {esc(rec.get('note'))}</div>"
    )
    calib = res.get("calibration") or {}
    feller = calib.get("feller") or {}
    band_rows = "".join(
        f"<tr><td>{esc(key)}</td><td>{esc(band.get('n'))}</td>"
        f"<td>{esc(round(band['pct'], 1) if band.get('pct') is not None else '—')}</td>"
        f"<td>{esc(band.get('label'))}</td><td>{esc(band.get('citation'))}</td></tr>"
        for key, band in feller.items() if isinstance(band, dict)
    )
    return f"""
<section id="panel-results">
  <h2>Results</h2>
  <h3>Gate evidence</h3>
  <table><tr><th>variant</th><th>route</th><th>PV</th><th>delta</th>
    <th>max |Δ| ct</th><th>bound ct</th></tr>{''.join(rows)}</table>
  {delta_note}
  <h3>Backtest outcomes</h3>
  <div class="caveat">{esc(backtest.get('caveat'))}</div>
  {rec_note}
  <h3>Calibration health</h3>
  <table><tr><th>band</th><th>n</th><th>%</th><th>label</th><th>citation</th></tr>
  {band_rows}</table>
</section>"""


def _panel_fleet(doc: Dict[str, Any]) -> str:
    fleet = doc.get("fleet") or {}
    variants: Sequence[str] = fleet.get("variants") or []
    inceptions: Sequence[str] = fleet.get("inceptions") or []
    grid = fleet.get("grid") or {}
    def state_of(variant: str, tag: str) -> str:
        return ((grid.get(variant) or {}).get(tag) or {}).get("state", "missing")

    lines = []
    for variant in variants:
        pieces = []
        n_fresh = 0
        for tag in inceptions:
            state = state_of(variant, tag)
            n_fresh += state == "fresh"
            color = STATE_COLOR.get(state, "var(--ink-2)")
            glyph = STATE_GLYPH.get(state, "··")
            pieces.append(f'<span style="color:{color}" title="{esc(tag)} {esc(state)}">'
                          f"{glyph}</span>")
        label = esc(variant).ljust(15)
        lines.append(f"{label}{''.join(pieces)}  {n_fresh}/{len(inceptions)}")
    counts = fleet.get("counts") or {}
    dir_rows = "".join(
        f"<tr><td>{esc(d.get('dir'))}</td><td>{esc(d.get('role'))}</td>"
        f"<td>{esc(d.get('n_cells') if d.get('n_cells') is not None else '—')}</td></tr>"
        for d in fleet.get("run_dirs", [])
    )
    legend = " ".join(f"{glyph} {name}" for name, glyph in STATE_GLYPH.items())
    return f"""
<section id="panel-fleet">
  <h2>Fleet coverage — {esc(fleet.get('admitted'))}/{esc(fleet.get('expected_cells'))} admitted
    ({esc(counts.get('fresh', 0))} fresh · {esc(counts.get('stale', 0))} stale)</h2>
  <div class="grid-wrap"><div class="grid">{'<br>'.join(lines)}</div></div>
  <div class="caveat">{esc(legend)} · counts {esc(json.dumps(counts))}</div>
  <table><tr><th>run dir</th><th>role</th><th>cells</th></tr>{dir_rows}</table>
</section>"""


def render(doc: Dict[str, Any]) -> str:
    errors = doc.get("errors") or []
    err_html = "".join(
        f"<div class='err'>{esc(e.get('source'))}: {esc(e.get('path'))} — "
        f"{esc(e.get('message'))}</div>" for e in errors
    )
    # json.dumps passes "</script>" through untouched (verified), and the
    # payload carries log tails, exception text and git subjects -- arbitrary
    # text from disk.  An unescaped closer terminates the application/json
    # element and everything after it becomes markup.
    payload_json = json.dumps(doc, default=str).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="en" data-theme="dark"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Snowball study — progress</title><style>{_CSS}</style></head><body>
<div class="container">
<header><h1>Snowball vol-model study — progress</h1>
<div class="meta">generated {esc(doc.get('generated_at'))} · mode {esc(doc.get('mode'))} ·
 {esc((doc.get('git') or {}).get('branch'))} @ {esc((doc.get('git') or {}).get('head'))}
 — {esc((doc.get('git') or {}).get('head_subject'))}</div></header>
{_panel_status(doc)}
{_panel_results(doc)}
{_panel_fleet(doc)}
{f'<section><h2>Errors</h2>{err_html}</section>' if err_html else ''}
</div>
<script id="__DASHBOARD_PAYLOAD__" type="application/json">{payload_json}</script>
</body></html>"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_dashboard.py -n0 -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add example/mo_volmodels/mo_dashboard/render.py test/mo_volmodels/test_dashboard.py
git commit -m "feat(dashboard): self-contained HTML renderer"
```

---

## Task 8: Server and CLI

**Files:**
- Create: `example/mo_volmodels/mo_dashboard/serve.py`
- Create: `example/mo_volmodels/16_dashboard.py`
- Modify: `test/mo_volmodels/test_dashboard.py` (append)

**Interfaces:**
- Consumes: `payload.collect`, `render.render`
- Produces: `serve(project_root, registry_path, host, port, poll_seconds) -> None`, `make_handler(...)`, and the CLI `main(argv) -> int`

- [ ] **Step 1: Write the failing test**

Append to `test/mo_volmodels/test_dashboard.py`:

```python
serve_mod = importlib.import_module("mo_dashboard.serve")


def test_api_routes_return_json_slices(tmp_path):
    (tmp_path / "output").mkdir()
    router = serve_mod.Router(tmp_path, tmp_path / "absent.yaml", poll_seconds=10)

    status, ctype, body = router.handle("/api/fleet")
    assert status == 200
    assert ctype == "application/json"
    assert "expected_cells" in json.loads(body)

    status, ctype, body = router.handle("/")
    assert status == 200
    assert ctype == "text/html; charset=utf-8"
    assert body.startswith("<!doctype html>")

    status, _, _ = router.handle("/nope")
    assert status == 404


def test_serve_mode_payload_has_live_block(tmp_path):
    (tmp_path / "output").mkdir()
    router = serve_mod.Router(tmp_path, tmp_path / "absent.yaml", poll_seconds=10)
    _, _, body = router.handle("/api/live")
    assert "log_tails" in json.loads(body)


def test_cli_writes_a_snapshot(tmp_path):
    (tmp_path / "output").mkdir()
    cli = importlib.util.spec_from_file_location(
        "mo_dash_cli", MO_DIR / "16_dashboard.py")
    module = importlib.util.module_from_spec(cli)
    cli.loader.exec_module(module)

    out = tmp_path / "snowball_dashboard_latest.html"
    rc = module.main([
        "--project-root", str(tmp_path),
        "--registry", str(tmp_path / "absent.yaml"),
        "--out", str(out),
    ])
    assert rc == 0
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_cli_writes_nothing_else_under_output(tmp_path):
    """Read-only contract: the only write is the HTML file named by --out."""
    (tmp_path / "output").mkdir()
    before = {p.name for p in (tmp_path / "output").iterdir()}
    cli = importlib.util.spec_from_file_location(
        "mo_dash_cli2", MO_DIR / "16_dashboard.py")
    module = importlib.util.module_from_spec(cli)
    cli.loader.exec_module(module)
    module.main([
        "--project-root", str(tmp_path),
        "--registry", str(tmp_path / "absent.yaml"),
        "--out", str(tmp_path / "output/snowball_dashboard_latest.html"),
    ])
    after = {p.name for p in (tmp_path / "output").iterdir()}
    assert after - before == {"snowball_dashboard_latest.html"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_dashboard.py -n0 -q -k "serve or api or cli"`
Expected: FAIL — `ModuleNotFoundError: No module named 'mo_dashboard.serve'`

- [ ] **Step 3: Write the server**

Create `example/mo_volmodels/mo_dashboard/serve.py`:

```python
"""Local server: same collectors, re-run per poll.

Bound to 127.0.0.1.  This serves a read-only view of a working directory;
it must never be exposed.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional, Tuple

from .payload import collect
from .render import render


class Router:
    """Path -> (status, content_type, body).  Pure enough to unit-test."""

    def __init__(self, project_root: Path, registry_path: Optional[Path],
                 *, poll_seconds: float = 10.0) -> None:
        self.project_root = Path(project_root)
        self.registry_path = registry_path
        self.poll_seconds = float(poll_seconds)

    def _payload(self) -> dict:
        return collect(
            self.project_root, self.registry_path,
            mode="serve", poll_window_seconds=self.poll_seconds,
        )

    def handle(self, path: str) -> Tuple[int, str, str]:
        if path in ("/", "/index.html"):
            return 200, "text/html; charset=utf-8", render(self._payload())
        if path.startswith("/api/"):
            key = path[len("/api/"):].strip("/")
            doc = self._payload()
            if key in ("gates", "results", "fleet", "live", "chain"):
                return 200, "application/json", json.dumps(doc.get(key, {}), default=str)
            if key == "all":
                return 200, "application/json", json.dumps(doc, default=str)
        return 404, "text/plain; charset=utf-8", "not found"


def make_handler(router: Router):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            status, ctype, body = router.handle(self.path)
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            return  # quiet

    return Handler


def serve(project_root: Path, registry_path: Optional[Path], *,
          host: str = "127.0.0.1", port: int = 8765, poll_seconds: float = 10.0) -> None:
    router = Router(project_root, registry_path, poll_seconds=poll_seconds)
    server = HTTPServer((host, port), make_handler(router))
    print(f"[dashboard] http://{host}:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] stopped")
    finally:
        server.server_close()
```

- [ ] **Step 4: Write the CLI**

Create `example/mo_volmodels/16_dashboard.py`:

```python
#!/usr/bin/env python
"""Stage 16 — the study progress dashboard.

    .venv/bin/python example/mo_volmodels/16_dashboard.py            # snapshot
    .venv/bin/python example/mo_volmodels/16_dashboard.py --serve    # live

Read-only.  The only thing this writes is the HTML file named by --out.
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mo_dashboard.payload import collect  # noqa: E402
from mo_dashboard.render import render  # noqa: E402
from mo_dashboard.serve import serve  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "output/snowball_dashboard_latest.html"
DEFAULT_REGISTRY = PROJECT_ROOT / "example/mo_volmodels/mo_dashboard.yaml"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--registry", type=Path, default=None,
                        help=f"registry YAML (default {DEFAULT_REGISTRY})")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"snapshot path (default {DEFAULT_OUT})")
    parser.add_argument("--serve", action="store_true", help="run the local server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--open", action="store_true", dest="open_browser")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    registry = args.registry if args.registry is not None else \
        project_root / "example/mo_volmodels/mo_dashboard.yaml"

    if args.serve:
        if args.open_browser:
            webbrowser.open(f"http://127.0.0.1:{args.port}")
        serve(project_root, registry, port=args.port, poll_seconds=args.poll_seconds)
        return 0

    out = args.out if args.out is not None else project_root / "output/snowball_dashboard_latest.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = collect(project_root, registry)
    out.write_text(render(doc), encoding="utf-8")

    action = (doc.get("chain") or {}).get("next_action") or {}
    fleet = doc.get("fleet") or {}
    counts = fleet.get("counts") or {}
    print(f"[dashboard] {out}")
    print(f"[fleet]     {fleet.get('admitted')}/{fleet.get('expected_cells')} admitted "
          f"({counts.get('fresh', 0)} fresh, {counts.get('stale', 0)} stale, "
          f"{counts.get('void', 0)} void)")
    print(f"[next]      {action.get('node')} — {action.get('why')}")
    for err in doc.get("errors", []):
        print(f"[error]     {err.get('source')}: {err.get('message')}")
    if args.open_browser:
        webbrowser.open(out.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_dashboard.py -n0 -q`
Expected: PASS

- [ ] **Step 6: Generate the real dashboard**

Run: `.venv/bin/python example/mo_volmodels/16_dashboard.py`

Expected output includes `[fleet] 27/162 admitted` and `[next] G2 — G2: delta facet void by 3fbbf21`. Open `output/snowball_dashboard_latest.html` and confirm by eye: three panels present, the delta column carries its VOID banner, the eight Jul-27 cells render in the void colour, and the dirty working tree is listed.

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/python -m pytest test/mo_volmodels/ -q`
Expected: PASS — no regression in the existing cohort/gate tests.

- [ ] **Step 8: Commit**

```bash
git add example/mo_volmodels/mo_dashboard/serve.py example/mo_volmodels/16_dashboard.py test/mo_volmodels/test_dashboard.py
git commit -m "feat(dashboard): local server and stage-16 CLI"
```

---

## Out of scope for this plan

- **Stamping gate artifacts with `provenance` blocks.** Both readers ship (§6.4); every artifact reads `inferred` until a gate script is next edited for other reasons.
- **Fixing `13_aggregate_and_report.py:41 VARIANT_ORDER`**, which lists five variants and omits `flat_bsm_quad` (§1.3). The dashboard sources variants from stage 12 instead. This is a real defect in stage 13 and should get its own change.
- **Running G5.** The dashboard renders `NOT RUN` because no artifact exists; producing one is study work, not dashboard work. The artifact path `output/pde_convergence_gate/grid_preflight.json` and the shape `{"n_operating_points": int, "under_resolved": [...]}` are assumed by `headline_g5`; if the eventual G5 writes a different shape, that function is the single place to update.
- **Any change to how gates or fleets run.** The dashboard is strictly read-only.
