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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from . import provenance as P
from .registry import Registry, classify_run_dirs, norm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MO_DIR = PROJECT_ROOT / "example/mo_volmodels"

STATES = ("unreadable", "running", "failed", "void", "stale", "fresh", "missing")

G4_ARTIFACT = "output/volmodel_backtest/inceptions.json"

# The canonical 6-tuple from 12_snowball_volmodel_backtest.py:143.  NOT
# 13_aggregate_and_report.py:41 VARIANT_ORDER, which lists five and omits
# flat_bsm_quad (spec section 1.3).
VARIANTS: Tuple[str, ...] = (
    "flat_bsm",
    "flat_bsm_quad",
    "ts_bsm",
    "localvol",
    "heston",
    "heston_slv",
)


def _load_stage(name: str, filename: str):
    """Test-only.  Never called at render time -- see ``inception_tags``."""
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
    # Normalise first: the registry stores absolute dirs, so a relative
    # project_root makes run_dir.relative_to(project_root) raise ValueError.
    project_root = norm(Path(project_root).absolute())
    errors: List[Dict[str, str]] = []
    variants, tags = fleet_dimensions(project_root)
    if not tags:
        errors.append(
            {
                "source": "fleet.dimensions",
                "path": G4_ARTIFACT,
                "message": "no G4 artifact: with no coupon solve there is no defined fleet",
            }
        )
    inception_tag_list = list(tags)

    roles = classify_run_dirs(reg, project_root / "output")
    commits, dirty, missing = P.collect_git_facts(project_root, P.DEPS["FLEET"])

    grid: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        grid[variant] = {
            tag: {"state": "missing", "run_dir": None, "mtime": None,
                  "provenance": None}
            for tag in inception_tag_list
        }

    for run_dir in reg.fleet_dirs:
        cells = walk_cells(run_dir)
        failures = manifest_failures(run_dir)

        # A run that failed early enough to leave no directory appears only in
        # the manifest.  Iterating walk_cells alone leaves that cell "missing"
        # and hides the execution failure, so synthesise facts for it.
        for key in failures - set(cells):
            tag, variant = key
            cells[key] = CellFacts(
                inception=tag,
                variant=variant,
                run_dir=run_dir,
                summary_mtime=None,
                dir_exists=False,
                summary_readable=True,
            )

        for (tag, variant), facts in cells.items():
            prov = P.Provenance()
            if facts.summary_mtime is not None:
                prov = P.freshness(
                    artifact_mtime=facts.summary_mtime,
                    scope="FLEET",
                    variant=variant,
                    facet="all",
                    dep_commits=commits,
                    dirty_deps=dirty,
                    missing_deps=missing,
                    invalidations=reg.invalidations,
                )
            state = cell_state(
                facts=facts,
                in_failures=(tag, variant) in failures,
                prov=prov,
                poll_window_seconds=poll_window_seconds,
                now=now,
            )
            if variant in grid and tag in grid[variant]:
                grid[variant][tag] = {
                    "state": state,
                    "run_dir": _shown(run_dir, project_root),
                    "mtime": facts.summary_mtime.isoformat() if facts.summary_mtime else None,
                    "provenance": prov.as_dict(),
                }
            else:
                errors.append(
                    {
                        "source": "fleet.offgrid",
                        "path": _shown(run_dir, project_root),
                        "message": f"cell {tag}/{variant} is outside the pinned "
                        f"{len(variants)}x{len(inception_tag_list)} grid",
                    }
                )

    states = [
        grid[variant][tag]["state"]
        for variant in variants
        for tag in inception_tag_list
    ]
    counts = count_states(states)

    # Newest admitted cell -- an aggregate older than this is summarising
    # results it predates (spec 4.1).
    admitted_mtimes = [
        cell["mtime"]
        for variant in variants
        for tag in inception_tag_list
        for cell in [grid[variant][tag]]
        if cell.get("mtime") and cell["state"] in ("fresh", "stale")
    ]

    run_dirs = []
    for path, role in sorted(roles.items()):
        run_dirs.append(
            {
                "dir": _shown(path, project_root),
                "role": role,
                "n_cells": len(walk_cells(path)),
            }
        )

    return {
        "variants": list(variants),
        "inceptions": inception_tag_list,
        "expected_cells": len(variants) * len(inception_tag_list),
        "grid": grid,
        "counts": counts,
        "admitted": admitted(counts),
        "newest_cell_mtime": max(admitted_mtimes) if admitted_mtimes else None,
        "run_dirs": run_dirs,
        "errors": errors,
    }


def _shown(path: Path, project_root: Path) -> str:
    path = Path(path)
    if path.is_relative_to(project_root):
        return str(path.relative_to(project_root))
    return str(path)
