"""Assemble one versioned payload; derive the chain and the next action.

The dashboard is a viewer, not a gate.  It reports state and its own
confidence in that state, and never certifies a verdict as valid -- only
that it has, or has not, found evidence against it.
"""
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import fleet as fleet_mod
from . import gates as gates_mod
from . import provenance as P
from . import results as results_mod
from .registry import load_registry, norm

SCHEMA_VERSION = 1

# G5 sits before fleet: study section 9 requires a grid sweep over every
# operating point first, and fdf3a70 made under-resolution fail closed.
CHAIN: Tuple[str, ...] = ("G1", "G4", "G2", "G5", "fleet", "aggregate")

DEFAULT_REGISTRY = "example/mo_volmodels/mo_dashboard.yaml"
AGGREGATE_ARTIFACT = "output/volmodel_backtest/aggregate.json"


def _git(project_root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
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
    """Cohort pin.  cohort.py is a pure JSON reader -- no pricing imports."""
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
                {
                    "date": r["date"].isoformat(),
                    "reason": r["reason"],
                    "n_expiries": r["n_expiries"],
                }
                for r in excluded
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "asof": None,
            "n_admitted": None,
            "n_excluded": None,
            "excluded": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


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
        return False, f"fleet: {admitted}/{expected} admitted cells", "inferred"

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
    project_root = norm(Path(project_root).absolute())
    registry_path = Path(registry_path or (project_root / DEFAULT_REGISTRY))
    reg = load_registry(registry_path, project_root)

    errors: List[Dict[str, str]] = list(reg.errors)
    gate_rows, gate_errors = gates_mod.collect_gates(project_root, reg)
    errors.extend(gate_errors)

    fleet_block = fleet_mod.collect_fleet(
        project_root,
        reg,
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
