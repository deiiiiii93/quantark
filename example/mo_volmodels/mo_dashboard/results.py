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


@dataclass(frozen=True)
class Read:
    """Absent and corrupt are different states.

    A reader that answers None to both lets a truncated run_manifest.json
    render as "0 runs completed" -- a legitimate-looking result produced by a
    parse failure.
    """

    state: str  # "ok" | "missing" | "unreadable"
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


def gate_evidence_block(g2_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not g2_row:
        return {"state": "MISSING", "variants": {}}
    facets = g2_row.get("facets") or {}
    headline = g2_row.get("headline") or {}
    return {
        "state": g2_row.get("status", "ok"),
        "variants": headline.get("variants") or {},
        "by_variant": g2_row.get("by_variant") or {},
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
        float(r["feller_ratio"]) for r in records if r.get("feller_ratio") is not None
    ]
    n = len(ratios)

    def band(lo: Optional[float], hi: Optional[float]) -> int:
        return sum(
            1 for x in ratios if (lo is None or x >= lo) and (hi is None or x < hi)
        )

    violated, usable, collapsed = band(None, 0.5), band(0.5, 10.0), band(10.0, None)
    return {
        "n": n,
        "violated": {
            "n": violated,
            "pct": 100.0 * violated / n if n else None,
            "label": "unconstrained fails G2",
            "citation": "study 7A.11",
        },
        "usable": {
            "n": usable,
            "pct": 100.0 * usable / n if n else None,
            "label": "usable",
            "citation": "",
        },
        "sigma_collapsed": {
            "n": collapsed,
            "pct": 100.0 * collapsed / n if n else None,
            "label": SIGMA_COLLAPSE_LABEL,
            "citation": SIGMA_COLLAPSE_CITATION,
        },
    }


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
        errors.append(
            {"source": "results.backtest", "path": read.path, "message": read.message}
        )
    manifest = read.doc if read.state == "ok" else {}
    counts = manifest.get("counts") or {}
    tree_total = sum(
        d.get("n_cells") or 0
        for d in fleet_block.get("run_dirs", [])
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
        errors.append(
            {
                "source": "results.calibration",
                "path": read.path,
                "message": read.message,
            }
        )
    status = read.doc if read.state == "ok" else None
    records = _calibration_records(status)
    costs = sorted(float(r["cost"]) for r in records if r.get("cost") is not None)

    def pct(fraction: float) -> Optional[float]:
        if not costs:
            return None
        return costs[min(len(costs) - 1, int(fraction * len(costs)))]

    return {
        "status_state": read.state,
        "as_of_date": (status or {}).get("as_of_date"),
        "n_records": len(records),
        "feller": feller_bands(records),
        "cost": {
            "median": pct(0.5),
            "p90": pct(0.9),
            "max": costs[-1] if costs else None,
        },
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
