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
        return {
            "state": "NOT_RUN",
            "n_under_resolved": None,
            "satisfied": False,
            "complete": False,
        }
    n_points = doc.get("n_operating_points")
    under = doc.get("under_resolved")
    complete = isinstance(n_points, int) and n_points > 0 and isinstance(under, list)
    if not complete:
        return {
            "state": "INCOMPLETE",
            "n_operating_points": n_points,
            "n_under_resolved": None,
            "satisfied": False,
            "complete": False,
        }
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
    GateSpec(
        "G1", "Surface admission", "output/gate_g1_admission.json",
        ("all",), headline_g1,
    ),
    GateSpec(
        "G4", "Fair coupon", "output/volmodel_backtest/inceptions.json",
        ("all",), headline_g4,
    ),
    GateSpec(
        "G2", "Engine admission", "output/pde_convergence_gate/gate_decision.json",
        ("pv", "delta"), headline_g2,
    ),
    GateSpec(
        "G5", "Grid pre-flight", "output/pde_convergence_gate/grid_preflight.json",
        ("all",), headline_g5,
    ),
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
            "by_variant": {},
            "status": "ok",
        }

        if mtime is None:
            row["status"] = "missing"
            row["headline"] = (
                spec.headline_fn(None) if spec.id == "G5" else {"satisfied": False}
            )
            rows.append(row)
            continue

        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            row["headline"] = spec.headline_fn(doc)
        except Exception as exc:  # noqa: BLE001
            row["status"] = "unreadable"
            row["headline"] = {
                "satisfied": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append(
                {
                    "source": f"gate.{spec.id}",
                    "path": str(path),
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
            rows.append(row)
            continue

        commits, dirty, missing = P.collect_git_facts(project_root, P.DEPS[spec.id])

        # G2's invalidations are variant-scoped (f97fba3 -> heston,
        # heston_slv).  Evaluating with variant=None makes every one of them
        # unreachable -- the scoping mechanism would be dead code for the one
        # gate it was written for.
        variants = (
            sorted((row["headline"].get("variants") or {}).keys())
            if spec.id == "G2"
            else []
        )
        by_variant: Dict[str, Dict[str, Any]] = {}

        for facet in spec.facets:
            if variants:
                per = {}
                for variant in variants:
                    prov = P.freshness(
                        artifact_mtime=mtime,
                        scope=spec.id,
                        variant=variant,
                        facet=facet,
                        dep_commits=commits,
                        dirty_deps=dirty,
                        missing_deps=missing,
                        invalidations=reg.invalidations,
                    )
                    per[variant] = prov
                    by_variant.setdefault(variant, {})[facet] = prov.as_dict()
                worst = worst_freshness([p.freshness for p in per.values()])
                pick = next(p for p in per.values() if p.freshness == worst)
                row["facets"][facet] = pick.as_dict()
            else:
                prov = P.freshness(
                    artifact_mtime=mtime,
                    scope=spec.id,
                    variant=None,
                    facet=facet,
                    dep_commits=commits,
                    dirty_deps=dirty,
                    missing_deps=missing,
                    invalidations=reg.invalidations,
                )
                row["facets"][facet] = prov.as_dict()

        row["by_variant"] = by_variant
        row["freshness"] = worst_freshness(
            [f["freshness"] for f in row["facets"].values()]
        )
        rows.append(row)

    return rows, errors
