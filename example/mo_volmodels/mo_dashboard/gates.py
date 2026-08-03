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


def gate_verdict(row: Dict[str, Any]) -> Tuple[bool, str, str]:
    """(ok, label, why) -- the SINGLE predicate for both chain and display.

    The renderer previously derived PASS from ``headline.satisfied`` alone,
    so G2 printed "PASS (inferred)" while its own delta facet badge, one
    cell to the right, read "void" -- and next_action correctly stopped at
    G2.  The most prominent verdict on the page contradicted the page.
    """
    gid = row.get("id", "?")
    status = row.get("status")
    if status in ("missing", "unreadable"):
        return False, status.upper(), f"{gid}: artifact {status}"

    facets = row.get("facets") or {}
    voided = [name for name, f in facets.items() if f.get("freshness") == P.VOID]
    if voided:
        by = facets[voided[0]].get("invalidated_by")
        return False, "VOID", f"{gid}: {'/'.join(voided)} facet void by {by}"

    if not (row.get("headline") or {}).get("satisfied"):
        return False, "FAIL (inferred)", f"{gid}: gate criteria not met"

    stale = [name for name, f in facets.items() if f.get("freshness") == P.STALE]
    if stale:
        return True, "PASS (stale, inferred)", f"{gid}: pass, {'/'.join(stale)} stale"
    return True, "PASS (inferred)", f"{gid}: pass"


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
    """G4 *defines* the inception universe, so it cannot be validated against
    it -- there is no render-time source for "how many inceptions should
    there be" that does not mean executing stage 12's ``schedule_inceptions``
    and with it the whole pricing stack (spec 5.3).

    So this checks internal completeness only, and says so.  The size of the
    universe is asserted in
    ``test_the_artifact_matches_stage12s_schedule``, which pays the import
    cost once in the suite rather than on every page render.  A silently
    shrunken G4 run would move the fleet denominator, so ``n_inceptions`` is
    rendered prominently and ``universe_verified`` is False by construction.
    """
    records = list(doc or [])
    solved = [r for r in records if (r.get("coupon_solution") or {}).get("solved")]
    coupons = [float(r["coupon"]) for r in solved if r.get("coupon") is not None]
    return {
        "n_inceptions": len(records),
        "n_solved": len(solved),
        "coupon_min": min(coupons) if coupons else None,
        "coupon_max": max(coupons) if coupons else None,
        "universe_verified": False,
        "universe_note": (
            "G4 defines the inception set; its size is asserted in the test "
            "suite, not at render time (spec 5.3)."
        ),
        "satisfied": bool(records) and len(solved) == len(records),
    }


def headline_g2(doc: Any, expected_variants: Sequence[str] = ()) -> Dict[str, Any]:
    """G2 must decide the WHOLE study universe, not whatever it contains.

    Without an expected set, an artifact holding two variants passes because
    both have routes -- and the per-variant provenance roll-up then iterates
    that same short list, so a missing variant's invalidation is never even
    evaluated.  Incompleteness must be a failure, not an empty loop.
    """
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
    expected = tuple(expected_variants)
    missing = [v for v in expected if v not in out]
    extra = [v for v in out if expected and v not in expected]
    return {
        "variants": out,
        "expected_variants": list(expected),
        "missing_variants": missing,
        "extra_variants": extra,
        "complete": bool(out) and not missing and not extra,
        "tolerance": doc.get("tolerance"),
        "mc_reference": doc.get("mc_reference"),
        "calibration_policy": doc.get("calibration_policy"),
        # The ROUTE is the decision; the comparison flags are evidence.
        # delta_pass=false on heston is *why* heston routes to mc rather than
        # pde.  Reading delta_pass as the predicate leaves G2 permanently
        # unsatisfiable, because the routes that exist are exactly the ones
        # chosen when a comparison did not pass.
        "satisfied": (
            bool(out)
            and not missing
            and not extra
            and all(v.get("route") for v in out.values())
        ),
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
    project_root: Path, reg: Registry, *, expected_variants: Sequence[str] = ()
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
            row["headline"] = (
                headline_g2(doc, expected_variants)
                if spec.id == "G2"
                else spec.headline_fn(doc)
            )
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
        # Iterate the canonical universe, not the artifact's keys: a
        # variant absent from the artifact must still be evaluated, or its
        # scoped invalidation is silently skipped.
        variants = (
            sorted(set(expected_variants) | set(row["headline"].get("variants") or {}))
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

    for row in rows:
        ok, label, why = gate_verdict(row)
        row["verdict"] = {"ok": ok, "label": label, "why": why}
        missing_variants = (row.get("headline") or {}).get("missing_variants") or []
        if missing_variants:
            errors.append({
                "source": f"gate.{row['id']}",
                "path": row["artifact_path"],
                "message": f"artifact omits study variants: {', '.join(missing_variants)}",
            })

    return rows, errors
