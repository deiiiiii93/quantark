"""Does main's continuous-KI work move the numbers this certification banked?

Main landed six commits that change how a knock-in barrier monitored
CONTINUOUSLY is priced: a first-passage transfer in the PDE (62f72b6), made
barrier-local for the vol-model solvers (c6b8401, 7baf042), and a Brownian
bridge in the MC that now uses the variance the paths actually accumulated
(512d08e).  Four of the files they touch sit inside this certification's
fail-closed implementation digest.

The certification's fourteen cells monitor their KI barrier DISCRETELY --
``make_snowball(case, dense_ki=True)`` builds a 252-per-year observation
schedule -- so none of that machinery should ever be constructed for them.
This probe checks that claim three ways, from weakest to strongest:

  1. STRUCTURAL.  After pricing the certification's own product, every PDE
     solver has ``_ki_fp is None`` and every MC engine has
     ``_ki_bridge_wanted False`` and no recorded step variance.  Positive
     controls (the same engines under continuous monitoring) keep the
     assertions from passing vacuously.  This extends main's own
     test/test_continuous_ki_scope.py to the SLV classes it does not cover,
     which are exactly the ones this certification certifies.

  2. NUMERICAL, against banked evidence.  For every cell, rebuild the target
     grid and re-run the deterministic candidate.  ``certifications[g]["pde"]``
     in the banked payload IS the raw ``central_bump_greeks`` output at that
     grid and SPOT_BUMP, so the comparison is exact equality or nothing.  This
     is stronger than a pre/post re-run: it tests the surviving artifact.

  3. ANCHORS.  ``deterministic_anchors`` prices its snowballs with
     ``dense_ki=False`` -- CONTINUOUS monitoring -- so three of the four
     anchors DO run the corrected machinery and are expected to move.  The
     probe quantifies the movement and re-checks each anchor's own
     cross-engine tolerance, because what those anchors assert is a degeneracy
     property (Heston at zero vol-of-vol must reduce to flat BSM), and that
     property either survives the correction or it does not.

Run from the worktree root:

    PYTHONPATH=$PWD .venv/bin/python \\
        docs/adi2d-greek-perf/probes/probe_merge_ki_invariance.py \\
        --evidence output/p18_strided/adi_greek_certification.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STAGE16_RELATIVE = "example/mo_volmodels/16_adi_greek_certification.py"
DEFAULT_EVIDENCE = ROOT / "output" / "p18_strided" / "adi_greek_certification.json"


def load_stage16():
    spec = importlib.util.spec_from_file_location(
        "s16_merge_probe", ROOT / STAGE16_RELATIVE
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# 1. structural


def structural_report(s16) -> list[dict]:
    """The KI machinery must not exist for a discretely monitored product."""
    import numpy as np

    from quantark.asset.equity.param import MCParams

    rows: list[dict] = []
    case = next(
        c for c in s16.certification_cases(quick=False) if c.name == "near_expiry"
    )
    env = s16.make_environment(
        case.spot, math.sqrt(max(case.params.v0, case.params.theta))
    )
    leverage = s16.make_leverage_surface(case.maturity)
    grid = s16.GridPoint(96, 24, 48)
    # The same constant artifact make_mc_engine supplies: the SLV construction
    # seam resolves a local-vol surface even when a precomputed leverage
    # surface makes it inert in the path loop.
    representative_vol = math.sqrt(max(case.params.v0, case.params.theta))
    constant_local_vol = s16.LocalVolSurface(
        strike_grid=np.array([40.0, 180.0]),
        time_grid=np.array([0.0, float(case.maturity)]),
        lv_grid=np.full((2, 2), representative_vol),
    )

    for dense_ki, label in ((True, "discrete_ki"), (False, "continuous_ki")):
        product = s16.make_snowball(case, dense_ki=dense_ki)
        for variant in ("heston", "heston_slv"):
            solver = s16.make_pde_engine(
                variant, case, grid, leverage if variant == "heston_slv" else None
            )
            solver.price(product, env)
            rows.append(
                {
                    "kind": "pde",
                    "engine": type(solver).__name__,
                    "variant": variant,
                    "monitoring": label,
                    "ki_first_passage_built": getattr(solver, "_ki_fp", None)
                    is not None,
                }
            )

        # (a) The certification's OWN reference engines, conditional-control
        #     machinery and all.  _build_time_grid is the seam main added the
        #     flag to, and the seam every reference flow passes through, so
        #     this settles the question for the configuration that was actually
        #     certified without simulating a single path.
        for variant in ("heston", "heston_slv"):
            engine = s16.make_mc_engine(
                variant,
                case,
                leverage if variant == "heston_slv" else None,
                paths_per_batch=256,
                batches=1,
                seed=7,
                substeps=1,
            )
            engine._build_time_grid(product, env, float(case.maturity))
            rows.append(
                {
                    "kind": "mc_reference",
                    "engine": type(engine).__name__,
                    "variant": variant,
                    "monitoring": label,
                    "bridge_wanted": bool(
                        getattr(engine, "_ki_bridge_wanted", False)
                    ),
                    "step_log_variance_recorded": getattr(
                        engine, "_step_log_variance", None
                    )
                    is not None,
                }
            )

        # (b) The same engine classes on their PLAIN (unconditioned) path, run
        #     end to end.  This is the branch the merge had to hand-resolve --
        #     the one where main's h2 buffer and the branch's rewritten
        #     simulate() bodies overlap -- so it must both stay silent under a
        #     discrete KI and actually record under a continuous one.
        for variant, engine in (
            (
                "heston",
                s16.QESnowballMCEngine(
                    case.params,
                    martingale_correction=True,
                    params=MCParams(num_paths=256, time_steps=13, seed=7),
                ),
            ),
            (
                "heston_slv",
                s16.HestonSLVQESnowballMCEngine(
                    case.params,
                    leverage_surface=leverage,
                    local_vol_surface=constant_local_vol,
                    martingale_correction=True,
                    params=MCParams(num_paths=256, time_steps=13, seed=7),
                ),
            ),
        ):
            engine.price(product, env)
            rows.append(
                {
                    "kind": "mc_plain",
                    "engine": type(engine).__name__,
                    "variant": variant,
                    "monitoring": label,
                    "bridge_wanted": bool(
                        getattr(engine, "_ki_bridge_wanted", False)
                    ),
                    "step_log_variance_recorded": getattr(
                        engine, "_step_log_variance", None
                    )
                    is not None,
                }
            )
    return rows


def structural_verdict(rows: list[dict]) -> tuple[bool, list[str]]:
    problems: list[str] = []
    for row in rows:
        discrete = row["monitoring"] == "discrete_ki"
        if row["kind"] == "pde":
            built = row["ki_first_passage_built"]
            if discrete and built:
                problems.append(
                    f"{row['engine']}: first-passage state built for a "
                    "DISCRETELY monitored KI"
                )
            if not discrete and not built:
                problems.append(
                    f"{row['engine']}: positive control failed -- no "
                    "first-passage state under continuous monitoring"
                )
        else:
            wanted = row["bridge_wanted"]
            if discrete and (wanted or row["step_log_variance_recorded"]):
                problems.append(
                    f"{row['engine']}: bridge armed for a DISCRETELY "
                    "monitored KI"
                )
            if not discrete and not wanted:
                problems.append(
                    f"{row['engine']}: positive control failed -- bridge not "
                    "armed under continuous monitoring"
                )
            # The plain path must actually record the variance it accumulated.
            # The conditional-control path deliberately does not: its nodes are
            # contractual with one row per (path, stratum), a shape the bridge
            # buffer cannot describe, so it leaves the recording None and
            # _ki_bridge_step_log_variance raises rather than substituting a
            # variance those paths never had.
            if (
                not discrete
                and row["kind"] == "mc_plain"
                and not row["step_log_variance_recorded"]
            ):
                problems.append(
                    f"{row['engine']}: bridge armed under continuous "
                    "monitoring but recorded no step variance"
                )
    return (not problems), problems


# --------------------------------------------------------------------------
# 2. numerical, against banked evidence


def reproduce_cell(s16, cell: dict) -> dict:
    """Re-run the deterministic candidate for one banked cell."""
    variant = cell["variant"]
    spec = cell["case"]
    case = next(
        c for c in s16.certification_cases(quick=False) if c.name == spec["name"]
    )
    product = s16.make_snowball(case, dense_ki=True)
    env = s16.make_environment(
        case.spot, math.sqrt(max(case.params.v0, case.params.theta))
    )
    leverage = (
        s16.make_leverage_surface(case.maturity) if variant == "heston_slv" else None
    )
    ladders = s16.grid_ladders(
        case.maturity, quick=False, dense_ki_stencil=(case.name == "near_ki")
    )
    target = ladders["target"]
    banked_grid = cell["target_grid"]
    if (target.n_x, target.n_v, target.n_t) != (
        banked_grid["n_x"],
        banked_grid["n_v"],
        banked_grid["n_t"],
    ):
        raise ValueError(
            f"{variant}/{case.name}: target grid drifted from the banked one"
        )

    started = time.perf_counter()
    engine = s16.make_pde_engine(variant, case, target, leverage)
    row = s16.central_bump_greeks(engine, product, env, s16.SPOT_BUMP)
    elapsed = time.perf_counter() - started

    result = {
        "variant": variant,
        "case": case.name,
        "grid": {"n_x": target.n_x, "n_v": target.n_v, "n_t": target.n_t},
        "seconds": elapsed,
        "ki_first_passage_built": getattr(engine, "_ki_fp", None) is not None,
        "greeks": {},
        "bitwise": True,
    }
    for greek in ("delta", "gamma"):
        banked = float(cell["certifications"][greek]["pde"])
        recomputed = float(row[greek])
        identical = recomputed == banked
        result["greeks"][greek] = {
            "banked": banked,
            "recomputed": recomputed,
            "bitwise_equal": identical,
            "abs_diff": abs(recomputed - banked),
        }
        result["bitwise"] = result["bitwise"] and identical
    return result


# --------------------------------------------------------------------------
# 3. anchors


def anchor_report(s16, banked_anchors: list[dict]) -> dict:
    started = time.perf_counter()
    recomputed = s16.deterministic_anchors(quick=False)
    elapsed = time.perf_counter() - started

    by_name = {row["name"]: row for row in banked_anchors}
    rows = []
    for row in recomputed:
        name = row["name"]
        old = by_name.get(name)
        statuses = [
            check["status"]
            for group in _iter_check_groups(row)
            for check in group.values()
        ]
        rows.append(
            {
                "name": name,
                "status_now": row.get("status"),
                "status_banked": None if old is None else old.get("status"),
                "all_checks_pass": all(s == "PASS" for s in statuses),
                "moved": _anchor_moved(row, old),
                "movement": _anchor_movement(row, old),
            }
        )
    return {"seconds": elapsed, "rows": rows}


def _iter_check_groups(row: dict):
    checks = row.get("checks", {})
    if checks and all(isinstance(v, dict) and "status" in v for v in checks.values()):
        yield checks
        return
    for group in checks.values():
        if isinstance(group, dict):
            yield group


def _numeric_leaves(node, prefix=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _numeric_leaves(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield prefix, float(node)


def _anchor_moved(row: dict, old: dict | None) -> bool | None:
    if old is None:
        return None
    now = dict(_numeric_leaves(row.get("pde", {})))
    was = dict(_numeric_leaves(old.get("pde", {})))
    return any(now.get(k) != was.get(k) for k in set(now) | set(was))


def _anchor_movement(row: dict, old: dict | None) -> dict:
    if old is None:
        return {}
    now = dict(_numeric_leaves(row.get("pde", {})))
    was = dict(_numeric_leaves(old.get("pde", {})))
    return {
        key: {"banked": was.get(key), "now": now.get(key)}
        for key in sorted(set(now) | set(was))
        if now.get(key) != was.get(key)
    }


# --------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="restrict the cell reproduction to these case names",
    )
    parser.add_argument(
        "--skip-cells", action="store_true", help="structural and anchors only"
    )
    parser.add_argument("--skip-anchors", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    s16 = load_stage16()
    payload = json.loads(args.evidence.read_text())
    report: dict = {"evidence": str(args.evidence)}

    print("== 1. structural: is the KI machinery built at all? ==", flush=True)
    structural = structural_report(s16)
    ok, problems = structural_verdict(structural)
    report["structural"] = {"rows": structural, "ok": ok, "problems": problems}
    for row in structural:
        detail = (
            f"first_passage={row['ki_first_passage_built']}"
            if row["kind"] == "pde"
            else f"bridge={row['bridge_wanted']} "
            f"variance={row['step_log_variance_recorded']}"
        )
        print(
            f"   {row['monitoring']:<14} {row['kind']:<13} "
            f"{row['engine']:<32} {detail}",
            flush=True,
        )
    print(f"   -> {'OK' if ok else 'PROBLEMS: ' + '; '.join(problems)}\n", flush=True)

    if not args.skip_cells:
        print("== 2. numerical: do the banked cells reproduce bitwise? ==", flush=True)
        cells = payload["cells"]
        if args.cases:
            cells = [c for c in cells if c["case"]["name"] in set(args.cases)]
        results = []
        for cell in cells:
            row = reproduce_cell(s16, cell)
            results.append(row)
            flags = " ".join(
                f"{greek}=" + ("==" if v["bitwise_equal"] else f"DIFF {v['abs_diff']:.3e}")
                for greek, v in row["greeks"].items()
            )
            print(
                f"   {row['variant']:<12} {row['case']:<18} "
                f"{row['seconds']:7.1f}s  {flags}",
                flush=True,
            )
        report["cells"] = results
        report["cells_all_bitwise"] = all(r["bitwise"] for r in results)
        print(
            f"   -> {sum(r['bitwise'] for r in results)}/{len(results)} cells "
            "reproduce bitwise\n",
            flush=True,
        )

    if not args.skip_anchors:
        print("== 3. anchors: continuous KI, so movement is expected ==", flush=True)
        anchors = anchor_report(s16, payload.get("anchors", []))
        report["anchors"] = anchors
        for row in anchors["rows"]:
            print(
                f"   {row['name']:<40} banked={row['status_banked']} "
                f"now={row['status_now']} moved={row['moved']}",
                flush=True,
            )
        print(f"   ({anchors['seconds']:.1f}s)\n", flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"wrote {args.out}", flush=True)

    failed = not report["structural"]["ok"] or not report.get("cells_all_bitwise", True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
