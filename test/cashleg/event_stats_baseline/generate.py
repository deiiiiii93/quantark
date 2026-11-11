"""Bank the cash-leg baseline (unified event statistics spec D9) into ``baseline.json``.

Run from the worktree root, at the base commit, with the worktree first on the path:

    env PYTHONPATH=$PWD:$PWD/test <venv>/bin/python test/cashleg/event_stats_baseline/generate.py

Every (engine, product) case in ``cases.CASES`` banks ``price``, ``event_npv``, and per leg the
event-route value, the folded TOTAL and the folded DRILLDOWN value (expected raises recorded by
exception class). The DRILLDOWN is one call over the legs whose TOTAL folded (the case's
``drilldown_set``); a leg outside it banks the outcome of a DRILLDOWN of that leg alone.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cases  # noqa: E402  (this directory's module; loaded by path in the test)


def _head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _record(case, meta, leg, route, outcome, scale, descriptor=None):
    return dict(
        id=f"{case.id}__{leg or '-'}__{route}",
        case=case.id, engine=meta["engine"], engine_class=meta["engine_class"], engine_kind=meta["engine_kind"],
        product=meta["product"], product_class=meta["product_class"], family=meta["family"],
        features=meta["features"], leg=leg, leg_descriptor=descriptor, route=route, scale=float(scale),
        outcome=outcome,
    )


def _scale(outcome, nominal):
    return max(float(nominal), abs(float(outcome["value"]))) if "value" in outcome else float(nominal)


def bank_case(case):
    meta = case.meta()
    specs = case.legs()
    records = []

    event = cases.event_route(case)
    for route in ("price", "event_npv"):
        out = event[route]
        records.append(_record(case, meta, None, route, out, _scale(out, 0.0)))
    for spec in specs:
        out = event["legs"][spec.name]
        records.append(_record(case, meta, spec.name, "event", out, _scale(out, spec.scale), spec.descriptor))

    totals = {spec.name: cases.fold_total(case, spec.name) for spec in specs}
    for spec in specs:
        out = totals[spec.name]
        records.append(_record(case, meta, spec.name, "fold_total", out, _scale(out, spec.scale), spec.descriptor))

    drilldown_set = [spec.name for spec in specs if "value" in totals[spec.name]]
    combined = cases.fold_drilldown(case, drilldown_set)
    for route in ("total", "npv"):
        out = combined[route]
        records.append(_record(case, meta, None, f"fold_drilldown_{route}", out, _scale(out, 0.0)))
    for spec in specs:
        if spec.name in drilldown_set:
            out = combined["per_leg"][spec.name]
        else:
            out = cases.fold_drilldown(case, [spec.name])["per_leg"][spec.name]
        records.append(_record(case, meta, spec.name, "fold_drilldown", out, _scale(out, spec.scale),
                               spec.descriptor))

    meta = dict(meta, drilldown_set=drilldown_set, event_distribution=event["diagnostics"])
    return meta, records


def main() -> int:
    head = _head()
    if head != cases.BASE_SHA:
        print(f"WARNING: HEAD {head} is not the base {cases.BASE_SHA}", file=sys.stderr)
    started = time.perf_counter()
    case_meta, records = {}, []
    for case in cases.all_cases():
        t0 = time.perf_counter()
        meta, recs = bank_case(case)
        case_meta[case.id] = meta
        records.extend(recs)
        n_raise = sum("raises" in r["outcome"] for r in recs)
        print(f"{case.id:45s} {len(recs):3d} records ({n_raise:3d} raises) {time.perf_counter() - t0:6.2f}s",
              flush=True)
    elapsed = time.perf_counter() - started
    import quantark

    checkout = HERE.parents[2]
    header = dict(
        base_sha=cases.BASE_SHA, generated_at_sha=head, generated_on=date.today().isoformat(),
        machine=platform.machine(), python=platform.python_version(), numpy=np.__version__,
        # the editable install points at another checkout unless this one leads PYTHONPATH
        quantark_from_this_checkout=Path(quantark.__file__).resolve().parent == (checkout / "quantark").resolve(),
        generator="test/cashleg/event_stats_baseline/generate.py", seconds=round(elapsed, 1),
        routes=["price", "event_npv", "event", "fold_total", "fold_drilldown", "fold_drilldown_total",
                "fold_drilldown_npv"],
        unit_notional=cases.UNIT,
    )
    if not header["quantark_from_this_checkout"]:
        print("ERROR: quantark is not imported from this checkout; put it first on PYTHONPATH", file=sys.stderr)
        return 1
    out = HERE / "baseline.json"
    # one record per line: a later re-bank diffs record by record
    lines = ["{", f' "header": {json.dumps(header)},', f' "cases": {json.dumps(case_meta, indent=1)},',
             ' "records": [']
    lines += [f"  {json.dumps(r)}," for r in records[:-1]] + [f"  {json.dumps(records[-1])}", " ]", "}"]
    out.write_text("\n".join(lines) + "\n")
    json.loads(out.read_text())                         # the hand-assembled file must parse
    n_raise = sum("raises" in r["outcome"] for r in records)
    print(f"banked {len(records)} records ({len(records) - n_raise} values, {n_raise} raises) "
          f"over {len(case_meta)} cases in {elapsed:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
