"""Bank the cash-leg baseline (unified event statistics spec D9) into ``baseline.json``.

Run from the worktree root, at the base commit, with the worktree first on the path:

    env PYTHONPATH=$PWD:$PWD/test <venv>/bin/python test/cashleg/event_stats_baseline/generate.py

Every (engine, product) case in ``cases.CASES`` banks ``price``, ``event_npv``, and per leg the
event-route value, the folded TOTAL and the folded DRILLDOWN value (expected raises recorded by
exception class). The DRILLDOWN is one call over the legs whose TOTAL folded (the case's
``drilldown_set``); a leg outside it banks the outcome of a DRILLDOWN of that leg alone.

A later task that moves a whole engine family on purpose re-banks only that family, in place, and
records the commit and its D9 cause in the header's ``rebanks`` list:

    env PYTHONPATH=$PWD:$PWD/test <venv>/bin/python test/cashleg/event_stats_baseline/generate.py \
        --rebank mc --cause f --note "MC statistics on price() paths"
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
from rebank_guard import RebankRefused, merge_rebank  # noqa: E402


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


def _engine_kind_matches(kind: str, families) -> bool:
    """``kind`` (``EngineSpec.kind``, e.g. ``mc_heston``, ``dcn_mc_lv``) belongs to one of ``families``
    (``mc`` matches every Monte Carlo kind, autocallable and DCN)."""
    return any(family in kind.split("_") for family in families)


def rebank(families, cause: str, note: str) -> int:
    """Re-bank, in place, only the cases whose engine kind is in ``families``; every other record and
    case meta is kept byte for byte. The header records the re-bank (commit, cases, D9 cause) and
    every record it moved with old and new outcome; ``rebank_guard`` refuses a moved ``price`` and an
    ``event_npv`` move outside the npv causes, writing nothing."""
    out = HERE / "baseline.json"
    banked = json.loads(out.read_text())
    head = _head()
    case_meta = dict(banked["cases"])
    fresh = []
    rebanked_cases = []
    started = time.perf_counter()
    for case in cases.all_cases():
        if not _engine_kind_matches(case.engine.kind, families):
            continue
        meta, recs = bank_case(case)
        fresh.extend(recs)
        case_meta[case.id] = meta
        rebanked_cases.append(case.id)
        print(f"{case.id:45s} re-banked {len(recs):3d} records", flush=True)
    try:
        records, moved = merge_rebank(banked["records"], fresh, cause)
    except RebankRefused as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    import quantark

    checkout = HERE.parents[2]
    if Path(quantark.__file__).resolve().parent != (checkout / "quantark").resolve():
        print("ERROR: quantark is not imported from this checkout; put it first on PYTHONPATH", file=sys.stderr)
        return 1
    header = dict(banked["header"])
    header.setdefault("rebanks", []).append(dict(
        at_sha=head, on=date.today().isoformat(), machine=platform.machine(), python=platform.python_version(),
        numpy=np.__version__, engine_families=sorted(families), cases=rebanked_cases, cause=cause, note=note,
        moved=moved, seconds=round(time.perf_counter() - started, 1),
    ))
    _write(out, header, case_meta, records)
    print(f"re-banked {len(rebanked_cases)} cases (cause {cause}), {len(moved)} records moved -> {out}")
    return 0


def _write(out: Path, header: dict, case_meta: dict, records: list) -> None:
    # one record per line: a later re-bank diffs record by record
    lines = ["{", f' "header": {json.dumps(header)},', f' "cases": {json.dumps(case_meta, indent=1)},',
             ' "records": [']
    lines += [f"  {json.dumps(r)}," for r in records[:-1]] + [f"  {json.dumps(records[-1])}", " ]", "}"]
    out.write_text("\n".join(lines) + "\n")
    json.loads(out.read_text())                         # the hand-assembled file must parse


def main() -> int:
    if len(sys.argv) > 1:
        # re-bank only some engine families in place:
        #   generate.py --rebank mc --cause f --note "MC statistics on price() paths"
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--rebank", nargs="+", required=True, help="engine families (EngineSpec.kind parts)")
        parser.add_argument("--cause", required=True, help="the D9 cause key of the re-bank")
        parser.add_argument("--note", required=True)
        args = parser.parse_args()
        return rebank(args.rebank, args.cause, args.note)
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
    _write(out, header, case_meta, records)
    n_raise = sum("raises" in r["outcome"] for r in records)
    print(f"banked {len(records)} records ({len(records) - n_raise} values, {n_raise} raises) "
          f"over {len(case_meta)} cases in {elapsed:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
