"""Bank the full cross-engine matrix's discretisation allowances and their evidence (spec AC20 full matrix; T24).

For every matrix entry (``matrix_cases.ENTRIES``: contract x model group) and every grid engine of it, the statistics at
the three levels ``L0`` (gated), ``L1`` and ``L2`` (refined twice), and per compared number (``matrix_fields_of``):

- ``L0``, ``L1``, ``L2``: the values (the convergence evidence itself);
- ``allowance``: ``max(|L0 - L1|, |L0 - L2|)`` per row -- the envelope of L0's distance to both refined levels (the
  convergence is not monotone per number on every engine, so ``|L0 - L2|`` alone could understate L0's error, and a
  single refinement difference is not a bound: the MC gate below is the independent check);
- ``refinement``: ``[max|L0 - L1|, max|L1 - L2|]`` (successive steps, recorded);
- ``residuals``: ``probability_residual`` and ``reconciliation_error / max(1, |pv|)`` per level; ``seconds`` per level.

Beside the allowances, every entry banks its MC reference (``mc_reference``: 16 independent PSEUDO runs, mean and
standard error per number, spec AC21) -- the SE the allowance is weighed against in the report; the matrix test
recomputes the reference live.

Run from the worktree root with the banking interpreter (main ``.venv``, py3.11), long runs detached::

    env PYTHONPATH=<wt>:<wt>/test nohup caffeinate -i -m -s \\
        /Users/fuxinyao/quant-ark/.venv/bin/python test/event_statistics/bank_matrix.py [--workers 6] [entry ...]

``entry`` (``name|group``) re-banks only those entries, keeping the rest. Never edit ``matrix_allowances.json`` by
hand; re-bank only for a deliberate engine change, with a field-level diff (D22).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "matrix_allowances.json"


def _setup() -> None:
    for path in (str(HERE.parent), str(HERE.parent.parent)):
        if path not in sys.path:
            sys.path.insert(0, path)
    warnings.filterwarnings("ignore")


def _lists(fields):
    return {path: np.asarray(value, dtype=float).tolist() for path, value in fields.items()}


def _commit() -> str:
    """The banked library's git commit (``+dirty`` when ``quantark/`` differs from it): a later D22 diff's base."""
    root = HERE.parent.parent
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "quantark"], cwd=root,
                           capture_output=True, text=True).stdout.strip()
    return head + ("+dirty" if dirty else "")


def bank_entry(key: str, commit: str = "") -> dict:
    """One entry's record: every grid engine at every level, and the MC reference."""
    _setup()
    from event_statistics import matrix_cases as M

    name, group = key.split("|")
    record = {"engines": {}, "commit": commit, "generated": time.strftime("%Y-%m-%d")}
    if M.CONTRACTS[name].terminated:
        return record
    start = time.perf_counter()
    reference = M.mc_reference(name, group)
    record["mc_reference"] = {"mean": _lists(reference.mean),
                              "se": {k: (None if v is None else np.asarray(v).tolist())
                                     for k, v in reference.se.items()},
                              "runs": reference.runs, "paths": reference.paths,
                              "knock_in_paths": reference.knock_in_paths, "unavailable": reference.unavailable,
                              "seconds": round(time.perf_counter() - start, 3)}
    for engine in M.grid_engines(name, group):
        levels, seconds, residuals = {}, {}, {}
        for level in M.LEVELS:
            t0 = time.perf_counter()
            stats = M.engine_statistics(name, group, engine, level)
            seconds[level] = round(time.perf_counter() - t0, 3)
            levels[level] = M.matrix_fields_of(stats)
            residuals[level] = [float(stats.probability_residual),
                                float(stats.reconciliation_error) / max(1.0, abs(float(stats.pv)))]
        out = {"L0": {}, "L1": {}, "L2": {}, "allowance": {}, "refinement": {}, "seconds": seconds,
               "residuals": residuals}
        for path, value in levels["L0"].items():
            for level in M.LEVELS:
                out[level][path] = np.asarray(levels[level][path], dtype=float).tolist()
            if path in M.AXES:
                continue
            gap01 = np.abs(value - levels["L1"][path])
            gap02 = np.abs(value - levels["L2"][path])
            gap12 = np.abs(levels["L1"][path] - levels["L2"][path])
            out["allowance"][path] = np.maximum(gap01, gap02).tolist()
            out["refinement"][path] = [float(gap01.max()) if gap01.size else 0.0,
                                       float(gap12.max()) if gap12.size else 0.0]
        record["engines"][engine] = out
    return record


def main() -> None:
    _setup()
    from event_statistics import matrix_cases as M

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("entries", nargs="*")
    args = parser.parse_args()
    keys = [f"{n}|{g}" for n, g in M.ENTRIES]
    todo = args.entries or keys
    unknown = set(todo) - set(keys)
    if unknown:
        raise SystemExit(f"unknown entries {sorted(unknown)}")
    bank = json.loads(OUT.read_text()) if (args.entries and OUT.exists()) else {}
    bank["_provenance"] = {"python": platform.python_version(), "machine": platform.machine(),
                           "generated": time.strftime("%Y-%m-%d"), "script": "test/event_statistics/bank_matrix.py",
                           "levels": "L0 gated; L1, L2 refined (matrix_cases.py)",
                           "mc_reference": f"{len(M.SEEDS)} PSEUDO runs x {M.MC_PATHS} paths",
                           "commit": _commit(),
                           "entries": "each entry records its own commit and date; entries without one were banked "
                                      "2026-11-14 on library 448a8f4e (T24)"}
    started = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {pool.submit(bank_entry, key, bank["_provenance"]["commit"]): key for key in todo}
        for future in as_completed(futures):
            key = futures[future]
            bank[key] = future.result()
            print(f"{time.perf_counter() - started:8.1f}s {key}", flush=True)
    ordered = {k: bank[k] for k in ["_provenance"] + [k for k in keys if k in bank]}
    OUT.write_text(json.dumps(ordered, sort_keys=False, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({len(ordered) - 1} entries) in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
