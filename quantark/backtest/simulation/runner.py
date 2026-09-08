"""Batching the ensemble over path ranges, in process or in a spawn pool (spec 11).

The split is bit-inert -- a path's numbers do not depend on which other
paths share its engine (the plan-2 oracle pinned that) -- so batching is a
memory and parallelism knob, not a numerical one, and the runner test
compares batched and unbatched cubes with ``array_equal``.
"""
from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from quantark.util.exceptions import ValidationError

from .config import EnsembleConfig
from .engine import BOOL_COLUMNS, FLOAT_COLUMNS, EnsembleBacktestEngine, EnsembleResults, StateCube
from .paths.market_path import MarketPath
from .pricing.base import GateReport


def batch_ranges(n_paths: int, batch_paths: Optional[int]) -> List[range]:
    """Consecutive index ranges of at most ``batch_paths`` paths (one range when None)."""
    if n_paths < 1:
        raise ValidationError("n_paths must be positive")
    size = n_paths if batch_paths is None else int(batch_paths)
    if size < 1:
        raise ValidationError("batch_paths must be at least 1")
    return [range(start, min(start + size, n_paths)) for start in range(0, n_paths, size)]


def _run_batch(config: EnsembleConfig, batch: MarketPath) -> EnsembleResults:
    """Module-level so a spawn worker can import it by reference."""
    return EnsembleBacktestEngine(config).run(batch)


def concat_results(parts: Sequence[EnsembleResults], *, path_fingerprint: str) -> EnsembleResults:
    """Stack per-batch results into one, paths renumbered in batch order.

    A batch whose paths all settle early stops its loop there and repeats
    its last active contract over the rest of the calendar, so the batches
    are checked on the contract prefix each one actually ran, and the
    longest run supplies the whole list.
    """
    parts = list(parts)
    if not parts:
        raise ValidationError("concat_results needs at least one batch")
    longest = max(parts, key=lambda p: int(p.manifest["days_run"]))
    contracts = list(longest.cube.active_contract)
    for part in parts:
        ran = int(part.manifest["days_run"])
        if (
            not part.cube.dates.equals(longest.cube.dates)
            or part.cube.active_contract[:ran] != contracts[:ran]
        ):
            raise ValidationError(
                "batches disagree on the calendar or the active contract; they are not one run"
            )
    n_paths = sum(p.n_paths for p in parts)
    cube = StateCube(longest.cube.dates, n_paths)
    cube.active_contract = contracts
    for name in FLOAT_COLUMNS + BOOL_COLUMNS:
        setattr(cube, name, np.concatenate([getattr(p.cube, name) for p in parts], axis=0))
    trades: List[Dict[str, Any]] = []
    events = []
    offset = 0
    for part in parts:
        trades += [{**t, "path": int(t["path"]) + offset} for t in part.trades]
        events += [replace(e, path=e.path + offset) for e in part.events]
        offset += part.n_paths
    manifest = _merge_manifests([p.manifest for p in parts], path_fingerprint)
    return EnsembleResults(
        cube=cube, trades=trades, events=events, manifest=manifest,
        last_day=np.concatenate([p.last_day for p in parts]),
        initial_book_value=np.concatenate([p.initial_book_value for p in parts]),
    )


def _merge_manifests(manifests: Sequence[Dict[str, Any]], path_fingerprint: str) -> Dict[str, Any]:
    head = dict(manifests[0])
    path_meta = {k: v for k, v in dict(head.get("path_meta", {})).items() if k != "path_indices"}
    return {
        **head,
        "path_fingerprint": path_fingerprint,
        "path_meta": path_meta,
        "days_run": max(int(m["days_run"]) for m in manifests),
        "engine_calls": sum(int(m["engine_calls"]) for m in manifests),
        "solves": sum(int(m.get("solves", 0)) for m in manifests),
        "surface_cache": [entry for m in manifests for entry in m.get("surface_cache", [])],
        "dividend_builds": sum(int(m["dividend_builds"]) for m in manifests),
        "data_end_paths": sum(int(m["data_end_paths"]) for m in manifests),
        "seconds": sum(float(m["seconds"]) for m in manifests),
        "gate": GateReport.combine([GateReport(**m["gate"]) for m in manifests]).as_dict(),
        "cache": _sum_counters([m["cache"] for m in manifests]),
        "batches": [dict(m) for m in manifests],
    }


def _sum_counters(dicts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum numeric counters across dicts, recursing into nested dicts; other values keep the last."""
    out: Dict[str, Any] = {}
    for d in dicts:
        if d is None:
            continue
        for key, value in d.items():
            if isinstance(value, dict):
                out[key] = _sum_counters([out.get(key, {}), value]) if key in out else dict(value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                out[key] = out.get(key, 0) + value
            else:
                out[key] = value
    return out


def run_ensemble(config: EnsembleConfig, paths: MarketPath) -> EnsembleResults:
    """Run one cell over ``paths`` in batches of ``config.batch_paths`` on ``config.workers`` processes.

    A worker failure aborts the whole run with the worker's error; no
    partial result is returned.
    """
    ranges = batch_ranges(paths.n_paths, config.batch_paths)
    batches = [paths.take(list(r)) for r in ranges]
    if int(config.workers) == 1 or len(batches) == 1:
        parts = [_run_batch(config, batch) for batch in batches]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=int(config.workers), mp_context=context) as pool:
            futures = [pool.submit(_run_batch, config, batch) for batch in batches]
            parts = []
            for future in futures:
                try:
                    parts.append(future.result())
                except Exception as exc:
                    for other in futures:
                        other.cancel()
                    raise ValidationError(f"a batch worker failed: {type(exc).__name__}: {exc}") from exc
    if len(parts) == 1:
        part = parts[0]
        part.manifest["path_fingerprint"] = paths.fingerprint()
        part.manifest["path_meta"] = dict(paths.meta)
        part.manifest.setdefault("batches", [dict(part.manifest)])
        return part
    result = concat_results(parts, path_fingerprint=paths.fingerprint())
    result.manifest["path_meta"] = dict(paths.meta)
    return result
