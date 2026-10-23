"""Static equity stress tasks on a spawn process pool (ship spec 2026-10-23 D13).

Each task's arguments are pickled on submit, so every task prices a fresh copy of its post-baseline inputs, as the
serial loop's deepcopy does; nothing prepared is shared between tasks or processes. Workers never receive the
config (a progress callback stays in the parent), and this module never changes thread settings.

Spawn re-imports the caller's main module in every worker, so a script that starts a run must guard it with
``if __name__ == "__main__":``. Without the guard each worker would start the run again while bootstrapping; that
attempt is refused, the worker dies, and the parent warns and runs serially.

Pool infrastructure failures (inputs that do not pickle, a pool that cannot start or breaks) return ``None`` after a
``RuntimeWarning`` so the caller runs serially; an exception raised by a task itself is re-raised unchanged.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from quantark.stresstest.equity.valuation import PositionValuation, batch_value_position, value_positions


def scenario_task(task: Tuple[Dict[str, Any], Dict[str, Any], Any, bool]) -> Dict[str, PositionValuation]:
    """One scenario: ``(positions, envs, greeks_calculator, use_analytical)``, each position valued once."""
    positions, envs, greeks_calculator, use_analytical = task
    return value_positions(positions, envs, greeks_calculator, trade=True, use_analytical=use_analytical)


def position_task(task: Tuple[Any, Sequence[Any], Any]) -> List[PositionValuation]:
    """One session-capable position over every scenario: ``(position, envs, greeks_calculator)``."""
    position, envs, greeks_calculator = task
    return batch_value_position(position, envs, greeks_calculator)


def _call(packed: Tuple[Callable[[Any], Any], Any]) -> Tuple[int, bool, Any]:
    """Run one task in a worker; a task's own exception travels back as a value."""
    fn, task = packed
    try:
        return os.getpid(), True, fn(task)
    except Exception as exc:  # re-raised in the parent, so it is never mistaken for a pool failure
        return os.getpid(), False, exc


def run_tasks(
    fn: Callable[[Any], Any],
    tasks: Sequence[Any],
    max_workers: int,
) -> Optional[Tuple[List[Any], int]]:
    """``[fn(t) for t in tasks]`` on a spawn pool, in order, with the number of distinct worker processes.

    Returns ``None`` (after a ``RuntimeWarning``) when the pool cannot be used, so the caller runs serially.
    """
    packed = [(fn, task) for task in tasks]
    try:
        pickle.dumps(packed)
    except Exception as exc:
        _warn_serial(f"inputs do not pickle ({type(exc).__name__}: {exc})")
        return None
    if getattr(multiprocessing.current_process(), "_inheriting", False):
        # a spawn worker re-running an unguarded main script: die, so the parent's pool breaks and it falls back
        raise RuntimeError("static stress started while a spawn worker bootstraps; guard the script's entry point")
    try:
        with ProcessPoolExecutor(
            max_workers=min(max_workers, len(packed)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            try:
                outcomes = list(pool.map(_call, packed))
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
    except (BrokenProcessPool, RuntimeError, OSError, pickle.PicklingError, AssertionError) as exc:
        # AssertionError: a daemonic process (e.g. a multiprocessing.Pool worker) may not start children
        _warn_serial(f"the process pool failed ({type(exc).__name__}: {exc})")
        return None
    for _pid, ok, value in outcomes:
        if not ok:
            raise value
    return [value for _pid, _ok, value in outcomes], len({pid for pid, _ok, _value in outcomes})


def _warn_serial(reason: str) -> None:
    warnings.warn(f"static stress parallel_execution: {reason}; running serial", RuntimeWarning, stacklevel=3)
