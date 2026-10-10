"""One node per instant on a simulation time grid (W17a, approved 2026-10-10).

A Monte Carlo grid is the union of the contract's observation times, a monitoring grid and maturity. Two spellings
of one instant can differ in the last ulp: a dated record's ``(d - a) / 365`` against an aged numeric record's
``d / 365 - a / 365`` against a ``np.linspace`` node ``T * j / n``. A union over bit-identical floats keeps both as
nodes ~1e-16 years apart. The empty step carries no variance, but it adds a column of random draws and re-deals
every one after it, so a dated contract and its aged numeric twin price on two independent samples.

``merge_time_nodes`` sorts the union and collapses every run of nodes closer than ``tol`` into the run's LAST
member; ``node_index`` maps a time onto the node that represents it. A grid with no such run is returned exactly as
the sorted set of its times (bit-identical), and each of its times maps to its own node.

No simulation import: the module is shared by any time-grid builder.
"""
from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np

from .constants import Tolerance


def merge_time_nodes(
    times: Iterable[float], tol: float = Tolerance.ZERO, *, pin: Optional[float] = None
) -> np.ndarray:
    """Sorted grid nodes, one per instant.

    Args:
        times: The times to place on the grid (any order, duplicates allowed).
        tol: Two consecutive nodes closer than ``tol`` are one instant; a run of such nodes (each step below
            ``tol``) becomes its LAST member.
        pin: Optional time that must stay a node exactly (a maturity): a node within ``tol`` of it is replaced by
            ``pin`` itself. With ``pin`` the largest time, the last node is ``pin`` bit for bit.

    Returns:
        A float64 array, strictly increasing, every step at least ``tol``.
    """
    nodes = np.unique(np.asarray(list(times), dtype=float))
    if nodes.size > 1:
        keep = np.empty(nodes.size, dtype=bool)
        keep[:-1] = np.diff(nodes) >= tol
        keep[-1] = True
        if not keep.all():
            nodes = nodes[keep]
    if pin is not None and nodes.size:
        near = np.abs(nodes - float(pin)) < tol
        if near.any():
            nodes[near] = float(pin)
            nodes = np.unique(nodes)
    return nodes


def node_index(
    nodes: np.ndarray, times: Union[float, Iterable[float]], tol: float = Tolerance.ZERO
) -> Union[int, np.ndarray]:
    """Index of the node that represents each time: the first node after ``time - tol``.

    For a time that went into ``merge_time_nodes`` this is the last member of its run (itself when it collided
    with nothing, which is ``np.searchsorted(nodes, time)``). A scalar time gives an ``int``.
    """
    shifted = np.asarray(times, dtype=float) - tol
    index = np.searchsorted(nodes, shifted, side="right")
    return int(index) if np.ndim(index) == 0 else index.astype(int)
