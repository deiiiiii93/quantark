"""Column groups and their policies (spec 2 §5.4, §5.5, decision 7).

A column group is exactly what one solve call advances today; the grouping,
the backend and the operation order inside a step decide bit-identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np

__all__ = ["Backend", "BoundaryPolicy", "GroupPolicy", "ColumnGroup", "GroupState",
           "VALUE"]


class Backend(Enum):
    SPARSE_LU = "sparse_lu"
    BANDED = "banded"


class BoundaryPolicy(Enum):
    DIRICHLET = "dirichlet"   # the adapter writes rows 0/-1 of step j before the solve


@dataclass(frozen=True)
class GroupPolicy:
    boundary: BoundaryPolicy
    zero_diffusion_theta_one: bool    # θ=1 on zero-diffusion coefficient sets


VALUE = GroupPolicy(BoundaryPolicy.DIRICHLET, True)


@dataclass(frozen=True)
class ColumnGroup:
    """``surfaces`` name the state arrays this group advances, in right-hand-side column order."""
    name: str
    surfaces: tuple
    backend: Backend
    policy: GroupPolicy
    one_dimensional_rhs: bool = False   # Phoenix banded: a 1-D right-hand side (nrhs=1)
    rolling: bool = False               # states hold ONE column set, replaced every step (statistics)


class GroupState(NamedTuple):
    """What a stepper sees: the group and its columns at step j+1 (read-only).

    A NamedTuple, not a dataclass: the march builds one per group per step."""
    group: ColumnGroup
    v_next: np.ndarray
