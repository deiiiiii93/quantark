"""Curve-node provenance metadata (spec WP3.2).

Purely informational for pricing; risk reports align buckets to CALIBRATED
nodes and disclose (``roles_inferred``) when roles were inferred rather than
supplied. Fallback when metadata is absent (normative): all supplied nodes
are CALIBRATED, interpolated queries are MODEL, queries beyond the last node
are EXTRAPOLATED with ``last_observable_tenor`` defaulting to the last node.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple

from quantark.util.exceptions import ValidationError


class NodeRole(Enum):
    CALIBRATED = "calibrated"
    MODEL = "model"
    EXTRAPOLATED = "extrapolated"


@dataclass(frozen=True)
class NodeRoleInfo:
    roles: Tuple[NodeRole, ...]
    last_observable_tenor: float
    roles_inferred: bool


def resolve_node_roles(
    node_tenors: Sequence[float],
    node_roles: Optional[Sequence[NodeRole]],
    last_observable_tenor: Optional[float],
) -> NodeRoleInfo:
    tenors = [float(t) for t in node_tenors]
    if not tenors:
        raise ValidationError("node_tenors must be non-empty")
    if node_roles is None:
        return NodeRoleInfo(
            roles=(NodeRole.CALIBRATED,) * len(tenors),
            last_observable_tenor=(
                float(last_observable_tenor)
                if last_observable_tenor is not None
                else tenors[-1]
            ),
            roles_inferred=True,
        )
    if len(node_roles) != len(tenors):
        raise ValidationError(
            f"node_roles length {len(node_roles)} != nodes {len(tenors)}"
        )
    return NodeRoleInfo(
        roles=tuple(node_roles),
        last_observable_tenor=(
            float(last_observable_tenor)
            if last_observable_tenor is not None
            else max(
                (t for t, r in zip(tenors, node_roles)
                 if r is NodeRole.CALIBRATED),
                default=tenors[-1],
            )
        ),
        roles_inferred=False,
    )
