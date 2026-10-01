"""Declarative PDE grid & event layer (spec: 2026-07-27-pde-grid-redesign-design.md).

Solvers declare a :class:`GridRequest`; an engine-owned :class:`GridBinder`
turns it into immutable layouts; a per-solve :class:`EventSchedule` applies
event semantics. One spatial builder, one time builder, no modes.
"""

from quantark.asset.equity.engine.pde.grid.request import GridRequest, MarketSnapshot, TimeRefinement
from quantark.asset.equity.engine.pde.grid.config import (
    ACCURACY_PROFILES,
    GridConfig,
    resolve_config,
)
from quantark.asset.equity.engine.pde.grid.time import TimeLayout, build_time, time_fill
from quantark.asset.equity.engine.pde.grid.space import SpatialLayout, build_space
from quantark.asset.equity.engine.pde.grid.events import (
    DiscreteEvent,
    EventOp,
    EventRecord,
    EventSchedule,
    Nodes,
    apply_discrete_event,
    breach_weights,
    project_between,
    project_piecewise,
)
from quantark.asset.equity.engine.pde.grid.binder import (
    GridBinder,
    GridLayerMixin,
    Layout,
    resolve_bound_layout,
    validate_external_layout,
)

__all__ = [
    "GridBinder",
    "GridLayerMixin",
    "Layout",
    "resolve_bound_layout",
    "validate_external_layout",
    "GridRequest",
    "MarketSnapshot",
    "TimeRefinement",
    "GridConfig",
    "ACCURACY_PROFILES",
    "resolve_config",
    "TimeLayout",
    "build_time",
    "time_fill",
    "SpatialLayout",
    "build_space",
    "EventSchedule",
    "DiscreteEvent",
    "EventOp",
    "EventRecord",
    "Nodes",
    "apply_discrete_event",
    "breach_weights",
    "project_between",
    "project_piecewise",
]
