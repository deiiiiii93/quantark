"""Declarative PDE grid & event layer (spec: 2026-07-27-pde-grid-redesign-design.md).

Solvers declare a :class:`GridRequest`; an engine-owned :class:`GridBinder`
turns it into immutable layouts; a per-solve :class:`EventSchedule` applies
event semantics. One spatial builder, one time builder, no modes.
"""

from quantark.asset.equity.engine.pde.grid.request import GridRequest, MarketSnapshot

__all__ = [
    "GridRequest",
    "MarketSnapshot",
]
