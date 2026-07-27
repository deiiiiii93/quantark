"""Compatibility shim — the certified projection math MOVED to
``quantark.asset.equity.engine.pde.grid.events`` (grid redesign, Phase 0).

This module re-exports the public API for the legacy solvers and tests that
still import it; it is deleted at Phase 4 of the migration along with its
remaining importers. Do not add new imports of this module.
"""

from quantark.asset.equity.engine.pde.grid.events import (  # noqa: F401
    breach_fractions,
    project_breach_jump,
    project_event_values,
    project_piecewise_event,
)

__all__ = [
    "breach_fractions",
    "project_breach_jump",
    "project_event_values",
    "project_piecewise_event",
]
