"""The statistics group protocol and its factory (spec 3B §5.1)."""
from __future__ import annotations

__all__ = ["RollingStatistics", "create_statistics"]


class RollingStatistics:
    """One rolling column group of the value march: its states hold one column set, replaced every step.

    A statistics group provides ``group`` (its :class:`ColumnGroup`), ``n_cols`` (columns per
    surface) and the march hooks ``allocate(states)``, ``terminal(states)``,
    ``step_events(states, captures, j)`` and ``read(states, captures, pv)``; the two below are
    the step's rows and the store of the solve.
    """

    group = None
    n_cols = 0

    def rows(self, j, v_next):
        """The step's Dirichlet row pair, one value per column of the hstacked surfaces."""
        raise NotImplementedError

    def store(self, states, out, lower, upper):
        """Write the solve's interior and the step's rows into the states."""
        raise NotImplementedError


def create_statistics(prepared, streams):
    """The family's statistics group for ``prepared``, or ``None`` without a knock-out observation."""
    return prepared.adapter.statistics_family(prepared, streams)
