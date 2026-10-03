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
        """The step's Dirichlet row pair, one value per column of the hstacked surfaces (legacy: carried)."""
        return v_next[0, :].copy(), v_next[-1, :].copy()

    def store(self, states, out, lower, upper):
        """Write the solve's interior into fresh state arrays (legacy: the carried rows stay)."""
        col = 0
        for name in self.group.surfaces:
            width = states[name].shape[1]
            new = states[name].copy()                   # legacy: v_cur = v_next.copy()
            new[1:-1, :] = out[:, col:col + width]
            states[name] = new
            col += width


def create_statistics(prepared, streams):
    """The statistics group for ``prepared`` in its family's mode, or ``None`` without a knock-out observation."""
    adapter = prepared.adapter
    if adapter.statistics_mode == "legacy":
        from .legacy import StatisticsLegacy
        return StatisticsLegacy.create(prepared, streams)
    return adapter.statistics_family(prepared, streams)
