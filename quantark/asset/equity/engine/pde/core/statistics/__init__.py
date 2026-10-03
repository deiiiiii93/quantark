"""Event statistics as a column group of the value march (spec 2 §5.4; spec 3B §5.1).

``columns`` lays out a statistics block, ``group`` holds the rolling-group protocol and the
factory that picks a family's mode, ``legacy`` is the 0.8.0 statistics sweep and ``readout``
assembles the dataclass and its ledger.
"""
from .columns import ColumnLayout
from .group import RollingStatistics, create_statistics
from .legacy import StatisticsLegacy

__all__ = ["ColumnLayout", "RollingStatistics", "StatisticsLegacy", "create_statistics"]
