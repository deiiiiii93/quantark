"""Event statistics as a column group of the value march (spec 2 §5.4; spec 3B §5.1).

``columns`` lays out a statistics block, ``group`` holds the rolling-group protocol and the
factory, ``snowball``/``phoenix``/``ko_reset`` are the families' groups and ``readout``
assembles the dataclass and its ledger.
"""
from .columns import ColumnLayout
from .group import RollingStatistics, create_statistics

__all__ = ["ColumnLayout", "RollingStatistics", "create_statistics"]
