"""The value's Dirichlet rows, allocated to the statistics columns (spec 3B §5.1 "Rows are allocated,
not designed"; decision 2).

Each value row is produced by one code branch from one cash term. The statistics send that term to the
column whose readout multiplies it: the maturity leg ``M`` (cash), a knock-out unit ``KO`` (times the
record's payoff), an EXPIRY termination unit ``T``/``U``. For a one-term row ``term.value *
term.operand == row.value`` bit for bit, so the cash columns' rows sum to the value's rows exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = ["RowTerm", "EdgeRow"]


@dataclass(frozen=True)
class RowTerm:
    """One cash term of a value row, sent to the column whose readout multiplies it."""
    column: str                      # "M", "KO", "T", "U" (KO-reset adds "M_pre", "M_post", "KO_post")
    value: float                     # the column's row: cash for M/U-type, a unit DF(t->settlement) for KO/T
    record_time: Optional[float] = None   # the knock-out record a KO/T unit belongs to
    operand: float = 1.0             # what the readout multiplies the column by (a KO payoff)


@dataclass(frozen=True)
class EdgeRow:
    """A value surface's Dirichlet row at one edge and step, with its allocation."""
    value: float                     # exactly what the value surface receives
    branch: str                      # the code branch that produced it (test_rows enumerates them)
    terms: tuple = ()                # sum(t.value * t.operand) == value, bit for bit, for one-term rows
    copy_from_ki: bool = False       # continuous KI: the alive lower row is the knocked-in row
