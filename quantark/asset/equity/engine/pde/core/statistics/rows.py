"""The value's Dirichlet rows, allocated to the statistics columns (spec 3B §5.1 "Rows are allocated,
not designed"; decision 2).

Each value row is produced by one code branch from one cash term. The statistics send that term to the
column whose readout multiplies it: the maturity leg ``M`` (cash), a knock-out unit ``KO`` (times the
record's payoff), an EXPIRY termination unit ``T``/``U``. For a one-term row ``term.value *
term.operand == row.value`` bit for bit, so the cash columns' rows sum to the value's rows exactly.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

__all__ = ["RowTerm", "EdgeRow", "CERTAIN_KI_BRANCHES", "knocks_in", "knocks_out"]

#: the not-knocked-in rows a discretely observed knock-in makes certain (``_knock_in_side_row``): the knocked-in
#: surface's row, while a KI observation remains and the edge lies beyond every KI level
CERTAIN_KI_BRANCHES = frozenset({"v0_lower_certain_ki", "v0_upper_certain_ki"})


# NamedTuples, not frozen dataclasses: the value builds four of each per step on every route, and a
# frozen dataclass costs 2.4x the construction (0.21 vs 0.09 us measured) -- immutable either way


class RowTerm(NamedTuple):
    """One cash term of a value row, sent to the column whose readout multiplies it."""
    column: str                      # "M", "KO", "T", "U" (KO-reset adds "M_pre", "M_post", "KO_post")
    value: float                     # the column's row: cash for M/U-type, a unit DF(t->settlement) for KO/T
    record_time: Optional[float] = None   # the knock-out record a KO/T unit belongs to
    operand: float = 1.0             # what the readout multiplies the column by (a KO payoff)


class EdgeRow(NamedTuple):
    """A value surface's Dirichlet row at one edge and step, with its allocation."""
    value: float                     # exactly what the value surface receives
    branch: str                      # the code branch that produced it (test_rows enumerates them)
    terms: tuple = ()                # sum(t.value * t.operand) == value, bit for bit, for one-term rows
    copy_from_ki: bool = False       # continuous KI: the alive knock-in-side row is the knocked-in row


def knocks_in(edge: EdgeRow) -> bool:
    """Whether the price's not-knocked-in row knocks in (unified event statistics D23: the statistics describe the
    price's own march): the knocked-in row copied under continuous KI (``copy_from_ki``, set while the edge lies
    beyond the knock-in level) or a discrete knock-in made certain (``CERTAIN_KI_BRANCHES``, set while a KI observation
    remains and the edge lies beyond every level). Any other row has no knock-in: it matures not knocked in."""
    return edge.copy_from_ki or edge.branch in CERTAIN_KI_BRANCHES


def knocks_out(edge: EdgeRow) -> bool:
    """Whether the price's row knocks out: it carries a knock-out term of a record ahead (``_boundary_rows_v0``'s
    knock-out edge). A row without one pays its maturity on its own surface."""
    return any(term.column == "KO" and term.record_time is not None for term in edge.terms)
