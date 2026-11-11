"""Re-bank guard for the cash-leg baseline (unified event statistics spec D9).

``generate.py --rebank`` replaces an engine family's records in place. Unchecked, that would let any
move pass under a cause label. The guard:

* refuses a moved ``price`` record: ``price()`` is bitwise across this change (AC4, AC22);
* refuses a moved ``event_npv`` record unless the cause is one that moves the event npv: (f) MC
  statistics on ``price()`` paths, (i) ``npv := price()`` on grid engines;
* returns every record whose outcome changed at all, with its old and new outcome, for the re-bank's
  header entry, so the test can audit the moves.

Moves are detected bitwise: a re-bank runs on the banking machine, so anything that changed is listed.
Stdlib only: the test loads this file by path.
"""
from __future__ import annotations

from typing import List, Tuple

#: D9 causes under which the event route's npv may move
NPV_MOVE_CAUSES = frozenset({"f", "i"})


class RebankRefused(ValueError):
    """A re-bank would absorb a move that its cause cannot explain."""


def merge_rebank(records: List[dict], new_records: List[dict], cause: str) -> Tuple[List[dict], List[dict]]:
    """``records`` with each of ``new_records`` replaced in place (by id), and the list of moves.

    Each move is ``{"id", "route", "old", "new"}`` (outcomes). Raises ``RebankRefused`` for a record
    that is not in the bank, any moved ``price`` record, or a moved ``event_npv`` record under a cause
    outside ``NPV_MOVE_CAUSES``. All refusals are collected before raising.
    """
    positions = {r["id"]: k for k, r in enumerate(records)}
    merged = list(records)
    moved: List[dict] = []
    refused: List[str] = []
    for rec in new_records:
        k = positions.get(rec["id"])
        if k is None:
            refused.append(f"{rec['id']} is not in the bank (re-bank never adds records)")
            continue
        old = merged[k]["outcome"]
        if rec["outcome"] != old:
            if rec["route"] == "price":
                refused.append(f"{rec['id']}: price moved {old} -> {rec['outcome']}")
            elif rec["route"] == "event_npv" and cause not in NPV_MOVE_CAUSES:
                refused.append(f"{rec['id']}: event_npv moved under cause {cause!r} "
                               f"(only {sorted(NPV_MOVE_CAUSES)}): {old} -> {rec['outcome']}")
            moved.append(dict(id=rec["id"], route=rec["route"], old=old, new=rec["outcome"]))
        merged[k] = rec
    if refused:
        raise RebankRefused("re-bank refused:\n  " + "\n  ".join(refused))
    return merged, moved
