"""Diff two ``measure.py`` outputs into ``moved_values.txt``: every value that moved, with the test that pins it.

    python summarize.py <base.txt> <branch.txt> > moved_values.txt

Exit 1 on an unexpected movement. Expected (W18): ``level`` / ``terms`` / ``greeks`` value -> value; ``payoff``
value -> value on a knock-out only; ``agg`` / ``agglimit`` value -> refused. ``plain`` and ``nolevel`` never move.
"""
from __future__ import annotations

import sys

PIN = {
    "level": "test_vol_barrier_schedule_level.py::test_a_common_record_level_is_the_level_priced",
    "terms": "test_vol_barrier_schedule_level.py::test_a_common_record_payoff_and_level_are_the_terms_priced",
    "payoff": "test_vol_barrier_schedule_level.py::test_a_common_record_payoff_and_level_are_the_terms_priced "
              "(the knock-out reads the record payoff)",
    "greeks": "test_vol_barrier_schedule_level.py::test_greeks_follow_the_common_record_level",
    "agg": "test_vol_barrier_schedule_level.py::test_a_non_first_hit_aggregation_is_refused",
    "agglimit": "test_vol_barrier_schedule_level.py::test_the_deterministic_limit_refuses_a_non_first_hit_aggregation_too",
}


def load(path):
    out = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        cell, form, quantity, value = (part.strip() for part in line.split(" | ", 3))
        out[(cell, form, quantity)] = value
    return out


def expected(key, old, new):
    group, rest = key[0].split(":", 1)
    priced = not old.startswith("RAISES") and not new.startswith("RAISES")
    if group in ("level", "terms", "greeks"):
        return priced
    if group == "payoff":
        return priced and "_out_" in rest
    if group == "agg":
        return not old.startswith("RAISES") and new.startswith("RAISES ValidationError")
    if group == "agglimit":
        # the 1D core behind the limit already refused BEST / WORST ("PDE solver does not support aggregation mode");
        # the route now refuses every non-first-hit mode itself, before the limit is reached
        return new.startswith("RAISES ValidationError") and (
            not old.startswith("RAISES") or old.startswith("RAISES PricingError: PDE solver does not support"))
    return False


def main(base_path, branch_path):
    base, branch = load(base_path), load(branch_path)
    assert base.keys() == branch.keys(), "the two trees measured different cells"
    moved = [(k, base[k], branch[k]) for k in base if base[k] != branch[k]]
    bad = [m for m in moved if not expected(*m)]
    stayed = {g: [k for k in base if k[0].startswith(g + ":") and base[k] == branch[k]] for g in ("plain", "nolevel")}
    total = {g: [k for k in base if k[0].startswith(g + ":")] for g in ("plain", "nolevel")}
    print("# Moved values: the vol-model barrier routes price a discrete schedule at its records' terms, first-hit "
          "only (W18)")
    print(f"# measure.py on 468dc2df (git archive) and the branch; summarize.py. {len(base)} values measured, "
          f"{len(moved)} moved, {len(bad)} unexpected.")
    print("# Expected: level / terms / greeks value -> value; payoff value -> value on knock-outs only; agg value -> "
          "refused; agglimit value -> refused (ACCUMULATE) or refused -> refused (BEST / WORST: the 1D core's "
          "PricingError, now the route's ValidationError).")
    for g in ("plain", "nolevel"):
        print(f"# unmoved {g} values (record level == product.barrier / unset, STOP_FIRST_HIT): "
              f"{len(stayed[g])} of {len(total[g])}")
    unmoved_payoff = [k for k in base if k[0].startswith("payoff:") and base[k] == branch[k]]
    print(f"# unmoved payoff values (knock-ins: the no-hit rebate stays product.rebate): {len(unmoved_payoff)}")
    print("# cell | form | quantity | old -> new | pinned by")
    for (cell, form, quantity), old, new in moved:
        print(f"{cell} | {form} | {quantity} | {old} -> {new} | {PIN[cell.split(':')[0]]}")
    if bad:
        print("# UNEXPECTED:")
        for key, old, new in bad:
            print(f"# {key} {old} -> {new}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
