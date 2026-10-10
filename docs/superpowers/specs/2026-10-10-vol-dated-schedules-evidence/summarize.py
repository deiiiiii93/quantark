"""Diff two ``measure.py`` outputs into ``moved_values.txt``: every value that moved, with the test that pins it.

    python summarize.py <base.txt> <branch.txt> > moved_values.txt

Exit 1 when a NUMERIC-form value moved, or a dated value moved other than "raised -> value" (W12's only expected
movement).
"""
from __future__ import annotations

import sys

PIN = {
    "autocall": "test_vol_dated_schedules.py::test_2d_autocallable_dated_twin_price / _greeks",
    "stats": "test_vol_dated_schedules.py::test_2d_snowball_dated_twin_event_stats",
    "barrier": "test_vol_dated_schedules.py::test_vol_barrier_dated_twin_price / _greeks",
    "carried_before": "test_expiry_carried_knock_in.py::test_a_carried_knock_in_the_day_before_is_on_the_settled_branch"
                      " / test_a_carried_autocallable_knock_in_the_day_before_is_on_the_settled_branch",
}


def load(path):
    out = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        cell, form, quantity, value = (part.strip() for part in line.split(" | ", 3))
        out[(cell, form, quantity)] = value
    return out


def main(base_path, branch_path):
    base, branch = load(base_path), load(branch_path)
    assert base.keys() == branch.keys(), "the two trees measured different cells"
    moved, bad = [], []
    for key in base:
        old, new = base[key], branch[key]
        if old == new:
            continue
        moved.append((key, old, new))
        expected = key[1] == "dated" and old.startswith("RAISES") and not new.startswith("RAISES")
        if not expected:
            bad.append((key, old, new))
    print("# Moved values: the vol-model routes resolve dated observation schedules (W12)")
    print(f"# measure.py on 0d1e4133 (git archive) and the branch; summarize.py. {len(base)} values measured, "
          f"{len(moved)} moved, {len(bad)} unexpected.")
    print("# Expected movement: DATED form only, raised -> value. Every numeric value is unmoved (listed below as "
          "the count of unmoved numeric values).")
    print(f"# unmoved numeric values: {sum(1 for k in base if k[1] == 'numeric' and base[k] == branch[k])} of "
          f"{sum(1 for k in base if k[1] == 'numeric')}")
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
