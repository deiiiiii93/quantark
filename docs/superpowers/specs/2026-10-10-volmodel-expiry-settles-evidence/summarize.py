"""Diff two ``measure.py`` runs: every (cell, form, quantity) whose value moved, old -> new, with its pinning test.

    python summarize.py today_e9daa6ed.txt after_branch.txt
"""
from __future__ import annotations

import sys

PIN = {
    ("w8", "price"): "test_dated_expiry_settles.py::test_dated_contract_on_its_exercise_date_prices_its_numeric_twins_"
                     "settled_value",
    ("w9", "price"): "test_dated_expiry_settles.py::test_vol_model_route_on_its_exercise_date_prices_its_black_"
                     "scholes_counterparts_settled_value",
    ("w9", "greeks"): "test_dated_expiry_settles.py::test_vol_model_route_greeks_on_its_exercise_date_are_its_black_"
                      "scholes_counterparts",
}


def load(path):
    rows = {}
    with open(path) as fh:
        next(fh)
        for line in fh:
            cell, form, quantity, value = (part.strip() for part in line.rstrip("\n").split(" | ", 3))
            rows[(cell, form, quantity)] = value
    return rows


def main(before_path, after_path):
    before, after = load(before_path), load(after_path)
    assert before.keys() == after.keys(), "the two runs measured different cells"
    moved = [(key, before[key], after[key]) for key in before if before[key] != after[key]]
    kinds = {}
    for key, old, new in moved:
        kind = ("raised -> settled" if old.startswith("RAISES") and not new.startswith("RAISES")
                else "OTHER")
        kinds[kind] = kinds.get(kind, 0) + 1
    print(f"# {len(moved)} of {len(before)} measured values moved: "
          + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())))
    print("# cell | form | quantity | old -> new | pinned by")
    for (cell, form, quantity), old, new in moved:
        pin = PIN[(cell.split(":", 1)[0], quantity)]
        print(f"{cell} | {form} | {quantity} | {old} -> {new} | {pin}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
