"""Diff the W18b knock-in-refusal measurements (base = 822b9433, the record-level commit) into ``moved_values.txt``.

    python summarize_w18b_ki.py <label>=<base.txt>,<branch.txt> [...] >> moved_values.txt

One pair per measure script (``measure_w18b_ki.py``, and the W18 / W18b ``measure.py`` / ``measure_w18b.py`` re-run,
whose knock-in cells with a record payoff 4.5 are now refused). Exit 1 on an unexpected movement. Expected: a
knock-in cell whose records pay other than ``product.rebate`` (groups ``kipay``, ``terms``, ``payoff``) moves
value -> "a knock-in has one rebate" refusal; nothing else moves.
"""
from __future__ import annotations

import sys

PIN = {
    "kipay": "test_knock_in_record_payoff_refused.py::test_a_knock_in_record_paying_other_than_the_rebate_is_refused",
    "terms": "test_vol_barrier_schedule_level.py::test_a_common_record_payoff_and_level_are_the_terms_priced",
    "payoff": "test_vol_barrier_schedule_level.py::test_a_common_record_payoff_and_level_are_the_terms_priced / "
              "test_barrier_mc_record_level_at_valuation.py::test_the_exercise_date_knock_out_pays_the_record_payoff",
}
REFUSED = "RAISES ValidationError: a knock-in has one rebate"


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
    return group in PIN and "_in_" in rest and not old.startswith("RAISES") and new.startswith(REFUSED)


def main(pairs):
    total = moved_n = 0
    moved, bad = [], []
    for spec in pairs:
        label, paths = spec.split("=", 1)
        base_path, branch_path = paths.split(",")
        base, branch = load(base_path), load(branch_path)
        assert base.keys() == branch.keys(), f"{label}: the two trees measured different cells"
        total += len(base)
        for key in base:
            if base[key] != branch[key]:
                moved_n += 1
                moved.append((label, key, base[key], branch[key]))
                if not expected(key, base[key], branch[key]):
                    bad.append((label, key, base[key], branch[key]))
    print()
    print("# ======================================================================================================")
    print("# W18b: a knock-in whose schedule records pay other than product.rebate is refused on every route "
          "(user ruling 2026-10-10, option (a))")
    print(f"# base 822b9433 (the record-level commit, git archive) vs the branch; summarize_w18b_ki.py over "
          f"{', '.join(p.split('=')[0] for p in pairs)}. {total} values measured, {moved_n} moved, {len(bad)} "
          "unexpected.")
    print("# Expected: knock-in cells with a record payoff 4.5 != rebate: price -> refused (the BS PDE, the LV PDE and "
          "the Heston / SLV PDE deterministic limit already refused them: unmoved). Nothing else moves.")
    print("# measure | cell | form | quantity | old -> new | pinned by")
    for label, (cell, form, quantity), old, new in moved:
        print(f"{label} | {cell} | {form} | {quantity} | {old} -> {new} | {PIN.get(cell.split(':')[0], '?')}")
    if bad:
        print("# UNEXPECTED:")
        for label, key, old, new in bad:
            print(f"# {label} {key} {old} -> {new}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
