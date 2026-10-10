"""Diff two ``measure_w18b.py`` outputs into the W18b section of ``moved_values.txt``.

    python summarize_w18b.py <base.txt> <branch.txt> >> moved_values.txt

Exit 1 on an unexpected movement. Expected: only the cells whose records sit at a level L != product.barrier
(``expiry`` / ``payoff`` / ``mid`` / ``next``) move, and only where the spot lies between the two levels ("between")
or the record's cash differs (``payoff``, a knock-out hit). Plain cells (``expiry0`` / ``mid0`` / ``next0``) and the
Black-Scholes PDE reference (``midpde``) never move.
"""
from __future__ import annotations

import sys

PIN = {
    "expiry": "test_barrier_mc_record_level_at_valuation.py::test_the_exercise_date_value_decides_on_the_record_level"
              " / _equals_the_black_scholes_pde",
    "payoff": "test_barrier_mc_record_level_at_valuation.py::test_the_exercise_date_knock_out_pays_the_record_payoff",
    "mid": "test_barrier_mc_record_level_at_valuation.py::test_a_valuation_date_observation_decides_on_its_level_as_"
           "the_pde_does",
    "next": "test_barrier_mc_record_level_at_valuation.py::test_hit_at_valuation_between_observations_reads_the_next_"
            "record",
}


def load(path):
    out = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        cell, form, quantity, value = (part.strip() for part in line.split(" | ", 3))
        out[(cell, form, quantity)] = value
    return out


def expected(key):
    group, rest = key[0].split(":", 1)
    if group in ("expiry", "mid", "next"):
        return rest.endswith("-between")
    if group == "payoff":
        return rest.endswith("-between") or ("_out_" in rest and rest.endswith("-beyond"))
    return False


def main(base_path, branch_path):
    base, branch = load(base_path), load(branch_path)
    assert base.keys() == branch.keys(), "the two trees measured different cells"
    moved = [(k, base[k], branch[k]) for k in base if base[k] != branch[k]]
    bad = [m for m in moved if not expected(m[0])]
    plain = [k for k in base if k[0].split(":")[0] in ("expiry0", "mid0", "next0", "midpde")]
    print()
    print("# ======================================================================================================")
    print("# W18b: the Black-Scholes barrier MC decides at the valuation instant on the schedule record's own level")
    print(f"# measure_w18b.py on ee008e9a (the rebased W18 head, git archive) and the branch; summarize_w18b.py. "
          f"{len(base)} values measured, {len(moved)} moved, {len(bad)} unexpected.")
    print("# Expected: records at L != product.barrier, spot between the levels (or a knock-out paying the record's "
          "4.5): value -> value. Plain contracts and the BS PDE reference never move.")
    print("# The 8 'mid' cells move raised -> value: before, the MC did not see the day-91 record's hit, simulated, and "
          "failed on the zero-length step of a record at the valuation instant ('dt_array must have strictly "
          "positive entries'); the same failure stays on the unmoved 'inside' cells (pre-existing).")
    print(f"# unmoved plain / reference values: {sum(1 for k in plain if base[k] == branch[k])} of {len(plain)}")
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
