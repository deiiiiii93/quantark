"""Diff two ``measure.py`` runs: every (cell, form, quantity) whose value moved, old -> new, with its pinning test.

    python summarize.py today_07c799cf.txt after_branch.txt            # every form
    python summarize.py main_6f9c6e74.txt after_branch.txt numeric    # the numeric form only (dated raises on main)
"""
from __future__ import annotations

import sys

_NEW = "test_expiry_carried_knock_in.py::"
_W9 = "test_dated_expiry_settles.py::"
PIN = {
    ("barrier", "price_carried_flag"): _NEW + "test_a_carried_knock_in_settles_the_payoff_on_the_exercise_date",
    ("barrier", "price_carried_state"): _NEW + "test_a_knock_in_carried_by_the_lifecycle_state_settles_the_payoff",
    ("barrier", "price_not_carried"): _NEW + "test_a_knock_in_not_carried_is_decided_by_todays_observation",
    ("greeks", "price"): _NEW + "test_carried_knock_in_greeks_on_the_exercise_date_are_the_payoffs",
    ("greeks", "delta"): _NEW + "test_carried_knock_in_greeks_on_the_exercise_date_are_the_payoffs",
    ("greeks", "gamma"): _NEW + "test_carried_knock_in_greeks_on_the_exercise_date_are_the_payoffs",
    ("w9", "price"): _W9 + "test_vol_model_route_settles_a_carried_knock_in_at_its_payoff",
    ("w9", "price_bsm"): _W9 + "test_vol_model_route_settles_a_carried_knock_in_at_its_payoff",
    ("w9", "greeks_price"): _W9 + "test_vol_model_route_greeks_on_its_exercise_date_are_its_black_scholes_counterparts",
    ("w9", "greeks_delta"): _W9 + "test_vol_model_route_greeks_on_its_exercise_date_are_its_black_scholes_counterparts",
    ("w9", "greeks_gamma"): _W9 + "test_vol_model_route_greeks_on_its_exercise_date_are_its_black_scholes_counterparts",
    ("before", "price_carried_flag"): _NEW + "test_a_carried_knock_in_the_day_before_is_on_the_settled_branch",
    ("autocall", "price"): _NEW + "test_an_autocallable_on_its_exercise_date_settles_its_knock_in_state",
    ("autocall", "price_carried_state"): _NEW + "test_quad_v2_settles_a_knock_in_carried_by_the_lifecycle_state",
    ("autocall", "price_day_before"): _NEW + "test_a_carried_autocallable_knock_in_the_day_before_is_on_the_settled_"
                                             "branch",
}


def load(path, form_filter=None):
    rows = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cell, form, quantity, value = (part.strip() for part in line.rstrip("\n").split(" | ", 3))
            if form_filter is None or form == form_filter:
                rows[(cell, form, quantity)] = value
    return rows


def main(before_path, after_path, form_filter=None):
    before, after = load(before_path, form_filter), load(after_path, form_filter)
    assert before.keys() == after.keys(), "the two runs measured different cells"
    moved = [(key, before[key], after[key]) for key in before if before[key] != after[key]]
    kinds = {}
    for _, old, new in moved:
        kind = ("raised -> value" if old.startswith("RAISES") and not new.startswith("RAISES")
                else "value -> raised" if new.startswith("RAISES") and not old.startswith("RAISES")
                else "value -> value")
        kinds[kind] = kinds.get(kind, 0) + 1
    print(f"# {len(moved)} of {len(before)} measured values moved"
          + (f" ({form_filter} form only)" if form_filter else "") + ": "
          + (", ".join(f"{n} {k}" for k, n in sorted(kinds.items())) or "none"))
    print("# cell | form | quantity | old -> new | pinned by")
    for (cell, form, quantity), old, new in moved:
        print(f"{cell} | {form} | {quantity} | {old} -> {new} | {PIN[(cell.split(':', 1)[0], quantity)]}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
