"""W11 moved values: base (0d1e4133) vs branch rows of measure.py, with the test that pins each moved row.

Usage: python summarize.py base_0d1e4133.txt after_branch.txt > moved_values.txt
"""
import sys
from collections import Counter

NEW = "test_ko_reset_past_records.py::"
DATED = "test_dated_expiry_settles.py::"
CARRIED = "test_expiry_carried_knock_in.py::"


def load(path):
    rows = {}
    for line in open(path):
        if not line.startswith("#"):
            key, value = line.rstrip("\n").split("\t")
            rows[key] = value
    return rows


def pinned_by(key):
    p = key.split("|")
    if p[0] == "A":
        if p[1] == "ko_reset":
            return NEW + "test_a_dated_ko_reset_with_past_records_prices_like_its_numeric_twin"
        return NEW + "test_a_dated_autocallable_with_past_knock_in_records_prices_like_its_numeric_twin"
    if p[0] == "B":
        if p[3] == "exercise":
            return DATED + "test_dated_contract_on_its_exercise_date_prices_its_numeric_twins_settled_value"
        return "measured only (tracker-clock k/12 twin is not the same contract mid-life); " + NEW + "pins the snapped twin"
    if p[0] == "C":
        if p[3] == "exercise":
            return CARRIED + ("test_quad_v2_settles_a_knock_in_carried_by_the_lifecycle_state" if p[4] == "state"
                              else "test_an_autocallable_on_its_exercise_date_settles_its_knock_in_state")
        if float(p[5]) < 103.0:
            return CARRIED + "test_a_carried_autocallable_knock_in_the_day_before_is_on_the_settled_branch"
        return "measured only (the day-before branch test takes spots below the 103 KO)"
    raise ValueError(key)


base, branch = load(sys.argv[1]), load(sys.argv[2])
moved, counts, unchanged = [], Counter(), 0
for key, new in branch.items():
    old = base.get(key, base.get(key.rsplit("|", 1)[0]))
    if old == new:
        unchanged += 1
        continue
    form = next(f for f in ("dated", "numeric") if f"|{f}" in key)
    counts[(key.split("|")[0], form, "raised" if old.startswith("raised") else "value",
            "raised" if new.startswith("raised") else "value")] += 1
    moved.append((key, old, new))

print("# Moved values: W11, a dated autocallable with past knock-in records (2026-10-10)")
print("# measure.py on two trees (git archive of quantark at 0d1e4133; the branch), summarize.py. Rows: section|kind|route|")
print("#   day or date|knock-in|spot|form|greek. A = test_ko_reset_past_records cases (price/delta/gamma by")
print("#   calculate_greeks); B = test_dated_expiry_settles KO_RESET (tracker-clock contract, exercise date and 292 days)")
print("#   C = test_expiry_carried_knock_in KO-reset routes (exercise date and the day before; flag and lifecycle state).")
print(f"# {len(branch)} rows measured on the branch; {unchanged} unchanged bitwise; {len(moved)} moved:")
for (section, form, a, b), n in sorted(counts.items()):
    print(f"#   section {section}, {form}: {a} -> {b}: {n}")
print("# No numeric row moved; no value -> value move; no value -> raised move.")
print("#")
print("# row | old -> new | pinned by")
for key, old, new in moved:
    old_s = "raised (end_date must be after start_date)" if "end_date-after-start_date" in old else old
    print(f"{key} | {old_s} -> {new} | {pinned_by(key)}")
