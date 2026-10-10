"""Diff two ``measure_moved.py`` outputs: every key whose value changed, grouped by section, plus unchanged counts.

    python summarize.py OLD.txt NEW.txt "label"
"""
from __future__ import annotations

import sys
from collections import defaultdict


def load(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            parts = [p.strip() for p in line.rstrip("\n").split(" | ", 2)]
            if len(parts) == 3:
                out[(parts[0], parts[1])] = parts[2]
    return out


def main(old_path, new_path, label):
    old, new = load(old_path), load(new_path)
    if set(old) != set(new):
        raise SystemExit(f"key sets differ: {sorted(set(old) ^ set(new))[:5]}")
    moved, same = defaultdict(list), defaultdict(int)
    for key in sorted(old):
        if old[key] == new[key]:
            same[key[0]] += 1
        else:
            moved[key[0]].append((key[1], old[key], new[key]))
    print(f"# {label}: {sum(len(v) for v in moved.values())} moved of {len(old)} measured")
    for section in sorted(set(same) | set(moved)):
        print(f"## section {section}: {len(moved[section])} moved, {same[section]} unchanged")
        for key, a, b in moved[section]:
            print(f"{key} | {a} -> {b}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
