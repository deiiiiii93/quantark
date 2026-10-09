"""Fold today_e4271b6d.txt and after_branch.txt (``measure.py`` output) into one row per (family, engine).

    python docs/superpowers/specs/2026-10-09-dated-expiry-settles-evidence/summarize.py
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    rows = OrderedDict()
    for line in (HERE / name).read_text().splitlines()[2:]:
        family, engine, spot, dated, numeric, settled, same, exact = line.split(" | ")
        rows.setdefault((family, engine), []).append((spot, dated, numeric, settled, same, exact))
    return rows


def status(cells, column):
    values = [c[column] for c in cells]
    raises = sorted({v.split(":")[0].replace("RAISES ", "") + ": " + v.split(": ", 1)[1][:48]
                     for v in values if v.startswith("RAISES")})
    if raises:
        return "refuses (" + "; ".join(raises) + ")"
    return "prices"


def main():
    today, after = load("today_e4271b6d.txt"), load("after_branch.txt")
    print("| family | engine | spots | today: dated | today: numeric T=0 | numeric == settled today | "
          "branch: dated == numeric == settled |")
    print("|---|---|---|---|---|---|---|")
    for key, cells in today.items():
        new = after[key]
        numeric_ok = "yes" if all(c[5] == "yes" for c in cells) else (
            "n/a" if all(c[2].startswith("RAISES") for c in cells) else
            "NO: " + ", ".join(f"{c[0]}: {c[2]} vs {c[3]}" for c in cells if c[5] != "yes"))
        branch = "yes" if all(c[4] == "yes" and c[5] == "yes" for c in new) else status(new, 1) + " / numeric " + \
            status(new, 2)
        print(f"| {key[0]} | {key[1]} | {', '.join(c[0] for c in cells)} | {status(cells, 1)} | "
              f"{status(cells, 2)} | {numeric_ok} | {branch} |")


if __name__ == "__main__":
    main()
