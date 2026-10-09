"""Measure every dated equity family on its exercise date against its numeric twin at maturity 0.0.

Run from the repository root (worktree), with the tree under test first on the path:

    env PYTHONPATH=<wt>:<wt>/test <venv>/bin/python docs/superpowers/specs/2026-10-09-dated-expiry-settles-evidence/measure.py [label]

Prints one row per (family, engine, spot): the dated value (or the refusal), the numeric twin's value (or refusal),
the hand settled value, and whether dated == numeric (rel 1e-12) and numeric == settled (abs 1e-12 + rel 1e-12).
The inventory is ``FAMILIES`` in test/test_dated_expiry_settles.py.
"""
from __future__ import annotations

import math
import sys
import warnings

from test_dated_expiry_settles import FAMILIES, VALUATION, env_for  # noqa: E402

warnings.simplefilter("ignore")


def attempt(fn):
    try:
        return float(fn()), None
    except Exception as exc:  # the measurement records every refusal verbatim
        return None, f"{type(exc).__name__}: {str(exc).splitlines()[0][:110]}"


def close(a, b):
    return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def fmt(value, error):
    return f"{value:.12g}" if error is None else f"RAISES {error}"


def main(label: str) -> None:
    print(f"# {label}: valuation {VALUATION.date()} == exercise date")
    print("family | engine | spot | dated | numeric(T=0) | settled | dated==numeric | numeric==settled")
    for family in FAMILIES:
        for name, factory in family.engines:
            for spot in family.spots:
                dated = attempt(lambda: factory().price(family.build("dated"), env_for(name, spot)))
                numeric = attempt(lambda: factory().price(family.build("numeric"), env_for(name, spot)))
                settled = family.settled(spot)
                print(" | ".join([family.name, name, f"{spot:g}", fmt(*dated), fmt(*numeric), f"{settled:.12g}",
                                  "yes" if close(dated[0], numeric[0]) else "NO",
                                  "yes" if close(numeric[0], settled) else "NO"]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "measurement")
