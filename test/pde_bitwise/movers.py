"""Which byte-book rows each stage-2 flip may move (spec 3A §7.1). Undeclared movers fail the
gate; declared rows that did not move are reported and must be explained in the commit message."""
from __future__ import annotations

import argparse
import sys

#: flip -> the tags whose rows it may move
FLIP_TAGS = {
    "a": {"bus244"},
    "b": {"capped"},
    "c": {"default_projection", "base_discrete"},
    "d": {"snowball_nodal"},
    "e": {"trading_clock_phoenix"},
    "f": {"default_damping"},
    "g": {"put_far_field", "put_far_field_v070", "barrier_call_kinked", "barrier_call_flat"},
}


def _tags(row):
    tags = set(row.tags)
    if {"trading_clock", "phoenix"} <= tags:
        tags.add("trading_clock_phoenix")
    return tags


def movers(flip, rows):
    want = FLIP_TAGS[flip]
    out = sorted(rid for rid, row in rows.items() if _tags(row) & want)
    if flip == "f":
        bad = [rid for rid in out if _tags(rows[rid]) & {"explicit_full_step", "continuous_ki", "v070_base"}]
        assert not bad, f"flip f may not move {bad}"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", nargs=2, metavar=("FLIP", "PATH"), required=True)
    a = ap.parse_args(argv)
    import capture_book as cb
    ids = movers(a.write[0], cb.rows())
    with open(a.write[1], "w") as fh:
        fh.write("\n".join(ids) + "\n")
    print(f"flip {a.write[0]}: {len(ids)} movers -> {a.write[1]}")


if __name__ == "__main__":
    sys.exit(main())
