"""Which byte-book rows each stage-2 flip may move (spec 3A §7.1). Undeclared movers fail the
gate; declared rows that did not move are reported and must be explained in the commit message."""
from __future__ import annotations

import argparse
import json
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


#: 3B stage -> how its movers are chosen (spec 3B §7.1 gates)
STAGE_3B = {
    "a": {"family": "snowball"},
    "b": {"family": "phoenix"},
    "c": {"family": "ko_reset"},
    "d": {"tags": {"valuation_rules"}},
    "e": {"tags": {"twod_default"}},
    "f": {"tags": {"release_rule"}},
    "g": {"tags": {"accrual"}},
}


def _carries_statistics(fields):
    return any(k.startswith(("stats_", "dist_")) for k in fields)


def phoenix_kind(fields):
    """``plain`` (non-memory INSTANT), ``memory`` or ``expiry`` from a row's previous capture, else ``None``.

    The legacy statistics reported no coupon cash for a memory coupon, and their EXPIRY ledger paid every
    coupon at the terminal payment; a plain coupon pays at its own observation."""
    if "stats_ko_times" not in fields or "stats_payment_times" not in fields:
        return None
    cash = fields.get("stats_expected_discounted_coupon_cashflow")
    if not isinstance(cash, list) or not cash:
        return "memory"
    n = len(fields["stats_ko_times"])
    pay = fields["stats_payment_times"]
    coupon_pay, terminal = pay[n:2 * n], pay[-1]
    return "expiry" if all(t == terminal for t in coupon_pay) else "plain"


def movers_3b(stage, rows, capture, kind=None):
    """Data-driven: a family stage moves every row of the family whose PREVIOUS capture carries statistics
    or distribution fields; a rule stage moves its tagged rows."""
    import capture_book as cb
    spec = STAGE_3B[stage]
    if "family" in spec:
        out = sorted(rid for rid in rows
                     if cb.family_tag(rid) == spec["family"] and _carries_statistics(capture.get(rid, {})))
        if kind is not None:
            out = [rid for rid in out if phoenix_kind(capture[rid]) == kind]
        return out
    return sorted(rid for rid, row in rows.items() if row.tags & spec["tags"])


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
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", nargs=2, metavar=("FLIP", "PATH"), help="3A flip movers")
    mode.add_argument("--stage", choices=sorted(STAGE_3B), help="3B stage movers (needs --capture and PATH)")
    ap.add_argument("--capture", default=None, help="the previous commit's capture (3B family stages)")
    ap.add_argument("--kind", choices=("plain", "memory", "expiry"), default=None,
                    help="stage b: only the Phoenix movers of this kind (phoenix_kind)")
    ap.add_argument("path", nargs="?", default=None)
    a = ap.parse_args(argv)
    import capture_book as cb
    if a.write:
        ids = movers(a.write[0], cb.rows())
        with open(a.write[1], "w") as fh:
            fh.write("\n".join(ids) + "\n")
        print(f"flip {a.write[0]}: {len(ids)} movers -> {a.write[1]}")
        return 0
    if not (a.capture and a.path):
        ap.error("--stage needs --capture PREV.json and PATH")
    with open(a.capture) as fh:
        capture = json.load(fh)
    ids = movers_3b(a.stage, cb.rows(), capture, a.kind)
    with open(a.path, "w") as fh:
        fh.write("\n".join(ids) + "\n")
    print(f"stage {a.stage}: {len(ids)} movers -> {a.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
