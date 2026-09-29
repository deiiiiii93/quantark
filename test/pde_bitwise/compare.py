"""Diff two capture_book JSONs (spec §8.1). Exit 1 on any unexpected diff."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("branch")
    ap.add_argument("--allow", action="append", default=[],
                    help="row-id prefix whose diffs are reported, not failed")
    a = ap.parse_args(argv)
    with open(a.base) as fh:
        base = json.load(fh)
    with open(a.branch) as fh:
        branch = json.load(fh)
    bad, allowed = [], []
    for row in sorted(set(base) | set(branch)):
        if base.get(row) == branch.get(row):
            continue
        (allowed if any(row.startswith(p) for p in a.allow) else bad).append(row)
    for row in allowed:
        b, n = base.get(row, {}), branch.get(row, {})
        for k in sorted(set(b) | set(n)):
            if b.get(k) != n.get(k) and isinstance(b.get(k), str) and isinstance(n.get(k), str):
                try:
                    fb, fn = float.fromhex(b[k]), float.fromhex(n[k])
                except ValueError:
                    print(f"ALLOWED {row}.{k}: base={b[k]!r} branch={n[k]!r}")
                    continue
                print(f"ALLOWED {row}.{k}: base={fb!r} branch={fn!r} delta={fn - fb:+.3e}")
    for row in bad:
        print(f"DIFF {row}\n  base:   {base.get(row)}\n  branch: {branch.get(row)}")
    print(f"{len(base)} base rows, {len(branch)} branch rows, {len(bad)} unexpected diffs, "
          f"{len(allowed)} allowed diffs")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
