"""Diff two capture_book JSONs (spec §8.1). Exit 1 on any unexpected diff.

With ``--manifest`` (spec 2 decision 20) the gate is a manifest, not a diff:
every manifest row present on both sides, every numerical output a finite hex
on both sides, a row may raise only when the manifest declares it with its
exception type, and an output may be ``None`` only when the manifest declares
it. Two books carrying the same error for a row are equal, so zero diffs alone
proves nothing about preservation.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import math
import sys

# Outputs that are integers or identifiers, not hexes: DCN step counts, the
# session adapter id, the life-surface step counts.
_NON_HEX_KEYS = ("adapter", "steps", "actual_steps", "requested_steps")


def _finite_hex(v) -> bool:
    try:
        vals = v if isinstance(v, list) else [v]
        return all(isinstance(x, str) and math.isfinite(float.fromhex(x)) for x in vals)
    except (TypeError, ValueError):
        return False


def _manifest_violations(book: dict, manifest: dict, side: str) -> list:
    out = []
    for row, decl in manifest.items():
        if row not in book:
            out.append(f"{side}: missing row {row}")
            continue
        got = book[row]
        err = got.get("__error__")
        if decl["error"] is not None:
            if err is None or not err.startswith(decl["error"] + ":"):
                out.append(f"{side}: {row} must raise {decl['error']}, got {err or 'no error'}")
            continue
        if err is not None:
            out.append(f"{side}: {row} raised undeclared {err}")
            continue
        for key, val in got.items():
            if key in decl["none"]:
                if val is not None:
                    out.append(f"{side}: {row}.{key} declared None, got {val!r}")
            elif isinstance(val, (int, str)) and key in _NON_HEX_KEYS:
                continue  # declared integer / identifier outputs
            elif not _finite_hex(val):
                out.append(f"{side}: {row}.{key} is not a finite hex: {val!r}")
    extra = sorted(set(book) - set(manifest))
    out += [f"{side}: row {r} not in manifest" for r in extra]
    return out


def _load(path) -> dict:
    with open(path) as fh:
        book = json.load(fh)
    # books written before the trace moved to stderr carry it per row
    return {k: {kk: vv for kk, vv in v.items() if kk != "__trace__"} for k, v in book.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("branch")
    ap.add_argument("--allow", action="append", default=[],
                    help="row-id prefix whose diffs are reported, not failed")
    ap.add_argument("--manifest", default=None,
                    help="manifest.json: declared rows, None outputs and failures")
    ap.add_argument("--allow-file", default=None, help="exact row ids, one per line, whose diffs are allowed")
    ap.add_argument("--bound", action="append", default=[], metavar="GLOB=REL",
                    help="allowed rows matching GLOB may move by at most REL relative")
    a = ap.parse_args(argv)
    base = _load(a.base)
    branch = _load(a.branch)
    violations = []
    if a.manifest:
        with open(a.manifest) as fh:
            manifest = json.load(fh)
        violations = (_manifest_violations(base, manifest, "base")
                      + _manifest_violations(branch, manifest, "branch"))
    bad, allowed = [], []
    exact = set()
    if a.allow_file:
        with open(a.allow_file) as fh:
            exact = {line.strip() for line in fh if line.strip()}
    bounds = [(g, float(r)) for g, r in (b.split("=", 1) for b in a.bound)]
    for row in sorted(set(base) | set(branch)):
        if base.get(row) == branch.get(row):
            continue
        ok = row in exact or any(row.startswith(p) for p in a.allow)
        (allowed if ok else bad).append(row)
    for row in allowed:
        for glob, rel in bounds:
            if fnmatch.fnmatch(row, glob):
                for k in set(base.get(row, {})) & set(branch.get(row, {})):
                    vb, vn = base[row][k], branch[row][k]
                    if vb == vn or k in _NON_HEX_KEYS:
                        continue
                    pairs = list(zip(vb, vn)) if isinstance(vb, list) and isinstance(vn, list) and len(vb) == len(vn) \
                        else [(vb, vn)]
                    try:
                        far = any(abs(float.fromhex(n) - float.fromhex(b)) > rel * max(1.0, abs(float.fromhex(b)))
                                  for b, n in pairs)
                    except (TypeError, ValueError):
                        far = True   # a shape or type change is not a bounded move
                    if far:
                        bad.append(f"{row}.{k} beyond bound {rel}")
    for row in sorted(exact - set(allowed)):
        print(f"ALLOWED-UNCHANGED {row}")
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
        if row in base or row in branch:
            print(f"DIFF {row}\n  base:   {base.get(row)}\n  branch: {branch.get(row)}")
        else:
            print(f"BOUND {row}")
    for v in violations:
        print(f"MANIFEST {v}")
    print(f"{len(base)} base rows, {len(branch)} branch rows, {len(bad)} unexpected diffs, "
          f"{len(allowed)} allowed diffs, {len(violations)} manifest violations")
    return 1 if (bad or violations) else 0


if __name__ == "__main__":
    sys.exit(main())
