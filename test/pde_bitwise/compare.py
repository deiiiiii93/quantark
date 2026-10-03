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


def _tagged(x) -> bool:
    """The capture's non-numeric outputs (spec 3B §7.1): ``none``, ``bool:0/1``, ``repr:...``."""
    return isinstance(x, str) and (x == "none" or x.startswith("bool:") or x.startswith("repr:"))


def _finite_hex(v) -> bool:
    try:
        vals = v if isinstance(v, list) else [v]
        return all(_tagged(x) or (isinstance(x, str) and math.isfinite(float.fromhex(x))) for x in vals)
    except (TypeError, ValueError):
        return False


def _same(b, n, added_globs):
    """Rows equal up to branch-only keys a glob allows (F3); a base-only key is always a diff."""
    if b == n:
        return True
    if b is None or n is None or "__error__" in b or "__error__" in n:
        return False
    if set(b) - set(n):
        return False
    extra = set(n) - set(b)
    if any(not any(fnmatch.fnmatchcase(k, g) for g in added_globs) for k in extra):
        return False
    return all(b[k] == n[k] for k in b)


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
    ap.add_argument("--bound-field", action="append", default=[], metavar="ROWGLOB:FIELDGLOB=ABS",
                    help="on allowed rows matching ROWGLOB, fields matching FIELDGLOB may move by at most ABS (absolute)")
    ap.add_argument("--movable-fields", action="append", default=[], metavar="FIELDGLOB",
                    help="on allowed rows, a differing field that matches none of these globs is a violation")
    ap.add_argument("--allow-added-key", action="append", default=[], metavar="KEYGLOB",
                    help="a key present only on the branch side is not a diff when it matches")
    ap.add_argument("--base-manifest", default=None, help="manifest for the base side (default: --manifest)")
    a = ap.parse_args(argv)
    base = _load(a.base)
    branch = _load(a.branch)
    violations = []
    if a.manifest:
        with open(a.manifest) as fh:
            manifest = json.load(fh)
        base_manifest = manifest
        if a.base_manifest:
            with open(a.base_manifest) as fh:
                base_manifest = json.load(fh)
        violations = (_manifest_violations(base, base_manifest, "base")
                      + _manifest_violations(branch, manifest, "branch"))
    bad, allowed = [], []
    exact = set()
    if a.allow_file:
        with open(a.allow_file) as fh:
            exact = {line.strip() for line in fh if line.strip()}
    bounds = [(g, float(r)) for g, r in (b.split("=", 1) for b in a.bound)]
    for row in sorted(set(base) | set(branch)):
        if _same(base.get(row), branch.get(row), a.allow_added_key):
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
    field_bounds = []
    for spec in a.bound_field:
        lhs, abs_tol = spec.rsplit("=", 1)
        rglob, fglob = lhs.split(":", 1)
        field_bounds.append((rglob, fglob, float(abs_tol)))
    maxdelta = {}
    for row in allowed:
        b_row, n_row = base.get(row) or {}, branch.get(row) or {}
        for k in sorted(set(b_row) & set(n_row)):
            if b_row[k] == n_row[k] or k in _NON_HEX_KEYS:
                continue
            if a.movable_fields and not any(fnmatch.fnmatchcase(k, g) for g in a.movable_fields):
                bad.append(f"{row}.{k} moved but is not a movable field")
                continue
            for rglob, fglob, tol in field_bounds:
                if not (fnmatch.fnmatchcase(row, rglob) and fnmatch.fnmatchcase(k, fglob)):
                    continue
                vb, vn = b_row[k], n_row[k]
                if isinstance(vb, list) != isinstance(vn, list) or (isinstance(vb, list) and len(vb) != len(vn)):
                    bad.append(f"{row}.{k} changed shape under bound {tol}")
                    continue
                pairs = list(zip(vb, vn)) if isinstance(vb, list) else [(vb, vn)]
                try:
                    d = max(abs(float.fromhex(y) - float.fromhex(x)) for x, y in pairs)
                except (TypeError, ValueError):
                    bad.append(f"{row}.{k} is not numeric under bound {tol}")
                    continue
                if d > maxdelta.get(k, (0.0, ""))[0]:
                    maxdelta[k] = (d, row)
                if d > tol:
                    bad.append(f"{row}.{k} beyond bound {tol} (delta {d:.3e})")
    for k, (d, row) in sorted(maxdelta.items()):
        print(f"MAXDELTA {k} {d:.3e} {row}")
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
