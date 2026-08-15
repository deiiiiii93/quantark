"""Does the numerical projection see any difference between two revisions?

The per-cell identity gates reuse on a numerical projection: the implementation
inputs with an explicitly declared set of non-numerical symbols removed. That
turns "may I resume across this commit?" into a computation instead of a
judgement call, and this probe performs it.

It materialises both revisions' implementation inputs into temporary trees with
identical relative paths -- the digest hashes paths as well as bytes -- projects
each through the CURRENT exemption list, and compares. When the digests differ it
reports which files moved and diffs the projected harness, so the answer is never
just "no".

Exemptions are intersected with each revision's own top-level symbols, because a
symbol added by the newer revision is absent from the older one and
``project_source`` refuses a name it cannot find. That intersection is sound only
because the exemption list is audited: a numerical function RENAMED into an
exempt one would be hidden by it, so read the reported diff rather than trusting
the verdict alone.

Usage:
    PYTHONPATH=$PWD python docs/adi2d-greek-perf/probes/probe_numerical_projection_equivalence.py \
        --base 350d323 --head HEAD
"""

from __future__ import annotations

import argparse
import ast
import difflib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from quantark.validation.cell_identity import project_source, source_projection_sha256

ROOT = Path(__file__).resolve().parents[3]
HARNESS = "example/mo_volmodels/16_adi_greek_certification.py"
OUTPUT_DIR = ROOT / "output" / "projection_equivalence"


def load_stage16():
    spec = importlib.util.spec_from_file_location("s16_proj", ROOT / HARNESS)
    module = importlib.util.module_from_spec(spec)
    sys.modules["s16_proj"] = module
    spec.loader.exec_module(module)
    return module


WORKTREE = "worktree"


def blob_at(revision: str, relative: str) -> Optional[bytes]:
    """Bytes of one input at a revision, or from the working tree.

    ``worktree`` reads the live files and is the DEFAULT for --head, because the
    committed HEAD is the wrong thing to compare when the change under test is
    still uncommitted. Defaulting to HEAD here produced a false clearance: the
    2026-08-14 migration proved 350d323 == 817bdee while the run that consumed
    its result was the uncommitted tree, whose projection differed. It stamped
    anyway; had it compared the right pair it would have refused.
    """
    if revision == WORKTREE:
        path = ROOT / relative
        return path.read_bytes() if path.exists() else None
    result = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{revision}:{relative}"],
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else None


def top_level_symbols(source: bytes) -> set[str]:
    """Every name a revision defines at module level, definitions and constants.

    Constants count because the exemption list names the provenance bookkeeping
    ones; omitting them here would leave them in the projection on one side and
    drop them on the other, manufacturing a difference out of the intersection.
    """
    try:
        tree = ast.parse(source.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError):
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def materialize(revision: str, relatives: Sequence[str], destination: Path) -> list[str]:
    """Write each input at `revision` into `destination`, preserving paths."""
    missing = []
    for relative in relatives:
        blob = blob_at(revision, relative)
        if blob is None:
            missing.append(relative)
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
    return missing


def digest(tree: Path, relatives: Sequence[str], exempt: Sequence[str]) -> str:
    inputs = []
    for relative in relatives:
        path = tree / relative
        applicable: tuple[str, ...] = ()
        if relative == HARNESS:
            present = top_level_symbols(path.read_bytes())
            applicable = tuple(name for name in exempt if name in present)
        inputs.append((path, applicable))
    return source_projection_sha256(inputs, root=tree)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="350d323", help="revision that banked evidence")
    parser.add_argument(
        "--head",
        default=WORKTREE,
        help=(
            "revision that wants to resume; defaults to the live working tree, "
            "because comparing committed HEAD silently clears uncommitted changes"
        ),
    )
    args = parser.parse_args(argv)

    module = load_stage16()
    exempt = tuple(module.NON_NUMERICAL_SYMBOLS)
    relatives = [
        relative
        for relative in module.IMPLEMENTATION_INPUTS
        if relative not in module.NUMERICAL_EXEMPT_INPUTS
    ]
    print(f"numerical inputs: {len(relatives)}   exempt symbols: {len(exempt)}")
    print(f"base {args.base}   head {args.head}\n")

    with tempfile.TemporaryDirectory() as raw:
        scratch = Path(raw)
        base_tree, head_tree = scratch / "base", scratch / "head"
        for revision, tree in ((args.base, base_tree), (args.head, head_tree)):
            missing = materialize(revision, relatives, tree)
            if missing:
                print(f"  {revision}: MISSING inputs {missing}")
                print(
                    "\nVERDICT: the revisions do not share an input list, so the "
                    "projection cannot be compared."
                )
                return 2

        base_digest = digest(base_tree, relatives, exempt)
        head_digest = digest(head_tree, relatives, exempt)
        print(f"  base projection {base_digest[:16]}")
        print(f"  head projection {head_digest[:16]}")

        differing = []
        for relative in relatives:
            one = (base_tree / relative).read_bytes()
            two = (head_tree / relative).read_bytes()
            if one == two:
                continue
            if relative == HARNESS:
                projected_one = project_source(
                    one.decode("utf-8"),
                    exempt=[n for n in exempt if n in top_level_symbols(one)],
                )
                projected_two = project_source(
                    two.decode("utf-8"),
                    exempt=[n for n in exempt if n in top_level_symbols(two)],
                )
                if projected_one == projected_two:
                    differing.append((relative, "raw bytes differ, projection identical"))
                    continue
                differing.append((relative, "PROJECTION DIFFERS"))
                diff = list(
                    difflib.unified_diff(
                        projected_one.splitlines(),
                        projected_two.splitlines(),
                        fromfile=f"{args.base}:{relative} (projected)",
                        tofile=f"{args.head}:{relative} (projected)",
                        lineterm="",
                        n=2,
                    )
                )
                print(f"\n  projected diff in {relative} ({len(diff)} diff lines):")
                for line in diff[:60]:
                    print(f"    {line}")
                if len(diff) > 60:
                    print(f"    ... and {len(diff) - 60} more")
            else:
                differing.append((relative, "PROJECTION DIFFERS (whole-file input)"))

        print("\n  per-input status:")
        for relative, status in differing:
            print(f"    {status:<42} {relative}")
        if not differing:
            print("    (every input byte-identical)")

        equivalent = base_digest == head_digest
        print()
        if equivalent:
            print(
                "VERDICT: the projections are IDENTICAL. Every change between these\n"
                "revisions is validation, reporting, or plan-derivation code, so a\n"
                "cell banked at base is numerically attributable to head."
            )
        else:
            print(
                "VERDICT: the projections DIFFER. Something that can change a cell's\n"
                "numbers moved between these revisions; banked cells are not reusable."
            )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        destination = OUTPUT_DIR / "projection_equivalence.json"
        destination.write_text(
            json.dumps(
                {
                    "base": args.base,
                    "head": args.head,
                    "base_projection_sha256": base_digest,
                    "head_projection_sha256": head_digest,
                    "numerical_inputs": list(relatives),
                    "differing_inputs": [
                        {"path": path, "status": status} for path, status in differing
                    ],
                    "projections_identical": bool(equivalent),
                },
                indent=1,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"wrote {destination}")
    return 0 if equivalent else 1


if __name__ == "__main__":
    raise SystemExit(main())
