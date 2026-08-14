"""Per-cell provenance identity for certification evidence.

A certification cell is expensive -- the 2026-08-12 stage-16 fleet spent 35.5
hours on fourteen of them -- so the question "may this banked cell be reused?"
has to be answered precisely. Answering it too broadly is not a safe default: it
does not make anything more correct, it just forces a re-run, and a re-run that
is not affordable gets skipped or worked around, which is how provenance
guarantees actually die.

The original guard hashed a fleet-wide configuration dict containing a digest of
whole files, so a comment, a validator fix, or an unrelated cell's sampling plan
invalidated every cell. This module builds a narrower identity from exactly what
can change ONE cell's numbers:

* the **numerical projection** of the implementation -- source with an explicitly
  declared set of non-numerical symbols removed;
* that cell's **own declared plan**;
* the **runtime** the arithmetic ran on;
* the **identities of the cells it consumes**, so a control that moves
  invalidates its consumer.

Two properties make the narrowing trustworthy rather than convenient.

**The default is "numerical".** ``project_source`` removes only symbols named in
the exemption list and refuses a list naming a symbol that is not there. New code
is therefore included and invalidates; renaming an exempt function breaks loudly
instead of quietly widening the exemption.

**Plan-derivation code is exempt because the plan itself is hashed.** Code whose
only route to a cell's numbers is the plan it declares -- argument parsing, the
policy builder, the case table -- is double-counted if its bytes are hashed too,
and that double counting is precisely what made the old guard hypersensitive.
Exempting it is sound ONLY while the plan projection is complete, so the plan
projection is the thing to scrutinise when adding to the exemption list.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "canonical_sha256",
    "cell_identity_sha256",
    "project_source",
    "source_projection_sha256",
]

_EXEMPTABLE = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def canonical_sha256(payload: Any) -> str:
    """Hash a JSON-serialisable payload independently of key order."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def project_source(source: str, *, exempt: Sequence[str]) -> str:
    """Return ``source`` with the named top-level definitions removed.

    Only top-level functions, async functions and classes may be exempted, and
    every name in ``exempt`` must resolve to exactly one of them. A name that
    resolves to nothing is an error rather than a no-op: the common way an
    exemption list goes wrong is a rename, and a silent no-op would widen the
    projection at exactly the moment it should have complained.

    Decorators are removed with the definition they decorate. Module-level
    statements -- imports, constants, dataclass declarations -- always remain,
    because a changed constant is a changed plan.

    Whitespace-only lines are dropped, so the projection is what the file would
    be if the exempt symbols had never existed rather than what is left after
    cutting them out. Without this, a removed definition leaves its PEP8 blank
    separators behind and ADDING an exempt helper shifts the digest -- a diff made
    entirely of blank lines, forcing a full re-run to pay for a validator tweak.
    The cost of the normalisation is that blank lines inside a string literal stop
    counting; no arithmetic in this package depends on them.
    """
    wanted = list(dict.fromkeys(exempt))
    if not wanted:
        return _without_blank_lines(source)

    tree = ast.parse(source)
    spans: dict[str, list[tuple[int, int]]] = {}
    for node in tree.body:
        if isinstance(node, _EXEMPTABLE) and node.name in wanted:
            first = node.lineno
            for decorator in getattr(node, "decorator_list", []) or []:
                first = min(first, decorator.lineno)
            spans.setdefault(node.name, []).append(
                (first, node.end_lineno or node.lineno)
            )

    missing = [name for name in wanted if name not in spans]
    if missing:
        raise ValueError(
            "exempt names are not a top-level function or class: "
            + ", ".join(sorted(missing))
        )
    duplicated = sorted(name for name, found in spans.items() if len(found) > 1)
    if duplicated:
        raise ValueError(
            "exempt names are defined more than once at top level: "
            + ", ".join(duplicated)
        )

    drop: set[int] = set()
    for found in spans.values():
        for start, end in found:
            drop.update(range(start, end + 1))
    lines = source.splitlines(keepends=True)
    return _without_blank_lines(
        "".join(
            line for number, line in enumerate(lines, start=1) if number not in drop
        )
    )


def _without_blank_lines(source: str) -> str:
    return "".join(
        line for line in source.splitlines(keepends=True) if line.strip()
    )


def source_projection_sha256(
    inputs: Iterable[tuple[Path, Sequence[str]]],
    *,
    root: Path,
) -> str:
    """Digest a list of ``(path, exempt_symbols)`` implementation inputs.

    The path is hashed alongside the bytes, so moving a file changes the digest
    even when its contents do not. Inputs are hashed in the order given, which
    the caller is expected to make deterministic.
    """
    digest = hashlib.sha256()
    for path, exempt in inputs:
        path = Path(path)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"cannot hash implementation input {path}") from exc
        if exempt:
            if path.suffix != ".py":
                raise ValueError(
                    f"cannot project a non-Python input: {path} names "
                    f"{len(tuple(exempt))} exempt symbol(s)"
                )
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"cannot project non-UTF-8 source {path}") from exc
            payload = project_source(text, exempt=exempt).encode("utf-8")
        else:
            payload = raw
        try:
            relative = path.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            relative = path.name
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def cell_identity_sha256(
    *,
    numerical_sha256: str,
    plan: Mapping[str, Any],
    runtime: Mapping[str, Any],
    consumed: Mapping[str, str],
) -> str:
    """The identity a banked cell must match to be reusable.

    ``consumed`` maps ``"variant/case"`` to that cell's identity, for cells this
    one reads as an input. It exists because the stage-16 fleet stopped
    ``heston/near_ki`` early and fed the truncated mean into
    ``heston_slv/near_ki``'s telescoping estimator with nothing in any hash
    linking the two, so the consumer stayed "valid" while its control moved.
    """
    return canonical_sha256(
        {
            "numerical_sha256": str(numerical_sha256),
            "plan": dict(plan),
            "runtime": dict(runtime),
            "consumed": dict(consumed),
        }
    )
