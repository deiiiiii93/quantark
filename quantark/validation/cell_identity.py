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
import io
import json
import tokenize
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

    Top-level functions, async functions, classes and simple assignments may be
    exempted, and every name in ``exempt`` must resolve to exactly one of them. A
    name that resolves to nothing is an error rather than a no-op: the common way
    an exemption list goes wrong is a rename, and a silent no-op would widen the
    projection at exactly the moment it should have complained.

    Assignments are exemptable so that the provenance BOOKKEEPING constants -- the
    exemption list itself, the input lists -- can sit outside the projection they
    govern. While they were inside it, adding a validator helper and listing it as
    exempt changed the list and invalidated every banked cell: the act of
    exempting defeated itself. Doing this loses nothing, because such a list's
    effect stays visible in the projection regardless -- exempting a pre-existing
    function still removes its body, and a newly added exempt function was never
    there to begin with. Exempt only constants that DESCRIBE the digest; a
    constant the arithmetic reads (an allocation table, a bound) must stay in.

    Decorators are removed with the definition they decorate. Module-level
    statements -- imports, constants, dataclass declarations -- always remain,
    because a changed constant is a changed plan.

    The result is canonical EXECUTABLE text: blank lines, comments and docstrings
    are all removed. Each of those was found the hard way to invalidate cells for
    free -- a removed definition left its PEP8 blank separators behind, and later
    an added explanatory comment shifted the digest on its own. None of them can
    change what the code computes, so none may decide whether an expensive banked
    cell is reusable. The cost is that blank lines and '#' characters inside string
    literals stop counting; no arithmetic here depends on either.
    """
    wanted = list(dict.fromkeys(exempt))
    tree = ast.parse(source)
    spans: dict[str, list[tuple[int, int]]] = {}
    for node in tree.body:
        for name in _exemptable_names(node):
            if name not in wanted:
                continue
            first = node.lineno
            for decorator in getattr(node, "decorator_list", []) or []:
                first = min(first, decorator.lineno)
            spans.setdefault(name, []).append((first, node.end_lineno or node.lineno))

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
    for start, end in _docstring_spans(tree):
        drop.update(range(start, end + 1))
    return _canonical_code(source, drop_lines=drop)


def _docstring_spans(tree: ast.AST) -> list[tuple[int, int]]:
    """Line spans of every docstring, at module, class and function level.

    A docstring is prose in a string literal, as inert as a comment, and dropped
    for the same reason. Only the conventional FIRST statement of a scope counts,
    so a string used as data is untouched.
    """
    spans = []
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            spans.append((first.lineno, first.end_lineno or first.lineno))
    return spans


def _canonical_code(source: str, *, drop_lines: set[int]) -> str:
    """Executable text only: no dropped spans, no comments, no blank lines.

    Comments are found by the tokenizer rather than by searching for '#', so a
    hash inside a string literal survives. Everything removed here is text that
    cannot change what the code computes, and so must not decide whether an
    expensive banked cell may be reused. If a comment costs a 36-hour re-run, the
    rational response is to stop writing comments -- a worse outcome than any
    digest precision gained.
    """
    comment_columns: dict[int, int] = {}
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                row, column = token.start
                comment_columns[row] = min(comment_columns.get(row, column), column)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Real Python would already have failed the ast.parse above; keep the
        # digest deterministic rather than raising a second, less useful error.
        comment_columns = {}

    kept = []
    for number, line in enumerate(source.splitlines(), start=1):
        if number in drop_lines:
            continue
        if number in comment_columns:
            line = line[: comment_columns[number]]
        line = line.rstrip()
        if line:
            kept.append(line)
    return "\n".join(kept) + ("\n" if kept else "")


def _exemptable_names(node: ast.stmt) -> tuple[str, ...]:
    """Names this top-level statement defines, if it may be exempted at all."""
    if isinstance(node, _EXEMPTABLE):
        return (node.name,)
    if isinstance(node, ast.Assign):
        return tuple(
            target.id for target in node.targets if isinstance(target, ast.Name)
        )
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return (node.target.id,)
    return ()




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
