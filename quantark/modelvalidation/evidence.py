"""Evidence primitives: canonical serialization, hashing, durable writes.

An evidence package has to survive three things: a crash mid-run, a reviewer
asking "is this the file that was produced?", and a re-run on the same machine
producing the same answer. That gives three mechanisms here.

**Canonical JSON + projected hash.** The certificate's digest covers a
*projection* of the payload that drops volatile fields (wall-clock, timestamps).
Re-running the same study on the same machine reproduces the digest; a changed
number does not.

**Atomic writes.** Everything lands via write-temp-then-rename, so a crash
leaves either the old file or the new one, never a half-written certificate.

**Identity-gated checkpoints.** A checkpoint is reused only when the
configuration that produced it hashes identically. A mismatched checkpoint is
quarantined (renamed ``.stale``), never silently reused and never silently
deleted -- resume must be sound, not merely convenient.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

from quantark.util.exceptions import ValidationError

#: Evidence schema version. Independent of any prior certification work.
SCHEMA_VERSION: int = 1

#: Fields excluded from the projected hash: they vary between identical runs.
VOLATILE_KEYS = frozenset(
    {
        "wall_clock_seconds",
        "timestamp",
        "started_at",
        "finished_at",
        "host_load",
        "projected_sha256",
    }
)

#: Checkpoint keys become filenames, so they stay boring on purpose.
_SAFE_KEY = re.compile(r"^[A-Za-z0-9._-]+$")


def _json_default(value: Any) -> Any:
    """Serialize numpy scalars; refuse anything else so silent coercion cannot
    smuggle an unreviewable object into the evidence."""
    item = getattr(value, "item", None)
    if callable(item) and hasattr(value, "dtype"):
        return item()
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable in evidence"
    )


def canonical_json(obj: Any) -> str:
    """Serialize deterministically: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def evidence_projection(value: Any) -> Any:
    """Recursively copy ``value``, dropping volatile keys.

    The input is never mutated -- the caller's payload keeps its timings.
    """
    if isinstance(value, Mapping):
        return {
            key: evidence_projection(item)
            for key, item in value.items()
            if key not in VOLATILE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [evidence_projection(item) for item in value]
    return value


def projected_sha256(payload: Mapping[str, Any]) -> str:
    """Hex digest of the payload's projection.

    Stable across re-runs on one machine; sensitive to every substantive value.
    """
    text = canonical_json(evidence_projection(payload))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def identity_hash(mapping: Mapping[str, Any]) -> str:
    """Hex digest of a configuration identity, used to gate checkpoint reuse."""
    return hashlib.sha256(canonical_json(mapping).encode("utf-8")).hexdigest()


def _temp_roots() -> tuple[Path, ...]:
    roots = {Path(tempfile.gettempdir()), Path("/tmp")}
    resolved = set()
    for root in roots:
        resolved.add(root)
        try:
            resolved.add(root.resolve())
        except OSError:  # pragma: no cover - unreachable on supported platforms
            pass
    return tuple(resolved)


def validate_durable_root(path: Path | str) -> Path:
    """Reject an output root that sits in system-temp territory.

    Rejects a temp root itself (``/tmp``, ``$TMPDIR``) and its immediate
    children (``/tmp/whatever``) -- the shapes a default output path actually
    takes when someone reaches for scratch space. Deeper managed directories
    (pytest's ``tmp_path``, an explicit workspace) are allowed.

    Returns:
        The resolved path.

    Raises:
        ValidationError: the path is a temp root or an immediate child of one.
    """
    resolved = Path(path).resolve()
    roots = _temp_roots()
    if resolved in roots or resolved.parent in roots:
        raise ValidationError(
            f"Refusing to bank evidence under system temp storage: {resolved}. "
            "Choose a durable output directory (e.g. output/modelvalidation)."
        )
    return resolved


def atomic_write_text(path: Path | str, text: str) -> None:
    """Write ``text`` to ``path`` atomically, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, target)


def atomic_write_json(path: Path | str, payload: Any) -> None:
    """Write ``payload`` as indented JSON atomically.

    Indented (not canonical) on disk so a human can read a certificate; the
    hash is computed over the canonical projection, never over the file bytes.
    """
    atomic_write_text(
        path, json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )


def read_json(path: Path | str) -> Any:
    """Read a JSON file written by :func:`atomic_write_json`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


class CheckpointStore:
    """Durable per-unit checkpoints, gated on configuration identity.

    Layout: ``<root>/<kind>/<key>.json``, each holding the identity mapping,
    its hash, and the payload.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _path(self, kind: str, key: str) -> Path:
        if not _SAFE_KEY.match(kind or ""):
            raise ValidationError(f"Unsafe checkpoint kind {kind!r}")
        if not _SAFE_KEY.match(key or ""):
            raise ValidationError(
                f"Unsafe checkpoint key {key!r}; expected characters matching "
                "[A-Za-z0-9._-]+"
            )
        return self.root / kind / f"{key}.json"

    def save(
        self,
        kind: str,
        key: str,
        identity: Mapping[str, Any],
        payload: Any,
    ) -> None:
        """Persist ``payload`` under ``(kind, key)`` with its identity."""
        record = {
            "identity": dict(identity),
            "identity_hash": identity_hash(identity),
            "payload": payload,
        }
        atomic_write_json(self._path(kind, key), record)

    def load(
        self,
        kind: str,
        key: str,
        identity: Mapping[str, Any],
    ) -> Optional[Any]:
        """Return the stored payload, or ``None`` when it cannot be trusted.

        A checkpoint whose identity does not match, or which cannot be parsed,
        is renamed ``.stale`` and treated as absent. Nothing is deleted: a
        quarantined checkpoint is evidence about the run that produced it.
        """
        path = self._path(kind, key)
        if not path.exists():
            return None

        try:
            record = read_json(path)
            stored_hash = record["identity_hash"]
            payload = record["payload"]
        except (ValueError, KeyError, TypeError):
            self._quarantine(path)
            return None

        if stored_hash != identity_hash(identity):
            self._quarantine(path)
            return None

        return payload

    @staticmethod
    def _quarantine(path: Path) -> None:
        os.replace(path, path.with_name(path.name + ".stale"))
