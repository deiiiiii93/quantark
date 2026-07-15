"""Canonical value fingerprints (spec section 10.1).

Keys derive from VALUES, never Python object identity. A type without a safe
canonicalizer raises ``Uncanonicalizable``; callers treat that artifact as
uncacheable and build fresh — correctness never depends on cacheability.
"""
import dataclasses
import hashlib
from datetime import date, datetime
from enum import Enum

import numpy as np

__all__ = ["Uncanonicalizable", "canonical_tree", "fingerprint", "try_fingerprint"]


class Uncanonicalizable(Exception):
    """No safe canonicalizer exists for this value."""


def canonical_tree(obj):
    if obj is None or isinstance(obj, (bool, int, str)):
        # The tag carries the concrete type, so 1 and True and 1.0 differ.
        return ("atom", type(obj).__name__, obj)
    if isinstance(obj, float):
        if obj != obj:  # NaN: hex() differs across signs/payloads
            return ("float", "nan")
        return ("float", obj.hex())
    if isinstance(obj, (datetime, date)):
        return ("dt", obj.isoformat())
    if isinstance(obj, Enum):
        return ("enum", f"{type(obj).__module__}.{type(obj).__qualname__}", obj.name)
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return (
            "nd", str(arr.dtype), tuple(arr.shape),
            hashlib.sha256(arr.tobytes()).hexdigest(),
        )
    if isinstance(obj, np.generic):
        return canonical_tree(obj.item())
    if isinstance(obj, (tuple, list)):
        return ("seq", tuple(canonical_tree(x) for x in obj))
    if isinstance(obj, dict):
        if not all(isinstance(k, str) for k in obj):
            raise Uncanonicalizable("dict keys must be strings")
        return (
            "map",
            tuple((k, canonical_tree(v)) for k, v in sorted(obj.items())),
        )
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cls = type(obj)
        return (
            "dc", f"{cls.__module__}.{cls.__qualname__}",
            tuple(
                (f.name, canonical_tree(getattr(obj, f.name)))
                for f in dataclasses.fields(cls)
            ),
        )
    raise Uncanonicalizable(
        f"no canonicalizer for {type(obj).__module__}.{type(obj).__qualname__}"
    )


def fingerprint(obj) -> str:
    tree = canonical_tree(obj)
    return hashlib.sha256(repr(tree).encode()).hexdigest()


def try_fingerprint(obj):
    try:
        return fingerprint(obj)
    except Uncanonicalizable:
        return None
