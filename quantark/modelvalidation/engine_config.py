"""Recording the engine configuration a candidate actually ran with.

A certificate that says an engine ran at ``accuracy: standard`` is weaker
evidence than it looks. "Standard" is an indirection into a profile table, and
if that table changes in a later version the certificate still reads the same
while describing a different grid. Worse, the candidate's identity hash would
not change either, so a stale checkpoint or anchor could be reused across a
genuine numerical change.

So candidates record their configuration *resolved*: the actual node counts,
step densities, bounds, and scheme switches the engine was handed. That makes
the certificate reproducible on its own terms, and makes the identity hash move
when the numerics move.

These are the **requested** settings. The achieved geometry (how many nodes a
grid ended up with after alignment, whether a step cap bit) is not exposed by
the engines through any public API, and is deliberately not guessed at here.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any, Dict, Mapping


def _plain(value: Any) -> Any:
    """Coerce a config value into something JSON- and hash-stable."""
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _plain(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def engine_config(params: Any, *, exclude: tuple[str, ...] = ()) -> Dict[str, Any]:
    """Serialize an engine parameter object into an evidence-safe dict.

    Args:
        params: An engine params dataclass (``PDEParams``, ``QuadParams``, ...).
        exclude: Field names to drop -- for settings that cannot change the
            numbers, such as cache sizes.

    Returns:
        A plain dict with enums flattened to their values and nested
        dataclasses expanded.
    """
    if not dataclasses.is_dataclass(params):
        raise TypeError(f"engine_config expects a dataclass, got {type(params).__name__}")
    return {
        field.name: _plain(getattr(params, field.name))
        for field in dataclasses.fields(params)
        if field.name not in exclude
    }


def flatten(config: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested config into ``dotted.key`` form for tabular display."""
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, prefix=f"{path}."))
        else:
            flat[path] = value
    return flat
