"""Per-family builder registry.

A *builder* is a small function that turns declarative YAML params into real
quantark objects (products, environments, engines). YAML study files name
builders; builders own construction. This is what lets studies be declarative
without inventing a serialization format for rich domain objects such as coupon
schedules or grid configurations.

The registry grows one engine family at a time, and only when a real study needs
it -- never speculatively.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from quantark.util.exceptions import ValidationError

#: The kinds of object a study assembles. Each kind is its own namespace, so a
#: product and a candidate may share a name (e.g. ``equity.snowball``).
BUILDER_KINDS: Tuple[str, ...] = (
    "product",
    "environment",
    "reference",
    "candidate",
    "economic_scale",
)

#: (kind, name) -> builder callable.
_REGISTRY: Dict[Tuple[str, str], Callable] = {}


def _validate_kind(kind: str) -> None:
    if kind not in BUILDER_KINDS:
        raise ValidationError(
            f"Unknown builder kind {kind!r}; expected one of {BUILDER_KINDS}"
        )


def register_builder(name: str, kind: str) -> Callable[[Callable], Callable]:
    """Register a builder under ``name`` within ``kind``.

    Returns the decorated function unchanged, so builders remain directly
    callable and unit-testable without going through the registry.

    Raises:
        ValidationError: unknown kind, empty name, or a duplicate registration.
    """
    _validate_kind(kind)
    if not name:
        raise ValidationError("Builder name must be a non-empty string")

    def decorator(builder: Callable) -> Callable:
        key = (kind, name)
        if key in _REGISTRY:
            raise ValidationError(
                f"Builder {name!r} is already registered for kind {kind!r}"
            )
        _REGISTRY[key] = builder
        return builder

    return decorator


def get_builder(name: str, kind: str) -> Callable:
    """Look up a registered builder.

    Raises:
        ValidationError: unknown kind, or unknown name -- the message lists the
            names registered for that kind, so a typo in a YAML study is
            self-diagnosing.
    """
    _validate_kind(kind)
    try:
        return _REGISTRY[(kind, name)]
    except KeyError:
        available = sorted(n for (k, n) in _REGISTRY if k == kind)
        raise ValidationError(
            f"Unknown {kind} builder {name!r}. Registered {kind} builders: "
            f"{available if available else '(none)'}"
        ) from None


def list_builders() -> Dict[str, Tuple[str, ...]]:
    """Return every registered builder name, grouped by kind and sorted."""
    return {
        kind: tuple(sorted(n for (k, n) in _REGISTRY if k == kind))
        for kind in BUILDER_KINDS
    }


def clear_registry() -> None:
    """Remove every registration.

    Test-only helper for isolating registry state. Production code never calls
    this: builders are registered once at import time.
    """
    _REGISTRY.clear()
