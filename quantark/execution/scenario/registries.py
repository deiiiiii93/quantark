"""Importable scenario registries: transformers, runners, factories.

Registration contract (spec section 13.1 "registered, importable pure
function"; hardened by plan-gate finding 1, 2026-07-17): the registered
callable must be resolvable by importing its module and walking its
qualified name — ``getattr(import_module(fn.__module__), fn.__qualname__)
is fn``. Lambdas, closures, and instance methods are rejected with
``ValidationGateError`` because a spawn/Dask worker could never rebuild
them. Component EXTRACTORS inside a transformer registration may be
lambdas: workers never unpickle them — a child resolves the whole
registration by ID after importing the registering module, which
re-creates the extractors at import time.

Re-registering the same ID with the same function object is a no-op
(idempotent module re-imports); a different object under an existing ID
raises ``ValidationGateError`` — silent replacement is never allowed.
"""
import importlib
import json
from dataclasses import dataclass

from quantark.execution.errors import CapabilityError, ValidationGateError
from quantark.execution.scenario.contracts import CallableRef

__all__ = [
    "TransformerRegistration",
    "RunnerRegistration",
    "FactoryRegistration",
    "register_transformer",
    "register_runner",
    "register_factory",
    "get_transformer",
    "get_runner",
    "get_factory",
    "callable_ref",
    "check_worker_payload",
]

_VALUE_KINDS = ("float", "native")


@dataclass(frozen=True)
class TransformerRegistration:
    transformer_id: str
    fn: object
    allowed_tags: frozenset
    components: tuple            # ((tag, extractor), ...)
    schema_version: str


@dataclass(frozen=True)
class RunnerRegistration:
    runner_id: str
    fn: object
    value_kind: str              # "float" | "native"
    schema_version: str


@dataclass(frozen=True)
class FactoryRegistration:
    factory_id: str
    fn: object
    schema_version: str


_TRANSFORMERS: dict = {}
_RUNNERS: dict = {}
_FACTORIES: dict = {}
_TABLES = {
    "transformer": _TRANSFORMERS,
    "runner": _RUNNERS,
    "factory": _FACTORIES,
}


def _require_importable(fn) -> None:
    qualname = getattr(fn, "__qualname__", "")
    module = getattr(fn, "__module__", "")
    if not module or not qualname or "<" in qualname:
        raise ValidationGateError(
            "registered callables must be module-level named functions "
            f"(got {module!r}:{qualname!r}); lambdas and closures cannot be "
            "reconstructed in a spawn worker"
        )
    try:
        resolved = importlib.import_module(module)
        for part in qualname.split("."):
            resolved = getattr(resolved, part)
    except (ImportError, AttributeError) as exc:
        raise ValidationGateError(
            f"registered callable {module}:{qualname} is not importable: {exc}"
        ) from exc
    if resolved is not fn:
        raise ValidationGateError(
            f"{module}:{qualname} does not resolve back to the registered "
            "object; register a module-level function"
        )


def _register(kind: str, key: str, registration) -> None:
    table = _TABLES[kind]
    existing = table.get(key)
    if existing is not None:
        if existing.fn is registration.fn:
            return  # idempotent re-import of the registering module
        raise ValidationGateError(
            f"{kind} id {key!r} is already registered to a different object"
        )
    table[key] = registration


def register_transformer(transformer_id: str, fn, *, allowed_tags: frozenset,
                         components: tuple, schema_version: str = "1") -> None:
    _require_importable(fn)
    _register(
        "transformer", transformer_id,
        TransformerRegistration(
            transformer_id=transformer_id, fn=fn,
            allowed_tags=frozenset(allowed_tags),
            components=tuple(components), schema_version=schema_version,
        ),
    )


def register_runner(runner_id: str, fn, *, value_kind: str = "native",
                    schema_version: str = "1") -> None:
    if value_kind not in _VALUE_KINDS:
        raise ValidationGateError(
            f"value_kind must be one of {_VALUE_KINDS}, got {value_kind!r}"
        )
    _require_importable(fn)
    _register(
        "runner", runner_id,
        RunnerRegistration(
            runner_id=runner_id, fn=fn, value_kind=value_kind,
            schema_version=schema_version,
        ),
    )


def register_factory(factory_id: str, fn, *, schema_version: str = "1") -> None:
    _require_importable(fn)
    _register(
        "factory", factory_id,
        FactoryRegistration(
            factory_id=factory_id, fn=fn, schema_version=schema_version,
        ),
    )


def _get(kind: str, key: str):
    registration = _TABLES[kind].get(key)
    if registration is None:
        raise CapabilityError(
            f"no {kind} registered under id {key!r}; scenario {kind}s must "
            "be registered by their defining module before use"
        )
    return registration


def get_registration(kind: str, ref_id: str):
    """Kind-dispatched lookup used by worker CallableRef verification."""
    return _get(kind, ref_id)


def get_transformer(transformer_id: str) -> TransformerRegistration:
    return _get("transformer", transformer_id)


def get_runner(runner_id: str) -> RunnerRegistration:
    return _get("runner", runner_id)


def get_factory(factory_id: str) -> FactoryRegistration:
    return _get("factory", factory_id)


def callable_ref(kind: str, ref_id: str) -> CallableRef:
    """Spawn-reconstructable reference for a registered callable."""
    registration = _get(kind, ref_id)
    fn = registration.fn
    return CallableRef(
        kind=kind,
        ref_id=ref_id,
        module=fn.__module__,
        qualname=fn.__qualname__,
        schema_version=registration.schema_version,
    )


def check_worker_payload(obj) -> None:
    """Reject payloads that are not canonical JSON primitives (spec 12.3)."""
    try:
        json.loads(json.dumps(obj))
    except (TypeError, ValueError) as exc:
        raise ValidationGateError(
            f"worker payload is not JSON-serializable: {exc}"
        ) from exc
