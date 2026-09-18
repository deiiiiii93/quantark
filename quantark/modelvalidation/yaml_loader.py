"""YAML study loader.

A study file is declarative and diffable: it names *builders* and their params,
and the builders construct the real pricing objects. That keeps studies
reviewable and deliverable (a certificate can carry its own definition verbatim)
without a serializer for every quantark type.

The loader validates structure and resolves names; builders validate semantics.
Every structural error names the YAML path that caused it, so a typo in a study
file diagnoses itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.registry import get_builder
from quantark.modelvalidation.study import (
    SUPPORTED_SCHEMAS,
    CaseSpec,
    CertificationStudy,
    GateBounds,
    QuantityBounds,
    SamplingPolicy,
)

TOP_LEVEL_KEYS = frozenset(
    {
        "study",
        "schema",
        "quantities",
        "bounds",
        "sampling",
        "economic_scale",
        "environment",
        "product",
        "reference",
        "candidates",
        "cases",
        "context",
        "quantity_bounds",
    }
)

_BOUNDS_OPTIONAL = ("se_budget_fraction", "interval_k", "envelope_fraction")
_SAMPLING_OPTIONAL = ("bump",)
_CASE_KEYS_SCHEMA_1 = frozenset({"name", "environment", "product"})
_CASE_KEYS_SCHEMA_2 = _CASE_KEYS_SCHEMA_1 | {"context", "expect"}


def _require(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if not isinstance(mapping, Mapping) or key not in mapping:
        raise ValidationError(f"Study is missing required key {path}")
    return mapping[key]


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{path} must be a mapping, got {type(value).__name__}")
    return value


def _builder_spec(document: Mapping[str, Any], key: str) -> tuple[str, dict]:
    spec = _require_mapping(_require(document, key, key), key)
    name = _require(spec, "builder", f"{key}.builder")
    if not isinstance(name, str):
        raise ValidationError(f"{key}.builder must be a string, got {name!r}")
    params = spec.get("params", {})
    return name, dict(_require_mapping(params, f"{key}.params"))


def _bounds(document: Mapping[str, Any]) -> GateBounds:
    spec = _require_mapping(_require(document, "bounds", "bounds"), "bounds")
    kwargs = {
        "cell": float(_require(spec, "cell", "bounds.cell")),
        "mean_signed_bias": float(
            _require(spec, "mean_signed_bias", "bounds.mean_signed_bias")
        ),
    }
    for optional in _BOUNDS_OPTIONAL:
        if optional in spec:
            kwargs[optional] = float(spec[optional])
    unknown = set(spec) - {"cell", "mean_signed_bias", *_BOUNDS_OPTIONAL}
    if unknown:
        raise ValidationError(f"Unknown keys in bounds: {sorted(unknown)}")
    return GateBounds(**kwargs)


def _sampling(document: Mapping[str, Any], schema: int) -> SamplingPolicy:
    spec = _require_mapping(_require(document, "sampling", "sampling"), "sampling")
    required = ("paths_per_batch", "min_batches", "max_batches", "seed")
    kwargs: dict[str, Any] = {
        key: int(_require(spec, key, f"sampling.{key}")) for key in required
    }
    for optional in _SAMPLING_OPTIONAL:
        if optional in spec:
            kwargs[optional] = float(spec[optional])
    # Each schema has its own seed scheme; the study's validation refuses the other one.
    kwargs["seed_scheme"] = str(spec.get("seed_scheme", "substream" if schema == 2 else "sequential"))
    unknown = set(spec) - {*required, *_SAMPLING_OPTIONAL, "seed_scheme"}
    if unknown:
        raise ValidationError(f"Unknown keys in sampling: {sorted(unknown)}")
    return SamplingPolicy(**kwargs)


def _quantity_bounds(document: Mapping[str, Any]) -> dict:
    raw = _require_mapping(_require(document, "quantity_bounds", "quantity_bounds"), "quantity_bounds")
    out = {}
    for quantity, spec in raw.items():
        path = f"quantity_bounds.{quantity}"
        spec = _require_mapping(spec, path)
        unknown = set(spec) - {"abs_floor", "rel"}
        if unknown:
            raise ValidationError(f"Unknown keys in {path}: {sorted(unknown)}")
        out[str(quantity)] = QuantityBounds(
            abs_floor=float(_require(spec, "abs_floor", f"{path}.abs_floor")),
            rel=float(spec.get("rel", 0.0)),
        )
    return out


def _cases(document: Mapping[str, Any], schema: int) -> tuple[CaseSpec, ...]:
    raw = _require(document, "cases", "cases")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValidationError("cases must be a list")
    if not raw:
        raise ValidationError("cases must not be empty")

    allowed = _CASE_KEYS_SCHEMA_2 if schema == 2 else _CASE_KEYS_SCHEMA_1
    cases = []
    for index, entry in enumerate(raw):
        path = f"cases[{index}]"
        spec = _require_mapping(entry, path)
        unknown = set(spec) - allowed
        if unknown:
            raise ValidationError(
                f"Unknown keys in {path}: {sorted(unknown)}; expected a subset of {sorted(allowed)}"
            )
        cases.append(
            CaseSpec(
                name=str(_require(spec, "name", f"{path}.name")),
                environment_params=dict(
                    _require_mapping(spec.get("environment", {}), f"{path}.environment")
                ),
                product_params=dict(
                    _require_mapping(spec.get("product", {}), f"{path}.product")
                ),
                context_params=dict(
                    _require_mapping(spec.get("context", {}), f"{path}.context")
                ),
                expected={
                    str(k): str(v)
                    for k, v in _require_mapping(spec.get("expect", {}), f"{path}.expect").items()
                },
            )
        )
    return tuple(cases)


def _quantities(document: Mapping[str, Any]) -> tuple[str, ...]:
    raw = _require(document, "quantities", "quantities")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValidationError("quantities must be a list")
    return tuple(str(q) for q in raw)


def _ensure_builtin_builders() -> None:
    """Import the builtin builder modules so their registrations exist.

    Imported lazily and from inside the loader rather than at module scope:
    builder modules import engines from across quantark, and a study file is the
    only thing that needs them. Doing it here means every entry point (CLI,
    library call, anchor replay) sees the same registry without callers having
    to remember an import.
    """
    from quantark.modelvalidation import builders  # noqa: F401


def load_study_text(text: str) -> CertificationStudy:
    """Parse a YAML study, resolving builders through the registry.

    Raises:
        ValidationError: malformed YAML, a structural violation (with its path),
            or an unknown builder name.
    """
    _ensure_builtin_builders()
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValidationError(f"Study is not valid YAML: {exc}") from exc

    document = _require_mapping(document, "study document")

    unknown = set(document) - TOP_LEVEL_KEYS
    if unknown:
        raise ValidationError(
            f"Unknown top-level keys in study: {sorted(unknown)}; expected a subset of "
            f"{sorted(TOP_LEVEL_KEYS)}"
        )

    schema = int(_require(document, "schema", "schema"))
    if schema not in SUPPORTED_SCHEMAS:
        raise ValidationError(f"Study schema must be one of {SUPPORTED_SCHEMAS}, got {schema}")
    if schema == 1 and ({"quantity_bounds", "context"} & set(document)):
        raise ValidationError("quantity_bounds and context are schema-2 keys; this study declares schema 1")

    name = str(_require(document, "study", "study"))
    quantities = _quantities(document)
    quantity_bounds = _quantity_bounds(document) if schema == 2 else {}
    bounds = _bounds(document)
    sampling = _sampling(document, schema)
    cases = _cases(document, schema)

    scale_name, scale_params = _builder_spec(document, "economic_scale")
    scale = get_builder(scale_name, kind="economic_scale")(scale_params)

    # Environment and product params are passed through, not built here: the
    # reference and candidate builders own construction, because only they know
    # which engine objects they need. The names are still resolved now, so an
    # unknown builder fails at load time rather than hours into a run.
    environment_builder, environment_params = _builder_spec(document, "environment")
    product_builder, product_params = _builder_spec(document, "product")
    get_builder(environment_builder, kind="environment")
    get_builder(product_builder, kind="product")

    # Schema 2: the study-level intraday context (valuation instant, clock, history) reaches every arm.
    arm_extra: dict = {}
    if "context" in document:
        context_builder, context_params = _builder_spec(document, "context")
        get_builder(context_builder, kind="context")
        arm_extra["context_params"] = context_params

    reference_name, reference_params = _builder_spec(document, "reference")
    reference = get_builder(reference_name, kind="reference")(
        environment_params=environment_params,
        product_params=product_params,
        sampling=sampling,
        quantities=quantities,
        params=reference_params,
        **arm_extra,
    )

    raw_candidates = _require(document, "candidates", "candidates")
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
        raise ValidationError("candidates must be a list")
    if not raw_candidates:
        raise ValidationError("candidates must not be empty")

    candidates = []
    for index, entry in enumerate(raw_candidates):
        path = f"candidates[{index}]"
        spec = _require_mapping(entry, path)
        builder_name = _require(spec, "builder", f"{path}.builder")
        params = dict(_require_mapping(spec.get("params", {}), f"{path}.params"))
        candidates.append(
            get_builder(str(builder_name), kind="candidate")(
                environment_params=environment_params,
                product_params=product_params,
                quantities=quantities,
                params=params,
                **arm_extra,
            )
        )

    return CertificationStudy(
        name=name,
        schema=schema,
        cases=cases,
        quantities=quantities,
        bounds=bounds,
        scale=scale,
        reference=reference,
        candidates=tuple(candidates),
        sampling=sampling,
        source_text=text,
        quantity_bounds=quantity_bounds,
    )


def load_study(path: str | Path) -> CertificationStudy:
    """Load a YAML study from disk, keeping its text verbatim for evidence."""
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"Cannot read study file {file_path}: {exc}") from exc
    return load_study_text(text)
