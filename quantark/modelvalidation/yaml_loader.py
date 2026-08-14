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
    SCHEMA,
    CaseSpec,
    CertificationStudy,
    GateBounds,
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
    }
)

_BOUNDS_OPTIONAL = ("se_budget_fraction", "interval_k", "envelope_fraction")
_SAMPLING_OPTIONAL = ("bump",)


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


def _sampling(document: Mapping[str, Any]) -> SamplingPolicy:
    spec = _require_mapping(_require(document, "sampling", "sampling"), "sampling")
    required = ("paths_per_batch", "min_batches", "max_batches", "seed")
    kwargs: dict[str, Any] = {
        key: int(_require(spec, key, f"sampling.{key}")) for key in required
    }
    for optional in _SAMPLING_OPTIONAL:
        if optional in spec:
            kwargs[optional] = float(spec[optional])
    unknown = set(spec) - {*required, *_SAMPLING_OPTIONAL}
    if unknown:
        raise ValidationError(f"Unknown keys in sampling: {sorted(unknown)}")
    return SamplingPolicy(**kwargs)


def _cases(document: Mapping[str, Any]) -> tuple[CaseSpec, ...]:
    raw = _require(document, "cases", "cases")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValidationError("cases must be a list")
    if not raw:
        raise ValidationError("cases must not be empty")

    cases = []
    for index, entry in enumerate(raw):
        path = f"cases[{index}]"
        spec = _require_mapping(entry, path)
        unknown = set(spec) - {"name", "environment", "product"}
        if unknown:
            raise ValidationError(f"Unknown keys in {path}: {sorted(unknown)}")
        cases.append(
            CaseSpec(
                name=str(_require(spec, "name", f"{path}.name")),
                environment_params=dict(
                    _require_mapping(spec.get("environment", {}), f"{path}.environment")
                ),
                product_params=dict(
                    _require_mapping(spec.get("product", {}), f"{path}.product")
                ),
            )
        )
    return tuple(cases)


def _quantities(document: Mapping[str, Any]) -> tuple[str, ...]:
    raw = _require(document, "quantities", "quantities")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValidationError("quantities must be a list")
    return tuple(str(q) for q in raw)


def load_study_text(text: str) -> CertificationStudy:
    """Parse a YAML study, resolving builders through the registry.

    Raises:
        ValidationError: malformed YAML, a structural violation (with its path),
            or an unknown builder name.
    """
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
    if schema != SCHEMA:
        raise ValidationError(f"Study schema must be {SCHEMA}, got {schema}")

    name = str(_require(document, "study", "study"))
    quantities = _quantities(document)
    bounds = _bounds(document)
    sampling = _sampling(document)
    cases = _cases(document)

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

    reference_name, reference_params = _builder_spec(document, "reference")
    reference = get_builder(reference_name, kind="reference")(
        environment_params=environment_params,
        product_params=product_params,
        sampling=sampling,
        quantities=quantities,
        params=reference_params,
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
    )


def load_study(path: str | Path) -> CertificationStudy:
    """Load a YAML study from disk, keeping its text verbatim for evidence."""
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"Cannot read study file {file_path}: {exc}") from exc
    return load_study_text(text)
