"""Loading a schema-2 study: quantity budgets, context, per-case context and expectations."""
import textwrap

import pytest

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation import registry
from quantark.modelvalidation.registry import register_builder
from quantark.modelvalidation.study import NormalizedScale, QuantityBounds
from quantark.modelvalidation.yaml_loader import _ensure_builtin_builders, load_study_text

VALID2 = textwrap.dedent(
    """
    study: s2-demo
    schema: 2
    quantities: [pv, point_delta, desk_gamma]
    quantity_bounds:
      pv: {abs_floor: 1.0e-6}
      point_delta: {abs_floor: 1.0e-5, rel: 1.0e-4}
      desk_gamma: {abs_floor: 1.0e-4, rel: 1.0e-3}
    bounds: {cell: 1.0, mean_signed_bias: 0.2, se_budget_fraction: 0.25, interval_k: 2.0, envelope_fraction: 0.5}
    sampling: {paths_per_batch: 4096, min_batches: 4, max_batches: 4, seed: 11}
    economic_scale: {builder: normalized, params: {notional: 100.0, spot_scale: 100.0}}
    environment: {builder: s2.env, params: {spot: 100.0, vol: 0.2}}
    product: {builder: s2.product, params: {strike: 100.0}}
    context: {builder: s2.context, params: {valuation: "2026-09-10T14:00:00+08:00", profile: desk}}
    reference: {builder: s2.reference, params: {}}
    candidates:
      - {builder: s2.candidate, params: {tag: a}}
    cases:
      - {name: ordinary}
      - {name: one_second, context: {valuation: "2026-09-10T14:59:59+08:00"}, environment: {spot: 75.2}}
      - {name: on_barrier, context: {valuation: "2026-09-10T15:00:00+08:00"}, environment: {spot: 75.0},
         expect: {point_delta: undefined}}
    """
).strip()


class _Arm:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def name(self):
        return "s2.candidate"

    def params(self):
        return {}

    def identity(self, case):
        return {"case": case.name}


@pytest.fixture(autouse=True)
def _builders():
    # Import the builtin builders BEFORE the snapshot: the study uses the builtin `normalized` scale, and a
    # lazy first import inside the test would register them only to have the restore below wipe them.
    _ensure_builtin_builders()
    saved = dict(registry._REGISTRY)
    try:
        register_builder("s2.env", kind="environment")(lambda params: dict(params))
        register_builder("s2.product", kind="product")(lambda params: dict(params))
        register_builder("s2.context", kind="context")(lambda params: dict(params))
        register_builder("s2.reference", kind="reference")(lambda **kwargs: _Arm(**kwargs))
        register_builder("s2.candidate", kind="candidate")(lambda **kwargs: _Arm(**kwargs))
        yield
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)


def test_loads_quantity_bounds_scale_and_context():
    study = load_study_text(VALID2)
    assert study.schema == 2
    assert study.quantities == ("pv", "point_delta", "desk_gamma")
    assert study.quantity_bounds["point_delta"] == QuantityBounds(abs_floor=1e-5, rel=1e-4)
    assert isinstance(study.scale, NormalizedScale) and study.scale.spot_scale == 100.0
    assert study.reference.kwargs["context_params"] == {"valuation": "2026-09-10T14:00:00+08:00", "profile": "desk"}
    assert study.candidates[0].kwargs["context_params"]["profile"] == "desk"


def test_case_context_and_expectations_are_loaded():
    study = load_study_text(VALID2)
    by_name = {case.name: case for case in study.cases}
    assert by_name["ordinary"].context_params == {} and by_name["ordinary"].expected == {}
    assert by_name["one_second"].context_params == {"valuation": "2026-09-10T14:59:59+08:00"}
    assert by_name["one_second"].environment_params == {"spot": 75.2}
    assert by_name["on_barrier"].expected == {"point_delta": "undefined"}


def test_schema_2_samples_on_substreams_and_refuses_the_sequential_scheme():
    assert load_study_text(VALID2).sampling.seed_scheme == "substream"
    sequential = VALID2.replace("seed: 11}", "seed: 11, seed_scheme: sequential}")
    with pytest.raises(ValidationError, match="substream"):
        load_study_text(sequential)


def test_schema_2_requires_quantity_bounds():
    text = "\n".join(line for line in VALID2.splitlines() if "abs_floor" not in line and "quantity_bounds" not in line)
    with pytest.raises(ValidationError, match="quantity_bounds"):
        load_study_text(text)


def test_schema_1_rejects_schema_2_keys():
    text = VALID2.replace("schema: 2", "schema: 1")
    with pytest.raises(ValidationError, match="schema-2"):
        load_study_text(text)


def test_unknown_case_key_is_named():
    with pytest.raises(ValidationError, match=r"cases\[0\]"):
        load_study_text(VALID2.replace("- {name: ordinary}", "- {name: ordinary, expects: {}}"))


def test_unknown_quantity_bounds_key_is_named():
    with pytest.raises(ValidationError, match="quantity_bounds.pv"):
        load_study_text(VALID2.replace("pv: {abs_floor: 1.0e-6}", "pv: {abs_floor: 1.0e-6, tol: 1}"))
