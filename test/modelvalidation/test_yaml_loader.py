"""Tests for the YAML study loader."""

import textwrap

import pytest

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation import registry
from quantark.modelvalidation.registry import register_builder
from quantark.modelvalidation.study import HedgeContractScale
from quantark.modelvalidation.yaml_loader import load_study, load_study_text

VALID = textwrap.dedent(
    """
    study: yaml-demo
    schema: 1
    quantities: [pv, delta]
    bounds: {cell: 0.5, mean_signed_bias: 0.1}
    sampling: {paths_per_batch: 4096, min_batches: 2, max_batches: 8, seed: 11}
    economic_scale:
      builder: test.scale
      params: {hedge_multiplier: 200, hedge_inception_spot: 100.0, notional: 50000000}
    environment: {builder: test.env, params: {spot: 100.0, vol: 0.2}}
    product: {builder: test.product, params: {strike: 100.0}}
    reference: {builder: test.reference, params: {}}
    candidates:
      - {builder: test.candidate, params: {tag: a}}
      - {builder: test.candidate, params: {tag: b}}
    cases:
      - {name: ordinary}
      - {name: near_ko, environment: {spot: 102.5}, product: {strike: 101.0}}
    """
).strip()


@pytest.fixture(autouse=True)
def _builders():
    saved = dict(registry._REGISTRY)
    try:

        @register_builder("test.scale", kind="economic_scale")
        def scale(params):
            return HedgeContractScale(**params)

        @register_builder("test.env", kind="environment")
        def env(params):
            return dict(params)

        @register_builder("test.product", kind="product")
        def product(params):
            return dict(params)

        @register_builder("test.reference", kind="reference")
        def reference(environment_params, product_params, sampling, quantities, params):
            return _Recorder(
                "reference",
                environment_params=environment_params,
                product_params=product_params,
                sampling=sampling,
                quantities=quantities,
                params=params,
            )

        @register_builder("test.candidate", kind="candidate")
        def candidate(environment_params, product_params, quantities, params):
            return _Recorder(
                "candidate",
                environment_params=environment_params,
                product_params=product_params,
                quantities=quantities,
                params=params,
            )

        yield
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)


class _Recorder:
    def __init__(self, kind, **kwargs):
        self.kind = kind
        self.kwargs = kwargs

    def name(self):
        return f"{self.kind}:{self.kwargs['params'].get('tag', 'x')}"

    def params(self):
        return dict(self.kwargs["params"])


def _yaml_without(line_prefix: str) -> str:
    return "\n".join(
        line for line in VALID.splitlines() if not line.strip().startswith(line_prefix)
    )


def test_loads_a_valid_study():
    study = load_study_text(VALID)
    assert study.name == "yaml-demo"
    assert study.schema == 1
    assert study.quantities == ("pv", "delta")
    assert study.bounds.cell == 0.5
    assert study.sampling.seed == 11
    assert len(study.candidates) == 2
    assert [c.name for c in study.cases] == ["ordinary", "near_ko"]


def test_source_text_round_trips_verbatim():
    study = load_study_text(VALID)
    assert study.source_text == VALID


def test_case_overrides_reach_the_case_spec():
    study = load_study_text(VALID)
    near_ko = study.cases[1]
    assert near_ko.environment_params == {"spot": 102.5}
    assert near_ko.product_params == {"strike": 101.0}


def test_builders_receive_the_study_level_specs():
    study = load_study_text(VALID)
    assert study.reference.kwargs["environment_params"] == {"spot": 100.0, "vol": 0.2}
    assert study.reference.kwargs["product_params"] == {"strike": 100.0}
    assert study.reference.kwargs["quantities"] == ("pv", "delta")
    assert study.reference.kwargs["sampling"].seed == 11
    assert study.candidates[0].kwargs["params"] == {"tag": "a"}


def test_economic_scale_is_built():
    study = load_study_text(VALID)
    assert isinstance(study.scale, HedgeContractScale)
    assert study.scale.hedge_multiplier == 200


def test_optional_bounds_fields_default():
    study = load_study_text(VALID)
    assert study.bounds.se_budget_fraction == 0.25
    assert study.bounds.interval_k == 2.0


def test_rejects_wrong_schema():
    with pytest.raises(ValidationError):
        load_study_text(VALID.replace("schema: 1", "schema: 2"))


def test_rejects_unknown_builder_and_names_the_registered_ones():
    with pytest.raises(ValidationError) as exc:
        load_study_text(VALID.replace("builder: test.candidate", "builder: test.nope"))
    assert "test.candidate" in str(exc.value)


def test_rejects_a_missing_required_section():
    with pytest.raises(ValidationError) as exc:
        load_study_text(_yaml_without("bounds:"))
    assert "bounds" in str(exc.value)


def test_rejects_a_missing_bounds_field_naming_the_path():
    broken = VALID.replace(
        "bounds: {cell: 0.5, mean_signed_bias: 0.1}", "bounds: {mean_signed_bias: 0.1}"
    )
    with pytest.raises(ValidationError) as exc:
        load_study_text(broken)
    assert "bounds.cell" in str(exc.value)


def test_rejects_duplicate_case_names():
    broken = VALID.replace("- {name: ordinary}", "- {name: near_ko}")
    with pytest.raises(ValidationError):
        load_study_text(broken)


def test_rejects_an_unknown_top_level_key():
    """A typo in a study file must fail loudly, not be ignored."""
    with pytest.raises(ValidationError) as exc:
        load_study_text(VALID + "\nquantitiez: [pv]\n")
    assert "quantitiez" in str(exc.value)


def test_rejects_a_case_that_is_not_a_mapping():
    broken = VALID.replace("- {name: ordinary}", "- ordinary")
    with pytest.raises(ValidationError):
        load_study_text(broken)


def test_rejects_empty_candidates():
    broken = VALID.split("candidates:")[0] + textwrap.dedent(
        """
        candidates: []
        cases:
          - {name: ordinary}
        """
    )
    with pytest.raises(ValidationError):
        load_study_text(broken)


def test_rejects_non_mapping_document():
    with pytest.raises(ValidationError):
        load_study_text("- just\n- a\n- list\n")


def test_load_study_reads_a_file(tmp_path):
    path = tmp_path / "study.yaml"
    path.write_text(VALID, encoding="utf-8")
    study = load_study(path)
    assert study.name == "yaml-demo"
    assert study.source_text == VALID


def test_load_study_reports_a_missing_file(tmp_path):
    with pytest.raises(ValidationError):
        load_study(tmp_path / "absent.yaml")
