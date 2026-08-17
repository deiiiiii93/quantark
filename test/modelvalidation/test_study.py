"""Tests for quantark.modelvalidation.study core types."""

import pytest

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.study import (
    QUANTITIES,
    CaseSpec,
    CertificationStudy,
    GateBounds,
    HedgeContractScale,
    SamplingPolicy,
)


def _scale() -> HedgeContractScale:
    return HedgeContractScale(
        hedge_multiplier=200.0,
        hedge_inception_spot=100.0,
        notional=50_000_000.0,
    )


def test_hedge_contract_scale_formulas():
    scale = _scale()
    dq = 200.0 * 100.0 / 50_000_000.0  # 4e-4 delta per contract
    assert scale.delta_quantum == pytest.approx(dq)
    assert scale.to_economic("delta", 2 * dq) == pytest.approx(2.0)
    assert scale.to_economic("gamma", 1.0) == pytest.approx(0.01 * 100.0 / dq)
    assert scale.to_economic("pv", dq * 0.01 * 100.0) == pytest.approx(1.0)


def test_hedge_contract_scale_is_linear():
    """Errors and SEs convert with the same linear map -- gates rely on this."""
    scale = _scale()
    for quantity in QUANTITIES:
        assert scale.to_economic(quantity, 3.0) == pytest.approx(
            3.0 * scale.to_economic(quantity, 1.0)
        )


def test_hedge_contract_scale_rejects_unknown_quantity():
    with pytest.raises(ValidationError):
        _scale().to_economic("vega", 1.0)


def test_hedge_contract_scale_validates_inputs():
    with pytest.raises(ValidationError):
        HedgeContractScale(
            hedge_multiplier=0.0, hedge_inception_spot=100.0, notional=1.0
        )
    with pytest.raises(ValidationError):
        HedgeContractScale(
            hedge_multiplier=200.0, hedge_inception_spot=100.0, notional=-1.0
        )


def test_gate_bounds_validation():
    with pytest.raises(ValidationError):
        GateBounds(cell=-0.5, mean_signed_bias=0.1)
    with pytest.raises(ValidationError):
        GateBounds(cell=0.5, mean_signed_bias=0.0)
    with pytest.raises(ValidationError):
        GateBounds(cell=0.5, mean_signed_bias=0.1, se_budget_fraction=0.0)
    with pytest.raises(ValidationError):
        GateBounds(cell=0.5, mean_signed_bias=0.1, interval_k=-1.0)


def test_gate_bounds_defaults():
    bounds = GateBounds(cell=0.5, mean_signed_bias=0.1)
    assert bounds.se_budget_fraction == 0.25
    assert bounds.interval_k == 2.0
    assert bounds.envelope_fraction == 0.5


def test_sampling_policy_validation():
    with pytest.raises(ValidationError):
        SamplingPolicy(paths_per_batch=1024, min_batches=1, max_batches=8, seed=7)
    with pytest.raises(ValidationError):
        SamplingPolicy(paths_per_batch=1024, min_batches=4, max_batches=2, seed=7)
    with pytest.raises(ValidationError):
        SamplingPolicy(paths_per_batch=0, min_batches=4, max_batches=8, seed=7)
    with pytest.raises(ValidationError):
        SamplingPolicy(
            paths_per_batch=1024, min_batches=4, max_batches=8, seed=7, bump=0.0
        )


def test_case_spec_defaults():
    case = CaseSpec(name="ordinary")
    assert case.environment_params == {}
    assert case.product_params == {}


def test_case_spec_requires_name():
    with pytest.raises(ValidationError):
        CaseSpec(name="")


class _StubArm:
    """Stand-in for a reference builder / candidate evaluator.

    The study does not type-check these in v1; it only validates structure.
    """

    def name(self) -> str:
        return "stub"


def _study(**overrides) -> CertificationStudy:
    kwargs = dict(
        name="demo",
        schema=1,
        cases=(CaseSpec(name="ordinary"), CaseSpec(name="near_ko")),
        quantities=("pv", "delta"),
        bounds=GateBounds(cell=0.5, mean_signed_bias=0.1),
        scale=_scale(),
        reference=_StubArm(),
        candidates=(_StubArm(),),
        sampling=SamplingPolicy(
            paths_per_batch=1024, min_batches=2, max_batches=8, seed=7
        ),
    )
    kwargs.update(overrides)
    return CertificationStudy(**kwargs)


def test_certification_study_accepts_valid_config():
    study = _study()
    assert study.name == "demo"
    assert study.source_text is None


def test_certification_study_rejects_wrong_schema():
    with pytest.raises(ValidationError):
        _study(schema=2)


def test_certification_study_rejects_duplicate_case_names():
    with pytest.raises(ValidationError):
        _study(cases=(CaseSpec(name="ordinary"), CaseSpec(name="ordinary")))


def test_certification_study_rejects_unknown_quantity():
    with pytest.raises(ValidationError):
        _study(quantities=("vega",))


def test_certification_study_rejects_empty_collections():
    with pytest.raises(ValidationError):
        _study(cases=())
    with pytest.raises(ValidationError):
        _study(candidates=())
    with pytest.raises(ValidationError):
        _study(quantities=())
