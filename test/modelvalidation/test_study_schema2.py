"""Schema-2 study types: catalogue quantities, per-quantity budgets, normalized scale."""
import pytest

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.study import (
    EXPECTED_STATUSES,
    QUANTITY_CATALOGUE,
    SUPPORTED_SCHEMAS,
    CaseSpec,
    CertificationStudy,
    GateBounds,
    NormalizedScale,
    QuantityBounds,
    SamplingPolicy,
    batch_seed,
)

from conftest import CASE_MEANS_C, OffsetCandidate, SteadyReference


def test_schema_versions():
    assert SUPPORTED_SCHEMAS == (1, 2)
    assert EXPECTED_STATUSES == ("undefined", "failed")


def test_catalogue_names_convention_scale_and_unit():
    assert QUANTITY_CATALOGUE["pv"].scale == "pv" and QUANTITY_CATALOGUE["pv"].convention == "pv"
    assert QUANTITY_CATALOGUE["point_gamma"].convention == "point"
    assert QUANTITY_CATALOGUE["point_gamma"].scale == "gamma"
    assert QUANTITY_CATALOGUE["desk_theta"].scale == "theta_per_hour"
    assert QUANTITY_CATALOGUE["desk_theta"].unit == "PnL per hour"
    assert QUANTITY_CATALOGUE["point_vega"].scale == "point_move"
    assert QUANTITY_CATALOGUE["desk_vega"].scale == "desk_move"
    assert set(QUANTITY_CATALOGUE) == {"pv"} | {
        f"{c}_{m}" for c in ("point", "desk") for m in ("delta", "gamma", "theta", "vega", "rho", "dividend_rho")
    }


def test_quantity_bounds_budget_is_the_larger_of_floor_and_relative_term():
    bounds = QuantityBounds(abs_floor=1e-4, rel=1e-3)
    assert bounds.budget(0.0) == 1e-4
    assert bounds.budget(0.5) == pytest.approx(5e-4)
    assert bounds.budget(-0.5) == pytest.approx(5e-4)
    assert QuantityBounds(abs_floor=1e-6).budget(1e9) == 1e-6
    with pytest.raises(ValidationError):
        QuantityBounds(abs_floor=0.0)
    with pytest.raises(ValidationError):
        QuantityBounds(abs_floor=1e-4, rel=-1.0)


def test_normalized_scale_follows_the_spec_table():
    scale = NormalizedScale(notional=100.0, spot_scale=100.0)
    assert scale.to_economic("pv", 1e-4) == pytest.approx(1e-6)             # PV / N
    assert scale.to_economic("point_delta", 0.5) == pytest.approx(0.5)      # delta * S / N
    assert scale.to_economic("desk_gamma", 0.01) == pytest.approx(1.0)      # gamma * S^2 / N
    assert scale.to_economic("desk_theta", -0.05) == pytest.approx(-5e-4)   # PnL per hour / N
    assert scale.to_economic("point_vega", 20.0) == pytest.approx(2e-3)     # vega * 0.01 / N
    assert scale.to_economic("desk_vega", 0.2) == pytest.approx(2e-3)       # PnL of the move / N
    with pytest.raises(ValidationError, match="catalogue"):
        scale.to_economic("delta", 1.0)                                      # a schema-1 name
    with pytest.raises(ValidationError):
        NormalizedScale(notional=0.0, spot_scale=100.0)


def _study2(**overrides):
    kwargs = dict(
        name="s2",
        schema=2,
        cases=(CaseSpec(name="a", context_params={"valuation": "2026-09-10T14:00:00+08:00"}),),
        quantities=("pv", "point_delta"),
        bounds=GateBounds(cell=1.0, mean_signed_bias=0.2),
        quantity_bounds={"pv": QuantityBounds(1e-6), "point_delta": QuantityBounds(1e-5, 1e-4)},
        scale=NormalizedScale(100.0, 100.0),
        reference=SteadyReference(means_c=CASE_MEANS_C),
        candidates=(OffsetCandidate(means_c=CASE_MEANS_C),),
        sampling=SamplingPolicy(paths_per_batch=8, min_batches=2, max_batches=2, seed=1, seed_scheme="substream"),
    )
    kwargs.update(overrides)
    return CertificationStudy(**kwargs)


def test_schema_2_requires_catalogue_quantities_and_a_bound_per_quantity():
    _study2()
    with pytest.raises(ValidationError, match="quantity_bounds"):
        _study2(quantity_bounds={"pv": QuantityBounds(1e-6)})
    with pytest.raises(ValidationError, match="catalogue"):
        _study2(quantities=("pv", "delta"),
                quantity_bounds={"pv": QuantityBounds(1e-6), "delta": QuantityBounds(1e-5)})
    with pytest.raises(ValidationError, match="quantity_bounds"):
        _study2(quantity_bounds={"pv": QuantityBounds(1e-6), "point_delta": QuantityBounds(1e-5),
                                 "desk_delta": QuantityBounds(1e-5)})


def test_schema_1_rejects_schema_2_fields():
    with pytest.raises(ValidationError, match="schema 1"):
        _study2(schema=1, quantities=("pv",), quantity_bounds={"pv": QuantityBounds(1e-6)},
                cases=(CaseSpec(name="a"),))
    with pytest.raises(ValidationError, match="schema 1"):
        _study2(schema=1, quantities=("pv",), quantity_bounds={},
                cases=(CaseSpec(name="a", expected={"pv": "undefined"}),))


def test_semantic_expectations_name_study_quantities_and_known_statuses():
    _study2(cases=(CaseSpec(name="a", expected={"point_delta": "undefined"}),))
    with pytest.raises(ValidationError, match="expected"):
        _study2(cases=(CaseSpec(name="a", expected={"point_gamma": "undefined"}),))
    with pytest.raises(ValidationError, match="expected"):
        CaseSpec(name="a", expected={"point_delta": "ok"})


def test_unknown_schema_is_rejected():
    with pytest.raises(ValidationError, match="schema"):
        _study2(schema=3)


# --- seed schemes (review R3) --------------------------------------------------------------------------------------
def test_sequential_seeds_are_the_schema_1_rule():
    policy = SamplingPolicy(paths_per_batch=8, min_batches=2, max_batches=4, seed=100)
    assert policy.seed_scheme == "sequential"
    assert [batch_seed(policy, "any case", i) for i in range(3)] == [100, 101, 102]


def test_substream_seeds_are_per_case_deterministic_and_share_nothing_with_a_neighbouring_study_seed():
    policy = SamplingPolicy(paths_per_batch=8, min_batches=2, max_batches=4, seed=20260918, seed_scheme="substream")
    a = [batch_seed(policy, "near_ki_1s", i) for i in range(64)]
    assert a == [batch_seed(policy, "near_ki_1s", i) for i in range(64)] and len(set(a)) == 64
    assert not set(a) & {batch_seed(policy, "ordinary", i) for i in range(64)}              # cases never share a scramble
    pilot = SamplingPolicy(paths_per_batch=8, min_batches=2, max_batches=4, seed=20260917, seed_scheme="substream")
    assert not set(a) & {batch_seed(pilot, "near_ki_1s", i) for i in range(64)}             # nor does a pilot
    assert all(0 <= seed < 2 ** 32 for seed in a)


def test_each_schema_has_its_seed_scheme():
    with pytest.raises(ValidationError, match="substream"):
        _study2(sampling=SamplingPolicy(paths_per_batch=8, min_batches=2, max_batches=2, seed=1))
    with pytest.raises(ValidationError, match="seed_scheme"):
        SamplingPolicy(paths_per_batch=8, min_batches=2, max_batches=2, seed=1, seed_scheme="per_case")
