"""Bucketed curve risk (spec WP3.3): RATE_KEYRATE + node-aligned buckets."""
import pytest

from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.riskmeasures.bucketed_greeks import (
    BucketedGreekCoordinate,
    BucketedGreeksRequest,
)
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.param.rrf.key_rate import key_rate_bumped_zero_curve

from dcn_fixtures import DCN_A, FLAT, make_dcn, term_env

PATHS = 2 ** 13


def _run(env, coordinate, **request_kw):
    return GreeksCalculator().calculate_bucketed_greeks(
        make_dcn(DCN_A),
        env,
        DCNMCEngine(num_paths=PATHS, seed=42),
        request=BucketedGreeksRequest(coordinates=(coordinate,), **request_kw),
    )


def test_key_rate_bump_moves_only_local_pillar():
    from quantark.param.rrf.rate_curve import LinearRateCurve

    curve = LinearRateCurve([(0.5, 0.03), (1.0, 0.035), (2.0, 0.04)])
    bumped = key_rate_bumped_zero_curve(curve, 1.0, 1e-4)
    assert bumped.get_rate(1.0) == pytest.approx(0.035 + 1e-4)
    assert bumped.get_rate(0.5) == pytest.approx(0.03)   # neighbor fixed
    assert bumped.get_rate(2.0) == pytest.approx(0.04)
    assert bumped.get_rate(0.75) > 0.0325                # triangle in between
    assert curve.get_rate(1.0) == pytest.approx(0.035)   # input not mutated


def test_rate_keyrate_buckets_reconcile_to_parallel():
    # pillar triangle bumps sum to the parallel shift in zero-rate space, so
    # the FD gradients must reconcile. The residual has two parts: discrete
    # KI/KO indicator noise (~1/n_paths: 9% @ 2^13 -> 2.2% @ 2^16) and a
    # ~2% floor from finite-bump cross-third-order terms near the barriers
    # (present at 2^17 too). Gate at 3%; the result also emits both numbers
    # per spec WP3.3 so the report can show them side by side.
    res = GreeksCalculator().calculate_bucketed_greeks(
        make_dcn(DCN_A),
        term_env(**FLAT),
        DCNMCEngine(num_paths=2 ** 16, seed=42),
        request=BucketedGreeksRequest(
            coordinates=(BucketedGreekCoordinate.RATE_KEYRATE,)
        ),
    )
    pillar_points = [pt for pt in res.points if "parallel" not in pt.name]
    assert len(pillar_points) == 3            # one per CALIBRATED pillar
    md = res.metadata
    assert "sum_of_buckets" in md and "parallel" in md and "reconciles" in md
    assert md["sum_of_buckets"] == pytest.approx(md["parallel"], rel=0.03)
    assert md["reconciles"] is True


def test_rate_keyrate_units_per_1bp():
    res = _run(term_env(**FLAT), BucketedGreekCoordinate.RATE_KEYRATE)
    for pt in res.points:
        assert pt.metadata.get("unit") == "per_1bp"


def test_carry_rhoq_buckets_align_to_curve_nodes():
    res = _run(term_env(**FLAT), BucketedGreekCoordinate.CARRY_RHOQ)
    tenors = sorted(
        pt.maturity for pt in res.points
        if pt.coordinate == BucketedGreekCoordinate.CARRY_RHOQ
    )
    assert tenors == [0.5, 1.0, 2.0]


def test_vol_tenor_vega_node_aligned_and_per_volpt():
    res = _run(term_env(**FLAT), BucketedGreekCoordinate.VOL_TENOR_VEGA)
    pts = [pt for pt in res.points
           if pt.coordinate == BucketedGreekCoordinate.VOL_TENOR_VEGA]
    assert sorted(pt.maturity for pt in pts) == [0.5, 1.0, 2.0]
    assert all(pt.metadata.get("unit") == "per_1volpt" for pt in pts)
    assert all(pt.metadata.get("adjusted_to_one_sided") is False for pt in pts)


def test_vol_bucket_one_sided_fallback_recorded():
    # near-flat calendar between 0.5y and 1.0y: a -1pt bump at the 1.0y node
    # sends w(1.0) below w(0.5) -> NumericalError -> one-sided-up recorded
    from quantark.param.vol.vol_surface import TermStructureVolSurface

    env = term_env(**FLAT)
    env.vol_surface = TermStructureVolSurface(
        times=[0.5, 1.0, 2.0], vols=[0.20, 0.1436, 0.15]
    )
    res = _run(env, BucketedGreekCoordinate.VOL_TENOR_VEGA)
    flagged = [pt for pt in res.points
               if pt.metadata.get("adjusted_to_one_sided")]
    assert flagged, "expected at least one recorded one-sided fallback"
    assert all(
        pt.difference_mode in ("one_sided_up", "one_sided_down")
        for pt in flagged if pt.status == "ok"
    )


def test_rate_keyrate_carry_invariant_with_none_div_yield():
    # review regression: div_yield=None means zero yield; the carry-invariant
    # wrapper must apply there too, so None and an explicit zero-dividend
    # environment produce IDENTICAL key-rate risk (pure discounting)
    from copy import deepcopy

    from quantark.param.div import ContinuousDividendYield

    env_zero = term_env(r=FLAT["r"], q=0.0, sigma=FLAT["sigma"])
    env_none = deepcopy(env_zero)
    env_none.div_yield = None
    env_zero.div_yield = ContinuousDividendYield(0.0)

    def _points(env):
        res = GreeksCalculator().calculate_bucketed_greeks(
            make_dcn(DCN_A), env, DCNMCEngine(num_paths=PATHS, seed=42),
            request=BucketedGreeksRequest(
                coordinates=(BucketedGreekCoordinate.RATE_KEYRATE,)
            ),
        )
        return {pt.name: pt.reported for pt in res.points}

    assert _points(env_none) == _points(env_zero)
