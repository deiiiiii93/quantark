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


def test_rate_keyrate_dividend_held_is_the_forward_moving_sensitivity():
    """Patch spec 2026-09-03 §6: opt-in DIVIDEND_HELD holds q, so the forward moves with r."""
    import math
    from datetime import datetime
    from scipy.stats import norm
    from quantark.asset.equity.engine.analytical import BlackScholesEngine
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.asset.equity.riskmeasures import RateKeyrateConvention
    from quantark.param import FlatVolSurface, SpotQuote
    from quantark.param.div import ContinuousDividendYield
    from quantark.param.rrf.rate_curve import LinearRateCurve
    from quantark.priceenv import PricingEnvironment
    from quantark.util.enum import OptionType
    from quantark.util.exceptions import ValidationError

    S, K, T, r, q, sig = 100.0, 100.0, 1.0, 0.03, 0.01, 0.2
    env = PricingEnvironment(spot_quote=SpotQuote(spot=S), vol_surface=FlatVolSurface(sig),
                             rate_curve=LinearRateCurve([(0.5, r), (1.0, r), (2.0, r)]),
                             div_yield=ContinuousDividendYield(q), valuation_date=datetime(2026, 1, 5))
    call = EuropeanVanillaOption(strike=K, option_type=OptionType.CALL, maturity=T)
    calc = GreeksCalculator()

    def run(conv):
        return calc.calculate_bucketed_greeks(call, env, BlackScholesEngine(), request=BucketedGreeksRequest(
            coordinates=(BucketedGreekCoordinate.RATE_KEYRATE,), rate_keyrate_convention=conv))

    held = run(RateKeyrateConvention.DIVIDEND_HELD)
    carry = run(RateKeyrateConvention.CARRY_INVARIANT)
    par_held = [p for p in held.points if p.name == "rate_keyrate.parallel"][0]
    par_carry = [p for p in carry.points if p.name == "rate_keyrate.parallel"][0]
    d2 = (math.log(S / K) + (r - q - 0.5 * sig**2) * T) / (sig * math.sqrt(T))
    bs_rho = K * T * math.exp(-r * T) * norm.cdf(d2)                 # dV/dr, dividend held
    assert par_held.derivative == pytest.approx(bs_rho, rel=1e-6)
    assert par_held.derivative > 0.0 > par_carry.derivative           # the two conventions differ in sign for a call
    assert all(p.metadata["convention"] == "dividend_held" for p in held.points)
    assert all(p.metadata["convention"] == "carry_invariant" for p in carry.points)
    assert all("dividend yield held" in p.metadata["rebuild_rule"] for p in held.points)
    assert held.metadata["rate_keyrate_convention"] == "dividend_held"
    assert carry.metadata["rate_keyrate_convention"] == "carry_invariant"
    # default request == carry-invariant, bitwise
    default = calc.calculate_bucketed_greeks(call, env, BlackScholesEngine(),
                                             request=BucketedGreeksRequest(coordinates=(BucketedGreekCoordinate.RATE_KEYRATE,)))
    assert [p.reported for p in default.points] == [p.reported for p in carry.points]
    with pytest.raises(ValidationError, match="rate_keyrate_convention"):
        BucketedGreeksRequest(coordinates=(BucketedGreekCoordinate.RATE_KEYRATE,), rate_keyrate_convention="dividend_held")
