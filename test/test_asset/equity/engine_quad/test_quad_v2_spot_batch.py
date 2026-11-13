"""QUAD V2 scoped spot-batch session (ship spec 2026-10-23 C1, D7, D10–D12, AC4).

Inside ``engine.spot_batch((lo, hi))`` a call reuses a prepared context when the engine's own cache key (plus the
compiled contract's metadata and the event phase) is equal and its spot is in range. The first use of a key
prepares the scalar's narrow context; a later use outside it prepares over the declared range.
"""
import copy
import math
from datetime import datetime

import numpy as np
import pytest

import quantark.asset.equity.engine.quad.v2.engine as v2_engine
from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.quad.v2 import PhoenixQuadEngineV2, SnowballQuadEngineV2
from quantark.asset.equity.param.quad_v2_params import QuadV2Params
from quantark.asset.equity.product.option import phoenix_helpers
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import NumericalError, ValidationError
from quantark.cashleg.event_distribution import EventType

VAL = datetime(2026, 3, 16)
LADDER = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0]
RANGE = (75.0, 115.0)


def _env(spot=100.0, vol=0.25, rate=0.03):
    return PricingEnvironment(spot_quote=SpotQuote(spot), vol_surface=FlatVolSurface(vol),
                              rate_curve=FlatRateCurve(rate), div_yield=ContinuousDividendYield(0.01),
                              valuation_date=VAL)


def _snowball():
    """Short continuous-KI snowball: cheap at the default V2 discretisation."""
    return SnowballOption(initial_price=100.0, strike=100.0, maturity=0.5,
                          barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15,
                                                       ko_observation_dates=[1 / 6, 2 / 6, 0.5],
                                                       ki_barrier=75.0, ki_continuous=True))


def _phoenix():
    return phoenix_helpers.create_standard_phoenix(initial_price=100.0, strike=100.0, maturity=0.5,
                                                   ko_barrier=103.0, ki_barrier=75.0, coupon_barrier=80.0,
                                                   coupon_rate=0.01, num_observations=6)


@pytest.fixture
def prepares(monkeypatch):
    """Counts every compiled-contract prepare the engine runs."""
    calls = []
    original = v2_engine.prepare_compiled

    def counting(contract, env, strike, params, spots, **kwargs):
        calls.append((float(min(spots)), float(max(spots))))
        return original(contract, env, strike, params, spots, **kwargs)

    monkeypatch.setattr(v2_engine, "prepare_compiled", counting)
    return calls


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def test_capability_flags():
    assert BaseEngine.supports_spot_batch is False
    assert SnowballQuadEngineV2.supports_spot_batch is True


@pytest.mark.parametrize("make_engine,make_product,ladder", [(SnowballQuadEngineV2, _snowball, LADDER),
                                                              (PhoenixQuadEngineV2, _phoenix, [100.0, 85.0, 110.0])])
def test_session_prices_match_the_scalar_engine(make_engine, make_product, ladder):
    product = make_product()
    scalar = [make_engine().price(product, _env(s)) for s in ladder]
    engine = make_engine()
    with engine.spot_batch(RANGE):
        batched = [engine.price(product, _env(s)) for s in ladder]
    assert batched[0] == scalar[0]                        # the first use of a key is the scalar's own context
    assert max(_rel(b, s) for b, s in zip(batched, scalar)) <= 1e-12


def test_one_narrow_and_one_wide_prepare_per_key(prepares):
    product, engine = _snowball(), SnowballQuadEngineV2()
    with engine.spot_batch(RANGE):
        for s in LADDER:
            engine.price(product, _env(s))
    assert len(prepares) == 2
    assert prepares[1] == RANGE


def test_without_a_session_each_far_spot_reprepares(prepares):
    product, engine = _snowball(), SnowballQuadEngineV2()
    for s in LADDER:
        engine.price(product, _env(s))
    assert len(prepares) == len(LADDER)


def test_distinct_keys_never_share_a_context():
    product, engine = _snowball(), SnowballQuadEngineV2()
    knocked = copy.deepcopy(product)
    knocked._otc_lifecycle_knocked_in = True
    states = [(product, {}), (product, {"vol": 0.30}), (product, {"rate": 0.04}), (knocked, {})]
    want = [SnowballQuadEngineV2().price(p, _env(90.0, **m)) for p, m in states]
    with engine.spot_batch(RANGE):
        for p, m in states:
            engine.price(p, _env(100.0, **m))             # each key's first use: its narrow context
        got = [engine.price(p, _env(90.0, **m)) for p, m in states]      # second use: its own wide context
    assert max(_rel(g, w) for g, w in zip(got, want)) <= 1e-12
    assert len(set(want)) == len(want)                    # the four states really price differently


def test_event_stats_and_events_contract_match_scalar():
    product, engine = _snowball(), SnowballQuadEngineV2()
    want = [SnowballQuadEngineV2().calculate_event_stats(product, _env(s)) for s in (100.0, 90.0)]
    with engine.spot_batch(RANGE):
        got = [engine.calculate_event_stats(product, _env(s)) for s in (100.0, 90.0)]
    for g, w in zip(got, want):
        assert _rel(g.pv, w.pv) <= 1e-12
        np.testing.assert_allclose(g.observations.probability, w.observations.probability, rtol=1e-10, atol=1e-13)
        np.testing.assert_array_equal(g.observations.times, w.observations.times)
        assert _rel(g.knock_in.probability, w.knock_in.probability) <= 1e-10


def test_folded_and_unfolded_keys_are_separate():
    from test_cashleg._autocallable_helpers import make_env, make_snowball
    from test_cashleg.test_folding import _leg_set, _route_dist
    from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
    from quantark.asset.equity.param import PDEParams
    from quantark.cashleg import fold_cash_legs

    env, product = make_env(), make_snowball(ko_dates=(0.25, 0.5, 0.75, 1.0))
    legs = _leg_set(_route_dist(SnowballPDESolver(PDEParams()), product, env).stream_times[EventType.KO])
    overlay = fold_cash_legs(legs, product, env, quantity=1.0, unit_notional=1.0)
    plain_want = SnowballQuadEngineV2().price(product, env)
    folded_want = SnowballQuadEngineV2().price_folded(product, env, overlay)
    engine = SnowballQuadEngineV2()
    with engine.spot_batch((env.spot * 0.8, env.spot * 1.2)):
        assert engine.price(product, env) == plain_want
        assert engine.price_folded(product, env, overlay) == folded_want
        assert engine.price(product, env) == plain_want


def test_out_of_range_spot_takes_the_scalar_path(prepares):
    product, engine = _snowball(), SnowballQuadEngineV2()
    want = SnowballQuadEngineV2().price(product, _env(130.0))
    with engine.spot_batch(RANGE):
        engine.price(product, _env(100.0))
        assert engine.price(product, _env(130.0)) == want
        assert len(engine._batch.contexts) == 1            # the out-of-range call never entered the session
    assert RANGE not in prepares                           # never widened for a spot it cannot cover


def test_wide_prepare_failure_falls_back_narrow_once_per_key(prepares):
    product = _snowball()
    probe = SnowballQuadEngineV2()
    narrow = probe.prepare(product, _env(100.0)).diagnostics["nodes"]
    wide = probe.prepare(product, _env(100.0), [RANGE[0], RANGE[1]]).diagnostics["nodes"]
    assert narrow < wide
    params = QuadV2Params(max_nodes=(narrow + wide) // 2)
    want = [SnowballQuadEngineV2(params).price(product, _env(s)) for s in LADDER]
    engine = SnowballQuadEngineV2(params)
    prepares.clear()
    with engine.spot_batch(RANGE):
        got = [engine.price(product, _env(s)) for s in LADDER]
    assert got == want
    assert prepares.count(RANGE) == 1                      # the failed wide prepare is not retried


def test_exit_restores_the_single_slot_cache():
    product, engine = _snowball(), SnowballQuadEngineV2()
    engine.price(product, _env(100.0))
    before = (engine._prepared, engine._prepared_key)
    with engine.spot_batch(RANGE):
        for s in LADDER:
            engine.price(product, _env(s))
    assert engine._prepared is before[0] and engine._prepared_key is before[1]
    assert engine._batch is None


def test_session_state_is_dropped_by_copies_and_pickles():
    import pickle

    product, engine = _snowball(), SnowballQuadEngineV2()
    with engine.spot_batch(RANGE):
        engine.price(product, _env(90.0))
        for clone in (copy.deepcopy(engine), pickle.loads(pickle.dumps(engine))):
            assert clone._batch is None and clone._prepared is None


def test_small_context_budget_stays_correct():
    product, engine = _snowball(), SnowballQuadEngineV2()
    want = [SnowballQuadEngineV2().price(product, _env(s, vol=v)) for s in (90.0, 110.0) for v in (0.2, 0.3)]
    with engine.spot_batch(RANGE, max_contexts=1):
        got = [engine.price(product, _env(s, vol=v)) for s in (90.0, 110.0) for v in (0.2, 0.3)]
    assert max(_rel(g, w) for g, w in zip(got, want)) <= 1e-12


@pytest.mark.parametrize("spot_range,max_contexts", [((0.0, 10.0), 4), ((10.0, 5.0), 4), ((math.nan, 5.0), 4),
                                                       ((5.0, math.inf), 4), ((5.0, 10.0), 0), ((5.0, 10.0), True),
                                                       ((5.0,), 4), (("a", "b"), 4), (None, 4)])
def test_bad_session_arguments_raise(spot_range, max_contexts):
    with pytest.raises(ValidationError):
        with SnowballQuadEngineV2().spot_batch(spot_range, max_contexts=max_contexts):
            pass


def test_sessions_do_not_nest():
    engine = SnowballQuadEngineV2()
    with engine.spot_batch(RANGE):
        with pytest.raises(ValidationError):
            with engine.spot_batch(RANGE):
                pass
    assert engine._batch is None


def test_unsupported_market_raises_inside_a_session_too():
    product = _snowball()
    env = _env()
    smile = copy.deepcopy(env)
    smile.vol_surface = _smile_surface()
    with pytest.raises(NotImplementedError) as scalar:
        SnowballQuadEngineV2().price(product, smile)
    engine = SnowballQuadEngineV2()
    with engine.spot_batch(RANGE):
        with pytest.raises(NotImplementedError) as batched:
            engine.price(product, smile)
    assert str(batched.value) == str(scalar.value)


def _smile_surface():
    from quantark.param.vol.vol_surface import GridVolSurface

    strikes = np.array([60.0, 80.0, 100.0, 120.0, 140.0])
    times = np.array([0.25, 0.5, 1.0])
    vols = np.array([[0.32, 0.28, 0.25, 0.24, 0.24]] * 3)
    return GridVolSurface(strikes=list(strikes), maturities=list(times), iv_grid=vols)


def test_unsupported_engines_do_not_offer_the_session():
    from quantark.asset.equity.engine.pde import SnowballPDESolver
    from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine

    for engine_type in (SnowballPDESolver, SnowballQuadEngine):
        assert getattr(engine_type, "supports_spot_batch", False) is False
