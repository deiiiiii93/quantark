from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.replay.engine_factory import create_pricing_engine
from quantark.backtest.replay.market import ImpliedBasisYield, SignedDividendYield
from quantark.backtest.simulation.config import CacheConfig, GateConfig
from quantark.backtest.simulation.pricing.base import DayStates, float_key
from quantark.backtest.simulation.pricing.cache import StateCache
from quantark.backtest.simulation.pricing.repricing import RepricingPricer, product_fingerprint
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment

from .conftest import RATE, SPOT, pde_engine_config, short_snowball

START = pd.Timestamp("2024-01-02")


def _pricer(cache_bytes: int = 8_000_000) -> RepricingPricer:
    return RepricingPricer(
        short_snowball(), engine_config=pde_engine_config(), start_date=START,
        underlying="CSI1000", cache=StateCache(CacheConfig(memory_bytes=cache_bytes)),
    )


def _states(day_index: int, spots, *, knocked_in=None, vol: float = 0.22) -> DayStates:
    spots = np.asarray(spots, dtype=float)
    m = spots.size
    q = np.full(m, 0.05)
    return DayStates(
        day_index=day_index, date=START + pd.Timedelta(days=day_index),
        path_index=np.arange(m), spot=spots, vol=np.full(m, vol),
        rate=np.full(m, RATE), q_T=q, div_yield=tuple(SignedDividendYield(float(x)) for x in q),
        basis_yield=np.full(m, -0.03),
        env_key=float_key(np.full(m, RATE)),
        knocked_in=np.zeros(m, dtype=bool) if knocked_in is None else np.asarray(knocked_in, dtype=bool),
    )


def test_the_aged_product_does_not_depend_on_the_market():
    pricer = _pricer()
    day = START + pd.Timedelta(days=3)
    a = pricer.aged_product(day, knocked_in=False)
    pricer.reset_aging_memo()
    b = pricer.aged_product(day, knocked_in=False)
    assert a.maturity == pytest.approx(b.maturity)
    assert a.barrier_config.ko_observation_dates == pytest.approx(b.barrier_config.ko_observation_dates)
    assert a.barrier_config.ki_observation_dates == pytest.approx(b.barrier_config.ki_observation_dates)


def test_exact_mode_equals_a_direct_engine_call_state_by_state():
    pricer = _pricer()
    day_index, day = 3, START + pd.Timedelta(days=3)
    states = _states(day_index, [SPOT * 0.9, SPOT, SPOT * 1.02])
    pv, delta, gamma = pricer.price_day(states)
    product = pricer.aged_product(day, knocked_in=False)
    engine = create_pricing_engine(product, pde_engine_config())
    for n in range(len(states)):
        env = PricingEnvironment(
            spot_quote=SpotQuote(spot=float(states.spot[n]), asset_name="CSI1000"),
            vol_surface=FlatVolSurface(volatility=float(states.vol[n])),
            rate_curve=FlatRateCurve(rate=float(states.rate[n])),
            div_yield=states.div_yield[n],
            basis_yield=ImpliedBasisYield(float(states.basis_yield[n])),
            valuation_date=day.to_pydatetime(),
        )
        assert pv[n] == float(engine.price(product, env))
        greeks = engine.calculate_greeks(product, env)
        assert delta[n] == float(greeks["delta"])
        assert gamma[n] == float(greeks["gamma"])


def test_the_ki_flag_selects_a_different_product_and_price():
    pricer = _pricer()
    states = _states(3, [SPOT * 0.7, SPOT * 0.7], knocked_in=[False, True])
    pv, _, _ = pricer.price_day(states)
    assert pv[0] != pv[1]
    assert pricer.aged_product(START + pd.Timedelta(days=3), knocked_in=True)._otc_lifecycle_knocked_in is True


def test_a_repeated_state_is_a_cache_hit_and_not_a_second_engine_call():
    pricer = _pricer()
    states = _states(3, [SPOT, SPOT, SPOT * 1.01])
    pv, delta, _ = pricer.price_day(states)
    assert pv[0] == pv[1] and delta[0] == delta[1]
    assert pricer.stats()["engine_calls"] == 2            # deduplicated within the day
    pricer.price_day(states)
    assert pricer.stats()["engine_calls"] == 2            # served from the cache
    assert pricer.stats()["cache"]["hits"] >= 3


def test_an_empty_day_prices_nothing():
    pricer = _pricer()
    pv, delta, gamma = pricer.price_day(_states(3, []))
    assert pv.shape == delta.shape == gamma.shape == (0,)
    assert pricer.stats()["engine_calls"] == 0


def test_the_exact_gate_reports_a_zero_gap():
    pricer = _pricer()
    report = pricer.verify(_states(3, [SPOT]), GateConfig(sample_states=5, pv_tolerance_bp=1.0,
                                                          delta_tolerance_hands=0.1))
    assert report.mode == "exact" and report.passed
    assert report.max_pv_gap_bp == 0.0 and report.max_delta_gap_hands == 0.0


def test_the_fingerprint_moves_with_the_engine_settings():
    a = _pricer().fingerprint()
    b = RepricingPricer(short_snowball(), engine_config=pde_engine_config(), start_date=START,
                        underlying="CSI1000",
                        cache=StateCache(CacheConfig(memory_bytes=1024)), delta_bump_size=0.02).fingerprint()
    assert a != b and len(a) == 32


def test_the_product_fingerprint_sees_every_contract_term():
    # SnowballOption.__repr__ is a 4-decimal summary that does not mention
    # the observation days; two products that differ only there must not
    # share cache entries.
    same = product_fingerprint(short_snowball())
    assert same == product_fingerprint(short_snowball())
    assert same != product_fingerprint(short_snowball(ko_days=(3, 5)))
    assert same != product_fingerprint(short_snowball(ki_days=(1, 4)))
    knocked_in = short_snowball()
    knocked_in._otc_lifecycle_knocked_in = True            # lifecycle state is NOT a term
    assert same == product_fingerprint(knocked_in)
