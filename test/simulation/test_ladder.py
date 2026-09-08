from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.replay.market import SignedDividendYield
from quantark.backtest.simulation.config import CacheConfig
from quantark.backtest.simulation.pricing.base import DayStates, bucket_centre, bucket_key, float_key
from quantark.backtest.simulation.pricing.cache import StateCache
from quantark.backtest.simulation.pricing.repricing import RepricingPricer

from .conftest import RATE, SPOT, pde_engine_config, short_snowball

START = pd.Timestamp("2024-01-02")


def _pricer(**steps) -> RepricingPricer:
    return RepricingPricer(
        short_snowball(), engine_config=pde_engine_config(), start_date=START, underlying="CSI1000",
        cache=StateCache(CacheConfig(memory_bytes=8_000_000)), **steps,
    )


def _states(day_index, spots, *, vol=0.22, q=0.05, knocked_in=None) -> DayStates:
    spots = np.asarray(spots, dtype=float)
    m = spots.size
    vols = np.full(m, vol) if np.isscalar(vol) else np.asarray(vol, dtype=float)
    qs = np.full(m, q) if np.isscalar(q) else np.asarray(q, dtype=float)
    return DayStates(
        day_index=day_index, date=START + pd.Timedelta(days=day_index), path_index=np.arange(m),
        spot=spots, vol=vols, rate=np.full(m, RATE), q_T=qs,
        div_yield=tuple(SignedDividendYield(float(x)) for x in qs), basis_yield=np.full(m, -0.03),
        env_key=float_key(np.full(m, RATE)),
        knocked_in=np.zeros(m, dtype=bool) if knocked_in is None else np.asarray(knocked_in, dtype=bool),
    )


def test_bucket_keys_and_centres():
    values = np.array([0.2149, 0.2151, 0.22, 0.0])
    assert list(bucket_key(values, 0.01)) == [21, 22, 22, 0]
    assert bucket_centre(values, 0.01) == pytest.approx([0.21, 0.22, 0.22, 0.0])
    assert list(bucket_key(values, 0.0)) == list(float_key(values))
    assert bucket_centre(values, None) is values


def test_ladder_nodes_bracket_each_spot():
    pricer = _pricer(spot_step=0.01, vol_step=0.0, q_step=0.0)
    states = _states(3, [SPOT, SPOT * np.exp(0.015), SPOT * np.exp(-0.004)])
    j, w = pricer.ladder_nodes(states)
    assert list(j) == [0, 1, -1]
    assert w == pytest.approx([0.0, 0.5, 0.6])


def test_ladder_prices_interpolate_between_the_exact_nodes():
    exact = _pricer()
    ladder = _pricer(spot_step=0.01, vol_step=0.0, q_step=0.0)
    spots = SPOT * np.exp(np.array([0.0, 0.004, 0.01, 0.0135, -0.006]))
    pv_l, delta_l, gamma_l = ladder.price_day(_states(3, spots))
    pv_e, delta_e, _ = exact.price_day(_states(3, spots))
    assert pv_l[0] == pv_e[0]                                 # on a node: the node's own price, bitwise
    # SPOT * exp(0.01) sits one ulp off node 1 (exp(ln SPOT + 0.01)), so it interpolates
    # with a weight of ~1e-16 instead of taking the node's value outright.
    assert pv_l[2] == pytest.approx(pv_e[2], rel=1e-12)
    assert pv_l == pytest.approx(pv_e, rel=2e-3)              # off a node: linear in x (measured 1.8e-3)
    assert ladder.stats()["engine_calls"] == 4                # nodes -1..2, each once; state 0 needs node 0 only
    assert exact.stats()["engine_calls"] == 5


def test_a_finer_ladder_converges_on_price_and_delta():
    """Six days from expiry the delta doubles across one 1% step, so the
    convergence is what to pin: PV gap 1.8e-3 / 4.2e-4 / 5.5e-5 and delta
    gap 24% / 3.4% / 1.6% at steps 0.01 / 0.005 / 0.002 (measured)."""
    exact = _pricer()
    fine = _pricer(spot_step=0.002, vol_step=0.0, q_step=0.0)
    spots = SPOT * np.exp(np.array([0.0, 0.004, 0.01, 0.0135, -0.006]))
    pv_f, delta_f, gamma_f = fine.price_day(_states(3, spots))
    pv_e, delta_e, gamma_e = exact.price_day(_states(3, spots))
    assert pv_f == pytest.approx(pv_e, rel=1e-4)
    assert delta_f == pytest.approx(delta_e, rel=2.5e-2)
    assert gamma_f == pytest.approx(gamma_e, rel=1e-2)


def test_the_same_node_is_one_cache_key_across_days_and_paths():
    ladder = _pricer(spot_step=0.01, vol_step=0.0, q_step=0.0)
    a = ladder.state_keys(_states(3, [SPOT * np.exp(0.003)]))
    b = ladder.state_keys(_states(3, [SPOT * np.exp(0.007)]))
    assert a == b                                              # both need nodes 0 and 1
    c = ladder.state_keys(_states(4, [SPOT * np.exp(0.003)]))
    assert a != c                                              # a different day is a different key


def test_vol_and_q_buckets_share_a_node_price():
    ladder = _pricer(spot_step=0.01, vol_step=0.01, q_step=0.005)
    states = _states(3, [SPOT, SPOT], vol=[0.2149, 0.2151], q=[0.0501, 0.0499])
    pv, _, _ = ladder.price_day(states)
    assert pv[0] != pv[1]                                      # 0.2149 -> 0.21, 0.2151 -> 0.22
    states = _states(3, [SPOT, SPOT], vol=[0.2226, 0.2174], q=[0.0501, 0.0499])
    pv, _, _ = ladder.price_day(states)
    assert pv[0] == pv[1]                                      # both -> vol 0.22, q 0.05
    assert ladder.stats()["engine_calls"] == 2                 # node 0 at vol 0.21 and at vol 0.22


def test_a_zero_step_keeps_vol_and_q_exact():
    ladder = _pricer(spot_step=0.01, vol_step=0.0, q_step=0.0)
    states = _states(3, [SPOT, SPOT], vol=[0.2200, np.nextafter(0.22, 1.0)])
    pv, _, _ = ladder.price_day(states)
    assert ladder.stats()["engine_calls"] == 2


def test_the_ki_flag_selects_the_product_in_ladder_mode():
    ladder = _pricer(spot_step=0.01, vol_step=0.0, q_step=0.0)
    pv, _, _ = ladder.price_day(_states(3, [SPOT * 0.7, SPOT * 0.7], knocked_in=[False, True]))
    assert pv[0] != pv[1]


def test_the_fingerprint_and_mode_move_with_the_steps():
    assert _pricer().mode == "exact"
    ladder = _pricer(spot_step=0.01, vol_step=0.0, q_step=0.0)
    assert ladder.mode == "ladder"
    assert ladder.fingerprint() != _pricer().fingerprint()
    assert ladder.fingerprint() != _pricer(spot_step=0.02, vol_step=0.0, q_step=0.0).fingerprint()
