from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.replay.market import SignedDividendYield
from quantark.backtest.simulation.config import CacheConfig, GateConfig
from quantark.backtest.simulation.paths.market_path import trading_calendar
from quantark.backtest.simulation.pricing.base import DayStates, GateScale, float_key
from quantark.backtest.simulation.pricing.cache import StateCache
from quantark.backtest.simulation.pricing.repricing import RepricingPricer
from quantark.backtest.simulation.pricing.surface import LifeSurfacePricer
from quantark.util.exceptions import ValidationError

from .conftest import RATE, SPOT, pde_engine_config, short_snowball

DATES = trading_calendar(pd.Timestamp("2024-01-02").date(), 8)
GATE = GateConfig(sample_states=6, pv_tolerance_bp=50.0, delta_tolerance_hands=3.0)
SCALE = GateScale(unit_notional=SPOT, hands_per_unit_delta=5.0)


def _surface(vol_step=0.01, q_step=0.0025, budget=200_000_000, compact=True):
    return LifeSurfacePricer(
        short_snowball(), engine_config=pde_engine_config(), start_date=DATES[0], dates=DATES,
        underlying="CSI1000", vol_step=vol_step, q_step=q_step, surface_cache_bytes=budget, gate=GATE,
        compact=compact,
    )


def _exact():
    return RepricingPricer(short_snowball(), engine_config=pde_engine_config(), start_date=DATES[0],
                           underlying="CSI1000", cache=StateCache(CacheConfig(memory_bytes=8_000_000)))


def _states(day_index, spots, *, vol=0.22, q=0.05, knocked_in=None) -> DayStates:
    spots = np.asarray(spots, dtype=float)
    m = spots.size
    qs = np.full(m, q)
    return DayStates(
        day_index=day_index, date=DATES[day_index], path_index=np.arange(m), spot=spots,
        vol=np.full(m, vol), rate=np.full(m, RATE), q_T=qs,
        div_yield=tuple(SignedDividendYield(float(x)) for x in qs), basis_yield=np.full(m, -0.03),
        env_key=float_key(np.full(m, RATE)),
        knocked_in=np.zeros(m, dtype=bool) if knocked_in is None else np.asarray(knocked_in, dtype=bool),
    )


def test_one_solve_serves_every_day_of_a_bucket():
    pricer = _surface()
    for d in range(0, 5):
        pricer.price_day(_states(d, SPOT * np.exp(np.linspace(-0.02, 0.02, 5))))
    stats = pricer.stats()
    assert stats["solves"] == 1 and stats["surface_cache"]["hits"] == 4


def test_the_readout_tracks_the_exact_engine_within_the_gate():
    pricer, exact = _surface(vol_step=0.0, q_step=0.0), _exact()
    worst = 0.0
    for d in (0, 1, 3, 4):
        states = _states(d, SPOT * np.exp(np.linspace(-0.03, 0.02, 6)))
        pv_s, delta_s, _ = pricer.price_day(states)
        pv_e, delta_e, _ = exact.price_day(states)
        worst = max(worst, float(np.max(np.abs(pv_s - pv_e))) / SPOT * 1e4)
        assert delta_s == pytest.approx(delta_e, abs=0.05)
    assert worst <= GATE.pv_tolerance_bp, worst


def test_day_zero_reads_the_engine_price_exactly_on_the_start_state():
    pricer, exact = _surface(vol_step=0.0, q_step=0.0), _exact()
    pv_s, _, _ = pricer.price_day(_states(0, [SPOT]))
    pv_e, _, _ = exact.price_day(_states(0, [SPOT]))
    assert pv_s[0] == pytest.approx(pv_e[0], rel=1e-6)


def test_the_ki_flag_reads_the_other_slab():
    pricer = _surface()
    # 5% down: inside the six-day product's auto grid (20% down is outside it and fails closed).
    pv, _, _ = pricer.price_day(_states(3, [SPOT * 0.95, SPOT * 0.95], knocked_in=[False, True]))
    assert pv[1] < pv[0]


def test_a_spot_outside_the_grid_fails_closed():
    pricer = _surface()
    with pytest.raises(ValidationError):
        pricer.price_day(_states(3, [SPOT * 50.0]))


def test_buckets_key_the_solve_and_the_budget_evicts():
    small = _surface(budget=1)          # below one surface: every solve evicts the last
    with pytest.raises(ValidationError):
        small.price_day(_states(1, [SPOT]))
    pricer = _surface()
    pricer.price_day(_states(1, [SPOT], vol=0.2149))
    pricer.price_day(_states(1, [SPOT], vol=0.2151))
    pricer.price_day(_states(1, [SPOT], vol=0.2226))
    assert pricer.stats()["solves"] == 2


def test_the_gate_reprices_the_reservoir_exactly():
    pricer = _surface(vol_step=0.0, q_step=0.0)
    for d in (1, 2, 3):
        pricer.price_day(_states(d, SPOT * np.exp(np.linspace(-0.03, 0.02, 4))))
    samples = pricer.sample_visited()
    assert len(samples) == 6
    report = pricer.verify(samples, GATE, SCALE)
    assert report.mode == "life_surface" and report.sampled == 6
    assert report.passed, report
    tight = GateConfig(sample_states=6, pv_tolerance_bp=1e-9, delta_tolerance_hands=1e-9)
    assert not pricer.verify(samples, tight, SCALE).passed


def test_the_fingerprint_names_the_provider_and_its_steps():
    assert _surface().fingerprint() != _exact().fingerprint()
    assert _surface(vol_step=0.02).fingerprint() != _surface().fingerprint()
    assert _surface().mode == "life_surface"


def test_select_keeps_columns_bit_for_bit_and_remaps_the_nodes():
    pricer = _surface(vol_step=0.0, q_step=0.0, compact=False)
    pricer.price_day(_states(1, [SPOT]))
    (key, full), = pricer._surfaces._store.items()
    node = full.node(pricer.elapsed(2))
    sub = full.select([0, node, full.t.size - 1])
    assert sub.t.size == 3 and sub.nbytes < full.nbytes / 5      # 3 of ~25 columns on the six-day fixture
    assert sub.node(pricer.elapsed(2)) == 1 and sub.step_of[pricer.elapsed(2)] == 1
    x = np.log(SPOT * np.exp(np.array([-0.01, 0.0, 0.004])))
    for ki in (False, True):
        flags = np.full(3, ki)
        for a, b in zip(full.readout(x, node, flags), sub.readout(x, 1, flags)):
            assert np.array_equal(a, b)
    assert np.array_equal(sub.v1[:, -1], full.v1[:, -1])
    with pytest.raises(ValidationError):
        full.select([full.t.size])


def test_a_compact_pricer_prices_exactly_like_the_full_one_with_less_memory():
    full, compact = _surface(vol_step=0.0, q_step=0.0, compact=False), _surface(vol_step=0.0, q_step=0.0)
    for d in range(0, 5):
        states = _states(d, SPOT * np.exp(np.linspace(-0.02, 0.02, 5)), knocked_in=[False, True, False, True, False])
        for a, b in zip(full.price_day(states), compact.price_day(states)):
            assert np.array_equal(a, b)
    assert compact.stats()["surface_cache"]["bytes_used"] < full.stats()["surface_cache"]["bytes_used"] / 2
