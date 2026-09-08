from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.replay.market import SignedDividendYield
from quantark.backtest.simulation.config import CacheConfig, GateConfig
from quantark.backtest.simulation.pricing.base import DayStates, GateFailure, GateReport, GateScale, float_key
from quantark.backtest.simulation.pricing.cache import StateCache
from quantark.backtest.simulation.pricing.repricing import RepricingPricer

from .conftest import RATE, SPOT, pde_engine_config, short_snowball

START = pd.Timestamp("2024-01-02")
SCALE = GateScale(unit_notional=SPOT, hands_per_unit_delta=1000.0 / 200.0)


def _pricer(gate: GateConfig, **steps) -> RepricingPricer:
    return RepricingPricer(
        short_snowball(), engine_config=pde_engine_config(), start_date=START, underlying="CSI1000",
        cache=StateCache(CacheConfig(memory_bytes=8_000_000)), gate=gate, **steps,
    )


def _states(day_index, spots) -> DayStates:
    spots = np.asarray(spots, dtype=float)
    m = spots.size
    q = np.full(m, 0.05)
    return DayStates(
        day_index=day_index, date=START + pd.Timedelta(days=day_index), path_index=np.arange(m),
        spot=spots, vol=np.full(m, 0.22), rate=np.full(m, RATE), q_T=q,
        div_yield=tuple(SignedDividendYield(float(x)) for x in q), basis_yield=np.full(m, -0.03),
        env_key=float_key(np.full(m, RATE)), knocked_in=np.zeros(m, dtype=bool),
    )


def test_exact_mode_never_samples_and_reports_zero():
    pricer = _pricer(GateConfig(sample_states=4, pv_tolerance_bp=0.1, delta_tolerance_hands=0.1))
    pricer.price_day(_states(3, SPOT * np.exp(np.linspace(-0.02, 0.02, 6))))
    assert pricer.sample_visited() == []
    report = pricer.verify([], pricer.gate, SCALE)
    assert report.mode == "exact" and report.passed and report.max_pv_gap_bp == 0.0


def test_the_reservoir_is_a_deterministic_sample_of_visited_states():
    gate = GateConfig(sample_states=3, pv_tolerance_bp=100.0, delta_tolerance_hands=10.0)
    spots = SPOT * np.exp(np.linspace(-0.03, 0.03, 10))
    a, b = _pricer(gate, spot_step=0.01, vol_step=0.0, q_step=0.0), _pricer(gate, spot_step=0.01, vol_step=0.0, q_step=0.0)
    for day in (3, 4):
        a.price_day(_states(day, spots))
        b.price_day(_states(day, spots))
    sa, sb = a.sample_visited(), b.sample_visited()
    assert len(sa) == 3 and [(s.day_index, float(s.spot[0])) for s in sa] == [(s.day_index, float(s.spot[0])) for s in sb]
    assert len({float(s.spot[0]) for s in sa}) == 3


def test_a_coarse_ladder_fails_a_tight_gate_and_passes_a_loose_one():
    tight = GateConfig(sample_states=6, pv_tolerance_bp=0.01, delta_tolerance_hands=0.01)
    loose = GateConfig(sample_states=6, pv_tolerance_bp=200.0, delta_tolerance_hands=20.0)
    spots = SPOT * np.exp(np.linspace(-0.045, 0.045, 7))       # nothing on a node
    coarse = _pricer(tight, spot_step=0.03, vol_step=0.0, q_step=0.0)
    coarse.price_day(_states(3, spots))
    samples = coarse.sample_visited()
    failing = coarse.verify(samples, tight, SCALE)
    assert failing.mode == "ladder" and not failing.passed and failing.sampled == 6
    assert failing.max_pv_gap_bp > 0.01
    passing = coarse.verify(samples, loose, SCALE)
    assert passing.passed and passing.max_pv_gap_bp == failing.max_pv_gap_bp


def test_a_gap_is_measured_in_bp_of_notional_and_in_hands():
    gate = GateConfig(sample_states=2, pv_tolerance_bp=1e9, delta_tolerance_hands=1e9)
    pricer = _pricer(gate, spot_step=0.03, vol_step=0.0, q_step=0.0)
    states = _states(3, [SPOT * np.exp(0.015)])
    pricer.price_day(states)
    one = pricer.verify(pricer.sample_visited(), gate, GateScale(unit_notional=SPOT, hands_per_unit_delta=1.0))
    ten = pricer.verify(pricer.sample_visited(), gate, GateScale(unit_notional=10 * SPOT, hands_per_unit_delta=10.0))
    assert one.max_pv_gap_bp == pytest.approx(10.0 * ten.max_pv_gap_bp)
    assert ten.max_delta_gap_hands == pytest.approx(10.0 * one.max_delta_gap_hands)


def test_reports_combine_to_the_worst_case():
    a = GateReport(mode="ladder", sampled=2, max_pv_gap_bp=1.0, max_delta_gap_hands=0.5, passed=True)
    b = GateReport(mode="ladder", sampled=3, max_pv_gap_bp=0.2, max_delta_gap_hands=0.9, passed=False)
    c = GateReport.combine([a, b])
    assert (c.sampled, c.max_pv_gap_bp, c.max_delta_gap_hands, c.passed) == (5, 1.0, 0.9, False)


def test_gate_failure_carries_the_report():
    report = GateReport(mode="ladder", sampled=1, max_pv_gap_bp=9.0, max_delta_gap_hands=0.0, passed=False)
    with pytest.raises(GateFailure) as excinfo:
        raise GateFailure(report)
    assert excinfo.value.report is report and "9" in str(excinfo.value)
