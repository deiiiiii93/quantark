"""Contract, market fixture and production adapters of the local-vol QUAD study."""
from __future__ import annotations

import math

import numpy as np
import pytest

from example.case_studies.localvol_transition_quad.contract import Contract, snowball_contract
from example.case_studies.localvol_transition_quad import adapters
from example.case_studies.localvol_transition_quad.market import Market, constant_surface, load_fixture

S0 = 100.0


def _snowball(maturity=2.0, months=24):
    return snowball_contract(S0, maturity, months)


@pytest.mark.parametrize("kwargs", [
    dict(kind="vanilla", maturity=1.0, strike=100.0, s0=S0, ko_times=(1.0,)),
    dict(kind="vanilla", maturity=0.0, strike=100.0, s0=S0),
    dict(kind="vanilla", maturity=1.0, strike=-1.0, s0=S0),
    dict(kind="up_out", maturity=1.0, strike=100.0, s0=S0, ko=120.0, ko_times=(0.5,)),
    dict(kind="up_out", maturity=1.0, strike=100.0, s0=S0, ko=120.0, ko_times=(0.5, 1.0), ki=80.0,
         ki_times=(0.5,)),
    dict(kind="snowball", maturity=1.0, strike=100.0, s0=S0, ko=103.0, ki=104.0, ko_times=(1.0,)),
    dict(kind="snowball", maturity=1.0, strike=100.0, s0=110.0, ko=103.0, ki=85.0, ko_times=(1.0,)),
    dict(kind="snowball", maturity=1.0, strike=100.0, s0=S0, ko=103.0, ki=85.0, ko_times=(0.5, 0.4)),
    dict(kind="snowball", maturity=1.0, strike=100.0, s0=S0, ko=103.0, ki=85.0, ko_times=(1.2,)),
    dict(kind="other", maturity=1.0, strike=100.0, s0=S0),
])
def test_contract_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        Contract(**kwargs)


def test_snowball_schedules_follow_d7():
    c = _snowball(1.0, 12)
    assert c.channels == 2
    assert c.ko_times == tuple(k / 12 for k in range(1, 13))
    assert len(c.ki_times) == 252 and c.ki_times[0] == 1 / 252 and c.ki_times[-1] == 1.0
    assert (c.ko, c.ki, c.strike) == (103.0, 85.0, 100.0)
    c2 = snowball_contract(S0, 0.75, 9)          # 0.75*252 = 189 exactly
    assert c2.ki_times[-1] == 0.75


def test_shifted_keeps_coupon_amounts_and_refuses_crossing():
    c = _snowball(1.0, 12)
    s = c.shifted(1 / 365)
    assert s.maturity == pytest.approx(1 - 1 / 365, abs=0)
    assert s.ko_times[0] == 1 / 12 - 1 / 365
    for t_old, t_new in zip(c.ko_times, s.ko_times):
        assert s.ko_payment(t_new) == pytest.approx(c.ko_payment(t_old), abs=1e-15)
    with pytest.raises(ValueError):
        c.shifted(1 / 252)                       # first KI date would land on valuation


def test_shifted_matches_production_time_shift():
    c = _snowball(2.0, 24)
    market = Market(S0, 0.02, 0.01, constant_surface(0.2))
    product = adapters.production_snowball(c)
    shift = adapters.theta_shift(product, adapters.environment(market))
    assert shift.time_bump == 1 / 365
    s = c.shifted(1 / 365)
    shifted = shift.product
    times = shifted.barrier_config.ko_observation_dates
    np.testing.assert_allclose(times, s.ko_times, rtol=0, atol=1e-14)
    records = shifted.resolve_ko_observations(shift.env)
    assert len(records) == len(s.ko_times)
    for rec, t in zip(records, s.ko_times):
        assert rec.observation_time == pytest.approx(t, abs=1e-14)
        assert rec.payoff / S0 == pytest.approx(s.ko_payment(t), abs=1e-12)


def test_production_snowball_pins_flat_rebate():
    c = _snowball(2.0, 24)
    product = adapters.production_snowball(c)
    assert product.get_maturity_payoff_v0(S0) == pytest.approx(S0 * (1 + 0.15), abs=1e-12)
    cfg = product.barrier_config
    assert cfg.ki_continuous is False
    assert list(cfg.ki_observation_dates) == list(c.ki_times)
    assert list(cfg.ko_observation_dates) == list(c.ko_times)


def test_fixture_market():
    m = load_fixture()
    assert m.r == 0.02 and m.spot == pytest.approx(4993.105)
    assert np.isfinite(m.q)
    assert m.surface.lv_grid.shape == (6, 33)
    up = m.bumped("vega", 1)
    np.testing.assert_allclose(up.surface.lv_grid - m.surface.lv_grid, 0.01, atol=1e-15)
    assert m.bumped("rho", -1).r == pytest.approx(0.02 - 1e-4, abs=0)
    assert m.bumped("carry", 1).q == pytest.approx(m.q + 1e-4, abs=0)


@pytest.mark.slow
def test_pde_full_risk_keys_and_units():
    c = snowball_contract(S0, 0.25, 3)
    market = Market(S0, 0.02, 0.01, constant_surface(0.2))
    from quantark.asset.equity.param import PDEParams
    out = adapters.pde_full_risk(c, market, PDEParams(accuracy="fast"))
    assert set(out["values"]) == {"pv", "delta", "gamma", "vega", "theta", "rho", "carry"}
    v = out["values"]
    assert 0.8 < v["pv"] < 1.1                      # per unit notional
    assert v["vega"] < 0 and math.isfinite(v["theta"])
    assert len(out["timings"]) == 8


@pytest.mark.slow
def test_mc_full_risk_keys():
    c = snowball_contract(S0, 0.25, 3)
    market = Market(S0, 0.02, 0.01, constant_surface(0.2))
    out = adapters.mc_full_risk(c, market, adapters.MCConfig(total_paths=2048, batches=4, substeps=1))
    assert set(out["values"]) == {"pv", "delta", "gamma", "vega", "theta", "rho", "carry"}
    assert set(out["stderr"]) == set(out["values"])
    assert len(out["timings"]) == 10
