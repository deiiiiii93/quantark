import numpy as np
import pytest

from quantark.volmodels.barrier import (
    BarrierSpec, mc_barrier_cashflows, discrete_survival, bridge_survival, validate_barrier,
)
from quantark.util.exceptions import ValidationError


def _disc(t):
    return np.exp(-0.02 * np.asarray(t, dtype=float))


def test_up_out_call_knocks_out_with_rebate():
    spec = BarrierSpec(is_up=True, is_out=True, is_call=True, barrier=120., strike=100., rebate=5., pay_at_hit=False)
    term = np.array([130., 110.])
    w = np.array([0.0, 1.0])  # path0 breached, path1 survived
    pv = mc_barrier_cashflows(term, w, np.array([0.5, 0.0]), spec, _disc, maturity=1.0)
    assert pv[0] == pytest.approx(5 * np.exp(-0.02 * 1.0), rel=1e-9)   # KO -> rebate at T
    assert pv[1] == pytest.approx(10 * np.exp(-0.02 * 1.0), rel=1e-9)  # survived -> call payoff at T


def test_pay_at_hit_discounts_to_hit_time():
    spec = BarrierSpec(True, True, True, 120., 100., 5., pay_at_hit=True)
    pv = mc_barrier_cashflows(np.array([130.]), np.array([0.0]), np.array([0.5]), spec, _disc, maturity=1.0)
    assert pv[0] == pytest.approx(5 * np.exp(-0.02 * 0.5), rel=1e-9)


def test_knock_in_pays_option_only_when_breached():
    spec = BarrierSpec(is_up=False, is_out=False, is_call=False, barrier=90., strike=100., rebate=0., pay_at_hit=False)
    # put; path0 breached down -> knocks in, pays (100-95)=5; path1 survived -> 0 (rebate 0)
    pv = mc_barrier_cashflows(np.array([95., 105.]), np.array([0.0, 1.0]), np.array([0.3, 0.0]), spec, _disc, maturity=1.0)
    assert pv[0] == pytest.approx(5 * np.exp(-0.02), rel=1e-9)
    assert pv[1] == pytest.approx(0.0, abs=1e-12)


def test_discrete_survival_hard_breach():
    spec = BarrierSpec(True, True, True, 120., 100., 0., False)
    samples = np.array([[105., 125., 110.], [105., 110., 115.]])  # path0 breaches at col1
    w, first = discrete_survival(samples, spec)
    assert w[0] == 0.0 and first[0] == 1
    assert w[1] == 1.0 and first[1] == 3


def test_bridge_survival_between_step_crossing():
    # both nodes below barrier but bridge assigns a positive crossing prob (survival < 1)
    spec = BarrierSpec(True, True, True, 120., 100., 0., False)
    nodes = np.array([[100., 118.]])          # ends just below 120
    vol = np.array([[0.4]])
    w, first = bridge_survival(nodes, vol, np.array([1.0]), spec)
    assert 0.0 < w[0] < 1.0                    # captures the between-step crossing risk
    assert first[0] == 1                       # no hard node breach
    # a node that ends beyond the barrier -> survival exactly 0
    nodes2 = np.array([[100., 130.]])
    w2, first2 = bridge_survival(nodes2, vol, np.array([1.0]), spec)
    assert w2[0] == 0.0 and first2[0] == 0


def test_validate_rejects_barrier_at_spot_and_negative():
    with pytest.raises(ValidationError):
        validate_barrier(BarrierSpec(True, True, True, 100., 100., 0., False), s0=100.)
    with pytest.raises(ValidationError):
        validate_barrier(BarrierSpec(True, True, True, -1., 100., 0., False), s0=100.)
