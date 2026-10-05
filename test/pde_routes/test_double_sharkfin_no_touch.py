"""DoubleSharkfinOptionAnalyticalEngine on double_no_touch (spec 3C §5.8 ¶, §7.2 'double sharkfin' row)."""
import json
import math
from pathlib import Path

import pytest
from scipy import integrate

from pde_routes.cases import env_const
from quantark.asset.equity.engine.analytical.double_no_touch import double_no_touch
from quantark.asset.equity.engine.analytical.double_sharkfin_option_analytical_engine import (
    DoubleSharkfinOptionAnalyticalEngine)
from quantark.asset.equity.product.option.double_sharkfin_option import DoubleSharkfinOption
from quantark.util.enum import ObservationType, OptionType

HERE = Path(__file__).parent
CAPTURE = json.loads((HERE / "analytic_capture.json").read_text())


def test_captured_legs_do_not_move():
    for tag, value in CAPTURE.items():
        if not tag.startswith("shf|"):
            continue
        _, obs, at_hit, vol, spot = tag.split("|")
        kw = dict(observation_type=ObservationType.CONTINUOUS)
        if obs == "monthly":
            from pde_routes.cases import obs_dates
            kw = dict(observation_type=ObservationType.DISCRETE, observation_dates=obs_dates("monthly"))
        p = DoubleSharkfinOption(strike=100.0, option_type=OptionType.CALL, upper_barrier=130.0, lower_barrier=80.0,
                                 maturity=1.0, knock_out_rebate=5.0, no_hit_rebate=5.0, pay_at_hit=at_hit == "hit",
                                 **kw)
        got = float(DoubleSharkfinOptionAnalyticalEngine().price(p, env_const(float(spot), vol=float(vol))))
        assert abs(got - float.fromhex(value)) <= 1e-12, tag                       # measured 3.6e-15


def _converged_hit_factor(spot, T, r, q, vol):
    """E[exp(-r tau); tau <= T] = 1 - exp(-r T) P(T) - r int_0^T exp(-r t) P(t) dt."""
    f = lambda t: math.exp(-r * t) * double_no_touch(spot, t, r, q, vol, 80.0, 130.0)     # noqa: E731
    integral, err = integrate.quad(f, 0.0, T, epsabs=1e-14, epsrel=1e-13, limit=500)
    assert err <= 1e-12
    return 1.0 - math.exp(-r * T) * double_no_touch(spot, T, r, q, vol, 80.0, 130.0) - r * integral


# 127 bends like 129.9: the engine's 48-node quadrature leaves 6.23e-8 there (evidence z6; spec 3C §7.2 corrected
# on the user's ruling of 2026-10-05; 1.5e-9 at 96 nodes, round-off at 400)
@pytest.mark.parametrize("spot,bound", [(85.0, 1e-8), (100.0, 1e-8), (127.0, 1e-7), (129.9, 5e-6)])
def test_continuous_paid_at_hit_factor(spot, bound):
    r, q, vol, T = 0.03, 0.01, 0.22, 1.0
    engine = DoubleSharkfinOptionAnalyticalEngine()
    surv = double_no_touch(spot, T, r, q, vol, 80.0, 130.0)
    got = engine._continuous_hit_discount_factor(spot=spot, maturity=T, rate=r, div=q, vol=vol, lower_barrier=80.0,
                                                 upper_barrier=130.0, survival_at_maturity=surv)
    assert abs(got - _converged_hit_factor(spot, T, r, q, vol)) <= bound     # measured 1.2e-9 / 6.2e-8 / 1.2e-6


def test_survival_at_the_review_cell():
    bank = json.loads((HERE / "no_touch_reference.json").read_text())["cells"]["reviewer"]
    S, T, r, q, vol, L, U = next(c["args"] for c in bank if c["args"][4] == 0.01)
    ref = next(c["reference"] for c in bank if c["args"][4] == 0.01)
    p = DoubleSharkfinOption(strike=100.0, option_type=OptionType.CALL, upper_barrier=U, lower_barrier=L,
                             maturity=T, knock_out_rebate=5.0, observation_type=ObservationType.CONTINUOUS)
    got = DoubleSharkfinOptionAnalyticalEngine()._survival_probability(p, T, r, q, vol, env_const(S, vol=vol, rate=r,
                                                                                                    div=q))
    assert abs(got - ref) <= 1e-12                                  # today 0.248 for 0.858


def test_max_terms_is_gone():
    with pytest.raises(TypeError):
        DoubleSharkfinOptionAnalyticalEngine(max_terms=80)
    with pytest.raises(TypeError):
        DoubleSharkfinOptionAnalyticalEngine(None, 48)
    assert not hasattr(DoubleSharkfinOptionAnalyticalEngine, "DEFAULT_MAX_TERMS")
    assert "max_terms" not in DoubleSharkfinOptionAnalyticalEngine.MODEL_FINGERPRINT_ATTRS
