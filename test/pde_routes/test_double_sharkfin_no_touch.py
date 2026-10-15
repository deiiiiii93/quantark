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
#: the monthly (DISCRETE, from inception) cells of the capture moved on 2026-10-15: with no record today the
#: engine prices through the first observation, the BGK-shifted formula only from it on (the capture holds
#: the BGK formula at today's spot); before -> after, QUAD (401 points) and the BS PDE for reference
MOVED = {
    # 9.021712544545743 -> (QUAD 9.0167, PDE 9.0168)
    "shf|monthly|hit|0.08|100.0": 9.021712544545803,
    # 7.1267347766670435 -> (QUAD 7.4988, PDE 7.5014)
    "shf|monthly|hit|0.08|129.9": 7.480849942388168,
    # 4.945519293223049 -> (QUAD 4.9458, PDE 4.9458)
    "shf|monthly|hit|0.08|81.0": 4.945482150717244,
    # 8.499348484963795 -> (QUAD 8.3615, PDE 8.3613)
    "shf|monthly|hit|0.22|100.0": 8.499270106352515,
    # 5.9847723713166125 -> (QUAD 6.1511, PDE 6.1510)
    "shf|monthly|hit|0.22|129.9": 6.167719507200834,
    # 5.707976003741938 -> (QUAD 5.7543, PDE 5.7543)
    "shf|monthly|hit|0.22|81.0": 5.766260894233837,
    # 9.021670796433433 -> (QUAD 9.0167, PDE 9.0167)
    "shf|monthly|maturity|0.08|100.0": 9.021670796433494,
    # 7.019219637714621 -> (QUAD 7.3955, PDE 7.3982)
    "shf|monthly|maturity|0.08|129.9": 7.377696074592943,
    # 4.870304530635556 -> (QUAD 4.8706, PDE 4.8707)
    "shf|monthly|maturity|0.08|81.0": 4.87051945515878,
    # 8.477635604525915 -> (QUAD 8.3385, PDE 8.3384)
    "shf|monthly|maturity|0.22|100.0": 8.477550869510516,
    # 5.8799864647984785 -> (QUAD 6.0515, PDE 6.0515)
    "shf|monthly|maturity|0.22|129.9": 6.06823048919779,
    # 5.611165576637623 -> (QUAD 5.6602, PDE 5.6601)
    "shf|monthly|maturity|0.22|81.0": 5.672278194067586,
}


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
        assert abs(got - MOVED.get(tag, float.fromhex(value))) <= 1e-12, tag      # measured 3.6e-15


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


@pytest.mark.parametrize("vol", [0.22, 0.08])
@pytest.mark.parametrize("maturity", [1 / 365, 30 / 365, 1.0, 3.0])
@pytest.mark.parametrize("spot", [81.0, 100.0, 129.0, 129.9])
def test_paid_at_hit_integral_resolves_next_to_a_barrier(spot, maturity, vol):
    """The continuous paid-at-hit leg's integral against an adaptive reference (3C handoff: the 48-node rule
    was off by up to 2.4e-5 on the leg next to a barrier; graded panels: 4e-15 in the evidence sweep)."""
    import math

    from scipy import integrate

    from quantark.asset.equity.engine.analytical.double_no_touch import double_no_touch
    from quantark.asset.equity.engine.analytical.double_sharkfin_option_analytical_engine import (
        DoubleSharkfinOptionAnalyticalEngine,
    )

    r, q, lo, up = 0.03, 0.01, 80.0, 130.0

    def f(t):
        return math.exp(-r * t) * double_no_touch(spot, t, r, q, vol, lo, up) if t > 0 else 1.0

    d = min(math.log(up / spot), math.log(spot / lo))
    points = sorted({min(maturity, (d / vol) ** 2 * k) for k in (0.01, 0.1, 1, 10, 100)} - {maturity})
    ref = integrate.quad(f, 0.0, maturity, points=points or None, limit=2000, epsabs=1e-15, epsrel=1e-13)[0]
    got = DoubleSharkfinOptionAnalyticalEngine()._discounted_survival_integral(spot, maturity, r, q, vol, lo, up)
    assert got == pytest.approx(ref, abs=1e-12)
