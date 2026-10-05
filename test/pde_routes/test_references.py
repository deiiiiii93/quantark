"""The references' own guards (spec 3C §7.2 'references' row; §4.1): conservation, closed forms, refinement,
the corridor reducing to one barrier, refusal of what they do not cover, and the term forms on flat curves."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pde_routes.barrier_reference import barrier_reference, barrier_reference_term, knock_out
from pde_routes.cases import CONST, black_scholes, dcn_case, env_const
from pde_routes.dcn_reference import dcn_exact, dcn_exact_term

VOL, R, Q = CONST["vol"], CONST["rate"], CONST["div"]


@pytest.mark.parametrize("name", ["A", "B", "SA", "SB"])
def test_dcn_recursion_conserves_mass(name):
    product, env = dcn_case(name)
    out = dcn_exact(product, env)
    assert out["conservation"] <= 1e-12            # measured 3.0e-14


def test_dcn_recursion_closed_forms():
    """Two closed-form limits, as $E/d1_check.py computes them (spec §4.1: +3.1e-8 and +3.5e-8 on contract A):
    no knock-in and no knock-out is a strip of digital coupons; knocked in at valuation with no knock-out is a
    forward-settled put on the loss strike."""
    from scipy.stats import norm
    from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
    r, q, vol = 0.0356, 0.1406, 0.184
    mu = r - q - 0.5 * vol * vol

    p1, e1 = dcn_case("A", ki_barrier_ratio=0.05, ko_barrier_ratio=20.0)
    ctx = build_dcn_grid_context(p1)
    strip = 0.0
    for j, c in enumerate(ctx.obs_cols):
        if ctx.obs_is_coupon[j]:
            t = float(ctx.times[c])
            d2 = (np.log(e1.spot / p1.coupon_barrier) + mu * t) / (vol * np.sqrt(t))
            strip += (p1.participation * p1.coupon_rate * ctx.coupon_accruals[j] * p1.notional
                      * np.exp(-r * ctx.coupon_pay_times[j]) * norm.cdf(d2))
    assert abs(float(dcn_exact(p1, e1)["pv"][0]) - strip) <= 1e-6

    p2, e2 = dcn_case("A", knocked_in_at_valuation=True, ko_barrier_ratio=20.0)
    ctx = build_dcn_grid_context(p2)
    T = float(ctx.times[-1])
    fwd, k = e2.spot * np.exp((r - q) * T), p2.k_loss
    d1 = (np.log(fwd / k) + 0.5 * vol * vol * T) / (vol * np.sqrt(T))
    put_fwd = k * norm.cdf(-(d1 - vol * np.sqrt(T))) - fwd * norm.cdf(-d1)
    put = -(p2.notional / p2.initial_price) * p2.participation * put_fwd * np.exp(-r * ctx.loss_pay_time)
    assert abs(float(dcn_exact(p2, e2)["pv"][0]) - put) <= 1e-6


def test_barrier_recursion_conserves_and_matches_expiry_closed_form():
    obs = [1.0]
    one = knock_out(payoff="one", strike=100.0, lower=None, upper=1e9, cash=0.0, at_hit=False, obs=obs,
                    spots=[100.0], vol=VOL, r=R, q=Q)[0][100.0][0]
    assert abs(one - math.exp(-R)) <= 1e-12        # conservation: measured 5.0e-14
    call = knock_out(payoff="call", strike=100.0, lower=None, upper=1e9, cash=0.0, at_hit=False, obs=obs,
                     spots=[100.0], vol=VOL, r=R, q=Q)[0][100.0][0]
    assert abs(call - black_scholes(True, 100.0, 100.0, 1.0)[0]) <= 1e-12     # measured 4.4e-15


def test_double_corridor_reduces_to_single_barrier():
    obs = [k / 12.0 for k in range(1, 13)]
    single = knock_out(payoff="call", strike=100.0, lower=None, upper=130.0, cash=5.0, at_hit=False, obs=obs,
                       spots=[100.0], vol=VOL, r=R, q=Q)[0][100.0][0]
    double = knock_out(payoff="call", strike=100.0, lower=1e-3, upper=130.0, cash=5.0, at_hit=False, obs=obs,
                       spots=[100.0], vol=VOL, r=R, q=Q)[0][100.0][0]
    assert abs(single - double) <= 1e-12


def test_term_forms_reproduce_flat_forms():
    obs = [k / 12.0 for k in range(1, 13)]
    df = lambda t: math.exp(-R * t)                                    # noqa: E731  (l_ref_term: df(t))
    growth = lambda t: math.exp((R - Q) * t)                           # noqa: E731  (F(t) = S0 growth(t))
    flat = barrier_reference(kind="UO", call=True, strike=100.0, level=130.0, rebate=5.0, at_hit=False, obs=obs,
                             spots=[100.0], vol=VOL, r=R, q=Q)[100.0][0]
    term = barrier_reference_term(kind="UO", call=True, strike=100.0, level=130.0, rebate=5.0, at_hit=False,
                                  obs=obs, spots=[100.0], vol=VOL, df=df, growth=growth)[100.0]
    assert abs(term - flat) <= 1e-12 * abs(flat)                       # measured 2.2e-15
    product, env = dcn_case("A")
    a, b = dcn_exact(product, env)["pv"][0], dcn_exact_term(product, env)["pv"][0]
    assert abs(a - b) <= 1e-12 * abs(a)                                # measured 2.7e-9 on -93,301.72 (2.9e-14 rel)


def test_references_refuse_what_they_do_not_cover():
    with pytest.raises(AssertionError):
        knock_out(payoff="call", strike=100.0, lower=None, upper=130.0, cash=0.0, at_hit=False, obs=[0.5, 0.5],
                  spots=[100.0], vol=VOL, r=R, q=Q)                      # non-increasing dates


def test_banked_no_touch_reference_is_complete():
    bank = json.loads((Path(__file__).parent / "no_touch_reference.json").read_text())["cells"]
    sizes = {k: len(v) for k, v in bank.items()}
    assert sizes == {"z1": 637, "equity": 108, "stress": 4536, "random_reachable": 2666, "random_any": 2663,
                     "reviewer": 2, "engine": 3}
