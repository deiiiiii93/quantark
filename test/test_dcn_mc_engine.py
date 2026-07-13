"""DCN Monte Carlo engine tests (spec WP1.3)."""
import pytest

from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.product.option.dcn_option import DCNDirection

from dcn_fixtures import DCN_A, DCN_B, FLAT, flat_env, make_dcn

PATHS = 2 ** 15  # test-speed paths; validation gate uses 2**17


def _engine(**kw):
    return DCNMCEngine(num_paths=PATHS, seed=42, **kw)


def test_deterministic_rerun_bit_reproduces():
    p = make_dcn(DCN_A)
    env = flat_env(**FLAT)
    r1 = _engine().price_detailed(p, env)
    r2 = _engine().price_detailed(p, env)
    assert r1.pv == r2.pv and r1.std_error == r2.std_error


def test_leg_invariant_and_sign_exact():
    p = make_dcn(DCN_A)
    env = flat_env(**FLAT)
    r = _engine().price_detailed(p, env)
    # exact by construction: pv is DEFINED as the sum of the signed legs
    assert r.pv == r.pv_fixed_coupons + r.pv_ko_coupons + r.pv_loss_leg
    assert sum(r.pv_ko_coupons_by_period) == pytest.approx(r.pv_ko_coupons)
    s = _engine().price_detailed(make_dcn(DCN_A, direction=DCNDirection.SELLER), env)
    assert s.pv == -r.pv                       # IEEE negation is exact
    assert s.pv_loss_leg == -r.pv_loss_leg


def test_probabilities_consistent():
    p = make_dcn(DCN_B)
    env = flat_env(**FLAT)
    r = _engine().price_detailed(p, env)
    assert 0.0 <= r.ki_probability <= 1.0
    assert sum(r.ko_timing_distribution) == pytest.approx(r.ko_probability)
    assert r.prob_survive_no_ki + r.prob_survive_ki + r.ko_probability == \
        pytest.approx(1.0)
    assert len(r.coupon_probability) == 34
    assert 0.0 < r.expected_life_years <= 3.05


def test_high_q_makes_ki_dominant():
    # q = 14.06% >> r drags the forward down hard: KI before termination is
    # substantial, almost no path survives to maturity without knocking in,
    # and the buyer PV is negative (loss leg dominates the coupons).
    # ki_probability counts KI on ALIVE paths only (contract terminates at
    # KO), so it is bounded by 1 - early-KO mass.
    p = make_dcn(DCN_B)
    r = _engine().price_detailed(p, flat_env(**FLAT))
    assert r.ki_probability > 0.3
    assert r.prob_survive_ki > 0.3           # KI'd survivors carry the loss
    assert r.prob_survive_no_ki < 0.05       # surviving without KI is rare
    assert r.pv < 0.0
    assert r.pv_loss_leg < 0.0


def test_knocked_in_seed_removes_all_coupons():
    p = make_dcn(DCN_A, knocked_in_at_valuation=True)
    r = _engine().price_detailed(p, flat_env(**FLAT))
    assert r.pv_fixed_coupons == 0.0
    assert r.pv_loss_leg < 0.0


def test_to_dict_json_safe():
    import json
    r = _engine().price_detailed(make_dcn(DCN_A), flat_env(**FLAT))
    json.dumps(r.to_dict())  # must not raise
