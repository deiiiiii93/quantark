import pytest

from quantark.param.vol.vannavolga import FXEnv, SmileQuotes, DeltaConvention
from quantark.asset.fx.engine.analytical.vannavolga.vv_vanilla_barrier import (
    price_vv_barrier,
)
from quantark.asset.fx.engine.analytical.vannavolga.barrier_bs import (
    reiner_rubinstein_barrier,
)

ENV = FXEnv(spot=1.20, rd=0.02, rf=0.01, tau=0.75)
SMILE = SmileQuotes(sigma_atm=0.10, rr25=-0.01, bf25_2vol=0.003)
FLAT = SmileQuotes(sigma_atm=0.10, rr25=0.0, bf25_2vol=0.0)


def test_breached_knock_in_returns_full_vv_vanilla():
    # Spot already above an up-barrier: the KI is a plain vanilla, so the price
    # must equal the FULL VV vanilla (including the vega term), not the
    # vanna/volga-only barrier correction.
    env = FXEnv(spot=1.40, rd=0.02, rf=0.01, tau=0.75)
    res = price_vv_barrier(
        env, SMILE, strike=1.20, barrier=1.35, is_up=True,
        is_call=True, knock_in=True, conv=DeltaConvention.SPOT,
    )
    assert res.vv == pytest.approx(res.vanilla)
    # Diagnostics must match a live vanilla: non-zero vega/vanna/volga.
    assert res.greeks["vega"] > 0.0
    assert res.greeks["vanna"] != 0.0
    assert res.greeks["volga"] != 0.0


def test_vv_reduces_to_bs_when_smile_flat():
    res = price_vv_barrier(
        ENV, FLAT, strike=1.20, barrier=1.35, is_up=True,
        is_call=True, knock_in=False, conv=DeltaConvention.SPOT,
    )
    bs = reiner_rubinstein_barrier(
        ENV.spot, 1.20, 1.35, FLAT.sigma_atm, ENV.tau, ENV.rd, ENV.rf,
        is_up=True, is_call=True, knock_in=False,
    )
    assert res.vv == pytest.approx(bs, abs=1e-9)


def test_far_ko_converges_to_vv_vanilla():
    # Canonical CM correction (incl. survival-weighted vega) must make a far
    # up-and-out call converge to the smile-consistent VV vanilla as H -> inf.
    res_far = price_vv_barrier(
        ENV, SMILE, strike=1.20, barrier=10.0, is_up=True,
        is_call=True, knock_in=False, conv=DeltaConvention.SPOT,
    )
    assert res_far.vv == pytest.approx(res_far.vanilla, rel=1e-6)


def test_near_ko_vega_term_suppressed_by_survival():
    # Near the barrier, survival is low so the (now-present) vega term is
    # attenuated: the KO stays well below the vanilla, not pinned to it.
    res_near = price_vv_barrier(
        ENV, SMILE, strike=1.20, barrier=1.35, is_up=True,
        is_call=True, knock_in=False, conv=DeltaConvention.SPOT,
    )
    assert 0.0 < res_near.vv < 0.6 * res_near.vanilla


def test_vv_ko_bounded_by_vanilla_and_nonneg():
    res = price_vv_barrier(
        ENV, SMILE, strike=1.20, barrier=1.35, is_up=True,
        is_call=True, knock_in=False, conv=DeltaConvention.SPOT,
    )
    assert 0.0 <= res.vv <= res.vanilla + 1e-12
    assert res.bstv == pytest.approx(
        reiner_rubinstein_barrier(
            ENV.spot, 1.20, 1.35, SMILE.sigma_atm, ENV.tau, ENV.rd, ENV.rf,
            is_up=True, is_call=True, knock_in=False,
        )
    )
