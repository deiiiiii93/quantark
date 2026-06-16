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
