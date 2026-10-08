"""KO-reset product inputs for lifecycle tracking: an explicit V1 tenor and the ABSOLUTE contract a knocked-in
REBASED one becomes (spec 2026-10-08 section 5)."""
import pytest

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.v2 import KOResetSnowballQuadEngineV2
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.util.enum.option_enums import ObservationType, PostKOScheduleMode
from quantark.util.exceptions import ValidationError

from .conftest import env, schedule

D = 1 / 365.0


def ko_reset(mode, post, *, pre=(91 * D, 182 * D), ki_times=(91 * D, 182 * D), annualized=False, maturity=None,
             accrued_offset=0.0):
    """Schedule 1 at 91 and 182 days (103), KI on `ki_times` (75), schedule 2 on `post` (95): 15% / 3% coupons,
    5% rebate, no principal. REBASED `post` holds offsets from the knock-in."""
    if maturity is None:
        maturity = max(ki_times) + max(post) if mode == PostKOScheduleMode.REBASED else max(post)
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=maturity, tenor=maturity if annualized else None,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=schedule(list(pre), 103.0), ki_barrier=75.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=schedule(list(ki_times), 75.0)),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=schedule(list(post), 95.0)),
        # V0/V1 read is_annualized_rebate / is_annualized_ki raw (None is not annualized): set them with it
        accrual_config=AccrualConfig(is_annualized=annualized, is_annualized_ki=annualized,
                                     is_annualized_rebate=annualized, accrued_offset=accrued_offset),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False), post_ko_mode=mode)


def test_an_explicit_v1_tenor_is_the_resolved_one_on_absolute():
    product = ko_reset(PostKOScheduleMode.ABSOLUTE, [273 * D, 365 * D], annualized=True)
    tenor = product._resolve_post_contract_tenor(env())
    assert (product.get_maturity_payoff_v1(60.0, env(), contract_tenor=tenor)
            == product.get_maturity_payoff_v1(60.0, env()))


def test_a_rebased_annualized_v1_with_its_tenor_prices():
    product = ko_reset(PostKOScheduleMode.REBASED, [91 * D, 365 * D], annualized=True)
    with pytest.raises(ValidationError, match="ki_time is required"):
        product.get_maturity_payoff_v1(60.0, env())
    converted = product.with_absolute_schedule_2(91 * D)
    assert product.get_maturity_payoff_v1(60.0, env(), contract_tenor=91 * D + 365 * D) == pytest.approx(
        converted.get_maturity_payoff_v1(60.0, env()), rel=1e-15)


def test_the_conversion_places_schedule_two_after_the_knock_in():
    product = ko_reset(PostKOScheduleMode.REBASED, [91 * D, 365 * D])
    converted = product.with_absolute_schedule_2(182 * D)
    times = [r.observation_time for r in converted.post_barrier_config.ko_observation_schedule.records]
    assert times == [182 * D + 91 * D, 182 * D + 365 * D]
    assert converted.post_ko_mode == PostKOScheduleMode.ABSOLUTE
    assert converted.maturity == 182 * D + 365 * D
    assert product.post_ko_mode == PostKOScheduleMode.REBASED          # the original is untouched
    with pytest.raises(ValidationError):
        ko_reset(PostKOScheduleMode.ABSOLUTE, [273 * D]).with_absolute_schedule_2(0.1)


@pytest.mark.parametrize("engine", [
    lambda: KOResetSnowballPDESolver(PDEParams()),
    lambda: KOResetSnowballQuadEngine(params=QuadParams(grid_points=2001)),
    lambda: KOResetSnowballQuadEngineV2(),
    lambda: SnowballMCEngine(params=MCParams(seed=7, num_paths=8192)),
], ids=["pde", "quad_v1", "quad_v2", "mc"])
def test_a_converted_knocked_in_contract_prices_as_the_absolute_contract(engine):
    rebased = ko_reset(PostKOScheduleMode.REBASED, [91 * D, 365 * D]).with_absolute_schedule_2(91 * D)
    direct = ko_reset(PostKOScheduleMode.ABSOLUTE, [182 * D, 456 * D], maturity=456 * D)
    for product in (rebased, direct):
        setattr(product, "_otc_lifecycle_knocked_in", True)
    assert engine().price(rebased, env(90.0)) == pytest.approx(engine().price(direct, env(90.0)), rel=1e-12)


def test_the_state_records_observed_schedule_two_indices():
    from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
    assert AutocallableLifecycleState().observed_post_ko_indices == set()
