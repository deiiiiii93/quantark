"""The legacy autocallable QUAD engines price a contract whose knock-in or knock-out side has ended (spec 2026-10-07,
plan Task 6): theta ages a contract past its last knock-in (or knock-out) date and leaves that side empty, which they
refused. No knock-in side means no knock-in monitoring; no knock-out side means no autocall before maturity. Checked
against the PDE solver within twice the two engines' gap on the same contract still alive."""
import pytest

from test_autocallable_observation_at_valuation import (COUPON_LEVEL, KI, KO, POST_KO, AccrualConfig, BarrierConfig,
                                                        CouponBarrierConfig, KnockOutResetSnowballOption,
                                                        ObservationType, PayoffConfig, PhoenixOption, SnowballOption,
                                                        _schedule, carrying_a_knock_in, env, price)

SPOT, AGE = 95.0, 0.3
NO_KI = ((0.5, 0.75, 1.0), (0.1, 0.2))           # (knock-out times, knock-in times): the knock-in side ends first
NO_KO = ((0.1, 0.2), (0.5, 0.75, 1.0))


def _barrier(ko_times, ki_times, ko_rate):
    return BarrierConfig(ko_barrier=KO, ko_rate=ko_rate, ko_observation_type=ObservationType.DISCRETE,
                         ko_observation_schedule=_schedule(KO, ko_times), ki_barrier=KI,
                         ki_observation_type=ObservationType.DISCRETE, ki_observation_schedule=_schedule(KI, ki_times))


def make(family, ko_times, ki_times, post_times=None):
    if family == "snowball":
        return SnowballOption(initial_price=100.0, strike=100.0, maturity=1.0,
                              barrier_config=_barrier(ko_times, ki_times, 0.15),
                              accrual_config=AccrualConfig(is_annualized=False),
                              payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False))
    if family == "phoenix":
        return PhoenixOption(initial_price=100.0, strike=100.0, maturity=1.0,
                             barrier_config=_barrier(ko_times, ki_times, 0.0),
                             coupon_config=CouponBarrierConfig(coupon_barrier=COUPON_LEVEL, coupon_rate=0.02,
                                                               memory_coupon=False, fixed_coupon_year_fraction=0.5),
                             accrual_config=AccrualConfig(is_annualized=False, is_annualized_coupon=True),
                             payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True))
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0, barrier_config=_barrier(ko_times, ki_times, 0.15),
        post_barrier_config=BarrierConfig(ko_barrier=POST_KO, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=_schedule(POST_KO, post_times or ko_times)),
        accrual_config=AccrualConfig(is_annualized=False),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False))


CASES = [("snowball", NO_KI, False), ("snowball", NO_KO, False), ("phoenix", NO_KI, False),
         ("phoenix", NO_KO, False), ("ko_reset", NO_KI, False),
         # a KO-reset whose pre-knock-in schedule ended un-knocked has matured; knocked in, it runs on schedule 2
         ("ko_reset", NO_KO, True)]


def test_legacy_quad_prices_a_ko_reset_whose_post_knock_in_schedule_has_ended():
    """Schedule 2 ended first: a knock-in now replaces schedule 1 by nothing, so a knocked-in contract runs to maturity."""
    times = ((0.5, 0.75, 1.0), (0.5, 0.75, 1.0), (0.1, 0.2))

    gap = abs(price("ko_reset", "pde", make("ko_reset", *times), SPOT)
              - price("ko_reset", "quad_v1", make("ko_reset", *times), SPOT))
    aged = make("ko_reset", *times)
    aged.time_shift(AGE, None, env(SPOT), keep_landing=True)
    assert not aged.post_barrier_config.ko_observation_schedule.records
    assert price("ko_reset", "quad_v1", aged, SPOT) == pytest.approx(price("ko_reset", "pde", aged, SPOT), abs=2.0 * gap)


@pytest.mark.parametrize("family, times, knocked_in", CASES,
                         ids=[f"{f}-{'no_ki' if t is NO_KI else 'no_ko'}{'-knocked_in' if k else ''}" for f, t, k in CASES])
def test_legacy_quad_prices_an_ended_side_as_the_pde_does(family, times, knocked_in):
    def build():
        product = make(family, *times)
        return carrying_a_knock_in(product) if knocked_in else product

    gap = abs(price(family, "pde", build(), SPOT) - price(family, "quad_v1", build(), SPOT))
    aged = build()
    aged.time_shift(AGE, None, env(SPOT), keep_landing=True)
    side = aged.barrier_config.ki_observation_schedule if times is NO_KI else aged.barrier_config.ko_observation_schedule
    assert not side.records
    assert price(family, "quad_v1", aged, SPOT) == pytest.approx(price(family, "pde", aged, SPOT), abs=2.0 * gap)
