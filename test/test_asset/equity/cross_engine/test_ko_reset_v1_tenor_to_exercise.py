"""Annualized knocked-in amounts of a KO-reset accrue to the PRODUCT'S EXERCISE DATE (owner ruling O2, 2026-11-27).

Owner ruling "Accrue to exercise" (run 9, replaces D4): the knocked-in (V1) tenor of an annualized-knock-in KO-reset
runs from the contract's accrual origin to the exercise date, on every engine and the tracker -- it was origin to
schedule 2's last contractual fixing (ABSOLUTE), and the knock-in anchor plus schedule 2's last offset (REBASED). The
product's ``post`` terminal period carries it (``_resolve_post_contract_tenor``); every engine reads that one place.
Bitwise where the exercise date IS schedule 2's last fixing.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from test_asset.equity.cross_engine.test_ko_reset_knocked_in_end import (
    ENGINES, PUT_TOLERANCE, SCHEDULE_2_END, _env_for, _referee, carried_env, make,
)
from test_asset.equity.engine_pde.test_same_instant_ko import env

from quantark.asset.equity.engine.mc import SnowballMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.util.enum.option_enums import ObservationType, PostKOScheduleMode


def _schedule(times, level):
    return ObservationSchedule(records=[ObservationRecord(observation_time=float(t), barrier=level) for t in times])


def numeric_ko_reset(mode, maturity):
    """Numeric: schedule 1 and the knock-in at 0.25 / 0.5 (103 / 75), schedule 2 at 0.75 / 1.0 (ABSOLUTE) or offsets
    0.25 / 0.5 from the knock-in (REBASED: the latest end 1.0); an annualized knocked-in leg."""
    post = (0.75, 1.0) if mode is PostKOScheduleMode.ABSOLUTE else (0.25, 0.5)
    product = KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=maturity,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_schedule((0.25, 0.5), 103.0), ki_barrier=75.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=_schedule((0.25, 0.5), 75.0)),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=_schedule(post, 95.0)),
        accrual_config=AccrualConfig(is_annualized=False, is_annualized_ki=True),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False), post_ko_mode=mode)
    product.validate()
    return product


def annualized(product):
    """The same contract with an annualized knocked-in leg (and nothing else annualized)."""
    product.accrual_config = replace(product.accrual_config, is_annualized_ki=True)
    product.validate()
    return product


def test_a_dated_absolute_v1_tenor_runs_to_the_exercise_date():
    """``ko_reset()``: schedule 2 ends 2025-12-03, exercise 2026-01-05. The V1 tenor is the contract tenor to the
    exercise date (origin 2025-01-03), longer than the tenor to 12-03 by the 33 days between them."""
    product = annualized(make())
    tenor = product._resolve_post_contract_tenor(env())
    assert tenor == product.get_contract_tenor()
    on_end = annualized(make(exercise=SCHEDULE_2_END))._resolve_post_contract_tenor(env())
    assert tenor > on_end


def test_a_dated_absolute_v1_tenor_ending_with_schedule_two_is_unchanged():
    """Exercise on schedule 2's last fixing: origin to that fixing, as before (the two ends are one date)."""
    product = annualized(make(exercise=SCHEDULE_2_END))
    assert product._resolve_post_contract_tenor(env()) == product.get_contract_tenor()


@pytest.mark.parametrize("mode", [PostKOScheduleMode.ABSOLUTE, PostKOScheduleMode.REBASED])
def test_a_numeric_v1_tenor_runs_to_the_maturity(mode):
    """Numeric: origin to the maturity (1.5) whatever schedule 2's end; REBASED needs no knock-in anchor (the tenor no
    longer hangs from it) and ignores one when given."""
    product = numeric_ko_reset(mode, maturity=1.5)
    assert product._resolve_post_contract_tenor(env()) == 1.5
    if mode is PostKOScheduleMode.REBASED:
        from quantark.asset.equity.product.option.accrual import knock_in_anchor_for

        record = product.barrier_config.ki_observation_schedule.records[0]
        assert product._resolve_post_contract_tenor(env(), knock_in=knock_in_anchor_for(record, 0)) == 1.5


def test_a_numeric_absolute_v1_tenor_ending_with_schedule_two_is_bitwise():
    """Maturity == schedule 2's last fixing (1.0): the tenor is that float, bitwise."""
    product = numeric_ko_reset(PostKOScheduleMode.ABSOLUTE, maturity=1.0)
    assert product._resolve_post_contract_tenor(env()) == 1.0


@pytest.mark.parametrize("engine", list(ENGINES))
def test_the_annualized_referee_is_the_tenor_times_the_put(engine):
    """The referee (carried knock-in + ``disable_ko_after_ki``) with an annualized knocked-in leg: V1 = tenor x min(S -
    K, 0) on the spot at the exercise date, the tenor to the exercise date -- minus the tenor times the put on every
    grid engine to the referee tolerance."""
    product = annualized(make(disable=True, carried=True))
    product._otc_lifecycle_knocked_in = True
    e = carried_env()
    tenor = product._resolve_post_contract_tenor(e)
    exact = tenor * _referee(product, e)
    value = ENGINES[engine][0]().price(product, _env_for(engine, carried_env))
    assert abs(value - exact) <= PUT_TOLERANCE[engine] * abs(exact), (value, exact, (value - exact) / exact)


def test_the_annualized_referee_on_mc():
    product = annualized(make(disable=True, carried=True))
    product._otc_lifecycle_knocked_in = True
    e = carried_env()
    exact = product._resolve_post_contract_tenor(e) * _referee(product, e)
    engine = SnowballMCEngine(params=MCParams(num_paths=1_000_000, seed=7))
    value = engine.price(product, e)
    se = engine.get_last_std_error()
    assert abs(value - exact) <= 3.0 * se, (value, exact, se, (value - exact) / se)
