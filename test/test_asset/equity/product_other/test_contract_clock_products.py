"""Contract clock wired into the autocallables (T3): construction, theta aging, the lifecycle day-snap and the
KO-reset REBASED schedule 2 all read one float per instant (spec .ship/2026-11-23-contract-clock AC2, AC2t, AC5, AC8).
"""
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.quad import KOResetSnowballQuadEngine, KOResetSnowballQuadEngineV2
from quantark.asset.equity.lifecycle.snapping import snap_contract
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.accrual import knock_in_anchor_for
from quantark.asset.equity.product.option.contract_clock import canonicalize_contract_clock, contract_instants
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
    at_valuation_instant,
)
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, PostKOScheduleMode
from quantark.util.exceptions import ValidationError

UP = lambda t: float(np.nextafter(t, np.inf))      # noqa: E731
DOWN = lambda t: float(np.nextafter(t, -np.inf))   # noqa: E731


def env():
    return PricingEnvironment(
        spot_quote=SpotQuote(100.0), vol_surface=FlatVolSurface(0.2), rate_curve=FlatRateCurve(0.03),
        div_yield=ContinuousDividendYield(0.01), valuation_date=datetime(2026, 9, 11),
    )


def barrier(ko_times, ki_times, **kw):
    return BarrierConfig(
        ko_barrier=103.0, ko_rate=0.12, ko_observation_dates=list(ko_times),
        ki_barrier=75.0, ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=list(ki_times), **kw,
    )


def snowball(ko_times=(0.25, 0.5, 0.75, 1.0), ki_times=(0.25, 0.5, 0.75, 1.0), maturity=None):
    return SnowballOption(initial_price=100.0, strike=100.0, contract_multiplier=1.0,
                          maturity=maturity or ko_times[-1], barrier_config=barrier(ko_times, ki_times))


def times(schedule):
    return [r.observation_time for r in schedule.records]


class TestConstruction:
    def test_snowball_construction_is_canonical(self):
        p = snowball(ki_times=(0.25, UP(0.5), 0.75, UP(1.0)))
        assert times(p.barrier_config.ko_observation_schedule)[1] == times(p.barrier_config.ki_observation_schedule)[1]
        assert times(p.barrier_config.ki_observation_schedule)[-1] == 1.0 == p.maturity
        assert canonicalize_contract_clock(p) is False

    def test_phoenix_construction_is_canonical(self):
        p = PhoenixOption(initial_price=100.0, strike=100.0, contract_multiplier=1.0, maturity=1.0,
                          barrier_config=barrier((0.25, 0.5, 0.75, 1.0), (0.25, DOWN(0.5), 0.75, 1.0)),
                          coupon_config=CouponBarrierConfig(coupon_barrier=80.0))
        assert times(p.barrier_config.ki_observation_schedule)[1] == 0.5

    def test_same_schedule_collision_refused_at_construction(self):
        with pytest.raises(ValidationError, match="observe one instant twice"):
            snowball(ko_times=(0.25, 0.5, UP(0.5), 1.0))

    def test_double_rounded_ko_reset_is_canonical_and_prices_on_v2_default(self):
        """AC5: the booking the lobe re-measure refused (schedule 1 at (i+1)/12*1.0, schedule 2 at (i+1)/18*1.5)."""
        pre = [(i + 1) / 12 * 1.0 for i in range(12)]
        post = [(i + 1) / 18 * 1.5 for i in range(18)]
        assert pre[10] != post[10]
        ki = [(i + 1) / 252 * 1.0 for i in range(252)]
        p = KnockOutResetSnowballOption(
            initial_price=100.0, strike=100.0, contract_multiplier=1.0, maturity=1.5,
            barrier_config=barrier(pre, ki), post_ko_mode=PostKOScheduleMode.ABSOLUTE,
            post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.12, ko_observation_dates=post),
        )
        assert times(p.barrier_config.ko_observation_schedule)[10] == times(p.post_barrier_config.ko_observation_schedule)[10]
        v2 = KOResetSnowballQuadEngineV2().price(p, env())
        v1 = KOResetSnowballQuadEngine(QuadParams()).price(p, env())
        assert np.isfinite(v2)
        assert v2 == pytest.approx(v1, rel=2e-3)


class TestThetaLanding:
    """AC2t: a landing record stays AT the valuation instant whatever residue its settlement or maturity leaves."""

    def _records(self):
        return ObservationSchedule(records=[
            ObservationRecord(observation_time=0.3, barrier=103.0, return_rate=0.03, settlement_time=0.3),
            ObservationRecord(observation_time=1.0, barrier=103.0, return_rate=0.03, settlement_time=1.0),
        ])

    def test_snowball_landing_keeps_exact_zero(self):
        p = SnowballOption(initial_price=100.0, strike=100.0, contract_multiplier=1.0, maturity=1.0,
                           barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.12,
                                                        ko_observation_schedule=self._records(), ki_barrier=75.0,
                                                        ki_observation_type=ObservationType.DISCRETE,
                                                        ki_observation_dates=[0.3, 1.0]))
        p.time_shift(DOWN(0.3), None, None, keep_landing=True)
        landing = p.barrier_config.ko_observation_schedule.records[0]
        assert landing.observation_time == 0.0 and at_valuation_instant(landing.observation_time)
        assert landing.settlement_time == 0.0
        assert p.barrier_config.ki_observation_schedule.records[0].observation_time == 0.0

    def test_ko_reset_shifts_both_schedules_before_canonicalizing(self):
        p = KnockOutResetSnowballOption(
            initial_price=100.0, strike=100.0, contract_multiplier=1.0, maturity=1.5,
            barrier_config=barrier((0.3, 1.0), (0.3, 1.0)), post_ko_mode=PostKOScheduleMode.ABSOLUTE,
            post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.12, ko_observation_dates=[0.3, 1.5]),
        )
        p.time_shift(DOWN(0.3), None, None, keep_landing=True)
        assert times(p.barrier_config.ko_observation_schedule)[0] == 0.0
        assert times(p.post_barrier_config.ko_observation_schedule)[0] == 0.0
        # schedule 2 at 1.5 - bump against schedule 1's 1.0 - bump: shifted together, never merged with each other
        assert times(p.post_barrier_config.ko_observation_schedule)[1] == 1.5 - DOWN(0.3)


class TestSnapContract:
    def test_day_snap_output_is_canonical(self):
        """AC8: a kept whole-day float 1 ulp off k/365 and an off-day neighbour rounding onto k/365."""
        whole = UP(100 / 365)
        p = snowball(ko_times=(whole, 0.5, 1.0), ki_times=(100.3 / 365, 0.5, 1.0))
        assert times(p.barrier_config.ko_observation_schedule)[0] != times(p.barrier_config.ki_observation_schedule)[0]
        snapped = snap_contract(p)
        ko = times(snapped.barrier_config.ko_observation_schedule)
        ki = times(snapped.barrier_config.ki_observation_schedule)
        assert ko[0] == ki[0]
        assert canonicalize_contract_clock(snapped) is False


class TestRebased:
    """AC8: every absolute REBASED schedule-2 time within the tolerance of a contract instant IS that instant."""

    def _contract(self):
        # schedule 1 ends at 0.3; a knock-in at 0.1 plus offset 0.2 spells 0.30000000000000004. The maturity covers
        # the latest end, the last knock-in 0.3 plus 0.2 (owner ruling O1, 2026-11-27)
        assert 0.1 + 0.2 != 0.3
        return KnockOutResetSnowballOption(
            initial_price=100.0, strike=100.0, contract_multiplier=1.0, maturity=0.5,
            barrier_config=barrier((0.2, 0.3), (0.1, 0.2, 0.3)), post_ko_mode=PostKOScheduleMode.REBASED,
            post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.12, ko_observation_dates=[0.1, 0.2]),
        )

    def test_generated_times_snap_onto_contract_instants(self):
        p = self._contract()
        assert 0.3 in contract_instants(p)
        rows = p.rebased_schedule_two_times(env(), [0.1], [0], np.array([0.1, 0.2]))
        assert list(rows[0]) == [0.2, 0.3]
        assert p.get_post_maturity_time(env(), ki_time=0.1) == 0.3

    def test_knocked_in_copy_agrees(self):
        p = self._contract()
        record = p.barrier_config.ki_observation_schedule.records[0]
        copy = p.with_absolute_schedule_2(0.1, anchor=knock_in_anchor_for(record, 0))
        assert times(copy.post_barrier_config.ko_observation_schedule) == [0.2, 0.3]
        assert copy.maturity == p.maturity == 0.5               # the exercise kept (run 9, D5)


class TestCheckpointOneFixes:
    def test_rebased_offsets_are_one_float_per_instant(self):
        """CP1 #1: a schedule-2 settlement offset 1 ulp after the next observation offset is that offset, so the
        knocked-in copy and the MC rows place both on one float."""
        records = [
            ObservationRecord(observation_time=0.1, barrier=95.0, return_rate=0.03, settlement_time=UP(0.2)),
            ObservationRecord(observation_time=0.2, barrier=95.0, return_rate=0.03, settlement_time=0.25),
        ]
        p = KnockOutResetSnowballOption(
            initial_price=100.0, strike=100.0, contract_multiplier=1.0, maturity=0.5,   # 0.3 + 0.2 (O1)
            barrier_config=barrier((0.2, 0.3), (0.1, 0.2, 0.3)), post_ko_mode=PostKOScheduleMode.REBASED,
            post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.12,
                                              ko_observation_schedule=ObservationSchedule(records=records)),
        )
        post = p.post_barrier_config.ko_observation_schedule.records
        assert post[0].settlement_time == post[1].observation_time == 0.2
        record = p.barrier_config.ki_observation_schedule.records[0]
        copy = p.with_absolute_schedule_2(0.1, anchor=knock_in_anchor_for(record, 0))
        rows = p.rebased_schedule_two_times(env(), [0.1], [0], np.array([r.observation_time for r in post]))
        copied = copy.post_barrier_config.ko_observation_schedule.records
        assert [r.observation_time for r in copied] == list(rows[0])
        assert copied[0].settlement_time == copied[1].observation_time

    def test_factory_ko_reset_prices_on_v2_default(self):
        """AC5, the literal factory call of the lobe re-measure."""
        from quantark.asset.equity.product.option.snowball_helpers import create_ko_reset_snowball

        p = create_ko_reset_snowball(initial_price=100.0, strike=100.0, maturity_pre=1.0, maturity_post=1.5)
        v2 = KOResetSnowballQuadEngineV2().price(p, env())
        v1 = KOResetSnowballQuadEngine(QuadParams()).price(p, env())
        assert v2 == pytest.approx(v1, rel=1e-4)
