"""Contract clock (spec .ship/2026-11-23-contract-clock, design 2026-11-23-contract-clock-design.md).

Layer 1: one schedule-time generator whose equal calendar instants are bit-identical floats (AC1).
"""
from fractions import Fraction

import pytest

from quantark.asset.equity.product.option.contract_clock import schedule_times
from quantark.asset.equity.product.option.snowball_helpers import (
    create_ko_reset_snowball,
    generate_ko_observation_dates,
)
from quantark.util.exceptions import ValidationError

MATURITIES = ("0.25", "0.5", "0.75", "1", "1.25", "1.5", "2", "2.5", "3")
COUNTS = (1, 2, 3, 4, 6, 12, 18, 24, 36, 52, 252, 378, 504)


class TestScheduleTimes:
    def test_single_rounding_formula_and_pinned_last(self):
        times = schedule_times(1.5, 18)
        assert len(times) == 18
        assert times[:-1] == [(i + 1) * 1.5 / 18 for i in range(17)]
        assert times[-1] == 1.5

    def test_equal_calendar_instants_are_bit_identical_across_schedules(self):
        """AC1: two schedules that share an exact rational instant hold the same float there."""
        floats_at = {}
        for text in MATURITIES:
            exact = Fraction(text)
            for n in COUNTS:
                for i, t in enumerate(schedule_times(float(exact), n)):
                    floats_at.setdefault(exact * (i + 1) / n, set()).add(t)
        clashes = {k: v for k, v in floats_at.items() if len(v) > 1}
        assert not clashes, f"{len(clashes)} instants with several floats, e.g. {next(iter(clashes.items()))}"

    def test_ko_reset_factory_pair_is_equal(self):
        """The 11/12 * 1.0 vs 11/18 * 1.5 pair the lobe re-measure tripped on."""
        assert schedule_times(1.0, 12)[10] == schedule_times(1.5, 18)[10]
        product = create_ko_reset_snowball(
            initial_price=100.0, strike=100.0, maturity_pre=1.0, maturity_post=1.5
        )
        pre = product.barrier_config.ko_observation_dates
        post = product.post_barrier_config.ko_observation_dates
        assert pre[10] == post[10] == 11 * 1.0 / 12

    @pytest.mark.parametrize("bad", [(1.0, 0), (0.0, 4), (-1.0, 4), (float("nan"), 4), (float("inf"), 4)])
    def test_refuses_bad_input(self, bad):
        with pytest.raises(ValidationError):
            schedule_times(*bad)

    def test_generate_ko_observation_dates_uses_the_generator(self):
        assert generate_ko_observation_dates(1.5, "monthly") == schedule_times(1.5, 18)
        assert generate_ko_observation_dates(1.5, "monthly", skip_first=3) == schedule_times(1.5, 18)[3:]


# ---------------------------------------------------------------------------------------------------------------
# Layer 2 primitives (T2): the canonical clock, the shared engine node merge and the instant snap.
# ---------------------------------------------------------------------------------------------------------------
import numpy as np

from quantark.asset.equity.product.option import contract_clock
from quantark.asset.equity.product.option.contract_clock import (
    canonical_clock_after,
    canonicalize_contract_clock,
    contract_instants,
    snap_to_instants,
)
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.util.enum import ObservationType
from quantark.util.numerical import Tolerance, merge_time_nodes, same_instant

UP = lambda t: float(np.nextafter(t, np.inf))      # noqa: E731 -- one ulp after
DOWN = lambda t: float(np.nextafter(t, -np.inf))   # noqa: E731 -- one ulp before
SECOND = 1.0 / (365.0 * 86400.0)


def _snowball(ko_times=(0.25, 0.5, 0.75, 1.0), ki_times=None, maturity=None, ko_schedule=None):
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        contract_multiplier=1.0,
        maturity=maturity if maturity is not None else ko_times[-1],
        barrier_config=BarrierConfig(
            ko_barrier=103.0,
            ko_rate=0.12,
            ko_observation_dates=None if ko_schedule is not None else list(ko_times),
            ko_observation_schedule=ko_schedule,
            ki_barrier=75.0,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=list(ki_times if ki_times is not None else ko_times),
        ),
    )


def _ko(product):
    return [r.observation_time for r in product.barrier_config.ko_observation_schedule.records]


def _ki(product):
    return [r.observation_time for r in product.barrier_config.ki_observation_schedule.records]


class TestCanonicalize:
    def test_cross_schedule_ulp_pair_becomes_the_last_member(self):
        """AC2(a): both spellings become the run's LAST float, in the schedules and in the legacy lists."""
        p = _snowball(ki_times=(0.25, UP(0.5), 0.75, 1.0))
        assert canonicalize_contract_clock(p) is True
        assert _ko(p)[1] == _ki(p)[1] == UP(0.5)
        assert p.barrier_config.ko_observation_dates[1] == p.barrier_config.ki_observation_dates[1] == UP(0.5)

    def test_time_within_tol_of_maturity_becomes_maturity(self):
        """AC2(b): maturity is pinned even when it is not the LAST member."""
        p = _snowball(ki_times=(0.25, 0.5, 0.75, UP(1.0)))
        assert canonicalize_contract_clock(p) is True
        assert _ki(p)[-1] == 1.0 and p.maturity == 1.0

    def test_one_second_apart_stays_two_instants(self):
        """AC2(c), owner risk #1: intraday events one second apart are distinct."""
        p = _snowball(ki_times=(0.25, 0.5 + SECOND, 0.75, 1.0))
        assert canonicalize_contract_clock(p) is False
        assert _ki(p)[1] == 0.5 + SECOND and _ko(p)[1] == 0.5

    def test_equal_times_unchanged_and_idempotent(self):
        """AC2(d,e,f): nothing moves, nothing is rebound; a second call after a move is a no-op."""
        p = _snowball()
        config, ko, ki = p.barrier_config, p.barrier_config.ko_observation_schedule, p.barrier_config.ki_observation_schedule
        assert canonicalize_contract_clock(p) is False
        assert p.barrier_config is config
        assert p.barrier_config.ko_observation_schedule is ko and p.barrier_config.ki_observation_schedule is ki
        q = _snowball(ki_times=(0.25, UP(0.5), 0.75, 1.0))
        assert canonicalize_contract_clock(q) is True
        moved = q.barrier_config
        assert canonicalize_contract_clock(q) is False
        assert q.barrier_config is moved

    def test_records_and_lists_are_rebuilt_not_mutated(self):
        """A shallow copy (QUAD V2 validates one) must not see the original's records or lists rewritten."""
        p = _snowball(ki_times=(0.25, UP(0.5), 0.75, 1.0))
        old_record = p.barrier_config.ko_observation_schedule.records[1]
        old_list = p.barrier_config.ko_observation_dates
        canonicalize_contract_clock(p)
        assert old_record.observation_time == 0.5 and old_list[1] == 0.5

    def test_same_schedule_distinct_observation_floats_refused(self):
        """AC2(g): two records of ONE schedule landing on one instant; the error names both times."""
        p = _snowball(ko_times=(0.25, 0.5, UP(0.5), 1.0), ki_times=(0.25, 0.5, 0.75, 1.0))
        with pytest.raises(ValidationError, match=repr(UP(0.5))):
            canonicalize_contract_clock(p)

    def test_settlement_float_one_ulp_off_next_observation_is_not_refused(self):
        """AC2(h): a T+lag settlement landing 1 ulp off the next observation merges and is never refused."""
        records = [
            ObservationRecord(observation_time=0.25, barrier=103.0, return_rate=0.03, settlement_time=DOWN(0.5)),
            ObservationRecord(observation_time=0.5, barrier=103.0, return_rate=0.03, settlement_time=0.75),
            ObservationRecord(observation_time=1.0, barrier=103.0, return_rate=0.03, settlement_time=1.0),
        ]
        p = _snowball(ko_schedule=ObservationSchedule(records=records), ki_times=(0.25, 0.5, 1.0), maturity=1.0)
        assert canonicalize_contract_clock(p) is True
        assert p.barrier_config.ko_observation_schedule.records[0].settlement_time == 0.5

    def test_time_within_tol_of_zero_becomes_exact_zero(self):
        """AC2(i): zero is pinned (a theta landing stays at the valuation instant)."""
        p = _snowball(ki_times=(5e-11, 0.5, 0.75, 1.0))
        assert canonicalize_contract_clock(p) is True
        assert _ki(p)[0] == 0.0

    def test_zero_beats_maturity_and_chains_are_refused(self):
        """AC2(i) for a maturity residue, AC2(j) for a run wider than the tolerance (core map)."""
        mapping = contract_clock._canonical_map([5e-11, -3e-17, 0.4], maturity=5e-11)
        assert mapping[5e-11] == 0.0 and mapping[-3e-17] == 0.0 and mapping[0.4] == 0.4
        step = 0.6 * Tolerance.ZERO
        with pytest.raises(ValidationError, match="wider than"):
            contract_clock._canonical_map([0.5, 0.5 + step, 0.5 + 2 * step], maturity=1.0)

    def test_contract_instants_sorted_unique(self):
        p = _snowball(ki_times=(0.25, 0.5, 0.6, 1.0))
        assert list(contract_instants(p)) == [0.25, 0.5, 0.6, 0.75, 1.0]


class TestSnapAndMerge:
    def test_snap_to_nearest_instant_earlier_on_tie(self):
        instants = np.array([0.5, 0.5 + Tolerance.ZERO, 1.0])
        out = snap_to_instants([UP(0.5), 0.5 + 0.5 * Tolerance.ZERO, 0.7, DOWN(1.0)], instants)
        assert list(out) == [0.5, 0.5, 0.7, 1.0]

    def test_merge_time_nodes_without_anchors_is_unchanged(self):
        times = [0.0, 0.25, UP(0.25), 0.5, 1.0]
        assert merge_time_nodes(times, pin=1.0).tobytes() == merge_time_nodes(times, pin=1.0, anchors=None).tobytes()

    def test_engine_node_snaps_onto_contract_anchor(self):
        """D2: an engine-added node a hair AFTER a contract instant does not displace it."""
        nodes = merge_time_nodes([0.25, UP(0.5), 0.5, 1.0], pin=1.0, anchors=[0.5, 1.0])
        assert list(nodes) == [0.25, 0.5, 1.0]
        assert list(merge_time_nodes([0.25, UP(0.5), 0.5, 1.0], pin=1.0)) == [0.25, UP(0.5), 1.0]

    def test_same_instant(self):
        assert same_instant(0.5, UP(0.5)) and not same_instant(0.5, 0.5 + SECOND)


class TestCanonicalClockAfter:
    def test_only_the_outermost_call_canonicalizes(self, monkeypatch):
        calls = []
        monkeypatch.setattr(contract_clock, "canonicalize_contract_clock", lambda p: calls.append(p) or False)

        class Base:
            @canonical_clock_after
            def time_shift(self):
                calls.append("base")

        class Child(Base):
            @canonical_clock_after
            def time_shift(self):
                super().time_shift()
                calls.append("child")

        c = Child()
        c.time_shift()
        assert calls == ["base", "child", c]
        c.time_shift()
        assert calls[-1] is c and len(calls) == 6
