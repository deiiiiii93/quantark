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
