"""
Unit tests for DoubleSharkfinOption.
"""

import pytest

from quantark.asset.equity.product.option import (
    DoubleSharkfinOption,
    ObservationRecord,
    ObservationSchedule,
)
from quantark.util.enum import ObservationFrequency, ObservationType, OptionType
from quantark.util.exceptions import ValidationError


class TestDoubleSharkfinOption:
    """Tests for double sharkfin product behavior."""

    def test_call_expiry_no_hit_payoff(self):
        option = DoubleSharkfinOption(
            strike=100.0,
            option_type=OptionType.CALL,
            lower_barrier=80.0,
            upper_barrier=120.0,
            maturity=1.0,
            observation_type=ObservationType.EXPIRY,
        )

        assert option.get_payoff(110.0) == pytest.approx(10.0)

    def test_call_expiry_hit_pays_knock_out_rebate(self):
        option = DoubleSharkfinOption(
            strike=100.0,
            option_type=OptionType.CALL,
            lower_barrier=80.0,
            upper_barrier=120.0,
            maturity=1.0,
            knock_out_rebate=2.0,
            observation_type=ObservationType.EXPIRY,
        )

        assert option.get_payoff(121.0) == pytest.approx(2.0)
        assert option.get_payoff(79.0) == pytest.approx(2.0)

    def test_put_expiry_no_hit_payoff(self):
        option = DoubleSharkfinOption(
            strike=100.0,
            option_type=OptionType.PUT,
            lower_barrier=80.0,
            upper_barrier=120.0,
            maturity=1.0,
            observation_type=ObservationType.EXPIRY,
        )

        assert option.get_payoff(90.0) == pytest.approx(10.0)

    def test_put_expiry_hit_pays_knock_out_rebate(self):
        option = DoubleSharkfinOption(
            strike=100.0,
            option_type=OptionType.PUT,
            lower_barrier=80.0,
            upper_barrier=120.0,
            maturity=1.0,
            knock_out_rebate=1.5,
            observation_type=ObservationType.EXPIRY,
        )

        assert option.get_payoff(79.0) == pytest.approx(1.5)
        assert option.get_payoff(121.0) == pytest.approx(1.5)

    def test_no_hit_payoff_is_capped_by_relevant_barrier(self):
        call_option = DoubleSharkfinOption(
            strike=100.0,
            option_type=OptionType.CALL,
            lower_barrier=80.0,
            upper_barrier=120.0,
            maturity=1.0,
            participation_rate=0.5,
            no_hit_rebate=1.0,
        )
        put_option = DoubleSharkfinOption(
            strike=100.0,
            option_type=OptionType.PUT,
            lower_barrier=80.0,
            upper_barrier=120.0,
            maturity=1.0,
            participation_rate=0.5,
            no_hit_rebate=1.0,
        )

        assert call_option.get_payoff(140.0, barrier_hit=False) == pytest.approx(11.0)
        assert put_option.get_payoff(60.0, barrier_hit=False) == pytest.approx(11.0)

    def test_payoff_scales_by_contract_multiplier(self):
        option = DoubleSharkfinOption(
            strike=100.0,
            option_type=OptionType.CALL,
            lower_barrier=80.0,
            upper_barrier=120.0,
            maturity=1.0,
            contract_multiplier=100.0,
        )

        assert option.get_payoff(110.0) == pytest.approx(1000.0)

    def test_discrete_daily_schedule_is_generated(self):
        option = DoubleSharkfinOption(
            strike=100.0,
            option_type=OptionType.CALL,
            lower_barrier=80.0,
            upper_barrier=120.0,
            maturity=1.0,
            observation_type=ObservationType.DISCRETE,
            observation_frequency=ObservationFrequency.DAILY,
        )

        times = option.get_observation_times()
        assert len(times) == 252
        assert times[-1] == pytest.approx(1.0)
        assert option.observation_schedule.frequency == ObservationFrequency.DAILY

    def test_discrete_custom_requires_observation_dates(self):
        with pytest.raises(ValidationError, match="Observation dates required"):
            DoubleSharkfinOption(
                strike=100.0,
                option_type=OptionType.CALL,
                lower_barrier=80.0,
                upper_barrier=120.0,
                maturity=1.0,
                observation_type=ObservationType.DISCRETE,
            )

    def test_observation_schedule_is_normalized(self):
        schedule = ObservationSchedule(
            records=[ObservationRecord(observation_time=0.5)]
        )
        option = DoubleSharkfinOption(
            strike=100.0,
            option_type=OptionType.CALL,
            lower_barrier=80.0,
            upper_barrier=120.0,
            maturity=1.0,
            observation_type=ObservationType.DISCRETE,
            observation_schedule=schedule,
            knock_out_rebate=3.0,
        )

        record = option.observation_schedule.records[0]
        assert record.lower_barrier == pytest.approx(80.0)
        assert record.upper_barrier == pytest.approx(120.0)
        assert record.payoff == pytest.approx(3.0)

    def test_discrete_observation_dates_must_be_sorted(self):
        with pytest.raises(ValidationError, match="sorted"):
            DoubleSharkfinOption(
                strike=100.0,
                option_type=OptionType.CALL,
                lower_barrier=80.0,
                upper_barrier=120.0,
                maturity=1.0,
                observation_type=ObservationType.DISCRETE,
                observation_dates=[0.5, 0.25],
            )

    def test_strike_must_be_inside_corridor(self):
        with pytest.raises(ValidationError, match="lower_barrier < strike"):
            DoubleSharkfinOption(
                strike=100.0,
                option_type=OptionType.CALL,
                lower_barrier=100.0,
                upper_barrier=120.0,
                maturity=1.0,
            )

    def test_path_barrier_detection_for_continuous_monitoring(self):
        option = DoubleSharkfinOption(
            strike=100.0,
            option_type=OptionType.CALL,
            lower_barrier=80.0,
            upper_barrier=120.0,
            maturity=1.0,
            observation_type=ObservationType.CONTINUOUS,
        )

        assert option.has_barrier_hit([100.0, 110.0, 121.0])
        assert option.has_barrier_hit([100.0, 90.0, 79.0])
        assert not option.has_barrier_hit([100.0, 110.0, 119.0])

    def test_pay_at_hit_flag_is_stored(self):
        option = DoubleSharkfinOption(
            strike=100.0,
            option_type=OptionType.CALL,
            lower_barrier=80.0,
            upper_barrier=120.0,
            maturity=1.0,
            pay_at_hit=True,
        )

        assert option.pay_at_hit is True

    def test_pay_at_hit_must_be_boolean(self):
        with pytest.raises(ValidationError, match="pay_at_hit must be boolean"):
            DoubleSharkfinOption(
                strike=100.0,
                option_type=OptionType.CALL,
                lower_barrier=80.0,
                upper_barrier=120.0,
                maturity=1.0,
                pay_at_hit="yes",
            )
