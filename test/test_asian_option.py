"""
Tests for Asian option product.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from asset.equity.product.option import AsianOption
from util.enum import OptionType, AveragingType, AsianStrikeType
from util.exceptions import ValidationError


class TestAsianOptionCreation:
    """Test Asian option construction and validation."""

    def test_fixed_strike_call_creation(self):
        """Test creating a fixed strike Asian call."""
        option = AsianOption(
            strike=100,
            option_type=OptionType.CALL,
            asian_strike_type=AsianStrikeType.FIXED,
            averaging_type=AveragingType.ARITHMETIC,
            maturity=1.0,
        )
        assert option.strike == 100
        assert option.option_type == OptionType.CALL
        assert option.asian_strike_type == AsianStrikeType.FIXED
        assert option.averaging_type == AveragingType.ARITHMETIC
        assert option.maturity == 1.0
        assert option.is_fixed_strike()
        assert option.is_arithmetic()

    def test_floating_strike_put_creation(self):
        """Test creating a floating strike Asian put."""
        option = AsianOption(
            option_type=OptionType.PUT,
            asian_strike_type=AsianStrikeType.FLOATING,
            averaging_type=AveragingType.ARITHMETIC,
            maturity=0.5,
        )
        assert option.option_type == OptionType.PUT
        assert option.asian_strike_type == AsianStrikeType.FLOATING
        assert option.is_floating_strike()
        assert not option.is_fixed_strike()

    def test_geometric_averaging(self):
        """Test creating Asian option with geometric averaging."""
        option = AsianOption(
            strike=100,
            option_type=OptionType.CALL,
            asian_strike_type=AsianStrikeType.FIXED,
            averaging_type=AveragingType.GEOMETRIC,
            maturity=1.0,
        )
        assert option.averaging_type == AveragingType.GEOMETRIC
        assert option.is_geometric()
        assert not option.is_arithmetic()

    def test_with_observation_times(self):
        """Test creating Asian option with explicit observation times."""
        obs_times = [0.25, 0.5, 0.75, 1.0]
        option = AsianOption(
            strike=100,
            option_type=OptionType.CALL,
            asian_strike_type=AsianStrikeType.FIXED,
            averaging_type=AveragingType.ARITHMETIC,
            maturity=1.0,
            observation_times=obs_times,
        )
        assert option.observation_times == obs_times
        assert option.get_observation_times() == obs_times

    def test_with_observation_dates(self):
        """Test creating Asian option with observation dates."""
        base_date = datetime(2024, 1, 1)
        obs_dates = [
            base_date + timedelta(days=90),
            base_date + timedelta(days=180),
            base_date + timedelta(days=270),
            base_date + timedelta(days=365),
        ]
        option = AsianOption(
            strike=100,
            option_type=OptionType.CALL,
            asian_strike_type=AsianStrikeType.FIXED,
            averaging_type=AveragingType.ARITHMETIC,
            exercise_date=obs_dates[-1],
            observation_dates=obs_dates,
        )
        assert option.observation_dates == obs_dates


class TestAsianOptionValidation:
    """Test Asian option validation."""

    def test_invalid_strike_for_fixed(self):
        """Test that negative strike raises error for fixed strike option."""
        with pytest.raises(ValidationError, match="Strike must be positive"):
            AsianOption(
                strike=-100,
                option_type=OptionType.CALL,
                asian_strike_type=AsianStrikeType.FIXED,
                maturity=1.0,
            )

    def test_empty_observation_times(self):
        """Test that empty observation times raises error."""
        with pytest.raises(ValidationError, match="observation_times cannot be empty"):
            AsianOption(
                strike=100,
                option_type=OptionType.CALL,
                asian_strike_type=AsianStrikeType.FIXED,
                maturity=1.0,
                observation_times=[],
            )

    def test_negative_observation_times(self):
        """Test that negative observation times raises error."""
        with pytest.raises(ValidationError, match="observation_times must be non-negative"):
            AsianOption(
                strike=100,
                option_type=OptionType.CALL,
                asian_strike_type=AsianStrikeType.FIXED,
                maturity=1.0,
                observation_times=[-0.5, 0.5, 1.0],
            )

    def test_unsorted_observation_times(self):
        """Test that unsorted observation times raises error."""
        with pytest.raises(ValidationError, match="observation_times must be sorted"):
            AsianOption(
                strike=100,
                option_type=OptionType.CALL,
                asian_strike_type=AsianStrikeType.FIXED,
                maturity=1.0,
                observation_times=[1.0, 0.5, 0.25],
            )

    def test_invalid_num_observations(self):
        """Test that num_observations < 1 raises error."""
        with pytest.raises(ValidationError, match="num_observations must be >= 1"):
            AsianOption(
                strike=100,
                option_type=OptionType.CALL,
                asian_strike_type=AsianStrikeType.FIXED,
                maturity=1.0,
                num_observations=0,
            )


class TestAsianOptionAveraging:
    """Test average computation."""

    def test_arithmetic_average(self):
        """Test arithmetic average calculation."""
        option = AsianOption(
            strike=100,
            option_type=OptionType.CALL,
            asian_strike_type=AsianStrikeType.FIXED,
            averaging_type=AveragingType.ARITHMETIC,
            maturity=1.0,
        )
        prices = [100, 105, 110, 115]
        avg = option.get_average(prices)
        assert avg == pytest.approx(107.5)

    def test_geometric_average(self):
        """Test geometric average calculation."""
        option = AsianOption(
            strike=100,
            option_type=OptionType.CALL,
            asian_strike_type=AsianStrikeType.FIXED,
            averaging_type=AveragingType.GEOMETRIC,
            maturity=1.0,
        )
        prices = [100, 105, 110, 115]
        avg = option.get_average(prices)
        # Geometric mean = (100 * 105 * 110 * 115)^(1/4)
        expected = (100 * 105 * 110 * 115) ** 0.25
        assert avg == pytest.approx(expected)

    def test_arithmetic_average_single_price(self):
        """Test arithmetic average with single price."""
        option = AsianOption(
            strike=100,
            option_type=OptionType.CALL,
            asian_strike_type=AsianStrikeType.FIXED,
            averaging_type=AveragingType.ARITHMETIC,
            maturity=1.0,
        )
        avg = option.get_average([105])
        assert avg == pytest.approx(105)

    def test_average_with_numpy_array(self):
        """Test average with numpy array input."""
        option = AsianOption(
            strike=100,
            option_type=OptionType.CALL,
            asian_strike_type=AsianStrikeType.FIXED,
            averaging_type=AveragingType.ARITHMETIC,
            maturity=1.0,
        )
        prices = np.array([100, 105, 110, 115])
        avg = option.get_average(prices)
        assert avg == pytest.approx(107.5)

    def test_average_empty_prices(self):
        """Test that empty prices raises error."""
        option = AsianOption(
            strike=100,
            option_type=OptionType.CALL,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=1.0,
        )
        with pytest.raises(ValidationError, match="prices cannot be empty"):
            option.get_average([])


class TestAsianOptionPayoff:
    """Test payoff calculations."""

    def test_fixed_strike_call_itm(self):
        """Test fixed strike call payoff when ITM."""
        option = AsianOption(
            strike=100,
            option_type=OptionType.CALL,
            asian_strike_type=AsianStrikeType.FIXED,
            averaging_type=AveragingType.ARITHMETIC,
            maturity=1.0,
        )
        prices = [100, 105, 110, 115]  # Average = 107.5
        payoff = option.get_payoff(spot=115, observed_prices=prices)
        assert payoff == pytest.approx(7.5)  # max(107.5 - 100, 0)

    def test_fixed_strike_call_otm(self):
        """Test fixed strike call payoff when OTM."""
        option = AsianOption(
            strike=110,
            option_type=OptionType.CALL,
            asian_strike_type=AsianStrikeType.FIXED,
            averaging_type=AveragingType.ARITHMETIC,
            maturity=1.0,
        )
        prices = [100, 105, 110, 100]  # Average = 103.75
        payoff = option.get_payoff(spot=100, observed_prices=prices)
        assert payoff == pytest.approx(0.0)  # max(103.75 - 110, 0)

    def test_fixed_strike_put_itm(self):
        """Test fixed strike put payoff when ITM."""
        option = AsianOption(
            strike=110,
            option_type=OptionType.PUT,
            asian_strike_type=AsianStrikeType.FIXED,
            averaging_type=AveragingType.ARITHMETIC,
            maturity=1.0,
        )
        prices = [100, 105, 100, 95]  # Average = 100
        payoff = option.get_payoff(spot=95, observed_prices=prices)
        assert payoff == pytest.approx(10.0)  # max(110 - 100, 0)

    def test_floating_strike_call_itm(self):
        """Test floating strike call payoff when ITM."""
        option = AsianOption(
            option_type=OptionType.CALL,
            asian_strike_type=AsianStrikeType.FLOATING,
            averaging_type=AveragingType.ARITHMETIC,
            maturity=1.0,
        )
        prices = [100, 105, 110, 105]  # Average = 105
        payoff = option.get_payoff(spot=115, observed_prices=prices)
        assert payoff == pytest.approx(10.0)  # max(115 - 105, 0)

    def test_floating_strike_call_otm(self):
        """Test floating strike call payoff when OTM."""
        option = AsianOption(
            option_type=OptionType.CALL,
            asian_strike_type=AsianStrikeType.FLOATING,
            averaging_type=AveragingType.ARITHMETIC,
            maturity=1.0,
        )
        prices = [100, 110, 115, 120]  # Average = 111.25
        payoff = option.get_payoff(spot=105, observed_prices=prices)
        assert payoff == pytest.approx(0.0)  # max(105 - 111.25, 0)

    def test_floating_strike_put_itm(self):
        """Test floating strike put payoff when ITM."""
        option = AsianOption(
            option_type=OptionType.PUT,
            asian_strike_type=AsianStrikeType.FLOATING,
            averaging_type=AveragingType.ARITHMETIC,
            maturity=1.0,
        )
        prices = [100, 105, 110, 115]  # Average = 107.5
        payoff = option.get_payoff(spot=95, observed_prices=prices)
        assert payoff == pytest.approx(12.5)  # max(107.5 - 95, 0)

    def test_payoff_with_precomputed_average(self):
        """Test payoff with pre-computed average."""
        option = AsianOption(
            strike=100,
            option_type=OptionType.CALL,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=1.0,
        )
        payoff = option.get_payoff(spot=110, average=108)
        assert payoff == pytest.approx(8.0)

    def test_payoff_missing_prices_and_average(self):
        """Test that payoff raises error without prices or average."""
        option = AsianOption(
            strike=100,
            option_type=OptionType.CALL,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=1.0,
        )
        with pytest.raises(ValidationError, match="Either observed_prices or average must be provided"):
            option.get_payoff(spot=110)

    def test_payoff_negative_spot(self):
        """Test that negative spot price raises error."""
        option = AsianOption(
            strike=100,
            option_type=OptionType.CALL,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=1.0,
        )
        with pytest.raises(ValidationError, match="Spot price must be non-negative"):
            option.get_payoff(spot=-10, average=105)


class TestAsianOptionObservationSchedule:
    """Test observation schedule generation."""

    def test_default_observations(self):
        """Test default uniform observation schedule."""
        option = AsianOption(
            strike=100,
            option_type=OptionType.CALL,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=1.0,
            num_observations=12,
        )
        obs_times = option.get_observation_times()
        assert len(obs_times) == 12
        assert obs_times[0] == pytest.approx(1.0 / 12)
        assert obs_times[-1] == pytest.approx(1.0)

    def test_custom_num_observations(self):
        """Test custom number of observations."""
        option = AsianOption(
            strike=100,
            option_type=OptionType.CALL,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=0.5,
            num_observations=6,
        )
        obs_times = option.get_observation_times()
        assert len(obs_times) == 6
        assert obs_times[-1] == pytest.approx(0.5)

    def test_explicit_observation_times_override(self):
        """Test that explicit observation times override generation."""
        explicit_times = [0.1, 0.3, 0.5, 0.8, 1.0]
        option = AsianOption(
            strike=100,
            option_type=OptionType.CALL,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=1.0,
            observation_times=explicit_times,
            num_observations=20,  # Should be ignored
        )
        obs_times = option.get_observation_times()
        assert obs_times == explicit_times


class TestAsianOptionRepr:
    """Test string representation."""

    def test_fixed_strike_repr(self):
        """Test repr for fixed strike option."""
        option = AsianOption(
            strike=100,
            option_type=OptionType.CALL,
            asian_strike_type=AsianStrikeType.FIXED,
            averaging_type=AveragingType.ARITHMETIC,
            maturity=1.0,
        )
        repr_str = repr(option)
        assert "AsianOption" in repr_str
        assert "Call" in repr_str
        assert "K=100.00" in repr_str
        assert "Fixed" in repr_str
        assert "Arithmetic" in repr_str

    def test_floating_strike_repr(self):
        """Test repr for floating strike option."""
        option = AsianOption(
            option_type=OptionType.PUT,
            asian_strike_type=AsianStrikeType.FLOATING,
            averaging_type=AveragingType.GEOMETRIC,
            maturity=0.5,
        )
        repr_str = repr(option)
        assert "floating" in repr_str
        assert "Floating" in repr_str
        assert "Geometric" in repr_str
