"""
American option implementation.

American options can be exercised at any time before maturity.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from .base_equity_option import BaseEquityOption
from quantark.asset.equity.settlement import SettlementConvention
from quantark.util.calendar import DayCountConvention
from quantark.util.enum import OptionType, ExerciseType, TenorEnd
from quantark.util.exceptions import ValidationError


@dataclass
class AmericanOption(BaseEquityOption):
    """
    American call or put option.

    An American option can be exercised at any time up to and including
    the expiration date. This early exercise feature makes American options
    more valuable than their European counterparts (or at least as valuable).

    For calls on non-dividend-paying stocks, early exercise is never optimal,
    so American calls equal European calls. However, for puts or when there
    are dividends, early exercise may be optimal.

    Attributes:
        strike: Strike price
        maturity: Time to maturity in years
        option_type: CALL or PUT
    """

    def __init__(
        self,
        strike: float,
        option_type: OptionType,
        maturity: Optional[float] = None,
        exercise_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
        contract_multiplier: float = 1.0,
        tenor: Optional[float] = None,
        initial_date: Optional[datetime] = None,
        maturity_date: Optional[datetime] = None,
        tenor_end: TenorEnd = TenorEnd.EXERCISE,
        annualization_day_count: DayCountConvention = DayCountConvention.ACT_365,
        settlement_convention: Optional[SettlementConvention] = None,
    ):
        """
        Initialize American option.

        Args:
            strike: Strike price
            option_type: CALL or PUT
            maturity: Time to maturity in years (optional if exercise_date provided)
            exercise_date: Date when option expires (optional if maturity provided)
            settlement_date: Date when settlement occurs (optional, defaults to exercise_date)
            contract_multiplier: Underlying units represented by one contract
            tenor: Full contract tenor in years (optional)
            initial_date: Product start/issue date (optional)
            maturity_date: Explicit contract maturity date (optional)
            tenor_end: End-point used when deriving contract tenor
            annualization_day_count: Day count convention for contract tenor

        Note:
            Provide one pricing-expiry form: maturity/tenor or
            exercise_date/maturity_date.
        """
        super().__init__(
            strike=strike,
            maturity=maturity,
            option_type=option_type,
            exercise_type=ExerciseType.AMERICAN,
            exercise_date=exercise_date,
            settlement_date=settlement_date,
            tenor=tenor,
            initial_date=initial_date,
            maturity_date=maturity_date,
            tenor_end=tenor_end,
            annualization_day_count=annualization_day_count,
            contract_multiplier=contract_multiplier,
            settlement_convention=settlement_convention,
        )

    def get_payoff(self, spot: float) -> float:
        """
        Calculate the option payoff at exercise.

        For a call: max(S - K, 0)
        For a put:  max(K - S, 0)

        Args:
            spot: Spot price at exercise

        Returns:
            Option payoff
        """
        if spot < 0:
            raise ValidationError(f"Spot price must be non-negative, got {spot}")

        if self.is_call():
            intrinsic = max(spot - self.strike, 0.0)
        else:  # put
            intrinsic = max(self.strike - spot, 0.0)

        return intrinsic * self.contract_multiplier

    def intrinsic_value(self, spot: float) -> float:
        """
        Calculate the intrinsic value of the option.

        For American options, this is the value that could be obtained
        by exercising immediately.

        Args:
            spot: Current spot price

        Returns:
            Intrinsic value
        """
        return self.get_payoff(spot)

    def __repr__(self):
        return (
            f"AmericanOption("
            f"{self.option_type}, K={self.strike:.2f}, {self._format_maturity()})"
        )
