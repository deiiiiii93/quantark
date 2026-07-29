"""
Base class for equity options.

This module provides the abstract base class for all equity option products,
defining common attributes and methods for date handling, contract scaling,
maturity/tenor calculations, and option type checks.
"""

from abc import abstractmethod
from typing import Optional
from datetime import datetime

from quantark.util.calendar import DayCountConvention, calculate_year_fraction
from quantark.asset.equity.settlement import SettlementConvention
from ..base_equity_product import BaseEquityProduct
from quantark.util.enum import OptionType, ExerciseType, TenorEnd
from quantark.util.exceptions import ValidationError


class BaseEquityOption(BaseEquityProduct):
    """
    Abstract base class for equity options.

    This class provides fundamental attributes and methods for equity option
    products including:
    - Date fields (initial, exercise, settlement, maturity dates)
    - Type fields (option type, exercise type)
    - Contract multiplier management
    - Tenor/maturity calculation methods
    - Moneyness utilities

    Attributes:
        initial_date: Product start/issue date (optional)
        exercise_date: Date when option can be exercised (optional)
        settlement_date: Date when settlement occurs (optional)
        maturity_date: Explicit maturity date, may differ from exercise_date (optional)
        tenor: Contract tenor in years from issue to expiry (optional)
        maturity: Time to maturity in years from valuation (optional)
        tenor_end: Determines which date to use for tenor calculations
        annualization_day_count: Day count convention for year fraction calculations
        option_type: CALL or PUT
        exercise_type: EUROPEAN, AMERICAN, or BERMUDAN
        strike: Strike price
        contract_multiplier: Underlying units represented by one contract

    ``initial_price`` is intentionally not part of this base contract.  A
    reference fixing is a contractual term only for products whose payoff is
    defined relative to inception (for example Asians and autocallables).
    Those products declare and validate it themselves; absolute-strike options
    obtain valuation spot from the pricing environment.
    """

    # ==========================================================================
    # Date parameters
    # ==========================================================================
    initial_date: Optional[datetime] = None
    exercise_date: Optional[datetime] = None
    settlement_date: Optional[datetime] = None
    maturity_date: Optional[datetime] = None
    tenor: Optional[float] = None
    maturity: Optional[float] = None
    tenor_end: TenorEnd = TenorEnd.EXERCISE
    annualization_day_count: DayCountConvention = DayCountConvention.ACT_365
    settlement_convention: Optional[SettlementConvention] = None

    # ==========================================================================
    # Type parameters
    # ==========================================================================
    option_type: OptionType
    exercise_type: ExerciseType

    # ==========================================================================
    # Price parameters
    # ==========================================================================
    strike: float = 0.0

    # ==========================================================================
    # Contract parameters
    # ==========================================================================
    contract_multiplier: float = 1.0

    def __init__(
        self,
        strike: float = 0.0,
        option_type: OptionType = None,
        exercise_type: ExerciseType = ExerciseType.EUROPEAN,
        maturity: Optional[float] = None,
        tenor: Optional[float] = None,
        initial_date: Optional[datetime] = None,
        exercise_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
        maturity_date: Optional[datetime] = None,
        tenor_end: TenorEnd = TenorEnd.EXERCISE,
        annualization_day_count: DayCountConvention = DayCountConvention.ACT_365,
        contract_multiplier: float = 1.0,
        settlement_convention: Optional[SettlementConvention] = None,
    ):
        """
        Initialize base equity option.

        Args:
            strike: Strike price
            option_type: CALL or PUT
            exercise_type: EUROPEAN, AMERICAN, or BERMUDAN
            maturity: Time to maturity in years from valuation (optional)
            tenor: Contract tenor in years from issue to expiry. At inception it
                also supplies maturity when no other expiry representation is set.
            initial_date: Product start/issue date (optional)
            exercise_date: Date when option can be exercised (optional)
            settlement_date: Date when settlement occurs (optional)
            maturity_date: Explicit contract maturity date. It also supplies the
                exercise/expiry date when no exercise_date or maturity is set.
            tenor_end: Determines which date to use for tenor calculations
            annualization_day_count: Day count convention for year fractions
            contract_multiplier: Underlying units represented by one contract
            settlement_convention: Optional contractual payment lag convention
        """
        # Normalize alternate term-sheet representations into the two canonical
        # pricing forms used by engines: a remaining year fraction (`maturity`)
        # or an explicit expiry (`exercise_date`).  Keep the original tenor and
        # maturity_date as contract metadata for annualization/settlement logic.
        maturity_is_unset = maturity is None or maturity == 0.0
        if exercise_date is None and maturity_is_unset:
            if maturity_date is not None:
                exercise_date = maturity_date
            elif tenor is not None:
                maturity = tenor

        self.strike = strike
        self.option_type = option_type
        self.exercise_type = exercise_type
        self.maturity = maturity
        self.tenor = tenor
        self.initial_date = initial_date
        self.exercise_date = exercise_date
        self.settlement_date = settlement_date
        self.maturity_date = maturity_date
        self.tenor_end = tenor_end
        self.annualization_day_count = annualization_day_count
        self.contract_multiplier = contract_multiplier
        self.settlement_convention = settlement_convention

        self.validate()

    def __post_init__(self):
        """Validate option parameters after dataclass initialization."""
        self.validate()

    # ==========================================================================
    # Validation
    # ==========================================================================

    def validate(self) -> None:
        """
        Validate option parameters.

        Performs comprehensive validation including:
        - Strike price positivity
        - Maturity/exercise_date mutual exclusivity
        - Date ordering consistency
        - Type enum validation
        - Tenor end configuration validity

        Raises:
            ValidationError: If any parameter is invalid
        """
        self._validate_strike()
        self._validate_maturity_dates()
        self._validate_date_ordering()
        self._validate_types()
        self._validate_tenor_end()
        self._validate_contract_multiplier()
        self._validate_settlement_convention()

    def _validate_strike(self) -> None:
        """Validate strike price is positive."""
        if self.strike <= 0:
            raise ValidationError(f"Strike must be positive, got {self.strike}")

    def _validate_contract_multiplier(self) -> None:
        """Validate contract multiplier is positive."""
        if self.contract_multiplier <= 0:
            raise ValidationError(
                f"Contract multiplier must be positive, got {self.contract_multiplier}"
            )

    def _validate_settlement_convention(self) -> None:
        """Validate the optional contractual payment-lag convention."""
        if (
            self.settlement_convention is not None
            and not isinstance(self.settlement_convention, SettlementConvention)
        ):
            raise ValidationError(
                "settlement_convention must be SettlementConvention or None, "
                f"got {type(self.settlement_convention).__name__}"
            )

    def _validate_maturity_dates(self) -> None:
        """Validate maturity and date parameters are properly specified."""
        has_dates = self.exercise_date is not None
        has_maturity = self.maturity is not None and self.maturity > 0

        if self.maturity is not None and self.maturity < 0:
            raise ValidationError(f"Maturity must be positive, got {self.maturity}")

        if self.tenor is not None and self.tenor <= 0:
            raise ValidationError(f"Tenor must be positive, got {self.tenor}")

        if not has_dates and not has_maturity:
            raise ValidationError(
                "Either maturity/tenor or exercise_date/maturity_date must be provided"
            )

        if has_dates and has_maturity:
            raise ValidationError("Cannot provide both maturity and exercise_date")

    def _validate_date_ordering(self) -> None:
        """Validate that dates are in proper chronological order."""
        # initial_date should be before exercise_date
        if (
            self.initial_date is not None
            and self.exercise_date is not None
            and self.initial_date >= self.exercise_date
        ):
            raise ValidationError(
                f"Initial date ({self.initial_date}) must be before "
                f"exercise date ({self.exercise_date})"
            )

        # settlement_date should be >= exercise_date
        if (
            self.settlement_date is not None
            and self.exercise_date is not None
            and self.settlement_date < self.exercise_date
        ):
            raise ValidationError(
                f"Settlement date ({self.settlement_date}) must be >= "
                f"exercise date ({self.exercise_date})"
            )

        # maturity_date should be >= exercise_date if both provided
        if (
            self.maturity_date is not None
            and self.exercise_date is not None
            and self.maturity_date < self.exercise_date
        ):
            raise ValidationError(
                f"Maturity date ({self.maturity_date}) must be >= "
                f"exercise date ({self.exercise_date})"
            )

    def _validate_types(self) -> None:
        """Validate enum types."""
        if not isinstance(self.option_type, OptionType):
            raise ValidationError(f"Invalid option type: {self.option_type}")
        if not isinstance(self.exercise_type, ExerciseType):
            raise ValidationError(f"Invalid exercise type: {self.exercise_type}")
        if not isinstance(self.tenor_end, TenorEnd):
            raise ValidationError(f"Invalid tenor_end: {self.tenor_end}")
        if not isinstance(self.annualization_day_count, DayCountConvention):
            raise ValidationError(
                f"Invalid annualization_day_count: {self.annualization_day_count}"
            )

    def _validate_tenor_end(self) -> None:
        """Validate tenor_end configuration is consistent with available dates."""
        if self.tenor_end == TenorEnd.SETTLEMENT and self.settlement_date is None:
            # Only raise if using date-based calculation
            if self.exercise_date is not None:
                raise ValidationError(
                    "settlement_date required when tenor_end is SETTLEMENT"
                )

        if self.tenor_end == TenorEnd.MATURITY:
            if (
                self.maturity_date is None
                and self.exercise_date is None
                and self.maturity is None
            ):
                raise ValidationError(
                    "maturity_date, exercise_date, or maturity required "
                    "when tenor_end is MATURITY"
                )

    # ==========================================================================
    # Maturity and Tenor Calculations
    # ==========================================================================

    def get_maturity(self, pricing_env=None) -> float:
        """
        Get time to maturity in years from valuation date.

        If exercise_date is provided, calculates maturity from pricing_env.valuation_date
        to exercise_date using the day count convention. Otherwise returns the
        pre-specified maturity float.

        Args:
            pricing_env: Pricing environment with valuation_date and day_count_convention
                        (required if using date-based calculation)

        Returns:
            Time to maturity in years

        Raises:
            ValidationError: If date-based but no pricing_env provided, or if
                           valuation date is on or after exercise date
        """
        if self.exercise_date is not None:
            if pricing_env is None:
                raise ValidationError(
                    "PricingEnvironment required for date-based maturity calculation"
                )

            # Validate valuation date is before exercise date
            if pricing_env.valuation_date >= self.exercise_date:
                raise ValidationError(
                    f"Valuation date ({pricing_env.valuation_date}) must be before "
                    f"exercise date ({self.exercise_date})"
                )

            return calculate_year_fraction(
                pricing_env.valuation_date,
                self.exercise_date,
                pricing_env.day_count_convention,
                pricing_env.bus_days_in_year,
                calendar=getattr(pricing_env, "calendar", None),
            )
        else:
            if self.maturity is None:
                raise ValidationError("Maturity is not set")
            return self.maturity

    def get_tenor(self, pricing_env=None) -> float:
        """
        Get the contract tenor in years.

        The tenor represents the full contract duration from initial_date to the
        tenor end date (determined by tenor_end setting). This differs from
        maturity which is the remaining time from valuation date.

        Resolution order:
        1. If tenor is directly set, return it
        2. If initial_date is available, calculate from initial_date to tenor end date
        3. Fall back to maturity as an approximation

        Args:
            pricing_env: Pricing environment for date-based calculations (optional)

        Returns:
            Contract tenor in years

        Raises:
            ValidationError: If tenor cannot be determined
        """
        # If tenor is directly specified, use it
        if self.tenor is not None:
            return self.tenor

        # Calculate from dates if available
        if self.initial_date is not None:
            end_date = self.get_tenor_end_date()
            if end_date is not None:
                return calculate_year_fraction(
                    self.initial_date,
                    end_date,
                    self.annualization_day_count,
                )

        # Fall back to maturity if no other information available
        if self.maturity is not None:
            return self.maturity

        # Cannot determine tenor
        raise ValidationError(
            "Tenor cannot be determined: set tenor, or provide initial_date with "
            "exercise_date/settlement_date/maturity_date"
        )

    def get_tenor_end_date(self) -> Optional[datetime]:
        """
        Get the tenor end date based on the tenor_end setting.

        Returns:
            The appropriate end date for tenor calculations, or None if not available

        The date returned depends on tenor_end:
        - EXERCISE: Returns exercise_date
        - SETTLEMENT: Returns settlement_date (falls back to exercise_date)
        - MATURITY: Returns maturity_date (falls back to exercise_date)
        """
        if self.tenor_end == TenorEnd.EXERCISE:
            return self.exercise_date
        elif self.tenor_end == TenorEnd.SETTLEMENT:
            return self.settlement_date or self.exercise_date
        elif self.tenor_end == TenorEnd.MATURITY:
            return self.maturity_date or self.exercise_date
        return self.exercise_date

    def get_time_to_date(self, target_date: datetime, pricing_env=None) -> float:
        """
        Calculate time in years from valuation date to a target date.

        Args:
            target_date: Target date to calculate time to
            pricing_env: Pricing environment with valuation_date (required)

        Returns:
            Time in years from valuation date to target date

        Raises:
            ValidationError: If pricing_env is not provided or target_date is invalid
        """
        if pricing_env is None:
            raise ValidationError(
                "PricingEnvironment required for date-based calculation"
            )

        if target_date is None:
            raise ValidationError("Target date cannot be None")

        if pricing_env.valuation_date >= target_date:
            raise ValidationError(
                f"Valuation date ({pricing_env.valuation_date}) must be before "
                f"target date ({target_date})"
            )

        return calculate_year_fraction(
            pricing_env.valuation_date,
            target_date,
            pricing_env.day_count_convention,
            getattr(pricing_env, "bus_days_in_year", 252),
            calendar=getattr(pricing_env, "calendar", None),
        )

    # ==========================================================================
    # Option Type Checks
    # ==========================================================================

    def is_call(self) -> bool:
        """
        Check if option is a call.

        Returns:
            True if option type is CALL, False otherwise

        Raises:
            ValidationError: If option_type is not set
        """
        if not hasattr(self, "option_type") or self.option_type is None:
            raise ValidationError(
                "option_type is not set. This method is only available for "
                "vanilla options with a defined option type."
            )
        return self.option_type == OptionType.CALL

    def is_put(self) -> bool:
        """
        Check if option is a put.

        Returns:
            True if option type is PUT, False otherwise

        Raises:
            ValidationError: If option_type is not set
        """
        if not hasattr(self, "option_type") or self.option_type is None:
            raise ValidationError(
                "option_type is not set. This method is only available for "
                "vanilla options with a defined option type."
            )
        return self.option_type == OptionType.PUT

    def is_european(self) -> bool:
        """
        Check if option has European exercise style.

        Returns:
            True if exercise type is EUROPEAN, False otherwise
        """
        if not hasattr(self, "exercise_type") or self.exercise_type is None:
            return False
        return self.exercise_type == ExerciseType.EUROPEAN

    def is_american(self) -> bool:
        """
        Check if option has American exercise style.

        Returns:
            True if exercise type is AMERICAN, False otherwise
        """
        if not hasattr(self, "exercise_type") or self.exercise_type is None:
            return False
        return self.exercise_type == ExerciseType.AMERICAN

    def is_bermudan(self) -> bool:
        """
        Check if option has Bermudan exercise style.

        Returns:
            True if exercise type is BERMUDAN, False otherwise
        """
        if not hasattr(self, "exercise_type") or self.exercise_type is None:
            return False
        return self.exercise_type == ExerciseType.BERMUDAN

    # ==========================================================================
    # Moneyness Utilities
    # ==========================================================================

    def moneyness(self, spot: float) -> float:
        """
        Calculate simple moneyness ratio (spot / strike).

        Args:
            spot: Current spot price

        Returns:
            Moneyness ratio S/K

        Raises:
            ValidationError: If spot is non-positive
        """
        if spot <= 0:
            raise ValidationError(f"Spot must be positive, got {spot}")
        return spot / self.strike

    def log_moneyness(self, spot: float) -> float:
        """
        Calculate log moneyness ln(spot / strike).

        Args:
            spot: Current spot price

        Returns:
            Log moneyness ln(S/K)

        Raises:
            ValidationError: If spot is non-positive
        """
        import math

        if spot <= 0:
            raise ValidationError(f"Spot must be positive, got {spot}")
        return math.log(spot / self.strike)

    def is_in_the_money(self, spot: float) -> bool:
        """
        Check if option is in the money.

        For calls: spot > strike
        For puts: spot < strike

        Args:
            spot: Current spot price

        Returns:
            True if option is in the money
        """
        if self.is_call():
            return spot > self.strike
        else:
            return spot < self.strike

    def is_out_of_the_money(self, spot: float) -> bool:
        """
        Check if option is out of the money.

        For calls: spot < strike
        For puts: spot > strike

        Args:
            spot: Current spot price

        Returns:
            True if option is out of the money
        """
        if self.is_call():
            return spot < self.strike
        else:
            return spot > self.strike

    def is_at_the_money(self, spot: float, tolerance: float = 0.001) -> bool:
        """
        Check if option is at the money within a tolerance.

        Args:
            spot: Current spot price
            tolerance: Relative tolerance for ATM check (default: 0.1%)

        Returns:
            True if |spot/strike - 1| <= tolerance
        """
        return abs(spot / self.strike - 1.0) <= tolerance

    def intrinsic_value(self, spot: float) -> float:
        """
        Calculate intrinsic value of the option.

        For calls: max(spot - strike, 0)
        For puts: max(strike - spot, 0)

        Args:
            spot: Current spot price

        Returns:
            Intrinsic value (non-negative)

        Raises:
            ValidationError: If spot is negative or option_type not set
        """
        if spot < 0:
            raise ValidationError(f"Spot must be non-negative, got {spot}")

        # Check if option_type is available
        if not hasattr(self, "option_type") or self.option_type is None:
            raise ValidationError(
                "option_type is not set. Cannot calculate intrinsic value for "
                "structured products. Use get_payoff() instead."
            )

        if self.is_call():
            intrinsic = max(spot - self.strike, 0.0)
        else:
            intrinsic = max(self.strike - spot, 0.0)

        return intrinsic * self.contract_multiplier

    def time_shift(self, time_bump: float, bumped_date: datetime, pricing_env) -> bool:
        """
        Shift option state for theta bumping.

        Returns True if all observations were dropped.
        """
        dropped_all = False
        schedule = getattr(self, "observation_schedule", None)
        if schedule is not None:
            if schedule.uses_dates():
                pricing_env.valuation_date = bumped_date
            bumped_schedule = schedule.time_shift(time_bump, bumped_date)
            if bumped_schedule is None:
                return True
            self.observation_schedule = bumped_schedule
            if hasattr(self, "observation_dates") and bumped_schedule.uses_times():
                self.observation_dates = bumped_schedule.times

        if getattr(self, "exercise_date", None) is None:
            if getattr(self, "maturity", None) is not None:
                self.maturity -= time_bump
        else:
            pricing_env.valuation_date = bumped_date

        return dropped_all

    # ==========================================================================
    # Abstract Methods
    # ==========================================================================

    @abstractmethod
    def get_payoff(self, spot: float) -> float:
        """
        Calculate the option payoff at maturity.

        This method must be implemented by subclasses to define the specific
        payoff structure of the option.

        Args:
            spot: Spot price at maturity

        Returns:
            Option payoff
        """
        pass

    # ==========================================================================
    # String Representation
    # ==========================================================================

    def _format_maturity(self) -> str:
        """Format the canonical numeric or date-based expiry representation."""
        if self.exercise_date is not None:
            return f"exp={self.exercise_date}"
        if self.maturity is not None:
            return f"T={self.maturity:.4f}"
        return "expiry=unset"

    def __repr__(self) -> str:
        """
        Return string representation of the option.

        Handles both maturity-based and date-based options safely.
        """
        return (
            f"{self.__class__.__name__}("
            f"{self.option_type}, K={self.strike:.2f}, {self._format_maturity()})"
        )
