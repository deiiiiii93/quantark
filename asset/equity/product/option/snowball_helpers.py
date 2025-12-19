"""
Factory functions for creating common Snowball option structures.

This module provides helper functions that simplify the creation of Snowball options
by providing sensible defaults for common market structures. Each helper accepts
minimal required parameters and allows full customization via **kwargs.

Available helpers:
- create_standard_snowball(): Basic snowball with flat KO barrier, continuous KI
- create_stepdown_snowball(): KO barrier decreases each observation period
- create_european_ki_snowball(): KI only observed at maturity
- create_parachute_snowball(): Last KO barrier equals KI barrier
- create_phoenix_snowball(): Periodic coupons above coupon barrier
- create_airbag_snowball(): Reduced participation below airbag barrier

Utilities:
- generate_ko_observation_dates(): Create evenly spaced observation dates
- generate_stepdown_barriers(): Generate decreasing barrier levels

Example:
    >>> from asset.equity.product.option import create_standard_snowball
    >>> snowball = create_standard_snowball(
    ...     initial_price=100.0,
    ...     strike=100.0,
    ...     maturity=1.0,
    ...     notional=1_000_000.0,
    ... )
"""

from dataclasses import fields
from typing import List, Optional

from util.exceptions import ValidationError
from util.enum import ObservationType, CouponPayType

from .snowball_option import SnowballOption
from .snowball_config import BarrierConfig, PayoffConfig, AccrualConfig, AirbagConfig


# =============================================================================
# Utility Functions
# =============================================================================


def generate_ko_observation_dates(
    maturity: float,
    frequency: str = "monthly",
    skip_first: int = 0,
) -> List[float]:
    """
    Generate evenly spaced observation dates as year fractions.

    Args:
        maturity: Time to maturity in years
        frequency: Observation frequency - "monthly", "quarterly", "weekly", or "daily"
        skip_first: Number of initial observations to skip (lock-out period)

    Returns:
        List of observation times as year fractions

    Raises:
        ValidationError: If frequency is invalid or maturity is non-positive

    Example:
        >>> generate_ko_observation_dates(1.0, "quarterly")
        [0.25, 0.5, 0.75, 1.0]
        >>> generate_ko_observation_dates(1.0, "monthly", skip_first=2)
        [0.25, 0.333..., 0.416..., ..., 1.0]
    """
    if maturity <= 0:
        raise ValidationError(f"maturity must be positive, got {maturity}")

    frequency_map = {
        "monthly": 12,
        "quarterly": 4,
        "weekly": 52,
        "daily": 252,
    }

    if frequency not in frequency_map:
        raise ValidationError(
            f"frequency must be one of {list(frequency_map.keys())}, got '{frequency}'"
        )

    num_periods = frequency_map[frequency]
    num_observations = int(maturity * num_periods)

    if num_observations < 1:
        raise ValidationError(
            f"maturity {maturity} with frequency '{frequency}' yields no observations"
        )

    # Generate dates evenly spaced up to maturity
    all_dates = [(i + 1) / num_observations * maturity for i in range(num_observations)]

    if skip_first >= len(all_dates):
        raise ValidationError(
            f"skip_first={skip_first} would skip all {len(all_dates)} observations"
        )

    return all_dates[skip_first:]


def generate_stepdown_barriers(
    initial_barrier: float,
    stepdown_amount: float,
    num_observations: int,
    min_barrier: Optional[float] = None,
) -> List[float]:
    """
    Generate decreasing barrier levels for step-down structures.

    Args:
        initial_barrier: Starting barrier level
        stepdown_amount: Amount to decrease each period (absolute value)
        num_observations: Number of observation dates
        min_barrier: Optional floor for barrier levels

    Returns:
        List of barrier levels, decreasing by stepdown_amount each period

    Raises:
        ValidationError: If parameters are invalid

    Example:
        >>> generate_stepdown_barriers(103.0, 0.5, 4)
        [103.0, 102.5, 102.0, 101.5]
        >>> generate_stepdown_barriers(103.0, 2.0, 4, min_barrier=100.0)
        [103.0, 101.0, 100.0, 100.0]
    """
    if initial_barrier <= 0:
        raise ValidationError(
            f"initial_barrier must be positive, got {initial_barrier}"
        )
    if stepdown_amount < 0:
        raise ValidationError(
            f"stepdown_amount must be non-negative, got {stepdown_amount}"
        )
    if num_observations < 1:
        raise ValidationError(
            f"num_observations must be at least 1, got {num_observations}"
        )

    barriers = []
    current = initial_barrier
    for _ in range(num_observations):
        if min_barrier is not None:
            current = max(current, min_barrier)
        barriers.append(current)
        current -= stepdown_amount

    return barriers


# =============================================================================
# Helper Functions
# =============================================================================


def _validate_core_params(
    initial_price: float,
    strike: float,
    maturity: float,
    notional: float,
    func_name: str,
) -> None:
    """Validate core parameters common to all helpers."""
    if initial_price <= 0:
        raise ValidationError(
            f"{func_name}: initial_price must be positive, got {initial_price}"
        )
    if strike <= 0:
        raise ValidationError(f"{func_name}: strike must be positive, got {strike}")
    if maturity <= 0:
        raise ValidationError(f"{func_name}: maturity must be positive, got {maturity}")
    if notional <= 0:
        raise ValidationError(f"{func_name}: notional must be positive, got {notional}")


def _extract_config_kwargs(kwargs: dict) -> tuple:
    """
    Extract kwargs for each config class using introspection.

    Raises:
        ValidationError: If kwargs contains unknown parameters.

    Returns:
        (barrier_kwargs, payoff_kwargs, accrual_kwargs, airbag_kwargs)
    """
    barrier_fields = {f.name for f in fields(BarrierConfig)}
    payoff_fields = {f.name for f in fields(PayoffConfig)}
    accrual_fields = {f.name for f in fields(AccrualConfig)}
    airbag_fields = {f.name for f in fields(AirbagConfig)}

    barrier_kwargs = {}
    payoff_kwargs = {}
    accrual_kwargs = {}
    airbag_kwargs = {}
    unknown_kwargs = []

    for key, value in kwargs.items():
        if key in barrier_fields:
            barrier_kwargs[key] = value
        elif key in payoff_fields:
            payoff_kwargs[key] = value
        elif key in accrual_fields:
            accrual_kwargs[key] = value
        elif key in airbag_fields:
            airbag_kwargs[key] = value
        else:
            unknown_kwargs.append(key)

    if unknown_kwargs:
        raise ValidationError(
            f"Unknown parameters provided: {', '.join(unknown_kwargs)}. "
            "Please check spelling or valid configuration fields."
        )

    return barrier_kwargs, payoff_kwargs, accrual_kwargs, airbag_kwargs


def create_standard_snowball(
    initial_price: float,
    strike: float,
    maturity: float,
    notional: float = 1_000_000.0,
    ko_barrier: Optional[float] = None,
    ko_rate: float = 0.15,
    ki_barrier: Optional[float] = None,
    num_observations: int = 12,
    is_reverse: bool = False,
    **kwargs,
) -> SnowballOption:
    """
    Create a standard snowball with flat KO barrier and continuous KI monitoring.

    This is the most common snowball structure with:
    - Discrete KO observations (monthly by default)
    - Continuous KI monitoring
    - Flat (constant) KO barrier
    - Annualized coupon rate

    Args:
        initial_price: Reference price for payoff calculations
        strike: Strike price for embedded option
        maturity: Time to maturity in years
        notional: Notional principal (default: 1,000,000)
        ko_barrier: Knock-out barrier (default: 103% of initial_price)
        ko_rate: Annualized knock-out rate (default: 15%)
        ki_barrier: Knock-in barrier (default: 75% of initial_price)
        num_observations: Number of KO observations (default: 12 for monthly)
        is_reverse: If True, create reverse snowball (default: False)
        **kwargs: Additional parameters passed to config objects or SnowballOption

    Returns:
        Configured SnowballOption instance

    Example:
        >>> snowball = create_standard_snowball(
        ...     initial_price=100.0,
        ...     strike=100.0,
        ...     maturity=1.0,
        ... )
        >>> snowball.barrier_config.ko_barrier
        103.0
    """
    _validate_core_params(
        initial_price, strike, maturity, notional, "create_standard_snowball"
    )

    # Apply defaults
    if ko_barrier is None:
        ko_barrier = 1.03 * initial_price
    if ki_barrier is None:
        ki_barrier = 0.75 * initial_price

    # Generate observation dates
    ko_observation_dates = [
        (i + 1) / num_observations * maturity for i in range(num_observations)
    ]

    # Extract config-specific kwargs
    barrier_kwargs, payoff_kwargs, accrual_kwargs, airbag_kwargs = _extract_config_kwargs(kwargs)

    # Build configs
    barrier_config = BarrierConfig(
        ko_barrier=ko_barrier,
        ko_rate=ko_rate,
        ko_observation_type=barrier_kwargs.pop(
            "ko_observation_type", ObservationType.DISCRETE
        ),
        ko_observation_dates=barrier_kwargs.pop(
            "ko_observation_dates", ko_observation_dates
        ),
        ki_barrier=ki_barrier,
        ki_observation_type=barrier_kwargs.pop(
            "ki_observation_type", ObservationType.CONTINUOUS
        ),
        ki_continuous=barrier_kwargs.pop("ki_continuous", True),
        **barrier_kwargs,
    )

    payoff_config = PayoffConfig(
        rebate_rate=payoff_kwargs.pop("rebate_rate", ko_rate),
        include_principal=payoff_kwargs.pop("include_principal", False),
        **payoff_kwargs,
    )

    accrual_config = AccrualConfig(
        coupon_pay_type=accrual_kwargs.pop("coupon_pay_type", CouponPayType.INSTANT),
        is_annualized=accrual_kwargs.pop("is_annualized", True),
        **accrual_kwargs,
    )

    airbag_config = AirbagConfig(**airbag_kwargs)

    return SnowballOption(
        initial_price=initial_price,
        strike=strike,
        maturity=maturity,
        notional=notional,
        barrier_config=barrier_config,
        payoff_config=payoff_config,
        accrual_config=accrual_config,
        airbag_config=airbag_config,
        is_reverse=is_reverse,
    )


def create_stepdown_snowball(
    initial_price: float,
    strike: float,
    maturity: float,
    notional: float = 1_000_000.0,
    initial_ko_barrier: Optional[float] = None,
    stepdown_rate: float = 0.005,
    ko_rate: float = 0.15,
    ki_barrier: Optional[float] = None,
    num_observations: int = 12,
    is_reverse: bool = False,
    **kwargs,
) -> SnowballOption:
    """
    Create a step-down snowball where KO barrier decreases each observation.

    Common in China structured products market (递减雪球). The KO barrier
    starts high and decreases each observation period, making knock-out
    progressively easier to achieve.

    Args:
        initial_price: Reference price for payoff calculations
        strike: Strike price for embedded option
        maturity: Time to maturity in years
        notional: Notional principal (default: 1,000,000)
        initial_ko_barrier: Starting KO barrier (default: 103% of initial_price)
        stepdown_rate: Rate of barrier decrease per period as fraction of initial_price
                      (default: 0.5% per period)
        ko_rate: Annualized knock-out rate (default: 15%)
        ki_barrier: Knock-in barrier (default: 75% of initial_price)
        num_observations: Number of KO observations (default: 12 for monthly)
        is_reverse: If True, create reverse snowball (default: False)
        **kwargs: Additional parameters passed to config objects or SnowballOption

    Returns:
        Configured SnowballOption instance with decreasing KO barriers

    Example:
        >>> snowball = create_stepdown_snowball(
        ...     initial_price=100.0,
        ...     strike=100.0,
        ...     maturity=1.0,
        ...     stepdown_rate=0.005,
        ... )
        >>> snowball.barrier_config.ko_barrier[:4]
        [103.0, 102.5, 102.0, 101.5]
    """
    _validate_core_params(
        initial_price, strike, maturity, notional, "create_stepdown_snowball"
    )

    # Apply defaults
    if initial_ko_barrier is None:
        initial_ko_barrier = 1.03 * initial_price
    if ki_barrier is None:
        ki_barrier = 0.75 * initial_price

    # Generate step-down barriers
    stepdown_amount = stepdown_rate * initial_price
    ko_barriers = generate_stepdown_barriers(
        initial_barrier=initial_ko_barrier,
        stepdown_amount=stepdown_amount,
        num_observations=num_observations,
    )

    # Generate observation dates
    ko_observation_dates = [
        (i + 1) / num_observations * maturity for i in range(num_observations)
    ]

    # Extract config-specific kwargs
    barrier_kwargs, payoff_kwargs, accrual_kwargs, airbag_kwargs = _extract_config_kwargs(kwargs)

    # Build configs
    barrier_config = BarrierConfig(
        ko_barrier=ko_barriers,
        ko_rate=ko_rate,
        ko_observation_type=barrier_kwargs.pop(
            "ko_observation_type", ObservationType.DISCRETE
        ),
        ko_observation_dates=barrier_kwargs.pop(
            "ko_observation_dates", ko_observation_dates
        ),
        ki_barrier=ki_barrier,
        ki_observation_type=barrier_kwargs.pop(
            "ki_observation_type", ObservationType.CONTINUOUS
        ),
        ki_continuous=barrier_kwargs.pop("ki_continuous", True),
        **barrier_kwargs,
    )

    payoff_config = PayoffConfig(
        rebate_rate=payoff_kwargs.pop("rebate_rate", ko_rate),
        include_principal=payoff_kwargs.pop("include_principal", False),
        **payoff_kwargs,
    )

    accrual_config = AccrualConfig(
        coupon_pay_type=accrual_kwargs.pop("coupon_pay_type", CouponPayType.INSTANT),
        is_annualized=accrual_kwargs.pop("is_annualized", True),
        **accrual_kwargs,
    )

    airbag_config = AirbagConfig(**airbag_kwargs)

    return SnowballOption(
        initial_price=initial_price,
        strike=strike,
        maturity=maturity,
        notional=notional,
        barrier_config=barrier_config,
        payoff_config=payoff_config,
        accrual_config=accrual_config,
        airbag_config=airbag_config,
        is_reverse=is_reverse,
    )


def create_european_ki_snowball(
    initial_price: float,
    strike: float,
    maturity: float,
    notional: float = 1_000_000.0,
    ko_barrier: Optional[float] = None,
    ko_rate: float = 0.15,
    ki_barrier: Optional[float] = None,
    num_ko_observations: int = 12,
    is_reverse: bool = False,
    **kwargs,
) -> SnowballOption:
    """
    Create a snowball with European-style KI (only observed at maturity).

    Unlike standard snowballs with continuous KI monitoring, this structure
    only checks the KI barrier at maturity. This increases the probability
    of the V0 outcome (no KO, no KI).

    Args:
        initial_price: Reference price for payoff calculations
        strike: Strike price for embedded option
        maturity: Time to maturity in years
        notional: Notional principal (default: 1,000,000)
        ko_barrier: Knock-out barrier (default: 103% of initial_price)
        ko_rate: Annualized knock-out rate (default: 15%)
        ki_barrier: Knock-in barrier (default: 75% of initial_price)
        num_ko_observations: Number of KO observations (default: 12 for monthly)
        is_reverse: If True, create reverse snowball (default: False)
        **kwargs: Additional parameters passed to config objects or SnowballOption

    Returns:
        Configured SnowballOption instance with European-style KI

    Example:
        >>> snowball = create_european_ki_snowball(
        ...     initial_price=100.0,
        ...     strike=100.0,
        ...     maturity=1.0,
        ... )
        >>> snowball.barrier_config.ki_continuous
        False
        >>> snowball.barrier_config.ki_observation_dates
        [1.0]
    """
    _validate_core_params(
        initial_price, strike, maturity, notional, "create_european_ki_snowball"
    )

    # Apply defaults
    if ko_barrier is None:
        ko_barrier = 1.03 * initial_price
    if ki_barrier is None:
        ki_barrier = 0.75 * initial_price

    # Generate KO observation dates
    ko_observation_dates = [
        (i + 1) / num_ko_observations * maturity for i in range(num_ko_observations)
    ]

    # KI only at maturity (European-style)
    ki_observation_dates = [maturity]

    # Extract config-specific kwargs
    barrier_kwargs, payoff_kwargs, accrual_kwargs, airbag_kwargs = _extract_config_kwargs(kwargs)

    # Build configs - force discrete KI with single observation at maturity
    barrier_config = BarrierConfig(
        ko_barrier=ko_barrier,
        ko_rate=ko_rate,
        ko_observation_type=barrier_kwargs.pop(
            "ko_observation_type", ObservationType.DISCRETE
        ),
        ko_observation_dates=barrier_kwargs.pop(
            "ko_observation_dates", ko_observation_dates
        ),
        ki_barrier=ki_barrier,
        ki_observation_type=ObservationType.DISCRETE,  # Force discrete
        ki_observation_dates=ki_observation_dates,
        ki_continuous=False,  # Force non-continuous
        **barrier_kwargs,
    )

    payoff_config = PayoffConfig(
        rebate_rate=payoff_kwargs.pop("rebate_rate", ko_rate),
        include_principal=payoff_kwargs.pop("include_principal", False),
        **payoff_kwargs,
    )

    accrual_config = AccrualConfig(
        coupon_pay_type=accrual_kwargs.pop("coupon_pay_type", CouponPayType.INSTANT),
        is_annualized=accrual_kwargs.pop("is_annualized", True),
        **accrual_kwargs,
    )

    airbag_config = AirbagConfig(**airbag_kwargs)

    return SnowballOption(
        initial_price=initial_price,
        strike=strike,
        maturity=maturity,
        notional=notional,
        barrier_config=barrier_config,
        payoff_config=payoff_config,
        accrual_config=accrual_config,
        airbag_config=airbag_config,
        is_reverse=is_reverse,
    )


def create_parachute_snowball(
    initial_price: float,
    strike: float,
    maturity: float,
    notional: float = 1_000_000.0,
    ko_barrier: Optional[float] = None,
    ko_rate: float = 0.15,
    ki_barrier: Optional[float] = None,
    num_observations: int = 12,
    is_reverse: bool = False,
    **kwargs,
) -> SnowballOption:
    """
    Create a parachute snowball where last KO barrier equals KI barrier.

    Also known as 降落伞雪球 in China market. At the final observation,
    the KO barrier drops to the KI level, guaranteeing an exit (knock-out)
    if the product hasn't been knocked in. This provides a "parachute"
    safety net for investors.

    Args:
        initial_price: Reference price for payoff calculations
        strike: Strike price for embedded option
        maturity: Time to maturity in years
        notional: Notional principal (default: 1,000,000)
        ko_barrier: Knock-out barrier for early observations (default: 103% of initial_price)
        ko_rate: Annualized knock-out rate (default: 15%)
        ki_barrier: Knock-in barrier, also final KO barrier (default: 75% of initial_price)
        num_observations: Number of KO observations (default: 12 for monthly)
        is_reverse: If True, create reverse snowball (default: False)
        **kwargs: Additional parameters passed to config objects or SnowballOption

    Returns:
        Configured SnowballOption instance with parachute structure

    Example:
        >>> snowball = create_parachute_snowball(
        ...     initial_price=100.0,
        ...     strike=100.0,
        ...     maturity=1.0,
        ... )
        >>> barriers = snowball.barrier_config.ko_barrier
        >>> barriers[-1] == snowball.barrier_config.ki_barrier
        True
    """
    _validate_core_params(
        initial_price, strike, maturity, notional, "create_parachute_snowball"
    )

    # Apply defaults
    if ko_barrier is None:
        ko_barrier = 1.03 * initial_price
    if ki_barrier is None:
        ki_barrier = 0.75 * initial_price

    # Build parachute KO barriers: flat until last, then drop to KI level
    ko_barriers = [ko_barrier] * (num_observations - 1) + [ki_barrier]

    # Generate observation dates
    ko_observation_dates = [
        (i + 1) / num_observations * maturity for i in range(num_observations)
    ]

    # Extract config-specific kwargs
    barrier_kwargs, payoff_kwargs, accrual_kwargs, airbag_kwargs = _extract_config_kwargs(kwargs)

    # Build configs
    barrier_config = BarrierConfig(
        ko_barrier=ko_barriers,
        ko_rate=ko_rate,
        ko_observation_type=barrier_kwargs.pop(
            "ko_observation_type", ObservationType.DISCRETE
        ),
        ko_observation_dates=barrier_kwargs.pop(
            "ko_observation_dates", ko_observation_dates
        ),
        ki_barrier=ki_barrier,
        ki_observation_type=barrier_kwargs.pop(
            "ki_observation_type", ObservationType.CONTINUOUS
        ),
        ki_continuous=barrier_kwargs.pop("ki_continuous", True),
        **barrier_kwargs,
    )

    payoff_config = PayoffConfig(
        rebate_rate=payoff_kwargs.pop("rebate_rate", ko_rate),
        include_principal=payoff_kwargs.pop("include_principal", False),
        **payoff_kwargs,
    )

    accrual_config = AccrualConfig(
        coupon_pay_type=accrual_kwargs.pop("coupon_pay_type", CouponPayType.INSTANT),
        is_annualized=accrual_kwargs.pop("is_annualized", True),
        **accrual_kwargs,
    )

    airbag_config = AirbagConfig(**airbag_kwargs)

    return SnowballOption(
        initial_price=initial_price,
        strike=strike,
        maturity=maturity,
        notional=notional,
        barrier_config=barrier_config,
        payoff_config=payoff_config,
        accrual_config=accrual_config,
        airbag_config=airbag_config,
        is_reverse=is_reverse,
    )





def create_airbag_snowball(
    initial_price: float,
    strike: float,
    maturity: float,
    notional: float = 1_000_000.0,
    ko_barrier: Optional[float] = None,
    ko_rate: float = 0.15,
    ki_barrier: Optional[float] = None,
    airbag_barrier: Optional[float] = None,
    participation_rate: float = 1.0,
    airbag_participation_rate: float = 0.5,
    airbag_strike: Optional[float] = None,
    num_observations: int = 12,
    is_reverse: bool = False,
    **kwargs,
) -> SnowballOption:
    """
    Create an airbag snowball with reduced participation below airbag barrier.

    Airbag snowballs provide additional protection for extreme downside
    by reducing the participation rate when spot falls below the airbag
    barrier. This limits losses in severe market downturns.

    Args:
        initial_price: Reference price for payoff calculations
        strike: Strike price for embedded option
        maturity: Time to maturity in years
        notional: Notional principal (default: 1,000,000)
        ko_barrier: Knock-out barrier (default: 103% of initial_price)
        ko_rate: Annualized knock-out rate (default: 15%)
        ki_barrier: Knock-in barrier (default: 75% of initial_price)
        airbag_barrier: Barrier below which participation is reduced
                       (default: 60% of initial_price)
        participation_rate: Participation rate for normal KI scenario (default: 100%)
        airbag_participation_rate: Participation rate when below airbag barrier (default: 50%)
        airbag_strike: Strike price for airbag payoff (optional, defaults to strike)
        num_observations: Number of observations (default: 12 for monthly)
        is_reverse: If True, create reverse snowball (default: False)
        **kwargs: Additional parameters passed to config objects or SnowballOption

    Returns:
        Configured SnowballOption instance with airbag structure

    Example:
        >>> snowball = create_airbag_snowball(
        ...     initial_price=100.0,
        ...     strike=100.0,
        ...     maturity=1.0,
        ...     participation_rate=1.0,
        ...     airbag_participation_rate=0.5,
        ... )
        >>> snowball.airbag_config.airbag_participation_rate
        0.5
    """
    _validate_core_params(
        initial_price, strike, maturity, notional, "create_airbag_snowball"
    )

    # Apply defaults
    if ko_barrier is None:
        ko_barrier = 1.03 * initial_price
    if ki_barrier is None:
        ki_barrier = 0.75 * initial_price
    if airbag_barrier is None:
        airbag_barrier = 0.60 * initial_price

    # Validate airbag barrier
    if airbag_barrier >= ki_barrier:
        raise ValidationError(
            f"create_airbag_snowball: airbag_barrier ({airbag_barrier}) "
            f"must be less than ki_barrier ({ki_barrier})"
        )

    # Generate observation dates
    ko_observation_dates = [
        (i + 1) / num_observations * maturity for i in range(num_observations)
    ]

    # Extract config-specific kwargs
    barrier_kwargs, payoff_kwargs, accrual_kwargs, airbag_kwargs = (
        _extract_config_kwargs(kwargs)
    )

    # Use airbag params in config
    airbag_kwargs["airbag_barrier"] = airbag_barrier
    airbag_kwargs["airbag_participation_rate"] = airbag_participation_rate
    airbag_kwargs["airbag_strike"] = airbag_strike

    # Build configs
    barrier_config = BarrierConfig(
        ko_barrier=ko_barrier,
        ko_rate=ko_rate,
        ko_observation_type=barrier_kwargs.pop(
            "ko_observation_type", ObservationType.DISCRETE
        ),
        ko_observation_dates=barrier_kwargs.pop(
            "ko_observation_dates", ko_observation_dates
        ),
        ki_barrier=ki_barrier,
        ki_observation_type=barrier_kwargs.pop(
            "ki_observation_type", ObservationType.CONTINUOUS
        ),
        ki_continuous=barrier_kwargs.pop("ki_continuous", True),
        **barrier_kwargs,
    )

    payoff_config = PayoffConfig(
        rebate_rate=payoff_kwargs.pop("rebate_rate", ko_rate),
        include_principal=payoff_kwargs.pop("include_principal", False),
        participation_rate=participation_rate,
        **payoff_kwargs,
    )

    accrual_config = AccrualConfig(
        coupon_pay_type=accrual_kwargs.pop("coupon_pay_type", CouponPayType.INSTANT),
        is_annualized=accrual_kwargs.pop("is_annualized", True),
        **accrual_kwargs,
    )

    airbag_config = AirbagConfig(**airbag_kwargs)

    return SnowballOption(
        initial_price=initial_price,
        strike=strike,
        maturity=maturity,
        notional=notional,
        barrier_config=barrier_config,
        payoff_config=payoff_config,
        accrual_config=accrual_config,
        airbag_config=airbag_config,
        is_reverse=is_reverse,
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Helper functions
    "create_standard_snowball",
    "create_stepdown_snowball",
    "create_european_ki_snowball",
    "create_parachute_snowball",
    "create_airbag_snowball",
    # Utility functions
    "generate_ko_observation_dates",
    "generate_stepdown_barriers",
]
