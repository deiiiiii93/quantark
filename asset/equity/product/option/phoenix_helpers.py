"""
Factory functions for creating common Phoenix option structures.

This module provides helper functions that simplify the creation of Phoenix options
by providing sensible defaults for common market structures. Each helper accepts
minimal required parameters and allows full customization via **kwargs.

Available helpers:
- create_standard_phoenix(): Basic phoenix with flat barriers, memory coupon
- create_stepdown_phoenix(): KO and coupon barriers decrease each observation period
- create_reverse_phoenix(): Reverse direction (down KO, up KI)
- create_memory_phoenix(): Explicit memory coupon enabled
- create_non_memory_phoenix(): Memory coupon disabled

Utilities (reused from snowball_helpers):
- generate_ko_observation_dates(): Create evenly spaced observation dates
- generate_stepdown_barriers(): Generate decreasing barrier levels

Example:
    >>> from asset.equity.product.option import create_standard_phoenix
    >>> phoenix = create_standard_phoenix(
    ...     initial_price=100.0,
    ...     strike=100.0,
    ...     maturity=1.0,
    ...     contract_multiplier=1.0,
    ... )
"""

from dataclasses import fields
from typing import List, Optional

from util.calendar.day_counter import DayCountConvention
from util.enum import CouponPayType, ObservationType
from util.exceptions import ValidationError

from .phoenix_config import CouponBarrierConfig
from .phoenix_option import PhoenixOption
from .snowball_config import AccrualConfig, AirbagConfig, BarrierConfig, PayoffConfig
from .snowball_helpers import (
    generate_ko_observation_dates,
    generate_stepdown_barriers,
)


# =============================================================================
# Internal Helper Functions
# =============================================================================


def _validate_core_params(
    initial_price: float,
    strike: float,
    maturity: float,
    contract_multiplier: float,
    func_name: str,
    num_observations: Optional[int] = None,
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
    if contract_multiplier <= 0:
        raise ValidationError(
            f"{func_name}: contract_multiplier must be positive, got {contract_multiplier}"
        )
    if num_observations is not None:
        if not isinstance(num_observations, int):
            raise ValidationError(
                f"{func_name}: num_observations must be an integer, got {type(num_observations)}"
            )
        if num_observations < 1:
            raise ValidationError(
                f"{func_name}: num_observations must be at least 1, got {num_observations}"
            )


def _extract_config_kwargs(kwargs: dict) -> tuple:
    """
    Extract kwargs for each config class using introspection.

    Returns:
        (barrier_kwargs, coupon_kwargs, payoff_kwargs, accrual_kwargs, airbag_kwargs)
    """
    barrier_fields = {f.name for f in fields(BarrierConfig)}
    coupon_fields = {f.name for f in fields(CouponBarrierConfig)}
    payoff_fields = {f.name for f in fields(PayoffConfig)}
    accrual_fields = {f.name for f in fields(AccrualConfig)}
    airbag_fields = {f.name for f in fields(AirbagConfig)}

    barrier_kwargs = {}
    coupon_kwargs = {}
    payoff_kwargs = {}
    accrual_kwargs = {}
    airbag_kwargs = {}
    unknown_kwargs = []

    for key, value in kwargs.items():
        if key in barrier_fields:
            barrier_kwargs[key] = value
        elif key in coupon_fields:
            coupon_kwargs[key] = value
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

    return barrier_kwargs, coupon_kwargs, payoff_kwargs, accrual_kwargs, airbag_kwargs


# =============================================================================
# Factory Functions
# =============================================================================


def create_standard_phoenix(
    initial_price: float,
    strike: float,
    maturity: float,
    contract_multiplier: float = 1.0,
    ko_barrier: Optional[float] = None,
    ko_rate: float = 0.0,
    ki_barrier: Optional[float] = None,
    coupon_barrier: Optional[float] = None,
    coupon_rate: float = 0.01,
    num_observations: int = 12,
    memory_coupon: bool = True,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365,
    coupon_pay_type: CouponPayType = CouponPayType.INSTANT,
    is_reverse: bool = False,
    **kwargs,
) -> PhoenixOption:
    """
    Create a standard phoenix with flat barriers and memory coupon.

    This is the most common phoenix structure with:
    - Discrete KO observations (monthly by default)
    - Continuous KI monitoring
    - Flat (constant) KO and coupon barriers
    - Memory coupon enabled by default
    - Day count convention for coupon calculation

    Args:
        initial_price: Reference price for payoff calculations
        strike: Strike price for embedded option
        maturity: Time to maturity in years
        contract_multiplier: Underlying units represented by one contract
        ko_barrier: Knock-out barrier (default: 103% of initial_price)
        ko_rate: Annualized knock-out rate (default: 0%)
        ki_barrier: Knock-in barrier (default: 75% of initial_price)
        coupon_barrier: Coupon barrier (default: 85% of initial_price)
        coupon_rate: Per-period coupon rate (default: 1%)
        num_observations: Number of observations (default: 12 for monthly)
        memory_coupon: If True, accumulate missed coupons (default: True)
        day_count_convention: Day count for coupon calculation (default: ACT/365)
        coupon_pay_type: INSTANT or EXPIRY (default: INSTANT)
        is_reverse: If True, create reverse phoenix (default: False)
        **kwargs: Additional parameters passed to config objects

    Returns:
        Configured PhoenixOption instance

    Example:
        >>> phoenix = create_standard_phoenix(
        ...     initial_price=100.0,
        ...     strike=100.0,
        ...     maturity=1.0,
        ... )
        >>> phoenix.coupon_config.coupon_barrier
        85.0
    """
    _validate_core_params(
        initial_price,
        strike,
        maturity,
        contract_multiplier,
        "create_standard_phoenix",
        num_observations
    )

    # Apply defaults
    if ko_barrier is None:
        ko_barrier = 1.03 * initial_price
    if ki_barrier is None:
        ki_barrier = 0.75 * initial_price
    if coupon_barrier is None:
        coupon_barrier = 0.85 * initial_price

    # Generate observation dates
    ko_observation_dates = [
        (i + 1) / num_observations * maturity for i in range(num_observations)
    ]

    # Extract config-specific kwargs
    (
        barrier_kwargs,
        coupon_kwargs,
        payoff_kwargs,
        accrual_kwargs,
        airbag_kwargs,
    ) = _extract_config_kwargs(kwargs)

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

    coupon_config = CouponBarrierConfig(
        coupon_barrier=coupon_kwargs.pop("coupon_barrier", coupon_barrier),
        coupon_rate=coupon_kwargs.pop("coupon_rate", coupon_rate),
        coupon_pay_type=coupon_kwargs.pop("coupon_pay_type", coupon_pay_type),
        day_count_convention=coupon_kwargs.pop(
            "day_count_convention", day_count_convention
        ),
        memory_coupon=coupon_kwargs.pop("memory_coupon", memory_coupon),
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

    return PhoenixOption(
        initial_price=initial_price,
        strike=strike,
        maturity=maturity,
        contract_multiplier=contract_multiplier,
        barrier_config=barrier_config,
        coupon_config=coupon_config,
        payoff_config=payoff_config,
        accrual_config=accrual_config,
        airbag_config=airbag_config,
        is_reverse=is_reverse,
    )


def create_stepdown_phoenix(
    initial_price: float,
    strike: float,
    maturity: float,
    contract_multiplier: float = 1.0,
    initial_ko_barrier: Optional[float] = None,
    initial_coupon_barrier: Optional[float] = None,
    ko_stepdown_rate: float = 0.005,
    coupon_stepdown_rate: float = 0.005,
    ko_rate: float = 0.0,
    ki_barrier: Optional[float] = None,
    coupon_rate: float = 0.01,
    num_observations: int = 12,
    memory_coupon: bool = True,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365,
    coupon_pay_type: CouponPayType = CouponPayType.INSTANT,
    is_reverse: bool = False,
    **kwargs,
) -> PhoenixOption:
    """
    Create a step-down phoenix where KO and coupon barriers decrease each observation.

    The barriers start high and decrease each observation period, making
    knock-out and coupon payments progressively easier to achieve.

    Args:
        initial_price: Reference price for payoff calculations
        strike: Strike price for embedded option
        maturity: Time to maturity in years
        contract_multiplier: Underlying units represented by one contract
        initial_ko_barrier: Starting KO barrier (default: 103% of initial_price)
        initial_coupon_barrier: Starting coupon barrier (default: 85% of initial_price)
        ko_stepdown_rate: KO barrier decrease per period as % of initial_price (default: 0.5%)
        coupon_stepdown_rate: Coupon barrier decrease per period (default: 0.5%)
        ko_rate: Annualized knock-out rate (default: 0%)
        ki_barrier: Knock-in barrier (default: 75% of initial_price)
        coupon_rate: Per-period coupon rate (default: 1%)
        num_observations: Number of observations (default: 12)
        memory_coupon: If True, accumulate missed coupons (default: True)
        day_count_convention: Day count for coupon calculation (default: ACT/365)
        coupon_pay_type: INSTANT or EXPIRY (default: INSTANT)
        is_reverse: If True, create reverse phoenix (default: False)
        **kwargs: Additional parameters passed to config objects

    Returns:
        Configured PhoenixOption instance with step-down barriers

    Example:
        >>> phoenix = create_stepdown_phoenix(
        ...     initial_price=100.0,
        ...     strike=100.0,
        ...     maturity=1.0,
        ...     ko_stepdown_rate=0.01,
        ... )
        >>> phoenix.barrier_config.ko_barrier  # List of decreasing barriers
        [103.0, 102.0, 101.0, ...]
    """
    _validate_core_params(
        initial_price,
        strike,
        maturity,
        contract_multiplier,
        "create_stepdown_phoenix",
        num_observations
    )

    # Apply defaults
    if initial_ko_barrier is None:
        initial_ko_barrier = 1.03 * initial_price
    if initial_coupon_barrier is None:
        initial_coupon_barrier = 0.85 * initial_price
    if ki_barrier is None:
        ki_barrier = 0.75 * initial_price

    # Generate step-down barriers
    ko_barriers = generate_stepdown_barriers(
        initial_barrier=initial_ko_barrier,
        stepdown_amount=ko_stepdown_rate * initial_price,
        num_observations=num_observations,
        min_barrier=ki_barrier,  # Don't go below KI barrier
    )

    coupon_barriers = generate_stepdown_barriers(
        initial_barrier=initial_coupon_barrier,
        stepdown_amount=coupon_stepdown_rate * initial_price,
        num_observations=num_observations,
        min_barrier=ki_barrier,  # Don't go below KI barrier
    )

    # Generate observation dates
    ko_observation_dates = [
        (i + 1) / num_observations * maturity for i in range(num_observations)
    ]

    # Extract config-specific kwargs
    (
        barrier_kwargs,
        coupon_kwargs,
        payoff_kwargs,
        accrual_kwargs,
        airbag_kwargs,
    ) = _extract_config_kwargs(kwargs)

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

    coupon_config = CouponBarrierConfig(
        coupon_barrier=coupon_barriers,
        coupon_rate=coupon_kwargs.pop("coupon_rate", coupon_rate),
        coupon_pay_type=coupon_kwargs.pop("coupon_pay_type", coupon_pay_type),
        day_count_convention=coupon_kwargs.pop(
            "day_count_convention", day_count_convention
        ),
        memory_coupon=coupon_kwargs.pop("memory_coupon", memory_coupon),
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

    return PhoenixOption(
        initial_price=initial_price,
        strike=strike,
        maturity=maturity,
        contract_multiplier=contract_multiplier,
        barrier_config=barrier_config,
        coupon_config=coupon_config,
        payoff_config=payoff_config,
        accrual_config=accrual_config,
        airbag_config=airbag_config,
        is_reverse=is_reverse,
    )


def create_reverse_phoenix(
    initial_price: float,
    strike: float,
    maturity: float,
    contract_multiplier: float = 1.0,
    ko_barrier: Optional[float] = None,
    ko_rate: float = 0.0,
    ki_barrier: Optional[float] = None,
    coupon_barrier: Optional[float] = None,
    coupon_rate: float = 0.01,
    num_observations: int = 12,
    memory_coupon: bool = True,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365,
    coupon_pay_type: CouponPayType = CouponPayType.INSTANT,
    **kwargs,
) -> PhoenixOption:
    """
    Create a reverse phoenix with down KO barrier and up KI barrier.

    Reverse phoenix:
    - KO barrier: DOWN (below initial price) - KO triggers when spot <= ko_barrier
    - KI barrier: UP (above initial price) - KI triggers when spot >= ki_barrier
    - Coupon barrier: UP - Coupon pays when spot <= coupon_barrier
    - Embedded option: CALL (investor is short call on KI)

    Args:
        initial_price: Reference price for payoff calculations
        strike: Strike price for embedded option
        maturity: Time to maturity in years
        contract_multiplier: Underlying units represented by one contract
        ko_barrier: Knock-out barrier (default: 97% of initial_price)
        ko_rate: Annualized knock-out rate (default: 0%)
        ki_barrier: Knock-in barrier (default: 125% of initial_price)
        coupon_barrier: Coupon barrier (default: 115% of initial_price)
        coupon_rate: Per-period coupon rate (default: 1%)
        num_observations: Number of observations (default: 12)
        memory_coupon: If True, accumulate missed coupons (default: True)
        day_count_convention: Day count for coupon calculation (default: ACT/365)
        coupon_pay_type: INSTANT or EXPIRY (default: INSTANT)
        **kwargs: Additional parameters passed to config objects

    Returns:
        Configured PhoenixOption instance with reverse direction
    """
    _validate_core_params(
        initial_price,
        strike,
        maturity,
        contract_multiplier,
        "create_reverse_phoenix",
        num_observations
    )

    # Reverse phoenix defaults
    if ko_barrier is None:
        ko_barrier = 0.97 * initial_price
    if ki_barrier is None:
        ki_barrier = 1.25 * initial_price
    if coupon_barrier is None:
        coupon_barrier = 1.15 * initial_price

    return create_standard_phoenix(
        initial_price=initial_price,
        strike=strike,
        maturity=maturity,
        contract_multiplier=contract_multiplier,
        ko_barrier=ko_barrier,
        ko_rate=ko_rate,
        ki_barrier=ki_barrier,
        coupon_barrier=coupon_barrier,
        coupon_rate=coupon_rate,
        num_observations=num_observations,
        memory_coupon=memory_coupon,
        day_count_convention=day_count_convention,
        coupon_pay_type=coupon_pay_type,
        is_reverse=True,
        **kwargs,
    )


def create_memory_phoenix(
    initial_price: float,
    strike: float,
    maturity: float,
    contract_multiplier: float = 1.0,
    **kwargs,
) -> PhoenixOption:
    """
    Create a phoenix with memory coupon explicitly enabled.

    Memory coupon: When coupon barrier is not hit at an observation, the
    coupon is accumulated and paid when the barrier is hit at a later
    observation (or at maturity if never knocked out).

    Args:
        initial_price: Reference price for payoff calculations
        strike: Strike price for embedded option
        maturity: Time to maturity in years
        contract_multiplier: Underlying units represented by one contract
        **kwargs: Additional parameters passed to create_standard_phoenix

    Returns:
        PhoenixOption with memory_coupon=True
    """
    return create_standard_phoenix(
        initial_price=initial_price,
        strike=strike,
        maturity=maturity,
        contract_multiplier=contract_multiplier,
        memory_coupon=True,
        **kwargs,
    )


def create_non_memory_phoenix(
    initial_price: float,
    strike: float,
    maturity: float,
    contract_multiplier: float = 1.0,
    **kwargs,
) -> PhoenixOption:
    """
    Create a phoenix with memory coupon disabled.

    Non-memory coupon: Each observation is independent. If coupon barrier
    is not hit, that period's coupon is forfeited.

    Args:
        initial_price: Reference price for payoff calculations
        strike: Strike price for embedded option
        maturity: Time to maturity in years
        contract_multiplier: Underlying units represented by one contract
        **kwargs: Additional parameters passed to create_standard_phoenix

    Returns:
        PhoenixOption with memory_coupon=False
    """
    return create_standard_phoenix(
        initial_price=initial_price,
        strike=strike,
        maturity=maturity,
        contract_multiplier=contract_multiplier,
        memory_coupon=False,
        **kwargs,
    )
