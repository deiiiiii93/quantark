"""
Barrier shift utility for discretely observed barrier options.
"""

import math
from util.exceptions import ValidationError


def apply_barrier_shift(
    barrier: float,
    is_up_barrier: bool,
    volatility: float,
    observation_interval: float,
    beta: float = 0.5825971579,
) -> float:
    """
    Apply Broadie-Glasserman-Kou style barrier shift for discrete monitoring.

    Args:
        barrier: Original barrier level.
        is_up_barrier: True if barrier is an up barrier, False if down.
        volatility: Volatility used for shifting.
        observation_interval: Observation spacing in years (dt).
        beta: Empirical constant for barrier shift (default from literature).

    Returns:
        Shifted barrier level.

    Raises:
        ValidationError: If inputs are invalid.
    """
    if barrier <= 0:
        raise ValidationError(f"Barrier must be positive, got {barrier}")
    if volatility <= 0:
        raise ValidationError(f"Volatility must be positive, got {volatility}")
    if observation_interval <= 0:
        raise ValidationError(
            f"Observation interval must be positive, got {observation_interval}"
        )

    shift = beta * volatility * math.sqrt(observation_interval)
    if is_up_barrier:
        return barrier * math.exp(shift)
    return barrier * math.exp(-shift)
