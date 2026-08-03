"""Exact one-factor conditioning for standard Snowball RQMC payoffs.

The Heston QE/QE-M path generator can expose each contractual log-spot as

``log(S_t) = log(base_path_t) + loading_t * Z,  Z ~ N(0, 1)``.

Conditional on the variance path and the remaining Brownian-bridge factors,
all discrete KO/KI events are therefore intervals on the same scalar ``Z``.
This module integrates those intervals, and the terminal short-put payoff,
analytically.  It is deliberately fail-closed for product features whose
payoff is not represented by that interval decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import log_ndtr, ndtr

from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, ProtectionType
from quantark.util.exceptions import PricingError, ValidationError


@dataclass(frozen=True)
class ConditionalSnowballMoments:
    """Per-outer-path conditional moments used by RQMC and diagnostics."""

    discounted_payoff: np.ndarray
    ko_probability: np.ndarray
    v0_probability: np.ndarray
    v1_probability: np.ndarray
    ko_time_numerator: np.ndarray


def _normal_interval_probability(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Stable ``P(lower <= Z < upper)`` for standard normal ``Z``."""

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    probability = np.where(
        lower > 0.0,
        ndtr(-lower) - ndtr(-upper),
        ndtr(upper) - ndtr(lower),
    )
    return np.where(upper > lower, np.clip(probability, 0.0, 1.0), 0.0)


def _truncated_lognormal_first_moment(
    base_spot: np.ndarray,
    loading: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Return ``E[base_spot * exp(loading*Z) 1{Z<=upper}]`` stably."""

    log_value = (
        np.log(np.asarray(base_spot, dtype=float))
        + 0.5 * np.square(np.asarray(loading, dtype=float))
        + log_ndtr(np.asarray(upper, dtype=float) - loading)
    )
    result = np.exp(log_value)
    if not np.all(np.isfinite(result)):
        raise PricingError("conditional lognormal moment is non-finite")
    return result


def _validate_supported_product(product: SnowballOption) -> None:
    if not isinstance(product, SnowballOption):
        raise ValidationError("affine spot conditioning requires SnowballOption")
    if product.is_reverse:
        raise ValidationError(
            "affine spot conditioning currently supports standard Snowballs only"
        )
    if product.barrier_config.ko_observation_type != ObservationType.DISCRETE:
        raise ValidationError(
            "affine spot conditioning requires discrete KO monitoring"
        )
    if product.barrier_config.ki_continuous or (
        product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
    ):
        raise ValidationError(
            "affine spot conditioning requires discrete KI monitoring"
        )
    if product.barrier_config.disable_ko_after_ki:
        raise ValidationError(
            "affine spot conditioning does not support disable_ko_after_ki"
        )
    if product.payoff_config.call_rebate_enabled:
        raise ValidationError(
            "affine spot conditioning does not support call-style V0 rebates"
        )
    if product.airbag_config.airbag_barrier is not None:
        raise ValidationError(
            "affine spot conditioning does not support airbag payoffs"
        )
    if product.payoff_config.protection_type != ProtectionType.NONE:
        raise ValidationError(
            "affine spot conditioning does not support protected V1 payoffs"
        )


def conditional_standard_snowball_moments(
    *,
    product: SnowballOption,
    pricing_env: PricingEnvironment,
    base_paths: np.ndarray,
    log_spot_factor_loadings: np.ndarray,
    ko_indices: np.ndarray,
    ki_indices: np.ndarray,
    maturity: float,
    discount_factors: Callable[[np.ndarray], np.ndarray],
) -> ConditionalSnowballMoments:
    """Integrate a standard discrete Snowball payoff over one Gaussian factor.

    ``base_paths`` and ``log_spot_factor_loadings`` must share shape
    ``(n_outer_paths, n_contractual_times + 1)``. Column zero is valuation
    spot; event index ``i`` maps to column ``i + 1`` just as in the native MC
    payoff kernel.
    """

    _validate_supported_product(product)
    base = np.asarray(base_paths, dtype=float)
    loading = np.asarray(log_spot_factor_loadings, dtype=float)
    ko_indices = np.asarray(ko_indices, dtype=int)
    ki_indices = np.asarray(ki_indices, dtype=int)
    if base.ndim != 2 or base.shape != loading.shape:
        raise ValidationError(
            "base_paths and log_spot_factor_loadings must be same-shaped 2D arrays"
        )
    if base.shape[0] == 0 or base.shape[1] < 2:
        raise ValidationError("affine spot conditioning requires non-empty paths")
    if not np.all(np.isfinite(base)) or np.any(base <= 0.0):
        raise ValidationError("conditional base paths must be finite and positive")
    if not np.all(np.isfinite(loading)) or np.any(loading < 0.0):
        raise ValidationError("conditional log-spot loadings must be finite and non-negative")
    if np.any(ko_indices < 0) or np.any(ko_indices + 1 >= base.shape[1]):
        raise ValidationError("KO indices are outside conditional path columns")
    if np.any(ki_indices < 0) or np.any(ki_indices + 1 >= base.shape[1]):
        raise ValidationError("KI indices are outside conditional path columns")

    event_columns = np.unique(np.concatenate((ko_indices + 1, ki_indices + 1)))
    if event_columns.size and np.any(loading[:, event_columns] <= 0.0):
        raise ValidationError(
            "affine spot conditioning requires positive loading at every barrier event"
        )

    n_outer = base.shape[0]
    ko_profile = product.get_ko_observation_profile(pricing_env)
    ko_barriers = np.asarray(ko_profile["barriers"], dtype=float)
    ko_payoffs = np.asarray(ko_profile["payoffs"], dtype=float)
    ko_times = np.asarray(ko_profile["observation_times"], dtype=float)
    ko_settlement_times = np.asarray(
        ko_profile["settlement_times"], dtype=float
    )
    n_ko = ko_indices.size
    if not all(
        values.shape == (n_ko,)
        for values in (ko_barriers, ko_payoffs, ko_times, ko_settlement_times)
    ):
        raise ValidationError("resolved KO profile does not match KO grid indices")
    ko_dfs = np.asarray(discount_factors(ko_settlement_times), dtype=float)
    if ko_dfs.shape == ():
        ko_dfs = np.full(n_ko, float(ko_dfs), dtype=float)
    if ko_dfs.shape != (n_ko,) or not np.all(np.isfinite(ko_dfs)):
        raise PricingError("invalid KO discount factors in conditional payoff")

    discounted_payoff = np.zeros(n_outer, dtype=float)
    ko_probability = np.zeros(n_outer, dtype=float)
    ko_time_numerator = np.zeros(n_outer, dtype=float)
    survivor_upper = np.full(n_outer, np.inf, dtype=float)

    # A path first knocks out at event j when its shared factor is above the
    # event-j threshold but below every preceding threshold.
    for j, column in enumerate(ko_indices + 1):
        threshold = (
            np.log(ko_barriers[j]) - np.log(base[:, column])
        ) / loading[:, column]
        event_probability = _normal_interval_probability(threshold, survivor_upper)
        discounted_payoff += event_probability * ko_payoffs[j] * ko_dfs[j]
        ko_probability += event_probability
        ko_time_numerator += event_probability * ko_times[j]
        survivor_upper = np.minimum(survivor_upper, threshold)

    survivor_probability = ndtr(survivor_upper)

    already_knocked_in = bool(
        getattr(product, "_otc_lifecycle_knocked_in", False)
    )
    if already_knocked_in:
        ki_upper = np.full(n_outer, np.inf, dtype=float)
    elif product.has_ki_barrier:
        ki_profile = product.get_ki_observation_profile(pricing_env)
        ki_barriers = np.asarray(ki_profile["barriers"], dtype=float)
        if ki_barriers.shape != (ki_indices.size,):
            raise ValidationError(
                "resolved KI profile does not match KI grid indices"
            )
        ki_thresholds = np.empty((n_outer, ki_indices.size), dtype=float)
        for j, column in enumerate(ki_indices + 1):
            ki_thresholds[:, j] = (
                np.log(ki_barriers[j]) - np.log(base[:, column])
            ) / loading[:, column]
        ki_upper = np.max(ki_thresholds, axis=1)
    else:
        ki_upper = np.full(n_outer, -np.inf, dtype=float)

    v1_upper = np.minimum(survivor_upper, ki_upper)
    v1_probability = ndtr(v1_upper)
    v0_probability = np.clip(
        survivor_probability - v1_probability, 0.0, 1.0
    )

    maturity_df_raw = np.asarray(
        discount_factors(np.array([float(maturity)], dtype=float)), dtype=float
    ).reshape(-1)
    if maturity_df_raw.size != 1 or not np.isfinite(maturity_df_raw[0]):
        raise PricingError("invalid maturity discount factor in conditional payoff")
    maturity_df = float(maturity_df_raw[0])

    # The supported V0 payoff is constant. The supported V1 payoff is a
    # principal plus a linear short put below strike; querying the product at
    # two points preserves its own annualization and multiplier semantics.
    v0_payoff = float(product.get_maturity_payoff_v0(product.strike, pricing_env))
    v1_principal = float(
        product.get_maturity_payoff_v1(product.strike, pricing_env)
    )
    probe = min(1.0, 0.5 * float(product.strike))
    v1_below = float(
        product.get_maturity_payoff_v1(product.strike - probe, pricing_env)
    )
    downside_slope = (v1_principal - v1_below) / probe
    if not np.isfinite(downside_slope) or downside_slope < 0.0:
        raise PricingError("unsupported V1 payoff slope in conditional payoff")

    terminal_base = base[:, -1]
    terminal_loading = loading[:, -1]
    if np.any(terminal_loading <= 0.0):
        raise ValidationError(
            "affine spot conditioning requires positive terminal loading"
        )
    strike_threshold = (
        np.log(float(product.strike)) - np.log(terminal_base)
    ) / terminal_loading
    downside_upper = np.minimum(v1_upper, strike_threshold)
    downside_probability = ndtr(downside_upper)
    truncated_spot = _truncated_lognormal_first_moment(
        terminal_base, terminal_loading, downside_upper
    )
    v1_expectation = (
        v1_principal * v1_probability
        + downside_slope
        * (truncated_spot - float(product.strike) * downside_probability)
    )
    discounted_payoff += maturity_df * (
        v0_payoff * v0_probability + v1_expectation
    )

    probability_total = ko_probability + v0_probability + v1_probability
    if (
        not np.all(np.isfinite(discounted_payoff))
        or not np.allclose(probability_total, 1.0, rtol=0.0, atol=2e-13)
    ):
        raise PricingError(
            "conditional Snowball state probabilities do not close to one"
        )

    return ConditionalSnowballMoments(
        discounted_payoff=discounted_payoff,
        ko_probability=ko_probability,
        v0_probability=v0_probability,
        v1_probability=v1_probability,
        ko_time_numerator=ko_time_numerator,
    )


__all__ = [
    "ConditionalSnowballMoments",
    "conditional_standard_snowball_moments",
]
