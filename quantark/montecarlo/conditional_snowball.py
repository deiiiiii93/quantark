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
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from scipy.special import log_ndtr, ndtr

from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, ProtectionType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import Tolerance

if TYPE_CHECKING:
    # ``montecarlo`` sits below ``asset``: equity MC engines import this
    # module, so a runtime import of the product here closes a cycle.
    from quantark.asset.equity.product.option.snowball_option import (
        SnowballOption,
    )


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
    # Deferred: see the TYPE_CHECKING import above.
    from quantark.asset.equity.product.option.snowball_option import (
        SnowballOption,
    )

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
    ko_cash: Optional[np.ndarray] = None,
    no_ki_cash: float = 0.0,
    ki_cash: float = 0.0,
) -> ConditionalSnowballMoments:
    """Integrate a standard discrete Snowball payoff over one Gaussian factor.

    ``base_paths`` and ``log_spot_factor_loadings`` must share shape
    ``(n_outer_paths, n_contractual_times + 1)``. Column zero is valuation
    spot; event index ``i`` maps to column ``i + 1`` just as in the native MC
    payoff kernel.

    ``ko_cash`` (one amount per KO record), ``no_ki_cash`` and ``ki_cash`` are
    folded cash legs (ship spec 2026-10-21 D4): each is added to the payoff of
    its event where the record's own cash sits -- the KO record's payoff under
    the record's settlement discount, the never-knocked-in / knocked-in
    redemption under the maturity discount. The payoff stays affine in them, so
    the integration is exact. The defaults leave every number unchanged.
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
    if ko_cash is not None:
        ko_cash = np.asarray(ko_cash, dtype=float)
        if ko_cash.shape != (n_ko,) or not np.all(np.isfinite(ko_cash)):
            raise ValidationError("folded KO cash must hold one finite amount per KO record")
        ko_payoffs = ko_payoffs + ko_cash

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

    # Folded branch cash shifts each redemption by a constant: the knocked-in
    # downside slope is the product's own, read before the shift.
    v0_payoff = v0_payoff + float(no_ki_cash)
    v1_principal = v1_principal + float(ki_cash)

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


def conditional_standard_snowball_event_record(
    *,
    product: SnowballOption,
    pricing_env: PricingEnvironment,
    base_paths: np.ndarray,
    log_spot_factor_loadings: np.ndarray,
    ko_indices: np.ndarray,
    ki_indices: np.ndarray,
    maturity: float,
    discount_factors: Callable[[np.ndarray], np.ndarray],
) -> dict:
    """One batch's event-statistics record for the exact one-factor estimator (MC event statistics on
    ``price()``'s paths, unified event statistics spec AC2).

    The estimator prices ``E[payoff | base path]`` by integrating the shared Gaussian factor ``Z``; its
    native outcomes are intervals of ``Z`` on the same base paths. The record holds, summed over the
    batch's rows, each outcome's exact conditional probability and conditional discounted / undiscounted
    cash, on the same thresholds :func:`conditional_standard_snowball_moments` integrates:

    - knock-out at record ``j``: ``Z`` in ``[threshold_j, min_{i<j} threshold_i)``;
    - the maturity branches as the moments split them (never knocked in / knocked in, no knock-out);
    - first knock-in at record ``j`` before any knock-out: the first-knock-in interval
      ``(max_{i<j} ki_threshold_i, ki_threshold_j]`` below every knock-out threshold observed at or before
      ``t_j`` (a knock-out wins a tie with a discrete knock-in);
    - knock-out at record ``j`` with no knock-in before it: the knock-out interval above every knock-in
      threshold observed strictly before ``t_j``.

    Keys and meaning follow the native kernel's record (``kind = "conditional"``: sums of probabilities,
    not path counts). Read-only: the price's moments are computed separately and unchanged.
    """
    _validate_supported_product(product)
    base = np.asarray(base_paths, dtype=float)
    loading = np.asarray(log_spot_factor_loadings, dtype=float)
    ko_indices = np.asarray(ko_indices, dtype=int)
    ki_indices = np.asarray(ki_indices, dtype=int)
    n_rows = base.shape[0]

    ko_profile = product.get_ko_observation_profile(pricing_env)
    ko_barriers = np.asarray(ko_profile["barriers"], dtype=float)
    ko_payoffs = np.asarray(ko_profile["payoffs"], dtype=float)
    ko_times = np.asarray(ko_profile["observation_times"], dtype=float)
    ko_settlement_times = np.asarray(ko_profile["settlement_times"], dtype=float)
    n_ko = ko_indices.size
    ko_dfs = np.asarray(discount_factors(ko_settlement_times), dtype=float)
    if ko_dfs.shape == ():
        ko_dfs = np.full(n_ko, float(ko_dfs), dtype=float)

    ko_count = np.zeros(n_ko, dtype=float)
    ko_disc_cash = np.zeros(n_ko, dtype=float)
    ko_cash = np.zeros(n_ko, dtype=float)
    ko_thresholds = np.empty((n_rows, n_ko), dtype=float)
    ko_lower = np.empty((n_rows, n_ko), dtype=float)
    ko_upper = np.empty((n_rows, n_ko), dtype=float)
    survivor_upper = np.full(n_rows, np.inf, dtype=float)
    for j, column in enumerate(ko_indices + 1):
        threshold = (np.log(ko_barriers[j]) - np.log(base[:, column])) / loading[:, column]
        event_probability = _normal_interval_probability(threshold, survivor_upper)
        mass = float(event_probability.sum())
        ko_count[j] = mass
        ko_cash[j] = mass * ko_payoffs[j]
        ko_disc_cash[j] = mass * ko_payoffs[j] * ko_dfs[j]
        ko_thresholds[:, j] = threshold
        ko_lower[:, j] = threshold
        ko_upper[:, j] = survivor_upper
        survivor_upper = np.minimum(survivor_upper, threshold)
    survivor_probability = ndtr(survivor_upper)

    already_knocked_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
    ki_row_times = np.zeros(0, dtype=float)
    ki_first_count = np.zeros(0, dtype=float)
    ko_clean_count = ko_count.copy()
    ki_before_ko = 0.0
    ki_time_sum = 0.0
    if already_knocked_in:
        ki_upper = np.full(n_rows, np.inf, dtype=float)
        ko_clean_count = np.zeros(n_ko, dtype=float)
        ki_before_ko = float(n_rows)
    elif product.has_ki_barrier:
        ki_profile = product.get_ki_observation_profile(pricing_env)
        ki_barriers = np.asarray(ki_profile["barriers"], dtype=float)
        ki_row_times = np.asarray(ki_profile["observation_times"], dtype=float)
        ki_first_count = np.zeros(ki_indices.size, dtype=float)
        ki_thresholds = np.empty((n_rows, ki_indices.size), dtype=float)
        previous = np.full(n_rows, -np.inf, dtype=float)
        for j, column in enumerate(ki_indices + 1):
            threshold = (np.log(ki_barriers[j]) - np.log(base[:, column])) / loading[:, column]
            ki_thresholds[:, j] = threshold
            # no knock-out at or before this knock-in observation: below every such knock-out threshold
            ko_before = ko_times <= ki_row_times[j] + Tolerance.ZERO
            no_ko_upper = (
                np.min(ko_thresholds[:, ko_before], axis=1) if np.any(ko_before)
                else np.full(n_rows, np.inf, dtype=float)
            )
            mass = float(_normal_interval_probability(previous, np.minimum(threshold, no_ko_upper)).sum())
            ki_first_count[j] = mass
            ki_before_ko += mass
            ki_time_sum += mass * float(ki_row_times[j])
            previous = np.maximum(previous, threshold)
        ki_upper = previous if ki_indices.size else np.full(n_rows, -np.inf, dtype=float)
        for j in range(n_ko):
            # no knock-in strictly before this knock-out observation: above every such knock-in threshold
            ki_before = ki_row_times < ko_times[j] - Tolerance.ZERO
            no_ki_lower = (
                np.max(ki_thresholds[:, ki_before], axis=1) if np.any(ki_before)
                else np.full(n_rows, -np.inf, dtype=float)
            )
            ko_clean_count[j] = float(
                _normal_interval_probability(np.maximum(ko_lower[:, j], no_ki_lower), ko_upper[:, j]).sum()
            )
    else:
        ki_upper = np.full(n_rows, -np.inf, dtype=float)

    v1_upper = np.minimum(survivor_upper, ki_upper)
    v1_probability = ndtr(v1_upper)
    v0_probability = np.clip(survivor_probability - v1_probability, 0.0, 1.0)

    maturity_df = float(
        np.asarray(discount_factors(np.array([float(maturity)], dtype=float)), dtype=float).reshape(-1)[0]
    )
    v0_payoff = float(product.get_maturity_payoff_v0(product.strike, pricing_env))
    v1_principal = float(product.get_maturity_payoff_v1(product.strike, pricing_env))
    probe = min(1.0, 0.5 * float(product.strike))
    v1_below = float(product.get_maturity_payoff_v1(product.strike - probe, pricing_env))
    downside_slope = (v1_principal - v1_below) / probe
    terminal_base = base[:, -1]
    terminal_loading = loading[:, -1]
    strike_threshold = (np.log(float(product.strike)) - np.log(terminal_base)) / terminal_loading
    downside_upper = np.minimum(v1_upper, strike_threshold)
    downside_probability = ndtr(downside_upper)
    truncated_spot = _truncated_lognormal_first_moment(terminal_base, terminal_loading, downside_upper)
    downside = downside_slope * (truncated_spot - float(product.strike) * downside_probability)
    v1_expectation = v1_principal * v1_probability + downside
    v0_cash = float((v0_payoff * v0_probability).sum())
    v1_cash = float(v1_expectation.sum())
    v0_disc_cash = maturity_df * v0_cash
    v1_disc_cash = maturity_df * v1_cash
    return {
        "kind": "conditional",
        "n": int(n_rows),
        "pv_sum": float(ko_disc_cash.sum() + v0_disc_cash + v1_disc_cash),
        "ko_count": ko_count,
        "ko_clean_count": ko_clean_count,
        "ko_disc_cash": ko_disc_cash,
        "ko_cash": ko_cash,
        "v0_count": float(v0_probability.sum()),
        "v1_count": float(v1_probability.sum()),
        "v0_disc_cash": v0_disc_cash,
        "v1_disc_cash": v1_disc_cash,
        "v0_cash": v0_cash,
        "v1_cash": v1_cash,
        # the knocked-in downside ``principal - redemption`` = -slope (S - K)^- integrated, discounted
        "v1_disc_loss": maturity_df * float(-downside.sum()),
        "ki_no_ko_count": float(v1_probability.sum()),
        "ki_before_ko_count": ki_before_ko,
        "ki_time_sum": ki_time_sum,
        "ki_first_count": ki_first_count,
        "ki_row_times": ki_row_times,
        "ki_breached_at_valuation": False,
    }


__all__ = [
    "ConditionalSnowballMoments",
    "conditional_standard_snowball_event_record",
    "conditional_standard_snowball_moments",
]
