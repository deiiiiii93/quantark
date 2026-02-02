"""
Engine-level event stats and cashflow decomposition types.

These types define an optional API that engines MAY implement to provide
per-observation event probabilities and expected discounted cashflows for
autocallable products (Snowball-first).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AutocallableEventStats:
    """
    Event probabilities and expected discounted cashflows.

    Attributes:
        pv: Present value produced by the engine for the given product/env.
        ko_times: KO observation times (year fractions from valuation date).
        ko_probability: Probability of KO occurring at each observation time.
        survival_probability: Probability of surviving (not KO'd) up to each observation.
        expected_discounted_ko_cashflow: Expected discounted KO redemption cashflow at each observation.
        ki_probability: Probability that KI occurred at least once before maturity (if applicable).
        expected_discounted_maturity_cashflow: Expected discounted maturity cashflow (conditional on no KO).
        reconciliation_error: pv minus sum(expected discounted cashflows) if computed, else 0.0.
    """

    pv: float
    ko_times: np.ndarray
    ko_probability: np.ndarray
    survival_probability: np.ndarray
    expected_discounted_ko_cashflow: np.ndarray
    ki_probability: float
    expected_discounted_maturity_cashflow: float
    reconciliation_error: float = 0.0


@dataclass(frozen=True)
class PhoenixEventStats(AutocallableEventStats):
    """
    Event stats for Phoenix options including coupon diagnostics.

    Attributes:
        coupon_probability: Coupon trigger probability at each observation time.
        expected_discounted_coupon_cashflow: Expected discounted coupon cashflow per observation.
    """

    coupon_probability: np.ndarray = field(default_factory=lambda: np.array([]))
    expected_discounted_coupon_cashflow: np.ndarray = field(
        default_factory=lambda: np.array([])
    )


@dataclass(frozen=True)
class KOResetEventStats(AutocallableEventStats):
    """
    Event stats for KO-reset snowball options.

    Attributes:
        pre_ko_times: Pre-KI KO observation times (absolute).
        pre_ko_probability: KO probability per pre-KI observation.
        post_ko_times: Post-KI KO observation times (absolute or offsets for REBASED).
        post_ko_probability: KO probability per post-KI observation (or offset).
        pre_ko_probability_total: Total probability of pre-KI KO.
        post_ko_probability_total: Total probability of post-KI KO.
        expected_discounted_post_ko_cashflow: Total expected discounted KO cashflow after KI.
    """

    pre_ko_times: np.ndarray = field(default_factory=lambda: np.array([]))
    pre_ko_probability: np.ndarray = field(default_factory=lambda: np.array([]))
    post_ko_times: np.ndarray = field(default_factory=lambda: np.array([]))
    post_ko_probability: np.ndarray = field(default_factory=lambda: np.array([]))
    pre_ko_probability_total: float = 0.0
    post_ko_probability_total: float = 0.0
    expected_discounted_post_ko_cashflow: float = 0.0
