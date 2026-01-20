"""
Engine-level event stats and cashflow decomposition types.

These types define an optional API that engines MAY implement to provide
per-observation event probabilities and expected discounted cashflows for
autocallable products (Snowball-first).
"""

from __future__ import annotations

from dataclasses import dataclass

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

