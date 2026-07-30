"""Request-scoped settlement timing bundles for autocallable MC engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from quantark.asset.equity.engine.settlement_support import (
    resolve_terminal_timing,
)
from quantark.asset.equity.product.option.observation_schedule import (
    ResolvedObservationRecord,
)
from quantark.asset.equity.settlement import ResolvedPaymentTiming
from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class AutocallablePaymentTimings:
    """Aligned event and terminal payment data for one valuation."""

    observation_times: np.ndarray
    observation_payment_times: np.ndarray
    observation_payment_dfs: np.ndarray
    terminal: ResolvedPaymentTiming

    def __post_init__(self) -> None:
        n = self.observation_times.size
        if (
            self.observation_payment_times.size != n
            or self.observation_payment_dfs.size != n
        ):
            raise ValidationError(
                "autocallable payment timing arrays must be aligned"
            )
        for value in (
            self.observation_times,
            self.observation_payment_times,
            self.observation_payment_dfs,
        ):
            value.setflags(write=False)


def resolve_autocallable_payment_timings(
    product,
    pricing_env,
    records: Sequence[ResolvedObservationRecord],
    *,
    event_paid: bool,
) -> AutocallablePaymentTimings:
    """Resolve observation and terminal payment arrays exactly once."""
    terminal = resolve_terminal_timing(product, pricing_env)
    observation_times = np.asarray(
        [record.observation_time for record in records],
        dtype=float,
    )
    if event_paid:
        payment_times = np.asarray(
            [record.settlement_time for record in records],
            dtype=float,
        )
    else:
        payment_times = np.full(
            observation_times.shape,
            float(terminal.payment_time),
            dtype=float,
        )
    payment_dfs = np.asarray(
        [
            pricing_env.get_discount_factor(float(time))
            for time in payment_times
        ],
        dtype=float,
    )
    return AutocallablePaymentTimings(
        observation_times=observation_times,
        observation_payment_times=payment_times,
        observation_payment_dfs=payment_dfs,
        terminal=terminal,
    )


__all__ = [
    "AutocallablePaymentTimings",
    "resolve_autocallable_payment_timings",
]
