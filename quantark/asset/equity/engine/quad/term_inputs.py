"""Bridge PricingEnvironment term structures onto QUAD observation grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from quantark.priceenv.term_sampling import TermCoefficients


@dataclass(frozen=True)
class QuadTermParams:
    """Per-observation-interval forward parameters for QUAD engines.

    Entry i of ``rate``/``div``/``vol`` covers (t_{i-1}, t_i] of the
    observation grid; ``node_dfs`` are DF(0, t_i) including t=0. The interval
    arrays plug straight into ``QuadratureCore(rate=, div=, vol=)`` (its
    ``_broadcast_param`` accepts length-n_obs sequences).
    """

    rate: np.ndarray
    div: np.ndarray
    vol: np.ndarray
    node_dfs: np.ndarray


def build_quad_term_params(
    pricing_env, ref_strike: float, observation_times: Sequence[float]
) -> QuadTermParams:
    grid = np.concatenate(([0.0], np.asarray(observation_times, dtype=float)))
    tc = TermCoefficients.from_env(pricing_env, grid, ref_strike=float(ref_strike))
    return QuadTermParams(
        rate=tc.fwd_rates, div=tc.fwd_carry, vol=tc.step_vols, node_dfs=tc.node_dfs
    )
