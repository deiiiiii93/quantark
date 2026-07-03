"""Bridge PricingEnvironment term structures onto MC generator time grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from quantark.asset.equity.process.bsm.qmc_path_generator import _build_time_grid
from quantark.priceenv.term_sampling import TermCoefficients


@dataclass(frozen=True)
class McTermInputs:
    """Per-step forward coefficients on an MC engine's time grid.

    ``times`` includes t=0 (shape ``(time_steps + 1,)``); the interval arrays
    ``rrf``/``div``/``vol`` have shape ``(time_steps,)`` — entry k applies over
    ``[times[k], times[k+1]]`` and matches GBMPathGenerator's step layout.
    """

    times: np.ndarray
    rrf: np.ndarray
    div: np.ndarray
    vol: np.ndarray
    node_dfs: np.ndarray


def build_mc_term_inputs(
    pricing_env,
    ref_strike: float,
    maturity: float,
    time_steps: int,
    dt_array: Optional[np.ndarray] = None,
) -> McTermInputs:
    """Sample forward r/q/vol and node DFs on the generator's exact grid.

    ``_build_time_grid`` returns step-terminal times only (``np.cumsum(dt)``,
    shape ``(time_steps,)``) — it does NOT include t=0. TermCoefficients needs
    the full node grid, so 0.0 is prepended here; interval arrays then have
    length ``time_steps`` and ``node_dfs`` length ``time_steps + 1``.
    """
    times, _ = _build_time_grid(
        maturity=maturity, time_steps=time_steps, dt_array=dt_array
    )
    grid = np.concatenate(([0.0], np.asarray(times, dtype=float)))
    tc = TermCoefficients.from_env(pricing_env, grid, ref_strike=ref_strike)
    return McTermInputs(
        times=tc.t_grid,
        rrf=tc.fwd_rates,
        div=tc.fwd_carry,
        vol=tc.step_vols,
        node_dfs=tc.node_dfs,
    )


def df_at(inputs: McTermInputs, t: float) -> float:
    """Discount factor at a grid node time; rejects off-grid times."""
    idx = int(np.argmin(np.abs(inputs.times - float(t))))
    if abs(float(inputs.times[idx]) - float(t)) > 1e-12:
        raise ValueError(f"t={t} is not a node of the MC time grid")
    return float(inputs.node_dfs[idx])
