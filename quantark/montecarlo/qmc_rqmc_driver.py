"""
Created on Mon Nov 17 2025

@description: RQMC batching and adaptive stopping driver for Monte Carlo
               pricing engines using GBMPathGenerator or similar path
               generators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol, Tuple

import numpy as np


class PathGenerator(Protocol):
    """
    Minimal protocol for path generators compatible with the RQMC driver.
    """

    num_paths: int

    def generate_paths(
        self,
        seed: Optional[int] = None,
        batch_id: Optional[int] = None,
        return_aux: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        ...


PricerFn = Callable[[np.ndarray, Optional[Dict[str, np.ndarray]]], np.ndarray]


@dataclass
class RQMCResult:
    """
    Result container for RQMC batching runs.
    """

    price: float
    std_error: float
    total_paths: int
    batches_used: int
    batch_means: np.ndarray


@dataclass(frozen=True)
class RQMCCheckpoint:
    """Post-batch stopping-criterion evaluation record (spec section 8.4).

    ``std_error`` is None strictly before ``min_batches`` (the criterion is
    not evaluated there); ``stopped`` marks the checkpoint at which the run
    terminated (target reached or ``max_batches`` exhausted).
    """

    batch_index: int
    batch_mean: float
    running_mean: float
    std_error: Optional[float]
    stopped: bool


@dataclass(frozen=True)
class RQMCRunSpec:
    """Engine-provided description of one adaptive RQMC run.

    ``finalize`` assembles the engine-native result object from the
    RQMCResult; it is the SAME callable on the direct and session paths, so
    result assembly (including any extra statistics batch) is shared code.
    """

    pricer_fn: PricerFn
    path_generator: object
    max_batches: int
    min_batches: int
    target_std: float
    paths_per_batch: int
    time_steps: int
    scheme: str
    finalize: object  # Callable[[RQMCResult], engine-native result]
    product: object = None  # the priced product (session postamble needs it)


def run_rqmc_traced(
    pricer_fn: PricerFn,
    path_generator: PathGenerator,
    max_batches: int,
    target_std: float,
    min_batches: int = 1,
) -> Tuple[RQMCResult, Tuple[RQMCCheckpoint, ...]]:
    """run_rqmc with a per-batch checkpoint trace (spec section 8.4).

    This is THE stopping-loop implementation; ``run_rqmc`` delegates here,
    so direct and session executions share one arithmetic path by
    construction.
    """
    if max_batches <= 0:
        raise ValueError("max_batches must be positive")
    if min_batches <= 0:
        raise ValueError("min_batches must be positive")
    if min_batches > max_batches:
        raise ValueError("min_batches cannot exceed max_batches")
    if target_std <= 0.0:
        raise ValueError("target_std must be positive")

    batch_means = []
    checkpoints = []
    n_paths_per_batch = path_generator.num_paths

    # Welford's algorithm over batch means
    mean = 0.0
    m2 = 0.0

    for batch_id in range(max_batches):
        paths, aux = path_generator.generate_paths(batch_id=batch_id, return_aux=True)
        payoffs = pricer_fn(paths, aux)
        payoffs = np.asarray(payoffs, dtype=float)
        if payoffs.ndim != 1 or payoffs.shape[0] != n_paths_per_batch:
            raise ValueError(
                "pricer_fn must return a 1D array with one payoff per path "
                f"(expected length {n_paths_per_batch}, got {payoffs.shape})."
            )

        batch_mean = float(payoffs.mean())
        batch_means.append(batch_mean)

        n = batch_id + 1
        delta = batch_mean - mean
        mean += delta / n
        m2 += delta * (batch_mean - mean)

        if n >= min_batches:
            if n > 1:
                variance = m2 / (n - 1)
            else:
                variance = 0.0
            std_error = np.sqrt(variance / n)

            stopped = bool(std_error <= target_std or n == max_batches)
            checkpoints.append(RQMCCheckpoint(
                batch_index=batch_id, batch_mean=batch_mean,
                running_mean=mean, std_error=float(std_error),
                stopped=stopped,
            ))
            if stopped:
                result = RQMCResult(
                    price=mean,
                    std_error=std_error,
                    total_paths=n * n_paths_per_batch,
                    batches_used=n,
                    batch_means=np.array(batch_means, dtype=float),
                )
                return result, tuple(checkpoints)
        else:
            checkpoints.append(RQMCCheckpoint(
                batch_index=batch_id, batch_mean=batch_mean,
                running_mean=mean, std_error=None, stopped=False,
            ))

    raise RuntimeError("No batches were run in RQMC driver.")


def run_rqmc(
    pricer_fn: PricerFn,
    path_generator: PathGenerator,
    max_batches: int,
    target_std: float,
    min_batches: int = 1,
) -> RQMCResult:
    """
    Run randomized QMC (RQMC) in batches with adaptive stopping.

    Each batch corresponds to an independent scrambled Sobol sequence (or
    independent pseudorandom run), and the estimator is the average of the
    batch means. The standard error is based on the sample variance of the
    batch means.

    Parameters
    ----------
    pricer_fn : callable
        Function with signature pricer_fn(paths, aux) -> payoffs_per_path,
        where paths is an array of shape (n_paths, n_steps + 1) and aux is
        a dictionary with auxiliary data (e.g., importance weights).
    path_generator : PathGenerator
        Path generator implementing the PathGenerator protocol.
    max_batches : int
        Maximum number of RQMC batches to run.
    target_std : float
        Target standard error for stopping the simulation.
    min_batches : int, optional
        Minimum number of batches to run before checking the stopping
        criterion.

    Returns
    -------
    RQMCResult
        Result object containing the estimated price, standard error,
        total number of paths used, and per-batch means.
    """
    return run_rqmc_traced(
        pricer_fn, path_generator, max_batches, target_std, min_batches
    )[0]


__all__ = [
    "RQMCCheckpoint",
    "RQMCResult",
    "RQMCRunSpec",
    "run_rqmc",
    "run_rqmc_traced",
]
