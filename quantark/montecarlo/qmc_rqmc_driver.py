"""
Created on Mon Nov 17 2025

@description: RQMC batching and adaptive stopping driver for Monte Carlo
               pricing engines using GBMPathGenerator or similar path
               generators.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
class PairedRQMCGreeksResult:
    """Central spot Greeks from jointly scrambled RQMC repricings.

    Each row of ``batch_estimates`` is formed from the *same* RQMC batch id
    at ``S-h``, ``S`` and ``S+h``.  Computing delta/gamma inside the batch
    preserves the common-random-number covariance before the standard error
    is estimated across independent Sobol scrambles.

    Columns are ``down_price``, ``base_price``, ``up_price``, ``delta`` and
    ``gamma`` in that order.
    """

    price: float
    price_std_error: float
    delta: float
    delta_std_error: float
    gamma: float
    gamma_std_error: float
    spot: float
    relative_bump: float
    absolute_bump: float
    paths_per_batch: int
    batches_used: int
    total_unique_paths: int
    total_path_valuations: int
    randomization_key: str
    batch_estimates: np.ndarray
    covariance: np.ndarray
    # Optional zero-bias conditional-control component using the same five
    # columns. This is retained per scramble so validation harnesses can
    # attribute target-minus-control variance without rerunning paths.
    control_batch_estimates: Optional[np.ndarray] = None

    @property
    def batch_price_down(self) -> np.ndarray:
        return self.batch_estimates[:, 0]

    @property
    def batch_price_base(self) -> np.ndarray:
        return self.batch_estimates[:, 1]

    @property
    def batch_price_up(self) -> np.ndarray:
        return self.batch_estimates[:, 2]

    @property
    def batch_delta(self) -> np.ndarray:
        return self.batch_estimates[:, 3]

    @property
    def batch_gamma(self) -> np.ndarray:
        return self.batch_estimates[:, 4]

    def as_dict(self, *, include_batches: bool = True) -> dict:
        payload = {
            "price": self.price,
            "price_std_error": self.price_std_error,
            "delta": self.delta,
            "delta_std_error": self.delta_std_error,
            "gamma": self.gamma,
            "gamma_std_error": self.gamma_std_error,
            "spot": self.spot,
            "relative_bump": self.relative_bump,
            "absolute_bump": self.absolute_bump,
            "paths_per_batch": self.paths_per_batch,
            "batches_used": self.batches_used,
            "total_unique_paths": self.total_unique_paths,
            "total_path_valuations": self.total_path_valuations,
            "randomization_key": self.randomization_key,
            "covariance": self.covariance.tolist(),
        }
        if include_batches:
            payload["batch_estimates"] = self.batch_estimates.tolist()
            if self.control_batch_estimates is not None:
                payload["control_batch_estimates"] = (
                    self.control_batch_estimates.tolist()
                )
        return payload


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
    # Resolved per-path draw dimension (streams_per_step * time_steps):
    # distinguishes model schemes with different stream counts and sizes
    # the memory admission estimate (code-gate finding 2026-07-16).
    dimension: int = 0
    # Stable engine-provided description of the scrambled point set. Paired
    # finite differences fail closed unless all three specs provide the same
    # key; an identical batch_id does not prove coupling when seeds can differ.
    randomization_key: object = None
    # Optional per-path exact conditional-control payoff. Paired Greek runs
    # evaluate it on the already generated paths/aux metadata and preserve its
    # batch price/delta/gamma beside the primary estimator.
    control_pricer_fn: Optional[PricerFn] = None
    # Number of target path states evaluated per outer path after conditional
    # stratification. This affects work accounting, not the independent sample
    # size used for standard errors.
    path_valuation_multiplier: int = 1


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


def _paired_spec_contract(
    specs: tuple[RQMCRunSpec, RQMCRunSpec, RQMCRunSpec],
) -> tuple[int, str]:
    """Validate that three specs describe one couplable RQMC experiment."""
    labels = ("down", "base", "up")
    paths = []
    for label, spec in zip(labels, specs):
        n_generator = int(getattr(spec.path_generator, "num_paths", -1))
        n_spec = int(spec.paths_per_batch)
        if n_generator <= 0 or n_generator != n_spec:
            raise ValueError(
                f"{label} RQMC spec has inconsistent paths_per_batch "
                f"({n_spec}) and generator.num_paths ({n_generator})"
            )
        paths.append(n_spec)
    if len(set(paths)) != 1:
        raise ValueError("paired RQMC specs must use the same paths_per_batch")

    for field in (
        "time_steps",
        "dimension",
        "scheme",
        "path_valuation_multiplier",
    ):
        values = [getattr(spec, field) for spec in specs]
        if len(set(values)) != 1:
            raise ValueError(f"paired RQMC specs must use the same {field}")
    randomization_keys = [spec.randomization_key for spec in specs]
    if any(key is None for key in randomization_keys):
        raise ValueError(
            "paired RQMC specs must declare a randomization_key proving "
            "that they share one scrambled point set"
        )
    if not all(key == randomization_keys[0] for key in randomization_keys[1:]):
        raise ValueError("paired RQMC specs must use the same randomization_key")
    control_flags = [spec.control_pricer_fn is not None for spec in specs]
    if any(control_flags) and not all(control_flags):
        raise ValueError(
            "paired RQMC specs must all provide the conditional control or none"
        )
    return paths[0], repr(randomization_keys[0])


def _paired_batch_components(
    spec: RQMCRunSpec,
    batch_id: int,
    expected_paths: int,
) -> tuple[float, Optional[float]]:
    paths, aux = spec.path_generator.generate_paths(
        batch_id=batch_id,
        return_aux=True,
    )
    payoffs = np.asarray(spec.pricer_fn(paths, aux), dtype=float)
    if payoffs.ndim != 1 or payoffs.shape[0] != expected_paths:
        raise ValueError(
            "paired RQMC pricer_fn must return one payoff per path "
            f"(expected {(expected_paths,)}, got {payoffs.shape})"
        )
    if not np.all(np.isfinite(payoffs)):
        raise ValueError("paired RQMC pricer_fn returned non-finite payoffs")
    control_mean = None
    if spec.control_pricer_fn is not None:
        control = np.asarray(spec.control_pricer_fn(paths, aux), dtype=float)
        if control.ndim != 1 or control.shape[0] != expected_paths:
            raise ValueError(
                "paired RQMC control_pricer_fn must return one payoff per path "
                f"(expected {(expected_paths,)}, got {control.shape})"
            )
        if not np.all(np.isfinite(control)):
            raise ValueError(
                "paired RQMC control_pricer_fn returned non-finite payoffs"
            )
        control_mean = float(np.mean(control))
    return float(np.mean(payoffs)), control_mean


def run_paired_rqmc_greeks(
    down_spec: RQMCRunSpec,
    base_spec: RQMCRunSpec,
    up_spec: RQMCRunSpec,
    *,
    spot: float,
    relative_bump: float = 0.01,
    batches: Optional[int] = None,
    batch_workers: int = 1,
) -> PairedRQMCGreeksResult:
    """Estimate price, delta and gamma with paired RQMC batches.

    The three engine specs must have matching scheme, draw dimension, time
    discretization and batch size.  Their path generators are called with an
    identical ``batch_id`` so independently constructed bumped engines use the
    same scrambled Sobol point set.  A fixed number of outer scrambles is used:
    optional stopping on a noisy Greek would invalidate the reported batch
    standard error.
    """
    spot = float(spot)
    relative_bump = float(relative_bump)
    if not np.isfinite(spot) or spot <= 0.0:
        raise ValueError("spot must be finite and positive")
    if not np.isfinite(relative_bump) or not 0.0 < relative_bump < 1.0:
        raise ValueError("relative_bump must be finite and between 0 and 1")

    specs = (down_spec, base_spec, up_spec)
    paths_per_batch, randomization_key = _paired_spec_contract(specs)
    max_coupled_batches = min(int(spec.max_batches) for spec in specs)
    batches_used = max_coupled_batches if batches is None else int(batches)
    if batches_used < 2:
        raise ValueError("paired RQMC Greeks require at least two batches")
    if batches_used > max_coupled_batches:
        raise ValueError(
            "requested paired RQMC batches exceed an engine spec maximum "
            f"({batches_used} > {max_coupled_batches})"
        )
    if isinstance(batch_workers, bool) or int(batch_workers) < 1:
        raise ValueError("batch_workers must be a positive integer")
    batch_workers = min(int(batch_workers), batches_used)

    absolute_bump = spot * relative_bump
    def greek_row(down: float, base: float, up: float):
        delta = (up - down) / (2.0 * absolute_bump)
        gamma = (up - 2.0 * base + down) / (absolute_bump * absolute_bump)
        return down, base, up, delta, gamma

    def estimate_batch(batch_id: int):
        components = [
            _paired_batch_components(spec, batch_id, paths_per_batch)
            for spec in specs
        ]
        primary = greek_row(*(component[0] for component in components))
        controls = [component[1] for component in components]
        control = (
            None
            if controls[0] is None
            else greek_row(*(float(value) for value in controls))
        )
        return primary, control

    if batch_workers == 1:
        rows = [estimate_batch(batch_id) for batch_id in range(batches_used)]
    else:
        # Each scramble owns independent generators and local path arrays.
        # executor.map preserves batch-id order, so threaded and serial
        # reductions are bitwise identical once each row has completed.
        with ThreadPoolExecutor(
            max_workers=batch_workers,
            thread_name_prefix="paired-rqmc",
        ) as executor:
            rows = list(executor.map(estimate_batch, range(batches_used)))
    estimates = np.asarray([row[0] for row in rows], dtype=float)
    control_estimates = (
        None
        if rows[0][1] is None
        else np.asarray([row[1] for row in rows], dtype=float)
    )

    means = np.mean(estimates, axis=0)
    covariance = np.asarray(np.cov(estimates, rowvar=False, ddof=1), dtype=float)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0) / batches_used)
    return PairedRQMCGreeksResult(
        price=float(means[1]),
        price_std_error=float(standard_errors[1]),
        delta=float(means[3]),
        delta_std_error=float(standard_errors[3]),
        gamma=float(means[4]),
        gamma_std_error=float(standard_errors[4]),
        spot=spot,
        relative_bump=relative_bump,
        absolute_bump=absolute_bump,
        paths_per_batch=paths_per_batch,
        batches_used=batches_used,
        total_unique_paths=paths_per_batch * batches_used,
        total_path_valuations=(
            3
            * paths_per_batch
            * batches_used
            * int(down_spec.path_valuation_multiplier)
        ),
        randomization_key=randomization_key,
        batch_estimates=estimates,
        covariance=covariance,
        control_batch_estimates=control_estimates,
    )


__all__ = [
    "RQMCCheckpoint",
    "PairedRQMCGreeksResult",
    "RQMCResult",
    "RQMCRunSpec",
    "run_rqmc",
    "run_rqmc_traced",
    "run_paired_rqmc_greeks",
]
