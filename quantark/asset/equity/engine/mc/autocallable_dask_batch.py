"""Shared legacy Dask batch fan-out/reduction for autocallable MC engines.

One implementation of the batch-size split, delayed fan-out, and sum/sum²
reduction that was previously triplicated across ``SnowballMCEngine``
(vanilla + KO-reset) and ``PhoenixMCEngine``. This is an INTRA-legacy
consolidation (spec §17.3): both legacy call paths now share one internal
reducer; behavior — messages, arithmetic order, result fields, lazy
``num_batches`` validation timing — is byte-preserved and gated by
``test/execution/test_legacy_dask_goldens.py``.
"""
import math
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional

from quantark.util.exceptions import PricingError, ValidationError

__all__ = ["DaskBatchTotals", "run_autocallable_dask_batches"]


@dataclass(frozen=True)
class DaskBatchTotals:
    """Reduced batch statistics shared by the autocallable MC result classes."""

    price: float
    std_error: float
    num_paths: int
    ko_probability: float
    v0_probability: float
    v1_probability: float
    avg_ko_time: Optional[float]
    batches_used: int


def run_autocallable_dask_batches(
    *,
    num_batches: int,
    total_paths: int,
    batch_fn: Callable[..., Dict[str, float]],
    batch_kwargs: Mapping[str, object],
) -> DaskBatchTotals:
    """Fan ``batch_fn`` out over Dask delayed batches and reduce the totals.

    ``batch_fn`` is called as
    ``batch_fn(batch_id=..., batch_num_paths=..., **batch_kwargs)`` and must
    return the legacy per-batch dict with keys ``n``, ``sum_x``, ``sum_x2``,
    ``ko_count``, ``v0_count``, ``v1_count``, ``ko_time_sum``,
    ``ko_time_count``. Validation stays lazy: this function is only reached
    once the engine has entered its Dask path.
    """
    # Dask is optional; the engines verify availability before dispatching
    # here, so this import cannot fail on the reachable path.
    from dask import compute, delayed

    if num_batches <= 0:
        raise ValidationError(
            f"num_batches must be positive, got {num_batches}"
        )

    # Split total paths across batches so parallel mode matches serial mode
    total_paths_target = int(total_paths)
    base = total_paths_target // num_batches
    remainder = total_paths_target % num_batches
    batch_sizes = [
        (base + 1 if i < remainder else base) for i in range(num_batches)
    ]

    # Create delayed tasks for non-empty batches
    batch_results = []
    batches_used = 0
    for batch_id, batch_num_paths in enumerate(batch_sizes):
        if batch_num_paths <= 0:
            continue
        batches_used += 1
        batch_results.append(
            delayed(batch_fn)(
                batch_id=batch_id,
                batch_num_paths=batch_num_paths,
                **batch_kwargs,
            )
        )

    # Compute all batches in parallel
    results = compute(*batch_results)

    total_n = 0
    total_sum_x = 0.0
    total_sum_x2 = 0.0
    total_ko_count = 0
    total_v0_count = 0
    total_v1_count = 0
    total_ko_time_sum = 0.0
    total_ko_time_count = 0

    for res in results:
        total_n += int(res["n"])
        total_sum_x += float(res["sum_x"])
        total_sum_x2 += float(res["sum_x2"])
        total_ko_count += int(res.get("ko_count", 0))
        total_v0_count += int(res.get("v0_count", 0))
        total_v1_count += int(res.get("v1_count", 0))
        total_ko_time_sum += float(res.get("ko_time_sum", 0.0))
        total_ko_time_count += int(res.get("ko_time_count", 0))

    if total_n <= 0:
        raise PricingError("Dask parallel pricing produced zero simulated paths")

    price = total_sum_x / total_n

    if total_n > 1:
        sample_var = (total_sum_x2 - (total_sum_x * total_sum_x) / total_n) / (
            total_n - 1
        )
        sample_var = max(sample_var, 0.0)
        std_error = math.sqrt(sample_var) / math.sqrt(total_n)
    else:
        std_error = 0.0

    ko_probability = float(total_ko_count / total_n)
    v0_probability = float(total_v0_count / total_n)
    v1_probability = float(total_v1_count / total_n)

    avg_ko_time: Optional[float]
    if total_ko_time_count > 0:
        avg_ko_time = float(total_ko_time_sum / total_ko_time_count)
    else:
        avg_ko_time = None

    return DaskBatchTotals(
        price=float(price),
        std_error=float(std_error),
        num_paths=int(total_n),
        ko_probability=ko_probability,
        v0_probability=v0_probability,
        v1_probability=v1_probability,
        avg_ko_time=avg_ko_time,
        batches_used=batches_used,
    )
