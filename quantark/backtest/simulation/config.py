"""Configuration for the simulated-path ensemble backtest (spec 11).

Nothing here has a hidden default: a cache budget, gate tolerances and a
hedge strategy are required arguments.  A setting this package cannot yet
honour is absent from these objects rather than accepted and ignored, so a
config that is constructed is a config that runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.backtest.replay import AutocallableEngineConfig, HedgeSpec, ReplayProduct
from quantark.backtest.transaction_costs import TransactionCostModel
from quantark.util.exceptions import ValidationError

PROVIDERS = ("repricing", "life_surface")
MODES = ("exact", "ladder", "life_surface")


@dataclass(frozen=True)
class GateConfig:
    """Accuracy budget for the pricing provider (spec 7.5).

    In exact repricing mode the gap is identically zero and the report says
    so; the sampling gate that consumes ``sample_states`` arrives with the
    approximate providers.
    """

    sample_states: int
    pv_tolerance_bp: float
    delta_tolerance_hands: float
    #: A second delta budget, as a fraction of the state's OWN exact delta,
    #: taken whenever it is the larger of the two.  Hands alone are
    #: calibrated on a typical state; beside a barrier near expiry the delta
    #: is an order of magnitude above that and neither the provider nor the
    #: exact solve it is scored against is converged, so an absolute rule
    #: there measures the mesh rather than the provider.  0.0 leaves hands
    #: as the only rule.
    delta_tolerance_rel: float = 0.0
    #: Signed fractions of the knock-in barrier the designed stress set
    #: places spots at.  Whether a run's reservoir ever lands beside the
    #: barrier is luck; these states are visited whether it does or not.
    #: An empty tuple turns the stress set off.
    barrier_offsets: Tuple[float, ...] = (-0.01, -0.002, 0.002, 0.01)
    #: How many knock-in observation dates the stress set probes.
    barrier_dates: int = 2

    def __post_init__(self) -> None:
        if int(self.sample_states) < 0:
            raise ValidationError("GateConfig.sample_states must be non-negative")
        if float(self.pv_tolerance_bp) < 0.0 or float(self.delta_tolerance_hands) < 0.0:
            raise ValidationError("GateConfig tolerances must be non-negative")
        rel = float(self.delta_tolerance_rel)
        if not np.isfinite(rel) or rel < 0.0:
            raise ValidationError(
                "GateConfig.delta_tolerance_rel must be a finite non-negative fraction"
            )
        offsets = tuple(float(x) for x in self.barrier_offsets)
        if any(x == 0.0 or not np.isfinite(x) or x <= -1.0 for x in offsets):
            raise ValidationError(
                "GateConfig.barrier_offsets must be finite, non-zero and above -100%"
            )
        object.__setattr__(self, "barrier_offsets", offsets)
        if int(self.barrier_dates) < 0:
            raise ValidationError("GateConfig.barrier_dates must be non-negative")

    def delta_allowance(self, exact_delta_hands: float) -> float:
        """The delta budget ONE state is judged against, in hands."""
        return max(float(self.delta_tolerance_hands),
                   float(self.delta_tolerance_rel) * abs(float(exact_delta_hands)))

    def passes_delta(self, *, gap_hands: float, exact_delta_hands: float) -> bool:
        """Whether one state's delta gap is inside its own budget."""
        return abs(float(gap_hands)) <= self.delta_allowance(exact_delta_hands)


@dataclass(frozen=True)
class CacheConfig:
    """State-cache budget (spec 7.4).

    ``disk_dir`` names the on-disk tier; ``None`` means memory only.  A
    shard written by another library version or engine is a miss, never
    reinterpreted (see ``pricing.cache.DiskTier``).
    """

    memory_bytes: int
    disk_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if int(self.memory_bytes) <= 0:
            raise ValidationError("CacheConfig.memory_bytes must be positive")
        if self.disk_dir is not None and not str(self.disk_dir):
            raise ValidationError("CacheConfig.disk_dir must be a directory path or None")


@dataclass(frozen=True)
class PricingProviderConfig:
    """Which pricer runs, how coarsely, and how much it may spend.

    ``spot_step`` (log-spot) switches the repricing provider from exact to
    ladder mode; ``vol_step`` / ``q_step`` bucket vol and the flat dividend
    yield in either approximate mode (0 keeps them exact); the life
    surface additionally needs ``surface_cache_bytes``.  A step that a
    mode cannot honour is rejected, not ignored.
    """

    provider: Literal["repricing", "life_surface"]
    cache: CacheConfig
    gate: GateConfig
    spot_step: Optional[float] = None
    vol_step: Optional[float] = None
    q_step: Optional[float] = None
    surface_cache_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValidationError(f"provider must be one of {PROVIDERS}, got {self.provider!r}")
        steps_given = self.vol_step is not None or self.q_step is not None
        if self.provider == "repricing":
            if self.surface_cache_bytes is not None:
                raise ValidationError("surface_cache_bytes belongs to the life_surface provider")
            if self.spot_step is None:
                if steps_given:
                    raise ValidationError(
                        "vol_step / q_step have no meaning in exact repricing mode; set spot_step for the ladder"
                    )
                return
            if float(self.spot_step) <= 0.0:
                raise ValidationError("spot_step must be positive (log-spot) or None for exact mode")
            self._require_bucket_steps("ladder")
            return
        if self.spot_step is not None:
            raise ValidationError("spot_step has no meaning for the life_surface provider")
        self._require_bucket_steps("life_surface")
        if self.surface_cache_bytes is None or int(self.surface_cache_bytes) <= 0:
            raise ValidationError("life_surface requires a positive surface_cache_bytes")

    def _require_bucket_steps(self, mode: str) -> None:
        if self.vol_step is None or self.q_step is None:
            raise ValidationError(f"{mode} mode requires vol_step and q_step (0 means exact)")
        if float(self.vol_step) < 0.0 or float(self.q_step) < 0.0:
            raise ValidationError("vol_step and q_step must be non-negative")

    @property
    def mode(self) -> str:
        """``exact`` | ``ladder`` | ``life_surface``."""
        if self.provider == "life_surface":
            return "life_surface"
        return "exact" if self.spot_step is None else "ladder"


@dataclass
class EnsembleConfig:
    """A book, an engine, a hedge and a pricer: one cell of a simulated run.

    The rate comes from the ``MarketPath`` (per path, per day), so there is
    no rate schedule here.  ``workers`` and ``batch_paths`` split the batch
    into index ranges for ``run_ensemble``; the split is bit-inert.
    """

    products: List[ReplayProduct]
    engine_config: AutocallableEngineConfig
    hedge: HedgeSpec
    strategy: Any
    transaction_cost_model: TransactionCostModel
    pricing: PricingProviderConfig
    underlying: str = "equity_index"
    delta_bump_size: Optional[float] = None
    gamma_bump_size: Optional[float] = None
    allow_data_end: bool = False
    workers: int = 1
    batch_paths: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.products:
            raise ValidationError("EnsembleConfig needs at least one product")
        for bp in self.products:
            if not isinstance(bp.product, SnowballOption):
                raise ValidationError(
                    "the simulated-path engine prices standard snowballs only, got "
                    f"{type(bp.product).__name__} for position {bp.position_id}"
                )
            if not bp.has_lifecycle:
                raise ValidationError(
                    f"position {bp.position_id} must carry a lifecycle in a simulated run"
                )
        if self.hedge.kind != "futures":
            raise ValidationError(
                f"the simulated-path engine hedges with futures only, got {self.hedge.kind!r}"
            )
        if self.strategy is None:
            raise ValidationError("EnsembleConfig.strategy is required")
        from quantark.backtest.strategy.futures_bucket_strategy import (
            FuturesBucketHedgeStrategy,
        )

        if isinstance(self.strategy, FuturesBucketHedgeStrategy):
            # Rejected BEFORE the scalar sizing path below, which would call
            # a multi-leg strategy through the single-contract protocol.
            raise ValidationError(
                "the simulated-path engine sizes one contract per day; a "
                "multi-leg bucket hedge needs the historical replay, whose "
                "actual futures chain supplies its risk coordinates"
            )
        source = getattr(self.engine_config, "dividend_source", None)
        if source not in (None, "active_contract", "futures_curve"):
            raise ValidationError(
                f"dividend_source {source!r} has no meaning on a simulated path; "
                "use None, 'active_contract' or 'futures_curve'"
            )
        if getattr(self.engine_config, "vol_source", "scalar") != "scalar":
            raise ValidationError(
                "a simulated path carries one ATM vol per day: vol_source must be 'scalar'"
            )
        if int(self.workers) < 1:
            raise ValidationError("workers must be at least 1")
        if self.batch_paths is not None and int(self.batch_paths) < 1:
            raise ValidationError("batch_paths must be at least 1 or None (one batch)")
        if self.pricing.provider == "life_surface":
            from quantark.util.enum.engine_enums import EngineType

            if self.engine_config.pricing_engine_type != EngineType.PDE:
                raise ValidationError(
                    "the life_surface provider needs a PDE engine; use provider='repricing' for "
                    f"{self.engine_config.pricing_engine_type!r}"
                )

    @property
    def quantities(self) -> np.ndarray:
        """Position sizes in product order; negative for the seller."""
        return np.array([float(bp.quantity) for bp in self.products])
