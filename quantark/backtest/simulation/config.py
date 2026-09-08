"""Configuration for the simulated-path ensemble backtest (spec 11).

Nothing here has a hidden default: a cache budget, gate tolerances and a
hedge strategy are required arguments.  A setting this package cannot yet
honour is absent from these objects rather than accepted and ignored, so a
config that is constructed is a config that runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.backtest.replay import AutocallableEngineConfig, HedgeSpec, ReplayProduct
from quantark.backtest.transaction_costs import TransactionCostModel
from quantark.util.exceptions import ValidationError

PROVIDERS = ("repricing",)  # "life_surface" joins this when that provider lands


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

    def __post_init__(self) -> None:
        if int(self.sample_states) < 0:
            raise ValidationError("GateConfig.sample_states must be non-negative")
        if float(self.pv_tolerance_bp) < 0.0 or float(self.delta_tolerance_hands) < 0.0:
            raise ValidationError("GateConfig tolerances must be non-negative")


@dataclass(frozen=True)
class CacheConfig:
    """State-cache budget (spec 7.4); the on-disk tier arrives with the ladder."""

    memory_bytes: int

    def __post_init__(self) -> None:
        if int(self.memory_bytes) <= 0:
            raise ValidationError("CacheConfig.memory_bytes must be positive")


@dataclass(frozen=True)
class PricingProviderConfig:
    """Which pricer runs and how much it may spend."""

    provider: Literal["repricing"]
    cache: CacheConfig
    gate: GateConfig

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValidationError(
                f"provider must be one of {PROVIDERS}, got {self.provider!r}"
            )


@dataclass
class EnsembleConfig:
    """A book, an engine, a hedge and a pricer: one cell of a simulated run.

    The rate comes from the ``MarketPath`` (per path, per day), so there is
    no rate schedule here.  Batch parallelism is not part of this
    configuration yet; the engine runs the whole batch in process.
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

    @property
    def quantities(self) -> np.ndarray:
        """Position sizes in product order; negative for the seller."""
        return np.array([float(bp.quantity) for bp in self.products])
