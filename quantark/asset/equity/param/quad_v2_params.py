"""Explicit controls for the opt-in Gaussian autocallable engine."""
from dataclasses import dataclass
import math

from .engine_params import EngineParams
from quantark.util.exceptions import ValidationError


@dataclass
class QuadV2Params(EngineParams):
    """Accuracy controls are refinement settings, not certified error bounds.

    ``backend`` changes application of a kernel, never its discretization.
    Memory budgets bound individual workspaces and the per-context cache.
    Floating event times are preserved unless a tolerance is explicitly set.
    """

    order: int = 8
    cells_per_sd: float = 2.0
    domain_sd: float = 11.0
    tail_sd: float = 10.0
    readout_order: int = 32
    backend: str = "auto"
    max_nodes: int = 100_000
    max_events: int = 4096
    max_states: int = 128
    max_work_bytes: int = 256 * 1024**2
    max_cache_bytes: int = 64 * 1024**2
    event_time_tolerance: float = 0.0
    continuous_steps_per_year: int = 48
    continuous_term_structure: str = "exact"

    def __post_init__(self):
        super().__post_init__()
        for name in (
            "order",
            "readout_order",
            "max_nodes",
            "max_events",
            "max_states",
            "max_work_bytes",
            "max_cache_bytes",
            "continuous_steps_per_year",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(f"{name} must be a positive integer")
        if not 4 <= self.order <= 16:
            raise ValidationError("order must be in [4, 16]")
        if not 16 <= self.readout_order <= 128:
            raise ValidationError("readout_order must be in [16, 128]")
        for name in ("cells_per_sd", "domain_sd", "tail_sd"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValidationError(f"{name} must be finite and positive")
        if self.tail_sd < 8 or self.domain_sd < self.tail_sd + 0.5:
            raise ValidationError("require tail_sd >= 8 and domain_sd >= tail_sd + 0.5")
        if self.backend not in {"auto", "direct", "fft"}:
            raise ValidationError("backend must be auto, direct or fft")
        if self.continuous_term_structure not in {"exact", "piecewise_constant"}:
            raise ValidationError(
                "continuous_term_structure must be exact or piecewise_constant"
            )
        if (
            not math.isfinite(self.event_time_tolerance)
            or self.event_time_tolerance < 0
        ):
            raise ValidationError("event_time_tolerance must be finite and nonnegative")
