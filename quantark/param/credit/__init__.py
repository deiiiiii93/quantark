"""Credit market-data parameters (hazard-rate curves)."""
from .hazard_curve import FlatHazardCurve, HazardCurve, ParallelShiftHazardCurve

__all__ = ["HazardCurve", "FlatHazardCurve", "ParallelShiftHazardCurve"]
