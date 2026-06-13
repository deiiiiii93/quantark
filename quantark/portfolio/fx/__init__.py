"""
FX portfolio subpackage.

Provides FX-specific position and portfolio implementations conforming to the
asset-agnostic ``BasePosition`` / ``BasePortfolio`` protocols, and to the SIMM
``SIMMSensitivityProvider`` protocol (``FXPosition.get_simm_sensitivities``).
"""
from .portfolio import FXPortfolio
from .position import FXPosition

__all__ = ["FXPosition", "FXPortfolio"]
