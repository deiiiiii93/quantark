"""
SIMM Calibration Version Information

This module contains version information for the SIMM (Standard Initial Margin Model)
implementation. SIMM is a methodology for calculating initial margin for non-cleared
derivatives as specified by ISDA.
"""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class SIMMVersion:
    """SIMM version information.

    This class tracks the version of SIMM calibration parameters implemented.
    All parameters in the calibration module follow the ISDA SIMM v2.6 specification.

    Attributes:
        version: The current SIMM version (v2.6)
        base_version: The base SIMM version this builds from
        effective_date: Date when v2.6 became effective (December 2, 2023)
        publication_date: Date when v2.6 was published (August 16, 2023)
    """
    version: str = "2.6"
    base_version: str = "2.5.6"
    effective_date: date = date(2023, 12, 2)
    publication_date: date = date(2023, 8, 16)


# The current version of SIMM calibration parameters
CURRENT_VERSION = SIMMVersion()
