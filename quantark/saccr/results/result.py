"""SA-CCR calculation result with full regulatory decomposition.

Reference: Basel Committee SA-CCR document, paragraphs 128-129.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SACCRResult:
    """Result of an SA-CCR EAD calculation.

    The ``rc``, ``pfe``, ``multiplier``, ``addon_aggregate``,
    ``addon_by_asset_class`` and ``addon_by_hedging_set`` fields ALWAYS describe
    the primary computation path (the margined path for margined sets, the
    unmargined path otherwise). The margined EAD cap only affects the final scalar
    ``ead``; ``ead_uncapped`` is the primary-path EAD before the cap.

    Attributes:
        ead: final Exposure at Default = alpha x (RC + PFE), capped for margined sets.
        ead_uncapped: primary-path EAD before applying the margined cap.
        ead_capped: True if the unmargined cap bound the margined EAD.
        rc: Replacement Cost (primary path).
        pfe: Potential Future Exposure (primary path).
        multiplier: PFE multiplier (primary path).
        alpha: regulatory multiplier (1.4).
        addon_aggregate: sum of asset-class add-ons (primary path).
        addon_by_asset_class: add-on per asset class, keyed by enum name.
        addon_by_hedging_set: add-on per hedging set (e.g. "IR:USD", "COMMODITY:ENERGY").
        v: netting-set market value (V).
        c: netting-set collateral (C).
        nica: net independent collateral amount.
        is_margined: whether the netting set is margined.
    """

    ead: float
    ead_uncapped: float
    ead_capped: bool
    rc: float
    pfe: float
    multiplier: float
    alpha: float
    addon_aggregate: float
    addon_by_asset_class: Dict[str, float] = field(default_factory=dict)
    addon_by_hedging_set: Dict[str, float] = field(default_factory=dict)
    v: float = 0.0
    c: float = 0.0
    nica: float = 0.0
    is_margined: bool = False
