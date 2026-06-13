"""Credit derivatives asset class (CDS, basket CDS)."""
from quantark.asset.credit.product import CDS, ProtectionSide
from quantark.asset.credit.engine.analytical import CDSConstantHazardEngine

__all__ = ["CDS", "ProtectionSide", "CDSConstantHazardEngine"]
