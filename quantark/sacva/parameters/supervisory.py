"""Supervisory parameters for SA-CVA (MAR50.54-50.77).

All tables frozen via ``MappingProxyType``; values stored as decimals. Lookups
raise ``ValidationError`` for unknown keys (no silent default).
See ``quantark/sacva/doc/sacva_basel.md``.
"""

from types import MappingProxyType
from typing import Optional

from quantark.sacva.models.enums import CreditQuality
from quantark.util.exceptions import ValidationError

SACVA_VERSION = "Basel SA-CVA (MAR50, consolidated)"


def _sym(matrix: dict) -> dict:
    """Mirror an upper-triangular {(i,j): v} dict to a full symmetric dict."""
    full = {}
    for (i, j), v in matrix.items():
        full[(i, j)] = v
        full[(j, i)] = v
    return full


class SupervisoryParameters:
    """SA-CVA supervisory risk weights and correlations (MAR50.54-50.77)."""

    M_CVA_DEFAULT = 1.0           # MAR50.41
    R_HEDGE_DISALLOWANCE = 0.01   # MAR50.53(1)

    # ---- Interest rate (MAR50.54-50.58) ----
    GAMMA_IR = 0.5
    SPECIFIED_IR_BASE = frozenset({"USD", "EUR", "GBP", "AUD", "CAD", "SEK", "JPY"})
    _IR_SPEC_RW = MappingProxyType({1.0: 0.0111, 2.0: 0.0093, 5.0: 0.0074,
                                    10.0: 0.0074, 30.0: 0.0074})
    _IR_SPEC_INFL_RW = 0.0111
    _IR_SPEC_TENOR_CORR = MappingProxyType(_sym({
        (1.0, 2.0): 0.91, (1.0, 5.0): 0.72, (1.0, 10.0): 0.55, (1.0, 30.0): 0.31,
        (2.0, 5.0): 0.87, (2.0, 10.0): 0.72, (2.0, 30.0): 0.45,
        (5.0, 10.0): 0.91, (5.0, 30.0): 0.68, (10.0, 30.0): 0.83,
    }))
    _IR_INFL_CORR = 0.40
    IR_OTHER_RW = 0.0158
    IR_OTHER_CORR = 0.40
    IR_VEGA_RW = 1.00
    IR_VEGA_CORR = 0.40

    @classmethod
    def ir_specified_rw(cls, tenor: Optional[float] = None,
                        inflation: bool = False) -> float:
        if inflation:
            return cls._IR_SPEC_INFL_RW
        if tenor not in cls._IR_SPEC_RW:
            raise ValidationError(f"Unknown IR specified tenor: {tenor!r}")
        return cls._IR_SPEC_RW[tenor]

    @classmethod
    def ir_specified_corr(cls, tenor_a: Optional[float] = None,
                         tenor_b: Optional[float] = None,
                         inflation_a: bool = False,
                         inflation_b: bool = False) -> float:
        if inflation_a and inflation_b:
            return 1.0
        if inflation_a or inflation_b:
            return cls._IR_INFL_CORR
        for t in (tenor_a, tenor_b):
            if t not in cls._IR_SPEC_RW:
                raise ValidationError(f"Unknown IR specified tenor: {t!r}")
        if tenor_a == tenor_b:
            return 1.0
        key = (tenor_a, tenor_b)
        if key not in cls._IR_SPEC_TENOR_CORR:
            raise ValidationError(f"Unknown IR tenor pair: {key!r}")
        return cls._IR_SPEC_TENOR_CORR[key]

    FX_DELTA_RW = 0.11
    FX_VEGA_RW = 1.00
    GAMMA_FX = 0.6

    # ---- Counterparty credit spread (MAR50.63-50.65) ----
    _CPTY_RW_IG = MappingProxyType({"1a": 0.005, "1b": 0.010, 2: 0.050, 3: 0.030,
                                    4: 0.030, 5: 0.020, 6: 0.015, 7: 0.050, 8: 0.015})
    _CPTY_RW_HY = MappingProxyType({"1a": 0.020, "1b": 0.040, 2: 0.120, 3: 0.070,
                                    4: 0.085, 5: 0.055, 6: 0.050, 7: 0.120, 8: 0.050})
    _CPTY_GAMMA = MappingProxyType(_sym({
        (1, 1): 1.0, (1, 2): 0.10, (1, 3): 0.20, (1, 4): 0.25, (1, 5): 0.20, (1, 6): 0.15, (1, 7): 0.0, (1, 8): 0.45,
        (2, 2): 1.0, (2, 3): 0.05, (2, 4): 0.15, (2, 5): 0.20, (2, 6): 0.05, (2, 7): 0.0, (2, 8): 0.45,
        (3, 3): 1.0, (3, 4): 0.20, (3, 5): 0.25, (3, 6): 0.05, (3, 7): 0.0, (3, 8): 0.45,
        (4, 4): 1.0, (4, 5): 0.25, (4, 6): 0.05, (4, 7): 0.0, (4, 8): 0.45,
        (5, 5): 1.0, (5, 6): 0.05, (5, 7): 0.0, (5, 8): 0.45,
        (6, 6): 1.0, (6, 7): 0.0, (6, 8): 0.45,
        (7, 7): 1.0, (7, 8): 0.0,
        (8, 8): 1.0,
    }))

    @classmethod
    def cpty_rw(cls, bucket: int, quality: CreditQuality,
                sub_bucket: Optional[str] = None) -> float:
        if quality == CreditQuality.IG:
            table = cls._CPTY_RW_IG
        elif quality == CreditQuality.HY_NR:
            table = cls._CPTY_RW_HY
        else:
            raise ValidationError(f"Unknown credit quality: {quality!r}")
        if bucket == 1:
            if sub_bucket not in ("a", "b"):
                raise ValidationError("Counterparty bucket 1 requires sub_bucket 'a'/'b'")
            return table[f"1{sub_bucket}"]
        if bucket not in table:
            raise ValidationError(f"Unknown counterparty bucket: {bucket!r}")
        return table[bucket]

    @classmethod
    def cpty_gamma(cls, b: int, c: int) -> float:
        key = (b, c)
        if key not in cls._CPTY_GAMMA:
            raise ValidationError(f"Unknown counterparty bucket pair: {key!r}")
        return cls._CPTY_GAMMA[key]

    @staticmethod
    def cpty_rho_tenor(same_tenor: bool) -> float:
        return 1.0 if same_tenor else 0.90

    @staticmethod
    def cpty_rho_quality(same_quality: bool) -> float:
        return 1.0 if same_quality else 0.80

    # ---- Reference credit spread (MAR50.66-50.69) ----
    _REF_RW = MappingProxyType({
        1: 0.005, 2: 0.010, 3: 0.050, 4: 0.030, 5: 0.030, 6: 0.020, 7: 0.015,
        8: 0.020, 9: 0.040, 10: 0.120, 11: 0.070, 12: 0.085, 13: 0.055, 14: 0.050,
        15: 0.120, 16: 0.015, 17: 0.050,
    })
    _REF_SECTOR_GAMMA = MappingProxyType(_sym({
        (1, 1): 1.0, (1, 2): 0.75, (1, 3): 0.10, (1, 4): 0.20, (1, 5): 0.25, (1, 6): 0.20, (1, 7): 0.15,
        (2, 2): 1.0, (2, 3): 0.05, (2, 4): 0.15, (2, 5): 0.20, (2, 6): 0.15, (2, 7): 0.10,
        (3, 3): 1.0, (3, 4): 0.05, (3, 5): 0.15, (3, 6): 0.20, (3, 7): 0.05,
        (4, 4): 1.0, (4, 5): 0.20, (4, 6): 0.25, (4, 7): 0.05,
        (5, 5): 1.0, (5, 6): 0.25, (5, 7): 0.05,
        (6, 6): 1.0, (6, 7): 0.05,
        (7, 7): 1.0,
    }))

    @classmethod
    def refcredit_rw(cls, bucket: int) -> float:
        if bucket not in cls._REF_RW:
            raise ValidationError(f"Unknown reference-credit bucket: {bucket!r}")
        return cls._REF_RW[bucket]

    @staticmethod
    def _ref_sector(bucket: int) -> int:
        if 1 <= bucket <= 7:
            return bucket
        if 8 <= bucket <= 14:
            return bucket - 7
        raise ValidationError(f"bucket {bucket} has no sector")

    @staticmethod
    def _ref_quality(bucket: int) -> CreditQuality:
        return CreditQuality.IG if 1 <= bucket <= 7 else CreditQuality.HY_NR

    @classmethod
    def refcredit_gamma(cls, b: int, c: int) -> float:
        for x in (b, c):
            if x not in cls._REF_RW:
                raise ValidationError(f"Unknown reference-credit bucket: {x!r}")
        if b == c:
            return 1.0
        special = {15, 16, 17}
        if b in special or c in special:
            pair = frozenset({b, c})
            if pair == frozenset({16, 17}):
                return 0.75
            if (b in {16, 17} and 1 <= c <= 14) or (c in {16, 17} and 1 <= b <= 14):
                return 0.45
            return 0.0  # anything involving bucket 15, and 15/16, 15/17
        base = cls._REF_SECTOR_GAMMA[(cls._ref_sector(b), cls._ref_sector(c))]
        if cls._ref_quality(b) != cls._ref_quality(c):
            base /= 2.0
        return base

    # ---- Equity (MAR50.70-50.73) ----
    _EQ_RW = MappingProxyType({1: 0.55, 2: 0.60, 3: 0.45, 4: 0.55, 5: 0.30, 6: 0.35,
                               7: 0.40, 8: 0.50, 9: 0.70, 10: 0.50, 11: 0.70,
                               12: 0.15, 13: 0.25})
    _EQ_LARGE_CAP = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 12})

    @classmethod
    def equity_delta_rw(cls, bucket: int) -> float:
        if bucket not in cls._EQ_RW:
            raise ValidationError(f"Unknown equity bucket: {bucket!r}")
        return cls._EQ_RW[bucket]

    @classmethod
    def equity_vega_rw(cls, bucket: int) -> float:
        if bucket not in cls._EQ_RW:
            raise ValidationError(f"Unknown equity bucket: {bucket!r}")
        return 0.78 if bucket in cls._EQ_LARGE_CAP else 1.00

    @classmethod
    def equity_gamma(cls, b: int, c: int) -> float:
        for x in (b, c):
            if x not in cls._EQ_RW:
                raise ValidationError(f"Unknown equity bucket: {x!r}")
        if b == c:
            return 1.0
        if 11 in (b, c):
            return 0.0
        pair = frozenset({b, c})
        if pair == frozenset({12, 13}):
            return 0.75
        if (b in {12, 13} and 1 <= c <= 10) or (c in {12, 13} and 1 <= b <= 10):
            return 0.45
        if 1 <= b <= 10 and 1 <= c <= 10:
            return 0.15
        return 0.0

    # ---- Commodity (MAR50.74-50.77) ----
    _CO_RW = MappingProxyType({1: 0.30, 2: 0.35, 3: 0.60, 4: 0.80, 5: 0.40, 6: 0.45,
                               7: 0.20, 8: 0.35, 9: 0.25, 10: 0.35, 11: 0.50})

    @classmethod
    def commodity_delta_rw(cls, bucket: int) -> float:
        if bucket not in cls._CO_RW:
            raise ValidationError(f"Unknown commodity bucket: {bucket!r}")
        return cls._CO_RW[bucket]

    @classmethod
    def commodity_vega_rw(cls, bucket: int) -> float:
        if bucket not in cls._CO_RW:
            raise ValidationError(f"Unknown commodity bucket: {bucket!r}")
        return 1.00

    @classmethod
    def commodity_gamma(cls, b: int, c: int) -> float:
        for x in (b, c):
            if x not in cls._CO_RW:
                raise ValidationError(f"Unknown commodity bucket: {x!r}")
        if b == c:
            return 1.0
        if 11 in (b, c):
            return 0.0
        return 0.20
