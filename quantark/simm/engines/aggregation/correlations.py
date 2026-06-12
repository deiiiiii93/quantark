"""
Correlation lookups for SIMM aggregation.

Implements the intra-bucket correlations rho_kl and inter-bucket
correlations gamma_bc of ISDA SIMM v2.6, Sections D-I. Risk factors are
identified by the hashable ``risk_factor`` keys defined in
quantark.simm.sensitivity:

- IR delta: ("Yield", vertex, sub_curve) | ("Inflation",) | ("XCcyBasis",)
- IR vega: ("Vol", vertex) | ("InflationVol", vertex)
- Credit delta: (issuer, vertex, payment_currency); vega: (issuer, vertex)
- Equity / Commodity: (name,)
- FX delta: (currency,); FX vega: frozenset({ccy1, ccy2})
"""

from typing import Hashable, Union

from quantark.simm.taxonomy import (
    MarginType,
    RiskClass,
    get_fx_volatility_group,
    is_residual_bucket,
)
from quantark.simm.calibration.ir import (
    IR_TENOR_CORRELATIONS,
    IR_TENOR_INDEX,
    IR_SUB_CURVE_CORRELATION,
    IR_INFLATION_CORRELATION,
    IR_XCCY_BASIS_CORRELATION,
    IR_INTER_CURRENCY_CORRELATION,
)
from quantark.simm.calibration.credit_qualifying import (
    CREDIT_QUALIFYING_SAME_ISSUER_CORRELATION,
    CREDIT_QUALIFYING_DIFFERENT_ISSUER_CORRELATION,
    CREDIT_QUALIFYING_RESIDUAL_SAME_ISSUER_CORRELATION,
    CREDIT_QUALIFYING_RESIDUAL_DIFFERENT_ISSUER_CORRELATION,
    CREDIT_QUALIFYING_INTER_BUCKET_CORRELATIONS,
)
from quantark.simm.calibration.credit_non_qualifying import (
    CREDIT_NON_QUALIFYING_SAME_GROUP_CORRELATION,
    CREDIT_NON_QUALIFYING_DIFFERENT_GROUP_CORRELATION,
    CREDIT_NON_QUALIFYING_RESIDUAL_SAME_GROUP_CORRELATION,
    CREDIT_NON_QUALIFYING_RESIDUAL_DIFFERENT_GROUP_CORRELATION,
    CREDIT_NON_QUALIFYING_INTER_BUCKET_CORRELATION,
)
from quantark.simm.calibration.equity import (
    EQUITY_INTRA_BUCKET_CORRELATIONS,
    EQUITY_INTER_BUCKET_CORRELATIONS,
)
from quantark.simm.calibration.commodity import (
    COMMODITY_INTRA_BUCKET_CORRELATIONS,
    COMMODITY_INTER_BUCKET_CORRELATIONS,
)
from quantark.simm.calibration.fx import (
    FX_DELTA_CORRELATIONS,
    FX_VEGA_CORRELATION,
)
from quantark.util.exceptions import ValidationError


def _ir_tenor_rho(vertex_1: str, vertex_2: str) -> float:
    """Tenor correlation rho_kl from the 12x12 matrix (paragraph 36)."""
    try:
        i = IR_TENOR_INDEX[vertex_1]
        j = IR_TENOR_INDEX[vertex_2]
    except KeyError as exc:
        raise ValidationError(f"Unknown IR tenor vertex: {exc}") from exc
    return float(IR_TENOR_CORRELATIONS[i, j])


def _ir_intra_correlation(
    rf1: tuple, rf2: tuple, margin_type: MarginType
) -> float:
    """IR intra-currency correlation (paragraph 36)."""
    kind_1, kind_2 = rf1[0], rf2[0]

    # Cross-currency basis spread vs any yield or inflation rate: 4%.
    if "XCcyBasis" in (kind_1, kind_2):
        return IR_XCCY_BASIS_CORRELATION

    if margin_type == MarginType.DELTA:
        # Inflation vs any yield: 24%.
        if "Inflation" in (kind_1, kind_2):
            return IR_INFLATION_CORRELATION
        # Yield vs yield: phi_ij * rho_kl (paragraph 7(c)).
        rho = _ir_tenor_rho(rf1[1], rf2[1])
        if rf1[2] != rf2[2]:
            rho *= IR_SUB_CURVE_CORRELATION
        return rho
    else:
        # Vega / curvature factors: ("Vol", vertex) or ("InflationVol", vertex).
        if kind_1 != kind_2:
            # Inflation volatility vs interest-rate volatility: 24%.
            return IR_INFLATION_CORRELATION
        return _ir_tenor_rho(rf1[1], rf2[1])


def intra_bucket_correlation(
    risk_class: RiskClass,
    margin_type: MarginType,
    bucket: Union[str, int],
    rf1: Hashable,
    rf2: Hashable,
    group_1: str = "",
    group_2: str = "",
    calculation_currency: str = "USD",
) -> float:
    """Intra-bucket correlation rho_kl between two distinct risk factors.

    Args:
        risk_class: The risk class.
        margin_type: DELTA, VEGA or CURVATURE (vega correlations are used
            for curvature, per paragraph 11(c)).
        bucket: The bucket containing both risk factors.
        rf1, rf2: Risk factor keys (see module docstring).
        group_1, group_2: Group names for Credit Non-Qualifying factors
            (paragraph 48).
        calculation_currency: SIMM calculation currency (drives the FX
            delta correlation table choice, paragraph 72).

    Returns:
        The correlation rho_kl.
    """
    if rf1 == rf2:
        return 1.0

    residual = is_residual_bucket(bucket)

    if risk_class == RiskClass.INTEREST_RATE:
        return _ir_intra_correlation(rf1, rf2, margin_type)  # type: ignore[arg-type]

    elif risk_class == RiskClass.CREDIT_QUALIFYING:
        same_issuer = rf1[0] == rf2[0]  # type: ignore[index]
        if residual:
            return (
                CREDIT_QUALIFYING_RESIDUAL_SAME_ISSUER_CORRELATION
                if same_issuer
                else CREDIT_QUALIFYING_RESIDUAL_DIFFERENT_ISSUER_CORRELATION
            )
        return (
            CREDIT_QUALIFYING_SAME_ISSUER_CORRELATION
            if same_issuer
            else CREDIT_QUALIFYING_DIFFERENT_ISSUER_CORRELATION
        )

    elif risk_class == RiskClass.CREDIT_NON_QUALIFYING:
        same_group = (group_1 or str(rf1[0])) == (group_2 or str(rf2[0]))  # type: ignore[index]
        if residual:
            return (
                CREDIT_NON_QUALIFYING_RESIDUAL_SAME_GROUP_CORRELATION
                if same_group
                else CREDIT_NON_QUALIFYING_RESIDUAL_DIFFERENT_GROUP_CORRELATION
            )
        return (
            CREDIT_NON_QUALIFYING_SAME_GROUP_CORRELATION
            if same_group
            else CREDIT_NON_QUALIFYING_DIFFERENT_GROUP_CORRELATION
        )

    elif risk_class == RiskClass.EQUITY:
        key = "Residual" if residual else bucket
        return EQUITY_INTRA_BUCKET_CORRELATIONS[key]

    elif risk_class == RiskClass.COMMODITY:
        return COMMODITY_INTRA_BUCKET_CORRELATIONS[int(bucket)]

    elif risk_class == RiskClass.FX:
        if margin_type != MarginType.DELTA:
            # FX volatility and curvature risk factors (paragraph 73).
            return FX_VEGA_CORRELATION
        calc_group = get_fx_volatility_group(calculation_currency)
        g1 = get_fx_volatility_group(str(rf1[0]))  # type: ignore[index]
        g2 = get_fx_volatility_group(str(rf2[0]))  # type: ignore[index]
        return FX_DELTA_CORRELATIONS[calc_group][(g1, g2)]

    raise ValidationError(f"Unknown risk class: {risk_class}")


def inter_bucket_correlation(
    risk_class: RiskClass,
    bucket_1: Union[str, int],
    bucket_2: Union[str, int],
) -> float:
    """Inter-bucket correlation gamma_bc between two non-residual buckets.

    Args:
        risk_class: The risk class.
        bucket_1, bucket_2: Bucket identifiers (currencies for IR).

    Returns:
        The correlation gamma_bc.
    """
    if bucket_1 == bucket_2:
        return 1.0

    if risk_class == RiskClass.INTEREST_RATE:
        return IR_INTER_CURRENCY_CORRELATION  # paragraph 37

    elif risk_class == RiskClass.CREDIT_QUALIFYING:
        i, j = int(bucket_1) - 1, int(bucket_2) - 1
        return float(CREDIT_QUALIFYING_INTER_BUCKET_CORRELATIONS[i, j])

    elif risk_class == RiskClass.CREDIT_NON_QUALIFYING:
        return CREDIT_NON_QUALIFYING_INTER_BUCKET_CORRELATION  # paragraph 49

    elif risk_class == RiskClass.EQUITY:
        i, j = int(bucket_1) - 1, int(bucket_2) - 1
        return float(EQUITY_INTER_BUCKET_CORRELATIONS[i, j])

    elif risk_class == RiskClass.COMMODITY:
        i, j = int(bucket_1) - 1, int(bucket_2) - 1
        return float(COMMODITY_INTER_BUCKET_CORRELATIONS[i, j])

    elif risk_class == RiskClass.FX:
        # FX has a single bucket (paragraph 66); two distinct buckets
        # should never be encountered.
        raise ValidationError("FX has a single bucket; no gamma_bc applies")

    raise ValidationError(f"Unknown risk class: {risk_class}")
