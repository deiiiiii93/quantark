"""Reusable intra-bucket correlation builders for SA-CVA (MAR50.56, 50.65)."""

from quantark.sacva.models.sensitivity import CVASensitivity
from quantark.sacva.parameters.supervisory import SupervisoryParameters as SP


def counterparty_rho_name(a: CVASensitivity, b: CVASensitivity) -> float:
    """rho_name for counterparty credit (MAR50.65(4)-(5))."""
    if a.bucket == 8 or b.bucket == 8:  # qualified-index bucket
        if a.name == b.name:
            return 1.0 if a.index_series == b.index_series else 0.90
        return 0.80
    if a.name == b.name:
        return 1.0
    if (a.legal_entity_group is not None
            and a.legal_entity_group == b.legal_entity_group):
        return 0.90  # distinct but legally related
    return 0.50


def counterparty_rho(a: CVASensitivity, b: CVASensitivity) -> float:
    """rho_kl = rho_tenor * rho_name * rho_quality (MAR50.65(4))."""
    rho_t = SP.cpty_rho_tenor(a.tenor == b.tenor)
    rho_n = counterparty_rho_name(a, b)
    rho_q = SP.cpty_rho_quality(a.credit_quality == b.credit_quality)
    return rho_t * rho_n * rho_q


def ir_specified_rho(a: CVASensitivity, b: CVASensitivity) -> float:
    """Intra-bucket rho for IR specified currencies (MAR50.56(4))."""
    return SP.ir_specified_corr(
        tenor_a=a.tenor, tenor_b=b.tenor,
        inflation_a=a.is_inflation, inflation_b=b.is_inflation)
