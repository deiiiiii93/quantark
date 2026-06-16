"""SA-CVA optional qualified-index treatment tests (MAR50.50)."""

import math
import pytest

from quantark.sacva import (
    SACVACalculator, CVAPortfolio, CVASensitivity, RiskClass, RiskType,
    CreditQuality, SACVA_VERSION,
)


def test_public_api_exports():
    assert SACVA_VERSION.startswith("Basel SA-CVA")


def test_counterparty_index_bucket8_series_corr():
    a = CVASensitivity(risk_class=RiskClass.COUNTERPARTY_CREDIT, risk_type=RiskType.DELTA,
                       bucket=8, risk_factor="CDX:s1:5y", s_cva=1000.0, name="CDX",
                       tenor=5.0, credit_quality=CreditQuality.IG, is_index=True,
                       index_series="S1")
    b = CVASensitivity(risk_class=RiskClass.COUNTERPARTY_CREDIT, risk_type=RiskType.DELTA,
                       bucket=8, risk_factor="CDX:s2:5y", s_cva=500.0, name="CDX",
                       tenor=5.0, credit_quality=CreditQuality.IG, is_index=True,
                       index_series="S2")
    pf = CVAPortfolio([a, b], reporting_currency="USD")
    res = SACVACalculator().calculate(pf)
    rw = 0.015
    ws1, ws2 = rw*1000.0, rw*500.0
    rho = 1.0 * 0.90 * 1.0  # tenor same, same-index-distinct-series, same quality
    kb = math.sqrt(ws1**2 + ws2**2 + 2*rho*ws1*ws2)
    assert res.by_bucket["COUNTERPARTY_CREDIT:DELTA:8"] == pytest.approx(kb, rel=1e-9)


def test_equity_index_buckets_12_13():
    a = CVASensitivity(risk_class=RiskClass.EQUITY, risk_type=RiskType.DELTA, bucket=12,
                       risk_factor="idx12", s_cva=1000.0, is_index=True)
    b = CVASensitivity(risk_class=RiskClass.EQUITY, risk_type=RiskType.DELTA, bucket=13,
                       risk_factor="idx13", s_cva=1000.0, is_index=True)
    pf = CVAPortfolio([a, b], reporting_currency="USD")
    res = SACVACalculator().calculate(pf)
    ws1, ws2 = 0.15*1000.0, 0.25*1000.0
    assert res.by_risk_class["EQUITY:DELTA"] == pytest.approx(
        math.sqrt(ws1**2 + ws2**2 + 2*0.75*ws1*ws2), rel=1e-9)


def test_reference_credit_index_buckets_16_17():
    a = CVASensitivity(risk_class=RiskClass.REFERENCE_CREDIT, risk_type=RiskType.DELTA,
                       bucket=16, risk_factor="idx16", s_cva=1000.0, is_index=True)
    b = CVASensitivity(risk_class=RiskClass.REFERENCE_CREDIT, risk_type=RiskType.DELTA,
                       bucket=17, risk_factor="idx17", s_cva=1000.0, is_index=True)
    pf = CVAPortfolio([a, b], reporting_currency="USD")
    res = SACVACalculator().calculate(pf)
    ws1, ws2 = 0.015*1000.0, 0.050*1000.0
    assert res.by_risk_class["REFERENCE_CREDIT:DELTA"] == pytest.approx(
        math.sqrt(ws1**2 + ws2**2 + 2*0.75*ws1*ws2), rel=1e-9)
