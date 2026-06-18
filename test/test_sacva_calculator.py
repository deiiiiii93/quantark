"""SA-CVA calculator end-to-end tests (hand-computed)."""

import pytest

from quantark.sacva.calculator import SACVACalculator
from quantark.sacva.models.enums import RiskClass, RiskType
from quantark.sacva.models.sensitivity import CVASensitivity
from quantark.sacva.models.portfolio import CVAPortfolio
from quantark.util.exceptions import ValidationError


def _eq(bucket, s_cva, rtype=RiskType.DELTA):
    return CVASensitivity(risk_class=RiskClass.EQUITY, risk_type=rtype, bucket=bucket,
                          risk_factor=f"b{bucket}", s_cva=s_cva, name="n")


def test_delta_plus_vega_simple_sum():
    pf = CVAPortfolio([_eq(1, 1000.0), _eq(1, 1000.0, RiskType.VEGA)],
                      reporting_currency="USD")
    res = SACVACalculator().calculate(pf)
    assert res.delta_capital == pytest.approx(550.0, rel=1e-12)
    assert res.vega_capital == pytest.approx(780.0, rel=1e-12)
    assert res.total_capital == pytest.approx(1330.0, rel=1e-12)
    assert res.by_risk_class["EQUITY:DELTA"] == pytest.approx(550.0, rel=1e-12)


def test_multi_class_delta_simple_sum():
    com = CVASensitivity(risk_class=RiskClass.COMMODITY, risk_type=RiskType.DELTA,
                         bucket=1, risk_factor="c1", s_cva=1000.0)
    pf = CVAPortfolio([_eq(1, 1000.0), com], reporting_currency="USD")
    res = SACVACalculator().calculate(pf)
    assert res.delta_capital == pytest.approx(850.0, rel=1e-12)


def test_m_cva_scaling():
    pf = CVAPortfolio([_eq(1, 1000.0)], reporting_currency="USD")
    res = SACVACalculator(m_cva=1.5).calculate(pf)
    assert res.total_capital == pytest.approx(1.5 * 550.0, rel=1e-12)
    assert res.m_cva == 1.5
    # decomposition building block stays raw
    assert res.by_bucket["EQUITY:DELTA:1"] == pytest.approx(550.0, rel=1e-12)


def test_m_cva_below_one_rejected():
    with pytest.raises(ValidationError):
        SACVACalculator(m_cva=0.9)


def test_m_cva_nonfinite_rejected():
    with pytest.raises(ValidationError):
        SACVACalculator(m_cva=float("nan"))
    with pytest.raises(ValidationError):
        SACVACalculator(m_cva=True)
