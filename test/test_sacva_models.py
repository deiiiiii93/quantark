"""SA-CVA model validation tests."""

import pytest
from quantark.sacva.models.enums import RiskClass, RiskType, CreditQuality
from quantark.util.exceptions import ValidationError


def test_enum_members():
    assert {c.name for c in RiskClass} == {
        "INTEREST_RATE", "FX", "COUNTERPARTY_CREDIT",
        "REFERENCE_CREDIT", "EQUITY", "COMMODITY",
    }
    assert {t.name for t in RiskType} == {"DELTA", "VEGA"}
    assert {q.name for q in CreditQuality} == {"IG", "HY_NR"}


from quantark.sacva.models.sensitivity import CVASensitivity
from quantark.sacva.models.portfolio import CVAPortfolio


def _eq(bucket=5, **kw):
    base = dict(risk_class=RiskClass.EQUITY, risk_type=RiskType.DELTA,
                bucket=bucket, risk_factor="EQ", s_cva=100.0, s_hdg=0.0,
                name="ACME")
    base.update(kw)
    return CVASensitivity(**base)


def test_sensitivity_valid():
    s = _eq()
    assert s.s_cva == 100.0 and s.bucket == 5


def test_sensitivity_type_validation():
    with pytest.raises(ValidationError):
        CVASensitivity(risk_class="EQUITY", risk_type=RiskType.DELTA, bucket=5,
                       risk_factor="x", s_cva=1.0, s_hdg=0.0)
    with pytest.raises(ValidationError):
        _eq(s_cva=float("nan"))
    with pytest.raises(ValidationError):
        _eq(is_index="yes")


def test_equity_index_bucket_requires_is_index():
    with pytest.raises(ValidationError):
        _eq(bucket=12, is_index=False)
    _eq(bucket=12, is_index=True)
    with pytest.raises(ValidationError):
        _eq(bucket=5, is_index=True)  # non-index bucket flagged as index


def test_counterparty_requires_discriminators():
    ok = dict(risk_class=RiskClass.COUNTERPARTY_CREDIT, risk_type=RiskType.DELTA,
              bucket=3, risk_factor="A:5y", s_cva=1.0, s_hdg=0.0,
              name="A", tenor=5.0, credit_quality=CreditQuality.IG)
    CVASensitivity(**ok)
    bad = dict(ok); bad.pop("credit_quality")
    with pytest.raises(ValidationError):
        CVASensitivity(**bad)
    bad2 = dict(ok); bad2["bucket"] = 1  # bucket 1 needs sub_bucket
    with pytest.raises(ValidationError):
        CVASensitivity(**bad2)
    CVASensitivity(**{**ok, "bucket": 1, "sub_bucket": "a"})


def test_counterparty_no_vega():
    with pytest.raises(ValidationError):
        CVASensitivity(risk_class=RiskClass.COUNTERPARTY_CREDIT, risk_type=RiskType.VEGA,
                       bucket=3, risk_factor="A", s_cva=1.0, s_hdg=0.0,
                       name="A", tenor=5.0, credit_quality=CreditQuality.IG)


def test_counterparty_index_only_bucket8():
    with pytest.raises(ValidationError):
        CVASensitivity(risk_class=RiskClass.COUNTERPARTY_CREDIT, risk_type=RiskType.DELTA,
                       bucket=3, risk_factor="A", s_cva=1.0, name="A", tenor=5.0,
                       credit_quality=CreditQuality.IG, is_index=True, index_series="S1")


def test_stray_sub_bucket_and_index_series_rejected():
    with pytest.raises(ValidationError):
        _eq(sub_bucket="a")  # equity cannot carry sub_bucket
    with pytest.raises(ValidationError):
        _eq(index_series="S1")  # index_series without is_index


def test_bucket_and_tenor_type_validation():
    with pytest.raises(ValidationError):
        _eq(bucket=True)
    with pytest.raises(ValidationError):
        CVASensitivity(risk_class=RiskClass.COUNTERPARTY_CREDIT, risk_type=RiskType.DELTA,
                       bucket=3, risk_factor="A", s_cva=1.0, name="A",
                       tenor=float("nan"), credit_quality=CreditQuality.IG)


def test_counterparty_tenor_must_be_on_grid():
    ok = dict(risk_class=RiskClass.COUNTERPARTY_CREDIT, risk_type=RiskType.DELTA,
              bucket=3, risk_factor="A", s_cva=1.0, name="A",
              credit_quality=CreditQuality.IG)
    CVASensitivity(tenor=5.0, **ok)
    with pytest.raises(ValidationError):
        CVASensitivity(tenor=7.0, **ok)   # off the {0.5,1,3,5,10} grid


def test_counterparty_bucket8_requires_index_series():
    base = dict(risk_class=RiskClass.COUNTERPARTY_CREDIT, risk_type=RiskType.DELTA,
                bucket=8, risk_factor="CDX", s_cva=1.0, name="CDX", tenor=5.0,
                credit_quality=CreditQuality.IG, is_index=True)
    with pytest.raises(ValidationError):
        CVASensitivity(**base)            # no index_series
    CVASensitivity(index_series="S1", **base)


def test_reference_credit_quality_consistency():
    with pytest.raises(ValidationError):
        CVASensitivity(risk_class=RiskClass.REFERENCE_CREDIT, risk_type=RiskType.DELTA,
                       bucket=1, risk_factor="b1", s_cva=1.0, name="a",
                       credit_quality=CreditQuality.HY_NR)  # bucket 1 is IG
    CVASensitivity(risk_class=RiskClass.REFERENCE_CREDIT, risk_type=RiskType.DELTA,
                   bucket=1, risk_factor="b1", s_cva=1.0, name="a",
                   credit_quality=CreditQuality.IG)
    # bucket 15 (other) has no defined quality -> any value accepted
    CVASensitivity(risk_class=RiskClass.REFERENCE_CREDIT, risk_type=RiskType.DELTA,
                   bucket=15, risk_factor="b15", s_cva=1.0, name="a",
                   credit_quality=CreditQuality.HY_NR)


def test_portfolio_fx_reporting_currency_excluded():
    fx = CVASensitivity(risk_class=RiskClass.FX, risk_type=RiskType.DELTA, bucket=0,
                        currency="USD", risk_factor="USD", s_cva=1.0, s_hdg=0.0)
    with pytest.raises(ValidationError):
        CVAPortfolio([fx], reporting_currency="USD")
    CVAPortfolio([fx], reporting_currency="EUR")


def test_portfolio_requires_sensitivities():
    with pytest.raises(ValidationError):
        CVAPortfolio([], reporting_currency="USD")
