"""Regression tests for fail-closed SIMM correctness behavior."""

import pytest

from quantark.simm import (
    CurvatureSensitivity,
    EquityDeltaSensitivity,
    EquityVegaSensitivity,
    MappingFXRateProvider,
    RiskClass,
    SIMMConfig,
    SIMMMarketData,
    SensitivityCollection,
)
from quantark.simm.engines.aggregation import SIMMCalculator
from quantark.simm.engines.factory import create_all_engines
from quantark.simm.engines.factory import create_engine
from quantark.simm.engines.portfolio_adapter import SIMMPortfolioAdapter
from quantark.util.exceptions import ValidationError


def test_all_six_risk_classes_have_engines():
    assert set(create_all_engines(SIMMConfig())) == set(RiskClass)


def test_amounts_and_usd_thresholds_convert_to_calculation_currency():
    usd = SensitivityCollection([
        EquityDeltaSensitivity("T", 48e6, issuer="BIG", bucket_number=5)
    ])
    eur = SensitivityCollection([
        EquityDeltaSensitivity(
            "T", 24e6, amount_currency="EUR", issuer="BIG", bucket_number=5
        )
    ])
    market_data = SIMMMarketData(
        MappingFXRateProvider({("EUR", "USD"): 2.0})
    )
    usd_result = SIMMCalculator().calculate(usd)
    eur_result = SIMMCalculator(
        SIMMConfig(calculation_currency="EUR"), market_data
    ).calculate(eur)
    assert eur_result.total_margin == pytest.approx(usd_result.total_margin / 2.0)


def test_missing_fx_rate_provider_fails_closed():
    sensitivities = SensitivityCollection([
        EquityDeltaSensitivity(
            "T", 100.0, amount_currency="EUR", issuer="A", bucket_number=5
        )
    ])
    with pytest.raises(ValidationError, match="SIMMMarketData required"):
        SIMMCalculator().calculate(sensitivities)


def test_empty_non_usd_calculation_does_not_require_fx_rates():
    result = SIMMCalculator(SIMMConfig(calculation_currency="EUR")).calculate([])
    assert result.total_margin == 0.0


def test_mixed_explicit_and_derived_curvature_is_rejected():
    sensitivities = SensitivityCollection([
        EquityVegaSensitivity("T", 100.0, issuer="A", bucket_number=5),
        CurvatureSensitivity(
            "T",
            10.0,
            risk_class_value=RiskClass.EQUITY,
            qualifier_value="A",
            bucket_value=5,
            risk_factor_value=("A",),
        ),
    ])
    with pytest.raises(ValidationError, match="cannot be combined"):
        SIMMCalculator().calculate(sensitivities)


def test_portfolio_adapter_rejects_positions_without_provider():
    class Unsupported:
        position_id = "unsupported"

    portfolio = {"positions": {"unsupported": Unsupported()}}
    with pytest.raises(ValidationError, match="does not implement"):
        SIMMPortfolioAdapter(SIMMConfig()).portfolio_to_sensitivities(portfolio)


def test_provider_engine_rejects_positions_without_provider():
    with pytest.raises(ValidationError, match="does not implement"):
        create_engine(RiskClass.FX, SIMMConfig()).calculate_sensitivities(
            [object()], {}, SIMMConfig()
        )


def test_crif_amount_usd_is_preferred_for_usd_calculation():
    records = [{
        "TradeID": "T",
        "ValuationDate": "2026-06-13",
        "RiskType": "Risk_Equity",
        "Qualifier": "A",
        "Bucket": "5",
        "Amount": 50.0,
        "AmountCurrency": "EUR",
        "AmountUSD": 100.0,
    }]
    from_crif = SIMMCalculator().calculate_from_crif(records)
    direct = SIMMCalculator().calculate(SensitivityCollection([
        EquityDeltaSensitivity("T", 100.0, issuer="A", bucket_number=5)
    ]))
    assert from_crif.total_margin == pytest.approx(direct.total_margin)


def test_crif_amount_usd_is_validated_when_fx_rates_are_available():
    records = [{
        "TradeID": "T",
        "ValuationDate": "2026-06-13",
        "RiskType": "Risk_Equity",
        "Qualifier": "A",
        "Bucket": "5",
        "Amount": 50.0,
        "AmountCurrency": "EUR",
        "AmountUSD": 99.0,
    }]
    market_data = SIMMMarketData(
        MappingFXRateProvider({("EUR", "USD"): 2.0})
    )
    with pytest.raises(ValidationError, match="inconsistent"):
        SIMMCalculator(market_data=market_data).calculate_from_crif(records)
