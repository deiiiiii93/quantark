"""Tests for the SIMM aggregation engine (ISDA SIMM v2.6).

Expected values are hand-computed from the methodology formulas
(paragraphs 5-13 and the calibration tables of Sections D-K).
"""
import math

import pytest

from quantark.simm.config import SIMMConfig
from quantark.simm.taxonomy import (
    IRSubCurve,
    MarginType,
    ProductClass,
    RiskClass,
)
from quantark.simm.sensitivity import (
    BaseCorrSensitivity,
    CreditDeltaSensitivity,
    EquityDeltaSensitivity,
    EquityVegaSensitivity,
    FXDeltaSensitivity,
    FXVegaSensitivity,
    IRDeltaSensitivity,
    IRInflationDeltaSensitivity,
    IRVegaSensitivity,
    IRXCcyBasisSensitivity,
    SensitivityCollection,
)
from quantark.simm.engines.aggregation import (
    ConcentrationCalculator,
    SIMMCalculator,
    net_by_risk_factor,
)
from quantark.simm.engines.aggregation.weighted_sensitivity import (
    delta_risk_weight,
    vega_risk_weight,
)
from quantark.simm.engines.aggregation.correlations import (
    inter_bucket_correlation,
    intra_bucket_correlation,
)
from quantark.simm.calibration.accessors import PHI_INV_995, scaling_function
from quantark.simm.calibration.ir import IR_HVR


def _calc(config=None):
    return SIMMCalculator(config or SIMMConfig())


class TestNetting:
    def test_ir_nets_by_tenor_and_subcurve(self):
        sens = [
            IRDeltaSensitivity("T1", 100.0, currency="USD", tenor=5.0, sub_curve=IRSubCurve.OIS),
            IRDeltaSensitivity("T2", -40.0, currency="USD", tenor=5.0, sub_curve=IRSubCurve.OIS),
            IRDeltaSensitivity("T3", 70.0, currency="USD", tenor=5.0, sub_curve=IRSubCurve.LIBOR_3M),
            IRDeltaSensitivity("T4", 30.0, currency="USD", tenor=10.0, sub_curve=IRSubCurve.OIS),
        ]
        netted = {n.risk_factor: n.amount for n in net_by_risk_factor(sens, MarginType.DELTA)}
        assert len(netted) == 3
        assert netted[("Yield", "5y", "OIS")] == pytest.approx(60.0)
        assert netted[("Yield", "5y", "Libor3m")] == pytest.approx(70.0)
        assert netted[("Yield", "10y", "OIS")] == pytest.approx(30.0)

    def test_equity_vega_netting_applies_hvr(self):
        # Paragraph 10(c): VR_ik = HVR * sigma * vega; HVR_equity = 0.6.
        sens = [
            EquityVegaSensitivity("T1", 1000.0, issuer="AAPL", bucket_number=8, option_tenor=1.0),
            EquityVegaSensitivity("T2", 500.0, issuer="AAPL", bucket_number=8, option_tenor=5.0),
        ]
        netted = net_by_risk_factor(sens, MarginType.VEGA)
        assert len(netted) == 1  # expiries net within the same risk factor
        assert netted[0].amount == pytest.approx(0.6 * 1500.0)


class TestRiskWeights:
    def test_ir_delta_weights_by_currency_group(self):
        rf = ("Yield", "5y", "OIS")
        assert delta_risk_weight(RiskClass.INTEREST_RATE, "USD", rf) == 60
        assert delta_risk_weight(RiskClass.INTEREST_RATE, "JPY", rf) == 23
        assert delta_risk_weight(RiskClass.INTEREST_RATE, "BRL", rf) == 97

    def test_ir_inflation_and_xccy_weights(self):
        assert delta_risk_weight(RiskClass.INTEREST_RATE, "USD", ("Inflation",)) == 61
        assert delta_risk_weight(RiskClass.INTEREST_RATE, "USD", ("XCcyBasis",)) == 21

    def test_fx_weight_depends_on_groups_and_calc_currency(self):
        # Paragraph 69: regular/regular 7.4; high given currency 14.7.
        assert delta_risk_weight(RiskClass.FX, 1, ("EUR",), "USD") == 7.4
        assert delta_risk_weight(RiskClass.FX, 1, ("BRL",), "USD") == 14.7
        assert delta_risk_weight(RiskClass.FX, 1, ("RUB",), "TRY") == 21.4
        # No FX risk factor for the calculation currency itself.
        assert delta_risk_weight(RiskClass.FX, 1, ("USD",), "USD") == 0.0

    def test_vega_risk_weights(self):
        assert vega_risk_weight(RiskClass.INTEREST_RATE, "USD") == 0.23
        assert vega_risk_weight(RiskClass.CREDIT_QUALIFYING, 1) == 0.76
        assert vega_risk_weight(RiskClass.EQUITY, 5) == 0.45
        assert vega_risk_weight(RiskClass.EQUITY, 12) == 0.96
        assert vega_risk_weight(RiskClass.COMMODITY, 2) == 0.55
        assert vega_risk_weight(RiskClass.FX, 1) == 0.48


class TestCorrelations:
    def test_ir_yield_yield_same_subcurve(self):
        rho = intra_bucket_correlation(
            RiskClass.INTEREST_RATE, MarginType.DELTA, "USD",
            ("Yield", "1y", "OIS"), ("Yield", "2y", "OIS"))
        assert rho == pytest.approx(0.94)

    def test_ir_yield_yield_cross_subcurve_applies_phi(self):
        rho = intra_bucket_correlation(
            RiskClass.INTEREST_RATE, MarginType.DELTA, "USD",
            ("Yield", "1y", "OIS"), ("Yield", "2y", "Libor3m"))
        assert rho == pytest.approx(0.94 * 0.993)

    def test_ir_inflation_and_xccy(self):
        rho_infl = intra_bucket_correlation(
            RiskClass.INTEREST_RATE, MarginType.DELTA, "USD",
            ("Yield", "1y", "OIS"), ("Inflation",))
        assert rho_infl == pytest.approx(0.24)
        rho_xccy = intra_bucket_correlation(
            RiskClass.INTEREST_RATE, MarginType.DELTA, "USD",
            ("Yield", "1y", "OIS"), ("XCcyBasis",))
        assert rho_xccy == pytest.approx(0.04)

    def test_credit_qualifying_issuer_correlations(self):
        same = intra_bucket_correlation(
            RiskClass.CREDIT_QUALIFYING, MarginType.DELTA, 2,
            ("ACME", "1y", ""), ("ACME", "5y", ""))
        diff = intra_bucket_correlation(
            RiskClass.CREDIT_QUALIFYING, MarginType.DELTA, 2,
            ("ACME", "1y", ""), ("OTHER", "1y", ""))
        assert same == 0.93
        assert diff == 0.46

    def test_fx_delta_correlation_depends_on_calc_currency(self):
        reg = intra_bucket_correlation(
            RiskClass.FX, MarginType.DELTA, 1, ("EUR",), ("GBP",),
            calculation_currency="USD")
        assert reg == 0.50
        high_calc = intra_bucket_correlation(
            RiskClass.FX, MarginType.DELTA, 1, ("EUR",), ("GBP",),
            calculation_currency="BRL")
        assert high_calc == 0.88

    def test_fx_vega_correlation(self):
        rho = intra_bucket_correlation(
            RiskClass.FX, MarginType.VEGA, 1,
            frozenset({"EUR", "USD"}), frozenset({"GBP", "USD"}))
        assert rho == 0.50

    def test_inter_bucket(self):
        assert inter_bucket_correlation(RiskClass.INTEREST_RATE, "USD", "EUR") == 0.32
        assert inter_bucket_correlation(RiskClass.EQUITY, 5, 6) == pytest.approx(0.29)
        assert inter_bucket_correlation(RiskClass.CREDIT_NON_QUALIFYING, 1, 2) == 0.43


class TestConcentration:
    def test_ir_below_threshold(self):
        sens = [IRDeltaSensitivity("T", 1000.0, currency="USD", tenor=5.0)]
        netted = net_by_risk_factor(sens, MarginType.DELTA)
        cr = ConcentrationCalculator().calculate(
            netted, RiskClass.INTEREST_RATE, MarginType.DELTA, "USD")
        assert cr.bucket_cr == 1.0

    def test_ir_above_threshold(self):
        # USD threshold 330mm/bp: net 1,320mm -> CR = sqrt(4) = 2.
        sens = [IRDeltaSensitivity("T", 1320e6, currency="USD", tenor=5.0)]
        netted = net_by_risk_factor(sens, MarginType.DELTA)
        cr = ConcentrationCalculator().calculate(
            netted, RiskClass.INTEREST_RATE, MarginType.DELTA, "USD")
        assert cr.bucket_cr == pytest.approx(2.0)

    def test_ir_xccy_excluded_and_unscaled(self):
        sens = [
            IRDeltaSensitivity("T", 1320e6, currency="USD", tenor=5.0),
            IRXCcyBasisSensitivity("T", 999e9, currency="USD"),
        ]
        netted = net_by_risk_factor(sens, MarginType.DELTA)
        cr = ConcentrationCalculator().calculate(
            netted, RiskClass.INTEREST_RATE, MarginType.DELTA, "USD")
        # XCcy excluded from the sum -> CR_b still 2; XCcy factor CR = 1.
        assert cr.bucket_cr == pytest.approx(2.0)
        assert cr.cr_values[("XCcyBasis",)] == 1.0
        assert cr.cr_values[("Yield", "5y", "OIS")] == pytest.approx(2.0)

    def test_equity_per_factor(self):
        # Bucket 5 threshold 12mm/%: 48mm -> CR = 2.
        sens = [
            EquityDeltaSensitivity("T", 48e6, issuer="BIG", bucket_number=5),
            EquityDeltaSensitivity("T", 1e6, issuer="SMALL", bucket_number=5),
        ]
        netted = net_by_risk_factor(sens, MarginType.DELTA)
        cr = ConcentrationCalculator().calculate(
            netted, RiskClass.EQUITY, MarginType.DELTA, 5)
        assert cr.cr_values[("BIG",)] == pytest.approx(2.0)
        assert cr.cr_values[("SMALL",)] == 1.0

    def test_credit_groups_by_issuer(self):
        # Bucket 2 threshold 0.17mm/bp: issuer net 0.68mm -> CR = 2.
        sens = [
            CreditDeltaSensitivity("T", 0.5e6, issuer="ACME", bucket_number=2, tenor=1.0),
            CreditDeltaSensitivity("T", 0.18e6, issuer="ACME", bucket_number=2, tenor=5.0),
        ]
        netted = net_by_risk_factor(sens, MarginType.DELTA)
        cr = ConcentrationCalculator().calculate(
            netted, RiskClass.CREDIT_QUALIFYING, MarginType.DELTA, 2)
        for rf, value in cr.cr_values.items():
            assert value == pytest.approx(2.0)

    def test_fx_delta_category_thresholds(self):
        # EUR category 1: 3300mm/%; net 13,200mm -> CR = 2.
        sens = [FXDeltaSensitivity("T", 13200e6, currency="EUR")]
        netted = net_by_risk_factor(sens, MarginType.DELTA)
        cr = ConcentrationCalculator().calculate(
            netted, RiskClass.FX, MarginType.DELTA, 1)
        assert cr.cr_values[("EUR",)] == pytest.approx(2.0)

    def test_f_and_g_factors(self):
        assert ConcentrationCalculator.f_factor(2.0, 1.0) == 0.5
        assert ConcentrationCalculator.g_factor(1.0, 1.0) == 1.0


class TestDeltaMargin:
    def test_single_equity_factor(self):
        # WS = 26 * 1000; single factor -> margin = WS.
        result = _calc().calculate(SensitivityCollection([
            EquityDeltaSensitivity("T", 1000.0, issuer="AAPL", bucket_number=5),
        ]))
        assert result.total_margin == pytest.approx(26000.0)

    def test_ir_two_tenors(self):
        s5, s10 = 10000.0, 20000.0
        result = _calc().calculate(SensitivityCollection([
            IRDeltaSensitivity("T", s5, currency="USD", tenor=5.0),
            IRDeltaSensitivity("T", s10, currency="USD", tenor=10.0),
        ]))
        ws5, ws10 = 60 * s5, 60 * s10
        expected = math.sqrt(ws5**2 + ws10**2 + 2 * 0.95 * ws5 * ws10)
        assert result.total_margin == pytest.approx(expected)

    def test_ir_inflation_correlation(self):
        result = _calc().calculate(SensitivityCollection([
            IRDeltaSensitivity("T", 10000.0, currency="USD", tenor=5.0),
            IRInflationDeltaSensitivity("T", 5000.0, currency="USD"),
        ]))
        ws_y, ws_i = 60 * 10000.0, 61 * 5000.0
        expected = math.sqrt(ws_y**2 + ws_i**2 + 2 * 0.24 * ws_y * ws_i)
        assert result.total_margin == pytest.approx(expected)

    def test_ir_cross_currency_buckets(self):
        result = _calc().calculate(SensitivityCollection([
            IRDeltaSensitivity("T", 10000.0, currency="USD", tenor=5.0),
            IRDeltaSensitivity("T", 10000.0, currency="EUR", tenor=5.0),
        ]))
        k = 60 * 10000.0
        expected = math.sqrt(2 * k**2 + 2 * 0.32 * k * k)
        assert result.total_margin == pytest.approx(expected)

    def test_equity_two_buckets_gamma(self):
        result = _calc().calculate(SensitivityCollection([
            EquityDeltaSensitivity("T", 1000.0, issuer="A", bucket_number=5),
            EquityDeltaSensitivity("T", 1000.0, issuer="B", bucket_number=6),
        ]))
        k5, k6 = 26 * 1000.0, 25 * 1000.0
        expected = math.sqrt(k5**2 + k6**2 + 2 * 0.29 * k5 * k6)
        assert result.total_margin == pytest.approx(expected)

    def test_equity_residual_added_without_diversification(self):
        result = _calc().calculate(SensitivityCollection([
            EquityDeltaSensitivity("T", 1000.0, issuer="A", bucket_number=5),
            EquityDeltaSensitivity("T", 1000.0, issuer="X", bucket_number=-1),
        ]))
        assert result.total_margin == pytest.approx(26 * 1000.0 + 50 * 1000.0)

    def test_equity_residual_zero_intra_correlation(self):
        # Paragraph 59: residual bucket rho = 0.
        result = _calc().calculate(SensitivityCollection([
            EquityDeltaSensitivity("T", 1000.0, issuer="X", bucket_number=-1),
            EquityDeltaSensitivity("T", 1000.0, issuer="Y", bucket_number=-1),
        ]))
        expected = math.sqrt(2) * 50 * 1000.0
        assert result.total_margin == pytest.approx(expected)

    def test_fx_pair_with_calc_currency_dependence(self):
        result = _calc().calculate(SensitivityCollection([
            FXDeltaSensitivity("T", 1000.0, currency="EUR"),
            FXDeltaSensitivity("T", 1000.0, currency="GBP"),
        ]))
        ws = 7.4 * 1000.0
        expected = math.sqrt(2 * ws**2 + 2 * 0.50 * ws * ws)
        assert result.total_margin == pytest.approx(expected)

    def test_fx_calc_currency_factor_dropped(self):
        result = _calc().calculate(SensitivityCollection([
            FXDeltaSensitivity("T", 1000.0, currency="USD"),
        ]))
        assert result.total_margin == 0.0


class TestVegaMargin:
    def test_ir_vega_single_expiry(self):
        # amount is vol-weighted vega; VR = VRW * amount (CR=1).
        config = SIMMConfig(calculate_curvature=False)
        result = _calc(config).calculate(SensitivityCollection([
            IRVegaSensitivity("T", 1000.0, currency="USD", option_tenor=5.0),
        ]))
        assert result.total_margin == pytest.approx(0.23 * 1000.0)

    def test_equity_vega_hvr_and_vrw(self):
        config = SIMMConfig(calculate_curvature=False)
        result = _calc(config).calculate(SensitivityCollection([
            EquityVegaSensitivity("T", 1000.0, issuer="AAPL", bucket_number=5, option_tenor=1.0),
        ]))
        # VR = VRW * HVR * amount = 0.45 * 0.6 * 1000.
        assert result.total_margin == pytest.approx(0.45 * 0.6 * 1000.0)

    def test_ir_vega_two_expiries_tenor_correlation(self):
        config = SIMMConfig(calculate_curvature=False)
        result = _calc(config).calculate(SensitivityCollection([
            IRVegaSensitivity("T", 1000.0, currency="USD", option_tenor=1.0),
            IRVegaSensitivity("T", 1000.0, currency="USD", option_tenor=2.0),
        ]))
        vr = 0.23 * 1000.0
        expected = math.sqrt(2 * vr**2 + 2 * 0.94 * vr * vr)
        assert result.total_margin == pytest.approx(expected)


class TestCurvatureMargin:
    def test_scaling_function(self):
        # Paragraph 11(a) example table.
        assert scaling_function(14.0) == pytest.approx(0.50)
        assert scaling_function(365.0 / 12.0) == pytest.approx(0.23, abs=0.005)
        assert scaling_function(365.0) == pytest.approx(0.019, abs=0.0005)
        assert scaling_function(365.0 * 5) == pytest.approx(0.004, abs=0.0005)

    def test_single_positive_vega_curvature(self):
        config = SIMMConfig(calculate_delta=False, calculate_vega=False)
        result = _calc(config).calculate(SensitivityCollection([
            IRVegaSensitivity("T", 1000.0, currency="USD", option_tenor=1.0),
        ]))
        cvr = scaling_function(365.0) * 1000.0
        lam = (PHI_INV_995**2 - 1.0)  # theta = 0 for net positive CVR
        expected = (cvr + lam * cvr) * IR_HVR ** (-2)
        assert result.by_margin_type[RiskClass.INTEREST_RATE][MarginType.CURVATURE] == pytest.approx(expected)

    def test_net_negative_curvature_floors_at_zero(self):
        config = SIMMConfig(calculate_delta=False, calculate_vega=False)
        result = _calc(config).calculate(SensitivityCollection([
            EquityVegaSensitivity("T", -1000.0, issuer="A", bucket_number=5, option_tenor=1.0),
        ]))
        # theta = -1 -> lambda = 1; CVR + lambda*|K| = -CVR_abs + CVR_abs = 0.
        curvature = result.by_margin_type.get(RiskClass.EQUITY, {}).get(MarginType.CURVATURE, 0.0)
        assert curvature == pytest.approx(0.0)

    def test_equity_volatility_index_bucket_has_zero_curvature(self):
        # Paragraph 11(b): bucket 12 curvature taken to be zero.
        config = SIMMConfig(calculate_delta=False, calculate_vega=False)
        result = _calc(config).calculate(SensitivityCollection([
            EquityVegaSensitivity("T", 1000.0, issuer="VIX", bucket_number=12, option_tenor=1.0),
        ]))
        assert result.by_margin_type.get(RiskClass.EQUITY, {}).get(MarginType.CURVATURE, 0.0) == 0.0


class TestBaseCorrMargin:
    def test_two_index_families(self):
        config = SIMMConfig(calculate_delta=False, calculate_vega=False, calculate_curvature=False)
        result = _calc(config).calculate(SensitivityCollection([
            BaseCorrSensitivity("T", 100.0, index_name="CDX IG"),
            BaseCorrSensitivity("T", -50.0, index_name="iTraxx Main"),
        ]))
        ws1, ws2 = 10 * 100.0, 10 * -50.0
        expected = math.sqrt(ws1**2 + ws2**2 + 2 * 0.29 * ws1 * ws2)
        assert result.total_margin == pytest.approx(expected)


class TestProductClassSeparation:
    def test_same_risk_class_in_different_product_classes_not_netted(self):
        # Paragraph 6: IR risk of an Equity-product trade stays in the
        # Equity product class; the offsetting RatesFX IR risk must NOT
        # net against it.
        offsetting = SensitivityCollection([
            IRDeltaSensitivity("T1", 10000.0, currency="USD", tenor=5.0,
                               product_class=ProductClass.RATES_FX),
            IRDeltaSensitivity("T2", -10000.0, currency="USD", tenor=5.0,
                               product_class=ProductClass.EQUITY),
        ])
        result = _calc().calculate(offsetting)
        # Each product class carries |WS| = 600k of IR delta margin.
        assert result.by_product_class[ProductClass.RATES_FX] == pytest.approx(600000.0)
        assert result.by_product_class[ProductClass.EQUITY] == pytest.approx(600000.0)
        assert result.total_margin == pytest.approx(1200000.0)

    def test_psi_correlation_within_product_class(self):
        result = _calc().calculate(SensitivityCollection([
            IRDeltaSensitivity("T", 10000.0, currency="USD", tenor=5.0),
            FXDeltaSensitivity("T", 10000.0, currency="EUR"),
        ]))
        im_ir = 60 * 10000.0
        im_fx = 7.4 * 10000.0
        expected = math.sqrt(im_ir**2 + im_fx**2 + 2 * 0.14 * im_ir * im_fx)
        assert result.total_margin == pytest.approx(expected)

    def test_total_is_sum_over_product_classes(self):
        result = _calc().calculate(SensitivityCollection([
            IRDeltaSensitivity("T", 10000.0, currency="USD", tenor=5.0),
            EquityDeltaSensitivity("T", 1000.0, issuer="AAPL", bucket_number=5),
        ]))
        assert result.total_margin == pytest.approx(60 * 10000.0 + 26 * 1000.0)


class TestAddOnsAndMultipliers:
    def test_fixed_addon(self):
        config = SIMMConfig(addon_fixed=1000.0)
        result = _calc(config).calculate(SensitivityCollection([
            EquityDeltaSensitivity("T", 1000.0, issuer="AAPL", bucket_number=5),
        ]))
        assert result.total_margin == pytest.approx(26000.0 + 1000.0)

    def test_multiplicative_scale(self):
        config = SIMMConfig(ms_equity=1.5)
        result = _calc(config).calculate(SensitivityCollection([
            EquityDeltaSensitivity("T", 1000.0, issuer="AAPL", bucket_number=5),
        ]))
        assert result.total_margin == pytest.approx(1.5 * 26000.0)


class TestEdgeCases:
    def test_empty_collection(self):
        result = _calc().calculate(SensitivityCollection())
        assert result.total_margin == 0.0

    def test_fully_offsetting_sensitivities(self):
        result = _calc().calculate(SensitivityCollection([
            EquityDeltaSensitivity("T1", 1000.0, issuer="AAPL", bucket_number=5),
            EquityDeltaSensitivity("T2", -1000.0, issuer="AAPL", bucket_number=5),
        ]))
        assert result.total_margin == pytest.approx(0.0)

    def test_result_to_dict(self):
        result = _calc().calculate(SensitivityCollection([
            EquityDeltaSensitivity("T", 1000.0, issuer="AAPL", bucket_number=5),
        ]))
        d = result.to_dict()
        assert d["total_margin"] == pytest.approx(26000.0)
        assert d["by_product_class"]["Equity"] == pytest.approx(26000.0)
        assert d["simm_version"] == "2.6"
