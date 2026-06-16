"""SA-CVA per-risk-class engine tests (hand-computed vs MAR50.51-50.53)."""

import math
import pytest

from quantark.sacva.models.enums import RiskClass, RiskType, CreditQuality
from quantark.sacva.models.sensitivity import CVASensitivity
from quantark.sacva.engines.base import BaseRiskClassEngine


class _StubSingleFactor(BaseRiskClassEngine):
    """Single-factor-per-bucket stub: RW=0.30, cross-bucket gamma=0.15."""
    def supports(self, risk_type): return True
    def is_single_factor(self): return True
    def factor_identity(self, s): return s.bucket
    def risk_weight(self, s, risk_type): return 0.30
    def group_buckets(self, sens, reporting_currency):
        out = {}
        for s in sens:
            out.setdefault(s.bucket, []).append(s)
        return out
    def intra_correlation(self, a, b): return 0.0
    def cross_bucket_correlation(self, b, c): return 0.15


def _eq(bucket, s_cva, s_hdg=0.0):
    return CVASensitivity(risk_class=RiskClass.EQUITY, risk_type=RiskType.DELTA,
                          bucket=bucket, risk_factor=f"b{bucket}",
                          s_cva=s_cva, s_hdg=s_hdg, name="n")


def test_single_factor_bucket_with_hedge_disallowance():
    eng = _StubSingleFactor()
    res = eng.calculate([_eq(5, 1000.0, 400.0)], RiskType.DELTA, "USD")
    assert res.k == pytest.approx(math.sqrt(180**2 + 0.01*120**2), rel=1e-12)
    assert res.by_bucket[5] == pytest.approx(math.sqrt(32544.0), rel=1e-12)


def test_pre_netting_sums_duplicate_rows():
    eng = _StubSingleFactor()
    res = eng.calculate([_eq(5, 600.0), _eq(5, 400.0)], RiskType.DELTA, "USD")
    assert res.k == pytest.approx(0.30 * 1000.0, rel=1e-12)


def test_perfect_hedge_leaves_disallowance_only():
    eng = _StubSingleFactor()
    res = eng.calculate([_eq(5, 1000.0, 1000.0)], RiskType.DELTA, "USD")
    assert res.k == pytest.approx(30.0, rel=1e-12)


def test_cross_bucket_aggregation():
    eng = _StubSingleFactor()
    res = eng.calculate([_eq(5, 600.0), _eq(6, 175.0/0.30)], RiskType.DELTA, "USD")
    expected = math.sqrt(180**2 + 175**2 + 2*0.15*180*175)
    assert res.k == pytest.approx(expected, rel=1e-12)


# ---- Counterparty credit ----
from quantark.sacva.engines.counterparty_credit import CounterpartyCreditEngine


def _cp(bucket, name, tenor, s_cva, quality=CreditQuality.IG, sub_bucket=None,
        s_hdg=0.0, leg=None):
    return CVASensitivity(risk_class=RiskClass.COUNTERPARTY_CREDIT, risk_type=RiskType.DELTA,
                          bucket=bucket, risk_factor=f"{name}:{tenor}", s_cva=s_cva,
                          s_hdg=s_hdg, name=name, tenor=tenor, credit_quality=quality,
                          sub_bucket=sub_bucket, legal_entity_group=leg)


def test_counterparty_two_distinct_names_s_b_capped():
    eng = CounterpartyCreditEngine()
    res = eng.calculate([_cp(3, "A", 5.0, 1000.0), _cp(3, "B", 5.0, 500.0)],
                        RiskType.DELTA, "USD")
    kb = math.sqrt(30**2 + 15**2 + 2*0.5*30*15)
    assert res.by_bucket[3] == pytest.approx(kb, rel=1e-12)
    assert res.bucket_s_b[3] == pytest.approx(kb, rel=1e-12)  # capped
    assert res.k == pytest.approx(kb, rel=1e-12)


def test_counterparty_capped_s_b_changes_cross_bucket_k():
    eng = CounterpartyCreditEngine()
    rows = [_cp(3, "A", 5.0, 1000.0), _cp(3, "B", 5.0, 500.0),
            _cp(5, "C", 5.0, 1000.0)]
    res = eng.calculate(rows, RiskType.DELTA, "USD")
    kb3 = math.sqrt(30**2 + 15**2 + 2*0.5*30*15)   # 39.6863
    s_b3 = kb3                                      # capped (sum_ws=45 > kb3)
    kb5 = 0.02 * 1000.0                             # 20
    expected = math.sqrt(kb3**2 + kb5**2 + 2*0.25*s_b3*kb5)
    assert res.bucket_s_b[3] == pytest.approx(s_b3, rel=1e-12)
    assert res.k == pytest.approx(expected, rel=1e-12)


def test_counterparty_legal_relation_corr():
    from quantark.sacva.engines.correlations import counterparty_rho_name
    a = _cp(3, "A", 5.0, 1.0, leg="G")
    b = _cp(3, "B", 5.0, 1.0, leg="G")
    c = _cp(3, "C", 5.0, 1.0)
    d = _cp(3, "D", 5.0, 1.0)
    assert counterparty_rho_name(a, a) == 1.0
    assert counterparty_rho_name(a, b) == 0.90
    assert counterparty_rho_name(c, d) == 0.50   # both None -> NOT related
    assert counterparty_rho_name(a, c) == 0.50


def test_counterparty_no_vega():
    eng = CounterpartyCreditEngine()
    assert eng.supports(RiskType.VEGA) is False
    res = eng.calculate([_cp(3, "A", 5.0, 1000.0)], RiskType.VEGA, "USD")
    assert res.k == 0.0


# ---- Interest rate ----
from quantark.sacva.engines.interest_rate import InterestRateEngine


def _ir(currency, s_cva, tenor=None, inflation=False, s_hdg=0.0):
    return CVASensitivity(risk_class=RiskClass.INTEREST_RATE, risk_type=RiskType.DELTA,
                          bucket=0, currency=currency, risk_factor=f"{currency}:{tenor}",
                          s_cva=s_cva, s_hdg=s_hdg, tenor=tenor, is_inflation=inflation)


def test_ir_specified_two_tenors():
    eng = InterestRateEngine()
    res = eng.calculate([_ir("USD", 10000.0, tenor=5.0), _ir("USD", 20000.0, tenor=10.0)],
                        RiskType.DELTA, "USD")
    kb = math.sqrt(74**2 + 148**2 + 2*0.91*74*148)
    assert res.by_bucket["USD"] == pytest.approx(kb, rel=1e-12)
    assert res.k == pytest.approx(kb, rel=1e-12)


def test_ir_other_currency_parallel_plus_inflation():
    eng = InterestRateEngine()
    res = eng.calculate([_ir("BRL", 1000.0), _ir("BRL", 500.0, inflation=True)],
                        RiskType.DELTA, "USD")
    kb = math.sqrt(15.8**2 + 7.9**2 + 2*0.40*15.8*7.9)
    assert res.by_bucket["BRL"] == pytest.approx(kb, rel=1e-9)


def test_ir_other_currency_parallel_factor_nets_across_tenors_hedge():
    eng = InterestRateEngine()
    rows = [_ir("BRL", 1000.0, tenor=5.0, s_hdg=600.0),
            _ir("BRL", 1000.0, tenor=10.0, s_hdg=400.0)]
    res = eng.calculate(rows, RiskType.DELTA, "USD")
    kb = math.sqrt(15.8**2 + 0.01*15.8**2)
    assert res.by_bucket["BRL"] == pytest.approx(kb, rel=1e-9)


def test_ir_reporting_currency_is_specified():
    eng = InterestRateEngine()
    res = eng.calculate([_ir("BRL", 10000.0, tenor=5.0)], RiskType.DELTA, "BRL")
    assert res.by_bucket["BRL"] == pytest.approx(0.0074 * 10000.0, rel=1e-12)


def test_ir_specified_inflation_is_single_factor_hedge():
    eng = InterestRateEngine()
    rows = [_ir("USD", 1000.0, tenor=2.0, inflation=True, s_hdg=600.0),
            _ir("USD", 1000.0, tenor=10.0, inflation=True, s_hdg=400.0)]
    res = eng.calculate(rows, RiskType.DELTA, "USD")
    kb = math.sqrt(11.1**2 + 0.01*11.1**2)
    assert res.by_bucket["USD"] == pytest.approx(kb, rel=1e-9)


def test_ir_cross_bucket_gamma_half():
    eng = InterestRateEngine()
    res = eng.calculate([_ir("USD", 10000.0, tenor=5.0), _ir("EUR", 10000.0, tenor=5.0)],
                        RiskType.DELTA, "USD")
    ws = 0.0074 * 10000.0
    expected = math.sqrt(ws**2 + ws**2 + 2*0.5*ws*ws)
    assert res.k == pytest.approx(expected, rel=1e-12)


# ---- FX / reference credit / equity / commodity ----
from quantark.sacva.engines.fx import FXEngine
from quantark.sacva.engines.reference_credit import ReferenceCreditEngine
from quantark.sacva.engines.equity import EquityEngine
from quantark.sacva.engines.commodity import CommodityEngine


def test_fx_single_factor_and_gamma():
    f1 = CVASensitivity(risk_class=RiskClass.FX, risk_type=RiskType.DELTA, bucket=0,
                        currency="EUR", risk_factor="EUR", s_cva=1000.0, s_hdg=0.0)
    f2 = CVASensitivity(risk_class=RiskClass.FX, risk_type=RiskType.DELTA, bucket=0,
                        currency="GBP", risk_factor="GBP", s_cva=2000.0, s_hdg=0.0)
    res = FXEngine().calculate([f1, f2], RiskType.DELTA, "USD")
    ws1, ws2 = 0.11*1000.0, 0.11*2000.0
    assert res.k == pytest.approx(math.sqrt(ws1**2 + ws2**2 + 2*0.6*ws1*ws2), rel=1e-12)


def test_reference_credit_single_factor_per_bucket():
    s1 = CVASensitivity(risk_class=RiskClass.REFERENCE_CREDIT, risk_type=RiskType.DELTA,
                        bucket=3, risk_factor="b3", s_cva=1000.0, s_hdg=0.0, name="x")
    res = ReferenceCreditEngine().calculate([s1], RiskType.DELTA, "USD")
    assert res.by_bucket[3] == pytest.approx(0.050 * 1000.0, rel=1e-12)


def test_reference_credit_diff_quality_gamma_halved():
    ig = CVASensitivity(risk_class=RiskClass.REFERENCE_CREDIT, risk_type=RiskType.DELTA,
                        bucket=1, risk_factor="b1", s_cva=1000.0, name="a")
    hy = CVASensitivity(risk_class=RiskClass.REFERENCE_CREDIT, risk_type=RiskType.DELTA,
                        bucket=10, risk_factor="b10", s_cva=1000.0, name="b")
    res = ReferenceCreditEngine().calculate([ig, hy], RiskType.DELTA, "USD")
    ws1 = 0.005 * 1000.0
    ws2 = 0.120 * 1000.0
    gamma = 0.10 / 2.0
    assert res.k == pytest.approx(math.sqrt(ws1**2 + ws2**2 + 2*gamma*ws1*ws2), rel=1e-9)


def test_equity_vega_large_cap_rw():
    s = CVASensitivity(risk_class=RiskClass.EQUITY, risk_type=RiskType.VEGA,
                       bucket=1, risk_factor="b1", s_cva=1000.0, name="a")
    res = EquityEngine().calculate([s], RiskType.VEGA, "USD")
    assert res.by_bucket[1] == pytest.approx(0.78 * 1000.0, rel=1e-12)


def test_commodity_cross_bucket():
    a = CVASensitivity(risk_class=RiskClass.COMMODITY, risk_type=RiskType.DELTA,
                       bucket=1, risk_factor="b1", s_cva=1000.0)
    b = CVASensitivity(risk_class=RiskClass.COMMODITY, risk_type=RiskType.DELTA,
                       bucket=2, risk_factor="b2", s_cva=1000.0)
    res = CommodityEngine().calculate([a, b], RiskType.DELTA, "USD")
    ws1, ws2 = 0.30*1000.0, 0.35*1000.0
    assert res.k == pytest.approx(math.sqrt(ws1**2 + ws2**2 + 2*0.20*ws1*ws2), rel=1e-12)
