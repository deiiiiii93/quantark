"""Surface diagnostics tests (spec WP4.3)."""
import json

import numpy as np
import pytest

from quantark.param.vol.marketquotes import clean_and_imply
from quantark.param.vol.svi import SVIVolSurface
from quantark.volmodels.diagnostics import (
    repricing_residual_report,
    static_no_arb_report,
)

from dcn_fixtures import synthetic_quote_book, synthetic_svi_slices


def _surface():
    return SVIVolSurface(
        synthetic_svi_slices(), forward_provider=lambda t: 5800.0
    )


def _cleaned():
    quotes, valuation_date, spot, rate_curve, carry_curve = (
        synthetic_quote_book()
    )
    return clean_and_imply(quotes, valuation_date, spot, rate_curve, carry_curve)


def test_no_arb_passes_on_healthy_surface():
    s = _surface()
    rep = static_no_arb_report(s, [0.25, 0.5])
    assert rep.passed
    assert all(v > 0.0 for v in rep.butterfly_min_g.values())
    assert rep.calendar_min_dw[(0.25, 0.5)] > 0.0


def test_calendar_violation_detected():
    class _BadSurface:
        # total variance DECREASES with T at fixed y: calendar arbitrage
        def total_variance(self, y, t):
            y = np.asarray(y, dtype=float)
            return (0.05 - 0.02 * t) * (1.0 + 0.1 * y * y)

    rep = static_no_arb_report(_BadSurface(), [0.25, 0.5])
    assert not rep.passed
    assert rep.calendar_min_dw[(0.25, 0.5)] < 0.0


def test_residual_report_exact_offset():
    cleaned = _cleaned()

    def model_iv_fn(strike, expiry_t):
        (t,) = cleaned.slices.keys()
        market = {q.strike: q.iv for q in cleaned.slices[t]}
        return market[strike] + 0.005

    rep = repricing_residual_report(cleaned, model_iv_fn)
    assert rep.rmse_iv == pytest.approx(0.005, abs=1e-12)
    assert rep.max_abs_iv == pytest.approx(0.005, abs=1e-12)
    # price errors consistent with vega * 0.005 to ~5% relative
    for row in rep.rows:
        assert row["price_error"] == pytest.approx(
            row["vega"] * 0.005, rel=0.05
        )
    assert all(v == pytest.approx(0.005, abs=1e-12)
               for v in rep.by_bucket.values())


def test_reports_to_dict_json_safe():
    rep1 = static_no_arb_report(_surface(), [0.25, 0.5])
    rep2 = repricing_residual_report(
        _cleaned(), lambda k, t: 0.2
    )
    json.dumps(rep1.to_dict())
    json.dumps(rep2.to_dict())
