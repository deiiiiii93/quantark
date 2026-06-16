"""SA-CVA dashboard tests (skipped if plotly is unavailable)."""

import pytest

pytest.importorskip("plotly")

from quantark.sacva import (SACVACalculator, CVAPortfolio, CVASensitivity,
                            RiskClass, RiskType, SACVADashboard)


def test_dashboard_generates_html(tmp_path):
    s = CVASensitivity(risk_class=RiskClass.EQUITY, risk_type=RiskType.DELTA, bucket=1,
                       risk_factor="b1", s_cva=1000.0, name="n")
    pf = CVAPortfolio([s], reporting_currency="USD")
    res = SACVACalculator().calculate(pf)
    out = tmp_path / "sacva.html"
    path = SACVADashboard(res, portfolio=pf).generate(out)
    html = out.read_text()
    assert "SA-CVA" in html
    assert "Delta" in html and "Vega" in html
    assert str(path) == str(out)
