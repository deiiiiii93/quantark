from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.asset.equity.report import autocallable_risk_report as report
from quantark.asset.equity.report.surfaces import GridSpec
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment


class DummyEngine(BaseEngine):
    def price(self, product, pricing_env) -> float:
        return float(pricing_env.spot) * 0.01


def _build_env() -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2024, 1, 1),
    )


def _build_product():
    return create_standard_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        ko_barrier=103.0,
        ko_rate=0.15,
        ki_barrier=75.0,
        num_observations=4,
        is_reverse=False,
        include_principal=False,
    )


def _patch_report(monkeypatch):
    monkeypatch.setattr(
        report,
        "_select_snowball_pricing_engine",
        lambda **kwargs: (DummyEngine(), "dummy"),
    )

    def _dummy_analyze(self, product, pricing_env):
        return report.RiskNeutralSnowballEventStats(
            pv_mc=1.0,
            std_error=0.0,
            num_paths=10,
            ko_times=np.array([0.5, 1.0]),
            ko_prob=np.array([0.1, 0.2]),
            survive_prob=np.array([0.9, 0.7]),
            expected_discounted_ko_cf=np.array([0.05, 0.1]),
            ki_probability=0.3,
            expected_discounted_maturity_cf=0.0,
            reconciliation_error=0.0,
        )

    monkeypatch.setattr(
        report.AutocallablePathAnalyzer,
        "analyze_snowball_risk_neutral",
        _dummy_analyze,
    )


def test_barrier_distance_metrics():
    product = _build_product()
    env = _build_env()
    barrier, pct_dist, sigma_dist = report._barrier_distance_metrics(
        spot=100.0,
        barrier=110.0,
        time_to_barrier=1.0,
        pricing_env=env,
        product=product,
    )
    assert barrier == 110.0
    assert pct_dist == pytest.approx(0.10)
    assert sigma_dist == pytest.approx(np.log(1.1) / 0.2)


def test_barrier_zoom_surface_shapes():
    product = _build_product()
    env = _build_env()
    engine = DummyEngine()
    vol_grid = np.linspace(0.19, 0.21, 5)

    spot_grid, vol_grid_out, gamma, vega = report._compute_barrier_zoom_surfaces(
        product=product,
        pricing_env=env,
        engine=engine,
        barrier_level=100.0,
        vol_grid=vol_grid,
        base_vol=0.2,
        base_div_yield=env.div_yield,
        spot_nodes=11,
        band_width=0.02,
    )

    assert spot_grid[0] == pytest.approx(98.0)
    assert spot_grid[-1] == pytest.approx(102.0)
    assert vol_grid_out.shape == vol_grid.shape
    assert gamma.shape == (spot_grid.size, vol_grid.size)
    assert vega.shape == (spot_grid.size, vol_grid.size)


def test_vanna_volga_linear_function():
    spot_grid = np.linspace(90.0, 110.0, 9)
    vol_grid = np.linspace(0.2, 0.3, 7)
    pv_sv = np.outer(spot_grid, vol_grid)

    vanna, volga = report._compute_vanna_volga(pv_sv, spot_grid, vol_grid)

    assert np.max(np.abs(vanna - 1.0)) < 1e-6
    assert np.max(np.abs(volga)) < 1e-6


def test_report_smoke_sections(tmp_path, monkeypatch):
    product = _build_product()
    env = _build_env()

    _patch_report(monkeypatch)

    grid = GridSpec(spot_nodes=5, q_nodes=5, vol_nodes=5, time_bump_years=1.0 / 252.0)
    result = report.generate_snowball_risk_report(
        product=product,
        pricing_env=env,
        output_dir=tmp_path,
        grid_spec=grid,
        engine_preference=["dummy"],
        skew_smile_shock=report.SkewSmileShock(skew=0.1, smile=0.0),
    )

    content = result.report_path.read_text()
    assert "Executive Dashboard" in content
    assert "Barrier Watch" in content
    assert "Barrier Risk (Zoom)" in content
    assert "Advanced Volatility Risk" in content
    assert "Higher-Order Time Greeks" in content
    assert "Lifecycle Context" in content
    assert "Stress Scenarios" in content
    assert "Conditional Cashflow Projection" in content


def test_report_high_accuracy_smoke(tmp_path, monkeypatch):
    product = _build_product()
    env = _build_env()

    _patch_report(monkeypatch)

    grid = GridSpec(spot_nodes=5, q_nodes=5, vol_nodes=5, time_bump_years=1.0 / 252.0)
    result = report.generate_snowball_risk_report(
        product=product,
        pricing_env=env,
        output_dir=tmp_path,
        grid_spec=grid,
        engine_preference=["dummy"],
        high_accuracy_surfaces=True,
    )

    content = result.report_path.read_text()
    assert "Surface mode: point-greeks" in content
