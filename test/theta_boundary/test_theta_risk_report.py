"""The autocallable risk report's theta takes the one step of ``theta_shift`` (spec 2026-10-07 R1, R4)."""
from datetime import datetime, timedelta

import numpy as np
import pytest

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.report.autocallable_risk_report import build_snowball_risk_snapshot
from quantark.asset.equity.report.surfaces import GridSpec
from quantark.asset.equity.riskmeasures.greeks.theta_shift import bumped_value, shift_for_theta
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType

TB = 1.0 / 252.0                                          # GridSpec.time_bump_years; the report moves one day


def _env(spot=100.0):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=0.2),
                              rate_curve=FlatRateCurve(rate=0.02), div_yield=ContinuousDividendYield(div_yield=0.01),
                              valuation_date=datetime(2024, 1, 1))


def _snowball(first_ko):
    cfg = BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                        ko_observation_dates=[first_ko, 0.25, 0.5, 0.75, 1.0], ki_barrier=75.0,
                        ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
    return SnowballOption(initial_price=100.0, strike=100.0, barrier_config=cfg, maturity=1.0)


def _snapshot(product, env, engine, monkeypatch):
    # The bucketed vol-tenor vega raises "total variance is decreasing" on a KO date one day out (pre-existing,
    # outside theta); it runs after the theta block and feeds nothing into it.
    import quantark.asset.equity.report.autocallable_risk_report as report
    monkeypatch.setattr(report, "_compute_bucketed_greeks", lambda *a, **k: __import__("pandas").DataFrame())
    return build_snowball_risk_snapshot(product=product, pricing_env=env, engine=engine, label="t",
                                        grid_spec=GridSpec(spot_nodes=3, q_nodes=3, vol_nodes=3))


@pytest.mark.parametrize("first_ko, spot", [(TB, 104.0), (0.0, 104.0)])
def test_report_theta_is_the_shared_theta_step(first_ko, spot, monkeypatch):
    """KO date on t1 (kept, observed at the new as-of) / KO at t0 (the ended contract carried)."""
    engine, env = SnowballPDESolver(PDEParams()), _env(spot)
    snap = _snapshot(_snowball(first_ko), env, engine, monkeypatch)
    shift = shift_for_theta(_snowball(first_ko), env, TB, env.valuation_date + timedelta(days=1))
    want = (bumped_value(engine, shift) - float(engine.price(_snowball(first_ko), env))) / TB
    assert snap["base"]["theta"] == pytest.approx(want, rel=1e-12, abs=1e-9)
    assert np.isfinite(snap["base"]["theta"])
