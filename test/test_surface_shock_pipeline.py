"""Quote-shock -> rebuild -> reprice pipeline tests (spec WP4.5)."""
import json
from datetime import datetime

import pytest

from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import (
    HestonDCNMCEngine,
    LocalVolDCNMCEngine,
)
from quantark.asset.equity.riskmeasures.surface_shock_pipeline import (
    SurfaceShockMode,
    run_surface_shock_pipeline,
    shock_cleaned_ivs,
)
from quantark.param import SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import DayCountConvention

from dcn_fixtures import DCN_A, make_dcn, synthetic_cleaned_set

PATHS = 2 ** 12  # convention tests, not accuracy gates
SPOT = 6000.0


def _env_builder_factory(rate_curve):
    def build(surface):
        env = PricingEnvironment(
            spot_quote=SpotQuote(spot=SPOT),
            vol_surface=surface,
            rate_curve=rate_curve,
            div_yield=None,
            valuation_date=datetime(2023, 1, 3),
            day_count_convention=DayCountConvention.ACT_365,
        )
        return env

    return build


def _engine_factory(model, artifact):
    if model == "local_vol":
        return LocalVolDCNMCEngine(
            local_vol_surface=artifact, num_paths=PATHS, seed=42
        )
    return HestonDCNMCEngine(model_params=artifact, num_paths=PATHS, seed=42)


def _run(model, mode, **kw):
    cleaned, rate_curve, carry_curve, _ = synthetic_cleaned_set()
    env_builder = _env_builder_factory(rate_curve)
    # attach the carry-derived q so drift is consistent with the fixture
    base_builder = env_builder

    def builder(surface):
        env = base_builder(surface)
        env.div_yield = carry_curve.to_dividend_yield(rate_curve)
        return env

    return run_surface_shock_pipeline(
        make_dcn(DCN_A),
        builder,
        cleaned,
        SPOT,
        rate_curve,
        carry_curve,
        model=model,
        mode=mode,
        engine_factory=_engine_factory,
        **kw,
    )


def test_shock_targets_only_selected_ivs():
    cleaned, *_ = synthetic_cleaned_set()
    ts = sorted(cleaned.slices)
    shocked = shock_cleaned_ivs(cleaned, 0.01, tenor_bucket=(0.0, ts[0]))
    for q_old, q_new in zip(cleaned.slices[ts[0]], shocked.slices[ts[0]]):
        assert q_new.iv == pytest.approx(q_old.iv + 0.01)
    for q_old, q_new in zip(cleaned.slices[ts[1]], shocked.slices[ts[1]]):
        assert q_new.iv == q_old.iv


def test_lv_recalibrate_transmits_the_shock():
    res = _run("local_vol", SurfaceShockMode.RECALIBRATE)
    assert res.pnl != 0.0
    assert res.no_arb_passed
    assert res.shock["layer"] == "cleaned_market_iv_nodes"
    diagnostics = res.artifact_diagnostics
    assert diagnostics["base_local_vol"]["min"] > 0.0
    assert diagnostics["shocked_local_vol"]["min"] > 0.0
    assert diagnostics["shocked_local_vol"]["max"] >= (
        diagnostics["shocked_local_vol"]["min"]
    )


def test_heston_frozen_is_zero_with_note():
    res = _run("heston", SurfaceShockMode.FROZEN)
    assert res.pnl == 0.0
    assert any("frozen params" in n for n in res.notes)


def test_lv_frozen_smaller_than_recalibrate():
    frozen = _run("local_vol", SurfaceShockMode.FROZEN)
    recal = _run("local_vol", SurfaceShockMode.RECALIBRATE)
    assert abs(frozen.pnl) < abs(recal.pnl)
    assert any("frozen Dupire" in n for n in frozen.notes)


def test_results_to_dict_json_safe():
    for model, mode in (("local_vol", SurfaceShockMode.RECALIBRATE),
                        ("heston", SurfaceShockMode.FROZEN)):
        payload = json.dumps(_run(model, mode).to_dict())
        assert "cleaned_market_iv_nodes" in payload
