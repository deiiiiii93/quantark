"""Smoke test: the full FX risk-stack demo runs end-to-end headless."""
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))


def test_fx_portfolio_risk_demo_runs():
    from example.asset.fx.fx_portfolio_risk_demo import main

    with redirect_stdout(io.StringIO()):
        main()


def test_individual_fx_demos_build_objects():
    """Each per-module FX demo exposes a working builder."""
    from example.asset.fx.fx_backtest_demo import build_book as bt_book
    from example.asset.fx.fx_dynamic_scenario_demo import build_book as dyn_book
    from example.asset.fx.fx_stress_test_demo import build_book as stress_book
    from example.asset.fx.fx_var_demo import build_book as var_book

    assert len(stress_book()) == 2
    assert len(var_book()) == 2
    assert len(dyn_book()) == 2
    assert len(bt_book()) == 1
