"""Static equity stress on a spawn process pool (ship spec 2026-10-23 D13, AC12).

Every task works on a freshly pickled copy of its post-baseline inputs, so the pool's results are bitwise the
serial results, in input order, in scalar and in native-batch mode. Pool infrastructure failures (inputs that do
not pickle, a pool that cannot start) warn and run serially; a pricing error inside a task propagates.
"""
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime

import pytest

import quantark.stresstest.equity.parallel as parallel
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.quad.v2 import PhoenixQuadEngineV2, SnowballQuadEngineV2
from quantark.asset.equity.param import PDEParams, QuadParams
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio import Portfolio
from quantark.priceenv import PricingEnvironment
from quantark.stresstest import ScenarioBuilder, StressTestConfig, StressTestEngine, StressType

from test_asset.equity.product_other.test_equity_position_batch import _phoenix, _rebate_leg, _snowball
from test_stresstest.test_equity_stress_native_batch import _add, _plain_snowball

VAL = datetime(2026, 3, 16)


def _env():
    return PricingEnvironment(spot_quote=SpotQuote(100.0, asset_name="X"), vol_surface=FlatVolSurface(0.25),
                              rate_curve=FlatRateCurve(0.03), div_yield=ContinuousDividendYield(0.01),
                              valuation_date=VAL)


def _book():
    pf = Portfolio(portfolio_name="pool", pricing_environments={"X": _env()}, creation_date=VAL)
    shared = SnowballQuadEngine(params=QuadParams())       # two positions on ONE engine object
    _add(pf, "v2_snowball", _snowball(), SnowballQuadEngineV2(), quantity=-2.0)
    _add(pf, "v2_phoenix_leg", _phoenix(), PhoenixQuadEngineV2(), legs=(_rebate_leg(),))
    _add(pf, "pde", _plain_snowball(), SnowballPDESolver(PDEParams()))
    _add(pf, "v1_a", _plain_snowball(), shared, quantity=2.0)
    _add(pf, "v1_b", _snowball(), shared, quantity=-1.0)
    return pf


def _scenarios():
    return [ScenarioBuilder().name(f"spot {s:+.2f}").spot_stress(s).build() for s in (-0.12, -0.04, 0.03)] + [
        ScenarioBuilder().name("vol +5").vol_stress(0.05, stress_type=StressType.ABSOLUTE).build()]


def _run(*, native_batch, parallel_execution, book=None):
    config = StressTestConfig(calculate_greeks=True, native_batch=native_batch,
                              parallel_execution=parallel_execution, max_workers=2)
    return StressTestEngine(config).run_static_scenarios(book or _book(), _scenarios())


def _numbers(results):
    """Everything a run reports except timings."""
    return (results.baseline_value, results.baseline_greeks,
            [(r.scenario.name, r.portfolio_value, r.portfolio_pnl, r.greeks, r.position_results,
              r.underlying_results) for r in results.scenario_results])


@pytest.mark.parametrize("native_batch", [False, True])
def test_pool_results_are_bitwise_serial(native_batch, recwarn):
    serial = _run(native_batch=native_batch, parallel_execution=False)
    pooled = _run(native_batch=native_batch, parallel_execution=True)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
    assert pooled.metadata["execution"] == "process_pool" and pooled.metadata["workers"] >= 1
    assert serial.metadata["execution"] == "serial"
    assert _numbers(pooled) == _numbers(serial)


def test_a_pool_that_cannot_start_warns_and_runs_serially(monkeypatch):
    class Broken:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, *args, **kwargs):
            raise BrokenProcessPool("simulated bootstrap failure")

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", Broken)
    serial = _run(native_batch=False, parallel_execution=False)
    with pytest.warns(RuntimeWarning, match="serial"):
        fallback = _run(native_batch=False, parallel_execution=True)
    assert fallback.metadata["execution"] == "serial"
    assert _numbers(fallback) == _numbers(serial)


def test_inputs_that_do_not_pickle_warn_and_run_serially():
    def book():
        pf = _book()
        pf.positions["pde"].engine.note = lambda: None    # deep-copies, never pickles
        return pf

    serial = _run(native_batch=False, parallel_execution=False, book=book())
    with pytest.warns(RuntimeWarning, match="serial"):
        fallback = _run(native_batch=False, parallel_execution=True, book=book())
    assert _numbers(fallback) == _numbers(serial)


def _explode(task):
    """A task that fails the way a pricing call can (module level, so spawn workers can import it)."""
    raise NotImplementedError(f"task {task} refuses")


def test_a_pricing_error_inside_a_task_propagates(recwarn):
    with pytest.raises(NotImplementedError, match="task 1 refuses|task 2 refuses"):
        parallel.run_tasks(_explode, [1, 2], max_workers=2)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_single_task_or_one_worker_stays_serial():
    config = StressTestConfig(calculate_greeks=False, parallel_execution=True, max_workers=1)
    result = StressTestEngine(config).run_static_scenarios(_book(), _scenarios())
    assert result.metadata["execution"] == "serial"
    config = StressTestConfig(calculate_greeks=False, parallel_execution=True, max_workers=2)
    result = StressTestEngine(config).run_static_scenarios(_book(), _scenarios()[:1])
    assert result.metadata["execution"] == "serial"
