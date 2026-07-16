"""12-engine adaptive RQMC bitwise matrix + checkpoint determinism (Phase 3).

Every Snowball/Phoenix engine (BSM/LV/Heston/QE/SLV/SLV-QE) in
RANDOMIZED_QUASI mode: session PRICE must equal the direct call BITWISE
(price, std_error, batches_used). Checkpoint traces are deterministic and
their complete-value fingerprint is part of the diagnostics evidence.
"""
import threading

import pytest

from execution.matrix_fixtures import (
    _eq_flat_env,
    _eq_grid_env,
    _hp,
    _phoenix,
    _snowball,
    _unit_leverage,
)
from execution.test_adaptive_adapter import (
    _ko_reset_product,
    _rqmc_params,
    _rqmc_snowball_engine,
)
from quantark.execution import PricingSession
from quantark.execution.contracts import PricingRequest

_VOL_MCP = dict(num_paths=1024, time_steps=24, seed=19,
                rqmc_min_batches=2, rqmc_max_batches=4,
                rqmc_target_std=1e-9)

_ENGINE_NAMES = [
    "SnowballMCEngine", "LocalVolSnowballMCEngine", "HestonSnowballMCEngine",
    "QESnowballMCEngine", "HestonSLVSnowballMCEngine",
    "HestonSLVQESnowballMCEngine",
    "PhoenixMCEngine", "LocalVolPhoenixMCEngine", "HestonPhoenixMCEngine",
    "QEPhoenixMCEngine", "HestonSLVPhoenixMCEngine",
    "HestonSLVQEPhoenixMCEngine",
]


def _build(name):
    """(engine_factory, product, env) for an RQMC-mode autocallable engine."""
    from quantark.asset.equity.engine.mc import (
        HestonPhoenixMCEngine, HestonSLVPhoenixMCEngine,
        HestonSLVQEPhoenixMCEngine, HestonSLVQESnowballMCEngine,
        HestonSLVSnowballMCEngine, HestonSnowballMCEngine,
        LocalVolPhoenixMCEngine, LocalVolSnowballMCEngine,
        PhoenixMCEngine, QEPhoenixMCEngine, QESnowballMCEngine,
        SnowballMCEngine,
    )
    from quantark.util.enum.engine_enums import MonteCarloMethod

    method = MonteCarloMethod.RANDOMIZED_QUASI
    mcp = lambda: _rqmc_params(**_VOL_MCP)  # noqa: E731
    product = _snowball() if "Snowball" in name else _phoenix()
    env = _eq_flat_env() if name in ("SnowballMCEngine", "PhoenixMCEngine") \
        else _eq_grid_env()
    factories = {
        "SnowballMCEngine": lambda: SnowballMCEngine(
            params=mcp(), method=method),
        "LocalVolSnowballMCEngine": lambda: LocalVolSnowballMCEngine(
            mcp(), method=method),
        "HestonSnowballMCEngine": lambda: HestonSnowballMCEngine(
            _hp(), mcp(), method=method),
        "QESnowballMCEngine": lambda: QESnowballMCEngine(
            _hp(), mcp(), method=method),
        "HestonSLVSnowballMCEngine": lambda: HestonSLVSnowballMCEngine(
            _hp(), params=mcp(), leverage_surface=_unit_leverage(),
            method=method),
        "HestonSLVQESnowballMCEngine": lambda: HestonSLVQESnowballMCEngine(
            _hp(), params=mcp(), leverage_surface=_unit_leverage(),
            method=method),
        "PhoenixMCEngine": lambda: PhoenixMCEngine(
            params=mcp(), method=method),
        "LocalVolPhoenixMCEngine": lambda: LocalVolPhoenixMCEngine(
            mcp(), method=method),
        "HestonPhoenixMCEngine": lambda: HestonPhoenixMCEngine(
            _hp(), mcp(), method=method),
        "QEPhoenixMCEngine": lambda: QEPhoenixMCEngine(
            _hp(), mcp(), method=method),
        "HestonSLVPhoenixMCEngine": lambda: HestonSLVPhoenixMCEngine(
            _hp(), params=mcp(), leverage_surface=_unit_leverage(),
            method=method),
        "HestonSLVQEPhoenixMCEngine": lambda: HestonSLVQEPhoenixMCEngine(
            _hp(), params=mcp(), leverage_surface=_unit_leverage(),
            method=method),
    }
    return factories[name], product, env


@pytest.mark.parametrize("name", _ENGINE_NAMES)
def test_session_bitwise_vs_direct(name):
    make_engine, product, env = _build(name)
    direct = make_engine()
    expected = direct.price(product, env)
    expected_result = direct.get_last_result()

    with PricingSession() as session:
        outcome = session.execute(
            make_engine(), PricingRequest(product=product, pricing_env=env),
        )
    assert outcome.value == expected, name
    econ = dict(outcome.normalized_economics)
    assert econ["std_error"] == float(expected_result.std_error), name
    assert outcome.manifest.adapter_id == "autocallable-adaptive-mc"
    assert (
        f"adaptive:batches_used={expected_result.batches_used}"
        in outcome.diagnostics.records
    ), name


def test_ko_reset_vol_engine_bitwise():
    from quantark.asset.equity.engine.mc import LocalVolSnowballMCEngine
    from quantark.util.enum.engine_enums import MonteCarloMethod

    product, env = _ko_reset_product(), _eq_grid_env()

    def make_engine():
        return LocalVolSnowballMCEngine(
            _rqmc_params(**_VOL_MCP),
            method=MonteCarloMethod.RANDOMIZED_QUASI,
        )

    direct = make_engine()
    expected = direct.price(product, env)
    with PricingSession() as session:
        outcome = session.execute(
            make_engine(), PricingRequest(product=product, pricing_env=env),
        )
    assert outcome.value == expected


def _session_outcome(make_engine, product, env):
    with PricingSession() as session:
        return session.execute(
            make_engine(), PricingRequest(product=product, pricing_env=env),
        )


def test_checkpoint_trace_deterministic_and_fingerprint_stable():
    product, env = _snowball(), _eq_flat_env()
    a = _session_outcome(_rqmc_snowball_engine, product, env)
    b = _session_outcome(_rqmc_snowball_engine, product, env)
    a_records = [r for r in a.diagnostics.records if r.startswith("adaptive:")]
    b_records = [r for r in b.diagnostics.records if r.startswith("adaptive:")]
    assert a_records == b_records
    assert any(
        r.startswith("adaptive:trace_fingerprint=") for r in a_records
    )
    assert a.manifest.plan_fingerprint == b.manifest.plan_fingerprint


def test_trace_fingerprint_changes_when_any_checkpoint_value_changes():
    from quantark.execution.cache.fingerprint import try_fingerprint
    from quantark.montecarlo.qmc_rqmc_driver import RQMCCheckpoint

    def make(batch_mean):
        return (
            RQMCCheckpoint(batch_index=0, batch_mean=1.0, running_mean=1.0,
                           std_error=None, stopped=False),
            RQMCCheckpoint(batch_index=1, batch_mean=batch_mean,
                           running_mean=(1.0 + batch_mean) / 2,
                           std_error=0.1, stopped=True),
        )

    base = try_fingerprint(make(2.0))
    same = try_fingerprint(make(2.0))
    perturbed = try_fingerprint(make(2.0 + 1e-12))
    assert base is not None
    assert base == same
    assert base != perturbed


def test_stop_boundaries():
    product, env = _snowball(), _eq_flat_env()

    # loose target: stops at min_batches
    def loose():
        return _rqmc_snowball_engine(rqmc_target_std=1e6)

    direct = loose()
    direct.price(product, env)
    assert direct.get_last_result().batches_used == 2
    outcome = _session_outcome(loose, product, env)
    assert "adaptive:batches_used=2" in outcome.diagnostics.records
    assert "adaptive:stopped_early=True" in outcome.diagnostics.records

    # tight target: runs to max_batches
    outcome = _session_outcome(_rqmc_snowball_engine, product, env)
    assert "adaptive:batches_used=6" in outcome.diagnostics.records
    assert "adaptive:stopped_early=False" in outcome.diagnostics.records


def test_concurrent_same_engine_dispatches_are_serialized():
    """Plan-gate finding 2026-07-16: one engine instance, two environments,
    two racing threads - each result must equal its own single-threaded
    direct price (no mixed-market cross-talk through _df/_term_ctx)."""
    from quantark.param import FlatRateCurve

    product = _snowball()
    env_a = _eq_flat_env()
    env_b = _eq_flat_env()
    env_b.rate_curve = FlatRateCurve(rate=0.15)

    expected = {}
    for key, env in (("a", env_a), ("b", env_b)):
        engine = _rqmc_snowball_engine()
        expected[key] = engine.price(product, env)
    assert expected["a"] != expected["b"]

    shared_engine = _rqmc_snowball_engine()
    results = {}
    errors = []
    barrier = threading.Barrier(2)

    def worker(key, env):
        try:
            barrier.wait(timeout=30)
            with PricingSession() as session:
                outcome = session.execute(
                    shared_engine,
                    PricingRequest(product=product, pricing_env=env),
                )
            results[key] = outcome.value
        except BaseException as exc:  # noqa: BLE001
            errors.append((key, exc))

    threads = [
        threading.Thread(target=worker, args=("a", env_a)),
        threading.Thread(target=worker, args=("b", env_b)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not errors, errors
    assert results["a"] == expected["a"]
    assert results["b"] == expected["b"]
