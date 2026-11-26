"""DCN session parity (0.16.0): the ``PricingSession`` ``EVENT_STATS`` operation on all 8 DCN engines.

The session value is the engine's own ``calculate_event_stats`` object, bit for bit (owner ruling: bitwise-only):
stateless, with a mid-life ``DCNLifecycleState`` (pending receivable), on the final observation day (native path,
stateless and stateful), with several MC batches on the serial and threads backends, and for terminated notes
(receivables only; a fully settled note refuses, as the legacy route does).
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
import warnings

import numpy as np
import pytest

from quantark.asset.equity.engine.event_stats import EventStatistics
from quantark.execution import PricingSession
from quantark.execution.contracts import OutputKind, PricingOperation, PricingRequest
from quantark.execution.errors import CapabilityError
from quantark.execution.registry import build_default_registry
from quantark.volmodels.heston import HestonParams

from dcn_fixtures import DCN_A, make_dcn
from execution.matrix_fixtures import FIXTURE_BUILDERS
from execution.test_dcn_batch_adapter import make_batch_context
from test_asset.equity.engine_mc.test_mc_dcn_event_statistics import (
    AGED, FINAL, ROW, ROW_PAY, SETTLED, env_on, flow, state_on,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

HESTON = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)
MC_ENGINES = ["DCNMCEngine", "LocalVolDCNMCEngine", "HestonDCNMCEngine", "QEDCNMCEngine",
              "CoupledCoarseHestonDCNMCEngine"]
PDE_1D = ["DCNPDEEngine", "LocalVolDCNPDEEngine"]
ALL_DCN = MC_ENGINES + PDE_1D + ["HestonDCNPDESolver"]
GRID_ENV = {"LocalVolDCNMCEngine", "LocalVolDCNPDEEngine"}


# --- helpers -----------------------------------------------------------------------------------------------------

def assert_same(a, b, path="stats"):
    """Full recursive bitwise compare: every dataclass field, arrays and floats NaN-aware."""
    assert type(a) is type(b), f"{path}: {type(a).__name__} != {type(b).__name__}"
    if dataclasses.is_dataclass(a):
        for f in dataclasses.fields(a):
            assert_same(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape, path
        if a.dtype.kind in "fc":
            np.testing.assert_array_equal(a, b, err_msg=path)
        else:
            assert np.array_equal(a, b), path
    elif isinstance(a, Mapping):
        assert a.keys() == b.keys(), path
        for k in a:
            assert_same(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert (math.isnan(a) and math.isnan(b)) or a == b, f"{path}: {a!r} != {b!r}"
    else:
        same = a == b
        assert same if isinstance(same, bool) else bool(np.all(same)), f"{path}: {a!r} != {b!r}"


def _request(product, env, state=None):
    return PricingRequest(product=product, pricing_env=env, operation=PricingOperation.EVENT_STATS,
                          outputs=frozenset({OutputKind.PV}), lifecycle_state=state)


def _session(engine, product, env, state=None):
    with PricingSession() as session:
        return session.execute(engine, _request(product, env, state))


def _kernel(engine, product, env, ctx, state=None):
    from quantark.execution.kernel import ExecutionKernel

    return ExecutionKernel.dispatch(engine, _request(product, env, state), ctx)


def _check(outcome, direct):
    assert isinstance(direct, EventStatistics)
    assert_same(outcome.value, direct)
    assert dict(outcome.normalized_economics)["pv"] == direct.pv


def _engine(name, num_batches=None):
    """A fresh DCN engine; ``num_batches`` builds the MC engines with several batches."""
    from quantark.asset.equity.engine.mc import (
        DCNMCEngine, HestonDCNMCEngine, LocalVolDCNMCEngine, QEDCNMCEngine,
    )
    from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import coupled_heston_ladder_pair
    from quantark.asset.equity.engine.pde import DCNPDEEngine, HestonDCNPDESolver, LocalVolDCNPDEEngine
    from quantark.asset.equity.engine.pde import GridConfig
    from quantark.asset.equity.param import PDEParams
    from quantark.util.enum.engine_enums import HestonMCScheme

    mc = {"num_paths": 2048, "seed": 7, "num_batches": num_batches or 1}
    if name == "DCNMCEngine":
        return DCNMCEngine(**mc)
    if name == "LocalVolDCNMCEngine":
        return LocalVolDCNMCEngine(**mc)
    if name == "HestonDCNMCEngine":
        return HestonDCNMCEngine(model_params=HESTON, **mc)
    if name == "QEDCNMCEngine":
        return QEDCNMCEngine(HESTON, **mc)
    if name == "CoupledCoarseHestonDCNMCEngine":
        coarse, _fine = coupled_heston_ladder_pair(HESTON, 2, HestonMCScheme.QUADEXP_M, use_sobol=True, **mc)
        return coarse
    pde = PDEParams(grid=GridConfig(points=301))
    if name == "DCNPDEEngine":
        return DCNPDEEngine(params=pde)
    if name == "LocalVolDCNPDEEngine":
        return LocalVolDCNPDEEngine(params=pde)
    if name == "HestonDCNPDESolver":
        return HestonDCNPDESolver(HESTON, n_x=151, n_v=41)
    raise KeyError(name)


def _env(name, d, spot=6000.0):
    return env_on(d, spot, grid=name in GRID_ENV)


def _receivable_state():
    return state_on(AGED, flow("coupon", ROW, ROW_PAY))


# --- AC1: stateless, every engine (matrix fixtures) -------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_DCN)
def test_session_event_stats_equals_direct_on_the_matrix_fixture(name):
    build = FIXTURE_BUILDERS[name]
    engine, product, env, _shape = build()
    direct = build()[0].calculate_event_stats(product, env)
    _check(_session(engine, product, env), direct)


@pytest.mark.parametrize("name", ALL_DCN)
def test_resolved_adapter_advertises_the_event_stats_operation(name):
    engine = FIXTURE_BUILDERS[name]()[0]
    adapter = build_default_registry().resolve(engine)
    assert PricingOperation.EVENT_STATS in adapter.capabilities().operations


# --- AC1b: several batches, serial and threads ------------------------------------------------------------------

@pytest.mark.parametrize("backend, workers", [("serial", 1), ("threads", 3)])
@pytest.mark.parametrize("name", MC_ENGINES)
def test_multi_batch_session_event_stats_is_bitwise(name, backend, workers):
    product, env = make_dcn(DCN_A), _env(name, DCN_A["valuation_date"])
    direct = _engine(name, num_batches=4).calculate_event_stats(product, env)
    assert direct.standard_errors is not None
    out = _kernel(_engine(name, num_batches=4), product, env, make_batch_context(workers=workers, backend=backend))
    _check(out, direct)


# --- AC2: lifecycle state, final observation day ------------------------------------------------------------------

@pytest.mark.parametrize("name", MC_ENGINES + PDE_1D)
def test_mid_life_receivable_state(name):
    product, env = make_dcn(DCN_A, valuation_date=AGED), _env(name, AGED)
    direct = _engine(name, num_batches=2).calculate_event_stats(product, env, lifecycle_state=_receivable_state())
    assert direct.receivables.size == 1
    _check(_session(_engine(name, num_batches=2), product, env, _receivable_state()), direct)


@pytest.mark.parametrize("stateful", [False, True])
@pytest.mark.parametrize("name", ["DCNMCEngine", "HestonDCNMCEngine", "DCNPDEEngine"])
def test_final_observation_day(name, stateful):
    product, env = make_dcn(DCN_A, valuation_date=FINAL), _env(name, FINAL, 5500.0)
    state = (lambda: state_on(FINAL)) if stateful else (lambda: None)
    direct = _engine(name).calculate_event_stats(product, env, lifecycle_state=state())
    _check(_session(_engine(name), product, env, state()), direct)


# --- AC2b: terminated notes ---------------------------------------------------------------------------------------

def _ko_unpaid():
    return state_on(AGED, flow("ko", ROW, ROW_PAY), knocked_out=True, ko_date=ROW)


@pytest.mark.parametrize("name", ["DCNMCEngine", "HestonDCNMCEngine", "DCNPDEEngine"])
def test_terminated_note_with_unpaid_cash(name):
    product, env = make_dcn(DCN_A, valuation_date=AGED), _env(name, AGED)
    direct = _engine(name).calculate_event_stats(product, env, lifecycle_state=_ko_unpaid())
    assert direct.terminated
    _check(_session(_engine(name), product, env, _ko_unpaid()), direct)


@pytest.mark.parametrize("name", ["DCNMCEngine", "DCNPDEEngine"])
def test_fully_settled_note_refuses(name):
    product, env = make_dcn(DCN_A, valuation_date=SETTLED), _env(name, SETTLED)
    settled = lambda: state_on(SETTLED, flow("ko", ROW, ROW_PAY), knocked_out=True, ko_date=ROW)  # noqa: E731
    assert _engine(name).calculate_event_stats(product, env, lifecycle_state=settled()) is None
    with pytest.raises(CapabilityError):
        _session(_engine(name), product, env, settled())


# --- AC3: the legacy route forwards a state only to engines that declare it --------------------------------------

def test_heston_dcn_pde_forwards_the_lifecycle_state():
    name = "HestonDCNPDESolver"
    product, env = make_dcn(DCN_A, valuation_date=AGED), _env(name, AGED)
    direct = _engine(name).calculate_event_stats(product, env, lifecycle_state=_receivable_state())
    assert direct.receivables.size == 1
    _check(_session(_engine(name), product, env, _receivable_state()), direct)


# --- AC4: DCN PDE adapters do not serve the EVENT_STATS output on PRICE --------------------------------------------

@pytest.mark.parametrize("name", PDE_1D)
def test_dcn_pde_price_with_event_output_still_refused(name):
    engine, product, env, _ = FIXTURE_BUILDERS[name]()
    request = PricingRequest(product=product, pricing_env=env,
                             outputs=frozenset({OutputKind.PV, OutputKind.EVENT_STATS}))
    with PricingSession() as session, pytest.raises(CapabilityError):
        session.execute(engine, request)


@pytest.mark.parametrize("name", ["DCNMCEngine", "DCNPDEEngine"])
def test_event_stats_operation_accepts_only_pv(name):
    engine, product, env, _ = FIXTURE_BUILDERS[name]()
    request = PricingRequest(product=product, pricing_env=env, operation=PricingOperation.EVENT_STATS,
                             outputs=frozenset({OutputKind.PV, OutputKind.ERROR_ESTIMATE}))
    with PricingSession() as session, pytest.raises(CapabilityError):
        session.execute(engine, request)


# --- AC3b (owner, mid-run): a lifecycle state is honoured or refused, never dropped ----------------------------------

def _stateless_statistics_engines():
    """Inventory engines with their own event statistics but no lifecycle support (Snowball / Phoenix / KO-reset
    MC, PDE and QUAD V1, the PDE facade)."""
    import importlib

    from quantark.asset.equity.engine.base_engine import BaseEngine
    from quantark.execution.inventory import ENGINE_INVENTORY

    names = []
    for record in ENGINE_INVENTORY:
        if record.role == "abstract" or record.name not in FIXTURE_BUILDERS or record.asset_family != "equity":
            continue
        module, _, cls_name = record.import_path.rpartition(".")
        cls = getattr(importlib.import_module(module), cls_name)
        if not issubclass(cls, BaseEngine) or cls.calculate_event_stats is BaseEngine.calculate_event_stats:
            continue
        if not getattr(cls, "supports_lifecycle_state", False):
            names.append(record.name)
    return names


STATELESS_STATS = _stateless_statistics_engines()


#: QUAD V1 engines are not inventoried; a session reaches them through the legacy fallback adapter
QUAD_V1 = {"SnowballQuadEngine": "SnowballPDESolver", "PhoenixQuadEngine": "PhoenixPDESolver",
           "KOResetSnowballQuadEngine": "KOResetSnowballPDESolver"}


def _stateless_case(name):
    if name in QUAD_V1:
        import quantark.asset.equity.engine.quad as quad

        _, product, env, _ = FIXTURE_BUILDERS[QUAD_V1[name]]()
        return getattr(quad, name)(), product, env
    engine, product, env, _ = FIXTURE_BUILDERS[name]()
    return engine, product, env


def test_the_stateless_statistics_set_is_the_autocallable_family():
    assert len(STATELESS_STATS) >= 25
    assert {"SnowballMCEngine", "SnowballPDESolver", "PhoenixMCEngine", "HestonSnowballPDESolver"} <= set(
        STATELESS_STATS)


@pytest.mark.parametrize("name", STATELESS_STATS + sorted(QUAD_V1))
def test_stateful_event_stats_refuses_like_stateful_price(name):
    from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState

    engine, product, env = _stateless_case(name)
    state = AutocallableLifecycleState(knocked_in=True)
    with pytest.raises(CapabilityError, match="lifecycle_state"):
        engine.price(product, env, lifecycle_state=state)
    with pytest.raises(CapabilityError, match="lifecycle_state"):
        _session(engine, product, env, state)


# --- final-review fix: the legacy PRICE_DETAILED call forwards the state too ----------------------------------------

def test_heston_dcn_pde_price_detailed_forwards_the_lifecycle_state():
    name = "HestonDCNPDESolver"
    product, env = make_dcn(DCN_A, valuation_date=AGED), _env(name, AGED)
    direct = _engine(name).price_detailed(product, env, lifecycle_state=_receivable_state())
    stateless = _engine(name).price_detailed(product, env)
    assert direct.pv != stateless.pv
    request = PricingRequest(product=product, pricing_env=env, operation=PricingOperation.PRICE_DETAILED,
                             outputs=frozenset({OutputKind.PV}), lifecycle_state=_receivable_state())
    with PricingSession() as session:
        outcome = session.execute(_engine(name), request)
    assert outcome.value.pv == direct.pv
    assert dict(outcome.normalized_economics)["pv"] == direct.pv
