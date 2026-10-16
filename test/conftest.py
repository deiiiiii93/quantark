"""Shared pytest fixtures for the PDE autocallable event-distribution redesign.

These fixtures back the Phase 1–4 tests of the PDE event-distribution redesign
(plan: docs/superpowers/plans/2026-07-01-pde-autocallable-event-distribution-redesign.md).
They construct real products / solvers / pricing environments so the tests
exercise production constructors, not mocks.
"""

import os as _os
import tempfile as _tempfile

# Point matplotlib at a shared writable config dir before anything imports it.
# The sandbox home (and therefore ~/.matplotlib) is not writable, so without
# this every test worker rebuilds the font cache in a fresh temp dir on every
# run. The shared dir lets the cache persist across workers and runs.
_mplconfig = _os.environ.get(
    "MPLCONFIGDIR", _os.path.join(_tempfile.gettempdir(), "quantark-mplconfig")
)
try:
    _os.makedirs(_mplconfig, exist_ok=True)
    _os.environ.setdefault("MPLCONFIGDIR", _mplconfig)
except OSError:
    pass
del _os, _tempfile, _mplconfig

from datetime import datetime
from typing import List, Optional

import pytest

# --- worker memory ---------------------------------------------------------
# The numerics tests allocate multi-GB grids. Python frees them, but the C
# allocator keeps the pages, so a long-lived xdist worker's RSS only ratchets up
# (measured: one worker reached ~9 GB, 4 workers ~24 GB). A 16 GB CI runner is
# then shut down mid-suite ("The runner has received a shutdown signal"). After
# any test that ran at least this long, collect cycles and hand freed pages back.
# Measured on the six heaviest files, serial: peak RSS 10.2 GB -> 4.3 GB, final
# 10.1 GB -> 2.4 GB, same wall-clock. Short tests are skipped so the 17k tiny ones
# pay nothing.
_RELIEF_AFTER_SECONDS = 0.5


def _build_allocator_relief():
    import ctypes
    import ctypes.util
    import sys

    try:
        if sys.platform == "darwin":
            lib = ctypes.CDLL(ctypes.util.find_library("System"))
            return lambda: lib.malloc_zone_pressure_relief(None, 0)
        if sys.platform.startswith("linux"):
            lib = ctypes.CDLL("libc.so.6")
            return lambda: lib.malloc_trim(0)
    except (OSError, AttributeError, TypeError):
        pass
    return None


_allocator_relief = _build_allocator_relief()


def pytest_runtest_logreport(report):
    if report.when != "call" or report.duration < _RELIEF_AFTER_SECONDS:
        return
    import gc

    gc.collect()
    if _allocator_relief is not None:
        _allocator_relief()


from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import (
    KOResetSnowballPDESolver,
)
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option import create_ko_reset_snowball
from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.util.enum import PostKOScheduleMode
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import BarrierType, ObservationType, OptionType


# ---------------------------------------------------------------------------
# Pricing environment
# ---------------------------------------------------------------------------
def _make_env(
    spot: float = 100.0,
    vol: float = 0.20,
    rate: float = 0.03,
    div_yield: float = 0.01,
) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div_yield),
        valuation_date=datetime(2024, 1, 1),
    )


@pytest.fixture
def pricing_env() -> PricingEnvironment:
    return _make_env()


# ---------------------------------------------------------------------------
# Vanilla discrete barrier (base-solver _time_grid_spec default path)
# ---------------------------------------------------------------------------
@pytest.fixture
def vanilla_barrier_product() -> BarrierOption:
    """Up-and-out call with a discrete monthly observation schedule.

    Used to exercise the BASE ``_time_grid_spec`` default (no KI-monitor
    concept; align to the generic observation schedule).
    """
    monthly = [i / 12 for i in range(1, 12)]  # 11 interior monthly dates
    return BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=120.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        observation_type=ObservationType.DISCRETE,
        observation_dates=monthly,
    )


# ---------------------------------------------------------------------------
# Snowball builders across the four KI regimes (Task 1.4 table)
# ---------------------------------------------------------------------------
def _daily_dates(n: int = 252) -> List[float]:
    return [i / n for i in range(1, n + 1)]


def _monthly_ko_dates() -> List[float]:
    return [i / 12 for i in range(1, 13)]  # includes 1.0 (maturity)


def _make_snowball(
    *,
    ki_barrier: Optional[float] = 75.0,
    ki_observation_type: ObservationType = ObservationType.DISCRETE,
    ki_observation_dates: Optional[List[float]] = None,
    ki_continuous: bool = False,
    maturity: float = 1.0,
    is_reverse: bool = False,
    ko_barrier: float = 103.0,
) -> SnowballOption:
    cfg = BarrierConfig(
        ko_barrier=ko_barrier,
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=_monthly_ko_dates(),
        ki_barrier=ki_barrier,
        ki_observation_type=ki_observation_type,
        ki_observation_dates=ki_observation_dates,
        ki_continuous=ki_continuous,
    )
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=cfg,
        contract_multiplier=10_000.0,
        maturity=maturity,
        is_reverse=is_reverse,
    )


@pytest.fixture
def snowball_daily_ki() -> SnowballOption:
    """Daily-discrete KI: KI observed on ~252 interior daily dates."""
    return _make_snowball(
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=_daily_dates(252),
        ki_continuous=False,
    )


@pytest.fixture
def snowball_euro_ki() -> SnowballOption:
    """European KI: KI can only trigger at maturity (single obs at t=tau)."""
    return _make_snowball(
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=[1.0],
        ki_continuous=False,
    )


@pytest.fixture
def snowball_cont_ki() -> SnowballOption:
    """Continuous KI."""
    return _make_snowball(
        ki_observation_type=ObservationType.CONTINUOUS,
        ki_observation_dates=None,
        ki_continuous=True,
    )


@pytest.fixture
def snowball_no_ki() -> SnowballOption:
    """No KI barrier (mandatory-put-free / pure KO note)."""
    return _make_snowball(ki_barrier=None, ki_continuous=False)


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------
@pytest.fixture
def snowball_solver() -> SnowballPDESolver:
    return SnowballPDESolver()


@pytest.fixture
def ko_reset_product():
    """KO-reset snowball with pre (0,1] KO and post (1,2] KO in ABSOLUTE mode."""
    return create_ko_reset_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity_pre=1.0,
        maturity_post=2.0,
        post_ko_mode=PostKOScheduleMode.ABSOLUTE,
        ki_continuous=False,
    )


@pytest.fixture
def ko_reset_solver() -> KOResetSnowballPDESolver:
    return KOResetSnowballPDESolver(PDEParams())


@pytest.fixture
def snowball_pde_solver_factory():
    """Factory building a SnowballPDESolver with overridden PDEParams fields."""

    def _factory(**param_overrides) -> SnowballPDESolver:
        return SnowballPDESolver(params=PDEParams(**param_overrides))

    return _factory


# ---------------------------------------------------------------------------
# Convergence-gate cases (Task 1.6): (product, env) per KI-regime row
# ---------------------------------------------------------------------------
@pytest.fixture
def autocallable_case():
    """Return a builder mapping a row name to (product, pricing_env).

    Rows exercise the three KI regimes that stress the time grid differently:
    daily-discrete KI (dense monitor nodes), European KI (KO-aligned only), and
    a monthly-KO continuous-KI note (sparse align, resolution-fill dependent).
    All share the standard env (spot 100, vol 0.20, r 0.03, q 0.01).
    """

    def _build(row: str):
        env = _make_env(spot=100.0, vol=0.20, rate=0.03, div_yield=0.01)
        if row == "daily_ki":
            product = _make_snowball(
                ki_observation_type=ObservationType.DISCRETE,
                ki_observation_dates=_daily_dates(252),
                ki_continuous=False,
            )
        elif row == "euro_ki":
            product = _make_snowball(
                ki_observation_type=ObservationType.DISCRETE,
                ki_observation_dates=[1.0],
                ki_continuous=False,
            )
        elif row == "monthly_ko":
            product = _make_snowball(
                ki_observation_type=ObservationType.CONTINUOUS,
                ki_observation_dates=None,
                ki_continuous=True,
            )
        else:
            raise ValueError(f"unknown autocallable row: {row}")
        return product, env

    return _build
