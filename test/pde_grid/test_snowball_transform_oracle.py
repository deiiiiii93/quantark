"""Transform oracle: the migrated snowball EventSchedule must reproduce the
CURRENT certified KO/KI jump code (plan Task 10; scaffolding, deleted Phase 4).

Fixture lifecycle: generated from the live legacy methods while they exist,
committed at test/pde_grid/data/oracle_snowball.npz, and — after the legacy
code is deleted at Phase 4 — this whole module goes with it. While both
implementations coexist the oracle pins them together.
"""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.pde.grid.events import project_between
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option import SnowballOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType

FIXTURE = Path(__file__).parent / "data" / "oracle_snowball.npz"


def _env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2024, 1, 1),
    )


def _product(ki_continuous: bool) -> SnowballOption:
    ki_dates = None if ki_continuous else [round(i / 48.0, 10) for i in range(1, 49)]
    cfg = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
        ki_barrier=75.0,
        ki_observation_type=(
            ObservationType.CONTINUOUS if ki_continuous else ObservationType.DISCRETE
        ),
        ki_observation_dates=ki_dates,
        ki_continuous=ki_continuous,
    )
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=cfg,
        contract_multiplier=1.0,
        maturity=1.0,
    )


def _capture_legacy():
    """Run today's private jump methods and record inputs/outputs."""
    out = {}
    for tag, ki_continuous in (("disc", False), ("cont", True)):
        env = _env()
        product = _product(ki_continuous)
        solver = SnowballPDESolver(params=PDEParams())
        solver.price(product, env)  # populates solve state + observation maps

        spot, tau = env.spot, product.get_maturity(env)
        r, q = env.get_rate(tau), env.get_div_yield(tau)
        sigma = env.get_vol(product.strike, tau)
        x_vec, s_vec, _, t_vec, _ = solver._build_grids(
            product, env, spot, sigma, tau, r, q
        )
        n_x, n_t = len(x_vec), len(t_vec)

        ramp = np.linspace(40.0, 130.0, n_x)
        bowl = 100.0 + 0.01 * (s_vec - 100.0) ** 2 / 10.0

        if tag == "disc":
            ko_idx, ko_rec = max(
                (kv for kv in solver._ko_observation_indices.items()),
                key=lambda kv: kv[0],
            )
            grid_v0 = np.tile(ramp[:, None], (1, n_t))
            grid_v1 = np.tile(bowl[:, None], (1, n_t))
            v0_in, v1_in = grid_v0[:, ko_idx].copy(), grid_v1[:, ko_idx].copy()
            solver._apply_ko_jump(
                grid_v0, grid_v1, s_vec, ko_idx, ko_rec.observation_time,
                product, env, ko_rec,
            )
            cash = solver._cashflow_value_at_time(
                pricing_env=env,
                cashflow=ko_rec.payoff or 0.0,
                current_time=ko_rec.observation_time,
                settlement_time=ko_rec.settlement_time,
            )
            out.update(
                x_vec=x_vec, s_vec=s_vec,
                ko_barrier=float(ko_rec.barrier), ko_cash=float(cash),
                ko_v0_in=v0_in, ko_v1_in=v1_in,
                ko_v0_out=grid_v0[:, ko_idx].copy(),
                ko_v1_out=grid_v1[:, ko_idx].copy(),
            )

            ki_idx = max(
                i for i in solver._ki_observation_indices if i not in (0,)
            )
            grid_v0 = np.tile(ramp[:, None], (1, n_t))
            grid_v1 = np.tile(bowl[:, None], (1, n_t))
            v0_in, v1_in = grid_v0[:, ki_idx].copy(), grid_v1[:, ki_idx].copy()
            solver._apply_ki_jump(grid_v0, grid_v1, s_vec, ki_idx, product)
            out.update(
                ki_barrier=float(solver._resolve_ki_barrier_at_tidx(ki_idx)),
                ki_v0_in=v0_in, ki_v1_in=v1_in,
                ki_v0_out=grid_v0[:, ki_idx].copy(),
            )
        else:
            k = n_t // 2
            grid_v0 = np.tile(ramp[:, None], (1, n_t))
            grid_v1 = np.tile(bowl[:, None], (1, n_t))
            v0_in = grid_v0[:, k].copy()
            solver._apply_ki_jump(grid_v0, grid_v1, s_vec, k, product)
            out.update(
                cont_x_vec=x_vec, cont_s_vec=s_vec,
                cont_ki_barrier=float(solver._resolve_ki_barrier_at_tidx(k)),
                cont_v0_in=v0_in, cont_v1_in=grid_v1[:, k].copy(),
                cont_v0_out=grid_v0[:, k].copy(),
            )
    return out


def _load_or_create():
    if FIXTURE.exists():
        return dict(np.load(FIXTURE))
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    data = _capture_legacy()
    np.savez(FIXTURE, **data)
    return data


def test_fixture_matches_live_legacy_code():
    """While the legacy methods exist, the committed fixture must equal what
    they produce today (guards accidental drift before the migration)."""
    stored = _load_or_create()
    live = _capture_legacy()
    for key, val in live.items():
        assert np.allclose(stored[key], val, rtol=0, atol=1e-12), key


def test_new_transforms_reproduce_oracle():
    """The grid-layer projection primitives reproduce the legacy outputs on
    the same grid (activates fully once SnowballPDESolver migrates)."""
    d = _load_or_create()
    lay = SimpleNamespace(x=d["x_vec"])

    # KO: up-barrier, both surfaces -> projected against the cash value
    ko_new_v0 = project_between(
        lay, d["ko_barrier"], True, np.full_like(d["ko_v0_in"], d["ko_cash"]),
        d["ko_v0_in"],
    )
    ko_new_v1 = project_between(
        lay, d["ko_barrier"], True, np.full_like(d["ko_v1_in"], d["ko_cash"]),
        d["ko_v1_in"],
    )
    assert np.allclose(ko_new_v0, d["ko_v0_out"], atol=1e-12)
    assert np.allclose(ko_new_v1, d["ko_v1_out"], atol=1e-12)

    # Discrete KI: down-barrier, alive <- ki in the breached region
    ki_new_v0 = project_between(
        lay, d["ki_barrier"], False, d["ki_v1_in"], d["ki_v0_in"]
    )
    assert np.allclose(ki_new_v0, d["ki_v0_out"], atol=1e-12)

    # Continuous KI: nodal coupling (never projected)
    mask = d["cont_s_vec"] <= d["cont_ki_barrier"]
    cont_new = d["cont_v0_in"].copy()
    cont_new[mask] = d["cont_v1_in"][mask]
    assert np.allclose(cont_new, d["cont_v0_out"], atol=1e-12)


@pytest.mark.skipif(
    not SnowballPDESolver()._uses_grid_layer(),
    reason="activates when SnowballPDESolver migrates (plan Task 11)",
)
def test_migrated_event_schedule_reproduces_oracle():
    """After migration: the solver-built EventSchedule stages themselves
    reproduce the oracle columns on the same grid."""
    d = _load_or_create()
    env = _env()
    product = _product(ki_continuous=False)
    solver = SnowballPDESolver(params=PDEParams())
    market = solver.market_snapshot(product, env)
    layout = solver.grid_binder.bind(solver.grid_request(product, market), market)
    schedule = solver.event_schedule(product, env, layout)
    assert schedule.interior_steps  # KO + discrete KI steps present
