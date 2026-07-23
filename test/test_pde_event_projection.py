"""Conservative cell-average projection of discrete PDE event operators.

Root cause: discrete coupon/KO/KI Heaviside transitions were applied through
one-sided Boolean nodal masks while auto_grid snaps thresholds onto nodes,
displacing the effective trigger by ~half a cell (see
quantark/asset/equity/engine/docs/pde_auto_grid_investigation.md).

The fix projects the event jump onto the grid by exact dual-cell averaging of
the piecewise-linear jump function, opt-in via
``PDEParams(event_projection="cell_average")`` (default stays ``"nodal"``).

Reference values in the integration gates come from the 2026-07-23
investigation reproduction (protected 24-observation Phoenix): PDE nodal
auto-grid error vs QUAD was +1.3e-3 relative at N=400 while the projected
solver sat within 2e-5 of QUAD on both auto and uniform meshes.
"""

import types
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.event_projection import (
    breach_fractions,
    project_breach_jump,
)
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import (
    KOResetSnowballPDESolver,
)
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import (
    KOResetSnowballQuadEngine,
)
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import PDEParams, QuadParams
from quantark.asset.equity.product.option import (
    SnowballOption,
    create_ko_reset_snowball,
    create_standard_phoenix,
)
from quantark.asset.equity.product.option.observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
)
from quantark.asset.equity.product.option.snowball_config import (
    BarrierConfig,
    ProtectionType,
)
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, PostKOScheduleMode
from quantark.util.enum.engine_enums import EventProjectionMode
from quantark.util.exceptions import ValidationError

# ---------------------------------------------------------------------------
# Projector unit properties (pure numerics, no solver)
# ---------------------------------------------------------------------------


def _uniform_x(n=41, x0=-1.0, x1=1.0):
    return np.linspace(x0, x1, n)


def _nonuniform_x(n=41):
    # smooth non-uniform mesh (sinh-clustered around 0.1)
    u = np.linspace(-1.0, 1.0, n)
    return 0.1 + 0.8 * np.sinh(1.5 * u) / np.sinh(1.5)


class TestBreachFractions:
    def test_constant_reproduction_up(self):
        x = _uniform_x()
        h = x[1] - x[0]
        b = 0.5 * (x[20] + x[21])  # exactly on a cell face
        w = breach_fractions(x, b, breach_up=True)
        assert np.all(w[:21] == 0.0)
        assert np.all(w[21:] == 1.0)

    def test_on_node_barrier_splits_dual_cell(self):
        x = _uniform_x()
        b = x[20]  # exactly on a node
        w = breach_fractions(x, b, breach_up=True)
        assert w[20] == pytest.approx(0.5, abs=1e-14)
        assert np.all(w[:20] == 0.0)
        assert np.all(w[21:] == 1.0)

    def test_on_node_nonuniform_weights_by_local_spacing(self):
        x = _nonuniform_x()
        j = 25
        b = x[j]
        dxl = x[j] - x[j - 1]
        dxr = x[j + 1] - x[j]
        w = breach_fractions(x, b, breach_up=True)
        assert w[j] == pytest.approx(dxr / (dxl + dxr), rel=1e-12)

    def test_down_breach_mirror(self):
        x = _uniform_x()
        b = x[20] + 0.3 * (x[21] - x[20])
        w_up = breach_fractions(x, b, breach_up=True)
        w_dn = breach_fractions(x, b, breach_up=False)
        np.testing.assert_allclose(w_up + w_dn, 1.0, atol=1e-14)

    def test_barrier_outside_domain(self):
        x = _uniform_x()
        assert np.all(breach_fractions(x, x[0] - 1.0, breach_up=True) == 1.0)
        assert np.all(breach_fractions(x, x[-1] + 1.0, breach_up=True) == 0.0)

    def test_fp_ownership_invariance(self):
        """A 1-ULP-scale shift of the barrier must not flip a whole cell.

        This is the root-cause regression: with nodal masks, trigger-cell
        ownership was decided by FP roundoff of exp(log(B)).
        """
        x = _uniform_x()
        b = x[20]
        w0 = breach_fractions(x, b, breach_up=True)
        w1 = breach_fractions(x, b * (1.0 + 1e-13) if b != 0 else 1e-13, breach_up=True)
        assert np.max(np.abs(w1 - w0)) < 1e-8


class TestProjectBreachJump:
    def test_constant_jump_reproduction(self):
        x = _uniform_x()
        d = np.full_like(x, 3.7)
        b = x[20]
        j = project_breach_jump(x, b, d, breach_up=True)
        assert np.all(j[:20] == 0.0)
        np.testing.assert_allclose(j[21:], 3.7, rtol=1e-14)
        assert j[20] == pytest.approx(3.7 * 0.5, rel=1e-12)

    def test_affine_jump_exact_in_straddle_cell(self):
        x = _uniform_x()
        h = x[1] - x[0]
        a, c = 0.4, 2.0
        d = a + c * x
        jnode = 20
        b = x[jnode] + 0.3 * h  # inside dual cell of node 20
        out = project_breach_jump(x, b, d, breach_up=True)
        # exact cell average of 1{x>=b}(a+cx) over [x_j - h/2, x_j + h/2]
        upper = x[jnode] + 0.5 * h
        seg = upper - b
        exact = seg * (a + c * 0.5 * (b + upper)) / h
        assert out[jnode] == pytest.approx(exact, rel=1e-12)
        np.testing.assert_allclose(out[jnode + 1 :], d[jnode + 1 :], rtol=1e-12)
        assert np.all(out[:jnode] == 0.0)

    def test_linearity(self):
        x = _nonuniform_x()
        rng = np.random.default_rng(7)
        d1 = rng.normal(size=x.shape)
        d2 = rng.normal(size=x.shape)
        b = 0.5 * (x[18] + x[19]) + 1e-3
        j12 = project_breach_jump(x, b, 2.0 * d1 - 0.5 * d2, breach_up=True)
        j1 = project_breach_jump(x, b, d1, breach_up=True)
        j2 = project_breach_jump(x, b, d2, breach_up=True)
        np.testing.assert_allclose(j12, 2.0 * j1 - 0.5 * j2, atol=1e-12)

    def test_constant_mass_conservation(self):
        """Sum of dual-cell masses of the projected jump equals the exact
        measure of the breach region for a constant jump."""
        x = _nonuniform_x()
        d = np.ones_like(x)
        b = x[22] + 0.37 * (x[23] - x[22])
        j = project_breach_jump(x, b, d, breach_up=True)
        edges = np.concatenate(([x[0]], 0.5 * (x[1:] + x[:-1]), [x[-1]]))
        cell_w = np.diff(edges)
        assert float(np.sum(cell_w * j)) == pytest.approx(x[-1] - b, rel=1e-12)

    def test_no_overshoot(self):
        x = _nonuniform_x()
        rng = np.random.default_rng(11)
        d = rng.uniform(0.5, 2.0, size=x.shape)
        b = x[20] + 1e-4
        j = project_breach_jump(x, b, d, breach_up=True)
        assert np.all(j >= -1e-14)
        lo = np.minimum(np.minimum(np.roll(d, 1), d), np.roll(d, -1))
        hi = np.maximum(np.maximum(np.roll(d, 1), d), np.roll(d, -1))
        assert np.all(j <= hi + 1e-14)
        interior = (j > 1e-13) & (np.abs(j - d) > 1e-13)
        assert np.all(j[interior] <= hi[interior] + 1e-14)
        assert np.all(j[interior] >= 0.0)

    def test_2d_columns_match_per_column(self):
        x = _uniform_x()
        rng = np.random.default_rng(3)
        d = rng.normal(size=(x.size, 3))
        b = x[15] + 2e-3
        j2 = project_breach_jump(x, b, d, breach_up=False)
        for k in range(3):
            jk = project_breach_jump(x, b, d[:, k], breach_up=False)
            np.testing.assert_allclose(j2[:, k], jk, atol=1e-14)


# ---------------------------------------------------------------------------
# Param plumbing
# ---------------------------------------------------------------------------


class TestEventProjectionParam:
    def test_default_is_nodal(self):
        assert PDEParams().event_projection is EventProjectionMode.NODAL

    def test_string_coercion(self):
        p = PDEParams(event_projection="cell_average")
        assert p.event_projection is EventProjectionMode.CELL_AVERAGE

    def test_invalid_value_raises(self):
        with pytest.raises(ValidationError):
            PDEParams(event_projection="midpoint_magic")


# ---------------------------------------------------------------------------
# Integration: investigation instrument (protected 24-obs Phoenix)
# ---------------------------------------------------------------------------

MAT = 2.0
N_OBS = 24
OBS_TIMES = [(i + 1) / N_OBS * MAT for i in range(N_OBS)]


def _phoenix(memory: bool):
    ki_sched = ObservationSchedule(
        records=[ObservationRecord(observation_time=t, barrier=80.0) for t in OBS_TIMES]
    )
    return create_standard_phoenix(
        initial_price=100.0,
        strike=100.0,
        maturity=MAT,
        contract_multiplier=1.0,
        ko_barrier=100.0,
        ko_rate=0.08,
        ki_barrier=80.0,
        coupon_barrier=85.0,
        coupon_rate=0.01,
        num_observations=N_OBS,
        memory_coupon=memory,
        ki_continuous=False,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_schedule=ki_sched,
        include_principal=True,
        protection_type=ProtectionType.PARTIAL,
        protection_rate=0.80,
    )


def _env(spot=100.0, vol=0.22, rate=0.025, q=0.04):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=q),
        valuation_date=datetime(2024, 1, 1),
    )


@pytest.fixture(scope="module")
def phoenix_env():
    return _env()


@pytest.fixture(scope="module")
def phoenix_product():
    return _phoenix(memory=False)


@pytest.fixture(scope="module")
def quad_ref(phoenix_product, phoenix_env):
    return float(
        PhoenixQuadEngine(params=QuadParams(grid_points=801)).price(
            phoenix_product, phoenix_env
        )
    )


def _pde_price(product, env, **params):
    return float(PhoenixPDESolver(params=PDEParams(grid_size=400, **params)).price(product, env))


class TestPhoenixCellAverage:
    def test_default_price_bitwise_unchanged(self, phoenix_product, phoenix_env):
        px_default = _pde_price(phoenix_product, phoenix_env)
        px_nodal = _pde_price(
            phoenix_product, phoenix_env, event_projection=EventProjectionMode.NODAL
        )
        assert px_default == px_nodal

    def test_cell_average_matches_quad(self, phoenix_product, phoenix_env, quad_ref):
        px_proj = _pde_price(phoenix_product, phoenix_env, event_projection="cell_average")
        px_nodal = _pde_price(phoenix_product, phoenix_env)
        err_proj = abs(px_proj - quad_ref) / quad_ref
        err_nodal = abs(px_nodal - quad_ref) / quad_ref
        assert err_proj < 3e-4, f"projected PDE {px_proj} vs quad {quad_ref}"
        # documents the bug the projection fixes (nodal ~1.3e-3 at N=400)
        assert err_nodal > 2.5e-4
        assert err_proj < 0.5 * err_nodal

    def test_cell_average_phase_robust(self, phoenix_product, phoenix_env, quad_ref):
        """Auto (snapped adaptive) and uniform meshes must agree once events
        are projected — the grid-phase lottery is the bug."""
        proj_auto = _pde_price(phoenix_product, phoenix_env, event_projection="cell_average")
        proj_unif = _pde_price(
            phoenix_product,
            phoenix_env,
            event_projection="cell_average",
            auto_grid=False,
            adaptive_grid=False,
            time_grid_type="event_aligned",
        )
        nodal_auto = _pde_price(phoenix_product, phoenix_env)
        nodal_unif = _pde_price(
            phoenix_product,
            phoenix_env,
            auto_grid=False,
            adaptive_grid=False,
            time_grid_type="event_aligned",
        )
        assert abs(proj_auto - proj_unif) / quad_ref < 2e-4
        assert abs(nodal_auto - nodal_unif) / quad_ref > 6e-4

    def test_memory_coupon_cell_average(self, phoenix_env):
        product = _phoenix(memory=True)
        env = _env(spot=86.0, vol=0.16, q=0.06)
        quad = float(
            PhoenixQuadEngine(params=QuadParams(grid_points=1601)).price(product, env)
        )
        px_proj = _pde_price(product, env, event_projection="cell_average")
        assert abs(px_proj - quad) / quad < 6e-4

    def test_event_stats_cell_average_consistent_with_quad(
        self, phoenix_product, phoenix_env
    ):
        quad_stats = PhoenixQuadEngine(
            params=QuadParams(grid_points=801)
        ).calculate_event_stats(phoenix_product, phoenix_env)
        pde_stats = PhoenixPDESolver(
            params=PDEParams(grid_size=400, event_projection="cell_average")
        ).calculate_event_stats(phoenix_product, phoenix_env)
        ko_diff = np.max(
            np.abs(np.asarray(pde_stats.ko_probability) - np.asarray(quad_stats.ko_probability))
        )
        cp_diff = np.max(
            np.abs(
                np.asarray(pde_stats.coupon_probability)
                - np.asarray(quad_stats.coupon_probability)
            )
        )
        assert ko_diff < 3e-3
        assert cp_diff < 3e-3
        assert float(np.min(np.asarray(pde_stats.ko_probability))) >= -1e-8
        assert float(np.sum(np.asarray(pde_stats.ko_probability))) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Integration: snowball (KO + discrete KI path)
# ---------------------------------------------------------------------------


def _snowball(ki_continuous: bool):
    cfg = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.15,
        ko_observation_dates=OBS_TIMES,
        ki_barrier=80.0,
        ki_continuous=ki_continuous,
        ki_observation_type=(
            ObservationType.CONTINUOUS if ki_continuous else ObservationType.DISCRETE
        ),
        ki_observation_dates=None if ki_continuous else OBS_TIMES,
    )
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=cfg,
        contract_multiplier=1.0,
        maturity=MAT,
    )


class TestSnowballCellAverage:
    def test_discrete_ki_snowball_matches_quad(self, phoenix_env):
        product = _snowball(ki_continuous=False)
        quad = float(
            SnowballQuadEngine(params=QuadParams(grid_points=801)).price(
                product, phoenix_env
            )
        )
        px_proj = float(
            SnowballPDESolver(
                params=PDEParams(grid_size=400, event_projection="cell_average")
            ).price(product, phoenix_env)
        )
        assert abs(px_proj - quad) / abs(quad) < 6e-4

    def test_continuous_ki_application_stays_nodal(self):
        """Continuous KI monitoring is a continuous-barrier treatment: the
        per-step V0<-V1 transfer must remain a nodal mask even when
        cell_average projection is enabled (projection is for discrete events
        only)."""
        solver = SnowballPDESolver(
            params=PDEParams(event_projection="cell_average")
        )
        solver._ki_continuous = True
        solver._bgk_active = False
        solver._resolve_ki_barrier_at_tidx = types.MethodType(
            lambda self, t_idx: 80.0, solver
        )
        s_vec = np.linspace(60.0, 120.0, 61)  # node exactly at 80
        grid_v0 = np.zeros((61, 3))
        grid_v1 = np.ones((61, 3))
        product = types.SimpleNamespace(is_reverse=False)
        solver._apply_ki_jump(grid_v0, grid_v1, s_vec, 1, product)
        mask = s_vec <= 80.0
        assert np.all(grid_v0[mask, 1] == 1.0)
        assert np.all(grid_v0[~mask, 1] == 0.0)
        # no fractional values anywhere: nodal application
        assert set(np.unique(grid_v0[:, 1])) <= {0.0, 1.0}

    def test_ko_reset_cell_average_engages_and_stays_consistent(self):
        """The KO-reset solver has its own KO application sites; the flag must
        reach them (prices move) without degrading quad agreement."""
        env = _env(vol=0.20, rate=0.05, q=0.02)
        product = create_ko_reset_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity_pre=1.0,
            maturity_post=2.0,
            post_ko_mode=PostKOScheduleMode.ABSOLUTE,
            ki_continuous=False,
        )
        nodal = float(
            KOResetSnowballPDESolver(PDEParams(grid_size=400)).price(product, env)
        )
        proj = float(
            KOResetSnowballPDESolver(
                PDEParams(grid_size=400, event_projection="cell_average")
            ).price(product, env)
        )
        quad = float(
            KOResetSnowballQuadEngine(QuadParams(grid_points=801)).price(product, env)
        )
        assert proj != nodal
        assert abs(proj - quad) / abs(quad) < 2e-3
        assert abs(proj - quad) <= 1.5 * abs(nodal - quad) + 1e-12

    def test_discrete_ki_application_is_projected(self):
        solver = SnowballPDESolver(
            params=PDEParams(event_projection="cell_average")
        )
        solver._ki_continuous = False
        solver._bgk_active = False
        solver._ki_observation_indices = {1}
        solver._resolve_ki_barrier_at_tidx = types.MethodType(
            lambda self, t_idx: 80.0, solver
        )
        s_vec = np.linspace(60.0, 120.0, 61)  # node exactly at 80
        grid_v0 = np.zeros((61, 3))
        grid_v1 = np.ones((61, 3))
        product = types.SimpleNamespace(is_reverse=False)
        solver._apply_ki_jump(grid_v0, grid_v1, s_vec, 1, product)
        j = int(np.argmin(np.abs(s_vec - 80.0)))
        assert 0.0 < grid_v0[j, 1] < 1.0, "straddle node must be fractionally blended"
        assert np.all(grid_v0[:j, 1] == 1.0)
        assert np.all(grid_v0[j + 1 :, 1] == 0.0)
