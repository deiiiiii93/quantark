"""Conservative cell-average projection of discrete PDE event operators.

Root cause: discrete coupon/KO/KI Heaviside transitions were applied through
one-sided Boolean nodal masks while auto_grid snaps thresholds onto nodes,
displacing the effective trigger by ~half a cell (see
quantark/asset/equity/engine/docs/pde_auto_grid_investigation.md).

The fix projects the event jump onto the grid by exact dual-cell averaging of
the piecewise-linear jump function. ``event_projection="cell_average"`` is the
default since the 2026-07-23 repricing review; ``"nodal"`` (with
``event_rannacher_steps=1``) is the explicit legacy opt-out used below to
characterize the bias the projection removes.

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
from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import (
    HestonPhoenixPDESolver,
)
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    HestonSnowballPDESolver,
)
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
    GridVolSurface,
    SpotQuote,
)
from quantark.volmodels.heston import HestonParams
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


class TestProjectEventValues:
    """Review 2026-07-23, blocking finding 1: the straddling node must carry
    the dual-cell average of the COMPLETE post-event function (survive branch
    on one side of the threshold, breach branch on the other). Adding a
    cell-averaged jump to the POINTWISE survive value mixes two inconsistent
    discretizations in that one cell and can leave the branch envelope
    entirely (e.g. produce negative values from two non-negative branches)
    when the survive branch is steep across the straddling cell."""

    def test_reviewer_counterexample_stays_in_envelope(self):
        # Grid x = [-1,-.5,0,.5,1], survive [1,1,1,100,100], breach 0, up
        # threshold at x=0. Both branches are non-negative; the correct cell
        # average over the straddling cell [-0.25, 0.25] is
        # (0.25*1 + 0)/0.5 = 0.5 — the jump-only form gave -11.875.
        x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        v_survive = np.array([1.0, 1.0, 1.0, 100.0, 100.0])
        solver = SnowballPDESolver(PDEParams())
        out = solver._project_event_values(
            np.exp(x), 1.0, False, True, v_survive, 0.0
        )
        assert np.all(out >= 0.0)
        assert out[2] == pytest.approx(0.5, abs=1e-12)
        np.testing.assert_allclose(out[:2], 1.0, atol=1e-14)
        np.testing.assert_allclose(out[3:], 0.0, atol=1e-14)

    def test_envelope_bound_random_branches(self):
        from quantark.asset.equity.engine.pde.event_projection import (
            project_event_values,
        )

        rng = np.random.default_rng(20260723)
        incs = rng.uniform(0.02, 0.1, size=30)
        x = np.concatenate(([0.0], np.cumsum(incs))) - 1.0
        for _ in range(50):
            v_s = rng.uniform(0.0, 10.0, size=x.size)
            v_b = rng.uniform(0.0, 10.0, size=x.size)
            b_x = float(rng.uniform(x[0], x[-1]))
            up = bool(rng.integers(0, 2))
            out = project_event_values(x, b_x, v_s, v_b, up)
            lo = min(v_s.min(), v_b.min())
            hi = max(v_s.max(), v_b.max())
            assert np.all(out >= lo - 1e-12)
            assert np.all(out <= hi + 1e-12)

    def test_straddle_equals_numeric_cell_average(self):
        from quantark.asset.equity.engine.pde.event_projection import (
            project_event_values,
        )

        x = _nonuniform_x()
        rng = np.random.default_rng(5)
        v_s = rng.uniform(-2.0, 5.0, size=x.size)
        v_b = rng.uniform(-2.0, 5.0, size=x.size)
        i = 17
        edges = np.concatenate(([x[0]], 0.5 * (x[1:] + x[:-1]), [x[-1]]))
        e_lo, e_hi = edges[i], edges[i + 1]
        b_x = e_lo + 0.63 * (e_hi - e_lo)
        out = project_event_values(x, b_x, v_s, v_b, breach_up=True)

        def _pl_int(vals, a, c):
            pts = np.unique(
                np.concatenate(([a, c], x[(x > a) & (x < c)]))
            )
            y = np.interp(pts, x, vals)
            return float(np.trapezoid(y, pts))

        expected = (_pl_int(v_s, e_lo, b_x) + _pl_int(v_b, b_x, e_hi)) / (
            e_hi - e_lo
        )
        assert out[i] == pytest.approx(expected, rel=1e-12)
        # away from the straddle: pointwise branch values, untouched
        np.testing.assert_allclose(out[:i], v_s[:i], atol=1e-14)
        np.testing.assert_allclose(out[i + 1 :], v_b[i + 1 :], atol=1e-14)

    def test_affine_survive_uniform_grid_matches_jump_form(self):
        # On a uniform grid the dual cell is symmetric, so the cell average
        # of an affine survive branch equals its nodal value and the
        # straddle correction vanishes: the validated jump-form results are
        # unchanged in this regime.
        from quantark.asset.equity.engine.pde.event_projection import (
            project_event_values,
        )

        x = _uniform_x()
        v_s = 1.4 - 0.8 * x
        rng = np.random.default_rng(9)
        v_b = rng.uniform(0.0, 3.0, size=x.size)
        b_x = x[20] + 0.3 * (x[21] - x[20])
        out = project_event_values(x, b_x, v_s, v_b, breach_up=True)
        jump_form = v_s + project_breach_jump(x, b_x, v_b - v_s, breach_up=True)
        np.testing.assert_allclose(out, jump_form, atol=1e-12)

    def test_columns_match_per_column(self):
        from quantark.asset.equity.engine.pde.event_projection import (
            project_event_values,
        )

        x = _nonuniform_x()
        rng = np.random.default_rng(13)
        v_s = rng.normal(size=(x.size, 3))
        v_b = rng.normal(size=(x.size, 3))
        b_x = x[12] + 1e-3
        out = project_event_values(x, b_x, v_s, v_b, breach_up=False)
        for k in range(3):
            col = project_event_values(
                x, b_x, v_s[:, k], v_b[:, k], breach_up=False
            )
            np.testing.assert_allclose(out[:, k], col, atol=1e-14)


# ---------------------------------------------------------------------------
# Param plumbing
# ---------------------------------------------------------------------------


class TestEventProjectionParam:
    def test_default_is_cell_average(self):
        """Corrected event semantics are the default; "nodal" stays available
        as the explicit legacy opt-out (repricing reviewed 2026-07-23)."""
        assert PDEParams().event_projection is EventProjectionMode.CELL_AVERAGE

    def test_default_event_rannacher_steps_is_two(self):
        """Two implicit solves per event is the field-tested minimum
        (Pooley/d'Halluin); one full BE step has no literature support."""
        assert PDEParams().event_rannacher_steps == 2

    def test_string_coercion(self):
        p = PDEParams(event_projection="nodal")
        assert p.event_projection is EventProjectionMode.NODAL

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
    def test_default_is_projected_and_nodal_optout_differs(
        self, phoenix_product, phoenix_env
    ):
        px_default = _pde_price(phoenix_product, phoenix_env)
        px_proj = _pde_price(
            phoenix_product,
            phoenix_env,
            event_projection=EventProjectionMode.CELL_AVERAGE,
        )
        px_nodal = _pde_price(
            phoenix_product, phoenix_env, event_projection=EventProjectionMode.NODAL,
            event_rannacher_steps=1,
        )
        assert px_default == px_proj
        assert px_default != px_nodal

    def test_cell_average_matches_quad(self, phoenix_product, phoenix_env, quad_ref):
        px_proj = _pde_price(phoenix_product, phoenix_env, event_projection="cell_average")
        px_nodal = _pde_price(
            phoenix_product, phoenix_env,
            event_projection="nodal", event_rannacher_steps=1,
        )
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
        nodal_auto = _pde_price(
            phoenix_product, phoenix_env,
            event_projection="nodal", event_rannacher_steps=1,
        )
        nodal_unif = _pde_price(
            phoenix_product,
            phoenix_env,
            event_projection="nodal",
            event_rannacher_steps=1,
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
            KOResetSnowballPDESolver(
                PDEParams(grid_size=400, event_projection="nodal", event_rannacher_steps=1)
            ).price(product, env)
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
        self._run_discrete_ki_projection_check(solver)

    @staticmethod
    def _run_discrete_ki_projection_check(solver):
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


# ---------------------------------------------------------------------------
# Event-local damping policy (Patch 2): decoupled from auto_grid
# ---------------------------------------------------------------------------


class TestValuationDateEvents:
    """Review 2026-07-23, blocking finding 2: an observation falling exactly
    on the valuation date is deterministic — today's spot is known — so its
    trigger must be applied with the product's exact inclusive comparison
    (e.g. ``PhoenixOption.is_coupon_triggered``), never cell-averaged. The
    nodal application at t=0 additionally owns nodes within ``is_close``
    tolerance of the barrier inclusively: grid nodes are ``exp(log(.))``
    round-trips, so a raw ``>=`` would let 1-ULP noise decide ownership."""

    def test_event_uses_projection_guard(self):
        solver = SnowballPDESolver(PDEParams())
        assert not solver._event_uses_projection(0)
        assert solver._event_uses_projection(5)
        nodal = SnowballPDESolver(PDEParams(event_projection="nodal"))
        assert not nodal._event_uses_projection(5)

    @staticmethod
    def _phoenix_t0(b0: float):
        from quantark.asset.equity.product.option.phoenix_config import (
            CouponBarrierConfig,
        )
        from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
        from quantark.asset.equity.product.option.snowball_config import PayoffConfig
        from quantark.util.calendar.day_counter import DayCountConvention
        from quantark.util.enum import CouponPayType

        bc = BarrierConfig(
            ko_barrier=200.0,
            ko_rate=0.0,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.0, 0.5, 1.0],
            ki_barrier=None,
        )
        cc = CouponBarrierConfig(
            coupon_barrier=[b0, 85.0, 85.0],
            coupon_rate=0.04,
            coupon_pay_type=CouponPayType.INSTANT,
            day_count_convention=DayCountConvention.ACT_365,
            memory_coupon=False,
            fixed_coupon_year_fraction=0.5,
        )
        pf = PayoffConfig(rebate_rate=0.0, include_principal=True)
        return PhoenixOption(
            initial_price=100.0,
            strike=100.0,
            barrier_config=bc,
            coupon_config=cc,
            payoff_config=pf,
            contract_multiplier=1.0,
            maturity=1.0,
        )

    def test_valuation_coupon_at_spot_pays_in_full(self):
        env = _env()

        def _price(b0):
            return float(
                PhoenixPDESolver(params=PDEParams(grid_size=200)).price(
                    self._phoenix_t0(b0), env
                )
            )

        pv_at_barrier = _price(100.0)  # spot exactly on today's coupon barrier
        pv_deep = _price(50.0)  # trivially triggered today
        pv_missed = _price(180.0)  # trivially missed today
        # Inclusive trigger at the known spot: on-barrier == deeply-triggered.
        # Tolerance: auto_grid pins each coupon barrier onto a node, so
        # changing b0 perturbs the MESH and the diffused values by ~1e-5 —
        # far below the ~0.56-coupon (~1.1) gap the cell-averaged t=0 coupon
        # produced, which is the regression this guards against.
        assert pv_at_barrier == pytest.approx(pv_deep, abs=2e-4)
        # ...and the trigger is worth one full (undiscounted, INSTANT) coupon
        expected = float(self._phoenix_t0(50.0).get_coupon_payoff(0, year_fraction=0.5))
        assert pv_deep - pv_missed == pytest.approx(expected, rel=1e-4)

    def test_snowball_valuation_ko_jump_is_exact(self):
        # (The price() path short-circuits a spot-at-or-above-barrier KO at
        # valuation before the PDE runs, so this is exercised at the
        # event-application seam: t_idx == 0 must apply the exact nodal
        # trigger, interior observations must still project.)
        solver = SnowballPDESolver(PDEParams())
        s_vec = np.linspace(60.0, 120.0, 61)
        rec = types.SimpleNamespace(barrier=100.0, payoff=100.0, settlement_time=None)
        product = types.SimpleNamespace(is_reverse=False)
        env = _env()
        j = int(np.argmin(np.abs(s_vec - 100.0)))

        grid_v0 = np.full((61, 3), 55.0)
        grid_v1 = np.full((61, 3), 45.0)
        solver._apply_ko_jump(grid_v0, grid_v1, s_vec, 0, 0.0, product, env, rec)
        assert grid_v0[j, 0] == 100.0
        assert np.all((grid_v0[:, 0] == 55.0) | (grid_v0[:, 0] == 100.0))
        assert np.all((grid_v1[:, 0] == 45.0) | (grid_v1[:, 0] == 100.0))

        grid_v0b = np.full((61, 3), 55.0)
        grid_v1b = np.full((61, 3), 45.0)
        solver._apply_ko_jump(grid_v0b, grid_v1b, s_vec, 1, 0.5, product, env, rec)
        assert 55.0 < grid_v0b[j, 1] < 100.0

    def test_vol_snowball_valuation_ko_is_exact(self):
        solver = HestonSnowballPDESolver(_hp(), params=PDEParams())
        core = types.SimpleNamespace(S_grid=np.linspace(60.0, 120.0, 61))
        product = types.SimpleNamespace(is_reverse=False)
        env = _heston_env()
        rec = types.SimpleNamespace(barrier=100.0, payoff=100.0, settlement_time=None)
        event_maps = {"ko": {4: [rec], 2: [rec]}, "ki": {}, "dt": 0.25}
        U = np.full((61, 3), 55.0)

        # tau == T: the event is at the valuation date -> exact nodal
        # application, no blended node anywhere.
        out = solver._apply_ko(U.copy(), core, product, env, 1.0, 1.0, event_maps)
        j = int(np.argmin(np.abs(core.S_grid - 100.0)))
        assert out[j, 0] == 100.0
        assert np.all((out == 55.0) | (out == 100.0))

        # interior observation (tau < T): the straddling node must still blend.
        out2 = solver._apply_ko(U.copy(), core, product, env, 1.0, 0.5, event_maps)
        assert 55.0 < out2[j, 0] < 100.0

    def test_vol_phoenix_valuation_coupon_is_exact(self):
        from quantark.util.enum import CouponPayType

        solver = HestonPhoenixPDESolver(_hp(), params=PDEParams())
        solver._coupon_barriers = np.array([100.0])
        solver._coupon_amounts = np.array([2.0])
        core = types.SimpleNamespace(S_grid=np.linspace(60.0, 120.0, 61))
        product = types.SimpleNamespace(
            is_reverse=False,
            coupon_config=types.SimpleNamespace(coupon_pay_type=CouponPayType.INSTANT),
        )
        env = _heston_env()
        U = np.full((61, 3), 55.0)

        # tau == T: today's coupon condition is exact — the on-barrier node
        # receives the full coupon, every node pays either 0 or the coupon.
        out = solver._apply_coupon(U.copy(), core, product, env, 1.0, 1.0, 0)
        j = int(np.argmin(np.abs(core.S_grid - 100.0)))
        assert out[j, 0] == 57.0
        assert np.all((out == 55.0) | (out == 57.0))

        # interior observation: the straddling node must still blend.
        out2 = solver._apply_coupon(U.copy(), core, product, env, 1.0, 0.5, 0)
        assert 55.0 < out2[j, 0] < 57.0


class TestEventDampingPolicy:
    def test_event_damping_decoupled_from_auto_grid(self):
        """Damping depends on event regularity, not on the mesh-selection
        mode: an event-aligned grid built with auto_grid=False must damp
        after each discrete event exactly like the auto grid does."""
        from quantark.asset.equity.engine.pde.backward_operator import (
            BackwardOperator,
        )

        t_vec = np.linspace(0.0, 1.0, 11)  # event at t=0.5 sits on node 5
        dt_vec = np.diff(t_vec)
        params = PDEParams(
            auto_grid=False,
            use_rannacher=True,
            rannacher_at_events=True,
            rannacher_steps=1,
            event_rannacher_steps=2,
            event_theta=1.0,
            theta=0.5,
        )
        th = BackwardOperator.theta_by_step(t_vec, dt_vec, params, [0.5])
        # backward steps j=4 (0.5 -> 0.4) and j=3 (0.4 -> 0.3) follow the event
        assert th[4] == 1.0
        assert th[3] == 1.0
        assert th[2] == 0.5

    def test_event_damping_respects_rannacher_at_events_off(self):
        from quantark.asset.equity.engine.pde.backward_operator import (
            BackwardOperator,
        )

        t_vec = np.linspace(0.0, 1.0, 11)
        dt_vec = np.diff(t_vec)
        params = PDEParams(
            auto_grid=False,
            use_rannacher=True,
            rannacher_at_events=False,
            rannacher_steps=1,
            event_rannacher_steps=2,
            event_theta=1.0,
            theta=0.5,
        )
        th = BackwardOperator.theta_by_step(t_vec, dt_vec, params, [0.5])
        assert th[4] == 0.5
        assert th[3] == 0.5


# ---------------------------------------------------------------------------
# Integration: Heston 2D solvers (spot-dimension projection, slice-wise)
# ---------------------------------------------------------------------------


def _heston_env(vol=0.20, s0=100.0, r=0.03, q=0.01):
    strikes = list(s0 * np.exp(np.linspace(-0.5, 0.5, 9)))
    maturities = list(np.linspace(0.25, 1.0, 4))
    surface = GridVolSurface(
        strikes, maturities, np.full((len(maturities), len(strikes)), vol)
    )
    return PricingEnvironment(
        rate_curve=FlatRateCurve(r),
        valuation_date=datetime(2026, 1, 1),
        spot_quote=SpotQuote(spot=s0),
        vol_surface=surface,
        div_yield=ContinuousDividendYield(q),
    )


def _hp():
    return HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)


QUARTERS = [0.25, 0.5, 0.75, 1.0]


class TestVolSolverCellAverage:
    def test_heston_snowball_flag_engages(self):
        env = _heston_env()
        product = SnowballOption(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=1.0,
            barrier_config=BarrierConfig(
                ko_barrier=105.0,
                ko_rate=0.12,
                ko_observation_type=ObservationType.DISCRETE,
                ko_observation_dates=QUARTERS,
                ki_barrier=80.0,
                ki_observation_type=ObservationType.DISCRETE,
                ki_continuous=False,
                ki_observation_dates=QUARTERS,
            ),
        )
        nodal = float(
            HestonSnowballPDESolver(
                _hp(), params=PDEParams(event_projection="nodal", event_rannacher_steps=1),
                n_x=60, n_v=20, n_t=24
            ).price(product, env)
        )
        proj = float(
            HestonSnowballPDESolver(
                _hp(),
                params=PDEParams(event_projection="cell_average"),
                n_x=60,
                n_v=20,
                n_t=24,
            ).price(product, env)
        )
        assert np.isfinite(proj) and proj > 0
        assert proj != nodal, "cell_average must engage in the 2D Heston solver"
        assert abs(proj - nodal) / abs(nodal) < 2e-2

    def test_heston_phoenix_flag_engages(self):
        env = _heston_env()
        ki_sched = ObservationSchedule(
            records=[
                ObservationRecord(observation_time=t, barrier=80.0) for t in QUARTERS
            ]
        )
        product = create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=1.0,
            ko_barrier=105.0,
            ko_rate=0.10,
            ki_barrier=80.0,
            coupon_barrier=90.0,
            coupon_rate=0.02,
            num_observations=4,
            memory_coupon=False,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=ki_sched,
        )
        nodal = float(
            HestonPhoenixPDESolver(
                _hp(), params=PDEParams(event_projection="nodal", event_rannacher_steps=1),
                n_x=60, n_v=20, n_t=24
            ).price(product, env)
        )
        proj = float(
            HestonPhoenixPDESolver(
                _hp(),
                params=PDEParams(event_projection="cell_average"),
                n_x=60,
                n_v=20,
                n_t=24,
            ).price(product, env)
        )
        assert np.isfinite(proj) and proj > 0
        assert proj != nodal, "cell_average must engage in the 2D Heston phoenix"
        # This phoenix has a small net premium (~1.9, no principal), so the
        # half-cell trigger error the projection removes is a large fraction
        # of it at n_x=60 (measured ~6%; the NODAL side is the biased one —
        # its 120->240 refinement moves 0.185 vs 0.005 projected). Loose
        # sanity bound only; accuracy gates live in the 1D suites.
        assert abs(proj - nodal) / abs(nodal) < 0.10

    def test_phoenix_vol_ki_dispatch(self):
        """2D KI transfer: nodal when continuous, projected when discrete."""
        solver = HestonPhoenixPDESolver(
            _hp(), params=PDEParams(event_projection="cell_average")
        )
        core = types.SimpleNamespace(S_grid=np.linspace(60.0, 120.0, 61))
        product = types.SimpleNamespace(is_reverse=False)
        v1 = np.ones((61, 3))

        solver._ki_continuous = True
        solver._bgk_active = False
        out = solver._apply_ki(np.zeros((61, 3)), core, product, 80.0, v1)
        assert set(np.unique(out)) <= {0.0, 1.0}, "continuous KI must stay nodal"

        solver._ki_continuous = False
        out2 = solver._apply_ki(np.zeros((61, 3)), core, product, 80.0, v1)
        j = int(np.argmin(np.abs(core.S_grid - 80.0)))
        assert 0.0 < out2[j, 1] < 1.0, "discrete KI straddle node must blend"
        assert np.all(out2[:j, 1] == 1.0)
        assert np.all(out2[j + 1 :, 1] == 0.0)
