"""The knocked-in surface's knock-in-edge row honours an annualized knock-in leg (unified event statistics D26, AC30).

``get_maturity_payoff_v1`` scales the knocked-in downside by the contract tenor when the knock-in leg is annualized
(Snowball: the contract tenor; KO-reset: the post-KI tenor). The knocked-in Dirichlet row on the knock-in edge (lower
on a standard product, upper on a reverse one; ``_knock_in_side_row`` copies it into the not-knocked-in surface, and
the Heston / SLV ADI writes the same row) must be the discounted forward value of THAT payoff's asymptote, read off the
payoff itself. Before D26 it omitted the tenor: an LV KO-reset priced 96.16 at the default width against a true 94.10.

Gates (AC30):

* the row, at any step, is ``E_fwd[P(S_T)]`` of the payoff's affine piece at the edge, with the slope set by the
  contract (participation x multiplier x tenor), for Snowball (standard and reverse, unprotected and partially
  protected) and KO-reset; the non-annualized row is the legacy closed form, bit for bit;
* the price no longer depends on the domain width: width (``num_std``) 3 == width 16 within the PDE's discretisation
  allowance, on PDE 1D (flat and local vol) and the Heston / SLV ADI;
* the price matches an independent MC (local vol: 1-day SDE steps; Heston / SLV: QE substeps) within 4 SE.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from quantark.asset.equity.engine.pde import GridConfig
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.pde.ko_reset_vol_pde_solvers import (
    HestonKOResetSnowballPDESolver,
    HestonSLVKOResetSnowballPDESolver,
    LocalVolKOResetSnowballPDESolver,
)
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    HestonSLVSnowballPDESolver,
    HestonSnowballPDESolver,
    LocalVolSnowballPDESolver,
)
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, PayoffConfig
from quantark.util.enum.option_enums import ProtectionType
from test_asset.equity.engine_mc import mc_price_path_routes as grid

QUARTERS_2Y = tuple(0.25 * (i + 1) for i in range(8))
QUARTERS_6M = (0.25, 0.5)
ANNUALIZED_KI = dict(is_annualized=True, is_annualized_ki=True, is_annualized_rebate=True)

#: PDE 1D width gate: |price(num_std=3, 400 points) - price(num_std=16, 1600 points)|. Measured with the fix (T14b
#: report): <= 4.5e-3 on local vol, <= 3.4e-4 on flat vol; with base's row 2.3 to 3.5 (LV) and 0.14 (flat)
WIDTH_TOL_1D = 1e-2
#: ADI width gate: num_std 3 on n_x 160 vs num_std 16 on n_x 640 (the same spacing). Measured with the fix (T14b
#: report): <= 1.6e-3 (Heston and SLV, Snowball and KO-reset); with base's row 0.71 to 1.46
WIDTH_TOL_ADI = 5e-3


def annualized_snowball(maturity: float = 2.0, *, reverse: bool = False, protection: float | None = None):
    """A discretely knocked-in Snowball whose knock-in leg is annualized (the contract tenor is its maturity)."""
    dates = QUARTERS_2Y if maturity == 2.0 else QUARTERS_6M
    product = grid.snowball("discrete", ko_dates=dates, maturity=maturity, reverse=reverse)
    product.accrual_config = AccrualConfig(**ANNUALIZED_KI)
    if protection is not None:
        product.payoff_config = PayoffConfig(rebate_rate=0.05, include_principal=True,
                                             protection_type=ProtectionType.PARTIAL, protection_rate=protection)
    product.validate()
    return product


def _solved_s_grid(solver, product, env):
    """Solve once (setting the solver's state) and return the spot nodes its knocked-in rows were written on."""
    seen = {}
    own = solver._boundary_rows_v1

    def spy(t_idx, tau, p, e, s_vec):
        seen["s"] = np.array(s_vec, dtype=float)
        return own(t_idx, tau, p, e, s_vec)

    solver._boundary_rows_v1 = spy
    try:
        solver.price(product, env)
    finally:
        del solver._boundary_rows_v1
    return seen["s"]


def _forward_affine(product, env, solver, tau, s_edge, tenor):
    """``E_fwd[P(S_T)]`` of the payoff's affine piece through the knock-in edge, from the contract terms alone:
    principal + participation x multiplier x tenor x (S_T - K) (standard) or x (K - S_T) (reverse)."""
    df, df_div = solver._get_asymptotic_discount_factors(env, tau)
    principal = product.initial_price * product.contract_multiplier
    k = product.payoff_config.participation_rate * product.contract_multiplier * tenor
    if product.is_reverse:
        return (principal + k * product.strike) * df - k * s_edge * df_div
    return (principal - k * product.strike) * df + k * s_edge * df_div


ROW_CASES = {
    "snowball_standard": (SnowballPDESolver, lambda: annualized_snowball(), lambda p, e: p.get_contract_tenor(e)),
    "snowball_reverse": (SnowballPDESolver, lambda: annualized_snowball(reverse=True),
                         lambda p, e: p.get_contract_tenor(e)),
    # partial protection (the legacy max-loss row): a 6-month tenor halves the loss at the edge, the 80% floor
    # never binds
    "snowball_partial": (SnowballPDESolver, lambda: annualized_snowball(0.5, protection=0.8),
                         lambda p, e: p.get_contract_tenor(e)),
    "ko_reset": (KOResetSnowballPDESolver, grid.ko_reset, lambda p, e: p._resolve_post_contract_tenor(e)),
}


@pytest.mark.parametrize("case", list(ROW_CASES))
@pytest.mark.parametrize("tau_fraction", [0.25, 1.0])
def test_knock_in_edge_row_is_the_annualized_payoffs_forward(case, tau_fraction):
    cls, make, tenor_of = ROW_CASES[case]
    product, env = make(), grid.flat_env()
    solver = cls(PDEParams(grid=GridConfig(num_std=3.0, points=400)))
    s_vec = _solved_s_grid(solver, product, env)
    tau = tau_fraction * solver._total_tau
    lower, upper = solver._boundary_rows_v1(0, tau, product, env, s_vec)
    row, s_edge = (upper, s_vec[-1]) if product.is_reverse else (lower, s_vec[0])
    tenor = tenor_of(product, env)
    assert not math.isclose(tenor, 1.0)
    # the edge lies on the payoff's affine piece (no floor binds there)
    assert product.is_reverse == (s_edge > product.strike)
    expected = _forward_affine(product, env, solver, tau, s_edge, tenor)
    assert row.value == pytest.approx(expected, rel=1e-12, abs=1e-10), (case, row.branch, row.value, expected)
    assert row.terms[0].value == row.value


def test_non_annualized_row_is_the_legacy_closed_form_bitwise():
    """Outside the annualized-KI class the row is the legacy formula, to the bit (AC30: those prices do not move)."""
    product, env = grid.snowball("discrete"), grid.flat_env()
    solver = SnowballPDESolver(PDEParams(grid=GridConfig(num_std=3.0, points=400)))
    s_vec = _solved_s_grid(solver, product, env)
    tau = 0.5 * solver._total_tau
    lower, _ = solver._boundary_rows_v1(0, tau, product, env, s_vec)
    df, df_div = solver._get_asymptotic_discount_factors(env, tau)
    slope = product.payoff_config.participation_rate * product.contract_multiplier
    legacy = 100.0 * df + slope * (s_vec[0] * df_div - product.strike * df)
    assert lower.branch == "v1_lower_none_slope"
    assert lower.value == legacy


# --- width invariance (AC30) --------------------------------------------------------------------------------------

def _price_1d(cls, product, env, num_std, points):
    return cls(params=PDEParams(grid=GridConfig(num_std=num_std, points=points))).price(product, env)


WIDTH_1D = {
    "flat_snowball": (SnowballPDESolver, annualized_snowball, grid.flat_env),
    "flat_ko_reset": (KOResetSnowballPDESolver, grid.ko_reset, grid.flat_env),
    "lv_snowball": (LocalVolSnowballPDESolver, annualized_snowball, grid.grid_env),
    "lv_ko_reset": (LocalVolKOResetSnowballPDESolver, grid.ko_reset, grid.grid_env),
    "lv_ko_reset_knocked_in": (LocalVolKOResetSnowballPDESolver, lambda: grid.ko_reset(knocked_in=True),
                               grid.grid_env),
    "lv_ko_reset_cki": (LocalVolKOResetSnowballPDESolver, lambda: grid.ko_reset(ki="continuous"), grid.grid_env),
}


@pytest.mark.parametrize("case", list(WIDTH_1D))
def test_pde_1d_price_does_not_depend_on_the_domain_width(case):
    cls, make, env = WIDTH_1D[case]
    narrow = _price_1d(cls, make(), env(), 3.0, 400)
    wide = _price_1d(cls, make(), env(), 16.0, 1600)
    assert abs(narrow - wide) <= WIDTH_TOL_1D, (case, narrow, wide)


def _adi_at_width(cls, num_std, **kw):
    """A Heston / SLV ADI solver whose S axis is bound at ``num_std`` instead of the default 8 (test-only binder)."""
    from quantark.asset.equity.engine.pde.grid import GridBinder

    class AtWidth(cls):
        def _layer_binder(self):
            if self._layer_binder_2d is None:
                self._layer_binder_2d = GridBinder("standard", GridConfig(points=int(self.n_x), num_std=num_std),
                                                   cache_enabled=self._is_cache_enabled())
            return self._layer_binder_2d

    extra = {"leverage_surface": grid.skew_leverage()} if "SLV" in cls.__name__ else {}
    return AtWidth(grid.HESTON, **extra, **kw)


ADI_KW = dict(n_x=160, n_v=32, n_t=200)
WIDTH_ADI = {
    "heston_snowball": (HestonSnowballPDESolver, annualized_snowball),
    "heston_ko_reset": (HestonKOResetSnowballPDESolver, grid.ko_reset),
    "slv_snowball": (HestonSLVSnowballPDESolver, annualized_snowball),
    "slv_ko_reset": (HestonSLVKOResetSnowballPDESolver, grid.ko_reset),
}


@pytest.mark.parametrize("case", list(WIDTH_ADI))
def test_adi_price_does_not_depend_on_the_domain_width(case):
    cls, make = WIDTH_ADI[case]
    env = grid.flat_env()
    narrow = _adi_at_width(cls, 3.0, **ADI_KW).price(make(), env)
    wide = _adi_at_width(cls, 16.0, **{**ADI_KW, "n_x": 4 * ADI_KW["n_x"]}).price(make(), env)
    assert abs(narrow - wide) <= WIDTH_TOL_ADI, (case, narrow, wide)


# --- independent MC (AC30) ----------------------------------------------------------------------------------------

def _mc_mean_se(price_one, seeds):
    values = np.array([price_one(seed) for seed in seeds], dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(values.size))


MC_SEEDS = tuple(range(101, 117))


@pytest.mark.slow
@pytest.mark.parametrize("case", ["lv_snowball", "lv_ko_reset"])
def test_local_vol_pde_matches_daily_step_mc(case):
    """LV PDE (default grid and width 3) against LV MC with 1-day SDE steps, 16 independent PSEUDO seeds."""
    from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import LocalVolSnowballMCEngine

    cls, make, env_of = WIDTH_1D[case]
    env = env_of()
    mean, se = _mc_mean_se(
        lambda seed: LocalVolSnowballMCEngine(MCParams(num_paths=20000, seed=seed), max_sde_step_days=1.0).price(
            make(), env), MC_SEEDS)
    for label, pde in (("default", cls(params=PDEParams()).price(make(), env)),
                       ("width3", _price_1d(cls, make(), env, 3.0, 400))):
        assert abs(pde - mean) <= 4.0 * se, (case, label, pde, mean, se)


@pytest.mark.slow
@pytest.mark.parametrize("case", list(WIDTH_ADI))
def test_heston_and_slv_adi_match_mc(case):
    """Heston / SLV ADI at width 3 and at its default width against Heston / SLV QE MC (32 substeps per contractual
    interval, 16 independent PSEUDO seeds). The SLV MC builds its local vol from a grid surface: both SLV engines price
    on a flat 22% grid surface (the ADI's price does not read it)."""
    from quantark.asset.equity.engine.mc import HestonSLVSnowballMCEngine, HestonSnowballMCEngine

    cls, make = WIDTH_ADI[case]
    slv = "SLV" in cls.__name__
    env = grid.grid_env(skew=False) if slv else grid.flat_env()
    extra = {"leverage_surface": grid.skew_leverage()} if slv else {}
    mc_cls = HestonSLVSnowballMCEngine if slv else HestonSnowballMCEngine
    mean, se = _mc_mean_se(
        lambda seed: mc_cls(grid.HESTON, **extra, params=MCParams(num_paths=20000, seed=seed),
                            substeps_per_interval=32).price(make(), env), MC_SEEDS)
    for label, solver in (("default", cls(grid.HESTON, **extra, **ADI_KW)),
                          ("width3", _adi_at_width(cls, 3.0, **ADI_KW))):
        pde = solver.price(make(), env)
        assert abs(pde - mean) <= 4.0 * se, (case, label, pde, mean, se)


def test_ko_reset_row_follows_the_payoffs_post_tenor_not_the_pre_tenor():
    """KO-reset: the knocked-in leg is annualized over the POST-KI tenor (2y here), not the first schedule's 1y."""
    product, env = grid.ko_reset(), grid.flat_env()
    assert product._resolve_post_contract_tenor(env) == pytest.approx(2.0)
    assert product.get_contract_tenor(env) == pytest.approx(1.0)
    solver = KOResetSnowballPDESolver(PDEParams(grid=GridConfig(num_std=3.0, points=400)))
    s_vec = _solved_s_grid(solver, product, env)
    lower, _ = solver._boundary_rows_v1(0, solver._total_tau, product, env, s_vec)
    expected = _forward_affine(product, env, solver, solver._total_tau, s_vec[0], 2.0)
    assert lower.value == pytest.approx(expected, rel=1e-12)
