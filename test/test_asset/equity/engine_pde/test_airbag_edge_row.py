"""The knocked-in surface's knock-in-edge row honours the airbag (unified event statistics D27, AC31).

Below its barrier (above it on a reverse product) an airbag swaps the knocked-in payoff's participation and strike
(``get_maturity_payoff_v1``). The closed-form rows of ``SnowballPDESolver._boundary_rows_v1`` that ignore it are the
protected ones: the max-loss row of a standard product (``participation x K`` at S = 0, floored) and the reverse
partial row (the floor). Under partial protection an airbag in force on the knock-in edge cell changes the payoff
there -- the floor may stop binding, or start to -- and the row was wrong (Snowball, Phoenix and KO-reset all reach it;
``_knock_in_side_row`` copies it into the not-knocked-in surface and the Heston / SLV ADI writes the same row).

D27: those contracts take T14b's payoff-read row (``_payoff_v1_edge_row``). The gate is as narrow as gives correct rows:

* protection NONE: the closed form already reads the airbag's participation and strike -- unchanged, bit for bit;
* protection FULL: the floor at principal binds everywhere, the airbag cannot change the payoff -- unchanged;
* partial protection where the airbag does not change the payoff on the edge cell (the floor binds with and without
  it, or the airbag is not in force there) -- the closed form is what the contract without an airbag gets: unchanged
  by D27 (D28 then reads the payoff where that floor does not bind on the cell: test_partial_protection_edge_row).

Gates (AC31): the row is ``E_fwd`` of the payoff's affine piece at the edge, from the contract terms; the price no
longer depends on the domain width (num_std 3 == 16 within the PDE's discretisation allowance) on PDE 1D (flat and
local vol) and the Heston / SLV ADI, for none / partial / full protection; it matches an independent MC within 4 SE.
"""
from __future__ import annotations

import math
import zlib

import numpy as np
import pytest

from quantark.asset.equity.engine.pde import GridConfig
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.pde.ko_reset_vol_pde_solvers import (
    HestonKOResetSnowballPDESolver,
    HestonSLVKOResetSnowballPDESolver,
    LocalVolKOResetSnowballPDESolver,
)
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import (
    HestonPhoenixPDESolver,
    HestonSLVPhoenixPDESolver,
    LocalVolPhoenixPDESolver,
)
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    HestonSLVSnowballPDESolver,
    HestonSnowballPDESolver,
    LocalVolSnowballPDESolver,
)
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, AirbagConfig, PayoffConfig
from quantark.util.enum.option_enums import ProtectionType
from test_asset.equity.engine_mc import mc_price_path_routes as grid
from test_asset.equity.engine_pde.test_annualized_ki_edge_row import _adi_at_width, _solved_s_grid

PROTECTION = {"none": ProtectionType.NONE, "partial": ProtectionType.PARTIAL, "full": ProtectionType.FULL}
#: the airbag of a standard product: below 80 the knocked-in loss runs at half rate
AIRBAG = AirbagConfig(airbag_barrier=80.0, airbag_participation_rate=0.5)
#: a reverse product's airbag: above 120 the knocked-in loss runs at half rate
AIRBAG_REVERSE = AirbagConfig(airbag_barrier=120.0, airbag_participation_rate=0.5)


def with_terms(product, protection: str = "partial", rate: float = 0.6, airbag: AirbagConfig | None = AIRBAG,
               annualized: bool = False):
    """``product`` with the protection, the airbag and (KO-reset) the knock-in leg's annualization set."""
    product.payoff_config = PayoffConfig(rebate_rate=product.payoff_config.rebate_rate, include_principal=True,
                                         protection_type=PROTECTION[protection],
                                         protection_rate=rate if protection == "partial" else 0.0)
    product.airbag_config = airbag if airbag is not None else AirbagConfig()
    if not annualized:
        product.accrual_config = AccrualConfig(coupon_pay_type=product.accrual_config.coupon_pay_type)
    product.validate()
    return product


def snowball(protection="partial", rate=0.6, airbag=AIRBAG):
    return with_terms(grid.snowball("discrete"), protection, rate, airbag)


def reverse_snowball(protection="partial", rate=0.6, airbag=AIRBAG_REVERSE):
    return with_terms(grid.snowball("discrete", reverse=True), protection, rate, airbag)


def phoenix(protection="partial", rate=0.6, airbag=AIRBAG):
    return with_terms(grid.phoenix(ki="discrete"), protection, rate, airbag)


def ko_reset(protection="partial", rate=0.6, airbag=AIRBAG, annualized=False):
    return with_terms(grid.ko_reset(), protection, rate, airbag, annualized=annualized)


#: an annualized KO-reset (post-KI tenor 2y) under a quarter-rate airbag: its largest loss (2 x 0.25 x 100) stays
#: inside the 60% floor, so the payoff beyond either edge is the airbag's affine piece (as for the other products)
AIRBAG_QUARTER = AirbagConfig(airbag_barrier=80.0, airbag_participation_rate=0.25)


def ko_reset_annualized(protection="partial", rate=0.6, airbag=AIRBAG_QUARTER):
    return ko_reset(protection, rate, airbag, annualized=True)


def _solved(cls, product, env):
    solver = cls(PDEParams(grid=GridConfig(num_std=3.0, points=400)))
    return solver, _solved_s_grid(solver, product, env)


def _edge(product, s_vec, lower, upper):
    if product.is_reverse:
        return upper, float(s_vec[-1]), float(s_vec[-2])
    return lower, float(s_vec[0]), float(s_vec[1])


# --- the row (D27) ------------------------------------------------------------------------------------------------

def _forward_airbag_piece(product, env, solver, tau, s_edge):
    """``E_fwd[P(S_T)]`` of the knocked-in payoff's piece through the edge, from the contract terms alone: under the
    airbag, principal + airbag participation x multiplier x (S_T - K_airbag) (standard) or x (K_airbag - S_T)
    (reverse), K_airbag the airbag strike or else the strike."""
    df, df_div = solver._get_asymptotic_discount_factors(env, tau)
    airbag = product.airbag_config
    principal = product.initial_price * product.contract_multiplier
    k = airbag.airbag_participation_rate * product.contract_multiplier
    strike = airbag.airbag_strike if airbag.airbag_strike is not None else product.strike
    if product.is_reverse:
        return (principal + k * strike) * df - k * s_edge * df_div
    return (principal - k * strike) * df + k * s_edge * df_div


ROW_CASES = {
    # partial 60%: at the width-3 edge (~52) the airbag's half-rate loss (~24) no longer reaches the floor (60)
    "snowball": (SnowballPDESolver, snowball),
    "snowball_airbag_strike": (SnowballPDESolver,
                               lambda: snowball(airbag=AirbagConfig(airbag_barrier=80.0, airbag_participation_rate=0.5,
                                                                    airbag_strike=90.0))),
    "snowball_reverse": (SnowballPDESolver, reverse_snowball),
    "phoenix": (PhoenixPDESolver, phoenix),
    "ko_reset": (KOResetSnowballPDESolver, ko_reset),
}


@pytest.mark.parametrize("case", list(ROW_CASES))
@pytest.mark.parametrize("tau_fraction", [0.25, 1.0])
def test_knock_in_edge_row_is_the_airbag_payoffs_forward(case, tau_fraction):
    cls, make = ROW_CASES[case]
    product, env = make(), grid.flat_env()
    solver, s_vec = _solved(cls, product, env)
    tau = tau_fraction * solver._total_tau
    lower, upper = solver._boundary_rows_v1(0, tau, product, env, s_vec)
    row, s_edge, s_next = _edge(product, s_vec, lower, upper)
    principal = product.initial_price * product.contract_multiplier
    floor = product.payoff_config.protection_rate * principal
    # the edge cell lies under the airbag, on its affine piece: the floor does not bind there
    for s in (s_edge, s_next):
        assert (s > 120.0) if product.is_reverse else (s < 80.0)
        assert product.get_maturity_payoff_v1(s, env) > principal - floor
    expected = _forward_airbag_piece(product, env, solver, tau, s_edge)
    assert row.value == pytest.approx(expected, rel=1e-12, abs=1e-10), (case, row.branch, row.value, expected)
    assert row.terms[0].value == row.value


def test_airbag_that_makes_the_floor_bind_reads_the_floor():
    """A 150% airbag rate drives the loss at the edge past the 60% floor although the plain rate would not: the row
    is the floored payoff's forward, (principal - floor) x DF -- not the closed form's plain max loss at S = 0."""
    product, env = snowball(airbag=AirbagConfig(airbag_barrier=80.0, airbag_participation_rate=1.5)), grid.flat_env()
    solver, s_vec = _solved(SnowballPDESolver, product, env)
    tau = 0.5 * solver._total_tau
    lower, _ = solver._boundary_rows_v1(0, tau, product, env, s_vec)
    df, _ = solver._get_asymptotic_discount_factors(env, tau)
    assert lower.value == pytest.approx((100.0 - 60.0) * df, rel=1e-12)


def _legacy_rows(solver, tau, product, env, s_vec):
    """Base's closed-form knock-in-edge row (the gate shut), written out here."""
    df_t = (solver._df_between_times(env, solver._total_tau - tau, solver._total_tau)
            * solver._structured_terminal_delay_df)
    principal = product.initial_price * product.contract_multiplier
    payoff = product.payoff_config
    if product.is_reverse:
        if payoff.protection_type == ProtectionType.PARTIAL:
            return (principal - payoff.protection_rate * principal) * df_t
        assert payoff.protection_type == ProtectionType.FULL
        return principal * df_t
    max_loss = payoff.participation_rate * (-product.strike / product.initial_price) * principal
    if payoff.protection_type == ProtectionType.FULL:
        max_loss = 0.0
    elif payoff.protection_type == ProtectionType.PARTIAL:
        max_loss = max(max_loss, -payoff.protection_rate * principal)
    return (principal + max_loss) * df_t


def _none_slope_row(solver, tau, product, env, s_vec):
    df, df_div = solver._get_asymptotic_discount_factors(env, tau)
    airbag = product.airbag_config
    k = airbag.airbag_participation_rate * product.contract_multiplier
    strike = airbag.airbag_strike if airbag.airbag_strike is not None else product.strike
    return 100.0 * df + k * (s_vec[0] * df_div - strike * df)


UNCHANGED = {
    # NONE: the closed form already reads the airbag's participation and strike
    "none": (SnowballPDESolver, lambda: snowball("none"), _none_slope_row),
    "phoenix_none": (PhoenixPDESolver, lambda: phoenix("none"), _none_slope_row),
    # FULL: the floor at principal binds everywhere
    "full": (SnowballPDESolver, lambda: snowball("full"), _legacy_rows),
    "reverse_full": (SnowballPDESolver, lambda: reverse_snowball("full"), _legacy_rows),
    # partial 30% with a 150% airbag: the floor binds on the edge cell with and without the airbag
    "partial_floor_binds_both_ways": (SnowballPDESolver,
                                      lambda: snowball(rate=0.3, airbag=AirbagConfig(airbag_barrier=80.0,
                                                                                     airbag_participation_rate=1.5)),
                                      _legacy_rows),
    "ko_reset_partial_floor_binds_both_ways": (KOResetSnowballPDESolver,
                                               lambda: ko_reset(rate=0.3, airbag=AirbagConfig(
                                                   airbag_barrier=80.0, airbag_participation_rate=1.5)),
                                               _legacy_rows),
    # the airbag barrier (40) lies below the width-3 edge (~52): not in force on the edge cell. A 30% floor (binding
    # below 70) keeps the cell on the floor: with the 60% floor the payoff there follows the spot and the row is D28's
    # (test_partial_protection_edge_row), not the closed form
    "partial_airbag_below_edge": (SnowballPDESolver,
                                  lambda: snowball(rate=0.3, airbag=AirbagConfig(airbag_barrier=40.0,
                                                                                 airbag_participation_rate=0.5)),
                                  _legacy_rows),
    # no airbag at all (the 30% floor binds on the cell; see above)
    "partial_no_airbag": (SnowballPDESolver, lambda: snowball(rate=0.3, airbag=None), _legacy_rows),
    "reverse_partial_no_airbag": (SnowballPDESolver, lambda: reverse_snowball(airbag=None), _legacy_rows),
}


@pytest.mark.parametrize("case", list(UNCHANGED))
def test_rows_the_airbag_does_not_make_wrong_stay_bitwise(case):
    """Outside D27's class the row is base's closed form, to the bit (AC31: those prices do not move)."""
    cls, make, legacy = UNCHANGED[case]
    product, env = make(), grid.flat_env()
    solver, s_vec = _solved(cls, product, env)
    tau = 0.5 * solver._total_tau
    lower, upper = solver._boundary_rows_v1(0, tau, product, env, s_vec)
    row = upper if product.is_reverse else lower
    assert row.value == legacy(solver, tau, product, env, s_vec), (case, row.branch)


# --- width invariance (AC31) --------------------------------------------------------------------------------------

#: PDE 1D width gate: |price(num_std 3, 375 points) - price(num_std 16, 2000 points)| (the same spacing). Measured
#: with the fix (T14c report): flat <= 3.0e-3; local vol <= 3.7e-2 (1.4e-2 without the annualized KO-reset). With
#: base's row the partial cases were off by 2.9e-2..9.2e-2 (flat) and 1.6..1.8 (local vol).
WIDTH_TOL_1D = {"flat": 1e-2, "lv": 5e-2}
#: Why local vol needs more: the airbag puts a jump in the knocked-in payoff at its barrier (80), and from the
#: width-3 local-vol edge (~53, where the surface quotes ~34% vol) that jump is within reach, which no asymptotic
#: Dirichlet row sees. The residual is the SAME with protection NONE, whose closed-form row D27 does not touch
#: (identical prices: ``test_protection_that_never_binds_leaves_the_price``); on the Snowball it does not shrink
#: under refinement (375 -> 1500 points moves it < 2e-3) and it is gone by width 6 (< 3e-4 from width 16): a
#: narrow-domain effect, not D27's.
#: ADI width gate: num_std 3 on n_x 160 vs num_std 16 on n_x 853 (the same spacing). Measured with the fix (T14c
#: report) <= 4.7e-3, and 9.2e-3 on the annualized KO-reset (identical with protection NONE: the same airbag jump);
#: with base's row the partial cases were off by 6.8e-3..1.07
WIDTH_TOL_ADI = 2e-2

PRODUCTS = {
    "snowball": (snowball, SnowballPDESolver, LocalVolSnowballPDESolver, HestonSnowballPDESolver,
                 HestonSLVSnowballPDESolver),
    "reverse_snowball": (reverse_snowball, SnowballPDESolver, LocalVolSnowballPDESolver, HestonSnowballPDESolver,
                         HestonSLVSnowballPDESolver),
    "phoenix": (phoenix, PhoenixPDESolver, LocalVolPhoenixPDESolver, HestonPhoenixPDESolver,
                HestonSLVPhoenixPDESolver),
    "ko_reset": (ko_reset, KOResetSnowballPDESolver, LocalVolKOResetSnowballPDESolver,
                 HestonKOResetSnowballPDESolver, HestonSLVKOResetSnowballPDESolver),
    "ko_reset_annualized": (ko_reset_annualized, KOResetSnowballPDESolver,
                            LocalVolKOResetSnowballPDESolver, HestonKOResetSnowballPDESolver,
                            HestonSLVKOResetSnowballPDESolver),
}
PROTECTIONS = ("none", "partial", "full")


def _price_1d(cls, product, env, num_std, points):
    return cls(params=PDEParams(grid=GridConfig(num_std=num_std, points=points))).price(product, env)


@pytest.mark.parametrize("protection", PROTECTIONS)
@pytest.mark.parametrize("market", ["flat", "lv"])
@pytest.mark.parametrize("product", list(PRODUCTS))
def test_pde_1d_airbag_price_does_not_depend_on_the_domain_width(product, market, protection):
    make, flat_cls, lv_cls, _, _ = PRODUCTS[product]
    cls, env = (flat_cls, grid.flat_env) if market == "flat" else (lv_cls, grid.grid_env)
    narrow = _price_1d(cls, make(protection), env(), 3.0, 375)
    wide = _price_1d(cls, make(protection), env(), 16.0, 2000)
    assert abs(narrow - wide) <= WIDTH_TOL_1D[market], (product, market, protection, narrow, wide)


ADI_KW = dict(n_x=160, n_v=32, n_t=200)
ADI_PRODUCTS = ("snowball", "reverse_snowball", "phoenix", "ko_reset", "ko_reset_annualized")


def _adi_cls(product, model):
    _, _, _, heston, slv = PRODUCTS[product]
    return heston if model == "heston" else slv


@pytest.mark.parametrize("protection", PROTECTIONS)
@pytest.mark.parametrize("model", ["heston", "slv"])
@pytest.mark.parametrize("product", ADI_PRODUCTS)
def test_adi_airbag_price_does_not_depend_on_the_domain_width(product, model, protection):
    make = PRODUCTS[product][0]
    cls, env = _adi_cls(product, model), grid.flat_env()
    narrow = _adi_at_width(cls, 3.0, **ADI_KW).price(make(protection), env)
    wide = _adi_at_width(cls, 16.0, **{**ADI_KW, "n_x": 853}).price(make(protection), env)
    assert abs(narrow - wide) <= WIDTH_TOL_ADI, (product, model, protection, narrow, wide)


def _width3_solver(product, engine):
    make, flat_cls, lv_cls, heston, slv = PRODUCTS[product]
    if engine in ("flat", "lv"):
        cls = flat_cls if engine == "flat" else lv_cls
        return lambda: cls(params=PDEParams(grid=GridConfig(num_std=3.0, points=375)))
    return lambda: _adi_at_width(heston if engine == "heston" else slv, 3.0, **ADI_KW)


@pytest.mark.parametrize("engine", ["flat", "lv", "heston", "slv"])
@pytest.mark.parametrize("product", ["snowball", "phoenix", "ko_reset", "ko_reset_annualized"])
def test_protection_that_never_binds_leaves_the_price(product, engine):
    """Under the half-rate (annualized: quarter-rate) airbag the largest knocked-in loss (50) stays inside the 60%
    floor, so the partially protected payoff IS the unprotected one: the two prices agree to round-off at width 3,
    where the edge is reached (base's max-loss row wrote the floor there and the partial price was off by up to 1.7).
    The unprotected row is base's closed form, untouched by D27, so this pins the partial row against it."""
    make = PRODUCTS[product][0]
    env = grid.grid_env() if engine == "lv" else grid.flat_env()
    unprotected = _width3_solver(product, engine)().price(make("none"), env)
    partial = _width3_solver(product, engine)().price(make("partial"), env)
    assert partial == pytest.approx(unprotected, rel=1e-11), (product, engine, partial, unprotected)


# --- independent MC (AC31) ----------------------------------------------------------------------------------------

def _seeds(*key) -> tuple:
    """16 PSEUDO seeds of their own per case (no two cases share MC noise)."""
    start = 1000 + 16 * (zlib.crc32("|".join(key).encode()) % 100_000)
    return tuple(range(start, start + 16))


def _mc_mean_se(price_one, seeds):
    values = np.array([price_one(seed) for seed in seeds], dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(values.size))


def _lv_mc(product_name):
    from quantark.asset.equity.engine.mc.phoenix_vol_mc_engines import LocalVolPhoenixMCEngine
    from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import LocalVolSnowballMCEngine

    return LocalVolPhoenixMCEngine if product_name == "phoenix" else LocalVolSnowballMCEngine


def _heston_mc(product_name, slv):
    from quantark.asset.equity.engine.mc import HestonSLVSnowballMCEngine, HestonSnowballMCEngine
    from quantark.asset.equity.engine.mc.phoenix_vol_mc_engines import HestonPhoenixMCEngine, HestonSLVPhoenixMCEngine

    if product_name == "phoenix":
        return HestonSLVPhoenixMCEngine if slv else HestonPhoenixMCEngine
    return HestonSLVSnowballMCEngine if slv else HestonSnowballMCEngine


@pytest.mark.slow
@pytest.mark.parametrize("protection", PROTECTIONS)
@pytest.mark.parametrize("product", list(PRODUCTS))
def test_local_vol_pde_matches_daily_step_mc(product, protection):
    """LV PDE (default grid and width 3) against LV MC with 1-day SDE steps, 16 independent PSEUDO seeds."""
    make, _, cls, _, _ = PRODUCTS[product]
    env = grid.grid_env()
    mc_cls = _lv_mc(product)
    mean, se = _mc_mean_se(
        lambda seed: mc_cls(MCParams(num_paths=20000, seed=seed), max_sde_step_days=1.0).price(make(protection), env),
        _seeds("lv", product, protection))
    for label, pde in (("default", cls(params=PDEParams()).price(make(protection), env)),
                       ("width3", _price_1d(cls, make(protection), env, 3.0, 400))):
        assert abs(pde - mean) <= 4.0 * se, (product, protection, label, pde, mean, se)


@pytest.mark.slow
@pytest.mark.parametrize("protection", PROTECTIONS)
@pytest.mark.parametrize("model", ["heston", "slv"])
@pytest.mark.parametrize("product", ADI_PRODUCTS)
def test_heston_and_slv_adi_match_mc(product, model, protection):
    """Heston / SLV ADI at width 3 and at its default width against Heston / SLV MC (32 substeps per contractual
    interval, 16 independent PSEUDO seeds); both SLV engines on a flat 20% grid surface (the ADI does not read it)."""
    make = PRODUCTS[product][0]
    slv = model == "slv"
    cls = _adi_cls(product, model)
    env = grid.grid_env(skew=False) if slv else grid.flat_env()
    extra = {"leverage_surface": grid.skew_leverage()} if slv else {}
    mc_cls = _heston_mc(product, slv)
    mean, se = _mc_mean_se(
        lambda seed: mc_cls(grid.HESTON, **extra, params=MCParams(num_paths=20000, seed=seed),
                            substeps_per_interval=32).price(make(protection), env),
        _seeds(model, product, protection))
    for label, solver in (("default", cls(grid.HESTON, **extra, **ADI_KW)),
                          ("width3", _adi_at_width(cls, 3.0, **ADI_KW))):
        pde = solver.price(make(protection), env)
        assert abs(pde - mean) <= 4.0 * se, (product, model, protection, label, pde, mean, se)
