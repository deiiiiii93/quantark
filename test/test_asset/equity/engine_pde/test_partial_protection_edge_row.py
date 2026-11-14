"""The knocked-in surface's knock-in-edge row honours partial protection without an airbag (unified event statistics
D28, AC33).

Under partial protection the knocked-in payoff (``get_maturity_payoff_v1``) is the unprotected loss floored at
``protection_rate x S0 x multiplier``. The closed-form rows of ``SnowballPDESolver._boundary_rows_v1`` write a
constant there: the standard max-loss row (``participation x K`` at S = 0, floored) and the reverse floor. A constant
is the payoff only where the floor binds. Where it does not bind on the knock-in edge cell (a deep floor, e.g. a 90%
maximum loss, or a reverse floor beyond the upper edge), the row wrote the wrong value and the price depended on how
far the domain reached (base: flat Snowball 0.9 width 3 vs 16 off by 0.107, local vol by 2.05).

D28: those contracts take T14b's payoff-read row (``_payoff_v1_edge_row``); the gate is "the knocked-in payoff on the
edge cell is not the closed-form constant". Contracts whose floor binds on the edge cell keep the closed form, bit for
bit.

Gates (AC33): the row is ``E_fwd`` of the payoff's affine piece at the edge, from the contract terms; a floor that
never binds prices exactly as no protection (whose row D28 does not touch); the price no longer depends on the domain
width (width 3 == 16 within the PDE's discretisation allowance) on PDE 1D (flat and local vol) and the Heston / SLV
ADI; at the default grid it matches an independent MC within 4 SE; floor-binding contracts are bitwise unchanged.
"""
from __future__ import annotations

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
from test_asset.equity.engine_pde.test_airbag_edge_row import (
    ADI_KW,
    WIDTH_TOL_1D,
    _heston_mc,
    _legacy_rows,
    _lv_mc,
    _mc_mean_se,
    _price_1d,
    _seeds,
    _solved,
)
from test_asset.equity.engine_pde.test_annualized_ki_edge_row import (
    WIDTH_TOL_ADI,
    _adi_at_width,
    _forward_affine,
)


def with_partial(product, rate: float, participation: float = 1.0, protection: ProtectionType = ProtectionType.PARTIAL):
    """``product`` partially protected at ``rate`` (no airbag, knock-in leg not annualized)."""
    product.payoff_config = PayoffConfig(rebate_rate=product.payoff_config.rebate_rate, include_principal=True,
                                         participation_rate=participation, protection_type=protection,
                                         protection_rate=rate if protection == ProtectionType.PARTIAL else 0.0)
    product.airbag_config = AirbagConfig()
    product.accrual_config = AccrualConfig(coupon_pay_type=product.accrual_config.coupon_pay_type)
    product.validate()
    return product


def snowball(rate=0.9, protection=ProtectionType.PARTIAL):
    """90% maximum loss: the floor binds below 10, far beyond every edge (the payoff there is the unprotected one)."""
    return with_partial(grid.snowball("discrete"), rate, protection=protection)


def reverse_snowball(rate=1.0, protection=ProtectionType.PARTIAL):
    """Half participation, 100% maximum loss: the floor binds above 300, beyond the width-3 upper edge (~194)."""
    return with_partial(grid.snowball("discrete", reverse=True), rate, participation=0.5, protection=protection)


def phoenix(rate=0.9, protection=ProtectionType.PARTIAL):
    return with_partial(grid.phoenix(ki="discrete"), rate, protection=protection)


def ko_reset(rate=0.9, protection=ProtectionType.PARTIAL):
    return with_partial(grid.ko_reset(), rate, protection=protection)


PRODUCTS = {
    "snowball": (snowball, SnowballPDESolver, LocalVolSnowballPDESolver, HestonSnowballPDESolver,
                 HestonSLVSnowballPDESolver),
    "reverse_snowball": (reverse_snowball, SnowballPDESolver, LocalVolSnowballPDESolver, HestonSnowballPDESolver,
                         HestonSLVSnowballPDESolver),
    "phoenix": (phoenix, PhoenixPDESolver, LocalVolPhoenixPDESolver, HestonPhoenixPDESolver,
                HestonSLVPhoenixPDESolver),
    "ko_reset": (ko_reset, KOResetSnowballPDESolver, LocalVolKOResetSnowballPDESolver,
                 HestonKOResetSnowballPDESolver, HestonSLVKOResetSnowballPDESolver),
}


def _edge(product, s_vec, lower, upper):
    if product.is_reverse:
        return upper, float(s_vec[-1]), float(s_vec[-2])
    return lower, float(s_vec[0]), float(s_vec[1])


# --- the row (D28) ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("product", list(PRODUCTS))
@pytest.mark.parametrize("tau_fraction", [0.25, 1.0])
def test_knock_in_edge_row_is_the_unfloored_payoffs_forward(product, tau_fraction):
    make, cls = PRODUCTS[product][0], PRODUCTS[product][1]
    contract, env = make(), grid.flat_env()
    solver, s_vec = _solved(cls, contract, env)
    tau = tau_fraction * solver._total_tau
    lower, upper = solver._boundary_rows_v1(0, tau, contract, env, s_vec)
    row, s_edge, s_next = _edge(contract, s_vec, lower, upper)
    principal = contract.initial_price * contract.contract_multiplier
    floor = contract.payoff_config.protection_rate * principal
    for s in (s_edge, s_next):                     # the floor does not bind on the edge cell
        assert contract.get_maturity_payoff_v1(s, env) > principal - floor + 1.0
    expected = _forward_affine(contract, env, solver, tau, s_edge, 1.0)
    assert row.value == pytest.approx(expected, rel=1e-12, abs=1e-10), (product, row.branch, row.value, expected)
    assert row.terms[0].value == row.value


#: the floor binds on the whole width-3 edge cell: 30% maximum loss (binds below 70; reverse at half participation
#: above 160); the closed-form constant is the payoff there
FLOOR_BINDS = {name: (lambda make=make: make(rate=0.3)) for name, (make, *_rest) in PRODUCTS.items()}


@pytest.mark.parametrize("product", list(PRODUCTS))
def test_floor_binding_on_the_edge_cell_keeps_the_closed_form_row_bitwise(product):
    cls = PRODUCTS[product][1]
    contract, env = FLOOR_BINDS[product](), grid.flat_env()
    solver, s_vec = _solved(cls, contract, env)
    tau = 0.5 * solver._total_tau
    lower, upper = solver._boundary_rows_v1(0, tau, contract, env, s_vec)
    row = upper if contract.is_reverse else lower
    assert row.value == _legacy_rows(solver, tau, contract, env, s_vec), (product, row.branch)


def _d28_shut():
    """The D28 gate forced shut (the D26 / D27 gates untouched): the closed-form row for every non-annualized,
    airbag-free contract."""
    real = SnowballPDESolver.__dict__["_partial_floor_unbound_on_knock_in_edge"]

    class Shut:
        def __enter__(self):
            SnowballPDESolver._partial_floor_unbound_on_knock_in_edge = staticmethod(lambda product, env, s_vec: False)

        def __exit__(self, *exc):
            SnowballPDESolver._partial_floor_unbound_on_knock_in_edge = real
            return False

    return Shut()


def _width3_solver(product, engine):
    _, flat_cls, lv_cls, heston, slv = PRODUCTS[product]
    if engine in ("flat", "lv"):
        cls = flat_cls if engine == "flat" else lv_cls
        return lambda: cls(params=PDEParams(grid=GridConfig(num_std=3.0, points=375)))
    return lambda: _adi_at_width(heston if engine == "heston" else slv, 3.0, **ADI_KW)


def _env(engine):
    return grid.grid_env() if engine == "lv" else grid.flat_env()


@pytest.mark.parametrize("engine", ["flat", "lv", "heston", "slv"])
@pytest.mark.parametrize("product", list(PRODUCTS))
def test_floor_binding_contract_price_is_bitwise_unchanged(product, engine):
    """AC33: where the floor binds on the edge cell the D28 gate stays shut -- the price is the closed-form row's,
    bit for bit, at width 3 (where the edge is reached)."""
    contract = FLOOR_BINDS[product]
    live = _width3_solver(product, engine)().price(contract(), _env(engine))
    with _d28_shut():
        closed = _width3_solver(product, engine)().price(contract(), _env(engine))
    assert float(live).hex() == float(closed).hex(), (product, engine, live, closed)


@pytest.mark.parametrize("engine", ["flat", "lv", "heston", "slv"])
@pytest.mark.parametrize("product", ["snowball", "phoenix", "ko_reset"])
@pytest.mark.parametrize("width", ["width3", "default"])
def test_floor_that_never_binds_prices_as_no_protection(product, engine, width):
    """A 100% maximum loss at full participation never binds (the loss reaches 100 only at S = 0): the partially
    protected payoff IS the unprotected one, so the two prices agree to round-off -- at width 3 and at the default
    grid. The unprotected row is base's closed form (``v1_lower_none_slope``), untouched by D28, so this pins the
    partial row against it (base wrote the max-loss constant there: off by up to 0.9 on local vol, default grid)."""
    make = PRODUCTS[product][0]
    if width == "width3":
        solver = _width3_solver(product, engine)
    else:
        _, flat_cls, lv_cls, heston, slv = PRODUCTS[product]
        solver = {"flat": lambda: flat_cls(params=PDEParams()), "lv": lambda: lv_cls(params=PDEParams()),
                  "heston": lambda: heston(grid.HESTON, **ADI_KW),
                  "slv": lambda: slv(grid.HESTON, leverage_surface=grid.skew_leverage(), **ADI_KW)}[engine]
    unprotected = solver().price(make(rate=0.0, protection=ProtectionType.NONE), _env(engine))
    partial = solver().price(make(rate=1.0), _env(engine))
    assert partial == pytest.approx(unprotected, rel=1e-11), (product, engine, width, partial, unprotected)


# --- width invariance (AC33) --------------------------------------------------------------------------------------
#
# PDE 1D: T14c's gate (flat 1e-2, local vol 5e-2; ``test_airbag_edge_row.WIDTH_TOL_1D``). Local vol needs T14c's
# allowance for the Phoenix only: its width-3 residual (1.42e-2) is the unprotected Phoenix's to the bit (the coupon
# barrier at 80 is in reach of the ~53 edge, which no Dirichlet row sees), not D28's. Measured with the fix (T25c
# report, tmp/T25c-ac33-width-*.txt): flat <= 3.3e-4, local vol <= 6.2e-3 apart from that Phoenix (base's row: 6.7e-2
# to 0.12 flat, 1.7 to 2.1 local vol). ADI: T14b's 5e-3 (``test_annualized_ki_edge_row.WIDTH_TOL_ADI``); measured
# <= 4.1e-3 (the SLV Phoenix, again its unprotected twin's to the bit; base 2.2e-2 to 1.29).

@pytest.mark.parametrize("market", ["flat", "lv"])
@pytest.mark.parametrize("product", list(PRODUCTS))
def test_pde_1d_partial_price_does_not_depend_on_the_domain_width(product, market):
    make, flat_cls, lv_cls, _, _ = PRODUCTS[product]
    cls = flat_cls if market == "flat" else lv_cls
    narrow = _price_1d(cls, make(), _env(market), 3.0, 375)
    wide = _price_1d(cls, make(), _env(market), 16.0, 2000)
    assert abs(narrow - wide) <= WIDTH_TOL_1D[market], (product, market, narrow, wide)


@pytest.mark.parametrize("model", ["heston", "slv"])
@pytest.mark.parametrize("product", list(PRODUCTS))
def test_adi_partial_price_does_not_depend_on_the_domain_width(product, model):
    make = PRODUCTS[product][0]
    cls = _adi_cls_d28(product, model)
    narrow = _adi_at_width(cls, 3.0, **ADI_KW).price(make(), grid.flat_env())
    wide = _adi_at_width(cls, 16.0, **{**ADI_KW, "n_x": 853}).price(make(), grid.flat_env())
    assert abs(narrow - wide) <= WIDTH_TOL_ADI, (product, model, narrow, wide)


def _adi_cls_d28(product, model):
    _, _, _, heston, slv = PRODUCTS[product]
    return heston if model == "heston" else slv


# --- independent MC at the default grid (AC33) --------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("product", list(PRODUCTS))
def test_local_vol_pde_default_grid_matches_daily_step_mc(product):
    """LV PDE at its default grid against LV MC with 1-day SDE steps, 16 independent PSEUDO seeds."""
    make, _, cls, _, _ = PRODUCTS[product]
    env = grid.grid_env()
    mc_cls = _lv_mc(product)
    mean, se = _mc_mean_se(
        lambda seed: mc_cls(MCParams(num_paths=20000, seed=seed), max_sde_step_days=1.0).price(make(), env),
        _seeds("d28", "lv", product))
    pde = cls(params=PDEParams()).price(make(), env)
    assert abs(pde - mean) <= 4.0 * se, (product, pde, mean, se)


@pytest.mark.slow
@pytest.mark.parametrize("model", ["heston", "slv"])
@pytest.mark.parametrize("product", list(PRODUCTS))
def test_heston_and_slv_adi_default_grid_match_mc(product, model):
    """Heston / SLV ADI at the default width against Heston / SLV MC (32 substeps per contractual interval, 16
    independent PSEUDO seeds); both SLV engines on a flat 20% grid surface (the ADI does not read it)."""
    make = PRODUCTS[product][0]
    slv = model == "slv"
    cls = _adi_cls_d28(product, model)
    env = grid.grid_env(skew=False) if slv else grid.flat_env()
    extra = {"leverage_surface": grid.skew_leverage()} if slv else {}
    mc_cls = _heston_mc(product, slv)
    mean, se = _mc_mean_se(
        lambda seed: mc_cls(grid.HESTON, **extra, params=MCParams(num_paths=20000, seed=seed),
                            substeps_per_interval=32).price(make(), env),
        _seeds("d28", model, product))
    pde = cls(grid.HESTON, **extra, **ADI_KW).price(make(), env)
    assert abs(pde - mean) <= 4.0 * se, (product, model, pde, mean, se)
