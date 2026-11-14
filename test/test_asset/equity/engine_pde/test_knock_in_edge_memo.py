"""The knock-in edge row's per-solve memo (unified event statistics T25, ``SnowballPDESolver._knock_in_edge_terms``).

The D26 / D27 rule (an annualized knock-in leg, or an airbag that moves the knock-in edge) and the payoff chord it
reads are computed once per solve and reused by every row of that solve. The memo is keyed on the product and
environment objects and the edge cell, so a contract edited in place between two solves on the same solver must not
read the previous solve's terms: each solve's state preamble clears it. A reused solver prices the edited contract
exactly as a fresh solver does, on PDE 1D (flat, local vol) and the Heston ADI.
"""
from __future__ import annotations

import pytest

from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    HestonSnowballPDESolver,
    LocalVolSnowballPDESolver,
)
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, AirbagConfig, PayoffConfig
from quantark.util.enum.option_enums import ProtectionType
from test_asset.equity.engine_mc import mc_price_path_routes as grid

AIRBAG = AirbagConfig(airbag_barrier=80.0, airbag_participation_rate=0.5)
SOLVERS = {
    "pde_1d": (lambda: SnowballPDESolver(PDEParams(accuracy="fast")), grid.flat_env),
    "pde_1d_lv": (lambda: LocalVolSnowballPDESolver(params=PDEParams(accuracy="fast")), grid.grid_env),
    "adi_heston": (lambda: HestonSnowballPDESolver(grid.HESTON, n_x=64, n_v=12, n_t=32), grid.flat_env),
}


def _partial_snowball(airbag=AIRBAG):
    product = grid.snowball("discrete")
    product.payoff_config = PayoffConfig(rebate_rate=product.payoff_config.rebate_rate, include_principal=True,
                                         protection_type=ProtectionType.PARTIAL, protection_rate=0.6)
    product.airbag_config = airbag
    product.accrual_config = AccrualConfig(coupon_pay_type=product.accrual_config.coupon_pay_type)
    product.validate()
    return product


def _drop_airbag(product):
    product.airbag_config = AirbagConfig()


def _add_airbag(product):
    product.airbag_config = AIRBAG


def _halve_airbag_rate(product):
    product.airbag_config = AirbagConfig(airbag_barrier=80.0, airbag_participation_rate=0.25)


#: edit -> (the contract solved first, the in-place edit): the payoff-read row switched off, switched on, and kept
#: on with a different chord
EDITS = {"drop_airbag": (_partial_snowball, _drop_airbag),
         "add_airbag": (lambda: _partial_snowball(AirbagConfig()), _add_airbag),
         "halve_airbag_rate": (_partial_snowball, _halve_airbag_rate)}


@pytest.mark.parametrize("edit", list(EDITS))
@pytest.mark.parametrize("route", list(SOLVERS))
def test_reused_solver_reads_the_edited_contract(route, edit):
    make_solver, make_env = SOLVERS[route]
    env = make_env()
    start, apply_edit = EDITS[edit]
    product = start()
    reused = make_solver()
    before = reused.price(product, env)
    apply_edit(product)                    # the same product and environment objects
    product.validate()
    after = reused.price(product, env)
    fresh = make_solver().price(product, env)
    assert float(after).hex() == float(fresh).hex()
    assert float(after).hex() != float(before).hex()    # the edit moves the knock-in edge row
