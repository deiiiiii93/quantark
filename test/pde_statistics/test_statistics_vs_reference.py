"""Statistics against the exact reference (spec 3B §7.2): bounds at twice the largest measured S1 error."""
import json
from pathlib import Path

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from pde_statistics.cases import CASES

BANK = json.loads((Path(__file__).parent / "reference_values.json").read_text())["cases"]
CLASS = {"sb_cont": "cont", "sb_cont_near_ki": "cont", "sb_cont_near_ko": "cont", "sb_cont_3m": "cont",
         "sb_cont_vol12": "cont", "sb_discrete": "disc", "sb_discrete_near_ki": "disc", "sb_european": "disc",
         "sb_daily": "daily", "ph_cont": "cont", "ph_cont_near_coupon": "cont"}
BOUNDS = {"cont": dict(ko=2e-4, ki_ever=2e-3, ki_survive=1e-3),
          "disc": dict(ko=3e-5, ki_ever=1e-4, ki_survive=1e-4),
          "daily": dict(ko=2e-4, ki_ever=7e-4, ki_survive=1e-4)}
COUPON, CASH, PV = 1e-4, 2e-2, 2e-2


@pytest.mark.parametrize("name", sorted(CLASS))
def test_statistics_within_measured_bounds(name):
    product, env = CASES[name][0](), CASES[name][1]()
    solver = (PhoenixPDESolver if name.startswith("ph") else SnowballPDESolver)(PDEParams())
    got, ref, b = solver.calculate_event_stats(product, env), BANK[name]["values"], BOUNDS[CLASS[name]]
    # unified event statistics T12: the knock-in-ever column is gone (a knock-in after a knock-out is no knock-in);
    # P(knock in before any knock-out) comes from the forward adjoint pass (T13), not from this reference
    assert np.max(np.abs(got.observations.probability - ref["ko_probability"])) <= b["ko"]
    assert abs(got.ki_settlement.probability - ref["ki_settlement_probability"]) <= b["ki_survive"]
    assert np.max(np.abs(got.observations.discounted_cash - ref["expected_discounted_ko_cashflow"])) <= CASH
    assert abs(got.pv - ref["pv"]) <= PV
    if "coupon_probability" in ref:
        assert np.max(np.abs(got.coupons.probability - ref["coupon_probability"])) <= COUPON
