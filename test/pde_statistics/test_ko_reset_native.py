"""Native KO-reset statistics (spec 3B §5.4, §7.2)."""
import json
from pathlib import Path

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.quad.v2.engine import KOResetSnowballQuadEngineV2
from quantark.asset.equity.param import PDEParams
from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset
from pde_3a.products import KO_RESET, env

REF = json.loads((Path(__file__).parent / "ko_reset_reference.json").read_text())["cases"]
SPOTS = {"kr_cont": 100.0, "kr_cont_spot90": 90.0, "kr_cont_spot101": 101.0}


@pytest.mark.parametrize("name", sorted(SPOTS))
def test_against_the_converged_quad(value_mode, name):
    value_mode("ko_reset")
    product, e = make_ko_reset(KO_RESET), env(spot=SPOTS[name])
    s, ref = KOResetSnowballPDESolver(PDEParams()).calculate_event_stats(product, e), REF[name]
    assert np.max(np.abs(s.pre_ko_probability - ref["pre_ko_probability"])) <= 5e-5
    assert np.max(np.abs(s.post_ko_probability - ref["post_ko_probability"])) <= 5e-5
    assert abs(s.ki_ever_probability - ref["ki_ever_probability"]) <= 5e-4
    assert abs(s.ki_survive_knocked_in_probability - ref["ki_survive_knocked_in_probability"]) <= 5e-4
    assert abs(s.reconciliation_error) <= 1e-10


# Bounds at twice the largest measured error (standard accuracy, 2026-10-03), the plan's convention:
# V2's price against the 32,001-point price: 8.9e-5, 2.32e-4, 7.9e-5 (spots 100, 90, 101);
# M_pre against V2's alive row: -1.51e-2, -1.18e-2, -7.0e-3; M_post against its knocked-in row: 2.2e-3,
# 1.1e-4, 8.7e-4. The pre-maturity bucket carries the value surface's continuous-knock-in first-passage
# bias (it tracks ki_ever's +2e-4..+4e-4 spot by spot and accuracy by accuracy -- non-monotone under
# refinement, also in the price): one unit of probability is a 100 notional of cash.
V2_QUALIFIED, M_PRE, M_POST = 5e-4, 3e-2, 5e-3


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(SPOTS))
def test_maturity_buckets_against_v2(value_mode, name):
    value_mode("ko_reset")
    product, e = make_ko_reset(KO_RESET), env(spot=SPOTS[name])
    s = KOResetSnowballPDESolver(PDEParams()).calculate_event_stats(product, e)
    v2 = KOResetSnowballQuadEngineV2().calculate_event_stats(product, e)
    assert abs(v2.pv - REF[name]["pv"]) <= V2_QUALIFIED              # F17: V2 is qualified here
    v2_alive, v2_ki = v2.expected_discounted_cashflows[-2], v2.expected_discounted_cashflows[-1]
    assert abs(s.expected_discounted_pre_maturity_cashflow - v2_alive) <= M_PRE
    assert abs(s.expected_discounted_post_maturity_cashflow - v2_ki) <= M_POST
    assert s.pre_maturity_time < s.post_maturity_time
    assert {s.pre_maturity_payment_time, s.post_maturity_payment_time} <= set(s.payment_times.tolist())


def test_field_layout_is_complete(value_mode):
    value_mode("ko_reset")
    s = KOResetSnowballPDESolver(PDEParams()).calculate_event_stats(make_ko_reset(KO_RESET), env())
    for f in ("pre_ko_payment_times", "post_ko_payment_times", "expected_discounted_pre_maturity_cashflow",
              "expected_discounted_post_maturity_cashflow", "pre_maturity_probability", "post_maturity_probability",
              "pre_maturity_time", "post_maturity_time", "pre_maturity_payment_time", "post_maturity_payment_time"):
        assert getattr(s, f) is not None, f
    total = s.pre_ko_probability.sum() + s.post_ko_probability.sum() + s.pre_maturity_probability \
        + s.post_maturity_probability
    assert abs(total - 1.0) <= 1e-6
