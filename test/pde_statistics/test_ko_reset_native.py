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
from pde_statistics.v2_frozen import v2_event_stats

REF = json.loads((Path(__file__).parent / "ko_reset_reference.json").read_text())["cases"]
SPOTS = {"kr_cont": 100.0, "kr_cont_spot90": 90.0, "kr_cont_spot101": 101.0}


def _rows(stats, schedule):
    """A schedule's knock-out probabilities, in its own (time) order, from the tagged observation table."""
    tags = np.asarray(stats.observations.schedule)
    return stats.observations.probability[tags == schedule]


@pytest.mark.parametrize("name", sorted(SPOTS))
def test_against_the_converged_quad(name):
    product, e = make_ko_reset(KO_RESET), env(spot=SPOTS[name])
    s, ref = KOResetSnowballPDESolver(PDEParams()).calculate_event_stats(product, e), REF[name]
    assert np.max(np.abs(_rows(s, "pre") - ref["pre_ko_probability"])) <= 5e-5
    assert np.max(np.abs(_rows(s, "post") - ref["post_ko_probability"])) <= 5e-5
    # T12: no knock-in-ever (a knock-in after a knock-out is no knock-in); P(KI before KO) is the forward pass's
    assert abs(s.ki_settlement.probability - ref["ki_settlement_probability"]) <= 5e-4
    assert abs(s.reconciliation_error) <= 1e-10


# Bounds at twice the largest measured error (standard accuracy, 2026-10-03), the plan's convention:
# V2's price against the 32,001-point price: 8.9e-5, 2.32e-4, 7.9e-5 (spots 100, 90, 101);
# M_pre against V2's alive row: -1.51e-2, -1.18e-2, -7.0e-3; M_post against its knocked-in row: 2.2e-3,
# 1.1e-4, 8.7e-4. The pre-maturity bucket carries the value surface's continuous-knock-in first-passage
# bias (it tracks ki_ever's +2e-4..+4e-4 spot by spot and accuracy by accuracy -- non-monotone under
# refinement, also in the price): one unit of probability is a 100 notional of cash.
V2_QUALIFIED, M_PRE, M_POST = 5e-4, 3e-2, 5e-3


@pytest.mark.parametrize("name", sorted(SPOTS))
def test_maturity_buckets_against_v2(name):
    product, e = make_ko_reset(KO_RESET), env(spot=SPOTS[name])
    s = KOResetSnowballPDESolver(PDEParams()).calculate_event_stats(product, e)
    v2 = v2_event_stats(f"ko_reset_{name}",
                        lambda: KOResetSnowballQuadEngineV2().calculate_event_stats(product, e))
    assert abs(v2.pv - REF[name]["pv"]) <= V2_QUALIFIED              # F17: V2 is qualified here
    v2_alive, v2_ki = v2.maturity.discounted_cash, v2.ki_settlement.discounted_cash
    assert abs(s.maturity.discounted_cash - v2_alive) <= M_PRE          # the first schedule's end, not knocked in
    assert abs(s.ki_settlement.discounted_cash - v2_ki) <= M_POST        # the second schedule's end, knocked in
    assert s.maturity.time < s.ki_settlement.time
    assert {s.maturity.payment_time, s.ki_settlement.payment_time} <= set(s.ledger.payment_times.tolist())


def test_field_layout_is_complete():
    s = KOResetSnowballPDESolver(PDEParams()).calculate_event_stats(make_ko_reset(KO_RESET), env())
    assert set(s.observations.schedule) == {"pre", "post"}
    assert s.observations.payment_times.size == s.observations.size
    total = s.observations.probability.sum() + s.maturity.probability + s.ki_settlement.probability
    assert abs(total - 1.0) <= 1e-6 and abs(s.probability_residual) <= 1e-6
