"""The allocation of the value's boundary rows (spec 3B §5.1, §7.2 test_rows): every branch, bit for bit."""
import numpy as np
import pytest

from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest, prepare
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.engine.pde.grid.config import GridConfig
from quantark.modelvalidation.builders.equity_phoenix import make_phoenix
from quantark.modelvalidation.builders.equity_snowball import make_snowball
from pde_3a.products import PHOENIX, SNOWBALL, env

SHORT = PDEParams(grid=GridConfig(points=60, steps_per_day=1.0))

SNOWBALLS = {
    "standard_cont": dict(SNOWBALL),
    "standard_disc": dict(SNOWBALL, ki_monitoring="discrete"),
    "reverse": dict(SNOWBALL, is_reverse=True, ko_barrier=97.0, ki_barrier=115.0),
    "airbag": dict(SNOWBALL, airbag_barrier=90.0, airbag_participation_rate=0.5),
    "full_protection": dict(SNOWBALL, protection_type="full"),
    "partial_protection": dict(SNOWBALL, protection_type="partial", protection_rate=0.2),
    "reverse_partial": dict(SNOWBALL, is_reverse=True, ko_barrier=97.0, ki_barrier=115.0,
                            protection_type="partial", protection_rate=0.2),
    "dkak": dict(SNOWBALL, disable_ko_after_ki=True),
    "call_rebate": dict(SNOWBALL, call_rebate=True, call_strike=105.0, call_participation_rate=0.5),
    # the upper edge (~248) below every knock-out level: the deep-OTM rows
    "unreachable_ko": dict(SNOWBALL, ko_barrier=1000.0),
    "call_rebate_unreachable_ko": dict(SNOWBALL, ko_barrier=1000.0, call_rebate=True, call_strike=105.0,
                                       call_participation_rate=0.5),
    "reverse_full": dict(SNOWBALL, is_reverse=True, ko_barrier=97.0, ki_barrier=115.0, protection_type="full"),
}


def _rows_over_layout(solver, product, e):
    p = prepare(solver, product, e, OutputRequest(Operation.PRICE))
    t, s = p.layout.time.t, p.layout.spatial.s
    seen = set()
    for j in range(len(t) - 1):
        tau_rem = p.tau - t[j]
        for rows in (solver._boundary_rows_v0(j, tau_rem, product, e, s),
                     solver._boundary_rows_v1(j, tau_rem, product, e, s)):
            for edge in rows:
                seen.add(edge.branch)
                if edge.copy_from_ki:
                    assert not edge.terms
                    continue
                assert len(edge.terms) == 1, edge
                term = edge.terms[0]
                assert term.value * term.operand == edge.value, (edge.branch, j)
    return seen


@pytest.mark.parametrize("name", sorted(SNOWBALLS))
def test_snowball_rows_allocate_bit_for_bit(name):
    _rows_over_layout(SnowballPDESolver(SHORT), make_snowball(SNOWBALLS[name]), env())


def test_every_snowball_branch_is_covered():
    seen = set()
    for spec in SNOWBALLS.values():
        seen |= _rows_over_layout(SnowballPDESolver(SHORT), make_snowball(spec), env())
    assert seen >= {
        "v0_lower_copy_ki", "v0_lower_principal_rebate", "v0_upper_ko", "v0_upper_principal_rebate",
        "v0_upper_call_rebate",
        "v1_lower_reverse", "v1_lower_none_slope", "v1_lower_max_loss",
        "v1_upper_reverse_none", "v1_upper_reverse_partial", "v1_upper_reverse_full", "v1_upper_principal",
    }


def test_phoenix_termination_rows_allocate_bit_for_bit():
    product = make_phoenix(dict(PHOENIX, memory_coupon=True, coupon_pay_type="expiry"))
    solver = PhoenixPDESolver(SHORT)
    solver.price(product, env())          # the march builds the termination tables
    assert solver._term_df_maturity is not None
    seen_t = False
    for j in range(len(solver._term_df_maturity) - 1):
        lower, upper = solver._termination_rows(j)
        assert (lower.branch, upper.branch) == ("w_lower", "w_upper")
        for edge in (lower, upper):
            (term,) = edge.terms
            assert term.value * term.operand == edge.value
        assert [t.column for t in lower.terms] == ["U"]
        next_obs = solver._term_next_ko_obs[j]
        if np.isnan(next_obs):
            # past the last interior knock-out W's upper edge is the maturity discount: a U unit
            assert [t.column for t in upper.terms] == ["U"]
        else:
            assert [t.column for t in upper.terms] == ["T"] and upper.terms[0].record_time == next_obs
            seen_t = True
    assert seen_t
