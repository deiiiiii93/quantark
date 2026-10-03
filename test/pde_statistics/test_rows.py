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


# --- the value-mode statistics rows (Task 12; spec 3B §5.1 "Rows", decision 14) ----------------

def _stats_rows(solver, product, e, price=True):
    request = (OutputRequest(Operation.EVENT_STATS, event_stats=True) if price
               else OutputRequest(Operation.EVENT_STATS, price=False, event_stats=True, external_pv=1.0))
    p = prepare(solver, product, e, request)
    stats, n = p.statistics, len(p.layout.spatial.x)
    out = [stats.rows(j, np.zeros((n, 2 * stats.n_cols))) for j in range(len(p.layout.time.t) - 1)]
    return p, stats, out


def _lagged(spec, lag):
    from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
    product = make_snowball(spec)
    product.settlement_convention = SettlementConvention(lag=lag, lag_unit=SettlementLagUnit.YEAR_FRACTION)
    return product


@pytest.mark.parametrize("lag", [0.0, 0.2])
def test_knock_in_rows_are_determination_units(lag):
    solver, e = SnowballPDESolver(SHORT), env()
    spec = SNOWBALLS["standard_disc"]
    product = _lagged(spec, lag) if lag else make_snowball(spec)
    p, stats, rows = _stats_rows(solver, product, e)
    L, w = stats.layout, stats.n_cols
    t, s = p.layout.time.t, p.layout.spatial.s
    for j, (lower, upper) in enumerate(rows):
        current = solver._total_tau - (p.tau - t[j])
        df_det = solver._df_between_times(e, current, p.tau)
        lo0, lo1, up1 = lower[:w], lower[w:], upper[w:]
        assert lo1[L.kis] == lo1[L.kie] == df_det, j          # knocked in, at the knock-in edge
        assert lo0[L.kis] == lo0[L.kie] == df_det, j          # a certain discrete knock-in
        assert up1[L.kis] == 0.0 and up1[L.kie] == df_det, j  # the knock-out survives the knock-in
        # the maturity leg's rows are the value's own rows (the lag's discount included)
        v_lo0, _ = solver._boundary_rows_v0(j, p.tau - t[j], product, e, s)
        v_lo1, _ = solver._boundary_rows_v1(j, p.tau - t[j], product, e, s)
        assert lo0[L.maturity[0]] == v_lo0.value and lo1[L.maturity[0]] == v_lo1.value, j


def test_lag_discounts_the_maturity_leg_only():
    e, spec = env(), SNOWBALLS["standard_disc"]
    _, stats, plain = _stats_rows(SnowballPDESolver(SHORT), make_snowball(spec), e)
    _, _, lagged = _stats_rows(SnowballPDESolver(SHORT), _lagged(spec, 0.2), e)
    assert len(plain) == len(lagged)
    L, w = stats.layout, stats.n_cols
    for (pl, pu), (ll, lu) in zip(plain, lagged):
        for a, b in ((pl, ll), (pu, lu)):
            for off in (0, w):
                assert a[off + L.kis] == b[off + L.kis] and a[off + L.kie] == b[off + L.kie]
        assert ll[L.maturity[0]] < pl[L.maturity[0]]        # the lag's extra discount on the leg


def test_rows_do_not_depend_on_the_value_march():
    e, product = env(), make_snowball(SNOWBALLS["standard_cont"])
    _, _, with_value = _stats_rows(SnowballPDESolver(SHORT), product, e, price=True)
    _, _, stats_only = _stats_rows(SnowballPDESolver(SHORT), product, e, price=False)
    assert len(with_value) == len(stats_only)
    for (a_lo, a_up), (b_lo, b_up) in zip(with_value, stats_only):
        assert np.array_equal(a_lo, b_lo) and np.array_equal(a_up, b_up)


def test_phoenix_rows():
    """Task 13: the alive lower row is the value's 0.0 (Phoenix never exposes the knocked-in grid), the
    coupon cash rows are 0 (the value's rows carry no coupon), and the knock-out edge triggers the coupon
    of the record it pays, in payment units (F8)."""
    solver, e = PhoenixPDESolver(SHORT), env()
    product = make_phoenix(dict(PHOENIX, memory_coupon=True))      # continuous KI
    p, stats, rows = _stats_rows(solver, product, e)
    L, w = stats.layout, stats.n_cols
    t = p.layout.time.t
    seen_trigger = False
    for j, (lower, upper) in enumerate(rows):
        lo0, up0 = lower[:w], upper[:w]
        cash = L.cash.ravel()
        assert not lo0[L.ko].any() and lo0[L.maturity[0]] == 0.0 and not lo0[L.trigger_cols].any(), j
        assert not lower[cash].any() and not upper[cash].any() and not upper[w + cash].any(), j
        _, v_up0 = solver._boundary_rows_v0(j, p.tau - t[j], product, e, p.layout.spatial.s)
        (term,) = v_up0.terms
        if term.column == "KO" and term.record_time is not None:
            k = stats.column_of_record(term.record_time)
            current = solver._total_tau - (p.tau - t[j])
            pay = solver._coupon_payment_time(product, e, stats._obs_of_k[k])
            assert up0[L.trigger_cols[k]] == solver._cashflow_value_at_time(e, 1.0, current, pay), j
            assert np.count_nonzero(up0[L.trigger_cols]) == 1, j
            seen_trigger = True
    assert seen_trigger
