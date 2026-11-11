"""The statistics-only path (spec 3B decision 14): no value march, and nothing in the statistics reads one."""
import numpy as np
import pytest

from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest, prepare
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.modelvalidation.builders.equity_phoenix import make_phoenix
from pde_3a.products import PHOENIX, env
from pde_statistics.cases import CASES

STATS_ONLY = OutputRequest(Operation.EVENT_STATS, price=False, event_stats=True, external_pv=1.0)

PRODUCTS = {
    "snowball": lambda: (SnowballPDESolver(PDEParams()), CASES["sb_cont"][0]()),
    "phoenix_plain": lambda: (PhoenixPDESolver(PDEParams()), CASES["ph_cont"][0]()),
    "phoenix_memory": lambda: (PhoenixPDESolver(PDEParams()), CASES["ph_memory"][0]()),
    "phoenix_expiry": lambda: (PhoenixPDESolver(PDEParams()),
                               make_phoenix(dict(PHOENIX, memory_coupon=True, coupon_pay_type="expiry"))),
    "ko_reset": lambda: (KOResetSnowballPDESolver(PDEParams()), CASES["kr_cont"][0]()),
}


def test_statistics_only_ko_reset_seeds_its_staged_surface():
    """The staged seed index is the value terminal's; a statistics-only march sets it itself (decision 14)."""
    solver, product = PRODUCTS["ko_reset"]()
    own = solver.calculate_event_stats(product, env())
    solver._v0_seed_idx = None
    passed = solver.calculate_event_stats(product, env(), npv=float(own.pv))
    assert passed.maturity.discounted_cash == own.maturity.discounted_cash
    assert passed.maturity.probability == own.maturity.probability > 0.0


def test_statistics_only_snowball_builds_its_event_schedule():
    solver, product = PRODUCTS["snowball"]()
    p = prepare(solver, product, env(), STATS_ONLY)
    assert p.events is not None and p.groups == (p.statistics.group,)


def test_statistics_only_expiry_phoenix_builds_its_termination_tables():
    solver, product = PRODUCTS["phoenix_expiry"]()
    solver._term_df_maturity = None
    prepare(solver, product, env(), STATS_ONLY)
    assert solver._term_df_maturity is not None and solver._term_next_ko_time is not None


@pytest.mark.parametrize("name", sorted(PRODUCTS))
def test_supplied_npv_equal_to_the_price_changes_no_field(name):
    solver, product = PRODUCTS[name]()
    own = solver.calculate_event_stats(product, env())
    passed = solver.calculate_event_stats(product, env(), npv=float(own.pv))
    _assert_same(own, passed, "stats")


def _assert_same(a, b, path):
    """Field-by-field bitwise equality through the nested tables (EventStatistics, T12)."""
    import dataclasses
    from collections.abc import Mapping

    if dataclasses.is_dataclass(a):
        assert type(a) is type(b), path
        for f in dataclasses.fields(a):
            _assert_same(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
    elif isinstance(a, Mapping):
        assert set(a) == set(b), path
        for k in a:
            _assert_same(a[k], b[k], f"{path}[{k}]")
    else:
        assert a == b, path
