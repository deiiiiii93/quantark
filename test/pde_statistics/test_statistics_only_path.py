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
}


def test_statistics_only_snowball_builds_its_event_schedule():
    solver, product = PRODUCTS["snowball"]()
    p = prepare(solver, product, env(), STATS_ONLY)
    assert p.events is not None and p.groups == (p.statistics.group,)


def test_statistics_only_expiry_phoenix_builds_its_termination_tables():
    solver, product = PRODUCTS["phoenix_expiry"]()
    solver._term_df_maturity = None
    prepare(solver, product, env(), STATS_ONLY)
    assert solver._term_df_maturity is not None and solver._term_next_ko_time is not None


@pytest.mark.parametrize("mode", ["legacy", "value"])
@pytest.mark.parametrize("name", sorted(PRODUCTS))
def test_supplied_npv_equal_to_the_price_changes_no_field(value_mode, name, mode):
    if mode == "value":
        # the EXPIRY Phoenix joins in Task 14, KO-reset in Task 15
        value_mode("snowball", *(() if name == "phoenix_expiry" else ("phoenix",)))
    solver, product = PRODUCTS[name]()
    own = solver.calculate_event_stats(product, env())
    passed = solver.calculate_event_stats(product, env(), npv=float(own.pv))
    for f in own.__dataclass_fields__:
        a, b = getattr(own, f), getattr(passed, f)
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b), f
        else:
            assert a == b, f
