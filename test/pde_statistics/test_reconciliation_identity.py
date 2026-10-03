"""The reconciliation identity (spec 3B §5.2, §7.2): |pv - Σ ledger| <= 1e-10 on every native 1D route."""
import itertools

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.cashleg.event_distribution import EventType
from quantark.execution import OutputKind, PricingRequest, PricingSession
from quantark.modelvalidation.builders.equity_snowball import make_snowball
from pde_3a.products import SNOWBALL, env
from pde_statistics.cases import CASES, trading_clock_env

BOUND = 1e-10


def _flagged(p):
    setattr(p, "_otc_lifecycle_knocked_in", True)
    return p


SNOWBALL_CASES = {
    "cont": (lambda: CASES["sb_cont"][0](), {}),
    "cont_86": (lambda: CASES["sb_cont"][0](), {"spot": 86.0}),
    "cont_102_5": (lambda: CASES["sb_cont"][0](), {"spot": 102.5}),
    "cont_3m": (lambda: CASES["sb_cont_3m"][0](), {}),
    "cont_knocked_in_by_spot": (lambda: CASES["sb_cont"][0](), {"spot": 82.45}),
    "cont_lifecycle_flag": (lambda: _flagged(CASES["sb_cont"][0]()), {}),
    "discrete": (lambda: CASES["sb_discrete"][0](), {}),
    "european": (lambda: CASES["sb_european"][0](), {}),
    "daily": (lambda: CASES["sb_daily"][0](), {}),
    "dkak_cont": (lambda: make_snowball(dict(SNOWBALL, disable_ko_after_ki=True)), {}),
    "dkak_disc": (lambda: make_snowball(dict(SNOWBALL, disable_ko_after_ki=True, ki_monitoring="discrete")), {}),
    "reverse": (lambda: make_snowball(dict(SNOWBALL, is_reverse=True, ko_barrier=97.0, ki_barrier=115.0)), {}),
    "airbag": (lambda: make_snowball(dict(SNOWBALL, airbag_barrier=90.0, airbag_participation_rate=0.5)), {}),
}
ENVS = {"flat": lambda **kw: env(**kw), "trading_clock": lambda **kw: trading_clock_env(**kw)[0]}
STREAMS = {"all": None, "ko_only": frozenset({EventType.KO})}


def _entry(entry, solver, product, e, streams):
    if entry == "direct":
        return solver.calculate_event_stats(product, e, streams=streams)
    if entry == "pwe":
        return solver._session_outputs(product, e, want_events=True, streams=streams).event_stats
    with PricingSession() as session:
        return session.execute(solver, PricingRequest(
            product=product, pricing_env=e, outputs=frozenset({OutputKind.PV, OutputKind.EVENT_STATS}))).value.event_stats


def _ids(*keys):
    return "-".join(map(str, keys))


def _check(stats):
    assert abs(stats.reconciliation_error) <= BOUND, stats.reconciliation_error
    ledger = float(np.sum(stats.expected_discounted_cashflows))
    assert abs(float(stats.pv) - ledger - stats.reconciliation_error) <= 1e-12


@pytest.mark.slow
@pytest.mark.parametrize("case,env_name,accuracy,projection,streams,entry", [
    p for p in itertools.product(sorted(SNOWBALL_CASES), sorted(ENVS), ("standard", "high"),
                                 (None, "nodal"), sorted(STREAMS), ("direct", "pwe", "session"))
    if p[5] == "direct" or (p[2] == "standard" and p[4] == "all")], ids=_ids)
def test_snowball_identity(value_mode, case, env_name, accuracy, projection, streams, entry):
    value_mode("snowball")
    make, over = SNOWBALL_CASES[case]
    solver = SnowballPDESolver(PDEParams(accuracy=accuracy, event_projection=projection))
    _check(_entry(entry, solver, make(), ENVS[env_name](**over), STREAMS[streams]))


@pytest.mark.parametrize("case", ["cont", "discrete", "reverse"])
def test_snowball_identity_fast(value_mode, case):
    """The matrix's quick slice: one standard flat-env direct run per KI class (the slow matrix has the rest)."""
    value_mode("snowball")
    make, over = SNOWBALL_CASES[case]
    _check(SnowballPDESolver(PDEParams()).calculate_event_stats(make(), env(**over)))


def test_local_vol_snowball_identity(value_mode):
    """Review Focus 5: the local-vol route marches the same value-mode statistics."""
    from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import LocalVolSnowballPDESolver
    from pde_bitwise.capture_book import env_grid_vol, snowball

    value_mode("snowball")
    stats = LocalVolSnowballPDESolver(params=PDEParams()).calculate_event_stats(snowball("continuous"), env_grid_vol())
    assert stats.provenance.statistics_model == "local_vol_1d"
    _check(stats)
