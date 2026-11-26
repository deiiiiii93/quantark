"""Taylor PnL explain on TradingClock-wrapped environments (0.16.0 trading-clock run D3, gate AC3).

Taylor used to refuse a wrapped environment. It now runs on the numerical greek route (forced: the analytical
formulas read env.get_vol / env.get_rate on the environment's axis while the factor moves are in the inners' units),
every bump moving the wrapper's inner. The gate:

(i)   each first-order greek equals an independent finite difference built by moving the INNER and re-wrapping on
      the same map, at the calculator's bump size and stencil (delta central; vega, rho, rho-q one-sided up);
(ii)  halving every market move (time step fixed) shrinks |unexplained| at least 2x: no residual ignores the move size
      (one-sided stencils and time cross terms keep it partly first order on every environment, so not 4x);
(iii) on a pure-time step the Taylor time row equals the waterfall TIME row.
"""
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_pnlexplain_trading_clock import CALL, ENG, _cal_env, _td_env  # noqa: E402

from quantark.asset.equity.param import EngineParams  # noqa: E402
from quantark.pnlexplain import ExplainMethod, Factor, PnLExplainConfig, ValuationSnapshot, explain  # noqa: E402
from quantark.util.exceptions import ValidationError  # noqa: E402

AXES = {"calendar_axis": _cal_env, "business_days": _td_env}
STEPS = {"plain": (datetime(2026, 3, 10), datetime(2026, 3, 11)),
         "holiday": (datetime(2026, 9, 30), datetime(2026, 10, 9))}
BASE = dict(spot=100.0, sigma_td=0.20, r=0.02, q=0.01)
MOVE = dict(spot=1.0, sigma_td=0.01, r=0.001, q=0.001)
BUMP = EngineParams().get_effective_bump_config()


def _moved(scale):
    return {k: BASE[k] + scale * MOVE[k] for k in BASE}


def _explain(mk, t0, t1, scale=1.0, greeks_method="auto"):
    return explain(ValuationSnapshot(CALL, ENG, mk(t0, **BASE), date=t0),
                   ValuationSnapshot(CALL, ENG, mk(t1, **_moved(scale)), date=t1),
                   config=PnLExplainConfig(greeks_method=greeks_method))


def _taylor(r, factor):
    rows = [x for x in r.rows if x.method is ExplainMethod.TAYLOR and x.factor is factor
            and x.kind.value == "component"]
    return rows[0]


def _central(mk, t0, key, h):
    up = dict(BASE, **{key: BASE[key] + h})
    dn = dict(BASE, **{key: BASE[key] - h})
    return (ENG.price(CALL, mk(t0, **up)) - ENG.price(CALL, mk(t0, **dn))) / (2.0 * h)


def _one_sided(mk, t0, key, h):
    """The calculator's vega, rho and rho-q stencil (numerical_vega / numerical_rho / numerical_dividend_rho: one-sided up)."""
    up = dict(BASE, **{key: BASE[key] + h})
    return (ENG.price(CALL, mk(t0, **up)) - ENG.price(CALL, mk(t0, **BASE))) / h


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("greeks_method", ["auto", "analytical"])
def test_first_order_greeks_equal_inner_finite_differences(axis, step, greeks_method):
    mk, (t0, t1) = AXES[axis], STEPS[step]
    r = _explain(mk, t0, t1, greeks_method=greeks_method)
    assert r.metadata["route"] == "numerical"
    fd = {Factor.SPOT: _central(mk, t0, "spot", BASE["spot"] * BUMP.spot_bump),
          Factor.VOL: _one_sided(mk, t0, "sigma_td", BUMP.vol_bump),
          Factor.RATE: _one_sided(mk, t0, "r", BUMP.rate_bump),
          Factor.DIVIDEND: _one_sided(mk, t0, "q", BUMP.div_bump)}
    for factor, expected in fd.items():
        assert _taylor(r, factor).greek == pytest.approx(expected, rel=1e-6), factor


def _shrink(mk, t0, t1):
    def residual(scale):
        return _taylor(_explain(mk, t0, t1, scale=scale, greeks_method="numerical"), Factor.UNEXPLAINED).pnl

    u0 = residual(0.0)
    return abs(residual(1.0) - u0) / abs(residual(0.5) - u0)


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("step", STEPS)
def test_halving_the_moves_shrinks_the_residual(axis, step):
    """Halving every market move at least halves |unexplained| (minus the pure-time part): no leftover that ignores
    the move size, such as a time row or a move read on the wrong axis. Not 4x: the calculator's one-sided stencils
    and the cross terms with the fixed time step are first order in the move on ANY environment (an unwrapped env
    measured 3.49x on the plain step, 2.05x across the holiday); gate (i) carries the greeks."""
    mk, (t0, t1) = AXES[axis], STEPS[step]
    assert _shrink(mk, t0, t1) >= 2.0


@pytest.mark.parametrize("axis", AXES)
def test_pure_time_step_matches_the_waterfall_time_row(axis):
    mk, (t0, t1) = AXES[axis], STEPS["plain"]
    r = _explain(mk, t0, t1, scale=0.0)
    wf = [x for x in r.rows if x.method is ExplainMethod.WATERFALL and x.factor is Factor.TIME][0].pnl
    assert _taylor(r, Factor.TIME).pnl == pytest.approx(wf, abs=1e-12)


def test_bucketed_taylor_on_a_wrapped_environment_refuses():
    t0, t1 = STEPS["plain"]
    with pytest.raises(ValidationError, match="bucketed Taylor"):
        explain(ValuationSnapshot(CALL, ENG, _td_env(t0), date=t0), ValuationSnapshot(CALL, ENG, _td_env(t1), date=t1),
                config=PnLExplainConfig(bucketed=True))


def test_snowball_on_pde_taylor_greeks_equal_inner_finite_differences():
    """AC3's path-dependent case: a dated snowball on the 1D PDE, BUSINESS_DAYS env with wrapped rate and dividend."""
    import pandas as pd

    from quantark.asset.equity.engine.pde import SnowballPDESolver
    from quantark.asset.equity.param import PDEParams
    from test_asset.equity.product_other.test_tracker_clock_parity import KINDS

    product = KINDS["snowball"]("dated", pd.Timestamp(datetime(2026, 2, 26)))
    engine = SnowballPDESolver(PDEParams())
    t0, t1 = STEPS["plain"]
    r = explain(ValuationSnapshot(product, engine, _td_env(t0, **BASE), date=t0),
                ValuationSnapshot(product, engine, _td_env(t1, **_moved(1.0)), date=t1),
                config=PnLExplainConfig())
    assert r.metadata["route"] == "numerical"

    from quantark.asset.equity.riskmeasures.greeks.bump_envs import resolve_bump_engine

    # the calculator reprices on its resolved bump engine (a re-gridded solver); the gate uses the same one
    bump_engine = resolve_bump_engine(product, _td_env(t0, **BASE), engine)

    def one_sided(key, h):
        up = dict(BASE, **{key: BASE[key] + h})
        return (bump_engine.price(product, _td_env(t0, **up)) - bump_engine.price(product, _td_env(t0, **BASE))) / h

    rows = {f: _taylor(r, f) for f in (Factor.RATE, Factor.DIVIDEND, Factor.VOL)}     # first component = vega
    assert rows[Factor.RATE].greek == pytest.approx(one_sided("r", BUMP.rate_bump), rel=1e-6)
    assert rows[Factor.DIVIDEND].greek == pytest.approx(one_sided("q", BUMP.div_bump), rel=1e-6)
    assert rows[Factor.VOL].greek == pytest.approx(one_sided("sigma_td", BUMP.vol_bump), rel=1e-6)
