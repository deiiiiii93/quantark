"""Byte-capture book for the PDE 1D surface cleanup (spec §8.1).

Prices a fixed book and writes every output as ``float.hex()``. Run it on the
base commit (detached worktree) and on the branch, same machine, back to back,
then diff with ``compare.py``. The base JSON is never committed (cross-arch
goldens are fragile); the diff report goes in the merge commit message.

Rows whose construction changed between base and branch detect the API they
run on per row (the legacy fields leave one family per task), so this one
script serves the base commit and every branch commit.

Not collected by pytest (no test_ prefix). Rows run in a process pool
(``--workers``); the ``__main__`` guard is what makes spawn workers safe.
"""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parents[1]
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))  # test/ helpers (term_structure_benchmarks, dcn_fixtures)

import numpy as np  # noqa: E402

from term_structure_benchmarks import make_term_env  # noqa: E402

from quantark.asset.equity.engine import PDEEngine  # noqa: E402
from quantark.asset.equity.engine.pde.grid.config import GridConfig  # noqa: E402
from quantark.asset.equity.param import PDEParams  # noqa: E402
from quantark.asset.equity.product.option import (  # noqa: E402
    AmericanOption, BarrierOption, DoubleBarrierOption, DoubleOneTouchOption,
    EuropeanVanillaOption, OneTouchOption,
)
from quantark.asset.equity.riskmeasures import GreeksCalculator  # noqa: E402
from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset  # noqa: E402
from quantark.modelvalidation.builders.equity_phoenix import make_phoenix  # noqa: E402
from quantark.modelvalidation.builders.equity_snowball import make_snowball  # noqa: E402
from quantark.param import FlatRateCurve, FlatVolSurface, GridVolSurface, SpotQuote  # noqa: E402
from quantark.param.div import ContinuousDividendYield  # noqa: E402
from quantark.priceenv import PricingEnvironment  # noqa: E402
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar  # noqa: E402
from quantark.util.enum import BarrierType, ObservationType, OptionType  # noqa: E402
from quantark.util.enum.option_enums import (  # noqa: E402
    BarrierDirection, DoubleBarrierType, TouchType,
)

ACCURACIES = ("fast", "standard", "high")
SNOWBALL = dict(initial_price=100.0, strike=100.0, ko_barrier=103.0, ki_barrier=85.0,
                ko_rate=0.15, rebate_rate=0.15, months=12, maturity=1.0,
                contract_multiplier=1.0)
PHOENIX = dict(initial_price=100.0, strike=100.0, ko_barrier=103.0, ko_rate=0.0,
               ki_barrier=75.0, coupon_barrier=85.0, coupon_rate=0.02,
               memory_coupon=True, num_observations=12, maturity=1.0,
               contract_multiplier=1.0)
KO_RESET = dict(initial_price=100.0, strike=100.0, maturity_pre=0.25, maturity_post=0.5,
                pre_ko_barrier=103.0, pre_ko_rate=0.15, post_ko_barrier=95.0,
                post_ko_rate=0.03, ki_barrier=80.0, ki_continuous=True,
                contract_multiplier=1.0)
BUMP_GREEKS = ("delta", "gamma", "vega", "theta", "rho")


def _fields(cls) -> set:
    return {f.name for f in dataclasses.fields(cls)}


def _hex(x):
    try:
        return float(x).hex()
    except (TypeError, ValueError):
        return None


# --- environments -----------------------------------------------------------

def env_flat():
    return make_term_env("flat")


def env_term():
    return make_term_env("kinked")


def env_244():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.02), div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2026, 6, 26),
        day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=244,
        calendar=create_calendar(CalendarType.CHINA_SSE),
    )


def env_grid_vol():
    strikes = list(100.0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    smile = [0.22 + 0.02 * abs(k - 4) / 4 for k in range(9)]
    surf = GridVolSurface(strikes, list(np.linspace(0.1, 2.0, 6)), np.array([smile] * 6))
    return PricingEnvironment(rate_curve=FlatRateCurve(0.03), valuation_date=datetime(2026, 1, 1),
                              spot_quote=SpotQuote(spot=100.0), vol_surface=surf,
                              div_yield=ContinuousDividendYield(0.01))


def env_grid_vol_dcn():
    from dcn_fixtures import DCN_A
    s0 = float(DCN_A["initial_price"])
    strikes = list(s0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    surf = GridVolSurface(strikes, list(np.linspace(0.1, 3.0, 6)), np.full((6, 9), 0.22))
    return PricingEnvironment(rate_curve=FlatRateCurve(0.02), valuation_date=datetime(2026, 1, 1),
                              spot_quote=SpotQuote(spot=s0), vol_surface=surf,
                              div_yield=ContinuousDividendYield(0.0))


# --- products ---------------------------------------------------------------

def european():
    return EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)


def american_tenor():
    return AmericanOption(100.0, OptionType.PUT, maturity=1.0)


def american_dated():
    return AmericanOption(100.0, OptionType.PUT, exercise_date=datetime(2027, 7, 5))


def barrier(observation):
    kw = {}
    if observation == ObservationType.DISCRETE:
        kw["observation_dates"] = [k / 12.0 for k in range(1, 13)]
    return BarrierOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL, barrier=130.0,
                         barrier_type=BarrierType.UP_OUT, observation_type=observation, **kw)


def double_barrier():
    return DoubleBarrierOption(100.0, OptionType.CALL, 120.0, 80.0,
                               DoubleBarrierType.KNOCK_OUT, maturity=1.0)


def one_touch():
    return OneTouchOption(barrier=115.0, barrier_direction=BarrierDirection.UP, maturity=1.0,
                          rebate=1.0, payment_at_hit=True, touch_type=TouchType.ONE_TOUCH)


def double_one_touch():
    return DoubleOneTouchOption(upper_barrier=115.0, lower_barrier=85.0, maturity=1.0,
                                rebate=1.0, payment_at_hit=True,
                                touch_type=TouchType.DOUBLE_ONE_TOUCH)


def snowball(ki="continuous"):
    if ki == "daily":
        from quantark.asset.equity.product.option.snowball_config import BarrierConfig
        from quantark.asset.equity.product.option.snowball_option import SnowballOption
        ref = make_snowball(dict(SNOWBALL, ki_monitoring="discrete"))
        bc = ref.barrier_config
        return SnowballOption(
            initial_price=ref.initial_price, strike=ref.strike,
            barrier_config=BarrierConfig(
                ko_barrier=bc.ko_barrier, ko_rate=bc.ko_rate,
                ko_observation_type=bc.ko_observation_type,
                ko_observation_dates=list(bc.ko_observation_dates),
                ki_barrier=bc.ki_barrier, ki_continuous=False,
                ki_observation_type=ObservationType.DISCRETE,
                ki_observation_dates=[k / 252.0 for k in range(1, 253)],
            ),
            payoff_config=ref.payoff_config, contract_multiplier=1.0,
            maturity=1.0, is_reverse=False,
        )
    if ki == "knocked_in":
        p = make_snowball(SNOWBALL)
        setattr(p, "_otc_lifecycle_knocked_in", True)
        return p
    return make_snowball(dict(SNOWBALL, ki_monitoring=ki))


# --- params with per-row API detection ---------------------------------------

def p_default(accuracy="standard"):
    return PDEParams(accuracy=accuracy)


def p_nodal():
    return PDEParams(event_projection="nodal")


def p_theta1():
    return PDEParams(theta=1.0)


def p_damping_0_1():
    """Base: the deleted booleans; branch: the two counts alone (§8.1)."""
    if "use_rannacher" in _fields(PDEParams):
        return PDEParams(use_rannacher=False, event_rannacher_steps=1)
    return PDEParams(grid=GridConfig(terminal_damping_steps=0, event_damping_steps=1))


def lv_params(points=None, steps=None):
    if points is None:
        return PDEParams()
    if "grid_size" in _fields(PDEParams):
        return PDEParams(grid_size=points, time_steps=steps)
    return PDEParams(grid=GridConfig(points=points, time_steps=steps))


def dcn_engine(cls, points=None):
    from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
    if points is None:
        return cls()
    # detect on the base class: LocalVolDCNPDEEngine forwards **kwargs
    if "num_space_nodes" in inspect.signature(DCNPDEEngine.__init__).parameters:
        return cls(num_space_nodes=points)
    return cls(params=PDEParams(grid=GridConfig(points=points)))


# --- outputs ----------------------------------------------------------------

def facade_outputs(product, env, params, event_stats=False):
    out = {"price": _hex(PDEEngine(params=params).price(product, env))}
    g = PDEEngine(params=params).calculate_greeks(product, env)
    for k in ("price", "delta", "gamma"):
        out[f"grid_{k}"] = _hex(g.get(k))
    bump = GreeksCalculator().calculate_numerical_greeks(
        product=product, pricing_env=env, engine=PDEEngine(params=params))
    for k in BUMP_GREEKS:
        out[f"bump_{k}"] = _hex(bump.get(k))
    if event_stats:
        stats = PDEEngine(params=params).calculate_event_stats(product, env)
        if stats is not None:
            for name, val in sorted(vars(stats).items()):
                if isinstance(val, bool):
                    continue
                if isinstance(val, (int, float)):
                    out[f"stats_{name}"] = _hex(val)
                elif isinstance(val, (list, tuple, np.ndarray)):
                    try:
                        arr = np.asarray(val, dtype=float).ravel()
                    except (TypeError, ValueError):
                        continue
                    out[f"stats_{name}"] = [x.hex() for x in arr.tolist()]
    return out


def solver_price(solver, product, env):
    return {"price": _hex(solver.price(product, env))}


def solver_price_greeks(solver, product, env):
    out = solver_price(solver, product, env)
    g = solver.calculate_greeks(product, env)
    for k in sorted(g):
        out[f"grid_{k}"] = _hex(g[k])
    return out


def budget_row(product, env):
    """Base: the standard density; branch: a budget equal to that density's fill."""
    from quantark.asset.equity.engine.pde import EuropeanPDESolver, SnowballPDESolver
    cls = EuropeanPDESolver if isinstance(product, EuropeanVanillaOption) else SnowballPDESolver
    if "time_steps" not in _fields(GridConfig):
        return solver_price_greeks(cls(params=PDEParams()), product, env)
    probe = cls(params=PDEParams())
    probe.price(product, env)
    n = int(probe._active_layout.time.actual_steps)
    return solver_price_greeks(cls(params=PDEParams(grid=GridConfig(time_steps=n))), product, env)


def session_row():
    from quantark.asset.equity.engine.pde import SnowballPDESolver
    from quantark.execution import PricingRequest, PricingSession
    engine = SnowballPDESolver(PDEParams())
    with PricingSession() as session:
        outcome = session.execute(
            engine, PricingRequest(product=make_snowball(SNOWBALL), pricing_env=env_flat()))
    return {"price": _hex(outcome.value), "adapter": outcome.manifest.adapter_id}


def intraday_row():
    from pde_bitwise.intraday_fixture import intraday_price
    return {"price": _hex(intraday_price())}


# --- the book ----------------------------------------------------------------

def book():
    rows = {}
    facade = {
        "european": (european, False), "american_tenor": (american_tenor, False),
        "american_dated": (american_dated, False),
        "barrier_cont": (lambda: barrier(ObservationType.CONTINUOUS), False),
        "barrier_disc": (lambda: barrier(ObservationType.DISCRETE), False),
        "double_barrier": (double_barrier, False), "one_touch": (one_touch, False),
        "double_one_touch": (double_one_touch, False),
        "snowball_cont": (lambda: snowball("continuous"), True),
        "snowball_monthly": (lambda: snowball("discrete"), True),
        "snowball_daily": (lambda: snowball("daily"), True),
        "snowball_knocked_in": (lambda: snowball("knocked_in"), True),
        "ko_reset": (lambda: make_ko_reset(KO_RESET), True),
        "phoenix_memory": (lambda: make_phoenix(PHOENIX), True),
    }
    for name, (make, stats) in facade.items():
        for env_name, env in (("flat", env_flat), ("term", env_term)):
            for acc in ACCURACIES:
                rows[f"{name}|{env_name}|{acc}"] = (
                    lambda m=make, e=env, a=acc, s=stats: facade_outputs(m(), e(), p_default(a), event_stats=s))
    rows["snowball_cont|flat|nodal"] = lambda: facade_outputs(snowball(), env_flat(), p_nodal(), True)
    for name, make in (("snowball_cont", snowball), ("european", european)):
        rows[f"{name}|flat|theta1"] = lambda m=make: facade_outputs(m(), env_flat(), p_theta1())
        rows[f"{name}|flat|damp01"] = lambda m=make: facade_outputs(m(), env_flat(), p_damping_0_1())
        rows[f"{name}|flat|budget_eq_density"] = lambda m=make: budget_row(m(), env_flat())
        rows[f"{name}|bus244"] = lambda m=make: facade_outputs(m(), env_244(), p_default())

    def vol_rows():
        from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
            HestonSLVSnowballPDESolver, HestonSnowballPDESolver, LocalVolSnowballPDESolver)
        from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import (
            HestonPhoenixPDESolver, LocalVolPhoenixPDESolver)
        from quantark.volmodels.heston import HestonParams
        from quantark.volmodels.slv.leverage import LeverageSurface
        hp = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.6)
        ks = np.array(list(100.0 * np.exp(np.linspace(-1.2, 1.2, 15))))
        lev = LeverageSurface(time_grid=np.linspace(0, 1, 6), strike_grid=ks,
                              leverage_grid=np.ones((6, ks.size)))
        return {
            "lv_snowball": lambda: solver_price(LocalVolSnowballPDESolver(params=PDEParams()), snowball(), env_grid_vol()),
            "lv_phoenix": lambda: solver_price(LocalVolPhoenixPDESolver(params=PDEParams()), make_phoenix(PHOENIX), env_grid_vol()),
            "heston_snowball": lambda: solver_price(HestonSnowballPDESolver(hp, params=PDEParams(), n_x=80, n_v=32, n_t=40), snowball(), env_flat()),
            "slv_snowball": lambda: solver_price(HestonSLVSnowballPDESolver(hp, lev, params=PDEParams(), n_x=80, n_v=32, n_t=40), snowball(), env_flat()),
            # the 2-D phoenix supports non-memory coupons only
            "heston_phoenix": lambda: solver_price(HestonPhoenixPDESolver(hp, params=PDEParams(), n_x=80, n_v=32, n_t=40), make_phoenix(dict(PHOENIX, memory_coupon=False)), env_flat()),
        }

    for key in ("lv_snowball", "lv_phoenix", "heston_snowball", "slv_snowball", "heston_phoenix"):
        rows[key] = lambda k=key: vol_rows()[k]()

    def lv_solvers():
        from quantark.asset.equity.engine.pde.barrier_vol_pde_solvers import LocalVolBarrierPDESolver
        from quantark.asset.equity.engine.pde.local_vol_pde_solver import LocalVolPDESolver
        return LocalVolPDESolver, LocalVolBarrierPDESolver

    for tag, kw in (("default", {}), ("600x200", dict(points=600, steps=200))):
        rows[f"lv_vanilla|{tag}"] = lambda kw=kw: solver_price_greeks(
            lv_solvers()[0](params=lv_params(**kw)), european(), env_grid_vol())
        for obs_tag, obs in (("cont", ObservationType.CONTINUOUS), ("expiry", ObservationType.EXPIRY),
                             ("discrete", ObservationType.DISCRETE)):
            rows[f"lv_barrier_{obs_tag}|{tag}"] = lambda kw=kw, o=obs: solver_price(
                lv_solvers()[1](lv_params(**kw)), barrier(o), env_grid_vol())

    def dcn_row(lv, pts):
        from dcn_fixtures import DCN_A, FLAT, flat_env, make_dcn
        from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
        from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import LocalVolDCNPDEEngine
        if lv:
            res = dcn_engine(LocalVolDCNPDEEngine, pts).price_detailed(make_dcn(DCN_A), env_grid_vol_dcn())
        else:
            res = dcn_engine(DCNPDEEngine, pts).price_detailed(
                make_dcn(DCN_A), flat_env(**FLAT, spot=DCN_A["initial_price"]))
        return {"pv": _hex(res.pv), "steps": res.num_time_steps}

    for pts in (None, 601):
        tag = "default" if pts is None else str(pts)
        rows[f"dcn|{tag}"] = lambda p=pts: dcn_row(False, p)
        rows[f"lv_dcn|{tag}"] = lambda p=pts: dcn_row(True, p)

    rows["session_snowball"] = session_row
    rows["intraday_snowball"] = intraday_row
    return rows


def run_row(row_id):
    try:
        return row_id, book()[row_id]()
    except Exception as exc:  # a row that raises is a diff, not a crash
        return row_id, {"__error__": f"{type(exc).__name__}: {exc}",
                        "__trace__": traceback.format_exc(limit=4)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="substring filter on row ids")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    ids = [r for r in book() if not args.only or args.only in r]
    if args.workers <= 1:
        results = dict(run_row(r) for r in ids)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = dict(pool.map(run_row, ids))
    Path(args.out).write_text(json.dumps(results, indent=1, sort_keys=True))
    errors = sorted(r for r, v in results.items() if "__error__" in v)
    print(f"{len(results)} rows -> {args.out}; {len(errors)} errors {errors}")


if __name__ == "__main__":
    main()
