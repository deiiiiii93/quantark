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
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

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


def p_stepper(**kw):
    """Branch after Task 12: the stepper; before: the 0.6.0 home of the same four settings."""
    if "stepper" in _fields(PDEParams):
        from quantark.asset.equity.engine.pde.core.components import CrankNicolson
        return PDEParams(stepper=CrankNicolson(**kw))
    grid_kw = {k: kw.pop(k) for k in ("terminal_damping_steps", "event_damping_steps") if k in kw}
    return PDEParams(grid=GridConfig(**grid_kw) if grid_kw else None, **kw)


def p_theta1():
    return p_stepper(theta=1.0)


def p_damping_0_1():
    """Base: the deleted booleans; branch: the two counts alone (§8.1)."""
    if "use_rannacher" in _fields(PDEParams):
        return PDEParams(use_rannacher=False, event_rannacher_steps=1)
    return p_stepper(terminal_damping_steps=0, event_damping_steps=1)


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
            out.update(_stats_fields(stats))
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
    if hasattr(cls, "explain"):
        n = int(cls(params=PDEParams()).explain(product, env).time.actual_steps)
    else:
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


# --- sub-project 2 outputs (spec 2 §9.1) --------------------------------------

def _field_value(val):
    """One output value (spec 3B §7.1): numbers and numeric arrays as hex, booleans as ``bool:0/1``,
    ``None`` as ``"none"``, anything else (dates, provenance, enums, sets) as ``repr:``."""
    if val is None:
        return "none"
    if isinstance(val, (bool, np.bool_)):
        return f"bool:{int(bool(val))}"
    if isinstance(val, (int, float, np.integer, np.floating)):
        return _hex(val)
    if isinstance(val, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(val, dtype=float).ravel()
        except (TypeError, ValueError):
            return f"repr:{val!r}"
        return [x.hex() for x in arr.tolist()]
    return f"repr:{val!r}"


def _stats_fields(stats, prefix="stats_"):
    """Every field of a statistics dataclass, in field order (booleans and objects included)."""
    return {f"{prefix}{f.name}": _field_value(getattr(stats, f.name)) for f in dataclasses.fields(stats)}


def _dist_fields(dist):
    """Every field of an EventDistribution; the three pre-3B keys keep their names."""
    out = {"dist_times": _field_value(dist.event_times),
           "dist_survival": _field_value(dist.survival_probability)}
    for et in sorted(dist.probabilities, key=str):
        out[f"dist_{et}"] = _field_value(np.atleast_1d(np.asarray(dist.probabilities[et], float)))
    for f in dataclasses.fields(dist):
        if f.name in ("event_times", "survival_probability", "probabilities"):
            continue
        val = getattr(dist, f.name)
        if isinstance(val, dict):
            for key in sorted(val, key=str):
                out[f"dist_{f.name}_{key}"] = _field_value(val[key])
        else:
            out[f"dist_{f.name}"] = _field_value(val)
    return out


def stats_only(solver, product, env, streams=None):
    """Direct calculate_event_stats, no npv: the march without a price (§5.10)."""
    stats = solver.calculate_event_stats(product, env, streams=streams)
    return {"stats": None} if stats is None else _stats_fields(stats)


def pwe(solver, product, env, streams=None):
    res = solver.price_with_events(product, env, streams=streams)
    out = {"npv": _hex(res.npv)}
    dist = res.event_distribution
    if dist is not None:
        out.update(_dist_fields(dist))
    return out


def phoenix_plain():
    return make_phoenix(dict(PHOENIX, memory_coupon=False))


def _aged_to_maturity(product):
    """The state ``time_shift`` leaves a tenor product in once aged to its maturity (tau = 0).

    A dated product cannot express it: ``get_maturity`` raises for a valuation on or after
    the exercise date, so every operation would raise instead of taking the short-circuit.
    """
    product.maturity = 0.0
    return product


def expired_snowball():
    return _aged_to_maturity(snowball())


def expired_phoenix():
    return _aged_to_maturity(make_phoenix(PHOENIX))


def expired_barrier():
    return _aged_to_maturity(barrier(ObservationType.CONTINUOUS))


def env_after_maturity(product):
    """The environment the expired products are priced in (tenor products: expiry is the product's)."""
    return env_flat()


def immediate_ko_snowball():
    """A dated snowball whose first KO observation is the valuation date."""
    from intraday.conftest import dated_snowball
    return dated_snowball(create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028)),
                          datetime(2026, 3, 16))


def immediate_ko_env():
    """Valued on the snowball's first KO date with the spot above its 103 barrier."""
    from intraday.conftest import flat_env
    first_ko = immediate_ko_snowball().barrier_config.ko_observation_schedule.records[0].observation_date
    return flat_env(first_ko, spot=130.0)


def env_with_spot(env, spot):
    """The spot bump ``GreeksCalculator`` applies: a deep copy with the quote moved."""
    out = deepcopy(env)
    out.spot_quote.spot = spot
    return out


def terminal_op(op, cls, product, env):
    if op == "price":
        return solver_price(cls(params=PDEParams()), product, env)
    if op == "greeks":
        g = cls(params=PDEParams()).calculate_greeks(product, env)
        return {f"grid_{k}": _hex(g[k]) for k in sorted(g)}
    if op == "pwe":
        return pwe(cls(params=PDEParams()), product, env)
    if op == "stats":
        return stats_only(cls(params=PDEParams()), product, env)
    if op == "life_surface":
        sol = cls(params=PDEParams()).solve_life_surface(product, env, extra_times=(0.25, 0.5, 0.75))
        return life_surface_outputs(sol)
    raise ValueError(op)


def life_surface_outputs(sol):
    def hexes(a):
        return None if a is None else [float(x).hex() for x in np.asarray(a, float).ravel()]
    return {"v0_first": hexes(sol.v0[:, 0]), "v1_first": hexes(sol.v1[:, 0]),
            "v0_last": hexes(sol.v0[:, -1]), "t0_readout": hexes(sol.t0_readout),
            "t": hexes(sol.t), "actual_steps": int(sol.actual_steps),
            "requested_steps": int(sol.requested_steps)}


def session_stats_row(cls, product):
    # PRICE with {PV, EVENT_STATS} is the prepared adapter's one-solve path
    # (PDESessionValue); it rejects PRICE_DETAILED.
    from quantark.execution import OutputKind, PricingOperation, PricingRequest, PricingSession
    engine = cls(PDEParams())
    with PricingSession() as session:
        outcome = session.execute(engine, PricingRequest(
            product=product, pricing_env=env_flat(), operation=PricingOperation.PRICE,
            outputs=frozenset({OutputKind.PV, OutputKind.EVENT_STATS})))
    out = {"price": _hex(outcome.value.pv), "adapter": outcome.manifest.adapter_id}
    out.update(_stats_fields(outcome.value.event_stats))
    return out


# --- the book ----------------------------------------------------------------

def _facade_products():
    """name -> (product factory, event stats) of the facade rows priced at each accuracy."""
    return {
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


def book():
    rows = {}
    facade = _facade_products()
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


# --- the sub-project 2 registry (spec 2 §9.1, decision 20) ----------------------

@dataclasses.dataclass(frozen=True)
class Row:
    """A book row: its thunk, outputs declared None, a declared failure type, and the tags that
    say which stage-2 flips may move it (``movers.py``, spec 3A §7.1)."""
    fn: object
    none: frozenset = frozenset()
    error: Optional[str] = None   # exception class name the row must raise on both sides
    tags: frozenset = frozenset()


def _row(fn, *, none=(), error=None, tags=()):
    return Row(fn, frozenset(none), error, frozenset(tags))


def _solvers():
    from quantark.asset.equity.engine.pde import (
        KOResetSnowballPDESolver, PhoenixPDESolver, SnowballPDESolver)
    return SnowballPDESolver, PhoenixPDESolver, KOResetSnowballPDESolver


def _autocallables():
    """name -> (product factory, solver class index into ``_solvers()``)."""
    return {
        "snowball_cont": (lambda: snowball("continuous"), 0),
        "snowball_monthly": (lambda: snowball("discrete"), 0),
        "snowball_daily": (lambda: snowball("daily"), 0),
        "snowball_knocked_in": (lambda: snowball("knocked_in"), 0),
        "phoenix_memory": (lambda: make_phoenix(PHOENIX), 1),
        "phoenix_plain": (phoenix_plain, 1),
        "ko_reset": (lambda: make_ko_reset(KO_RESET), 2),
    }


def knock_in_products(knocked_in=False):
    """name -> (product, spot). Spot beyond the barrier when ``knocked_in``."""
    return {
        "up_in_call": (BarrierOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL,
                                     barrier=130.0, barrier_type=BarrierType.UP_IN,
                                     observation_type=ObservationType.CONTINUOUS),
                       131.0 if knocked_in else None),
        "down_in_put": (BarrierOption(strike=100.0, maturity=1.0, option_type=OptionType.PUT,
                                      barrier=80.0, barrier_type=BarrierType.DOWN_IN,
                                      observation_type=ObservationType.CONTINUOUS),
                        79.0 if knocked_in else None),
        "double_in": (DoubleBarrierOption(100.0, OptionType.CALL, 120.0, 80.0,
                                          DoubleBarrierType.KNOCK_IN, maturity=1.0),
                      121.0 if knocked_in else None),
    }


def _vol_solver(name):
    from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import HestonPhoenixPDESolver
    from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
        HestonSLVSnowballPDESolver, HestonSnowballPDESolver)
    from quantark.volmodels.heston import HestonParams
    from quantark.volmodels.slv.leverage import LeverageSurface
    hp = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.6)
    if name == "heston_snowball":
        return HestonSnowballPDESolver(hp, params=PDEParams(), n_x=80, n_v=32, n_t=40)
    if name == "slv_snowball":
        ks = np.array(list(100.0 * np.exp(np.linspace(-1.2, 1.2, 15))))
        lev = LeverageSurface(time_grid=np.linspace(0, 1, 6), strike_grid=ks,
                              leverage_grid=np.ones((6, ks.size)))
        return HestonSLVSnowballPDESolver(hp, lev, params=PDEParams(), n_x=80, n_v=32, n_t=40)
    return HestonPhoenixPDESolver(hp, params=PDEParams(), n_x=80, n_v=32, n_t=40)


def bump_frozen_vs_fresh():
    from quantark.asset.equity.engine.pde import SnowballPDESolver
    p = snowball()
    ctx = SnowballPDESolver(params=PDEParams()).create_bump_context(p, env_flat())
    env101 = env_with_spot(env_flat(), 101.0)
    return {"frozen": _hex(ctx.price(p, env101)),
            "fresh": _hex(SnowballPDESolver(params=PDEParams()).price(p, env101))}


def lv_snowball_prebuilt():
    from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import LocalVolSnowballPDESolver
    surf = LocalVolSnowballPDESolver(params=PDEParams())._build_surface(env_grid_vol())
    return solver_price_greeks(LocalVolSnowballPDESolver(params=PDEParams(), local_vol_surface=surf),
                               snowball(), env_grid_vol())


def settlement_lag_row(kind):
    """Through the solver: the ``PDEEngine`` facade declares no settlement support."""
    from test_structured_numerical_settlement import _convention, _env, _phoenix, _snowball
    from quantark.asset.equity.engine.pde import PhoenixPDESolver, SnowballPDESolver
    product = _snowball(_convention()) if kind == "snowball" else _phoenix(_convention())
    cls = SnowballPDESolver if kind == "snowball" else PhoenixPDESolver
    env = _env()
    out = solver_price_greeks(cls(params=p_default()), product, env)
    bump = GreeksCalculator().calculate_numerical_greeks(
        product=product, pricing_env=env, engine=cls(params=p_default()))
    for k in BUMP_GREEKS:
        out[f"bump_{k}"] = _hex(bump.get(k))
    out.update(stats_only(cls(params=p_default()), product, env))
    return out


def engine_price_greeks(product, env, params):
    """Price and solver Greeks through the facade, without the bump Greeks."""
    out = {"price": _hex(PDEEngine(params=params).price(product, env))}
    g = PDEEngine(params=params).calculate_greeks(product, env)
    for k in ("price", "delta", "gamma"):
        out[f"grid_{k}"] = _hex(g.get(k))
    return out


def engine_bump_greeks(product, env, params):
    bump = GreeksCalculator().calculate_numerical_greeks(
        product=product, pricing_env=env, engine=PDEEngine(params=params))
    return {f"bump_{k}": _hex(bump.get(k)) for k in BUMP_GREEKS}


# --- sub-project 3A (spec 3A §7.1) ---------------------------------------------------

def _cn(**kw):
    """CrankNicolson kwargs; ``damping`` only where the API has it (base: 0.7.0)."""
    from quantark.asset.equity.engine.pde.core.components import CrankNicolson
    if "damping" not in _fields(CrankNicolson):
        kw.pop("damping", None)
    return CrankNicolson(**kw)


def p_full_step(**extra):
    """The explicit 0.7.0 scheme: one whole damped step at the terminal node, two at events."""
    return PDEParams(stepper=_cn(damping="full_step", terminal_damping_steps=1, event_damping_steps=2), **extra)


def p_v070(**extra):
    """The 0.7.0 configuration spelled out: full-step, nodal, the 0.7.0 cap."""
    return p_full_step(event_projection="nodal", grid=GridConfig(max_steps=5000), **extra)


def env_tc():
    """Flat vol on the SSE trading clock, 252 days a year.

    252, not 244, so that flip (a) does not move these rows and flip (e) is the only
    flip that does.
    """
    from quantark.param.vol.trading_clock_surface import TradingClockVolSurface
    from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
    anchor = datetime(2026, 2, 9)
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2029))
    tmap = BusinessTimeMap(TradingClock(cal, 252), anchor, datetime(2028, 6, 30))
    return PricingEnvironment(rate_curve=FlatRateCurve(0.025), valuation_date=anchor,
                              spot_quote=SpotQuote(100.0),
                              vol_surface=TradingClockVolSurface(FlatVolSurface(0.22), tmap),
                              div_yield=ContinuousDividendYield(0.03), bus_days_in_year=252)


def solver_greeks_lag(cls, product, env, params):
    """Solver-class price and Greeks: the facade declares no settlement support (spec §14, review 2)."""
    return solver_price_greeks(cls(params=params), product, env)


def _p3a():
    """The 3A evidence's products (test/pde_3a/products.py)."""
    from pde_3a import products
    return products


#: Tags of the rows that predate 3A, by row-id glob (spec 3A §7.1). The rules:
#:   default_damping     the params leave the stepper's damping at its default and the product
#:                       has no continuous knock-in (the core class that flips at (f)); American
#:                       is a core route, the local-vol vanilla/barrier kernels and DCN and 2-D
#:                       have their own keys and do not flip
#:   continuous_ki       the product's knock-in is monitored continuously (never moves at (f))
#:   default_projection  base-family discrete monitoring under the route's default projection
#:   bus244              the solve's day basis is not 252: env_244, and the intraday route,
#:                       whose desk profile runs on 244 days (F1)
#:   snowball_nodal      the snowball priced with an explicit nodal projection (flip d)
#:   barrier_call_*      a discrete-monitoring Barrier call whose upper row is a far-field row
#:                       (flip g; flat curves move by ULPs at most, F2)
#:   put_far_field       a put priced through the European put's lower row (flip g)
#: Settlement-lag products: continuous knock-in (test_structured_numerical_settlement._barriers).
_EXISTING_TAGS = (
    ("default_damping", (
        "european|*", "american_tenor|*", "american_dated|*", "barrier_cont|*", "barrier_disc|*",
        "double_barrier|*", "one_touch|*", "double_one_touch|*",
        "snowball_monthly|*", "snowball_daily|*", "stats|snowball_monthly|*", "stats|snowball_daily|*",
        "pwe|snowball_monthly|*", "pwe|snowball_daily|*", "scheme|european|*", "knock_in|*",
        "terminal|expired|barrier|*", "terminal|immediate_ko|snowball|*", "intraday_snowball")),
    ("continuous_ki", (
        "snowball_cont|*", "snowball_knocked_in|*", "ko_reset|*", "phoenix_memory|*", "phoenix_plain|*",
        "stats|snowball_cont|*", "stats|snowball_knocked_in|*", "stats|phoenix_*", "stats|ko_reset|*",
        "pwe|snowball_cont|*", "pwe|snowball_knocked_in|*", "pwe|phoenix_*", "pwe|ko_reset|*",
        "scheme|snowball_cont|*", "terminal|expired|snowball|*", "terminal|expired|phoenix|*",
        "session_*", "life_surface|snowball", "bump|frozen_vs_fresh|snowball", "lv_snowball*", "lv_phoenix",
        "heston_*", "slv_snowball*", "lu_fallback|*", "settlement_lag|*")),
    ("default_projection", ("barrier_disc|*",)),
    ("bus244", ("*|bus244", "intraday_snowball")),
    ("snowball_nodal", ("snowball_cont|flat|nodal", "pwe|snowball_monthly|flat|nodal",
                        "stats|snowball_monthly|flat|nodal")),
    ("barrier_call_kinked", ("barrier_disc|term|*",)),
    ("barrier_call_flat", ("barrier_disc|flat|*",)),
    ("put_far_field", ("knock_in|down_in_put|*",)),
)


def _existing_tags(rid):
    import fnmatch
    return frozenset(tag for tag, globs in _EXISTING_TAGS if any(fnmatch.fnmatchcase(rid, g) for g in globs))


def _coincident_product():
    """Two records at t = 0.5 whose thresholds share one dual cell of the standard layout.

    g = 0.01 (measured, Task 1): log(130) and log(130.01) sit in the same dual cell of
    ``BarrierPDESolver(PDEParams()).explain(product, env_flat())``'s space grid, a cell
    2.6e-3 wide in log spot against a 7.7e-5 gap.
    """
    from quantark.asset.equity.product.option.observation_schedule import (
        ObservationRecord, ObservationSchedule)
    from quantark.util.enum.option_enums import ObservationAggregation
    g = 0.01
    return BarrierOption(
        strike=100.0, maturity=1.0, option_type=OptionType.CALL, barrier=130.0,
        barrier_type=BarrierType.UP_OUT, observation_type=ObservationType.DISCRETE,
        observation_schedule=ObservationSchedule(records=[
            ObservationRecord(observation_time=0.5, barrier=130.0, payoff=2.0),
            ObservationRecord(observation_time=0.5, barrier=130.0 + g, payoff=3.0),
            ObservationRecord(observation_time=1.0, barrier=130.0, payoff=1.0)],
            aggregation_mode=ObservationAggregation.ACCUMULATE))


def _farfield_products(lag):
    """e17's products (``e17_put_boundary_term_curves.py::PRODUCTS``) and its down-in put."""
    from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
    obs = [k / 12.0 for k in range(1, 13)]
    kw = dict(settlement_convention=SettlementConvention(lag=lag, lag_unit=SettlementLagUnit.YEAR_FRACTION)) if lag else {}

    def bar(kind, level, btype):
        return BarrierOption(strike=100.0, option_type=kind, barrier=level, barrier_type=btype,
                             observation_type=ObservationType.DISCRETE, observation_dates=obs,
                             maturity=1.0, **kw)
    return {
        "european_put": EuropeanVanillaOption(strike=100.0, option_type=OptionType.PUT, maturity=1.0, **kw),
        "uo_put_130": bar(OptionType.PUT, 130.0, BarrierType.UP_OUT),
        "di_put_85": bar(OptionType.PUT, 85.0, BarrierType.DOWN_IN),
        "do_call_85": bar(OptionType.CALL, 85.0, BarrierType.DOWN_OUT),
        "european_call": EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0, **kw),
    }


def _class_product(family, mode, state):
    """The damping-class rows' products (spec 3A decision 12)."""
    p3 = _p3a()
    if family == "snowball":
        p = snowball("continuous" if mode == "cont" else "discrete")
    elif family == "phoenix":
        p = make_phoenix(PHOENIX) if mode == "cont" else p3.phoenix(100.0, {}, "monthly")
    else:
        p = make_ko_reset(KO_RESET) if mode == "cont" else p3.ko_reset_discrete()
    if state == "knocked_in":
        setattr(p, "_otc_lifecycle_knocked_in", True)
    return p


def _bumpcross_product(family, mode):
    """Knock-in barrier at 85 on every family, so 85.5 and 84.9 sit either side of it."""
    disc = {} if mode == "cont" else dict(ki_monitoring="discrete")
    if family == "snowball":
        return make_snowball(dict(SNOWBALL, **disc))
    if family == "phoenix":
        return make_phoenix(dict(PHOENIX, ki_barrier=85.0, coupon_barrier=90.0, **disc))
    return make_ko_reset(dict(KO_RESET, ki_barrier=85.0, **disc))


def _d16_row(market, variant, at):
    p3 = _p3a()
    if market == "flat":
        spot = 100.0 if at == "1.0" else 76.0
        return facade_outputs(p3.phoenix(100.0, *p3.PHOENIX_VARIANTS[variant]), p3.env(spot=spot), p_default())
    from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import LocalVolPhoenixPDESolver
    product, env, lv = p3.localvol_phoenix_case(market, 1.0 if at == "1.0" else 0.76, variant)
    return solver_price_greeks(LocalVolPhoenixPDESolver(params=PDEParams(), local_vol_surface=lv), product, env)


def _lv_class_row(family, mode):
    from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import LocalVolPhoenixPDESolver
    from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import LocalVolSnowballPDESolver
    if family == "snowball":
        return solver_price(LocalVolSnowballPDESolver(params=PDEParams()),
                            snowball("continuous" if mode == "cont" else "discrete"), env_grid_vol())
    product = make_phoenix(PHOENIX) if mode == "cont" else _p3a().phoenix(100.0, {}, "monthly")
    return solver_price(LocalVolPhoenixPDESolver(params=PDEParams()), product, env_grid_vol())


def _base6_row(name, mode):
    p3 = _p3a()
    product = p3.PRODUCTS[name][1](p3.dates(1.0 / 12.0))
    return facade_outputs(product, env_flat(), p_default() if mode == "default" else p_v070())


def rows_3a(existing):
    """The 3A rows (spec 3A §7.1). ``existing`` is the registry of the rows that predate 3A."""
    from quantark.asset.equity.engine.pde import BarrierPDESolver, EuropeanPDESolver
    from quantark.asset.equity.engine.pde.american_pde_solver import AmericanPDESolver
    out = {}
    disc_tags = ("discrete_ki", "default_damping")

    # explicit full-step copies of every facade row priced with the default params
    copies = {}
    for name, (make, stats) in _facade_products().items():
        for env_name, env in (("flat", env_flat), ("term", env_term)):
            for acc in ACCURACIES:
                copies[f"{name}|{env_name}|{acc}"] = (
                    lambda m=make, e=env, a=acc, s=stats: facade_outputs(m(), e(), p_full_step(accuracy=a), event_stats=s))
    for env_name, env in (("flat", env_flat), ("term", env_term)):
        for acc in ACCURACIES:
            copies[f"phoenix_plain|{env_name}|{acc}"] = (
                lambda e=env, a=acc: facade_outputs(phoenix_plain(), e(), p_full_step(accuracy=a), event_stats=True))
    copies["snowball_cont|flat|nodal"] = lambda: facade_outputs(
        snowball(), env_flat(), p_full_step(event_projection="nodal"), True)
    for name, make in (("snowball_cont", snowball), ("european", european)):
        copies[f"{name}|bus244"] = lambda m=make: facade_outputs(m(), env_244(), p_full_step())
    for rid, fn in copies.items():
        assert rid in existing, rid
        tags = (set(existing[rid].tags) - {"default_damping"}) | {"explicit_full_step"}
        out[f"3a|fullstep|{rid}"] = _row(fn, tags=tags)

    # the six base-family discrete products: today's default, and 0.7.0 spelled out
    for name in ("uo_call", "di_put", "double_ko_call", "one_touch_up_at_hit", "no_touch_up",
                 "double_one_touch_at_hit"):
        for mode in ("default", "v070"):
            if mode == "default":
                tags = {"base_discrete", "default_damping", "default_projection"}
                if name == "di_put":
                    tags.add("put_far_field")
            else:
                tags = {"v070_base"} | ({"put_far_field_v070"} if name == "di_put" else set())
            if name == "uo_call":
                tags.add("barrier_call_flat")
            out[f"3a|base6|{name}|{mode}"] = _row(lambda n=name, m=mode: _base6_row(n, m), tags=tags)

    # day basis 244 (flip a)
    for name, make, stats in (("snowball_cont", lambda: snowball("continuous"), True),
                              ("phoenix_memory", lambda: make_phoenix(PHOENIX), True),
                              ("european", european, False)):
        tags = {"bus244"} | ({"default_damping"} if name == "european" else {"continuous_ki"})
        out[f"3a|bus244|{name}"] = _row(
            lambda m=make, s=stats: facade_outputs(m(), env_244(), p_default(), event_stats=s), tags=tags)

    # the step cap (flip b): a fill that the 0.7.0 cap of 5000 scales, and the same under a cap of 20000
    for name, spec, acc in (("sb5y_standard", dict(SNOWBALL, months=60, maturity=5.0), "standard"),
                            ("sb3y_high", dict(SNOWBALL, months=36, maturity=3.0), "high")):
        out[f"3a|capped|{name}|default"] = _row(
            lambda s=spec, a=acc: facade_outputs(make_snowball(s), env_flat(), p_default(a)),
            tags={"capped", "continuous_ki"})
        out[f"3a|capped|{name}|cap20000"] = _row(
            lambda s=spec, a=acc: facade_outputs(make_snowball(s), env_flat(),
                                                 PDEParams(accuracy=a, grid=GridConfig(max_steps=20000))),
            tags={"continuous_ki"})

    # the trading clock (flip e moves the Phoenix only); the Phoenix here has continuous knock-in
    out["3a|tc|snowball_monthly"] = _row(
        lambda: facade_outputs(snowball("discrete"), env_tc(), p_default(), event_stats=True),
        tags={"trading_clock"} | set(disc_tags))
    out["3a|tc|phoenix_plain"] = _row(
        lambda: facade_outputs(phoenix_plain(), env_tc(), p_default(), event_stats=True),
        tags={"trading_clock", "phoenix", "continuous_ki"})

    # continuous knock-in keeps full-step on every solve (spec §5.2)
    for acc in ACCURACIES:
        out[f"3a|cont_ki|snowball|{acc}"] = _row(
            lambda a=acc: facade_outputs(snowball("continuous"), env_flat(), p_default(a), event_stats=True),
            tags={"continuous_ki"})

    # damping classes (decision 12): keyed on the contract, alive or knocked in
    for family in ("snowball", "phoenix", "ko_reset"):
        for mode in ("cont", "disc"):
            for state in ("alive", "knocked_in"):
                out[f"3a|class|{family}|{mode}|{state}"] = _row(
                    lambda f=family, m=mode, s=state: facade_outputs(_class_product(f, m, s), env_flat(),
                                                                     p_default(), event_stats=True),
                    tags={"continuous_ki"} if mode == "cont" else set(disc_tags))
    out["3a|class|snowball_daily"] = _row(
        lambda: facade_outputs(snowball("daily"), env_flat(), p_default(), True), tags=disc_tags)
    for family in ("snowball", "phoenix"):
        for mode in ("cont", "disc"):
            tags = {"continuous_ki"} if mode == "cont" else set(disc_tags)
            if mode == "disc" and family == "phoenix":
                tags.add("lv_phoenix")
            out[f"3a|class|lv_{family}|{mode}"] = _row(lambda f=family, m=mode: _lv_class_row(f, m), tags=tags)

    # decision 16: discrete-KI Phoenix variants, flat and local vol, and the discrete-KI KO-reset
    for market in ("flat", "crash", "calm"):
        for variant in ("daily_ki", "daily_ki_memory", "coupon_at_ko", "coupon_near_ko"):
            for at in ("1.0", "ki+1"):
                tags = set(disc_tags) | ({"lv_phoenix"} if market != "flat" else set())
                out[f"3a|d16|{market}|phoenix|{variant}|{at}"] = _row(
                    lambda mk=market, v=variant, a=at: _d16_row(mk, v, a), tags=tags)
    for at, spot in (("1.0", 100.0), ("ki+1", 81.0)):
        out[f"3a|d16|ko_reset|{at}"] = _row(
            lambda s=spot: facade_outputs(_p3a().ko_reset_discrete(), _p3a().env(spot=s), p_default()),
            tags=disc_tags)

    # a 1% spot bump either side of the knock-in barrier at 85 must not change the class
    for family in ("snowball", "phoenix", "ko_reset"):
        for mode in ("cont", "disc"):
            for spot in ("85.5", "84.9"):
                out[f"3a|bumpcross|{family}|{mode}|{spot}"] = _row(
                    lambda f=family, m=mode, s=float(spot): engine_bump_greeks(
                        _bumpcross_product(f, m), env_with_spot(env_flat(), s), p_default()),
                    tags={"continuous_ki"} if mode == "cont" else set(disc_tags))

    # far-field rows (flip g), on flat and kinked curves, with and without a settlement lag
    for name in ("european_put", "uo_put_130", "di_put_85", "do_call_85", "european_call"):
        is_barrier = name not in ("european_put", "european_call")
        for shape in ("flat", "kinked"):
            for spot in (100, 80):
                for lag_tag, lag in (("lag0", 0.0), ("lag01", 0.1)):
                    tags = {"default_damping"} | ({"default_projection"} if is_barrier else set())
                    if "put" in name:
                        tags.add("put_far_field")
                    elif name == "do_call_85":
                        tags.add("barrier_call_kinked" if shape == "kinked" else "barrier_call_flat")
                    cls = BarrierPDESolver if is_barrier else EuropeanPDESolver
                    out[f"3a|farfield|{name}|{shape}|{spot}|{lag_tag}"] = _row(
                        lambda c=cls, n=name, sh=shape, s=float(spot), lg=lag: solver_greeks_lag(
                            c, _farfield_products(lg)[n], make_term_env(sh, spot=s), p_default()),
                        tags=tags)
    for shape in ("flat", "kinked"):
        out[f"3a|farfield|american_put|{shape}"] = _row(
            lambda sh=shape: solver_price_greeks(AmericanPDESolver(params=p_default()),
                                                 AmericanOption(100.0, OptionType.PUT, maturity=1.0),
                                                 make_term_env(sh)),
            tags={"default_damping"})

    # two records at one node (decision 14); a discrete up-and-out call, so its upper row is far field
    for mode, proj in (("nodal", "nodal"), ("projected", "cell_average")):
        out[f"3a|coincident|{mode}"] = _row(
            lambda p=proj: facade_outputs(_coincident_product(), env_flat(), p_full_step(event_projection=p)),
            tags={"coincident", "barrier_call_flat"})
    return out


def rows():
    out = {rid: Row(fn) for rid, fn in book().items()}
    for env_name, env in (("flat", env_flat), ("term", env_term)):
        for acc in ACCURACIES:
            out[f"phoenix_plain|{env_name}|{acc}"] = _row(
                lambda e=env, a=acc: facade_outputs(phoenix_plain(), e(), p_default(a), event_stats=True))
    for name, (make, si) in _autocallables().items():
        for env_name, env in (("flat", env_flat), ("term", env_term)):
            for acc in ACCURACIES:
                out[f"stats|{name}|{env_name}|{acc}"] = _row(
                    lambda m=make, e=env, a=acc, i=si: stats_only(_solvers()[i](params=p_default(a)), m(), e()))
            out[f"pwe|{name}|{env_name}|standard"] = _row(
                lambda m=make, e=env, i=si: pwe(_solvers()[i](params=p_default()), m(), e()))

    def ko_streams():
        from quantark.cashleg.event_distribution import EventType
        return frozenset({EventType.KO})

    out["stats|snowball_cont|flat|standard|streams_ko"] = _row(
        lambda: stats_only(_solvers()[0](params=p_default()), snowball(), env_flat(), streams=ko_streams()))
    out["pwe|snowball_cont|flat|standard|streams_ko"] = _row(
        lambda: pwe(_solvers()[0](params=p_default()), snowball(), env_flat(), streams=ko_streams()))
    out["stats|snowball_monthly|flat|nodal"] = _row(
        lambda: stats_only(_solvers()[0](params=p_nodal()), snowball("discrete"), env_flat()))
    out["pwe|snowball_monthly|flat|nodal"] = _row(
        lambda: pwe(_solvers()[0](params=p_nodal()), snowball("discrete"), env_flat()))

    out["session_phoenix"] = _row(lambda: session_stats_row(_solvers()[1], make_phoenix(PHOENIX)))
    out["session_ko_reset"] = _row(lambda: session_stats_row(_solvers()[2], make_ko_reset(KO_RESET)))
    # t0_readout is None when the valuation date carries no events (LifeSurfaceSolution).
    out["life_surface|snowball"] = _row(lambda: life_surface_outputs(
        PDEEngine(params=PDEParams()).solve_life_surface(snowball(), env_flat(), extra_times=(0.25, 0.5, 0.75))),
        none=("t0_readout",))

    for name in ("heston_snowball", "slv_snowball", "heston_phoenix"):
        prod = phoenix_plain if name == "heston_phoenix" else snowball
        out[f"{name}|stats"] = _row(lambda n=name, m=prod: stats_only(_vol_solver(n), m(), env_flat()))
    out["heston_snowball|greeks"] = _row(
        lambda: solver_price_greeks(_vol_solver("heston_snowball"), snowball(), env_flat()))

    for name, make in (("european", european), ("snowball_cont", snowball)):
        is_sb = name != "european"
        out[f"scheme|{name}|theta1"] = _row(
            lambda m=make, s=is_sb: facade_outputs(m(), env_flat(), p_theta1(), event_stats=s))
        out[f"scheme|{name}|damp01"] = _row(
            lambda m=make, s=is_sb: facade_outputs(m(), env_flat(), p_damping_0_1(), event_stats=s))

    for pname, make, si in (("snowball", expired_snowball, 0), ("phoenix", expired_phoenix, 1)):
        for op in ("price", "greeks", "pwe", "stats"):
            out[f"terminal|expired|{pname}|{op}"] = _row(
                lambda m=make, i=si, o=op: terminal_op(o, _solvers()[i], m(), env_after_maturity(m())),
                none=("stats",) if op == "stats" else ())

    def barrier_solver():
        from quantark.asset.equity.engine.pde import BarrierPDESolver
        return BarrierPDESolver

    for op in ("price", "greeks"):
        out[f"terminal|expired|barrier|{op}"] = _row(
            lambda o=op: terminal_op(o, barrier_solver(), expired_barrier(), env_after_maturity(expired_barrier())))
    for op in ("price", "greeks", "pwe", "stats"):
        out[f"terminal|immediate_ko|snowball|{op}"] = _row(
            lambda o=op: terminal_op(o, _solvers()[0], immediate_ko_snowball(), immediate_ko_env()))
    out["terminal|immediate_ko|snowball|life_surface"] = _row(
        lambda: terminal_op("life_surface", _solvers()[0], immediate_ko_snowball(), immediate_ko_env()),
        error="ValidationError")

    out["bump|frozen_vs_fresh|snowball"] = _row(bump_frozen_vs_fresh)
    out["lv_snowball|prebuilt"] = _row(lv_snowball_prebuilt)

    # Knock-in bump Greeks raise on base whenever a bumped state needs the
    # knock-out leg: the bump context freezes the knock-in request's layout
    # (no hard bounds) and the knock-out solve refuses it. Recorded as
    # declared failures; price and solver Greeks are their own rows.
    def knocked_in_env(n):
        return env_with_spot(env_flat(), knock_in_products(True)[n][1])

    for name in ("up_in_call", "down_in_put", "double_in"):
        for env_name, env in (("flat", env_flat), ("term", env_term)):
            out[f"knock_in|{name}|{env_name}|standard"] = _row(
                lambda n=name, e=env: engine_price_greeks(knock_in_products()[n][0], e(), p_default()))
            out[f"knock_in|{name}|{env_name}|standard|bump"] = _row(
                lambda n=name, e=env: engine_bump_greeks(knock_in_products()[n][0], e(), p_default()),
                error="PricingError")
        out[f"knock_in|{name}|knocked_in"] = _row(
            lambda n=name: engine_price_greeks(knock_in_products(True)[n][0], knocked_in_env(n), p_default()))
        # down_in_put at 79 stays knocked in under both 1% spot bumps; the
        # other two cross back inside their barriers on one of them.
        out[f"knock_in|{name}|knocked_in|bump"] = _row(
            lambda n=name: engine_bump_greeks(knock_in_products(True)[n][0], knocked_in_env(n), p_default()),
            error=None if name == "down_in_put" else "PricingError")

    def lu_params():
        return PDEParams(grid=GridConfig(points=4, eps_crit=1.0))

    out["lu_fallback|snowball"] = _row(
        lambda: solver_price_greeks(_solvers()[0](params=lu_params()), snowball(), env_flat()))
    out["lu_fallback|phoenix"] = _row(
        lambda: solver_price_greeks(_solvers()[1](params=lu_params()), make_phoenix(PHOENIX), env_flat()))
    out["lu_fallback|snowball|stats"] = _row(
        lambda: stats_only(_solvers()[0](params=lu_params()), snowball(), env_flat()),
        error="ValidationError")

    for kind in ("snowball", "phoenix"):
        out[f"settlement_lag|{kind}"] = _row(lambda k=kind: settlement_lag_row(k))

    # spec 3A §7.1: tag the rows above by rule (_EXISTING_TAGS), then add the 3A rows
    out = {rid: dataclasses.replace(r, tags=r.tags | _existing_tags(rid)) for rid, r in out.items()}
    out.update(rows_3a(out))
    return out


def run_row(row_id):
    try:
        return row_id, rows()[row_id].fn()
    except Exception as exc:  # a row that raises is a diff, not a crash
        # The trace names this worktree's paths: stderr only, so books from
        # worktrees at different paths compare equal.
        print(f"[{row_id}] {traceback.format_exc(limit=4)}", file=sys.stderr)
        return row_id, {"__error__": f"{type(exc).__name__}: {exc}"}


def write_manifest(path):
    manifest = {rid: {"none": sorted(r.none), "error": r.error}
                for rid, r in sorted(rows().items())}
    Path(path).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"{len(manifest)} rows -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="substring filter on row ids")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--write-manifest", default=None, metavar="PATH",
                    help="write every row id with its declared None outputs and failure type, then exit")
    args = ap.parse_args(argv)
    if args.write_manifest:
        write_manifest(args.write_manifest)
        return
    if not args.out:
        ap.error("--out is required unless --write-manifest is given")
    ids = [r for r in rows() if not args.only or args.only in r]
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
