"""Explicit intraday capability matrix (design §Engine integration and limits).

A row is published only for a (product, engine, monitoring) combination whose
semantics are implemented; ``status="qualified"`` additionally means Gate C
evidence exists for the declared profiles/outputs (Plans 2/3 flip rows).
Explicit requests outside the matrix raise ``CapabilityError`` naming
alternatives. Engine classes match exactly: a subclass does not inherit a row.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, replace
from enum import Enum
from datetime import timedelta
from functools import lru_cache
from typing import Iterable, Optional, Tuple

from quantark.execution.errors import CapabilityError

_QUAD_V2 = "quantark.asset.equity.engine.quad.v2.engine."
_ANALYTICAL = "quantark.asset.equity.engine.analytical."

#: Every intraday request prices on a ``TradingClockVolSurface``. QUAD V2's exact
#: continuous-barrier classifier recognizes only its flat/term curve families, so the
#: clock wrapper falls through to a time-refinement approximation that nothing here has
#: qualified, and its grid builder does not split at the intraday map's session knots.
#: Publishing the row would advertise an accuracy claim no evidence supports.
_CONTINUOUS_QUAD_NOTE = (
    "continuous monitoring on the intraday clock needs an interval survival/crossing operator split at every "
    "session and coefficient knot, with its own time-refinement and first-passage qualification (design "
    "\u00a7Engine integration). QUAD V2's exact continuous classifier does not recognize TradingClockVolSurface "
    "and its continuous grid builder does not carry the intraday map's knots")


@dataclass(frozen=True)
class IntradayCapability:
    product_type: type
    engine_class_path: str                 # f"{module}.{qualname}" of the engine class
    monitoring: str                        # "discrete" | "continuous" | "terminal"
    profiles: frozenset                    # {"any"} or the profile names the route is qualified for
    outputs: frozenset                     # {"price", "delta", ...} the route can deliver
    status: str                            # "supported" | "qualified" | "unsupported"
    note: str = ""
    alternatives: Tuple[str, ...] = ()
    #: shortest time to the next fixing at which every Gate C price cell of the row passed (``qualified`` requires it);
    #: timestamp support below it does not imply an accuracy certificate there
    qualified_horizon: Optional[timedelta] = None

    def __post_init__(self):
        if (self.status == "qualified") != (self.qualified_horizon is not None):
            raise ValueError(f"{self.engine_class_path}: a qualified row needs a qualified horizon and only it has one")


def engine_class_path(engine) -> str:
    cls = engine if isinstance(engine, type) else type(engine)
    return f"{cls.__module__}.{cls.__qualname__}"


def _rows() -> Tuple[IntradayCapability, ...]:
    from quantark.asset.equity.product.option.barrier_option import BarrierOption
    from quantark.asset.equity.product.option.digital_option import CashOrNothingDigitalOption
    from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
    from quantark.asset.equity.product.option.one_touch_option import OneTouchOption
    from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
    from quantark.asset.equity.product.option.snowball_option import SnowballOption
    # Every priced route delivers desk-bump greeks: re-evaluations of the same route on bumped environments.
    price_only = frozenset({"price", "delta", "gamma", "vega", "rho", "dividend_rho", "theta"})
    anywhere = frozenset({"any"})
    barrier_rows = []
    for product, engine in ((BarrierOption, "barrier_analytical_engine.BarrierAnalyticalEngine"),
                            (OneTouchOption, "one_touch_analytical_engine.OneTouchAnalyticalEngine")):
        barrier_rows += [
            IntradayCapability(product, _ANALYTICAL + engine, "continuous", anywhere, price_only, "supported",
                               "exact only under a uniform calendar variance rate with flat carry, or zero carry "
                               "(variance-time change); otherwise CapabilityError"),
            IntradayCapability(product, _ANALYTICAL + engine, "terminal", anywhere, price_only, "supported",
                               "expiry-only monitoring: terminal distribution, exact"),
            IntradayCapability(product, _ANALYTICAL + engine, "discrete", anywhere, price_only, "unsupported",
                               "BGK is an approximation; discrete barriers route to PDE/QUAD/MC"),
        ]
    pde = "quantark.asset.equity.engine.pde."
    pde_note = ("resolution reported per price (resolved / under_resolved / deterministic): cells and time steps across the "
                "diffusion layer, grid-mode damping, barrier placement; refinement bounded by a grid-cell memory budget")
    pde_rows = []
    for product, path, modes in (
            (SnowballOption, "snowball_pde_solver.SnowballPDESolver", ("discrete", "continuous")),
            (PhoenixOption, "phoenix_pde_solver.PhoenixPDESolver", ("discrete", "continuous")),
            (KnockOutResetSnowballOption, "ko_reset_snowball_pde_solver.KOResetSnowballPDESolver", ("discrete", "continuous")),
            (BarrierOption, "barrier_pde_solver.BarrierPDESolver", ("discrete", "continuous", "terminal")),
            (OneTouchOption, "one_touch_pde_solver.OneTouchPDESolver", ("discrete", "continuous", "terminal"))):
        pde_rows += [IntradayCapability(product, pde + path, mode, anywhere, price_only, "supported", pde_note) for mode in modes]
    mc = "quantark.asset.equity.engine.mc."
    mc_note = "paths, seed, standard error and estimator reported per price"
    mc_rows = []
    for product, path, modes in (
            (SnowballOption, "snowball_mc_engine.SnowballMCEngine", ("discrete", "continuous")),
            (PhoenixOption, "phoenix_mc_engine.PhoenixMCEngine", ("discrete", "continuous")),
            (BarrierOption, "barrier_option_mc_engine.BarrierOptionMCEngine", ("discrete", "continuous", "terminal")),
            (CashOrNothingDigitalOption, "digital_option_mc_engine.DigitalOptionMCEngine", ("terminal",))):
        mc_rows += [IntradayCapability(product, mc + path, mode, anywhere, price_only, "supported", mc_note) for mode in modes]
    return tuple(barrier_rows) + tuple(pde_rows) + tuple(mc_rows) + (
        IntradayCapability(CashOrNothingDigitalOption, _ANALYTICAL + "digital_option_engine.DigitalOptionAnalyticalEngine",
                           "terminal", anywhere, price_only, "supported",
                           "integrated carry/variance via TradingClockVolSurface; zero variance priced as the exact forward limit"),
        IntradayCapability(SnowballOption, _QUAD_V2 + "SnowballQuadEngineV2", "discrete", anywhere, price_only, "supported",
                           "exact Gaussian interval moments; a t=0 event under 'before' is decided at spot"),
        IntradayCapability(SnowballOption, _QUAD_V2 + "SnowballQuadEngineV2", "continuous", anywhere, price_only,
                           "unsupported", _CONTINUOUS_QUAD_NOTE),
        IntradayCapability(PhoenixOption, _QUAD_V2 + "PhoenixQuadEngineV2", "discrete", anywhere, price_only, "supported",
                           "realized coupons replay at their contractual amount; memory outstanding at the instant "
                           "needs equal periods (it reaches the twin as a count)"),
        IntradayCapability(PhoenixOption, _QUAD_V2 + "PhoenixQuadEngineV2", "continuous", anywhere, price_only,
                           "unsupported", _CONTINUOUS_QUAD_NOTE),
        IntradayCapability(KnockOutResetSnowballOption, _QUAD_V2 + "KOResetSnowballQuadEngineV2", "discrete", anywhere,
                           price_only, "supported",
                           "absolute post-KI schedules; due fixings must be covered by the checkpoint (no tracker replay)"),
        IntradayCapability(KnockOutResetSnowballOption, _QUAD_V2 + "KOResetSnowballQuadEngineV2", "continuous", anywhere,
                           price_only, "unsupported", _CONTINUOUS_QUAD_NOTE),
    )


#: Rows every Gate C price cell of which passed at the horizon and above (packaged evidence gate_c_results.json,
#: profiles uniform / desk / sessions_only, eleven spot offsets per barrier; test/intraday/test_capability_evidence.py
#: cross-checks). (product, engine, monitoring) -> (horizon, evidence scope appended to the note)
_QUALIFIED = {
    ("SnowballOption", "SnowballQuadEngineV2", "discrete"):
        (timedelta(seconds=1), "Gate C qualified to 1 s before a fixing around the KO and KI barriers"),
    ("CashOrNothingDigitalOption", "DigitalOptionAnalyticalEngine", "terminal"):
        (timedelta(seconds=1), "Gate C qualified to 1 s before expiry around the strike"),
    ("BarrierOption", "BarrierAnalyticalEngine", "continuous"):
        (timedelta(seconds=1), "Gate C qualified to 1 s before expiry on zero-carry up-and-out calls"),
    ("OneTouchOption", "OneTouchAnalyticalEngine", "continuous"):
        (timedelta(seconds=1), "Gate C qualified to 1 s before expiry on zero-carry up one-touches"),
}


def _publish(rows):
    out = []
    for row in rows:
        key = (row.product_type.__name__, row.engine_class_path.rsplit(".", 1)[-1], row.monitoring)
        if key in _QUALIFIED:
            horizon, scope = _QUALIFIED[key]
            row = replace(row, status="qualified", qualified_horizon=horizon, note=f"{row.note}; {scope}")
        out.append(row)
    return tuple(out)


INTRADAY_CAPABILITIES: Tuple[IntradayCapability, ...] = _publish(_rows())


def find_capability(product, engine, *, monitoring: str) -> Optional[IntradayCapability]:
    path = engine_class_path(engine)
    for row in INTRADAY_CAPABILITIES:
        if type(product) is row.product_type and row.engine_class_path == path and row.monitoring == monitoring:
            return row
    return None


def _alternatives(product, monitoring):
    return sorted({r.engine_class_path.rsplit(".", 1)[-1] for r in INTRADAY_CAPABILITIES
                   if type(product) is r.product_type and r.monitoring == monitoring and r.status != "unsupported"})


def require_capability(product, engine, *, monitoring: str, outputs: Iterable[str] = ("price",)) -> IntradayCapability:
    """The matrix row for this combination, or ``CapabilityError`` naming the limitation and alternatives."""
    row = find_capability(product, engine, monitoring=monitoring)
    name = engine_class_path(engine).rsplit(".", 1)[-1]
    if row is None or row.status == "unsupported":
        alts = _alternatives(product, monitoring)
        why = f" ({row.note})" if row is not None and row.note else " (not in the intraday inventory)"
        raise CapabilityError(
            f"{name} has no intraday route for {type(product).__name__} with {monitoring} monitoring{why}. "
            f"Intraday alternatives: {alts or 'none'}.")
    missing = sorted(set(outputs) - set(row.outputs))
    if missing:
        raise CapabilityError(f"{name} intraday route for {type(product).__name__} does not deliver {missing}; "
                              f"available outputs: {sorted(row.outputs)}")
    return row


PRICE_EVIDENCE_FILE = "gate_c_results.json"
GREEK_EVIDENCE_FILE = "gate_c_greeks.json"


def _evidence(name: str) -> dict:
    import json
    from importlib import resources

    resource = resources.files("quantark.intraday.evidence").joinpath(name)
    if not resource.is_file():
        return {}
    return json.loads(resource.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def capability_evidence() -> dict:
    """The packaged Gate C price evidence (``quantark/intraday/evidence/gate_c_results.json``); {} when absent."""
    return _evidence(PRICE_EVIDENCE_FILE)


@lru_cache(maxsize=None)
def greek_evidence() -> dict:
    """The packaged Gate C greek evidence (``quantark/intraday/evidence/gate_c_greeks.json``); {} when absent."""
    return _evidence(GREEK_EVIDENCE_FILE)


def _plain(value):
    """Tuples and lists compared alike, so a JSON round-trip of an identity still matches."""
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


#: Engine parameters that bound a resource or choose how a kernel is APPLIED, never what it computes: exceeding a
#: QUAD V2 memory/size cap raises, and its backend "changes application of a kernel, never its discretization"
#: (QuadV2Params). Every other field -- and every field of every other params class -- is part of the configuration
#: a certificate is bound to.
_APPLICATION_ONLY_PARAMS = {
    "QuadV2Params": frozenset({"backend", "max_nodes", "max_events", "max_states", "max_work_bytes", "max_cache_bytes"}),
}
#: Engine attributes set outside ``params`` that change the estimator (MC method and batching, bridge sampling).
_ESTIMATOR_ATTRIBUTES = ("method", "num_batches", "use_brownian_bridge")


def _canonical(value):
    """A JSON-stable value: enums by value, dataclasses and mappings as sorted dicts, sequences as lists."""
    if isinstance(value, Enum):
        return _canonical(value.value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _canonical(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def accuracy_settings(engine) -> dict:
    """Everything in ``engine`` that can move a number: the configuration a Gate C certificate is bound to.

    A demonstration earned at one quadrature, mesh or path count says nothing about another: QUAD V2 at
    ``cells_per_sd=0.1`` returns a point delta 30% off the reference with the same kernel-derivative evidence as
    the demonstrated ``cells_per_sd=2`` (review 2026-09-16 R4). So the certificate records the engine class, every
    params field except the resource/application ones above, the effective bump configuration, and the estimator
    attributes set outside ``params``; a request is covered only by an identical record.
    """
    params = getattr(engine, "params", None)
    out = {"engine": engine_class_path(engine)}
    if params is not None and dataclasses.is_dataclass(params):
        skip = _APPLICATION_ONLY_PARAMS.get(type(params).__name__, frozenset())
        out["params"] = {f.name: _canonical(getattr(params, f.name)) for f in dataclasses.fields(params)
                         if f.name not in skip}
        getter = getattr(params, "get_effective_bump_config", None)
        if getter is not None:
            out["bump_config"] = _canonical(getter())
    for name in _ESTIMATOR_ATTRIBUTES:
        if hasattr(engine, name):
            out[name] = _canonical(getattr(engine, name))
    return out


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    return {prefix[:-1]: value}


def _differences(requested: dict, demonstrated: dict, limit: int = 4) -> str:
    """The first few settings in which a request differs from a demonstrated configuration, for the reason text."""
    a, b = _flatten(_plain(requested) or {}), _flatten(_plain(demonstrated) or {})
    keys = sorted(k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>"))
    shown = ", ".join(f"{k}={a.get(k, '<absent>')!r} (demonstrated {b.get(k, '<absent>')!r})" for k in keys[:limit])
    return shown + (f", and {len(keys) - limit} more" if len(keys) > limit else "")


def economic_identity(ctx) -> str:
    """Tested conditional contract and market, excluding spot and valuation time.

    Historical observations enter through their resolved state. Keep every
    product term except observation schedules, keyed instead by their resolved
    FUTURE events (cash, barriers and absolute payment times included). A long
    first period can then qualify the same remaining claim just after a fixing,
    without certifying different payoffs, curves, event suffixes or KI states.
    """
    from quantark.execution.cache.fingerprint import fingerprint
    from quantark.intraday.context import value_tree
    from quantark.param.vol import TradingClockVolSurface

    product = ctx.request.product
    terms = {f.name: getattr(product, f.name) for f in dataclasses.fields(product)}
    barriers = terms.get("barrier_config")
    if barriers is not None and dataclasses.is_dataclass(barriers):
        terms["barrier_config"] = {f.name: getattr(barriers, f.name) for f in dataclasses.fields(barriers)
            if f.name not in ("ko_observation_schedule", "ki_observation_schedule",
                              "ko_observation_dates", "ki_observation_dates")}
    events = [(e.kind.value, e.timestamp, e.payment_timestamp, e.barrier, e.cash, e.regime)
              for e in ctx.numerical.remaining_events]
    # Bump contexts intentionally retain their original request. Match the
    # market actually delivered to the engine, unwrapping only the known clock.
    env = ctx.pricing_env
    surface = env.vol_surface.inner if type(env.vol_surface) is TradingClockVolSurface else env.vol_surface
    state = ctx.numerical.lifecycle_state
    ledger = () if state is None else tuple((cf.amount, cf.payment_time) for cf in state.ledger.cashflows
                                           if cf.payment_time > 0.0)
    return fingerprint(value_tree((terms, events, ctx.numerical.knocked_in, ledger,
                                   env.rate_curve, env.div_yield, surface, env.basis_yield,
                                   ctx.request.session_calendar.identity())))


def output_qualification_gap(product_name: str, route_name: str, measure: str, seconds_to_event: float, *,
                             monitoring: str, profile_identity, settings: dict,
                             measure_settings: Optional[dict] = None, economics: Optional[str] = None) -> str:
    """"" when Gate C demonstrated this exact combination here; otherwise why it did not.

    A certificate covers only the configurations its cells actually ran. The key is
    therefore the whole tested family — product, route, measure, MONITORING, the
    exact variance PROFILE, the engine's ACCURACY SETTINGS (``accuracy_settings``)
    and, for a measure with its own knob such as a desk theta's step, those
    MEASURE SETTINGS — inside the horizon WINDOW the ladder swept. Outside any of
    those the evidence is silent, and silence is not a pass: no evidence file, no
    matching row, or a horizon the ladder never reached all fail closed.

    Numerical routes also match ``economic_identity``: conditional future
    contract, lifecycle state, settlement, market curves/levels and calendar.
    No generalisation to untested economics is inferred from mesh density.
    The context-level caller additionally checks the demonstrated spot envelope.
    """
    rows = [r for r in greek_evidence().get("demonstrated", ())
            if (r["product"], r["route"], r["measure"]) == (product_name, route_name, measure)]
    if not rows:
        return f"Gate C demonstrated no {measure} for {product_name} on {route_name}"
    rows_here = [r for r in rows if r.get("monitoring") == monitoring]
    if not rows_here:
        return (f"Gate C demonstrated {measure} for {product_name} on {route_name} only under "
                f"{sorted({r.get('monitoring') for r in rows})} monitoring, not {monitoring!r}")
    wanted = _plain(profile_identity)
    rows_profile = [r for r in rows_here if _plain(r.get("profile_identity")) == wanted]
    if not rows_profile:
        return (f"Gate C demonstrated {measure} for {product_name} on {route_name} only for the variance profiles "
                f"{sorted({r.get('profile') for r in rows_here})}, not this request's "
                f"{profile_identity[0] if profile_identity else 'profile'!r}")
    rows_settings = [r for r in rows_profile if _plain(r.get("settings")) == _plain(settings)]
    if not rows_settings:
        return (f"Gate C demonstrated {measure} for {product_name} on {route_name} only at other engine settings: "
                f"{_differences(settings, rows_profile[0].get('settings'))}")
    rows_measure = [r for r in rows_settings if _plain(r.get("measure_settings") or {}) == _plain(measure_settings or {})]
    if not rows_measure:
        return (f"Gate C demonstrated {measure} for {product_name} on {route_name} only with "
                f"{_differences(measure_settings or {}, rows_settings[0].get('measure_settings') or {})}")
    if route_name in ("QuadV2Route", "PDERoute", "MCRoute"):
        rows_measure = [r for r in rows_measure if economics and r.get("economic_identity") == economics]
        if not rows_measure:
            return (f"Gate C demonstrated {measure} only for other conditional contract/market economics "
                    "(payoff, future events, state, settlement, curve families and levels)")
    windows = [(float(r["horizon_s"]), float(r["horizon_max_s"])) for r in rows_measure]
    if any(lo <= seconds_to_event <= hi for lo, hi in windows):
        return ""
    return (f"Gate C swept {measure} for {product_name} on {route_name} over "
            f"{[(lo, hi) for lo, hi in sorted(windows)]} s to the next event; this request is "
            f"{seconds_to_event:g} s away, outside every tested window")


def point_output_qualified(product_name: str, route_name: str, measure: str, seconds_to_event: float, *,
                           monitoring: str, profile_identity, settings: dict, economics: Optional[str] = None) -> bool:
    """Whether Gate C demonstrated point ``measure`` for this exact configuration (see the gap function)."""
    return not output_qualification_gap(product_name, route_name, f"point_{measure}", seconds_to_event,
                                        monitoring=monitoring, profile_identity=profile_identity, settings=settings,
                                        economics=economics)


def _horizon_label(horizon: Optional[timedelta]) -> str:
    if horizon is None:
        return "—"
    seconds = int(horizon.total_seconds())
    for unit, size in (("d", 86400), ("h", 3600), ("min", 60)):
        if seconds % size == 0 and seconds >= size:
            return f"{seconds // size} {unit}"
    return f"{seconds} s"


def render_capability_matrix() -> str:
    lines = ["| Product | Engine | Monitoring | Profiles | Outputs | Status | Qualified horizon | Note |",
             "|---|---|---|---|---|---|---|---|"]
    for r in INTRADAY_CAPABILITIES:
        lines.append(f"| {r.product_type.__name__} | {r.engine_class_path.rsplit('.', 1)[-1]} | {r.monitoring} | "
                     f"{', '.join(sorted(r.profiles))} | {', '.join(sorted(r.outputs))} | {r.status} | "
                     f"{_horizon_label(r.qualified_horizon)} | {r.note} |")
    return "\n".join(lines) + "\n"
