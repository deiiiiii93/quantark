"""Shared construction for intraday certification studies.

An intraday case is a (market, contract, context) triple: the flat market the
day-level studies already declare, the daily-KI snowball, and the intraday
context -- valuation instant, event phase, variance profile, session calendar,
confirmed history and lifecycle checkpoint. Every arm of a study resolves the
same request through the runtime's own ``resolve_context`` and prices through
the ordinary APIs. Nothing here bypasses the runtime or licenses anything.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import quantark
from quantark.asset.equity.lifecycle.cashflows import ValuationPoint
from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.intraday.context import resolve_context
from quantark.intraday.events import EventKind
from quantark.intraday.fixings import Fixing
from quantark.intraday.profile import VarianceProfile
from quantark.intraday.request import IntradayValuationRequest
from quantark.intraday.session import TradingSession, TradingSessionCalendar
from quantark.intraday.timestamp import to_utc
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.enum.option_enums import ObservationType
from quantark.util.exceptions import ValidationError
from quantark.validation.cell_identity import source_projection_sha256

from quantark.modelvalidation.registry import register_builder

SHANGHAI = timezone(timedelta(hours=8))

_PRODUCT_KEYS = ("initial_date", "months", "initial_price", "strike", "ko_barrier", "ki_barrier", "ko_rate",
                 "rebate_rate", "contract_multiplier", "ki_observation", "settlement_lag_days")
_KI_OBSERVATION = ("daily_close", "ko_dates")
_CONTEXT_KEYS = ("valuation", "phase", "profile", "calendar", "history_level", "fixings", "unfixed_from", "checkpoint")
_PROFILES = ("desk", "uniform", "sessions_only")
_CALENDARS = ("SSE",)
_PHASES = ("before", "after")

#: Everything whose source or data decides an intraday number, as repository-relative files or whole
#: trees. A class's own module is not enough: the PDE gamma readout lives in ``base_pde_solver.py``, the
#: quadrature kernels in helper modules, the holidays in CSV files. Their bytes enter every arm's identity,
#: so an edit invalidates checkpoints, anchors and amendment carry-forward instead of reusing them.
COMMON_TREES: Tuple[str, ...] = (
    "quantark/intraday",
    "quantark/asset/equity/product/option",
    "quantark/asset/equity/lifecycle",
    "quantark/asset/equity/settlement.py",
    "quantark/asset/equity/riskmeasures/greeks",
    "quantark/asset/equity/engine/settlement_support.py",
    "quantark/param",
    "quantark/priceenv",
    "quantark/util/calendar",
    "quantark/util/numerical",
    "quantark/modelvalidation/builders/intraday_common.py",
)
QUAD_V2_TREES: Tuple[str, ...] = ("quantark/asset/equity/engine/quad/v2", "quantark/asset/equity/param")
PDE_TREES: Tuple[str, ...] = ("quantark/asset/equity/engine/pde", "quantark/asset/equity/param")
MC_TREES: Tuple[str, ...] = ("quantark/asset/equity/engine/mc", "quantark/montecarlo", "quantark/asset/equity/param")
_FINGERPRINT_SUFFIXES = (".py", ".csv")


def _as_datetime(value: Any) -> datetime:
    """A naive contract date from a YAML date, datetime or ISO string."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    return datetime.fromisoformat(str(value)).replace(tzinfo=None)


@register_builder("equity.snowball.intraday", kind="product")
def build_intraday_snowball_spec(params: Mapping[str, Any]) -> dict:
    """Validate the daily-KI snowball spec (the 2026-09-17 fixture's terms, declared)."""
    spec = dict(params)
    unknown = set(spec) - set(_PRODUCT_KEYS)
    if unknown:
        raise ValidationError(
            f"Unknown equity.snowball.intraday product keys: {sorted(unknown)}; expected a subset of {_PRODUCT_KEYS}"
        )
    for key in ("initial_date", "months", "initial_price", "strike", "ko_barrier", "ki_barrier"):
        if key not in spec:
            raise ValidationError(f"equity.snowball.intraday product is missing {key!r}")
    _as_datetime(spec["initial_date"])
    if int(spec["months"]) <= 0:
        raise ValidationError(f"months must be positive, got {spec['months']}")
    observation = str(spec.get("ki_observation", "daily_close"))
    if observation not in _KI_OBSERVATION:
        raise ValidationError(f"ki_observation must be one of {_KI_OBSERVATION}, got {observation!r}")
    if int(spec.get("settlement_lag_days", 0)) < 0:
        raise ValidationError("settlement_lag_days must be non-negative")
    return spec


@register_builder("intraday.sse", kind="context")
def build_intraday_context_spec(params: Mapping[str, Any]) -> dict:
    """Validate an intraday context spec: instant, phase, clock, history and checkpoint."""
    spec = dict(params)
    unknown = set(spec) - set(_CONTEXT_KEYS)
    if unknown:
        raise ValidationError(f"Unknown intraday context keys: {sorted(unknown)}; expected a subset of {_CONTEXT_KEYS}")
    valuation_timestamp(spec)
    if str(spec.get("phase", "before")) not in _PHASES:
        raise ValidationError(f"context.phase must be one of {_PHASES}, got {spec.get('phase')!r}")
    if str(spec.get("profile", "desk")) not in _PROFILES:
        raise ValidationError(f"context.profile must be one of {_PROFILES}, got {spec.get('profile')!r}")
    if str(spec.get("calendar", "SSE")) not in _CALENDARS:
        raise ValidationError(f"context.calendar must be one of {_CALENDARS}, got {spec.get('calendar')!r}")
    for fixing in spec.get("fixings", ()):
        if not {"date", "level"} <= set(fixing):
            raise ValidationError("each context.fixings entry needs date and level")
        _as_datetime(fixing["date"])
    checkpoint_spec = spec.get("checkpoint")
    if checkpoint_spec is not None:
        if "as_of" not in checkpoint_spec:
            raise ValidationError("context.checkpoint needs as_of")
        _as_datetime(checkpoint_spec["as_of"])
    return spec


def valuation_timestamp(spec: Mapping[str, Any]) -> datetime:
    raw = spec.get("valuation")
    if raw is None:
        raise ValidationError("context.valuation is required")
    ts = raw if isinstance(raw, datetime) else datetime.fromisoformat(str(raw))
    if ts.tzinfo is None:
        raise ValidationError("context.valuation must carry a UTC offset, e.g. 2026-09-10T14:00:00+08:00")
    return ts


@lru_cache(maxsize=None)
def exchange_calendar(name: str = "SSE"):
    if name not in _CALENDARS:
        raise ValidationError(f"unknown calendar {name!r}")
    return create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))


@lru_cache(maxsize=None)
def session_calendar(name: str = "SSE") -> TradingSessionCalendar:
    return TradingSessionCalendar(
        name=name, tz=SHANGHAI, calendar=exchange_calendar(name),
        sessions=(TradingSession(time(9, 30), time(11, 30)), TradingSession(time(13, 0), time(15, 0))),
    )


def variance_profile(name: str, calendar: TradingSessionCalendar) -> VarianceProfile:
    if name == "desk":
        return VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))
    if name == "uniform":
        return VarianceProfile.uniform(calendar, 244, reference_date=date(2026, 9, 15))
    if name == "sessions_only":
        return VarianceProfile.sessions_only(calendar, 244)
    raise ValidationError(f"unknown variance profile {name!r}")


def _monthly_trading_dates(cal, t0: datetime, months: int) -> list:
    out = []
    for k in range(1, months + 1):
        month = (t0.month - 1 + k) % 12 + 1
        year = t0.year + (t0.month - 1 + k) // 12
        d = datetime(year, month, min(t0.day, 28))
        while not cal.is_business_day(d):
            d += timedelta(days=1)
        out.append(d)
    return out


def make_snowball(spec: Mapping[str, Any]) -> SnowballOption:
    """The daily-KI snowball: monthly KO dates rolled to SSE business days, KI at every close."""
    cal = exchange_calendar("SSE")
    t0 = _as_datetime(spec["initial_date"])
    dates = _monthly_trading_dates(cal, t0, int(spec["months"]))
    ko, ki = float(spec["ko_barrier"]), float(spec["ki_barrier"])
    lag = int(spec.get("settlement_lag_days", 0))

    def ko_record(d: datetime) -> ObservationRecord:
        kwargs = {"observation_date": d, "barrier": ko}
        if lag > 0:
            kwargs["settlement_date"] = d + timedelta(days=lag)
        return ObservationRecord(**kwargs)

    if str(spec.get("ki_observation", "daily_close")) == "daily_close":
        maturity = dates[-1]
        ki_dates = [d for d in (t0 + timedelta(days=k) for k in range(1, (maturity - t0).days + 1))
                    if cal.is_business_day(d)]
    else:
        ki_dates = dates
    ko_rate = float(spec.get("ko_rate", 0.12))
    return SnowballOption(
        initial_price=float(spec["initial_price"]), strike=float(spec["strike"]),
        contract_multiplier=float(spec.get("contract_multiplier", 1.0)), initial_date=t0, exercise_date=dates[-1],
        barrier_config=BarrierConfig(
            ko_barrier=ko, ko_rate=ko_rate, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=ObservationSchedule(records=[ko_record(d) for d in dates]),
            ki_barrier=ki, ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=ObservationSchedule(
                records=[ObservationRecord(observation_date=d, barrier=ki) for d in ki_dates]),
        ),
        payoff_config=PayoffConfig(rebate_rate=float(spec.get("rebate_rate", ko_rate)), include_principal=False),
    )


def make_environment(env_spec: Mapping[str, Any], ts: datetime, spot: Optional[float] = None) -> PricingEnvironment:
    """A flat market quoted at the valuation instant (aware), optionally at another spot."""
    return PricingEnvironment(
        rate_curve=FlatRateCurve(float(env_spec["rate"])), valuation_date=ts,
        spot_quote=SpotQuote(float(env_spec["spot"] if spot is None else spot), timestamp=ts),
        vol_surface=FlatVolSurface(float(env_spec["vol"])),
        div_yield=ContinuousDividendYield(float(env_spec["div_yield"])),
    )


def bumped_env(env: PricingEnvironment, factor: float) -> PricingEnvironment:
    bumped = deepcopy(env)
    bumped.spot_quote.spot *= factor
    return bumped


def _checkpoint_day(context_spec: Mapping[str, Any]) -> Optional[date]:
    cp = context_spec.get("checkpoint")
    return None if cp is None else _as_datetime(cp["as_of"]).date()


def checkpoint(context_spec: Mapping[str, Any]) -> Optional[AutocallableLifecycleState]:
    """A date-only checkpoint: 'after that day's close', carrying the declared KI flag."""
    cp = context_spec.get("checkpoint")
    if cp is None:
        return None
    ki_date = cp.get("ki_date")
    return AutocallableLifecycleState(
        knocked_in=bool(cp.get("knocked_in", False)),
        ki_date=None if ki_date is None else _as_datetime(ki_date),
        valuation_point=ValuationPoint(date=_as_datetime(cp["as_of"])),
    )


def confirmed_history(product, calendar: TradingSessionCalendar, profile: VarianceProfile,
                      env_spec: Mapping[str, Any], context_spec: Mapping[str, Any], ts: datetime) -> Tuple[Fixing, ...]:
    """Every contractual event before ``ts`` fixed at ``history_level`` (overrides by date).

    A checkpoint covers the events up to its day; ``unfixed_from`` leaves the events from that date on
    without a fixing, which is how a study declares a due-but-missing observation; a knock-out fixing ends
    the history, because a terminated claim has no later observations.
    """
    level = context_spec.get("history_level")
    overrides = {_as_datetime(f["date"]).date(): float(f["level"]) for f in context_spec.get("fixings", ())}
    if level is None and not overrides:
        return ()
    probe = resolve_context(IntradayValuationRequest(
        product=product, pricing_env=make_environment(env_spec, ts), session_calendar=calendar, variance_profile=profile))
    covered_until = _checkpoint_day(context_spec)
    unfixed_from = None if context_spec.get("unfixed_from") is None else _as_datetime(context_spec["unfixed_from"]).date()
    events_at: dict = {}
    for event in probe.timeline.events:
        events_at.setdefault(to_utc(event.timestamp), []).append(event)
    out = []
    for key in sorted(events_at):
        when = events_at[key][0].timestamp
        if key >= to_utc(ts):
            break
        if unfixed_from is not None and when.date() >= unfixed_from:
            break                   # due and deliberately missing: the runtime resolves it provisionally
        if covered_until is not None and when.date() <= covered_until:
            continue
        value = overrides.get(when.date(), level)
        if value is None:
            continue
        out.append(Fixing(when, float(value)))
        if any(e.kind is EventKind.KO and e.barrier is not None and float(value) >= float(e.barrier)
               for e in events_at[key]):
            break   # knocked out: the claim has terminated, and later events carry no fixings
    return tuple(out)


def build_request(env_spec: Mapping[str, Any], product_spec: Mapping[str, Any], context_spec: Mapping[str, Any], *,
                  greeks: Sequence[str] = (), greek_convention: Optional[str] = None,
                  theta_unit: str = "hour") -> IntradayValuationRequest:
    """The runtime request every arm of an intraday study prices."""
    calendar = session_calendar(str(context_spec.get("calendar", "SSE")))
    profile = variance_profile(str(context_spec.get("profile", "desk")), calendar)
    ts = valuation_timestamp(context_spec)
    product = make_snowball(product_spec)
    return IntradayValuationRequest(
        product=product, pricing_env=make_environment(env_spec, ts), session_calendar=calendar,
        variance_profile=profile, lifecycle_state=checkpoint(context_spec),
        fixings=confirmed_history(product, calendar, profile, env_spec, context_spec, ts),
        event_phase=str(context_spec.get("phase", "before")), greeks=tuple(greeks),
        greek_convention=greek_convention, theta_unit=theta_unit,
    )


class IntradayArm:
    """Shared spec handling for every arm of an intraday study."""

    def __init__(self, environment_params: Mapping[str, Any], product_params: Mapping[str, Any],
                 context_params: Mapping[str, Any], quantities: Sequence[str], params: Mapping[str, Any]) -> None:
        self.environment_params = dict(environment_params)
        self.product_params = dict(product_params)
        self.context_params = dict(context_params)
        self.quantities = tuple(quantities)
        self._params = dict(params)

    def specs(self, case) -> Tuple[dict, dict, dict]:
        environment = {**self.environment_params, **case.environment_params}
        product = {**self.product_params, **case.product_params}
        context = {**self.context_params, **case.context_params}
        # Case overrides never reach the study-level validators; validate the merged specs here
        # or a typo builds a different case and certifies it under the wrong name.
        build_intraday_snowball_spec(product)
        build_intraday_context_spec(context)
        return environment, product, context

    def resolved_inputs(self, case) -> dict:
        """The merged study-level and case inputs this arm prices, JSON-stable: what a schema-2 identity is over."""
        environment, product, context = self.specs(case)
        return {"environment": plain(environment), "product": plain(product), "context": plain(context),
                "quantities": list(self.quantities)}


def plain(value: Any) -> Any:
    """JSON-stable spec values: YAML dates become ISO strings, containers recurse."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def _repo_root() -> Path:
    return Path(quantark.__file__).resolve().parent.parent


def fingerprint_inputs(*trees: str, root: Optional[Path] = None) -> List[Path]:
    """Every source and data file under the given repository-relative files and trees, sorted.

    Raises:
        ValidationError: a declared path does not exist. A silently empty tree would hash to a
            constant and protect nothing.
    """
    base = Path(root) if root is not None else _repo_root()
    found = set()
    for tree in trees:
        path = base / tree
        if path.is_file():
            found.add(path)
        elif path.is_dir():
            found.update(p for p in path.rglob("*") if p.is_file() and p.suffix in _FINGERPRINT_SUFFIXES
                         and "__pycache__" not in p.parts)
        else:
            raise ValidationError(f"implementation tree {tree!r} does not exist under {base}")
    return sorted(found)


def implementation_fingerprint(*trees: str, root: Optional[Path] = None) -> str:
    """SHA-256 over the raw bytes and repository-relative paths of every file in ``trees``."""
    base = Path(root) if root is not None else _repo_root()
    return source_projection_sha256([(path, ()) for path in fingerprint_inputs(*trees, root=base)], root=base)
