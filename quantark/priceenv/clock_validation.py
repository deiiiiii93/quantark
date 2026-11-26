"""One trading clock per environment, and how its cash curves are quoted.

Spec: docs/superpowers/specs/2026-09-01-trading-clock-vol-design.md §4.3
(review iter-2 P2b): product times on the trading axis come from the env
resolver (day_count_convention / bus_days_in_year / calendar) while the
cash/vol wrappers carry a TradingClock — a 252-resolver paired with a
244-map would silently mis-date every query while the DF invariant breaks.

0.15.0: a native (unwrapped) cash curve on a ``BUSINESS_DAYS`` environment is
integrated over trading-year time, which is right only if it is quoted per
trading year. The environment records the quoting basis per curve
(``CurveQuotingBasis``) and owns one ``BusinessTimeMap`` for the curves it
wraps; ``resolve_cash_curve_quoting`` applies the declarations at
construction and ``re_anchor_clock_wrappers`` keeps every trading-clock
wrapper (cash curves and the vol surface) on the environment's valuation
date when it is reassigned.
"""
from contextlib import contextmanager

from quantark.priceenv.curve_quoting import CurveQuotingBasis
from quantark.util.calendar import DayCountConvention
from quantark.util.calendar.trading_clock import BusinessTimeMap
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close

#: (curve field, its quoting field) for every cash curve the environment can wrap.
CASH_CURVE_FIELDS = (
    ("rate_curve", "rate_curve_quoting"),
    ("div_yield", "div_yield_quoting"),
    ("basis_yield", "basis_yield_quoting"),
)


def validate_trading_clock_configuration(env) -> None:
    """Every wrapped curve's TradingClock must match the env resolver."""
    for name in ("rate_curve", "div_yield", "basis_yield", "vol_surface"):
        obj = getattr(env, name, None)
        clock = getattr(obj, "clock", None)
        if clock is None:
            # TradingClockVolSurface carries its clock on the time map, not on itself
            clock = getattr(getattr(obj, "time_map", None), "clock", None)
        if clock is None:
            continue
        _check_clock(name, clock, env)


def _check_clock(name, clock, env) -> None:
    if clock.days_per_year != env.bus_days_in_year:
        raise ValidationError(
            f"{name} carries a TradingClock with days_per_year="
            f"{clock.days_per_year} but the environment resolves times "
            f"with bus_days_in_year={env.bus_days_in_year}; one clock "
            "per configuration"
        )
    if env.calendar is not None and clock.calendar is not env.calendar:
        raise ValidationError(
            f"{name} carries a TradingClock whose calendar is not the "
            "environment's calendar object; one clock per configuration"
        )


def is_clock_wrapped(obj) -> bool:
    """True for a TradingClock wrapper (it has an ``inner`` and a ``time_map``)."""
    return obj is not None and getattr(obj, "time_map", None) is not None and hasattr(obj, "inner")


def _is_identically_zero(obj) -> bool:
    """None, or a curve that says it is exactly zero. A curve without the method is not zero."""
    if obj is None:
        return True
    probe = getattr(obj, "is_identically_zero", None)
    return bool(probe()) if callable(probe) else False


def _wrapper_for(field: str):
    if field == "rate_curve":
        from quantark.param.rrf.trading_clock_curve import TradingClockRateCurve
        return TradingClockRateCurve
    if field == "div_yield":
        from quantark.param.div.trading_clock_yield import TradingClockDividendYield
        return TradingClockDividendYield
    from quantark.param.basis.trading_clock_basis import TradingClockBasisYield
    return TradingClockBasisYield


class _EnvMap:
    """The environment's own BusinessTimeMap, built once and only if a curve needs it."""

    def __init__(self, env) -> None:
        self._env = env
        self._map = None

    def get(self, field: str):
        if self._map is None:
            env = self._env
            if env.calendar is None:
                raise ValidationError(
                    f"{field} is declared CALENDAR_YEAR on a BUSINESS_DAYS environment without a calendar; "
                    "the environment needs its calendar to build the trading clock it wraps the curve onto"
                )
            horizon = env.trading_clock_horizon
            if horizon is None:
                raise ValidationError(
                    f"{field} is declared CALENDAR_YEAR on a BUSINESS_DAYS environment; set "
                    "trading_clock_horizon (a date strictly after the last date priced, e.g. last date + 1 day) "
                    "so the environment can build the trading clock it wraps the curve onto"
                )
            if env.valuation_date >= horizon:
                raise ValidationError(
                    f"valuation_date {env.valuation_date} is not before trading_clock_horizon {horizon}; "
                    f"the trading clock {field} is wrapped onto must run past the valuation date"
                )
            from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
            self._map = BusinessTimeMap(TradingClock(env.calendar, env.bus_days_in_year), env.valuation_date,
                                        horizon)
        return self._map


def _on_env_map(env, field: str, obj, env_map: _EnvMap):
    """A CALENDAR_YEAR wrapper sits on the env's map: same anchor, same horizon; otherwise its inner moves there.

    Returns the object the field should hold (``obj`` itself when it already sits on the env's map).
    """
    _check_clock(field, obj.clock, env)
    tm = obj.time_map
    if tm.anchor_date == env.valuation_date and tm.horizon_date == env.trading_clock_horizon:
        return obj
    return obj.with_time_map(env_map.get(field))


def _on_own_horizon(env, field: str, obj):
    """An undeclared hand-wrapped curve or the vol wrapper sits at the valuation date on its own map's horizon
    (owner rulings D7, D19).

    Returns ``obj`` itself when its map is already anchored at the valuation date, else the same inner re-exposed on
    ``time_map.re_anchored(valuation_date)`` (same clock, same horizon: sticky trading tenor). Only a
    ``BusinessTimeMap`` (the calendar trading clock) is moved; an ``IntradayTimeMap`` is the intraday route's own
    clock and is left as it is.
    """
    tm = obj.time_map
    if not isinstance(tm, BusinessTimeMap):
        # an IntradayTimeMap belongs to the intraday route, which resolves its own clock per timestamp
        return obj
    if tm.anchor_date == env.valuation_date:
        return obj
    if env.valuation_date >= tm.horizon_date:
        advice = "rebuild the wrapper with a later horizon"
        if field != "vol_surface" and env.day_count_convention == DayCountConvention.BUSINESS_DAYS:
            advice += f", or pass the native curve with {field}_quoting=CALENDAR_YEAR and trading_clock_horizon"
        raise ValidationError(
            f"valuation_date {env.valuation_date} is not before the horizon {tm.horizon_date} of the trading clock "
            f"{field} is wrapped onto; {advice}"
        )
    return obj.with_time_map(tm.re_anchored(env.valuation_date))


def resolve_cash_curve_quoting(env) -> None:
    """Apply the cash-curve quoting declarations at construction (0.15.0 spec D1-D9, D14, D15).

    On ``BUSINESS_DAYS``: an undeclared native curve must be exactly zero; ``TRADING_YEAR`` keeps a native curve
    and refuses a wrapper; ``CALENDAR_YEAR`` wraps a native curve on the environment's map and moves an existing
    wrapper onto it. On any other convention ``CALENDAR_YEAR`` reads the calendar-quoted curve as is (a wrapper
    is unwrapped to its inner) and ``TRADING_YEAR`` is refused.
    """
    business = env.day_count_convention == DayCountConvention.BUSINESS_DAYS
    if business and env.calendar is None:
        # D14 at construction: a later valuation-date move would need the calendar to re-anchor
        for _field, qfield in CASH_CURVE_FIELDS:
            if getattr(env, qfield) is CurveQuotingBasis.CALENDAR_YEAR:
                raise ValidationError(
                    f"{qfield}=CALENDAR_YEAR on a BUSINESS_DAYS environment without a calendar; the environment "
                    "needs its calendar to build the trading clock it wraps calendar-quoted curves onto"
                )
    env_map = _EnvMap(env)
    for field, qfield in CASH_CURVE_FIELDS:
        obj, quoting = getattr(env, field), getattr(env, qfield)
        wrapped = is_clock_wrapped(obj)
        if not business:
            if quoting is CurveQuotingBasis.TRADING_YEAR:
                raise ValidationError(
                    f"{qfield}=TRADING_YEAR needs a BUSINESS_DAYS environment; on "
                    f"{env.day_count_convention.name} there is no trading clock to read a trading-year quote on"
                )
            if quoting is CurveQuotingBasis.CALENDAR_YEAR and wrapped:
                setattr(env, field, obj.inner)
            elif quoting is None and wrapped:
                setattr(env, field, _on_own_horizon(env, field, obj))
            continue
        if quoting is None:
            if wrapped:
                setattr(env, field, _on_own_horizon(env, field, obj))
                continue
            if _is_identically_zero(obj):
                continue
            raise ValidationError(
                f"{field} is a native curve on a BUSINESS_DAYS environment, so it would be integrated over "
                f"trading-year time. Declare how it is quoted: {qfield}=CurveQuotingBasis.CALENDAR_YEAR (per "
                "calendar year; the environment wraps it onto its trading clock and needs trading_clock_horizon) "
                f"or {qfield}=CurveQuotingBasis.TRADING_YEAR (per trading year; read as is)"
            )
        if quoting is CurveQuotingBasis.TRADING_YEAR:
            if wrapped:
                raise ValidationError(
                    f"{field} is a TradingClock wrapper (a calendar-quoted curve on the trading clock) but "
                    f"{qfield}=TRADING_YEAR says it is quoted per trading year; pass the native curve or "
                    "declare CALENDAR_YEAR"
                )
            continue
        # CALENDAR_YEAR on BUSINESS_DAYS
        if obj is None:
            continue
        if wrapped:
            setattr(env, field, _on_env_map(env, field, obj, env_map))
        else:
            setattr(env, field, _wrapper_for(field)(obj, env_map.get(field)))
    surface = env.vol_surface
    if is_clock_wrapped(surface):
        # owner ruling D19: the vol wrapper follows the valuation date like a hand-wrapped cash curve
        setattr(env, "vol_surface", _on_own_horizon(env, "vol_surface", surface))


def re_anchor_clock_wrappers(env) -> None:
    """Keep every trading-clock wrapper on the environment's valuation date after it is reassigned.

    A ``CALENDAR_YEAR`` wrapper on a ``BUSINESS_DAYS`` environment moves onto the environment's own map (spec D13);
    an undeclared hand-wrapped cash curve and the vol wrapper are re-anchored on their own horizon (owner rulings
    D7, D19), on any convention. Native
    curves are never touched and the declaration refusals of construction are not re-run, so a bumped copy that
    carries a curve assigned after construction keeps working. It raises only when a re-anchor cannot be built (the
    new date is not before the relevant horizon, or a wrapper's clock no longer matches the environment). Every new
    wrapper is built before any field is assigned, so a raise leaves the curve fields as they were.
    """
    business = env.day_count_convention == DayCountConvention.BUSINESS_DAYS
    env_map = _EnvMap(env)
    updates = {}
    for field, qfield in CASH_CURVE_FIELDS:
        obj = getattr(env, field)
        if not is_clock_wrapped(obj):
            continue
        quoting = getattr(env, qfield)
        if business and quoting is CurveQuotingBasis.CALENDAR_YEAR:
            updates[field] = _on_env_map(env, field, obj, env_map)
        elif quoting is None:
            updates[field] = _on_own_horizon(env, field, obj)
    if is_clock_wrapped(env.vol_surface):
        updates["vol_surface"] = _on_own_horizon(env, "vol_surface", env.vol_surface)
    for field, obj in updates.items():
        setattr(env, field, obj)


def relationship_tenor(env, mode, time_to_maturity: float) -> float:
    """The tenor a basis<->dividend shift reads its curves at, on the curves' own axis (0.16.0 trading-clock run D2).

    ``time_to_maturity`` is on the environment's axis. With no TradingClock-wrapped cash curve it is returned as is.
    Otherwise the shift runs on the calendar-quoted inners (``calendar_quoted_view``), so every read at the tenor --
    the source curve's current value and the relationship's rate / dividend / basis -- uses the calendar time the
    wrappers map it to. A relationship mode (not INDEPENDENT) also needs one clock for every involved curve: it is
    refused when a non-zero cash curve is native next to a wrapped one, when a slot it may derive into (``None`` or
    exactly zero) is declared ``TRADING_YEAR``, or when the wrappers map the tenor to different calendar times.
    """
    t = float(time_to_maturity)
    if env.day_count_convention != DayCountConvention.BUSINESS_DAYS:
        return t
    curves = [(f, getattr(env, f)) for f, _ in CASH_CURVE_FIELDS]
    wrapped = [(f, obj) for f, obj in curves if is_clock_wrapped(obj)]
    if not wrapped:
        return t
    name = getattr(mode, "name", str(mode)).upper()
    native = [f for f, obj in curves if not is_clock_wrapped(obj) and not _is_identically_zero(obj)]
    if name == "INDEPENDENT" and native:
        # a native (trading-year) curve may be the one read: keep 0.15.0's reading at T (checkpoint finding)
        return t
    if name != "INDEPENDENT":
        if native:
            raise ValidationError(
                f"basis/dividend relationship mode {name} is not supported on a partially wrapped environment: "
                f"{', '.join(native)} is a native non-zero curve next to TradingClock-wrapped "
                f"{', '.join(f for f, _ in wrapped)}; deriving one curve from the others would mix calendar- and "
                "trading-year quotes"
            )
        trading = [f for f, q in CASH_CURVE_FIELDS
                   if not is_clock_wrapped(getattr(env, f)) and getattr(env, q) is CurveQuotingBasis.TRADING_YEAR]
        if trading:
            raise ValidationError(
                f"basis/dividend relationship mode {name} could derive a calendar-quoted curve into "
                f"{', '.join(trading)}, declared TRADING_YEAR; declare CALENDAR_YEAR or use INDEPENDENT"
            )
    times = [float(obj.to_inner_time(t)) for _, obj in wrapped]
    if any(not is_close(times[0], x) for x in times[1:]):
        raise ValidationError(
            f"basis/dividend relationship mode {name}: the wrapped cash curves map the tenor {t} to different "
            f"calendar times {times}; they are not on one clock"
        )
    return times[0]


def refuse_cross_clock_relationship(env, mode) -> None:
    """Refuse a basis<->dividend relationship mode on an environment holding a TradingClock-wrapped cash curve.

    A relationship derives one cash curve from the others at one ``time_to_maturity``. Across a trading-clock
    wrapper that needs the tenor mapped onto the calendar axis and the derived curve placed on the same clock as its
    sources; 0.15.0 does not build that (spec D16, CP2), so it refuses rather than mix units.
    TODO: map time_to_maturity through time_map.to_calendar and wrap the derived curve on the sources' map.
    """
    if getattr(mode, "name", str(mode)).upper() == "INDEPENDENT":
        return
    wrapped = [f for f, _ in CASH_CURVE_FIELDS if is_clock_wrapped(getattr(env, f))]
    if wrapped:
        raise ValidationError(
            f"basis/dividend relationship mode {getattr(mode, 'name', mode)} is not supported on trading-clock-"
            f"wrapped cash curves ({', '.join(wrapped)}): deriving one curve from the others would mix calendar- "
            "and trading-year quotes; use INDEPENDENT"
        )


def apply_declared_quoting(env, field: str) -> None:
    """Re-apply the environment's declaration to a cash-curve field assigned after construction (spec D16).

    Only ``CALENDAR_YEAR`` on ``BUSINESS_DAYS`` acts: a native curve is wrapped on the environment's map and a
    wrapper is moved onto it. Any other combination leaves the field exactly as assigned.
    """
    if env.day_count_convention != DayCountConvention.BUSINESS_DAYS:
        return
    if getattr(env, dict(CASH_CURVE_FIELDS)[field]) is not CurveQuotingBasis.CALENDAR_YEAR:
        return
    obj = getattr(env, field)
    if obj is None:
        return
    env_map = _EnvMap(env)
    if is_clock_wrapped(obj):
        setattr(env, field, _on_env_map(env, field, obj, env_map))
    else:
        setattr(env, field, _wrapper_for(field)(obj, env_map.get(field)))


def _holds_clock_wrapper(curve, depth: int = 3) -> bool:
    """True when ``curve`` is, or is built over, a TradingClock wrapper: searched through its attributes
    (``__dict__`` and ``__slots__``) and the items of lists, tuples and dicts it holds, ``depth`` levels deep."""
    if is_clock_wrapped(curve):
        return True
    if depth <= 0 or curve is None or isinstance(curve, (str, bytes, int, float, bool, type)):
        return False
    if isinstance(curve, dict):
        children = list(curve.values())
    elif isinstance(curve, (list, tuple)):
        children = list(curve)
    else:
        children = list(getattr(curve, "__dict__", {}).values())
        for slot in getattr(type(curve), "__slots__", ()) or ():
            if isinstance(slot, str) and hasattr(curve, slot):
                children.append(getattr(curve, slot))
    return any(_holds_clock_wrapper(v, depth - 1) for v in children)


def declared_time_map(env, field: str):
    """The environment's own map when ``field`` is declared ``CALENDAR_YEAR`` on a ``BUSINESS_DAYS`` environment
    (a curve created for that field is a calendar quote and lives on it), else ``None``."""
    if env.day_count_convention != DayCountConvention.BUSINESS_DAYS:
        return None
    if getattr(env, dict(CASH_CURVE_FIELDS)[field]) is not CurveQuotingBasis.CALENDAR_YEAR:
        return None
    return _EnvMap(env).get(field)


def place_created_cash_curve(env, field: str, curve) -> None:
    """Assign a cash curve a bump CREATED from nothing (the field was ``None``) per the environment's declaration.

    The created curve is a calendar-year quote like every other cash curve: on a ``BUSINESS_DAYS`` environment whose
    field is declared ``CALENDAR_YEAR`` it is wrapped on the environment's map (``apply_declared_quoting``); anywhere
    else it is assigned as is. This is NOT a general assignment hook (0.15.0 D13): a curve built over a wrapper would
    be wrapped twice, so one is refused.
    """
    if curve is not None and _holds_clock_wrapper(curve):
        raise ValidationError(
            f"cannot place a created {field} built over a TradingClock wrapper: it would be wrapped twice"
        )
    setattr(env, field, curve)
    if curve is not None:
        apply_declared_quoting(env, field)


@contextmanager
def calendar_quoted_view(env):
    """Run a shift on the calendar-quoted curves behind the environment's trading-clock wrappers (spec D16).

    Inside the block every clock-wrapped cash curve (rate, dividend, basis) holds its calendar-quoted ``inner``, so a
    shift written for native curves moves the curve as quoted (per calendar year, like ``parallel_shifted``) and
    cross-curve reads (basis from rate and dividend) see one consistent clock. On exit a wrapped field is re-exposed
    on the map it came from: untouched it gets its very wrapper object back; a new curve is wrapped as
    ``type(wrapper)(new, wrapper.time_map)``; None stays None. A field that was not wrapped keeps what the block
    assigned, except that a curve assigned to a ``CALENDAR_YEAR`` field of a ``BUSINESS_DAYS`` environment is
    wrapped (``apply_declared_quoting``). If the block or the re-wrapping on exit raises, every field gets its
    original object back.
    """
    names = [f for f, _ in CASH_CURVE_FIELDS]
    originals = {f: getattr(env, f) for f in names}
    # the clock the block's calendar quotes came from: a curve it creates in an empty slot is re-exposed on it
    common_map = next((obj.time_map for obj in originals.values() if is_clock_wrapped(obj)), None)
    for f, obj in originals.items():
        if is_clock_wrapped(obj):
            setattr(env, f, obj.inner)
    try:
        yield env
        exits, declared = {}, []
        quoting = dict(CASH_CURVE_FIELDS)
        for f, obj in originals.items():
            new = getattr(env, f)
            if is_clock_wrapped(obj):
                if new is obj.inner:
                    exits[f] = obj
                elif new is not None and not is_clock_wrapped(new):
                    exits[f] = type(obj)(new, obj.time_map)
            elif new is not obj:
                if (common_map is not None and new is not None and not is_clock_wrapped(new)
                        and env.day_count_convention == DayCountConvention.BUSINESS_DAYS
                        and _is_identically_zero(obj) and getattr(env, quoting[f]) is None):
                    # an empty (None / exactly zero) undeclared slot filled from calendar-quoted inners holds a
                    # calendar quote too (0.16.0 trading-clock run D2): on the sources' clock, never native
                    exits[f] = _wrapper_for(f)(new, common_map)
                else:
                    declared.append(f)
        for f, obj in exits.items():
            setattr(env, f, obj)
        for f in declared:
            apply_declared_quoting(env, f)
    except BaseException:
        for f, obj in originals.items():
            setattr(env, f, obj)
        raise
