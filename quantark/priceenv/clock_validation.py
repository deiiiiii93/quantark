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
construction and ``re_anchor_declared_curves`` keeps that map on the
environment's valuation date when it is reassigned.
"""
from quantark.priceenv.curve_quoting import CurveQuotingBasis
from quantark.util.calendar import DayCountConvention
from quantark.util.exceptions import ValidationError

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


def _rewrap_if_moved(env, field: str, obj, env_map: _EnvMap) -> None:
    """A CALENDAR_YEAR wrapper sits on the env's map: same anchor, same horizon; otherwise its inner moves there."""
    _check_clock(field, obj.clock, env)
    tm = obj.time_map
    if tm.anchor_date == env.valuation_date and tm.horizon_date == env.trading_clock_horizon:
        return
    setattr(env, field, obj.with_time_map(env_map.get(field)))


def resolve_cash_curve_quoting(env) -> None:
    """Apply the cash-curve quoting declarations at construction (0.15.0 spec D1-D9, D14, D15).

    On ``BUSINESS_DAYS``: an undeclared native curve must be exactly zero; ``TRADING_YEAR`` keeps a native curve
    and refuses a wrapper; ``CALENDAR_YEAR`` wraps a native curve on the environment's map and moves an existing
    wrapper onto it. On any other convention ``CALENDAR_YEAR`` reads the calendar-quoted curve as is (a wrapper
    is unwrapped to its inner) and ``TRADING_YEAR`` is refused.
    """
    business = env.day_count_convention == DayCountConvention.BUSINESS_DAYS
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
            continue
        if quoting is None:
            if wrapped or _is_identically_zero(obj):
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
            _rewrap_if_moved(env, field, obj, env_map)
        else:
            setattr(env, field, _wrapper_for(field)(obj, env_map.get(field)))


def re_anchor_declared_curves(env) -> None:
    """Keep every CALENDAR_YEAR wrapper on the environment's valuation date after it is reassigned (spec D13).

    Only re-anchors: an undeclared or TRADING_YEAR curve is never touched and nothing is refused here, so a
    bumped copy that carries a curve assigned after construction keeps working.
    """
    if env.day_count_convention != DayCountConvention.BUSINESS_DAYS:
        return
    env_map = _EnvMap(env)
    for field, qfield in CASH_CURVE_FIELDS:
        obj = getattr(env, field)
        if getattr(env, qfield) is CurveQuotingBasis.CALENDAR_YEAR and is_clock_wrapped(obj):
            _rewrap_if_moved(env, field, obj, env_map)
