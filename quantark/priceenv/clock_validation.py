"""One trading clock per environment.

Spec: docs/superpowers/specs/2026-09-01-trading-clock-vol-design.md §4.3
(review iter-2 P2b): product times on the trading axis come from the env
resolver (day_count_convention / bus_days_in_year / calendar) while the
cash/vol wrappers carry a TradingClock — a 252-resolver paired with a
244-map would silently mis-date every query while the DF invariant breaks.
"""
from quantark.util.exceptions import ValidationError


def validate_trading_clock_configuration(env) -> None:
    """Every wrapped curve's TradingClock must match the env resolver."""
    for name in ("rate_curve", "div_yield", "vol_surface"):
        obj = getattr(env, name, None)
        clock = getattr(obj, "clock", None)
        if clock is None:
            # TradingClockVolSurface carries its clock on the time map, not on itself
            clock = getattr(getattr(obj, "time_map", None), "clock", None)
        if clock is None:
            continue
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
