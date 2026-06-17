"""As-of (roll-down) pricing-environment helpers for MC exposure (spec §3.2).

A value surface reprices a trade at a future simulation time ``t`` and a simulated
``spot``. For date-based equity products ``get_maturity`` rolls the remaining tenor
down from ``env.valuation_date`` to ``exercise_date`` (real-API gotcha), so the
as-of env must ADVANCE ``valuation_date`` by ``t`` years — changing the spot alone
leaves the tenor unchanged and mis-prices.

v1 advances the date by whole calendar days (``CALENDAR_DAYS`` / ``ACT_365`` use
``delta.days / 365``); other day-count conventions raise rather than approximate.
"""

from dataclasses import replace
from datetime import timedelta

from quantark.param import SpotQuote
from quantark.util.calendar import DayCountConvention
from quantark.util.exceptions import ValidationError

_DAY_DENOM = {DayCountConvention.CALENDAR_DAYS: 365.0, DayCountConvention.ACT_365: 365.0}


def equity_asof_env(base_env, spot, t):
    """Clone ``base_env`` rolled forward to year-fraction ``t`` with ``spot``.

    Keeps the (deterministic) rate/vol/dividend term structures; advances
    ``valuation_date`` by ``round(t * 365)`` days so date-based products roll down.
    """
    if base_env.spot_quote is None:
        raise ValidationError("base_env requires a spot_quote for equity roll-down")
    denom = _DAY_DENOM.get(base_env.day_count_convention)
    if denom is None:
        raise ValidationError(
            f"as-of roll-down supports CALENDAR_DAYS/ACT_365 only (v1); got "
            f"{base_env.day_count_convention}")
    if not (t >= 0):
        raise ValidationError("as-of time t must be >= 0")
    days = int(round(float(t) * denom))
    new_val = base_env.valuation_date + timedelta(days=days)
    new_spot = SpotQuote(spot=float(spot),
                         asset_name=getattr(base_env.spot_quote, "asset_name", None))
    return replace(base_env, valuation_date=new_val, spot_quote=new_spot)
