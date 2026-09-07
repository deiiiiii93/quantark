"""FactorMoves: scalar moves at the product coordinate + change detection (spec §5.3)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Optional

from quantark.param.basis.basis_yield import FlatBasisYield, ZeroBasis
from quantark.param.div import ContinuousDividendYield, NoDividend
from quantark.pnlexplain.base import Factor
from quantark.pnlexplain.equity.clock import (
    is_clock_wrapped, validate_clock_env, validate_clock_pair, wrapped_equal,
)
from quantark.pnlexplain.equity.coordinate import FactorCoordinate
from quantark.pnlexplain.equity.fingerprints import calendars_equal, engines_equivalent
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot
from quantark.util.calendar import DayCountConvention, calculate_year_fraction
from quantark.util.exceptions import NumericalError, ValidationError
from quantark.util.numerical import is_close, safe_divide

POINT_TOL = 1e-12       # spec §5.2: numeric valuation points advance by calendar_days/365


def objects_equal(a: Any, b: Any) -> bool:
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 - a comparison that raises counts as changed
        return False


def _is_zero_yield(obj: Any) -> bool:
    if obj is None or isinstance(obj, NoDividend):
        return True
    return isinstance(obj, ContinuousDividendYield) and float(getattr(obj, "div_yield", 1.0)) == 0.0


def _yields_equal(a: Any, b: Any) -> bool:
    if _is_zero_yield(a) and _is_zero_yield(b):
        return True
    return objects_equal(a, b)


def _is_zero_basis(obj: Any) -> bool:
    """None, ZeroBasis and a flat zero basis all price as no basis (spec §5.3)."""
    if obj is None or isinstance(obj, ZeroBasis):
        return True
    return isinstance(obj, FlatBasisYield) and float(obj.basis_yield) == 0.0


def _basis_equal(a: Any, b: Any) -> bool:
    if _is_zero_basis(a) and _is_zero_basis(b):
        return True
    return objects_equal(a, b)


def market_objects_equal(a: Any, b: Any, base: Callable[[Any, Any], bool] = objects_equal) -> bool:
    """Clock wrappers compare by (class, inner, clock); everything else by ``base`` (patch spec §8)."""
    if is_clock_wrapped(a) and is_clock_wrapped(b):
        return wrapped_equal(a, b, base)
    return base(a, b)


def _float_maturity_only(product: Any) -> bool:
    return (getattr(product, "exercise_date", None) is None
            and getattr(product, "maturity_date", None) is None
            and getattr(product, "maturity", None) is not None)


@dataclass(frozen=True)
class FactorMoves:
    coordinate: FactorCoordinate
    spot_t0: float
    spot_t1: float
    d_spot: float
    spot_return: float
    vol_t0: Optional[float]
    vol_t1: Optional[float]
    d_vol: Optional[float]
    rate_t0: Optional[float]
    rate_t1: Optional[float]
    d_rate: Optional[float]
    div_t0: Optional[float]
    div_t1: Optional[float]
    d_div: Optional[float]
    basis_t0: Optional[float]
    basis_t1: Optional[float]
    d_basis: Optional[float]
    calendar_days: int
    trading_days: Optional[int]
    year_fraction: float
    changed: FrozenSet[Factor]

    def display(self, factor: Factor) -> Dict[str, float]:
        """Display-unit moves for one factor; unavailable keys are omitted."""
        if factor is Factor.SPOT:
            return {"spot_return": self.spot_return}
        if factor is Factor.VOL:
            return {} if self.d_vol is None else {"vol_pts": self.d_vol * 100.0}
        if factor is Factor.RATE:
            return {} if self.d_rate is None else {"rate_pct": self.d_rate * 100.0}
        if factor is Factor.DIVIDEND:
            return {} if self.d_div is None else {"div_pct": self.d_div * 100.0}
        if factor is Factor.BASIS:
            return {} if self.d_basis is None else {"basis_pct": self.d_basis * 100.0}
        if factor is Factor.TIME:
            out = {"days": float(self.calendar_days)}
            if self.trading_days is not None:
                out["trading_days"] = float(self.trading_days)
            return out
        return {}


def validate_pair(snap0: ValuationSnapshot, snap1: ValuationSnapshot) -> None:
    if snap1.date <= snap0.date:
        raise ValidationError(f"date_t1 {snap1.date} must be after date_t0 {snap0.date}")
    e0, e1 = snap0.pricing_env, snap1.pricing_env
    if e0.day_count_convention != e1.day_count_convention or e0.bus_days_in_year != e1.bus_days_in_year:
        raise ValidationError("day count convention / bus_days_in_year differ between snapshots")
    if not calendars_equal(getattr(e0, "calendar", None), getattr(e1, "calendar", None)):
        raise ValidationError("calendars differ between snapshots (semantic comparison)")
    validate_clock_env(e0, "snapshot_t0.pricing_env")
    validate_clock_env(e1, "snapshot_t1.pricing_env")
    validate_clock_pair(e0, e1)
    # A float maturity on a BUSINESS_DAYS environment was rejected here while the roll rule
    # was fixed at days/365, which does not describe a trading-time contract. The rule now
    # takes the decrement from the environment's own clock (lifecycle.float_maturity_decrement),
    # so the contract is checked against the right number instead of being turned away.
    if e0.day_count_convention is DayCountConvention.BUSINESS_DAYS \
            and _float_maturity_only(snap0.product) and getattr(e0, "calendar", None) is None:
        raise ValidationError(
            "a float-maturity product on a BUSINESS_DAYS environment ages in trading time, "
            "which needs a calendar on the pricing environment to measure"
        )
    if snap0.quantity != snap1.quantity:
        raise ValidationError(
            "quantities differ between snapshots; a quantity change is a trade (use explain_position)"
        )
    if snap0.currency is not None and snap1.currency is not None and snap0.currency != snap1.currency:
        raise ValidationError("currency labels differ between snapshots")
    p0, p1 = snap0.point, snap1.point
    if (p0.date is None) != (p1.date is None):
        raise ValidationError("valuation points must share one representation (date or time)")
    if p0.date is None:
        days = (snap1.date - snap0.date).days
        if not is_close(p1.time - p0.time, days / 365.0, rel_tol=0.0, abs_tol=POINT_TOL):
            raise ValidationError("numeric valuation points must advance by calendar_days/365")


#: the environment field a factor reads, for the three clock-wrappable objects
_CLOCK_FIELD = {Factor.VOL: "vol_surface", Factor.RATE: "rate_curve", Factor.DIVIDEND: "div_yield"}


def _sample(env: Any, coordinate: FactorCoordinate, factor: Factor) -> Optional[float]:
    tenor = coordinate.tenor_t1
    if factor not in coordinate.applicable or tenor is None:
        return None
    if factor is Factor.VOL:
        return float(env.get_vol(coordinate.reference_strike, tenor))
    if factor is Factor.RATE:
        return float(env.get_rate(tenor))
    if factor is Factor.DIVIDEND:
        return float(env.get_div_yield(tenor))
    if factor is Factor.BASIS:
        return float(env.get_basis_yield(tenor))
    return None


def _read_inner(inner: Any, factor: Factor, point: float, strike: Any, spot: float) -> float:
    if factor is Factor.VOL:
        return float(inner.get_vol(strike, point, spot))
    if factor is Factor.RATE:
        return float(inner.get_rate(point))
    return float(inner.get_yield(point))                      # DIVIDEND


def sample_pair(e0: Any, e1: Any, coordinate: FactorCoordinate, factor: Factor):
    """(t0 level, t1 level, move) for one factor at ONE shared coordinate.

    A clock wrapper converts the coordinate with its OWN anchor, and the two
    snapshots are anchored at their own valuation dates, so sampling through the
    environment reads two different points of one unchanged market (patch spec
    2026-09-03 §14). Both sides therefore read the INNER at the t1 coordinate
    mapped onto the inner's axis: an unchanged market differences to exactly
    zero, matching ``changed`` (which ignores the anchor) and the waterfall
    (which re-anchors on the TIME step, so the anchor slide is time, like the
    tenor slide the shared coordinate already keeps out of the market moves).
    The move then carries the unit the bump moves: sigma_td for vol, the
    calendar-quoted r and q for the rate and dividend wrappers.
    """
    tenor = coordinate.tenor_t1
    if factor in _CLOCK_FIELD and tenor is not None and factor in coordinate.applicable:
        o0, o1 = getattr(e0, _CLOCK_FIELD[factor], None), getattr(e1, _CLOCK_FIELD[factor], None)
        # validate_clock_pair has already rejected a wrapper on one side only
        if is_clock_wrapped(o0) and is_clock_wrapped(o1):
            point = float(o1.to_inner_time(float(tenor)))
            if point > 0.0:
                a = _read_inner(o0.inner, factor, point, coordinate.reference_strike, float(e0.spot))
                b = _read_inner(o1.inner, factor, point, coordinate.reference_strike, float(e1.spot))
                return a, b, b - a
            # a horizon with no trading time left has no inner coordinate: fall
            # through to the wrapper's own documented degenerate reading
    a, b = _sample(e0, coordinate, factor), _sample(e1, coordinate, factor)
    return a, b, (None if a is None or b is None else b - a)


def build_factor_moves(
    snap0: ValuationSnapshot,
    snap1: ValuationSnapshot,
    coordinate: FactorCoordinate,
    *,
    engine_alive_t1: Any,
    lifecycle_changed: bool,
) -> FactorMoves:
    e0, e1 = snap0.pricing_env, snap1.pricing_env
    s0, s1 = float(e0.spot), float(e1.spot)
    days = (snap1.date - snap0.date).days
    cal = getattr(e0, "calendar", None)
    trading = None
    if cal is not None and hasattr(cal, "count_business_days"):
        trading = int(cal.count_business_days(snap0.date, snap1.date, include_start=False, include_end=True))
    yf = float(calculate_year_fraction(snap0.date, snap1.date, e0.day_count_convention,
                                       e0.bus_days_in_year, calendar=cal))

    vol = sample_pair(e0, e1, coordinate, Factor.VOL)
    rate = sample_pair(e0, e1, coordinate, Factor.RATE)
    div = sample_pair(e0, e1, coordinate, Factor.DIVIDEND)
    basis = sample_pair(e0, e1, coordinate, Factor.BASIS)
    for label, triple in (("vol", vol), ("rate", rate), ("dividend", div), ("basis", basis)):
        for v in triple:
            if v is not None and not math.isfinite(v):
                raise NumericalError(f"non-finite {label} sample at the product coordinate: {v!r}")

    changed = {Factor.TIME}
    app = coordinate.applicable
    if Factor.SPOT in app and s0 != s1:
        changed.add(Factor.SPOT)
    if Factor.VOL in app and not market_objects_equal(e0.vol_surface, e1.vol_surface):
        changed.add(Factor.VOL)
    if Factor.RATE in app and not market_objects_equal(e0.rate_curve, e1.rate_curve):
        changed.add(Factor.RATE)
    if Factor.DIVIDEND in app and not market_objects_equal(e0.div_yield, e1.div_yield, _yields_equal):
        changed.add(Factor.DIVIDEND)
    if Factor.BASIS in app and not _basis_equal(e0.basis_yield, e1.basis_yield):
        changed.add(Factor.BASIS)
    if Factor.MODEL in app and engine_alive_t1 is not None \
            and not engines_equivalent(snap0.engine, engine_alive_t1):
        changed.add(Factor.MODEL)
    if lifecycle_changed:
        changed.add(Factor.LIFECYCLE_EVENT)

    return FactorMoves(
        coordinate=coordinate, spot_t0=s0, spot_t1=s1, d_spot=s1 - s0, spot_return=safe_divide(s1 - s0, s0),
        vol_t0=vol[0], vol_t1=vol[1], d_vol=vol[2],
        rate_t0=rate[0], rate_t1=rate[1], d_rate=rate[2],
        div_t0=div[0], div_t1=div[1], d_div=div[2],
        basis_t0=basis[0], basis_t1=basis[1], d_basis=basis[2],
        calendar_days=days, trading_days=trading, year_fraction=yf, changed=frozenset(changed),
    )
