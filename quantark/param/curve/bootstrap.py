"""Curve bootstrappers: quotes + configuration → ZeroCurve / CarryCurve.

Spec: ``.ship/2026-11-05-curve-construction/spec.md`` §Decisions "Bootstrap
algorithm", "Carry nodes are closed-form", "Clock rule".

Rate curves are solved by an **iterative sequential bootstrap**: one pillar per
quote at the quote's last payment time, each pillar's zero rate found by Brent's
method so that the quote reprices while the other pillars are held. Sweep 1 is
*truncated-sequential* — while pillar ``i`` is solved the curve holds only
pillars ``1..i`` and the long-end rule extrapolates beyond — so no artificial
initial guess ever enters the interpolant. Later sweeps re-solve every pillar
on the full curve until the largest pillar change is at most
:data:`SWEEP_TOLERANCE`. Local schemes (linear, log-linear) settle in two
sweeps; cubic and monotone-convex schemes need more because the interpolant
between earlier pillars depends on later ones. Non-convergence or a bracket
without a sign change raises ``NumericalError`` — there is no fallback.

Carry nodes need no solver: ``B = ln(F/S)`` from a futures price, or from the
put–call-parity forward ``F = K + (C − P)/DF(T)`` of an option pair.

Clock rule: a run is numeric (``valuation_date=None``, quotes carry
``tenor_years``) or dated (quotes carry dates). Mixing raises; so does handing a
builder a curve anchored at a different ``valuation_date``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq

from quantark.param.curve.carry_curve import CarryCurve
from quantark.param.curve.interpolation import CurveInterpolation
from quantark.param.curve.quotes import (
    BondQuote,
    FuturesQuote,
    OptionPairQuote,
    SwapQuote,
    check_run_clock,
)
from quantark.param.curve.strip import InstrumentStrip, build_strip
from quantark.param.curve.zero_curve import ZeroCurve
from quantark.param.extrapolation import CarryExtrapolation, RateExtrapolation
from quantark.param.node_roles import NodeRole
from quantark.param.rrf.rate_curve import RateCurve
from quantark.util.exceptions import NumericalError, ValidationError

SWEEP_TOLERANCE: float = 1e-14
"""Sweep-to-sweep convergence: largest pillar change, in zero-rate units (two
orders above the Brent root jitter at PILLAR_XTOL)."""

PILLAR_XTOL: float = 1e-16
"""Brent absolute tolerance on a pillar's zero rate."""

PILLAR_RTOL: float = float(4 * np.finfo(float).eps)
"""Brent relative tolerance (scipy's minimum)."""

PILLAR_MAXITER: int = 200
"""Brent iteration cap per pillar solve."""

_BRACKET: Tuple[float, float] = (-1.0, 1.0)
_BRACKET_WIDENINGS: int = 4
_CurvePair = Callable[[ZeroCurve], Tuple[RateCurve, RateCurve]]


@dataclass(frozen=True)
class BootstrapConfig:
    """Interpolation, long-end rule and sweep cap of a rate-curve bootstrap."""

    interpolation: CurveInterpolation = CurveInterpolation.LOG_LINEAR_DF
    extrapolation: RateExtrapolation = RateExtrapolation.FLAT_FORWARD_RATE
    max_sweeps: int = 50

    def __post_init__(self) -> None:
        if not isinstance(self.interpolation, CurveInterpolation):
            raise ValidationError(f"interpolation must be a CurveInterpolation, got {self.interpolation!r}")
        if not isinstance(self.extrapolation, RateExtrapolation):
            raise ValidationError(f"extrapolation must be a RateExtrapolation, got {self.extrapolation!r}")
        if int(self.max_sweeps) < 1:
            raise ValidationError(f"max_sweeps must be >= 1, got {self.max_sweeps}")


# --------------------------------------------------------------------------- helpers


def _label(quote) -> str:
    return getattr(quote, "label", None) or repr(quote)


def _check_anchor(curve, valuation_date: Optional[datetime], what: str) -> None:
    """A curve carrying a ``valuation_date`` must be on the run's clock; others pass."""
    if curve is None:
        raise ValidationError(f"{what} is required")
    anchor = getattr(curve, "valuation_date", None)
    if anchor is not None and anchor != valuation_date:
        run = "numeric (valuation_date=None)" if valuation_date is None else f"dated {valuation_date:%Y-%m-%d}"
        raise ValidationError(
            f"{what} is anchored at {anchor:%Y-%m-%d} but this run is {run}: one clock per run"
        )


def _sorted_strips(quotes: Sequence, valuation_date: Optional[datetime]) -> List[InstrumentStrip]:
    check_run_clock(quotes, valuation_date)
    strips = [build_strip(q, valuation_date) for q in quotes]
    strips.sort(key=lambda s: s.pillar_time)
    for a, b in zip(strips, strips[1:]):
        if b.pillar_time == a.pillar_time:
            raise ValidationError(
                f"quotes {_label(a.quote)} and {_label(b.quote)} share the pillar time {a.pillar_time}: "
                "one pillar per quote"
            )
    return strips


def _solve_pillars(
    strips: Sequence[InstrumentStrip],
    config: BootstrapConfig,
    valuation_date: Optional[datetime],
    curves_of: _CurvePair,
) -> ZeroCurve:
    """Iterative sequential bootstrap over ``strips`` (sorted by pillar time).

    ``curves_of(curve_under_construction)`` returns the ``(discount, projection)``
    pair the residuals see: ``(c, c)`` for a discount bootstrap, ``(external, c)``
    for a projection bootstrap.
    """
    times = [s.pillar_time for s in strips]
    n = len(strips)

    def curve_for(levels: Sequence[float]) -> ZeroCurve:
        k = len(levels)
        return ZeroCurve(
            list(zip(times[:k], levels)),
            interpolation=config.interpolation,
            extrapolation=config.extrapolation,
            node_roles=(NodeRole.CALIBRATED,) * k,
            last_observable_tenor=times[k - 1],
            valuation_date=valuation_date,
        )

    def solve_one(i: int, levels: Sequence[float]) -> float:
        base = list(levels)

        def residual(z: float) -> float:
            trial = list(base)
            if i < len(trial):
                trial[i] = z
            else:
                trial.append(z)
            discount, projection = curves_of(curve_for(trial))
            return float(strips[i].residual(discount, projection))

        lo, hi = _BRACKET
        f_lo, f_hi = residual(lo), residual(hi)
        widenings = 0
        while f_lo * f_hi > 0.0:
            if widenings >= _BRACKET_WIDENINGS:
                raise NumericalError(
                    f"no sign change for {_label(strips[i].quote)} in the zero-rate bracket [{lo}, {hi}]"
                )
            lo, hi = 2.0 * lo, 2.0 * hi
            widenings += 1
            f_lo, f_hi = residual(lo), residual(hi)
        if f_lo == 0.0:
            return lo
        if f_hi == 0.0:
            return hi
        try:
            root = brentq(residual, lo, hi, xtol=PILLAR_XTOL, rtol=PILLAR_RTOL, maxiter=PILLAR_MAXITER)
        except RuntimeError as exc:  # scipy: convergence failure
            raise NumericalError(f"pillar solve failed for {_label(strips[i].quote)}: {exc}") from exc
        return float(root)

    # sweep 1: truncated-sequential — the curve holds only the pillars solved so far
    levels: List[float] = []
    for i in range(n):
        levels.append(solve_one(i, levels))

    # sweeps 2..max_sweeps: every pillar on the full curve, until the fixed point
    delta = float("inf")
    for _sweep in range(2, int(config.max_sweeps) + 1):
        delta = 0.0
        for i in range(n):
            new = solve_one(i, levels)
            delta = max(delta, abs(new - levels[i]))
            levels[i] = new
        if delta <= SWEEP_TOLERANCE:
            return curve_for(levels)
    raise NumericalError(
        f"bootstrap did not converge in {config.max_sweeps} sweep(s): last max pillar change "
        f"{delta:.3e} > SWEEP_TOLERANCE {SWEEP_TOLERANCE}"
    )


# --------------------------------------------------------------------------- public API


def bootstrap_discount_curve(
    quotes: Sequence,
    config: BootstrapConfig = BootstrapConfig(),
    valuation_date: Optional[datetime] = None,
) -> ZeroCurve:
    """Single-curve bootstrap of deposits, par swaps and bonds.

    Swaps project their floating legs off the curve under construction; the
    result discounts and projects. One pillar per quote at its last payment time.
    """
    if not isinstance(config, BootstrapConfig):
        raise ValidationError(f"config must be a BootstrapConfig, got {config!r}")
    strips = _sorted_strips(quotes, valuation_date)
    return _solve_pillars(strips, config, valuation_date, lambda c: (c, c))


def bootstrap_projection_curve(
    quotes: Sequence,
    discount_curve: RateCurve,
    config: BootstrapConfig = BootstrapConfig(),
    valuation_date: Optional[datetime] = None,
) -> ZeroCurve:
    """Projection ("float") curve of one index from deposits and par swaps given a discount curve.

    Swap fixed and floating legs are discounted on ``discount_curve``; floating
    rates and deposit pseudo-discount-factors come from the curve under
    construction. All swaps must share one ``RateIndex``; bonds are refused.
    """
    if not isinstance(config, BootstrapConfig):
        raise ValidationError(f"config must be a BootstrapConfig, got {config!r}")
    _check_anchor(discount_curve, valuation_date, "discount_curve")
    indices = set()
    for q in quotes:
        if isinstance(q, BondQuote):
            raise ValidationError(f"bond quote {_label(q)} does not define a projection curve")
        if isinstance(q, SwapQuote):
            indices.add(q.index.name)
    if len(indices) > 1:
        raise ValidationError(
            f"a projection curve is per index; got swaps on {sorted(indices)} — bootstrap one curve per index"
        )
    strips = _sorted_strips(quotes, valuation_date)
    return _solve_pillars(strips, config, valuation_date, lambda c: (discount_curve, c))


def build_carry_curve(
    spot: float,
    quotes: Sequence,
    rate_curve: RateCurve,
    interpolation: CurveInterpolation = CurveInterpolation.LOG_LINEAR_DF,
    extrapolation: CarryExtrapolation = CarryExtrapolation.FLAT_FORWARD_CARRY,
    valuation_date: Optional[datetime] = None,
) -> CarryCurve:
    """Carry curve ``B(T) = ln(F/S)`` from futures prices and option pairs.

    Futures give ``F`` directly; an option pair gives ``F = K + (C − P)/DF(T)`` on
    ``rate_curve``. One quote per expiry (duplicates raise); ``F <= 0`` raises.
    """
    try:
        s0 = float(spot)
    except (TypeError, ValueError):
        raise ValidationError(f"spot must be a number, got {spot!r}") from None
    if not np.isfinite(s0) or s0 <= 0.0:
        raise ValidationError(f"spot must be positive and finite, got {spot}")
    _check_anchor(rate_curve, valuation_date, "rate_curve")
    check_run_clock(quotes, valuation_date)
    nodes: List[Tuple[float, float]] = []
    labels: List[str] = []
    for q in quotes:
        if isinstance(q, FuturesQuote):
            T = q.pillar_time(valuation_date)
            forward = float(q.price)
        elif isinstance(q, OptionPairQuote):
            T = q.pillar_time(valuation_date)
            df = float(rate_curve.get_discount_factor(T))
            forward = float(q.strike) + (float(q.call_price) - float(q.put_price)) / df
        else:
            raise ValidationError(f"not a carry quote (FuturesQuote / OptionPairQuote): {q!r}")
        if not np.isfinite(forward) or forward <= 0.0:
            raise ValidationError(f"implied forward {forward} <= 0 for {_label(q)}: no log-carry node")
        nodes.append((T, float(np.log(forward / s0))))
        labels.append(_label(q))
    order = sorted(range(len(nodes)), key=lambda k: nodes[k][0])
    for a, b in zip(order, order[1:]):
        if nodes[a][0] == nodes[b][0]:
            raise ValidationError(
                f"quotes {labels[a]} and {labels[b]} share the expiry {nodes[a][0]}: one node per expiry"
            )
    sorted_nodes = [nodes[k] for k in order]
    return CarryCurve(
        sorted_nodes,
        interpolation=interpolation,
        extrapolation=extrapolation,
        node_roles=(NodeRole.CALIBRATED,) * len(sorted_nodes),
        last_observable_tenor=sorted_nodes[-1][0],
        valuation_date=valuation_date,
    )


def reprice_quotes(
    quotes: Sequence,
    discount_curve: RateCurve,
    projection_curve: Optional[RateCurve] = None,
    valuation_date: Optional[datetime] = None,
) -> List[float]:
    """Residual of every rate quote on the given curves, in the caller's order.

    Deposit: ``P(t_s) − P(t_e)(1 + rτ)``; swap (unit notional): ``PV_fixed − PV_float``;
    bond (per 100 face): ``PV_flows − dirty``. ``projection_curve`` defaults to the
    discount curve (single-curve).
    """
    _check_anchor(discount_curve, valuation_date, "discount_curve")
    projection = projection_curve if projection_curve is not None else discount_curve
    _check_anchor(projection, valuation_date, "projection_curve")
    check_run_clock(quotes, valuation_date)
    return [float(build_strip(q, valuation_date).residual(discount_curve, projection)) for q in quotes]
