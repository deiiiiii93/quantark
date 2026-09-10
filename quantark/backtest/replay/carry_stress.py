"""Finite scenarios for carry risk the listed futures cannot span.

A nodal bump moves a quote, so a futures hedge sized against it responds.
The scenarios here deliberately do NOT move spot or any listed quote, which
means every futures hedge P&L in them is exactly zero while a product with
exposure between or beyond the quoted tenors still moves.  That is the whole
point: "nodal rhoq is zero" is not "the carry risk is hedged".

Two families (design section 5.1), both written as a log-forward
displacement ``L(t)`` applied to the base curve::

    q_stress(t) = q_base(t) - L(t) / t

tail
    ``L(t) = -lambda * max(t - T_n, 0)``, so the extra yield is
    ``lambda * max(t - T_n, 0) / t`` -- zero at every listed tenor, zero at
    ``t = 0``, and growing beyond the last node.  For a discounted claim on
    ``S_T`` this gives ``dV/dlambda = -(T - T_n) V``.
shape
    ``L(t) = epsilon * 4u(1-u)`` on ``[A, B]`` with ``u = (t-A)/(B-A)``, zero
    outside.  It vanishes at both anchors and peaks at ``epsilon`` in the
    middle, including on the first interval ``[0, T_1]``.  At ``t = 0`` with
    ``A = 0`` the yield increment has the finite limit ``-4 epsilon / B``.

Both adapters wrap the base dividend object and evaluate the displacement at
each pricing time.  They must not be sampled at a few tenors and rebuilt
through the interpolator: re-fitting a stress that vanishes at every anchor
would erase it, and the scenario would silently become a no-op.

A joint scenario moves spot and the curve together at FIXED holdings and is
compared against the linear prediction.  Its repricing residual is carry
curvature and cross terms; ``V = F^2`` with unchanged spot exists precisely
to show that this residual is not spot gamma.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from quantark.backtest.replay.carry_context import CarryCurveContext
from quantark.param.div.dividend_yield import DividendYield
from quantark.util.exceptions import ValidationError

#: Every scenario family the recorder knows how to run.
SCENARIO_FAMILIES: Tuple[str, ...] = (
    "spot",
    "nodal_carry",
    "parallel_carry",
    "joint_spot_carry",
    "independent_tail",
    "interpolation_shape",
)

__all__ = [
    "SCENARIO_FAMILIES",
    "CarryScenario",
    "ScenarioResult",
    "ShapeStressedDividendYield",
    "TailStressedDividendYield",
    "run_scenario",
    "shape_scenario",
    "tail_scenario",
]


def _as_array(time_to_maturity):
    return np.asarray(time_to_maturity, dtype=float)


class _DisplacedDividendYield(DividendYield):
    """``q(t) = q_base(t) - L(t)/t`` for a log-forward displacement ``L``."""

    def __init__(self, base: DividendYield):
        self._base = base

    # subclasses implement the displacement
    def _displacement(self, t: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _limit_at_zero(self) -> float:
        """Finite value of ``-L(t)/t`` as ``t -> 0``; defaults to zero."""
        return 0.0

    def get_yield(self, time_to_maturity):
        t = _as_array(time_to_maturity)
        displacement = self._displacement(t)
        increment = np.zeros_like(t)
        np.divide(-displacement, t, out=increment, where=t > 0.0)
        limit = self._limit_at_zero()
        if limit != 0.0:
            increment = np.where(t > 0.0, increment, limit)
        base = self._base.get_yield(t if t.ndim else float(t))
        out = np.asarray(base, dtype=float) + increment
        if t.ndim == 0:
            return float(out)
        return out

    def parallel_shifted(self, shift: float) -> DividendYield:
        from quantark.param.div.dividend_yield import ParallelShiftDividendYield

        return ParallelShiftDividendYield(self, float(shift))


class TailStressedDividendYield(_DisplacedDividendYield):
    """Extra instantaneous carry yield beyond the last listed tenor."""

    def __init__(self, base: DividendYield, last_tenor: float, rate_shift: float):
        super().__init__(base)
        if not math.isfinite(float(last_tenor)) or float(last_tenor) <= 0.0:
            raise ValidationError("last_tenor must be finite and positive")
        if not math.isfinite(float(rate_shift)):
            raise ValidationError("rate_shift must be finite")
        self.last_tenor = float(last_tenor)
        self.rate_shift = float(rate_shift)

    def _displacement(self, t: np.ndarray) -> np.ndarray:
        return -self.rate_shift * np.maximum(t - self.last_tenor, 0.0)

    def __repr__(self) -> str:
        return (
            f"TailStressedDividendYield(lambda={self.rate_shift:+.4%}, "
            f"T_n={self.last_tenor:.4f})"
        )


class ShapeStressedDividendYield(_DisplacedDividendYield):
    """Log-forward bump inside one interval, vanishing at both anchors."""

    def __init__(
        self, base: DividendYield, start: float, end: float, log_forward_shift: float
    ):
        super().__init__(base)
        start, end = float(start), float(end)
        if not math.isfinite(start) or start < 0.0:
            raise ValidationError("shape interval start must be finite and >= 0")
        if not math.isfinite(end) or end <= start:
            raise ValidationError("shape interval must have positive width")
        if not math.isfinite(float(log_forward_shift)):
            raise ValidationError("log_forward_shift must be finite")
        self.start = start
        self.end = end
        self.log_forward_shift = float(log_forward_shift)

    def _displacement(self, t: np.ndarray) -> np.ndarray:
        u = np.zeros_like(t)
        inside = (t > self.start) & (t < self.end)
        np.divide(
            t - self.start, self.end - self.start, out=u, where=inside
        )
        return np.where(inside, self.log_forward_shift * 4.0 * u * (1.0 - u), 0.0)

    def _limit_at_zero(self) -> float:
        # L(t) = eps * 4u(1-u) with u = t/B near zero, so L(t)/t -> 4 eps / B
        # and the yield increment -L/t tends to -4 eps / B.
        if self.start != 0.0:
            return 0.0
        return -4.0 * self.log_forward_shift / self.end

    def __repr__(self) -> str:
        return (
            f"ShapeStressedDividendYield(eps={self.log_forward_shift:+.4%}, "
            f"[{self.start:.4f}, {self.end:.4f}])"
        )


# ---------------------------------------------------------------------------
# Scenario definitions and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CarryScenario:
    """One finite, fully specified market move at FIXED holdings.

    A joint scenario names every change together; nothing is applied twice
    and no quantity is re-sized inside it.  ``finite`` distinguishes a full
    repricing shock from a derivative estimate, which report different
    things and must not be compared as if they were the same number.
    """

    scenario_id: str
    family: str
    spot_shift_rel: float = 0.0
    parallel_yield_shift: float = 0.0
    node_yield_shifts: Dict[str, float] = field(default_factory=dict)
    tail_rate_shift: float = 0.0
    shape_interval: Optional[Tuple[float, float]] = None
    shape_log_forward_shift: float = 0.0
    finite: bool = True

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValidationError("scenario_id must be non-empty")
        if self.family not in SCENARIO_FAMILIES:
            raise ValidationError(
                f"family must be one of {SCENARIO_FAMILIES}, got {self.family!r}"
            )
        for name in (
            "spot_shift_rel",
            "parallel_yield_shift",
            "tail_rate_shift",
            "shape_log_forward_shift",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValidationError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "node_yield_shifts",
            {str(k): float(v) for k, v in self.node_yield_shifts.items()},
        )
        for contract, shift in self.node_yield_shifts.items():
            if not math.isfinite(shift):
                raise ValidationError(f"node shift for {contract} must be finite")
        if self.shape_interval is not None:
            start, end = (float(v) for v in self.shape_interval)
            if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
                raise ValidationError("shape_interval must be an increasing pair")
            object.__setattr__(self, "shape_interval", (start, end))
        if self.spot_shift_rel <= -1.0:
            raise ValidationError("spot_shift_rel must leave a positive spot")

    @property
    def moves_listed_quotes(self) -> bool:
        """True when a futures hedge can respond to this scenario at all."""
        return bool(
            self.spot_shift_rel
            or self.parallel_yield_shift
            or self.node_yield_shifts
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "family": self.family,
            "spot_shift_rel": self.spot_shift_rel,
            "parallel_yield_shift": self.parallel_yield_shift,
            "node_yield_shifts": dict(self.node_yield_shifts),
            "tail_rate_shift": self.tail_rate_shift,
            "shape_interval": self.shape_interval,
            "shape_log_forward_shift": self.shape_log_forward_shift,
            "finite": self.finite,
            "yield_units": "absolute decimal annual yield",
            "log_forward_units": "absolute log forward displacement",
            "spot_units": "relative",
        }


@dataclass(frozen=True)
class ScenarioResult:
    """Product, hedge and total P&L of one scenario, with its own validity."""

    scenario_id: str
    family: str
    holdings_kind: str
    status: str
    reason: str
    spot: float
    stressed_spot: float
    base_prices: Dict[str, float]
    stressed_prices: Dict[str, float]
    product_pnl: float = float("nan")
    hedge_pnl: float = float("nan")
    book_pnl: float = float("nan")
    linear_prediction: float = float("nan")
    repricing_error: float = float("nan")
    source_basis_assumption: str = "listed futures quotes"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in ("ok", "failed", "not_measured"):
            raise ValidationError(f"unknown scenario status: {self.status!r}")


def tail_scenario(rate_shift: float, *, scenario_id: str = "") -> CarryScenario:
    """The mandatory independent-tail stress."""
    return CarryScenario(
        scenario_id=scenario_id or f"tail{rate_shift:+.4f}",
        family="independent_tail",
        tail_rate_shift=float(rate_shift),
    )


def shape_scenario(
    log_forward_shift: float,
    interval: Tuple[float, float],
    *,
    scenario_id: str = "",
) -> CarryScenario:
    """An interpolation-shape stress on one interval, anchors unchanged."""
    start, end = interval
    return CarryScenario(
        scenario_id=scenario_id
        or f"shape{log_forward_shift:+.4f}@[{start:g},{end:g}]",
        family="interpolation_shape",
        shape_interval=(start, end),
        shape_log_forward_shift=float(log_forward_shift),
    )


def stressed_dividend(
    context: CarryCurveContext, scenario: CarryScenario
) -> Tuple[CarryCurveContext, Any]:
    """The scenario's context and its dividend object.

    Quote transformations go through the context so the builder rebuilds
    them; the tail and shape displacements wrap the RESULT, because they are
    not expressible as quote moves and re-fitting would erase them.
    """
    moved = context
    if scenario.spot_shift_rel:
        moved = moved.with_spot(context.spot * (1.0 + scenario.spot_shift_rel))
    if scenario.parallel_yield_shift:
        moved = moved.parallel_yield_shift(scenario.parallel_yield_shift)
    for contract, shift in scenario.node_yield_shifts.items():
        moved = moved.bump_node_yield(contract, shift)
    dividend = moved.dividend()
    if scenario.tail_rate_shift:
        dividend = TailStressedDividendYield(
            dividend,
            last_tenor=float(moved.quotes[-1].maturity),
            rate_shift=scenario.tail_rate_shift,
        )
    if scenario.shape_interval is not None and scenario.shape_log_forward_shift:
        start, end = scenario.shape_interval
        dividend = ShapeStressedDividendYield(
            dividend,
            start=start,
            end=end,
            log_forward_shift=scenario.shape_log_forward_shift,
        )
    return moved, dividend


def run_scenario(
    price_at,
    context: CarryCurveContext,
    scenario: CarryScenario,
    holdings: Mapping[str, float],
    *,
    holdings_kind: str = "actual",
    base_price: Optional[float] = None,
    linear_prediction: float = float("nan"),
) -> ScenarioResult:
    """Reprice the book under one finite scenario at fixed holdings."""
    if holdings_kind not in ("actual", "ideal"):
        raise ValidationError(f"unknown holdings kind: {holdings_kind!r}")
    base_prices = context.prices
    try:
        if base_price is None:
            base_price = float(price_at(context.spot, context.dividend()))
        moved, dividend = stressed_dividend(context, scenario)
        stressed_price = float(price_at(moved.spot, dividend))
        if not math.isfinite(base_price) or not math.isfinite(stressed_price):
            raise ValidationError("scenario repricing produced a non-finite value")
    except ValidationError as error:
        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            family=scenario.family,
            holdings_kind=holdings_kind,
            status="failed",
            reason=str(error),
            spot=context.spot,
            stressed_spot=float("nan"),
            base_prices=dict(base_prices),
            stressed_prices={},
            metadata=scenario.metadata(),
        )
    product_pnl = stressed_price - base_price
    hedge_pnl = sum(
        float(holdings.get(q.contract, 0.0))
        * q.multiplier
        * (q.price - base_prices[q.contract])
        for q in moved.quotes
    )
    book_pnl = product_pnl + hedge_pnl
    return ScenarioResult(
        scenario_id=scenario.scenario_id,
        family=scenario.family,
        holdings_kind=holdings_kind,
        status="ok",
        reason="",
        spot=context.spot,
        stressed_spot=moved.spot,
        base_prices=dict(base_prices),
        stressed_prices=moved.prices,
        product_pnl=product_pnl,
        hedge_pnl=hedge_pnl,
        book_pnl=book_pnl,
        linear_prediction=float(linear_prediction),
        repricing_error=book_pnl - float(linear_prediction),
        metadata=scenario.metadata(),
    )
