"""The six carry tails past the last pillar ``T_n``, and the trailing far-IM anchor.

Design §4.2.  Three tails are the library's ``CarryExtrapolation`` rules; the
three converging tails are :class:`ConvergingCarryCurve`, a study-local
``CarryCurve`` subclass (no library change).  With ``c_n = B_n/T_n``,
``κ = ln 2 / half_life`` and ``τ = t − T_n > 0``::

    b(t) = b_∞ + (c_n − b_∞)·e^{−κτ}                     (forward carry)
    B(t) = B_n + b_∞·τ + (c_n − b_∞)·(1 − e^{−κτ})/κ     (cumulative carry)

The tail starts from the last zero-carry LEVEL ``c_n`` (not the last slope), so
it reads only node ``n`` — no calendar-spread term.  ``κ → 0`` is ``hold_yield``
(``B_n + c_n·τ = B_n·t/T_n``); ``κ → ∞`` is forward carry ``b_∞`` right past ``T_n``.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from quantark.param.curve.carry_curve import CarryCurve
from quantark.param.curve.interpolation import CurveInterpolation
from quantark.param.extrapolation import CarryExtrapolation
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_valid_number, safe_exp, safe_log, validate_positive

from . import HALF_LIVES, IM_MULT, MO_MULT, RATE
from .chain import DayChain

ArrayLike = Union[float, np.ndarray]


def _converging_id(half_life: float) -> str:
    return f"converge_hl{int(round(100 * half_life)):03d}"


#: Tail id -> half-life (years) of the converging tails.
CONVERGING_HALF_LIFE: Dict[str, float] = {_converging_id(hl): float(hl) for hl in HALF_LIVES}
#: Tail id -> library long-end rule of the three non-converging tails.
LIBRARY_TAILS: Dict[str, CarryExtrapolation] = {
    "hold_forward": CarryExtrapolation.FLAT_FORWARD_CARRY,
    "hold_yield": CarryExtrapolation.FLAT_CARRY_YIELD,
    "zero_carry": CarryExtrapolation.ZERO_FORWARD_CARRY,
}
TAILS: Tuple[str, ...] = tuple(LIBRARY_TAILS) + tuple(CONVERGING_HALF_LIFE)


def is_converging(tail: str) -> bool:
    """True for the anchor-converging tails."""
    return tail in CONVERGING_HALF_LIFE


class ConvergingCarryCurve(CarryCurve):
    """``CarryCurve`` whose tail past ``T_n`` converges to the forward carry ``b_inf``.

    Identical to the base curve on ``[0, T_n]``.  The base is built with
    ``ZERO_FORWARD_CARRY`` on purpose: ``CarryImpliedDividendYield.parallel_shifted``
    then takes its exact wrapper path (a node shift does not move ``b_inf`` and
    would not be a parallel ``q`` shift of this tail).

    Args:
        nodes: ``(T_i, B_i)`` pairs, as for :class:`CarryCurve`.
        b_inf: Long-run forward carry the tail converges to (per year).
        half_life: Half-life of the convergence, years (> 0).
        **kw: ``interpolation``, ``node_roles``, ``last_observable_tenor``,
            ``valuation_date``; ``extrapolation`` may only be ``ZERO_FORWARD_CARRY``.
    """

    def __init__(self, nodes: Sequence[Tuple[float, float]], *, b_inf: float, half_life: float, **kw) -> None:
        extrapolation = kw.pop("extrapolation", CarryExtrapolation.ZERO_FORWARD_CARRY)
        if extrapolation is not CarryExtrapolation.ZERO_FORWARD_CARRY:
            raise ValidationError(
                "ConvergingCarryCurve is built on ZERO_FORWARD_CARRY (exact parallel-shift "
                f"wrapper path); got {extrapolation!r}"
            )
        if not is_valid_number(float(b_inf)):
            raise ValidationError(f"b_inf must be finite, got {b_inf}")
        validate_positive(float(half_life), "half_life")
        super().__init__(nodes, extrapolation=CarryExtrapolation.ZERO_FORWARD_CARRY, **kw)
        self.b_inf = float(b_inf)
        self.half_life = float(half_life)
        self.kappa = math.log(2.0) / self.half_life
        self._tn, self._bn = self._nodes[-1]
        self.c_n = self._bn / self._tn

    # ----------------------------------------------------------------- tail

    def _tail_carry(self, t: np.ndarray) -> np.ndarray:
        tau = t - self._tn
        # (1 − e^{−κτ})/κ as −expm1(−κτ)/κ: no cancellation at tiny κ
        return self._bn + self.b_inf * tau + (self.c_n - self.b_inf) * (-np.expm1(-self.kappa * tau)) / self.kappa

    def _tail_forward(self, t: np.ndarray) -> np.ndarray:
        return self.b_inf + (self.c_n - self.b_inf) * np.asarray(safe_exp(-self.kappa * (t - self._tn)), dtype=float)

    def _with_tail(self, base: ArrayLike, T: ArrayLike, tail_fn, inclusive: bool) -> ArrayLike:
        arr = np.asarray(T, dtype=float)
        if arr.ndim == 0:
            t = float(arr)
            in_tail = t >= self._tn if inclusive else t > self._tn
            return float(tail_fn(np.array([t]))[0]) if in_tail else float(base)
        out = np.array(base, dtype=float)  # a copy
        mask = arr >= self._tn if inclusive else arr > self._tn
        if np.any(mask):
            out[mask] = tail_fn(arr[mask])
        return out

    # ----------------------------------------------------------------- evaluation

    def carry(self, T: ArrayLike) -> ArrayLike:
        """``B(T)``: the base curve on ``[0, T_n]``, the converging tail beyond."""
        return self._with_tail(super().carry(T), T, self._tail_carry, inclusive=False)

    def forward_carry(self, T: ArrayLike) -> ArrayLike:
        """Right-continuous ``dB/dt+``; at ``T_n`` the tail's start ``c_n``."""
        return self._with_tail(super().forward_carry(T), T, self._tail_forward, inclusive=True)

    def interval_carry(self, t0: float, t1: float) -> float:
        """``(B(t1) − B(t0))/(t1 − t0)``; for ``t1 == t0`` the forward carry at ``t0``."""
        t0, t1 = float(t0), float(t1)
        if t1 < t0:
            raise ValidationError("t1 must be >= t0")
        if t1 == t0:
            return float(self.forward_carry(t0))
        return (float(self.carry(t1)) - float(self.carry(t0))) / (t1 - t0)

    # ----------------------------------------------------------------- derived curves

    def with_nodes(self, nodes: Sequence[Tuple[float, float]]) -> "ConvergingCarryCurve":
        """Same tail (``b_inf``, ``half_life``) and configuration on new nodes."""
        return ConvergingCarryCurve(
            nodes,
            b_inf=self.b_inf,
            half_life=self.half_life,
            interpolation=self.interpolation,
            node_roles=self.node_roles,
            last_observable_tenor=self.last_observable_tenor,
            valuation_date=self.valuation_date,
        )

    def __repr__(self) -> str:
        return (
            f"ConvergingCarryCurve(nodes={len(self._nodes)}, interpolation={self.interpolation.value}, "
            f"b_inf={self.b_inf:.6g}, half_life={self.half_life:g})"
        )


# ---------------------------------------------------------------------------
# Anchor and the daily curve
# ---------------------------------------------------------------------------


def anchor_series(chains: Sequence[DayChain], window: int) -> Dict[date, float]:
    """Trailing mean of ``far_carry`` over the ``window`` days ending at each ``d``.

    Point-in-time: ``anchor[d]`` reads only chains up to and including ``d``.
    Days with fewer than ``window`` observations are absent.
    """
    if int(window) != window or window < 1:
        raise ValidationError(f"window must be a positive integer, got {window}")
    days = [dc.day for dc in chains]
    if any(b <= a for a, b in zip(days, days[1:])):
        raise ValidationError("chains must be strictly increasing in day")
    vals = np.array([dc.far_carry for dc in chains], dtype=float)
    w = int(window)
    return {days[i]: float(np.mean(vals[i - w + 1:i + 1])) for i in range(w - 1, len(chains))}


def curve_nodes(dc: DayChain, mo_forward: Optional[float] = None) -> List[Tuple[float, float]]:
    """``(tenor, ln(F/S))``: every IM close, then the MO pillar (or ``mo_forward``)."""
    s = dc.spot
    nodes = [(q.tenor, float(safe_log(q.close / s))) for q in dc.im]
    f_mo = dc.mo.forward if mo_forward is None else float(mo_forward)
    nodes.append((dc.mo.tenor, float(safe_log(f_mo / s))))
    return nodes


def build_curve(dc: DayChain, tail: str, b_inf: Optional[float], *,
                mo_forward: Optional[float] = None) -> CarryCurve:
    """The day's ``LOG_LINEAR_DF`` carry curve under ``tail``.

    ``b_inf`` is required for the converging tails and ignored otherwise;
    ``mo_forward`` overrides the MO pillar's forward (switch-floor / robustness).
    """
    nodes = curve_nodes(dc, mo_forward)
    if tail in LIBRARY_TAILS:
        return CarryCurve(nodes, interpolation=CurveInterpolation.LOG_LINEAR_DF,
                          extrapolation=LIBRARY_TAILS[tail])
    if tail in CONVERGING_HALF_LIFE:
        if b_inf is None:
            raise ValidationError(f"{tail} needs an anchor b_inf (none on {dc.day})")
        return ConvergingCarryCurve(nodes, b_inf=b_inf, half_life=CONVERGING_HALF_LIFE[tail],
                                    interpolation=CurveInterpolation.LOG_LINEAR_DF)
    raise ValidationError(f"unknown tail {tail!r}; expected one of {TAILS}")


def pillar_inputs(dc: DayChain) -> Tuple[List[str], List[float]]:
    """Pillar labels and lot multipliers: 200 per IM lot, ``100·DF(T_n)`` per MO C−P pair."""
    labels = [q.code for q in dc.im] + [f"{dc.mo.series} C-P"]
    mults = [IM_MULT] * len(dc.im) + [MO_MULT * float(safe_exp(-RATE * dc.mo.tenor))]
    return labels, mults
