"""Strike-quoted listed-option normalizer: parity, OTM filter, IV inversion."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from quantark.util.exceptions import NumericalError, ValidationError
from quantark.util.numerical import safe_log
from quantark.volcalibration.quotes import ExpiryQuotes, IvNode, QuoteSet
from quantark.volcalibration.snapshot import QuoteSnapshot
from quantark.volmodels.black_scholes import implied_vol_call

# Parity quality gates, promoted from
# example/mo_volmodels/10_calibration_diagnostics.py.
MAX_ABS_PARITY_IMPLIED_RATE = 0.10
MAX_PARITY_RMSE_FORWARD_RATIO = 0.01

MIN_PARITY_PAIRS = 3
MIN_STRIKES_PER_EXPIRY = 5
MIN_USABLE_EXPIRIES = 2
DEFAULT_MONEYNESS_WIDTH = 0.35


@dataclass
class ExpirySlice:
    """One expiry's quotes, indexed by strike for calls and puts separately."""

    expiry_label: str
    expiry_date: Optional[str]
    T: float
    calls: Dict[float, float]
    puts: Dict[float, float]
    volume: Dict[Tuple[float, str], int] = field(default_factory=dict)


@dataclass(frozen=True)
class ParityResult:
    """Carry recovered from one expiry via put-call parity."""

    r: float
    forward: float
    discount_factor: float
    q: float
    n_pairs: int
    rmse_over_forward: float


def iter_expiries(snapshot: QuoteSnapshot) -> List[ExpirySlice]:
    """Reshape each expiry's flat quote list into strike-indexed maps."""
    out: List[ExpirySlice] = []
    for exp in snapshot.expiries:
        calls: Dict[float, float] = {}
        puts: Dict[float, float] = {}
        volume: Dict[Tuple[float, str], int] = {}
        for quote in exp["quotes"]:
            strike = float(quote["strike"])
            kind = quote["type"]
            price = snapshot.quote_price(quote)
            (calls if kind == "C" else puts)[strike] = price
            volume[(strike, kind)] = int(quote.get("volume", 0) or 0)
        label = str(exp.get("expiry_date") or exp.get("contract_month"))
        out.append(
            ExpirySlice(
                expiry_label=label,
                expiry_date=exp.get("expiry_date"),
                T=float(exp["T_years"]),
                calls=calls,
                puts=puts,
                volume=volume,
            )
        )
    return out


def imply_forward_and_rate(sl: ExpirySlice, s0: float) -> ParityResult:
    """Recover ``(r, F, DF, q)`` for one expiry from put-call parity.

    Parity is model-free: ``C(K) - P(K) = DF * (F - K)`` is a straight line in
    ``K`` with slope ``-DF`` and intercept ``DF * F``, so one OLS fit yields the
    market discount factor and the forward together.  A slice that violates the
    quality gates is rejected, never repaired.
    """
    pairs = sorted(set(sl.calls) & set(sl.puts))
    if len(pairs) < MIN_PARITY_PAIRS:
        raise ValidationError(
            f"expiry {sl.expiry_label}: only {len(pairs)} paired strikes "
            f"(< {MIN_PARITY_PAIRS})"
        )
    K = np.asarray(pairs, dtype=float)
    y = np.asarray([sl.calls[k] - sl.puts[k] for k in pairs], dtype=float)
    slope, intercept = np.polyfit(K, y, 1)
    df = float(-slope)
    if df <= 0.0:
        raise ValidationError(
            f"expiry {sl.expiry_label}: non-positive discount factor {df:.4g} "
            "(arbitrage-violating quotes) — excluded, not fabricated"
        )
    forward = float(intercept / df)
    if not math.isfinite(forward) or forward <= 0.0:
        raise ValidationError(
            f"expiry {sl.expiry_label}: non-positive implied forward {forward:.4g}"
        )
    residual = y - (slope * K + intercept)
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    rmse_over_forward = rmse / forward
    if rmse_over_forward > MAX_PARITY_RMSE_FORWARD_RATIO:
        raise ValidationError(
            f"expiry {sl.expiry_label}: parity RMSE/forward "
            f"{rmse_over_forward:.4g} exceeds {MAX_PARITY_RMSE_FORWARD_RATIO}"
        )
    r = float(-safe_log(df) / sl.T)
    if abs(r) > MAX_ABS_PARITY_IMPLIED_RATE:
        raise ValidationError(
            f"expiry {sl.expiry_label}: implied rate {r:.4g} exceeds "
            f"+/-{MAX_ABS_PARITY_IMPLIED_RATE}"
        )
    q = float(r - safe_log(forward / s0) / sl.T)
    if not math.isfinite(q):
        raise ValidationError(f"expiry {sl.expiry_label}: non-finite carry q={q}")
    return ParityResult(
        r=r,
        forward=forward,
        discount_factor=df,
        q=q,
        n_pairs=len(pairs),
        rmse_over_forward=rmse_over_forward,
    )


@dataclass(frozen=True)
class OtmQuote:
    """A single out-of-the-money quote surviving the liquidity filter."""

    strike: float
    kind: str  # "C" or "P"
    price: float


def select_otm(sl: ExpirySlice, forward: float, min_volume: int = 1) -> List[OtmQuote]:
    """Keep only OTM options: puts below the forward, calls at/above it.

    Only OTM options carry clean volatility information.  Deep-ITM quotes are
    dominated by intrinsic value and are typically stale and wide, so a small
    price error there becomes a large IV error.
    """
    out: List[OtmQuote] = []
    for k in sorted(set(sl.calls) | set(sl.puts)):
        kind = "P" if k < forward else "C"
        book = sl.puts if kind == "P" else sl.calls
        if k not in book:
            continue
        if sl.volume.get((k, kind), 0) < min_volume:
            continue
        price = book[k]
        if price <= 0.0:
            continue
        out.append(OtmQuote(strike=k, kind=kind, price=price))
    return out


def otm_implied_vol(oq, s0, r, q_carry, forward, discount_factor, T):
    """Invert an OTM quote to Black IV via its call-equivalent price.

    An OTM put becomes the call at the same strike through parity,
    ``C = P + DF*(F - K)``, so one call inverter serves both wings and the two
    wings agree at the forward by construction -- the no-arbitrage property the
    Dupire builder needs.  A quote outside the no-arb band yields ``None``:
    excluded, never fabricated.
    """
    if oq.kind == "P":
        call_equiv = oq.price + discount_factor * (forward - oq.strike)
    else:
        call_equiv = oq.price
    try:
        return implied_vol_call(s0, oq.strike, T, call_equiv, r, q_carry)
    except NumericalError:
        return None


class ListedNormalizer:
    """Normalize a strike-quoted listed-option snapshot into a QuoteSet."""

    convention = "listed_strike"

    def __init__(
        self,
        *,
        min_volume: int = 1,
        min_strikes_per_expiry: int = MIN_STRIKES_PER_EXPIRY,
        moneyness_width: float = DEFAULT_MONEYNESS_WIDTH,
    ) -> None:
        if min_strikes_per_expiry < 3:
            raise ValidationError("min_strikes_per_expiry must be >= 3")
        if moneyness_width <= 0.0:
            raise ValidationError("moneyness_width must be positive")
        self._min_volume = int(min_volume)
        self._min_strikes = int(min_strikes_per_expiry)
        self._width = float(moneyness_width)

    def normalize(self, snapshot: QuoteSnapshot) -> QuoteSet:
        """Parity -> OTM filter -> IV inversion, one ExpiryQuotes per usable expiry.

        An expiry that fails parity or falls under the strike floor is dropped,
        not repaired; if fewer than two survive, the whole snapshot is rejected.
        """
        s0 = snapshot.spot
        expiries: List[ExpiryQuotes] = []
        for sl in iter_expiries(snapshot):
            try:
                par = imply_forward_and_rate(sl, s0)
            except ValidationError:
                # A rejected expiry is dropped; the failure surfaces below if
                # too few survive.
                continue
            nodes: List[IvNode] = []
            for oq in select_otm(sl, par.forward, min_volume=self._min_volume):
                iv = otm_implied_vol(
                    oq, s0, par.r, par.q, par.forward, par.discount_factor, sl.T
                )
                if iv is None or not (0.0 < iv < 2.0):
                    continue
                weight = math.exp(
                    -0.5 * (math.log(oq.strike / par.forward) / self._width) ** 2
                )
                nodes.append(IvNode(strike=oq.strike, iv=float(iv), weight_hint=weight))
            if len(nodes) < self._min_strikes:
                continue
            nodes.sort(key=lambda n: n.strike)
            expiries.append(
                ExpiryQuotes(
                    expiry_label=sl.expiry_label,
                    expiry_date=sl.expiry_date,
                    T=sl.T,
                    forward=par.forward,
                    discount_factor=par.discount_factor,
                    r=par.r,
                    q=par.q,
                    nodes=tuple(nodes),
                    diagnostics={
                        "n_pairs": float(par.n_pairs),
                        "parity_rmse_over_forward": par.rmse_over_forward,
                        "n_nodes": float(len(nodes)),
                    },
                )
            )
        if len(expiries) < MIN_USABLE_EXPIRIES:
            raise ValidationError(
                f"{snapshot.trade_date.isoformat()}: only {len(expiries)} usable "
                f"expiries (need >= {MIN_USABLE_EXPIRIES}) after parity and the "
                f"{self._min_strikes}-strike floor"
            )
        expiries.sort(key=lambda e: e.T)
        return QuoteSet(
            trade_date=snapshot.trade_date,
            spot=s0,
            convention=self.convention,
            expiries=tuple(expiries),
        )
