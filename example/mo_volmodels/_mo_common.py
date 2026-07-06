"""Shared helpers for the MO vol-model suite: snapshot IO, put-call parity, OTM
filtering, Black-IV inversion, and plotting. Pure quantark (no akshare)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from quantark.util.numerical import safe_log
from quantark.volmodels.black_scholes import implied_vol_call
from quantark.util.exceptions import NumericalError

_REQUIRED = {"fetched_at", "underlying", "expiries"}


def load_snapshot(path) -> dict:
    """Load and lightly validate an MO snapshot JSON."""
    snap = json.loads(Path(path).read_text())
    missing = _REQUIRED - set(snap)
    if missing:
        raise ValueError(f"snapshot missing keys: {sorted(missing)}")
    if "spot" not in snap.get("underlying", {}):
        raise ValueError("snapshot underlying missing 'spot'")
    return snap


@dataclass
class ExpirySlice:
    """One expiry's quotes, indexed by strike for calls and puts separately."""

    expiry_date: str
    T: float
    calls: Dict[float, float]  # strike -> last price
    puts: Dict[float, float]
    volume: Dict[Tuple[float, str], int] = field(default_factory=dict)


def _quote_price(q: dict) -> float:
    """Price used for a quote: bid/ask mid when both sides are live, else last.

    Real MO deep quotes can carry a stale 'last' far from the current book; the mid is
    the cleaner mark. The synthetic sample sets bid/ask symmetrically around last, so the
    mid equals last there — this choice is neutral on the sample and better on live data.
    """
    bid, ask = q.get("bid"), q.get("ask")
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return 0.5 * (float(bid) + float(ask))
    return float(q["last"])


def iter_expiries(snapshot: dict) -> List[ExpirySlice]:
    """Reshape the flat quote list of each expiry into strike-indexed maps."""
    out: List[ExpirySlice] = []
    for exp in snapshot["expiries"]:
        calls: Dict[float, float] = {}
        puts: Dict[float, float] = {}
        vol: Dict[Tuple[float, str], int] = {}
        for q in exp["quotes"]:
            (calls if q["type"] == "C" else puts)[float(q["strike"])] = _quote_price(q)
            vol[(float(q["strike"]), q["type"])] = int(q.get("volume", 0))
        out.append(ExpirySlice(exp["expiry_date"], float(exp["T_years"]), calls, puts, vol))
    return out


@dataclass
class ParityResult:
    """Carry recovered from one expiry via put-call parity."""

    r: float
    forward: float
    discount_factor: float
    q: float
    n_pairs: int


def imply_forward_and_rate(sl: "ExpirySlice", s0: float) -> "ParityResult":
    """Recover (r, forward, DF, q) for one expiry from put-call parity.

    Model: for each paired strike K,  C(K) - P(K) = DF * (F - K),
    which is linear in K with slope = -DF and intercept = DF*F.
    """
    pairs = sorted(set(sl.calls) & set(sl.puts))
    if len(pairs) < 3:
        raise ValueError(f"expiry {sl.expiry_date}: only {len(pairs)} paired strikes (<3)")
    K = np.array(pairs)
    y = np.array([sl.calls[k] - sl.puts[k] for k in pairs])
    # Put-call parity is model-free: C - P = DF*(F - K) is a straight line in K.
    # Slope = -DF, intercept = DF*F. A single OLS fit yields both the market
    # discount factor and the forward. (Near-ATM weighting is possible via the
    # `w=` arg of polyfit; OLS is used here — clean on arbitrage-consistent quotes.)
    a, b = np.polyfit(K, y, 1)
    df = -a
    if df <= 0.0:
        raise ValueError(
            f"expiry {sl.expiry_date}: non-positive discount factor {df:.4g} "
            "(arbitrage-violating quotes) — excluded, not fabricated"
        )
    forward = float(b / df)
    r = float(-safe_log(df) / sl.T)
    q = float(r - safe_log(forward / s0) / sl.T)
    return ParityResult(r=r, forward=forward, discount_factor=float(df), q=q, n_pairs=len(pairs))


@dataclass
class OtmQuote:
    """A single out-of-the-money quote surviving the liquidity filter."""

    strike: float
    kind: str  # "C" or "P"
    price: float


def select_otm(sl: "ExpirySlice", forward: float, min_volume: int = 1) -> List["OtmQuote"]:
    """Keep only OTM options: puts below the forward, calls at/above it, liquid & sane.

    The desk convention: only OTM options carry clean volatility information. Deep-ITM
    quotes are dominated by intrinsic value and are typically stale/wide, so a small
    pricing error there is a large IV error. We therefore take the put wing below the
    forward and the call wing at/above it.
    """
    out: List[OtmQuote] = []
    strikes = sorted(set(sl.calls) | set(sl.puts))
    for k in strikes:
        kind = "P" if k < forward else "C"
        book = sl.puts if kind == "P" else sl.calls
        if k not in book:
            continue  # that side not quoted at this strike
        if sl.volume.get((k, kind), 0) < min_volume:
            continue  # illiquid / no trades
        price = book[k]
        if price <= 0.0:
            continue  # non-positive quote
        out.append(OtmQuote(strike=k, kind=kind, price=price))
    return out


def otm_implied_vol(oq: "OtmQuote", s0, r, q_carry, forward, discount_factor, T):
    """Invert an OTM quote to Black IV via its call-equivalent price. None if uninvertible.

    An OTM put is turned into the price of the call at the same strike using put-call
    parity, C = P + DF*(F - K). A single call inverter then handles both wings, and the
    put/call smiles agree at the forward by construction — the no-arbitrage property the
    Dupire builder needs. A quote outside the no-arb band yields None (excluded, never
    fabricated) per the project's no-fallback rule.
    """
    if oq.kind == "P":
        call_equiv = oq.price + discount_factor * (forward - oq.strike)
    else:
        call_equiv = oq.price
    try:
        return implied_vol_call(s0, oq.strike, T, call_equiv, r, q_carry)
    except NumericalError:
        return None


def build_env(surface_json: dict):
    """Reconstruct (PricingEnvironment, GridVolSurface, s0) from stage-02 output.

    The rate curve and dividend/carry curve are built from the per-expiry parity
    pillars, so the pricing environment carries the exact term structure the surface
    was calibrated against. Both curve types require >= 2 pillars (guaranteed by the
    stage-02 >=2-expiry check).
    """
    from datetime import datetime

    from quantark.param import GridVolSurface, SpotQuote
    from quantark.param.rrf.rate_curve import LinearRateCurve
    from quantark.param.div import TermStructureDividendYield
    from quantark.priceenv import PricingEnvironment

    s0 = float(surface_json["s0"])
    surf = GridVolSurface(surface_json["strikes"], surface_json["maturities"],
                          np.array(surface_json["iv_grid"]))
    pe = surface_json["per_expiry"]
    ts = [p["T"] for p in pe]
    rate = LinearRateCurve([(p["T"], p["r"]) for p in pe])
    div = TermStructureDividendYield(times=ts, yields=[p["q"] for p in pe])
    env = PricingEnvironment(rate_curve=rate, valuation_date=datetime(2026, 7, 6),
                             spot_quote=SpotQuote(spot=s0), vol_surface=surf, div_yield=div)
    return env, surf, s0


def plot_smiles(rows, path, title="MO implied-vol smiles"):
    """rows = list of (label, strikes, ivs). Saves a PNG; returns the path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, strikes, ivs in rows:
        ax.plot(strikes, np.array(ivs) * 100, marker="o", ms=3, label=label)
    ax.set_xlabel("strike")
    ax.set_ylabel("implied vol (%)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
