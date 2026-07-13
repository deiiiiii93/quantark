"""Option-quote ingestion and cleaning (spec WP4.1).

Input = already-frozen snapshot rows (live fetching stays in the consumer
repo). The pipeline, with normative defaults (each step configurable):

1. validity      — drop bid<=0, ask<=0, or crossed (bid>=ask); price source
                   is mid=(bid+ask)/2; ``last`` only if both sides missing
                   AND allow_last_price (default False -> drop).
2. liquidity     — keep if volume>=1 OR open_interest>=10; drop when the
                   relative spread (ask-bid)/mid exceeds 20%.
3. OTM selection — calls with K >= F(0,T), puts with K < F(0,T); call/put
                   unification via forward parity with the SAME D/F as Q2.
4. no-arb bounds — mid within the European bounds priced off D and F.
5. implied vol   — Black-76 with D, F; Brent on sigma in [1e-4, 5.0];
                   solver failure -> exclusion. Expiry time = ACT/365F to
                   15:00 local (intraday fraction matters only same-day).
6. weights       — vega-weighted (default) or spread-weighted.

Every dropped quote is recorded in the exclusion log with its reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import erf, exp, log, pi, sqrt
from typing import Dict, Optional, Tuple

from scipy.optimize import brentq

from quantark.util.exceptions import ValidationError

DAYS_PER_YEAR = 365.0            # ACT/365F fixed by the problem
EXPIRY_HOUR_FRACTION = 15.0 / 24.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def black_price(
    forward: float, strike: float, expiry_t: float, sigma: float,
    df: float, is_call: bool,
) -> float:
    """Black-76 price on the forward."""
    if expiry_t <= 0.0 or sigma <= 0.0:
        intrinsic = (forward - strike) if is_call else (strike - forward)
        return df * max(intrinsic, 0.0)
    s = sigma * sqrt(expiry_t)
    d1 = (log(forward / strike) + 0.5 * s * s) / s
    d2 = d1 - s
    if is_call:
        return df * (forward * _norm_cdf(d1) - strike * _norm_cdf(d2))
    return df * (strike * _norm_cdf(-d2) - forward * _norm_cdf(-d1))


def black_vega(
    forward: float, strike: float, expiry_t: float, sigma: float, df: float
) -> float:
    if expiry_t <= 0.0 or sigma <= 0.0:
        return 0.0
    s = sigma * sqrt(expiry_t)
    d1 = (log(forward / strike) + 0.5 * s * s) / s
    return df * forward * _norm_pdf(d1) * sqrt(expiry_t)


def implied_vol_black(
    price: float, forward: float, strike: float, expiry_t: float,
    df: float, is_call: bool, bracket: Tuple[float, float] = (1e-4, 5.0),
) -> float:
    """Invert Black-76 with Brent; raises ValidationError when unbracketable."""
    lo, hi = bracket

    def objective(sigma: float) -> float:
        return black_price(forward, strike, expiry_t, sigma, df, is_call) - price

    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo * f_hi > 0.0:
        raise ValidationError(
            f"implied vol not bracketed in [{lo}, {hi}] for price {price}"
        )
    return float(brentq(objective, lo, hi, xtol=1e-12, rtol=1e-12))


@dataclass(frozen=True)
class OptionQuote:
    expiry: datetime
    strike: float
    call_put: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[float] = None
    open_interest: Optional[float] = None
    quote_time: Optional[datetime] = None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["expiry"] = self.expiry.isoformat()
        d["quote_time"] = (
            self.quote_time.isoformat() if self.quote_time else None
        )
        return d


@dataclass(frozen=True)
class QuoteCleaningConfig:
    allow_last_price: bool = False
    min_volume: float = 1.0
    min_open_interest: float = 10.0
    max_rel_spread: float = 0.20
    iv_bracket: Tuple[float, float] = (1e-4, 5.0)
    weight_scheme: str = "vega"          # "vega" | "spread"
    expiry_hour_fraction: float = EXPIRY_HOUR_FRACTION


@dataclass(frozen=True)
class CleanedQuote:
    expiry_t: float
    strike: float
    log_moneyness: float   # y = ln(K / F(0,T))
    iv: float
    weight: float
    source: OptionQuote

    def to_dict(self) -> dict:
        return {
            "expiry_t": self.expiry_t,
            "strike": self.strike,
            "log_moneyness": self.log_moneyness,
            "iv": self.iv,
            "weight": self.weight,
            "source": self.source.to_dict(),
        }


@dataclass(frozen=True)
class CleanedQuoteSet:
    valuation_date: datetime
    spot: float
    slices: Dict[float, Tuple[CleanedQuote, ...]]     # keyed by expiry_t
    exclusions: Tuple[Tuple[OptionQuote, str], ...]   # (quote, reason)
    forwards: Dict[float, float]                      # F(0,T) per slice
    dfs: Dict[float, float]                           # D(0,T) per slice
    config: QuoteCleaningConfig = field(default_factory=QuoteCleaningConfig)

    @property
    def all_quotes(self) -> Tuple[CleanedQuote, ...]:
        return tuple(
            q for t in sorted(self.slices) for q in self.slices[t]
        )

    def to_dict(self) -> dict:
        return {
            "valuation_date": self.valuation_date.isoformat(),
            "spot": self.spot,
            "used": [q.to_dict() for q in self.all_quotes],
            "excluded": [
                {"quote": q.to_dict(), "reason": reason}
                for q, reason in self.exclusions
            ],
            "forwards": {f"{t:g}": f for t, f in self.forwards.items()},
            "dfs": {f"{t:g}": d for t, d in self.dfs.items()},
        }


def _normalize_call_put(raw: str) -> bool:
    """True = call, False = put."""
    token = str(raw).strip().lower()
    if token in ("c", "call"):
        return True
    if token in ("p", "put"):
        return False
    raise ValidationError(f"call_put must be one of C/P/call/put, got {raw!r}")


def _expiry_year_fraction(
    valuation_date: datetime, expiry: datetime, hour_fraction: float
) -> float:
    days = (expiry.date() - valuation_date.date()).days
    return (days + hour_fraction) / DAYS_PER_YEAR if days == 0 else (
        days / DAYS_PER_YEAR
    )


def clean_and_imply(
    quotes,
    valuation_date: datetime,
    spot: float,
    rate_curve,
    carry_curve,
    config: Optional[QuoteCleaningConfig] = None,
) -> CleanedQuoteSet:
    cfg = config or QuoteCleaningConfig()
    slices: Dict[float, list] = {}
    exclusions: list = []
    forwards: Dict[float, float] = {}
    dfs: Dict[float, float] = {}

    for quote in quotes:
        is_call = _normalize_call_put(quote.call_put)
        expiry_t = _expiry_year_fraction(
            valuation_date, quote.expiry, cfg.expiry_hour_fraction
        )
        if expiry_t <= 0.0:
            exclusions.append((quote, "expired"))
            continue
        if expiry_t not in forwards:
            forwards[expiry_t] = float(carry_curve.forward(spot, expiry_t))
            dfs[expiry_t] = float(rate_curve.get_discount_factor(expiry_t))
        fwd, df = forwards[expiry_t], dfs[expiry_t]

        # 1) validity + price source
        bid, ask = quote.bid, quote.ask
        if bid is not None and ask is not None:
            if bid <= 0.0 or ask <= 0.0 or bid >= ask:
                exclusions.append((quote, "crossed_or_zero"))
                continue
            mid = 0.5 * (bid + ask)
        elif cfg.allow_last_price and quote.last is not None and quote.last > 0:
            mid = float(quote.last)
        else:
            exclusions.append((quote, "no_price"))
            continue

        # 2) liquidity
        volume = quote.volume or 0.0
        open_interest = quote.open_interest or 0.0
        if volume < cfg.min_volume and open_interest < cfg.min_open_interest:
            exclusions.append((quote, "illiquid"))
            continue
        if bid is not None and ask is not None:
            if (ask - bid) / mid > cfg.max_rel_spread:
                exclusions.append((quote, "wide_spread"))
                continue

        # 3) OTM selection (parity unification with the same D/F)
        if is_call and quote.strike < fwd:
            exclusions.append((quote, "itm_side"))
            continue
        if not is_call and quote.strike >= fwd:
            exclusions.append((quote, "itm_side"))
            continue

        # 4) European no-arb bounds
        intrinsic = max(
            (fwd - quote.strike) if is_call else (quote.strike - fwd), 0.0
        )
        upper = df * (fwd if is_call else quote.strike)
        if mid < df * intrinsic:
            exclusions.append((quote, "below_intrinsic"))
            continue
        if mid > upper:
            exclusions.append((quote, "above_upper_bound"))
            continue

        # 5) implied vol
        try:
            iv = implied_vol_black(
                mid, fwd, quote.strike, expiry_t, df, is_call,
                bracket=cfg.iv_bracket,
            )
        except (ValidationError, ValueError):
            exclusions.append((quote, "iv_solve_failed"))
            continue

        # 6) weight
        if cfg.weight_scheme == "vega":
            weight = black_vega(fwd, quote.strike, expiry_t, iv, df)
        elif cfg.weight_scheme == "spread":
            tick = 1e-4 * mid
            weight = 1.0 / max((ask - bid) if bid is not None else tick, tick)
        else:
            raise ValidationError(
                f"unknown weight_scheme: {cfg.weight_scheme!r}"
            )
        slices.setdefault(expiry_t, []).append(
            CleanedQuote(
                expiry_t=expiry_t,
                strike=float(quote.strike),
                log_moneyness=float(log(quote.strike / fwd)),
                iv=iv,
                weight=float(weight),
                source=quote,
            )
        )

    return CleanedQuoteSet(
        valuation_date=valuation_date,
        spot=float(spot),
        slices={
            t: tuple(sorted(qs, key=lambda q: q.strike))
            for t, qs in slices.items()
        },
        exclusions=tuple(exclusions),
        forwards={t: forwards[t] for t in slices},
        dfs={t: dfs[t] for t in slices},
        config=cfg,
    )
