"""Settlement-quoted listed-option normalizer (official exchange EOD marks).

Distinct from :mod:`~quantark.volcalibration.normalize.listed`, which handles
live bid/ask/last books.  Settlement marks are not executable quotes, so this
path is stricter and fully auditable:

* the expiry calendar is verified (third Friday) rather than trusted;
* maturity is derived as ``calendar_days / 365``, not read from the payload;
* a node needs BOTH positive volume and positive open interest;
* IV is inverted in normalized units (``S=1``, ``K/F``, ``C/(DF*F)``, r=q=0),
  which removes the carry from the inversion entirely;
* every quote that does not become a node is counted under a named reason, and
  every expiry that drops out is recorded -- nothing disappears silently.

Ported from ``example/mo_volmodels/10_calibration_diagnostics.py``
(``build_calibration_nodes``), which built the admitted MO surface history.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date, timedelta
from typing import Any, Dict, List

import numpy as np

from quantark.util.exceptions import NumericalError
from quantark.volcalibration.admission import AdmissionError, AdmissionReason
from quantark.volcalibration.quotes import ExpiryQuotes, IvNode, QuoteSet
from quantark.volcalibration.snapshot import QuoteSnapshot
from quantark.volmodels.black_scholes import implied_vol_call

MIN_CALENDAR_DAYS = 7
MAX_CALENDAR_DAYS = 365
MIN_PARITY_PAIRS = 3
MAX_ABS_PARITY_IMPLIED_RATE = 0.10
MAX_PARITY_RMSE_FORWARD_RATIO = 0.01

# CFFEX shifts expiry off the third Friday when it falls on a holiday.  The
# generic rule cannot predict these, so known shifts are tabulated; an unknown
# shift surfaces as a third-Friday mismatch rather than a silently wrong
# maturity.  Ported from 10_calibration_diagnostics.EXPIRY_DATE_OVERRIDES.
EXPIRY_DATE_OVERRIDES = {"2606": date(2026, 6, 22)}


def third_friday(year_month: str) -> date:
    """Expiry date of a ``YYMM`` contract month: third Friday, or a known shift."""
    if not re.fullmatch(r"\d{4}", str(year_month)):
        raise AdmissionError(
            AdmissionReason.INVALID_SNAPSHOT,
            f"contract_month must be YYMM, got {year_month!r}",
        )
    year = 2000 + int(str(year_month)[:2])
    month = int(str(year_month)[2:])
    if not 1 <= month <= 12:
        raise AdmissionError(
            AdmissionReason.INVALID_SNAPSHOT, f"invalid contract_month {year_month!r}"
        )
    cursor = date(year, month, 1)
    fridays: List[date] = []
    while cursor.month == month:
        if cursor.weekday() == 4:
            fridays.append(cursor)
        cursor += timedelta(days=1)
    return EXPIRY_DATE_OVERRIDES.get(str(year_month), fridays[2])


def _parse_date(value, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise AdmissionError(
            AdmissionReason.INVALID_SNAPSHOT, f"{field}: cannot parse {value!r}"
        ) from exc


def _finite_positive(value, name: str) -> float:
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {out}")
    return out


class SettlementNormalizer:
    """Normalize an exchange settlement snapshot into a QuoteSet plus audit."""

    convention = "listed_strike"

    def __init__(self, *, min_expiries: int = 2) -> None:
        self._min_expiries = int(min_expiries)

    def normalize(self, snapshot: QuoteSnapshot) -> QuoteSet:
        """Parity -> OTM/liquidity filter -> normalized IV inversion, per expiry."""
        trade_date = snapshot.trade_date
        excluded: List[Dict[str, Any]] = []
        filtered: Counter = Counter()
        expiries: List[ExpiryQuotes] = []
        total_nodes = 0

        for expiry in snapshot.expiries:
            contract_month = str(expiry.get("contract_month", ""))
            expected = third_friday(contract_month)
            supplied = _parse_date(expiry.get("expiry_date"), "expiry_date")
            if supplied != expected:
                raise AdmissionError(
                    AdmissionReason.INVALID_SNAPSHOT,
                    f"{contract_month}: expiry_date {supplied} is not third Friday "
                    f"{expected}",
                )
            calendar_days = (expected - trade_date).days
            base = {
                "contract_month": contract_month,
                "expiry_date": expected.isoformat(),
                "calendar_days": calendar_days,
            }
            if not MIN_CALENDAR_DAYS <= calendar_days <= MAX_CALENDAR_DAYS:
                excluded.append({**base, "reason": "outside_maturity_window"})
                continue
            maturity = calendar_days / 365.0

            by_strike: Dict[float, Dict[str, dict]] = {}
            for quote in expiry.get("quotes", []):
                option_type = str(quote.get("type", ""))
                if option_type not in {"C", "P"}:
                    raise AdmissionError(
                        AdmissionReason.INVALID_SNAPSHOT,
                        f"{contract_month}: quote type must be C or P",
                    )
                strike = _finite_positive(quote.get("strike"), "strike")
                if option_type in by_strike.setdefault(strike, {}):
                    raise AdmissionError(
                        AdmissionReason.INVALID_SNAPSHOT,
                        f"{contract_month}: duplicate {option_type} quote at "
                        f"strike {strike}",
                    )
                by_strike[strike][option_type] = dict(quote)

            paired = sorted(k for k, q in by_strike.items() if set(q) >= {"C", "P"})
            if len(paired) < MIN_PARITY_PAIRS:
                excluded.append(
                    {**base, "reason": "fewer_than_three_settlement_pairs"}
                )
                continue

            strikes_for_parity: List[float] = []
            differences: List[float] = []
            for strike in paired:
                try:
                    call_price = _finite_positive(
                        snapshot.quote_price(by_strike[strike]["C"]), "call settlement"
                    )
                    put_price = _finite_positive(
                        snapshot.quote_price(by_strike[strike]["P"]), "put settlement"
                    )
                except ValueError:
                    filtered["invalid_parity_settlement"] += 1
                    continue
                strikes_for_parity.append(strike)
                differences.append(call_price - put_price)
            if len(strikes_for_parity) < MIN_PARITY_PAIRS:
                excluded.append(
                    {**base, "reason": "fewer_than_three_valid_settlement_pairs"}
                )
                continue

            strike_array = np.asarray(strikes_for_parity, dtype=float)
            difference_array = np.asarray(differences, dtype=float)
            slope, intercept = np.polyfit(strike_array, difference_array, 1)
            discount_factor = -float(slope)
            if not math.isfinite(discount_factor) or discount_factor <= 0.0:
                excluded.append(
                    {
                        **base,
                        "reason": "non_positive_parity_discount_factor",
                        "discount_factor": discount_factor,
                    }
                )
                continue
            forward = float(intercept / discount_factor)
            if not math.isfinite(forward) or forward <= 0.0:
                excluded.append(
                    {**base, "reason": "non_positive_parity_forward", "forward": forward}
                )
                continue

            residuals = difference_array - (
                -discount_factor * strike_array + discount_factor * forward
            )
            implied_rate = -math.log(discount_factor) / maturity
            parity_rmse_points = float(np.sqrt(np.mean(np.square(residuals))))
            parity_rmse_ratio = parity_rmse_points / forward
            if (
                abs(implied_rate) > MAX_ABS_PARITY_IMPLIED_RATE
                or parity_rmse_ratio > MAX_PARITY_RMSE_FORWARD_RATIO
            ):
                excluded.append(
                    {
                        **base,
                        "T": maturity,
                        "pair_count": len(strikes_for_parity),
                        "forward": forward,
                        "discount_factor": discount_factor,
                        "implied_rate": implied_rate,
                        "parity_rmse_points": parity_rmse_points,
                        "parity_rmse_forward_ratio": parity_rmse_ratio,
                        "reason": "parity_quality_gate_failed",
                        "maximum_absolute_implied_rate": MAX_ABS_PARITY_IMPLIED_RATE,
                        "maximum_rmse_forward_ratio": MAX_PARITY_RMSE_FORWARD_RATIO,
                    }
                )
                continue

            nodes: List[IvNode] = []
            wing = Counter()
            for strike in paired:
                option_type = "P" if strike < forward else "C"
                quote = by_strike[strike][option_type]
                try:
                    option_price = _finite_positive(
                        snapshot.quote_price(quote), "selected settlement"
                    )
                except ValueError:
                    filtered["invalid_selected_settlement"] += 1
                    continue
                try:
                    volume = int(quote.get("volume", 0))
                    open_interest = int(quote.get("oi", 0))
                except (TypeError, ValueError) as exc:
                    raise AdmissionError(
                        AdmissionReason.INVALID_SNAPSHOT,
                        f"{quote.get('contract')}: invalid volume/OI",
                    ) from exc
                if volume <= 0:
                    filtered["zero_selected_volume"] += 1
                    continue
                if open_interest <= 0:
                    filtered["zero_selected_open_interest"] += 1
                    continue
                call_equivalent = (
                    option_price + discount_factor * (forward - strike)
                    if option_type == "P"
                    else option_price
                )
                normalized_strike = strike / forward
                normalized_price = call_equivalent / (discount_factor * forward)
                if not (
                    math.isfinite(normalized_price)
                    and max(1.0 - normalized_strike, 0.0) < normalized_price < 1.0
                ):
                    filtered["normalized_price_outside_no_arbitrage_bounds"] += 1
                    continue
                try:
                    implied_vol = implied_vol_call(
                        1.0, normalized_strike, maturity, normalized_price, 0.0, 0.0
                    )
                except (NumericalError, ValueError, OverflowError):
                    filtered["iv_inversion_failure"] += 1
                    continue
                if not math.isfinite(implied_vol) or not 0.0 < implied_vol < 2.0:
                    filtered["iv_outside_sanity_range"] += 1
                    continue
                wing[option_type] += 1
                nodes.append(
                    IvNode(strike=strike, iv=float(implied_vol), weight_hint=1.0)
                )

            if not nodes:
                excluded.append({**base, "reason": "no_usable_otm_nodes"})
                continue
            if wing["P"] == 0 or wing["C"] == 0:
                excluded.append(
                    {
                        **base,
                        "reason": "missing_liquid_otm_wing",
                        "put_node_count": wing["P"],
                        "call_node_count": wing["C"],
                    }
                )
                continue

            # Each expiry carries total objective weight one, so a dense front
            # ladder cannot dominate the fit.
            weight = 1.0 / len(nodes)
            nodes = [IvNode(n.strike, n.iv, weight) for n in nodes]
            nodes.sort(key=lambda n: n.strike)
            total_nodes += len(nodes)
            r = -math.log(discount_factor) / maturity
            q = r - math.log(forward / snapshot.spot) / maturity
            expiries.append(
                ExpiryQuotes(
                    expiry_label=expected.isoformat(),
                    expiry_date=expected.isoformat(),
                    T=maturity,
                    forward=forward,
                    discount_factor=discount_factor,
                    r=r,
                    q=q,
                    nodes=tuple(nodes),
                    diagnostics={
                        "n_pairs": float(len(strikes_for_parity)),
                        "parity_rmse_points": parity_rmse_points,
                        "parity_rmse_over_forward": parity_rmse_ratio,
                        "implied_rate": implied_rate,
                        "n_nodes": float(len(nodes)),
                    },
                )
            )

        if len(expiries) < self._min_expiries:
            raise AdmissionError(
                AdmissionReason.INSUFFICIENT_EXPIRIES,
                f"{trade_date.isoformat()}: {len(expiries)} usable expiries "
                f"(need >= {self._min_expiries}); excluded={excluded}",
            )
        expiries.sort(key=lambda e: e.T)
        return QuoteSet(
            trade_date=trade_date,
            spot=snapshot.spot,
            convention=self.convention,
            expiries=tuple(expiries),
            universe={
                "node_count": total_nodes,
                "expiry_count": len(expiries),
                "filtered_quote_counts": dict(filtered),
                "excluded_expiries": excluded,
            },
        )
