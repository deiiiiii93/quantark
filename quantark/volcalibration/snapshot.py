"""Canonical quote-snapshot envelope and lifters for the legacy MO shapes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Mapping, Optional, Tuple

from quantark.util.exceptions import ValidationError

SCHEMA_VERSION = 1

CONVENTION_LISTED = "listed_strike"
CONVENTION_FX_DELTA = "fx_delta"
CONVENTIONS = (CONVENTION_LISTED, CONVENTION_FX_DELTA)

PRICE_FIELD_SETTLEMENT = "settlement"
PRICE_FIELD_MID_OR_LAST = "mid_or_last"
PRICE_FIELDS = (PRICE_FIELD_SETTLEMENT, PRICE_FIELD_MID_OR_LAST)


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
    raise ValidationError(f"Cannot interpret {value!r} as a trade date")


def _positive(value, what: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{what} must be numeric, got {value!r}") from exc
    if not math.isfinite(out) or out <= 0.0:
        raise ValidationError(f"{what} must be positive and finite, got {out}")
    return out


@dataclass(frozen=True)
class QuoteSnapshot:
    """One trading date's option quotes in the canonical envelope."""

    schema_version: int
    convention: str
    trade_date: date
    symbol: str
    spot: float
    price_field: str
    source: Mapping[str, Any]
    expiries: Tuple[Dict[str, Any], ...]

    # ---------------------------------------------------------------- build
    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "QuoteSnapshot":
        """Validate a canonical snapshot payload; fail closed on any violation."""
        if not isinstance(payload, Mapping):
            raise ValidationError("snapshot payload must be a mapping")
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValidationError(
                f"snapshot schema_version must be {SCHEMA_VERSION}, got {version!r}"
            )
        convention = payload.get("convention")
        if convention not in CONVENTIONS:
            raise ValidationError(
                f"snapshot convention must be one of {CONVENTIONS}, got {convention!r}"
            )
        underlying = payload.get("underlying")
        if not isinstance(underlying, Mapping) or "spot" not in underlying:
            raise ValidationError("snapshot underlying must define 'spot'")
        source = payload.get("source")
        if not isinstance(source, Mapping):
            raise ValidationError("snapshot must define a 'source' object")
        price_field = source.get("price_field")
        if price_field not in PRICE_FIELDS:
            raise ValidationError(
                f"source.price_field must be one of {PRICE_FIELDS}, got {price_field!r}"
            )
        expiries = payload.get("expiries")
        if not isinstance(expiries, list):
            raise ValidationError("snapshot expiries must be a list")
        return cls(
            schema_version=SCHEMA_VERSION,
            convention=str(convention),
            trade_date=_as_date(payload.get("trade_date")),
            symbol=str(underlying.get("symbol", "")),
            spot=_positive(underlying["spot"], "underlying.spot"),
            price_field=str(price_field),
            source=dict(source),
            expiries=tuple(dict(e) for e in expiries),
        )

    @classmethod
    def from_legacy_live(
        cls,
        payload: Mapping[str, Any],
        *,
        trade_date,
        symbol: str,
        source_sha256: Optional[str] = None,
    ) -> "QuoteSnapshot":
        """Lift a live ``01_fetch_mo_snapshot`` payload into the canonical envelope.

        The live shape carries no trade date and no price-field declaration, so
        both are supplied by the caller rather than inferred.
        """
        underlying = payload.get("underlying")
        if not isinstance(underlying, Mapping) or "spot" not in underlying:
            raise ValidationError("live snapshot underlying missing 'spot'")
        expiries = payload.get("expiries")
        if not isinstance(expiries, list) or not expiries:
            raise ValidationError("live snapshot requires a non-empty 'expiries' list")
        return cls(
            schema_version=SCHEMA_VERSION,
            convention=CONVENTION_LISTED,
            trade_date=_as_date(trade_date),
            symbol=str(symbol),
            spot=_positive(underlying["spot"], "underlying.spot"),
            price_field=PRICE_FIELD_MID_OR_LAST,
            source={
                "vendor": "mo_live_snapshot",
                "price_field": PRICE_FIELD_MID_OR_LAST,
                "fetched_at": payload.get("fetched_at"),
                "market_open": payload.get("market_open"),
                "sha256": source_sha256,
            },
            expiries=tuple(dict(e) for e in expiries),
        )

    @classmethod
    def from_legacy_settlement(
        cls,
        payload: Mapping[str, Any],
        *,
        trade_date,
        spot,
        symbol: str,
        source_sha256: Optional[str] = None,
        source_url: Optional[str] = None,
    ) -> "QuoteSnapshot":
        """Lift a CFFEX settlement payload into the canonical envelope.

        The settlement shape carries no spot -- it comes from the separate spot
        CSV -- so ``spot`` is a required argument.  Nothing is inferred.

        ``source_url`` and ``source_sha256`` are fetcher provenance: snapshots
        written by the live pipeline carry them in the payload, while the
        committed samples are stripped of them, so the payload wins and the
        arguments fill in.  They are recorded in the artifact body, which is
        where the existing artifacts already keep them.
        """
        declared = payload.get("price_field")
        if declared != PRICE_FIELD_SETTLEMENT:
            raise ValidationError(
                "settlement snapshot must declare price_field "
                f"{PRICE_FIELD_SETTLEMENT!r}, got {declared!r}"
            )
        expiries = payload.get("expiries")
        if not isinstance(expiries, list) or not expiries:
            raise ValidationError(
                "settlement snapshot requires a non-empty 'expiries' list"
            )
        return cls(
            schema_version=SCHEMA_VERSION,
            convention=CONVENTION_LISTED,
            trade_date=_as_date(trade_date),
            symbol=str(symbol),
            spot=_positive(spot, "spot"),
            price_field=PRICE_FIELD_SETTLEMENT,
            source={
                "vendor": payload.get("source_class", "official_cffex_eod_settlement"),
                "price_field": PRICE_FIELD_SETTLEMENT,
                "record_count": payload.get("record_count"),
                "sha256": payload.get("source_sha256", source_sha256),
                "source_url": payload.get("source_url", source_url),
            },
            expiries=tuple(dict(e) for e in expiries),
        )

    # ------------------------------------------------------------ accessors
    @property
    def sha256(self) -> Optional[str]:
        """The source payload's sha256, as recorded by whoever fetched it."""
        value = self.source.get("sha256")
        return str(value) if value else None

    def to_payload(self) -> Dict[str, Any]:
        """The canonical envelope; round-trips through :meth:`from_payload`."""
        return {
            "schema_version": self.schema_version,
            "convention": self.convention,
            "trade_date": self.trade_date.isoformat(),
            "underlying": {"symbol": self.symbol, "spot": self.spot},
            "source": dict(self.source),
            "expiries": [dict(e) for e in self.expiries],
        }

    # ---------------------------------------------------------------- price
    def quote_price(self, quote: Mapping[str, Any]) -> float:
        """Resolve one quote's price through the declared ``price_field``.

        There is exactly one rule per price field and no cross-field fallback:
        a quote that cannot satisfy the declared field is a
        ``price_field_mismatch``, never silently priced off another key.
        """
        if self.price_field == PRICE_FIELD_SETTLEMENT:
            value = quote.get("settlement")
            if value is None:
                raise ValidationError(
                    "price_field_mismatch: snapshot declares "
                    f"{PRICE_FIELD_SETTLEMENT!r} but quote has no 'settlement' key"
                )
            return float(value)
        if self.price_field == PRICE_FIELD_MID_OR_LAST:
            bid, ask = quote.get("bid"), quote.get("ask")
            if bid is not None and ask is not None:
                bid_f, ask_f = float(bid), float(ask)
                if bid_f > 0.0 and ask_f > 0.0:
                    return 0.5 * (bid_f + ask_f)
            last = quote.get("last")
            if last is None:
                raise ValidationError(
                    "price_field_mismatch: snapshot declares "
                    f"{PRICE_FIELD_MID_OR_LAST!r} but quote has no usable "
                    "bid/ask pair and no 'last' key"
                )
            return float(last)
        raise ValidationError(f"unsupported price_field {self.price_field!r}")
