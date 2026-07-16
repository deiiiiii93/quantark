"""End-to-end prepared-dispatch mutation guard (code-gate finding 2026-07-16).

A prepared adapter clones the engine and injects artifacts built from ONE
market state; execution then reads the live product/environment. Without a
post-execution verification, a concurrent market mutation between prepare
and execute silently prices a MIXED market. This helper implements the same
capture-and-verify contract the DCN batch adapters use
(``_verify_captured_inputs``): object-identity for field replacement plus a
value fingerprint for in-place mutation, failing closed with
``DeterminismViolation``.
"""
from dataclasses import dataclass

from quantark.execution.cache.fingerprint import try_fingerprint
from quantark.execution.errors import DeterminismViolation

__all__ = ["MarketCapture", "capture_market", "verify_market"]


@dataclass(frozen=True)
class MarketCapture:
    fields: tuple          # captured market objects, identity-checked
    market_fp: str | None  # value fingerprint of the fields (None: uncanonicalizable)
    product_fp: str | None


def capture_market(fields: tuple, product) -> MarketCapture:
    return MarketCapture(
        fields=tuple(fields),
        market_fp=try_fingerprint(tuple(fields)),
        product_fp=try_fingerprint(product),
    )


def verify_market(capture: MarketCapture, current_fields: tuple, product) -> None:
    """Raise DeterminismViolation if the market or product no longer matches
    the prepared capture (replacement OR in-place value mutation)."""
    if capture is None:
        return
    current = tuple(current_fields)
    replaced = len(current) != len(capture.fields) or any(
        a is not b for a, b in zip(current, capture.fields)
    )
    fp_changed = (
        capture.market_fp is not None
        and try_fingerprint(current) != capture.market_fp
    )
    product_changed = (
        capture.product_fp is not None
        and try_fingerprint(product) != capture.product_fp
    )
    if replaced or fp_changed or product_changed:
        raise DeterminismViolation(
            "pricing environment or product mutated between preparation and "
            "execution; prepared artifacts no longer match the live market "
            "(fail closed)"
        )
