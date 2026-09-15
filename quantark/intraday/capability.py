"""Explicit intraday capability matrix (design §Engine integration and limits).

A row is published only for a (product, engine, monitoring) combination whose
semantics are implemented; ``status="qualified"`` additionally means Gate C
evidence exists for the declared profiles/outputs (Plans 2/3 flip rows).
Explicit requests outside the matrix raise ``CapabilityError`` naming
alternatives. Engine classes match exactly: a subclass does not inherit a row.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from quantark.execution.errors import CapabilityError

_QUAD_V2 = "quantark.asset.equity.engine.quad.v2.engine."
_ANALYTICAL = "quantark.asset.equity.engine.analytical."


@dataclass(frozen=True)
class IntradayCapability:
    product_type: type
    engine_class_path: str                 # f"{module}.{qualname}" of the engine class
    monitoring: str                        # "discrete" | "continuous" | "terminal"
    profiles: frozenset                    # {"any"} or the profile names the route is qualified for
    outputs: frozenset                     # {"price", "delta", ...} the route can deliver
    status: str                            # "supported" | "qualified" | "unsupported"
    note: str = ""
    alternatives: Tuple[str, ...] = ()


def engine_class_path(engine) -> str:
    cls = engine if isinstance(engine, type) else type(engine)
    return f"{cls.__module__}.{cls.__qualname__}"


def _rows() -> Tuple[IntradayCapability, ...]:
    from quantark.asset.equity.product.option.digital_option import CashOrNothingDigitalOption
    from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
    from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
    from quantark.asset.equity.product.option.snowball_option import SnowballOption
    price_only = frozenset({"price"})
    anywhere = frozenset({"any"})
    return (
        IntradayCapability(CashOrNothingDigitalOption, _ANALYTICAL + "digital_option_engine.DigitalOptionAnalyticalEngine",
                           "terminal", anywhere, price_only, "supported",
                           "integrated carry/variance via TradingClockVolSurface; zero variance priced as the exact forward limit"),
        IntradayCapability(SnowballOption, _QUAD_V2 + "SnowballQuadEngineV2", "discrete", anywhere, price_only, "supported",
                           "exact Gaussian interval moments; a t=0 event under 'before' is decided at spot"),
        IntradayCapability(SnowballOption, _QUAD_V2 + "SnowballQuadEngineV2", "continuous", anywhere, price_only, "supported",
                           "continuous KI via the QUAD V2 survival kernel; touch history disclosed as an assumption"),
        IntradayCapability(PhoenixOption, _QUAD_V2 + "PhoenixQuadEngineV2", "discrete", anywhere, price_only, "supported",
                           "valuations before the first due coupon (coupon replay fails closed)"),
        IntradayCapability(PhoenixOption, _QUAD_V2 + "PhoenixQuadEngineV2", "continuous", anywhere, price_only, "supported",
                           "valuations before the first due coupon (coupon replay fails closed)"),
        IntradayCapability(KnockOutResetSnowballOption, _QUAD_V2 + "KOResetSnowballQuadEngineV2", "discrete", anywhere,
                           price_only, "supported",
                           "absolute post-KI schedules; due fixings must be covered by the checkpoint (no tracker replay)"),
        IntradayCapability(KnockOutResetSnowballOption, _QUAD_V2 + "KOResetSnowballQuadEngineV2", "continuous", anywhere,
                           price_only, "supported",
                           "absolute post-KI schedules; due fixings must be covered by the checkpoint (no tracker replay)"),
    )


INTRADAY_CAPABILITIES: Tuple[IntradayCapability, ...] = _rows()


def find_capability(product, engine, *, monitoring: str) -> Optional[IntradayCapability]:
    path = engine_class_path(engine)
    for row in INTRADAY_CAPABILITIES:
        if type(product) is row.product_type and row.engine_class_path == path and row.monitoring == monitoring:
            return row
    return None


def _alternatives(product, monitoring):
    return sorted({r.engine_class_path.rsplit(".", 1)[-1] for r in INTRADAY_CAPABILITIES
                   if type(product) is r.product_type and r.monitoring == monitoring and r.status != "unsupported"})


def require_capability(product, engine, *, monitoring: str, outputs: Iterable[str] = ("price",)) -> IntradayCapability:
    """The matrix row for this combination, or ``CapabilityError`` naming the limitation and alternatives."""
    row = find_capability(product, engine, monitoring=monitoring)
    name = engine_class_path(engine).rsplit(".", 1)[-1]
    if row is None or row.status == "unsupported":
        alts = _alternatives(product, monitoring)
        raise CapabilityError(
            f"{name} has no intraday route for {type(product).__name__} with {monitoring} monitoring "
            f"(PDE/MC/analytical-barrier routes arrive in Plan 2). Intraday alternatives: {alts or 'none'}.")
    missing = sorted(set(outputs) - set(row.outputs))
    if missing:
        raise CapabilityError(f"{name} intraday route for {type(product).__name__} does not deliver {missing}; "
                              f"available outputs: {sorted(row.outputs)}")
    return row


def render_capability_matrix() -> str:
    lines = ["| Product | Engine | Monitoring | Profiles | Outputs | Status | Note |", "|---|---|---|---|---|---|---|"]
    for r in INTRADAY_CAPABILITIES:
        lines.append(f"| {r.product_type.__name__} | {r.engine_class_path.rsplit('.', 1)[-1]} | {r.monitoring} | "
                     f"{', '.join(sorted(r.profiles))} | {', '.join(sorted(r.outputs))} | {r.status} | {r.note} |")
    return "\n".join(lines) + "\n"
