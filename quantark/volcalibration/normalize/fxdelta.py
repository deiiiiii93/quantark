"""Delta-quoted FX books: CFETS five-delta tenor slices -> ``QuoteSet``.

The delta convention is CFETS non-premium-adjusted **spot** delta:

    Delta_call = exp(-r_f T) N(d1)
    Delta_put  = exp(-r_f T) (N(d1) - 1)

with forward moneyness in ``d1``.  CFETS publishes strikes alongside the
deltas, so this normalizer does not invert them -- it *verifies* the round trip
and refuses a snapshot whose published strikes and deltas disagree.  That check
is the FX analogue of the listed path's put-call-parity quality gate: both ask
whether the venue's own numbers are internally consistent before anything
downstream trusts them.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Dict, List, Sequence

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.quotes import ExpiryQuotes, IvNode, QuoteSet
from quantark.volcalibration.snapshot import CONVENTION_FX_DELTA, QuoteSnapshot
from quantark.volcalibration.surface import (
    DEFAULT_UNIFORM_GRID_SIZE,
    STRIKE_GRID_UNIFORM_OVER_OVERLAP,
)

PILLAR_ORDER = ("10P", "25P", "ATM", "25C", "10C")
PILLAR_DELTA = {"10P": -0.10, "25P": -0.25, "ATM": None, "25C": 0.25, "10C": 0.10}
TENOR_ORDER = (
    "1D",
    "1W",
    "2W",
    "3W",
    "1M",
    "2M",
    "3M",
    "6M",
    "9M",
    "1Y",
    "18M",
    "2Y",
    "3Y",
)
TENOR_SETS = {
    "core": ("1M", "2M", "3M", "6M", "9M", "1Y"),
    "liquid": ("1W", "2W", "3W", "1M", "2M", "3M", "6M", "9M", "1Y"),
    "full": (
        "1W",
        "2W",
        "3W",
        "1M",
        "2M",
        "3M",
        "6M",
        "9M",
        "1Y",
        "18M",
        "2Y",
        "3Y",
    ),
}

# The published strike and delta must agree to this absolute tolerance.  It is
# the same bound the suite's snapshot loader has always applied; a looser one
# would let a mis-stamped delta through, and a tighter one would reject CFETS's
# own rounding.
DELTA_ROUND_TRIP_ATOL = 2e-10


def normalise_tenor(tenor: str) -> str:
    """The canonical tenor label."""
    value = str(tenor).strip().upper()
    return "18M" if value in {"1.5Y", "1Y6M"} else value


def strike_from_spot_delta(
    forward: float,
    iv: float,
    maturity: float,
    foreign_rate: float,
    delta: float,
) -> float:
    """Invert the CFETS non-premium-adjusted spot delta into a strike.

    ``delta`` is signed: positive for calls, negative for puts.
    """
    values = (forward, iv, maturity)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValidationError("forward, iv and maturity must be finite and positive")
    if not math.isfinite(foreign_rate):
        raise ValidationError("foreign_rate must be finite")
    if not math.isfinite(delta) or delta == 0.0 or abs(delta) >= 1.0:
        raise ValidationError(
            "delta must be finite, non-zero and have magnitude below one"
        )

    foreign_df = math.exp(-foreign_rate * maturity)
    if delta > 0.0:
        probability = delta / foreign_df
    else:
        probability = 1.0 + delta / foreign_df
    if not 0.0 < probability < 1.0:
        raise ValidationError(
            f"delta {delta} is incompatible with foreign discount factor "
            f"{foreign_df:.8f}"
        )
    d1 = NormalDist().inv_cdf(probability)
    vol_time = iv * math.sqrt(maturity)
    return float(forward * math.exp(-d1 * vol_time + 0.5 * vol_time * vol_time))


def spot_delta_from_strike(
    forward: float,
    strike: float,
    iv: float,
    maturity: float,
    foreign_rate: float,
    *,
    is_call: bool,
) -> float:
    """Evaluate the CFETS spot delta; the round-trip check uses this."""
    if min(forward, strike, iv, maturity) <= 0.0:
        raise ValidationError("forward, strike, iv and maturity must be positive")
    vol_time = iv * math.sqrt(maturity)
    d1 = (math.log(forward / strike) + 0.5 * vol_time * vol_time) / vol_time
    n_d1 = NormalDist().cdf(d1)
    foreign_df = math.exp(-foreign_rate * maturity)
    return float(foreign_df * (n_d1 if is_call else n_d1 - 1.0))


def selected_slices(
    snapshot: QuoteSnapshot, tenor_set="core"
) -> List[Dict[str, Any]]:
    """Slices in canonical tenor order for one calibration universe."""
    if isinstance(tenor_set, str):
        if tenor_set not in TENOR_SETS:
            raise ValidationError(
                f"unknown tenor_set {tenor_set!r}; choose {sorted(TENOR_SETS)}"
            )
        tenors = set(TENOR_SETS[tenor_set])
    else:
        tenors = {normalise_tenor(t) for t in tenor_set}
    rows = [
        dict(row)
        for row in snapshot.expiries
        if normalise_tenor(row.get("tenor", "")) in tenors
    ]
    order = {tenor: index for index, tenor in enumerate(TENOR_ORDER)}
    rows.sort(key=lambda row: order[normalise_tenor(row["tenor"])])
    missing = tenors - {normalise_tenor(row["tenor"]) for row in rows}
    if missing:
        raise ValidationError(f"snapshot missing requested tenors: {sorted(missing)}")
    return rows


class FxDeltaNormalizer:
    """CFETS five-delta tenor slices -> ``QuoteSet``.

    Emits exactly the container the listed normalizers emit, so smoothing,
    admission and model calibration never branch on quote convention.
    """

    def __init__(
        self, tenor_set="core", *, grid_size: int = DEFAULT_UNIFORM_GRID_SIZE
    ) -> None:
        self.tenor_set = tenor_set
        self.grid_size = int(grid_size)

    def normalize(self, snapshot: QuoteSnapshot) -> QuoteSet:
        if snapshot.convention != CONVENTION_FX_DELTA:
            raise ValidationError(
                f"FxDeltaNormalizer needs a {CONVENTION_FX_DELTA!r} snapshot, "
                f"got {snapshot.convention!r}"
            )
        expiries: List[ExpiryQuotes] = []
        checked = 0
        for row in selected_slices(snapshot, self.tenor_set):
            tenor = normalise_tenor(row["tenor"])
            maturity = float(row["maturity"])
            forward = float(row["forward"])
            domestic_rate = float(row["domestic_rate"])
            foreign_rate = float(row["foreign_rate"])
            quotes = row.get("quotes", [])
            if tuple(q.get("pillar") for q in quotes) != PILLAR_ORDER:
                raise ValidationError(
                    f"tenor {tenor}: quotes must follow {PILLAR_ORDER}"
                )
            checked += _verify_delta_round_trip(row, tenor)
            nodes = tuple(
                IvNode(
                    strike=float(quote["strike"]),
                    iv=float(quote["mid_iv"]),
                    weight_hint=1.0,
                )
                for quote in quotes
            )
            expiries.append(
                ExpiryQuotes(
                    expiry_label=tenor,
                    expiry_date=row.get("expiry_date"),
                    T=maturity,
                    forward=forward,
                    # Domestic discounting: the CNY leg is the numeraire.
                    discount_factor=math.exp(-domestic_rate * maturity),
                    r=domestic_rate,
                    q=foreign_rate,
                    diagnostics={
                        "pillar_count": float(len(nodes)),
                        "delta_round_trip_atol": DELTA_ROUND_TRIP_ATOL,
                    },
                    nodes=nodes,
                )
            )
        return QuoteSet(
            trade_date=snapshot.trade_date,
            spot=snapshot.spot,
            convention=CONVENTION_FX_DELTA,
            expiries=tuple(expiries),
            universe={
                "expiry_count": len(expiries),
                "node_count": sum(len(e.nodes) for e in expiries),
                "delta_round_trips_verified": checked,
                "filtered_quote_counts": {},
                "excluded_expiries": [],
                # Every tenor's 25-delta strike is its own, so no strike is
                # shared across tenors and there is no observed ladder to build
                # a grid from.  Declared here rather than inferred downstream.
                "strike_grid_rule": STRIKE_GRID_UNIFORM_OVER_OVERLAP,
                "strike_grid_size": self.grid_size,
            },
        )


def _verify_delta_round_trip(row: Dict[str, Any], tenor: str) -> int:
    """Do the published strikes reproduce the published deltas? Count checked.

    This is the FX analogue of the listed path's parity gate -- the one check
    that says the venue's own numbers are internally consistent -- so a slice
    that carries no ``effective_foreign_rate_for_delta`` is refused rather than
    waved through with zero checks.  ``foreign_rate`` is NOT substituted: CFETS
    publishes a separate effective rate for its delta convention, and using the
    pricing rate instead would test a different quantity and pass for the wrong
    reason.
    """
    if "effective_foreign_rate_for_delta" not in row:
        raise ValidationError(
            f"tenor {tenor}: snapshot carries no "
            "'effective_foreign_rate_for_delta', so the published deltas cannot "
            "be verified against the published strikes"
        )
    delta_rate = float(row["effective_foreign_rate_for_delta"])
    checked = 0
    for quote in row["quotes"]:
        delta = quote.get("delta")
        if delta is None:
            continue
        recovered = spot_delta_from_strike(
            float(row["forward"]),
            float(quote["strike"]),
            float(quote["mid_iv"]),
            float(row["maturity"]),
            delta_rate,
            is_call=float(delta) > 0.0,
        )
        if not math.isclose(
            recovered, float(delta), rel_tol=0.0, abs_tol=DELTA_ROUND_TRIP_ATOL
        ):
            raise ValidationError(
                f"tenor {tenor} {quote['pillar']}: published strike implies spot "
                f"delta {recovered:.12f}, not the published {float(delta):.12f}"
            )
        checked += 1
    return checked
