"""Research contracts priced by the local-vol transition QUAD (per unit notional S0).

Semantics mirror the production ``SnowballOption`` as the model-validation
builders configure it (spec D7): KO at an observation when ``S >= KO`` pays
``1 + ko_rate * (t_k + accrual_offset)`` at that observation; KO stays live
after KI; never-KI maturity pays the flat ``1 + rebate_rate``; KI'd maturity pays
``1 + min(S_T - K, 0) / S0``; KI when ``S <= KI``. Times are float years from
the valuation instant.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

from quantark.util.numerical import Tolerance

KI_PER_YEAR = 252
EPS = Tolerance.ZERO          # year-fraction coincidence of a daily date with maturity


@dataclass(frozen=True)
class Contract:
    kind: Literal["vanilla", "up_out", "snowball"]
    maturity: float
    strike: float
    s0: float
    ko: float = math.inf
    ki: float = 0.0
    ko_times: tuple[float, ...] = ()
    ki_times: tuple[float, ...] = ()
    ko_rate: float = 0.0
    rebate_rate: float = 0.0
    option: Literal["call", "put"] = "call"
    accrual_offset: float = 0.0

    def __post_init__(self):
        if self.kind not in ("vanilla", "up_out", "snowball"):
            raise ValueError("supported kinds: vanilla, up_out, snowball")
        if self.option not in ("call", "put"):
            raise ValueError("option must be 'call' or 'put'")
        scalars = (self.maturity, self.strike, self.s0, self.ko_rate, self.rebate_rate, self.accrual_offset)
        if not all(math.isfinite(x) for x in scalars) or min(self.maturity, self.strike, self.s0) <= 0:
            raise ValueError("finite positive maturity, strike and s0 required")
        for times in (self.ko_times, self.ki_times):
            if any(not math.isfinite(t) or t <= 0 or t > self.maturity for t in times):
                raise ValueError("event times must lie in (0, maturity]")
            if any(b <= a for a, b in zip(times, times[1:])):
                raise ValueError("event times must be strictly increasing")
        if self.kind == "vanilla" and (self.ko_times or self.ki_times):
            raise ValueError("vanilla carries no barrier events")
        if self.kind == "up_out":
            if not self.ko_times or self.ko_times[-1] != self.maturity or self.ki_times:
                raise ValueError("up_out requires a KO observation at maturity and no KI events")
            if not (math.isfinite(self.ko) and self.ko > self.strike) or self.option != "call":
                raise ValueError("up_out is a call with a finite KO above the strike")
        if self.kind == "snowball":
            if not self.ko_times or not math.isfinite(self.ko):
                raise ValueError("snowball requires a finite KO with observations")
            if not 0 < self.ki < self.strike <= self.ko or not self.ki < self.s0 < self.ko:
                raise ValueError("snowball requires KI < strike <= KO and KI < s0 < KO")

    @property
    def channels(self) -> int:
        return 2 if self.kind == "snowball" else 1

    def ko_payment(self, t: float) -> float:
        """Cash per unit notional paid at a KO observation at time ``t``."""
        if self.kind == "snowball":
            return 1.0 + self.ko_rate * (t + self.accrual_offset)
        return 0.0

    def shifted(self, dt: float) -> "Contract":
        """The contract one valuation step ``dt`` later (production theta, spec D8)."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("positive finite shift required")
        ko = tuple(t - dt for t in self.ko_times)
        ki = tuple(t - dt for t in self.ki_times)
        if any(t <= 0 for t in ko + ki) or self.maturity <= dt:
            raise ValueError("shift crosses an observation; lifecycle resolution is out of scope")
        return replace(self, maturity=self.maturity - dt, ko_times=ko, ki_times=ki,
                       accrual_offset=self.accrual_offset + dt)


def ki_schedule(maturity: float, per_year: int = KI_PER_YEAR) -> tuple[float, ...]:
    """Daily KI dates k/per_year up to maturity, plus maturity if it is off that grid."""
    count = int(math.floor(maturity * per_year + EPS))
    times = [k / per_year for k in range(1, count + 1)]
    if not times or abs(times[-1] - maturity) > EPS:
        times.append(maturity)
    else:
        times[-1] = maturity
    return tuple(times)


def snowball_contract(s0: float, maturity: float, months: int, *, ko_m: float = 1.03, ki_m: float = 0.85,
                      strike_m: float = 1.0, ko_rate: float = 0.15, rebate_rate: float = 0.15,
                      ki_per_year: int = KI_PER_YEAR) -> Contract:
    """Headline Snowball: monthly KO from month 1, daily KI (pinned study terms, spec D7)."""
    ko_times = tuple(maturity * (k + 1) / months for k in range(months))
    return Contract("snowball", maturity, strike_m * s0, s0, ko=ko_m * s0, ki=ki_m * s0, ko_times=ko_times,
                    ki_times=ki_schedule(maturity, ki_per_year), ko_rate=ko_rate, rebate_rate=rebate_rate)


def up_out_contract(s0: float, maturity: float, months: int, *, strike_m: float = 1.0,
                    ko_m: float = 1.2) -> Contract:
    """Discrete monthly up-and-out call, zero rebate."""
    ko_times = tuple(maturity * (k + 1) / months for k in range(months))
    return Contract("up_out", maturity, strike_m * s0, s0, ko=ko_m * s0, ko_times=ko_times)


def vanilla_contract(s0: float, maturity: float, *, strike_m: float = 1.0, option: str = "put") -> Contract:
    return Contract("vanilla", maturity, strike_m * s0, s0, option=option)
