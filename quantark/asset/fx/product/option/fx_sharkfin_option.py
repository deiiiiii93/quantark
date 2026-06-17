"""
FX single sharkfin option: a knock-out vanilla capped at the barrier, with an
optional knock-out rebate and a no-hit bonus.

The "fin" shape comes from the capped knock-out payoff: an up-and-out call rises
with spot up to the barrier (where it caps / knocks out), then drops to the
rebate. Only the two sensible single-barrier combinations are allowed:

  - up-and-out call  (barrier H above the strike; cap = H)
  - down-and-out put (barrier L below the strike; cap = L)

Under continuous monitoring the cap is implied by the knock-out (a surviving
path never breaches the barrier, so the cap only binds under discrete monitoring
when the terminal fixing is unobserved and lands beyond the barrier).
"""

from datetime import datetime
from typing import List, Optional

from quantark.util.enum import OptionType, ObservationType
from quantark.util.exceptions import ValidationError
from ..base_fx_product import BaseFxProduct
from ..currency_pair import CurrencyPair


class FxSharkfinOption(BaseFxProduct):
    """
    FX single sharkfin (capped knock-out vanilla + rebates).

    Attributes:
        strike: Strike rate (quote per base).
        barrier: Knock-out barrier; also the payoff cap.
        is_up: True for up-and-out (call), False for down-and-out (put).
        option_type: CALL (requires is_up) or PUT (requires not is_up).
        participation: Multiplier on the capped vanilla payoff.
        ko_rebate: Cash rebate (domestic) paid on knock-out.
        rebate_at_hit: Pay the KO rebate at hit (True) vs at expiry (False).
        no_hit_rebate: Cash bonus (domestic) added on surviving paths, paid at
            expiry.
        monitoring: CONTINUOUS or DISCRETE barrier monitoring.
        observation_times: Year-fraction fixing times for DISCRETE monitoring.
    """

    def __init__(
        self,
        strike: float,
        barrier: float,
        is_up: bool,
        option_type: OptionType,
        participation: float = 1.0,
        ko_rebate: float = 0.0,
        rebate_at_hit: bool = False,
        no_hit_rebate: float = 0.0,
        monitoring: ObservationType = ObservationType.CONTINUOUS,
        observation_times: Optional[List[float]] = None,
        currency_pair: Optional[CurrencyPair] = None,
        maturity: Optional[float] = None,
        expiry_date: Optional[datetime] = None,
        delivery: Optional[float] = None,
        delivery_date: Optional[datetime] = None,
    ):
        super().__init__(
            currency_pair=currency_pair,
            maturity=maturity,
            expiry_date=expiry_date,
            delivery=delivery,
            delivery_date=delivery_date,
        )
        self.strike = strike
        self.barrier = barrier
        self.is_up = is_up
        self.option_type = option_type
        self.participation = participation
        self.ko_rebate = ko_rebate
        self.rebate_at_hit = rebate_at_hit
        self.no_hit_rebate = no_hit_rebate
        self.monitoring = monitoring
        self.observation_times = observation_times
        self.validate()

    def validate(self) -> None:
        if self.strike <= 0:
            raise ValidationError(f"Strike must be positive, got {self.strike}")
        if self.barrier <= 0:
            raise ValidationError(f"Barrier must be positive, got {self.barrier}")
        if not isinstance(self.option_type, OptionType):
            raise ValidationError(f"Invalid option_type: {self.option_type}")
        if self.participation <= 0:
            raise ValidationError(
                f"participation must be positive, got {self.participation}"
            )
        if self.ko_rebate < 0:
            raise ValidationError(f"ko_rebate must be non-negative, got {self.ko_rebate}")
        if self.no_hit_rebate < 0:
            raise ValidationError(
                f"no_hit_rebate must be non-negative, got {self.no_hit_rebate}"
            )
        # Only up-out call / down-out put make sense as a sharkfin (the barrier
        # caps the in-the-money side).
        if self.is_up and self.option_type != OptionType.CALL:
            raise ValidationError("An up-and-out sharkfin must be a CALL")
        if not self.is_up and self.option_type != OptionType.PUT:
            raise ValidationError("A down-and-out sharkfin must be a PUT")
        if self.is_up and self.barrier <= self.strike:
            raise ValidationError("Up-and-out call requires barrier > strike")
        if not self.is_up and self.barrier >= self.strike:
            raise ValidationError("Down-and-out put requires barrier < strike")
        if not isinstance(self.monitoring, ObservationType):
            raise ValidationError(f"Invalid monitoring: {self.monitoring}")
        if self.monitoring == ObservationType.DISCRETE:
            self._validate_observation_times()
        self._validate_maturity_inputs()

    def _validate_observation_times(self) -> None:
        times = self.observation_times
        if not times:
            raise ValidationError(
                "DISCRETE monitoring requires a non-empty observation_times list"
            )
        if list(times) != sorted(times):
            raise ValidationError("observation_times must be ascending")
        if len(set(times)) != len(times):
            raise ValidationError("observation_times must be unique")
        T = self.maturity
        for t in times:
            if t <= 0.0:
                raise ValidationError("observation_times must be strictly positive")
            if T is not None and t > T:
                raise ValidationError("observation_times must not exceed maturity")

    def is_call(self) -> bool:
        return self.option_type == OptionType.CALL

    def capped_intrinsic(self, spot: float) -> float:
        """Participation-scaled capped vanilla payoff (no rebate / KO logic)."""
        if self.is_call():
            return self.participation * max(min(spot, self.barrier) - self.strike, 0.0)
        return self.participation * max(self.strike - max(spot, self.barrier), 0.0)

    def get_payoff(self, spot: float) -> float:
        """Surviving-path terminal payoff (capped intrinsic + no-hit bonus).

        The knock-out / rebate logic is applied by the engine over the path.
        """
        return self.capped_intrinsic(spot) + self.no_hit_rebate

    def __repr__(self):
        side = "up-out" if self.is_up else "down-out"
        return (
            f"FxSharkfinOption({self.currency_pair}, {side}, {self.option_type}, "
            f"K={self.strike:.6f}, H={self.barrier:.6f}, part={self.participation})"
        )
