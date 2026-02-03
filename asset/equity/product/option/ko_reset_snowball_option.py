"""
Knock-out reset snowball option implementation.

This product behaves like a standard snowball before knock-in (KI). Once KI
occurs, the knock-out (KO) schedule switches to a second, independent schedule.
The post-KI schedule can be applied in two modes:
- ABSOLUTE: post-KI schedule uses absolute times/dates.
- REBASED: post-KI schedule uses offsets from the KI time.
"""

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from util.calendar import calculate_year_fraction
from util.calendar.day_counter import DayCountConvention
from util.enum import (
    BarrierType,
    ObservationAggregation,
    ObservationType,
    PostKOScheduleMode,
    ProtectionType,
    TenorEnd,
)
from util.exceptions import ValidationError

from .observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
    PricingEnv,
    ResolvedObservationRecord,
)
from .snowball_config import AccrualConfig, AirbagConfig, BarrierConfig, PayoffConfig
from .snowball_option import SnowballOption


@dataclass
class KnockOutResetSnowballOption(SnowballOption):
    """
    Snowball option with KO schedule reset after KI.

    The pre-KI KO schedule is defined by `barrier_config` (same as SnowballOption).
    The post-KI KO schedule is defined by `post_barrier_config` (KO-only).
    """

    post_barrier_config: BarrierConfig = field(
        default_factory=lambda: BarrierConfig(ko_barrier=1.0, ko_rate=0.0)
    )
    post_ko_mode: PostKOScheduleMode = PostKOScheduleMode.ABSOLUTE

    def __init__(
        self,
        initial_price: float,
        strike: float,
        barrier_config: BarrierConfig,
        post_barrier_config: BarrierConfig,
        payoff_config: Optional[PayoffConfig] = None,
        accrual_config: Optional[AccrualConfig] = None,
        airbag_config: Optional[AirbagConfig] = None,
        contract_multiplier: float = 1.0,
        is_reverse: bool = False,
        maturity: Optional[float] = None,
        tenor: Optional[float] = None,
        initial_date: Optional[datetime] = None,
        exercise_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
        maturity_date: Optional[datetime] = None,
        tenor_end: TenorEnd = TenorEnd.EXERCISE,
        annualization_day_count: DayCountConvention = DayCountConvention.ACT_365,
        post_ko_mode: PostKOScheduleMode = PostKOScheduleMode.ABSOLUTE,
    ):
        self.post_barrier_config = post_barrier_config
        self.post_ko_mode = post_ko_mode

        super().__init__(
            initial_price=initial_price,
            strike=strike,
            barrier_config=barrier_config,
            payoff_config=payoff_config,
            accrual_config=accrual_config,
            airbag_config=airbag_config,
            contract_multiplier=contract_multiplier,
            is_reverse=is_reverse,
            maturity=maturity,
            tenor=tenor,
            initial_date=initial_date,
            exercise_date=exercise_date,
            settlement_date=settlement_date,
            maturity_date=maturity_date,
            tenor_end=tenor_end,
            annualization_day_count=annualization_day_count,
        )

    def validate(self) -> None:
        super().validate()
        self._validate_post_ko_mode()

    def _validate_barrier_parameters(self) -> None:
        super()._validate_barrier_parameters()
        self._validate_post_barrier_parameters()

    def _validate_observation_parameters(self) -> None:
        super()._validate_observation_parameters()
        self._validate_post_observation_parameters()

    def _validate_post_barrier_parameters(self) -> None:
        """Validate post-KI KO barrier configuration."""
        config = self.post_barrier_config

        self._validate_barrier_array(config.ko_barrier, "Post KO barrier")
        self._validate_rate_array(config.ko_rate, "Post KO rate")

        obs_len = self._get_observation_length_for_config(config)
        if obs_len is not None:
            self._validate_array_length(config.ko_barrier, obs_len, "Post KO barrier")
            self._validate_array_length(config.ko_rate, obs_len, "Post KO rate")

        if config.ki_barrier is not None:
            raise ValidationError("post_barrier_config must not define KI barriers")

    def _validate_post_observation_parameters(self) -> None:
        """Validate post-KI KO observation schedules."""
        config = self.post_barrier_config

        if not isinstance(config.ko_observation_type, ObservationType):
            raise ValidationError(
                f"Invalid post KO observation type: {config.ko_observation_type}"
            )
        if config.ko_observation_type != ObservationType.DISCRETE:
            raise ValidationError("Post KO monitoring must be discrete")

        if (
            config.ko_observation_schedule is None
            and config.ko_observation_dates is None
        ):
            raise ValidationError(
                "Post KO observation dates or schedule required for discrete monitoring"
            )
        self._validate_observation_dates(
            config.ko_observation_dates, "Post KO"
        )

        if self.post_ko_mode == PostKOScheduleMode.REBASED:
            if self.barrier_config.ki_barrier is None:
                raise ValidationError("Rebased post-KO schedule requires KI barrier")
            if (
                self.barrier_config.ki_observation_type != ObservationType.DISCRETE
                or self.barrier_config.ki_continuous
            ):
                raise ValidationError(
                    "Rebased post-KO schedule requires discrete KI monitoring"
                )
            schedule = config.ko_observation_schedule
            if schedule is not None:
                for rec in schedule.records:
                    if rec.observation_date is not None:
                        raise ValidationError(
                            "Rebased post-KO schedule must use observation_time offsets, not dates"
                        )

    def _validate_post_ko_mode(self) -> None:
        if not isinstance(self.post_ko_mode, PostKOScheduleMode):
            raise ValidationError(f"Invalid post_ko_mode: {self.post_ko_mode}")

    def _get_observation_length_for_config(
        self, config: BarrierConfig
    ) -> Optional[int]:
        if config.ko_observation_schedule is not None:
            return len(config.ko_observation_schedule.records)
        if config.ko_observation_dates is not None:
            return len(config.ko_observation_dates)
        return None

    def _build_observation_schedules(self) -> None:
        super()._build_observation_schedules()

        needs_post_schedule = (
            self.post_barrier_config.ko_observation_schedule is None
            and self.post_barrier_config.ko_observation_dates is not None
        )
        if not needs_post_schedule:
            return

        post_barriers = (
            self.post_barrier_config.ko_barrier
            if isinstance(self.post_barrier_config.ko_barrier, list)
            else None
        )
        post_rates = (
            self.post_barrier_config.ko_rate
            if isinstance(self.post_barrier_config.ko_rate, list)
            else None
        )

        records = []
        for i, t in enumerate(self.post_barrier_config.ko_observation_dates):
            barrier_val = (
                post_barriers[i] if post_barriers else self.post_barrier_config.ko_barrier
            )
            rate_val = (
                post_rates[i] if post_rates else self.post_barrier_config.ko_rate
            )
            records.append(
                ObservationRecord(
                    observation_time=t,
                    barrier=barrier_val,
                    return_rate=rate_val,
                    is_rate_annualized=False,
                )
            )
        post_schedule = ObservationSchedule(
            records=records,
            aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
        )

        self.post_barrier_config = replace(
            self.post_barrier_config, ko_observation_schedule=post_schedule
        )

    def _resolve_ko_schedule(
        self, config: BarrierConfig, pricing_env: PricingEnv
    ) -> Tuple[List[ResolvedObservationRecord], List[float], List[ObservationRecord]]:
        schedule = config.ko_observation_schedule
        if schedule is None:
            raise ValidationError("KO observation schedule is required to resolve KO observations.")

        default_barrier = (
            None if isinstance(config.ko_barrier, list) else config.ko_barrier
        )
        resolved_schedule = schedule.resolve(
            pricing_env=pricing_env,
            default_barrier=default_barrier,
            default_payoff=0.0,
            require_single=True,
        )

        rates: List[float] = []
        for idx, _ in enumerate(resolved_schedule):
            rate = schedule.records[idx].return_rate
            if rate is None:
                rate = self._get_barrier_at(config.ko_rate, idx, "KO rate")
            rates.append(rate)

        return resolved_schedule, rates, schedule.records

    def get_pre_ko_observation_profile(self, pricing_env: PricingEnv) -> Dict[str, List[float]]:
        resolved, rates, records = self._resolve_ko_schedule(
            self.barrier_config, pricing_env
        )
        return {
            "observation_times": [rec.observation_time for rec in resolved],
            "barriers": [rec.barrier for rec in resolved],
            "rates": rates,
            "records": records,
        }

    def get_post_ko_observation_profile(self, pricing_env: PricingEnv) -> Dict[str, List[float]]:
        resolved, rates, records = self._resolve_ko_schedule(
            self.post_barrier_config, pricing_env
        )
        return {
            "observation_times": [rec.observation_time for rec in resolved],
            "barriers": [rec.barrier for rec in resolved],
            "rates": rates,
            "records": records,
            "mode": self.post_ko_mode,
        }

    def get_pre_maturity_time(self, pricing_env: PricingEnv) -> float:
        profile = self.get_pre_ko_observation_profile(pricing_env)
        times = profile["observation_times"]
        if times:
            return max(times)
        return super().get_maturity(pricing_env)

    def _resolve_schedule_end_date(
        self, schedule: Optional[ObservationSchedule]
    ) -> Optional[datetime]:
        if schedule is None:
            return None
        dates = [
            rec.observation_date
            for rec in schedule.records
            if rec.observation_date is not None
        ]
        return max(dates) if dates else None

    def _resolve_pre_contract_tenor(self, pricing_env: PricingEnv) -> float:
        end_date = self._resolve_schedule_end_date(
            self.barrier_config.ko_observation_schedule
        )
        if end_date is not None and self.initial_date is not None:
            return calculate_year_fraction(
                self.initial_date,
                end_date,
                self.annualization_day_count,
                pricing_env.bus_days_in_year,
                calendar=getattr(pricing_env, "calendar", None),
            )
        return self.get_pre_maturity_time(pricing_env)

    def _resolve_post_contract_tenor(self, pricing_env: PricingEnv) -> float:
        if self.post_ko_mode == PostKOScheduleMode.ABSOLUTE:
            end_date = self._resolve_schedule_end_date(
                self.post_barrier_config.ko_observation_schedule
            )
            if end_date is not None and self.initial_date is not None:
                return calculate_year_fraction(
                    self.initial_date,
                    end_date,
                    self.annualization_day_count,
                    pricing_env.bus_days_in_year,
                    calendar=getattr(pricing_env, "calendar", None),
                )
        return self.get_post_maturity_time(pricing_env)

    def get_contract_tenor(self, pricing_env: PricingEnv = None) -> float:
        """
        For KO-reset snowball, default contract tenor follows the pre-KI schedule.
        Post-KI tenor is handled explicitly in get_maturity_payoff_v1.
        """
        if pricing_env is None:
            return super().get_contract_tenor(pricing_env)
        return self._resolve_pre_contract_tenor(pricing_env)

    def get_post_maturity_time(
        self, pricing_env: PricingEnv, ki_time: Optional[float] = None
    ) -> float:
        profile = self.get_post_ko_observation_profile(pricing_env)
        times = profile["observation_times"]
        if not times:
            return self.get_pre_maturity_time(pricing_env)
        if self.post_ko_mode == PostKOScheduleMode.ABSOLUTE:
            return max(times)
        if ki_time is None:
            raise ValidationError("ki_time is required for rebased post-KO maturity.")
        return float(ki_time + max(times))

    def get_max_maturity_time(self, pricing_env: PricingEnv) -> float:
        base_maturity = super().get_maturity(pricing_env)
        pre_maturity = self.get_pre_maturity_time(pricing_env)
        post_profile = self.get_post_ko_observation_profile(pricing_env)
        post_times = post_profile["observation_times"]
        if not post_times:
            return max(base_maturity, pre_maturity)

        if self.post_ko_mode == PostKOScheduleMode.ABSOLUTE:
            post_max = max(post_times)
        else:
            ki_profile = self.get_ki_observation_profile(pricing_env)
            ki_times = ki_profile["observation_times"]
            if not ki_times:
                post_max = pre_maturity
            else:
                post_max = max(ki_times) + max(post_times)
        return max(base_maturity, pre_maturity, post_max)

    def get_maturity_payoff_v0(
        self, spot: float, accumulated_coupons: float = 0.0, pricing_env: PricingEnv = None
    ) -> float:
        """
        KO-reset V0 payoff uses pre-KI contract tenor when annualized.
        """
        if spot < 0:
            raise ValidationError(f"Spot price must be non-negative, got {spot}")

        principal = (
            self.initial_price * self.contract_multiplier
            if self.payoff_config.include_principal
            else 0.0
        )
        contract_tenor: Optional[float] = None

        if (
            self.payoff_config.call_rebate_enabled
            and self.payoff_config.call_strike is not None
        ):
            call_payoff = max(spot - self.payoff_config.call_strike, 0.0)
            rebate = (
                self.payoff_config.call_participation_rate
                * self.contract_multiplier
                * call_payoff
            )
            if self.accrual_config.is_annualized_rebate:
                contract_tenor = contract_tenor or self._resolve_pre_contract_tenor(
                    pricing_env
                )
                rebate *= contract_tenor
        else:
            contract_tenor = contract_tenor or self._resolve_pre_contract_tenor(
                pricing_env
            )
            if self.accrual_config.is_annualized_rebate:
                rebate = (
                    self.payoff_config.rebate_rate
                    * self.initial_price
                    * self.contract_multiplier
                    * contract_tenor
                )
            else:
                rebate = (
                    self.payoff_config.rebate_rate
                    * self.initial_price
                    * self.contract_multiplier
                )

        # Note: accumulated_coupons is typically 0 for KO reset as they don't have coupons
        # but added for interface compatibility.
        return principal + rebate + accumulated_coupons

    def get_maturity_payoff_v1(
        self, spot: float, pricing_env: PricingEnv = None
    ) -> float:
        """
        KO-reset V1 payoff uses post-KI tenor when annualized.
        """
        if spot < 0:
            raise ValidationError(f"Spot price must be non-negative, got {spot}")

        principal = (
            self.initial_price * self.contract_multiplier
            if self.payoff_config.include_principal
            else 0.0
        )

        airbag_barrier = self.airbag_config.airbag_barrier
        participation_rate = self.payoff_config.participation_rate
        effective_strike = self.strike

        if airbag_barrier is not None:
            if self.is_reverse:
                is_unsafe = spot > airbag_barrier
            else:
                is_unsafe = spot < airbag_barrier

            if is_unsafe:
                participation_rate = self.airbag_config.airbag_participation_rate
                effective_strike = (
                    self.airbag_config.airbag_strike
                    if self.airbag_config.airbag_strike is not None
                    else self.strike
                )

        if self.is_reverse:
            raw_diff = effective_strike - spot
        else:
            raw_diff = spot - effective_strike

        downside = (
            participation_rate * min(raw_diff, 0.0) * self.contract_multiplier
        )
        if self.accrual_config.is_annualized_ki:
            contract_tenor = self._resolve_post_contract_tenor(pricing_env)
            downside *= contract_tenor

        if self.payoff_config.protection_type == ProtectionType.FULL:
            floor = 0.0
            downside = max(downside, -floor)
        elif self.payoff_config.protection_type == ProtectionType.PARTIAL:
            floor = (
                self.payoff_config.protection_rate
                * self.initial_price
                * self.contract_multiplier
            )
            downside = max(downside, -floor)

        return principal + downside

    def compute_ko_accrual_factor(
        self,
        observation_time: float,
        schedule_record: ObservationRecord,
        pricing_env: PricingEnv,
    ) -> float:
        """Compute accrual factor for KO coupon given observation time."""
        annualized_ko = self._effective_annualized_flag(
            self.accrual_config.is_annualized_ko
        )
        if not annualized_ko:
            return 1.0

        accrual_start_date = self.initial_date
        if schedule_record.observation_date is not None:
            if accrual_start_date is None:
                if pricing_env is None:
                    raise ValidationError(
                        "PricingEnvironment required to resolve KO accrual from observation_date."
                    )
                accrual_start_date = pricing_env.valuation_date
            return calculate_year_fraction(
                accrual_start_date,
                schedule_record.observation_date,
                self.annualization_day_count,
                pricing_env.bus_days_in_year,
                calendar=getattr(pricing_env, "calendar", None),
            )

        if accrual_start_date is None:
            return float(observation_time)

        if pricing_env is None:
            raise ValidationError(
                "PricingEnvironment required to resolve KO accrual without observation_date."
            )
        if pricing_env.valuation_date < accrual_start_date:
            raise ValidationError(
                "valuation_date must be on or after initial_date to resolve KO accrual."
            )
        if pricing_env.valuation_date == accrual_start_date:
            initial_to_valuation = 0.0
        else:
            initial_to_valuation = calculate_year_fraction(
                accrual_start_date,
                pricing_env.valuation_date,
                self.annualization_day_count,
                pricing_env.bus_days_in_year,
                calendar=getattr(pricing_env, "calendar", None),
            )
        return float(initial_to_valuation + observation_time)

    def cache_key(self) -> Dict[str, object]:
        key = super().cache_key()

        def _serialize_enum(value) -> Optional[str]:
            return value.name if hasattr(value, "name") else None

        def _serialize_schedule(
            schedule: Optional[ObservationSchedule],
        ) -> Optional[Dict[str, object]]:
            if schedule is None:
                return None
            frequency = schedule.frequency
            if hasattr(frequency, "name"):
                frequency_value = frequency.name
            else:
                frequency_value = frequency
            return {
                "aggregation_mode": _serialize_enum(schedule.aggregation_mode),
                "frequency": frequency_value,
                "records": [_serialize_value(rec) for rec in schedule.records],
            }

        def _serialize_value(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, datetime):
                return value.isoformat()
            if hasattr(value, "name"):
                return value.name
            if isinstance(value, ObservationSchedule):
                return _serialize_schedule(value)
            if hasattr(value, "__dataclass_fields__"):
                return {
                    f.name: _serialize_value(getattr(value, f.name))
                    for f in value.__dataclass_fields__.values()
                }
            if isinstance(value, dict):
                return {k: _serialize_value(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_serialize_value(v) for v in value]
            return str(value)

        key["post_barrier_config"] = _serialize_value(self.post_barrier_config)
        key["post_ko_mode"] = _serialize_value(self.post_ko_mode)
        return key
