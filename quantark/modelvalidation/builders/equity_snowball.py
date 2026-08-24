"""Snowball under flat Black-Scholes: the module's demonstration study.

Two deterministic engines are certified against one paired-RQMC benchmark: the
two-surface PDE solver and the quadrature engine. That is the shape a real
engine release takes -- several implementations of the same payoff, one
statistically controlled reference, and per-engine decisions.

Snowballs make the certification non-trivial in a way vanillas cannot: the
payoff is discontinuous at the knock-out barrier and path-dependent through
knock-in, so a deterministic grid can agree on price while disagreeing sharply
on gamma near a barrier. Those are exactly the cells the gates exist to catch.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.grid.config import resolve_config
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.snowball_config import (
    AccrualConfig,
    AirbagConfig,
    BarrierConfig,
    PayoffConfig,
)
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import CouponPayType, ObservationType, ProtectionType
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import ValidationError

from quantark.modelvalidation.candidate import CandidateResult, LadderRung
from quantark.modelvalidation.engine_config import engine_config
from quantark.modelvalidation.reference import BatchResult
from quantark.modelvalidation.registry import register_builder
from quantark.modelvalidation.study import SamplingPolicy

VALUATION_DATE = datetime(2024, 1, 1)

_ENVIRONMENT_KEYS = ("spot", "vol", "rate", "div_yield")
_PRODUCT_KEYS = (
    "initial_price",
    "strike",
    "ko_barrier",
    "ki_barrier",
    "ko_rate",
    "rebate_rate",
    "months",
    "maturity",
    "contract_multiplier",
    # Variant knobs. Absent, they leave the product exactly as the original
    # certification built it, so the already-certified cells keep their identity.
    "ki_monitoring",
    "ko_stepdown",
    "parachute",
    "ki_stepdown",
    "ko_rate_step",
    "is_reverse",
    "disable_ko_after_ki",
    "airbag_barrier",
    "airbag_participation_rate",
    "protection_type",
    "protection_rate",
    "participation_rate",
    "call_rebate",
    "call_strike",
    "call_participation_rate",
    "coupon_pay_type",
    "is_annualized",
)

_PROTECTION = {
    "none": ProtectionType.NONE,
    "partial": ProtectionType.PARTIAL,
    "full": ProtectionType.FULL,
}

_COUPON_PAY = {"instant": CouponPayType.INSTANT, "expiry": CouponPayType.EXPIRY}

#: How the knock-in barrier is watched. ``continuous`` is the structure the
#: original certification covered; the other two are separate engine code paths
#: (no Brownian-bridge crossing correction, no per-step regime transfer).
_KI_MONITORING = ("continuous", "discrete", "european")

#: One profile coarser than each target, for the refinement ladder.
_COARSER_ACCURACY = {"high": "standard", "standard": "fast", "fast": "fast"}

#: Parameters the flat-BSM snowball benchmark can actually honour. It is empty
#: on purpose: SnowballMCEngine derives its whole time grid from the product's
#: observation schedule -- a daily grid plus a Brownian-bridge crossing
#: correction when KI is continuous -- so there is no sampling knob to turn.
_REFERENCE_KEYS: frozenset = frozenset()

#: Quadrature settings that cannot move the answer, so they stay out of the
#: recorded configuration and out of the identity hash.
#: ``event_stats_mode`` selects the event-DISTRIBUTION algorithm only; the
#: certified quantities are pv/greeks from price(), which both modes share
#: (npv equality is hex-asserted in test_quad_forward_density_stats).
_QUAD_NON_NUMERIC = ("bump_size", "bump_config", "auto_converge",
                     "convergence_rel_tol", "convergence_abs_tol",
                     "max_convergence_grid_points", "event_stats_mode")


@register_builder("equity.snowball", kind="product")
def build_snowball_product_spec(params: Mapping[str, Any]) -> dict:
    """Validate a snowball product spec."""
    spec = dict(params)
    unknown = set(spec) - set(_PRODUCT_KEYS)
    if unknown:
        raise ValidationError(
            f"Unknown equity.snowball product keys: {sorted(unknown)}; expected a "
            f"subset of {_PRODUCT_KEYS}"
        )
    for key in ("initial_price", "strike", "ko_barrier", "ki_barrier", "maturity", "months"):
        if key not in spec:
            raise ValidationError(f"equity.snowball product is missing {key!r}")

    monitoring = str(spec.get("ki_monitoring", "continuous"))
    if monitoring not in _KI_MONITORING:
        raise ValidationError(
            f"ki_monitoring must be one of {_KI_MONITORING}, got {monitoring!r}"
        )

    stepdown = float(spec.get("ko_stepdown", 0.0))
    if stepdown < 0.0:
        raise ValidationError(f"ko_stepdown must be non-negative, got {stepdown}")

    # A KO schedule that walks below the KI barrier is not a step-down snowball:
    # the two barriers cross and the payoff means something else entirely. Only
    # a parachute may land the *final* barrier on KI, and it does so exactly.
    schedule = _ko_schedule(spec)
    live = schedule[:-1] if bool(spec.get("parachute", False)) else schedule
    ki_barrier = float(spec["ki_barrier"])
    if isinstance(live, list) and any(level <= ki_barrier for level in live):
        raise ValidationError(
            f"ko_stepdown={stepdown} walks below the ki_barrier ({ki_barrier}): "
            f"schedule reaches {min(live)}. Crossing barriers is a different "
            "product, not a step-down."
        )
    ki_stepdown = float(spec.get("ki_stepdown", 0.0))
    if ki_stepdown:
        if monitoring != "discrete":
            raise ValidationError(
                "ki_stepdown requires ki_monitoring='discrete': both deterministic "
                "engines refuse a per-observation KI barrier under continuous "
                "monitoring, so a stepping KI is a discretely observed product."
            )
        levels = _ki_schedule(spec)
        if min(levels) <= 0.0:
            raise ValidationError(
                f"ki_stepdown={ki_stepdown} walks the ki_barrier to {min(levels)}; "
                "a barrier must stay positive."
            )

    protection = str(spec.get("protection_type", "none"))
    if protection not in _PROTECTION:
        raise ValidationError(
            f"protection_type must be one of {tuple(_PROTECTION)}, got {protection!r}"
        )
    coupon_pay = str(spec.get("coupon_pay_type", "instant"))
    if coupon_pay not in _COUPON_PAY:
        raise ValidationError(
            f"coupon_pay_type must be one of {tuple(_COUPON_PAY)}, got {coupon_pay!r}"
        )
    return spec


def _ki_schedule(spec: Mapping[str, Any]):
    """The KI barrier: scalar when flat, per-observation list when stepping."""
    barrier = float(spec["ki_barrier"])
    rate = float(spec.get("ki_stepdown", 0.0))
    if rate == 0.0:
        return barrier
    step = rate * float(spec["initial_price"])
    return [barrier - step * i for i in range(int(spec["months"]))]


def _ko_rate_schedule(spec: Mapping[str, Any]):
    """The KO rate: scalar when flat, per-observation list when stepping."""
    rate = float(spec.get("ko_rate", 0.15))
    step = float(spec.get("ko_rate_step", 0.0))
    if step == 0.0:
        return rate
    return [rate + step * i for i in range(int(spec["months"]))]


def _payoff_config(spec: Mapping[str, Any]) -> PayoffConfig:
    """Payoff features: protection, participation, and a call-style rebate."""
    kwargs: dict[str, Any] = {
        "rebate_rate": float(spec.get("rebate_rate", spec.get("ko_rate", 0.15))),
        "protection_type": _PROTECTION[str(spec.get("protection_type", "none"))],
    }
    for key in ("protection_rate", "participation_rate", "call_participation_rate"):
        if key in spec:
            kwargs[key] = float(spec[key])
    if bool(spec.get("call_rebate", False)):
        kwargs["call_rebate_enabled"] = True
        kwargs["call_strike"] = float(spec.get("call_strike", spec["strike"]))
    return PayoffConfig(**kwargs)


def _accrual_config(spec: Mapping[str, Any]) -> Optional[AccrualConfig]:
    """Only built when a knob asks for it, so the certified cells keep the
    product's own default accrual object rather than an equal-looking copy."""
    if "coupon_pay_type" not in spec and "is_annualized" not in spec:
        return None
    return AccrualConfig(
        coupon_pay_type=_COUPON_PAY[str(spec.get("coupon_pay_type", "instant"))],
        is_annualized=bool(spec.get("is_annualized", True)),
    )


def _airbag_config(spec: Mapping[str, Any]) -> Optional[AirbagConfig]:
    if "airbag_barrier" not in spec:
        return None
    kwargs: dict[str, Any] = {"airbag_barrier": float(spec["airbag_barrier"])}
    if "airbag_participation_rate" in spec:
        kwargs["airbag_participation_rate"] = float(spec["airbag_participation_rate"])
    return AirbagConfig(**kwargs)


def _ko_observation_dates(spec: Mapping[str, Any]) -> list:
    """Evenly spaced KO observations, the schedule every case shares."""
    months = int(spec["months"])
    maturity = float(spec["maturity"])
    return [maturity * (i + 1) / months for i in range(months)]


def _ko_schedule(spec: Mapping[str, Any]):
    """The KO barrier: a scalar when flat, a per-observation list when not.

    Staying scalar in the flat case matters beyond tidiness -- it is what keeps
    the originally certified cells byte-identical rather than merely equivalent.
    """
    barrier = float(spec["ko_barrier"])
    stepdown = float(spec.get("ko_stepdown", 0.0))
    parachute = bool(spec.get("parachute", False))
    if stepdown == 0.0 and not parachute:
        return barrier

    months = int(spec["months"])
    step = stepdown * float(spec["initial_price"])
    levels = [barrier - step * i for i in range(months)]
    if parachute:
        # The parachute: at the final observation the KO barrier drops onto the
        # KI barrier, so anything that never knocked in necessarily knocks out.
        levels[-1] = float(spec["ki_barrier"])
    return levels


def _ki_monitoring(spec: Mapping[str, Any]) -> dict:
    """BarrierConfig keyword arguments for this case's KI monitoring."""
    monitoring = str(spec.get("ki_monitoring", "continuous"))
    if monitoring not in _KI_MONITORING:
        raise ValidationError(
            f"ki_monitoring must be one of {_KI_MONITORING}, got {monitoring!r}"
        )
    if monitoring == "continuous":
        return {"ki_continuous": True}
    dates = (
        [float(spec["maturity"])]
        if monitoring == "european"
        else _ko_observation_dates(spec)
    )
    return {
        "ki_continuous": False,
        "ki_observation_type": ObservationType.DISCRETE,
        "ki_observation_dates": dates,
    }


def make_environment(spec: Mapping[str, Any], spot: float | None = None) -> PricingEnvironment:
    """Build a flat-BSM environment, optionally at a bumped spot."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=float(spec["spot"] if spot is None else spot)),
        vol_surface=FlatVolSurface(volatility=float(spec["vol"])),
        rate_curve=FlatRateCurve(rate=float(spec["rate"])),
        div_yield=ContinuousDividendYield(div_yield=float(spec["div_yield"])),
        valuation_date=VALUATION_DATE,
    )


def make_snowball(spec: Mapping[str, Any]) -> SnowballOption:
    """Build a snowball with a monthly discrete KO schedule.

    Barriers are absolute levels, so a case that moves spot moves the trade's
    position relative to its barriers -- which is the point of the near_ko and
    near_ki cases.

    Three product variants are reachable from the spec, each a distinct engine
    code path rather than a re-parameterization: ``ki_monitoring`` selects
    continuous / discrete / European knock-in observation, ``ko_stepdown``
    turns the flat KO barrier into a declining schedule, and ``parachute``
    drops the final KO barrier onto the KI barrier.
    """
    maturity = float(spec["maturity"])
    ko_rate = float(spec.get("ko_rate", 0.15))

    barrier_config = BarrierConfig(
        ko_barrier=_ko_schedule(spec),
        ko_rate=_ko_rate_schedule(spec),
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=_ko_observation_dates(spec),
        ki_barrier=_ki_schedule(spec),
        disable_ko_after_ki=bool(spec.get("disable_ko_after_ki", False)),
        **_ki_monitoring(spec),
    )

    optional: dict[str, Any] = {}
    accrual = _accrual_config(spec)
    if accrual is not None:
        optional["accrual_config"] = accrual
    airbag = _airbag_config(spec)
    if airbag is not None:
        optional["airbag_config"] = airbag

    return SnowballOption(
        initial_price=float(spec["initial_price"]),
        strike=float(spec["strike"]),
        barrier_config=barrier_config,
        payoff_config=_payoff_config(spec),
        contract_multiplier=float(spec.get("contract_multiplier", 1.0)),
        maturity=maturity,
        is_reverse=bool(spec.get("is_reverse", False)),
        **optional,
    )


def _central_difference_greeks(price_at, spot: float, relative_bump: float) -> dict:
    """PV plus central-difference delta and gamma at one spot."""
    h = relative_bump * spot
    down, base, up = price_at(spot - h), price_at(spot), price_at(spot + h)
    return {
        "pv": base,
        "delta": (up - down) / (2.0 * h),
        "gamma": (up - 2.0 * base + down) / (h * h),
    }


class _SnowballArm:
    """Shared spec handling for every arm of this study."""

    def __init__(
        self,
        environment_params: Mapping[str, Any],
        product_params: Mapping[str, Any],
        quantities: Sequence[str],
        params: Mapping[str, Any],
    ) -> None:
        self.environment_params = dict(environment_params)
        self.product_params = dict(product_params)
        self.quantities = tuple(quantities)
        self._params = dict(params)

    def _specs(self, case) -> tuple[dict, dict]:
        environment = dict(self.environment_params)
        environment.update(case.environment_params)
        product = dict(self.product_params)
        product.update(case.product_params)
        # A variant is expressed as a case override, which the study-level
        # validator never sees. Validate the merged spec here or a typo builds
        # a different product and gets certified under the wrong case name.
        build_snowball_product_spec(product)
        return environment, product


class SnowballMCReference(_SnowballArm):
    """Paired-RQMC benchmark: one randomization per batch, shared across bumps."""

    def __init__(self, sampling: SamplingPolicy, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sampling = sampling
        # A knob this engine cannot honour would still be recorded in the
        # certificate and folded into the benchmark identity hash, so changing
        # it would invalidate checkpoints while moving no number. Refuse it
        # rather than ignore it.
        unsupported = set(self._params) - _REFERENCE_KEYS
        if unsupported:
            raise ValidationError(
                f"equity.snowball.mc_rqmc does not support {sorted(unsupported)}. "
                "SnowballMCEngine builds its time grid from the observation "
                "schedule, so these would be banked as benchmark settings but "
                "never applied."
            )

    def config(self) -> Mapping[str, Any]:
        """The benchmark's own settings -- it is half of every comparison."""
        return {
            "engine": "SnowballMCEngine",
            "method": MonteCarloMethod.RANDOMIZED_QUASI.value,
            "paths_per_batch": self.sampling.paths_per_batch,
            "greeks": "paired central difference (common random numbers)",
        }

    def identity(self, case) -> Mapping[str, Any]:
        environment, product = self._specs(case)
        return {
            "builder": "equity.snowball.mc_rqmc",
            "case": case.name,
            "environment": environment,
            "product": product,
            "quantities": list(self.quantities),
            "params": dict(self._params),
            "config": dict(self.config()),
            "sampling": {
                "paths_per_batch": self.sampling.paths_per_batch,
                "min_batches": self.sampling.min_batches,
                "max_batches": self.sampling.max_batches,
                "seed": self.sampling.seed,
                "bump": self.sampling.bump,
            },
        }

    def run_batch(self, case, batch_index: int) -> BatchResult:
        environment, product_spec = self._specs(case)
        product = make_snowball(product_spec)
        seed = self.sampling.seed + batch_index

        def price_at(spot: float) -> float:
            # A fresh engine per pricing call: engine instances are not safe to
            # reuse across calls, and the shared seed is what pairs the three
            # bump arms onto one set of paths.
            engine = SnowballMCEngine(
                params=MCParams(
                    seed=seed,
                    num_paths=self.sampling.paths_per_batch,
                    use_qmc=True,
                    rqmc_min_batches=1,
                    rqmc_max_batches=1,
                    rqmc_paths_mode="per_batch",
                ),
                method=MonteCarloMethod.RANDOMIZED_QUASI,
            )
            return engine.price(product, make_environment(environment, spot))

        values = _central_difference_greeks(
            price_at, float(environment["spot"]), self.sampling.bump
        )
        return BatchResult(index=batch_index, seed=seed, values=values)


class SnowballPDECandidate(_SnowballArm):
    """Two-surface PDE solver; delta and gamma come from the solver directly."""

    def name(self) -> str:
        return "equity.snowball.pde"

    def params(self) -> Mapping[str, Any]:
        """Declared settings plus the grid the accuracy profile resolves to.

        Recording the resolved grid rather than just the profile name is what
        makes the certificate self-describing, and what makes the identity hash
        move if a future release redefines the profile.
        """
        accuracy = str(self._params.get("accuracy", "standard"))
        return {
            **self._params,
            "engine": "SnowballPDESolver",
            "grid": engine_config(resolve_config(accuracy, None)),
        }

    def _greeks(self, case, accuracy: str) -> dict:
        environment, product_spec = self._specs(case)
        solver = SnowballPDESolver(params=PDEParams(accuracy=accuracy))
        result = solver.calculate_greeks(
            make_snowball(product_spec), make_environment(environment)
        )
        return {
            "pv": result["price"],
            "delta": result["delta"],
            "gamma": result["gamma"],
        }

    def evaluate(self, case) -> CandidateResult:
        accuracy = str(self._params.get("accuracy", "standard"))
        values = self._greeks(case, accuracy)
        coarser = _COARSER_ACCURACY[accuracy]
        rungs = [LadderRung(axis="accuracy", level="target", values=values)]
        if coarser != accuracy:
            rungs.append(
                LadderRung(axis="accuracy", level="medium", values=self._greeks(case, coarser))
            )
        return CandidateResult(values=values, ladders=tuple(rungs))


class SnowballQuadCandidate(_SnowballArm):
    """Quadrature engine; Greeks by central difference on the same bump width."""

    def name(self) -> str:
        return "equity.snowball.quad"

    def params(self) -> Mapping[str, Any]:
        """Declared settings plus the full resolved quadrature configuration.

        Every numerically relevant knob is recorded, including the ones taken
        from defaults: a default that changes in a later release is a numerics
        change, and the identity hash has to notice.
        """
        grid_points = int(self._params.get("grid_points", 1001))
        return {
            **self._params,
            "engine": "SnowballQuadEngine",
            "grid": engine_config(
                QuadParams(grid_points=grid_points), exclude=_QUAD_NON_NUMERIC
            ),
        }

    def _greeks(self, case, grid_points: int) -> dict:
        environment, product_spec = self._specs(case)
        product = make_snowball(product_spec)
        bump = float(self._params.get("bump", 0.01))

        def price_at(spot: float) -> float:
            engine = SnowballQuadEngine(params=QuadParams(grid_points=grid_points))
            return engine.price(product, make_environment(environment, spot))

        return _central_difference_greeks(price_at, float(environment["spot"]), bump)

    def evaluate(self, case) -> CandidateResult:
        grid_points = int(self._params.get("grid_points", 1001))
        values = self._greeks(case, grid_points)
        # Nested odd grid: 1001 -> 501, so the coarse rung reuses half the nodes.
        medium_points = (grid_points - 1) // 2 + 1
        rungs = [LadderRung(axis="grid_points", level="target", values=values)]
        if medium_points >= 11 and medium_points != grid_points:
            rungs.append(
                LadderRung(
                    axis="grid_points", level="medium", values=self._greeks(case, medium_points)
                )
            )
        return CandidateResult(values=values, ladders=tuple(rungs))


@register_builder("equity.snowball.mc_rqmc", kind="reference")
def build_snowball_mc_reference(
    environment_params: Mapping[str, Any],
    product_params: Mapping[str, Any],
    sampling: SamplingPolicy,
    quantities: Sequence[str],
    params: Mapping[str, Any],
) -> SnowballMCReference:
    return SnowballMCReference(
        sampling=sampling,
        environment_params=environment_params,
        product_params=product_params,
        quantities=quantities,
        params=params,
    )


@register_builder("equity.snowball.pde", kind="candidate")
def build_snowball_pde_candidate(
    environment_params: Mapping[str, Any],
    product_params: Mapping[str, Any],
    quantities: Sequence[str],
    params: Mapping[str, Any],
) -> SnowballPDECandidate:
    return SnowballPDECandidate(
        environment_params=environment_params,
        product_params=product_params,
        quantities=quantities,
        params=params,
    )


@register_builder("equity.snowball.quad", kind="candidate")
def build_snowball_quad_candidate(
    environment_params: Mapping[str, Any],
    product_params: Mapping[str, Any],
    quantities: Sequence[str],
    params: Mapping[str, Any],
) -> SnowballQuadCandidate:
    return SnowballQuadCandidate(
        environment_params=environment_params,
        product_params=product_params,
        quantities=quantities,
        params=params,
    )
