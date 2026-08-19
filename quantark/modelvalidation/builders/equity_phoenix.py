"""Certification builders for the equity phoenix family.

The same three arms as the snowball study -- the two-surface PDE solver, the
quadrature engine, and a paired-RQMC Monte Carlo benchmark -- over a payoff
that adds one thing snowballs do not have: a *coupon barrier*. A phoenix pays a
coupon at every observation where spot is above that barrier, whether or not the
trade knocks out, and (optionally) remembers the coupons it missed and pays them
later. That turns each observation date into a digital, which is where a
deterministic grid tends to disagree with Monte Carlo first.

Flat-BSM market data, the environment helpers, and the central-difference Greek
helper are shared with ``equity_snowball``. That is deliberate rather than
accidental coupling: PhoenixPDESolver subclasses SnowballPDESolver and
PhoenixQuadEngine subclasses SnowballQuadEngine, so the two studies certify
overlapping code and must not drift in how they build market data.

Two payoff switches are pinned here rather than exposed to the study, the way
``equity_snowball`` pins continuous KI: coupons pay at the observation
(``CouponPayType.INSTANT``) and the structure is not reversed.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.pde.grid.config import resolve_config
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.phoenix_helpers import (
    create_standard_phoenix,
)
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.util.enum import CouponPayType
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import ValidationError

from quantark.modelvalidation.builders.equity_snowball import (
    _COARSER_ACCURACY,
    _QUAD_NON_NUMERIC,
    _central_difference_greeks,
    make_environment,
)
from quantark.modelvalidation.candidate import CandidateResult, LadderRung
from quantark.modelvalidation.engine_config import engine_config
from quantark.modelvalidation.reference import BatchResult
from quantark.modelvalidation.registry import register_builder
from quantark.modelvalidation.study import SamplingPolicy

_PRODUCT_KEYS = (
    "initial_price",
    "strike",
    "ko_barrier",
    "ko_rate",
    "ki_barrier",
    "coupon_barrier",
    "coupon_rate",
    "num_observations",
    "memory_coupon",
    "maturity",
    "contract_multiplier",
    # Step-down knobs. Absent, the product is exactly what the original
    # certification built, so its cells keep their identity.
    "ko_stepdown",
    "coupon_stepdown",
)

_REQUIRED_PRODUCT_KEYS = (
    "initial_price",
    "strike",
    "ko_barrier",
    "ki_barrier",
    "coupon_barrier",
    "coupon_rate",
    "maturity",
    "num_observations",
)

#: See ``equity_snowball._REFERENCE_KEYS``: PhoenixMCEngine likewise derives its
#: time grid from the observation schedule, so there is no sampling knob here
#: that a study could set and the engine would honour.
_REFERENCE_KEYS: frozenset = frozenset()


@register_builder("equity.phoenix", kind="product")
def build_phoenix_product_spec(params: Mapping[str, Any]) -> dict:
    """Validate a phoenix product spec."""
    spec = dict(params)
    unknown = set(spec) - set(_PRODUCT_KEYS)
    if unknown:
        raise ValidationError(
            f"Unknown equity.phoenix product keys: {sorted(unknown)}; expected a "
            f"subset of {_PRODUCT_KEYS}"
        )
    for key in _REQUIRED_PRODUCT_KEYS:
        if key not in spec:
            raise ValidationError(f"equity.phoenix product is missing {key!r}")

    ki_barrier = float(spec["ki_barrier"])
    for key, start in (("ko_stepdown", "ko_barrier"), ("coupon_stepdown", "coupon_barrier")):
        rate = float(spec.get(key, 0.0))
        if rate < 0.0:
            raise ValidationError(f"{key} must be non-negative, got {rate}")
        schedule = _stepdown_schedule(spec, start, key)
        if isinstance(schedule, list) and any(level <= ki_barrier for level in schedule):
            raise ValidationError(
                f"{key}={rate} walks {start} below the ki_barrier ({ki_barrier}): "
                f"schedule reaches {min(schedule)}. Crossing barriers is a "
                "different product, not a step-down."
            )
    return spec


def _stepdown_schedule(spec: Mapping[str, Any], barrier_key: str, rate_key: str):
    """A barrier as a scalar when flat, a per-observation list when stepping.

    Staying scalar in the flat case is what keeps the originally certified
    cells byte-identical rather than merely equivalent.
    """
    start = float(spec[barrier_key])
    rate = float(spec.get(rate_key, 0.0))
    if rate == 0.0:
        return start
    step = rate * float(spec["initial_price"])
    return [start - step * i for i in range(int(spec["num_observations"]))]


def make_phoenix(spec: Mapping[str, Any]) -> PhoenixOption:
    """Build a phoenix with a discrete KO/coupon schedule and continuous KI.

    Barriers are absolute levels, so a case that moves spot moves the trade
    relative to its barriers -- which is what the near_ko, near_coupon and
    near_ki cases exist to exercise.

    ``ko_stepdown`` and ``coupon_stepdown`` turn the corresponding flat barrier
    into a declining per-observation schedule, which is a different engine code
    path: every observation then projects onto its own barrier level, and the
    grid has a dozen levels to align rather than one.
    """
    return create_standard_phoenix(
        initial_price=float(spec["initial_price"]),
        strike=float(spec["strike"]),
        maturity=float(spec["maturity"]),
        contract_multiplier=float(spec.get("contract_multiplier", 1.0)),
        ko_barrier=_stepdown_schedule(spec, "ko_barrier", "ko_stepdown"),
        ko_rate=float(spec.get("ko_rate", 0.0)),
        ki_barrier=float(spec["ki_barrier"]),
        coupon_barrier=_stepdown_schedule(spec, "coupon_barrier", "coupon_stepdown"),
        coupon_rate=float(spec["coupon_rate"]),
        num_observations=int(spec["num_observations"]),
        memory_coupon=bool(spec.get("memory_coupon", False)),
        coupon_pay_type=CouponPayType.INSTANT,
        is_reverse=False,
    )


class _PhoenixArm:
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
        build_phoenix_product_spec(product)
        return environment, product


class PhoenixMCReference(_PhoenixArm):
    """Paired-RQMC benchmark: one randomization per batch, shared across bumps."""

    def __init__(self, sampling: SamplingPolicy, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sampling = sampling
        unsupported = set(self._params) - _REFERENCE_KEYS
        if unsupported:
            raise ValidationError(
                f"equity.phoenix.mc_rqmc does not support {sorted(unsupported)}. "
                "PhoenixMCEngine builds its time grid from the observation "
                "schedule, so these would be banked as benchmark settings but "
                "never applied."
            )

    def config(self) -> Mapping[str, Any]:
        """The benchmark's own settings -- it is half of every comparison."""
        return {
            "engine": "PhoenixMCEngine",
            "method": MonteCarloMethod.RANDOMIZED_QUASI.value,
            "paths_per_batch": self.sampling.paths_per_batch,
            "greeks": "paired central difference (common random numbers)",
        }

    def identity(self, case) -> Mapping[str, Any]:
        environment, product = self._specs(case)
        return {
            "builder": "equity.phoenix.mc_rqmc",
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
        product = make_phoenix(product_spec)
        seed = self.sampling.seed + batch_index

        def price_at(spot: float) -> float:
            # A fresh engine per pricing call: engine instances are not safe to
            # reuse across calls, and the shared seed is what pairs the three
            # bump arms onto one set of paths.
            engine = PhoenixMCEngine(
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


class PhoenixPDECandidate(_PhoenixArm):
    """Two-surface PDE solver; delta and gamma come from the solver directly."""

    def name(self) -> str:
        return "equity.phoenix.pde"

    def params(self) -> Mapping[str, Any]:
        """Declared settings plus the grid the accuracy profile resolves to."""
        accuracy = str(self._params.get("accuracy", "standard"))
        return {
            **self._params,
            "engine": "PhoenixPDESolver",
            "grid": engine_config(resolve_config(accuracy, None)),
        }

    def _greeks(self, case, accuracy: str) -> dict:
        environment, product_spec = self._specs(case)
        solver = PhoenixPDESolver(params=PDEParams(accuracy=accuracy))
        result = solver.calculate_greeks(
            make_phoenix(product_spec), make_environment(environment)
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


class PhoenixQuadCandidate(_PhoenixArm):
    """Quadrature engine; Greeks by central difference on the same bump width."""

    def name(self) -> str:
        return "equity.phoenix.quad"

    def params(self) -> Mapping[str, Any]:
        """Declared settings plus the full resolved quadrature configuration."""
        grid_points = int(self._params.get("grid_points", 1001))
        return {
            **self._params,
            "engine": "PhoenixQuadEngine",
            "grid": engine_config(
                QuadParams(grid_points=grid_points), exclude=_QUAD_NON_NUMERIC
            ),
        }

    def _greeks(self, case, grid_points: int) -> dict:
        environment, product_spec = self._specs(case)
        product = make_phoenix(product_spec)
        bump = float(self._params.get("bump", 0.01))

        def price_at(spot: float) -> float:
            engine = PhoenixQuadEngine(params=QuadParams(grid_points=grid_points))
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


@register_builder("equity.phoenix.mc_rqmc", kind="reference")
def build_phoenix_mc_reference(
    environment_params: Mapping[str, Any],
    product_params: Mapping[str, Any],
    sampling: SamplingPolicy,
    quantities: Sequence[str],
    params: Mapping[str, Any],
) -> PhoenixMCReference:
    return PhoenixMCReference(
        sampling=sampling,
        environment_params=environment_params,
        product_params=product_params,
        quantities=quantities,
        params=params,
    )


@register_builder("equity.phoenix.pde", kind="candidate")
def build_phoenix_pde_candidate(
    environment_params: Mapping[str, Any],
    product_params: Mapping[str, Any],
    quantities: Sequence[str],
    params: Mapping[str, Any],
) -> PhoenixPDECandidate:
    return PhoenixPDECandidate(
        environment_params=environment_params,
        product_params=product_params,
        quantities=quantities,
        params=params,
    )


@register_builder("equity.phoenix.quad", kind="candidate")
def build_phoenix_quad_candidate(
    environment_params: Mapping[str, Any],
    product_params: Mapping[str, Any],
    quantities: Sequence[str],
    params: Mapping[str, Any],
) -> PhoenixQuadCandidate:
    return PhoenixQuadCandidate(
        environment_params=environment_params,
        product_params=product_params,
        quantities=quantities,
        params=params,
    )
