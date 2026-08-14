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
from typing import Any, Mapping, Sequence

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.grid.config import resolve_config
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.snowball_config import (
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
from quantark.util.enum import ObservationType
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
)

#: One profile coarser than each target, for the refinement ladder.
_COARSER_ACCURACY = {"high": "standard", "standard": "fast", "fast": "fast"}

#: Quadrature settings that cannot move the answer, so they stay out of the
#: recorded configuration and out of the identity hash.
_QUAD_NON_NUMERIC = ("bump_size", "bump_config", "auto_converge",
                     "convergence_rel_tol", "convergence_abs_tol",
                     "max_convergence_grid_points")


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
    return spec


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
    """Build a snowball with a monthly discrete KO schedule and continuous KI.

    Barriers are absolute levels, so a case that moves spot moves the trade's
    position relative to its barriers -- which is the point of the near_ko and
    near_ki cases.
    """
    months = int(spec["months"])
    maturity = float(spec["maturity"])
    ko_rate = float(spec.get("ko_rate", 0.15))

    barrier_config = BarrierConfig(
        ko_barrier=float(spec["ko_barrier"]),
        ko_rate=ko_rate,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[
            maturity * (i + 1) / months for i in range(months)
        ],
        ki_barrier=float(spec["ki_barrier"]),
        ki_continuous=True,
    )
    payoff_config = PayoffConfig(rebate_rate=float(spec.get("rebate_rate", ko_rate)))

    return SnowballOption(
        initial_price=float(spec["initial_price"]),
        strike=float(spec["strike"]),
        barrier_config=barrier_config,
        payoff_config=payoff_config,
        contract_multiplier=float(spec.get("contract_multiplier", 1.0)),
        maturity=maturity,
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
        return environment, product


class SnowballMCReference(_SnowballArm):
    """Paired-RQMC benchmark: one randomization per batch, shared across bumps."""

    def __init__(self, sampling: SamplingPolicy, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sampling = sampling

    def config(self) -> Mapping[str, Any]:
        """The benchmark's own settings -- it is half of every comparison."""
        return {
            "engine": "SnowballMCEngine",
            "method": MonteCarloMethod.RANDOMIZED_QUASI.value,
            "paths_per_batch": self.sampling.paths_per_batch,
            "substeps_per_interval": int(self._params.get("substeps_per_interval", 1)),
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
        substeps = int(self._params.get("substeps_per_interval", 1))

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
            if substeps > 1:
                engine.substeps_per_interval = substeps
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
