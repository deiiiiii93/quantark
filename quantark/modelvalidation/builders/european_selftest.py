"""European flat-BSM self-test: the framework's own calibration check.

The candidate here is the analytical Black-Scholes engine, whose answer is known
in closed form. If the framework cannot admit an analytically exact engine
against its own Monte Carlo benchmark, the framework is wrong -- not the engine.
That makes this study a permanent CI test of the certification machinery, and
the template a new engine family copies.

It is deliberately small: three cases, a few thousand paths, seconds to run.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from quantark.asset.equity.engine.analytical.black_scholes_engine import (
    BlackScholesEngine,
)
from quantark.asset.equity.engine.mc.euro_mc_engine import EuropeanMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.european_vanilla_option import (
    EuropeanVanillaOption,
)
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import ValidationError

from quantark.modelvalidation.candidate import CandidateResult, LadderRung
from quantark.modelvalidation.reference import BatchResult
from quantark.modelvalidation.registry import register_builder
from quantark.modelvalidation.study import HedgeContractScale, SamplingPolicy

VALUATION_DATE = datetime(2024, 1, 1)

_ENVIRONMENT_KEYS = ("spot", "vol", "rate", "div_yield")
_PRODUCT_KEYS = ("strike", "maturity", "option_type")


@register_builder("hedge_contracts", kind="economic_scale")
def build_hedge_contract_scale(params: Mapping[str, Any]) -> HedgeContractScale:
    """Economic scale in hedge contracts."""
    return HedgeContractScale(**dict(params))


@register_builder("flat_bsm", kind="environment")
def build_flat_bsm_environment(params: Mapping[str, Any]) -> dict:
    """Validate a flat Black-Scholes market spec.

    Returns the spec, not a ``PricingEnvironment``: the reference and candidate
    builders construct environments per case, after applying that case's
    overrides.
    """
    spec = dict(params)
    unknown = set(spec) - set(_ENVIRONMENT_KEYS)
    if unknown:
        raise ValidationError(
            f"Unknown flat_bsm environment keys: {sorted(unknown)}; expected a subset "
            f"of {_ENVIRONMENT_KEYS}"
        )
    for key in _ENVIRONMENT_KEYS:
        if key not in spec:
            raise ValidationError(f"flat_bsm environment is missing {key!r}")
    return spec


@register_builder("equity.european", kind="product")
def build_european_product_spec(params: Mapping[str, Any]) -> dict:
    """Validate a European vanilla product spec."""
    spec = dict(params)
    unknown = set(spec) - set(_PRODUCT_KEYS)
    if unknown:
        raise ValidationError(
            f"Unknown equity.european product keys: {sorted(unknown)}; expected a "
            f"subset of {_PRODUCT_KEYS}"
        )
    for key in ("strike", "maturity"):
        if key not in spec:
            raise ValidationError(f"equity.european product is missing {key!r}")
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


def make_product(spec: Mapping[str, Any]) -> EuropeanVanillaOption:
    """Build a European vanilla option from a product spec."""
    option_type = str(spec.get("option_type", "call")).lower()
    if option_type not in ("call", "put"):
        raise ValidationError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return EuropeanVanillaOption(
        strike=float(spec["strike"]),
        option_type=OptionType.CALL if option_type == "call" else OptionType.PUT,
        maturity=float(spec["maturity"]),
    )


def resolve_specs(
    environment_params: Mapping[str, Any],
    product_params: Mapping[str, Any],
    case,
) -> tuple[dict, dict]:
    """Apply a case's overrides on top of the study-level specs."""
    environment = dict(environment_params)
    environment.update(case.environment_params)
    product = dict(product_params)
    product.update(case.product_params)
    return environment, product


class _FlatBsmArm:
    """Shared spec handling for both arms of this study."""

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

    def _specs(self, case):
        return resolve_specs(self.environment_params, self.product_params, case)


class EuropeanAnalyticalCandidate(_FlatBsmArm):
    """The analytically exact candidate: closed-form Black-Scholes."""

    def name(self) -> str:
        return "equity.european.analytical"

    def params(self) -> Mapping[str, Any]:
        """A closed-form engine has no grid; the bump width is its only knob."""
        return {
            **self._params,
            "engine": "BlackScholesEngine",
            "grid": {"method": "closed_form", "bump": float(self._params.get("bump", 0.01))},
        }

    def evaluate(self, case) -> CandidateResult:
        environment, product_spec = self._specs(case)
        product = make_product(product_spec)
        engine = BlackScholesEngine()
        bump = float(self._params.get("bump", 0.01))
        values = _central_difference_greeks(
            lambda spot: engine.price(product, make_environment(environment, spot)),
            float(environment["spot"]),
            bump,
        )
        # A closed-form engine has no grid to refine: target and medium rungs
        # coincide, so the envelope is exactly zero rather than absent.
        rungs = (
            LadderRung(axis="analytic", level="target", values=values),
            LadderRung(axis="analytic", level="medium", values=dict(values)),
        )
        return CandidateResult(values=values, ladders=rungs)


class EuropeanMCReference(_FlatBsmArm):
    """The stochastic benchmark: RQMC Monte Carlo with paired seeds."""

    def __init__(self, sampling: SamplingPolicy, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sampling = sampling

    def config(self) -> Mapping[str, Any]:
        """The benchmark's own settings -- it is half of every comparison."""
        return {
            "engine": "EuropeanMCEngine",
            "method": MonteCarloMethod.RANDOMIZED_QUASI.value,
            "paths_per_batch": self.sampling.paths_per_batch,
            "greeks": "paired central difference (common random numbers)",
        }

    def identity(self, case) -> Mapping[str, Any]:
        environment, product_spec = self._specs(case)
        return {
            "builder": "equity.european.mc",
            "case": case.name,
            "environment": environment,
            "product": product_spec,
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
        product = make_product(product_spec)
        seed = self.sampling.seed + batch_index

        def price_at(spot: float) -> float:
            # One engine instance per pricing call: engine instances are not
            # safe to reuse across calls, and a fresh one cannot carry a stale
            # term-structure context.
            engine = EuropeanMCEngine(
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


def _central_difference_greeks(price_at, spot: float, relative_bump: float) -> dict:
    """PV plus central-difference delta and gamma at one spot.

    The three prices share a seed on the Monte Carlo side, so the differences
    are taken along one set of paths (common random numbers) rather than across
    independent simulations -- without that pairing the Greek noise would swamp
    the signal.
    """
    h = relative_bump * spot
    down, base, up = price_at(spot - h), price_at(spot), price_at(spot + h)
    return {
        "pv": base,
        "delta": (up - down) / (2.0 * h),
        "gamma": (up - 2.0 * base + down) / (h * h),
    }


@register_builder("equity.european.analytical", kind="candidate")
def build_european_analytical_candidate(
    environment_params: Mapping[str, Any],
    product_params: Mapping[str, Any],
    quantities: Sequence[str],
    params: Mapping[str, Any],
) -> EuropeanAnalyticalCandidate:
    return EuropeanAnalyticalCandidate(
        environment_params=environment_params,
        product_params=product_params,
        quantities=quantities,
        params=params,
    )


@register_builder("equity.european.mc", kind="reference")
def build_european_mc_reference(
    environment_params: Mapping[str, Any],
    product_params: Mapping[str, Any],
    sampling: SamplingPolicy,
    quantities: Sequence[str],
    params: Mapping[str, Any],
) -> EuropeanMCReference:
    return EuropeanMCReference(
        sampling=sampling,
        environment_params=environment_params,
        product_params=product_params,
        quantities=quantities,
        params=params,
    )
