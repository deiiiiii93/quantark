"""Monte Carlo routes (Snowball, Phoenix, Barrier, Digital) with explicit sampling evidence.

The MC engines draw on the clock-wrapped surface's exact per-step variance and
decide an observation exactly at valuation on the known spot. Every price
carries its paths, seed, standard error and the estimator it came from; a
plain QMC run is labelled as having no valid error estimate. Pending
receivables are added by the service, so stateful engines are priced without
the lifecycle state here.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite

from quantark.execution.errors import CapabilityError
from quantark.intraday.engines.base import (TERMINATED_POINT_GREEKS, EnginePriceOutcome, PointGreeks, decided_at_valuation,
                                            decided_point_greeks, decided_price_outcome, plain_snowball,
                                            point_greeks_from_estimates)


def _estimator(engine, result) -> str:
    method = getattr(getattr(engine, "method", None), "name", "")
    if method == "RANDOMIZED_QUASI":
        return "scramble_means" if getattr(result, "batches_used", None) else "pathwise_iid"
    if method == "QUASI":
        return "pathwise_iid_on_qmc (not a valid QMC error)"
    return "pathwise_iid"


class MCRoute:
    def price(self, ctx, engine) -> EnginePriceOutcome:
        from quantark.asset.equity.engine.mc import EuropeanMCEngine
        from quantark.asset.equity.product.option import EuropeanVanillaOption

        num, env = ctx.numerical, ctx.pricing_env
        if num.terminated:
            return EnginePriceOutcome(0.0, "terminated", {"reason": "lifecycle state is not alive; only the ledger remains"}, {},
                                      exact=True)
        twin = num.product
        if plain_snowball(twin) and decided_at_valuation(ctx):
            # nothing is left to simulate: the day-level engine's zero-maturity shortcut means "already processed"
            return decided_price_outcome(ctx)
        if isinstance(twin, EuropeanVanillaOption):
            solver = EuropeanMCEngine(params=deepcopy(engine.params), method=engine.method)
            method = "mc_vanilla_after_ki"
        else:
            solver = deepcopy(engine)
            method = f"mc_{getattr(engine.method, 'name', 'unknown').lower()}"
        pv = float(solver.price(twin, env))
        result = solver.get_last_result() if hasattr(solver, "get_last_result") else None
        std_error = solver.get_last_std_error() if hasattr(solver, "get_last_std_error") else None
        params = solver.params
        numerical = {
            "n_paths": int(getattr(params, "num_paths", 0) or 0),
            "seed": getattr(params, "seed", None),
            "std_error": None if std_error is None else float(std_error),
            "estimator": _estimator(solver, result),
            "batches": getattr(result, "batches_used", None),
            "bridge": bool(getattr(solver, "use_brownian_bridge", False)),
            "resolution": "sampling_uncertainty_reported",
        }
        return EnginePriceOutcome(pv, method, numerical, {}, engine_used=solver)

    def point_greeks(self, ctx, engine) -> PointGreeks:
        """Paired RQMC: the engine's session spec at spot*(1-h), spot, spot*(1+h) on identical scramble batches.

        That is a central difference at the finite relative bump h, not a derivative: near a fixing a 1% move
        spans the diffusion layer.
        """
        from quantark.intraday.greeks import bump_config_for
        from quantark.montecarlo import run_paired_rqmc_greeks
        from quantark.util.enum.engine_enums import MonteCarloMethod

        num = ctx.numerical
        if num.terminated:
            return TERMINATED_POINT_GREEKS
        if plain_snowball(num.product) and decided_at_valuation(ctx):
            return decided_point_greeks(ctx)
        if getattr(engine, "method", None) != MonteCarloMethod.RANDOMIZED_QUASI:
            raise CapabilityError(f"point greeks on MC need RQMC: {type(engine).__name__} runs "
                                  f"{getattr(getattr(engine, 'method', None), 'name', 'an unknown method')}")
        if not hasattr(engine, "build_rqmc_session_spec"):
            raise CapabilityError(f"point greeks on MC need an RQMC session spec; {type(engine).__name__} has none")
        h = float(bump_config_for(engine).spot_bump)
        specs = []
        for factor in (1.0 - h, 1.0, 1.0 + h):
            env = deepcopy(ctx.pricing_env)
            env.spot_quote.spot = float(ctx.spot) * factor
            spec = deepcopy(engine).build_rqmc_session_spec(num.product, env)
            if spec is None:
                raise CapabilityError(f"{type(engine).__name__} takes no RQMC session for this claim (near-expiry shortcut)")
            specs.append(spec)
        res = run_paired_rqmc_greeks(*specs, spot=float(ctx.spot), relative_bump=h)
        uncertainty = {"delta": float(res.delta_std_error), "gamma": float(res.gamma_std_error),
                       "delta_estimate": float(res.delta), "gamma_estimate": float(res.gamma),
                       "batches": float(res.batches_used), "relative_bump": h}
        min_batches = int(getattr(engine.params, "rqmc_min_batches", 2))
        note = f"paired RQMC central difference at relative bump {h:g}; standard errors in uncertainty"
        if not (isfinite(res.delta_std_error) and isfinite(res.gamma_std_error) and res.batches_used >= min_batches):
            batches = int(res.batches_used)
            note = (f"paired RQMC central difference at relative bump {h:g}; standard error unavailable: "
                    f"{batches} batch{'' if batches == 1 else 'es'} (minimum {min_batches}) or a non-finite estimate of it")
        return point_greeks_from_estimates(float(res.delta), float(res.gamma), "paired_rqmc", note=note,
                                           uncertainty=uncertainty)
