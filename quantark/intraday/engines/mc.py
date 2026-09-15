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

from quantark.intraday.engines.base import EnginePriceOutcome


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
            return EnginePriceOutcome(0.0, "terminated", {"reason": "lifecycle state is not alive; only the ledger remains"}, {})
        twin = num.product
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
