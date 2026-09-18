"""Prepared, snapshot-based valuation contexts and vector spot queries."""
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
import numpy as np

from quantark.util.exceptions import NumericalError, ValidationError
from .basis import make_mesh
from .contract import affine, grid_function
from .gaussian import readout
from .operator import Operator


@dataclass(frozen=True)
class PreparedQuad:
    """Immutable market/contract snapshot; evaluation does not mutate caches.

    Point Greeks differentiate the same Gaussian integral used for PV.
    Desk finite-bump Greeks remain a separately named engine operation.
    """

    _function: object = field(repr=False)
    _mean: float
    _variance: float
    _discount: float
    _params: object = field(repr=False)
    spot_range: tuple[float, float]
    channels: tuple[str, ...]
    diagnostics: object
    pending_pv: float = 0.0
    _continuous: object = field(default=None, repr=False)

    def evaluate(self, spots):
        spots = np.asarray(spots, dtype=float).reshape(-1)
        if np.any(~np.isfinite(spots)) or np.any(spots <= 0):
            raise ValidationError("query spots must be positive and finite")
        if np.any(spots < self.spot_range[0]) or np.any(spots > self.spot_range[1]):
            raise ValidationError(
                f"query outside prepared spot range {self.spot_range}; prepare a wider context"
            )
        if self._continuous is None:
            r = readout(
                self._function,
                np.log(spots),
                self._mean,
                self._variance,
                self._discount,
                self._params,
            )
        else:
            from .continuous import continuous_readout

            hit, barrier, reverse = self._continuous
            r = continuous_readout(
                self._function,
                hit,
                np.log(spots),
                self._mean,
                self._variance,
                self._discount,
                barrier,
                reverse,
                self._params,
            )
        result = {
            "price": r[0, 0] + self.pending_pv,
            "delta": r[1, 0] / spots,
            "gamma": (r[2, 0] - r[1, 0]) / spots**2,
        }
        if np.any(~np.isfinite(r[0])) or np.any(~np.isfinite(result["price"])):
            raise NumericalError("QUAD V2 produced a nonfinite price or cashflow")
        # NaN is intentional at a contractual jump or kink; infinite risk is
        # an arithmetic failure, never a qualified result.
        if np.any(np.isinf(r[1:])) or any(
            np.any(np.isinf(result[name])) for name in ("delta", "gamma")
        ):
            raise NumericalError("QUAD V2 produced an infinite spot Greek")
        if len(self.channels) > 1:
            result["components"] = {
                name: r[0, i].copy() for i, name in enumerate(self.channels) if i
            }
            result["components"]["pending"] = np.full(len(spots), self.pending_pv)
        return result


def interval_moments(env, strike, times):
    positive = np.asarray([t for t in times if t > 0])
    means = np.zeros(len(times))
    variances = np.zeros(len(times))
    discounts = np.ones(len(times))
    if len(positive):
        grid = np.r_[0.0, positive]
        dfs = np.r_[1.0, [env.get_discount_factor(float(t)) for t in positive]]
        carry = np.r_[0.0, [env.get_div_yield(float(t)) * t for t in positive]]
        surface = env.vol_surface
        if getattr(surface, "exposes_exact_total_variance", False):
            total = np.asarray(
                surface.total_variance(strike, grid, env.spot), dtype=float
            )
        else:
            vols = np.array([env.get_vol(strike, float(t)) for t in positive])
            if np.any(vols < 0) or np.any(~np.isfinite(vols)):
                raise ValidationError("QUAD V2 requires finite nonnegative volatility")
            total = np.r_[0.0, vols**2 * positive]
        if (
            total.shape != grid.shape
            or np.any(~np.isfinite(total))
            or np.any(total < 0)
        ):
            raise ValidationError("invalid cumulative variance")
        var = np.diff(total)
        if np.any(var < 0) or np.any(~np.isfinite(var)):
            raise ValidationError(
                "QUAD V2 requires finite nonnegative forward variance"
            )
        mask = np.asarray(times) > 0
        if np.any(dfs <= 0) or np.any(~np.isfinite(dfs)) or np.any(~np.isfinite(carry)):
            raise ValidationError("invalid deterministic discount/carry inputs")
        step_dfs = dfs[1:] / dfs[:-1]
        means[mask] = -np.log(step_dfs) - np.diff(carry) - var / 2
        variances[mask] = var
        discounts[mask] = step_dfs
    return means, variances, discounts


def prepare_compiled(contract, env, strike, params, spots, *, pending_pv=0.0):
    spots = np.asarray(spots, dtype=float).reshape(-1)
    if not len(spots) or np.any(~np.isfinite(spots)) or np.any(spots <= 0):
        raise ValidationError("prepare needs positive finite spots")
    params = deepcopy(params)
    events = contract.events
    means, variances, discounts = interval_moments(
        env, strike, [e.time for e in events]
    )
    mesh = make_mesh(
        np.log(spots), means, variances, params, align=contract.continuous_barrier
    )
    n_states, n_channels = len(contract.state_names), len(contract.channels)
    if (
        mesh
        and n_states * n_channels * mesh.cells * mesh.order * 8 * 4
        > params.max_work_bytes
    ):
        raise NumericalError("QUAD V2 state arrays exceed workspace budget")
    op = Operator(mesh, params) if mesh else None
    states = tuple(affine(np.zeros(n_channels)) for _ in range(n_states))
    continuous_pair = None
    for i in range(len(events) - 1, -1, -1):
        functions = events[i].apply(states)
        if i == 0:
            if contract.continuous_barrier is not None:
                pair = dict(contract.ki_pairs).get(contract.initial_state)
                if pair is not None:
                    continuous_pair = (
                        functions[pair],
                        contract.continuous_barrier,
                        contract.reverse,
                    )
            break
        if variances[i] == 0:
            states = tuple(f.shifted(means[i], discounts[i]) for f in functions)
            if contract.continuous_barrier is not None:
                from .continuous import deterministic_continuous

                states = deterministic_continuous(
                    states,
                    contract.ki_pairs,
                    contract.continuous_barrier,
                    contract.reverse,
                    means[i],
                )
        else:
            rolled = [
                op.integrate(f, means[i], variances[i], discounts[i]) for f in functions
            ]
            if contract.continuous_barrier is not None:
                from .continuous import continuous_readout

                for alive, hit in contract.ki_pairs:
                    rolled[alive] = continuous_readout(
                        functions[alive],
                        functions[hit],
                        mesh.nodes.ravel(),
                        means[i],
                        variances[i],
                        discounts[i],
                        contract.continuous_barrier,
                        contract.reverse,
                        params,
                    )[0].reshape(n_channels, mesh.cells, mesh.order)
            states = tuple(grid_function(mesh, values) for values in rolled)
    diagnostics = MappingProxyType(
        {
            "nodes": 0 if mesh is None else mesh.cells * mesh.order,
            "cells": 0 if mesh is None else mesh.cells,
            "log_domain": None if mesh is None else (mesh.lo, mesh.hi),
            "cell_width": None if mesh is None else mesh.h,
            "events": len(events),
            "states": n_states,
            "kernel_cache_bytes": 0 if op is None else op.cache.bytes,
            "backends": () if op is None else tuple(sorted(op.backends_used)),
            "model": "deterministic Gaussian log increments",
            "qualification": "requires independent accuracy/refinement evidence",
            **contract.metadata,
        }
    )
    return PreparedQuad(
        functions[contract.initial_state],
        float(means[0]),
        float(variances[0]),
        float(discounts[0]),
        params,
        (float(min(spots)), float(max(spots))),
        contract.channels,
        diagnostics,
        float(pending_pv),
        continuous_pair,
    )
