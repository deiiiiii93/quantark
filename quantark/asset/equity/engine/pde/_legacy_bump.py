"""Deep-copy bump clone kept for the 2D Heston/SLV solvers only.

Their ``calculate_greeks`` (``snowball_vol_pde_solvers.py``, ``phoenix_vol_pde_solvers.py``)
and 2D S-axis freezing rely on it. Sub-project 3 moves the 2D solvers onto the
core and deletes this. The bodies are the 0.6.0 ``BasePDESolver`` /
``SnowballPDESolver`` methods, verbatim, less the per-solver factorization and
session attributes the 1D base no longer has.
"""
from __future__ import annotations

from copy import deepcopy

__all__ = ["LegacyBumpCloneMixin"]


class LegacyBumpCloneMixin:
    """``create_bump_context`` by deep copy (the 2D bases list it first in their MRO)."""

    def create_bump_context(self, product, pricing_env):
        """
        Return a PDE solver clone with the base spatial domain frozen.

        Migrated solvers freeze the whole base-market ``Layout`` (spec §4.8):
        spot/vol/rate/div bumps reuse it by identity; calendar (theta) bumps
        rebuild only the time layout via ``rebind_time``.
        """
        if self._uses_grid_layer():
            tau = product.get_maturity(pricing_env)
            if tau <= 0:
                return self
            clone = self._bump_clone()
            prep = getattr(clone, "_prepare_for_request", None)
            if prep is not None:
                prep(product, pricing_env)
            market = clone.market_snapshot(product, pricing_env)
            request = clone.grid_request(product, market, tau)
            clone._frozen_base_layout = clone.grid_binder.bind(request, market)
            return clone

        raise NotImplementedError(
            "legacy params-freeze bump contexts were removed with the "
            "declarative grid layer (0.4.0)"
        )

    def _bump_clone(self):
        """A fully configured engine clone for bump contexts.

        ``deepcopy`` of the live instance — not ``type(self)(params=...)`` —
        so subclasses with richer constructors (Heston ``model_params``, SLV
        ``leverage_surface``, ADI dimensions) keep their full configuration.
        Transient per-solve state (layouts, binder caches, session artefacts)
        is excluded and reset on the clone.
        """
        transient = tuple(name for name in self._bump_transient_attrs() if hasattr(self, name))
        saved = {name: getattr(self, name) for name in transient}
        for name in transient:
            setattr(self, name, None)
        try:
            clone = deepcopy(self)
        finally:
            for name, value in saved.items():
                setattr(self, name, value)
        return clone

    def _bump_transient_attrs(self) -> tuple:
        """Per-solve state excluded from the bump-clone deepcopy (layouts and
        binder caches hold mappingproxies deepcopy cannot handle; sessions
        must not leak into clones). Subclasses with extra grid state extend
        this tuple; names the instance lacks are skipped."""
        return (
            "_grid_binder",
            "_frozen_base_layout",
            "_session_artefacts",
        )

    def _prepare_for_request(self, product, pricing_env) -> float:
        """Resolve the solve state grid_request depends on; returns tau."""
        self._grid_request_env = pricing_env
        tau = product.get_maturity(pricing_env)
        self._prepare_solve_state(product, pricing_env)
        return tau
