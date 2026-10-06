"""The surface members of the local-vol PDE solvers on the core (spec 3C §5.1)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import numpy as np

from quantark.asset.equity.engine.pde.base_pde_solver import StepCoefficients
from quantark.asset.equity.engine.pde.core.components import CentralDifference, nodal_stencil
from quantark.asset.equity.engine.pde.core.samples import _validated_local_vol_row, sample_local_vol
from quantark.param import GridVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import PricingError
from quantark.volmodels.localvol import LocalVolSurface, build_dupire_local_vol


class LocalVolSurfaceMixin:
    """The surface members of the local-vol solvers on the core (spec 3C §5.1): the Dupire surface built once
    around a whole call (``_with_surface``), sampled one local-vol row per step on the solve's S grid."""

    def _init_local_vol(self, local_vol_surface: Optional[LocalVolSurface]) -> None:
        self._prebuilt = local_vol_surface
        self._active_lv_surface: Optional[LocalVolSurface] = None
        # the environment the active surface was built for (a direct prepare builds its own when it differs)
        self._active_lv_env: Optional[PricingEnvironment] = None
        self._active_s_vec: Optional[np.ndarray] = None

    def _build_surface(self, env: PricingEnvironment) -> LocalVolSurface:
        if self._prebuilt is not None:
            return self._prebuilt
        if not isinstance(env.vol_surface, GridVolSurface):
            raise PricingError(
                f"{type(self).__name__} needs a GridVolSurface or a prebuilt LocalVolSurface"
            )
        return build_dupire_local_vol(
            env.vol_surface,
            spot=env.spot,
            rate_curve=env.rate_curve,
            div_yield=env.get_div_yield,
        )

    @contextmanager
    def _surface_scope(self, env: PricingEnvironment):
        previous = (self._active_lv_surface, self._active_lv_env)
        self._active_lv_surface = self._build_surface(env)
        self._active_lv_env = env
        try:
            yield
        finally:
            self._active_lv_surface, self._active_lv_env = previous

    def _with_surface(self, env: PricingEnvironment, fn):
        with self._surface_scope(env):
            return fn()

    def representative_vol(self, product, pricing_env) -> float:
        # A surface that is not the environment's own (the Heston/SLV deterministic limit) states the vol its grid
        # must span; a Dupire surface states none and the domain stays the environment's strike vol.
        vol = super().representative_vol(product, pricing_env)
        surface = self._active_lv_surface if self._active_lv_surface is not None else self._prebuilt
        domain_vol = getattr(surface, "domain_vol", None)
        if domain_vol is None:
            return vol
        return max(vol, domain_vol(float(product.get_maturity(pricing_env))))

    def _constructor_kwargs(self) -> dict:
        return {**super()._constructor_kwargs(), "local_vol_surface": self._prebuilt}

    def _build_step_coefficients(
        self,
        pricing_env: PricingEnvironment,
        ref_strike: float,
        t_vec: np.ndarray,
        dx_vec: np.ndarray,
        num_x: int,
    ) -> StepCoefficients:
        # The route's model samples (a local-vol row per step at the interval
        # midpoint); the built-in operator builds the stencil (spec 2 §5.2 step 8).
        samples = sample_local_vol(
            pricing_env, t_vec, self._active_s_vec, self._active_lv_surface,
            ref_strike=ref_strike,
        )
        return CentralDifference().assemble_sets(samples, dx_vec, num_x)

    def _flat_exact_step_coefficients(self, sc, r, q, sigma, dx_vec, num_x):
        return sc

    @staticmethod
    def _local_vol_coefficients(r, q, sigma_vec, dx_vec, num_x):
        return nodal_stencil(r, q, _validated_local_vol_row(sigma_vec, num_x), dx_vec, num_x)
