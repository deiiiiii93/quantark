"""DCN two-surface PDE engine on the shared 1D core (spec 3C §5.4) - independent cross-check of MC.

V1 = knocked-in surface (no coupons ever; KO still live; terminal = loss leg discounted from settlement_date).
V0 = never-knocked-in surface (coupons + KO; projected onto V1 below the KI barrier at every daily date). Backward
event order at each observation date: coupon injection -> KO overwrite -> KI projection (the reverse of the MC
forward event priority KI -> KO -> coupon, so the two engines are path-equivalent). ``core/adapters/dcn.py`` holds
the family: the events as cell-average projections (the legacy nodal masks under ``event_projection="nodal"``) and
Dirichlet rows from the contract's certain edge outcomes.

Deterministic by construction: no Monte Carlo imports (the DCN grid context lives in the neutral product-side
module ``dcn_grid``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.engine.pde.base_pde_solver import BasePDESolver
from quantark.asset.equity.engine.pde.core.adapters.dcn import DCNAdapter
from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest, prepare, run
from quantark.asset.equity.engine.pde.grid import GridRequest
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.param.pde_removed_names import (
    REMOVED_DCN_NAMES,
    reject_removed_kwargs,
)
from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import PricingError, ValidationError


@dataclass(frozen=True)
class DCNPDEResult:
    pv: float
    direction_sign: float
    num_space_nodes: int
    num_time_steps: int

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def apply_dcn_events(
    v0: np.ndarray,
    v1: np.ndarray,
    cpn_mask: np.ndarray,
    ko_mask: np.ndarray,
    ki_mask: np.ndarray,
    coupon_amount: Optional[float],
    ko_amount: Optional[float],
):
    """Backward event operator at one observation date (spec WP1.4 order):
    (1) coupon injection on V0, (2) KO overwrite on both surfaces, (3) KI
    projection ``V0 := V1`` below the KI barrier. ``None`` amount = that
    event type is not active at this date."""
    if coupon_amount is not None:
        v0 = np.where(cpn_mask, v0 + coupon_amount, v0)
    if ko_amount is not None:
        v0 = np.where(ko_mask, ko_amount, v0)
        v1 = np.where(ko_mask, ko_amount, v1)
    v0 = np.where(ki_mask, v1, v0)
    return v0, v1


class DCNPDEEngine(BasePDESolver):
    """Two-surface DCN PDE on the shared core (spec 3C §5.4): the shared presets (200 / 400 / 800 points, 4 / 4 / 8
    steps per day), the cell-average projection of each daily event, half-step damping, Dirichlet rows from the
    contract's certain edge outcomes, the core's readout. Deterministic: no Monte Carlo."""

    engine_type = EngineType.PDE
    settlement_support = SettlementSupport.EVENT_AND_TERMINAL
    supports_lifecycle_state = True  # DCNLifecycleState (W14), read by DCNAdapter
    _adapter_class = DCNAdapter

    def __init__(self, params: Optional[PDEParams] = None, **removed):
        reject_removed_kwargs(type(self).__name__, removed, REMOVED_DCN_NAMES)
        if params is not None and not isinstance(params, PDEParams):
            # 0.5's first positional argument was num_space_nodes.
            raise ValidationError(
                f"{type(self).__name__}(params=...) takes a PDEParams, got "
                f"{type(params).__name__} {params!r}; the positional node count "
                f"was removed in 0.6.0: {REMOVED_DCN_NAMES['num_space_nodes']}"
            )
        super().__init__(params)

    def _uses_grid_layer(self) -> bool:
        return True

    def grid_request(self, product, market, tau) -> GridRequest:
        # The DCN PDE horizon is the final observation date from the grid
        # context, not the product's nominal maturity; the passed tau is
        # superseded so external validation compares on equal footing.
        ctx = build_dcn_grid_context(product)
        tau = float(ctx.times[-1])
        interior = tuple(
            float(t) for t in ctx.times[1:-1] if 0.0 < float(t) < tau
        )
        criticals = [
            market.spot,
            float(product.ki_barrier),
            float(product.ko_barrier),
            float(product.coupon_barrier),
        ]
        return GridRequest(
            tau=tau,
            bound_anchors=(market.spot, float(product.initial_price)),
            critical_prices=tuple(p for p in criticals if p and p > 0),
            hard_lower=None,
            hard_upper=None,
            event_times=interior,
            day_count=market.bus_days_in_year,
            refinement=self._time_refinement(product),
        )

    def get_critical_points(self, product, pricing_env):
        return sorted({float(product.ki_barrier), float(product.ko_barrier), float(product.coupon_barrier)})

    def set_boundary_conditions(self, grid, x_vec, s_vec, t_idx, tau, product, pricing_env):
        raise PricingError("DCNAdapter.boundaries writes both surfaces; the single-surface hook is not used")

    def price(self, product, pricing_env, *, lifecycle_state=None) -> float:
        return self._run(product, pricing_env, OutputRequest(Operation.PRICE), lifecycle_state=lifecycle_state)

    def calculate_greeks(self, product, pricing_env, *, lifecycle_state=None):
        return self._run(product, pricing_env, OutputRequest(Operation.GREEKS, greeks=True),
                         lifecycle_state=lifecycle_state)

    def price_detailed(self, product, pricing_env, *, lifecycle_state=None, **removed) -> DCNPDEResult:
        reject_removed_kwargs(f"{type(self).__name__}.price_detailed", removed, REMOVED_DCN_NAMES)
        problem = prepare(self, product, pricing_env, OutputRequest(Operation.PRICE), lifecycle_state=lifecycle_state,
                          frozen=self._frozen_base_layout, session=self._session_artefacts)
        pv = float(run(problem))
        if problem.terminal is not None:     # the final observation date (W10): settled, no grid
            return DCNPDEResult(pv=pv, direction_sign=product.direction_sign, num_space_nodes=0, num_time_steps=0)
        return DCNPDEResult(pv=pv, direction_sign=product.direction_sign,
                            num_space_nodes=len(problem.layout.spatial.x),
                            num_time_steps=int(problem.layout.time.actual_steps))
