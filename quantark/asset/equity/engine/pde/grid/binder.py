"""GridBinder — the engine-owned layout constructor and cache (spec §4.2).

One binder per engine instance. ``bind`` memoizes layouts in a bounded LRU
keyed by ``(request, market, config_key)`` — layouts are deterministic pure
data, so cache hits return the SAME object (observable via ``is``; layout
classes are eq=False by design). ``bind_shared`` builds one spatial layout
for a same-underlying book; ``rebind_time`` supports calendar (theta) bumps
by rebuilding only the time layout while the spatial layout is retained by
identity.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Sequence

from quantark.asset.equity.engine.pde.grid.config import (
    GridConfig,
    reject_legacy_resolution_knobs,
    resolve_config,
)
from quantark.asset.equity.engine.pde.grid.request import GridRequest, MarketSnapshot
from quantark.asset.equity.engine.pde.grid.space import SpatialLayout, build_space
from quantark.asset.equity.engine.pde.grid.time import TimeLayout, build_time
from quantark.util.exceptions import NumericalError, PricingError, ValidationError
from quantark.util.numerical import is_close


@dataclass(frozen=True, eq=False)
class Layout:
    """A bound grid: spatial + time + provenance; compared by identity."""

    spatial: SpatialLayout
    time: TimeLayout
    request: GridRequest
    config_key: tuple


def validate_external_layout(
    layout: Layout, fresh_request: GridRequest, market: MarketSnapshot
) -> None:
    """Validate an externally supplied layout against a fresh request (§4.6).

    Alignment-critical fields must match exactly (a layout from another
    product or schedule cannot silently misprice); spot and interior critical
    prices must sit strictly inside the frozen domain, with an ``is_close``
    exemption for prices that ARE a hard edge; concentration drift
    (critical_prices / bound_anchors moved under a bump) is allowed.
    """
    old = layout.request
    for field in ("tau", "event_times", "hard_lower", "hard_upper"):
        if getattr(old, field) != getattr(fresh_request, field):
            raise ValidationError(
                f"external layout mismatch on {field}: layout has "
                f"{getattr(old, field)}, product requires "
                f"{getattr(fresh_request, field)}"
            )
    lo, hi = layout.spatial.bounds

    def _covered(price: float, label: str) -> None:
        if (fresh_request.hard_lower is not None and is_close(price, lo)) or (
            fresh_request.hard_upper is not None and is_close(price, hi)
        ):
            return  # the price IS the domain edge (continuous barrier)
        if not (lo < price < hi):
            raise NumericalError(
                f"frozen layout does not cover {label} {price} "
                f"(domain [{lo}, {hi}]); rebuild at base market"
            )

    _covered(market.spot, "spot")
    for p in fresh_request.critical_prices:
        _covered(p, "critical price")


def resolve_bound_layout(
    binder: "GridBinder",
    frozen: Optional[Layout],
    request: GridRequest,
    market: MarketSnapshot,
) -> Layout:
    """The layout for a solve: frozen (bump clone) > time-rebind > bind (§4.8).

    An alignment-identical request reuses the frozen layout by identity
    (coverage validated). A changed schedule/tau (calendar roll) rebinds the
    time layout on the SAME spatial object — but only when the hard bounds
    are unchanged: hard bounds ARE the absorbing domain edges, so a frozen
    spatial layout built for different barriers would silently apply the
    wrong boundary. That case fails closed instead.
    """
    if frozen is None:
        return binder.bind(request, market)
    fr = frozen.request
    if request.hard_lower != fr.hard_lower or request.hard_upper != fr.hard_upper:
        raise PricingError(
            "frozen bump layout was built for hard bounds "
            f"({fr.hard_lower}, {fr.hard_upper}) but this solve requires "
            f"({request.hard_lower}, {request.hard_upper}); a bump context "
            "cannot be reused across products with different absorbing "
            "barriers — create a new bump context for this product"
        )
    if request.tau == fr.tau and request.event_times == fr.event_times:
        validate_external_layout(frozen, request, market)
        return frozen
    rebound = binder.rebind_time(frozen, request)
    validate_external_layout(rebound, request, market)
    return rebound


class GridLayerMixin:
    """Grid-layer seam for engines NOT derived from BasePDESolver.

    ``DCNPDEEngine`` and the Phase-3 LV/Heston/SLV engines derive from
    ``BaseEngine`` and inherit none of the BasePDESolver seam; this mixin
    supplies the same contract: binder ownership, market snapshot, external
    -layout validation, and a frozen-layout bump context. Adopters implement
    ``grid_request(product, market, tau)`` / ``representative_vol`` and may
    override ``_default_grid_config`` for engine-specific geometry defaults.
    """

    _frozen_base_layout = None  # set on create_bump_context clones

    def _default_grid_accuracy(self) -> str:
        return "standard"

    def _default_grid_config(self):
        return None

    def _grid_layer_binder(self, params=None) -> "GridBinder":
        binder = getattr(self, "_grid_binder", None)
        if binder is None:
            reject_legacy_resolution_knobs(params)
            accuracy = getattr(params, "accuracy", None) or self._default_grid_accuracy()
            override = getattr(params, "grid", None) or self._default_grid_config()
            cache_enabled = bool(getattr(params, "cache_enabled", True)) and (
                getattr(params, "cache_strategy", "standard") != "disable"
            )
            binder = GridBinder(
                accuracy,
                override,
                cache_enabled=cache_enabled,
                cache_max_entries=int(
                    getattr(params, "grid_cache_max_entries", 128) or 128
                ),
            )
            self._grid_binder = binder
        return binder

    def representative_vol(self, product, pricing_env) -> float:
        tau = product.get_maturity(pricing_env)
        strike = getattr(product, "strike", pricing_env.spot)
        return float(pricing_env.get_vol(strike, tau))

    def market_snapshot(self, product, pricing_env) -> MarketSnapshot:
        tau = product.get_maturity(pricing_env)
        return MarketSnapshot(
            spot=float(pricing_env.spot),
            sigma_ref=self.representative_vol(product, pricing_env),
            r_ref=float(pricing_env.get_rate(tau)),
            q_ref=float(pricing_env.get_div_yield(tau)),
        )

    def _external_layout_check(self, product, pricing_env, layout) -> None:
        market = self.market_snapshot(product, pricing_env)
        tau = product.get_maturity(pricing_env)
        validate_external_layout(
            layout, self.grid_request(product, market, tau), market
        )

    def create_bump_context(self, product, pricing_env):
        """Clone with the base-market layout frozen (spec §4.8)."""
        import copy

        tau = product.get_maturity(pricing_env)
        if tau <= 0:
            return self
        clone = copy.deepcopy(self)
        clone._grid_binder = None
        market = clone.market_snapshot(product, pricing_env)
        request = clone.grid_request(product, market, tau)
        clone._frozen_base_layout = clone._grid_layer_binder(
            getattr(clone, "params", None)
        ).bind(request, market)
        return clone

    def _bound_layout_for_solve(self, request, market):
        """Frozen (bump clone) > time-rebind (calendar roll) > plain bind."""
        return resolve_bound_layout(
            self._grid_layer_binder(getattr(self, "params", None)),
            getattr(self, "_frozen_base_layout", None),
            request,
            market,
        )


class GridBinder:
    """Engine-owned constructor + LRU cache for grid layouts."""

    def __init__(
        self,
        accuracy: str,
        override: Optional[GridConfig],
        *,
        cache_enabled: bool = True,
        cache_max_entries: int = 128,
    ):
        self.config = resolve_config(accuracy, override)
        self._cache_enabled = bool(cache_enabled)
        self._cache_max = int(cache_max_entries)
        self._cache: "OrderedDict[tuple, Layout]" = OrderedDict()

    # -- construction -----------------------------------------------------

    def bind(self, request: GridRequest, market: MarketSnapshot) -> Layout:
        key = (request, market, self.config.key)
        if self._cache_enabled:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
        self._validate_bounds_composition(request)
        spatial = build_space(request, market, self.config)
        time = build_time(request, self.config)
        layout = Layout(
            spatial=spatial, time=time, request=request, config_key=self.config.key
        )
        if self._cache_enabled:
            self._cache[key] = layout
            if len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)
        return layout

    def bind_shared(
        self, requests: Sequence[GridRequest], market: MarketSnapshot
    ) -> List[Layout]:
        """One spatial layout for a same-underlying book (spec §4.2)."""
        if not requests:
            raise ValidationError("bind_shared requires at least one request")
        for r in requests:
            if r.hard_lower is not None or r.hard_upper is not None:
                raise ValidationError(
                    "bind_shared cannot include hard-bounded (continuous "
                    "barrier) products; bind them privately"
                )
        union = GridRequest(
            tau=max(r.tau for r in requests),
            bound_anchors=tuple(
                sorted({a for r in requests for a in r.bound_anchors})
            ),
            critical_prices=tuple(
                sorted({p for r in requests for p in r.critical_prices})
            ),
            hard_lower=None,
            hard_upper=None,
            event_times=(),
        )
        spatial = build_space(union, market, self.config)
        time_by_request: dict = {}
        layouts = []
        for r in requests:
            tl = time_by_request.get(r)
            if tl is None:
                tl = build_time(r, self.config)
                time_by_request[r] = tl
            layouts.append(
                Layout(
                    spatial=spatial,
                    time=tl,
                    request=r,
                    config_key=self.config.key,
                )
            )
        return layouts

    def rebind_time(self, layout: Layout, new_request: GridRequest) -> Layout:
        """Calendar-roll rebuild: new time layout, spatial kept by identity."""
        return Layout(
            spatial=layout.spatial,
            time=build_time(new_request, self.config),
            request=new_request,
            config_key=layout.config_key,
        )

    # -- validation -------------------------------------------------------

    def _validate_bounds_composition(self, request: GridRequest) -> None:
        cfg_lo, cfg_hi = self.config.bounds
        if cfg_lo is not None and request.hard_lower is not None:
            raise ValidationError(
                "GridConfig.bounds lower side conflicts with the product's "
                f"hard_lower ({request.hard_lower}); hard bounds are the "
                "domain edge and cannot be overridden"
            )
        if cfg_hi is not None and request.hard_upper is not None:
            raise ValidationError(
                "GridConfig.bounds upper side conflicts with the product's "
                f"hard_upper ({request.hard_upper}); hard bounds are the "
                "domain edge and cannot be overridden"
            )
        lo = request.hard_lower if request.hard_lower is not None else cfg_lo
        hi = request.hard_upper if request.hard_upper is not None else cfg_hi
        anchors = request.bound_anchors
        if lo is not None and all(a <= lo for a in anchors):
            raise PricingError(
                f"domain lower bound {lo} excludes every bound anchor {anchors}"
            )
        if hi is not None and all(a >= hi for a in anchors):
            raise PricingError(
                f"domain upper bound {hi} excludes every bound anchor {anchors}"
            )
