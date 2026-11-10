"""Local-volatility and native Heston/SLV ADI KO-reset routes."""
from .ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from .local_vol_surface import LocalVolSurfaceMixin
from .vol_continuous_ki import LocalVolBarrierCrossingMixin
from .core.adapters.ko_reset import KOResetAdapter
from .core.adapters.snowball import LocalVolAdapterMixin
from .snowball_vol_pde_solvers import HestonSnowballPDESolver, HestonSLVSnowballPDESolver
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption


class _LocalVolKOResetAdapter(LocalVolAdapterMixin, KOResetAdapter):
    pass


class LocalVolKOResetSnowballPDESolver(LocalVolSurfaceMixin, LocalVolBarrierCrossingMixin,
                                      KOResetSnowballPDESolver):
    _adapter_class = _LocalVolKOResetAdapter

    def __init__(self, params=None, local_vol_surface=None):
        super().__init__(params=params)
        self._init_local_vol(local_vol_surface)

    def price(self, product, pricing_env):
        return self._with_surface(pricing_env, lambda: super(LocalVolKOResetSnowballPDESolver, self).price(product, pricing_env))

    def calculate_event_stats(self, product, pricing_env, **kwargs):
        return self._with_surface(pricing_env, lambda: super(LocalVolKOResetSnowballPDESolver, self).calculate_event_stats(product, pricing_env, **kwargs))

    def price_folded(self, product, pricing_env, overlay, **kwargs):
        return self._with_surface(pricing_env, lambda: super(LocalVolKOResetSnowballPDESolver, self).price_folded(product, pricing_env, overlay, **kwargs))

    def _constructor_kwargs(self):
        return dict(params=self.params, local_vol_surface=self._prebuilt)


class _KOResetADI:
    _supported_product_type = KnockOutResetSnowballOption

    def _event_stats_product_type(self):
        return KnockOutResetSnowballOption

    def _validate_product(self, product):
        KOResetSnowballPDESolver(params=self.params)._validate_product(product)

    def get_critical_points(self, product, pricing_env=None):
        return KOResetSnowballPDESolver(params=self.params).get_critical_points(product, pricing_env)

    def price(self, product, pricing_env, *, lifecycle_state=None):
        from .vol_model_statistics import _family, native_adi
        from quantark.asset.equity.engine.settlement_support import validate_settlement_capability
        validate_settlement_capability(self, product, lifecycle_state)
        self._check_product_type(product)
        if pricing_env is None:
            from quantark.util.exceptions import ValidationError
            raise ValidationError(f"PricingEnvironment is required for {self._solver_name}")
        family = _family(self, product)
        family._validate_product(product)
        if float(product.get_maturity(pricing_env)) <= 0:
            return family._calculate_terminal_value(product, pricing_env.spot, pricing_env)
        if family._valuation_decision(product, pricing_env).knocked_out:
            return family._get_immediate_ko_payoff(product, pricing_env)
        return native_adi(self, product, pricing_env, fold=self._native_fold_overlay)

    _native_fold_overlay = None

    def price_folded(self, product, pricing_env, overlay, *, lifecycle_state=None):
        from quantark.asset.equity.engine.settlement_support import validate_settlement_capability
        from .vol_model_statistics import _family
        validate_settlement_capability(self, product, lifecycle_state)
        self._check_product_type(product)
        if overlay.is_empty:
            return self.price(product, pricing_env, lifecycle_state=lifecycle_state)
        family = _family(self, product)
        family._validate_product(product)
        if float(product.get_maturity(pricing_env)) <= 0:
            family._refuse_expired_fold()
        decision = family._valuation_decision(product, pricing_env)
        if decision.knocked_out:
            return family.price_folded(product, pricing_env, overlay, lifecycle_state=lifecycle_state)
        self._native_fold_overlay = overlay
        try:
            return self.price(product, pricing_env)
        finally:
            self._native_fold_overlay = None

    def calculate_greeks(self, product, pricing_env, *, lifecycle_state=None):
        from quantark.asset.equity.engine.base_engine import BaseEngine
        from quantark.asset.equity.engine.settlement_support import validate_settlement_capability
        validate_settlement_capability(self, product, lifecycle_state)
        return BaseEngine.calculate_greeks(self, product, pricing_env)


class HestonKOResetSnowballPDESolver(_KOResetADI, HestonSnowballPDESolver):
    """Absolute pre/post schedules on the native Heston ADI operator."""
    _solver_name = 'HestonKOResetSnowballPDESolver'


class HestonSLVKOResetSnowballPDESolver(_KOResetADI, HestonSLVSnowballPDESolver):
    """Absolute pre/post schedules on the native SLV ADI operator."""
    _solver_name = 'HestonSLVKOResetSnowballPDESolver'
