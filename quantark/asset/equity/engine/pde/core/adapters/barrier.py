"""The barrier family: barrier, double barrier, one-touch, double one-touch.

Each solver's ``price`` / ``calculate_greeks`` moved verbatim, in the order it
tested its branches: the raising checks become ``validate``; the outcomes known
at valuation (knocked out, touched, expired) become ``terminal_outcome``; a
knock-in is a composite of prepared problems (decision 21) -- the vanilla alone
once knocked in, participation × vanilla − knock-out before. The session seam
keeps the base semantics (no facade branches), as the base ``_price_with_solution``
did.
"""
from __future__ import annotations

from quantark.util.exceptions import PricingError

from ..prepare import CompositeProblem, Operation, SessionArtefacts, TerminalOutcome, prepare
from .base import FamilyAdapter

__all__ = ["BarrierAdapter", "DoubleBarrierAdapter", "OneTouchAdapter", "DoubleOneTouchAdapter"]

_FACADE = (Operation.PRICE, Operation.GREEKS)
_GREEK_KEYS = ("price", "delta", "gamma")


class _BarrierFamilyAdapter(FamilyAdapter):
    def _product_class(self):
        """The one product type the solver's facades accept."""
        raise NotImplementedError

    def validate(self, product, env, request, lifecycle_state) -> None:
        if request.operation not in _FACADE:
            return
        product_class = self._product_class()
        if not isinstance(product, product_class):
            raise PricingError(
                f"{type(self.solver).__name__} only supports {product_class.__name__}, "
                f"got {type(product).__name__}"
            )
        self._validate_observation(product)
        # The lifecycle wrapper's check on every price / calculate_greeks call
        # (the knock-out leg of a composite went through it as super().price).
        from quantark.asset.equity.engine.settlement_support import validate_settlement_capability
        validate_settlement_capability(self.solver, product, lifecycle_state)

    def _validate_observation(self, product) -> None:
        pass

    # --- the outcomes known at valuation --------------------------------------

    def terminal_outcome(self, product, env, request, lifecycle_state):
        op = request.operation
        if op is Operation.PRICE:
            # price(): the hit test comes first
            if product.is_barrier_hit(env.spot):
                hit = self._hit_price(product, env)
                if hit is not None:
                    return TerminalOutcome("barrier_hit", hit)
            return super().terminal_outcome(product, env, request, lifecycle_state)
        if op is Operation.GREEKS:
            # calculate_greeks(): expiry first, then the hit test
            expired = super().terminal_outcome(product, env, request, lifecycle_state)
            if expired is not None:
                return expired
            if product.is_barrier_hit(env.spot):
                hit = self._hit_greeks(product, env)
                if hit is not None:
                    return TerminalOutcome("barrier_hit", hit)
            return None
        return super().terminal_outcome(product, env, request, lifecycle_state)

    def _hit_price(self, product, env):
        """The value once the barrier is hit at valuation; None = the knock-in composite."""
        raise NotImplementedError

    def _hit_greeks(self, product, env):
        raise NotImplementedError


class _KnockInAdapter(_BarrierFamilyAdapter):
    """Barrier and double barrier: knock-in = participation × vanilla − knock-out."""

    def composite(self, product, env, request, lifecycle_state):
        op = request.operation
        if op not in _FACADE:
            return None
        if op is Operation.GREEKS and product.get_maturity(env) <= 0:
            return None                          # calculate_greeks: expiry is tested first
        if product.is_barrier_hit(env.spot):
            if product.is_knock_out:
                return None                      # the rebate (terminal_outcome)
            return self._compose(product, env, request, knock_out=False)   # knocked in: the vanilla
        if product.is_knock_in:
            return self._compose(product, env, request, knock_out=True)
        return None

    def _compose(self, product, env, request, *, knock_out):
        from quantark.asset.equity.engine.pde.european_pde_solver import EuropeanPDESolver

        solver = self.solver
        vanilla_solver = EuropeanPDESolver(solver.params)    # fresh on every call, as today: no frozen layout
        parts = [prepare(vanilla_solver, self._vanilla_product(product), env, request)]
        if knock_out:
            ko = self._knock_out_product(product)
            parts.append(prepare(solver, ko, env, request, frozen=solver._frozen_base_layout,
                                 session=SessionArtefacts.from_solver(solver, ko, env)))
        pr = self._participation(product)
        greeks = request.operation is Operation.GREEKS
        if greeks:
            if pr is None:
                scale = lambda g: g                                            # noqa: E731
            else:
                scale = lambda g: {k: pr * g[k] for k in _GREEK_KEYS}          # noqa: E731
            if knock_out:
                def compose(outs):
                    v = scale(outs[0])
                    return {k: v[k] - outs[1][k] for k in _GREEK_KEYS}
            else:
                def compose(outs):
                    return scale(outs[0])
        else:
            if pr is None:
                scale = lambda p: p                                            # noqa: E731
            else:
                scale = lambda p: pr * p                                       # noqa: E731
            if knock_out:
                def compose(outs):
                    return scale(outs[0]) - outs[1]
            else:
                def compose(outs):
                    return scale(outs[0])
        description = ("knock-in = participation × vanilla − knock-out" if knock_out
                       else "knocked in = participation × vanilla")
        return CompositeProblem(parts=tuple(parts), compose=compose, request=request,
                                description=description)

    @staticmethod
    def _vanilla_product(product):
        # _price_vanilla / _calculate_greeks_vanilla: the product construction, verbatim
        from quantark.asset.equity.product.option import EuropeanVanillaOption

        return EuropeanVanillaOption(
            strike=product.strike,
            option_type=product.option_type,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
            settlement_convention=product.settlement_convention,
        )

    def _knock_out_product(self, product):
        raise NotImplementedError

    def _participation(self, product):
        raise NotImplementedError


class BarrierAdapter(_KnockInAdapter):
    """``barrier_pde_solver.py``."""

    def _product_class(self):
        from quantark.asset.equity.product.option.barrier_option import BarrierOption
        return BarrierOption

    def _validate_observation(self, product) -> None:
        from quantark.util.enum import ObservationType

        if getattr(product, "observation_type", None) == ObservationType.EXPIRY:
            raise PricingError(
                "BarrierPDESolver does not support EXPIRY observation_type. "
                "Use BarrierAnalyticalEngine for expiry-only monitoring."
            )

    def _rebate_at_valuation(self, product, env):
        solver = self.solver
        maturity = product.get_maturity(env)  # noqa: F841  (read as price() did: an expired dated product raises)
        settlement_time = (
            solver._event_payment_time(product, env, 0.0)
            if product.pay_at_hit
            else solver._terminal_payment_time(product, env)
        )
        return solver._cashflow_value_at_time(
            pricing_env=env,
            cashflow=product.rebate,
            current_time=0.0,
            settlement_time=settlement_time,
        )

    def _hit_price(self, product, env):
        return self._rebate_at_valuation(product, env) if product.is_knock_out else None

    def _hit_greeks(self, product, env):
        # Knocked out: return rebate Greeks (rebate is constant, so delta=gamma=0)
        if not product.is_knock_out:
            return None
        return {"price": self._rebate_at_valuation(product, env), "delta": 0.0, "gamma": 0.0}

    def observation_maps(self, product, env, layout, tau) -> None:
        # The deleted _build_grids override: discrete-monitoring indices on the bound layout
        self.solver._setup_observation_indices(
            product,
            env,
            tau,
            layout.time.t,
            resolve_kwargs={
                "default_barrier": product.barrier,
                "default_payoff": product.rebate,
                "require_single": True,
            },
        )

    def _knock_out_product(self, product):
        # _price_knock_out / _calculate_greeks_knock_out, verbatim
        from quantark.asset.equity.product.option.barrier_option import BarrierOption
        from quantark.util.enum import BarrierType

        if product.barrier_type == BarrierType.UP_IN:
            ko_type = BarrierType.UP_OUT
        elif product.barrier_type == BarrierType.DOWN_IN:
            ko_type = BarrierType.DOWN_OUT
        else:
            ko_type = product.barrier_type

        return BarrierOption(
            strike=product.strike,
            option_type=product.option_type,
            barrier=product.barrier,
            barrier_type=ko_type,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
            rebate=0.0,  # Zero rebate for decomposition
            participation_rate=product.participation_rate,
            pay_at_hit=product.pay_at_hit,
            observation_type=product.observation_type,
            observation_dates=product.observation_dates,
            observation_schedule=product.observation_schedule,
            settlement_convention=product.settlement_convention,
        )

    def _participation(self, product):
        return product.participation_rate


class DoubleBarrierAdapter(_KnockInAdapter):
    """``double_barrier_pde_solver.py``: no EXPIRY check; the vanilla carries no participation."""

    def _product_class(self):
        from quantark.asset.equity.product.option.double_barrier_option import DoubleBarrierOption
        return DoubleBarrierOption

    def _hit_price(self, product, env):
        if not product.is_knock_out:
            return None
        solver = self.solver
        return solver._cashflow_value_at_time(
            env,
            product.rebate,
            0.0,
            solver._event_payment_time(product, env, 0.0),
        )

    def _hit_greeks(self, product, env):
        # Already knocked out: rebate is constant, so delta=gamma=0 (undiscounted, as today)
        if not product.is_knock_out:
            return None
        return {"price": product.rebate, "delta": 0.0, "gamma": 0.0}

    def _knock_out_product(self, product):
        from quantark.asset.equity.product.option.double_barrier_option import DoubleBarrierOption
        from quantark.util.enum import DoubleBarrierType

        return DoubleBarrierOption(
            strike=product.strike,
            option_type=product.option_type,
            upper_barrier=product.upper_barrier,
            lower_barrier=product.lower_barrier,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
            rebate=0.0,  # Zero rebate for decomposition
            observation_type=product.observation_type,
            observation_dates=product.observation_dates,
            observation_schedule=product.observation_schedule,
            settlement_convention=product.settlement_convention,
        )

    def _participation(self, product):
        return None     # _price_vanilla multiplies by nothing (not by 1.0)


class _TouchAdapter(_BarrierFamilyAdapter):
    """One-touch and double one-touch: a touched one-touch pays the rebate, a touched no-touch nothing."""

    def _is_touch_paying(self, product) -> bool:
        raise NotImplementedError

    def _touched_value(self, product, env):
        solver = self.solver
        settlement_time = (
            solver._event_payment_time(product, env, 0.0)
            if product.payment_at_hit
            else solver._terminal_payment_time(product, env)
        )
        return solver._cashflow_value_at_time(
            env,
            product.rebate,
            0.0,
            settlement_time,
        )

    def _hit_price(self, product, env):
        # No-touch already failed: 0.0
        return self._touched_value(product, env) if self._is_touch_paying(product) else 0.0

    def _hit_greeks(self, product, env):
        # Already touched, fixed rebate (delta=gamma=0); no-touch already failed
        if self._is_touch_paying(product):
            return {"price": self._touched_value(product, env), "delta": 0.0, "gamma": 0.0}
        return {"price": 0.0, "delta": 0.0, "gamma": 0.0}


class OneTouchAdapter(_TouchAdapter):
    """``one_touch_pde_solver.py``."""

    def _product_class(self):
        from quantark.asset.equity.product.option.one_touch_option import OneTouchOption
        return OneTouchOption

    def _is_touch_paying(self, product) -> bool:
        return product.is_one_touch


class DoubleOneTouchAdapter(_TouchAdapter):
    """``double_one_touch_pde_solver.py``."""

    def _product_class(self):
        from quantark.asset.equity.product.option.double_one_touch_option import DoubleOneTouchOption
        return DoubleOneTouchOption

    def _is_touch_paying(self, product) -> bool:
        return product.is_double_one_touch
