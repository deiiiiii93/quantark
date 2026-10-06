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

import numpy as np

from quantark.asset.equity.engine.pde.grid.events import DiscreteEvent, EventOp, EventRecord, fold_decision
from quantark.util.enum import ObservationAggregation, ObservationType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import is_close

from ..prepare import CompositeProblem, Operation, TerminalOutcome, prepare
from .base import FamilyAdapter

__all__ = ["BaseFamilyEvents", "BarrierAdapter", "DoubleBarrierAdapter", "OneTouchAdapter", "DoubleOneTouchAdapter"]

_FACADE = (Operation.PRICE, Operation.GREEKS, Operation.SESSION_OUTPUTS)
_MATURITY_OPS = (Operation.PRICE, Operation.GREEKS, Operation.SESSION_OUTPUTS)
_GREEK_KEYS = ("price", "delta", "gamma")


class BaseFamilyEvents:
    """The base family's discrete observations as ``explain`` reports them (spec 3A §5.4): per
    observation node its records and the projection in force. Evidence only; the march applies
    events through the solver (``_apply_step_modifications``). The valuation instant's observation
    is not marched: it is reported as the readout's decision (spec 3C decision 25)."""

    def __init__(self, rows: tuple, projection):
        self.rows, self.projection = rows, projection

    def evidence(self) -> tuple:
        return (("projection", self.projection.value),) + self.rows


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
        self._validate_expiry_march(product, env)
        # The lifecycle wrapper's check on every price / calculate_greeks call
        # (the knock-out leg of a composite went through it as super().price).
        from quantark.asset.equity.engine.settlement_support import validate_settlement_capability
        validate_settlement_capability(self.solver, product, lifecycle_state)

    def _validate_observation(self, product) -> None:
        pass

    def _validate_expiry_march(self, product, env) -> None:
        """Only the single barrier marches EXPIRY monitoring (one record at maturity, its far-field row; spec 3C
        §5.2 (b)). The double barrier and the touch products have no such row (§9), and their march applied the
        barrier at every step: before maturity they refuse EXPIRY. At maturity it is decision 30's terminal value."""
        if self._marches_expiry or getattr(product, "observation_type", None) != ObservationType.EXPIRY:
            return
        if product.get_maturity(env) > 0:
            raise ValidationError(
                f"{type(self.solver).__name__} does not march EXPIRY monitoring before maturity: it has no far-field "
                "row for a barrier observed only at maturity. Use the analytic engine, or DISCRETE monitoring with "
                "one observation at maturity")

    #: whether the family marches EXPIRY before maturity (the single barrier does; spec 3C §5.2 (b))
    _marches_expiry = False

    # --- the outcomes known at valuation --------------------------------------

    def _continuous_hit_record(self, product, env) -> EventRecord:
        """The hit outcome at valuation under continuous monitoring, as one SET record (spec 3C decision 25)."""
        raise NotImplementedError

    def valuation_event(self, product, env):
        """The valuation instant's event as the march would build it (spec 3C decision 25): under continuous
        monitoring the barrier's SET record; under a schedule the records observed now, built by the march's own
        helper with their level, cash from their own payment time and operation; else None."""
        if getattr(product, "observation_type", None) == ObservationType.CONTINUOUS:
            return DiscreteEvent((self._continuous_hit_record(product, env),))
        solver = self.solver
        records = solver._valuation_instant_records(product, env, solver._observation_resolve_kwargs(product))
        if records is None:
            return None
        schedule = getattr(product, "observation_schedule", None)
        aggregation = schedule.aggregation_mode if schedule is not None else ObservationAggregation.STOP_FIRST_HIT
        return solver._event_for_records(product, env, records, 0.0, aggregation)

    def decision_at(self, product, env, spot, price=0.0, delta=0.0, gamma=0.0):
        """``fold_decision`` of the valuation instant's event at ``spot`` (spec 3C decision 25)."""
        return fold_decision(self.valuation_event(product, env), spot, price, delta, gamma)

    def _maturity_event(self, product, env):
        """The decision at maturity (spec 3C decision 30): continuous monitoring, the hit outcome of any other
        instant; a schedule (EXPIRY: one record at maturity), the maturity records as the march's ``_terminal_event``
        builds them; a DISCRETE schedule with no maturity record, none."""
        if product.observation_type == ObservationType.CONTINUOUS:
            return DiscreteEvent((self._continuous_hit_record(product, env),))
        solver = self.solver
        solver._setup_maturity_observation(product, env, solver._observation_resolve_kwargs(product))
        if product.observation_type == ObservationType.DISCRETE and not solver._has_terminal_observation:
            return None
        return solver._terminal_event(product, env)

    def maturity_value(self, product, env, spot):
        """(price, delta, gamma) at maturity: the solver's terminal payoff at ``spot`` (the continuation), then the
        decision there (spec 3C decision 30). Gamma is zero."""
        value, slope = self.solver._terminal_payoff(product, env, np.array([float(spot)]))
        price, delta, gamma, _ = fold_decision(self._maturity_event(product, env), spot, float(value[0]),
                                               float(slope[0]), 0.0)
        return price, delta, gamma

    def _outputs(self, op, price, delta, gamma):
        """A value known without a march, in the operation's own form (no grid: decided at the base spot)."""
        if op is Operation.GREEKS:
            return {"price": price, "delta": delta, "gamma": gamma}
        if op is Operation.SESSION_OUTPUTS:
            return self._session_outputs(price, None)
        return price

    def terminal_outcome(self, product, env, request, lifecycle_state):
        op = request.operation
        if op in _MATURITY_OPS and product.get_maturity(env) <= 0:
            return TerminalOutcome("expired", self._outputs(op, *self.maturity_value(product, env, env.spot)))
        if op in _FACADE and not getattr(product, "is_knock_in", False):
            price, delta, gamma, fired = self.decision_at(product, env, env.spot)
            if fired:                                             # a SET record fires at the spot: nothing is marched
                return TerminalOutcome("barrier_hit", self._outputs(op, price, delta, gamma))
        return FamilyAdapter.terminal_outcome(self, product, env, request, lifecycle_state)

    def read(self, prepared, result):
        out = super().read(prepared, result)
        op = prepared.request.operation
        if op not in _FACADE:
            return out
        event, spot = self.valuation_event(prepared.product, prepared.env), prepared.env.spot
        if op is Operation.SESSION_OUTPUTS:
            npv = fold_decision(event, spot, out.npv)[0]
            solution = out.solution if out.solution is None else out.solution._replace(decision=event)
            return self._session_outputs(npv, solution)
        if op is Operation.GREEKS:
            price, delta, gamma, _ = fold_decision(event, spot, out["price"], out["delta"], out["gamma"])
            return {"price": price, "delta": delta, "gamma": gamma}
        return fold_decision(event, spot, out)[0]

    def _hit_price(self, product, env):
        """The knock-out's value once its barrier is hit at valuation."""
        raise NotImplementedError

    def prepare_events(self, product, env, layout, tau, request):
        solver = self.solver
        if getattr(product, "observation_type", None) == ObservationType.CONTINUOUS:
            return ()                                   # continuous: the barrier is the domain edge
        t, projection = layout.time.t, solver._base_family_projection()
        total = solver._total_tau if solver._total_tau > 0 else tau
        rows = []
        if 0 in solver._observation_indices:                            # the readout's decision, not marched
            rows.append(("valuation_decision", 0, 0.0, "readout", self.valuation_event(product, env).evidence()))
        for idx in sorted(solver._observation_indices):
            if idx == 0:
                continue
            current = solver._current_time(total, tau - t[idx])         # as _apply_step_modifications
            event = solver._step_event(product, env, idx, current)
            rows.append(("observation", int(idx), float(t[idx]),
                         "nodal" if idx == 0 else projection.value, event.evidence()))
        if solver._has_terminal_observation or product.observation_type == ObservationType.EXPIRY:
            rows.append(("maturity", len(t) - 1, float(t[-1]), projection.value,
                         solver._terminal_event(product, env).evidence()))
        return BaseFamilyEvents(tuple(rows), projection)


class _KnockInAdapter(_BarrierFamilyAdapter):
    """Barrier and double barrier: knock-in = participation × vanilla + R DF(T_pay) − knock-out(R)."""

    def validate(self, product, env, request, lifecycle_state) -> None:
        super().validate(product, env, request, lifecycle_state)
        if request.operation not in _FACADE or not getattr(product, "is_knock_in", False):
            return
        schedule = getattr(product, "observation_schedule", None)
        if schedule is None:
            return
        if schedule.aggregation_mode != ObservationAggregation.STOP_FIRST_HIT:
            raise ValidationError(
                "a knock-in is priced as participation x vanilla + rebate - knock-out, which holds only under "
                f"first-hit monitoring; aggregation {schedule.aggregation_mode.value} is refused (spec 3C decision 8)")
        for rec in schedule.records:
            if rec.payoff is not None and not is_close(rec.payoff, product.rebate):
                raise ValidationError(
                    f"a knock-in has one rebate ({product.rebate}); a schedule record pays {rec.payoff} "
                    "(spec 3C §5.2 (a))")

    def composite(self, product, env, request, lifecycle_state):
        op = request.operation
        if op not in _FACADE:
            return None
        if product.is_knock_out:
            return None
        if product.get_maturity(env) <= 0:
            return None                          # maturity is read before any composite is built (decision 30, P1)
        knocked_in = self.decision_at(product, env, env.spot)[3]      # a SET record fires at the spot (decision 25)
        return self._compose(product, env, request, knock_out=not knocked_in)

    def bump_parts(self, product, env):
        if product.is_knock_out or product.get_maturity(env) <= 0:
            return None                          # at maturity no part product is built (decision 30), nothing to freeze
        solver = self.solver
        parts = {"vanilla": (solver._vanilla_solver(), self._vanilla_product(product))}
        if not self.decision_at(product, env, env.spot)[3]:
            parts["knock_out"] = (solver, self._knock_out_product(product))
        return parts

    def _composition(self, product, env):
        """(participation, constant) of knock-in = participation x vanilla + constant - knock-out (spec 3C §5.2 (a));
        the constant is R x DF(0, T_pay), per contract."""
        return self._participation(product), self._rebate_constant(product, env)

    def _rebate_constant(self, product, env) -> float:
        """R x DF(0, T_pay), per contract: the cash a knock-in pays at the terminal payment time when it never
        knocks in (spec 3C §5.2 (a))."""
        solver = self.solver
        return solver._cashflow_value_at_time(env, product.rebate, 0.0,
                                              solver._terminal_payment_time(product, env)) * solver._multiplier(product)

    def maturity_value(self, product, env, spot):
        """Repair (a)'s identity on terminal values (spec 3C decision 30 table): knocked in, participation x the
        payoff x the terminal payment discount (delta likewise); otherwise the rebate x that discount, delta 0. No
        part product is constructed (a product cannot be built with zero maturity, §4.11). A knock-out takes the
        family's rule."""
        if not product.is_knock_in:
            return super().maturity_value(product, env, spot)
        value, slope = self.solver._vanilla_solver()._terminal_payoff(product, env, np.array([float(spot)]))
        pr = self._participation(product)
        van, dvan = float(value[0]), float(slope[0])
        if pr is not None:
            van, dvan = pr * van, pr * dvan
        if fold_decision(self._maturity_event(product, env), spot, 0.0)[3]:
            return van, dvan, 0.0
        return self._rebate_constant(product, env), 0.0, 0.0

    def _compose(self, product, env, request, *, knock_out):
        solver = self.solver
        # A session's artefacts belong to the knock-in product's own request (its automatic domain); each part's
        # domain is another layout, so a part binds its own in a session. A bump context freezes one layout per
        # part at the base market (spec 3C decision 10); a part with none binds fresh.
        frozen = solver._frozen_parts
        vanilla_solver = solver._vanilla_solver()
        parts = [prepare(vanilla_solver, self._vanilla_product(product), env, request, frozen=frozen.get("vanilla"))]
        if knock_out:
            ko = self._knock_out_product(product)
            parts.append(prepare(solver, ko, env, request, frozen=frozen.get("knock_out")))
        pr, constant = self._composition(product, env)
        op = request.operation
        if op is Operation.SESSION_OUTPUTS:
            from quantark.asset.equity.engine.pde.base_pde_solver import CompositeSolution
            decision = self.valuation_event(product, env)

            def compose(outs):
                van = outs[0]
                vnpv = van.npv if pr is None else pr * van.npv
                if not knock_out:                                   # knocked in: no grid (decided at the base spot)
                    return self._session_outputs(vnpv, None)
                leg = outs[1]
                solution = None
                if request.grid:
                    solution = CompositeSolution(vanilla=van.solution, leg=leg.solution, participation=pr,
                                                 constant=constant, decision=decision)
                npv = vnpv + constant - leg.npv if constant else vnpv - leg.npv
                return self._session_outputs(npv, solution)
        elif op is Operation.GREEKS:
            if pr is None:
                scale = lambda g: g                                            # noqa: E731
            else:
                scale = lambda g: {k: pr * g[k] for k in _GREEK_KEYS}          # noqa: E731
            if knock_out:
                def compose(outs):
                    v = scale(outs[0])
                    out = {k: v[k] - outs[1][k] for k in _GREEK_KEYS}
                    if constant:
                        out["price"] = v["price"] + constant - outs[1]["price"]
                    return out
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
                    if constant:
                        return scale(outs[0]) + constant - outs[1]
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
            contract_multiplier=product.contract_multiplier,
        )

    def _knock_out_product(self, product):
        raise NotImplementedError

    def _participation(self, product):
        raise NotImplementedError


class BarrierAdapter(_KnockInAdapter):
    """``barrier_pde_solver.py``."""

    _marches_expiry = True

    def _product_class(self):
        from quantark.asset.equity.product.option.barrier_option import BarrierOption
        return BarrierOption

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
        ) * solver._multiplier(product)

    def _hit_price(self, product, env):
        return self._rebate_at_valuation(product, env) if product.is_knock_out else None

    def _continuous_hit_record(self, product, env) -> EventRecord:
        return EventRecord(*self.solver._sides(product, product.barrier),
                           self._rebate_at_valuation(product, env) if product.is_knock_out else 0.0, EventOp.SET)

    def observation_maps(self, product, env, layout, tau) -> None:
        # Discrete-monitoring indices on the bound layout (the 0.6.0 grid-build override)
        self.solver._setup_observation_indices(
            product,
            env,
            tau,
            layout.time.t,
            resolve_kwargs=self.solver._observation_resolve_kwargs(product),
        )
        # the far-field row's observations (spec 3C decision 9): after the valuation instant, in time order
        self.solver._far_field_timeline = self.solver._observation_timeline(product, env)

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

        schedule = product.observation_schedule
        if schedule is not None:
            # the leg pays the knock-in's one rebate on every record (validate refused any other payoff)
            import dataclasses
            from quantark.asset.equity.product.option.observation_schedule import ObservationSchedule
            schedule = ObservationSchedule(records=[dataclasses.replace(r, payoff=product.rebate) for r in schedule.records],
                                           aggregation_mode=schedule.aggregation_mode, frequency=schedule.frequency)
        return BarrierOption(
            strike=product.strike,
            option_type=product.option_type,
            barrier=product.barrier,
            barrier_type=ko_type,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
            rebate=product.rebate,          # the leg carries R, paid at maturity: a knock-in's rebate is known then
            participation_rate=product.participation_rate,
            pay_at_hit=False,
            observation_type=product.observation_type,
            observation_dates=product.observation_dates,
            observation_schedule=schedule,
            settlement_convention=product.settlement_convention,
            contract_multiplier=product.contract_multiplier,
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
        ) * solver._multiplier(product)

    def _continuous_hit_record(self, product, env) -> EventRecord:
        return EventRecord(product.lower_barrier, product.upper_barrier,
                           self._hit_price(product, env) if product.is_knock_out else 0.0, EventOp.SET)

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
            rebate=product.rebate,          # the leg carries R, paid at maturity (spec 3C §5.2 (a))
            observation_type=product.observation_type,
            observation_dates=product.observation_dates,
            observation_schedule=product.observation_schedule,
            settlement_convention=product.settlement_convention,
            contract_multiplier=product.contract_multiplier,
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



class OneTouchAdapter(_TouchAdapter):
    """``one_touch_pde_solver.py``."""

    def _product_class(self):
        from quantark.asset.equity.product.option.one_touch_option import OneTouchOption
        return OneTouchOption

    def _is_touch_paying(self, product) -> bool:
        return product.is_one_touch

    def _continuous_hit_record(self, product, env) -> EventRecord:
        return EventRecord(*self.solver._sides(product, product.barrier),
                           self._touched_value(product, env) if product.is_one_touch else 0.0, EventOp.SET)


class DoubleOneTouchAdapter(_TouchAdapter):
    """``double_one_touch_pde_solver.py``."""

    def _product_class(self):
        from quantark.asset.equity.product.option.double_one_touch_option import DoubleOneTouchOption
        return DoubleOneTouchOption

    def _is_touch_paying(self, product) -> bool:
        return product.is_double_one_touch

    def _continuous_hit_record(self, product, env) -> EventRecord:
        return EventRecord(product.lower_barrier, product.upper_barrier,
                           self._touched_value(product, env) if product.is_double_one_touch else 0.0, EventOp.SET)
