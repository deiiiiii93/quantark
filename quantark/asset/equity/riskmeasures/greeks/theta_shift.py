"""One theta step for every theta caller (spec ``docs/superpowers/specs/2026-10-07-theta-observation-boundary-design.md``).

Theta moves the contract from the as-of date ``t0`` to the bumped date ``t1``. ``t0`` is gone: its observation is
decided here on the spot, as the engines decide it at ``t0``, and only its OUTCOME is carried (R4): a knock-in as
lifecycle state, an accumulator's fixing into ``past_observations``, a Phoenix's memory arrears into the product,
cash decided at ``t0`` and paid after ``t1`` as a fixed receivable, a contract ended at ``t0`` as the receivable of
its terminal cash. Then the product moves with ``time_shift(..., keep_landing=True)``: an observation on ``t1`` is
kept at time 0 and observed at the new as-of instant (R1); a schedule with nothing left is ``monitoring_ended`` (R3).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Optional, Tuple
from quantark.execution.errors import CapabilityError
from quantark.util.numerical import is_close, is_zero

__all__ = ["ThetaShift", "shift_for_theta", "bumped_value", "bumped_trade_value", "carried_ended_value",
           "shift_cash_legs"]


@dataclass(frozen=True)
class ThetaShift:
    """The bumped contract and what it carries from ``t0``.

    Attributes:
        product: the bumped product (a copy).
        env: the bumped environment (a copy; ``valuation_date`` is the bumped date).
        lifecycle_state: state the bumped price carries (a knock-in decided at ``t0``), or the caller's.
        fixed_pv: cash decided at ``t0`` and paid after ``t1``, in bumped money.
        ended_at_t0: ``(product, env, payment_time, time_bump)`` when ``t0`` ended the contract: the bumped value
            is then the engine's own ``t0`` value of it, re-discounted to ``t1`` (0 once paid within the bump).
        time_bump: the step ``t1 - t0`` in years.
        settled_value: when ``t0`` ended a contract whose remaining value no engine prices (an accumulator knocked
            out at ``t0``): its booked value at ``t1``; the bumped value is this plus ``fixed_pv``.
    """

    product: object
    env: object
    lifecycle_state: object = None
    fixed_pv: float = 0.0
    ended_at_t0: Optional[Tuple[object, object, float, float]] = None
    time_bump: float = 0.0
    settled_value: Optional[float] = None

    @property
    def terminated(self) -> bool:
        return self.ended_at_t0 is not None or self.settled_value is not None


def _carried_to_t1(cash_pv_at_t0: float, payment_time: float, env_t0, env_t1, time_bump: float) -> float:
    """A fixed cash's value at ``t1`` from its value at ``t0``: re-discounted, or 0 once paid before ``t1``.

    Cash paid ON ``t1`` is still in the value (discount 1 at the new as-of instant, as an engine prices a payment
    at time 0), whatever float residue the two times carry."""
    if payment_time is None:
        return 0.0
    if is_close(payment_time, time_bump):
        payment_time = time_bump
    elif payment_time < time_bump:
        return 0.0
    df0 = float(env_t0.get_discount_factor(payment_time))
    return cash_pv_at_t0 * float(env_t1.get_discount_factor(max(payment_time - time_bump, 0.0))) / df0


def _price(engine, product, env, lifecycle_state) -> float:
    if lifecycle_state is None:
        return float(engine.price(product, env))
    return float(engine.price(product, env, lifecycle_state=lifecycle_state))


# --- autocallables --------------------------------------------------------------------------------------------

def _autocallable_outcome(product, env, env_t1, time_bump, state):
    """(knocked_in, fixed_pv, ended_payment_time, arrears) from the product's own
    ``decide_observations_at_valuation`` (``ended_payment_time`` None while alive).

    The autocallable engines carry a knock-in on the product (``_otc_lifecycle_knocked_in``, as the lifecycle
    tracker and the OTC adapters do), not as ``lifecycle_state``."""
    carried = bool(getattr(state, "knocked_in", False) or getattr(product, "_otc_lifecycle_knocked_in", False))
    decision = product.decide_observations_at_valuation(env.spot, env, knocked_in=carried)
    if decision.knocked_out:
        return carried, 0.0, float(decision.ko_record.settlement_time), None
    fixed_pv = 0.0
    coupon = float(getattr(decision, "coupon", 0.0) or 0.0)
    if coupon:
        from quantark.util.enum import CouponPayType
        if product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY:
            # TODO(theta boundary): an EXPIRY coupon rolls up to the note's end, a knock-out or maturity, so its value
            # at t1 is coupon x E[DF(note end)]; no product or lifecycle state carries a rolled-up coupon yet.
            raise CapabilityError(
                "theta: a coupon decided at the as-of date under CouponPayType.EXPIRY rolls up to the note's end; "
                "no state carries it into the bumped contract")
        pay = _coupon_payment_time(product, env, decision.coupon_index)
        fixed_pv = _carried_to_t1(coupon * float(env.get_discount_factor(pay)), pay, env, env_t1, time_bump)
    knocked_in = bool(decision.knocked_in)
    ended = _ko_reset_schedule_1_end(product, env, time_bump) if not knocked_in else None
    return knocked_in, fixed_pv, ended, getattr(decision, "arrears", None)


def _ko_reset_schedule_1_end(product, env, time_bump):
    """The payment time when a KO-reset's schedule 1 ends within the bump un-knocked: it matured then (rule 2),
    paid with the terminal settlement lag after its last observation; else None."""
    from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
    from quantark.asset.equity.product.option import KnockOutResetSnowballOption

    if not isinstance(product, KnockOutResetSnowballOption):
        return None
    times = [float(t) for t in product.get_pre_ko_observation_profile(env)["observation_times"]]
    if not times or any(t >= time_bump or is_close(t, time_bump) for t in times):
        return None
    terminal = resolve_terminal_timing(product, env)
    return max(times) + float(terminal.payment_time - terminal.determination_time)


def _coupon_payment_time(product, env, index: int) -> float:
    from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
    from quantark.util.enum import CouponPayType

    if product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY:
        return float(resolve_terminal_timing(product, env).payment_time)
    return float(product.resolve_ko_observations(env)[index].settlement_time)


# --- the barrier family ---------------------------------------------------------------------------------------

def _single_up(product) -> bool:
    up = getattr(product, "is_up_barrier", None)
    if up is not None:
        return bool(up)
    return bool(product.is_call())                       # single sharkfin: a call's barrier is above


def _breached(product, spot: float, rec) -> bool:
    if rec.upper_barrier is not None or rec.lower_barrier is not None:
        return spot >= rec.upper_barrier or spot <= rec.lower_barrier
    return spot >= rec.barrier if _single_up(product) else spot <= rec.barrier


def _valuation_records(product, env):
    from quantark.asset.equity.engine.analytical.first_observation import (observed_at_valuation,
                                                                           resolve_current_records)

    kwargs = {}
    for attr, key in (("barrier", "default_barrier"), ("upper_barrier", "default_upper"),
                      ("lower_barrier", "default_lower")):
        if getattr(product, attr, None) is not None:
            kwargs[key] = getattr(product, attr)
    payoff = getattr(product, "rebate", None)
    if payoff is None:
        payoff = getattr(product, "knock_out_rebate", 0.0)
    return observed_at_valuation(resolve_current_records(product.observation_schedule, env, default_payoff=payoff,
                                                         product=product, **kwargs))


def _hit_payment_time(product, env, record) -> float:
    from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
    from quantark.asset.equity.settlement import CashflowKind, SettlementRequest, SettlementResolver

    pays_at_hit = bool(getattr(product, "pay_at_hit", False) or getattr(product, "payment_at_hit", False))
    if not pays_at_hit:
        return float(resolve_terminal_timing(product, env).payment_time)
    if record is not None and record.settlement_time is not None:
        return float(record.settlement_time)
    timing = SettlementResolver.resolve_contingent(
        product,
        SettlementRequest(kind=CashflowKind.HIT, determination_date=env.valuation_date, determination_time=0.0,
                          cashflow_id="theta:hit-at-t0"),
        env,
    )
    return float(timing.payment_time)


def _barrier_outcome(product, env, state):
    """(knocked_in, ended_payment_time) from the records observed at ``t0``, at their own levels."""
    from quantark.util.enum import ObservationType

    if getattr(state, "knocked_in", False) or getattr(product, "_otc_lifecycle_knocked_in", False):
        return False, None                             # already carried; nothing new decided
    record = None
    if product.observation_type == ObservationType.CONTINUOUS:
        hit = product.is_barrier_hit(env.spot)
    elif product.observation_type == ObservationType.DISCRETE:
        hits = [rec for rec in _valuation_records(product, env) if _breached(product, env.spot, rec)]
        hit, record = bool(hits), (hits[0] if hits else None)
    else:                                               # EXPIRY: nothing is observed before maturity
        return False, None
    if not hit:
        return False, None
    if getattr(product, "is_knock_in", False):
        return True, None
    return False, _hit_payment_time(product, env, record)     # ended: the engine's own hit value, re-discounted


# --- the accumulator ------------------------------------------------------------------------------------------

def _accumulator_t0(product, env, env_t1, time_bump):
    """(fixed_pv, terminated) of a fixing at ``t0``, settled on the spot as the engines settle it (R4): its
    settlement is engine-determined cash, a receivable only when paid after ``t1``; a TERMINATION breach ends the
    contract with the knock-out rebate, a SINGLE_DAY breach accrues nothing that day."""
    from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
    from quantark.util.enum import AccumulatorKnockOutType

    schedule = product.observation_schedule
    if schedule is None or not schedule.records:
        return 0.0, False
    resolved = schedule.resolve(env, product=product)
    now = [(raw, rec) for raw, rec in zip(schedule.records, resolved) if is_zero(float(rec.observation_time))]
    if not now:
        return 0.0, False
    raw, rec = now[0]
    use_terminal = (product.settlement_at_expiry and raw.settlement_date is None and raw.settlement_time is None)
    pay = float(resolve_terminal_timing(product, env).payment_time if use_terminal else rec.settlement_time)
    spot = float(env.spot)
    if spot >= product.knock_out_barrier:
        if product.knock_out_type != AccumulatorKnockOutType.TERMINATION:
            return 0.0, False
        cash = product.get_knock_out_rebate_cash() * product.contract_multiplier
        return _carried_to_t1(cash * float(env.get_discount_factor(pay)), pay, env, env_t1, time_bump), True
    cash = product.get_observation_payoff(spot)
    return _carried_to_t1(cash * float(env.get_discount_factor(pay)), pay, env, env_t1, time_bump), False


# --- entry points ---------------------------------------------------------------------------------------------

def shift_for_theta(product, pricing_env, time_bump: float, bumped_date, *, lifecycle_state=None) -> ThetaShift:
    """Decide ``t0``'s observation on the spot, carry its outcome, and move the product to ``bumped_date``."""
    from quantark.asset.equity.product.option import (AccumulatorOption, BarrierOption, DoubleBarrierOption,
                                                      DoubleOneTouchOption, DoubleSharkfinOption,
                                                      KnockOutResetSnowballOption, OneTouchOption, PhoenixOption,
                                                      SingleSharkfinOption, SnowballOption)

    env_t1 = deepcopy(pricing_env)
    env_t1.valuation_date = bumped_date
    shifted = deepcopy(product)
    knocked_in, fixed_pv, ended_pay, arrears = False, 0.0, None, None
    accumulator_ended = False

    if isinstance(product, (SnowballOption, PhoenixOption, KnockOutResetSnowballOption)):
        knocked_in, fixed_pv, ended_pay, arrears = _autocallable_outcome(
            product, pricing_env, env_t1, time_bump, lifecycle_state)
    elif isinstance(product, (BarrierOption, DoubleBarrierOption, OneTouchOption, DoubleOneTouchOption,
                              SingleSharkfinOption, DoubleSharkfinOption)):
        knocked_in, ended_pay = _barrier_outcome(product, pricing_env, lifecycle_state)
    elif isinstance(product, AccumulatorOption):
        fixed_pv, accumulator_ended = _accumulator_t0(product, pricing_env, env_t1, time_bump)

    if ended_pay is not None:
        return ThetaShift(shifted, env_t1, lifecycle_state, fixed_pv, (product, pricing_env, ended_pay, time_bump),
                          time_bump)
    shifted.time_shift(time_bump, bumped_date, env_t1, keep_landing=True)
    if accumulator_ended:                               # no engine prices a terminated accumulator: its booked value
        from quantark.asset.equity.engine.accumulator_valuation import realized_value
        return ThetaShift(shifted, env_t1, lifecycle_state, fixed_pv, None, time_bump,
                          settled_value=realized_value(shifted, env_t1))
    if knocked_in:                                      # carried on the product, as every autocallable engine reads it
        shifted._otc_lifecycle_knocked_in = True
    if arrears is not None and getattr(shifted, "has_memory_coupon", False):
        shifted.coupon_config = replace(shifted.coupon_config, initial_coupon_arrears=float(arrears))
    return ThetaShift(shifted, env_t1, lifecycle_state, fixed_pv, None, time_bump)


def carried_ended_value(shift: ThetaShift, value_t0: float) -> float:
    """A ``t0`` value of a contract ``t0`` ended, at ``t1``: re-discounted from its payment, 0 once paid within the
    bump. Cash legs riding on the ended contract settle with it and carry the same way."""
    _product_t0, env_t0, payment_time, time_bump = shift.ended_at_t0
    return _carried_to_t1(value_t0, payment_time, env_t0, shift.env, time_bump)


def shift_cash_legs(legs, time_bump: float) -> list:
    """The trade's cash legs moved with the product (a leg observation on ``t1`` kept at 0, as the product keeps
    it); a leg whose observations are all past drops out."""
    moved = []
    for leg in legs:
        shift = getattr(leg, "time_shift", None)
        shifted = shift(time_bump, keep_landing=True) if callable(shift) else leg
        if shifted is not None:
            moved.append(shifted)
    return moved


def bumped_value(engine, shift: ThetaShift) -> float:
    """The bumped contract's value: when ``t0`` ended it, the engine's own ``t0`` value re-discounted to ``t1``
    (0 once paid within the bump); else the engine's price with the carried state. Plus the carried receivable."""
    if shift.settled_value is not None:
        return shift.settled_value + shift.fixed_pv
    if shift.ended_at_t0 is not None:
        product_t0, env_t0, _payment_time, _time_bump = shift.ended_at_t0
        value_t0 = _price(engine, product_t0, env_t0, shift.lifecycle_state)
        return carried_ended_value(shift, value_t0) + shift.fixed_pv
    if shift.lifecycle_state is not None and not getattr(engine, "supports_lifecycle_state", False):
        raise CapabilityError(
            f"{type(engine).__name__} cannot carry the caller's lifecycle state into theta "
            "(it does not support lifecycle_state)")
    return _price(engine, shift.product, shift.env, shift.lifecycle_state) + shift.fixed_pv


def bumped_trade_value(bump_engine, shift: ThetaShift, legs, *, streams, unit_of):
    """``(npv, leg_sum)`` of a trade with cash legs at ``t1``, one ``price_with_events`` solve.

    When ``t0`` ended the contract, the trade's ``t0`` values (product and legs, on the ``t0`` distribution) are
    carried to ``t1``; else the bumped product is solved and the legs, moved with it, are valued on its distribution.
    ``unit_of(env)`` is the caller's unit notional."""
    from quantark.cashleg.leg_valuator import value_leg

    if shift.settled_value is not None:
        if legs:
            raise CapabilityError("theta: cash legs on a contract settled at the as-of date have no distribution")
        return shift.settled_value + shift.fixed_pv, 0.0
    if shift.ended_at_t0 is not None:
        product_t0, env_t0, _payment_time, _time_bump = shift.ended_at_t0
        result = bump_engine.price_with_events(product_t0, env_t0, streams=streams)
        unit = unit_of(env_t0)
        legs_t0 = sum(value_leg(leg, result.event_distribution, env_t0, unit) for leg in legs)
        return (carried_ended_value(shift, float(result.npv)) + shift.fixed_pv,
                carried_ended_value(shift, legs_t0))
    result = bump_engine.price_with_events(shift.product, shift.env, streams=streams)
    unit = unit_of(shift.env)
    moved = shift_cash_legs(legs, shift.time_bump)
    return (float(result.npv) + shift.fixed_pv,
            sum(value_leg(leg, result.event_distribution, shift.env, unit) for leg in moved))
