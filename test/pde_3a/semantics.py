"""Base-family products covering every row of spec 3A §5.4's operation table (Task 6)."""
from quantark.asset.equity.engine.pde import (BarrierPDESolver, DoubleBarrierPDESolver, DoubleOneTouchPDESolver,
                                              OneTouchPDESolver)
from quantark.asset.equity.product.option import (BarrierOption, DoubleBarrierOption, DoubleOneTouchOption,
                                                  OneTouchOption)
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.util.enum import BarrierType, ObservationAggregation, ObservationType, OptionType
from quantark.util.enum.option_enums import BarrierDirection, DoubleBarrierType, TouchType

MODES = {"stop": ObservationAggregation.STOP_FIRST_HIT, "acc": ObservationAggregation.ACCUMULATE}
DATES = [0.25, 0.5, 0.75, 1.0]
DISC = dict(observation_type=ObservationType.DISCRETE, maturity=1.0)


def _sched(mode, *records):
    return ObservationSchedule(records=[ObservationRecord(**r) for r in records], aggregation_mode=MODES[mode])


def cases():
    """name -> (solver class, product factory, the expected ops (interior at 0.5, maturity))."""
    out = {}
    for m in MODES:
        op = "add" if m == "acc" else "set"
        single = lambda b, m=m: _sched(m, dict(observation_time=0.5, barrier=b, payoff=2.0),   # noqa: E731
                                       dict(observation_time=1.0, barrier=b, payoff=3.0))
        corridor = lambda m=m: _sched(m, dict(observation_time=0.5, upper_barrier=120.0, lower_barrier=80.0, payoff=1.0),  # noqa: E731
                                      dict(observation_time=1.0, upper_barrier=118.0, lower_barrier=82.0, payoff=2.0))
        out[f"barrier_uo_call|{m}"] = (BarrierPDESolver, lambda s=single: BarrierOption(
            strike=100.0, option_type=OptionType.CALL, barrier=125.0, barrier_type=BarrierType.UP_OUT, rebate=2.0,
            pay_at_hit=False, observation_schedule=s(125.0), **DISC), (op, op))
        out[f"barrier_do_put|{m}"] = (BarrierPDESolver, lambda s=single: BarrierOption(
            strike=100.0, option_type=OptionType.PUT, barrier=85.0, barrier_type=BarrierType.DOWN_OUT, rebate=2.0,
            pay_at_hit=True, observation_schedule=s(85.0), **DISC), (op, op))
        out[f"barrier_di_put|{m}"] = (BarrierPDESolver, lambda s=single: BarrierOption(
            strike=100.0, option_type=OptionType.PUT, barrier=85.0, barrier_type=BarrierType.DOWN_IN,
            observation_schedule=s(85.0), **DISC), (op, op))
        out[f"double_ko|{m}"] = (DoubleBarrierPDESolver, lambda c=corridor: DoubleBarrierOption(
            100.0, OptionType.CALL, 120.0, 80.0, DoubleBarrierType.KNOCK_OUT, rebate=1.0,
            observation_schedule=c(), **DISC), (op, op))
        out[f"double_ki|{m}"] = (DoubleBarrierPDESolver, lambda c=corridor: DoubleBarrierOption(
            100.0, OptionType.CALL, 120.0, 80.0, DoubleBarrierType.KNOCK_IN, observation_schedule=c(), **DISC), (op, op))
        for tt, name in ((TouchType.ONE_TOUCH, "one_touch"), (TouchType.NO_TOUCH, "no_touch")):
            ops = (op, op) if tt is TouchType.ONE_TOUCH else (op, "set")       # a no-touch zeroes with SET at maturity
            out[f"{name}|{m}"] = (OneTouchPDESolver, lambda tt=tt, s=single: OneTouchOption(
                barrier=115.0, barrier_direction=BarrierDirection.UP, rebate=1.0, payment_at_hit=False,
                touch_type=tt, observation_schedule=s(115.0), **DISC), ops)
        for tt, name in ((TouchType.DOUBLE_ONE_TOUCH, "double_one_touch"), (TouchType.DOUBLE_NO_TOUCH, "double_no_touch")):
            ops = (op, op) if tt is TouchType.DOUBLE_ONE_TOUCH else (op, "set")
            out[f"{name}|{m}"] = (DoubleOneTouchPDESolver, lambda tt=tt, m=m: DoubleOneTouchOption(
                upper_barrier=115.0, lower_barrier=85.0, rebate=1.0, payment_at_hit=False, touch_type=tt,
                observation_schedule=_sched(m, dict(observation_time=0.5, upper_barrier=115.0, lower_barrier=85.0, payoff=1.0),
                                            dict(observation_time=1.0, upper_barrier=115.0, lower_barrier=85.0, payoff=1.0)),
                **DISC), ops)
    out["barrier_uo_call|dates"] = (BarrierPDESolver, lambda: BarrierOption(
        strike=100.0, option_type=OptionType.CALL, barrier=125.0, barrier_type=BarrierType.UP_OUT, rebate=2.0,
        observation_dates=DATES, **DISC), ("set", "set"))
    out["double_ko|dates"] = (DoubleBarrierPDESolver, lambda: DoubleBarrierOption(
        100.0, OptionType.CALL, 120.0, 80.0, DoubleBarrierType.KNOCK_OUT, rebate=1.0, observation_dates=DATES, **DISC),
        ("set", "set"))
    out["one_touch|dates"] = (OneTouchPDESolver, lambda: OneTouchOption(
        barrier=115.0, barrier_direction=BarrierDirection.UP, rebate=1.0, payment_at_hit=True,
        touch_type=TouchType.ONE_TOUCH, observation_dates=DATES, **DISC), ("set", "set"))
    out["double_one_touch|dates"] = (DoubleOneTouchPDESolver, lambda: DoubleOneTouchOption(
        upper_barrier=115.0, lower_barrier=85.0, rebate=1.0, payment_at_hit=True,
        touch_type=TouchType.DOUBLE_ONE_TOUCH, observation_dates=DATES, **DISC), ("set", "set"))
    out["valuation_date_only"] = (BarrierPDESolver, lambda: BarrierOption(
        strike=100.0, option_type=OptionType.CALL, barrier=125.0, barrier_type=BarrierType.UP_OUT, rebate=2.0,
        observation_dates=[0.0], **DISC), None)
    return out
