"""
Adapter helpers to map products into quadrature core inputs.
"""

import math
from typing import Sequence

import numpy as np

from asset.equity.product.option import BarrierOption, OneTouchOption
from asset.equity.product.option.observation_schedule import ResolvedObservationRecord
from util.exceptions import ValidationError
from util.numerical import Tolerance

from .quad_core import QuadCoreInputs


def build_barrier_quad_inputs(
    product: BarrierOption,
    resolved: Sequence[ResolvedObservationRecord],
    maturity: float,
    rate: float,
) -> QuadCoreInputs:
    obs_times, barriers, payoffs, settlement_times = _extract_observations(
        resolved, maturity, product.is_up_barrier
    )

    if product.is_up_barrier:
        k_plus = np.array(barriers, dtype=float)
        k_minus = np.zeros_like(k_plus)
    else:
        k_minus = np.array(barriers, dtype=float)
        k_plus = np.full_like(k_minus, math.inf, dtype=float)

    a_minus = np.zeros_like(k_minus, dtype=float)
    b_minus = np.zeros_like(k_minus, dtype=float)
    a_plus = np.zeros_like(k_plus, dtype=float)
    b_plus = np.zeros_like(k_plus, dtype=float)

    rebate_values = _discount_rebates(
        payoffs=payoffs,
        observation_times=obs_times,
        settlement_times=settlement_times,
        maturity=maturity,
        rate=rate,
        pay_at_hit=product.pay_at_hit,
    )

    if product.rebate > 0.0:
        if product.is_up_barrier:
            b_plus = rebate_values
        else:
            b_minus = rebate_values

    maturity_barrier = _resolve_maturity_barrier(obs_times, barriers, maturity)
    a_terminal, b_terminal = _apply_terminal_payoff_structure(
        product,
        maturity_barrier,
        k_minus,
        k_plus,
        a_minus,
        b_minus,
        a_plus,
        b_plus,
    )

    return QuadCoreInputs(
        observation_times=obs_times,
        k_minus=k_minus,
        k_plus=k_plus,
        a_minus=a_minus,
        b_minus=b_minus,
        a_plus=a_plus,
        b_plus=b_plus,
        a_terminal=a_terminal,
        b_terminal=b_terminal,
    )


def build_one_touch_quad_inputs(
    product: OneTouchOption,
    resolved: Sequence[ResolvedObservationRecord],
    maturity: float,
    rate: float,
) -> QuadCoreInputs:
    obs_times, barriers, payoffs, settlement_times = _extract_observations(
        resolved, maturity, product.is_up_barrier
    )

    if product.is_up_barrier:
        k_plus = np.array(barriers, dtype=float)
        k_minus = np.zeros_like(k_plus)
    else:
        k_minus = np.array(barriers, dtype=float)
        k_plus = np.full_like(k_minus, math.inf, dtype=float)

    a_minus = np.zeros_like(k_minus, dtype=float)
    b_minus = np.zeros_like(k_minus, dtype=float)
    a_plus = np.zeros_like(k_plus, dtype=float)
    b_plus = np.zeros_like(k_plus, dtype=float)

    rebate_values = _discount_rebates(
        payoffs=payoffs,
        observation_times=obs_times,
        settlement_times=settlement_times,
        maturity=maturity,
        rate=rate,
        pay_at_hit=product.payment_at_hit,
    )

    if product.rebate > 0.0:
        if product.is_up_barrier:
            b_plus = rebate_values
        else:
            b_minus = rebate_values

    return QuadCoreInputs(
        observation_times=obs_times,
        k_minus=k_minus,
        k_plus=k_plus,
        a_minus=a_minus,
        b_minus=b_minus,
        a_plus=a_plus,
        b_plus=b_plus,
        a_terminal=0.0,
        b_terminal=0.0,
    )


def _extract_observations(
    resolved: Sequence[ResolvedObservationRecord],
    maturity: float,
    is_up_barrier: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs_times = np.array([rec.observation_time for rec in resolved], dtype=float)
    barriers = np.array([rec.barrier for rec in resolved], dtype=float)
    payoffs = np.array([rec.payoff for rec in resolved], dtype=float)
    settlement_times = np.array(
        [
            rec.settlement_time if rec.settlement_time is not None else rec.observation_time
            for rec in resolved
        ],
        dtype=float,
    )

    if obs_times.size == 0:
        raise ValidationError("Resolved observation schedule is empty.")

    if maturity - obs_times[-1] > Tolerance.ZERO:
        obs_times = np.append(obs_times, maturity)
        if is_up_barrier:
            barriers = np.append(barriers, math.inf)
        else:
            barriers = np.append(barriers, 0.0)
        payoffs = np.append(payoffs, 0.0)
        settlement_times = np.append(settlement_times, maturity)

    return obs_times, barriers, payoffs, settlement_times


def _discount_rebates(
    payoffs: np.ndarray,
    observation_times: np.ndarray,
    settlement_times: np.ndarray,
    maturity: float,
    rate: float,
    pay_at_hit: bool,
) -> np.ndarray:
    if payoffs.size == 0:
        return payoffs
    if pay_at_hit:
        delays = np.maximum(settlement_times - observation_times, 0.0)
        discount = np.exp(-rate * delays)
    else:
        discount = np.exp(-rate * (maturity - observation_times))
    return payoffs * discount


def _resolve_maturity_barrier(
    observation_times: np.ndarray, barriers: np.ndarray, maturity: float
) -> float | None:
    maturity_mask = np.isclose(
        observation_times, maturity, atol=Tolerance.ZERO, rtol=0.0
    )
    if not np.any(maturity_mask):
        return None
    value = float(barriers[np.where(maturity_mask)[0][-1]])
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def _apply_terminal_payoff_structure(
    product: BarrierOption,
    maturity_barrier: float | None,
    k_minus: np.ndarray,
    k_plus: np.ndarray,
    a_minus: np.ndarray,
    b_minus: np.ndarray,
    a_plus: np.ndarray,
    b_plus: np.ndarray,
) -> tuple[float, float]:
    participation = product.participation_rate
    strike = product.strike
    is_call = product.is_call()

    a_terminal = 0.0
    b_terminal = 0.0

    if maturity_barrier is None:
        if is_call:
            k_minus[-1] = strike
            a_terminal = participation
            b_terminal = -participation * strike
        else:
            k_plus[-1] = strike
            a_terminal = -participation
            b_terminal = participation * strike
        return a_terminal, b_terminal

    barrier = maturity_barrier

    if product.is_up_barrier:
        k_plus[-1] = barrier
        if is_call:
            if barrier > strike:
                k_minus[-1] = strike
                a_terminal = participation
                b_terminal = -participation * strike
        else:
            if barrier > strike:
                k_minus[-1] = strike
                a_minus[-1] = -participation
                b_minus[-1] = participation * strike
            else:
                a_terminal = -participation
                b_terminal = participation * strike
        return a_terminal, b_terminal

    k_minus[-1] = barrier
    if is_call:
        if barrier < strike:
            k_plus[-1] = strike
            a_plus[-1] = participation
            b_plus[-1] = -participation * strike
        else:
            a_terminal = participation
            b_terminal = -participation * strike
    else:
        if barrier < strike:
            k_plus[-1] = strike
            a_terminal = -participation
            b_terminal = participation * strike

    return a_terminal, b_terminal
