"""Shared deterministic-GBM coefficient clock for continuous KI.

The engines retain their own spatial kernels and native time density. Known
coefficient knots are mandatory; smooth curves use explicit piecewise-constant
model controls. Exact describes the coefficient model, not numerical accuracy.
"""
import numpy as np

from quantark.param import FlatRateCurve, FlatVolSurface, ContinuousDividendYield
from quantark.param.rrf.rate_curve import LogLinearRateCurve, ParallelShiftRateCurve
from quantark.param.vol.vol_surface import TermStructureVolSurface, ParallelShiftVolSurface
from quantark.param.div.dividend_yield import NoDividend, ParallelShiftDividendYield
from quantark.param.div.forward_carry_curve import _CarryImpliedDividendYield
from quantark.util.exceptions import NumericalError, ValidationError


def continuous_coefficient_model(env):
    rate, div, vol = env.rate_curve, env.div_yield, env.vol_surface
    while type(rate) is ParallelShiftRateCurve:
        rate = rate.base_curve
    while type(div) is ParallelShiftDividendYield:
        div = div.base
    base_vol = vol
    while type(base_vol) is ParallelShiftVolSurface:
        base_vol = base_vol.base
    exact_vol = type(vol) in {FlatVolSurface, TermStructureVolSurface} or type(base_vol) is FlatVolSurface
    exact_rate = type(rate) in {FlatRateCurve, LogLinearRateCurve}
    carry = div._carry if type(div) is _CarryImpliedDividendYield else None
    carry_rate = div._rate if carry is not None else None
    while type(carry_rate) is ParallelShiftRateCurve:
        carry_rate = carry_rate.base_curve
    exact_div = div is None or type(div) in {ContinuousDividendYield, NoDividend} or (
        carry is not None and type(carry_rate) in {FlatRateCurve, LogLinearRateCurve}
    )
    curves = [rate, div, base_vol]
    if carry is not None:
        curves.extend([carry, carry_rate])
    knots = set()
    for curve in curves:
        knots.update(float(t) for t in getattr(curve, 'times', ()))
        knots.update(float(p[0]) for p in getattr(curve, 'pillars', ()))
        if carry is not None and curve is carry:
            knots.update(float(p[0]) for p in carry.nodes)
    return exact_rate and exact_div and exact_vol, knots


def continuous_time_grid(env, times, params):
    """Known knots plus optional smooth-curve refinement on the engine's clock."""
    times = [float(t) for t in times]
    if not times or any(not np.isfinite(t) or t < 0 for t in times):
        raise ValidationError('continuous KI event times must be finite and nonnegative')
    exact, knots = continuous_coefficient_model(env)
    mode = params.continuous_term_structure
    if not exact and mode == 'exact':
        raise NotImplementedError(
            "continuous KI with these term curves requires continuous_term_structure='piecewise_constant' and a time-refinement qualification"
        )
    horizon = max(times)
    nodes = set(times) | {0.0} | {t for t in knots if 0 < t < horizon}
    maximum = getattr(params, 'max_events', 100_000)
    density = params.continuous_steps_per_year
    if not exact and density:
        base = sorted(nodes)
        predicted = len(nodes)
        for left, right in zip(base[:-1], base[1:]):
            count = max(1, int(np.ceil((right-left)*density)))
            predicted += count - 1
            if predicted > maximum + 1:
                raise NumericalError('continuous KI time subdivision exceeds max_events')
            nodes.update(left + (right-left)*i/count for i in range(1, count))
    if 0.0 not in times:
        nodes.discard(0.0)
    if len(nodes) > maximum:
        raise NumericalError('continuous KI event count exceeds max_events')
    return sorted(nodes), 'piecewise_constant_exact' if exact else 'piecewise_constant_approximation'


def continuous_ki_ever(product, env, params):
    """Underlying monitoring after an immediate KO, without repricing the note."""
    from types import SimpleNamespace
    from quantark.asset.equity.param import QuadV2Params
    from quantark.asset.equity.engine.quad.v2.basis import make_mesh
    from quantark.asset.equity.engine.quad.v2.context import interval_moments
    from quantark.asset.equity.engine.quad.v2.continuous import prepare_continuous_ki_monitor
    from quantark.asset.equity.engine.quad.v2.operator import Operator

    if bool(getattr(product, '_otc_lifecycle_knocked_in', False)):
        return 1.0
    horizon = float(product.get_pre_maturity_time(env) if hasattr(product, 'get_pre_maturity_time') else product.get_maturity(env))
    barrier = float(product.barrier_config.ki_barrier)
    if horizon <= 0:
        return float(env.spot >= barrier if product.is_reverse else env.spot <= barrier)
    settings = QuadV2Params(continuous_term_structure=params.continuous_term_structure,
                            continuous_steps_per_year=max(1, params.continuous_steps_per_year))
    times, _ = continuous_time_grid(env, [horizon], settings)
    means, variances, discounts = interval_moments(env, product.strike, times)
    mesh = make_mesh(np.log([env.spot]), means, variances, settings, align=np.log(barrier))
    contract = SimpleNamespace(events=[SimpleNamespace(time=t) for t in times],
                               metadata={'ki_monitoring_end': horizon},
                               continuous_barrier=float(np.log(barrier)), reverse=product.is_reverse)
    monitor = prepare_continuous_ki_monitor(contract, means, variances, discounts, mesh,
                                           settings, None if mesh is None else Operator(mesh, settings))
    return float(monitor.probability([env.spot])[0])
