"""Greeks-based Taylor explain (spec §7)."""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple

from quantark.asset.equity.param import EngineParams
from quantark.asset.equity.riskmeasures.greeks.bump_envs import resolve_theta_bump_mode
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.pnlexplain.base import ExplainMethod, ExplainRow, Factor, RowKind
from quantark.pnlexplain.config import PnLExplainConfig, TERM_FACTOR, resolve_stencil, resolved_subrows
from quantark.pnlexplain.equity.scenario import ScenarioCache
from quantark.pnlexplain.equity.snapshot import is_terminal
from quantark.util.exceptions import ValidationError

# coefficient, exponents (spot, vol, rate, div, time)
TERM_SPEC: Dict[str, Tuple[float, Tuple[int, int, int, int, int]]] = {
    "delta": (1.0, (1, 0, 0, 0, 0)), "gamma": (0.5, (2, 0, 0, 0, 0)), "speed": (1.0 / 6.0, (3, 0, 0, 0, 0)),
    "vega": (1.0, (0, 1, 0, 0, 0)), "volga": (0.5, (0, 2, 0, 0, 0)), "vanna": (1.0, (1, 1, 0, 0, 0)),
    "zomma": (0.5, (2, 1, 0, 0, 0)),
    "theta": (1.0, (0, 0, 0, 0, 1)), "charm": (1.0, (1, 0, 0, 0, 1)), "color": (0.5, (2, 0, 0, 0, 1)),
    "vega_theta": (1.0, (0, 1, 0, 0, 1)),
    "rho": (1.0, (0, 0, 1, 0, 0)), "dividend_rho": (1.0, (0, 0, 0, 1, 0)),
    "dividend_volga": (0.5, (0, 0, 0, 2, 0)), "delta_q": (1.0, (1, 0, 0, 1, 0)),
}
VEGA_SCALED = ("vega", "vega_theta")
PER_PCT = ("rho", "dividend_rho")
TIME_GREEKS = ("theta", "r_theta", "q_theta", "convexity_theta", "gamma_theta", "charm", "color", "vega_theta")
SUBROW_NAMES = ("r_theta", "q_theta", "convexity_theta", "gamma_theta", "theta_contract", "ledger_carry")


def cash_greek(name: str, raw_position_greek: float, spot_t0: float, per_day_divisor: float) -> float:
    """Desk cash convention (spec §7.5): raw x S^a / 100^max(a-1,0) x 0.01^(b+c+d); time greeks per day."""
    if name in SUBROW_NAMES:
        return raw_position_greek / per_day_divisor
    _, (a, b, c, d, e) = TERM_SPEC[name]
    scale = spot_t0 ** a / (100.0 ** max(a - 1, 0)) * (0.01 ** (b + c + d))
    if e:
        scale /= per_day_divisor
    return raw_position_greek * scale


def display_moves(exponents, moves) -> Dict[str, float]:
    out: Dict[str, float] = {}
    a, b, c, d, e = exponents
    if a:
        out["spot_return"] = moves.spot_return
    if b and moves.d_vol is not None:
        out["vol_pts"] = moves.d_vol * 100.0
    if c and moves.d_rate is not None:
        out["rate_pct"] = moves.d_rate * 100.0
    if d and moves.d_div is not None:
        out["div_pct"] = moves.d_div * 100.0
    if e:
        out.update(moves.display(Factor.TIME))
    return out


def _resolve_steps(cache: ScenarioCache, config: PnLExplainConfig, bump) -> Tuple[int, str]:
    """(n, clock_label) for the Taylor time terms; the label is 'gap', '1d' or '1td'.

    Under ``per_step`` an unset ``config.clock`` is resolved from the bump
    config's theta mode, and the RESOLVED clock is what the calculator is asked
    for (``theta_1d`` / ``theta_1td``): the unsuffixed theta is the configured
    ``time_bump_days`` step, which is not a one-day value.
    """
    mv = cache.moves
    if config.time_term == "exact_gap":
        return 1, "gap"
    clock = config.clock
    if clock is None:
        mode = resolve_theta_bump_mode(cache.snap0.pricing_env, bump.time_bump_mode)
        clock = "1td" if mode == "business_days" else "1d"
    if clock == "1td":
        if mv.trading_days is None:
            raise ValidationError("time_term='per_step' with the '1td' clock requires pricing_env.calendar")
        n = mv.trading_days
    else:
        n = mv.calendar_days
    if n <= 0:
        raise ValidationError(
            f"time_term='per_step' with the '{clock}' clock: no steps elapsed between the snapshots "
            "(use clock='1d' or time_term='exact_gap')"
        )
    return n, clock


def _clock_suffix(clock_label: str) -> str:
    """Registry suffix of the resolved clock: '' for the exact gap, '_1d' / '_1td' per step."""
    return "" if clock_label == "gap" else f"_{clock_label}"


def _calculator(params, bump, config: PnLExplainConfig, days: int) -> GreeksCalculator:
    if config.time_term == "exact_gap":
        bump = dataclasses.replace(bump, time_bump_days=days, time_bump_mode="calendar_days")
        params = dataclasses.replace(params, bump_config=bump)
    return GreeksCalculator(params=params, greeks_mode=config.greeks_mode)


def _row(level, name, factor, pnl, greek=None, cash=None, kind=RowKind.COMPONENT, extra=None, mv=None):
    return ExplainRow(factor=factor, term=name, method=ExplainMethod.TAYLOR, kind=kind, level=level,
                      pnl=pnl, moves=mv if mv is not None else {}, greek=greek, cash_greek=cash,
                      metadata=extra or {})


def _info(level, term, pnl, per_day, moves, formula, extra=None):
    return ExplainRow(factor=Factor.TIME, term=term, method=ExplainMethod.TAYLOR, kind=RowKind.INFORMATIONAL,
                      level=level, pnl=pnl, greek=pnl / per_day, cash_greek=pnl / per_day,
                      moves=moves.display(Factor.TIME), metadata={"formula": formula, **(extra or {})})


def _theta_subrows(level, greeks, q, n, days, per_day, gap_scale, config, subrows, time_pure,
                   theta_pnl, terminal, moves) -> List[ExplainRow]:
    """Informational time sub-rows: pnl over the whole step, greek/cash per day (exact_gap)
    or per step (per_step), i.e. pnl divided by the number of clock steps."""
    exact_gap = config.time_term == "exact_gap"
    divisor = per_day if exact_gap else float(n)
    if terminal:
        out = [_info(level, "ledger_carry", time_pure, divisor, moves, "ledger_carry = time_pure - theta_contract")]
        if exact_gap:
            out.insert(0, _info(level, "theta_contract", 0.0, divisor, moves, "no contingent leg"))
        return out
    out: List[ExplainRow] = []
    if exact_gap:
        theta_contract = q * greeks["theta"] * gap_scale
        out.append(_info(level, "theta_contract", theta_contract, divisor, moves,
                         "calculator theta over the calendar gap x quantity (analytical: per day x days)"))
        out.append(_info(level, "ledger_carry", time_pure - theta_contract, divisor, moves,
                         "ledger_carry = time_pure - theta_contract"))
        base = theta_contract
    else:
        base = theta_pnl
    if not subrows:
        return out
    # r_theta / q_theta are numerical-only: estimate mode returns per-day values (x days under
    # exact_gap), exact mode reprices over the configured step (x1 under exact_gap); per_step x n.
    estimate = config.theta_decomposition_mode == "estimate"
    step_scale = (float(days) if estimate else 1.0) if exact_gap else float(n)
    r_theta = q * greeks["r_theta"] * step_scale
    q_theta = q * greeks["q_theta"] * step_scale
    if "r_theta" in subrows:
        out.append(_info(level, "r_theta", r_theta, divisor, moves, "calculator r_theta x steps"))
    if "q_theta" in subrows:
        out.append(_info(level, "q_theta", q_theta, divisor, moves, "calculator q_theta x steps"))
    convexity = base - r_theta - q_theta
    if "convexity_theta" in subrows:
        out.append(_info(level, "convexity_theta", convexity, divisor, moves, "theta_contract - r_theta - q_theta"))
    if "gamma_theta" in subrows:
        gamma_theta = q * greeks["gamma_theta"] * (float(days) if exact_gap else float(n))
        out.append(_info(level, "gamma_theta", gamma_theta, divisor, moves,
                         "-1/2 sigma^2 S^2 Gamma per day x days", {"theta_residual": convexity - gamma_theta}))
    return out


def taylor_rows(cache: ScenarioCache, config: PnLExplainConfig, level: str = "instrument"
                ) -> Tuple[Tuple[ExplainRow, ...], float, Dict[str, Any]]:
    snap0, moves = cache.snap0, cache.moves
    q, days, S0 = snap0.quantity, moves.calendar_days, moves.spot_t0
    terms, subrows = resolve_stencil(config), resolved_subrows(config)
    alive_move = cache.all_market().total - cache.value_for(()).total
    time_pure = cache.time_pure()
    params = config.params if config.params is not None else getattr(snap0.engine, "params", None)
    if params is None:
        params = EngineParams()        # the GreeksCalculator's own default for a params-less engine
    bump = params.get_effective_bump_config()
    n, clock_label = _resolve_steps(cache, config, bump)
    per_day = float(days) if config.time_term == "exact_gap" else 1.0
    terminal = is_terminal(snap0.lifecycle_state)

    greeks: Dict[str, float] = {}
    route: Optional[str] = None
    vega_scale: Optional[float] = None
    gap_scale = 1.0
    bucket_rows: Mapping[Factor, Tuple[ExplainRow, ...]] = {}
    covered: FrozenSet[Factor] = frozenset()
    # contract_roll_days == 0: the holder repriced the SAME float-maturity contract without
    # rolling it, so there is no contract theta to measure; every time greek is zero and the
    # time row carries only the valuation-date effect (spec §8).
    rolled = getattr(cache.transition, "contract_roll_days", None) != 0
    if not terminal:
        calc = _calculator(params, bump, config, days)
        wanted = list(terms)
        if "theta" in terms and subrows:                      # sub-rows are numerical-only in the calculator
            wanted += ["r_theta", "q_theta"] + (["gamma_theta"] if "gamma_theta" in subrows else [])
        suffix = _clock_suffix(clock_label)
        requested = [name for name in wanted if rolled or name not in TIME_GREEKS]
        request = [f"{name}{suffix}" if name in TIME_GREEKS else name for name in requested]
        route = config.greeks_method if config.greeks_method != "auto" \
            else calc.resolve_route(snap0.product, request)
        raw = calc.calculate(snap0.product, snap0.pricing_env, cache.bump_engine_t0, method=route,
                             greeks=request, theta_decomposition_mode=config.theta_decomposition_mode)
        greeks = {name: float(raw[f"{name}{suffix}" if name in TIME_GREEKS else name]) for name in requested}
        for name in wanted:
            if name not in greeks:
                greeks[name] = 0.0                           # unrolled contract: no time greek
        vega_scale = 0.01 if route == "analytical" else float(bump.vol_bump)
        # analytical time greeks are per-day rates; numerical ones are gap-valued under exact_gap
        gap_scale = float(days) if (route == "analytical" and config.time_term == "exact_gap") else 1.0
        if config.bucketed:
            from quantark.pnlexplain.equity.bucketed import bucketed_rows
            bucket_rows, covered = bucketed_rows(cache, calc, bump, level)
    meta = {"route": route, "vega_scale": vega_scale, "n_steps": n, "clock": clock_label, "gap_scale": gap_scale,
            "contract_rolled": rolled, "bucketed_factors": tuple(sorted(f.value for f in covered))}

    def derivative(name: str) -> float:
        g = greeks[name]
        if name in VEGA_SCALED:
            return g / vega_scale
        if name in PER_PCT:
            return g / 0.01
        return g

    raw_moves = (moves.d_spot, moves.d_vol, moves.d_rate, moves.d_div, float(n))
    rows: List[ExplainRow] = []
    explained = 0.0
    for name in terms:
        factor = TERM_FACTOR[name]
        coeff, exps = TERM_SPEC[name]
        applicable = factor in moves.coordinate.applicable and all(
            raw_moves[i] is not None for i, e in enumerate(exps[:4]) if e)
        if name == "vega" and Factor.VOL in covered:
            # tenor-vega buckets take the scalar row's slot exactly once (spec §7.6)
            for row in bucket_rows[Factor.VOL]:
                rows.append(row)
                if row.kind is RowKind.COMPONENT:
                    explained += row.pnl
            continue
        if name == "theta":
            if config.time_term == "exact_gap":
                pnl, greek = time_pure, time_pure / per_day
                extra = {"basis": "revaluation", "formula": "time_pure = V(alive@t1, t0 market) - V(t0)"}
            else:
                greek = q * greeks["theta"] if not terminal else 0.0
                pnl = greek * n
                extra = {"basis": "greek", "formula": "theta_per_step x n"}
            rows.append(_row(level, name, factor, pnl, greek=greek, cash=greek, extra=extra,
                             mv=moves.display(Factor.TIME)))
            explained += pnl
            rows.extend(_theta_subrows(level, greeks, q, n, days, per_day, gap_scale, config, subrows,
                                       time_pure, pnl, terminal, moves))
            continue
        if terminal or not applicable:
            rows.append(_row(level, name, factor, 0.0, extra={"applicable": applicable}))
            continue
        g_pos = q * derivative(name) * (gap_scale if exps[4] else 1.0)
        move_product = 1.0
        for i, e in enumerate(exps):
            if e:
                move_product *= raw_moves[i] ** e
        pnl = coeff * g_pos * move_product
        greek_disp = g_pos / per_day if exps[4] else g_pos
        rows.append(_row(level, name, factor, pnl, greek=greek_disp,
                         cash=cash_greek(name, g_pos, S0, per_day), mv=display_moves(exps, moves)))
        explained += pnl
        if name == "rho" and Factor.RATE in bucket_rows:
            # carry-invariant key-rate view: informational rows beneath the scalar rho (spec §7.6)
            rows.extend(bucket_rows[Factor.RATE])
    unexplained = alive_move - explained
    rows.append(_row(level, "unexplained", Factor.UNEXPLAINED, unexplained,
                     extra={"basis_and_model_effects_included": True}))
    return tuple(rows), unexplained, meta
