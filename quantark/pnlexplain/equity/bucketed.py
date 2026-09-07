"""Opt-in tenor-vega / key-rate rows (spec §7.6, amended by patch spec 2026-09-03 §6).

Tenor-vega rows are COMPONENT rows that replace the scalar ``vega`` row: the
node-aligned bumps of the bucketed-greeks machinery sum to the scalar vega, so
``Σ_τ dV/dσ_τ × Δσ_τ`` is the scalar row split by pillar.

Key-rate rows are COMPONENT rows that replace the scalar ``rho`` row in the
same way. They are requested under the DIVIDEND_HELD convention (the rate
curve alone moves, the dividend yield is held), which IS the factor model's
rate step split by pillar; the desk-default carry-invariant convention (the
forward held, the dividend yield re-derived) measures pure discounting and is
a different sensitivity, so it is never booked here. The
``rate_keyrate.parallel`` row stays INFORMATIONAL as the reconciliation view.
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Mapping, Tuple

from quantark.asset.equity.riskmeasures.bucketed_greeks import (
    BucketedGreekCoordinate,
    BucketedGreeksRequest,
    RateKeyrateConvention,
)
from quantark.param import TermStructureVolSurface
from quantark.param.rrf.rate_curve import InterpolatedRateCurve
from quantark.pnlexplain.base import ExplainMethod, ExplainRow, Factor, RowKind
from quantark.pnlexplain.equity.scenario import ScenarioCache
from quantark.util.exceptions import ValidationError

RATE_CONVENTION = RateKeyrateConvention.DIVIDEND_HELD    # the factor model's rate step: q held, F moves


def _check_pair(name: str, a: Any, b: Any, cls) -> bool:
    """True when both snapshots carry a `cls` term structure.

    Rule (spec §7.6): the canonical pillars are the t0 object's; the t1 object
    is sampled at those pillars with its own interpolation, so a differing t1
    grid is allowed. A term structure on one side only is rejected.
    """
    is_a, is_b = isinstance(a, cls), isinstance(b, cls)
    if is_a != is_b:
        raise ValidationError(f"bucketed mode: {name} is a term structure on one side only")
    return is_a


def bucketed_rows(cache: ScenarioCache, calc: Any, bump: Any, level: str
                  ) -> Tuple[Mapping[Factor, Tuple[ExplainRow, ...]], FrozenSet[Factor]]:
    """Rows partitioned by factor + the factors whose scalar row they REPLACE (⊆ {VOL, RATE}).

    ``rows[Factor.VOL]`` are the ``vega.<τ>`` COMPONENT rows (t0 pillars).
    ``rows[Factor.RATE]`` are the dividend-held ``rate_keyrate.<τ>`` COMPONENT
    rows plus the informational ``rate_keyrate.parallel`` row; the scalar
    ``rho`` is replaced.
    """
    snap0, snap1, moves = cache.snap0, cache.snap1, cache.moves
    e0, e1 = snap0.pricing_env, snap1.pricing_env
    q, K = snap0.quantity, moves.coordinate.reference_strike
    T = moves.coordinate.tenor_t1
    coords = []
    if _check_pair("vol_surface", e0.vol_surface, e1.vol_surface, TermStructureVolSurface) \
            and Factor.VOL in moves.coordinate.applicable:
        coords.append(BucketedGreekCoordinate.VOL_TENOR_VEGA)
    if _check_pair("rate_curve", e0.rate_curve, e1.rate_curve, InterpolatedRateCurve) \
            and Factor.RATE in moves.coordinate.applicable:
        coords.append(BucketedGreekCoordinate.RATE_KEYRATE)
    if not coords:
        return {}, frozenset()
    request = BucketedGreeksRequest(coordinates=tuple(coords), vol_bump=float(bump.vol_bump),
                                    rate_bump=float(bump.rate_bump), rate_keyrate_convention=RATE_CONVENTION)
    result = calc.calculate_bucketed_greeks(snap0.product, e0, cache.bump_engine_t0, request)
    by_factor: Dict[Factor, List[ExplainRow]] = {Factor.VOL: [], Factor.RATE: []}
    for pt in result.points:
        mode = getattr(pt, "difference_mode", None)
        mode = mode.value if hasattr(mode, "value") else mode
        if pt.coordinate is BucketedGreekCoordinate.VOL_TENOR_VEGA:
            tau = float(pt.maturity)
            d = float(e1.get_vol(K, tau)) - float(e0.get_vol(K, tau))
            g = q * float(pt.derivative)
            by_factor[Factor.VOL].append(ExplainRow(
                factor=Factor.VOL, term=f"vega.{tau:g}", method=ExplainMethod.TAYLOR, kind=RowKind.COMPONENT,
                level=level, pnl=g * d, greek=g, cash_greek=g * 0.01,
                moves={"vol_pts": d * 100.0, "tenor": tau},
                metadata={"pillar": tau, "bump_size": float(pt.bump_size), "difference_mode": mode}))
        elif pt.coordinate is BucketedGreekCoordinate.RATE_KEYRATE:
            if pt.name == "rate_keyrate.parallel":
                # the parallel move IS the factor model's scalar rate move, read at the coordinate
                if moves.d_rate is None or T is None:
                    raise ValidationError(
                        "bucketed mode: the parallel key-rate row needs the scalar rate move at the "
                        "product's tenor, which this coordinate does not define"
                    )
                d = float(moves.d_rate)
                g = q * float(pt.derivative)
                by_factor[Factor.RATE].append(ExplainRow(
                    factor=Factor.RATE, term="rate_keyrate.parallel", method=ExplainMethod.TAYLOR,
                    kind=RowKind.INFORMATIONAL, level=level, pnl=g * d, greek=g, cash_greek=g * 0.01,
                    moves={"rate_pct": d * 100.0},
                    metadata={**dict(pt.metadata), "bump_size": float(pt.bump_size), "difference_mode": mode,
                              "convention": RATE_CONVENTION.value}))
                continue
            tau = float(pt.maturity)
            d = float(e1.get_rate(tau)) - float(e0.get_rate(tau))
            g = q * float(pt.derivative)
            by_factor[Factor.RATE].append(ExplainRow(
                factor=Factor.RATE, term=f"rate_keyrate.{tau:g}", method=ExplainMethod.TAYLOR,
                kind=RowKind.COMPONENT, level=level, pnl=g * d, greek=g, cash_greek=g * 0.01,
                moves={"rate_pct": d * 100.0, "tenor": tau},
                metadata={"pillar": tau, "bump_size": float(pt.bump_size), "difference_mode": mode,
                          "convention": RATE_CONVENTION.value}))
    rows = {f: tuple(r) for f, r in by_factor.items() if r}
    # A factor is COVERED only when it has a component row to replace the scalar term with.
    # rate_keyrate always emits the informational parallel row, so a curve whose pillars are
    # all uncalibrated would otherwise drop the scalar rho and leave the rate move unexplained
    # (Kimi review 2026-09-03).
    covered = frozenset(
        f for f in (Factor.VOL, Factor.RATE)
        if any(r.kind is RowKind.COMPONENT for r in rows.get(f, ()))
    )
    return rows, covered
