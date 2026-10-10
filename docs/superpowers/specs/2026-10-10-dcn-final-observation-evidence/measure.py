"""W10 evidence: the DCN on (and just before) its final observation date. No library change; runtime patches only.

    env PYTHONPATH=<wt>:<wt>/test <venv>/bin/python docs/superpowers/specs/2026-10-10-dcn-final-observation-evidence/measure.py

Sections (all printed; raw numbers, no rounding beyond %.6f):
  A. HAND   - the proposed settled value from the written rule (independent arithmetic in this file).
     MCROW  - the MC path kernel ``compute_dcn_cashflows`` on a ONE-column path at the final date (the engine's guard
              ``dt_array.size == 0`` bypassed by calling the kernel directly): the MC's own final row.
     PDEROW - the PDE's terminal column (``DCNAdapter.terminal``, captured by a wrapper during a day-before solve) read
              at the spot, plus the readout's column switch (carried OR spot <= KI): the PDE's own final row.
     TODAY  - what the engines do today when valued on the final observation date (raise).
  B. Day before (previous SSE trading day): MC and PDE prices per state x spot, beside HAND.
  C. Shrinking last step: the day-before contract with the final gap h (1 day, 1 hour, 1 minute, 1 second, 1e-9 y);
     payment gaps after the final observation held fixed. MC and PDE -> HAND as h -> 0 (away from a barrier).
  D. A valuation date that is itself a (non-final) monthly observation: MC vs PDE (does today's observation pay?).
"""
from __future__ import annotations

import dataclasses
import math
import warnings
from datetime import datetime, timedelta

import numpy as np

from dcn_fixtures import DCN_A, DCN_B, SSE, make_dcn

from quantark.asset.equity.engine.mc import DCNMCEngine
from quantark.asset.equity.engine.mc import dcn_mc_engine as mc_mod
from quantark.asset.equity.engine.mc.dcn_payoff import compute_dcn_cashflows
from quantark.asset.equity.engine.pde import dcn_pde_solver as pde_mod
from quantark.asset.equity.engine.pde.core.adapters import dcn as adapter_mod
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
from quantark.asset.equity.product.option import dcn_grid as grid_mod
from quantark.asset.equity.product.option.dcn_option import DCNDirection
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.priceenv.term_sampling import make_df_fn

warnings.simplefilter("ignore")

R, Q, SIGMA = 0.0356, 0.1406, 0.184
MC_PATHS = 1 << 16
ORIG_CTX = grid_mod.build_dcn_grid_context


def env(spot, valuation_date):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=SIGMA),
                              rate_curve=FlatRateCurve(rate=R), div_yield=ContinuousDividendYield(Q),
                              valuation_date=valuation_date)


def prev_trading_day(d):
    d -= timedelta(days=1)
    while not SSE.is_business_day(d):
        d -= timedelta(days=1)
    return d


def attempt(fn):
    try:
        return f"{float(fn()):.6f}"
    except Exception as exc:  # noqa: BLE001 - evidence: record the refusal text
        return f"RAISES {type(exc).__name__}: {str(exc).splitlines()[0][:90]}"


# ---------------------------------------------------------------- the written rule (proposal section 2)
def hand_settled(c, spot, carried, direction=DCNDirection.BUYER):
    """Final-row cash flows from the proposal, by hand. Returns (total, legs)."""
    p = make_dcn(c, valuation_date=c["maturity_date"], knocked_in_at_valuation=carried, direction=direction)
    row = p.schedule.monthly[-1]
    v = c["maturity_date"]
    df = lambda d: math.exp(-R * (d - v).days / 365.0)  # flat curve, ACT/365F from the valuation date
    sign = 1.0 if direction is DCNDirection.BUYER else -1.0
    a = c["coupon_counted_days"] / c["coupon_days_denom"]
    n, s0, part = c["notional"], c["initial_price"], c["participation"]
    ki = carried or spot <= c["ki_barrier_ratio"] * s0                      # carried state first, then today's close
    ko = row.is_ko_obs and spot >= c["ko_barrier_ratio"] * s0               # KO is live in both states
    cpn = row.is_coupon_obs and not ko and not ki and spot >= c["coupon_barrier_ratio"] * s0
    legs = {
        "ko": part * c["ko_coupon_rate"] * a * n * df(row.ko_payment_date) if ko else 0.0,
        "coupon": part * c["coupon_rate"] * a * n * df(row.coupon_payment_date) if cpn else 0.0,
        "loss": (-(n / s0) * part * max(c["ki_put_strike_ratio"] * s0 - spot, 0.0) * df(c["settlement_date"])
                 if (ki and not ko) else 0.0),
    }
    legs = {k: sign * x for k, x in legs.items()}
    return sum(legs.values()), legs


def mc_row(c, spot, carried, direction=DCNDirection.BUYER):
    """The MC path kernel on the one-column grid at the final date (guard bypassed)."""
    p = make_dcn(c, valuation_date=c["maturity_date"], knocked_in_at_valuation=carried, direction=direction)
    ctx = ORIG_CTX(p)
    cf = compute_dcn_cashflows(np.array([[float(spot)]]), p, ctx, make_df_fn(env(spot, c["maturity_date"])))
    return p.direction_sign * float(cf.total_pv[0])


_CAPTURE = {}
_ORIG_TERMINAL = adapter_mod.DCNAdapter.terminal


def _capturing_terminal(self, prepared, states, captures):
    _ORIG_TERMINAL(self, prepared, states, captures)
    _CAPTURE["s"] = np.array(prepared.events.s, copy=True)
    _CAPTURE["alive"] = np.array(states["alive"][:, -1], copy=True)
    _CAPTURE["ki"] = np.array(states["ki"][:, -1], copy=True)


adapter_mod.DCNAdapter.terminal = _capturing_terminal


def pde_row(c, spot, carried, direction=DCNDirection.BUYER):
    """The PDE terminal column (values in final-observation money) from a day-before solve, at the spot.

    On a flat curve df(pay)/df(T) from the day-before origin equals the DF from the final date to pay, so the row is
    directly comparable with the settled value. Column choice = the readout's own rule (carried OR spot <= KI)."""
    vday = prev_trading_day(c["maturity_date"])
    p = make_dcn(c, valuation_date=vday, knocked_in_at_valuation=carried, direction=direction)
    DCNPDEEngine().price(p, env(spot, vday))
    col = _CAPTURE["ki"] if (carried or spot <= p.ki_barrier) else _CAPTURE["alive"]
    s = _CAPTURE["s"]
    i = int(np.argmin(np.abs(s - spot)))
    return p.direction_sign * float(np.interp(spot, s, col)), float(s[i]), p.direction_sign * float(col[i])


def mc_price(p, e):
    return DCNMCEngine(num_paths=MC_PATHS).price(p, e)


def pde_price(p, e):
    return DCNPDEEngine().price(p, e)


# ---------------------------------------------------------------- shrinking final gap (section C)
def shrunk_ctx_factory(h):
    def build(product):
        ctx = ORIG_CTX(product)
        if ctx.times.size != 2:
            raise RuntimeError("shrink patch expects the day-before contract (2 grid columns)")
        shift = float(ctx.times[-1]) - h
        return dataclasses.replace(
            ctx,
            times=np.array([0.0, h]),
            coupon_pay_times=ctx.coupon_pay_times - shift,
            ko_pay_times=ctx.ko_pay_times - shift,
            loss_pay_time=ctx.loss_pay_time - shift,
        )
    return build


def patch_ctx(builder):
    for mod in (grid_mod, mc_mod, pde_mod, adapter_mod):
        mod.build_dcn_grid_context = builder


def main():
    contracts = {"DCN_A": DCN_A, "DCN_B": DCN_B}
    spots_a = (7000.0, 6300.0, 6000.0, 5999.0, 5500.0, 4800.0, 4650.0, 4500.0, 4000.0)

    for name, c in contracts.items():
        s0 = c["initial_price"]
        print(f"=== {name}: final obs {c['maturity_date']:%Y-%m-%d}, day before {prev_trading_day(c['maturity_date']):%Y-%m-%d}, "
              f"loss settlement {c['settlement_date']:%Y-%m-%d}; KI {c['ki_barrier_ratio']*s0:g} coupon "
              f"{c['coupon_barrier_ratio']*s0:g} KO {c['ko_barrier_ratio']*s0:g} K_loss {c['ki_put_strike_ratio']*s0:g}")
        p_last = make_dcn(c, valuation_date=c["maturity_date"])
        row = p_last.schedule.monthly[-1]
        print(f"final row: months {row.month_indices} coupon_obs={row.is_coupon_obs} ko_obs={row.is_ko_obs} "
              f"coupon_pay {row.coupon_payment_date:%Y-%m-%d} ko_pay {row.ko_payment_date:%Y-%m-%d}; "
              f"grid times {ORIG_CTX(p_last).times.tolist()} get_maturity {p_last.get_maturity()}")

        print("\n-- A. final row: HAND | MCROW | PDEROW(interp; nearest node) | TODAY mc | TODAY pde")
        for carried in (False, True):
            for spot in spots_a:
                hand, legs = hand_settled(c, spot, carried)
                mrow = mc_row(c, spot, carried)
                prow, s_node, prow_node = pde_row(c, spot, carried)
                pm = make_dcn(c, valuation_date=c["maturity_date"], knocked_in_at_valuation=carried)
                e = env(spot, c["maturity_date"])
                print(f"carried={carried!s:5} spot={spot:7g} | HAND {hand:14.6f} {{{', '.join(f'{k}={v:.6f}' for k, v in legs.items())}}}"
                      f" | MCROW {mrow:14.6f} | PDEROW {prow:14.6f} (node {s_node:.3f}: {prow_node:.6f})"
                      f" | TODAY mc {attempt(lambda: mc_price(pm, e))} | pde {attempt(lambda: pde_price(pm, e))}")
        # direction
        for carried, spot in ((False, 5500.0), (True, 4000.0)):
            hand, _ = hand_settled(c, spot, carried, DCNDirection.SELLER)
            print(f"SELLER carried={carried} spot={spot:g} | HAND {hand:.6f} | MCROW {mc_row(c, spot, carried, DCNDirection.SELLER):.6f}"
                  f" | PDEROW {pde_row(c, spot, carried, DCNDirection.SELLER)[0]:.6f}")

        print("\n-- B. day before: MC | PDE | HAND(final) | MC-HAND | PDE-HAND | one-day spot sd")
        vday = prev_trading_day(c["maturity_date"])
        dt_last = (c["maturity_date"] - vday).days / 365.0
        for carried in (False, True):
            for spot in spots_a:
                p = make_dcn(c, valuation_date=vday, knocked_in_at_valuation=carried)
                e = env(spot, vday)
                mc, pde = mc_price(p, e), pde_price(p, e)
                hand, _ = hand_settled(c, spot, carried)
                print(f"carried={carried!s:5} spot={spot:7g} | MC {mc:14.6f} | PDE {pde:14.6f} | HAND {hand:14.6f}"
                      f" | MC-HAND {mc-hand:12.6f} | PDE-HAND {pde-hand:12.6f} | sd {spot*SIGMA*math.sqrt(dt_last):.2f}")

        print("\n-- C. shrinking final step h (day-before contract, payment gaps after T held): MC | PDE | HAND")
        for h_label, h in (("1d", 1 / 365.0), ("1h", 1 / 365.0 / 24), ("1min", 1 / 365.0 / 1440),
                           ("1s", 1 / 365.0 / 86400), ("1e-9y", 1e-9)):
            patch_ctx(shrunk_ctx_factory(h))
            try:
                for carried in (False, True):
                    for spot in (7000.0, 6300.0, 5500.0, 4650.0, 4000.0):
                        p = make_dcn(c, valuation_date=vday, knocked_in_at_valuation=carried)
                        e = env(spot, vday)
                        hand, _ = hand_settled(c, spot, carried)
                        print(f"h={h_label:6} carried={carried!s:5} spot={spot:7g} | MC {attempt(lambda: mc_price(p, e)):>16}"
                              f" | PDE {attempt(lambda: pde_price(p, e)):>16} | HAND {hand:14.6f}")
            finally:
                patch_ctx(ORIG_CTX)
        print()

    print("=== D. valuation ON a non-final monthly observation (DCN_A, 2024-12-03, month 23): MC | PDE | MC-PDE")
    vobs = datetime(2024, 12, 3)
    assert SSE.is_business_day(vobs)
    for carried in (False, True):
        for spot in (6300.0, 5500.0, 4650.0, 4000.0):
            p = make_dcn(DCN_A, valuation_date=vobs, knocked_in_at_valuation=carried)
            e = env(spot, vobs)
            mc, pde = mc_price(p, e), pde_price(p, e)
            print(f"carried={carried!s:5} spot={spot:7g} | MC {mc:14.6f} | PDE {pde:14.6f} | MC-PDE {mc-pde:12.6f}")
    vprev = prev_trading_day(vobs)
    for spot in (6300.0, 5500.0):
        p = make_dcn(DCN_A, valuation_date=vprev, knocked_in_at_valuation=False)
        e = env(spot, vprev)
        print(f"day before ({vprev:%Y-%m-%d}) carried=False spot={spot:g} | MC {mc_price(p, e):.6f} | PDE {pde_price(p, e):.6f}")


if __name__ == "__main__":
    main()
