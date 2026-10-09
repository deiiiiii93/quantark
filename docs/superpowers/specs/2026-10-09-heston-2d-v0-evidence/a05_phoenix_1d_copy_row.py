"""A05: the 1D Phoenix path (PhoenixAdapter never sets ``solver._grid_v1``; core/statistics/phoenix.py documents the
continuous-KI copy row as 0.0). Same defect? Same fix? Before / after a runtime patch, back to back.

Patch: ``PhoenixAdapter.boundaries`` sets each V0 surface's rows through
``_set_boundary_conditions_v0_beside_v1_rows`` (the helper the 2D hooks now use: V1's rows of the step are computed
and exposed as ``_grid_v1`` for the copy row). Prices of continuous-KI Phoenixes (memory and not, standard and
reverse) on the default grid and on a narrow grid (short maturity, where the edge sits nearest the KI level).
"""
import numpy as np

from common import S0, barrier, env
from quantark.asset.equity.engine.pde.core.adapters.phoenix import PhoenixAdapter
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig, CouponPayType
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import ObservationType

_ORIG = PhoenixAdapter.boundaries


def _patched(self, prepared, group, states, j):
    name = group.surfaces[0]
    if not name.startswith("v0["):
        return _ORIG(self, prepared, group, states, j)
    solver, product, env_ = self.solver, prepared.product, prepared.env
    x_vec, s_vec = prepared.layout.spatial.x, prepared.layout.spatial.s
    tau_remaining = prepared.tau - prepared.layout.time.t[j]
    solver._set_boundary_conditions_v0_beside_v1_rows(states[name], x_vec, s_vec, j, tau_remaining, product, env_)


def phoenix(reverse, memory, T, ki=None):
    ko, ki0 = (97.0, 125.0) if reverse else (103.0, 75.0)
    ki = ki0 if ki is None else ki
    n = max(int(round(12 * T)), 1)
    obs = [(i + 1) / n * T for i in range(n)]
    bc = BarrierConfig(
        ko_barrier=ko, ko_rate=0.0,
        ko_observation_schedule=ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=ko)
                                                             for t in obs]),
        ki_barrier=ki, ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
    return PhoenixOption(
        initial_price=S0, strike=S0, maturity=T, contract_multiplier=1.0, is_reverse=reverse, barrier_config=bc,
        coupon_config=CouponBarrierConfig(coupon_barrier=ki, coupon_rate=0.01, memory_coupon=memory,
                                          coupon_pay_type=CouponPayType.INSTANT,
                                          day_count_convention=DayCountConvention.ACT_365),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True))


if __name__ == "__main__":
    for T in (2.0, 0.25):
        for reverse in (False, True):
            for memory in (False, True):
                for ki in (None,) + ((60.0, 50.0) if not reverse else (150.0, 200.0)):
                    p = phoenix(reverse, memory, T, ki)
                    PhoenixAdapter.boundaries = _ORIG
                    s0 = PhoenixPDESolver()
                    b = s0.price(p, env())
                    PhoenixAdapter.boundaries = _patched
                    s1 = PhoenixPDESolver()
                    a = s1.price(p, env())
                    PhoenixAdapter.boundaries = _ORIG
                    rows = s1._rows_v0_stash
                    edge = rows[5] if reverse else rows[4]
                    tag = f"T={T} {'rev' if reverse else 'std'} {'memory' if memory else 'plain '} KI={p.barrier_config.ki_barrier}"
                    print(f"{tag:<34} before {b:.12f} after {a:.12f} diff {a - b:+.3e}  "
                          f"{'bit-identical' if float(a).hex() == float(b).hex() else 'MOVED'}  t0 ki-edge {edge.branch}",
                          flush=True)
