"""Pathwise delta for the one-step-survival estimator (RESEARCH.md §5.4).

The OSS estimator makes the per-path value a smooth function of S0, which is why
its FD gamma stops blowing up. But it is smooth enough for more than that: it is
differentiable in closed form, so delta needs no bump at all. That removes the
O(h^2) FD bias from the reference delta outright, and it turns gamma into a
FIRST difference of a smooth quantity instead of a second difference of a rough
one.

Why bother, given OSS alone already drops the certification's delta requirement
to the batch floor: the floor estimates rest on noise-inflated pilot terms and on
gains measured on a single fixture, and the aggregate gate is delta-only. Bias
that no allocation can buy down is exactly the kind of term worth eliminating
rather than bounding.

## The recursion

For Heston, mu_i and s_i are functions of the variance path alone -- z_var and
u_var -- and carry no S0 dependence, so differentiating the asset recursion is
self-contained. (Under SLV they would depend on S through the leverage surface,
which is the extra work that port needs, on top of the C1 interpolation issue.)

Writing A = d log_a / dS0 and B = d log_b / dS0, per step:

  Run A at a KO node, with zb = (ln_ko - log_a - mu)/s:
      dp_surv = phi(zb) * (-A/s)
      z_a     = Phi^-1(p_surv * u)
      dz_a    = u * dp_surv / phi(z_a)
      A      <- A + s * dz_a          (and A unchanged at non-KO nodes)
      w_a    <- w_a * p_surv,   dw_a <- dw_a * p_surv + w_a * dp_surv

  Run B at a monitored node, lo = Phi(z_ki), hi = Phi(z_ko), p = hi - lo:
      dlo = phi(z_ki)*(-B/s),  dhi = phi(z_ko)*(-B/s)
      z_b = Phi^-1(lo + p*u)
      dz_b = ((1-u)*dlo + u*dhi) / phi(z_b)
      B   <- B + s * dz_b
      w_b <- w_b * p,   dw_b <- dw_b * p + w_b * (dhi - dlo)

The phi-ratios are the numerical trap. p_surv*u <= p_surv means z_a <= zb
always, so in the lower tail phi(zb)/phi(z_a) = exp((z_a^2 - zb^2)/2) is a large
number computed as a ratio of two tiny ones. Evaluated as a quotient it
overflows or returns nan; evaluated as one exponential of a difference of squares
it is exact and merely large. Same for Run B.

Clips and maxima in the value assembly must be differentiated as what they are:
where `max(., 0)` is clipped, or a probability hits its clip, the derivative is
zero, not the unclipped value.

## Gates

1. PV parity: this loop must reproduce `OSSSnowballPricer.oss_price` to ~1e-12.
   Without that, any delta it reports belongs to a different estimator.
2. Correctness: pathwise delta must match a tiny-bump FD of the SAME estimator on
   the SAME draws to O(h^2). This is the real test of the algebra above.
3. Variance: pathwise delta stderr vs OSS-FD and plain-engine-FD delta stderr.
4. Gamma-from-pathwise-delta: FD of pathwise delta vs OSS-FD gamma.

Usage:
    PYTHONPATH=$PWD python docs/adi2d-greek-perf/probes/probe_pathwise_delta.py
    ... --seeds 10 --paths 50000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.special import log_ndtr, ndtr, ndtri

ROOT = Path(__file__).resolve().parents[3]
DEMOS = Path("/Users/fuxinyao/quant-ark/docs/mc2d-gamma-convergence/demos")
OUTPUT_DIR = ROOT / "output" / "pathwise_delta"

if not DEMOS.is_dir():
    raise SystemExit(f"mc2d-gamma-convergence demos not found at {DEMOS}")
sys.path.insert(0, str(DEMOS))

import demo_b_oss as oss_demo  # noqa: E402
from common import (  # noqa: E402
    HESTON,
    KI_BARRIER,
    KO_BARRIER,
    SPOT0,
    batch_seeds,
    engine_price,
)

_UEPS = oss_demo._UEPS
_EXP_CAP = 600.0
_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


def _phi(z: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(-0.5 * z * z, -_EXP_CAP, _EXP_CAP)) / np.sqrt(2.0 * np.pi)


def _log_phi(z: np.ndarray) -> np.ndarray:
    # s_i floors at 1e-300 whenever the variance path hits zero, which sends the
    # z-scores to ~1e300; squaring that overflows before any clip can help, so
    # bound z first. -0.5*(1e150)^2 is already far below every exp() floor here.
    zc = np.clip(z, -1e150, 1e150)
    return -0.5 * zc * zc - _LOG_SQRT_2PI


def _phi_over_s(z: np.ndarray, log_s: np.ndarray) -> np.ndarray:
    """phi(z)/s, via logs, so a degenerate step yields 0 instead of 0*inf=nan.

    The 1/s here is real, not an artefact: as s->0 the survival probability
    becomes a step function of S0 and its derivative is a delta. But phi(z)
    vanishes faster than 1/s grows for any path not sitting exactly on the
    barrier, so the almost-everywhere limit is 0 -- which is what this returns.
    """
    return np.exp(np.clip(_log_phi(z) - log_s, -_EXP_CAP, _EXP_CAP))


def _phi_ratio(z_num: np.ndarray, z_den: np.ndarray) -> np.ndarray:
    """phi(z_num)/phi(z_den) as one exponential of a difference of logs."""
    return np.exp(np.clip(_log_phi(z_num) - _log_phi(z_den), -_EXP_CAP, _EXP_CAP))


def _log_hazard_lower(z: np.ndarray) -> np.ndarray:
    """log of phi(z)/Phi(z) -- the hazard rate for LOWER-truncated draws."""
    return _log_phi(z) - log_ndtr(z)


def _log_hazard_upper(z: np.ndarray) -> np.ndarray:
    """log of phi(z)/(1-Phi(z)), via Phi(-z) so the upper tail stays exact."""
    return _log_phi(z) - log_ndtr(-z)


def _hazard_ratio(log_num: np.ndarray, log_den: np.ndarray) -> np.ndarray:
    """exp(log_num - log_den), clipped. Both are ~|z| in the tail, so bounded.

    This is the identity that makes the pathwise derivative computable. For a
    one-sided truncated draw z = Phi^-1(Phi(zb)*u), the chain rule wants
    u*phi(zb)/phi(z), which as u->0 is 0*exp(+huge) and overflows. But u is
    exactly Phi(z)/Phi(zb), so the product collapses to hazard(zb)/hazard(z) --
    a ratio of two quantities that each behave like |z|.
    """
    return np.exp(np.clip(log_num - log_den, -_EXP_CAP, _EXP_CAP))


class PathwiseOSSPricer(oss_demo.OSSSnowballPricer):
    """OSS value and its exact S0-derivative from one pass over the same draws."""

    def oss_price_and_delta(self, spot: float) -> tuple[float, float, np.ndarray, np.ndarray]:
        engine, env, T, all_times, dt_array, ko_idx, ki_idx, term, kop = self._setup(
            spot
        )
        p = HESTON
        M = len(dt_array)
        principal, v0_const, pm_eff, K = self._closed
        rebate_amt = v0_const - principal

        ko_node = np.zeros(M + 1, dtype=bool)
        ko_node[np.asarray(ko_idx) + 1] = True
        ko_leg_of_node = np.full(M + 1, -1, dtype=int)
        ko_leg_of_node[np.asarray(ko_idx) + 1] = np.arange(len(ko_idx))
        ki_node = np.zeros(M + 1, dtype=bool)
        ki_node[np.asarray(ki_idx) + 1] = True

        ko_cash = np.asarray(kop["payoffs"], dtype=float)
        ko_settle = np.asarray(kop["settlement_times"], dtype=float)
        df = engine._df
        df_ko = df(ko_settle)
        df_T = float(df(np.array([T]))[0])

        z_var, z_ind, u_var = self._draws(M)
        u_ind = ndtr(z_ind)

        ln_ko = np.log(KO_BARRIER)
        ln_ki = np.log(KI_BARRIER)
        ln_k = np.log(K)

        n = self.num_paths
        s0 = max(float(spot), 1e-12)
        log_a = np.full(n, np.log(s0))
        log_b = log_a.copy()
        w_a = np.ones(n)
        w_b = np.ones(n)
        val = np.zeros(n)
        var = np.full(n, max(float(p.v0), 0.0))

        # Derivatives w.r.t. S0. log S0 differentiates to 1/S0; weights start flat.
        d_log_a = np.full(n, 1.0 / s0)
        d_log_b = np.full(n, 1.0 / s0)
        d_w_a = np.zeros(n)
        d_w_b = np.zeros(n)
        d_val = np.zeros(n)

        for i, dt in enumerate(dt_array):
            node = i + 1
            last = node == M
            drift = float(term.rrf[i] - term.div[i])
            v_np, v_bar, mu_extra, s_cond = self._qe_step(
                var, z_var[:, i], u_var[:, i], p, dt, self.martingale
            )
            mu_i = (drift - 0.5 * v_bar) * dt + mu_extra
            s_i = np.maximum(s_cond, 1e-300)
            u = u_ind[:, i]

            log_s_i = np.log(s_i)

            if last:
                # ---- Run A terminal: KO leg + E[1{S<B_ko} V1(S)] -------------
                fa = np.exp(log_a + mu_i + 0.5 * s_i * s_i)
                d_fa = fa * d_log_a
                zb_a = (ln_ko - log_a - mu_i) / s_i
                zk_a = (ln_k - log_a - mu_i) / s_i
                # d z/dS0 = -d_log/s, so every phi(z)*dz below is -phi(z)/s*d_log.
                p_ko_surv = ndtr(zb_a)
                d_p_ko = -_phi_over_s(zb_a, log_s_i) * d_log_a

                leg = ko_leg_of_node[node]
                cash = df_ko[leg] * ko_cash[leg]
                val += w_a * (1.0 - p_ko_surv) * cash
                d_val += (d_w_a * (1.0 - p_ko_surv) - w_a * d_p_ko) * cash

                e_v1 = principal * p_ko_surv - pm_eff * (
                    K * ndtr(zk_a) - fa * ndtr(zk_a - s_i)
                )
                d_e_v1 = principal * d_p_ko - pm_eff * (
                    -K * _phi_over_s(zk_a, log_s_i) * d_log_a
                    - d_fa * ndtr(zk_a - s_i)
                    + fa * _phi_over_s(zk_a - s_i, log_s_i) * d_log_a
                )
                val += w_a * df_T * e_v1
                d_val += df_T * (d_w_a * e_v1 + w_a * d_e_v1)

                # ---- Run B terminal: corridor legs ---------------------------
                fb = np.exp(log_b + mu_i + 0.5 * s_i * s_i)
                d_fb = fb * d_log_b
                zb_b = (ln_ko - log_b - mu_i) / s_i
                zk_b = (ln_k - log_b - mu_i) / s_i
                zki_b = (ln_ki - log_b - mu_i) / s_i

                raw_corridor = ndtr(zb_b) - ndtr(zki_b)
                corridor = np.maximum(raw_corridor, 0.0)
                # Where the max clamps, the derivative is zero, not the raw one.
                live = raw_corridor > 0.0
                d_corridor = np.where(
                    live,
                    -(_phi_over_s(zb_b, log_s_i) - _phi_over_s(zki_b, log_s_i))
                    * d_log_b,
                    0.0,
                )

                raw_cash = ndtr(zk_b) - ndtr(zki_b)
                cash_leg = np.maximum(raw_cash, 0.0)
                live_cash = raw_cash > 0.0
                d_cash_leg = np.where(
                    live_cash,
                    -(_phi_over_s(zk_b, log_s_i) - _phi_over_s(zki_b, log_s_i))
                    * d_log_b,
                    0.0,
                )

                raw_fwd = ndtr(zk_b - s_i) - ndtr(zki_b - s_i)
                fwd_leg = np.maximum(raw_fwd, 0.0)
                live_fwd = raw_fwd > 0.0
                d_fwd_leg = np.where(
                    live_fwd,
                    -(
                        _phi_over_s(zk_b - s_i, log_s_i)
                        - _phi_over_s(zki_b - s_i, log_s_i)
                    )
                    * d_log_b,
                    0.0,
                )

                e_put_corr = pm_eff * (K * cash_leg - fb * fwd_leg)
                d_e_put = pm_eff * (
                    K * d_cash_leg - d_fb * fwd_leg - fb * d_fwd_leg
                )
                payoff_b = rebate_amt * corridor + e_put_corr
                d_payoff_b = rebate_amt * d_corridor + d_e_put
                val += w_b * df_T * payoff_b
                d_val += df_T * (d_w_b * payoff_b + w_b * d_payoff_b)

                # Kept for the localisation gate: lets a caller check the path
                # recursion (d_log_a/d_log_b) separately from the value assembly.
                self.last_state = {
                    "log_a": log_a.copy(),
                    "log_b": log_b.copy(),
                    "d_log_a": d_log_a.copy(),
                    "d_log_b": d_log_b.copy(),
                    "w_a": w_a.copy(),
                    "w_b": w_b.copy(),
                    "d_w_a": d_w_a.copy(),
                    "d_w_b": d_w_b.copy(),
                }
                var = v_np
                break

            # ---- Run A: truncate below KO at KO nodes only -------------------
            if ko_node[node]:
                zb = (ln_ko - log_a - mu_i) / s_i
                raw_surv = ndtr(zb)
                p_surv = np.clip(raw_surv, _UEPS, 1.0)
                # Derivative vanishes where the clip is active.
                unclipped = (raw_surv > _UEPS) & (raw_surv < 1.0)
                d_p_surv = np.where(
                    unclipped, -_phi_over_s(zb, log_s_i) * d_log_a, 0.0
                )

                leg = ko_leg_of_node[node]
                cash = df_ko[leg] * ko_cash[leg]
                val += w_a * (1.0 - p_surv) * cash
                d_val += (d_w_a * (1.0 - p_surv) - w_a * d_p_surv) * cash

                d_w_a = d_w_a * p_surv + w_a * d_p_surv
                w_a = w_a * p_surv

                arg = p_surv * u
                arg_clipped = np.clip(arg, _UEPS, 1.0 - _UEPS)
                z_a = ndtri(arg_clipped)
                # Survival is "below the barrier", so this is a LOWER-truncated
                # draw: Phi(z_a) = Phi(zb)*u exactly, hence
                #   u * phi(zb)/phi(z_a) == hazard_lower(zb)/hazard_lower(z_a).
                # s_i cancels: s_i * dz_a = s_i * ratio * (-d_log_a/s_i).
                inside = (arg > _UEPS) & (arg < 1.0 - _UEPS)
                ratio = _hazard_ratio(
                    _log_hazard_lower(zb), _log_hazard_lower(z_a)
                )
                d_log_a = d_log_a - np.where(inside & unclipped, ratio, 0.0) * d_log_a
            else:
                z_a = z_ind[:, i]
                # d_log_a unchanged: the draw carries no S0 dependence here.
            log_a = log_a + mu_i + s_i * z_a

            # ---- Run B: corridor truncation at every monitored node ----------
            lo = np.zeros(n)
            hi = np.ones(n)
            d_lo = np.zeros(n)
            d_hi = np.zeros(n)
            has_ki = bool(ki_node[node])
            has_ko = bool(ko_node[node])
            z_ki = z_ko = None
            if has_ki:
                z_ki = (ln_ki - log_b - mu_i) / s_i
                lo = ndtr(z_ki)
                d_lo = -_phi_over_s(z_ki, log_s_i) * d_log_b
            if has_ko:
                z_ko = (ln_ko - log_b - mu_i) / s_i
                hi = ndtr(z_ko)
                d_hi = -_phi_over_s(z_ko, log_s_i) * d_log_b

            raw_corr = hi - lo
            p_corr = np.clip(raw_corr, 0.0, 1.0)
            live_corr = (raw_corr > 0.0) & (raw_corr < 1.0)
            d_p_corr = np.where(live_corr, d_hi - d_lo, 0.0)

            d_w_b = d_w_b * p_corr + w_b * d_p_corr
            w_b = w_b * p_corr

            alive = p_corr > _UEPS
            arg_b = lo + np.where(alive, p_corr, 1.0) * u
            arg_b_clipped = np.clip(arg_b, _UEPS, 1.0 - _UEPS)
            z_b = np.where(alive, ndtri(arg_b_clipped), 0.0)
            inside_b = (arg_b > _UEPS) & (arg_b < 1.0 - _UEPS)
            usable = alive & inside_b

            # Same cancellation as Run A, but the corridor has three regimes and
            # only the one-sided ones admit the clean hazard identity.
            # As in Run A, s_i cancels out of s_i * dz_b, leaving a pure factor.
            if has_ki and has_ko:
                # Two-sided: z_b lies between z_ki and z_ko, so phi(z_b) is
                # bounded below by the smaller endpoint density and the ratios
                # are safe. Dead paths (p_corr ~ 0) are excluded by `usable`.
                factor = (1.0 - u) * _phi_ratio(z_ki, z_b) + u * _phi_ratio(z_ko, z_b)
            elif has_ko:
                # Lower-truncated (survive below KO): Phi(z_b) = Phi(z_ko)*u, so
                # the u cancels into a ratio of lower hazards.
                factor = _hazard_ratio(
                    _log_hazard_lower(z_ko), _log_hazard_lower(z_b)
                )
            elif has_ki:
                # Upper-truncated (survive above KI):
                # 1-Phi(z_b) = (1-Phi(z_ki))*(1-u) -> ratio of upper hazards.
                factor = _hazard_ratio(
                    _log_hazard_upper(z_ki), _log_hazard_upper(z_b)
                )
            else:
                # Untruncated: z_b is the raw draw and carries no S0 dependence.
                factor = np.zeros(n)
            d_log_b = d_log_b - np.where(usable, factor, 0.0) * d_log_b
            log_b = log_b + mu_i + s_i * z_b

            var = v_np

        return float(val.mean()), float(d_val.mean()), val, d_val


def gate_pv_parity(paths: int, seed: int) -> dict:
    """The derivative pass must price identically to the published estimator."""
    pricer = PathwiseOSSPricer(paths, seed)
    pv_mine, delta, _, _ = pricer.oss_price_and_delta(SPOT0)
    pv_theirs = oss_demo.OSSSnowballPricer(paths, seed).oss_price(SPOT0)
    return {
        "pv_pathwise_loop": pv_mine,
        "pv_published": pv_theirs,
        "abs_diff": abs(pv_mine - pv_theirs),
        "delta": delta,
        "pass": bool(abs(pv_mine - pv_theirs) < 1e-10),
    }


def gate_fd_agreement(paths: int, seed: int, bumps: Sequence[float]) -> dict:
    """Pathwise delta vs FD of the same estimator on the same draws: O(h^2)."""
    pricer = PathwiseOSSPricer(paths, seed)
    _, pathwise, _, _ = pricer.oss_price_and_delta(SPOT0)
    rows = []
    for h_rel in bumps:
        h = h_rel * SPOT0
        up, _, _, _ = pricer.oss_price_and_delta(SPOT0 + h)
        down, _, _, _ = pricer.oss_price_and_delta(SPOT0 - h)
        fd = (up - down) / (2.0 * h)
        rows.append(
            {
                "bump": h_rel,
                "fd_delta": fd,
                "pathwise_delta": pathwise,
                "abs_diff": abs(fd - pathwise),
                "rel_diff": abs(fd - pathwise) / max(abs(pathwise), 1e-300),
            }
        )
    # An O(h^2) discrepancy must fall ~4x per halving of h; check the trend.
    trend = None
    if len(rows) >= 2:
        trend = [
            rows[i]["abs_diff"] / max(rows[i + 1]["abs_diff"], 1e-300)
            for i in range(len(rows) - 1)
        ]
    return {"rows": rows, "successive_ratios": trend}


def gate_variance(seeds: Sequence[int], paths: int, bump: float) -> dict:
    """Pathwise vs OSS-FD vs plain-engine-FD delta stderr at equal paths."""
    h = bump * SPOT0
    pathwise, oss_fd, engine_fd = [], [], []
    seconds = {"pathwise": 0.0, "oss_fd": 0.0, "engine_fd": 0.0}
    for seed in seeds:
        started = time.perf_counter()
        pricer = PathwiseOSSPricer(paths, seed)
        _, d_pathwise, _, _ = pricer.oss_price_and_delta(SPOT0)
        seconds["pathwise"] += time.perf_counter() - started
        pathwise.append(d_pathwise)

        started = time.perf_counter()
        p2 = oss_demo.OSSSnowballPricer(paths, seed)
        oss_fd.append((p2.oss_price(SPOT0 + h) - p2.oss_price(SPOT0 - h)) / (2.0 * h))
        seconds["oss_fd"] += time.perf_counter() - started

        started = time.perf_counter()
        up = engine_price(SPOT0 + h, seed, paths).price
        down = engine_price(SPOT0 - h, seed, paths).price
        seconds["engine_fd"] += time.perf_counter() - started
        engine_fd.append((up - down) / (2.0 * h))

    out = {"bump": bump, "paths": paths, "seeds": len(seeds), "seconds": seconds}
    for name, values in (
        ("pathwise", pathwise),
        ("oss_fd", oss_fd),
        ("engine_fd", engine_fd),
    ):
        arr = np.asarray(values, dtype=float)
        out[name] = {
            "mean": float(arr.mean()),
            "stderr": float(arr.std(ddof=1) / np.sqrt(arr.size)),
        }
    for base in ("oss_fd", "engine_fd"):
        out[f"variance_ratio_vs_{base}"] = float(
            (out[base]["stderr"] / out["pathwise"]["stderr"]) ** 2
        )
    return out


def gate_gamma_from_pathwise(seeds: Sequence[int], paths: int, bump: float) -> dict:
    """Gamma as a FIRST difference of pathwise delta, vs OSS second difference."""
    h = bump * SPOT0
    from_pathwise, from_oss = [], []
    for seed in seeds:
        pricer = PathwiseOSSPricer(paths, seed)
        _, d_up, _, _ = pricer.oss_price_and_delta(SPOT0 + h)
        _, d_down, _, _ = pricer.oss_price_and_delta(SPOT0 - h)
        from_pathwise.append((d_up - d_down) / (2.0 * h))

        p2 = oss_demo.OSSSnowballPricer(paths, seed)
        up, mid, down = (
            p2.oss_price(SPOT0 + h),
            p2.oss_price(SPOT0),
            p2.oss_price(SPOT0 - h),
        )
        from_oss.append((up - 2.0 * mid + down) / (h * h))
    out = {"bump": bump}
    for name, values in (("pathwise_fd", from_pathwise), ("oss_second_diff", from_oss)):
        arr = np.asarray(values, dtype=float)
        out[name] = {
            "mean": float(arr.mean()),
            "stderr": float(arr.std(ddof=1) / np.sqrt(arr.size)),
        }
    out["variance_ratio"] = float(
        (out["oss_second_diff"]["stderr"] / out["pathwise_fd"]["stderr"]) ** 2
    )
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--paths", type=int, default=50_000)
    parser.add_argument("--bump", type=float, default=0.003)
    parser.add_argument("--parity-paths", type=int, default=20_000)
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR / "pathwise_delta.json"
    )
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "purpose": "pathwise delta for the OSS estimator (RESEARCH.md 5.4)",
        "paths_per_seed": args.paths,
        "seeds": args.seeds,
    }

    print("gate 1: PV parity against the published estimator", flush=True)
    payload["gate_pv_parity"] = gate_pv_parity(args.parity_paths, 20260811)
    g = payload["gate_pv_parity"]
    print(
        f"  pv mine {g['pv_pathwise_loop']:.10f} vs theirs {g['pv_published']:.10f}"
        f"  diff {g['abs_diff']:.2e}  {'PASS' if g['pass'] else 'FAIL'}",
        flush=True,
    )
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    if not g["pass"]:
        print("  refusing to report derivative numbers for a different estimator")
        return 1

    print("gate 2: pathwise vs FD of the same estimator (must be O(h^2))", flush=True)
    payload["gate_fd_agreement"] = gate_fd_agreement(
        args.parity_paths, 20260811, (0.01, 0.005, 0.0025)
    )
    for row in payload["gate_fd_agreement"]["rows"]:
        print(
            f"  h={row['bump']:<7.4f} fd {row['fd_delta']:+.8f}  "
            f"pathwise {row['pathwise_delta']:+.8f}  diff {row['abs_diff']:.3e}",
            flush=True,
        )
    print(
        f"  successive ratios (expect ~4): "
        f"{payload['gate_fd_agreement']['successive_ratios']}",
        flush=True,
    )
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    seeds = batch_seeds(999_331, args.seeds)
    print("gate 3: delta variance, pathwise vs OSS-FD vs engine-FD", flush=True)
    payload["gate_variance"] = gate_variance(seeds, args.paths, args.bump)
    v = payload["gate_variance"]
    print(
        f"  pathwise {v['pathwise']['mean']:+.5f}±{v['pathwise']['stderr']:.5f}   "
        f"oss_fd {v['oss_fd']['mean']:+.5f}±{v['oss_fd']['stderr']:.5f}   "
        f"engine_fd {v['engine_fd']['mean']:+.5f}±{v['engine_fd']['stderr']:.5f}",
        flush=True,
    )
    print(
        f"  variance ratio vs OSS-FD {v['variance_ratio_vs_oss_fd']:.2f}x, "
        f"vs engine-FD {v['variance_ratio_vs_engine_fd']:.2f}x",
        flush=True,
    )
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print("gate 4: gamma as a first difference of pathwise delta", flush=True)
    payload["gate_gamma"] = gate_gamma_from_pathwise(seeds, args.paths, args.bump)
    gg = payload["gate_gamma"]
    print(
        f"  from pathwise {gg['pathwise_fd']['mean']:+.5f}±"
        f"{gg['pathwise_fd']['stderr']:.5f}   "
        f"from OSS 2nd diff {gg['oss_second_diff']['mean']:+.5f}±"
        f"{gg['oss_second_diff']['stderr']:.5f}   "
        f"ratio {gg['variance_ratio']:.2f}x",
        flush=True,
    )
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
