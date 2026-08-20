"""Shared helpers for the CFETS USD/CNY volatility-model example suite.

The public CFETS curve is a composite benchmark, not a licensed executable
quote history.  This module keeps that distinction visible in every artifact:
raw five-delta nodes drive Heston calibration; a separately labelled SABR and
calendar projection supplies the differentiable grid required by Dupire/SLV.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Sequence

import numpy as np


SCHEMA_VERSION = 1
PILLAR_ORDER = ("10P", "25P", "ATM", "25C", "10C")
PILLAR_DELTA = {"10P": -0.10, "25P": -0.25, "ATM": None, "25C": 0.25, "10C": 0.10}
TENOR_ORDER = (
    "1D", "1W", "2W", "3W", "1M", "2M", "3M", "6M", "9M", "1Y", "18M", "2Y", "3Y"
)
TENOR_SETS = {
    "core": ("1M", "2M", "3M", "6M", "9M", "1Y"),
    "liquid": ("1W", "2W", "3W", "1M", "2M", "3M", "6M", "9M", "1Y"),
    "full": ("1W", "2W", "3W", "1M", "2M", "3M", "6M", "9M", "1Y", "18M", "2Y", "3Y"),
}

HESTON_BOUNDS = (
    (1e-7, 0.01, 1e-7, 0.0005, -0.99),
    (0.02, 20.0, 0.02, 1.0, 0.99),
)


def normalise_tenor(tenor: str) -> str:
    """Return the suite's canonical tenor label."""
    value = str(tenor).strip().upper()
    return "18M" if value in {"1.5Y", "1Y6M"} else value


def strike_from_spot_delta(
    forward: float,
    iv: float,
    maturity: float,
    foreign_rate: float,
    delta: float,
) -> float:
    """Invert the CFETS non-premium-adjusted spot delta into a strike.

    CFETS publishes ``Delta_call = exp(-r_f T) N(d1)`` and
    ``Delta_put = exp(-r_f T) (N(d1)-1)`` with forward moneyness in ``d1``.
    ``delta`` is signed: positive for calls and negative for puts.
    """
    values = (forward, iv, maturity)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("forward, iv and maturity must be finite and positive")
    if not math.isfinite(foreign_rate):
        raise ValueError("foreign_rate must be finite")
    if not math.isfinite(delta) or delta == 0.0 or abs(delta) >= 1.0:
        raise ValueError("delta must be finite, non-zero and have magnitude below one")

    foreign_df = math.exp(-foreign_rate * maturity)
    if delta > 0.0:
        probability = delta / foreign_df
    else:
        probability = 1.0 + delta / foreign_df
    if not 0.0 < probability < 1.0:
        raise ValueError(
            f"delta {delta} is incompatible with foreign discount factor {foreign_df:.8f}"
        )
    d1 = NormalDist().inv_cdf(probability)
    vol_time = iv * math.sqrt(maturity)
    return float(forward * math.exp(-d1 * vol_time + 0.5 * vol_time * vol_time))


def spot_delta_from_strike(
    forward: float,
    strike: float,
    iv: float,
    maturity: float,
    foreign_rate: float,
    *,
    is_call: bool,
) -> float:
    """Evaluate the CFETS spot delta; useful for round-trip validation."""
    if min(forward, strike, iv, maturity) <= 0.0:
        raise ValueError("forward, strike, iv and maturity must be positive")
    vol_time = iv * math.sqrt(maturity)
    d1 = (math.log(forward / strike) + 0.5 * vol_time * vol_time) / vol_time
    n_d1 = NormalDist().cdf(d1)
    foreign_df = math.exp(-foreign_rate * maturity)
    return float(foreign_df * (n_d1 if is_call else n_d1 - 1.0))


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def load_snapshot(path: str | Path) -> dict:
    """Load and validate the stable, offline CFETS snapshot schema."""
    snapshot = load_json(path)
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"snapshot schema_version must be {SCHEMA_VERSION}, got {snapshot.get('schema_version')!r}"
        )
    if snapshot.get("currency_pair") != "USD.CNY":
        raise ValueError("snapshot currency_pair must be USD.CNY")
    if not snapshot.get("trade_date") or not snapshot.get("quote_time"):
        raise ValueError("snapshot requires trade_date and quote_time")
    if not math.isfinite(float(snapshot.get("spot", 0.0))) or float(snapshot["spot"]) <= 0.0:
        raise ValueError("snapshot spot must be finite and positive")
    slices = snapshot.get("slices")
    if not isinstance(slices, list) or not slices:
        raise ValueError("snapshot requires non-empty slices")
    seen = set()
    for row in slices:
        tenor = normalise_tenor(row.get("tenor", ""))
        if tenor not in TENOR_ORDER or tenor in seen:
            raise ValueError(f"invalid or duplicate tenor {tenor!r}")
        seen.add(tenor)
        if float(row.get("maturity", 0.0)) <= 0.0 or float(row.get("forward", 0.0)) <= 0.0:
            raise ValueError(f"tenor {tenor}: maturity and forward must be positive")
        quotes = row.get("quotes", [])
        if tuple(q.get("pillar") for q in quotes) != PILLAR_ORDER:
            raise ValueError(f"tenor {tenor}: quotes must follow {PILLAR_ORDER}")
        strikes = [float(q["strike"]) for q in quotes]
        if any(not math.isfinite(k) or k <= 0.0 for k in strikes) or any(
            a >= b for a, b in zip(strikes, strikes[1:])
        ):
            raise ValueError(f"tenor {tenor}: strikes must be finite, positive and increasing")
        atm_quote = quotes[PILLAR_ORDER.index("ATM")]
        if not math.isclose(
            float(atm_quote["strike"]), float(row["forward"]), rel_tol=0.0, abs_tol=2e-10
        ):
            raise ValueError(f"tenor {tenor}: ATM must be ATMF with strike equal to forward")
        for quote in quotes:
            bid, mid, ask = (float(quote[name]) for name in ("bid_iv", "mid_iv", "ask_iv"))
            if not (0.0 < bid <= ask and math.isfinite(mid) and mid > 0.0):
                raise ValueError(
                    f"tenor {tenor} {quote['pillar']}: require positive bid <= ask and positive mid"
                )
        if "effective_foreign_rate_for_delta" in row:
            delta_rate = float(row["effective_foreign_rate_for_delta"])
            for quote in quotes:
                delta = quote.get("delta")
                if delta is None:
                    continue
                recovered = spot_delta_from_strike(
                    float(row["forward"]),
                    float(quote["strike"]),
                    float(quote["mid_iv"]),
                    float(row["maturity"]),
                    delta_rate,
                    is_call=float(delta) > 0.0,
                )
                if not math.isclose(recovered, float(delta), rel_tol=0.0, abs_tol=2e-10):
                    raise ValueError(
                        f"tenor {tenor} {quote['pillar']}: reconstructed spot delta does not round-trip"
                    )
        if "pricing_foreign_rate" in row:
            reconstructed_forward = float(snapshot["spot"]) * math.exp(
                (float(row["domestic_rate"]) - float(row["pricing_foreign_rate"]))
                * float(row["maturity"])
            )
            if not math.isclose(
                reconstructed_forward, float(row["forward"]), rel_tol=0.0, abs_tol=2e-10
            ):
                raise ValueError(f"tenor {tenor}: pricing rates do not reproduce published forward")
    return snapshot


def selected_slices(snapshot: dict, tenor_set: str | Sequence[str] = "core") -> list[dict]:
    """Return slices in canonical tenor order for one calibration universe."""
    if isinstance(tenor_set, str):
        if tenor_set not in TENOR_SETS:
            raise ValueError(f"unknown tenor_set {tenor_set!r}; choose {sorted(TENOR_SETS)}")
        tenors = set(TENOR_SETS[tenor_set])
    else:
        tenors = {normalise_tenor(t) for t in tenor_set}
    rows = [row for row in snapshot["slices"] if normalise_tenor(row["tenor"]) in tenors]
    order = {tenor: index for index, tenor in enumerate(TENOR_ORDER)}
    rows.sort(key=lambda row: order[normalise_tenor(row["tenor"])])
    missing = tenors - {normalise_tenor(row["tenor"]) for row in rows}
    if missing:
        raise ValueError(f"snapshot missing requested tenors: {sorted(missing)}")
    return rows


def iter_nodes(snapshot: dict, tenor_set: str | Sequence[str] = "core") -> list[dict]:
    """Flatten selected five-delta slices into calibration nodes."""
    nodes: list[dict] = []
    for row in selected_slices(snapshot, tenor_set):
        for quote in row["quotes"]:
            nodes.append(
                {
                    "trade_date": snapshot["trade_date"],
                    "tenor": normalise_tenor(row["tenor"]),
                    "maturity": float(row["maturity"]),
                    "forward": float(row["forward"]),
                    "domestic_rate": float(row["domestic_rate"]),
                    "foreign_rate": float(row["foreign_rate"]),
                    "pillar": quote["pillar"],
                    "delta": quote.get("delta"),
                    "strike": float(quote["strike"]),
                    "strike_over_forward": float(quote["strike"]) / float(row["forward"]),
                    "mid_iv": float(quote["mid_iv"]),
                    "bid_iv": float(quote["bid_iv"]),
                    "ask_iv": float(quote["ask_iv"]),
                }
            )
    return nodes


def _heston_initials():
    from quantark.volmodels.heston import HestonParams

    return (
        HestonParams(v0=0.0004, kappa=1.0, theta=0.0010, sigma=0.05, rho=-0.2),
        HestonParams(v0=0.0004, kappa=3.0, theta=0.0010, sigma=0.10, rho=-0.5),
        HestonParams(v0=0.0010, kappa=0.5, theta=0.0005, sigma=0.20, rho=0.2),
        HestonParams(v0=0.0003, kappa=5.0, theta=0.0012, sigma=0.20, rho=-0.8),
        HestonParams(v0=0.0008, kappa=10.0, theta=0.0008, sigma=0.30, rho=0.5),
    )


def _node_weights(nodes: Sequence[dict], mode: str) -> np.ndarray:
    if mode == "equal":
        return np.ones(len(nodes), dtype=float)
    if mode != "spread":
        raise ValueError("weight_mode must be 'equal' or 'spread'")
    half = np.array([0.5 * (n["ask_iv"] - n["bid_iv"]) for n in nodes], dtype=float)
    # Zero-width public composites are not executable zero-spread markets.  A
    # 2.5 vol-bp floor prevents them from receiving infinite weight.
    scale = np.maximum(half, 0.00025)
    weights = 1.0 / np.square(scale)
    return weights / float(np.mean(weights))


def heston_model_ivs(nodes: Sequence[dict], params) -> np.ndarray:
    """Lewis-path model IVs grouped by maturity, matching calibration numerics."""
    from quantark.volmodels.black_scholes import implied_vol_call
    from quantark.volmodels.heston import heston_call_prices_vectorized

    maturities = np.array([node["maturity"] for node in nodes], dtype=float)
    strikes = np.array([node["strike_over_forward"] for node in nodes], dtype=float)
    model = np.empty(len(nodes), dtype=float)
    for maturity in np.unique(maturities):
        indices = np.flatnonzero(maturities == maturity)
        prices = heston_call_prices_vectorized(
            1.0, strikes[indices], float(maturity), params, 0.0, 0.0
        )
        model[indices] = [
            implied_vol_call(1.0, float(strike), float(maturity), float(price), 0.0, 0.0)
            for strike, price in zip(strikes[indices], prices)
        ]
    return model


def heston_fit_diagnostics(nodes: Sequence[dict], result) -> dict:
    """Compute fit, spread and Feller diagnostics for a calibration result."""
    model = heston_model_ivs(nodes, result.params)
    market = np.array([node["mid_iv"] for node in nodes], dtype=float)
    bid = np.array([node["bid_iv"] for node in nodes], dtype=float)
    ask = np.array([node["ask_iv"] for node in nodes], dtype=float)
    error = model - market
    inside = (model >= bid - 1e-12) & (model <= ask + 1e-12)
    nonzero = ask > bid + 1e-12
    half_spread = 0.5 * (ask - bid)
    params = result.params
    sigma2 = params.sigma * params.sigma
    feller_ratio = math.inf if sigma2 == 0.0 else 2.0 * params.kappa * params.theta / sigma2
    tenor_rmse = {}
    for tenor in TENOR_ORDER:
        mask = np.array([node["tenor"] == tenor for node in nodes], dtype=bool)
        if np.any(mask):
            tenor_rmse[tenor] = float(np.sqrt(np.mean(np.square(error[mask]))) * 100.0)
    rows = []
    for node, model_iv, err, in_band in zip(nodes, model, error, inside):
        rows.append(
            {
                "tenor": node["tenor"],
                "pillar": node["pillar"],
                "maturity": node["maturity"],
                "strike_over_forward": node["strike_over_forward"],
                "bid_iv": node["bid_iv"],
                "market_iv": node["mid_iv"],
                "ask_iv": node["ask_iv"],
                "model_iv": float(model_iv),
                "error_iv": float(err),
                "inside_public_band": bool(in_band),
            }
        )
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "optimizer": result.optimizer,
        "nfev": int(result.nfev),
        "params": asdict(params),
        "feller_ratio": float(feller_ratio),
        "feller_margin": float(result.feller_margin),
        "rmse_iv": float(np.sqrt(np.mean(np.square(error)))),
        "rmse_vol_points": float(np.sqrt(np.mean(np.square(error))) * 100.0),
        "mae_vol_points": float(np.mean(np.abs(error)) * 100.0),
        "max_abs_vol_points": float(np.max(np.abs(error)) * 100.0),
        "inside_public_band_pct": float(np.mean(inside) * 100.0),
        "inside_nonzero_public_band_pct": float(np.mean(inside[nonzero]) * 100.0) if np.any(nonzero) else None,
        "zero_spread_nodes": int(np.sum(~nonzero)),
        "median_nonzero_half_spread_vol_points": (
            float(np.median(half_spread[nonzero]) * 100.0) if np.any(nonzero) else None
        ),
        "rmse_by_tenor_vol_points": tenor_rmse,
        "rows": rows,
    }


def calibrate_heston_multistart(
    snapshot: dict,
    *,
    tenor_set: str = "core",
    hard_feller: bool = False,
    weight_mode: str = "equal",
    starts: int = 5,
    max_nfev: int = 500,
) -> list[dict]:
    """Fit normalized-forward Heston from several deterministic starts."""
    from quantark.volmodels.heston import MarketOption, calibrate_heston

    nodes = iter_nodes(snapshot, tenor_set)
    weights = _node_weights(nodes, weight_mode)
    options = [
        MarketOption(K=n["strike_over_forward"], T=n["maturity"], iv=n["mid_iv"], weight=float(w))
        for n, w in zip(nodes, weights)
    ]
    fits = []
    for initial in _heston_initials()[:starts]:
        if hard_feller:
            from quantark.volmodels.heston import HestonParams

            # SLSQP is materially more reliable when its initial point is
            # feasible.  Keep a small interior margin; the optimizer may still
            # choose the active Feller boundary.
            feasible_sigma = min(
                initial.sigma,
                0.95 * math.sqrt(2.0 * initial.kappa * initial.theta),
            )
            initial = HestonParams(
                v0=initial.v0,
                kappa=initial.kappa,
                theta=initial.theta,
                sigma=max(feasible_sigma, HESTON_BOUNDS[0][3]),
                rho=initial.rho,
            )
        try:
            result = calibrate_heston(
                s0=1.0,
                options=options,
                r=0.0,
                carry=0.0,
                initial=initial,
                bounds=HESTON_BOUNDS,
                target="iv",
                method="lewis",
                regularize_feller=0.0,
                enforce_feller=hard_feller,
                max_nfev=max_nfev,
                xtol=1e-9,
                ftol=1e-10,
                gtol=1e-9,
            )
            fit = heston_fit_diagnostics(nodes, result)
            fit["initial"] = asdict(initial)
            fits.append(fit)
        except Exception as exc:  # report individual start failure; fail if every start fails
            fits.append({"success": False, "initial": asdict(initial), "error": f"{type(exc).__name__}: {exc}"})
    return sorted(
        fits,
        key=lambda item: item.get("rmse_iv", math.inf) if item.get("success") else math.inf,
    )


def summarise_multistart(fits: Sequence[dict], tolerance_vol_points: float = 0.001) -> dict:
    valid = [fit for fit in fits if "rmse_vol_points" in fit and fit.get("success")]
    if not valid:
        return {"valid": 0, "near_best": 0, "parameter_ranges": {}}
    best = min(fit["rmse_vol_points"] for fit in valid)
    near = [fit for fit in valid if fit["rmse_vol_points"] <= best + tolerance_vol_points]
    names = ("v0", "kappa", "theta", "sigma", "rho")
    return {
        "valid": len(valid),
        "near_best": len(near),
        "near_best_tolerance_vol_points": tolerance_vol_points,
        "best_rmse_vol_points": best,
        "parameter_ranges": {
            name: [min(fit["params"][name] for fit in near), max(fit["params"][name] for fit in near)]
            for name in names
        },
    }


def build_fx_environment(surface: dict):
    """Reconstruct an FxPricingEnvironment and GridVolSurface from stage 02."""
    from quantark.param import GridVolSurface, SpotQuote
    from quantark.param.rrf.rate_curve import LinearRateCurve
    from quantark.priceenv import FxPricingEnvironment

    grid = GridVolSurface(surface["strikes"], surface["maturities"], np.asarray(surface["iv_grid"], dtype=float))
    domestic = LinearRateCurve([(row["maturity"], row["domestic_rate"]) for row in surface["slices"]])
    foreign = LinearRateCurve([(row["maturity"], row["foreign_rate"]) for row in surface["slices"]])
    env = FxPricingEnvironment(
        valuation_date=datetime.fromisoformat(surface["trade_date"]),
        spot_quote=SpotQuote(spot=float(surface["spot"])),
        domestic_curve=domestic,
        foreign_curve=foreign,
        vol_surface=grid,
    )
    for row in surface["slices"]:
        maturity = float(row["maturity"])
        published = float(row["forward"])
        reconstructed = float(env.get_forward(maturity))
        if not math.isclose(reconstructed, published, rel_tol=0.0, abs_tol=2e-10):
            raise ValueError(
                f"tenor {row['tenor']}: pricing curves imply forward {reconstructed:.10f}, "
                f"not published CFETS forward {published:.10f}"
            )
    return env, grid


def plot_smiles(rows: Iterable[tuple[str, Sequence[float], Sequence[float]]], path: str | Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, x_values, vols in rows:
        ax.plot(x_values, np.asarray(vols) * 100.0, marker="o", ms=3, label=label)
    ax.set_xlabel("strike / forward")
    ax.set_ylabel("implied volatility (%)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output
