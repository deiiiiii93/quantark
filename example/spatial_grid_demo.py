#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spatial grid visualization demo (uniform vs non-uniform).

This script visualizes the non-uniform Tavella–Randall spatial grids used by
PDE solvers, for:
  1) A single critical point (classic Tavella–Randall)
  2) Multiple critical points (piecewise multi-critical construction)

It produces plots comparing:
  - Grid points in spot space (S) vs index
  - Spacing in log-space (dx) and in spot-space (dS)

Usage:
  python example/spatial_grid_demo.py --show
  python example/spatial_grid_demo.py --save-dir example/output
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from quantark.asset.equity.engine.pde.spatial_grid import SpatialGrid


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_grids(
    *,
    s_min: float,
    s_max: float,
    grid_size: int,
    critical_point: float,
    critical_points: List[float],
    eps_crit: float,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    x_u, s_u, dx_u = SpatialGrid.build_uniform_log(s_min, s_max, grid_size)
    dx_u_vec = np.full(grid_size - 1, dx_u, dtype=float)

    x_1, s_1, dx_1 = SpatialGrid.build_tavella_randall(
        s_min=s_min,
        s_max=s_max,
        num_points=grid_size,
        critical_point=critical_point,
        eps_crit=eps_crit,
    )

    x_m, s_m, dx_m = SpatialGrid.build_tavella_randall_multi(
        s_min=s_min,
        s_max=s_max,
        num_points=grid_size,
        critical_points=critical_points,
        eps_crit=eps_crit,
    )

    return (x_u, s_u, dx_u_vec), (x_1, s_1, dx_1), (x_m, s_m, dx_m)


def _plot_grid_points(
    *,
    ax,
    s_vec: np.ndarray,
    label: str,
    critical_lines: Optional[List[float]] = None,
) -> None:
    idx = np.arange(len(s_vec))
    ax.plot(idx, s_vec, linewidth=1.2, label=label)
    ax.set_xlabel("Grid Index")
    ax.set_ylabel("Spot Grid (S)")
    ax.grid(True, alpha=0.3)

    if critical_lines:
        for c in critical_lines:
            ax.axhline(c, color="black", linestyle="--", linewidth=0.8, alpha=0.5)


def _plot_spacings(
    *,
    ax_dx,
    ax_ds,
    x_vec: np.ndarray,
    s_vec: np.ndarray,
    dx_vec: np.ndarray,
    label: str,
) -> None:
    idx = np.arange(len(dx_vec))
    ds_vec = np.diff(s_vec)

    ax_dx.plot(idx, dx_vec, linewidth=1.2, label=label)
    ax_dx.set_xlabel("Interval Index")
    ax_dx.set_ylabel("dx (log-space)")
    ax_dx.grid(True, alpha=0.3)

    ax_ds.plot(idx, ds_vec, linewidth=1.2, label=label)
    ax_ds.set_xlabel("Interval Index")
    ax_ds.set_ylabel("dS (spot-space)")
    ax_ds.grid(True, alpha=0.3)


def _figure_single_critical(
    *,
    uniform: Tuple[np.ndarray, np.ndarray, np.ndarray],
    single: Tuple[np.ndarray, np.ndarray, np.ndarray],
    s_min: float,
    s_max: float,
    critical_point: float,
    eps_crit: float,
) -> plt.Figure:
    (x_u, s_u, dx_u), (x_1, s_1, dx_1) = uniform, single

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(gs[0, :])
    _plot_grid_points(
        ax=ax0,
        s_vec=s_u,
        label="Uniform (log-space)",
        critical_lines=[critical_point],
    )
    _plot_grid_points(
        ax=ax0,
        s_vec=s_1,
        label=f"Tavella–Randall (single crit={critical_point:g}, eps={eps_crit:g})",
        critical_lines=[critical_point],
    )
    ax0.set_title(
        f"Spatial Grid Points (Single Critical Point)  S∈[{s_min:g}, {s_max:g}]"
    )
    ax0.legend()

    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    _plot_spacings(
        ax_dx=ax1, ax_ds=ax2, x_vec=x_u, s_vec=s_u, dx_vec=dx_u, label="Uniform"
    )
    _plot_spacings(
        ax_dx=ax1, ax_ds=ax2, x_vec=x_1, s_vec=s_1, dx_vec=dx_1, label="Tavella–Randall"
    )
    ax1.set_title("dx per interval")
    ax2.set_title("dS per interval")
    ax1.legend()
    ax2.legend()

    return fig


def _figure_multi_critical(
    *,
    uniform: Tuple[np.ndarray, np.ndarray, np.ndarray],
    multi: Tuple[np.ndarray, np.ndarray, np.ndarray],
    s_min: float,
    s_max: float,
    critical_points: List[float],
    eps_crit: float,
) -> plt.Figure:
    (x_u, s_u, dx_u), (x_m, s_m, dx_m) = uniform, multi

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(gs[0, :])
    _plot_grid_points(
        ax=ax0,
        s_vec=s_u,
        label="Uniform (log-space)",
        critical_lines=critical_points,
    )
    _plot_grid_points(
        ax=ax0,
        s_vec=s_m,
        label=f"Multi-critical (crits={critical_points}, eps={eps_crit:g})",
        critical_lines=critical_points,
    )
    ax0.set_title(
        f"Spatial Grid Points (Multiple Critical Points)  S∈[{s_min:g}, {s_max:g}]"
    )
    ax0.legend()

    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    _plot_spacings(
        ax_dx=ax1, ax_ds=ax2, x_vec=x_u, s_vec=s_u, dx_vec=dx_u, label="Uniform"
    )
    _plot_spacings(
        ax_dx=ax1, ax_ds=ax2, x_vec=x_m, s_vec=s_m, dx_vec=dx_m, label="Multi-critical"
    )
    ax1.set_title("dx per interval")
    ax2.set_title("dS per interval")
    ax1.legend()
    ax2.legend()

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize SpatialGrid non-uniform grids (single vs multi critical points)."
    )
    parser.add_argument(
        "--grid-size", type=int, default=300, help="Number of grid points"
    )
    parser.add_argument("--s-min", type=float, default=40.0, help="Minimum spot bound")
    parser.add_argument("--s-max", type=float, default=160.0, help="Maximum spot bound")
    parser.add_argument(
        "--single-critical",
        type=float,
        default=100.0,
        help="Critical point for single-point Tavella–Randall grid",
    )
    parser.add_argument(
        "--multi-critical",
        type=float,
        nargs="+",
        default=[80.0, 100.0, 120.0],
        help="Critical points for multi-critical grid",
    )
    parser.add_argument(
        "--eps-crit",
        type=float,
        default=0.003,
        help="Target relative spacing near critical points (in spot space)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(Path("example") / "output"),
        help="Directory to save output figures (PNG)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plots instead of only saving",
    )
    args = parser.parse_args()

    mpl_dir = Path(tempfile.gettempdir()) / "quantark-mplconfig"
    _ensure_dir(mpl_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_dir))

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s_min = float(args.s_min)
    s_max = float(args.s_max)
    grid_size = int(args.grid_size)
    single_crit = float(args.single_critical)
    multi_crits = [float(x) for x in args.multi_critical]
    eps_crit = float(args.eps_crit)

    uniform, single, multi = _build_grids(
        s_min=s_min,
        s_max=s_max,
        grid_size=grid_size,
        critical_point=single_crit,
        critical_points=multi_crits,
        eps_crit=eps_crit,
    )

    fig_single = _figure_single_critical(
        uniform=uniform,
        single=single,
        s_min=s_min,
        s_max=s_max,
        critical_point=single_crit,
        eps_crit=eps_crit,
    )
    fig_multi = _figure_multi_critical(
        uniform=uniform,
        multi=multi,
        s_min=s_min,
        s_max=s_max,
        critical_points=multi_crits,
        eps_crit=eps_crit,
    )

    save_dir = Path(args.save_dir)
    _ensure_dir(save_dir)
    out_single = save_dir / "spatial_grid_single_critical.png"
    out_multi = save_dir / "spatial_grid_multi_critical.png"

    fig_single.savefig(out_single, dpi=160)
    fig_multi.savefig(out_multi, dpi=160)
    print(f"Saved: {out_single}")
    print(f"Saved: {out_multi}")

    if args.show:
        plt.show()
    else:
        plt.close(fig_single)
        plt.close(fig_multi)


if __name__ == "__main__":
    main()
