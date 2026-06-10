"""Plot helpers for risk reports."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from quantark.util.exceptions import ValidationError


def _require_matplotlib():
    # In sandboxed environments, ~/.matplotlib may be non-writable.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ValidationError(
            "matplotlib is required to generate plots for risk reports."
        ) from exc
    return plt


def save_line_plot(
    *,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    x_markers: Optional[dict[str, float]] = None,
) -> None:
    plt = _require_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if x_markers:
        for name, x0 in x_markers.items():
            ax.axvline(x=x0, linestyle="--", linewidth=1, alpha=0.6)
            ax.text(x0, ax.get_ylim()[1], f" {name}", rotation=90, va="top", alpha=0.7)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_heatmap(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    path: Path,
) -> None:
    plt = _require_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)

    if z.shape != (x.size, y.size):
        raise ValidationError(f"z must be shape (len(x), len(y)), got {z.shape}")

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        extent=[float(y[0]), float(y[-1]), float(x[0]), float(x[-1])],
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
