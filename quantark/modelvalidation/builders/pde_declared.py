"""The PDEParams a study declares for a PDE candidate (spec 2 §4.6, decision 15).

A candidate executes exactly the params it records: the YAML ``params`` mapping
goes through ``PDEParams.from_config`` once, when the study loads, so a key the
engine cannot honour fails the study there instead of being banked as a setting
that moved no number.
"""

from __future__ import annotations

from typing import Any, Mapping

from quantark.asset.equity.param import PDEParams
from quantark.util.exceptions import ValidationError


def declared_pde_params(params: Any, *, exclude: tuple = ()) -> PDEParams:
    """``PDEParams.from_config`` over the declared mapping, minus the study-only
    keys in ``exclude`` (a candidate label, say)."""
    if not isinstance(params, Mapping):
        raise ValidationError(
            f"a PDE candidate is declared by a params mapping, got {type(params).__name__}; "
            "custom components cannot be certified (from_config cannot name them)"
        )
    return PDEParams.from_config({k: v for k, v in params.items() if k not in exclude})
