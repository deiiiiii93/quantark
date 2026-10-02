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


def environment_day_count(environment_params: Mapping[str, Any]) -> int:
    """The business days per year of the environment a study builds (spec 3A §5.8): the declared
    ``bus_days_in_year``, else ``PricingEnvironment``'s default. The study's environments and its
    candidates' grid record read this one function, so the two cannot disagree."""
    import dataclasses

    from quantark.priceenv import PricingEnvironment

    if "bus_days_in_year" in environment_params:
        return int(environment_params["bus_days_in_year"])
    return next(f.default for f in dataclasses.fields(PricingEnvironment) if f.name == "bus_days_in_year")


def refuse_case_day_count(base: Mapping[str, Any], case: Mapping[str, Any]) -> None:
    """A candidate records one day count, so a case may not declare another."""
    if "bus_days_in_year" in case and environment_day_count(case) != environment_day_count(base):
        raise ValidationError("a case may not override bus_days_in_year: the candidate's grid record states one "
                              "day count for every case")
