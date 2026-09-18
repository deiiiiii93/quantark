"""Human-readable certification report.

The certificate is the machine record; this is what a reviewer actually reads.
It carries no timestamps, so the report is a pure function of the certified
evidence -- two identical certifications produce identical reports, and a diff
between two reports shows only what actually changed.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from quantark.modelvalidation.engine_config import flatten

_NA = "--"


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return _NA
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        if value in (float("inf"), float("-inf")):
            return "inf" if value > 0 else "-inf"
        return f"{value:.{digits}g}"
    return str(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _first_line(text: str) -> str:
    """Last line of a traceback -- the exception, not the call stack."""
    lines = [line.strip() for line in (text or "").strip().splitlines() if line.strip()]
    return lines[-1] if lines else _NA


def _flat(value) -> str:
    """A nested error-model entry on one table line."""
    if isinstance(value, dict):
        return "; ".join(f"{k}: {_flat(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(_flat(v) for v in value)
    return str(value)


def _reference_spread(reference):
    """A cell reference's uncertainty: the radius of a deterministic value, else the standard error."""
    return reference.get("radius") if reference.get("kind") == "deterministic" else reference["se"]


def _deterministic_reference_section(payload) -> list:
    """The deterministic reference: its declared error model, each case's radii, and the qualifying arm's checks."""
    contract = payload["contract"]
    parts = ["## Deterministic reference", ""]
    parts.append("The reference is a deterministic solve. Its uncertainty is a declared radius, not a standard error: "
                 "`analytical` where the solve names an exactness basis, otherwise a `calibrated_estimate` from a refinement "
                 "ladder -- a numerical estimate with calibration evidence, not a proved bound. A radius consumes the budget "
                 "whole (no interval multiplier) and radii add linearly across cells.")
    parts.append("")
    parts.append(_table(["error model", "value"], [[str(k), _flat(v)] for k, v in sorted(contract["reference_error_model"].items())]))
    parts.append("")
    rows = []
    for case, block in sorted(payload["references"].items()):
        if "error" in block:
            rows.append([case, _first_line(block["error"]), _NA])
            continue
        rows.append([case, ", ".join(f"{q}: {_fmt(r, 3)} ({block['radius_basis'][q]})" for q, r in sorted(block["radii"].items())),
                     ", ".join(f"{q} ({why})" for q, why in sorted(block["undefined"].items())) or "none"])
    parts.append(_table(["case", "radii (raw)", "undefined here"], rows))
    parts.append("")
    qualification = payload.get("qualification")
    if qualification is None:
        parts.append("No qualifying arm was declared.")
        parts.append("")
        return parts
    policy = contract["qualification"]
    parts.append("## Reference qualification")
    parts.append("")
    parts.append(f"An independent stochastic arm simulates every case. The deterministic value must sit within "
                 f"{_fmt(policy['max_z'])} of its standard errors plus the reference's radius; a case that does not is "
                 "not qualified and its cells are unresolved.")
    parts.append("")
    rows = []
    for case, block in sorted(qualification.items()):
        if "error" in block:
            rows.append([case, _NA, _NA, _NA, _NA, _first_line(block["error"])])
            continue
        for quantity, check in block["checks"].items():
            rows.append([case, quantity, _fmt(check["difference"], 3), _fmt(check["qualifier_se"], 3),
                         _NA if check["z"] is None else _fmt(check["z"], 3), "yes" if check["within"] else "NO"])
    parts.append(_table(["case", "quantity", "reference - qualifier", "qualifier SE", "z", "within"], rows))
    parts.append("")
    return parts


def render_markdown(payload: Mapping[str, Any]) -> str:
    """Render a certificate as a markdown report."""
    study = payload["study"]
    runtime = payload["runtime"]
    bounds = study["bounds"]
    sampling = study["sampling"]

    parts: list[str] = []

    parts.append(f"# Certification report: {study['name']}")
    parts.append("")
    if study.get("quick"):
        parts.append(
            "> **Quick mode.** Sampling was shrunk for a wiring check; this run is "
            "not bankable evidence."
        )
        parts.append("")

    parts.append(f"Evidence digest: `{payload['projected_sha256']}`")
    parts.append("")
    parts.append(
        f"Machine: `{runtime['machine']}` / {runtime['platform']} - Python "
        f"{runtime['python']}, NumPy {runtime['numpy']}, quantark "
        f"`{runtime.get('quantark_git_sha') or 'unknown'}`"
    )
    parts.append("")

    amendment = payload.get("amendment")
    if amendment:
        # Without this the report reads as though every cell was freshly
        # measured. Carried cells were not re-priced; they are the parent's
        # numbers, admitted here because both identities still matched.
        parts.append("## Amendment")
        parts.append("")
        parts.append(
            _table(
                ["field", "value"],
                [
                    ["parent", str(amendment["parent"])],
                    ["parent digest", f"`{amendment['parent_projected_sha256']}`"],
                    ["reason", str(amendment["reason"])],
                    ["re-priced", f"{len(amendment['replaced_cells'])} cell(s)"],
                    ["carried forward", f"{len(amendment['carried_cells'])} cell(s)"],
                ],
            )
        )
        parts.append("")

    schema2 = payload.get("schema") == 2

    parts.append("## Decisions")
    parts.append("")
    parts.append(
        _table(
            ["candidate", "decision"],
            [[name, decision] for name, decision in sorted(payload["decisions"].items())],
        )
    )
    parts.append("")
    uncertified = study.get("uncertified_quantities") or []
    if schema2 and uncertified:
        parts.append(
            "Uncertified quantities (no reference estimator; outside every decision): "
            + ", ".join(f"`{q}`" for q in uncertified)
            + "."
        )
        parts.append("")
    if payload.get("contract", {}).get("reference_kind") == "deterministic":
        # the standard-error fraction and interval k belong to a stochastic reference; printing them here would
        # describe a policy this certificate did not use
        parts.append(
            f"Bounds: cell {_fmt(bounds['cell'])} c, mean signed bias "
            f"{_fmt(bounds['mean_signed_bias'])} c, reference radius at most "
            f"{_fmt(bounds['radius_budget_fraction'])} x cell (deterministic reference: a radius is consumed whole, "
            "with no interval multiplier)."
        )
    else:
        parts.append(
            f"Bounds: cell {_fmt(bounds['cell'])} c, mean signed bias "
            f"{_fmt(bounds['mean_signed_bias'])} c, standard-error budget "
            f"{_fmt(bounds['se_budget_fraction'])} x cell, interval k "
            f"{_fmt(bounds['interval_k'])}."
        )
    parts.append("")
    if schema2:
        parts.append(
            "Schema 2: gate values are a fraction of each cell's own budget (quantity_bounds below); "
            "the cell bound of 1 is that budget."
        )
        parts.append("")
        parts.append(
            _table(
                ["quantity", "abs floor", "relative"],
                [
                    [quantity, _fmt(spec["abs_floor"]), _fmt(spec["rel"])]
                    for quantity, spec in (study.get("quantity_bounds") or {}).items()
                ],
            )
        )
        parts.append("")

    config_rows = []
    for candidate in study["candidates"]:
        for key, value in sorted(flatten(candidate.get("params") or {}).items()):
            config_rows.append([candidate["name"], key, _fmt(value)])
    for key, value in sorted(flatten(payload.get("reference_config") or {}).items()):
        config_rows.append(["(benchmark)", key, _fmt(value)])
    if config_rows:
        parts.append("## Engine configuration")
        parts.append("")
        parts.append(
            "Resolved rather than named: a profile such as `standard` is an indirection "
            "whose meaning can change between releases. These are the requested settings."
        )
        parts.append("")
        parts.append(_table(["engine", "setting", "value"], config_rows))
        parts.append("")

    deterministic = payload.get("contract", {}).get("reference_kind") == "deterministic"
    if deterministic:
        parts.extend(_deterministic_reference_section(payload))
    parts.append("## Qualifying arm sampling" if deterministic else "## Benchmark sampling")
    parts.append("")
    reference_rows = []
    for case, block in sorted((payload.get("qualification", {}) if deterministic else payload["references"]).items()):
        if "error" in block:
            reference_rows.append([case, _NA, _first_line(block["error"]), _NA])
            continue
        reference_rows.append(
            [
                case,
                str(block["batches"]),
                block["stopped_reason"],
                ", ".join(
                    f"{q}: {_fmt(se, 3)}" for q, se in sorted(block["std_errors"].items())
                ),
            ]
        )
    parts.append(
        _table(["case", "batches", "stopped because", "standard errors (raw)"], reference_rows)
    )
    parts.append("")
    parts.append(
        f"Sampling policy: {sampling['paths_per_batch']} paths/batch, "
        f"{sampling['min_batches']}-{sampling['max_batches']} batches, seed "
        f"{sampling['seed']}, bump {_fmt(sampling['bump'])}."
    )
    parts.append("")

    parts.append("## Cells")
    parts.append("")
    cell_rows = []
    for cell in payload["cells"]:
        gate = cell["gate"]
        reference = cell["reference"]
        cell_rows.append(
            [
                cell["candidate"],
                cell["case"],
                cell["quantity"],
                _fmt(reference["value"]) if reference else _NA,
                _fmt(_reference_spread(reference), 3) if reference else _NA,
                _fmt(cell["candidate_value"]),
                _fmt(gate["signed_err_c"], 4) if gate else _NA,
                _fmt(gate["interval_c"], 4) if gate else _NA,
                _fmt(gate["envelope_c"], 4) if gate else _NA,
                cell["verdict"],
            ]
        )
    parts.append(
        _table(
            [
                "candidate",
                "case",
                "quantity",
                "reference",
                "radius" if deterministic else "SE",
                "candidate",
                "err (c)",
                "interval (c)",
                "envelope (c)",
                "verdict",
            ],
            cell_rows,
        )
    )
    parts.append("")

    if schema2:
        reason_rows = [
            [cell["candidate"], cell["case"], cell["quantity"], cell.get("kind") or _NA, cell["verdict"],
             str(cell["reason"])]
            for cell in payload["cells"]
            if cell.get("reason")
        ]
        if reason_rows:
            parts.append("## Semantic, uncertified and unresolved cells")
            parts.append("")
            parts.append(_table(["candidate", "case", "quantity", "kind", "verdict", "reason"], reason_rows))
            parts.append("")
        convergence_rows = []
        for cell in payload["cells"]:
            block = cell.get("convergence")
            if not block:
                continue
            orders = block.get("observed_orders") or {}
            convergence_rows.append(
                [
                    cell["candidate"],
                    cell["case"],
                    cell["quantity"],
                    _fmt(block.get("envelope_raw"), 3),
                    ", ".join(f"{axis}: {_fmt(order, 3)}" for axis, order in sorted(orders.items())) or _NA,
                    ", ".join(block.get("non_monotone") or []) or _NA,
                ]
            )
        if convergence_rows:
            parts.append("## Convergence")
            parts.append("")
            parts.append(
                _table(
                    ["candidate", "case", "quantity", "envelope", "observed order per axis", "non-monotone axes"],
                    convergence_rows,
                )
            )
            parts.append("")

    parts.append("## Aggregate bias")
    parts.append("")
    aggregate_rows = [
        [
            aggregate["candidate"],
            aggregate["quantity"],
            str(aggregate["cells"]),
            _fmt(aggregate["mean_signed_bias_c"], 4),
            _fmt(aggregate.get("radius_of_mean_c") if deterministic else aggregate["se_of_mean_c"], 3),
            "yes" if aggregate["passed"] else "no",
        ]
        for aggregate in payload["aggregates"]
    ]
    if aggregate_rows:
        parts.append(
            _table(
                ["candidate", "quantity", "cells", "mean bias (c)",
                 "mean radius (c)" if deterministic else "SE (c)", "passed"],
                aggregate_rows,
            )
        )
    else:
        parts.append("No aggregate gates ran (every cell errored).")
    parts.append("")

    errors = [cell for cell in payload["cells"] if cell.get("error")]
    if errors:
        parts.append("## Errors")
        parts.append("")
        parts.append(
            _table(
                ["candidate", "case", "quantity", "exception"],
                [
                    [
                        cell["candidate"],
                        cell["case"],
                        cell["quantity"],
                        _first_line(cell["error"]),
                    ]
                    for cell in errors
                ],
            )
        )
        parts.append("")
        parts.append(
            "Full tracebacks are in `certificate.json`. An errored cell makes "
            "ADMITTED unreachable for that candidate."
        )
        parts.append("")

    return "\n".join(parts)
