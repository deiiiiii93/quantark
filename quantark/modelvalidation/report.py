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

    parts.append("## Decisions")
    parts.append("")
    parts.append(
        _table(
            ["candidate", "decision"],
            [[name, decision] for name, decision in sorted(payload["decisions"].items())],
        )
    )
    parts.append("")
    parts.append(
        f"Bounds: cell {_fmt(bounds['cell'])} c, mean signed bias "
        f"{_fmt(bounds['mean_signed_bias'])} c, standard-error budget "
        f"{_fmt(bounds['se_budget_fraction'])} x cell, interval k "
        f"{_fmt(bounds['interval_k'])}."
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

    parts.append("## Benchmark sampling")
    parts.append("")
    reference_rows = []
    for case, block in sorted(payload["references"].items()):
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
                _fmt(reference["se"], 3) if reference else _NA,
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
                "SE",
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

    parts.append("## Aggregate bias")
    parts.append("")
    aggregate_rows = [
        [
            aggregate["candidate"],
            aggregate["quantity"],
            str(aggregate["cells"]),
            _fmt(aggregate["mean_signed_bias_c"], 4),
            _fmt(aggregate["se_of_mean_c"], 3),
            "yes" if aggregate["passed"] else "no",
        ]
        for aggregate in payload["aggregates"]
    ]
    if aggregate_rows:
        parts.append(
            _table(
                ["candidate", "quantity", "cells", "mean bias (c)", "SE (c)", "passed"],
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
