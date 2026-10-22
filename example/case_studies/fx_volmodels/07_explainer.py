"""Stage 07 — render the CFETS USD/CNY calibration study as one HTML file.

The explainer is deliberately a *derived* artifact.  Every displayed market or
calibration number is read from the five upstream JSON artifacts; the renderer
does not silently substitute sample values.  The public CFETS curve is always
labelled as a composite benchmark, never as executable bid/ask history.

Run:
    .venv/bin/python example/fx_volmodels/07_explainer.py --tag latest
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
ARTIFACT_STEMS = {
    "surface": "cfets_usdcny_surface",
    "localvol": "cfets_usdcny_localvol",
    "heston": "cfets_usdcny_heston",
    "slv": "cfets_usdcny_slv",
    "diagnostics": "cfets_usdcny_diagnostics",
}
PARAMETERS = ("v0", "kappa", "theta", "sigma", "rho")
SAFE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
MARKET_EVIDENCE = (
    {
        "fact": "H1 2025 domestic RMB FX turnover",
        "value": "USD 21.0tn total; USD 13.6tn derivatives (65%)",
        "scope": "SAFE aggregate includes the derivative complex; it is not option-only volume.",
        "url": "https://www.safe.gov.cn/en/2025/0722/2328.html",
    },
    {
        "fact": "Full-year 2025 Chinese FX turnover",
        "value": "USD 42.64tn",
        "scope": "SAFE total FX market turnover; December derivatives were USD 2.51tn.",
        "url": "https://www.safe.gov.cn/en/2026/0130/2384.html",
    },
    {
        "fact": "Public USD/CNY option curve",
        "value": "13 tenors; ATM, direct wings, RR and BF",
        "scope": "CFETS says inputs include trades, broker quotes and bilateral platform quotes.",
        "url": "https://www.chinamoney.com.cn/english/bmkycvivc/",
    },
    {
        "fact": "RMB/FX option trading",
        "value": "USD/CNY vanilla, risk reversal and butterfly",
        "scope": "CFETS describes bilateral inquiry; public curve depth is not executable size.",
        "url": "https://www.chinamoney.com.cn/english/prddmkopt/",
    },
)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must contain a JSON object: {path}")
    if "schema_version" not in payload:
        raise ValueError(f"artifact is missing schema_version: {path}")
    return payload


def _walk(payload: Any, path: Sequence[str]) -> Any:
    value = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _first(payload: Mapping[str, Any], *paths: Sequence[str]) -> Any:
    for path in paths:
        value = _walk(payload, path)
        if value is not None:
            return value
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    number = _number(value)
    if number is None:
        return "—"
    return f"{number:,.{digits}f}{suffix}"


def _fmt_param(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "—"
    if abs(number) < 0.01:
        return f"{number:.6f}"
    return f"{number:.4f}"


def _metric_vol_points(payload: Mapping[str, Any]) -> float | None:
    direct = _first(
        payload,
        ("overall_rmse_vol_points",),
        ("rmse_vol_points",),
        ("fit", "rmse_vol_points"),
        ("repricing", "rmse_vol_points"),
        ("vanilla_repricing", "rmse_vol_points"),
        ("diagnostics", "rmse_vol_points"),
        ("raw_composite_fit", "in_prepared_domain", "rmse_vol_points"),
        ("raw_composite_fit", "rmse_vol_points"),
        ("prepared_target_fit", "rmse_vol_points"),
    )
    value = _number(direct)
    if value is not None:
        return value
    iv = _first(
        payload,
        ("overall_rmse_iv",),
        ("rmse_iv",),
        ("repricing", "overall_rmse_iv"),
        ("vanilla_repricing", "overall_rmse_iv"),
    )
    value = _number(iv)
    return None if value is None else value * 100.0


def load_artifacts(data_dir: str | Path, tag: str) -> dict[str, dict[str, Any]]:
    """Load and cross-check the five required stage artifacts.

    Missing inputs are a hard error: an explainer without one of the model or
    diagnostics artifacts would look complete while silently omitting evidence.
    """
    if not SAFE_TAG.fullmatch(tag):
        raise ValueError("tag must contain only letters, numbers, '.', '_' or '-'")
    directory = Path(data_dir)
    paths = {
        name: directory / f"{stem}_{tag}.json"
        for name, stem in ARTIFACT_STEMS.items()
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(f"required FX explainer artifacts are missing:\n  - {formatted}")

    artifacts = {name: _read_json(path) for name, path in paths.items()}
    dates = {
        str(payload["trade_date"])
        for payload in artifacts.values()
        if payload.get("trade_date")
    }
    if len(dates) > 1:
        raise ValueError(f"FX explainer artifacts have inconsistent trade_date values: {sorted(dates)}")
    for name, payload in artifacts.items():
        pair = payload.get("currency_pair")
        if pair is not None and pair != "USD.CNY":
            raise ValueError(f"{name} artifact currency_pair must be USD.CNY, got {pair!r}")
    return artifacts


def _surface_slices(surface: Mapping[str, Any]) -> list[dict[str, Any]]:
    slices = surface.get("slices")
    if not isinstance(slices, list) or not slices:
        raise ValueError("surface artifact requires a non-empty slices list")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(slices):
        if isinstance(row, Mapping) and row.get("role", "calibration_target") != "calibration_target":
            continue
        if not isinstance(row, Mapping) or not row.get("tenor"):
            raise ValueError(f"surface slice {index} requires tenor")
        maturity = _number(row.get("maturity"))
        forward = _number(row.get("forward"))
        if maturity is None or maturity <= 0.0:
            raise ValueError(f"surface slice {row.get('tenor')} requires a positive maturity")
        if forward is None or forward <= 0.0:
            raise ValueError(f"surface slice {row.get('tenor')} requires a positive forward")
        quotes = row.get("raw_quotes", row.get("quotes"))
        if not isinstance(quotes, list) or not quotes:
            raise ValueError(f"surface slice {row.get('tenor')} requires raw_quotes")
        normalized_quotes: list[dict[str, Any]] = []
        for quote in quotes:
            if not isinstance(quote, Mapping) or not quote.get("pillar"):
                raise ValueError(f"surface slice {row.get('tenor')} has an invalid quote")
            mid = _number(quote.get("mid_iv", quote.get("market_iv")))
            strike = _number(quote.get("strike"))
            bid = _number(quote.get("bid_iv"))
            ask = _number(quote.get("ask_iv"))
            if (
                mid is None
                or bid is None
                or ask is None
                or strike is None
                or min(mid, bid, ask, strike) <= 0.0
            ):
                raise ValueError(f"surface quote {row.get('tenor')} {quote.get('pillar')} is invalid")
            normalized_quotes.append(
                {
                    "pillar": str(quote["pillar"]),
                    "delta": quote.get("delta"),
                    "strike": strike,
                    "bid_iv": bid,
                    "mid_iv": mid,
                    "ask_iv": ask,
                }
            )
        result.append(
            {
                "tenor": str(row["tenor"]),
                "maturity": maturity,
                "expiry_date": row.get("expiry_date", "—"),
                "forward": forward,
                "domestic_rate": _number(row.get("domestic_rate")),
                "foreign_rate": _number(row.get("foreign_rate")),
                "quotes": normalized_quotes,
            }
        )
    return result


def _select_universe(heston: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    universes = heston.get("universes")
    if isinstance(universes, Mapping) and universes:
        requested = str(diagnostics.get("universe") or "core")
        if requested not in universes:
            requested = "core" if "core" in universes else next(iter(universes))
        universe = universes[requested]
        if not isinstance(universe, Mapping):
            raise ValueError(f"heston universe {requested!r} must be an object")
        return requested, universe
    return str(heston.get("universe", "reported")), heston


def _heston_best(heston: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    universe_name, universe = _select_universe(heston, diagnostics)
    best = _first(
        universe,
        ("free", "best"),
        ("selected_fit",),
        ("best",),
        ("fit",),
    )
    if not isinstance(best, Mapping):
        raise ValueError("heston artifact requires universes.<name>.free.best calibration evidence")
    params = best.get("params")
    if not isinstance(params, Mapping) or any(_number(params.get(name)) is None for name in PARAMETERS):
        raise ValueError(f"heston best fit requires parameters {PARAMETERS}")
    return universe_name, best


def _model_row_lookup(best: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows = best.get("rows", [])
    if not isinstance(rows, list):
        return {}
    result = {}
    for row in rows:
        if isinstance(row, Mapping) and row.get("tenor") and row.get("pillar"):
            result[(str(row["tenor"]), str(row["pillar"]))] = row
    return result


def _history(diagnostics: Mapping[str, Any], best: Mapping[str, Any], trade_date: str) -> list[dict[str, Any]]:
    included = diagnostics.get("included")
    rows: list[dict[str, Any]] = []
    if isinstance(included, list):
        for item in included:
            if not isinstance(item, Mapping):
                continue
            mode = _first(item, ("modes", "free"), ("free",), ("fit",))
            if not isinstance(mode, Mapping):
                continue
            params = mode.get("params")
            if not isinstance(params, Mapping):
                continue
            rows.append(
                {
                    "tag": str(item.get("tag", "—")),
                    "trade_date": str(item.get("trade_date", "—")),
                    "params": {name: _number(params.get(name)) for name in PARAMETERS},
                    "rmse_vol_points": _number(mode.get("rmse_vol_points")),
                    "feller_ratio": _number(mode.get("feller_ratio")),
                    "jacobian_condition": _number(
                        mode.get("jacobian_condition", _first(mode, ("jacobian", "condition_number")))
                    ),
                }
            )
    if not rows:
        params = best["params"]
        rows.append(
            {
                "tag": "current",
                "trade_date": trade_date,
                "params": {name: _number(params.get(name)) for name in PARAMETERS},
                "rmse_vol_points": _number(best.get("rmse_vol_points")),
                "feller_ratio": _number(best.get("feller_ratio")),
                "jacobian_condition": _number(
                    _first(best, ("jacobian_condition",), ("jacobian", "condition_number"))
                ),
            }
        )
    return rows


def _verdict_rows(diagnostics: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    raw = diagnostics.get("verdicts", [])
    result: list[tuple[str, str, str]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                result.append(("conditional", item, "Artifact diagnostic"))
            elif isinstance(item, Mapping):
                status = str(item.get("status", item.get("verdict", "conditional"))).lower()
                title = str(item.get("title", item.get("name", item.get("test", "Diagnostic"))))
                detail = str(
                    item.get(
                        "detail",
                        item.get(
                            "interpretation",
                            item.get("reason", item.get("message", "")),
                        ),
                    )
                )
                result.append((status, title, detail))
    if not result:
        result.append(
            (
                "conditional",
                "Public-composite calibration",
                "The artifacts support an in-sample model study, not an executable-liquidity conclusion.",
            )
        )
    return result


def _limitations(*payloads: Mapping[str, Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for payload in payloads:
        values = payload.get("limitations", [])
        if not isinstance(values, list):
            continue
        for value in values:
            text = str(value).strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
    return result


def _status_class(status: str) -> str:
    text = status.lower()
    if any(token in text for token in ("pass", "support", "qualified", "green")):
        return "pass"
    if any(token in text for token in ("fail", "reject", "insufficient", "red")):
        return "fail"
    if any(token in text for token in ("not_test", "not test", "missing")):
        return "neutral"
    return "warn"


def _artifact_table(tag: str, artifacts: Mapping[str, Mapping[str, Any]]) -> str:
    rows = []
    for name, stem in ARTIFACT_STEMS.items():
        payload = artifacts[name]
        rows.append(
            "<tr>"
            f"<td><code>{_escape(stem)}_{_escape(tag)}.json</code></td>"
            f"<td class='n'>{_escape(payload.get('schema_version'))}</td>"
            f"<td>{_escape(payload.get('trade_date', 'shared study'))}</td>"
            f"<td>{_escape(name)}</td>"
            "</tr>"
        )
    return "".join(rows)


CSS = r"""
:root{
  --paper:#F7F6F2;--panel:#EFEDE6;--panel2:#E7E4DB;--ink:#20242C;
  --ink2:#4A5160;--faint:#858B98;--line:#D8D5CC;--grid:#DCDAD2;
  --cinnabar:#BE3A2B;--cinnabar-soft:#BE3A2B22;--jade:#2F7D6D;
  --jade-soft:#2F7D6D1E;--amber:#A66B1F;--amber-soft:#A66B1F1F;
  --slate:#5B6B84;--slate-soft:#5B6B841B;--code-bg:#ECEAE2;
}
@media(prefers-color-scheme:dark){:root{
  --paper:#151A21;--panel:#1C222B;--panel2:#232A35;--ink:#E2E4E8;
  --ink2:#AAB1BE;--faint:#79808E;--line:#2E3541;--grid:#252C37;
  --cinnabar:#E0604C;--cinnabar-soft:#E0604C26;--jade:#4FA98F;
  --jade-soft:#4FA98F22;--amber:#D9A03F;--amber-soft:#D9A03F22;
  --slate:#8FA2BE;--slate-soft:#8FA2BE20;--code-bg:#1A202A}}
:root[data-theme='dark']{--paper:#151A21;--panel:#1C222B;--panel2:#232A35;--ink:#E2E4E8;--ink2:#AAB1BE;--faint:#79808E;--line:#2E3541;--grid:#252C37;--cinnabar:#E0604C;--cinnabar-soft:#E0604C26;--jade:#4FA98F;--jade-soft:#4FA98F22;--amber:#D9A03F;--amber-soft:#D9A03F22;--slate:#8FA2BE;--slate-soft:#8FA2BE20;--code-bg:#1A202A}
:root[data-theme='light']{--paper:#F7F6F2;--panel:#EFEDE6;--panel2:#E7E4DB;--ink:#20242C;--ink2:#4A5160;--faint:#858B98;--line:#D8D5CC;--grid:#DCDAD2;--cinnabar:#BE3A2B;--cinnabar-soft:#BE3A2B22;--jade:#2F7D6D;--jade-soft:#2F7D6D1E;--amber:#A66B1F;--amber-soft:#A66B1F1F;--slate:#5B6B84;--slate-soft:#5B6B841B;--code-bg:#ECEAE2}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:Charter,"Bitstream Charter",Cambria,Georgia,"Noto Serif SC",serif;font-size:17px;line-height:1.62}
main,.hero,.footer{max-width:72ch;margin:0 auto;padding-left:20px;padding-right:20px}main{padding-bottom:6rem}
h1,h2,h3,h4{font-family:Palatino,"Palatino Linotype","URW Palladio L",Georgia,"Noto Serif SC",serif;line-height:1.18;text-wrap:balance}h1{font-size:clamp(2.2rem,6vw,3.5rem);font-weight:600;margin:.35rem 0 .8rem;letter-spacing:-.025em}h2{font-size:1.75rem;margin:0 0 1rem}h3{font-size:1.2rem;margin:2rem 0 .55rem}p{margin:.85rem 0}section{margin-top:4.7rem;scroll-margin-top:4rem}a{color:var(--cinnabar);text-underline-offset:2px}.mono,code,.num{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}code{font-size:.84em;background:var(--code-bg);padding:.08em .35em;border-radius:3px}
.topnav{position:sticky;top:0;z-index:50;background:color-mix(in srgb,var(--paper) 94%,transparent);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}.topnav-inner{max-width:1060px;margin:0 auto;padding:.48rem 20px;display:flex;align-items:center;gap:1rem;flex-wrap:wrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.7rem}.brand{letter-spacing:.12em;text-transform:uppercase;color:var(--ink2);font-weight:700}.topnav nav{display:flex;gap:.78rem;flex:1;flex-wrap:wrap}.topnav a{color:var(--ink2);text-decoration:none;text-transform:uppercase;letter-spacing:.05em}.topnav a:hover{color:var(--cinnabar)}
.hero{padding-top:3.5rem}.eyebrow{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.7rem;letter-spacing:.15em;text-transform:uppercase;color:var(--cinnabar);margin:0 0 .3rem}.kicker{color:var(--ink2);font-size:1.08rem;max-width:62ch}.hero-meta{display:flex;gap:1rem 1.5rem;flex-wrap:wrap;margin-top:1.3rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.73rem;color:var(--faint)}.hero-meta b{color:var(--ink)}
.composite-banner{margin:1.5rem 0;padding:.8rem 1rem;border:1px solid var(--amber);border-left:5px solid var(--amber);background:var(--amber-soft);border-radius:0 7px 7px 0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.75rem;line-height:1.55}.composite-banner b{display:block;color:var(--amber);letter-spacing:.12em;text-transform:uppercase}
.toc{columns:2;gap:2.4rem;margin:2rem 0 0}.toc a{display:block;color:var(--ink);text-decoration:none;padding:.2rem 0;border-bottom:1px dotted var(--line);break-inside:avoid}.toc .no{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.68rem;color:var(--cinnabar);margin-right:.55rem}
.wide{width:min(1000px,calc(100vw - 32px));margin-left:50%;transform:translateX(-50%)}figure{margin:2rem 0}.fig-frame{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:1rem 1.1rem 1.15rem;overflow-x:auto;box-shadow:0 18px 50px color-mix(in srgb,var(--ink) 5%,transparent)}figcaption{font-size:.82rem;color:var(--ink2);margin-top:.65rem;line-height:1.48;max-width:90ch}.figno{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.69rem;letter-spacing:.1em;text-transform:uppercase;color:var(--cinnabar);margin-right:.5em}canvas{display:block;width:100%;height:auto}
.controls{display:flex;gap:.8rem 1.3rem;flex-wrap:wrap;align-items:end;margin-bottom:.8rem}.ctl{display:flex;flex-direction:column;gap:.18rem}.ctl label{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.66rem;letter-spacing:.09em;text-transform:uppercase;color:var(--ink2)}button,select{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.72rem;background:var(--panel2);color:var(--ink);border:1px solid var(--line);border-radius:5px;padding:.4rem .72rem}button{cursor:pointer}button:hover,button:focus-visible,select:focus-visible{border-color:var(--cinnabar);outline:2px solid transparent}button[aria-pressed='true']{background:var(--ink);color:var(--paper);border-color:var(--ink)}.chiprow{display:flex;gap:.35rem;flex-wrap:wrap}.readout{display:flex;gap:.7rem 1.3rem;flex-wrap:wrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.72rem;margin-top:.65rem}.readout .k{color:var(--faint)}.readout .v{font-weight:700}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:.75rem;margin:1.2rem 0}.metric-card{border:1px solid var(--line);border-radius:7px;background:var(--panel);padding:.8rem .9rem}.metric-card .label{display:block;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.64rem;letter-spacing:.08em;text-transform:uppercase;color:var(--faint)}.metric-card .value{display:block;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:1.15rem;font-weight:700;margin:.16rem 0}.metric-card .sub{font-size:.78rem;color:var(--ink2)}
.tbl-wrap{overflow-x:auto;margin:1.2rem 0}table{border-collapse:collapse;width:100%;font-size:.84rem}th,td{padding:.42rem .62rem;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}thead th{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.64rem;letter-spacing:.08em;text-transform:uppercase;color:var(--ink2);border-bottom:2px solid var(--ink2)}td.n,th.n{text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums;white-space:nowrap}.fallback-evidence{margin-top:1rem}.fallback-evidence summary{font-weight:700}
.chip{display:inline-block;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.64rem;letter-spacing:.05em;padding:.12em .55em;border-radius:99px;font-weight:700}.chip.pass{background:var(--jade-soft);color:var(--jade)}.chip.warn{background:var(--amber-soft);color:var(--amber)}.chip.fail{background:var(--cinnabar-soft);color:var(--cinnabar)}.chip.neutral{background:var(--slate-soft);color:var(--slate)}
.note{border:1px solid var(--line);border-left:4px solid var(--jade);background:var(--jade-soft);border-radius:0 6px 6px 0;padding:.82rem 1.05rem;margin:1.35rem 0;font-size:.92rem}.note.warn{border-left-color:var(--amber);background:var(--amber-soft)}.note.risk{border-left-color:var(--cinnabar);background:var(--cinnabar-soft)}.note .t{display:block;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;font-weight:700;color:var(--jade);margin-bottom:.2rem}.note.warn .t{color:var(--amber)}.note.risk .t{color:var(--cinnabar)}
.eq{margin:1.2rem 0;padding:1rem 1.15rem;background:var(--panel);border-left:3px solid var(--slate);border-radius:0 6px 6px 0;overflow-x:auto;font-family:Palatino,"Palatino Linotype",Georgia,serif;font-size:1rem;line-height:1.9}.eq .lbl{float:right;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.66rem;color:var(--faint);margin-left:1rem}
details{border:1px solid var(--line);border-radius:6px;margin:1rem 0;background:var(--panel)}details summary{cursor:pointer;padding:.7rem 1rem;font-weight:700;font-size:.9rem}details summary::marker{color:var(--cinnabar)}details .body{padding:0 1.05rem 1rem;font-size:.9rem}.pipe{display:flex;flex-wrap:wrap;gap:.45rem;align-items:stretch;margin:1.2rem 0}.pipe .box{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:.55rem .7rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.68rem;line-height:1.45;flex:1 1 145px}.pipe .box b{display:block;color:var(--cinnabar)}.pipe .arrow{align-self:center;color:var(--faint)}
.footer{padding-top:2rem;padding-bottom:4rem;border-top:1px solid var(--line);color:var(--faint);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.76rem}
@media(max-width:700px){.toc{columns:1}.topnav nav{display:none}.wide{width:calc(100vw - 20px)}body{font-size:16px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@media print{.topnav,.controls,.interactive-only{display:none!important}.wide{width:100%;margin-left:0;transform:none}.fig-frame{box-shadow:none}details{break-inside:avoid}details>.body{display:block}.footer{padding-bottom:1rem}}
"""


JS = r"""
(function(){
"use strict";
var DATA=window.__FX_REPORT_DATA__;
function css(name){return getComputedStyle(document.documentElement).getPropertyValue(name).trim();}
function finite(v){return typeof v==='number'&&Number.isFinite(v);}
function fmt(v,d){return finite(v)?v.toFixed(d===undefined?3:d):'—';}
function setupCanvas(canvas,height){var dpr=window.devicePixelRatio||1,W=canvas.clientWidth||880,H=height;canvas.width=W*dpr;canvas.height=H*dpr;var g=canvas.getContext('2d');g.scale(dpr,dpr);g.clearRect(0,0,W,H);return{g:g,W:W,H:H};}
function smileSlice(){var sel=document.getElementById('tenorSelect');return DATA.smiles[Number(sel.value)||0];}
function drawSmile(){
  var canvas=document.getElementById('smileCanvas'),c=setupCanvas(canvas,350),g=c.g,W=c.W,H=c.H,s=smileSlice();
  var pad={l:60,r:18,t:20,b:46},nodes=s.nodes.slice().sort(function(a,b){return a.strike-b.strike;});
  var xs=nodes.map(function(n){return n.strike/s.forward;}),ys=[];
  nodes.forEach(function(n){ys.push(n.bid*100,n.mid*100,n.ask*100);if(finite(n.model))ys.push(n.model*100);});
  var xmin=Math.min.apply(null,xs),xmax=Math.max.apply(null,xs),ymin=Math.min.apply(null,ys),ymax=Math.max.apply(null,ys);
  var xp=Math.max((xmax-xmin)*.12,.004),yp=Math.max((ymax-ymin)*.22,.03);xmin-=xp;xmax+=xp;ymin-=yp;ymax+=yp;
  function X(x){return pad.l+(W-pad.l-pad.r)*(x-xmin)/(xmax-xmin);}function Y(y){return pad.t+(H-pad.t-pad.b)*(ymax-y)/(ymax-ymin);}
  g.font='10px ui-monospace,Menlo,monospace';g.lineWidth=1;g.strokeStyle=css('--grid');g.fillStyle=css('--faint');
  for(var i=0;i<=4;i++){var y=ymin+(ymax-ymin)*i/4;g.beginPath();g.moveTo(pad.l,Y(y));g.lineTo(W-pad.r,Y(y));g.stroke();g.textAlign='right';g.fillText(y.toFixed(2)+'%',pad.l-7,Y(y)+3);}
  nodes.forEach(function(n,i){var x=X(n.strike/s.forward);g.strokeStyle=css('--amber');g.lineWidth=5;g.beginPath();g.moveTo(x,Y(n.bid*100));g.lineTo(x,Y(n.ask*100));g.stroke();g.fillStyle=css('--ink2');g.textAlign='center';g.fillText(n.pillar,x,H-16);});
  g.strokeStyle=css('--slate');g.lineWidth=1.5;g.beginPath();nodes.forEach(function(n,i){var x=X(n.strike/s.forward),y=Y(n.mid*100);if(i===0)g.moveTo(x,y);else g.lineTo(x,y);});g.stroke();
  nodes.forEach(function(n){g.fillStyle=css('--slate');g.beginPath();g.arc(X(n.strike/s.forward),Y(n.mid*100),4,0,Math.PI*2);g.fill();});
  var model=nodes.filter(function(n){return finite(n.model);});if(model.length){g.strokeStyle=css('--cinnabar');g.lineWidth=2.2;g.beginPath();model.forEach(function(n,i){var x=X(n.strike/s.forward),y=Y(n.model*100);if(i===0)g.moveTo(x,y);else g.lineTo(x,y);});g.stroke();}
  g.fillStyle=css('--faint');g.textAlign='center';g.fillText('strike / forward',pad.l+(W-pad.l-pad.r)/2,H-2);
  document.getElementById('smileReadout').innerHTML='<span><span class="k">tenor </span><span class="v">'+s.tenor+'</span></span><span><span class="k">forward </span><span class="v">'+fmt(s.forward,4)+'</span></span><span><span class="k">nodes </span><span class="v">'+nodes.length+'</span></span><span><span class="k">amber bar </span><span class="v">public composite bid/ask display</span></span>';
}
var activeMetric='v0';
function drawStability(){
  var canvas=document.getElementById('stabilityCanvas'),c=setupCanvas(canvas,330),g=c.g,W=c.W,H=c.H,rows=DATA.history;
  var pad={l:68,r:18,t:22,b:52},values=rows.map(function(r){return activeMetric in r.params?r.params[activeMetric]:r[activeMetric];}).filter(finite);
  if(!values.length){g.fillStyle=css('--faint');g.font='13px ui-monospace,Menlo,monospace';g.fillText('No finite '+activeMetric+' history in diagnostics artifact.',pad.l,60);return;}
  var ymin=Math.min.apply(null,values),ymax=Math.max.apply(null,values);if(ymin===ymax){var bump=Math.max(Math.abs(ymin)*.15,.001);ymin-=bump;ymax+=bump;}else{var yp=(ymax-ymin)*.15;ymin-=yp;ymax+=yp;}
  function X(i){return rows.length===1?(pad.l+W-pad.r)/2:pad.l+(W-pad.l-pad.r)*i/(rows.length-1);}function Y(y){return pad.t+(H-pad.t-pad.b)*(ymax-y)/(ymax-ymin);}
  g.font='10px ui-monospace,Menlo,monospace';g.strokeStyle=css('--grid');g.fillStyle=css('--faint');g.lineWidth=1;
  for(var i=0;i<=4;i++){var y=ymin+(ymax-ymin)*i/4;g.beginPath();g.moveTo(pad.l,Y(y));g.lineTo(W-pad.r,Y(y));g.stroke();g.textAlign='right';g.fillText(y.toPrecision(4),pad.l-8,Y(y)+3);}
  g.strokeStyle=css('--cinnabar');g.lineWidth=2.1;g.beginPath();var started=false;rows.forEach(function(r,i){var v=activeMetric in r.params?r.params[activeMetric]:r[activeMetric];if(!finite(v))return;if(!started){g.moveTo(X(i),Y(v));started=true;}else g.lineTo(X(i),Y(v));});g.stroke();
  rows.forEach(function(r,i){var v=activeMetric in r.params?r.params[activeMetric]:r[activeMetric];if(!finite(v))return;g.fillStyle=css('--cinnabar');g.beginPath();g.arc(X(i),Y(v),4,0,Math.PI*2);g.fill();g.fillStyle=css('--ink2');var label=W<650?r.trade_date.slice(5):r.trade_date;g.textAlign=i===0?'left':(i===rows.length-1?'right':'center');g.fillText(label,X(i),H-18);});
  document.getElementById('stabilityReadout').innerHTML='<span><span class="k">metric </span><span class="v">'+activeMetric+'</span></span><span><span class="k">dates </span><span class="v">'+rows.length+'</span></span><span><span class="k">range </span><span class="v">'+fmt(Math.min.apply(null,values),6)+' → '+fmt(Math.max.apply(null,values),6)+'</span></span>';
}
function boot(){
  var tenor=document.getElementById('tenorSelect');DATA.smiles.forEach(function(s,i){var o=document.createElement('option');o.value=String(i);o.textContent=s.tenor;tenor.appendChild(o);});tenor.addEventListener('change',drawSmile);
  document.querySelectorAll('[data-stability-metric]').forEach(function(button){button.addEventListener('click',function(){document.querySelectorAll('[data-stability-metric]').forEach(function(b){b.setAttribute('aria-pressed','false');});button.setAttribute('aria-pressed','true');activeMetric=button.getAttribute('data-stability-metric');drawStability();});});
  document.getElementById('themeToggle').addEventListener('click',function(){var root=document.documentElement,current=root.getAttribute('data-theme');if(!current){current=window.matchMedia&&window.matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light';}root.setAttribute('data-theme',current==='dark'?'light':'dark');drawSmile();drawStability();});
  var timer;window.addEventListener('resize',function(){clearTimeout(timer);timer=setTimeout(function(){drawSmile();drawStability();},120);});drawSmile();drawStability();
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',boot);else boot();
})();
"""


def render_html(artifacts: Mapping[str, Mapping[str, Any]], tag: str) -> str:
    surface = artifacts["surface"]
    localvol = artifacts["localvol"]
    heston = artifacts["heston"]
    slv = artifacts["slv"]
    diagnostics = artifacts["diagnostics"]
    slices = _surface_slices(surface)
    universe_name, universe = _select_universe(heston, diagnostics)
    _, best = _heston_best(heston, diagnostics)
    hard_best = _first(universe, ("hard_feller", "best"))
    model_rows = _model_row_lookup(best)
    trade_date = str(surface.get("trade_date", heston.get("trade_date", "—")))
    quote_time = str(surface.get("quote_time", heston.get("quote_time", "—")))
    params = best["params"]
    history = _history(diagnostics, best, trade_date)
    verdicts = _verdict_rows(diagnostics)

    smiles = []
    node_rows = []
    for row in slices:
        forward = row["forward"]
        nodes = []
        for quote in row["quotes"]:
            model = model_rows.get((row["tenor"], quote["pillar"]), {})
            model_iv = _number(model.get("model_iv"))
            nodes.append(
                {
                    "pillar": quote["pillar"],
                    "strike": quote["strike"],
                    "bid": quote["bid_iv"],
                    "mid": quote["mid_iv"],
                    "ask": quote["ask_iv"],
                    "model": model_iv,
                }
            )
            node_rows.append(
                "<tr>"
                f"<td>{_escape(row['tenor'])}</td><td>{_escape(quote['pillar'])}</td>"
                f"<td class='n'>{_fmt(quote['strike'], 4)}</td>"
                f"<td class='n'>{_fmt(quote['bid_iv'] * 100, 3, '%')}</td>"
                f"<td class='n'>{_fmt(quote['mid_iv'] * 100, 3, '%')}</td>"
                f"<td class='n'>{_fmt(quote['ask_iv'] * 100, 3, '%')}</td>"
                f"<td class='n'>{_fmt(None if model_iv is None else model_iv * 100, 3, '%')}</td>"
                "</tr>"
            )
        smiles.append({"tenor": row["tenor"], "forward": forward, "nodes": nodes})

    history_table = []
    for row in history:
        history_table.append(
            "<tr>"
            f"<td>{_escape(row['tag'])}</td><td>{_escape(row['trade_date'])}</td>"
            + "".join(f"<td class='n'>{_fmt_param(row['params'].get(name))}</td>" for name in PARAMETERS)
            + f"<td class='n'>{_fmt(row.get('rmse_vol_points'), 4)}</td>"
            + f"<td class='n'>{_fmt(row.get('feller_ratio'), 4)}</td>"
            + f"<td class='n'>{_fmt(row.get('jacobian_condition'), 3)}</td>"
            + "</tr>"
        )

    tenor_table = []
    for row in slices:
        tenor_table.append(
            "<tr>"
            f"<td>{_escape(row['tenor'])}</td><td>{_escape(row['expiry_date'])}</td>"
            f"<td class='n'>{_fmt(row['maturity'], 4)}</td><td class='n'>{_fmt(row['forward'], 4)}</td>"
            f"<td class='n'>{_fmt(None if row['domestic_rate'] is None else row['domestic_rate'] * 100, 3, '%')}</td>"
            f"<td class='n'>{_fmt(None if row['foreign_rate'] is None else row['foreign_rate'] * 100, 3, '%')}</td>"
            f"<td class='n'>{len(row['quotes'])}</td></tr>"
        )

    verdict_html = "".join(
        "<div class='metric-card'>"
        f"<span class='label'><span class='chip {_status_class(status)}'>{_escape(status)}</span></span>"
        f"<span class='value'>{_escape(title)}</span><span class='sub'>{_escape(detail)}</span></div>"
        for status, title, detail in verdicts
    )
    limitation_values = _limitations(surface, localvol, heston, slv, diagnostics)
    limitation_html = "".join(f"<li>{_escape(value)}</li>" for value in limitation_values)
    if not limitation_html:
        limitation_html = "<li>No limitations array was published; rely on the public-composite qualification above.</li>"

    prep = surface.get("surface_preparation", {})
    if not isinstance(prep, Mapping):
        prep = {}
    localvol_rmse = _metric_vol_points(localvol)
    slv_rmse = _metric_vol_points(slv)
    heston_rmse = _number(best.get("rmse_vol_points"))
    hard_heston_rmse = (
        _number(hard_best.get("rmse_vol_points")) if isinstance(hard_best, Mapping) else None
    )
    fit_gate_vol_points = 0.10
    fit_gate_pass = heston_rmse is not None and heston_rmse <= fit_gate_vol_points
    coverage = _number(
        _first(
            best,
            ("inside_nonzero_public_band_pct",),
            ("inside_public_band_pct",),
            ("coverage", "inside_nonzero_public_band_pct"),
            ("coverage", "inside_public_band_pct"),
        )
    )
    feller = _number(best.get("feller_ratio"))
    source_node_count = int(surface.get("observed_node_count", sum(len(row["quotes"]) for row in slices)))
    cross_date_status = "MULTI-DATE" if len(history) >= 2 else "SINGLE SNAPSHOT"
    cross_date_class = "pass" if len(history) >= 2 else "warn"
    provenance = surface.get("provenance", {})
    if not isinstance(provenance, Mapping):
        provenance = {}
    source_links = []
    for label, key in (
        ("CFETS volatility-curve methodology", "methodology_url"),
        ("CFETS delta-parameter page", "delta_parameters_url"),
        ("CFETS public curve endpoint", "curve_endpoint"),
        ("CFETS public delta endpoint", "delta_endpoint"),
    ):
        url = provenance.get(key)
        if isinstance(url, str) and url.startswith("https://"):
            source_links.append(f'<li><a href="{_escape(url)}">{_escape(label)}</a></li>')
    source_links_html = "".join(source_links) or "<li>Source URLs were not carried by the surface artifact.</li>"
    market_evidence_rows = "".join(
        "<tr>"
        f'<td><a href="{_escape(item["url"])}">{_escape(item["fact"])}</a></td>'
        f'<td>{_escape(item["value"])}</td><td>{_escape(item["scope"])}</td>'
        "</tr>"
        for item in MARKET_EVIDENCE
    )
    report_data = json.dumps(
        {"smiles": smiles, "history": history}, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")

    param_rows = "".join(
        f"<tr><td><code>{name}</code></td><td class='n'>{_fmt_param(params.get(name))}</td><td>{description}</td></tr>"
        for name, description in (
            ("v0", "initial variance; short-tenor level"),
            ("kappa", "variance mean-reversion speed; term-structure curvature"),
            ("theta", "long-run variance; long-tenor level"),
            ("sigma", "vol-of-vol; smile convexity"),
            ("rho", "spot/variance correlation; risk-reversal skew"),
        )
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>CFETS USD/CNY Heston calibration — evidence-led explainer</title>
<style>{CSS}</style>
</head>
<body>
<div class="topnav"><div class="topnav-inner"><span class="brand">CFETS · USD/CNY · {_escape(trade_date)}</span><nav>
<a href="#s1">Verdict</a><a href="#s2">Quotes</a><a href="#s3">Surface</a><a href="#s4">Heston</a><a href="#s5">Dynamics</a><a href="#s6">Stability</a><a href="#s7">Limits</a><a href="#s8">Appendix</a>
</nav><button id="themeToggle" type="button" aria-label="Toggle light and dark theme">Theme</button></div></div>

<header class="hero">
<p class="eyebrow">Technical explainer · CFETS USD/CNY + QuantArk</p>
<h1>Can China’s onshore FX market support Heston calibration?</h1>
<p class="kicker">The surface geometry is rich enough to run a serious vanilla calibration. This report keeps the harder question separate: whether a fit to a public composite curve is stable, executable, and good enough for model promotion.</p>
<div class="composite-banner"><b>Public composite · not executable history</b>The displayed CFETS bid, mid and ask values are benchmark outputs. A zero displayed spread is not evidence of a zero executable spread, and no fit percentage on this page is a liquidity claim.</div>
<div class="hero-meta"><span>Pair <b>USD/CNY</b></span><span>Trade date <b>{_escape(trade_date)}</b></span><span>Quote time <b>{_escape(quote_time)}</b></span><span>Spot <b>{_fmt(surface.get('spot'),4)}</b></span><span>Artifact tag <b>{_escape(tag)}</b></span></div>
<div class="toc">
<a href="#s1"><span class="no">§1</span>Decision: what the artifacts prove</a><a href="#s2"><span class="no">§2</span>The five-delta CFETS smile</a>
<a href="#s3"><span class="no">§3</span>From sparse quotes to a surface</a><a href="#s4"><span class="no">§4</span>Heston fit and identification</a>
<a href="#s5"><span class="no">§5</span>Local vol, SLV, and dynamics</a><a href="#s6"><span class="no">§6</span>Cross-date robustness</a>
<a href="#s7"><span class="no">§7</span>Comparison, assumptions, limits</a><a href="#s8"><span class="no">§8</span>Artifacts and reproduction</a>
</div></header>

<main>
<section id="s1"><p class="eyebrow">§1 · Decision</p><h2>Calibration-ready is not calibration-proven</h2>
<p>The artifact set contains <b class="num">{source_node_count}</b> observed five-delta nodes across <b class="num">{len(slices)}</b> selected tenors. That is enough cross-sectional geometry to fit Heston, but the source remains a public composite. The defensible conclusion is therefore conditional: USD/CNY supports the calibration exercise; promotion requires executable-history and stability evidence.</p>
<div class="metric-grid">
<div class="metric-card"><span class="label">Surface geometry</span><span class="value"><span class="chip pass">SUPPORTED</span></span><span class="sub">{source_node_count} artifact nodes · {len(slices)} tenors</span></div>
<div class="metric-card"><span class="label">Raw-node fit gate</span><span class="value"><span class="chip {'pass' if fit_gate_pass else 'fail'}">{'PASS' if fit_gate_pass else 'FAIL'}</span></span><span class="sub">{_fmt(heston_rmse,4)} ≤ {_fmt(fit_gate_vol_points,2)} vol pts (10 vol bp)</span></div>
<div class="metric-card"><span class="label">Heston universe</span><span class="value">{_escape(universe_name)}</span><span class="sub">reported by the Heston artifact</span></div>
<div class="metric-card"><span class="label">Cross-date evidence</span><span class="value"><span class="chip {cross_date_class}">{cross_date_status}</span></span><span class="sub">{len(history)} included calibration date(s)</span></div>
<div class="metric-card"><span class="label">H1 2025 RMB FX scale</span><span class="value">USD 21.0tn</span><span class="sub">USD 13.6tn derivatives; not option-only volume</span></div>
<div class="metric-card"><span class="label">Production promotion</span><span class="value"><span class="chip warn">CONDITIONAL</span></span><span class="sub">licensed executable history still governs</span></div>
</div><div class="metric-grid">{verdict_html}</div>
<div class="note warn"><span class="t">Decision boundary</span>A low in-sample RMSE can demonstrate model fit to these composite nodes. It cannot establish executable liquidity, parameter identification, next-day stability, or exotic-model adequacy.</div></section>

<section id="s2"><p class="eyebrow">§2 · Quote geometry</p><h2>The CFETS five-delta smile</h2>
<p>Each tenor contributes 10-delta put, 25-delta put, ATM, 25-delta call and 10-delta call coordinates. CFETS uses premium-excluded <b>spot delta</b> here, and ATM is ATMF (<code>K = F</code>). The stage-02 artifact stores the reconstructed strikes and displayed composite bands; the chart below reads those rows directly.</p>
<figure class="wide"><div class="fig-frame"><div class="controls"><div class="ctl"><label for="tenorSelect">Tenor</label><select id="tenorSelect"></select></div><div class="ctl"><label>Chart language</label><div class="chiprow"><span class="chip neutral">slate = composite mid</span><span class="chip warn">amber = displayed band</span><span class="chip fail">red = Heston</span></div></div></div><canvas id="smileCanvas" height="350" aria-label="CFETS five-delta market smile and Heston fit by tenor"></canvas><div class="readout" id="smileReadout" aria-live="polite"></div></div><figcaption><span class="figno">Fig 1 · Tenor smile explorer</span>Every plotted value is embedded from the surface and Heston artifacts. The displayed band remains labelled composite; it is not treated as an executable spread.</figcaption></figure>
<details class="fallback-evidence"><summary>Fallback evidence table — all market and model nodes</summary><div class="body"><div class="tbl-wrap"><table><thead><tr><th>Tenor</th><th>Pillar</th><th class="n">Strike</th><th class="n">Bid IV</th><th class="n">Mid IV</th><th class="n">Ask IV</th><th class="n">Heston IV</th></tr></thead><tbody>{''.join(node_rows)}</tbody></table></div></div></details>
<h3>Selected tenor inputs</h3><div class="tbl-wrap"><table><thead><tr><th>Tenor</th><th>Expiry</th><th class="n">T</th><th class="n">Forward</th><th class="n">CNY rate</th><th class="n">USD rate</th><th class="n">Nodes</th></tr></thead><tbody>{''.join(tenor_table)}</tbody></table></div></section>

<section id="s3"><p class="eyebrow">§3 · Surface preparation</p><h2>Observed nodes and model-based interpolation stay separate</h2>
<div class="pipe wide"><div class="box"><b>CFETS composite</b>five delta nodes per selected tenor</div><div class="arrow">→</div><div class="box"><b>Strike reconstruction</b>artifact forward, rates, delta convention</div><div class="arrow">→</div><div class="box"><b>SABR slices</b>smooth each sparse smile</div><div class="arrow">→</div><div class="box"><b>Calendar projection</b>non-decreasing total variance</div><div class="arrow">→</div><div class="box"><b>Consumers</b>Dupire and SLV diagnostics</div></div>
<div class="metric-grid"><div class="metric-card"><span class="label">Preparation method</span><span class="value">SABR + calendar</span><span class="sub">{_escape(prep.get('method','reported by surface artifact'))}</span></div><div class="metric-card"><span class="label">Prepared grid</span><span class="value">{_escape(prep.get('grid_size', len(surface.get('strikes',[]))))} strikes</span><span class="sub">{len(surface.get('maturities',[]))} maturities</span></div><div class="metric-card"><span class="label">Raw-node smoothing RMSE</span><span class="value">{_fmt(None if _number(prep.get('raw_five_delta_sabr_rmse_iv')) is None else _number(prep.get('raw_five_delta_sabr_rmse_iv'))*100,4)} vol pts</span><span class="sub">artifact diagnostic</span></div><div class="metric-card"><span class="label">Calendar-adjusted nodes</span><span class="value">{_escape(prep.get('calendar_adjusted_nodes','—'))}</span><span class="sub">interpolation, not new liquidity</span></div></div>
<div class="note"><span class="t">Clean separation</span>Heston is fitted to the raw five-delta observations. Dupire and SLV require derivatives of a smooth surface, so their input is explicitly model-based interpolation rather than a claim that a dense grid traded.</div></section>

<section id="s4"><p class="eyebrow">§4 · Heston</p><h2>A five-parameter fit, with identification left visible</h2>
<div class="eq"><span class="lbl">risk-neutral FX Heston</span>dS/S = (r<sub>d</sub>−r<sub>f</sub>)dt + √v dW<sub>S</sub><br>dv = κ(θ−v)dt + σ√v dW<sub>v</sub>, &nbsp; d⟨W<sub>S</sub>,W<sub>v</sub>⟩ = ρdt</div>
<div class="metric-grid"><div class="metric-card"><span class="label">Free fit RMSE</span><span class="value">{_fmt(heston_rmse,4)} vol pts</span><span class="sub">raw five-delta nodes</span></div><div class="metric-card"><span class="label">Hard-Feller RMSE</span><span class="value">{_fmt(hard_heston_rmse,4)} vol pts</span><span class="sub">separate model-risk stress</span></div><div class="metric-card"><span class="label">Displayed-band coverage</span><span class="value">{_fmt(coverage,1,'%')}</span><span class="sub">composite band, not execution</span></div><div class="metric-card"><span class="label">Feller ratio</span><span class="value">{_fmt(feller,4)}</span><span class="sub">2κθ / σ²</span></div><div class="metric-card"><span class="label">Optimizer status</span><span class="value"><span class="chip {'pass' if best.get('success', True) else 'fail'}">{'SUCCESS' if best.get('success', True) else 'FAILED'}</span></span><span class="sub">{_escape(best.get('message','artifact-selected fit'))}</span></div></div>
<div class="tbl-wrap"><table><thead><tr><th>Parameter</th><th class="n">Value</th><th>Primary surface information</th></tr></thead><tbody>{param_rows}</tbody></table></div>
<div class="note warn"><span class="t">Identification test</span>Fit error alone does not qualify κ, θ, and σ. Read it together with the multi-start ranges, Jacobian conditioning, hard-Feller penalty, and cross-date chart in §6.</div></section>

<section id="s5"><p class="eyebrow">§5 · Dynamics</p><h2>Local volatility and SLV answer a different question</h2>
<p>Local volatility asks for the diffusion that reproduces the prepared vanilla surface. SLV adds stochastic variance and a leverage function. Their calibration diagnostics test the prepared surface and numerical engines; they do not create additional observed market information.</p>
<div class="metric-grid"><div class="metric-card"><span class="label">Local-vol raw in-domain RMSE</span><span class="value">{_fmt(localvol_rmse,4)} vol pts</span><span class="sub">from <code>cfets_usdcny_localvol</code></span></div><div class="metric-card"><span class="label">SLV raw in-domain RMSE</span><span class="value">{_fmt(slv_rmse,4)} vol pts</span><span class="sub">from <code>cfets_usdcny_slv</code></span></div><div class="metric-card"><span class="label">Heston observed-node RMSE</span><span class="value">{_fmt(heston_rmse,4)} vol pts</span><span class="sub">same observed five-delta language</span></div></div>
<div class="note warn"><span class="t">Interpretation</span>The Heston gate is scored directly on observed five-delta nodes. Local-vol and SLV are downstream diagnostics against a smoothed surface; their errors are not independent evidence of market depth, and the public sample does not justify promoting SLV to production calibration.</div>
<div class="note risk"><span class="t">Vanilla fit is not an exotic validation</span>A snapshot fit does not test the daily fixing, managed-band regime, CNY/CNH basis dynamics, or intervention jumps that can matter for barriers and fixing-sensitive payoffs.</div></section>

<section id="s6"><p class="eyebrow">§6 · Robustness</p><h2>Cross-date parameter evidence</h2>
<p>The diagnostics artifact contributes every included date below. {'Only one snapshot is included, so this is a display rather than a stability test.' if len(history)<2 else f'The strict comparability gate admitted {len(history)} dates; parameter movement is evidence, while date count alone is not a stability guarantee.'}</p>
<figure class="wide"><div class="fig-frame"><div class="controls"><div class="ctl"><label>Parameter or diagnostic</label><div class="chiprow">{''.join(f'<button type="button" data-stability-metric="{name}" aria-pressed="{"true" if name=="v0" else "false"}">{name}</button>' for name in (*PARAMETERS,'rmse_vol_points','feller_ratio','jacobian_condition'))}</div></div></div><canvas id="stabilityCanvas" height="330" aria-label="Cross-date Heston parameter and diagnostic chart"></canvas><div class="readout" id="stabilityReadout" aria-live="polite"></div></div><figcaption><span class="figno">Fig 2 · Parameter stability explorer</span>Dates, parameters, RMSE, Feller ratio and Jacobian condition are read from <code>diagnostics.included[].modes.free</code>. Missing metrics remain missing.</figcaption></figure>
<details class="fallback-evidence"><summary>Fallback evidence table — every included calibration date</summary><div class="body"><div class="tbl-wrap"><table><thead><tr><th>Tag</th><th>Date</th><th class="n">v0</th><th class="n">κ</th><th class="n">θ</th><th class="n">σ</th><th class="n">ρ</th><th class="n">RMSE</th><th class="n">Feller</th><th class="n">Jacobian cond.</th></tr></thead><tbody>{''.join(history_table)}</tbody></table></div></div></details>
<div class="note {'warn' if len(history)<2 else ''}"><span class="t">Evidence status</span>{'Only one date is included; parameter stability and next-day performance are not tested.' if len(history)<2 else f'{len(history)} dates are included. Interpret the chart with the artifact verdicts and exclusions; date count alone does not prove stability.'}</div></section>

<section id="s7"><p class="eyebrow">§7 · Honesty section</p><h2>Comparison and limitations, ranked openly</h2>
<div class="tbl-wrap"><table><thead><tr><th>Dimension</th><th>CFETS USD/CNY</th><th>MO index options</th></tr></thead><tbody><tr><td>Quote geometry</td><td>standardized delta smile across a broad tenor ladder</td><td>listed strike ladder across contract months</td></tr><tr><td>Transparency</td><td>public composite; executable history is licensed</td><td>exchange contract and market-data transparency</td></tr><tr><td>Main Heston question</td><td>executable depth and parameter stability</td><td>expiry/strike concentration and observed calibration stability</td></tr><tr><td>Research conclusion</td><td><span class="chip warn">CONDITIONAL YES</span></td><td>retain the separately established MO conclusion</td></tr></tbody></table></div>
<h3>Official market-development context</h3><div class="tbl-wrap"><table><thead><tr><th>Evidence</th><th>Published fact</th><th>Correct scope</th></tr></thead><tbody>{market_evidence_rows}</tbody></table></div>
<div class="note warn"><span class="t">No category error</span>Large aggregate FX and derivative turnover supports market infrastructure, but cannot substitute for USD/CNY option-specific executable depth. The calibration conclusion therefore remains stronger than the liquidity conclusion.</div>
<ol>{limitation_html}</ol>
<h3>Artifact flow</h3><div class="pipe wide"><div class="box"><b>01 snapshot</b>public CFETS curve, frozen</div><div class="arrow">→</div><div class="box"><b>02 surface</b>raw nodes + prepared grid</div><div class="arrow">→</div><div class="box"><b>03–05 models</b>LV · Heston · SLV</div><div class="arrow">→</div><div class="box"><b>06 diagnostics</b>cross-date evidence</div><div class="arrow">→</div><div class="box"><b>07 explainer</b>this derived document</div></div></section>

<section id="s8"><p class="eyebrow">§8 · Appendix</p><h2>Artifact identity and reproduction contract</h2>
<p>This document has no runtime stylesheet, font, image, script, library, or network dependency. CSS, JavaScript, evidence tables and chart data are inline. Re-running the renderer requires all five named JSON artifacts.</p>
<div class="tbl-wrap"><table><thead><tr><th>Artifact</th><th class="n">Schema</th><th>Trade date</th><th>Role</th></tr></thead><tbody>{_artifact_table(tag,artifacts)}</tbody></table></div>
<details><summary>Official CFETS source registry</summary><div class="body"><ul>{source_links_html}</ul><p>These links identify the public benchmark and convention sources. The HTML does not fetch them at runtime.</p></div></details>
<details><summary>Exact command and fail-closed behavior</summary><div class="body"><pre><code>.venv/bin/python example/fx_volmodels/07_explainer.py --tag {_escape(tag)}</code></pre><p>The command fails if any required artifact is absent, malformed, not a JSON object, lacks <code>schema_version</code>, reports a non-USD/CNY pair, or conflicts on trade date. It never downgrades missing model evidence to an empty section.</p></div></details>
<details><summary>What would be needed for production promotion</summary><div class="body"><ul><li>Licensed node-level executable bid/ask and transaction history.</li><li>Quote age, contributor/sample counts and size metadata.</li><li>Spread-relative fit and next-day holdout tests.</li><li>Multi-start/Jacobian/bootstrap identification evidence.</li><li>Stress-window and CNY/CNH basis diagnostics for the intended product.</li></ul></div></details></section>
</main>
<div class="footer">QuantArk · CFETS USD/CNY volatility-model study · artifact tag {_escape(tag)} · public composite clearly separated from executable evidence</div>
<script>window.__FX_REPORT_DATA__={report_data};</script>
<script>{JS}</script>
</body></html>
"""


def generate(data_dir: str | Path, tag: str, output: str | Path | None = None) -> Path:
    artifacts = load_artifacts(data_dir, tag)
    document = render_html(artifacts, tag)
    destination = Path(output) if output is not None else Path(data_dir) / f"fx_calibration_explainer_{tag}.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = generate(args.data_dir, args.tag, args.out)
    print(output)


if __name__ == "__main__":
    main()
