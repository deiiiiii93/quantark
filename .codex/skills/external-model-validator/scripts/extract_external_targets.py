#!/usr/bin/env python3
"""Extract external benchmark targets from .docx/.txt/.md case documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET


ENGINE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "RQMC": (
        r"\bRQMC\b",
        r"\bRANDOMI[ZS]ED\s+QMC\b",
        r"\bRANDOMI[ZS]ED\s+QUASI[-\s]?MONTE[-\s]?CARLO\b",
    ),
    "QMC": (
        r"\bQMC\b",
        r"\bQUASI[-\s]?MONTE[-\s]?CARLO\b",
        r"\bSOBOL\b",
        r"\bHALTON\b",
    ),
    "MC": (
        r"\bMC\b",
        r"\bMONTE[-\s]?CARLO\b",
        r"蒙特卡洛",
    ),
    "PDE": (
        r"\bPDE\b",
        r"\bFDM\b",
        r"\bFINITE\s+DIFFERENCE\b",
        r"有限差分",
    ),
    "QUAD": (
        r"\bQUAD(?:RATURE)?\b",
        r"\bGAUSS(?:IAN)?\s+QUADRATURE\b",
        r"数值积分",
        r"高斯积分",
    ),
    "TREE": (
        r"\bTREE\b",
        r"\bBINOMIAL\b",
        r"\bTRINOMIAL\b",
        r"树法",
        r"晶格",
    ),
    "ANALYTICAL": (
        r"\bANALYTICAL\b",
        r"\bCLOSED[-\s]?FORM\b",
        r"解析法",
        r"解析解",
    ),
    "EXTERNAL_MODEL": (
        r"校验模型",
        r"独立.*模型",
        r"第三方模型",
        r"\bEXTERNAL\s+MODEL\b",
        r"\bVALIDATION\s+MODEL\b",
    ),
    "INTERNAL_MODEL": (
        r"金融创新部",
        r"自建模型",
        r"自主开发模型",
        r"\bINTERNAL\s+MODEL\b",
        r"\bPRODUCTION\s+MODEL\b",
    ),
    "BENCHMARK": (
        r"\bBENCHMARK\b",
        r"基准模型",
        r"基准值",
        r"参考模型",
    ),
}

METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "price": (
        r"\bPRICE\b",
        r"\bPV\b",
        r"\bNPV\b",
        r"\bFAIR\s+VALUE\b",
        r"\bMODEL\s+VALUE\b",
        r"估值",
        r"价格",
        r"净现值",
    ),
    "delta_cash": (r"\bDELTA\b", r"\bDELTA\s+CASH\b", r"Delta"),
    "gamma_cash": (r"\bGAMMA\b", r"Gamma"),
    "vega_1pct": (r"\bVEGA\b", r"\bVEGA\s*\(1%?\)\b", r"\b1%\s*VEGA\b", r"Vega"),
    "theta_1d": (r"\bTHETA\b", r"Theta"),
    "rho_1pct": (r"\bRHO\b", r"Rho"),
    "std_error": (
        r"\bSTD(?:\.|_|\s)?ERR(?:OR)?\b",
        r"\bSTANDARD\s+ERROR\b",
        r"标准误",
    ),
    "probability": (r"\bPROBABILIT(?:Y|IES)\b", r"概率"),
}

CASE_HINTS = (
    r"\bcase\b",
    r"\bscenario\b",
    r"\bvariant\b",
    r"图表\s*\d+",
    r"\bchart\s*\d+\b",
    r"\bfigure\s*\d+\b",
    r"估值结果对比",
    r"参与率",
    r"\bparticipation\b",
)

SKIP_HINTS = (
    r"绝对差异",
    r"相对差异",
    r"\bdiff(?:erence)?\b",
    r"误差率",
)

NUMBER_RE = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
)


@dataclass
class Target:
    case: str
    engine: str
    metric: str
    value: float
    source_line: int
    raw_line: str


def _read_docx(path: Path) -> str:
    try:
        from docx import Document  # type: ignore

        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text)
    except Exception:
        with zipfile.ZipFile(path) as zf:
            xml_bytes = zf.read("word/document.xml")
        root = ET.fromstring(xml_bytes)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        texts = [node.text for node in root.findall(".//w:t", ns) if node.text]
        return "\n".join(texts)


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        return _read_docx(path)
    return path.read_text(encoding="utf-8")


def normalize_number(token: str) -> float:
    return float(token.replace(",", ""))


def is_skippable(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in SKIP_HINTS)


def match_alias(text: str, aliases: Dict[str, Tuple[str, ...]]) -> Optional[str]:
    for canonical, patterns in aliases.items():
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            return canonical
    return None


def split_key_value(line: str) -> Tuple[Optional[str], Optional[str]]:
    text = line.strip()
    if not text:
        return None, None
    if "\t" in text:
        key, value = text.split("\t", 1)
        return key.strip(), value.strip()
    match = re.match(r"^\s*(.{2,}?)\s*[:：=]\s*(.+?)\s*$", text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, None


def infer_case_label(line: str, fallback_idx: int) -> str:
    stripped = re.sub(r"\s+", " ", line.strip())

    cn_match = re.search(r"参与率\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*%", stripped)
    if cn_match:
        return f"Participation {cn_match.group(1)}%"

    en_match = re.search(
        r"participation\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*%", stripped, flags=re.IGNORECASE
    )
    if en_match:
        return f"Participation {en_match.group(1)}%"

    chart_match = re.search(r"(图表|chart|figure)\s*(\d+)", stripped, flags=re.IGNORECASE)
    if chart_match:
        return f"Case {chart_match.group(2)}"

    heading_match = re.search(
        r"\b(case|scenario|variant)\s*[:#-]?\s*([A-Za-z0-9_%./() \-]+)",
        stripped,
        flags=re.IGNORECASE,
    )
    if heading_match:
        suffix = heading_match.group(2).strip()
        if suffix:
            return f"{heading_match.group(1).title()} {suffix}"

    if stripped:
        return stripped[:120]
    return f"Case {fallback_idx}"


def looks_like_case_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(re.search(pattern, stripped, flags=re.IGNORECASE) for pattern in CASE_HINTS)


def extract_value(text: str) -> Optional[float]:
    match = NUMBER_RE.search(text)
    if not match:
        return None
    return normalize_number(match.group(0))


def extract_targets(text: str) -> List[Target]:
    lines = text.splitlines()
    targets: List[Target] = []
    current_case = "Case 1"
    case_counter = 1

    for idx, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        if looks_like_case_header(line):
            case_counter += 1
            current_case = infer_case_label(line, case_counter)

        if is_skippable(line):
            continue

        key, value_text = split_key_value(line)
        search_zone = key if key else line
        engine = match_alias(search_zone, ENGINE_ALIASES)
        metric = match_alias(search_zone, METRIC_ALIASES)

        if (engine is None or metric is None) and key and value_text:
            # Some formats place engine or metric in value columns.
            both = f"{key} {value_text}"
            engine = engine or match_alias(both, ENGINE_ALIASES)
            metric = metric or match_alias(both, METRIC_ALIASES)

        if engine is None or metric is None:
            continue

        value_source = value_text if value_text else line
        numeric_value = extract_value(value_source)
        if numeric_value is None:
            continue

        targets.append(
            Target(
                case=current_case,
                engine=engine,
                metric=metric,
                value=numeric_value,
                source_line=idx,
                raw_line=line,
            )
        )

    return targets


def deduplicate(targets: Iterable[Target]) -> List[Target]:
    seen = set()
    out: List[Target] = []
    for item in targets:
        key = (item.case, item.engine, item.metric, item.value, item.source_line)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def build_output(input_path: Path, text: str, targets: List[Target]) -> dict:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "input_file": str(input_path),
        "input_sha256": digest,
        "num_targets": len(targets),
        "engines_detected": sorted({item.engine for item in targets}),
        "metrics_detected": sorted({item.metric for item in targets}),
        "cases_detected": sorted({item.case for item in targets}),
        "targets": [asdict(item) for item in targets],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract external benchmark targets from .docx/.txt/.md files."
    )
    parser.add_argument("--input", required=True, help="Input file path")
    parser.add_argument("--output", help="Optional output JSON file path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    text = read_text(input_path)
    targets = deduplicate(extract_targets(text))
    payload = build_output(input_path, text, targets)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(f"Saved extracted targets to: {output_path}")
        print(f"Targets extracted: {payload['num_targets']}")
        return

    print(rendered)


if __name__ == "__main__":
    main()
