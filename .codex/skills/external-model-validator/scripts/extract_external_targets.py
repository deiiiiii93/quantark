#!/usr/bin/env python3
"""Extract benchmark cases and engine targets from docx/txt/md files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List
from xml.etree import ElementTree as ET


ENGINE_PATTERNS: Dict[str, str] = {
    "MC": r"\bMC(?:\s*\([^)]*\))?\b",
    "QMC": r"\bQMC(?:\s*\([^)]*\))?\b",
    "RQMC": r"\bRQMC(?:\s*\([^)]*\))?\b",
    "PDE": r"\bPDE\b",
    "QUAD": r"\bQUAD(?:RATURE)?\b",
}

CASE_HINTS = (
    r"\bcase\b",
    r"\bscenario\b",
    r"\bvariant\b",
    r"\bparticipation\b",
    r"\bKI\b",
    r"\bKO\b",
    r"\b\d{1,3}\s*%\b",
    r"图表\s*\d+",
    r"敲入参与率",
)

NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
CN_TARGET_PATTERNS = (
    ("PDE", "price", r"校验模型估值"),
    ("PDE", "delta_cash", r"校验模型Delta"),
    ("PDE", "vega_1pct", r"校验模型Vega"),
    ("MC_BENCHMARK", "price", r"金融创新部估值"),
    ("MC_BENCHMARK", "delta_cash", r"金融创新部Delta"),
    ("MC_BENCHMARK", "vega_1pct", r"金融创新部Vega"),
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
    ext = path.suffix.lower()
    if ext == ".docx":
        return _read_docx(path)
    return path.read_text(encoding="utf-8")


def normalize_number(token: str) -> float:
    return float(token.replace(",", ""))


def looks_like_case_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(re.search(pattern, stripped, flags=re.IGNORECASE) for pattern in CASE_HINTS)


def infer_case_label(line: str, fallback_idx: int) -> str:
    if "图表 5" in line or ("60%" in line and "估值结果对比" in line):
        return "KI participation 60%"
    if "图表 4" in line:
        return "KI participation 100%"
    text = line.strip().strip("#-*: ")
    text = re.sub(r"\s+", " ", text)
    return text if text else f"Case {fallback_idx}"


def extract_targets(text: str) -> List[Target]:
    lines = text.splitlines()
    current_case = "Case 1"
    case_counter = 1
    targets: List[Target] = []

    for idx, line in enumerate(lines, start=1):
        if looks_like_case_header(line):
            case_counter += 1
            current_case = infer_case_label(line, case_counter)

        if "绝对差异" in line or "相对差异" in line:
            continue

        for engine, metric, cn_pattern in CN_TARGET_PATTERNS:
            if not re.search(rf"^\s*{cn_pattern}\s*", line):
                continue
            match = NUMBER_RE.search(line)
            if not match:
                continue
            value = normalize_number(match.group(0))
            targets.append(
                Target(
                    case=current_case,
                    engine=engine,
                    metric=metric,
                    value=value,
                    source_line=idx,
                    raw_line=line.strip(),
                )
            )

        for engine, engine_pattern in ENGINE_PATTERNS.items():
            if not re.search(engine_pattern, line, flags=re.IGNORECASE):
                continue
            for num_token in NUMBER_RE.findall(line):
                value = normalize_number(num_token)
                metric = "price"
                if re.search(r"\bdelta\b", line, flags=re.IGNORECASE):
                    metric = "delta_cash"
                elif re.search(r"\bvega\b", line, flags=re.IGNORECASE):
                    metric = "vega_1pct"
                targets.append(
                    Target(
                        case=current_case,
                        engine=engine,
                        metric=metric,
                        value=value,
                        source_line=idx,
                        raw_line=line.strip(),
                    )
                )

    return targets


def deduplicate(targets: List[Target]) -> List[Target]:
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
    engines = sorted({item.engine for item in targets})
    cases = sorted({item.case for item in targets})
    return {
        "input_file": str(input_path),
        "input_sha256": digest,
        "num_targets": len(targets),
        "engines_detected": engines,
        "cases_detected": cases,
        "targets": [asdict(t) for t in targets],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract external benchmark targets from docx/txt/md case files."
    )
    parser.add_argument("--input", required=True, help="Path to .docx/.txt/.md source file")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    text = read_text(input_path)
    targets = deduplicate(extract_targets(text))
    payload = build_output(input_path, text, targets)

    output_json = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json, encoding="utf-8")
        print(f"Saved extracted targets to: {output_path}")
        print(f"Targets extracted: {payload['num_targets']}")
        return

    print(output_json)


if __name__ == "__main__":
    main()
