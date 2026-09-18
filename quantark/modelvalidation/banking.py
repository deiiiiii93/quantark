"""Banking: copy a validated certificate into the evidence bank without ever overwriting one.

A child certificate records its parent's digest, so a banked directory that is replaced
leaves an unverifiable chain. The directory is therefore created, never reused: a second
certification of the same study on the same day takes the next numeric suffix.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.amendment import validate_parent
from quantark.modelvalidation.anchors import extract_anchors
from quantark.modelvalidation.evidence import atomic_write_json
from quantark.modelvalidation.pipeline import CERTIFICATE_NAME, HTML_REPORT_NAME, REPORT_NAME
from quantark.modelvalidation.yaml_loader import load_study_text

ANCHORS_NAME = "anchors.json"


def bank_certificate(run_dir: str | Path, bank_root: str | Path, date: str, *, study=None) -> Path:
    """Bank ``run_dir`` under ``<bank_root>/<study>/<date>[-n]`` and extract its anchors.

    ``study`` is re-loaded from the certificate's own source text unless given (tests over fakes give it).

    Raises:
        ValidationError: an invalid or tampered certificate, a quick run, a missing report, or a
            certificate without study source text (anchors could not be re-run from it).
    """
    run_dir = Path(run_dir)
    payload = validate_parent(run_dir / CERTIFICATE_NAME)
    if payload["study"].get("quick"):
        raise ValidationError("a quick run is a wiring check and cannot be banked")
    source_text = payload["study"].get("source_text")
    if not source_text:
        raise ValidationError("this certificate carries no study source text, so its anchors could not be re-run")
    for name in (REPORT_NAME, HTML_REPORT_NAME):
        if not (run_dir / name).is_file():
            raise ValidationError(f"{run_dir / name} is missing; bank the run directory the certification wrote")
    anchors = extract_anchors(payload, study if study is not None else load_study_text(source_text))

    study_dir = Path(bank_root) / payload["study"]["name"]
    dest, suffix = study_dir / date, 2
    while dest.exists():
        dest, suffix = study_dir / f"{date}-{suffix}", suffix + 1
    dest.mkdir(parents=True)                    # never exist_ok: this directory must be new
    for name in (CERTIFICATE_NAME, REPORT_NAME, HTML_REPORT_NAME):
        shutil.copyfile(run_dir / name, dest / name)
    atomic_write_json(dest / ANCHORS_NAME, anchors)
    return dest
