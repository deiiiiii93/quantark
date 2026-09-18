"""Banking copies a validated certificate into a directory that did not exist, and nothing else."""
import hashlib

import pytest

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.banking import bank_certificate
from quantark.modelvalidation.pipeline import certify

from conftest import OffsetCandidate, CASE_MEANS_C, make_study


def _digest(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())}


def test_banks_the_three_artifacts_and_anchors_but_never_the_checkpoints(tmp_path, study):
    run = certify(study, out_dir=tmp_path / "out")
    dest = bank_certificate(run.path.parent, tmp_path / "bank", "2026-09-18", study=study)
    assert dest == tmp_path / "bank" / study.name / "2026-09-18"
    assert sorted(p.name for p in dest.iterdir()) == ["anchors.json", "certificate.json", "report.html", "report.md"]


def test_a_second_bank_on_the_same_day_takes_a_suffix_and_leaves_the_first_untouched(tmp_path, study):
    first_run = certify(study, out_dir=tmp_path / "out1")
    first = bank_certificate(first_run.path.parent, tmp_path / "bank", "2026-09-18", study=study)
    before = _digest(first)
    other = make_study(candidates=(OffsetCandidate(offset_c=0.1, means_c=CASE_MEANS_C),))
    second_run = certify(other, out_dir=tmp_path / "out2")
    second = bank_certificate(second_run.path.parent, tmp_path / "bank", "2026-09-18", study=other)
    third = bank_certificate(second_run.path.parent, tmp_path / "bank", "2026-09-18", study=other)
    assert second.name == "2026-09-18-2" and third.name == "2026-09-18-3"
    assert _digest(first) == before


def test_a_quick_run_cannot_be_banked(tmp_path, study):
    run = certify(study, out_dir=tmp_path / "out", quick=True)
    with pytest.raises(ValidationError, match="quick"):
        bank_certificate(run.path.parent, tmp_path / "bank", "2026-09-18", study=study)
    assert not (tmp_path / "bank").exists()


def test_a_tampered_certificate_is_refused(tmp_path, study):
    run = certify(study, out_dir=tmp_path / "out")
    text = run.path.read_text(encoding="utf-8").replace('"ADMITTED"', '"REJECTED"')
    run.path.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError, match="digest"):
        bank_certificate(run.path.parent, tmp_path / "bank", "2026-09-18", study=study)
