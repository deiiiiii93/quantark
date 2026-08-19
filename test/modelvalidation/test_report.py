"""Tests for the markdown certification report."""

from quantark.modelvalidation.pipeline import certify
from quantark.modelvalidation.report import render_markdown

from conftest import CASE_MEANS_C, ExplodingCandidate, OffsetCandidate, make_study


def test_report_states_the_decision(tmp_path, study):
    payload = certify(study, out_dir=tmp_path).payload
    text = render_markdown(payload)
    assert "fake.candidate" in text
    assert "ADMITTED" in text
    assert "fake-study" in text


def test_report_lists_every_cell(tmp_path, study):
    payload = certify(study, out_dir=tmp_path).payload
    text = render_markdown(payload)
    for case in ("ordinary", "near_ko"):
        assert case in text
    for quantity in ("pv", "delta", "gamma"):
        assert quantity in text
    assert text.count("PASS") >= len(payload["cells"])


def test_report_surfaces_errors_without_the_full_traceback(tmp_path):
    study = make_study(candidates=(ExplodingCandidate(name="fake.exploding"),))
    payload = certify(study, out_dir=tmp_path).payload
    text = render_markdown(payload)
    assert "Errors" in text
    assert "engine blew up" in text
    # The full traceback stays in the JSON; the report keeps one line per cell.
    assert "Traceback (most recent call last)" not in text


def test_report_shows_reference_sampling(tmp_path, study):
    payload = certify(study, out_dir=tmp_path).payload
    text = render_markdown(payload)
    assert "se_budget_met" in text or "Benchmark" in text
    assert "batches" in text.lower()


def test_report_is_stable_for_identical_payloads(tmp_path, study):
    payload = certify(study, out_dir=tmp_path).payload
    assert render_markdown(payload) == render_markdown(payload)


def test_report_does_not_mutate_the_payload(tmp_path, study):
    payload = certify(study, out_dir=tmp_path).payload
    before = dict(payload)
    render_markdown(payload)
    assert payload == before


def test_report_marks_a_quick_run_as_not_bankable(tmp_path, study):
    payload = certify(study, out_dir=tmp_path, quick=True).payload
    text = render_markdown(payload)
    assert "quick" in text.lower()


def test_report_written_next_to_the_certificate(tmp_path, study):
    certificate = certify(study, out_dir=tmp_path)
    report = certificate.path.parent / "report.md"
    assert report.exists()
    assert "fake-study" in report.read_text(encoding="utf-8")


def test_report_includes_the_digest_and_runtime(tmp_path, study):
    certificate = certify(study, out_dir=tmp_path)
    text = render_markdown(certificate.payload)
    assert certificate.payload["projected_sha256"][:12] in text
    assert certificate.payload["runtime"]["machine"] in text


def test_report_shows_aggregate_bias(tmp_path):
    study = make_study(
        candidates=(OffsetCandidate(name="fake.tilted", offset_c=0.3, means_c=CASE_MEANS_C),)
    )
    payload = certify(study, out_dir=tmp_path).payload
    text = render_markdown(payload)
    assert "Aggregate" in text
    assert "REJECTED" in text


def test_report_discloses_an_amendment(tmp_path, study):
    """The markdown report is the diff-friendly record of what was measured.

    An amendment re-prices some cells and carries the rest forward by
    reference. A report that omits that reads as though every cell was freshly
    measured -- the HTML review copy has always disclosed it, and the markdown
    one must not disagree with it.
    """
    from quantark.modelvalidation.amendment import amend

    parent = certify(study, out_dir=tmp_path / "parent")
    amended = amend(
        make_study(),
        parent=parent.path,
        out_dir=tmp_path / "amended",
        reason="grid refinement",
    )
    text = render_markdown(amended.payload)
    assert "Amendment" in text
    assert "grid refinement" in text
    assert parent.payload["projected_sha256"] in text
    assert str(len(amended.payload["amendment"]["carried_cells"])) in text


def test_report_omits_the_amendment_block_for_a_fresh_certification(tmp_path, study):
    payload = certify(study, out_dir=tmp_path).payload
    assert "Amendment" not in render_markdown(payload)
