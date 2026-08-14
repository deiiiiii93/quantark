"""Tests for the self-contained HTML certification report."""

import re

from quantark.modelvalidation.html_report import render_html
from quantark.modelvalidation.pipeline import certify

from conftest import CASE_MEANS_C, ExplodingCandidate, OffsetCandidate, make_study


def test_page_is_well_formed(tmp_path, study):
    html = render_html(certify(study, out_dir=tmp_path).payload)
    assert html.startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")
    assert html.count("<table") == html.count("</table>")
    assert html.count("<section") == html.count("</section>")


def test_page_is_self_contained(tmp_path, study):
    """No network requests: it must survive being emailed or opened offline."""
    html = render_html(certify(study, out_dir=tmp_path).payload)
    assert "<script" not in html
    assert "http://" not in html and "https://" not in html
    assert "src=" not in html
    assert "@import" not in html


def test_page_defines_both_themes(tmp_path, study):
    """A colour defined only inside a media query renders unreadable in the other theme."""
    html = render_html(certify(study, out_dir=tmp_path).payload)
    assert "prefers-color-scheme: dark" in html
    assert ':root[data-theme="dark"]' in html
    # The base palette must exist outside any conditional block.
    base = html.split("@media")[0]
    for token in ("--ink", "--paper", "--ground", "--pass", "--fail"):
        assert token in base


def test_decision_and_verdicts_are_shown(tmp_path, study):
    html = render_html(certify(study, out_dir=tmp_path).payload)
    assert "fake.candidate" in html
    assert "ADMITTED" in html
    assert "PASS" in html


def test_gauge_encodes_margin_not_just_the_verdict(tmp_path):
    """The reason this report exists: 4% of budget and 96% are different evidence."""
    tight = make_study(
        candidates=(OffsetCandidate(name="fake.tight", offset_c=0.45, means_c=CASE_MEANS_C),)
    )
    html = render_html(certify(tight, out_dir=tmp_path / "tight").payload)
    widths = [float(w) for w in re.findall(r'class="fill [a-z]+" style="width:([\d.]+)%', html)]
    assert widths, "expected margin gauges in the cells table"
    # 0.45 c of a 0.5 c bound, plus interval slack -> most of the budget consumed.
    assert max(widths) > 80.0

    loose = make_study(
        candidates=(OffsetCandidate(name="fake.loose", offset_c=0.0, means_c=CASE_MEANS_C),)
    )
    loose_html = render_html(certify(loose, out_dir=tmp_path / "loose").payload)
    loose_widths = [
        float(w) for w in re.findall(r'class="fill [a-z]+" style="width:([\d.]+)%', loose_html)
    ]
    assert max(loose_widths) < 20.0


def test_quick_mode_is_flagged_prominently(tmp_path, study):
    html = render_html(certify(study, out_dir=tmp_path, quick=True).payload)
    assert "Quick mode" in html
    assert "not bankable evidence" in html


def test_errors_are_escaped_not_injected(tmp_path):
    """A traceback is untrusted text on a page; it must never become markup."""
    study = make_study(
        candidates=(
            ExplodingCandidate(
                name="fake.exploding", exc=RuntimeError("<script>alert(1)</script>")
            ),
        )
    )
    html = render_html(certify(study, out_dir=tmp_path).payload)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_error_section_appears_only_when_needed(tmp_path, study):
    clean = render_html(certify(study, out_dir=tmp_path / "clean").payload)
    assert "<h2>Errors</h2>" not in clean

    broken = make_study(candidates=(ExplodingCandidate(name="fake.exploding"),))
    broken_html = render_html(certify(broken, out_dir=tmp_path / "broken").payload)
    assert "<h2>Errors</h2>" in broken_html
    assert "engine blew up" in broken_html


def test_amendment_block_appears_for_amendments(tmp_path, study):
    from quantark.modelvalidation.amendment import amend

    parent = certify(study, out_dir=tmp_path / "parent")
    amended = amend(
        make_study(),
        parent=parent.path,
        out_dir=tmp_path / "amended",
        reason="grid refinement",
    )
    html = render_html(amended.payload)
    assert "<h2>Amendment</h2>" in html
    assert "grid refinement" in html
    assert parent.payload["projected_sha256"] in html


def test_provenance_is_carried(tmp_path, study):
    certificate = certify(study, out_dir=tmp_path)
    html = render_html(certificate.payload)
    assert certificate.payload["projected_sha256"] in html
    assert certificate.payload["runtime"]["machine"] in html


def test_render_is_deterministic_and_non_mutating(tmp_path, study):
    payload = certify(study, out_dir=tmp_path).payload
    before = dict(payload)
    assert render_html(payload) == render_html(payload)
    assert payload == before


def test_report_written_next_to_the_certificate(tmp_path, study):
    certificate = certify(study, out_dir=tmp_path)
    report = certificate.path.parent / "report.html"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "fake-study" in text
    assert text.startswith("<!doctype html>")


def test_unresolved_cells_are_surfaced_in_the_banner(tmp_path):
    """A benchmark too noisy to decide must be visible without reading the table."""
    from quantark.modelvalidation.study import GateBounds

    study = make_study(bounds=GateBounds(cell=0.0005, mean_signed_bias=0.0001))
    html = render_html(certify(study, out_dir=tmp_path).payload)
    assert "UNRESOLVED" in html
