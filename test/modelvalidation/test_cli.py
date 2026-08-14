"""Tests for the universal certification CLI."""

import textwrap

import pytest

from quantark.modelvalidation import registry
from quantark.modelvalidation.cli import main
from quantark.modelvalidation.evidence import read_json
from quantark.modelvalidation.registry import register_builder
from quantark.modelvalidation.study import HedgeContractScale

from conftest import CASE_MEANS_C, OffsetCandidate, SteadyReference

STUDY_YAML = textwrap.dedent(
    """
    study: cli-demo
    schema: 1
    quantities: [pv, delta]
    bounds: {cell: 0.5, mean_signed_bias: 0.1}
    sampling: {paths_per_batch: 512, min_batches: 2, max_batches: 4, seed: 100}
    economic_scale:
      builder: cli.scale
      params: {hedge_multiplier: 200, hedge_inception_spot: 100.0, notional: 50000000}
    environment: {builder: cli.env, params: {spot: 100.0}}
    product: {builder: cli.product, params: {strike: 100.0}}
    reference: {builder: cli.reference, params: {}}
    candidates:
      - {builder: cli.candidate, params: {}}
    cases:
      - {name: ordinary}
      - {name: near_ko}
    """
).strip()


@pytest.fixture(autouse=True)
def _builders():
    saved = dict(registry._REGISTRY)
    try:

        @register_builder("cli.scale", kind="economic_scale")
        def scale(params):
            return HedgeContractScale(**params)

        @register_builder("cli.env", kind="environment")
        def env(params):
            return dict(params)

        @register_builder("cli.product", kind="product")
        def product(params):
            return dict(params)

        @register_builder("cli.reference", kind="reference")
        def reference(environment_params, product_params, sampling, quantities, params):
            return SteadyReference(means_c=CASE_MEANS_C)

        @register_builder("cli.candidate", kind="candidate")
        def candidate(environment_params, product_params, quantities, params):
            return OffsetCandidate(name="cli.candidate", means_c=CASE_MEANS_C)

        yield
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)


@pytest.fixture
def study_file(tmp_path):
    path = tmp_path / "study.yaml"
    path.write_text(STUDY_YAML, encoding="utf-8")
    return path


def test_run_writes_a_certificate(tmp_path, study_file, capsys):
    code = main(["run", str(study_file), "--out", str(tmp_path / "out")])
    assert code == 0

    certificate = tmp_path / "out" / "cli-demo" / "certificate.json"
    assert certificate.exists()
    assert (tmp_path / "out" / "cli-demo" / "report.md").exists()

    output = capsys.readouterr().out
    assert "cli.candidate" in output
    assert "ADMITTED" in output
    assert str(certificate) in output


def test_run_quick_flag(tmp_path, study_file):
    main(["run", str(study_file), "--quick", "--out", str(tmp_path / "out")])
    payload = read_json(tmp_path / "out" / "cli-demo" / "certificate.json")
    assert payload["study"]["quick"] is True


def test_run_resume_flag(tmp_path, study_file):
    out = str(tmp_path / "out")
    main(["run", str(study_file), "--out", out])
    assert main(["run", str(study_file), "--resume", "--out", out]) == 0


def test_run_returns_zero_even_when_rejected(tmp_path, study_file, capsys):
    """A REJECTED decision is a successful run: the framework did its job."""
    biased = STUDY_YAML.replace("params: {}\n    cases:", "params: {}\n    cases:")
    path = tmp_path / "biased.yaml"
    path.write_text(biased, encoding="utf-8")

    registry._REGISTRY[("candidate", "cli.candidate")] = (
        lambda environment_params, product_params, quantities, params: OffsetCandidate(
            name="cli.candidate", offset_c=5.0, means_c=CASE_MEANS_C
        )
    )
    code = main(["run", str(path), "--out", str(tmp_path / "out")])
    assert code == 0
    assert "REJECTED" in capsys.readouterr().out


def test_run_reports_a_bad_study_file(tmp_path, capsys):
    path = tmp_path / "broken.yaml"
    path.write_text("study: broken\nschema: 2\n", encoding="utf-8")
    code = main(["run", str(path), "--out", str(tmp_path / "out")])
    assert code == 1
    assert "schema" in capsys.readouterr().err


def test_amend_requires_a_reason(tmp_path, study_file):
    out = str(tmp_path / "out")
    main(["run", str(study_file), "--out", out])
    parent = tmp_path / "out" / "cli-demo" / "certificate.json"
    with pytest.raises(SystemExit):
        main(["amend", str(study_file), "--parent", str(parent), "--out", out])


def test_amend_writes_an_amended_certificate(tmp_path, study_file, capsys):
    out = str(tmp_path / "out")
    main(["run", str(study_file), "--out", out])
    parent = tmp_path / "out" / "cli-demo" / "certificate.json"

    code = main(
        [
            "amend",
            str(study_file),
            "--parent",
            str(parent),
            "--reason",
            "no-op amendment",
            "--out",
            str(tmp_path / "amended"),
        ]
    )
    assert code == 0
    payload = read_json(tmp_path / "amended" / "cli-demo" / "certificate.json")
    assert payload["amendment"]["reason"] == "no-op amendment"
    assert "amend" in capsys.readouterr().out.lower() or payload["amendment"]


def test_anchors_writes_next_to_the_certificate(tmp_path, study_file):
    out = str(tmp_path / "out")
    main(["run", str(study_file), "--out", out])
    certificate = tmp_path / "out" / "cli-demo" / "certificate.json"

    assert main(["anchors", str(certificate)]) == 0
    anchors = read_json(certificate.parent / "anchors.json")
    assert anchors["schema"] == 1
    assert anchors["anchors"]


def test_anchors_honours_an_explicit_output(tmp_path, study_file):
    out = str(tmp_path / "out")
    main(["run", str(study_file), "--out", out])
    certificate = tmp_path / "out" / "cli-demo" / "certificate.json"
    target = tmp_path / "custom_anchors.json"

    assert main(["anchors", str(certificate), "--out", str(target)]) == 0
    assert target.exists()


def test_list_shows_registered_builders(capsys):
    assert main(["list"]) == 0
    output = capsys.readouterr().out
    assert "cli.candidate" in output
    assert "candidate" in output


def test_unknown_subcommand_exits_two():
    with pytest.raises(SystemExit) as exc:
        main(["frobnicate"])
    assert exc.value.code == 2


def test_no_subcommand_exits_two():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
