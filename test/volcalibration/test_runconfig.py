import pytest

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.config import (
    CalibrationRunConfig,
    RunConfig,
    SurfaceBuildConfig,
    UnderlyingConfig,
)
from quantark.volcalibration.store import builder_fingerprint
from quantark.volcalibration.yaml_loader import load_run_config_text

# Read back from the live surface manifest's "config" block on 2026-09-01.
# 787 admitted artifacts were built with exactly these values; a default
# SurfaceBuildConfig must fingerprint identically or a resumed run rebuilds
# every one of them (spec 5.3, plan DP-2).
FROZEN_CONFIG_BLOCK = {
    "artifact_schema_version": 1,
    "min_common_strikes": 3,
    "min_expiries": 2,
    "min_strikes_per_expiry": 5,
    "sabr_beta": 1.0,
}

MINIMAL_YAML = """
schema_version: 1
name: mo-daily
underlying:
  symbol: "000852.SH"
  convention: listed_strike
  price_field: settlement
paths:
  root: data/history
"""


def test_the_fingerprint_declares_the_same_artifact_schema_as_the_builder():
    from quantark.volcalibration.surface import ARTIFACT_SCHEMA_VERSION

    assert (
        SurfaceBuildConfig().fingerprint_payload()["artifact_schema_version"]
        == ARTIFACT_SCHEMA_VERSION
    )


def test_default_surface_config_fingerprints_as_the_frozen_history():
    assert SurfaceBuildConfig().fingerprint_payload() == FROZEN_CONFIG_BLOCK
    assert builder_fingerprint(
        SurfaceBuildConfig().fingerprint_payload()
    ) == builder_fingerprint(FROZEN_CONFIG_BLOCK)


def test_a_moved_knob_enters_the_fingerprint():
    moved = SurfaceBuildConfig(max_abs_implied_rate=0.05)
    payload = moved.fingerprint_payload()
    assert payload["max_abs_implied_rate"] == 0.05
    assert set(payload) - set(FROZEN_CONFIG_BLOCK) == {"max_abs_implied_rate"}
    assert builder_fingerprint(payload) != builder_fingerprint(FROZEN_CONFIG_BLOCK)


def test_calibration_manifest_payload_matches_the_live_shape():
    # Keys and content of the live calibration manifest's per-record "config"
    # block, so an existing record stays "current".
    assert CalibrationRunConfig().manifest_payload() == {
        "variants": ["localvol", "heston", "heston_slv"],
        "heston_preset": "mo_frozen",
        "heston_max_nfev": 200,
        "slv_eta": 1.0,
        "slv_n_steps": 40,
        "slv_n_x": 161,
        "slv_n_z": 81,
    }


def test_temporal_smoothing_adds_the_scheme_block():
    payload = CalibrationRunConfig(
        temporal_smoothing=True, structural_ewma_span=19
    ).manifest_payload()
    scheme = payload["temporal_scheme"]
    assert scheme["name"] == "daily_v0_structural_ewma"
    assert scheme["structural_ewma_span"] == 19
    assert scheme["structural_ewma_alpha"] == pytest.approx(2.0 / 20.0)
    assert scheme["daily_parameters"] == ["v0"]
    assert scheme["structural_parameters"] == ["kappa", "theta", "sigma", "rho"]


def test_yaml_defaults_reproduce_today(tmp_path):
    config = load_run_config_text(MINIMAL_YAML, base_dir=tmp_path)
    assert config.name == "mo-daily"
    assert config.underlying.price_field == "settlement"
    assert config.history_dir == tmp_path / "data/history"
    # runtime defaults to the surface root: one directory unless told otherwise
    assert config.runtime_dir == config.history_dir
    assert config.spot_csv is None
    assert config.surface.fingerprint_payload() == FROZEN_CONFIG_BLOCK
    assert config.workers == 1


def test_split_roots_are_expressible(tmp_path):
    text = (
        MINIMAL_YAML
        + "  runtime: ../out/mo\n"
        + "  spot_csv: data/history/csi1000_spot.csv\n"
    )
    config = load_run_config_text(text, base_dir=tmp_path)
    assert config.runtime_dir == (tmp_path / "../out/mo").resolve()
    assert config.spot_csv == tmp_path / "data/history/csi1000_spot.csv"


def test_the_run_config_echo_carries_both_blocks(tmp_path):
    config = load_run_config_text(MINIMAL_YAML, base_dir=tmp_path)
    echo = config.echo()
    assert echo["name"] == "mo-daily"
    assert echo["underlying"]["convention"] == "listed_strike"
    assert echo["surface"] == FROZEN_CONFIG_BLOCK
    assert echo["calibration"]["heston_preset"] == "mo_frozen"


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("schema_version: 2", "schema_version"),
        ("  price_field: mid\n", "price_field"),
        ("  convention: guess\n", "convention"),
        ("surprise: 1\n", "surprise"),
    ],
)
def test_the_loader_is_fail_closed(tmp_path, mutation, message):
    if mutation.startswith("schema_version"):
        text = MINIMAL_YAML.replace("schema_version: 1", mutation)
    elif mutation.startswith("surprise"):
        text = MINIMAL_YAML + mutation
    else:
        key = mutation.strip().split(":")[0]
        text = (
            "\n".join(
                line
                for line in MINIMAL_YAML.splitlines()
                if not line.strip().startswith(key)
            )
            + "\n"
            + mutation
        )
    with pytest.raises(ValidationError) as exc:
        load_run_config_text(text, base_dir=tmp_path)
    assert message in str(exc.value)


def test_a_missing_required_block_names_its_path(tmp_path):
    text = "\n".join(
        line for line in MINIMAL_YAML.splitlines() if not line.startswith("paths")
    )
    text = text.replace("  root: data/history", "")
    with pytest.raises(ValidationError) as exc:
        load_run_config_text(text, base_dir=tmp_path)
    assert "paths" in str(exc.value)


def test_an_unknown_variant_is_refused(tmp_path):
    text = MINIMAL_YAML + "calibration:\n  variants: [localvol, wishart]\n"
    with pytest.raises(ValidationError) as exc:
        load_run_config_text(text, base_dir=tmp_path)
    assert "wishart" in str(exc.value)


def test_run_config_rejects_a_non_positive_worker_count(tmp_path):
    with pytest.raises(ValidationError):
        RunConfig(
            name="x",
            underlying=UnderlyingConfig("S", "listed_strike", "settlement"),
            history_dir=tmp_path,
            runtime_dir=tmp_path,
            spot_csv=None,
            surface=SurfaceBuildConfig(),
            calibration=CalibrationRunConfig(),
            workers=0,
        )
