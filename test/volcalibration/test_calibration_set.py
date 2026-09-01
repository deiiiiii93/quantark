import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# replay_golden/fixtures.py already writes a minimal admitted history.
from replay_golden.fixtures import (  # noqa: E402
    DATE_A,
    DATE_EXCLUDED,
    write_localvol_history,
)

from quantark.util.exceptions import ValidationError  # noqa: E402
from quantark.volcalibration.calibration_set import CalibrationSet  # noqa: E402


@pytest.fixture
def history(tmp_path):
    return write_localvol_history(tmp_path)


def test_open_validates_and_lists_admitted_dates(history):
    cs = CalibrationSet.open(history)
    assert DATE_A in cs.dates()
    assert DATE_EXCLUDED not in cs.dates()


def test_surface_for_applies_manifest_carry_forward(history):
    cs = CalibrationSet.open(history)
    artifact = cs.surface_for(DATE_A)
    assert artifact.sha256
    # The excluded date carries the previous admitted surface forward.
    assert cs.surface_for(DATE_EXCLUDED).sha256 == artifact.sha256


def test_open_is_fail_closed_on_a_missing_manifest(tmp_path):
    with pytest.raises(ValidationError):
        CalibrationSet.open(tmp_path / "nowhere")


def test_environment_for_carries_the_parity_term_structure(history):
    cs = CalibrationSet.open(history)
    env, surface, s0 = cs.environment_for(DATE_A)
    artifact = cs.surface_for(DATE_A)
    assert s0 == pytest.approx(artifact.s0)
    assert env.valuation_date.date() == DATE_A, (
        "the environment must be dated to the trade date, not a hardcoded day"
    )
    for pillar in artifact.per_expiry:
        assert env.rate_curve.get_rate(pillar["T"]) == pytest.approx(pillar["r"])
        assert env.div_yield.get_yield(pillar["T"]) == pytest.approx(pillar["q"])
    assert list(surface.strikes) == list(artifact.strikes)
    assert env.spot_quote.spot == pytest.approx(artifact.s0)


def test_environment_for_a_carried_forward_date_keeps_that_date(history):
    """The surface carries forward; the valuation date does not."""
    cs = CalibrationSet.open(history)
    env, _surface, _s0 = cs.environment_for(DATE_EXCLUDED)
    assert env.valuation_date.date() == DATE_EXCLUDED


def test_model_for_reports_a_missing_calibration_by_name(history):
    cs = CalibrationSet.open(history)
    with pytest.raises(ValidationError) as exc:
        cs.model_for(DATE_A, "heston")
    assert str(DATE_A) in str(exc.value)
    assert "volcalibration run" in str(exc.value)


def test_status_of_a_store_that_never_ran_is_empty(history):
    assert CalibrationSet.open(history).status() == {}


def _write_calibration_record(history, tag, record):
    from quantark.volcalibration.store import StoreLayout, atomic_write_json

    layout = StoreLayout(history, history)
    atomic_write_json(
        layout.calibration_manifest,
        {"schema_version": 1, "records": [dict(record, date=tag)]},
    )


def test_a_record_for_a_different_surface_is_refused(history):
    """A record authorizes the model it describes, not whatever is on disk now."""
    from quantark.volcalibration.config import CalibrationRunConfig

    _write_calibration_record(
        history,
        DATE_A.strftime("%Y%m%d"),
        {
            "status": "ok",
            "surface_sha": "0" * 64,
            "config": CalibrationRunConfig().manifest_payload(),
            "variants": {"localvol": {"status": "ok", "record": {}}},
        },
    )
    cs = CalibrationSet.open(history)
    with pytest.raises(ValidationError) as exc:
        cs.model_for(DATE_A, "localvol")
    assert "surface in force" in str(exc.value)


def test_a_record_from_another_configuration_is_refused(history):
    from quantark.volcalibration.config import CalibrationRunConfig

    cs = CalibrationSet.open(history)
    _write_calibration_record(
        history,
        DATE_A.strftime("%Y%m%d"),
        {
            "status": "ok",
            "surface_sha": cs.surface_for(DATE_A).sha256,
            "config": {**CalibrationRunConfig().manifest_payload(), "slv_n_x": 81},
            "variants": {"localvol": {"status": "ok", "record": {}}},
        },
    )
    with pytest.raises(ValidationError) as exc:
        CalibrationSet.open(history).model_for(DATE_A, "localvol")
    assert "different configuration" in str(exc.value)


def test_a_temporally_smoothed_record_is_not_refitted(history):
    """Its EWMA reference cannot be rebuilt from the run config alone."""
    from quantark.volcalibration.config import CalibrationRunConfig

    cs = CalibrationSet.open(history)
    _write_calibration_record(
        history,
        DATE_A.strftime("%Y%m%d"),
        {
            "status": "ok",
            "surface_sha": cs.surface_for(DATE_A).sha256,
            "config": CalibrationRunConfig().manifest_payload(),
            "temporal_scheme": {"name": "daily_v0_structural_ewma", "status": "ok"},
            "variants": {"heston": {"status": "ok", "record": {}}},
        },
    )
    with pytest.raises(ValidationError) as exc:
        CalibrationSet.open(history).model_for(DATE_A, "heston")
    assert "temporal scheme" in str(exc.value)


def test_backtest_accepts_a_calibration_set_instead_of_a_directory(history):
    import pandas as pd

    from quantark.backtest.replay.market import AutocallableMarketDataSet

    cs = CalibrationSet.open(history)
    empty = pd.DataFrame({"date": [], "spot": []})
    data = AutocallableMarketDataSet(
        spot_data=empty.set_index("date"),
        vol_data=empty.rename(columns={"spot": "volatility"}).set_index("date"),
        rate_data=empty.rename(columns={"spot": "rate"}).set_index("date"),
        futures_data=pd.DataFrame(
            columns=["date", "contract", "expiry_date", "price"]
        ),
        calibration_set=cs,
    )
    # The surface channel is derived, so nothing downstream changes.
    assert data.surface_history is cs.surface_history


def test_supplying_both_channels_is_refused(history):
    import pandas as pd

    from quantark.backtest.replay.market import AutocallableMarketDataSet

    cs = CalibrationSet.open(history)
    empty = pd.DataFrame({"date": [], "spot": []})
    with pytest.raises(ValidationError):
        AutocallableMarketDataSet(
            spot_data=empty.set_index("date"),
            vol_data=empty.rename(columns={"spot": "volatility"}).set_index("date"),
            rate_data=empty.rename(columns={"spot": "rate"}).set_index("date"),
            futures_data=pd.DataFrame(
                columns=["date", "contract", "expiry_date", "price"]
            ),
            surface_history=cs.surface_history,
            calibration_set=cs,
        )
