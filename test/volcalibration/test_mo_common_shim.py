"""The shim must preserve _mo_common's exact legacy surface."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "example/mo_volmodels"))
import _mo_common as mc  # noqa: E402

SNAP = ROOT / "example/mo_volmodels/data/mo_snapshot_sample.json"


def test_load_snapshot_still_returns_a_dict():
    snap = mc.load_snapshot(SNAP)
    assert isinstance(snap, dict)
    assert snap["underlying"]["spot"] == 6000.0


def test_iter_expiries_still_takes_a_dict_and_yields_legacy_slices():
    slices = mc.iter_expiries(mc.load_snapshot(SNAP))
    assert len(slices) == 4
    sl = slices[0]
    assert sl.expiry_date  # legacy attribute name preserved
    assert sl.T > 0.0
    assert len(set(sl.calls) & set(sl.puts)) >= 5


def test_parity_delegates_to_the_library():
    from quantark.volcalibration.normalize import listed

    sl = mc.iter_expiries(mc.load_snapshot(SNAP))[0]
    legacy = mc.imply_forward_and_rate(sl, 6000.0)
    lib = listed.imply_forward_and_rate(
        listed.ExpirySlice(
            sl.expiry_date, sl.expiry_date, sl.T, sl.calls, sl.puts, sl.volume
        ),
        6000.0,
    )
    assert legacy.r == lib.r
    assert legacy.forward == lib.forward
    assert legacy.discount_factor == lib.discount_factor
    assert legacy.q == lib.q
    assert legacy.n_pairs == lib.n_pairs


def test_legacy_five_field_constructor_still_works():
    sl = mc.ExpirySlice("x", 0.2, {6000.0: 10.0}, {6000.0: 9.0})
    with pytest.raises(ValueError):
        mc.imply_forward_and_rate(sl, 6000.0)


def test_otm_helpers_are_the_library_objects():
    from quantark.volcalibration.normalize import listed

    assert mc.select_otm is listed.select_otm
    assert mc.otm_implied_vol is listed.otm_implied_vol
