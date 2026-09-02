import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from quantark.volcalibration.admission import AdmissionError, AdmissionReason
from quantark.volcalibration.normalize.listed import ListedNormalizer
from quantark.volcalibration.quotes import IvNode
from quantark.volcalibration.snapshot import QuoteSnapshot
from quantark.volcalibration.surface import build_raw_surface

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "example/mo_volmodels/data/mo_snapshot_sample.json"


def _quotes():
    snap = QuoteSnapshot.from_legacy_live(
        json.loads(LIVE.read_text()), trade_date=date(2026, 7, 6), symbol="000852.SH"
    )
    return ListedNormalizer().normalize(snap)


def test_grid_shape_and_schema():
    surface = build_raw_surface(_quotes())
    grid = np.asarray(surface["iv_grid"])
    assert grid.shape == (len(surface["maturities"]), len(surface["strikes"]))
    assert np.all(np.isfinite(grid)) and np.all(grid > 0.0)
    assert surface["maturities"] == sorted(surface["maturities"])
    assert surface["strikes"] == sorted(surface["strikes"])
    pe = surface["per_expiry"][0]
    assert set(pe) >= {
        "expiry_date",
        "T",
        "r",
        "q",
        "forward",
        "df",
        "points",
        "pair_count",
        "parity_rmse_points",
    }
    assert pe["points"] == sorted(pe["points"])


def test_grid_rows_interpolate_each_expiry_own_smile():
    quotes = _quotes()
    surface = build_raw_surface(quotes)
    strikes = np.asarray(surface["strikes"])
    row0 = np.asarray(surface["iv_grid"][0])
    own = {n.strike: n.iv for n in quotes.expiries[0].nodes}
    for k, v in own.items():
        if strikes.min() <= k <= strikes.max():
            assert row0[np.argmin(np.abs(strikes - k))] == pytest.approx(v, abs=1e-9)


def test_too_few_common_strikes_is_rejected():
    quotes = _quotes()
    shifted = replace(
        quotes.expiries[1],
        nodes=tuple(
            IvNode(n.strike + 3.7, n.iv, n.weight_hint)
            for n in quotes.expiries[1].nodes
        ),
    )
    broken = replace(quotes, expiries=(quotes.expiries[0], shifted))
    with pytest.raises(AdmissionError) as exc:
        build_raw_surface(broken)
    assert exc.value.reason is AdmissionReason.INSUFFICIENT_COMMON_STRIKES


def test_grid_domain_is_the_overlap_of_quoted_ranges_not_their_union():
    """A union-range grid would evaluate SABR wings the market never quoted."""
    quotes = _quotes()
    surface = build_raw_surface(quotes)
    lo = max(min(n.strike for n in e.nodes) for e in quotes.expiries)
    hi = min(max(n.strike for n in e.nodes) for e in quotes.expiries)
    assert min(surface["strikes"]) >= lo
    assert max(surface["strikes"]) <= hi
    # Trimmed wing nodes are counted for audit, never silently dropped.
    assert all("off_grid_node_count" in pe for pe in surface["per_expiry"])
