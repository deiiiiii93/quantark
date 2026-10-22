"""SVIVolSurface tests (spec WP4.2)."""
import numpy as np
import pytest

from quantark.param.vol.svi import SVIVolSurface
from quantark.util.exceptions import NumericalError

from dcn_fixtures import synthetic_quote_book, synthetic_svi_slices

F0 = 5800.0


def _surface(**kw):
    return SVIVolSurface(
        synthetic_svi_slices(), forward_provider=lambda t: F0, **kw
    )


def test_slice_round_trip():
    s = _surface()
    for fit in synthetic_svi_slices():
        t = fit.expiry_t
        for y in (-0.4, -0.1, 0.0, 0.2, 0.5):
            k = F0 * np.exp(y)
            expected = np.sqrt(fit.params.total_variance(y) / t)
            assert s.get_vol(k, t) == pytest.approx(float(expected), abs=1e-10)


def test_total_variance_interpolation():
    s = _surface()
    f1, f2 = synthetic_svi_slices()
    y = 0.1
    w_expected = 0.5 * (
        f1.params.total_variance(y) + f2.params.total_variance(y)
    )
    assert float(s.total_variance(y, 0.375)) == pytest.approx(
        float(w_expected), abs=1e-12
    )


def test_forwards_frozen():
    forwards = {"f": F0}
    s = SVIVolSurface(
        synthetic_svi_slices(), forward_provider=lambda t: forwards["f"]
    )
    v0 = s.get_vol(6000.0, 0.375)
    # the provider is captured, but a STICKY surface must be built with a
    # provider over frozen data; here we prove the surface itself adds no
    # spot dependence: get_vol ignores the spot argument entirely
    assert s.get_vol(6000.0, 0.375, 9999.0) == v0


def test_calendar_violation_raises():
    f1, f2 = synthetic_svi_slices()
    # swap expiries: slice with SMALLER w gets the LATER expiry
    import dataclasses
    bad = [
        dataclasses.replace(f2, expiry_t=0.25),
        dataclasses.replace(f1, expiry_t=0.5),
    ]
    with pytest.raises(NumericalError):
        SVIVolSurface(bad, forward_provider=lambda t: F0)


def test_plugs_into_dupire():
    from quantark.param import FlatRateCurve
    from quantark.volmodels.localvol import build_dupire_local_vol

    s = _surface()
    lv = build_dupire_local_vol(
        s,
        spot=6000.0,
        rate_curve=FlatRateCurve(rate=0.0356),
        div_yield=lambda t: 0.02,
    )
    spots = np.linspace(4500.0, 7500.0, 11)
    out = np.asarray(lv.local_vol(spots, 0.3), dtype=float)
    assert np.all(np.isfinite(out)) and np.all(out > 0.0)


def test_fit_from_quotes_rmse_gate():
    quotes, valuation_date, spot, rate_curve, carry_curve = (
        synthetic_quote_book()
    )
    from quantark.param.vol.marketquotes import clean_and_imply

    cleaned = clean_and_imply(
        quotes, valuation_date, spot, rate_curve, carry_curve
    )
    # 4 quotes on one expiry is below the 5-point SVI minimum; widen the book
    from quantark.param.vol.marketquotes import OptionQuote, black_price
    extra = []
    (t,) = cleaned.slices.keys()
    df, f = cleaned.dfs[t], cleaned.forwards[t]
    for k in (5300.0, 5450.0, 5750.0, 5900.0, 6150.0, 6550.0):
        is_call = k >= f
        mid = black_price(f, k, t, 0.20, df, is_call)
        extra.append(OptionQuote(
            expiry=quotes[0].expiry, strike=k,
            call_put="C" if is_call else "P",
            bid=mid - 0.01, ask=mid + 0.01, volume=50.0, open_interest=100.0,
        ))
    cleaned = clean_and_imply(
        list(quotes) + extra, valuation_date, spot, rate_curve, carry_curve
    )
    surface = SVIVolSurface.fit_from_quotes(cleaned, carry_curve, spot)
    ivs = [
        (q.iv, surface.get_vol(q.strike, q.expiry_t))
        for q in cleaned.all_quotes
    ]
    rmse = np.sqrt(np.mean([(a - b) ** 2 for a, b in ivs]))
    assert rmse < 0.0025  # spec LV round-trip IV gate: RMSE < 0.25 vol pt
