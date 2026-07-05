import numpy as np
import pytest

from quantark.volmodels.slv.leverage import BinMethod, bin_conditional


def _bin_conditional_reference(stock_values, variance_values, num_bins, method):
    """Verbatim snapshot of the pre-WS-B4 mask-based implementation (the equality oracle)."""
    n = stock_values.size
    order = np.argsort(stock_values)
    s_sorted = stock_values[order]
    v_sorted = variance_values[order]

    s_min, s_max = s_sorted[0], s_sorted[-1]
    boundaries = np.empty(num_bins + 1)
    boundaries[0] = s_min
    boundaries[-1] = s_max
    if method == BinMethod.EQUIDISTANT:
        for k in range(1, num_bins):
            boundaries[k] = s_min + (k / num_bins) * (s_max - s_min)
    elif method == BinMethod.EQUAL_WEIGHTED:
        for k in range(1, num_bins):
            idx = min(max(int(k * n / num_bins), 0), n - 1)
            boundaries[k] = s_sorted[idx]

    bin_means = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins, dtype=int)
    for k in range(num_bins):
        bl, br = boundaries[k], boundaries[k + 1]
        mask = (s_sorted >= bl) & (s_sorted <= br) if k == 0 else (s_sorted > bl) & (s_sorted <= br)
        idx = np.where(mask)[0]
        bin_counts[k] = idx.size
        if idx.size > 0:
            bin_means[k] = float(np.mean(v_sorted[idx]))
    global_mean = float(np.mean(v_sorted))
    for k in range(num_bins):
        if bin_counts[k] == 0:
            filled = False
            for off in range(1, num_bins):
                if k - off >= 0 and bin_counts[k - off] > 0:
                    bin_means[k] = bin_means[k - off]; filled = True; break
                if k + off < num_bins and bin_counts[k + off] > 0:
                    bin_means[k] = bin_means[k + off]; filled = True; break
            if not filled:
                bin_means[k] = global_mean
    return boundaries, bin_means


@pytest.mark.parametrize("method", [BinMethod.EQUIDISTANT, BinMethod.EQUAL_WEIGHTED])
def test_bin_conditional_exact_equality_random(method):
    rng = np.random.default_rng(3)
    for _ in range(5):
        S = rng.lognormal(4.6, 0.3, size=5000)
        v = rng.gamma(2.0, 0.02, size=5000)
        b_new, m_new = bin_conditional(S, v, 20, method)
        b_ref, m_ref = _bin_conditional_reference(S, v, 20, method)
        assert np.array_equal(b_new, b_ref)
        assert np.array_equal(m_new, m_ref)


@pytest.mark.parametrize("method", [BinMethod.EQUIDISTANT, BinMethod.EQUAL_WEIGHTED])
def test_bin_conditional_exact_equality_ties_and_empty_bins(method):
    rng = np.random.default_rng(4)
    # heavy ties: quantized spots -> many samples exactly on equal-weighted boundaries;
    # a hole in the support -> empty equidistant bins (neighbor-fill path)
    S = np.round(rng.lognormal(4.6, 0.4, size=3000), 0)
    S[S > 120] += 60.0
    v = rng.gamma(2.0, 0.02, size=3000)
    b_new, m_new = bin_conditional(S, v, 15, method)
    b_ref, m_ref = _bin_conditional_reference(S, v, 15, method)
    assert np.array_equal(b_new, b_ref)
    assert np.array_equal(m_new, m_ref)
