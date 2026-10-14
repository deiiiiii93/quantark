"""T7 (W16): antithetic MC standard error uses pair means (spec I3, contract C5 as amended by A9).

Within one batch the antithetic samples are ``[z-half; -z-half][:n]``; with ``m = ceil(n / 2)`` the pairs are
``(i, m + i)`` for ``i < n - m`` and, for odd ``n``, the unpaired ``z[m - 1]`` is left out of the standard error only.
The reported standard error is ``std(pair means, ddof=1) / sqrt(#pairs)``.  Prices stay bit-identical to commit
a354490b (pinned in ``mc_se_pins.py``); non-antithetic runs keep the iid standard error bitwise.
"""
import math
import sys

import numpy as np
import pytest

import mc_se_cases
from golden_compare import GOLDEN_ABS_TOL, GOLDEN_REL_TOL
from mc_se_pins import BEFORE
from quantark.montecarlo import antithetic_pair_means, mean_and_std_error, sample_std_error
from quantark.util.exceptions import ValidationError


def _ref_se(units):
    return float(np.std(units, ddof=1)) / math.sqrt(len(units))


def _pair_se(x):
    """Independent re-derivation of the A9 pairing (no helper)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = (n + 1) // 2
    units = 0.5 * (x[: n - m] + x[m:])
    return _ref_se(units)


# -- helper unit tests --------------------------------------------------------------------------------------------

def test_pair_means_even_layout():
    x = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 9.0])  # pairs (0,3),(1,4),(2,5)
    assert np.array_equal(antithetic_pair_means(x, 6), 0.5 * (x[:3] + x[3:]))


def test_pair_means_odd_n_drops_unpaired_center():
    x = np.array([1.0, 2.0, 3.0, 5.0, 7.0])  # n=5: m=3, pairs (0,3),(1,4); sample 2 is z[m-1], unpaired
    assert np.array_equal(antithetic_pair_means(x, 5), 0.5 * (x[:2] + x[3:]))


def test_pooled_batches_form_one_unit_set():
    a, b = np.arange(8.0), np.arange(8.0) ** 2
    units = np.concatenate([antithetic_pair_means(a, 8), antithetic_pair_means(b, 8)])
    mean, se = mean_and_std_error(units)
    assert mean == float(units.mean())
    assert se == _ref_se(units)


def test_n_paths_mismatch_rejected():
    with pytest.raises(ValidationError):
        antithetic_pair_means(np.arange(4.0), 6)


def test_sample_std_error_plain_is_bitwise_the_iid_formula():
    x = np.random.default_rng(0).standard_normal(1001)
    assert sample_std_error(x, antithetic=False) == float(x.std(ddof=1)) / math.sqrt(len(x))


# -- engines ------------------------------------------------------------------------------------------------------

def _spy_modules(monkeypatch):
    """Record every ``sample_std_error`` call made by an engine module."""
    calls = []
    real = sample_std_error

    def spy(samples, *, antithetic):
        calls.append((np.array(samples, dtype=float), bool(antithetic)))
        return real(samples, antithetic=antithetic)

    for name, mod in list(sys.modules.items()):
        if name.startswith("quantark.") and getattr(mod, "sample_std_error", None) is real:
            monkeypatch.setattr(mod, "sample_std_error", spy)
    return calls


CASE_NAMES = sorted({k[0] for k in BEFORE})
PAIRED_NAMES = [c for c in CASE_NAMES if c not in mc_se_cases.UNPAIRED and c not in mc_se_cases.BATCHED]
BATCHED_NAMES = [c for c in CASE_NAMES if c in mc_se_cases.BATCHED]


@pytest.mark.parametrize("case", PAIRED_NAMES)
@pytest.mark.parametrize("n", [2000, 2001])
def test_antithetic_se_is_pair_mean_formula(case, n, monkeypatch):
    mc_se_cases.run_case(case, True, 64)  # import the engines first so the spy can find them
    calls = _spy_modules(monkeypatch)
    price, se = mc_se_cases.run_case(case, True, n)

    assert calls, f"{case}: the engine did not go through sample_std_error"
    samples, antithetic = calls[-1]
    assert antithetic is True
    assert len(samples) == (n + n % 2 if case in mc_se_cases.EVEN_ROWS else n)
    # the price is the mean of the very samples the s.e. is computed on, up to the contract multiplier
    scale = price / float(samples.mean())
    assert se == pytest.approx(_pair_se(samples) * scale, rel=1e-9)


@pytest.mark.parametrize("case", sorted(mc_se_cases.UNPAIRED))
def test_iid_engines_keep_the_iid_se_with_antithetic_flag(case, monkeypatch):
    """The plain GBM snowball ignores ``use_antithetic`` (iid paths): it must not pair."""
    mc_se_cases.run_case(case, True, 64)
    calls = _spy_modules(monkeypatch)
    _, se = mc_se_cases.run_case(case, True, 2000)
    samples, antithetic = calls[-1]
    assert antithetic is False
    assert se == float(samples.std(ddof=1)) / math.sqrt(len(samples))


@pytest.mark.parametrize("case", BATCHED_NAMES)
@pytest.mark.parametrize("n", [2000, 2001])
def test_dcn_pairs_within_each_batch_and_pools(case, n, monkeypatch):
    """A9: pairs are per batch (batch size n, odd allowed); the pair means of all batches are one unit set."""
    import quantark.asset.equity.engine.mc.dcn_mc_engine as dcn_mod

    seen = []
    real = dcn_mod.antithetic_pair_means

    def spy(x, k):
        seen.append(np.array(x, dtype=float))
        return real(x, k)

    monkeypatch.setattr(dcn_mod, "antithetic_pair_means", spy)
    _, se = mc_se_cases.run_case(case, True, n)
    assert len(seen) == mc_se_cases.BATCHED[case]
    units = []
    for x in seen:
        assert len(x) == n
        m = (n + 1) // 2
        units.append(0.5 * (x[: n - m] + x[m:]))
    assert se == _ref_se(np.concatenate(units))


@pytest.mark.parametrize("case", CASE_NAMES)
@pytest.mark.parametrize("n", [2000, 2001])
def test_price_unchanged_antithetic(case, n):
    price, _ = mc_se_cases.run_case(case, True, n)
    # frozen on ARM64 -> golden tolerance (test/golden_compare.py)
    assert price == pytest.approx(BEFORE[(case, True, n)][0], rel=GOLDEN_REL_TOL, abs=GOLDEN_ABS_TOL)


@pytest.mark.parametrize("case", CASE_NAMES)
@pytest.mark.parametrize("n", [2000, 2001])
def test_non_antithetic_unchanged_bitwise(case, n):
    price, se = mc_se_cases.run_case(case, False, n)
    assert (price, se) == pytest.approx(BEFORE[(case, False, n)], rel=GOLDEN_REL_TOL, abs=GOLDEN_ABS_TOL)


@pytest.mark.parametrize("case", [c for c in PAIRED_NAMES + BATCHED_NAMES if c not in mc_se_cases.EVEN_ROWS])
def test_odd_n_pairs_the_same_draws_as_even_n(case):
    """n = 2001 pairs the same 1000 draws as n = 2000; the unpaired z[m-1] only leaves the s.e."""
    _, se_even = mc_se_cases.run_case(case, True, 2000)
    _, se_odd = mc_se_cases.run_case(case, True, 2001)
    if case == "american":  # LSM regression coefficients see the extra path, so the pair means differ slightly
        assert se_odd == pytest.approx(se_even, rel=0.1)
    else:
        assert se_odd == se_even


def test_vanilla_pairs_are_negatively_correlated(monkeypatch):
    """The pair layout is real: for a vanilla the partner payoff moves against the base payoff."""
    mc_se_cases.run_case("euro", True, 64)
    calls = _spy_modules(monkeypatch)
    mc_se_cases.run_case("euro", True, 2000)
    x, _ = calls[-1]
    assert np.corrcoef(x[:1000], x[1000:])[0, 1] < -0.3
