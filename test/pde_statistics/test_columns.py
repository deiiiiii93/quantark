"""The statistics column layout (spec 3B §5.1, §5.3, §5.4; F4)."""
import numpy as np

from quantark.asset.equity.engine.pde.core.statistics.columns import ColumnLayout


def test_legacy_snowball_order_is_ko_then_ki():
    L = ColumnLayout(n_ko=3)
    assert list(L.ko) == [0, 1, 2] and (L.kis, L.kie) == (3, 4) and L.n_cols == 5
    assert L.maturity.size == 0


def test_legacy_phoenix_order_is_ko_trigger_ki():
    L = ColumnLayout(n_ko=2, trigger=True)
    assert list(L.ko) == [0, 1] and list(L.trigger_cols) == [2, 3] and (L.kis, L.kie) == (4, 5)


def test_pruned_ki_has_no_ki_columns():
    L = ColumnLayout(n_ko=2, ki=False)
    assert L.kis is None and L.kie is None and L.n_cols == 2


def test_memory_expiry_layout_counts_match_the_spec_plus_f4():
    L = ColumnLayout(n_ko=12, trigger=True, leg=True, coupon_cash=True, n_states=13, expiry=True)
    # 12 KO + 12 P + 2 KI + 1 M + 156 C + 169 D + 12 T + 1 U
    assert L.n_cols == 365
    assert L.cash.shape == (12, 13) and L.bucket.shape == (13, 13) and L.term_unit.shape == (12,)
    assert len(set(np.r_[L.ko, L.trigger_cols, L.cash.ravel(), L.bucket.ravel(), L.term_unit, [L.kis, L.kie, L.unit],
                         L.maturity])) == L.n_cols


def test_ko_reset_layout_has_two_maturities_and_units():
    L = ColumnLayout(n_ko=6, n_post=12, leg=True, two_maturities=True)
    assert L.maturity.size == 2 and L.maturity_unit.size == 2 and L.post_ko.size == 12
