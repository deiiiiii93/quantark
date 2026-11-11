"""The statistics column layout (spec 3B §5.1, §5.3, §5.4; F4; unified event statistics T12)."""
import numpy as np

from quantark.asset.equity.engine.pde.core.statistics.columns import ColumnLayout


def test_snowball_order_is_ko_then_the_knock_in_settlement_unit():
    L = ColumnLayout(n_ko=3)
    assert list(L.ko) == [0, 1, 2] and L.kis == 3 and L.kie is None and L.n_cols == 4
    assert L.maturity.size == 0 and L.ki_unit == 3


def test_phoenix_order_is_ko_trigger_ki():
    L = ColumnLayout(n_ko=2, trigger=True)
    assert list(L.ko) == [0, 1] and list(L.trigger_cols) == [2, 3] and L.kis == 4


def test_no_settlement_unit_without_ki():
    L = ColumnLayout(n_ko=2, ki=False)
    assert L.kis is None and L.kie is None and L.n_cols == 2


def test_one_maturity_has_both_outcome_legs_and_the_not_knocked_in_unit():
    """T12: M0 (never knocked in), M1 (knocked in) and U0; the knocked-in unit is KIS."""
    L = ColumnLayout(n_ko=2, leg=True)
    assert list(L.maturity) == [3, 4] and list(L.maturity_unit) == [5] and L.kis == 2 and L.n_cols == 6


def test_memory_expiry_layout_counts_match_the_spec_plus_f4():
    L = ColumnLayout(n_ko=12, trigger=True, leg=True, coupon_cash=True, n_states=13, expiry=True, undiscounted=True)
    # 12 KO + 12 P + 1 KIS + 2 M + 1 U0 + 156 C + 169 D + 12 T + 1 U + 156 Ĉ
    assert L.n_cols == 522
    assert L.cash.shape == (12, 13) and L.bucket.shape == (13, 13) and L.term_unit.shape == (12,)
    assert L.cash_u.shape == (12, 13)
    assert len(set(np.r_[L.ko, L.trigger_cols, L.cash.ravel(), L.bucket.ravel(), L.term_unit, [L.kis, L.unit],
                         L.maturity, L.maturity_unit, L.cash_u.ravel()])) == L.n_cols


def test_ko_reset_layout_has_two_maturities_and_units():
    L = ColumnLayout(n_ko=6, n_post=12, ki=False, leg=True, two_maturities=True)
    assert L.maturity.size == 2 and L.maturity_unit.size == 2 and L.post_ko.size == 12
    assert L.kis is None and L.ki_unit == int(L.maturity_unit[1])
