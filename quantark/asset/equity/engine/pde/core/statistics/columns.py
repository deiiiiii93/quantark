"""The statistics block's columns, per surface (spec 3B §5.1, §5.3, §5.4; plan F4).

Order: the legacy columns first, in their legacy order -- knock-outs, the post-schedule knock-outs,
the Phoenix triggers, the two knock-in columns -- then the 3B columns: the maturity leg(s) and
units, the per-date coupon cash per memory state, the EXPIRY payment buckets per memory state, the
termination units. A layout with only legacy flags is the legacy block, column for column.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["ColumnLayout"]


@dataclass(frozen=True)
class ColumnLayout:
    n_ko: int                      # knock-out records (the pre schedule on a KO-reset)
    n_post: int = 0                # KO-reset post-schedule records
    trigger: bool = False          # Phoenix P_k
    ki: bool = True                # KIS, KIE
    leg: bool = False              # M (or M_pre, M_post, U_pre, U_post)
    two_maturities: bool = False   # KO-reset
    coupon_cash: bool = False      # C_k^(m)
    n_states: int = 1              # memory states K + 1
    expiry: bool = False           # D^(j,m), T_j, U
    ko: np.ndarray = field(init=False, repr=False)
    post_ko: np.ndarray = field(init=False, repr=False)
    trigger_cols: np.ndarray = field(init=False, repr=False)
    kis: Optional[int] = field(init=False, default=None)
    kie: Optional[int] = field(init=False, default=None)
    maturity: np.ndarray = field(init=False, repr=False)
    maturity_unit: np.ndarray = field(init=False, repr=False)
    cash_cols: np.ndarray = field(init=False, repr=False)     # (n_ko, n_states)
    bucket: np.ndarray = field(init=False, repr=False)        # (n_ko + 1, n_states); row n_ko is the terminal
    term_unit: np.ndarray = field(init=False, repr=False)     # (n_ko,)
    unit: Optional[int] = field(init=False, default=None)     # U
    n_cols: int = field(init=False, default=0)

    def __post_init__(self):
        nxt = 0

        def take(n):
            nonlocal nxt
            out = np.arange(nxt, nxt + n, dtype=int)
            nxt += n
            return out

        set_ = lambda k, v: object.__setattr__(self, k, v)   # noqa: E731
        set_("ko", take(self.n_ko))
        set_("post_ko", take(self.n_post))
        set_("trigger_cols", take(self.n_ko if self.trigger else 0))
        if self.ki:
            kis, kie = take(2)
            set_("kis", int(kis)); set_("kie", int(kie))
        if self.leg:
            set_("maturity", take(2 if self.two_maturities else 1))
            set_("maturity_unit", take(2 if self.two_maturities else 0))
        else:
            set_("maturity", take(0)); set_("maturity_unit", take(0))
        set_("cash_cols", take(self.n_ko * self.n_states if self.coupon_cash else 0)
             .reshape(self.n_ko if self.coupon_cash else 0, self.n_states))
        nb = (self.n_ko + 1) * self.n_states if self.expiry else 0
        set_("bucket", take(nb).reshape(self.n_ko + 1 if self.expiry else 0, self.n_states))
        set_("term_unit", take(self.n_ko if self.expiry else 0))
        if self.expiry:
            set_("unit", int(take(1)[0]))
        set_("n_cols", nxt)

    @property
    def cash(self) -> np.ndarray:
        """C_k^(m) column indices, shape (n_ko, n_states) (the spec's name)."""
        return self.cash_cols
