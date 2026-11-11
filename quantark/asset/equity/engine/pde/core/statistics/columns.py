"""The statistics block's columns, per surface (spec 3B §5.1, §5.3, §5.4; plan F4; unified event statistics T12).

Order: knock-outs, the post-schedule knock-outs, the Phoenix triggers, the knock-in settlement unit ``KIS``,
then the maturity legs and units, the per-date coupon cash per memory state, the EXPIRY payment buckets per
memory state, the termination units, and the undiscounted coupon carriers per memory state.

The termination outcomes are columns of their own (unified event statistics DESIGN §4.1, §4.7): every path
ends in a knock-out row, the maturity row (never knocked out, never knocked in) or the knock-in settlement row
(knocked in, never knocked out), and each outcome's probability and cash is marched directly:

* one maturity date (Snowball, Phoenix): ``maturity = (M0, M1)`` the maturity cash of the not-knocked-in and
  the knocked-in outcome, ``maturity_unit = (U0,)`` the not-knocked-in outcome's unit, ``KIS`` the knocked-in
  one's (both paid at the determination, ``DF(t, T)``);
* two maturities (KO-reset): ``maturity = (M_pre, M_post)``, ``maturity_unit = (U_pre, U_post)``; the knocked-in
  outcome's unit is ``U_post`` (no ``KIS``).

The legacy knock-in-ever column (a knock-in counted after a knock-out) is gone: the statistics count a
knock-in before any knock-out only.
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
    ki: bool = True                # KIS (the knocked-in settlement unit; KO-reset uses U_post instead)
    leg: bool = False              # M0, M1, U0 (or M_pre, M_post, U_pre, U_post)
    two_maturities: bool = False   # KO-reset
    coupon_cash: bool = False      # C_k^(m)
    n_states: int = 1              # memory states: 2 with memory (state 0, state 0 + one unit), else 1
    expiry: bool = False           # D^(j,m), T_j, U
    undiscounted: bool = False     # Ĉ_k^(m): the coupon earned at k, valued at k (memory EXPIRY cash to pay)
    ko: np.ndarray = field(init=False, repr=False)
    post_ko: np.ndarray = field(init=False, repr=False)
    trigger_cols: np.ndarray = field(init=False, repr=False)
    kis: Optional[int] = field(init=False, default=None)
    kie: Optional[int] = field(init=False, default=None)     # always None: no knock-in-ever column
    maturity: np.ndarray = field(init=False, repr=False)
    maturity_unit: np.ndarray = field(init=False, repr=False)
    cash_cols: np.ndarray = field(init=False, repr=False)     # (n_ko, n_states)
    bucket: np.ndarray = field(init=False, repr=False)        # (n_ko + 1, n_states); row n_ko is the terminal
    term_unit: np.ndarray = field(init=False, repr=False)     # (n_ko,)
    unit: Optional[int] = field(init=False, default=None)     # U
    cash_u: np.ndarray = field(init=False, repr=False)        # (n_ko, n_states) when ``undiscounted``
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
            set_("kis", int(take(1)[0]))
        if self.leg:
            set_("maturity", take(2))
            set_("maturity_unit", take(2 if self.two_maturities else 1))
        else:
            set_("maturity", take(0)); set_("maturity_unit", take(0))
        set_("cash_cols", take(self.n_ko * self.n_states if self.coupon_cash else 0)
             .reshape(self.n_ko if self.coupon_cash else 0, self.n_states))
        nb = (self.n_ko + 1) * self.n_states if self.expiry else 0
        set_("bucket", take(nb).reshape(self.n_ko + 1 if self.expiry else 0, self.n_states))
        set_("term_unit", take(self.n_ko if self.expiry else 0))
        if self.expiry:
            set_("unit", int(take(1)[0]))
        nu = self.n_ko * self.n_states if self.undiscounted else 0
        set_("cash_u", take(nu).reshape(self.n_ko if self.undiscounted else 0, self.n_states))
        set_("n_cols", nxt)

    @property
    def cash(self) -> np.ndarray:
        """C_k^(m) column indices, shape (n_ko, n_states) (the spec's name)."""
        return self.cash_cols

    @property
    def ki_unit(self) -> Optional[int]:
        """The knocked-in settlement outcome's unit: ``KIS``, or ``U_post`` on a KO-reset."""
        if self.two_maturities:
            return int(self.maturity_unit[1])
        return self.kis
