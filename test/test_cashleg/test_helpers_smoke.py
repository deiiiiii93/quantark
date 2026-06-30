"""Smoke test that the shared autocallable builders resolve and work."""

import numpy as np

from test_cashleg._autocallable_helpers import (
    make_env,
    make_snowball,
    make_phoenix,
    make_engine,
    future_event_times,
)


def test_builders_and_future_times():
    env = make_env()
    sb = make_snowball()
    make_phoenix()  # constructs without error
    assert make_engine("pde", "snowball").price(sb, env) > 0
    et = future_event_times(sb, make_engine("mc", "snowball"), env)
    assert et.ndim == 1 and et.size >= 1 and np.all(np.diff(et) > 0)
