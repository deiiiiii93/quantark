"""Knock-out records on one environment instant (0.16.0 DCN/dated run, D2).

Owner ruling: the records an instant holds are tested at the same spot in contract order; the first whose barrier
is breached knocks out and pays its own payoff. A later record whose breach region lies inside an earlier one's can
never trigger first, so it is dropped; what remains is one group of strictly monotone barriers.
"""
from quantark.asset.equity.product.option.observation_schedule import (
    ResolvedObservationRecord as R,
    same_instant_ko_groups,
)

T1, T2 = 0.25, 0.5


def ids(groups):
    return [[r.payoff for r in g] for g in groups]


def test_distinct_times_give_singletons_in_time_order():
    a, b = R(T2, barrier=103.0, payoff=2.0), R(T1, barrier=103.0, payoff=1.0)
    assert ids(same_instant_ko_groups([b, a], is_reverse=False)) == [[1.0], [2.0]]


def test_a_duplicate_is_dropped_whatever_its_payoff():
    groups = same_instant_ko_groups([R(T1, barrier=103.0, payoff=1.0), R(T1, barrier=103.0, payoff=9.0)], False)
    assert ids(groups) == [[1.0]]


def test_a_step_down_is_kept_after_the_earlier_record():
    groups = same_instant_ko_groups([R(T1, barrier=103.0, payoff=1.0), R(T1, barrier=100.0, payoff=2.0)], False)
    assert ids(groups) == [[1.0, 2.0]]


def test_a_step_up_is_dropped():
    groups = same_instant_ko_groups([R(T1, barrier=100.0, payoff=1.0), R(T1, barrier=103.0, payoff=2.0)], False)
    assert ids(groups) == [[1.0]]


def test_reverse_products_mirror_the_rule():
    up_then_down = [R(T1, barrier=97.0, payoff=1.0), R(T1, barrier=100.0, payoff=2.0)]   # reverse: breach S <= b
    assert ids(same_instant_ko_groups(up_then_down, is_reverse=True)) == [[1.0, 2.0]]
    assert ids(same_instant_ko_groups(list(reversed(up_then_down)), is_reverse=True)) == [[2.0]]


def test_three_records_keep_the_strictly_decreasing_chain():
    recs = [R(T1, barrier=105.0, payoff=1.0), R(T1, barrier=106.0, payoff=2.0), R(T1, barrier=100.0, payoff=3.0),
            R(T1, barrier=100.0, payoff=4.0), R(T1, barrier=95.0, payoff=5.0)]
    assert ids(same_instant_ko_groups(recs, False)) == [[1.0, 3.0, 5.0]]


def test_records_without_a_barrier_pass_through_in_place():
    recs = [R(T1, barrier=103.0, payoff=1.0), R(T1, barrier=None, payoff=7.0), R(T1, barrier=103.0, payoff=2.0)]
    assert ids(same_instant_ko_groups(recs, False)) == [[1.0, 7.0]]


def test_output_records_are_the_input_objects():
    a = R(T1, barrier=103.0, payoff=1.0)
    assert same_instant_ko_groups([a], False)[0][0] is a


def test_same_instant_uses_the_contract_clock_rule():
    from quantark.util.numerical import same_instant

    t = 0.5
    u = t + 1e-15
    assert same_instant(t, u)
    groups = same_instant_ko_groups([R(t, barrier=103.0, payoff=1.0), R(u, barrier=100.0, payoff=2.0)], False)
    assert ids(groups) == [[1.0, 2.0]]
