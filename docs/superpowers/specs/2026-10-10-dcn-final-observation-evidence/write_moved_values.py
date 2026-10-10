"""Assemble moved_values.txt from the two summarize.py diffs, with the test that pins each value.

    python write_moved_values.py > moved_values.txt
"""
from __future__ import annotations

W10_TESTS = "test/test_dcn_final_observation.py::"
W10B_TESTS = "test/test_dcn_valuation_date_observation.py::"


def w10_pin(section, key):
    if section == "theta":
        return W10_TESTS + "test_penultimate_day_theta_is_settled_minus_today"
    if key.endswith(".price_detailed"):
        return W10_TESTS + "test_mc_result_fields_are_the_decisions"
    if key.endswith(".greeks"):
        return W10_TESTS + "test_pde_greeks_are_the_slope_of_the_settled_row"
    if " session[" in key:
        return W10_TESTS + "test_session_answers_natively_and_equals_direct_bitwise"
    return (W10_TESTS + "test_every_route_settles_the_final_row_at_the_hand_value, "
            "test_every_route_gives_the_same_number_bitwise")


def w10b_pin(key, new):
    ko_today = "spot=6300" in key or "spot=6000" in key
    if key.endswith(".greeks"):
        return W10B_TESTS + ("test_knock_out_today_greeks" if ko_today
                             else "test_greeks_and_spot_curve_carry_the_row_today")
    if ko_today:
        return W10B_TESTS + "test_knock_out_today_settles_at_the_knock_out_coupon"
    return W10B_TESTS + "test_pde_equals_mc_on_an_observation_date, test_coupon_today_is_no_overnight_jump"


def body(path):
    section = None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("## section "):
            section = line.split()[2].rstrip(":")
            continue
        if line.startswith("#") or not line:
            continue
        key, change = line.split(" | ", 1)
        yield section, key, change


def main():
    print("DCN final observation (W10) and valuation-date observation (W10b): every moved value, old -> new, with the")
    print("test that pins it. Measured by measure_moved.py (816 values per tree: final date, controls, penultimate")
    print("theta, observation dates); diffs by summarize.py (moved_w10_raw.txt, moved_w10b_raw.txt).")
    print()
    print("== W10: 0d1e4133 -> d37b431f. 492 of 816 moved, every one RAISES -> settled value (468 final-date values on")
    print("   8 routes + MC price_detailed + PDE greeks + sessions) or RAISES -> value (24 penultimate-day theta_1d).")
    print("   Unchanged: 108 controls (day before the final date, day before each observation date), 216 observation-")
    print("   date values (W10 changes no T > 0 value).")
    for section, key, change in body("moved_w10_raw.txt"):
        print(f"{key} | {change} | pinned by {w10_pin(section, key)}")
    print()
    print("== W10b: d37b431f -> W10b. 90 of 816 moved, all PDE routes (GBM, local-vol, Heston) on a valuation date that")
    print("   is a monthly observation (value -> value): a knock-out today now settles at the knock-out coupon; a coupon")
    print("   today is added. Unchanged: every final-date value, every control, every theta, every MC value, and the")
    print("   observation-date cells where nothing pays today (no coupon below the coupon barrier, knocked in, or")
    print("   carried below KO). Cells literally in the tests: DCN_A 2024-12-03; the other dates follow the same rule.")
    for _, key, change in body("moved_w10b_raw.txt"):
        print(f"{key} | {change} | pinned by {w10b_pin(key, change)}")


if __name__ == "__main__":
    main()
