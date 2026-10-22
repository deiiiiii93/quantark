from quantark.util.enum import GreekConvention


def test_greek_convention_members():
    names = {c.name for c in GreekConvention}
    assert names == {
        "STICKY_STRIKE", "STICKY_MONEYNESS", "STICKY_DELTA", "MODEL",
        "BARTLETT",
    }


def test_greek_convention_is_hashable_and_valued():
    assert GreekConvention.STICKY_DELTA.value == "sticky_delta"
