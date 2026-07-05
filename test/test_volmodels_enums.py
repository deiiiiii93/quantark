from quantark.util.enum.engine_enums import (
    EngineType,
    HestonAnalyticalMethod,
    HestonMCScheme,
    ADIScheme,
    LeverageCalibrationMethod,
)
from quantark.util.enum import HestonAnalyticalMethod as HAM_TOPLEVEL  # re-export check


def test_method_enum_members():
    assert {m.name for m in HestonAnalyticalMethod} == {"LEWIS", "GATHERAL", "WEBER"}
    assert {m.name for m in HestonMCScheme} == {"EULER", "EULERLOG", "QUADEXP", "QUADEXP_M"}
    assert {m.name for m in ADIScheme} == {"DOUGLAS", "CRAIG_SNEYD", "MCS"}
    assert {m.name for m in LeverageCalibrationMethod} == {
        "MC_BINNING", "FORWARD_FOKKER_PLANCK", "UNCONDITIONAL_MEAN"
    }


def test_two_level_engine_pattern():
    assert EngineType.ANALYTICAL(HestonAnalyticalMethod.GATHERAL) == (
        EngineType.ANALYTICAL, HestonAnalyticalMethod.GATHERAL
    )


def test_enum_reexported_from_package():
    assert HAM_TOPLEVEL is HestonAnalyticalMethod
