"""
Generate a standalone HTML demo that explains how r, q, and vol affect the
quoted KO rate for a Snowball RFQ.

Structure:
- Standard Snowball
- 103 monthly KO
- 75 daily KI
- 2Y maturity
- principal excluded PV convention
- fair KO rate solved from Snowball PV + financing-leg PV = 0.0
"""

from __future__ import annotations

import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option import (
    SnowballOption,
    create_european_ki_snowball,
    create_parachute_snowball,
    create_standard_snowball,
    create_stepdown_snowball,
)
from quantark.asset.equity.product.option.snowball_config import (
    AccrualConfig,
    BarrierConfig,
    PayoffConfig,
)
from quantark.asset.equity.product.option.snowball_helpers import generate_ko_observation_dates
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, ProtectionType


TEMPLATE_PATH = ROOT / "example" / "templates" / "snowball_rfq_ko_rate_dashboard.html"

PDE_HTML_OUTPUT_PATH = ROOT / "output" / "snowball_rfq_ko_rate_demo.html"
PDE_DATA_OUTPUT_PATH = ROOT / "output" / "snowball_rfq_ko_rate_demo_data.json"
PDE_CSV_OUTPUT_PATH = ROOT / "output" / "snowball_rfq_ko_rate_scenarios.csv"

QUAD_HTML_OUTPUT_PATH = ROOT / "output" / "snowball_rfq_ko_rate_demo_quad_1001.html"
QUAD_DATA_OUTPUT_PATH = ROOT / "output" / "snowball_rfq_ko_rate_demo_quad_1001_data.json"
QUAD_CSV_OUTPUT_PATH = ROOT / "output" / "snowball_rfq_ko_rate_scenarios_quad_1001.csv"

OUTPUT_PATH = PDE_HTML_OUTPUT_PATH
DATA_OUTPUT_PATH = PDE_DATA_OUTPUT_PATH
CSV_OUTPUT_PATH = PDE_CSV_OUTPUT_PATH


def _linspace(start: float, stop: float, num_points: int) -> list[float]:
    if num_points < 2:
        return [start]
    step = (stop - start) / (num_points - 1)
    return [round(start + i * step, 6) for i in range(num_points)]


R_GRID = _linspace(0.01, 0.05, 4)
Q_GRID = _linspace(0.05, 0.15, 4)
VOL_GRID = _linspace(0.15, 0.35, 4)
TENOR_GRID = [1.0, 2.0, 3.0]
KO_GRID = [97.5, 100.0, 102.5, 105.0]
KI_GRID = [65.0, 70.0, 75.0, 80.0]

DEFAULT_R = 0.03
DEFAULT_Q = 0.10
DEFAULT_VOL = 0.20
DEFAULT_TENOR = 2.0
PDE_GRID_SIZE = 400
PDE_TIME_STEPS = 400
DEMO_BUSINESS_DAYS_PER_YEAR = 244
R_IMPACT_BUMP = 0.01
Q_IMPACT_BUMP = 0.01
KO_RATE_BOUNDS = (0.0, 5.0)
AFFINE_KO_RATE_PAIR = (0.0, 2.0)
PREPAYMENT = 100.0
BASE_KO_BARRIER = 102.5
BASE_KI_BARRIER = 75.0
KO_BARRIER_BUMP = 0.25
KI_BARRIER_BUMP = 0.25
DEFAULT_VARIANT = "standard"
VARIANTS = {
    "standard": {
        "family": "standard",
        "label": "Standard",
        "description": "Monthly KO with daily KI.",
        "product_protection_type": "NONE",
        "interest_protection_type": "FULL",
    },
    "european_ki": {
        "family": "european_ki",
        "label": "European KI",
        "description": "Monthly KO with KI observed only at maturity.",
        "product_protection_type": "NONE",
        "interest_protection_type": "FULL",
    },
    "parachute": {
        "family": "parachute",
        "label": "Parachute",
        "description": "Monthly KO that drops to the KI barrier on the final KO observation.",
        "product_protection_type": "NONE",
        "interest_protection_type": "FULL",
    },
    "stepdown": {
        "family": "stepdown",
        "label": "Stepdown",
        "description": "Monthly KO barrier steps down from 103 toward 75 over the 2Y life.",
        "product_protection_type": "NONE",
        "interest_protection_type": "FULL",
    },
    "standard_partial": {
        "family": "standard",
        "label": "Standard Partial-Protected",
        "description": "Standard Snowball with partial protection tied to the KI level.",
        "product_protection_type": "PARTIAL",
        "interest_protection_type": "PARTIAL",
    },
    "european_ki_partial": {
        "family": "european_ki",
        "label": "European KI Partial-Protected",
        "description": "European KI Snowball with partial protection tied to the KI level.",
        "product_protection_type": "PARTIAL",
        "interest_protection_type": "PARTIAL",
    },
    "parachute_partial": {
        "family": "parachute",
        "label": "Parachute Partial-Protected",
        "description": "Parachute Snowball with partial protection tied to the KI level.",
        "product_protection_type": "PARTIAL",
        "interest_protection_type": "PARTIAL",
    },
    "stepdown_partial": {
        "family": "stepdown",
        "label": "Stepdown Partial-Protected",
        "description": "Stepdown Snowball with partial protection tied to the KI level.",
        "product_protection_type": "PARTIAL",
        "interest_protection_type": "PARTIAL",
    },
}

DEMO_VARIANT_KEYS = (
    "standard",
    "european_ki",
    "standard_partial",
    "european_ki_partial",
)
DEMO_VARIANTS = {key: VARIANTS[key] for key in DEMO_VARIANT_KEYS}

DEFAULT_HTML_I18N = {
    "en": {
        "headerOverline": "Risk analyst demo",
        "pageTitle": "Snowball RFQ KO-rate dashboard",
        "pageSubtitle": "Interactive standalone page for quote convention review, local market shocks, and structure assumptions.",
        "engineTag": "Engine",
        "generatedTag": "Generated",
        "dataNoteTag": "Payload",
        "engineLabel": "Snowball PDE surface + interpolation",
        "dataNote": "Embedded grid payload with browser-side interpolation. Use real market materials for production analysis.",
        "controlsTitle": "Pricing params",
        "controlsTag": "Interpolated point",
        "controlsTagExact": "Exact structure",
        "structureControlsTitle": "Structure terms",
        "structureControlsTag": "Variant / tenor / barriers",
        "variantLabel": "Snowball variant",
        "tenorLabel": "Tenor",
        "rLabel": "Risk-free rate r",
        "qLabel": "Dividend yield q",
        "rqLinkLabel": "q-r linkage",
        "rqLinkCopy": "Keep the current q-r spread when either slider moves",
        "volLabel": "Flat vol sigma",
        "koLabel": "KO barrier",
        "kiLabel": "KI barrier",
        "quoteTitle": "Quoted result",
        "quoteStatus": "Interpolated output",
        "quoteStatusExact": "Exact structure / interpolated pricing",
        "quoteKpiLabel": "Quoted KO rate",
        "interestKpiLabel": "Financing-leg PV",
        "protectedKpiLabel": "Protected-leg PV",
        "formulaTitle": "Quote convention",
        "formulaHtml": "Interest PV = Prepayment - PV(Protected Snowball, principal excluded, KO rate = 100% non-annualized)<br />Solve ko_rate such that<br />V_snowball_ex_principal(T, r, q, sigma; ko_rate) - Interest PV = 0.0",
        "selectedStateTitle": "Selected market point",
        "valuationBreakdownTitle": "Valuation breakdown",
        "variantRow": "Variant",
        "tenorRow": "Tenor",
        "rRow": "r",
        "qRow": "q",
        "volRow": "Vol",
        "koRow": "KO barrier",
        "kiRow": "KI barrier",
        "targetSnowballRow": "Target snowball PV",
        "protectedSnowballRow": "Protected snowball PV",
        "prepaymentRow": "Prepayment",
        "residualRow": "Residual",
        "gridModeRow": "Barrier handling",
        "shockTitle": "Local shocks",
        "shockTag": "First-order read",
        "shockRTitle": "Quote change if r moves +100bp",
        "shockQTitle": "Quote change if q moves +100bp",
        "shockVolTitle": "Quote change if vol moves +1 vol pt",
        "assumptionsTitle": "Assumptions and runtime",
        "assumptionsTag": "Model context",
        "structureTableTitle": "Structure terms",
        "runtimeTableTitle": "Runtime metadata",
        "notesTitle": "Usage notes",
        "structureSpotLabel": "Spot",
        "structureStrikeLabel": "Strike",
        "structureBaseTenorLabel": "Base maturity",
        "structureBaseKoLabel": "Base KO",
        "structureBaseKiLabel": "Base KI",
        "structureFreqLabel": "Observation schedule",
        "runtimeEngineLabel": "Engine name",
        "runtimeGridLabel": "Solver grid",
        "runtimeTimeLabel": "Time steps",
        "runtimeQuoteLabel": "Quote rule",
        "runtimeRangesLabel": "Embedded ranges",
        "runtimeDataNoteLabel": "Data note",
        "runtimeQuoteValue": "Fair KO rate solved from Snowball PV vs financing-leg PV",
        "exactNodeControlCaption": "Exact embedded node selection for the structure dimensions.",
        "quoteCaption": "Fair KO coupon required to clear the embedded financing convention at the selected point.",
        "offGridExactQuoteCaption": "Off-grid exact quad quote is not embedded in this HTML. Move to a grid node or rerun the calculation for this market point.",
        "interestCaption": "Prepayment {prepayment} minus protected-leg PV {protected_pv}.",
        "protectedCaption": "Protected structure valued with principal excluded and KO rate fixed at 100% non-annualized.",
        "noQuoteCaption": "No positive fair KO rate is available inside the embedded quote range.",
        "noDataCaption": "Value unavailable for the selected point.",
        "noCombined": "Selected point is outside the available quote surface or produces no valid combined result.",
        "offGridExactNoQuote": "The saved quad payload contains exact KO quotes only at embedded grid nodes. This off-grid point requires a fresh solve; direct quote interpolation is disabled.",
        "rqLinked": "Linked",
        "rqIndependent": "Independent",
        "rqHolding": "Holding q-r spread = {spread}",
        "rqIndependentCaption": "Move r and q independently.",
        "gridModeExact": "Exact KO/KI cube",
        "gridModeApprox": "Base cube plus KO/KI sensitivities",
        "shockUnavailable": "Shock output unavailable at this point.",
        "shockFlat": "Locally flat around the selected point.",
        "highRUp": "Higher r lifts the required quote in this slice.",
        "highRDown": "Higher r eases the required quote in this slice.",
        "highQUp": "Higher q increases the quote because forward carry softens.",
        "highQDown": "Higher q reduces the quote in this slice.",
        "highVolUp": "Higher vol increases the quote as downside KI risk dominates.",
        "highVolDown": "Higher vol reduces the quote in this slice.",
        "summary": "{variant} at {tenor}, r={r}, q={q}, vol={vol}, KO={ko}, KI={ki}: fair KO rate {quote}, financing-leg PV {interest_pv}, protected-leg PV {protected_pv}.",
        "variantNames": {
            "standard": "Standard",
            "european_ki": "European KI",
            "standard_partial": "Standard Partial-Protected",
            "european_ki_partial": "European KI Partial-Protected",
        },
        "variantDescriptions": {
            "standard": "Monthly KO with daily KI.",
            "european_ki": "Monthly KO with KI observed at maturity only.",
            "standard_partial": "Standard Snowball with partial protection linked to KI.",
            "european_ki_partial": "European KI Snowball with partial protection linked to KI.",
        },
        "notes": [
            "Daily KI is modeled with the demo's 244-business-day convention and scales with tenor.",
            "The financing leg is valued as the selected protected Snowball with principal excluded and KO fixed at 100% non-annualized.",
            "The page is for workflow review and intuition building, not production RFQ approval.",
            "Use real term sheets, market data snapshots, and validation reports alongside this demo."
        ],
    },
    "cn": {
        "headerOverline": "风控分析演示",
        "pageTitle": "雪球 RFQ KO 票息看板",
        "pageSubtitle": "独立交互页面，用于查看报价口径、局部市场冲击和结构假设。",
        "engineTag": "引擎",
        "generatedTag": "生成时间",
        "dataNoteTag": "载荷",
        "engineLabel": "雪球 PDE 曲面 + 浏览器插值",
        "dataNote": "页面内嵌网格载荷并在浏览器端插值。生产分析请结合正式市场材料。",
        "controlsTitle": "定价参数",
        "controlsTag": "插值点位",
        "controlsTagExact": "精确结构节点",
        "structureControlsTitle": "结构条款",
        "structureControlsTag": "变体 / 期限 / 障碍",
        "variantLabel": "雪球变体",
        "tenorLabel": "期限",
        "rLabel": "无风险利率 r",
        "qLabel": "分红率 q",
        "rqLinkLabel": "q-r 联动",
        "rqLinkCopy": "移动任一滑块时保持当前 q-r 利差",
        "volLabel": "平坦波动率 sigma",
        "koLabel": "KO 障碍",
        "kiLabel": "KI 障碍",
        "quoteTitle": "报价结果",
        "quoteStatus": "插值输出",
        "quoteStatusExact": "精确结构 / 定价参数插值",
        "quoteKpiLabel": "KO 报价票息",
        "interestKpiLabel": "融资腿 PV",
        "protectedKpiLabel": "保本腿 PV",
        "formulaTitle": "报价口径",
        "formulaHtml": "Interest PV = Prepayment - PV(Protected Snowball, principal excluded, KO rate = 100% non-annualized)<br />求解 ko_rate，使得<br />V_snowball_ex_principal(T, r, q, sigma; ko_rate) - Interest PV = 0.0",
        "selectedStateTitle": "当前市场点",
        "valuationBreakdownTitle": "估值拆解",
        "variantRow": "变体",
        "tenorRow": "期限",
        "rRow": "r",
        "qRow": "q",
        "volRow": "波动率",
        "koRow": "KO 障碍",
        "kiRow": "KI 障碍",
        "targetSnowballRow": "目标雪球 PV",
        "protectedSnowballRow": "保本雪球 PV",
        "prepaymentRow": "预付金",
        "residualRow": "残差",
        "gridModeRow": "障碍处理",
        "shockTitle": "局部冲击",
        "shockTag": "一阶观察",
        "shockRTitle": "r 上移 100bp 时的报价变化",
        "shockQTitle": "q 上移 100bp 时的报价变化",
        "shockVolTitle": "vol 上移 1 vol 点时的报价变化",
        "assumptionsTitle": "假设与运行信息",
        "assumptionsTag": "模型上下文",
        "structureTableTitle": "结构条款",
        "runtimeTableTitle": "运行元数据",
        "notesTitle": "使用说明",
        "structureSpotLabel": "现价",
        "structureStrikeLabel": "行权价",
        "structureBaseTenorLabel": "基础期限",
        "structureBaseKoLabel": "基础 KO",
        "structureBaseKiLabel": "基础 KI",
        "structureFreqLabel": "观察频率",
        "runtimeEngineLabel": "引擎名称",
        "runtimeGridLabel": "求解网格",
        "runtimeTimeLabel": "时间步数",
        "runtimeQuoteLabel": "报价规则",
        "runtimeRangesLabel": "内嵌区间",
        "runtimeDataNoteLabel": "数据说明",
        "runtimeQuoteValue": "通过雪球 PV 与融资腿 PV 匹配求解公平 KO 票息",
        "exactNodeControlCaption": "结构维度只允许选择内嵌的精确网格节点。",
        "quoteCaption": "在当前点位下，使内嵌融资口径平衡所需的公平 KO 票息。",
        "offGridExactQuoteCaption": "当前 HTML 未内嵌该离网点位的精确 Quad KO 报价。请移动到网格节点，或按该市场点重新计算。",
        "interestCaption": "预付金 {prepayment} 减去保本腿 PV {protected_pv}。",
        "protectedCaption": "保本腿按去本金、KO=100% 非年化口径估值。",
        "noQuoteCaption": "当前内嵌报价区间内没有正的公平 KO 票息。",
        "noDataCaption": "当前点位无可用数值。",
        "noCombined": "当前点位超出可用报价曲面，或无法得到有效组合结果。",
        "offGridExactNoQuote": "保存的 Quad 载荷只在内嵌网格节点提供精确 KO 报价。当前离网点位需要重新求解，已禁用直接插值报价。",
        "rqLinked": "联动",
        "rqIndependent": "独立",
        "rqHolding": "保持 q-r 利差 = {spread}",
        "rqIndependentCaption": "r 与 q 独立变动。",
        "gridModeExact": "精确 KO/KI 立方体",
        "gridModeApprox": "基础立方体 + KO/KI 敏感度",
        "shockUnavailable": "该点位无法提供冲击输出。",
        "shockFlat": "该点位附近基本平坦。",
        "highRUp": "在这一切片下，更高 r 会抬升所需报价。",
        "highRDown": "在这一切片下，更高 r 会降低所需报价。",
        "highQUp": "更高 q 会削弱远期漂移，因此报价上升。",
        "highQDown": "在这一切片下，更高 q 会压低报价。",
        "highVolUp": "更高波动放大下敲风险，因此报价上升。",
        "highVolDown": "在这一切片下，更高波动会压低报价。",
        "summary": "{variant}，期限 {tenor}，r={r}，q={q}，vol={vol}，KO={ko}，KI={ki}：公平 KO 票息 {quote}，融资腿 PV {interest_pv}，保本腿 PV {protected_pv}。",
        "variantNames": {
            "standard": "标准型",
            "european_ki": "欧式 KI",
            "standard_partial": "标准部分保本",
            "european_ki_partial": "欧式 KI 部分保本",
        },
        "variantDescriptions": {
            "standard": "月度敲出，日度敲入。",
            "european_ki": "月度敲出，仅到期观察 KI。",
            "standard_partial": "与 KI 绑定部分保本的标准雪球。",
            "european_ki_partial": "与 KI 绑定部分保本的欧式 KI 雪球。",
        },
        "notes": [
            "日度 KI 使用演示中的每年 244 个交易日口径，并随期限缩放。",
            "融资腿按所选保本雪球估值，去本金，KO 固定为 100% 非年化。",
            "本页面用于工作流复核和定价直觉，不直接用于生产 RFQ 审批。",
            "正式分析仍需结合真实条款、市场快照和验证报告。"
        ],
    },
}

LEGACY_UI_COPY_TO_I18N = {
    "eyebrow_en": ("en", "headerOverline"),
    "chip_engine_en": ("en", "engineLabel"),
    "cube_note_en": ("en", "dataNote"),
    "eyebrow_cn": ("cn", "headerOverline"),
    "chip_engine_cn": ("cn", "engineLabel"),
    "cube_note_cn": ("cn", "dataNote"),
}


@dataclass(frozen=True)
class DemoMeta:
    generated_at: str
    engine: str
    solver_grid_size: int | None
    solver_time_steps: int | None
    structure: dict[str, Any]
    ranges: dict[str, list[float]]


@dataclass(frozen=True)
class ExactBarrierTask:
    scenario_id: int
    variant: str
    tenor_idx: int
    rate_idx: int
    q_idx: int
    vol_idx: int
    ko_idx: int
    ki_idx: int
    tenor: float
    rate: float
    div_yield: float
    vol: float
    ko_barrier: float
    ki_barrier: float


@dataclass(frozen=True)
class AnchorTask:
    scenario_id: int
    variant: str
    tenor_idx: int
    rate_idx: int
    q_idx: int
    vol_idx: int
    tenor: float
    rate: float
    div_yield: float
    vol: float


DEFAULT_HTML_UI_COPY = {
    "eyebrow_en": "PDE-backed RFQ explainer",
    "chip_engine_en": "Snowball PDE surface + interpolation",
    "cube_note_en": "The HTML embeds a coarse PDE-solved cube and interpolates between nodes in-browser.",
    "eyebrow_cn": "PDE 驱动 RFQ 解释器",
    "chip_engine_cn": "雪球 PDE 曲面 + 插值",
    "cube_note_cn": "页面内嵌较粗 PDE 曲面，并在浏览器端做插值。",
}


def build_product(ko_rate: float, variant: str) -> SnowballOption:
    return build_product_with_barriers(
        ko_rate=ko_rate,
        variant=variant,
        maturity=DEFAULT_TENOR,
        ko_barrier=BASE_KO_BARRIER,
        ki_barrier=BASE_KI_BARRIER,
    )


def get_variant_config(variant: str) -> dict[str, str]:
    try:
        return VARIANTS[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown variant: {variant}") from exc


def get_partial_protection_rate(ki_barrier: float) -> float:
    return max(0.0, min(1.0, 1.0 - ki_barrier / 100.0))


def get_num_monthly_observations(maturity: float) -> int:
    return max(1, int(round(maturity * 12)))


def get_num_daily_ki_observations(maturity: float) -> int:
    return max(1, int(round(maturity * DEMO_BUSINESS_DAYS_PER_YEAR)))


def generate_daily_ki_observation_dates(maturity: float) -> list[float]:
    """Generate evenly spaced KI dates using the demo's 244 business-day convention."""
    num_observations = get_num_daily_ki_observations(maturity)
    return [(i + 1) / num_observations * maturity for i in range(num_observations)]


def build_product_with_barriers(
    ko_rate: float,
    variant: str,
    *,
    maturity: float,
    ko_barrier: float,
    ki_barrier: float,
) -> SnowballOption:
    variant_config = get_variant_config(variant)
    product_protection = ProtectionType[variant_config["product_protection_type"]]
    protection_rate = (
        get_partial_protection_rate(ki_barrier)
        if product_protection == ProtectionType.PARTIAL
        else 0.0
    )
    num_observations = get_num_monthly_observations(maturity)
    common = {
        "initial_price": 100.0,
        "strike": 100.0,
        "maturity": maturity,
        "contract_multiplier": 1.0,
        "ko_rate": ko_rate,
        "ki_barrier": ki_barrier,
        "is_reverse": False,
        "rebate_rate": ko_rate,
        "include_principal": False,
        "protection_type": product_protection,
        "protection_rate": protection_rate,
    }
    stepdown_rate = (ko_barrier - ki_barrier) / (max(num_observations - 1, 1) * 100.0)
    daily_ki_dates = generate_daily_ki_observation_dates(maturity)

    if variant_config["family"] == "standard":
        return create_standard_snowball(
            **common,
            ko_barrier=ko_barrier,
            num_observations=num_observations,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=daily_ki_dates,
        )
    if variant_config["family"] == "european_ki":
        return create_european_ki_snowball(
            **common,
            ko_barrier=ko_barrier,
            num_ko_observations=num_observations,
        )
    if variant_config["family"] == "parachute":
        return create_parachute_snowball(
            **common,
            ko_barrier=ko_barrier,
            num_observations=num_observations,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=daily_ki_dates,
        )
    if variant_config["family"] == "stepdown":
        return create_stepdown_snowball(
            **common,
            num_observations=num_observations,
            initial_ko_barrier=ko_barrier,
            stepdown_rate=stepdown_rate,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=daily_ki_dates,
        )

    raise ValueError(f"Unknown variant family for {variant}")


def build_protected_product(variant: str):
    return build_protected_product_with_barriers(
        variant=variant,
        maturity=DEFAULT_TENOR,
        ko_barrier=BASE_KO_BARRIER,
        ki_barrier=BASE_KI_BARRIER,
    )


def build_protected_product_with_barriers(
    variant: str,
    *,
    maturity: float,
    ko_barrier: float,
    ki_barrier: float,
):
    variant_config = get_variant_config(variant)
    interest_protection = ProtectionType[variant_config["interest_protection_type"]]
    protection_rate = (
        get_partial_protection_rate(ki_barrier)
        if interest_protection == ProtectionType.PARTIAL
        else 0.0
    )
    num_observations = get_num_monthly_observations(maturity)
    if variant_config["family"] == "parachute":
        ko_barrier_value = [ko_barrier] * (num_observations - 1) + [ki_barrier]
    elif variant_config["family"] == "stepdown":
        stepdown_amount = (ko_barrier - ki_barrier) / max(num_observations - 1, 1)
        ko_barrier_value = [ko_barrier - i * stepdown_amount for i in range(num_observations)]
    else:
        ko_barrier_value = ko_barrier

    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=ko_barrier_value,
            ko_rate=1.0,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=generate_ko_observation_dates(maturity, "monthly"),
            ki_barrier=None,
        ),
        payoff_config=PayoffConfig(
            rebate_rate=1.0,
            include_principal=False,
            protection_type=interest_protection,
            protection_rate=protection_rate,
        ),
        accrual_config=AccrualConfig(is_annualized=False),
        maturity=maturity,
        contract_multiplier=1.0,
        is_reverse=False,
    )


def build_env(rate: float, div_yield: float, vol: float) -> PricingEnvironment:
    return PricingEnvironment(
        valuation_date=datetime(2024, 1, 1),
        spot_quote=SpotQuote(spot=100.0, asset_name="Snowball Demo"),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div_yield),
    )


def solve_fair_ko_rate_with_engine(
    engine: Any,
    *,
    rate: float,
    div_yield: float,
    vol: float,
    tenor: float,
    variant: str,
    ko_barrier: float = BASE_KO_BARRIER,
    ki_barrier: float = BASE_KI_BARRIER,
) -> dict[str, float]:
    """Solve the fair KO rate using the supplied pricing engine."""
    env = build_env(rate=rate, div_yield=div_yield, vol=vol)
    protected_pv = engine.price(
        build_protected_product_with_barriers(
            variant=variant,
            maturity=tenor,
            ko_barrier=ko_barrier,
            ki_barrier=ki_barrier,
        ),
        env,
    )
    interest_component_pv = PREPAYMENT - protected_pv
    target_snowball_pv = interest_component_pv
    k0, k1 = AFFINE_KO_RATE_PAIR
    p0 = engine.price(
        build_product_with_barriers(
            ko_rate=k0,
            variant=variant,
            maturity=tenor,
            ko_barrier=ko_barrier,
            ki_barrier=ki_barrier,
        ),
        env,
    )
    p1 = engine.price(
        build_product_with_barriers(
            ko_rate=k1,
            variant=variant,
            maturity=tenor,
            ko_barrier=ko_barrier,
            ki_barrier=ki_barrier,
        ),
        env,
    )
    slope = (p1 - p0) / (k1 - k0)
    if abs(slope) < 1e-12:
        raise ValueError("KO rate slope is numerically flat")
    fair_ko_rate = k0 + (target_snowball_pv - p0) / slope
    if not (KO_RATE_BOUNDS[0] <= fair_ko_rate <= KO_RATE_BOUNDS[1]):
        raise ValueError("Fair KO rate falls outside configured display bounds")
    combined_pv = target_snowball_pv - interest_component_pv
    return {
        "quoted_ko_rate": fair_ko_rate,
        "snowball_target_pv": target_snowball_pv,
        "interest_component_pv": interest_component_pv,
        "protected_snowball_pv": protected_pv,
        "combined_pv": combined_pv,
    }


def solve_fair_ko_rate(
    rate: float,
    div_yield: float,
    vol: float,
    tenor: float,
    variant: str,
    *,
    ko_barrier: float = BASE_KO_BARRIER,
    ki_barrier: float = BASE_KI_BARRIER,
    pde_params: PDEParams,
) -> dict[str, float]:
    engine = SnowballPDESolver(params=pde_params)
    return solve_fair_ko_rate_with_engine(
        engine,
        rate=rate,
        div_yield=div_yield,
        vol=vol,
        tenor=tenor,
        variant=variant,
        ko_barrier=ko_barrier,
        ki_barrier=ki_barrier,
    )


def serialize_engine(engine: Any) -> dict[str, Any]:
    """Serialize supported engine settings for worker-process construction."""
    if isinstance(engine, SnowballPDESolver):
        return {
            "engine_type": "pde",
            "params": {
                "grid_size": engine.params.grid_size,
                "time_steps": engine.params.time_steps,
            },
        }
    if type(engine).__name__ == "SnowballQuadEngine":
        return {
            "engine_type": "quad",
            "params": {
                "grid_points": engine.params.grid_points,
            },
        }
    raise ValueError(f"Unsupported engine for parallel exact-grid build: {type(engine).__name__}")


def build_engine_from_config(engine_config: dict[str, Any]) -> Any:
    """Construct a pricing engine from its serialized config."""
    engine_type = engine_config["engine_type"]
    params = engine_config["params"]
    if engine_type == "pde":
        return SnowballPDESolver(
            params=PDEParams(
                grid_size=params["grid_size"],
                time_steps=params["time_steps"],
            )
        )
    if engine_type == "quad":
        from asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
        from asset.equity.param import QuadParams

        return SnowballQuadEngine(
            params=QuadParams(
                grid_points=params["grid_points"],
            )
        )
    raise ValueError(f"Unknown engine_type: {engine_type}")


_EXACT_BARRIER_WORKER_ENGINE: Any | None = None
_ANCHOR_WORKER_ENGINE: Any | None = None
_ANCHOR_WORKER_BUMP_ENGINE: Any | None = None


def init_exact_barrier_worker(engine_config: dict[str, Any]) -> None:
    """Initialize one pricing engine per worker process."""
    global _EXACT_BARRIER_WORKER_ENGINE
    _EXACT_BARRIER_WORKER_ENGINE = build_engine_from_config(engine_config)


def init_anchor_worker(
    engine_config: dict[str, Any],
    bump_engine_config: dict[str, Any],
) -> None:
    """Initialize anchor pricing engines per worker process."""
    global _ANCHOR_WORKER_ENGINE, _ANCHOR_WORKER_BUMP_ENGINE
    _ANCHOR_WORKER_ENGINE = build_engine_from_config(engine_config)
    _ANCHOR_WORKER_BUMP_ENGINE = build_engine_from_config(bump_engine_config)


def solve_exact_barrier_task(task: ExactBarrierTask) -> tuple[ExactBarrierTask, dict[str, float] | None]:
    """Solve one exact KO/KI scenario inside a worker process."""
    if _EXACT_BARRIER_WORKER_ENGINE is None:
        raise RuntimeError("Exact barrier worker engine is not initialized.")
    try:
        result = solve_fair_ko_rate_with_engine(
            _EXACT_BARRIER_WORKER_ENGINE,
            rate=task.rate,
            div_yield=task.div_yield,
            vol=task.vol,
            tenor=task.tenor,
            variant=task.variant,
            ko_barrier=task.ko_barrier,
            ki_barrier=task.ki_barrier,
        )
    except Exception:
        result = None
    return (task, result)


def solve_anchor_task(
    task: AnchorTask,
) -> tuple[
    AnchorTask,
    dict[str, float] | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    """Solve one 4D anchor scenario and its KO/KI sensitivities in a worker process."""
    if _ANCHOR_WORKER_ENGINE is None or _ANCHOR_WORKER_BUMP_ENGINE is None:
        raise RuntimeError("Anchor worker engines are not initialized.")

    try:
        result = solve_fair_ko_rate_with_engine(
            _ANCHOR_WORKER_ENGINE,
            rate=task.rate,
            div_yield=task.div_yield,
            vol=task.vol,
            tenor=task.tenor,
            variant=task.variant,
            ko_barrier=BASE_KO_BARRIER,
            ki_barrier=BASE_KI_BARRIER,
        )
    except Exception:
        return (task, None, None, None, None, None)

    quote_ko_sensitivity: float | None = None
    quote_ki_sensitivity: float | None = None
    interest_ko_sensitivity: float | None = None
    interest_ki_sensitivity: float | None = None

    try:
        ko_up = solve_fair_ko_rate_with_engine(
            _ANCHOR_WORKER_BUMP_ENGINE,
            rate=task.rate,
            div_yield=task.div_yield,
            vol=task.vol,
            tenor=task.tenor,
            variant=task.variant,
            ko_barrier=BASE_KO_BARRIER + KO_BARRIER_BUMP,
            ki_barrier=BASE_KI_BARRIER,
        )
        quote_ko_sensitivity = (ko_up["quoted_ko_rate"] - result["quoted_ko_rate"]) / KO_BARRIER_BUMP
        interest_ko_sensitivity = (
            (ko_up["interest_component_pv"] - result["interest_component_pv"]) / KO_BARRIER_BUMP
        )
    except Exception:
        pass

    try:
        ki_up = solve_fair_ko_rate_with_engine(
            _ANCHOR_WORKER_BUMP_ENGINE,
            rate=task.rate,
            div_yield=task.div_yield,
            vol=task.vol,
            tenor=task.tenor,
            variant=task.variant,
            ko_barrier=BASE_KO_BARRIER,
            ki_barrier=BASE_KI_BARRIER + KI_BARRIER_BUMP,
        )
        quote_ki_sensitivity = (ki_up["quoted_ko_rate"] - result["quoted_ko_rate"]) / KI_BARRIER_BUMP
        interest_ki_sensitivity = (
            (ki_up["interest_component_pv"] - result["interest_component_pv"]) / KI_BARRIER_BUMP
        )
    except Exception:
        pass

    return (
        task,
        result,
        quote_ko_sensitivity,
        quote_ki_sensitivity,
        interest_ko_sensitivity,
        interest_ki_sensitivity,
    )


def apply_barrier_adjustment(
    base_value: float | None,
    ko_sensitivity: float | None,
    ki_sensitivity: float | None,
    ko_barrier: float,
    ki_barrier: float,
) -> float | None:
    """Apply first-order KO/KI barrier adjustment around the base anchor."""
    if base_value is None:
        return None
    adjusted = base_value
    if ko_sensitivity is not None:
        adjusted += ko_sensitivity * (ko_barrier - BASE_KO_BARRIER)
    if ki_sensitivity is not None:
        adjusted += ki_sensitivity * (ki_barrier - BASE_KI_BARRIER)
    return adjusted


def expand_scenario_rows_with_barriers(
    anchor_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand base-barrier rows onto an explicit KO/KI CSV export grid."""
    rows: list[dict[str, Any]] = []
    scenario_id = 0
    for row in anchor_rows:
        for ko_barrier in KO_GRID:
            for ki_barrier in KI_GRID:
                scenario_id += 1
                expanded = dict(row)
                expanded["scenario_id"] = scenario_id
                expanded["ko_barrier"] = ko_barrier
                expanded["ki_barrier"] = ki_barrier
                if row["product_protection_type"] == "PARTIAL":
                    expanded["product_protection_rate"] = get_partial_protection_rate(ki_barrier)
                if row["interest_protection_type"] == "PARTIAL":
                    expanded["interest_protection_rate"] = get_partial_protection_rate(ki_barrier)
                expanded["quoted_ko_rate"] = apply_barrier_adjustment(
                    row["quoted_ko_rate"],
                    row["quote_ko_sensitivity"],
                    row["quote_ki_sensitivity"],
                    ko_barrier,
                    ki_barrier,
                )
                expanded["interest_pv"] = apply_barrier_adjustment(
                    row["interest_pv"],
                    row["interest_ko_sensitivity"],
                    row["interest_ki_sensitivity"],
                    ko_barrier,
                    ki_barrier,
                )
                if expanded["interest_pv"] is not None:
                    expanded["combined_pv"] = 0.0
                    expanded["product_pv"] = expanded["interest_pv"]
                else:
                    expanded["combined_pv"] = None
                    expanded["product_pv"] = None
                # Keep the protected-leg anchor explicit so downstream users know what changed exactly.
                expanded["protected_snowball_pv"] = row["protected_snowball_pv"]
                rows.append(expanded)
    return rows


def build_scenario_row(
    *,
    scenario_id: int,
    variant: str,
    tenor: float,
    rate: float,
    div_yield: float,
    vol: float,
    ko_barrier: float,
    ki_barrier: float,
    variant_config: dict[str, str],
    result: dict[str, float] | None,
    quote_ko_sensitivity: float | None = None,
    quote_ki_sensitivity: float | None = None,
    interest_ko_sensitivity: float | None = None,
    interest_ki_sensitivity: float | None = None,
) -> dict[str, Any]:
    """Build one exported scenario row."""
    return {
        "scenario_id": scenario_id,
        "variant": variant,
        "tenor": tenor,
        "r": rate,
        "q": div_yield,
        "vol": vol,
        "ko_barrier": ko_barrier,
        "ki_barrier": ki_barrier,
        "product_protection_type": variant_config["product_protection_type"],
        "product_protection_rate": (
            get_partial_protection_rate(ki_barrier)
            if variant_config["product_protection_type"] == "PARTIAL"
            else 0.0
        ),
        "interest_protection_type": variant_config["interest_protection_type"],
        "interest_protection_rate": (
            get_partial_protection_rate(ki_barrier)
            if variant_config["interest_protection_type"] == "PARTIAL"
            else 0.0
        ),
        "quoted_ko_rate": None if result is None else result["quoted_ko_rate"],
        "product_pv": None if result is None else result["snowball_target_pv"],
        "interest_pv": None if result is None else result["interest_component_pv"],
        "combined_pv": None if result is None else result["combined_pv"],
        "protected_snowball_pv": None if result is None else result["protected_snowball_pv"],
        "quote_ko_sensitivity": quote_ko_sensitivity,
        "quote_ki_sensitivity": quote_ki_sensitivity,
        "interest_ko_sensitivity": interest_ko_sensitivity,
        "interest_ki_sensitivity": interest_ki_sensitivity,
    }


def build_exact_barrier_tasks() -> list[ExactBarrierTask]:
    """Enumerate exact KO/KI scenario tasks in deterministic export order."""
    tasks: list[ExactBarrierTask] = []
    scenario_id = 0
    for variant in DEMO_VARIANTS:
        for tenor_idx, tenor in enumerate(TENOR_GRID):
            for rate_idx, rate in enumerate(R_GRID):
                for q_idx, div_yield in enumerate(Q_GRID):
                    for vol_idx, vol in enumerate(VOL_GRID):
                        for ko_idx, ko_barrier in enumerate(KO_GRID):
                            for ki_idx, ki_barrier in enumerate(KI_GRID):
                                scenario_id += 1
                                tasks.append(
                                    ExactBarrierTask(
                                        scenario_id=scenario_id,
                                        variant=variant,
                                        tenor_idx=tenor_idx,
                                        rate_idx=rate_idx,
                                        q_idx=q_idx,
                                        vol_idx=vol_idx,
                                        ko_idx=ko_idx,
                                        ki_idx=ki_idx,
                                        tenor=tenor,
                                        rate=rate,
                                        div_yield=div_yield,
                                        vol=vol,
                                        ko_barrier=ko_barrier,
                                        ki_barrier=ki_barrier,
                                    )
                                )
    return tasks


def build_anchor_tasks() -> list[AnchorTask]:
    """Enumerate 4D anchor scenarios in deterministic export order."""
    tasks: list[AnchorTask] = []
    scenario_id = 0
    for variant in DEMO_VARIANTS:
        for tenor_idx, tenor in enumerate(TENOR_GRID):
            for rate_idx, rate in enumerate(R_GRID):
                for q_idx, div_yield in enumerate(Q_GRID):
                    for vol_idx, vol in enumerate(VOL_GRID):
                        scenario_id += 1
                        tasks.append(
                            AnchorTask(
                                scenario_id=scenario_id,
                                variant=variant,
                                tenor_idx=tenor_idx,
                                rate_idx=rate_idx,
                                q_idx=q_idx,
                                vol_idx=vol_idx,
                                tenor=tenor,
                                rate=rate,
                                div_yield=div_yield,
                                vol=vol,
                            )
                        )
    return tasks


def make_nested_none(dimensions: list[int]) -> Any:
    """Create a nested list filled with None for the requested dimensions."""
    if len(dimensions) == 1:
        return [None] * dimensions[0]
    return [make_nested_none(dimensions[1:]) for _ in range(dimensions[0])]


def build_exact_barrier_cube_parallel(
    *,
    engine: Any,
    progress_label: str,
    parallel_workers: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the exact 6D KO/KI cube using all available worker processes."""
    worker_count = parallel_workers or max(1, os.cpu_count() or 1)
    tasks = build_exact_barrier_tasks()
    total = len(tasks)
    engine_config = serialize_engine(engine)
    variant_cubes: dict[str, Any] = {}
    dimensions = [
        len(TENOR_GRID),
        len(R_GRID),
        len(Q_GRID),
        len(VOL_GRID),
        len(KO_GRID),
        len(KI_GRID),
    ]
    for variant in DEMO_VARIANTS:
        variant_cubes[variant] = {
            "quote": make_nested_none(dimensions),
            "interest": make_nested_none(dimensions),
            "protected": make_nested_none(dimensions),
            "snowballTarget": make_nested_none(dimensions),
        }

    rows: list[dict[str, Any] | None] = [None] * total
    done = 0
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=init_exact_barrier_worker,
        initargs=(engine_config,),
    ) as executor:
        futures = [executor.submit(solve_exact_barrier_task, task) for task in tasks]
        for future in as_completed(futures):
            task, result = future.result()
            done += 1
            print(
                f"[{done:05d}/{total}] {task.variant} fair ko_rate via {progress_label} for "
                f"T={task.tenor:.2f}, r={task.rate:.4f}, q={task.div_yield:.4f}, vol={task.vol:.4f}, "
                f"ko={task.ko_barrier:.1f}, ki={task.ki_barrier:.1f}"
            )
            variant_cube = variant_cubes[task.variant]
            idx = (
                task.tenor_idx,
                task.rate_idx,
                task.q_idx,
                task.vol_idx,
                task.ko_idx,
                task.ki_idx,
            )
            if result is not None:
                variant_cube["quote"][idx[0]][idx[1]][idx[2]][idx[3]][idx[4]][idx[5]] = result["quoted_ko_rate"]
                variant_cube["interest"][idx[0]][idx[1]][idx[2]][idx[3]][idx[4]][idx[5]] = result["interest_component_pv"]
                variant_cube["protected"][idx[0]][idx[1]][idx[2]][idx[3]][idx[4]][idx[5]] = result["protected_snowball_pv"]
                variant_cube["snowballTarget"][idx[0]][idx[1]][idx[2]][idx[3]][idx[4]][idx[5]] = result["snowball_target_pv"]

            rows[task.scenario_id - 1] = build_scenario_row(
                scenario_id=task.scenario_id,
                variant=task.variant,
                tenor=task.tenor,
                rate=task.rate,
                div_yield=task.div_yield,
                vol=task.vol,
                ko_barrier=task.ko_barrier,
                ki_barrier=task.ki_barrier,
                variant_config=get_variant_config(task.variant),
                result=result,
            )

    return (variant_cubes, [row for row in rows if row is not None])


def build_anchor_cube_parallel(
    *,
    engine: Any,
    bump_engine: Any,
    progress_label: str,
    parallel_workers: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the 4D anchor cube using all available worker processes."""
    worker_count = parallel_workers or max(1, os.cpu_count() or 1)
    tasks = build_anchor_tasks()
    total = len(tasks)
    engine_config = serialize_engine(engine)
    bump_engine_config = serialize_engine(bump_engine)
    dimensions = [len(TENOR_GRID), len(R_GRID), len(Q_GRID), len(VOL_GRID)]
    variant_cubes: dict[str, Any] = {}
    for variant in DEMO_VARIANTS:
        variant_cubes[variant] = {
            "quote": make_nested_none(dimensions),
            "interest": make_nested_none(dimensions),
            "protected": make_nested_none(dimensions),
            "snowballTarget": make_nested_none(dimensions),
            "quoteKoSens": make_nested_none(dimensions),
            "quoteKiSens": make_nested_none(dimensions),
            "interestKoSens": make_nested_none(dimensions),
            "interestKiSens": make_nested_none(dimensions),
        }

    rows: list[dict[str, Any] | None] = [None] * total
    done = 0
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=init_anchor_worker,
        initargs=(engine_config, bump_engine_config),
    ) as executor:
        futures = [executor.submit(solve_anchor_task, task) for task in tasks]
        for future in as_completed(futures):
            (
                task,
                result,
                quote_ko_sensitivity,
                quote_ki_sensitivity,
                interest_ko_sensitivity,
                interest_ki_sensitivity,
            ) = future.result()
            done += 1
            print(
                f"[{done:04d}/{total}] {task.variant} fair ko_rate via {progress_label} for "
                f"T={task.tenor:.2f}, r={task.rate:.4f}, q={task.div_yield:.4f}, vol={task.vol:.4f}"
            )
            variant_cube = variant_cubes[task.variant]
            idx = (task.tenor_idx, task.rate_idx, task.q_idx, task.vol_idx)
            if result is not None:
                variant_cube["quote"][idx[0]][idx[1]][idx[2]][idx[3]] = result["quoted_ko_rate"]
                variant_cube["interest"][idx[0]][idx[1]][idx[2]][idx[3]] = result["interest_component_pv"]
                variant_cube["protected"][idx[0]][idx[1]][idx[2]][idx[3]] = result["protected_snowball_pv"]
                variant_cube["snowballTarget"][idx[0]][idx[1]][idx[2]][idx[3]] = result["snowball_target_pv"]
                variant_cube["quoteKoSens"][idx[0]][idx[1]][idx[2]][idx[3]] = quote_ko_sensitivity
                variant_cube["quoteKiSens"][idx[0]][idx[1]][idx[2]][idx[3]] = quote_ki_sensitivity
                variant_cube["interestKoSens"][idx[0]][idx[1]][idx[2]][idx[3]] = interest_ko_sensitivity
                variant_cube["interestKiSens"][idx[0]][idx[1]][idx[2]][idx[3]] = interest_ki_sensitivity

            rows[task.scenario_id - 1] = build_scenario_row(
                scenario_id=task.scenario_id,
                variant=task.variant,
                tenor=task.tenor,
                rate=task.rate,
                div_yield=task.div_yield,
                vol=task.vol,
                ko_barrier=BASE_KO_BARRIER,
                ki_barrier=BASE_KI_BARRIER,
                variant_config=get_variant_config(task.variant),
                result=result,
                quote_ko_sensitivity=quote_ko_sensitivity,
                quote_ki_sensitivity=quote_ki_sensitivity,
                interest_ko_sensitivity=interest_ko_sensitivity,
                interest_ki_sensitivity=interest_ki_sensitivity,
            )

    return (variant_cubes, expand_scenario_rows_with_barriers([row for row in rows if row is not None]))


def build_cube_with_engines(
    *,
    engine: Any,
    bump_engine: Any,
    progress_label: str | None = None,
    exact_barrier_grid: bool = False,
    parallel_workers: int | None = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    variant_cubes: dict[str, Any] = {}
    anchor_rows: list[dict[str, Any]] = []
    engine_label = progress_label or type(engine).__name__
    if exact_barrier_grid and (parallel_workers or 1) > 1:
        return build_exact_barrier_cube_parallel(
            engine=engine,
            progress_label=engine_label,
            parallel_workers=parallel_workers,
        )
    if not exact_barrier_grid and (parallel_workers or 1) > 1:
        return build_anchor_cube_parallel(
            engine=engine,
            bump_engine=bump_engine,
            progress_label=engine_label,
            parallel_workers=parallel_workers,
        )
    barrier_points = len(KO_GRID) * len(KI_GRID) if exact_barrier_grid else 1
    total = (
        len(DEMO_VARIANTS)
        * len(TENOR_GRID)
        * len(R_GRID)
        * len(Q_GRID)
        * len(VOL_GRID)
        * barrier_points
    )
    done = 0
    for variant in DEMO_VARIANTS:
        variant_config = get_variant_config(variant)
        quote_cube: list[Any] = []
        interest_cube: list[Any] = []
        protected_cube: list[Any] = []
        target_cube: list[Any] = []
        quote_ko_sens_cube: list[Any] = []
        quote_ki_sens_cube: list[Any] = []
        interest_ko_sens_cube: list[Any] = []
        interest_ki_sens_cube: list[Any] = []
        for tenor in TENOR_GRID:
            quote_r_slice: list[Any] = []
            interest_r_slice: list[Any] = []
            protected_r_slice: list[Any] = []
            target_r_slice: list[Any] = []
            quote_ko_sens_r_slice: list[Any] = []
            quote_ki_sens_r_slice: list[Any] = []
            interest_ko_sens_r_slice: list[Any] = []
            interest_ki_sens_r_slice: list[Any] = []
            for rate in R_GRID:
                quote_q_slice: list[Any] = []
                interest_q_slice: list[Any] = []
                protected_q_slice: list[Any] = []
                target_q_slice: list[Any] = []
                quote_ko_sens_q_slice: list[Any] = []
                quote_ki_sens_q_slice: list[Any] = []
                interest_ko_sens_q_slice: list[Any] = []
                interest_ki_sens_q_slice: list[Any] = []
                for div_yield in Q_GRID:
                    quote_vol_slice: list[Any] = []
                    interest_vol_slice: list[Any] = []
                    protected_vol_slice: list[Any] = []
                    target_vol_slice: list[Any] = []
                    quote_ko_sens_vol_slice: list[Any] = []
                    quote_ki_sens_vol_slice: list[Any] = []
                    interest_ko_sens_vol_slice: list[Any] = []
                    interest_ki_sens_vol_slice: list[Any] = []
                    for vol in VOL_GRID:
                        if exact_barrier_grid:
                            quote_ko_slice: list[list[float | None]] = []
                            interest_ko_slice: list[list[float | None]] = []
                            protected_ko_slice: list[list[float | None]] = []
                            target_ko_slice: list[list[float | None]] = []
                            for ko_barrier in KO_GRID:
                                quote_ki_slice: list[float | None] = []
                                interest_ki_slice: list[float | None] = []
                                protected_ki_slice: list[float | None] = []
                                target_ki_slice: list[float | None] = []
                                for ki_barrier in KI_GRID:
                                    done += 1
                                    print(
                                        f"[{done:03d}/{total}] {variant} fair ko_rate via {engine_label} for "
                                        f"T={tenor:.2f}, r={rate:.4f}, q={div_yield:.4f}, vol={vol:.4f}, "
                                        f"ko={ko_barrier:.1f}, ki={ki_barrier:.1f}"
                                    )
                                    try:
                                        result = solve_fair_ko_rate_with_engine(
                                            engine,
                                            rate=rate,
                                            div_yield=div_yield,
                                            vol=vol,
                                            tenor=tenor,
                                            variant=variant,
                                            ko_barrier=ko_barrier,
                                            ki_barrier=ki_barrier,
                                        )
                                        quote_ki_slice.append(result["quoted_ko_rate"])
                                        interest_ki_slice.append(result["interest_component_pv"])
                                        protected_ki_slice.append(result["protected_snowball_pv"])
                                        target_ki_slice.append(result["snowball_target_pv"])
                                        anchor_rows.append(
                                            build_scenario_row(
                                                scenario_id=done,
                                                variant=variant,
                                                tenor=tenor,
                                                rate=rate,
                                                div_yield=div_yield,
                                                vol=vol,
                                                ko_barrier=ko_barrier,
                                                ki_barrier=ki_barrier,
                                                variant_config=variant_config,
                                                result=result,
                                            )
                                        )
                                    except Exception:
                                        quote_ki_slice.append(None)
                                        interest_ki_slice.append(None)
                                        protected_ki_slice.append(None)
                                        target_ki_slice.append(None)
                                        anchor_rows.append(
                                            build_scenario_row(
                                                scenario_id=done,
                                                variant=variant,
                                                tenor=tenor,
                                                rate=rate,
                                                div_yield=div_yield,
                                                vol=vol,
                                                ko_barrier=ko_barrier,
                                                ki_barrier=ki_barrier,
                                                variant_config=variant_config,
                                                result=None,
                                            )
                                        )
                                quote_ko_slice.append(quote_ki_slice)
                                interest_ko_slice.append(interest_ki_slice)
                                protected_ko_slice.append(protected_ki_slice)
                                target_ko_slice.append(target_ki_slice)
                            quote_vol_slice.append(quote_ko_slice)
                            interest_vol_slice.append(interest_ko_slice)
                            protected_vol_slice.append(protected_ko_slice)
                            target_vol_slice.append(target_ko_slice)
                        else:
                            done += 1
                            print(
                                f"[{done:03d}/{total}] {variant} fair ko_rate via {engine_label} for "
                                f"T={tenor:.2f}, r={rate:.4f}, q={div_yield:.4f}, vol={vol:.4f}"
                            )
                            try:
                                result = solve_fair_ko_rate_with_engine(
                                    engine,
                                    rate=rate,
                                    div_yield=div_yield,
                                    vol=vol,
                                    tenor=tenor,
                                    variant=variant,
                                    ko_barrier=BASE_KO_BARRIER,
                                    ki_barrier=BASE_KI_BARRIER,
                                )
                                quote_vol_slice.append(result["quoted_ko_rate"])
                                interest_vol_slice.append(result["interest_component_pv"])
                                protected_vol_slice.append(result["protected_snowball_pv"])
                                target_vol_slice.append(result["snowball_target_pv"])
                                try:
                                    ko_up = solve_fair_ko_rate_with_engine(
                                        bump_engine,
                                        rate=rate,
                                        div_yield=div_yield,
                                        vol=vol,
                                        tenor=tenor,
                                        variant=variant,
                                        ko_barrier=BASE_KO_BARRIER + KO_BARRIER_BUMP,
                                        ki_barrier=BASE_KI_BARRIER,
                                    )
                                    quote_ko_sens_vol_slice.append(
                                        (ko_up["quoted_ko_rate"] - result["quoted_ko_rate"])
                                        / KO_BARRIER_BUMP
                                    )
                                    interest_ko_sens_vol_slice.append(
                                        (ko_up["interest_component_pv"] - result["interest_component_pv"])
                                        / KO_BARRIER_BUMP
                                    )
                                except Exception:
                                    quote_ko_sens_vol_slice.append(None)
                                    interest_ko_sens_vol_slice.append(None)

                                try:
                                    ki_up = solve_fair_ko_rate_with_engine(
                                        bump_engine,
                                        rate=rate,
                                        div_yield=div_yield,
                                        vol=vol,
                                        tenor=tenor,
                                        variant=variant,
                                        ko_barrier=BASE_KO_BARRIER,
                                        ki_barrier=BASE_KI_BARRIER + KI_BARRIER_BUMP,
                                    )
                                    quote_ki_sens_vol_slice.append(
                                        (ki_up["quoted_ko_rate"] - result["quoted_ko_rate"])
                                        / KI_BARRIER_BUMP
                                    )
                                    interest_ki_sens_vol_slice.append(
                                        (ki_up["interest_component_pv"] - result["interest_component_pv"])
                                        / KI_BARRIER_BUMP
                                    )
                                except Exception:
                                    quote_ki_sens_vol_slice.append(None)
                                    interest_ki_sens_vol_slice.append(None)
                                anchor_rows.append(
                                    build_scenario_row(
                                        scenario_id=done,
                                        variant=variant,
                                        tenor=tenor,
                                        rate=rate,
                                        div_yield=div_yield,
                                        vol=vol,
                                        ko_barrier=BASE_KO_BARRIER,
                                        ki_barrier=BASE_KI_BARRIER,
                                        variant_config=variant_config,
                                        result=result,
                                        quote_ko_sensitivity=quote_ko_sens_vol_slice[-1],
                                        quote_ki_sensitivity=quote_ki_sens_vol_slice[-1],
                                        interest_ko_sensitivity=interest_ko_sens_vol_slice[-1],
                                        interest_ki_sensitivity=interest_ki_sens_vol_slice[-1],
                                    )
                                )
                            except Exception:
                                quote_vol_slice.append(None)
                                interest_vol_slice.append(None)
                                protected_vol_slice.append(None)
                                target_vol_slice.append(None)
                                quote_ko_sens_vol_slice.append(None)
                                quote_ki_sens_vol_slice.append(None)
                                interest_ko_sens_vol_slice.append(None)
                                interest_ki_sens_vol_slice.append(None)
                                anchor_rows.append(
                                    build_scenario_row(
                                        scenario_id=done,
                                        variant=variant,
                                        tenor=tenor,
                                        rate=rate,
                                        div_yield=div_yield,
                                        vol=vol,
                                        ko_barrier=BASE_KO_BARRIER,
                                        ki_barrier=BASE_KI_BARRIER,
                                        variant_config=variant_config,
                                        result=None,
                                    )
                                )
                    quote_q_slice.append(quote_vol_slice)
                    interest_q_slice.append(interest_vol_slice)
                    protected_q_slice.append(protected_vol_slice)
                    target_q_slice.append(target_vol_slice)
                    if not exact_barrier_grid:
                        quote_ko_sens_q_slice.append(quote_ko_sens_vol_slice)
                        quote_ki_sens_q_slice.append(quote_ki_sens_vol_slice)
                        interest_ko_sens_q_slice.append(interest_ko_sens_vol_slice)
                        interest_ki_sens_q_slice.append(interest_ki_sens_vol_slice)
                quote_r_slice.append(quote_q_slice)
                interest_r_slice.append(interest_q_slice)
                protected_r_slice.append(protected_q_slice)
                target_r_slice.append(target_q_slice)
                if not exact_barrier_grid:
                    quote_ko_sens_r_slice.append(quote_ko_sens_q_slice)
                    quote_ki_sens_r_slice.append(quote_ki_sens_q_slice)
                    interest_ko_sens_r_slice.append(interest_ko_sens_q_slice)
                    interest_ki_sens_r_slice.append(interest_ki_sens_q_slice)
            quote_cube.append(quote_r_slice)
            interest_cube.append(interest_r_slice)
            protected_cube.append(protected_r_slice)
            target_cube.append(target_r_slice)
            if not exact_barrier_grid:
                quote_ko_sens_cube.append(quote_ko_sens_r_slice)
                quote_ki_sens_cube.append(quote_ki_sens_r_slice)
                interest_ko_sens_cube.append(interest_ko_sens_r_slice)
                interest_ki_sens_cube.append(interest_ki_sens_r_slice)
        variant_cube = {
            "quote": quote_cube,
            "interest": interest_cube,
            "protected": protected_cube,
            "snowballTarget": target_cube,
        }
        if not exact_barrier_grid:
            variant_cube.update(
                {
                    "quoteKoSens": quote_ko_sens_cube,
                    "quoteKiSens": quote_ki_sens_cube,
                    "interestKoSens": interest_ko_sens_cube,
                    "interestKiSens": interest_ki_sens_cube,
                }
            )
        variant_cubes[variant] = variant_cube
    rows = anchor_rows if exact_barrier_grid else expand_scenario_rows_with_barriers(anchor_rows)
    return (variant_cubes, rows)


def build_cube(
    *, pde_params: PDEParams
) -> tuple[dict[str, dict[str, list[list[list[list[float | None]]]]]], list[dict[str, Any]]]:
    """Build the demo cube with the legacy PDE engine configuration."""
    bump_params = PDEParams(
        grid_size=max(50, pde_params.grid_size // 2),
        time_steps=max(80, pde_params.time_steps // 2),
    )
    return build_cube_with_engines(
        engine=SnowballPDESolver(params=pde_params),
        bump_engine=SnowballPDESolver(params=bump_params),
        progress_label="SnowballPDESolver",
    )


def write_scenario_csv(
    rows: list[dict[str, Any]],
    csv_output_path: Path = CSV_OUTPUT_PATH,
) -> None:
    """Write scenario PV table for downstream analysis."""
    fieldnames = [
        "scenario_id",
        "variant",
        "tenor",
        "r",
        "q",
        "vol",
        "ko_barrier",
        "ki_barrier",
        "product_protection_type",
        "product_protection_rate",
        "interest_protection_type",
        "interest_protection_rate",
        "quoted_ko_rate",
        "product_pv",
        "interest_pv",
        "combined_pv",
        "protected_snowball_pv",
        "quote_ko_sensitivity",
        "quote_ki_sensitivity",
        "interest_ko_sensitivity",
        "interest_ki_sensitivity",
    ]
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_demo_data_json(
    data: dict[str, Any],
    *,
    data_output_path: Path,
) -> None:
    """Write the embedded demo payload to a standalone JSON file."""
    data_output_path.parent.mkdir(parents=True, exist_ok=True)
    data_output_path.write_text(
        json.dumps(data, separators=(",", ":")),
        encoding="utf-8",
    )


def read_demo_data_json(data_input_path: Path) -> dict[str, Any]:
    """Read a standalone demo payload JSON file."""
    return json.loads(data_input_path.read_text(encoding="utf-8"))


def build_demo_data(
    *,
    cubes: dict[str, dict[str, list[list[list[list[float | None]]]]]],
    engine_name: str,
    solver_grid_size: int | None,
    solver_time_steps: int | None,
    exact_barrier_grid: bool = False,
) -> dict[str, Any]:
    """Build the embedded payload for one engine configuration."""
    return {
        "tenorGrid": TENOR_GRID,
        "rGrid": R_GRID,
        "qGrid": Q_GRID,
        "volGrid": VOL_GRID,
        "defaults": {
            "tenor": DEFAULT_TENOR,
            "r": DEFAULT_R,
            "q": DEFAULT_Q,
            "vol": DEFAULT_VOL,
            "variant": DEFAULT_VARIANT,
        },
        "koRateBounds": list(KO_RATE_BOUNDS),
        "variants": cubes,
        "variantMeta": DEMO_VARIANTS,
        "meta": asdict(
            DemoMeta(
                generated_at=datetime.utcnow().isoformat() + "Z",
                engine=engine_name,
                solver_grid_size=solver_grid_size,
                solver_time_steps=solver_time_steps,
                structure={
                    "spot": 100.0,
                    "strike": 100.0,
                    "maturity_years": DEFAULT_TENOR,
                    "tenor_years": TENOR_GRID,
                    "ko_barrier_grid": KO_GRID,
                    "ki_barrier_grid": KI_GRID,
                    "base_ko_barrier": BASE_KO_BARRIER,
                    "base_ki_barrier": BASE_KI_BARRIER,
                    "base_ko_frequency": "monthly",
                    "base_ki_frequency": "daily",
                    "exact_barrier_grid": exact_barrier_grid,
                    "include_principal": False,
                    "target_pv": 0.0,
                    "prepayment": PREPAYMENT,
                    "protected_leg_ko_rate": 1.0,
                    "protected_leg_protection": "VARIANT_SPECIFIC",
                    "protected_leg_include_principal": False,
                    "protected_leg_annualized": False,
                },
                ranges={
                    "tenor": TENOR_GRID,
                    "r": R_GRID,
                    "q": Q_GRID,
                    "vol": VOL_GRID,
                    "ko": KO_GRID,
                    "ki": KI_GRID,
                },
            )
        ),
    }


def _deep_merge_dict(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_dict(base[key], value)
        else:
            base[key] = value


def _resolve_html_i18n(ui_copy: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = json.loads(json.dumps(DEFAULT_HTML_I18N, ensure_ascii=False))
    if not ui_copy:
        return resolved

    for key, value in ui_copy.items():
        if key in LEGACY_UI_COPY_TO_I18N and isinstance(value, str):
            language, i18n_key = LEGACY_UI_COPY_TO_I18N[key]
            resolved[language][i18n_key] = value
        elif key in resolved and isinstance(value, dict):
            _deep_merge_dict(resolved[key], value)
    return resolved


def render_demo_html(data: dict[str, Any], ui_copy: dict[str, Any] | None = None) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    i18n = _resolve_html_i18n(ui_copy)
    return (
        template.replace("__DATA__", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
        .replace("__I18N__", json.dumps(i18n, separators=(",", ":"), ensure_ascii=False))
    )


def render_html(data: dict[str, Any], ui_copy: dict[str, Any] | None = None) -> str:
    """Backward-compatible alias for the standalone HTML renderer."""
    return render_demo_html(data, ui_copy=ui_copy)


def write_demo_html(
    data: dict[str, Any],
    *,
    html_output_path: Path = OUTPUT_PATH,
    ui_copy: dict[str, Any] | None = None,
) -> None:
    html_output_path.parent.mkdir(parents=True, exist_ok=True)
    html_output_path.write_text(
        render_demo_html(data, ui_copy=ui_copy),
        encoding="utf-8",
    )


def write_demo_artifacts(
    data: dict[str, Any],
    scenario_rows: list[dict[str, Any]] | None = None,
    *,
    html_output_path: Path | None = None,
    data_output_path: Path | None = None,
    csv_output_path: Path | None = None,
    ui_copy: dict[str, Any] | None = None,
) -> None:
    if html_output_path is not None:
        write_demo_html(data, html_output_path=html_output_path, ui_copy=ui_copy)
    if data_output_path is not None:
        write_demo_data_json(data, data_output_path=data_output_path)
    if csv_output_path is not None and scenario_rows is not None:
        write_scenario_csv(scenario_rows, csv_output_path=csv_output_path)


def write_demo_files(
    data: dict[str, Any],
    scenario_rows: list[dict[str, Any]],
    *,
    output_path: Path = OUTPUT_PATH,
    csv_output_path: Path = CSV_OUTPUT_PATH,
    ui_copy: dict[str, Any] | None = None,
) -> None:
    """Backward-compatible alias for writing HTML plus CSV."""
    write_demo_artifacts(
        data,
        scenario_rows,
        html_output_path=output_path,
        csv_output_path=csv_output_path,
        ui_copy=ui_copy,
    )


def calculate_demo_data(
    *,
    engine: Any,
    bump_engine: Any,
    engine_name: str,
    solver_grid_size: int | None,
    solver_time_steps: int | None,
    exact_barrier_grid: bool = False,
    parallel_workers: int | None = 1,
    progress_label: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cubes, scenario_rows = build_cube_with_engines(
        engine=engine,
        bump_engine=bump_engine,
        progress_label=progress_label or engine_name,
        exact_barrier_grid=exact_barrier_grid,
        parallel_workers=parallel_workers,
    )
    return (
        build_demo_data(
            cubes=cubes,
            engine_name=engine_name,
            solver_grid_size=solver_grid_size,
            solver_time_steps=solver_time_steps,
            exact_barrier_grid=exact_barrier_grid,
        ),
        scenario_rows,
    )


def calculate_pde_demo_data(
    *,
    grid_size: int = PDE_GRID_SIZE,
    time_steps: int = PDE_TIME_STEPS,
    parallel_workers: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pde_params = PDEParams(grid_size=grid_size, time_steps=time_steps)
    bump_params = PDEParams(
        grid_size=max(50, pde_params.grid_size // 2),
        time_steps=max(80, pde_params.time_steps // 2),
    )
    return calculate_demo_data(
        engine=SnowballPDESolver(params=pde_params),
        bump_engine=SnowballPDESolver(params=bump_params),
        engine_name="SnowballPDESolver",
        solver_grid_size=pde_params.grid_size,
        solver_time_steps=pde_params.time_steps,
        exact_barrier_grid=True,
        parallel_workers=parallel_workers or max(1, os.cpu_count() or 1),
        progress_label="SnowballPDESolver",
    )


def calculate_quad_demo_data(
    *,
    grid_points: int,
    parallel_workers: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
    from asset.equity.param import QuadParams

    quad_params = QuadParams(grid_points=grid_points)
    return calculate_demo_data(
        engine=SnowballQuadEngine(params=quad_params),
        bump_engine=SnowballQuadEngine(params=QuadParams(grid_points=grid_points)),
        engine_name="SnowballQuadEngine",
        solver_grid_size=grid_points,
        solver_time_steps=None,
        exact_barrier_grid=True,
        parallel_workers=parallel_workers or max(1, os.cpu_count() or 1),
        progress_label=f"SnowballQuadEngine({grid_points})",
    )
