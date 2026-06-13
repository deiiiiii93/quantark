"""
Main SIMM Calculator.

Orchestrates the full ISDA SIMM v2.6 margin calculation:

1. Sensitivities are split by product class (paragraph 6): the six risk
   classes take their component risks only from trades of that product
   class.
2. Within each (product class, risk class): net sensitivities per risk
   factor, compute concentration risk factors, weight, aggregate within
   buckets (K_b) and across buckets (Delta/Vega margins); derive
   curvature exposures from vega sensitivities (paragraph 11) and the
   Base Correlation margin for Credit Qualifying (paragraph 13).
3. Risk class margins combine with the psi correlations into
   SIMM_product (Section K); the total SIMM is the sum over product
   classes with multiplicative scales and add-ons (Section L).
"""

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, Hashable, List, Optional, Tuple, Union

from quantark.simm.config import SIMMConfig
from quantark.simm.market_data import SIMMMarketData
from quantark.util.exceptions import ValidationError
from quantark.simm.taxonomy import (
    EQUITY_VOLATILITY_INDEX_BUCKET,
    MarginType,
    ProductClass,
    RiskClass,
    TENOR_LABEL_DAYS,
    is_residual_bucket,
)
from quantark.simm.sensitivity import (
    AnySensitivity,
    BaseCorrSensitivity,
    CurvatureSensitivity,
    SensitivityCollection,
)
from quantark.simm.calibration.accessors import scaling_function
from quantark.simm.calibration.credit_qualifying import (
    CREDIT_QUALIFYING_BASE_CORRELATION_RISK_WEIGHT,
)
from quantark.simm.engines.aggregation.concentration import ConcentrationCalculator
from quantark.simm.engines.aggregation.weighted_sensitivity import (
    WeightedSensitivityCalculator,
    net_by_risk_factor,
)
from quantark.simm.engines.aggregation.bucket_aggregator import (
    BucketAggregator,
    BucketResult,
)
from quantark.simm.engines.aggregation.risk_class_aggregator import (
    CurvatureExposure,
    RiskClassAggregator,
    RiskClassResult,
)
from quantark.simm.engines.aggregation.product_class_aggregator import (
    ProductClassAggregator,
    ProductClassResult,
)
from quantark.simm.engines.aggregation.addon import AddOnCalculator, AddOnResult


@dataclass
class SIMMAggregationResult:
    """Complete SIMM calculation result with attribution.

    Attributes:
        total_margin: Total SIMM (sum over product classes, after
            multiplicative scales, plus add-ons).
        by_product_class: SIMM per product class (after multiplicative
            scales).
        product_class_results: Detailed per-product-class results.
        by_risk_class: Risk class IM summed across product classes.
        by_margin_type: Margin per risk class and margin type, summed
            across product classes.
        addon: Add-on calculation result.
        calculation_currency: Currency of the margin amounts.
        calculation_timestamp: When the calculation was performed.
        simm_version: SIMM version used.
        bucket_details: Per product class, risk class and margin type,
            the bucket-level results (when enabled in the config).
    """
    total_margin: float
    by_product_class: Dict[ProductClass, float] = field(default_factory=dict)
    product_class_results: Dict[ProductClass, ProductClassResult] = field(default_factory=dict)
    by_risk_class: Dict[RiskClass, float] = field(default_factory=dict)
    by_margin_type: Dict[RiskClass, Dict[MarginType, float]] = field(default_factory=dict)
    addon: Optional[AddOnResult] = None
    calculation_currency: str = "USD"
    calculation_timestamp: Optional[str] = None
    simm_version: str = "2.6"
    bucket_details: Dict[ProductClass, Dict[RiskClass, Dict[MarginType, Dict[Any, BucketResult]]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "total_margin": self.total_margin,
            "by_product_class": {pc.value: m for pc, m in self.by_product_class.items()},
            "by_risk_class": {rc.value: m for rc, m in self.by_risk_class.items()},
            "by_margin_type": {
                rc.value: {mt.value: m for mt, m in mt_dict.items()}
                for rc, mt_dict in self.by_margin_type.items()
            },
            "calculation_currency": self.calculation_currency,
            "calculation_timestamp": self.calculation_timestamp,
            "simm_version": self.simm_version,
        }
        if self.addon:
            result["addon"] = {
                "fixed": self.addon.fixed_addon,
                "factor": self.addon.factor_addon,
                "total": self.addon.total_addon,
            }
        return result

    def __str__(self) -> str:
        lines = [
            f"SIMM Result (v{self.simm_version})",
            f"  Total Margin: {self.total_margin:,.2f} {self.calculation_currency}",
            "",
            "  By Product Class:",
        ]
        for pc in ProductClass:
            margin = self.by_product_class.get(pc, 0.0)
            if margin != 0.0:
                lines.append(f"    {pc.value}: {margin:,.2f}")
        lines.append("")
        lines.append("  By Risk Class:")
        for rc in RiskClass:
            margin = self.by_risk_class.get(rc, 0.0)
            if margin != 0.0:
                lines.append(f"    {rc.value}: {margin:,.2f}")
                for mt, m in self.by_margin_type.get(rc, {}).items():
                    if m != 0.0:
                        lines.append(f"      {mt.value}: {m:,.2f}")
        return "\n".join(lines)


class SIMMCalculator:
    """Main SIMM calculation engine (ISDA SIMM v2.6).

    Example:
        >>> config = SIMMConfig()
        >>> calculator = SIMMCalculator(config)
        >>> result = calculator.calculate(sensitivities)
        >>> print(result.total_margin)
    """

    def __init__(
        self,
        config: Optional[SIMMConfig] = None,
        market_data: Optional[SIMMMarketData] = None,
    ):
        """Initialize the calculator.

        Args:
            config: SIMM configuration. Defaults to SIMMConfig().
        """
        self.config = config or SIMMConfig()
        self.market_data = market_data
        ccy = self.config.calculation_currency
        self.concentration_calc = ConcentrationCalculator()
        self.weighted_sens_calc = WeightedSensitivityCalculator(ccy)
        self.bucket_agg = BucketAggregator(ccy)
        self.risk_class_agg = RiskClassAggregator(ccy)
        self.product_class_agg = ProductClassAggregator()
        self.addon_calc = AddOnCalculator(self.config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(
        self,
        sensitivities: Union[SensitivityCollection, List[AnySensitivity]],
        notionals: Optional[Dict[str, float]] = None,
        market_data: Optional[SIMMMarketData] = None,
    ) -> SIMMAggregationResult:
        """Calculate total SIMM margin.

        Args:
            sensitivities: Collection (or list) of sensitivities.
            notionals: Optional notionals for factor-based add-ons
                (Section L).

        Returns:
            SIMMAggregationResult with full attribution.
        """
        if not isinstance(sensitivities, SensitivityCollection):
            sensitivities = SensitivityCollection(list(sensitivities))
        sensitivities = self._normalize_sensitivities(
            sensitivities, market_data or self.market_data
        )
        self._validate_curvature_mode(sensitivities)

        product_class_results: Dict[ProductClass, ProductClassResult] = {}
        bucket_details: Dict[ProductClass, Dict[RiskClass, Dict[MarginType, Dict[Any, BucketResult]]]] = {}

        for product_class in sensitivities.product_classes():
            subset = SensitivityCollection(sensitivities.by_product_class(product_class))
            rc_results, rc_buckets = self._calculate_risk_classes(subset)
            product_class_results[product_class] = self.product_class_agg.aggregate(
                product_class, rc_results
            )
            if self.config.include_bucket_detail:
                bucket_details[product_class] = rc_buckets

        # Multiplicative scales (Section L): total uses MS_p * SIMM_p.
        by_product_class = {
            pc: result.margin * self.config.get_product_class_multiplier(pc.value)
            for pc, result in product_class_results.items()
        }

        addon_result = self.addon_calc.calculate(
            notionals=notionals,
            product_class_margins={
                pc: r.margin for pc, r in product_class_results.items()
            },
        )

        total_margin = sum(by_product_class.values()) + addon_result.total_addon

        # Cross-product-class sums for reporting.
        by_risk_class: Dict[RiskClass, float] = {}
        by_margin_type: Dict[RiskClass, Dict[MarginType, float]] = {}
        for pc_result in product_class_results.values():
            for rc, im in pc_result.risk_class_margins.items():
                by_risk_class[rc] = by_risk_class.get(rc, 0.0) + im
            for rc, detail in pc_result.margin_type_detail.items():
                target = by_margin_type.setdefault(rc, {})
                for mt, m in detail.items():
                    target[mt] = target.get(mt, 0.0) + m

        return SIMMAggregationResult(
            total_margin=total_margin,
            by_product_class=by_product_class,
            product_class_results=product_class_results,
            by_risk_class=by_risk_class,
            by_margin_type=by_margin_type,
            addon=addon_result,
            calculation_currency=self.config.calculation_currency,
            calculation_timestamp=datetime.now(timezone.utc).isoformat(),
            simm_version=str(self.config.version),
            bucket_details=bucket_details,
        )

    def calculate_from_crif(
        self,
        crif_records: List[Dict[str, Any]],
        notionals: Optional[Dict[str, float]] = None,
        market_data: Optional[SIMMMarketData] = None,
    ) -> SIMMAggregationResult:
        """Calculate SIMM from CRIF record dictionaries.

        Args:
            crif_records: List of CRIF record dicts.
            notionals: Optional notionals for add-ons.

        Returns:
            SIMMAggregationResult.
        """
        from quantark.simm.crif.parser import crif_records_to_sensitivities

        collection = crif_records_to_sensitivities(
            crif_records,
            calculation_currency=self.config.calculation_currency,
            market_data=market_data or self.market_data,
        )
        return self.calculate(collection, notionals, market_data)

    def _normalize_sensitivities(
        self,
        sensitivities: SensitivityCollection,
        market_data: Optional[SIMMMarketData],
    ) -> SensitivityCollection:
        """Convert all amounts and USD thresholds into calculation currency."""
        target = self.config.calculation_currency
        normalized: List[AnySensitivity] = []
        for sensitivity in sensitivities:
            source = sensitivity.amount_currency.upper()
            if source == target:
                normalized.append(replace(sensitivity, amount_currency=target))
                continue
            if market_data is None:
                raise ValidationError(
                    f"SIMMMarketData required to convert sensitivity amount from "
                    f"{source} to {target}"
                )
            normalized.append(
                replace(
                    sensitivity,
                    amount=sensitivity.amount * market_data.fx_rate(source, target),
                    amount_currency=target,
                )
            )

        if target == "USD" or not normalized:
            threshold_scale = 1.0
        else:
            if market_data is None:
                raise ValidationError(
                    f"SIMMMarketData required to convert USD concentration thresholds "
                    f"to {target}"
                )
            threshold_scale = market_data.fx_rate("USD", target)
        self.concentration_calc = ConcentrationCalculator(threshold_scale)
        return SensitivityCollection(normalized)

    def _validate_curvature_mode(self, sensitivities: SensitivityCollection) -> None:
        if not self.config.derive_curvature_from_vega:
            return
        explicit = sensitivities.by_margin_type(MarginType.CURVATURE)
        vega = sensitivities.by_margin_type(MarginType.VEGA)
        if explicit and vega:
            raise ValidationError(
                "Explicit curvature sensitivities cannot be combined with "
                "vega-derived curvature"
            )

    # ------------------------------------------------------------------
    # Per-(product class, risk class) margins
    # ------------------------------------------------------------------

    def _calculate_risk_classes(
        self,
        sensitivities: SensitivityCollection,
    ) -> Tuple[
        Dict[RiskClass, Dict[MarginType, RiskClassResult]],
        Dict[RiskClass, Dict[MarginType, Dict[Any, BucketResult]]],
    ]:
        """Compute all margin types for the six risk classes of one
        product class."""
        results: Dict[RiskClass, Dict[MarginType, RiskClassResult]] = {}
        buckets: Dict[RiskClass, Dict[MarginType, Dict[Any, BucketResult]]] = {}

        for risk_class in RiskClass:
            rc_results: Dict[MarginType, RiskClassResult] = {}
            rc_buckets: Dict[MarginType, Dict[Any, BucketResult]] = {}

            if self.config.calculate_delta:
                result, detail = self._delta_or_vega_margin(
                    sensitivities, risk_class, MarginType.DELTA
                )
                rc_results[MarginType.DELTA] = result
                rc_buckets[MarginType.DELTA] = detail

            if self.config.calculate_vega:
                result, detail = self._delta_or_vega_margin(
                    sensitivities, risk_class, MarginType.VEGA
                )
                rc_results[MarginType.VEGA] = result
                rc_buckets[MarginType.VEGA] = detail

            if self.config.calculate_curvature:
                rc_results[MarginType.CURVATURE] = self._curvature_margin(
                    sensitivities, risk_class
                )

            if (
                self.config.calculate_base_corr
                and risk_class == RiskClass.CREDIT_QUALIFYING
            ):
                rc_results[MarginType.BASE_CORR] = self._base_corr_margin(sensitivities)

            if any(r.margin != 0.0 for r in rc_results.values()):
                results[risk_class] = rc_results
                buckets[risk_class] = rc_buckets

        return results, buckets

    def _delta_or_vega_margin(
        self,
        sensitivities: SensitivityCollection,
        risk_class: RiskClass,
        margin_type: MarginType,
    ) -> Tuple[RiskClassResult, Dict[Any, BucketResult]]:
        """Delta or vega margin for one risk class (paragraphs 7, 8, 10)."""
        # Exclude base-corr records (separate margin) and explicit
        # curvature records.
        records = [
            s for s in sensitivities.by_risk_class_and_margin_type(risk_class, margin_type)
        ]
        if not records:
            return (
                RiskClassResult(risk_class=risk_class, margin_type=margin_type, margin=0.0),
                {},
            )

        by_bucket: Dict[Any, List[AnySensitivity]] = {}
        for s in records:
            by_bucket.setdefault(s.bucket, []).append(s)

        bucket_results = []
        details: Dict[Any, BucketResult] = {}
        for bucket, bucket_sens in by_bucket.items():
            netted = net_by_risk_factor(bucket_sens, margin_type)
            cr = self.concentration_calc.calculate(netted, risk_class, margin_type, bucket)
            weighted = self.weighted_sens_calc.calculate(
                netted, risk_class, margin_type, bucket, cr.cr_values
            )
            bucket_result = self.bucket_agg.aggregate(
                weighted, risk_class, margin_type, bucket, bucket_cr=cr.bucket_cr
            )
            bucket_results.append(bucket_result)
            details[bucket] = bucket_result

        result = self.risk_class_agg.aggregate_delta_vega(
            bucket_results, risk_class, margin_type
        )
        return result, details

    def _curvature_margin(
        self,
        sensitivities: SensitivityCollection,
        risk_class: RiskClass,
    ) -> RiskClassResult:
        """Curvature margin for one risk class (paragraph 11).

        Curvature exposures are derived from vega sensitivities using the
        scaling function SF(t) (when enabled); explicit
        CurvatureSensitivity records are added as supplied.
        """
        exposures: Dict[Tuple[Any, Hashable], CurvatureExposure] = {}

        def add(bucket: Any, rf: Hashable, amount: float, group: str = "") -> None:
            key = (bucket, rf)
            entry = exposures.get(key)
            if entry is None:
                exposures[key] = CurvatureExposure(
                    bucket=bucket, risk_factor=rf, amount=amount, group=group
                )
            else:
                entry.amount += amount

        if self.config.derive_curvature_from_vega:
            for sens in sensitivities.by_risk_class_and_margin_type(
                risk_class, MarginType.VEGA
            ):
                # Equity bucket 12 (Volatility Indexes) has zero curvature
                # exposure (paragraph 11(b)).
                if (
                    risk_class == RiskClass.EQUITY
                    and not is_residual_bucket(sens.bucket)
                    and int(sens.bucket) == EQUITY_VOLATILITY_INDEX_BUCKET
                ):
                    continue
                expiry_days = TENOR_LABEL_DAYS[sens.vertex]  # type: ignore[union-attr]
                cvr = scaling_function(expiry_days) * sens.amount
                add(sens.bucket, sens.risk_factor, cvr, getattr(sens, "group_name", ""))

        for sens in sensitivities.by_risk_class_and_margin_type(
            risk_class, MarginType.CURVATURE
        ):
            if isinstance(sens, CurvatureSensitivity):
                add(
                    sens.bucket,
                    sens.risk_factor,
                    sens.amount,
                    getattr(sens, "group_name", ""),
                )

        return self.risk_class_agg.aggregate_curvature(
            list(exposures.values()), risk_class
        )

    def _base_corr_margin(
        self,
        sensitivities: SensitivityCollection,
    ) -> RiskClassResult:
        """Base correlation margin (paragraph 13, Credit Qualifying only).

        Base correlation sensitivities take CR = 1 (paragraph 8(b)).
        """
        records = sensitivities.by_risk_class_and_margin_type(
            RiskClass.CREDIT_QUALIFYING, MarginType.BASE_CORR
        )
        if not records:
            return RiskClassResult(
                risk_class=RiskClass.CREDIT_QUALIFYING,
                margin_type=MarginType.BASE_CORR,
                margin=0.0,
            )

        rw = CREDIT_QUALIFYING_BASE_CORRELATION_RISK_WEIGHT
        ws_by_index: Dict[str, float] = {}
        for sens in records:
            if isinstance(sens, BaseCorrSensitivity):
                name = sens.index_name
            else:
                name = sens.qualifier
            ws_by_index[name] = ws_by_index.get(name, 0.0) + rw * sens.amount

        return self.risk_class_agg.aggregate_base_corr(ws_by_index)
