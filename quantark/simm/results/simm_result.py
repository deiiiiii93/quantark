"""
SIMM Result Dataclasses.

This module defines comprehensive result structures for SIMM calculations,
providing hierarchical breakdown of margin components.
"""
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Dict, List, Optional, Any, Union
import json

from ..taxonomy import ProductClass, RiskClass, MarginType
from ..sensitivity import Sensitivity


@dataclass
class SensitivityContribution:
    """Contribution from a single sensitivity to a bucket."""
    sensitivity_id: str
    position_id: str
    ws_value: float  # Weighted sensitivity value
    pct_of_bucket: float  # Percentage of bucket WS sum


@dataclass
class AddonBreakdown:
    """Breakdown of SIMM add-ons by type."""
    supervision_addon: float = 0.0
    viral_addon: float = 0.0
    total: float = 0.0


@dataclass
class BucketDetail:
    """Detailed results for a single bucket."""
    bucket: Union[int, str]
    k_value: float  # Bucket-level K
    s_value: float  # Capped sum S_b
    ws_sum: float  # Sum of weighted sensitivities
    concentration_factor: float  # CR for bucket

    # Sensitivity contributions
    sensitivities: List[SensitivityContribution] = field(default_factory=list)

    @property
    def net_sensitivity(self) -> float:
        """Calculate net sensitivity (uncapped S_b)."""
        return self.ws_sum / self.concentration_factor if self.concentration_factor > 0 else 0.0


@dataclass
class RiskClassMargin:
    """Margin breakdown for a single risk class."""
    risk_class: RiskClass
    product_class: ProductClass

    # Margin components
    delta_margin: float = 0.0
    vega_margin: float = 0.0
    curvature_margin: float = 0.0
    base_corr_margin: float = 0.0  # Credit Q only
    total_margin: float = 0.0

    # Bucket-level detail
    bucket_detail: Dict[Union[int, str], BucketDetail] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and compute total margin."""
        if self.total_margin == 0.0:
            self.total_margin = (
                self.delta_margin +
                self.vega_margin +
                self.curvature_margin +
                self.base_corr_margin
            )


@dataclass
class SIMMResult:
    """Complete SIMM calculation result."""

    # Total and summary
    total_simm: float
    calculation_currency: str
    calculation_date: date
    simm_version: str

    # By product class
    product_class_simm: Dict[ProductClass, float]

    # By risk class (within each product class)
    risk_class_margin: Dict[ProductClass, Dict[RiskClass, RiskClassMargin]]

    # Add-ons
    addon_amount: float = 0.0
    addon_details: Optional[AddonBreakdown] = None

    # Attribution
    attribution: Optional[Any] = None  # SIMMAttribution, set after import

    # Metadata
    execution_time_seconds: float = 0.0
    config_summary: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def get_margin_by_risk_class(self) -> Dict[RiskClass, float]:
        """Get aggregated margin by risk class across all product classes.

        Returns:
            Dictionary mapping RiskClass to total margin.
        """
        margin_by_risk = {rc: 0.0 for rc in RiskClass}
        for pc_margin_dict in self.risk_class_margin.values():
            for rc, rc_margin in pc_margin_dict.items():
                margin_by_risk[rc] += rc_margin.total_margin
        return margin_by_risk

    def get_margin_by_margin_type(self) -> Dict[MarginType, float]:
        """Get aggregated margin by margin type across all risk classes.

        Returns:
            Dictionary mapping MarginType to total margin.
        """
        margin_by_type = {mt: 0.0 for mt in MarginType}
        for pc_margin_dict in self.risk_class_margin.values():
            for rc_margin in pc_margin_dict.values():
                margin_by_type[MarginType.DELTA] += rc_margin.delta_margin
                margin_by_type[MarginType.VEGA] += rc_margin.vega_margin
                margin_by_type[MarginType.CURVATURE] += rc_margin.curvature_margin
                margin_by_type[MarginType.BASE_CORR] += rc_margin.base_corr_margin
        return margin_by_type

    def get_top_buckets(self, n: int) -> List[tuple]:
        """Get top n buckets by margin contribution.

        Args:
            n: Number of top buckets to return.

        Returns:
            List of (risk_class, bucket, margin) tuples sorted by margin descending.
        """
        bucket_margins = []
        for pc, rc_dict in self.risk_class_margin.items():
            for rc, rc_margin in rc_dict.items():
                for bucket, bucket_detail in rc_margin.bucket_detail.items():
                    bucket_margins.append((rc, bucket, bucket_detail.k_value))

        bucket_margins.sort(key=lambda x: x[2], reverse=True)
        return bucket_margins[:n]

    def validate(self) -> bool:
        """Validate result consistency.

        Returns:
            True if result is consistent.

        Raises:
            ValidationError: If validation fails.
        """
        try:
            from quantark.util.exceptions import ValidationError
        except ImportError:
            # Fallback for when running from tests
            from quantark.util.exceptions import ValidationError

        # Check total equals sum of product class margins plus addon
        product_class_sum = sum(self.product_class_simm.values())
        expected_total = product_class_sum + self.addon_amount

        if abs(self.total_simm - expected_total) > 1e-6:
            raise ValidationError(
                f"Total SIMM {self.total_simm} does not match sum of product classes "
                f"{product_class_sum} plus addon {self.addon_amount}"
            )

        # Check risk class margins sum correctly
        for pc, rc_dict in self.risk_class_margin.items():
            for rc, rc_margin in rc_dict.items():
                if rc_margin.total_margin < 0:
                    raise ValidationError(f"Negative margin in {rc} risk class")

                # Validate bucket detail sums
                for bucket, bucket_detail in rc_margin.bucket_detail.items():
                    if bucket_detail.k_value < 0:
                        raise ValidationError(f"Negative bucket K value in {rc}/{bucket}")

        # Check product class margins match risk class sums
        for pc, expected_margin in self.product_class_simm.items():
            if pc not in self.risk_class_margin:
                continue

            actual_margin = sum(
                rc.total_margin for rc in self.risk_class_margin[pc].values()
            )
            if abs(actual_margin - expected_margin) > 1e-6:
                raise ValidationError(
                    f"Product class {pc} margin {actual_margin} does not match "
                    f"sum of risk classes {expected_margin}"
                )

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to nested dictionary.

        Returns:
            Dictionary representation of the result.
        """
        result = asdict(self)

        # Convert enums to strings
        def convert_enums(obj):
            if isinstance(obj, dict):
                # Convert enum keys to strings
                return {str(k) if hasattr(k, 'value') else k: convert_enums(v)
                       for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enums(item) for item in obj]
            elif hasattr(obj, 'value'):  # Enum
                return obj.value
            else:
                return obj

        return convert_enums(result)

    def to_json(self) -> str:
        """Convert result to JSON string.

        Returns:
            JSON representation of the result.
        """
        def json_serializer(obj):
            """Custom JSON serializer for enums and dataclasses."""
            if hasattr(obj, 'value'):  # Enum
                return obj.value
            elif isinstance(obj, (dict, list, tuple, set)):
                return obj
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return str(obj)

        return json.dumps(self.to_dict(), default=json_serializer)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SIMMResult":
        """Create result from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            SIMMResult instance.
        """
        # Convert string enums back to enum objects
        def convert_enums(obj):
            if isinstance(obj, dict):
                return {k: convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enums(item) for item in obj]
            elif isinstance(obj, str):
                # Try to convert to enum
                for enum_cls in [ProductClass, RiskClass, MarginType]:
                    try:
                        return enum_cls(obj)
                    except ValueError:
                        pass
                return obj
            else:
                return obj

        data = convert_enums(data)
        return cls(**data)

    def compare(self, other: "SIMMResult") -> Dict[str, Any]:
        """Compare this result with another.

        Args:
            other: SIMMResult to compare against.

        Returns:
            Comparison summary with deltas.
        """
        return {
            "total_simm": {
                "current": self.total_simm,
                "other": other.total_simm,
                "delta": self.total_simm - other.total_simm,
                "delta_pct": ((self.total_simm - other.total_simm) / other.total_simm * 100
                             if other.total_simm != 0 else 0)
            },
            "product_class_delta": {
                pc.value: self.product_class_simm.get(pc, 0) - other.product_class_simm.get(pc, 0)
                for pc in ProductClass
            },
            "risk_class_delta": {
                rc.value: {
                    "current": self.get_margin_by_risk_class().get(rc, 0),
                    "other": other.get_margin_by_risk_class().get(rc, 0),
                    "delta": (self.get_margin_by_risk_class().get(rc, 0) -
                             other.get_margin_by_risk_class().get(rc, 0))
                }
                for rc in RiskClass
            }
        }

    def diff(self, other: "SIMMResult") -> Dict[str, Any]:
        """Get detailed differences between two results.

        Args:
            other: SIMMResult to compare against.

        Returns:
            Detailed difference breakdown.
        """
        diff_result = self.compare(other)

        # Add bucket-level differences
        diff_result["bucket_changes"] = []
        for pc, rc_dict in self.risk_class_margin.items():
            for rc, rc_margin in rc_dict.items():
                for bucket, bucket_detail in rc_margin.bucket_detail.items():
                    other_bucket = (
                        other.risk_class_margin.get(pc, {})
                        .get(rc, RiskClassMargin(rc, pc))
                        .bucket_detail.get(bucket)
                    )
                    if other_bucket:
                        delta_k = bucket_detail.k_value - other_bucket.k_value
                        if abs(delta_k) > 1e-6:
                            diff_result["bucket_changes"].append({
                                "risk_class": rc.value,
                                "bucket": bucket,
                                "current_k": bucket_detail.k_value,
                                "other_k": other_bucket.k_value,
                                "delta_k": delta_k
                            })

        return diff_result
