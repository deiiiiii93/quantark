"""
SIMM Attribution Module.

This module provides margin attribution to various dimensions including
position-level, bucket-level, and risk class breakdowns.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict

from ..taxonomy import ProductClass, RiskClass, MarginType


@dataclass
class ContributorInfo:
    """Information about a top SIMM contributor."""
    identifier: str  # position_id, bucket, risk_class, etc.
    contribution_type: str  # "position", "bucket", "risk_class"
    amount: float
    pct_of_total: float


@dataclass
class PositionAttribution:
    """SIMM attribution to a single position."""
    position_id: str
    trade_id: Optional[str]
    underlying: str

    # Attributed margins
    delta_contribution: float = 0.0
    vega_contribution: float = 0.0
    curvature_contribution: float = 0.0
    total_contribution: float = 0.0

    # Percentage of total
    pct_of_total: float = 0.0

    # Sensitivities
    sensitivities_count: int = 0


@dataclass
class SIMMAttribution:
    """Attribution of SIMM to various dimensions."""

    # By product class
    by_product_class: Dict[ProductClass, float]

    # By risk class
    by_risk_class: Dict[RiskClass, float]

    # By margin type
    by_margin_type: Dict[MarginType, float]

    # By bucket (within each risk class)
    by_bucket: Dict[RiskClass, Dict[Union[int, str], float]]

    # By position/trade (approximate contribution)
    by_position: Dict[str, PositionAttribution]

    # Top contributors
    top_contributors: List[ContributorInfo]

    @classmethod
    def calculate_from_result(cls, result: Any, max_positions: int = 100) -> "SIMMAttribution":
        """Calculate attribution from a SIMMResult.

        Args:
            result: SIMMResult to attribute.
            max_positions: Maximum number of positions to include.

        Returns:
            SIMMAttribution instance.
        """
        # Calculate attribution by product class
        by_product_class = result.product_class_simm.copy()

        # Calculate attribution by risk class
        by_risk_class = result.get_margin_by_risk_class()

        # Calculate attribution by margin type
        by_margin_type = result.get_margin_by_margin_type()

        # Calculate attribution by bucket
        by_bucket: Dict[RiskClass, Dict[Any, float]] = defaultdict(dict)
        for pc, rc_dict in result.risk_class_margin.items():
            for rc, rc_margin in rc_dict.items():
                for bucket, bucket_detail in rc_margin.bucket_detail.items():
                    by_bucket[rc][bucket] = bucket_detail.k_value

        # Calculate position attribution (simplified approximation)
        by_position = cls._calculate_position_attribution(result)

        # Identify top contributors
        top_contributors = cls._identify_top_contributors(
            by_position, by_bucket, by_risk_class, result.total_simm, max_positions
        )

        return cls(
            by_product_class=by_product_class,
            by_risk_class=by_risk_class,
            by_margin_type=by_margin_type,
            by_bucket=by_bucket,
            by_position=by_position,
            top_contributors=top_contributors
        )

    @classmethod
    def _calculate_position_attribution(cls, result: Any) -> Dict[str, PositionAttribution]:
        """Calculate approximate position-level attribution.

        This is a simplified approximation. Due to correlation effects,
        the sum of position attributions may not exactly equal total SIMM.

        Args:
            result: SIMMResult to attribute.

        Returns:
            Dictionary mapping position_id to PositionAttribution.
        """
        position_attribution: Dict[str, PositionAttribution] = {}

        # Iterate through all buckets and attribute to positions
        for pc, rc_dict in result.risk_class_margin.items():
            for rc, rc_margin in rc_dict.items():
                for bucket, bucket_detail in rc_margin.bucket_detail.items():
                    if not bucket_detail.sensitivities:
                        continue

                    # Calculate total WS for this bucket
                    total_ws = bucket_detail.ws_sum
                    if total_ws == 0:
                        continue

                    # Attribute bucket K proportionally by WS^2 contribution
                    for sens_contrib in bucket_detail.sensitivities:
                        position_id = sens_contrib.position_id

                        if position_id not in position_attribution:
                            position_attribution[position_id] = PositionAttribution(
                                position_id=position_id,
                                trade_id=None,
                                underlying=""
                            )

                        # Calculate proportional attribution
                        ws_ratio = (sens_contrib.ws_value ** 2) / (total_ws ** 2)
                        attributed_margin = bucket_detail.k_value * ws_ratio

                        # Add to position attribution based on margin type
                        # This is simplified - in reality we'd track which sensitivity types
                        pos_attr = position_attribution[position_id]

                        # Determine margin type from sensitivity (simplified)
                        # In practice, we'd need to track sensitivity types
                        if rc in [RiskClass.CREDIT_QUALIFYING]:
                            pos_attr.base_corr_margin += attributed_margin * 0.1  # Simplified
                        if "vega" in str(sens_contrib).lower() or "vol" in str(sens_contrib).lower():
                            pos_attr.vega_contribution += attributed_margin
                        else:
                            pos_attr.delta_contribution += attributed_margin

                        pos_attr.sensitivities_count += 1

        # Calculate totals and percentages
        for pos_attr in position_attribution.values():
            pos_attr.total_contribution = (
                pos_attr.delta_contribution +
                pos_attr.vega_contribution +
                pos_attr.curvature_contribution +
                pos_attr.base_corr_margin
            )

        return position_attribution

    @classmethod
    def _identify_top_contributors(
        cls,
        by_position: Dict[str, PositionAttribution],
        by_bucket: Dict[RiskClass, Dict[Any, float]],
        by_risk_class: Dict[RiskClass, float],
        total_simm: float,
        max_positions: int
    ) -> List[ContributorInfo]:
        """Identify top contributors across all dimensions.

        Args:
            by_position: Position-level attribution.
            by_bucket: Bucket-level attribution.
            by_risk_class: Risk class attribution.
            total_simm: Total SIMM for percentage calculation.
            max_positions: Maximum number of position contributors.

        Returns:
            List of top contributors.
        """
        contributors: List[ContributorInfo] = []

        # Top positions
        sorted_positions = sorted(
            by_position.values(),
            key=lambda p: p.total_contribution,
            reverse=True
        )[:max_positions]

        for pos in sorted_positions:
            contributors.append(ContributorInfo(
                identifier=pos.position_id,
                contribution_type="position",
                amount=pos.total_contribution,
                pct_of_total=(pos.total_contribution / total_simm * 100
                             if total_simm != 0 else 0)
            ))

        # Top buckets
        for rc, bucket_dict in by_bucket.items():
            for bucket, amount in sorted(
                bucket_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]:  # Top 5 buckets per risk class
                contributors.append(ContributorInfo(
                    identifier=f"{rc.value}:{bucket}",
                    contribution_type="bucket",
                    amount=amount,
                    pct_of_total=(amount / total_simm * 100 if total_simm != 0 else 0)
                ))

        # Top risk classes
        for rc, amount in sorted(
            by_risk_class.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            contributors.append(ContributorInfo(
                identifier=rc.value,
                contribution_type="risk_class",
                amount=amount,
                pct_of_total=(amount / total_simm * 100 if total_simm != 0 else 0)
            ))

        # Sort all contributors by amount
        contributors.sort(key=lambda c: c.amount, reverse=True)

        return contributors

    def get_diversification_benefit(
        self,
        standalone_margins: Dict[RiskClass, float]
    ) -> Dict[str, float]:
        """Calculate diversification benefit.

        Args:
            standalone_margins: Standalone margins by risk class.

        Returns:
            Dictionary with diversification benefit metrics.
        """
        total_standalone = sum(standalone_margins.values())
        total_simm = sum(self.by_risk_class.values())

        diversification_benefit = total_standalone - total_simm
        diversification_pct = (
            (diversification_benefit / total_standalone * 100)
            if total_standalone != 0 else 0
        )

        return {
            "diversification_benefit": diversification_benefit,
            "diversification_pct": diversification_pct,
            "total_standalone": total_standalone,
            "total_simm": total_simm
        }

    def get_concentration_metrics(self) -> Dict[str, Any]:
        """Analyze concentration risk.

        Returns:
            Dictionary with concentration metrics.
        """
        total_simm = sum(self.by_risk_class.values())

        # Risk class concentration
        risk_class_concentration = {
            rc: {
                "amount": amount,
                "pct": (amount / total_simm * 100 if total_simm != 0 else 0)
            }
            for rc, amount in self.by_risk_class.items()
        }

        # Position concentration
        sorted_positions = sorted(
            self.by_position.values(),
            key=lambda p: p.total_contribution,
            reverse=True
        )

        position_concentration = {
            "top_10_pct": sum(
                p.total_contribution for p in sorted_positions[:10]
            ) / total_simm * 100 if total_simm != 0 else 0,
            "top_20_pct": sum(
                p.total_contribution for p in sorted_positions[:20]
            ) / total_simm * 100 if total_simm != 0 else 0,
            "total_positions": len(sorted_positions)
        }

        return {
            "risk_class_concentration": risk_class_concentration,
            "position_concentration": position_concentration
        }
