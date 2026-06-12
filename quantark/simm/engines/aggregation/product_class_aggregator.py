"""
Product class aggregation for SIMM.

Implements paragraph 6: within each product class, the initial margin for
each of the six risk classes is combined as

    SIMM_product = sqrt( sum_r IM_r^2 + sum_r sum_{s != r} psi_rs IM_r IM_s )

with the correlation matrix psi_rs given in Section K. The total SIMM is
the sum of the four product class SIMM values.
"""

import math
from dataclasses import dataclass, field
from typing import Dict

from quantark.simm.taxonomy import MarginType, ProductClass, RiskClass
from quantark.simm.engines.aggregation.risk_class_aggregator import RiskClassResult
from quantark.simm.calibration.accessors import get_inter_risk_class_correlation


@dataclass
class ProductClassResult:
    """Result of product class aggregation.

    Attributes:
        product_class: The product class.
        margin: SIMM_product (before any multiplicative scale).
        risk_class_margins: Total IM per risk class within this product
            class (Delta + Vega + Curvature + BaseCorr).
        margin_type_detail: Per risk class, the margin by margin type.
    """
    product_class: ProductClass
    margin: float
    risk_class_margins: Dict[RiskClass, float] = field(default_factory=dict)
    margin_type_detail: Dict[RiskClass, Dict[MarginType, float]] = field(default_factory=dict)


class ProductClassAggregator:
    """Aggregator across risk classes within a product class."""

    def aggregate(
        self,
        product_class: ProductClass,
        risk_class_results: Dict[RiskClass, Dict[MarginType, RiskClassResult]],
    ) -> ProductClassResult:
        """Aggregate risk class margins into the product class SIMM.

        Args:
            product_class: The product class.
            risk_class_results: Margin results per risk class and margin
                type, computed from the sensitivities of trades in this
                product class only (paragraph 6).

        Returns:
            ProductClassResult with SIMM_product.
        """
        risk_class_margins: Dict[RiskClass, float] = {}
        margin_type_detail: Dict[RiskClass, Dict[MarginType, float]] = {}

        for risk_class, by_margin_type in risk_class_results.items():
            margin_type_detail[risk_class] = {
                mt: r.margin for mt, r in by_margin_type.items()
            }
            # IM_X = DeltaMargin + VegaMargin + CurvatureMargin
            #        (+ BaseCorrMargin for Credit Qualifying), paragraph 5.
            risk_class_margins[risk_class] = sum(
                r.margin for r in by_margin_type.values()
            )

        classes = [rc for rc, im in risk_class_margins.items() if im != 0.0]
        total = 0.0
        for i, rc_i in enumerate(classes):
            im_i = risk_class_margins[rc_i]
            total += im_i ** 2
            for rc_j in classes[i + 1:]:
                psi = get_inter_risk_class_correlation(rc_i, rc_j)
                total += 2.0 * psi * im_i * risk_class_margins[rc_j]

        return ProductClassResult(
            product_class=product_class,
            margin=math.sqrt(max(0.0, total)),
            risk_class_margins=risk_class_margins,
            margin_type_detail=margin_type_detail,
        )
