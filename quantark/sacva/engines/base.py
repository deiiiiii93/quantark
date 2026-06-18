"""Generic SA-CVA sensitivity-based aggregation engine (MAR50.51-50.53).

Subclasses supply risk weights, correlations, and bucketing; the base performs
risk-factor netting (design spec §2.0) and the bucket/cross-bucket aggregation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Hashable, List

from quantark.sacva.models.enums import RiskType
from quantark.sacva.models.sensitivity import CVASensitivity
from quantark.sacva.parameters.supervisory import SupervisoryParameters
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import safe_sqrt


@dataclass
class _Factor:
    """A netted regulatory risk factor within a bucket."""
    ws: float          # WS_k = RW*(s_cva - s_hdg)
    ws_hdg: float      # WS_k^Hdg = RW*s_hdg


@dataclass
class RiskClassResult:
    k: float
    by_bucket: Dict = field(default_factory=dict)
    bucket_s_b: Dict = field(default_factory=dict)
    bucket_sum_ws: Dict = field(default_factory=dict)
    hedge_disallowance: Dict = field(default_factory=dict)


class BaseRiskClassEngine(ABC):
    """Abstract SA-CVA risk-class engine."""

    R = SupervisoryParameters.R_HEDGE_DISALLOWANCE

    @abstractmethod
    def supports(self, risk_type: RiskType) -> bool: ...

    @abstractmethod
    def is_single_factor(self) -> bool: ...

    @abstractmethod
    def factor_identity(self, s: CVASensitivity) -> Hashable: ...

    @abstractmethod
    def risk_weight(self, s: CVASensitivity, risk_type: RiskType) -> float: ...

    @abstractmethod
    def group_buckets(self, sens: List[CVASensitivity], reporting_currency: str) -> Dict:
        """Return {bucket_key: [sensitivities]}."""

    @abstractmethod
    def intra_correlation(self, a: CVASensitivity, b: CVASensitivity) -> float: ...

    @abstractmethod
    def cross_bucket_correlation(self, b, c) -> float: ...

    # ------------------------------------------------------------------
    def calculate(self, sensitivities: List[CVASensitivity], risk_type: RiskType,
                  reporting_currency: str) -> RiskClassResult:
        if not self.supports(risk_type):
            return RiskClassResult(k=0.0)
        sens = [s for s in sensitivities if s.risk_type == risk_type]
        if not sens:
            return RiskClassResult(k=0.0)

        buckets = self.group_buckets(sens, reporting_currency)
        kb_sq_sum = 0.0
        cross = 0.0
        s_b_by_bucket: Dict = {}
        result = RiskClassResult(k=0.0)

        for bkey, blist in buckets.items():
            factors, reps = self._net_factors(blist, risk_type)
            kb, sum_ws, disallow = self._bucket_k(factors, reps)
            s_b = max(-kb, min(sum_ws, kb))
            result.by_bucket[bkey] = kb
            result.bucket_s_b[bkey] = s_b
            result.bucket_sum_ws[bkey] = sum_ws
            result.hedge_disallowance[bkey] = disallow
            s_b_by_bucket[bkey] = s_b
            kb_sq_sum += kb * kb

        keys = list(buckets.keys())
        for i, b in enumerate(keys):
            for c in keys[i + 1:]:
                cross += 2.0 * self.cross_bucket_correlation(b, c) \
                    * s_b_by_bucket[b] * s_b_by_bucket[c]

        result.k = float(safe_sqrt(max(0.0, kb_sq_sum + cross)))
        return result

    # ------------------------------------------------------------------
    def _net_factors(self, blist: List[CVASensitivity], risk_type: RiskType):
        """Net rows into regulatory factors. Returns (factors, reps).

        ``factors``: list[_Factor]; ``reps``: parallel list of representative
        CVASensitivity (for intra-rho metadata). For single-factor classes the
        whole bucket collapses to one factor.
        """
        groups: Dict[Hashable, List[CVASensitivity]] = {}
        for s in blist:
            key = "ALL" if self.is_single_factor() else self.factor_identity(s)
            groups.setdefault(key, []).append(s)

        factors: List[_Factor] = []
        reps: List[CVASensitivity] = []
        for rows in groups.values():
            rep = rows[0]
            self._check_consistent(rows, rep)
            rw = self.risk_weight(rep, risk_type)
            s_cva = sum(r.s_cva for r in rows)
            s_hdg = sum(r.s_hdg for r in rows)
            ws = rw * (s_cva - s_hdg)
            ws_hdg = rw * s_hdg
            factors.append(_Factor(ws=ws, ws_hdg=ws_hdg))
            reps.append(rep)
        return factors, reps

    def consistency_attrs(self) -> tuple:
        """Attributes that netted rows for one factor must agree on.

        Default is empty: for single-factor risk classes the whole bucket nets and
        RW/correlation depend only on the bucket, so per-row metadata (e.g. an
        optional ``credit_quality``) need not match. Multi-factor engines override
        this to enforce the RW/correlation-driving attributes not already pinned by
        ``factor_identity``.
        """
        return ()

    def _check_consistent(self, rows, rep):
        """Reject netted rows that disagree on RW/correlation-driving attributes."""
        for r in rows:
            for attr in self.consistency_attrs():
                if getattr(r, attr) != getattr(rep, attr):
                    raise ValidationError(
                        f"Inconsistent {attr} among netted rows for factor "
                        f"{rep.risk_factor!r}")

    def _bucket_k(self, factors: List[_Factor], reps: List[CVASensitivity]):
        n = len(factors)
        sum_ws = sum(f.ws for f in factors)
        disallow = self.R * sum(f.ws_hdg ** 2 for f in factors)
        quad = sum(f.ws ** 2 for f in factors)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                rho = self.intra_correlation(reps[i], reps[j])
                quad += rho * factors[i].ws * factors[j].ws
        kb = float(safe_sqrt(max(0.0, quad) + disallow))
        return kb, sum_ws, disallow
