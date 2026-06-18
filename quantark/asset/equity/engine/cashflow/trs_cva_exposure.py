"""Pathwise SA-CVA exposure engine for path-dependent (market-value) TRS.

The default :class:`MonteCarloExposureEngine` reprices each trade as a *Markovian*
value surface (one spot column per node). A market-value-financing TRS is
path-dependent — its future financing is the spot time-integral — so it must be
valued over the *whole* simulated path. This thin subclass overrides the per-trade
value step: a trade whose engine advertises ``supports_pathwise_value`` is valued
by handing it the full spot-path array (``value_paths``); every other trade falls
through to the unchanged Markovian path, so vanilla trades and option co-nettings
behave exactly as before.

This is the opt-in carrier for the explicit financing **approximation** in
:class:`~quantark.asset.equity.engine.cashflow.trs_cva_repricer.TRSPathwiseCVARepricer`
— construct ``SACVAEngine(exposure_engine=TRSPathwiseExposureEngine(...))`` to use
it. The aggregation, netting, discounting and SBA capital path are entirely
unchanged.
"""

from quantark.sacva.exposure.simulator import MonteCarloExposureEngine
from quantark.util.exceptions import ValidationError


class TRSPathwiseExposureEngine(MonteCarloExposureEngine):
    """MC exposure engine that values ``supports_pathwise_value`` trades pathwise."""

    def _trade_value_array(self, trade, paths, times):
        if getattr(trade.engine, "supports_pathwise_value", False):
            valuer = getattr(trade.engine, "value_paths", None)
            if valuer is None:
                raise ValidationError(
                    f"{trade.trade_id}: engine advertises supports_pathwise_value "
                    "but has no value_paths method"
                )
            key = self._underlying_key(trade)
            values = valuer(trade.product, paths[key], times, trade.env)
            return values * float(trade.quantity)
        return super()._trade_value_array(trade, paths, times)
