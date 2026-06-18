"""Multi-asset Total Return Swap: a basket of single-asset TRS under one contract."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import safe_divide
from quantark.asset.equity.product.swap.trs_params import (
    TRSParams,
    AssetParams,
    FixLegParams,
    FloatLegParams,
    EventParams,
    MarginParams,
)
from quantark.asset.equity.product.swap.one_asset_trs import OneAssetTotalReturnSwap


@dataclass
class MultiAssetInfo:
    """One leg of a multi-asset TRS basket."""

    asset_id: str
    asset_initial_price: float
    asset_prices: pd.Series
    notional: float
    long_or_short: int = 1  # +1 long, -1 short
    fix_rate: Optional[float] = None  # falls back to the contract fixed rate


class MultiAssetTRS:
    """A basket of single-asset Total Return Swaps sharing one contract."""

    def __init__(self, params: TRSParams, assets: List[MultiAssetInfo]):
        self._validate_assets(assets)

        self.contract_id = params.contract_id
        self.fix_rate = params.fix_leg.rate
        self.valuation_date = params.pricing.valuation_date
        self.output_mode = params.pricing.output_mode

        self.asset_notionals = [asset.notional for asset in assets]
        self.total_notional = sum(self.asset_notionals)
        self.fix_notional = self.total_notional
        self.float_notional = self.total_notional

        self.single_asset_trs_list: List[OneAssetTotalReturnSwap] = []
        self._create_single_asset_trs_list(params, assets)
        self.margin_account = self._create_merged_margin_account()

    @staticmethod
    def _validate_assets(assets: List[MultiAssetInfo]) -> None:
        if not assets:
            raise ValidationError("assets must not be empty")
        for asset in assets:
            if asset.long_or_short not in (1, -1):
                raise ValidationError("long_or_short must be either 1 or -1")

    def _create_single_asset_trs_list(
        self, params: TRSParams, assets: List[MultiAssetInfo]
    ) -> None:
        for i, asset in enumerate(assets):
            asset_float_direction = params.float_leg.direction * asset.long_or_short

            asset_params = AssetParams(
                asset_id=asset.asset_id,
                asset_initial_price=asset.asset_initial_price,
                asset_prices=asset.asset_prices,
                asset_price_precision=params.asset.asset_price_precision,
            )
            fix_leg_params = FixLegParams(
                rate=asset.fix_rate if asset.fix_rate is not None else params.fix_leg.rate,
                tenor=params.fix_leg.tenor,
                day_count=params.fix_leg.day_count,
                notional=asset.notional,
                initial_notional=asset.notional,
                start_date=params.fix_leg.start_date,
                end_date=params.fix_leg.end_date,
                payment_lag=params.fix_leg.payment_lag,
                payment_calendar=params.fix_leg.payment_calendar,
                accrual_type=params.fix_leg.accrual_type,
                accrual_side=params.fix_leg.accrual_side,
                direction=params.fix_leg.direction,
            )
            float_leg_params = FloatLegParams(
                rate=params.float_leg.rate,
                tenor=params.float_leg.tenor,
                day_count=params.float_leg.day_count,
                notional=asset.notional,
                initial_notional=asset.notional,
                start_date=params.float_leg.start_date,
                end_date=params.float_leg.end_date,
                payment_lag=params.float_leg.payment_lag,
                payment_calendar=params.float_leg.payment_calendar,
                direction=asset_float_direction,
            )
            weight = safe_divide(asset.notional, self.total_notional)
            margin_params = MarginParams(
                initial_margin=(
                    params.margin.initial_margin * weight
                    if params.margin.initial_margin > 0
                    else 0
                ),
                outstanding_margin=(
                    params.margin.outstanding_margin * weight
                    if params.margin.outstanding_margin > 0
                    else 0
                ),
                margin_events=self._filter_margin_events_for_asset(
                    params.margin.margin_events, asset.asset_id, weight
                ),
                settle_type=params.margin.settle_type,
            )

            event_params = EventParams()
            if params.events and params.events.events:
                filtered = self._filter_events_for_asset(
                    params.events.events, asset.asset_id
                )
                if filtered:
                    event_params = EventParams(
                        dividend_events=filtered.get("div_cash", []),
                        share_dividend_events=filtered.get("div_share", []),
                        redemption_events=filtered.get("redm", []),
                        upfront_fee_events=filtered.get("upfront_fee", []),
                        unwind_fee_events=filtered.get("unwind_fee", []),
                    )

            single_params = TRSParams(
                contract_id=f"{self.contract_id}_{i + 1}",
                asset=asset_params,
                fix_leg=fix_leg_params,
                float_leg=float_leg_params,
                events=event_params,
                margin=margin_params,
                pricing=params.pricing,
            )
            self.single_asset_trs_list.append(OneAssetTotalReturnSwap(single_params))

    @staticmethod
    def _filter_events_for_asset(events: Dict, asset_id: str) -> Optional[Dict]:
        if not events:
            return None
        filtered: Dict[str, List[Dict]] = {}
        for key in ("div_cash", "div_share", "redm", "upfront_fee", "unwind_fee"):
            if key in events:
                filtered[key] = [
                    e for e in events[key] if e.get("asset_id") == asset_id
                ]
        return filtered if any(filtered.values()) else None

    @staticmethod
    def _filter_margin_events_for_asset(
        margin_events: List[Dict], asset_id: str, weight: float
    ) -> List[Dict]:
        if not margin_events:
            return []
        result: List[Dict] = []
        for event in margin_events:
            if event.get("asset_id") == asset_id:
                result.append(event)
            elif event.get("asset_id") is None:
                # Contract-level event: allocate the amount by notional weight.
                allocated = dict(event)
                allocated["amount"] = event.get("amount", 0) * weight
                result.append(allocated)
        return result

    def _create_merged_margin_account(self) -> Dict:
        initial_margin = sum(
            trs.margin_account.initial_margin for trs in self.single_asset_trs_list
        )
        outstanding_margin = sum(
            trs.margin_account.outstanding_margin
            for trs in self.single_asset_trs_list
        )
        history = pd.concat(
            [trs.margin_account.history for trs in self.single_asset_trs_list],
            ignore_index=True,
        ).sort_values(by="date")
        return {
            "initial_margin": initial_margin,
            "outstanding_margin": outstanding_margin,
            "history": history,
        }

    def price(self, precision: int = 2) -> pd.DataFrame:
        """Price each single-asset TRS and aggregate the basket."""
        price_results = []
        for i, trs in enumerate(self.single_asset_trs_list):
            result = trs.price(precision)
            result["asset_id"] = trs.params.asset.asset_id
            result["notional"] = self.asset_notionals[i]
            result["notional_ratio"] = safe_divide(
                self.asset_notionals[i], self.total_notional
            )
            price_results.append(result)

        if self.output_mode.lower() == "full":
            merged = self._merge_full_results(price_results)
        else:
            merged = self._merge_spot_results(price_results)
        return merged.round(precision).fillna("-")

    def _merge_full_results(self, price_results: List[pd.DataFrame]) -> pd.DataFrame:
        # Align on period_start only - the date each asset's row is written to.
        # Including period_end here would create blank rows for any period end
        # that is not also a period start.
        all_dates = set()
        for result in price_results:
            if "period_start" in result.columns:
                all_dates.update(result["period_start"])

        merged_df = pd.DataFrame(index=sorted(all_dates))
        merged_df.index.name = "date"

        skip_cols = {
            "period_start", "period_end", "asset_id", "notional", "notional_ratio",
        }
        for result in price_results:
            asset_id = result["asset_id"].iloc[0]
            if "period_start" in result.columns:
                temp = result.set_index("period_start")
                for col in result.columns:
                    if col not in skip_cols:
                        merged_df[f"{col}_{asset_id}"] = temp[col]

        self._calculate_total_values(merged_df)
        return merged_df.reset_index()

    def _merge_spot_results(self, price_results: List[pd.DataFrame]) -> pd.DataFrame:
        merged_df = pd.concat(price_results, ignore_index=True)

        def _col_sum(name):
            return (
                pd.to_numeric(merged_df[name], errors="coerce").sum()
                if name in merged_df.columns
                else None
            )

        total_row = {
            "contract_id": self.contract_id,
            "valuation_date": self.valuation_date,
            # Sum the per-asset current notionals so the total reflects
            # redemptions to date rather than the initial basket notional.
            "fix_notional": _col_sum("fix_notional"),
            "float_notional": _col_sum("float_notional"),
            "present_value": _col_sum("present_value"),
            "margin_mtm": _col_sum("margin_mtm"),
        }
        return pd.concat([merged_df, pd.DataFrame([total_row])], ignore_index=True)

    @staticmethod
    def _calculate_total_values(merged_df: pd.DataFrame) -> None:
        for prefix in (
            "present_value", "margin_mtm", "fix_notional", "float_notional",
        ):
            cols = [c for c in merged_df.columns if c.startswith(f"{prefix}_")]
            if cols:
                merged_df[f"{prefix}_total"] = merged_df[cols].sum(axis=1)
