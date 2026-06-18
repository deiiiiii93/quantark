"""Total Return Swap cashflow pricing engine.

The engine walks an observed asset price path over a trading calendar and builds
the realized cashflows of the swap: a per-date notional schedule, the fixed
(financing) leg, the floating (total-return) leg, and the combined present value
and margin mark-to-market.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from quantark.util.exceptions import MarketDataError, ValidationError
from quantark.util.numerical import safe_divide
from quantark.asset.equity.product.swap.trs_params import (
    TRSParams,
    AccrualType,
    AccrualSide,
    SettleType,
)
from quantark.asset.equity.engine.cashflow.accrual_calculator import (
    AccrualCalculatorFactory,
)


class TotalReturnSwapEngine:
    """Realized-cashflow pricing engine for a single-asset TRS."""

    @staticmethod
    def _price_at(params: TRSParams, date: str) -> float:
        """Return the observed asset price on ``date`` or raise ``MarketDataError``."""
        prices = params.asset.asset_prices
        if date not in prices.index:
            raise MarketDataError(
                f"asset_prices is missing a required pivot date: {date!r}"
            )
        return prices[date]

    def cal_interest_by_date_range(
        self,
        notional: float,
        start_date: str,
        end_date: str,
        interest_rate: float,
        calendar: Any,
        side: str = "left",
        day_count_basis: str = "act/365",
        force_no_zero: bool = False,
        **kwargs: Any,
    ) -> float:
        """Convenience: accrue simple interest on a fixed notional over a range."""
        calculator = AccrualCalculatorFactory.create_calculator(AccrualType.NOTIONAL)
        return calculator.calculate_accrual(
            notional, start_date, end_date, interest_rate, calendar, side,
            day_count_basis, force_no_zero, **kwargs,
        )

    def create_notional_schedule(self, params: TRSParams) -> List[Dict]:
        """
        Build the per-date notional schedule.

        Walks every calendar day from contract start to contract end, applying
        fees, dividends, share dividends, maturity and redemption events, and
        margin movements, and records the resulting notional/quantity state.
        """
        contract_start_date = min(
            params.fix_leg.start_date, params.float_leg.start_date
        )
        contract_end_date = max(params.fix_leg.end_date, params.float_leg.end_date)

        date_range = params.fix_leg.payment_calendar.get_calendar_days(
            contract_start_date, contract_end_date, side="both"
        )

        all_events = params.events.events or {}
        # Copy the redemption list so the maturity redemption appended below never
        # mutates the caller's shared event list across repeated pricing calls.
        redm_events = list(all_events.get("redm", []))
        cash_div_events = all_events.get("div_cash")
        share_div_events = all_events.get("div_share")
        upfront_fee_events = all_events.get("upfront_fee")
        unwind_fee_events = all_events.get("unwind_fee")

        notional_schedule: List[Dict] = []
        fix_notional = params.fix_leg.initial_notional
        float_notional = params.float_leg.initial_notional
        asset_quantity = safe_divide(
            params.float_leg.initial_notional, params.asset.asset_initial_price
        )
        asset_initial_price = params.asset.asset_initial_price

        cash_div_accrual_cum = 0.0
        outstanding_margin = params.margin.outstanding_margin

        for date in date_range:
            pivot_date = date.strftime("%Y-%m-%d")

            contract_event: List[str] = []
            redeem_fee = 0.0
            redeem_notional = 0.0
            redeem_ratio = 0.0
            cash_div_realized = 0.0
            asset_interest_realized = 0.0
            fix_interest_realized = 0.0
            upfront_fee = 0.0
            unwind_fee = 0.0

            if upfront_fee_events is not None:
                for fee_event in upfront_fee_events:
                    if pivot_date == fee_event["date"]:
                        upfront_fee = self._compute_fee(
                            params, fee_event, "upfront_fee",
                            asset_initial_price, asset_quantity, pivot_date,
                        )
                        contract_event.append("UPFRONT_FEE")

            if unwind_fee_events is not None:
                for fee_event in unwind_fee_events:
                    if pivot_date == fee_event["date"]:
                        unwind_fee = self._compute_fee(
                            params, fee_event, "unwind_fee",
                            asset_initial_price, asset_quantity, pivot_date,
                        )
                        contract_event.append("UNWIND_FEE")

            if cash_div_events is not None:
                for div_event in cash_div_events:
                    if div_event.get("date") == pivot_date:
                        if div_event.get("cash_div_per_share") is None:
                            raise ValidationError(
                                "cash dividend event is missing required field "
                                f"'cash_div_per_share': {div_event!r}"
                            )
                        cash_div_per_share = div_event.get("cash_div_per_share")
                        deliver_ratio = div_event.get("deliver_ratio", 1.0)
                        cash_div_accrual_cum += (
                            cash_div_per_share
                            * asset_quantity
                            * deliver_ratio
                            * params.float_leg.direction
                        )
                        contract_event.append("CASH_DIV")

            if share_div_events is not None:
                for div_event in share_div_events:
                    if div_event.get("date") == pivot_date:
                        share_div_per_share = div_event.get("share_div_per_share")
                        asset_quantity = asset_quantity * (1 + share_div_per_share)
                        asset_initial_price = safe_divide(
                            asset_initial_price, (1 + share_div_per_share)
                        )
                        contract_event.append("SHARE_DIV")

            if pivot_date >= contract_end_date:
                contract_event.append("MAT")
                asset_price_at_maturity = self._price_at(
                    params, min(contract_end_date, params.pricing.valuation_date)
                )
                # Appended last so any explicit same-day redemptions are processed
                # first; the maturity event then redeems whatever notional remains
                # (resolved at processing time via the ``_maturity`` flag).
                redm_events.append(
                    {
                        "date": pivot_date,
                        "redeem_notional": None,
                        "redeem_price": asset_price_at_maturity,
                        "redeem_fee_rate": 0,
                        "redeem_settle_option": ["asset", "int", "cash_div"],
                        "_maturity": True,
                    }
                )

            for redm_event in redm_events:
                if redm_event.get("date") == pivot_date:
                    if redm_event.get("_maturity"):
                        # Redeem the outstanding notional remaining at maturity.
                        redeem_notional = fix_notional
                    else:
                        redeem_notional = redm_event.get("redeem_notional")
                    redeem_price = redm_event.get("redeem_price")
                    redeem_fee_rate = redm_event.get("redeem_fee_rate")
                    redeem_settle_option = redm_event.get("redeem_settle_option")
                    redeem_quantity = safe_divide(redeem_notional, asset_initial_price)
                    redeem_ratio = redm_event.get(
                        "redeem_ratio", safe_divide(redeem_quantity, asset_quantity)
                    )
                    fix_notional -= redeem_notional
                    float_notional -= redeem_notional
                    asset_quantity -= redeem_quantity
                    redeem_fee = redeem_fee_rate * redeem_price * redeem_quantity
                    asset_interest_realized = (
                        (redeem_price - asset_initial_price)
                        * params.float_leg.direction
                        * redeem_quantity
                        if "asset" in redeem_settle_option
                        else 0
                    )
                    cash_div_realized = (
                        redeem_ratio * cash_div_accrual_cum
                        if "cash_div" in redeem_settle_option
                        else 0
                    )
                    fix_interest_realized = (
                        redeem_ratio if "int" in redeem_settle_option else 0
                    )
                    cash_div_accrual_cum -= cash_div_realized
                    if "MAT" not in contract_event:
                        if abs(redeem_notional) > 0:
                            contract_event.append("REDM")
                        else:
                            if "cash_div" in redeem_settle_option:
                                contract_event.append("CASH_DIV")
                            if "int" in redeem_settle_option:
                                contract_event.append("INT")

            if params.margin.margin_events:
                for event in params.margin.margin_events:
                    if event.get("date") == pivot_date:
                        outstanding_margin += event.get("amount") * (
                            1 if event.get("type").lower() == "in" else -1
                        )
                        contract_event.append("MARGIN_" + event.get("type").upper())

            notional_schedule.append(
                {
                    "date": pivot_date,
                    "fix_notional": fix_notional,
                    "float_notional": float_notional,
                    "asset_quantity": asset_quantity,
                    "asset_initial_price": asset_initial_price,
                    "redeem_notional": redeem_notional,
                    "redeem_fee": redeem_fee,
                    "fix_interest_realized": fix_interest_realized,
                    "upfront_fee": upfront_fee,
                    "unwind_fee": unwind_fee,
                    "asset_interest_realized": asset_interest_realized,
                    "cash_div_accrual": cash_div_accrual_cum,
                    "cash_div_realized": cash_div_realized,
                    "outstanding_margin": outstanding_margin,
                    "contract_event": contract_event,
                }
            )
        return notional_schedule

    @staticmethod
    def _compute_fee(
        params: TRSParams, fee_event: Dict, fee_kind: str,
        asset_initial_price: float, asset_quantity: float, pivot_date: str,
    ) -> float:
        """Compute an upfront/unwind fee for the matched event date."""
        rate = fee_event[f"{fee_kind}_rate"]
        fee_type = fee_event[f"{fee_kind}_type"].lower()
        if fee_type == "notional":
            return asset_initial_price * rate * asset_quantity
        if fee_type == "marketvalue":
            return (
                TotalReturnSwapEngine._price_at(params, pivot_date)
                * rate * asset_quantity
            )
        if fee_type == "initial_notional":
            return params.float_leg.initial_notional * rate
        raise ValidationError(
            f"invalid {fee_kind}_type: {fee_type!r}; expected one of "
            f"'notional', 'marketvalue', 'initial_notional'"
        )

    @staticmethod
    def _validate_business_day_maturity(params: TRSParams) -> None:
        """Reject a non-business-day contract maturity.

        The per-period leg tables are built from working days and merged onto the
        notional schedule by exact pivot date. A maturity that falls on a
        weekend/holiday would therefore have no matching pricing row, silently
        dropping the maturity redemption, realized interest and dividends. Rather
        than mis-price, fail loudly; callers should set the contract end to the
        intended settlement trading date.
        """
        calendar = params.fix_leg.payment_calendar
        contract_end_date = max(params.fix_leg.end_date, params.float_leg.end_date)
        end_dt = calendar._to_datetime(contract_end_date)
        if not calendar.is_business_day(end_dt):
            raise ValidationError(
                f"contract maturity {contract_end_date!r} is not a business day; "
                "set the contract end to a trading (settlement) date"
            )

    @staticmethod
    def _effective_valuation_date(params: TRSParams) -> str:
        """Cap the valuation date at contract maturity.

        The notional schedule is only built through the contract end date, so a
        valuation date past maturity (a matured contract) must not extend the leg
        pricing range beyond it.
        """
        contract_end_date = max(params.fix_leg.end_date, params.float_leg.end_date)
        return min(params.pricing.valuation_date, contract_end_date)

    def price_fixed_leg(
        self, params: TRSParams, notional_schedule: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Compute the fixed (financing) leg accrual schedule."""
        effective_val = self._effective_valuation_date(params)
        if params.fix_leg.accrual_type == AccrualType.LAST_MARKET_VALUE:
            range_start = params.fix_leg.payment_calendar.get_next_trading_date(
                params.fix_leg.start_date, n=-1, only_holidays=False
            )
            range_end = effective_val
        else:
            range_start = params.fix_leg.start_date
            range_end = params.fix_leg.payment_calendar.get_next_trading_date(
                effective_val, only_holidays=False
            )

        date_range = params.fix_leg.payment_calendar.get_working_days(
            range_start, range_end
        )

        fix_leg_price_res: List[Dict] = []

        asset_quantity_schedule = {
            item["date"]: item["asset_quantity"] for item in notional_schedule
        }
        asset_initial_price_schedule = {
            item["date"]: item["asset_initial_price"] for item in notional_schedule
        }
        redeem_int_schedule = {
            item["date"]: item["fix_interest_realized"] for item in notional_schedule
        }

        last_asset_quantity = asset_quantity_schedule[params.fix_leg.start_date]
        last_asset_price = self._price_at(params, params.fix_leg.start_date)

        accrual_interest_cum = 0.0
        calculator = AccrualCalculatorFactory.create_calculator(
            params.fix_leg.accrual_type
        )

        for period_start, period_end in zip(date_range[:-1], date_range[1:]):
            period_start_str = max(
                period_start.strftime("%Y-%m-%d"), params.fix_leg.start_date
            )
            period_end_str = period_end.strftime("%Y-%m-%d")

            pivot_date = (
                period_end_str
                if params.fix_leg.accrual_type == AccrualType.LAST_MARKET_VALUE
                else period_start_str
            )

            asset_quantity = asset_quantity_schedule[pivot_date]
            asset_initial_price = asset_initial_price_schedule[pivot_date]

            accrual_kwargs: Dict[str, Any] = {}
            if params.fix_leg.accrual_type == AccrualType.NOTIONAL:
                accrual_notional = asset_initial_price * asset_quantity
            elif params.fix_leg.accrual_type == AccrualType.MARKET_VALUE:
                accrual_notional = 0
                accrual_kwargs = {
                    "asset_quantity": asset_quantity,
                    "asset_price": self._price_at(params, pivot_date),
                }
            elif params.fix_leg.accrual_type == AccrualType.LAST_MARKET_VALUE:
                accrual_notional = 0
                accrual_kwargs = {
                    "last_asset_quantity": last_asset_quantity,
                    "last_asset_price": last_asset_price,
                }
            else:
                accrual_notional = params.fix_leg.notional

            accrual_interest = calculator.calculate_accrual(
                accrual_notional, period_start_str, period_end_str,
                params.fix_leg.rate, params.fix_leg.payment_calendar,
                day_count_basis=params.pricing.interest_basis,
                force_no_zero=True, **accrual_kwargs,
            )

            if (
                params.fix_leg.accrual_side in (AccrualSide.RIGHT, AccrualSide.NEITHER)
                and pivot_date == params.fix_leg.start_date
            ):
                accrual_interest = 0

            if (
                params.fix_leg.accrual_side in (AccrualSide.RIGHT, AccrualSide.BOTH)
                and period_end_str == params.fix_leg.end_date
            ):
                accrual_interest = calculator.calculate_accrual(
                    accrual_notional, period_start_str, period_end_str,
                    params.fix_leg.rate, params.fix_leg.payment_calendar,
                    side="both", day_count_basis=params.pricing.interest_basis,
                    force_no_zero=True, **accrual_kwargs,
                )

            accrual_interest_cum += accrual_interest

            accrual_interest_realized = 0.0
            redeem_int_ratio = redeem_int_schedule.get(pivot_date, 0)
            if redeem_int_ratio != 0:
                accrual_interest_realized = redeem_int_ratio * accrual_interest_cum
                accrual_interest_cum -= accrual_interest_realized

            last_asset_quantity = asset_quantity
            last_asset_price = self._price_at(params, pivot_date)

            fix_leg_price_res.append(
                {
                    "period_start": period_start_str,
                    "period_end": period_end_str,
                    "fix_notional": asset_initial_price * asset_quantity,
                    "accrual_interest": accrual_interest * params.fix_leg.direction,
                    "accrual_interest_cum": accrual_interest_cum
                    * params.fix_leg.direction,
                    "fix_interest_realized": accrual_interest_realized
                    * params.fix_leg.direction,
                }
            )

        return fix_leg_price_res

    def price_float_leg(
        self, params: TRSParams, notional_schedule: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Compute the floating (total-return) leg mark-to-market schedule."""
        effective_val = self._effective_valuation_date(params)
        if params.fix_leg.accrual_type == AccrualType.LAST_MARKET_VALUE:
            range_start = params.float_leg.payment_calendar.get_next_trading_date(
                params.float_leg.start_date, n=-1, only_holidays=False
            )
            range_end = effective_val
        else:
            range_start = params.float_leg.start_date
            range_end = params.float_leg.payment_calendar.get_next_trading_date(
                effective_val, only_holidays=False
            )

        date_range = params.float_leg.payment_calendar.get_working_days(
            range_start, range_end
        )

        asset_quantity_schedule = {
            item["date"]: item["asset_quantity"] for item in notional_schedule
        }
        asset_initial_price_schedule = {
            item["date"]: item["asset_initial_price"] for item in notional_schedule
        }

        float_leg_price_res: List[Dict] = []
        for period_start, period_end in zip(date_range[:-1], date_range[1:]):
            period_start_str = max(
                period_start.strftime("%Y-%m-%d"), params.float_leg.start_date
            )
            period_end_str = period_end.strftime("%Y-%m-%d")

            pivot_date = (
                period_end_str
                if params.fix_leg.accrual_type == AccrualType.LAST_MARKET_VALUE
                else period_start_str
            )

            asset_quantity = asset_quantity_schedule[pivot_date]
            asset_initial_price = asset_initial_price_schedule[pivot_date]

            asset_price = self._price_at(params, pivot_date)
            market_value = asset_price * asset_quantity
            float_interest = market_value - asset_initial_price * asset_quantity

            float_leg_price_res.append(
                {
                    "period_start": period_start_str,
                    "period_end": period_end_str,
                    "float_notional": asset_initial_price * asset_quantity,
                    "asset_price": asset_price,
                    "market_value": market_value * params.float_leg.direction,
                    "float_interest": float_interest * params.float_leg.direction,
                }
            )

        return float_leg_price_res

    def price(self, params: TRSParams, precision: int = 2) -> pd.DataFrame:
        """Compute the combined per-period cashflow table and present value."""
        self._validate_business_day_maturity(params)
        notional_schedule = self.create_notional_schedule(params)
        fix_leg_price_res = self.price_fixed_leg(params, notional_schedule)
        float_leg_price_res = self.price_float_leg(params, notional_schedule)

        df_schedule = pd.DataFrame(notional_schedule)
        df_fix = pd.DataFrame(fix_leg_price_res).set_index(
            ["period_start", "period_end"]
        )
        df_float = pd.DataFrame(float_leg_price_res).set_index(
            ["period_start", "period_end"]
        )
        df_price = pd.concat([df_fix, df_float], axis=1).reset_index()

        df_price["present_value"] = (
            df_price["accrual_interest_cum"] + df_price["float_interest"]
        )

        on_column = (
            "period_end"
            if params.fix_leg.accrual_type == AccrualType.LAST_MARKET_VALUE
            else "period_start"
        )

        df_price = pd.merge(
            df_price, df_schedule, how="left",
            left_on=on_column, right_on="date", suffixes=("", "_schedule"),
        )

        df_price["present_value"] += df_price["cash_div_accrual"]

        if params.margin.settle_type == SettleType.MARGIN:
            df_price["outstanding_margin"] += (
                df_price["asset_interest_realized"].cumsum()
                + df_price["fix_interest_realized"].cumsum()
                + df_price["cash_div_realized"].cumsum()
                - df_price["redeem_fee"].cumsum()
                - df_price["upfront_fee"].cumsum()
                - df_price["unwind_fee"].cumsum()
            )

        df_price["margin_mtm"] = (
            df_price["outstanding_margin"] + df_price["present_value"]
        )

        cols_to_drop = [col for col in df_price.columns if "_schedule" in col]
        df_price = df_price.drop(columns=cols_to_drop)

        if params.pricing.output_mode.lower() == "full":
            return df_price.round(precision).fillna("-")
        return df_price.iloc[-1].to_frame().T.round(precision).fillna("-")
