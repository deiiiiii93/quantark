# -*- coding: utf-8 -*-
"""
Created on Thu May 30 2024

@author: yaofuxin
@description: Price snowball option using quadrature method
"""

import numpy as np
from datetime import datetime, timedelta
from asset.equity.engine.bsm.quad.option_quad import (
    price_one_touch,
    price_single_barrier,
    price_double_barrier,
    price_double_touch,
)
import pandas as pd
from asset.equity.engine.bsm.quad.option_greeks import (
    calc_delta,
    calc_gamma,
    calc_vega,
    calc_rho,
    calc_rho_q,
    calc_theta,
)

from asset.equity.engine.bsm.quad.util import (
    count_calendar_days,
    count_trading_days,
)

from util.enum.XFinanceEnums import ObsFreqType


def price_snowball(
    bus_days: int,
    cal_days: int,
    ko_bus_days: list[int],
    ko_cal_days: list[int],
    ko_prices: list[float],
    ki_price: float,
    spot: float,
    r: float,
    q: float,
    vol: float,
    notional: float,
    ko_rate: float,
    coupon_rate: float,
    initial: float,
    strike: float,
    grid_x: int = 1000,
    grid_t: int = None,
    participation_rate: float = 1.0,
    is_knocked_in: bool = False,
    protection_rate: float = 0.0,
    day_count_basis: dict = {"calendar": 365, "trading": 244},
    ki_obs_type: ObsFreqType = ObsFreqType.DAILY,
    ki_bus_days: list[int] = None,
    is_coupon_annualized: bool = True,
    is_ki_annualized: bool = False,
    market_spot: float = 0,
) -> dict:
    """使用数值积分方法定价雪球期权。

    Args:
        bus_days: 交易日天数
        cal_days: 日历日天数
        ko_bus_days: 敲出观察交易日天数列表
        ko_prices: 敲出价格列表，与敲出观察日期一一对应
        ki_price: 敲入价格
        spot: 标的资产现价
        r: 无风险利率（年化）
        q: 股息率（年化）
        vol: 波动率（年化）
        notional: 名义本金
        ko_rate: 敲出收益率（年化）
        coupon_rate: 红利票息率（年化）
        initial: 期初价格
        strike: 敲入期权行权价格
        grid_x: x方向（价格维度）网格点数，默认1000
        grid_t: t方向（时间维度）网格点数，默认为观察日期数量的4倍
        participation_rate: 敲入参与率，默认1.0
        is_knocked_in: 是否已敲入，默认False
        protection_rate: 保本率，0表示无保本，1表示全保本，默认0.0
        day_count_basis: 日计数基准，默认{"calendar": 365, "trading": 244}
        ki_obs_type: 敲入观察类型，支持"daily"、"euro"、"custom"，默认"daily"
        ki_bus_days: 自定义敲入观察交易日列表，当ki_obs_type为"custom"时必须提供
        is_coupon_annualized: 是否对票息收益进行年化计算，默认True
        is_ki_annualized: 是否对敲入收益进行年化计算，默认False
        market_spot: 实际标的资产现价 float，默认0

    Returns:
        dict: 包含以下键值对的字典：
            - total_value: 期权总价值
            - knockout_value: 敲出收益部分的价值（包含敲出收益）
            - knockin_value: 敲入风险部分的价值（如果is_ki_annualized为True则进行年化）
            - coupon_value: 红利票息部分的价值（如果is_coupon_annualized为True则进行年化）
    """
    # 计算期权参数
    maturity = bus_days / day_count_basis["trading"]  # 年化到期时间
    df = np.exp(-r * maturity)

    # 计算敲出观察日的年化时间
    ko_bus_days_annual = np.array(ko_bus_days) / day_count_basis["trading"]

    # 计算敲出观察日的票息计算系数
    ko_cal_days_annual = (
        (np.array(ko_cal_days) / day_count_basis["calendar"])
        if is_coupon_annualized
        else np.ones(len(ko_bus_days))
    )

    if grid_t is None:
        grid_t = len(ko_bus_days) * 4  # 默认每个观察日期4个时间步长

    # 标准化价格
    if market_spot == 0:
        market_spot = spot
    ko_prices = np.array(ko_prices) / market_spot
    ki_price = ki_price / market_spot
    strike = strike / market_spot
    initial = initial / market_spot
    spot = spot / market_spot  # 标准化后的现价

    # 保本属性
    protection_type = (
        "full"
        if protection_rate >= 1
        else "none" if np.isclose(protection_rate, 0, 1e-6) else "partial"
    )
    ki_price = np.inf if protection_type == "full" else ki_price

    # 1. 计算敲出部分的价值
    knockout_value = 0
    ko_df = np.exp(-r * ko_bus_days_annual)
    ko_rate = np.array(ko_rate)
    ko_payoff = ko_rate * ko_cal_days_annual * ko_df

    knockout_value += price_one_touch(
        grid_x,
        grid_t,
        maturity,
        ko_prices,
        ko_bus_days_annual,
        ko_cal_days_annual,
        ko_payoff,
        False,
        True,
        spot,
        r,
        q,
        vol,
    )

    # 2. 计算敲入部分的价值（向下敲入看跌期权 = 双障碍敲出看跌-向上单障碍敲出看跌）
    knockin_value = 0

    # 计算敲入观察时间
    options = {
        ObsFreqType.DAILY: np.array(
            [i / day_count_basis["trading"] for i in range(bus_days)]
        ),
        ObsFreqType.EUROPEAN: np.array([maturity]),
        ObsFreqType.CUSTOM: (
            np.array(ki_bus_days) / day_count_basis["trading"]
            if ki_bus_days is not None
            else ValueError("knockin obs dates should be input, as obs type is custom.")
        ),
    }

    if ki_obs_type not in options:
        raise ValueError(f"invalid obs type {ki_obs_type} input.")

    ki_bus_days_annual = options[ki_obs_type]

    knockin_value -= price_single_barrier(
        grid_x,
        grid_t,
        maturity,
        ko_prices,
        ko_bus_days_annual,
        "put",
        strike,
        spot,
        r,
        q,
        vol,
        "up",
    )

    if not is_knocked_in:
        knockin_value += price_double_barrier(
            grid_x,
            grid_t,
            maturity,
            ko_prices,
            ki_price,
            ko_bus_days_annual,
            ki_bus_days_annual,
            "put",
            strike,
            spot,
            r,
            q,
            vol,
        )

    if protection_type == "partial":
        knockin_value += price_single_barrier(
            grid_x,
            grid_t,
            maturity,
            ko_prices,
            ko_bus_days_annual,
            "put",
            initial * protection_rate,
            spot,
            r,
            q,
            vol,
        )

        if not is_knocked_in:
            knockin_value -= price_double_barrier(
                grid_x,
                grid_t,
                maturity,
                ko_prices,
                ki_price,
                ko_bus_days_annual,
                ki_bus_days_annual,
                "put",
                initial * protection_rate,
                spot,
                r,
                q,
                vol,
            )

    knockin_value *= df * participation_rate

    if is_ki_annualized:
        knockin_value *= cal_days / day_count_basis["calendar"]

    # 3.红利票息
    coupon_value = 0
    if not is_knocked_in:
        coupon_value = price_double_touch(
            grid_x,
            grid_t,
            maturity,
            ko_prices,
            ki_price,
            ko_bus_days_annual,
            ki_bus_days_annual,
            "no_touch",
            coupon_rate * ko_cal_days_annual[-1] * df,
            spot,
            r,
            q,
            vol,
        )

    # 合并结果
    total_value = (knockout_value + knockin_value + coupon_value) * notional

    return {
        "total_value": total_value,
        "knockout_value": knockout_value * notional,
        "knockin_value": knockin_value * notional,
        "coupon_value": coupon_value * notional,
    }


def calc_snowball_greeks(
    bus_days: int,
    cal_days: int,
    ko_bus_days: list[int],
    ko_cal_days: list[int],
    ko_prices: list[float],
    ki_price: float,
    spot: float,
    r: float,
    q: float,
    vol: float,
    notional: float,
    ko_rate: float,
    coupon_rate: float,
    initial: float,
    strike: float,
    grid_x: int = 1000,
    grid_t: int = None,
    participation_rate: float = 1.0,
    is_knocked_in: bool = False,
    protection_rate: float = 0.0,
    day_count_basis: dict = {"calendar": 365, "trading": 244},
    market_spot: float = 0.0,
    eps: dict = {
        "spot": 1e-2,  # 标的价格扰动
        "r": 1e-4,  # 利率扰动
        "q": 1e-4,  # 分红率扰动
        "vol": 1e-2,  # 波动率扰动
        "t": 1,  # 日期扰动
    },
) -> dict:
    """计算雪球期权的Greeks。

    Args:
        bus_days: 交易日天数
        cal_days: 日历日天数
        ko_bus_days: 敲出观察交易日天数列表
        ko_cal_days: 敲出观察日历日天数列表
        ko_prices: 敲出价格列表，与敲出观察日期一一对应
        ki_price: 敲入价格
        spot: 标的资产现价
        r: 无风险利率（年化）
        q: 股息率（年化）
        vol: 波动率（年化）
        notional: 名义本金
        ko_rate: 敲出收益率（年化）
        coupon_rate: 红利票息率（年化）
        initial: 期初价格
        strike: 敲入期权行权价格
        grid_x: x方向（价格维度）网格点数，默认1000
        grid_t: t方向（时间维度）网格点数，默认为观察日期数量的4倍
        participation_rate: 敲入参与率，默认1.0
        is_knocked_in: 是否已敲入，默认False
        protection_rate: 保本率，0表示无保本，1表示全保本，默认0.0
        day_count_basis: 日计数基准，默认{"calendar": 365, "trading": 244}
        market_spot: 实际标的资产现价，默认0.0
        eps: 各类扰动的大小，用于计算Greeks

    Returns:
        dict: 包含希腊字母的字典
    """
    # 准备基础参数字典，移除将作为独立参数传递的值
    base_kwargs = {
        "bus_days": bus_days,
        "cal_days": cal_days,
        "ko_bus_days": ko_bus_days,
        "ko_cal_days": ko_cal_days,
        "ko_prices": ko_prices,
        "ki_price": ki_price,
        "notional": notional,
        "ko_rate": ko_rate,
        "coupon_rate": coupon_rate,
        "initial": initial,
        "strike": strike,
        "grid_x": grid_x,
        "grid_t": grid_t,
        "participation_rate": participation_rate,
        "is_knocked_in": is_knocked_in,
        "protection_rate": protection_rate,
        "day_count_basis": day_count_basis,
        "market_spot": market_spot,
    }

    # 计算各类Greeks
    delta = calc_delta(
        price_snowball, spot, eps["spot"], **base_kwargs, r=r, q=q, vol=vol
    )
    gamma = calc_gamma(
        price_snowball, spot, eps["spot"], **base_kwargs, r=r, q=q, vol=vol
    )
    vega = calc_vega(
        price_snowball, vol, eps["vol"], **base_kwargs, spot=spot, r=r, q=q
    )
    theta = calc_theta(
        price_snowball, eps["t"], **base_kwargs, spot=spot, r=r, q=q, vol=vol
    )  # 1天的扰动
    rho = calc_rho(price_snowball, r, eps["r"], **base_kwargs, spot=spot, q=q, vol=vol)
    rho_q = calc_rho_q(
        price_snowball, q, eps["q"], **base_kwargs, spot=spot, r=r, vol=vol
    )

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,  # 每日theta
        "rho": rho,
        "rho_q": rho_q,
    }


def example_snowball():
    """运行雪球期权定价和Greeks计算的示例"""
    import time

    # 测试参数设置
    ko_bus_days = [
        64,
        86,
        106,
        127,
        148,
        168,
        189,
        210,
        230,
        251,
        276,
        295,
        316,
        337,
        358,
        378,
        399,
        419,
        440,
        460,
        481,
        502,
    ]
    ko_prices = [5587.29] * len(ko_bus_days)
    ko_cal_days = [
        127,
        148,
        168,
        189,
        210,
        230,
        251,
        276,
        295,
        316,
        337,
        358,
        378,
        399,
        419,
        440,
        460,
        481,
    ]

    # 定价计时
    start_time = time.time()
    result = price_snowball(
        bus_days=502,
        cal_days=731,
        ko_bus_days=ko_bus_days,
        ko_cal_days=ko_cal_days,
        ko_prices=ko_prices,
        ki_price=4469.832,
        spot=5587.29,
        r=0.025,
        q=0.06,
        vol=0.20,
        notional=-1000000,
        ko_rate=0.15,
        coupon_rate=0.15,
        strike=5587.29,
        initial=5587.29,
        is_knocked_in=False,
        grid_x=1001,
        grid_t=len(ko_bus_days) * 2,
    )
    pricing_time = time.time() - start_time

    print("\n====== 雪球期权定价结果 ======")
    print(f"计算耗时: {pricing_time:.2f} 秒")
    for key, value in result.items():
        print(f"{key}: {value:,.2f}")

    # Greeks计时
    start_time = time.time()
    greeks = calc_snowball_greeks(
        bus_days=502,
        cal_days=731,
        ko_bus_days=ko_bus_days,
        ko_cal_days=ko_cal_days,
        ko_prices=ko_prices,
        ki_price=4469.832,
        spot=5587.29,
        r=0.025,
        q=0.06,
        vol=0.20,
        notional=-1000000,
        ko_rate=0.15,
        coupon_rate=0.15,
        strike=5587.29,
        initial=5587.29,
        is_knocked_in=False,
        grid_x=1001,
        grid_t=len(ko_bus_days) * 2,
        market_spot=5587.29,
    )
    greeks_time = time.time() - start_time

    print("\n====== 雪球期权Greeks ======")
    print(f"计算耗时: {greeks_time:.2f} 秒")
    for key, value in greeks.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v:,.6f}")
        else:
            print(f"{key}: {value:,.6f}")

    # 总耗时统计
    print(f"\n====== 总计耗时 ======")
    print(f"定价计算: {pricing_time:.2f} 秒")
    print(f"Greeks计算: {greeks_time:.2f} 秒")
    print(f"总耗时: {pricing_time + greeks_time:.2f} 秒")


if __name__ == "__main__":
    example_snowball()
