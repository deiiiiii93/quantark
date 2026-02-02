# -*- coding: utf-8 -*-
"""
敲出重置雪球
验证用示例

Author: Kejian Shi
Created Date: 2026-01-21
"""
import sys

if sys.version_info < (3, 10):
    raise ImportError('Requires Python >= 3.10')

import datetime
from collections.abc import Callable
from dataclasses import dataclass, InitVar

import numpy as np
from numpy.typing import NDArray


# region MC

class BSProcess:

    def __init__(self, s0: float, sigma: float, q: float, r: float):
        self.s0 = s0
        self.sigma = sigma
        self.q = q
        self.r = r

    def evolve(
        self,
        st: float | NDArray[np.number],
        dt: float,
        dw: float | NDArray[np.number],
    ) -> float:
        return st * np.exp(
            (self.r - self.q - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * dw,
        )


class PseudoRNG:

    def __init__(self, shape: int | tuple[int, ...], seed: int):
        self.shape: tuple[int, ...] = (shape,) if isinstance(shape, int) else shape
        self._rng = np.random.default_rng(seed=seed)

    def next(self) -> NDArray[np.number]:
        return self._rng.standard_normal(self.shape, dtype=np.float64)


class PathGenerator:

    def __init__(self, process, rng, obs_time: NDArray[np.number]):
        self.process = process
        self.rng = rng
        self.obs_time = obs_time

    def next(self):
        sequence = self.rng.next()
        path = np.empty(sequence.shape, dtype=np.float64)
        path[0] = self.process.s0
        for i in range(self.obs_time.size - 1):
            path[i + 1] = self.process.evolve(
                st=path[i],
                dt=self.obs_time[i + 1] - self.obs_time[i],
                dw=sequence[i],
            )
        return path


def mc_solve(path_gen, payoff_func: Callable, option: 'OptionInfo') -> float:
    tau_value, payoff_value = zip(
        *[
            payoff_func(path_i, option)
            for path_i in path_gen.next().T
        ],
    )
    tau_value, payoff_value = np.array(tau_value), np.array(payoff_value)
    return float(
        (np.exp(-option.r * tau_value) * payoff_value).mean() * option.nominal_value,
    )


# endregion

# region Option

@dataclass
class OptionInfo:
    days_in_year: int  # 交易日数量
    natural_days_in_year: int  # 自然日数量

    # 其他要素
    s0: float  # 期初价格
    st: float  # 当前价格
    nominal_value: float  # 名本
    ptc_rate: float  # 参与率
    sigma: float  # 波动率
    q: float  # 分红率
    r: float  # 无风险
    coupon: float  # 红利票息

    # 敲出要素（敲入前）
    obs_days: InitVar[list[int]]  # 敲出观察天数-交易日计数
    obs_natural_days: InitVar[list[int]]  # 敲出观察天数-自然日计数
    obs_dates: list[datetime.date]  # 观察日
    barriers: list[float]  # 敲出障碍
    rebates: list[float]  # 敲出票息

    # 敲出要素（敲入后）
    obs_days_ki: InitVar[list[int]]  # 敲出观察天数-交易日计数
    obs_natural_days_ki: InitVar[list[int]]  # 敲出观察天数-自然日计数
    obs_dates_ki: list[datetime.date]  # 观察日
    barriers_ki: list[float]  # 敲出障碍
    rebates_ki: list[float]  # 敲出票息

    # 敲入要素
    # 常规每日观察的如果要素也都显式展示，行数太多
    # 此处做简化
    low_barrier: float  # 敲入障碍价格
    strike: float  # 敲入行权价
    ptc_put: float  # 敲入参与率

    def __post_init__(
        self,
        obs_days: list[int],
        obs_natural_days: list[int],
        obs_days_ki: list[int],
        obs_natural_days_ki: list[int],
    ):
        self.time_to_obs = [days_i / self.days_in_year for days_i in obs_days]
        self.natural_time_to_obs = [days_i / self.natural_days_in_year for days_i in obs_natural_days]
        self.time_to_obs_ki = [days_i / self.days_in_year for days_i in obs_days_ki]
        self.natural_time_to_obs_ki = [days_i / self.natural_days_in_year for days_i in obs_natural_days_ki]
        self.tau_nki = max(self.time_to_obs)  # 敲入前存续期
        self.tau_ki = max(self.time_to_obs_ki)  # 敲入后存续期
        self.tau = max(self.tau_nki, self.tau_ki)  # tau 需要两者取其大，保证路径生成长度足够
        self.coupon *= max(self.natural_time_to_obs)  # 年化转绝对
        self.rebates = [r_i * t_i for r_i, t_i in zip(self.rebates, self.natural_time_to_obs)]
        self.rebates_ki = [r_i * t_i for r_i, t_i in zip(self.rebates_ki, self.natural_time_to_obs_ki)]


def option_payoff(s: NDArray[np.number], option: 'OptionInfo') -> tuple[float, float]:
    observe_idx = (np.array(option.time_to_obs) * option.days_in_year).astype(int)
    observe_idx_ki = (np.array(option.time_to_obs_ki) * option.days_in_year).astype(int)
    in_flag = (s < np.array(option.low_barrier))[:int(min(observe_idx.max(), observe_idx.max()))]
    if np.any(in_flag):
        ki_index = int(np.argmax(in_flag))
        # 在敲入时点前的用未敲入要素，从敲入起往后的用敲入后要素
        idx = np.where(observe_idx < ki_index)[0]
        idx_ki = np.where(observe_idx_ki >= ki_index)[0]
        # 拼接实际要素
        _observe_idx = np.concatenate(
            [
                observe_idx[idx],
                observe_idx_ki[idx_ki],
            ]
        )
        _time_to_obs = np.concatenate(
            [
                np.array(option.time_to_obs)[idx],
                np.array(option.time_to_obs_ki)[idx_ki],
            ]
        )
        _high_barrier = np.concatenate(
            [
                np.array(option.barriers)[idx],
                np.array(option.barriers_ki)[idx_ki],
            ]
        )
        _rebates = np.concatenate(
            [
                np.array(option.rebates)[idx],
                np.array(option.rebates_ki)[idx_ki],
            ]
        )
        out_flag = s[np.maximum(_observe_idx - 1, 0)] >= _high_barrier
        if np.any(out_flag):
            out_index = int(np.argmax(out_flag))
            return _time_to_obs[out_index], _rebates[out_index]
        return (
            _time_to_obs[-1],
            -max(option.strike - s[-1], 0.0) / option.s0 * option.ptc_put
        )
    else:
        # 不敲入时和经典雪球一致
        out_flag = s[np.maximum(observe_idx - 1, 0)] >= np.array(option.barriers)
        if np.any(out_flag):
            out_index = int(np.argmax(out_flag))
            return option.time_to_obs[out_index], option.rebates[out_index]
        return option.tau_nki, option.coupon


# endregion

if __name__ == '__main__':
    config = {
        'days_in_year': 244,  # 每年交易日天数
        'natural_days_in_year': 365,  # 每年自然日天数
        'dt': 1 / 244,  # 最小观察时间间隔对应每日观察
        'path_num': 200_000,  # MC 路径数
        'seed': 100,  # 随机数
    }
    option = OptionInfo(
        **{
            'days_in_year': config['days_in_year'],
            'natural_days_in_year': config['natural_days_in_year'],
            #
            's0': (s0 := 8340.11),
            'st': 1.0 * s0,
            'nominal_value': 50_000_000.00,
            'ptc_rate': 1.0,
            'sigma': 0.20,
            'q': 0.04,
            'r': 0.02,
            'coupon': 0.15,
            #
            'obs_dates': [
                datetime.date(2026, 4, 21),
                datetime.date(2026, 5, 21),
                datetime.date(2026, 6, 22),
                datetime.date(2026, 7, 21),
                datetime.date(2026, 8, 21),
                #
                datetime.date(2026, 9, 21),
                datetime.date(2026, 10, 21),
                datetime.date(2026, 11, 23),
                datetime.date(2026, 12, 21),
                datetime.date(2027, 1, 21),
                #
                datetime.date(2027, 2, 22),
                datetime.date(2027, 3, 22),
                datetime.date(2027, 4, 21),
                datetime.date(2027, 5, 21),
                datetime.date(2027, 6, 21),
                #
                datetime.date(2027, 7, 21),
                datetime.date(2027, 8, 23),
                datetime.date(2027, 9, 21),
                datetime.date(2027, 10, 21),
                datetime.date(2027, 11, 22),
                #
                datetime.date(2027, 12, 21),
                datetime.date(2028, 1, 21),
            ],
            'obs_days': [
                58,
                77,
                98,
                119,
                142,
                #
                163,
                179,
                202,
                222,
                244,
                #
                261,
                281,
                302,
                321,
                341,
                #
                363,
                386,
                406,
                423,
                445,
                #
                466,
                488,
            ],
            'obs_natural_days': [
                91,
                121,
                153,
                182,
                213,
                #
                244,
                274,
                307,
                335,
                366,
                #
                398,
                426,
                456,
                486,
                517,
                #
                547,
                580,
                609,
                639,
                671,
                #
                700,
                731,
            ],
            'barriers': [
                1.03 * s0,
                1.03 * s0,
                1.03 * s0,
                1.03 * s0,
                1.03 * s0,
                #
                1.03 * s0,
                1.03 * s0,
                1.03 * s0,
                1.03 * s0,
                1.03 * s0,
                #
                1.03 * s0,
                1.03 * s0,
                1.03 * s0,
                1.03 * s0,
                1.03 * s0,
                #
                1.03 * s0,
                1.03 * s0,
                1.03 * s0,
                1.03 * s0,
                1.03 * s0,
                #
                1.03 * s0,
                1.03 * s0,
            ],
            'rebates': [
                0.15,
                0.15,
                0.15,
                0.15,
                0.15,
                #
                0.15,
                0.15,
                0.15,
                0.15,
                0.15,
                #
                0.15,
                0.15,
                0.15,
                0.15,
                0.15,
                #
                0.15,
                0.15,
                0.15,
                0.15,
                0.15,
                #
                0.15,
                0.15,
            ],
            #
            'obs_dates_ki': [
                datetime.date(2026, 4, 21),
                datetime.date(2026, 5, 21),
                datetime.date(2026, 6, 22),
                datetime.date(2026, 7, 21),
                datetime.date(2026, 8, 21),
                #
                datetime.date(2026, 9, 21),
                datetime.date(2026, 10, 21),
                datetime.date(2026, 11, 23),
                datetime.date(2026, 12, 21),
                datetime.date(2027, 1, 21),
                #
                datetime.date(2027, 2, 22),
                datetime.date(2027, 3, 22),
                datetime.date(2027, 4, 21),
                datetime.date(2027, 5, 21),
                datetime.date(2027, 6, 21),
                #
                datetime.date(2027, 7, 21),
                datetime.date(2027, 8, 23),
                datetime.date(2027, 9, 21),
                datetime.date(2027, 10, 21),
                datetime.date(2027, 11, 22),
                #
                datetime.date(2027, 12, 21),
                datetime.date(2028, 1, 21),
                datetime.date(2028, 2, 21),
                datetime.date(2028, 3, 21),
                datetime.date(2028, 4, 21),
                #
                datetime.date(2028, 5, 22),
                datetime.date(2028, 6, 21),
                datetime.date(2028, 7, 21),
                datetime.date(2028, 8, 21),
                datetime.date(2028, 9, 21),
                #
                datetime.date(2028, 10, 23),
                datetime.date(2028, 11, 21),
                datetime.date(2028, 12, 21),
                datetime.date(2029, 1, 22),
                datetime.date(2029, 2, 21),
                #
                datetime.date(2029, 3, 21),
                datetime.date(2029, 4, 23),
                datetime.date(2029, 5, 21),
                datetime.date(2029, 6, 21),
                datetime.date(2029, 7, 23),
                #
                datetime.date(2029, 8, 21),
                datetime.date(2029, 9, 21),
                datetime.date(2029, 10, 22),
                datetime.date(2029, 11, 21),
                datetime.date(2029, 12, 21),
                #
                datetime.date(2030, 1, 21),
            ],
            'obs_days_ki': [
                58,
                77,
                98,
                119,
                142,
                #
                163,
                179,
                202,
                222,
                244,
                #
                261,
                281,
                302,
                321,
                341,
                #
                363,
                386,
                406,
                423,
                445,
                #
                466,
                488,
                503,
                524,
                545,
                #
                563,
                584,
                606,
                627,
                650,
                #
                667,
                688,
                710,
                731,
                747,
                #
                767,
                789,
                807,
                829,
                851,
                #
                872,
                895,
                910,
                932,
                954,
                #
                973,
            ],
            'obs_natural_days_ki': [
                91,
                121,
                153,
                182,
                213,
                #
                244,
                274,
                307,
                335,
                366,
                #
                398,
                426,
                456,
                486,
                517,
                #
                547,
                580,
                609,
                639,
                671,
                #
                700,
                731,
                762,
                791,
                822,
                #
                852,
                883,
                913,
                944,
                975,
                #
                1007,
                1036,
                1066,
                1098,
                1128,
                #
                1156,
                1189,
                1217,
                1248,
                1280,
                #
                1309,
                1340,
                1371,
                1401,
                1431,
                #
                1462,
            ],
            'barriers_ki': [
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                #
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                #
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                #
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                #
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                #
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                #
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                #
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                #
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                0.95 * s0,
                #
                0.95 * s0,
            ],
            'rebates_ki': [
                0.03,
                0.03,
                0.03,
                0.03,
                0.03,
                #
                0.03,
                0.03,
                0.03,
                0.03,
                0.03,
                #
                0.03,
                0.03,
                0.03,
                0.03,
                0.03,
                #
                0.03,
                0.03,
                0.03,
                0.03,
                0.03,
                #
                0.03,
                0.03,
                0.03,
                0.03,
                0.03,
                #
                0.03,
                0.03,
                0.03,
                0.03,
                0.03,
                #
                0.03,
                0.03,
                0.03,
                0.03,
                0.03,
                #
                0.03,
                0.03,
                0.03,
                0.03,
                0.03,
                #
                0.03,
                0.03,
                0.03,
                0.03,
                0.03,
                #
                0.03,
            ],
            #
            'low_barrier': 0.8 * s0,
            'strike': 1.0 * s0,
            'ptc_put': 1.0,
        },
    )
    time = np.arange(0, int(np.ceil(option.tau / config['dt']) + 1), dtype=np.float64) * config['dt']
    time[-1] = option.tau

    path_generator = PathGenerator(
        process=BSProcess(
            s0=option.st,
            sigma=option.sigma,
            q=option.q,
            r=option.r,
        ),
        obs_time=time,
        rng=PseudoRNG(
            shape=(time.size, int(config['path_num'])),
            seed=config['seed'],
        ),
    )
    result = mc_solve(path_generator, option_payoff, option)
    result = -result  # 实际方向为我方卖

    print(result)
