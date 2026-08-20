"""
Created on Tue Mar 23 2023

@author: yaofuxin
"""
import numpy as np
from scipy.stats import norm


class OneTouch:
    def __init__(
        self,
        initial,
        rrf,
        div,
        vol,
        H,
        spot,
        cal_days,
        bus_days,
        disc_mode,
        rebate,
        N,
        is_call,
        tenor=0,
        obs_type="Daily",
        pay_type="Instant",
        bus_days_in_year=245,
        is_knocked_out=False,
    ) -> None:
        self.initial = initial
        self.rrf = rrf
        self.div = div
        self.b = self.rrf - self.div
        self.vol = vol
        self.H = H
        self.shifted_barrier = self.H
        self.spot = spot
        self.cal_days = cal_days
        self.bus_days = bus_days
        self.T = cal_days / 365
        self.disc_mode = disc_mode
        self.bus_days_in_year = bus_days_in_year
        self.tau = self.bus_days / bus_days_in_year
        self.df = (
            np.exp(-1.0 * self.rrf * self.tau)
            if self.disc_mode == "BD"
            else np.exp(-1.0 * self.rrf * self.T)
        )
        self.rebate = rebate
        self.N = N
        self.is_call = is_call
        self.obs_type = obs_type
        self.pay_type = pay_type
        self.tenor = cal_days if tenor <= 0 else tenor
        self.is_knocked_out = is_knocked_out
        self.value = 0.0
        self.delta = 0.0
        self.delta_cash = 0.0
        self.gamma = 0.0
        self.gamma_cash = 0.0
        self.vega = 0.0
        self.theta = 0.0
        self.rho = 0.0
        self.rhoQ = 0.0

    def __barrier_shift__(self):
        beta = 0.5825971579390107
        dt = 1.0 / self.bus_days_in_year
        if self.obs_type.upper() == "DAILY":
            if self.is_call:
                self.shifted_barrier = self.H * np.exp(beta * self.vol * np.sqrt(dt))
            if not self.is_call:
                self.shifted_barrier = self.H * np.exp(
                    -1.0 * beta * self.vol * np.sqrt(dt)
                )

    def __cal_factors__(self):
        self.mu = (self.b - self.vol**2 / 2) / self.vol**2
        self.lamda = np.sqrt(self.mu**2 + 2 * self.rrf / self.vol**2)
        self.x2 = np.log(self.spot / self.shifted_barrier) / self.vol / np.sqrt(
            self.tau
        ) + (1 + self.mu) * self.vol * np.sqrt(self.tau)
        self.y2 = np.log(self.shifted_barrier / self.spot) / self.vol / np.sqrt(
            self.tau
        ) + (1 + self.mu) * self.vol * np.sqrt(self.tau)
        self.z = np.log(self.shifted_barrier / self.spot) / self.vol / np.sqrt(
            self.tau
        ) + self.lamda * self.vol * np.sqrt(self.tau)

    def price_bsm(self):
        is_knocked_out = (
            (self.spot >= self.H) if self.is_call else (self.spot <= self.H)
        ) or self.is_knocked_out
        if not is_knocked_out:
            self.__price_bsm__()
        else:
            df = self.df if self.pay_type.lower() == "expiry" else 1
            self.value = self.rebate * self.N * self.initial * self.tenor / 365 * df

    def __price_bsm__(self):
        self.b = self.rrf - self.div
        if self.obs_type.upper() == "DAILY":
            self.__barrier_shift__()
        self.__cal_factors__()
        if self.is_call:
            phi = 1.0
            yita = -1.0
            if self.spot >= self.H:
                self.value = self.initial * self.rebate * self.tenor / 365 * self.N
                if self.pay_type.upper() == "EXPIRY":
                    self.value = self.value * self.df
            else:
                if self.pay_type.upper() == "EXPIRY":
                    self.value = (
                        self.initial
                        * self.rebate
                        * self.tenor
                        / 365
                        * self.N
                        * self.df
                        * (
                            norm.cdf(
                                phi * self.x2 - phi * self.vol * np.sqrt(self.tau),
                                loc=0,
                                scale=1,
                            )
                            + np.power(self.shifted_barrier / self.spot, 2 * self.mu)
                            * norm.cdf(
                                yita * self.y2 - yita * self.vol * np.sqrt(self.tau),
                                loc=0,
                                scale=1,
                            )
                        )
                    )
                if self.pay_type.upper() == "INSTANT":
                    self.value = (
                        self.initial
                        * self.rebate
                        * self.tenor
                        / 365
                        * self.N
                        * (
                            np.power(
                                self.shifted_barrier / self.spot, self.mu + self.lamda
                            )
                            * norm.cdf(yita * self.z, loc=0, scale=1)
                            + np.power(
                                self.shifted_barrier / self.spot, self.mu - self.lamda
                            )
                            * norm.cdf(
                                yita * self.z
                                - 2 * yita * self.lamda * self.vol * np.sqrt(self.tau),
                                loc=0,
                                scale=1,
                            )
                        )
                    )
        else:
            phi = -1.0
            yita = 1.0
            if self.spot <= self.H:
                self.value = self.initial * self.rebate * self.tenor / 365 * self.N
                if self.pay_type.upper() == "EXPIRY":
                    self.value = self.value * self.df
            else:
                if self.pay_type.upper() == "EXPIRY":
                    self.value = (
                        self.initial
                        * self.rebate
                        * self.tenor
                        / 365
                        * self.N
                        * self.df
                        * (
                            norm.cdf(
                                phi * self.x2 - phi * self.vol * np.sqrt(self.tau),
                                loc=0,
                                scale=1,
                            )
                            + np.power(self.shifted_barrier / self.spot, 2 * self.mu)
                            * norm.cdf(
                                yita * self.y2 - yita * self.vol * np.sqrt(self.tau),
                                loc=0,
                                scale=1,
                            )
                        )
                    )
                if self.pay_type.upper() == "INSTANT":
                    self.value = (
                        self.initial
                        * self.rebate
                        * self.tenor
                        / 365
                        * self.N
                        * (
                            np.power(
                                self.shifted_barrier / self.spot, self.mu + self.lamda
                            )
                            * norm.cdf(yita * self.z, loc=0, scale=1)
                            + np.power(
                                self.shifted_barrier / self.spot, self.mu - self.lamda
                            )
                            * norm.cdf(
                                yita * self.z
                                - 2 * yita * self.lamda * self.vol * np.sqrt(self.tau),
                                loc=0,
                                scale=1,
                            )
                        )
                    )